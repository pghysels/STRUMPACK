/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The
 * Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals
 * from the U.S. Dept. of Energy).  All rights reserved.
 *
 * If you have questions about your rights to use or distribute this
 * software, please contact Berkeley Lab's Technology Transfer
 * Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As
 * such, the U.S. Government has been granted for itself and others
 * acting on its behalf a paid-up, nonexclusive, irrevocable,
 * worldwide license in the Software to reproduce, prepare derivative
 * works, and perform publicly and display publicly.  Beginning five
 * (5) years after the date permission to assert copyright is obtained
 * from the U.S. Department of Energy, and subject to any subsequent
 * five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
#ifndef ELIMINATION_TREE_MPI_HPP
#define ELIMINATION_TREE_MPI_HPP

#include <iostream>
#include <algorithm>
#include <random>
#include <algorithm>

#include "StrumpackParameters.hpp"
#include "CompressedSparseMatrix.hpp"
#include "CSRMatrixMPI.hpp"
#include "MatrixReorderingMPI.hpp"
#include "FrontalMatrix.hpp"
#include "FrontalMatrixMPI.hpp"
#include "FrontalMatrixDenseMPI.hpp"
#include "FrontalMatrixHSSMPI.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  class EliminationTreeMPI : public EliminationTree<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using Reord_t = MatrixReordering<scalar_t,integer_t>;
    using Tree_t = SeparatorTree<integer_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FD_t = FrontalMatrixDense<scalar_t,integer_t>;
    using FBLR_t = FrontalMatrixBLR<scalar_t,integer_t>;
    using FHSS_t = FrontalMatrixHSS<scalar_t,integer_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using FDMPI_t = FrontalMatrixDenseMPI<scalar_t,integer_t>;
    using FHSSMPI_t = FrontalMatrixHSSMPI<scalar_t,integer_t>;
    using FBLRMPI_t = FrontalMatrixHSSMPI<scalar_t,integer_t>;
    using SepRange = std::pair<std::size_t,std::size_t>;

  public:
    EliminationTreeMPI(const MPIComm& comm);

    EliminationTreeMPI
    (const SPOptions<scalar_t>& opts, const SpMat_t& A,
     Reord_t& nd, const MPIComm& comm);

    virtual ~EliminationTreeMPI() {}

    void multifrontal_solve(DenseM_t& x) const override;
    integer_t maximum_rank() const override;
    long long factor_nonzeros() const override;
    long long dense_factor_nonzeros() const override;
    std::vector<SepRange> subtree_ranges;

  protected:
    const MPIComm& comm_;
    int rank_;
    int P_;

    virtual int nr_HSS_fronts() const override {
      return comm_.all_reduce(this->nr_HSS_fronts_, MPI_SUM);
    }
    virtual int nr_dense_fronts() const override {
      return comm_.all_reduce(this->nr_dense_fronts_, MPI_SUM);
    }

  private:
    struct ParFront {
      // TODO store a pointer to the actual front??
      ParFront
      (integer_t _sep_begin, integer_t _dim_sep,
       int _P0, int _P, BLACSGrid* g)
        : sep_begin(_sep_begin), dim_sep(_dim_sep),
          P0(_P0), P(_P), grid(g) {}
      integer_t sep_begin, dim_sep;
      int P0, P;
      const BLACSGrid* grid;
    };
    std::vector<ParFront> parallel_fronts_;
    integer_t active_pfronts_;

    void symbolic_factorization
    (const SpMat_t& A, const Tree_t& tree, integer_t sep,
     std::vector<integer_t>* upd, float* subtree_work, int depth=0) const;

    std::unique_ptr<F_t> proportional_mapping
    (const Tree_t& tree, const SPOptions<scalar_t>& opts,
     std::vector<integer_t>* upd, float* subtree_work, SepRange& local_range,
     integer_t sep, int P0, int P, const MPIComm& fcomm,
     bool keep, bool is_hss, int level=0);

    void sequential_to_block_cyclic(DenseM_t& x, DistM_t*& x_dist) const;
    void block_cyclic_to_sequential(DenseM_t& x, DistM_t*& x_dist) const;
  };

  template<typename scalar_t,typename integer_t>
  EliminationTreeMPI<scalar_t,integer_t>::EliminationTreeMPI
  (const MPIComm& comm) : EliminationTree<scalar_t,integer_t>(),
    comm_(comm), rank_(comm.rank()), P_(comm.size()) {
  }

  template<typename scalar_t,typename integer_t>
  EliminationTreeMPI<scalar_t,integer_t>::EliminationTreeMPI
  (const SPOptions<scalar_t>& opts, const SpMat_t& A,
   Reord_t& nd, const MPIComm& comm)
    : EliminationTree<scalar_t,integer_t>(),
    comm_(comm), rank_(comm.rank()), P_(comm.size()), active_pfronts_(0) {
    auto& tree = nd.tree();

    // use vector instead? problem with OpenMP??
    auto upd = new std::vector<integer_t>[tree.separators()];
    auto subtree_work = new float[tree.separators()];

#pragma omp parallel default(shared)
#pragma omp single
    symbolic_factorization(A, tree, tree.root(), upd, subtree_work);

    SepRange local_range{A.size(), 0};
    this->root_ = proportional_mapping
      (tree, opts, upd, subtree_work, local_range, tree.root(),
       0, comm_.size(), comm_, true, true, 0);

    subtree_ranges.resize(P_);
    MPI_Allgather
      (&local_range, sizeof(SepRange), MPI_BYTE, subtree_ranges.data(),
       sizeof(SepRange), MPI_BYTE, comm_.comm());
    nd.clear_tree_data();
    delete[] upd;
    delete[] subtree_work;
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPI<scalar_t,integer_t>::sequential_to_block_cyclic
  (DenseM_t& x, DistM_t*& x_dist) const {
    size_t pos = 0;
    for (auto& pf : parallel_fronts_)
      if (rank_ >= pf.P0 && rank_ < pf.P0+pf.P) pos++;
    x_dist = new DistM_t[pos];
    pos = 0;
    for (auto& pf : parallel_fronts_)
      if (rank_ >= pf.P0 && rank_ < pf.P0+pf.P)
        // TODO this also does a pgemr2d!
        // TODO check if this is correct?!
        x_dist[pos++] = DistM_t
          (pf.grid, DenseMW_t(pf.dim_sep, x.cols(), x, pf.sep_begin, 0));
  }

  // TODO: rewrite this with a single alltoallv/allgatherv
  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPI<scalar_t,integer_t>::block_cyclic_to_sequential
  (DenseM_t& x, DistM_t*& x_dist) const {
    auto cnts = new int[2*P_];
    auto disp = cnts + P_;
    for (int p=0; p<P_; p++) {
      cnts[p] = std::max
        (std::size_t(0), subtree_ranges[p].second - subtree_ranges[p].first);
      disp[p] = subtree_ranges[p].first;
    }
    MPI_Allgatherv
      (MPI_IN_PLACE, 0, mpi_type<scalar_t>(), x.data(),
       cnts, disp, mpi_type<scalar_t>(), comm_.comm());
    delete[] cnts;

    auto xd = x_dist;
    for (auto& pf : parallel_fronts_)
      if (rank_ >= pf.P0 && rank_ < pf.P0+pf.P) {
        // TODO check if this is correct
        DenseMW_t x_loc(pf.dim_sep, x.cols(), x, pf.sep_begin, 0);
        x_loc = (xd++)->gather();
        MPI_Bcast
          (x_loc.data(), pf.dim_sep, mpi_type<scalar_t>(), pf.P0, comm_.comm());
      }
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPI<scalar_t,integer_t>::multifrontal_solve
  (DenseM_t& x) const {
    DistM_t* x_dist;
    sequential_to_block_cyclic(x, x_dist);
    this->root_->multifrontal_solve(x, x_dist);
    block_cyclic_to_sequential(x, x_dist);
    delete[] x_dist;
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPI<scalar_t,integer_t>::symbolic_factorization
  (const SpMat_t& A, const Tree_t& tree, const integer_t sep,
   std::vector<integer_t>* upd, float* subtree_work, int depth) const {
    auto chl = tree.lch(sep);
    auto chr = tree.rch(sep);
    if (depth < params::task_recursion_cutoff_level) {
      if (chl != -1)
#pragma omp task untied default(shared)                                 \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        symbolic_factorization(A, tree, chl, upd, subtree_work, depth+1);
      if (chr != -1)
#pragma omp task untied default(shared)                                 \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        symbolic_factorization(A, tree, chr, upd, subtree_work, depth+1);
#pragma omp taskwait
    } else {
      if (chl != -1)
        symbolic_factorization(A, tree, chl, upd, subtree_work, depth);
      if (chr != -1)
        symbolic_factorization(A, tree, chr, upd, subtree_work, depth);
    }
    auto sep_begin = tree.sizes(sep);
    auto sep_end = tree.sizes(sep+1);
    if (sep != tree.separators()-1) { // not necessary for the root
      for (integer_t c=sep_begin; c<sep_end; c++) {
        auto ice = A.ind()+A.ptr(c+1);
        auto icb = std::lower_bound
          (A.ind()+A.ptr(c), ice, sep_end);
        auto mid = upd[sep].size();
        std::copy(icb, ice, std::back_inserter(upd[sep]));
        std::inplace_merge
          (upd[sep].begin(), upd[sep].begin() + mid, upd[sep].end());
        upd[sep].erase
          (std::unique(upd[sep].begin(), upd[sep].end()), upd[sep].end());
      }
      if (chl != -1) {
        auto icb = std::lower_bound
          (upd[chl].begin(), upd[chl].end(), sep_end);
        auto mid = upd[sep].size();
        std::copy(icb, upd[chl].end(), std::back_inserter(upd[sep]));
        std::inplace_merge
          (upd[sep].begin(), upd[sep].begin() + mid, upd[sep].end());
        upd[sep].erase
          (std::unique(upd[sep].begin(), upd[sep].end()), upd[sep].end());
      }
      if (chr != -1) {
        auto icb = std::lower_bound
          (upd[chr].begin(), upd[chr].end(), sep_end);
        auto mid = upd[sep].size();
        std::copy(icb, upd[chr].end(), std::back_inserter(upd[sep]));
        std::inplace_merge
          (upd[sep].begin(), upd[sep].begin() + mid, upd[sep].end());
        upd[sep].erase
          (std::unique(upd[sep].begin(), upd[sep].end()), upd[sep].end());
      }
    }
    integer_t dim_blk = (sep_end - sep_begin) + upd[sep].size();
    // assume amount of work per front is N^3, work per subtree is
    // work on front plus children
    float wl = (chl != -1) ? subtree_work[chl] : 0.;
    float wr = (chr != -1) ? subtree_work[chr] : 0.;
    subtree_work[sep] = (float(dim_blk)*dim_blk*dim_blk) + wl + wr;
  }

  // keep track of [P0_pa, P0_pa+P_pa) -> can be used to stop iso keep_subtree
  template<typename scalar_t,typename integer_t>
  std::unique_ptr<FrontalMatrix<scalar_t,integer_t>>
  EliminationTreeMPI<scalar_t,integer_t>::proportional_mapping
  (const Tree_t& tree, const SPOptions<scalar_t>& opts,
   std::vector<integer_t>* upd, float* subtree_work, SepRange& local_range,
   integer_t sep, int P0, int P, const MPIComm& fcomm,
   bool keep, bool hss_parent, int level) {
    auto sep_begin = tree.sizes(sep);
    auto sep_end = tree.sizes(sep+1);
    auto dim_sep = sep_end - sep_begin;
    std::unique_ptr<F_t> front;
    bool is_hss = opts.use_HSS() && hss_parent &&
      (dim_sep >= opts.HSS_min_front_size());
    bool is_blr = opts.use_BLR() && (dim_sep >= opts.BLR_min_front_size());
    if (rank_ == P0) {
      if (is_hss) this->nr_HSS_fronts_++;
      else if (is_blr) this->nr_BLR_fronts_++;
      else this->nr_dense_fronts_++;
    }
    if (P == 1) {
      if (keep) {
        if (is_hss) {
          front = std::unique_ptr<F_t>
            (new FHSS_t(sep, sep_begin, sep_end, upd[sep]));
          front->set_HSS_partitioning(opts, tree.HSS_tree(sep), level == 0);
        } else if (is_blr) {
          front = std::unique_ptr<F_t>
            (new FBLR_t(sep, sep_begin, sep_end, upd[sep]));
          front->set_BLR_partitioning
            (opts, tree.HSS_tree(sep), tree.admissibility(sep), level == 0);
        } else
          front = std::unique_ptr<F_t>
            (new FD_t(sep, sep_begin, sep_end, upd[sep]));
      }
      if (P0 == rank_) {
        local_range.first  = std::min
          (local_range.first, std::size_t(sep_begin));
        local_range.second = std::max
          (local_range.second, std::size_t(sep_end));
      }
    } else {
      if (keep) {
        if (is_hss) {
          front = std::unique_ptr<F_t>
            (new FHSSMPI_t(active_pfronts_, sep_begin, sep_end, upd[sep], fcomm, P));
          front->set_HSS_partitioning(opts, tree.HSS_tree(sep), level == 0);
        } else {
          if (is_blr) {
            front = std::unique_ptr<F_t>
              (new FBLRMPI_t(active_pfronts_, sep_begin, sep_end, upd[sep], fcomm, P));
            front->set_BLR_partitioning
              (opts, tree.HSS_tree(sep), tree.admissibility(sep), level == 0);
          } else
            front = std::unique_ptr<F_t>
              (new FDMPI_t(active_pfronts_, sep_begin, sep_end, upd[sep], fcomm, P));
        }
        if (rank_ >= P0 && rank_ < P0+P) active_pfronts_++;
      }
      auto g = front ? static_cast<FMPI_t*>(front.get())->grid() : nullptr;
      parallel_fronts_.emplace_back
        (sep_begin, sep_end-sep_begin, P0, P, g);
    }

    // only store a node if you are part of its communicator
    // and also store your siblings!! needed for extend-add
    if (rank_ < P0 || rank_ >= P0+P) keep = false;
    if (P == 1 && P0 != rank_) return front;

    auto chl = tree.lch(sep);
    auto chr = tree.rch(sep);
    if (chl != -1) {
      float wl = subtree_work[chl];
      float wr = (chr != -1) ? subtree_work[chr] : 0.;
      int Pl = std::max(1, std::min(int(std::round(P * wl / (wl + wr))), P-1));
      int Pr = std::max(1, P - Pl);
      auto fl = proportional_mapping
        (tree, opts, upd, subtree_work, local_range, chl, P0, Pl,
         fcomm.sub(0, Pl), keep, is_hss, level+1);
      if (front) front->set_lchild(std::move(fl));
      if (chr != -1) {
        auto fr = proportional_mapping
          (tree, opts, upd, subtree_work, local_range, chr, P0+P-Pr, Pr,
           fcomm.sub(P-Pr, Pr), keep, is_hss, level+1);
        if (front) front->set_rchild(std::move(fr));
      }
    } else {
      if (chr != -1) {
        auto fr = proportional_mapping
          (tree, opts, upd, subtree_work, local_range, chr, P0, P,
           fcomm, keep, is_hss, level+1);
        if (front) front->set_rchild(std::move(fr));
      }
    }
    return front;
  }

  template<typename scalar_t,typename integer_t> integer_t
  EliminationTreeMPI<scalar_t,integer_t>::maximum_rank() const {
    return comm_.all_reduce
      (EliminationTree<scalar_t,integer_t>::maximum_rank(), MPI_MAX);
  }

  template<typename scalar_t,typename integer_t> long long
  EliminationTreeMPI<scalar_t,integer_t>::factor_nonzeros() const {
    return comm_.all_reduce
      (EliminationTree<scalar_t,integer_t>::factor_nonzeros(), MPI_SUM);
  }

  template<typename scalar_t,typename integer_t> long long
  EliminationTreeMPI<scalar_t,integer_t>::dense_factor_nonzeros() const {
    return comm_.all_reduce
      (EliminationTree<scalar_t,integer_t>::dense_factor_nonzeros(), MPI_SUM);
  }

} // end namespace strumpack

#endif
