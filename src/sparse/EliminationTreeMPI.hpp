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
    using FHSS_t = FrontalMatrixHSS<scalar_t,integer_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using FDMPI_t = FrontalMatrixDenseMPI<scalar_t,integer_t>;
    using FHSSMPI_t = FrontalMatrixHSSMPI<scalar_t,integer_t>;
    using SepRange = std::pair<std::size_t,std::size_t>;

  public:
    EliminationTreeMPI(MPI_Comm comm);
    EliminationTreeMPI
    (const SPOptions<scalar_t>& opts, const SpMat_t& A,
     Reord_t& nd, MPI_Comm comm);
    virtual ~EliminationTreeMPI() { mpi_free_comm(&_comm); }

    void multifrontal_solve(DenseM_t& x) const override;
    integer_t maximum_rank() const override;
    long long factor_nonzeros() const override;
    long long dense_factor_nonzeros() const override;
    std::vector<SepRange> subtree_ranges;

  protected:
    MPI_Comm _comm;
    int _rank;

    virtual int nr_HSS_fronts() const override {
      int f = this->_nr_HSS_fronts;
      MPI_Allreduce
        (MPI_IN_PLACE, &f, 1, MPI_INT, MPI_SUM, _comm);
      return f;
    }
    virtual int nr_dense_fronts() const override {
      int f = this->_nr_dense_fronts;
      MPI_Allreduce
        (MPI_IN_PLACE, &f, 1, MPI_INT, MPI_SUM, _comm);
      return f;
    }

  private:
    struct ParFront {
      // TODO store a pointer to the actual front??
      ParFront
      (integer_t _sep_begin, integer_t _dim_sep,
       int _P0, int _P, int _ctxt, int _ctxt_all)
        : sep_begin(_sep_begin), dim_sep(_dim_sep), P0(_P0), P(_P),
          ctxt(_ctxt), ctxt_all(_ctxt_all) {}
      integer_t sep_begin, dim_sep;
      int P0, P, ctxt, ctxt_all;
    };
    std::vector<ParFront> _parallel_fronts;
    integer_t _active_pfronts;

    void symbolic_factorization
    (const SpMat_t& A, const Tree_t& tree, integer_t sep,
     std::vector<integer_t>* upd, float* subtree_work, int depth=0) const;

    F_t* proportional_mapping
    (const SpMat_t& A, const Tree_t& tree, const SPOptions<scalar_t>& opts,
     std::vector<integer_t>* upd, float* subtree_work, SepRange& local_range,
     integer_t sep, int P0, int P, MPI_Comm front_comm,
     bool keep, bool is_hss, int level=0);

    void sequential_to_block_cyclic(DenseM_t& x, DistM_t*& x_dist) const;
    void block_cyclic_to_sequential(DenseM_t& x, DistM_t*& x_dist) const;
  };

  template<typename scalar_t,typename integer_t>
  EliminationTreeMPI<scalar_t,integer_t>::EliminationTreeMPI(MPI_Comm comm)
    : EliminationTree<scalar_t,integer_t>() {
    MPI_Comm_dup(comm, &_comm);
    _rank = mpi_rank(_comm);
  }

  template<typename scalar_t,typename integer_t>
  EliminationTreeMPI<scalar_t,integer_t>::EliminationTreeMPI
  (const SPOptions<scalar_t>& opts, const SpMat_t& A,
   Reord_t& nd, MPI_Comm comm)
    : EliminationTree<scalar_t,integer_t>(), _active_pfronts(0) {
    auto tree = *(nd.sep_tree);
    MPI_Comm_dup(comm, &_comm);
    _rank = mpi_rank(_comm);
    auto P = mpi_nprocs(_comm);
    auto upd = new std::vector<integer_t>[tree.separators()];
    auto subtree_work = new float[tree.separators()];

#pragma omp parallel default(shared)
#pragma omp single
    symbolic_factorization(A, tree, tree.root(), upd, subtree_work);

    SepRange local_range{A.size(), 0};
    MPI_Comm tree_comm;
    if (P>1) MPI_Comm_dup(_comm, &tree_comm);
    else tree_comm = _comm;

    this->_root = std::unique_ptr<F_t>
      (proportional_mapping
       (A, tree, opts, upd, subtree_work, local_range, tree.root(),
        0, P, tree_comm, true, true, 0));

    subtree_ranges.resize(P);
    MPI_Allgather
      (&local_range, sizeof(SepRange), MPI_BYTE, subtree_ranges.data(),
       sizeof(SepRange), MPI_BYTE, _comm);
    nd.clear_tree_data();
    delete[] upd;
    delete[] subtree_work;
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPI<scalar_t,integer_t>::sequential_to_block_cyclic
  (DenseM_t& x, DistM_t*& x_dist) const {
    size_t pos = 0;
    for (auto& pf : _parallel_fronts)
      if (_rank >= pf.P0 && _rank < pf.P0+pf.P) pos++;
    x_dist = new DistM_t[pos];
    pos = 0;
    for (auto& pf : _parallel_fronts)
      if (_rank >= pf.P0 && _rank < pf.P0+pf.P)
        // TODO this also does a pgemr2d!
        // TODO check if this is correct?!
        x_dist[pos++] = DistM_t
          (pf.ctxt, DenseMW_t(pf.dim_sep, x.cols(), x, pf.sep_begin, 0));
  }

  // TODO: rewrite this with a single alltoallv/allgatherv
  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPI<scalar_t,integer_t>::block_cyclic_to_sequential
  (DenseM_t& x, DistM_t*& x_dist) const {
    auto P = mpi_nprocs(_comm);
    auto cnts = new int[2*P];
    auto disp = cnts + P;
    for (int p=0; p<P; p++) {
      cnts[p] = std::max
        (std::size_t(0), subtree_ranges[p].second - subtree_ranges[p].first);
      disp[p] = subtree_ranges[p].first;
    }
    MPI_Allgatherv
      (MPI_IN_PLACE, 0, mpi_type<scalar_t>(), x.data(),
       cnts, disp, mpi_type<scalar_t>(), _comm);
    delete[] cnts;

    auto xd = x_dist;
    for (auto& pf : _parallel_fronts)
      if (_rank >= pf.P0 && _rank < pf.P0+pf.P) {
        // TODO check if this is correct
        DenseMW_t x_loc(pf.dim_sep, x.cols(), x, pf.sep_begin, 0);
        x_loc = (xd++)->gather();
        MPI_Bcast
          (x_loc.data(), pf.dim_sep, mpi_type<scalar_t>(), pf.P0, _comm);
      }
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPI<scalar_t,integer_t>::multifrontal_solve
  (DenseM_t& x) const {
    DistM_t* x_dist;
    sequential_to_block_cyclic(x, x_dist);
    this->_root->multifrontal_solve(x, x_dist);
    block_cyclic_to_sequential(x, x_dist);
    delete[] x_dist;
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPI<scalar_t,integer_t>::symbolic_factorization
  (const SpMat_t& A, const Tree_t& tree, const integer_t sep,
   std::vector<integer_t>* upd, float* subtree_work, int depth) const {
    auto chl = tree.lch()[sep];
    auto chr = tree.rch()[sep];
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
    auto sep_begin = tree.sizes()[sep];
    auto sep_end = tree.sizes()[sep+1];
    if (sep != tree.separators()-1) { // not necessary for the root
      for (integer_t c=sep_begin; c<sep_end; c++) {
        auto ice = A.get_ind()+A.get_ptr()[c+1];
        auto icb = std::lower_bound
          (A.get_ind()+A.get_ptr()[c], ice, sep_end);
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
  FrontalMatrix<scalar_t,integer_t>*
  EliminationTreeMPI<scalar_t,integer_t>::proportional_mapping
  (const SpMat_t& A, const Tree_t& tree, const SPOptions<scalar_t>& opts,
   std::vector<integer_t>* upd, float* subtree_work, SepRange& local_range,
   integer_t sep, int P0, int P, MPI_Comm front_comm,
   bool keep, bool hss_parent, int level) {
    auto sep_begin = tree.sizes()[sep];
    auto sep_end = tree.sizes()[sep+1];
    auto dim_sep = sep_end - sep_begin;
    F_t* front = nullptr;
    bool is_hss = opts.use_HSS() && hss_parent &&
      (dim_sep >= opts.HSS_min_front_size());
    if (_rank == P0) {
      if (is_hss) this->_nr_HSS_fronts++;
      else this->_nr_dense_fronts++;
    }
    if (P == 1) {
      if (keep) {
        if (is_hss)
          front = new FHSS_t(sep, sep_begin, sep_end, upd[sep]);
        else front = new FD_t(sep, sep_begin, sep_end, upd[sep]);
      }
      if (P0 == _rank) {
        local_range.first  = std::min
          (local_range.first, std::size_t(sep_begin));
        local_range.second = std::max
          (local_range.second, std::size_t(sep_end));
      }
    } else {
      if (keep) {
        if (is_hss)
          front = new FHSSMPI_t
            (_active_pfronts, sep_begin, sep_end, upd[sep], front_comm, P);
        else
          front = new FDMPI_t
            (_active_pfronts, sep_begin, sep_end, upd[sep], front_comm, P);
        if (_rank >= P0 && _rank < P0+P) _active_pfronts++;
      } else mpi_free_comm(&front_comm);
      int ctxt = front ? static_cast<FMPI_t*>(front)->blacs_context() : -1;
      int ctxt_all = front ?
        static_cast<FMPI_t*>(front)->blacs_context_all() : -1;
      _parallel_fronts.emplace_back
        (sep_begin, sep_end-sep_begin, P0, P, ctxt, ctxt_all);
    }

    // only store a node if you are part of its communicator
    // and also store your siblings!! needed for extend-add
    if (_rank < P0 || _rank >= P0+P) keep = false;
    if (P == 1 && P0 != _rank) return front;

    auto chl = tree.lch()[sep];
    auto chr = tree.rch()[sep];
    if (chl != -1) {
      float wl = subtree_work[chl];
      float wr = (chr != -1) ? subtree_work[chr] : 0.;
      int Pl = std::max(1, std::min(int(std::round(P * wl / (wl + wr))), P-1));
      int Pr = std::max(1, P - Pl);
      auto fl = proportional_mapping
        (A, tree, opts, upd, subtree_work, local_range, chl, P0, Pl,
         mpi_sub_comm(front_comm, 0, Pl), keep, is_hss, level+1);
      if (front) front->lchild = fl;
      if (chr != -1) {
        auto fr = proportional_mapping
          (A, tree, opts, upd, subtree_work, local_range, chr, P0+P-Pr, Pr,
           mpi_sub_comm(front_comm, P-Pr, Pr), keep, is_hss, level+1);
        if (front) front->rchild = fr;
      }
    } else {
      if (chr != -1) {
        auto fr = proportional_mapping
          (A, tree, opts, upd, subtree_work, local_range, chr, P0, P,
           front_comm, keep, is_hss, level+1);
        if (front) front->rchild = fr;
      }
    }
    return front;
  }

  template<typename scalar_t,typename integer_t> integer_t
  EliminationTreeMPI<scalar_t,integer_t>::maximum_rank() const {
    integer_t max_rank = EliminationTree<scalar_t,integer_t>::maximum_rank();
    MPI_Allreduce
      (MPI_IN_PLACE, &max_rank, 1, mpi_type<integer_t>(), MPI_MAX, _comm);
    return max_rank;
  }

  template<typename scalar_t,typename integer_t> long long
  EliminationTreeMPI<scalar_t,integer_t>::factor_nonzeros() const {
    long long nonzeros =
      EliminationTree<scalar_t,integer_t>::factor_nonzeros();
    MPI_Allreduce
      (MPI_IN_PLACE, &nonzeros, 1, MPI_LONG_LONG_INT, MPI_SUM, _comm);
    return nonzeros;
  }

  template<typename scalar_t,typename integer_t> long long
  EliminationTreeMPI<scalar_t,integer_t>::dense_factor_nonzeros() const {
    long long nonzeros =
      EliminationTree<scalar_t,integer_t>::dense_factor_nonzeros();
    MPI_Allreduce
      (MPI_IN_PLACE, &nonzeros, 1, MPI_LONG_LONG_INT, MPI_SUM, _comm);
    return nonzeros;
  }

} // end namespace strumpack

#endif
