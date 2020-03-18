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
#include <algorithm>
#include <vector>
#include <memory>

#include "EliminationTreeMPI.hpp"

#include "ordering/MatrixReorderingMPI.hpp"
#include "dense/DistributedMatrix.hpp"
#include "fronts/FrontFactory.hpp"
#include "fronts/FrontalMatrixMPI.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  EliminationTreeMPI<scalar_t,integer_t>::EliminationTreeMPI
  (const MPIComm& comm) : EliminationTree<scalar_t,integer_t>(),
    comm_(comm), rank_(comm.rank()), P_(comm.size()) {
  }

  template<typename scalar_t,typename integer_t>
  EliminationTreeMPI<scalar_t,integer_t>::~EliminationTreeMPI() {}

  template<typename scalar_t,typename integer_t>
  EliminationTreeMPI<scalar_t,integer_t>::EliminationTreeMPI
  (const SPOptions<scalar_t>& opts, const SpMat_t& A,
   Reord_t& nd, const MPIComm& comm)
    : EliminationTree<scalar_t,integer_t>(),
    comm_(comm), rank_(comm.rank()), P_(comm.size()), active_pfronts_(0) {
    auto& tree = nd.tree();
    std::vector<std::vector<integer_t>> upd(tree.separators());
    std::vector<float> subtree_work(tree.separators());
#pragma omp parallel default(shared)
#pragma omp single
    symbolic_factorization(A, tree, tree.root(), upd, subtree_work);
    local_range_ = {A.size(), 0};
    this->root_ = proportional_mapping
      (tree, opts, upd, subtree_work, tree.root(),
       0, comm_.size(), comm_, true, true, 0);
    subtree_ranges_.resize(P_);
    MPI_Allgather
      (&local_range_, sizeof(SepRange), MPI_BYTE, subtree_ranges_.data(),
       sizeof(SepRange), MPI_BYTE, comm_.comm());
    /* do not clear the tree data, because if we update the matrix
     * values, we want to reuse this information */
    // nd.clear_tree_data();
  }

  template<typename scalar_t,typename integer_t> FrontCounter
  EliminationTreeMPI<scalar_t,integer_t>::front_counter() const {
    return this->nr_fronts_.reduce(comm_);
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPI<scalar_t,integer_t>::update_local_ranges
  (integer_t lo, integer_t hi) {
    local_range_.first  = std::min(local_range_.first, lo);
    local_range_.second = std::max(local_range_.second, hi);
  }

  // TODO: rewrite this with a single alltoallv/allgatherv
  template<typename scalar_t,typename integer_t>
  std::unique_ptr<DistributedMatrix<scalar_t>[]>
  EliminationTreeMPI<scalar_t,integer_t>::sequential_to_block_cyclic
  (DenseM_t& x) const {
    std::size_t pos = 0;
    for (auto& pf : parallel_fronts_)
      if (rank_ >= pf.P0 && rank_ < pf.P0+pf.P) pos++;
    std::unique_ptr<DistM_t[]> x_dist(new DistM_t[pos]);
    pos = 0;
    for (auto& pf : parallel_fronts_)
      if (rank_ >= pf.P0 && rank_ < pf.P0+pf.P) {
        x_dist[pos] = DistM_t(pf.grid, pf.dim_sep, x.cols());
        DenseM_t xloc(pf.dim_sep, x.cols());
        copy(pf.dim_sep, x.cols(), x, pf.sep_begin, 0, xloc, 0, 0);
        x_dist[pos].scatter(xloc);
        pos++;
      }
    return x_dist;
  }

  // TODO: rewrite this with a single alltoallv/allgatherv
  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPI<scalar_t,integer_t>::block_cyclic_to_sequential
  (DenseM_t& x, const DistM_t* x_dist) const {
    auto iwork = std::unique_ptr<int[]>(new int[2*P_]);
    auto cnts = iwork.get();
    auto disp = cnts + P_;
    for (int p=0; p<P_; p++) {
      cnts[p] = std::max
        (integer_t(0), subtree_ranges_[p].second - subtree_ranges_[p].first);
      disp[p] = subtree_ranges_[p].first;
    }
    for (std::size_t c=0; c<x.cols(); c++)
      MPI_Allgatherv
        (MPI_IN_PLACE, 0, mpi_type<scalar_t>(), x.ptr(0, c),
         cnts, disp, mpi_type<scalar_t>(), comm_.comm());
    auto xd = x_dist;
    for (auto& pf : parallel_fronts_) {
      if (rank_ >= pf.P0 && rank_ < pf.P0+pf.P) {
        auto xloc = (xd++)->gather();
        if (rank_ == pf.P0)
          copy(xloc.rows(), xloc.cols(), xloc, 0, 0, x, pf.sep_begin, 0);
      }
      for (std::size_t c=0; c<x.cols(); c++)
        MPI_Bcast
          (x.ptr(pf.sep_begin, c), pf.dim_sep,
           mpi_type<scalar_t>(), pf.P0, comm_.comm());
    }
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPI<scalar_t,integer_t>::multifrontal_solve
  (DenseM_t& x) const {
    auto x_dist = sequential_to_block_cyclic(x);
    this->root_->multifrontal_solve(x, x_dist.get());
    block_cyclic_to_sequential(x, x_dist.get());
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPI<scalar_t,integer_t>::symbolic_factorization
  (const SpMat_t& A, const Tree_t& tree, const integer_t sep,
   std::vector<std::vector<integer_t>>& upd,
   std::vector<float>& subtree_work, int depth) const {
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
  (Tree_t& tree, const SPOptions<scalar_t>& opts,
   std::vector<std::vector<integer_t>>& upd, std::vector<float>& subtree_work,
   integer_t sep, int P0, int P, const MPIComm& fcomm,
   bool keep, bool hss_parent, int level) {
    auto sep_begin = tree.sizes(sep);
    auto sep_end = tree.sizes(sep+1);
    auto dim_sep = sep_end - sep_begin;
    std::unique_ptr<F_t> front;
    if (P == 1) {
      if (keep) {
        front = create_frontal_matrix<scalar_t,integer_t>
          (opts, sep, sep_begin, sep_end, upd[sep], hss_parent,
           level, this->nr_fronts_, rank_ == P0);
      }
      if (P0 == rank_) update_local_ranges(sep_begin, sep_end);
    } else {
      if (keep) {
        front = create_frontal_matrix<scalar_t,integer_t>
          (opts, active_pfronts_, sep_begin, sep_end, upd[sep],
           hss_parent, level, this->nr_fronts_, fcomm, P, rank_ == P0);
        if (rank_ >= P0 && rank_ < P0+P) active_pfronts_++;
      }
      using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
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
    bool use_compression = is_compressed
      (dim_sep, upd[sep].size(), hss_parent, opts);
    if (chl != -1) {
      float wl = subtree_work[chl];
      float wr = (chr != -1) ? subtree_work[chr] : 0.;
      int Pl = std::max(1, std::min(int(std::round(P * wl / (wl + wr))), P-1));
      int Pr = std::max(1, P - Pl);
      auto fl = proportional_mapping
        (tree, opts, upd, subtree_work, chl, P0, Pl,
         fcomm.sub(0, Pl), keep, use_compression, level+1);
      if (front) front->set_lchild(std::move(fl));
      if (chr != -1) {
        auto fr = proportional_mapping
          (tree, opts, upd, subtree_work, chr, P0+P-Pr, Pr,
           fcomm.sub(P-Pr, Pr), keep, use_compression, level+1);
        if (front) front->set_rchild(std::move(fr));
      }
    } else {
      if (chr != -1) {
        auto fr = proportional_mapping
          (tree, opts, upd, subtree_work, chr, P0, P,
           fcomm, keep, use_compression, level+1);
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

  // explicit template specializations
  template class EliminationTreeMPI<float,int>;
  template class EliminationTreeMPI<double,int>;
  template class EliminationTreeMPI<std::complex<float>,int>;
  template class EliminationTreeMPI<std::complex<double>,int>;

  template class EliminationTreeMPI<float,long int>;
  template class EliminationTreeMPI<double,long int>;
  template class EliminationTreeMPI<std::complex<float>,long int>;
  template class EliminationTreeMPI<std::complex<double>,long int>;

  template class EliminationTreeMPI<float,long long int>;
  template class EliminationTreeMPI<double,long long int>;
  template class EliminationTreeMPI<std::complex<float>,long long int>;
  template class EliminationTreeMPI<std::complex<double>,long long int>;

} // end namespace strumpack
