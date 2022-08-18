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

#include <unordered_map>
#include <algorithm>

#include "MatrixReorderingMPI.hpp"
#include "StrumpackConfig.hpp"
#include "sparse/CSRGraph.hpp"
#include "sparse/SeparatorTree.hpp"
#include "sparse/fronts/FrontalMatrix.hpp"
#if defined(STRUMPACK_USE_SCOTCH)
#include "ScotchReordering.hpp"
#endif
#if defined(STRUMPACK_USE_PTSCOTCH)
#include "PTScotchReordering.hpp"
#endif
#include "MetisReordering.hpp"
#if defined(STRUMPACK_USE_PARMETIS)
#include "ParMetisReordering.hpp"
#endif
#include "GeometricReorderingMPI.hpp"
#include "RCMReordering.hpp"
#include "ANDSparspak.hpp"
#include "minimum_degree/AMDReordering.hpp"
#include "minimum_degree/MMDReordering.hpp"
// #include "spectral/SpectralReordering.hpp"


namespace strumpack {

  template<typename scalar_t,typename integer_t>
  MatrixReorderingMPI<scalar_t,integer_t>::MatrixReorderingMPI
  (integer_t n, const MPIComm& c)
    : MatrixReordering<scalar_t,integer_t>(n), comm_(&c), dsep_leaf_(-1) {
  }

  template<typename scalar_t,typename integer_t>
  MatrixReorderingMPI<scalar_t,integer_t>::~MatrixReorderingMPI() {}

  template<typename scalar_t,typename integer_t> int
  MatrixReorderingMPI<scalar_t,integer_t>::nested_dissection
  (const Opts_t& opts, const CSRMPI_t& A,
   int nx, int ny, int nz, int components, int width) {
    if (!is_parallel(opts.reordering_method())) {
      auto rank = comm_->rank();
      auto P = comm_->size();
      auto Aseq = A.gather_graph();
      SeparatorTree<integer_t> global_sep_tree;
      if (Aseq) { // only root
        switch (opts.reordering_method()) {
        case ReorderingStrategy::NATURAL: {
          std::iota(perm_.begin(), perm_.end(), 0);
          global_sep_tree = build_sep_tree_from_perm
            (Aseq->ptr(), Aseq->ind(), perm_, iperm_);
          break;
        }
        case ReorderingStrategy::METIS: {
          global_sep_tree = metis_nested_dissection
            (*Aseq, perm_, iperm_, opts);
          break;
        }
        case ReorderingStrategy::SCOTCH: {
#if defined(STRUMPACK_USE_SCOTCH)
          global_sep_tree = scotch_nested_dissection
            (*Aseq, perm_, iperm_, opts);
#else
          std::cerr << "ERROR: STRUMPACK was not configured with Scotch support"
                    << std::endl;
          abort();
#endif
          break;
        }
        case ReorderingStrategy::RCM: {
          global_sep_tree = rcm_reordering(*Aseq, perm_, iperm_);
          break;
        }
        case ReorderingStrategy::AMD: {
          global_sep_tree = ordering::amd_reordering(*Aseq, perm_, iperm_);
          break;
        }
        case ReorderingStrategy::MMD: {
          global_sep_tree = ordering::mmd_reordering(*Aseq, perm_, iperm_);
          break;
        }
        case ReorderingStrategy::AND: {
          global_sep_tree = ordering::and_reordering(*Aseq, perm_, iperm_);
          break;
        }
        case ReorderingStrategy::MLF: {
          std::cerr << "# ERROR: MLF ordering not supported." << std::endl;
          return 1;
        }
        case ReorderingStrategy::SPECTRAL: {
          std::cerr << "# ERROR: spectral ordering not supported." << std::endl;
          return 1;
          // global_sep_tree = ordering::spectral_nd
          //   (*Aseq, perm_, iperm_, opts.ND_options());
          // break;
        }
        default: assert(true);
        }
        Aseq.reset();
        global_sep_tree.check();
      }
      comm_->broadcast(perm_);
      comm_->broadcast(iperm_);
      integer_t nbsep;
      if (!rank) nbsep = global_sep_tree.separators();
      comm_->broadcast(nbsep);
      if (rank)
        global_sep_tree = SeparatorTree<integer_t>(nbsep);
      global_sep_tree.broadcast(*comm_);
      ltree_ = global_sep_tree.subtree(rank, P);
      tree_ = global_sep_tree.toptree(P);
      ltree_.check();
      tree_.check();
      for (std::size_t i=0; i<perm_.size(); i++)
        iperm_[perm_[i]] = i;
      get_local_graphs(A);
    } else {
      switch (opts.reordering_method()) {
      case ReorderingStrategy::GEOMETRIC: {
        if (nx*ny*nz*components != A.size()) {
          nx = opts.nx();
          ny = opts.ny();
          nz = opts.nz();
          components = opts.components();
          width = opts.separator_width();
        }
        if (nx*ny*nz*components != A.size()) {
          std::cerr << "# ERROR: Geometric reordering failed. \n"
            "# Geometric reordering only works on a regular grid "
            "and you need to provide the mesh sizes."
                    << std::endl;
          return 1;
        }
        std::tie(tree_, ltree_) =
          geometric_ND_dist(nx, ny, nz, components, width,
                            A.begin_row(), A.end_row(), *comm_, perm_, iperm_,
                            opts.nd_param(), opts.nd_planar_levels());
        break;
      }
      case ReorderingStrategy::PARMETIS: {
#if defined(STRUMPACK_USE_PARMETIS)
        tree_ = parmetis_nested_dissection
          (A, comm_->comm(), true, perm_, opts);
#else
        std::cerr << "ERROR: STRUMPACK was not configured with ParMetis support"
                  << std::endl;
        abort();
#endif
        break;
      }
      case ReorderingStrategy::PTSCOTCH: {
#if defined(STRUMPACK_USE_PTSCOTCH)
        tree_ = ptscotch_nested_dissection
          (A, comm_->comm(), true, perm_, opts);
#else
        std::cerr << "ERROR: STRUMPACK was not configured with Scotch support"
                  << std::endl;
        abort();
#endif
        break;
      }
      case ReorderingStrategy::SPECTRAL: {
        std::cerr << "# ERROR: Parallel spectral ordering not yet implemented." << std::endl;
        return 1;
      }
      default: assert(true);
      }
      tree_.check();
      get_local_graphs(A);
      if (opts.reordering_method() == ReorderingStrategy::PARMETIS ||
          opts.reordering_method() == ReorderingStrategy::PTSCOTCH)
        build_local_tree(A);
      ltree_.check();
    }
    nested_dissection_print(opts, A.nnz());
    return 0;
  }

  template<typename scalar_t,typename integer_t> int
  MatrixReorderingMPI<scalar_t,integer_t>::set_permutation
  (const Opts_t& opts, const CSRMPI_t& A, const int* p, int base) {
    auto n = perm_.size();
    assert(A.size() == integer_t(n));
    if (base == 0) std::copy(p, p+n, perm_.data());
    else for (std::size_t i=0; i<n; i++) perm_[i] = p[i] - base;
    auto Aseq = A.gather_graph();
    SeparatorTree<integer_t> global_sep_tree;
    if (Aseq) {
      global_sep_tree = build_sep_tree_from_perm
        (Aseq->ptr(), Aseq->ind(), perm_, iperm_);
      Aseq.reset();
    }
    auto rank = comm_->rank();
    auto P = comm_->size();
    integer_t nbsep;
    if (!rank) nbsep = global_sep_tree.separators();
    comm_->broadcast(nbsep);
    if (rank)
      global_sep_tree = SeparatorTree<integer_t>(nbsep);
    global_sep_tree.broadcast(*comm_);
    ltree_ = global_sep_tree.subtree(rank, P);
    tree_ = global_sep_tree.toptree(P);
    for (std::size_t i=0; i<perm_.size(); i++)
      iperm_[perm_[i]] = i;
    get_local_graphs(A);
    nested_dissection_print(opts, A.nnz());
    return 0;
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::separator_reordering
  (const Opts_t& opts, CSM_t& A, F_t* F) {
    if (opts.compression() == CompressionType::NONE)
      return;
    auto n = A.size();
    std::vector<integer_t> sorder(n, -1);
#pragma omp parallel
#pragma omp single
    F->partition_fronts(opts, A, sorder.data());
    // not all processes have all fronts, so the sorder vector needs
    // to be communicated
    comm_->all_reduce(sorder.data(), sorder.size(), MPI_MAX);
    // product of perm_ and sep_order
    for (integer_t i=0; i<n; i++) iperm_[i] = sorder[perm_[i]];
    for (integer_t i=0; i<n; i++) perm_[iperm_[i]] = i;
    std::swap(perm_, iperm_);
#pragma omp parallel
#pragma omp single
    F->permute_CB(sorder.data());
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::get_local_graphs
  (const CSRMPI_t& A) {
    auto P = comm_->size();
    auto rank = comm_->rank();
    sub_graph_ranges.resize(P);
    dist_sep_ranges.resize(P);
    proc_dist_sep.resize(tree_.separators());
    for (integer_t sep=0, p_local=0, p_dist=0;
         sep<tree_.separators(); sep++) {
      if (tree_.is_leaf(sep)) {
        // sep is a leaf, so it is the local graph of proces p
        if (p_local == rank) dsep_leaf_ = sep;
        sub_graph_ranges[p_local] =
          std::make_pair(tree_.sizes[sep], tree_.sizes[sep+1]);
        proc_dist_sep[sep] = p_local++;
      } else {
        // sep was computed using distributed nested dissection,
        // assign it to process p_dist
        dist_sep_ranges[p_dist] =
          std::make_pair(tree_.sizes[sep], tree_.sizes[sep+1]);
        proc_dist_sep[sep] = p_dist++;
      }
    }
    sub_graph_range = sub_graph_ranges[rank];
    dist_sep_range = dist_sep_ranges[rank];
    my_sub_graph = A.get_sub_graph(perm_, sub_graph_ranges);
    my_dist_sep = A.get_sub_graph(perm_, dist_sep_ranges);
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::build_local_tree
  (const CSRMPI_t& A) {
    auto P = comm_->size();
    auto rank = comm_->rank();
    auto n = A.size();
    auto sub_n = my_sub_graph.size();
    auto sub_etree =
      spsymetree(my_sub_graph.ptr(), my_sub_graph.ptr()+1,
                 my_sub_graph.ind(), sub_n, sub_graph_range.first);
    std::vector<integer_t> iwork(sub_n), post(sub_n);
    auto seps = separators_from_etree(sub_etree, post);
    for (integer_t i=0; i<sub_n; ++i)
      iwork[post[i]] = post[sub_etree[i]];
    for (integer_t i=0; i<sub_n; ++i)
      sub_etree[i] = iwork[i];
    ltree_ = SeparatorTree<integer_t>(seps);
    for (integer_t i=0; i<sub_n; i++) {
      iwork[post[i]] = i;
      post[i] += sub_graph_range.first;
    }
    my_sub_graph.permute_local
      (post, iwork, sub_graph_range.first, sub_graph_range.second);
    std::vector<integer_t> gpost(n);
    std::iota(gpost.begin(), gpost.end(), 0);
    std::unique_ptr<int[]> rcnts(new int[2*P]);
    auto displs = rcnts.get() + P;
    for (int p=0; p<P; p++) {
      rcnts[p] = sub_graph_ranges[p].second - sub_graph_ranges[p].first;
      displs[p] = sub_graph_ranges[p].first;
    }
    MPI_Allgatherv
      (post.data(), rcnts[rank], mpi_type<integer_t>(), gpost.data(),
       rcnts.get(), displs, mpi_type<integer_t>(), comm_->comm());
    for (integer_t i=0; i<n; i++)
      iperm_[perm_[i]] = i;
    for (integer_t i=0; i<n; i++)
      perm_[gpost[i]] = iperm_[i];
    for (integer_t i=0; i<n; i++)
      iperm_[perm_[i]] = i;
    std::swap(perm_, iperm_);
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::clear_tree_data() {
    MatrixReordering<scalar_t,integer_t>::clear_tree_data();
    ltree_ = SeparatorTree<integer_t>();
    my_sub_graph = CSRGraph<integer_t>();
    my_dist_sep = CSRGraph<integer_t>();
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::nested_dissection_print
  (const Opts_t& opts, integer_t nnz) const {
    if (opts.verbose()) {
      auto total_separators =
        comm_->all_reduce(ltree_.separators(), MPI_SUM) +
        std::max(integer_t(0), tree_.separators() - comm_->size());
      integer_t local_levels;
      if (dsep_leaf_ != -1)
        local_levels = ltree_.levels() + tree_.level(dsep_leaf_) - 1;
      else local_levels = tree_.levels();
      integer_t max_level = comm_->all_reduce(local_levels, MPI_MAX);
      if (comm_->is_root())
        MatrixReordering<scalar_t,integer_t>::nested_dissection_print
          (opts, nnz, max_level, total_separators, true);
    }
  }

  // explicit template instantiations
  template class MatrixReorderingMPI<float,int>;
  template class MatrixReorderingMPI<double,int>;
  template class MatrixReorderingMPI<std::complex<float>,int>;
  template class MatrixReorderingMPI<std::complex<double>,int>;

  template class MatrixReorderingMPI<float,long int>;
  template class MatrixReorderingMPI<double,long int>;
  template class MatrixReorderingMPI<std::complex<float>,long int>;
  template class MatrixReorderingMPI<std::complex<double>,long int>;

  template class MatrixReorderingMPI<float,long long int>;
  template class MatrixReorderingMPI<double,long long int>;
  template class MatrixReorderingMPI<std::complex<float>,long long int>;
  template class MatrixReorderingMPI<std::complex<double>,long long int>;

} // end namespace strumpack

