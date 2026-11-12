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
#include <cstdlib>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/statvfs.h>
#endif

#include "MatrixReorderingMPI.hpp"
#include "StrumpackConfig.hpp"
#include "sparse/CSRGraph.hpp"
#include "sparse/SeparatorTree.hpp"
#include "sparse/fronts/Front.hpp"
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
    : MatrixReordering<scalar_t,integer_t>(0), comm_(&c), n_(n),
      dsep_leaf_(-1) {
    // perm_ and iperm_ are allocated when the permutation is
    // computed, and not at all when it is shared per node
  }

  template<typename scalar_t,typename integer_t>
  MatrixReorderingMPI<scalar_t,integer_t>::~MatrixReorderingMPI() {
    free_shared_perm();
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::alloc_perm() {
    perm_.resize(n_);
    iperm_.resize(n_);
  }

  /**
   * Collective. Frees the node-shared permutation, if any.
   */
  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::free_shared_perm() {
    if (shm_win_ == MPI_WIN_NULL) return;
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Win_unlock_all(shm_win_);
      MPI_Win_free(&shm_win_);
      MPI_Comm_free(&shm_comm_);
    }
    shm_win_ = MPI_WIN_NULL;
    shm_comm_ = MPI_COMM_NULL;
    shm_perm_ = shm_iperm_ = nullptr;
  }

  /**
   * Collective. On entry the root holds the permutation in perm_ (and
   * the other processes nothing). The lowest rank of every node
   * allocates a shared window of 2n integers, the root broadcasts the
   * permutation to these node leaders, which compute the inverse, and
   * all processes of the node read both with perm_data() and
   * iperm_data(). perm_ and iperm_ are then released. Returns false,
   * on all processes, if a window could not be allocated; perm_ and
   * iperm_ are then unchanged.
   *
   * For testing, STRUMPACK_SHARED_PERM_GROUP=k makes groups of k
   * consecutive ranks act as nodes (the ranks of a group must still
   * be able to share memory), so several node leaders can be tested
   * on one node.
   */
  template<typename scalar_t,typename integer_t> bool
  MatrixReorderingMPI<scalar_t,integer_t>::share_perm() {
    free_shared_perm();
    auto comm = comm_->comm();
    int rank = comm_->rank(), lrank;
    if (auto g = std::getenv("STRUMPACK_SHARED_PERM_GROUP"))
      MPI_Comm_split(comm, rank / std::max(1, std::atoi(g)), rank, &shm_comm_);
    else
      MPI_Comm_split_type
        (comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shm_comm_);
    MPI_Comm_rank(shm_comm_, &lrank);
    MPI_Comm_set_errhandler(shm_comm_, MPI_ERRORS_RETURN);
    MPI_Aint bytes = (lrank == 0) ? shared_perm_bytes() : 0;
    // The window is usually backed by a file in /dev/shm (Linux). A
    // failure for lack of space there cannot be handled. Open MPI
    // (5.0) asks for 5% more free space than the size of the file
    // (the window and a header), and when the node leader cannot
    // create it, the other processes of the node keep waiting in
    // MPI_Win_allocate_shared. And the pages of the file are not
    // reserved: if the space runs out later, writing to the window
    // raises SIGBUS. So check the free space of /dev/shm, if there
    // is one, before allocating, with a margin of 10% and 1 MB. This
    // is a best effort: the free space can change, and the MPI
    // library can put the file elsewhere.
    int ok = 1;
#if defined(__unix__) || defined(__APPLE__)
    struct statvfs fs;
    if (bytes && !statvfs("/dev/shm", &fs) &&
        double(fs.f_bavail) * double(fs.f_frsize) <
        1.1 * double(bytes) + double(1 << 20))
      ok = 0;
#endif
    // An error of MPI_Win_allocate_shared on all processes of a node
    // (an MPI library or configuration without shared memory
    // windows) is handled: all processes then keep a copy.
    integer_t* base = nullptr;
    ok = comm_->all_reduce(ok, MPI_MIN);
    if (ok) {
      int ierr = MPI_Win_allocate_shared
        (bytes, sizeof(integer_t), MPI_INFO_NULL, shm_comm_, &base, &shm_win_);
      ok = comm_->all_reduce(int(ierr == MPI_SUCCESS), MPI_MIN);
      if (!ok && ierr == MPI_SUCCESS) MPI_Win_free(&shm_win_);
    }
    if (!ok) {
      shm_win_ = MPI_WIN_NULL;
      MPI_Comm_free(&shm_comm_);
      shm_comm_ = MPI_COMM_NULL;
      return false;
    }
    MPI_Aint qbytes;
    int qdisp;
    MPI_Win_shared_query(shm_win_, 0, &qbytes, &qdisp, &base);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, shm_win_);
    shm_perm_ = base;
    shm_iperm_ = base + n_;
    MPI_Comm leaders;
    MPI_Comm_split(comm, lrank == 0 ? 0 : MPI_UNDEFINED, rank, &leaders);
    if (lrank == 0) {
      // the root is the leader of its node, and rank 0 of leaders
      if (rank == 0) std::copy(perm_.begin(), perm_.end(), shm_perm_);
      const std::size_t n = n_, chunk = std::size_t(1) << 28;
      for (std::size_t i=0; i<n; i+=chunk)
        MPI_Bcast(shm_perm_+i, int(std::min(chunk, n-i)),
                  mpi_type<integer_t>(), 0, leaders);
      for (integer_t i=0; i<n_; i++) shm_iperm_[shm_perm_[i]] = i;
      MPI_Comm_free(&leaders);
    }
    MPI_Win_sync(shm_win_);
    MPI_Barrier(shm_comm_);
    MPI_Win_sync(shm_win_);
    std::vector<integer_t>().swap(perm_);
    std::vector<integer_t>().swap(iperm_);
    return true;
  }

  /**
   * Collective. Replaces the node-shared permutation, if any, by a
   * copy per process.
   */
  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::unshare_perm() {
    if (shm_win_ == MPI_WIN_NULL) return;
    alloc_perm();
    std::copy(shm_perm_, shm_perm_+n_, perm_.data());
    std::copy(shm_iperm_, shm_iperm_+n_, iperm_.data());
    free_shared_perm();
  }

  template<typename scalar_t,typename integer_t> int
  MatrixReorderingMPI<scalar_t,integer_t>::nested_dissection
  (const Opts_t& opts, const CSRMPI_t& A,
   int nx, int ny, int nz, int components, int width) {
    free_shared_perm();
    if (!is_parallel(opts.reordering_method())) {
      auto rank = comm_->rank();
      auto P = comm_->size();
      // separator_reordering (compression) modifies the permutation
      bool shared = opts.use_shared_permutation() &&
        opts.compression() == CompressionType::NONE;
      if (!shared || !rank) alloc_perm();
      auto Aseq = A.gather_graph();
      SeparatorTree<integer_t> global_sep_tree;
      // with Schur variables, order the graph of the other variables
      std::vector<integer_t> other;
      if (Aseq && !this->schur_.empty()) {
        *Aseq = this->schur_free_graph(*Aseq, other);
        perm_.resize(Aseq->size());
        iperm_.resize(Aseq->size());
      }
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
        this->add_schur_root(global_sep_tree, other);
        global_sep_tree.check();
      }
      if (shared && !share_perm()) {
        if (!rank)
          std::cerr << "# WARNING: enable_shared_permutation: could not"
                    << " allocate a shared memory window of "
                    << shared_perm_bytes() / 1e6 << " MB per node"
                    << " (MPI_Win_allocate_shared failed, or not enough"
                    << " free space in /dev/shm), using a copy of the"
                    << " permutation per process" << std::endl;
        shared = false;
        if (rank) alloc_perm();
      }
      if (!shared) {
        comm_->broadcast(perm_);
        comm_->broadcast(iperm_);
      }
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
      if (!shared)
        for (std::size_t i=0; i<perm_.size(); i++)
          iperm_[perm_[i]] = i;
      get_local_graphs(A);
    } else {
      alloc_perm();
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
        int thread_level;
        MPI_Query_thread(&thread_level);
        if (thread_level != MPI_THREAD_MULTIPLE && comm_->is_root())
          std::cerr << "WARNING: MPI_THREAD_MULTIPLE might be required for PTSCOTCH"
                    << std::endl;
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
    free_shared_perm();
    alloc_perm();
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
    // build_sep_tree_from_perm changes the permutation (postorder of
    // the elimination tree), on the root only
    comm_->broadcast(perm_);
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
    // compression enabled after a reordering with the shared
    // permutation, which is modified here
    unshare_perm();
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
    my_sub_graph = A.get_sub_graph(perm_data(), sub_graph_ranges);
    my_dist_sep = A.get_sub_graph(perm_data(), dist_sep_ranges);
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
    for (integer_t i = 0; i < sub_n; ++i) {
      const auto parent = sub_etree[i];
      iwork[post[i]] =
        (parent >= 0 && parent < sub_n) ? post[parent] : sub_n;
      }
      for (integer_t i = 0; i < sub_n; ++i) {
      sub_etree[i] = iwork[i];
    }
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
          (opts, n_, nnz, max_level, total_separators, true);
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

