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
#ifndef MATRIX_REORDERING_MPI_HPP
#define MATRIX_REORDERING_MPI_HPP
#include <unordered_map>
#include <algorithm>
#include "CSRGraph.hpp"
#include "StrumpackConfig.hpp"
#if defined(STRUMPACK_USE_SCOTCH)
#include "PTScotchReordering.hpp"
#endif
#if defined(STRUMPACK_USE_PARMETIS)
#include "ParMetisReordering.hpp"
#endif
#include "GeometricReorderingMPI.hpp"

namespace strumpack {

  /**
   * A MatrixReorderingMPI has a distributed separator tree. This tree
   * is stored using 2 trees, one in the base class MatrixReordering
   * and one here in MatrixReorderingMPI. The tree in MatrixReordering
   * (sep_tree_) is the top of the tree, corresponding to the
   * distributed separators. The tree stored here (local_tree_)
   * corresponds to the local subtree.
   *
   * The distributed tree should have P leafs.  Lets number the
   * distributed separators level by level, root=0.  For instance: for
   * P=5, the distributed tree will look like:
   *                   1
   *                  / \
   *                 /   \
   *                2     3
   *               / \   / \
   *              4   5 6   7
   *             / \
   *            8   9
   *
   * The number of the node is given by dist_sep_id, left child has
   * id 2*dist_sep_id, right child 2*dist_sep_id+1.  Nodes for which
   * P<=dist_sep_id<2*P, are leafs of the distributed separator tree,
   * and they form the roots of the local subtree for process
   * dist_sep_id-P.
   *
   * Furthermore, each proces has a local tree rooted at one of the
   * leafs of the distributed tree.  To have the same convention as used
   * for PTScotch, the leafs are assigned to procs as in a postordering
   * of the distributed tree, hence for this example, proc_dist_sep:
   *                   3
   *                  / \
   *                 /   \
   *                1     2
   *               / \   / \
   *              0   2 3   4
   *             / \
   *            0   1
   */
  template<typename scalar_t,typename integer_t>
  class MatrixReorderingMPI
    : public MatrixReordering<scalar_t,integer_t> {

  public:

    MatrixReorderingMPI(integer_t n, const MPIComm& c);

    virtual ~MatrixReorderingMPI() {}

    int nested_dissection
    (const SPOptions<scalar_t>& opts,
     const CSRMatrixMPI<scalar_t,integer_t>& A,
     int nx, int ny, int nz, int components, int width);

    void separator_reordering
    (const SPOptions<scalar_t>& opts,
     const CSRMatrixMPI<scalar_t,integer_t>& A,
     FrontalMatrix<scalar_t,integer_t>& F);

    void separator_reordering
    (const SPOptions<scalar_t>& opts,
     const CSRMatrixMPI<scalar_t,integer_t>& A, bool verbose);

    void clear_tree_data();

    /**
     * proc_dist_sep[sep] holds the rank of the process responsible
     * for distributed separator sep
     * - if sep is a leaf of the distributed tree, proc_dist_sep
     *    points to the process holding that subgraph as my_sub_graph
     * - if sep is a non-leaf, proc_dist_sep points to the process
     *    holding the graph of the distributed separator sep as
     *    my_dist_sep
     */
    std::vector<integer_t> proc_dist_sep;

    /**
     * Every process is responsible for one local subgraph of A.  The
     * distributed nested dissection will create a separator tree with
     * exactly P leafs.  Each process takes one of those leafs and
     * stores the corresponding part of the permuted matrix A in
     * sub_graph_A.
     */
    CSRGraph<integer_t> my_sub_graph;

    /**
     * Every process is responsible for one separator from the
     * distributed part of nested dissection.  The graph of the
     * distributed separator for which this rank is responsible is
     * stored as dist_sep_A.
     */
    CSRGraph<integer_t> my_dist_sep;

    std::vector<std::pair<integer_t,integer_t>> sub_graph_ranges;
    std::vector<std::pair<integer_t,integer_t>> dist_sep_ranges;

    std::pair<integer_t,integer_t> sub_graph_range;
    std::pair<integer_t,integer_t> dist_sep_range;

    /**
     * Number of the node in sep_tree corresponding to the distributed
     * separator owned by this processor.
     */
    integer_t dsep_internal;
    /**
     * Number of the node in sep_tree corresponding to the root of the
     * local subtree.
     */
    integer_t dsep_leaf;

    const SeparatorTree<integer_t>& local_tree() const { return *local_tree_; }

  private:
    const MPIComm* comm_;
    std::unique_ptr<SeparatorTree<integer_t>> local_tree_;

    void get_local_graphs(const CSRMatrixMPI<scalar_t,integer_t>& Ampi);

    void build_local_tree(const CSRMatrixMPI<scalar_t,integer_t>& Ampi);

    void distributed_separator_reordering
    (const SPOptions<scalar_t>& opts, std::vector<integer_t>& gorder);

    void local_separator_reordering
    (const SPOptions<scalar_t>& opts, std::vector<integer_t>& gorder);

    void local_separator_reordering_recursive
    (const SPOptions<scalar_t>& opts, integer_t sep, bool compressed_parent,
     std::vector<integer_t>& lorder, std::vector<integer_t>& liorder);

    void split_recursive_local
    (const SPOptions<scalar_t>& opts, integer_t sep,
     std::vector<integer_t>& lorder, HSS::HSSPartitionTree& partition_tree,
     integer_t& nr_parts, integer_t part, integer_t count,
     const Length2Edges<integer_t>& l2);

    void split_recursive_dist
    (const SPOptions<scalar_t>& opts, std::vector<integer_t>& dorder,
     HSS::HSSPartitionTree& partition_tree,
     integer_t& nr_parts, integer_t part, integer_t count,
     const Length2Edges<integer_t>& l2);

    void nested_dissection_print
    (const SPOptions<scalar_t>& opts, integer_t nnz) const;

    using MatrixReordering<scalar_t,integer_t>::perm_;
    using MatrixReordering<scalar_t,integer_t>::iperm_;
    using MatrixReordering<scalar_t,integer_t>::sep_tree_;
  };

  template<typename scalar_t,typename integer_t>
  MatrixReorderingMPI<scalar_t,integer_t>::MatrixReorderingMPI
  (integer_t n, const MPIComm& c)
    : MatrixReordering<scalar_t,integer_t>(n), dsep_internal(0),
    dsep_leaf(0), comm_(&c) {
  }

  template<typename scalar_t,typename integer_t> int
  MatrixReorderingMPI<scalar_t,integer_t>::nested_dissection
  (const SPOptions<scalar_t>& opts, const CSRMatrixMPI<scalar_t,integer_t>& A,
   int nx, int ny, int nz, int components, int width) {
    if (!is_parallel(opts.reordering_method())) {
      auto rank = comm_->rank();
      auto P = comm_->size();
      // TODO only gather the graph? without the values or the diagonal
      auto Aseq = A.gather();
      std::unique_ptr<SeparatorTree<integer_t>> global_sep_tree;
      if (Aseq) { // only root
        switch (opts.reordering_method()) {
        case ReorderingStrategy::NATURAL: {
          for (integer_t i=0; i<A.size(); i++) perm_[i] = i;
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
        default: assert(true);
        }
        Aseq.reset();
        global_sep_tree->check();
      }
      comm_->broadcast(perm_);
      comm_->broadcast(iperm_);
      integer_t nbsep;
      if (!rank) nbsep = global_sep_tree->separators();
      comm_->broadcast(nbsep);
      if (rank)
        global_sep_tree = std::unique_ptr<SeparatorTree<integer_t>>
          (new SeparatorTree<integer_t>(nbsep));
      global_sep_tree->broadcast(comm_->comm());
      local_tree_ = global_sep_tree->subtree(rank, P);
      sep_tree_ = global_sep_tree->toptree(P);
      local_tree_->check();
      sep_tree_->check();
      for (std::size_t i=0; i<perm_.size(); i++) iperm_[perm_[i]] = i;
      get_local_graphs(A);
    } else {
      switch (opts.reordering_method()) {
      case ReorderingStrategy::GEOMETRIC: {
        if (nx*ny*nz*components != A.size()) {
          nx = opts.nx();
          ny = opts.nz();
          nz = opts.nz();
          components = opts.components();
          width = opts.separator_width();
        }
        if (nx*ny*nz*components != A.size()) {
          std::cerr << "# ERROR: Geometric reordering failed. \n"
            "# Geometric reordering only works"
            "on a simple 3 point wide stencil\n"
            "# on a regular grid and you need to provide the mesh sizes."
                    << std::endl;
          return 1;
        }
        std::tie(sep_tree_, local_tree_) =
          geometric_nested_dissection_dist
          (nx, ny, nz, components, width, A.begin_row(), A.end_row(),
           comm_->comm(), perm_, iperm_, opts.nd_param(),
           opts.compression_leaf_size(), opts.compression_min_sep_size());
        break;
      }
      case ReorderingStrategy::PARMETIS: {
#if defined(STRUMPACK_USE_PARMETIS)
        sep_tree_ = parmetis_nested_dissection
          (A, comm_->comm(), true, perm_, opts);
#else
        std::cerr << "ERROR: STRUMPACK was not configured with ParMetis support"
                  << std::endl;
        abort();
#endif
        break;
      }
      case ReorderingStrategy::PTSCOTCH: {
#if defined(STRUMPACK_USE_SCOTCH)
        sep_tree_ = ptscotch_nested_dissection
          (A, comm_->comm(), true, perm_, opts);
#else
        std::cerr << "ERROR: STRUMPACK was not configured with Scotch support"
                  << std::endl;
        abort();
#endif
        break;
      }
      default: assert(true);
      }
      sep_tree_->check();
      get_local_graphs(A);
      if (opts.reordering_method() == ReorderingStrategy::PARMETIS ||
          opts.reordering_method() == ReorderingStrategy::PTSCOTCH)
        build_local_tree(A);
      local_tree_->check();
    }
    nested_dissection_print(opts, A.nnz());
    return 0;
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::separator_reordering
  (const SPOptions<scalar_t>& opts, const CSRMatrixMPI<scalar_t,integer_t>& A,
   bool verbose) {
    // if (opts.reordering_method() == ReorderingStrategy::GEOMETRIC ||
    //     opts.compression() == CompressionType::NONE)
    //   return;

    if (opts.compression() == CompressionType::NONE)
      return;
    auto n = A.size();
    std::vector<integer_t> gorder(n);
    std::iota(gorder.begin(), gorder.end(), 0);
    distributed_separator_reordering(opts, gorder);
    local_separator_reordering(opts, gorder);
    for (integer_t i=0; i<n; i++) perm_[gorder[i]] = iperm_[i];
    for (integer_t i=0; i<n; i++) iperm_[perm_[i]] = i;
    std::swap(perm_, iperm_);
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::separator_reordering
  (const SPOptions<scalar_t>& opts, const CSRMatrixMPI<scalar_t,integer_t>& A,
   FrontalMatrix<scalar_t,integer_t>& F) {
    if (opts.reordering_method() == ReorderingStrategy::GEOMETRIC ||
        opts.compression() == CompressionType::NONE)
      return;
    std::vector<integer_t> sorder(A.size());
#pragma omp parallel
#pragma omp single
    F.bisection_partitioning(opts, sorder.data());
    std::cout << "TODO MatrixReorderingMPI::separator_reordering"
              << std::endl;
  }


  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::split_recursive_local
  (const SPOptions<scalar_t>& opts, integer_t sep,
   std::vector<integer_t>& lorder, HSS::HSSPartitionTree& partition_tree,
   integer_t& nr_parts, integer_t part, integer_t count,
   const Length2Edges<integer_t>& l2) {
    auto sep_begin = local_tree_->sizes(sep);
    auto sep_end = local_tree_->sizes(sep+1);
    auto dim_sep = sep_end - sep_begin;
    auto sg = my_sub_graph.extract_subgraph
      (opts.separator_ordering_level(), sub_graph_range.first,
       sep_begin, sep_end, part, &lorder[sep_begin], l2);
    idx_t edge_cut = 0, nvtxs = sg.size();
    std::vector<idx_t> partitioning(nvtxs);
    int info = WRAPPER_METIS_PartGraphRecursive
      (nvtxs, 1, sg.ptr(), sg.ind(), 2, edge_cut, partitioning);
    if (info !=  METIS_OK) {
      std::cerr << "METIS_PartGraphRecursive for separator"
        " reordering returned: " << info << std::endl;
      exit(1);
    }
    partition_tree.c.resize(2);
    for (integer_t i=sep_begin, j=0; i<sep_end; i++)
      if (lorder[i] == part) {
        auto p = partitioning[j++];
        lorder[i] = -count - p;
        partition_tree.c[p].size++;
      }
    for (integer_t p=0; p<2; p++) {
      if (partition_tree.c[p].size > 2 * opts.compression_leaf_size())
        split_recursive_local(opts, sep, lorder, partition_tree.c[p],
                              nr_parts, -count-p, count+2, l2);
      else
        std::replace(&lorder[sep_begin], &lorder[sep_end],
                     -count-p, nr_parts++);
    }
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::split_recursive_dist
  (const SPOptions<scalar_t>& opts, std::vector<integer_t>& dorder,
   HSS::HSSPartitionTree& partition_tree,
   integer_t& nr_parts, integer_t part, integer_t count,
   const Length2Edges<integer_t>& l2) {
    auto dsep_begin = dist_sep_range.first;
    auto dsep_end = dist_sep_range.second;
    auto dim_dsep = dsep_end - dsep_begin;
    auto sg = my_dist_sep.extract_subgraph
      (opts.separator_ordering_level(), dsep_begin, 0, dim_dsep,
       part, dorder.data(), l2);
    idx_t edge_cut = 0, nvtxs = sg.size();
    std::vector<idx_t> partitioning(dim_dsep);
    int info = WRAPPER_METIS_PartGraphRecursive
      (nvtxs, 1, sg.ptr(), sg.ind(), 2, edge_cut, partitioning);
    if (info !=  METIS_OK) {
      std::cerr << "METIS_PartGraphRecursive for separator"
        " reordering returned: " << info << std::endl;
      exit(1);
    }
    partition_tree.c.resize(2);
    for (integer_t i=0, j=0; i<dim_dsep; i++)
      if (dorder[i] == part) {
        auto p = partitioning[j++];
        dorder[i] = -count - p;
        partition_tree.c[p].size++;
      }
    for (integer_t p=0; p<2; p++)
      if (partition_tree.c[p].size > 2 * opts.compression_leaf_size())
        split_recursive_dist
          (opts, dorder, partition_tree.c[p], nr_parts, -count-p, count+2, l2);
      else
        std::replace
          (dorder.begin(), dorder.end(), -count-p, nr_parts++);
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::local_separator_reordering_recursive
  (const SPOptions<scalar_t>& opts, integer_t sep, bool compressed_parent,
   std::vector<integer_t>& lorder, std::vector<integer_t>& liorder) {
    auto sep_begin = local_tree_->sizes(sep);
    auto sep_end = local_tree_->sizes(sep+1);
    auto dim_sep = sep_end - sep_begin;
    bool compressed = is_compressed(dim_sep, compressed_parent, opts);
    if (compressed) {
      HSS::HSSPartitionTree tree(dim_sep);
      int leaf = opts.compression_leaf_size();
      if (dim_sep > 2 * leaf) {
        std::fill(&lorder[sep_begin], &lorder[sep_end], integer_t(0));
        integer_t parts = 0;
        auto l2 = my_sub_graph.length_2_edges(sub_graph_range.first);
        split_recursive_local(opts, sep, lorder, tree, parts, 0, 1, l2);
        for (integer_t part=0, count=sep_begin+sub_graph_range.first;
             part<parts; part++)
          for (integer_t i=sep_begin; i<sep_end; i++)
            if (lorder[i] == part)
              lorder[i] = -count++;
        for (integer_t i=sep_begin; i<sep_end; i++) {
          lorder[i] = -lorder[i];
          liorder[lorder[i]-sub_graph_range.first] = i;
        }
      }
      if (opts.use_BLR() || opts.use_HODLR()) {
        auto tiles = tree.leaf_sizes();
        integer_t nt = tiles.size();
        DenseMatrix<bool> adm(nt, nt);
        adm.fill(true);
        for (integer_t t=0; t<nt; t++)
          adm(t, t) = false;
        if (opts.use_HODLR() ||
            opts.BLR_options().admissibility() ==
            BLR::Admissibility::STRONG) {
          std::vector<integer_t> ts(nt+1);
          for (integer_t i=0; i<nt; i++)
            ts[i+1] = tiles[i] + ts[i];
          auto& G = my_sub_graph;
          for (integer_t t=0; t<nt; t++) {
            for (integer_t i=ts[t]; i<ts[t+1]; i++) {
              auto Gi = liorder[i+sep_begin];
              auto hij = G.ind() + G.ptr(Gi+1);
              for (auto pj=G.ind()+G.ptr(Gi); pj!=hij; pj++) {
                auto Gj = *pj - sub_graph_range.first;
                if (Gj < sep_begin || Gj >= sep_end) continue;
                integer_t tj = std::distance
                  (ts.begin(), std::upper_bound
                   (ts.begin(), ts.end(),
                    lorder[Gj]-sub_graph_range.first-sep_begin)) - 1;
                if (t != tj) adm(t, tj) = adm(tj, t) = false;
              }
            }
          }
        }
        local_tree_->admissibility[sep] = std::move(adm);
      }
      local_tree_->partition_tree[sep] = std::move(tree);
    }
    if (local_tree_->lch(sep) != -1)
      local_separator_reordering_recursive
        (opts, local_tree_->lch(sep), compressed, lorder, liorder);
    if (local_tree_->rch(sep) != -1)
      local_separator_reordering_recursive
        (opts, local_tree_->rch(sep), compressed, lorder, liorder);
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::local_separator_reordering
  (const SPOptions<scalar_t>& opts, std::vector<integer_t>& gorder) {
    auto P = comm_->size();
    auto n_local = my_sub_graph.size();
    std::vector<integer_t> lorder(n_local), liorder(n_local);
    std::iota(lorder.begin(), lorder.end(), sub_graph_range.first);
    std::iota(liorder.begin(), liorder.end(), 0);
    if (local_tree_->separators() > 0)
      local_separator_reordering_recursive
        (opts, local_tree_->root(), true, lorder, liorder);
    my_sub_graph.permute_local
      (lorder, liorder, sub_graph_range.first, sub_graph_range.second);

    if (opts.use_HODLR()) {
      for (auto& s : local_tree_->partition_tree) {
        auto sep = s.first;
        std::cout << "LOCAL SEP GRAPH!!! sep=" << sep << std::endl;
        auto sep_begin = local_tree_->sizes(sep);
        auto sep_end = local_tree_->sizes(sep+1);
        local_tree_->separator_graph[sep] =
          my_sub_graph.extract_subgraph
          (sub_graph_range.first, sep_begin, sep_end);
      }
    }

    std::unique_ptr<int[]> rcnts(new int[2*P]);
    auto displs = rcnts.get() + P;
    for (int p=0; p<P; p++) {
      rcnts[p] = sub_graph_ranges[p].second - sub_graph_ranges[p].first;
      displs[p] = sub_graph_ranges[p].first;
    }
    MPI_Allgatherv
      (lorder.data(), rcnts[comm_->rank()], mpi_type<integer_t>(),
       gorder.data(), rcnts.get(), displs, mpi_type<integer_t>(),
       comm_->comm());
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::distributed_separator_reordering
  (const SPOptions<scalar_t>& opts, std::vector<integer_t>& gorder) {
    auto rank = comm_->rank();
    auto P = comm_->size();
    auto dsep_begin = dist_sep_range.first;
    auto dsep_end = dist_sep_range.second;
    auto dim_dsep = dsep_end - dsep_begin;
    std::vector<integer_t> dorder(dim_dsep), diorder(dim_dsep);
    if (dim_dsep) {
      int min_sep = opts.compression_min_sep_size();
      int leaf = opts.compression_leaf_size();
      HSS::HSSPartitionTree tree(dim_dsep);
      if ((dim_dsep >= min_sep) && (dim_dsep > 2 * leaf)) {
        std::fill(dorder.begin(), dorder.end(), integer_t(0));
        integer_t parts = 0;
        auto l2 = my_dist_sep.length_2_edges(dist_sep_range.first);
        split_recursive_dist(opts, dorder, tree, parts, 0, 1, l2);
        for (integer_t part=0, count=dsep_begin; part<parts; part++)
          for (integer_t i=0; i<dim_dsep; i++)
            if (dorder[i] == part) dorder[i] = -count++;
        for (integer_t i=0; i<dim_dsep; i++)
          dorder[i] = -dorder[i];
      } else
        for (integer_t i=0; i<dim_dsep; i++)
          dorder[i] = i+dsep_begin;
      for (integer_t i=0; i<dim_dsep; i++)
        diorder[dorder[i]-dsep_begin] = i;
      if (opts.use_BLR() || opts.use_HODLR()) {
        auto tiles = tree.leaf_sizes();
        integer_t nt = tiles.size();
        DenseMatrix<bool> adm(nt, nt);
        adm.fill(true);
        for (integer_t t=0; t<nt; t++)
          adm(t, t) = false;
        if (opts.use_HODLR() ||
            opts.BLR_options().admissibility() ==
            BLR::Admissibility::STRONG) {
          std::vector<integer_t> ts(nt+1);
          for (integer_t i=0; i<nt; i++)
            ts[i+1] = tiles[i] + ts[i];
          auto& G = my_dist_sep;
          for (integer_t t=0; t<nt; t++) {
            for (integer_t i=ts[t]; i<ts[t+1]; i++) {
              auto Gi = diorder[i];
              auto hij = G.ind() + G.ptr(Gi+1);
              for (auto pj=G.ind()+G.ptr(Gi); pj!=hij; pj++) {
                auto Gj = *pj - dsep_begin;
                if (Gj < 0 || Gj >= dim_dsep) continue;
                integer_t tj = std::distance
                  (ts.begin(), std::upper_bound
                   (ts.begin(), ts.end(), dorder[Gj]-dsep_begin)) - 1;
                if (t != tj) adm(t, tj) = adm(tj, t) = false;
              }
            }
          }
        }
        sep_tree_->admissibility[dsep_internal] = std::move(adm);
      }
      sep_tree_->partition_tree[dsep_internal] = std::move(tree);
      // if (opts.use_HODLR())
      //   sep_tree_->separator_graph[dsep_internal] =
      //     my_dist_sep.extract_subgraph(dsep_begin, 0, dim_dsep);
    }
    std::unique_ptr<int[]> rcnts(new int[2*P]);
    auto displs = rcnts.get() + P;
    for (int p=0; p<P; p++) {
      rcnts[p] = dist_sep_ranges[p].second - dist_sep_ranges[p].first;
      displs[p] = dist_sep_ranges[p].first;
    }
    MPI_Allgatherv
      (dorder.data(), rcnts[rank], mpi_type<integer_t>(), gorder.data(),
       rcnts.get(), displs, mpi_type<integer_t>(), comm_->comm());
    my_dist_sep.permute_rows_local_cols_global
      (gorder, diorder, dsep_begin, dsep_end);

    if (opts.use_HODLR()) {
      std::cout << "DISTRIBUTED SEP GRAPH!!! dsep_internal="
                << dsep_internal << std::endl;
      sep_tree_->separator_graph[dsep_internal] =
        my_dist_sep.extract_subgraph(dsep_begin, 0, dim_dsep);
    }

    my_sub_graph.permute_columns(gorder);
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::get_local_graphs
  (const CSRMatrixMPI<scalar_t,integer_t>& A) {
    auto P = comm_->size();
    auto rank = comm_->rank();
    sub_graph_ranges.resize(P);
    dist_sep_ranges.resize(P);
    proc_dist_sep.resize(sep_tree_->separators());
    for (integer_t sep=0, p_local=0, p_dist=0;
         sep<sep_tree_->separators(); sep++) {
      if (sep_tree_->lch(sep) == -1) {
        // sep is a leaf, so it is the local graph of proces p
        if (p_local == rank) dsep_leaf = sep;
        sub_graph_ranges[p_local] =
          std::make_pair(sep_tree_->sizes(sep), sep_tree_->sizes(sep+1));
        proc_dist_sep[sep] = p_local++;
      } else {
        // sep was computed using distributed nested dissection,
        // assign it to process p_dist
        if (p_dist == rank)
          dsep_internal = sep;
        dist_sep_ranges[p_dist] =
          std::make_pair(sep_tree_->sizes(sep), sep_tree_->sizes(sep+1));
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
  (const CSRMatrixMPI<scalar_t,integer_t>& A) {
    auto P = comm_->size();
    auto rank = comm_->rank();
    auto n = A.size();
    auto sub_n = my_sub_graph.size();
    auto sub_etree =
      spsymetree(my_sub_graph.ptr(), my_sub_graph.ptr()+1,
                 my_sub_graph.ind(), sub_n, sub_graph_range.first);
    auto post = etree_postorder(sub_etree);
    std::vector<integer_t> iwork(sub_n);
    for (integer_t i=0; i<sub_n; ++i)
      iwork[post[i]] = post[sub_etree[i]];
    for (integer_t i=0; i<sub_n; ++i)
      sub_etree[i] = iwork[i];
    local_tree_ = std::unique_ptr<SeparatorTree<integer_t>>
      (new SeparatorTree<integer_t>(sub_etree));
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
    for (integer_t i=0; i<n; i++) iperm_[perm_[i]] = i;
    for (integer_t i=0; i<n; i++) perm_[gpost[i]] = iperm_[i];
    std::swap(perm_, iperm_);
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::clear_tree_data() {
    MatrixReordering<scalar_t,integer_t>::clear_tree_data();
    local_tree_ = nullptr;
    my_sub_graph = CSRGraph<integer_t>();
    my_dist_sep = CSRGraph<integer_t>();
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::nested_dissection_print
  (const SPOptions<scalar_t>& opts, integer_t nnz) const {
    if (opts.verbose()) {
      auto total_separators =
        comm_->all_reduce(local_tree_->separators(), MPI_SUM) +
        std::max(integer_t(0), sep_tree_->separators() - comm_->size());
      auto max_level = comm_->all_reduce
        (local_tree_->levels() + sep_tree_->level(dsep_leaf) - 1, MPI_MAX);
      if (comm_->is_root())
        MatrixReordering<scalar_t,integer_t>::nested_dissection_print
          (opts, nnz, max_level, total_separators, true);
    }
  }

} // end namespace strumpack

#endif // MATRIX_REORDERING_MPI_HPP
