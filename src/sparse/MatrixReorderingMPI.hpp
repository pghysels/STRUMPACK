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
#include "PTScotchReordering.hpp"
#include "ParMetisReordering.hpp"
#include "GeometricReorderingMPI.hpp"

namespace strumpack {

  /**
   * A MatrixReorderingMPI has a distributed separator tree. This tree
   * is stored using 2 trees, one in the base class MatrixReordering
   * and one here in MatrixReorderingMPI. The tree in MatrixReordering
   * (sep_tree) is the top of the tree, corresponding to the
   * distributed separators. The tree stored here (local_sep_tree)
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
  class MatrixReorderingMPI : public MatrixReordering<scalar_t,integer_t> {
  public:
    MatrixReorderingMPI(integer_t n, MPI_Comm c);
    virtual ~MatrixReorderingMPI();

    int nested_dissection(const SPOptions<scalar_t>& opts,
                          CSRMatrixMPI<scalar_t,integer_t>* A,
                          int nx, int ny, int nz);

    void separator_reordering(const SPOptions<scalar_t>& opts,
                              CSRMatrixMPI<scalar_t,integer_t>* A,
                              FrontalMatrix<scalar_t,integer_t>* F);
    void separator_reordering(const SPOptions<scalar_t>& opts,
                              CSRMatrixMPI<scalar_t,integer_t>* A,
                              bool verbose);
    void clear_tree_data();

    std::unique_ptr<SeparatorTree<integer_t>> local_sep_tree;

    /**
     * proc_dist_sep[sep] holds the rank of the process responsible
     * for distributed separator sep
     * - if sep is a leaf of the distributed tree, proc_dist_sep
     *    points to the process holding that subgraph as my_sub_graph
     * - if sep is a non-leaf, proc_dist_sep points to the process
     *    holding the graph of the distributed separator sep as
     *    my_dist_sep
     */
    integer_t* proc_dist_sep = nullptr;
    /**
     * Every process is responsible for one local subgraph of A.  The
     * distributed nested dissection will create a separator tree with
     * exactly P leafs.  Each process takes one of those leafs and
     * stores the corresponding part of the permuted matrix A in
     * sub_graph_A.
     */
    CSRGraph<integer_t>* my_sub_graph = nullptr;
    /**
     * Every process is responsible for one separator from the
     * distributed part of nested dissection.  The graph of the
     * distributed separator for which this rank is responsible is
     * stored as dist_sep_A.
     */
    CSRGraph<integer_t>* my_dist_sep = nullptr;

    std::pair<integer_t,integer_t>* sub_graph_ranges = nullptr;
    std::pair<integer_t,integer_t>* dist_sep_ranges = nullptr;

    std::pair<integer_t,integer_t> sub_graph_range;
    std::pair<integer_t,integer_t> dist_sep_range;

    /** Number of the node in sep_tree corresponding to the
     *  distributed separator owned by this processor. */
    integer_t dsep_internal;
    /** Number of the node in sep_tree corresponding to the
     *  root of the local subtree. */
    integer_t dsep_leaf;

  private:
    MPI_Comm _comm;
    void get_local_graphs(CSRMatrixMPI<scalar_t,integer_t>* A_mpi);
    void build_local_sep_tree(CSRMatrixMPI<scalar_t,integer_t>* A_mpi);
    void allgather_local_order(CSRMatrixMPI<scalar_t,integer_t>* A_mpi,
                               integer_t* local_order);
    void distributed_separator_reordering(const SPOptions<scalar_t>& opts,
                                          integer_t* global_sep_order);
    void local_separator_reordering(const SPOptions<scalar_t>& opts,
                                    integer_t* global_sep_order);
    void nested_dissection_print(const SPOptions<scalar_t>& opts,
                                 integer_t n, integer_t nnz);
  };

  template<typename scalar_t,typename integer_t>
  MatrixReorderingMPI<scalar_t,integer_t>::MatrixReorderingMPI
  (integer_t n, MPI_Comm c)
    : MatrixReordering<scalar_t,integer_t>(n), dsep_internal(0),
    dsep_leaf(0), _comm(c) {
  }

  template<typename scalar_t,typename integer_t>
  MatrixReorderingMPI<scalar_t,integer_t>::~MatrixReorderingMPI() {
    delete my_sub_graph;
    delete my_dist_sep;
    delete sub_graph_ranges;
    delete[] proc_dist_sep;
  }

  template<typename scalar_t,typename integer_t> int
  MatrixReorderingMPI<scalar_t,integer_t>::nested_dissection
  (const SPOptions<scalar_t>& opts, CSRMatrixMPI<scalar_t,integer_t>* A,
   int nx, int ny, int nz) {
    if (!is_parallel(opts.reordering_method())) {
      auto rank = mpi_rank(_comm);
      auto P = mpi_nprocs(_comm);
      // TODO only gather the graph? without the values or the diagonal
      auto Aseq = A->gather();
      std::unique_ptr<SeparatorTree<integer_t>> global_sep_tree;
      if (Aseq) { // only root
        switch (opts.reordering_method()) {
        case ReorderingStrategy::NATURAL: {
          for (integer_t i=0; i<A->size(); i++) this->perm[i] = i;
          global_sep_tree = build_sep_tree_from_perm
            (Aseq->size(), Aseq->get_ptr(), Aseq->get_ind(),
             this->perm, this->iperm);
          break;
        }
        case ReorderingStrategy::METIS: {
          global_sep_tree = metis_nested_dissection
            (Aseq.get(), this->perm, this->iperm, opts);
          break;
        }
        case ReorderingStrategy::SCOTCH: {
          global_sep_tree = scotch_nested_dissection
            (Aseq.get(), this->perm, this->iperm, opts);
          break;
        }
        case ReorderingStrategy::RCM: {
          global_sep_tree = rcm_reordering
            (Aseq.get(), this->perm, this->iperm);
          break;
        }
        default: assert(true);
        }
        Aseq.reset();
        global_sep_tree->check();
      }
      MPI_Bcast(this->perm, 2*this->n, mpi_type<integer_t>(), 0, _comm);
      integer_t nbsep;
      if (!rank) nbsep = global_sep_tree->separators();
      MPI_Bcast(&nbsep, 1, mpi_type<integer_t>(), 0, _comm);
      if (rank)
        global_sep_tree = std::unique_ptr<SeparatorTree<integer_t>>
          (new SeparatorTree<integer_t>(nbsep));
      global_sep_tree->broadcast(_comm);
      local_sep_tree = global_sep_tree->subtree(rank, P);
      this->sep_tree = global_sep_tree->toptree(P);
      local_sep_tree->check();
      this->sep_tree->check();
      for (integer_t i=0; i<this->n; i++) this->iperm[this->perm[i]] = i;
      get_local_graphs(A);
    } else {
      switch (opts.reordering_method()) {
      case ReorderingStrategy::GEOMETRIC: {
        if (nx*ny*nz != A->size()) {
          std::cerr << "# ERROR: Geometric reordering failed. \n"
            "# Geometric reordering only works"
            "on a simple 3 point wide stencil\n"
            "# on a regular grid and you need to provide the mesh sizes."
                    << std::endl;
          return 1;
        }
        std::tie(this->sep_tree, local_sep_tree) =
          geometric_nested_dissection_dist
          (nx, ny, nz, A->begin_row(), A->end_row(), _comm,
           this->perm, this->iperm, opts.nd_param());
        break;
      }
      case ReorderingStrategy::PARMETIS: {
        this->sep_tree = parmetis_nested_dissection
          (A, _comm, true, this->perm, opts);
        break;
      }
      case ReorderingStrategy::PTSCOTCH: {
        this->sep_tree = ptscotch_nested_dissection
          (A, _comm, true, this->perm, opts);
        break;
      }
      default: assert(true);
      }
      this->sep_tree->check();
      get_local_graphs(A);
      if (opts.reordering_method() == ReorderingStrategy::PARMETIS ||
          opts.reordering_method() == ReorderingStrategy::PTSCOTCH)
        build_local_sep_tree(A);
      local_sep_tree->check();
    }
    nested_dissection_print(opts, this->n, A->nnz());
    return 0;
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::separator_reordering
  (const SPOptions<scalar_t>& opts, CSRMatrixMPI<scalar_t,integer_t>* A,
   bool verbose) {
    auto n = A->size();
    auto global_sep_order = new integer_t[n];
    for (integer_t i=0; i<n; i++) global_sep_order[i] = i;
    distributed_separator_reordering(opts, global_sep_order);
    local_separator_reordering(opts, global_sep_order);
    for (integer_t i=0; i<n; i++)
      this->perm[global_sep_order[i]] = this->iperm[i];
    for (integer_t i=0; i<n; i++)
      this->iperm[this->perm[i]] = i;
    for (integer_t i=0; i<n; i++)
      std::swap(this->perm[i], this->iperm[i]);
    delete[] global_sep_order;
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::separator_reordering
  (const SPOptions<scalar_t>& opts, CSRMatrixMPI<scalar_t,integer_t>* A,
   FrontalMatrix<scalar_t,integer_t>* F) {
    auto N = A->size();
    auto sorder = new integer_t[N];
    std::fill(sorder, sorder+N, integer_t(0));

#pragma omp parallel
#pragma omp single
    F->bisection_partitioning(opts, sorder);

    std::cout << "TODO MatrixReorderingMPI::separator_reordering"
              << std::endl;

//     auto iwork = this->iperm;
//     for (integer_t i=0; i<N; i++) sorder[i] = -sorder[i];
//     for (integer_t i=0; i<N; i++) iwork[sorder[i]] = i;
//     A->permute(iwork, sorder);

//     // product of perm and sep_order
//     for (integer_t i=0; i<N; i++) iwork[i] = sorder[this->perm[i]];
//     for (integer_t i=0; i<N; i++) this->perm[i] = iwork[i];
//     for (integer_t i=0; i<N; i++) this->iperm[this->perm[i]] = i;

// #pragma omp parallel
// #pragma omp single
//     F->permute_upd_indices(sorder);

    delete[] sorder;
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::local_separator_reordering
  (const SPOptions<scalar_t>& opts, integer_t* global_sep_order) {
    auto rank = mpi_rank(_comm);
    auto P = mpi_nprocs(_comm);
    auto n_local = my_sub_graph->size();
    auto local_sep_order = new integer_t[2*n_local];
    auto local_sep_iorder = local_sep_order + n_local;
    for (integer_t i=0; i<n_local; i++)
      local_sep_order[i] = i + sub_graph_range.first;
    std::vector<integer_t> sep_csr_ptr, sep_csr_ind;

    std::function<void(integer_t)> reorder_separator = [&](integer_t sep) {
      auto sep_begin = local_sep_tree->sizes()[sep];
      auto sep_end = local_sep_tree->sizes()[sep+1];
      auto dim_sep = sep_end - sep_begin;
      // this front (and its descendants) are not HSS
      if (dim_sep < opts.HSS_min_sep_size()) return;
      HSS::HSSPartitionTree tree(dim_sep);
      if (dim_sep > 2 * opts.HSS_options().leaf_size()) {
        std::function<void(HSS::HSSPartitionTree&, integer_t&,
                           integer_t, integer_t)> split =
          [&](HSS::HSSPartitionTree& hss_tree, integer_t& nr_parts,
              integer_t part, integer_t count) {
          my_sub_graph->extract_separator_subgraph
          (opts.separator_ordering_level(), sub_graph_range.first,
           sep_begin, sep_end, part, local_sep_order+sep_begin,
           sep_csr_ptr, sep_csr_ind);
          idx_t ncon = 1, edge_cut = 0, two = 2, nvtxs = sep_csr_ptr.size()-1;
          auto partitioning = new idx_t[nvtxs];
          int info = WRAPPER_METIS_PartGraphRecursive
          (&nvtxs, &ncon, sep_csr_ptr, sep_csr_ind,
           &two, &edge_cut, partitioning);
          if (info !=  METIS_OK) {
            std::cerr << "METIS_PartGraphRecursive for separator"
              " reordering returned: " << info << std::endl;
            exit(1);
          }
          hss_tree.c.resize(2);
          for (integer_t i=sep_begin, j=0; i<sep_end; i++)
            if (local_sep_order[i] == part) {
              auto p = partitioning[j++];
              local_sep_order[i] = -count - p;
              hss_tree.c[p].size++;
            }
          for (integer_t p=0; p<2; p++)
            if (hss_tree.c[p].size > 2 * opts.HSS_options().leaf_size())
              split(hss_tree.c[p], nr_parts, -count-p, count+2);
            else
              std::replace(local_sep_order+sep_begin, local_sep_order+sep_end,
                           -count-p, nr_parts++);
          delete[] partitioning;
        };
        std::fill(local_sep_order+sep_begin,
                  local_sep_order+sep_end, integer_t(0));
        integer_t parts = 0;
        split(tree, parts, 0, 1);
        for (integer_t part=0, count=sep_begin+sub_graph_range.first;
             part<parts; part++)
          for (integer_t i=sep_begin; i<sep_end; i++)
            if (local_sep_order[i] == part)
              local_sep_order[i] = -count++;
        for (integer_t i=sep_begin; i<sep_end; i++)
          local_sep_order[i] = -local_sep_order[i];
      }
      local_sep_tree->HSS_trees()[sep] = tree;
      if (local_sep_tree->lch()[sep] != -1)
        reorder_separator(local_sep_tree->lch()[sep]);
      if (local_sep_tree->rch()[sep] != -1)
        reorder_separator(local_sep_tree->rch()[sep]);
    };
    if (local_sep_tree->separators() > 0)
      reorder_separator(local_sep_tree->root());
    my_sub_graph->clear_temp_data();
    for (integer_t i=0; i<n_local; i++)
      local_sep_iorder[local_sep_order[i]-sub_graph_range.first] = i;
    my_sub_graph->permute_local
      (local_sep_order, local_sep_iorder,
       sub_graph_range.first, sub_graph_range.second);
    auto rcnts = new int[2*P];
    auto displs = rcnts + P;
    for (int p=0; p<P; p++) {
      rcnts[p] = sub_graph_ranges[p].second - sub_graph_ranges[p].first;
      displs[p] = sub_graph_ranges[p].first;
    }
    MPI_Allgatherv(local_sep_order, rcnts[rank], mpi_type<integer_t>(),
                   global_sep_order, rcnts, displs,
                   mpi_type<integer_t>(), _comm);
    delete[] rcnts;
    delete[] local_sep_order;
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::distributed_separator_reordering
  (const SPOptions<scalar_t>& opts, integer_t* global_sep_order) {
    auto rank = mpi_rank(_comm);
    auto P = mpi_nprocs(_comm);
    auto dsep_begin = dist_sep_range.first;
    auto dsep_end = dist_sep_range.second;
    auto dim_dsep = dsep_end - dsep_begin;
    auto dsep_order = new integer_t[2*dim_dsep];
    auto dsep_iorder = dsep_order + dim_dsep;
    if (dim_dsep) {
      HSS::HSSPartitionTree tree(dim_dsep);
      if ((dim_dsep >= opts.HSS_min_front_size()) &&
          (dim_dsep > 2 * opts.HSS_options().leaf_size())) {
        std::vector<integer_t> sep_csr_ptr, sep_csr_ind;
        std::function<void(HSS::HSSPartitionTree&, integer_t&,
                           integer_t, integer_t)> split =
          [&](HSS::HSSPartitionTree& hss_tree, integer_t& nr_parts,
              integer_t part, integer_t count) {
          my_dist_sep->extract_separator_subgraph
          (opts.separator_ordering_level(), dsep_begin, 0, dim_dsep,
           part, dsep_order, sep_csr_ptr, sep_csr_ind);
          idx_t ncon = 1, edge_cut = 0, two = 2, nvtxs = sep_csr_ptr.size()-1;
          auto partitioning = new idx_t[dim_dsep];
          int info = WRAPPER_METIS_PartGraphRecursive
          (&nvtxs, &ncon, sep_csr_ptr, sep_csr_ind,
           &two, &edge_cut, partitioning);
          if (info !=  METIS_OK) {
            std::cerr << "METIS_PartGraphRecursive for separator"
              " reordering returned: " << info << std::endl;
            exit(1);
          }
          hss_tree.c.resize(2);
          for (integer_t i=0, j=0; i<dim_dsep; i++)
            if (dsep_order[i] == part) {
              auto p = partitioning[j++];
              dsep_order[i] = -count - p;
              hss_tree.c[p].size++;
            }
          for (integer_t p=0; p<2; p++)
            if (hss_tree.c[p].size > 2 * opts.HSS_options().leaf_size())
              split(hss_tree.c[p], nr_parts, -count-p, count+2);
            else
              std::replace(dsep_order, dsep_order+dim_dsep,
                           -count-p, nr_parts++);
          delete[] partitioning;
        };
        std::fill(dsep_order, dsep_order+dim_dsep, integer_t(0));
        integer_t parts = 0;
        split(tree, parts, 0, 1);
        my_dist_sep->clear_temp_data();
        for (integer_t part=0, count=dsep_begin; part<parts; part++)
          for (integer_t i=0; i<dim_dsep; i++)
            if (dsep_order[i] == part)
              dsep_order[i] = -count++;
        for (integer_t i=0; i<dim_dsep; i++)
          dsep_order[i] = -dsep_order[i];
      } else
        for (integer_t i=0; i<dim_dsep; i++)
          dsep_order[i] = i+dsep_begin;
      this->sep_tree->HSS_trees()[dsep_internal] = tree;
      for (integer_t i=0; i<dim_dsep; i++)
        dsep_iorder[dsep_order[i]-dsep_begin] = i;
    }
    auto rcnts = new int[2*P];
    auto displs = rcnts + P;
    for (int p=0; p<P; p++) {
      rcnts[p] = dist_sep_ranges[p].second - dist_sep_ranges[p].first;
      displs[p] = dist_sep_ranges[p].first;
    }
    MPI_Allgatherv(dsep_order, rcnts[rank], mpi_type<integer_t>(),
                   global_sep_order, rcnts, displs,
                   mpi_type<integer_t>(), _comm);
    delete[] rcnts;
    my_dist_sep->permute_rows_local_cols_global
      (global_sep_order, dsep_iorder, dsep_begin, dsep_end);
    delete[] dsep_order;
    my_sub_graph->permute_columns(global_sep_order);
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::get_local_graphs
  (CSRMatrixMPI<scalar_t,integer_t>* A) {
    auto P = mpi_nprocs(_comm);
    auto rank = mpi_rank(_comm);
    sub_graph_ranges = new std::pair<integer_t,integer_t>[2*P];
    dist_sep_ranges = sub_graph_ranges + P;
    proc_dist_sep = new integer_t[this->sep_tree->separators()];
    for (integer_t sep=0, p_local=0, p_dist=0;
         sep<this->sep_tree->separators(); sep++) {
      if (this->sep_tree->lch()[sep] == -1) {
        // sep is a leaf, so it is the local graph of proces p
        if (p_local==rank) dsep_leaf = sep;
        sub_graph_ranges[p_local] =
          std::make_pair(this->sep_tree->sizes()[sep],
                         this->sep_tree->sizes()[sep+1]);
        proc_dist_sep[sep] = p_local++;
      } else {
        // sep was computed using distributed nested dissection,
        // assign it to process p_dist
        if (p_dist==rank)
          dsep_internal = sep;
        dist_sep_ranges[p_dist] =
          std::make_pair(this->sep_tree->sizes()[sep],
                         this->sep_tree->sizes()[sep+1]);
        proc_dist_sep[sep] = p_dist++;
      }
    }
    sub_graph_range = sub_graph_ranges[rank];
    dist_sep_range = dist_sep_ranges[rank];
    my_sub_graph = A->get_sub_graph(this->perm, sub_graph_ranges);
    my_dist_sep = A->get_sub_graph(this->perm, dist_sep_ranges);
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::build_local_sep_tree
  (CSRMatrixMPI<scalar_t,integer_t>* A) {
    auto P = mpi_nprocs(_comm);
    auto rank = mpi_rank(_comm);
    auto n = A->size();
    auto sub_n = my_sub_graph->size();
    auto sub_graph_etree =
      spsymetree(my_sub_graph->get_ptr(), my_sub_graph->get_ptr()+1,
                 my_sub_graph->get_ind(),
                 sub_n, sub_graph_range.first);
    auto post_order = etree_postorder(sub_graph_etree);
    auto iwork = new integer_t[sub_n];
    for (integer_t i=0; i<sub_n; ++i)
      iwork[post_order[i]] = post_order[sub_graph_etree[i]];
    for (integer_t i=0; i<sub_n; ++i)
      sub_graph_etree[i] = iwork[i];
    local_sep_tree = std::unique_ptr<SeparatorTree<integer_t>>
      (new SeparatorTree<integer_t>(sub_graph_etree));
    for (integer_t i=0; i<sub_n; i++) {
      iwork[post_order[i]] = i;
      post_order[i] += sub_graph_range.first;
    }
    my_sub_graph->permute_local(post_order.data(), iwork,
                                sub_graph_range.first, sub_graph_range.second);
    delete[] iwork;

    auto global_post_order = new integer_t[n];
    for (integer_t i=0; i<n; i++) global_post_order[i] = i;
    auto rcnts = new int[2*P];
    auto displs = rcnts + P;
    for (int p=0; p<P; p++) {
      rcnts[p] = sub_graph_ranges[p].second - sub_graph_ranges[p].first;
      displs[p] = sub_graph_ranges[p].first;
    }
    MPI_Allgatherv(post_order.data(), rcnts[rank], mpi_type<integer_t>(),
                   global_post_order, rcnts, displs,
                   mpi_type<integer_t>(), _comm);
    delete[] rcnts;
    for (integer_t i=0; i<n; i++)
      this->iperm[this->perm[i]] = i;
    for (integer_t i=0; i<n; i++)
      this->perm[global_post_order[i]] = this->iperm[i];
    delete[] global_post_order;
    for (integer_t i=0; i<n; i++) this->iperm[this->perm[i]] = i;
    for (integer_t i=0; i<n; i++) std::swap(this->perm[i], this->iperm[i]);
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::clear_tree_data() {
    MatrixReordering<scalar_t,integer_t>::clear_tree_data();
    local_sep_tree.reset(nullptr);
    delete[] sub_graph_ranges;
    sub_graph_ranges = nullptr;
    dist_sep_ranges = nullptr;
    delete my_sub_graph; my_sub_graph = nullptr;
    delete my_dist_sep;  my_dist_sep = nullptr;
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReorderingMPI<scalar_t,integer_t>::nested_dissection_print
  (const SPOptions<scalar_t>& opts, integer_t n, integer_t nnz) {
    if (opts.verbose()) {
      auto total_separators = local_sep_tree->separators();
      MPI_Allreduce(MPI_IN_PLACE, &total_separators, 1,
                    mpi_type<integer_t>(), MPI_SUM, _comm);
      total_separators +=
        std::max(integer_t(0),
                 this->sep_tree->separators() - mpi_nprocs(_comm));
      auto max_level = local_sep_tree->levels() +
        this->sep_tree->level(dsep_leaf) - 1;
      MPI_Allreduce(MPI_IN_PLACE, &max_level, 1,
                    mpi_type<integer_t>(), MPI_MAX, _comm);
      if (!mpi_rank(_comm)) {
        std::cout << "# initial matrix:" << std::endl;
        std::cout << "#   - number of unknowns = "
                  << number_format_with_commas(n) << std::endl;
        std::cout << "#   - number of nonzeros = "
                  << number_format_with_commas(nnz) << std::endl;
        std::cout << "# nested dissection reordering:" << std::endl;
        std::cout << "#   - " << get_name(opts.reordering_method())
                  << " reordering" << std::endl;
        if (opts.reordering_method() == ReorderingStrategy::METIS) {
          if (opts.use_METIS_NodeNDP()) {
            std::cout << "#      - used METIS_NodeNDP (iso METIS_NodeND)"
                      << std::endl;
            if (opts.use_MUMPS_SYMQAMD())
              std::cout
                << "#      - supernodal tree was built using MUMPS_SYMQAMD "
                << (opts.use_agg_amalg() ? "with" : "without")
                << " aggressive amalgamation" << std::endl;
            else
              std::cout
                << "#      - supernodal tree from METIS_NodeNDP is used"
                << std::endl;
          } else {
            std::cout << "#      - used METIS_NodeND (iso METIS_NodeNDP)"
                      << std::endl;
            if (opts.use_MUMPS_SYMQAMD())
              std::cout
                << "#      - supernodal tree was built using MUMPS_SYMQAMD "
                << (opts.use_agg_amalg() ? "with" : "without")
                << " aggressive amalgamation" << std::endl;
            else
              std::cout << "#      - supernodal tree was built from etree"
                        << std::endl;
          }
        }
        std::cout << "#   - strategy parameter = "
                  << opts.nd_param() << std::endl;
        std::cout << "#   - number of separators = "
                  << number_format_with_commas(total_separators) << std::endl;
        std::cout << "#   - number of levels = " << max_level << std::endl;
      }
    }
  }

} // end namespace strumpack

#endif // MATRIX_REORDERING_MPI_HPP
