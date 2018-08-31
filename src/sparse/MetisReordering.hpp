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
#ifndef METIS_REORDERING_HPP
#define METIS_REORDERING_HPP

#include <functional>
#include <typeinfo>
#include <memory>
#include <metis.h>
#include "SeparatorTree.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "mumps_symqamd.hpp"
#endif

namespace strumpack {

  template<typename integer_t> inline int WRAPPER_METIS_NodeNDP
  (std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy,
   idx_t* vwgt, idx_t seps, idx_t* options, std::vector<integer_t>& perm,
   std::vector<integer_t>& iperm, std::vector<idx_t>& sizes) {
    idx_t n = perm.size();
    std::vector<idx_t> order(n);
    std::vector<idx_t> iorder(n);
    int ierr = METIS_NodeNDP
      (n, xadj.data(), adjncy.data(), vwgt, seps, options,
       order.data(), iorder.data(), sizes.data());
    perm.assign(order.begin(), order.end());
    iperm.assign(iorder.begin(), iorder.end());
    return ierr;
  }
  template<> inline int WRAPPER_METIS_NodeNDP
  (std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy,
   idx_t* vwgt, idx_t seps, idx_t* options, std::vector<idx_t>& perm,
   std::vector<idx_t>& iperm, std::vector<idx_t>& sizes) {
    idx_t n = perm.size();
    return METIS_NodeNDP
      (n, xadj.data(), adjncy.data(), vwgt, seps, options,
       perm.data(), iperm.data(), sizes.data());
  }

  template<typename integer_t> inline int WRAPPER_METIS_NodeND
  (std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy,
   idx_t* vwgt, idx_t* options, std::vector<integer_t>& perm,
   std::vector<integer_t>& iperm) {
    idx_t n = perm.size();
    std::vector<idx_t> order(n);
    std::vector<idx_t> iorder(n);
    int ierr = METIS_NodeND
      (&n, xadj.data(), adjncy.data(), vwgt, options,
       order.data(), iorder.data());
    perm.assign(order.begin(), order.end());
    iperm.assign(iorder.begin(), iorder.end());
    return ierr;
  }
  template<> inline int WRAPPER_METIS_NodeND
  (std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy,
   idx_t* vwgt, idx_t* options, std::vector<idx_t>& perm,
   std::vector<idx_t>& iperm) {
    idx_t n = perm.size();
    return METIS_NodeND
      (&n, xadj.data(), adjncy.data(), vwgt, options,
       perm.data(), iperm.data());
  }

  // this is used in separator reordering
  template<typename integer_t> inline int WRAPPER_METIS_PartGraphRecursive
  (idx_t nvtxs, idx_t ncon, std::vector<integer_t>& csr_ptr,
   std::vector<integer_t>& csr_ind, idx_t nparts,
   idx_t& edge_cut, std::vector<idx_t>& partitioning) {
    std::vector<idx_t> ptr(csr_ptr.size());
    std::vector<idx_t> ind(csr_ind.size());
    ptr.assign(csr_ptr.begin(), csr_ptr.end());
    ind.assign(csr_ind.begin(), csr_ind.end());
    int ierr = METIS_PartGraphRecursive
      (&nvtxs, &ncon, ptr.data(), ind.data(), NULL, NULL, NULL,
       &nparts, NULL, NULL, NULL, &edge_cut, partitioning.data());
    return ierr;
  }
  template<> inline int WRAPPER_METIS_PartGraphRecursive
  (idx_t nvtxs, idx_t ncon, std::vector<idx_t>& csr_ptr,
   std::vector<idx_t>& csr_ind, idx_t nparts,
   idx_t& edge_cut, std::vector<idx_t>& partitioning) {
    return METIS_PartGraphRecursive
      (&nvtxs, &ncon, csr_ptr.data(), csr_ind.data(),
       NULL, NULL, NULL, &nparts, NULL, NULL, NULL,
       &edge_cut, partitioning.data());
  }


  template<typename integer_t> std::unique_ptr<SeparatorTree<integer_t>>
  sep_tree_from_metis_sizes
  (integer_t nodes, integer_t separators, std::vector<idx_t>& sizes) {
    std::unique_ptr<SeparatorTree<integer_t>>
      sep_tree(new SeparatorTree<integer_t>(nodes));

    // Generates the parent positions of all the nodes in a
    // subtree, except for its root, and returns the number
    // of nodes in the subtree. The subtree is traversed in
    // left-to-right postorder and the parent positions are
    // written into parent starting from position pos.
    // The root of the subtree is specified by v, its
    // position in a top-down level-by-level right-to-left
    // traversal of the tree.
    std::function<integer_t(integer_t,integer_t)>
      parent = [&](integer_t pos, integer_t v) -> integer_t {
      integer_t s0, s1;
      if (v < separators) {
        integer_t root;
        s0 = parent(pos, 2 * v + 2);
        s1 = parent(pos + s0, 2 * v + 1);
        root = pos + s0 + s1;
        sep_tree->pa(pos + s0 - 1) = root;
        sep_tree->pa(pos + s0 + s1 - 1) = root;
      } else std::tie(s0, s1) = std::make_tuple(0, 0);
      return s0 + s1 + 1;
    };
    parent(0, 0);
    sep_tree->pa(nodes - 1) = -1;

    // Generates the children positions of all the nodes in
    // a subtree and returns the number of nodes in that
    // subtree. Leaves are assigned position -1.
    std::function<integer_t(integer_t,integer_t)>
      children = [&](integer_t pos, integer_t v) -> integer_t {
      integer_t s0, s1;
      if (v < separators) {
        integer_t root;
        s0 = children(pos, 2 * v + 2);
        s1 = children(pos + s0, 2 * v + 1);
        root = pos + s0 + s1;
        sep_tree->lch(root) = pos + s0 - 1;
        sep_tree->rch(root) = pos + s0 + s1 - 1;
      } else {
        std::tie(s0, s1) = std::make_tuple(0, 0);
        sep_tree->lch(pos) = -1;
        sep_tree->rch(pos) = -1;
      }
      return s0 + s1 + 1;
    };
    children(0, 0);

    // Generates the offsets of the nodes of a subtree in
    // the graph ordering and returns the numbers of nodes
    // and vertices in the subtree. The argument offs
    // indicates the offset of the subtree in the graph
    // ordering.
    using ss = std::pair<integer_t,integer_t>;
    std::function<ss(integer_t,integer_t,integer_t)>
      offset = [&](integer_t pos, integer_t offs, integer_t v) -> ss {
      integer_t s0, s1, q0, q1;
      if (v < separators) {
        std::tie(s0, q0) = offset(pos, offs, 2 * v + 2);
        std::tie(s1, q1) = offset(pos + s0, offs + q0, 2 * v + 1);
      } else {
        std::tie(s0, s1) = std::make_pair(0, 0);
        std::tie(q0, q1) = std::make_pair(0, 0);
      }
      sep_tree->sizes(pos + s0 + s1) = offs + q0 + q1;
      return std::make_pair(s0 + s1 + 1, q0 + q1 + sizes[nodes - (v + 1)]);
    };
    integer_t vertices;
    std::tie(std::ignore, vertices) = offset(0, 0, 0);
    sep_tree->sizes(nodes) = vertices;
    return sep_tree;
  }

  // TODO throw an exception
  template<typename scalar_t,typename integer_t>
  std::unique_ptr<SeparatorTree<integer_t>>
  metis_nested_dissection
  (const CompressedSparseMatrix<scalar_t,integer_t>& A,
   std::vector<integer_t>& perm, std::vector<integer_t>& iperm,
   const SPOptions<scalar_t>& opts) {
    auto n = A.size();
    auto ptr = A.ptr();
    auto ind = A.ind();
    std::vector<idx_t> xadj(n+1), adjncy(ptr[n]);
    integer_t e = 0;
    for (integer_t j=0; j<n; j++) {
      xadj[j] = e;
      for (integer_t t=ptr[j]; t<ptr[j+1]; t++)
        if (ind[t] != j) adjncy[e++] = ind[t];
    }
    xadj[n] = e;
    if (e==0)
      if (mpi_root())
        std::cerr << "# WARNING: matrix seems to be diagonal!" << std::endl;
    int ierr;
    std::unique_ptr<SeparatorTree<integer_t>> sep_tree;
    if (opts.use_METIS_NodeNDP()) {
      integer_t nodes =
        std::max(integer_t(3), ( n / opts.nd_param() ) / 2 * 2 + 1);
      integer_t separators = nodes / 2;
      std::vector<idx_t> sizes(nodes + 1);
      ierr = WRAPPER_METIS_NodeNDP
        (xadj, adjncy, NULL, separators + 1, NULL, iperm, perm, sizes);
#if defined(STRUMPACK_USE_MPI)
      if (opts.use_MUMPS_SYMQAMD())
        sep_tree = aggressive_amalgamation(A, perm, iperm, opts);
      else
        sep_tree = sep_tree_from_metis_sizes(nodes, separators, sizes);
#else
      sep_tree = sep_tree_from_metis_sizes(nodes, separators, sizes);
#endif
    } else {
      ierr = WRAPPER_METIS_NodeND(xadj, adjncy, NULL, NULL, iperm, perm);
#if defined(STRUMPACK_USE_MPI)
      if (opts.use_MUMPS_SYMQAMD())
        sep_tree = aggressive_amalgamation(A, perm, iperm, opts);
      else
        sep_tree = build_sep_tree_from_perm(ptr, ind, perm, iperm);
#else
      sep_tree = build_sep_tree_from_perm(ptr, ind, perm, iperm);
#endif
    }
    if (ierr != METIS_OK) {
      std::cerr << "# ERROR: Metis nested dissection reordering failed"
                << " with error code " << ierr << std::endl;
      // TODO throw an exception
    }
    return sep_tree;
  }

} // end namespace strumpack

#endif
