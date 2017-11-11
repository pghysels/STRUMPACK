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
#include "mumps_symqamd.hpp"

namespace strumpack {

  template<typename integer_t> inline int WRAPPER_METIS_NodeNDP
  (integer_t n, idx_t* xadj, idx_t* adjncy, idx_t* vwgt, idx_t seps,
   idx_t* options, integer_t* perm, integer_t* iperm, idx_t* sizes) {
    auto order = new idx_t[2*n];
    auto iorder = order + n;
    int ierr = METIS_NodeNDP(n, xadj, adjncy, vwgt, seps,
                             options, order, iorder, sizes);
    std::copy(order, order+n, perm);
    std::copy(iorder, iorder+n, iperm);
    delete[] order;
    return ierr;
  }
  template<> inline int
  WRAPPER_METIS_NodeNDP(idx_t n, idx_t* xadj, idx_t* adjncy, idx_t* vwgt,
                        idx_t seps, idx_t* options,
                        idx_t* perm, idx_t* iperm, idx_t* sizes) {
    return METIS_NodeNDP(n, xadj, adjncy, vwgt, seps, options,
                         perm, iperm, sizes);
  }

  template<typename integer_t> inline int WRAPPER_METIS_NodeND
  (integer_t n, idx_t* xadj, idx_t* adjncy, idx_t* vwgt,
   idx_t* options, integer_t* perm, integer_t* iperm) {
    auto order = new idx_t[2*n];
    auto iorder = order + n;
    idx_t _n(n);
    int ierr = METIS_NodeND(&_n, xadj, adjncy, vwgt, options, order, iorder);
    std::copy(order, order+n, perm);
    std::copy(iorder, iorder+n, iperm);
    delete[] order;
    return ierr;
  }
  template<> inline int
  WRAPPER_METIS_NodeND(idx_t n, idx_t* xadj, idx_t* adjncy, idx_t* vwgt,
                       idx_t* options, idx_t* perm, idx_t* iperm) {
    return METIS_NodeND(&n, xadj, adjncy, vwgt, options, perm, iperm);
  }

  // this is used in separator reordering
  template<typename integer_t> inline int WRAPPER_METIS_PartGraphRecursive
  (idx_t* nvtxs, idx_t* ncon, std::vector<integer_t>& sep_csr_ptr,
   std::vector<integer_t>& sep_csr_ind, idx_t* two,
   idx_t* edge_cut, idx_t* partitioning) {
    auto ptr = new idx_t[sep_csr_ptr.size()+sep_csr_ind.size()];
    auto ind = ptr + sep_csr_ptr.size();
    for (size_t i=0; i<sep_csr_ptr.size(); i++) ptr[i] = sep_csr_ptr[i];
    for (size_t i=0; i<sep_csr_ind.size(); i++) ind[i] = sep_csr_ind[i];
    int ierr = METIS_PartGraphRecursive
      (nvtxs, ncon, ptr, ind, NULL, NULL, NULL,
       two, NULL, NULL, NULL, edge_cut, partitioning);
    delete[] ptr;
    return ierr;
  }
  template<> inline int WRAPPER_METIS_PartGraphRecursive
  (idx_t* nvtxs, idx_t* ncon, std::vector<idx_t>& sep_csr_ptr,
   std::vector<idx_t>& sep_csr_ind, idx_t* two,
   idx_t* edge_cut, idx_t* partitioning) {
    return METIS_PartGraphRecursive
      (nvtxs, ncon, sep_csr_ptr.data(), sep_csr_ind.data(),
       NULL, NULL, NULL, two, NULL, NULL, NULL, edge_cut, partitioning);
  }


  template<typename integer_t> std::unique_ptr<SeparatorTree<integer_t>>
  sep_tree_from_metis_sizes(integer_t nodes, integer_t
                            separators, idx_t* sizes) {
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
        sep_tree->pa()[pos + s0 - 1] = root;
        sep_tree->pa()[pos + s0 + s1 - 1] = root;
      } else std::tie(s0, s1) = std::make_tuple(0, 0);
      return s0 + s1 + 1;
    };
    parent(0, 0);
    sep_tree->pa()[nodes - 1] = -1;

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
        sep_tree->lch()[root] = pos + s0 - 1;
        sep_tree->rch()[root] = pos + s0 + s1 - 1;
      } else {
        std::tie(s0, s1) = std::make_tuple(0, 0);
        sep_tree->lch()[pos] = -1;
        sep_tree->rch()[pos] = -1;
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
      sep_tree->sizes()[pos + s0 + s1] = offs + q0 + q1;
      return std::make_pair(s0 + s1 + 1, q0 + q1 + sizes[nodes - (v + 1)]);
    };
    integer_t vertices;
    std::tie(std::ignore, vertices) = offset(0, 0, 0);
    sep_tree->sizes()[nodes] = vertices;
    return sep_tree;
  }

  // TODO throw an exception
  template<typename scalar_t,typename integer_t>
  std::unique_ptr<SeparatorTree<integer_t>>
  metis_nested_dissection(CompressedSparseMatrix<scalar_t,integer_t>* A,
                          integer_t* perm, integer_t* iperm,
                          const SPOptions<scalar_t>& opts) {
    auto n = A->size();
    auto ptr = A->get_ptr();
    auto ind = A->get_ind();
    auto xadj = new idx_t[n+1 + ptr[n]];
    auto adjncy = xadj + n+1;
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
      auto sizes = new idx_t[nodes + 1];
      ierr = WRAPPER_METIS_NodeNDP(n, xadj, adjncy, NULL, separators + 1,
                                   NULL, iperm, perm, sizes);
      delete[] xadj;
      if (opts.use_MUMPS_SYMQAMD())
        sep_tree = aggressive_amalgamation(A, perm, iperm, opts);
      else sep_tree = sep_tree_from_metis_sizes(nodes, separators, sizes);
      delete[] sizes;
    } else {
      ierr = WRAPPER_METIS_NodeND(n, xadj, adjncy, NULL, NULL, iperm, perm);
      delete[] xadj;
      if (opts.use_MUMPS_SYMQAMD())
        sep_tree = aggressive_amalgamation(A, perm, iperm, opts);
      else sep_tree = build_sep_tree_from_perm(n, ptr, ind, perm, iperm);
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
