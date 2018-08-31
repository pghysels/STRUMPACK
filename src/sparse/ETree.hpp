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
#ifndef STRUMPACK_ETREE_HPP
#define STRUMPACK_ETREE_HPP

#include <cassert>
#include <stack>

namespace strumpack {

  template<typename integer_t> inline integer_t
  make_set(integer_t i, std::vector<integer_t>& pp) {
    pp[i] = i;
    return i;
  }

  template<typename integer_t> inline integer_t
  link(integer_t s, integer_t t, std::vector<integer_t>& pp) {
    pp[s] = t;
    return t;
  }

  /** path halving */
  template<typename integer_t> inline integer_t
  find(integer_t i, std::vector<integer_t>& pp) {
    integer_t p = pp[i];
    integer_t gp = pp[p];
    while (gp != p) {
      pp[i] = gp;
      i = gp;
      p = pp[i];
      gp = pp[p];
    }
    return p;
  }

  /*! \brief Symmetric elimination tree
   *
   * <pre>
   *      p = spsymetree (A);
   *
   *      Find the elimination tree for symmetric matrix A.
   *      This uses Liu's algorithm, and runs in time O(nz*log n).
   *
   *      Input:
   *        Square sparse matrix A.  No check is made for symmetry;
   *        elements below and on the diagonal are ignored.
   *        Numeric values are ignored, so any explicit zeros are
   *        treated as nonzero.
   *      Output:
   *        Integer array of parents representing the etree, with n
   *        meaning a root of the elimination forest.
   *      Note:
   *        This routine uses only the upper triangle, while sparse
   *        Cholesky (as in spchol.c) uses only the lower.  Matlab's
   *        dense Cholesky uses only the upper.  This routine could
   *        be modified to use the lower triangle either by transposing
   *        the matrix or by traversing it by rows with auxiliary
   *        pointer and link arrays.
   *
   *      John R. Gilbert, Xerox, 10 Dec 1990
   *      Based on code by JRG dated 1987, 1988, and 1990.
   *      Modified by X.S. Li, November 1999.
   * </pre>
   */
  template<typename integer_t> std::vector<integer_t>
  spsymetree(integer_t *acolst, integer_t *acolend, // column starts and ends past 1
             integer_t *arow,                       // row indices of A
             integer_t n,                           // dimension of A
             integer_t subgraph_begin=0) {          // first row/column of subgraph
    // if working on subgraph, acolst/end only for subgraph and n is number of vertices in the subgraph
    std::vector<integer_t> root(n, 0); // root of subtee of etree
    std::vector<integer_t> pp(n, 0);
    std::vector<integer_t> parent(n);
    integer_t rset, cset, rroot, row, col, p;
    for (col=0; col<n; col++) {
      cset = make_set(col, pp);
      root[cset] = col;
      parent[col] = n;
      for (p=acolst[col]; p<acolend[col]; p++) {
        row = arow[p] - subgraph_begin;
        if (row >= col) continue;
        rset = find(row, pp);
        rroot = root[rset];
        if (rroot != col) {
          parent[rroot] = col;
          cset = link(cset, rset, pp);
          root[cset] = col;
        }
      }
    }
    return parent;
  }

  /**
   * Depth-first search from vertex n on the etree.
   * Non-recursive version.
   */
  template<typename integer_t> void etdfs_non_recursive
  (integer_t n, const integer_t* parent, const integer_t* first_kid,
   const integer_t* next_kid, std::vector<integer_t>& post,
   integer_t postnum) {
    integer_t current = n, first, next;
    while (postnum != n) {
      first = first_kid[current];     // no kid for the current node
      if (first == -1) {              // no first kid for the current node
        post[current] = postnum++;    // numbering this node because it has no kid
        next = next_kid[current];     // looking for the next kid
        while (next == -1) {
          current = parent[current];  // no more kids : back to the parent node
          post[current] = postnum++;  // numbering the parent node
          next = next_kid[current];   // get the next kid
        }
        if (postnum == n+1) return;   // stopping criterion
        current = next;               // updating current node
      } else current = first;         // updating current node
    }
  }

  /**
   * Post-order an etree.
   */
  template<typename integer_t> std::vector<integer_t>
  etree_postorder(integer_t n, const integer_t* parent) {
    auto first_kid = new integer_t[2*(n+1)];
    auto next_kid = first_kid + n+1;
    std::fill(first_kid, first_kid+n+1, integer_t(-1));
    next_kid[n] = 0;
    // set up structure describing children
    for (integer_t v = n-1; v >= 0; v--) {
      integer_t dad = parent[v];
      next_kid[v] = first_kid[dad];
      first_kid[dad] = v;
    }
    // depth-first search from dummy root vertex #n
    std::vector<integer_t> post(n+1);
    etdfs_non_recursive(n, parent, first_kid, next_kid, post, integer_t(0));
    delete[] first_kid;
    return post;
  }

  template<typename integer_t> std::vector<integer_t>
  etree_postorder(const std::vector<integer_t>& etree) {
    return etree_postorder<integer_t>(etree.size(), etree.data());
  }

} // end namespace strumpack

#endif // STRUMPACK_ETREE_HPP
