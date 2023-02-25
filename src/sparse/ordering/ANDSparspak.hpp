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
#ifndef STRUMPACK_ORDERING_AND_SPARSPAK_HPP
#define STRUMPACK_ORDERING_AND_SPARSPAK_HPP

#include <memory>
#include <iostream>

#include "sparse/SeparatorTree.hpp"
#include "misc/Tools.hpp"


namespace strumpack {
  namespace ordering {

    template<typename integer> SeparatorTree<integer>
    gennd(integer neqns, integer* xadj, integer* adjncy, integer* perm);

#if defined(STRUMPACK_USE_CUDA)
    template<typename integer> SeparatorTree<integer>
    nd_bfs_cuda(integer neqns, integer* xadj, integer* adjncy,
                std::vector<integer>& perm,
                std::vector<integer>& iperm);
#endif

    template<typename integer_t>
    SeparatorTree<integer_t>
    and_reordering(integer_t n, const integer_t* ptr, const integer_t* ind,
                   std::vector<integer_t>& perm,
                   std::vector<integer_t>& iperm) {
      std::vector<integer_t> xadj(n+1), adjncy(ptr[n]);
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
#if defined(STRUMPACK_USE_CUDA)
      auto stree = nd_bfs_cuda(n, xadj.data(), adjncy.data(), iperm, perm);
#else
      auto stree = gennd(n, xadj.data(), adjncy.data(), iperm.data());
#endif
      for (integer_t i=0; i<n; i++)
        perm[iperm[i]] = i;
      return stree;
    }

    template<typename integer_t,typename G>
    SeparatorTree<integer_t>
    and_reordering(const G& A, std::vector<integer_t>& perm,
                   std::vector<integer_t>& iperm) {
      return and_reordering<integer_t>
        (A.size(), A.ptr(), A.ind(), perm, iperm);
    }

  } // end namespace ordering
} // end namespace strumpack

#endif // STRUMPACK_ORDERING_AND_SPARSPAK_HPP
