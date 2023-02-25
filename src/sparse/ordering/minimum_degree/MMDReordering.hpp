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
#ifndef STRUMPACK_ORDERING_MMD_HPP
#define STRUMPACK_ORDERING_MMD_HPP

#include "StrumpackFortranCInterface.h"
// #include "sparse/ordering/Graph.hpp"

#if MMDIDXSIZE==64
typedef std::int64_t MMDInt;
#else
typedef std::int32_t MMDInt;
#endif

extern "C" {
#define ORDMMD_FC STRUMPACK_FC_GLOBAL(ordmmd,ORDMMD)
  void ORDMMD_FC(MMDInt* neqns, MMDInt* nadj, MMDInt* xadj,
                 MMDInt* adjncy, MMDInt* invp, MMDInt* perm,
                 MMDInt* iwsiz, MMDInt* iwork, MMDInt* nofsub,
                 MMDInt* iflag);
}

namespace strumpack {
  namespace ordering {

    template<typename intt> inline void
    WRAPPER_mmd(intt n, intt* xadj, intt* adjncy, intt* perm, intt* iperm) {
      auto nnz = xadj[n];
      std::vector<MMDInt> ixadj(n+1), iadjncy(nnz), p(n), ip(n);
      std::copy(xadj, xadj+n+1, ixadj.data());
      std::copy(adjncy, adjncy+nnz, iadjncy.data());
      WRAPPER_mmd<MMDInt>(n, ixadj.data(), iadjncy.data(), p.data(), ip.data());
      std::copy(ip.begin(), ip.end(), iperm);
      std::copy(p.begin(), p.end(), perm);
    }
    template<> inline void
    WRAPPER_mmd(MMDInt n, MMDInt* xadj, MMDInt* adjncy,
                MMDInt* perm, MMDInt* iperm) {
      MMDInt iwsize = 4*n, nadj = xadj[n]-1, nofsub = 0, iflag = 0;
      std::unique_ptr<MMDInt[]> iwork(new MMDInt[iwsize]);
      ORDMMD_FC(&n, &nadj, xadj, adjncy, iperm, perm,
                &iwsize, iwork.get(), &nofsub, &iflag);
      assert(iflag == 0);
    }

    // template<typename intt> PPt<intt> mmd(const Graph<intt>& g) {
    //   auto g1 = g.get_1_based();
    //   PPt<intt> p(g1.n());
    //   WRAPPER_mmd(g1.n(), g1.ptr(), g1.ind(), p.Pt(), p.P());
    //   p.to_0_based();
    //   assert(p.valid());
    //   return p;
    // }

    // template<typename intt> PPt<intt> mmd(Graph<intt>&& g) {
    //   g.to_1_based(); // no need to make a copy here!
    //   PPt<intt> p(g.n());
    //   WRAPPER_mmd(g.n(), g.ptr(), g.ind(), p.Pt(), p.P());
    //   p.to_0_based();
    //   assert(p.valid());
    //   return p;
    // }


    template<typename integer_t> inline void
    WRAPPER_mmd(MMDInt n,
                std::vector<MMDInt>& xadj, std::vector<MMDInt>& adjncy,
                std::vector<integer_t>& perm, std::vector<integer_t>& iperm) {
      std::vector<MMDInt> p(n), ip(n);
      WRAPPER_mmd(n, xadj.data(), adjncy.data(), p.data(), ip.data());
      iperm.assign(ip.begin(), ip.end());
      perm.assign(p.begin(), p.end());
    }
    template<> inline void
    WRAPPER_mmd(MMDInt n,
                std::vector<MMDInt>& xadj, std::vector<MMDInt>& adjncy,
                std::vector<MMDInt>& perm, std::vector<MMDInt>& iperm) {
      WRAPPER_mmd(n, xadj.data(), adjncy.data(), perm.data(), iperm.data());
    }

    template<typename integer_t>
    SeparatorTree<integer_t>
    mmd_reordering(integer_t n, const integer_t* ptr, const integer_t* ind,
                   std::vector<integer_t>& perm,
                   std::vector<integer_t>& iperm) {
      std::vector<MMDInt> xadj(n+1), adjncy(ptr[n]);
      integer_t e = 0;
      for (integer_t j=0; j<n; j++) {
        xadj[j] = e+1;
        for (integer_t t=ptr[j]; t<ptr[j+1]; t++)
          if (ind[t] != j) adjncy[e++] = ind[t]+1;
      }
      xadj[n] = e+1;
      if (e==0)
        if (mpi_root())
          std::cerr << "# WARNING: matrix seems to be diagonal!" << std::endl;
      WRAPPER_mmd(n, xadj, adjncy, iperm, perm);
      for (integer_t i=0; i<n; i++) {
        iperm[i]--;
        perm[i]--;
      }
      return build_sep_tree_from_perm(ptr, ind, perm, iperm);
    }

    template<typename integer_t,typename G>
    SeparatorTree<integer_t>
    mmd_reordering(const G& A, std::vector<integer_t>& perm,
                   std::vector<integer_t>& iperm) {
      return mmd_reordering<integer_t>
        (A.size(), A.ptr(), A.ind(), perm, iperm);
    }

  } // end namespace ordering
} // end namespace strumpack

#endif
