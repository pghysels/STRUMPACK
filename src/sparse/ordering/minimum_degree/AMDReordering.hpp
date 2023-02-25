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
#ifndef STRUMPACK_ORDERING_AMD_HPP
#define STRUMPACK_ORDERING_AMD_HPP

#include "StrumpackFortranCInterface.h"
// #include "sparse/ordering/Graph.hpp"

#if AMDIDXSIZE==64
  typedef int64_t AMDInt;
#else
  typedef int32_t AMDInt;
#endif

extern "C" {
#define AMDBAR_FC STRUMPACK_FC_GLOBAL(amdbar,AMDBAR)
  void AMDBAR_FC(AMDInt* N, AMDInt* PE, AMDInt* IW, AMDInt* LEN,
                 AMDInt* IWLEN, AMDInt* PFREE, AMDInt* NV,
                 AMDInt* NEXT, AMDInt* LAST, AMDInt* HEAD,
                 AMDInt* ELEN, AMDInt* DEGREE, AMDInt* NCMPA,
                 AMDInt* W, AMDInt* IOVFLO);
}

namespace strumpack {
  namespace ordering {

    template<typename intt> inline void
    WRAPPER_amd(intt n, intt* xadj, intt* adjncy, intt* perm, intt* iperm) {
      auto nnz = xadj[n];
      std::vector<AMDInt> ixadj(n+1), iadjncy(nnz), p(n), ip(n);
      std::copy(xadj, xadj+n+1, ixadj.data());
      std::copy(adjncy, adjncy+nnz, iadjncy.data());
      WRAPPER_amd<AMDInt>(n, ixadj.data(), iadjncy.data(), p.data(), ip.data());
      std::copy(ip.begin(), ip.end(), iperm);
      std::copy(p.begin(), p.end(), perm);
    }
    template<> inline void
    WRAPPER_amd(AMDInt n, AMDInt* xadj, AMDInt* adjncy,
                AMDInt* perm, AMDInt* iperm) {
      AMDInt iovflo = std::numeric_limits<AMDInt>::max();
      AMDInt ncmpa = 0;
      AMDInt iwsize = 4*n;
      AMDInt nnz = xadj[n]-1;
      std::unique_ptr<AMDInt[]> iwork    // iwsize
        (new AMDInt[iwsize + 4*n + n+1 + nnz + n + 1]);
      auto vtxdeg = iwork.get() + iwsize; // n
      auto qsize  = vtxdeg + n;     // n
      auto ecforw = qsize + n;      // n
      auto marker = ecforw + n;     // n
      auto nvtxs  = marker + n;     // n+1
      auto rowind = nvtxs + n+1;    // nnz + n + 1
      for (AMDInt i=0; i<n; i++)
        nvtxs[i] = xadj[i+1] - xadj[i];
      for (AMDInt i=0; i<nnz; i++)
        rowind[i] = adjncy[i];
      AMDInt pfree = xadj[n-1] + nvtxs[n-1];
      AMDInt iwlen = pfree + n;
      AMDBAR_FC(&n, xadj, rowind, nvtxs, &iwlen, &pfree, qsize, ecforw,
                perm, iwork.get(), iperm, vtxdeg, &ncmpa, marker, &iovflo);
    }

    // template<typename intt> PPt<intt> amd(const Graph<intt>& g) {
    //   auto g1 = g.get_1_based();
    //   PPt<intt> p(g1.n());
    //   WRAPPER_amd(g1.n(), g1.ptr(), g1.ind(), p.Pt(), p.P());
    //   p.to_0_based();
    //   assert(p.valid());
    //   return p;
    // }

    // template<typename intt> PPt<intt> amd(Graph<intt>&& g) {
    //   g.to_1_based();
    //   PPt<intt> p(g.n());
    //   WRAPPER_amd(g.n(), g.ptr(), g.ind(), p.Pt(), p.P());
    //   p.to_0_based();
    //   assert(p.valid());
    //   return p;
    // }

    template<typename integer_t> inline void
    WRAPPER_amd(AMDInt n, std::vector<AMDInt>& xadj,
                std::vector<AMDInt>& adjncy,
                std::vector<integer_t>& perm,
                std::vector<integer_t>& iperm) {
      std::vector<AMDInt> p(n), ip(n);
      WRAPPER_amd(n, xadj.data(), adjncy.data(), p.data(), ip.data());
      iperm.assign(ip.begin(), ip.end());
      perm.assign(p.begin(), p.end());
    }
    template<> inline void
    WRAPPER_amd(AMDInt n, std::vector<AMDInt>& xadj,
                std::vector<AMDInt>& adjncy,
                std::vector<AMDInt>& perm,
                std::vector<AMDInt>& iperm) {
      WRAPPER_amd(n, xadj.data(), adjncy.data(), perm.data(), iperm.data());
    }

    template<typename integer_t>
    SeparatorTree<integer_t>
    amd_reordering(integer_t n, const integer_t* ptr, const integer_t* ind,
                   std::vector<integer_t>& perm,
                   std::vector<integer_t>& iperm) {
      std::vector<AMDInt> xadj(n+1), adjncy(ptr[n]);
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
      WRAPPER_amd(n, xadj, adjncy, iperm, perm);
      for (integer_t i=0; i<n; i++) {
        iperm[i]--;
        perm[i]--;
      }
      return build_sep_tree_from_perm(ptr, ind, perm, iperm);
    }

    template<typename integer_t,typename G>
    SeparatorTree<integer_t>
    amd_reordering(const G& A, std::vector<integer_t>& perm,
                   std::vector<integer_t>& iperm) {
      return amd_reordering<integer_t>
        (A.size(), A.ptr(), A.ind(), perm, iperm);
    }

  } // end namespace ordering
} // end namespace strumpack

#endif
