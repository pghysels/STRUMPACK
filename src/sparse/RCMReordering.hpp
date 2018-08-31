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
#ifndef RCM_REORDERING_HPP
#define RCM_REORDERING_HPP

#include "StrumpackFortranCInterface.h"

extern "C" {
#define GENRCM_FC FC_GLOBAL(genrcm,GENRCM)
  void GENRCM_FC
  (int* neqns, int* xadj, int* adjncy, int* perm, int* mask, int* xls);
}

namespace strumpack {

  template<typename integer_t> inline void WRAPPER_rcm
  (std::vector<int>& xadj, std::vector<int>& adjncy,
   std::vector<integer_t>& perm) {
    int n = perm.size();
    std::vector<int> mask(2*n), xls(2*n), int_perm(n);
    GENRCM_FC(&n, xadj.data(), adjncy.data(), int_perm.data(),
              mask.data(), xls.data());
    for (int i=0; i<n; i++)
      perm[i] = int_perm[i]-1;
  }
  template<> inline void WRAPPER_rcm
  (std::vector<int>& xadj, std::vector<int>& adjncy,
   std::vector<int>& perm) {
    int n = perm.size();
    std::vector<int> mask(2*n), xls(2*n);
    GENRCM_FC
      (&n, xadj.data(), adjncy.data(), perm.data(), mask.data(), xls.data());
    for (int i=0; i<n; i++) perm[i]--;
  }

  template<typename scalar_t,typename integer_t>
  std::unique_ptr<SeparatorTree<integer_t>> rcm_reordering
  (const CompressedSparseMatrix<scalar_t,integer_t>& A,
   std::vector<integer_t>& perm, std::vector<integer_t>& iperm) {
    auto n = A.size();
    auto ptr = A.ptr();
    auto ind = A.ind();
    std::vector<int> xadj(n+1), adjncy(ptr[n]);
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
    WRAPPER_rcm(xadj, adjncy, perm);
    return build_sep_tree_from_perm(ptr, ind, perm, iperm);
  }

} // end namespace strumpack

#endif
