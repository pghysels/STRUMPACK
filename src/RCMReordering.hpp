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

#include "strumpack_config.h"

extern "C" {
#define GENRCM_FC FC_GLOBAL(genrcm,GENRCM)
  void GENRCM_FC(int *neqns, int *xadj, int *adjncy, int *perm,
                 int *mask, int *xls);
}

namespace strumpack {

  template<typename integer_t> inline void WRAPPER_rcm
  (int n, int* xadj, int* adjncy, integer_t* perm) {
    auto mask = new int[5*n];
    auto xls = mask + 2*n;
    auto int_perm = xls + 2*n;
    GENRCM_FC(&n, xadj, adjncy, int_perm, mask, xls);
    for (int i=0; i<n; i++) perm[i] = int_perm[i]-1;
    delete[] mask;
  }
  template<> inline void WRAPPER_rcm
  (int n, int* xadj, int* adjncy, int* perm) {
    auto mask = new int[4*n];
    auto xls = mask + 2*n;
    GENRCM_FC(&n, xadj, adjncy, perm, mask, xls);
    for (int i=0; i<n; i++) perm[i]--;
    delete[] mask;
  }

  template<typename scalar_t,typename integer_t>
  std::unique_ptr<SeparatorTree<integer_t>>
  rcm_reordering(CompressedSparseMatrix<scalar_t,integer_t>* A,
                 integer_t* perm, integer_t* iperm) {
    auto n = A->size();
    auto ptr = A->get_ptr();
    auto ind = A->get_ind();
    auto xadj = new int[n+1 + ptr[n]];
    auto adjncy = xadj + n+1;
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
    WRAPPER_rcm(n, xadj, adjncy, perm);
    delete[] xadj;
    return build_sep_tree_from_perm(n, ptr, ind, perm, iperm);
  }

} // end namespace strumpack

#endif
