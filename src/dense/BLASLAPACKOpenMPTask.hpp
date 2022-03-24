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
#ifndef BLASLAPACKOMPTASK_HPP
#define BLASLAPACKOMPTASK_HPP
#include <iostream>
#include <stdlib.h>
#include "BLASLAPACKWrapper.hpp"
#include "StrumpackParameters.hpp"

namespace strumpack {

  template<typename scalar> void gemm_omp_task(char ta, char tb, int m, int n, int k, scalar alpha, const scalar* a, int lda, const scalar* b, int ldb, scalar beta, scalar* c, int ldc, int depth);
  template<typename scalar> void gemv_omp_task(char t, int m, int n, scalar alpha, const scalar *a, int lda, const scalar *x, int incx, scalar beta, scalar *y, int incy, int depth);
  template<typename scalar> void trsv_omp_task(char ul, char ta, char d, int n, const scalar* a, int lda, scalar* x, int incx, int depth);
  template<typename scalar> void trmm_omp_task(char s, char ul, char ta, char d, int m, int n, scalar alpha, const scalar* a, int lda, scalar* b, int ldb, int depth);
  template<typename scalar> void trmv_omp_task(char ul, char ta, char d, int n, const scalar* a, int lda, scalar* x, int incx, int depth);
  template<typename scalar> void trsm_omp_task(char s, char ul, char ta, char d, int m, int n, scalar alpha, const scalar* a, int lda, scalar* b, int ldb, int depth);
  template<typename scalar> void laswp_omp_task(int n, scalar* a, int lda, int k1, int k2, const int* ipiv, int incx, int depth);
  template<typename scalar> int getrf_omp_task(int m, int n, scalar* a, int lda, int* ipiv, int depth);
  template<typename scalar> int getrs_omp_task(char t, int m, int n, const scalar *a, int lda, const int* piv, scalar *b, int ldb, int depth);

} // end namespace strumpack

#endif
