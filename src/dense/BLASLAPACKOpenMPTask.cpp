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
#include "BLASLAPACKOpenMPTask.hpp"

namespace strumpack {

  // these are wrappers to the templated version of gemm_omp_task
  // to make it easily callable from fortran
  void SGEMM_OMP_TASK_FC
  (char* ta, char* tb, int* m, int* n, int* k, float* alpha,
   const float* a, int* lda, const float* b, int* ldb, float* beta,
   float* c, int* ldc, int* depth) {
    gemm_omp_task
      (*ta, *tb, *m, *n, *k, *alpha, a, *lda,
       b, *ldb, *beta, c, *ldc, *depth);
  }
  void DGEMM_OMP_TASK_FC
  (char* ta, char* tb, int* m, int* n, int* k, double* alpha,
   const double* a, int* lda, const double* b, int* ldb, double* beta,
   double* c, int* ldc, int* depth) {
    gemm_omp_task
      (*ta, *tb, *m, *n, *k, *alpha, a, *lda,
       b, *ldb, *beta, c, *ldc, *depth);
  }
  void CGEMM_OMP_TASK_FC
  (char* ta, char* tb, int* m, int* n, int* k, std::complex<float>* alpha,
   const std::complex<float>* a, int* lda,
   const std::complex<float>* b, int* ldb, std::complex<float>* beta,
   std::complex<float>* c, int* ldc, int* depth) {
    gemm_omp_task
      (*ta, *tb, *m, *n, *k, *alpha, a, *lda,
       b, *ldb, *beta, c, *ldc, *depth);
  }
  void ZGEMM_OMP_TASK_FC
  (char* ta, char* tb, int* m, int* n, int* k, std::complex<double>* alpha,
   const std::complex<double>* a, int* lda,
   const std::complex<double>* b, int* ldb, std::complex<double>* beta,
   std::complex<double>* c, int* ldc, int* depth) {
    gemm_omp_task
      (*ta, *tb, *m, *n, *k, *alpha, a, *lda,
       b, *ldb, *beta, c, *ldc, *depth);
  }


  void STRSM_OMP_TASK_FC
  (char* s, char* u, char* t, char* d, int* m, int* n, float* alpha,
   const float* a, int* lda, float* b, int* ldb, int* depth) {
    trsm_omp_task(*s, *u, *t, *d, *m, *n, *alpha, a, *lda, b, *ldb, *depth);
  }
  void DTRSM_OMP_TASK_FC
  (char* s, char* u, char* ta, char* d, int* m, int* n, double* alpha,
   const double* a, int* lda, double* b, int* ldb, int* depth) {
    trsm_omp_task(*s, *u, *ta, *d, *m, *n, *alpha, a, *lda, b, *ldb, *depth);
  }
  void CTRSM_OMP_TASK_FC
  (char* s, char* u, char* ta, char* d, int* m, int* n,
   std::complex<float>* alpha, const std::complex<float>* a, int* lda,
   std::complex<float>* b, int* ldb, int* depth) {
    trsm_omp_task(*s, *u, *ta, *d, *m, *n, *alpha, a, *lda, b, *ldb, *depth);
  }
  void ZTRSM_OMP_TASK_FC
  (char* s, char* u, char* ta, char* d, int* m, int* n,
   std::complex<double>* alpha, const std::complex<double>* a, int* lda,
   std::complex<double>* b, int* ldb, int* depth) {
    trsm_omp_task(*s, *u, *ta, *d, *m, *n, *alpha, a, *lda, b, *ldb, *depth);
  }


  void STRMM_OMP_TASK_FC
  (char* s, char* u, char* ta, char* d, int* m, int* n, float* alpha,
   const float* a, int* lda, float* b, int* ldb, int* depth) {
    trmm_omp_task(*s, *u, *ta, *d, *m, *n, *alpha, a, *lda, b, *ldb, *depth);
  }
  void DTRMM_OMP_TASK_FC
  (char* s, char* u, char* ta, char* d, int* m, int* n, double* alpha,
   const double* a, int* lda, double* b, int* ldb, int* depth) {
    trmm_omp_task(*s, *u, *ta, *d, *m, *n, *alpha, a, *lda, b, *ldb, *depth);
  }
  void CTRMM_OMP_TASK_FC
  (char* s, char* u, char* ta, char* d, int* m, int* n,
   std::complex<float>* alpha, const std::complex<float>* a, int* lda,
   std::complex<float>* b, int* ldb, int* depth) {
    trmm_omp_task(*s, *u, *ta, *d, *m, *n, *alpha, a, *lda, b, *ldb, *depth);
  }
  void ZTRMM_OMP_TASK_FC
  (char* s, char* u, char* ta, char* d, int* m, int* n,
   std::complex<double>* alpha, const std::complex<double>* a, int* lda,
   std::complex<double>* b, int* ldb, int* depth) {
    trmm_omp_task(*s, *u, *ta, *d, *m, *n, *alpha, a, *lda, b, *ldb, *depth);
  }


  void STRMV_OMP_TASK_FC
  (char* u, char* ta, char* d, int* n, const float* a, int* lda,
   float* x, int* incx, int* depth) {
    trmv_omp_task(*u, *ta, *d, *n, a, *lda, x, *incx, *depth);
  }
  void DTRMV_OMP_TASK_FC
  (char* u, char* ta, char* d, int* n, const double* a, int* lda,
   double* x, int* incx, int* depth) {
    trmv_omp_task(*u, *ta, *d, *n, a, *lda, x, *incx, *depth);
  }
  void CTRMV_OMP_TASK_FC
  (char* u, char* ta, char* d, int* n,
   const std::complex<float>* a, int* lda,
   std::complex<float>* x, int* incx, int* depth) {
    trmv_omp_task(*u, *ta, *d, *n, a, *lda, x, *incx, *depth);
  }
  void ZTRMV_OMP_TASK_FC
  (char* u, char* ta, char* d, int* n,
   const std::complex<double>* a, int* lda,
   std::complex<double>* x, int* incx, int* depth) {
    trmv_omp_task(*u, *ta, *d, *n, a, *lda, x, *incx, *depth);
  }


  void SGER_OMP_TASK_FC
  (int* m, int* n, float* alpha, const float* x, int* incx,
   const float* y, int* incy, float* a, int* lda, int* depth) {
    geru_omp_task(*m, *n, *alpha, x, *incx, y, *incy, a, *lda, *depth);
  }
  void DGER_OMP_TASK_FC
  (int* m, int* n, double* alpha, const double* x, int* incx,
   const double* y, int* incy, double* a, int* lda, int* depth) {
    geru_omp_task(*m, *n, *alpha, x, *incx, y, *incy, a, *lda, *depth);
  }
  void CGERU_OMP_TASK_FC
  (int* m, int* n, std::complex<float>* alpha,
   const std::complex<float>* x, int* incx,
   const std::complex<float>* y, int* incy,
   std::complex<float>* a, int* lda, int* depth) {
    geru_omp_task(*m, *n, *alpha, x, *incx, y, *incy, a, *lda, *depth);
  }
  void ZGERU_OMP_TASK_FC
  (int* m, int* n, std::complex<double>* alpha,
   const std::complex<double>* x, int* incx,
   const std::complex<double>* y, int* incy,
   std::complex<double>* a, int* lda, int* depth) {
    geru_omp_task(*m, *n, *alpha, x, *incx, y, *incy, a, *lda, *depth);
  }
  void CGERC_OMP_TASK_FC
  (int* m, int* n, std::complex<float>* alpha,
   const std::complex<float>* x, int* incx,
   const std::complex<float>* y, int* incy,
   std::complex<float>* a, int* lda, int* depth) {
    gerc_omp_task(*m, *n, *alpha, x, *incx, y, *incy, a, *lda, *depth);
  }
  void ZGERC_OMP_TASK_FC
  (int* m, int* n, std::complex<double>* alpha,
   const std::complex<double>* x, int* incx,
   const std::complex<double>* y, int* incy,
   std::complex<double>* a, int* lda, int* depth) {
    gerc_omp_task(*m, *n, *alpha, x, *incx, y, *incy, a, *lda, *depth);
  }


  void SLASWP_OMP_TASK_FC
  (int* n, float* a, int* lda, int* k1, int* k2,
   const int* ipiv, int* incx, int* depth) {
    laswp_omp_task(*n, a, *lda, *k1, *k2, ipiv, *incx, *depth);
  }
  void DLASWP_OMP_TASK_FC
  (int* n, double* a, int* lda, int* k1, int* k2,
   const int* ipiv, int* incx, int* depth) {
    laswp_omp_task(*n, a, *lda, *k1, *k2, ipiv, *incx, *depth);
  }
  void CLASWP_OMP_TASK_FC
  (int* n, std::complex<float>* a, int* lda, int* k1, int* k2,
   const int* ipiv, int* incx, int* depth) {
    laswp_omp_task(*n, a, *lda, *k1, *k2, ipiv, *incx, *depth);
  }
  void ZLASWP_OMP_TASK_FC
  (int* n, std::complex<double>* a, int* lda, int* k1, int* k2,
   const int* ipiv, int* incx, int* depth) {
    laswp_omp_task(*n, a, *lda, *k1, *k2, ipiv, *incx, *depth);
  }


  void SGEMV_OMP_TASK_FC
  (char* t, int* m, int* n, float* alpha, const float *a, int* lda,
   const float* x, int* incx, float* beta, float* y, int* incy, int* depth) {
    gemv_omp_task
      (*t, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy, *depth);
  }
  void DGEMV_OMP_TASK_FC
  (char* t, int* m, int* n, double* alpha, const double *a, int* lda,
   const double* x, int* incx, double* beta,
   double* y, int* incy, int* depth) {
    gemv_omp_task
      (*t, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy, *depth);
  }
  void CGEMV_OMP_TASK_FC
  (char* t, int* m, int* n, std::complex<float>* alpha,
   const std::complex<float> *a, int* lda,
   const std::complex<float>* x, int* incx, std::complex<float>* beta,
   std::complex<float>* y, int* incy, int* depth) {
    gemv_omp_task
      (*t, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy, *depth);
  }
  void ZGEMV_OMP_TASK_FC
  (char* t, int* m, int* n, std::complex<double>* alpha,
   const std::complex<double> *a, int* lda,
   const std::complex<double>* x, int* incx, std::complex<double>* beta,
   std::complex<double>* y, int* incy, int* depth) {
    gemv_omp_task
      (*t, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy, *depth);
  }

} // end namespace strumpack
