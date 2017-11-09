/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The Regents of
 * the University of California, through Lawrence Berkeley National Laboratory
 * (subject to receipt of any required approvals from the U.S. Dept. of Energy).
 * All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Technology Transfer Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, prepare derivative works, and perform publicly and display publicly.
 * Beginning five (5) years after the date permission to assert copyright is
 * obtained from the U.S. Department of Energy, and subject to any subsequent five
 * (5) year renewals, the U.S. Government is granted for itself and others acting
 * on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, prepare derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 *
 */
#include "BLASLAPACKOpenMPTask.hpp"

namespace strumpack {

  // these are wrappers to the templated version of gemm_omp_task
  // to make it easily callable from fortran
  void SGEMM_OMP_TASK_FC(char* transa, char* transb, int* m, int* n, int* k, float* alpha, float* a, int* lda, float* b, int* ldb, float* beta, float* c, int* ldc, int* depth)
  { gemm_omp_task(*transa, *transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc, *depth); }
  void DGEMM_OMP_TASK_FC(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc, int* depth)
  { gemm_omp_task(*transa, *transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc, *depth); }
  void CGEMM_OMP_TASK_FC(char* transa, char* transb, int* m, int* n, int* k, c_float* alpha, c_float* a, int* lda, c_float* b, int* ldb, c_float* beta, c_float* c, int* ldc, int* depth)
  { gemm_omp_task(*transa, *transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc, *depth); }
  void ZGEMM_OMP_TASK_FC(char* transa, char* transb, int* m, int* n, int* k, c_double* alpha, c_double* a, int* lda, c_double* b, int* ldb, c_double* beta, c_double* c, int* ldc, int* depth)
  { gemm_omp_task(*transa, *transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc, *depth); }

  void STRSM_OMP_TASK_FC(char* side, char* uplo, char* transa, char* diag, int* M, int* N, float* alpha, float* A, int* lda, float* B, int* ldb, int* depth)
  { trsm_omp_task(*side, *uplo, *transa, *diag, *M, *N, *alpha, A, *lda, B, *ldb, *depth); }
  void DTRSM_OMP_TASK_FC(char* side, char* uplo, char* transa, char* diag, int* M, int* N, double* alpha, double* A, int* lda, double* B, int* ldb, int* depth)
  { trsm_omp_task(*side, *uplo, *transa, *diag, *M, *N, *alpha, A, *lda, B, *ldb, *depth); }
  void CTRSM_OMP_TASK_FC(char* side, char* uplo, char* transa, char* diag, int* M, int* N, c_float* alpha, c_float* A, int* lda, c_float* B, int* ldb, int* depth)
  { trsm_omp_task(*side, *uplo, *transa, *diag, *M, *N, *alpha, A, *lda, B, *ldb, *depth); }
  void ZTRSM_OMP_TASK_FC(char* side, char* uplo, char* transa, char* diag, int* M, int* N, c_double* alpha, c_double* A, int* lda, c_double* B, int* ldb, int* depth)
  { trsm_omp_task(*side, *uplo, *transa, *diag, *M, *N, *alpha, A, *lda, B, *ldb, *depth); }

  void STRMM_OMP_TASK_FC(char* side, char* uplo, char* transa, char* diag, int* M, int* N, float* alpha, float* A, int* lda, float* B, int* ldb, int* depth)
  { trmm_omp_task(*side, *uplo, *transa, *diag, *M, *N, *alpha, A, *lda, B, *ldb, *depth); }
  void DTRMM_OMP_TASK_FC(char* side, char* uplo, char* transa, char* diag, int* M, int* N, double* alpha, double* A, int* lda, double* B, int* ldb, int* depth)
  { trmm_omp_task(*side, *uplo, *transa, *diag, *M, *N, *alpha, A, *lda, B, *ldb, *depth); }
  void CTRMM_OMP_TASK_FC(char* side, char* uplo, char* transa, char* diag, int* M, int* N, c_float* alpha, c_float* A, int* lda, c_float* B, int* ldb, int* depth)
  { trmm_omp_task(*side, *uplo, *transa, *diag, *M, *N, *alpha, A, *lda, B, *ldb, *depth); }
  void ZTRMM_OMP_TASK_FC(char* side, char* uplo, char* transa, char* diag, int* M, int* N, c_double* alpha, c_double* A, int* lda, c_double* B, int* ldb, int* depth)
  { trmm_omp_task(*side, *uplo, *transa, *diag, *M, *N, *alpha, A, *lda, B, *ldb, *depth); }

  void STRMV_OMP_TASK_FC(char* uplo, char* transa, char* diag, int* N, float* A, int* lda, float* X, int* incx, int* depth)
  { trmv_omp_task(*uplo, *transa, *diag, *N, A, *lda, X, *incx, *depth); }
  void DTRMV_OMP_TASK_FC(char* uplo, char* transa, char* diag, int* N, double* A, int* lda, double* X, int* incx, int* depth)
  { trmv_omp_task(*uplo, *transa, *diag, *N, A, *lda, X, *incx, *depth); }
  void CTRMV_OMP_TASK_FC(char* uplo, char* transa, char* diag, int* N, c_float* A, int* lda, c_float* X, int* incx, int* depth)
  { trmv_omp_task(*uplo, *transa, *diag, *N, A, *lda, X, *incx, *depth); }
  void ZTRMV_OMP_TASK_FC(char* uplo, char* transa, char* diag, int* N, c_double* A, int* lda, c_double* X, int* incx, int* depth)
  { trmv_omp_task(*uplo, *transa, *diag, *N, A, *lda, X, *incx, *depth); }

  void SGER_OMP_TASK_FC(int* m, int* n, float* alpha, float* x, int* incx, float* y, int* incy, float* a, int* lda, int* depth)
  { geru_omp_task(*m, *n, *alpha, x, *incx, y, *incy, a, *lda, *depth); }
  void DGER_OMP_TASK_FC(int* m, int* n, double* alpha, double* x, int* incx, double* y, int* incy, double* a, int* lda, int* depth)
  { geru_omp_task(*m, *n, *alpha, x, *incx, y, *incy, a, *lda, *depth); }
  void CGERU_OMP_TASK_FC(int* m, int* n, c_float* alpha, c_float* x, int* incx, c_float* y, int* incy, c_float* a, int* lda, int* depth)
  { geru_omp_task(*m, *n, *alpha, x, *incx, y, *incy, a, *lda, *depth); }
  void ZGERU_OMP_TASK_FC(int* m, int* n, c_double* alpha, c_double* x, int* incx, c_double* y, int* incy, c_double* a, int* lda, int* depth)
  { geru_omp_task(*m, *n, *alpha, x, *incx, y, *incy, a, *lda, *depth); }
  void CGERC_OMP_TASK_FC(int* m, int* n, c_float* alpha, c_float* x, int* incx, c_float* y, int* incy, c_float* a, int* lda, int* depth)
  { gerc_omp_task(*m, *n, *alpha, x, *incx, y, *incy, a, *lda, *depth); }
  void ZGERC_OMP_TASK_FC(int* m, int* n, c_double* alpha, c_double* x, int* incx, c_double* y, int* incy, c_double* a, int* lda, int* depth)
  { gerc_omp_task(*m, *n, *alpha, x, *incx, y, *incy, a, *lda, *depth); }

  void SLASWP_OMP_TASK_FC(int* N, float* A, int* lda, int* k1, int* k2, int* ipiv, int* incx, int* depth)
  { laswp_omp_task(*N, A, *lda, *k1, *k2, ipiv, *incx, *depth); }
  void DLASWP_OMP_TASK_FC(int* N, double* A, int* lda, int* k1, int* k2, int* ipiv, int* incx, int* depth)
  { laswp_omp_task(*N, A, *lda, *k1, *k2, ipiv, *incx, *depth); }
  void CLASWP_OMP_TASK_FC(int* N, c_float* A, int* lda, int* k1, int* k2, int* ipiv, int* incx, int* depth)
  { laswp_omp_task(*N, A, *lda, *k1, *k2, ipiv, *incx, *depth); }
  void ZLASWP_OMP_TASK_FC(int* N, c_double* A, int* lda, int* k1, int* k2, int* ipiv, int* incx, int* depth)
  { laswp_omp_task(*N, A, *lda, *k1, *k2, ipiv, *incx, *depth); }

  void SGEMV_OMP_TASK_FC(char* trans, int* M, int* N, float* alpha, float *A, int* lda, float* X, int* incx, float* beta, float* Y, int* incy, int* depth)
  { gemv_omp_task(*trans, *M, *N, *alpha, A, *lda, X, *incx, *beta, Y, *incy, *depth); }
  void DGEMV_OMP_TASK_FC(char* trans, int* M, int* N, double* alpha, double *A, int* lda, double* X, int* incx, double* beta, double* Y, int* incy, int* depth)
  { gemv_omp_task(*trans, *M, *N, *alpha, A, *lda, X, *incx, *beta, Y, *incy, *depth); }
  void CGEMV_OMP_TASK_FC(char* trans, int* M, int* N, c_float* alpha, c_float *A, int* lda, c_float* X, int* incx, c_float* beta, c_float* Y, int* incy, int* depth)
  { gemv_omp_task(*trans, *M, *N, *alpha, A, *lda, X, *incx, *beta, Y, *incy, *depth); }
  void ZGEMV_OMP_TASK_FC(char* trans, int* M, int* N, c_double* alpha, c_double *A, int* lda, c_double* X, int* incx, c_double* beta, c_double* Y, int* incy, int* depth)
  { gemv_omp_task(*trans, *M, *N, *alpha, A, *lda, X, *incx, *beta, Y, *incy, *depth); }

} // end namespace strumpack
