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
#include "BLASLAPACKWrapper.hpp"
#include "StrumpackFortranCInterface.h"

namespace strumpack {

  namespace blas {

    ///////////////////////////////////////////////////////////
    ///////// BLAS1 ///////////////////////////////////////////
    ///////////////////////////////////////////////////////////
    extern "C" {
      void STRUMPACK_FC_GLOBAL(scopy,SCOPY)
        (int* n, const float* x, int* incx, float* y, int* incy);
      void STRUMPACK_FC_GLOBAL(dcopy,DCOPY)
        (int* n, const double* x, int* incx, double* y, int* incy);
      void STRUMPACK_FC_GLOBAL(ccopy,CCOPY)
        (int* n, const std::complex<float>* x, int* incx,
         std::complex<float>* y, int* incy);
      void STRUMPACK_FC_GLOBAL(zcopy,ZCOPY)
        (int* n, const std::complex<double>* x, int* incx,
         std::complex<double>* y, int* incy);

      void STRUMPACK_FC_GLOBAL(sscal,SSCAL)
        (int* n, float* alpha, float* x, int* incx);
      void STRUMPACK_FC_GLOBAL(dscal,DSCAL)
        (int* n, double* alpha, double* x, int* incx);
      void STRUMPACK_FC_GLOBAL(cscal,CSCAL)
        (int* n, std::complex<float>* alpha,
         std::complex<float>* x, int* incx);
      void STRUMPACK_FC_GLOBAL(zscal,ZSCAL)
        (int* n, std::complex<double>* alpha,
         std::complex<double>* x, int* incx);

      int STRUMPACK_FC_GLOBAL(isamax,ISAMAX)
        (int* n, const float* dx, int* incx);
      int STRUMPACK_FC_GLOBAL(idamax,IDAMAX)
        (int* n, const double* dx, int* incx);
      int STRUMPACK_FC_GLOBAL(icamax,ICAMAX)
        (int* n, const std::complex<float>* dx, int* incx);
      int STRUMPACK_FC_GLOBAL(izamax,IZAMAX)
        (int* n, const std::complex<double>* dx, int* incx);

      float STRUMPACK_FC_GLOBAL(snrm2,SNRM2)
        (int* n, const float* x, int* incx);
      double STRUMPACK_FC_GLOBAL(dnrm2,DNRM2)
        (int* n, const double* x, int* incx);
      float STRUMPACK_FC_GLOBAL(scnrm2,SCNRM2)
        (int* n, const std::complex<float>* x, int* incx);
      double STRUMPACK_FC_GLOBAL(dznrm2,DZNRM2)
        (int* n, const std::complex<double>* x, int* incx);

      void STRUMPACK_FC_GLOBAL(saxpy,SAXPY)
        (int* n, float* alpha, const float* x, int* incx,
         float* y, int* incy);
      void STRUMPACK_FC_GLOBAL(daxpy,DAXPY)
        (int* n, double* alpha, const double* x, int* incx,
         double* y, int* incy);
      void STRUMPACK_FC_GLOBAL(caxpy,CAXPY)
        (int* n, std::complex<float>* alpha,
         const std::complex<float>* x, int* incx,
         std::complex<float>* y, int* incy);
      void STRUMPACK_FC_GLOBAL(zaxpy,ZAXPY)
        (int* n, std::complex<double>* alpha,
         const std::complex<double>* x, int* incx,
         std::complex<double>* y, int* incy);

      void STRUMPACK_FC_GLOBAL(sswap,SSWAP)
        (int* n, float* x, int* ldx, float* y, int* ldy);
      void STRUMPACK_FC_GLOBAL(dswap,DSWAP)
        (int* n, double* x, int* ldx, double* y, int* ldy);
      void STRUMPACK_FC_GLOBAL(cswap,CSWAP)
        (int* n, std::complex<float>* x, int* ldx,
         std::complex<float>* y, int* ldy);
      void STRUMPACK_FC_GLOBAL(zswap,ZSWAP)
        (int* n, std::complex<double>* x, int* ldx,
         std::complex<double>* y, int* ldy);

      float STRUMPACK_FC_GLOBAL(sdot,SDOT)
        (int* n, const float* x, int* incx, const float* y, int* incy);
      double STRUMPACK_FC_GLOBAL(ddot,DDOT)
        (int* n, const double* x, int* incx, const double* y, int* incy);


      ///////////////////////////////////////////////////////////
      ///////// BLAS2 ///////////////////////////////////////////
      ///////////////////////////////////////////////////////////
      void STRUMPACK_FC_GLOBAL(sgemv,SGEMV)
        (char* t, int* m, int* n, float* alpha, const float *a, int* lda,
         const float* x, int* incx, float* beta, float* y, int* incy);
      void STRUMPACK_FC_GLOBAL(dgemv,DGEMV)
        (char* t, int* m, int* n, double* alpha, const double *a, int* lda,
         const double* x, int* incx, double* beta, double* y, int* incy);
      void STRUMPACK_FC_GLOBAL(cgemv,CGEMV)
        (char* t, int* m, int* n, std::complex<float>* alpha,
         const std::complex<float> *a, int* lda,
         const std::complex<float>* x, int* incx, std::complex<float>* beta,
         std::complex<float>* y, int* incy);
      void STRUMPACK_FC_GLOBAL(zgemv,ZGEMV)
        (char* t, int* m, int* n, std::complex<double>* alpha,
         const std::complex<double> *a, int* lda,
         const std::complex<double>* x, int* incx, std::complex<double>* beta,
         std::complex<double>* y, int* incy);

      void STRUMPACK_FC_GLOBAL(sger,SGER)
        (int* m, int* n, float* alpha, const float* x, int* incx,
         const float* y, int* incy, float* a, int* lda);
      void STRUMPACK_FC_GLOBAL(dger,DGER)
        (int* m, int* n, double* alpha, const double* x, int* incx,
         const double* y, int* incy, double* a, int* lda);
      void STRUMPACK_FC_GLOBAL(cgeru,CGERU)
        (int* m, int* n, std::complex<float>* alpha,
         const std::complex<float>* x, int* incx,
         const std::complex<float>* y, int* incy,
         std::complex<float>* a, int* lda);
      void STRUMPACK_FC_GLOBAL(zgeru,ZGERU)
        (int* m, int* n, std::complex<double>* alpha,
         const std::complex<double>* x, int* incx,
         const std::complex<double>* y, int* incy,
         std::complex<double>* a, int* lda);
      void STRUMPACK_FC_GLOBAL(cgerc,CGERC)
        (int* m, int* n, std::complex<float>* alpha,
         const std::complex<float>* x, int* incx,
         const std::complex<float>* y, int* incy,
         std::complex<float>* a, int* lda);
      void STRUMPACK_FC_GLOBAL(zgerc,ZGERC)
        (int* m, int* n, std::complex<double>* alpha,
         const std::complex<double>* x, int* incx,
         const std::complex<double>* y, int* incy,
         std::complex<double>* a, int* lda);

      void STRUMPACK_FC_GLOBAL(strmv,STRMV)
        (char* ul, char* t, char* d, int* n,
         const float* a, int* lda, float* x, int* incx);
      void STRUMPACK_FC_GLOBAL(dtrmv,DTRMV)
        (char* ul, char* t, char* d, int* n,
         const double* a, int* lda, double* x, int* incx);
      void STRUMPACK_FC_GLOBAL(ctrmv,CTRMV)
        (char* ul, char* t, char* d, int* n,
         const std::complex<float>* a, int* lda,
         std::complex<float>* x, int* incx);
      void STRUMPACK_FC_GLOBAL(ztrmv,ZTRMV)
        (char* ul, char* t, char* d, int* n,
         const std::complex<double>* a, int* lda,
         std::complex<double>* x, int* incx);

      void STRUMPACK_FC_GLOBAL(strsv,STRSV)
        (char* ul, char* t, char* d, int* m, const float* a, int* lda,
         float* b, int* incb);
      void STRUMPACK_FC_GLOBAL(dtrsv,DTRSV)
        (char* ul, char* t, char* d, int* m, const double* a, int* lda,
         double* b, int* incb);
      void STRUMPACK_FC_GLOBAL(ctrsv,CTRSV)
        (char* ul, char* t, char* d, int* m,
         const std::complex<float>* a, int* lda,
         std::complex<float>* b, int* incb);
      void STRUMPACK_FC_GLOBAL(ztrsv,ZTRSV)
        (char* ul, char* t, char* d, int* m,
         const std::complex<double>* a, int* lda,
         std::complex<double>* b, int* incb);


      ///////////////////////////////////////////////////////////
      ///////// BLAS3 ///////////////////////////////////////////
      ///////////////////////////////////////////////////////////
      void STRUMPACK_FC_GLOBAL(sgemm,SGEMM)
        (char* ta, char* tb, int* m, int* n, int* k,
         float* alpha, const float* a, int* lda, const float* b, int* ldb,
         float* beta, float* c, int* ldc);
      void STRUMPACK_FC_GLOBAL(dgemm,DGEMM)
        (char* ta, char* tb, int* m, int* n, int* k,
         double* alpha, const double* A, int* lda, const double* b, int* ldb,
         double* beta, double* c, int* ldc);
      void STRUMPACK_FC_GLOBAL(cgemm,CGEMM)
        (char* ta, char* tb, int* m, int* n, int* k,
         std::complex<float>* alpha, const std::complex<float>* a, int* lda,
         const std::complex<float>* b, int* ldb, std::complex<float>* beta,
         std::complex<float>* c, int* ldc);
      void STRUMPACK_FC_GLOBAL(zgemm,ZGEMM)
        (char* ta, char* tb, int* m, int* n, int* k,
         std::complex<double>* alpha, const std::complex<double>* a, int* lda,
         const std::complex<double>* b, int* ldb, std::complex<double>* beta,
         std::complex<double>* c, int* ldc);

      void STRUMPACK_FC_GLOBAL(strsm,STRSM)
        (char* s, char* ul, char* t, char* d, int* m, int* n,
         float* alpha, const float* a, int* lda, float* b, int* ldb);
      void STRUMPACK_FC_GLOBAL(dtrsm,DTRSM)
        (char* s, char* ul, char* t, char* d, int* m, int* n,
         double* alpha, const double* a, int* lda, double* b, int* ldb);
      void STRUMPACK_FC_GLOBAL(ctrsm,CTRSM)
        (char* s, char* ul, char* t, char* d, int* m, int* n,
         std::complex<float>* alpha, const std::complex<float>* a, int* lda,
         std::complex<float>* b, int* ldb);
      void STRUMPACK_FC_GLOBAL(ztrsm,ZTRSM)
        (char* s, char* ul, char* t, char* d, int* m, int* n,
         std::complex<double>* alpha, const std::complex<double>* a, int* lda,
         std::complex<double>* b, int* ldb);

      void STRUMPACK_FC_GLOBAL(strmm,STRMM)
        (char* s, char* ul, char* t, char* d, int* m, int* n, float* alpha,
         const float* a, int* lda, float* b, int* ldb);
      void STRUMPACK_FC_GLOBAL(dtrmm,DTRMM)
        (char* s, char* ul, char* t, char* d, int* m, int* n, double* alpha,
         const double* a, int* lda, double* b, int* ldb);
      void STRUMPACK_FC_GLOBAL(ctrmm,CTRMM)
        (char* s, char* ul, char* t, char* d, int* m, int* n,
         std::complex<float>* alpha, const std::complex<float>* a, int* lda,
         std::complex<float>* b, int* ldb);
      void STRUMPACK_FC_GLOBAL(ztrmm,ZTRMM)
        (char* s, char* ul, char* t, char* d, int* m, int* n,
         std::complex<double>* alpha, const std::complex<double>* a, int* lda,
         std::complex<double>* b, int* ldb);


      ///////////////////////////////////////////////////////////
      ///////// LAPACK //////////////////////////////////////////
      ///////////////////////////////////////////////////////////
      float STRUMPACK_FC_GLOBAL(slamch,SLAMCH)(char* cmach);
      double STRUMPACK_FC_GLOBAL(dlamch,DLAMCH)(char* cmach);

      int STRUMPACK_FC_GLOBAL(ilaenv,ILAENV)
        (int* ispec, char* name, char* opts,
         int* n1, int* n2, int* n3, int* n4);

      void STRUMPACK_FC_GLOBAL(clacgv,CLACGV)
        (int* n, std::complex<float>* x, int* incx);
      void STRUMPACK_FC_GLOBAL(zlacgv,ZLACGV)
        (int* n, std::complex<double>* x, int* incx);

      int STRUMPACK_FC_GLOBAL(slacpy,SLACPY)
        (char* uplo, int* m, int* n, const float* a, int* lda,
         float* b, int* ldb);
      int STRUMPACK_FC_GLOBAL(dlacpy,DLACPY)
        (char* uplo, int* m, int* n, const double* a, int* lda,
         double* b, int* ldb);
      int STRUMPACK_FC_GLOBAL(clacpy,CLACPY)
        (char* uplo, int* m, int* n, const std::complex<float>* a, int* lda,
         std::complex<float>* b, int* ldb);
      int STRUMPACK_FC_GLOBAL(zlacpy,ZLACPY)
        (char* uplo, int* m, int* n, const std::complex<double>* a, int* lda,
         std::complex<double>* b, int* ldb);

      void STRUMPACK_FC_GLOBAL(slaswp,SLASWP)
        (int* n, float* a, int* lda, int* k1, int* k2,
         const int* ipiv, int* incx);
      void STRUMPACK_FC_GLOBAL(dlaswp,DLASWP)
        (int* n, double* a, int* lda, int* k1, int* k2,
         const int* ipiv, int* incx);
      void STRUMPACK_FC_GLOBAL(claswp,CLASWP)
        (int* n, std::complex<float>* a, int* lda, int* k1, int* k2,
         const int* ipiv, int* incx);
      void STRUMPACK_FC_GLOBAL(zlaswp,ZLASWP)
        (int* n, std::complex<double>* a, int* lda, int* k1, int* k2,
         const int* ipiv, int* incx);

      void STRUMPACK_FC_GLOBAL(slapmr,SLAPMR)
        (int* fwd, int* m, int* n, float* a, int* lda, const int* ipiv);
      void STRUMPACK_FC_GLOBAL(dlapmr,DLAPMR)
        (int* fwd, int* m, int* n, double* a, int* lda, const int* ipiv);
      void STRUMPACK_FC_GLOBAL(clapmr,CLAPMR)
        (int* fwd, int* m, int* n, std::complex<float>* a, int* lda,
         const int* ipiv);
      void STRUMPACK_FC_GLOBAL(zlapmr,ZLAPMR)
        (int* fwd, int* m, int* n, std::complex<double>* a, int* lda,
         const int* ipiv);

      void STRUMPACK_FC_GLOBAL(slapmt,SLAPMT)
        (int* fwd, int* m, int* n, float* a, int* lda, const int* ipiv);
      void STRUMPACK_FC_GLOBAL(dlapmt,DLAPMT)
        (int* fwd, int* m, int* n, double* a, int* lda, const int* ipiv);
      void STRUMPACK_FC_GLOBAL(clapmt,CLAPMT)
        (int* fwd, int* m, int* n, std::complex<float>* a, int* lda,
         const int* ipiv);
      void STRUMPACK_FC_GLOBAL(zlapmt,ZLAPMT)
        (int* fwd, int* m, int* n, std::complex<double>* a, int* lda,
         const int* ipiv);

      void STRUMPACK_FC_GLOBAL(slaset,SLASET)
        (char* s, int* m, int* n, float* alpha,
         float* beta, float* a, int* lda);
      void STRUMPACK_FC_GLOBAL(dlaset,DLASET)
        (char* s, int* m, int* n, double* alpha,
         double* beta, double* a, int* lda);
      void STRUMPACK_FC_GLOBAL(claset,CLASET)
        (char* s, int* m, int* n, std::complex<float>* alpha,
         std::complex<float>* beta, std::complex<float>* a, int* lda);
      void STRUMPACK_FC_GLOBAL(zlaset,ZLASET)
        (char* s, int* m, int* n, std::complex<double>* alpha,
         std::complex<double>* beta, std::complex<double>* a, int* lda);

      void STRUMPACK_FC_GLOBAL(sgeqp3,SGEQP3)
        (int* m, int* n, float* a, int* lda, int* jpvt,
         float* tau, float* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(dgeqp3,DGEQP3)
        (int* m, int* n, double* a, int* lda, int* jpvt,
         double* tau, double* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(cgeqp3,CGEQP3)
        (int* m, int* n, std::complex<float>* a, int* lda, int* jpvt,
         std::complex<float>* tau, std::complex<float>* work, int* lwork,
         std::complex<float>* rwork, int* info);
      void STRUMPACK_FC_GLOBAL(zgeqp3,ZGEQP3)
        (int* m, int* n, std::complex<double>* a, int* lda, int* jpvt,
         std::complex<double>* tau, std::complex<double>* work, int* lwork,
         std::complex<double>* rwork, int* info);

      void STRUMPACK_FC_GLOBAL(sgeqp3tol,SGEQP3TOL)
        (int* m, int* n, float* a, int* lda, int* jpvt,
         float* tau, float* work, int* lwork, int* info,
         int* rank, float* rtol, float* atol, int* depth);
      void STRUMPACK_FC_GLOBAL(dgeqp3tol,DGEQP3TOL)
        (int* m, int* n, double* a, int* lda, int* jpvt,
         double* tau, double* work, int* lwork, int* info,
         int* rank, double* rtol, double* atol, int* depth);
      void STRUMPACK_FC_GLOBAL(cgeqp3tol,CGEQP3TOL)
        (int* m, int* n, std::complex<float>* a, int* lda, int* jpvt,
         std::complex<float>* tau, std::complex<float>* work, int* lwork,
         float* rwork, int* info, int* rank,
         float* rtol, float* atol, int* depth);
      void STRUMPACK_FC_GLOBAL(zgeqp3tol,ZGEQP3TOL)
        (int* m, int* n, std::complex<double>* a, int* lda, int* jpvt,
         std::complex<double>* tau, std::complex<double>* work, int* lwork,
         double* rwork, int* info, int* rank,
         double* rtol, double* atol, int* depth);

      void STRUMPACK_FC_GLOBAL(sgeqrf,SGEQRF)
        (int* m, int* n, float* a, int* lda, float* tau,
         float* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(dgeqrf,DGEQRF)
        (int* m, int* n, double* a, int* lda, double* tau,
         double* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(cgeqrf,CGEQRF)
        (int* m, int* n, std::complex<float>* a, int* lda,
         std::complex<float>* tau, std::complex<float>* work,
         int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(zgeqrf,ZGEQRF)
        (int* m, int* n, std::complex<double>* a, int* lda,
         std::complex<double>* tau, std::complex<double>* work,
         int* lwork, int* info);

      void STRUMPACK_FC_GLOBAL(sgeqrfmod,SGEQRFMOD)
        (int* m, int* n, float* a, int* lda, float* tau,
         float* work, int* lwork, int* info, int* depth);
      void STRUMPACK_FC_GLOBAL(dgeqrfmod,DGEQRFMOD)
        (int* m, int* n, double* a, int* lda, double* tau,
         double* work, int* lwork, int* info, int* depth);
      void STRUMPACK_FC_GLOBAL(cgeqrfmod,CGEQRFMOD)
        (int* m, int* n, std::complex<float>* a, int* lda,
         std::complex<float>* tau, std::complex<float>* work, int* lwork,
         int* info, int* depth);
      void STRUMPACK_FC_GLOBAL(zgeqrfmod,ZGEQRFMOD)
        (int* m, int* n, std::complex<double>* a, int* lda,
         std::complex<double>* tau, std::complex<double>* work, int* lwork,
         int* info, int* depth);

      void STRUMPACK_FC_GLOBAL(sgelqf,SGELQF)
        (int* m, int* n, float* a, int* lda, float* tau,
         float* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(dgelqf,DGELQF)
        (int* m, int* n, double* a, int* lda, double* tau,
         double* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(cgelqf,CGELQF)
        (int* m, int* n, std::complex<float>* a, int* lda,
         std::complex<float>* tau, std::complex<float>* work,
         int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(zgelqf,ZGELQF)
        (int* m, int* n, std::complex<double>* a, int* lda,
         std::complex<double>* tau, std::complex<double>* work,
         int* lwork, int* info);

      void STRUMPACK_FC_GLOBAL(sgelqfmod,SGELQFMOD)
        (int* m, int* n, float* a, int* lda, float* tau,
         float* work, int* lwork, int* info, int* depth);
      void STRUMPACK_FC_GLOBAL(dgelqfmod,DGELQFMOD)
        (int* m, int* n, double* a, int* lda, double* tau,
         double* work, int* lwork, int* info, int* depth);
      void STRUMPACK_FC_GLOBAL(cgelqfmod,CGELQFMOD)
        (int* m, int* n, std::complex<float>* a, int* lda,
         std::complex<float>* tau, std::complex<float>* work, int* lwork,
         int* info, int* depth);
      void STRUMPACK_FC_GLOBAL(zgelqfmod,ZGELQFMOD)
        (int* m, int* n, std::complex<double>* a, int* lda,
         std::complex<double>* tau, std::complex<double>* work, int* lwork,
         int* info, int* depth);

      void STRUMPACK_FC_GLOBAL(sgetrf,SGETRF)
        (int* m, int* n, float* a, int* lda, int* ipiv, int* info);
      void STRUMPACK_FC_GLOBAL(dgetrf,DGETRF)
        (int* m, int* n, double* a, int* lda, int* ipiv, int* info);
      void STRUMPACK_FC_GLOBAL(cgetrf,CGETRF)
        (int* m, int* n, std::complex<float>* a, int* lda,
         int* ipiv, int* info);
      void STRUMPACK_FC_GLOBAL(zgetrf,ZGETRF)
        (int* m, int* n, std::complex<double>* a, int* lda,
         int* ipiv, int* info);

      void STRUMPACK_FC_GLOBAL(sgetrfmod,SGETRFMOD)
        (int* m, int* n, float* a, int* lda,
         int* ipiv, int* info, int* depth);
      void STRUMPACK_FC_GLOBAL(dgetrfmod,DGETRFMOD)
        (int* m, int* n, double* a, int* lda,
         int* ipiv, int* info, int* depth);
      void STRUMPACK_FC_GLOBAL(cgetrfmod,CGETRFMOD)
        (int* m, int* n, std::complex<float>* a, int* lda,
         int* ipiv, int* info, int* depth);
      void STRUMPACK_FC_GLOBAL(zgetrfmod,ZGETRFMOD)
        (int* m, int* n, std::complex<double>* a, int* lda,
         int* ipiv, int* info, int* depth);

      void STRUMPACK_FC_GLOBAL(sgetf2,SGETF2)
        (int* m, int* n, float* a, int* lda, int* ipiv, int* info);
      void STRUMPACK_FC_GLOBAL(dgetf2,DGETF2)
        (int* m, int* n, double* a, int* lda, int* ipiv, int* info);
      void STRUMPACK_FC_GLOBAL(cgetf2,CGETF2)
        (int* m, int* n, std::complex<float>* a, int* lda,
         int* ipiv, int* info);
      void STRUMPACK_FC_GLOBAL(zgetf2,ZGETF2)
        (int* m, int* n, std::complex<double>* a, int* lda,
         int* ipiv, int* info);

      void STRUMPACK_FC_GLOBAL(sgetrs,SGETRS)
        (char* t, int* n, int* nrhs, const float* a, int* lda,
         const int* ipiv, float* b, int* ldb, int* info);
      void STRUMPACK_FC_GLOBAL(dgetrs,dgetrs)
        (char* t, int* n, int* nrhs, const double* a, int* lda,
         const int* ipiv, double* b, int* ldb, int* info);
      void STRUMPACK_FC_GLOBAL(cgetrs,cgetrs)
        (char* t, int* n, int* nrhs, const std::complex<float>* a, int* lda,
         const int* ipiv, std::complex<float>* b, int* ldb, int* info);
      void STRUMPACK_FC_GLOBAL(zgetrs,ZGETRS)
        (char* t, int* n, int* nrhs, const std::complex<double>* a, int* lda,
         const int* ipiv, std::complex<double>* b, int* ldb, int* info);

      void STRUMPACK_FC_GLOBAL(spotrf,SPOTRF)
        (char* ul, int* n, float* a, int* lda, int* info);
      void STRUMPACK_FC_GLOBAL(dpotrf,DPOTRF)
        (char* ul, int* n, double* a, int* lda, int* info);
      void STRUMPACK_FC_GLOBAL(cpotrf,CPOTRF)
        (char* ul, int* n, std::complex<float>* a, int* lda, int* info);
      void STRUMPACK_FC_GLOBAL(zpotrf,ZPOTRF)
        (char* ul, int* n, std::complex<double>* a,
         int* lda, int* info);

      void STRUMPACK_FC_GLOBAL(sgetri,SGETRI)
        (int* n, float* a, int* lda, int* ipiv,
         float* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(dgetri,DGETRI)
        (int* n, double* a, int* lda, int* ipiv,
         double* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(cgetri,CGETRI)
        (int* n, std::complex<float>* a, int* lda, int* ipiv,
         std::complex<float>* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(zgetri,ZGETRI)
        (int* n, std::complex<double>* a, int* lda, int* ipiv,
         std::complex<double>* work, int* lwork, int* info);

      void STRUMPACK_FC_GLOBAL(sorglq,SORGLQ)
        (int* m, int* n, int* k, float* a, int* lda, const float* tau,
         float* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(dorglq,DORGLQ)
        (int* m, int* n, int* k, double* a, int* lda, const double* tau,
         double* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(cunglq,CORGLQ)
        (int* m, int* n, int* k, std::complex<float>* a, int* lda,
         const std::complex<float>* tau, std::complex<float>* work,
         int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(zunglq,ZORGLQ)
        (int* m, int* n, int* k, std::complex<double>* a, int* lda,
         const std::complex<double>* tau, std::complex<double>* work,
         int* lwork, int* info);

      void STRUMPACK_FC_GLOBAL(sorglqmod,SORQLQMOD)
        (int* m, int* n, int* k, float* a, int* lda, const float* tau,
         float* work, int* lwork, int* info, int* depth);
      void STRUMPACK_FC_GLOBAL(dorglqmod,DORQLQMOD)
        (int* m, int* n, int* k, double* a, int* lda, const double* tau,
         double* work, int* lwork, int* info, int* depth);
      void STRUMPACK_FC_GLOBAL(cunglqmod,CORQLQMOD)
        (int* m, int* n, int* k, std::complex<float>* a, int* lda,
         const std::complex<float>* tau, std::complex<float>* work,
         int* lwork, int* info, int* depth);
      void STRUMPACK_FC_GLOBAL(zunglqmod,ZORQLQMOD)
        (int* m, int* n, int* k, std::complex<double>* a, int* lda,
         const std::complex<double>* tau, std::complex<double>* work,
         int* lwork, int* info, int* depth);

      void STRUMPACK_FC_GLOBAL(sorgqr,SORGQR)
        (int* m, int* n, int* k, float* a, int* lda, const float* tau,
         float* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(dorgqr,DORGQR)
        (int* m, int* n, int* k, double* a, int* lda, const double* tau,
         double* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(cungqr,CORGQR)
        (int* m, int* n, int* k, std::complex<float>* a, int* lda,
         const std::complex<float>* tau, std::complex<float>* work,
         int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(zungqr,ZORGQR)
        (int* m, int* n, int* k, std::complex<double>* a, int* lda,
         const std::complex<double>* tau, std::complex<double>* work,
         int* lwork, int* info);

      void STRUMPACK_FC_GLOBAL(sormqr,SORMQR)
        (char* side, char* trans, int* m, int* n, int* k,
         const float* a, int* lda, const float* tau, float* c, int* ldc,
         float* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(dormqr,DORMQR)
        (char* size, char* trans, int* m, int* n, int* k,
         const double* a, int* lda, const double* tau, double* c, int* ldc,
         double* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(cunmqr,CORMQR)
        (char* size, char* trans, int* m, int* n, int* k,
         const std::complex<float>* a, int* lda,
         const std::complex<float>* tau, std::complex<float>* c, int* ldc,
         std::complex<float>* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(zunmqr,ZORMQR)
        (char* size, char* trans, int* m, int* n, int* k,
         const std::complex<double>* a, int* lda,
         const std::complex<double>* tau, std::complex<double>* c, int* ldc,
         std::complex<double>* work, int* lwork, int* info);

      void STRUMPACK_FC_GLOBAL(sgesv,SGESV)
        (int* n, int* nrhs, float* a, int* lda, int* ipiv,
         float* b, int* ldb, int* info);
      void STRUMPACK_FC_GLOBAL(dgesv,DGESV)
        (int* n, int* nrhs, double* a, int* lda, int* ipiv,
         double* b, int* ldb, int* info);
      void STRUMPACK_FC_GLOBAL(cgesv,CGESV)
        (int* n, int* nrhs, std::complex<float>* a, int* lda, int* ipiv,
         std::complex<float>* b, int* ldb, int* info);
      void STRUMPACK_FC_GLOBAL(zgesv,ZGESV)
        (int* n, int* nrhs, std::complex<double>* a, int* lda, int* ipiv,
         std::complex<double>* b, int* ldb, int* info);

      void STRUMPACK_FC_GLOBAL(slarnv,SLARNV)
        (int* idist, int* iseed, int* n, float* x);
      void STRUMPACK_FC_GLOBAL(dlarnv,DLARNV)
        (int* idist, int* iseed, int* n, double* x);
      void STRUMPACK_FC_GLOBAL(clarnv,CLARNV)
        (int* idist, int* iseed, int* n, std::complex<float>* x);
      void STRUMPACK_FC_GLOBAL(zlarnv,ZLARNV)
        (int* idist, int* iseed, int* n, std::complex<double>* x);

      float STRUMPACK_FC_GLOBAL(slange,SLANGE)
        (char* norm, int* m, int* n, const float* a, int* lda, float* work);
      double STRUMPACK_FC_GLOBAL(dlange,DLANGE)
        (char* norm, int* m, int* n, const double* a,int* lda, double* work);
      float STRUMPACK_FC_GLOBAL(clange,CLANGE)
        (char* norm, int* m, int* n,
         const std::complex<float>* a, int* lda, float* work);
      double STRUMPACK_FC_GLOBAL(zlange,ZLANGE)
        (char* norm, int* m, int* n,
         const std::complex<double>* a, int* lda, double* work);

      void STRUMPACK_FC_GLOBAL(sgesvd,SGESVD)
        (char* jobu, char* jobvt, int* m, int* n, float* a, int* lda,
         float* s, float* u, int* ldu, float* vt, int* ldvt,
         float* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(dgesvd,DGESVD)
        (char* jobu, char* jobvt, int* m, int* n, double* a, int* lda,
         double* s, double* u, int* ldu, double* vt, int* ldvt,
         double* work, int* lwork, int* info);

      void STRUMPACK_FC_GLOBAL(ssyevx,SSYEVX)
        (char* jobz, char* range, char* uplo, int* n,
         float* a, int* lda, float* vl, float* vu, int* il, int* iu,
         float* abstol, int* m, float* w, float* z, int* ldz,
         float* work, int* lwork, int* iwork, int* ifail, int* info);
      void STRUMPACK_FC_GLOBAL(dsyevx,DSYEVX)
        (char* jobz, char* range, char* uplo, int* n,
         double* a, int* lda, double* vl, double* vu, int* il, int* iu,
         double* abstol, int* m, double* w, double* z, int* ldz,
         double* work, int* lwork, int* iwork, int* ifail, int* info);

      void STRUMPACK_FC_GLOBAL(ssyev,SSYEV)
        (char* jobz, char* uplo, int* n, float* a, int* lda, float* w,
         float* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(dsyev,DSYEV)
        (char* jobz, char* uplo, int* n, double* a, int* lda, double* w,
         double* work, int* lwork, int* info);


      void STRUMPACK_FC_GLOBAL(ssytrf,SSYTRF)
         (char* s, int* n, float* a, int*lda, int* ipiv, float* work,
            int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(dsytrf,DSYTRF)
         (char* s, int* n, double* a, int*lda, int* ipiv, double* work,
            int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(csytrf,CSYTRF)
         (char* s, int* n, std::complex<float>* a, int*lda, int* ipiv,
            std::complex<float>* work, int* lwork, int* info);
      void STRUMPACK_FC_GLOBAL(zsytrf,ZSYTRF)
         (char* s, int* n, std::complex<double>* a, int*lda, int* ipiv,
            std::complex<double>* work, int* lwork, int* info);

      // void STRUMPACK_FC_GLOBAL_(ssytrf_rook,SSYTRF_ROOK)
      //    (char* s, int* n, float* a, int*lda, int* ipiv, float* work,
      //       int* lwork, int* info);
      // void STRUMPACK_FC_GLOBAL_(dsytrf_rook,DSYTRF_ROOK)
      //    (char* s, int* n, double* a, int*lda, int* ipiv, double* work,
      //       int* lwork, int* info);
      // void STRUMPACK_FC_GLOBAL_(csytrf_rook,CSYTRF_ROOK)
      //    (char* s, int* n, std::complex<float>* a, int*lda, int* ipiv,
      //       std::complex<float>* work, int* lwork, int* info);
      // void STRUMPACK_FC_GLOBAL_(zsytrf_rook,ZSYTRF_ROOK)
      //    (char* s, int* n, std::complex<double>* a, int*lda, int* ipiv,
      //       std::complex<double>* work, int* lwork, int* info);

      void STRUMPACK_FC_GLOBAL(ssytrs,SSYTRS)
        (char* s, int* n, int* nrhs, const float* a, int* lda, const int* ipiv,
         float* b, int* ldb, int* info);
      void STRUMPACK_FC_GLOBAL(dsytrs,DSYTRS)
        (char* s, int* n, int* nrhs, const double* a, int* lda, const int* ipiv,
         double* b, int* ldb, int* info);
      void STRUMPACK_FC_GLOBAL(csytrs,CSYTRS)
        (char* s, int* n, int* nrhs, const std::complex<float>* a, int* lda,
         const int* ipiv, std::complex<float>* b, int* ldb, int* info);
      void STRUMPACK_FC_GLOBAL(zsytrs,ZSYTRS)
        (char* s, int* n, int* nrhs, const std::complex<double>* a, int* lda,
         const int* ipiv, std::complex<double>* b, int* ldb, int* info);

      // void STRUMPACK_FC_GLOBAL_(ssytrs_rook,SSYTRS_ROOK)
      //   (char* s, int* n, int* nrhs, const float* a, int* lda, const int* ipiv,
      //    float* b, int* ldb, int* info);
      // void STRUMPACK_FC_GLOBAL_(dsytrs_rook,DSYTRS_ROOK)
      //   (char* s, int* n, int* nrhs, const double* a, int* lda, const int* ipiv,
      //    double* b, int* ldb, int* info);
      // void STRUMPACK_FC_GLOBAL_(csytrs_rook,CSYTRS_ROOK)
      //   (char* s, int* n, int* nrhs, const std::complex<float>* a, int* lda,
      //    const int* ipiv, std::complex<float>* b, int* ldb, int* info);
      // void STRUMPACK_FC_GLOBAL_(zsytrs_rook,ZSYTRS_ROOK)
      //   (char* s, int* n, int* nrhs, const std::complex<double>* a, int* lda,
      //    const int* ipiv, std::complex<double>* b, int* ldb, int* info);
    }

    int ilaenv
    (int ispec, char name[], char opts[], int n1, int n2, int n3, int n4) {
      return STRUMPACK_FC_GLOBAL
        (ilaenv,ILAENV)(&ispec, name, opts, &n1, &n2, &n3, &n4);
    }

    template<> float lamch<float>(char cmach) {
      return STRUMPACK_FC_GLOBAL(slamch,SLAMCH)(&cmach);
    }
    template<> double lamch<double>(char cmach) {
      return STRUMPACK_FC_GLOBAL(dlamch,DLAMCH)(&cmach);
    }

    void gemm
    (char ta, char tb, int m, int n, int k, float alpha,
     const float *a, int lda, const float *b, int ldb,
     float beta, float *c, int ldc) {
      STRUMPACK_FC_GLOBAL(sgemm,SGEMM)
        (&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      STRUMPACK_FLOPS(gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(4*gemm_moves(m,n,k));
    }
    void gemm
    (char ta, char tb, int m, int n, int k, double alpha,
     const double *a, int lda, const double *b, int ldb,
     double beta, double *c, int ldc) {
      STRUMPACK_FC_GLOBAL(dgemm,DGEMM)
        (&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      STRUMPACK_FLOPS(gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(8*gemm_moves(m,n,k));
    }
    void gemm
    (char ta, char tb, int m, int n, int k, std::complex<float> alpha,
     const std::complex<float>* a, int lda,
     const std::complex<float>* b, int ldb, std::complex<float> beta,
     std::complex<float>* c, int ldc) {
      STRUMPACK_FC_GLOBAL(cgemm,CGEMM)
        (&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      STRUMPACK_FLOPS(4*gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(2*4*gemm_moves(m,n,k));
    }
    void gemm
    (char ta, char tb, int m, int n, int k, std::complex<double> alpha,
     const std::complex<double>* a, int lda,
     const std::complex<double>* b, int ldb, std::complex<double> beta,
     std::complex<double>* c, int ldc) {
      STRUMPACK_FC_GLOBAL(zgemm,ZGEMM)
        (&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      STRUMPACK_FLOPS(4*gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(2*8*gemm_moves(m,n,k));
    }

    void gemv
    (char t, int m, int n, float alpha, const float *a, int lda,
     const float *x, int incx, float beta, float *y, int incy) {
      STRUMPACK_FC_GLOBAL(sgemv,SGEMV)
        (&t, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
      STRUMPACK_FLOPS(gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(4*gemv_moves(m,n));
    }
    void gemv
    (char t, int m, int n, double alpha, const double *a, int lda,
     const double *x, int incx, double beta, double *y, int incy) {
      STRUMPACK_FC_GLOBAL(dgemv,DGEMV)
        (&t, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
      STRUMPACK_FLOPS(gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(8*gemv_moves(m,n));
    }
    void gemv
    (char t, int m, int n, std::complex<float> alpha,
     const std::complex<float> *a, int lda,
     const std::complex<float> *x, int incx, std::complex<float> beta,
     std::complex<float> *y, int incy) {
      STRUMPACK_FC_GLOBAL(cgemv,CGEMV)
        (&t, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
      STRUMPACK_FLOPS(4*gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(2*4*gemv_moves(m,n));
    }
    void gemv
    (char t, int m, int n, std::complex<double> alpha,
     const std::complex<double> *a, int lda,
     const std::complex<double> *x, int incx, std::complex<double> beta,
     std::complex<double> *y, int incy) {
      STRUMPACK_FC_GLOBAL(zgemv,ZGEMV)
        (&t, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
      STRUMPACK_FLOPS(4*gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(2*8*gemv_moves(m,n));
    }


    void geru
    (int m, int n, float alpha, const float* x, int incx,
     const float* y, int incy, float* a, int lda) {
      STRUMPACK_FC_GLOBAL(sger,SGER)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(ger_flops(m,n,alpha));
    }
    void geru
    (int m, int n, double alpha, const double* x, int incx,
     const double* y, int incy, double* a, int lda) {
      STRUMPACK_FC_GLOBAL(dger,DGER)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(ger_flops(m,n,alpha));
    }
    void geru
    (int m, int n, std::complex<float> alpha,
     const std::complex<float>* x, int incx,
     const std::complex<float>* y, int incy,
     std::complex<float>* a, int lda) {
      STRUMPACK_FC_GLOBAL(cgeru,CGERU)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(4*ger_flops(m,n,alpha));
    }
    void geru
    (int m, int n, std::complex<double> alpha,
     const std::complex<double>* x, int incx,
     const std::complex<double>* y, int incy,
     std::complex<double>* a, int lda) {
      STRUMPACK_FC_GLOBAL(zgeru,ZGERU)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(4*ger_flops(m,n,alpha));
    }


    void gerc
    (int m, int n, float alpha, const float* x, int incx,
     const float* y, int incy, float* a, int lda) {
      STRUMPACK_FC_GLOBAL(sger,SGER)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(ger_flops(m,n,alpha));
    }
    void gerc
    (int m, int n, double alpha, const double* x, int incx,
     const double* y, int incy, double* a, int lda) {
      STRUMPACK_FC_GLOBAL(dger,DGER)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(ger_flops(m,n,alpha));
    }
    void gerc
    (int m, int n, std::complex<float> alpha,
     const std::complex<float>* x, int incx,
     const std::complex<float>* y, int incy,
     std::complex<float>* a, int lda) {
      STRUMPACK_FC_GLOBAL(cgerc,CGERC)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(4*ger_flops(m,n,alpha));
    }
    void gerc
    (int m, int n, std::complex<double> alpha,
     const std::complex<double>* x, int incx,
     const std::complex<double>* y, int incy,
     std::complex<double>* a, int lda) {
      STRUMPACK_FC_GLOBAL(zgerc,ZGERC)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(4*ger_flops(m,n,alpha));
    }


    void lacgv(int, float *, int ) { }
    void lacgv(int, double *, int ) { } //Nothing to do.
    void lacgv(int n, std::complex<float> *x, int incx) {
      STRUMPACK_FC_GLOBAL(clacgv,CLACGV)(&n, x, &incx);
    }
    void lacgv(int n, std::complex<double> *x, int incx) {
      STRUMPACK_FC_GLOBAL(zlacgv,ZLACGV)(&n, x, &incx);
    }


    void lacpy
    (char ul, int m, int n, float* a, int lda, float* b, int ldb) {
      STRUMPACK_FC_GLOBAL(slacpy,SLACPY)(&ul, &m, &n, a, &lda, b, &ldb);
    }
    void lacpy
    (char ul, int m, int n, double* a, int lda, double* b, int ldb) {
      STRUMPACK_FC_GLOBAL(dlacpy,DLACPY)(&ul, &m, &n, a, &lda, b, &ldb);
    }
    void lacpy
    (char ul, int m, int n, std::complex<float>* a, int lda,
     std::complex<float>* b, int ldb) {
      STRUMPACK_FC_GLOBAL(clacpy,CLACPY)(&ul, &m, &n, a, &lda, b, &ldb);
    }
    void lacpy
    (char ul, int m, int n, std::complex<double>* a, int lda,
     std::complex<double>* b, int ldb) {
      STRUMPACK_FC_GLOBAL(zlacpy,ZLACPY)(&ul, &m, &n, a, &lda, b, &ldb);
    }


    void axpy
    (int n, float alpha, float* x, int incx, float* y, int incy) {
      STRUMPACK_FC_GLOBAL(saxpy,SAXPY)(&n, &alpha, x, &incx, y, &incy);
      STRUMPACK_FLOPS(axpy_flops(n,alpha));
    }
    void axpy
    (int n, double alpha, double* x, int incx, double* y, int incy) {
      STRUMPACK_FC_GLOBAL(daxpy,DAXPY)(&n, &alpha, x, &incx, y, &incy);
      STRUMPACK_FLOPS(axpy_flops(n,alpha));
    }
    void axpy
    (int n, std::complex<float> alpha,
     const std::complex<float>* x, int incx,
     std::complex<float>* y, int incy) {
      STRUMPACK_FC_GLOBAL(caxpy,CAXPY)(&n, &alpha, x, &incx, y, &incy);
      STRUMPACK_FLOPS(4*axpy_flops(n,alpha));
    }
    void axpy
    (int n, std::complex<double> alpha,
     const std::complex<double>* x, int incx,
     std::complex<double>* y, int incy) {
      STRUMPACK_FC_GLOBAL(zaxpy,ZAXPY)(&n, &alpha, x, &incx, y, &incy);
      STRUMPACK_FLOPS(4*axpy_flops(n,alpha));
    }


    void copy
    (int n, const float* x, int incx, float* y, int incy) {
      STRUMPACK_FC_GLOBAL(scopy,SCOPY)(&n, x, &incx, y, &incy);
    }
    void copy
    (int n, const double* x, int incx, double* y, int incy) {
      STRUMPACK_FC_GLOBAL(dcopy,DCOPY)(&n, x, &incx, y, &incy);
    }
    void copy
    (int n, const std::complex<float>* x, int incx,
     std::complex<float>* y, int incy) {
      STRUMPACK_FC_GLOBAL(ccopy,CCOPY)(&n, x, &incx, y, &incy);
    }
    void copy
    (int n, const std::complex<double>* x, int incx,
     std::complex<double>* y, int incy) {
      STRUMPACK_FC_GLOBAL(zcopy,ZCOPY)(&n, x, &incx, y, &incy);
    }

    void scal(int n, float alpha, float* x, int incx) {
      STRUMPACK_FC_GLOBAL(sscal,SSCAL)(&n, &alpha, x, &incx);
      STRUMPACK_FLOPS(scal_flops(n,alpha));
    }
    void scal(int n, double alpha, double* x, int incx) {
      STRUMPACK_FC_GLOBAL(dscal,DSCAL)(&n, &alpha, x, &incx);
      STRUMPACK_FLOPS(scal_flops(n,alpha));
    }
    void scal
    (int n, std::complex<float> alpha, std::complex<float>* x, int incx) {
      STRUMPACK_FC_GLOBAL(cscal,CSCAL)(&n, &alpha, x, &incx);
      STRUMPACK_FLOPS(4*scal_flops(n,alpha));
    }
    void scal
    (int n, std::complex<double> alpha, std::complex<double>* x, int incx) {
      STRUMPACK_FC_GLOBAL(zscal,ZSCAL)(&n, &alpha, x, &incx);
      STRUMPACK_FLOPS(4*scal_flops(n,alpha));
    }


    int iamax
    (int n, const float* x, int incx) {
      return STRUMPACK_FC_GLOBAL(isamax,ISAMAX)(&n, x, &incx);
    }
    int iamax
    (int n, const double* x, int incx) {
      return STRUMPACK_FC_GLOBAL(idamax,IDAMAX)(&n, x, &incx);
    }
    int iamax
    (int n, const std::complex<float>* x, int incx) {
      return STRUMPACK_FC_GLOBAL(icamax,ICAMAX)(&n, x, &incx);
    }
    int iamax
    (int n, const std::complex<double>* x, int incx) {
      return STRUMPACK_FC_GLOBAL(izamax,IZAMAX)(&n, x, &incx);
    }


    void swap
    (int n, float* x, int incx, float* y, int incy) {
      STRUMPACK_FC_GLOBAL(sswap,SSWAP)(&n, x, &incx, y, &incy);
    }
    void swap
    (int n, double* x, int incx, double* y, int incy) {
      STRUMPACK_FC_GLOBAL(dswap,DSWAP)(&n, x, &incx, y, &incy);
    }
    void swap
    (int n, std::complex<float>* x, int incx,
     std::complex<float>* y, int incy) {
      STRUMPACK_FC_GLOBAL(cswap,CSWAP)(&n, x, &incx, y, &incy);
    }
    void swap
    (int n, std::complex<double>* x, int incx,
     std::complex<double>* y, int incy) {
      STRUMPACK_FC_GLOBAL(zswap,ZSWAP)(&n, x, &incx, y, &incy);
    }


    float nrm2(int n, const float* x, int incx) {
      STRUMPACK_FLOPS(nrm2_flops(n));
      return STRUMPACK_FC_GLOBAL(snrm2,SNRM2)(&n, x, &incx);
    }
    double nrm2(int n, const double* x, int incx) {
      STRUMPACK_FLOPS(nrm2_flops(n));
      return STRUMPACK_FC_GLOBAL(dnrm2,DNRM2)(&n, x, &incx);
    }
    float nrm2(int n, const std::complex<float>* x, int incx) {
      STRUMPACK_FLOPS(4*nrm2_flops(n));
      return STRUMPACK_FC_GLOBAL(scnrm2,SCNRM2)(&n, x, &incx);
    }
    double nrm2(int n, const std::complex<double>* x, int incx) {
      STRUMPACK_FLOPS(4*nrm2_flops(n));
      return STRUMPACK_FC_GLOBAL(dznrm2,DZNRM2)(&n, x, &incx);
    }


    float dotu
    (int n, const float* x, int incx, const float* y, int incy) {
      STRUMPACK_FLOPS(dot_flops(n))
        return STRUMPACK_FC_GLOBAL(sdot,SDOTU)(&n, x, &incx, y, &incy);
    }
    double dotu
    (int n, const double* x, int incx, const double* y, int incy) {
      STRUMPACK_FLOPS(dot_flops(n));
      return STRUMPACK_FC_GLOBAL(ddot,DDOT)(&n, x, &incx, y, &incy);
    }
    float dotc
    (int n, const float* x, int incx, const float* y, int incy) {
      STRUMPACK_FLOPS(dot_flops(n));
      return STRUMPACK_FC_GLOBAL(sdot,SDOT)(&n, x, &incx, y, &incy);
    }
    double dotc
    (int n, const double* x, int incx, const double* y, int incy) {
      STRUMPACK_FLOPS(dot_flops(n));
      return STRUMPACK_FC_GLOBAL(ddot,DDOT)(&n, x, &incx, y, &incy);
    }

    std::complex<float> dotu
    (int n, const std::complex<float>* x, int incx,
     const std::complex<float>* y, int incy) {
      STRUMPACK_FLOPS(4*dot_flops(n));
      std::complex<float> r(0.);
      for (int i=0; i<n; i++)
        r += x[i*incx]*y[i*incy];
      return r;
    }
    std::complex<double> dotu
    (int n, const std::complex<double>* x, int incx,
     const std::complex<double>* y, int incy) {
      STRUMPACK_FLOPS(4*dot_flops(n));
      std::complex<double> r(0.);
      for (int i=0; i<n; i++)
        r += x[i*incx]*y[i*incy];
      return r;
    }
    std::complex<float> dotc
    (int n, const std::complex<float>* x, int incx,
     const std::complex<float>* y, int incy) {
      STRUMPACK_FLOPS(4*dot_flops(n));
      std::complex<float> r(0.);
      for (int i=0; i<n; i++)
        r += std::conj(x[i*incx])*y[i*incy];
      return r;
    }
    std::complex<double> dotc
    (int n, const std::complex<double>* x, int incx,
     const std::complex<double>* y, int incy) {
      STRUMPACK_FLOPS(4*dot_flops(n));
      std::complex<double> r(0.);
      for (int i=0; i<n; i++)
        r += std::conj(x[i*incx])*y[i*incy];
      return r;
    }


    void laswp
    (int n, float* a, int lda, int k1, int k2, const int* ipiv, int incx) {
      STRUMPACK_FC_GLOBAL(slaswp,SLASWP)(&n, a, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(4*laswp_moves(n,k1,k2));
    }
    void laswp
    (int n, double* a, int lda, int k1, int k2, const int* ipiv, int incx) {
      STRUMPACK_FC_GLOBAL(dlaswp,DLASWP)(&n, a, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(8*laswp_moves(n,k1,k2));
    }
    void laswp
    (int n, std::complex<float>* a, int lda, int k1, int k2,
     const int* ipiv, int incx) {
      STRUMPACK_FC_GLOBAL(claswp,CLASWP)(&n, a, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(2*4*laswp_moves(n,k1,k2));
    }
    void laswp
    (int n, std::complex<double>* a, int lda, int k1, int k2,
     const int* ipiv, int incx) {
      STRUMPACK_FC_GLOBAL(zlaswp,ZLASWP)(&n, a, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(2*8*laswp_moves(n,k1,k2));
    }


    void lapmr
    (bool fwd, int m, int n, float* a, int lda, const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(slapmr,SLAPMR)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(4*lapmr_moves(n,m));
    }
    void lapmr
    (bool fwd, int m, int n, double* a, int lda, const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(dlapmr,DLAPMR)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(8*lapmr_moves(n,m));
    }
    void lapmr
    (bool fwd, int m, int n, std::complex<float>* a, int lda,
     const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(clapmr,CLAPMR)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(2*4*lapmr_moves(n,m));
    }
    void lapmr
    (bool fwd, int m, int n, std::complex<double>* a, int lda,
     const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(zlapmr,ZLAPMR)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(2*8*lapmr_moves(n,m));
    }

    void lapmt
    (bool fwd, int m, int n, float* a, int lda, const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(slapmt,SLAPMT)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(4*lapmt_moves(n,m));
    }
    void lapmt
    (bool fwd, int m, int n, double* a, int lda, const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(dlapmt,DLAPMT)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(8*lapmt_moves(n,m));
    }
    void lapmt
    (bool fwd, int m, int n, std::complex<float>* a, int lda,
     const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(clapmt,CLAPMT)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(2*4*lapmt_moves(n,m));
    }
    void lapmt
    (bool fwd, int m, int n, std::complex<double>* a, int lda,
     const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(zlapmt,ZLAPMT)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(2*8*lapmt_moves(n,m));
    }

    void trsm
    (char s, char ul, char t, char d, int m, int n, float alpha,
     const float* a, int lda, float* b, int ldb) {
      STRUMPACK_FC_GLOBAL(strsm,STRSM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(trsm_flops(m, n, alpha, s));
      STRUMPACK_BYTES(4*trsm_moves(m, n));
    }
    void trsm
    (char s, char ul, char t, char d, int m, int n, double alpha,
     const double* a, int lda, double* b, int ldb) {
      STRUMPACK_FC_GLOBAL(dtrsm,DTRSM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(trsm_flops(m, n, alpha, s));
      STRUMPACK_BYTES(8*trsm_moves(m, n));
    }
    void trsm
    (char s, char ul, char t, char d, int m, int n,
     std::complex<float> alpha, const std::complex<float>* a, int lda,
     std::complex<float>* b, int ldb) {
      STRUMPACK_FC_GLOBAL(ctrsm,CTRSM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(4*trsm_flops(m, n, alpha, s));
      STRUMPACK_BYTES(2*4*trsm_moves(m, n));
    }
    void trsm
    (char s, char ul, char t, char d, int m, int n,
     std::complex<double> alpha, const std::complex<double>* a, int lda,
     std::complex<double>* b, int ldb) {
      STRUMPACK_FC_GLOBAL(ztrsm,ZTRSM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(4*trsm_flops(m, n, alpha, s));
      STRUMPACK_BYTES(2*8*trsm_moves(m, n));
    }

    void trmm
    (char s, char ul, char t, char d, int m, int n, float alpha,
     const float* a, int lda, float* b, int ldb) {
      STRUMPACK_FC_GLOBAL(strmm,STRMM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(trmm_flops(m,n,alpha,s));
      STRUMPACK_BYTES(4*trmm_moves(m,n));
    }
    void trmm
    (char s, char ul, char t, char d, int m, int n, double alpha,
     const double* a, int lda, double* b, int ldb) {
      STRUMPACK_FC_GLOBAL(dtrmm,DTRMM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(trmm_flops(m,n,alpha,s));
      STRUMPACK_BYTES(8*trmm_moves(m,n));
    }
    void trmm
    (char s, char ul, char t, char d, int m, int n, std::complex<float> alpha,
     const std::complex<float>* a, int lda, std::complex<float>* b, int ldb) {
      STRUMPACK_FC_GLOBAL(ctrmm,CTRMM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(4*trmm_flops(m,n,alpha,s));
      STRUMPACK_BYTES(2*4*trmm_moves(m,n));
    }
    void trmm
    (char s, char ul, char t, char d, int m, int n,
     std::complex<double> alpha, const std::complex<double>* a, int lda,
     std::complex<double>* b, int ldb) {
      STRUMPACK_FC_GLOBAL(ztrmm,ZTRMM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(4*trmm_flops(m,n,alpha,s));
      STRUMPACK_BYTES(2*8*trmm_moves(m,n));
    }


    void trmv
    (char ul, char t, char d, int n, const float* a, int lda,
     float* x, int incx) {
      STRUMPACK_FC_GLOBAL(strmv,STRMV)(&ul, &t, &d, &n, a, &lda, x, &incx);
      STRUMPACK_FLOPS(trmv_flops(n));
      STRUMPACK_BYTES(4*trmv_moves(n));
    }
    void trmv
    (char ul, char t, char d, int n, const double* a, int lda,
     double* x, int incx) {
      STRUMPACK_FC_GLOBAL(dtrmv,DTRMV)(&ul, &t, &d, &n, a, &lda, x, &incx);
      STRUMPACK_FLOPS(trmv_flops(n));
      STRUMPACK_BYTES(8*trmv_moves(n));
    }
    void trmv
    (char ul, char t, char d, int n, const std::complex<float>* a, int lda,
     std::complex<float>* x, int incx) {
      STRUMPACK_FC_GLOBAL(ctrmv,CTRMV)(&ul, &t, &d, &n, a, &lda, x, &incx);
      STRUMPACK_FLOPS(4*trmv_flops(n));
      STRUMPACK_BYTES(2*4*trmv_moves(n));
    }
    void trmv
    (char ul, char t, char d, int n, const std::complex<double>* a, int lda,
     std::complex<double>* x, int incx) {
      STRUMPACK_FC_GLOBAL(ztrmv,ZTRMV)(&ul, &t, &d, &n, a, &lda, x, &incx);
      STRUMPACK_FLOPS(4*trmv_flops(n));
      STRUMPACK_BYTES(2*8*trmv_moves(n));
    }


    void trsv
    (char ul, char t, char d, int m, const float* a, int lda,
     float* b, int incb) {
      STRUMPACK_FC_GLOBAL(strsv,STRSV)(&ul, &t, &d, &m, a, &lda, b, &incb);
      STRUMPACK_FLOPS(trsv_flops(m));
      STRUMPACK_BYTES(4*trsv_moves(m));
    }
    void trsv
    (char ul, char t, char d, int m, const double* a, int lda,
     double* b, int incb) {
      STRUMPACK_FC_GLOBAL(dtrsv,DTRSV)(&ul, &t, &d, &m, a, &lda, b, &incb);
      STRUMPACK_FLOPS(trsv_flops(m));
      STRUMPACK_BYTES(8*trsv_moves(m));
    }
    void trsv
    (char ul, char t, char d, int m, const std::complex<float>* a, int lda,
     std::complex<float>* b, int incb) {
      STRUMPACK_FC_GLOBAL(ctrsv,CTRSV)(&ul, &t, &d, &m, a, &lda, b, &incb);
      STRUMPACK_FLOPS(4*trsv_flops(m));
      STRUMPACK_BYTES(2*4*trsv_moves(m));
    }
    void trsv
    (char ul, char t, char d, int m, const std::complex<double>* a, int lda,
     std::complex<double>* b, int incb) {
      STRUMPACK_FC_GLOBAL(ztrsv,ZTRSV)(&ul, &t, &d, &m, a, &lda, b, &incb);
      STRUMPACK_FLOPS(4*trsv_flops(m));
      STRUMPACK_BYTES(2*8*trsv_moves(m));
    }


    void laset
    (char s, int m, int n, float alpha, float beta, float* x, int ldx) {
      STRUMPACK_FC_GLOBAL(slaset,SLASET)(&s, &m, &n, &alpha, &beta, x, &ldx);
    }
    void laset
    (char s, int m, int n, double alpha, double beta, double* x, int ldx) {
      STRUMPACK_FC_GLOBAL(dlaset,DLASET)(&s, &m, &n, &alpha, &beta, x, &ldx);
    }
    void laset
    (char s, int m, int n, std::complex<float> alpha,
     std::complex<float> beta, std::complex<float>* x, int ldx) {
      STRUMPACK_FC_GLOBAL(claset,CLASET)(&s, &m, &n, &alpha, &beta, x, &ldx);
    }
    void laset
    (char s, int m, int n, std::complex<double> alpha,
     std::complex<double> beta, std::complex<double>* x, int ldx) {
      STRUMPACK_FC_GLOBAL(zlaset,ZLASET)(&s, &m, &n, &alpha, &beta, x, &ldx);
    }

    int geqp3
    (int m, int n, float* a, int lda, int* jpvt, float* tau,
     float* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(sgeqp3,SGEQP3)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(geqp3_flops(m,n));
      return info;
    }
    int geqp3
    (int m, int n, double* a, int lda, int* jpvt, double* tau,
     double* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(dgeqp3,DGEQP3)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(geqp3_flops(m,n));
      return info;
    }
    int geqp3
    (int m, int n, std::complex<float>* a, int lda, int* jpvt,
     std::complex<float>* tau, std::complex<float>* work, int lwork) {
      int info;
      std::unique_ptr<std::complex<float>[]> rwork
        (new std::complex<float>[std::max(1, 2*n)]);
      STRUMPACK_FC_GLOBAL(cgeqp3,CGEQP3)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork.get(), &info);
      STRUMPACK_FLOPS(4*geqp3_flops(m,n));
      return info;
    }
    int geqp3
    (int m, int n, std::complex<double>* a, int lda, int* jpvt,
     std::complex<double>* tau, std::complex<double>* work,
     int lwork) {
      int info;
      std::unique_ptr<std::complex<double>[]> rwork
        (new std::complex<double>[std::max(1, 2*n)]);
      STRUMPACK_FC_GLOBAL(zgeqp3,ZGEQP3)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork.get(), &info);
      STRUMPACK_FLOPS(4*geqp3_flops(m,n));
      return info;
    }


    int geqp3tol
    (int m, int n, float* a, int lda, int* jpvt, float* tau, float* work,
     int lwork, int& rank, float rtol, float atol, int depth) {
      int info;
      STRUMPACK_FC_GLOBAL(sgeqp3tol,SGEQP3TOL)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, &info,
         &rank, &rtol, &atol, &depth);
      return info;
    }
    int geqp3tol
    (int m, int n, double* a, int lda, int* jpvt, double* tau, double* work,
     int lwork, int& rank, double rtol, double atol, int depth) {
      int info;
      STRUMPACK_FC_GLOBAL(dgeqp3tol,DGEQP3TOL)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, &info,
         &rank, &rtol, &atol, &depth);
      return info;
    }
    int geqp3tol
    (int m, int n, std::complex<float>* a, int lda, int* jpvt,
     std::complex<float>* tau, std::complex<float>* work, int lwork,
     int& rank, float rtol, float atol, int depth) {
      std::unique_ptr<float[]> rwork(new float[std::max(1, 2*n)]);
      bool tasked = depth < params::task_recursion_cutoff_level;
      if (tasked) {
        int loop_tasks = std::max(params::num_threads / (depth+1), 1);
        int B = std::max(n / loop_tasks, 1);
        for (int task=0; task<std::ceil(n/float(B)); task++) {
#pragma omp task default(shared) firstprivate(task)
          for (int i=task*B; i<std::min((task+1)*B,n); i++)
            rwork[i] = nrm2(m, &a[i*lda], 1);
        }
#pragma omp taskwait
      } else
        for (int i=0; i<n; i++)
          rwork[i] = nrm2(m, &a[i*lda], 1);
      int info;
      STRUMPACK_FC_GLOBAL(cgeqp3tol,CGEQP3TOL)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork.get(), &info,
         &rank, &rtol, &atol, &depth);
      return info;
    }
    int geqp3tol
    (int m, int n, std::complex<double>* a, int lda, int* jpvt,
     std::complex<double>* tau, std::complex<double>* work, int lwork,
     int& rank, double rtol, double atol, int depth) {
      std::unique_ptr<double[]> rwork(new double[std::max(1, 2*n)]);
      bool tasked = depth < params::task_recursion_cutoff_level;
      if (tasked) {
        int loop_tasks = std::max(params::num_threads / (depth+1), 1);
        int B = std::max(n / loop_tasks, 1);
        for (int task=0; task<std::ceil(n/float(B)); task++) {
#pragma omp task default(shared) firstprivate(task)
          for (int i=task*B; i<std::min((task+1)*B,n); i++)
            rwork[i] = nrm2(m, &a[i*lda], 1);
        }
#pragma omp taskwait
      } else
        for (int i=0; i<n; i++)
          rwork[i] = nrm2(m, &a[i*lda], 1);
      int info;
      STRUMPACK_FC_GLOBAL(zgeqp3tol,ZGEQP3TOL)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork.get(), &info,
         &rank, &rtol, &atol, &depth);
      return info;
    }



    int geqrf
    (int m, int n, float* a, int lda, float* tau, float* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(sgeqrf,SGEQRF)(&m, &n, a, &lda, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(geqrf_flops(m,n));
      return info;
    }
    int geqrf
    (int m, int n, double* a, int lda, double* tau, double* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(dgeqrf,DGEQRF)(&m, &n, a, &lda, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(geqrf_flops(m,n));
      return info;
    }
    int geqrf
    (int m, int n, std::complex<float>* a, int lda, std::complex<float>* tau,
     std::complex<float>* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(cgeqrf,CGEQRF)(&m, &n, a, &lda, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(4*geqrf_flops(m,n));
      return info;
    }
    int geqrf
    (int m, int n, std::complex<double>* a, int lda,
     std::complex<double>* tau, std::complex<double>* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(zgeqrf,ZGEQRF)(&m, &n, a, &lda, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(4*geqrf_flops(m,n));
      return info;
    }


    int geqrfmod
    (int m, int n, float* a, int lda,
     float* tau, float* work, int lwork, int depth) {
      int info;
      STRUMPACK_FC_GLOBAL(sgeqrfmod,SGEQRFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, &info, &depth);
      return info;
    }
    int geqrfmod
    (int m, int n, double* a, int lda, double* tau,
     double* work, int lwork, int depth) {
      int info;
      STRUMPACK_FC_GLOBAL(dgeqrfmod,DGEQRFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, &info, &depth);
      return info;
    }
    int geqrfmod
    (int m, int n, std::complex<float>* a, int lda, std::complex<float>* tau,
     std::complex<float>* work, int lwork, int depth) {
      int info;
      STRUMPACK_FC_GLOBAL(cgeqrfmod,CGEQRFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, &info, &depth);
      return info;
    }
    int geqrfmod
    (int m, int n, std::complex<double>* a, int lda,
     std::complex<double>* tau, std::complex<double>* work,
     int lwork, int depth) {
      int info;
      STRUMPACK_FC_GLOBAL(zgeqrfmod,ZGEQRFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, &info, &depth);
      return info;
    }


    int gelqf
    (int m, int n, float* a, int lda, float* tau, float* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(sgelqf,SGELQF)(&m, &n, a, &lda, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(gelqf_flops(m,n));
      return info;
    }
    int gelqf
    (int m, int n, double* a, int lda, double* tau, double* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(dgelqf,DGELQF)(&m, &n, a, &lda, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(gelqf_flops(m,n));
      return info;
    }
    int gelqf
    (int m, int n, std::complex<float>* a, int lda, std::complex<float>* tau,
     std::complex<float>* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(cgelqf,CGELQF)(&m, &n, a, &lda, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(4*gelqf_flops(m,n));
      return info;
    }
    int gelqf
    (int m, int n, std::complex<double>* a, int lda,
     std::complex<double>* tau, std::complex<double>* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(zgelqf,ZGELQF)(&m, &n, a, &lda, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(4*gelqf_flops(m,n));
      return info;
    }



    void gelqfmod
    (int m, int n, float* a, int lda, float* tau,
     float* work, int lwork, int* info, int depth) {
      STRUMPACK_FC_GLOBAL(sgelqfmod,SGELQFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, info, &depth);
    }
    void gelqfmod
    (int m, int n, double* a, int lda, double* tau,
     double* work, int lwork, int* info, int depth) {
      STRUMPACK_FC_GLOBAL(dgelqfmod,DGELQFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, info, &depth);
    }
    void gelqfmod
    (int m, int n, std::complex<float>* a, int lda, std::complex<float>* tau,
     std::complex<float>* work, int lwork, int* info, int depth) {
      STRUMPACK_FC_GLOBAL(cgelqfmod,CGELQFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, info, &depth);
    }
    void gelqfmod
    (int m, int n, std::complex<double>* a, int lda,
     std::complex<double>* tau, std::complex<double>* work,
     int lwork, int* info, int depth) {
      STRUMPACK_FC_GLOBAL(zgelqfmod,ZGELQFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, info, &depth);
    }


    void getrf
    (int m, int n, float* a, int lda, int* ipiv, int* info) {
      STRUMPACK_FC_GLOBAL(sgetrf,SGETRF)(&m, &n, a, &lda, ipiv, info);
      STRUMPACK_FLOPS(getrf_flops(m,n));
    }
    void getrf
    (int m, int n, double* a, int lda, int* ipiv, int* info) {
      STRUMPACK_FC_GLOBAL(dgetrf,DGETRF)(&m, &n, a, &lda, ipiv, info);
      STRUMPACK_FLOPS(getrf_flops(m,n));
    }
    void getrf
    (int m, int n, std::complex<float>* a, int lda, int* ipiv, int* info) {
      STRUMPACK_FC_GLOBAL(cgetrf,CGETRF)(&m, &n, a, &lda, ipiv, info);
      STRUMPACK_FLOPS(4*getrf_flops(m,n));
    }
    void getrf
    (int m, int n, std::complex<double>* a, int lda, int* ipiv, int* info) {
      STRUMPACK_FC_GLOBAL(zgetrf,ZGETRF)(&m, &n, a, &lda, ipiv, info);
      STRUMPACK_FLOPS(4*getrf_flops(m,n));
    }


    void getrfmod
    (int m, int n, float* a, int lda, int* ipiv, int* info, int depth) {
      STRUMPACK_FC_GLOBAL(sgetrfmod,SGETRFMOD)(&m, &n, a, &lda, ipiv, info, &depth);
    }
    void getrfmod
    (int m, int n, double* a, int lda, int* ipiv, int* info, int depth) {
      STRUMPACK_FC_GLOBAL(dgetrfmod,DGETRFMOD)(&m, &n, a, &lda, ipiv, info, &depth);
    }
    void getrfmod
    (int m, int n, std::complex<float>* a, int lda,
     int* ipiv, int* info, int depth) {
      STRUMPACK_FC_GLOBAL(cgetrfmod,CGETRFMOD)(&m, &n, a, &lda, ipiv, info, &depth);
    }
    void getrfmod
    (int m, int n, std::complex<double>* a, int lda,
     int* ipiv, int* info, int depth) {
      STRUMPACK_FC_GLOBAL(zgetrfmod,ZGETRFMOD)(&m, &n, a, &lda, ipiv, info, &depth);
    }


    void getrs
    (char t, int n, int nrhs, const float* a, int lda,
     const int* ipiv, float* b, int ldb, int* info) {
      STRUMPACK_FC_GLOBAL(sgetrs,SGETRS)(&t, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
      STRUMPACK_FLOPS(getrs_flops(n, nrhs));
    }
    void getrs
    (char t, int n, int nrhs, const double* a, int lda,
     const int* ipiv, double* b, int ldb, int* info) {
      STRUMPACK_FC_GLOBAL(dgetrs,DGETRS)(&t, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
      STRUMPACK_FLOPS(getrs_flops(n, nrhs));
    }
    void getrs
    (char t, int n, int nrhs, const std::complex<float>* a, int lda,
     const int* ipiv, std::complex<float>* b, int ldb, int* info) {
      STRUMPACK_FC_GLOBAL(cgetrs,CGETRS)(&t, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
      STRUMPACK_FLOPS(4*getrs_flops(n, nrhs));
    }
    void getrs
    (char t, int n, int nrhs, const std::complex<double>* a, int lda,
     const int* ipiv, std::complex<double>* b, int ldb, int* info) {
      zgetrs_(&t, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
      STRUMPACK_FLOPS(4*getrs_flops(n, nrhs));
    }


    int potrf
    (char ul, int n, float* a, int lda) {
      int info;
      STRUMPACK_FC_GLOBAL(spotrf,SPOTRF)(&ul, &n, a, &lda, &info);
      STRUMPACK_FLOPS(potrf_flops(n));
      return info;
    }
    int potrf
    (char ul, int n, double* a, int lda) {
      int info;
      STRUMPACK_FC_GLOBAL(dpotrf,DPOTRF)(&ul, &n, a, &lda, &info);
      STRUMPACK_FLOPS(potrf_flops(n));
      return info;
    }
    int potrf
    (char ul, int n, std::complex<float>* a, int lda) {
      int info;
      STRUMPACK_FC_GLOBAL(cpotrf,CPOTRF)(&ul, &n, a, &lda, &info);
      STRUMPACK_FLOPS(4*potrf_flops(n));
      return info;
    }
    int potrf
    (char ul, int n, std::complex<double>* a, int lda) {
      int info;
      STRUMPACK_FC_GLOBAL(zpotrf,ZPOTRF)(&ul, &n, a, &lda, &info);
      STRUMPACK_FLOPS(4*potrf_flops(n));
      return info;
    }


    void xxglq
    (int m, int n, int k, float* a, int lda, const float* tau,
     float* work, int lwork, int* info) {
      STRUMPACK_FC_GLOBAL(sorglq,STRUMPACK_FC_GLOBAL)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(xxglq_flops(m,n,k));
    }
    void xxglq
    (int m, int n, int k, double* a, int lda, const double* tau,
     double* work, int lwork, int* info) {
      STRUMPACK_FC_GLOBAL(dorglq,DORGLQ)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(xxglq_flops(m,n,k));
    }
    void xxglq
    (int m, int n, int k, std::complex<float>* a, int lda,
     const std::complex<float>* tau, std::complex<float>* work,
     int lwork, int* info) {
      STRUMPACK_FC_GLOBAL(cunglq,CUNGLQ)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(4*xxglq_flops(m,n,k));
    }
    void xxglq
    (int m, int n, int k, std::complex<double>* a, int lda,
     const std::complex<double>* tau, std::complex<double>* work,
     int lwork, int* info) {
      STRUMPACK_FC_GLOBAL(zunglq,ZUNGLQ)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(4*xxglq_flops(m,n,k));
    }

    // do not count flops here, they are counted in the blas routines
    void xxglqmod
    (int m, int n, int k, float* a, int lda, const float* tau,
     float* work, int lwork, int* info, int depth) {
      STRUMPACK_FC_GLOBAL(sorglqmod,SORGLQMOD)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info, &depth);
    }
    void xxglqmod
    (int m, int n, int k, double* a, int lda, const double* tau,
     double* work, int lwork, int* info, int depth) {
      STRUMPACK_FC_GLOBAL(dorglqmod,DORGLQMOD)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info, &depth);
    }
    void xxglqmod
    (int m, int n, int k, std::complex<float>* a, int lda,
     const std::complex<float>* tau, std::complex<float>* work,
     int lwork, int* info, int depth) {
      STRUMPACK_FC_GLOBAL(cunglqmod,CUNGLQMOD)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info, &depth);
    }
    void xxglqmod
    (int m, int n, int k, std::complex<double>* a, int lda,
     const std::complex<double>* tau, std::complex<double>* work,
     int lwork, int* info, int depth) {
      STRUMPACK_FC_GLOBAL(zunglqmod,ZUNGLQMOD)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info, &depth);
    }


    int xxgqr
    (int m, int n, int k, float* a, int lda, const float* tau,
     float* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(sorgqr,SORGQR)
        (&m, &n, &k, a, &lda, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(xxgqr_flops(m,n,k));
      return info;
    }
    int xxgqr
    (int m, int n, int k, double* a, int lda, const double* tau,
     double* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(dorgqr,DORGQR)
        (&m, &n, &k, a, &lda, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(xxgqr_flops(m,n,k));
      return info;
    }
    int xxgqr
    (int m, int n, int k, std::complex<float>* a, int lda,
     const std::complex<float>* tau, std::complex<float>* work,
     int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(cungqr,CUNGQR)
        (&m, &n, &k, a, &lda, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(4*xxgqr_flops(m,n,k));
      return info;
    }
    int xxgqr
    (int m, int n, int k, std::complex<double>* a, int lda,
     const std::complex<double>* tau, std::complex<double>* work,
     int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(zungqr,ZUNGQR)
        (&m, &n, &k, a, &lda, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(4*xxgqr_flops(m,n,k));
      return info;
    }


    int xxmqr
    (char side, char trans, int m, int n, int k, float* a, int lda,
     const float* tau, float* c, int ldc, float* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(sormqr,SORMQR)
        (&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
      STRUMPACK_FLOPS(xxmqr_flops(m,n,k));
      return info;
    }
    int xxmqr
    (char side, char trans, int m, int n, int k, double* a, int lda,
     const double* tau, double* c, int ldc, double* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(dormqr,DORMQR)
        (&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
      STRUMPACK_FLOPS(xxmqr_flops(m,n,k));
      return info;
    }
    int xxmqr
    (char side, char trans, int m, int n, int k, std::complex<float>* a, int lda,
     const std::complex<float>* tau, std::complex<float>* c, int ldc,
     std::complex<float>* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(cunmqr,CUNMQR)
        (&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
      STRUMPACK_FLOPS(4*xxmqr_flops(m,n,k));
      return info;
    }
    int xxmqr
    (char side, char trans, int m, int n, int k, std::complex<double>* a, int lda,
     const std::complex<double>* tau, std::complex<double>* c, int ldc,
     std::complex<double>* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(zunmqr,ZUNMQR)
        (&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
      STRUMPACK_FLOPS(4*xxmqr_flops(m,n,k));
      return info;
    }


    int lange(char norm, int m, int n, const int *a, int lda) { return -1; }
    unsigned int lange(char norm, int m, int n, const unsigned int *a, int lda) { return 0; }
    std::size_t lange(char norm, int m, int n, const std::size_t *a, int lda) { return 0; }
    bool lange(char norm, int m, int n, const bool *a, int lda) { return false; }
    float lange(char norm, int m, int n, const float *a, int lda) {
      if (norm == 'I' || norm == 'i') {
        std::unique_ptr<float[]> work(new float[m]);
        return STRUMPACK_FC_GLOBAL(slange,SLANGE)(&norm, &m, &n, a, &lda, work.get());
      } else return STRUMPACK_FC_GLOBAL(slange,SLANGE)(&norm, &m, &n, a, &lda, nullptr);
    }
    double lange(char norm, int m, int n, const double *a, int lda) {
      if (norm == 'I' || norm == 'i') {
        std::unique_ptr<double[]> work(new double[m]);
        return STRUMPACK_FC_GLOBAL(dlange,DLANGE)(&norm, &m, &n, a, &lda, work.get());
      } else return STRUMPACK_FC_GLOBAL(dlange,DLANGE)(&norm, &m, &n, a, &lda, nullptr);
    }
    float lange
      (char norm, int m, int n, const std::complex<float> *a, int lda) {
      if (norm == 'I' || norm == 'i') {
        std::unique_ptr<float[]> work(new float[m]);
        return STRUMPACK_FC_GLOBAL(clange,CLANGE)(&norm, &m, &n, a, &lda, work.get());
      } else return STRUMPACK_FC_GLOBAL(clange,CLANGE)(&norm, &m, &n, a, &lda, nullptr);
    }
    double lange
      (char norm, int m, int n, const std::complex<double> *a, int lda) {
      if (norm == 'I' || norm == 'i') {
        std::unique_ptr<double[]> work(new double[m]);
        return STRUMPACK_FC_GLOBAL(zlange,ZLANGE)(&norm, &m, &n, a, &lda, work.get());
      } else return STRUMPACK_FC_GLOBAL(zlange,ZLANGE)(&norm, &m, &n, a, &lda, nullptr);
    }


    int gesvd
    (char jobu, char jobvt, int m, int n, float* a, int lda,
     float* s, float* u, int ldu, float* vt, int ldvt) {
      int info;
      int lwork = -1;
      float swork;
      STRUMPACK_FC_GLOBAL(sgesvd,SGESVD)
        (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
         &swork, &lwork, &info);
      lwork = int(swork);
      std::unique_ptr<float[]> work(new float[lwork]);
      STRUMPACK_FC_GLOBAL(sgesvd,SGESVD)
        (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
         work.get(), &lwork, &info);
      return info;
    }
    int gesvd
      (char jobu, char jobvt, int m, int n, double* a, int lda,
       double* s, double* u, int ldu, double* vt, int ldvt) {
      int info;
      int lwork = -1;
      double dwork;
      STRUMPACK_FC_GLOBAL(dgesvd,DGESVD)
        (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
         &dwork, &lwork, &info);
      lwork = int(dwork);
      std::unique_ptr<double[]> work(new double[lwork]);
      STRUMPACK_FC_GLOBAL(dgesvd,DGESVD)
        (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
         work.get(), &lwork, &info);
      return info;
    }
    int gesvd
      (char jobu, char jobvt, int m, int n, std::complex<float>* a, int lda,
       std::complex<float>* s, std::complex<float>* u, int ldu,
       std::complex<float>* vt, int ldvt) {
      std::cout << "TODO gesvd for std::complex<float>" << std::endl;
      return 0;
    }
    int gesvd
      (char jobu, char jobvt, int m, int n, std::complex<double>* a, int lda,
       std::complex<double>* s, std::complex<double>* u, int ldu,
       std::complex<double>* vt, int ldvt) {
      std::cout << "TODO gesvd for std::complex<double>" << std::endl;
      return 0;
    }

    int syevx
    (char jobz, char range, char uplo, int n, float* a, int lda,
     float vl, float vu, int il, int iu, float abstol, int& m,
     float* w, float* z, int ldz) {
      int info;
      std::unique_ptr<int[]> iwork(new int[5*n+n]);
      auto ifail = iwork.get()+5*n;
      int lwork = -1;
      float swork;
      STRUMPACK_FC_GLOBAL(ssyevx,SSYEVX)
        (&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m,
         w, z, &ldz, &swork, &lwork, iwork.get(), ifail, &info);
      lwork = int(swork);
      std::unique_ptr<float[]> work(new float[lwork]);
      STRUMPACK_FC_GLOBAL(ssyevx,SSYEVX)
        (&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m,
         w, z, &ldz, work.get(), &lwork, iwork.get(), ifail, &info);
      return info;
    }
    int syevx
    (char jobz, char range, char uplo, int n, double* a, int lda,
     double vl, double vu, int il, int iu, double abstol, int& m,
     double* w, double* z, int ldz) {
      int info;
      std::unique_ptr<int[]> iwork(new int[5*n+n]);
      auto ifail = iwork.get()+5*n;
      int lwork = -1;
      double dwork;
      STRUMPACK_FC_GLOBAL(dsyevx,DSYEVX)
        (&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m,
         w, z, &ldz, &dwork, &lwork, iwork.get(), ifail, &info);
      lwork = int(dwork);
      std::unique_ptr<double[]> work(new double[lwork]);
      STRUMPACK_FC_GLOBAL(dsyevx,DSYEVX)
        (&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m,
         w, z, &ldz, work.get(), &lwork, iwork.get(), ifail, &info);
      return info;
    }

    int syev(char jobz, char uplo, int n, float* a, int lda, float* w) {
      int info;
      int lwork = -1;
      float swork;
      STRUMPACK_FC_GLOBAL(ssyev,SSYEV)
        (&jobz, &uplo, &n, a, &lda, w, &swork, &lwork, &info);
      lwork = int(swork);
      std::unique_ptr<float[]> work(new float[lwork]);
      STRUMPACK_FC_GLOBAL(ssyev,SSYEV)
        (&jobz, &uplo, &n, a, &lda, w, work.get(), &lwork, &info);
      return info;
    }
    int syev(char jobz, char uplo, int n, double* a, int lda, double* w) {
      int info;
      int lwork = -1;
      double swork;
      STRUMPACK_FC_GLOBAL(dsyev,DSYEV)
        (&jobz, &uplo, &n, a, &lda, w, &swork, &lwork, &info);
      lwork = int(swork);
      std::unique_ptr<double[]> work(new double[lwork]);
      STRUMPACK_FC_GLOBAL(dsyev,DSYEV)
        (&jobz, &uplo, &n, a, &lda, w, work.get(), &lwork, &info);
      return info;
    }
    int syev(char jobz, char uplo, int n, std::complex<float>* a, int lda,
             std::complex<float>* w) {
      std::cerr << "ERROR: STRUMPACK does not implement syev for complex"
                << std::endl;;
      assert(false);
      return 0;
    }
    int syev(char jobz, char uplo, int n, std::complex<double>* a, int lda,
             std::complex<double>* w) {
      std::cerr << "ERROR: STRUMPACK does not implement syev for complex"
                << std::endl;;
      assert(false);
      return 0;
    }

    int sytrf
    (char s, int n, float* a, int lda, int* ipiv, float* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(ssytrf,SSYTRF)
        (&s, &n, a, &lda, ipiv, work, &lwork, &info);
      STRUMPACK_FLOPS(sytrf_flops(n));
      return info;
    }
    int sytrf
    (char s, int n, double* a, int lda, int* ipiv, double* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(dsytrf,DSYTRF)
        (&s, &n, a, &lda, ipiv, work, &lwork, &info);
      STRUMPACK_FLOPS(sytrf_flops(n));
      return info;
    }
    int sytrf
    (char s, int n, std::complex<float>* a, int lda, int* ipiv,
     std::complex<float>* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(csytrf,CSYTRF)
        (&s, &n, a, &lda, ipiv, work, &lwork, &info);
      STRUMPACK_FLOPS(4*sytrf_flops(n));
      return info;
    }
    int sytrf
    (char s, int n, std::complex<double>* a, int lda, int* ipiv,
     std::complex<double>* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(zsytrf,ZSYTRF)
        (&s, &n, a, &lda, ipiv, work, &lwork, &info);
      STRUMPACK_FLOPS(4*sytrf_flops(n));
      return info;
    }

    // int sytrf_rook
    // (char s, int n, float* a, int lda, int* ipiv, float* work, int lwork) {
    //   int info;
    //   STRUMPACK_FC_GLOBAL_(ssytrf_rook,SSYTRF_ROOK)
    //     (&s, &n, a, &lda, ipiv, work, &lwork, &info);
    //   STRUMPACK_FLOPS(sytrf_rook_flops(n));
    //   return info;
    // }
    // int sytrf_rook
    // (char s, int n, double* a, int lda, int* ipiv, double* work, int lwork) {
    //   int info;
    //   STRUMPACK_FC_GLOBAL_(dsytrf_rook,DSYTRF_ROOK)
    //     (&s, &n, a, &lda, ipiv, work, &lwork, &info);
    //   STRUMPACK_FLOPS(sytrf_rook_flops(n));
    //   return info;
    // }
    // int sytrf_rook
    // (char s, int n, std::complex<float>* a, int lda, int* ipiv,
    //  std::complex<float>* work, int lwork) {
    //   int info;
    //   STRUMPACK_FC_GLOBAL_(csytrf_rook,CSYTRF_ROOK)
    //     (&s, &n, a, &lda, ipiv, work, &lwork, &info);
    //   STRUMPACK_FLOPS(4*sytrf_rook_flops(n));
    //   return info;
    // }
    // int sytrf_rook
    // (char s, int n, std::complex<double>* a, int lda, int* ipiv,
    //  std::complex<double>* work, int lwork){
    //   int info;
    //   STRUMPACK_FC_GLOBAL_(zsytrf_rook,ZSYTRF_ROOK)
    //     (&s, &n, a, &lda, ipiv, work, &lwork, &info);
    //   STRUMPACK_FLOPS(4*sytrf_rook_flops(n));
    //   return info;
    // }


    int sytrs
    (char s, int n, int nrhs, const float* a, int lda,
     const int* ipiv, float* b, int ldb) {
      int info;
      STRUMPACK_FC_GLOBAL(ssytrs,SSYTRS)
        (&s, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      STRUMPACK_FLOPS(sytrs_flops(n,n,nrhs));
      return info;
    }
    int sytrs
    (char s, int n, int nrhs, const double* a, int lda,
     const int* ipiv, double* b, int ldb) {
      int info;
      STRUMPACK_FC_GLOBAL(dsytrs,DSYTRS)
        (&s, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      STRUMPACK_FLOPS(sytrs_flops(n,n,nrhs));
      return info;
    }
    int sytrs
    (char s, int n, int nrhs, const std::complex<float>* a, int lda,
     const int* ipiv, std::complex<float>* b, int ldb) {
      int info;
      STRUMPACK_FC_GLOBAL(csytrs,CSYTRS)
        (&s, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      STRUMPACK_FLOPS(4*sytrs_flops(n,n,nrhs));
      return info;
    }
    int sytrs
    (char s, int n, int nrhs, const std::complex<double>* a, int lda,
     const int* ipiv, std::complex<double>* b, int ldb) {
      int info;
      STRUMPACK_FC_GLOBAL(zsytrs,ZSYTRS)
        (&s, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      STRUMPACK_FLOPS(4*sytrs_flops(n,n,nrhs));
      return info;
    }

    // int sytrs_rook
    // (char s, int n, int nrhs, const float* a, int lda,
    //  const int* ipiv, float* b, int ldb) {
    //   int info;
    //   STRUMPACK_FC_GLOBAL_(ssytrs_rook,SSYTRS_ROOK)
    //     (&s, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    //   STRUMPACK_FLOPS(sytrs_rook_flops(n,n,nrhs));
    //   return info;
    // }
    // int sytrs_rook
    // (char s, int n, int nrhs, const double* a, int lda,
    //  const int* ipiv, double* b, int ldb) {
    //   int info;
    //   STRUMPACK_FC_GLOBAL_(dsytrs_rook,DSYTRS_ROOK)
    //     (&s, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    //   STRUMPACK_FLOPS(sytrs_rook_flops(n,n,nrhs));
    //   return info;
    // }
    // int sytrs_rook
    // (char s, int n, int nrhs, const std::complex<float>* a, int lda,
    //  const int* ipiv, std::complex<float>* b, int ldb) {
    //   int info;
    //   STRUMPACK_FC_GLOBAL_(csytrs_rook,CSYTRS_ROOK)
    //     (&s, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    //   STRUMPACK_FLOPS(4*sytrs_rook_flops(n,n,nrhs));
    //   return info;
    // }
    // int sytrs_rook
    // (char s, int n, int nrhs, const std::complex<double>* a, int lda,
    //  const int* ipiv, std::complex<double>* b, int ldb) {
    //   int info;
    //   STRUMPACK_FC_GLOBAL_(zsytrs_rook,ZSYTRS_ROOK)
    //     (&s, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    //   STRUMPACK_FLOPS(4*sytrs_rook_flops(n,n,nrhs));
    //   return info;
    // }

  } //end namespace blas
} // end namespace strumpack
