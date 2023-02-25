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
#include <vector>

#include "BLASLAPACKWrapper.hpp"
#include "StrumpackFortranCInterface.h"

namespace strumpack {
  namespace blas {

    ///////////////////////////////////////////////////////////
    ///////// BLAS1 ///////////////////////////////////////////
    ///////////////////////////////////////////////////////////
    extern "C" {
      void STRUMPACK_FC_GLOBAL(scopy,SCOPY)
        (strumpack_blas_int* n, const float* x, strumpack_blas_int* incx, float* y, strumpack_blas_int* incy);
      void STRUMPACK_FC_GLOBAL(dcopy,DCOPY)
        (strumpack_blas_int* n, const double* x, strumpack_blas_int* incx, double* y, strumpack_blas_int* incy);
      void STRUMPACK_FC_GLOBAL(ccopy,CCOPY)
        (strumpack_blas_int* n, const std::complex<float>* x, strumpack_blas_int* incx,
         std::complex<float>* y, strumpack_blas_int* incy);
      void STRUMPACK_FC_GLOBAL(zcopy,ZCOPY)
        (strumpack_blas_int* n, const std::complex<double>* x, strumpack_blas_int* incx,
         std::complex<double>* y, strumpack_blas_int* incy);

      void STRUMPACK_FC_GLOBAL(sscal,SSCAL)
        (strumpack_blas_int* n, float* alpha, float* x, strumpack_blas_int* incx);
      void STRUMPACK_FC_GLOBAL(dscal,DSCAL)
        (strumpack_blas_int* n, double* alpha, double* x, strumpack_blas_int* incx);
      void STRUMPACK_FC_GLOBAL(cscal,CSCAL)
        (strumpack_blas_int* n, std::complex<float>* alpha,
         std::complex<float>* x, strumpack_blas_int* incx);
      void STRUMPACK_FC_GLOBAL(zscal,ZSCAL)
        (strumpack_blas_int* n, std::complex<double>* alpha,
         std::complex<double>* x, strumpack_blas_int* incx);

      strumpack_blas_int STRUMPACK_FC_GLOBAL(isamax,ISAMAX)
        (strumpack_blas_int* n, const float* dx, strumpack_blas_int* incx);
      strumpack_blas_int STRUMPACK_FC_GLOBAL(idamax,IDAMAX)
        (strumpack_blas_int* n, const double* dx, strumpack_blas_int* incx);
      strumpack_blas_int STRUMPACK_FC_GLOBAL(icamax,ICAMAX)
        (strumpack_blas_int* n, const std::complex<float>* dx, strumpack_blas_int* incx);
      strumpack_blas_int STRUMPACK_FC_GLOBAL(izamax,IZAMAX)
        (strumpack_blas_int* n, const std::complex<double>* dx, strumpack_blas_int* incx);

      float STRUMPACK_FC_GLOBAL(snrm2,SNRM2)
        (strumpack_blas_int* n, const float* x, strumpack_blas_int* incx);
      double STRUMPACK_FC_GLOBAL(dnrm2,DNRM2)
        (strumpack_blas_int* n, const double* x, strumpack_blas_int* incx);
      float STRUMPACK_FC_GLOBAL(scnrm2,SCNRM2)
        (strumpack_blas_int* n, const std::complex<float>* x, strumpack_blas_int* incx);
      double STRUMPACK_FC_GLOBAL(dznrm2,DZNRM2)
        (strumpack_blas_int* n, const std::complex<double>* x, strumpack_blas_int* incx);

      void STRUMPACK_FC_GLOBAL(saxpy,SAXPY)
        (strumpack_blas_int* n, float* alpha, const float* x, strumpack_blas_int* incx,
         float* y, strumpack_blas_int* incy);
      void STRUMPACK_FC_GLOBAL(daxpy,DAXPY)
        (strumpack_blas_int* n, double* alpha, const double* x, strumpack_blas_int* incx,
         double* y, strumpack_blas_int* incy);
      void STRUMPACK_FC_GLOBAL(caxpy,CAXPY)
        (strumpack_blas_int* n, std::complex<float>* alpha,
         const std::complex<float>* x, strumpack_blas_int* incx,
         std::complex<float>* y, strumpack_blas_int* incy);
      void STRUMPACK_FC_GLOBAL(zaxpy,ZAXPY)
        (strumpack_blas_int* n, std::complex<double>* alpha,
         const std::complex<double>* x, strumpack_blas_int* incx,
         std::complex<double>* y, strumpack_blas_int* incy);

      void STRUMPACK_FC_GLOBAL(sswap,SSWAP)
        (strumpack_blas_int* n, float* x, strumpack_blas_int* ldx, float* y, strumpack_blas_int* ldy);
      void STRUMPACK_FC_GLOBAL(dswap,DSWAP)
        (strumpack_blas_int* n, double* x, strumpack_blas_int* ldx, double* y, strumpack_blas_int* ldy);
      void STRUMPACK_FC_GLOBAL(cswap,CSWAP)
        (strumpack_blas_int* n, std::complex<float>* x, strumpack_blas_int* ldx,
         std::complex<float>* y, strumpack_blas_int* ldy);
      void STRUMPACK_FC_GLOBAL(zswap,ZSWAP)
        (strumpack_blas_int* n, std::complex<double>* x, strumpack_blas_int* ldx,
         std::complex<double>* y, strumpack_blas_int* ldy);

      float STRUMPACK_FC_GLOBAL(sdot,SDOT)
        (strumpack_blas_int* n, const float* x, strumpack_blas_int* incx, const float* y, strumpack_blas_int* incy);
      double STRUMPACK_FC_GLOBAL(ddot,DDOT)
        (strumpack_blas_int* n, const double* x, strumpack_blas_int* incx, const double* y, strumpack_blas_int* incy);


      ///////////////////////////////////////////////////////////
      ///////// BLAS2 ///////////////////////////////////////////
      ///////////////////////////////////////////////////////////
      void STRUMPACK_FC_GLOBAL(sgemv,SGEMV)
        (char* t, strumpack_blas_int* m, strumpack_blas_int* n, float* alpha, const float *a, strumpack_blas_int* lda,
         const float* x, strumpack_blas_int* incx, float* beta, float* y, strumpack_blas_int* incy);
      void STRUMPACK_FC_GLOBAL(dgemv,DGEMV)
        (char* t, strumpack_blas_int* m, strumpack_blas_int* n, double* alpha, const double *a, strumpack_blas_int* lda,
         const double* x, strumpack_blas_int* incx, double* beta, double* y, strumpack_blas_int* incy);
      void STRUMPACK_FC_GLOBAL(cgemv,CGEMV)
        (char* t, strumpack_blas_int* m, strumpack_blas_int* n, std::complex<float>* alpha,
         const std::complex<float> *a, strumpack_blas_int* lda,
         const std::complex<float>* x, strumpack_blas_int* incx, std::complex<float>* beta,
         std::complex<float>* y, strumpack_blas_int* incy);
      void STRUMPACK_FC_GLOBAL(zgemv,ZGEMV)
        (char* t, strumpack_blas_int* m, strumpack_blas_int* n, std::complex<double>* alpha,
         const std::complex<double> *a, strumpack_blas_int* lda,
         const std::complex<double>* x, strumpack_blas_int* incx, std::complex<double>* beta,
         std::complex<double>* y, strumpack_blas_int* incy);

      void STRUMPACK_FC_GLOBAL(sger,SGER)
        (strumpack_blas_int* m, strumpack_blas_int* n, float* alpha, const float* x, strumpack_blas_int* incx,
         const float* y, strumpack_blas_int* incy, float* a, strumpack_blas_int* lda);
      void STRUMPACK_FC_GLOBAL(dger,DGER)
        (strumpack_blas_int* m, strumpack_blas_int* n, double* alpha, const double* x, strumpack_blas_int* incx,
         const double* y, strumpack_blas_int* incy, double* a, strumpack_blas_int* lda);
      void STRUMPACK_FC_GLOBAL(cgeru,CGERU)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<float>* alpha,
         const std::complex<float>* x, strumpack_blas_int* incx,
         const std::complex<float>* y, strumpack_blas_int* incy,
         std::complex<float>* a, strumpack_blas_int* lda);
      void STRUMPACK_FC_GLOBAL(zgeru,ZGERU)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<double>* alpha,
         const std::complex<double>* x, strumpack_blas_int* incx,
         const std::complex<double>* y, strumpack_blas_int* incy,
         std::complex<double>* a, strumpack_blas_int* lda);
      void STRUMPACK_FC_GLOBAL(cgerc,CGERC)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<float>* alpha,
         const std::complex<float>* x, strumpack_blas_int* incx,
         const std::complex<float>* y, strumpack_blas_int* incy,
         std::complex<float>* a, strumpack_blas_int* lda);
      void STRUMPACK_FC_GLOBAL(zgerc,ZGERC)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<double>* alpha,
         const std::complex<double>* x, strumpack_blas_int* incx,
         const std::complex<double>* y, strumpack_blas_int* incy,
         std::complex<double>* a, strumpack_blas_int* lda);

      void STRUMPACK_FC_GLOBAL(strmv,STRMV)
        (char* ul, char* t, char* d, strumpack_blas_int* n,
         const float* a, strumpack_blas_int* lda, float* x, strumpack_blas_int* incx);
      void STRUMPACK_FC_GLOBAL(dtrmv,DTRMV)
        (char* ul, char* t, char* d, strumpack_blas_int* n,
         const double* a, strumpack_blas_int* lda, double* x, strumpack_blas_int* incx);
      void STRUMPACK_FC_GLOBAL(ctrmv,CTRMV)
        (char* ul, char* t, char* d, strumpack_blas_int* n,
         const std::complex<float>* a, strumpack_blas_int* lda,
         std::complex<float>* x, strumpack_blas_int* incx);
      void STRUMPACK_FC_GLOBAL(ztrmv,ZTRMV)
        (char* ul, char* t, char* d, strumpack_blas_int* n,
         const std::complex<double>* a, strumpack_blas_int* lda,
         std::complex<double>* x, strumpack_blas_int* incx);

      void STRUMPACK_FC_GLOBAL(strsv,STRSV)
        (char* ul, char* t, char* d, strumpack_blas_int* m, const float* a, strumpack_blas_int* lda,
         float* b, strumpack_blas_int* incb);
      void STRUMPACK_FC_GLOBAL(dtrsv,DTRSV)
        (char* ul, char* t, char* d, strumpack_blas_int* m, const double* a, strumpack_blas_int* lda,
         double* b, strumpack_blas_int* incb);
      void STRUMPACK_FC_GLOBAL(ctrsv,CTRSV)
        (char* ul, char* t, char* d, strumpack_blas_int* m,
         const std::complex<float>* a, strumpack_blas_int* lda,
         std::complex<float>* b, strumpack_blas_int* incb);
      void STRUMPACK_FC_GLOBAL(ztrsv,ZTRSV)
        (char* ul, char* t, char* d, strumpack_blas_int* m,
         const std::complex<double>* a, strumpack_blas_int* lda,
         std::complex<double>* b, strumpack_blas_int* incb);


      ///////////////////////////////////////////////////////////
      ///////// BLAS3 ///////////////////////////////////////////
      ///////////////////////////////////////////////////////////
      void STRUMPACK_FC_GLOBAL(sgemm,SGEMM)
        (char* ta, char* tb, strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k,
         float* alpha, const float* a, strumpack_blas_int* lda, const float* b, strumpack_blas_int* ldb,
         float* beta, float* c, strumpack_blas_int* ldc);
      void STRUMPACK_FC_GLOBAL(dgemm,DGEMM)
        (char* ta, char* tb, strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k,
         double* alpha, const double* A, strumpack_blas_int* lda, const double* b, strumpack_blas_int* ldb,
         double* beta, double* c, strumpack_blas_int* ldc);
      void STRUMPACK_FC_GLOBAL(cgemm,CGEMM)
        (char* ta, char* tb, strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k,
         std::complex<float>* alpha, const std::complex<float>* a, strumpack_blas_int* lda,
         const std::complex<float>* b, strumpack_blas_int* ldb, std::complex<float>* beta,
         std::complex<float>* c, strumpack_blas_int* ldc);
      void STRUMPACK_FC_GLOBAL(zgemm,ZGEMM)
        (char* ta, char* tb, strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k,
         std::complex<double>* alpha, const std::complex<double>* a, strumpack_blas_int* lda,
         const std::complex<double>* b, strumpack_blas_int* ldb, std::complex<double>* beta,
         std::complex<double>* c, strumpack_blas_int* ldc);

      void STRUMPACK_FC_GLOBAL(strsm,STRSM)
        (char* s, char* ul, char* t, char* d, strumpack_blas_int* m, strumpack_blas_int* n,
         float* alpha, const float* a, strumpack_blas_int* lda, float* b, strumpack_blas_int* ldb);
      void STRUMPACK_FC_GLOBAL(dtrsm,DTRSM)
        (char* s, char* ul, char* t, char* d, strumpack_blas_int* m, strumpack_blas_int* n,
         double* alpha, const double* a, strumpack_blas_int* lda, double* b, strumpack_blas_int* ldb);
      void STRUMPACK_FC_GLOBAL(ctrsm,CTRSM)
        (char* s, char* ul, char* t, char* d, strumpack_blas_int* m, strumpack_blas_int* n,
         std::complex<float>* alpha, const std::complex<float>* a, strumpack_blas_int* lda,
         std::complex<float>* b, strumpack_blas_int* ldb);
      void STRUMPACK_FC_GLOBAL(ztrsm,ZTRSM)
        (char* s, char* ul, char* t, char* d, strumpack_blas_int* m, strumpack_blas_int* n,
         std::complex<double>* alpha, const std::complex<double>* a, strumpack_blas_int* lda,
         std::complex<double>* b, strumpack_blas_int* ldb);

      void STRUMPACK_FC_GLOBAL(strmm,STRMM)
        (char* s, char* ul, char* t, char* d, strumpack_blas_int* m, strumpack_blas_int* n, float* alpha,
         const float* a, strumpack_blas_int* lda, float* b, strumpack_blas_int* ldb);
      void STRUMPACK_FC_GLOBAL(dtrmm,DTRMM)
        (char* s, char* ul, char* t, char* d, strumpack_blas_int* m, strumpack_blas_int* n, double* alpha,
         const double* a, strumpack_blas_int* lda, double* b, strumpack_blas_int* ldb);
      void STRUMPACK_FC_GLOBAL(ctrmm,CTRMM)
        (char* s, char* ul, char* t, char* d, strumpack_blas_int* m, strumpack_blas_int* n,
         std::complex<float>* alpha, const std::complex<float>* a, strumpack_blas_int* lda,
         std::complex<float>* b, strumpack_blas_int* ldb);
      void STRUMPACK_FC_GLOBAL(ztrmm,ZTRMM)
        (char* s, char* ul, char* t, char* d, strumpack_blas_int* m, strumpack_blas_int* n,
         std::complex<double>* alpha, const std::complex<double>* a, strumpack_blas_int* lda,
         std::complex<double>* b, strumpack_blas_int* ldb);


      ///////////////////////////////////////////////////////////
      ///////// LAPACK //////////////////////////////////////////
      ///////////////////////////////////////////////////////////
      float STRUMPACK_FC_GLOBAL(slamch,SLAMCH)(char* cmach);
      double STRUMPACK_FC_GLOBAL(dlamch,DLAMCH)(char* cmach);

      strumpack_blas_int STRUMPACK_FC_GLOBAL(ilaenv,ILAENV)
        (strumpack_blas_int* ispec, char* name, char* opts,
         strumpack_blas_int* n1, strumpack_blas_int* n2, strumpack_blas_int* n3, strumpack_blas_int* n4);

      void STRUMPACK_FC_GLOBAL(clacgv,CLACGV)
        (strumpack_blas_int* n, std::complex<float>* x, strumpack_blas_int* incx);
      void STRUMPACK_FC_GLOBAL(zlacgv,ZLACGV)
        (strumpack_blas_int* n, std::complex<double>* x, strumpack_blas_int* incx);

      strumpack_blas_int STRUMPACK_FC_GLOBAL(slacpy,SLACPY)
        (char* uplo, strumpack_blas_int* m, strumpack_blas_int* n, const float* a, strumpack_blas_int* lda,
         float* b, strumpack_blas_int* ldb);
      strumpack_blas_int STRUMPACK_FC_GLOBAL(dlacpy,DLACPY)
        (char* uplo, strumpack_blas_int* m, strumpack_blas_int* n, const double* a, strumpack_blas_int* lda,
         double* b, strumpack_blas_int* ldb);
      strumpack_blas_int STRUMPACK_FC_GLOBAL(clacpy,CLACPY)
        (char* uplo, strumpack_blas_int* m, strumpack_blas_int* n, const std::complex<float>* a, strumpack_blas_int* lda,
         std::complex<float>* b, strumpack_blas_int* ldb);
      strumpack_blas_int STRUMPACK_FC_GLOBAL(zlacpy,ZLACPY)
        (char* uplo, strumpack_blas_int* m, strumpack_blas_int* n, const std::complex<double>* a, strumpack_blas_int* lda,
         std::complex<double>* b, strumpack_blas_int* ldb);

      void STRUMPACK_FC_GLOBAL(slaswp,SLASWP)
        (strumpack_blas_int* n, float* a, strumpack_blas_int* lda, strumpack_blas_int* k1, strumpack_blas_int* k2,
         const strumpack_blas_int* ipiv, strumpack_blas_int* incx);
      void STRUMPACK_FC_GLOBAL(dlaswp,DLASWP)
        (strumpack_blas_int* n, double* a, strumpack_blas_int* lda, strumpack_blas_int* k1, strumpack_blas_int* k2,
         const strumpack_blas_int* ipiv, strumpack_blas_int* incx);
      void STRUMPACK_FC_GLOBAL(claswp,CLASWP)
        (strumpack_blas_int* n, std::complex<float>* a, strumpack_blas_int* lda, strumpack_blas_int* k1, strumpack_blas_int* k2,
         const strumpack_blas_int* ipiv, strumpack_blas_int* incx);
      void STRUMPACK_FC_GLOBAL(zlaswp,ZLASWP)
        (strumpack_blas_int* n, std::complex<double>* a, strumpack_blas_int* lda, strumpack_blas_int* k1, strumpack_blas_int* k2,
         const strumpack_blas_int* ipiv, strumpack_blas_int* incx);

      void STRUMPACK_FC_GLOBAL(myslapmr,MYSLAPMR)
        (strumpack_blas_int* fwd, strumpack_blas_int* m, strumpack_blas_int* n, float* a, strumpack_blas_int* lda, const strumpack_blas_int* ipiv);
      void STRUMPACK_FC_GLOBAL(mydlapmr,MYDLAPMR)
        (strumpack_blas_int* fwd, strumpack_blas_int* m, strumpack_blas_int* n, double* a, strumpack_blas_int* lda, const strumpack_blas_int* ipiv);
      void STRUMPACK_FC_GLOBAL(myclapmr,MYCLAPMR)
        (strumpack_blas_int* fwd, strumpack_blas_int* m, strumpack_blas_int* n, std::complex<float>* a, strumpack_blas_int* lda,
         const strumpack_blas_int* ipiv);
      void STRUMPACK_FC_GLOBAL(myzlapmr,MYZLAPMR)
        (strumpack_blas_int* fwd, strumpack_blas_int* m, strumpack_blas_int* n, std::complex<double>* a, strumpack_blas_int* lda,
         const strumpack_blas_int* ipiv);

      void STRUMPACK_FC_GLOBAL(slapmt,SLAPMT)
        (strumpack_blas_int* fwd, strumpack_blas_int* m, strumpack_blas_int* n, float* a, strumpack_blas_int* lda, const strumpack_blas_int* ipiv);
      void STRUMPACK_FC_GLOBAL(dlapmt,DLAPMT)
        (strumpack_blas_int* fwd, strumpack_blas_int* m, strumpack_blas_int* n, double* a, strumpack_blas_int* lda, const strumpack_blas_int* ipiv);
      void STRUMPACK_FC_GLOBAL(clapmt,CLAPMT)
        (strumpack_blas_int* fwd, strumpack_blas_int* m, strumpack_blas_int* n, std::complex<float>* a, strumpack_blas_int* lda,
         const strumpack_blas_int* ipiv);
      void STRUMPACK_FC_GLOBAL(zlapmt,ZLAPMT)
        (strumpack_blas_int* fwd, strumpack_blas_int* m, strumpack_blas_int* n, std::complex<double>* a, strumpack_blas_int* lda,
         const strumpack_blas_int* ipiv);

      void STRUMPACK_FC_GLOBAL(slaset,SLASET)
        (char* s, strumpack_blas_int* m, strumpack_blas_int* n, float* alpha,
         float* beta, float* a, strumpack_blas_int* lda);
      void STRUMPACK_FC_GLOBAL(dlaset,DLASET)
        (char* s, strumpack_blas_int* m, strumpack_blas_int* n, double* alpha,
         double* beta, double* a, strumpack_blas_int* lda);
      void STRUMPACK_FC_GLOBAL(claset,CLASET)
        (char* s, strumpack_blas_int* m, strumpack_blas_int* n, std::complex<float>* alpha,
         std::complex<float>* beta, std::complex<float>* a, strumpack_blas_int* lda);
      void STRUMPACK_FC_GLOBAL(zlaset,ZLASET)
        (char* s, strumpack_blas_int* m, strumpack_blas_int* n, std::complex<double>* alpha,
         std::complex<double>* beta, std::complex<double>* a, strumpack_blas_int* lda);

      void STRUMPACK_FC_GLOBAL(sgeqp3,SGEQP3)
        (strumpack_blas_int* m, strumpack_blas_int* n, float* a, strumpack_blas_int* lda, strumpack_blas_int* jpvt,
         float* tau, float* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dgeqp3,DGEQP3)
        (strumpack_blas_int* m, strumpack_blas_int* n, double* a, strumpack_blas_int* lda, strumpack_blas_int* jpvt,
         double* tau, double* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(cgeqp3,CGEQP3)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<float>* a, strumpack_blas_int* lda, strumpack_blas_int* jpvt,
         std::complex<float>* tau, std::complex<float>* work, strumpack_blas_int* lwork,
         std::complex<float>* rwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zgeqp3,ZGEQP3)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<double>* a, strumpack_blas_int* lda, strumpack_blas_int* jpvt,
         std::complex<double>* tau, std::complex<double>* work, strumpack_blas_int* lwork,
         std::complex<double>* rwork, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(sgeqp3tol,SGEQP3TOL)
        (strumpack_blas_int* m, strumpack_blas_int* n, float* a, strumpack_blas_int* lda, strumpack_blas_int* jpvt,
         float* tau, float* work, strumpack_blas_int* lwork, strumpack_blas_int* info,
         strumpack_blas_int* rank, float* rtol, float* atol);
      void STRUMPACK_FC_GLOBAL(dgeqp3tol,DGEQP3TOL)
        (strumpack_blas_int* m, strumpack_blas_int* n, double* a, strumpack_blas_int* lda, strumpack_blas_int* jpvt,
         double* tau, double* work, strumpack_blas_int* lwork, strumpack_blas_int* info,
         strumpack_blas_int* rank, double* rtol, double* atol);
      void STRUMPACK_FC_GLOBAL(cgeqp3tol,CGEQP3TOL)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<float>* a, strumpack_blas_int* lda, strumpack_blas_int* jpvt,
         std::complex<float>* tau, std::complex<float>* work, strumpack_blas_int* lwork,
         float* rwork, strumpack_blas_int* info, strumpack_blas_int* rank, float* rtol, float* atol);
      void STRUMPACK_FC_GLOBAL(zgeqp3tol,ZGEQP3TOL)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<double>* a, strumpack_blas_int* lda, strumpack_blas_int* jpvt,
         std::complex<double>* tau, std::complex<double>* work, strumpack_blas_int* lwork,
         double* rwork, strumpack_blas_int* info, strumpack_blas_int* rank, double* rtol, double* atol);

      void STRUMPACK_FC_GLOBAL(sgeqrf,SGEQRF)
        (strumpack_blas_int* m, strumpack_blas_int* n, float* a, strumpack_blas_int* lda, float* tau,
         float* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dgeqrf,DGEQRF)
        (strumpack_blas_int* m, strumpack_blas_int* n, double* a, strumpack_blas_int* lda, double* tau,
         double* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(cgeqrf,CGEQRF)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<float>* a, strumpack_blas_int* lda,
         std::complex<float>* tau, std::complex<float>* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zgeqrf,ZGEQRF)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<double>* a, strumpack_blas_int* lda,
         std::complex<double>* tau, std::complex<double>* work, strumpack_blas_int* lwork, strumpack_blas_int* info);


      void STRUMPACK_FC_GLOBAL(sgelqf,SGELQF)
        (strumpack_blas_int* m, strumpack_blas_int* n, float* a, strumpack_blas_int* lda, float* tau,
         float* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dgelqf,DGELQF)
        (strumpack_blas_int* m, strumpack_blas_int* n, double* a, strumpack_blas_int* lda, double* tau,
         double* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(cgelqf,CGELQF)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<float>* a, strumpack_blas_int* lda,
         std::complex<float>* tau, std::complex<float>* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zgelqf,ZGELQF)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<double>* a, strumpack_blas_int* lda,
         std::complex<double>* tau, std::complex<double>* work, strumpack_blas_int* lwork, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(sgetrf,SGETRF)
        (strumpack_blas_int* m, strumpack_blas_int* n, float* a, strumpack_blas_int* lda, strumpack_blas_int* ipiv, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dgetrf,DGETRF)
        (strumpack_blas_int* m, strumpack_blas_int* n, double* a, strumpack_blas_int* lda, strumpack_blas_int* ipiv, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(cgetrf,CGETRF)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<float>* a, strumpack_blas_int* lda,
         strumpack_blas_int* ipiv, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zgetrf,ZGETRF)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<double>* a, strumpack_blas_int* lda,
         strumpack_blas_int* ipiv, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(sgetf2,SGETF2)
        (strumpack_blas_int* m, strumpack_blas_int* n, float* a, strumpack_blas_int* lda, strumpack_blas_int* ipiv, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dgetf2,DGETF2)
        (strumpack_blas_int* m, strumpack_blas_int* n, double* a, strumpack_blas_int* lda, strumpack_blas_int* ipiv, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(cgetf2,CGETF2)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<float>* a, strumpack_blas_int* lda,
         strumpack_blas_int* ipiv, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zgetf2,ZGETF2)
        (strumpack_blas_int* m, strumpack_blas_int* n, std::complex<double>* a, strumpack_blas_int* lda,
         strumpack_blas_int* ipiv, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(sgetrs,SGETRS)
        (char* t, strumpack_blas_int* n, strumpack_blas_int* nrhs, const float* a, strumpack_blas_int* lda,
         const strumpack_blas_int* ipiv, float* b, strumpack_blas_int* ldb, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dgetrs,dgetrs)
        (char* t, strumpack_blas_int* n, strumpack_blas_int* nrhs, const double* a, strumpack_blas_int* lda,
         const strumpack_blas_int* ipiv, double* b, strumpack_blas_int* ldb, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(cgetrs,cgetrs)
        (char* t, strumpack_blas_int* n, strumpack_blas_int* nrhs, const std::complex<float>* a, strumpack_blas_int* lda,
         const strumpack_blas_int* ipiv, std::complex<float>* b, strumpack_blas_int* ldb, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zgetrs,ZGETRS)
        (char* t, strumpack_blas_int* n, strumpack_blas_int* nrhs, const std::complex<double>* a, strumpack_blas_int* lda,
         const strumpack_blas_int* ipiv, std::complex<double>* b, strumpack_blas_int* ldb, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(spotrf,SPOTRF)
        (char* ul, strumpack_blas_int* n, float* a, strumpack_blas_int* lda, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dpotrf,DPOTRF)
        (char* ul, strumpack_blas_int* n, double* a, strumpack_blas_int* lda, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(cpotrf,CPOTRF)
        (char* ul, strumpack_blas_int* n, std::complex<float>* a, strumpack_blas_int* lda, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zpotrf,ZPOTRF)
        (char* ul, strumpack_blas_int* n, std::complex<double>* a,
         strumpack_blas_int* lda, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(sgetri,SGETRI)
        (strumpack_blas_int* n, float* a, strumpack_blas_int* lda, strumpack_blas_int* ipiv,
         float* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dgetri,DGETRI)
        (strumpack_blas_int* n, double* a, strumpack_blas_int* lda, strumpack_blas_int* ipiv,
         double* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(cgetri,CGETRI)
        (strumpack_blas_int* n, std::complex<float>* a, strumpack_blas_int* lda, strumpack_blas_int* ipiv,
         std::complex<float>* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zgetri,ZGETRI)
        (strumpack_blas_int* n, std::complex<double>* a, strumpack_blas_int* lda, strumpack_blas_int* ipiv,
         std::complex<double>* work, strumpack_blas_int* lwork, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(sorglq,SORGLQ)
        (strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k, float* a, strumpack_blas_int* lda, const float* tau,
         float* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dorglq,DORGLQ)
        (strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k, double* a, strumpack_blas_int* lda, const double* tau,
         double* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(cunglq,CORGLQ)
        (strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k, std::complex<float>* a, strumpack_blas_int* lda,
         const std::complex<float>* tau, std::complex<float>* work,
         strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zunglq,ZORGLQ)
        (strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k, std::complex<double>* a, strumpack_blas_int* lda,
         const std::complex<double>* tau, std::complex<double>* work,
         strumpack_blas_int* lwork, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(sorgqr,SORGQR)
        (strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k, float* a, strumpack_blas_int* lda, const float* tau,
         float* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dorgqr,DORGQR)
        (strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k, double* a, strumpack_blas_int* lda, const double* tau,
         double* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(cungqr,CORGQR)
        (strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k, std::complex<float>* a, strumpack_blas_int* lda,
         const std::complex<float>* tau, std::complex<float>* work,
         strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zungqr,ZORGQR)
        (strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k, std::complex<double>* a, strumpack_blas_int* lda,
         const std::complex<double>* tau, std::complex<double>* work,
         strumpack_blas_int* lwork, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(sormqr,SORMQR)
        (char* side, char* trans, strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k,
         const float* a, strumpack_blas_int* lda, const float* tau, float* c, strumpack_blas_int* ldc,
         float* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dormqr,DORMQR)
        (char* size, char* trans, strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k,
         const double* a, strumpack_blas_int* lda, const double* tau, double* c, strumpack_blas_int* ldc,
         double* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(cunmqr,CORMQR)
        (char* size, char* trans, strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k,
         const std::complex<float>* a, strumpack_blas_int* lda,
         const std::complex<float>* tau, std::complex<float>* c, strumpack_blas_int* ldc,
         std::complex<float>* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zunmqr,ZORMQR)
        (char* size, char* trans, strumpack_blas_int* m, strumpack_blas_int* n, strumpack_blas_int* k,
         const std::complex<double>* a, strumpack_blas_int* lda,
         const std::complex<double>* tau, std::complex<double>* c, strumpack_blas_int* ldc,
         std::complex<double>* work, strumpack_blas_int* lwork, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(sgesv,SGESV)
        (strumpack_blas_int* n, strumpack_blas_int* nrhs, float* a, strumpack_blas_int* lda, strumpack_blas_int* ipiv,
         float* b, strumpack_blas_int* ldb, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dgesv,DGESV)
        (strumpack_blas_int* n, strumpack_blas_int* nrhs, double* a, strumpack_blas_int* lda, strumpack_blas_int* ipiv,
         double* b, strumpack_blas_int* ldb, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(cgesv,CGESV)
        (strumpack_blas_int* n, strumpack_blas_int* nrhs, std::complex<float>* a, strumpack_blas_int* lda, strumpack_blas_int* ipiv,
         std::complex<float>* b, strumpack_blas_int* ldb, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zgesv,ZGESV)
        (strumpack_blas_int* n, strumpack_blas_int* nrhs, std::complex<double>* a, strumpack_blas_int* lda, strumpack_blas_int* ipiv,
         std::complex<double>* b, strumpack_blas_int* ldb, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(slarnv,SLARNV)
        (strumpack_blas_int* idist, strumpack_blas_int* iseed, strumpack_blas_int* n, float* x);
      void STRUMPACK_FC_GLOBAL(dlarnv,DLARNV)
        (strumpack_blas_int* idist, strumpack_blas_int* iseed, strumpack_blas_int* n, double* x);
      void STRUMPACK_FC_GLOBAL(clarnv,CLARNV)
        (strumpack_blas_int* idist, strumpack_blas_int* iseed, strumpack_blas_int* n, std::complex<float>* x);
      void STRUMPACK_FC_GLOBAL(zlarnv,ZLARNV)
        (strumpack_blas_int* idist, strumpack_blas_int* iseed, strumpack_blas_int* n, std::complex<double>* x);

      float STRUMPACK_FC_GLOBAL(slange,SLANGE)
        (char* norm, strumpack_blas_int* m, strumpack_blas_int* n, const float* a, strumpack_blas_int* lda, float* work);
      double STRUMPACK_FC_GLOBAL(dlange,DLANGE)
        (char* norm, strumpack_blas_int* m, strumpack_blas_int* n, const double* a,strumpack_blas_int* lda, double* work);
      float STRUMPACK_FC_GLOBAL(clange,CLANGE)
        (char* norm, strumpack_blas_int* m, strumpack_blas_int* n,
         const std::complex<float>* a, strumpack_blas_int* lda, float* work);
      double STRUMPACK_FC_GLOBAL(zlange,ZLANGE)
        (char* norm, strumpack_blas_int* m, strumpack_blas_int* n,
         const std::complex<double>* a, strumpack_blas_int* lda, double* work);

      void STRUMPACK_FC_GLOBAL(sgesvd,SGESVD)
        (char* jobu, char* jobvt, strumpack_blas_int* m, strumpack_blas_int* n, float* a, strumpack_blas_int* lda,
         float* s, float* u, strumpack_blas_int* ldu, float* vt, strumpack_blas_int* ldvt,
         float* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dgesvd,DGESVD)
        (char* jobu, char* jobvt, strumpack_blas_int* m, strumpack_blas_int* n, double* a, strumpack_blas_int* lda,
         double* s, double* u, strumpack_blas_int* ldu, double* vt, strumpack_blas_int* ldvt,
         double* work, strumpack_blas_int* lwork, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(ssyevx,SSYEVX)
        (char* jobz, char* range, char* uplo, strumpack_blas_int* n,
         float* a, strumpack_blas_int* lda, float* vl, float* vu, strumpack_blas_int* il, strumpack_blas_int* iu,
         float* abstol, strumpack_blas_int* m, float* w, float* z, strumpack_blas_int* ldz,
         float* work, strumpack_blas_int* lwork, strumpack_blas_int* iwork, strumpack_blas_int* ifail, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dsyevx,DSYEVX)
        (char* jobz, char* range, char* uplo, strumpack_blas_int* n,
         double* a, strumpack_blas_int* lda, double* vl, double* vu, strumpack_blas_int* il, strumpack_blas_int* iu,
         double* abstol, strumpack_blas_int* m, double* w, double* z, strumpack_blas_int* ldz,
         double* work, strumpack_blas_int* lwork, strumpack_blas_int* iwork, strumpack_blas_int* ifail, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(ssyev,SSYEV)
        (char* jobz, char* uplo, strumpack_blas_int* n, float* a, strumpack_blas_int* lda, float* w,
         float* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dsyev,DSYEV)
        (char* jobz, char* uplo, strumpack_blas_int* n, double* a, strumpack_blas_int* lda, double* w,
         double* work, strumpack_blas_int* lwork, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(ssytrf,SSYTRF)
         (char* s, strumpack_blas_int* n, float* a, strumpack_blas_int*lda, strumpack_blas_int* ipiv, float* work,
            strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dsytrf,DSYTRF)
         (char* s, strumpack_blas_int* n, double* a, strumpack_blas_int*lda, strumpack_blas_int* ipiv, double* work,
            strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(csytrf,CSYTRF)
         (char* s, strumpack_blas_int* n, std::complex<float>* a, strumpack_blas_int*lda, strumpack_blas_int* ipiv,
            std::complex<float>* work, strumpack_blas_int* lwork, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zsytrf,ZSYTRF)
         (char* s, strumpack_blas_int* n, std::complex<double>* a, strumpack_blas_int*lda, strumpack_blas_int* ipiv,
            std::complex<double>* work, strumpack_blas_int* lwork, strumpack_blas_int* info);

      void STRUMPACK_FC_GLOBAL(ssytrs,SSYTRS)
        (char* s, strumpack_blas_int* n, strumpack_blas_int* nrhs, const float* a, strumpack_blas_int* lda, const strumpack_blas_int* ipiv,
         float* b, strumpack_blas_int* ldb, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(dsytrs,DSYTRS)
        (char* s, strumpack_blas_int* n, strumpack_blas_int* nrhs, const double* a, strumpack_blas_int* lda, const strumpack_blas_int* ipiv,
         double* b, strumpack_blas_int* ldb, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(csytrs,CSYTRS)
        (char* s, strumpack_blas_int* n, strumpack_blas_int* nrhs, const std::complex<float>* a, strumpack_blas_int* lda,
         const strumpack_blas_int* ipiv, std::complex<float>* b, strumpack_blas_int* ldb, strumpack_blas_int* info);
      void STRUMPACK_FC_GLOBAL(zsytrs,ZSYTRS)
        (char* s, strumpack_blas_int* n, strumpack_blas_int* nrhs, const std::complex<double>* a, strumpack_blas_int* lda,
         const strumpack_blas_int* ipiv, std::complex<double>* b, strumpack_blas_int* ldb, strumpack_blas_int* info);
    }

    int ilaenv(int ispec, char name[], char opts[], int n1, int n2, int n3, int n4) {
      strumpack_blas_int ispec_ = ispec, n1_ = n1, n2_ = n2, n3_ = n3, n4_ = n4;
      return STRUMPACK_FC_GLOBAL
        (ilaenv,ILAENV)(&ispec_, name, opts, &n1_, &n2_, &n3_, &n4_);
    }

    template<> float lamch<float>(char cmach) {
      return STRUMPACK_FC_GLOBAL(slamch,SLAMCH)(&cmach);
    }
    template<> double lamch<double>(char cmach) {
      return STRUMPACK_FC_GLOBAL(dlamch,DLAMCH)(&cmach);
    }

    void gemm(char ta, char tb, int m, int n, int k, float alpha,
              const float *a, int lda, const float *b, int ldb,
              float beta, float *c, int ldc) {
      strumpack_blas_int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
      STRUMPACK_FC_GLOBAL(sgemm,SGEMM)
        (&ta, &tb, &m_, &n_, &k_, &alpha, a, &lda_, b, &ldb_, &beta, c, &ldc_);
      STRUMPACK_FLOPS(gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(4*gemm_moves(m,n,k));
    }
    void gemm(char ta, char tb, int m, int n, int k, double alpha,
              const double *a, int lda, const double *b, int ldb,
              double beta, double *c, int ldc) {
      strumpack_blas_int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
      STRUMPACK_FC_GLOBAL(dgemm,DGEMM)
        (&ta, &tb, &m_, &n_, &k_, &alpha, a, &lda_, b, &ldb_, &beta, c, &ldc_);
      STRUMPACK_FLOPS(gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(8*gemm_moves(m,n,k));
    }
    void gemm(char ta, char tb, int m, int n, int k, std::complex<float> alpha,
              const std::complex<float>* a, int lda,
              const std::complex<float>* b, int ldb, std::complex<float> beta,
              std::complex<float>* c, int ldc) {
      strumpack_blas_int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
      STRUMPACK_FC_GLOBAL(cgemm,CGEMM)
        (&ta, &tb, &m_, &n_, &k_, &alpha, a, &lda_, b, &ldb_, &beta, c, &ldc_);
      STRUMPACK_FLOPS(4*gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(2*4*gemm_moves(m,n,k));
    }
    void gemm(char ta, char tb, int m, int n, int k, std::complex<double> alpha,
              const std::complex<double>* a, int lda,
              const std::complex<double>* b, int ldb, std::complex<double> beta,
              std::complex<double>* c, int ldc) {
      strumpack_blas_int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
      STRUMPACK_FC_GLOBAL(zgemm,ZGEMM)
        (&ta, &tb, &m_, &n_, &k_, &alpha, a, &lda_, b, &ldb_, &beta, c, &ldc_);
      STRUMPACK_FLOPS(4*gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(2*8*gemm_moves(m,n,k));
    }

    void gemv(char t, int m, int n, float alpha, const float *a, int lda,
              const float *x, int incx, float beta, float *y, int incy) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(sgemv,SGEMV)
        (&t, &m_, &n_, &alpha, a, &lda_, x, &incx_, &beta, y, &incy_);
      STRUMPACK_FLOPS(gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(4*gemv_moves(m,n));
    }
    void gemv(char t, int m, int n, double alpha, const double *a, int lda,
              const double *x, int incx, double beta, double *y, int incy) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(dgemv,DGEMV)
        (&t, &m_, &n_, &alpha, a, &lda_, x, &incx_, &beta, y, &incy_);
      STRUMPACK_FLOPS(gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(8*gemv_moves(m,n));
    }
    void gemv(char t, int m, int n, std::complex<float> alpha,
              const std::complex<float> *a, int lda,
              const std::complex<float> *x, int incx, std::complex<float> beta,
              std::complex<float> *y, int incy) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(cgemv,CGEMV)
        (&t, &m_, &n_, &alpha, a, &lda_, x, &incx_, &beta, y, &incy_);
      STRUMPACK_FLOPS(4*gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(2*4*gemv_moves(m,n));
    }
    void gemv(char t, int m, int n, std::complex<double> alpha,
              const std::complex<double> *a, int lda,
              const std::complex<double> *x, int incx, std::complex<double> beta,
              std::complex<double> *y, int incy) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(zgemv,ZGEMV)
        (&t, &m_, &n_, &alpha, a, &lda_, x, &incx_, &beta, y, &incy_);
      STRUMPACK_FLOPS(4*gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(2*8*gemv_moves(m,n));
    }


    void geru(int m, int n, float alpha, const float* x, int incx,
              const float* y, int incy, float* a, int lda) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(sger,SGER)(&m_, &n_, &alpha, x, &incx_, y, &incy_, a, &lda_);
      STRUMPACK_FLOPS(ger_flops(m,n,alpha));
    }
    void geru(int m, int n, double alpha, const double* x, int incx,
              const double* y, int incy, double* a, int lda) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(dger,DGER)(&m_, &n_, &alpha, x, &incx_, y, &incy_, a, &lda_);
      STRUMPACK_FLOPS(ger_flops(m,n,alpha));
    }
    void geru(int m, int n, std::complex<float> alpha,
              const std::complex<float>* x, int incx,
              const std::complex<float>* y, int incy,
              std::complex<float>* a, int lda) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(cgeru,CGERU)(&m_, &n_, &alpha, x, &incx_, y, &incy_, a, &lda_);
      STRUMPACK_FLOPS(4*ger_flops(m,n,alpha));
    }
    void geru(int m, int n, std::complex<double> alpha,
              const std::complex<double>* x, int incx,
              const std::complex<double>* y, int incy,
              std::complex<double>* a, int lda) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(zgeru,ZGERU)(&m_, &n_, &alpha, x, &incx_, y, &incy_, a, &lda_);
      STRUMPACK_FLOPS(4*ger_flops(m,n,alpha));
    }


    void gerc(int m, int n, float alpha, const float* x, int incx,
              const float* y, int incy, float* a, int lda) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(sger,SGER)(&m_, &n_, &alpha, x, &incx_, y, &incy_, a, &lda_);
      STRUMPACK_FLOPS(ger_flops(m,n,alpha));
    }
    void gerc(int m, int n, double alpha, const double* x, int incx,
              const double* y, int incy, double* a, int lda) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(dger,DGER)(&m_, &n_, &alpha, x, &incx_, y, &incy_, a, &lda_);
      STRUMPACK_FLOPS(ger_flops(m,n,alpha));
    }
    void gerc(int m, int n, std::complex<float> alpha,
              const std::complex<float>* x, int incx,
              const std::complex<float>* y, int incy,
              std::complex<float>* a, int lda) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(cgerc,CGERC)(&m_, &n_, &alpha, x, &incx_, y, &incy_, a, &lda_);
      STRUMPACK_FLOPS(4*ger_flops(m,n,alpha));
    }
    void gerc(int m, int n, std::complex<double> alpha,
              const std::complex<double>* x, int incx,
              const std::complex<double>* y, int incy,
              std::complex<double>* a, int lda) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(zgerc,ZGERC)(&m_, &n_, &alpha, x, &incx_, y, &incy_, a, &lda_);
      STRUMPACK_FLOPS(4*ger_flops(m,n,alpha));
    }


    void lacgv(int, float *, int ) { }
    void lacgv(int, double *, int ) { } //Nothing to do.
    void lacgv(int n, std::complex<float> *x, int incx) {
      strumpack_blas_int n_ = n, incx_ = incx;
      STRUMPACK_FC_GLOBAL(clacgv,CLACGV)(&n_, x, &incx_);
    }
    void lacgv(int n, std::complex<double> *x, int incx) {
      strumpack_blas_int n_ = n, incx_ = incx;
      STRUMPACK_FC_GLOBAL(zlacgv,ZLACGV)(&n_, x, &incx_);
    }


    void lacpy(char ul, int m, int n, float* a, int lda, float* b, int ldb) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, ldb_ = ldb;
      STRUMPACK_FC_GLOBAL(slacpy,SLACPY)(&ul, &m_, &n_, a, &lda_, b, &ldb_);
    }
    void lacpy(char ul, int m, int n, double* a, int lda, double* b, int ldb) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, ldb_ = ldb;
      STRUMPACK_FC_GLOBAL(dlacpy,DLACPY)(&ul, &m_, &n_, a, &lda_, b, &ldb_);
    }
    void lacpy(char ul, int m, int n, std::complex<float>* a, int lda,
               std::complex<float>* b, int ldb) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, ldb_ = ldb;
      STRUMPACK_FC_GLOBAL(clacpy,CLACPY)(&ul, &m_, &n_, a, &lda_, b, &ldb_);
    }
    void lacpy(char ul, int m, int n, std::complex<double>* a, int lda,
               std::complex<double>* b, int ldb) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, ldb_ = ldb;
      STRUMPACK_FC_GLOBAL(zlacpy,ZLACPY)(&ul, &m_, &n_, a, &lda_, b, &ldb_);
    }


    void axpy(int n, float alpha, float* x, int incx, float* y, int incy) {
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(saxpy,SAXPY)(&n_, &alpha, x, &incx_, y, &incy_);
      STRUMPACK_FLOPS(axpy_flops(n,alpha));
    }
    void axpy(int n, double alpha, double* x, int incx, double* y, int incy) {
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(daxpy,DAXPY)(&n_, &alpha, x, &incx_, y, &incy_);
      STRUMPACK_FLOPS(axpy_flops(n,alpha));
    }
    void axpy(int n, std::complex<float> alpha,
              const std::complex<float>* x, int incx,
              std::complex<float>* y, int incy) {
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(caxpy,CAXPY)(&n_, &alpha, x, &incx_, y, &incy_);
      STRUMPACK_FLOPS(4*axpy_flops(n,alpha));
    }
    void axpy(int n, std::complex<double> alpha,
              const std::complex<double>* x, int incx,
              std::complex<double>* y, int incy) {
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(zaxpy,ZAXPY)(&n_, &alpha, x, &incx_, y, &incy_);
      STRUMPACK_FLOPS(4*axpy_flops(n,alpha));
    }


    void copy(int n, const float* x, int incx, float* y, int incy) {
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(scopy,SCOPY)(&n_, x, &incx_, y, &incy_);
    }
    void copy(int n, const double* x, int incx, double* y, int incy) {
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(dcopy,DCOPY)(&n_, x, &incx_, y, &incy_);
    }
    void copy(int n, const std::complex<float>* x, int incx,
              std::complex<float>* y, int incy) {
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(ccopy,CCOPY)(&n_, x, &incx_, y, &incy_);
    }
    void copy(int n, const std::complex<double>* x, int incx,
              std::complex<double>* y, int incy) {
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(zcopy,ZCOPY)(&n_, x, &incx_, y, &incy_);
    }

    void scal(int n, float alpha, float* x, int incx) {
      strumpack_blas_int n_ = n, incx_ = incx;
      STRUMPACK_FC_GLOBAL(sscal,SSCAL)(&n_, &alpha, x, &incx_);
      STRUMPACK_FLOPS(scal_flops(n,alpha));
    }
    void scal(int n, double alpha, double* x, int incx) {
      strumpack_blas_int n_ = n, incx_ = incx;
      STRUMPACK_FC_GLOBAL(dscal,DSCAL)(&n_, &alpha, x, &incx_);
      STRUMPACK_FLOPS(scal_flops(n,alpha));
    }
    void scal(int n, std::complex<float> alpha, std::complex<float>* x, int incx) {
      strumpack_blas_int n_ = n, incx_ = incx;
      STRUMPACK_FC_GLOBAL(cscal,CSCAL)(&n_, &alpha, x, &incx_);
      STRUMPACK_FLOPS(4*scal_flops(n,alpha));
    }
    void scal(int n, std::complex<double> alpha, std::complex<double>* x, int incx) {
      strumpack_blas_int n_ = n, incx_ = incx;
      STRUMPACK_FC_GLOBAL(zscal,ZSCAL)(&n_, &alpha, x, &incx_);
      STRUMPACK_FLOPS(4*scal_flops(n,alpha));
    }


    int iamax(int n, const float* x, int incx) {
      strumpack_blas_int n_ = n, incx_ = incx;
      return STRUMPACK_FC_GLOBAL(isamax,ISAMAX)(&n_, x, &incx_);
    }
    int iamax(int n, const double* x, int incx) {
      strumpack_blas_int n_ = n, incx_ = incx;
      return STRUMPACK_FC_GLOBAL(idamax,IDAMAX)(&n_, x, &incx_);
    }
    int iamax(int n, const std::complex<float>* x, int incx) {
      strumpack_blas_int n_ = n, incx_ = incx;
      return STRUMPACK_FC_GLOBAL(icamax,ICAMAX)(&n_, x, &incx_);
    }
    int iamax(int n, const std::complex<double>* x, int incx) {
      strumpack_blas_int n_ = n, incx_ = incx;
      return STRUMPACK_FC_GLOBAL(izamax,IZAMAX)(&n_, x, &incx_);
    }


    void swap(int n, float* x, int incx, float* y, int incy) {
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(sswap,SSWAP)(&n_, x, &incx_, y, &incy_);
    }
    void swap(int n, double* x, int incx, double* y, int incy) {
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(dswap,DSWAP)(&n_, x, &incx_, y, &incy_);
    }
    void swap(int n, std::complex<float>* x, int incx,
              std::complex<float>* y, int incy) {
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(cswap,CSWAP)(&n_, x, &incx_, y, &incy_);
    }
    void swap(int n, std::complex<double>* x, int incx,
              std::complex<double>* y, int incy) {
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      STRUMPACK_FC_GLOBAL(zswap,ZSWAP)(&n_, x, &incx_, y, &incy_);
    }


    float nrm2(int n, const float* x, int incx) {
      STRUMPACK_FLOPS(nrm2_flops(n));
      strumpack_blas_int n_ = n, incx_ = incx;
      return STRUMPACK_FC_GLOBAL(snrm2,SNRM2)(&n_, x, &incx_);
    }
    double nrm2(int n, const double* x, int incx) {
      STRUMPACK_FLOPS(nrm2_flops(n));
      strumpack_blas_int n_ = n, incx_ = incx;
      return STRUMPACK_FC_GLOBAL(dnrm2,DNRM2)(&n_, x, &incx_);
    }
    float nrm2(int n, const std::complex<float>* x, int incx) {
      STRUMPACK_FLOPS(4*nrm2_flops(n));
      strumpack_blas_int n_ = n, incx_ = incx;
      return STRUMPACK_FC_GLOBAL(scnrm2,SCNRM2)(&n_, x, &incx_);
    }
    double nrm2(int n, const std::complex<double>* x, int incx) {
      STRUMPACK_FLOPS(4*nrm2_flops(n));
      strumpack_blas_int n_ = n, incx_ = incx;
      return STRUMPACK_FC_GLOBAL(dznrm2,DZNRM2)(&n_, x, &incx_);
    }


    float dotu(int n, const float* x, int incx, const float* y, int incy) {
      STRUMPACK_FLOPS(dot_flops(n));
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      return STRUMPACK_FC_GLOBAL(sdot,SDOTU)(&n_, x, &incx_, y, &incy_);
    }
    double dotu(int n, const double* x, int incx, const double* y, int incy) {
      STRUMPACK_FLOPS(dot_flops(n));
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      return STRUMPACK_FC_GLOBAL(ddot,DDOT)(&n_, x, &incx_, y, &incy_);
    }
    float dotc(int n, const float* x, int incx, const float* y, int incy) {
      STRUMPACK_FLOPS(dot_flops(n));
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      return STRUMPACK_FC_GLOBAL(sdot,SDOT)(&n_, x, &incx_, y, &incy_);
    }
    double dotc(int n, const double* x, int incx, const double* y, int incy) {
      STRUMPACK_FLOPS(dot_flops(n));
      strumpack_blas_int n_ = n, incx_ = incx, incy_ = incy;
      return STRUMPACK_FC_GLOBAL(ddot,DDOT)(&n_, x, &incx_, y, &incy_);
    }

    std::complex<float> dotu(int n, const std::complex<float>* x, int incx,
                             const std::complex<float>* y, int incy) {
      STRUMPACK_FLOPS(4*dot_flops(n));
      std::complex<float> r(0.);
      for (int i=0; i<n; i++)
        r += x[i*incx]*y[i*incy];
      return r;
    }
    std::complex<double> dotu(int n, const std::complex<double>* x, int incx,
                              const std::complex<double>* y, int incy) {
      STRUMPACK_FLOPS(4*dot_flops(n));
      std::complex<double> r(0.);
      for (int i=0; i<n; i++)
        r += x[i*incx]*y[i*incy];
      return r;
    }
    std::complex<float> dotc(int n, const std::complex<float>* x, int incx,
                             const std::complex<float>* y, int incy) {
      STRUMPACK_FLOPS(4*dot_flops(n));
      std::complex<float> r(0.);
      for (int i=0; i<n; i++)
        r += std::conj(x[i*incx])*y[i*incy];
      return r;
    }
    std::complex<double> dotc(int n, const std::complex<double>* x, int incx,
                              const std::complex<double>* y, int incy) {
      STRUMPACK_FLOPS(4*dot_flops(n));
      std::complex<double> r(0.);
      for (int i=0; i<n; i++)
        r += std::conj(x[i*incx])*y[i*incy];
      return r;
    }


#if defined(STRUMPACK_USE_BLAS64)
    void laswp(int n, float* a, int lda, int k1, int k2, const int* ipiv, int incx) {
      strumpack_blas_int n_ = n, lda_ = lda,
        k1_ = 1, k2_ = k2-k1+1, incx_ = (incx > 0) ? 1 : -1;
      std::vector<strumpack_blas_int> ipiv_(k2_);
      for (int i=0; i<k2_; i++) ipiv_[i] = ipiv[k1-1+incx_*i*incx];
      STRUMPACK_FC_GLOBAL(slaswp,SLASWP)(&n_, a, &lda_, &k1_, &k2_, ipiv_.data(), &incx_);
      STRUMPACK_BYTES(4*laswp_moves(n,k1,k2));
    }
    void laswp(int n, double* a, int lda, int k1, int k2, const int* ipiv, int incx) {
      strumpack_blas_int n_ = n, lda_ = lda,
        k1_ = 1, k2_ = k2-k1+1, incx_ = (incx > 0) ? 1 : -1;
      std::vector<strumpack_blas_int> ipiv_(k2_);
      for (int i=0; i<k2_; i++) ipiv_[i] = ipiv[k1-1+incx_*i*incx];
      STRUMPACK_FC_GLOBAL(dlaswp,DLASWP)(&n_, a, &lda_, &k1_, &k2_, ipiv_.data(), &incx_);
      STRUMPACK_BYTES(8*laswp_moves(n,k1,k2));
    }
    void laswp(int n, std::complex<float>* a, int lda, int k1, int k2,
               const int* ipiv, int incx) {
      strumpack_blas_int n_ = n, lda_ = lda,
        k1_ = 1, k2_ = k2-k1+1, incx_ = (incx > 0) ? 1 : -1;
      std::vector<strumpack_blas_int> ipiv_(k2_);
      for (int i=0; i<k2_; i++) ipiv_[i] = ipiv[k1-1+incx_*i*incx];
      STRUMPACK_FC_GLOBAL(claswp,CLASWP)(&n_, a, &lda_, &k1_, &k2_, ipiv_.data(), &incx_);
      STRUMPACK_BYTES(2*4*laswp_moves(n,k1,k2));
    }
    void laswp(int n, std::complex<double>* a, int lda, int k1, int k2,
               const int* ipiv, int incx) {
      strumpack_blas_int n_ = n, lda_ = lda,
        k1_ = 1, k2_ = k2-k1+1, incx_ = (incx > 0) ? 1 : -1;
      std::vector<strumpack_blas_int> ipiv_(k2_);
      for (int i=0; i<k2_; i++) ipiv_[i] = ipiv[k1-1+incx_*i*incx];
      STRUMPACK_FC_GLOBAL(zlaswp,ZLASWP)(&n_, a, &lda_, &k1_, &k2_, ipiv_.data(), &incx_);
      STRUMPACK_BYTES(2*8*laswp_moves(n,k1,k2));
    }
#else
    void laswp(int n, float* a, int lda, int k1, int k2, const int* ipiv, int incx) {
      STRUMPACK_FC_GLOBAL(slaswp,SLASWP)(&n, a, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(4*laswp_moves(n,k1,k2));
    }
    void laswp(int n, double* a, int lda, int k1, int k2, const int* ipiv, int incx) {
      STRUMPACK_FC_GLOBAL(dlaswp,DLASWP)(&n, a, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(8*laswp_moves(n,k1,k2));
    }
    void laswp(int n, std::complex<float>* a, int lda, int k1, int k2,
               const int* ipiv, int incx) {
      STRUMPACK_FC_GLOBAL(claswp,CLASWP)(&n, a, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(2*4*laswp_moves(n,k1,k2));
    }
    void laswp(int n, std::complex<double>* a, int lda, int k1, int k2,
               const int* ipiv, int incx) {
      STRUMPACK_FC_GLOBAL(zlaswp,ZLASWP)(&n, a, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(2*8*laswp_moves(n,k1,k2));
    }
#endif


#if defined(STRUMPACK_USE_BLAS64)
    void lapmr(bool fwd, int m, int n, float* a, int lda, const int* ipiv) {
      strumpack_blas_int forward = fwd ? 1 : 0, m_ = m, n_ = n, lda_ = lda;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+m);
      STRUMPACK_FC_GLOBAL(myslapmr,MYSLAPMR)(&forward, &m_, &n_, a, &lda_, ipiv_.data());
      STRUMPACK_BYTES(4*lapmr_moves(n,m));
    }
    void lapmr(bool fwd, int m, int n, double* a, int lda, const int* ipiv) {
      strumpack_blas_int forward = fwd ? 1 : 0, m_ = m, n_ = n, lda_ = lda;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+m);
      STRUMPACK_FC_GLOBAL(mydlapmr,MYDLAPMR)(&forward, &m_, &n_, a, &lda_, ipiv_.data());
      STRUMPACK_BYTES(8*lapmr_moves(n,m));
    }
    void lapmr(bool fwd, int m, int n, std::complex<float>* a, int lda, const int* ipiv) {
      strumpack_blas_int forward = fwd ? 1 : 0, m_ = m, n_ = n, lda_ = lda;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+m);
      STRUMPACK_FC_GLOBAL(myclapmr,MYCLAPMR)(&forward, &m_, &n_, a, &lda_, ipiv_.data());
      STRUMPACK_BYTES(2*4*lapmr_moves(n,m));
    }
    void lapmr(bool fwd, int m, int n, std::complex<double>* a, int lda, const int* ipiv) {
      strumpack_blas_int forward = fwd ? 1 : 0, m_ = m, n_ = n, lda_ = lda;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+m);
      STRUMPACK_FC_GLOBAL(myzlapmr,MYZLAPMR)(&forward, &m_, &n_, a, &lda_, ipiv_.data());
      STRUMPACK_BYTES(2*8*lapmr_moves(n,m));
    }
#else
    void lapmr(bool fwd, int m, int n, float* a, int lda, const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(myslapmr,MYSLAPMR)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(4*lapmr_moves(n,m));
    }
    void lapmr(bool fwd, int m, int n, double* a, int lda, const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(mydlapmr,MYDLAPMR)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(8*lapmr_moves(n,m));
    }
    void lapmr(bool fwd, int m, int n, std::complex<float>* a, int lda, const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(myclapmr,MYCLAPMR)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(2*4*lapmr_moves(n,m));
    }
    void lapmr(bool fwd, int m, int n, std::complex<double>* a, int lda, const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(myzlapmr,MYZLAPMR)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(2*8*lapmr_moves(n,m));
    }
#endif

#if defined(STRUMPACK_USE_BLAS64)
    void lapmt(bool fwd, int m, int n, float* a, int lda, const int* ipiv) {
      strumpack_blas_int forward = fwd ? 1 : 0, m_ = m, n_ = n, lda_ = lda;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+n);
      STRUMPACK_FC_GLOBAL(slapmt,SLAPMT)(&forward, &m_, &n_, a, &lda_, ipiv_.data());
      STRUMPACK_BYTES(4*lapmt_moves(n,m));
    }
    void lapmt(bool fwd, int m, int n, double* a, int lda, const int* ipiv) {
      strumpack_blas_int forward = fwd ? 1 : 0, m_ = m, n_ = n, lda_ = lda;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+n);
      STRUMPACK_FC_GLOBAL(dlapmt,DLAPMT)(&forward, &m_, &n_, a, &lda_, ipiv_.data());
      STRUMPACK_BYTES(8*lapmt_moves(n,m));
    }
    void lapmt(bool fwd, int m, int n, std::complex<float>* a, int lda,
               const int* ipiv) {
      strumpack_blas_int forward = fwd ? 1 : 0, m_ = m, n_ = n, lda_ = lda;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+n);
      STRUMPACK_FC_GLOBAL(clapmt,CLAPMT)(&forward, &m_, &n_, a, &lda_, ipiv_.data());
      STRUMPACK_BYTES(2*4*lapmt_moves(n,m));
    }
    void lapmt(bool fwd, int m, int n, std::complex<double>* a, int lda,
               const int* ipiv) {
      strumpack_blas_int forward = fwd ? 1 : 0, m_ = m, n_ = n, lda_ = lda;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+n);
      STRUMPACK_FC_GLOBAL(zlapmt,ZLAPMT)(&forward, &m_, &n_, a, &lda_, ipiv_.data());
      STRUMPACK_BYTES(2*8*lapmt_moves(n,m));
    }
#else
    void lapmt(bool fwd, int m, int n, float* a, int lda, const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(slapmt,SLAPMT)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(4*lapmt_moves(n,m));
    }
    void lapmt(bool fwd, int m, int n, double* a, int lda, const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(dlapmt,DLAPMT)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(8*lapmt_moves(n,m));
    }
    void lapmt(bool fwd, int m, int n, std::complex<float>* a, int lda,
               const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(clapmt,CLAPMT)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(2*4*lapmt_moves(n,m));
    }
    void lapmt(bool fwd, int m, int n, std::complex<double>* a, int lda,
               const int* ipiv) {
      int forward = fwd ? 1 : 0;
      STRUMPACK_FC_GLOBAL(zlapmt,ZLAPMT)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(2*8*lapmt_moves(n,m));
    }
#endif

    void trsm(char s, char ul, char t, char d, int m, int n, float alpha,
              const float* a, int lda, float* b, int ldb) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, ldb_ = ldb;
      STRUMPACK_FC_GLOBAL(strsm,STRSM)
        (&s, &ul, &t, &d, &m_, &n_, &alpha, a, &lda_, b, &ldb_);
      STRUMPACK_FLOPS(trsm_flops(m, n, alpha, s));
      STRUMPACK_BYTES(4*trsm_moves(m, n));
    }
    void trsm(char s, char ul, char t, char d, int m, int n, double alpha,
              const double* a, int lda, double* b, int ldb) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, ldb_ = ldb;
      STRUMPACK_FC_GLOBAL(dtrsm,DTRSM)
        (&s, &ul, &t, &d, &m_, &n_, &alpha, a, &lda_, b, &ldb_);
      STRUMPACK_FLOPS(trsm_flops(m, n, alpha, s));
      STRUMPACK_BYTES(8*trsm_moves(m, n));
    }
    void trsm(char s, char ul, char t, char d, int m, int n,
              std::complex<float> alpha, const std::complex<float>* a, int lda,
              std::complex<float>* b, int ldb) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, ldb_ = ldb;
      STRUMPACK_FC_GLOBAL(ctrsm,CTRSM)
        (&s, &ul, &t, &d, &m_, &n_, &alpha, a, &lda_, b, &ldb_);
      STRUMPACK_FLOPS(4*trsm_flops(m, n, alpha, s));
      STRUMPACK_BYTES(2*4*trsm_moves(m, n));
    }
    void trsm(char s, char ul, char t, char d, int m, int n,
              std::complex<double> alpha, const std::complex<double>* a, int lda,
              std::complex<double>* b, int ldb) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, ldb_ = ldb;
      STRUMPACK_FC_GLOBAL(ztrsm,ZTRSM)
        (&s, &ul, &t, &d, &m_, &n_, &alpha, a, &lda_, b, &ldb_);
      STRUMPACK_FLOPS(4*trsm_flops(m, n, alpha, s));
      STRUMPACK_BYTES(2*8*trsm_moves(m, n));
    }

    void trmm(char s, char ul, char t, char d, int m, int n, float alpha,
              const float* a, int lda, float* b, int ldb) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, ldb_ = ldb;
      STRUMPACK_FC_GLOBAL(strmm,STRMM)
        (&s, &ul, &t, &d, &m_, &n_, &alpha, a, &lda_, b, &ldb_);
      STRUMPACK_FLOPS(trmm_flops(m,n,alpha,s));
      STRUMPACK_BYTES(4*trmm_moves(m,n));
    }
    void trmm(char s, char ul, char t, char d, int m, int n, double alpha,
              const double* a, int lda, double* b, int ldb) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, ldb_ = ldb;
      STRUMPACK_FC_GLOBAL(dtrmm,DTRMM)
        (&s, &ul, &t, &d, &m_, &n_, &alpha, a, &lda_, b, &ldb_);
      STRUMPACK_FLOPS(trmm_flops(m,n,alpha,s));
      STRUMPACK_BYTES(8*trmm_moves(m,n));
    }
    void trmm(char s, char ul, char t, char d, int m, int n, std::complex<float> alpha,
              const std::complex<float>* a, int lda, std::complex<float>* b, int ldb) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, ldb_ = ldb;
      STRUMPACK_FC_GLOBAL(ctrmm,CTRMM)
        (&s, &ul, &t, &d, &m_, &n_, &alpha, a, &lda_, b, &ldb_);
      STRUMPACK_FLOPS(4*trmm_flops(m,n,alpha,s));
      STRUMPACK_BYTES(2*4*trmm_moves(m,n));
    }
    void trmm(char s, char ul, char t, char d, int m, int n,
              std::complex<double> alpha, const std::complex<double>* a, int lda,
              std::complex<double>* b, int ldb) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda, ldb_ = ldb;
      STRUMPACK_FC_GLOBAL(ztrmm,ZTRMM)
        (&s, &ul, &t, &d, &m_, &n_, &alpha, a, &lda_, b, &ldb_);
      STRUMPACK_FLOPS(4*trmm_flops(m,n,alpha,s));
      STRUMPACK_BYTES(2*8*trmm_moves(m,n));
    }


    void trmv(char ul, char t, char d, int n, const float* a, int lda,
              float* x, int incx) {
      strumpack_blas_int n_ = n, lda_ = lda, incx_ = incx;
      STRUMPACK_FC_GLOBAL(strmv,STRMV)(&ul, &t, &d, &n_, a, &lda_, x, &incx_);
      STRUMPACK_FLOPS(trmv_flops(n));
      STRUMPACK_BYTES(4*trmv_moves(n));
    }
    void trmv(char ul, char t, char d, int n, const double* a, int lda,
              double* x, int incx) {
      strumpack_blas_int n_ = n, lda_ = lda, incx_ = incx;
      STRUMPACK_FC_GLOBAL(dtrmv,DTRMV)(&ul, &t, &d, &n_, a, &lda_, x, &incx_);
      STRUMPACK_FLOPS(trmv_flops(n));
      STRUMPACK_BYTES(8*trmv_moves(n));
    }
    void trmv(char ul, char t, char d, int n, const std::complex<float>* a, int lda,
              std::complex<float>* x, int incx) {
      strumpack_blas_int n_ = n, lda_ = lda, incx_ = incx;
      STRUMPACK_FC_GLOBAL(ctrmv,CTRMV)(&ul, &t, &d, &n_, a, &lda_, x, &incx_);
      STRUMPACK_FLOPS(4*trmv_flops(n));
      STRUMPACK_BYTES(2*4*trmv_moves(n));
    }
    void trmv(char ul, char t, char d, int n, const std::complex<double>* a, int lda,
              std::complex<double>* x, int incx) {
      strumpack_blas_int n_ = n, lda_ = lda, incx_ = incx;
      STRUMPACK_FC_GLOBAL(ztrmv,ZTRMV)(&ul, &t, &d, &n_, a, &lda_, x, &incx_);
      STRUMPACK_FLOPS(4*trmv_flops(n));
      STRUMPACK_BYTES(2*8*trmv_moves(n));
    }


    void trsv(char ul, char t, char d, int m, const float* a, int lda,
              float* b, int incb) {
      strumpack_blas_int m_ = m, lda_ = lda, incb_ = incb;
      STRUMPACK_FC_GLOBAL(strsv,STRSV)(&ul, &t, &d, &m_, a, &lda_, b, &incb_);
      STRUMPACK_FLOPS(trsv_flops(m));
      STRUMPACK_BYTES(4*trsv_moves(m));
    }
    void trsv(char ul, char t, char d, int m, const double* a, int lda,
              double* b, int incb) {
      strumpack_blas_int m_ = m, lda_ = lda, incb_ = incb;
      STRUMPACK_FC_GLOBAL(dtrsv,DTRSV)(&ul, &t, &d, &m_, a, &lda_, b, &incb_);
      STRUMPACK_FLOPS(trsv_flops(m));
      STRUMPACK_BYTES(8*trsv_moves(m));
    }
    void trsv(char ul, char t, char d, int m, const std::complex<float>* a, int lda,
              std::complex<float>* b, int incb) {
      strumpack_blas_int m_ = m, lda_ = lda, incb_ = incb;
      STRUMPACK_FC_GLOBAL(ctrsv,CTRSV)(&ul, &t, &d, &m_, a, &lda_, b, &incb_);
      STRUMPACK_FLOPS(4*trsv_flops(m));
      STRUMPACK_BYTES(2*4*trsv_moves(m));
    }
    void trsv(char ul, char t, char d, int m, const std::complex<double>* a, int lda,
              std::complex<double>* b, int incb) {
      strumpack_blas_int m_ = m, lda_ = lda, incb_ = incb;
      STRUMPACK_FC_GLOBAL(ztrsv,ZTRSV)(&ul, &t, &d, &m_, a, &lda_, b, &incb_);
      STRUMPACK_FLOPS(4*trsv_flops(m));
      STRUMPACK_BYTES(2*8*trsv_moves(m));
    }


    void laset(char s, int m, int n, float alpha, float beta, float* x, int ldx) {
      strumpack_blas_int m_ = m, n_ = n, ldx_ = ldx;
      STRUMPACK_FC_GLOBAL(slaset,SLASET)(&s, &m_, &n_, &alpha, &beta, x, &ldx_);
    }
    void laset(char s, int m, int n, double alpha, double beta, double* x, int ldx) {
      strumpack_blas_int m_ = m, n_ = n, ldx_ = ldx;
      STRUMPACK_FC_GLOBAL(dlaset,DLASET)(&s, &m_, &n_, &alpha, &beta, x, &ldx_);
    }
    void laset(char s, int m, int n, std::complex<float> alpha,
               std::complex<float> beta, std::complex<float>* x, int ldx) {
      strumpack_blas_int m_ = m, n_ = n, ldx_ = ldx;
      STRUMPACK_FC_GLOBAL(claset,CLASET)(&s, &m_, &n_, &alpha, &beta, x, &ldx_);
    }
    void laset(char s, int m, int n, std::complex<double> alpha,
               std::complex<double> beta, std::complex<double>* x, int ldx) {
      strumpack_blas_int m_ = m, n_ = n, ldx_ = ldx;
      STRUMPACK_FC_GLOBAL(zlaset,ZLASET)(&s, &m_, &n_, &alpha, &beta, x, &ldx_);
    }


#if defined(STRUMPACK_USE_BLAS64)
    int geqp3(int m, int n, float* a, int lda, int* jpvt, float* tau,
              float* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
      std::vector<strumpack_blas_int> jpvt_(jpvt, jpvt+n);
      STRUMPACK_FC_GLOBAL(sgeqp3,SGEQP3)
        (&m_, &n_, a, &lda_, jpvt_.data(), tau, work, &lwork_, &info);
      std::copy(jpvt_.begin(), jpvt_.end(), jpvt);
      STRUMPACK_FLOPS(geqp3_flops(m,n));
      return info;
    }
    int geqp3(int m, int n, double* a, int lda, int* jpvt, double* tau,
              double* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
      std::vector<strumpack_blas_int> jpvt_(jpvt, jpvt+n);
      STRUMPACK_FC_GLOBAL(dgeqp3,DGEQP3)
        (&m_, &n_, a, &lda_, jpvt_.data(), tau, work, &lwork_, &info);
      std::copy(jpvt_.begin(), jpvt_.end(), jpvt);
      STRUMPACK_FLOPS(geqp3_flops(m,n));
      return info;
    }
    int geqp3(int m, int n, std::complex<float>* a, int lda, int* jpvt,
              std::complex<float>* tau, std::complex<float>* work, int lwork) {
      std::unique_ptr<std::complex<float>[]> rwork
        (new std::complex<float>[std::max(1, 2*n)]);
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
      std::vector<strumpack_blas_int> jpvt_(jpvt, jpvt+n);
      STRUMPACK_FC_GLOBAL(cgeqp3,CGEQP3)
        (&m_, &n_, a, &lda_, jpvt_.data(), tau, work, &lwork_, rwork.get(), &info);
      std::copy(jpvt_.begin(), jpvt_.end(), jpvt);
      STRUMPACK_FLOPS(4*geqp3_flops(m,n));
      return info;
    }
    int geqp3(int m, int n, std::complex<double>* a, int lda, int* jpvt,
              std::complex<double>* tau, std::complex<double>* work,
              int lwork) {
      std::unique_ptr<std::complex<double>[]> rwork
        (new std::complex<double>[std::max(1, 2*n)]);
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
      std::vector<strumpack_blas_int> jpvt_(jpvt, jpvt+n);
      STRUMPACK_FC_GLOBAL(zgeqp3,ZGEQP3)
        (&m_, &n_, a, &lda_, jpvt_.data(), tau, work, &lwork_, rwork.get(), &info);
      std::copy(jpvt_.begin(), jpvt_.end(), jpvt);
      STRUMPACK_FLOPS(4*geqp3_flops(m,n));
      return info;
    }
#else
    int geqp3(int m, int n, float* a, int lda, int* jpvt, float* tau,
              float* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(sgeqp3,SGEQP3)(&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(geqp3_flops(m,n));
      return info;
    }
    int geqp3(int m, int n, double* a, int lda, int* jpvt, double* tau,
              double* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(dgeqp3,DGEQP3)(&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
      STRUMPACK_FLOPS(geqp3_flops(m,n));
      return info;
    }
    int geqp3(int m, int n, std::complex<float>* a, int lda, int* jpvt,
              std::complex<float>* tau, std::complex<float>* work, int lwork) {
      std::unique_ptr<std::complex<float>[]> rwork
        (new std::complex<float>[std::max(1, 2*n)]);
      int info;
      STRUMPACK_FC_GLOBAL(cgeqp3,CGEQP3)(&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork.get(), &info);
      STRUMPACK_FLOPS(4*geqp3_flops(m,n));
      return info;
    }
    int geqp3(int m, int n, std::complex<double>* a, int lda, int* jpvt,
              std::complex<double>* tau, std::complex<double>* work,
              int lwork) {
      std::unique_ptr<std::complex<double>[]> rwork
        (new std::complex<double>[std::max(1, 2*n)]);
      int info;
      STRUMPACK_FC_GLOBAL(zgeqp3,ZGEQP3)(&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork.get(), &info);
      STRUMPACK_FLOPS(4*geqp3_flops(m,n));
      return info;
    }
#endif


    int geqp3tol(int m, int n, float* a, int lda, int* jpvt, float* tau, float* work,
                 int lwork, int& rank, float rtol, float atol) {
#if defined(STRUMPACK_USE_BLAS64)
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork, rank_ = rank;
      std::vector<strumpack_blas_int> jpvt_(jpvt, jpvt+n);
      STRUMPACK_FC_GLOBAL(sgeqp3tol,SGEQP3TOL)
        (&m_, &n_, a, &lda_, jpvt_.data(), tau, work, &lwork_, &info, &rank_, &rtol, &atol);
      std::copy(jpvt_.begin(), jpvt_.end(), jpvt);
      rank = rank_;
#else
      int info;
      STRUMPACK_FC_GLOBAL(sgeqp3tol,SGEQP3TOL)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, &info, &rank, &rtol, &atol);
#endif
      return info;
    }
    int geqp3tol(int m, int n, double* a, int lda, int* jpvt, double* tau, double* work,
                 int lwork, int& rank, double rtol, double atol) {
#if defined(STRUMPACK_USE_BLAS64)
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork, rank_ = rank;
      std::vector<strumpack_blas_int> jpvt_(jpvt, jpvt+n);
      STRUMPACK_FC_GLOBAL(dgeqp3tol,DGEQP3TOL)
        (&m_, &n_, a, &lda_, jpvt_.data(), tau, work, &lwork_, &info, &rank_, &rtol, &atol);
      std::copy(jpvt_.begin(), jpvt_.end(), jpvt);
      rank = rank_;
#else
      int info;
      STRUMPACK_FC_GLOBAL(dgeqp3tol,DGEQP3TOL)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, &info, &rank, &rtol, &atol);
#endif
      return info;
    }
    int geqp3tol(int m, int n, std::complex<float>* a, int lda, int* jpvt,
                 std::complex<float>* tau, std::complex<float>* work, int lwork,
                 int& rank, float rtol, float atol) {
      std::unique_ptr<float[]> rwork(new float[std::max(1, 2*n)]);
      for (int i=0; i<n; i++)
        rwork[i] = nrm2(m, &a[i*lda], 1);
#if defined(STRUMPACK_USE_BLAS64)
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork, rank_ = rank;
      std::vector<strumpack_blas_int> jpvt_(jpvt, jpvt+n);
      STRUMPACK_FC_GLOBAL(cgeqp3tol,CGEQP3TOL)
        (&m_, &n_, a, &lda_, jpvt_.data(), tau, work, &lwork_, rwork.get(), &info, &rank_, &rtol, &atol);
      std::copy(jpvt_.begin(), jpvt_.end(), jpvt);
      rank = rank_;
#else
      int info;
      STRUMPACK_FC_GLOBAL(cgeqp3tol,CGEQP3TOL)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork.get(), &info, &rank, &rtol, &atol);
#endif
      return info;
    }
    int geqp3tol(int m, int n, std::complex<double>* a, int lda, int* jpvt,
                 std::complex<double>* tau, std::complex<double>* work, int lwork,
                 int& rank, double rtol, double atol) {
      std::unique_ptr<double[]> rwork(new double[std::max(1, 2*n)]);
      for (int i=0; i<n; i++)
        rwork[i] = nrm2(m, &a[i*lda], 1);
#if defined(STRUMPACK_USE_BLAS64)
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork, rank_ = rank;
      std::vector<strumpack_blas_int> jpvt_(jpvt, jpvt+n);
      STRUMPACK_FC_GLOBAL(zgeqp3tol,ZGEQP3TOL)
        (&m_, &n_, a, &lda_, jpvt_.data(), tau, work, &lwork_, rwork.get(), &info, &rank_, &rtol, &atol);
      std::copy(jpvt_.begin(), jpvt_.end(), jpvt);
      rank = rank_;
#else
      int info;
      STRUMPACK_FC_GLOBAL(zgeqp3tol,ZGEQP3TOL)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork.get(), &info, &rank, &rtol, &atol);
#endif
      return info;
    }


    int geqrf(int m, int n, float* a, int lda, float* tau, float* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(sgeqrf,SGEQRF)(&m_, &n_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(geqrf_flops(m,n));
      return info;
    }
    int geqrf(int m, int n, double* a, int lda, double* tau, double* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(dgeqrf,DGEQRF)(&m_, &n_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(geqrf_flops(m,n));
      return info;
    }
    int geqrf(int m, int n, std::complex<float>* a, int lda, std::complex<float>* tau,
              std::complex<float>* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(cgeqrf,CGEQRF)(&m_, &n_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(4*geqrf_flops(m,n));
      return info;
    }
    int geqrf(int m, int n, std::complex<double>* a, int lda,
              std::complex<double>* tau, std::complex<double>* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(zgeqrf,ZGEQRF)(&m_, &n_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(4*geqrf_flops(m,n));
      return info;
    }


    int gelqf(int m, int n, float* a, int lda, float* tau, float* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(sgelqf,SGELQF)(&m_, &n_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(gelqf_flops(m,n));
      return info;
    }
    int gelqf(int m, int n, double* a, int lda, double* tau, double* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(dgelqf,DGELQF)(&m_, &n_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(gelqf_flops(m,n));
      return info;
    }
    int gelqf(int m, int n, std::complex<float>* a, int lda, std::complex<float>* tau,
              std::complex<float>* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(cgelqf,CGELQF)(&m_, &n_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(4*gelqf_flops(m,n));
      return info;
    }
    int gelqf(int m, int n, std::complex<double>* a, int lda,
              std::complex<double>* tau, std::complex<double>* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(zgelqf,ZGELQF)(&m_, &n_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(4*gelqf_flops(m,n));
      return info;
    }


#if defined(STRUMPACK_USE_BLAS64)
    int getrf(int m, int n, float* a, int lda, int* ipiv) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, minmn = std::min(m, n);
      std::vector<strumpack_blas_int> ipiv_(minmn);
      STRUMPACK_FC_GLOBAL(sgetrf,SGETRF)(&m_, &n_, a, &lda_, ipiv_.data(), &info);
      std::copy(ipiv_.begin(), ipiv_.end(), ipiv);
      STRUMPACK_FLOPS(getrf_flops(m,n));
      return info;
    }
    int getrf(int m, int n, double* a, int lda, int* ipiv) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, minmn = std::min(m, n);
      std::vector<strumpack_blas_int> ipiv_(minmn);
      STRUMPACK_FC_GLOBAL(dgetrf,DGETRF)(&m_, &n_, a, &lda_, ipiv_.data(), &info);
      std::copy(ipiv_.begin(), ipiv_.end(), ipiv);
      STRUMPACK_FLOPS(getrf_flops(m,n));
      return info;
    }
    int getrf(int m, int n, std::complex<float>* a, int lda, int* ipiv) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, minmn = std::min(m, n);
      std::vector<strumpack_blas_int> ipiv_(minmn);
      STRUMPACK_FC_GLOBAL(cgetrf,CGETRF)(&m_, &n_, a, &lda_, ipiv_.data(), &info);
      std::copy(ipiv_.begin(), ipiv_.end(), ipiv);
      STRUMPACK_FLOPS(4*getrf_flops(m,n));
      return info;
    }
    int getrf(int m, int n, std::complex<double>* a, int lda, int* ipiv) {
      strumpack_blas_int info, m_ = m, n_ = n, lda_ = lda, minmn = std::min(m, n);
      std::vector<strumpack_blas_int> ipiv_(minmn);
      STRUMPACK_FC_GLOBAL(zgetrf,ZGETRF)(&m_, &n_, a, &lda_, ipiv_.data(), &info);
      std::copy(ipiv_.begin(), ipiv_.end(), ipiv);
      STRUMPACK_FLOPS(4*getrf_flops(m,n));
      return info;
    }
#else
    int getrf(int m, int n, float* a, int lda, int* ipiv) {
      int info;
      STRUMPACK_FC_GLOBAL(sgetrf,SGETRF)(&m, &n, a, &lda, ipiv, &info);
      STRUMPACK_FLOPS(getrf_flops(m,n));
      return info;
    }
    int getrf(int m, int n, double* a, int lda, int* ipiv) {
      int info;
      STRUMPACK_FC_GLOBAL(dgetrf,DGETRF)(&m, &n, a, &lda, ipiv, &info);
      STRUMPACK_FLOPS(getrf_flops(m,n));
      return info;
    }
    int getrf(int m, int n, std::complex<float>* a, int lda, int* ipiv) {
      int info;
      STRUMPACK_FC_GLOBAL(cgetrf,CGETRF)(&m, &n, a, &lda, ipiv, &info);
      STRUMPACK_FLOPS(4*getrf_flops(m,n));
      return info;
    }
    int getrf(int m, int n, std::complex<double>* a, int lda, int* ipiv) {
      int info;
      STRUMPACK_FC_GLOBAL(zgetrf,ZGETRF)(&m, &n, a, &lda, ipiv, &info);
      STRUMPACK_FLOPS(4*getrf_flops(m,n));
      return info;
    }
#endif


#if defined(STRUMPACK_USE_BLAS64)
    int getrs(char t, int n, int nrhs, const float* a, int lda,
              const int* ipiv, float* b, int ldb) {
      strumpack_blas_int info, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+n);
      STRUMPACK_FC_GLOBAL(sgetrs,SGETRS)(&t, &n_, &nrhs_, a, &lda_, ipiv_.data(), b, &ldb_, &info);
      STRUMPACK_FLOPS(getrs_flops(n, nrhs));
      return info;
    }
    int getrs(char t, int n, int nrhs, const double* a, int lda,
              const int* ipiv, double* b, int ldb) {
      strumpack_blas_int info, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+n);
      STRUMPACK_FC_GLOBAL(dgetrs,DGETRS)(&t, &n_, &nrhs_, a, &lda_, ipiv_.data(), b, &ldb_, &info);
      STRUMPACK_FLOPS(getrs_flops(n, nrhs));
      return info;
    }
    int getrs(char t, int n, int nrhs, const std::complex<float>* a, int lda,
              const int* ipiv, std::complex<float>* b, int ldb) {
      strumpack_blas_int info, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+n);
      STRUMPACK_FC_GLOBAL(cgetrs,CGETRS)(&t, &n_, &nrhs_, a, &lda_, ipiv_.data(), b, &ldb_, &info);
      STRUMPACK_FLOPS(4*getrs_flops(n, nrhs));
      return info;
    }
    int getrs(char t, int n, int nrhs, const std::complex<double>* a, int lda,
              const int* ipiv, std::complex<double>* b, int ldb) {
      strumpack_blas_int info, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+n);
      STRUMPACK_FC_GLOBAL(zgetrs,ZGETRS)(&t, &n_, &nrhs_, a, &lda_, ipiv_.data(), b, &ldb_, &info);
      STRUMPACK_FLOPS(4*getrs_flops(n, nrhs));
      return info;
    }
#else
    int getrs(char t, int n, int nrhs, const float* a, int lda,
              const int* ipiv, float* b, int ldb) {
      int info;
      STRUMPACK_FC_GLOBAL(sgetrs,SGETRS)(&t, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      STRUMPACK_FLOPS(getrs_flops(n, nrhs));
      return info;
    }
    int getrs(char t, int n, int nrhs, const double* a, int lda,
              const int* ipiv, double* b, int ldb) {
      int info;
      STRUMPACK_FC_GLOBAL(dgetrs,DGETRS)(&t, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      STRUMPACK_FLOPS(getrs_flops(n, nrhs));
      return info;
    }
    int getrs(char t, int n, int nrhs, const std::complex<float>* a, int lda,
              const int* ipiv, std::complex<float>* b, int ldb) {
      int info;
      STRUMPACK_FC_GLOBAL(cgetrs,CGETRS)(&t, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      STRUMPACK_FLOPS(4*getrs_flops(n, nrhs));
      return info;
    }
    int getrs(char t, int n, int nrhs, const std::complex<double>* a, int lda,
              const int* ipiv, std::complex<double>* b, int ldb) {
      int info;
      STRUMPACK_FC_GLOBAL(zgetrs,ZGETRS)(&t, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      STRUMPACK_FLOPS(4*getrs_flops(n, nrhs));
      return info;
    }
#endif


    int potrf(char ul, int n, float* a, int lda) {
      strumpack_blas_int info, n_ = n, lda_ = lda;
      STRUMPACK_FC_GLOBAL(spotrf,SPOTRF)(&ul, &n_, a, &lda_, &info);
      STRUMPACK_FLOPS(potrf_flops(n));
      return info;
    }
    int potrf(char ul, int n, double* a, int lda) {
      strumpack_blas_int info, n_ = n, lda_ = lda;
      STRUMPACK_FC_GLOBAL(dpotrf,DPOTRF)(&ul, &n_, a, &lda_, &info);
      STRUMPACK_FLOPS(potrf_flops(n));
      return info;
    }
    int potrf(char ul, int n, std::complex<float>* a, int lda) {
      strumpack_blas_int info, n_ = n, lda_ = lda;
      STRUMPACK_FC_GLOBAL(cpotrf,CPOTRF)(&ul, &n_, a, &lda_, &info);
      STRUMPACK_FLOPS(4*potrf_flops(n));
      return info;
    }
    int potrf(char ul, int n, std::complex<double>* a, int lda) {
      strumpack_blas_int info, n_ = n, lda_ = lda;
      STRUMPACK_FC_GLOBAL(zpotrf,ZPOTRF)(&ul, &n_, a, &lda_, &info);
      STRUMPACK_FLOPS(4*potrf_flops(n));
      return info;
    }


    int xxglq(int m, int n, int k, float* a, int lda, const float* tau,
              float* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, k_ = k, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(sorglq,STRUMPACK_FC_GLOBAL)
        (&m_, &n_, &k_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(xxglq_flops(m,n,k));
      return info;
    }
    int xxglq(int m, int n, int k, double* a, int lda, const double* tau,
              double* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, k_ = k, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(dorglq,DORGLQ)
        (&m_, &n_, &k_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(xxglq_flops(m,n,k));
      return info;
    }
    int xxglq(int m, int n, int k, std::complex<float>* a, int lda,
              const std::complex<float>* tau, std::complex<float>* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, k_ = k, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(cunglq,CUNGLQ)
        (&m_, &n_, &k_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(4*xxglq_flops(m,n,k));
      return info;
    }
    int xxglq(int m, int n, int k, std::complex<double>* a, int lda,
              const std::complex<double>* tau, std::complex<double>* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, k_ = k, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(zunglq,ZUNGLQ)
        (&m_, &n_, &k_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(4*xxglq_flops(m,n,k));
      return info;
    }


    int xxgqr(int m, int n, int k, float* a, int lda, const float* tau,
              float* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, k_ = k, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(sorgqr,SORGQR)
        (&m_, &n_, &k_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(xxgqr_flops(m,n,k));
      return info;
    }
    int xxgqr(int m, int n, int k, double* a, int lda, const double* tau,
              double* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, k_ = k, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(dorgqr,DORGQR)
        (&m_, &n_, &k_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(xxgqr_flops(m,n,k));
      return info;
    }
    int xxgqr(int m, int n, int k, std::complex<float>* a, int lda,
              const std::complex<float>* tau, std::complex<float>* work,
              int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, k_ = k, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(cungqr,CUNGQR)
        (&m_, &n_, &k_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(4*xxgqr_flops(m,n,k));
      return info;
    }
    int xxgqr(int m, int n, int k, std::complex<double>* a, int lda,
              const std::complex<double>* tau, std::complex<double>* work,
              int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, k_ = k, lda_ = lda, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(zungqr,ZUNGQR)
        (&m_, &n_, &k_, a, &lda_, tau, work, &lwork_, &info);
      STRUMPACK_FLOPS(4*xxgqr_flops(m,n,k));
      return info;
    }


    int xxmqr(char side, char trans, int m, int n, int k, float* a, int lda,
              const float* tau, float* c, int ldc, float* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, k_ = k, lda_ = lda, ldc_ = ldc, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(sormqr,SORMQR)
        (&side, &trans, &m_, &n_, &k_, a, &lda_, tau, c, &ldc_, work, &lwork_, &info);
      STRUMPACK_FLOPS(xxmqr_flops(m,n,k));
      return info;
    }
    int xxmqr(char side, char trans, int m, int n, int k, double* a, int lda,
              const double* tau, double* c, int ldc, double* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, k_ = k, lda_ = lda, ldc_ = ldc, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(dormqr,DORMQR)
        (&side, &trans, &m_, &n_, &k_, a, &lda_, tau, c, &ldc_, work, &lwork_, &info);
      STRUMPACK_FLOPS(xxmqr_flops(m,n,k));
      return info;
    }
    int xxmqr(char side, char trans, int m, int n, int k, std::complex<float>* a, int lda,
              const std::complex<float>* tau, std::complex<float>* c, int ldc,
              std::complex<float>* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, k_ = k, lda_ = lda, ldc_ = ldc, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(cunmqr,CUNMQR)
        (&side, &trans, &m_, &n_, &k_, a, &lda_, tau, c, &ldc_, work, &lwork_, &info);
      STRUMPACK_FLOPS(4*xxmqr_flops(m,n,k));
      return info;
    }
    int xxmqr(char side, char trans, int m, int n, int k, std::complex<double>* a, int lda,
              const std::complex<double>* tau, std::complex<double>* c, int ldc,
              std::complex<double>* work, int lwork) {
      strumpack_blas_int info, m_ = m, n_ = n, k_ = k, lda_ = lda, ldc_ = ldc, lwork_ = lwork;
      STRUMPACK_FC_GLOBAL(zunmqr,ZUNMQR)
        (&side, &trans, &m_, &n_, &k_, a, &lda_, tau, c, &ldc_, work, &lwork_, &info);
      STRUMPACK_FLOPS(4*xxmqr_flops(m,n,k));
      return info;
    }


    int lange(char norm, int m, int n, const int *a, int lda) { return -1; }
    unsigned int lange(char norm, int m, int n, const unsigned int *a, int lda) { return 0; }
    std::size_t lange(char norm, int m, int n, const std::size_t *a, int lda) { return 0; }
    bool lange(char norm, int m, int n, const bool *a, int lda) { return false; }
    float lange(char norm, int m, int n, const float *a, int lda) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda;
      if (norm == 'I' || norm == 'i') {
        std::unique_ptr<float[]> work(new float[m]);
        return STRUMPACK_FC_GLOBAL(slange,SLANGE)(&norm, &m_, &n_, a, &lda_, work.get());
      } else return STRUMPACK_FC_GLOBAL(slange,SLANGE)(&norm, &m_, &n_, a, &lda_, nullptr);
    }
    double lange(char norm, int m, int n, const double *a, int lda) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda;
      if (norm == 'I' || norm == 'i') {
        std::unique_ptr<double[]> work(new double[m]);
        return STRUMPACK_FC_GLOBAL(dlange,DLANGE)(&norm, &m_, &n_, a, &lda_, work.get());
      } else return STRUMPACK_FC_GLOBAL(dlange,DLANGE)(&norm, &m_, &n_, a, &lda_, nullptr);
    }
    float lange(char norm, int m, int n, const std::complex<float> *a, int lda) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda;
      if (norm == 'I' || norm == 'i') {
        std::unique_ptr<float[]> work(new float[m]);
        return STRUMPACK_FC_GLOBAL(clange,CLANGE)(&norm, &m_, &n_, a, &lda_, work.get());
      } else return STRUMPACK_FC_GLOBAL(clange,CLANGE)(&norm, &m_, &n_, a, &lda_, nullptr);
    }
    double lange(char norm, int m, int n, const std::complex<double> *a, int lda) {
      strumpack_blas_int m_ = m, n_ = n, lda_ = lda;
      if (norm == 'I' || norm == 'i') {
        std::unique_ptr<double[]> work(new double[m]);
        return STRUMPACK_FC_GLOBAL(zlange,ZLANGE)(&norm, &m_, &n_, a, &lda_, work.get());
      } else return STRUMPACK_FC_GLOBAL(zlange,ZLANGE)(&norm, &m_, &n_, a, &lda_, nullptr);
    }

    int gesvd(char jobu, char jobvt, int m, int n, float* a, int lda,
              float* s, float* u, int ldu, float* vt, int ldvt) {
      strumpack_blas_int info, lwork = -1, m_ = m, n_ = n, lda_ = lda, ldu_ = ldu, ldvt_ = ldvt;
      float swork;
      STRUMPACK_FC_GLOBAL(sgesvd,SGESVD)
        (&jobu, &jobvt, &m_, &n_, a, &lda_, s, u, &ldu_, vt, &ldvt_,
         &swork, &lwork, &info);
      lwork = (int)swork;
      std::unique_ptr<float[]> work(new float[lwork]);
      STRUMPACK_FC_GLOBAL(sgesvd,SGESVD)
        (&jobu, &jobvt, &m_, &n_, a, &lda_, s, u, &ldu_, vt, &ldvt_,
         work.get(), &lwork, &info);
      return info;
    }
    int gesvd(char jobu, char jobvt, int m, int n, double* a, int lda,
              double* s, double* u, int ldu, double* vt, int ldvt) {
      strumpack_blas_int info, lwork = -1, m_ = m, n_ = n, lda_ = lda, ldu_ = ldu, ldvt_ = ldvt;
      double dwork;
      STRUMPACK_FC_GLOBAL(dgesvd,DGESVD)
        (&jobu, &jobvt, &m_, &n_, a, &lda_, s, u, &ldu_, vt, &ldvt_,
         &dwork, &lwork, &info);
      lwork = (int)dwork;
      std::unique_ptr<double[]> work(new double[lwork]);
      STRUMPACK_FC_GLOBAL(dgesvd,DGESVD)
        (&jobu, &jobvt, &m_, &n_, a, &lda_, s, u, &ldu_, vt, &ldvt_,
         work.get(), &lwork, &info);
      return info;
    }
    int gesvd(char jobu, char jobvt, int m, int n, std::complex<float>* a, int lda,
              std::complex<float>* s, std::complex<float>* u, int ldu,
              std::complex<float>* vt, int ldvt) {
      std::cout << "TODO gesvd for std::complex<float>" << std::endl;
      return 0;
    }
    int gesvd(char jobu, char jobvt, int m, int n, std::complex<double>* a, int lda,
              std::complex<double>* s, std::complex<double>* u, int ldu,
              std::complex<double>* vt, int ldvt) {
      std::cout << "TODO gesvd for std::complex<double>" << std::endl;
      return 0;
    }

    int syevx(char jobz, char range, char uplo, int n, float* a, int lda,
              float vl, float vu, int il, int iu, float abstol, int& m,
              float* w, float* z, int ldz) {
      strumpack_blas_int info, lwork = -1, n_ = n, lda_ = lda, il_ = il, iu_ = iu, m_ = m, ldz_ = ldz;
      std::unique_ptr<strumpack_blas_int[]> iwork(new strumpack_blas_int[5*n+n]);
      auto ifail = iwork.get()+5*n;
      float swork;
      STRUMPACK_FC_GLOBAL(ssyevx,SSYEVX)
        (&jobz, &range, &uplo, &n_, a, &lda_, &vl, &vu, &il_, &iu_, &abstol, &m_,
         w, z, &ldz_, &swork, &lwork, iwork.get(), ifail, &info);
      lwork = (strumpack_blas_int)swork;
      std::unique_ptr<float[]> work(new float[lwork]);
      STRUMPACK_FC_GLOBAL(ssyevx,SSYEVX)
        (&jobz, &range, &uplo, &n_, a, &lda_, &vl, &vu, &il_, &iu_, &abstol, &m_,
         w, z, &ldz_, work.get(), &lwork, iwork.get(), ifail, &info);
      m = m_;
      return info;
    }
    int syevx(char jobz, char range, char uplo, int n, double* a, int lda,
              double vl, double vu, int il, int iu, double abstol, int& m,
              double* w, double* z, int ldz) {
      strumpack_blas_int info, lwork = -1, n_ = n, lda_ = lda, il_ = il, iu_ = iu, m_ = m, ldz_ = ldz;
      std::unique_ptr<strumpack_blas_int[]> iwork(new strumpack_blas_int[5*n+n]);
      auto ifail = iwork.get()+5*n;
      double dwork;
      STRUMPACK_FC_GLOBAL(dsyevx,DSYEVX)
        (&jobz, &range, &uplo, &n_, a, &lda_, &vl, &vu, &il_, &iu_, &abstol, &m_,
         w, z, &ldz_, &dwork, &lwork, iwork.get(), ifail, &info);
      lwork = (strumpack_blas_int)dwork;
      std::unique_ptr<double[]> work(new double[lwork]);
      STRUMPACK_FC_GLOBAL(dsyevx,DSYEVX)
        (&jobz, &range, &uplo, &n_, a, &lda_, &vl, &vu, &il_, &iu_, &abstol, &m_,
         w, z, &ldz_, work.get(), &lwork, iwork.get(), ifail, &info);
      m = m_;
      return info;
    }

    int syev(char jobz, char uplo, int n, float* a, int lda, float* w) {
      strumpack_blas_int info, lwork = -1, n_ = n, lda_ = lda;
      float swork;
      STRUMPACK_FC_GLOBAL(ssyev,SSYEV)
        (&jobz, &uplo, &n_, a, &lda_, w, &swork, &lwork, &info);
      lwork = (int)swork;
      std::unique_ptr<float[]> work(new float[lwork]);
      STRUMPACK_FC_GLOBAL(ssyev,SSYEV)
        (&jobz, &uplo, &n_, a, &lda_, w, work.get(), &lwork, &info);
      return info;
    }
    int syev(char jobz, char uplo, int n, double* a, int lda, double* w) {
      strumpack_blas_int info, lwork = -1, n_ = n, lda_ = lda;
      double swork;
      STRUMPACK_FC_GLOBAL(dsyev,DSYEV)
        (&jobz, &uplo, &n_, a, &lda_, w, &swork, &lwork, &info);
      lwork = (int)swork;
      std::unique_ptr<double[]> work(new double[lwork]);
      STRUMPACK_FC_GLOBAL(dsyev,DSYEV)
        (&jobz, &uplo, &n_, a, &lda_, w, work.get(), &lwork, &info);
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

#if defined(STRUMPACK_USE_BLAS64)
    int sytrf(char s, int n, float* a, int lda, int* ipiv, float* work, int lwork) {
      strumpack_blas_int info, n_ = n, lda_ = lda, lwork_ = lwork;
      std::vector<strumpack_blas_int> ipiv_(n);
      STRUMPACK_FC_GLOBAL(ssytrf,SSYTRF)
        (&s, &n_, a, &lda_, ipiv_.data(), work, &lwork_, &info);
      std::copy(ipiv_.begin(), ipiv_.end(), ipiv);
      STRUMPACK_FLOPS(sytrf_flops(n));
      return info;
    }
    int sytrf(char s, int n, double* a, int lda, int* ipiv, double* work, int lwork) {
      strumpack_blas_int info, n_ = n, lda_ = lda, lwork_ = lwork;
      std::vector<strumpack_blas_int> ipiv_(n);
      STRUMPACK_FC_GLOBAL(dsytrf,DSYTRF)
        (&s, &n_, a, &lda_, ipiv_.data(), work, &lwork_, &info);
      std::copy(ipiv_.begin(), ipiv_.end(), ipiv);
      STRUMPACK_FLOPS(sytrf_flops(n));
      return info;
    }
    int sytrf(char s, int n, std::complex<float>* a, int lda, int* ipiv,
              std::complex<float>* work, int lwork) {
      strumpack_blas_int info, n_ = n, lda_ = lda, lwork_ = lwork;
      std::vector<strumpack_blas_int> ipiv_(n);
      STRUMPACK_FC_GLOBAL(csytrf,CSYTRF)
        (&s, &n_, a, &lda_, ipiv_.data(), work, &lwork_, &info);
      std::copy(ipiv_.begin(), ipiv_.end(), ipiv);
      STRUMPACK_FLOPS(4*sytrf_flops(n));
      return info;
    }
    int sytrf(char s, int n, std::complex<double>* a, int lda, int* ipiv,
              std::complex<double>* work, int lwork) {
      strumpack_blas_int info, n_ = n, lda_ = lda, lwork_ = lwork;
      std::vector<strumpack_blas_int> ipiv_(n);
      STRUMPACK_FC_GLOBAL(zsytrf,ZSYTRF)
        (&s, &n_, a, &lda_, ipiv_.data(), work, &lwork_, &info);
      std::copy(ipiv_.begin(), ipiv_.end(), ipiv);
      STRUMPACK_FLOPS(4*sytrf_flops(n));
      return info;
    }
#else
    int sytrf(char s, int n, float* a, int lda, int* ipiv, float* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(ssytrf,SSYTRF)(&s, &n, a, &lda, ipiv, work, &lwork, &info);
      STRUMPACK_FLOPS(sytrf_flops(n));
      return info;
    }
    int sytrf(char s, int n, double* a, int lda, int* ipiv, double* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(dsytrf,DSYTRF)(&s, &n, a, &lda, ipiv, work, &lwork, &info);
      STRUMPACK_FLOPS(sytrf_flops(n));
      return info;
    }
    int sytrf(char s, int n, std::complex<float>* a, int lda, int* ipiv,
              std::complex<float>* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(csytrf,CSYTRF)(&s, &n, a, &lda, ipiv, work, &lwork, &info);
      STRUMPACK_FLOPS(4*sytrf_flops(n));
      return info;
    }
    int sytrf(char s, int n, std::complex<double>* a, int lda, int* ipiv,
              std::complex<double>* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(zsytrf,ZSYTRF)(&s, &n, a, &lda, ipiv, work, &lwork, &info);
      STRUMPACK_FLOPS(4*sytrf_flops(n));
      return info;
    }
#endif

#if defined(STRUMPACK_USE_BLAS64)
    int sytrs(char s, int n, int nrhs, const float* a, int lda,
              const int* ipiv, float* b, int ldb) {
      strumpack_blas_int info, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+n);
      STRUMPACK_FC_GLOBAL(ssytrs,SSYTRS)
        (&s, &n_, &nrhs_, a, &lda_, ipiv_.data(), b, &ldb_, &info);
      STRUMPACK_FLOPS(sytrs_flops(n,n,nrhs));
      return info;
    }
    int sytrs(char s, int n, int nrhs, const double* a, int lda,
     const int* ipiv, double* b, int ldb) {
      strumpack_blas_int info, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+n);
      STRUMPACK_FC_GLOBAL(dsytrs,DSYTRS)
        (&s, &n_, &nrhs_, a, &lda_, ipiv_.data(), b, &ldb_, &info);
      STRUMPACK_FLOPS(sytrs_flops(n,n,nrhs));
      return info;
    }
    int sytrs(char s, int n, int nrhs, const std::complex<float>* a, int lda,
              const int* ipiv, std::complex<float>* b, int ldb) {
      strumpack_blas_int info, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+n);
      STRUMPACK_FC_GLOBAL(csytrs,CSYTRS)
        (&s, &n_, &nrhs_, a, &lda_, ipiv_.data(), b, &ldb_, &info);
      STRUMPACK_FLOPS(4*sytrs_flops(n,n,nrhs));
      return info;
    }
    int sytrs(char s, int n, int nrhs, const std::complex<double>* a, int lda,
              const int* ipiv, std::complex<double>* b, int ldb) {
      strumpack_blas_int info, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;
      std::vector<strumpack_blas_int> ipiv_(ipiv, ipiv+n);
      STRUMPACK_FC_GLOBAL(zsytrs,ZSYTRS)
        (&s, &n_, &nrhs_, a, &lda_, ipiv_.data(), b, &ldb_, &info);
      STRUMPACK_FLOPS(4*sytrs_flops(n,n,nrhs));
      return info;
    }
#else
    int sytrs(char s, int n, int nrhs, const float* a, int lda,
              const int* ipiv, float* b, int ldb) {
      int info;
      STRUMPACK_FC_GLOBAL(ssytrs,SSYTRS)(&s, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      STRUMPACK_FLOPS(sytrs_flops(n,n,nrhs));
      return info;
    }
    int sytrs(char s, int n, int nrhs, const double* a, int lda,
              const int* ipiv, double* b, int ldb) {
      int info;
      STRUMPACK_FC_GLOBAL(dsytrs,DSYTRS)(&s, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      STRUMPACK_FLOPS(sytrs_flops(n,n,nrhs));
      return info;
    }
    int sytrs(char s, int n, int nrhs, const std::complex<float>* a, int lda,
              const int* ipiv, std::complex<float>* b, int ldb) {
      int info;
      STRUMPACK_FC_GLOBAL(csytrs,CSYTRS)(&s, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      STRUMPACK_FLOPS(4*sytrs_flops(n,n,nrhs));
      return info;
    }
    int sytrs(char s, int n, int nrhs, const std::complex<double>* a, int lda,
              const int* ipiv, std::complex<double>* b, int ldb) {
      int info;
      STRUMPACK_FC_GLOBAL(zsytrs,ZSYTRS)(&s, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      STRUMPACK_FLOPS(4*sytrs_flops(n,n,nrhs));
      return info;
    }
#endif

  } //end namespace blas
} // end namespace strumpack
