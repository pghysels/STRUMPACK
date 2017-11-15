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
#ifndef BLASLAPACKWRAPPER_H
#define BLASLAPACKWRAPPER_H

#include <cmath>
#include <complex>
#include <iostream>
#include <cassert>
#include "StrumpackParameters.hpp"
#include "strumpack_config.h"

namespace strumpack {

  template<typename scalar> inline bool is_complex() {
    return false;
  }
  template<> inline bool is_complex<std::complex<float>>() {
    return true;
  }
  template<> inline bool is_complex<std::complex<double>>() {
    return true;
  }

  template<class T> struct RealType {
    typedef T value_type;
  };
  template<class T> struct RealType<std::complex<T>> {
    typedef T value_type;
  };

  namespace blas {

    inline float my_conj(float a) { return a; }
    inline double my_conj(double a) { return a; }
    inline std::complex<float> my_conj(std::complex<float> a) {
      return std::conj(a);
    }
    inline std::complex<double> my_conj(std::complex<double> a) {
      return std::conj(a);
    }

    template<typename scalar_t> inline long long axpby_flops
    (long long n, scalar_t alpha, scalar_t beta) {
      return (alpha != scalar_t(0)) * n * (is_complex<scalar_t>() ? 2 : 1)
        + (alpha != scalar_t(0) && alpha != scalar_t(1) &&
           alpha != scalar_t(-1)) * n * (is_complex<scalar_t>() ? 6 : 1)
        + (beta != scalar_t(0) && beta != scalar_t(1) &&
           beta != scalar_t(-1)) * n * (is_complex<scalar_t>() ? 6 : 1);
    }
    template<typename scalar> inline void axpby
    (int n, scalar alpha, const scalar* x, int incx,
     scalar beta, scalar* y, int incy) {
      if (incx==1 && incy==1) {
#pragma omp parallel for
        for (int i=0; i<n; i++)
          y[i] = alpha * x[i] + beta * y[i];
      } else {
#pragma omp parallel for
        for (int i=0; i<n; i++)
          y[i*incy] = alpha * x[i*incx] + beta * y[i*incy];
      }
      STRUMPACK_FLOPS(axpby_flops(n, alpha, beta));
      STRUMPACK_BYTES(sizeof(scalar) * static_cast<long long>(n)*3);
    }

    template<typename scalar> inline void omatcopy
    (char opA, int m, int n, const scalar* a, int lda, scalar* b, int ldb) {
      // TODO do this in blocks??
      if (opA=='T' || opA=='t')
        for (int c=0; c<n; c++)
          for (int r=0; r<m; r++)
            b[c+r*ldb] = a[r+c*lda];
      else if (opA=='C' || opA=='c')
        for (int c=0; c<n; c++)
          for (int r=0; r<m; r++)
            b[c+r*ldb] = my_conj(a[r+c*lda]);
      else if (opA=='R' || opA=='r')
        for (int c=0; c<n; c++)
          for (int r=0; r<m; r++)
            b[r+c*ldb] = my_conj(a[r+c*lda]);
      else if (opA=='N' || opA=='n')
        for (int c=0; c<n; c++)
          for (int r=0; r<m; r++)
            b[r+c*ldb] = a[r+c*lda];
    }

#if defined(__HAVE_MKL)
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include "mkl_trans.h"
    template<> inline void omatcopy<float>
    (char opa, int m, int n, const float* a, int lda, float* b, int ldb) {
      if (m && n)
        mkl_somatcopy('C', opa, m, n, 1., a, lda, b, ldb);
    }
    template<> inline void omatcopy<double>
    (char opa, int m, int n, const double* a, int lda, double* b, int ldb) {
      if (m && n)
        mkl_domatcopy('C', opa, m, n, 1., a, lda, b, ldb);
    }
    template<> inline void omatcopy<std::complex<float>>
    (char opa, int m, int n, const std::complex<float>* a, int lda,
     std::complex<float>* b, int ldb) {
      MKL_Complex8 cone = {1.,0.};
      if (m && n)
        mkl_comatcopy('C', opa, m, n, cone, a, lda, b, ldb);
    }
    template<> inline void omatcopy<std::complex<double>>
    (char opa, int m, int n, const std::complex<double>* a, int lda,
     std::complex<double>* b, int ldb) {
      MKL_Complex16 zone = {1.,0.};
      if (m && n)
        mkl_zomatcopy('C', opa, m, n, zone, a, lda, b, ldb);
    }
#elif defined(__HAVE_OPENBLAS)
    extern "C" {
      // TODO this does not work on FH's laptop with openblas
      void somatcopy_
      (char* order, char* opa, int* m, int* n, float* alpha,
       const float* a, int* lda, float* b, int* ldb);
      void domatcopy_
      (char* order, char* opa, int* m, int* n, double* alpha,
       const double* a, int* lda, double* b, int* ldb);
      void comatcopy_
      (char* order, char* opa, int* m, int* n,
       std::complex<float>* alpha, const std::complex<float>* a, int* lda,
       std::complex<float>* b, int* ldb);
      void zomatcopy_
      (char* order, char* opa, int* m, int* n,
       std::complex<double>* alpha, const std::complex<double>* a, int* lda,
       std::complex<double>* b, int* ldb);
    }
    template<> inline void omatcopy<float>
    (char opa, int m, int n, const float* a, int lda, float* b, int ldb) {
      char c = 'C';
      float o = 1.;
      if (m && n)
        somatcopy_(&c, &opa, &m, &n, &o, a, &lda, b, &ldb);
    }
    template<> inline void omatcopy<double>
    (char opa, int m, int n, const double* a, int lda, double* b, int ldb) {
      char c = 'C';
      double o = 1.;
      if (m && n)
        domatcopy_(&c, &opa, &m, &n, &o, a, &lda, b, &ldb);
    }
    template<> inline void omatcopy<std::complex<float>>
    (char opa, int m, int n, const std::complex<float>* a, int lda,
     std::complex<float>* b, int ldb) {
      char c = 'C';
      std::complex<float> o(1.);
      if (m && n)
        comatcopy_(&c, &opa, &m, &n, &o, a, &lda, b, &ldb);
    }
    template<> inline void omatcopy<std::complex<double>>
    (char opa, int m, int n, const std::complex<double>* a, int lda,
     std::complex<double>* b, int ldb) {
      char c = 'C';
      std::complex<double> o(1.);
      if (m && n)
        zomatcopy_(&c, &opa, &m, &n, &o, a, &lda, b, &ldb);
    }
#endif

    // sparse blas 1 routines
    template<typename scalar> inline void axpyi
    (std::size_t nz, scalar a, const scalar* x, const int* indx, scalar* y) {
      if (a == scalar(-1)) {
        for (int i=0; i<nz; i++) y[indx[i]] -= x[i];
        STRUMPACK_FLOPS((is_complex<scalar>() ? 2 : 1) * nz);
      } else {
        if (a == scalar(1)) {
          for (int i=0; i<nz; i++) y[indx[i]] += x[i];
          STRUMPACK_FLOPS((is_complex<scalar>() ? 2 : 1) * nz);
        } else {
          for (int i=0; i<nz; i++) y[indx[i]] += a * x[i];
          STRUMPACK_FLOPS((is_complex<scalar>() ? 4 : 1) * nz * 2);
        }
      }
      STRUMPACK_BYTES(sizeof(scalar) * nz * 3 + sizeof(int) * nz);
    }

    template<typename scalar_t,typename integer_t> inline void gthr
    (std::size_t nz, const scalar_t* y, scalar_t* x, const integer_t* indx) {
      for (integer_t i=0; i<nz; i++)
        x[i] = y[indx[i]];
      STRUMPACK_BYTES(sizeof(scalar_t) * nz * 3 + sizeof(integer_t) * nz);
    }


    ///////////////////////////////////////////////////////////
    ///////// BLAS1 ///////////////////////////////////////////
    ///////////////////////////////////////////////////////////
    extern "C" {
      void FC_GLOBAL(scopy,SCOPY)
        (int* n, const float* x, int* incx, float* y, int* incy);
      void FC_GLOBAL(dcopy,DCOPY)
        (int* n, const double* x, int* incx, double* y, int* incy);
      void FC_GLOBAL(ccopy,CCOPY)
        (int* n, const std::complex<float>* x, int* incx,
         std::complex<float>* y, int* incy);
      void FC_GLOBAL(zcopy,ZCOPY)
        (int* n, const std::complex<double>* x, int* incx,
         std::complex<double>* y, int* incy);

      void FC_GLOBAL(sscal,SSCAL)
        (int* n, float* alpha, float* x, int* incx);
      void FC_GLOBAL(dscal,DSCAL)
        (int* n, double* alpha, double* x, int* incx);
      void FC_GLOBAL(cscal,CSCAL)
        (int* n, std::complex<float>* alpha,
         std::complex<float>* x, int* incx);
      void FC_GLOBAL(zscal,ZSCAL)
        (int* n, std::complex<double>* alpha,
         std::complex<double>* x, int* incx);

      int FC_GLOBAL(isamax,ISAMAX)
        (int* n, const float* dx, int* incx);
      int FC_GLOBAL(idamax,IDAMAX)
        (int* n, const double* dx, int* incx);
      int FC_GLOBAL(icamax,ICAMAX)
        (int* n, const std::complex<float>* dx, int* incx);
      int FC_GLOBAL(izamax,IZAMAX)
        (int* n, const std::complex<double>* dx, int* incx);

      float FC_GLOBAL(snrm2,SNRM2)
        (int* n, const float* x, int* incx);
      double FC_GLOBAL(dnrm2,DNRM2)
        (int* n, const double* x, int* incx);
      float FC_GLOBAL(scnrm2,SCNRM2)
        (int* n, const std::complex<float>* x, int* incx);
      double FC_GLOBAL(dznrm2,DZNRM2)
        (int* n, const std::complex<double>* x, int* incx);

      void FC_GLOBAL(saxpy,SAXPY)
        (int* n, float* alpha, const float* x, int* incx,
         float* y, int* incy);
      void FC_GLOBAL(daxpy,DAXPY)
        (int* n, double* alpha, const double* x, int* incx,
         double* y, int* incy);
      void FC_GLOBAL(caxpy,CAXPY)
        (int* n, std::complex<float>* alpha,
         const std::complex<float>* x, int* incx,
         std::complex<float>* y, int* incy);
      void FC_GLOBAL(zaxpy,ZAXPY)
        (int* n, std::complex<double>* alpha,
         const std::complex<double>* x, int* incx,
         std::complex<double>* y, int* incy);

      void FC_GLOBAL(sswap,SSWAP)
        (int* n, float* x, int* ldx, float* y, int* ldy);
      void FC_GLOBAL(dswap,DSWAP)
        (int* n, double* x, int* ldx, double* y, int* ldy);
      void FC_GLOBAL(cswap,CSWAP)
        (int* n, std::complex<float>* x, int* ldx,
         std::complex<float>* y, int* ldy);
      void FC_GLOBAL(zswap,ZSWAP)
        (int* n, std::complex<double>* x, int* ldx,
         std::complex<double>* y, int* ldy);

      float FC_GLOBAL(sdot,SDOT)
        (int* n, const float* x, int* incx, const float* y, int* incy);
      double FC_GLOBAL(ddot,DDOT)
        (int* n, const double* x, int* incx, const double* y, int* incy);


      ///////////////////////////////////////////////////////////
      ///////// BLAS2 ///////////////////////////////////////////
      ///////////////////////////////////////////////////////////
      void FC_GLOBAL(sgemv,SGEMV)
        (char* t, int* m, int* n, float* alpha, const float *a, int* lda,
         const float* x, int* incx, float* beta, float* y, int* incy);
      void FC_GLOBAL(dgemv,DGEMV)
        (char* t, int* m, int* n, double* alpha, const double *a, int* lda,
         const double* x, int* incx, double* beta, double* y, int* incy);
      void FC_GLOBAL(cgemv,CGEMV)
        (char* t, int* m, int* n, std::complex<float>* alpha,
         const std::complex<float> *a, int* lda,
         const std::complex<float>* x, int* incx, std::complex<float>* beta,
         std::complex<float>* y, int* incy);
      void FC_GLOBAL(zgemv,ZGEMV)
        (char* t, int* m, int* n, std::complex<double>* alpha,
         const std::complex<double> *a, int* lda,
         const std::complex<double>* x, int* incx, std::complex<double>* beta,
         std::complex<double>* y, int* incy);

      void FC_GLOBAL(sger,SGER)
        (int* m, int* n, float* alpha, const float* x, int* incx,
         const float* y, int* incy, float* a, int* lda);
      void FC_GLOBAL(dger,DGER)
        (int* m, int* n, double* alpha, const double* x, int* incx,
         const double* y, int* incy, double* a, int* lda);
      void FC_GLOBAL(cgeru,CGERU)
        (int* m, int* n, std::complex<float>* alpha,
         const std::complex<float>* x, int* incx,
         const std::complex<float>* y, int* incy,
         std::complex<float>* a, int* lda);
      void FC_GLOBAL(zgeru,ZGERU)
        (int* m, int* n, std::complex<double>* alpha,
         const std::complex<double>* x, int* incx,
         const std::complex<double>* y, int* incy,
         std::complex<double>* a, int* lda);
      void FC_GLOBAL(cgerc,CGERC)
        (int* m, int* n, std::complex<float>* alpha,
         const std::complex<float>* x, int* incx,
         const std::complex<float>* y, int* incy,
         std::complex<float>* a, int* lda);
      void FC_GLOBAL(zgerc,ZGERC)
        (int* m, int* n, std::complex<double>* alpha,
         const std::complex<double>* x, int* incx,
         const std::complex<double>* y, int* incy,
         std::complex<double>* a, int* lda);

      void FC_GLOBAL(strmv,STRMV)
        (char* ul, char* t, char* d, int* n,
         const float* a, int* lda, float* x, int* incx);
      void FC_GLOBAL(dtrmv,DTRMV)
        (char* ul, char* t, char* d, int* n,
         const double* a, int* lda, double* x, int* incx);
      void FC_GLOBAL(ctrmv,CTRMV)
        (char* ul, char* t, char* d, int* n,
         const std::complex<float>* a, int* lda,
         std::complex<float>* x, int* incx);
      void FC_GLOBAL(ztrmv,ZTRMV)
        (char* ul, char* t, char* d, int* n,
         const std::complex<double>* a, int* lda,
         std::complex<double>* x, int* incx);

      void FC_GLOBAL(strsv,STRSV)
        (char* ul, char* t, char* d, int* m, const float* a, int* lda,
         float* b, int* incb);
      void FC_GLOBAL(dtrsv,DTRSV)
        (char* ul, char* t, char* d, int* m, const double* a, int* lda,
         double* b, int* incb);
      void FC_GLOBAL(ctrsv,CTRSV)
        (char* ul, char* t, char* d, int* m,
         const std::complex<float>* a, int* lda,
         std::complex<float>* b, int* incb);
      void FC_GLOBAL(ztrsv,ZTRSV)
        (char* ul, char* t, char* d, int* m,
         const std::complex<double>* a, int* lda,
         std::complex<double>* b, int* incb);


      ///////////////////////////////////////////////////////////
      ///////// BLAS3 ///////////////////////////////////////////
      ///////////////////////////////////////////////////////////
      void FC_GLOBAL(sgemm,SGEMM)
        (char* ta, char* tb, int* m, int* n, int* k,
         float* alpha, const float* a, int* lda, const float* b, int* ldb,
         float* beta, float* c, int* ldc);
      void FC_GLOBAL(dgemm,DGEMM)
        (char* ta, char* tb, int* m, int* n, int* k,
         double* alpha, const double* A, int* lda, const double* b, int* ldb,
         double* beta, double* c, int* ldc);
      void FC_GLOBAL(cgemm,CGEMM)
        (char* ta, char* tb, int* m, int* n, int* k,
         std::complex<float>* alpha, const std::complex<float>* a, int* lda,
         const std::complex<float>* b, int* ldb, std::complex<float>* beta,
         std::complex<float>* c, int* ldc);
      void FC_GLOBAL(zgemm,ZGEMM)
        (char* ta, char* tb, int* m, int* n, int* k,
         std::complex<double>* alpha, const std::complex<double>* a, int* lda,
         const std::complex<double>* b, int* ldb, std::complex<double>* beta,
         std::complex<double>* c, int* ldc);

      void FC_GLOBAL(strsm,STRSM)
        (char* s, char* ul, char* t, char* d, int* m, int* n,
         float* alpha, const float* a, int* lda, float* b, int* ldb);
      void FC_GLOBAL(dtrsm,DTRSM)
        (char* s, char* ul, char* t, char* d, int* m, int* n,
         double* alpha, const double* a, int* lda, double* b, int* ldb);
      void FC_GLOBAL(ctrsm,CTRSM)
        (char* s, char* ul, char* t, char* d, int* m, int* n,
         std::complex<float>* alpha, const std::complex<float>* a, int* lda,
         std::complex<float>* b, int* ldb);
      void FC_GLOBAL(ztrsm,ZTRSM)
        (char* s, char* ul, char* t, char* d, int* m, int* n,
         std::complex<double>* alpha, const std::complex<double>* a, int* lda,
         std::complex<double>* b, int* ldb);

      void FC_GLOBAL(strmm,STRMM)
        (char* s, char* ul, char* t, char* d, int* m, int* n, float* alpha,
         const float* a, int* lda, float* b, int* ldb);
      void FC_GLOBAL(dtrmm,DTRMM)
        (char* s, char* ul, char* t, char* d, int* m, int* n, double* alpha,
         const double* a, int* lda, double* b, int* ldb);
      void FC_GLOBAL(ctrmm,CTRMM)
        (char* s, char* ul, char* t, char* d, int* m, int* n,
         std::complex<float>* alpha, const std::complex<float>* a, int* lda,
         std::complex<float>* b, int* ldb);
      void FC_GLOBAL(ztrmm,ZTRMM)
        (char* s, char* ul, char* t, char* d, int* m, int* n,
         std::complex<double>* alpha, const std::complex<double>* a, int* lda,
         std::complex<double>* b, int* ldb);


      ///////////////////////////////////////////////////////////
      ///////// LAPACK //////////////////////////////////////////
      ///////////////////////////////////////////////////////////
      float FC_GLOBAL(slamch,SLAMCH)(char* cmach);
      double FC_GLOBAL(dlamch,DLAMCH)(char* cmach);

      int FC_GLOBAL(ilaenv,ILAENV)
        (int* ispec, char* name, char* opts,
         int* n1, int* n2, int* n3, int* n4);

      void FC_GLOBAL(clacgv,CLACGV)
        (int* n, std::complex<float>* x, int* incx);
      void FC_GLOBAL(zlacgv,ZLACGV)
        (int* n, std::complex<double>* x, int* incx);

      int FC_GLOBAL(slacpy,SLACPY)
        (char* uplo, int* m, int* n, const float* a, int* lda,
         float* b, int* ldb);
      int FC_GLOBAL(dlacpy,DLACPY)
        (char* uplo, int* m, int* n, const double* a, int* lda,
         double* b, int* ldb);
      int FC_GLOBAL(clacpy,CLACPY)
        (char* uplo, int* m, int* n, const std::complex<float>* a, int* lda,
         std::complex<float>* b, int* ldb);
      int FC_GLOBAL(zlacpy,ZLACPY)
        (char* uplo, int* m, int* n, const std::complex<double>* a, int* lda,
         std::complex<double>* b, int* ldb);

      void FC_GLOBAL(slaswp,SLASWP)
        (int* n, float* a, int* lda, int* k1, int* k2,
         const int* ipiv, int* incx);
      void FC_GLOBAL(dlaswp,DLASWP)
        (int* n, double* a, int* lda, int* k1, int* k2,
         const int* ipiv, int* incx);
      void FC_GLOBAL(claswp,CLASWP)
        (int* n, std::complex<float>* a, int* lda, int* k1, int* k2,
         const int* ipiv, int* incx);
      void FC_GLOBAL(zlaswp,ZLASWP)
        (int* n, std::complex<double>* a, int* lda, int* k1, int* k2,
         const int* ipiv, int* incx);

      void FC_GLOBAL(slapmr,SLAPMR)
        (int* fwd, int* m, int* n, float* a, int* lda, const int* ipiv);
      void FC_GLOBAL(dlapmr,DLAPMR)
        (int* fwd, int* m, int* n, double* a, int* lda, const int* ipiv);
      void FC_GLOBAL(clapmr,CLAPMR)
        (int* fwd, int* m, int* n, std::complex<float>* a, int* lda,
         const int* ipiv);
      void FC_GLOBAL(zlapmr,ZLAPMR)
        (int* fwd, int* m, int* n, std::complex<double>* a, int* lda,
         const int* ipiv);

      void FC_GLOBAL(slaset,SLASET)
        (char* s, int* m, int* n, float* alpha,
         float* beta, float* a, int* lda);
      void FC_GLOBAL(dlaset,DLASET)
        (char* s, int* m, int* n, double* alpha,
         double* beta, double* a, int* lda);
      void FC_GLOBAL(claset,CLASET)
        (char* s, int* m, int* n, std::complex<float>* alpha,
         std::complex<float>* beta, std::complex<float>* a, int* lda);
      void FC_GLOBAL(zlaset,ZLASET)
        (char* s, int* m, int* n, std::complex<double>* alpha,
         std::complex<double>* beta, std::complex<double>* a, int* lda);

      void FC_GLOBAL(sgeqp3,SGEQP3)
        (int* m, int* n, float* a, int* lda, int* jpvt,
         float* tau, float* work, int* lwork, int* info);
      void FC_GLOBAL(dgeqp3,DGEQP3)
        (int* m, int* n, double* a, int* lda, int* jpvt,
         double* tau, double* work, int* lwork, int* info);
      void FC_GLOBAL(cgeqp3,CGEQP3)
        (int* m, int* n, std::complex<float>* a, int* lda, int* jpvt,
         std::complex<float>* tau, std::complex<float>* work, int* lwork,
         std::complex<float>* rwork, int* info);
      void FC_GLOBAL(zgeqp3,ZGEQP3)
        (int* m, int* n, std::complex<double>* a, int* lda, int* jpvt,
         std::complex<double>* tau, std::complex<double>* work, int* lwork,
         std::complex<double>* rwork, int* info);

      void FC_GLOBAL(sgeqp3tol,SGEQP3TOL)
        (int* m, int* n, float* a, int* lda, int* jpvt,
         float* tau, float* work, int* lwork, int* info,
         int* rank, float* rtol, float* atol, int* depth);
      void FC_GLOBAL(dgeqp3tol,DGEQP3TOL)
        (int* m, int* n, double* a, int* lda, int* jpvt,
         double* tau, double* work, int* lwork, int* info,
         int* rank, double* rtol, double* atol, int* depth);
      void FC_GLOBAL(cgeqp3tol,CGEQP3TOL)
        (int* m, int* n, std::complex<float>* a, int* lda, int* jpvt,
         std::complex<float>* tau, std::complex<float>* work, int* lwork,
         float* rwork, int* info, int* rank,
         float* rtol, float* atol, int* depth);
      void FC_GLOBAL(zgeqp3tol,ZGEQP3TOL)
        (int* m, int* n, std::complex<double>* a, int* lda, int* jpvt,
         std::complex<double>* tau, std::complex<double>* work, int* lwork,
         double* rwork, int* info, int* rank,
         double* rtol, double* atol, int* depth);

      void FC_GLOBAL(sgeqrf,SGEQRF)
        (int* m, int* n, float* a, int* lda, float* tau,
         float* work, int* lwork, int* info);
      void FC_GLOBAL(dgeqrf,DGEQRF)
        (int* m, int* n, double* a, int* lda, double* tau,
         double* work, int* lwork, int* info);
      void FC_GLOBAL(cgeqrf,CGEQRF)
        (int* m, int* n, std::complex<float>* a, int* lda,
         std::complex<float>* tau, std::complex<float>* work,
         int* lwork, int* info);
      void FC_GLOBAL(zgeqrf,ZGEQRF)
        (int* m, int* n, std::complex<double>* a, int* lda,
         std::complex<double>* tau, std::complex<double>* work,
         int* lwork, int* info);

      void FC_GLOBAL(sgeqrfmod,SGEQRFMOD)
        (int* m, int* n, float* a, int* lda, float* tau,
         float* work, int* lwork, int* info, int* depth);
      void FC_GLOBAL(dgeqrfmod,DGEQRFMOD)
        (int* m, int* n, double* a, int* lda, double* tau,
         double* work, int* lwork, int* info, int* depth);
      void FC_GLOBAL(cgeqrfmod,CGEQRFMOD)
        (int* m, int* n, std::complex<float>* a, int* lda,
         std::complex<float>* tau, std::complex<float>* work, int* lwork,
         int* info, int* depth);
      void FC_GLOBAL(zgeqrfmod,ZGEQRFMOD)
        (int* m, int* n, std::complex<double>* a, int* lda,
         std::complex<double>* tau, std::complex<double>* work, int* lwork,
         int* info, int* depth);

      void FC_GLOBAL(sgelqf,SGELQF)
        (int* m, int* n, float* a, int* lda, float* tau,
         float* work, int* lwork, int* info);
      void FC_GLOBAL(dgelqf,DGELQF)
        (int* m, int* n, double* a, int* lda, double* tau,
         double* work, int* lwork, int* info);
      void FC_GLOBAL(cgelqf,CGELQF)
        (int* m, int* n, std::complex<float>* a, int* lda,
         std::complex<float>* tau, std::complex<float>* work,
         int* lwork, int* info);
      void FC_GLOBAL(zgelqf,ZGELQF)
        (int* m, int* n, std::complex<double>* a, int* lda,
         std::complex<double>* tau, std::complex<double>* work,
         int* lwork, int* info);

      void FC_GLOBAL(sgelqfmod,SGELQFMOD)
        (int* m, int* n, float* a, int* lda, float* tau,
         float* work, int* lwork, int* info, int* depth);
      void FC_GLOBAL(dgelqfmod,DGELQFMOD)
        (int* m, int* n, double* a, int* lda, double* tau,
         double* work, int* lwork, int* info, int* depth);
      void FC_GLOBAL(cgelqfmod,CGELQFMOD)
        (int* m, int* n, std::complex<float>* a, int* lda,
         std::complex<float>* tau, std::complex<float>* work, int* lwork,
         int* info, int* depth);
      void FC_GLOBAL(zgelqfmod,ZGELQFMOD)
        (int* m, int* n, std::complex<double>* a, int* lda,
         std::complex<double>* tau, std::complex<double>* work, int* lwork,
         int* info, int* depth);

      void FC_GLOBAL(sgetrf,SGETRF)
        (int* m, int* n, float* a, int* lda, int* ipiv, int* info);
      void FC_GLOBAL(dgetrf,DGETRF)
        (int* m, int* n, double* a, int* lda, int* ipiv, int* info);
      void FC_GLOBAL(cgetrf,CGETRF)
        (int* m, int* n, std::complex<float>* a, int* lda,
         int* ipiv, int* info);
      void FC_GLOBAL(zgetrf,ZGETRF)
        (int* m, int* n, std::complex<double>* a, int* lda,
         int* ipiv, int* info);

      void FC_GLOBAL(sgetrfmod,SGETRFMOD)
        (int* m, int* n, float* a, int* lda,
         int* ipiv, int* info, int* depth);
      void FC_GLOBAL(dgetrfmod,DGETRFMOD)
        (int* m, int* n, double* a, int* lda,
         int* ipiv, int* info, int* depth);
      void FC_GLOBAL(cgetrfmod,CGETRFMOD)
        (int* m, int* n, std::complex<float>* a, int* lda,
         int* ipiv, int* info, int* depth);
      void FC_GLOBAL(zgetrfmod,ZGETRFMOD)
        (int* m, int* n, std::complex<double>* a, int* lda,
         int* ipiv, int* info, int* depth);

      void FC_GLOBAL(sgetf2,SGETF2)
        (int* m, int* n, float* a, int* lda, int* ipiv, int* info);
      void FC_GLOBAL(dgetf2,DGETF2)
        (int* m, int* n, double* a, int* lda, int* ipiv, int* info);
      void FC_GLOBAL(cgetf2,CGETF2)
        (int* m, int* n, std::complex<float>* a, int* lda,
         int* ipiv, int* info);
      void FC_GLOBAL(zgetf2,ZGETF2)
        (int* m, int* n, std::complex<double>* a, int* lda,
         int* ipiv, int* info);

      void FC_GLOBAL(sgetrs,SGETRS)
        (char* t, int* n, int* nrhs, const float* a, int* lda,
         const int* ipiv, float* b, int* ldb, int* info);
      void FC_GLOBAL(dgetrs,dgetrs)
        (char* t, int* n, int* nrhs, const double* a, int* lda,
         const int* ipiv, double* b, int* ldb, int* info);
      void FC_GLOBAL(cgetrs,cgetrs)
        (char* t, int* n, int* nrhs, const std::complex<float>* a, int* lda,
         const int* ipiv, std::complex<float>* b, int* ldb, int* info);
      void FC_GLOBAL(zgetrs,ZGETRS)
        (char* t, int* n, int* nrhs, const std::complex<double>* a, int* lda,
         const int* ipiv, std::complex<double>* b, int* ldb, int* info);

      void FC_GLOBAL(spotrf,SPOTRF)
        (char* ul, int* n, float* a, int* lda, int* info);
      void FC_GLOBAL(dpotrf,DPOTRF)
        (char* ul, int* n, double* a, int* lda, int* info);
      void FC_GLOBAL(cpotrf,CPOTRF)
        (char* ul, int* n, std::complex<float>* a, int* lda, int* info);
      void FC_GLOBAL(zpotrf,ZPOTRF)
        (char* ul, int* n, std::complex<double>* a,
         int* lda, int* info);

      void FC_GLOBAL(sgetri,SGETRI)
        (int* n, float* a, int* lda, int* ipiv,
         float* work, int* lwork, int* info);
      void FC_GLOBAL(dgetri,DGETRI)
        (int* n, double* a, int* lda, int* ipiv,
         double* work, int* lwork, int* info);
      void FC_GLOBAL(cgetri,CGETRI)
        (int* n, std::complex<float>* a, int* lda, int* ipiv,
         std::complex<float>* work, int* lwork, int* info);
      void FC_GLOBAL(zgetri,ZGETRI)
        (int* n, std::complex<double>* a, int* lda, int* ipiv,
         std::complex<double>* work, int* lwork, int* info);

      void FC_GLOBAL(sorglq,SORGLQ)
        (int* m, int* n, int* k, float* a, int* lda, const float* tau,
         float* work, int* lwork, int* info);
      void FC_GLOBAL(dorglq,DORGLQ)
        (int* m, int* n, int* k, double* a, int* lda, const double* tau,
         double* work, int* lwork, int* info);
      void FC_GLOBAL(cunglq,CORGLQ)
        (int* m, int* n, int* k, std::complex<float>* a, int* lda,
         const std::complex<float>* tau, std::complex<float>* work,
         int* lwork, int* info);
      void FC_GLOBAL(zunglq,ZORGLQ)
        (int* m, int* n, int* k, std::complex<double>* a, int* lda,
         const std::complex<double>* tau, std::complex<double>* work,
         int* lwork, int* info);

      void FC_GLOBAL(sorglqmod,SORQLQMOD)
        (int* m, int* n, int* k, float* a, int* lda, const float* tau,
         float* work, int* lwork, int* info, int* depth);
      void FC_GLOBAL(dorglqmod,DORQLQMOD)
        (int* m, int* n, int* k, double* a, int* lda, const double* tau,
         double* work, int* lwork, int* info, int* depth);
      void FC_GLOBAL(cunglqmod,CORQLQMOD)
        (int* m, int* n, int* k, std::complex<float>* a, int* lda,
         const std::complex<float>* tau, std::complex<float>* work,
         int* lwork, int* info, int* depth);
      void FC_GLOBAL(zunglqmod,ZORQLQMOD)
        (int* m, int* n, int* k, std::complex<double>* a, int* lda,
         const std::complex<double>* tau, std::complex<double>* work,
         int* lwork, int* info, int* depth);

      void FC_GLOBAL(sorgqr,SORGQR)
        (int* m, int* n, int* k, float* a, int* lda, const float* tau,
         float* work, int* lwork, int* info);
      void FC_GLOBAL(dorgqr,DORGQR)
        (int* m, int* n, int* k, double* a, int* lda, const double* tau,
         double* work, int* lwork, int* info);
      void FC_GLOBAL(cungqr,CORGQR)
        (int* m, int* n, int* k, std::complex<float>* a, int* lda,
         const std::complex<float>* tau, std::complex<float>* work,
         int* lwork, int* info);
      void FC_GLOBAL(zungqr,ZORGQR)
        (int* m, int* n, int* k, std::complex<double>* a, int* lda,
         const std::complex<double>* tau, std::complex<double>* work,
         int* lwork, int* info);

      void FC_GLOBAL(sgesv,SGESV)
        (int* n, int* nrhs, float* a, int* lda, int* ipiv,
         float* b, int* ldb, int* info);
      void FC_GLOBAL(dgesv,DGESV)
        (int* n, int* nrhs, double* a, int* lda, int* ipiv,
         double* b, int* ldb, int* info);
      void FC_GLOBAL(cgesv,CGESV)
        (int* n, int* nrhs, std::complex<float>* a, int* lda, int* ipiv,
         std::complex<float>* b, int* ldb, int* info);
      void FC_GLOBAL(zgesv,ZGESV)
        (int* n, int* nrhs, std::complex<double>* a, int* lda, int* ipiv,
         std::complex<double>* b, int* ldb, int* info);

      void FC_GLOBAL(slarnv,SLARNV)
        (int* idist, int* iseed, int* n, float* x);
      void FC_GLOBAL(dlarnv,DLARNV)
        (int* idist, int* iseed, int* n, double* x);
      void FC_GLOBAL(clarnv,CLARNV)
        (int* idist, int* iseed, int* n, std::complex<float>* x);
      void FC_GLOBAL(zlarnv,ZLARNV)
        (int* idist, int* iseed, int* n, std::complex<double>* x);

      float FC_GLOBAL(slange,SLANGE)
        (char* norm, int* m, int* n, const float* a, int* lda, float* work);
      double FC_GLOBAL(dlange,DLANGE)
        (char* norm, int* m, int* n, const double* a,int* lda, double* work);
      float FC_GLOBAL(clange,CLANGE)
        (char* norm, int* m, int* n,
         const std::complex<float>* a, int* lda, float* work);
      double FC_GLOBAL(zlange,ZLANGE)
        (char* norm, int* m, int* n,
         const std::complex<double>* a, int* lda, double* work);

      void FC_GLOBAL(sgesvd,SGESVD)
        (char* jobu, char* jobvt, int* m, int* n, float* a, int* lda,
         float* s, float* u, int* ldu, float* vt, int* ldvt,
         float* work, int* lwork, int* info);
      void FC_GLOBAL(dgesvd,DGESVD)
        (char* jobu, char* jobvt, int* m, int* n, double* a, int* lda,
         double* s, double* u, int* ldu, double* vt, int* ldvt,
         double* work, int* lwork, int* info);
    }

    inline int ilaenv
    (int ispec, char name[], char opts[], int n1, int n2, int n3, int n4) {
      return FC_GLOBAL(ilaenv,ILAENV)(&ispec, name, opts, &n1, &n2, &n3, &n4);
    }

    template<typename real> inline real lamch(char cmach);
    template<> inline float lamch<float>(char cmach) {
      return FC_GLOBAL(slamch,SLAMCH)(&cmach);
    }
    template<> inline double lamch<double>(char cmach) {
      return FC_GLOBAL(dlamch,DLAMCH)(&cmach);
    }


    template<typename scalar> inline long long gemm_flops
    (long long m, long long n, long long k, scalar alpha, scalar beta) {
      return (alpha != scalar(0.)) * m * n * (k * 2 - 1) +
        (alpha != scalar(0.) && beta != scalar(0.)) * m * n +
        (alpha != scalar(0.) && alpha != scalar(1.)) * m* n +
        (beta != scalar(0.) && beta != scalar(1.)) * m * n;
    }
    inline long long gemm_moves(long long m, long long n, long long k) {
      return 2 * m * n + m * k + k * n;
    }
    inline void gemm
    (char ta, char tb, int m, int n, int k, float alpha,
     const float *a, int lda, const float *b, int ldb,
     float beta, float *c, int ldc) {
      lda = std::max(lda, 1);
      FC_GLOBAL(sgemm,SGEMM)
        (&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      STRUMPACK_FLOPS(gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(4*gemm_moves(m,n,k));
    }
    inline void gemm
    (char ta, char tb, int m, int n, int k, double alpha,
     const double *a, int lda, const double *b, int ldb,
     double beta, double *c, int ldc) {
      lda = std::max(lda, 1);
      FC_GLOBAL(dgemm,DGEMM)
        (&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      STRUMPACK_FLOPS(gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(8*gemm_moves(m,n,k));
    }
    inline void gemm
    (char ta, char tb, int m, int n, int k, std::complex<float> alpha,
     const std::complex<float>* a, int lda,
     const std::complex<float>* b, int ldb, std::complex<float> beta,
     std::complex<float>* c, int ldc) {
      lda = std::max(lda, 1);
      FC_GLOBAL(cgemm,CGEMM)
        (&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      STRUMPACK_FLOPS(4*gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(2*4*gemm_moves(m,n,k));
    }
    inline void gemm
    (char ta, char tb, int m, int n, int k, std::complex<double> alpha,
     const std::complex<double>* a, int lda,
     const std::complex<double>* b, int ldb, std::complex<double> beta,
     std::complex<double>* c, int ldc) {
      lda = std::max(lda, 1);
      FC_GLOBAL(zgemm,ZGEMM)
        (&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      STRUMPACK_FLOPS(4*gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(2*8*gemm_moves(m,n,k));
    }


    template<typename scalar> inline long long gemv_flops
    (long long m, long long n, scalar alpha, scalar beta) {
      return (alpha != scalar(0.)) * m * (n * 2 - 1) +
        (alpha != scalar(1.) && alpha != scalar(0.)) * m +
        (beta != scalar(0.) && beta != scalar(1.)) * m +
        (alpha != scalar(0.) && beta != scalar(0.)) * m;
    }
    inline long long gemv_moves(long long m, long long n) {
      return 2 * m + m * n + n;
    }
    inline void gemv
    (char t, int m, int n, float alpha, const float *a, int lda,
     const float *x, int incx, float beta, float *y, int incy) {
      FC_GLOBAL(sgemv,SGEMV)
        (&t, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
      STRUMPACK_FLOPS(gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(4*gemv_moves(m,n));
    }
    inline void gemv
    (char t, int m, int n, double alpha, const double *a, int lda,
     const double *x, int incx, double beta, double *y, int incy) {
      FC_GLOBAL(dgemv,DGEMV)
        (&t, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
      STRUMPACK_FLOPS(gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(8*gemv_moves(m,n));
    }
    inline void gemv
    (char t, int m, int n, std::complex<float> alpha,
     const std::complex<float> *a, int lda,
     const std::complex<float> *x, int incx, std::complex<float> beta,
     std::complex<float> *y, int incy) {
      FC_GLOBAL(cgemv,CGEMV)
        (&t, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
      STRUMPACK_FLOPS(4*gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(2*4*gemv_moves(m,n));
    }
    inline void gemv
    (char t, int m, int n, std::complex<double> alpha,
     const std::complex<double> *a, int lda,
     const std::complex<double> *x, int incx, std::complex<double> beta,
     std::complex<double> *y, int incy) {
      FC_GLOBAL(zgemv,ZGEMV)
        (&t, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
      STRUMPACK_FLOPS(4*gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(2*8*gemv_moves(m,n));
    }


    template<typename scalar_t> inline long long ger_flops
    (long long m, long long n, scalar_t alpha) {
      // TODO check this?
      return (alpha != scalar_t(0)) * m * n +
        (alpha != scalar_t(0) && alpha != scalar_t(1)) * m * n +
        (alpha != scalar_t(0)) * m * n;
    }
    inline void geru
    (int m, int n, float alpha, const float* x, int incx,
     const float* y, int incy, float* a, int lda) {
      FC_GLOBAL(sger,SGER)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(ger_flops(m,n,alpha));
    }
    inline void geru
    (int m, int n, double alpha, const double* x, int incx,
     const double* y, int incy, double* a, int lda) {
      FC_GLOBAL(dger,DGER)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(ger_flops(m,n,alpha));
    }
    inline void geru
    (int m, int n, std::complex<float> alpha,
     const std::complex<float>* x, int incx,
     const std::complex<float>* y, int incy,
     std::complex<float>* a, int lda) {
      FC_GLOBAL(cgeru,CGERU)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(4*ger_flops(m,n,alpha));
    }
    inline void geru
    (int m, int n, std::complex<double> alpha,
     const std::complex<double>* x, int incx,
     const std::complex<double>* y, int incy,
     std::complex<double>* a, int lda) {
      FC_GLOBAL(zgeru,ZGERU)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(4*ger_flops(m,n,alpha));
    }


    inline void gerc
    (int m, int n, float alpha, const float* x, int incx,
     const float* y, int incy, float* a, int lda) {
      FC_GLOBAL(sger,SGER)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(ger_flops(m,n,alpha));
    }
    inline void gerc
    (int m, int n, double alpha, const double* x, int incx,
     const double* y, int incy, double* a, int lda) {
      FC_GLOBAL(dger,DGER)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(ger_flops(m,n,alpha));
    }
    inline void gerc
    (int m, int n, std::complex<float> alpha,
     const std::complex<float>* x, int incx,
     const std::complex<float>* y, int incy,
     std::complex<float>* a, int lda) {
      FC_GLOBAL(cgerc,CGERC)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(4*ger_flops(m,n,alpha));
    }
    inline void gerc
    (int m, int n, std::complex<double> alpha,
     const std::complex<double>* x, int incx,
     const std::complex<double>* y, int incy,
     std::complex<double>* a, int lda) {
      FC_GLOBAL(zgerc,ZGERC)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
      STRUMPACK_FLOPS(4*ger_flops(m,n,alpha));
    }


    inline void lacgv(int, float *, int ) { }
    inline void lacgv(int, double *, int ) { } //Nothing to do.
    inline void lacgv(int n, std::complex<float> *x, int incx) {
      FC_GLOBAL(clacgv,CLACGV)(&n, x, &incx);
    }
    inline void lacgv(int n, std::complex<double> *x, int incx) {
      FC_GLOBAL(zlacgv,ZLACGV)(&n, x, &incx);
    }


    inline void lacpy
    (char ul, int m, int n, float* a, int lda, float* b, int ldb) {
      FC_GLOBAL(slacpy,SLACPY)(&ul, &m, &n, a, &lda, b, &ldb);
    }
    inline void lacpy
    (char ul, int m, int n, double* a, int lda, double* b, int ldb) {
      FC_GLOBAL(dlacpy,DLACPY)(&ul, &m, &n, a, &lda, b, &ldb);
    }
    inline void lacpy
    (char ul, int m, int n, std::complex<float>* a, int lda,
     std::complex<float>* b, int ldb) {
      FC_GLOBAL(clacpy,CLACPY)(&ul, &m, &n, a, &lda, b, &ldb);
    }
    inline void lacpy
    (char ul, int m, int n, std::complex<double>* a, int lda,
     std::complex<double>* b, int ldb) {
      FC_GLOBAL(zlacpy,ZLACPY)(&ul, &m, &n, a, &lda, b, &ldb);
    }


    template<typename scalar_t> inline long long axpy_flops
    (long long n, scalar_t alpha) {
      return (alpha != scalar_t(0) && alpha != scalar_t(1)) * n +
        (alpha != scalar_t(0)) * n;
    }

    inline void axpy
    (int n, float alpha, float* x, int incx, float* y, int incy) {
      FC_GLOBAL(saxpy,SAXPY)(&n, &alpha, x, &incx, y, &incy);
      STRUMPACK_FLOPS(axpy_flops(n,alpha));
    }
    inline void axpy
    (int n, double alpha, double* x, int incx, double* y, int incy) {
      FC_GLOBAL(daxpy,DAXPY)(&n, &alpha, x, &incx, y, &incy);
      STRUMPACK_FLOPS(axpy_flops(n,alpha));
    }
    inline void axpy
    (int n, std::complex<float> alpha,
     const std::complex<float>* x, int incx,
     std::complex<float>* y, int incy) {
      FC_GLOBAL(caxpy,CAXPY)(&n, &alpha, x, &incx, y, &incy);
      STRUMPACK_FLOPS(4*axpy_flops(n,alpha));
    }
    inline void axpy
    (int n, std::complex<double> alpha,
     const std::complex<double>* x, int incx,
     std::complex<double>* y, int incy) {
      FC_GLOBAL(zaxpy,ZAXPY)(&n, &alpha, x, &incx, y, &incy);
      STRUMPACK_FLOPS(4*axpy_flops(n,alpha));
    }



    inline void copy
    (int n, const float* x, int incx, float* y, int incy) {
      FC_GLOBAL(scopy,SCOPY)(&n, x, &incx, y, &incy);
    }
    inline void copy
    (int n, const double* x, int incx, double* y, int incy) {
      FC_GLOBAL(dcopy,DCOPY)(&n, x, &incx, y, &incy);
    }
    inline void copy
    (int n, const std::complex<float>* x, int incx,
     std::complex<float>* y, int incy) {
      FC_GLOBAL(ccopy,CCOPY)(&n, x, &incx, y, &incy);
    }
    inline void copy
    (int n, const std::complex<double>* x, int incx,
     std::complex<double>* y, int incy) {
      FC_GLOBAL(zcopy,ZCOPY)(&n, x, &incx, y, &incy);
    }

    template<typename scalar_t> inline long long scal_flops
    (long long n, scalar_t alpha) {
      if (alpha == scalar_t(1)) return 0;
      else return n;
    }
    template<typename scalar_t> inline long long scal_moves
    (long long n, scalar_t alpha) {
      if (alpha == scalar_t(1)) return 0;
      else return 2 * n;
    }
    inline void scal(int n, float alpha, float* x, int incx) {
      FC_GLOBAL(sscal,SSCAL)(&n, &alpha, x, &incx);
      STRUMPACK_FLOPS(scal_flops(n,alpha));
    }
    inline void scal(int n, double alpha, double* x, int incx) {
      FC_GLOBAL(dscal,DSCAL)(&n, &alpha, x, &incx);
      STRUMPACK_FLOPS(scal_flops(n,alpha));
    }
    inline void scal
    (int n, std::complex<float> alpha, std::complex<float>* x, int incx) {
      FC_GLOBAL(cscal,CSCAL)(&n, &alpha, x, &incx);
      STRUMPACK_FLOPS(4*scal_flops(n,alpha));
    }
    inline void scal
    (int n, std::complex<double> alpha, std::complex<double>* x, int incx) {
      FC_GLOBAL(zscal,ZSCAL)(&n, &alpha, x, &incx);
      STRUMPACK_FLOPS(4*scal_flops(n,alpha));
    }


    inline int iamax
    (int n, const float* x, int incx) {
      return FC_GLOBAL(isamax,ISAMAX)(&n, x, &incx);
    }
    inline int iamax
    (int n, const double* x, int incx) {
      return FC_GLOBAL(idamax,IDAMAX)(&n, x, &incx);
    }
    inline int iamax
    (int n, const std::complex<float>* x, int incx) {
      return FC_GLOBAL(icamax,ICAMAX)(&n, x, &incx);
    }
    inline int iamax
    (int n, const std::complex<double>* x, int incx) {
      return FC_GLOBAL(izamax,IZAMAX)(&n, x, &incx);
    }


    inline void swap
    (int n, float* x, int incx, float* y, int incy) {
      FC_GLOBAL(sswap,SSWAP)(&n, x, &incx, y, &incy);
    }
    inline void swap
    (int n, double* x, int incx, double* y, int incy) {
      FC_GLOBAL(dswap,DSWAP)(&n, x, &incx, y, &incy);
    }
    inline void swap
    (int n, std::complex<float>* x, int incx,
     std::complex<float>* y, int incy) {
      FC_GLOBAL(cswap,CSWAP)(&n, x, &incx, y, &incy);
    }
    inline void swap
    (int n, std::complex<double>* x, int incx,
     std::complex<double>* y, int incy) {
      FC_GLOBAL(zswap,ZSWAP)(&n, x, &incx, y, &incy);
    }


    inline long long nrm2_flops(long long n) {
      return n * 2;
    }
    inline float nrm2(int n, const float* x, int incx) {
      STRUMPACK_FLOPS(nrm2_flops(n));
      return FC_GLOBAL(snrm2,SNRM2)(&n, x, &incx);
    }
    inline double nrm2(int n, const double* x, int incx) {
      STRUMPACK_FLOPS(nrm2_flops(n));
      return FC_GLOBAL(dnrm2,DNRM2)(&n, x, &incx);
    }
    inline float nrm2(int n, const std::complex<float>* x, int incx) {
      STRUMPACK_FLOPS(4*nrm2_flops(n));
      return FC_GLOBAL(scnrm2,SCNRM2)(&n, x, &incx);
    }
    inline double nrm2(int n, const std::complex<double>* x, int incx) {
      STRUMPACK_FLOPS(4*nrm2_flops(n));
      return FC_GLOBAL(dznrm2,DZNRM2)(&n, x, &incx);
    }


    inline long long dot_flops(long long n) {
      return 2 * n;
    }
    inline float dotu
    (int n, const float* x, int incx, const float* y, int incy) {
      STRUMPACK_FLOPS(dot_flops(n))
      return FC_GLOBAL(sdot,SDOTU)(&n, x, &incx, y, &incy);
    }
    inline double dotu
    (int n, const double* x, int incx, const double* y, int incy) {
      STRUMPACK_FLOPS(dot_flops(n));
      return FC_GLOBAL(ddot,DDOT)(&n, x, &incx, y, &incy);
    }
    inline float dotc
    (int n, const float* x, int incx, const float* y, int incy) {
      STRUMPACK_FLOPS(dot_flops(n));
      return FC_GLOBAL(sdot,SDOT)(&n, x, &incx, y, &incy);
    }
    inline double dotc
    (int n, const double* x, int incx, const double* y, int incy) {
      STRUMPACK_FLOPS(dot_flops(n));
      return FC_GLOBAL(ddot,DDOT)(&n, x, &incx, y, &incy);
    }

    // MKL does not follow the fortran conventions regarding calling
    // fortran from C.  Calling MKL fortran functions that return
    // complex numbers from C/C++ seems impossible.
    // See:
    // http://www.hpc.ut.ee/dokumendid/ics_2013/composer_xe/Documentation/en_US/mkl/Release_Notes.htm
    //   "Linux* OS only: The Intel MKL single dynamic library
    //   libmkl_rt.so does not conform to the gfortran calling
    //   convention for functions returning COMPLEX values. An
    //   application compiled with gfortran and linked with
    //   libmkl_rt.so might crash if it calls the following functions:
    //              BLAS: CDOTC, CDOTU, CDOTCI, CDOTUI, ZDOTC, ZDOTU
    //              LAPACK: CLADIV, ZLADIV  "
    // But the problem is not only there with gfortran.
    // https://software.intel.com/en-us/articles/intel-math-kernel-library-intel-mkl-blas-cblas-and-lapack-compilinglinking-functions-fortran-and-cc-calls#1
    // The following code should always work:
    inline std::complex<float> dotu
    (int n, const std::complex<float>* x, int incx,
     const std::complex<float>* y, int incy) {
      STRUMPACK_FLOPS(4*dot_flops(n));
      std::complex<float> r(0.);
      for (int i=0; i<n; i++)
        r += x[i*incx]*y[i*incy];
      return r;
    }
    inline std::complex<double> dotu
    (int n, const std::complex<double>* x, int incx,
     const std::complex<double>* y, int incy) {
      STRUMPACK_FLOPS(4*dot_flops(n));
      std::complex<double> r(0.);
      for (int i=0; i<n; i++)
        r += x[i*incx]*y[i*incy];
      return r;
    }
    inline std::complex<float> dotc
    (int n, const std::complex<float>* x, int incx,
     const std::complex<float>* y, int incy) {
      STRUMPACK_FLOPS(4*dot_flops(n));
      std::complex<float> r(0.);
      for (int i=0; i<n; i++)
        r += std::conj(x[i*incx])*y[i*incy];
      return r;
    }
    inline std::complex<double> dotc
    (int n, const std::complex<double>* x, int incx,
     const std::complex<double>* y, int incy) {
      STRUMPACK_FLOPS(4*dot_flops(n));
      std::complex<double> r(0.);
      for (int i=0; i<n; i++)
        r += std::conj(x[i*incx])*y[i*incy];
      return r;
    }


    inline long long laswp_moves(long long n, long long k1, long long k2) {
      return 2 * (k2 - k1) * n;
    }
    inline void laswp
    (int n, float* a, int lda, int k1, int k2, const int* ipiv, int incx) {
      FC_GLOBAL(slaswp,SLASWP)(&n, a, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(4*laswp_moves(n,k1,k2));
    }
    inline void laswp
    (int n, double* a, int lda, int k1, int k2, const int* ipiv, int incx) {
      FC_GLOBAL(dlaswp,DLASWP)(&n, a, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(8*laswp_moves(n,k1,k2));
    }
    inline void laswp
    (int n, std::complex<float>* a, int lda, int k1, int k2,
     const int* ipiv, int incx) {
      FC_GLOBAL(claswp,CLASWP)(&n, a, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(2*4*laswp_moves(n,k1,k2));
    }
    inline void laswp
    (int n, std::complex<double>* a, int lda, int k1, int k2,
     const int* ipiv, int incx) {
      FC_GLOBAL(zlaswp,ZLASWP)(&n, a, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(2*8*laswp_moves(n,k1,k2));
    }


    inline long long lapmr_moves(long long n, long long m) {
      return 2 * m * n;
    }
    inline void lapmr
    (bool fwd, int m, int n, float* a, int lda, const int* ipiv) {
      int forward = fwd ? 1 : 0;
      FC_GLOBAL(slapmr,SLAPMR)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(4*lapmr_moves(n,m));
    }
    inline void lapmr
    (bool fwd, int m, int n, double* a, int lda, const int* ipiv) {
      int forward = fwd ? 1 : 0;
      FC_GLOBAL(dlapmr,DLAPMR)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(8*lapmr_moves(n,m));
    }
    inline void lapmr
    (bool fwd, int m, int n, std::complex<float>* a, int lda,
     const int* ipiv) {
      int forward = fwd ? 1 : 0;
      FC_GLOBAL(clapmr,CLAPMR)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(2*4*lapmr_moves(n,m));
    }
    inline void lapmr
    (bool fwd, int m, int n, std::complex<double>* a, int lda,
     const int* ipiv) {
      int forward = fwd ? 1 : 0;
      FC_GLOBAL(zlapmr,ZLAPMR)(&forward, &m, &n, a, &lda, ipiv);
      STRUMPACK_BYTES(2*8*lapmr_moves(n,m));
    }


    template<typename scalar_t> inline long long trsm_flops
    (long long m, long long n, scalar_t alpha, char s) {
      if (s=='L' || s=='l')
        return (alpha != scalar_t(0)) * n * m *(m + 1) +
          (alpha != scalar_t(1) && alpha != scalar_t(0)) * n * m;
      else return (alpha != scalar_t(0)) * m * n * (n + 1) +
             (alpha != scalar_t(1) && alpha != scalar_t(0)) * n * m;
    }
    inline long long trsm_moves(long long m, long long n) {
      return n * n / 2 + 2 * m * n;
    }
    inline void trsm
    (char s, char ul, char t, char d, int m, int n, float alpha,
     const float* a, int lda, float* b, int ldb) {
      FC_GLOBAL(strsm,STRSM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(trsm_flops(m, n, alpha, s));
      STRUMPACK_BYTES(4*trsm_moves(m, n));
    }
    inline void trsm
      (char s, char ul, char t, char d, int m, int n, double alpha,
       const double* a, int lda, double* b, int ldb) {
      FC_GLOBAL(dtrsm,DTRSM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(trsm_flops(m, n, alpha, s));
      STRUMPACK_BYTES(8*trsm_moves(m, n));
    }
    inline void trsm
      (char s, char ul, char t, char d, int m, int n,
       std::complex<float> alpha, const std::complex<float>* a, int lda,
       std::complex<float>* b, int ldb) {
      FC_GLOBAL(ctrsm,CTRSM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(4*trsm_flops(m, n, alpha, s));
      STRUMPACK_BYTES(2*4*trsm_moves(m, n));
    }
    inline void trsm
      (char s, char ul, char t, char d, int m, int n,
       std::complex<double> alpha, const std::complex<double>* a, int lda,
       std::complex<double>* b, int ldb) {
      FC_GLOBAL(ztrsm,ZTRSM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(4*trsm_flops(m, n, alpha, s));
      STRUMPACK_BYTES(2*8*trsm_moves(m, n));
    }


    template<typename scalar_t> inline long long trmm_flops
    (long long m, long long n, scalar_t alpha, char s) {
      if (s=='L' || s=='l')
        return (alpha != scalar_t(0)) * n * m * (m + 1) +
          (alpha != scalar_t(1) && alpha != scalar_t(0)) * n * m;
      else return (alpha != scalar_t(0)) * m * n * (n + 1) +
             (alpha != scalar_t(1) && alpha != scalar_t(0)) * n * m;
    }
    inline long long trmm_moves(long long m, long long n) {
      return m * m / 2 + 2 * m * n;
    }
    inline void trmm
    (char s, char ul, char t, char d, int m, int n, float alpha,
     const float* a, int lda, float* b, int ldb) {
      FC_GLOBAL(strmm,STRMM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(trmm_flops(m,n,alpha,s));
      STRUMPACK_BYTES(4*trmm_moves(m,n));
    }
    inline void trmm
    (char s, char ul, char t, char d, int m, int n, double alpha,
     const double* a, int lda, double* b, int ldb) {
      FC_GLOBAL(dtrmm,DTRMM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(trmm_flops(m,n,alpha,s));
      STRUMPACK_BYTES(8*trmm_moves(m,n));
    }
    inline void trmm
    (char s, char ul, char t, char d, int m, int n, std::complex<float> alpha,
     const std::complex<float>* a, int lda, std::complex<float>* b, int ldb) {
      FC_GLOBAL(ctrmm,CTRMM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(4*trmm_flops(m,n,alpha,s));
      STRUMPACK_BYTES(2*4*trmm_moves(m,n));
    }
    inline void trmm
    (char s, char ul, char t, char d, int m, int n,
     std::complex<double> alpha, const std::complex<double>* a, int lda,
     std::complex<double>* b, int ldb) {
      FC_GLOBAL(ztrmm,ZTRMM)
        (&s, &ul, &t, &d, &m, &n, &alpha, a, &lda, b, &ldb);
      STRUMPACK_FLOPS(4*trmm_flops(m,n,alpha,s));
      STRUMPACK_BYTES(2*8*trmm_moves(m,n));
    }


    inline long long trmv_flops(long long n) {
      return n * (n + 1);
    }
    inline long long trmv_moves(long long n) {
      return n * n / 2 + 2 * n;
    }
    inline void trmv
    (char ul, char t, char d, int n, const float* a, int lda,
     float* x, int incx) {
      FC_GLOBAL(strmv,STRMV)(&ul, &t, &d, &n, a, &lda, x, &incx);
      STRUMPACK_FLOPS(trmv_flops(n));
      STRUMPACK_BYTES(4*trmv_moves(n));
    }
    inline void trmv
    (char ul, char t, char d, int n, const double* a, int lda,
     double* x, int incx) {
      FC_GLOBAL(dtrmv,DTRMV)(&ul, &t, &d, &n, a, &lda, x, &incx);
      STRUMPACK_FLOPS(trmv_flops(n));
      STRUMPACK_BYTES(8*trmv_moves(n));
    }
    inline void trmv
    (char ul, char t, char d, int n, const std::complex<float>* a, int lda,
     std::complex<float>* x, int incx) {
      FC_GLOBAL(ctrmv,CTRMV)(&ul, &t, &d, &n, a, &lda, x, &incx);
      STRUMPACK_FLOPS(4*trmv_flops(n));
      STRUMPACK_BYTES(2*4*trmv_moves(n));
    }
    inline void trmv
    (char ul, char t, char d, int n, const std::complex<double>* a, int lda,
     std::complex<double>* x, int incx) {
      FC_GLOBAL(ztrmv,ZTRMV)(&ul, &t, &d, &n, a, &lda, x, &incx);
      STRUMPACK_FLOPS(4*trmv_flops(n));
      STRUMPACK_BYTES(2*8*trmv_moves(n));
    }



    inline long long trsv_flops(long long m) {
      return m * (m + 1);
    }
    inline long long trsv_moves(long long m) {
      return m * m / 2 + 2 * m;
    }
    inline void trsv
    (char ul, char t, char d, int m, const float* a, int lda,
     float* b, int incb) {
      FC_GLOBAL(strsv,STRSV)(&ul, &t, &d, &m, a, &lda, b, &incb);
      STRUMPACK_FLOPS(trsv_flops(m));
      STRUMPACK_BYTES(4*trsv_moves(m));
    }
    inline void trsv
    (char ul, char t, char d, int m, const double* a, int lda,
     double* b, int incb) {
      FC_GLOBAL(dtrsv,DTRSV)(&ul, &t, &d, &m, a, &lda, b, &incb);
      STRUMPACK_FLOPS(trsv_flops(m));
      STRUMPACK_BYTES(8*trsv_moves(m));
    }
    inline void trsv
    (char ul, char t, char d, int m, const std::complex<float>* a, int lda,
     std::complex<float>* b, int incb) {
      FC_GLOBAL(ctrsv,CTRSV)(&ul, &t, &d, &m, a, &lda, b, &incb);
      STRUMPACK_FLOPS(4*trsv_flops(m));
      STRUMPACK_BYTES(2*4*trsv_moves(m));
    }
    inline void trsv
    (char ul, char t, char d, int m, const std::complex<double>* a, int lda,
     std::complex<double>* b, int incb) {
      FC_GLOBAL(ztrsv,ZTRSV)(&ul, &t, &d, &m, a, &lda, b, &incb);
      STRUMPACK_FLOPS(4*trsv_flops(m));
      STRUMPACK_BYTES(2*8*trsv_moves(m));
    }


    inline void laset
    (char s, int m, int n, float alpha, float beta, float* x, int ldx) {
      FC_GLOBAL(slaset,SLASET)(&s, &m, &n, &alpha, &beta, x, &ldx);
    }
    inline void laset
    (char s, int m, int n, double alpha, double beta, double* x, int ldx) {
      FC_GLOBAL(dlaset,DLASET)(&s, &m, &n, &alpha, &beta, x, &ldx);
    }
    inline void laset
    (char s, int m, int n, std::complex<float> alpha,
     std::complex<float> beta, std::complex<float>* x, int ldx) {
      FC_GLOBAL(claset,CLASET)(&s, &m, &n, &alpha, &beta, x, &ldx);
    }
    inline void laset
    (char s, int m, int n, std::complex<double> alpha,
     std::complex<double> beta, std::complex<double>* x, int ldx) {
      FC_GLOBAL(zlaset,ZLASET)(&s, &m, &n, &alpha, &beta, x, &ldx);
    }


    inline long long geqp3_flops(long long m, long long n) {
      if (m == n) return n * n * n * 4 / 3;
      else {
        if (m > n) return n * n * 2 / 3 * (3 * m - n);
        else return m * m * 2 / 3 * (3 * n - m);
      }
    }
    inline void geqp3
    (int m, int n, float* a, int lda, int* jpvt, float* tau,
     float* work, int lwork, int* info) {
      FC_GLOBAL(sgeqp3,SGEQP3)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, info);
      STRUMPACK_FLOPS(geqp3_flops(m,n));
    }
    inline void geqp3
    (int m, int n, double* a, int lda, int* jpvt, double* tau,
     double* work, int lwork, int* info) {
      dgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, info);
      STRUMPACK_FLOPS(geqp3_flops(m,n));
    }
    inline void geqp3
    (int m, int n, std::complex<float>* a, int lda, int* jpvt,
     std::complex<float>* tau, std::complex<float>* work,
     int lwork, int* info) {
      auto rwork = new std::complex<float>[std::max(1, 2*n)];
      cgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork, info);
      delete[] rwork;
      STRUMPACK_FLOPS(4*geqp3_flops(m,n));
    }
    inline void geqp3
    (int m, int n, std::complex<double>* a, int lda, int* jpvt,
     std::complex<double>* tau, std::complex<double>* work,
     int lwork, int* info) {
      auto rwork = new std::complex<double>[std::max(1, 2*n)];
      zgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork, info);
      delete[] rwork;
      STRUMPACK_FLOPS(4*geqp3_flops(m,n));
    }
    template<typename scalar> inline void geqp3
    (int m, int n, scalar* a, int lda, int* jpvt, scalar* tau, int* info) {
      scalar lwork;
      geqp3(m, n, a, lda, jpvt, tau, &lwork, -1, info);
      int ilwork = int(lwork);
      auto work = new scalar[ilwork];
      geqp3(m, n, a, lda, jpvt, tau, work, ilwork, info);
      delete[] work;
    }


    inline void geqp3tol
    (int m, int n, float* a, int lda, int* jpvt, float* tau, float* work,
     int lwork, int* info, int& rank, float rtol, float atol, int depth) {
      FC_GLOBAL(sgeqp3tol,SGEQP3TOL)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, info,
         &rank, &rtol, &atol, &depth);
    }
    inline void geqp3tol
    (int m, int n, double* a, int lda, int* jpvt, double* tau, double* work,
     int lwork, int* info, int& rank, double rtol, double atol, int depth) {
      FC_GLOBAL(dgeqp3tol,DGEQP3TOL)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, info,
         &rank, &rtol, &atol, &depth);
    }
    inline void geqp3tol
    (int m, int n, std::complex<float>* a, int lda, int* jpvt,
     std::complex<float>* tau, std::complex<float>* work, int lwork,
     int* info, int& rank, float rtol, float atol, int depth) {
      auto rwork = new float[std::max(1, 2*n)];
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
      FC_GLOBAL(cgeqp3tol,CGEQP3TOL)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork, info,
         &rank, &rtol, &atol, &depth);
      delete[] rwork;
    }
    inline void geqp3tol
    (int m, int n, std::complex<double>* a, int lda, int* jpvt,
     std::complex<double>* tau, std::complex<double>* work, int lwork,
     int* info, int& rank, double rtol, double atol, int depth) {
      auto rwork = new double[std::max(1, 2*n)];
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
      FC_GLOBAL(zgeqp3tol,ZGEQP3TOL)
        (&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork, info,
         &rank, &rtol, &atol, &depth);
      delete[] rwork;
    }

    template<typename scalar, typename real> inline void geqp3tol
    (int m, int n, scalar* a, int lda, int* jpvt, scalar* tau, int* info,
     int& rank, real rtol, real atol, int depth) {
      scalar lwork;
      geqp3tol
        (m, n, a, lda, jpvt, tau, &lwork, -1, info, rank, rtol, atol, depth);
      int ilwork = int(std::real(lwork));
      auto work = new scalar[ilwork];
      if (! is_complex<scalar>()) {
        bool tasked = depth < params::task_recursion_cutoff_level;
        if (tasked) {
          int loop_tasks = std::max(params::num_threads / (depth+1), 1);
          int B = std::max(n / loop_tasks, 1);
          for (int task=0; task<std::ceil(n/float(B)); task++) {
#pragma omp task default(shared) firstprivate(task)
            for (int i=task*B; i<std::min((task+1)*B,n); i++)
              work[i] = nrm2(m, &a[i*lda], 1);
          }
#pragma omp taskwait
        } else
          for (int i=0; i<n; i++)
            work[i] = nrm2(m, &a[i*lda], 1);
      }
      geqp3tol
        (m, n, a, lda, jpvt, tau, work, ilwork,
         info, rank, rtol, atol, depth);
      delete[] work;
    }


    inline long long geqrf_flops(long long m, long long n) {
      if (m > n)
        return n * (n * (.5-(1./3.)*n+m) + m + 23./6.) +
          n * (n * (.5-(1./3.) * n + m) + 5./6.);
      else
        return m * (m * (-.5 - (1./3.) * m + n) + 2 * n + 23./6.) +
          m * (m * (-.5 - (1./3.) * m + n) + n + 5./6.);
    }
    inline void geqrf
    (int m, int n, float* a, int lda, float* tau,
     float* work, int lwork, int* info) {
      FC_GLOBAL(sgeqrf,SGEQRF)(&m, &n, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(geqrf_flops(m,n));
    }
    inline void geqrf
    (int m, int n, double* a, int lda, double* tau,
     double* work, int lwork, int* info) {
      FC_GLOBAL(dgeqrf,DGEQRF)(&m, &n, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(geqrf_flops(m,n));
    }
    inline void geqrf
    (int m, int n, std::complex<float>* a, int lda, std::complex<float>* tau,
     std::complex<float>* work, int lwork, int* info) {
      FC_GLOBAL(cgeqrf,CGEQRF)(&m, &n, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(4*geqrf_flops(m,n));
    }
    inline void geqrf
    (int m, int n, std::complex<double>* a, int lda,
     std::complex<double>* tau, std::complex<double>* work,
     int lwork, int* info) {
      FC_GLOBAL(zgeqrf,ZGEQRF)(&m, &n, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(4*geqrf_flops(m,n));
    }
    template<typename scalar> inline void geqrf
    (int m, int n, scalar* a, int lda, scalar* tau, int* info) {
      scalar lwork;
      geqrf(m, n, a, lda, tau, &lwork, -1, info);
      int ilwork = int(std::real(lwork));
      auto work = new scalar[ilwork];
      geqrf(m, n, a, lda, tau, work, ilwork, info);
      delete[] work;
    }


    inline void geqrfmod
    (int m, int n, float* a, int lda,
     float* tau, float* work, int lwork, int* info, int depth) {
      FC_GLOBAL(sgeqrfmod,SGEQRFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, info, &depth);
    }
    inline void geqrfmod
    (int m, int n, double* a, int lda, double* tau,
     double* work, int lwork, int* info, int depth) {
      FC_GLOBAL(dgeqrfmod,DGEQRFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, info, &depth);
    }
    inline void geqrfmod
    (int m, int n, std::complex<float>* a, int lda, std::complex<float>* tau,
     std::complex<float>* work, int lwork, int* info, int depth) {
      FC_GLOBAL(cgeqrfmod,CGEQRFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, info, &depth);
    }
    inline void geqrfmod
    (int m, int n, std::complex<double>* a, int lda,
     std::complex<double>* tau, std::complex<double>* work,
     int lwork, int* info, int depth) {
      FC_GLOBAL(zgeqrfmod,ZGEQRFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, info, &depth);
    }
    template<typename scalar> inline void geqrfmod
    (int m, int n, scalar* a, int lda, scalar* tau, int* info, int depth) {
      scalar lwork;
      geqrfmod(m, n, a, lda, tau, &lwork, -1, info, depth);
      int ilwork = int(std::real(lwork));
      auto work = new scalar[ilwork];
      geqrfmod(m, n, a, lda, tau, work, ilwork, info, depth);
      delete[] work;
    }


    inline long long gelqf_flops(long long m, long long n) {
      if (m > n)
        return n * (n * (.5 - (1./3.) * n + m) + m + 29./6.) +
                              n * (n * (-.5 - (1./3.) * n + m) + m + 5./6.);
      else
        return m * (m * (-.5 - (1./3.) * m + n) + 2 * n + 29./6.) +
                              m * (m * (.5 - (1./3.) * m + n) + 5./6.);
    }
    inline void gelqf
    (int m, int n, float* a, int lda, float* tau,
     float* work, int lwork, int* info) {
      FC_GLOBAL(sgelqf,SGELQF)(&m, &n, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(gelqf_flops(m,n));
    }
    inline void gelqf
    (int m, int n, double* a, int lda, double* tau,
     double* work, int lwork, int* info) {
      FC_GLOBAL(dgelqf,DGELQF)(&m, &n, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(gelqf_flops(m,n));
    }
    inline void gelqf
    (int m, int n, std::complex<float>* a, int lda, std::complex<float>* tau,
     std::complex<float>* work, int lwork, int* info) {
      FC_GLOBAL(cgelqf,CGELQF)(&m, &n, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(4*gelqf_flops(m,n));
    }
    inline void gelqf
    (int m, int n, std::complex<double>* a, int lda,
     std::complex<double>* tau, std::complex<double>* work,
     int lwork, int* info) {
      FC_GLOBAL(zgelqf,ZGELQF)(&m, &n, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(4*gelqf_flops(m,n));
    }
    template<typename scalar> inline void gelqf
    (int m, int n, scalar* a, int lda, scalar* tau, int* info) {
      scalar lwork;
      gelqf(m, n, a, lda, tau, &lwork, -1, info);
      int ilwork = int(std::real(lwork));
      auto work = new scalar[ilwork];
      gelqf(m, n, a, lda, tau, work, ilwork, info);
      delete[] work;
    }


    inline void gelqfmod
    (int m, int n, float* a, int lda, float* tau,
     float* work, int lwork, int* info, int depth) {
      FC_GLOBAL(sgelqfmod,SGELQFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, info, &depth);
    }
    inline void gelqfmod
    (int m, int n, double* a, int lda, double* tau,
     double* work, int lwork, int* info, int depth) {
      FC_GLOBAL(dgelqfmod,DGELQFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, info, &depth);
    }
    inline void gelqfmod
    (int m, int n, std::complex<float>* a, int lda, std::complex<float>* tau,
     std::complex<float>* work, int lwork, int* info, int depth) {
      FC_GLOBAL(cgelqfmod,CGELQFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, info, &depth);
    }
    inline void gelqfmod
    (int m, int n, std::complex<double>* a, int lda,
     std::complex<double>* tau, std::complex<double>* work,
     int lwork, int* info, int depth) {
      FC_GLOBAL(zgelqfmod,ZGELQFMOD)
        (&m, &n, a, &lda, tau, work, &lwork, info, &depth);
    }
    template<typename scalar> inline void gelqfmod
    (int m, int n, scalar* a, int lda, scalar* tau, int* info, int depth) {
      scalar lwork;
      gelqfmod(m, n, a, lda, tau, &lwork, -1, info, depth);
      int ilwork = int(std::real(lwork));
      auto work = new scalar[ilwork];
      gelqfmod(m, n, a, lda, tau, work, ilwork, info, depth);
      delete[] work;
    }


    inline long long getrf_flops(long long m, long long n) {
      // TODO check this
      if (m < n) return (m / 2 * (m * (n - m / 3 - 1) + n) + 2 * m / 3) +
                   (m / 2 * (m * (n - m / 3) - n) + m / 6);
      else return n * n * (m - n/3 - 1) / 2 + m + 2 * n / 3 +
             n * (n * (m - (1./3.) * n - 1) / 2 - m) + n / 6;
    }
    inline void getrf
    (int m, int n, float* a, int lda, int* ipiv, int* info) {
      FC_GLOBAL(sgetrf,SGETRF)(&m, &n, a, &lda, ipiv, info);
      STRUMPACK_FLOPS(getrf_flops(m,n));
    }
    inline void getrf
    (int m, int n, double* a, int lda, int* ipiv, int* info) {
      FC_GLOBAL(dgetrf,DGETRF)(&m, &n, a, &lda, ipiv, info);
      STRUMPACK_FLOPS(getrf_flops(m,n));
    }
    inline void getrf
    (int m, int n, std::complex<float>* a, int lda, int* ipiv, int* info) {
      FC_GLOBAL(cgetrf,CGETRF)(&m, &n, a, &lda, ipiv, info);
      STRUMPACK_FLOPS(4*getrf_flops(m,n));
    }
    inline void getrf
    (int m, int n, std::complex<double>* a, int lda, int* ipiv, int* info) {
      FC_GLOBAL(zgetrf,ZGETRF)(&m, &n, a, &lda, ipiv, info);
      STRUMPACK_FLOPS(4*getrf_flops(m,n));
    }


    inline void getrfmod
    (int m, int n, float* a, int lda, int* ipiv, int* info, int depth) {
      FC_GLOBAL(sgetrfmod,SGETRFMOD)(&m, &n, a, &lda, ipiv, info, &depth);
    }
    inline void getrfmod
    (int m, int n, double* a, int lda, int* ipiv, int* info, int depth) {
      FC_GLOBAL(dgetrfmod,DGETRFMOD)(&m, &n, a, &lda, ipiv, info, &depth);
    }
    inline void getrfmod
    (int m, int n, std::complex<float>* a, int lda,
     int* ipiv, int* info, int depth) {
      FC_GLOBAL(cgetrfmod,CGETRFMOD)(&m, &n, a, &lda, ipiv, info, &depth);
    }
    inline void getrfmod
    (int m, int n, std::complex<double>* a, int lda,
     int* ipiv, int* info, int depth) {
      FC_GLOBAL(zgetrfmod,ZGETRFMOD)(&m, &n, a, &lda, ipiv, info, &depth);
    }


    inline long long getrs_flops(long long n, long long nrhs) {
      return 2 * n * n * nrhs;
    }
    inline void getrs
    (char t, int n, int nrhs, const float* a, int lda,
     const int* ipiv, float* b, int ldb, int* info) {
      FC_GLOBAL(sgetrs,SGETRS)(&t, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
      STRUMPACK_FLOPS(getrs_flops(n, nrhs));
    }
    inline void getrs
    (char t, int n, int nrhs, const double* a, int lda,
     const int* ipiv, double* b, int ldb, int* info) {
      FC_GLOBAL(dgetrs,DGETRS)(&t, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
      STRUMPACK_FLOPS(getrs_flops(n, nrhs));
    }
    inline void getrs
    (char t, int n, int nrhs, const std::complex<float>* a, int lda,
     const int* ipiv, std::complex<float>* b, int ldb, int* info) {
      FC_GLOBAL(cgetrs,CGETRS)(&t, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
      STRUMPACK_FLOPS(4*getrs_flops(n, nrhs));
    }
    inline void getrs
    (char t, int n, int nrhs, const std::complex<double>* a, int lda,
     const int* ipiv, std::complex<double>* b, int ldb, int* info) {
      zgetrs_(&t, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
      STRUMPACK_FLOPS(4*getrs_flops(n, nrhs));
    }


    inline long long potrf_flops(long long n) {
      std::cout << "TODO count flops for spotrf" << std::endl;
      return 0;
    }
    inline void potrf
    (char ul, int n, float* a, int lda, int* info) {
      FC_GLOBAL(spotrf,SPOTRF)
        (&ul, &n, a, &lda, info);
      STRUMPACK_FLOPS(potrf_flops(n));
    }
    inline void potrf
    (char ul, int n, double* a, int lda, int* info) {
      FC_GLOBAL(dpotrf,DPOTRF)
        (&ul, &n, a, &lda, info);
      STRUMPACK_FLOPS(potrf_flops(n));
    }
    inline void potrf
    (char ul, int n, std::complex<float>* a, int lda, int* info) {
      FC_GLOBAL(cpotrf,CPOTRF)
        (&ul, &n, a, &lda, info);
      STRUMPACK_FLOPS(4*potrf_flops(n));
    }
    inline void potrf
    (char ul, int n, std::complex<double>* a, int lda, int* info) {
      FC_GLOBAL(zpotrf,ZPOTRF)
        (&ul, &n, a, &lda, info);
      STRUMPACK_FLOPS(4*potrf_flops(n));
    }


    inline long long xxglq_flops(long long m, long long n, long long k) {
      if (m == k) return 2 * m * m *(3 * n - m) / 3;
      else return 4 * m * n * k - 2 * (m + n) * k * k + 4 * k * k * k / 3;
    }
    inline void xxglq
    (int m, int n, int k, float* a, int lda, const float* tau,
     float* work, int lwork, int* info) {
      FC_GLOBAL(sorglq,FC_GLOBAL)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(xxglq_flops(m,n,k));
    }
    inline void xxglq
    (int m, int n, int k, double* a, int lda, const double* tau,
     double* work, int lwork, int* info) {
      FC_GLOBAL(dorglq,DORGLQ)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(xxglq_flops(m,n,k));
    }
    inline void xxglq
    (int m, int n, int k, std::complex<float>* a, int lda,
     const std::complex<float>* tau, std::complex<float>* work,
     int lwork, int* info) {
      FC_GLOBAL(cunglq,CUNGLQ)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(4*xxglq_flops(m,n,k));
    }
    inline void xxglq
    (int m, int n, int k, std::complex<double>* a, int lda,
     const std::complex<double>* tau, std::complex<double>* work,
     int lwork, int* info) {
      FC_GLOBAL(zunglq,ZUNGLQ)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(4*xxglq_flops(m,n,k));
    }
    template<typename scalar> inline void xxglq
    (int m, int n, int k, scalar* a, int lda, const scalar* tau, int* info) {
      scalar lwork;
      xxglq(m, n, k, a, lda, tau, &lwork, -1, info);
      int ilwork = int(std::real(lwork));
      scalar* work = new scalar[ilwork];
      xxglq(m, n, k, a, lda, tau, work, ilwork, info);
      delete[] work;
    }

    // do not count flops here, they are counted in the blas routines
    inline void xxglqmod
    (int m, int n, int k, float* a, int lda, const float* tau,
     float* work, int lwork, int* info, int depth) {
      FC_GLOBAL(sorglqmod,SORGLQMOD)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info, &depth);
    }
    inline void xxglqmod
    (int m, int n, int k, double* a, int lda, const double* tau,
     double* work, int lwork, int* info, int depth) {
      FC_GLOBAL(dorglqmod,DORGLQMOD)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info, &depth);
    }
    inline void xxglqmod
    (int m, int n, int k, std::complex<float>* a, int lda,
     const std::complex<float>* tau, std::complex<float>* work,
     int lwork, int* info, int depth) {
      FC_GLOBAL(cunglqmod,CUNGLQMOD)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info, &depth);
    }
    inline void xxglqmod
    (int m, int n, int k, std::complex<double>* a, int lda,
     const std::complex<double>* tau, std::complex<double>* work,
     int lwork, int* info, int depth) {
      FC_GLOBAL(zunglqmod,ZUNGLQMOD)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info, &depth);
    }
    template<typename scalar> inline void xxglqmod
    (int m, int n, int k, scalar* a, int lda,
     const scalar* tau, int* info, int depth) {
      scalar lwork;
      xxglqmod(m, n, k, a, lda, tau, &lwork, -1, info, depth);
      int ilwork = int(std::real(lwork));
      scalar* work = new scalar[ilwork];
      xxglqmod(m, n, k, a, lda, tau, work, ilwork, info, depth);
      delete[] work;
    }


    inline long long xxgqr_flops(long long m, long long n, long long k) {
      if (n == k) return 2 * n * n * (3 * m - n) / 3;
      else return 4 * m * n * k - 2 * (m + n) * k * k + 4 * k * k * k / 3;
    }
    inline void xxgqr
    (int m, int n, int k, float* a, int lda, const float* tau,
     float* work, int lwork, int* info) {
      FC_GLOBAL(sorgqr,SORGQR)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(xxgqr_flops(m,n,k));
    }
    inline void xxgqr
    (int m, int n, int k, double* a, int lda, const double* tau,
     double* work, int lwork, int* info) {
      FC_GLOBAL(dorgqr,DORGQR)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(xxgqr_flops(m,n,k));
    }
    inline void xxgqr
    (int m, int n, int k, std::complex<float>* a, int lda,
     const std::complex<float>* tau, std::complex<float>* work,
     int lwork, int* info) {
      FC_GLOBAL(cungqr,FC_GLOBAL)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(4*xxgqr_flops(m,n,k));
    }
    inline void xxgqr
    (int m, int n, int k, std::complex<double>* a, int lda,
     const std::complex<double>* tau, std::complex<double>* work,
     int lwork, int* info) {
      FC_GLOBAL(zungqr,ZUNGQR)
        (&m, &n, &k, a, &lda, tau, work, &lwork, info);
      STRUMPACK_FLOPS(4*xxgqr_flops(m,n,k));
    }
    template<typename scalar> inline void xxgqr
    (int m, int n, int k, scalar* a, int lda, const scalar* tau, int* info) {
      scalar lwork;
      xxgqr(m, n, k, a, lda, tau, &lwork, -1, info);
      int ilwork = int(std::real(lwork));
      auto work = new scalar[ilwork];
      xxgqr(m, n, k, a, lda, tau, work, ilwork, info);
      delete[] work;
    }


    inline float lange(char norm, int m, int n, const float *a, int lda) {
      if (norm == 'I' || norm == 'i') {
        auto work = new float[m];
        auto ret = FC_GLOBAL(slange,SLANGE)(&norm, &m, &n, a, &lda, work);
        delete[] work;
        return ret;
      } else return FC_GLOBAL(slange,SLANGE)(&norm, &m, &n, a, &lda, nullptr);
    }
    inline double lange(char norm, int m, int n, const double *a, int lda) {
      if (norm == 'I' || norm == 'i') {
        double* work = new double[m];
        auto ret = FC_GLOBAL(dlange,DLANGE)(&norm, &m, &n, a, &lda, work);
        delete[] work;
        return ret;
      } else return FC_GLOBAL(dlange,DLANGE)(&norm, &m, &n, a, &lda, nullptr);
    }
    inline float lange
      (char norm, int m, int n, const std::complex<float> *a, int lda) {
      if (norm == 'I' || norm == 'i') {
        auto work = new float[m];
        auto ret = FC_GLOBAL(clange,CLANGE)(&norm, &m, &n, a, &lda, work);
        delete[] work;
        return ret;
      } else return FC_GLOBAL(clange,CLANGE)(&norm, &m, &n, a, &lda, nullptr);
    }
    inline double lange
      (char norm, int m, int n, const std::complex<double> *a, int lda) {
      if (norm == 'I' || norm == 'i') {
        auto work = new double[m];
        auto ret = FC_GLOBAL(zlange,ZLANGE)(&norm, &m, &n, a, &lda, work);
        delete[] work;
        return ret;
      } else return FC_GLOBAL(zlange,ZLANGE)(&norm, &m, &n, a, &lda, nullptr);
    }


    inline int gesvd
      (char jobu, char jobvt, int m, int n, float* a, int lda,
       float* s, float* u, int ldu, float* vt, int ldvt) {
      int info;
      int lwork = -1;
      float swork;
      FC_GLOBAL(sgesvd,SGESVD)
        (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
         &swork, &lwork, &info);
      lwork = int(swork);
      auto work = new float[lwork];
      FC_GLOBAL(sgesvd,SGESVD)
        (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
         work, &lwork, &info);
      delete[] work;
      return info;
    }
    inline int gesvd
      (char jobu, char jobvt, int m, int n, double* a, int lda,
       double* s, double* u, int ldu, double* vt, int ldvt) {
      int info;
      int lwork = -1;
      double dwork;
      FC_GLOBAL(dgesvd,DGESVD)
        (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
         &dwork, &lwork, &info);
      lwork = int(dwork);
      auto work = new double[lwork];
      FC_GLOBAL(dgesvd,DGESVD)
        (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
         work, &lwork, &info);
      delete[] work;
      return info;
    }
    inline int gesvd
      (char jobu, char jobvt, int m, int n, std::complex<float>* a, int lda,
       std::complex<float>* s, std::complex<float>* u, int ldu,
       std::complex<float>* vt, int ldvt) {
      std::cout << "TODO gesvd for std::complex<float>" << std::endl;
      return 0;
    }
    inline int gesvd
      (char jobu, char jobvt, int m, int n, std::complex<double>* a, int lda,
       std::complex<double>* s, std::complex<double>* u, int ldu,
       std::complex<double>* vt, int ldvt) {
      std::cout << "TODO gesvd for std::complex<double>" << std::endl;
      return 0;
    }

  } //end namespace blas
} // end namespace strumpack

#endif // BLASLAPACKWRAPPER_H
