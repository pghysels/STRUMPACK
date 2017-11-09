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
#ifndef BLASLAPACKWRAPPER_H
#define BLASLAPACKWRAPPER_H

#include <cmath>
#include <complex>
#include <iostream>
#include <cassert>
// TODO avoid this include!, this file should be standalone
#include "StrumpackParameters.hpp"
// TODO make wrappers const correct

namespace strumpack {

  typedef std::complex<float> c_float;
  typedef std::complex<double> c_double;

  template<typename scalar> inline bool is_complex() { return false; }
  template<> inline bool is_complex<c_float>() { return true; }
  template<> inline bool is_complex<c_double>() { return true; }

  template<class T> struct RealType { typedef T value_type; };
  template<class T> struct RealType<std::complex<T>> { typedef T value_type; };

  namespace blas {

    template<typename scalar> inline void axpby(int N, scalar alpha, scalar* X, int incx, scalar beta, scalar* Y, int incy) {
      if (incx==1 && incy==1) {
#pragma omp parallel for
	for (int i=0; i<N; i++) Y[i] = alpha*X[i] + beta*Y[i];
      } else {
#pragma omp parallel for
	for (int i=0; i<N; i++) Y[i*incy] = alpha*X[i*incx] + beta*Y[i*incy];
      }
      STRUMPACK_FLOPS(static_cast<long long int>((alpha!=scalar(0))*double(N)*(is_complex<scalar>()?2:1)
						 + (alpha!=scalar(0) && alpha!=scalar(1) && alpha!=scalar(-1))*double(N)*(is_complex<scalar>()?6:1)
						 + (beta!=scalar(0) && beta!=scalar(1) && beta!=scalar(-1))*double(N)*(is_complex<scalar>()?6:1)));
      STRUMPACK_BYTES(sizeof(scalar)*static_cast<long long int>(N)*3);
    }

    template<typename scalar> inline scalar my_conj(scalar a);
    template<> inline float my_conj<float>(float a) { return a; }
    template<> inline double my_conj<double>(double a) { return a; }
    template<> inline c_float my_conj<c_float>(c_float a) { return std::conj(a); }
    template<> inline c_double my_conj<c_double>(c_double a) { return std::conj(a); }

    template<typename scalar> inline void omatcopy(char opA, int m, int n, scalar* a, int lda, scalar* b, int ldb) {
      // TODO do this in blocks??
      if      (opA=='T'||opA=='t') for (int c=0; c<n; c++) for (int r=0; r<m; r++) b[c+r*ldb] = a[r+c*lda];
      else if (opA=='C'||opA=='c') for (int c=0; c<n; c++) for (int r=0; r<m; r++) b[c+r*ldb] = my_conj(a[r+c*lda]);
      else if (opA=='R'||opA=='r') for (int c=0; c<n; c++) for (int r=0; r<m; r++) b[r+c*ldb] = my_conj(a[r+c*lda]);
      else if (opA=='N'||opA=='n') for (int c=0; c<n; c++) for (int r=0; r<m; r++) b[r+c*ldb] = a[r+c*lda];
    }

#if defined(__HAVE_MKL)
#include "mkl_trans.h"
    template<> inline void omatcopy<float>(char opA, int m, int n, float* a, int lda, float* b, int ldb)
    {  if (m && n) mkl_somatcopy('C', opA, m, n, 1., a, lda, b, ldb); }
    template<> inline void omatcopy<double>(char opA, int m, int n, double* a, int lda, double* b, int ldb)
    {  if (m && n) mkl_domatcopy('C', opA, m, n, 1., a, lda, b, ldb); }
    template<> inline void omatcopy<c_float>(char opA, int m, int n, c_float* a, int lda, c_float* b, int ldb)
    { MKL_Complex8 cone = {1.,0.};  if (m && n) mkl_comatcopy('C', opA, m, n, cone, reinterpret_cast<MKL_Complex8*>(a), lda, reinterpret_cast<MKL_Complex8*>(b), ldb); }
    template<> inline void omatcopy<c_double>(char opA, int m, int n, c_double* a, int lda, c_double* b, int ldb)
    { MKL_Complex16 zone = {1.,0.};  if (m && n) mkl_zomatcopy('C', opA, m, n, zone, reinterpret_cast<MKL_Complex16*>(a), lda, reinterpret_cast<MKL_Complex16*>(b), ldb); }
#elif defined(__HAVE_OPENBLAS)
    extern "C" {
      // TODO this does not work on FH's laptop with openblas
      void somatcopy_(char*, char*, int*, int*, float*, float*, int*, float*, int*);
      void domatcopy_(char*, char*, int*, int*, double*, double*, int*, double*, int*);
      void comatcopy_(char*, char*, int*, int*, c_float*, c_float*, int*, c_float*, int*);
      void zomatcopy_(char*, char*, int*, int*, c_double*, c_double*, int*, c_double*, int*);
    }
    template<> inline void omatcopy<float>(char opA, int m, int n, float* a, int lda, float* b, int ldb)
    { char c='C'; float o=1.; if (m && n) somatcopy_(&c, &opA, &m, &n, &o, a, &lda, b, &ldb); }
    template<> inline void omatcopy<double>(char opA, int m, int n, double* a, int lda, double* b, int ldb)
    { char c='C'; double o=1.; if (m && n) domatcopy_(&c, &opA, &m, &n, &o, a, &lda, b, &ldb); }
    template<> inline void omatcopy<c_float>(char opA, int m, int n, c_float* a, int lda, c_float* b, int ldb)
    { char c='C'; c_float o(1.); if (m && n) comatcopy_(&c, &opA, &m, &n, &o, a, &lda, b, &ldb); }
    template<> inline void omatcopy<c_double>(char opA, int m, int n, c_double* a, int lda, c_double* b, int ldb)
    { char c='C'; c_double o(1.); if (m && n) zomatcopy_(&c, &opA, &m, &n, &o, a, &lda, b, &ldb); }
#endif

    // sparse blas 1 routines
    template<typename scalar> inline void axpyi(int nz, scalar a, scalar* x, int* indx, scalar* y) {
      if (a == scalar(-1)) {
	for (int i=0; i<nz; i++) y[indx[i]] -= x[i];
	STRUMPACK_FLOPS((is_complex<scalar>()?2:1) * static_cast<long long int>(nz));
      } else {
	if (a == scalar(1)) {
	  for (int i=0; i<nz; i++) y[indx[i]] += x[i];
	  STRUMPACK_FLOPS((is_complex<scalar>()?2:1) * static_cast<long long int>(nz));
	} else {
	  for (int i=0; i<nz; i++) y[indx[i]] += a * x[i];
	  STRUMPACK_FLOPS((is_complex<scalar>()?4:1) * static_cast<long long int>(nz)*2);
	}
      }
      STRUMPACK_BYTES(sizeof(scalar)*static_cast<long long int>(nz*3)+sizeof(int)*nz);
    }

    template<typename scalar_t,typename integer_t> inline void
    gthr(integer_t nz, scalar_t* y, scalar_t* x, integer_t* indx)
    { for (integer_t i=0; i<nz; i++) x[i] = y[indx[i]];
      STRUMPACK_BYTES(sizeof(scalar_t)*static_cast<long long int>(nz*3)+sizeof(integer_t)*nz); }

    extern "C" {
      float slamch_(char* cmach);
      double dlamch_(char* cmach);

      int ilaenv_(int* ispec, char* name, char* opts, int* n1, int* n2, int* n3, int* n4);

      void sgemv_(char* trans, int* M, int* N, float* alpha, float *A, int* lda, float* X, int* incx, float* beta, float* Y, int* incy);
      void dgemv_(char* trans, int* M, int* N, double* alpha, double *A, int* lda, double* X, int* incx, double* beta, double* Y, int* incy);
      void cgemv_(char* trans, int* M, int* N, c_float* alpha, c_float *A, int* lda, c_float* X, int* incx, c_float* beta, c_float* Y, int* incy);
      void zgemv_(char* trans, int* M, int* N, c_double* alpha, c_double *A, int* lda, c_double* X, int* incx, c_double* beta, c_double* Y, int* incy);

      void sger_(int* M, int* N, float* alpha, float* X, int* incx, float* Y, int* incy, float *A, int* lda);
      void dger_(int* M, int* N, double* alpha, double* X, int* incx, double* Y, int* incy, double *A, int* lda);
      void cgeru_(int* M, int* N, c_float* alpha, c_float* X, int* incx, c_float* Y, int* incy, c_float *A, int* lda);
      void zgeru_(int* M, int* N, c_double* alpha, c_double* X, int* incx, c_double* Y, int* incy, c_double *A, int* lda);
      void cgerc_(int* M, int* N, c_float* alpha, c_float* X, int* incx, c_float* Y, int* incy, c_float *A, int* lda);
      void zgerc_(int* M, int* N, c_double* alpha, c_double* X, int* incx, c_double* Y, int* incy, c_double *A, int* lda);

      void clacgv_(int* n, c_float* x, int* incx);
      void zlacgv_(int* n, c_double* x, int* incx);

      int slacpy_(char* uplo, int* m, int* n, float* a, int* lda, float* b, int* ldb);
      int dlacpy_(char* uplo, int* m, int* n, double* a, int* lda, double* b, int* ldb);
      int clacpy_(char* uplo, int* m, int* n, c_float* a, int* lda, c_float* b, int* ldb);
      int zlacpy_(char* uplo, int* m, int* n, c_double* a, int* lda, c_double* b, int* ldb);

      void sgemm_(char* TransA, char* TransB, int* M, int* N, int* K, float* alpha, float* A, int* lda, float* B, int* ldb, float* beta, float* C, int* ldc);
      void dgemm_(char* TransA, char* TransB, int* M, int* N, int* K, double* alpha, double* A, int* lda, double* B, int* ldb, double* beta, double* C, int* ldc);
      void cgemm_(char* TransA, char* TransB, int* M, int* N, int* K, c_float* alpha, c_float* A, int* lda, c_float* B, int* ldb, c_float* beta, c_float* C, int* ldc);
      void zgemm_(char* TransA, char* TransB, int* M, int* N, int* K, c_double* alpha, c_double* A, int* lda, c_double* B, int* ldb, c_double* beta, c_double* C, int* ldc);

      void slaswp_(int* N, float* A, int* lda, int* k1, int* k2, int* ipiv, int* incx);
      void dlaswp_(int* N, double* A, int* lda, int* k1, int* k2, int* ipiv, int* incx);
      void claswp_(int* N, c_float* A, int* lda, int* k1, int* k2, int* ipiv, int* incx);
      void zlaswp_(int* N, c_double* A, int* lda, int* k1, int* k2, int* ipiv, int* incx);

      void scopy_(int* N, float* X, int* incx, float* Y, int* incy);
      void dcopy_(int* N, double* X, int* incx, double* Y, int* incy);
      void ccopy_(int* N, c_float* X, int* incx, c_float* Y, int* incy);
      void zcopy_(int* N, c_double* X, int* incx, c_double* Y, int* incy);

      void sscal_(int* N, float* alpha, float* X, int* incx);
      void dscal_(int* N, double* alpha, double* X, int* incx);
      void cscal_(int* N, c_float* alpha, c_float* X, int* incx);
      void zscal_(int* N, c_double* alpha, c_double* X, int* incx);

      int isamax_(int* N, float* dx, int* incx);
      int idamax_(int* N, double* dx, int* incx);
      int icamax_(int* N, c_float* dx, int* incx);
      int izamax_(int* N, c_double* dx, int* incx);

      float snrm2_(int* N, float* X, int* incx);
      double dnrm2_(int* N, double* X, int* incx);
      float scnrm2_(int* N, c_float* X, int* incx);
      double dznrm2_(int* N, c_double* X, int* incx);

      void saxpy_(int* N, float* alpha, float* X, int* incx, float* Y, int* incy);
      void daxpy_(int* N, double* alpha, double* X, int* incx, double* Y, int* incy);
      void caxpy_(int* N, c_float* alpha, c_float* X, int* incx, c_float* Y, int* incy);
      void zaxpy_(int* N, c_double* alpha, c_double* X, int* incx, c_double* Y, int* incy);

      void sswap_(int* N, float* X, int* ldX, float* Y, int* ldY);
      void dswap_(int* N, double* X, int* ldX, double* Y, int* ldY);
      void cswap_(int* N, c_float* X, int* ldX, c_float* Y, int* ldY);
      void zswap_(int* N, c_double* X, int* ldX, c_double* Y, int* ldY);

      float sdot_(int* N, float* X, int* incx, float* Y, int* incy);
      double ddot_(int* N, double* X, int* incx, double* Y, int* incy);

      void slaset_(char* side, int* M, int* N, float* alpha, float* beta, float* X, int* ldX);
      void dlaset_(char* side, int* M, int* N, double* alpha, double* beta, double* X, int* ldX);
      void claset_(char* side, int* M, int* N, c_float* alpha, c_float* beta, c_float* X, int* ldX);
      void zlaset_(char* side, int* M, int* N, c_double* alpha, c_double* beta, c_double* X, int* ldX);

      void strsm_(char* side, char* uplo, char* transa, char* diag, int* M, int* N, float* alpha, float* A, int* lda, float* B, int* ldb);
      void dtrsm_(char* side, char* uplo, char* transa, char* diag, int* M, int* N, double* alpha, double* A, int* lda, double* B, int* ldb);
      void ctrsm_(char* side, char* uplo, char* transa, char* diag, int* M, int* N, c_float* alpha, c_float* A, int* lda, c_float* B, int* ldb);
      void ztrsm_(char* side, char* uplo, char* transa, char* diag, int* M, int* N, c_double* alpha, c_double* A, int* lda, c_double* B, int* ldb);

      void strmm_(char* side, char* uplo, char* transa, char* diag, int* M, int* N, float* alpha, float* A, int* lda, float* B, int* ldb);
      void dtrmm_(char* side, char* uplo, char* transa, char* diag, int* M, int* N, double* alpha, double* A, int* lda, double* B, int* ldb);
      void ctrmm_(char* side, char* uplo, char* transa, char* diag, int* M, int* N, c_float* alpha, c_float* A, int* lda, c_float* B, int* ldb);
      void ztrmm_(char* side, char* uplo, char* transa, char* diag, int* M, int* N, c_double* alpha, c_double* A, int* lda, c_double* B, int* ldb);

      void strmv_(char* uplo, char* transa, char* diag, int* N, float* A, int* lda, float* X, int* incx);
      void dtrmv_(char* uplo, char* transa, char* diag, int* N, double* A, int* lda, double* X, int* incx);
      void ctrmv_(char* uplo, char* transa, char* diag, int* N, c_float* A, int* lda, c_float* X, int* incx);
      void ztrmv_(char* uplo, char* transa, char* diag, int* N, c_double* A, int* lda, c_double* X, int* incx);

      void strsv_(char* uplo, char* transa, char* diag, int* M, float* A, int* lda, float* B, int* incb);
      void dtrsv_(char* uplo, char* transa, char* diag, int* M, double* A, int* lda, double* B, int* incb);
      void ctrsv_(char* uplo, char* transa, char* diag, int* M, c_float* A, int* lda, c_float* B, int* incb);
      void ztrsv_(char* uplo, char* transa, char* diag, int* M, c_double* A, int* lda, c_double* B, int* incb);

      void sgeqp3_(int* M, int* N, float* a, int* lda, int* jpvt, float* tau, float* work, int* lwork, int* info);
      void dgeqp3_(int* M, int* N, double* a, int* lda, int* jpvt, double* tau, double* work, int* lwork, int* info);
      void cgeqp3_(int* M, int* N, c_float* a, int* lda, int* jpvt, c_float* tau, c_float* work, int* lwork, c_float* rwork, int* info);
      void zgeqp3_(int* M, int* N, c_double* a, int* lda, int* jpvt, c_double* tau, c_double* work, int* lwork, c_double* rwork, int* info);

      void sgeqp3tol_(int* M, int* N, float* a, int* lda, int* jpvt, float* tau, float* work, int* lwork, int* info, int* rank, float* rtol, float* atol, int* depth);
      void dgeqp3tol_(int* M, int* N, double* a, int* lda, int* jpvt, double* tau, double* work, int* lwork, int* info, int* rank, double* rtol, double* atol, int* depth);
      void cgeqp3tol_(int* M, int* N, c_float* a, int* lda, int* jpvt, c_float* tau, c_float* work, int* lwork, float* rwork, int* info, int* rank, float* rtol, float* atol, int* depth);
      void zgeqp3tol_(int* M, int* N, c_double* a, int* lda, int* jpvt, c_double* tau, c_double* work, int* lwork, double* rwork, int* info, int* rank, double* rtol, double* atol, int* depth);

      void sgeqrf_(int* M, int* N, float* a, int* lda, float* tau, float* work, int* lwork, int* info);
      void dgeqrf_(int* M, int* N, double* a, int* lda, double* tau, double* work, int* lwork, int* info);
      void cgeqrf_(int* M, int* N, c_float* a, int* lda, c_float* tau, c_float* work, int* lwork, int* info);
      void zgeqrf_(int* M, int* N, c_double* a, int* lda, c_double* tau, c_double* work, int* lwork, int* info);

      void sgeqrfmod_(int* M, int* N, float* a, int* lda, float* tau, float* work, int* lwork, int* info, int* depth);
      void dgeqrfmod_(int* M, int* N, double* a, int* lda, double* tau, double* work, int* lwork, int* info, int* depth);
      void cgeqrfmod_(int* M, int* N, c_float* a, int* lda, c_float* tau, c_float* work, int* lwork, int* info, int* depth);
      void zgeqrfmod_(int* M, int* N, c_double* a, int* lda, c_double* tau, c_double* work, int* lwork, int* info, int* depth);

      void sgelqf_(int* M, int* N, float* a, int* lda, float* tau, float* work, int* lwork, int* info);
      void dgelqf_(int* M, int* N, double* a, int* lda, double* tau, double* work, int* lwork, int* info);
      void cgelqf_(int* M, int* N, c_float* a, int* lda, c_float* tau, c_float* work, int* lwork, int* info);
      void zgelqf_(int* M, int* N, c_double* a, int* lda, c_double* tau, c_double* work, int* lwork, int* info);

      void sgelqfmod_(int* M, int* N, float* a, int* lda, float* tau, float* work, int* lwork, int* info, int* depth);
      void dgelqfmod_(int* M, int* N, double* a, int* lda, double* tau, double* work, int* lwork, int* info, int* depth);
      void cgelqfmod_(int* M, int* N, c_float* a, int* lda, c_float* tau, c_float* work, int* lwork, int* info, int* depth);
      void zgelqfmod_(int* M, int* N, c_double* a, int* lda, c_double* tau, c_double* work, int* lwork, int* info, int* depth);

      void sgetrf_(int* M, int* N, float* a, int* lda, int* ipiv, int* info);
      void dgetrf_(int* M, int* N, double* a, int* lda, int* ipiv, int* info);
      void cgetrf_(int* M, int* N, c_float* a, int* lda, int* ipiv, int* info);
      void zgetrf_(int* M, int* N, c_double* a, int* lda, int* ipiv, int* info);

      void sgetrfmod_(int* M, int* N, float* a, int* lda, int* ipiv, int* info, int* depth);
      void dgetrfmod_(int* M, int* N, double* a, int* lda, int* ipiv, int* info, int* depth);
      void cgetrfmod_(int* M, int* N, c_float* a, int* lda, int* ipiv, int* info, int* depth);
      void zgetrfmod_(int* M, int* N, c_double* a, int* lda, int* ipiv, int* info, int* depth);

      void sgetf2_(int* M, int* N, float* a, int* lda, int* ipiv, int* info);
      void dgetf2_(int* M, int* N, double* a, int* lda, int* ipiv, int* info);
      void cgetf2_(int* M, int* N, c_float* a, int* lda, int* ipiv, int* info);
      void zgetf2_(int* M, int* N, c_double* a, int* lda, int* ipiv, int* info);

      void sgetrs_(char* trans, int* N, int* nrhs, float* a, int* lda, int* ipiv, float* b, int* ldb, int* info);
      void dgetrs_(char* trans, int* N, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info);
      void cgetrs_(char* trans, int* N, int* nrhs, c_float* a, int* lda, int* ipiv, c_float* b, int* ldb, int* info);
      void zgetrs_(char* trans, int* N, int* nrhs, c_double* a, int* lda, int* ipiv, c_double* b, int* ldb, int* info);

      void spotrf_(char* uplo, int* n, float* a, int* lda, int* info);
      void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info);
      void cpotrf_(char* uplo, int* n, c_float* a, int* lda, int* info);
      void zpotrf_(char* uplo, int* n, c_double* a, int* lda, int* info);

      void sgetri_(int* N, float* a, int* lda, int* ipiv, float* work, int* lwork, int* info);
      void dgetri_(int* N, double* a, int* lda, int* ipiv, double* work, int* lwork, int* info);
      void cgetri_(int* N, c_float* a, int* lda, int* ipiv, c_float* work, int* lwork, int* info);
      void zgetri_(int* N, c_double* a, int* lda, int* ipiv, c_double* work, int* lwork, int* info);

      void sorglq_(int* M, int* N, int* K, float* a, int* lda, float* tau, float* work, int* lwork, int* info);
      void dorglq_(int* M, int* N, int* K, double* a, int* lda, double* tau, double* work, int* lwork, int* info);
      void cunglq_(int* M, int* N, int* K, c_float* a, int* lda, c_float* tau, c_float* work, int* lwork, int* info);
      void zunglq_(int* M, int* N, int* K, c_double* a, int* lda, c_double* tau, c_double* work, int* lwork, int* info);

      void sorglqmod_(int* M, int* N, int* K, float* a, int* lda, float* tau, float* work, int* lwork, int* info, int* depth);
      void dorglqmod_(int* M, int* N, int* K, double* a, int* lda, double* tau, double* work, int* lwork, int* info, int* depth);
      void cunglqmod_(int* M, int* N, int* K, c_float* a, int* lda, c_float* tau, c_float* work, int* lwork, int* info, int* depth);
      void zunglqmod_(int* M, int* N, int* K, c_double* a, int* lda, c_double* tau, c_double* work, int* lwork, int* info, int* depth);

      void sorgqr_(int* M, int* N, int* K, float* a, int* lda, float* tau, float* work, int* lwork, int* info);
      void dorgqr_(int* M, int* N, int* K, double* a, int* lda, double* tau, double* work, int* lwork, int* info);
      void cungqr_(int* M, int* N, int* K, c_float* a, int* lda, c_float* tau, c_float* work, int* lwork, int* info);
      void zungqr_(int* M, int* N, int* K, c_double* a, int* lda, c_double* tau, c_double* work, int* lwork, int* info);

      void sgesv_(int* N, int* NRHS, float* A, int* LDA, int* IPIV, float* B, int* LDB, int* info);
      void dgesv_(int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B, int* LDB, int* info);
      void cgesv_(int* N, int* NRHS, c_float* A, int* LDA, int* IPIV, c_float* B, int* LDB, int* info);
      void zgesv_(int* N, int* NRHS, c_double* A, int* LDA, int* IPIV, c_double* B, int* LDB, int* info);

      void slarnv_(int* idist, int* iseed, int* n, float* x);
      void dlarnv_(int* idist, int* iseed, int* n, double* x);
      void clarnv_(int* idist, int* iseed, int* n, c_float* x);
      void zlarnv_(int* idist, int* iseed, int* n, c_double* x);

      float slange_(char* norm, int* m, int* n, float* a, int* lda, float* work);
      double dlange_(char* norm, int* m, int* n, double* a,int* lda, double* work);
      float clange_(char* norm, int* m, int* n, c_float* a, int* lda, float* work);
      double zlange_(char* norm, int* m, int* n, c_double* a, int* lda, double* work);

      void sgesvd_(char* JOBU, char* JOBVT, int* M, int* N,
                   float* A, int* LDA, float* S, float* U, int* LDU,
                   float* VT, int* LDVT, float* WORK, int* LWORK,
                   int* INFO);
      void dgesvd_(char* JOBU, char* JOBVT, int* M, int* N,
                   double* A, int* LDA, double* S, double* U, int* LDU,
                   double* VT, int* LDVT, double* WORK, int* LWORK,
                   int* INFO);
    }

    inline int ilaenv(int ispec, char name[], char opts[], int n1, int n2, int n3, int n4) {
      return ilaenv_(&ispec, name, opts, &n1, &n2, &n3, &n4);
    }

    template<typename real> inline real lamch(char cmach);
    template<> inline float lamch<float>(char cmach)
    { return slamch_(&cmach); }
    template<> inline double lamch<double>(char cmach)
    { return dlamch_(&cmach); }

    template<typename scalar> inline void gemm(char TransA, char TransB, int M, int N, int K, scalar alpha, scalar *A, int lda, scalar *B, int ldb, scalar beta, scalar *C, int ldc);
    template<> inline void gemm<float>(char TransA, char TransB, int M, int N, int K, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc)
    { lda = std::max(lda, 1);
      sgemm_(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
      STRUMPACK_FLOPS(static_cast<long long int>((alpha!=0)*double(M)*double(N)*(double(K)*2.-1.) + (alpha!=0 && beta!=0)*double(M)*double(N)
						 + (alpha!=0 && alpha!=1)*double(M)*double(N) + (beta!=0 && beta!=1)*double(M)*double(N)));
      STRUMPACK_BYTES(4*static_cast<long long int>(2.*double(M)*double(N)+double(M)*double(K)+double(K)*double(N)));
    }
    template<> inline void gemm<double>(char TransA, char TransB, int M, int N, int K, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
    { lda = std::max(lda, 1);
      dgemm_(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
      STRUMPACK_FLOPS(static_cast<long long int>((alpha!=0)*double(M)*double(N)*(double(K)*2.-1.) + (alpha!=0 && beta!=0)*double(M)*double(N)
						 + (alpha!=0 && alpha!=1)*double(M)*double(N) + (beta!=0 && beta!=1)*double(M)*double(N)));
      STRUMPACK_BYTES(8*static_cast<long long int>(2.*double(M)*double(N)+double(M)*double(K)+double(K)*double(N)));
    }
    template<> inline void gemm<c_float>(char TransA, char TransB, int M, int N, int K, c_float alpha, c_float *A, int lda, c_float *B, int ldb, c_float beta, c_float *C, int ldc)
    { lda = std::max(lda, 1);
      cgemm_(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
      STRUMPACK_FLOPS(4*static_cast<long long int>((alpha!=c_float(0))*double(M)*double(N)*(double(K)*2.-1.) + (alpha!=c_float(0) && beta!=c_float(0))*double(M)*double(N)
						   + (alpha!=c_float(0) && alpha!=c_float(1))*double(M)*double(N) + (beta!=c_float(0) && beta!=c_float(1))*double(M)*double(N)));
      STRUMPACK_BYTES(2*4*static_cast<long long int>(2.*double(M)*double(N)+double(M)*double(K)+double(K)*double(N))); }
    template<> inline void gemm<c_double>(char TransA, char TransB, int M, int N, int K, c_double alpha, c_double *A, int lda, c_double *B, int ldb, c_double beta, c_double *C, int ldc)
    { lda = std::max(lda, 1);
      zgemm_(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
      STRUMPACK_FLOPS(4*static_cast<long long int>((alpha!=c_double(0))*double(M)*double(N)*(double(K)*2.-1.) + (alpha!=c_double(0) && beta!=c_double(0))*double(M)*double(N)
						   + (alpha!=c_double(0) && alpha!=c_double(1))*double(M)*double(N) + (beta!=c_double(0) && beta!=c_double(1))*double(M)*double(N)));
      STRUMPACK_BYTES(2*8*static_cast<long long int>(2.*double(M)*double(N)+double(M)*double(K)+double(K)*double(N))); }

    template<typename scalar> inline void gemv(char trans, int M, int N, scalar alpha, scalar *A, int lda, scalar *X, int incx, scalar beta, scalar *Y, int incy);
    template<> inline void gemv<float>(char trans, int M, int N, float alpha, float *A, int lda, float *X, int incx, float beta, float *Y, int incy)
    { sgemv_(&trans, &M, &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy);
      STRUMPACK_FLOPS(static_cast<long long int>((alpha!=0)*double(M)*(double(N)*2.-1.) + (alpha!=1 && alpha!=0)*double(M) + (beta!=0 && beta!=1)*double(M) + (alpha!=0 && beta!=0)*double(M)));
      STRUMPACK_BYTES(4*static_cast<long long int>(2.*double(M)+double(M)*double(N)+double(N))); }
    template<> inline void gemv<double>(char trans, int M, int N, double alpha, double *A, int lda, double *X, int incx, double beta, double *Y, int incy)
    { dgemv_(&trans, &M, &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy);
      STRUMPACK_FLOPS(static_cast<long long int>((alpha!=0)*double(M)*(double(N)*2.-1.) + (alpha!=1 && alpha!=0)*double(M) + (beta!=0 && beta!=1)*double(M) + (alpha!=0 && beta!=0)*double(M)));
      STRUMPACK_BYTES(8*static_cast<long long int>(2.*double(M)+double(M)*double(N)+double(N))); }
    template<> inline void gemv<c_float>(char trans, int M, int N, c_float alpha, c_float *A, int lda, c_float *X, int incx, c_float beta, c_float *Y, int incy)
    { cgemv_(&trans, &M, &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy);
      STRUMPACK_FLOPS(4*static_cast<long long int>((alpha!=c_float(0))*double(M)*(double(N)*2.-1.) + (alpha!=c_float(1) && alpha!=c_float(0))*double(M) + (beta!=c_float(0) && beta!=c_float(1))*double(M) + (alpha!=c_float(0) && beta!=c_float(0))*double(M)));
      STRUMPACK_BYTES(2*4*static_cast<long long int>(2.*double(M)+double(M)*double(N)+double(N))); }
    template<> inline void gemv<c_double>(char trans, int M, int N, c_double alpha, c_double *A, int lda, c_double *X, int incx, c_double beta, c_double *Y, int incy)
    { zgemv_(&trans, &M, &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy);
      STRUMPACK_FLOPS(4*static_cast<long long int>((alpha!=c_double(0))*double(M)*(double(N)*2.-1.) + (alpha!=c_double(1) && alpha!=c_double(0))*double(M) + (beta!=c_double(0) && beta!=c_double(1))*double(M) + (alpha!=c_double(0) && beta!=c_double(0))*double(M)));
      STRUMPACK_BYTES(2*8*static_cast<long long int>(2.*double(M)+double(M)*double(N)+double(N))); }

    template<typename scalar> inline void geru(int M, int N, scalar alpha, scalar* X, int incx, scalar* Y, int incy, scalar* A, int lda);
    template<> inline void geru<float>(int M, int N, float alpha, float* X, int incx, float* Y, int incy, float* A, int lda)
    { sger_(&M, &N, &alpha, X, &incx, Y, &incy, A, &lda);
      STRUMPACK_FLOPS(static_cast<long long int>((alpha!=0)*double(M)*double(N) + (alpha!=0 && alpha!=1)*double(M)*double(N) + (alpha!=0)*double(M)*double(N))); }
    template<> inline void geru<double>(int M, int N, double alpha, double* X, int incx, double* Y, int incy, double* A, int lda)
    { dger_(&M, &N, &alpha, X, &incx, Y, &incy, A, &lda);
      STRUMPACK_FLOPS(static_cast<long long int>((alpha!=0)*double(M)*double(N) + (alpha!=0 && alpha!=1)*double(M)*double(N) + (alpha!=0)*double(M)*double(N))); }
    template<> inline void geru<c_float>(int M, int N, c_float alpha, c_float* X, int incx, c_float* Y, int incy, c_float* A, int lda)
    { cgeru_(&M, &N, &alpha, X, &incx, Y, &incy, A, &lda);
      STRUMPACK_FLOPS(4*static_cast<long long int>((alpha!=c_float(0))*double(M)*double(N) + (alpha!=c_float(0) && alpha!=c_float(1))*double(M)*double(N) + (alpha!=c_float(0))*double(M)*double(N))); }
    template<> inline void geru<c_double>(int M, int N, c_double alpha, c_double* X, int incx, c_double* Y, int incy, c_double* A, int lda)
    { zgeru_(&M, &N, &alpha, X, &incx, Y, &incy, A, &lda);
      STRUMPACK_FLOPS(4*static_cast<long long int>((alpha!=c_double(0))*double(M)*double(N) + (alpha!=c_double(0) && alpha!=c_double(1))*double(M)*double(N) + (alpha!=c_double(0))*double(M)*double(N))); }

    template<typename scalar> inline void gerc(int M, int N, scalar alpha, scalar* X, int incx, scalar* Y, int incy, scalar* A, int lda);
    template<> inline void gerc<float>(int M, int N, float alpha, float* X, int incx, float* Y, int incy, float* A, int lda)
    { sger_(&M, &N, &alpha, X, &incx, Y, &incy, A, &lda);
      STRUMPACK_FLOPS(static_cast<long long int>((alpha!=0)*double(M)*double(N) + (alpha!=0 && alpha!=1)*double(M)*double(N) + (alpha!=0)*double(M)*double(N))); }
    template<> inline void gerc<double>(int M, int N, double alpha, double* X, int incx, double* Y, int incy, double* A, int lda)
    { dger_(&M, &N, &alpha, X, &incx, Y, &incy, A, &lda);
      STRUMPACK_FLOPS(static_cast<long long int>((alpha!=0)*double(M)*double(N) + (alpha!=0 && alpha!=1)*double(M)*double(N) + (alpha!=0)*double(M)*double(N))); }
    template<> inline void gerc<c_float>(int M, int N, c_float alpha, c_float* X, int incx, c_float* Y, int incy, c_float* A, int lda)
    { cgerc_(&M, &N, &alpha, X, &incx, Y, &incy, A, &lda);
      STRUMPACK_FLOPS(4*static_cast<long long int>((alpha!=c_float(0))*double(M)*double(N) + (alpha!=c_float(0) && alpha!=c_float(1))*double(M)*double(N) + (alpha!=c_float(0))*double(M)*double(N))); }
    template<> inline void gerc<c_double>(int M, int N, c_double alpha, c_double* X, int incx, c_double* Y, int incy, c_double* A, int lda)
    { zgerc_(&M, &N, &alpha, X, &incx, Y, &incy, A, &lda);
      STRUMPACK_FLOPS(4*static_cast<long long int>((alpha!=c_double(0))*double(M)*double(N) + (alpha!=c_double(0) && alpha!=c_double(1))*double(M)*double(N) + (alpha!=c_double(0))*double(M)*double(N))); }

    template<typename scalar> inline void lacgv(int n, scalar* x, int incx);
    template<> inline void lacgv<float>(int, float *, int ) { }
    template<> inline void lacgv<double>(int, double *, int ) { } //Nothing to do.
    template<> inline void lacgv<c_float>(int n, c_float *x, int incx) { clacgv_(&n, x, &incx); }
    template<> inline void lacgv<c_double>(int n, c_double *x, int incx) { zlacgv_(&n, x, &incx); }

    template<typename scalar> inline void lacpy(char uplo, int m, int n, scalar* a, int lda, scalar* b, int ldb);
    template<> inline void lacpy<float>(char uplo, int m, int n, float* a, int lda, float* b, int ldb)
    { slacpy_(&uplo, &m, &n, a, &lda, b, &ldb); }
    template<> inline void lacpy<double>(char uplo, int m, int n, double* a, int lda, double* b, int ldb)
    { dlacpy_(&uplo, &m, &n, a, &lda, b, &ldb); }
    template<> inline void lacpy<c_float>(char uplo, int m, int n, c_float* a, int lda, c_float* b, int ldb)
    { clacpy_(&uplo, &m, &n, a, &lda, b, &ldb); }
    template<> inline void lacpy<c_double>(char uplo, int m, int n, c_double* a, int lda, c_double* b, int ldb)
    { zlacpy_(&uplo, &m, &n, a, &lda, b, &ldb); }

    template<typename scalar> inline void axpy(int N, scalar alpha, scalar* X, int incx, scalar* Y, int incy);
    template<> inline void axpy<float>(int N, float alpha, float* X, int incx, float* Y, int incy)
    { saxpy_(&N, &alpha, X, &incx, Y, &incy);
      STRUMPACK_FLOPS(static_cast<long long int>((alpha!=0 && alpha!=1)*double(N) + (alpha!=0)*double(N))); }
    template<> inline void axpy<double>(int N, double alpha, double* X, int incx, double* Y, int incy)
    { daxpy_(&N, &alpha, X, &incx, Y, &incy);
      STRUMPACK_FLOPS(static_cast<long long int>((alpha!=0 && alpha!=1)*double(N) + (alpha!=0)*double(N))); }
    template<> inline void axpy<c_float>(int N, c_float alpha, c_float* X, int incx, c_float* Y, int incy)
    { caxpy_(&N, &alpha, X, &incx, Y, &incy); }
    template<> inline void axpy<c_double>(int N, c_double alpha, c_double* X, int incx, c_double* Y, int incy)
    { zaxpy_(&N, &alpha, X, &incx, Y, &incy); }

    template<typename scalar> inline void copy(int N, scalar* X, int incx, scalar* Y, int incy);
    template<> inline void copy<float>(int N, float* X, int incx, float* Y, int incy)
    { scopy_(&N, X, &incx, Y, &incy); }
    template<> inline void copy<double>(int N, double* X, int incx, double* Y, int incy)
    { dcopy_(&N, X, &incx, Y, &incy); }
    template<> inline void copy<c_float>(int N, c_float* X, int incx, c_float* Y, int incy)
    { ccopy_(&N, X, &incx, Y, &incy); }
    template<> inline void copy<c_double>(int N, c_double* X, int incx, c_double* Y, int incy)
    { zcopy_(&N, X, &incx, Y, &incy); }

    template<typename scalar> inline void scal(int N, scalar alpha, scalar* X, int incx);
    template<> inline void scal<float>(int N, float alpha, float* X, int incx)
    { sscal_(&N, &alpha, X, &incx);
      STRUMPACK_FLOPS((alpha==1) ? 0 : static_cast<long long int>(N)); }
    template<> inline void scal<double>(int N, double alpha, double* X, int incx)
    { dscal_(&N, &alpha, X, &incx);
      STRUMPACK_FLOPS((alpha==1) ? 0 : static_cast<long long int>(N)); }
    template<> inline void scal<c_float>(int N, c_float alpha, c_float* X, int incx)
    { cscal_(&N, &alpha, X, &incx);
      STRUMPACK_FLOPS(4*((alpha==c_float(1)) ? 0 : static_cast<long long int>(N))); }
    template<> inline void scal<c_double>(int N, c_double alpha, c_double* X, int incx)
    { zscal_(&N, &alpha, X, &incx);
      STRUMPACK_FLOPS(4*((alpha==c_double(1)) ? 0 : static_cast<long long int>(N))); }

    template<typename scalar> inline int iamax(int N, scalar* dx, int incx);
    template<> inline int iamax(int N, float* dx, int incx) { return isamax_(&N, dx, &incx); }
    template<> inline int iamax(int N, double* dx, int incx) { return idamax_(&N, dx, &incx); }
    template<> inline int iamax(int N, c_float* dx, int incx) { return icamax_(&N, dx, &incx); }
    template<> inline int iamax(int N, c_double* dx, int incx) { return izamax_(&N, dx, &incx); }

    template<typename scalar> inline void swap(int N, scalar* X, int incx, scalar* Y, int incy);
    template<> inline void swap<float>(int N, float* X, int incx, float* Y, int incy)
    { sswap_(&N, X, &incx, Y, &incy); }
    template<> inline void swap<double>(int N, double* X, int incx, double* Y, int incy)
    { dswap_(&N, X, &incx, Y, &incy); }
    template<> inline void swap<c_float>(int N, c_float* X, int incx, c_float* Y, int incy)
    { cswap_(&N, X, &incx, Y, &incy); }
    template<> inline void swap<c_double>(int N, c_double* X, int incx, c_double* Y, int incy)
    { zswap_(&N, X, &incx, Y, &incy); }

    template<typename scalar> inline typename RealType<scalar>::value_type nrm2(int N, scalar* X, int incx);
    template<> inline float nrm2<float>(int N, float* X, int incx)
    { STRUMPACK_FLOPS(static_cast<long long int>(N)*2); return snrm2_(&N, X, &incx); }
    template<> inline double nrm2<double>(int N, double* X, int incx)
    { STRUMPACK_FLOPS(static_cast<long long int>(N)*2); return dnrm2_(&N, X, &incx); }
    template<> inline float nrm2<c_float>(int N, c_float* X, int incx)
    { STRUMPACK_FLOPS(static_cast<long long int>(N)*2*4); return scnrm2_(&N, X, &incx); }
    template<> inline double nrm2<c_double>(int N, c_double* X, int incx)
    { STRUMPACK_FLOPS(static_cast<long long int>(N)*2*4); return dznrm2_(&N, X, &incx); }


    template<typename scalar> inline scalar dotu(int N, scalar* X, int incx, scalar* Y, int incy);
    template<> inline float dotu<float>(int N, float* X, int incx, float* Y, int incy)
    { STRUMPACK_FLOPS(static_cast<long long int>(N)*2); return sdot_(&N, X, &incx, Y, &incy);}
    template<> inline double dotu<double>(int N, double* X, int incx, double* Y, int incy)
    { STRUMPACK_FLOPS(static_cast<long long int>(N)*2); return ddot_(&N, X, &incx, Y, &incy);}

    template<typename scalar> inline scalar dotc(int N, scalar* X, int incx, scalar* Y, int incy);
    template<> inline float dotc<float>(int N, float* X, int incx, float* Y, int incy)
    { STRUMPACK_FLOPS(static_cast<long long int>(N)*2); return sdot_(&N, X, &incx, Y, &incy); }
    template<> inline double dotc<double>(int N, double* X, int incx, double* Y, int incy)
    { STRUMPACK_FLOPS(static_cast<long long int>(N)*2); return ddot_(&N, X, &incx, Y, &incy); }

    // MKL does not follow the fortran conventions regarding calling fortran from C.
    // Calling MKL fortran functions that return complex numbers from C/C++ seems impossible.
    // See: http://www.hpc.ut.ee/dokumendid/ics_2013/composer_xe/Documentation/en_US/mkl/Release_Notes.htm
    //   "Linux* OS only: The Intel MKL single dynamic library libmkl_rt.so does not conform to the gfortran calling convention for functions returning COMPLEX values. An application compiled with gfortran and linked with libmkl_rt.so might crash if it calls the following functions:
    //              BLAS: CDOTC, CDOTU, CDOTCI, CDOTUI, ZDOTC, ZDOTU
    //              LAPACK: CLADIV, ZLADIV  "
    // But the problem is not only there with gfortran.
    // https://software.intel.com/en-us/articles/intel-math-kernel-library-intel-mkl-blas-cblas-and-lapack-compilinglinking-functions-fortran-and-cc-calls#1
    // The following code should always work:
    template<> inline c_float dotu<c_float>(int N, c_float* X, int incx, c_float* Y, int incy)
    { STRUMPACK_FLOPS(static_cast<long long int>(N)*2*4); c_float r(0.); for (int i=0; i<N; i++) r += X[i*incx]*Y[i*incy]; return r; }
    template<> inline c_double dotu<c_double>(int N, c_double* X, int incx, c_double* Y, int incy)
    { STRUMPACK_FLOPS(static_cast<long long int>(N)*2*4); c_double r(0.); for (int i=0; i<N; i++) r += X[i*incx]*Y[i*incy]; return r; }
    template<> inline c_float dotc<c_float>(int N, c_float* X, int incx, c_float* Y, int incy)
    { STRUMPACK_FLOPS(static_cast<long long int>(N)*2*4); c_float r(0.); for (int i=0; i<N; i++) r += std::conj(X[i*incx])*Y[i*incy]; return r; }
    template<> inline c_double dotc<c_double>(int N, c_double* X, int incx, c_double* Y, int incy)
    { STRUMPACK_FLOPS(static_cast<long long int>(N)*2*4); c_double r(0.); for (int i=0; i<N; i++) r += std::conj(X[i*incx])*Y[i*incy]; return r; }

    template<typename scalar> inline void laswp(int N, scalar* A, int lda, int k1, int k2, int* ipiv, int incx);
    template<> inline void laswp(int N, float* A, int lda, int k1, int k2, int* ipiv, int incx)
    { slaswp_(&N, A, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(4*static_cast<long long int>(2.*double(k2-k1)*double(N))); }
    template<> inline void laswp(int N, double* A, int lda, int k1, int k2, int* ipiv, int incx)
    { dlaswp_(&N, A, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(8*static_cast<long long int>(2.*double(k2-k1)*double(N))); }
    template<> inline void laswp(int N, c_float* A, int lda, int k1, int k2, int* ipiv, int incx)
    { claswp_(&N, A, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(2*4*static_cast<long long int>(2.*double(k2-k1)*double(N))); }
    template<> inline void laswp(int N, c_double* A, int lda, int k1, int k2, int* ipiv, int incx)
    { zlaswp_(&N, A, &lda, &k1, &k2, ipiv, &incx);
      STRUMPACK_BYTES(2*8*static_cast<long long int>(2.*double(k2-k1)*double(N))); }

    template<typename scalar> inline void trsm(char side, char uplo, char transa, char diag, int M, int N, scalar alpha, scalar* A, int lda, scalar* B, int ldb);
    template<> inline void trsm<float>(char side, char uplo, char transa, char diag, int M, int N, float alpha, float* A, int lda, float* B, int ldb)
    { strsm_(&side, &uplo, &transa, &diag, &M, &N, &alpha, A, &lda, B, &ldb);
      STRUMPACK_FLOPS(static_cast<long long int>((side=='L'||side=='l') ?
						 (alpha!=0)*double(N)*double(M)*(double(M)+1.) + (alpha!=1 && alpha!=0)*double(N)*double(M) :
						 (alpha!=0)*double(M)*double(N)*(double(N)+1.) + (alpha!=1 && alpha!=0)*double(N)*double(M) ));
      STRUMPACK_BYTES(4*static_cast<long long int>(double(N)*double(N)*.5+2.*double(M)*double(N))); }
    template<> inline void trsm<double>(char side, char uplo, char transa, char diag, int M, int N, double alpha, double* A, int lda, double* B, int ldb)
    { dtrsm_(&side, &uplo, &transa, &diag, &M, &N, &alpha, A, &lda, B, &ldb);
      STRUMPACK_FLOPS(static_cast<long long int>((side=='L'||side=='l') ?
						 (alpha!=0)*double(N)*double(M)*(double(M)+1.) + (alpha!=1 && alpha!=0)*double(N)*double(M) :
						 (alpha!=0)*double(M)*double(N)*(double(N)+1.) + (alpha!=1 && alpha!=0)*double(N)*double(M) ));
      STRUMPACK_BYTES(8*static_cast<long long int>(double(N)*double(N)*.5+2.*double(M)*double(N))); }
    template<> inline void trsm<c_float>(char side, char uplo, char transa, char diag, int M, int N, c_float alpha, c_float* A, int lda, c_float* B, int ldb)
    { ctrsm_(&side, &uplo, &transa, &diag, &M, &N, &alpha, A, &lda, B, &ldb);
      STRUMPACK_FLOPS(4*static_cast<long long int>((side=='L'||side=='l') ?
						   (alpha!=c_float(0))*double(N)*double(M)*(double(M)+1.) + (alpha!=c_float(1) && alpha!=c_float(0))*double(N)*double(M) :
						   (alpha!=c_float(0))*double(M)*double(N)*(double(N)+1.) + (alpha!=c_float(1) && alpha!=c_float(0))*double(N)*double(M) ));
      STRUMPACK_BYTES(2*4*static_cast<long long int>(double(N)*double(N)*.5+2.*double(M)*double(N))); }
    template<> inline void trsm<c_double>(char side, char uplo, char transa, char diag, int M, int N, c_double alpha, c_double* A, int lda, c_double* B, int ldb)
    { ztrsm_(&side, &uplo, &transa, &diag, &M, &N, &alpha, A, &lda, B, &ldb);
      STRUMPACK_FLOPS(4*static_cast<long long int>((side=='L'||side=='l') ?
						   (alpha!=c_double(0))*double(N)*double(M)*(double(M)+1.) + (alpha!=c_double(1) && alpha!=c_double(0))*double(N)*double(M) :
						   (alpha!=c_double(0))*double(M)*double(N)*(double(N)+1.) + (alpha!=c_double(1) && alpha!=c_double(0))*double(N)*double(M) ));
      STRUMPACK_BYTES(2*8*static_cast<long long int>(double(N)*double(N)*.5+2.*double(M)*double(N))); }

    template<typename scalar> inline void trmm(char side, char uplo, char transa, char diag, int M, int N, scalar alpha, scalar* A, int lda, scalar* B, int ldb);
    template<> inline void trmm<float>(char side, char uplo, char transa, char diag, int M, int N, float alpha, float* A, int lda, float* B, int ldb)
    { strmm_(&side, &uplo, &transa, &diag, &M, &N, &alpha, A, &lda, B, &ldb);
      STRUMPACK_FLOPS(static_cast<long long int>((side=='L'||side=='l') ?
						 (alpha!=0)*double(N)*double(M)*(double(M)+1.) + (alpha!=1 && alpha!=0)*double(N)*double(M) :
						 (alpha!=0)*double(M)*double(N)*(double(N)+1.) + (alpha!=1 && alpha!=0)*double(N)*double(M) ));
      STRUMPACK_BYTES(4*static_cast<long long int>(double(M)*double(M)*.5+2.*double(M)*double(N))); }
    template<> inline void trmm<double>(char side, char uplo, char transa, char diag, int M, int N, double alpha, double* A, int lda, double* B, int ldb)
    { dtrmm_(&side, &uplo, &transa, &diag, &M, &N, &alpha, A, &lda, B, &ldb);
      STRUMPACK_FLOPS(static_cast<long long int>((side=='L'||side=='l') ?
						 (alpha!=0)*double(N)*double(M)*(double(M)+1.) + (alpha!=1 && alpha!=0)*double(N)*double(M) :
						 (alpha!=0)*double(M)*double(N)*(double(N)+1.) + (alpha!=1 && alpha!=0)*double(N)*double(M) ));
      STRUMPACK_BYTES(8*static_cast<long long int>(double(M)*double(M)*.5+2.*double(M)*double(N))); }
    template<> inline void trmm<c_float>(char side, char uplo, char transa, char diag, int M, int N, c_float alpha, c_float* A, int lda, c_float* B, int ldb)
    { ctrmm_(&side, &uplo, &transa, &diag, &M, &N, &alpha, A, &lda, B, &ldb);
      STRUMPACK_FLOPS(4*static_cast<long long int>((side=='L'||side=='l') ?
						   (alpha!=c_float(0))*double(N)*double(M)*(double(M)+1.) + (alpha!=c_float(1) && alpha!=c_float(0))*double(N)*double(M) :
						   (alpha!=c_float(0))*double(M)*double(N)*(double(N)+1.) + (alpha!=c_float(1) && alpha!=c_float(0))*double(N)*double(M) ));
      STRUMPACK_BYTES(2*4*static_cast<long long int>(double(M)*double(M)*.5+2.*double(M)*double(N))); }
    template<> inline void trmm<c_double>(char side, char uplo, char transa, char diag, int M, int N, c_double alpha, c_double* A, int lda, c_double* B, int ldb)
    { ztrmm_(&side, &uplo, &transa, &diag, &M, &N, &alpha, A, &lda, B, &ldb);
      STRUMPACK_FLOPS(4*static_cast<long long int>((side=='L'||side=='l') ?
						   (alpha!=c_double(0))*double(N)*double(M)*(double(M)+1.) + (alpha!=c_double(1) && alpha!=c_double(0))*double(N)*double(M) :
						   (alpha!=c_double(0))*double(M)*double(N)*(double(N)+1.) + (alpha!=c_double(1) && alpha!=c_double(0))*double(N)*double(M) ));
      STRUMPACK_BYTES(2*8*static_cast<long long int>(double(M)*double(M)*.5+2.*double(M)*double(N))); }

    template<typename scalar> inline void trmv(char uplo, char transa, char diag, int N, scalar* A, int lda, scalar* X, int incx);
    template<> inline void trmv<float>(char uplo, char transa, char diag, int N, float* A, int lda, float* X, int incx)
    { strmv_(&uplo, &transa, &diag, &N, A, &lda, X, &incx);
      STRUMPACK_FLOPS(static_cast<long long int>(double(N)*(double(N)+1.)));
      STRUMPACK_BYTES(4*static_cast<long long int>(double(N)*double(N)*.5+2.*double(N))); }
    template<> inline void trmv<double>(char uplo, char transa, char diag, int N, double* A, int lda, double* X, int incx)
    { dtrmv_(&uplo, &transa, &diag, &N, A, &lda, X, &incx);
      STRUMPACK_FLOPS(static_cast<long long int>(double(N)*(double(N)+1.)));
      STRUMPACK_BYTES(8*static_cast<long long int>(double(N)*double(N)*.5+2.*double(N))); }
    template<> inline void trmv<c_float>(char uplo, char transa, char diag, int N, c_float* A, int lda, c_float* X, int incx)
    { ctrmv_(&uplo, &transa, &diag, &N, A, &lda, X, &incx);
      STRUMPACK_FLOPS(4*static_cast<long long int>(double(N)*(double(N)+1.)));
      STRUMPACK_BYTES(2*4*static_cast<long long int>(double(N)*double(N)*.5+2.*double(N))); }
    template<> inline void trmv<c_double>(char uplo, char transa, char diag, int N, c_double* A, int lda, c_double* X, int incx)
    { ztrmv_(&uplo, &transa, &diag, &N, A, &lda, X, &incx);
      STRUMPACK_FLOPS(4*static_cast<long long int>(double(N)*(double(N)+1.)));
      STRUMPACK_BYTES(2*8*static_cast<long long int>(double(N)*double(N)*.5+2.*double(N))); }

    template<typename scalar> inline void trsv(char uplo, char transa, char diag, int M, scalar* A, int lda, scalar* B, int incb);
    template<> inline void trsv<float>(char uplo, char transa, char diag, int M, float* A, int lda, float* B, int incb)
    { strsv_(&uplo, &transa, &diag, &M, A, &lda, B, &incb);
      STRUMPACK_FLOPS(static_cast<long long int>(double(M)*(double(M)+1.)));
      STRUMPACK_BYTES(4*static_cast<long long int>(double(M)*double(M)*.5+2.*double(M))); }
    template<> inline void trsv<double>(char uplo, char transa, char diag, int M, double* A, int lda, double* B, int incb)
    { dtrsv_(&uplo, &transa, &diag, &M, A, &lda, B, &incb);
      STRUMPACK_FLOPS(static_cast<long long int>(double(M)*(double(M)+1.)));
      STRUMPACK_BYTES(8*static_cast<long long int>(double(M)*double(M)*.5+2.*double(M))); }
    template<> inline void trsv<c_float>(char uplo, char transa, char diag, int M, c_float* A, int lda, c_float* B, int incb)
    { ctrsv_(&uplo, &transa, &diag, &M, A, &lda, B, &incb);
      STRUMPACK_FLOPS(4*static_cast<long long int>(double(M)*(double(M)+1.)));
      STRUMPACK_BYTES(2*4*static_cast<long long int>(double(M)*double(M)*.5+2.*double(M))); }
    template<> inline void trsv<c_double>(char uplo, char transa, char diag, int M, c_double* A, int lda, c_double* B, int incb)
    { ztrsv_(&uplo, &transa, &diag, &M, A, &lda, B, &incb);
      STRUMPACK_FLOPS(4*static_cast<long long int>(double(M)*(double(M)+1.)));
      STRUMPACK_BYTES(2*8*static_cast<long long int>(double(M)*double(M)*.5+2.*double(M))); }

    template<typename scalar> inline void laset(char side, int M, int N, scalar alpha, scalar beta, scalar* X, int ldX);
    template<> inline void laset(char side, int M, int N, float alpha, float beta, float* X, int ldX)
    { slaset_(&side, &M, &N, &alpha, &beta, X, &ldX); }
    template<> inline void laset(char side, int M, int N, double alpha, double beta, double* X, int ldX)
    { dlaset_(&side, &M, &N, &alpha, &beta, X, &ldX); }
    template<> inline void laset(char side, int M, int N, c_float alpha, c_float beta, c_float* X, int ldX)
    { claset_(&side, &M, &N, &alpha, &beta, X, &ldX); }
    template<> inline void laset(char side, int M, int N, c_double alpha, c_double beta, c_double* X, int ldX)
    { zlaset_(&side, &M, &N, &alpha, &beta, X, &ldX); }

    template<typename scalar> inline void geqp3(int M, int N, scalar* a, int lda, int* jpvt, scalar* tau, scalar* work, int lwork, int* info);
    template<> inline void geqp3(int M, int N, float* a, int lda, int* jpvt, float* tau, float* work, int lwork, int* info)
    { sgeqp3_(&M, &N, a, &lda, jpvt, tau, work, &lwork, info); }
    template<> inline void geqp3(int M, int N, double* a, int lda, int* jpvt, double* tau, double* work, int lwork, int* info)
    { dgeqp3_(&M, &N, a, &lda, jpvt, tau, work, &lwork, info); }
    template<> inline void geqp3(int M, int N, c_float* a, int lda, int* jpvt, c_float* tau, c_float* work, int lwork, int* info)
    { c_float* rwork = new c_float[std::max(1, 2*N)];
      cgeqp3_(&M, &N, a, &lda, jpvt, tau, work, &lwork, rwork, info);
      delete[] rwork; }
    template<> inline void geqp3(int M, int N, c_double* a, int lda, int* jpvt, c_double* tau, c_double* work, int lwork, int* info)
    { c_double* rwork = new c_double[std::max(1, 2*N)];
      zgeqp3_(&M, &N, a, &lda, jpvt, tau, work, &lwork, rwork, info);
      delete[] rwork; }

    template<typename scalar> inline void geqp3(int M, int N, scalar* a, int lda, int* jpvt, scalar* tau, int* info) {
      scalar lwork;
      geqp3(M, N, a, lda, jpvt, tau, &lwork, -1, info);
      int ilwork = int(lwork);
      scalar* work = new scalar[ilwork];
      geqp3(M, N, a, lda, jpvt, tau, work, ilwork, info);
      STRUMPACK_FLOPS((is_complex<scalar>()?4:1) * static_cast<long long int>(((M==N) ? double(N)*double(N)*double(N)*4./3. : ((M>N) ? double(N)*double(N)*2./3.*(3.*double(M)-double(N)) : double(M)*double(M)*2./3.*(3.*double(N)-double(M))))));
      delete[] work;
    }

    template<typename scalar, typename real> inline void geqp3tol(int M, int N, scalar* a, int lda, int* jpvt, scalar* tau, scalar* work, int lwork, int* info, int& rank, real rtol, real atol, int depth);
    template<> inline void geqp3tol<float,float>(int M, int N, float* a, int lda, int* jpvt, float* tau, float* work, int lwork, int* info, int& rank, float rtol, float atol, int depth)
    { sgeqp3tol_(&M, &N, a, &lda, jpvt, tau, work, &lwork, info, &rank, &rtol, &atol, &depth); }
    template<> inline void geqp3tol<double,double>(int M, int N, double* a, int lda, int* jpvt, double* tau, double* work, int lwork, int* info, int& rank, double rtol, double atol, int depth)
    { dgeqp3tol_(&M, &N, a, &lda, jpvt, tau, work, &lwork, info, &rank, &rtol, &atol, &depth); }
    template<> inline void geqp3tol<c_float,float>(int M, int N, c_float* a, int lda, int* jpvt, c_float* tau, c_float* work, int lwork, int* info, int& rank, float rtol, float atol, int depth) {
      float* rwork = new float[std::max(1, 2*N)];
      bool tasked = depth<params::task_recursion_cutoff_level;
      if (tasked) {
	int loop_tasks = std::max(params::num_threads / (depth+1), 1);
	int B = std::max(N / loop_tasks, 1);
	for (int task=0; task<std::ceil(N/float(B)); task++) {
#pragma omp task default(shared) firstprivate(task)
	  for (int i=task*B; i<std::min((task+1)*B,N); i++)
	    rwork[i] = nrm2(M, &a[i*lda], 1);
	}
#pragma omp taskwait
      } else for (int i=0; i<N; i++) rwork[i] = nrm2(M, &a[i*lda], 1);
      cgeqp3tol_(&M, &N, a, &lda, jpvt, tau, work, &lwork, rwork, info, &rank, &rtol, &atol, &depth);
      delete[] rwork;
    }
    template<> inline void geqp3tol<c_double,double>(int M, int N, c_double* a, int lda, int* jpvt, c_double* tau, c_double* work, int lwork, int* info, int& rank, double rtol, double atol, int depth)
    { double* rwork = new double[std::max(1, 2*N)];
      bool tasked = depth<params::task_recursion_cutoff_level;
      if (tasked) {
	int loop_tasks = std::max(params::num_threads / (depth+1), 1);
	int B = std::max(N / loop_tasks, 1);
	for (int task=0; task<std::ceil(N/float(B)); task++) {
#pragma omp task default(shared) firstprivate(task)
	  for (int i=task*B; i<std::min((task+1)*B,N); i++)
	    rwork[i] = nrm2(M, &a[i*lda], 1);
	}
#pragma omp taskwait
      } else for (int i=0; i<N; i++) rwork[i] = nrm2(M, &a[i*lda], 1);
      zgeqp3tol_(&M, &N, a, &lda, jpvt, tau, work, &lwork, rwork, info, &rank, &rtol, &atol, &depth);
      delete[] rwork;
    }

    template<typename scalar, typename real> inline void geqp3tol(int M, int N, scalar* a, int lda, int* jpvt, scalar* tau, int* info, int& rank, real rtol, real atol, int depth) {
      scalar lwork;
      geqp3tol(M, N, a, lda, jpvt, tau, &lwork, -1, info, rank, rtol, atol, depth);
      int ilwork = int(std::real(lwork));
      scalar* work = new scalar[ilwork];
      if (! is_complex<scalar>()) {
	bool tasked = depth<params::task_recursion_cutoff_level;
	if (tasked) {
	  int loop_tasks = std::max(params::num_threads / (depth+1), 1);
	  int B = std::max(N / loop_tasks, 1);
	  for (int task=0; task<std::ceil(N/float(B)); task++) {
#pragma omp task default(shared) firstprivate(task)
	    for (int i=task*B; i<std::min((task+1)*B,N); i++)
	      work[i] = nrm2(M, &a[i*lda], 1);
	  }
#pragma omp taskwait
	} else for (int i=0; i<N; i++) work[i] = nrm2(M, &a[i*lda], 1);
      }
      geqp3tol(M, N, a, lda, jpvt, tau, work, ilwork, info, rank, rtol, atol, depth);
      delete[] work;
    }

    template<typename scalar> inline void geqrf(int M, int N, scalar* a, int lda, scalar* tau, scalar* work, int lwork, int* info);
    template<> inline void geqrf(int M, int N, float* a, int lda, float* tau, float* work, int lwork, int* info)
    { sgeqrf_(&M, &N, a, &lda, tau, work, &lwork, info); }
    template<> inline void geqrf(int M, int N, double* a, int lda, double* tau, double* work, int lwork, int* info)
    { dgeqrf_(&M, &N, a, &lda, tau, work, &lwork, info); }
    template<> inline void geqrf(int M, int N, c_float* a, int lda, c_float* tau, c_float* work, int lwork, int* info)
    { cgeqrf_(&M, &N, a, &lda, tau, work, &lwork, info); }
    template<> inline void geqrf(int M, int N, c_double* a, int lda, c_double* tau, c_double* work, int lwork, int* info)
    { zgeqrf_(&M, &N, a, &lda, tau, work, &lwork, info); }
    template<typename scalar> inline void geqrf(int M, int N, scalar* a, int lda, scalar* tau, int* info) {
      scalar lwork;
      geqrf(M, N, a, lda, tau, &lwork, -1, info);
      int ilwork = int(std::real(lwork));
      auto work = new scalar[ilwork];
      geqrf(M, N, a, lda, tau, work, ilwork, info);
      STRUMPACK_FLOPS((is_complex<scalar>()?4:1)*
		      static_cast<long long int>(((M>N) ? (double(N)*(double(N)*(.5-(1./3.)*double(N)+double(M)) + double(M) + 23./6.)) : (double(M)*(double(M)*(-.5-(1./3.)*double(M)+double(N)) + 2.*double(N) + 23./6.)))
						 + ((M>N) ? (double(N)*(double(N)*(.5-(1./3.)*double(N)+double(M)) + 5./6.)) : (double(M)*(double(M)*(-.5-(1./3.)*double(M)+double(N)) + double(N) + 5./6.)))));
      delete[] work;
    }

    template<typename scalar> inline void geqrfmod(int M, int N, scalar* a, int lda, scalar* tau, scalar* work, int lwork, int* info, int depth);
    template<> inline void geqrfmod(int M, int N, float* a, int lda, float* tau, float* work, int lwork, int* info, int depth)
    { sgeqrfmod_(&M, &N, a, &lda, tau, work, &lwork, info, &depth); }
    template<> inline void geqrfmod(int M, int N, double* a, int lda, double* tau, double* work, int lwork, int* info, int depth)
    { dgeqrfmod_(&M, &N, a, &lda, tau, work, &lwork, info, &depth); }
    template<> inline void geqrfmod(int M, int N, c_float* a, int lda, c_float* tau, c_float* work, int lwork, int* info, int depth)
    { cgeqrfmod_(&M, &N, a, &lda, tau, work, &lwork, info, &depth); }
    template<> inline void geqrfmod(int M, int N, c_double* a, int lda, c_double* tau, c_double* work, int lwork, int* info, int depth)
    { zgeqrfmod_(&M, &N, a, &lda, tau, work, &lwork, info, &depth); }
    template<typename scalar> inline void geqrfmod(int M, int N, scalar* a, int lda, scalar* tau, int* info, int depth) {
      scalar lwork;
      geqrfmod(M, N, a, lda, tau, &lwork, -1, info, depth);
      int ilwork = int(std::real(lwork));
      auto work = new scalar[ilwork];
      geqrfmod(M, N, a, lda, tau, work, ilwork, info, depth);
      delete[] work;
    }

    template<typename scalar> inline void gelqf(int M, int N, scalar* a, int lda, scalar* tau, scalar* work, int lwork, int* info);
    template<> inline void gelqf(int M, int N, float* a, int lda, float* tau, float* work, int lwork, int* info)
    { sgelqf_(&M, &N, a, &lda, tau, work, &lwork, info); }
    template<> inline void gelqf(int M, int N, double* a, int lda, double* tau, double* work, int lwork, int* info)
    { dgelqf_(&M, &N, a, &lda, tau, work, &lwork, info); }
    template<> inline void gelqf(int M, int N, c_float* a, int lda, c_float* tau, c_float* work, int lwork, int* info)
    { cgelqf_(&M, &N, a, &lda, tau, work, &lwork, info); }
    template<> inline void gelqf(int M, int N, c_double* a, int lda, c_double* tau, c_double* work, int lwork, int* info)
    { zgelqf_(&M, &N, a, &lda, tau, work, &lwork, info); }

    template<typename scalar> inline void gelqf(int M, int N, scalar* a, int lda, scalar* tau, int* info) {
      scalar lwork;
      gelqf(M, N, a, lda, tau, &lwork, -1, info);
      int ilwork = int(std::real(lwork));
      scalar* work = new scalar[ilwork];
      gelqf(M, N, a, lda, tau, work, ilwork, info);
      STRUMPACK_FLOPS((is_complex<scalar>()?4:1) * static_cast<long long int>(((M>N) ? (double(N)*(double(N)*(.5-(1./3.)*double(N)+double(M)) + double(M) + 29./6.)) : (double(M)*(double(M)*(-.5-(1./3.)*double(M)+double(N)) + 2.*double(N) + 29./6.)))
									      + ((M>N) ? (double(N)*(double(N)*(-.5-(1./3.)*double(N)+double(M)) + double(M) + 5./6.)) : (double(M)*(double(M)*(.5-(1./3.)*double(M)+double(N)) + 5./6.)))));
      delete[] work;
    }

    template<typename scalar> inline void gelqfmod(int M, int N, scalar* a, int lda, scalar* tau, scalar* work, int lwork, int* info, int depth);
    template<> inline void gelqfmod(int M, int N, float* a, int lda, float* tau, float* work, int lwork, int* info, int depth)
    { sgelqfmod_(&M, &N, a, &lda, tau, work, &lwork, info, &depth); }
    template<> inline void gelqfmod(int M, int N, double* a, int lda, double* tau, double* work, int lwork, int* info, int depth)
    { dgelqfmod_(&M, &N, a, &lda, tau, work, &lwork, info, &depth); }
    template<> inline void gelqfmod(int M, int N, c_float* a, int lda, c_float* tau, c_float* work, int lwork, int* info, int depth)
    { cgelqfmod_(&M, &N, a, &lda, tau, work, &lwork, info, &depth); }
    template<> inline void gelqfmod(int M, int N, c_double* a, int lda, c_double* tau, c_double* work, int lwork, int* info, int depth)
    { zgelqfmod_(&M, &N, a, &lda, tau, work, &lwork, info, &depth); }

    template<typename scalar> inline void gelqfmod(int M, int N, scalar* a, int lda, scalar* tau, int* info, int depth) {
      scalar lwork;
      gelqfmod(M, N, a, lda, tau, &lwork, -1, info, depth);
      int ilwork = int(std::real(lwork));
      scalar* work = new scalar[ilwork];
      gelqfmod(M, N, a, lda, tau, work, ilwork, info, depth);
      delete[] work;
    }

    template<typename scalar> inline void getrf(int M, int N, scalar* a, int lda, int* ipiv, int* info);
    template<> inline void getrf(int M, int N, float* a, int lda, int* ipiv, int* info)
    { sgetrf_(&M, &N, a, &lda, ipiv, info);
      STRUMPACK_FLOPS(static_cast<long long int>(((M<N) ? (.5*double(M)*(double(M)*(double(N)-(1./3.)*double(M)-1.) + double(N)) + (2./3.) * double(M)) : (.5*double(N)*(double(N)*(double(M)-(1./3.)*double(N)-1.) + double(M)) + (2./3.) * double(N)))
						 + ((M<N) ? (.5*double(M)*(double(M)*(double(N)-(1./3.)*double(M)) - double(N)) + (1./6.) * double(M)) : (.5*double(N)*(double(N)*(double(M)-(1./3.)*double(N)-1.) - double(M)) + (1./6.) * double(N))))); }
    template<> inline void getrf(int M, int N, double* a, int lda, int* ipiv, int* info)
    { dgetrf_(&M, &N, a, &lda, ipiv, info);
      STRUMPACK_FLOPS(static_cast<long long int>(((M<N) ? (.5*double(M)*(double(M)*(double(N)-(1./3.)*double(M)-1.) + double(N)) + (2./3.) * double(M)) : (.5*double(N)*(double(N)*(double(M)-(1./3.)*double(N)-1.) + double(M)) + (2./3.) * double(N)))
						 + ((M<N) ? (.5*double(M)*(double(M)*(double(N)-(1./3.)*double(M)) - double(N)) + (1./6.) * double(M)) : (.5*double(N)*(double(N)*(double(M)-(1./3.)*double(N)-1.) - double(M)) + (1./6.) * double(N))))); }
    template<> inline void getrf(int M, int N, c_float* a, int lda, int* ipiv, int* info)
    { cgetrf_(&M, &N, a, &lda, ipiv, info);
      STRUMPACK_FLOPS(4*static_cast<long long int>(((M<N) ? (.5*double(M)*(double(M)*(double(N)-(1./3.)*double(M)-1.) + double(N)) + (2./3.) * double(M)) : (.5*double(N)*(double(N)*(double(M)-(1./3.)*double(N)-1.) + double(M)) + (2./3.) * double(N)))
						   + ((M<N) ? (.5*double(M)*(double(M)*(double(N)-(1./3.)*double(M)) - double(N)) + (1./6.) * double(M)) : (.5*double(N)*(double(N)*(double(M)-(1./3.)*double(N)-1.) - double(M)) + (1./6.) * double(N))))); }
    template<> inline void getrf(int M, int N, c_double* a, int lda, int* ipiv, int* info)
    { zgetrf_(&M, &N, a, &lda, ipiv, info);
      STRUMPACK_FLOPS(4*static_cast<long long int>(((M<N) ? (.5*double(M)*(double(M)*(double(N)-(1./3.)*double(M)-1.) + double(N)) + (2./3.) * double(M)) : (.5*double(N)*(double(N)*(double(M)-(1./3.)*double(N)-1.) + double(M)) + (2./3.) * double(N)))
						   + ((M<N) ? (.5*double(M)*(double(M)*(double(N)-(1./3.)*double(M)) - double(N)) + (1./6.) * double(M)) : (.5*double(N)*(double(N)*(double(M)-(1./3.)*double(N)-1.) - double(M)) + (1./6.) * double(N))))); }

    template<typename scalar> inline void getrfmod(int M, int N, scalar* a, int lda, int* ipiv, int* info, int depth);
    template<> inline void getrfmod(int M, int N, float* a, int lda, int* ipiv, int* info, int depth)
    { sgetrfmod_(&M, &N, a, &lda, ipiv, info, &depth); }
    template<> inline void getrfmod(int M, int N, double* a, int lda, int* ipiv, int* info, int depth)
    { dgetrfmod_(&M, &N, a, &lda, ipiv, info, &depth); }
    template<> inline void getrfmod(int M, int N, c_float* a, int lda, int* ipiv, int* info, int depth)
    { cgetrfmod_(&M, &N, a, &lda, ipiv, info, &depth); }
    template<> inline void getrfmod(int M, int N, c_double* a, int lda, int* ipiv, int* info, int depth)
    { zgetrfmod_(&M, &N, a, &lda, ipiv, info, &depth); }

    template<typename scalar> inline void getrs(char trans, int N, int nrhs, scalar* a, int lda, int* ipiv, scalar* b, int ldb, int* info);
    template<> inline void getrs(char trans, int N, int nrhs, float* a, int lda, int* ipiv, float* b, int ldb, int* info)
    { sgetrs_(&trans, &N, &nrhs, a, &lda, ipiv, b, &ldb, info);
      STRUMPACK_FLOPS(static_cast<long long int>(2.*double(N)*double(N)*double(nrhs))); }
    template<> inline void getrs(char trans, int N, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb, int* info)
    { dgetrs_(&trans, &N, &nrhs, a, &lda, ipiv, b, &ldb, info);
      STRUMPACK_FLOPS(static_cast<long long int>(2.*double(N)*double(N)*double(nrhs))); }
    template<> inline void getrs(char trans, int N, int nrhs, c_float* a, int lda, int* ipiv, c_float* b, int ldb, int* info)
    { cgetrs_(&trans, &N, &nrhs, a, &lda, ipiv, b, &ldb, info);
      STRUMPACK_FLOPS(4*static_cast<long long int>(2.*double(N)*double(N)*double(nrhs))); }
    template<> inline void getrs(char trans, int N, int nrhs, c_double* a, int lda, int* ipiv, c_double* b, int ldb, int* info)
    { zgetrs_(&trans, &N, &nrhs, a, &lda, ipiv, b, &ldb, info);
      STRUMPACK_FLOPS(4*static_cast<long long int>(2.*double(N)*double(N)*double(nrhs))); }

    template<typename scalar> inline void potrf(char uplo, int N, scalar* a, int lda, int* info);
    template<> inline void potrf(char uplo, int N, float* a, int lda, int* info)
    { spotrf_(&uplo, &N, a, &lda, info); std::cout << "TODO count flops for spotrf" << std::endl; }
    template<> inline void potrf(char uplo, int N, double* a, int lda, int* info)
    { dpotrf_(&uplo, &N, a, &lda, info); std::cout << "TODO count flops for dpotrf" << std::endl; }
    template<> inline void potrf(char uplo, int N, c_float* a, int lda, int* info)
    { cpotrf_(&uplo, &N, a, &lda, info); std::cout << "TODO count flops for cpotrf" << std::endl; }
    template<> inline void potrf(char uplo, int N, c_double* a, int lda, int* info)
    { zpotrf_(&uplo, &N, a, &lda, info); std::cout << "TODO count flops for zpotrf" << std::endl; }

    template<typename scalar> inline void xxglq(int M, int N, int K, scalar* a, int lda, scalar* tau, scalar* work, int lwork, int* info);
    template<> inline void xxglq(int M, int N, int K, float* a, int lda, float* tau, float* work, int lwork, int* info)
    { sorglq_(&M, &N, &K, a, &lda, tau, work, &lwork, info); }
    template<> inline void xxglq(int M, int N, int K, double* a, int lda, double* tau, double* work, int lwork, int* info)
    { dorglq_(&M, &N, &K, a, &lda, tau, work, &lwork, info); }
    template<> inline void xxglq(int M, int N, int K, c_float* a, int lda, c_float* tau, c_float* work, int lwork, int* info)
    { cunglq_(&M, &N, &K, a, &lda, tau, work, &lwork, info); }
    template<> inline void xxglq(int M, int N, int K, c_double* a, int lda, c_double* tau, c_double* work, int lwork, int* info)
    { zunglq_(&M, &N, &K, a, &lda, tau, work, &lwork, info); }

    template<typename scalar> inline void xxglq(int M, int N, int K, scalar* a, int lda, scalar* tau, int* info) {
      scalar lwork;
      xxglq(M, N, K, a, lda, tau, &lwork, -1, info);
      int ilwork = int(std::real(lwork));
      scalar* work = new scalar[ilwork];
      xxglq(M, N, K, a, lda, tau, work, ilwork, info);
      STRUMPACK_FLOPS((is_complex<scalar>()?4:1) * static_cast<long long int>((M==K) ? ((2./3.)*double(M)*double(M)*(3.*double(N) - double(M))) : 4.*double(M)*double(N)*double(K) - 2.*(double(M) + double(N))*double(K)*double(K)+ (4./3.)*double(K)*double(K)*double(K)));
      delete[] work;
    }

    template<typename scalar> inline void xxglqmod(int M, int N, int K, scalar* a, int lda, scalar* tau, scalar* work, int lwork, int* info, int depth);
    template<> inline void xxglqmod(int M, int N, int K, float* a, int lda, float* tau, float* work, int lwork, int* info, int depth)
    { sorglqmod_(&M, &N, &K, a, &lda, tau, work, &lwork, info, &depth); }
    template<> inline void xxglqmod(int M, int N, int K, double* a, int lda, double* tau, double* work, int lwork, int* info, int depth)
    { dorglqmod_(&M, &N, &K, a, &lda, tau, work, &lwork, info, &depth); }
    template<> inline void xxglqmod(int M, int N, int K, c_float* a, int lda, c_float* tau, c_float* work, int lwork, int* info, int depth)
    { cunglqmod_(&M, &N, &K, a, &lda, tau, work, &lwork, info, &depth); }
    template<> inline void xxglqmod(int M, int N, int K, c_double* a, int lda, c_double* tau, c_double* work, int lwork, int* info, int depth)
    { zunglqmod_(&M, &N, &K, a, &lda, tau, work, &lwork, info, &depth); }

    template<typename scalar> inline void xxglqmod(int M, int N, int K, scalar* a, int lda, scalar* tau, int* info, int depth) {
      scalar lwork;
      xxglqmod(M, N, K, a, lda, tau, &lwork, -1, info, depth);
      int ilwork = int(std::real(lwork));
      scalar* work = new scalar[ilwork];
      xxglqmod(M, N, K, a, lda, tau, work, ilwork, info, depth);
      delete[] work;
    }

    template<typename scalar> inline void xxgqr(int M, int N, int K, scalar* a, int lda, scalar* tau, scalar* work, int lwork, int* info);
    template<> inline void xxgqr(int M, int N, int K, float* a, int lda, float* tau, float* work, int lwork, int* info)
    { sorgqr_(&M, &N, &K, a, &lda, tau, work, &lwork, info); }
    template<> inline void xxgqr(int M, int N, int K, double* a, int lda, double* tau, double* work, int lwork, int* info)
    { dorgqr_(&M, &N, &K, a, &lda, tau, work, &lwork, info); }
    template<> inline void xxgqr(int M, int N, int K, c_float* a, int lda, c_float* tau, c_float* work, int lwork, int* info)
    { cungqr_(&M, &N, &K, a, &lda, tau, work, &lwork, info); }
    template<> inline void xxgqr(int M, int N, int K, c_double* a, int lda, c_double* tau, c_double* work, int lwork, int* info)
    { zungqr_(&M, &N, &K, a, &lda, tau, work, &lwork, info); }

    template<typename scalar> inline void xxgqr(int M, int N, int K, scalar* a, int lda, scalar* tau, int* info) {
      scalar lwork;
      xxgqr(M, N, K, a, lda, tau, &lwork, -1, info);
      int ilwork = int(std::real(lwork));
      scalar* work = new scalar[ilwork];
      xxgqr(M, N, K, a, lda, tau, work, ilwork, info);
      STRUMPACK_FLOPS((is_complex<scalar>()?4:1)*static_cast<long long int>((N==K) ? ((2./3.)*double(N)*double(N)*(3.*double(M) - double(N))) : (4.*double(M)*double(N)*double(K) - 2.*(double(M) + double(N))*double(K)*double(K) + (4./3.)*double(K)*double(K)*double(K))));
      delete[] work;
    }

    template<typename scalar, typename real> inline real lange(char norm, int m, int n, scalar *a, int lda);
    template<> inline float lange<float, float>(char norm, int m, int n, float *a, int lda) {
      if (norm == 'I' || norm == 'i') {
	float* work = new float[m];
	auto ret = slange_(&norm, &m, &n, a, &lda, work);
	delete[] work;
	return ret;
      } else return slange_(&norm, &m, &n, a, &lda, nullptr);
    }
    template<> inline double lange<double,double>(char norm, int m, int n, double *a, int lda) {
      if (norm == 'I' || norm == 'i') {
	double* work = new double[m];
	auto ret = dlange_(&norm, &m, &n, a, &lda, work);
	delete[] work;
	return ret;
      } else return dlange_(&norm, &m, &n, a, &lda, nullptr);
    }
    template<> inline  float lange<c_float, float>(char norm, int m, int n, c_float *a, int lda) {
      if (norm == 'I' || norm == 'i') {
	float* work = new float[m];
	auto ret = clange_(&norm, &m, &n, a, &lda, work);
	delete[] work;
	return ret;
      } else return clange_(&norm, &m, &n, a, &lda, nullptr);
    }
    template<> inline double lange<c_double, double>(char norm, int m, int n, c_double *a, int lda) {
      if (norm == 'I' || norm == 'i') {
	double* work = new double[m];
	auto ret = zlange_(&norm, &m, &n, a, &lda, work);
	delete[] work;
	return ret;
      } else return zlange_(&norm, &m, &n, a, &lda, nullptr);
    }


    inline int
    gesvd(char JOBU, char JOBVT, int M, int N, float* A, int LDA,
          float* S, float* U, int LDU, float* VT, int LDVT) {
      int INFO;
      int LWORK = -1;
      float SWORK;
      sgesvd_(&JOBU, &JOBVT, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT,
              &SWORK, &LWORK, &INFO);
      LWORK = int(SWORK);
      auto WORK = new float[LWORK];
      sgesvd_(&JOBU, &JOBVT, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT,
              WORK, &LWORK, &INFO);
      delete[] WORK;
      return INFO;
    }
    inline int
    gesvd(char JOBU, char JOBVT, int M, int N, double* A, int LDA,
          double* S, double* U, int LDU, double* VT, int LDVT) {
      int INFO;
      int LWORK = -1;
      double DWORK;
      dgesvd_(&JOBU, &JOBVT, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT,
              &DWORK, &LWORK, &INFO);
      LWORK = int(DWORK);
      auto WORK = new double[LWORK];
      dgesvd_(&JOBU, &JOBVT, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT,
              WORK, &LWORK, &INFO);
      delete[] WORK;
      return INFO;
    }
    inline int
    gesvd(char JOBU, char JOBVT, int M, int N, c_float* A, int LDA,
          c_float* S, c_float* U, int LDU, c_float* VT, int LDVT) {
      std::cout << "TODO gesvd for c_float" << std::endl;
      return 0;
    }
    inline int
    gesvd(char JOBU, char JOBVT, int M, int N, c_double* A, int LDA,
          c_double* S, c_double* U, int LDU, c_double* VT, int LDVT) {
      std::cout << "TODO gesvd for c_double" << std::endl;
      return 0;
    }


  } //end namespace blas
} // end namespace strumpack

#endif // BLASLAPACKWRAPPER_H
