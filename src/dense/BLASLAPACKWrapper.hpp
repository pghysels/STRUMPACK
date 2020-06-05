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
#ifndef STRUMPACK_BLASLAPACKWRAPPER_HPP
#define STRUMPACK_BLASLAPACKWRAPPER_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <cassert>
#include <memory>
#include "StrumpackParameters.hpp"

namespace strumpack {

  template<typename scalar> bool is_complex() { return false; }
  template<> inline bool is_complex<std::complex<float>>() { return true; }
  template<> inline bool is_complex<std::complex<double>>() { return true; }

  template<class T> struct RealType { typedef T value_type; };
  template<class T> struct RealType<std::complex<T>> { typedef T value_type; };

  namespace blas {

    inline bool my_conj(bool a) { return a; }
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

    int ilaenv(int ispec, char name[], char opts[],
               int n1, int n2, int n3, int n4);

    template<typename real> inline real lamch(char cmach);
    template<> float lamch<float>(char cmach);
    template<> double lamch<double>(char cmach);


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
    void gemm(char ta, char tb, int m, int n, int k, float alpha,
              const float *a, int lda, const float *b, int ldb,
              float beta, float *c, int ldc);
    void gemm(char ta, char tb, int m, int n, int k, double alpha,
              const double *a, int lda, const double *b, int ldb,
              double beta, double *c, int ldc);
    void gemm(char ta, char tb, int m, int n, int k,
              std::complex<float> alpha,
              const std::complex<float>* a, int lda,
              const std::complex<float>* b, int ldb,
              std::complex<float> beta,
              std::complex<float>* c, int ldc);
    void gemm(char ta, char tb, int m, int n, int k,
              std::complex<double> alpha,
              const std::complex<double>* a, int lda,
              const std::complex<double>* b, int ldb,
              std::complex<double> beta,
              std::complex<double>* c, int ldc);

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
    void gemv(char t, int m, int n, float alpha, const float *a, int lda,
              const float *x, int incx, float beta, float *y, int incy);
    void gemv(char t, int m, int n, double alpha, const double *a, int lda,
              const double *x, int incx, double beta, double *y, int incy);
    void gemv(char t, int m, int n, std::complex<float> alpha,
              const std::complex<float> *a, int lda,
              const std::complex<float> *x, int incx,
              std::complex<float> beta,
              std::complex<float> *y, int incy);
    void gemv(char t, int m, int n, std::complex<double> alpha,
              const std::complex<double> *a, int lda,
              const std::complex<double> *x, int incx,
              std::complex<double> beta,
              std::complex<double> *y, int incy);

    template<typename scalar_t> inline long long ger_flops
    (long long m, long long n, scalar_t alpha) {
      // TODO check this?
      return (alpha != scalar_t(0)) * m * n +
        (alpha != scalar_t(0) && alpha != scalar_t(1)) * m * n +
        (alpha != scalar_t(0)) * m * n;
    }
    void geru(int m, int n, float alpha, const float* x, int incx,
              const float* y, int incy, float* a, int lda);
    void geru(int m, int n, double alpha, const double* x, int incx,
              const double* y, int incy, double* a, int lda);
    void geru(int m, int n, std::complex<float> alpha,
              const std::complex<float>* x, int incx,
              const std::complex<float>* y, int incy,
              std::complex<float>* a, int lda);
    void geru(int m, int n, std::complex<double> alpha,
              const std::complex<double>* x, int incx,
              const std::complex<double>* y, int incy,
              std::complex<double>* a, int lda);

    void gerc(int m, int n, float alpha, const float* x, int incx,
              const float* y, int incy, float* a, int lda);
    void gerc(int m, int n, double alpha, const double* x, int incx,
              const double* y, int incy, double* a, int lda);
    void gerc(int m, int n, std::complex<float> alpha,
              const std::complex<float>* x, int incx,
              const std::complex<float>* y, int incy,
              std::complex<float>* a, int lda);
    void gerc(int m, int n, std::complex<double> alpha,
              const std::complex<double>* x, int incx,
              const std::complex<double>* y, int incy,
              std::complex<double>* a, int lda);

    void lacgv(int, float *, int );
    void lacgv(int, double *, int );
    void lacgv(int n, std::complex<float> *x, int incx);
    void lacgv(int n, std::complex<double> *x, int incx);

    void lacpy(char ul, int m, int n, float* a, int lda, float* b, int ldb);
    void lacpy(char ul, int m, int n, double* a, int lda, double* b, int ldb);
    void lacpy(char ul, int m, int n, std::complex<float>* a, int lda,
               std::complex<float>* b, int ldb);
    void lacpy(char ul, int m, int n, std::complex<double>* a, int lda,
               std::complex<double>* b, int ldb);

    template<typename scalar_t> inline long long axpy_flops
    (long long n, scalar_t alpha) {
      return (alpha != scalar_t(0) && alpha != scalar_t(1)) * n +
        (alpha != scalar_t(0)) * n;
    }

    void axpy(int n, float alpha, float* x, int incx, float* y, int incy);
    void axpy(int n, double alpha, double* x, int incx, double* y, int incy);
    void axpy(int n, std::complex<float> alpha,
              const std::complex<float>* x, int incx,
              std::complex<float>* y, int incy);
    void axpy(int n, std::complex<double> alpha,
              const std::complex<double>* x, int incx,
              std::complex<double>* y, int incy);

    void copy(int n, const float* x, int incx, float* y, int incy);
    void copy(int n, const double* x, int incx, double* y, int incy);
    void copy(int n, const std::complex<float>* x, int incx,
              std::complex<float>* y, int incy);
    void copy(int n, const std::complex<double>* x, int incx,
              std::complex<double>* y, int incy);

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
    void scal(int n, float alpha, float* x, int incx);
    void scal(int n, double alpha, double* x, int incx);
    void scal(int n, std::complex<float> alpha,
              std::complex<float>* x, int incx);
    void scal(int n, std::complex<double> alpha,
              std::complex<double>* x, int incx);

    int iamax(int n, const float* x, int incx);
    int iamax(int n, const double* x, int incx);
    int iamax(int n, const std::complex<float>* x, int incx);
    int iamax(int n, const std::complex<double>* x, int incx);

    void swap(int n, float* x, int incx, float* y, int incy);
    void swap(int n, double* x, int incx, double* y, int incy);
    void swap(int n, std::complex<float>* x, int incx,
              std::complex<float>* y, int incy);
    void swap(int n, std::complex<double>* x, int incx,
              std::complex<double>* y, int incy);

    inline long long nrm2_flops(long long n) {
      return n * 2;
    }
    float nrm2(int n, const float* x, int incx);
    double nrm2(int n, const double* x, int incx);
    float nrm2(int n, const std::complex<float>* x, int incx);
    double nrm2(int n, const std::complex<double>* x, int incx);

    inline long long dot_flops(long long n) {
      return 2 * n;
    }
    float dotu(int n, const float* x, int incx, const float* y, int incy);
    double dotu(int n, const double* x, int incx, const double* y, int incy);
    float dotc(int n, const float* x, int incx, const float* y, int incy);
    double dotc(int n, const double* x, int incx, const double* y, int incy);

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
    std::complex<float> dotu(int n, const std::complex<float>* x, int incx,
                             const std::complex<float>* y, int incy);
    std::complex<double> dotu(int n, const std::complex<double>* x, int incx,
                              const std::complex<double>* y, int incy);
    std::complex<float> dotc(int n, const std::complex<float>* x, int incx,
                             const std::complex<float>* y, int incy);
    std::complex<double> dotc(int n, const std::complex<double>* x, int incx,
                              const std::complex<double>* y, int incy);


    inline long long laswp_moves(long long n, long long k1, long long k2) {
      return 2 * (k2 - k1) * n;
    }
    void laswp(int n, float* a, int lda, int k1, int k2,
               const int* ipiv, int incx);
    void laswp(int n, double* a, int lda, int k1, int k2,
               const int* ipiv, int incx);
    void laswp(int n, std::complex<float>* a, int lda, int k1, int k2,
               const int* ipiv, int incx);
    void laswp(int n, std::complex<double>* a, int lda, int k1, int k2,
               const int* ipiv, int incx);

    inline long long lapmr_moves(long long n, long long m) {
      return 2 * m * n;
    }
    void lapmr(bool fwd, int m, int n, float* a, int lda, const int* ipiv);
    void lapmr(bool fwd, int m, int n, double* a, int lda, const int* ipiv);
    void lapmr(bool fwd, int m, int n, std::complex<float>* a, int lda,
               const int* ipiv);
    void lapmr(bool fwd, int m, int n, std::complex<double>* a, int lda,
               const int* ipiv);

    inline long long lapmt_moves(long long n, long long m) {
      return 2 * m * n;
    }
    void lapmt(bool fwd, int m, int n, float* a, int lda, const int* ipiv);
    void lapmt(bool fwd, int m, int n, double* a, int lda, const int* ipiv);
    void lapmt(bool fwd, int m, int n, std::complex<float>* a, int lda,
               const int* ipiv);
    void lapmt(bool fwd, int m, int n, std::complex<double>* a, int lda,
               const int* ipiv);

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
    void trsm(char s, char ul, char t, char d, int m, int n, float alpha,
              const float* a, int lda, float* b, int ldb);
    void trsm(char s, char ul, char t, char d, int m, int n, double alpha,
              const double* a, int lda, double* b, int ldb);
    void trsm(char s, char ul, char t, char d, int m, int n,
              std::complex<float> alpha,
              const std::complex<float>* a, int lda,
              std::complex<float>* b, int ldb);
    void trsm(char s, char ul, char t, char d, int m, int n,
              std::complex<double> alpha,
              const std::complex<double>* a, int lda,
              std::complex<double>* b, int ldb);

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
    void trmm(char s, char ul, char t, char d, int m, int n, float alpha,
              const float* a, int lda, float* b, int ldb);
    void trmm(char s, char ul, char t, char d, int m, int n, double alpha,
              const double* a, int lda, double* b, int ldb);
    void trmm(char s, char ul, char t, char d, int m, int n,
              std::complex<float> alpha,
              const std::complex<float>* a, int lda,
              std::complex<float>* b, int ldb);
    void trmm(char s, char ul, char t, char d, int m, int n,
              std::complex<double> alpha,
              const std::complex<double>* a, int lda,
              std::complex<double>* b, int ldb);

    inline long long trmv_flops(long long n) {
      return n * (n + 1);
    }
    inline long long trmv_moves(long long n) {
      return n * n / 2 + 2 * n;
    }
    void trmv(char ul, char t, char d, int n, const float* a, int lda,
              float* x, int incx);
    void trmv(char ul, char t, char d, int n, const double* a, int lda,
              double* x, int incx);
    void trmv(char ul, char t, char d, int n,
              const std::complex<float>* a, int lda,
              std::complex<float>* x, int incx);
    void trmv(char ul, char t, char d, int n,
              const std::complex<double>* a, int lda,
              std::complex<double>* x, int incx);

    inline long long trsv_flops(long long m) {
      return m * (m + 1);
    }
    inline long long trsv_moves(long long m) {
      return m * m / 2 + 2 * m;
    }
    void trsv(char ul, char t, char d, int m, const float* a, int lda,
              float* b, int incb);
    void trsv(char ul, char t, char d, int m, const double* a, int lda,
              double* b, int incb);
    void trsv(char ul, char t, char d, int m,
              const std::complex<float>* a, int lda,
              std::complex<float>* b, int incb);
    void trsv(char ul, char t, char d, int m,
              const std::complex<double>* a, int lda,
              std::complex<double>* b, int incb);

    void laset(char s, int m, int n, float alpha,
               float beta, float* x, int ldx);
    void laset(char s, int m, int n, double alpha,
               double beta, double* x, int ldx);
    void laset(char s, int m, int n, std::complex<float> alpha,
               std::complex<float> beta, std::complex<float>* x, int ldx);
    void laset(char s, int m, int n, std::complex<double> alpha,
               std::complex<double> beta, std::complex<double>* x, int ldx);

    inline long long geqp3_flops(long long m, long long n) {
      if (m == n) return n * n * n * 4 / 3;
      else {
        if (m > n) return n * n * 2 / 3 * (3 * m - n);
        else return m * m * 2 / 3 * (3 * n - m);
      }
    }
    int geqp3(int m, int n, float* a, int lda, int* jpvt, float* tau,
              float* work, int lwork);
    int geqp3(int m, int n, double* a, int lda, int* jpvt, double* tau,
              double* work, int lwork);
    int geqp3(int m, int n, std::complex<float>* a, int lda, int* jpvt,
              std::complex<float>* tau, std::complex<float>* work, int lwork);
    int geqp3(int m, int n, std::complex<double>* a, int lda, int* jpvt,
              std::complex<double>* tau, std::complex<double>* work,
              int lwork);
    template<typename scalar> inline int geqp3
    (int m, int n, scalar* a, int lda, int* jpvt, scalar* tau) {
      scalar lwork;
      geqp3(m, n, a, lda, jpvt, tau, &lwork, -1);
      int ilwork = int(std::real(lwork));
      std::unique_ptr<scalar[]> work(new scalar[ilwork]);
      return geqp3(m, n, a, lda, jpvt, tau, work.get(), ilwork);
    }


    int geqp3tol(int m, int n, float* a, int lda, int* jpvt,
                 float* tau, float* work, int lwork, int& rank,
                 float rtol, float atol, int depth);
    int geqp3tol(int m, int n, double* a, int lda, int* jpvt,
                 double* tau, double* work, int lwork, int& rank,
                 double rtol, double atol, int depth);
    int geqp3tol(int m, int n, std::complex<float>* a, int lda, int* jpvt,
                 std::complex<float>* tau, std::complex<float>* work,
                 int lwork, int& rank, float rtol, float atol, int depth);
    int geqp3tol(int m, int n, std::complex<double>* a, int lda, int* jpvt,
                 std::complex<double>* tau, std::complex<double>* work,
                 int lwork, int& rank, double rtol, double atol, int depth);

    template<typename scalar, typename real> inline int geqp3tol
    (int m, int n, scalar* a, int lda, int* jpvt, scalar* tau,
     int& rank, real rtol, real atol, int depth) {
      scalar lwork;
      geqp3tol
        (m, n, a, lda, jpvt, tau, &lwork, -1, rank, rtol, atol, depth);
      int ilwork = int(std::real(lwork));
      std::unique_ptr<scalar[]> work(new scalar[ilwork]);
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
      return geqp3tol
        (m, n, a, lda, jpvt, tau, work.get(), ilwork,
         rank, rtol, atol, depth);
    }


    inline long long geqrf_flops(long long m, long long n) {
      if (m > n)
        return n * (n * (.5-(1./3.)*n+m) + m + 23./6.) +
          n * (n * (.5-(1./3.) * n + m) + 5./6.);
      else
        return m * (m * (-.5 - (1./3.) * m + n) + 2 * n + 23./6.) +
          m * (m * (-.5 - (1./3.) * m + n) + n + 5./6.);
    }
    int geqrf(int m, int n, float* a, int lda, float* tau,
              float* work, int lwork);
    int geqrf(int m, int n, double* a, int lda, double* tau,
              double* work, int lwork);
    int geqrf(int m, int n, std::complex<float>* a, int lda,
              std::complex<float>* tau, std::complex<float>* work,
              int lwork);
    int geqrf(int m, int n, std::complex<double>* a, int lda,
              std::complex<double>* tau, std::complex<double>* work,
              int lwork);
    template<typename scalar> inline int geqrf
    (int m, int n, scalar* a, int lda, scalar* tau) {
      scalar lwork;
      geqrf(m, n, a, lda, tau, &lwork, -1);
      int ilwork = int(std::real(lwork));
      std::unique_ptr<scalar[]> work(new scalar[ilwork]);
      return geqrf(m, n, a, lda, tau, work.get(), ilwork);
    }

    int geqrfmod(int m, int n, float* a, int lda,
                 float* tau, float* work, int lwork, int depth);
    int geqrfmod(int m, int n, double* a, int lda, double* tau,
                 double* work, int lwork, int depth);
    int geqrfmod(int m, int n, std::complex<float>* a, int lda,
                 std::complex<float>* tau, std::complex<float>* work,
                 int lwork, int depth);
    int geqrfmod(int m, int n, std::complex<double>* a, int lda,
                 std::complex<double>* tau, std::complex<double>* work,
                 int lwork, int depth);
    template<typename scalar> inline int geqrfmod
    (int m, int n, scalar* a, int lda, scalar* tau, int depth) {
      scalar lwork;
      geqrfmod(m, n, a, lda, tau, &lwork, -1, depth);
      int ilwork = int(std::real(lwork));
      std::unique_ptr<scalar[]> work(new scalar[ilwork]);
      return geqrfmod(m, n, a, lda, tau, work.get(), ilwork, depth);
    }


    inline long long gelqf_flops(long long m, long long n) {
      if (m > n)
        return n * (n * (.5 - (1./3.) * n + m) + m + 29./6.) +
                              n * (n * (-.5 - (1./3.) * n + m) + m + 5./6.);
      else
        return m * (m * (-.5 - (1./3.) * m + n) + 2 * n + 29./6.) +
                              m * (m * (.5 - (1./3.) * m + n) + 5./6.);
    }
    int gelqf(int m, int n, float* a, int lda, float* tau,
              float* work, int lwork);
    int gelqf(int m, int n, double* a, int lda, double* tau,
              double* work, int lwork);
    int gelqf(int m, int n, std::complex<float>* a, int lda,
              std::complex<float>* tau, std::complex<float>* work,
              int lwork);
    int gelqf(int m, int n, std::complex<double>* a, int lda,
              std::complex<double>* tau, std::complex<double>* work,
              int lwork);
    template<typename scalar> inline int gelqf
    (int m, int n, scalar* a, int lda, scalar* tau) {
      scalar lwork;
      gelqf(m, n, a, lda, tau, &lwork, -1);
      int ilwork = int(std::real(lwork));
      std::unique_ptr<scalar[]> work(new scalar[ilwork]);
      return gelqf(m, n, a, lda, tau, work.get(), ilwork);
    }


    void gelqfmod(int m, int n, float* a, int lda, float* tau,
                  float* work, int lwork, int* info, int depth);
    void gelqfmod(int m, int n, double* a, int lda, double* tau,
                  double* work, int lwork, int* info, int depth);
    void gelqfmod(int m, int n, std::complex<float>* a, int lda,
                  std::complex<float>* tau, std::complex<float>* work,
                  int lwork, int* info, int depth);
    void gelqfmod(int m, int n, std::complex<double>* a, int lda,
                  std::complex<double>* tau, std::complex<double>* work,
                  int lwork, int* info, int depth);
    template<typename scalar> inline void gelqfmod
    (int m, int n, scalar* a, int lda, scalar* tau, int* info, int depth) {
      scalar lwork;
      gelqfmod(m, n, a, lda, tau, &lwork, -1, info, depth);
      int ilwork = int(std::real(lwork));
      std::unique_ptr<scalar[]> work(new scalar[ilwork]);
      gelqfmod(m, n, a, lda, tau, work.get(), ilwork, info, depth);
    }


    inline long long getrf_flops(long long m, long long n) {
      // TODO check this
      if (m < n) return (m / 2 * (m * (n - m / 3 - 1) + n) + 2 * m / 3) +
                   (m / 2 * (m * (n - m / 3) - n) + m / 6);
      else return n * n * (m - n/3 - 1) / 2 + m + 2 * n / 3 +
             n * (n * (m - (1./3.) * n - 1) / 2 - m) + n / 6;
    }
    void getrf(int m, int n, float* a, int lda, int* ipiv, int* info);
    void getrf(int m, int n, double* a, int lda, int* ipiv, int* info);
    void getrf(int m, int n, std::complex<float>* a, int lda,
               int* ipiv, int* info);
    void getrf(int m, int n, std::complex<double>* a, int lda,
               int* ipiv, int* info);

    void getrfmod(int m, int n, float* a, int lda, int* ipiv,
                  int* info, int depth);
    void getrfmod(int m, int n, double* a, int lda, int* ipiv,
                  int* info, int depth);
    void getrfmod(int m, int n, std::complex<float>* a, int lda,
                  int* ipiv, int* info, int depth);
    void getrfmod(int m, int n, std::complex<double>* a, int lda,
                  int* ipiv, int* info, int depth);

    inline long long getrs_flops(long long n, long long nrhs) {
      return 2 * n * n * nrhs;
    }
    void getrs(char t, int n, int nrhs, const float* a, int lda,
               const int* ipiv, float* b, int ldb, int* info);
    void getrs(char t, int n, int nrhs, const double* a, int lda,
               const int* ipiv, double* b, int ldb, int* info);
    void getrs(char t, int n, int nrhs,
               const std::complex<float>* a, int lda,
               const int* ipiv, std::complex<float>* b, int ldb,
               int* info);
    void getrs(char t, int n, int nrhs,
               const std::complex<double>* a, int lda,
               const int* ipiv, std::complex<double>* b, int ldb,
               int* info);

    inline long long potrf_flops(long long n) {
      return n*n*n/6 + n*n/2 + n/3 + n*n*n/6 - n/6;
    }
    int potrf(char ul, int n, float* a, int lda);
    int potrf(char ul, int n, double* a, int lda);
    int potrf(char ul, int n, std::complex<float>* a, int lda);
    int potrf(char ul, int n, std::complex<double>* a, int lda);

    inline long long xxglq_flops(long long m, long long n, long long k) {
      if (m == k) return 2 * m * m *(3 * n - m) / 3;
      else return 4 * m * n * k - 2 * (m + n) * k * k + 4 * k * k * k / 3;
    }
    void xxglq(int m, int n, int k, float* a, int lda, const float* tau,
               float* work, int lwork, int* info);
    void xxglq(int m, int n, int k, double* a, int lda, const double* tau,
               double* work, int lwork, int* info);
    void xxglq(int m, int n, int k, std::complex<float>* a, int lda,
               const std::complex<float>* tau, std::complex<float>* work,
               int lwork, int* info);
    void xxglq(int m, int n, int k, std::complex<double>* a, int lda,
               const std::complex<double>* tau, std::complex<double>* work,
               int lwork, int* info);
    template<typename scalar> inline void xxglq
    (int m, int n, int k, scalar* a, int lda, const scalar* tau, int* info) {
      scalar lwork;
      xxglq(m, n, k, a, lda, tau, &lwork, -1, info);
      int ilwork = int(std::real(lwork));
      std::unique_ptr<scalar[]> work(new scalar[ilwork]);
      xxglq(m, n, k, a, lda, tau, work.get(), ilwork, info);
    }

    // do not count flops here, they are counted in the blas routines
    void xxglqmod(int m, int n, int k, float* a, int lda, const float* tau,
                  float* work, int lwork, int* info, int depth);
    void xxglqmod(int m, int n, int k, double* a, int lda, const double* tau,
                  double* work, int lwork, int* info, int depth);
    void xxglqmod(int m, int n, int k, std::complex<float>* a, int lda,
                  const std::complex<float>* tau, std::complex<float>* work,
                  int lwork, int* info, int depth);
    void xxglqmod(int m, int n, int k, std::complex<double>* a, int lda,
                  const std::complex<double>* tau, std::complex<double>* work,
                  int lwork, int* info, int depth);
    template<typename scalar> inline void xxglqmod
    (int m, int n, int k, scalar* a, int lda,
     const scalar* tau, int* info, int depth) {
      scalar lwork;
      xxglqmod(m, n, k, a, lda, tau, &lwork, -1, info, depth);
      int ilwork = int(std::real(lwork));
      std::unique_ptr<scalar[]> work(new scalar[ilwork]);
      xxglqmod(m, n, k, a, lda, tau, work.get(), ilwork, info, depth);
    }


    inline long long xxgqr_flops(long long m, long long n, long long k) {
      if (n == k) return 2 * n * n * (3 * m - n) / 3;
      else return 4 * m * n * k - 2 * (m + n) * k * k + 4 * k * k * k / 3;
    }
    int xxgqr(int m, int n, int k, float* a, int lda, const float* tau,
              float* work, int lwork);
    int xxgqr(int m, int n, int k, double* a, int lda, const double* tau,
              double* work, int lwork);
    int xxgqr(int m, int n, int k, std::complex<float>* a, int lda,
              const std::complex<float>* tau, std::complex<float>* work,
              int lwork);
    int xxgqr(int m, int n, int k, std::complex<double>* a, int lda,
              const std::complex<double>* tau, std::complex<double>* work,
              int lwork);
    template<typename scalar> inline int xxgqr
    (int m, int n, int k, scalar* a, int lda, const scalar* tau) {
      scalar lwork;
      xxgqr(m, n, k, a, lda, tau, &lwork, -1);
      int ilwork = int(std::real(lwork));
      std::unique_ptr<scalar[]> work(new scalar[ilwork]);
      return xxgqr(m, n, k, a, lda, tau, work.get(), ilwork);
    }


    inline long long xxmqr_flops(long long m, long long n, long long k) {
      // TODO
      // if (n == k) return 2 * n * n * (3 * m - n) / 3;
      // else return 4 * m * n * k - 2 * (m + n) * k * k + 4 * k * k * k / 3;
      return 0;
    }
    int xxmqr(char side, char trans, int m, int n, int k, float* a, int lda,
              const float* tau, float* c, int ldc, float* work, int lwork);
    int xxmqr(char side, char trans, int m, int n, int k, double* a, int lda,
              const double* tau, double* c, int ldc, double* work, int lwork);
    int xxmqr(char side, char trans, int m, int n, int k,
              std::complex<float>* a, int lda,
              const std::complex<float>* tau, std::complex<float>* c, int ldc,
              std::complex<float>* work, int lwork);
    int xxmqr(char side, char trans, int m, int n, int k,
              std::complex<double>* a, int lda,
              const std::complex<double>* tau,
              std::complex<double>* c, int ldc,
              std::complex<double>* work, int lwork);
    template<typename scalar> inline int xxmqr
    (char side, char trans, int m, int n, int k, scalar* a, int lda,
     const scalar* tau, scalar* c, int ldc) {
      scalar lwork;
      xxmqr(side, trans, m, n, k, a, lda, tau, c, ldc, &lwork, -1);
      int ilwork = int(std::real(lwork));
      std::unique_ptr<scalar[]> work(new scalar[ilwork]);
      return xxmqr
        (side, trans, m, n, k, a, lda, tau, c, ldc, work.get(), ilwork);
    }

    int lange(char norm, int m, int n, const int *a, int lda);
    unsigned int lange(char norm, int m, int n, const unsigned int *a, int lda);
    std::size_t lange(char norm, int m, int n, const std::size_t *a, int lda);
    bool lange(char norm, int m, int n, const bool *a, int lda);

    float lange(char norm, int m, int n, const float *a, int lda);
    double lange(char norm, int m, int n, const double *a, int lda);
    float lange(char norm, int m, int n,
                const std::complex<float> *a, int lda);
    double lange(char norm, int m, int n,
                 const std::complex<double> *a, int lda);

    int gesvd(char jobu, char jobvt, int m, int n, float* a, int lda,
              float* s, float* u, int ldu, float* vt, int ldvt);
    int gesvd(char jobu, char jobvt, int m, int n, double* a, int lda,
              double* s, double* u, int ldu, double* vt, int ldvt);
    int gesvd(char jobu, char jobvt, int m, int n,
              std::complex<float>* a, int lda,
              std::complex<float>* s, std::complex<float>* u, int ldu,
              std::complex<float>* vt, int ldvt);
    int gesvd(char jobu, char jobvt, int m, int n,
              std::complex<double>* a, int lda,
              std::complex<double>* s, std::complex<double>* u, int ldu,
              std::complex<double>* vt, int ldvt);

    int syevx(char jobz, char range, char uplo, int n, float* a, int lda,
              float vl, float vu, int il, int iu, float abstol, int& m,
              float* w, float* z, int ldz);
    int syevx(char jobz, char range, char uplo, int n, double* a, int lda,
              double vl, double vu, int il, int iu, double abstol, int& m,
              double* w, double* z, int ldz);

    int syev(char jobz, char uplo, int n, float* a, int lda, float* w);
    int syev(char jobz, char uplo, int n, double* a, int lda, double* w);
    int syev(char jobz, char uplo, int n, std::complex<float>* a, int lda,
             std::complex<float>* w);
    int syev(char jobz, char uplo, int n, std::complex<double>* a, int lda,
             std::complex<double>* w);

    inline long long sytrf_flops(long long n) {
      return n * n * n / 3;
    }
    int sytrf(char s, int n, float* a, int lda, int* ipiv,
              float* work, int lwork);
    int sytrf(char s, int n, double* a, int lda, int* ipiv,
              double* work, int lwork);
    int sytrf(char s, int n, std::complex<float>* a, int lda, int* ipiv,
              std::complex<float>* work, int lwork);
    int sytrf(char s, int n, std::complex<double>* a, int lda, int* ipiv,
              std::complex<double>* work, int lwork);
    template<typename scalar> inline int sytrf
    (char s, int n, scalar* a, int lda, int* ipiv) {
      scalar lwork;
      sytrf(s, n, a, lda, ipiv, &lwork, -1);
      int ilwork = int(std::real(lwork));
      std::unique_ptr<scalar[]> work(new scalar[ilwork]);
      return sytrf(s, n, a, lda, ipiv, work.get(), ilwork);
    }

    // inline long long sytrf_rook_flops(long long n) {
    //   return n * n * n / 3;
    // }
    // int sytrf_rook(char s, int n, float* a, int lda, int* ipiv,
    //                float* work, int lwork);
    // int sytrf_rook(char s, int n, double* a, int lda, int* ipiv,
    //                double* work, int lwork);
    // int sytrf_rook(char s, int n, std::complex<float>* a, int lda,
    //                int* ipiv, std::complex<float>* work, int lwork);
    // int sytrf_rook(char s, int n, std::complex<double>* a, int lda,
    //                int* ipiv, std::complex<double>* work, int lwork);
    // template<typename scalar> inline int sytrf_rook
    // (char s, int n, scalar* a, int lda, int* ipiv) {
    //   scalar lwork;
    //   sytrf_rook(s, n, a, lda, ipiv, &lwork, -1);
    //   int ilwork = int(std::real(lwork));
    //   std::unique_ptr<scalar[]> work(new scalar[ilwork]);
    //   return sytrf_rook(s, n, a, lda, ipiv, work.get(), ilwork);
    // }


    inline long long sytrs_flops(long long m, long long n, long long k) {
      return 2*m*n*k;
    }
    int sytrs(char s, int n, int nrhs, const float* a, int lda,
              const int* ipiv, float* b, int ldb);
    int sytrs(char s, int n, int nrhs, const double* a, int lda,
              const int* ipiv, double* b, int ldb);
    int sytrs(char s, int n, int nrhs, const std::complex<float>* a, int lda,
              const int* ipiv, std::complex<float>* b, int ldb);
    int sytrs(char s, int n, int nrhs, const std::complex<double>* a, int lda,
              const int* ipiv, std::complex<double>* b, int ldb);

    inline long long sytrs_rook_flops(long long m, long long n, long long k) {
      return 2*m*n*k;
    }
    int sytrs_rook(char s, int n, int nrhs, const float* a, int lda,
                   const int* ipiv, float* b, int ldb);
    int sytrs_rook(char s, int n, int nrhs, const double* a, int lda,
                   const int* ipiv, double* b, int ldb);
    int sytrs_rook(char s, int n, int nrhs,
                   const std::complex<float>* a, int lda,
                   const int* ipiv, std::complex<float>* b, int ldb);
    int sytrs_rook(char s, int n, int nrhs,
                   const std::complex<double>* a, int lda,
                   const int* ipiv, std::complex<double>* b, int ldb);

  } //end namespace blas
} // end namespace strumpack

#endif // BLASLAPACKWRAPPER_H
