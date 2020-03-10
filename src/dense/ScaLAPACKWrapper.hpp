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
 * Developers: Francois-Henry Rouet, Xiaoye S. Li, Pieter Ghysels
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */

#ifndef SCALAPACK_HPP
#define SCALAPACK_HPP

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include "BLASLAPACKWrapper.hpp"


namespace strumpack {
  namespace scalapack {

    extern "C" {
      ///////////////////////////////////////////////
      ////// BLACS //////////////////////////////////
      ///////////////////////////////////////////////
      void Cblacs_get(int, int, int *);
      void Cblacs_gridinit(int *, const char *, int, int);
      void Cblacs_gridmap(int *, int *, int, int, int);
      void Cblacs_gridinfo(int, int *, int *, int *, int *);
      void Cblacs_gridexit(int);
      void Cblacs_exit(int);
      int Csys2blacs_handle(MPI_Comm);
      MPI_Comm Cblacs2sys_handle(int);
    }

    int descinit(int* desc, int m, int n, int mb, int nb,
                 int rsrc, int csrc, int ictxt, int mxllda);

    void descset(int* desc, int m, int n, int mb, int nb,
                 int rsrc, int csrc, int ictxt, int mxllda);

    int numroc(int n, int nb, int iproc, int isrcproc, int nprocs);

    int infog1l(int GINDX, int NB, int NPROCS, int MYROC,
                int ISRCPROC, int& ROCSRC);

    int infog1l(int GINDX, int NB, int NPROCS, int MYROC, int ISRCPROC);

    void infog2l(int GRINDX, int GCINDX, const int* DESC,
                 int NPROW, int NPCOL, int MYROW,
                 int MYCOL, int& LRINDX, int& LCINDX, int& RSRC, int& CSRC);
    void infog2l(int GRINDX, int GCINDX, const int* DESC,
                 int NPROW, int NPCOL, int MYROW,
                 int MYCOL, int& LRINDX, int& LCINDX);

    void gebs2d(int ctxt, char scope, char top, int m, int n,
                int* a, int lda);
    void gebs2d(int ctxt, char scope, char top, int m, int n,
                float* a, int lda);
    void gebs2d(int ctxt, char scope, char top, int m, int n,
                double* a, int lda);
    void gebs2d(int ctxt, char scope, char top, int m, int n,
                std::complex<float>* a, int lda);
    void gebs2d(int ctxt, char scope, char top, int m, int n,
                std::complex<double>* a, int lda);


    void gebr2d(int ctxt, char scope, char top, int m, int n,
                int* a, int lda, int rsrc, int csrc);
    void gebr2d(int ctxt, char scope, char top, int m, int n,
                float* a, int lda, int rsrc, int csrc);
    void gebr2d(int ctxt, char scope, char top, int m, int n,
                double* a, int lda, int rsrc, int csrc);
    void gebr2d(int ctxt, char scope, char top, int m, int n,
                std::complex<float>* a, int lda, int rsrc, int csrc);
    void gebr2d(int ctxt, char scope, char top, int m, int n,
                std::complex<double>* a, int lda, int rsrc, int csrc);

    void gsum2d(int ctxt, char scope, char top, int m, int n,
                float* a, int lda, int rdest, int cdest);
    void gsum2d(int ctxt, char scope, char top, int m, int n,
                double* a, int lda, int rdest, int cdest);

    void gamx2d(int ctxt, char scope, char top, int m, int n,
                float* a, int lda, int *ra, int *ca, int ldia,
                int rdest, int cdest);
    void gamx2d(int ctxt, char scope, char top, int m, int n,
                double* a, int lda, int *ra, int *ca, int ldia,
                int rdest, int cdest);
    void gamx2d(int ctxt, char scope, char top, int m, int n,
                std::complex<float>* a, int lda, int *ra, int *ca,
                int ldia, int rdest, int cdest);
    void gamx2d(int ctxt, char scope, char top, int m, int n,
                std::complex<double>* a, int lda, int *ra, int *ca,
                int ldia, int rdest, int cdest);

    void gamn2d(int ctxt, char scope, char top, int m, int n,
                float* a, int lda, int *ra, int *ca, int ldia,
                int rdest, int cdest);
    void gamn2d(int ctxt, char scope, char top, int m, int n,
                double* a, int lda, int *ra, int *ca, int ldia,
                int rdest, int cdest);
    void gamn2d(int ctxt, char scope, char top, int m, int n,
                std::complex<float>* a, int lda, int *ra, int *ca,
                int ldia, int rdest, int cdest);
    void gamn2d(int ctxt, char scope, char top, int m, int n,
                std::complex<double>* a, int lda, int *ra, int *ca,
                int ldia, int rdest, int cdest);

    void pamax(int n, float *amax, int *indx, float* x, int ix,
               int jx, int *descx, int incx);
    void pamax(int n, double *amax, int *indx, double* x,
               int ix, int jx, int *descx, int incx);
    void pamax(int n, std::complex<float> *amax, int *indx,
               std::complex<float>* x, int ix, int jx, int *descx, int incx);
    void pamax(int n, std::complex<double> *amax, int *indx,
               std::complex<double>* x, int ix, int jx, int *descx, int incx);

    void pswap(int n, float* x, int ix, int jx, int *descx, int incx,
               float* y, int iy, int jy, int *descy, int incy);
    void pswap(int n, double* x, int ix, int jx, int *descx, int incx,
               double* y, int iy, int jy, int *descy, int incy);
    void pswap(int n, std::complex<float>* x, int ix, int jx,
               int *descx, int incx, std::complex<float>* y,
               int iy, int jy, int *descy, int incy);
    void pswap(int n, std::complex<double>* x, int ix, int jx,
               int *descx, int incx, std::complex<double>* y,
               int iy, int jy, int *descy, int incy);

    void pscal(int n, float a, float* x, int ix, int jx,
               int *descx, int incx);
    void pscal(int n, double a, double* x, int ix, int jx,
               int *descx, int incx);
    void pscal(int n, std::complex<float> a, std::complex<float>* x,
               int ix, int jx, int *descx, int incx);
    void pscal(int n, std::complex<double> a, std::complex<double>* x,
               int ix, int jx, int *descx, int incx);

    void pgemv(char ta, int m, int n, float alpha,
               const float* a, int ia, int ja, const int *desca,
               const float* x, int ix, int jx, const int *descx,
               int incx, float beta, float* y, int iy, int jy,
               const int *descy, int incy);
    void pgemv(char ta, int m, int n, double alpha,
               const double* a, int ia, int ja, const int *desca,
               const double* x, int ix, int jx, const int *descx,
               int incx, double beta, double* y, int iy, int jy,
               const int *descy, int incy);
    void pgemv(char ta, int m, int n, std::complex<float> alpha,
               const std::complex<float>* a, int ia, int ja, const int *desca,
               const std::complex<float>* x, int ix, int jx, const int *descx,
               int incx, std::complex<float> beta, std::complex<float>* y,
               int iy, int jy, const int *descy, int incy);
    void pgemv(char ta, int m, int n, std::complex<double> alpha,
               const std::complex<double>* a,
               int ia, int ja, const int *desca,
               const std::complex<double>* x,
               int ix, int jx, const int *descx, int incx,
               std::complex<double> beta, std::complex<double>* y,
               int iy, int jy, const int *descy, int incy);

    void pgemm(char ta, char tb, int m, int n, int k, float alpha,
               const float* a, int ia, int ja, const int *desca,
               const float* b, int ib, int jb, const int *descb, float beta,
               float *c, int ic, int jc, const int *descC);
    void pgemm(char ta, char tb, int m, int n, int k, double alpha,
               const double* a, int ia, int ja, const int *desca,
               const double* b, int ib, int jb, const int *descb, double beta,
               double *c, int ic, int jc, const int *descC);
    void pgemm(char ta, char tb, int m, int n, int k,
               std::complex<float> alpha,
               const std::complex<float>* a, int ia, int ja, const int *desca,
               const std::complex<float>* b, int ib, int jb, const int *descb,
               std::complex<float> beta, std::complex<float> *c,
               int ic, int jc, const int *descC);
    void pgemm(char ta, char tb, int m, int n, int k,
               std::complex<double> alpha,
               const std::complex<double>* a,
               int ia, int ja, const int *desca,
               const std::complex<double>* b,
               int ib, int jb, const int *descb,
               std::complex<double> beta, std::complex<double> *c,
               int ic, int jc, const int *descC);

    void placgv(int n, float* x, int ix, int jx, const int *descx, int incx);
    void placgv(int n, double* x, int ix, int jx, const int *descx, int incx);
    void placgv(int n, std::complex<double>* x, int ix, int jx,
                const int *descx, int incx);
    void placgv(int n, std::complex<float>* x, int ix, int jx,
                const int *descx, int incx);

    void pgeru(int m, int n, float alpha,
               const float* x, int ix, int jx, const int *descx, int incx,
               const float* y, int iy, int jy, const int *descy, int incy,
               float* a, int ia, int ja, const int *desca);
    void pgeru(int m, int n, double alpha,
               const double* x, int ix, int jx, const int *descx, int incx,
               const double* y, int iy, int jy, const int *descy, int incy,
               double* a, int ia, int ja, const int *desca);
    void pgeru(int m, int n, std::complex<float> alpha,
               const std::complex<float>* x, int ix, int jx,
               const int *descx, int incx,
               const std::complex<float>* y, int iy, int jy,
               const int *descy, int incy,
               std::complex<float>* a, int ia, int ja, const int *desca);
    void pgeru(int m, int n, std::complex<double> alpha,
               const std::complex<double>* x, int ix, int jx,
               const int *descx, int incx,
               const std::complex<double>* y, int iy, int jy,
               const int *descy, int incy,
               std::complex<double>* a, int ia, int ja, const int *desca);

    void plaswp(char direc, char rowcol, int n,
                float* a, int ia, int ja, const int* desca,
                int k1, int k2, const int* ipiv);
    void plaswp(char direc, char rowcol, int n,
                double* a, int ia, int ja, const int* desca,
                int k1, int k2, const int* ipiv);
    void plaswp(char direc, char rowcol, int n,
                std::complex<float>* a, int ia, int ja, const int* desca,
                int k1, int k2, const int* ipiv);
    void plaswp(char direc, char rowcol, int n,
                std::complex<double>* a, int ia, int ja, const int* desca,
                int k1, int k2, const int* ipiv);

    void plapiv(char direc, char rowcol, char pivroc, int m, int n,
                float* a, int ia, int ja, const int* desca,
                const int* ipiv, int ip, int jp, const int* descip,
                int* iwork);
    void plapiv(char direc, char rowcol, char pivroc, int m, int n,
                double* a, int ia, int ja, const int* desca,
                const int* ipiv, int ip, int jp, const int* descip,
                int* iwork);
    void plapiv(char direc, char rowcol, char pivroc, int m, int n,
                std::complex<float>* a, int ia, int ja, const int* desca,
                const int* ipiv, int ip, int jp, const int* descip,
                int* iwork);
    void plapiv(char direc, char rowcol, char pivroc, int m, int n,
                std::complex<double>* a, int ia, int ja, int* desca,
                const int* ipiv, int ip, int jp, const int* descip,
                int* iwork);

    void ptrsm(char side, char uplo, char trans, char diag, int m, int n,
               float alpha, const float* a, int ia, int ja, const int *desca,
               float* b, int ib, int jb, const int *descb);
    void ptrsm(char side, char uplo, char trans, char diag, int m, int n,
               double alpha, const double* a, int ia, int ja,
               const int *desca, double* b, int ib, int jb, const int *descb);
    void ptrsm(char side, char uplo, char trans, char diag, int m, int n,
               std::complex<float> alpha, const std::complex<float>* a,
               int ia, int ja, const int *desca,
               std::complex<float>* b, int ib, int jb, const int *descb);
    void ptrsm(char side, char uplo, char trans, char diag, int m, int n,
               std::complex<double> alpha, const std::complex<double>* a,
               int ia, int ja, const int *desca,
               std::complex<double>* b, int ib, int jb, const int *descb);

    void ptrsv(char uplo, char trans, char diag, int m,
               const float* a, int ia, int ja, const int *desca,
               float* b, int ib, int jb, const int *descb, int incb);
    void ptrsv(char uplo, char trans, char diag, int m,
               const double* a, int ia, int ja, const int *desca,
               double* b, int ib, int jb, int const *descb, int incb);
    void ptrsv(char uplo, char trans, char diag, int m,
               const std::complex<float>* a, int ia, int ja, const int *desca,
               std::complex<float>* b, int ib, int jb,
               const int *descb, int incb);
    void ptrsv(char uplo, char trans, char diag, int m,
               const std::complex<double>* a, int ia, int ja,
               const int *desca, std::complex<double>* b,
               int ib, int jb, const int *descb, int incb);

    float plange(char norm, int m, int n, const float* a, int ia, int ja,
                 const int *desca, float *work);
    double plange(char norm, int m, int n, const double* a, int ia, int ja,
                  const int *desca, double *work);
    float plange(char norm, int m, int n, const std::complex<float>* a,
                 int ia, int ja, const int *desca, float *work);
    double plange(char norm, int m, int n, const std::complex<double>* a,
                  int ia, int ja, const int *desca, double *work);

    void pgeadd(char trans, int m, int n, float alpha,
                const float* a, int ia, int ja, int *desca, float beta,
                float *c, int ic, int jc, const int *descc);
    void pgeadd(char trans, int m, int n, double alpha,
                const double* a, int ia, int ja, int *desca, double beta,
                double *c, int ic, int jc, const int *descC);
    void pgeadd(char trans, int m, int n, std::complex<float> alpha,
                const std::complex<float>* a, int ia, int ja, int *desca,
                std::complex<float> beta,
                std::complex<float> *c, int ic, int jc, const int *descC);
    void pgeadd(char trans, int m, int n, std::complex<double> alpha,
                const std::complex<double>* a, int ia, int ja, int *desca,
                std::complex<double> beta,
                std::complex<double> *c, int ic, int jc, const int *descC);

    void placpy(char trans, int m, int n,
                const float* a, int ia, int ja, const int *desca,
                float *c, int ic, int jc, const int *descc);
    void placpy(char trans, int m, int n,
                const double* a, int ia, int ja, const int *desca,
                double *c, int ic, int jc, const int *descc);
    void placpy(char trans, int m, int n, const std::complex<float>* a,
                int ia, int ja, const int *desca,
                std::complex<float> *c, int ic, int jc, const int *descc);
    void placpy(char trans, int m, int n, const std::complex<double>* a,
                int ia, int ja, const int *desca,
                std::complex<double> *c, int ic, int jc, const int *descc);

    void pgemr2d(int m, int n, const float* a, int ia, int ja,
                 const int *desca, float* b, int ib, int jb,
                 const int *descb, int ctxt);
    void pgemr2d(int m, int n, const double* a, int ia, int ja,
                 const int *desca, double* b, int ib, int jb,
                 const int *descb, int ctxt);
    void pgemr2d(int m, int n, const std::complex<float>* a, int ia, int ja,
                 const int *desca, std::complex<float>* b, int ib, int jb,
                 const int *descb, int ctxt);
    void pgemr2d(int m, int n, const std::complex<double>* a, int ia, int ja,
                 const int *desca, std::complex<double>* b, int ib, int jb,
                 const int *descb, int ctxt);

    void ptranc(int m, int n, float alpha,
                const float* a, int ia, int ja, const int *desca,
                float beta, float *c, int ic, int jc, const int *descc);
    void ptranc(int m, int n, double alpha,
                const double* a, int ia, int ja, const int *desca,
                double beta, double *c, int ic, int jc, const int *descc);
    void ptranc(int m, int n, std::complex<float> alpha,
                const std::complex<float>* a, int ia, int ja,
                const int *desca, std::complex<float> beta,
                std::complex<float> *c, int ic, int jc, const int *descc);
    void ptranc(int m, int n, std::complex<double> alpha,
                const std::complex<double>* a, int ia, int ja,
                const int *desca, std::complex<double> beta,
                std::complex<double> *c, int ic, int jc, const int *descc);

    void pgeqpftol(int m, int n, float* a, int ia, int ja, const int *desca,
                   int *J, int *piv, int *r, float rtol, float atol);
    void pgeqpftol(int m, int n, double* a, int ia, int ja, const int *desca,
                   int *J, int *piv, int *r, double rtol, double atol);
    void pgeqpftol(int m, int n, std::complex<float>* a, int ia, int ja,
                   const int *desca, int *J, int *piv, int *r,
                   float rtol, float atol);
    void pgeqpftol(int m, int n, std::complex<double>* a, int ia, int ja,
                   const int *desca, int *J, int *piv, int *r,
                   double rtol, double atol);

    int pgetrf(int m, int n, float* a, int ia, int ja,
               const int *desca, int *ipiv);
    int pgetrf(int m, int n, double* a, int ia, int ja,
               const int *desca, int *ipiv);
    int pgetrf(int m, int n, std::complex<float>* a, int ia, int ja,
               const int *desca, int *ipiv);
    int pgetrf(int m, int n, std::complex<double>* a, int ia, int ja,
               const int *desca, int *ipiv);

    int pgetrs(char trans, int m, int n, const float* a, int ia, int ja,
               const int *desca, const int *ipiv,
               float* b, int ib, int jb, const int *descb);
    int pgetrs(char trans, int m, int n, const double* a, int ia, int ja,
               const int *desca, const int *ipiv,
               double* b, int ib, int jb, const int *descb);
    int pgetrs(char trans, int m, int n, const std::complex<float>* a,
               int ia, int ja, const int *desca, const int *ipiv,
               std::complex<float>* b, int ib, int jb, const int *descb);
    int pgetrs(char trans, int m, int n, const std::complex<double>* a,
               int ia, int ja, const int *desca, const int *ipiv,
               std::complex<double>* b, int ib, int jb, const int *descb);

    void pgelqf(int m, int n, float* a, int ia, int ja,
                const int *desca, float *tau);
    void pgelqf(int m, int n, double* a, int ia, int ja,
                const int *desca, double *tau);
    void pgelqf(int m, int n, std::complex<float>* a, int ia, int ja,
                const int *desca, std::complex<float> *tau);
    void pgelqf(int m, int n, std::complex<double>* a, int ia, int ja,
                const int *desca, std::complex<double> *tau);

    void pxxglq(int m, int n, int k, float* a, int ia, int ja,
                const int *desca, const float *tau);
    void pxxglq(int m, int n, int k, double* a, int ia, int ja,
                const int *desca, const double *tau);
    void pxxglq(int m, int n, int k, std::complex<float>* a, int ia, int ja,
                const int *desca, const std::complex<float> *tau);
    void pxxglq(int m, int n, int k, std::complex<double>* a, int ia, int ja,
                const int *desca, const std::complex<double> *tau);

    int pgeqrf(int m, int n, float* a, int ia, int ja, const int *desca,
               float *tau, float* work, int lwork);
    int pgeqrf(int m, int n, double* a, int ia, int ja, const int *desca,
               double *tau, double* work, int lwork);
    int pgeqrf(int m, int n, std::complex<float>* a, int ia, int ja,
               const int *desca, std::complex<float> *tau,
               std::complex<float>* work, int lwork);
    int pgeqrf(int m, int n, std::complex<double>* a, int ia, int ja,
               const int *desca, std::complex<double> *tau,
               std::complex<double>* work, int lwork);
    template<typename T>
    int pgeqrf(int m, int n, T* a, int ia, int ja, const int *desca, T* tau) {
      T lwork;
      pgeqrf(m, n, a, ia, ja, desca, tau, &lwork, -1);
      int ilwork = int(std::real(lwork));
      std::unique_ptr<T[]> work(new T[ilwork]);
      return pgeqrf(m, n, a, ia, ja, desca, tau, work.get(), ilwork);
    }

    int pxxgqr(int m, int n, int k, float* a, int ia, int ja,
               const int* desca, const float* tau, float* work, int lwork);
    int pxxgqr(int m, int n, int k, double* a, int ia, int ja,
               const int* desca, const double* tau, double* work, int lwork);
    int pxxgqr(int m, int n, int k, std::complex<float>* a, int ia, int ja,
               const int* desca, const std::complex<float>* tau,
               std::complex<float>* work, int lwork);
    int pxxgqr(int m, int n, int k, std::complex<double>* a, int ia, int ja,
               const int* desca, const std::complex<double>* tau,
               std::complex<double>* work, int lwork);
    template<typename T>
    int pxxgqr(int m, int n, int k, T* a, int ia, int ja,
               int* desca, T* tau) {
      T lwork;
      int info = pxxgqr(m, n, k, a, ia, ja, desca, tau, &lwork, -1);
      int ilwork = int(std::real(lwork));
      std::unique_ptr<T[]> work(new T[ilwork]);
      info = pxxgqr(m, n, k, a, ia, ja, desca, tau, work.get(), ilwork);
      return info;
    }

    char topget(int ctxt, char B, char R);

  } // end namespace scalapack
} // end namespace strumpack

#endif // SCALAPACK_HPP
