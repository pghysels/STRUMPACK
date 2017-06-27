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
 * Developers: Francois-Henry Rouet, Xiaoye S. Li, Pieter Ghysels
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 *
 */

#ifndef SCALAPACK_HPP
#define SCALAPACK_HPP

#include "blas_lapack_wrapper.hpp"
#include "blacs.h"

namespace strumpack {
  namespace scalapack {

    extern "C" {
      /* Arithmetic-independent routines */
      int numroc_(int *, int *, int *, int *, int *);
      void descinit_(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *);
      void descset_(int *, int *, int *, int *, int *, int *, int *, int *, int *);
      // int indxg2p_(int *, int *, int *, int *, int *);
      // int indxg2l_(int *, int *, int *, int *, int *);
      // int indxl2g_(int *, int *, int *, int *, int *);
      void infog1l_(int *, int *, int *, int *, int *, int *, int *);
      void infog2l_(int *, int *, const int *, int *, int *, int *, int *, int *, int *, int *, int *);
      void igamn2d_(int *, const char *, const char *, int *, int *, int *, int *, int *, int *, int *, int *, int *);

      /* xGEBS2D */
      void igebs2d_(int *, const char *, const char *, int *, int *,      int *, int *);
      void dgebs2d_(int *, const char *, const char *, int *, int *,   double *, int *);
      void sgebs2d_(int *, const char *, const char *, int *, int *,    float *, int *);
      void zgebs2d_(int *, const char *, const char *, int *, int *, c_double *, int *);
      void cgebs2d_(int *, const char *, const char *, int *, int *, c_float *, int *);

      /* xGEBR2D */
      void igebr2d_(int *, const char *, const char *, int *, int *,      int *, int *, int *, int *);
      void dgebr2d_(int *, const char *, const char *, int *, int *,   double *, int *, int *, int *);
      void sgebr2d_(int *, const char *, const char *, int *, int *,    float *, int *, int *, int *);
      void zgebr2d_(int *, const char *, const char *, int *, int *, c_double *, int *, int *, int *);
      void cgebr2d_(int *, const char *, const char *, int *, int *, c_float *, int *, int *, int *);

      /* xGSUM2D */
      void dgsum2d_(int *, char *, char *, int *, int *, double *, int *, int *, int *);
      void sgsum2d_(int *, char *, char *, int *, int *,  float *, int *, int *, int *);

      /* xGAMX2D */
      void dgamx2d_(int *, char *, char *, int *, int *, double *, int *, int *, int *, int *, int *, int *);
      void sgamx2d_(int *, char *, char *, int *, int *,  float *, int *, int *, int *, int *, int *, int *);

      /* PxAMAX */
      void pdamax_(int *,   double *, int *,   double *, int *, int *, int *, int *);
      void psamax_(int *,    float *, int *,    float *, int *, int *, int *, int *);
      void pzamax_(int *, c_double *, int *, c_double *, int *, int *, int *, int *);
      void pcamax_(int *, c_float *, int *, c_float *, int *, int *, int *, int *);

      /* PxSWAP */
      void pdswap_ (int *,   double *, int *, int *, int *, int *,   double *, int *, int *, int *, int *);
      void psswap_ (int *,    float *, int *, int *, int *, int *,    float *, int *, int *, int *, int *);
      void pzswap_ (int *, c_double *, int *, int *, int *, int *, c_double *, int *, int *, int *, int *);
      void pcswap_ (int *, c_float *, int *, int *, int *, int *, c_float *, int *, int *, int *, int *);

      /* PxSCAL */
      void pdscal_(int *,   double *,   double *, int *, int *, int *, int *);
      void psscal_(int *,    float *,    float *, int *, int *, int *, int *);
      void pzscal_(int *, c_double *, c_double *, int *, int *, int *, int *);
      void pcscal_(int *, c_float *, c_float *, int *, int *, int *, int *);

      /* PxGEMV */
      void pdgemv_(char *, int *, int *,   double *,   double *, int *, int *, int *,   double *, int *, int *, int *, int *,   double *,   double *, int *, int *, int *, int *);
      void psgemv_(char *, int *, int *,    float *,    float *, int *, int *, int *,    float *, int *, int *, int *, int *,    float *,    float *, int *, int *, int *, int *);
      void pzgemv_(char *, int *, int *, c_double *, c_double *, int *, int *, int *, c_double *, int *, int *, int *, int *, c_double *, c_double *, int *, int *, int *, int *);
      void pcgemv_(char *, int *, int *, c_float *, c_float *, int *, int *, int *, c_float *, int *, int *, int *, int *, c_float *, c_float *, int *, int *, int *, int *);

      /* PxGEMM */
      void pdgemm_(char *, char *, int *, int *, int *,   double *,   double *, int *, int *, int *,   double *, int *, int *, int *,   double *,   double *, int *, int *, int *);
      void psgemm_(char *, char *, int *, int *, int *,    float *,    float *, int *, int *, int *,    float *, int *, int *, int *,    float *,    float *, int *, int *, int *);
      void pzgemm_(char *, char *, int *, int *, int *, c_double *, c_double *, int *, int *, int *, c_double *, int *, int *, int *, c_double *, c_double *, int *, int *, int *);
      void pcgemm_(char *, char *, int *, int *, int *, c_float *, c_float *, int *, int *, int *, c_float *, int *, int *, int *, c_float *, c_float *, int *, int *, int *);

      /* PxLACGV */
      void pzlacgv_(int*, c_double *X, int *, int *, int *, int *);
      void pclacgv_(int*, c_float *X, int *, int *, int *, int *);

      /* PxGER */
      void pdger_ (int *, int *,   double *,   double *, int *, int *, int *, int *,     double *, int *, int *, int *, int *, double *, int *, int *, int *);
      void psger_ (int *, int *,    float *,    float *, int *, int *, int *, int *,    float *, int *, int *, int *, int *,    float *, int *, int *, int *);
      void pzgeru_(int *, int *, c_double *, c_double *, int *, int *, int *, int *, c_double *, int *, int *, int *, int *, c_double *, int *, int *, int *);
      void pcgeru_(int *, int *, c_float *, c_float *, int *, int *, int *, int *, c_float *, int *, int *, int *, int *, c_float *, int *, int *, int *);

      /* PxLASWP */
      void pdlaswp_(char *, char *, int *,   double *, int *, int *, int *, int *, int *, int *);
      void pslaswp_(char *, char *, int *,    float *, int *, int *, int *, int *, int *, int *);
      void pzlaswp_(char *, char *, int *, c_double *, int *, int *, int *, int *, int *, int *);
      void pclaswp_(char *, char *, int *, c_float *, int *, int *, int *, int *, int *, int *);

      /* PxLAPIV */
      void pdlapiv_(char *, char *, char *, int *, int *,   double *, int *, int *, int *, int *, int *, int *, int *, int *);
      void pslapiv_(char *, char *, char *, int *, int *,    float *, int *, int *, int *, int *, int *, int *, int *, int *);
      void pzlapiv_(char *, char *, char *, int *, int *, c_double *, int *, int *, int *, int *, int *, int *, int *, int *);
      void pclapiv_(char *, char *, char *, int *, int *, c_float *, int *, int *, int *, int *, int *, int *, int *, int *);

      /* PxTRSM */
      void pdtrsm_(char *, char *, char *, char *, int *, int *,   double *,   double *, int *, int *, int *,   double *, int *, int *, int *);
      void pstrsm_(char *, char *, char *, char *, int *, int *,    float *,    float *, int *, int *, int *,    float *, int *, int *, int *);
      void pztrsm_(char *, char *, char *, char *, int *, int *, c_double *, c_double *, int *, int *, int *, c_double *, int *, int *, int *);
      void pctrsm_(char *, char *, char *, char *, int *, int *, c_float *, c_float *, int *, int *, int *, c_float *, int *, int *, int *);

      /* PxTRSV */
      void pdtrsv_(char *, char *, char *, int *,   double *, int *, int *, int *,   double *, int *, int *, int *, int *);
      void pstrsv_(char *, char *, char *, int *,    float *, int *, int *, int *,    float *, int *, int *, int *, int *);
      void pztrsv_(char *, char *, char *, int *, c_double *, int *, int *, int *, c_double *, int *, int *, int *, int *);
      void pctrsv_(char *, char *, char *, int *, c_float *, int *, int *, int *, c_float *, int *, int *, int *, int *);

      /* PxLANGE */
      double pdlange_(char*, int*, int*,   double*, int*, int*, int*, double*);
      float pslange_(char*, int*, int*,    float*, int*, int*, int*,  float*);
      double pzlange_(char*, int*, int*, c_double*, int*, int*, int*, double*);
      float pclange_(char*, int*, int*, c_float*, int*, int*, int*,  float*);

      /* PxGEADD */
      void pdgeadd_(char *, int *, int *,   double *,   double *, int *, int *, int *,   double *,   double *, int *, int *, int *);
      void psgeadd_(char *, int *, int *,    float *,    float *, int *, int *, int *,    float *,    float *, int *, int *, int *);
      void pzgeadd_(char *, int *, int *, c_double *, c_double *, int *, int *, int *, c_double *, c_double *, int *, int *, int *);
      void pcgeadd_(char *, int *, int *, c_float *, c_float *, int *, int *, int *, c_float *, c_float *, int *, int *, int *);

      /* PxLACPY */
      void pdlacpy_(char *, int *, int *,   double *, int *, int *, int *,   double *, int *, int *, int *);
      void pslacpy_(char *, int *, int *,    float *, int *, int *, int *,    float *, int *, int *, int *);
      void pzlacpy_(char *, int *, int *, c_double *, int *, int *, int *, c_double *, int *, int *, int *);
      void pclacpy_(char *, int *, int *, c_float *, int *, int *, int *, c_float *, int *, int *, int *);

      /* pxGEMR2D */
      void pdgemr2d_(int *, int *,   double *, int *, int *, int *,   double *, int *, int *, int *, int *);
      void psgemr2d_(int *, int *,    float *, int *, int *, int *,    float *, int *, int *, int *, int *);
      void pzgemr2d_(int *, int *, c_double *, int *, int *, int *, c_double *, int *, int *, int *, int *);
      void pcgemr2d_(int *, int *, c_float *, int *, int *, int *, c_float *, int *, int *, int *, int *);

      /* PxTRAN */
      void pdtran_(int *, int *,   double *,   double *, int *, int *, int *,   double *,   double *, int *, int *, int *);
      void pstran_(int *, int *,    float *,    float *, int *, int *, int *,    float *,    float *, int *, int *, int *);
      void pztranc_(int *, int *, c_double *, c_double *, int *, int *, int *, c_double *, c_double *, int *, int *, int *);
      void pctranc_(int *, int *, c_float *, c_float *, int *, int *, int *, c_float *, c_float *, int *, int *, int *);

      /* PxGEQPF; this one is used only in the examples */
      void pdgeqpf_(int *, int *,   double *, int *, int *, int *, int *,   double *,   double *, int *, int *);
      void psgeqpf_(int *, int *,    float *, int *, int *, int *, int *,    float *,    float *, int *, int *);
      void pzgeqpf_(int *, int *, c_double *, int *, int *, int *, int *, c_double *, c_double *, int *, double *, int *, int *);
      void pcgeqpf_(int *, int *, c_float *, int *, int *, int *, int *, c_float *, c_float *, int *,  float *, int *, int *);

      /* PxGEQPFmod */
      void pdgeqpfmod_(int *, int *,   double *, int *, int *, int *, int *,   double *,   double *, int *, int *, int *, int *, int *, double *);
      void psgeqpfmod_(int *, int *,    float *, int *, int *, int *, int *,    float *,    float *, int *, int *, int *, int *, int *,  float *);
      void pzgeqpfmod_(int *, int *, c_double *, int *, int *, int *, int *, c_double *, c_double *, int *, double *, int *, int *, int *, int *, int *, double *);
      void pcgeqpfmod_(int *, int *, c_float *, int *, int *, int *, int *, c_float *, c_float *, int *,  float *, int *, int *, int *, int *, int *,  float *);

      /* PxGETRF */
      void pdgetrf_(int *, int *,   double *, int *, int *, int *, int *, int *);
      void psgetrf_(int *, int *,    float *, int *, int *, int *, int *, int *);
      void pzgetrf_(int *, int *, c_double *, int *, int *, int *, int *, int *);
      void pcgetrf_(int *, int *, c_float *, int *, int *, int *, int *, int *);

      /* PxGETRS */
      void pdgetrs_(char *, int *, int *,   double *, int *, int *, int *, int *,   double *, int *, int *, int *, int *);
      void psgetrs_(char *, int *, int *,    float *, int *, int *, int *, int *,    float *, int *, int *, int *, int *);
      void pzgetrs_(char *, int *, int *, c_double *, int *, int *, int *, int *, c_double *, int *, int *, int *, int *);
      void pcgetrs_(char *, int *, int *, c_float *, int *, int *, int *, int *, c_float *, int *, int *, int *, int *);

      /* PxGELQF */
      void pdgelqf_(int *, int *,   double *, int *, int *, int *,   double *,   double *, int *, int *);
      void psgelqf_(int *, int *,    float *, int *, int *, int *,    float *,    float *, int *, int *);
      void pzgelqf_(int *, int *, c_double *, int *, int *, int *, c_double *, c_double *, int *, int *);
      void pcgelqf_(int *, int *, c_float *, int *, int *, int *, c_float *, c_float *, int *, int *);

      /* PxxxGLQ */
      void pdorglq_(int *, int *, int *,   double *, int *, int *, int *,   double *,   double *, int *, int *);
      void psorglq_(int *, int *, int *,    float *, int *, int *, int *,    float *,    float *, int *, int *);
      void pzunglq_(int *, int *, int *, c_double *, int *, int *, int *, c_double *, c_double *, int *, int *);
      void pcunglq_(int *, int *, int *, c_float *, int *, int *, int *, c_float *, c_float *, int *, int *);

      /* PxNRM2; this one is used only in the examples */
      void pdnrm2_(int *,  double *,   double *, int *, int *, int *, int *);
      void psnrm2_(int *,   float *,    float *, int *, int *, int *, int *);
      void pdznrm2_(int *, double *, c_double *, int *, int *, int *, int *);
      void pscnrm2_(int *,  float *, c_float *, int *, int *, int *, int *);

      /* PxDOT; this is used only in the examples */
      void pddot_(int *,   double *,   double *, int *, int *, int *, int *,   double *, int *, int *, int *, int *);
      void psdot_(int *,    float *,    float *, int *, int *, int *, int *,    float *, int *, int *, int *, int *);
      void pzdot_(int *, c_double *, c_double *, int *, int *, int *, int *, c_double *, int *, int *, int *, int *);
      void pcdot_(int *, c_float *, c_float *, int *, int *, int *, int *, c_float *, int *, int *, int *, int *);

      void pdgeqrf_(int *, int *,   double *, int *, int *, int *,   double *,   double *, int *, int *);
      void psgeqrf_(int *, int *,    float *, int *, int *, int *,    float *,    float *, int *, int *);
      void pzgeqrf_(int *, int *, c_double *, int *, int *, int *, c_double *, c_double *, int *, int *);
      void pcgeqrf_(int *, int *,  c_float *, int *, int *, int *,  c_float *,  c_float *, int *, int *);

      void pdorgqr_(int *, int *, int *,   double *, int *, int *, int *,   double *,   double *, int *, int *);
      void psorgqr_(int *, int *, int *,    float *, int *, int *, int *,    float *,    float *, int *, int *);
      void pzungqr_(int *, int *, int *, c_double *, int *, int *, int *, c_double *, c_double *, int *, int *);
      void pcungqr_(int *, int *, int *,  c_float *, int *, int *, int *,  c_float *,  c_float *, int *, int *);
    }

    /* ScaLAPACK routines */
    /* Arithmetic-independent routines */
    inline int descinit(int* desc, int m, int n, int mb, int nb, int rsrc, int csrc, int ictxt, int mxllda) {
      int info;
      descinit_(desc, &m, &n, &mb, &nb, &rsrc, &csrc, &ictxt, &mxllda, &info);
      return info;
    }
    inline void descset(int* desc, int m, int n, int mb, int nb, int rsrc, int csrc, int ictxt, int mxllda) {
      descset_(desc, &m, &n, &mb, &nb, &rsrc, &csrc, &ictxt, &mxllda);
    }

    inline int numroc(int n, int nb, int iproc, int isrcproc, int nprocs) {
      return numroc_(&n, &nb, &iproc, &isrcproc, &nprocs);
    }

    inline int infog1l(int GINDX, int NB, int NPROCS, int MYROC, int ISRCPROC, int& ROCSRC) {
      int LINDX;
      infog1l_(&GINDX, &NB, &NPROCS, &MYROC, &ISRCPROC, &LINDX, &ROCSRC);
      return LINDX;
    }
    inline int infog1l(int GINDX, int NB, int NPROCS, int MYROC, int ISRCPROC) {
      int LINDX, ROCSRC;
      infog1l_(&GINDX, &NB, &NPROCS, &MYROC, &ISRCPROC, &LINDX, &ROCSRC);
      return LINDX;
    }
    inline void infog2l(int GRINDX, int GCINDX, const int* DESC, int NPROW, int NPCOL, int MYROW,
			int MYCOL, int& LRINDX, int& LCINDX, int& RSRC, int& CSRC) {
      infog2l_(&GRINDX, &GCINDX, DESC, &NPROW, &NPCOL, &MYROW, &MYCOL, &LRINDX, &LCINDX, &RSRC, &CSRC);
    }
    inline void infog2l(int GRINDX, int GCINDX, const int* DESC, int NPROW, int NPCOL, int MYROW,
			int MYCOL, int& LRINDX, int& LCINDX) {
      int RSRC, CSRC;
      infog2l_(&GRINDX, &GCINDX, DESC, &NPROW, &NPCOL, &MYROW, &MYCOL, &LRINDX, &LCINDX, &RSRC, &CSRC);
    }

    /* xGEBS2D */
    template<typename S> inline void gebs2d(int, char, char, int, int, S *, int);
    template<> inline void gebs2d<int>(int ctxt, char scope, char top, int m, int n, int *A, int lda) {
      igebs2d_(&ctxt,&scope,&top,&m,&n,A,&lda);
    }
    template<> inline void gebs2d<double>(int ctxt, char scope, char top, int m, int n, double *A, int lda) {
      dgebs2d_(&ctxt,&scope,&top,&m,&n,A,&lda);
    }
    template<> inline void gebs2d< float>(int ctxt, char scope, char top, int m, int n, float *A, int lda) {
      sgebs2d_(&ctxt,&scope,&top,&m,&n,A,&lda);
    }
    template<> inline void gebs2d<c_double>(int ctxt, char scope, char top, int m, int n, c_double *A, int lda) {
      zgebs2d_(&ctxt,&scope,&top,&m,&n,A,&lda);
    }
    template<> inline void gebs2d<c_float>(int ctxt, char scope, char top, int m, int n, c_float *A, int lda) {
      cgebs2d_(&ctxt,&scope,&top,&m,&n,A,&lda);
    }

    /* xGEBR2D */
    template<typename S> inline void gebr2d(int, char, char, int, int, S *, int, int, int);
    template<> inline void gebr2d<  int>(int ctxt, char scope, char top, int m, int n, int *A, int lda, int rsrc, int csrc) {
      igebr2d_(&ctxt,&scope,&top,&m,&n,A,&lda,&rsrc,&csrc);
    }
    template<> inline void gebr2d<double>(int ctxt, char scope, char top, int m, int n, double *A, int lda, int rsrc, int csrc) {
      dgebr2d_(&ctxt,&scope,&top,&m,&n,A,&lda,&rsrc,&csrc);
    }
    template<> inline void gebr2d< float>(int ctxt, char scope, char top, int m, int n, float *A, int lda, int rsrc, int csrc) {
      sgebr2d_(&ctxt,&scope,&top,&m,&n,A,&lda,&rsrc,&csrc);
    }
    template<> inline void gebr2d<c_double>(int ctxt, char scope, char top, int m, int n, c_double *A, int lda, int rsrc, int csrc) {
      zgebr2d_(&ctxt,&scope,&top,&m,&n,A,&lda,&rsrc,&csrc);
    }
    template<> inline void gebr2d<c_float>(int ctxt, char scope, char top, int m, int n, c_float *A, int lda, int rsrc, int csrc) {
      cgebr2d_(&ctxt,&scope,&top,&m,&n,A,&lda,&rsrc,&csrc);
    }

    /* xGSUM2D */
    template<typename S> inline void gsum2d(int, char, char, int, int, S *, int, int, int);
    template<> inline void gsum2d<double>(int ctxt, char scope, char top, int m, int n, double *A, int lda, int rdest, int cdest) {
      dgsum2d_(&ctxt,&scope,&top,&m,&n,A,&lda,&rdest,&cdest);
    }
    template<> inline void gsum2d< float>(int ctxt, char scope, char top, int m, int n,  float *A, int lda, int rdest, int cdest) {
      sgsum2d_(&ctxt,&scope,&top,&m,&n,A,&lda,&rdest,&cdest);
    }

    /* xGAMX2D */
    template<typename S> inline void gamx2d(int, char, char, int, int, S *, int, int *, int *, int, int, int);
    template<> inline void gamx2d<double>(int ctxt, char scope, char top, int m, int n, double *A, int lda, int *ra, int *ca, int ldia, int rdest, int cdest) {
      dgamx2d_(&ctxt,&scope,&top,&m,&n,A,&lda,ra,ca,&ldia,&rdest,&cdest);
    }
    template<> inline void gamx2d< float>(int ctxt, char scope, char top, int m, int n,  float *A, int lda, int *ra, int *ca, int ldia, int rdest, int cdest) {
      sgamx2d_(&ctxt,&scope,&top,&m,&n,A,&lda,ra,ca,&ldia,&rdest,&cdest);
    }

    /* PxAMAX */
    template<typename T> inline void pamax(int, T *, int *, T *, int, int, int *, int);
    template<> inline void pamax<  double>(int n,   double *amax, int *indx,   double *X, int ix, int jx, int *descX, int incx) {
      pdamax_(&n,amax,indx,X,&ix,&jx,descX,&incx);
    }
    template<> inline void pamax<   float>(int n,    float *amax, int *indx,    float *X, int ix, int jx, int *descX, int incx) {
      psamax_(&n,amax,indx,X,&ix,&jx,descX,&incx);
    }
    template<> inline void pamax<c_double>(int n, c_double *amax, int *indx, c_double *X, int ix, int jx, int *descX, int incx) {
      pzamax_(&n,amax,indx,X,&ix,&jx,descX,&incx);
    }
    template<> inline void pamax<c_float>(int n, c_float *amax, int *indx, c_float *X, int ix, int jx, int *descX, int incx) {
      pcamax_(&n,amax,indx,X,&ix,&jx,descX,&incx);
    }

    /* PxSWAP */
    template<typename T> inline void pswap(int, T *, int, int, int *, int, T *, int, int, int *, int);
    template<> inline void pswap<  double>(int n,   double *X, int ix, int jx, int *descX, int incx,   double *Y, int iy, int jy, int *descY, int incy) {
      pdswap_(&n,X,&ix,&jx,descX,&incx,Y,&iy,&jy,descY,&incy);
    }
    template<> inline void pswap<   float>(int n,    float *X, int ix, int jx, int *descX, int incx,    float *Y, int iy, int jy, int *descY, int incy) {
      psswap_(&n,X,&ix,&jx,descX,&incx,Y,&iy,&jy,descY,&incy);
    }
    template<> inline void pswap<c_double>(int n, c_double *X, int ix, int jx, int *descX, int incx, c_double *Y, int iy, int jy, int *descY, int incy) {
      pzswap_(&n,X,&ix,&jx,descX,&incx,Y,&iy,&jy,descY,&incy);
    }
    template<> inline void pswap<c_float>(int n, c_float *X, int ix, int jx, int *descX, int incx, c_float *Y, int iy, int jy, int *descY, int incy) {
      pcswap_(&n,X,&ix,&jx,descX,&incx,Y,&iy,&jy,descY,&incy);
    }

    /* PxSCAL */
    template<typename T> inline void pscal(int, T, T *, int, int, int *, int);
    template<> inline void pscal<  double>(int n,   double a,   double *X, int ix, int jx, int *descX, int incx) {
      pdscal_(&n,&a,X,&ix,&jx,descX,&incx);
    }
    template<> inline void pscal<   float>(int n,    float a,    float *X, int ix, int jx, int *descX, int incx) {
      psscal_(&n,&a,X,&ix,&jx,descX,&incx);
    }
    template<> inline void pscal<c_double>(int n, c_double a, c_double *X, int ix, int jx, int *descX, int incx) {
      pzscal_(&n,&a,X,&ix,&jx,descX,&incx);
    }
    template<> inline void pscal<c_float>(int n, c_float a, c_float *X, int ix, int jx, int *descX, int incx) {
      pcscal_(&n,&a,X,&ix,&jx,descX,&incx);
    }

    /* PxGEMV */
    template<typename T> inline void pgemv(char, int, int, T, T *, int, int, int *, T *, int, int, int*, int, T, T *, int, int, int *, int);
    template<> inline void pgemv<  double>(char transA, int m, int n,   double alpha,   double *A, int ia, int ja, int *descA,   double *X, int ix, int jx, int *descX, int incx,   double beta,   double *Y, int iy, int jy, int *descY, int incy) {
      pdgemv_(&transA,&m,&n,&alpha,A,&ia,&ja,descA,X,&ix,&jx,descX,&incx,&beta,Y,&iy,&jy,descY,&incy);
    }
    template<> inline void pgemv<   float>(char transA, int m, int n,    float alpha,    float *A, int ia, int ja, int *descA,    float *X, int ix, int jx, int *descX, int incx,    float beta,    float *Y, int iy, int jy, int *descY, int incy) {
      psgemv_(&transA,&m,&n,&alpha,A,&ia,&ja,descA,X,&ix,&jx,descX,&incx,&beta,Y,&iy,&jy,descY,&incy);
    }
    template<> inline void pgemv<c_double>(char transA, int m, int n, c_double alpha, c_double *A, int ia, int ja, int *descA, c_double *X, int ix, int jx, int *descX, int incx, c_double beta, c_double *Y, int iy, int jy, int *descY, int incy) {
      pzgemv_(&transA,&m,&n,&alpha,A,&ia,&ja,descA,X,&ix,&jx,descX,&incx,&beta,Y,&iy,&jy,descY,&incy);
    }
    template<> inline void pgemv<c_float>(char transA, int m, int n, c_float alpha, c_float *A, int ia, int ja, int *descA, c_float *X, int ix, int jx, int *descX, int incx, c_float beta, c_float *Y, int iy, int jy, int *descY, int incy) {
      pcgemv_(&transA,&m,&n,&alpha,A,&ia,&ja,descA,X,&ix,&jx,descX,&incx,&beta,Y,&iy,&jy,descY,&incy);
    }

    /* PxGEMM */
    template<typename T> inline void pgemm(char, char, int, int, int, T, T*, int, int, int*, T*, int, int, int*, T, T*, int, int, int*);
    template<> inline void pgemm<double> (char transA, char transB, int m, int n, int k,    double alpha,   double *A, int ia, int ja, int *descA,   double *B, int ib, int jb, int *descB,   double beta, double *C, int ic, int jc, int *descC) {
      pdgemm_(&transA,&transB,&m,&n,&k,&alpha,A,&ia,&ja,descA,B,&ib,&jb,descB,&beta,C,&ic,&jc,descC);
    }
    template<> inline void pgemm<float>  (char transA, char transB, int m, int n, int k,     float alpha,    float *A, int ia, int ja, int *descA,    float *B, int ib, int jb, int *descB,    float beta,   float *C, int ic, int jc, int *descC) {
      psgemm_(&transA,&transB,&m,&n,&k,&alpha,A,&ia,&ja,descA,B,&ib,&jb,descB,&beta,C,&ic,&jc,descC);
    }
    template<> inline void pgemm<c_double>(char transA, char transB, int m, int n, int k, c_double alpha, c_double *A, int ia, int ja, int *descA, c_double *B, int ib, int jb, int *descB, c_double beta, c_double *C, int ic, int jc, int *descC) {
      pzgemm_(&transA,&transB,&m,&n,&k,&alpha,A,&ia,&ja,descA,B,&ib,&jb,descB,&beta,C,&ic,&jc,descC);
    }
    template<> inline void pgemm<c_float>(char transA, char transB, int m, int n, int k, c_float alpha, c_float *A, int ia, int ja, int *descA, c_float *B, int ib, int jb, int *descB, c_float beta, c_float *C, int ic, int jc, int *descC) {
      pcgemm_(&transA,&transB,&m,&n,&k,&alpha,A,&ia,&ja,descA,B,&ib,&jb,descB,&beta,C,&ic,&jc,descC);
    }

    /* PxLACGV */
    template<typename T> inline void placgv(int, T *, int, int, int *, int);
    template<> inline void placgv<  double>(int n,   double *X, int ix, int jx, int *descX, int incx) {
      // Nothing to do
    }
    template<> inline void placgv<   float>(int n,    float *X, int ix, int jx, int *descX, int incx) {
      // Nothing to do
    }
    template<> inline void placgv<c_double>(int n, c_double *X, int ix, int jx, int *descX, int incx) {
      pzlacgv_(&n,X,&ix,&jx,descX,&incx);
    }
    template<> inline void placgv<c_float>(int n, c_float *X, int ix, int jx, int *descX, int incx) {
      pclacgv_(&n,X,&ix,&jx,descX,&incx);
    }

    /* PxGER */
    template<typename T> inline void pgeru(int, int, T, T*, int, int, int*, int, T*, int, int, int *, int, T*, int, int, int*);
    template<> inline void pgeru<  double>(int m, int n,   double alpha,   double *X, int ix, int jx, int *descX, int incx,   double *Y, int iy, int jy, int *descY, int incy,   double *A, int ia, int ja, int *descA) {
      pdger_ (&m,&n,&alpha,X,&ix,&jx,descX,&incx,Y,&iy,&jy,descY,&incy,A,&ia,&ja,descA);
    }
    template<> inline void pgeru<   float>(int m, int n,    float alpha,    float *X, int ix, int jx, int *descX, int incx,    float *Y, int iy, int jy, int *descY, int incy,    float *A, int ia, int ja, int *descA) {
      psger_ (&m,&n,&alpha,X,&ix,&jx,descX,&incx,Y,&iy,&jy,descY,&incy,A,&ia,&ja,descA);
    }
    template<> inline void pgeru<c_double>(int m, int n, c_double alpha, c_double *X, int ix, int jx, int *descX, int incx, c_double *Y, int iy, int jy, int *descY, int incy, c_double *A, int ia, int ja, int *descA) {
      pzgeru_(&m,&n,&alpha,X,&ix,&jx,descX,&incx,Y,&iy,&jy,descY,&incy,A,&ia,&ja,descA);
    }
    template<> inline void pgeru<c_float>(int m, int n, c_float alpha, c_float *X, int ix, int jx, int *descX, int incx, c_float *Y, int iy, int jy, int *descY, int incy, c_float *A, int ia, int ja, int *descA) {
      pcgeru_(&m,&n,&alpha,X,&ix,&jx,descX,&incx,Y,&iy,&jy,descY,&incy,A,&ia,&ja,descA);
    }

    /* PxLASWP */
    template<typename scalar> inline void plaswp(char direc, char rowcol, int n, scalar* a, int ia, int ja, int* desca, int k1, int k2, int* ipiv);
    template<> inline void plaswp(char direc, char rowcol, int n, double* a, int ia, int ja, int* desca, int k1, int k2, int* ipiv) {
      pdlaswp_(&direc,&rowcol,&n,a,&ia,&ja,desca,&k1,&k2,ipiv);
    }
    template<> inline void plaswp(char direc, char rowcol, int n, float* a, int ia, int ja, int* desca, int k1, int k2, int* ipiv) {
      pslaswp_(&direc,&rowcol,&n,a,&ia,&ja,desca,&k1,&k2,ipiv);
    }
    template<> inline void plaswp(char direc, char rowcol, int n, c_double* a, int ia, int ja, int* desca, int k1, int k2, int* ipiv) {
      pzlaswp_(&direc,&rowcol,&n,a,&ia,&ja,desca,&k1,&k2,ipiv);
    }
    template<> inline void plaswp(char direc, char rowcol, int n, c_float* a, int ia, int ja, int* desca, int k1, int k2, int* ipiv) {
      pclaswp_(&direc,&rowcol,&n,a,&ia,&ja,desca,&k1,&k2,ipiv);
    }

    /* PxLAPIV */
    template<typename scalar> inline void plapiv(char direc, char rowcol, char pivroc, int m, int n, scalar* a, int ia, int ja, int* desca, int* ipiv, int ip, int jp, int* descip, int* iwork);
    template<> inline void plapiv(char direc, char rowcol, char pivroc, int m, int n, double* a, int ia, int ja, int* desca, int* ipiv, int ip, int jp, int* descip, int* iwork) {
      pdlapiv_(&direc,&rowcol,&pivroc,&m,&n,a,&ia,&ja,desca,ipiv,&ip,&jp,descip,iwork);
    }
    template<> inline void plapiv(char direc, char rowcol, char pivroc, int m, int n, float* a, int ia, int ja, int* desca, int* ipiv, int ip, int jp, int* descip, int* iwork) {
      pslapiv_(&direc,&rowcol,&pivroc,&m,&n,a,&ia,&ja,desca,ipiv,&ip,&jp,descip,iwork);
    }
    template<> inline void plapiv(char direc, char rowcol, char pivroc, int m, int n, c_double* a, int ia, int ja, int* desca, int* ipiv, int ip, int jp, int* descip, int* iwork) {
      pzlapiv_(&direc,&rowcol,&pivroc,&m,&n,a,&ia,&ja,desca,ipiv,&ip,&jp,descip,iwork);
    }
    template<> inline void plapiv(char direc, char rowcol, char pivroc, int m, int n, c_float* a, int ia, int ja, int* desca, int* ipiv, int ip, int jp, int* descip, int* iwork) {
      pclapiv_(&direc,&rowcol,&pivroc,&m,&n,a,&ia,&ja,desca,ipiv,&ip,&jp,descip,iwork);
    }

    /* PxTRSM */
    template<typename T> inline void ptrsm(char, char, char, char, int, int, T, T *, int, int , int *, T *, int ,int, int *);
    template<> inline void ptrsm<  double>(char side, char uplo, char trans, char diag, int m, int n,   double alpha,   double *A, int ia, int ja, int *descA,   double *B, int ib, int jb, int *descB) {
      pdtrsm_(&side,&uplo,&trans,&diag,&m,&n,&alpha,A,&ia,&ja,descA,B,&ib,&jb,descB);
    }
    template<> inline void ptrsm<   float>(char side, char uplo, char trans, char diag, int m, int n,    float alpha,    float *A, int ia, int ja, int *descA,    float *B, int ib, int jb, int *descB) {
      pstrsm_(&side,&uplo,&trans,&diag,&m,&n,&alpha,A,&ia,&ja,descA,B,&ib,&jb,descB);
    }
    template<> inline void ptrsm<c_double>(char side, char uplo, char trans, char diag, int m, int n, c_double alpha, c_double *A, int ia, int ja, int *descA, c_double *B, int ib, int jb, int *descB) {
      pztrsm_(&side,&uplo,&trans,&diag,&m,&n,&alpha,A,&ia,&ja,descA,B,&ib,&jb,descB);
    }
    template<> inline void ptrsm<c_float>(char side, char uplo, char trans, char diag, int m, int n, c_float alpha, c_float *A, int ia, int ja, int *descA, c_float *B, int ib, int jb, int *descB) {
      pctrsm_(&side,&uplo,&trans,&diag,&m,&n,&alpha,A,&ia,&ja,descA,B,&ib,&jb,descB);
    }

    /* PxTRSV */
    template<typename T> inline void ptrsv(char, char, char, int, T *, int, int , int *, T *, int ,int, int *, int);
    template<> inline void ptrsv<  double>(char uplo, char trans, char diag, int m, double *A, int ia, int ja, int *descA,   double *B, int ib, int jb, int *descB, int incB) {
      pdtrsv_(&uplo,&trans,&diag,&m,A,&ia,&ja,descA,B,&ib,&jb,descB,&incB);
    }
    template<> inline void ptrsv<   float>(char uplo, char trans, char diag, int m, float *A, int ia, int ja, int *descA,    float *B, int ib, int jb, int *descB, int incB) {
      pstrsv_(&uplo,&trans,&diag,&m,A,&ia,&ja,descA,B,&ib,&jb,descB,&incB);
    }
    template<> inline void ptrsv<c_double>(char uplo, char trans, char diag, int m, c_double *A, int ia, int ja, int *descA, c_double *B, int ib, int jb, int *descB, int incB) {
      pztrsv_(&uplo,&trans,&diag,&m,A,&ia,&ja,descA,B,&ib,&jb,descB,&incB);
    }
    template<> inline void ptrsv<c_float>(char uplo, char trans, char diag, int m, c_float *A, int ia, int ja, int *descA, c_float *B, int ib, int jb, int *descB, int incB) {
      pctrsv_(&uplo,&trans,&diag,&m,A,&ia,&ja,descA,B,&ib,&jb,descB,&incB);
    }

    /* PxLANGE */
    template<typename T, typename S> inline S plange(char , int , int , T *, int, int, int *, S *);
    template<> inline double plange<  double,double>(char norm, int m, int n,   double *A, int ia, int ja, int *descA, double *work) {
      return pdlange_(&norm,&m,&n,A,&ia,&ja,descA,work);
    }
    template<> inline  float plange<   float, float>(char norm, int m, int n,    float *A, int ia, int ja, int *descA,  float *work) {
      return pslange_(&norm,&m,&n,A,&ia,&ja,descA,work);
    }
    template<> inline double plange<c_double,double>(char norm, int m, int n, c_double *A, int ia, int ja, int *descA, double *work) {
      return pzlange_(&norm,&m,&n,A,&ia,&ja,descA,work);
    }
    template<> inline  float plange<c_float, float>(char norm, int m, int n, c_float *A, int ia, int ja, int *descA,  float *work) {
      return pclange_(&norm,&m,&n,A,&ia,&ja,descA,work);
    }

    /* PxGEADD */
    template<typename T> inline void pgeadd(char, int, int, T, T *, int, int, int *, T, T *, int, int, int *);
    template<> inline void pgeadd<  double>(char trans, int m, int n,   double alpha,   double *A, int ia, int ja, int *descA,   double beta,   double *C, int ic, int jc, int *descC) {
      pdgeadd_(&trans,&m,&n,&alpha,A,&ia,&ja,descA,&beta,C,&ic,&jc,descC);
    }
    template<> inline void pgeadd<   float>(char trans, int m, int n,    float alpha,    float *A, int ia, int ja, int *descA,    float beta,    float *C, int ic, int jc, int *descC) {
      psgeadd_(&trans,&m,&n,&alpha,A,&ia,&ja,descA,&beta,C,&ic,&jc,descC);
    }
    template<> inline void pgeadd<c_double>(char trans, int m, int n, c_double alpha, c_double *A, int ia, int ja, int *descA, c_double beta, c_double *C, int ic, int jc, int *descC) {
      pzgeadd_(&trans,&m,&n,&alpha,A,&ia,&ja,descA,&beta,C,&ic,&jc,descC);
    }
    template<> inline void pgeadd<c_float>(char trans, int m, int n, c_float alpha, c_float *A, int ia, int ja, int *descA, c_float beta, c_float *C, int ic, int jc, int *descC) {
      pcgeadd_(&trans,&m,&n,&alpha,A,&ia,&ja,descA,&beta,C,&ic,&jc,descC);
    }

    /* PxLACPY */
    template<typename T> inline void placpy(char, int, int, T *, int, int, int *, T *, int, int, int *);
    template<> inline void placpy<  double>(char trans, int m, int n,   double *A, int ia, int ja, int *descA,   double *C, int ic, int jc, int *descC) {
      pdlacpy_(&trans,&m,&n,A,&ia,&ja,descA,C,&ic,&jc,descC);
    }
    template<> inline void placpy<   float>(char trans, int m, int n,    float *A, int ia, int ja, int *descA,    float *C, int ic, int jc, int *descC) {
      pslacpy_(&trans,&m,&n,A,&ia,&ja,descA,C,&ic,&jc,descC);
    }
    template<> inline void placpy<c_double>(char trans, int m, int n, c_double *A, int ia, int ja, int *descA, c_double *C, int ic, int jc, int *descC) {
      pzlacpy_(&trans,&m,&n,A,&ia,&ja,descA,C,&ic,&jc,descC);
    }
    template<> inline void placpy<c_float>(char trans, int m, int n, c_float *A, int ia, int ja, int *descA, c_float *C, int ic, int jc, int *descC) {
      pclacpy_(&trans,&m,&n,A,&ia,&ja,descA,C,&ic,&jc,descC);
    }

    /* pxGEMR2D */
    template<typename T> inline void pgemr2d(int, int, T *, int, int, int *, T *, int, int, int *, int);
    template<> inline void pgemr2d<  double>(int m, int n,   double *A, int ia, int ja, int *descA,   double *B, int ib, int jb, int *descB, int ctxt) {
      pdgemr2d_(&m,&n,A,&ia,&ja,descA,B,&ib,&jb,descB,&ctxt);
    }
    template<> inline void pgemr2d<   float>(int m, int n,    float *A, int ia, int ja, int *descA,    float *B, int ib, int jb, int *descB, int ctxt) {
      psgemr2d_(&m,&n,A,&ia,&ja,descA,B,&ib,&jb,descB,&ctxt);
    }
    template<> inline void pgemr2d<c_double>(int m, int n, c_double *A, int ia, int ja, int *descA, c_double *B, int ib, int jb, int *descB, int ctxt) {
      pzgemr2d_(&m,&n,A,&ia,&ja,descA,B,&ib,&jb,descB,&ctxt);
    }
    template<> inline void pgemr2d<c_float>(int m, int n, c_float *A, int ia, int ja, int *descA, c_float *B, int ib, int jb, int *descB, int ctxt) {
      pcgemr2d_(&m,&n,A,&ia,&ja,descA,B,&ib,&jb,descB,&ctxt);
    }

    /* PxTRAN */
    template<typename T> inline void ptranc(int, int, T, T*, int, int, int *, T, T*, int, int, int *);
    template<> inline void ptranc<  double>(int m, int n,   double alpha,   double *A, int ia, int ja, int *descA,   double beta,   double *C, int ic, int jc, int *descC) {
      pdtran_(&m,&n,&alpha,A,&ia,&ja,descA,&beta,C,&ic,&jc,descC);
    }
    template<> inline void ptranc<   float>(int m, int n,    float alpha,    float *A, int ia, int ja, int *descA,    float beta,    float *C, int ic, int jc, int *descC) {
      pstran_(&m,&n,&alpha,A,&ia,&ja,descA,&beta,C,&ic,&jc,descC);
    }
    template<> inline void ptranc<c_double>(int m, int n, c_double alpha, c_double *A, int ia, int ja, int *descA, c_double beta, c_double *C, int ic, int jc, int *descC) {
      pztranc_(&m,&n,&alpha,A,&ia,&ja,descA,&beta,C,&ic,&jc,descC);
    }
    template<> inline void ptranc<c_float>(int m, int n, c_float alpha, c_float *A, int ia, int ja, int *descA, c_float beta, c_float *C, int ic, int jc, int *descC) {
      pctranc_(&m,&n,&alpha,A,&ia,&ja,descA,&beta,C,&ic,&jc,descC);
    }

    /* PxGEQPF; this one is used only in the examples */
    template<typename T> inline void pgeqpf (int, int, T *A, int, int , int *, int, int, int, int, int, int);
    template<> inline void pgeqpf< double>(int m, int n,   double *A, int ia, int ja, int *descA, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int ierr;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      auto ipiv = new int[loccA];
      auto tau = new double[loccA];
      int lwork = 3*(1+locrA+loccA);
      auto work = new double[lwork];
      pdgeqpf_(&m, &n, A, &ia, &ja, descA, ipiv, tau, work, &lwork, &ierr);
      delete[] tau;
      delete[] work;
      delete[] ipiv;
    }
    template<> inline void pgeqpf<  float>(int m, int n,    float *A, int ia, int ja, int *descA, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int ierr;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      auto ipiv = new int[loccA];
      auto tau = new float[loccA];
      int lwork = 3*(1+locrA+loccA);
      auto work = new float[lwork];
      psgeqpf_(&m, &n, A, &ia, &ja, descA, ipiv, tau, work, &lwork, &ierr);
      delete[] tau;
      delete[] work;
      delete[] ipiv;
    }
    template<> inline void pgeqpf<c_double>(int m, int n, c_double *A, int ia, int ja, int *descA, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int ierr;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      auto ipiv = new int[loccA];
      auto tau = new c_double[loccA];
      int lwork = 3*(1+locrA+loccA);
      auto work = new c_double[lwork];
      int lrwork = 2*(1+loccA);
      auto rwork = new double[lrwork];
      pzgeqpf_(&m, &n, A, &ia, &ja, descA, ipiv, tau, work, &lwork, rwork, &lrwork, &ierr);
      delete[] tau;
      delete[] work;
      delete[] ipiv;
      delete[] rwork;
    }
    template<> inline void pgeqpf<c_float>(int m, int n, c_float *A, int ia, int ja, int *descA, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int ierr;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      auto ipiv = new int[loccA];
      auto tau = new c_float[loccA];
      int lwork = 3*(1+locrA+loccA);
      auto work = new c_float[lwork];
      int lrwork = 2*(1+loccA);
      auto rwork = new float[lrwork];
      pcgeqpf_(&m, &n, A, &ia, &ja, descA, ipiv, tau, work, &lwork, rwork, &lrwork, &ierr);
      delete[] tau;
      delete[] work;
      delete[] ipiv;
      delete[] rwork;
    }

    /* PxGEQPFmod */
    template<typename T, typename S> inline void pgeqpfmod(int, int, T *, int, int , int *, int *, int *, int *, S, int, int, int, int, int, int);
    template<> inline void pgeqpfmod(int m, int n, double *A, int ia, int ja, int *descA, int *J, int *piv, int *r, double tol, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int info;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      int lwork = 3*(1+locrA+loccA);
      auto twork = new double[lwork];
      auto tau = new double[loccA];
      auto ipiv = new int[n];
      int IONE = 1;
      pdgeqpfmod_(&m, &n, A, &IONE, &IONE, descA, ipiv, tau, twork, &lwork, &info, J, piv, r, &tol);
      delete[] tau;
      delete[] twork;
      delete[] ipiv;
    }
    template<> inline void pgeqpfmod(int m, int n,  float *A, int ia, int ja, int *descA, int *J, int *piv, int *r,  float tol, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int info;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      int lwork=3*(1+locrA+loccA);
      auto twork = new float[lwork];
      auto tau = new float[loccA];
      auto ipiv = new int[n];
      int IONE = 1;
      psgeqpfmod_(&m, &n, A, &IONE, &IONE, descA, ipiv, tau, twork, &lwork, &info, J, piv, r, &tol);
      delete[] tau;
      delete[] twork;
      delete[] ipiv;
    }
    template<> inline void pgeqpfmod(int m, int n, c_double *A, int ia, int ja, int *descA, int *J, int *piv, int *r, double tol, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int info;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      int lwork = 3*(1+locrA+loccA);
      auto twork = new c_double[lwork];
      int lrwork = 2*(1+loccA);
      auto rwork = new double[lrwork];
      auto tau = new c_double[loccA];
      auto ipiv = new int[n];
      int IONE = 1;
      pzgeqpfmod_(&m, &n, A, &IONE, &IONE, descA, ipiv, tau, twork, &lwork, rwork, &lrwork, &info, J, piv, r, &tol);
      delete[] tau;
      delete[] twork;
      delete[] rwork;
      delete[] ipiv;
    }
    template<> inline void pgeqpfmod(int m, int n, c_float *A, int ia, int ja, int *descA, int *J, int *piv, int *r,  float tol, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int info;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      int lwork = 3*(1+locrA+loccA);
      auto twork = new c_float[lwork];
      int lrwork = 2*(1+loccA);
      auto rwork = new float[lrwork];
      auto tau = new c_float[loccA];
      auto ipiv = new int[n];
      int IONE = 1;
      pcgeqpfmod_(&m, &n, A, &IONE, &IONE, descA, ipiv, tau, twork, &lwork, rwork, &lrwork, &info, J, piv, r, &tol);
      delete[] tau;
      delete[] twork;
      delete[] rwork;
      delete[] ipiv;
    }

    /* PxGETRF */
    template<typename T> inline int pgetrf(int, int, T *, int, int, int *, int *);
    template<> inline int pgetrf(int m, int n,   double *A, int ia, int ja, int *descA, int *ipiv) {
      int info;
      pdgetrf_(&m,&n,A,&ia,&ja,descA,ipiv,&info);
      return info;
    }
    template<> inline int pgetrf(int m, int n,    float *A, int ia, int ja, int *descA, int *ipiv) {
      int info;
      psgetrf_(&m,&n,A,&ia,&ja,descA,ipiv,&info);
      return info;
    }
    template<> inline int pgetrf(int m, int n, c_double *A, int ia, int ja, int *descA, int *ipiv) {
      int info;
      pzgetrf_(&m,&n,A,&ia,&ja,descA,ipiv,&info);
      return info;
    }
    template<> inline int pgetrf(int m, int n, c_float *A, int ia, int ja, int *descA, int *ipiv) {
      int info;
      pcgetrf_(&m,&n,A,&ia,&ja,descA,ipiv,&info);
      return info;
    }

    /* PxGETRS */
    template<typename T> inline int pgetrs(char, int, int, T *, int, int, int *, int *, T *, int, int, int *);
    template<> inline int pgetrs(char trans, int m, int n,   double *A, int ia, int ja, int *descA, int *ipiv, double *B, int ib, int jb, int *descB) {
      int info;
      pdgetrs_(&trans,&m,&n,A,&ia,&ja,descA,ipiv,B,&ib,&jb,descB,&info);
      return info;
    }
    template<> inline int pgetrs(char trans, int m, int n,    float *A, int ia, int ja, int *descA, int *ipiv, float *B, int ib, int jb, int *descB) {
      int info;
      psgetrs_(&trans,&m,&n,A,&ia,&ja,descA,ipiv,B,&ib,&jb,descB,&info);
      return info;
    }
    template<> inline int pgetrs(char trans, int m, int n, c_double *A, int ia, int ja, int *descA, int *ipiv, c_double *B, int ib, int jb, int *descB) {
      int info;
      pzgetrs_(&trans,&m,&n,A,&ia,&ja,descA,ipiv,B,&ib,&jb,descB,&info);
      return info;
    }
    template<> inline int pgetrs(char trans, int m, int n, c_float *A, int ia, int ja, int *descA, int *ipiv, c_float *B, int ib, int jb, int *descB) {
      int info;
      pcgetrs_(&trans,&m,&n,A,&ia,&ja,descA,ipiv,B,&ib,&jb,descB,&info);
      return info;
    }

    /* PxGELQF */
    template<typename T> inline void pgelqf (int, int, T *A, int, int , int *, T *, int, int, int, int, int, int);
    template<> inline void pgelqf< double>(int m, int n,   double *A, int ia, int ja, int *descA, double * tau, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int ierr;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      int lwork = mb*(mb+locrA+loccA);
      auto work = new double[lwork];
      pdgelqf_(&m, &n, A, &ia, &ja, descA, tau, work, &lwork, &ierr);
      delete[] work;
    }
    template<> inline void pgelqf<  float>(int m, int n,    float *A, int ia, int ja, int *descA, float *tau, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int ierr;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      int lwork = mb*(mb+locrA+loccA);
      auto work = new float[lwork];
      psgelqf_(&m, &n, A, &ia, &ja, descA, tau, work, &lwork, &ierr);
      delete[] work;
    }
    template<> inline void pgelqf<c_double>(int m, int n, c_double *A, int ia, int ja, int *descA, c_double *tau, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int ierr;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      int lwork = mb*(mb+locrA+loccA);
      auto work = new c_double[lwork];
      pzgelqf_(&m, &n, A, &ia, &ja, descA, tau, work, &lwork, &ierr);
      delete[] work;
    }
    template<> inline void pgelqf<c_float>(int m, int n, c_float *A, int ia, int ja, int *descA, c_float *tau, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int ierr;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      int lwork = mb*(mb+locrA+loccA);
      auto work = new c_float[lwork];
      pcgelqf_(&m, &n, A, &ia, &ja, descA, tau, work, &lwork, &ierr);
      delete[] work;
    }

    /* PxxxGLQ */
    template<typename T> inline void pxxglq(int, int, int, T *, int, int, int *, T *, int, int, int, int, int, int);
    template<> inline void pxxglq<double>(int m, int n, int k, double *A, int ia, int ja, int *descA, double *tau, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int ierr;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      int lwork=mb*(mb+locrA+loccA);
      auto work = new double[lwork];
      pdorglq_(&m, &n, &k, A, &ia, &ja, descA, tau, work, &lwork, &ierr);
      delete[] work;
    }
    template<> inline void pxxglq<  float>(int m, int n, int k, float *A, int ia, int ja, int *descA, float *tau, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int ierr;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      int lwork = mb*(mb+locrA+loccA);
      auto work = new float[lwork];
      psorglq_(&m, &n, &k, A, &ia, &ja, descA, tau, work, &lwork, &ierr);
      delete[] work;
    }
    template<> inline void pxxglq<c_double>(int m, int n, int k, c_double *A, int ia, int ja, int *descA, c_double *tau, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int ierr;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      int lwork = mb*(mb+locrA+loccA);
      auto work = new c_double[lwork];
      pzunglq_(&m, &n, &k, A, &ia, &ja, descA, tau, work, &lwork, &ierr);
      delete[] work;
    }
    template<> inline void pxxglq<c_float>(int m, int n, int k, c_float *A, int ia, int ja, int *descA, c_float *tau, int mb, int nb, int myrow, int mycol, int nprow, int npcol) {
      int ierr;
      int locrA = numroc(m, mb, myrow, 0, nprow);
      int loccA = numroc(n, nb, mycol, 0, npcol);
      int lwork = mb*(mb+locrA+loccA);
      auto work = new c_float[lwork];
      pcunglq_(&m, &n, &k, A, &ia, &ja, descA, tau, work, &lwork, &ierr);
      delete[] work;
    }

    /* PxNRM2; this one is used only in the examples */
    template<typename T, typename S> inline S pnrm2(int, T *, int, int, int *, int);
    template<> inline double pnrm2<double, double>(int n, double *X, int ix, int jx, int *descX, int incx) {
      double nrm;
      pdnrm2_(&n,&nrm,X,&ix,&jx,descX,&incx);
      return nrm;
    }
    template<> inline float pnrm2<float, float>(int n, float *X, int ix, int jx, int *descX, int incx) {
      float nrm;
      psnrm2_(&n,&nrm,X,&ix,&jx,descX,&incx);
      return nrm;
    }
    template<> inline double pnrm2<c_double, double>(int n, c_double *X, int ix, int jx, int *descX, int incx) {
      double nrm;
      pdznrm2_(&n,&nrm,X,&ix,&jx,descX,&incx);
      return nrm;
    }
    template<> inline float pnrm2<c_float, float>(int n, c_float *X, int ix, int jx, int *descX, int incx) {
      float nrm;
      pscnrm2_(&n,&nrm,X,&ix,&jx,descX,&incx);
      return nrm;
    }

    // New
    /* PxDOT; this one is used only in the examples */
    template<typename T> inline void pdot(int, T *, T *, int, int, int *, int, T *, int, int, int *, int);
    template<> inline void pdot<double>(int n, double *dot, double *X, int iX, int jX, int *descX, int incX, double *Y, int iY, int jY, int *descY, int incY) {
      pddot_(&n,dot,X,&iX,&jX,descX,&incX,Y,&iY,&jY,descY,&incY);
    }
    template<> inline void pdot<float>(int n, float *dot, float *X, int iX, int jX, int *descX, int incX, float *Y, int iY, int jY, int *descY, int incY) {
      psdot_(&n,dot,X,&iX,&jX,descX,&incX,Y,&iY,&jY,descY,&incY);
    }
    template<> inline void pdot<c_double>(int n, c_double *dot, c_double *X, int iX, int jX, int *descX, int incX, c_double *Y, int iY, int jY, int *descY, int incY) {
      pzdot_(&n,dot,X,&iX,&jX,descX,&incX,Y,&iY,&jY,descY,&incY);
    }
    template<> inline void pdot<c_float>(int n, c_float *dot, c_float *X, int iX, int jX, int *descX, int incX, c_float *Y, int iY, int jY, int *descY, int incY) {
      pcdot_(&n,dot,X,&iX,&jX,descX,&incX,Y,&iY,&jY,descY,&incY);
    }


    template<typename T> inline int pgeqrf(int, int, T*, int, int, int *, T*, T*, int);
    template<> inline int pgeqrf<double>(int m, int n, double *A, int ia, int ja, int *descA, double *tau, double* work, int lwork)
    { int info; pdgeqrf_(&m, &n, A, &ia, &ja, descA, tau, work, &lwork, &info); return info; }
    template<> inline int pgeqrf<float>(int m, int n, float *A, int ia, int ja, int *descA, float *tau, float* work, int lwork)
    { int info; psgeqrf_(&m, &n, A, &ia, &ja, descA, tau, work, &lwork, &info); return info; }
    template<> inline int pgeqrf<c_double>(int m, int n, c_double *A, int ia, int ja, int *descA, c_double *tau, c_double* work, int lwork)
    { int info; pzgeqrf_(&m, &n, A, &ia, &ja, descA, tau, work, &lwork, &info); return info; }
    template<> inline int pgeqrf<c_float>(int m, int n, c_float *A, int ia, int ja, int *descA, c_float *tau, c_float* work, int lwork)
    { int info; pcgeqrf_(&m, &n, A, &ia, &ja, descA, tau, work, &lwork, &info); return info; }
    template<typename T> inline int pgeqrf(int m, int n, T* A, int ia, int ja, int *descA, T* tau) {
      T lwork;
      pgeqrf(m, n, A, ia, ja, descA, tau, &lwork, -1);
      int ilwork = int(std::real(lwork));
      auto work = new T[ilwork];
      int info = pgeqrf(m, n, A, ia, ja, descA, tau, work, ilwork);
      delete[] work;
      STRUMPACK_FLOPS((is_complex<T>()?4:1)*
		      static_cast<long long int>(((m>n) ? (double(n)*(double(n)*(.5-(1./3.)*double(n)+double(m)) + double(m) + 23./6.)) : (double(m)*(double(m)*(-.5-(1./3.)*double(m)+double(n)) + 2.*double(n) + 23./6.)))
						 + ((m>n) ? (double(n)*(double(n)*(.5-(1./3.)*double(n)+double(m)) + 5./6.)) : (double(m)*(double(m)*(-.5-(1./3.)*double(m)+double(n)) + double(n) + 5./6.)))));
      return info;
    }


    template<typename T> inline int pxxgqr(int m, int n, int k, T* A, int ia, int ja, int* descA, T* tau, T* work, int lwork);
    template<> inline int pxxgqr(int m, int n, int k, double* A, int ia, int ja, int* descA, double* tau, double* work, int lwork)
    { int info; pdorgqr_(&m, &n, &k, A, &ia, &ja, descA, tau, work, &lwork, &info); return info; }
    template<> inline int pxxgqr(int m, int n, int k, float* A, int ia, int ja, int* descA, float* tau, float* work, int lwork)
    { int info; psorgqr_(&m, &n, &k, A, &ia, &ja, descA, tau, work, &lwork, &info); return info; }
    template<> inline int pxxgqr(int m, int n, int k, c_double* A, int ia, int ja, int* descA, c_double* tau, c_double* work, int lwork)
    { int info; pzungqr_(&m, &n, &k, A, &ia, &ja, descA, tau, work, &lwork, &info); return info; }
    template<> inline int pxxgqr(int m, int n, int k, c_float* A, int ia, int ja, int* descA, c_float* tau, c_float* work, int lwork)
    { int info; pcungqr_(&m, &n, &k, A, &ia, &ja, descA, tau, work, &lwork, &info); return info; }

    template<typename T> inline int pxxgqr(int m, int n, int k, T* A, int ia, int ja, int* descA, T* tau) {
      T lwork;
      pxxgqr(m, n, k, A, ia, ja, descA, tau, &lwork, -1);
      int ilwork = int(std::real(lwork));
      auto work = new T[ilwork];
      int info = pxxgqr(m, n, k, A, ia, ja, descA, tau, work, ilwork);
      STRUMPACK_FLOPS((is_complex<T>()?4:1)*static_cast<long long int>((n==k) ? ((2./3.)*double(n)*double(n)*(3.*double(m) - double(n))) : (4.*double(m)*double(n)*double(k) - 2.*(double(m) + double(n))*double(k)*double(k) + (4./3.)*double(k)*double(k)*double(k))));
      delete[] work;
      return info;
    }

  } // end namespace scalapack
} // end namespace strumpack

#endif // SCALAPACK_HPP
