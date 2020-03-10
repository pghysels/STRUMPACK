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

#include "ScaLAPACKWrapper.hpp"
#include "StrumpackFortranCInterface.h"

#define BLACSCTXTSIZE 9
#define BLACSdtype    0
#define BLACSctxt     1
#define BLACSm        2
#define BLACSn        3
#define BLACSmb       4
#define BLACSnb       5
#define BLACSrsrc     6
#define BLACScsrc     7
#define BLACSlld      8


namespace strumpack {
  namespace scalapack {

    extern "C" {

      ///////////////////////////////////////////////
      ////// ScaLAPACK //////////////////////////////
      ///////////////////////////////////////////////
      int STRUMPACK_FC_GLOBAL(numroc,NUMROC)
        (int*, int*, int* , int *, int *);
      void STRUMPACK_FC_GLOBAL(descinit,DESCINIT)
        (int *, int *, int *, int *, int *, int *,
         int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(descset,DESCSET)
        (int *, int *, int *, int *, int *, int *,
         int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(infog1l,INFOG1L)
        (int *, int *, int *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(infog2l,INFOG2L)
        (int *, int *, const int *, int *, int *, int *,
         int *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(igamn2d,IGAMN2D)
        (int *, const char *, const char *, int *, int *,
         int *, int *, int *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL_(pb_topget,PB_TOPGET)
        (const int*, const char*, const char*, char*);
      void STRUMPACK_FC_GLOBAL_(pb_topset,PB_TOPSET)
        (const int*, const char*, const char*, const char*);

      void STRUMPACK_FC_GLOBAL(igebs2d,IGEBS2D)
        (int *, const char *, const char *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(sgebs2d,SGEBS2D)
        (int *, const char *, const char *, int *, int *, float *, int *);
      void STRUMPACK_FC_GLOBAL(dgebs2d,DGEBS2D)
        (int *, const char *, const char *, int *, int *, double *, int *);
      void STRUMPACK_FC_GLOBAL(cgebs2d,CGEBS2D)
        (int *, const char *, const char *, int *, int *,
         std::complex<float> *, int *);
      void STRUMPACK_FC_GLOBAL(zgebs2d,ZGEBS2D)
        (int *, const char *, const char *, int *, int *,
         std::complex<double> *, int *);

      void STRUMPACK_FC_GLOBAL(igebr2d,IGEBR2D)
        (int *, const char *, const char *, int *, int *,
         int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(sgebr2d,SGEBR2D)
        (int *, const char *, const char *, int *, int *,
         float *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(dgebr2d,DGEBR2D)
        (int *, const char *, const char *, int *, int *,
         double *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(cgebr2d,CGEBR2D)
        (int *, const char *, const char *, int *, int *,
         std::complex<float> *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(zgebr2d,ZGEBR2D)
        (int *, const char *, const char *, int *, int *,
         std::complex<double> *, int *, int *, int *);

      void STRUMPACK_FC_GLOBAL(sgsum2d,SGSUM2D)
        (int *, char *, char *, int *, int *,  float *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(dgsum2d,DGSUM2D)
        (int *, char *, char *, int *, int *, double *, int *, int *, int *);

      void STRUMPACK_FC_GLOBAL(sgamx2d,SGAMX2D)
        (int *, char *, char *, int *, int *, float *, int *,
         int *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(dgamx2d,DGAMX2D)
        (int *, char *, char *, int *, int *, double *, int *,
         int *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(cgamx2d,CGAMX2D)
        (int *, char *, char *, int *, int *,  std::complex<float> *,
         int *, int *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(zgamx2d,ZGAMX2D)
        (int *, char *, char *, int *, int *, std::complex<double> *,
         int *, int *, int *, int *, int *, int *);

      void STRUMPACK_FC_GLOBAL(sgamn2d,SGAMN2D)
        (int *, char *, char *, int *, int *, float *, int *,
         int *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(dgamn2d,DGAMN2D)
        (int *, char *, char *, int *, int *, double *, int *,
         int *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(cgamn2d,CGAMN2D)
        (int *, char *, char *, int *, int *, std::complex<float> *, int *,
         int *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(zgamn2d,ZGAMN2D)
        (int *, char *, char *, int *, int *, std::complex<double> *, int *,
         int *, int *, int *, int *, int *);

      void STRUMPACK_FC_GLOBAL(psamax,PSAMAX)
        (int *, float *, int *, float *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pdamax,PDAMAX)
        (int *, double *, int *, double *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pcamax,PCAMAX)
        (int *, std::complex<float> *, int *, std::complex<float> *,
         int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pzamax,PZAMAX)
        (int *, std::complex<double> *, int *, std::complex<double> *,
         int *, int *, int *, int *);

      void STRUMPACK_FC_GLOBAL(psswap,PSSWAP)
        (int *, float *, int *, int *, int *, int *,
         float *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pdswap,PDSWAP)
        (int *, double *, int *, int *, int *, int *,
         double *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pcswap,PCSWAP)
        (int *, std::complex<float> *, int *, int *, int *, int *,
         std::complex<float> *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pzswap,PZSWAP)
        (int *, std::complex<double> *, int *, int *, int *, int *,
         std::complex<double> *, int *, int *, int *, int *);

      void STRUMPACK_FC_GLOBAL(psscal,PSSCAL)
        (int *, float *, float *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pdscal,PDSCAL)
        (int *, double *, double *, int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pcscal,PCSCAL)
        (int *, std::complex<float> *, std::complex<float> *,
         int *, int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pzscal,PZSCAL)
        (int *, std::complex<double> *, std::complex<double> *,
         int *, int *, int *, int *);

      void STRUMPACK_FC_GLOBAL(psgemv,PSGEMV)
        (char *, int *, int *, float *,
         const float *, int *, int *, const int *,
         const float *, int *, int *, const int *, int *,
         float *, float *, int *,  int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pdgemv,PDGEMV)
        (char *, int *, int *, double *,
         const double *, int *, int *, const int *,
         const double *, int *, int *, const int *, int *,
         double *, double *, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pcgemv,PCGEMV)
        (char *, int *, int *, std::complex<float> *,
         const std::complex<float> *, int *, int *, const int *,
         const std::complex<float> *, int *, int *, const int *,
         int *, std::complex<float> *, std::complex<float> *,
         int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pzgemv,PZGEMV)
        (char *, int *, int *, std::complex<double> *,
         const std::complex<double> *, int *, int *, const int *,
         const std::complex<double> *, int *, int *, const int *,
         int *, std::complex<double> *, std::complex<double> *,
         int *, int *, const int *, int *);

      void STRUMPACK_FC_GLOBAL(psgemm,PSGEMM)
        (char *, char *, int *, int *, int *, float *, const float *,
         int *, int *, const int *, const float *, int *, int *, const int *,
         float *, float *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pdgemm,PDGEMM)
        (char *, char *, int *, int *, int *, double *, const double *,
         int *, int *, const int *, const double *, int *, int *, const int *,
         double *, double *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pcgemm,PCGEMM)
        (char *, char *, int *, int *, int *, std::complex<float> *,
         const std::complex<float> *, int *, int *, const int *,
         const std::complex<float> *, int *, int *, const int *,
         std::complex<float> *, std::complex<float> *,
         int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pzgemm,PZGEMM)
        (char *, char *, int *, int *, int *, std::complex<double> *,
         const std::complex<double> *, int *, int *, const int *,
         const std::complex<double> *, int *, int *, const int *,
         std::complex<double> *, std::complex<double> *,
         int *, int *, const int *);

      void STRUMPACK_FC_GLOBAL(pclacgv,PCLACGV)
        (int*, std::complex<float>* x, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pzlacgv,PZLACGV)
        (int*, std::complex<double>* x, int *, int *, const int *, int *);

      void STRUMPACK_FC_GLOBAL(psger,PSGER)
        (int *, int *, float *,
         const float *, int *, int *, const int *, int *,
         const float *, int *, int *, const int *, int *,
         float *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pdger,PDGER)
        (int *, int *, double *,
         const double *, int *, int *, const int *, int *,
         const double *, int *, int *, const int *, int *,
         double *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pcgeru,PCGERU)
        (int *, int *, std::complex<float> *,
         const std::complex<float> *, int *, int *, const int *, int *,
         const std::complex<float> *, int *, int *, const int *, int *,
         std::complex<float> *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pzgeru,PZGERU)
        (int *, int *, std::complex<double> *,
         const std::complex<double> *, int *, int *, const int *, int *,
         const std::complex<double> *, int *, int *, const int *, int *,
         std::complex<double> *, int *, int *, const int *);

      void STRUMPACK_FC_GLOBAL(pslaswp,PSLASWP)
        (char *, char *, int *,
         float *, int *, int *, const int *,
         int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pdlaswp,PDLASWP)
        (char *, char *, int *,
         double *, int *, int *, const int *,
         int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pclaswp,PCLASWP)
        (char *, char *, int *,
         std::complex<float> *, int *, int *, const int *,
         int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pzlaswp,PZLASWP)
        (char *, char *, int *,
         std::complex<double> *, int *, int *, const int *,
         int *, int *, const int *);

      void STRUMPACK_FC_GLOBAL(pslapiv,PSLAPIV)
        (char *, char *, char *, int *, int *,
         float *, int *, int *, const int *,
         const int *, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pdlapiv,PDLAPIV)
        (char *, char *, char *, int *, int *,
         double *, int *, int *, const int *,
         const int *, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pclapiv,PCLAPIV)
        (char *, char *, char *, int *, int *,
         std::complex<float> *, int *, int *, const int *,
         const int *, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pzlapiv,PZLAPIV)
        (char *, char *, char *, int *, int *,
         std::complex<double> *, int *, int *, const int *,
         const int *, int *, int *, const int *, int *);

      void STRUMPACK_FC_GLOBAL(pstrsm,PSTRSM)
        (char *, char *, char *, char *, int *, int *, float *,
         const float *, int *, int *, const int *,
         float *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pdtrsm,PDTRSM)
        (char *, char *, char *, char *, int *, int *, double *,
         const double *, int *, int *, const int *,
         double *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pctrsm,PCTRSM)
        (char *, char *, char *, char *, int *, int *, std::complex<float> *,
         const std::complex<float> *, int *, int *, const int *,
         std::complex<float> *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pztrsm,PZTRSM)
        (char *, char *, char *, char *, int *, int *, std::complex<double> *,
         const std::complex<double> *, int *, int *, const int *,
         std::complex<double> *, int *, int *, const int *);

      void STRUMPACK_FC_GLOBAL(pstrsv,PSTRSV)
        (char *, char *, char *, int *,
         const float *, int *, int *, const int *,
         float *, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pdtrsv,PDTRSV)
        (char *, char *, char *, int *,
         const double *, int *, int *, const int *,
         double *, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pctrsv,PCTRSV)
        (char *, char *, char *, int *,
         const std::complex<float> *, int *, int *, const int *,
         std::complex<float> *, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pztrsv,PZTRSV)
        (char *, char *, char *, int *,
         const std::complex<double> *, int *, int *, const int *,
         std::complex<double> *, int *, int *, const int *, int *);

      float STRUMPACK_FC_GLOBAL(pslange,PSLANGE)
        (char*, int*, int*, const float*, int*, int*, const int*, float*);
      double STRUMPACK_FC_GLOBAL(pdlange,PDLANGE)
        (char*, int*, int*, const double*, int*, int*, const int*, double*);
      float STRUMPACK_FC_GLOBAL(pclange,PCLANGE)
        (char*, int*, int*, const std::complex<float>*, int*, int*,
         const int*, float*);
      double STRUMPACK_FC_GLOBAL(pzlange,PZLANGE)
        (char*, int*, int*, const std::complex<double>*, int*, int*,
         const int*, double*);

      void STRUMPACK_FC_GLOBAL(psgeadd,PSGEADD)
        (char *, int *, int *, float *,
         const float *, int *, int *, const int *,
         float *, float *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pdgeadd,PDGEADD)
        (char *, int *, int *, double *,
         const double *, int *, int *, const int *,
         double *, double *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pcgeadd,PCGEADD)
        (char *, int *, int *, std::complex<float> *,
         const std::complex<float> *, int *, int *, const int *,
         std::complex<float> *,
         std::complex<float> *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pzgeadd,PZGEADD)
        (char *, int *, int *, std::complex<double> *,
         const std::complex<double> *, int *, int *, const int *,
         std::complex<double> *,
         std::complex<double> *, int *, int *, const int *);

      void STRUMPACK_FC_GLOBAL(pslacpy,PSLACPY)
        (char *, int *, int *,
         const float *, int *, int *, const int *,
         float *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pdlacpy,PDLACPY)
        (char *, int *, int *,
         const double *, int *, int *, const int *,
         double *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pclacpy,PCLACPY)
        (char *, int *, int *,
         const std::complex<float> *, int *, int *, const int *,
         std::complex<float> *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pzlacpy,PZLACPY)
        (char *, int *, int *,
         const std::complex<double> *, int *, int *, const int *,
         std::complex<double> *, int *, int *, const int *);

      void STRUMPACK_FC_GLOBAL(psgemr2d,PSGEMR2D)
        (int *, int *, const float *, int *, int *, const int *,
         float *, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pdgemr2d,PDGEMR2D)
        (int *, int *, const double *, int *, int *, const int *,
         double *, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pcgemr2d,PCGEMR2D)
        (int *, int *, const std::complex<float> *, int *, int *, const int *,
         std::complex<float> *, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pzgemr2d,PZGEMR2D)
        (int *, int *,
         const std::complex<double> *, int *, int *, const int *,
         std::complex<double> *, int *, int *, const int *, int *);

      void STRUMPACK_FC_GLOBAL(pstran,PSTRAN)
        (int *, int *, float *,
         const float *, int *, int *, const int *,
         float *, float *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pdtran,PDTRAN)
        (int *, int *, double *,
         const double *, int *, int *, const int *,
         double *, double *, int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pctranc,PCTRANC)
        (int *, int *, std::complex<float> *,
         const std::complex<float> *, int *, int *, const int *,
         std::complex<float> *, std::complex<float> *,
         int *, int *, const int *);
      void STRUMPACK_FC_GLOBAL(pztranc,PZTRANC)
        (int *, int *, std::complex<double> *,
         const std::complex<double> *, int *, int *, const int *,
         std::complex<double> *, std::complex<double> *,
         int *, int *, const int *);

      void STRUMPACK_FC_GLOBAL(psgeqpftol,PSGEQPFTOL)
        (int *, int *, float *, int *, int *, const int *, int *,
         float *, float *, int *, int *, int *, int *, int *,
         float *, float*);
      void STRUMPACK_FC_GLOBAL(pdgeqpftol,PDGEQPFTOL)
        (int *, int *, double *, int *, int *, const int *, int *,
         double *, double *, int *, int *, int *, int *, int *,
         double *, double *);
      void STRUMPACK_FC_GLOBAL(pcgeqpftol,PCGEQPFTOL)
        (int *, int *, std::complex<float> *, int *, int *, const int *,
         int *, std::complex<float> *, std::complex<float> *, int *, float *,
         int *, int *, int *, int *, int *, float *, float *);
      void STRUMPACK_FC_GLOBAL(pzgeqpftol,PZGEQPFTOL)
        (int *, int *, std::complex<double> *, int *, int *, const int *,
         int *, std::complex<double> *, std::complex<double> *, int *,
         double *, int *, int *, int *, int *, int *, double *, double *);

      void STRUMPACK_FC_GLOBAL(psgetrf,PSGETRF)
        (int *, int *, float *, int *, int *, const int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pdgetrf,PDGETRF)
        (int *, int *, double *, int *, int *, const int *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pcgetrf,PCGETRF)
        (int *, int *, std::complex<float> *, int *, int *, const int *,
         int *, int *);
      void STRUMPACK_FC_GLOBAL(pzgetrf,PZGETRF)
        (int *, int *, std::complex<double> *, int *, int *, const int *,
         int *, int *);

      void STRUMPACK_FC_GLOBAL(psgetrs,PSGETRS)
        (char *, int *, int *, const float *, int *, int *, const int *,
         const int *, float *, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pdgetrs,PDGETRS)
        (char *, int *, int *, const double *, int *, int *, const int *,
         const int *, double *, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pcgetrs,PCGETRS)
        (char *, int *, int *,
         const std::complex<float> *, int *, int *, const int *, const  int *,
         std::complex<float> *, int *, int *, const int *, int *);
      void STRUMPACK_FC_GLOBAL(pzgetrs,PZGETRS)
        (char *, int *, int *,
         const std::complex<double> *, int *, int *, const int *, const int *,
         std::complex<double> *, int *, int *, const int *, int *);

      void STRUMPACK_FC_GLOBAL(psgelqf,PSGELQF)
        (int *, int *, float *, int *, int *, const int *,
         float *, float *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pdgelqf,PDGELQF)
        (int *, int *, double *, int *, int *, const int *,
         double *, double *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pcgelqf,PCGELQF)
        (int *, int *, std::complex<float> *, int *, int *, const int *,
         std::complex<float> *, std::complex<float> *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pzgelqf,PZGELQF)
        (int *, int *, std::complex<double> *, int *, int *, const int *,
         std::complex<double> *, std::complex<double> *, int *, int *);

      void STRUMPACK_FC_GLOBAL(psorglq,PSORGLQ)
        (int *, int *, int *, float *, int *, int *, const int *,
         const float *, float *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pdorglq,PDORGLQ)
        (int *, int *, int *, double *, int *, int *, const int *,
         const double *, double *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pcunglq,PCUNGLQ)
        (int *, int *, int *,
         std::complex<float> *, int *, int *, const int *,
         const std::complex<float> *, std::complex<float> *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pzunglq,PZUNGLQ)
        (int *, int *, int *,
         std::complex<double> *, int *, int *, const int *,
         const std::complex<double> *, std::complex<double> *, int *, int *);

      void STRUMPACK_FC_GLOBAL(psgeqrf,PSGEQRF)
        (int *, int *, float *, int *, int *, const int *,
         float *, float *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pdgeqrf,PDGEQRF)
        (int *, int *, double *, int *, int *, const int *,
         double *, double *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pcgeqrf,PCGEQRF)
        (int *, int *, std::complex<float> *, int *, int *, const int *,
         std::complex<float> *, std::complex<float> *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pzgeqrf,PZGEQRF)
        (int *, int *, std::complex<double> *, int *, int *, const int *,
         std::complex<double> *, std::complex<double> *, int *, int *);

      void STRUMPACK_FC_GLOBAL(psorgqr,PSORGQR)
        (int *, int *, int *, float *, int *, int *, const int *,
         const float *, float *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pdorgqr,PDORGQR)
        (int *, int *, int *, double *, int *, int *, const int *,
         const double *, double *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pcungqr,PCUNGQR)
        (int *, int *, int *,
         std::complex<float> *, int *, int *, const int *,
         const std::complex<float> *, std::complex<float> *, int *, int *);
      void STRUMPACK_FC_GLOBAL(pzungqr,PZUNGQR)
        (int *, int *, int *,
         std::complex<double> *, int *, int *, const int *,
         const std::complex<double> *, std::complex<double> *, int *, int *);
    }

    int descinit
    (int* desc, int m, int n, int mb, int nb,
     int rsrc, int csrc, int ictxt, int mxllda) {
      int info;
      STRUMPACK_FC_GLOBAL(descinit,DESCINIT)
        (desc, &m, &n, &mb, &nb, &rsrc, &csrc, &ictxt, &mxllda, &info);
      return info;
    }

    void descset
    (int* desc, int m, int n, int mb, int nb,
     int rsrc, int csrc, int ictxt, int mxllda) {
      STRUMPACK_FC_GLOBAL(descset,DESCSET)
        (desc, &m, &n, &mb, &nb, &rsrc, &csrc, &ictxt, &mxllda);
    }

    int numroc
    (int n, int nb, int iproc, int isrcproc, int nprocs) {
      return STRUMPACK_FC_GLOBAL(numroc,NUMROC)
        (&n, &nb, &iproc, &isrcproc, &nprocs);
    }

    int infog1l
    (int GINDX, int NB, int NPROCS, int MYROC, int ISRCPROC, int& ROCSRC) {
      int LINDX;
      STRUMPACK_FC_GLOBAL(infog1l,INFOG1L)
        (&GINDX, &NB, &NPROCS, &MYROC, &ISRCPROC, &LINDX, &ROCSRC);
      return LINDX;
    }

    int infog1l
    (int GINDX, int NB, int NPROCS, int MYROC, int ISRCPROC) {
      int LINDX, ROCSRC;
      STRUMPACK_FC_GLOBAL(infog1l,INFOG1L)
        (&GINDX, &NB, &NPROCS, &MYROC, &ISRCPROC, &LINDX, &ROCSRC);
      return LINDX;
    }

    void infog2l
    (int GRINDX, int GCINDX, const int* DESC, int NPROW, int NPCOL, int MYROW,
     int MYCOL, int& LRINDX, int& LCINDX, int& RSRC, int& CSRC) {
      STRUMPACK_FC_GLOBAL(infog2l,INFOG2L)
        (&GRINDX, &GCINDX, DESC, &NPROW, &NPCOL, &MYROW, &MYCOL,
         &LRINDX, &LCINDX, &RSRC, &CSRC);
    }
    void infog2l
    (int GRINDX, int GCINDX, const int* DESC, int NPROW, int NPCOL, int MYROW,
     int MYCOL, int& LRINDX, int& LCINDX) {
      int RSRC, CSRC;
      STRUMPACK_FC_GLOBAL(infog2l,INFOG2L)
        (&GRINDX, &GCINDX, DESC, &NPROW, &NPCOL, &MYROW, &MYCOL,
         &LRINDX, &LCINDX, &RSRC, &CSRC);
    }

    void gebs2d
    (int ctxt, char scope, char top, int m, int n, int* a, int lda) {
      STRUMPACK_FC_GLOBAL(igebs2d,IGEBS2D)(&ctxt, &scope, &top, &m, &n, a, &lda);
    }
    void gebs2d
    (int ctxt, char scope, char top, int m, int n, float* a, int lda) {
      STRUMPACK_FC_GLOBAL(sgebs2d,SGEBS2D)(&ctxt, &scope, &top, &m, &n, a, &lda);
    }
    void gebs2d
    (int ctxt, char scope, char top, int m, int n, double* a, int lda) {
      STRUMPACK_FC_GLOBAL(dgebs2d,DGEBS2D)(&ctxt, &scope, &top, &m, &n, a, &lda);
    }
    void gebs2d
    (int ctxt, char scope, char top, int m, int n,
     std::complex<float>* a, int lda) {
      STRUMPACK_FC_GLOBAL(cgebs2d,CGEBS2D)(&ctxt, &scope, &top, &m, &n, a, &lda);
    }
    void gebs2d
    (int ctxt, char scope, char top, int m, int n,
     std::complex<double>* a, int lda) {
      STRUMPACK_FC_GLOBAL(zgebs2d,ZGEBS2D)(&ctxt, &scope, &top, &m, &n, a, &lda);
    }


    void gebr2d
    (int ctxt, char scope, char top, int m, int n,
     int* a, int lda, int rsrc, int csrc) {
      STRUMPACK_FC_GLOBAL(igebr2d,IGEBR2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda, &rsrc, &csrc);
    }
    void gebr2d
    (int ctxt, char scope, char top, int m, int n,
     float* a, int lda, int rsrc, int csrc) {
      STRUMPACK_FC_GLOBAL(sgebr2d,SGEBR2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda, &rsrc, &csrc);
    }
    void gebr2d
    (int ctxt, char scope, char top, int m, int n,
     double* a, int lda, int rsrc, int csrc) {
      STRUMPACK_FC_GLOBAL(dgebr2d,DGEBR2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda, &rsrc, &csrc);
    }
    void gebr2d
    (int ctxt, char scope, char top, int m, int n,
     std::complex<float>* a, int lda, int rsrc, int csrc) {
      STRUMPACK_FC_GLOBAL(cgebr2d,CGEBR2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda, &rsrc, &csrc);
    }
    void gebr2d
    (int ctxt, char scope, char top, int m, int n,
     std::complex<double>* a, int lda, int rsrc, int csrc) {
      STRUMPACK_FC_GLOBAL(zgebr2d,ZGEBR2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda, &rsrc, &csrc);
    }


    void gsum2d
    (int ctxt, char scope, char top, int m, int n,
     float* a, int lda, int rdest, int cdest) {
      STRUMPACK_FC_GLOBAL(sgsum2d,SGSUM2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda, &rdest, &cdest);
    }
    void gsum2d
    (int ctxt, char scope, char top, int m, int n,
     double* a, int lda, int rdest, int cdest) {
      STRUMPACK_FC_GLOBAL(dgsum2d,DGSUM2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda, &rdest, &cdest);
    }


    void gamx2d
    (int ctxt, char scope, char top, int m, int n, float* a, int lda,
     int *ra, int *ca, int ldia, int rdest, int cdest) {
      STRUMPACK_FC_GLOBAL(sgamx2d,SGAMX2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda,ra,ca, &ldia, &rdest, &cdest);
    }
    void gamx2d
    (int ctxt, char scope, char top, int m, int n, double* a, int lda,
     int *ra, int *ca, int ldia, int rdest, int cdest) {
      STRUMPACK_FC_GLOBAL(dgamx2d,DGAMX2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda,ra,ca, &ldia, &rdest, &cdest);
    }
    void gamx2d
    (int ctxt, char scope, char top, int m, int n,
     std::complex<float>* a, int lda, int *ra, int *ca,
     int ldia, int rdest, int cdest) {
      STRUMPACK_FC_GLOBAL(cgamx2d,CGAMX2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda,ra,ca, &ldia, &rdest, &cdest);
    }
    void gamx2d
    (int ctxt, char scope, char top, int m, int n,
     std::complex<double>* a, int lda, int *ra, int *ca,
     int ldia, int rdest, int cdest) {
      STRUMPACK_FC_GLOBAL(zgamx2d,ZGAMX2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda,ra,ca, &ldia, &rdest, &cdest);
    }

    void gamn2d
    (int ctxt, char scope, char top, int m, int n,  float* a, int lda,
     int *ra, int *ca, int ldia, int rdest, int cdest) {
      STRUMPACK_FC_GLOBAL(sgamn2d,SGAMN2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda,ra,ca, &ldia, &rdest, &cdest);
    }
    void gamn2d
    (int ctxt, char scope, char top, int m, int n, double* a, int lda,
     int *ra, int *ca, int ldia, int rdest, int cdest) {
      STRUMPACK_FC_GLOBAL(dgamn2d,DGAMN2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda,ra,ca, &ldia, &rdest, &cdest);
    }
    void gamn2d
    (int ctxt, char scope, char top, int m, int n,
     std::complex<float>* a, int lda, int *ra, int *ca,
     int ldia, int rdest, int cdest) {
      STRUMPACK_FC_GLOBAL(cgamn2d,CGAMN2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda,ra,ca, &ldia, &rdest, &cdest);
    }
    void gamn2d
    (int ctxt, char scope, char top, int m, int n,
     std::complex<double>* a, int lda, int *ra, int *ca,
     int ldia, int rdest, int cdest) {
      STRUMPACK_FC_GLOBAL(zgamn2d,ZGAMN2D)
        (&ctxt, &scope, &top, &m, &n, a, &lda,ra,ca, &ldia, &rdest, &cdest);
    }

    void pamax
    (int n, float *amax, int *indx, float* x, int ix,
     int jx, int *descx, int incx) {
      STRUMPACK_FC_GLOBAL(psamax,PSAMAX)
        (&n,amax,indx, x, &ix, &jx, descx, &incx);
    }
    void pamax
    (int n, double *amax, int *indx, double* x,
     int ix, int jx, int *descx, int incx) {
      STRUMPACK_FC_GLOBAL(pdamax,PDAMAX)
        (&n,amax,indx, x, &ix, &jx, descx, &incx);
    }
    void pamax
    (int n, std::complex<float> *amax, int *indx, std::complex<float>* x,
     int ix, int jx, int *descx, int incx) {
      STRUMPACK_FC_GLOBAL(pcamax,PCAMAX)
        (&n,amax,indx, x, &ix, &jx, descx, &incx);
    }
    void pamax
    (int n, std::complex<double> *amax, int *indx, std::complex<double>* x,
     int ix, int jx, int *descx, int incx) {
      STRUMPACK_FC_GLOBAL(pzamax,PZAMAX)
        (&n,amax,indx, x, &ix, &jx, descx, &incx);
    }

    void pswap
    (int n, float* x, int ix, int jx, int *descx, int incx,
     float* y, int iy, int jy, int *descy, int incy) {
      STRUMPACK_FC_GLOBAL(psswap,PSSWAP)
        (&n, x, &ix, &jx, descx, &incx, y, &iy, &jy, descy, &incy);
    }
    void pswap
    (int n, double* x, int ix, int jx, int *descx, int incx,
     double* y, int iy, int jy, int *descy, int incy) {
      STRUMPACK_FC_GLOBAL(pdswap,PDSWAP)
        (&n, x, &ix, &jx, descx, &incx, y, &iy, &jy, descy, &incy);
    }
    void pswap
    (int n, std::complex<float>* x, int ix, int jx, int *descx, int incx,
     std::complex<float>* y, int iy, int jy, int *descy, int incy) {
      STRUMPACK_FC_GLOBAL(pcswap,PCSWAP)
        (&n, x, &ix, &jx, descx, &incx, y, &iy, &jy, descy, &incy);
    }
    void pswap
    (int n, std::complex<double>* x, int ix, int jx, int *descx, int incx,
     std::complex<double>* y, int iy, int jy, int *descy, int incy) {
      STRUMPACK_FC_GLOBAL(pzswap,PZSWAP)
        (&n, x, &ix, &jx, descx, &incx, y, &iy, &jy, descy, &incy);
    }

    void pscal
    (int n, float a, float* x, int ix, int jx, int *descx, int incx) {
      STRUMPACK_FC_GLOBAL(psscal,PSSCAL)
        (&n, &a, x, &ix, &jx, descx, &incx);
    }
    void pscal
    (int n, double a, double* x, int ix, int jx, int *descx, int incx) {
      STRUMPACK_FC_GLOBAL(pdscal,PDSCAL)
        (&n, &a, x, &ix, &jx, descx, &incx);
    }
    void pscal
    (int n, std::complex<float> a, std::complex<float>* x,
     int ix, int jx, int *descx, int incx) {
      STRUMPACK_FC_GLOBAL
        (pcscal,PCSCAL)(&n, &a, x, &ix, &jx, descx, &incx);
    }
    void pscal
    (int n, std::complex<double> a, std::complex<double>* x,
     int ix, int jx, int *descx, int incx) {
      STRUMPACK_FC_GLOBAL(pzscal,PZSCAL)
        (&n, &a, x, &ix, &jx, descx, &incx);
    }


    void pgemv
    (char ta, int m, int n, float alpha,
     const float* a, int ia, int ja, const int *desca,
     const float* x, int ix, int jx, const int *descx, int incx, float beta,
     float* y, int iy, int jy, const int *descy, int incy) {
      STRUMPACK_FC_GLOBAL(psgemv,PSGEMV)
        (&ta, &m, &n, &alpha, a, &ia, &ja, desca,
         x, &ix, &jx, descx, &incx, &beta, y, &iy, &jy, descy, &incy);
    }
    void pgemv
    (char ta, int m, int n, double alpha,
     const double* a, int ia, int ja, const int *desca,
     const double* x, int ix, int jx, const int *descx, int incx, double beta,
     double* y, int iy, int jy, const int *descy, int incy) {
      STRUMPACK_FC_GLOBAL(pdgemv,PDGEMV)
        (&ta, &m, &n, &alpha, a, &ia, &ja, desca,
         x, &ix, &jx, descx, &incx, &beta, y, &iy, &jy, descy, &incy);
    }
    void pgemv
    (char ta, int m, int n, std::complex<float> alpha,
     const std::complex<float>* a, int ia, int ja, const int *desca,
     const std::complex<float>* x, int ix, int jx, const int *descx, int incx,
     std::complex<float> beta,
     std::complex<float>* y, int iy, int jy, const int *descy, int incy) {
      STRUMPACK_FC_GLOBAL(pcgemv,PCGEMV)
        (&ta, &m, &n, &alpha, a, &ia, &ja, desca,
         x, &ix, &jx, descx, &incx, &beta, y, &iy, &jy, descy, &incy);
    }
    void pgemv
    (char ta, int m, int n, std::complex<double> alpha,
     const std::complex<double>* a, int ia, int ja, const int *desca,
     const std::complex<double>* x, int ix, int jx, const int *descx, int incx,
     std::complex<double> beta,
     std::complex<double>* y, int iy, int jy, const int *descy, int incy) {
      STRUMPACK_FC_GLOBAL(pzgemv,PZGEMV)
        (&ta, &m, &n, &alpha, a, &ia, &ja, desca,
         x, &ix, &jx, descx, &incx, &beta, y, &iy, &jy, descy, &incy);
    }

    void pgemm
    (char ta, char tb, int m, int n, int k, float alpha,
     const float* a, int ia, int ja, const int *desca,
     const float* b, int ib, int jb, const int *descb, float beta,
     float *c, int ic, int jc, const int *descC) {
      STRUMPACK_FC_GLOBAL(psgemm,PSGEMM)
        (&ta, &tb, &m, &n, &k, &alpha, a, &ia, &ja, desca,
         b, &ib, &jb, descb, &beta, c, &ic, &jc, descC);
    }
    void pgemm
    (char ta, char tb, int m, int n, int k, double alpha,
     const double* a, int ia, int ja, const int *desca,
     const double* b, int ib, int jb, const int *descb, double beta,
     double *c, int ic, int jc, const int *descC) {
      STRUMPACK_FC_GLOBAL(pdgemm,PDGEMM)
        (&ta, &tb, &m, &n, &k, &alpha, a, &ia, &ja, desca,
         b, &ib, &jb, descb, &beta, c, &ic, &jc, descC);
    }
    void pgemm
    (char ta, char tb, int m, int n, int k, std::complex<float> alpha,
     const std::complex<float>* a, int ia, int ja, const int *desca,
     const std::complex<float>* b, int ib, int jb, const int *descb,
     std::complex<float> beta, std::complex<float> *c,
     int ic, int jc, const int *descC) {
      STRUMPACK_FC_GLOBAL(pcgemm,PCGEMM)
        (&ta, &tb, &m, &n, &k, &alpha, a, &ia, &ja, desca,
         b, &ib, &jb, descb, &beta, c, &ic, &jc, descC);
    }
    void pgemm
    (char ta, char tb, int m, int n, int k, std::complex<double> alpha,
     const std::complex<double>* a, int ia, int ja, const int *desca,
     const std::complex<double>* b, int ib, int jb, const int *descb,
     std::complex<double> beta, std::complex<double> *c,
     int ic, int jc, const int *descC) {
      STRUMPACK_FC_GLOBAL(pzgemm,PZGEMM)
        (&ta, &tb, &m, &n, &k, &alpha, a, &ia, &ja, desca,
         b, &ib, &jb, descb, &beta, c, &ic, &jc, descC);
    }

    void placgv
    (int n, float* x, int ix, int jx, const int *descx, int incx) {
      // Nothing to do
    }
    void placgv
    (int n, double* x, int ix, int jx, const int *descx, int incx) {
      // Nothing to do
    }
    void placgv
    (int n, std::complex<double>* x, int ix, int jx,
     const int *descx, int incx) {
      STRUMPACK_FC_GLOBAL(pzlacgv,PZLACGV)
        (&n, x, &ix, &jx, descx, &incx);
    }
    void placgv
    (int n, std::complex<float>* x, int ix, int jx,
     const int *descx, int incx) {
      STRUMPACK_FC_GLOBAL(pclacgv,PCLACGV)
        (&n, x, &ix, &jx, descx, &incx);
    }

    void pgeru
    (int m, int n, float alpha,
     const float* x, int ix, int jx, const int *descx, int incx,
     const float* y, int iy, int jy, const int *descy, int incy,
     float* a, int ia, int ja, const int *desca) {
      STRUMPACK_FC_GLOBAL(psger,PSGER)
        (&m, &n, &alpha, x, &ix, &jx, descx, &incx,
         y, &iy, &jy, descy, &incy, a, &ia, &ja, desca);
    }
    void pgeru
    (int m, int n, double alpha,
     const double* x, int ix, int jx, const int *descx, int incx,
     const double* y, int iy, int jy, const int *descy, int incy,
     double* a, int ia, int ja, const int *desca) {
      STRUMPACK_FC_GLOBAL(pdger,PDGER)
        (&m, &n, &alpha, x, &ix, &jx, descx, &incx,
         y, &iy, &jy, descy, &incy, a, &ia, &ja, desca);
    }
    void pgeru
    (int m, int n, std::complex<float> alpha,
     const std::complex<float>* x, int ix, int jx, const int *descx, int incx,
     const std::complex<float>* y, int iy, int jy, const int *descy, int incy,
     std::complex<float>* a, int ia, int ja, const int *desca) {
      STRUMPACK_FC_GLOBAL(pcgeru,PCGERU)
        (&m, &n, &alpha, x, &ix, &jx, descx, &incx,
         y, &iy, &jy, descy, &incy, a, &ia, &ja, desca);
    }
    void pgeru
    (int m, int n, std::complex<double> alpha,
     const std::complex<double>* x, int ix, int jx, const int *descx, int incx,
     const std::complex<double>* y, int iy, int jy, const int *descy, int incy,
     std::complex<double>* a, int ia, int ja, const int *desca) {
      STRUMPACK_FC_GLOBAL(pzgeru,PZGERU)
        (&m, &n, &alpha, x, &ix, &jx, descx, &incx,
         y, &iy, &jy, descy, &incy, a, &ia, &ja, desca);
    }

    void plaswp
    (char direc, char rowcol, int n,
     float* a, int ia, int ja, const int* desca,
     int k1, int k2, const int* ipiv) {
      STRUMPACK_FC_GLOBAL(pslaswp,PSLASWP)
        (&direc, &rowcol, &n,a, &ia, &ja, desca, &k1, &k2, ipiv);
    }
    void plaswp
    (char direc, char rowcol, int n,
     double* a, int ia, int ja, const int* desca,
     int k1, int k2, const int* ipiv) {
      STRUMPACK_FC_GLOBAL(pdlaswp,PDLASWP)
        (&direc, &rowcol, &n,a, &ia, &ja, desca, &k1, &k2, ipiv);
    }
    void plaswp
    (char direc, char rowcol, int n,
     std::complex<float>* a, int ia, int ja, const int* desca,
     int k1, int k2, const int* ipiv) {
      STRUMPACK_FC_GLOBAL(pclaswp,PCLASWP)
        (&direc, &rowcol, &n,a, &ia, &ja, desca, &k1, &k2, ipiv);
    }
    void plaswp
    (char direc, char rowcol, int n,
     std::complex<double>* a, int ia, int ja, const int* desca,
     int k1, int k2, const int* ipiv) {
      STRUMPACK_FC_GLOBAL(pzlaswp,PZLASWP)
        (&direc, &rowcol, &n,a, &ia, &ja, desca, &k1, &k2, ipiv);
    }

    void plapiv
    (char direc, char rowcol, char pivroc, int m, int n,
     float* a, int ia, int ja, const int* desca,
     const int* ipiv, int ip, int jp, const int* descip, int* iwork) {
      STRUMPACK_FC_GLOBAL(pslapiv,PSLAPIV)
        (&direc, &rowcol, &pivroc, &m, &n,
         a, &ia, &ja, desca, ipiv, &ip, &jp, descip, iwork);
    }
    void plapiv
    (char direc, char rowcol, char pivroc, int m, int n,
     double* a, int ia, int ja, const int* desca,
     const int* ipiv, int ip, int jp, const int* descip, int* iwork) {
      STRUMPACK_FC_GLOBAL(pdlapiv,PDLAPIV)
        (&direc, &rowcol, &pivroc, &m, &n,
         a, &ia, &ja, desca, ipiv, &ip, &jp, descip, iwork);
    }
    void plapiv
    (char direc, char rowcol, char pivroc, int m, int n,
     std::complex<float>* a, int ia, int ja, const int* desca,
     const int* ipiv, int ip, int jp, const int* descip, int* iwork) {
      STRUMPACK_FC_GLOBAL(pclapiv,PCLAPIV)
        (&direc, &rowcol, &pivroc, &m, &n,
         a, &ia, &ja, desca, ipiv, &ip, &jp, descip, iwork);
    }
    void plapiv
    (char direc, char rowcol, char pivroc, int m, int n,
     std::complex<double>* a, int ia, int ja, int* desca,
     const int* ipiv, int ip, int jp, const int* descip, int* iwork) {
      STRUMPACK_FC_GLOBAL(pzlapiv,PZLAPIV)
        (&direc, &rowcol, &pivroc, &m, &n,
         a, &ia, &ja, desca, ipiv, &ip, &jp, descip, iwork);
    }

    void ptrsm
    (char side, char uplo, char trans, char diag, int m, int n,
     float alpha, const float* a, int ia, int ja, const int *desca,
     float* b, int ib, int jb, const int *descb) {
      STRUMPACK_FC_GLOBAL(pstrsm,PSTRSM)
        (&side, &uplo, &trans, &diag, &m, &n, &alpha,
         a, &ia, &ja, desca, b, &ib, &jb, descb);
    }
    void ptrsm
    (char side, char uplo, char trans, char diag, int m, int n,
     double alpha, const double* a, int ia, int ja, const int *desca,
     double* b, int ib, int jb, const int *descb) {
      STRUMPACK_FC_GLOBAL(pdtrsm,PDTRSM)
        (&side, &uplo, &trans, &diag, &m, &n, &alpha,
         a, &ia, &ja, desca, b, &ib, &jb, descb);
    }
    void ptrsm
    (char side, char uplo, char trans, char diag, int m, int n,
     std::complex<float> alpha,
     const std::complex<float>* a, int ia, int ja, const int *desca,
     std::complex<float>* b, int ib, int jb, const int *descb) {
      STRUMPACK_FC_GLOBAL(pctrsm,PCTRSM)
        (&side, &uplo, &trans, &diag, &m, &n, &alpha,
         a, &ia, &ja, desca, b, &ib, &jb, descb);
    }
    void ptrsm
    (char side, char uplo, char trans, char diag, int m, int n,
     std::complex<double> alpha,
     const std::complex<double>* a, int ia, int ja, const int *desca,
     std::complex<double>* b, int ib, int jb, const int *descb) {
      STRUMPACK_FC_GLOBAL(pztrsm,PZTRSM)
        (&side, &uplo, &trans, &diag, &m, &n, &alpha,
         a, &ia, &ja, desca, b, &ib, &jb, descb);
    }

    void ptrsv
    (char uplo, char trans, char diag, int m,
     const float* a, int ia, int ja, const int *desca,
     float* b, int ib, int jb, const int *descb, int incb) {
      STRUMPACK_FC_GLOBAL(pstrsv,PSTRSV)
        (&uplo, &trans, &diag, &m, a, &ia, &ja, desca,
         b, &ib, &jb, descb, &incb);
    }
    void ptrsv
    (char uplo, char trans, char diag, int m,
     const double* a, int ia, int ja, const int *desca,
     double* b, int ib, int jb, int const *descb, int incb) {
      STRUMPACK_FC_GLOBAL(pdtrsv,PDTRSV)
        (&uplo, &trans, &diag, &m, a, &ia, &ja, desca,
         b, &ib, &jb, descb, &incb);
    }
    void ptrsv
    (char uplo, char trans, char diag, int m,
     const std::complex<float>* a, int ia, int ja, const int *desca,
     std::complex<float>* b, int ib, int jb, const int *descb, int incb) {
      STRUMPACK_FC_GLOBAL(pctrsv,PCTRSV)
        (&uplo, &trans, &diag, &m, a, &ia, &ja, desca,
         b, &ib, &jb, descb, &incb);
    }
    void ptrsv
    (char uplo, char trans, char diag, int m,
     const std::complex<double>* a, int ia, int ja, const int *desca,
     std::complex<double>* b, int ib, int jb, const int *descb, int incb) {
      STRUMPACK_FC_GLOBAL(pztrsv,PZTRSV)
        (&uplo, &trans, &diag, &m, a, &ia, &ja, desca,
         b, &ib, &jb, descb, &incb);
    }

    float plange
    (char norm, int m, int n, const float* a, int ia, int ja,
     const int *desca, float *work) {
      return STRUMPACK_FC_GLOBAL(pslange,PSLANGE)
        (&norm, &m, &n, a, &ia, &ja, desca, work);
    }
    double plange
    (char norm, int m, int n, const double* a, int ia, int ja,
     const int *desca, double *work) {
      return STRUMPACK_FC_GLOBAL(pdlange,PDLANGE)
        (&norm, &m, &n, a, &ia, &ja, desca, work);
    }
    float plange
    (char norm, int m, int n, const std::complex<float>* a, int ia, int ja,
     const int *desca, float *work) {
      return STRUMPACK_FC_GLOBAL(pclange,PCLANGE)
        (&norm, &m, &n, a, &ia, &ja, desca, work);
    }
    double plange
    (char norm, int m, int n, const std::complex<double>* a, int ia, int ja,
     const int *desca, double *work) {
      return STRUMPACK_FC_GLOBAL(pzlange,PZLANGE)
        (&norm, &m, &n, a, &ia, &ja, desca, work);
    }

    void pgeadd
    (char trans, int m, int n, float alpha,
     const float* a, int ia, int ja, int *desca, float beta,
     float *c, int ic, int jc, const int *descc) {
      STRUMPACK_FC_GLOBAL(psgeadd,PSGEADD)
        (&trans, &m, &n, &alpha, a, &ia, &ja, desca,
         &beta, c, &ic, &jc, descc);
    }
    void pgeadd
    (char trans, int m, int n, double alpha,
     const double* a, int ia, int ja, int *desca, double beta,
     double *c, int ic, int jc, const int *descC) {
      STRUMPACK_FC_GLOBAL(pdgeadd,PDGEADD)
        (&trans, &m, &n, &alpha, a, &ia, &ja, desca,
         &beta, c, &ic, &jc, descC);
    }
    void pgeadd
    (char trans, int m, int n, std::complex<float> alpha,
     const std::complex<float>* a, int ia, int ja, int *desca,
     std::complex<float> beta,
     std::complex<float> *c, int ic, int jc, const int *descC) {
      STRUMPACK_FC_GLOBAL(pcgeadd,PCGEADD)
        (&trans, &m, &n, &alpha, a, &ia, &ja, desca,
         &beta, c, &ic, &jc, descC);
    }
    void pgeadd
    (char trans, int m, int n, std::complex<double> alpha,
     const std::complex<double>* a, int ia, int ja, int *desca,
     std::complex<double> beta,
     std::complex<double> *c, int ic, int jc, const int *descC) {
      STRUMPACK_FC_GLOBAL(pzgeadd,PZGEADD)
        (&trans, &m, &n, &alpha, a, &ia, &ja, desca,
         &beta, c, &ic, &jc, descC);
    }

    void placpy
    (char trans, int m, int n,
     const float* a, int ia, int ja, const int *desca,
     float *c, int ic, int jc, const int *descc) {
      STRUMPACK_FC_GLOBAL(pslacpy,PSLACPY)
        (&trans, &m, &n, a, &ia, &ja, desca, c, &ic, &jc, descc);
    }
    void placpy
    (char trans, int m, int n,
     const double* a, int ia, int ja, const int *desca,
     double *c, int ic, int jc, const int *descc) {
      STRUMPACK_FC_GLOBAL(pdlacpy,PDLACPY)
        (&trans, &m, &n, a, &ia, &ja, desca, c, &ic, &jc, descc);
    }
    void placpy
    (char trans, int m, int n,
     const std::complex<float>* a, int ia, int ja, const int *desca,
     std::complex<float> *c, int ic, int jc, const int *descc) {
      STRUMPACK_FC_GLOBAL(pclacpy,PCLACPY)
        (&trans, &m, &n, a, &ia, &ja, desca, c, &ic, &jc, descc);
    }
    void placpy
    (char trans, int m, int n,
     const std::complex<double>* a, int ia, int ja, const int *desca,
     std::complex<double> *c, int ic, int jc, const int *descc) {
      STRUMPACK_FC_GLOBAL(pzlacpy,PZLACPY)
        (&trans, &m, &n, a, &ia, &ja, desca, c, &ic, &jc, descc);
    }

    void pgemr2d
    (int m, int n, const float* a, int ia, int ja, const int *desca,
     float* b, int ib, int jb, const int *descb, int ctxt) {
      STRUMPACK_FC_GLOBAL(psgemr2d,PSGEMR2D)
        (&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ctxt);
    }
    void pgemr2d
    (int m, int n, const double* a, int ia, int ja, const int *desca,
     double* b, int ib, int jb, const int *descb, int ctxt) {
      assert(desca[BLACSctxt]==-1 || m+ia-1 <= desca[BLACSm]);
      assert(descb[BLACSctxt]==-1 || m+ib-1 <= descb[BLACSm]);
      assert(desca[BLACSctxt]==-1 || n+ja-1 <= desca[BLACSn]);
      assert(descb[BLACSctxt]==-1 || n+jb-1 <= descb[BLACSn]);
      assert(ia >= 0 && ja >= 0 && ib >= 0 && jb >= 0);
      STRUMPACK_FC_GLOBAL(pdgemr2d,PDGEMR2D)
        (&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ctxt);
    }
    void pgemr2d
    (int m, int n,
     const std::complex<float>* a, int ia, int ja, const int *desca,
     std::complex<float>* b, int ib, int jb, const int *descb, int ctxt) {
      STRUMPACK_FC_GLOBAL(pcgemr2d,PCGEMR2D)
        (&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ctxt);
    }
    void pgemr2d
    (int m, int n,
     const std::complex<double>* a, int ia, int ja, const int *desca,
     std::complex<double>* b, int ib, int jb, const int *descb, int ctxt) {
      STRUMPACK_FC_GLOBAL(pzgemr2d,PZGEMR2D)
        (&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ctxt);
    }

    void ptranc
    (int m, int n, float alpha,
     const float* a, int ia, int ja, const int *desca,
     float beta, float *c, int ic, int jc, const int *descc) {
      STRUMPACK_FC_GLOBAL(pstran,PSTRAN)
        (&m, &n, &alpha, a, &ia, &ja, desca, &beta, c, &ic, &jc, descc);
    }
    void ptranc
    (int m, int n, double alpha,
     const double* a, int ia, int ja, const int *desca,
     double beta, double *c, int ic, int jc, const int *descc) {
      STRUMPACK_FC_GLOBAL(pdtran,PDTRAN)
        (&m, &n, &alpha, a, &ia, &ja, desca, &beta, c, &ic, &jc, descc);
    }
    void ptranc
    (int m, int n, std::complex<float> alpha,
     const std::complex<float>* a, int ia, int ja, const int *desca,
     std::complex<float> beta,
     std::complex<float> *c, int ic, int jc, const int *descc) {
      STRUMPACK_FC_GLOBAL(pctranc,PCTRANC)
        (&m, &n, &alpha, a, &ia, &ja, desca, &beta, c, &ic, &jc, descc);
    }
    void ptranc
    (int m, int n, std::complex<double> alpha,
     const std::complex<double>* a, int ia, int ja, const int *desca,
     std::complex<double> beta,
     std::complex<double> *c, int ic, int jc, const int *descc) {
      STRUMPACK_FC_GLOBAL(pztranc,PZTRANC)
        (&m, &n, &alpha, a, &ia, &ja, desca, &beta, c, &ic, &jc, descc);
    }

    void pgeqpftol
    (int m, int n, float* a, int ia, int ja, const int *desca,
     int *J, int *piv, int *r, float rtol, float atol) {
      int mb = desca[BLACSmb], nb = desca[BLACSnb];
      int info, prow, pcol, nprow, npcol;
      Cblacs_gridinfo(desca[BLACSctxt], &nprow, &npcol, &prow, &pcol);
      int locra = numroc(m, mb, prow, 0, nprow);
      int locca = numroc(n, nb, pcol, 0, npcol);
      int lwork = 3*(1+locra+locca);
      std::unique_ptr<float[]> twork(new float[lwork+locca]);
      auto tau = twork.get() + lwork;
      std::unique_ptr<int[]> ipiv(new int[n]);
      int IONE = 1;
      STRUMPACK_FC_GLOBAL(psgeqpftol,PSGEQPFTOL)
        (&m, &n, a, &IONE, &IONE, desca, ipiv.get(), tau,
         twork.get(), &lwork, &info, J, piv, r, &rtol, &atol);
    }
    void pgeqpftol
    (int m, int n, double* a, int ia, int ja, const int *desca,
     int *J, int *piv, int *r, double rtol, double atol) {
      int mb = desca[BLACSmb], nb = desca[BLACSnb];
      int info, prow, pcol, nprow, npcol;
      Cblacs_gridinfo(desca[BLACSctxt], &nprow, &npcol, &prow, &pcol);
      int locra = numroc(m, mb, prow, 0, nprow);
      int locca = numroc(n, nb, pcol, 0, npcol);
      int lwork = 3*(1+locra+locca);
      std::unique_ptr<double[]> twork(new double[lwork+locca]);
      auto tau = twork.get() + lwork;
      std::unique_ptr<int[]> ipiv(new int[n]);
      int IONE = 1;
      STRUMPACK_FC_GLOBAL(pdgeqpftol,PDGEQPFTOL)
        (&m, &n, a, &IONE, &IONE, desca, ipiv.get(), tau,
         twork.get(), &lwork, &info, J, piv, r, &rtol, &atol);
    }
    void pgeqpftol
    (int m, int n, std::complex<float>* a, int ia, int ja, const int *desca,
     int *J, int *piv, int *r, float rtol, float atol) {
      int mb = desca[BLACSmb], nb = desca[BLACSnb];
      int info, prow, pcol, nprow, npcol;
      Cblacs_gridinfo(desca[BLACSctxt], &nprow, &npcol, &prow, &pcol);
      int locra = numroc(m, mb, prow, 0, nprow);
      int locca = numroc(n, nb, pcol, 0, npcol);
      int lwork = 3*(1+locra+locca);
      std::unique_ptr<std::complex<float>[]> twork
        (new std::complex<float>[lwork+locca]);
      auto tau = twork.get() + lwork;
      int lrwork = 2*(1+locca);
      std::unique_ptr<float[]> rwork(new float[lrwork]);
      std::unique_ptr<int[]> ipiv(new int[n]);
      int IONE = 1;
      STRUMPACK_FC_GLOBAL(pcgeqpftol,PCGEQPFTOL)
        (&m, &n, a, &IONE, &IONE, desca, ipiv.get(), tau,
         twork.get(), &lwork, rwork.get(), &lrwork, &info, J, piv, r,
         &rtol, &atol);
    }
    void pgeqpftol
    (int m, int n, std::complex<double>* a, int ia, int ja, const int *desca,
     int *J, int *piv, int *r, double rtol, double atol) {
      int mb = desca[BLACSmb], nb = desca[BLACSnb];
      int info, prow, pcol, nprow, npcol;
      Cblacs_gridinfo(desca[BLACSctxt], &nprow, &npcol, &prow, &pcol);
      int locra = numroc(m, mb, prow, 0, nprow);
      int locca = numroc(n, nb, pcol, 0, npcol);
      int lwork = 3*(1+locra+locca);
      std::unique_ptr<std::complex<double>[]> twork
        (new std::complex<double>[lwork+locca]);
      auto tau = twork.get() + lwork;
      int lrwork = 2*(1+locca);
      std::unique_ptr<double[]> rwork(new double[lrwork]);
      std::unique_ptr<int[]> ipiv(new int[n]);
      int IONE = 1;
      STRUMPACK_FC_GLOBAL(pzgeqpftol,PZGEQPFTOL)
        (&m, &n, a, &IONE, &IONE, desca, ipiv.get(), tau,
         twork.get(), &lwork, rwork.get(), &lrwork, &info, J, piv, r,
         &rtol, &atol);
    }

    int pgetrf
    (int m, int n, float* a, int ia, int ja, const int *desca, int *ipiv) {
      int info;
      STRUMPACK_FC_GLOBAL(psgetrf,PSGETRF)
        (&m, &n, a, &ia, &ja, desca, ipiv, &info);
      return info;
    }
    int pgetrf
    (int m, int n, double* a, int ia, int ja, const int *desca, int *ipiv) {
      int info;
      STRUMPACK_FC_GLOBAL(pdgetrf,PDGETRF)
        (&m, &n, a, &ia, &ja, desca, ipiv, &info);
      return info;
    }
    int pgetrf
    (int m, int n, std::complex<float>* a, int ia, int ja,
     const int *desca, int *ipiv) {
      int info;
      STRUMPACK_FC_GLOBAL(pcgetrf,PCGETRF)
        (&m, &n, a, &ia, &ja, desca, ipiv, &info);
      return info;
    }
    int pgetrf
    (int m, int n, std::complex<double>* a, int ia, int ja,
     const int *desca, int *ipiv) {
      int info;
      STRUMPACK_FC_GLOBAL(pzgetrf,PZGETRF)
        (&m, &n, a, &ia, &ja, desca, ipiv, &info);
      return info;
    }

    int pgetrs
    (char trans, int m, int n,
     const float* a, int ia, int ja, const int *desca, const int *ipiv,
     float* b, int ib, int jb, const int *descb) {
      int info;
      STRUMPACK_FC_GLOBAL(psgetrs,PSGETRS)
        (&trans, &m, &n, a, &ia, &ja, desca, ipiv,
         b, &ib, &jb, descb, &info);
      return info;
    }
    int pgetrs
    (char trans, int m, int n,
     const double* a, int ia, int ja, const int *desca,
     const int *ipiv, double* b, int ib, int jb, const int *descb) {
      int info;
      STRUMPACK_FC_GLOBAL(pdgetrs,PDGETRS)
        (&trans, &m, &n, a, &ia, &ja, desca, ipiv,
         b, &ib, &jb, descb, &info);
      return info;
    }
    int pgetrs
    (char trans, int m, int n, const std::complex<float>* a,
     int ia, int ja, const int *desca, const int *ipiv,
     std::complex<float>* b, int ib, int jb, const int *descb) {
      int info;
      STRUMPACK_FC_GLOBAL(pcgetrs,PCGETRS)
        (&trans, &m, &n, a, &ia, &ja, desca, ipiv, b, &ib, &jb, descb, &info);
      return info;
    }
    int pgetrs
    (char trans, int m, int n, const std::complex<double>* a,
     int ia, int ja, const int *desca, const int *ipiv,
     std::complex<double>* b, int ib, int jb, const int *descb) {
      int info;
      STRUMPACK_FC_GLOBAL(pzgetrs,PZGETRS)
        (&trans, &m, &n, a, &ia, &ja, desca, ipiv, b, &ib, &jb, descb, &info);
      return info;
    }

    void pgelqf
    (int m, int n, float* a, int ia, int ja, const int *desca, float *tau) {
      int mb = desca[BLACSmb], nb = desca[BLACSnb];
      int info, prow, pcol, nprow, npcol;
      Cblacs_gridinfo(desca[BLACSctxt], &nprow, &npcol, &prow, &pcol);
      int locra = numroc(m, mb, prow, 0, nprow);
      int locca = numroc(n, nb, pcol, 0, npcol);
      int lwork = mb*(mb+locra+locca);
      std::unique_ptr<float[]> work(new float[lwork]);
      STRUMPACK_FC_GLOBAL(psgelqf,PSGELQF)
        (&m, &n, a, &ia, &ja, desca, tau, work.get(), &lwork, &info);
    }
    void pgelqf
    (int m, int n, double* a, int ia, int ja, const int *desca, double *tau) {
      int mb = desca[BLACSmb], nb = desca[BLACSnb];
      int info, prow, pcol, nprow, npcol;
      Cblacs_gridinfo(desca[BLACSctxt], &nprow, &npcol, &prow, &pcol);
      int locra = numroc(m, mb, prow, 0, nprow);
      int locca = numroc(n, nb, pcol, 0, npcol);
      int lwork = mb*(mb+locra+locca);
      std::unique_ptr<double[]> work(new double[lwork]);
      STRUMPACK_FC_GLOBAL(pdgelqf,PDGELQF)
        (&m, &n, a, &ia, &ja, desca, tau, work.get(), &lwork, &info);
    }
    void pgelqf
    (int m, int n, std::complex<float>* a, int ia, int ja, const int *desca,
     std::complex<float> *tau) {
      int mb = desca[BLACSmb], nb = desca[BLACSnb];
      int info, prow, pcol, nprow, npcol;
      Cblacs_gridinfo(desca[BLACSctxt], &nprow, &npcol, &prow, &pcol);
      int locra = numroc(m, mb, prow, 0, nprow);
      int locca = numroc(n, nb, pcol, 0, npcol);
      int lwork = mb*(mb+locra+locca);
      std::unique_ptr<std::complex<float>[]> work
        (new std::complex<float>[lwork]);
      STRUMPACK_FC_GLOBAL(pcgelqf,PCGELQF)
        (&m, &n, a, &ia, &ja, desca, tau, work.get(), &lwork, &info);
    }
    void pgelqf
    (int m, int n, std::complex<double>* a, int ia, int ja, const int *desca,
     std::complex<double> *tau) {
      int mb = desca[BLACSmb], nb = desca[BLACSnb];
      int info, prow, pcol, nprow, npcol;
      Cblacs_gridinfo(desca[BLACSctxt], &nprow, &npcol, &prow, &pcol);
      int locra = numroc(m, mb, prow, 0, nprow);
      int locca = numroc(n, nb, pcol, 0, npcol);
      int lwork = mb*(mb+locra+locca);
      std::unique_ptr<std::complex<double>[]> work
        (new std::complex<double>[lwork]);
      STRUMPACK_FC_GLOBAL(pzgelqf,PZGELQF)
        (&m, &n, a, &ia, &ja, desca, tau, work.get(), &lwork, &info);
    }

    void pxxglq
    (int m, int n, int k, float* a, int ia, int ja, const int *desca,
     const float *tau) {
      int mb = desca[BLACSmb], nb = desca[BLACSnb];
      int info, prow, pcol, nprow, npcol;
      Cblacs_gridinfo(desca[BLACSctxt], &nprow, &npcol, &prow, &pcol);
      int locra = numroc(m, mb, prow, 0, nprow);
      int locca = numroc(n, nb, pcol, 0, npcol);
      int lwork = mb*(mb+locra+locca);
      std::unique_ptr<float[]> work(new float[lwork]);
      STRUMPACK_FC_GLOBAL(psorglq,PSORGLQ)
        (&m, &n, &k, a, &ia, &ja, desca, tau, work.get(), &lwork, &info);
    }
    void pxxglq
    (int m, int n, int k, double* a, int ia, int ja, const int *desca,
     const double *tau) {
      int mb = desca[BLACSmb], nb = desca[BLACSnb];
      int info, prow, pcol, nprow, npcol;
      Cblacs_gridinfo(desca[BLACSctxt], &nprow, &npcol, &prow, &pcol);
      int locra = numroc(m, mb, prow, 0, nprow);
      int locca = numroc(n, nb, pcol, 0, npcol);
      int lwork = mb*(mb+locra+locca);
      std::unique_ptr<double[]> work(new double[lwork]);
      STRUMPACK_FC_GLOBAL(pdorglq,PDORGLQ)
        (&m, &n, &k, a, &ia, &ja, desca, tau, work.get(), &lwork, &info);
    }
    void pxxglq
    (int m, int n, int k,
     std::complex<float>* a, int ia, int ja, const int *desca,
     const std::complex<float> *tau) {
      int mb = desca[BLACSmb], nb = desca[BLACSnb];
      int info, prow, pcol, nprow, npcol;
      Cblacs_gridinfo(desca[BLACSctxt], &nprow, &npcol, &prow, &pcol);
      int locra = numroc(m, mb, prow, 0, nprow);
      int locca = numroc(n, nb, pcol, 0, npcol);
      int lwork = mb*(mb+locra+locca);
      std::unique_ptr<std::complex<float>[]> work
        (new std::complex<float>[lwork]);
      STRUMPACK_FC_GLOBAL(pcunglq,PCUNGLQ)
        (&m, &n, &k, a, &ia, &ja, desca, tau, work.get(), &lwork, &info);
    }
    void pxxglq
    (int m, int n, int k,
     std::complex<double>* a, int ia, int ja, const int *desca,
     const std::complex<double> *tau) {
      int mb = desca[BLACSmb], nb = desca[BLACSnb];
      int info, prow, pcol, nprow, npcol;
      Cblacs_gridinfo(desca[BLACSctxt], &nprow, &npcol, &prow, &pcol);
      int locra = numroc(m, mb, prow, 0, nprow);
      int locca = numroc(n, nb, pcol, 0, npcol);
      int lwork = mb*(mb+locra+locca);
      std::unique_ptr<std::complex<double>[]> work
        (new std::complex<double>[lwork]);
      STRUMPACK_FC_GLOBAL(pzunglq,PZUNGLQ)
        (&m, &n, &k, a, &ia, &ja, desca, tau, work.get(), &lwork, &info);
    }

    int pgeqrf
    (int m, int n, float* a, int ia, int ja, const int *desca,
     float *tau, float* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(psgeqrf,PSGEQRF)
        (&m, &n, a, &ia, &ja, desca, tau, work, &lwork, &info);
      return info;
    }
    int pgeqrf
    (int m, int n, double* a, int ia, int ja, const int *desca,
     double *tau, double* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(pdgeqrf,PDGEQRF)
        (&m, &n, a, &ia, &ja, desca, tau, work, &lwork, &info);
      return info;
    }
    int pgeqrf
    (int m, int n, std::complex<float>* a, int ia, int ja, const int *desca,
     std::complex<float> *tau, std::complex<float>* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(pcgeqrf,PCGEQRF)
        (&m, &n, a, &ia, &ja, desca, tau, work, &lwork, &info);
      return info;
    }
    int pgeqrf
    (int m, int n, std::complex<double>* a, int ia, int ja, const int *desca,
     std::complex<double> *tau, std::complex<double>* work, int lwork) {
      int info;
      STRUMPACK_FC_GLOBAL(pzgeqrf,PZGEQRF)
        (&m, &n, a, &ia, &ja, desca, tau, work, &lwork, &info);
      return info;
    }

    int pxxgqr
    (int m, int n, int k, float* a, int ia, int ja,
     const int* desca, const float* tau, float* work, int lwork) {
      int info;
      char R = 'R', C = 'C', B = 'B', rowbtop, colbtop;
      STRUMPACK_FC_GLOBAL(pb_topget,PB_TOPGET)(&desca[BLACSctxt], &B, &R, &rowbtop);
      STRUMPACK_FC_GLOBAL(pb_topget,PB_TOPGET)(&desca[BLACSctxt], &B, &C, &colbtop);
      STRUMPACK_FC_GLOBAL(psorgqr,PSORGQR)
        (&m, &n, &k, a, &ia, &ja, desca, tau, work, &lwork, &info);
      STRUMPACK_FC_GLOBAL(pb_topset,PB_TOPSET)(&desca[BLACSctxt], &B, &R, &rowbtop);
      STRUMPACK_FC_GLOBAL(pb_topset,PB_TOPSET)(&desca[BLACSctxt], &B, &C, &colbtop);
      return info;
    }
    int pxxgqr
    (int m, int n, int k, double* a, int ia, int ja,
     const int* desca, const double* tau, double* work, int lwork) {
      int info;
      // workaround for ScaLAPACK bug:
      //   http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=2&t=4510
      char R = 'R', C = 'C', B = 'B', rowbtop, colbtop;
      STRUMPACK_FC_GLOBAL(pb_topget,PB_TOPGET)(&desca[BLACSctxt], &B, &R, &rowbtop);
      STRUMPACK_FC_GLOBAL(pb_topget,PB_TOPGET)(&desca[BLACSctxt], &B, &C, &colbtop);
      STRUMPACK_FC_GLOBAL(pdorgqr,PDORGQR)
        (&m, &n, &k, a, &ia, &ja, desca, tau, work, &lwork, &info);
      STRUMPACK_FC_GLOBAL(pb_topset,PB_TOPSET)(&desca[BLACSctxt], &B, &R, &rowbtop);
      STRUMPACK_FC_GLOBAL(pb_topset,PB_TOPSET)(&desca[BLACSctxt], &B, &C, &colbtop);
      return info;
    }
    int pxxgqr
    (int m, int n, int k, std::complex<float>* a, int ia, int ja,
     const int* desca, const std::complex<float>* tau,
     std::complex<float>* work, int lwork) {
      int info;
      char R = 'R', C = 'C', B = 'B', rowbtop, colbtop;
      STRUMPACK_FC_GLOBAL(pb_topget,PB_TOPGET)(&desca[BLACSctxt], &B, &R, &rowbtop);
      STRUMPACK_FC_GLOBAL(pb_topget,PB_TOPGET)(&desca[BLACSctxt], &B, &C, &colbtop);
      STRUMPACK_FC_GLOBAL(pcungqr,PCUNGQR)
        (&m, &n, &k, a, &ia, &ja, desca, tau, work, &lwork, &info);
      STRUMPACK_FC_GLOBAL(pb_topset,PB_TOPSET)(&desca[BLACSctxt], &B, &R, &rowbtop);
      STRUMPACK_FC_GLOBAL(pb_topset,PB_TOPSET)(&desca[BLACSctxt], &B, &C, &colbtop);
      return info;
    }
    int pxxgqr
    (int m, int n, int k, std::complex<double>* a, int ia, int ja,
     const int* desca, const std::complex<double>* tau,
     std::complex<double>* work, int lwork) {
      int info;
      char R = 'R', C = 'C', B = 'B', rowbtop, colbtop;
      STRUMPACK_FC_GLOBAL(pb_topget,PB_TOPGET)(&desca[BLACSctxt], &B, &R, &rowbtop);
      STRUMPACK_FC_GLOBAL(pb_topget,PB_TOPGET)(&desca[BLACSctxt], &B, &C, &colbtop);
      STRUMPACK_FC_GLOBAL(pzungqr,PZUNGQR)
        (&m, &n, &k, a, &ia, &ja, desca, tau, work, &lwork, &info);
      STRUMPACK_FC_GLOBAL(pb_topset,PB_TOPSET)(&desca[BLACSctxt], &B, &R, &rowbtop);
      STRUMPACK_FC_GLOBAL(pb_topset,PB_TOPSET)(&desca[BLACSctxt], &B, &C, &colbtop);
      return info;
    }

    char topget(int ctxt, char B, char R) {
      char ret;
      STRUMPACK_FC_GLOBAL(pb_topget,PB_TOPGET)(&ctxt, &B, &R, &ret);
      return ret;
    }

  } // end namespace scalapack
} // end namespace strumpack
