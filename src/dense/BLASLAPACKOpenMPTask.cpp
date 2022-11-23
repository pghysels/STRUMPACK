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
#include "StrumpackFortranCInterface.h"

namespace strumpack {

  // TODO put in a namespace?
  const int OMPTileSize = 64;
  const int OMPThreshold = OMPTileSize*OMPTileSize*OMPTileSize;
  const int gemmOMPThreshold = OMPThreshold;
  const int trsmOMPThreshold = OMPThreshold;

  template<typename scalar>
  void gemm_omp_task(char ta, char tb, int m, int n, int k, scalar alpha,
                     const scalar* a, int lda, const scalar* b, int ldb,
                     scalar beta, scalar* c, int ldc, int depth) {
    if (depth>=params::task_recursion_cutoff_level ||
        double(m)*n*k <= gemmOMPThreshold)
      blas::gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    else {
      bool opA = ta=='T'||ta=='t'||ta=='C'||ta=='c';
      bool opB = tb=='T'||tb=='t'||tb=='C'||tb=='c';
      if (n >= std::max(m,k)) {
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
        gemm_omp_task
          (ta, tb, m, n/2, k, alpha, a, lda, b, ldb, beta, c, ldc, depth+1);
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
        gemm_omp_task
          (ta, tb, m, n-n/2, k, alpha, a, lda, opB ? b+n/2 : b+(n/2)*ldb, ldb,
           beta, c+(n/2)*ldc, ldc, depth+1);
#pragma omp taskwait
      } else if (m >= k) {
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
        gemm_omp_task
          (ta, tb, m/2, n, k, alpha, a, lda, b, ldb, beta, c, ldc, depth+1);
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
        gemm_omp_task
          (ta, tb, m-m/2, n, k, alpha, opA ? a+(m/2)*lda : a+m/2, lda, b, ldb,
           beta, c+m/2, ldc, depth+1);
#pragma omp taskwait
      } else {
        gemm_omp_task
          (ta, tb, m, n, k/2, alpha, a, lda, b, ldb, beta, c, ldc, depth);
        gemm_omp_task
          (ta, tb, m, n, k-k/2, alpha, opA ? a+k/2 : a+(k/2)*lda, lda,
           opB ? b+(k/2)*ldb : b+k/2, ldb, scalar(1.), c, ldc, depth);
      }
    }
  }

  template<typename scalar>
  void gemv_omp_task(char t, int m, int n, scalar alpha,
                     const scalar *a, int lda,
                     const scalar *x, int incx, scalar beta,
                     scalar *y, int incy, int depth) {
    if (depth>=params::task_recursion_cutoff_level ||
        2*double(m)*n <= OMPTileSize*OMPTileSize/*OMPThreshold*/)
      blas::gemv(t, m, n, alpha, a, lda, x, incx, beta, y, incy);
    else {
      if (t=='T' || t=='t' || t=='C' || t=='c') {
        if (n >= m) {
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          gemv_omp_task
            (t, m, n/2, alpha, a, lda, x, incx, beta, y, incy, depth+1);
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          gemv_omp_task
            (t, m, n-n/2, alpha, a+(n/2)*lda, lda, x, incx,
             beta, y+(n/2)*incy, incy, depth+1);
#pragma omp taskwait
        } else {
          if (n <= /*OMPTileSize*/64) {
            scalar tmp[/*OMPTileSize*/64];
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
            gemv_omp_task
              (t, m/2, n, alpha, a, lda, x, incx, beta,
               y, incy, depth+1);
#pragma omp task shared(tmp)                                            \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
            gemv_omp_task
              (t, m-m/2, n, alpha, a+m/2, lda, x+(m/2)*incx, incx,
               scalar(0.), tmp, 1, depth+1);
#pragma omp taskwait
            for (int i=0; i<n; i++) y[i*incy] += tmp[i];
          } else {
            gemv_omp_task
              (t, m/2, n, alpha, a, lda, x, incx, beta, y, incy, depth);
            gemv_omp_task
              (t, m-m/2, n, alpha, a+m/2, lda, x+(m/2)*incx, incx,
               scalar(1.), y, incy, depth);
          }
        }
      } else if (t=='N' || t=='n') {
        if (m >= n) {
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          gemv_omp_task
            (t, m/2, n, alpha, a, lda, x, incx, beta, y, incy, depth+1);
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          gemv_omp_task
            (t, m-m/2, n, alpha, a+m/2, lda, x, incx, beta,
             y+(m/2)*incy, incy, depth+1);
#pragma omp taskwait
        } else {
          if (m <= /*OMPTileSize*/64) {
            scalar tmp[/*OMPTileSize*/64];
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
            gemv_omp_task
              (t, m, n/2, alpha, a, lda, x, incx, beta, y, incy, depth+1);
#pragma omp task shared(tmp)                                            \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
            gemv_omp_task
              (t, m, n-n/2, alpha, a+(n/2)*lda, lda,
               x+(n/2)*incx, incx, scalar(0.), tmp, 1, depth+1);
#pragma omp taskwait
            for (int i=0; i<m; i++) y[i*incy] += tmp[i];
          } else {
            gemv_omp_task
              (t, m, n/2, alpha, a, lda, x, incx, beta, y, incy, depth);
            gemv_omp_task
              (t, m, n-n/2, alpha, a+(n/2)*lda, lda, x+(n/2)*incx, incx,
               scalar(1.), y, incy, depth);
          }
        }
      }
    }
  }

  // TODO test this code, or replace with SLATE
#if 0 //_OPENMP >= 201307
  //#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
  template<typename scalar>
  void trsv_omp_task(char ul, char ta, char d, int n, const scalar* a, int lda,
                     scalar* x, int incx, int depth) {
    if (depth>=params::task_recursion_cutoff_level ||
        double(n) <= OMPTileSize)
      blas::trsv(ul, ta, d, n, a, lda, x, incx);
    else {
      if ((ul=='L' || ul=='l') && (ta=='N' || ta=='n')) {
#pragma omp taskgroup
        {
          for (int i=0; i<n; i+=OMPTileSize) {
#pragma omp task firstprivate(i) default(shared) depend(inout:x[i])
            blas::trsv
              (ul, ta, d, std::min(OMPTileSize, n-i),
               a+i+i*lda, lda, x+i*incx, incx);
            for (int j=i+OMPTileSize; j<n; j+=OMPTileSize)
#pragma omp task firstprivate(i,j) default(shared) depend(in:x[i])      \
  depend(inout:x[j])
              blas::gemv
                (ta, std::min(OMPTileSize, n-j), OMPTileSize, scalar(-1.),
                 a+j+i*lda, lda, x+i*incx, 1, scalar(1.), x+j*incx, incx);
          }
        }
      } else if ((ul=='U' || ul=='u') && (ta=='N' || ta=='n')) {
#pragma omp taskgroup
        {
          for (int i=n-n%OMPTileSize; i>=0; i-=OMPTileSize) {
#pragma omp task firstprivate(i) default(shared) depend(inout:x[i])
            blas::trsv
              (ul, ta, d, std::min(OMPTileSize, n-i), a+i+i*lda, lda,
               x+i*incx, incx);
            for (int j=i-OMPTileSize; j>=0; j-=OMPTileSize)
#pragma omp task firstprivate(i,j) default(shared) depend(in:x[i])      \
  depend(inout:x[j])
              blas::gemv
                (ta, OMPTileSize, std::min(OMPTileSize, n-i), scalar(-1.),
                 a+j+i*lda, lda, x+i*incx, 1, scalar(1.), x+j*incx, incx);
          }
        }
      } else {
        std::cerr << "trsv_omp_task not implemented with this combination of"
                  << " side, uplo and transpose" << std::endl;
        abort();
      }
    }
  }

#else

  template<typename scalar>
  void trsv_omp_task(char ul, char ta, char d, int n, const scalar* a, int lda,
                     scalar* x, int incx, int depth) {
    if (depth>=params::task_recursion_cutoff_level ||
        double(n)*n <= OMPTileSize*OMPTileSize/*OMPThreshold*/)
      blas::trsv(ul, ta, d, n, a, lda, x, incx);
    else {
      if ((ul=='L' || ul=='l') && (ta=='N' || ta=='n')) {
        trsv_omp_task(ul, ta, d, n/2, a, lda, x, incx, depth);
        gemv_omp_task
          (ta, n-n/2, n/2, scalar(-1.), a+n/2, lda, x, incx,
           scalar(1.), x+(n/2)*incx, incx, depth);
        trsv_omp_task
          (ul, ta, d, n-n/2, a+n/2+(n/2)*lda, lda,
           x+(n/2)*incx, incx, depth);
      } else if ((ul=='U' || ul=='u') && (ta=='N' || ta=='n')) {
        trsv_omp_task
          (ul, ta, d, n-n/2, a+n/2+(n/2)*lda, lda, x+(n/2)*incx, incx, depth);
        gemv_omp_task
          (ta, n/2, n-n/2, scalar(-1.), a+n/2*lda, lda, x+(n/2)*incx, incx,
           scalar(1.), x, incx, depth);
        trsv_omp_task(ul, ta, d, n/2, a, lda, x, incx, depth);
      } else {
        std::cerr << "trsv_omp_task not implemented with this combination of"
                  << " side, uplo and transpose" << std::endl;
        abort();
      }
    }
  }
#endif


  template<typename scalar>
  void trmm_omp_task(char s, char ul, char ta, char d, int m, int n,
                     scalar alpha, const scalar* a, int lda,
                     scalar* b, int ldb, int depth) {
    if (m == 0 || n == 0) return;
    if (depth>=params::task_recursion_cutoff_level ||
        double(m)*n*n <= OMPThreshold)
      blas::trmm(s, ul, ta, d, m, n, alpha, a, lda, b, ldb);
    else {
      if (s == 'L' || s == 'l') {
        if (n >= m) {
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          trmm_omp_task
            (s, ul, ta, d, m, n/2, alpha, a, lda, b, ldb, depth+1);
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          trmm_omp_task
            (s, ul, ta, d, m, n-n/2, alpha, a, lda,
             b+(n/2)*ldb, ldb, depth+1);
#pragma omp taskwait
        } else {
          bool opA = ta=='T'||ta=='t'||ta=='C'||ta=='c';
          if (ul == 'U' || ul == 'u') {
            if (opA) {
              trmm_omp_task
                (s, ul, ta, d, m-m/2, m-m/2, alpha,
                 a+m/2+(m/2)*lda, lda, b+m/2, ldb, depth);
              gemm_omp_task
                (ta, 'N', m-m/2, n, m/2, alpha, a+(m/2)*lda, lda,
                 b, ldb, scalar(1.), b+m/2, ldb, depth);
              trmm_omp_task
                (s, ul, ta, d, m/2, m/2, alpha, a, lda, b, ldb, depth);
            } else {
              trmm_omp_task
                (s, ul, ta, d, m/2, m/2, alpha, a, lda, b, ldb, depth);
              gemm_omp_task
                (ta, 'N', m/2, n, m-m/2, alpha, a+(m/2)*lda, lda,
                 b+m/2, ldb, scalar(1.), b, ldb, depth);
              trmm_omp_task
                (s, ul, ta, d, m-m/2, m-m/2, alpha,
                 a+m/2+(m/2)*lda, lda, b+m/2, ldb, depth);
            }
          } else {
            if (opA) {
              trmm_omp_task
                (s, ul, ta, d, m/2, m/2, alpha, a, lda, b, ldb, depth);
              gemm_omp_task
                (ta, 'N', m/2, n, m-m/2, alpha, a+m/2, lda,
                 b+m/2, ldb, scalar(1.), b, ldb, depth);
              trmm_omp_task
                (s, ul, ta, d, m-m/2, m-m/2, alpha,
                 a+m/2+(m/2)*lda, lda, b+m/2, ldb, depth);
            } else {
              trmm_omp_task
                (s, ul, ta, d, m-m/2, m-m/2, alpha,
                 a+m/2+(m/2)*lda, lda, b+m/2, ldb, depth);
              gemm_omp_task
                (ta, 'N', m-m/2, n, m/2, alpha, a+m/2, lda,
                 b, ldb, scalar(1.), b+m/2, ldb, depth);
              trmm_omp_task
                (s, ul, ta, d, m/2, m/2, alpha, a, lda, b, ldb, depth);
            }
          }
        }
      } else { // s == R/r
        if (n >= m) {
          bool opA = ta=='T'||ta=='t'||ta=='C'||ta=='c';
          if (ul == 'U' || ul == 'u') {
            if (opA) {
              trmm_omp_task
                (s, ul, ta, d, m, n/2, alpha, a, lda, b, ldb, depth);
              gemm_omp_task
                ('N', ta, m, n/2, n-n/2, alpha, b+(n/2)*ldb, ldb,
                 a+(n/2)*lda, lda, scalar(1.), b, ldb, depth);
              trmm_omp_task
                (s, ul, ta, d, m, n-n/2, alpha, a+n/2+(n/2)*lda, lda,
                 b+(n/2)*ldb, ldb, depth);
            } else {
              trmm_omp_task
                (s, ul, ta, d, m, n-n/2, alpha,
                 a+n/2+(n/2)*lda, lda, b+(n/2)*ldb, ldb, depth);
              gemm_omp_task
                ('N', ta, m, n-n/2, n/2, alpha, b, ldb,
                 a+(n/2)*lda, lda, scalar(1.), b+(n/2)*ldb, ldb, depth);
              trmm_omp_task
                (s, ul, ta, d, m, n/2, alpha, a, lda, b, ldb, depth);
            }
          } else {
            if (opA) {
              trmm_omp_task
                (s, ul, ta, d, m, n-n/2, alpha,
                 a+n/2+(n/2)*lda, lda, b+(n/2)*ldb, ldb, depth);
              gemm_omp_task
                ('N', ta, m, n-n/2, n/2, alpha, b, ldb, a+n/2, lda,
                 scalar(1.), b+(n/2)*ldb, ldb, depth);
              trmm_omp_task
                (s, ul, ta, d, m, n/2, alpha, a, lda, b, ldb, depth);
            } else {
              trmm_omp_task
                (s, ul, ta, d, m, n/2, alpha, a, lda, b, ldb, depth);
              gemm_omp_task
                ('N', ta, m, n/2, n-n/2, alpha, b+(n/2)*ldb, ldb,
                 a+n/2, lda, scalar(1.), b, ldb, depth);
              trmm_omp_task
                (s, ul, ta, d, m, n-n/2, alpha,
                 a+n/2+(n/2)*lda, lda, b+(n/2)*ldb, ldb, depth);
            }
          }
        } else {
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          trmm_omp_task
            (s, ul, ta, d, m/2, n, alpha, a, lda, b, ldb, depth+1);
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          trmm_omp_task
            (s, ul, ta, d, m-m/2, n, alpha, a, lda, b+m/2, ldb, depth+1);
#pragma omp taskwait
        }
      }
    }
  }

  // it seems this is never called?! so optimized code is not tested
  template<typename scalar>
  void trmv_omp_task(char ul, char ta, char d, int n, const scalar* a, int lda,
                     scalar* x, int incx, int depth) {
    blas::trmv(ul, ta, d, n, a, lda, x, incx);

    // if (incx == 1) trmm_omp_task('L', ul, ta, d, n, 1, scalar(1.), A, lda, x, n, depth);
    // else trmv(ul, ta, d, n, A, lda, x, incx);

    // if (n == 0) return;
    // if (depth>=params::task_recursion_cutoff_level || double(n)*n <= OMPTileSize*OMPTileSize)
    //   trmv(ul, ta, d, n, A, lda, x, incx);
    // else {
    //   std::cout << "hello from trmv!!!" << std::endl;
    //   if (ul == 'U' || ul == 'u') {
    //     trmv_omp_task(ul, ta, d, n/2, A, lda, x, incx, depth);
    //     if (ta=='T'||ta=='t'||ta=='C'||ta=='c')
    //          gemv_omp_task(ta, n-n/2, n/2, scalar(1.), A+(n/2)*lda, lda, x+(n/2)*incx, incx, scalar(1.), x, incx, depth);
    //     else gemv_omp_task(ta, n/2, n-n/2, scalar(1.), A+(n/2)*lda, lda, x+(n/2)*incx, incx, scalar(1.), x, incx, depth);
    //     trmv_omp_task(ul, ta, d, n-n/2, A+n/2+(n/2)*lda, lda, x+(n/2)*incx, incx, depth);
    //   } else {
    //     trmv_omp_task(ul, ta, d, n-n/2, A+n/2+(n/2)*lda, lda, x+(n/2)*incx, incx, depth);
    //     if (ta=='T'||ta=='t'||ta=='C'||ta=='c')
    //          gemv_omp_task(ta, n/2, n-n/2, scalar(1.), A+n/2, lda, x, incx, scalar(1.), x+(n/2)*incx, incx, depth);
    //     else gemv_omp_task(ta, n-n/2, n/2, scalar(1.), A+n/2, lda, x, incx, scalar(1.), x+(n/2)*incx, incx, depth);
    //     trmv_omp_task(ul, ta, d, n/2, A, lda, x, incx, depth);
    //   }
    // }
  }

  template<typename scalar>
  void trsm_omp_task(char s, char ul, char ta, char d, int m, int n,
                     scalar alpha, const scalar* a, int lda,
                     scalar* b, int ldb, int depth) {
    if (double(m)*m*n <= trsmOMPThreshold ||
        depth>=params::task_recursion_cutoff_level)
      blas::trsm(s, ul, ta, d, m, n, alpha, a, lda, b, ldb);
    else {
      if ((s=='L' || s=='l') && (ul=='L' || ul=='l') &&
          (ta=='N' || ta=='n')) {
        if (n >= m) {
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          trsm_omp_task
            (s, ul, ta, d, m, n/2, alpha, a, lda, b, ldb, depth+1);
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          trsm_omp_task
            (s, ul, ta, d, m, n-n/2, alpha, a, lda,
             b+n/2*ldb, ldb, depth+1);
#pragma omp taskwait
        } else {
          trsm_omp_task
            (s, ul, ta, d, m/2, n, alpha, a, lda, b, ldb, depth);
          gemm_omp_task
            ('N', 'N', m-m/2, n, m/2, scalar(-1.),
             a+m/2, lda, b, ldb, alpha, b+m/2, ldb, depth);
          trsm_omp_task
            (s, ul, ta, d, m-m/2, n, scalar(1.),
             a+m/2+m/2*lda, lda, b+m/2, ldb, depth);
        }
      } else if ((s=='R' || s=='r') &&
                 (ul=='U' || ul=='u') && (ta=='N' || ta=='n')) {
        if (m >= n) {
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          trsm_omp_task
            (s, ul, ta, d, m/2, n, alpha, a, lda, b, ldb, depth+1);
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          trsm_omp_task
            (s, ul, ta, d, m-m/2, n, alpha, a, lda, b+m/2, ldb, depth+1);
#pragma omp taskwait
        } else {
          trsm_omp_task
            (s, ul, ta, d, m, n/2, alpha, a, lda, b, ldb, depth);
          gemm_omp_task
            ('N', 'N', m, n-n/2, n/2, scalar(-1.), b, ldb,
             a+n/2*lda, lda, alpha, b+n/2*ldb, ldb, depth);
          trsm_omp_task
            (s, ul, ta, d, m, n-n/2, scalar(1.),
             a+n/2+n/2*lda, lda, b+n/2*ldb, ldb, depth);
        }
      } else if ((s=='L' || s=='l') &&
                 (ul=='U' || ul=='u') && (ta=='N' || ta=='n')) {
        if (n >= m) {
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          trsm_omp_task
            (s, ul, ta, d, m, n/2, alpha, a, lda, b, ldb, depth+1);
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
          trsm_omp_task
            (s, ul, ta, d, m, n-n/2, alpha, a, lda,
             b+n/2*ldb, ldb, depth+1);
#pragma omp taskwait
        } else {
          trsm_omp_task
            (s, ul, ta, d, m-m/2, n, alpha,
             a+m/2+m/2*lda, lda, b+m/2, ldb, depth);
          gemm_omp_task
            ('N', 'N', m/2, n, m-m/2, scalar(-1.),
             a+m/2*lda, lda, b+m/2, ldb, alpha, b, ldb, depth);
          trsm_omp_task
            (s, ul, ta, d, m/2, n, scalar(1.), a, lda, b, ldb, depth);
        }
      } else {
        // std::cerr << "trsm_omp_task not implemented with this combination of"
        //           << " side, uplo and transpose" << std::endl;
        blas::trsm(s, ul, ta, d, m, n, alpha, a, lda, b, ldb);
        //abort();
      }
    }
  }

  template<typename scalar>
  void laswp_omp_task(int n, scalar* a, int lda, int k1, int k2,
                      const int* ipiv, int incx, int depth) {
    if (depth>=params::task_recursion_cutoff_level || n <= OMPTileSize)
      blas::laswp(n, a, lda, k1, k2, ipiv, incx);
    else {
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
      laswp_omp_task(n/2, a, lda, k1, k2, ipiv, incx, depth+1);
#pragma omp task final(depth >= params::task_recursion_cutoff_level-1)  \
  mergeable
      laswp_omp_task(n-n/2, a+(n/2)*lda, lda, k1, k2, ipiv, incx, depth+1);
#pragma omp taskwait
    }
  }

  // recursive with row pivoting, drop in replacement for regular
  // getrf
  template<typename scalar>
  int getrf_omp_task(int m, int n, scalar* a, int lda, int* ipiv, int depth) {
    int info = 0;
    //  getrfmod(m, n, a, lda, ipiv, info, depth);
    if (depth>=params::task_recursion_cutoff_level || n <= 1)
      // sequential part, can make this very skinny or add parallelism
      info = blas::getrf(m, n, a, lda, ipiv);
    else {
      int k = std::min(m,n/2);
      info = getrf_omp_task(m, k, a, lda, ipiv, depth);
      laswp_omp_task(n-k, a+k*lda, lda, 1, k, ipiv, 1, depth);
      trsm_omp_task
        ('L', 'L', 'N', 'U', k, n-k, scalar(1.), a, lda, a+k*lda, lda, depth);
      if (m > k){
        gemm_omp_task
          ('N', 'N', m-k, n-k, k, scalar(-1.), a+k, lda,
           a+k*lda, lda, scalar(1.), a+k+k*lda, lda, depth);
        int ierr = getrf_omp_task
          (m-k, n-k, a+k+k*lda, lda, &ipiv[k], depth);
        if (!info && ierr) info = ierr + k;
        laswp_omp_task
          (k, a+k, lda, 1, std::min(m-k, n-k), &ipiv[k], 1, depth);
        for (int i=0; i<std::min(m-k, n-k); i++) ipiv[k+i] += k;
      }
    }
    return info;
  }


  template<typename scalar>
  int getrs_omp_task(char t, int m, int n, const scalar *a, int lda,
                     const int* piv, scalar *b, int ldb,
                     int depth) {
    if (depth>=params::task_recursion_cutoff_level ||
        2*double(m)*n <= OMPThreshold)
      return blas::getrs(t, m, n, a, lda, piv, b, ldb);
    else {
      if (t=='N' || t=='n') {
        if (n==1) {
          blas::laswp(1, b, ldb, 1, m, piv, 1);
          trsv_omp_task('L', 'N', 'U', m, a, lda, b, 1, depth);
          trsv_omp_task('U', 'N', 'N', m, a, lda, b, 1, depth);
        } else {
          laswp_omp_task(n, b, ldb, 1, m, piv, 1, depth);
          trsm_omp_task
            ('L', 'L', 'N', 'U', m, n, scalar(1.), a, lda, b, ldb, depth);
          trsm_omp_task
            ('L', 'U', 'N', 'N', m, n, scalar(1.), a, lda, b, ldb, depth);
        }
      } else {
        std::cerr << "getrs_omp_task not implemented for transpose"
                  << std::endl;
        abort();
      }
      return 0; // check division by zero?
    }
  }


  // explicit template declarations
  template void gemm_omp_task(char ta, char tb, int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc, int depth);
  template void gemm_omp_task(char ta, char tb, int m, int n, int k, double alpha, const double* a, int lda, const double* b, int ldb, double beta, double* c, int ldc, int depth);
  template void gemm_omp_task(char ta, char tb, int m, int n, int k, std::complex<float> alpha, const std::complex<float>* a, int lda, const std::complex<float>* b, int ldb, std::complex<float> beta, std::complex<float>* c, int ldc, int depth);
  template void gemm_omp_task(char ta, char tb, int m, int n, int k, std::complex<double> alpha, const std::complex<double>* a, int lda, const std::complex<double>* b, int ldb, std::complex<double> beta, std::complex<double>* c, int ldc, int depth);

  template void gemv_omp_task(char t, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy, int depth);
  template void gemv_omp_task(char t, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy, int depth);
  template void gemv_omp_task(char t, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x, int incx, std::complex<float> beta, std::complex<float> *y, int incy, int depth);
  template void gemv_omp_task(char t, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x, int incx, std::complex<double> beta, std::complex<double> *y, int incy, int depth);

  template void trsv_omp_task(char ul, char ta, char d, int n, const float* a, int lda, float* x, int incx, int depth);
  template void trsv_omp_task(char ul, char ta, char d, int n, const double* a, int lda, double* x, int incx, int depth);
  template void trsv_omp_task(char ul, char ta, char d, int n, const std::complex<float>* a, int lda, std::complex<float>* x, int incx, int depth);
  template void trsv_omp_task(char ul, char ta, char d, int n, const std::complex<double>* a, int lda, std::complex<double>* x, int incx, int depth);

  template void trmm_omp_task(char s, char ul, char ta, char d, int m, int n, float alpha, const float* a, int lda, float* b, int ldb, int depth);
  template void trmm_omp_task(char s, char ul, char ta, char d, int m, int n, double alpha, const double* a, int lda, double* b, int ldb, int depth);
  template void trmm_omp_task(char s, char ul, char ta, char d, int m, int n, std::complex<float> alpha, const std::complex<float>* a, int lda, std::complex<float>* b, int ldb, int depth);
  template void trmm_omp_task(char s, char ul, char ta, char d, int m, int n, std::complex<double> alpha, const std::complex<double>* a, int lda, std::complex<double>* b, int ldb, int depth);

  template void trmv_omp_task(char ul, char ta, char d, int n, const float* a, int lda, float* x, int incx, int depth);
  template void trmv_omp_task(char ul, char ta, char d, int n, const double* a, int lda, double* x, int incx, int depth);
  template void trmv_omp_task(char ul, char ta, char d, int n, const std::complex<float>* a, int lda, std::complex<float>* x, int incx, int depth);
  template void trmv_omp_task(char ul, char ta, char d, int n, const std::complex<double>* a, int lda, std::complex<double>* x, int incx, int depth);

  template void trsm_omp_task(char s, char ul, char ta, char d, int m, int n, float alpha, const float* a, int lda, float* b, int ldb, int depth);
  template void trsm_omp_task(char s, char ul, char ta, char d, int m, int n, double alpha, const double* a, int lda, double* b, int ldb, int depth);
  template void trsm_omp_task(char s, char ul, char ta, char d, int m, int n, std::complex<float> alpha, const std::complex<float>* a, int lda, std::complex<float>* b, int ldb, int depth);
  template void trsm_omp_task(char s, char ul, char ta, char d, int m, int n, std::complex<double> alpha, const std::complex<double>* a, int lda, std::complex<double>* b, int ldb, int depth);

  template int getrf_omp_task(int m, int n, float* a, int lda, int* ipiv, int depth);
  template int getrf_omp_task(int m, int n, double* a, int lda, int* ipiv, int depth);
  template int getrf_omp_task(int m, int n, std::complex<float>* a, int lda, int* ipiv, int depth);
  template int getrf_omp_task(int m, int n, std::complex<double>* a, int lda, int* ipiv, int depth);

  template int getrs_omp_task(char t, int m, int n, const float *a, int lda, const int* piv, float *b, int ldb, int depth);
  template int getrs_omp_task(char t, int m, int n, const double *a, int lda, const int* piv, double *b, int ldb, int depth);
  template int getrs_omp_task(char t, int m, int n, const std::complex<float> *a, int lda, const int* piv, std::complex<float> *b, int ldb, int depth);
  template int getrs_omp_task(char t, int m, int n, const std::complex<double> *a, int lda, const int* piv, std::complex<double> *b, int ldb, int depth);

  template void laswp_omp_task(int n, float* a, int lda, int k1, int k2, const int* ipiv, int incx, int depth);
  template void laswp_omp_task(int n, double* a, int lda, int k1, int k2, const int* ipiv, int incx, int depth);
  template void laswp_omp_task(int n, std::complex<float>* a, int lda, int k1, int k2, const int* ipiv, int incx, int depth);
  template void laswp_omp_task(int n, std::complex<double>* a, int lda, int k1, int k2, const int* ipiv, int incx, int depth);

} // end namespace strumpack
