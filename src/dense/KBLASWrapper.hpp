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
#ifndef STRUMPACK_KBLAS_WRAPPER_HPP
#define STRUMPACK_KBLAS_WRAPPER_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <cassert>
#include <memory>

#include "DenseMatrix.hpp"
#include "CUDAWrapper.hpp"

#include "kblas_operators.h"
#include "batch_geqp.h"
#include "batch_qr.h"
#include "batch_ara.h"
#include "kblas.h"

namespace strumpack {
  namespace gpu {
    namespace kblas {

      // void geqp2(kblasHandle_t kblas_handle, int m, int n,
      //            float** A, int lda, float** tau, int** piv,
      //            int* ranks, float tol, int num_ops) {
      //   kblasSgeqp2_batch(kblas_handle, m, n, A, lda, tau, piv, ranks,
      //                     tol, num_ops);
      // }
      // void geqp2(kblasHandle_t kblas_handle, int m, int n,
      //            double** A, int lda, double** tau, int** piv,
      //            int* ranks, double tol, int num_ops) {
      //   kblasDgeqp2_batch(kblas_handle, m, n, A, lda, tau, piv, ranks,
      //                     tol, num_ops);
      // }
      // void geqp2(kblasHandle_t kblas_handle, int m, int n,
      //            std::complex<float>** A, int lda, std::complex<float>** tau,
      //            int** piv, int* ranks, std::complex<float> tol, int num_ops) {
      //   assert(false);
      // }
      // void geqp2(kblasHandle_t kblas_handle, int m, int n,
      //            std::complex<double>** A, int lda, std::complex<double>** tau,
      //            int** piv, int* ranks, std::complex<double> tol, int num_ops) {
      //   assert(false);
      // }

      // void geqp2_strided(kblasHandle_t kblas_handle, int m, int n,
      //                    float* A, int lda, int stride_a,
      //                    float* tau, int stride_tau, int* piv, int stride_piv,
      //                    int* ranks, float tol, int num_ops) {
      //   kblasSgeqp2_batch_strided
      //     (kblas_handle, m, n, A, lda, stride_a, tau, stride_tau,
      //      piv, stride_piv, ranks, tol, num_ops);
      // }
      // void geqp2_strided(kblasHandle_t kblas_handle, int m, int n,
      //                    double* A, int lda, int stride_a,
      //                    double* tau, int stride_tau, int* piv, int stride_piv,
      //                    int* ranks, double tol, int num_ops) {
      //   kblasDgeqp2_batch_strided
      //     (kblas_handle, m, n, A, lda, stride_a, tau, stride_tau,
      //      piv, stride_piv, ranks, tol, num_ops);
      // }
      // void geqp2_strided(kblasHandle_t kblas_handle, int m, int n,
      //                    std::complex<float>* A, int lda, int stride_a,
      //                    std::complex<float>* tau, int stride_tau,
      //                    int* piv, int stride_piv, int* ranks,
      //                    std::complex<float> tol, int num_ops) {
      //   assert(false);
      // }
      // void geqp2_strided(kblasHandle_t kblas_handle, int m, int n,
      //                    std::complex<double>* A, int lda, int stride_a,
      //                    std::complex<double>* tau, int stride_tau,
      //                    int* piv, int stride_piv, int* ranks,
      //                    std::complex<double> tol, int num_ops) {
      //   assert(false);
      // }

      // void orgqr(kblasHandle_t kblas_handle, int m, int n,
      //            float** A, int lda, float** tau, int num_ops) {
      //   kblasSorgqr_batch(kblas_handle, m, n, A, lda, tau, num_ops);
      // }
      // void orgqr(kblasHandle_t kblas_handle, int m, int n,
      //            double** A, int lda, double** tau, int num_ops) {
      //   kblasDorgqr_batch(kblas_handle, m, n, A, lda, tau, num_ops);
      // }
      // void orgqr(kblasHandle_t kblas_handle, int m, int n,
      //            std::complex<float>** A, int lda,
      //            std::complex<float>** tau, int num_ops) {
      //   assert(false);
      // }
      // void orgqr(kblasHandle_t kblas_handle, int m, int n,
      //            std::complex<double>** A, int lda,
      //            std::complex<double>** tau, int num_ops) {
      //   assert(false);
      // }

      // void orgqr_strided(kblasHandle_t kblas_handle, int m, int n,
      //                    float* A, int lda, int stride_a,
      //                    float* tau, int stride_tau, int num_ops) {
      //   kblasSorgqr_batch_strided(kblas_handle, m, n, A, lda, stride_a, tau,
      //                             stride_tau, num_ops);
      // }
      // void orgqr_strided(kblasHandle_t kblas_handle, int m, int n,
      //                    double* A, int lda, int stride_a,
      //                    double* tau, int stride_tau, int num_ops) {
      //   kblasDorgqr_batch_strided(kblas_handle, m, n, A, lda, stride_a, tau,
      //                             stride_tau, num_ops);
      // }
      // void orgqr_strided(kblasHandle_t kblas_handle, int m, int n,
      //                    std::complex<float>* A, int lda, int stride_a,
      //                    std::complex<float>* tau, int stride_tau,
      //                    int num_ops) {
      //   assert(false);
      // }
      // void orgqr_strided(kblasHandle_t kblas_handle, int m, int n,
      //                    std::complex<double>* A, int lda, int stride_a,
      //                    std::complex<double>* tau, int stride_tau,
      //                    int num_ops) {
      //   assert(false);
      // }

      void ara(BLASHandle& handle, int* rows_batch, int* cols_batch,
               float** M_batch, int* ldm_batch,
               float** A_batch, int* lda_batch,
               float** B_batch, int* ldb_batch, int* ranks_batch,
               float tol, int max_rows, int max_cols, int max_rank,
               int bs, int r, int relative, int num_ops) {
        kblas_sara_batch
          (handle, rows_batch, cols_batch, M_batch, ldm_batch,
           A_batch, lda_batch, B_batch, ldb_batch, ranks_batch,
           tol, max_rows, max_cols, max_rank, bs, r,
           handle.kblas_rand_state(), relative, num_ops);
      }
      void ara(BLASHandle& handle, int* rows_batch, int* cols_batch,
               double** M_batch, int* ldm_batch,
               double** A_batch, int* lda_batch,
               double** B_batch, int* ldb_batch, int* ranks_batch,
               double tol, int max_rows, int max_cols, int max_rank,
               int bs, int r, int relative, int num_ops) {
        kblas_dara_batch
          (handle, rows_batch, cols_batch, M_batch, ldm_batch,
           A_batch, lda_batch, B_batch, ldb_batch, ranks_batch,
           tol, max_rows, max_cols, max_rank, bs, r,
           handle.kblas_rand_state(), relative, num_ops);
      }
      void ara(kblasHandle_t handle, int* rows_batch, int* cols_batch,
               std::complex<float>** M_batch, int* ldm_batch,
               std::complex<float>** A_batch, int* lda_batch,
               std::complex<float>** B_batch, int* ldb_batch,
               int* ranks_batch, std::complex<float> tol,
               int max_rows, int max_cols, int max_rank,
               int bs, int r, int relative, int num_ops) {
        assert(false);
      }
      void ara(kblasHandle_t handle, int* rows_batch, int* cols_batch,
               std::complex<double>** M_batch, int* ldm_batch,
               std::complex<double>** A_batch, int* lda_batch,
               std::complex<double>** B_batch, int* ldb_batch,
               int* ranks_batch, std::complex<double> tol,
               int max_rows, int max_cols, int max_rank, int bs, int r,
               int relative, int num_ops) {
        assert(false);
      }

    } // end namespace kblas
  } // end namespace gpu
} // end namespace strumpack

#endif // STRUMPACK_KBLAS_WRAPPER_HPP
