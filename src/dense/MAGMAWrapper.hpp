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
#ifndef STRUMPACK_MAGMA_WRAPPER_HPP
#define STRUMPACK_MAGMA_WRAPPER_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <cassert>
#include <memory>

#include "DenseMatrix.hpp"
#if defined(STRUMPACK_USE_CUDA)
#include "dense/CUDAWrapper.hpp"
#endif
#if defined(STRUMPACK_USE_HIP)
#include "dense/HIPWrapper.hpp"
#endif

#include <magma_v2.h>
#include <magma_auxiliary.h>
#include <magma_svbatched.h>
#include <magma_dvbatched.h>
#include <magma_cvbatched.h>
#include <magma_zvbatched.h>


namespace strumpack {
  namespace gpu {
    namespace magma {

      class MAGMAQueue {
      public:
        MAGMAQueue(Stream& s, BLASHandle& h) {
          magma_init();
#if defined(STRUMPACK_USE_CUDA)
          magma_queue_create_from_cuda(0, s, h, NULL, &q_);
#elif defined(STRUMPACK_USE_HIP)
          magma_queue_create_from_hip(0, s, h, NULL, &q_);
#endif
        }
        ~MAGMAQueue() {
          magma_queue_destroy(q_);
          magma_finalize();
        }
        operator magma_queue_t&() { return q_; }
        operator const magma_queue_t&() const { return q_; }
      private:
        magma_queue_t q_;
      };


      inline int getrf(DenseMatrix<float>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        //magma_sgetrf_native
        magma_sgetrf_gpu
          (A.rows(), A.cols(), A.data(), A.ld(), piv.data(), &info);
        gpu_check(gpu::copy_host_to_device(dpiv, piv.data(), A.rows()));
        return info;
      }
      inline int getrf(DenseMatrix<double>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        //magma_dgetrf_native
        magma_dgetrf_gpu
          (A.rows(), A.cols(), A.data(), A.ld(), piv.data(), &info);
        gpu_check(gpu::copy_host_to_device(dpiv, piv.data(), A.rows()));
        return info;
      }
      inline int getrf(DenseMatrix<std::complex<float>>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        //magma_cgetrf_native
        magma_cgetrf_gpu
          (A.rows(), A.cols(), reinterpret_cast<magmaFloatComplex*>(A.data()),
           A.ld(), piv.data(), &info);
        gpu_check(gpu::copy_host_to_device(dpiv, piv.data(), A.rows()));
        return info;
      }
      inline int getrf(DenseMatrix<std::complex<double>>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        //magma_zgetrf_native
        magma_zgetrf_gpu
          (A.rows(), A.cols(), reinterpret_cast<magmaDoubleComplex*>(A.data()),
           A.ld(), piv.data(), &info);
        gpu_check(gpu::copy_host_to_device(dpiv, piv.data(), A.rows()));
        return info;
      }


      inline magma_int_t getrf_vbatched_max_nocheck_work
      (magma_int_t* m, magma_int_t* n,
       magma_int_t max_m, magma_int_t max_n,
       magma_int_t max_minmn, magma_int_t max_mxn,
       float **dA_array, magma_int_t *ldda,
       magma_int_t **dipiv_array, magma_int_t *info_array,
       void* work, magma_int_t* lwork,
       magma_int_t batchCount, magma_queue_t queue) {
        if (!batchCount) return 0;
        return magma_sgetrf_vbatched_max_nocheck_work
          (m, n, max_m, max_n, max_minmn, max_mxn,
           dA_array, ldda, dipiv_array, info_array,
           work, lwork, batchCount, queue);
      }
      inline magma_int_t getrf_vbatched_max_nocheck_work
      (magma_int_t* m, magma_int_t* n,
       magma_int_t max_m, magma_int_t max_n,
       magma_int_t max_minmn, magma_int_t max_mxn,
       double **dA_array, magma_int_t *ldda,
       magma_int_t **dipiv_array, magma_int_t *info_array,
       void* work, magma_int_t* lwork,
       magma_int_t batchCount, magma_queue_t queue) {
        if (!batchCount) return 0;
        return magma_dgetrf_vbatched_max_nocheck_work
          (m, n, max_m, max_n, max_minmn, max_mxn,
           dA_array, ldda, dipiv_array, info_array,
           work, lwork, batchCount, queue);
      }
      inline magma_int_t getrf_vbatched_max_nocheck_work
      (magma_int_t* m, magma_int_t* n,
       magma_int_t max_m, magma_int_t max_n,
       magma_int_t max_minmn, magma_int_t max_mxn,
       std::complex<float>** dA_array, magma_int_t *ldda,
       magma_int_t **dipiv_array, magma_int_t *info_array,
       void* work, magma_int_t* lwork,
       magma_int_t batchCount, magma_queue_t queue) {
        if (!batchCount) return 0;
        return magma_cgetrf_vbatched_max_nocheck_work
          (m, n, max_m, max_n, max_minmn, max_mxn,
           (magmaFloatComplex**)dA_array, ldda,
           dipiv_array, info_array, work, lwork, batchCount, queue);
      }
      inline magma_int_t getrf_vbatched_max_nocheck_work
      (magma_int_t* m, magma_int_t* n,
       magma_int_t max_m, magma_int_t max_n,
       magma_int_t max_minmn, magma_int_t max_mxn,
       std::complex<double>** dA_array, magma_int_t *ldda,
       magma_int_t **dipiv_array, magma_int_t *info_array,
       void* work, magma_int_t* lwork,
       magma_int_t batchCount, magma_queue_t queue) {
        if (!batchCount) return 0;
        return magma_zgetrf_vbatched_max_nocheck_work
          (m, n, max_m, max_n, max_minmn, max_mxn,
           (magmaDoubleComplex**)dA_array, ldda,
           dipiv_array, info_array, work, lwork, batchCount, queue);
      }

      inline void trsm_vbatched_max_nocheck
      (magma_side_t side, magma_uplo_t uplo, magma_trans_t transA,
       magma_diag_t diag, magma_int_t max_m, magma_int_t max_n,
       magma_int_t *m, magma_int_t *n,
       float alpha, float **dA_array, magma_int_t *ldda,
       float **dB_array, magma_int_t *lddb, magma_int_t batchCount,
       magma_queue_t queue) {
        if (!batchCount) return;
        magmablas_strsm_vbatched_max_nocheck
          (side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array,
           ldda, dB_array, lddb, batchCount, queue);
      }
      inline void trsm_vbatched_max_nocheck
      (magma_side_t side, magma_uplo_t uplo, magma_trans_t transA,
       magma_diag_t diag, magma_int_t max_m, magma_int_t max_n,
       magma_int_t *m, magma_int_t *n,
       double alpha, double **dA_array, magma_int_t *ldda,
       double **dB_array, magma_int_t *lddb, magma_int_t batchCount,
       magma_queue_t queue) {
        if (!batchCount) return;
        magmablas_dtrsm_vbatched_max_nocheck
          (side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array,
           ldda, dB_array, lddb, batchCount, queue);
      }
      inline void trsm_vbatched_max_nocheck
      (magma_side_t side, magma_uplo_t uplo, magma_trans_t transA,
       magma_diag_t diag, magma_int_t max_m, magma_int_t max_n,
       magma_int_t *m, magma_int_t *n,
       std::complex<float> alpha,
       std::complex<float> **dA_array, magma_int_t *ldda,
       std::complex<float> **dB_array, magma_int_t *lddb,
       magma_int_t batchCount, magma_queue_t queue) {
        if (!batchCount) return;
        magmaFloatComplex alpha_ = {alpha.real(), alpha.imag()};
        magmablas_ctrsm_vbatched_max_nocheck
          (side, uplo, transA, diag, max_m, max_n, m, n, alpha_,
           (magmaFloatComplex**)dA_array, ldda,
           (magmaFloatComplex**)dB_array, lddb, batchCount, queue);
      }
      inline void trsm_vbatched_max_nocheck
      (magma_side_t side, magma_uplo_t uplo, magma_trans_t transA,
       magma_diag_t diag, magma_int_t max_m, magma_int_t max_n,
       magma_int_t *m, magma_int_t *n,
       std::complex<double> alpha,
       std::complex<double> **dA_array, magma_int_t *ldda,
       std::complex<double> **dB_array, magma_int_t *lddb,
       magma_int_t batchCount, magma_queue_t queue) {
        if (!batchCount) return;
        magmaDoubleComplex alpha_ = {alpha.real(), alpha.imag()};
        magmablas_ztrsm_vbatched_max_nocheck
          (side, uplo, transA, diag, max_m, max_n, m, n, alpha_,
           (magmaDoubleComplex**)dA_array, ldda,
           (magmaDoubleComplex**)dB_array, lddb, batchCount, queue);
      }

      inline void gemm_vbatched_max_nocheck
      (magma_trans_t transA, magma_trans_t transB,
       magma_int_t *m, magma_int_t *n, magma_int_t *k, float alpha,
       float const *const *dA_array, magma_int_t *ldda,
       float const *const *dB_array, magma_int_t *lddb,
       float beta, float **dC_array, magma_int_t *lddc,
       magma_int_t batchCount,
       magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
       magma_queue_t queue) {
        if (!batchCount) return;
        magmablas_sgemm_vbatched_max_nocheck
          (transA, transB, m, n, k, alpha, dA_array, ldda,
           dB_array, lddb, beta, dC_array, lddc, batchCount,
           max_m, max_n, max_k, queue);
      }
      inline void gemm_vbatched_max_nocheck
      (magma_trans_t transA, magma_trans_t transB,
       magma_int_t *m, magma_int_t *n, magma_int_t *k, double alpha,
       double const *const *dA_array, magma_int_t *ldda,
       double const *const *dB_array, magma_int_t *lddb,
       double beta, double **dC_array, magma_int_t *lddc,
       magma_int_t batchCount,
       magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
       magma_queue_t queue) {
        if (!batchCount) return;
        magmablas_dgemm_vbatched_max_nocheck
          (transA, transB, m, n, k, alpha, dA_array, ldda,
           dB_array, lddb, beta, dC_array, lddc, batchCount,
           max_m, max_n, max_k, queue);
      }
      inline void gemm_vbatched_max_nocheck
      (magma_trans_t transA, magma_trans_t transB,
       magma_int_t *m, magma_int_t *n, magma_int_t *k,
       std::complex<float> alpha,
       std::complex<float> const *const *dA_array, magma_int_t *ldda,
       std::complex<float> const *const *dB_array, magma_int_t *lddb,
       std::complex<float> beta,
       std::complex<float> **dC_array, magma_int_t *lddc,
       magma_int_t batchCount,
       magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
       magma_queue_t queue) {
        if (!batchCount) return;
        magmaFloatComplex alpha_ = {alpha.real(), alpha.imag()},
          beta_ = {beta.real(), beta.imag()};
        magmablas_cgemm_vbatched_max_nocheck
          (transA, transB, m, n, k, alpha_,
           (magmaFloatComplex**)dA_array, ldda,
           (magmaFloatComplex**)dB_array, lddb, beta_,
           (magmaFloatComplex**)dC_array, lddc, batchCount,
           max_m, max_n, max_k, queue);
      }
      inline void gemm_vbatched_max_nocheck
      (magma_trans_t transA, magma_trans_t transB,
       magma_int_t *m, magma_int_t *n, magma_int_t *k,
       std::complex<double> alpha,
       std::complex<double> const *const *dA_array, magma_int_t *ldda,
       std::complex<double> const *const *dB_array, magma_int_t *lddb,
       std::complex<double> beta,
       std::complex<double> **dC_array, magma_int_t *lddc,
       magma_int_t batchCount,
       magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
       magma_queue_t queue) {
        if (!batchCount) return;
        magmaDoubleComplex alpha_ = {alpha.real(), alpha.imag()},
          beta_ = {beta.real(), beta.imag()};
        magmablas_zgemm_vbatched_max_nocheck
          (transA, transB, m, n, k, alpha_,
           (magmaDoubleComplex**)dA_array, ldda,
           (magmaDoubleComplex**)dB_array, lddb, beta_,
           (magmaDoubleComplex**)dC_array, lddc, batchCount,
           max_m, max_n, max_k, queue);
      }

      inline void gemv_vbatched_max_nocheck
      (magma_trans_t trans, magma_int_t *m, magma_int_t *n, float alpha,
       float const *const *dA_array, magma_int_t *ldda,
       float const *const *dB_array, magma_int_t *lddb,
       float beta, float **dC_array, magma_int_t *lddc,
       magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
       magma_queue_t queue) {
        if (!batchCount) return;
        magmablas_sgemv_vbatched_max_nocheck
          (trans, m, n, alpha,
           const_cast<float**>(dA_array), ldda,
           const_cast<float**>(dB_array), lddb, beta,
           dC_array, lddc, batchCount, max_m, max_n, queue);
      }
      inline void gemv_vbatched_max_nocheck
      (magma_trans_t trans, magma_int_t *m, magma_int_t *n, double alpha,
       double const *const *dA_array, magma_int_t *ldda,
       double const *const *dB_array, magma_int_t *lddb,
       double beta, double **dC_array, magma_int_t *lddc,
       magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
       magma_queue_t queue) {
        if (!batchCount) return;
        magmablas_dgemv_vbatched_max_nocheck
          (trans, m, n, alpha,
           const_cast<double**>(dA_array), ldda,
           const_cast<double**>(dB_array), lddb, beta,
           dC_array, lddc, batchCount, max_m, max_n, queue);
      }
      inline void gemv_vbatched_max_nocheck
      (magma_trans_t trans, magma_int_t *m, magma_int_t *n, std::complex<float> alpha,
       std::complex<float> const *const *dA_array, magma_int_t *ldda,
       std::complex<float> const *const *dB_array, magma_int_t *lddb,
       std::complex<float> beta,
       std::complex<float> **dC_array, magma_int_t *lddc,
       magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
       magma_queue_t queue) {
        if (!batchCount) return;
        magmaFloatComplex alpha_ = {alpha.real(), alpha.imag()},
          beta_ = {beta.real(), beta.imag()};
        magmablas_cgemv_vbatched_max_nocheck
          (trans, m, n, alpha_,
           (magmaFloatComplex**)(const_cast<std::complex<float>**>(dA_array)), ldda,
           (magmaFloatComplex**)(const_cast<std::complex<float>**>(dB_array)), lddb, beta_,
           (magmaFloatComplex**)dC_array, lddc, batchCount,
           max_m, max_n, queue);
      }
      inline void gemv_vbatched_max_nocheck
      (magma_trans_t trans, magma_int_t *m, magma_int_t *n,
       std::complex<double> alpha,
       std::complex<double> const *const *dA_array, magma_int_t *ldda,
       std::complex<double> const *const *dB_array, magma_int_t *lddb,
       std::complex<double> beta,
       std::complex<double> **dC_array, magma_int_t *lddc,
       magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
       magma_queue_t queue) {
        if (!batchCount) return;
        magmaDoubleComplex alpha_ = {alpha.real(), alpha.imag()},
          beta_ = {beta.real(), beta.imag()};
        magmablas_zgemv_vbatched_max_nocheck
          (trans, m, n, alpha_,
           (magmaDoubleComplex**)(const_cast<std::complex<double>**>(dA_array)), ldda,
           (magmaDoubleComplex**)(const_cast<std::complex<double>**>(dB_array)), lddb, beta_,
           (magmaDoubleComplex**)dC_array, lddc, batchCount,
           max_m, max_n, queue);
      }

    } // end namespace magma
  } // end namespace gpu
} // end namespace strumpack

#endif // STRUMPACK_CUDA_WRAPPER_HPP
