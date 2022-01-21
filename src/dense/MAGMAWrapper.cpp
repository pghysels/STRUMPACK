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
#include "MAGMAWrapper.hpp"

namespace strumpack {
  namespace gpu {
    namespace magma {

      int getrf(DenseMatrix<float>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        //magma_sgetrf_native
        magma_sgetrf_gpu
          (A.rows(), A.cols(), A.data(), A.ld(), piv.data(), &info);
        gpu::copy_host_to_device(dpiv, piv.data(), A.rows());
        return info;
      }
      int getrf(DenseMatrix<double>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        //magma_dgetrf_native
        magma_dgetrf_gpu
          (A.rows(), A.cols(), A.data(), A.ld(), piv.data(), &info);
        gpu::copy_host_to_device(dpiv, piv.data(), A.rows());
        return info;
      }
      int getrf(DenseMatrix<std::complex<float>>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        //magma_cgetrf_native
        magma_cgetrf_gpu
          (A.rows(), A.cols(), reinterpret_cast<magmaFloatComplex*>(A.data()),
           A.ld(), piv.data(), &info);
        gpu::copy_host_to_device(dpiv, piv.data(), A.rows());
        return info;
      }
      int getrf(DenseMatrix<std::complex<double>>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        //magma_zgetrf_native
        magma_zgetrf_gpu
          (A.rows(), A.cols(), reinterpret_cast<magmaDoubleComplex*>(A.data()),
           A.ld(), piv.data(), &info);
        gpu::copy_host_to_device(dpiv, piv.data(), A.rows());
        return info;
      }

      void laswpx(DenseMatrix<float>& A, int* dpiv, magma_queue_t queue, bool fwd) {
        std::vector<int> tpiv(A.rows());
        gpu::copy_device_to_host(tpiv.data(), dpiv, A.rows()); 
        magmablas_slaswpx(A.cols(), A.data(), 1, A.ld(), 1, 
                          A.rows(), tpiv.data(), fwd ? 1 : -1, queue);
      }

      void laswpx(DenseMatrix<double>& A, int* dpiv, magma_queue_t queue, bool fwd) {
        std::vector<int> tpiv(A.rows());
        gpu::copy_device_to_host(tpiv.data(), dpiv, A.rows()); 
        magmablas_dlaswpx(A.cols(), A.data(), 1, A.ld(), 1, 
                          A.rows(), tpiv.data(), fwd ? 1 : -1, queue);
      }

      void laswpx(DenseMatrix<std::complex<float>>& A, int* dpiv, 
                  magma_queue_t queue, bool fwd) {
        std::vector<int> tpiv(A.rows());
        gpu::copy_device_to_host(tpiv.data(), dpiv, A.rows()); 
        magmablas_claswpx(A.cols(), reinterpret_cast<magmaFloatComplex*>(A.data()), 
                          1, A.ld(), 1, A.rows(), tpiv.data(), fwd ? 1 : -1, queue);
      }

      void laswpx(DenseMatrix<std::complex<double>>& A, int* dpiv, 
                  magma_queue_t queue, bool fwd) {
        std::vector<int> tpiv(A.rows());
        gpu::copy_device_to_host(tpiv.data(), dpiv, A.rows()); 
        magmablas_zlaswpx(A.cols(), reinterpret_cast<magmaDoubleComplex*>(A.data()), 
                          1, A.ld(), 1, A.rows(), tpiv.data(), fwd ? 1 : -1, queue);
      }

      void gemm_vbatched(magma_trans_t transA, magma_trans_t transB,
                         magma_int_t * m, magma_int_t * n, magma_int_t * k,
                         float alpha,
                         float const *const * dA_array, magma_int_t * ldda,
                         float const *const * dB_array, magma_int_t * lddb,
                         float beta, float ** dC_array, magma_int_t * lddc,
                         magma_int_t batchCount,
                         magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                         magma_queue_t queue ) {
        magmablas_sgemm_vbatched_max_nocheck
          (transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb,
           beta, dC_array, lddc, batchCount, max_m, max_n, max_k, queue);
      }
      void gemm_vbatched(magma_trans_t transA, magma_trans_t transB,
                         magma_int_t * m, magma_int_t * n, magma_int_t * k,
                         double alpha,
                         double const *const * dA_array, magma_int_t * ldda,
                         double const *const * dB_array, magma_int_t * lddb,
                         double beta, double ** dC_array, magma_int_t * lddc,
                         magma_int_t batchCount,
                         magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                         magma_queue_t queue) {
        magmablas_dgemm_vbatched
          (transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb,
           beta, dC_array, lddc, batchCount, queue);
      }
      void gemm_vbatched(magma_trans_t transA, magma_trans_t transB,
                         magma_int_t * m, magma_int_t * n, magma_int_t * k,
                         std::complex<float> alpha,
                         std::complex<float> const *const * dA_array, magma_int_t * ldda,
                         std::complex<float> const *const * dB_array, magma_int_t * lddb,
                         std::complex<float> beta,
                         std::complex<float> ** dC_array, magma_int_t * lddc,
                         magma_int_t batchCount,
                         magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                         magma_queue_t queue) {
        magmaFloatComplex m_alpha, m_beta;
        memcpy(&m_alpha, &alpha, sizeof(magmaFloatComplex));
        memcpy(&m_beta, &beta, sizeof(magmaFloatComplex));
        magmablas_cgemm_vbatched
          (transA, transB, m, n, k, m_alpha,
           reinterpret_cast<const magmaFloatComplex* const*>(dA_array), ldda,
           reinterpret_cast<const magmaFloatComplex* const*>(dB_array), lddb,
           m_beta,
           reinterpret_cast<magmaFloatComplex* *>(dC_array), lddc,
           batchCount, queue);
      }
      void gemm_vbatched(magma_trans_t transA, magma_trans_t transB,
                         magma_int_t * m, magma_int_t * n, magma_int_t * k,
                         std::complex<double> alpha,
                         std::complex<double> const *const * dA_array, magma_int_t * ldda,
                         std::complex<double> const *const * dB_array, magma_int_t * lddb,
                         std::complex<double> beta,
                         std::complex<double> ** dC_array, magma_int_t * lddc,
                         magma_int_t batchCount,
                         magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                         magma_queue_t queue) {
        magmaDoubleComplex m_alpha, m_beta;
        memcpy(&m_alpha, &alpha, sizeof(magmaDoubleComplex));
        memcpy(&m_beta, &beta, sizeof(magmaDoubleComplex));
        magmablas_zgemm_vbatched
          (transA, transB, m, n, k, m_alpha,
           reinterpret_cast<const magmaDoubleComplex* const*>(dA_array), ldda,
           reinterpret_cast<const magmaDoubleComplex* const*>(dB_array), lddb,
           m_beta,
           reinterpret_cast<magmaDoubleComplex* *>(dC_array), lddc,
           batchCount, queue);
      }

    } // end namespace magma
  } // end namespace gpu
} // end namespace strumpack
