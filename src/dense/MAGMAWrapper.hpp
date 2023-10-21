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

#include "GPUWrapper.hpp"
#if defined(STRUMPACK_USE_CUDA)
#include "CUDAWrapper.hpp" // for get_magma_queue, magma_int_t, ..
#endif
#if defined(STRUMPACK_USE_HIP)
#include "HIPWrapper.hpp"
#endif

namespace strumpack {
  namespace gpu {
    namespace magma {

      int getrf(DenseMatrix<float>& A, int* dpiv);
      int getrf(DenseMatrix<double>& A, int* dpiv);
      int getrf(DenseMatrix<std::complex<float>>& A, int* dpiv);
      int getrf(DenseMatrix<std::complex<double>>& A, int* dpiv);

      template<typename scalar_t,
               typename real_t=typename RealType<scalar_t>::value_type> void
      gesvd_magma(magma_vec_t jobu, magma_vec_t jobvt, real_t* S,
                  DenseMatrix<scalar_t>& A, DenseMatrix<scalar_t>& U,
                  DenseMatrix<scalar_t>& V);

      magma_int_t
      getrf_vbatched_max_nocheck_work(magma_int_t* m, magma_int_t* n,
                                      magma_int_t max_m, magma_int_t max_n,
                                      magma_int_t max_minmn, magma_int_t max_mxn,
                                      float **dA_array, magma_int_t *ldda,
                                      magma_int_t **dipiv_array, magma_int_t *info_array,
                                      void* work, magma_int_t* lwork,
                                      magma_int_t batchCount, Handle& queue);
      magma_int_t
      getrf_vbatched_max_nocheck_work(magma_int_t* m, magma_int_t* n,
                                      magma_int_t max_m, magma_int_t max_n,
                                      magma_int_t max_minmn, magma_int_t max_mxn,
                                      double **dA_array, magma_int_t *ldda,
                                      magma_int_t **dipiv_array, magma_int_t *info_array,
                                      void* work, magma_int_t* lwork,
                                      magma_int_t batchCount, Handle& queue);
      magma_int_t
      getrf_vbatched_max_nocheck_work(magma_int_t* m, magma_int_t* n,
                                      magma_int_t max_m, magma_int_t max_n,
                                      magma_int_t max_minmn, magma_int_t max_mxn,
                                      std::complex<float>** dA_array, magma_int_t *ldda,
                                      magma_int_t **dipiv_array, magma_int_t *info_array,
                                      void* work, magma_int_t* lwork,
                                      magma_int_t batchCount, Handle& queue);
      magma_int_t
      getrf_vbatched_max_nocheck_work(magma_int_t* m, magma_int_t* n,
                                      magma_int_t max_m, magma_int_t max_n,
                                      magma_int_t max_minmn, magma_int_t max_mxn,
                                      std::complex<double>** dA_array, magma_int_t *ldda,
                                      magma_int_t **dipiv_array, magma_int_t *info_array,
                                      void* work, magma_int_t* lwork,
                                      magma_int_t batchCount, Handle& queue);

      void
      trsm_vbatched_max_nocheck(magma_side_t side, magma_uplo_t uplo,
                                magma_trans_t transA, magma_diag_t diag,
                                magma_int_t max_m, magma_int_t max_n,
                                magma_int_t *m, magma_int_t *n,
                                float alpha, float **dA_array,
                                magma_int_t *ldda, float **dB_array,
                                magma_int_t *lddb, magma_int_t batchCount,
                                Handle& queue);
      void
      trsm_vbatched_max_nocheck(magma_side_t side, magma_uplo_t uplo,
                                magma_trans_t transA, magma_diag_t diag,
                                magma_int_t max_m, magma_int_t max_n,
                                magma_int_t *m, magma_int_t *n,
                                double alpha, double **dA_array,
                                magma_int_t *ldda, double **dB_array,
                                magma_int_t *lddb, magma_int_t batchCount,
                                Handle& queue);
      void
      trsm_vbatched_max_nocheck(magma_side_t side, magma_uplo_t uplo,
                                magma_trans_t transA, magma_diag_t diag,
                                magma_int_t max_m, magma_int_t max_n,
                                magma_int_t *m, magma_int_t *n,
                                std::complex<float> alpha,
                                std::complex<float> **dA_array, magma_int_t *ldda,
                                std::complex<float> **dB_array, magma_int_t *lddb,
                                magma_int_t batchCount, Handle& queue);
      void
      trsm_vbatched_max_nocheck(magma_side_t side, magma_uplo_t uplo, magma_trans_t transA,
                                magma_diag_t diag, magma_int_t max_m, magma_int_t max_n,
                                magma_int_t *m, magma_int_t *n, std::complex<double> alpha,
                                std::complex<double> **dA_array, magma_int_t *ldda,
                                std::complex<double> **dB_array, magma_int_t *lddb,
                                magma_int_t batchCount, Handle& queue);

      void
      gemm_vbatched_max_nocheck(magma_trans_t transA, magma_trans_t transB,
                                magma_int_t *m, magma_int_t *n, magma_int_t *k, float alpha,
                                float const *const *dA_array, magma_int_t *ldda,
                                float const *const *dB_array, magma_int_t *lddb,
                                float beta, float **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                Handle& queue);
      void
      gemm_vbatched_max_nocheck(magma_trans_t transA, magma_trans_t transB,
                                magma_int_t *m, magma_int_t *n, magma_int_t *k, double alpha,
                                double const *const *dA_array, magma_int_t *ldda,
                                double const *const *dB_array, magma_int_t *lddb,
                                double beta, double **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                Handle& queue);
      void
      gemm_vbatched_max_nocheck(magma_trans_t transA, magma_trans_t transB,
                                magma_int_t *m, magma_int_t *n, magma_int_t *k,
                                std::complex<float> alpha,
                                std::complex<float> const *const *dA_array, magma_int_t *ldda,
                                std::complex<float> const *const *dB_array, magma_int_t *lddb,
                                std::complex<float> beta,
                                std::complex<float> **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                Handle& queue);
      void
      gemm_vbatched_max_nocheck(magma_trans_t transA, magma_trans_t transB,
                                magma_int_t *m, magma_int_t *n, magma_int_t *k,
                                std::complex<double> alpha,
                                std::complex<double> const *const *dA_array, magma_int_t *ldda,
                                std::complex<double> const *const *dB_array, magma_int_t *lddb,
                                std::complex<double> beta,
                                std::complex<double> **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                Handle& queue);

      void
      gemv_vbatched_max_nocheck(magma_trans_t trans, magma_int_t *m, magma_int_t *n, float alpha,
                                float const *const *dA_array, magma_int_t *ldda,
                                float const *const *dB_array, magma_int_t *lddb,
                                float beta, float **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
                                Handle& queue);
      void
      gemv_vbatched_max_nocheck(magma_trans_t trans, magma_int_t *m, magma_int_t *n, double alpha,
                                double const *const *dA_array, magma_int_t *ldda,
                                double const *const *dB_array, magma_int_t *lddb,
                                double beta, double **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
                                Handle& queue);
      void
      gemv_vbatched_max_nocheck(magma_trans_t trans, magma_int_t *m, magma_int_t *n, std::complex<float> alpha,
                                std::complex<float> const *const *dA_array, magma_int_t *ldda,
                                std::complex<float> const *const *dB_array, magma_int_t *lddb,
                                std::complex<float> beta,
                                std::complex<float> **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
                                Handle& queue);
      void
      gemv_vbatched_max_nocheck(magma_trans_t trans, magma_int_t *m, magma_int_t *n,
                                std::complex<double> alpha,
                                std::complex<double> const *const *dA_array, magma_int_t *ldda,
                                std::complex<double> const *const *dB_array, magma_int_t *lddb,
                                std::complex<double> beta,
                                std::complex<double> **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
                                Handle& queue);

    } // end namespace magma
  } // end namespace gpu
} // end namespace strumpack

#endif // STRUMPACK_MAGMA_WRAPPER_HPP
