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
#include "CUDAWrapper.hpp"

#include <magma_v2.h>

namespace strumpack {
  namespace gpu {
    namespace magma {

      int getrf(DenseMatrix<float>& A, int* dpiv);
      int getrf(DenseMatrix<double>& A, int* dpiv);
      int getrf(DenseMatrix<std::complex<float>>& A, int* dpiv);
      int getrf(DenseMatrix<std::complex<double>>& A, int* dpiv);

      void laswpx(DenseMatrix<float>& A, const int* dpiv, 
                  magma_queue_t queue, bool fwd);
      void laswpx(DenseMatrix<double>& A, const int* dpiv, 
                  magma_queue_t queue, bool fwd);
      void laswpx(DenseMatrix<std::complex<float>>& A, const int* dpiv, 
                  magma_queue_t queue, bool fwd);
      void laswpx(DenseMatrix<std::complex<double>>& A, const int* dpiv, 
                  magma_queue_t queue, bool fwd);
      
      template<typename scalar_t,
             typename real_t=typename RealType<scalar_t>::value_type> void
      gesvd_magma(magma_vec_t jobu, magma_vec_t jobvt, real_t* S, 
            DenseMatrix<scalar_t>& A, DenseMatrix<scalar_t>& U, 
            DenseMatrix<scalar_t>& V);

      void gemm_vbatched(magma_trans_t transA, magma_trans_t transB,
                         magma_int_t * m, magma_int_t * n, magma_int_t * k,
                         float alpha,
                         float const *const * dA_array, magma_int_t * ldda,
                         float const *const * dB_array, magma_int_t * lddb,
                         float beta, float ** dC_array, magma_int_t * lddc,
                         magma_int_t batchCount,
                         magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                         magma_queue_t queue);
      void gemm_vbatched(magma_trans_t transA, magma_trans_t transB,
                         magma_int_t * m, magma_int_t * n, magma_int_t * k,
                         double alpha,
                         double const *const * dA_array, magma_int_t * ldda,
                         double const *const * dB_array, magma_int_t * lddb,
                         double beta, double ** dC_array, magma_int_t * lddc,
                         magma_int_t batchCount,
                         magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                         magma_queue_t queue);
      void gemm_vbatched(magma_trans_t transA, magma_trans_t transB,
                         magma_int_t * m, magma_int_t * n, magma_int_t * k,
                         std::complex<float> alpha,
                         std::complex<float> const *const * dA_array, magma_int_t * ldda,
                         std::complex<float> const *const * dB_array, magma_int_t * lddb,
                         std::complex<float> beta,
                         std::complex<float> ** dC_array, magma_int_t * lddc,
                         magma_int_t batchCount,
                         magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                         magma_queue_t queue);
      void gemm_vbatched(magma_trans_t transA, magma_trans_t transB,
                         magma_int_t * m, magma_int_t * n, magma_int_t * k,
                         std::complex<double> alpha,
                         std::complex<double> const *const * dA_array, magma_int_t * ldda,
                         std::complex<double> const *const * dB_array, magma_int_t * lddb,
                         std::complex<double> beta,
                         std::complex<double> ** dC_array, magma_int_t * lddc,
                         magma_int_t batchCount,
                         magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                         magma_queue_t queue);

    } // end namespace magma
  } // end namespace gpu
} // end namespace strumpack

#endif // STRUMPACK_MAGMA_WRAPPER_HPP
