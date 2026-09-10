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

#include "GPUWrapper.hpp"
#if defined(STRUMPACK_USE_CUDA)
#include "CUDAWrapper.hpp" // for get_kblas_handle and get_kblas_rand_state
#endif
#if defined(STRUMPACK_USE_HIP)
#include "HIPWrapper.hpp"
#endif
#include "batch_ara.h" // KBLAS_HAS_ARA_TOL_ARRAY, when supported by KBLAS

namespace strumpack {
  namespace gpu {
    namespace kblas {

      template<typename scalar_t> void
      ara_workspace(Handle& handle, int blocksize, int batchcount);

      template<typename scalar_t,
               typename real_t=typename RealType<scalar_t>::value_type>
      void ara(Handle& handle, int* rows_batch, int* cols_batch,
               scalar_t** M_batch, int* ldm_batch,
               scalar_t** A_batch, int* lda_batch,
               scalar_t** B_batch, int* ldb_batch, int* ranks_batch,
               real_t tol, int max_rows, int max_cols, int* max_rank,
               int bs, int r, int* info, int relative, int num_ops);

#if defined(KBLAS_HAS_ARA_TOL_ARRAY) && KBLAS_HAS_ARA_TOL_ARRAY
      // Per-tile absolute thresholds in device memory. The array must remain
      // valid until the operation finishes on the handle's stream.
      template<typename scalar_t,
               typename real_t=typename RealType<scalar_t>::value_type>
      void ara_tol(Handle& handle, int* rows_batch, int* cols_batch,
                   scalar_t** M_batch, int* ldm_batch,
                   scalar_t** A_batch, int* lda_batch,
                   scalar_t** B_batch, int* ldb_batch, int* ranks_batch,
                   const real_t* tol_batch, int max_rows, int max_cols,
                   int* max_rank, int bs, int r, int* info, int num_ops);

      // Match the CPU RRQR scale: max_j ||M(:,j)||_2 = |R(0,0)|.
      // Writes max(abs_tol, rel_tol * scale) per tile, on the handle's stream.
      template<typename scalar_t,
               typename real_t=typename RealType<scalar_t>::value_type>
      void ara_tolerances(Handle& handle, const int* rows, const int* cols,
                          scalar_t* const* matrices, const int* ld,
                          real_t rel_tol, real_t abs_tol,
                          real_t* tolerances, int batch_count);

      template<typename scalar_t>
      typename RealType<scalar_t>::value_type
      front_norm(const DenseMatrix<scalar_t>& F11,
                 const DenseMatrix<scalar_t>& F12,
                 const DenseMatrix<scalar_t>& F21);
#endif

    } // end namespace kblas
  } // end namespace gpu
} // end namespace strumpack

#endif // STRUMPACK_KBLAS_WRAPPER_HPP
