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
#if defined(STRUMPACK_USE_KBLAS)
#include "batch_ara.h" // KBLAS_HAS_ARA_TOL_ARRAY, when supported by KBLAS
#elif defined(STRUMPACK_USE_GPU_ARA)
// native HIP/CUDA implementation (HIPBatchedARA.hip,
// CUDABatchedARA.cu), same interface
#define KBLAS_HAS_ARA_TOL_ARRAY 1
#define KBLAS_Success 0
#endif

namespace strumpack {
  namespace gpu {
    namespace kblas {

      template<typename scalar_t> void
      ara_workspace(Handle& handle, int blocksize, int batchcount);

#if defined(STRUMPACK_USE_KBLAS)
      template<typename scalar_t,
               typename real_t=typename RealType<scalar_t>::value_type>
      void ara(Handle& handle, int* rows_batch, int* cols_batch,
               scalar_t** M_batch, int* ldm_batch,
               scalar_t** A_batch, int* lda_batch,
               scalar_t** B_batch, int* ldb_batch, int* ranks_batch,
               real_t tol, int max_rows, int max_cols, int* max_rank,
               int bs, int r, int* info, int relative, int num_ops);

#endif

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

#if defined(STRUMPACK_USE_GPU_ARA)
      /**
       * Batched post-processing of ara_tol: for every tile with
       * info == 0, copy Q (m x rank) to the start of the tile storage
       * M, followed by V^H (rank x n). Same stream as the ARA, no
       * synchronization.
       */
      template<typename scalar_t> void
      ara_finalize(Handle& handle, int* rows_batch, int* cols_batch,
                   scalar_t** M_batch, scalar_t** Q_batch, int* ldq_batch,
                   scalar_t** V_batch, int* ldv_batch,
                   int* ranks_batch, int* info, int num_ops);
#endif

#if defined(STRUMPACK_USE_GPU_ARA)
      /**
       * Data for the ARA variant that uses batched GEMMs (the GEMMs
       * are called from BLRBatch.cpp, these routines do the rest). All
       * pointers are device pointers. See ara_magma_worksize and
       * ara_magma_layout for the workspace.
       */
      template<typename scalar_t,
               typename real_t=typename RealType<scalar_t>::value_type>
      struct AraMagma {
        int B, bs, rstop, maxm, maxn, cap;
        unsigned long long seed;
        int *rows, *cols, *ldm, *ldq, *ldv, *ranks, *info, *maxr;
        scalar_t **M, **Q, **V;
        real_t* tol;
        int *ymrow, *nbv, *rkm, *ldc, *state, *small, *active;
        scalar_t **omega, **Y, **C;
        scalar_t *omega_data, *c_data;
      };

      template<typename scalar_t> std::size_t
      ara_magma_worksize(int B, int maxn, int cap, int bs) {
        return round_up(7*std::size_t(B)*sizeof(int)) +
          round_up(3*std::size_t(B)*sizeof(scalar_t*)) +
          round_up(std::size_t(B)*maxn*bs*sizeof(scalar_t)) +
          round_up(std::size_t(B)*cap*bs*sizeof(scalar_t));
      }

      template<typename scalar_t, typename real_t> void
      ara_magma_layout(AraMagma<scalar_t,real_t>& a, void* dwork) {
        const std::size_t B = a.B;
        char* p = static_cast<char*>(dwork);
        int* ip = reinterpret_cast<int*>(p);
        a.ymrow = ip;  a.nbv = ip + B;  a.rkm = ip + 2*B;
        a.ldc = ip + 3*B;  a.state = ip + 4*B;  a.small = ip + 5*B;
        a.active = ip + 6*B;
        p += round_up(7*B*sizeof(int));
        a.omega = reinterpret_cast<scalar_t**>(p);
        a.Y = a.omega + B;  a.C = a.Y + B;
        p += round_up(3*B*sizeof(scalar_t*));
        a.omega_data = reinterpret_cast<scalar_t*>(p);
        p += round_up(B*a.maxn*a.bs*sizeof(scalar_t));
        a.c_data = reinterpret_cast<scalar_t*>(p);
      }

      template<typename scalar_t, typename real_t> void
      ara_magma_init(Handle& handle, AraMagma<scalar_t,real_t>& a);
      template<typename scalar_t, typename real_t> void
      ara_magma_round(Handle& handle, const AraMagma<scalar_t,real_t>& a,
                      int iter);
      template<typename scalar_t, typename real_t> void
      ara_magma_inblock(Handle& handle, const AraMagma<scalar_t,real_t>& a);
      template<typename scalar_t, typename real_t> void
      ara_magma_final(Handle& handle, const AraMagma<scalar_t,real_t>& a);
#endif

      // number of thread blocks used per matrix in front_norm
      constexpr int front_norm_blocks = 128;

      /**
       * Device workspace (in bytes) needed by front_norm. No ARA
       * routine allocates device memory, the caller provides it.
       */
      template<typename scalar_t> std::size_t front_norm_worksize() {
        using real_t = typename RealType<scalar_t>::value_type;
        return round_up(9*sizeof(int)) + round_up(3*sizeof(scalar_t*)) +
          round_up(3*front_norm_blocks*sizeof(real_t));
      }

      /**
       * Frobenius norm of [F11 F12; F21 0], with F11, F12 and F21 on
       * the device. dwork is device memory of size
       * front_norm_worksize<scalar_t>(). Synchronizes the default stream.
       */
      template<typename scalar_t>
      typename RealType<scalar_t>::value_type
      front_norm(const DenseMatrix<scalar_t>& F11,
                 const DenseMatrix<scalar_t>& F12,
                 const DenseMatrix<scalar_t>& F21, void* dwork);
#endif

    } // end namespace kblas
  } // end namespace gpu
} // end namespace strumpack

#endif // STRUMPACK_KBLAS_WRAPPER_HPP
