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
#include "KBLASWrapper.hpp"

#include "kblas.h"
#include "kblas_operators.h"
#include "batch_ara.h"
#include "kblas_defs.h"

namespace strumpack {
  namespace gpu {
    namespace kblas {

      template<typename scalar_t> void
      ara_workspace(Handle& handle, int blocksize, int batchcount) {
        kblas_ara_batch_wsquery<scalar_t>(get_kblas_handle(handle), blocksize, batchcount);
        kblasAllocateWorkspace(get_kblas_handle(handle));
      }
      template void ara_workspace<float>(Handle& handle, int blocksize, int batchcount);
      template void ara_workspace<double>(Handle& handle, int blocksize, int batchcount);
      template<> void
      ara_workspace<std::complex<float>>(Handle& handle, int blocksize, int batchcount) {
        kblas_ara_batch_wsquery<cuComplex>(get_kblas_handle(handle), blocksize, batchcount);
        kblasAllocateWorkspace(get_kblas_handle(handle));
      }
      template<> void
      ara_workspace<std::complex<double>>(Handle& handle, int blocksize, int batchcount) {
        kblas_ara_batch_wsquery<cuDoubleComplex>(get_kblas_handle(handle), blocksize, batchcount);
        kblasAllocateWorkspace(get_kblas_handle(handle));
      }

      template<> void
      ara(Handle& handle, int* rows_batch, int* cols_batch,
          float** M_batch, int* ldm_batch, float** A_batch, int* lda_batch,
          float** B_batch, int* ldb_batch, int* ranks_batch, float tol,
          int max_rows, int max_cols, int* max_rank,
          int bs, int r, int* info, int relative, int num_ops) {
        kblas_sara_batch
          (get_kblas_handle(handle), rows_batch, cols_batch, M_batch,
           ldm_batch, A_batch, lda_batch, B_batch, ldb_batch, ranks_batch,
           tol, max_rows, max_cols, max_rank, bs, r, info,
           get_kblas_rand_state(handle), relative, num_ops);
      }
      template<> void
      ara(Handle& handle, int* rows_batch, int* cols_batch,
          double** M_batch, int* ldm_batch, double** A_batch, int* lda_batch,
          double** B_batch, int* ldb_batch, int* ranks_batch, double tol,
          int max_rows, int max_cols, int* max_rank,
          int bs, int r, int* info, int relative, int num_ops) {
        kblas_dara_batch
          (get_kblas_handle(handle), rows_batch, cols_batch, M_batch,
           ldm_batch, A_batch, lda_batch, B_batch, ldb_batch, ranks_batch,
           tol, max_rows, max_cols, max_rank, bs, r, info,
           get_kblas_rand_state(handle), relative, num_ops);
      }
      template<> void
      ara(Handle& handle, int* rows_batch, int* cols_batch,
          std::complex<float>** M_batch, int* ldm_batch,
          std::complex<float>** A_batch, int* lda_batch,
          std::complex<float>** B_batch, int* ldb_batch,
          int* ranks_batch, float tol,
          int max_rows, int max_cols, int* max_rank,
          int bs, int r, int* info, int relative, int num_ops) {
        kblas_cara_batch
          (get_kblas_handle(handle), rows_batch, cols_batch,
           (cuComplex**)M_batch, ldm_batch,
           (cuComplex**)A_batch, lda_batch,
           (cuComplex**)B_batch, ldb_batch, ranks_batch,
           tol, max_rows, max_cols, max_rank, bs, r, info,
           get_kblas_rand_state(handle), relative, num_ops);
      }
      template<> void
      ara(Handle& handle, int* rows_batch, int* cols_batch,
          std::complex<double>** M_batch, int* ldm_batch,
          std::complex<double>** A_batch, int* lda_batch,
          std::complex<double>** B_batch, int* ldb_batch,
          int* ranks_batch, double tol,
          int max_rows, int max_cols, int* max_rank,
          int bs, int r, int* info, int relative, int num_ops) {
        kblas_zara_batch
          (get_kblas_handle(handle), rows_batch, cols_batch,
           (cuDoubleComplex**)M_batch, ldm_batch,
           (cuDoubleComplex**)A_batch, lda_batch,
           (cuDoubleComplex**)B_batch, ldb_batch, ranks_batch,
           tol, max_rows, max_cols, max_rank, bs, r, info,
           get_kblas_rand_state(handle), relative, num_ops);
      }

    } // end namespace kblas
  } // end namespace gpu
} // end namespace strumpack


