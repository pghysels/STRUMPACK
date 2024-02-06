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
#include <cassert>
#include <memory>
#include <functional>
#include <algorithm>

#include "misc/TaskTimer.hpp"

#include "BLRMatrix.hpp"
#include "BLRTileBLAS.hpp"
#include "BLRBatch.hpp"

#include "dense/GPUWrapper.hpp"
#include "sparse/fronts/FrontalMatrixGPUKernels.hpp"

namespace strumpack {
  namespace BLR {

    /*
     * Copy from device to device, from column major to tile layout.
     * dA is a pointer to work memory to temporarily store the entire
     * block column.
     */
    template<typename scalar_t> void
    BLRMatrix<scalar_t>::create_from_column_major_gpu
    (DenseM_t& A, scalar_t* work) {
      auto dA = A.data();
      for (std::size_t j=0; j<colblocks(); j++) {
        auto n = tilecols(j);
        DenseMW_t Aj(rows(), n, A, 0, tilecoff(j)),
          Ajtmp(rows(), n, work, rows());
        gpu::copy_device_to_device(Ajtmp, Aj);
        for (std::size_t i=0; i<rowblocks(); i++) {
          auto m = tilerows(i);
          DenseMW_t Aij(m, n, dA, m), Aijtmp(m, n, Ajtmp, tileroff(i), 0);
          gpu::copy_device_to_device<scalar_t>(Aij, Aijtmp);
          block(i, j) = DenseTile<scalar_t>::create_as_wrapper(Aij);
          dA += m * n;
        }
      }
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::construct_and_partial_factor_gpu
    (DenseM_t& A11, DenseM_t& A12, DenseM_t& A21, DenseM_t& A22,
     BLRM_t& B11, BLRM_t& B12, BLRM_t& B21,
     const std::vector<std::size_t>& tiles1,
     const std::vector<std::size_t>& tiles2,
     const DenseMatrix<bool>& admissible,
     VectorPool<scalar_t>& workspace, const Opts_t& opts) {
      // using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      auto dsep = A11.rows();
      auto dupd = A12.cols();
      B11 = BLRM_t(dsep, tiles1, dsep, tiles1);
      B12 = BLRM_t(dsep, tiles1, dupd, tiles2);
      B21 = BLRM_t(dupd, tiles2, dsep, tiles1);
      B11.piv_.resize(dsep);
      auto rb = B11.rowblocks();
      auto rb2 = B21.rowblocks();
      gpu::Stream copy_stream, comp_stream;
      gpu::Handle handle(comp_stream), handle2(copy_stream);

      int max_batchcount = std::pow(rb-1+rb2, 2); // Schur GEMM
      max_batchcount = std::max(int(2*(rb+rb2-1)), max_batchcount); // ARA, TRSM
      std::size_t max_m1 = B11.maxtilerows();
      std::size_t max_m = std::max(max_m1, B21.maxtilerows());

      gpu::HostMemory<scalar_t> pinned =
        workspace.get_pinned(max_m*std::max(dsep,dupd)); //max_mn);

      int compress_lwork = 0;

      VBatchedARA<scalar_t>::kblas_wsquery(handle, max_batchcount);

      int getrf_work_size =
        gpu::getrf_buffersize<scalar_t>(handle, max_m1);
      auto d_batch_meta = VBatchedGEMM<scalar_t>::dwork_bytes(max_batchcount);

      // TODO KBLAS
      std::size_t max_work = std::max(getrf_work_size, compress_lwork);
      std::size_t d_mem_size =
        gpu::round_up(sizeof(scalar_t)*
                      (std::max(dsep,dupd)*max_m + max_work)) +
        gpu::round_up(sizeof(int)*(dsep+1)) +
        3*gpu::round_up(d_batch_meta);

      auto dmem = workspace.get_device_bytes(d_mem_size);
      auto temp_col = dmem.template as<scalar_t>();
      auto d_work_mem = temp_col + std::max(dsep,dupd)*max_m;
      auto dpiv = gpu::aligned_ptr<int>(d_work_mem+max_work);
      auto dinfo = dpiv + dsep;
      auto d_batch_mem = gpu::aligned_ptr<char>(dinfo+1);

      gpu::DeviceMemory<char> d_batch_matrix_mem;

      B11.create_from_column_major_gpu(A11, temp_col);
      B12.create_from_column_major_gpu(A12, temp_col);
      B21.create_from_column_major_gpu(A21, temp_col);

      for (std::size_t i=0; i<rb; i++) {
        auto& D = B11.tile(i, i).D();
        // handle2 uses copy_stream, overlap this with ARA
        gpu::getrf(handle2, D, d_work_mem, getrf_work_size,
                   dpiv+B11.tileroff(i), dinfo);
        if (opts.pivot_threshold() > 0)
          gpu::replace_pivots
            (D.rows(), D.data(), opts.pivot_threshold(), &copy_stream);

        VBatchedARA<scalar_t> ara;
        for (std::size_t j=i+1; j<rb; j++) {
          if (admissible(i, j)) ara.add(B11.block(i, j));
          if (admissible(j, i)) ara.add(B11.block(j, i));
        }
        for (std::size_t j=0; j<rb2; j++) {
          ara.add(B12.block(i, j));
          ara.add(B21.block(j, i));
        }
        ara.run(handle, workspace, opts.rel_tol());
        copy_stream.synchronize(); // stream used for getrf

        VBatchedTRSMLeftRight<scalar_t> batched_trsm;
        for (std::size_t j=i+1; j<rb; j++) {
          B11.tile(i, j).laswp(handle, dpiv+B11.tileroff(i), true);
          batched_trsm.add(D, B11.tile(i, j).U(), B11.tile(j, i).V());
        }
        for (std::size_t j=0; j<rb2; j++) {
          B12.tile(i, j).laswp(handle, dpiv+B11.tileroff(i), true);
          batched_trsm.add(D, B12.tile(i, j).U(), B21.tile(j, i).V());
        }
        batched_trsm.run(handle, workspace);

        // Schur complement update
        std::size_t sVU = 0, sUVU = 0;
        for (std::size_t j=i+1; j<rb; j++) {
          for (std::size_t k=i+1; k<rb; k++)
            multiply_inc_work_size
              (B11.tile(k, i), B11.tile(i, j), sVU, sUVU);
          for (std::size_t k=0; k<rb2; k++) {
            multiply_inc_work_size
              (B11.tile(j, i), B12.tile(i, k), sVU, sUVU);
            multiply_inc_work_size
              (B21.tile(k, i), B11.tile(i, j), sVU, sUVU);
          }
        }
        for (std::size_t j=0; j<rb2; j++)
          for (std::size_t k=0; k<rb2; k++)
            multiply_inc_work_size
              (B21.tile(k, i), B12.tile(i, j), sVU, sUVU);

        if ((sVU+sUVU)*sizeof(scalar_t) > d_batch_matrix_mem.size()) {
          workspace.restore(d_batch_matrix_mem);
          d_batch_matrix_mem = workspace.get_device_bytes
            ((sVU+sUVU)*sizeof(scalar_t));
        }
        auto dVU = d_batch_matrix_mem.template as<scalar_t>();
        auto dUVU = dVU + sVU;

        int batchcount = std::pow(rb-(i+1)+rb2, 2);
        VBatchedGEMM<scalar_t> b1(batchcount, d_batch_mem),
          b2(batchcount, d_batch_mem+gpu::round_up(d_batch_meta)),
          b3(batchcount, d_batch_mem+2*gpu::round_up(d_batch_meta));

        for (std::size_t j=i+1; j<rb; j++) {
          for (std::size_t k=i+1; k<rb; k++)
            add_tile_mult(B11.tile(k, i), B11.tile(i, j), B11.tile(k, j).D(),
                          b1, b2, b3, dVU, dUVU);
          for (std::size_t k=0; k<rb2; k++) {
            add_tile_mult(B11.tile(j, i), B12.tile(i, k), B12.tile(j, k).D(),
                          b1, b2, b3, dVU, dUVU);
            add_tile_mult(B21.tile(k, i), B11.tile(i, j), B21.tile(k, j).D(),
                          b1, b2, b3, dVU, dUVU);
          }
        }
        for (std::size_t j=0; j<rb2; j++)
          for (std::size_t k=0; k<rb2; k++) {
            DenseMW_t dAkj(B21.tilerows(k), B12.tilecols(j), A22,
                           B21.tileroff(k), B12.tilecoff(j));
            add_tile_mult(B21.tile(k, i), B12.tile(i, j), dAkj,
                          b1, b2, b3, dVU, dUVU);
          }

#pragma omp parallel
#pragma omp single nowait
        {
#pragma omp task
          {
            b1.run(scalar_t(1.), scalar_t(0.), comp_stream, handle);
            b2.run(scalar_t(1.), scalar_t(0.), comp_stream, handle);
            b3.run(scalar_t(-1.), scalar_t(1.), comp_stream, handle);
          }
          if (i > 0) {
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop
#endif
              for (std::size_t j=0; j<rb; j++)
                B11.tile(j, i-1).move_to_cpu
                  (copy_stream, pinned+B11.tileroff(j)*B11.tilecols(i-1));
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop
#endif
              for (std::size_t j=0; j<rb2; j++)
                B12.tile(i-1, j).move_to_cpu
                  (copy_stream, pinned+B12.tilecoff(j)*B12.tilerows(i-1));
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop
#endif
              for (std::size_t j=0; j<rb2; j++)
                B21.tile(j, i-1).move_to_cpu
                  (copy_stream, pinned+B21.tileroff(j)*B21.tilecols(i-1));
          }
        }
        comp_stream.synchronize();
      }
      if (rb > 0) {
#pragma omp parallel for schedule(static,1)
        for (std::size_t j=0; j<rb; j++)
          B11.tile(j, rb-1).move_to_cpu
            (copy_stream, pinned+B11.tileroff(j)*B11.tilecols(rb-1));
#pragma omp parallel for schedule(static,1)
        for (std::size_t j=0; j<rb2; j++)
          B12.tile(rb-1, j).move_to_cpu
            (copy_stream, pinned+B12.tilecoff(j)*B12.tilerows(rb-1));
#pragma omp parallel for schedule(static,1)
        for (std::size_t j=0; j<rb2; j++)
          B21.tile(j, rb-1).move_to_cpu
            (copy_stream, pinned+B21.tileroff(j)*B21.tilecols(rb-1));
      }
      gpu::copy_device_to_host(B11.piv_.data(), dpiv, dsep);
      for (std::size_t i=0; i<rb; i++)
        for (std::size_t l=B11.tileroff(i); l<B11.tileroff(i+1); l++)
          B11.piv_[l] += B11.tileroff(i);
      workspace.restore(pinned);
      workspace.restore(d_batch_matrix_mem);
      workspace.restore(dmem);
    }

    // explicit template instantiations
    template class BLRMatrix<float>;
    template class BLRMatrix<double>;
    template class BLRMatrix<std::complex<float>>;
    template class BLRMatrix<std::complex<double>>;

  } // end namespace BLR
} // end namespace strumpack
