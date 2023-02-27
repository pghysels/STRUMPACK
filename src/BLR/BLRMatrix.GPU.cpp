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

#include "BLRMatrix.hpp"
#include "BLRTileBLAS.hpp"
#include "misc/TaskTimer.hpp"
// for gpu::round_up
#include "sparse/fronts/FrontalMatrixGPUKernels.hpp"

#if defined(STRUMPACK_USE_CUDA)
#include "dense/CUDAWrapper.hpp"
#endif
#if defined(STRUMPACK_USE_HIP)
#include "dense/HIPWrapper.hpp"
#endif
#if defined(STRUMPACK_USE_MAGMA)
#include "dense/MAGMAWrapper.hpp"
#endif
#if defined(STRUMPACK_USE_KBLAS)
#include "kblas_operators.h"
#include "kblas.h"
#include "dense/KBLASWrapper.hpp"
#endif

namespace strumpack {
  namespace BLR {


    template<typename scalar_t> void
    multiply_inc_work_size(const BLRTile<scalar_t>& A,
                           const BLRTile<scalar_t>& B,
                           std::size_t& temp1, std::size_t& temp2) {
      if (A.is_low_rank()) {
        if (B.is_low_rank()) {
          temp1 += A.rank() * B.rank();
          temp2 += (B.rank() < A.rank()) ?
            A.rows() * B.rank() : A.rank() * B.cols();
        } else
          temp1 += A.rank() * B.cols();
      } else if (B.is_low_rank())
        temp1 += A.rows() * B.rank();
    }

    template<typename scalar_t> void
    add_tile_mult(BLRTile<scalar_t>& A, BLRTile<scalar_t>& B,
                  DenseMatrix<scalar_t>& C, VBatchedGEMM<scalar_t>& b1,
                  VBatchedGEMM<scalar_t>& b2, VBatchedGEMM<scalar_t>& b3,
                  scalar_t*& d1, scalar_t*& d2) {
      auto m = A.rows(), n = B.cols(), k = A.cols(),
        r1 = A.rank(), r2 = B.rank();
      if (A.is_low_rank()) {
        if (B.is_low_rank()) {
          b1.add(r1, r2, k, A.V().data(), B.U().data(), d1);
          if (r2 < r1) {
            b2.add(m, r2, r1, A.U().data(), d1, d2);
            b3.add(m, n, r2, d2, B.V().data(), C.data(), C.ld());
            d2 += m * r2;
          } else {
            b2.add(r1, n, r2, d1, B.V().data(), d2);
            b3.add(m, n, r1, A.U().data(), d2, C.data(), C.ld());
            d2 += r1 * n;
          }
          d1 += r1 * r2;
        } else {
          b1.add(r1, n, k, A.V().data(), B.D().data(), d1);
          b3.add(m, n, r1, A.U().data(), d1, C.data(), C.ld());
          d1 += r1 * n;
        }
      } else {
        if (B.is_low_rank()) {
          b1.add(m, r2, k, A.D().data(), B.U().data(), d1);
          b3.add(m, n, r2, d1, B.V().data(), C.data(), C.ld());
          d1 += m * r2;
        } else
          b3.add(m, n, k, A.D().data(), B.D().data(), C.data(), C.ld());
      }
    }

    template<typename scalar_t>
    void BLRMatrix<scalar_t>::create_dense_tile_gpu
    (std::size_t i, std::size_t j, DenseM_t& A, scalar_t*& dA) {
      DenseMW_t dAij(tilerows(i), tilecols(j), dA, tilerows(i));
      block(i, j) = std::unique_ptr<DenseTile<scalar_t>>
        (new DenseTile<scalar_t>(dAij));
      dA += tilerows(i) * tilecols(j);
      gpu_check(gpu::copy_device_to_device(tile(i, j).D(), tile(A, i, j)));
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::compress_tile_gpu
    (gpu::SOLVERHandle& handle, gpu::BLASHandle& blashandle,
     std::size_t i, std::size_t j, DenseM_t& A,
     int* info, scalar_t* work, const Opts_t& opts) {
      auto m = tilerows(i), n = tilecols(j);
      if (m != 0 && n != 0) {
        std::size_t minmn = std::min(m, n);
        auto d_sval_real = reinterpret_cast<real_t*>(work);
        auto d_sval_scalar = work + minmn;
        auto dUmem = d_sval_scalar + minmn;
        auto dVmem = dUmem + m*minmn;
        auto dAmem = dVmem + n*minmn;
        auto svd_work = dAmem + m*n;
        DenseMW_t Atmp(m, n, dAmem, m);
        gpu_check(gpu::copy_device_to_device(Atmp, A));
        DenseMW_t dU(m, minmn, dUmem, m), dV(n, minmn, dVmem, n);
        std::vector<real_t> h_sval_real(minmn);
        const double tol = opts.rel_tol();
        int lwork = gpu::gesvdj_buffersize<scalar_t>
          (handle, Jobz::V, m, n);
        gpu::gesvdj<scalar_t>
          (handle, Jobz::V, Atmp, d_sval_real, dU, dV, info,
           svd_work, lwork, tol);
        gpu_check(gpu::copy_device_to_host
                  (h_sval_real.data(), d_sval_real, minmn));
        std::size_t rank = 0;
        while (rank < minmn && h_sval_real[rank] >= tol) rank++;
        if (rank*(m+n) < m*n) {
          auto dA = tile(i, j).D().data();
          DenseMW_t tU(m, rank, dA, m), tV(rank, n, dA+m*rank, rank);
          block(i, j) = std::unique_ptr<LRTile<scalar_t>>
            (new LRTile<scalar_t>(tU, tV));
          DenseMW_t dU_tmp(m, rank, dU, 0, 0);
          gpu_check(gpu::copy_device_to_device(tU, dU_tmp));
          gpu::geam<scalar_t>
            (blashandle, Trans::C, Trans::N, 1.0, dV, 0.0, dV, tV);
          gpu_check(gpu::copy_real_to_scalar<scalar_t>
                    (d_sval_scalar, d_sval_real, rank));
          gpu::dgmm<scalar_t>(blashandle, Side::L, tV, d_sval_scalar, tV);
        }
      }
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::construct_and_partial_factor_gpu
    (DenseMatrix<scalar_t>& A11, DenseMatrix<scalar_t>& A12,
     DenseMatrix<scalar_t>& A21, DenseMatrix<scalar_t>& A22,
     BLRMatrix<scalar_t>& B11, std::vector<int>& piv,
     BLRMatrix<scalar_t>& B12, BLRMatrix<scalar_t>& B21,
     const std::vector<std::size_t>& tiles1,
     const std::vector<std::size_t>& tiles2,
     const DenseMatrix<bool>& admissible, const Opts_t& opts) {
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      auto dsep = A11.rows();
      auto dupd = A12.cols();
      B11 = BLRMatrix<scalar_t>(dsep, tiles1, dsep, tiles1);
      B12 = BLRMatrix<scalar_t>(dsep, tiles1, dupd, tiles2);
      B21 = BLRMatrix<scalar_t>(dupd, tiles2, dsep, tiles1);
      piv.resize(dsep);
      auto rb = B11.rowblocks();
      auto rb2 = B21.rowblocks();
      gpu::Stream copy_stream, comp_stream;
      gpu::BLASHandle handle(comp_stream);
      gpu::SOLVERHandle solvehandle(comp_stream);

      std::size_t max_m1 = 0;
      for (std::size_t k=0; k<rb; k++)
        max_m1 = std::max(max_m1, B11.tilerows(k));
      auto max_m = max_m1;
      for (std::size_t k=0; k<rb2; k++)
        max_m = std::max(max_m, B21.tilerows(k));
      auto max_mn = max_m*max_m;

      gpu::HostMemory<scalar_t> pinned(max_mn);

      int getrf_work_size =
        gpu::getrf_buffersize<scalar_t>(solvehandle, max_m1);
      // max buffersize is not the buffersize of the largest matrix,
      // rectangular matrix seems to need more workspace than square
      int compress_lwork = 0;
      for (std::size_t i=0; i<rb; i++) {
        for (std::size_t j=0; j<rb; j++)
          compress_lwork =
            std::max(compress_lwork, gpu::gesvdj_buffersize<scalar_t>
                     (solvehandle, Jobz::V, B11.tilerows(i), B11.tilecols(j)));
        for (std::size_t j=0; j<rb2; j++) {
          compress_lwork =
            std::max(compress_lwork, gpu::gesvdj_buffersize<scalar_t>
                     (solvehandle, Jobz::V, B11.tilerows(i), B12.tilecols(j)));
          compress_lwork =
            std::max(compress_lwork, gpu::gesvdj_buffersize<scalar_t>
                     (solvehandle, Jobz::V, B21.tilerows(j), B11.tilecols(i)));
        }
      }
      // singular values (real and scalar), U, V and A
      compress_lwork += 2*max_m + 3*max_mn;

      int max_batchcount = std::pow(rb-1+rb2, 2);
      auto d_batch_meta = VBatchedGEMM<scalar_t>::dwork_bytes(max_batchcount);
      std::size_t max_work = std::max(getrf_work_size, compress_lwork);
      std::size_t d_mem_size =
        gpu::round_up(sizeof(scalar_t)*
                      (dsep*dsep + 2*dsep*dupd + max_work)) +
        gpu::round_up(sizeof(int)*(dsep+1)) +
        3*gpu::round_up(d_batch_meta);
      gpu::DeviceMemory<char> dmem(d_mem_size);
      auto dA11 = dmem.template as<scalar_t>();
      auto dA12 = dA11 + dsep*dsep;
      auto dA21 = dA12 + dsep*dupd;
      auto d_work_mem = dA21 + dupd*dsep;
      auto dpiv = gpu::aligned_ptr<int>(d_work_mem+max_work);
      auto dinfo = dpiv + dsep;
      auto d_batch_mem = gpu::aligned_ptr<char>(dinfo+1);

      std::size_t batch_scalar_size = 0;
      gpu::DeviceMemory<scalar_t> d_batch_scalar_mem;

      for (std::size_t i=0; i<rb; i++) {
        for (std::size_t j=0; j<rb; j++)
          B11.create_dense_tile_gpu(i, j, A11, dA11);
        for (std::size_t j=0; j<rb2; j++) {
          B12.create_dense_tile_gpu(i, j, A12, dA12);
          B21.create_dense_tile_gpu(j, i, A21, dA21);
        }
      }

      for (std::size_t i=0; i<rb; i++) {
        gpu::getrf(solvehandle, B11.tile(i, i).D(),
                   d_work_mem, dpiv+B11.tileroff(i), dinfo);
        for (std::size_t j=i+1; j<rb; j++) {
          if (admissible(i, j))
            B11.compress_tile_gpu
              (solvehandle, handle, i, j, B11.tile(i, j).D(),
               dinfo, d_work_mem, opts);
          B11.tile(i, j).laswp(handle, dpiv+B11.tileroff(i), true);
          trsm(handle, Side::L, UpLo::L, Trans::N, Diag::U,
               scalar_t(1.), B11.tile(i, i), B11.tile(i, j));
        }
        for (std::size_t j=i+1; j<rb; j++) {
          if (admissible(j, i))
            B11.compress_tile_gpu
              (solvehandle, handle, j, i, B11.tile(j, i).D(),
               dinfo, d_work_mem, opts);
          trsm(handle, Side::R, UpLo::U, Trans::N, Diag::N,
               scalar_t(1.), B11.tile(i, i), B11.tile(j, i));
        }
        // B12, B21 on GPU
        for (std::size_t j=0; j<rb2; j++) {
          B12.compress_tile_gpu
            (solvehandle, handle, i, j, B12.tile(i, j).D(),
             dinfo, d_work_mem, opts);
          B12.tile(i, j).laswp(handle, dpiv+B11.tileroff(i), true);
          trsm(handle, Side::L, UpLo::L, Trans::N, Diag::U,
               scalar_t(1.), B11.tile(i, i), B12.tile(i, j));
          B21.compress_tile_gpu
            (solvehandle, handle, j, i, B21.tile(j, i).D(),
             dinfo, d_work_mem, opts);
          trsm(handle, Side::R, UpLo::U, Trans::N, Diag::N,
               scalar_t(1.), B11.tile(i, i), B21.tile(j, i));
        }

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

        if (sVU + sUVU > batch_scalar_size) {
          batch_scalar_size = sVU + sUVU;
          d_batch_scalar_mem = gpu::DeviceMemory<scalar_t>(batch_scalar_size);
        }
        scalar_t* dVU = d_batch_scalar_mem;
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
#pragma omp task
          {
            for (std::size_t j=0; j<rb; j++)
              B11.tile(i, j).move_gpu_tile_to_cpu(copy_stream, pinned);
            for (std::size_t j=0; j<rb2; j++) {
              B12.tile(i, j).move_gpu_tile_to_cpu(copy_stream, pinned);
              B21.tile(j, i).move_gpu_tile_to_cpu(copy_stream, pinned);
            }
          }
        }
        comp_stream.synchronize();
        copy_stream.synchronize();
      }
      gpu::copy_device_to_host(piv.data(), dpiv, dsep);
      for (std::size_t i=0; i<rb; i++)
        for (std::size_t l=B11.tileroff(i); l<B11.tileroff(i+1); l++)
          piv[l] += B11.tileroff(i);
    }

    // explicit template instantiations
    template class BLRMatrix<float>;
    template class BLRMatrix<double>;
    template class BLRMatrix<std::complex<float>>;
    template class BLRMatrix<std::complex<double>>;

  } // end namespace BLR
} // end namespace strumpack
