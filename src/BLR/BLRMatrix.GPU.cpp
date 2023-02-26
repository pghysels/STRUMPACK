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
      if (A.is_low_rank()) {
        if (B.is_low_rank()) {
          b1.add(A.rank(), B.rank(), A.cols(),
                 A.V().data(), B.U().data(), d1);
          if (B.rank() < A.rank()) {
            b2.add(A.rows(), B.rank(), A.rank(),
                   A.U().data(), d1, d2);
            b3.add(A.rows(), B.cols(), B.rank(),
                   d2, B.V().data(), C.data(), C.ld());
            d2 += A.rows() * B.rank();
          } else {
            b2.add(A.rank(), B.cols(), B.rank(),
                   d1, B.V().data(), d2);
            b3.add(A.rows(), B.cols(), A.rank(),
                   A.U().data(), d2, C.data(), C.ld());
            d2 += A.rank() * B.cols();
          }
          d1 += A.rank() * B.rank();
        } else {
          b1.add(A.rank(), B.cols(), A.cols(),
                 A.V().data(), B.D().data(), d1);
          b3.add(A.rows(), B.cols(), A.rank(),
                 A.U().data(), d1, C.data(), C.ld());
          d1 += A.rank() * B.cols();
        }
      } else {
        if (B.is_low_rank()) {
          b1.add(A.rows(), B.rank(), A.cols(),
                 A.D().data(), B.U().data(), d1);
          b3.add(A.rows(), B.cols(), B.rank(),
                 d1, B.V().data(), C.data(), C.ld());
          d1 += A.rows() * B.rank();
        } else
          b3.add(A.rows(), B.cols(), A.cols(),
                 A.D().data(), B.D().data(), C.data(), C.ld());
      }
    }


    template<typename scalar_t>
    void BLRMatrix<scalar_t>::create_dense_gpu_tile
    (std::size_t i, std::size_t j, DenseM_t& A, DenseMW_t& dB) {
      block(i, j) = std::unique_ptr<DenseTile<scalar_t>>
        (new DenseTile<scalar_t>(dB));
      gpu_check(gpu::copy_device_to_device(tile(i, j).D(), tile(A, i, j)));
    }

    template<typename scalar_t> int
    compress_buffersize(gpu::SOLVERHandle& solvehandle,
                        std::size_t max_m, std::size_t max_mn) {
#if defined(STRUMPACK_USE_CUDA)
#if defined(STRUMPACK_USE_MAGMA)
#if defined(STRUMPACK_USE_KBLAS)
      int svd_size =
        gpu::round_up(sizeof(scalar_t)*(2*max_mn+3)) +
        gpu::round_up(sizeof(int)*3);
#else
      int svd_size =
        gpu::round_up(sizeof(scalar_t)*2*max_m);
#endif
#else
      gesvdjInfo_t params = nullptr;
      int gesvd_work_size = gpu::gesvdj_buffersize<scalar_t>
        (solvehandle, Jobz::V, max_m, max_m+1, params);
      int svd_size =
        gpu::round_up(sizeof(scalar_t) *
                      (2*max_m+2*max_mn+gesvd_work_size));
#endif
#else
#if defined(STRUMPACK_USE_HIP)
      int svd_size = 0;
#endif
#endif
      return svd_size;
    }


#if defined(STRUMPACK_USE_CUDA)
    template<typename scalar_t> void
    BLRMatrix<scalar_t>::compress_tile_CUDA
    (gpu::SOLVERHandle& handle, gpu::BLASHandle& blashandle,
     std::size_t i, std::size_t j, DenseM_t& A,
     scalar_t* d_U, scalar_t* d_V, int* dpiv, char* svd_mem,
     const Opts_t& opts) {
      if (tilerows(i) != 0 && tilecols(j) != 0) {
        std::size_t minmn = std::min(tilerows(i), tilecols(j));
        DenseMW_t dU(tilerows(i), minmn, d_U, tilerows(i)),
          dV(tilecols(j), minmn, d_V, tilecols(j));
        auto dS = gpu::aligned_ptr<real_t>(svd_mem);
        std::vector<real_t> S_tmp;
        S_tmp.resize(minmn);
        int rank = 0;
        const double tol = opts.rel_tol();
        int gesvd_work_size = gpu::gesvd<scalar_t>
          (handle, Jobz::V, dS, A, dU, dV, dpiv, svd_mem, tol);
        gpu_check(gpu::copy_device_to_host(S_tmp.data(), dS, minmn));
        while (S_tmp[rank] >= tol) rank++;
        if (rank*(dU.rows() + dV.rows()) < dU.rows()*dV.rows()) {
          DenseMW_t dU_tmp(dU.rows(), rank, dU, 0, 0);
          auto d_V = gpu::aligned_ptr<scalar_t>(svd_mem);
          d_V += minmn + (A.rows() * A.cols()) + gesvd_work_size;
          DenseMW_t dV_T(rank, dV.rows(), d_V, rank);
          gpu::geam<scalar_t>
            (blashandle, Trans::C, Trans::N, 1.0, dV, 0.0, dV_T, dV_T);
          d_V += rank * dV.rows();
          gpu_check(gpu::copy_real_to_scalar<scalar_t>(d_V, dS, rank));
          gpu::dgmm<scalar_t>(blashandle, Side::L, dV_T, d_V, dV_T);
          scalar_t* dA = tile(i, j).D().data();
          DenseMW_t dAijU(tilerows(i), rank, dA, tilerows(i));
          dA += tilerows(i) * rank;
          DenseMW_t dAijV(rank, tilecols(j), dA, rank);
          dA += rank * tilecols(j);
          block(i, j) = std::unique_ptr<LRTile<scalar_t>>
            (new LRTile<scalar_t>(dAijU, dAijV));
          gpu_check(gpu::copy_device_to_device(tile(i, j).U(), dU_tmp));
          gpu_check(gpu::copy_device_to_device(tile(i, j).V(), dV_T));
        }
      }
    }
#endif

#if defined(STRUMPACK_USE_MAGMA)
    template<typename scalar_t> void
    BLRMatrix<scalar_t>::compress_tile_MAGMA
    (gpu::SOLVERHandle& handle, gpu::BLASHandle& blashandle,
     std::size_t i, std::size_t j, DenseM_t& A,
     scalar_t* d_U, scalar_t* d_V, int* dpiv, char* svd_mem,
     const Opts_t& opts) {
      if (tilerows(i) != 0 && tilecols(j) != 0) {
        std::size_t minmn = std::min(tilerows(i), tilecols(j));
        DenseMW_t dU(tilerows(i), minmn, d_U, tilerows(i)),
          dV(minmn, tilecols(j), d_V, minmn);
        auto dS = reinterpret_cast<real_t*>(svd_mem);
        std::vector<real_t> S_tmp;
        S_tmp.resize(minmn);
        int rank = 0;
        const double tol = opts.rel_tol();
        gpu::magma::gesvd_magma<scalar_t>
          (MagmaSomeVec, MagmaSomeVec, dS, A, dU, dV);
        gpu_check(gpu::copy_device_to_host(S_tmp.data(), dS, minmn));
        while (S_tmp[rank] >= tol) rank++;
        if (rank*(dU.rows() + dV.cols()) < dU.rows()*dV.cols()) {
          DenseMW_t dU_tmp(dU.rows(), rank, dU, 0, 0),
            dV_tmp(rank, dV.cols(), dV, 0, 0);
          auto d_x = reinterpret_cast<scalar_t*>(svd_mem);
          d_x += minmn;
          gpu_check(gpu::copy_real_to_scalar<scalar_t>(d_x, dS, rank));
          gpu::dgmm<scalar_t>(blashandle, Side::L, dV_tmp, d_x, dV_tmp);
          scalar_t* dA = tile(i, j).D().data();
          DenseMW_t dAijU(tilerows(i), rank, dA, tilerows(i));
          dA += tilerows(i) * rank;
          DenseMW_t dAijV(rank, tilecols(j), dA, rank);
          dA += rank * tilecols(j);
          block(i, j) = std::unique_ptr<LRTile<scalar_t>>
            (new LRTile<scalar_t>(dAijU, dAijV));
          gpu_check(gpu::copy_device_to_device(tile(i, j).U(), dU_tmp));
          gpu_check(gpu::copy_device_to_device(tile(i, j).V(), dV_tmp));
        }
      }
    }
#endif

#if defined(STRUMPACK_USE_KBLAS)
    template<typename scalar_t> void
    BLRMatrix<scalar_t>::compress_tile_KBLAS
    (gpu::SOLVERHandle& handle, gpu::BLASHandle& blashandle,
     std::size_t i, std::size_t j, DenseM_t& A,
     scalar_t* d_U, scalar_t* d_V, int* dpiv, char* svd_mem,
     const Opts_t& opts) {
      if (tilerows(i) > 15 && tilecols(j) > 15) {
        std::size_t minmn = std::min(tilerows(i), tilecols(j));
        DenseMW_t dU(tilerows(i), minmn, d_U, tilerows(i));
        DenseMW_t dV(tilecols(j), minmn, d_V, tilecols(j));
        auto dmemA = reinterpret_cast<scalar_t*>(svd_mem);
        DenseMW_t dA(A.rows(), A.cols(), dmemA, A.rows());
        gpu_check(gpu::copy_device_to_device(dA, A));
        const double tol = opts.rel_tol();
        int rows = dA.rows(), cols = dA.cols();
        auto drows = reinterpret_cast<int*>(svd_mem);
        drows += A.rows()*A.cols();
        gpu_check(gpu::copy_host_to_device(drows, &rows, 1));
        auto dcols = reinterpret_cast<int*>(svd_mem);
        dcols += A.rows()*A.cols() + 1;
        gpu_check(gpu::copy_host_to_device(dcols, &cols, 1));
        scalar_t* hA_ptr = dA.data();
        auto dApptr = reinterpret_cast<scalar_t**>(svd_mem);
        dApptr += A.rows()*A.cols() + 2;
        scalar_t** dA_pptr = dApptr;
        gpu_check(gpu::copy_host_to_device(dA_pptr, &hA_ptr, 1));
        scalar_t* hU_ptr = dU.data();
        auto dUpptr = reinterpret_cast<scalar_t**>(svd_mem);
        dUpptr += A.rows()*A.cols() + 3;
        scalar_t** dU_pptr = dUpptr;
        gpu_check(gpu::copy_host_to_device(dU_pptr, &hU_ptr, 1));
        scalar_t* hV_ptr = dV.data();
        auto dVpptr = reinterpret_cast<scalar_t**>(svd_mem);
        dVpptr += A.rows()*A.cols() + 4;
        scalar_t** dV_pptr = dVpptr;
        gpu_check(gpu::copy_host_to_device(dV_pptr, &hV_ptr, 1));
        int rank = 0;
        auto drank = reinterpret_cast<int*>(svd_mem);
        drank += A.rows()*A.cols() + 5;
        gpu::kblas::ara(blashandle.kblas_handle(), drows, dcols,
                        dA_pptr, drows, dU_pptr, drows, dV_pptr,
                        dcols, drank, tol, rows, cols,
                        minmn, 32, 10, blashandle.kblas_rand_state(), 1, 1);
        gpu_check(gpu::copy_device_to_host(&rank, drank, 1));
        STRUMPACK_FLOPS(blas::ara_flops(rows, cols, rank, 10));
        if (rank*(dU.rows() + dV.rows()) < dU.rows()*dV.rows()) {
          DenseMW_t dU_tmp(dU.rows(), rank, dU, 0, 0);
          auto d_V = reinterpret_cast<scalar_t*>(svd_mem);
          d_V += A.rows()*A.cols() + 6;
          DenseMW_t dV_tmp(rank, dV.rows(), d_V, rank);
          gpu::geam<scalar_t>
            (blashandle, Trans::C, Trans::N, 1.0, dV, 0.0, dV_tmp, dV_tmp);
          scalar_t* dA_tmp = tile(i, j).D().data();
          DenseMW_t dAijU(tilerows(i), rank, dA_tmp, tilerows(i));
          dA_tmp += tilerows(i) * rank;
          DenseMW_t dAijV(rank, tilecols(j), dA_tmp, rank);
          dA_tmp += rank * tilecols(j);
          block(i, j) = std::unique_ptr<LRTile<scalar_t>>
            (new LRTile<scalar_t>(dAijU, dAijV));
          gpu_check(gpu::copy_device_to_device(tile(i, j).U(), dU_tmp));
          gpu_check(gpu::copy_device_to_device(tile(i, j).V(), dV_tmp));
        }
      }
    }
#endif

#if defined(STRUMPACK_USE_HIP)
    template<typename scalar_t> void
    BLRMatrix<scalar_t>::compress_tile_HIP
    (gpu::SOLVERHandle& handle, gpu::BLASHandle& blashandle,
     std::size_t i, std::size_t j, DenseM_t& A,
     scalar_t* d_U, scalar_t* d_V, int* dpiv, char* svd_mem,
     const Opts_t& opts) {
      if (tilerows(i) != 0 && tilecols(j) != 0) {
        using real_t = typename RealType<scalar_t>::value_type;
        std::size_t minmn = std::min(tilerows(i), tilecols(j));
        DenseMW_t dU(tilerows(i), minmn, d_U, tilerows(i));
        DenseMW_t dV(minmn, tilecols(j), d_V, minmn);
        gpu::DeviceMemory<real_t> d_S(minmn);
        real_t* dS = d_S;
        std::vector<real_t> S_tmp;
        S_tmp.resize(minmn);
        int rank = 0;
        const double tol = opts.rel_tol();
        gpu::gesvd_hip<scalar_t>(handle, dS, A, dU, dV, dpiv);
        gpu_check(gpu::copy_device_to_host(S_tmp.data(), dS, minmn));
        while (S_tmp[rank] >= tol) rank++;
        if (rank*(dU.rows() + dV.cols()) < dU.rows()*dV.cols()){
          DenseMW_t dU_tmp(dU.rows(), rank, dU, 0, 0);
          DenseMW_t dV_tmp(rank, dV.cols(), dV, 0, 0);
          gpu::DeviceMemory<scalar_t> d_x(rank);
          gpu_check(gpu::copy_real_to_scalar<scalar_t>(d_x, dS, rank));
          gpu::dgmm<scalar_t>(blashandle, Side::L, dV_tmp, d_x, dV_tmp);
          scalar_t* dA = tile(i, j).D().data();
          DenseMW_t dAijU(tilerows(i), rank, dA, tilerows(i));
          dA += tilerows(i) * rank;
          DenseMW_t dAijV(rank, tilecols(j), dA, rank);
          dA += rank * tilecols(j);
          block(i, j) = std::unique_ptr<LRTile<scalar_t>>
          (new LRTile<scalar_t>(dAijU, dAijV));
          gpu_check(gpu::copy_device_to_device(tile(i, j).U(), dU_tmp));
          gpu_check(gpu::copy_device_to_device(tile(i, j).V(), dV_tmp));
        }
      }
    }
#endif

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::compress_tile_gpu
    (gpu::SOLVERHandle& handle, gpu::BLASHandle& blashandle,
     std::size_t i, std::size_t j, DenseM_t& A,
     scalar_t* d_U, scalar_t* d_V, int* dpiv, char* svd_mem,
     const Opts_t& opts) {
#if defined(STRUMPACK_USE_CUDA)
#if defined(STRUMPACK_USE_MAGMA)
#if defined(STRUMPACK_USE_KBLAS)
      compress_tile_KBLAS
        (handle, blashandle, i, j, A, d_U, d_V, dpiv, svd_mem, opts);
#else //CUDA && MAGMA && !KBLAS
      compress_tile_MAGMA
        (handle, blashandle, i, j, A, d_U, d_V, dpiv, svd_mem, opts);
#endif //end KBLAS
#else //CUDA && !MAGMA
      compress_tile_CUDA
        (handle, blashandle, i, j, A, d_U, d_V, dpiv, svd_mem, opts);
#endif //end MAGMA
#else //!CUDA
#if defined(STRUMPACK_USE_HIP)
      compress_tile_HIP
        (handle, blashandle, i, j, A, d_U, d_V, dpiv, svd_mem, opts);
#endif
#endif
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

#if defined(STRUMPACK_USE_KBLAS)
      const int BLOCK_SIZE = 32;
      kblas_ara_batch_wsquery<real_t>
        (handle.kblas_handle(), BLOCK_SIZE, 1);
      kblasAllocateWorkspace(handle.kblas_handle());
#endif

      std::size_t max_m1 = 0;
      for (std::size_t k=0; k<rb; k++)
        max_m1 = std::max(max_m1, B11.tilerows(k));
      auto max_m = max_m1;
      for (std::size_t k=0; k<rb2; k++)
        max_m = std::max(max_m, B21.tilerows(k));
      auto max_mn = max_m*max_m;
      std::size_t compress_work_bytes =
        compress_buffersize<scalar_t>(solvehandle, max_m, max_mn);
      std::size_t getrf_work_size =
        gpu::getrf_buffersize<scalar_t>(solvehandle, max_m1);
      std::size_t d_mem_size =
        gpu::round_up(compress_work_bytes) +
        gpu::round_up(sizeof(scalar_t)*
                      (dsep*dsep + 2*dsep*dupd + 2*max_mn +
                       getrf_work_size)) +
        gpu::round_up(sizeof(int)*(dsep+1));
      gpu::DeviceMemory<char> dmem(d_mem_size);
      char* compress_work_mem = dmem; // gpu::aligned_ptr<char>()
      auto dA11 = gpu::aligned_ptr<scalar_t>(compress_work_mem+compress_work_bytes);
      auto dA12 = dA11 + dsep*dsep;
      auto dA21 = dA12 + dsep*dupd;
      auto d_U = dA21 + dupd*dsep;
      auto d_V = d_U + max_mn;
      auto getrf_work_mem = d_V + max_mn;
      auto dpiv = gpu::aligned_ptr<int>(getrf_work_mem+getrf_work_size);
      auto dgetrf_err = dpiv + dsep;

      for (std::size_t i=0; i<rb; i++)
        for (std::size_t j=0; j<rb; j++) {
          DenseMW_t dAij(B11.tilerows(i), B11.tilecols(j),
                         dA11, B11.tilerows(i));
          B11.create_dense_gpu_tile(i, j, A11, dAij);
          dA11 += B11.tilerows(i) * B11.tilecols(j);
        }
      for (std::size_t i=0; i<rb; i++)
        for (std::size_t j=0; j<rb2; j++) {
          DenseMW_t dAij(B12.tilerows(i), B12.tilecols(j),
                         dA12, B12.tilerows(i));
          B12.create_dense_gpu_tile(i, j, A12, dAij);
          dA12 += B12.tilerows(i) * B12.tilecols(j);
        }
      for (std::size_t i=0; i<rb2; i++)
        for (std::size_t j=0; j<rb; j++) {
          DenseMW_t dAij(B21.tilerows(i), B21.tilecols(j),
                         dA21, B21.tilerows(i));
          B21.create_dense_gpu_tile(i, j, A21, dAij);
          dA21 += B21.tilerows(i) * B21.tilecols(j);
        }

      for (std::size_t i=0; i<rb; i++) {
        gpu::getrf(solvehandle, B11.tile(i, i).D(),
                   getrf_work_mem, dpiv+B11.tileroff(i), dgetrf_err);
        for (std::size_t j=i+1; j<rb; j++) {
          if (admissible(i, j))
            B11.compress_tile_gpu
              (solvehandle, handle, i, j, B11.tile(i, j).D(),
               d_U, d_V, dpiv+dsep, compress_work_mem, opts);
          B11.tile(i, j).laswp(handle, dpiv+B11.tileroff(i), true);
          trsm(handle, Side::L, UpLo::L, Trans::N, Diag::U,
               scalar_t(1.), B11.tile(i, i), B11.tile(i, j));
        }
        for (std::size_t j=i+1; j<rb; j++) {
          if (admissible(j, i))
            B11.compress_tile_gpu
              (solvehandle, handle, j, i, B11.tile(j, i).D(),
               d_U, d_V, dpiv+dsep, compress_work_mem, opts);
          trsm(handle, Side::R, UpLo::U, Trans::N, Diag::N,
               scalar_t(1.), B11.tile(i, i), B11.tile(j, i));
        }
        // B12, B21 on GPU
        for (std::size_t j=0; j<rb2; j++) {
          B12.compress_tile_gpu
            (solvehandle, handle, i, j, B12.tile(i, j).D(),
             d_U, d_V, dpiv+dsep, compress_work_mem, opts);
          B12.tile(i, j).laswp(handle, dpiv+B11.tileroff(i), true);
          trsm(handle, Side::L, UpLo::L, Trans::N, Diag::U,
               scalar_t(1.), B11.tile(i, i), B12.tile(i, j));
          B21.compress_tile_gpu
            (solvehandle, handle, j, i, B21.tile(j, i).D(),
             d_U, d_V, dpiv+dsep, compress_work_mem, opts);
          trsm(handle, Side::R, UpLo::U, Trans::N, Diag::N,
               scalar_t(1.), B11.tile(i, i), B21.tile(j, i));
        }

        // Schur complement update
        std::size_t sVU = 0, sUVU = 0;
        int batchcount = std::pow(rb-(i+1)+rb2, 2);
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

        // TODO move allocation out of the loop
        auto bdwork = VBatchedGEMM<scalar_t>::dwork_bytes(batchcount);
        // TODO align
        gpu::DeviceMemory<char> bdmem(bdwork*3 + (sVU+sUVU)*sizeof(scalar_t));
        auto dVU = gpu::aligned_ptr<scalar_t>(bdmem + bdwork*3);
        auto dUVU = dVU + sVU;
        VBatchedGEMM<scalar_t> b1(batchcount, bdmem),
          b2(batchcount, bdmem+bdwork), b3(batchcount, bdmem+2*bdwork);

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

        // TODO overlap this with copy back of the computed F11, F12, F21
        b1.run(scalar_t(1.), scalar_t(0.), comp_stream, handle);
        b2.run(scalar_t(1.), scalar_t(0.), comp_stream, handle);
        b3.run(scalar_t(-1.), scalar_t(1.), comp_stream, handle);
        comp_stream.synchronize();
      }
      gpu::synchronize();
      gpu::copy_device_to_host(piv.data(), dpiv, dsep);
      for (std::size_t i=0; i<rb; i++) {
        for (std::size_t l=B11.tileroff(i); l<B11.tileroff(i+1); l++)
          piv[l] += B11.tileroff(i);
        for (std::size_t j=0; j<rb; j++)
          B11.tile(i, j).move_gpu_tile_to_cpu(copy_stream);
        for (std::size_t j=0; j<rb2; j++) {
          B12.tile(i, j).move_gpu_tile_to_cpu(copy_stream);
          B21.tile(j, i).move_gpu_tile_to_cpu(copy_stream);
        }
      }
      gpu::synchronize();
      A11.clear();
      if (dupd) {
        A12.clear();
        A21.clear();
      }
    }

    // explicit template instantiations
    template class BLRMatrix<float>;
    template class BLRMatrix<double>;
    template class BLRMatrix<std::complex<float>>;
    template class BLRMatrix<std::complex<double>>;

  } // end namespace BLR
} // end namespace strumpack
