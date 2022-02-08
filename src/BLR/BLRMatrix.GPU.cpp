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
#include <cassert>

#include "BLRMatrix.hpp"
#include "BLRTileBLAS.hpp"
#include "misc/TaskTimer.hpp"

#if defined(STRUMPACK_USE_CUDA)
#include "dense/CUDAWrapper.hpp"
#else
#if defined(STRUMPACK_USE_HIP)
#include "dense/HIPWrapper.hpp"
#endif
#endif

#if defined(STRUMPACK_USE_MAGMA)
#include "dense/MAGMAWrapper.hpp"
#endif

// #include "cuda_profiler_api.h"
// #include "cudaProfiler.h"

namespace strumpack {
  namespace BLR {

    uintptr_t round_to_16(uintptr_t p) { return (p + 15) & ~15; }
    uintptr_t round_to_16(void* p) {
      return round_to_16(reinterpret_cast<uintptr_t>(p));
    }

    template<typename scalar_t> class VBatchedGEMM {
    public:
      VBatchedGEMM(std::size_t B, char* dmem) : dmem_(dmem) { reserve(B); }
      void reserve(std::size_t B) {
        m_.reserve(B+1);  ldA_.reserve(B+1);  A_.reserve(B);
        n_.reserve(B+1);  ldB_.reserve(B+1);  B_.reserve(B);
        k_.reserve(B+1);  ldC_.reserve(B+1);  C_.reserve(B);
      }
      void add(int m, int n, int k,
               scalar_t* A, scalar_t* B, scalar_t* C) {
        add(m, n, k, A, m, B, k, C, m);
      }
      void add(int m, int n, int k, scalar_t* A, int ldA,
               scalar_t* B, int ldB, scalar_t* C, int ldC) {
        assert(ldA >= m && ldB >= k && ldC >= m);
        m_.push_back(m);  ldA_.push_back(ldA);  A_.push_back(A);
        n_.push_back(n);  ldB_.push_back(ldB);  B_.push_back(B);
        k_.push_back(k);  ldC_.push_back(ldC);  C_.push_back(C);
      }
      std::size_t count() { return m_.size(); }
      static std::size_t dwork_bytes(int batchcount) {
#if defined(STRUMPACK_USE_MAGMA)
        return round_to_16((batchcount+1)*6*sizeof(magma_int_t)) +
          round_to_16(batchcount*3*sizeof(scalar_t*));
#else
        return 0;
#endif
      }
#if defined(STRUMPACK_USE_MAGMA)
      void run(scalar_t alpha, scalar_t beta,
               magma_queue_t& q, gpu::Stream& s) {
        magma_int_t B = m_.size();
        if (!B) return;
        auto dimem = reinterpret_cast<magma_int_t*>(dmem_);
        auto dsmem = reinterpret_cast<scalar_t**>
          (dmem_ + round_to_16((B+1)*6*sizeof(magma_int_t)));

        std::vector<magma_int_t> imem((B+1)*6);
        auto iptr = imem.begin();
        std::copy(m_.begin(), m_.end(), iptr);   iptr += B+1;
        std::copy(n_.begin(), n_.end(), iptr);   iptr += B+1;
        std::copy(k_.begin(), k_.end(), iptr);   iptr += B+1;
        std::copy(ldA_.begin(), ldA_.end(), iptr);   iptr += B+1;
        std::copy(ldB_.begin(), ldB_.end(), iptr);   iptr += B+1;
        std::copy(ldC_.begin(), ldC_.end(), iptr);   iptr += B+1;
        gpu::copy_host_to_device_async(dimem, imem.data(), (B+1)*6, s);

        std::vector<scalar_t*> smem(B*3);
        auto sptr = smem.begin();
        std::copy(A_.begin(), A_.end(), sptr);   sptr += B;
        std::copy(B_.begin(), B_.end(), sptr);   sptr += B;
        std::copy(C_.begin(), C_.end(), sptr);   sptr += B;
        gpu::copy_host_to_device_async(dsmem, smem.data(), B*3, s);

        for (magma_int_t i=0; i<B; i++) {
          STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::gemm_flops(m_[i],n_[i],k_[i],alpha,beta));
          STRUMPACK_BYTES(sizeof(scalar_t)*blas::gemm_moves(m_[i],n_[i],k_[i]));
        }
        auto max_m = *std::max_element(m_.begin(), m_.end());
        auto max_n = *std::max_element(n_.begin(), n_.end());
        auto max_k = *std::max_element(k_.begin(), k_.end());
        gpu::magma::gemm_vbatched
          (MagmaNoTrans, MagmaNoTrans, dimem, dimem+(B+1), dimem+2*(B+1),
           alpha, dsmem, dimem+3*(B+1), dsmem+B, dimem+4*(B+1),
           beta, dsmem+2*B, dimem+5*(B+1), B, max_m, max_n, max_k, q);
      }
#endif
      void run(scalar_t alpha, scalar_t beta,
               gpu::BLASHandle& h) {
        std::size_t batchcount = m_.size();
        if (!batchcount) return;
        // gpu::synchronize();
        for (std::size_t i=0; i<batchcount; i++) {
          DenseMatrixWrapper<scalar_t>
            A(m_[i], k_[i], A_[i], ldA_[i]),
            B(k_[i], n_[i], B_[i], ldB_[i]),
            C(m_[i], n_[i], C_[i], ldC_[i]);
          gpu::gemm(h, Trans::N, Trans::N, alpha, A, B, beta, C);
        }
      }
    private:
#if defined(STRUMPACK_USE_MAGMA)
      std::vector<magma_int_t> m_, n_, k_, ldA_, ldB_, ldC_;
#else
      std::vector<int> m_, n_, k_, ldA_, ldB_, ldC_;
#endif
      std::vector<scalar_t*> A_, B_, C_;
      char* dmem_ = nullptr; // only needed for MAGMA
    };

    template<typename scalar_t>
    void BLRMatrix<scalar_t>::create_dense_gpu_tile
    (std::size_t i, std::size_t j, DenseM_t& A, DenseMW_t& dB) {
      block(i, j) = std::unique_ptr<DenseTile<scalar_t>>
        (new DenseTile<scalar_t>(dB));
      gpu::copy_device_to_device(tile(i, j).D(), tile(A, i, j));
    }

    template<typename scalar_t>
    void BLRMatrix<scalar_t>::move_dense_gpu_tile_to_cpu
    (std::size_t i, std::size_t j, DenseM_t& dD) {
      DenseM_t hD(dD.rows(), dD.cols());
      gpu::copy_device_to_host(hD, dD);
      block(i, j) = std::unique_ptr<DenseTile<scalar_t>>
        (new DenseTile<scalar_t>(hD));
    }

    template<typename scalar_t>
    void BLRMatrix<scalar_t>::move_LR_gpu_tile_to_cpu
    (std::size_t i, std::size_t j, DenseM_t& dU, DenseM_t& dV) {
      DenseM_t hU(dU.rows(), dU.cols());
      DenseM_t hV(dV.rows(), dV.cols());
      gpu::copy_device_to_host(hU, dU);
      gpu::copy_device_to_host(hV, dV);
      block(i, j) = std::unique_ptr<LRTile<scalar_t>>
        (new LRTile<scalar_t>(hU, hV));
    }  
#if defined(STRUMPACK_USE_CUDA)
    template<typename scalar_t> void
    BLRMatrix<scalar_t>::compress_tile_gpu
    (gpu::SOLVERHandle& handle, gpu::BLASHandle& blashandle, std::size_t i, 
     std::size_t j, DenseM_t& A, DenseM_t& dU, DenseM_t& dV, int* dpiv, 
     char* svd_mem, const Opts_t& opts) {
      if (dU.rows() != 0 && dV.rows() != 0) {
        using real_t = typename RealType<scalar_t>::value_type;
        std::size_t minmn = std::min(dU.rows(), dV.rows());
        auto dS = reinterpret_cast<real_t*>(svd_mem);
        std::vector<real_t> S_tmp;
        S_tmp.resize(minmn);
        int rank = 0;
        const double tol = opts.rel_tol();
        int gesvd_work_size = gpu::gesvd<scalar_t>(handle, Jobz::V, dS, 
                                A, dU, dV, dpiv, svd_mem, tol);
        gpu::copy_device_to_host(S_tmp.data(), dS, minmn);
        while(S_tmp[rank] >= tol){
          rank++;
        }
        if (rank*(dU.rows() + dV.rows()) < dU.rows()*dV.rows()){
          DenseMW_t dU_tmp(dU.rows(), rank, dU, 0, 0);
          auto d_V = reinterpret_cast<scalar_t*>(svd_mem);
          d_V += minmn + (A.rows() * A.cols()) + gesvd_work_size;
          DenseMW_t dV_T(rank, dV.rows(), d_V, rank);
          gpu::geam<scalar_t>(blashandle, Trans::C, Trans::N, 1.0, dV, 0.0, 
                              dV_T, dV_T);
          d_V += rank * dV.rows();
          gpu::copy_real_to_scalar<scalar_t>(d_V, dS, rank);
          gpu::dgmm<scalar_t>(blashandle, Side::L, dV_T, 
                              d_V, dV_T);
          scalar_t* dA = tile(i, j).D().data();
          DenseMW_t dAijU(tilerows(i), rank, dA, tilerows(i));
          dA += tilerows(i) * rank;
          DenseMW_t dAijV(rank, tilecols(j), dA, rank);
          dA += rank * tilecols(j);
          block(i, j) = std::unique_ptr<LRTile<scalar_t>>
          (new LRTile<scalar_t>(dAijU, dAijV));
          gpu::copy_device_to_device(tile(i, j).U(), dU_tmp);
          gpu::copy_device_to_device(tile(i, j).V(), dV_T);
        }
      }
    }
#endif
#if defined(STRUMPACK_USE_HIP)
    template<typename scalar_t> void
    BLRMatrix<scalar_t>::compress_tile_gpu_hip
    (gpu::SOLVERHandle& handle, gpu::BLASHandle& blashandle, std::size_t i, 
     std::size_t j, DenseM_t& A, DenseM_t& dU, DenseM_t& dV, int* dpiv, const Opts_t& opts) {
      if (dU.rows() != 0 && dV.cols() != 0) {
        using real_t = typename RealType<scalar_t>::value_type;
        std::size_t minmn = std::min(dU.rows(), dV.cols());
        gpu::DeviceMemory<real_t> d_S(minmn);
        real_t* dS = d_S;
        std::vector<real_t> S_tmp;
        S_tmp.resize(minmn);
        int rank = 0;
        const double tol = opts.rel_tol();
        gpu::gesvd_hip<scalar_t>(handle, dS, A, dU, dV, dpiv);
        gpu::copy_device_to_host(S_tmp.data(), dS, minmn);
        while(S_tmp[rank] >= tol){
          rank++;
        }
        if (rank*(dU.rows() + dV.cols()) < dU.rows()*dV.cols()){
          DenseMW_t dU_tmp(dU.rows(), rank, dU, 0, 0);
          gpu::DeviceMemory<scalar_t> d_V(rank*dV.cols());
          DenseMW_t dV_T(rank, dV.cols(), d_V, rank);
          gpu::DeviceMemory<scalar_t> d_x(rank);
          gpu::copy_real_to_scalar<scalar_t>(d_x, dS, rank);
          gpu::dgmm<scalar_t>(blashandle, Side::L, dV_T, 
                              d_x, dV_T);
          scalar_t* dA = tile(i, j).D().data();
          DenseMW_t dAijU(tilerows(i), rank, dA, tilerows(i));
          dA += tilerows(i) * rank;
          DenseMW_t dAijV(rank, tilecols(j), dA, rank);
          dA += rank * tilecols(j);
          block(i, j) = std::unique_ptr<LRTile<scalar_t>>
          (new LRTile<scalar_t>(dAijU, dAijV));
          gpu::copy_device_to_device(tile(i, j).U(), dU_tmp);
          gpu::copy_device_to_device(tile(i, j).V(), dV_T);
        }
      }
    }
#endif
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
      B11 = BLRMatrix<scalar_t>(A11.rows(), tiles1, A11.cols(), tiles1);
      B12 = BLRMatrix<scalar_t>(A12.rows(), tiles1, A12.cols(), tiles2);
      B21 = BLRMatrix<scalar_t>(A21.rows(), tiles2, A21.cols(), tiles1);
      piv.resize(B11.rows());
      auto rb = B11.rowblocks();
      auto rb2 = B21.rowblocks();
#if 1 // B11, B12, B21, B22 on GPU
        auto dsep = A11.rows();
        auto d2 = A22.rows();
        int nr_streams = 1;
        std::vector<gpu::Stream> streams(nr_streams);
        std::vector<gpu::SOLVERHandle> solvehandles(nr_streams);
        std::vector<gpu::BLASHandle> handles(nr_streams);
        for (int i=0; i<nr_streams; i++) {
          solvehandles[i].set_stream(streams[i]);
          handles[i].set_stream(streams[i]);
        }
#if defined(STRUMPACK_USE_MAGMA)
        magma_init();
        magma_queue_t q;
#if defined(STRUMPACK_USE_CUDA)
        gpu::Stream comp_stream;
        gpu::BLASHandle handle(comp_stream);
        magma_queue_create_from_cuda
          (0, comp_stream, handle, nullptr, &q);
#else
        magma_queue_create(0, &q);
#endif
#endif
        gpu::DeviceMemory<scalar_t> dmB11(dsep*dsep), dmB12(dsep*A12.cols()), 
                                    dmB21(A21.rows()*dsep);
        int getrf_work_size = gpu::getrf_buffersize<scalar_t>
           (solvehandles[0], *std::max_element(tiles1.begin(), tiles1.end()));
        gpu::DeviceMemory<scalar_t> getrf_work(getrf_work_size);
        gpu::DeviceMemory<int> dpiv(dsep+1);
        std::size_t max_m = 0, max_mn = 0;
        for (std::size_t k1=0; k1<rb; k1++) {
          max_m = std::max(max_m, B11.tilerows(k1));
          for (std::size_t k2 = 0; k2 < rb; k2++) {
            if (k1 != k2) 
              max_mn = std::max(max_mn, B11.tilerows(k1)*B11.tilecols(k2));
          }
        }
        std::size_t max_m12 = 0, max_n12 = 0, 
                    max_m21 = 0, max_n21 = 0;
        for (std::size_t k=0; k<rb; k++) {
          max_m12 = std::max(max_m12, B12.tilerows(k));
          max_n21 = std::max(max_n21, B21.tilecols(k));
        }
        for (std::size_t k=0; k<rb2; k++) {
          max_n12 = std::max(max_n12, B12.tilecols(k));
          max_m21 = std::max(max_m21, B21.tilerows(k));
        }
        std::size_t maxmn_all = std::max(max_mn, max_m12*max_n12),
                    maxm_all = std::max(max_m, max_m12);
        maxmn_all = std::max(maxmn_all, max_m21*max_n21);
        maxm_all = std::max(maxm_all, max_m21);
#if defined(STRUMPACK_USE_CUDA)
        gesvdjInfo_t params = nullptr;
        int gesvd_work_size = gpu::gesvdj_buffersize<scalar_t>
          (solvehandles[0], Jobz::V, maxm_all, maxm_all+1, params);
        int svd_size = round_to_16(sizeof(scalar_t) * (2*maxm_all + maxmn_all + gesvd_work_size));
        gpu::DeviceMemory<char> svd_mem(svd_size);
#endif
        gpu::DeviceMemory<scalar_t> dVU_tmp(2*max_mn), dVU12_tmp(2*max_m12*max_n12),
                                    dVU21_tmp(2*max_m21*max_n21);
        scalar_t* dVU = dVU_tmp, *dUVU = dVU + max_mn, *dVU12 = dVU12_tmp, 
                *dUVU12 = dVU12 + max_m12*max_n12, *dVU21 = dVU21_tmp, 
                *dUVU21 = dVU21 + max_m21*max_n21;
        gpu::DeviceMemory<scalar_t> d_U(max_mn);
        gpu::DeviceMemory<scalar_t> d_V(max_mn);
        gpu::DeviceMemory<scalar_t> d_U12(max_m12*max_n12);
        gpu::DeviceMemory<scalar_t> d_V12(max_m12*max_n12);
        gpu::DeviceMemory<scalar_t> d_U21(max_m21*max_n21);
        gpu::DeviceMemory<scalar_t> d_V21(max_m21*max_n21);
        scalar_t* dU = d_U, *dV = d_V;
        scalar_t* dU12 = d_U12, *dV12 = d_V12,
                 *dU21 = d_U21, *dV21 = d_V21;
        scalar_t* dA11 = dmB11, *dA12 = dmB12, *dA21 = dmB21;
        gpu::DeviceMemory<scalar_t> dmemA22(2*max_m21*max_n12);
        scalar_t* dVU22 = dmemA22, *dUVU22 = dVU22 + max_m21*max_n12;
        for (std::size_t i=0; i<rb; i++) {
          for (std::size_t j=0; j<rb; j++){
            DenseMW_t dAij(B11.tilerows(i), B11.tilecols(j), dA11, B11.tilerows(i));
            B11.create_dense_gpu_tile(i, j, A11, dAij);
            dA11 += B11.tilerows(i) * B11.tilecols(j);
          }
        }
        for (std::size_t i=0; i<rb; i++) {
          for (std::size_t j=0; j<rb2; j++){
            DenseMW_t dAij(B12.tilerows(i), B12.tilecols(j), dA12, B12.tilerows(i));
            B12.create_dense_gpu_tile(i, j, A12, dAij);
            dA12 += B12.tilerows(i) * B12.tilecols(j);
          }
        }
        for (std::size_t i=0; i<rb2; i++) {
          for (std::size_t j=0; j<rb; j++){
            DenseMW_t dAij(B21.tilerows(i), B21.tilecols(j), dA21, B21.tilerows(i));
            B21.create_dense_gpu_tile(i, j, A21, dAij);
            dA21 += B21.tilerows(i) * B21.tilecols(j);
          }
        }
        for (std::size_t i=0, s=0; i<rb; i++) {
          gpu::getrf(solvehandles[s], B11.tile(i, i).D(), 
                     reinterpret_cast<scalar_t*>(getrf_work_size),
                     dpiv+B11.tileroff(i), dpiv+dsep);
          for (std::size_t j=i+1; j<rb; j++) {
            if (admissible(i, j)) {
              std::size_t minmn = std::min(B11.tilerows(i), B11.tilecols(j));
              DenseMW_t dAijU(B11.tilerows(i), minmn, dU, B11.tilerows(i));
#if defined(STRUMPACK_USE_CUDA)
              DenseMW_t dAijV(B11.tilecols(j), minmn, dV, B11.tilecols(j));
              B11.compress_tile_gpu(solvehandles[s], handles[s], i, j, B11.tile(i, j).D(),
                                    dAijU, dAijV, dpiv+dsep, svd_mem, opts);
#else
#if defined(STRUMPACK_USE_HIP)
              DenseMW_t dAijV(minmn, B11.tilecols(j), dV, minmn);
              B11.compress_tile_gpu_hip(solvehandles[s], handles[s], i, j, B11.tile(i, j).D(),
                                    dAijU, dAijV, dpiv+dsep, opts);
#endif
#endif   
#if defined(STRUMPACK_USE_MAGMA)
              if (B11.tile(i, j).is_low_rank()){
                gpu::magma::laswpx(B11.tile(i, j).U(), dpiv+B11.tileroff(i), 
                                   q, true);
              } else {
                gpu::magma::laswpx(B11.tile(i, j).D(), dpiv+B11.tileroff(i), 
                                   q, true);
              }
#else
              if (B11.tile(i, j).is_low_rank()){
                gpu::laswp(solvehandles[s], B11.tile(i, j).U(), 1, 
                           B11.tile(i, j).U().rows(), dpiv+B11.tileroff(i), 1);
              } else{
                gpu::laswp(solvehandles[s], B11.tile(i, j).D(), 1, 
                           B11.tile(i, j).D().rows(), dpiv+B11.tileroff(i), 1);
              }
#endif
              if (B11.tile(i, j).is_low_rank()){
                gpu::trsm(handles[s], Side::L, UpLo::L, Trans::N, Diag::U,
                          scalar_t(1.), B11.tile(i, i).D(), B11.tile(i, j).U());
              } else {
                gpu::trsm(handles[s], Side::L, UpLo::L, Trans::N, Diag::U,
                          scalar_t(1.), B11.tile(i, i).D(), B11.tile(i, j).D());
              }
            } else {
#if defined(STRUMPACK_USE_MAGMA)
              gpu::magma::laswpx(B11.tile(i, j).D(), dpiv+B11.tileroff(i), 
                                 q, true);
#else
              gpu::laswp(solvehandles[s], B11.tile(i, j).D(), 1, 
                         B11.tile(i, j).D().rows(), dpiv+B11.tileroff(i), 1);
#endif
              gpu::trsm(handles[s], Side::L, UpLo::L, Trans::N, Diag::U,
                        scalar_t(1.), B11.tile(i, i).D(), B11.tile(i, j).D());
            }
          }
          for (std::size_t j=i+1; j<rb; j++) {
            if (admissible(j, i)) {
              std::size_t minmn = std::min(B11.tilerows(j), B11.tilecols(i));
              DenseMW_t dAijU(B11.tilerows(j), minmn, dU, B11.tilerows(j));
#if defined(STRUMPACK_USE_CUDA)
              DenseMW_t dAijV(B11.tilecols(i), minmn, dV, B11.tilecols(i));
              B11.compress_tile_gpu(solvehandles[s], handles[s], j, i, B11.tile(j, i).D(), 
                                    dAijU, dAijV, dpiv+dsep, svd_mem, opts);
#else
#if defined(STRUMPACK_USE_HIP)
              DenseMW_t dAijV(minmn, B11.tilecols(i), dV, minmn);
              B11.compress_tile_gpu_hip(solvehandles[s], handles[s], j, i, B11.tile(j, i).D(),
                                        dAijU, dAijV, dpiv+dsep, opts);
#endif
#endif 
              if (B11.tile(j, i).is_low_rank()){
                gpu::trsm(handles[s], Side::R, UpLo::U, Trans::N, Diag::N,
                          scalar_t(1.), B11.tile(i, i).D(), B11.tile(j, i).V());
              } else{
                gpu::trsm(handles[s], Side::R, UpLo::U, Trans::N, Diag::N,
                          scalar_t(1.), B11.tile(i, i).D(), B11.tile(j, i).D());
              }
            } else{
              gpu::trsm(handles[s], Side::R, UpLo::U, Trans::N, Diag::N,
                        scalar_t(1.), B11.tile(i, i).D(), B11.tile(j, i).D());
            }
          }
          //B12, B21 on GPU
          for (std::size_t j=0; j<rb2; j++) {
            std::size_t minmn = std::min(B12.tilerows(i), B12.tilecols(j));
            DenseMW_t dAijU(B12.tilerows(i), minmn, dU12, B12.tilerows(i));
#if defined(STRUMPACK_USE_CUDA)
            DenseMW_t dAijV(B12.tilecols(j), minmn, dV12, B12.tilecols(j));
            B12.compress_tile_gpu(solvehandles[s], handles[s], i, j, B12.tile(i, j).D(), 
                                  dAijU, dAijV, dpiv+dsep, svd_mem, opts);
#else
#if defined(STRUMPACK_USE_HIP)
            DenseMW_t dAijV(minmn, B12.tilecols(j), dV12, minmn);
            B12.compress_tile_gpu_hip(solvehandles[s], handles[s], i, j, B12.tile(i, j).D(),
                                      dAijU, dAijV, dpiv+dsep, opts);
#endif
#endif 
#if defined(STRUMPACK_USE_MAGMA)
            if (B12.tile(i, j).is_low_rank()){
              gpu::magma::laswpx(B12.tile(i, j).U(), dpiv+B11.tileroff(i), 
                                 q, true);
            } else {
              gpu::magma::laswpx(B12.tile(i, j).D(), dpiv+B11.tileroff(i), 
                                 q, true);
            }
#else
            if (B12.tile(i, j).is_low_rank()){
              gpu::laswp(solvehandles[s], B12.tile(i, j).U(), 1, 
                         B12.tile(i, j).U().rows(), dpiv+B11.tileroff(i), 1);
            } else{
              gpu::laswp(solvehandles[s], B12.tile(i, j).D(), 1, 
                         B12.tile(i, j).D().rows(), dpiv+B11.tileroff(i), 1);
            }
#endif
            if (B12.tile(i, j).is_low_rank()){
              gpu::trsm(handles[s], Side::L, UpLo::L, Trans::N, Diag::U,
                        scalar_t(1.), B11.tile(i, i).D(), B12.tile(i, j).U());
            } else{
              gpu::trsm(handles[s], Side::L, UpLo::L, Trans::N, Diag::U,
                        scalar_t(1.), B11.tile(i, i).D(), B12.tile(i, j).D());
            }
            minmn = std::min(B21.tilerows(j), B21.tilecols(i));
            DenseMW_t dAijU21(B21.tilerows(j), minmn, dU21, B21.tilerows(j));
#if defined(STRUMPACK_USE_CUDA)
            DenseMW_t dAijV21(B21.tilecols(i), minmn, dV21, B21.tilecols(i));
            B21.compress_tile_gpu(solvehandles[s], handles[s], j, i, B21.tile(j, i).D(), 
                                  dAijU21, dAijV21, dpiv+dsep, svd_mem, opts);
#else
#if defined(STRUMPACK_USE_HIP)
            DenseMW_t dAijV21(minmn, B21.tilecols(i), dV21, minmn);
            B21.compress_tile_gpu_hip(solvehandles[s], handles[s], j, i, B21.tile(j, i).D(),
                                      dAijU21, dAijV21, dpiv+dsep, opts);
#endif
#endif 
            if (B21.tile(j, i).is_low_rank()){
              gpu::trsm(handles[s], Side::R, UpLo::U, Trans::N, Diag::N,
                        scalar_t(1.), B11.tile(i, i).D(), B21.tile(j, i).V());
            } else{
              gpu::trsm(handles[s], Side::R, UpLo::U, Trans::N, Diag::N,
                        scalar_t(1.), B11.tile(i, i).D(), B21.tile(j, i).D());
            }
          }
          //GEMM B11
          for (std::size_t j=i+1; j<rb; j++) {
            auto& Tij = B11.tile(i, j);
            for (std::size_t k=i+1; k<rb; k++) {
              auto& Tki = B11.tile(k, i);
              if (Tki.is_low_rank()) {
                if (Tij.is_low_rank()) {
                  DenseMW_t VU(Tki.rank(), Tij.rank(), dVU, Tki.rank());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(1.), Tki.V(), Tij.U(), scalar_t(0.), VU);
                  if (Tij.rank() < Tki.rank()) {
                    DenseMW_t UVU(Tki.rows(), Tij.rank(), dUVU, Tki.rows());
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(1.), Tki.U(), VU, scalar_t(0.), UVU);
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(-1.), UVU, Tij.V(), scalar_t(1.), 
                              B11.tile(k, j).D());
                  } else {
                    DenseMW_t UVU(Tki.rank(), Tij.cols(), dUVU, Tki.rank());
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(1.), VU, Tij.V(), scalar_t(0.), UVU);
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(-1.), Tki.U(), UVU, scalar_t(1.), 
                              B11.tile(k, j).D());
                  }
                } else { // Tij is dense
                  DenseMW_t VU(Tki.rank(), Tij.cols(), dVU, Tki.rank());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(1.), Tki.V(), Tij.D(), scalar_t(0.), VU);
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(-1.), Tki.U(), VU, scalar_t(1.), 
                            B11.tile(k, j).D());
                }
              } else { // Tki is dense
                if (Tij.is_low_rank()) {
                  DenseMW_t VU(Tki.rows(), Tij.rank(), dVU, Tki.rows());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(1.), Tki.D(), Tij.U(), scalar_t(0.), VU);
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(-1.), VU, Tij.V(), scalar_t(1.), 
                            B11.tile(k, j).D());
                } else {
                  gpu::gemm(handles[s], Trans::N, Trans::N, scalar_t(-1.), 
                            Tki.D(), Tij.D(), scalar_t(1.), B11.tile(k, j).D());
                }
              }
            }
          }
          //GEMM B12
          for (std::size_t k=i+1; k<rb; k++) {
            auto& Tki = B11.tile(k, i);
            for (std::size_t j=0; j<rb2; j++) {
              auto& Tij = B12.tile(i, j);
              if (Tki.is_low_rank()) {
                if (Tij.is_low_rank()) {
                  DenseMW_t VU(Tki.rank(), Tij.rank(), dVU12, Tki.rank());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(1.), Tki.V(), Tij.U(), scalar_t(0.), VU);
                  if (Tij.rank() < Tki.rank()) {
                    DenseMW_t UVU(Tki.rows(), Tij.rank(), dUVU12, Tki.rows());
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(1.), Tki.U(), VU, scalar_t(0.), UVU);
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(-1.), UVU, Tij.V(), scalar_t(1.), 
                              B12.tile(k, j).D());
                  } else{
                    DenseMW_t UVU(Tki.rank(), Tij.cols(), dUVU12, Tki.rank());
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(1.), VU, Tij.V(), scalar_t(0.), UVU);
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(-1.), Tki.U(), UVU, scalar_t(1.), 
                              B12.tile(k, j).D());
                  }
                } else { // Tij is dense
                  DenseMW_t VU(Tki.rank(), Tij.cols(), dVU12, Tki.rank());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(1.), Tki.V(), Tij.D(), scalar_t(0.), VU);
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(-1.), Tki.U(), VU, scalar_t(1.), 
                            B12.tile(k, j).D());
                }
              } else { // Tki is dense
                if (Tij.is_low_rank()){
                  DenseMW_t VU(Tki.rows(), Tij.rank(), dVU12, Tki.rows());
                  gpu::gemm(handles[s], Trans::N, Trans::N, scalar_t(1.), 
                            Tki.D(), Tij.U(), scalar_t(0.), VU);
                  gpu::gemm(handles[s], Trans::N, Trans::N, scalar_t(-1.), 
                            VU, Tij.V(), scalar_t(1.), B12.tile(k, j).D());
                } else {
                  gpu::gemm(handles[s], Trans::N, Trans::N, scalar_t(-1.), 
                            Tki.D(), Tij.D(), scalar_t(1.), B12.tile(k, j).D());
                }
              }
            }
          }
          //GEMM B21
          for (std::size_t k=i+1; k<rb; k++) {
            auto& Tik = B11.tile(i, k);
            for (std::size_t j=0; j<rb2; j++) {
              auto& Tji = B21.tile(j, i);
              if (Tji.is_low_rank()) {
                if (Tik.is_low_rank()) {
                  DenseMW_t VU(Tji.rank(), Tik.rank(), dVU21, Tji.rank());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(1.), Tji.V(), Tik.U(), scalar_t(0.), VU);
                  if (Tik.rank() < Tji.rank()) {
                    DenseMW_t UVU(Tji.rows(), Tik.rank(), dUVU21, Tji.rows());
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(1.), Tji.U(), VU, scalar_t(0.), UVU);
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(-1.), UVU, Tik.V(), scalar_t(1.), 
                              B21.tile(j, k).D());
                  } else{
                    DenseMW_t UVU(Tji.rank(), Tik.cols(), dUVU21, Tji.rank());
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(1.), VU, Tik.V(), scalar_t(0.), UVU);
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(-1.), Tji.U(), UVU, scalar_t(1.), 
                              B21.tile(j, k).D());
                  }
                } else { // Tik is dense
                  DenseMW_t VU(Tji.rank(), Tik.cols(), dVU21, Tji.rank());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(1.), Tji.V(), Tik.D(), scalar_t(0.), VU);
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(-1.), Tji.U(), VU, scalar_t(1.), 
                            B21.tile(j, k).D());
                }
              } else { // Tji is dense
                if (Tik.is_low_rank()) {
                  DenseMW_t VU(Tji.rows(), Tik.rank(), dVU21, Tji.rows());
                  gpu::gemm(handles[s], Trans::N, Trans::N, scalar_t(1.), 
                            Tji.D(), Tik.U(), scalar_t(0.), VU);
                  gpu::gemm(handles[s], Trans::N, Trans::N, scalar_t(-1.), 
                            VU, Tik.V(), scalar_t(1.), B21.tile(j, k).D());
                } else {
                  gpu::gemm(handles[s], Trans::N, Trans::N, scalar_t(-1.), 
                            Tji.D(), Tik.D(), scalar_t(1.), B21.tile(j, k).D());
                }
              }
            }
          }
        }
        //GEMM B22
        for (std::size_t i=0, s=0; i<rb; i++) {
          for (std::size_t j=0; j<rb2; j++) {
            auto& Tij = B12.tile(i, j);
            for (std::size_t k=0; k<rb2; k++) {
              DenseMW_t dAkj(B21.tilerows(k), B12.tilecols(j), A22,
                             B21.tileroff(k), B12.tilecoff(j));
              auto& Tki = B21.tile(k, i);
              if (Tki.is_low_rank()) {
                if (Tij.is_low_rank()) {
                  DenseMW_t VU(Tki.rank(), Tij.rank(), dVU22, Tki.rank());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(1.), Tki.V(), Tij.U(), scalar_t(0.), VU);
                  if (Tij.rank() < Tki.rank()) {
                    DenseMW_t UVU(Tki.rows(), Tij.rank(), dUVU22, Tki.rows());
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(1.), Tki.U(), VU, scalar_t(0.), UVU);
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(-1.), UVU, Tij.V(), scalar_t(1.), dAkj);
                  } else {
                    DenseMW_t UVU(Tki.rank(), Tij.cols(), dUVU22, Tki.rank());
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(1.), VU, Tij.V(), scalar_t(0.), UVU);
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(-1.), Tki.U(), UVU, scalar_t(1.), dAkj);
                  }
                } else { // Tij is dense
                  DenseMW_t VU(Tki.rank(), Tij.cols(), dVU22, Tki.rank());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(1.), Tki.V(), Tij.D(), scalar_t(0.), VU);
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(-1.), Tki.U(), VU, scalar_t(1.), dAkj);
                }
              } else { // Tki is dense
                if (Tij.is_low_rank()) {
                  DenseMW_t VU(Tki.rows(), Tij.rank(), dVU22, Tki.rows());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(1.), Tki.D(), Tij.U(), scalar_t(0.), VU);
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(-1.), VU, Tij.V(), scalar_t(1.), dAkj);
                } else {
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(-1.), Tki.D(), Tij.D(), scalar_t(1.), dAkj);
                }
              }
            }
          }
        }
        gpu::copy_device_to_host(piv.data(), dpiv.as<int>(), B11.rows());
        for (std::size_t i=0; i<rb; i++)
          for (std::size_t l=B11.tileroff(i); l<B11.tileroff(i+1); l++)
            piv[l] += B11.tileroff(i);
        for (std::size_t i=0; i<rb; i++) {
          for (std::size_t j=0; j<rb; j++){
            if (B11.tile(i, j).is_low_rank()){
              B11.move_LR_gpu_tile_to_cpu(i, j, B11.tile(i, j).U(), B11.tile(i, j).V());
            } else{
              B11.move_dense_gpu_tile_to_cpu(i, j, B11.tile(i, j).D());
            }
          }
        }
        for (std::size_t i=0; i<rb; i++) {
          for (std::size_t j=0; j<rb2; j++){
            if (B12.tile(i, j).is_low_rank()){
              B12.move_LR_gpu_tile_to_cpu(i, j, B12.tile(i, j).U(), B12.tile(i, j).V());
            } else{
              B12.move_dense_gpu_tile_to_cpu(i, j, B12.tile(i, j).D());
            }
          }
        }
        for (std::size_t i=0; i<rb2; i++) {
          for (std::size_t j=0; j<rb; j++){
            if (B21.tile(i, j).is_low_rank()){
              B21.move_LR_gpu_tile_to_cpu(i, j, B21.tile(i, j).U(), B21.tile(i, j).V());
            } else{
              B21.move_dense_gpu_tile_to_cpu(i, j, B21.tile(i, j).D());
            }
          }
        }
        A11.clear();
        if(d2){
          A12.clear();
          A21.clear();
        }

#if defined(STRUMPACK_USE_MAGMA)
        magma_queue_destroy(q);
        magma_finalize();
#endif

#else // B11 on CPU, B22 on GPU
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
        auto lrb = rb+rb2;
        // dummy for task synchronization
        std::unique_ptr<int[]> B_(new int[lrb*lrb]()); auto B = B_.get();
#pragma omp taskgroup
#else
        int* B = nullptr;
#endif
        {
          for (std::size_t i=0; i<rb; i++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
            std::size_t ii = i+lrb*i;
#pragma omp task default(shared) firstprivate(i,ii) depend(inout:B[ii])
#endif
            {
              B11.create_dense_tile(i, i, A11);
              auto tpiv = B11.tile(i, i).LU();
              std::copy(tpiv.begin(), tpiv.end(),
                        piv.begin()+B11.tileroff(i));
            }
            for (std::size_t j=i+1; j<rb; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ij = i+lrb*j;
#pragma omp task default(shared) firstprivate(i,j,ij,ii)        \
  depend(in:B[ii]) depend(inout:B[ij]) priority(rb-j)
#endif
              { // these blocks have received all updates, compress now
                if (admissible(i, j)) B11.create_LR_tile(i, j, A11, opts);
                else B11.create_dense_tile(i, j, A11);
                // permute and solve with L, blocks right from the
                // diagonal block
                std::vector<int> tpiv
                  (piv.begin()+B11.tileroff(i),
                   piv.begin()+B11.tileroff(i+1));
                B11.tile(i, j).laswp(tpiv, true);
                trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                     scalar_t(1.), B11.tile(i, i), B11.tile(i, j));
              }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ji = j+lrb*i;
#pragma omp task default(shared) firstprivate(i,j,ji,ii)        \
  depend(in:B[ii]) depend(inout:B[ji]) priority(rb-j)
#endif
              {
                if (admissible(j, i)) B11.create_LR_tile(j, i, A11, opts);
                else B11.create_dense_tile(j, i, A11);
                // solve with U, the blocks under the diagonal block
                trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                     scalar_t(1.), B11.tile(i, i), B11.tile(j, i));
              }
            }
            for (std::size_t j=0; j<rb2; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ij2 = i+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(i,j,ij2,ii)       \
  depend(in:B[ii]) depend(inout:B[ij2])
#endif
              {
                B12.create_LR_tile(i, j, A12, opts);
                // permute and solve with L blocks right from the
                // diagonal block
                std::vector<int> tpiv
                  (piv.begin()+B11.tileroff(i),
                   piv.begin()+B11.tileroff(i+1));
                B12.tile(i, j).laswp(tpiv, true);
                trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                     scalar_t(1.), B11.tile(i, i), B12.tile(i, j));
              }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t j2i = (rb+j)+lrb*i;
#pragma omp task default(shared) firstprivate(i,j,j2i,ii)       \
  depend(in:B[ii]) depend(inout:B[j2i])
#endif
              {
                B21.create_LR_tile(j, i, A21, opts);
                // solve with U, the blocks under the diagonal block
                trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                     scalar_t(1.), B11.tile(i, i), B21.tile(j, i));
              }
            }
            for (std::size_t j=i+1; j<rb; j++) {
              for (std::size_t k=i+1; k<rb; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t ij = i+lrb*j, ki = k+lrb*i, kj = k+lrb*j;
#pragma omp task default(shared) firstprivate(i,j,k,ij,ki,kj)   \
  depend(in:B[ij],B[ki]) depend(inout:B[kj]) priority(rb-j)
#endif
                { // Schur complement updates, always into full rank
                  auto Akj = B11.tile(A11, k, j);
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       B11.tile(k, i), B11.tile(i, j), scalar_t(1.), Akj);
                }
              }
            }
            for (std::size_t k=i+1; k<rb; k++) {
              for (std::size_t j=0; j<rb2; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t ki = k+lrb*i, ij2 = i+lrb*(rb+j),
                  kj2 = k+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(i,k,j,ki,ij2,kj2) \
  depend(in:B[ki],B[ij2]) depend(inout:B[kj2])
#endif
                { // Schur complement updates, always into full rank
                  auto Akj = B12.tile(A12, k, j);
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       B11.tile(k, i), B12.tile(i, j), scalar_t(1.), Akj);
                }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t ik = i+lrb*k, j2i = (j+rb)+lrb*i,
                  j2k = (rb+j)+lrb*k;
#pragma omp task default(shared) firstprivate(i,k,j,ik,j2i,j2k) \
  depend(in:B[ik],B[j2i]) depend(inout:B[j2k])
#endif
                { // Schur complement updates, always into full rank
                  auto Ajk = B21.tile(A21, j, k);
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       B21.tile(j, i), B11.tile(i, k), scalar_t(1.), Ajk);
                }
              }
            }
          }
        }
      //}
      for (std::size_t i=0; i<rb; i++)
        for (std::size_t l=B11.tileroff(i); l<B11.tileroff(i+1); l++)
          piv[l] += B11.tileroff(i);
      A11.clear();
      A12.clear();
      A21.clear();

      auto d2 = A22.rows();
      if (d2) {
        TaskTimer t("BLR_Schur_GPU");
        t.start();
#if 0
        // cudaProfilerStart();
        // cuProfilerStart();
#pragma omp critical
        {
          gpu::Stream h2d_stream, d2h_stream, comp_stream;
          gpu::BLASHandle handle(comp_stream);
#if defined(STRUMPACK_USE_MAGMA)
          magma_init();
          magma_queue_t q;
#if defined(STRUMPACK_USE_CUDA)
          magma_queue_create_from_cuda
            (0, comp_stream, handle, nullptr, &q);
#else
          magma_queue_create(0, &q);
#endif
#endif
          std::size_t sU0 = 0, sV0 = 0;
          std::vector<std::size_t> sU1(rb2), sV1(rb2), sVU(rb2), sUVU(rb2);
          std::size_t lwork = 0;
          // count how much device memory will be required
          for (std::size_t i=0; i<rb; i++) {
            for (std::size_t k=0; k<rb2; k++) {
              auto& Tki = B21.tile(k, i);
              if (Tki.is_low_rank()) {
                sU0 += Tki.U().nonzeros();
                sV0 += Tki.V().nonzeros();
              } else sU0 += Tki.D().nonzeros();
            }
          }
          for (std::size_t j=0; j<rb2; j++) {
            for (std::size_t i=0; i<rb; i++) {
              auto& Tij = B12.tile(i, j);
              if (Tij.is_low_rank()) {
                sU1[j] += Tij.U().nonzeros();
                sV1[j] += Tij.V().nonzeros();
              } else sU1[j] += Tij.D().nonzeros();
              for (std::size_t k=0; k<rb2; k++) {
                auto& Tki = B21.tile(k, i);
                if (Tki.is_low_rank()) {
                  if (Tij.is_low_rank()) {
                    sVU[j] += Tki.rank() * Tij.rank();
                    sUVU[j] += (Tij.rank() < Tki.rank()) ?
                      Tki.rows() * Tij.rank() : Tki.rank() * Tij.cols();
                  } else sVU[j] += Tki.rank() * Tij.cols();
                } else if (Tij.is_low_rank()) sVU[j] += Tki.rows() * Tij.rank();
              }
            }
            lwork = std::max(lwork, sU1[j]+sV1[j]+sVU[j]+sUVU[j]);
          }
          lwork += sU0 + sV0;
          auto bdwork = VBatchedGEMM<scalar_t>::dwork_bytes(rb2);
          gpu::DeviceMemory<char> bdmem(bdwork*3 + (d2*d2+2*lwork)*sizeof(scalar_t));
          DenseMW_t dA22
            (d2, d2, reinterpret_cast<scalar_t*>(bdmem + bdwork*3), d2);
          for (std::size_t j=0; j<rb2; j++) {
            scalar_t *dU0 = dA22.end(), *dV0 = dU0 + sU0,
              *dU1 = dV0 + sV0, *dV1 = dU1 + sU1[j],
              *dVU = dV1 + sV1[j], *dUVU = dVU + sVU[j];
            gpu::copy_host_to_device_async
              (dA22.ptr(0, B12.tilecoff(j)), A22.ptr(0, B12.tilecoff(j)),
               d2*B12.tilecols(j), h2d_stream);
            for (std::size_t i=0; i<rb; i++) {
              VBatchedGEMM<scalar_t> b1(rb2, bdmem),
                b2(rb2, bdmem+bdwork), b3(rb2, bdmem+2*bdwork);
              auto& Tij = B12.tile(i, j);
              for (std::size_t k=0; k<rb2; k++) {
                auto& Tki = B21.tile(k, i);
                auto dAkj = dA22.ptr(B21.tileroff(k), B12.tilecoff(j));
                if (Tki.is_low_rank()) {
                  if (Tij.is_low_rank()) {
                    b1.add(Tki.rank(), Tij.rank(), Tki.cols(), dV0, dU1, dVU);
                    if (Tij.rank() < Tki.rank()) {
                      b2.add(Tki.rows(), Tij.rank(), Tki.rank(), dU0, dVU, dUVU);
                      b3.add(Tki.rows(), Tij.cols(), Tij.rank(), dUVU, Tki.rows(),
                             dV1, Tij.rank(), dAkj, d2);
                      dUVU += Tki.rows() * Tij.rank();
                    } else {
                      b2.add(Tki.rank(), Tij.cols(), Tij.rank(), dVU, dV1, dUVU);
                      b3.add(Tki.rows(), Tij.cols(), Tki.rank(), dU0, Tki.rows(),
                             dUVU, Tki.rank(), dAkj, d2);
                      dUVU += Tki.rank() * Tij.cols();
                    }
                    dVU += Tki.rank() * Tij.rank();
                  } else {
                    b1.add(Tki.rank(), Tij.cols(), Tki.cols(), dV0, dU1, dVU);
                    b3.add(Tki.rows(), Tij.cols(), Tki.rank(), dU0, Tki.rows(),
                           dVU, Tki.rank(), dAkj, d2);
                    dVU += Tki.rank() * Tij.cols();
                  }
                  if (j == 0) {
                    gpu::copy_host_to_device_async(dU0, Tki.U(), h2d_stream);
                    gpu::copy_host_to_device_async(dV0, Tki.V(), h2d_stream);
                  }
                  dU0 += Tki.U().nonzeros();
                  dV0 += Tki.V().nonzeros();
                } else {
                  if (Tij.is_low_rank()) {
                    b1.add(Tki.rows(), Tij.rank(), Tki.cols(), dU0, dU1, dVU);
                    b3.add(Tki.rows(), Tij.cols(), Tij.rank(), dVU, Tki.rows(),
                           dV1, Tij.rank(), dAkj, d2);
                    dVU += Tki.rows() * Tij.rank();
                  } else
                    b3.add(Tki.rows(), Tij.cols(), Tki.cols(), dU0, Tki.rows(),
                           dU1, Tki.cols(), dAkj, d2);
                  if (j == 0)
                    gpu::copy_host_to_device_async(dU0, Tki.D(), h2d_stream);
                  dU0 += Tki.D().nonzeros();
                }
              }
              if (Tij.is_low_rank()) {
                gpu::copy_host_to_device_async(dU1, Tij.U(), h2d_stream);
                gpu::copy_host_to_device_async(dV1, Tij.V(), h2d_stream);
                dU1 += Tij.U().nonzeros();
                dV1 += Tij.V().nonzeros();
              } else {
                gpu::copy_host_to_device_async(dU1, Tij.D(), h2d_stream);
                dU1 += Tij.D().nonzeros();
              }
              comp_stream.synchronize();
              h2d_stream.synchronize();
#if defined(STRUMPACK_USE_MAGMA)
              b1.run(scalar_t(1.), scalar_t(0.), q, comp_stream);
              b2.run(scalar_t(1.), scalar_t(0.), q, comp_stream);
              b3.run(scalar_t(-1.), scalar_t(1.), q, comp_stream);
#else
              b1.run(scalar_t(1.), scalar_t(0.), handle);
              b2.run(scalar_t(1.), scalar_t(0.), handle);
              b3.run(scalar_t(-1.), scalar_t(1.), handle);
#endif
            }
            comp_stream.synchronize();
            gpu::copy_device_to_host_async
              (A22.ptr(0, B12.tilecoff(j)), dA22.ptr(0, B12.tilecoff(j)),
               d2*B12.tilecols(j), d2h_stream);
          }
          gpu::synchronize();
#if defined(STRUMPACK_USE_MAGMA)
          magma_queue_destroy(q);
          magma_finalize();
#endif
        }
        // cudaProfilerStop();
        // cuProfilerStop();
#else

#if 1
        int nr_streams = 4;
        std::vector<gpu::Stream> streams(nr_streams);
        std::vector<gpu::BLASHandle> handles(nr_streams);
        for (int i=0; i<nr_streams; i++)
          handles[i].set_stream(streams[i]);
        std::size_t max_m = 0, max_n = 0;
        for (std::size_t k=0; k<rb2; k++) {
          max_m = std::max(max_m, B21.tilerows(k));
          max_n = std::max(max_n, B12.tilecols(k));
        }
        std::size_t sU0 = 0, sV0 = 0, sU1 = 0, sV1 = 0;
        // count how much device memory will be required
        for (std::size_t i=0; i<rb; i++) {
          for (std::size_t k=0; k<rb2; k++) {
            auto& Tki = B21.tile(k, i);
            if (Tki.is_low_rank()) {
              sU0 += Tki.U().nonzeros();
              sV0 += Tki.V().nonzeros();
            } else sU0 += Tki.D().nonzeros();
            auto& Tik = B12.tile(i, k);
            if (Tik.is_low_rank()) {
              sU1 += Tik.U().nonzeros();
              sV1 += Tik.V().nonzeros();
            } else sU1 += Tik.D().nonzeros();
          }
        }
        gpu::DeviceMemory<scalar_t> dmemUV
          (d2*d2+sU0+sV0+sU1+sV1+2*max_m*max_n);
        DenseMW_t dA22(d2, d2, dmemUV, d2);
        scalar_t* dU0 = dmemUV+d2*d2, *dV0 = dU0 + sU0,
          *dU1 = dV0 + sV0, *dV1 = dU1 + sU1,
          *dVU = dV1 + sV1, *dUVU = dVU + max_m*max_n;
        for (std::size_t i=0, s=0; i<rb; i++) {
          for (std::size_t k=0; k<rb2; k++) {
            auto& Tki = B21.tile(k, i);
            if (Tki.is_low_rank()) {
              gpu::copy_host_to_device_async(dU0, Tki.U(), streams[s]);
              gpu::copy_host_to_device_async(dV0, Tki.V(), streams[s]);
              dU0 += Tki.U().nonzeros();
              dV0 += Tki.V().nonzeros();
            } else {
              gpu::copy_host_to_device_async(dU0, Tki.D(), streams[s]);
              dU0 += Tki.D().nonzeros();
            }
            auto& Tik = B12.tile(i, k);
            if (Tik.is_low_rank()) {
              gpu::copy_host_to_device_async(dU1, Tik.U(), streams[s]);
              gpu::copy_host_to_device_async(dV1, Tik.V(), streams[s]);
              dU1 += Tik.U().nonzeros();
              dV1 += Tik.V().nonzeros();
            } else {
              gpu::copy_host_to_device_async(dU1, Tik.D(), streams[s]);
              dU1 += Tik.D().nonzeros();
            }
            s = (s + 1) % nr_streams;
          }
        }
        gpu::synchronize();
        dU0 = dmemUV + d2*d2;  dV0 = dU0 + sU0;
        dU1 = dV0 + sV0;       dV1 = dU1 + sU1;
        for (std::size_t i=0, s=0; i<rb; i++) {
          auto dU0i = dU0;
          auto dV0i = dV0;
          for (std::size_t j=0; j<rb2; j++) {
            dU0 = dU0i;
            dV0 = dV0i;
            auto& Tij = B12.tile(i, j);
            for (std::size_t k=0; k<rb2; k++) {
              DenseMW_t dAkj(B21.tilerows(k), B12.tilecols(j), dA22,
                             B21.tileroff(k), B12.tilecoff(j));
              if (i == 0) {
                DenseMW_t Akj(B21.tilerows(k), B12.tilecols(j), A22,
                              B21.tileroff(k), B12.tilecoff(j));
                gpu::copy_host_to_device_async(dAkj, Akj, streams[s]);
              }
              auto& Tki = B21.tile(k, i);
              if (Tki.is_low_rank()) {
                DenseMW_t U0(Tki.rows(), Tki.rank(), dU0, Tki.rows()),
                  V0(Tki.rank(), Tki.cols(), dV0, Tki.rank());
                if (Tij.is_low_rank()) {
                  DenseMW_t U1(Tij.rows(), Tij.rank(), dU1, Tij.rows()),
                    V1(Tij.rank(), Tij.cols(), dV1, Tij.rank()),
                    VU(Tki.rank(), Tij.rank(), dVU, Tki.rank());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(1.), V0, U1, scalar_t(0.), VU);
                  if (Tij.rank() < Tki.rank()) {
                    DenseMW_t UVU(Tki.rows(), Tij.rank(), dUVU, Tki.rows());
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(1.), U0, VU, scalar_t(0.), UVU);
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(-1.), UVU, V1, scalar_t(1.), dAkj);
                  } else {
                    DenseMW_t UVU(Tki.rank(), Tij.cols(), dUVU, Tki.rank());
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(1.), VU, V1, scalar_t(0.), UVU);
                    gpu::gemm(handles[s], Trans::N, Trans::N,
                              scalar_t(-1.), U0, UVU, scalar_t(1.), dAkj);
                  }
                } else { // Tij is dense
                  DenseMW_t D1(Tij.rows(), Tij.cols(), dU1, Tij.rows()),
                    VU(Tki.rank(), Tij.cols(), dVU, Tki.rank());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(1.), V0, D1, scalar_t(0.), VU);
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(-1.), U0, VU, scalar_t(1.), dAkj);
                }
                dU0 += Tki.U().nonzeros();
                dV0 += Tki.V().nonzeros();
              } else { // Tki is dense
                DenseMW_t D0(Tki.rows(), Tki.cols(), dU0, Tki.rows());
                if (Tij.is_low_rank()) {
                  DenseMW_t U1(Tij.rows(), Tij.rank(), dU1, Tij.rows()),
                    V1(Tij.rank(), Tij.cols(), dV1, Tij.rank()),
                    VU(Tki.rows(), Tij.rank(), dVU, Tki.rows());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(1.), D0, U1, scalar_t(0.), VU);
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(-1.), VU, V1, scalar_t(1.), dAkj);
                } else {
                  DenseMW_t U1(Tij.rows(), Tij.cols(), dU1, Tij.rows());
                  gpu::gemm(handles[s], Trans::N, Trans::N,
                            scalar_t(-1.), D0, U1, scalar_t(1.), dAkj);
                }
                dU0 += Tki.D().nonzeros();
              }
              if (i == rb-1) {
                DenseMW_t Akj(B21.tilerows(k), B12.tilecols(j), A22,
                              B21.tileroff(k), B12.tilecoff(j));
                gpu::copy_device_to_host_async(Akj, dAkj, streams[s]);
              }
              s = (s + 1) % nr_streams;
            }
            if (Tij.is_low_rank()) {
              dU1 += Tij.U().nonzeros();
              dV1 += Tij.V().nonzeros();
            } else
              dU1 += Tij.D().nonzeros();
          }
        }
        gpu::synchronize();
        // gpu::copy_device_to_host_async(A22, dA22, d2h_stream);
        // gpu::synchronize();
#else
        for (std::size_t i=0; i<rb; i++) {
#pragma omp taskloop collapse(2) grainsize(1) default(shared)
          for (std::size_t j=0; j<rb2; j++) {
            for (std::size_t k=0; k<rb2; k++) {
              DenseMW_t Akj
                (B21.tilerows(k), B12.tilecols(j), A22,
                 B21.tileroff(k), B12.tilecoff(j));
              gemm(Trans::N, Trans::N, scalar_t(-1.),
                   B21.tile(k, i), B12.tile(i, j), scalar_t(1.), Akj);
            }
          }
        }
#endif
#endif
        auto time = t.elapsed();
        std::cout << "#   BLR GPU Schur time = "
                  << time << " sec" << std::endl;
      }
#endif
    }


    // explicit template instantiations
    template class BLRMatrix<float>;
    template class BLRMatrix<double>;
    template class BLRMatrix<std::complex<float>>;
    template class BLRMatrix<std::complex<double>>;

  } // end namespace BLR
} // end namespace strumpack
