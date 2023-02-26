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
/*! \file BLRTileBLAS.hpp
 * \brief Contains BLAS routines on BLRTiles.
 */
#ifndef BLR_TILE_BLAS_HPP
#define BLR_TILE_BLAS_HPP

#include <cassert>

#include "LRTile.hpp"
#include "DenseTile.hpp"

#if defined(STRUMPACK_USE_CUDA) || defined(STRUMPACK_USE_HIP)
// for gpu::round_up
#include "sparse/fronts/FrontalMatrixGPUKernels.hpp"
#endif

namespace strumpack {
  namespace BLR {

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const BLRTile<scalar_t>& a,
         const BLRTile<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c) {
      a.gemm_a(ta, tb, alpha, b, beta, c);
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const BLRTile<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c, int task_depth) {
      b.gemm_b(ta, tb, alpha, a, beta, c);
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const BLRTile<scalar_t>& a,
         const DenseMatrix<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c, int task_depth) {
      a.gemm_a(ta, tb, alpha, b, beta, c, task_depth);
    }

    template<typename scalar_t> void
    trsm(Side s, UpLo ul, Trans ta, Diag d, scalar_t alpha,
         const BLRTile<scalar_t>& a, BLRTile<scalar_t>& b) {
      b.trsm_b(s, ul, ta, d, alpha, a.D());
    }

    template<typename scalar_t> void
    trsm(Side s, UpLo ul, Trans ta, Diag d, scalar_t alpha,
         const BLRTile<scalar_t>& a, DenseMatrix<scalar_t>& b,
         int task_depth) {
      trsm(s, ul, ta, d, alpha, a.D(), b, task_depth);
    }
#if defined(STRUMPACK_USE_CUDA) || defined(STRUMPACK_USE_HIP)
    template<typename scalar_t> void
    trsm(gpu::BLASHandle& handle, Side s, UpLo ul, Trans ta, Diag d,
         scalar_t alpha, BLRTile<scalar_t>& a, BLRTile<scalar_t>& b) {
      b.trsm_b(handle, s, ul, ta, d, alpha, a.D());
    }
#endif

    template<typename scalar_t> void Schur_update_col
    (std::size_t j, const BLRTile<scalar_t>& a,
     const BLRTile<scalar_t>& b, scalar_t* c, scalar_t* work) {
      a.Schur_update_col_a(j, b, c, work);
    }

    template<typename scalar_t> void Schur_update_row
    (std::size_t i, const BLRTile<scalar_t>& a,
     const BLRTile<scalar_t>& b, scalar_t* c, scalar_t* work) {
      a.Schur_update_row_a(i, b, c, work);
    }

    template<typename scalar_t> void Schur_update_cols
    (const std::vector<std::size_t>& cols, const BLRTile<scalar_t>& a,
     const BLRTile<scalar_t>& b, DenseMatrix<scalar_t>& c, scalar_t* work) {
      a.Schur_update_cols_a(cols, b, c, work);
    }

    template<typename scalar_t> void Schur_update_rows
    (const std::vector<std::size_t>& rows, const BLRTile<scalar_t>& a,
     const BLRTile<scalar_t>& b, DenseMatrix<scalar_t>& c, scalar_t* work) {
      a.Schur_update_rows_a(rows, b, c, work);
    }

#if defined(STRUMPACK_USE_CUDA) || defined(STRUMPACK_USE_HIP)
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
      void add(int m, int n, int k,
               scalar_t* A, scalar_t* B, scalar_t* C, int ldC) {
        add(m, n, k, A, m, B, k, C, ldC);
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
        return
          gpu::round_up((batchcount+1)*6*sizeof(magma_int_t)) +
          gpu::round_up(batchcount*3*sizeof(scalar_t*));
#else
        return 0;
#endif
      }

      void run(scalar_t alpha, scalar_t beta,
               // magma_queue_t& q,
               gpu::Stream& s,
               gpu::BLASHandle& h) {
#if defined(STRUMPACK_USE_MAGMA)
        magma_int_t batchcount = m_.size();
        if (!batchcount) return;
        auto dimem = reinterpret_cast<magma_int_t*>(dmem_);
        auto dsmem = reinterpret_cast<scalar_t**>
          (dmem_ + gpu::round_up((batchcount+1)*6*sizeof(magma_int_t)));

        // TODO HostMemory?
        std::vector<magma_int_t> imem((batchcount+1)*6);
        auto iptr = imem.begin();
        std::copy(m_.begin(), m_.end(), iptr);   iptr += batchcount+1;
        std::copy(n_.begin(), n_.end(), iptr);   iptr += batchcount+1;
        std::copy(k_.begin(), k_.end(), iptr);   iptr += batchcount+1;
        std::copy(ldA_.begin(), ldA_.end(), iptr);   iptr += batchcount+1;
        std::copy(ldB_.begin(), ldB_.end(), iptr);   iptr += batchcount+1;
        std::copy(ldC_.begin(), ldC_.end(), iptr);   iptr += batchcount+1;
        gpu_check(gpu::copy_host_to_device_async
                  (dimem, imem.data(), (batchcount+1)*6, s));

        std::vector<scalar_t*> smem(batchcount*3);
        auto sptr = smem.begin();
        std::copy(A_.begin(), A_.end(), sptr);   sptr += batchcount;
        std::copy(B_.begin(), B_.end(), sptr);   sptr += batchcount;
        std::copy(C_.begin(), C_.end(), sptr);   sptr += batchcount;
        gpu_check(gpu::copy_host_to_device_async
                  (dsmem, smem.data(), batchcount*3, s));

        for (magma_int_t i=0; i<batchcount; i++) {
          STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*
                          blas::gemm_flops(m_[i],n_[i],k_[i],alpha,beta));
          STRUMPACK_BYTES(sizeof(scalar_t)*
                          blas::gemm_moves(m_[i],n_[i],k_[i]));
        }
        auto max_m = *std::max_element(m_.begin(), m_.end());
        auto max_n = *std::max_element(n_.begin(), n_.end());
        auto max_k = *std::max_element(k_.begin(), k_.end());
        gpu::magma::gemm_vbatched_max_nocheck
          (MagmaNoTrans, MagmaNoTrans,
           dimem, dimem+(batchcount+1), dimem+2*(batchcount+1),
           alpha, dsmem, dimem+3*(batchcount+1),
           dsmem+batchcount, dimem+4*(batchcount+1),
           beta, dsmem+2*batchcount, dimem+5*(batchcount+1),
           batchcount, max_m, max_n, max_k, h);
#else
        std::size_t batchcount = m_.size();
        if (!batchcount) return;
        // gpu::synchronize();
        for (std::size_t i=0; i<batchcount; i++) {
          STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*
                          blas::gemm_flops(m_[i],n_[i],k_[i],alpha,beta));
          STRUMPACK_BYTES(sizeof(scalar_t)*
                          blas::gemm_moves(m_[i],n_[i],k_[i]));
          DenseMatrixWrapper<scalar_t>
            A(m_[i], k_[i], A_[i], ldA_[i]),
            B(k_[i], n_[i], B_[i], ldB_[i]),
            C(m_[i], n_[i], C_[i], ldC_[i]);
          gpu::gemm(h, Trans::N, Trans::N, alpha, A, B, beta, C);
        }
#endif
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
#endif

  } // end namespace BLR
} // end namespace strumpack

#endif // BLR_TILE_BLAS_HPP
