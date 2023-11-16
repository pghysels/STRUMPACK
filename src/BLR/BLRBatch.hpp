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
/*! \file BLRBatch.hpp
 * \brief Contains Batch routines on BLRTiles.
 */
#ifndef BLR_BATCH_HPP
#define BLR_BATCH_HPP

#include <cassert>

#include "misc/Tools.hpp"
#include "BLRTileBLAS.hpp"
#if defined(STRUMPACK_USE_MAGMA)
#include "dense/MAGMAWrapper.hpp"
#endif

namespace strumpack {
  namespace BLR {

    template<typename scalar_t> class VBatchedGEMM {
    public:
      VBatchedGEMM(std::size_t B, char* dmem);
      void add(int m, int n, int k,
               scalar_t* A, scalar_t* B, scalar_t* C);
      void add(int m, int n, int k,
               scalar_t* A, scalar_t* B, scalar_t* C, int ldC);
      void add(int m, int n, int k, scalar_t* A, int ldA,
               scalar_t* B, int ldB, scalar_t* C, int ldC);

      static std::size_t dwork_bytes(int batchcount);

      void run(scalar_t alpha, scalar_t beta, gpu::Stream& s, gpu::Handle& h);

    private:
#if defined(STRUMPACK_USE_MAGMA)
      std::vector<magma_int_t> m_, n_, k_, ldA_, ldB_, ldC_;
#else
      std::vector<int> m_, n_, k_, ldA_, ldB_, ldC_;
#endif
      std::vector<scalar_t*> A_, B_, C_;
      char* dmem_ = nullptr; // only needed for MAGMA
    };


    template<typename scalar_t> void
    multiply_inc_work_size(const BLRTile<scalar_t>& A,
                           const BLRTile<scalar_t>& B,
                           std::size_t& temp1, std::size_t& temp2) {
      if (A.is_low_rank()) {
        if (B.is_low_rank()) {
          temp1 += A.rank() * B.rank();
          temp2 += (B.rank() < A.rank()) ?
            A.rows() * B.rank() : A.rank() * B.cols();
        } else temp1 += A.rank() * B.cols();
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

    template<typename scalar_t> class VBatchedTRSMLeftRight {
      using DenseM_t = DenseMatrix<scalar_t>;
    public:
      void add(DenseM_t& A, DenseM_t& Bl, DenseM_t& Br);
      void run(gpu::Handle& h, VectorPool<scalar_t>& workspace);

    private:
      std::vector<DenseM_t*> A_, Bl_, Br_;
    };

    template<typename scalar_t> class VBatchedTRSM {
      using DenseM_t = DenseMatrix<scalar_t>;
    public:
      void add(DenseM_t& A, DenseM_t& B);
      void run(gpu::Handle& h, VectorPool<scalar_t>& workspace,
               bool left);

    private:
      std::vector<DenseM_t*> A_, B_;
    };

    template<typename scalar_t> class VBatchedARA {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using real_t = typename RealType<scalar_t>::value_type;

    public:
      void add(std::unique_ptr<BLRTile<scalar_t>>& tile);
      void run(gpu::Handle& handle, VectorPool<scalar_t>& workspace,
               real_t tol);

      static void kblas_wsquery(gpu::Handle& handle, int batchcount);

    private:
      std::vector<std::unique_ptr<BLRTile<scalar_t>>*> tile_;

      void run_kblas(gpu::Handle& handle, VectorPool<scalar_t>& workspace,
                     real_t tol);
      void run_magma(gpu::Handle& handle, VectorPool<scalar_t>& workspace,
                     real_t tol);
      void run_svd(gpu::Handle& handle, VectorPool<scalar_t>& workspace,
                   real_t tol);
      void compress(gpu::Handle& handle,
                    std::unique_ptr<BLRTile<scalar_t>>& t,
                    scalar_t* work, int* dinfo, real_t tol);

      static const int KBLAS_ARA_BLOCK_SIZE;
    };

  } // end namespace BLR
} // end namespace strumpack

#endif // BLR_BATCH_HPP
