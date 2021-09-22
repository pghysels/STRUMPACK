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

#if defined(STRUMPACK_USE_CUDA)
#include "dense/CUDAWrapper.hpp"
#else
#if defined(STRUMPACK_USE_HIP)
#include "dense/HIPWrapper.hpp"
#endif
#endif

namespace strumpack {
  namespace BLR {

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::construct_and_partial_factor_gpu
    (DenseMatrix<scalar_t>& A11, DenseMatrix<scalar_t>& A12,
     DenseMatrix<scalar_t>& A21, DenseMatrix<scalar_t>& A22,
     BLRMatrix<scalar_t>& B11, std::vector<int>& piv,
     BLRMatrix<scalar_t>& B12, BLRMatrix<scalar_t>& B21,
     const std::vector<std::size_t>& tiles1,
     const std::vector<std::size_t>& tiles2,
     const DenseMatrix<bool>& admissible,
     const Opts_t& opts) {

      std::cout << "In construct_and_partial_factor_gpu!!" << std::endl;

      B11 = BLRMatrix<scalar_t>(A11.rows(), tiles1, A11.cols(), tiles1);
      B12 = BLRMatrix<scalar_t>(A12.rows(), tiles1, A12.cols(), tiles2);
      B21 = BLRMatrix<scalar_t>(A21.rows(), tiles2, A21.cols(), tiles1);
      piv.resize(B11.rows());
      auto rb = B11.rowblocks();
      auto rb2 = B21.rowblocks();

      for (std::size_t i=0; i<rb; i++) {
        B11.create_dense_tile(i, i, A11);
        auto tpiv = B11.tile(i, i).LU();
        std::copy(tpiv.begin(), tpiv.end(), piv.begin()+B11.tileroff(i));
        for (std::size_t j=i+1; j<rb; j++) {
          // these blocks have received all updates, compress now
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
          if (admissible(j, i)) B11.create_LR_tile(j, i, A11, opts);
          else B11.create_dense_tile(j, i, A11);
          // solve with U, the blocks under the diagonal block
          trsm(Side::R, UpLo::U, Trans::N, Diag::N,
               scalar_t(1.), B11.tile(i, i), B11.tile(j, i));
        }
        for (std::size_t j=0; j<rb2; j++) {
          B12.create_LR_tile(i, j, A12, opts);
          // permute and solve with L blocks right from the
          // diagonal block
          std::vector<int> tpiv
            (piv.begin()+B11.tileroff(i), piv.begin()+B11.tileroff(i+1));
          B12.tile(i, j).laswp(tpiv, true);
          trsm(Side::L, UpLo::L, Trans::N, Diag::U,
               scalar_t(1.), B11.tile(i, i), B12.tile(i, j));
          B21.create_LR_tile(j, i, A21, opts);
          // solve with U, the blocks under the diagonal block
          trsm(Side::R, UpLo::U, Trans::N, Diag::N,
               scalar_t(1.), B11.tile(i, i), B21.tile(j, i));
        }


        /// TODO do these GEMMs on the GPU! //////////////////////////////////
        // use for instance magmablas_dgemm_vbatched()
        // or an alternative in cuBLAS??
        //
        // A complication here is that the A and B arguments here can
        // be LRTiles instead of regular dense matrices.  So we will
        // need a sequence of 3 calls to magmablas_dgemm_vbatched()
        //
        // A11, A12, A21, A22 need to be copied to the GPU (at the
        // beginning of this routine?).  The blocks of A11, A12, A21
        // that have received all updates need to be copied back. A22
        // can be copied back at the end of this routine.
        for (std::size_t j=i+1; j<rb; j++) {
          for (std::size_t k=i+1; k<rb; k++) {
            // Schur complement updates, always into full rank
            auto Akj = B11.tile(A11, k, j);
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                 B11.tile(k, i), B11.tile(i, j), scalar_t(1.), Akj);
          }
        }
        for (std::size_t k=i+1; k<rb; k++) {
          for (std::size_t j=0; j<rb2; j++) {
            // Schur complement updates, always into full rank
            auto Akj = B12.tile(A12, k, j);
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                 B11.tile(k, i), B12.tile(i, j), scalar_t(1.), Akj);
            // Schur complement updates, always into full rank
            auto Ajk = B21.tile(A21, j, k);
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                 B21.tile(j, i), B11.tile(i, k), scalar_t(1.), Ajk);
          }
        }
        for (std::size_t j=0; j<rb2; j++) {
          for (std::size_t k=0; k<rb2; k++) {
            // Schur complement updates, always into full rank
            DenseMatrixWrapper<scalar_t> Akj
              (B21.tilerows(k), B12.tilecols(j), A22,
               B21.tileroff(k), B12.tilecoff(j));
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                 B21.tile(k, i), B12.tile(i, j), scalar_t(1.), Akj);
          }
        }
        //////////////////////////////////////////////////////////////////////


      }
      for (std::size_t i=0; i<rb; i++)
        for (std::size_t l=B11.tileroff(i); l<B11.tileroff(i+1); l++)
          piv[l] += B11.tileroff(i);
      A11.clear();
      A12.clear();
      A21.clear();
    }


    // explicit template instantiations
    template class BLRMatrix<float>;
    template class BLRMatrix<double>;
    template class BLRMatrix<std::complex<float>>;
    template class BLRMatrix<std::complex<double>>;

  } // end namespace BLR
} // end namespace strumpack
