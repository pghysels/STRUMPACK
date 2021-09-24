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

#if defined(STRUMPACK_USE_MAGMA)
#include "dense/MAGMAWrapper.hpp"
#endif

namespace strumpack {
  namespace BLR {

    template<typename scalar_t> //typename integer_t=magma_int_t>
    class VBatchedMeta {
    public:
      VBatchedMeta(std::size_t B) { reserve(B); }
      void reserve(std::size_t B) {
        m_.reserve(B+1);  ldA_.reserve(B+1);  A_.reserve(B);
        n_.reserve(B+1);  ldB_.reserve(B+1);  B_.reserve(B);
        k_.reserve(B+1);  ldC_.reserve(B+1);  C_.reserve(B);
      }
      void add(int m, int n, int k,
               scalar_t*& A, scalar_t*& B, scalar_t*& C) {
        m_.push_back(m);  ldA_.push_back(m);  A_.push_back(A);
        n_.push_back(n);  ldB_.push_back(k);  B_.push_back(B);
        k_.push_back(k);  ldC_.push_back(m);  C_.push_back(C);
      }
      void clear() {
        m_.clear();  ldA_.clear();  A_.clear();
        n_.clear();  ldB_.clear();  B_.clear();
        k_.clear();  ldC_.clear();  C_.clear();
      }
      std::size_t count() { return m_.size(); }
      void run(scalar_t alpha, scalar_t beta, magma_queue_t q) {
        magma_int_t batchcount = m_.size();
        if (!batchcount) return;
        m_.push_back(0);  ldA_.push_back(0);
        n_.push_back(0);  ldB_.push_back(0);
        k_.push_back(0);  ldC_.push_back(0);

        // gpu::magma::gemm_vbatched
        //   (MagmaNoTrans, MagmaNoTrans, m_.data(), n_.data(), k_.data(),
        //    alpha, A_.data(), ldA_.data(), B_.data(), ldB_.data(),
        //    beta, C_.data(), ldC_.data(), batchcount, q);

        gpu::Stream s;
        gpu::BLASHandle h;
        h.set_stream(s);
        for (magma_int_t i=0; i<batchcount; i++) {
          DenseMatrixWrapper<scalar_t>
            A(m_[i], k_[i], A_[i], ldA_[i]),
            B(k_[i], n_[i], B_[i], ldB_[i]),
            C(m_[i], n_[i], C_[i], ldC_[i]);
          gpu::gemm(h, Trans::N, Trans::N, alpha, A, B, beta, C);
        }
        gpu::synchronize();
      }
    private:
      std::vector<magma_int_t> m_, n_, k_, ldA_, ldB_, ldC_;
      std::vector<scalar_t*> A_, B_, C_;
    };


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
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
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

        // TODO also do these GEMMs on the GPU?
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
      }
      A11.clear();
      A12.clear();
      A21.clear();

      // magma_init();
      magma_queue_t q;
      // magma_queue_create(0, &q);

      auto d2 = A22.rows();
      if (d2) {
        std::vector<std::size_t> sU0(rb), sV0(rb),
          sU1(rb), sV1(rb), sVU(rb), sUVU(rb);
        std::size_t lwork = 0;
        for (std::size_t i=0; i<rb; i++) {
          for (std::size_t k=0; k<rb2; k++) {
            auto& Tk = B21.tile(k, i);
            if (Tk.is_low_rank()) {
              sU0[i] += Tk.U().nonzeros();
              sV0[i] += Tk.V().nonzeros();
            } else sU0[i] += Tk.D().nonzeros();
          }
          for (std::size_t j=0; j<rb2; j++) {
            auto& Tj = B12.tile(i, j);
            if (Tj.is_low_rank()) {
              sU1[i] += Tj.U().nonzeros();
              sV1[i] += Tj.V().nonzeros();
            } else sU1[i] += Tj.D().nonzeros();
            for (std::size_t k=0; k<rb2; k++) {
              auto& Tk = B21.tile(k, i);
              if (Tk.is_low_rank()) {
                if (Tj.is_low_rank()) {
                  sVU[i] += Tk.rank() * Tj.rank();
                  sUVU[i] += (Tj.rank() < Tk.rank()) ?
                    Tk.rows() * Tj.rank() : Tk.rank() * Tj.cols();
                } else sVU[i] += Tk.rank() * Tj.cols();
              } else if (Tj.is_low_rank()) sVU[i] += Tk.rows() * Tj.rank();
            }
          }
          lwork = std::max(lwork, sU0[i]+sV0[i]+sU1[i]+sV1[i]+sVU[i]+sUVU[i]);
        }

        gpu::DeviceMemory<scalar_t> dmem(d2*d2 + lwork);
        DenseMW_t dA22(d2, d2, dmem, d2);
        gpu::copy_host_to_device(dA22, A22);

        for (std::size_t i=0; i<rb; i++) {
          scalar_t *dU0 = dA22.data(), *dV0 = dU0 + sU0[i],
            *dU1 = dV0 + sV0[i], *dV1 = dU1 + sU1[i],
            *dVU = dV1 + sV1[i], *dUVU = dVU + sVU[i];
          for (std::size_t k=0; k<rb2; k++) {
            auto& Tk = B21.tile(k, i);
            if (Tk.is_low_rank()) {
              gpu::copy_host_to_device(dU0, Tk.U());
              gpu::copy_host_to_device(dV0, Tk.V());
              dU0 += Tk.U().nonzeros();
              dV0 += Tk.V().nonzeros();
            } else {
              gpu::copy_host_to_device(dU0, Tk.D());
              dU0 += Tk.D().nonzeros();
            }
          }
          VBatchedMeta<scalar_t> b1(rb2*rb2), b2(rb2*rb2), b3(rb2*rb2);
          for (std::size_t j=0; j<rb2; j++) {
            auto& Tj = B12.tile(i, j);
            dU0 = dA22.data();
            dV0 = dU0 + sU0[i];
            for (std::size_t k=0; k<rb2; k++) {
              auto& Tk = B21.tile(k, i);
              auto dAkj = dA22.ptr(B21.tileroff(k), B12.tilecoff(j));
              if (Tk.is_low_rank()) {
                if (Tj.is_low_rank()) {
                  // V0*U1
                  b1.add(Tk.rank(), Tj.rank(), Tk.cols(), dV0, dU1, dVU);
                  if (Tj.rank() < Tk.rank()) {
                    // U0*(V0*U1)
                    b2.add(Tk.rows(), Tj.rank(), Tk.rank(), dU0, dVU, dUVU);
                    // A - (U0*(V0*U1))*V1
                    b3.add(Tk.rows(), Tj.cols(), Tj.rank(), dUVU, dV1, dAkj);
                  } else {
                    // (V0*U1)*V1
                    b2.add(Tk.rank(), Tj.cols(), Tj.rank(), dVU, dV1, dUVU);
                    // A - U0*((V0*U1))*V1)
                    b3.add(Tk.rows(), Tj.cols(), Tk.rank(), dU0, dUVU, dAkj);
                  }
                } else { // Tk low-rank, Tj dense
                  // V0*D1
                  b1.add(Tk.rank(), Tj.cols(), Tk.cols(), dV0, dU1, dVU);
                  // U0*(V0*D1)
                  b3.add(Tk.rows(), Tj.cols(), Tk.rank(), dU0, dVU, dAkj);
                }
                dU0 += Tk.U().nonzeros();
                dV0 += Tk.V().nonzeros();
              } else { // Tk is dense
                if (Tj.is_low_rank()) { // Tk dense, Tj low-rank
                  // D0*U1
                  b1.add(Tk.rows(), Tj.rank(), Tk.cols(), dU0, dU1, dVU);
                  // (D0*U1)*V1
                  b3.add(Tk.rows(), Tj.cols(), Tj.rank(), dVU, dV1, dAkj);
                } else // Tk and Tj are dense, D0*D1
                  b3.add(Tk.rows(), Tj.cols(), Tk.cols(), dU0, dU1, dAkj);
                dU0 += Tk.D().nonzeros();
              }
            }
            if (Tj.is_low_rank()) {
              gpu::copy_host_to_device(dU1, Tj.U());
              gpu::copy_host_to_device(dV1, Tj.V());
              dU1 += Tj.U().nonzeros();
              dV1 += Tj.V().nonzeros();
            } else {
              gpu::copy_host_to_device(dU1, Tj.D());
              dU1 += Tj.D().nonzeros();
            }
          }
          b1.run(scalar_t(1.), scalar_t(0.), q);
          b2.run(scalar_t(1.), scalar_t(0.), q);
          b3.run(scalar_t(-1.), scalar_t(1.), q);
        }
        gpu::copy_device_to_host(A22, dA22);

        // magma_finalize();
      }

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
