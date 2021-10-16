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
      {
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
      }
      for (std::size_t i=0; i<rb; i++)
        for (std::size_t l=B11.tileroff(i); l<B11.tileroff(i+1); l++)
          piv[l] += B11.tileroff(i);
      A11.clear();
      A12.clear();
      A21.clear();

      auto d2 = A22.rows();
      if (d2) {
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
      }
    }


    // explicit template instantiations
    template class BLRMatrix<float>;
    template class BLRMatrix<double>;
    template class BLRMatrix<std::complex<float>>;
    template class BLRMatrix<std::complex<double>>;

  } // end namespace BLR
} // end namespace strumpack
