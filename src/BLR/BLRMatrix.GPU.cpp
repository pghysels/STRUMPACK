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
      void clear() {
        m_.clear();  ldA_.clear();  A_.clear();
        n_.clear();  ldB_.clear();  B_.clear();
        k_.clear();  ldC_.clear();  C_.clear();
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
      void run(scalar_t alpha, scalar_t beta, magma_queue_t q) {
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
        gpu::copy_host_to_device(dimem, imem.data(), (B+1)*6);

        std::vector<scalar_t*> smem(B*3);
        auto sptr = smem.begin();
        std::copy(A_.begin(), A_.end(), sptr);   sptr += B;
        std::copy(B_.begin(), B_.end(), sptr);   sptr += B;
        std::copy(C_.begin(), C_.end(), sptr);   sptr += B;
        gpu::copy_host_to_device(dsmem, smem.data(), B*3);

        // TODO count flops
        gpu::synchronize();
        gpu::magma::gemm_vbatched
          (MagmaNoTrans, MagmaNoTrans, dimem, dimem+(B+1), dimem+2*(B+1),
           alpha, dsmem, dimem+3*(B+1), dsmem+B, dimem+4*(B+1),
           beta, dsmem+2*B, dimem+5*(B+1), B, q);
      }
#endif
      void run(scalar_t alpha, scalar_t beta,
               std::vector<gpu::BLASHandle>& h) {
        std::size_t batchcount = m_.size();
        if (!batchcount) return;
        // TODO count flops ? or already in gpu::gemm?
        gpu::synchronize();
        for (std::size_t i=0; i<batchcount; i++) {
          DenseMatrixWrapper<scalar_t>
            A(m_[i], k_[i], A_[i], ldA_[i]),
            B(k_[i], n_[i], B_[i], ldB_[i]),
            C(m_[i], n_[i], C_[i], ldC_[i]);
          gpu::gemm(h[i % h.size()], Trans::N, Trans::N,
                    alpha, A, B, beta, C);
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
          int nr_streams = 4; // TODO get from options, only in sparse now
          std::vector<gpu::Stream> streams(nr_streams);
          std::vector<gpu::BLASHandle> handles(nr_streams);
          for (int i=0; i<nr_streams; i++)
            handles[i].set_stream(streams[i]);
#if defined(STRUMPACK_USE_MAGMA)
          magma_init();
          magma_queue_t q;
#if defined(STRUMPACK_USE_CUDA)
          magma_queue_create_from_cuda
            (0, streams[0], handles[0], nullptr, &q);
#else
          magma_queue_create(0, &q);
#endif
#endif
          std::vector<std::size_t> sU0(rb), sV0(rb),
            sU1(rb), sV1(rb), sVU(rb), sUVU(rb);
          std::size_t lwork = 0;
          // count how much device memory will be required
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
          auto bdwork = VBatchedGEMM<scalar_t>::dwork_bytes(rb2*rb2);
          gpu::DeviceMemory<char> bdmem(bdwork*3 + (d2*d2+lwork)*sizeof(scalar_t));
          scalar_t* dmem = reinterpret_cast<scalar_t*>(bdmem + bdwork*3);
          DenseMW_t dA22(d2, d2, dmem, d2);
          gpu::copy_host_to_device_async(dA22, A22, streams[0]);
          for (std::size_t i=0, s=1; i<rb; i++) {
            scalar_t *dU0 = dA22.end()/*dmem*/, *dV0 = dU0 + sU0[i],
              *dU1 = dV0 + sV0[i], *dV1 = dU1 + sU1[i],
              *dVU = dV1 + sV1[i], *dUVU = dVU + sVU[i];
            for (std::size_t k=0; k<rb2; k++) {
              auto& Tk = B21.tile(k, i);
              if (Tk.is_low_rank()) {
                gpu::copy_host_to_device_async
                  (dU0, Tk.U(), streams[s++ % nr_streams]);
                gpu::copy_host_to_device_async
                  (dV0, Tk.V(), streams[s++ % nr_streams]);
                dU0 += Tk.U().nonzeros();
                dV0 += Tk.V().nonzeros();
              } else {
                gpu::copy_host_to_device_async
                  (dU0, Tk.D(), streams[s++ % nr_streams]);
                dU0 += Tk.D().nonzeros();
              }
            }
            VBatchedGEMM<scalar_t> b1(rb2*rb2, bdmem),
              b2(rb2*rb2, bdmem+bdwork), b3(rb2*rb2, bdmem+2*bdwork);
            for (std::size_t j=0; j<rb2; j++) {
              auto& Tj = B12.tile(i, j);
              dU0 = dA22.end()/*dmem*/;
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
                      b3.add(Tk.rows(), Tj.cols(), Tj.rank(),
                             dUVU, Tk.rows(), dV1, Tj.rank(), dAkj, d2);
                      dUVU += Tk.rows() * Tj.rank();
                    } else {
                      // (V0*U1)*V1
                      b2.add(Tk.rank(), Tj.cols(), Tj.rank(), dVU, dV1, dUVU);
                      // A - U0*((V0*U1))*V1)
                      b3.add(Tk.rows(), Tj.cols(), Tk.rank(),
                             dU0, Tk.rows(), dUVU, Tk.rank(), dAkj, d2);
                      dUVU += Tk.rank() * Tj.cols();
                    }
                    dVU += Tk.rank() * Tj.rank();
                  } else { // Tk low-rank, Tj dense
                    // V0*D1
                    b1.add(Tk.rank(), Tj.cols(), Tk.cols(), dV0, dU1, dVU);
                    // U0*(V0*D1)
                    b3.add(Tk.rows(), Tj.cols(), Tk.rank(),
                           dU0, Tk.rows(), dVU, Tk.rank(), dAkj, d2);
                    dVU += Tk.rank() * Tj.cols();
                  }
                  dU0 += Tk.U().nonzeros();
                  dV0 += Tk.V().nonzeros();
                } else { // Tk is dense
                  if (Tj.is_low_rank()) { // Tk dense, Tj low-rank
                    // D0*U1
                    b1.add(Tk.rows(), Tj.rank(), Tk.cols(), dU0, dU1, dVU);
                    // (D0*U1)*V1
                    b3.add(Tk.rows(), Tj.cols(), Tj.rank(), dVU, Tk.rows(),
                           dV1, Tj.rank(), dAkj, d2);
                    dVU += Tk.rows(), Tj.rank();
                  } else // Tk and Tj are dense, D0*D1
                    b3.add(Tk.rows(), Tj.cols(), Tk.cols(), dU0, Tk.rows(),
                           dU1, Tk.cols(), dAkj, d2);
                  dU0 += Tk.D().nonzeros();
                }
              }
              if (Tj.is_low_rank()) {
                gpu::copy_host_to_device_async
                  (dU1, Tj.U(), streams[s++ % nr_streams]);
                gpu::copy_host_to_device_async
                  (dV1, Tj.V(), streams[s++ % nr_streams]);
                dU1 += Tj.U().nonzeros();
                dV1 += Tj.V().nonzeros();
              } else {
                gpu::copy_host_to_device_async
                  (dU1, Tj.D(), streams[s++ % nr_streams]);
                dU1 += Tj.D().nonzeros();
              }
            }
#if defined(STRUMPACK_USE_MAGMA)
            b1.run(scalar_t(1.), scalar_t(0.), q);
            b2.run(scalar_t(1.), scalar_t(0.), q);
            b3.run(scalar_t(-1.), scalar_t(1.), q);
#else
            b1.run(scalar_t(1.), scalar_t(0.), handles);
            b2.run(scalar_t(1.), scalar_t(0.), handles);
            b3.run(scalar_t(-1.), scalar_t(1.), handles);
#endif
            gpu::synchronize();
          }
          gpu::copy_device_to_host(A22, dA22);
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
