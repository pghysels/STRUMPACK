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
#include <stdlib.h>

#include "HIPWrapper.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "misc/MPIWrapper.hpp"
#endif

namespace strumpack {
  namespace gpu {

    void peek_at_last_error() {
      gpu_check(hipPeekAtLastError());
    }

    void get_last_error() {
      // this is used to reset the last error. Some MAGMA calls fail
      // on purpose, then use a different algorithm
      auto e = hipGetLastError();
      ((void)e); // silence unused warning
    }

    void synchronize_default_stream() {
      gpu_check(hipStreamSynchronize(0));
    }

    struct Stream::StreamImpl {
      StreamImpl() { gpu_check(hipStreamCreate(&s_)); }
      ~StreamImpl() { gpu_check(hipStreamDestroy(s_)); }
      operator hipStream_t&() { return s_; }
      operator const hipStream_t&() const { return s_; }
      void synchronize() { gpu_check(hipStreamSynchronize(s_)); }
      hipStream_t s_;
    };
    Stream::Stream() { s_ = std::make_unique<StreamImpl>(); }
    Stream::~Stream() = default;
    void Stream::synchronize() { s_->synchronize(); }

    const hipStream_t& get_hip_stream(const Stream& s) { return *(s.s_); }
    hipStream_t& get_hip_stream(Stream& s) { return *(s.s_); }

    struct Handle::HandleImpl {
      HandleImpl() {
        gpu_check(hipblasCreate(&bh_));
        gpu_check(rocblas_create_handle(&sh_));
#if defined(STRUMPACK_USE_MAGMA)
        magma_queue_create_from_hip(0, NULL, bh_, NULL, &q_);
#endif
#if defined(STRUMPACK_USE_KBLAS)
        kblasCreate(&k_);
        kblasInitRandState(k_, &r_, 16384*2, 0);
        kblasEnableMagma(k_);
#endif
      }
      HandleImpl(Stream& s) {
        gpu_check(hipblasCreate(&bh_));
        gpu_check(rocblas_create_handle(&sh_));
#if defined(STRUMPACK_USE_MAGMA)
        magma_queue_create_from_hip(0, get_hip_stream(s), bh_, NULL, &q_);
#endif
#if defined(STRUMPACK_USE_KBLAS)
        kblasCreate(&k_);
        kblasInitRandState(k_, &r_, 16384*2, 0);
        kblasEnableMagma(k_);
#endif
        set_stream(s);
      }
      ~HandleImpl() {
#if defined(STRUMPACK_USE_KBLAS)
        kblasDestroy(&k_);
#endif
#if defined(STRUMPACK_USE_MAGMA)
        magma_queue_destroy(q_);
#endif
        gpu_check(rocblas_destroy_handle(sh_));
        gpu_check(hipblasDestroy(bh_));
      }
      void set_stream(Stream& s) {
        gpu_check(hipblasSetStream(bh_, get_hip_stream(s)));
        gpu_check(rocblas_set_stream(sh_, get_hip_stream(s)));
#if defined(STRUMPACK_USE_KBLAS)
        kblasSetStream(k_, get_hip_stream(s));
#endif
      }

      operator hipblasHandle_t&() { return bh_; }
      operator const hipblasHandle_t&() const { return bh_; }
      operator rocblas_handle&() { return sh_; }
      operator const rocblas_handle&() const { return sh_; }
      hipblasHandle_t bh_;
      rocblas_handle sh_;

#if defined(STRUMPACK_USE_MAGMA)
      operator magma_queue_t&() { return q_; }
      operator const magma_queue_t&() const { return q_; }
      magma_queue_t q_;
#endif

#if defined(STRUMPACK_USE_KBLAS)
      operator kblasHandle_t&() { return k_; }
      operator const kblasHandle_t&() const { return k_; }
      operator kblasRandState_t&() { return r_; }
      operator const kblasRandState_t&() const { return r_; }
      kblasHandle_t k_;
      kblasRandState_t r_;
#endif
    };
    Handle::Handle() { h_ = std::make_unique<HandleImpl>(); }
    Handle::Handle(Stream& s) { h_ = std::make_unique<HandleImpl>(s); }
    Handle::~Handle() = default;
    void Handle::set_stream(Stream& s) { h_->set_stream(s); }

    const hipblasHandle_t& get_hipblas_handle(const Handle& h) { return *(h.h_); }
    hipblasHandle_t& get_hipblas_handle(Handle& h) { return *(h.h_); }
    const rocblas_handle& get_rocblas_handle(const Handle& h) { return *(h.h_); }
    rocblas_handle& get_rocblas_handle(Handle& h) { return *(h.h_); }
#if defined(STRUMPACK_USE_MAGMA)
    const magma_queue_t& get_magma_queue(const Handle& h) { return *(h.h_); }
    magma_queue_t& get_magma_queue(Handle& h) { return *(h.h_); }
#endif
#if defined(STRUMPACK_USE_KBLAS)
    const kblasHandle_t& get_kblas_handle(const Handle& h) { return *(h.h_); }
    kblasHandle_t& get_kblas_handle(Handle& h) { return *(h.h_); }
    const kblasRandState_t& get_kblas_rand_state(const Handle& h) { return *(h.h_); }
    kblasRandState_t& get_kblas_rand_state(Handle& h) { return *(h.h_); }
#endif

    struct Event::EventImpl {
      EventImpl() {
        gpu_check(hipEventCreateWithFlags
                  (&e_, hipEventDisableTiming)); }
      ~EventImpl() { gpu_check(hipEventDestroy(e_)); }
      void record() { gpu_check(hipEventRecord(e_)); }
      void record(Stream& s) { gpu_check(hipEventRecord(e_, get_hip_stream(s))); }
      void wait(Stream& s) { gpu_check(hipStreamWaitEvent(get_hip_stream(s), e_, 0)); }
      void synchronize() { gpu_check(hipEventSynchronize(e_));}
      hipEvent_t e_;
    };
    Event::Event() { e_ = std::make_unique<EventImpl>(); }
    Event::~Event() = default;
    void Event::record() { e_->record(); }
    void Event::record(Stream& s) { e_->record(s); }
    void Event::wait(Stream& s) { e_->wait(s); }
    void Event::synchronize() { e_->synchronize(); }

    void device_malloc(void** ptr, std::size_t size) {
      if (hipMalloc(ptr, size) != hipSuccess) {
        std::cerr << "HIP Failed to allocate " << size << " bytes on device" << std::endl;
        throw std::bad_alloc();
      }
    }
    void host_malloc(void** ptr, std::size_t size) {
      if (hipHostMalloc(ptr, size) != hipSuccess) {
        std::cerr << "HIP Failed to allocate " << size << " bytes on host " << std::endl;
        throw std::bad_alloc();
      }
    }
    void device_free(void* ptr) { gpu_check(hipFree(ptr)); }
    void host_free(void* ptr) { gpu_check(hipHostFree(ptr)); }

    hipMemcpyKind CD2hipMK(CopyDir d) {
      switch (d) {
      case CopyDir::H2H: return hipMemcpyHostToHost;
      case CopyDir::H2D: return hipMemcpyHostToDevice;
      case CopyDir::D2H: return hipMemcpyDeviceToHost;
      case CopyDir::D2D: return hipMemcpyDeviceToDevice;
      case CopyDir::DEF: return hipMemcpyDefault;
      default: assert(false); return hipMemcpyDefault;
      }
    }

    hipblasOperation_t T2hipOp(Trans op) {
      switch (op) {
      case Trans::N: return HIPBLAS_OP_N;
      case Trans::T: return HIPBLAS_OP_T;
      case Trans::C: return HIPBLAS_OP_C;
      default:
        assert(false);
        return HIPBLAS_OP_N;
      }
    }
    rocblas_operation T2rocOp(Trans op) {
      switch (op) {
      case Trans::N: return rocblas_operation_none;
      case Trans::T: return rocblas_operation_transpose;
      case Trans::C: return rocblas_operation_conjugate_transpose;
      default:
        assert(false);
        return rocblas_operation_none;
      }
    }
    hipblasSideMode_t S2hipOp(Side op) {
      switch (op) {
      case Side::L: return HIPBLAS_SIDE_LEFT;
      case Side::R: return HIPBLAS_SIDE_RIGHT;
      default:
        assert(false);
        return HIPBLAS_SIDE_LEFT;
      }
    }
    hipblasFillMode_t U2hipOp(UpLo op) {
      switch (op) {
      case UpLo::L: return HIPBLAS_FILL_MODE_LOWER;
      case UpLo::U: return HIPBLAS_FILL_MODE_UPPER;
      default:
        assert(false);
        return HIPBLAS_FILL_MODE_LOWER;
      }
    }
    hipblasDiagType_t D2hipOp(Diag op) {
      switch (op) {
      case Diag::N: return HIPBLAS_DIAG_NON_UNIT;
      case Diag::U: return HIPBLAS_DIAG_UNIT;
      default:
        assert(false);
        return HIPBLAS_DIAG_UNIT;
      }
    }
    // hipsolverEigMode_t E2hipOp(Jobz op) {
    //   switch (op) {
    //   case Jobz::N: return HIPSOLVER_EIG_MODE_NOVECTOR;
    //   case Jobz::V: return HIPSOLVER_EIG_MODE_VECTOR;
    //   default:
    //     assert(false);
    //     return HIPSOLVER_EIG_MODE_VECTOR;
    //   }
    // }

    void hip_assert(hipError_t code, const char *file, int line,
                     bool abort) {
      if (code != hipSuccess) {
        std::cerr << "HIP assertion failed: "
                  << hipGetErrorString(code) << " "
                  <<  file << " " << line << std::endl;
        if (abort) exit(code);
      }
    }

    void hip_assert(rocblas_status code, const char *file, int line, bool abort) {
      if (code != rocblas_status_success) {
        std::cerr << "rocsolver/rocblas assertion failed: " << code << " "
                  <<  file << " " << line << std::endl;
        if (abort) exit(code);
      }
    }

    void hip_assert(hipblasStatus_t code, const char *file, int line, bool abort) {
      if (code != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipBLAS assertion failed: " << code << " "
                  <<  file << " " << line << std::endl;
        if (abort) exit(code);
      }
    }

    void init() {
#if defined(STRUMPACK_USE_MPI)
      int devs;
      gpu_check(hipGetDeviceCount(&devs));
      if (devs > 1) {
        int flag, rank = 0;
        MPI_Initialized(&flag);
        if (flag) {
          MPIComm c;
          rank = c.rank();
        }
        gpu_check(hipSetDevice(rank % devs));
      }
#endif
    }

    void device_memset(void* dptr, int value, std::size_t count) {
      gpu_check(hipMemset(dptr, value, count));
    }

    void device_copy(void* dest, const void* src,
                     std::size_t count, CopyDir dir) {
      gpu_check(hipMemcpy(dest, src, count, CD2hipMK(dir)));
    }
    void device_copy_async(void* dest, const void* src, std::size_t count,
                           CopyDir dir, Stream& s) {
      gpu_check(hipMemcpyAsync(dest, src, count, CD2hipMK(dir),
                               get_hip_stream(s)));
    }
    void device_copy_2D(void* dest, std::size_t dpitch,
                        const void* src, std::size_t spitch,
                        std::size_t width, std::size_t height, CopyDir dir) {
      gpu_check(hipMemcpy2D(dest, dpitch, src, spitch,
                            width, height, CD2hipMK(dir)));
    }
    void device_copy_2D_async(void* dest, std::size_t dpitch,
                              const void* src, std::size_t spitch,
                              std::size_t width, std::size_t height,
                              CopyDir dir, Stream& s) {
      gpu_check(hipMemcpy2DAsync(dest, dpitch, src, spitch, width , height,
                                 CD2hipMK(dir), get_hip_stream(s)));
    }

    std::size_t available_memory() {
      std::size_t free_device_mem, total_device_mem;
      gpu_check(hipMemGetInfo(&free_device_mem, &total_device_mem));
      return free_device_mem;
    }

    void gemm(hipblasHandle_t& handle, hipblasOperation_t transa,
              hipblasOperation_t transb, int m, int n, int k,
              float alpha, const float* A, int lda,
              const float* B, int ldb,
              float beta, float* C, int ldc) {
      STRUMPACK_FLOPS(blas::gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(4*blas::gemm_moves(m,n,k));
      gpu_check(hipblasSgemm
                (handle, transa, transb, m, n, k, &alpha, A, lda,
                 B, ldb, &beta, C, ldc));
    }
    void gemm(hipblasHandle_t& handle, hipblasOperation_t transa,
              hipblasOperation_t transb, int m, int n, int k,
              double alpha, const double* A, int lda,
              const double* B, int ldb,
              double beta, double* C, int ldc) {
      STRUMPACK_FLOPS(blas::gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(8*blas::gemm_moves(m,n,k));
      gpu_check(hipblasDgemm
                (handle, transa, transb, m, n, k, &alpha, A, lda,
                 B, ldb, &beta, C, ldc));
    }
    void gemm(hipblasHandle_t& handle, hipblasOperation_t transa,
              hipblasOperation_t transb, int m, int n, int k,
              std::complex<float> alpha,
              const std::complex<float>* A, int lda,
              const std::complex<float>* B, int ldb,
              std::complex<float> beta, std::complex<float> *C,
              int ldc) {
      STRUMPACK_FLOPS(4*blas::gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(2*4*blas::gemm_moves(m,n,k));
      gpu_check(hipblasCgemm
                (handle, transa, transb, m, n, k,
                 reinterpret_cast<hipblasComplex*>(&alpha),
                 reinterpret_cast<const hipblasComplex*>(A), lda,
                 reinterpret_cast<const hipblasComplex*>(B), ldb,
                 reinterpret_cast<hipblasComplex*>(&beta),
                 reinterpret_cast<hipblasComplex*>(C), ldc));
    }
    void gemm(hipblasHandle_t& handle, hipblasOperation_t transa,
              hipblasOperation_t transb, int m, int n, int k,
              std::complex<double> alpha,
              const std::complex<double> *A, int lda,
              const std::complex<double> *B, int ldb,
              std::complex<double> beta,
              std::complex<double> *C, int ldc) {
      STRUMPACK_FLOPS(4*blas::gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(2*8*blas::gemm_moves(m,n,k));
      gpu_check(hipblasZgemm
                (handle, transa, transb, m, n, k,
                 reinterpret_cast<hipblasDoubleComplex*>(&alpha),
                 reinterpret_cast<const hipblasDoubleComplex*>(A), lda,
                 reinterpret_cast<const hipblasDoubleComplex*>(B), ldb,
                 reinterpret_cast<hipblasDoubleComplex*>(&beta),
                 reinterpret_cast<hipblasDoubleComplex*>(C), ldc));
    }

    template<typename scalar_t> void
    gemm(Handle& handle, Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c) {
      assert((ta==Trans::N && a.rows()==c.rows()) ||
             (ta!=Trans::N && a.cols()==c.rows()));
      assert((tb==Trans::N && b.cols()==c.cols()) ||
             (tb!=Trans::N && b.rows()==c.cols()));
      assert((ta==Trans::N && tb==Trans::N && a.cols()==b.rows()) ||
             (ta!=Trans::N && tb==Trans::N && a.rows()==b.rows()) ||
             (ta==Trans::N && tb!=Trans::N && a.cols()==b.cols()) ||
             (ta!=Trans::N && tb!=Trans::N && a.rows()==b.cols()));
      gemm(get_hipblas_handle(handle), T2hipOp(ta), T2hipOp(tb),
           c.rows(), c.cols(), (ta==Trans::N) ? a.cols() : a.rows(),
           alpha, a.data(), a.ld(), b.data(), b.ld(), beta, c.data(), c.ld());
    }
    template void gemm(Handle&, Trans, Trans, float, const DenseMatrix<float>&,
                       const DenseMatrix<float>&, float, DenseMatrix<float>&);
    template void gemm(Handle&, Trans, Trans, double, const DenseMatrix<double>&,
                       const DenseMatrix<double>&, double, DenseMatrix<double>&);
    template void gemm(Handle&, Trans, Trans, std::complex<float>,
                       const DenseMatrix<std::complex<float>>&, const DenseMatrix<std::complex<float>>&,
                       std::complex<float>, DenseMatrix<std::complex<float>>&);
    template void gemm(Handle&, Trans, Trans, std::complex<double>,
                       const DenseMatrix<std::complex<double>>&, const DenseMatrix<std::complex<double>>&,
                       std::complex<double>, DenseMatrix<std::complex<double>>&);

    // it seems like rocsolver doesn't need work memory?
    template<typename scalar_t> std::int64_t
    getrf_buffersize(Handle& handle, int n) { return 0; }
    template std::int64_t getrf_buffersize<float>(Handle&, int);
    template std::int64_t getrf_buffersize<double>(Handle&, int);
    template std::int64_t getrf_buffersize<std::complex<float>>(Handle&, int);
    template std::int64_t getrf_buffersize<std::complex<double>>(Handle&, int);

    void getrf(rocblas_handle& handle, int m, int n, float* A, int lda,
               float*, std::int64_t, int* dpiv, int* dinfo) {
      STRUMPACK_FLOPS(blas::getrf_flops(m,n));
      gpu_check(rocsolver_sgetrf(handle, m, n, A, lda, dpiv, dinfo));
    }
    void getrf(rocblas_handle& handle, int m, int n, double* A, int lda,
               double*, std::int64_t, int* dpiv, int* dinfo) {
      STRUMPACK_FLOPS(blas::getrf_flops(m,n));
      gpu_check(rocsolver_dgetrf(handle, m, n, A, lda, dpiv, dinfo));
    }
    void getrf(rocblas_handle& handle, int m, int n, std::complex<float>* A, int lda,
               std::complex<float>*, std::int64_t, int* dpiv, int* dinfo) {
      STRUMPACK_FLOPS(4*blas::getrf_flops(m,n));
      gpu_check(rocsolver_cgetrf
                (handle, m, n, reinterpret_cast<rocblas_float_complex*>(A),
                 lda, dpiv, dinfo));
    }
    void getrf(rocblas_handle& handle, int m, int n, std::complex<double>* A, int lda,
               std::complex<double>*, std::int64_t, int* dpiv, int* dinfo) {
      STRUMPACK_FLOPS(4*blas::getrf_flops(m,n));
      gpu_check(rocsolver_zgetrf
                (handle, m, n, reinterpret_cast<rocblas_double_complex*>(A),
                 lda, dpiv, dinfo));
    }

    template<typename scalar_t> void
    getrf(Handle& handle, DenseMatrix<scalar_t>& A,
          scalar_t* work, std::int64_t lwork, int* dpiv, int* dinfo) {
      getrf(get_rocblas_handle(handle), A.rows(), A.cols(), A.data(), A.ld(),
            work, lwork, dpiv, dinfo);
    }
    template void getrf(Handle&, DenseMatrix<float>&, float*, std::int64_t, int*, int*);
    template void getrf(Handle&, DenseMatrix<double>&, double*, std::int64_t, int*, int*);
    template void getrf(Handle&, DenseMatrix<std::complex<float>>&, std::complex<float>*, std::int64_t, int*, int*);
    template void getrf(Handle&, DenseMatrix<std::complex<double>>& A, std::complex<double>*, std::int64_t, int*, int*);

    template<typename scalar_t> std::int64_t
    getrs_buffersize(Handle&, Trans, int, int, int, int) {
      return 0;
    }
    template std::int64_t getrs_buffersize<float>(Handle&, Trans, int, int, int, int);
    template std::int64_t getrs_buffersize<double>(Handle&, Trans, int, int, int, int);
    template std::int64_t getrs_buffersize<std::complex<float>>(Handle&, Trans, int, int, int, int);
    template std::int64_t getrs_buffersize<std::complex<double>>(Handle&, Trans, int, int, int, int);

    void getrs(rocblas_handle& handle, rocblas_operation trans,
               int n, int nrhs, const float* A, int lda,
               const int* dpiv, float* B, int ldb, int* dinfo) {
      STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
      gpu_check(rocsolver_sgetrs
                (handle, trans, n, nrhs,
                 const_cast<float*>(A), lda, dpiv, B, ldb));
    }
    void getrs(rocblas_handle& handle, rocblas_operation trans,
               int n, int nrhs, const double* A, int lda,
               const int* dpiv, double* B, int ldb, int* dinfo) {
      STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
      gpu_check(rocsolver_dgetrs
                (handle, trans, n, nrhs,
                 const_cast<double*>(A), lda, dpiv, B, ldb));
    }
    void getrs(rocblas_handle& handle, rocblas_operation trans,
               int n, int nrhs, const std::complex<float>* A, int lda,
               const int* dpiv, std::complex<float>* B, int ldb, int* dinfo) {
      STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
      gpu_check(rocsolver_cgetrs
                (handle, trans, n, nrhs,
                 reinterpret_cast<rocblas_float_complex*>
                 (const_cast<std::complex<float>*>(A)), lda, dpiv,
                 reinterpret_cast<rocblas_float_complex*>
                 (const_cast<std::complex<float>*>(B)), ldb));
    }
    void getrs(rocblas_handle& handle, rocblas_operation trans,
               int n, int nrhs, const std::complex<double>* A, int lda,
               const int* dpiv, std::complex<double>* B, int ldb,
               int *dinfo) {
      STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
      gpu_check(rocsolver_zgetrs
                (handle, trans, n, nrhs,
                 reinterpret_cast<rocblas_double_complex*>
                 (const_cast<std::complex<double>*>(A)), lda, dpiv,
                 reinterpret_cast<rocblas_double_complex*>
                 (const_cast<std::complex<double>*>(B)), ldb));
    }

    template<typename scalar_t> void
    getrs(Handle& handle, Trans trans, const DenseMatrix<scalar_t>& A,
          const int* dpiv, DenseMatrix<scalar_t>& B, int *dinfo,
          scalar_t*, std::int64_t) {
      getrs(get_rocblas_handle(handle), T2rocOp(trans), A.rows(), B.cols(),
            A.data(), A.ld(), dpiv, B.data(), B.ld(), dinfo);
    }
    template void getrs(Handle&, Trans, const DenseMatrix<float>&,
                        const int*, DenseMatrix<float>&, int*, float*, std::int64_t);
    template void getrs(Handle&, Trans, const DenseMatrix<double>&,
                        const int*, DenseMatrix<double>&, int*, double*, std::int64_t);
    template void getrs(Handle&, Trans, const DenseMatrix<std::complex<float>>&, const int*,
                        DenseMatrix<std::complex<float>>&, int*, std::complex<float>*, std::int64_t);
    template void getrs(Handle&, Trans, const DenseMatrix<std::complex<double>>&, const int*,
                        DenseMatrix<std::complex<double>>&, int*, std::complex<double>*, std::int64_t);

    void trsm(hipblasHandle_t& handle, hipblasSideMode_t side,
              hipblasFillMode_t uplo, hipblasOperation_t trans,
              hipblasDiagType_t diag, int m, int n, const float* alpha,
              float* A, int lda, float* B, int ldb) {
      STRUMPACK_FLOPS(blas::trsm_flops(m, n, alpha, side));
      STRUMPACK_BYTES(4*blas::trsm_moves(m, n));
      gpu_check(hipblasStrsm(handle, side, uplo, trans, diag, m,
                             n, alpha, A, lda, B, ldb));
    }
    void trsm(hipblasHandle_t& handle, hipblasSideMode_t side,
              hipblasFillMode_t uplo, hipblasOperation_t trans,
              hipblasDiagType_t diag, int m, int n, const double* alpha,
              double* A, int lda, double* B, int ldb) {
      STRUMPACK_FLOPS(blas::trsm_flops(m, n, alpha, side));
      STRUMPACK_BYTES(8*blas::trsm_moves(m, n));
      gpu_check(hipblasDtrsm(handle, side, uplo, trans, diag, m,
                             n, alpha, A, lda, B, ldb));
    }
    void trsm(hipblasHandle_t& handle, hipblasSideMode_t side,
              hipblasFillMode_t uplo, hipblasOperation_t trans,
              hipblasDiagType_t diag, int m, int n,
              const std::complex<float>* alpha, std::complex<float>* A,
              int lda, std::complex<float>* B, int ldb) {
      STRUMPACK_FLOPS(4*blas::trsm_flops(m, n, alpha, side));
      STRUMPACK_BYTES(2*4*blas::trsm_moves(m, n));
      gpu_check(hipblasCtrsm(handle, side, uplo, trans, diag, m, n,
                             reinterpret_cast<const hipblasComplex*>(alpha),
                             reinterpret_cast<hipblasComplex*>(A), lda,
                             reinterpret_cast<hipblasComplex*>(B), ldb));
    }
    void trsm(hipblasHandle_t& handle, hipblasSideMode_t side,
              hipblasFillMode_t uplo, hipblasOperation_t trans,
              hipblasDiagType_t diag, int m, int n,
              const std::complex<double>* alpha, std::complex<double>* A,
              int lda, std::complex<double>* B, int ldb) {
      STRUMPACK_FLOPS(4*blas::trsm_flops(m, n, alpha, side));
      STRUMPACK_BYTES(2*8*blas::trsm_moves(m, n));
      gpu_check(hipblasZtrsm(handle, side, uplo, trans, diag, m, n,
                             reinterpret_cast<const hipblasDoubleComplex*>(alpha), 
                             reinterpret_cast<hipblasDoubleComplex*>(A), lda,
                             reinterpret_cast<hipblasDoubleComplex*>(B), ldb));
    }
    template<typename scalar_t> void
    trsm(Handle& handle, Side side, UpLo uplo, Trans trans, Diag diag,
         const scalar_t alpha, DenseMatrix<scalar_t>& A, DenseMatrix<scalar_t>& B) {
      trsm(get_hipblas_handle(handle),
           S2hipOp(side), U2hipOp(uplo), T2hipOp(trans), D2hipOp(diag),
           B.rows(), B.cols(), &alpha, A.data(), A.ld(), B.data(), B.ld());
    }
    template void trsm(Handle&, Side, UpLo, Trans, Diag, const float,
                       DenseMatrix<float>&, DenseMatrix<float>&);
    template void trsm(Handle&, Side, UpLo, Trans, Diag, const double,
                       DenseMatrix<double>&, DenseMatrix<double>&);
    template void trsm(Handle&, Side, UpLo, Trans, Diag, const std::complex<float>,
                       DenseMatrix<std::complex<float>>&, DenseMatrix<std::complex<float>>&);
    template void trsm(Handle&, Side, UpLo, Trans, Diag, const std::complex<double>,
                       DenseMatrix<std::complex<double>>&, DenseMatrix<std::complex<double>>&);

    template<typename scalar_t, typename real_t>
    int gesvdj_buffersize(Handle& handle, Jobz jobz, int m, int n) {
      std::cerr << "TODO gesvdj_buffersize not implemented for HIP" << std::endl;
      return 0;
    }
    template int gesvdj_buffersize<float,float>(Handle&, Jobz, int, int);
    template int gesvdj_buffersize<double,double>(Handle&, Jobz, int, int);
    template int gesvdj_buffersize<std::complex<float>,float>(Handle&, Jobz, int, int);
    template int gesvdj_buffersize<std::complex<double>,double>(Handle&, Jobz, int, int);


    template<typename scalar_t, typename real_t> void
    gesvdj(Handle& handle, Jobz jobz, DenseMatrix<scalar_t>& A, real_t* S,
           DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V, int* dinfo,
           scalar_t* work, int lwork, const double tol) {
      std::cerr << "TODO gesvdj not implemented for HIP" << std::endl;
    }
    template void gesvdj(Handle&, Jobz, DenseMatrix<float>&, float*,
                         DenseMatrix<float>&, DenseMatrix<float>&,
                         int*, float*, int, const double);
    template void gesvdj(Handle&, Jobz, DenseMatrix<double>&, double*,
                         DenseMatrix<double>&, DenseMatrix<double>&,
                         int*, double*, int, const double);
    template void gesvdj(Handle&, Jobz, DenseMatrix<std::complex<float>>&, float*,
                         DenseMatrix<std::complex<float>>&,
                         DenseMatrix<std::complex<float>>&,
                         int*, std::complex<float>*, int, const double);
    template void gesvdj(Handle&, Jobz, DenseMatrix<std::complex<double>>&, double*,
                         DenseMatrix<std::complex<double>>&,
                         DenseMatrix<std::complex<double>>&,
                         int*, std::complex<double>*, int, const double);


    void geam(hipblasHandle_t& handle, hipblasOperation_t transa,
              hipblasOperation_t transb, int m, int n,
              const float* alpha, const float* A, int lda,
              const float* beta, const float* B, int ldb, float* C, int ldc) {
      STRUMPACK_FLOPS(blas::geam_flops(m, n, alpha, beta));
      gpu_check(hipblasSgeam(handle, transa, transb, m, n, alpha,
                             A, lda, beta, B, ldb, C, ldc));
    }
    void geam(hipblasHandle_t& handle, hipblasOperation_t transa,
              hipblasOperation_t transb, int m, int n,
              const double* alpha, const double* A, int lda,
              const double* beta, const double* B, int ldb,
              double* C, int ldc) {
      STRUMPACK_FLOPS(blas::geam_flops(m, n, alpha, beta));
      gpu_check(hipblasDgeam(handle, transa, transb, m, n, alpha,
                             A, lda, beta, B, ldb, C, ldc));
    }
    void geam(hipblasHandle_t& handle, hipblasOperation_t transa,
              hipblasOperation_t transb, int m, int n,
              const std::complex<float>* alpha,
              const std::complex<float>* A, int lda,
              const std::complex<float>* beta,
              const std::complex<float>* B, int ldb,
              std::complex<float>* C, int ldc) {
      STRUMPACK_FLOPS(4*blas::geam_flops(m, n, alpha, beta));
      gpu_check
        (hipblasCgeam(handle, transa, transb, m, n,
                      reinterpret_cast<const hipblasComplex*>(alpha),
                      reinterpret_cast<const hipblasComplex*>(A), lda,
                      reinterpret_cast<const hipblasComplex*>(beta),
                      reinterpret_cast<const hipblasComplex*>(B), ldb,
                      reinterpret_cast<hipblasComplex*>(C), ldc));
    }
    void geam(hipblasHandle_t& handle, hipblasOperation_t transa,
              hipblasOperation_t transb, int m, int n,
              const std::complex<double>* alpha,
              const std::complex<double>* A, int lda,
              const std::complex<double>* beta,
              const std::complex<double>* B, int ldb,
              std::complex<double>* C, int ldc) {
      STRUMPACK_FLOPS(4*blas::geam_flops(m, n, alpha, beta));
      gpu_check
        (hipblasZgeam(handle, transa, transb, m, n,
                      reinterpret_cast<const hipblasDoubleComplex*>(alpha),
                      reinterpret_cast<const hipblasDoubleComplex*>(A), lda,
                      reinterpret_cast<const hipblasDoubleComplex*>(beta),
                      reinterpret_cast<const hipblasDoubleComplex*>(B), ldb,
                      reinterpret_cast<hipblasDoubleComplex*>(C), ldc));
    }

    template<typename scalar_t> void
    geam(Handle& handle, Trans transa, Trans transb, const scalar_t alpha,
         const DenseMatrix<scalar_t>& A, const scalar_t beta,
         const DenseMatrix<scalar_t>& B, DenseMatrix<scalar_t>& C){
      geam(get_hipblas_handle(handle), T2hipOp(transa), T2hipOp(transb),
           C.rows(), C.cols(), &alpha, A.data(), A.ld(), &beta,
           B.data(), B.ld(), C.data(), C.ld());
    }
    template void geam(Handle&, Trans, Trans, const float,
                       const DenseMatrix<float>&, const float,
                       const DenseMatrix<float>&, DenseMatrix<float>&);
    template void geam(Handle&, Trans, Trans, const double,
                       const DenseMatrix<double>&, const double,
                       const DenseMatrix<double>&, DenseMatrix<double>&);
    template void geam(Handle&, Trans, Trans, const std::complex<float>,
                       const DenseMatrix<std::complex<float>>&, const std::complex<float>,
                       const DenseMatrix<std::complex<float>>&, DenseMatrix<std::complex<float>>&);
    template void geam(Handle&, Trans, Trans, const std::complex<double>,
                       const DenseMatrix<std::complex<double>>&, const std::complex<double>,
                       const DenseMatrix<std::complex<double>>&, DenseMatrix<std::complex<double>>&);

    void dgmm(hipblasHandle_t& handle, hipblasSideMode_t side, int m, int n,
              const float* A, int lda, const float* x, int incx, float* C, int ldc) {
      STRUMPACK_FLOPS(blas::dgmm_flops(m, n));
      gpu_check(hipblasSdgmm(handle, side, m, n, A, lda, x, incx, C, ldc));
    }
    void dgmm(hipblasHandle_t& handle, hipblasSideMode_t side, int m, int n,
              const double* A, int lda, const double* x, int incx, double* C, int ldc) {
      STRUMPACK_FLOPS(blas::dgmm_flops(m, n));
      gpu_check(hipblasDdgmm(handle, side, m, n, A, lda, x, incx, C, ldc));
    }
    void dgmm(hipblasHandle_t& handle, hipblasSideMode_t side, int m, int n,
              const std::complex<float>* A, int lda,
              const std::complex<float>* x, int incx, std::complex<float>* C, int ldc) {
      STRUMPACK_FLOPS(4*blas::dgmm_flops(m, n));
      gpu_check
        (hipblasCdgmm(handle, side, m, n,
                      reinterpret_cast<const hipblasComplex*>(A), lda,
                      reinterpret_cast<const hipblasComplex*>(x), incx,
                      reinterpret_cast<hipblasComplex*>(C), ldc));
    }
    void dgmm(hipblasHandle_t& handle, hipblasSideMode_t side, int m, int n,
              const std::complex<double>* A, int lda,
              const std::complex<double>* x, int incx,
              std::complex<double>* C, int ldc) {
      STRUMPACK_FLOPS(4*blas::dgmm_flops(m, n));
      gpu_check
        (hipblasZdgmm(handle, side, m, n,
                      reinterpret_cast<const hipblasDoubleComplex*>(A), lda,
                      reinterpret_cast<const hipblasDoubleComplex*>(x), incx,
                      reinterpret_cast<hipblasDoubleComplex*>(C), ldc));
    }

    template<typename scalar_t> void
    dgmm(Handle& handle, Side side, const DenseMatrix<scalar_t>& A,
         const scalar_t* x, DenseMatrix<scalar_t>& C){
      dgmm(get_hipblas_handle(handle), S2hipOp(side), A.rows(), A.cols(),
           A.data(), A.ld(), x, 1, C.data(), C.ld());
    }
    template void
    dgmm(Handle&, Side, const DenseMatrix<float>&, const float*, DenseMatrix<float>&);
    template void
    dgmm(Handle&, Side, const DenseMatrix<double>&, const double*, DenseMatrix<double>&);
    template void
    dgmm(Handle&, Side, const DenseMatrix<std::complex<float>>&,
         const std::complex<float>*, DenseMatrix<std::complex<float>>&);
    template void
    dgmm(Handle&, Side, const DenseMatrix<std::complex<double>>&,
         const std::complex<double>*, DenseMatrix<std::complex<double>>&);

    void gemv(hipblasHandle_t& handle, hipblasOperation_t transa,
              int m, int n, float alpha,
              const float* A, int lda, const float* B, int incb,
              float beta, float* C, int incc) {
      STRUMPACK_FLOPS(blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(4*blas::gemv_moves(m,n));
      gpu_check(hipblasSgemv
                (handle, transa, m, n, &alpha,
                 A, lda, B, incb, &beta, C, incc));
    }
    void gemv(hipblasHandle_t& handle, hipblasOperation_t transa,
              int m, int n, double alpha,
              const double* A, int lda, const double* B, int incb,
              double beta, double* C, int incc) {
      STRUMPACK_FLOPS(blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(8*blas::gemv_moves(m,n));
      gpu_check(hipblasDgemv
                (handle, transa, m, n, &alpha,
                 A, lda, B, incb, &beta, C, incc));
    }
    void gemv(hipblasHandle_t& handle, hipblasOperation_t transa,
              int m, int n, std::complex<float> alpha,
              const std::complex<float>* A, int lda,
              const std::complex<float>* B, int incb,
              std::complex<float> beta,
              std::complex<float> *C, int incc) {
      STRUMPACK_FLOPS(4*blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(2*4*blas::gemv_moves(m,n));
      gpu_check(hipblasCgemv
                (handle, transa, m, n,
                 reinterpret_cast<hipblasComplex*>(&alpha),
                 reinterpret_cast<const hipblasComplex*>(A), lda,
                 reinterpret_cast<const hipblasComplex*>(B), incb,
                 reinterpret_cast<hipblasComplex*>(&beta),
                 reinterpret_cast<hipblasComplex*>(C), incc));
    }
    void gemv(hipblasHandle_t& handle, hipblasOperation_t transa,
              int m, int n, std::complex<double> alpha,
              const std::complex<double> *A, int lda,
              const std::complex<double> *B, int incb,
              std::complex<double> beta,
              std::complex<double> *C, int incc) {
      STRUMPACK_FLOPS(4*blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(2*8*blas::gemv_moves(m,n));
      gpu_check(hipblasZgemv
                (handle, transa, m, n,
                 reinterpret_cast<hipblasDoubleComplex*>(&alpha),
                 reinterpret_cast<const hipblasDoubleComplex*>(A), lda,
                 reinterpret_cast<const hipblasDoubleComplex*>(B), incb,
                 reinterpret_cast<hipblasDoubleComplex*>(&beta),
                 reinterpret_cast<hipblasDoubleComplex*>(C), incc));
    }
    template<typename scalar_t> void
    gemv(Handle& handle, Trans ta, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& x,
         scalar_t beta, DenseMatrix<scalar_t>& y) {
      gemv(get_hipblas_handle(handle), T2hipOp(ta), a.rows(), a.cols(),
           alpha, a.data(), a.ld(), x.data(), 1, beta, y.data(), 1);
    }
    template void gemv(Handle&, Trans, float, const DenseMatrix<float>&,
                       const DenseMatrix<float>&, float, DenseMatrix<float>&);
    template void gemv(Handle&, Trans, double, const DenseMatrix<double>&,
                       const DenseMatrix<double>&, double, DenseMatrix<double>&);
    template void gemv(Handle&, Trans, std::complex<float>,
                       const DenseMatrix<std::complex<float>>&,
                       const DenseMatrix<std::complex<float>>&,
                       std::complex<float>, DenseMatrix<std::complex<float>>&);
    template void gemv(Handle&, Trans, std::complex<double>,
                       const DenseMatrix<std::complex<double>>&,
                       const DenseMatrix<std::complex<double>>&,
                       std::complex<double>, DenseMatrix<std::complex<double>>&);

  } // end namespace gpu
} // end namespace strumpack
