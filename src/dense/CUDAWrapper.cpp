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

#include "CUDAWrapper.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "misc/MPIWrapper.hpp"
#endif

namespace strumpack {
  namespace gpu {

    void peek_at_last_error() {
      gpu_check(cudaPeekAtLastError());
    }

    void get_last_error() {
      // this is used to reset the last error. Some MAGMA calls fail
      // on purpose, then use a different algorithm
      auto e = cudaGetLastError();
      ((void)e); // silence unused warning
    }

    void synchronize_default_stream() {
      gpu_check(cudaStreamSynchronize(0));
    }

    struct Stream::StreamImpl {
      StreamImpl() { gpu_check(cudaStreamCreate(&s_)); }
      ~StreamImpl() { gpu_check(cudaStreamDestroy(s_)); }
      operator cudaStream_t&() { return s_; }
      operator const cudaStream_t&() const { return s_; }
      void synchronize() { gpu_check(cudaStreamSynchronize(s_)); }
      cudaStream_t s_;
    };
    Stream::Stream() { s_ = std::make_unique<StreamImpl>(); }
    Stream::~Stream() = default;
    void Stream::synchronize() { s_->synchronize(); }

    const cudaStream_t& get_cuda_stream(const Stream& s) { return *(s.s_); }
    cudaStream_t& get_cuda_stream(Stream& s) { return *(s.s_); }

    struct Handle::HandleImpl {
      HandleImpl() {
        gpu_check(cublasCreate(&bh_));
        gpu_check(cusolverDnCreate(&sh_));
#if defined(STRUMPACK_USE_MAGMA)
        magma_queue_create_from_cuda(0, NULL, bh_, NULL, &q_);
#endif
#if defined(STRUMPACK_USE_KBLAS)
        kblasCreate(&k_);
        kblasInitRandState(k_, &r_, 16384*2, 0);
        kblasEnableMagma(k_);
#endif
      }
      HandleImpl(Stream& s) {
        gpu_check(cublasCreate(&bh_));
        gpu_check(cusolverDnCreate(&sh_));
#if defined(STRUMPACK_USE_MAGMA)
        magma_queue_create_from_cuda(0, get_cuda_stream(s), bh_, NULL, &q_);
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
        gpu_check(cusolverDnDestroy(sh_));
        gpu_check(cublasDestroy(bh_));
      }
      void set_stream(Stream& s) {
        gpu_check(cublasSetStream(bh_, get_cuda_stream(s)));
        gpu_check(cusolverDnSetStream(sh_, get_cuda_stream(s)));
#if defined(STRUMPACK_USE_KBLAS)
        kblasSetStream(k_, get_cuda_stream(s));
#endif
      }

      operator cublasHandle_t&() { return bh_; }
      operator const cublasHandle_t&() const { return bh_; }
      operator cusolverDnHandle_t&() { return sh_; }
      operator const cusolverDnHandle_t&() const { return sh_; }
      cublasHandle_t bh_;
      cusolverDnHandle_t sh_;

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

    const cublasHandle_t& get_cublas_handle(const Handle& h) { return *(h.h_); }
    cublasHandle_t& get_cublas_handle(Handle& h) { return *(h.h_); }
    const cusolverDnHandle_t& get_cusolver_handle(const Handle& h) { return *(h.h_); }
    cusolverDnHandle_t& get_cusolver_handle(Handle& h) { return *(h.h_); }
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
        gpu_check(cudaEventCreateWithFlags
                  (&e_, cudaEventDisableTiming)); }
      ~EventImpl() { gpu_check(cudaEventDestroy(e_)); }
      void record() { gpu_check(cudaEventRecord(e_)); }
      void record(Stream& s) { gpu_check(cudaEventRecord(e_, get_cuda_stream(s))); }
      void wait(Stream& s) { gpu_check(cudaStreamWaitEvent(get_cuda_stream(s), e_, 0)); }
      void synchronize() { gpu_check(cudaEventSynchronize(e_));}
      cudaEvent_t e_;
    };
    Event::Event() { e_ = std::make_unique<EventImpl>(); }
    Event::~Event() = default;
    void Event::record() { e_->record(); }
    void Event::record(Stream& s) { e_->record(s); }
    void Event::wait(Stream& s) { e_->wait(s); }
    void Event::synchronize() { e_->synchronize(); }

    void device_malloc(void** ptr, std::size_t size) {
      if (cudaMalloc(ptr, size) != cudaSuccess) {
        std::cerr << "CUDA Failed to allocate " << size << " bytes on device" << std::endl;
        throw std::bad_alloc();
      }
    }
    void host_malloc(void** ptr, std::size_t size) {
      if (cudaMallocHost(ptr, size) != cudaSuccess) {
        std::cerr << "CUDA Failed to allocate " << size << " bytes on host " << std::endl;
        throw std::bad_alloc();
      }
    }
    void device_free(void* ptr) { gpu_check(cudaFree(ptr)); }
    void host_free(void* ptr) { gpu_check(cudaFreeHost(ptr)); }

    cudaMemcpyKind CD2cuMK(CopyDir d) {
      switch (d) {
      case CopyDir::H2H: return cudaMemcpyHostToHost;
      case CopyDir::H2D: return cudaMemcpyHostToDevice;
      case CopyDir::D2H: return cudaMemcpyDeviceToHost;
      case CopyDir::D2D: return cudaMemcpyDeviceToDevice;
      case CopyDir::DEF: return cudaMemcpyDefault;
      default: assert(false); return cudaMemcpyDefault;
      }
    }

    cublasOperation_t T2cuOp(Trans op) {
      switch (op) {
      case Trans::N: return CUBLAS_OP_N;
      case Trans::T: return CUBLAS_OP_T;
      case Trans::C: return CUBLAS_OP_C;
      default: assert(false); return CUBLAS_OP_N;
      }
    }

    cublasFillMode_t F2cuOp(UpLo op) {
      switch (op) {
      case UpLo::U: return CUBLAS_FILL_MODE_UPPER;
      case UpLo::L: return CUBLAS_FILL_MODE_LOWER;
      case UpLo::F: return CUBLAS_FILL_MODE_FULL;
      default:
        assert(false);
        return CUBLAS_FILL_MODE_LOWER;
      }
    }

    cublasSideMode_t S2cuOp(Side op) {
      switch (op) {
      case Side::L: return CUBLAS_SIDE_LEFT;
      case Side::R: return CUBLAS_SIDE_RIGHT;
      default: assert(false); return CUBLAS_SIDE_LEFT;
      }
    }

    cublasFillMode_t U2cuOp(UpLo op) {
      switch (op) {
      case UpLo::L: return CUBLAS_FILL_MODE_LOWER;
      case UpLo::U: return CUBLAS_FILL_MODE_UPPER;
      default: assert(false); return CUBLAS_FILL_MODE_LOWER;
      }
    }

    cublasDiagType_t D2cuOp(Diag op) {
      switch (op) {
      case Diag::N: return CUBLAS_DIAG_NON_UNIT;
      case Diag::U: return CUBLAS_DIAG_UNIT;
      default: assert(false); return CUBLAS_DIAG_UNIT;
      }
    }

    cusolverEigMode_t E2cuOp(Jobz op) {
      switch (op) {
      case Jobz::N: return CUSOLVER_EIG_MODE_NOVECTOR;
      case Jobz::V: return CUSOLVER_EIG_MODE_VECTOR;
      default: assert(false); return CUSOLVER_EIG_MODE_VECTOR;
      }
    }

    void cuda_assert(cudaError_t code, const char *file, int line,
                     bool abrt) {
      if (code != cudaSuccess) {
        std::cerr << "CUDA assertion failed: "
                  << cudaGetErrorString(code) << " "
                  <<  file << " " << line << std::endl;
        abort();
        //if (abrt) exit(code);
      }
    }
    void cuda_assert(cusolverStatus_t code, const char *file, int line,
                     bool abrt) {
      if (code != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER assertion failed: " << code << " "
                  <<  file << " " << line << std::endl;
        switch (code) {
        case CUSOLVER_STATUS_SUCCESS:                                 std::cerr << "CUSOLVER_STATUS_SUCCESS" << std::endl; break;
        case CUSOLVER_STATUS_NOT_INITIALIZED:                         std::cerr << "CUSOLVER_STATUS_NOT_INITIALIZED" << std::endl; break;
        case CUSOLVER_STATUS_ALLOC_FAILED:                            std::cerr << "CUSOLVER_STATUS_ALLOC_FAILED" << std::endl; break;
        case CUSOLVER_STATUS_INVALID_VALUE:                           std::cerr << "CUSOLVER_STATUS_INVALID_VALUE" << std::endl; break;
        case CUSOLVER_STATUS_ARCH_MISMATCH:                           std::cerr << "CUSOLVER_STATUS_ARCH_MISMATCH" << std::endl; break;
        case CUSOLVER_STATUS_EXECUTION_FAILED:                        std::cerr << "CUSOLVER_STATUS_EXECUTION_FAILED" << std::endl; break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:                          std::cerr << "CUSOLVER_STATUS_INTERNAL_ERROR" << std::endl; break;
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:               std::cerr << "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED" << std::endl; break;
        case CUSOLVER_STATUS_MAPPING_ERROR:                           std::cerr << "CUSOLVER_STATUS_MAPPING_ERROR" << std::endl; break;
        case CUSOLVER_STATUS_NOT_SUPPORTED:                           std::cerr << "CUSOLVER_STATUS_NOT_SUPPORTED" << std::endl; break;
        case CUSOLVER_STATUS_ZERO_PIVOT:                              std::cerr << "CUSOLVER_STATUS_ZERO_PIVOT" << std::endl; break;
        case CUSOLVER_STATUS_INVALID_LICENSE:                         std::cerr << "CUSOLVER_STATUS_INVALID_LICENSE" << std::endl; break;
          // case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:              std::cerr << "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED" << std::endl; break;
          // case CUSOLVER_STATUS_IRS_PARAMS_INVALID:                      std::cerr << "CUSOLVER_STATUS_IRS_PARAMS_INVALID" << std::endl; break;
          // case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC:                 std::cerr << "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC" << std::endl; break;
          // case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE:               std::cerr << "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE" << std::endl; break;
          // case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER:              std::cerr << "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER" << std::endl; break;
          // case CUSOLVER_STATUS_IRS_INTERNAL_ERROR:                      std::cerr << "CUSOLVER_STATUS_IRS_INTERNAL_ERROR" << std::endl; break;
          // case CUSOLVER_STATUS_IRS_NOT_SUPPORTED:                       std::cerr << "CUSOLVER_STATUS_IRS_NOT_SUPPORTED" << std::endl; break;
          // case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:                        std::cerr << "CUSOLVER_STATUS_IRS_OUT_OF_RANGE" << std::endl; break;
          // case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES: std::cerr << "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES" << std::endl; break;
          // case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED:               std::cerr << "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED" << std::endl; break;
          // case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED:                 std::cerr << "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED" << std::endl; break;
          // case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR:                     std::cerr << "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR" << std::endl; break;
          // case CUSOLVER_STATUS_INVALID_WORKSPACE:                       std::cerr << "CUSOLVER_STATUS_INVALID_WORKSPACE" << std::endl; break;
        default: std::cerr << "unknown cusolver error" << std::endl;
        }
        if (abrt) exit(code);
      }
    }
    void cuda_assert(cublasStatus_t code, const char *file, int line,
                     bool abrt) {
      if (code != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS assertion failed: " << code << " "
                  <<  file << " " << line << std::endl;
        switch (code) {
        case CUBLAS_STATUS_SUCCESS:          std::cerr << "CUBLAS_STATUS_SUCCESS" << std::endl; break;
        case CUBLAS_STATUS_NOT_INITIALIZED:  std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED" << std::endl; break;
        case CUBLAS_STATUS_ALLOC_FAILED:     std::cerr << "CUBLAS_STATUS_ALLOC_FAILED" << std::endl; break;
        case CUBLAS_STATUS_INVALID_VALUE:    std::cerr << "CUBLAS_STATUS_INVALID_VALUE" << std::endl; break;
        case CUBLAS_STATUS_ARCH_MISMATCH:    std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH" << std::endl; break;
        case CUBLAS_STATUS_MAPPING_ERROR:    std::cerr << "CUBLAS_STATUS_MAPPING_ERROR" << std::endl; break;
        case CUBLAS_STATUS_EXECUTION_FAILED: std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED" << std::endl; break;
        case CUBLAS_STATUS_INTERNAL_ERROR:   std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR" << std::endl; break;
        case CUBLAS_STATUS_NOT_SUPPORTED:    std::cerr << "CUBLAS_STATUS_NOT_SUPPORTED" << std::endl; break;
        case CUBLAS_STATUS_LICENSE_ERROR:    std::cerr << "CUBLAS_STATUS_LICENSE_ERROR" << std::endl; break;
        default: std::cerr << "unknown cublas error" << std::endl;
        }
        if (abrt) exit(code);
      }
    }

    void init() {
#if defined(STRUMPACK_USE_MPI)
      int devs;
      cudaGetDeviceCount(&devs);
      if (devs > 1) {
        int flag, rank = 0;
        MPI_Initialized(&flag);
        if (flag) {
          MPIComm c;
          rank = c.rank();
        }
        cudaSetDevice(rank % devs);
      }
#endif
      //       gpu_check(cudaFree(0));
      // #if defined(STRUMPACK_USE_MAGMA)
      //       magma_init();
      // #endif
      //       gpu::BLASHandle hb;
      //       gpu::SOLVERHandle hs;
    }


    void device_memset(void* dptr, int value, std::size_t count) {
      gpu_check(cudaMemset(dptr, value, count));
    }

    void device_copy(void* dest, const void* src,
                     std::size_t count, CopyDir dir) {
      gpu_check(cudaMemcpy(dest, src, count, CD2cuMK(dir)));
    }
    void device_copy_async(void* dest, const void* src, std::size_t count,
                           CopyDir dir, Stream& s) {
      gpu_check(cudaMemcpyAsync(dest, src, count, CD2cuMK(dir),
                                get_cuda_stream(s)));
    }
    void device_copy_2D(void* dest, std::size_t dpitch,
                        const void* src, std::size_t spitch,
                        std::size_t width, std::size_t height, CopyDir dir) {
      gpu_check(cudaMemcpy2D(dest, dpitch, src, spitch,
                             width , height, CD2cuMK(dir)));
    }
    void device_copy_2D_async(void* dest, std::size_t dpitch,
                              const void* src, std::size_t spitch,
                              std::size_t width, std::size_t height,
                              CopyDir dir, Stream& s) {
      gpu_check(cudaMemcpy2DAsync(dest, dpitch, src, spitch, width, height,
                                  CD2cuMK(dir), get_cuda_stream(s)));
    }

    std::size_t available_memory() {
      std::size_t free_device_mem, total_device_mem;
      gpu_check(cudaMemGetInfo(&free_device_mem, &total_device_mem));
      return free_device_mem;
    }

    void gemm(cublasHandle_t& handle, cublasOperation_t transa,
              cublasOperation_t transb, int m, int n, int k,
              float alpha, const float* A, int lda,
              const float* B, int ldb,
              float beta, float* C, int ldc) {
      STRUMPACK_FLOPS(blas::gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(4*blas::gemm_moves(m,n,k));
      gpu_check(cublasSgemm
                (handle, transa, transb, m, n, k, &alpha,
                 A, lda, B, ldb, &beta, C, ldc));
    }
    void gemm(cublasHandle_t& handle, cublasOperation_t transa,
              cublasOperation_t transb, int m, int n, int k,
              double alpha, const double* A, int lda,
              const double* B, int ldb,
              double beta, double* C, int ldc) {
      STRUMPACK_FLOPS(blas::gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(8*blas::gemm_moves(m,n,k));
      gpu_check(cublasDgemm
                (handle, transa, transb, m, n, k, &alpha, A, lda,
                 B, ldb, &beta, C, ldc));
    }
    void gemm(cublasHandle_t& handle, cublasOperation_t transa,
              cublasOperation_t transb, int m, int n, int k,
              std::complex<float> alpha,
              const std::complex<float>* A, int lda,
              const std::complex<float>* B, int ldb,
              std::complex<float> beta, std::complex<float> *C,
              int ldc) {
      STRUMPACK_FLOPS(4*blas::gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(2*4*blas::gemm_moves(m,n,k));
      gpu_check(cublasCgemm
                (handle, transa, transb, m, n, k,
                 reinterpret_cast<cuComplex*>(&alpha),
                 reinterpret_cast<const cuComplex*>(A), lda,
                 reinterpret_cast<const cuComplex*>(B), ldb,
                 reinterpret_cast<cuComplex*>(&beta),
                 reinterpret_cast<cuComplex*>(C), ldc));
    }
    void gemm(cublasHandle_t& handle, cublasOperation_t transa,
              cublasOperation_t transb, int m, int n, int k,
              std::complex<double> alpha,
              const std::complex<double> *A, int lda,
              const std::complex<double> *B, int ldb,
              std::complex<double> beta,
              std::complex<double> *C, int ldc) {
      STRUMPACK_FLOPS(4*blas::gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(2*8*blas::gemm_moves(m,n,k));
      gpu_check(cublasZgemm
                (handle, transa, transb, m, n, k,
                 reinterpret_cast<cuDoubleComplex*>(&alpha),
                 reinterpret_cast<const cuDoubleComplex*>(A), lda,
                 reinterpret_cast<const cuDoubleComplex*>(B), ldb,
                 reinterpret_cast<cuDoubleComplex*>(&beta),
                 reinterpret_cast<cuDoubleComplex*>(C), ldc));
    }

    template<typename scalar_t> void
    gemm(Handle& handle, Trans ta, Trans tb,
         scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const DenseMatrix<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c) {
      assert((ta==Trans::N && a.rows()==c.rows()) ||
             (ta!=Trans::N && a.cols()==c.rows()));
      assert((tb==Trans::N && b.cols()==c.cols()) ||
             (tb!=Trans::N && b.rows()==c.cols()));
      assert((ta==Trans::N && tb==Trans::N && a.cols()==b.rows()) ||
             (ta!=Trans::N && tb==Trans::N && a.rows()==b.rows()) ||
             (ta==Trans::N && tb!=Trans::N && a.cols()==b.cols()) ||
             (ta!=Trans::N && tb!=Trans::N && a.rows()==b.cols()));
      gemm(get_cublas_handle(handle), T2cuOp(ta), T2cuOp(tb), c.rows(), c.cols(),
           (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(), a.ld(),
           b.data(), b.ld(), beta, c.data(), c.ld());
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

    void syrk(Handle& handle, cublasFillMode_t uplo,
              cublasOperation_t transa, int n, int k,
              float alpha, const float* A, int lda,
              float beta, float* C, int ldc) {
      STRUMPACK_FLOPS(blas::gemm_flops(n,n,k,alpha,beta));
      STRUMPACK_BYTES(4*blas::gemm_moves(n,n,k));
      gpu_check(cublasSsyrk_v2(get_cublas_handle(handle), uplo, transa, n, k, &alpha, A, lda, &beta, C, ldc));
    }
    void syrk(Handle& handle, cublasFillMode_t uplo,
              cublasOperation_t transa, int n, int k,
              double alpha, const double* A, int lda,
              double beta, double* C, int ldc) {
      STRUMPACK_FLOPS(blas::gemm_flops(n,n,k,alpha,beta));
      STRUMPACK_BYTES(8*blas::gemm_moves(n,n,k));
      gpu_check(cublasDsyrk_v2(get_cublas_handle(handle), uplo, transa, n, k, &alpha, A, lda, &beta, C, ldc));
    }
    void syrk(Handle& handle, cublasFillMode_t uplo,
              cublasOperation_t transa, int n, int k,
              std::complex<float> alpha,
              const std::complex<float>* A, int lda,
              std::complex<float> beta, std::complex<float> *C,
              int ldc) {
      STRUMPACK_FLOPS(4*blas::gemm_flops(n,n,k,alpha,beta));
      STRUMPACK_BYTES(2*4*blas::gemm_moves(n,n,k));
      gpu_check(cublasCsyrk_v2(get_cublas_handle(handle), uplo, transa, n, k,
                               reinterpret_cast<cuComplex*>(&alpha),
                               reinterpret_cast<const cuComplex*>(A), lda,
                               reinterpret_cast<cuComplex*>(&beta),
                               reinterpret_cast<cuComplex*>(C), ldc));
    }
    void syrk(Handle& handle, cublasFillMode_t uplo,
              cublasOperation_t transa, int n, int k,
              std::complex<double> alpha,
              const std::complex<double> *A, int lda,
              std::complex<double> beta,
              std::complex<double> *C, int ldc) {
      STRUMPACK_FLOPS(4*blas::gemm_flops(n,n,k,alpha,beta));
      STRUMPACK_BYTES(2*8*blas::gemm_moves(n,n,k));
      gpu_check(cublasZsyrk_v2(get_cublas_handle(handle), uplo, transa, n, k,
                               reinterpret_cast<cuDoubleComplex*>(&alpha),
                               reinterpret_cast<const cuDoubleComplex*>(A), lda,
                               reinterpret_cast<cuDoubleComplex*>(&beta),
                               reinterpret_cast<cuDoubleComplex*>(C), ldc));
    }

    template<typename scalar_t> void
    syrk(Handle& handle, UpLo uplo, Trans ta,
         scalar_t alpha, const DenseMatrix<scalar_t>& a,
         scalar_t beta, DenseMatrix<scalar_t>& c) {
      assert((ta==Trans::N && a.rows()==c.rows()) ||
             (ta!=Trans::N && a.cols()==c.rows()));
      syrk(handle, F2cuOp(uplo), T2cuOp(ta), c.rows(),
           (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(), a.ld(),
           beta, c.data(), c.ld());
    }
    template void syrk(Handle&, UpLo, Trans,
                       float, const DenseMatrix<float>&,
                       float, DenseMatrix<float>&);
    template void syrk(Handle&, UpLo, Trans,
                       double, const DenseMatrix<double>&,
                       double, DenseMatrix<double>&);
    template void syrk(Handle&, UpLo, Trans, std::complex<float>,
                       const DenseMatrix<std::complex<float>>&,
                       std::complex<float>,
                       DenseMatrix<std::complex<float>>&);
    template void syrk(Handle&, UpLo, Trans, std::complex<double>,
                       const DenseMatrix<std::complex<double>>&,
                       std::complex<double>,
                       DenseMatrix<std::complex<double>>&);

    void getrf_buffersize(cusolverDnHandle_t& handle, int m, int n, float* A, int lda, int* Lwork) {
      gpu_check(cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork));
    }
    void getrf_buffersize(cusolverDnHandle_t& handle, int m, int n, double *A, int lda, int* Lwork) {
      gpu_check(cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork));
    }
    void getrf_buffersize(cusolverDnHandle_t& handle, int m, int n, std::complex<float>* A, int lda, int *Lwork) {
      gpu_check(cusolverDnCgetrf_bufferSize(handle, m, n, reinterpret_cast<cuComplex*>(A), lda, Lwork));
    }
    void getrf_buffersize(cusolverDnHandle_t& handle, int m, int n, std::complex<double>* A, int lda, int *Lwork) {
      gpu_check(cusolverDnZgetrf_bufferSize(handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, Lwork));
    }

    template<typename scalar_t>
    std::int64_t getrf_buffersize(Handle& handle, int n) {
      int Lwork;
      getrf_buffersize(get_cusolver_handle(handle), n, n, static_cast<scalar_t*>(nullptr), n, &Lwork);
      return Lwork;
    }
    template std::int64_t getrf_buffersize<float>(Handle&, int);
    template std::int64_t getrf_buffersize<double>(Handle&, int);
    template std::int64_t getrf_buffersize<std::complex<float>>(Handle&, int);
    template std::int64_t getrf_buffersize<std::complex<double>>(Handle&, int);


    void getrf(cusolverDnHandle_t& handle, int m, int n, float* A, int lda,
               float* work, std::size_t, int* dpiv, int* dinfo) {
      STRUMPACK_FLOPS(blas::getrf_flops(m,n));
      gpu_check(cusolverDnSgetrf
                (handle, m, n, A, lda, work, dpiv, dinfo));
    }
    void getrf(cusolverDnHandle_t& handle, int m, int n, double* A, int lda,
               double* work, std::int64_t, int* dpiv, int* dinfo) {
      STRUMPACK_FLOPS(blas::getrf_flops(m,n));
      gpu_check(cusolverDnDgetrf
                (handle, m, n, A, lda, work, dpiv, dinfo));
    }
    void getrf(cusolverDnHandle_t& handle, int m, int n, std::complex<float>* A, int lda,
               std::complex<float>* work, std::int64_t, int* dpiv, int* dinfo) {
      STRUMPACK_FLOPS(4*blas::getrf_flops(m,n));
      gpu_check(cusolverDnCgetrf
                (handle, m, n, reinterpret_cast<cuComplex*>(A), lda,
                 reinterpret_cast<cuComplex*>(work), dpiv, dinfo));
    }
    void getrf(cusolverDnHandle_t& handle, int m, int n, std::complex<double>* A, int lda,
               std::complex<double>* work, std::int64_t, int* dpiv, int* dinfo) {
      STRUMPACK_FLOPS(4*blas::getrf_flops(m,n));
      gpu_check(cusolverDnZgetrf
                (handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda,
                 reinterpret_cast<cuDoubleComplex*>(work),
                 dpiv, dinfo));
    }

    template<typename scalar_t> void
    getrf(Handle& handle, DenseMatrix<scalar_t>& A,
          scalar_t* work, std::int64_t lwork, int* dpiv, int* dinfo) {
      getrf(get_cusolver_handle(handle), A.rows(), A.cols(), A.data(), A.ld(),
            work, lwork, dpiv, dinfo);
    }
    template void getrf(Handle&, DenseMatrix<float>&, float*, std::int64_t, int*, int*);
    template void getrf(Handle&, DenseMatrix<double>&, double*, std::int64_t, int*, int*);
    template void getrf(Handle&, DenseMatrix<std::complex<float>>&, std::complex<float>*, std::int64_t, int*, int*);
    template void getrf(Handle&, DenseMatrix<std::complex<double>>& A, std::complex<double>*, std::int64_t, int*, int*);

    template<typename scalar_t> std::int64_t
    getrs_buffersize(Handle& handle, Trans, int, int, int, int) {
      return 0;
    }
    template std::int64_t getrs_buffersize<float>(Handle&, Trans, int, int, int, int);
    template std::int64_t getrs_buffersize<double>(Handle&, Trans, int, int, int, int);
    template std::int64_t getrs_buffersize<std::complex<float>>(Handle&, Trans, int, int, int, int);
    template std::int64_t getrs_buffersize<std::complex<double>>(Handle&, Trans, int, int, int, int);

    void getrs(cusolverDnHandle_t& handle, cublasOperation_t trans,
               int n, int nrhs, const float* A, int lda,
               const int* dpiv, float* B, int ldb, int* dinfo) {
      STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
      gpu_check(cusolverDnSgetrs
                (handle, trans, n, nrhs, A, lda, dpiv, B, ldb, dinfo));
    }
    void getrs(cusolverDnHandle_t& handle, cublasOperation_t trans,
               int n, int nrhs, const double* A, int lda,
               const int* dpiv, double* B, int ldb, int* dinfo) {
      STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
      gpu_check(cusolverDnDgetrs
                (handle, trans, n, nrhs, A, lda, dpiv, B, ldb, dinfo));
    }
    void getrs(cusolverDnHandle_t& handle, cublasOperation_t trans,
               int n, int nrhs, const std::complex<float>* A, int lda,
               const int* dpiv, std::complex<float>* B, int ldb,
               int* dinfo) {
      STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
      gpu_check(cusolverDnCgetrs
                (handle, trans, n, nrhs,
                 reinterpret_cast<const cuComplex*>(A), lda,
                 dpiv, reinterpret_cast<cuComplex*>(B), ldb, dinfo));
    }
    void getrs(cusolverDnHandle_t& handle, cublasOperation_t trans,
               int n, int nrhs, const std::complex<double>* A, int lda,
               const int* dpiv, std::complex<double>* B, int ldb,
               int *dinfo) {
      STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
      gpu_check(cusolverDnZgetrs
                (handle, trans, n, nrhs,
                 reinterpret_cast<const cuDoubleComplex*>(A), lda, dpiv,
                 reinterpret_cast<cuDoubleComplex*>(B), ldb, dinfo));
    }
    template<typename scalar_t> void
    getrs(Handle& handle, Trans trans,
          const DenseMatrix<scalar_t>& A, const int* dpiv,
          DenseMatrix<scalar_t>& B, int *dinfo, scalar_t*, std::int64_t) {
      getrs(get_cusolver_handle(handle), T2cuOp(trans), A.rows(), B.cols(),
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

    void potrf_buffersize
    (Handle& handle, cublasFillMode_t uplo, int m, float* A, int lda, int* Lwork) {
      gpu_check(cusolverDnSpotrf_bufferSize(get_cusolver_handle(handle), uplo, m, A, lda, Lwork));
    }
    void potrf_buffersize
    (Handle& handle, cublasFillMode_t uplo, int m, double * A, int lda, int* Lwork) {
      gpu_check(cusolverDnDpotrf_bufferSize(get_cusolver_handle(handle), uplo, m, A, lda, Lwork));
    }
    void potrf_buffersize
    (Handle& handle, cublasFillMode_t uplo, int m, std::complex<float>* A, int lda,
     int *Lwork) {
      gpu_check(cusolverDnCpotrf_bufferSize(get_cusolver_handle(handle), uplo, m, reinterpret_cast<cuComplex*>(A), lda, Lwork));
    }
    void potrf_buffersize
    (Handle& handle, cublasFillMode_t uplo, int m, std::complex<double>* A, int lda,
     int *Lwork) {
      gpu_check(cusolverDnZpotrf_bufferSize(get_cusolver_handle(handle), uplo, m, reinterpret_cast<cuDoubleComplex*>(A), lda, Lwork));
    }

    template<typename scalar_t>
    int potrf_buffersize(Handle& handle, UpLo uplo, int n) {
      int Lwork;
      potrf_buffersize(handle, F2cuOp(uplo), n, static_cast<scalar_t*>(nullptr), n, &Lwork);
      return Lwork;
    }
    template int potrf_buffersize<float>(Handle&, UpLo, int);
    template int potrf_buffersize<double>(Handle&, UpLo, int);
    template int potrf_buffersize<std::complex<float>>(Handle&, UpLo, int);
    template int potrf_buffersize<std::complex<double>>(Handle&, UpLo, int);


    void potrf(Handle& handle, cublasFillMode_t uplo, int m, float* A, int lda,
               float* Workspace, int Lwork, int* devInfo) {
      STRUMPACK_FLOPS(blas::potrf_flops(m));
      gpu_check(cusolverDnSpotrf(get_cusolver_handle(handle), uplo, m, A, lda, Workspace, Lwork, devInfo));
    }
    void potrf(Handle& handle, cublasFillMode_t uplo, int m, double* A, int lda,
               double* Workspace, int Lwork, int* devInfo) {
      STRUMPACK_FLOPS(blas::potrf_flops(m));
      gpu_check(cusolverDnDpotrf(get_cusolver_handle(handle), uplo, m, A, lda, Workspace, Lwork, devInfo));
    }
    void potrf(Handle& handle, cublasFillMode_t uplo, int m, std::complex<float>* A, int lda,
               std::complex<float>* Workspace, int Lwork, int* devInfo) {
      STRUMPACK_FLOPS(4*blas::potrf_flops(m));
      gpu_check(cusolverDnCpotrf(get_cusolver_handle(handle), uplo, m, reinterpret_cast<cuComplex*>(A), lda,
                                 reinterpret_cast<cuComplex*>(Workspace), Lwork, devInfo));
    }
    void potrf(Handle& handle, cublasFillMode_t uplo, int m, std::complex<double>* A, int lda,
               std::complex<double>* Workspace, int Lwork, int* devInfo) {
      STRUMPACK_FLOPS(4*blas::potrf_flops(m));
      gpu_check(cusolverDnZpotrf(get_cusolver_handle(handle), uplo, m, reinterpret_cast<cuDoubleComplex*>(A), lda,
                                 reinterpret_cast<cuDoubleComplex*>(Workspace), Lwork, devInfo));
    }

    template<typename scalar_t> void
    potrf(Handle& handle, UpLo uplo, DenseMatrix<scalar_t>& A,
          scalar_t* Workspace, int Lwork, int* devInfo) {
      potrf(handle, F2cuOp(uplo), A.rows(), A.data(), A.ld(), Workspace, Lwork, devInfo);
    }
    template void potrf(Handle&, UpLo, DenseMatrix<float>&,
                        float*, int, int*);
    template void potrf(Handle&, UpLo, DenseMatrix<double>&,
                        double*, int, int*);
    template void potrf(Handle&, UpLo, DenseMatrix<std::complex<float>>&,
                        std::complex<float>*, int, int*);
    template void potrf(Handle&, UpLo, DenseMatrix<std::complex<double>>&,
                        std::complex<double>*, int, int*);

    void trsm(cublasHandle_t& handle, cublasSideMode_t side,
              cublasFillMode_t uplo, cublasOperation_t trans,
              cublasDiagType_t diag, int m, int n, const float* alpha,
              const float* A, int lda, float* B, int ldb) {
      STRUMPACK_FLOPS(blas::trsm_flops(m, n, alpha, side));
      STRUMPACK_BYTES(4*blas::trsm_moves(m, n));
      gpu_check(cublasStrsm(handle, side, uplo, trans, diag, m,
                            n, alpha, A, lda, B, ldb));
    }
    void trsm(cublasHandle_t& handle, cublasSideMode_t side,
              cublasFillMode_t uplo, cublasOperation_t trans,
              cublasDiagType_t diag, int m, int n, const double* alpha,
              const double* A, int lda, double* B, int ldb) {
      STRUMPACK_FLOPS(blas::trsm_flops(m, n, alpha, side));
      STRUMPACK_BYTES(8*blas::trsm_moves(m, n));
      gpu_check(cublasDtrsm(handle, side, uplo, trans, diag, m,
                            n, alpha, A, lda, B, ldb));
    }
    void trsm(cublasHandle_t& handle, cublasSideMode_t side,
              cublasFillMode_t uplo, cublasOperation_t trans,
              cublasDiagType_t diag, int m, int n,
              const std::complex<float>* alpha, const std::complex<float>* A,
              int lda, std::complex<float>* B, int ldb) {
      STRUMPACK_FLOPS(4*blas::trsm_flops(m, n, alpha, side));
      STRUMPACK_BYTES(2*4*blas::trsm_moves(m, n));
      gpu_check(cublasCtrsm(handle, side, uplo, trans, diag, m, n,
                            reinterpret_cast<const cuComplex*>(alpha),
                            reinterpret_cast<const cuComplex*>(A), lda,
                            reinterpret_cast<cuComplex*>(B), ldb));
    }
    void trsm(cublasHandle_t& handle, cublasSideMode_t side,
              cublasFillMode_t uplo, cublasOperation_t trans,
              cublasDiagType_t diag, int m, int n,
              const std::complex<double>* alpha,
              const std::complex<double>* A,
              int lda, std::complex<double>* B, int ldb) {
      STRUMPACK_FLOPS(4*blas::trsm_flops(m, n, alpha, side));
      STRUMPACK_BYTES(2*8*blas::trsm_moves(m, n));
      gpu_check(cublasZtrsm(handle, side, uplo, trans, diag, m, n,
                            reinterpret_cast<const cuDoubleComplex*>(alpha),
                            reinterpret_cast<const cuDoubleComplex*>(A), lda,
                            reinterpret_cast<cuDoubleComplex*>(B), ldb));
    }
    template<typename scalar_t> void
    trsm(Handle& handle, Side side, UpLo uplo,
         Trans trans, Diag diag, const scalar_t alpha,
         DenseMatrix<scalar_t>& A, DenseMatrix<scalar_t>& B) {
      trsm(get_cublas_handle(handle),
           S2cuOp(side), U2cuOp(uplo), T2cuOp(trans), D2cuOp(diag),
           B.rows(), B.cols(), &alpha, A.data(), A.ld(), B.data(), B.ld());
    }
    template void trsm(Handle&, Side, UpLo, Trans, Diag, const float,
                       DenseMatrix<float>&, DenseMatrix<float>&);
    template void trsm(Handle&, Side, UpLo, Trans, Diag, const double,
                       DenseMatrix<double>&, DenseMatrix<double>&);
    template void trsm(Handle&, Side, UpLo, Trans, Diag, const std::complex<float>,
                       DenseMatrix<std::complex<float>>&,
                       DenseMatrix<std::complex<float>>&);
    template void trsm(Handle&, Side, UpLo, Trans, Diag, const std::complex<double>,
                       DenseMatrix<std::complex<double>>&,
                       DenseMatrix<std::complex<double>>&);

    void gesvdj_info_create(gesvdjInfo_t *info) {
      gpu_check(cusolverDnCreateGesvdjInfo(info));
    }
    void gesvdj_info_destroy(gesvdjInfo_t& info) {
      gpu_check(cusolverDnDestroyGesvdjInfo(info));
    }

    void gesvdj_buffersize(cusolverDnHandle_t& handle, cusolverEigMode_t jobz,
                           int econ, int m, int n, const float* A, int lda,
                           const float* S, const float* U, int ldu,
                           const float* V, int ldv, int* lwork,
                           gesvdjInfo_t& params) {
      gpu_check(cusolverDnSgesvdj_bufferSize
                (handle, jobz, econ, m, n, A, lda, S,
                 U, ldu, V, ldv, lwork, params));
    }
    void gesvdj_buffersize(cusolverDnHandle_t& handle, cusolverEigMode_t jobz,
                           int econ, int m, int n, const double* A, int lda,
                           const double* S, const double* U, int ldu,
                           const double* V, int ldv, int* lwork,
                           gesvdjInfo_t& params) {
      gpu_check(cusolverDnDgesvdj_bufferSize
                (handle, jobz, econ, m, n, A, lda, S,
                 U, ldu, V, ldv, lwork, params));
    }
    void gesvdj_buffersize(cusolverDnHandle_t& handle, cusolverEigMode_t jobz,
                           int econ, int m, int n, const std::complex<float>* A, int lda,
                           const float* S, const std::complex<float>* U, int ldu,
                           const std::complex<float>* V,
                           int ldv, int* lwork, gesvdjInfo_t& params) {
      gpu_check(cusolverDnCgesvdj_bufferSize
                (handle, jobz, econ, m, n,
                 reinterpret_cast<const cuComplex*>(A), lda, S,
                 reinterpret_cast<const cuComplex*>(U), ldu,
                 reinterpret_cast<const cuComplex*>(V), ldv,
                 lwork, params));
    }
    void gesvdj_buffersize(cusolverDnHandle_t& handle, cusolverEigMode_t jobz,
                           int econ, int m, int n, const std::complex<double>* A, int lda,
                           const double* S, const std::complex<double>* U, int ldu,
                           const std::complex<double>* V,
                           int ldv, int* lwork, gesvdjInfo_t& params) {
      gpu_check(cusolverDnZgesvdj_bufferSize
                (handle, jobz, econ, m, n,
                 reinterpret_cast<const cuDoubleComplex*>(A), lda, S,
                 reinterpret_cast<const cuDoubleComplex*>(U), ldu,
                 reinterpret_cast<const cuDoubleComplex*>(V), ldv,
                 lwork, params));
    }
    template<typename scalar_t, typename real_t>
    int gesvdj_buffersize(Handle& handle, Jobz jobz, int m, int n) {
      gesvdjInfo_t params = nullptr;
      int Lwork;
      gesvdj_buffersize
        (get_cusolver_handle(handle),
         E2cuOp(jobz), 1, m, n, static_cast<scalar_t*>(nullptr), m,
         static_cast<real_t*>(nullptr), static_cast<scalar_t*>(nullptr), m,
         static_cast<scalar_t*>(nullptr), n, &Lwork, params);
      return Lwork;
    }
    template int gesvdj_buffersize<float,float>(Handle&, Jobz, int, int);
    template int gesvdj_buffersize<double,double>(Handle&, Jobz, int, int);
    template int gesvdj_buffersize<std::complex<float>,float>(Handle&, Jobz, int, int);
    template int gesvdj_buffersize<std::complex<double>,double>(Handle&, Jobz, int, int);

    void gesvdj(cusolverDnHandle_t& handle, cusolverEigMode_t jobz, int econ,
                int m, int n, float* A, int lda, float* S, float* U, int ldu,
                float* V, int ldv, float* Workspace, int lwork, int *info,
                gesvdjInfo_t& params) {
      STRUMPACK_FLOPS(blas::gesvd_flops(m,n));
      gpu_check(cusolverDnSgesvdj
                (handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                 Workspace, lwork, info, params));
    }

    void gesvdj(cusolverDnHandle_t& handle, cusolverEigMode_t jobz, int econ,
                int m, int n, double* A, int lda, double* S, double* U,
                int ldu, double* V, int ldv, double* Workspace, int lwork,
                int *info, gesvdjInfo_t& params) {
      STRUMPACK_FLOPS(blas::gesvd_flops(m,n));
      gpu_check(cusolverDnDgesvdj
                (handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                 Workspace, lwork, info, params));
    }
    void gesvdj(cusolverDnHandle_t& handle, cusolverEigMode_t jobz, int econ, int m,
                int n, std::complex<float>* A, int lda, float* S,
                std::complex<float>* U, int ldu, std::complex<float>* V,
                int ldv, std::complex<float>* Workspace, int lwork,
                int *info, gesvdjInfo_t& params) {
      STRUMPACK_FLOPS(4*blas::gesvd_flops(m,n));
      gpu_check(cusolverDnCgesvdj
                (handle, jobz, econ, m, n, reinterpret_cast<cuComplex*>(A),
                 lda, S, reinterpret_cast<cuComplex*>(U), ldu,
                 reinterpret_cast<cuComplex*>(V), ldv,
                 reinterpret_cast<cuComplex*>(Workspace),
                 lwork, info, params));
    }
    void gesvdj(cusolverDnHandle_t& handle, cusolverEigMode_t jobz, int econ, int m,
                int n, std::complex<double>* A, int lda, double* S,
                std::complex<double>* U, int ldu, std::complex<double>* V,
                int ldv, std::complex<double>* Workspace, int lwork,
                int *info, gesvdjInfo_t& params) {
      STRUMPACK_FLOPS(4*blas::gesvd_flops(m,n));
      gpu_check(cusolverDnZgesvdj
                (handle, jobz, econ, m, n,
                 reinterpret_cast<cuDoubleComplex*>(A), lda, S,
                 reinterpret_cast<cuDoubleComplex*>(U), ldu,
                 reinterpret_cast<cuDoubleComplex*>(V), ldv,
                 reinterpret_cast<cuDoubleComplex*>(Workspace), lwork, info,
                 params));
    }
    template<typename scalar_t, typename real_t> void
    gesvdj(Handle& handle, Jobz jobz, DenseMatrix<scalar_t>& A, real_t* d_S,
           DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V,
           scalar_t* Workspace, int Lwork, int* dinfo, gesvdjInfo_t& params) {
      gesvdj(get_cusolver_handle(handle), E2cuOp(jobz), 1, A.rows(), A.cols(),
             A.data(), A.ld(), d_S, U.data(), A.ld(), V.data(), A.cols(),
             Workspace, Lwork, dinfo, params);
    }
    template void gesvdj(Handle&, Jobz, DenseMatrix<float>&, float*,
                         DenseMatrix<float>&, DenseMatrix<float>&,
                         float*, int, int*, gesvdjInfo_t&);
    template void gesvdj(Handle&, Jobz, DenseMatrix<double>&, double*,
                         DenseMatrix<double>&, DenseMatrix<double>&,
                         double*, int, int*, gesvdjInfo_t&);
    template void gesvdj(Handle&, Jobz, DenseMatrix<std::complex<float>>&,
                         float*, DenseMatrix<std::complex<float>>&,
                         DenseMatrix<std::complex<float>>&,
                         std::complex<float>*, int, int*, gesvdjInfo_t&);
    template void gesvdj(Handle&, Jobz, DenseMatrix<std::complex<double>>&,
                         double*, DenseMatrix<std::complex<double>>&,
                         DenseMatrix<std::complex<double>>&,
                         std::complex<double>*, int, int*, gesvdjInfo_t&);

    template<typename scalar_t, typename real_t> void
    gesvdj(Handle& handle, Jobz jobz, DenseMatrix<scalar_t>& A, real_t* S,
           DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V, int* dinfo,
           scalar_t* work, int lwork, const double tol) {
      gesvdjInfo_t params;
      gesvdj_info_create(&params);
      cusolverDnXgesvdjSetTolerance(params, tol);
      gesvdj<scalar_t>
        (handle, jobz, A, S, U, V, work, lwork, dinfo, params);
      gesvdj_info_destroy(params);
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

    void geam(cublasHandle_t& handle, cublasOperation_t transa,
              cublasOperation_t transb, int m, int n, const float* alpha,
              const float* A, int lda, const float* beta,
              const float* B, int ldb, float* C, int ldc) {
      STRUMPACK_FLOPS(blas::geam_flops(m, n, alpha, beta));
      gpu_check(cublasSgeam(handle, transa, transb, m, n, alpha,
                            A, lda, beta, B, ldb, C, ldc));
    }
    void geam(cublasHandle_t& handle, cublasOperation_t transa,
              cublasOperation_t transb, int m, int n, const double* alpha,
              const double* A, int lda, const double* beta,
              const double* B, int ldb, double* C, int ldc){
      STRUMPACK_FLOPS(blas::geam_flops(m, n, alpha, beta));
      gpu_check(cublasDgeam(handle, transa, transb, m, n, alpha,
                            A, lda, beta, B, ldb, C, ldc));
    }
    void geam(cublasHandle_t& handle, cublasOperation_t transa,
              cublasOperation_t transb, int m, int n,
              const std::complex<float>* alpha,
              const std::complex<float>* A, int lda,
              const std::complex<float>* beta,
              const std::complex<float>* B, int ldb,
              std::complex<float>* C, int ldc) {
      STRUMPACK_FLOPS(4*blas::geam_flops(m, n, alpha, beta));
      gpu_check(cublasCgeam(handle, transa, transb, m, n,
                            reinterpret_cast<const cuComplex*>(alpha),
                            reinterpret_cast<const cuComplex*>(A), lda,
                            reinterpret_cast<const cuComplex*>(beta),
                            reinterpret_cast<const cuComplex*>(B), ldb,
                            reinterpret_cast<cuComplex*>(C), ldc));
    }
    void geam(cublasHandle_t& handle, cublasOperation_t transa,
              cublasOperation_t transb, int m, int n,
              const std::complex<double>* alpha,
              const std::complex<double>* A, int lda,
              const std::complex<double>* beta,
              const std::complex<double>* B, int ldb,
              std::complex<double>* C, int ldc) {
      STRUMPACK_FLOPS(4*blas::geam_flops(m, n, alpha, beta));
      gpu_check(cublasZgeam(handle, transa, transb, m, n,
                            reinterpret_cast<const cuDoubleComplex*>(alpha),
                            reinterpret_cast<const cuDoubleComplex*>(A), lda,
                            reinterpret_cast<const cuDoubleComplex*>(beta),
                            reinterpret_cast<const cuDoubleComplex*>(B), ldb,
                            reinterpret_cast<cuDoubleComplex*>(C), ldc));
    }
    template<typename scalar_t>
    void geam(Handle& handle, Trans transa, Trans transb,
              const scalar_t alpha, const DenseMatrix<scalar_t>& A,
              const scalar_t beta, const DenseMatrix<scalar_t>& B,
              DenseMatrix<scalar_t>& C) {
      geam(get_cublas_handle(handle),
           T2cuOp(transa), T2cuOp(transb), C.rows(), C.cols(), &alpha,
           A.data(), A.ld(), &beta, B.data(), B.ld(), C.data(), C.ld());
    }
    template void geam(Handle&, Trans, Trans, const float,
                       const DenseMatrix<float>&, const float,
                       const DenseMatrix<float>&, DenseMatrix<float>&);
    template void geam(Handle&, Trans, Trans, const double,
                       const DenseMatrix<double>&, const double,
                       const DenseMatrix<double>&, DenseMatrix<double>&);
    template void geam(Handle&, Trans, Trans, const std::complex<float>,
                       const DenseMatrix<std::complex<float>>&,
                       const std::complex<float>,
                       const DenseMatrix<std::complex<float>>&,
                       DenseMatrix<std::complex<float>>&);
    template void geam(Handle&, Trans, Trans, const std::complex<double>,
                       const DenseMatrix<std::complex<double>>&,
                       const std::complex<double>,
                       const DenseMatrix<std::complex<double>>&,
                       DenseMatrix<std::complex<double>>&);

    void dgmm(cublasHandle_t& handle, cublasSideMode_t side, int m, int n,
              const float* A, int lda, const float* x, int incx,
              float* C, int ldc) {
      STRUMPACK_FLOPS(blas::dgmm_flops(m, n));
      gpu_check(cublasSdgmm(handle, side, m, n, A, lda, x, incx, C, ldc));
    }
    void dgmm(cublasHandle_t& handle, cublasSideMode_t side, int m, int n,
              const double* A, int lda, const double* x, int incx,
              double* C, int ldc) {
      STRUMPACK_FLOPS(blas::dgmm_flops(m, n));
      gpu_check(cublasDdgmm(handle, side, m, n, A, lda, x, incx, C, ldc));
    }
    void dgmm(cublasHandle_t& handle, cublasSideMode_t side, int m, int n,
              const std::complex<float>* A, int lda,
              const std::complex<float>* x, int incx,
              std::complex<float>* C, int ldc) {
      STRUMPACK_FLOPS(4*blas::dgmm_flops(m, n));
      gpu_check(cublasCdgmm(handle, side, m, n,
                            reinterpret_cast<const cuComplex*>(A), lda,
                            reinterpret_cast<const cuComplex*>(x), incx,
                            reinterpret_cast<cuComplex*>(C), ldc));
    }
    void dgmm(cublasHandle_t& handle, cublasSideMode_t side, int m, int n,
              const std::complex<double>* A, int lda,
              const std::complex<double>* x, int incx,
              std::complex<double>* C, int ldc){
      STRUMPACK_FLOPS(4*blas::dgmm_flops(m, n));
      gpu_check(cublasZdgmm(handle, side, m, n,
                            reinterpret_cast<const cuDoubleComplex*>(A), lda,
                            reinterpret_cast<const cuDoubleComplex*>(x), incx,
                            reinterpret_cast<cuDoubleComplex*>(C), ldc));
    }
    template<typename scalar_t> void
    dgmm(Handle& handle, Side side, const DenseMatrix<scalar_t>& A,
         const scalar_t* x, DenseMatrix<scalar_t>& C){
      dgmm(get_cublas_handle(handle), S2cuOp(side), A.rows(), A.cols(),
           A.data(), A.ld(), x, 1, C.data(), C.ld());
    }
    template void dgmm(Handle&, Side, const DenseMatrix<float>&,
                       const float*, DenseMatrix<float>&);
    template void dgmm(Handle&, Side, const DenseMatrix<double>&,
                       const double*, DenseMatrix<double>&);
    template void dgmm(Handle&, Side, const DenseMatrix<std::complex<float>>&,
                       const std::complex<float>*, DenseMatrix<std::complex<float>>&);
    template void dgmm(Handle&, Side, const DenseMatrix<std::complex<double>>&,
                       const std::complex<double>*, DenseMatrix<std::complex<double>>&);

    void gemv(cublasHandle_t& handle, cublasOperation_t transa,
              int m, int n, float alpha,
              const float* A, int lda, const float* B, int incb,
              float beta, float* C, int incc) {
      STRUMPACK_FLOPS(blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(4*blas::gemv_moves(m,n));
      gpu_check(cublasSgemv
                (handle, transa, m, n, &alpha,
                 A, lda, B, incb, &beta, C, incc));
    }
    void gemv(cublasHandle_t& handle, cublasOperation_t transa,
              int m, int n, double alpha,
              const double* A, int lda, const double* B, int incb,
              double beta, double* C, int incc) {
      STRUMPACK_FLOPS(blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(8*blas::gemv_moves(m,n));
      gpu_check(cublasDgemv
                (handle, transa, m, n, &alpha,
                 A, lda, B, incb, &beta, C, incc));
    }
    void gemv(cublasHandle_t& handle, cublasOperation_t transa,
              int m, int n, std::complex<float> alpha,
              const std::complex<float>* A, int lda,
              const std::complex<float>* B, int incb,
              std::complex<float> beta,
              std::complex<float> *C, int incc) {
      STRUMPACK_FLOPS(4*blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(2*4*blas::gemv_moves(m,n));
      gpu_check(cublasCgemv
                (handle, transa, m, n,
                 reinterpret_cast<cuComplex*>(&alpha),
                 reinterpret_cast<const cuComplex*>(A), lda,
                 reinterpret_cast<const cuComplex*>(B), incb,
                 reinterpret_cast<cuComplex*>(&beta),
                 reinterpret_cast<cuComplex*>(C), incc));
    }
    void gemv(cublasHandle_t& handle, cublasOperation_t transa,
              int m, int n, std::complex<double> alpha,
              const std::complex<double> *A, int lda,
              const std::complex<double> *B, int incb,
              std::complex<double> beta,
              std::complex<double> *C, int incc) {
      STRUMPACK_FLOPS(4*blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(2*8*blas::gemv_moves(m,n));
      gpu_check(cublasZgemv
                (handle, transa, m, n,
                 reinterpret_cast<cuDoubleComplex*>(&alpha),
                 reinterpret_cast<const cuDoubleComplex*>(A), lda,
                 reinterpret_cast<const cuDoubleComplex*>(B), incb,
                 reinterpret_cast<cuDoubleComplex*>(&beta),
                 reinterpret_cast<cuDoubleComplex*>(C), incc));
    }
    template<typename scalar_t> void
    gemv(Handle& handle, Trans ta, scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const DenseMatrix<scalar_t>& x, scalar_t beta, DenseMatrix<scalar_t>& y) {
      gemv(get_cublas_handle(handle), T2cuOp(ta), a.rows(), a.cols(), alpha,
           a.data(), a.ld(), x.data(), 1, beta, y.data(), 1);
    }
    template void
    gemv(Handle&, Trans, float, const DenseMatrix<float>&,
         const DenseMatrix<float>&, float, DenseMatrix<float>&);
    template void
    gemv(Handle&, Trans, double, const DenseMatrix<double>&,
         const DenseMatrix<double>&, double, DenseMatrix<double>&);
    template void
    gemv(Handle&, Trans, std::complex<float>, const DenseMatrix<std::complex<float>>&,
         const DenseMatrix<std::complex<float>>&, std::complex<float>,
         DenseMatrix<std::complex<float>>&);
    template void
    gemv(Handle&, Trans, std::complex<double>, const DenseMatrix<std::complex<double>>&,
         const DenseMatrix<std::complex<double>>&, std::complex<double>,
         DenseMatrix<std::complex<double>>&);

  } // end namespace gpu
} // end namespace strumpack
