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

#include "SYCLWrapper.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "misc/MPIWrapper.hpp"
#endif

namespace strumpack {
  namespace gpu {

    void peek_at_last_error() {
      std::cout << "TODO SYCL peek_at_last_error" << std::endl;
      // gpu_check(cudaPeekAtLastError());
    }

    void get_last_error() {
      std::cout << "TODO SYCL get_last_error" << std::endl;
      // cudaGetLastError();
    }

    void synchronize() {
      get_sycl_queue().wait();
    }

    struct Stream::StreamImpl {
      StreamImpl() {
        q_ = sycl::queue
          (get_sycl_queue().get_context(),
           get_sycl_queue().get_device(), async_handler,
           sycl::property_list{sycl::property::queue::in_order{}} );
      }
      // ~StreamImpl() = default
      operator sycl::queue&() { return q_; }
      operator const sycl::queue&() const { return q_; }
      void synchronize() { q_.wait(); }
      sycl::queue q_;
    };
    Stream::Stream() { s_ = std::make_unique<StreamImpl>(); }
    Stream::~Stream() = default;
    void Stream::synchronize() { s_->synchronize(); }

    const sycl::queue& get_sycl_queue(const Stream& s) { return *(s.s_); }
    sycl::queue& get_sycl_queue(Stream& s) { return *(s.s_); }


    struct Handle::HandleImpl {
      // TODO magma, kblas handles are stored here? see CUDAWrapper.cpp
      HandleImpl() { q_ = &get_sycl_queue(); }
      HandleImpl(Stream& s) { set_stream(s); }
      // ~HandleImpl() = default;
      void set_stream(Stream& s) { q_ = &get_sycl_queue(s); }

      operator sycl::queue&() { return *q_; }
      operator const sycl::queue&() const { return *q_; }
      sycl::queue* q_;
    };
    Handle::Handle() { h_ = std::make_unique<HandleImpl>(); }
    Handle::Handle(Stream& s) { h_ = std::make_unique<HandleImpl>(s); }
    Handle::~Handle() = default;
    void Handle::set_stream(Stream& s) { h_->set_stream(s); }

    const sycl::queue& get_sycl_queue(const Handle& h) { return *(h.h_); }
    sycl::queue& get_sycl_queue(Handle& h) { return *(h.h_); }


    struct Event::EventImpl {
      // EventImpl() = default;
      // ~EventImpl() = default;
      void record() {
        std::cout << "TODO EventImpl::record" << std::endl;
      }
      void record(Stream& s) {
        e_ = get_sycl_queue(s).ext_oneapi_submit_barrier();
      }
      void wait(Stream& s) {
        std::cout << "TODO EventImpl::wait(Stream)" << std::endl;
      }
      void synchronize() {
        std::cout << "TODO EventImpl::synchronize" << std::endl;
      }

      sycl::event e_;
    };
    Event::Event() { e_ = std::make_unique<EventImpl>(); }
    Event::~Event() = default;
    void Event::record() { e_->record(); }
    void Event::record(Stream& s) { e_->record(s); }
    void Event::wait(Stream& s) { e_->wait(s); }
    void Event::synchronize() { e_->synchronize(); }


    void device_malloc(void** ptr, std::size_t size) {
      *ptr = (void*)sycl::malloc_device(size, get_sycl_queue());
    }
    void host_malloc(void** ptr, std::size_t size) {
      *ptr = (void*)sycl::malloc_host(size, get_sycl_queue());
    }
    void device_free(void* ptr) {
      sycl::free(ptr, get_sycl_queue().get_context());
    }
    void host_free(void* ptr) {
      sycl::free(ptr, get_sycl_queue().get_context());
    }

    // cudaMemcpyKind CD2cuMK(CopyDir d) {
    //   switch (d) {
    //   case CopyDir::H2H: return cudaMemcpyHostToHost;
    //   case CopyDir::H2D: return cudaMemcpyHostToDevice;
    //   case CopyDir::D2H: return cudaMemcpyDeviceToHost;
    //   case CopyDir::D2D: return cudaMemcpyDeviceToDevice;
    //   case CopyDir::DEF: return cudaMemcpyDefault;
    //   default: assert(false); return cudaMemcpyDefault;
    //   }
    // }

    oneapi::mkl::transpose T2MKLOp(Trans op) {
      switch (op) {
      case Trans::N: return oneapi::mkl::transpose::N;
      case Trans::T: return oneapi::mkl::transpose::T;
      case Trans::C: return oneapi::mkl::transpose::C;
      default:
        assert(false);
        return oneapi::mkl::transpose::N;
      }
    }

    // cublasSideMode_t S2cuOp(Side op) {
    //   switch (op) {
    //   case Side::L: return CUBLAS_SIDE_LEFT;
    //   case Side::R: return CUBLAS_SIDE_RIGHT;
    //   default: assert(false); return CUBLAS_SIDE_LEFT;
    //   }
    // }

    // cublasFillMode_t U2cuOp(UpLo op) {
    //   switch (op) {
    //   case UpLo::L: return CUBLAS_FILL_MODE_LOWER;
    //   case UpLo::U: return CUBLAS_FILL_MODE_UPPER;
    //   default: assert(false); return CUBLAS_FILL_MODE_LOWER;
    //   }
    // }

    // cublasDiagType_t D2cuOp(Diag op) {
    //   switch (op) {
    //   case Diag::N: return CUBLAS_DIAG_NON_UNIT;
    //   case Diag::U: return CUBLAS_DIAG_UNIT;
    //   default: assert(false); return CUBLAS_DIAG_UNIT;
    //   }
    // }

    // cusolverEigMode_t E2cuOp(Jobz op) {
    //   switch (op) {
    //   case Jobz::N: return CUSOLVER_EIG_MODE_NOVECTOR;
    //   case Jobz::V: return CUSOLVER_EIG_MODE_VECTOR;
    //   default: assert(false); return CUSOLVER_EIG_MODE_VECTOR;
    //   }
    // }


    void init() {
#if defined(STRUMPACK_USE_MPI)
      int devs = DeviceManager::instance().device_count();
      if (devs > 1) {
        int flag, rank = 0;
        MPI_Initialized(&flag);
        if (flag) {
          MPIComm c;
          rank = c.rank();
        }
        DeviceManager::instance().select_device(rank % devs);
      }
#endif
    }

    void device_memset(void* dptr, int value, std::size_t count) {
      get_sycl_queue().memset(dptr, value, count).wait();
    }

    void device_copy(void* dest, const void* src,
                     std::size_t count, CopyDir) {
      // gpu_check(cudaMemcpy(dest, src, count, CD2cuMK(dir)));
      get_sycl_queue().memcpy(dest, src, count).wait();
    }
    void device_copy_async(void* dest, const void* src, std::size_t count,
                           CopyDir dir, Stream& s) {
      // gpu_check(cudaMemcpyAsync(dest, src, count, CD2cuMK(dir),
      //                           get_cuda_stream(s)));
      get_sycl_queue(s).memcpy(dest, src, count);
    }
    void device_copy_2D(void* dest, std::size_t dpitch,
                        const void* src, std::size_t spitch,
                        std::size_t width, std::size_t height, CopyDir dir) {
      // gpu_check(cudaMemcpy2D(dest, dpitch, src, spitch,
      //                        width , height, CD2cuMK(dir)));
      std::cout << "TODO SYCL device_copy_2D" << std::endl;
    }
    void device_copy_2D_async(void* dest, std::size_t dpitch,
                              const void* src, std::size_t spitch,
                              std::size_t width, std::size_t height,
                              CopyDir dir, Stream& s) {
      // gpu_check(cudaMemcpy2DAsync(dest, dpitch, src, spitch, width, height,
      //                             CD2cuMK(dir), get_cuda_stream(s)));
      std::cout << "TODO SYCL device_copy_2D_async" << std::endl;
    }

    std::size_t available_memory() {
      return get_sycl_queue().get_device().
        get_info<sycl::ext::intel::info::device::free_memory>();
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
      oneapi::mkl::blas::gemm
        (get_sycl_queue(handle), T2MKLOp(ta), T2MKLOp(tb), c.rows(), c.cols(),
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


    // TODO make return value std::int64_t in GPUWrapper
    template<typename scalar_t> std::int64_t
    getrf_buffersize(Handle& h, int n) {
      return oneapi::mkl::lapack::getrf_scratchpad_size<scalar_t>
        (get_sycl_queue(h), n, n, n);
    }
    template std::int64_t getrf_buffersize<float>(Handle&, int);
    template std::int64_t getrf_buffersize<double>(Handle&, int);
    template std::int64_t getrf_buffersize<std::complex<float>>(Handle&, int);
    template std::int64_t getrf_buffersize<std::complex<double>>(Handle&, int);

    // TODO add this in GPUWrapper, use when needed
    // template<typename scalar_t> std::int64_t
    // getrs_buffersize(Handle& h, Trans t, int n, int nrhs, int lda, int ldb) {
    //   return oneapi::mkl::lapack::getrs_scratchpad_size<scalar_t>
    //     (get_sycl_queue(h), T2MKLOp(t), n, nrhs, lda, ldb);
    // }


    template<typename scalar_t> void
    getrf(Handle& h, DenseMatrix<scalar_t>& A,
          scalar_t* workspace, int* devIpiv, int* devInfo) {
      STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::getrf_flops(A.rows(),A.cols()));
      try {
        // TODO scratchpad_size??
        std::int64_t scratchpad_size = 0;
        std::cout << "TODO SYCL getrf scratchpad_size" << std::endl;
        std::cout << "TODO SYCL getrf piv 64!" << std::endl;
        std::int64_t* dpiv = nullptr;
        oneapi::mkl::lapack::getrf
          (get_sycl_queue(h), A.rows(), A.cols(), A.data(), A.ld(),
           // devIpiv,
           dpiv, workspace, scratchpad_size);
      } catch (oneapi::mkl::lapack::exception e) {
        std::cerr << "Exception in oneapi::mkl::lapack::getrf, info = "
                  << e.info() << std::endl;
      }
    }
    template void getrf(Handle&, DenseMatrix<float>&,float*, int*, int*);
    template void getrf(Handle&, DenseMatrix<double>&, double*, int*, int*);
    template void getrf(Handle&, DenseMatrix<std::complex<float>>&, std::complex<float>*, int*, int*);
    template void getrf(Handle&, DenseMatrix<std::complex<double>>& A, std::complex<double>*, int*, int*);

    template<typename scalar_t> void
    getrs(Handle& handle, Trans trans, const DenseMatrix<scalar_t>& A,
          const int* ipiv, DenseMatrix<scalar_t>& B, int *devinfo) {
      STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::getrs_flops(A.rows(),B.cols()));
      try {
        std::cout << "TODO SYCL getrs scratchpad_size" << std::endl;
        std::cout << "TODO SYCL getrs piv 64!" << std::endl;
        std::int64_t* dpiv = nullptr;
        scalar_t* scratchpad = nullptr;
        std::int64_t scratchpad_size = 0;
        oneapi::mkl::lapack::getrs
          (get_sycl_queue(handle), T2MKLOp(trans), A.rows(), B.cols(),
           const_cast<scalar_t*>(A.data()), A.ld(),
           // const_cast<std::int64_t*>(ipiv),
           dpiv, B.data(), B.ld(), scratchpad, scratchpad_size);
      } catch (oneapi::mkl::lapack::exception e) {
        std::cerr << "Exception in oneapi::mkl::lapack::getrs, info = "
                  << e.info() << std::endl;
      }
    }
    template void getrs(Handle&, Trans, const DenseMatrix<float>&,
                        const int*, DenseMatrix<float>&, int*);
    template void getrs(Handle&, Trans, const DenseMatrix<double>&,
                        const int*, DenseMatrix<double>&, int*);
    template void getrs(Handle&, Trans, const DenseMatrix<std::complex<float>>&, const int*,
                        DenseMatrix<std::complex<float>>&, int*);
    template void getrs(Handle&, Trans, const DenseMatrix<std::complex<double>>&, const int*,
                        DenseMatrix<std::complex<double>>&, int*);

    template<typename scalar_t> void
    trsm(Handle& handle, Side side, UpLo uplo,
         Trans trans, Diag diag, const scalar_t alpha,
         DenseMatrix<scalar_t>& A, DenseMatrix<scalar_t>& B) {
      std::cout << "TODO SYCL trsm" << std::endl;
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

    template<typename scalar_t, typename real_t>
    int gesvdj_buffersize(Handle& handle, Jobz jobz, int m, int n) {
      std::cout << "TODO SYCL gesvdj_buffersize" << std::endl;
      return 0;
    }
    template int gesvdj_buffersize<float,float>(Handle&, Jobz, int, int);
    template int gesvdj_buffersize<double,double>(Handle&, Jobz, int, int);
    template int gesvdj_buffersize<std::complex<float>,float>(Handle&, Jobz, int, int);
    template int gesvdj_buffersize<std::complex<double>,double>(Handle&, Jobz, int, int);


    template<typename scalar_t, typename real_t> void
    gesvdj(Handle& handle, Jobz jobz, DenseMatrix<scalar_t>& A, real_t* S,
           DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V, int* devInfo,
           scalar_t* work, int lwork, const double tol) {
      std::cout << "TODO SYCL gesvdj" << std::endl;
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

    template<typename scalar_t>
    void geam(Handle& handle, Trans transa, Trans transb,
              const scalar_t alpha, const DenseMatrix<scalar_t>& A,
              const scalar_t beta, const DenseMatrix<scalar_t>& B,
              DenseMatrix<scalar_t>& C) {
      std::cout << "TODO SYCL geam" << std::endl;
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

    template<typename scalar_t> void
    dgmm(Handle& handle, Side side, const DenseMatrix<scalar_t>& A,
         const scalar_t* x, DenseMatrix<scalar_t>& C){
      std::cout << "TODO SYCL dgmm" << std::endl;
    }
    template void dgmm(Handle&, Side, const DenseMatrix<float>&,
                       const float*, DenseMatrix<float>&);
    template void dgmm(Handle&, Side, const DenseMatrix<double>&,
                       const double*, DenseMatrix<double>&);
    template void dgmm(Handle&, Side, const DenseMatrix<std::complex<float>>&,
                       const std::complex<float>*, DenseMatrix<std::complex<float>>&);
    template void dgmm(Handle&, Side, const DenseMatrix<std::complex<double>>&,
                       const std::complex<double>*, DenseMatrix<std::complex<double>>&);

    template<typename scalar_t> void
    gemv(Handle& handle, Trans ta, scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const DenseMatrix<scalar_t>& x, scalar_t beta, DenseMatrix<scalar_t>& y) {
      STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::gemv_flops(a.rows(),a.cols(),alpha,beta));
      STRUMPACK_BYTES((is_complex<scalar_t>()?2:1)*sizeof(scalar_t)*blas::gemv_moves(a.rows(),a.cols()));
      oneapi::mkl::blas::gemv
        (get_sycl_queue(handle), T2MKLOp(ta), a.rows(), a.cols(), alpha,
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
