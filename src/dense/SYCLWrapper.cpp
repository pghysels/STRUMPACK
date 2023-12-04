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

// TODO is this portable?
#include <sys/syscall.h>
#include <unistd.h>

#include "SYCLWrapper.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "misc/MPIWrapper.hpp"
#endif

namespace strumpack {
  namespace gpu {

    auto async_handler = [](sycl::exception_list exceptions) {
      for (std::exception_ptr const &e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch (sycl::exception const &e) {
          std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                    << e.what() << std::endl
                    << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
        }
      }
    };

    // TODO is this portable?
    int get_tid() { return syscall(SYS_gettid); }

    class DeviceManager {
    public:
      int current_device() {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = _thread2dev_map.find(get_tid());
        if (it != _thread2dev_map.end()) {
          check_id(it->second);
          return it->second;
        }
        std::cerr
          << "WARNING: no SYCL device found in the map, returning DEFAULT_DEVICE_ID"
          << std::endl;
        return DEFAULT_DEVICE_ID;
      }
      sycl::queue* current_queue() {
        return _queues[current_device()];
      }

      void select_device(int id) {
        std::lock_guard<std::mutex> lock(m_mutex);
        check_id(id);
        _thread2dev_map[get_tid()] = id;
      }
      int device_count() { return _queues.size(); }

      /// Returns the instance of device manager singleton.
      static DeviceManager& instance() {
        static DeviceManager d_m;
        return d_m;
      }
      DeviceManager(const DeviceManager&)            = delete;
      DeviceManager& operator=(const DeviceManager&) = delete;
      DeviceManager(DeviceManager&&)                 = delete;
      DeviceManager& operator=(DeviceManager&&)      = delete;

    private:
      mutable std::mutex m_mutex;

      DeviceManager() {
        sycl::device dev(sycl::gpu_selector_v);
        _queues.push_back
          (new sycl::queue
           (dev, async_handler,
            sycl::property_list{sycl::property::queue::in_order{}}));
      }

      void check_id(int id) const {
        if (id >= _queues.size())
          throw std::runtime_error("invalid device id");
      }

      // Note: only 1 out-of-order SYCL queue is created per device
      std::vector<sycl::queue*> _queues;

      /// DEFAULT_DEVICE_ID is used, if current_device() can not find
      /// current thread id in _thread2dev_map, which means default
      /// device should be used for the current thread.
      const int DEFAULT_DEVICE_ID = 0;
      /// thread-id to device-id map.
      std::map<int, int> _thread2dev_map;
    };

    int get_sycl_device() {
      return DeviceManager::instance().current_device();
    }

    sycl::queue& get_sycl_queue() {
      return *(DeviceManager::instance().current_queue());
    }

    void peek_at_last_error() {}
    void get_last_error() {}

    void synchronize_default_stream() {
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
      void record() { e_ = get_sycl_queue().ext_oneapi_submit_barrier(); }
      void record(Stream& s) { e_ = get_sycl_queue(s).ext_oneapi_submit_barrier(); }
      void wait(Stream& s) { get_sycl_queue(s).ext_oneapi_submit_barrier({e_}).wait(); }
      void synchronize() { e_.wait(); }

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

    oneapi::mkl::side S2MKLOp(Side op) {
      switch (op) {
      case Side::L: return oneapi::mkl::side::L;
      case Side::R: return oneapi::mkl::side::R;
      default: assert(false); return oneapi::mkl::side::L;
      }
    }

    oneapi::mkl::uplo U2MKLOp(UpLo op) {
      switch (op) {
      case UpLo::L: return oneapi::mkl::uplo::L;
      case UpLo::U: return oneapi::mkl::uplo::U;
      default: assert(false); return oneapi::mkl::uplo::L;
      }
    }

    oneapi::mkl::diag D2MKLOp(Diag op) {
      switch (op) {
      case Diag::N: return oneapi::mkl::diag::N;
      case Diag::U: return oneapi::mkl::diag::U;
      default: assert(false); return oneapi::mkl::diag::N;
      }
    }


    void init() {
      int rank = 0, devs = DeviceManager::instance().device_count();
#if defined(STRUMPACK_USE_MPI)
      int flag = 0;
      MPI_Initialized(&flag);
      if (flag) {
        MPIComm c;
        rank = c.rank();
      }
#endif
      DeviceManager::instance().select_device(rank % devs);
    }

    void device_memset(void* dptr, int value, std::size_t count) {
      get_sycl_queue().memset(dptr, value, count).wait();
    }

    void device_copy(void* dest, const void* src,
                     std::size_t count, CopyDir) {
      get_sycl_queue().memcpy(dest, src, count).wait();
    }
    void device_copy_async(void* dest, const void* src, std::size_t count,
                           CopyDir dir, Stream& s) {
      get_sycl_queue(s).memcpy(dest, src, count);
    }
    void device_copy_2D(void* dest, std::size_t dpitch,
                        const void* src, std::size_t spitch,
                        std::size_t width, std::size_t height, CopyDir dir) {
      get_sycl_queue().ext_oneapi_memcpy2d
        (dest, dpitch, src, spitch, width, height).wait();
    }
    void device_copy_2D_async(void* dest, std::size_t dpitch,
                              const void* src, std::size_t spitch,
                              std::size_t width, std::size_t height,
                              CopyDir dir, Stream& s) {
      get_sycl_queue().ext_oneapi_memcpy2d
        (dest, dpitch, src, spitch, width, height);
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
      oneapi::mkl::blas::column_major::gemm
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

    template<typename scalar_t> std::size_t scalars_for_int64(std::size_t n) {
      // == ceil(n * sizeof(std::int64_t) / sizeof(scalar_t))
      return (n * sizeof(std::int64_t) + sizeof(scalar_t) - 1) / sizeof(scalar_t);
    }

    template<typename scalar_t> std::int64_t
    getrf_buffersize(Handle& h, int n) {
      return oneapi::mkl::lapack::getrf_scratchpad_size<scalar_t>
        (get_sycl_queue(h), n, n, n)
        // add some extra space to convert pivot data int/int64_t
        + scalars_for_int64<scalar_t>(n);
    }
    template std::int64_t getrf_buffersize<float>(Handle&, int);
    template std::int64_t getrf_buffersize<double>(Handle&, int);
    template std::int64_t getrf_buffersize<std::complex<float>>(Handle&, int);
    template std::int64_t getrf_buffersize<std::complex<double>>(Handle&, int);


    template<typename scalar_t> void
    getrf(Handle& h, DenseMatrix<scalar_t>& A,
          scalar_t* work, std::int64_t lwork, int* dpiv, int* dinfo) {
      STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::getrf_flops(A.rows(),A.cols()));
      auto np = scalars_for_int64<scalar_t>(A.rows());
      auto dpiv64 = reinterpret_cast<std::int64_t*>(work);
      try {
        oneapi::mkl::lapack::getrf
          (get_sycl_queue(h), A.rows(), A.cols(), A.data(), A.ld(),
           dpiv64, work+np, lwork-np);
        memset<int>(dinfo, 0, 1);
      } catch (oneapi::mkl::lapack::exception e) {
        std::cerr << "Exception in oneapi::mkl::lapack::getrf, info = "
                  << e.info() << std::endl;
      }
      get_sycl_queue(h).parallel_for
        (sycl::range<1>(A.rows()), [=](sycl::id<1> i) {
          dpiv[i[0]] = dpiv64[i[0]];
        });
    }
    template void getrf(Handle&, DenseMatrix<float>&, float*, std::int64_t, int*, int*);
    template void getrf(Handle&, DenseMatrix<double>&, double*, std::int64_t, int*, int*);
    template void getrf(Handle&, DenseMatrix<std::complex<float>>&, std::complex<float>*, std::int64_t, int*, int*);
    template void getrf(Handle&, DenseMatrix<std::complex<double>>& A, std::complex<double>*, std::int64_t, int*, int*);

    template<typename scalar_t> std::int64_t
    getrs_buffersize(Handle& h, Trans t, int n, int nrhs, int lda, int ldb) {
      return oneapi::mkl::lapack::getrs_scratchpad_size<scalar_t>
        (get_sycl_queue(h), T2MKLOp(t), n, nrhs, lda, ldb)
        // add some extra space to convert pivot data int/int64_t
        + scalars_for_int64<scalar_t>(n);
    }
    template std::int64_t getrs_buffersize<float>(Handle&, Trans, int, int, int, int);
    template std::int64_t getrs_buffersize<double>(Handle&, Trans, int, int, int, int);
    template std::int64_t getrs_buffersize<std::complex<float>>(Handle&, Trans, int, int, int, int);
    template std::int64_t getrs_buffersize<std::complex<double>>(Handle&, Trans, int, int, int, int);

    template<typename scalar_t> void
    getrs(Handle& handle, Trans trans, const DenseMatrix<scalar_t>& A,
          const int* ipiv, DenseMatrix<scalar_t>& B, int* dinfo,
          scalar_t* work, std::int64_t lwork) {
      STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::getrs_flops(A.rows(),B.cols()));
      DeviceMemory<scalar_t> dwork;
      scalar_t* scratchpad = nullptr;
      std::int64_t scratchpad_size = 0;
      std::int64_t* dpiv64 = nullptr;
      auto np = scalars_for_int64<scalar_t>(A.rows());
      if (work) {
        dpiv64 = reinterpret_cast<std::int64_t*>(work);
        scratchpad = work + np;
        scratchpad_size = lwork - np;
      } else {
        scratchpad_size = getrs_buffersize<scalar_t>
          (handle, trans, A.rows(), B.cols(), A.ld(), B.ld());
        dwork = DeviceMemory<scalar_t>(scratchpad_size);
        scratchpad = dwork;
        dpiv64 = reinterpret_cast<std::int64_t*>(scratchpad);
        scratchpad += np;
        scratchpad_size -= np;
      }
      get_sycl_queue(handle).parallel_for
        (sycl::range<1>(A.rows()), [=](sycl::id<1> i) {
          dpiv64[i[0]] = ipiv[i[0]];
        });
      try {
        oneapi::mkl::lapack::getrs
          (get_sycl_queue(handle), T2MKLOp(trans), A.rows(), B.cols(),
           const_cast<scalar_t*>(A.data()), A.ld(),
           dpiv64, B.data(), B.ld(), scratchpad, scratchpad_size);
        memset<int>(dinfo, 0, 1);
      } catch (oneapi::mkl::lapack::exception e) {
        std::cerr << "Exception in oneapi::mkl::lapack::getrs, info = "
                  << e.info() << std::endl;
      }
    }

    template void getrs(Handle&, Trans, const DenseMatrix<float>&,
                        const int*, DenseMatrix<float>&, int*, float*, std::int64_t);
    template void getrs(Handle&, Trans, const DenseMatrix<double>&,
                        const int*, DenseMatrix<double>&, int*, double*, std::int64_t);
    template void getrs(Handle&, Trans, const DenseMatrix<std::complex<float>>&, const int*,
                        DenseMatrix<std::complex<float>>&, int*, std::complex<float>*, std::int64_t);
    template void getrs(Handle&, Trans, const DenseMatrix<std::complex<double>>&, const int*,
                        DenseMatrix<std::complex<double>>&, int*, std::complex<double>*, std::int64_t);

    template<typename scalar_t> void
    trsm(Handle& handle, Side side, UpLo uplo,
         Trans trans, Diag diag, const scalar_t alpha,
         DenseMatrix<scalar_t>& A, DenseMatrix<scalar_t>& B) {
      STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::trsm_flops(A.rows(),B.cols(),alpha,char(side)));
      oneapi::mkl::blas::column_major::trsm
        (get_sycl_queue(handle), S2MKLOp(side), U2MKLOp(uplo), T2MKLOp(trans), D2MKLOp(diag),
         B.rows(), B.cols(), alpha, A.data(), A.ld(), B.data(), B.ld());
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


    /// code below was created from
    /// dpct --cuda-include-path=/soft/libraries/cuda-headers/12.0.0/targets/x86_64-linux/include/ CUDAWrapper.cu

    template<typename scalar_t> void
    laswp_kernel(int n, scalar_t* dA, int lddA,
                 int npivots, int* dipiv, int inci,
                 const sycl::nd_item<3> &item_ct1) {
      int tid = item_ct1.get_local_id(2) +
                item_ct1.get_local_range(2) * item_ct1.get_group(2);
      if (tid < n) {
        dA += tid * lddA;
        auto A1 = dA;
        for (int i1=0; i1<npivots; i1++) {
          int i2 = dipiv[i1*inci] - 1;
          auto A2 = dA + i2;
          auto temp = *A1;
          *A1 = *A2;
          *A2 = temp;
          A1++;
        }
      }
    }

    template<typename scalar_t> void
    laswp(Handle& handle, DenseMatrix<scalar_t>& dA,
          int k1, int k2, int* dipiv, int inci) {
      if (!dA.rows() || !dA.cols()) return;
      int n = dA.cols(), nt = 256;
      int grid = (n + nt - 1) / nt;
      /*
        DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
      */
      get_sycl_queue(handle).submit([&](sycl::handler &cgh) {
          auto dA_data_ct1 = dA.data();
          auto dA_ld_ct2 = dA.ld();
          cgh.parallel_for
            (sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                               sycl::range<3>(1, 1, nt),
                               sycl::range<3>(1, 1, nt)),
             [=](sycl::nd_item<3> item_ct1) {
              laswp_kernel<scalar_t>(n, dA_data_ct1, dA_ld_ct2,
                                     k2 - k1 + 1, dipiv + k1 - 1,
                                     inci, item_ct1);
            });
        });
    }

    template<typename T>  void
    laswp_vbatch_kernel(int* dn, T** dA, int* lddA, int** dipiv,
                        int* npivots, unsigned int batchCount,
                        const sycl::nd_item<3> &item_ct1) {
      // assume dn = cols, inc = 1
      int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2),
        f = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
        item_ct1.get_local_id(1);
      if (f >= batchCount) return;
      if (x >= dn[f]) return;
      auto A = dA[f];
      auto P = dipiv[f];
      auto ldA = lddA[f];
      auto npiv = npivots[f];
      A += x * ldA;
      auto A1 = A;
      for (int i=0; i<npiv; i++) {
        auto p = P[i] - 1;
        if (p != i) {
          auto A2 = A + p;
          auto temp = *A1;
          *A1 = *A2;
          *A2 = temp;
        }
        A1++;
      }
    }

    template<typename scalar_t> void
    laswp_fwd_vbatched(Handle& handle, int* dn, int max_n,
                       scalar_t** dA, int* lddA, int** dipiv, int* npivots,
                       unsigned int batchCount) {
      if (max_n <= 0 || !batchCount) return;
      unsigned int nt = 512, ops = 1;
      while (nt > max_n) {
        nt /= 2;
        ops *= 2;
      }
      ops = std::min(ops, batchCount);
      unsigned int nbx = (max_n + nt - 1) / nt,
        nbf = (batchCount + ops - 1) / ops;
      sycl::range<3> block(1, ops, nt);
      for (unsigned int f=0; f<nbf; f+=MAX_BLOCKS_Y) {
        sycl::range<3> grid(nbx, std::min(nbf - f, MAX_BLOCKS_Y), 1);
        auto f0 = f * ops;
        /*
          DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
          limit. To get the device limit, query info::device::max_work_group_size.
          Adjust the work-group size if needed.
        */
        get_sycl_queue(handle).parallel_for
          (sycl::nd_range<3>(grid * block, block),
           [=](sycl::nd_item<3> item_ct1) {
            laswp_vbatch_kernel(dn + f0, dA + f0, lddA + f0, dipiv + f0,
                                npivots + f0, batchCount - f0, item_ct1);
          });
      }
    }

    // explicit template instantiations
    template void laswp(Handle&, DenseMatrix<float>&, int, int, int*, int);
    template void laswp(Handle&, DenseMatrix<double>&, int, int, int*, int);
    template void laswp(Handle&, DenseMatrix<std::complex<float>>&, int, int, int*, int);
    template void laswp(Handle&, DenseMatrix<std::complex<double>>&, int, int, int*, int);

    template void laswp_fwd_vbatched(Handle&, int*, int, float**, int*, int**, int*, unsigned int);
    template void laswp_fwd_vbatched(Handle&, int*, int, double**, int*, int**, int*, unsigned int);
    template void laswp_fwd_vbatched(Handle&, int*, int, std::complex<float>**, int*, int**, int*, unsigned int);
    template void laswp_fwd_vbatched(Handle&, int*, int, std::complex<double>**, int*, int**, int*, unsigned int);

  } // end namespace gpu
} // end namespace strumpack
