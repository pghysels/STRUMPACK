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
#ifndef STRUMPACK_GPU_WRAPPER_HPP
#define STRUMPACK_GPU_WRAPPER_HPP

#include <memory>
#include "DenseMatrix.hpp"

namespace strumpack {
  namespace gpu {
#if defined(STRUMPACK_USE_GPU)

    class Stream {
    public:
      Stream();
      ~Stream();
      void synchronize();

      struct StreamImpl;
      std::unique_ptr<StreamImpl> s_;
    };

    class Handle {
    public:
      Handle();
      Handle(Stream& s);
      ~Handle();
      void set_stream(Stream& s);

      struct HandleImpl;
      std::unique_ptr<HandleImpl> h_;
    };

    class Event {
    public:
      Event();
      ~Event();
      void record();
      void record(Stream& s);
      void wait(Stream& s);
      void synchronize();

      struct EventImpl;
      std::unique_ptr<EventImpl> e_;
    };

    void device_malloc(void** ptr, std::size_t size);
    void device_free(void* ptr);
    void host_malloc(void** ptr, std::size_t size);
    void host_free(void* ptr);

    template<typename T> class DeviceMemory {
    public:
      DeviceMemory() {}
      DeviceMemory(std::size_t size) {
        if (size) {
          size_ = size;
          device_malloc((void**)&data_, size*sizeof(T));
          STRUMPACK_ADD_DEVICE_MEMORY(size*sizeof(T));
        }
      }
      DeviceMemory(const DeviceMemory&) = delete;
      DeviceMemory(DeviceMemory<T>&& d) {
        *this = std::forward<DeviceMemory<T>>(d);
      }
      DeviceMemory<T>& operator=(const DeviceMemory<T>&) = delete;
      DeviceMemory<T>& operator=(DeviceMemory<T>&& d) {
        if (this != &d) {
          release();
          data_ = d.data_;
          size_ = d.size_;
          d.data_ = nullptr;
          d.release();
        }
        return *this;
      }
      ~DeviceMemory() { release(); }
      std::size_t size() const { return size_; }
      operator T*() { return data_; }
      operator const T*() const { return data_; }
      template<typename S> S* as() {
        return reinterpret_cast<S*>(data_);
      }
      template<typename S> const S* as() const {
        return reinterpret_cast<S*>(data_);
      }
      void release() {
        if (data_) {
          STRUMPACK_SUB_DEVICE_MEMORY(size_*sizeof(T));
          device_free(data_);
          data_ = nullptr;
        }
        size_ = 0;
      }
    private:
      T* data_ = nullptr;
      std::size_t size_ = 0;
    };

    template<typename T> class HostMemory {
    public:
      HostMemory() {}
      HostMemory(std::size_t size) {
        size_ = size;
        host_malloc((void**)&data_, size*sizeof(T));
        STRUMPACK_ADD_MEMORY(size*sizeof(T));
      }
      HostMemory(const HostMemory&) = delete;
      HostMemory(HostMemory<T>&& d) {
        *this = std::forward<HostMemory<T>>(d);
      }
      HostMemory<T>& operator=(const HostMemory<T>&) = delete;
      HostMemory<T>& operator=(HostMemory<T>&& d) {
        if (this != & d) {
          release();
          data_ = d.data_;
          size_ = d.size_;
          d.data_ = nullptr;
          d.release();
        }
        return *this;
      }
      ~HostMemory() { release(); }
      std::size_t size() const { return size_; }
      T* data() { return data_; }
      const T* data() const { return data_; }
      operator T*() { return data_; }
      operator const T*() const { return data_; }
      template<typename S> S* as() { return reinterpret_cast<S*>(data_); }
      void release() {
        if (data_) {
          STRUMPACK_SUB_MEMORY(size_*sizeof(T));
          host_free(data_);
          data_ = nullptr;
        }
        size_ = 0;
      }
    private:
      T* data_ = nullptr;
      std::size_t size_ = 0;
    };

    void init();
    void peek_at_last_error();
    /**
     * This is used to reset the last error. Some MAGMA internal CUDA
     * kernels calls can fail, but MAGMA detects this and uses a
     * different algorithm.
     */
    void get_last_error();
    void synchronize_default_stream();
    std::size_t available_memory();

    void device_memset(void* dptr, int value, std::size_t count);

    template<typename T>
    void memset(void* dptr, int value, std::size_t count) {
      device_memset(dptr, value, sizeof(T)*count);
    }

    enum class CopyDir { H2H, H2D, D2H, D2D, DEF };

    void device_copy(void* dest, const void* src,
                     std::size_t count, CopyDir dir);

    void device_copy_async(void* dest, const void* src, std::size_t count,
                           CopyDir dir, Stream& s);

    void device_copy_2D(void* dest, std::size_t dpitch,
                        const void* src, std::size_t spitch,
                        std::size_t width, std::size_t height, CopyDir dir);

    void device_copy_2D_async(void* dest, std::size_t dpitch,
                              const void* src, std::size_t spitch,
                              std::size_t width, std::size_t height,
                              CopyDir dir, Stream& s);


    template<typename T> void
    copy(T* dst, const T* src, std::size_t count) {
      device_copy(dst, src, count*sizeof(T), CopyDir::DEF);
    }
    template<typename T> void
    copy_device_to_host(T* hptr, const T* dptr, std::size_t count) {
      device_copy(hptr, dptr, count*sizeof(T), CopyDir::D2H);
    }
    template<typename T> void
    copy_host_to_device(T* dptr, const T* hptr, std::size_t count) {
      device_copy(dptr, hptr, count*sizeof(T), CopyDir::H2D);
    }
    template<typename T> void
    copy_device_to_device(T* d1ptr, const T* d2ptr, std::size_t count) {
      device_copy(d1ptr, d2ptr, count*sizeof(T), CopyDir::D2D);
    }

    template<typename T> void
    copy_async(T* hptr, const T* dptr,
               std::size_t count, Stream& s) {
      device_copy_async(hptr, dptr, count*sizeof(T), CopyDir::DEF, s);
    }
    template<typename T> void
    copy_device_to_host_async(T* hptr, const T* dptr,
                              std::size_t count, Stream& s) {
      device_copy_async(hptr, dptr, count*sizeof(T), CopyDir::D2H, s);
    }
    template<typename T> void
    copy_host_to_device_async(T* dptr, const T* hptr,
                              std::size_t count, Stream& s) {
      device_copy_async(dptr, hptr, count*sizeof(T), CopyDir::H2D, s);
    }

    template<typename T> void
    copy_async(DenseMatrix<T>& h, const DenseMatrix<T>& d,
               Stream& s) {
      if (!d.rows() || !d.cols()) return;
      assert(d.rows() == h.rows() && d.cols() == h.cols());
      if (d.rows() != d.ld() || h.rows() != h.ld())
        device_copy_2D_async
          (h.data(), h.ld()*sizeof(T), d.data(), d.ld()*sizeof(T),
           h.rows()*sizeof(T), h.cols(), CopyDir::DEF, s);
      else
        copy_async(h.data(), d.data(), d.rows()*d.cols(), s);
    }
    template<typename T> void
    copy_device_to_host_async(DenseMatrix<T>& h, const DenseMatrix<T>& d,
                              Stream& s) {
      if (!d.rows() || !d.cols()) return;
      assert(d.rows() == h.rows() && d.cols() == h.cols());
      if (d.rows() != d.ld() || h.rows() != h.ld())
        device_copy_2D_async
          (h.data(), h.ld()*sizeof(T), d.data(), d.ld()*sizeof(T),
           h.rows()*sizeof(T), h.cols(), CopyDir::D2H, s);
      else
        copy_device_to_host_async(h.data(), d.data(), d.rows()*d.cols(), s);
    }
    template<typename T> void
    copy_host_to_device_async(DenseMatrix<T>& d, const DenseMatrix<T>& h,
                              Stream& s) {
      if (!d.rows() || !d.cols()) return;
      assert(d.rows() == h.rows() && d.cols() == h.cols());
      assert(d.rows() == d.ld() && h.rows() == h.ld());
      copy_host_to_device_async(d.data(), h.data(), std::size_t(d.rows())*d.cols(), s);
    }

    template<typename T> void
    copy_device_to_device(DenseMatrix<T>& d1, const DenseMatrix<T>& d2) {
      if (!d1.rows() || !d1.cols()) return;
      assert(d1.rows() == d2.rows() && d1.cols() == d2.cols());
      if (d1.rows() != d1.ld() || d2.rows() != d2.ld()) {
        device_copy_2D
          (d1.data(), d1.ld()*sizeof(T), d2.data(), d2.ld()*sizeof(T),
           d2.rows()*sizeof(T), d2.cols(), CopyDir::D2D);
      } else
        copy_device_to_device(d1.data(), d2.data(), d1.rows()*d1.cols());
    }

    template<typename T> void
    copy(DenseMatrix<T>& dst, const DenseMatrix<T>& src) {
      if (!dst.rows() || !dst.cols()) return;
      assert(src.rows() == dst.rows() && src.cols() == dst.cols());
      assert(src.rows() == src.ld() && dst.rows() == dst.ld());
      copy(dst.data(), src.data(), std::size_t(src.rows())*src.cols());
    }
    template<typename T> void
    copy_device_to_host(DenseMatrix<T>& h, const DenseMatrix<T>& d) {
      if (!h.rows() || !h.cols()) return;
      assert(d.rows() == h.rows() && d.cols() == h.cols());
      assert(d.rows() == d.ld() && h.rows() == h.ld());
      copy_device_to_host(h.data(), d.data(), std::size_t(d.rows())*d.cols());
    }
    template<typename T> void
    copy_host_to_device(DenseMatrix<T>& d, const DenseMatrix<T>& h) {
      if (!d.rows() || !d.cols()) return;
      assert(d.rows() == h.rows() && d.cols() == h.cols());
      assert(d.rows() == d.ld() && h.rows() == h.ld());
      copy_host_to_device(d.data(), h.data(), std::size_t(d.rows())*d.cols());
    }

    template<typename T> void
    copy(DenseMatrix<T>& dst, const T* src) {
      if (!dst.rows() || !dst.cols()) return;
      assert(dst.rows() == dst.ld());
      copy(dst.data(), src, std::size_t(dst.rows())*dst.cols());
    }
    template<typename T> void
    copy_device_to_host(DenseMatrix<T>& h, const T* d) {
      if (!h.rows() || !h.cols()) return;
      assert(h.rows() == h.ld());
      copy_device_to_host(h.data(), d, std::size_t(h.rows())*h.cols());
    }
    template<typename T> void
    copy_host_to_device(DenseMatrix<T>& d, const T* h) {
      if (!d.rows() || !d.cols()) return;
      assert(d.rows() == d.ld());
      copy_host_to_device(d.data(), h, std::size_t(d.rows())*d.cols());
    }
    template<typename T> void
    copy_device_to_device(DenseMatrix<T>& d1, const T* d2) {
      if (!d1.rows() || !d1.cols()) return;
      assert(d1.rows() == d1.ld());
      copy_device_to_device
        (d1.data(), d2, std::size_t(d1.rows())*d1.cols());
    }
    template<typename T> void
    copy_host_to_device_async(DenseMatrix<T>& d, const T* h,
                              Stream& s) {
      if (!d.rows() || !d.cols()) return;
      assert(d.rows() == d.ld());
      copy_host_to_device_async(d.data(), h, d.rows()*d.cols(), s);
    }

    template<typename T> void
    copy(T* dst, const DenseMatrix<T>& src) {
      if (!src.rows() || !src.cols()) return;
      assert(src.rows() == src.ld());
      copy(dst, src.data(), std::size_t(src.rows())*src.cols());
    }
    template<typename T> void
    copy_device_to_host(T* h, const DenseMatrix<T>& d) {
      if (!d.rows() || !d.cols()) return;
      assert(d.rows() == d.ld());
      copy_device_to_host(h, d.data(), std::size_t(d.rows())*d.cols());
    }
    template<typename T> void
    copy_host_to_device(T* d, const DenseMatrix<T>& h) {
      if (!h.rows() || !h.cols()) return;
      assert(h.rows() == h.ld());
      copy_host_to_device(d, h.data(), std::size_t(h.rows())*h.cols());
    }
    template<typename T> void
    copy_device_to_device(T* d1, const DenseMatrix<T>& d2) {
      if (!d2.rows() || !d2.cols()) return;
      assert(d2.rows() == d2.ld());
      copy_device_to_device(d1, d2.data(), std::size_t(d2.rows())*d2.cols());
    }
    template<typename T> void
    copy_host_to_device_async(T* d, const DenseMatrix<T>& h, Stream& s) {
      if (!h.rows() || !h.cols()) return;
      assert(h.rows() == h.ld());
      copy_host_to_device_async(d, h.data(), h.rows()*h.cols(), s);
    }

    template<typename scalar_t,
             typename real_t=typename RealType<scalar_t>::value_type>
    void
    copy_real_to_scalar(scalar_t* dest, const real_t* src, std::size_t size) {
      memset<scalar_t>(dest, 0, size);
      device_copy_2D(dest, sizeof(scalar_t), src, sizeof(real_t),
                     sizeof(real_t), size, CopyDir::D2D);
    }


    template<typename scalar_t> std::int64_t
    getrf_buffersize(Handle& handle, int n);

    template<typename scalar_t> void
    getrf(Handle& handle, DenseMatrix<scalar_t>& A,
          scalar_t* work, std::int64_t lwork, int* dpiv, int* dinfo);

    template<typename scalar_t> std::int64_t
    getrs_buffersize(Handle& handle, Trans t,
                     int n, int nrhs, int lda, int ldb);

    template<typename scalar_t> void
    getrs(Handle& handle, Trans trans,
          const DenseMatrix<scalar_t>& A, const int* dpiv,
          DenseMatrix<scalar_t>& B, int *dinfo,
          scalar_t* work=nullptr, std::int64_t lwork=0);

    template<typename scalar_t>
    int potrf_buffersize(Handle& handle, UpLo uplo, int n);

    template<typename scalar_t> void
    potrf(Handle& handle, UpLo uplo, DenseMatrix<scalar_t>& A,
          scalar_t* Workspace, int Lwork, int* devInfo);

    template<typename scalar_t> void
    syrk(Handle& handle, UpLo uplo, Trans ta,
         scalar_t alpha, const DenseMatrix<scalar_t>& a,
         scalar_t beta, DenseMatrix<scalar_t>& c);

        template<typename scalar_t> void
    trsm(Handle& handle, Side side, UpLo uplo,
         Trans trans, Diag diag, const scalar_t alpha,
         DenseMatrix<scalar_t>& A, DenseMatrix<scalar_t>& B);

    template<typename scalar_t,
             typename real_t=typename RealType<scalar_t>::value_type> int
    gesvdj_buffersize(Handle& handle, Jobz jobz, int m, int n);

    template<typename scalar_t,
             typename real_t=typename RealType<scalar_t>::value_type> void
    gesvdj(Handle& handle, Jobz jobz, DenseMatrix<scalar_t>& A,
           real_t* S, DenseMatrix<scalar_t>& U,
           DenseMatrix<scalar_t>& V, int* devInfo,
           scalar_t* work, int lwork, const double tol);

    template<typename scalar_t> void
    geam(Handle& handle, Trans transa, Trans transb, const scalar_t alpha,
         const DenseMatrix<scalar_t>& A, const scalar_t beta,
         const DenseMatrix<scalar_t>& B, DenseMatrix<scalar_t>& C);

    template<typename scalar_t> void
    dgmm(Handle& handle, Side side, const DenseMatrix<scalar_t>& A,
         const scalar_t* x, DenseMatrix<scalar_t>& C);

    template<typename scalar_t> void
    gemm(Handle& handle, Trans ta, Trans tb,
         scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const DenseMatrix<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c);

    template<typename scalar_t> void
    gemv(Handle& handle, Trans ta,
         scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const DenseMatrix<scalar_t>& x, scalar_t beta,
         DenseMatrix<scalar_t>& y);

    template<typename scalar_t> void
    laswp(Handle& handle, DenseMatrix<scalar_t>& A,
          int k1, int k2, int* ipiv, int inc);

    // assume inc = 1
    template<typename scalar_t> void
    laswp_fwd_vbatched(Handle& handle, int* dn, int max_n,
                       scalar_t** dA, int* lddA, int** dipiv, int* npivots,
                       unsigned int batchCount);

#endif // STRUMPACK_USE_GPU

  } // end namespace gpu
} // end namespace strumpack

#endif // STRUMPACK_GPU_WRAPPER_HPP

