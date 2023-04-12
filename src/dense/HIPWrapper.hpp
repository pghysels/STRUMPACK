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
#ifndef STRUMPACK_HIP_WRAPPER_HPP
#define STRUMPACK_HIP_WRAPPER_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <cassert>
#include <memory>

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocsolver/rocsolver.h>

#include "DenseMatrix.hpp"

namespace strumpack {
  namespace gpu {

    const unsigned int MAX_BLOCKS_Y = 65535;
    const unsigned int MAX_BLOCKS_Z = 65535;

#define gpu_check(err) {                                              \
      strumpack::gpu::hip_assert((err), __FILE__, __LINE__);          \
    }
    void hip_assert(hipError_t code, const char *file, int line,
                    bool abort=true);
    void hip_assert(rocblas_status code, const char *file, int line,
                    bool abort=true);
    void hip_assert(hipblasStatus_t code, const char *file, int line,
                    bool abort=true);

    void init();

    inline void synchronize() {
      gpu_check(hipDeviceSynchronize());
    }

    class Stream {
    public:
      Stream() { gpu_check(hipStreamCreate(&s_)); }
      ~Stream() { gpu_check(hipStreamDestroy(s_)); }
      operator hipStream_t&() { return s_; }
      operator const hipStream_t&() const { return s_; }
      void synchronize() { gpu_check(hipStreamSynchronize(s_)); }
    private:
      hipStream_t s_;
    };

    class BLASHandle {
    public:
      BLASHandle() { gpu_check(hipblasCreate(&h_)); }
      BLASHandle(Stream& s) : BLASHandle() { set_stream(s); }
      ~BLASHandle() { gpu_check(hipblasDestroy(h_)); }
      void set_stream(Stream& s) { gpu_check(hipblasSetStream(h_, s)); }
      operator hipblasHandle_t&() { return h_; }
      operator const hipblasHandle_t&() const { return h_; }
    private:
      hipblasHandle_t h_;
    };

    // TODO there is no such thing as hipSOLVER yet :(
    class SOLVERHandle {
    public:
      SOLVERHandle() { gpu_check(rocblas_create_handle(&h_)); }
      SOLVERHandle(Stream& s) : SOLVERHandle() { set_stream(s); }
      ~SOLVERHandle() { gpu_check(rocblas_destroy_handle(h_)); }
      void set_stream(Stream& s) { rocblas_set_stream(h_, s); }
      operator rocblas_handle&() { return h_; }
      operator const rocblas_handle&() const { return h_; }
    private:
      rocblas_handle h_;
    };

    class Event {
    public:
      Event() { gpu_check(hipEventCreateWithFlags
                          (&e_, hipEventDisableTiming)); }
      ~Event() { gpu_check(hipEventDestroy(e_)); }
      void record() { gpu_check(hipEventRecord(e_)); }
      void record(Stream& s) { gpu_check(hipEventRecord(e_, s)); }
      void wait(Stream& s) { gpu_check(hipStreamWaitEvent(s, e_, 0)); }
    private:
      hipEvent_t e_;
    };

    template<typename T> hipError_t
    memset(void* dptr, int value, std::size_t count) {
      return hipMemset(dptr, value, count*sizeof(T));
    }

    template<typename T> hipError_t
    copy_device_to_host(T* hptr, const T* dptr, std::size_t count) {
      return hipMemcpy(hptr, dptr, count*sizeof(T),
                       hipMemcpyDeviceToHost);
    }
    template<typename T> hipError_t
    copy_device_to_host_async(T* hptr, const T* dptr,
                              std::size_t count, const Stream& s) {
      return hipMemcpyAsync(hptr, dptr, count*sizeof(T),
                            hipMemcpyDeviceToHost, s);
    }
    template<typename T> hipError_t
    copy_host_to_device(T* dptr, const T* hptr, std::size_t count) {
      return hipMemcpy(dptr, hptr, count*sizeof(T),
                       hipMemcpyHostToDevice);
    }
    template<typename T> hipError_t
    copy_host_to_device_async(T* dptr, const T* hptr,
                              std::size_t count, const Stream& s) {
      return hipMemcpyAsync(dptr, hptr, count*sizeof(T),
                            hipMemcpyHostToDevice, s);
    }

    template<typename T> hipError_t
    copy_device_to_host(DenseMatrix<T>& h, const DenseMatrix<T>& d) {
      assert(d.rows() == h.rows() && d.cols() == h.cols());
      assert(d.rows() == d.ld() && h.rows() == h.ld());
      return copy_device_to_host
        (h.data(), d.data(), std::size_t(d.rows())*d.cols());
    }
    template<typename T> hipError_t
    copy_device_to_host(DenseMatrix<T>& h, const T* d) {
      assert(h.rows() == h.ld());
      return copy_device_to_host
        (h.data(), d, std::size_t(h.rows())*h.cols());
    }
    template<typename T> hipError_t
    copy_device_to_host(T* h, const DenseMatrix<T>& d) {
      assert(d.rows() == d.ld());
      return copy_device_to_host
        (h, d.data(), std::size_t(d.rows())*d.cols());
    }
    template<typename T> hipError_t
    copy_host_to_device(DenseMatrix<T>& d, const DenseMatrix<T>& h) {
      assert(d.rows() == h.rows() && d.cols() == h.cols());
      assert(d.rows() == d.ld() && h.rows() == h.ld());
      return copy_host_to_device
        (d.data(), h.data(), std::size_t(d.rows())*d.cols());
    }
    template<typename T> hipError_t
    copy_host_to_device(DenseMatrix<T>& d, const T* h) {
      assert(d.rows() == d.ld());
      return copy_host_to_device
        (d.data(), h, std::size_t(d.rows())*d.cols());
    }
    template<typename T> hipError_t
    copy_host_to_device(T* d, const DenseMatrix<T>& h) {
      assert(h.rows() == h.ld());
      return copy_host_to_device
        (d, h.data(), std::size_t(h.rows())*h.cols());
    }


    inline std::size_t available_memory() {
      std::size_t free_device_mem, total_device_mem;
      gpu_check(hipMemGetInfo(&free_device_mem, &total_device_mem));
      return free_device_mem;
    }

    template<typename T> class DeviceMemory {
    public:
      DeviceMemory() {}
      DeviceMemory(std::size_t size) {
        if (size) {
          size_ = size;
#if 0
          if (hipMalloc(&data_, size*sizeof(T)) != hipSuccess) {
            std::cerr << "Failed to allocate " << size << " "
                      << typeid(T).name() << " objects" << std::endl;
            throw std::bad_alloc();
          }
          STRUMPACK_ADD_DEVICE_MEMORY(size*sizeof(T));
          is_managed_ = false;
#else
          auto e = hipMalloc(&data_, size*sizeof(T));
          if (e == hipSuccess) {
            STRUMPACK_ADD_DEVICE_MEMORY(size*sizeof(T));
            is_managed_ = false;
          } else {
            STRUMPACK_ADD_MEMORY(size*sizeof(T));
            std::cerr << "#  hipMalloc failed: "
                      << hipGetErrorString(e) << std::endl;
            std::cerr << "#  Trying hipMallocManaged instead ..."
                      << std::endl;
            gpu_check(hipGetLastError()); // reset to hipSuccess
            gpu_check(hipMallocManaged(&data_, size*sizeof(T)));
            is_managed_ = true;
          }
#endif
        }
      }
      DeviceMemory(const DeviceMemory&) = delete;
      DeviceMemory(DeviceMemory<T>&& d) { *this = d;}
      DeviceMemory<T>& operator=(const DeviceMemory<T>&) = delete;
      DeviceMemory<T>& operator=(DeviceMemory<T>&& d) {
        if (this != &d) {
          release();
          data_ = d.data_;
          size_ = d.size_;
          is_managed_ = d.is_managed_;
          d.data_ = nullptr;
          d.release();
        }
        return *this;
      }
      ~DeviceMemory() { release(); }
      std::size_t size() const { return size_; }
      operator T*() { return data_; }
      operator const T*() const { return data_; }
      // operator void*() { return data_; }
      template<typename S> S* as() { return reinterpret_cast<S*>(data_); }
      void release() {
        if (data_) {
          if (is_managed_) {
            STRUMPACK_SUB_MEMORY(size_*sizeof(T));
          } else {
            STRUMPACK_SUB_DEVICE_MEMORY(size_*sizeof(T));
          }
          gpu_check(hipFree(data_));
        }
        data_ = nullptr;
        size_ = 0;
        is_managed_ = false;
      }
    private:
      T* data_ = nullptr;
      std::size_t size_ = 0;
      bool is_managed_ = false;
    };

    template<typename T> class HostMemory {
    public:
      HostMemory() {}
      HostMemory(std::size_t size) {
        if (size) {
          STRUMPACK_ADD_MEMORY(size*sizeof(T));
          auto e = hipHostMalloc(&data_, size*sizeof(T));
          is_managed_ = false;
          size_ = size;
          if (e != hipSuccess) {
            std::cerr << "#  hipMalloc failed: "
                      << hipGetErrorString(e) << std::endl;
            std::cerr << "#  Trying hipMallocManaged instead ..."
                      << std::endl;
            gpu_check(hipGetLastError()); // reset to hipSuccess
            gpu_check(hipMallocManaged(&data_, size*sizeof(T)));
            is_managed_ = true;
          }
        }
      }
      HostMemory(const HostMemory&) = delete;
      HostMemory(HostMemory<T>&& d) { *this = d; }
      HostMemory<T>& operator=(const HostMemory<T>&) = delete;
      HostMemory<T>& operator=(HostMemory<T>&& d) {
        if (this != & d) {
          release();
          data_ = d.data_;
          size_ = d.size_;
          is_managed_ = d.is_managed_;
          d.data_ = nullptr;
          d.release();
        }
        return *this;
      }
      ~HostMemory() { release(); }
      std::size_t size() const { return size_; }
      operator T*() { return data_; }
      operator const T*() const { return data_; }
      // operator void*() { return data_; }
      template<typename S> S* as() { return reinterpret_cast<S*>(data_); }
      void release() {
        if (data_) {
          STRUMPACK_SUB_MEMORY(size_*sizeof(T));
          if (!is_managed_) {
            gpu_check(hipHostFree(data_));
          } else {
            gpu_check(hipFree(data_));
          }
        }
        data_ = nullptr;
        size_ = 0;
        is_managed_ = false;
      }
    private:
      T* data_ = nullptr;
      std::size_t size_ = 0;
      bool is_managed_ = false;
    };

    template<typename scalar_t>
    int getrf_buffersize(SOLVERHandle& handle, int n);

    template<typename scalar_t> void
    getrf(SOLVERHandle& handle, DenseMatrix<scalar_t>& A,
          scalar_t* Workspace, int* devIpiv, int* devInfo);

    template<typename scalar_t> void
    getrs(SOLVERHandle& handle, Trans trans,
          const DenseMatrix<scalar_t>& A, const int* devIpiv,
          DenseMatrix<scalar_t>& B, int *devInfo);

    template<typename scalar_t> void
    gemm(BLASHandle& handle, Trans ta, Trans tb,
         scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const DenseMatrix<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c);

    template<typename scalar_t> void
    gemv(BLASHandle& handle, Trans ta,
         scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const DenseMatrix<scalar_t>& x, scalar_t beta,
         DenseMatrix<scalar_t>& y);

    template<typename scalar_t> void
    laswp(BLASHandle& handle, DenseMatrix<scalar_t>& A,
          int k1, int k2, int* ipiv, int inc);

    // assume inc = 1
    template<typename scalar_t> void
    laswp_fwd_vbatched(BLASHandle& handle, int* dn, int max_n,
                       scalar_t** dA, int* lddA, int** dipiv, int* npivots,
                       unsigned int batchCount);

  } // end namespace gpu
} // end namespace strumpack

#endif // STRUMPACK_HIP_WRAPPER_HPP
