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
#ifndef STRUMPACK_DPCPP_WRAPPER_HPP
#define STRUMPACK_DPCPP_WRAPPER_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <cassert>
#include <memory>

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

#include "DenseMatrix.hpp"


namespace strumpack {
  namespace gpu {

    inline void init() {
      std::cout << "TODO DPC++ init" << std::endl;
    }

    class Stream {
    public:
      Stream() { 
	// TODO query devices, allow multiple?
	q_ = cl::sycl::queue
	  (cl::sycl::device(cl::sycl::gpu_selector()),
	   {cl::sycl::property::queue::in_order()});
      }
      operator cl::sycl::queue&() { return q_; }
      operator const cl::sycl::queue&() const { return q_; }
      cl::sycl::queue& queue() { return q_; }
      const cl::sycl::queue& queue() const { return q_; }
    private:
      cl::sycl::queue q_;
    };

    inline void synchronize(Stream& s) {
      static_cast<cl::sycl::queue&>(s).wait();
    }
    inline void synchronize(std::vector<Stream>& s) {
      for (auto& i : s) static_cast<cl::sycl::queue&>(i).wait();
    }

    class BLASHandle {
    public:
      void set_stream(Stream& s) { q_ = &s.queue(); }
      operator cl::sycl::queue&() { assert(q_); return *q_; }
      operator const cl::sycl::queue&() const { assert(q_); return *q_; }
    private:
      cl::sycl::queue* q_ = nullptr;
    };

    using SOLVERHandle = BLASHandle;

    // TODO create an Event class????

    template<typename T> void memset
    (T* dptr, int value, std::size_t count, Stream& s) {
      s.queue().memset(dptr, value, count).wait();
    }

    template<typename T> void copy_device_to_host
    (T* hptr, const T* dptr, std::size_t count, Stream& s) {
      s.queue().memcpy(hptr, dptr, count*sizeof(T)).wait();
    }
    template<typename T> void copy_device_to_host_async
    (T* hptr, const T* dptr, std::size_t count, Stream& s) {
      // event is ignored!
      s.queue().memcpy(hptr, dptr, count*sizeof(T));
    }
    template<typename T> void copy_host_to_device
    (T* dptr, const T* hptr, std::size_t count, Stream& s) {
      s.queue().memcpy(dptr, hptr, count*sizeof(T)).wait();
    }
    template<typename T> void copy_host_to_device_async
    (T* dptr, const T* hptr, std::size_t count, Stream& s) {
      // event is ignored!
      s.queue().memcpy(dptr, hptr, count*sizeof(T));
    }

    template<typename T> void copy_device_to_host
    (DenseMatrix<T>& h, const DenseMatrix<T>& d, Stream& s) {
      assert(d.rows() == h.rows() && d.cols() == h.cols());
      assert(d.rows() == d.ld() && h.rows() == h.ld());
      copy_device_to_host(h.data(), d.data(), d.rows()*d.cols(), s);
    }
    template<typename T> void copy_device_to_host
    (DenseMatrix<T>& h, const T* d, Stream& s) {
      assert(h.rows() == h.ld());
      copy_device_to_host(h.data(), d, h.rows()*h.cols(), s);
    }
    template<typename T> void copy_device_to_host
    (T* h, const DenseMatrix<T>& d, Stream& s) {
      assert(d.rows() == d.ld());
      copy_device_to_host(h, d.data(), d.rows()*d.cols(), s);
    }
    template<typename T> void copy_host_to_device
    (DenseMatrix<T>& d, const DenseMatrix<T>& h, Stream& s) {
      assert(d.rows() == h.rows() && d.cols() == h.cols());
      assert(d.rows() == d.ld() && h.rows() == h.ld());
      copy_host_to_device(d.data(), h.data(), d.rows()*d.cols(), s);
    }
    template<typename T> void copy_host_to_device
    (DenseMatrix<T>& d, const T* h, Stream& s) {
      assert(d.rows() == d.ld());
      copy_host_to_device(d.data(), h, d.rows()*d.cols(), s);
    }
    template<typename T> void copy_host_to_device
    (T* d, const DenseMatrix<T>& h, Stream& s) {
      assert(h.rows() == h.ld());
      copy_host_to_device(d, h.data(), h.rows()*h.cols(), s);
    }


    inline std::size_t available_memory() {
      std::size_t free_device_mem = 0, total_device_mem = 0;
      // gpu_check(cudaMemGetInfo(&free_device_mem, &total_device_mem));
      std::cout << "TODO available_memory" << std::endl;
      return free_device_mem;
    }

    template<typename T> class DeviceMemory {
    public:
      DeviceMemory() {}
      DeviceMemory(std::size_t size, Stream& s) {
        if (size) {
	  data_ = cl::sycl::malloc_device<T>(size, s);	
          size_ = size;
	  s_ = &s.queue();
	  if (data_) {
	    STRUMPACK_ADD_DEVICE_MEMORY(size*sizeof(T));
	    is_managed_ = false;
	  } else {
            std::cerr << "#  Device memory allocation failed. "
		      << "#  Trying shared memory instead ..."
                      << std::endl;
	    data_ = cl::sycl::malloc_shared<T>(size, s);
            STRUMPACK_ADD_MEMORY(size*sizeof(T));
            is_managed_ = true;
          }
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
	  s_ = d.s_;
          is_managed_ = d.is_managed_;
          d.data_ = nullptr;
          d.release();
        }
        return *this;
      }
      ~DeviceMemory() { release(); }
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
          free(data_, *s_);
        }
        data_ = nullptr;
        size_ = 0;
	s_ = nullptr;
        is_managed_ = false;
      }
    private:
      T* data_ = nullptr;
      std::size_t size_ = 0;
      cl::sycl::queue* s_ = nullptr;
      bool is_managed_ = false;
    };

    template<typename T> class HostMemory {
    public:
      HostMemory() {}
      HostMemory(std::size_t size, Stream& s) {
        if (size) {
	  data_ = cl::sycl::malloc_host<T>(size);
          size_ = size;
	  s_ = &s.queue();
          STRUMPACK_ADD_MEMORY(size*sizeof(T));
          if (!data_)
            std::cerr << "#  Malloc failed." << std::endl;
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
	  s_ = d.s_;
          d.data_ = nullptr;
	  d.s_ = nullptr;
          d.release();
        }
        return *this;
      }
      ~HostMemory() { release(); }
      operator T*() { return data_; }
      operator const T*() const { return data_; }
      // operator void*() { return data_; }
      template<typename S> S* as() { return reinterpret_cast<S*>(data_); }
      void release() {
        if (data_) {
          STRUMPACK_SUB_MEMORY(size_*sizeof(T));
	  free(data_, *s_);
        }
        data_ = nullptr;
	s_ = nullptr;
        size_ = 0;
      }
    private:
      T* data_ = nullptr;
      std::size_t size_ = 0;
      cl::sycl::queue* s_ = nullptr;
    };


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

    template<typename scalar_t>
    int getrf_buffersize(SOLVERHandle& handle, int m, int n, int ld) {
      return oneapi::mkl::lapack::getrf_scratchpad_size<scalar_t>
	(handle, m, n, ld);
    }
    template<typename scalar_t>
    int getrf_buffersize(SOLVERHandle& handle, int n) {
      return getrf_buffersize<scalar_t>(handle, n, n, n);
    }

    template<typename scalar_t> void
    getrf(SOLVERHandle& handle, DenseMatrix<scalar_t>& A,
          scalar_t* Workspace, int* devIpiv, int* devInfo) {
      if (!is_complex<scalar_t>()) {
	STRUMPACK_FLOPS(blas::getrf_flops(m,n));
      } else {
	STRUMPACK_FLOPS(4*blas::getrf_flops(m,n));
      }
      try {
	// event is ignored?!
	oneapi::mkl::lapack::getrf
	  (handle, A.rows(), A.cols(), A.data(), A.ld(), devIpiv,
	   // this assumes that getrf_buffersize return the same value!!
	   Workspace, getrf_buffersize<scalar_t>(handle, A.rows(), A.cols(), A.ld()));
      } catch (oneapi::mkl::lapack::exception e) {
	std::cout << "Exception in oneapi::mkl::lapack::getrf, info = "
		  << e.info() << std::endl;
      }
      *devInfo = 0;
    }

    template<typename scalar_t> void
    getrs(SOLVERHandle& handle, Trans trans,
          const DenseMatrix<scalar_t>& A, const int* devIpiv,
          DenseMatrix<scalar_t>& B, int *devInfo) {
      if (!is_complex<scalar_t>()) {
	STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
      } else {
	STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
      }
      try {
	// event is ignored?!
	oneapi::mkl::lapack::getrs
	  (handle, T2MKLOp(trans), A.rows(), B.cols(), 
	   A.data(), A.ld(), devIpiv, B.data(), B.ld(), 
	   // TODO 
	   nullptr, 0);
      } catch (oneapi::mkl::lapack::exception e) {
	std::cout << "Exception in oneapi::mkl::lapack::getrs, info = "
		  << e.info() << std::endl;
      }
      *devInfo = 0;
    }

    template<typename scalar_t> void
    gemm(BLASHandle& handle, Trans ta, Trans tb,
         scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const DenseMatrix<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c) {
      if (!is_complex<scalar_t>()) {
	STRUMPACK_FLOPS(blas::gemm_flops(m,n,k,alpha,beta));
	STRUMPACK_BYTES(sizeof(scalar_t)*blas::gemm_moves(m,n,k));
      } else {
	STRUMPACK_FLOPS(4*blas::gemm_flops(m,n,k,alpha,beta));
	STRUMPACK_BYTES(2*sizeof(scalar_t)*blas::gemm_moves(m,n,k));
      }
      assert((ta==Trans::N && a.rows()==c.rows()) ||
             (ta!=Trans::N && a.cols()==c.rows()));
      assert((tb==Trans::N && b.cols()==c.cols()) ||
             (tb!=Trans::N && b.rows()==c.cols()));
      assert((ta==Trans::N && tb==Trans::N && a.cols()==b.rows()) ||
             (ta!=Trans::N && tb==Trans::N && a.rows()==b.rows()) ||
             (ta==Trans::N && tb!=Trans::N && a.cols()==b.cols()) ||
             (ta!=Trans::N && tb!=Trans::N && a.rows()==b.cols()));
      // event is ignored?!
      oneapi::mkl::blas::gemm
	(handle, T2MKLOp(ta), T2MKLOp(tb), c.rows(), c.cols(),
	 (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(), a.ld(),
	 b.data(), b.ld(), beta, c.data(), c.ld());
    }

    template<typename scalar_t> void
    gemv(BLASHandle& handle, Trans ta,
         scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const DenseMatrix<scalar_t>& x, scalar_t beta,
         DenseMatrix<scalar_t>& y) {
      if (!is_complex<scalar_t>()) {
	STRUMPACK_FLOPS(blas::gemv_flops(m,n,alpha,beta));
	STRUMPACK_BYTES(sizeof(scalar_t)*blas::gemv_moves(m,n));
      } else {
	STRUMPACK_FLOPS(4*blas::gemv_flops(m,n,alpha,beta));
	STRUMPACK_BYTES(2*sizeof(scalar_t)*blas::gemv_moves(m,n));
      }
      // event is ignored?!
      oneapi::mkl::blas::gemv
	(handle, T2MKLOp(ta), a.rows(), a.cols(), alpha,
	 a.data(), a.ld(), x.data(), 1, beta, y.data(), 1);
    }

  } // end namespace gpu
} // end namespace strumpack

#endif // STRUMPACK_DPCPP_WRAPPER_HPP
