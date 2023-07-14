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

#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

#include "DenseMatrix.hpp"


namespace strumpack {
  namespace dpcpp {

    inline void init() {
      std::cout << "# initializing DPC++/SYCL" << std::endl;
      cl::sycl::queue q; //(cl::sycl::default_selector{});
      cl::sycl::malloc_device<int>(0, q);
    }

    template<typename T> cl::sycl::event
    memcpy(cl::sycl::queue& q, T* dest, const T* src,
           std::size_t count) {
      return q.memcpy(dest, src, count*sizeof(T));
    }

    template<typename T> cl::sycl::event
    memcpy(cl::sycl::queue& q, DenseMatrix<T>& dest,
           const DenseMatrix<T>& src) {
      assert(dest.rows() == src.rows());
      assert(dest.cols() == src.cols());
      return memcpy(q, dest.data(), src.data(), dest.rows()*dest.cols());
    }

    template<typename T> cl::sycl::event
    fill(cl::sycl::queue& q, T* ptr, T value,
         std::size_t count) {
      return q.fill(ptr, value, count);
    }

    // inline std::size_t available_memory() {
    //   std::size_t free_device_mem = 0, total_device_mem = 0;
    //   // gpu_check(cudaMemGetInfo(&free_device_mem, &total_device_mem));
    //   std::cout << "TODO available_memory" << std::endl;
    //   return free_device_mem;
    // }

    template<typename T> class DeviceMemory {
    public:
      DeviceMemory() {}
      DeviceMemory(std::size_t size, cl::sycl::queue& q,
                   bool try_shared=true) {
        if (size) {
          data_ = cl::sycl::malloc_device<T>(size, q);
          size_ = size;
          q_ = &q;
          if (data_) {
            STRUMPACK_ADD_DEVICE_MEMORY(size*sizeof(T));
            is_managed_ = false;
          } else {
            if (!try_shared) throw std::bad_alloc();
            std::cerr << "#  Device memory allocation failed. "
                      << "#  Trying shared memory instead ..."
                      << std::endl;
            data_ = cl::sycl::malloc_shared<T>(size, q);
            if (!data_) throw std::bad_alloc();
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
          q_ = d.q_;
          is_managed_ = d.is_managed_;
          d.data_ = nullptr;
          d.release();
        }
        return *this;
      }
      ~DeviceMemory() { release(); }
      operator T*() { return data_; }
      operator const T*() const { return data_; }
      T* get() { return data_; }
      const T* get() const { return data_; }
      // operator void*() { return data_; }
      template<typename S> S* as() { return reinterpret_cast<S*>(data_); }
      void release() {
        if (data_) {
          if (is_managed_) {
            STRUMPACK_SUB_MEMORY(size_*sizeof(T));
          } else {
            STRUMPACK_SUB_DEVICE_MEMORY(size_*sizeof(T));
          }
          cl::sycl::free(data_, *q_);
        }
        data_ = nullptr;
        size_ = 0;
        q_ = nullptr;
        is_managed_ = false;
      }
    private:
      T* data_ = nullptr;
      std::size_t size_ = 0;
      cl::sycl::queue* q_ = nullptr;
      bool is_managed_ = false;
    };

    template<typename T> class HostMemory {
    public:
      HostMemory() {}
      HostMemory(std::size_t size, cl::sycl::queue& q) {
        if (size) {
          data_ = cl::sycl::malloc_host<T>(size, q);
          size_ = size;
          q_ = &q;
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
          q_ = d.q_;
          d.data_ = nullptr;
          d.q_ = nullptr;
          d.release();
        }
        return *this;
      }
      ~HostMemory() { release(); }
      operator T*() { return data_; }
      operator const T*() const { return data_; }
      T* get() { return data_; }
      const T* get() const { return data_; }
      // operator void*() { return data_; }
      template<typename S> S* as() { return reinterpret_cast<S*>(data_); }
      void release() {
        if (data_) {
          STRUMPACK_SUB_MEMORY(size_*sizeof(T));
          cl::sycl::free(data_, *q_);
        }
        data_ = nullptr;
        q_ = nullptr;
        size_ = 0;
      }
    private:
      T* data_ = nullptr;
      std::size_t size_ = 0;
      cl::sycl::queue* q_ = nullptr;
    };


    inline oneapi::mkl::transpose T2MKLOp(Trans op) {
      switch (op) {
      case Trans::N: return oneapi::mkl::transpose::N;
      case Trans::T: return oneapi::mkl::transpose::T;
      case Trans::C: return oneapi::mkl::transpose::C;
      default:
        assert(false);
        return oneapi::mkl::transpose::N;
      }
    }

    template<typename scalar_t> std::int64_t
    getrf_buffersize(cl::sycl::queue& q, int m, int n, int ld) {
      return oneapi::mkl::lapack::getrf_scratchpad_size<scalar_t>
        (q, m, n, ld);
    }

    template<typename scalar_t> std::int64_t
    getrs_buffersize(cl::sycl::queue& q, Trans t,
                     int n, int nrhs, int lda, int ldb) {
      return oneapi::mkl::lapack::getrs_scratchpad_size<scalar_t>
        (q, T2MKLOp(t), n, nrhs, lda, ldb);
    }

    template<typename scalar_t> cl::sycl::event
    getrf(cl::sycl::queue& q, DenseMatrix<scalar_t>& A,
          std::int64_t* ipiv, scalar_t* scratchpad,
          std::int64_t scratchpad_size,
          const std::vector<cl::sycl::event>& deps={}) {
      STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::getrf_flops(A.rows(),A.cols()));
      try {
        return oneapi::mkl::lapack::getrf
          (q, A.rows(), A.cols(), A.data(), A.ld(), ipiv,
           scratchpad, scratchpad_size, deps);
      } catch (oneapi::mkl::lapack::exception e) {
        std::cerr << "Exception in oneapi::mkl::lapack::getrf, info = "
                  << e.info() << std::endl;
      }
      return cl::sycl::event();
    }

    template<typename scalar_t> cl::sycl::event
    getrs(cl::sycl::queue& q, Trans trans,
          const DenseMatrix<scalar_t>& A, const std::int64_t* ipiv,
          DenseMatrix<scalar_t>& B,
          scalar_t* scratchpad, std::int64_t scratchpad_size,
          const std::vector<cl::sycl::event>& deps={}) {
      STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::getrs_flops(A.rows(),B.cols()));
      try {
        return oneapi::mkl::lapack::getrs
          (q, T2MKLOp(trans), A.rows(), B.cols(),
           const_cast<scalar_t*>(A.data()), A.ld(),
           const_cast<std::int64_t*>(ipiv),
           B.data(), B.ld(), scratchpad, scratchpad_size, deps);
      } catch (oneapi::mkl::lapack::exception e) {
        std::cerr << "Exception in oneapi::mkl::lapack::getrs, info = "
                  << e.info() << std::endl;
      }
      return cl::sycl::event();
    }

    template<typename scalar_t> cl::sycl::event
    gemm(cl::sycl::queue& q, Trans ta, Trans tb,
         scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const DenseMatrix<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c, const std::vector<cl::sycl::event>& deps={}) {
      STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::gemm_flops(c.rows(),c.cols(),(ta==Trans::N)?a.cols():a.rows(),alpha,beta));
      STRUMPACK_BYTES((is_complex<scalar_t>()?2:1)*sizeof(scalar_t)*blas::gemm_moves(c.rows(),c.cols(),(ta==Trans::N)?a.cols():a.rows()));
      assert((ta==Trans::N && a.rows()==c.rows()) ||
             (ta!=Trans::N && a.cols()==c.rows()));
      assert((tb==Trans::N && b.cols()==c.cols()) ||
             (tb!=Trans::N && b.rows()==c.cols()));
      assert((ta==Trans::N && tb==Trans::N && a.cols()==b.rows()) ||
             (ta!=Trans::N && tb==Trans::N && a.rows()==b.rows()) ||
             (ta==Trans::N && tb!=Trans::N && a.cols()==b.cols()) ||
             (ta!=Trans::N && tb!=Trans::N && a.rows()==b.cols()));
      return oneapi::mkl::blas::gemm
        (q, T2MKLOp(ta), T2MKLOp(tb), c.rows(), c.cols(),
         (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(), a.ld(),
         b.data(), b.ld(), beta, c.data(), c.ld(), deps);
    }

    template<typename scalar_t> cl::sycl::event
    gemv(cl::sycl::queue& q, Trans ta,
         scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const DenseMatrix<scalar_t>& x, scalar_t beta,
         DenseMatrix<scalar_t>& y, const std::vector<cl::sycl::event>& deps={}) {
      STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::gemv_flops(a.rows(),a.cols(),alpha,beta));
      STRUMPACK_BYTES((is_complex<scalar_t>()?2:1)*sizeof(scalar_t)*blas::gemv_moves(a.rows(),a.cols()));
      return oneapi::mkl::blas::gemv
        (q, T2MKLOp(ta), a.rows(), a.cols(), alpha,
         a.data(), a.ld(), x.data(), 1, beta, y.data(), 1, deps);
    }

  } // end namespace dpcpp
} // end namespace strumpack

#endif // STRUMPACK_DPCPP_WRAPPER_HPP
