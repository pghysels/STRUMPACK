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
#ifndef STRUMPACK_CUDA_WRAPPER_HPP
#define STRUMPACK_CUDA_WRAPPER_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <cassert>
#include <memory>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "BLASLAPACKWrapper.hpp"


namespace strumpack {

  namespace cuda {

    template<typename T> class CudaDeviceMemory {
    public:
      CudaDeviceMemory() {}
      CudaDeviceMemory(std::size_t size) {
        STRUMPACK_ADD_DEVICE_MEMORY(size*sizeof(T));
        cudaMalloc(&data_, size*sizeof(T));
        size_ = size;
      }
      CudaDeviceMemory(const CudaDeviceMemory&) = delete;
      CudaDeviceMemory(CudaDeviceMemory<T>&& d) { *this = d;}
      CudaDeviceMemory<T>& operator=(const CudaDeviceMemory<T>&) = delete;
      CudaDeviceMemory<T>& operator=(CudaDeviceMemory<T>&& d) {
        if (this != &d) {
          release();
          data_ = d.data_;
          d.data_ = nullptr;
        }
        return *this;
      }
      ~CudaDeviceMemory() { release(); }
      operator T*() { return data_; }
      operator void*() { return data_; }
      template<typename S> S* as() { return reinterpret_cast<S*>(data_); }
      void release() {
        if (data_) {
          STRUMPACK_SUB_DEVICE_MEMORY(size_*sizeof(T));
          cudaFree(data_);
        }
        data_ = nullptr;
        size_ = 0;
      }
    private:
      T* data_ = nullptr;
      std::size_t size_ = 0;
    };

    template<typename T> class CudaHostMemory {
    public:
      CudaHostMemory() {}
      CudaHostMemory(std::size_t size) {
        STRUMPACK_ADD_MEMORY(size*sizeof(T));
        cudaMallocHost(&data_, size*sizeof(T));
      }
      CudaHostMemory(const CudaHostMemory&) = delete;
      CudaHostMemory(CudaHostMemory<T>&& d) { *this = d; }
      CudaHostMemory<T>& operator=(const CudaHostMemory<T>&) = delete;
      CudaHostMemory<T>& operator=(CudaHostMemory<T>&& d) {
        if (this != & d) {
          release();
          data_ = d.data_;
          d.data_ = nullptr;
        }
        return *this;
      }
      ~CudaHostMemory() { release(); }
      operator T*() { return data_; }
      operator void*() { return data_; }
      template<typename S> S* as() { return reinterpret_cast<S*>(data_); }
      void release() {
        if (data_) {
          STRUMPACK_ADD_MEMORY(size_*sizeof(T));
          cudaFreeHost(data_);
        }
        data_ = nullptr;
        size_ = 0;
      }
    private:
      T* data_ = nullptr;
      std::size_t size_ = 0;
    };


    inline cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa,
     cublasOperation_t transb, int m, int n, int k, float alpha,
     const float* A, int lda, const float* B, int ldb,
     float beta, float* C, int ldc) {
      STRUMPACK_FLOPS(blas::gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(4*blas::gemm_moves(m,n,k));
      return cublasSgemm
        (handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    inline cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa,
     cublasOperation_t transb, int m, int n, int k, double alpha,
     const double* A, int lda, const double* B, int ldb,
     double beta, double* C, int ldc) {
      STRUMPACK_FLOPS(blas::gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(8*blas::gemm_moves(m,n,k));
      return cublasDgemm
        (handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    inline cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa,
     cublasOperation_t transb, int m, int n, int k, std::complex<float> alpha,
     const std::complex<float>* A, int lda,
     const std::complex<float>* B, int ldb,
     std::complex<float> beta, std::complex<float> *C, int ldc) {
      STRUMPACK_FLOPS(4*blas::gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(2*4*blas::gemm_moves(m,n,k));
      return cublasCgemm
        (handle, transa, transb, m, n, k,
         reinterpret_cast<cuComplex*>(&alpha),
         reinterpret_cast<const cuComplex*>(A), lda,
         reinterpret_cast<const cuComplex*>(B), ldb,
         reinterpret_cast<cuComplex*>(&beta),
         reinterpret_cast<cuComplex*>(C), ldc);
    }
    inline cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa,
     cublasOperation_t transb, int m, int n, int k,
     std::complex<double> alpha, const std::complex<double> *A, int lda,
     const std::complex<double> *B, int ldb, std::complex<double> beta,
     std::complex<double> *C, int ldc) {
      STRUMPACK_FLOPS(4*blas::gemm_flops(m,n,k,alpha,beta));
      STRUMPACK_BYTES(2*8*blas::gemm_moves(m,n,k));
      return cublasZgemm
        (handle, transa, transb, m, n, k,
         reinterpret_cast<cuDoubleComplex*>(&alpha),
         reinterpret_cast<const cuDoubleComplex*>(A), lda,
         reinterpret_cast<const cuDoubleComplex*>(B), ldb,
         reinterpret_cast<cuDoubleComplex*>(&beta),
         reinterpret_cast<cuDoubleComplex*>(C), ldc);
    }

    inline cublasStatus_t cublasgemv
    (cublasHandle_t handle, cublasOperation_t transa,
     int m, int n, float alpha,
     const float* A, int lda, const float* B, int incb,
     float beta, float* C, int incc) {
      STRUMPACK_FLOPS(blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(4*blas::gemv_moves(m,n));
      return cublasSgemv
        (handle, transa, m, n, &alpha, A, lda, B, incb, &beta, C, incc);
    }
    inline cublasStatus_t cublasgemv
    (cublasHandle_t handle, cublasOperation_t transa,
     int m, int n, double alpha,
     const double* A, int lda, const double* B, int incb,
     double beta, double* C, int incc) {
      STRUMPACK_FLOPS(blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(8*blas::gemv_moves(m,n));
      return cublasDgemv
        (handle, transa, m, n, &alpha, A, lda, B, incb, &beta, C, incc);
    }
    inline cublasStatus_t cublasgemv
    (cublasHandle_t handle, cublasOperation_t transa,
     int m, int n, std::complex<float> alpha,
     const std::complex<float>* A, int lda,
     const std::complex<float>* B, int incb,
     std::complex<float> beta, std::complex<float> *C, int incc) {
      STRUMPACK_FLOPS(4*blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(2*4*blas::gemv_moves(m,n));
      return cublasCgemv
        (handle, transa, m, n,
         reinterpret_cast<cuComplex*>(&alpha),
         reinterpret_cast<const cuComplex*>(A), lda,
         reinterpret_cast<const cuComplex*>(B), incb,
         reinterpret_cast<cuComplex*>(&beta),
         reinterpret_cast<cuComplex*>(C), incc);
    }
    inline cublasStatus_t cublasgemv
    (cublasHandle_t handle, cublasOperation_t transa,
     int m, int n, std::complex<double> alpha,
     const std::complex<double> *A, int lda,
     const std::complex<double> *B, int incb, std::complex<double> beta,
     std::complex<double> *C, int incc) {
      STRUMPACK_FLOPS(4*blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(2*8*blas::gemv_moves(m,n));
      return cublasZgemv
        (handle, transa, m, n,
         reinterpret_cast<cuDoubleComplex*>(&alpha),
         reinterpret_cast<const cuDoubleComplex*>(A), lda,
         reinterpret_cast<const cuDoubleComplex*>(B), incb,
         reinterpret_cast<cuDoubleComplex*>(&beta),
         reinterpret_cast<cuDoubleComplex*>(C), incc);
    }


    inline cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* Lwork) {
      return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
    }
    inline cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, double *A, int lda,
     int* Lwork) {
      return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
    }
    inline cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, std::complex<float>* A, int lda,
     int *Lwork) {
      return cusolverDnCgetrf_bufferSize
        (handle, m, n, reinterpret_cast<cuComplex*>(A), lda, Lwork);
    }
    inline cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, std::complex<double>* A, int lda,
     int *Lwork) {
      return cusolverDnZgetrf_bufferSize
        (handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, Lwork);
    }


    inline cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, float* A, int lda,
     float* Workspace, int* devIpiv, int* devInfo) {
      STRUMPACK_FLOPS(blas::getrf_flops(m,n));
      return cusolverDnSgetrf
        (handle, m, n, A, lda, Workspace, devIpiv, devInfo);
    }
    inline cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, double* A, int lda,
     double* Workspace, int* devIpiv, int* devInfo) {
      STRUMPACK_FLOPS(blas::getrf_flops(m,n));
      return cusolverDnDgetrf
        (handle, m, n, A, lda, Workspace, devIpiv, devInfo);
    }
    inline cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, std::complex<float>* A, int lda,
     std::complex<float>* Workspace, int* devIpiv, int* devInfo) {
      STRUMPACK_FLOPS(4*blas::getrf_flops(m,n));
      return cusolverDnCgetrf
        (handle, m, n, reinterpret_cast<cuComplex*>(A), lda,
         reinterpret_cast<cuComplex*>(Workspace), devIpiv, devInfo);
    }
    inline cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, std::complex<double>* A, int lda,
     std::complex<double>* Workspace, int* devIpiv, int* devInfo) {
      STRUMPACK_FLOPS(4*blas::getrf_flops(m,n));
      return cusolverDnZgetrf
        (handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda,
         reinterpret_cast<cuDoubleComplex*>(Workspace), devIpiv, devInfo);
    }


    inline cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
     const float* A, int lda, const int* devIpiv, float* B, int ldb,
     int* devInfo) {
      STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
      return cusolverDnSgetrs
        (handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
    }
    inline cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
     const double* A, int lda, const int* devIpiv, double* B, int ldb,
     int* devInfo) {
      STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
      return cusolverDnDgetrs
        (handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
    }
    inline cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
     const std::complex<float>* A, int lda, const int* devIpiv,
     std::complex<float>* B, int ldb, int* devInfo) {
      STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
      return cusolverDnCgetrs
        (handle, trans, n, nrhs, reinterpret_cast<const cuComplex*>(A), lda,
         devIpiv, reinterpret_cast<cuComplex*>(B), ldb, devInfo);
    }
    inline cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
     const std::complex<double>* A, int lda, const int* devIpiv,
     std::complex<double>* B, int ldb, int *devInfo) {
      STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
      return cusolverDnZgetrs
        (handle, trans, n, nrhs,
         reinterpret_cast<const cuDoubleComplex*>(A), lda, devIpiv,
         reinterpret_cast<cuDoubleComplex*>(B), ldb, devInfo);
    }

  } // end namespace cuda
} // end namespace strumpack

#endif // STRUMPACK_CUDA_WRAPPER_HPP
