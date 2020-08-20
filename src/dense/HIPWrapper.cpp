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


#include <hip/hip_runtime.h>
#include "HIPWrapper.hpp"

namespace strumpack {
  namespace gpu {

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

    void hip_assert(hipError_t code, const char *file, int line,
                     bool abort) {
      if (code != hipSuccess) {
        std::cerr << "HIP assertion failed: "
                  << hipGetErrorString(code) << " "
                  <<  file << " " << line << std::endl;
        if (abort) exit(code);
      }
    }

#if defined(STRUMPACK_HIP_PLATFORM_NVCC)
    void hip_assert(cusolverStatus_t code, const char *file, int line,
                     bool abort) {
      if (code != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "hipSOLVER assertion failed: " << code << " "
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
        if (abort) exit(code);
      }
    }
#else
    void hip_assert(rocblas_status code, const char *file, int line,
                    bool abort) {
      if (code != rocblas_status_success) {
        std::cerr << "rocsolver/rocblas assertion failed: " << code << " "
                  <<  file << " " << line << std::endl;
        if (abort) exit(code);
      }
    }
#endif

    void hip_assert(hipblasStatus_t code, const char *file, int line,
                     bool abort) {
      if (code != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipBLAS assertion failed: " << code << " "
                  <<  file << " " << line << std::endl;
        if (abort) exit(code);
      }
    }


    void gemm(BLASHandle& handle, hipblasOperation_t transa,
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
    void gemm(BLASHandle& handle, hipblasOperation_t transa,
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
    void gemm(BLASHandle& handle, hipblasOperation_t transa,
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
    void gemm(BLASHandle& handle, hipblasOperation_t transa,
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
    gemm(BLASHandle& handle, Trans ta, Trans tb,
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
      gemm(handle, T2hipOp(ta), T2hipOp(tb), c.rows(), c.cols(),
           (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(), a.ld(),
           b.data(), b.ld(), beta, c.data(), c.ld());
    }
    template void gemm(BLASHandle&, Trans, Trans,
                       float, const DenseMatrix<float>&,
                       const DenseMatrix<float>&, float,
                       DenseMatrix<float>&);
    template void gemm(BLASHandle&, Trans, Trans,
                       double, const DenseMatrix<double>&,
                       const DenseMatrix<double>&, double,
                       DenseMatrix<double>&);
    template void gemm(BLASHandle&, Trans, Trans, std::complex<float>,
                       const DenseMatrix<std::complex<float>>&,
                       const DenseMatrix<std::complex<float>>&,
                       std::complex<float>,
                       DenseMatrix<std::complex<float>>&);
    template void gemm(BLASHandle&, Trans, Trans, std::complex<double>,
                       const DenseMatrix<std::complex<double>>&,
                       const DenseMatrix<std::complex<double>>&,
                       std::complex<double>,
                       DenseMatrix<std::complex<double>>&);

#if defined(STRUMPACK_HIP_PLATFORM_NVCC)
    void getrf_buffersize
    (SOLVERHandle& handle, int m, int n, float* A, int lda, int* Lwork) {
      gpu_check(cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork));
    }
    void getrf_buffersize
    (SOLVERHandle& handle, int m, int n, double *A, int lda,
     int* Lwork) {
      gpu_check(cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork));
    }
    void getrf_buffersize
    (SOLVERHandle& handle, int m, int n, std::complex<float>* A, int lda,
     int *Lwork) {
      gpu_check(cusolverDnCgetrf_bufferSize
                (handle, m, n, reinterpret_cast<hipComplex*>(A), lda, Lwork));
    }
    void getrf_buffersize
    (SOLVERHandle& handle, int m, int n, std::complex<double>* A, int lda,
     int *Lwork) {
      gpu_check(cusolverDnZgetrf_bufferSize
                (handle, m, n,
                 reinterpret_cast<hipblasDoubleComplex*>(A), lda, Lwork));
    }
    template<typename scalar_t> int getrf_buffersize
    (SOLVERHandle& handle, int n) {
      int Lwork;
      getrf_buffersize
        (handle, n, n, static_cast<scalar_t*>(nullptr), n, &Lwork);
      return Lwork;
    }
#else
    // it seems like rocsolver doesn't need work memory?
    template<typename scalar_t> int 
    getrf_buffersize(SOLVERHandle& handle, int n) {
      return 0;
    }
#endif

    template int getrf_buffersize<float>(SOLVERHandle&, int);
    template int getrf_buffersize<double>(SOLVERHandle&, int);
    template int getrf_buffersize<std::complex<float>>(SOLVERHandle&, int);
    template int getrf_buffersize<std::complex<double>>(SOLVERHandle&, int);


#if defined(STRUMPACK_HIP_PLATFORM_NVCC)
    void getrf(SOLVERHandle& handle, int m, int n, float* A, int lda,
               float* Workspace, int* devIpiv, int* devInfo) {
      STRUMPACK_FLOPS(blas::getrf_flops(m,n));
      gpu_check(cusolverDnSgetrf
                (handle, m, n, A, lda, Workspace, devIpiv, devInfo));
    }
    void getrf(SOLVERHandle& handle, int m, int n, double* A,
               int lda, double* Workspace,
               int* devIpiv, int* devInfo) {
      STRUMPACK_FLOPS(blas::getrf_flops(m,n));
      gpu_check(cusolverDnDgetrf
                (handle, m, n, A, lda, Workspace, devIpiv, devInfo));
    }
    void getrf(SOLVERHandle& handle, int m, int n,
               std::complex<float>* A, int lda,
               std::complex<float>* Workspace,
               int* devIpiv, int* devInfo) {
      STRUMPACK_FLOPS(4*blas::getrf_flops(m,n));
      gpu_check(cusolverDnCgetrf
                (handle, m, n, reinterpret_cast<cuComplex*>(A), lda,
                 reinterpret_cast<cuComplex*>(Workspace), devIpiv, devInfo));
    }
    void getrf(SOLVERHandle& handle, int m, int n,
               std::complex<double>* A, int lda,
               std::complex<double>* Workspace,
               int* devIpiv, int* devInfo) {
      STRUMPACK_FLOPS(4*blas::getrf_flops(m,n));
      gpu_check(cusolverDnZgetrf
                (handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda,
                 reinterpret_cast<cuDoubleComplex*>(Workspace),
                 devIpiv, devInfo));
    }
#else
    void getrf(SOLVERHandle& handle, int m, int n, float* A, int lda,
               float* Workspace, int* devIpiv, int* devInfo) {
      STRUMPACK_FLOPS(blas::getrf_flops(m,n));
      gpu_check(rocsolver_sgetrf(handle, m, n, A, lda, devIpiv, devInfo));
    }
    void getrf(SOLVERHandle& handle, int m, int n, double* A,
               int lda, double* Workspace,
               int* devIpiv, int* devInfo) {
      STRUMPACK_FLOPS(blas::getrf_flops(m,n));
      gpu_check(rocsolver_dgetrf(handle, m, n, A, lda, devIpiv, devInfo));
    }
    void getrf(SOLVERHandle& handle, int m, int n,
               std::complex<float>* A, int lda,
               std::complex<float>* Workspace,
               int* devIpiv, int* devInfo) {
      STRUMPACK_FLOPS(4*blas::getrf_flops(m,n));
      gpu_check(rocsolver_cgetrf
		(handle, m, n, reinterpret_cast<rocblas_float_complex*>(A),
		 lda, devIpiv, devInfo));
    }
    void getrf(SOLVERHandle& handle, int m, int n,
               std::complex<double>* A, int lda,
               std::complex<double>* Workspace,
               int* devIpiv, int* devInfo) {
      STRUMPACK_FLOPS(4*blas::getrf_flops(m,n));
      gpu_check(rocsolver_zgetrf
		(handle, m, n, reinterpret_cast<rocblas_double_complex*>(A),
		 lda, devIpiv, devInfo));
    }
#endif

    template<typename scalar_t> void
    getrf(SOLVERHandle& handle, DenseMatrix<scalar_t>& A,
          scalar_t* Workspace, int* devIpiv, int* devInfo) {
      getrf(handle, A.rows(), A.cols(), A.data(), A.ld(),
            Workspace, devIpiv, devInfo);
    }
    template void getrf(SOLVERHandle&, DenseMatrix<float>&,
                        float*, int*, int*);
    template void getrf(SOLVERHandle&, DenseMatrix<double>&,
                        double*, int*, int*);
    template void getrf(SOLVERHandle&, DenseMatrix<std::complex<float>>&,
                        std::complex<float>*, int*, int*);
    template void getrf(SOLVERHandle&, DenseMatrix<std::complex<double>>& A,
                        std::complex<double>*, int*, int*);


#if defined(STRUMPACK_HIP_PLATFORM_NVCC)
    void getrs(SOLVERHandle& handle, hipblasOperation_t trans,
               int n, int nrhs, const float* A, int lda,
               const int* devIpiv, float* B, int ldb, int* devInfo) {
      STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
      gpu_check(cusolverDnSgetrs
                (handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo));
    }
    void getrs(SOLVERHandle& handle, hipblasOperation_t trans,
               int n, int nrhs, const double* A, int lda,
               const int* devIpiv, double* B, int ldb,
               int* devInfo) {
      STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
      gpu_check(cusolverDnDgetrs
                (handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo));
    }
    void getrs(SOLVERHandle& handle, hipblasOperation_t trans,
               int n, int nrhs, const std::complex<float>* A, int lda,
               const int* devIpiv, std::complex<float>* B, int ldb,
               int* devInfo) {
      STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
      gpu_check(cusolverDnCgetrs
                (handle, trans, n, nrhs,
                 reinterpret_cast<const cuComplex*>(A), lda,
                 devIpiv, reinterpret_cast<cuComplex*>(B), ldb, devInfo));
    }
    void getrs(SOLVERHandle& handle, hipblasOperation_t trans,
               int n, int nrhs, const std::complex<double>* A, int lda,
               const int* devIpiv, std::complex<double>* B, int ldb,
               int *devInfo) {
      STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
      gpu_check(cusolverDnZgetrs
                (handle, trans, n, nrhs,
                 reinterpret_cast<const cuDoubleComplex*>(A), lda, devIpiv,
                 reinterpret_cast<cuDoubleComplex*>(B), ldb, devInfo));
    }
#else
    void getrs(SOLVERHandle& handle, hipblasOperation_t trans,
               int n, int nrhs, const float* A, int lda,
               const int* devIpiv, float* B, int ldb, int* devInfo) {
      STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
      // TODO
    }
    void getrs(SOLVERHandle& handle, hipblasOperation_t trans,
               int n, int nrhs, const double* A, int lda,
               const int* devIpiv, double* B, int ldb,
               int* devInfo) {
      STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
      // TODO
    }
    void getrs(SOLVERHandle& handle, hipblasOperation_t trans,
               int n, int nrhs, const std::complex<float>* A, int lda,
               const int* devIpiv, std::complex<float>* B, int ldb,
               int* devInfo) {
      STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
      // TODO
    }
    void getrs(SOLVERHandle& handle, hipblasOperation_t trans,
               int n, int nrhs, const std::complex<double>* A, int lda,
               const int* devIpiv, std::complex<double>* B, int ldb,
               int *devInfo) {
      STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
      // TODO
    }
#endif

    template<typename scalar_t> void
    getrs(SOLVERHandle& handle, Trans trans,
          const DenseMatrix<scalar_t>& A, const int* devIpiv,
          DenseMatrix<scalar_t>& B, int *devInfo) {
      getrs(handle, T2hipOp(trans), A.rows(), B.cols(), A.data(), A.ld(),
            devIpiv, B.data(), B.ld(), devInfo);
    }
    template void getrs(SOLVERHandle&, Trans, const DenseMatrix<float>&,
                        const int*, DenseMatrix<float>&, int*);
    template void getrs(SOLVERHandle&, Trans, const DenseMatrix<double>&,
                        const int*, DenseMatrix<double>&, int*);
    template void getrs(SOLVERHandle&, Trans,
                        const DenseMatrix<std::complex<float>>&, const int*,
                        DenseMatrix<std::complex<float>>&, int*);
    template void getrs(SOLVERHandle&, Trans,
                        const DenseMatrix<std::complex<double>>&, const int*,
                        DenseMatrix<std::complex<double>>&, int*);


    void gemv(BLASHandle& handle, hipblasOperation_t transa,
              int m, int n, float alpha,
              const float* A, int lda, const float* B, int incb,
              float beta, float* C, int incc) {
      STRUMPACK_FLOPS(blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(4*blas::gemv_moves(m,n));
      gpu_check(hipblasSgemv
                (handle, transa, m, n, &alpha,
                 A, lda, B, incb, &beta, C, incc));
    }
    void gemv(BLASHandle& handle, hipblasOperation_t transa,
              int m, int n, double alpha,
              const double* A, int lda, const double* B, int incb,
              double beta, double* C, int incc) {
      STRUMPACK_FLOPS(blas::gemv_flops(m,n,alpha,beta));
      STRUMPACK_BYTES(8*blas::gemv_moves(m,n));
      gpu_check(hipblasDgemv
                (handle, transa, m, n, &alpha,
                 A, lda, B, incb, &beta, C, incc));
    }
    void gemv(BLASHandle& handle, hipblasOperation_t transa,
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
    void gemv(BLASHandle& handle, hipblasOperation_t transa,
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


  } // end namespace gpu
} // end namespace strumpack
