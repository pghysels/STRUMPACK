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
#if defined(STRUMPACK_USE_MAGMA)
#include "MAGMAWrapper.hpp"
#endif
#if defined(STRUMPACK_USE_MPI)
#include "misc/MPIWrapper.hpp"
#endif

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
    /*hipsolverEigMode_t E2hipOp(Jobz op) {
      switch (op) {
      case Jobz::N: return HIPSOLVER_EIG_MODE_NOVECTOR;
      case Jobz::V: return HIPSOLVER_EIG_MODE_VECTOR;
      default:
        assert(false);
        return HIPSOLVER_EIG_MODE_VECTOR;
      }
    }*/

    void hip_assert(hipError_t code, const char *file, int line,
                     bool abort) {
      if (code != hipSuccess) {
        std::cerr << "HIP assertion failed: "
                  << hipGetErrorString(code) << " "
                  <<  file << " " << line << std::endl;
        if (abort) exit(code);
      }
    }

    void hip_assert(rocblas_status code, const char *file, int line,
                    bool abort) {
      if (code != rocblas_status_success) {
        std::cerr << "rocsolver/rocblas assertion failed: " << code << " "
                  <<  file << " " << line << std::endl;
        if (abort) exit(code);
      }
    }

    void hip_assert(hipblasStatus_t code, const char *file, int line,
                     bool abort) {
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
      gpu_check(hipFree(0));
      gpu::BLASHandle hb;
      gpu::SOLVERHandle hs;
#if defined(STRUMPACK_USE_MAGMA)
      gpu::magma::MAGMAQueue mq;
#endif
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

    // it seems like rocsolver doesn't need work memory?
    template<typename scalar_t> int
    getrf_buffersize(SOLVERHandle& handle, int n) {
      return 0;
    }
    template int getrf_buffersize<float>(SOLVERHandle&, int);
    template int getrf_buffersize<double>(SOLVERHandle&, int);
    template int getrf_buffersize<std::complex<float>>(SOLVERHandle&, int);
    template int getrf_buffersize<std::complex<double>>(SOLVERHandle&, int);


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

    void getrs(SOLVERHandle& handle, rocblas_operation trans,
               int n, int nrhs, const float* A, int lda,
               const int* devIpiv, float* B, int ldb, int* devInfo) {
      STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
      gpu_check(rocsolver_sgetrs
                (handle, trans, n, nrhs,
                 const_cast<float*>(A), lda, devIpiv, B, ldb));
    }
    void getrs(SOLVERHandle& handle, rocblas_operation trans,
               int n, int nrhs, const double* A, int lda,
               const int* devIpiv, double* B, int ldb,
               int* devInfo) {
      STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
      gpu_check(rocsolver_dgetrs
                (handle, trans, n, nrhs,
                 const_cast<double*>(A), lda, devIpiv, B, ldb));
    }
    void getrs(SOLVERHandle& handle, rocblas_operation trans,
               int n, int nrhs, const std::complex<float>* A, int lda,
               const int* devIpiv, std::complex<float>* B, int ldb,
               int* devInfo) {
      STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
      gpu_check(rocsolver_cgetrs
                (handle, trans, n, nrhs,
                 reinterpret_cast<rocblas_float_complex*>
                 (const_cast<std::complex<float>*>(A)), lda, devIpiv,
                 reinterpret_cast<rocblas_float_complex*>
                 (const_cast<std::complex<float>*>(B)), ldb));
    }
    void getrs(SOLVERHandle& handle, rocblas_operation trans,
               int n, int nrhs, const std::complex<double>* A, int lda,
               const int* devIpiv, std::complex<double>* B, int ldb,
               int *devInfo) {
      STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
      gpu_check(rocsolver_zgetrs
                (handle, trans, n, nrhs,
                 reinterpret_cast<rocblas_double_complex*>
                 (const_cast<std::complex<double>*>(A)), lda, devIpiv,
                 reinterpret_cast<rocblas_double_complex*>
                 (const_cast<std::complex<double>*>(B)), ldb));
    }

    template<typename scalar_t> void
    getrs(SOLVERHandle& handle, Trans trans,
          const DenseMatrix<scalar_t>& A, const int* devIpiv,
          DenseMatrix<scalar_t>& B, int *devInfo) {
      getrs(handle, T2rocOp(trans), A.rows(), B.cols(), A.data(), A.ld(),
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

    void trsm(BLASHandle& handle, hipblasSideMode_t side,
              hipblasFillMode_t uplo, hipblasOperation_t trans,
              hipblasDiagType_t diag, int m, int n, const float* alpha,
              float* A, int lda, float* B, int ldb) {
      STRUMPACK_FLOPS(blas::trsm_flops(m, n, alpha, side));
      STRUMPACK_BYTES(4*blas::trsm_moves(m, n));
      gpu_check(hipblasStrsm(handle, side, uplo, trans, diag, m,
                             n, alpha, A, lda, B, ldb));
    }
    void trsm(BLASHandle& handle, hipblasSideMode_t side,
              hipblasFillMode_t uplo, hipblasOperation_t trans,
              hipblasDiagType_t diag, int m, int n, const double* alpha,
              double* A, int lda, double* B, int ldb) {
      STRUMPACK_FLOPS(blas::trsm_flops(m, n, alpha, side));
      STRUMPACK_BYTES(8*blas::trsm_moves(m, n));
      gpu_check(hipblasDtrsm(handle, side, uplo, trans, diag, m,
                             n, alpha, A, lda, B, ldb));
    }
    void trsm(BLASHandle& handle, hipblasSideMode_t side,
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
    void trsm(BLASHandle& handle, hipblasSideMode_t side,
              hipblasFillMode_t uplo, hipblasOperation_t trans,
              hipblasDiagType_t diag, int m, int n,
              const std::complex<double>* alpha,
              std::complex<double>* A,
              int lda, std::complex<double>* B, int ldb) {
      STRUMPACK_FLOPS(4*blas::trsm_flops(m, n, alpha, side));
      STRUMPACK_BYTES(2*8*blas::trsm_moves(m, n));
      gpu_check(hipblasZtrsm(handle, side, uplo, trans, diag, m, n,
                             reinterpret_cast<const hipblasDoubleComplex*>(alpha), 
                             reinterpret_cast<hipblasDoubleComplex*>(A), lda,
                             reinterpret_cast<hipblasDoubleComplex*>(B), ldb));
    }
    template<typename scalar_t> void
    trsm(BLASHandle& handle, Side side, UpLo uplo,
         Trans trans, Diag diag, const scalar_t alpha,
          DenseMatrix<scalar_t>& A, DenseMatrix<scalar_t>& B) {
      trsm(handle, S2hipOp(side), U2hipOp(uplo), T2hipOp(trans), D2hipOp(diag), 
           B.rows(), B.cols(), &alpha, A.data(), A.ld(), B.data(), B.ld());
    }

    template void trsm(BLASHandle&, Side, UpLo, Trans, Diag, const float,
                       DenseMatrix<float>&, DenseMatrix<float>&);
    template void trsm(BLASHandle&, Side, UpLo, Trans, Diag, const double,
                       DenseMatrix<double>&, DenseMatrix<double>&);
    template void trsm(BLASHandle&, Side, UpLo, Trans, Diag,
                       const std::complex<float>,
                       DenseMatrix<std::complex<float>>&,
                       DenseMatrix<std::complex<float>>&);
    template void trsm(BLASHandle&, Side, UpLo, Trans, Diag,
                       const std::complex<double>,
                       DenseMatrix<std::complex<double>>&,
                       DenseMatrix<std::complex<double>>&);

    /*void gesvdj_info_create(hipsolverGesvdjInfo_t *info) {
      gpu_check(hipsolverDnCreateGesvdjInfo(info));
    }

    void gesvdj_buffersize
    (SOLVERHandle& handle, hipsolverEigMode_t jobz, int econ, int m, int n,
     const float* A, int lda, const float* S, const float* U,
     int ldu, const float* V, int ldv, int* lwork,
     hipsolverGesvdjInfo_t params) {
      gpu_check(hipsolverDnSgesvdj_bufferSize
                (handle, jobz, econ, m, n, A, lda, S,
                 U, ldu, V, ldv, lwork, params));
    }

    void gesvdj_buffersize
    (SOLVERHandle& handle, hipsolverEigMode_t jobz, int econ, int m, int n,
     const double* A, int lda, const double* S, const double* U,
     int ldu, const double* V, int ldv, int* lwork,
     hipsolverGesvdjInfo_t params) {
      gpu_check(hipsolverDnDgesvdj_bufferSize
                (handle, jobz, econ, m, n, A, lda, S,
                 U, ldu, V, ldv, lwork, params));
    }

    void gesvdj_buffersize
    (SOLVERHandle& handle, hipsolverEigMode_t jobz, int econ, int m, int n,
     const std::complex<float>* A, int lda, const float* S,
     const std::complex<float>* U, int ldu, const std::complex<float>* V,
     int ldv, int* lwork, hipsolverGesvdjInfo_t params) {
      gpu_check(hipsolverDnCgesvdj_bufferSize
                (handle, jobz, econ, m, n,
                 reinterpret_cast<const hipFloatComplex*>(A), lda, S,
                 reinterpret_cast<const hipFloatComplex*>(U), ldu,
                 reinterpret_cast<const hipFloatComplex*>(V), ldv,
                 lwork, params));
    }

    void gesvdj_buffersize
    (SOLVERHandle& handle, hipsolverEigMode_t jobz, int econ, int m, int n,
     const std::complex<double>* A, int lda, const double* S,
     const std::complex<double>* U, int ldu, const std::complex<double>* V,
     int ldv, int* lwork, hipsolverGesvdjInfo_t params) {
      gpu_check(hipsolverDnZgesvdj_bufferSize
                (handle, jobz, econ, m, n,
                 reinterpret_cast<const hipDoubleComplex*>(A), lda, S,
                 reinterpret_cast<const hipDoubleComplex*>(U), ldu,
                 reinterpret_cast<const hipDoubleComplex*>(V), ldv,
                 lwork, params));
    }

    template<typename scalar_t, typename real_t> int gesvdj_buffersize
    (SOLVERHandle& handle, Jobz jobz, int m, int n, real_t* S,
     hipsolverGesvdjInfo_t params) {
      int econ = 1;
      int Lwork;
      gesvdj_info_create(&params);
      gesvdj_buffersize
        (handle, E2hipOp(jobz), econ, m, n, static_cast<scalar_t*>(nullptr), n,
         S, static_cast<scalar_t*>(nullptr), m,
         static_cast<scalar_t*>(nullptr), n, &Lwork, params);
      return Lwork;
    }

    template int gesvdj_buffersize<float, float>
      (SOLVERHandle&, Jobz, int, int, float*, hipsolverGesvdjInfo_t);

    template int gesvdj_buffersize<double, double>
      (SOLVERHandle&, Jobz, int, int, double*, hipsolverGesvdjInfo_t);

    template int gesvdj_buffersize<std::complex<float>, float>
      (SOLVERHandle&, Jobz, int, int, float*, hipsolverGesvdjInfo_t);

    template int gesvdj_buffersize<std::complex<double>, double>
      (SOLVERHandle&, Jobz, int, int, double*, hipsolverGesvdjInfo_t);

    void gesvdj(SOLVERHandle& handle, hipsolverEigMode_t jobz, int econ,
               int m, int n, float* A, int lda, float* S, float* U, int ldu,
               float* V, int ldv, float* Workspace, int lwork, int* info,
               hipsolverGesvdjInfo_t params) {
      STRUMPACK_FLOPS(blas::gesvd_flops(m,n));
      gpu_check(hipsolverDnSsgesvdj
                (handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                 Workspace, lwork, info, params));
    }

    void gesvdj(SOLVERHandle& handle, hipsolverEigMode_t jobz, int econ,
               int m, int n, double* A, int lda, double* S, double* U, int ldu,
               double* V, int ldv, float* Workspace, int lwork, int* info,
               hipsolverGesvdjInfo_t params) {
      STRUMPACK_FLOPS(blas::gesvd_flops(m,n));
      gpu_check(hipsolverDnDsgesvdj
                (handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                 Workspace, lwork, info, params));
    }

    void gesvdj(SOLVERHandle& handle, hipsolverEigMode_t jobz, int econ,
               int m, int n, std::complex<float>* A, int lda, float* S,
               std::complex<float>* U, int ldu, std::complex<float>* V,
               int ldv, std::complex<float>* Workspace, int lwork,
               int* info, hipsolverGesvdjInfo_t params) {
      STRUMPACK_FLOPS(4*blas::gesvd_flops(m,n));
      gpu_check(hipsolverDnCgesvdj
                (handle, jobz, econ, m, n, reinterpret_cast<hipFloatComplex*>(A),
                 lda, S, reinterpret_cast<hipFloatComplex*>(U), ldu,
                 reinterpret_cast<hipFloatComplex*>(V), ldv,
                 reinterpret_cast<hipFloatComplex*>(Workspace), lwork, info, params));
    }

    void gesvdj(SOLVERHandle& handle, hipsolverEigMode_t jobz, int econ,
               int m, int n, std::complex<double>* A, int lda, double* S,
               std::complex<double>* U, int ldu, std::complex<double>* V,
               int ldv, std::complex<double>* Workspace, int lwork,
               int* info, hipsolverGesvdjInfo_t params) {
      STRUMPACK_FLOPS(4*blas::gesvd_flops(m,n));
      gpu_check(hipsolverDnZgesvdj
                (handle, jobz, econ, m, n, reinterpret_cast<hipDoubleComplex*>(A),
                 lda, S, reinterpret_cast<hipDoubleComplex*>(U), ldu,
                 reinterpret_cast<hipDoubleComplex*>(V), ldv,
                 reinterpret_cast<hipDoubleComplex*>(Workspace), lwork, info, params));
    }

    template<typename scalar_t, typename real_t> void
    gesvdj(SOLVERHandle& handle, Jobz jobz, DenseMatrix<scalar_t>& A, real_t* d_S,
          DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V,
          scalar_t* Workspace, int Lwork, int* devInfo, hipsolverGesvdjInfo_t params) {
      int econ = 1;
      gesvdj(handle, E2hipOp(jobz), econ, A.rows(), A.cols(), A.data(), A.ld(), d_S,
             U.data(), A.ld(), V.data(), A.cols(), Workspace, Lwork, devInfo,
             params);
    }

    template void gesvdj(SOLVERHandle&, Jobz, DenseMatrix<float>&, float*,
                         DenseMatrix<float>&, DenseMatrix<float>&,
                         float*, int, int*, hipsolverGesvdjInfo_t);

    template void gesvdj(SOLVERHandle&, Jobz, DenseMatrix<double>&, double*,
                         DenseMatrix<double>&, DenseMatrix<double>&,
                         double*, int, int*, hipsolverGesvdjInfo_t);

    template void gesvdj(SOLVERHandle&, Jobz, DenseMatrix<std::complex<float>>&,
                         float*, DenseMatrix<std::complex<float>>&,
                         DenseMatrix<std::complex<float>>&,
                         std::complex<float>*, int, int*, hipsolverGesvdjInfo_t);

    template void gesvdj(SOLVERHandle&, Jobz, DenseMatrix<std::complex<double>>&,
                         double*, DenseMatrix<std::complex<double>>&,
                         DenseMatrix<std::complex<double>>&,
                         std::complex<double>*, int, int*, hipsolverGesvdjInfo_t);

    template<typename scalar_t, typename real_t> void
    gesvd(SOLVERHandle& handle, Jobz jobz, int m, int n, real_t* S,
          DenseMatrix<scalar_t>& A, DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V,
          int* devInfo, const double tol) {
      hipsolverGesvdjInfo_t params = nullptr;
      hipsolverDnCreateGesvdjInfo(&params);
      hipsolverDnXgesvdjSetTolerance(params, tol);
      int gesvd_work_size = gesvdj_buffersize<scalar_t>
        (handle, jobz, m, n, S, params);
      DeviceMemory<scalar_t> gesvd_work(gesvd_work_size);
      DeviceMemory<scalar_t> dmemA(A.rows()*A.cols());
      DenseMatrixWrapper<scalar_t> d_A(A.rows(), A.cols(), dmemA, A.rows());
      copy_device_to_device(d_A, A);
      gesvdj<scalar_t>(handle, jobz, d_A, S, U, V, gesvd_work,
                       gesvd_work_size, devInfo, params);
    }
    */

    void gesvd(SOLVERHandle& handle, rocblas_svect left, rocblas_svect right,
               int m, int n, float* A, int lda, float* S, float* U, int ldu,
               float* V, int ldv, float* E, rocblas_workmode alg, int* info) {
      STRUMPACK_FLOPS(blas::gesvd_flops(m,n));
      gpu_check(rocsolver_sgesvd
                (handle, left, right, m, n, A, lda, S, U, ldu, V, ldv,
                 E, alg, info));
    }
    void gesvd(SOLVERHandle& handle, rocblas_svect left, rocblas_svect right,
               int m, int n, double* A, int lda, double* S, double* U, int ldu,
               double* V, int ldv, double* E, rocblas_workmode alg, int* info) {
      STRUMPACK_FLOPS(blas::gesvd_flops(m,n));
      gpu_check(rocsolver_dgesvd
                (handle, left, right, m, n, A, lda, S, U, ldu, V, ldv,
                 E, alg, info));
    }
    void gesvd(SOLVERHandle& handle, rocblas_svect left, rocblas_svect right,
               int m, int n, std::complex<float>* A, int lda, float* S,
               std::complex<float>* U, int ldu, std::complex<float>* V,
               int ldv, float* E, rocblas_workmode alg, int* info) {
      STRUMPACK_FLOPS(4*blas::gesvd_flops(m,n));
      gpu_check(rocsolver_cgesvd
                (handle, left, right, m, n,
                 reinterpret_cast<rocblas_float_complex*>(A), lda, S,
                 reinterpret_cast<rocblas_float_complex*>(U), ldu,
                 reinterpret_cast<rocblas_float_complex*>(V), ldv,
                 E, alg, info));
    }
    void gesvd(SOLVERHandle& handle, rocblas_svect left, rocblas_svect right,
               int m, int n, std::complex<double>* A, int lda, double* S,
               std::complex<double>* U, int ldu, std::complex<double>* V,
               int ldv, double* E, rocblas_workmode alg, int* info) {
      STRUMPACK_FLOPS(4*blas::gesvd_flops(m,n));
      gpu_check(rocsolver_zgesvd
                (handle, left, right, m, n,
                 reinterpret_cast<rocblas_double_complex*>(A), lda, S,
                 reinterpret_cast<rocblas_double_complex*>(U), ldu,
                 reinterpret_cast<rocblas_double_complex*>(V), ldv,
                 E, alg, info));
    }

    template<typename scalar_t, typename real_t> void
    gesvd(SOLVERHandle& handle, DenseMatrix<scalar_t>& A, real_t* d_S,
          DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V,
          real_t* E, int* devInfo) {
      rocblas_svect left = rocblas_svect_singular;
      rocblas_svect right = rocblas_svect_singular;
      rocblas_workmode alg = rocblas_outofplace;
      gesvd(handle, left, right, A.rows(), A.cols(), A.data(), A.ld(), d_S,
             U.data(), U.ld(), V.data(), V.ld(), E, alg, devInfo);
    }
    template void gesvd(SOLVERHandle&, DenseMatrix<float>&, float*,
                        DenseMatrix<float>&, DenseMatrix<float>&,
                        float*, int*);
    template void gesvd(SOLVERHandle&, DenseMatrix<double>&, double*,
                        DenseMatrix<double>&, DenseMatrix<double>&,
                        double*, int*);
    template void gesvd(SOLVERHandle&, DenseMatrix<std::complex<float>>&,
                        float*, DenseMatrix<std::complex<float>>&,
                        DenseMatrix<std::complex<float>>&, float*, int*);
    template void gesvd(SOLVERHandle&, DenseMatrix<std::complex<double>>&,
                        double*, DenseMatrix<std::complex<double>>&,
                        DenseMatrix<std::complex<double>>&, double*, int*);

    template<typename scalar_t, typename real_t> void
    gesvd_hip(SOLVERHandle& handle, real_t* S, DenseMatrix<scalar_t>& A,
              DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V,
              int* devInfo) {
      DeviceMemory<scalar_t> dmemA(A.rows()*A.cols());
      DenseMatrixWrapper<scalar_t> d_A(A.rows(), A.cols(), dmemA, A.rows());
      copy_device_to_device(d_A, A);
      DeviceMemory<real_t> dE(A.rows()-1);
      real_t* E = dE;
      gesvd<scalar_t>(handle, d_A, S, U, V, E, devInfo);
    }

    template void gesvd_hip(SOLVERHandle&, float*, DenseMatrix<float>&,
                            DenseMatrix<float>&, DenseMatrix<float>&,
                            int*);
    template void gesvd_hip(SOLVERHandle&, double*, DenseMatrix<double>&,
                            DenseMatrix<double>&, DenseMatrix<double>&,
                            int*);
    template void gesvd_hip(SOLVERHandle&, float*,
                            DenseMatrix<std::complex<float>>&,
                            DenseMatrix<std::complex<float>>&,
                            DenseMatrix<std::complex<float>>&,
                            int*);
    template void gesvd_hip(SOLVERHandle&, double*,
                            DenseMatrix<std::complex<double>>&,
                            DenseMatrix<std::complex<double>>&,
                            DenseMatrix<std::complex<double>>&,
                            int*);

    void geam(BLASHandle& handle, hipblasOperation_t transa,
              hipblasOperation_t transb, int m, int n,
              const float* alpha, const float* A, int lda,
              const float* beta, const float* B, int ldb, float* C, int ldc) {
      STRUMPACK_FLOPS(blas::geam_flops(m, n, alpha, beta));
      gpu_check(hipblasSgeam(handle, transa, transb, m, n, alpha,
                             A, lda, beta, B, ldb, C, ldc));
    }
    void geam(BLASHandle& handle, hipblasOperation_t transa,
              hipblasOperation_t transb, int m, int n,
              const double* alpha, const double* A, int lda,
              const double* beta, const double* B, int ldb,
              double* C, int ldc) {
      STRUMPACK_FLOPS(blas::geam_flops(m, n, alpha, beta));
      gpu_check(hipblasDgeam(handle, transa, transb, m, n, alpha,
                             A, lda, beta, B, ldb, C, ldc));
    }
    void geam(BLASHandle& handle, hipblasOperation_t transa,
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
    void geam(BLASHandle& handle, hipblasOperation_t transa,
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
    geam(BLASHandle& handle, Trans transa, Trans transb, const scalar_t alpha,
         const DenseMatrix<scalar_t>& A, const scalar_t beta,
         const DenseMatrix<scalar_t>& B, DenseMatrix<scalar_t>& C){
      geam(handle, T2hipOp(transa), T2hipOp(transb),
           C.rows(), C.cols(), &alpha, A.data(), A.ld(), &beta,
           B.data(), B.ld(), C.data(), C.ld());
    }

    template void geam(BLASHandle&, Trans, Trans, const float,
                       const DenseMatrix<float>&, const float,
                       const DenseMatrix<float>&, DenseMatrix<float>&);
    template void geam(BLASHandle&, Trans, Trans, const double,
                       const DenseMatrix<double>&, const double,
                       const DenseMatrix<double>&, DenseMatrix<double>&);
    template void geam(BLASHandle&, Trans, Trans, const std::complex<float>,
                       const DenseMatrix<std::complex<float>>&,
                       const std::complex<float>,
                       const DenseMatrix<std::complex<float>>&,
                       DenseMatrix<std::complex<float>>&);
    template void geam(BLASHandle&, Trans, Trans, const std::complex<double>,
                       const DenseMatrix<std::complex<double>>&,
                       const std::complex<double>,
                       const DenseMatrix<std::complex<double>>&,
                       DenseMatrix<std::complex<double>>&);

    void dgmm(BLASHandle& handle, hipblasSideMode_t side, int m, int n,
              const float* A, int lda, const float* x, int incx,
              float* C, int ldc) {
      STRUMPACK_FLOPS(blas::dgmm_flops(m, n));
      gpu_check(hipblasSdgmm(handle, side, m, n, A, lda, x, incx, C, ldc));
    }
    void dgmm(BLASHandle& handle, hipblasSideMode_t side, int m, int n,
              const double* A, int lda, const double* x, int incx,
              double* C, int ldc) {
      STRUMPACK_FLOPS(blas::dgmm_flops(m, n));
      gpu_check(hipblasDdgmm(handle, side, m, n, A, lda, x, incx, C, ldc));
    }
    void dgmm(BLASHandle& handle, hipblasSideMode_t side, int m, int n,
              const std::complex<float>* A, int lda,
              const std::complex<float>* x, int incx,
              std::complex<float>* C, int ldc) {
      STRUMPACK_FLOPS(4*blas::dgmm_flops(m, n));
      gpu_check
        (hipblasCdgmm(handle, side, m, n,
                      reinterpret_cast<const hipblasComplex*>(A), lda,
                      reinterpret_cast<const hipblasComplex*>(x), incx,
                      reinterpret_cast<hipblasComplex*>(C), ldc));
    }
    void dgmm(BLASHandle& handle, hipblasSideMode_t side, int m, int n,
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
    dgmm(BLASHandle& handle, Side side, const DenseMatrix<scalar_t>& A,
         const scalar_t* x, DenseMatrix<scalar_t>& C){
      int incx = 1;
      dgmm(handle, S2hipOp(side), A.rows(), A.cols(), A.data(), A.ld(), x, incx,
           C.data(), C.ld());
    }

    template void dgmm(BLASHandle&, Side, const DenseMatrix<float>&, 
                       const float*, DenseMatrix<float>&);

    template void dgmm(BLASHandle&, Side, const DenseMatrix<double>&, 
                       const double*, DenseMatrix<double>&);

    template void dgmm(BLASHandle&, Side, const DenseMatrix<std::complex<float>>&, 
                       const std::complex<float>*, DenseMatrix<std::complex<float>>&);

    template void dgmm(BLASHandle&, Side, const DenseMatrix<std::complex<double>>&, 
                       const std::complex<double>*, DenseMatrix<std::complex<double>>&);
    
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
    template<typename scalar_t> void
    gemv(BLASHandle& handle, Trans ta,
         scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const DenseMatrix<scalar_t>& x, scalar_t beta,
         DenseMatrix<scalar_t>& y) {
      gemv(handle, T2hipOp(ta), a.rows(), a.cols(), alpha, a.data(), a.ld(),
           x.data(), 1, beta, y.data(), 1);
    }
    template void gemv(BLASHandle&, Trans, float,
                       const DenseMatrix<float>&, const DenseMatrix<float>&,
                       float, DenseMatrix<float>&);
    template void gemv(BLASHandle&, Trans, double,
                       const DenseMatrix<double>&, const DenseMatrix<double>&,
                       double, DenseMatrix<double>&);
    template void gemv(BLASHandle&, Trans, std::complex<float>,
                       const DenseMatrix<std::complex<float>>&,
                       const DenseMatrix<std::complex<float>>&,
                       std::complex<float>,
                       DenseMatrix<std::complex<float>>&);
    template void gemv(BLASHandle&, Trans, std::complex<double>,
                       const DenseMatrix<std::complex<double>>&,
                       const DenseMatrix<std::complex<double>>&,
                       std::complex<double>,
                       DenseMatrix<std::complex<double>>&);

    void laswp(SOLVERHandle& handle, DenseMatrix<float>& A,
               int k1, int k2, int* ipiv, int inc) {
      STRUMPACK_BYTES(4*blas::laswp_moves(A.cols(), 1, A.rows()));
      gpu_check(rocsolver_slaswp
                (handle, A.cols(), A.data(), A.ld(),
                 k1, k2, ipiv, inc));
    }

    void laswp(SOLVERHandle& handle, DenseMatrix<double>& A,
               int k1, int k2, int* ipiv, int inc) {
      STRUMPACK_BYTES(8*blas::laswp_moves(A.cols(), 1, A.rows()));
      gpu_check(rocsolver_dlaswp
                (handle, A.cols(), A.data(), A.ld(),
                 k1, k2, ipiv, inc));
    }

    void laswp(SOLVERHandle& handle, DenseMatrix<std::complex<float>>& A,
               int k1, int k2, int* ipiv, int inc) {
      STRUMPACK_BYTES(2*4*blas::laswp_moves(A.cols(), 1, A.rows()));
      gpu_check(rocsolver_claswp
                (handle, A.cols(),
                 reinterpret_cast<rocblas_float_complex*>(A.data()),
                 A.ld(), k1, k2, ipiv, inc));
    }

    void laswp(SOLVERHandle& handle, DenseMatrix<std::complex<double>>& A,
               int k1, int k2, int* ipiv, int inc) {
      STRUMPACK_BYTES(2*8*blas::laswp_moves(A.cols(), 1, A.rows()));
      gpu_check(rocsolver_zlaswp
                (handle, A.cols(),
                 reinterpret_cast<rocblas_double_complex*>(A.data()),
                 A.ld(), k1, k2, ipiv, inc));
    }

  } // end namespace gpu
} // end namespace strumpack
