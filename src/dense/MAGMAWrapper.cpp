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
#include "MAGMAWrapper.hpp"
#if defined(STRUMPACK_USE_CUDA)
#include "dense/CUDAWrapper.hpp"
#endif
#if defined(STRUMPACK_USE_HIP)
#include "dense/HIPWrapper.hpp"
#endif

#include <magma_auxiliary.h>
#include <magma_svbatched.h>
#include <magma_dvbatched.h>
#include <magma_cvbatched.h>
#include <magma_zvbatched.h>


namespace strumpack {
  namespace gpu {
    namespace magma {

      magma_int_t get_geqp3_nb(const DenseMatrix<float>& A) {
        return magma_get_sgeqp3_nb(A.rows(), A.cols());
      }
      magma_int_t get_geqp3_nb(const DenseMatrix<double>& A) {
        return magma_get_dgeqp3_nb(A.rows(), A.cols());
      }
      magma_int_t get_geqp3_nb(const DenseMatrix<std::complex<float>>& A) {
        return magma_get_cgeqp3_nb(A.rows(), A.cols());
      }
      magma_int_t get_geqp3_nb(const DenseMatrix<std::complex<double>>& A) {
        return magma_get_zgeqp3_nb(A.rows(), A.cols());
      }

      std::size_t geqp3_scalar_worksize(const DenseMatrix<float>& A) {
        auto N = A.cols();
        std::size_t NB = get_geqp3_nb(A);
        return (N+1)*NB + 2*N;
      }
      std::size_t geqp3_scalar_worksize(const DenseMatrix<double>& A) {
        auto N = A.cols();
        auto NB = get_geqp3_nb(A);
        return (N+1)*NB + 2*N;
      }
      std::size_t geqp3_scalar_worksize(const DenseMatrix<std::complex<float>>& A) {
        auto N = A.cols();
        auto NB = get_geqp3_nb(A);
        return (N+1)*NB;
      }
      std::size_t geqp3_scalar_worksize(const DenseMatrix<std::complex<double>>& A) {
        auto N = A.cols();
        auto NB = get_geqp3_nb(A);
        return (N+1)*NB;
      }

      std::size_t geqp3_real_worksize(const DenseMatrix<float>& A) {
        return 0;
      }
      std::size_t geqp3_real_worksize(const DenseMatrix<double>& A) {
        return 0;
      }
      std::size_t geqp3_real_worksize(const DenseMatrix<std::complex<float>>& A) {
        return 2 * A.cols();
      }
      std::size_t geqp3_real_worksize(const DenseMatrix<std::complex<double>>& A) {
        return 2 * A.cols();
      }

      void geqp3(DenseMatrix<float>& A, magma_int_t* jpvt,
                 float* tau, float* dwork, magma_int_t lwork,
                 float* rwork, magma_int_t* info) {
        magma_sgeqp3_gpu(A.rows(), A.cols(), A.data(), A.ld(), jpvt,
                         tau, dwork, lwork, info);
      }
      void geqp3(DenseMatrix<double>& A, magma_int_t* jpvt,
                 double* tau, double* dwork, magma_int_t lwork,
                 double* rwork, magma_int_t* info) {
        magma_dgeqp3_gpu(A.rows(), A.cols(), A.data(), A.ld(), jpvt,
                         tau, dwork, lwork, info);
      }
      void geqp3(DenseMatrix<std::complex<float>>& A, magma_int_t* jpvt,
                 std::complex<float>* tau, std::complex<float>* dwork,
                 magma_int_t lwork, float* rwork, magma_int_t* info) {
        magma_cgeqp3_gpu(A.rows(), A.cols(),
                         reinterpret_cast<magmaFloatComplex*>(A.data()),
                         A.ld(), jpvt,
                         reinterpret_cast<magmaFloatComplex*>(tau),
                         reinterpret_cast<magmaFloatComplex*>(dwork),
                         lwork, rwork, info);
      }
      void geqp3(DenseMatrix<std::complex<double>>& A, magma_int_t* jpvt,
                 std::complex<double>* tau, std::complex<double>* dwork,
                 magma_int_t lwork, double* rwork, magma_int_t* info) {
        magma_zgeqp3_gpu(A.rows(), A.cols(),
                         reinterpret_cast<magmaDoubleComplex*>(A.data()),
                         A.ld(), jpvt,
                         reinterpret_cast<magmaDoubleComplex*>(tau),
                         reinterpret_cast<magmaDoubleComplex*>(dwork),
                         lwork, rwork, info);
      }

      void xxgqr(magma_int_t m, magma_int_t n, magma_int_t k,
                 DenseMatrix<float>& Q, float* tau, float* dT,
                 magma_int_t nb, int* info) {
        magma_sorgqr_gpu(m, n, k, Q.data(), Q.ld(), tau, dT, nb, info);
      }
      void xxgqr(magma_int_t m, magma_int_t n, magma_int_t k,
                 DenseMatrix<double>& Q, double* tau, double* dT,
                 magma_int_t nb, int* info) {
        magma_dorgqr_gpu(m, n, k, Q.data(), Q.ld(), tau, dT, nb, info);
      }
      void xxgqr(magma_int_t m, magma_int_t n, magma_int_t k,
                 DenseMatrix<std::complex<float>>& Q, std::complex<float>* tau,
                 std::complex<float>* dT, magma_int_t nb, int* info) {
        magma_cungqr_gpu(m, n, k, reinterpret_cast<magmaFloatComplex*>(Q.data()),
                         Q.ld(), reinterpret_cast<magmaFloatComplex*>(tau),
                         reinterpret_cast<magmaFloatComplex*>(dT), nb, info);
      }
      void xxgqr(magma_int_t m, magma_int_t n, magma_int_t k,
                 DenseMatrix<std::complex<double>>& Q, std::complex<double>* tau,
                 std::complex<double>* dT, magma_int_t nb, int* info) {
        magma_zungqr_gpu(m, n, k, reinterpret_cast<magmaDoubleComplex*>(Q.data()),
                         Q.ld(), reinterpret_cast<magmaDoubleComplex*>(tau),
                         reinterpret_cast<magmaDoubleComplex*>(dT), nb, info);
      }

      int getrf(DenseMatrix<float>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        STRUMPACK_FLOPS(blas::getrf_flops(A.rows(),A.cols()));
        //magma_sgetrf_native
        magma_sgetrf_gpu(A.rows(), A.cols(), A.data(), A.ld(), piv.data(), &info);
        gpu::copy_host_to_device(dpiv, piv.data(), A.rows());
        return info;
      }
      int getrf(DenseMatrix<double>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        STRUMPACK_FLOPS(blas::getrf_flops(A.rows(),A.cols()));
        //magma_dgetrf_native
        magma_dgetrf_gpu(A.rows(), A.cols(), A.data(), A.ld(), piv.data(), &info);
        gpu::copy_host_to_device(dpiv, piv.data(), A.rows());
        return info;
      }
      int getrf(DenseMatrix<std::complex<float>>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        STRUMPACK_FLOPS(4*blas::getrf_flops(A.rows(),A.cols()));
        //magma_cgetrf_native
        magma_cgetrf_gpu
          (A.rows(), A.cols(), reinterpret_cast<magmaFloatComplex*>(A.data()),
           A.ld(), piv.data(), &info);
        gpu::copy_host_to_device(dpiv, piv.data(), A.rows());
        return info;
      }
      int getrf(DenseMatrix<std::complex<double>>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        STRUMPACK_FLOPS(4*blas::getrf_flops(A.rows(),A.cols()));
        //magma_zgetrf_native
        magma_zgetrf_gpu
          (A.rows(), A.cols(),
           reinterpret_cast<magmaDoubleComplex*>(A.data()),
           A.ld(), piv.data(), &info);
        gpu::copy_host_to_device(dpiv, piv.data(), A.rows());
        return info;
      }

      int gesvd_nb(DenseMatrix<float>& A) {
        int nb = magma_get_sgesvd_nb(A.rows(), A.cols());
        std::size_t minmn = std::min(A.rows(), A.cols());
        return 2*minmn*minmn + 3*minmn + 2*minmn*nb;
      }
      int gesvd_nb(DenseMatrix<double>& A) {
        int nb = magma_get_dgesvd_nb(A.rows(), A.cols());
        std::size_t minmn = std::min(A.rows(), A.cols());
        return 2*minmn*minmn + 3*minmn + 2*minmn*nb;
      }
      int gesvd_nb(DenseMatrix<std::complex<float>>& A) {
        int nb = magma_get_dgesvd_nb(A.rows(), A.cols());
        std::size_t minmn = std::min(A.rows(), A.cols());
        return 2*minmn*minmn + 3*minmn + 2*minmn*nb;
      }
      int gesvd_nb(DenseMatrix<std::complex<double>>& A) {
        int nb = magma_get_dgesvd_nb(A.rows(), A.cols());
        std::size_t minmn = std::min(A.rows(), A.cols());
        return 2*minmn*minmn + 3*minmn + 2*minmn*nb;
      }

      void gesvd(magma_vec_t jobu, magma_vec_t jobvt,
                 DenseMatrix<float>& A, float* S,
                 DenseMatrix<float>& U, DenseMatrix<float>& V,
                 float* Workspace, int lwork, float* E, int *info) {
        STRUMPACK_FLOPS(blas::gesvd_flops(A.rows(), A.cols()));
        magma_sgesvd(jobu, jobvt, A.rows(), A.cols(), A.data(), A.ld(), S,
                     U.data(), U.ld(), V.data(), V.ld(),
                     Workspace, lwork, info);
      }
      void gesvd(magma_vec_t jobu, magma_vec_t jobvt,
                 DenseMatrix<double>& A, double* S,
                 DenseMatrix<double>& U, DenseMatrix<double>& V,
                 double* Workspace, int lwork, double* E, int* info) {
        STRUMPACK_FLOPS(blas::gesvd_flops(A.rows(), A.cols()));
        magma_dgesvd(jobu, jobvt, A.rows(), A.cols(), A.data(), A.ld(), S,
                     U.data(), U.ld(), V.data(), V.ld(),
                     Workspace, lwork, info);
      }
      void gesvd(magma_vec_t jobu, magma_vec_t jobvt,
                 DenseMatrix<std::complex<float>>& A,
                 float* S, DenseMatrix<std::complex<float>>& U,
                 DenseMatrix<std::complex<float>>& V,
                 std::complex<float>* Workspace,
                 int lwork, float* E, int *info) {
        STRUMPACK_FLOPS(4*blas::gesvd_flops(A.rows(), A.cols()));
        magma_cgesvd(jobu, jobvt, A.rows(), A.cols(),
                     reinterpret_cast<magmaFloatComplex*>(A.data()), A.ld(),
                     S,
                     reinterpret_cast<magmaFloatComplex*>(U.data()), U.ld(),
                     reinterpret_cast<magmaFloatComplex*>(V.data()), V.ld(),
                     reinterpret_cast<magmaFloatComplex*>(Workspace),
                     lwork, E, info);
      }
      void gesvd(magma_vec_t jobu, magma_vec_t jobvt,
                 DenseMatrix<std::complex<double>>& A, double* S,
                 DenseMatrix<std::complex<double>>& U,
                 DenseMatrix<std::complex<double>>& V,
                 std::complex<double>* Workspace,
                 int lwork, double* E, int* info){
        STRUMPACK_FLOPS(4*blas::gesvd_flops(A.rows(), A.cols()));
        magma_zgesvd(jobu, jobvt, A.rows(), A.cols(),
                     reinterpret_cast<magmaDoubleComplex*>(A.data()), A.ld(),
                     S,
                     reinterpret_cast<magmaDoubleComplex*>(U.data()), U.ld(),
                     reinterpret_cast<magmaDoubleComplex*>(V.data()), V.ld(),
                     reinterpret_cast<magmaDoubleComplex*>(Workspace),
                     lwork, E, info);
      }

      template<typename scalar_t, typename real_t> void
      gesvd_magma(magma_vec_t jobu, magma_vec_t jobvt, real_t* S,
                  DenseMatrix<scalar_t>& A, DenseMatrix<scalar_t>& U,
                  DenseMatrix<scalar_t>& V) {
        int info = 0;
        std::size_t minmn = std::min(U.rows(), V.cols());
        DenseMatrix<scalar_t> hA(A.rows(), A.cols());
        copy_device_to_host(hA, A);
        int gesvd_work_size = gesvd_nb(hA);
        std::vector<real_t> hS(minmn);
        DenseMatrix<scalar_t> hU(U.rows(), U.cols());
        DenseMatrix<scalar_t> hV(V.rows(), V.cols());
        std::vector<scalar_t> hwork(gesvd_work_size);
        std::vector<real_t> hE(5*minmn);
        gesvd(jobu, jobvt, hA, hS.data(), hU, hV, hwork.data(),
              gesvd_work_size, hE.data(), &info);
        copy_host_to_device(S, hS.data(), minmn);
        copy_host_to_device(U, hU);
        copy_host_to_device(V, hV);
      }

      template
      void gesvd_magma(magma_vec_t, magma_vec_t, float*,
                       DenseMatrix<float>&, DenseMatrix<float>&,
                       DenseMatrix<float>&);
      template
      void gesvd_magma(magma_vec_t, magma_vec_t, double*,
                       DenseMatrix<double>&, DenseMatrix<double>&,
                       DenseMatrix<double>&);
      template
      void gesvd_magma(magma_vec_t, magma_vec_t, float*,
                       DenseMatrix<std::complex<float>>&,
                       DenseMatrix<std::complex<float>>&,
                       DenseMatrix<std::complex<float>>&);
      template
      void gesvd_magma(magma_vec_t, magma_vec_t, double*,
                       DenseMatrix<std::complex<double>>&,
                       DenseMatrix<std::complex<double>>&,
                       DenseMatrix<std::complex<double>>&);

      magma_int_t
      getrf_vbatched_max_nocheck_work(magma_int_t* m, magma_int_t* n,
                                      magma_int_t max_m, magma_int_t max_n,
                                      magma_int_t max_minmn, magma_int_t max_mxn,
                                      float **dA_array, magma_int_t *ldda,
                                      magma_int_t **dipiv_array,
                                      magma_int_t *info_array,
                                      void* work, magma_int_t* lwork,
                                      magma_int_t batchCount, Handle& h) {
        if (!batchCount) return 0;
        auto info = magma_sgetrf_vbatched_max_nocheck_work
          (m, n, max_m, max_n, max_minmn, max_mxn,
           dA_array, ldda, dipiv_array, info_array,
           work, lwork, batchCount, get_magma_queue(h));
        if (info)
          std::cerr << "ERROR: magma_sgetrf_vbatched_max_nocheck_work "
                    << "failed with info= " << info << std::endl;
        get_last_error();
        return info;
      }
      magma_int_t
      getrf_vbatched_max_nocheck_work(magma_int_t* m, magma_int_t* n,
                                      magma_int_t max_m, magma_int_t max_n,
                                      magma_int_t max_minmn, magma_int_t max_mxn,
                                      double **dA_array, magma_int_t *ldda,
                                      magma_int_t **dipiv_array,
                                      magma_int_t *info_array,
                                      void* work, magma_int_t* lwork,
                                      magma_int_t batchCount, Handle& h) {
        if (!batchCount) return 0;
        auto info = magma_dgetrf_vbatched_max_nocheck_work
          (m, n, max_m, max_n, max_minmn, max_mxn,
           dA_array, ldda, dipiv_array, info_array,
           work, lwork, batchCount, get_magma_queue(h));
        if (info)
          std::cerr << "ERROR: magma_dgetrf_vbatched_max_nocheck_work "
                    << "failed with info= " << info << std::endl;
        get_last_error();
        return info;
      }
      magma_int_t
      getrf_vbatched_max_nocheck_work(magma_int_t* m, magma_int_t* n,
                                      magma_int_t max_m, magma_int_t max_n,
                                      magma_int_t max_minmn, magma_int_t max_mxn,
                                      std::complex<float>** dA_array, magma_int_t *ldda,
                                      magma_int_t **dipiv_array,
                                      magma_int_t *info_array,
                                      void* work, magma_int_t* lwork,
                                      magma_int_t batchCount, Handle& h) {
        if (!batchCount) return 0;
        auto info = magma_cgetrf_vbatched_max_nocheck_work
          (m, n, max_m, max_n, max_minmn, max_mxn,
           (magmaFloatComplex**)dA_array, ldda,
           dipiv_array, info_array, work, lwork, batchCount, get_magma_queue(h));
        if (info)
          std::cerr << "ERROR: magma_cgetrf_vbatched_max_nocheck_work "
                    << "failed with info= " << info << std::endl;
        get_last_error();
        return info;
      }
      magma_int_t
      getrf_vbatched_max_nocheck_work(magma_int_t* m, magma_int_t* n,
                                      magma_int_t max_m, magma_int_t max_n,
                                      magma_int_t max_minmn, magma_int_t max_mxn,
                                      std::complex<double>** dA_array,
                                      magma_int_t *ldda,
                                      magma_int_t **dipiv_array,
                                      magma_int_t *info_array,
                                      void* work, magma_int_t* lwork,
                                      magma_int_t batchCount, Handle& h) {
        if (!batchCount) return 0;
        auto info = magma_zgetrf_vbatched_max_nocheck_work
          (m, n, max_m, max_n, max_minmn, max_mxn,
           (magmaDoubleComplex**)dA_array, ldda,
           dipiv_array, info_array, work, lwork, batchCount, get_magma_queue(h));
        if (info)
          std::cerr << "ERROR: magma_zgetrf_vbatched_max_nocheck_work "
                    << "failed with info= " << info << std::endl;
        get_last_error();
        return info;
      }

      void
      trsm_vbatched_max_nocheck(magma_side_t side, magma_uplo_t uplo,
                                magma_trans_t transA, magma_diag_t diag,
                                magma_int_t max_m, magma_int_t max_n,
                                magma_int_t *m, magma_int_t *n,
                                float alpha, float **dA_array,
                                magma_int_t *ldda, float **dB_array,
                                magma_int_t *lddb, magma_int_t batchCount,
                                Handle& h) {
        if (!batchCount) return;
        magmablas_strsm_vbatched_max_nocheck
          (side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array,
           ldda, dB_array, lddb, batchCount, get_magma_queue(h));
        get_last_error();
      }
      void
      trsm_vbatched_max_nocheck(magma_side_t side, magma_uplo_t uplo,
                                magma_trans_t transA, magma_diag_t diag,
                                magma_int_t max_m, magma_int_t max_n,
                                magma_int_t *m, magma_int_t *n,
                                double alpha, double **dA_array,
                                magma_int_t *ldda, double **dB_array,
                                magma_int_t *lddb, magma_int_t batchCount,
                                Handle& h) {
        if (!batchCount) return;
        magmablas_dtrsm_vbatched_max_nocheck
          (side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array,
           ldda, dB_array, lddb, batchCount, get_magma_queue(h));
        get_last_error();
      }
      void
      trsm_vbatched_max_nocheck(magma_side_t side, magma_uplo_t uplo,
                                magma_trans_t transA, magma_diag_t diag,
                                magma_int_t max_m, magma_int_t max_n,
                                magma_int_t *m, magma_int_t *n,
                                std::complex<float> alpha,
                                std::complex<float> **dA_array, magma_int_t *ldda,
                                std::complex<float> **dB_array, magma_int_t *lddb,
                                magma_int_t batchCount, Handle& h) {
        if (!batchCount) return;
        magmaFloatComplex alpha_ = {alpha.real(), alpha.imag()};
        magmablas_ctrsm_vbatched_max_nocheck
          (side, uplo, transA, diag, max_m, max_n, m, n, alpha_,
           (magmaFloatComplex**)dA_array, ldda,
           (magmaFloatComplex**)dB_array, lddb, batchCount, get_magma_queue(h));
        get_last_error();
      }
      void
      trsm_vbatched_max_nocheck(magma_side_t side, magma_uplo_t uplo,
                                magma_trans_t transA,
                                magma_diag_t diag, magma_int_t max_m, magma_int_t max_n,
                                magma_int_t *m, magma_int_t *n,
                                std::complex<double> alpha,
                                std::complex<double> **dA_array, magma_int_t *ldda,
                                std::complex<double> **dB_array, magma_int_t *lddb,
                                magma_int_t batchCount, Handle& h) {
        if (!batchCount) return;
        magmaDoubleComplex alpha_ = {alpha.real(), alpha.imag()};
        magmablas_ztrsm_vbatched_max_nocheck
          (side, uplo, transA, diag, max_m, max_n, m, n, alpha_,
           (magmaDoubleComplex**)dA_array, ldda,
           (magmaDoubleComplex**)dB_array, lddb, batchCount, get_magma_queue(h));
        get_last_error();
      }

      void
      gemm_vbatched_max_nocheck(magma_trans_t transA, magma_trans_t transB,
                                magma_int_t *m, magma_int_t *n, magma_int_t *k, float alpha,
                                float const *const *dA_array, magma_int_t *ldda,
                                float const *const *dB_array, magma_int_t *lddb,
                                float beta, float **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                Handle& h) {
        if (!batchCount) return;
        magmablas_sgemm_vbatched_max_nocheck
          (transA, transB, m, n, k, alpha, dA_array, ldda,
           dB_array, lddb, beta, dC_array, lddc, batchCount,
           max_m, max_n, max_k, get_magma_queue(h));
        get_last_error();
      }
      void
      gemm_vbatched_max_nocheck(magma_trans_t transA, magma_trans_t transB,
                                magma_int_t *m, magma_int_t *n, magma_int_t *k, double alpha,
                                double const *const *dA_array, magma_int_t *ldda,
                                double const *const *dB_array, magma_int_t *lddb,
                                double beta, double **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                Handle& h) {
        if (!batchCount) return;
        magmablas_dgemm_vbatched_max_nocheck
          (transA, transB, m, n, k, alpha, dA_array, ldda,
           dB_array, lddb, beta, dC_array, lddc, batchCount,
           max_m, max_n, max_k, get_magma_queue(h));
        get_last_error();
      }
      void
      gemm_vbatched_max_nocheck(magma_trans_t transA, magma_trans_t transB,
                                magma_int_t *m, magma_int_t *n, magma_int_t *k,
                                std::complex<float> alpha,
                                std::complex<float> const *const *dA_array, magma_int_t *ldda,
                                std::complex<float> const *const *dB_array, magma_int_t *lddb,
                                std::complex<float> beta,
                                std::complex<float> **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                Handle& h) {
        if (!batchCount) return;
        magmaFloatComplex alpha_ = {alpha.real(), alpha.imag()},
          beta_ = {beta.real(), beta.imag()};
        magmablas_cgemm_vbatched_max_nocheck
          (transA, transB, m, n, k, alpha_,
           (magmaFloatComplex**)dA_array, ldda,
           (magmaFloatComplex**)dB_array, lddb, beta_,
           (magmaFloatComplex**)dC_array, lddc, batchCount,
           max_m, max_n, max_k, get_magma_queue(h));
        get_last_error();
      }
      void
      gemm_vbatched_max_nocheck(magma_trans_t transA, magma_trans_t transB,
                                magma_int_t *m, magma_int_t *n, magma_int_t *k,
                                std::complex<double> alpha,
                                std::complex<double> const *const *dA_array, magma_int_t *ldda,
                                std::complex<double> const *const *dB_array, magma_int_t *lddb,
                                std::complex<double> beta,
                                std::complex<double> **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                Handle& h) {
        if (!batchCount) return;
        magmaDoubleComplex alpha_ = {alpha.real(), alpha.imag()},
          beta_ = {beta.real(), beta.imag()};
        magmablas_zgemm_vbatched_max_nocheck
          (transA, transB, m, n, k, alpha_,
           (magmaDoubleComplex**)dA_array, ldda,
           (magmaDoubleComplex**)dB_array, lddb, beta_,
           (magmaDoubleComplex**)dC_array, lddc, batchCount,
           max_m, max_n, max_k, get_magma_queue(h));
        get_last_error();
      }

      void
      gemv_vbatched_max_nocheck(magma_trans_t trans, magma_int_t *m, magma_int_t *n,
                                float alpha,
                                float const *const *dA_array, magma_int_t *ldda,
                                float const *const *dB_array, magma_int_t *lddb,
                                float beta, float **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
                                Handle& h) {
        if (!batchCount) return;
        magmablas_sgemv_vbatched_max_nocheck
          (trans, m, n, alpha,
           const_cast<float**>(dA_array), ldda,
           const_cast<float**>(dB_array), lddb, beta,
           dC_array, lddc, batchCount, max_m, max_n, get_magma_queue(h));
        get_last_error();
      }
      void
      gemv_vbatched_max_nocheck(magma_trans_t trans, magma_int_t *m, magma_int_t *n,
                                double alpha,
                                double const *const *dA_array, magma_int_t *ldda,
                                double const *const *dB_array, magma_int_t *lddb,
                                double beta, double **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
                                Handle& h) {
        if (!batchCount) return;
        magmablas_dgemv_vbatched_max_nocheck
          (trans, m, n, alpha,
           const_cast<double**>(dA_array), ldda,
           const_cast<double**>(dB_array), lddb, beta,
           dC_array, lddc, batchCount, max_m, max_n, get_magma_queue(h));
        get_last_error();
      }
      void
      gemv_vbatched_max_nocheck(magma_trans_t trans, magma_int_t *m, magma_int_t *n,
                                std::complex<float> alpha,
                                std::complex<float> const *const *dA_array, magma_int_t *ldda,
                                std::complex<float> const *const *dB_array, magma_int_t *lddb,
                                std::complex<float> beta,
                                std::complex<float> **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
                                Handle& h) {
        if (!batchCount) return;
        magmaFloatComplex alpha_ = {alpha.real(), alpha.imag()},
          beta_ = {beta.real(), beta.imag()};
        magmablas_cgemv_vbatched_max_nocheck
          (trans, m, n, alpha_,
           (magmaFloatComplex**)(const_cast<std::complex<float>**>(dA_array)), ldda,
           (magmaFloatComplex**)(const_cast<std::complex<float>**>(dB_array)), lddb, beta_,
           (magmaFloatComplex**)dC_array, lddc, batchCount,
           max_m, max_n, get_magma_queue(h));
        get_last_error();
      }
      void
      gemv_vbatched_max_nocheck(magma_trans_t trans, magma_int_t *m, magma_int_t *n,
                                std::complex<double> alpha,
                                std::complex<double> const *const *dA_array, magma_int_t *ldda,
                                std::complex<double> const *const *dB_array, magma_int_t *lddb,
                                std::complex<double> beta,
                                std::complex<double> **dC_array, magma_int_t *lddc,
                                magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
                                Handle& h) {
        if (!batchCount) return;
        magmaDoubleComplex alpha_ = {alpha.real(), alpha.imag()},
          beta_ = {beta.real(), beta.imag()};
        magmablas_zgemv_vbatched_max_nocheck
          (trans, m, n, alpha_,
           (magmaDoubleComplex**)(const_cast<std::complex<double>**>(dA_array)), ldda,
           (magmaDoubleComplex**)(const_cast<std::complex<double>**>(dB_array)), lddb, beta_,
           (magmaDoubleComplex**)dC_array, lddc, batchCount,
           max_m, max_n, get_magma_queue(h));
        get_last_error();
      }

    } // end namespace magma
  } // end namespace gpu
} // end namespace strumpack
