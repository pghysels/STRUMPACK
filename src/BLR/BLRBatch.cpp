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

#include "BLRBatch.hpp"

#if defined(STRUMPACK_USE_MAGMA)
#include "dense/MAGMAWrapper.hpp"
#endif
#if defined(STRUMPACK_USE_KBLAS)
#include "dense/KBLASWrapper.hpp"
#endif

// for gpu::round_up
#include "sparse/fronts/FrontalMatrixGPUKernels.hpp"

namespace strumpack {
  namespace BLR {

    template<typename scalar_t>
    VBatchedGEMM<scalar_t>::VBatchedGEMM
    (std::size_t B, char* dmem) : dmem_(dmem) {
      m_.reserve(B+1);  ldA_.reserve(B+1);  A_.reserve(B);
      n_.reserve(B+1);  ldB_.reserve(B+1);  B_.reserve(B);
      k_.reserve(B+1);  ldC_.reserve(B+1);  C_.reserve(B);
    }

    template<typename scalar_t> void
    VBatchedGEMM<scalar_t>::add(int m, int n, int k,
                                scalar_t* A, scalar_t* B, scalar_t* C) {
      add(m, n, k, A, m, B, k, C, m);
    }

    template<typename scalar_t> void
    VBatchedGEMM<scalar_t>::add(int m, int n, int k,
                                scalar_t* A, scalar_t* B,
                                scalar_t* C, int ldC) {
      add(m, n, k, A, m, B, k, C, ldC);
    }

    template<typename scalar_t> void
    VBatchedGEMM<scalar_t>::add(int m, int n, int k,
                                scalar_t* A, int ldA,
                                scalar_t* B, int ldB,
                                scalar_t* C, int ldC) {
      assert(ldA >= m && ldB >= k && ldC >= m);
      m_.push_back(m);  ldA_.push_back(ldA);  A_.push_back(A);
      n_.push_back(n);  ldB_.push_back(ldB);  B_.push_back(B);
      k_.push_back(k);  ldC_.push_back(ldC);  C_.push_back(C);
    }

    template<typename scalar_t> std::size_t
    VBatchedGEMM<scalar_t>::dwork_bytes(int batchcount) {
#if defined(STRUMPACK_USE_MAGMA)
      return
        gpu::round_up((batchcount+1)*6*sizeof(magma_int_t)) +
        gpu::round_up(batchcount*3*sizeof(scalar_t*));
#else
      return 0;
#endif
    }

    template<typename scalar_t> void
    VBatchedGEMM<scalar_t>::run(scalar_t alpha, scalar_t beta,
                                gpu::Stream& s, gpu::Handle& h) {
#if defined(STRUMPACK_USE_MAGMA)
      magma_int_t batchcount = m_.size();
      if (!batchcount) return;
      auto dimem = reinterpret_cast<magma_int_t*>(dmem_);
      auto dsmem = reinterpret_cast<scalar_t**>
        (dmem_ + gpu::round_up((batchcount+1)*6*sizeof(magma_int_t)));
      // TODO HostMemory?
      std::vector<magma_int_t> imem((batchcount+1)*6);
      auto iptr = imem.begin();
      std::copy(m_.begin(), m_.end(), iptr);   iptr += batchcount+1;
      std::copy(n_.begin(), n_.end(), iptr);   iptr += batchcount+1;
      std::copy(k_.begin(), k_.end(), iptr);   iptr += batchcount+1;
      std::copy(ldA_.begin(), ldA_.end(), iptr);   iptr += batchcount+1;
      std::copy(ldB_.begin(), ldB_.end(), iptr);   iptr += batchcount+1;
      std::copy(ldC_.begin(), ldC_.end(), iptr);   iptr += batchcount+1;
      gpu::copy_host_to_device_async(dimem, imem.data(), (batchcount+1)*6, s);

      std::vector<scalar_t*> smem(batchcount*3);
      auto sptr = smem.begin();
      std::copy(A_.begin(), A_.end(), sptr);   sptr += batchcount;
      std::copy(B_.begin(), B_.end(), sptr);   sptr += batchcount;
      std::copy(C_.begin(), C_.end(), sptr);   sptr += batchcount;
      gpu::copy_host_to_device_async(dsmem, smem.data(), batchcount*3, s);

      for (magma_int_t i=0; i<batchcount; i++) {
        STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*
                        blas::gemm_flops(m_[i],n_[i],k_[i],alpha,beta));
        STRUMPACK_BYTES(sizeof(scalar_t)*
                        blas::gemm_moves(m_[i],n_[i],k_[i]));
      }
      auto max_m = *std::max_element(m_.begin(), m_.end());
      auto max_n = *std::max_element(n_.begin(), n_.end());
      auto max_k = *std::max_element(k_.begin(), k_.end());
      gpu::magma::gemm_vbatched_max_nocheck
        (MagmaNoTrans, MagmaNoTrans,
         dimem, dimem+(batchcount+1), dimem+2*(batchcount+1),
         alpha, dsmem, dimem+3*(batchcount+1),
         dsmem+batchcount, dimem+4*(batchcount+1),
         beta, dsmem+2*batchcount, dimem+5*(batchcount+1),
         batchcount, max_m, max_n, max_k, h);
#else
      std::size_t batchcount = m_.size();
      if (!batchcount) return;
      for (std::size_t i=0; i<batchcount; i++) {
        DenseMatrixWrapper<scalar_t>
          A(m_[i], k_[i], A_[i], ldA_[i]),
          B(k_[i], n_[i], B_[i], ldB_[i]),
          C(m_[i], n_[i], C_[i], ldC_[i]);
        gpu::gemm(h, Trans::N, Trans::N, alpha, A, B, beta, C);
      }
#endif
    }

    template<typename scalar_t> void
    VBatchedTRSMLeftRight<scalar_t>::add(DenseM_t& A,
                                         DenseM_t& Bl, DenseM_t& Br) {
      if (!Bl.rows() || !Bl.cols()) return;
      A_.push_back(&A);
      Bl_.push_back(&Bl);
      Br_.push_back(&Br);
    }

    template<typename scalar_t> void
    VBatchedTRSMLeftRight<scalar_t>::run(gpu::Handle& h,
                                         VectorPool<scalar_t>& workspace) {
      auto B = A_.size();
      if (!B) return;
#if defined(STRUMPACK_USE_MAGMA)
      std::vector<int> mn(3*B);
      std::vector<scalar_t*> AB(3*B);
      int maxm = 0, maxnl = 0, maxnr = 0;
      for (std::size_t i=0; i<B; i++) {
        mn[i    ] = A_[i]->rows();
        mn[i+  B] = Bl_[i]->cols();
        mn[i+2*B] = Br_[i]->rows();
        maxm = std::max(maxm, mn[i]);
        maxnl = std::max(maxnl, mn[i+B]);
        maxnr = std::max(maxnr, mn[i+2*B]);
        AB[i    ] = A_[i]->data();
        AB[i+  B] = Bl_[i]->data();
        AB[i+2*B] = Br_[i]->data();
        STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::trsm_flops(A_[i]->rows(),Bl_[i]->cols(),scalar_t(1.),'L'));
        STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::trsm_flops(A_[i]->rows(),Br_[i]->rows(),scalar_t(1.),'R'));
        STRUMPACK_BYTES(sizeof(scalar_t)*blas::trsm_moves(A_[i]->rows(),Bl_[i]->cols()));
        STRUMPACK_BYTES(sizeof(scalar_t)*blas::trsm_moves(A_[i]->rows(),Br_[i]->rows()));
      }
      std::size_t dmem_size =
        gpu::round_up(3*B*sizeof(int)) +
        gpu::round_up(3*B*sizeof(scalar_t*));
      auto dmem = workspace.get_device_bytes(dmem_size);
      auto dm = dmem.template as<int>();
      auto dnl = dm + B;
      auto dnr = dnl + B;
      auto dA = gpu::aligned_ptr<scalar_t*>(dnr+B);
      auto dBl = dA + B;
      auto dBr = dBl + B;
      gpu::copy_host_to_device(dm, mn.data(), 3*B);
      gpu::copy_host_to_device(dA, AB.data(), 3*B);
      gpu::magma::trsm_vbatched_max_nocheck
        (MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
         maxm, maxnl, dm, dnl, scalar_t(1.),
         dA, dm, dBl, dm, B, h);
      gpu::magma::trsm_vbatched_max_nocheck
        (MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
         maxnr, maxm, dnr, dm, scalar_t(1.),
         dA, dm, dBr, dnr, B, h);
      workspace.restore(dmem);
#else
      for (std::size_t i=0; i<B; i++) {
        gpu::trsm(h, Side::L, UpLo::L, Trans::N, Diag::U,
                  scalar_t(1.), *A_[i], *Bl_[i]);
        gpu::trsm(h, Side::R, UpLo::U, Trans::N, Diag::N,
                  scalar_t(1.), *A_[i], *Br_[i]);
      }
#endif
    }

    template<typename scalar_t> void
    VBatchedTRSM<scalar_t>::add(DenseM_t& A, DenseM_t& B) {
      if (!B.rows() || !B.cols()) return;
      A_.push_back(&A);
      B_.push_back(&B);
    }

    template<typename scalar_t> void
    VBatchedTRSM<scalar_t>::run(gpu::Handle& h,
                                VectorPool<scalar_t>& workspace,
                                bool left) {
      auto B = A_.size();
      if (!B) return;
#if defined(STRUMPACK_USE_MAGMA)
      std::vector<int> mn(2*B);
      std::vector<scalar_t*> AB(2*B);
      int maxm = 0, maxn = 0;
      for (std::size_t i=0; i<B; i++) {
        mn[i  ] = A_[i]->rows();
        mn[i+B] = left ? B_[i]->cols() : B_[i]->rows();
        maxm = std::max(maxm, mn[i]);
        maxn = std::max(maxn, mn[i+B]);
        AB[i  ] = A_[i]->data();
        AB[i+B] = B_[i]->data();
        if (left) {
          STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::trsm_flops(A_[i]->rows(),B_[i]->cols(),scalar_t(1.),'L'));
          STRUMPACK_BYTES(sizeof(scalar_t)*blas::trsm_moves(A_[i]->rows(),B_[i]->cols()));
        } else {
          STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*blas::trsm_flops(A_[i]->rows(),B_[i]->rows(),scalar_t(1.),'R'));
          STRUMPACK_BYTES(sizeof(scalar_t)*blas::trsm_moves(A_[i]->rows(),B_[i]->rows()));
        }
      }
      std::size_t dmem_size =
        gpu::round_up(2*B*sizeof(int)) +
        gpu::round_up(2*B*sizeof(scalar_t*));
      auto dmem = workspace.get_device_bytes(dmem_size);
      auto dm = dmem.template as<int>();
      auto dn = dm + B;
      auto dA = gpu::aligned_ptr<scalar_t*>(dn+B);
      auto dB = dA + B;
      gpu::copy_host_to_device(dm, mn.data(), 2*B);
      gpu::copy_host_to_device(dA, AB.data(), 2*B);
      if (left)
        gpu::magma::trsm_vbatched_max_nocheck
          (MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
           maxm, maxn, dm, dn, scalar_t(1.),
           dA, dm, dB, dm, B, h);
      else
        gpu::magma::trsm_vbatched_max_nocheck
          (MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
           maxn, maxm, dn, dm, scalar_t(1.),
           dA, dm, dB, dn, B, h);
      workspace.restore(dmem);
#else
      for (std::size_t i=0; i<B; i++) {
        if (left)
          gpu::trsm(h, Side::L, UpLo::L, Trans::N, Diag::U,
                    scalar_t(1.), *A_[i], *B_[i]);
        else
          gpu::trsm(h, Side::R, UpLo::U, Trans::N, Diag::N,
                    scalar_t(1.), *A_[i], *B_[i]);
      }
#endif
    }

    template<typename scalar_t>
    const int VBatchedARA<scalar_t>::KBLAS_ARA_BLOCK_SIZE = 16;

    template<typename scalar_t> void
    VBatchedARA<scalar_t>::add(std::unique_ptr<BLRTile<scalar_t>>& tile) {
#if defined(STRUMPACK_USE_KBLAS)
      if (tile->D().rows() <= KBLAS_ARA_BLOCK_SIZE ||
          tile->D().cols() <= KBLAS_ARA_BLOCK_SIZE)
        return;
#else
      if (!tile->D().rows() || !tile->D().cols())
        return;
#endif
      tile_.push_back(&tile);
    }

    template<typename scalar_t> void
    VBatchedARA<scalar_t>::compress(gpu::Handle& handle,
                                    std::unique_ptr<BLRTile<scalar_t>>& t,
                                    scalar_t* work, int* dinfo, real_t tol) {
      auto& A = t->D();
      auto m = A.rows(), n = A.cols();
      if (m == 0 || n == 0) return;
      std::size_t minmn = std::min(m, n);
      auto d_sval_real = reinterpret_cast<real_t*>(work);
      auto d_sval_scalar = work + minmn;
      auto dUmem = d_sval_scalar + minmn;
      auto dVmem = dUmem + m*minmn;
      auto dAmem = dVmem + n*minmn;
      auto svd_work = dAmem + m*n;
      DenseMW_t Atmp(m, n, dAmem, m);
      gpu::copy_device_to_device(Atmp, A);
      DenseMW_t dU(m, minmn, dUmem, m), dV(n, minmn, dVmem, n);
      std::vector<real_t> h_sval_real(minmn);
      int lwork = gpu::gesvdj_buffersize<scalar_t>(handle, Jobz::V, m, n);
      gpu::gesvdj<scalar_t>
        (handle, Jobz::V, Atmp, d_sval_real, dU, dV, dinfo,
         svd_work, lwork, tol);
      gpu::copy_device_to_host(h_sval_real.data(), d_sval_real, minmn);
      std::size_t rank = 0;
      while (rank < minmn && h_sval_real[rank] >= tol) rank++;
      if (rank*(m+n) >= m*n) return;
      auto dA = A.data();
      DenseMW_t tU(m, rank, dA, m), tV(rank, n, dA+m*rank, rank);
      auto t_lr = LRTile<scalar_t>::create_as_wrapper(tU, tV);
      DenseMW_t dU_tmp(m, rank, dU, 0, 0), dV_tmp(n, rank, dV, 0, 0);
      gpu::copy_device_to_device(tU, dU_tmp);
      gpu::geam<scalar_t>
        (handle, Trans::C, Trans::N, 1., dV_tmp, 0., dV_tmp, tV);
      gpu::copy_real_to_scalar<scalar_t>(d_sval_scalar, d_sval_real, rank);
      gpu::dgmm(handle, Side::L, tV, d_sval_scalar, tV);
      t = std::move(t_lr);

      // TODO flops!
    }

    template<typename scalar_t> void
    VBatchedARA<scalar_t>::run_kblas(gpu::Handle& handle,
                                     VectorPool<scalar_t>& workspace,
                                     real_t tol) {
#if defined(STRUMPACK_USE_KBLAS)
      auto B = tile_.size();
      if (!B) return;
      int maxm = 0, maxn = 0, maxminmn = KBLAS_ARA_BLOCK_SIZE;
      std::vector<int> m_n_maxr(3*B);
      for (std::size_t i=0; i<B; i++) {
        int m = tile_[i]->get()->D().rows(),
          n = tile_[i]->get()->D().cols();
        auto minmn = std::min(m, n);
        maxminmn = std::max(maxminmn, minmn);
        maxm = std::max(maxm, m);
        maxn = std::max(maxn, n);
        m_n_maxr[i    ] = m;
        m_n_maxr[i+B  ] = n;
        m_n_maxr[i+2*B] = m*n/(m+n);
      }
      std::size_t smem_size = 0;
      for (std::size_t i=0; i<B; i++)
        smem_size += tile_[i]->get()->D().rows()*maxminmn +
          tile_[i]->get()->D().cols()*maxminmn;
      std::size_t dmem_size =
        gpu::round_up(5*B*sizeof(int)) +
        gpu::round_up(3*B*sizeof(scalar_t*)) +
        gpu::round_up(smem_size*sizeof(scalar_t));
      auto dmem = workspace.get_device_bytes(dmem_size);
      auto dm = dmem.template as<int>();
      auto dn = dm + B;
      auto dmaxr = dn + B;
      auto dr = dmaxr + B;
      auto dinfo = dr + B;
      auto dA = gpu::aligned_ptr<scalar_t*>(dinfo+B);
      auto dU = dA + B;
      auto dV = dU + B;
      auto smem = gpu::aligned_ptr<scalar_t>(dV+B);
      std::vector<scalar_t*> AUV(3*B);
      for (std::size_t i=0; i<B; i++) {
        auto m = m_n_maxr[i], n = m_n_maxr[i+B];
        AUV[i    ] = tile_[i]->get()->D().data();
        AUV[i+  B] = smem;  smem += m*maxminmn;
        AUV[i+2*B] = smem;  smem += n*maxminmn;
      }
      gpu::copy_host_to_device(dm, m_n_maxr.data(), 3*B);
      gpu::copy_host_to_device(dA, AUV.data(), 3*B);
      gpu::kblas::ara
        (handle, dm, dn, dA, dm, dU, dm, dV, dn, dr,
         tol, maxm, maxn, dmaxr, KBLAS_ARA_BLOCK_SIZE, 10, dinfo, 1, B);
      std::vector<int> ranks(B), info(B);
      gpu::copy_device_to_host(ranks.data(), dr, B);
      gpu::copy_device_to_host(info.data(), dinfo, B);
      for (std::size_t i=0; i<B; i++) {
        auto rank = ranks[i], m = m_n_maxr[i], n = m_n_maxr[i+B];
        STRUMPACK_FLOPS(blas::ara_flops(m, n, rank, 10));
        if (info[i] == KBLAS_Success) {
          auto dA = AUV[i];
          DenseMW_t tU(m, rank, dA, m),
            tV(rank, n, dA+m*rank, rank);
          *tile_[i] = LRTile<scalar_t>::create_as_wrapper(tU, tV);
          DenseMW_t dUtmp(m, rank, AUV[i+B], m),
            dVtmp(n, rank, AUV[i+2*B], n);
          gpu::copy_device_to_device(tU, dUtmp);
          gpu::geam<scalar_t>
            (handle, Trans::C, Trans::N, 1., dVtmp, 0., dVtmp, tV);
        }
      }
      workspace.restore(dmem);
#endif
    }

    template<typename scalar_t> void
    VBatchedARA<scalar_t>::run_svd(gpu::Handle& handle,
                                   VectorPool<scalar_t>& workspace,
                                   real_t tol) {
      auto B = tile_.size();
      if (!B) return;
      std::size_t compress_lwork = 0;
      for (std::size_t i=0; i<B; i++) {
        auto m = tile_[i]->get()->D().rows(), n = tile_[i]->get()->D().cols();
        auto minmn = std::min(m, n);
        compress_lwork =
          std::max(compress_lwork,
                   gpu::gesvdj_buffersize<scalar_t>(handle, Jobz::V, m, n)
                   // singular values (real and scalar), U, V and A
                   + minmn * (2 + m + n) + m*n);
      }
      auto lwork = gpu::round_up(compress_lwork*sizeof(scalar_t)) + sizeof(int);
      auto work = workspace.get_device_bytes(lwork);
      auto wptr = work.template as<scalar_t>();
      auto dinfo = gpu::aligned_ptr<int>(wptr + compress_lwork);
      for (std::size_t i=0; i<B; i++)
        compress(handle, *tile_[i], wptr, dinfo, tol);
      workspace.restore(work);
    }

    template<typename scalar_t> void
    VBatchedARA<scalar_t>::run_magma(gpu::Handle& handle,
                                     VectorPool<scalar_t>& workspace,
                                     real_t tol) {
#if defined(STRUMPACK_USE_MAGMA)
      auto B = tile_.size();
      if (!B) return;
      std::size_t lwork = 0, lrwork = 0, max_mn = 0, max_minmn = 0;
      for (std::size_t i=0; i<B; i++) {
        auto& A = (*tile_[i])->D();
        auto m = A.rows(), n = A.cols();
        max_mn = std::max(max_mn, std::size_t(m)*n);
        max_minmn = std::max(max_minmn, std::min(m, n));
        lwork = std::max(lwork, gpu::magma::geqp3_scalar_worksize(A));
        lrwork = std::max(lrwork, gpu::magma::geqp3_real_worksize(A));
      }

      // TODO combine and get from workspace
      gpu::DeviceMemory<scalar_t> Acopy(max_mn), dtau(max_minmn), dwork(lwork);
      gpu::DeviceMemory<real_t> rwork(lrwork);

      for (std::size_t i=0; i<B; i++) {
        auto& A = (*tile_[i])->D();
        auto m = A.rows(), n = A.cols();
        auto minmn = std::min(m, n);
        DenseMW_t Ac(m, n, Acopy, m);
        gpu::copy(Ac, A);

        // TODO flops!

        int info;
        std::vector<magma_int_t> jpvt(n);
        gpu::magma::geqp3(Ac, jpvt.data(), dtau, dwork, lwork, rwork, &info);
        std::vector<scalar_t> tau(minmn);
        gpu::copy<scalar_t>(tau.data(), dtau, minmn);
        if (info) std::cerr << "ERROR: magma_geqp3 info= " << info << std::endl;
        std::vector<scalar_t> diagR(minmn);
        // copy diagonal elements
        // for (std::size_t i=0; i<minmn; i++)
        //   gpu::copy(diagR.data()+i, Ac.ptr(i, i), 1);
        gpu::device_copy_2D(diagR.data(), sizeof(scalar_t),
                            Ac.data(), sizeof(scalar_t)*(Ac.ld()+1),
                            sizeof(scalar_t), minmn, gpu::CopyDir::D2H);
        std::size_t rank = 0;
        while (rank < minmn && std::abs(diagR[rank]/diagR[0]) >= tol) rank++;
        if (rank*(m+n) >= m*n) continue;
        DenseMW_t Q(m, rank, Ac, 0, 0), R(rank, n, Ac, 0, 0),
          U(m, rank, A.data(), m), V(rank, n, A.data()+m*rank, rank);
        *tile_[i] = LRTile<scalar_t>::create_as_wrapper(U, V);
        gpu::memset<scalar_t>(V.data(), 0, rank*n);
        // TODO optimize?
        for (std::size_t i=0; i<n; i++)
          gpu::copy(V.ptr(0, jpvt[i]-1), R.ptr(0, i), std::min(rank, i+1));
        auto nb = gpu::magma::get_geqp3_nb(Ac);
        gpu::magma::xxgqr(m, rank, rank, Q, tau.data(), dwork, nb, &info);
        if (info) std::cerr << "ERROR: magma_xxgqr info= " << info << std::endl;
        gpu::copy(U, Q);
      }
#endif
    }

    template<typename scalar_t> void
    VBatchedARA<scalar_t>::run(gpu::Handle& handle,
                               VectorPool<scalar_t>& workspace,
                               real_t tol) {
#if defined(STRUMPACK_USE_KBLAS)
      run_kblas(handle, workspace, tol);
#else
#if 0 //defined(STRUMPACK_USE_MAGMA)
      run_magma(handle, workspace, tol);
#else
      run_svd(handle, workspace, tol);
#endif
#endif
    }

    template<typename scalar_t> void
    VBatchedARA<scalar_t>::kblas_wsquery(gpu::Handle& handle,
                                         int batchcount) {
#if defined(STRUMPACK_USE_KBLAS)
      gpu::kblas::ara_workspace<scalar_t>
        (handle, KBLAS_ARA_BLOCK_SIZE, batchcount);
#endif
    }

    // explicit template instantiation
    template class VBatchedGEMM<float>;
    template class VBatchedGEMM<double>;
    template class VBatchedGEMM<std::complex<float>>;
    template class VBatchedGEMM<std::complex<double>>;

    template class VBatchedTRSMLeftRight<float>;
    template class VBatchedTRSMLeftRight<double>;
    template class VBatchedTRSMLeftRight<std::complex<float>>;
    template class VBatchedTRSMLeftRight<std::complex<double>>;

    template class VBatchedTRSM<float>;
    template class VBatchedTRSM<double>;
    template class VBatchedTRSM<std::complex<float>>;
    template class VBatchedTRSM<std::complex<double>>;

    template class VBatchedARA<float>;
    template class VBatchedARA<double>;
    template class VBatchedARA<std::complex<float>>;
    template class VBatchedARA<std::complex<double>>;

  } // end namespace BLR
} // end namespace strumpack

