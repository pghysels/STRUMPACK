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
 * five (5) year renewals, the U.S. Government igs granted for itself
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
#include <fstream>

#include "FrontalMatrixDenseMPI.hpp"
#include "FrontalMatrixBLRMPI.hpp"
#include "ExtendAdd.hpp"
#include "BLR/BLRExtendAdd.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  FrontalMatrixDenseMPI<scalar_t,integer_t>::FrontalMatrixDenseMPI
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd, const MPIComm& comm, int P)
    : FrontalMatrixMPI<scalar_t,integer_t>
    (sep, sep_begin, sep_end, upd, comm, P) {
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::release_work_memory() {
    F22_.clear(); // remove the update block
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_copy_to_buffers
      (F22_, sbuf, pa, this->upd_to_parent(pa));
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::extadd_blr_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf, const FBLRMPI_t* pa) const {
    BLR::BLRExtendAdd<scalar_t,integer_t>::copy_to_buffers
      (F22_, sbuf, pa, this->upd_to_parent(pa));
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::extadd_blr_copy_to_buffers_col
  (std::vector<std::vector<scalar_t>>& sbuf, const FBLRMPI_t* pa,
  integer_t begin_col, integer_t end_col, const SPOptions<scalar_t>& opts) const {
    BLR::BLRExtendAdd<scalar_t,integer_t>::copy_to_buffers_col
      (F22_, sbuf, pa, this->upd_to_parent(pa), begin_col, end_col);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::extadd_blr_copy_from_buffers
  (BLRMPI_t& F11, BLRMPI_t& F12, BLRMPI_t& F21, BLRMPI_t& F22,
   scalar_t** pbuf, const FBLRMPI_t* pa) const {
    BLR::BLRExtendAdd<scalar_t,integer_t>::copy_from_buffers
      (F11, F12, F21, F22, pbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::extadd_blr_copy_from_buffers_col
  (BLRMPI_t& F11, BLRMPI_t& F12, BLRMPI_t& F21, BLRMPI_t& F22,
   scalar_t** pbuf, const FBLRMPI_t* pa,
   integer_t begin_col, integer_t end_col) const {
    BLR::BLRExtendAdd<scalar_t,integer_t>::copy_from_buffers_col
      (F11, F12, F21, F22, pbuf, pa, this, begin_col, end_col);
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixDenseMPI<scalar_t,integer_t>::node_factor_nonzeros() const {
#if defined(STRUMPACK_USE_ZFP)
    if (compressed_)
      return (F11c_.compressed_size() + F12c_.compressed_size() +
              F21c_.compressed_size()) / sizeof(scalar_t);
    else
#endif
      return FMPI_t::node_factor_nonzeros();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::extend_add() {
    if (!lchild_ && !rchild_) return;
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    for (auto& ch : {lchild_.get(), rchild_.get()}) {
      if (ch) {
        STRUMPACK_FLOPS
          (static_cast<long long int>(ch->dim_upd())*ch->dim_upd()/
           grid()->npactives());
      }
      if (!visit(ch)) continue;
      ch->extend_add_copy_to_buffers(sbuf, this);
    }
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    for (auto& ch : {lchild_.get(), rchild_.get()}) {
      if (!ch) continue;
      ch->extend_add_copy_from_buffers
        (F11_, F12_, F21_, F22_, pbuf.data()+this->master(ch), this);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::build_front
  (const SpMat_t& A) {
    const auto dupd = this->dim_upd();
    const auto dsep = this->dim_sep();
    if (dsep) {
      F11_ = DistM_t(grid(), dsep, dsep);
      using ExFront = ExtractFront<scalar_t,integer_t>;
      ExFront::extract_F11(F11_, A, this->sep_begin_, dsep);
      if (this->dim_upd()) {
        F12_ = DistM_t(grid(), dsep, dupd);
        ExFront::extract_F12
          (F12_, A, this->sep_begin_, this->sep_end_, this->upd_);
        F21_ = DistM_t(grid(), dupd, dsep);
        ExFront::extract_F21
          (F21_, A, this->sep_end_, this->sep_begin_, this->upd_);
      }
    }
    if (dupd) {
      F22_ = DistM_t(grid(), dupd, dupd);
      F22_.zero();
    }
    extend_add();
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontalMatrixDenseMPI<scalar_t,integer_t>::partial_factorization
  (const SPOptions<scalar_t>& opts) {
    ReturnCode err_code = ReturnCode::SUCCESS;
    if (!this->dim_sep() || !grid()->active())
      return err_code;
    TaskTimer pf("FrontalMatrixDenseMPI_factor");
    pf.start();
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
    if (opts.use_gpu())
      slate_opts_.insert({slate::Option::Target, slate::Target::Devices});
    auto slateF11 = slate_matrix(F11_);
    // TODO get return value
    slate::getrf(slateF11, slate_piv_, slate_opts_);
#else
    if (F11_.LU(piv))
      err_code = ReturnCode::ZERO_PIVOT;
#endif
    if (opts.replace_tiny_pivots()) {
      auto thresh = opts.pivot_threshold();
      int prow = F11_.prow(), pcol = F11_.pcol();
      for (int i=0; i<F11_.rows(); i++) {
        int pr = F11_.rowg2p_fixed(i);
        if (pr != prow) continue;
        int pc = F11_.colg2p_fixed(i);
        if (pc != pcol) continue;
        auto& Fii = F11_.global_fixed(i,i);
        if (std::abs(Fii) < thresh)
          Fii = (std::real(Fii) < 0) ? -thresh : thresh;
      }
    }
    long long flops = LU_flops(F11_);
    if (this->dim_upd()) {
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
      auto slateF12 = slate_matrix(F12_);
      auto slateF21 = slate_matrix(F21_);
      auto slateF22 = slate_matrix(F22_);
      slate::getrs(slateF11, slate_piv_, slateF12, slate_opts_);
      slate::gemm(scalar_t(-1.), slateF21, slateF12,
                  scalar_t(1.), slateF22, slate_opts_);
#else
      F12_.laswp(piv, true);
      trsm(Side::L, UpLo::L, Trans::N, Diag::U, scalar_t(1.), F11_, F12_);
      trsm(Side::R, UpLo::U, Trans::N, Diag::N, scalar_t(1.), F11_, F21_);
      gemm(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.), F22_);
#endif
      flops += gemm_flops(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.)) +
        trsm_flops(Side::L, scalar_t(1.), F11_, F12_) +
        trsm_flops(Side::R, scalar_t(1.), F11_, F21_);
    }
    if (Comm().is_root() && opts.verbose()) {
      auto time = pf.elapsed();
      std::cout << "# DenseMPI factorization complete, "
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
                << "GPU=" << opts.use_gpu()
#else
                << "no GPU support"
#endif
                << ", P=" << Comm().size() << ", T=" << params::num_threads
                << ": " << time << " seconds, "
                << flops*F11_.npactives() / 1.e9  << " GFLOPS, "
                << (float(flops)*F11_.npactives() / time) / 1.e9 << " GFLOP/s, "
                << " ds=" << this->dim_sep()
                << ", du=" << this->dim_upd() << std::endl;
    }
    STRUMPACK_FULL_RANK_FLOPS(flops);
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
    STRUMPACK_FLOPS(flops);
#endif
    return err_code;
  }


#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
  template<typename scalar_t,typename integer_t> slate::Matrix<scalar_t>
  FrontalMatrixDenseMPI<scalar_t,integer_t>::slate_matrix(const DistM_t& M) const {
    return slate::Matrix<scalar_t>::fromScaLAPACK
      (M.rows(), M.cols(), const_cast<scalar_t*>(M.data()), M.ld(),
       M.MB(), M.nprows(), M.npcols(), M.grid()->Comm_active().comm());
  }
#endif

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontalMatrixDenseMPI<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    ReturnCode err_code = ReturnCode::SUCCESS;
    if (visit(lchild_)) {
      auto el = lchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
      if (el != ReturnCode::SUCCESS) err_code = el;
    }
    if (visit(rchild_)) {
      auto er = rchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
      if (er != ReturnCode::SUCCESS) err_code = er;
    }
    build_front(A);
    if (etree_level == 0 && opts.write_root_front()) {
      auto Fs = F11_.gather();
      std::string fname = is_complex<scalar_t>() ?
        "Froot_colmajor_complex_" : "Froot_colmajor_real_";
      fname += (sizeof(typename RealType<scalar_t>::value_type) == 4) ?
        "single_" : "double_";
      fname += std::to_string(this->dim_sep()) + "x"
        + std::to_string(this->dim_sep()) + ".dat";
      std::ofstream f(fname, std::ios::out | std::ios::binary);
      f.write((char*)Fs.data(), Fs.rows()*Fs.cols()*sizeof(scalar_t));
      Comm().barrier();
    }
    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
    auto ef = partial_factorization(opts);
    if (ef != ReturnCode::SUCCESS) err_code = ef;
#if defined(STRUMPACK_USE_ZFP)
    compress(opts);
#endif
    return err_code;
  }

#if defined(STRUMPACK_USE_ZFP)
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::compress
  (const SPOptions<scalar_t>& opts) {
    if (opts.compression() == CompressionType::LOSSY ||
        opts.compression() == CompressionType::LOSSLESS ||
        opts.compression() == CompressionType::ZFP_BLR_HODLR) {
      auto prec = opts.lossy_precision();
      F11c_ = LossyMatrix<scalar_t>(F11_.dense_wrapper(), prec);
      F12c_ = LossyMatrix<scalar_t>(F12_.dense_wrapper(), prec);
      F21c_ = LossyMatrix<scalar_t>(F21_.dense_wrapper(), prec);
      F11_.clear();
      F12_.clear();
      F21_.clear();
      compressed_ = true;
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::decompress
  (DistM_t& F11, DistM_t& F12, DistM_t& F21) const {
    if (this->dim_sep()) {
      auto wF11 = F11.dense_wrapper();
      F11c_.decompress(wF11);
      if (this->dim_upd()) {
        auto wF12 = F12.dense_wrapper();
        F12c_.decompress(wF12);
        auto wF21 = F21.dense_wrapper();
        F21c_.decompress(wF21);
      }
    }
  }
#endif

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::fwd_solve_phase2
  (const DistM_t& F11, const DistM_t& F12, const DistM_t& F21,
   DistM_t& b, DistM_t& bupd) const {
    if (!grid()->active()) return;
    TIMER_TIME(TaskType::SOLVE_LOWER, 0, t_s);
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
    if (this->dim_sep()) {
      auto sbloc = slate_matrix(b);
      auto slateF11 = slate_matrix(F11);
      // there is a bug in SLATE's gemmA (when the A matrix is large
      // and the C matrix is small) for SLATE <= 20220700
#if SLATE_VERSION > 20220700
      auto& slopts = slate_opts_;
#else // run on host
      std::map<slate::Option, slate::Value> slopts;
#endif
      slate::getrs
        (slateF11, const_cast<slate::Pivots&>(slate_piv_), sbloc, slopts);
      // TODO count flops
      if (this->dim_upd()) {
        auto sbupd = slate_matrix(bupd);
        auto slateF21 = slate_matrix(F21);
        slate::gemm
          (scalar_t(-1.), slateF21, sbloc, scalar_t(1.), sbupd, slopts);
        // TODO count flops
      }
    }
#else
    if (this->dim_sep()) {
      b.laswp(piv, true);
      if (b.cols() == 1) {
        trsv(UpLo::L, Trans::N, Diag::U, F11, b);
        if (this->dim_upd())
          gemv(Trans::N, scalar_t(-1.), F21, b, scalar_t(1.), bupd);
      } else {
        trsm(Side::L, UpLo::L, Trans::N, Diag::U, scalar_t(1.), F11, b);
        if (this->dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F21, b, scalar_t(1.), bupd);
      }
    }
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t&,
   int etree_level) const {
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    if (visit(lchild_))
      lchild_->forward_multifrontal_solve
        (bloc, bdist, CBl, seqCBl, etree_level);
    if (visit(rchild_))
      rchild_->forward_multifrontal_solve
        (bloc, bdist, CBr, seqCBr, etree_level);
    DistM_t& b = bdist[this->sep_];
    bupd = DistM_t(grid(), this->dim_upd(), b.cols());
    bupd.zero();
    this->extend_add_b(b, bupd, CBl, CBr, seqCBl, seqCBr);
#if defined(STRUMPACK_USE_ZFP)
    if (compressed_) {
      const auto dupd = this->dim_upd();
      const auto dsep = this->dim_sep();
      DistM_t F11(grid(), dsep, dsep), F12(grid(), dsep, dupd),
        F21(grid(), dupd, dsep);
      decompress(F11, F12, F21);
      fwd_solve_phase2(F11, F12, F21, b, bupd);
    } else
#endif
      fwd_solve_phase2(F11_, F12_, F21_, b, bupd);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::bwd_solve_phase1
  (const DistM_t& F11, const DistM_t& F12, const DistM_t& F21,
   DistM_t& y, DistM_t& yupd) const {
    if (!grid()->active()) return;
#if defined(STRUMPACK_USE_SLATE_SCALAPACK) // && (SLATE_VERSION > 20220700)
    if (this->dim_sep()) {
      if (this->dim_upd()) {
#if SLATE_VERSION > 20220700
        auto& slopts = slate_opts_;
#else // run on host
        std::map<slate::Option, slate::Value> slopts;
#endif
        TIMER_TIME(TaskType::SOLVE_UPPER, 0, t_s);
        auto sy = slate_matrix(y);
        auto syupd = slate_matrix(yupd);
        auto slateF12 = slate_matrix(F12);
        slate::gemm
          (scalar_t(-1.), slateF12, syupd, scalar_t(1.), sy, slopts);
        // TODO count flops
      }
    }
#else
    if (this->dim_sep()) {
      TIMER_TIME(TaskType::SOLVE_UPPER, 0, t_s);
      if (y.cols() == 1) {
        if (this->dim_upd())
          gemv(Trans::N, scalar_t(-1.), F12, yupd, scalar_t(1.), y);
        trsv(UpLo::U, Trans::N, Diag::N, F11, y);
      } else {
        if (this->dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F12, yupd, scalar_t(1.), y);
        trsm(Side::L, UpLo::U, Trans::N, Diag::N, scalar_t(1.), F11, y);
      }
    }
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t&,
   int etree_level) const {
    DistM_t& y = ydist[this->sep_];
#if defined(STRUMPACK_USE_ZFP)
    if (compressed_) {
      const auto dupd = this->dim_upd();
      const auto dsep = this->dim_sep();
      DistM_t F11(grid(), dsep, dsep), F12(grid(), dsep, dupd),
        F21(grid(), dupd, dsep);
      decompress(F11, F12, F21);
      bwd_solve_phase1(F11, F12, F21, y, yupd);
    } else
#endif
      bwd_solve_phase1(F11_, F12_, F21_, y, yupd);
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    this->extract_b(y, yupd, CBl, CBr, seqCBl, seqCBr);
    if (visit(lchild_))
      lchild_->backward_multifrontal_solve
        (yloc, ydist, CBl, seqCBl, etree_level);
    if (visit(rchild_))
      rchild_->backward_multifrontal_solve
        (yloc, ydist, CBr, seqCBr, etree_level);
  }

  /**
   * Note that B should be defined on the same context as used in this
   * front. This simplifies communication.
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::extract_CB_sub_matrix_2d
  (const VecVec_t& I, const VecVec_t& J, std::vector<DistM_t>& B) const {
    if (Comm().is_null() || !this->dim_upd()) return;
    // TIMER_TIME(TaskType::HSS_EXTRACT_SCHUR, 3, t_ex_schur);
    auto nB = I.size();
    std::vector<std::vector<std::size_t>> lI(nB), lJ(nB), oI(nB), oJ(nB);
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    using ExtAdd = ExtendAdd<scalar_t,integer_t>;
    for (std::size_t i=0; i<nB; i++) {
      this->find_upd_indices(I[i], lI[i], oI[i]);
      this->find_upd_indices(J[i], lJ[i], oJ[i]);
      ExtAdd::extract_copy_to_buffers
        (F22_, lI[i], lJ[i], oI[i], oJ[i], B[i], sbuf);
    }
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    {
      TIMER_TIME(TaskType::GET_SUBMATRIX_2D_A2A, 2, t_a2a);
      Comm().all_to_all_v(sbuf, rbuf, pbuf);
    }
    for (std::size_t i=0; i<nB; i++)
      ExtAdd::extract_copy_from_buffers
        (B[i], lI[i], lJ[i], oI[i], oJ[i], F22_, pbuf);
  }

  /**
   *  Sr = F22 * R
   *  Sc = F22^* * R
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::sample_CB
  (const DistM_t& R, DistM_t& Sr, DistM_t& Sc,
   FrontalMatrix<scalar_t,integer_t>* pa) const {
    if (F11_.active() || F22_.active()) {
      auto b = R.cols();
      Sr = DistM_t(grid(), this->dim_upd(), b);
      Sc = DistM_t(grid(), this->dim_upd(), b);
      gemm(Trans::N, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.), Sr);
      gemm(Trans::C, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.), Sc);
      STRUMPACK_CB_SAMPLE_FLOPS
        (gemm_flops(Trans::N, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.)) +
         gemm_flops(Trans::C, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.)));
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::sample_CB
  (Trans op, const DistM_t& R, DistM_t& S,
   FrontalMatrix<scalar_t,integer_t>* pa) const {
    if (F11_.active() || F22_.active()) {
      auto b = R.cols();
      S = DistM_t(grid(), this->dim_upd(), b);
      TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
      gemm(op, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.), S);
      TIMER_STOP(t_f22mult);
      STRUMPACK_CB_SAMPLE_FLOPS
        (gemm_flops(op, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.)));
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::delete_factors() {
    if (visit(lchild_)) lchild_->delete_factors();
    if (visit(rchild_)) rchild_->delete_factors();
    F11_ = DistM_t();
    F12_ = DistM_t();
    F21_ = DistM_t();
    F22_ = DistM_t();
    piv = std::vector<int>();
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontalMatrixDenseMPI<scalar_t,integer_t>::matrix_inertia
  (const DistM_t& F, integer_t& neg, integer_t& zero, integer_t& pos) const {
    using real_t = typename RealType<scalar_t>::value_type;
    int prow = F.prow(), pcol = F.pcol();
    for (int i=0; i<F.rows(); i++) {
      int pr = F.rowg2p_fixed(i);
      if (pr != prow) continue;
      int pc = F.colg2p_fixed(i);
      if (pc != pcol) continue;
      if (piv[F.rowg2l(i)] != int(i+1))
        return ReturnCode::INACCURATE_INERTIA;
      real_t Fii = std::abs(F.global(i,i));
      if (Fii > real_t(0.)) pos++;
      else if (Fii < real_t(0.)) neg++;
      else if (Fii == real_t(0.)) zero++;
      else std::cerr << "F(" << i << "," << i << ")=" << Fii << std::endl;
    }
    return ReturnCode::SUCCESS;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontalMatrixDenseMPI<scalar_t,integer_t>::node_inertia
  (integer_t& neg, integer_t& zero, integer_t& pos) const {
    if (!this->dim_sep() || !grid()->active())
      return ReturnCode::SUCCESS;
#if defined(STRUMPACK_USE_ZFP)
    if (compressed_) {
      DistM_t F11(grid(), this->dim_sep(), this->dim_sep());
      auto f = F11.dense_wrapper();
      F11c_.decompress(f);
      return matrix_inertia(F11, neg, zero, pos);
    }
#endif
    return matrix_inertia(F11_, neg, zero, pos);
  }

  // explicit template instantiations
  template class FrontalMatrixDenseMPI<float,int>;
  template class FrontalMatrixDenseMPI<double,int>;
  template class FrontalMatrixDenseMPI<std::complex<float>,int>;
  template class FrontalMatrixDenseMPI<std::complex<double>,int>;

  template class FrontalMatrixDenseMPI<float,long int>;
  template class FrontalMatrixDenseMPI<double,long int>;
  template class FrontalMatrixDenseMPI<std::complex<float>,long int>;
  template class FrontalMatrixDenseMPI<std::complex<double>,long int>;

  template class FrontalMatrixDenseMPI<float,long long int>;
  template class FrontalMatrixDenseMPI<double,long long int>;
  template class FrontalMatrixDenseMPI<std::complex<float>,long long int>;
  template class FrontalMatrixDenseMPI<std::complex<double>,long long int>;

} // end namespace strumpack
