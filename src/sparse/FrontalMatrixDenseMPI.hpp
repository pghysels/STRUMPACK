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
#ifndef FRONTAL_MATRIX_DENSE_MPI_HPP
#define FRONTAL_MATRIX_DENSE_MPI_HPP

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "misc/TaskTimer.hpp"
#include "dense/DistributedMatrix.hpp"
#include "CompressedSparseMatrix.hpp"
#include "FrontalMatrixMPI.hpp"
#if defined(STRUMPACK_USE_ZFP)
#include "FrontalMatrixLossy.hpp"
#endif
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
#define LAPACK_COMPLEX_CPP
#include <slate/slate.hh>
#endif

namespace strumpack {

  template<typename scalar_t,typename integer_t> class ExtractFront;

  template<typename scalar_t,typename integer_t>
  class FrontalMatrixDenseMPI : public FrontalMatrixMPI<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using DistMW_t = DistributedMatrixWrapper<scalar_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using FDMPI_t = FrontalMatrixDenseMPI<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using ExtAdd = ExtendAdd<scalar_t,integer_t>;
    using VecVec_t = std::vector<std::vector<std::size_t>>;
    template<typename _scalar_t,typename _integer_t> friend class ExtendAdd;

  public:
    FrontalMatrixDenseMPI
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd, const MPIComm& comm, int P);
    FrontalMatrixDenseMPI(const FDMPI_t&) = delete;
    FrontalMatrixDenseMPI& operator=(FDMPI_t const&) = delete;

    void release_work_memory() override;

    void extend_add();
    void extend_add_copy_to_buffers
    (std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const override {
      ExtAdd::extend_add_copy_to_buffers(F22_, sbuf, pa, this->upd_to_parent(pa));
    }

    void sample_CB
    (const DistM_t& R, DistM_t& Sr, DistM_t& Sc, F_t* pa) const override;
    void sample_CB
    (Trans op, const DistM_t& R, DistM_t& S,
     FrontalMatrix<scalar_t,integer_t>* pa) const override;

    void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level=0, int task_depth=0) override;

    void forward_multifrontal_solve
    (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
     int etree_level=0) const override;
    void backward_multifrontal_solve
    (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
     int etree_level=0) const override;

    void extract_CB_sub_matrix_2d
    (const VecVec_t& I, const VecVec_t& J,
     std::vector<DistM_t>& B) const override;

    std::string type() const override { return "FrontalMatrixDenseMPI"; }

    long long node_factor_nonzeros() const override;

  private:
    DistM_t F11_, F12_, F21_, F22_;
    std::vector<int> piv;

    void build_front(const SpMat_t& A);
    void partial_factorization();

    void fwd_solve_phase2
    (const DistM_t& F11, const DistM_t& F12, const DistM_t& F21,
     DistM_t& b, DistM_t& bupd) const;
    void bwd_solve_phase1
    (const DistM_t& F11, const DistM_t& F12, const DistM_t& F21,
     DistM_t& y, DistM_t& yupd) const;

#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
    //slate::Matrix<scalar_t> slateF11_, slateF12_, slateF21_, slateF22_;
    slate::Pivots slate_piv_;
    //#if defined(STRUMPACK_USE_CUDA)
    // std::map<slate::Option, slate::Value> slate_opts_ =
    //   {{slate::Option::Target, slate::Target::Devices}};
    //#else
    std::map<slate::Option, slate::Value> slate_opts_;
    //#endif

    slate::Matrix<scalar_t> slate_matrix(DistM_t& M) const;
#endif

#if defined(STRUMPACK_USE_ZFP)
    LossyMatrix<scalar_t> F11c_, F12c_, F21c_;
    bool compressed_ = false;

    void compress(const SPOptions<scalar_t>& opts);
    void decompress(DistM_t& F11, DistM_t& F12, DistM_t& F21) const;
#endif

    using FrontalMatrix<scalar_t,integer_t>::lchild_;
    using FrontalMatrix<scalar_t,integer_t>::rchild_;
    using FrontalMatrixMPI<scalar_t,integer_t>::visit;
    using FrontalMatrixMPI<scalar_t,integer_t>::grid;
    using FrontalMatrixMPI<scalar_t,integer_t>::Comm;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixDenseMPI<scalar_t,integer_t>::FrontalMatrixDenseMPI
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd, const MPIComm& comm, int P)
    : FrontalMatrixMPI<scalar_t,integer_t>
    (sep, sep_begin, sep_end, upd, comm, P) {
    // slate_opts_.insert({slate::Option::Target, slate::Target::Devices});
    // auto it_bool = slate_opts_.insert({slate::Option::Lookahead, slate::Value(1)});
    // std::cout << "insertion success " << (it_bool.second ? " YES" : "NO") << std::endl;
  }


  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::release_work_memory() {
    F22_.clear(); // remove the update block
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
      if (ch && Comm().is_root()) {
        STRUMPACK_FLOPS
          (static_cast<long long int>(ch->dim_upd())*ch->dim_upd());
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

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::partial_factorization() {
    if (this->dim_sep() && grid()->active()) {
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
      auto slateF11 = slate_matrix(F11_);
      slate::getrf(slateF11, slate_piv_, slate_opts_);
#else
      piv = F11_.LU();
#endif
      STRUMPACK_FULL_RANK_FLOPS(LU_flops(F11_));
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
        STRUMPACK_FULL_RANK_FLOPS
          (gemm_flops(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.)) +
           trsm_flops(Side::L, scalar_t(1.), F11_, F12_) +
           trsm_flops(Side::R, scalar_t(1.), F11_, F21_));
      }
    }
  }

#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
  template<typename scalar_t,typename integer_t> slate::Matrix<scalar_t>
  FrontalMatrixDenseMPI<scalar_t,integer_t>::slate_matrix(DistM_t& M) const {
    return slate::Matrix<scalar_t>::fromScaLAPACK
      (M.rows(), M.cols(), M.data(), M.ld(), M.MB(),
       M.nprows(), M.npcols(), M.comm());
  }
#endif

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (visit(lchild_))
      lchild_->multifrontal_factorization(A, opts, etree_level+1, task_depth);
    if (visit(rchild_))
      rchild_->multifrontal_factorization(A, opts, etree_level+1, task_depth);
    TaskTimer t("");
    if (etree_level == 0 && opts.print_root_front_stats()) t.start();
    build_front(A);
    if (etree_level == 0 && opts.write_root_front()) {
      F11_.print_to_files("Froot");
      Comm().barrier();
    }
    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
    partial_factorization();
#if defined(STRUMPACK_USE_ZFP)
    compress(opts);
#endif
    if (etree_level == 0 && opts.print_root_front_stats()) {
      auto time = t.elapsed();
      if (Comm().is_root())
        std::cout << "#   - DenseMPI root front: Nsep= " << this->dim_sep()
                  << " , time= " << time << " sec" << std::endl;
    }
  }

#if defined(STRUMPACK_USE_ZFP)
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::compress
  (const SPOptions<scalar_t>& opts) {
    if (opts.compression() == CompressionType::LOSSY) {
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
    TIMER_TIME(TaskType::SOLVE_LOWER, 0, t_s);
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
    if (this->dim_sep()) {
      auto sbloc = slate_matrix(b);
      auto slateF11 = slate_matrix(F11);
      slate::getrs(slateF11, const_cast<slate::Pivots&>(slate_piv_),
                   sbloc, slate_opts_);
      if (this->dim_upd()) {
        auto sbupd = slate_matrix(bupd);
        auto slateF21 = slate_matrix(F21);
        slate::gemm
          (scalar_t(-1.), slateF21, sbloc, scalar_t(1.), sbupd, slate_opts_);
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
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
    if (this->dim_sep()) {
      if (this->dim_upd()) {
        TIMER_TIME(TaskType::SOLVE_UPPER, 0, t_s);
        auto sy = slate_matrix(y);
        auto syupd = slate_matrix(yupd);
        auto slateF12 = slate_matrix(F12);
        slate::gemm
          (scalar_t(-1.), slateF12, syupd, scalar_t(1.), sy, slate_opts_);
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
    TIMER_TIME(TaskType::HSS_EXTRACT_SCHUR, 3, t_ex_schur);
    auto nB = I.size();
    std::vector<std::vector<std::size_t>> lI(nB), lJ(nB), oI(nB), oJ(nB);
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    for (std::size_t i=0; i<nB; i++) {
      this->find_upd_indices(I[i], lI[i], oI[i]);
      this->find_upd_indices(J[i], lJ[i], oJ[i]);
      ExtAdd::extract_copy_to_buffers
        (F22_, lI[i], lJ[i], oI[i], oJ[i], B[i], sbuf);
    }
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
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

} // end namespace strumpack

#endif //FRONTAL_MATRIX_DENSE_MPI_HPP
