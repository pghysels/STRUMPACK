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
#ifndef FRONTAL_MATRIX_HSS_MPI_HPP
#define FRONTAL_MATRIX_HSS_MPI_HPP

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "misc/TaskTimer.hpp"
#include "FrontalMatrixMPI.hpp"
#include "HSS/HSSMatrixMPI.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  class FrontalMatrixHSSMPI : public FrontalMatrixMPI<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using DistMW_t = DistributedMatrixWrapper<scalar_t>;
    using ExtAdd = ExtendAdd<scalar_t,integer_t>;
    using Opts = SPOptions<scalar_t>;
    template<typename _scalar_t,typename _integer_t> friend class ExtendAdd;

  public:
    FrontalMatrixHSSMPI
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd, const MPIComm& comm, int _total_procs);
    FrontalMatrixHSSMPI(const FrontalMatrixHSSMPI&) = delete;
    FrontalMatrixHSSMPI& operator=(FrontalMatrixHSSMPI const&) = delete;

    void release_work_memory() override;
    void extend_add(int task_depth) {}

    void random_sampling
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     const DistM_t& R, DistM_t& Sr, DistM_t& Sc);
    void sample_CB
    (const DistM_t& R, DistM_t& Sr, DistM_t& Sc, F_t* pa) const override;
    void sample_children_CB
    (const SPOptions<scalar_t>& opts, const DistM_t& R,
     DistM_t& Sr, DistM_t& Sc);

    void multifrontal_factorization
    (const SpMat_t& A, const Opts& opts, int etree_level=0, int task_depth=0) override;

    void forward_multifrontal_solve
    (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
     int etree_level=0) const override;
    void backward_multifrontal_solve
    (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
     int etree_level=0) const override;

    void element_extraction
    (const SpMat_t& A, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J, DistM_t& B) const;
    void element_extraction
    (const SpMat_t& A, const std::vector<std::vector<std::size_t>>& I,
     const std::vector<std::vector<std::size_t>>& J,
     std::vector<DistMW_t>& B) const;
    void extract_CB_sub_matrix_2d
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DistM_t& B) const override;
    void extract_CB_sub_matrix_2d
    (const std::vector<std::vector<std::size_t>>& I,
     const std::vector<std::vector<std::size_t>>& J,
     std::vector<DistM_t>& B) const override;

    long long node_factor_nonzeros() const override;
    integer_t maximum_rank(int task_depth) const override;
    bool isHSS() const override { return true; };
    std::string type() const override { return "FrontalMatrixHSSMPI"; }

    void bisection_partitioning
    (const Opts& opts, integer_t* sorder, bool isroot=true,
     int task_depth=0) override;
    void set_HSS_partitioning
    (const Opts& opts, const HSS::HSSPartitionTree& sep_tree,
     bool is_root) override;

  private:
    std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>> _H;
    HSS::HSSFactorsMPI<scalar_t> _ULV;

    // TODO get rid of this!!!
    mutable std::unique_ptr<HSS::WorkSolveMPI<scalar_t>> _ULVwork;

    /** Schur complement update:
     *    S = F22 - _Theta * Vhat^C * _Phi^C
     **/
    DistM_t _Theta;
    DistM_t _Phi;
    DistM_t _Vhat;
    DistM_t _ThetaVhatC;
    DistM_t _VhatCPhiC;
    DistM_t _DUB01;

    /** these are saved during/after randomized compression and are
        then later used to sample the Schur complement when
        compressing the parent front */
    DistM_t R1;        /* top of the random matrix used to construct
                          HSS matrix of this front */
    DistM_t Sr2, Sc2;  /* bottom of the sample matrix used to
                          construct HSS matrix of this front */
    std::uint32_t _sampled_columns = 0;

    using FrontalMatrix<scalar_t,integer_t>::lchild_;
    using FrontalMatrix<scalar_t,integer_t>::rchild_;
    using FrontalMatrix<scalar_t,integer_t>::dim_sep;
    using FrontalMatrix<scalar_t,integer_t>::dim_upd;
    using FrontalMatrix<scalar_t,integer_t>::dim_blk;
    using FrontalMatrixMPI<scalar_t,integer_t>::visit;
    using FrontalMatrixMPI<scalar_t,integer_t>::grid;
    using FrontalMatrixMPI<scalar_t,integer_t>::Comm;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixHSSMPI<scalar_t,integer_t>::FrontalMatrixHSSMPI
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd, const MPIComm& comm, int total_procs)
    : FrontalMatrixMPI<scalar_t,integer_t>
    (sep, sep_begin, sep_end, upd, comm, total_procs) {
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::release_work_memory() {
    _ThetaVhatC.clear();
    _VhatCPhiC.clear();
    _Vhat.clear();
    if (_H) _H->delete_trailing_block();
    R1.clear();
    Sr2.clear();
    Sc2.clear();
    _DUB01.clear();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::random_sampling
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   const DistM_t& R, DistM_t& Sr, DistM_t& Sc) {
    Sr.zero();
    Sc.zero();
    TIMER_TIME(TaskType::FRONT_MULTIPLY_2D, 1, t_fmult);
    A.front_multiply_2d
      (this->sep_begin_, this->sep_end_, this->upd_, R, Sr, Sc, 0);
    TIMER_STOP(t_fmult);
    TIMER_TIME(TaskType::UUTXR, 1, t_UUtxR);
    sample_children_CB(opts, R, Sr, Sc);
    TIMER_STOP(t_UUtxR);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::sample_CB
  (const DistM_t& R, DistM_t& Sr, DistM_t& Sc, F_t* pa) const {
    if (!dim_upd()) return;
    if (Comm().is_null()) return;
    auto b = R.cols();
    Sr = DistM_t(grid(), dim_upd(), b);
    Sc = DistM_t(grid(), dim_upd(), b);
    TIMER_TIME(TaskType::HSS_SCHUR_PRODUCT, 2, t_sprod);
    _H->Schur_product_direct
      (_Theta, _Vhat, _DUB01, _Phi, _ThetaVhatC, _VhatCPhiC, R, Sr, Sc);
    TIMER_STOP(t_sprod);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::sample_children_CB
  (const SPOptions<scalar_t>& opts, const DistM_t& R,
   DistM_t& Sr, DistM_t& Sc) {
    if (!lchild_ && !rchild_) return;
    DistM_t cSrl, cSrr, cScl, cScr, Rl, Rr;
    DenseM_t seqSrl, seqSrr, seqScl, seqScr, seqRl, seqRr;
    if (lchild_) {
      lchild_->extract_from_R2D(R, Rl, seqRl, this, visit(lchild_));
      seqSrl = DenseM_t(seqRl.rows(), seqRl.cols());
      seqScl = DenseM_t(seqRl.rows(), seqRl.cols());
      seqSrl.zero();
      seqScl.zero();
    }
    if (rchild_) {
      rchild_->extract_from_R2D(R, Rr, seqRr, this, visit(rchild_));
      seqSrr = DenseM_t(seqRr.rows(), seqRr.cols());
      seqScr = DenseM_t(seqRr.rows(), seqRr.cols());
      seqSrr.zero();
      seqScr.zero();
    }
    if (visit(lchild_))
      lchild_->sample_CB(opts, Rl, cSrl, cScl, seqRl, seqSrl, seqScl, this);
    if (visit(rchild_))
      rchild_->sample_CB(opts, Rr, cSrr, cScr, seqRr, seqSrr, seqScr, this);
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    if (visit(lchild_)) {
      lchild_->skinny_ea_to_buffers(cSrl, seqSrl, sbuf, this);
      lchild_->skinny_ea_to_buffers(cScl, seqScl, sbuf, this);
    }
    if (visit(rchild_)) {
      rchild_->skinny_ea_to_buffers(cSrr, seqSrr, sbuf, this);
      rchild_->skinny_ea_to_buffers(cScr, seqScr, sbuf, this);
    }
    std::vector<scalar_t> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    if (lchild_) {
      lchild_->skinny_ea_from_buffers(Sr, pbuf.data(), this);
      lchild_->skinny_ea_from_buffers(Sc, pbuf.data(), this);
    }
    if (rchild_) {
      rchild_->skinny_ea_from_buffers
        (Sr, pbuf.data()+this->master(rchild_), this);
      rchild_->skinny_ea_from_buffers
        (Sc, pbuf.data()+this->master(rchild_), this);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const Opts& opts, int etree_level, int task_depth) {
    if (visit(lchild_))
      lchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    if (visit(rchild_))
      rchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    if (!dim_blk()) return;

    auto mult = [&](DistM_t& R, DistM_t& Sr, DistM_t& Sc) {
      TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
      random_sampling(A, opts, R, Sr, Sc);
      _sampled_columns += R.cols();
      TIMER_STOP(t_sampling);
    };
    // auto elem = [&]
    //   (const std::vector<std::size_t>& I,
    //    const std::vector<std::size_t>& J, DistM_t& B) {
    //   element_extraction(A, I, J, B);
    // };
    auto elem_blocks = [&]
      (const std::vector<std::vector<std::size_t>>& I,
       const std::vector<std::vector<std::size_t>>& J,
       std::vector<DistMW_t>& B) {
      element_extraction(A, I, J, B);
    };

    //_H->compress(mult, elem, opts.HSS_options());
    _H->compress(mult, elem_blocks, opts.HSS_options());

    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();

    if (dim_sep()) {
      if (etree_level > 0) {
        TIMER_TIME(TaskType::HSS_PARTIALLY_FACTOR, 0, t_pfact);
        _ULV = _H->partial_factor();
        TIMER_STOP(t_pfact);
        TIMER_TIME(TaskType::HSS_COMPUTE_SCHUR, 0, t_comp_schur);
        _H->Schur_update(_ULV, _Theta, _Vhat, _DUB01, _Phi);
        if (_Theta.cols() < _Phi.cols()) {
          _VhatCPhiC = DistM_t(_Phi.grid(), _Vhat.cols(), _Phi.rows());
          gemm(Trans::C, Trans::C, scalar_t(1.), _Vhat, _Phi,
               scalar_t(0.), _VhatCPhiC);
          STRUMPACK_SCHUR_FLOPS
            (gemm_flops(Trans::C, Trans::C, scalar_t(1.), _Vhat, _Phi,
                        scalar_t(0.)));
        } else {
          _ThetaVhatC = DistM_t(_Theta.grid(), _Theta.rows(), _Vhat.rows());
          gemm(Trans::N, Trans::C, scalar_t(1.), _Theta, _Vhat,
               scalar_t(0.), _ThetaVhatC);
          STRUMPACK_SCHUR_FLOPS
            (gemm_flops(Trans::N, Trans::C, scalar_t(1.), _Theta, _Vhat,
                        scalar_t(0.)));
        }
        TIMER_STOP(t_comp_schur);
      } else {
        TIMER_TIME(TaskType::HSS_FACTOR, 0, t_fact);
        _ULV = _H->factor();
        TIMER_STOP(t_fact);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
   int etree_level) const {
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    if (visit(lchild_))
      lchild_->forward_multifrontal_solve
        (bloc, bdist, CBl, seqCBl, etree_level+1);
    if (visit(rchild_))
      rchild_->forward_multifrontal_solve
        (bloc, bdist, CBr, seqCBr, etree_level+1);
    DistM_t& b = bdist[this->sep_];
    bupd = DistM_t(_H->grid(), dim_upd(), b.cols());
    bupd.zero();
    this->extend_add_b(b, bupd, CBl, CBr, seqCBl, seqCBr);
    if (etree_level) {
      if (_Theta.cols() && _Phi.cols()) {
        TIMER_TIME(TaskType::SOLVE_LOWER, 0, t_reduce);
        _ULVwork = std::unique_ptr<HSS::WorkSolveMPI<scalar_t>>
          (new HSS::WorkSolveMPI<scalar_t>());
        DistM_t lb(_H->child(0)->grid(_H->grid_local()), b.rows(), b.cols());
        copy(b.rows(), b.cols(), b, 0, 0, lb, 0, 0, grid()->ctxt_all());
        _H->child(0)->forward_solve(_ULV, *_ULVwork, lb, true);
        DistM_t rhs(_H->grid(), _Theta.cols(), bupd.cols());
        copy(rhs.rows(), rhs.cols(), _ULVwork->reduced_rhs, 0, 0,
             rhs, 0, 0, grid()->ctxt_all());
        _ULVwork->reduced_rhs.clear();
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), _Theta, rhs,
               scalar_t(1.), bupd);
        TIMER_STOP(t_reduce);
      }
    } else {
      TIMER_TIME(TaskType::SOLVE_LOWER_ROOT, 0, t_solve);
      DistM_t lb(_H->grid(), b.rows(), b.cols());
      copy(b.rows(), b.cols(), b, 0, 0, lb, 0, 0, grid()->ctxt_all());
      _ULVwork = std::unique_ptr<HSS::WorkSolveMPI<scalar_t>>
        (new HSS::WorkSolveMPI<scalar_t>());
      _H->forward_solve(_ULV, *_ULVwork, lb, false);
      TIMER_STOP(t_solve);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
   int etree_level) const {
    DistM_t& y = ydist[this->sep_];
    TIMER_TIME(TaskType::SOLVE_UPPER, 0, t_expand);
    if (etree_level) {
      if (_Phi.cols() && _Theta.cols()) {
        DistM_t ly
          (_H->child(0)->grid(_H->grid_local()), y.rows(), y.cols());
        copy(y.rows(), y.cols(), y, 0, 0, ly, 0, 0, grid()->ctxt_all());
        if (dim_upd()) {
          // TODO can these copies be avoided??
          DistM_t wx
            (_H->grid(), _Phi.cols(), yupd.cols(), _ULVwork->x, grid()->ctxt_all());
          DistM_t yupdHctxt
            (_H->grid(), dim_upd(), y.cols(), yupd, grid()->ctxt_all());
          gemm(Trans::C, Trans::N, scalar_t(-1.), _Phi, yupdHctxt,
               scalar_t(1.), wx);
          copy(wx.rows(), wx.cols(), wx, 0, 0, _ULVwork->x, 0, 0, grid()->ctxt_all());
        }
        _H->child(0)->backward_solve(_ULV, *_ULVwork, ly);
        copy(y.rows(), y.cols(), ly, 0, 0, y, 0, 0, grid()->ctxt_all());
        _ULVwork.reset();
      }
    } else {
      DistM_t ly(_H->grid(), y.rows(), y.cols());
      copy(y.rows(), y.cols(), y, 0, 0, ly, 0, 0, grid()->ctxt_all());
      _H->backward_solve(_ULV, *_ULVwork, ly);
      copy(y.rows(), y.cols(), ly, 0, 0, y, 0, 0, grid()->ctxt_all());
    }
    TIMER_STOP(t_expand);
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    this->extract_b(y, yupd, CBl, CBr, seqCBl, seqCBr);
    if (visit(lchild_))
      lchild_->backward_multifrontal_solve
        (yloc, ydist, CBl, seqCBl, etree_level+1);
    if (visit(rchild_))
      rchild_->backward_multifrontal_solve
        (yloc, ydist, CBr, seqCBr, etree_level+1);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::element_extraction
  (const SpMat_t& A, const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J, DistM_t& B) const {
    if (I.empty() || J.empty()) return;
    std::vector<std::size_t> gI, gJ;
    gI.reserve(I.size());
    gJ.reserve(J.size());
    const std::size_t dsep = dim_sep();
    for (auto i : I) {
      assert(i < std::size_t(dim_blk()));
      gI.push_back((i < dsep) ? i+this->sep_begin_ : this->upd_[i-dsep]);
    }
    for (auto j : J) {
      assert(j < std::size_t(dim_blk()));
      gJ.push_back((j < dsep) ? j+this->sep_begin_ : this->upd_[j-dsep]);
    }
    TIMER_TIME(TaskType::EXTRACT_2D, 1, t_ex);
    this->extract_2d(A, gI, gJ, B);
    TIMER_STOP(t_ex);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::element_extraction
  (const SpMat_t& A, const std::vector<std::vector<std::size_t>>& I,
   const std::vector<std::vector<std::size_t>>& J,
   std::vector<DistMW_t>& B) const {
    //if (I.empty() || J.empty()) return;
    std::vector<std::vector<std::size_t>> gI(I.size()), gJ(J.size());
    for (std::size_t j=0; j<I.size(); j++) {
      gI[j].reserve(I[j].size());
      gJ[j].reserve(J[j].size());
      const std::size_t dsep = dim_sep();
      for (auto i : I[j]) {
        assert(i < std::size_t(dim_blk()));
        gI[j].push_back((i < dsep) ? i+this->sep_begin_ : this->upd_[i-dsep]);
      }
      for (auto i : J[j]) {
        assert(i < std::size_t(dim_blk()));
        gJ[j].push_back((i < dsep) ? i+this->sep_begin_ : this->upd_[i-dsep]);
      }
    }
    TIMER_TIME(TaskType::EXTRACT_2D, 1, t_ex);
    this->extract_2d(A, gI, gJ, B);
    TIMER_STOP(t_ex);
  }

  /**
   * Extract from (HSS - theta Vhat^* phi*^).
   *
   * Note that B has the same context as this front, otherwise the
   * communication pattern would be hard to figure out.
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::extract_CB_sub_matrix_2d
  (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
   DistM_t& B) const {
    if (Comm().is_null() || !dim_upd()) return;
    TIMER_TIME(TaskType::HSS_EXTRACT_SCHUR, 3, t_ex_schur);
    std::vector<std::size_t> lI, lJ, oI, oJ;
    this->find_upd_indices(J, lJ, oJ);
    this->find_upd_indices(I, lI, oI);

    // TODO extract from child(1) directly
    auto gI = lI;  for (auto& i : gI) i += dim_sep();
    auto gJ = lJ;  for (auto& j : gJ) j += dim_sep();
    DistM_t e = _H->extract(gI, gJ, grid());

    if (_Theta.cols() < _Phi.cols()) {
      DistM_t tr(grid(), lI.size(), _Theta.cols(),
                 _Theta.extract_rows(lI), grid()->ctxt_all());
      DistM_t tc(grid(), _VhatCPhiC.rows(), lJ.size(),
                 _VhatCPhiC.extract_cols(lJ), grid()->ctxt_all());
      gemm(Trans::N, Trans::N, scalar_t(-1), tr, tc, scalar_t(1.), e);
      STRUMPACK_EXTRACTION_FLOPS
        (gemm_flops(Trans::N, Trans::N, scalar_t(-1), tr, tc, scalar_t(1.)));
    } else {
      DistM_t tr(grid(), lI.size(), _ThetaVhatC.cols(),
                 _ThetaVhatC.extract_rows(lI), grid()->ctxt_all());
      DistM_t tc(grid(), lJ.size(), _Phi.cols(),
                 _Phi.extract_rows(lJ), grid()->ctxt_all());
      gemm(Trans::N, Trans::C, scalar_t(-1), tr, tc, scalar_t(1.), e);
      STRUMPACK_EXTRACTION_FLOPS
        (gemm_flops(Trans::N, Trans::C, scalar_t(-1), tr, tc, scalar_t(1.)));
    }

    std::vector<std::vector<scalar_t>> sbuf(this->P());
    ExtAdd::extend_copy_to_buffers(e, oI, oJ, B, sbuf);
    std::vector<scalar_t> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    ExtAdd::extend_copy_from_buffers(B, oI, oJ, e, pbuf);
    TIMER_STOP(t_ex_schur);
  }

  /**
   * Extract from (HSS - theta Vhat^* phi*^).
   *
   * Note that B has the same context as this front, otherwise the
   * communication pattern would be hard to figure out.
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::extract_CB_sub_matrix_2d
  (const std::vector<std::vector<std::size_t>>& I,
   const std::vector<std::vector<std::size_t>>& J,
   std::vector<DistM_t>& B) const {
    if (Comm().is_null() || !dim_upd()) return;
    TIMER_TIME(TaskType::HSS_EXTRACT_SCHUR, 3, t_ex_schur);
    auto nB = I.size();
    std::vector<std::vector<std::size_t>>
      lI(nB), lJ(nB), oI(nB), oJ(nB), gI(nB), gJ(nB);
    for (std::size_t i=0; i<nB; i++) {
      this->find_upd_indices(I[i], lI[i], oI[i]);
      this->find_upd_indices(J[i], lJ[i], oJ[i]);
      gI[i] = lI[i];  for (auto& idx : gI[i]) idx += dim_sep();
      gJ[i] = lJ[i];  for (auto& idx : gJ[i]) idx += dim_sep();
    }
    std::vector<DistM_t> e_vec = _H->extract(gI, gJ, grid());

    std::vector<std::vector<scalar_t>> sbuf(this->P());
    // TODO extract all rows at once?????
    for (std::size_t i=0; i<nB; i++) {
      if (_Theta.cols() < _Phi.cols()) {
        DistM_t tr(grid(), lI[i].size(), _Theta.cols(),
                   _Theta.extract_rows(lI[i]), grid()->ctxt_all());
        DistM_t tc(grid(), _VhatCPhiC.rows(), lJ[i].size(),
                   _VhatCPhiC.extract_cols(lJ[i]), grid()->ctxt_all());
        gemm(Trans::N, Trans::N, scalar_t(-1), tr, tc, scalar_t(1.), e_vec[i]);
        STRUMPACK_EXTRACTION_FLOPS
          (gemm_flops(Trans::N, Trans::N, scalar_t(-1), tr, tc, scalar_t(1.)));
      } else {
        DistM_t tr(grid(), lI[i].size(), _ThetaVhatC.cols(),
                   _ThetaVhatC.extract_rows(lI[i]), grid()->ctxt_all());
        DistM_t tc(grid(), lJ[i].size(), _Phi.cols(),
                   _Phi.extract_rows(lJ[i]), grid()->ctxt_all());
        gemm(Trans::N, Trans::C, scalar_t(-1), tr, tc, scalar_t(1.), e_vec[i]);
        STRUMPACK_EXTRACTION_FLOPS
          (gemm_flops(Trans::N, Trans::C, scalar_t(-1), tr, tc, scalar_t(1.)));
      }
      ExtAdd::extend_copy_to_buffers(e_vec[i], oI[i], oJ[i], B[i], sbuf);
    }
    std::vector<scalar_t> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    for (std::size_t i=0; i<nB; i++)
      ExtAdd::extend_copy_from_buffers(B[i], oI[i], oJ[i], e_vec[i], pbuf);
    TIMER_STOP(t_ex_schur);
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixHSSMPI<scalar_t,integer_t>::node_factor_nonzeros() const {
    // TODO does this return only when root??
    return (_H ? _H->nonzeros() : 0) + _ULV.nonzeros() +
      _Theta.nonzeros() + _Phi.nonzeros();
  }

  template<typename scalar_t,typename integer_t> integer_t
  FrontalMatrixHSSMPI<scalar_t,integer_t>::maximum_rank(int task_depth) const {
    integer_t mr = _H->max_rank();
    if (visit(lchild_))
      mr = std::max(mr, lchild_->maximum_rank(task_depth));
    if (visit(rchild_))
      mr = std::max(mr, rchild_->maximum_rank(task_depth));
    return mr;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::set_HSS_partitioning
  (const Opts& opts, const HSS::HSSPartitionTree& sep_tree, bool is_root) {
    //if (!this->active()) return;
    if (Comm().is_null()) return;
    assert(sep_tree.size == dim_sep());
    if (is_root)
      _H = std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>>
        (new HSS::HSSMatrixMPI<scalar_t>
         (sep_tree, grid(), opts.HSS_options()));
    else {
      HSS::HSSPartitionTree hss_tree(dim_blk());
      hss_tree.c.reserve(2);
      hss_tree.c.push_back(sep_tree);
      hss_tree.c.emplace_back(dim_upd());
      hss_tree.c.back().refine(opts.HSS_options().leaf_size());
      _H = std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>>
        (new HSS::HSSMatrixMPI<scalar_t>
         (hss_tree, grid(), opts.HSS_options()));
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::bisection_partitioning
  (const Opts& opts, integer_t* sorder, bool isroot, int task_depth) {
    if (visit(lchild_))
      lchild_->bisection_partitioning(opts, sorder, false, task_depth);
    if (visit(rchild_))
      rchild_->bisection_partitioning(opts, sorder, false, task_depth);

    HSS::HSSPartitionTree sep_tree(dim_sep());
    sep_tree.refine(opts.HSS_options().leaf_size());

    std::cout << "TODO FrontalMatrixHSSMPI::bisection_partitioning"
              << std::endl;
    for (integer_t i=this->sep_begin_; i<this->sep_end_; i++) sorder[i] = -i;

    // TODO also communicate the tree to everyone working on this front!!
    // see code in EliminationTreeMPIDist

    if (isroot)
      _H = std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>>
        (new HSS::HSSMatrixMPI<scalar_t>
         (sep_tree, grid(), opts.HSS_options()));
    else {
      HSS::HSSPartitionTree hss_tree(dim_blk());
      hss_tree.c.reserve(2);
      hss_tree.c.push_back(sep_tree);
      hss_tree.c.emplace_back(dim_upd());
      hss_tree.c.back().refine(opts.HSS_options().leaf_size());
      _H = std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>>
        (new HSS::HSSMatrixMPI<scalar_t>
         (hss_tree, grid(), opts.HSS_options()));
    }
  }

} // end namespace strumpack

#endif //FRONTAL_MATRIX_HSS_MPI_HPP
