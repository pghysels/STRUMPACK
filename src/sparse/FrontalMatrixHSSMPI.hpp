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
#include "misc/MPIWrapper.hpp"
#include "dense/DistributedMatrix.hpp"
#include "CompressedSparseMatrix.hpp"
#include "MatrixReordering.hpp"
#include "FrontalMatrixMPI.hpp"

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
    template<typename _scalar_t,typename _integer_t> friend class ExtendAdd;

  public:
    FrontalMatrixHSSMPI
    (integer_t _sep, integer_t _sep_begin, integer_t _sep_end,
     std::vector<integer_t>& _upd, MPI_Comm _front_comm, int _total_procs);
    FrontalMatrixHSSMPI(const FrontalMatrixHSSMPI&) = delete;
    FrontalMatrixHSSMPI& operator=(FrontalMatrixHSSMPI const&) = delete;
    ~FrontalMatrixHSSMPI() {}

    void release_work_memory() override;
    void extend_add(int task_depth) {}
    void random_sampling
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     DistM_t& R, DistM_t& Sr, DistM_t& Sc, int etree_level);
    void sample_CB
    (const SPOptions<scalar_t>& opts, const DistM_t& R, DistM_t& Sr,
     DistM_t& Sc, F_t* pa) const override;
    void sample_children_CB
    (const SPOptions<scalar_t>& opts,
     DistM_t& R, DistM_t& Sr, DistM_t& Sc);
    void sample_children_CB_seqseq
    (const SPOptions<scalar_t>& opts, const DistM_t& R,
     DistM_t& Sr, DistM_t& Sc);
    void skinny_extend_add
    (DistM_t& cSrl, DistM_t& cScl, DistM_t& cSrr, DistM_t& cScr,
     DistM_t& Sr, DistM_t& Sc);

    void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level=0, int task_depth=0) override;

    void forward_multifrontal_solve
    (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
     int etree_level=0) const override;
    void backward_multifrontal_solve
    (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
     int etree_level=0) const override;

    void element_extraction
    (const SpMat_t& A, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J, DistM_t& B);
    void extract_CB_sub_matrix_2d
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DistM_t& B) const override;

    long long node_factor_nonzeros() const;
    integer_t maximum_rank(int task_depth) const;
    bool isHSS() const override { return true; };
    std::string type() const override { return "FrontalMatrixHSSMPI"; }

    void bisection_partitioning
    (const SPOptions<scalar_t>& opts, integer_t* sorder, bool isroot=true,
     int task_depth=0) override;
    void set_HSS_partitioning
    (const SPOptions<scalar_t>& opts, const HSS::HSSPartitionTree& sep_tree,
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
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixHSSMPI<scalar_t,integer_t>::FrontalMatrixHSSMPI
  (integer_t _sep, integer_t _sep_begin, integer_t _sep_end,
   std::vector<integer_t>& _upd, MPI_Comm _front_comm, int _total_procs)
    : FrontalMatrixMPI<scalar_t,integer_t>
    (_sep, _sep_begin, _sep_end, _upd, _front_comm, _total_procs) {
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

  /**
   * Simply generate Rrow as a random matrix, but in such a way that
   * rows which also occur in the sibling or in the parent are the same
   * there.
   * Rcol is just a copy of Rrow.
   * Compute F * Rrow and F^T Rcol.
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::random_sampling
  (const SpMat_t& A, const SPOptions<scalar_t>& opts, DistM_t& R,
   DistM_t& Sr, DistM_t& Sc, int etree_level) {
    Sr.zero();
    Sc.zero();
    auto f0 = params::flops;
    TIMER_TIME(TaskType::FRONT_MULTIPLY_2D, 1, t_fmult);
    A.front_multiply_2d
      (this->sep_begin, this->sep_end, this->upd, R, Sr, Sc,
       this->ctxt_all, this->front_comm, 0);
    TIMER_STOP(t_fmult);
    TIMER_TIME(TaskType::UUTXR, 1, t_UUtxR);
    params::sparse_sample_flops += params::flops - f0;
    auto f1 = params::flops;
    sample_children_CB(opts, R, Sr, Sc);
    params::CB_sample_flops += params::flops - f1;
    TIMER_STOP(t_UUtxR);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::sample_CB
  (const SPOptions<scalar_t>& opts, const DistM_t& R,
   DistM_t& Sr, DistM_t& Sc, F_t* pa) const {
    if (!this->dim_upd()) return;
    if (this->comm() == MPI_COMM_NULL) return;
    auto b = R.cols();
    Sr = DistM_t(this->ctxt, this->dim_upd(), b);
    Sc = DistM_t(this->ctxt, this->dim_upd(), b);
    TIMER_TIME(TaskType::HSS_SCHUR_PRODUCT, 2, t_sprod);
    _H->Schur_product_direct
      (_Theta, _Vhat, _DUB01, _Phi, _ThetaVhatC, _VhatCPhiC, R, Sr, Sc);
    TIMER_STOP(t_sprod);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::sample_children_CB
  (const SPOptions<scalar_t>& opts, DistM_t& R, DistM_t& Sr, DistM_t& Sc) {
    if (!this->lchild && !this->rchild) return;
    // both children are sequential
    if (this->lchild && !this->lchild->isMPI() &&
        this->rchild && !this->rchild->isMPI()) {
      sample_children_CB_seqseq(opts, R, Sr, Sc);
      return;
    }
    // both children are MPI
    if (this->lchild->isMPI() && this->rchild->isMPI()) {
      auto lch = dynamic_cast<FMPI_t*>(this->lchild);
      auto rch = dynamic_cast<FMPI_t*>(this->rchild);
      DistM_t cSrl, cScl, cSrr, cScr;
      // this needs to be done first, because it is collective, then
      // the sampling for left/right can be done concurrently
      auto I = lch->upd_to_parent(this);
      DistM_t cRl(lch->ctxt, I.size(), R.cols(),
                  R.extract_rows(I, this->front_comm), this->ctxt_all);
      I = rch->upd_to_parent(this);
      DistM_t cRr(rch->ctxt, I.size(), R.cols(),
                  R.extract_rows(I, this->front_comm), this->ctxt_all);
      lch->sample_CB(opts, cRl, cSrl, cScl, this);
      rch->sample_CB(opts, cRr, cSrr, cScr, this);
      TIMER_TIME(TaskType::SKINNY_EXTEND_ADD_MPIMPI, 2, t_sea);
      skinny_extend_add(cSrl, cScl, cSrr, cScr, Sr, Sc);
      TIMER_STOP(t_sea);
      return;
    }
    // one child is seq or NULL, the other is MPI or NULL
    F_t* ch_seq = nullptr;
    FMPI_t* ch_mpi = dynamic_cast<FMPI_t*>(this->lchild);
    if (!ch_mpi)
      ch_mpi = dynamic_cast<FMPI_t*>(this->rchild);
    if (ch_mpi)
      ch_seq = (ch_mpi == this->lchild) ? this->rchild : this->lchild;
    else ch_seq = this->lchild ? this->lchild : this->rchild;
    DenseM_t Rseq, Srseq, Scseq;
    int m = 0, n = 0;
    auto pch = this->child_master(ch_seq);
    if (ch_seq) {
      m = R.rows(), n = R.cols();
      auto p = mpi_rank(this->front_comm);
      if (p == pch) {
        Rseq = DenseM_t(m, n);
        Srseq = DenseM_t(m, n);
        Scseq = DenseM_t(m, n);
        Srseq.zero();
        Scseq.zero();
      }
      strumpack::copy(m, n, R, 0, 0, Rseq, pch, this->ctxt_all);
      if (p == pch) ch_seq->sample_CB(opts, Rseq, Srseq, Scseq, this);
    }
    if (ch_mpi) {
      DistM_t cSr, cSc, S_dummy;
      auto I = ch_mpi->upd_to_parent(this);
      DistM_t cR(ch_mpi->ctxt, I.size(), R.cols(),
                 R.extract_rows(I, this->front_comm), this->ctxt_all);
      ch_mpi->sample_CB(opts, cR, cSr, cSc, this);
      TIMER_TIME(TaskType::SKINNY_EXTEND_ADD_MPI1, 2, t_sea);
      if (ch_mpi == this->lchild)
        skinny_extend_add(cSr, cSc, S_dummy, S_dummy, Sr, Sc);
      else skinny_extend_add(S_dummy, S_dummy, cSr, cSc, Sr, Sc);
      TIMER_STOP(t_sea);
    }
    if (ch_seq) {
      TIMER_TIME(TaskType::SKINNY_EXTEND_ADD_SEQ1, 2, t_sea);
      DistM_t Stmp(this->ctxt, m, n);
      strumpack::copy(m, n, Srseq, pch, Stmp, 0, 0, this->ctxt_all);
      Sr.add(Stmp);
      strumpack::copy(m, n, Scseq, pch, Stmp, 0, 0, this->ctxt_all);
      Sc.add(Stmp);
      TIMER_STOP(t_sea);
    }
  }

  /**
   * Both children are sequential. This means this front only has 2
   * processes working on it. Both these processes are the master of 1
   * of the children, so they will both need to construct a sequential
   * copy of R, then compute Srowseq and Scolseq, and add it to the 2D
   * block-cyclicly distributed Srow and Scol of the parent.
   *
   * It is very important to combine this function for the two
   * children together, to avoid that the children are handled one
   * after the other!!
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::sample_children_CB_seqseq
  (const SPOptions<scalar_t>& opts, const DistM_t& R, DistM_t& Sr, DistM_t& Sc) {
    auto m = R.rows();
    auto n = R.cols();
    auto rank = mpi_rank(this->front_comm);
    auto pl = this->child_master(this->lchild);
    auto pr = this->child_master(this->rchild);
    DenseM_t Rseq(m, n), Srseq(m, n), Scseq(m, n);
    Srseq.zero();
    Scseq.zero();
    strumpack::copy(m, n, R, 0, 0, Rseq, pl, this->ctxt_all);
    strumpack::copy(m, n, R, 0, 0, Rseq, pr, this->ctxt_all);
    if (rank == pl) this->lchild->sample_CB(opts, Rseq, Srseq, Scseq, this);
    if (rank == pr) this->rchild->sample_CB(opts, Rseq, Srseq, Scseq, this);
    TIMER_TIME(TaskType::SKINNY_EXTEND_ADD_SEQSEQ, 2, t_sea);
    DistM_t Stmp(this->ctxt, m, n);
    // TODO combine these 4 copies into 1 all-to-all?
    strumpack::copy(m, n, Srseq, pl, Stmp, 0, 0, this->ctxt_all);
    Sr.add(Stmp);
    strumpack::copy(m, n, Scseq, pl, Stmp, 0, 0, this->ctxt_all);
    Sc.add(Stmp);
    strumpack::copy(m, n, Srseq, pr, Stmp, 0, 0, this->ctxt_all);
    Sr.add(Stmp);
    strumpack::copy(m, n, Scseq, pr, Stmp, 0, 0, this->ctxt_all);
    Sc.add(Stmp);
    TIMER_STOP(t_sea);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::skinny_extend_add
  (DistM_t& cSrl, DistM_t& cScl, DistM_t& cSrr, DistM_t& cScr,
   DistM_t& Sr, DistM_t& Sc) {
    std::vector<std::vector<scalar_t>> sbuf(mpi_nprocs(this->comm()));
    auto lch = dynamic_cast<FMPI_t*>(this->lchild);
    auto rch = dynamic_cast<FMPI_t*>(this->rchild);
    if (cSrl.active())
      ExtAdd::skinny_extend_add_copy_to_buffers
        (cSrl, cScl, sbuf, this, lch->upd_to_parent(this));
    if (cSrr.active())
      ExtAdd::skinny_extend_add_copy_to_buffers
        (cSrr, cScr, sbuf, this, rch->upd_to_parent(this));
    scalar_t *rbuf = nullptr, **pbuf = nullptr;
    all_to_all_v(sbuf, rbuf, pbuf, this->comm());
    if (lch) // unpack left child contribution
      ExtAdd::skinny_extend_add_copy_from_buffers
        (Sr, Sc, pbuf, this, lch);
    if (rch) // unpack right child contribution
      ExtAdd::skinny_extend_add_copy_from_buffers
        (Sr, Sc, pbuf+this->child_master(rch), this, rch);
    delete[] pbuf;
    delete[] rbuf;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (this->visit(this->lchild))
      this->lchild->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    if (this->visit(this->rchild))
      this->rchild->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    if (!this->dim_blk()) return;

    auto mult = [&](DistM_t& R, DistM_t& Sr, DistM_t& Sc) {
      auto f0 = params::flops;
      TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
      random_sampling(A, opts, R, Sr, Sc, etree_level);
      params::sample_flops += params::flops - f0;
      if (_sampled_columns == 0)
        params::initial_sample_flops += params::flops - f0;
      _sampled_columns += R.cols();
      TIMER_STOP(t_sampling);
    };
    auto elem = [&](const std::vector<std::size_t>& I,
                    const std::vector<std::size_t>& J, DistM_t& B) {
      auto f0 = params::flops;
      element_extraction(A, I, J, B);
      params::extraction_flops += params::flops - f0;
    };

    // auto N = this->dim_blk();
    // DistM_t eye(this->ctxt, N, N), FI(this->ctxt, N, N),
    //   FtI(this->ctxt, N, N);
    // eye.eye();
    // mult(eye, FI, FtI);
    // std::vector<std::size_t> I(N);
    // std::iota(I.begin(), I.end(), 0);
    // DistM_t Fdense(this->ctxt, N, N);
    // elem(I, I, Fdense);
    // auto Ftdense = Fdense.transpose();
    // Fdense.scaled_add(scalar_t(-1.), FI);
    // Ftdense.scaled_add(scalar_t(-1.), FtI);
    // auto errF = Fdense.normF() / FI.normF();
    // auto errFt = Ftdense.normF() / FtI.normF();
    // if (mpi_rank(this->front_comm)==0)
    //   std::cout << "||FI-Fdense||/||FI|| = " << errF
    //             << " ||FtI-Ftdense||/||FtI|| = " << errFt << std::endl;
    // if (errF > 1e-8) { // || errFt > 1e-8) {
    //   Fdense.print("Fdense_err");
    //   //Ftdense.print("Ftdense_err");
    //   elem(I, I, Fdense);
    //   Fdense.print("Fdense");
    //   FI.print("FI");
    //   exit(1);
    // }

    auto f0 = params::flops;
    // auto HSSopts = opts.HSS_options();
    // int child_samples = 0;
    // TODO do the child random samples need to be communicated??
    // if (this->lchild)
    //   child_samples = this->lchild->random_samples();
    // if (this->rchild)
    //   child_samples = std::max(child_samples, this->rchild->random_samples());
    // HSSopts.set_d0(std::max(child_samples - HSSopts.dd(), HSSopts.d0()));
    // if (opts.indirect_sampling()) HSSopts.set_user_defined_random(true);
    // _H->compress(mult, elem, HSSopts);

    TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_compress);
    _H->compress(mult, elem, opts.HSS_options());
    params::compression_flops += params::flops - f0;
    TIMER_STOP(t_compress);

    // // _H->print_info(std::cout, 0, 0);
    // auto Hdense = _H->dense(this->ctxt); //.print("_H");
    // Hdense.scaled_add(scalar_t(-1.), FI);
    // auto errH = Hdense.normF() / FI.normF();
    // if (mpi_rank(this->front_comm)==0)
    //   std::cout << "||FI-Hdense||/||FI|| = " << errH << std::endl;
    // if (errH > opts.HSS_options().rel_tol() * 1e2) {
    //   Hdense.print("Hdense_err");
    //   exit(1);
    // }

    if (this->lchild) this->lchild->release_work_memory();
    if (this->rchild) this->rchild->release_work_memory();

    if (this->dim_sep()) {
      if (etree_level > 0) {
        auto f0 = params::flops;
        TIMER_TIME(TaskType::HSS_PARTIALLY_FACTOR, 0, t_pfact);
        _ULV = _H->partial_factor();
        TIMER_STOP(t_pfact);
        params::ULV_factor_flops += params::flops - f0;
        auto f1 = params::flops;
        TIMER_TIME(TaskType::HSS_COMPUTE_SCHUR, 0, t_comp_schur);
        _H->Schur_update(_ULV, _Theta, _Vhat, _DUB01, _Phi);
        if (_Theta.cols() < _Phi.cols()) {
          _VhatCPhiC = DistM_t(_Phi.ctxt(), _Vhat.cols(), _Phi.rows());
          gemm(Trans::C, Trans::C, scalar_t(1.), _Vhat, _Phi,
               scalar_t(0.), _VhatCPhiC);
        } else {
          _ThetaVhatC = DistM_t(_Theta.ctxt(), _Theta.rows(), _Vhat.rows());
          gemm(Trans::N, Trans::C, scalar_t(1.), _Theta, _Vhat,
               scalar_t(0.), _ThetaVhatC);
        }
        TIMER_STOP(t_comp_schur);
        params::schur_flops += params::flops - f1;
      } else {
        auto f0 = params::flops;
        TIMER_TIME(TaskType::HSS_FACTOR, 0, t_fact);
        _ULV = _H->factor();
        TIMER_STOP(t_fact);
        params::ULV_factor_flops += params::flops - f0;
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
   int etree_level) const {
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    if (this->visit(this->lchild))
      this->lchild->forward_multifrontal_solve
        (bloc, bdist, CBl, seqCBl, etree_level+1);
    if (this->visit(this->rchild))
      this->rchild->forward_multifrontal_solve
        (bloc, bdist, CBr, seqCBr, etree_level+1);
    DistM_t& b = bdist[this->sep];
    bupd = DistM_t(_H->ctxt(), this->dim_upd(), b.cols());
    bupd.zero();
    this->extend_add_b(b, bupd, CBl, CBr, seqCBl, seqCBr);
    if (etree_level) {
      if (_Theta.cols() && _Phi.cols()) {
        TIMER_TIME(TaskType::SOLVE_LOWER, 0, t_reduce);
        _ULVwork = std::unique_ptr<HSS::WorkSolveMPI<scalar_t>>
          (new HSS::WorkSolveMPI<scalar_t>());
        DistM_t lb(_H->child(0)->ctxt(_H->ctxt_loc()), b.rows(), b.cols());
        copy(b.rows(), b.cols(), b, 0, 0, lb, 0, 0, this->ctxt_all);
        _H->child(0)->forward_solve(_ULV, *_ULVwork, lb, true);
        DistM_t rhs(_H->ctxt(), _Theta.cols(), bupd.cols());
        copy(rhs.rows(), rhs.cols(), _ULVwork->reduced_rhs, 0, 0,
             rhs, 0, 0, this->ctxt_all);
        _ULVwork->reduced_rhs.clear();
        if (this->dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), _Theta, rhs,
               scalar_t(1.), bupd);
        TIMER_STOP(t_reduce);
      }
    } else {
      TIMER_TIME(TaskType::SOLVE_LOWER_ROOT, 0, t_solve);
      DistM_t lb(_H->ctxt(), b.rows(), b.cols());
      copy(b.rows(), b.cols(), b, 0, 0, lb, 0, 0, this->ctxt_all);
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
    DistM_t& y = ydist[this->sep];
    TIMER_TIME(TaskType::SOLVE_UPPER, 0, t_expand);
    if (etree_level) {
      if (_Phi.cols() && _Theta.cols()) {
        DistM_t ly
          (_H->child(0)->ctxt(_H->ctxt_loc()), y.rows(), y.cols());
        copy(y.rows(), y.cols(), y, 0, 0, ly, 0, 0, this->ctxt_all);
        if (this->dim_upd()) {
          // TODO can these copies be avoided??
          DistM_t wx
            (_H->ctxt(), _Phi.cols(), yupd.cols(), _ULVwork->x, this->ctxt_all);
          DistM_t yupdHctxt
            (_H->ctxt(), this->dim_upd(), y.cols(), yupd, this->ctxt_all);
          gemm(Trans::C, Trans::N, scalar_t(-1.), _Phi, yupdHctxt,
               scalar_t(1.), wx);
          copy(wx.rows(), wx.cols(), wx, 0, 0,
               _ULVwork->x, 0, 0, this->ctxt_all);
        }
        _H->child(0)->backward_solve(_ULV, *_ULVwork, ly);
        copy(y.rows(), y.cols(), ly, 0, 0, y, 0, 0, this->ctxt_all);
        _ULVwork.reset();
      }
    } else {
      DistM_t ly(_H->ctxt(), y.rows(), y.cols());
      copy(y.rows(), y.cols(), y, 0, 0, ly, 0, 0, this->ctxt_all);
      _H->backward_solve(_ULV, *_ULVwork, ly);
      copy(y.rows(), y.cols(), ly, 0, 0, y, 0, 0, this->ctxt_all);
    }
    TIMER_STOP(t_expand);
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    this->extract_b(y, yupd, CBl, CBr, seqCBl, seqCBr);
    if (this->visit(this->lchild))
      this->lchild->backward_multifrontal_solve
        (yloc, ydist, CBl, seqCBl, etree_level+1);
    if (this->visit(this->rchild))
      this->rchild->backward_multifrontal_solve
        (yloc, ydist, CBr, seqCBr, etree_level+1);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::element_extraction
  (const SpMat_t& A, const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J, DistM_t& B) {
    if (I.empty() || J.empty()) return;
    std::vector<std::size_t> gI, gJ;
    gI.reserve(I.size());
    gJ.reserve(J.size());
    const std::size_t dsep = this->dim_sep();
    for (auto i : I) {
      assert(i < std::size_t(this->dim_blk()));
      gI.push_back((i < dsep) ? i+this->sep_begin : this->upd[i-dsep]);
    }
    for (auto j : J) {
      assert(j < std::size_t(this->dim_blk()));
      gJ.push_back((j < dsep) ? j+this->sep_begin : this->upd[j-dsep]);
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
    if (this->front_comm == MPI_COMM_NULL || !this->dim_upd()) return;
    TIMER_TIME(TaskType::HSS_EXTRACT_SCHUR, 3, t_ex_schur);
    std::vector<std::size_t> lI, lJ, oI, oJ;
    this->find_upd_indices(J, lJ, oJ);
    this->find_upd_indices(I, lI, oI);

    // TODO extract from child(1) directly
    auto gI = lI;  for (auto& i : gI) i += this->dim_sep();
    auto gJ = lJ;  for (auto& j : gJ) j += this->dim_sep();
    DistM_t e = _H->extract(gI, gJ, this->ctxt, this->np_rows(),
                            this->np_cols());

    if (_Theta.cols() < _Phi.cols()) {
      DistM_t tr(this->ctxt, lI.size(), _Theta.cols(),
                 _Theta.extract_rows(lI, this->front_comm), this->ctxt_all);
      DistM_t tc(this->ctxt, _VhatCPhiC.rows(), lJ.size(),
                 _VhatCPhiC.extract_cols(lJ, this->front_comm), this->ctxt_all);
      gemm(Trans::N, Trans::N, scalar_t(-1), tr, tc, scalar_t(1.), e);
    } else {
      DistM_t tr(this->ctxt, lI.size(), _ThetaVhatC.cols(),
                 _ThetaVhatC.extract_rows(lI, this->front_comm), this->ctxt_all);
      DistM_t tc(this->ctxt, lJ.size(), _Phi.cols(),
                 _Phi.extract_rows(lJ, this->front_comm), this->ctxt_all);
      gemm(Trans::N, Trans::C, scalar_t(-1), tr, tc, scalar_t(1.), e);
    }

    auto P = mpi_nprocs(this->front_comm);
    std::vector<std::vector<scalar_t>> sbuf(P);
    ExtAdd::extend_copy_to_buffers(e, oI, oJ, B, sbuf);
    scalar_t* rbuf = nullptr, **pbuf = nullptr;
    all_to_all_v(sbuf, rbuf, pbuf, this->front_comm);
    ExtAdd::extend_copy_from_buffers(B, oI, oJ, e, pbuf);
    delete[] rbuf;
    delete[] pbuf;
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
    if (this->visit(this->lchild))
      mr = std::max(mr, this->lchild->maximum_rank(task_depth));
    if (this->visit(this->rchild))
      mr = std::max(mr, this->rchild->maximum_rank(task_depth));
    return mr;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::set_HSS_partitioning
  (const SPOptions<scalar_t>& opts, const HSS::HSSPartitionTree& sep_tree,
   bool is_root) {
    //if (!this->active()) return;
    if (this->front_comm == MPI_COMM_NULL) return;
    assert(sep_tree.size == this->dim_sep());
    if (is_root)
      _H = std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>>
        (new HSS::HSSMatrixMPI<scalar_t>
         (sep_tree, opts.HSS_options(), this->comm()));
    else {
      HSS::HSSPartitionTree hss_tree(this->dim_blk());
      hss_tree.c.reserve(2);
      hss_tree.c.push_back(sep_tree);
      hss_tree.c.emplace_back(this->dim_upd());
      hss_tree.c.back().refine(opts.HSS_options().leaf_size());
      _H = std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>>
        (new HSS::HSSMatrixMPI<scalar_t>
         (hss_tree, opts.HSS_options(), this->comm()));
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::bisection_partitioning
  (const SPOptions<scalar_t>& opts, integer_t* sorder,
   bool isroot, int task_depth) {
    if (this->visit(this->lchild))
      this->lchild->bisection_partitioning(opts, sorder, false, task_depth);
    if (this->visit(this->rchild))
      this->rchild->bisection_partitioning(opts, sorder, false, task_depth);

    HSS::HSSPartitionTree sep_tree(this->dim_sep());
    sep_tree.refine(opts.HSS_options().leaf_size());

    std::cout << "TODO FrontalMatrixHSSMPI::bisection_partitioning"
              << std::endl;
    for (integer_t i=this->sep_begin; i<this->sep_end; i++) sorder[i] = -i;

    // TODO also communicate the tree to everyone working on this front!!
    // see code in EliminationTreeMPIDist

    if (isroot)
      _H = std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>>
        (new HSS::HSSMatrixMPI<scalar_t>
         (sep_tree, opts.HSS_options(), this->comm()));
    else {
      HSS::HSSPartitionTree hss_tree(this->dim_blk());
      hss_tree.c.reserve(2);
      hss_tree.c.push_back(sep_tree);
      hss_tree.c.emplace_back(this->dim_upd());
      hss_tree.c.back().refine(opts.HSS_options().leaf_size());
      _H = std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>>
        (new HSS::HSSMatrixMPI<scalar_t>
         (hss_tree, opts.HSS_options(), this->comm()));
    }
  }

} // end namespace strumpack

#endif //FRONTAL_MATRIX_HSS_MPI_HPP
