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
#ifndef FRONTAL_MATRIX_HODLR_MPI_HPP
#define FRONTAL_MATRIX_HODLR_MPI_HPP

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
#include "HODLR/HODLRMatrix.hpp"
#include "HODLR/LRBFMatrix.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  class FrontalMatrixHODLRMPI : public FrontalMatrixMPI<scalar_t,integer_t> {
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
    FrontalMatrixHODLRMPI
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd, const MPIComm& comm, int _total_procs);
    FrontalMatrixHODLRMPI(const FrontalMatrixHODLRMPI&) = delete;
    FrontalMatrixHODLRMPI& operator=(FrontalMatrixHODLRMPI const&) = delete;

    void release_work_memory() override;
    void extend_add(int task_depth) {}

    void sample_CB
    (Trans op, const DistM_t& R, DistM_t& S, F_t* pa) const override;
    void sample_children_CB(Trans op, const DistM_t& R, DistM_t& S);
    void skinny_extend_add(DistM_t& cSl, DistM_t& cSr, DistM_t& S);

    void multifrontal_factorization
    (const SpMat_t& A, const Opts& opts,
     int etree_level=0, int task_depth=0) override;

    void forward_multifrontal_solve
    (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
     int etree_level=0) const override;
    void backward_multifrontal_solve
    (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
     int etree_level=0) const override;

    long long node_factor_nonzeros() const;
    integer_t maximum_rank(int task_depth) const;
    std::string type() const override { return "FrontalMatrixHODLRMPI"; }

    void set_HODLR_partitioning
    (const Opts& opts, const HSS::HSSPartitionTree& sep_tree,
     bool is_root) override;

  private:
    HODLR::HODLRMatrix<scalar_t> F11_;
    HODLR::LRBFMatrix<scalar_t> F12_, F21_;
    std::unique_ptr<HODLR::HODLRMatrix<scalar_t>> F22_;

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
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::FrontalMatrixHODLRMPI
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd, const MPIComm& comm, int total_procs)
    : FrontalMatrixMPI<scalar_t,integer_t>
    (sep, sep_begin, sep_end, upd, comm, total_procs) {
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::release_work_memory() {
    F22_.reset(nullptr);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::sample_CB
  (Trans op, const DistM_t& R, DistM_t& S, F_t* pa) const {
    if (!dim_upd()) return;
    if (Comm().is_null()) return;
    auto b = R.cols();
    S = DistM_t(grid(), dim_upd(), b);
    TIMER_TIME(TaskType::HSS_SCHUR_PRODUCT, 2, t_sprod);
    F22_->mult(op, R, S);
    TIMER_STOP(t_sprod);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::sample_children_CB
  (Trans op, const DistM_t& R, DistM_t& S) {
    if (!lchild_ && !rchild_) return;
    DistM_t Sl, Sr, Rl, Rr;
    DenseM_t seqSl, seqSr, seqRl, seqRr;
    if (lchild_) {
      lchild_->extract_from_R2D(R, Rl, seqRl, this, visit(lchild_));
      seqSl = DenseM_t(seqRl.rows(), seqRl.cols());
      seqSl.zero();
    }
    if (rchild_) {
      rchild_->extract_from_R2D(R, Rr, seqRr, this, visit(rchild_));
      seqSr = DenseM_t(seqRr.rows(), seqRr.cols());
      seqSr.zero();
    }
    if (visit(lchild_)) lchild_->sample_CB(op, Rl, Sl, seqRl, seqSl, this);
    if (visit(rchild_)) rchild_->sample_CB(op, Rr, Sr, seqRr, seqSr, this);
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    if (visit(lchild_)) lchild_->skinny_ea_to_buffers(Sl, seqSl, sbuf, this);
    if (visit(rchild_)) rchild_->skinny_ea_to_buffers(Sr, seqSr, sbuf, this);
    std::vector<scalar_t> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    if (lchild_)
      lchild_->skinny_ea_from_buffers(S, pbuf.data(), this);
    if (rchild_)
      rchild_->skinny_ea_from_buffers
        (S, pbuf.data()+this->master(rchild_), this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const Opts& opts, int etree_level, int task_depth) {
    if (visit(lchild_))
      lchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    if (visit(rchild_))
      rchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    if (!dim_blk()) return;

    const auto dsep = dim_sep();
    const auto dupd = dim_upd();

    auto sample_front = [&](Trans op, const DistM_t& R, DistM_t& S) {
      S.zero();
      DistM_t Sdummy(grid(), dsep+dupd, R.cols());
      if (op == Trans::N)
        A.front_multiply_2d // TODO only perform op, remove dummy
          (this->sep_begin_, this->sep_end_, this->upd_, R, S, Sdummy, 0);
      else
        A.front_multiply_2d // TODO only perform op, remove dummy
          (this->sep_begin_, this->sep_end_, this->upd_, R, Sdummy, S, 0);
      sample_children_CB(op, R, S);
    };

    auto sample_F11 = [&](Trans op, const DenseM_t& R1, DenseM_t& S1) {
      TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
      auto n = R1.cols();
      DistM_t R(grid(), dsep+dupd, n), S(grid(), dsep+dupd, n);
      DistMW_t R1dist(dsep, n, R, 0, 0);
      DistMW_t(dupd, n, R, dsep, 0).zero();
      F11_.redistribute_1D_to_2D(R1, R1dist);
      sample_front(op, R, S);
      F11_.redistribute_2D_to_1D(DistMW_t(dsep, n, S, 0, 0), S1);
      TIMER_STOP(t_sampling);
    };
    F11_.compress(sample_F11);
    TIMER_TIME(TaskType::HSS_FACTOR, 0, t_fact);
    F11_.factor();
    TIMER_STOP(t_fact);

    if (dupd) {
      auto sample_F12 = [&]
        (Trans op, scalar_t a, const DenseM_t& Rl, scalar_t b, DenseM_t& Sl) {
        TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
        auto n = Rl.cols();
        DistM_t R(grid(), dsep+dupd, n), S(grid(), dsep+dupd, n);
        if (op == Trans::N) {
          DistMW_t R2dist(dupd, n, R, dsep, 0);
          DistMW_t(dsep, n, R, 0, 0).zero();
          F12_.redistribute_1D_to_2D(Rl, R2dist, F12_.cdist());
          sample_front(op, R, S);
          F12_.redistribute_2D_to_1D
            (a, DistMW_t(dsep, n, S, 0, 0), b, Sl, F12_.rdist());
        } else {
          DistMW_t R1dist(dsep, n, R, 0, 0);
          DistMW_t(dupd, n, R, dsep, 0).zero();
          F12_.redistribute_1D_to_2D(Rl, R1dist, F12_.rdist());
          sample_front(op, R, S);
          F12_.redistribute_2D_to_1D
          (a, DistMW_t(dupd, n, S, dsep, 0), b, Sl, F12_.cdist());
        }
        TIMER_STOP(t_sampling);
      };
      auto sample_F21 = [&]
        (Trans op, scalar_t a, const DenseM_t& Rl, scalar_t b, DenseM_t& Sl) {
        TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
        auto n = Rl.cols();
        DistM_t R(grid(), dsep+dupd, n), S(grid(), dsep+dupd, n);
        if (op == Trans::N) {
          DistMW_t R1dist(dsep, n, R, 0, 0);
          DistMW_t(dupd, n, R, dsep, 0).zero();
          F21_.redistribute_1D_to_2D(Rl, R1dist, F21_.cdist());
          sample_front(op, R, S);
          F12_.redistribute_2D_to_1D
            (a, DistMW_t(dupd, n, S, dsep, 0), b, Sl, F21_.rdist());
        } else {
          DistMW_t R2dist(dupd, n, R, dsep, 0);
          DistMW_t(dsep, n, R, 0, 0).zero();
          F21_.redistribute_1D_to_2D(Rl, R2dist, F21_.rdist());
          sample_front(op, R, S);
          F12_.redistribute_2D_to_1D
            (a, DistMW_t(dsep, n, S, 0, 0), b, Sl, F21_.cdist());
        }
        TIMER_STOP(t_sampling);
      };
      F12_ = HODLR::LRBFMatrix<scalar_t>(F11_, *F22_);
      F12_.compress(sample_F12);
      F21_ = HODLR::LRBFMatrix<scalar_t>(*F22_, F11_);
      F21_.compress(sample_F21);

      auto sample_F22 = [&](Trans op, const DenseM_t& R2, DenseM_t& S2) {
        TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
        auto n = R2.cols();
        DistM_t R(grid(), dsep+dupd, n), S(grid(), dsep+dupd, n);
        DistMW_t(dsep, n, R, 0, 0).zero();
        DistMW_t R2dist(dupd, n, R, dsep, 0);
        F22_->redistribute_1D_to_2D(R2, R2dist);
        sample_front(op, R, S);
        F22_->redistribute_2D_to_1D(DistMW_t(dupd, n, S, dsep, 0), S2);
        DenseM_t F12R2(F12_.lrows(), R2.cols()),
        invF11F12R2(F11_.lrows(), R2.cols()), S2tmp(S2.rows(), S2.cols());
        if (op == Trans::N) {
          F12_.mult(op, R2, F12R2);
          F11_.inv_mult(op, F12R2, invF11F12R2);
          //F11_.solve(F12R, invF11F12R);
          F21_.mult(op, invF11F12R2, S2tmp);
        } else {
          F21_.mult(op, R2, F12R2);
          F11_.inv_mult(op, F12R2, invF11F12R2);
          F12_.mult(op, invF11F12R2, S2tmp);
        }
        S2.scaled_add(scalar_t(-1.), S2tmp);
        TIMER_STOP(t_sampling);
      };
      F22_->compress(sample_F22);
    }
    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::forward_multifrontal_solve
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
    bupd = DistM_t(grid(), dim_upd(), b.cols());
    bupd.zero();
    this->extend_add_b(b, bupd, CBl, CBr, seqCBl, seqCBr);
    if (this->dim_sep()) {
      TIMER_TIME(TaskType::SOLVE_LOWER, 0, t_s);
      DistM_t rhs(b);
      F11_.solve(rhs, b);
      if (this->dim_upd()) {
        DistM_t tmp(bupd.grid(), bupd.rows(), bupd.cols());
        F21_.mult(Trans::N, b, tmp);
        bupd.scaled_add(scalar_t(-1.), tmp);
      }
      TIMER_STOP(t_s);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
   int etree_level) const {
    DistM_t& y = ydist[this->sep_];
    if (this->dim_sep() && this->dim_upd()) {
      TIMER_TIME(TaskType::SOLVE_UPPER, 0, t_s);
      DistM_t tmp(y.grid(), y.rows(), y.cols());
      F12_.mult(Trans::N, yupd, tmp);
      y.scaled_add(scalar_t(-1.), tmp);
      TIMER_STOP(t_s);
    }
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

  template<typename scalar_t,typename integer_t> integer_t
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::maximum_rank
  (int task_depth) const {
    integer_t mr = -1; //_H->max_rank();
    if (visit(lchild_))
      mr = std::max(mr, lchild_->maximum_rank(task_depth));
    if (visit(rchild_))
      mr = std::max(mr, rchild_->maximum_rank(task_depth));
    return mr;
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::node_factor_nonzeros() const {
    // TODO does this return only when root??
    return -1; //(_H ? _H->nonzeros() : 0) + _ULV.nonzeros() +
    //_Theta.nonzeros() + _Phi.nonzeros();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::set_HODLR_partitioning
  (const Opts& opts, const HSS::HSSPartitionTree& sep_tree, bool is_root) {
    //if (!this->active()) return;
    if (Comm().is_null()) return;
    assert(sep_tree.size == dim_sep());
    F11_ = std::move
      (HODLR::HODLRMatrix<scalar_t>
       (Comm(), sep_tree, opts.HODLR_options()));
    if (!is_root && dim_upd()) {
      HSS::HSSPartitionTree CB_tree(dim_upd());
      CB_tree.refine(opts.HODLR_options().leaf_size());
      F22_ = std::unique_ptr<HODLR::HODLRMatrix<scalar_t>>
        (new HODLR::HODLRMatrix<scalar_t>
         (Comm(), CB_tree, opts.HODLR_options()));
    }
  }

} // end namespace strumpack

#endif //FRONTAL_MATRIX_HODLR_MPI_HPP
