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
    void random_sampling
    (const SpMat_t& A, const Opts& opts,
     DistM_t& R, DistM_t& Sr, DistM_t& Sc, int etree_level);
    void sample_CB
    (const Opts& opts, const DistM_t& R, DistM_t& Sr,
     DistM_t& Sc, F_t* pa) const override;
    void sample_children_CB
    (const Opts& opts,
     DistM_t& R, DistM_t& Sr, DistM_t& Sc);
    void sample_children_CB_seqseq
    (const Opts& opts, const DistM_t& R,
     DistM_t& Sr, DistM_t& Sc);
    void skinny_extend_add
    (DistM_t& cSrl, DistM_t& cScl, DistM_t& cSrr, DistM_t& cScr,
     DistM_t& Sr, DistM_t& Sc);

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
    std::unique_ptr<HODLR::HODLRMatrix<scalar_t>> F11_, F22_;

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
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::random_sampling
  (const SpMat_t& A, const Opts& opts, DistM_t& R,
   DistM_t& Sr, DistM_t& Sc, int etree_level) {
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
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::sample_CB
  (const Opts& opts, const DistM_t& R,
   DistM_t& Sr, DistM_t& Sc, F_t* pa) const {
    if (!dim_upd()) return;
    if (Comm().is_null()) return;
    auto b = R.cols();
    Sr = DistM_t(grid(), dim_upd(), b);
    Sc = DistM_t(grid(), dim_upd(), b);
    TIMER_TIME(TaskType::HSS_SCHUR_PRODUCT, 2, t_sprod);

    std::cout << "TODO: sample with child CB" << std::endl;
    // _H->Schur_product_direct
    //   (_Theta, _Vhat, _DUB01, _Phi, _ThetaVhatC, _VhatCPhiC, R, Sr, Sc);

    TIMER_STOP(t_sprod);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::sample_children_CB
  (const Opts& opts, DistM_t& R, DistM_t& Sr, DistM_t& Sc) {
    if (!lchild_ && !rchild_) return;

    std::cout << "TODO: call sample_CB on the children" << std::endl;
  }

  // TODO avoid duplication with HSS
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::skinny_extend_add
  (DistM_t& cSrl, DistM_t& cScl, DistM_t& cSrr, DistM_t& cScr,
   DistM_t& Sr, DistM_t& Sc) {
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    if (cSrl.active())
      ExtAdd::skinny_extend_add_copy_to_buffers
        (cSrl, cScl, sbuf, this, lchild_->upd_to_parent(this));
    if (cSrr.active())
      ExtAdd::skinny_extend_add_copy_to_buffers
        (cSrr, cScr, sbuf, this, rchild_->upd_to_parent(this));
    std::vector<scalar_t> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    if (lchild_) // unpack left child contribution
      ExtAdd::skinny_extend_add_copy_from_buffers
        (Sr, Sc, pbuf.data(), this, lchild_);
    if (rchild_) // unpack right child contribution
      ExtAdd::skinny_extend_add_copy_from_buffers
        (Sr, Sc, pbuf.data()+this->master(rchild_), this, rchild_);
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

    // TODO implement the sampling routine!!
    auto mult = [&](DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc) {
      TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
      //random_sampling(A, opts, R, Sr, Sc, etree_level);
      TIMER_STOP(t_sampling);
    };
    //F11_->compress(mult);
    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();

    if (dim_sep()) {
      if (etree_level > 0) {
        TIMER_TIME(TaskType::HSS_PARTIALLY_FACTOR, 0, t_pfact);
        F11_->factor();
        TIMER_STOP(t_pfact);
        TIMER_TIME(TaskType::HSS_COMPUTE_SCHUR, 0, t_comp_schur);

        // TODO construct F12_ and F21_
        // TODO construct F22 as (F22 - F21 inv(F11) F12)

        std::cout << "TODO HODLR Schur update" << std::endl;
        TIMER_STOP(t_comp_schur);
      } else {
        TIMER_TIME(TaskType::HSS_FACTOR, 0, t_fact);
        F11_->factor();
        TIMER_STOP(t_fact);
      }
    }
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
    //bupd = DistM_t(_H->grid(), dim_upd(), b.cols());
    bupd.zero();
    this->extend_add_b(b, bupd, CBl, CBr, seqCBl, seqCBr);

    std::cout << "TODO solve fwd" << std::endl;

    // if (this->dim_sep()) {
    //   TIMER_TIME(TaskType::SOLVE_LOWER, 0, t_s);
    //   b.laswp(piv, true);
    //   if (b.cols() == 1) {
    //     trsv(UpLo::L, Trans::N, Diag::U, F11_, b);
    //     if (this->dim_upd())
    //       gemv(Trans::N, scalar_t(-1.), F21_, b, scalar_t(1.), bupd);
    //   } else {
    //     trsm(Side::L, UpLo::L, Trans::N, Diag::U, scalar_t(1.), F11_, b);
    //     if (this->dim_upd())
    //       gemm(Trans::N, Trans::N, scalar_t(-1.), F21_, b, scalar_t(1.), bupd);
    //   }
    //   TIMER_STOP(t_s);
    // }

  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
   int etree_level) const {
    DistM_t& y = ydist[this->sep_];

    std::cout << "TODO solve bwd" << std::endl;

    // if (this->dim_sep()) {
    //   TIMER_TIME(TaskType::SOLVE_UPPER, 0, t_s);
    //   if (y.cols() == 1) {
    //     if (this->dim_upd())
    //       gemv(Trans::N, scalar_t(-1.), F12_, yupd, scalar_t(1.), y);
    //     trsv(UpLo::U, Trans::N, Diag::N, F11_, y);
    //   } else {
    //     if (this->dim_upd())
    //       gemm(Trans::N, Trans::N, scalar_t(-1.), F12_, yupd, scalar_t(1.), y);
    //     trsm(Side::L, UpLo::U, Trans::N, Diag::N, scalar_t(1.), F11_, y);
    //   }
    //   TIMER_STOP(t_s);
    // }

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
    std::cout << "TODO create HODLR matrix hierarchy" << std::endl;
    if (is_root) {
      // _H = std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>>
      //   (new HSS::HSSMatrixMPI<scalar_t>
      //    (sep_tree, grid(), opts.HSS_options()));
    } else {
      HSS::HSSPartitionTree hss_tree(dim_blk());
      hss_tree.c.reserve(2);
      hss_tree.c.push_back(sep_tree);
      hss_tree.c.emplace_back(dim_upd());
      hss_tree.c.back().refine(opts.HSS_options().leaf_size());
      // _H = std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>>
      //   (new HSS::HSSMatrixMPI<scalar_t>
      //    (hss_tree, grid(), opts.HSS_options()));
    }
  }


} // end namespace strumpack

#endif //FRONTAL_MATRIX_HODLR_MPI_HPP
