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
#ifndef FRONTAL_MATRIX_BLR_MPI_HPP
#define FRONTAL_MATRIX_BLR_MPI_HPP

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "misc/TaskTimer.hpp"
#include "misc/MPIWrapper.hpp"
#include "dense/DistributedMatrix.hpp"
#include "BLR/BLRMatrixMPI.hpp"
#include "CompressedSparseMatrix.hpp"
#include "MatrixReordering.hpp"
#include "FrontalMatrixMPI.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class ExtractFront;

  template<typename scalar_t,typename integer_t>
  class FrontalMatrixBLRMPI : public FrontalMatrixMPI<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using DistMW_t = DistributedMatrixWrapper<scalar_t>;
    using BLRMPI_t = BLR::BLRMatrixMPI<scalar_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using FBLRMPI_t = FrontalMatrixBLRMPI<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using Opts_t = SPOptions<scalar_t>;
    using ExtAdd = ExtendAdd<scalar_t,integer_t>;
    template<typename _scalar_t,typename _integer_t> friend class ExtendAdd;

  public:
    FrontalMatrixBLRMPI
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd, const MPIComm& comm, int P);

    void release_work_memory() override;
    void build_front(const SpMat_t& A);
    void partial_factorization(const Opts_t& opts);

    void extend_add();
    void extend_add_copy_to_buffers
    (std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const override;

    void sample_CB
    (const DistM_t& R, DistM_t& Sr, DistM_t& Sc, F_t* pa) const override;

    void multifrontal_factorization
    (const SpMat_t& A, const Opts_t& opts,
     int etree_level=0, int task_depth=0) override;

    void forward_multifrontal_solve
    (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
     int etree_level=0) const override;
    void backward_multifrontal_solve
    (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
     int etree_level=0) const override;

    void extract_CB_sub_matrix_2d
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DistM_t& B) const override;

    std::string type() const override { return "FrontalMatrixBLRMPI"; }

    void partition
    (const Opts_t& opts, const SpMat_t& A,
     integer_t* sorder, bool is_root, int task_depth) override;

  private:
    DistM_t F11_, F12_, F21_, F22_;
    BLRMPI_t F11blr_, F12blr_, F21blr_;
    std::vector<int> piv_;
    std::vector<std::size_t> sep_tiles_;
    std::vector<std::size_t> upd_tiles_;
    DenseMatrix<bool> adm_;
    BLR::ProcessorGrid2D blr_grid_;

    long long node_factor_nonzeros() const override;

    using FrontalMatrix<scalar_t,integer_t>::lchild_;
    using FrontalMatrix<scalar_t,integer_t>::rchild_;
    using FrontalMatrix<scalar_t,integer_t>::sep_begin_;
    using FrontalMatrix<scalar_t,integer_t>::sep_end_;
    using FrontalMatrix<scalar_t,integer_t>::dim_sep;
    using FrontalMatrix<scalar_t,integer_t>::dim_upd;
    using FrontalMatrixMPI<scalar_t,integer_t>::visit;
    using FrontalMatrixMPI<scalar_t,integer_t>::Comm;
    using FrontalMatrixMPI<scalar_t,integer_t>::grid;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixBLRMPI<scalar_t,integer_t>::FrontalMatrixBLRMPI
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd, const MPIComm& comm, int P)
    : FrontalMatrixMPI<scalar_t,integer_t>
    (sep, sep_begin, sep_end, upd, comm, P), blr_grid_(Comm()) {}


  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::release_work_memory() {
    F22_.clear(); // remove the update block
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::extend_add() {
    if (!lchild_ && !rchild_) return;
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    for (auto& ch : {lchild_.get(), rchild_.get()}) {
      if (ch && Comm().is_root()) {
        STRUMPACK_FLOPS
          (static_cast<long long int>(ch->dim_upd())*ch->dim_upd());
      }
      if (visit(ch)) ch->extend_add_copy_to_buffers(sbuf, this);
    }
    std::vector<scalar_t> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    for (auto& ch : {lchild_.get(), rchild_.get()})
      if (ch)
        ch->extend_add_copy_from_buffers
          (F11_, F12_, F21_, F22_, pbuf.data()+this->master(ch), this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtAdd::extend_add_copy_to_buffers
      (F22_, sbuf, pa, this->upd_to_parent(pa));
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::build_front
  (const SpMat_t& A) {
    const auto dupd = dim_upd();
    const auto dsep = dim_sep();
    if (dsep) {
      F11_ = DistM_t(grid(), dsep, dsep);
      using ExFront = ExtractFront<scalar_t,integer_t>;
      ExFront::extract_F11(F11_, A, sep_begin_, dsep);
      if (dim_upd()) {
        F12_ = DistM_t(grid(), dsep, dupd);
        ExFront::extract_F12(F12_, A, sep_begin_, sep_end_, this->upd_);
        F21_ = DistM_t(grid(), dupd, dsep);
        ExFront::extract_F21(F21_, A, sep_end_, sep_begin_, this->upd_);
      }
    }
    if (dupd) {
      F22_ = DistM_t(grid(), dupd, dupd);
      F22_.zero();
    }
    extend_add();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::partial_factorization
  (const Opts_t& opts) {
    if (dim_sep() && grid()->active()) {
      F11blr_ = BLRMPI_t(blr_grid_, sep_tiles_, adm_,
                         // [&](std::size_t i, std::size_t j) -> bool {
                         //   return adm_(i,j); },
                         F11_, piv_, opts.BLR_options());
      // TODO flops?
      F11_.clear();
      if (dim_upd()) {
        F12_.laswp(piv_, true);
        trsm(Side::L, UpLo::L, Trans::N, Diag::U, scalar_t(1.), F11blr_, F12_);
        F12blr_ = BLRMPI_t(blr_grid_, sep_tiles_, upd_tiles_, F12_, opts.BLR_options());
        F12_.clear();
        trsm(Side::R, UpLo::U, Trans::N, Diag::N, scalar_t(1.), F11blr_, F21_);
        F21blr_ = BLRMPI_t(blr_grid_, upd_tiles_, sep_tiles_, F21_, opts.BLR_options());
        F21_.clear();
        gemm(Trans::N, Trans::N, scalar_t(-1.), F21blr_, F12blr_, scalar_t(1.), F22_);
        // TODO flops?
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const Opts_t& opts, int etree_level, int task_depth) {
    if (visit(lchild_))
      lchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    if (visit(rchild_))
      rchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    build_front(A);
    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
    partial_factorization(opts);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
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
    bupd = DistM_t(grid(), dim_upd(), b.cols());
    bupd.zero();
    this->extend_add_b(b, bupd, CBl, CBr, seqCBl, seqCBr);
    if (dim_sep()) {
      TIMER_TIME(TaskType::SOLVE_LOWER, 0, t_s);
      b.laswp(piv_, true);
      if (b.cols() == 1) {
        trsv(UpLo::L, Trans::N, Diag::U, F11blr_, b);
        if (dim_upd())
          gemv(Trans::N, scalar_t(-1.), F21blr_, b, scalar_t(1.), bupd);
      } else {
        trsm(Side::L, UpLo::L, Trans::N, Diag::U, scalar_t(1.), F11blr_, b);
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F21blr_, b, scalar_t(1.), bupd);
      }
      TIMER_STOP(t_s);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
   int etree_level) const {
    DistM_t& y = ydist[this->sep_];
    if (dim_sep()) {
      TIMER_TIME(TaskType::SOLVE_UPPER, 0, t_s);
      if (y.cols() == 1) {
        if (dim_upd())
          gemv(Trans::N, scalar_t(-1.), F12blr_, yupd, scalar_t(1.), y);
        trsv(UpLo::U, Trans::N, Diag::N, F11blr_, y);
      } else {
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F12blr_, yupd, scalar_t(1.), y);
        trsm(Side::L, UpLo::U, Trans::N, Diag::N, scalar_t(1.), F11blr_, y);
      }
      TIMER_STOP(t_s);
    }
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
  FrontalMatrixBLRMPI<scalar_t,integer_t>::extract_CB_sub_matrix_2d
  (const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J, DistM_t& B) const {
    if (Comm().is_null()) return;
    std::vector<std::size_t> lJ, oJ, lI, oI;
    this->find_upd_indices(J, lJ, oJ);
    this->find_upd_indices(I, lI, oI);
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    ExtAdd::extract_copy_to_buffers(F22_, lI, lJ, oI, oJ, B, sbuf);
    std::vector<scalar_t> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    ExtAdd::extract_copy_from_buffers(B, lI, lJ, oI, oJ, F22_, pbuf);
  }

  /**
   *  Sr = F22 * R
   *  Sc = F22^* * R
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::sample_CB
  (const DistM_t& R, DistM_t& Sr, DistM_t& Sc,
   FrontalMatrix<scalar_t,integer_t>* pa) const {
    if (F11_.active() || F22_.active()) {
      auto b = R.cols();
      Sr = DistM_t(grid(), dim_upd(), b);
      Sc = DistM_t(grid(), dim_upd(), b);
      gemm(Trans::N, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.), Sr);
      gemm(Trans::C, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.), Sc);
      STRUMPACK_CB_SAMPLE_FLOPS
        (gemm_flops(Trans::N, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.)) +
         gemm_flops(Trans::C, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.)));
    }
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixBLRMPI<scalar_t,integer_t>::node_factor_nonzeros() const {
    return F11_.nonzeros() + F12_.nonzeros() + F21_.nonzeros();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::partition
  (const Opts_t& opts, const SpMat_t& A,
   integer_t* sorder, bool is_root, int task_depth) {
    if (Comm().is_null()) return;
    if (dim_sep()) {
      auto g = A.extract_graph
        (opts.separator_ordering_level(), sep_begin_, sep_end_);
      auto sep_tree = g.recursive_bisection
        (opts.compression_leaf_size(), 0,
         sorder+sep_begin_, nullptr, 0, 0, dim_sep());
      std::vector<integer_t> siorder(dim_sep());
      for (integer_t i=sep_begin_; i<sep_end_; i++)
        siorder[sorder[i]] = i - sep_begin_;
      g.permute(sorder+sep_begin_, siorder.data());
      for (integer_t i=sep_begin_; i<sep_end_; i++)
        sorder[i] += sep_begin_;
      sep_tiles_ = sep_tree.template leaf_sizes<std::size_t>();
      adm_ = g.admissibility(sep_tiles_);
    }
    if (dim_upd()) {
      auto leaf = opts.BLR_options().leaf_size();
      auto nt = std::ceil(float(dim_upd()) / leaf);
      upd_tiles_.resize(nt, leaf);
      upd_tiles_.back() = dim_upd() - leaf*(nt-1);
    }
  }

} // end namespace strumpack

#endif //FRONTAL_MATRIX_DENSE_MPI_HPP
