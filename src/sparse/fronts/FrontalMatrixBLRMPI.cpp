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
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

#include "FrontalMatrixBLRMPI.hpp"
#include "sparse/CSRGraph.hpp"
#include "ExtendAdd.hpp"
#include "BLR/BLRExtendAdd.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  FrontalMatrixBLRMPI<scalar_t,integer_t>::FrontalMatrixBLRMPI
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd, const MPIComm& comm, int P, int leaf)
    : FrontalMatrixMPI<scalar_t,integer_t>
    (sep, sep_begin, sep_end, upd, comm, P),
      pgrid_(Comm(), P), leaf_(leaf) {}

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::release_work_memory() {
    // F22_.clear(); // remove the update block
    F22blr_ = BLRMPI_t(); //.clear(); // remove the update block
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
      if (!visit(ch)) continue;
      ch->extadd_blr_copy_to_buffers(sbuf, this);
    }
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    for (auto& ch : {lchild_.get(), rchild_.get()}) {
      if (!ch) continue;
      ch->extadd_blr_copy_from_buffers
        (F11blr_, F12blr_, F21blr_, F22blr_,
         pbuf.data()+this->master(ch), this);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    DistM_t F22(grid(), dim_upd(), dim_upd());
    F22blr_.to_ScaLAPACK(F22);
    ExtendAdd<scalar_t,integer_t>::extend_add_copy_to_buffers
      (F22, sbuf, pa, this->upd_to_parent(pa));
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::extadd_blr_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf, const FBLRMPI_t* pa) const {
    BLR::BLRExtendAdd<scalar_t,integer_t>::copy_to_buffers
      (F22blr_, sbuf, pa, this->upd_to_parent(pa));
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::extadd_blr_copy_from_buffers
  (BLRMPI_t& F11, BLRMPI_t& F12, BLRMPI_t& F21, BLRMPI_t& F22,
   scalar_t** pbuf, const FBLRMPI_t* pa) const {
    BLR::BLRExtendAdd<scalar_t,integer_t>::copy_from_buffers
      (F11, F12, F21, F22, pbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::build_front
  (const SpMat_t& A) {
    const auto dupd = dim_upd();
    const auto dsep = dim_sep();
    if (dsep) {
      // TODO directly to BLR matrices, as in the GPU code???
      using ExFront = ExtractFront<scalar_t,integer_t>;
      DistM_t F11(grid(), dsep, dsep);
      ExFront::extract_F11(F11, A, sep_begin_, dsep);
      F11blr_ = BLRMPI_t::from_ScaLAPACK(F11, sep_tiles_, sep_tiles_, pgrid_);
      if (dim_upd()) {
        DistM_t F12(grid(), dsep, dupd), F21(grid(), dupd, dsep);
        ExFront::extract_F12(F12, A, sep_begin_, sep_end_, this->upd_);
        ExFront::extract_F21(F21, A, sep_end_, sep_begin_, this->upd_);
        F12blr_ = BLRMPI_t::from_ScaLAPACK(F12, sep_tiles_, upd_tiles_, pgrid_);
        F21blr_ = BLRMPI_t::from_ScaLAPACK(F21, upd_tiles_, sep_tiles_, pgrid_);
      }
    }
    if (dupd) {
      DistM_t F22(grid(), dupd, dupd);
      F22.zero();
      F22blr_ = BLRMPI_t::from_ScaLAPACK(F22, upd_tiles_, upd_tiles_, pgrid_);
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
    extend_add();

    // if (etree_level == 0) {
    //   auto Fd = F11blr_.to_ScaLAPACK(this->grid());
    //   Fd.print("Froot");
    // }

    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
    if (dim_sep() && grid2d().active()) {
      if (dim_upd())
        piv_ = BLRMPI_t::partial_factor
          (F11blr_, F12blr_, F21blr_, F22blr_, adm_, opts.BLR_options());
      else piv_ = F11blr_.factor(adm_, opts.BLR_options());
      // TODO flops?
    }
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
      std::vector<std::size_t> col_tiles(1, b.cols());
      auto b_blr = BLRMPI_t::from_ScaLAPACK
        (b, sep_tiles_, col_tiles, pgrid_);
      auto bupd_blr = BLRMPI_t::from_ScaLAPACK
        (bupd, upd_tiles_, col_tiles, pgrid_);
      b_blr.laswp(piv_, true);
      if (b.cols() == 1) {
        trsv(UpLo::L, Trans::N, Diag::U, F11blr_, b_blr);
        if (dim_upd())
          gemv(Trans::N, scalar_t(-1.), F21blr_, b_blr, scalar_t(1.), bupd_blr);
      } else {
        trsm(Side::L, UpLo::L, Trans::N, Diag::U, scalar_t(1.), F11blr_, b_blr);
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F21blr_, b_blr, scalar_t(1.), bupd_blr);
      }
      b_blr.to_ScaLAPACK(b);
      bupd_blr.to_ScaLAPACK(bupd);
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
      std::vector<std::size_t> col_tiles(1, y.cols());
      auto y_blr = BLRMPI_t::from_ScaLAPACK
        (y, sep_tiles_, col_tiles, pgrid_);
      auto yupd_blr = BLRMPI_t::from_ScaLAPACK
        (yupd, upd_tiles_, col_tiles, pgrid_);
      if (y.cols() == 1) {
        if (dim_upd())
          gemv(Trans::N, scalar_t(-1.), F12blr_, yupd_blr, scalar_t(1.), y_blr);
        trsv(UpLo::U, Trans::N, Diag::N, F11blr_, y_blr);
      } else {
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F12blr_, yupd_blr, scalar_t(1.), y_blr);
        trsm(Side::L, UpLo::U, Trans::N, Diag::N, scalar_t(1.), F11blr_, y_blr);
      }
      y_blr.to_ScaLAPACK(y);
      yupd_blr.to_ScaLAPACK(yupd);
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

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixBLRMPI<scalar_t,integer_t>::node_factor_nonzeros() const {
    return F11blr_.nonzeros() + F12blr_.nonzeros() + F21blr_.nonzeros();
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
        (opts.BLR_options().leaf_size(), 0,
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

  // explicit template instantiations
  template class FrontalMatrixBLRMPI<float,int>;
  template class FrontalMatrixBLRMPI<double,int>;
  template class FrontalMatrixBLRMPI<std::complex<float>,int>;
  template class FrontalMatrixBLRMPI<std::complex<double>,int>;

  template class FrontalMatrixBLRMPI<float,long int>;
  template class FrontalMatrixBLRMPI<double,long int>;
  template class FrontalMatrixBLRMPI<std::complex<float>,long int>;
  template class FrontalMatrixBLRMPI<std::complex<double>,long int>;

  template class FrontalMatrixBLRMPI<float,long long int>;
  template class FrontalMatrixBLRMPI<double,long long int>;
  template class FrontalMatrixBLRMPI<std::complex<float>,long long int>;
  template class FrontalMatrixBLRMPI<std::complex<double>,long long int>;

} // end namespace strumpack
