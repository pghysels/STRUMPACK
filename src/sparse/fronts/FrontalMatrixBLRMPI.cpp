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
    F22blr_ = BLRMPI_t(); // remove the update block
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
      F11blr_ = BLRMPI_t(pgrid_, sep_tiles_, sep_tiles_);
      F11blr_.fill(0.);
      if (dim_upd()) {
        F12blr_ = BLRMPI_t(pgrid_, sep_tiles_, upd_tiles_);
        F21blr_ = BLRMPI_t(pgrid_, upd_tiles_, sep_tiles_);
        F12blr_.fill(0.);
        F21blr_.fill(0.);
      }
    }
    if (dupd) {
      F22blr_ = BLRMPI_t(pgrid_, upd_tiles_, upd_tiles_);
      F22blr_.fill(0.);
    }
    using Trip_t = Triplet<scalar_t>;
    std::vector<Trip_t> e11, e12, e21;
    A.push_front_elements(sep_begin_, sep_end_, this->upd(), e11, e12, e21);
    int npr = grid2d().nprows();
    // TODO combine these 3 all_to_all calls?
    {
      std::vector<std::vector<Trip_t>> sbuf(this->P());
      for (auto& e : e11) sbuf[sep_rg2p(e.r)+sep_cg2p(e.c)*npr].push_back(e);
      auto rbuf = Comm().all_to_all_v(sbuf);
      for (auto& e : rbuf) F11blr_.global(e.r, e.c) = e.v;
    } {
      std::vector<std::vector<Trip_t>> sbuf(this->P());
      for (auto& e : e12) sbuf[sep_rg2p(e.r)+upd_cg2p(e.c)*npr].push_back(e);
      auto rbuf = Comm().all_to_all_v(sbuf);
      for (auto& e : rbuf) F12blr_.global(e.r, e.c) = e.v;
    } {
      std::vector<std::vector<Trip_t>> sbuf(this->P());
      for (auto& e : e21) sbuf[upd_rg2p(e.r)+sep_cg2p(e.c)*npr].push_back(e);
      auto rbuf = Comm().all_to_all_v(sbuf);
      for (auto& e : rbuf) F21blr_.global(e.r, e.c) = e.v;
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
    TaskTimer t("FrontalMatrixBLRMPI_factor");
    if (/*etree_level == 0 && */opts.print_root_front_stats()) t.start();
    build_front(A);
    extend_add();
    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
    if (dim_sep() && grid2d().active()) {
      if (dim_upd())
        piv_ = BLRMPI_t::partial_factor
          (F11blr_, F12blr_, F21blr_, F22blr_, adm_, opts.BLR_options());
      else piv_ = F11blr_.factor(adm_, opts.BLR_options());
      // TODO flops?
    }
    if (/*etree_level == 0 && */opts.print_root_front_stats()) {
      auto time = t.elapsed();
      if (Comm().is_root())
        std::cout << "#   - BLRMPI front: Nsep= " << dim_sep()
                  << ", Nupd= " << dim_upd()
                  << " , time= " << time << " sec" << std::endl;
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
        (b, pgrid_, sep_tiles_, col_tiles);
      auto bupd_blr = BLRMPI_t::from_ScaLAPACK
        (bupd, pgrid_, upd_tiles_, col_tiles);
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
        (y, pgrid_, sep_tiles_, col_tiles);
      auto yupd_blr = BLRMPI_t::from_ScaLAPACK
        (yupd, pgrid_, upd_tiles_, col_tiles);
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

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLRMPI<scalar_t,integer_t>::extract_CB_sub_matrix_2d
  (const VecVec_t& I, const VecVec_t& J, std::vector<DistM_t>& B) const {
    auto nB = I.size();
    std::vector<std::vector<std::size_t>> lI(nB), lJ(nB), oI(nB), oJ(nB);
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    using ExtAdd = ExtendAdd<scalar_t,integer_t>;
    for (std::size_t i=0; i<nB; i++) {
      this->find_upd_indices(I[i], lI[i], oI[i]);
      this->find_upd_indices(J[i], lJ[i], oJ[i]);
      ExtAdd::extract_copy_to_buffers
        (F22blr_, lI[i], lJ[i], oI[i], oJ[i], B[i], sbuf);
    }
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    {
      TIMER_TIME(TaskType::GET_SUBMATRIX_2D_A2A, 2, t_a2a);
      Comm().all_to_all_v(sbuf, rbuf, pbuf);
    }
    for (std::size_t i=0; i<nB; i++)
      ExtAdd::extract_copy_from_buffers
        (B[i], lI[i], lJ[i], oI[i], oJ[i], F22blr_, pbuf);
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
