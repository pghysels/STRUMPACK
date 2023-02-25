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
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <cmath>

#include "FrontalMatrix.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#include "FrontalMatrixMPI.hpp"
#include "BLR/BLRExtendAdd.hpp"
#include "FrontalMatrixBLRMPI.hpp"
#endif

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  FrontalMatrix<scalar_t,integer_t>::FrontalMatrix
  (F_t* lchild, F_t* rchild, integer_t sep, integer_t sep_begin,
   integer_t sep_end, std::vector<integer_t>& upd)
    : sep_(sep), sep_begin_(sep_begin), sep_end_(sep_end),
      upd_(std::move(upd)), lchild_(lchild), rchild_(rchild) {
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::draw
  (std::ostream& of, int etree_level) const {
    draw_node(of, etree_level == 0);
    for (auto u : upd_) {
      char prev = std::cout.fill('0');
      of << "set obj rect from "
         << sep_begin_ << ", " << u << " to "
         << sep_end_ << ", " << u+1
         << " fc rgb '#FF0000'" << std::endl
         << "set obj rect from "
         << u << ", " << sep_begin_ << " to "
         << u+1 << ", " << sep_end_
         << " fc rgb '#FF0000'" << std::endl;
      std::cout.fill(prev);
    }
    if (lchild_) lchild_->draw(of, etree_level+1);
    if (rchild_) rchild_->draw(of, etree_level+1);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::draw_node
  (std::ostream& of, bool is_root) const {
    char prev = std::cout.fill('0');
    of << "set obj rect from "
       << sep_begin_ << ", " << sep_begin_ << " to "
       << sep_end_ << ", " << sep_end_
       << " fc rgb '#FF0000'" << std::endl;
    std::cout.fill(prev);
  }

  /**
   * I[0:n-1] contains a list of unsorted global indices. This routine
   * finds the corresponding indices into the upd part of this front
   * and puts those indices in lI. Not all indices from I need to be
   * in lI, so oI stores for each element in lI, to which element in I
   * that element corresponds.
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::find_upd_indices
  (const std::vector<std::size_t>& I, std::vector<std::size_t>& lI,
   std::vector<std::size_t>& oI) const {
    auto n = I.size();
    lI.reserve(n);
    oI.reserve(n);
    for (std::size_t i=0; i<n; i++) {
      auto l = std::lower_bound(upd_.begin(), upd_.end(), I[i]);
      if (l != upd_.end() && *l == int(I[i])) {
        lI.push_back(std::distance(upd_.begin(), l));
        oI.push_back(i);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::upd_to_parent
  (const F_t* pa, std::size_t& upd2sep, std::size_t* I) const {
    integer_t r = 0, dupd = dim_upd(), pa_dsep = pa->dim_sep();
    for (; r<dupd; r++) {
      auto up = upd_[r];
      if (up >= pa->sep_end_) break;
      I[r] = up - pa->sep_begin_;
    }
    upd2sep = r;
    for (integer_t t=0; r<dupd; r++) {
      auto up = upd_[r];
      while (pa->upd_[t] < up) t++;
      I[r] = t + pa_dsep;
    }
  }
  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::upd_to_parent
  (const F_t* pa, std::size_t* I) const {
    integer_t r = 0, dupd = dim_upd(), pa_dsep = pa->dim_sep();
    for (; r<dupd; r++) {
      auto up = upd_[r];
      if (up >= pa->sep_end_) break;
      I[r] = up - pa->sep_begin_;
    }
    for (integer_t t=0; r<dupd; r++) {
      auto up = upd_[r];
      while (pa->upd_[t] < up) t++;
      I[r] = t + pa_dsep;
    }
  }

  template<typename scalar_t,typename integer_t> std::vector<std::size_t>
  FrontalMatrix<scalar_t,integer_t>::upd_to_parent
  (const F_t* pa) const {
    std::size_t upd2sep;
    return upd_to_parent(pa, upd2sep);
  }

  template<typename scalar_t,typename integer_t> std::vector<std::size_t>
  FrontalMatrix<scalar_t,integer_t>::upd_to_parent
  (const F_t* pa, std::size_t& upd2sep) const {
    std::vector<std::size_t> I(dim_upd());
    upd_to_parent(pa, upd2sep, I.data());
    return I;
  }

  template<typename scalar_t,typename integer_t> inline void
  FrontalMatrix<scalar_t,integer_t>::extend_add_b
  (DenseM_t& b, DenseM_t& bupd, const DenseM_t& CB, const F_t* pa) const {
    std::size_t upd2sep;
    auto I = upd_to_parent(pa, upd2sep);
    for (std::size_t c=0; c<b.cols(); c++) {
      for (std::size_t r=0; r<upd2sep; r++)
        b(I[r]+pa->sep_begin_, c) += CB(r, c);
      for (std::size_t r=upd2sep; r<CB.rows(); r++) {
        bupd(I[r]-pa->dim_sep(), c) += CB(r, c);
      }
    }
    STRUMPACK_FLOPS
      ((is_complex<scalar_t>()?2:1)*
       static_cast<long long int>(CB.rows())*b.cols());
    STRUMPACK_BYTES
      (b.cols()*(sizeof(scalar_t)*static_cast<long long int>(3*CB.rows())+
                 sizeof(integer_t)*(CB.rows()+bupd.rows())));
  }

  /**
   * Assemble CB=b(I^{upd}) from [b(I^{sep});b(I^{upd})] of the
   * parent.
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::extract_b
  (const DenseM_t& y, const DenseM_t& yupd, DenseM_t& CB, const F_t* pa) const {
    std::size_t upd2sep;
    auto I = upd_to_parent(pa, upd2sep);
    for (std::size_t c=0; c<y.cols(); c++) {
      for (std::size_t r=0; r<upd2sep; r++)
        CB(r,c) = y(I[r]+pa->sep_begin_, c);
      for (std::size_t r=upd2sep; r<CB.rows(); r++)
        CB(r,c) = yupd(I[r]-pa->dim_sep(), c);
    }
    // TODO adjust flops for multiple columns
    // STRUMPACK_FLOPS
    //   ((is_complex<scalar_t>()?2:1)*
    //    static_cast<long long int>(CB.rows()));
    STRUMPACK_BYTES
      (sizeof(scalar_t)*static_cast<long long int>
       (3*CB.rows())+sizeof(integer_t)*(CB.rows()+yupd.rows()));
  }

  template<typename scalar_t,typename integer_t> integer_t
  FrontalMatrix<scalar_t,integer_t>::maximum_rank(int task_depth) const {
    integer_t r = front_rank(), rl = 0, rr = 0;
    if (lchild_)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
      rl = lchild_->maximum_rank(task_depth+1);
    if (rchild_)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
      rr = rchild_->maximum_rank(task_depth+1);
#pragma omp taskwait
    return std::max(r, std::max(rl, rr));
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::multifrontal_solve(DenseM_t& b) const {
    auto max_dupd = max_dim_upd();
    auto lvls = levels();
    std::vector<DenseM_t> CB(lvls);
    // for (auto& cb : CB)
    //   cb = DenseM_t(max_dupd, b.cols());
    for (std::size_t i=0; i<CB.size(); i++)
      CB[i] = DenseM_t(max_dupd, b.cols());
    TIMER_TIME(TaskType::FORWARD_SOLVE, 0, t_fwd);
    forward_multifrontal_solve(b, CB.data());
    TIMER_STOP(t_fwd);
    TIMER_TIME(TaskType::BACKWARD_SOLVE, 0, t_bwd);
    backward_multifrontal_solve(b, CB.data());
    TIMER_STOP(t_bwd);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t bupd(dim_upd(), b.cols(), work[0], 0, 0);
    bupd.zero();
    if (task_depth == 0) {
      // tasking when calling the children
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      this->fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
      // no tasking for the root node computations, use system blas threading!
      fwd_solve_phase2(b, bupd, etree_level, params::task_recursion_cutoff_level);
    } else {
      this->fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
      fwd_solve_phase2(b, bupd, etree_level, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::fwd_solve_phase1
  (DenseM_t& b, DenseM_t& bupd, DenseM_t* work,
   int etree_level, int task_depth) const {
    if (task_depth < params::task_recursion_cutoff_level) {
      if (lchild_)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        lchild_->forward_multifrontal_solve
          (b, work+1, etree_level+1, task_depth+1);
      if (rchild_)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          std::vector<DenseM_t> work2(rchild_->levels());
          // for (auto& cb : work2)
          //   cb = DenseM_t(rchild_->max_dim_upd(), b.cols());
          for (std::size_t i=0; i<work2.size(); i++)
            work2[i] = DenseM_t(rchild_->max_dim_upd(), b.cols());
          rchild_->forward_multifrontal_solve
            (b, work2.data(), etree_level+1, task_depth+1);
          DenseMW_t CBch(rchild_->dim_upd(), b.cols(), work2[0], 0, 0);
          rchild_->extend_add_b(b, bupd, CBch, this);
        }
#pragma omp taskwait
      if (lchild_) {
        DenseMW_t CBch(lchild_->dim_upd(), b.cols(), work[1], 0, 0);
        lchild_->extend_add_b(b, bupd, CBch, this);
      }
    } else {
      if (lchild_) {
        lchild_->forward_multifrontal_solve
          (b, work+1, etree_level+1, task_depth);
        DenseMW_t CBch(lchild_->dim_upd(), b.cols(), work[1], 0, 0);
        lchild_->extend_add_b(b, bupd, CBch, this);
      }
      if (rchild_) {
        rchild_->forward_multifrontal_solve
          (b, work+1, etree_level+1, task_depth);
        DenseMW_t CBch(rchild_->dim_upd(), b.cols(), work[1], 0, 0);
        rchild_->extend_add_b(b, bupd, CBch, this);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t yupd(dim_upd(), y.cols(), work[0], 0, 0);
    if (task_depth == 0) {
      // no tasking in blas routines, use system threaded blas instead
      bwd_solve_phase1
        (y, yupd, etree_level, params::task_recursion_cutoff_level);
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      // tasking when calling children
      this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
    } else {
      bwd_solve_phase1(y, yupd, etree_level, task_depth);
      this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::bwd_solve_phase2
  (DenseM_t& y, DenseM_t& yupd, DenseM_t* work,
   int etree_level, int task_depth) const {
    if (task_depth < params::task_recursion_cutoff_level) {
      if (lchild_) {
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          DenseMW_t CB(lchild_->dim_upd(), y.cols(), work[1], 0, 0);
          lchild_->extract_b(y, yupd, CB, this);
          lchild_->backward_multifrontal_solve
            (y, work+1, etree_level+1, task_depth+1);
        }
      }
      if (rchild_) {
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          std::vector<DenseM_t> work2(rchild_->levels());
          // for (auto& cb : work2)
          //   cb = DenseM_t(rchild_->max_dim_upd(), y.cols());
          for (std::size_t i=0; i<work2.size(); i++)
            work2[i] = DenseM_t(rchild_->max_dim_upd(), y.cols());
          DenseMW_t CB(rchild_->dim_upd(), y.cols(), work2[0], 0, 0);
          rchild_->extract_b(y, yupd, CB, this);
          rchild_->backward_multifrontal_solve
            (y, work2.data(), etree_level+1, task_depth+1);
        }
      }
#pragma omp taskwait
    } else {
      if (lchild_) {
        DenseMW_t CB(lchild_->dim_upd(), y.cols(), work[1], 0, 0);
        lchild_->extract_b(y, yupd, CB, this);
        lchild_->backward_multifrontal_solve
          (y, work+1, etree_level+1, task_depth);
      }
      if (rchild_) {
        DenseMW_t CB(rchild_->dim_upd(), y.cols(), work[1], 0, 0);
        rchild_->extract_b(y, yupd, CB, this);
        rchild_->backward_multifrontal_solve
          (y, work+1, etree_level+1, task_depth);
      }
    }
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrix<scalar_t,integer_t>::factor_nonzeros(int task_depth) const {
    long long nnz = node_factor_nonzeros(), nnzl = 0, nnzr = 0;
    if (lchild_)
#pragma omp task default(shared)                        \
  if(task_depth < params::task_recursion_cutoff_level)
      nnzl = lchild_->factor_nonzeros(task_depth+1);
    if (rchild_)
#pragma omp task default(shared)                        \
  if(task_depth < params::task_recursion_cutoff_level)
      nnzr = rchild_->factor_nonzeros(task_depth+1);
#pragma omp taskwait
    return nnz + nnzl + nnzr;
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrix<scalar_t,integer_t>::dense_factor_nonzeros
  (int task_depth) const {
    long long nnz = dense_node_factor_nonzeros(), nnzl = 0, nnzr = 0;
    if (lchild_)
#pragma omp task default(shared)                        \
  if(task_depth < params::task_recursion_cutoff_level)
      nnzl = lchild_->dense_factor_nonzeros(task_depth+1);
    if (rchild_)
#pragma omp task default(shared)                        \
  if(task_depth < params::task_recursion_cutoff_level)
      nnzr = rchild_->dense_factor_nonzeros(task_depth+1);
#pragma omp taskwait
    return nnz + nnzl + nnzr;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontalMatrix<scalar_t,integer_t>::inertia
  (integer_t& neg, integer_t& zero, integer_t& pos) const {
    ReturnCode el = ReturnCode::SUCCESS, er = ReturnCode::SUCCESS;
    if (lchild_) el = lchild_->inertia(neg, zero, pos);
    if (rchild_) er = rchild_->inertia(neg, zero, pos);
    if (el != ReturnCode::SUCCESS) return el;
    if (er != ReturnCode::SUCCESS) return er;
    return node_inertia(neg, zero, pos);
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::multifrontal_solve
  (DenseM_t& bloc, DistM_t* bdist) const {
    DistM_t CB;
    DenseM_t seqCB;
    TIMER_TIME(TaskType::FORWARD_SOLVE, 0, t_fwd);
    forward_multifrontal_solve(bloc, bdist, CB, seqCB);
    TIMER_STOP(t_fwd);
    TIMER_TIME(TaskType::BACKWARD_SOLVE, 0, t_bwd);
    backward_multifrontal_solve(bloc, bdist, CB, seqCB);
    TIMER_STOP(t_bwd);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
   int etree_level) const {
    auto max_dupd = max_dim_upd();
    auto lvls = levels();
    std::vector<DenseM_t> CB(lvls);
    // for (auto& cb : CB)
    //   cb = DenseM_t(max_dupd, bloc.cols());
    for (std::size_t i=0; i<CB.size(); i++)
      CB[i] = DenseM_t(max_dupd, bloc.cols());
    forward_multifrontal_solve(bloc, CB.data(), etree_level, 0);
    seqbupd = CB[0];
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
   int etree_level) const {
    auto max_dupd = max_dim_upd();
    auto lvls = levels();
    std::vector<DenseM_t> CB(lvls);
    // for (auto& cb : CB)
    //   cb = DenseM_t(max_dupd, yloc.cols());
    for (std::size_t i=0; i<CB.size(); i++)
      CB[i] = DenseM_t(max_dupd, yloc.cols());
    CB[0] = seqyupd;
    backward_multifrontal_solve(yloc, CB.data(), etree_level, 0);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::extend_add_copy_from_buffers
  (DistM_t& F11, DistM_t& F12, DistM_t& F21, DistM_t& F22,
   scalar_t** pbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_seq_copy_from_buffers
      (F11, F12, F21, F22, *pbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::extadd_blr_copy_from_buffers
  (BLRMPI_t& F11, BLRMPI_t& F12, BLRMPI_t& F21, BLRMPI_t& F22,
   scalar_t** pbuf, const FBLRMPI_t* pa) const {
    BLR::BLRExtendAdd<scalar_t,integer_t>::seq_copy_from_buffers
      (F11, F12, F21, F22, *pbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::extadd_blr_copy_from_buffers_col
  (BLRMPI_t& F11, BLRMPI_t& F12, BLRMPI_t& F21, BLRMPI_t& F22,
   scalar_t** pbuf, const FBLRMPI_t* pa, integer_t begin_col, integer_t end_col) const {
    BLR::BLRExtendAdd<scalar_t,integer_t>::seq_copy_from_buffers_col
      (F11, F12, F21, F22, *pbuf, pa, this, begin_col, end_col);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::extend_add_column_copy_to_buffers
  (const DistM_t& CB, const DenseM_t& seqCB,
   std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_column_seq_copy_to_buffers
      (seqCB, sbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::extend_add_column_copy_from_buffers
  (DistM_t& B, DistM_t& Bupd, scalar_t** pbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_column_seq_copy_from_buffers
      (B, Bupd, *pbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::extract_column_copy_to_buffers
  (const DistM_t& b, const DistM_t& bupd, int ch_master,
   std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::extract_column_seq_copy_to_buffers
      (b, bupd, sbuf[ch_master], pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::extract_column_copy_from_buffers
  (const DistM_t& b, DistM_t& CB, DenseM_t& seqCB,
   std::vector<scalar_t*>& pbuf, const FMPI_t* pa) const {
    seqCB = DenseM_t(dim_upd(), b.cols());
    ExtendAdd<scalar_t,integer_t>::extract_column_seq_copy_from_buffers
      (seqCB, pbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::skinny_ea_to_buffers
  (const DistM_t& S, const DenseM_t& seqS,
   std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::skinny_extend_add_seq_copy_to_buffers
      (seqS, sbuf, pa);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::skinny_ea_from_buffers
  (DistM_t& S, scalar_t** pbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::skinny_extend_add_seq_copy_from_buffers
      (S, *pbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::extract_from_R2D
  (const DistM_t& R, DistM_t& cR, DenseM_t& seqcR,
   const FMPI_t* pa, bool visit) const {
    if (visit) seqcR = DenseM_t(R.rows(), R.cols());
    copy(R.rows(), R.cols(), R, 0, 0, seqcR,
         pa->master(this), pa->grid()->ctxt_all());
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::get_submatrix_2d
  (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
   DistM_t&, DenseM_t& Bseq) const {
    TIMER_TIME(TaskType::GET_SUBMATRIX, 2, t_getsub);
    Bseq = DenseM_t(I.size(), J.size());
    Bseq.zero();
    extract_CB_sub_matrix(I, J, Bseq, 0);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::get_submatrix_2d
  (const std::vector<std::vector<std::size_t>>& I,
   const std::vector<std::vector<std::size_t>>& J,
   std::vector<DistM_t>&, std::vector<DenseM_t>& Bseq) const {
    TIMER_TIME(TaskType::GET_SUBMATRIX, 2, t_getsub);
    extract_CB_sub_matrix_blocks(I, J, Bseq, 0);
  }
#endif

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::extract_CB_sub_matrix_blocks
  (const std::vector<std::vector<std::size_t>>& I,
   const std::vector<std::vector<std::size_t>>& J,
   std::vector<DenseM_t>& Bseq, int task_depth) const {
    for (std::size_t i=0; i<I.size(); i++) {
      Bseq.emplace_back(I[i].size(), J[i].size());
      Bseq[i].zero();
      extract_CB_sub_matrix(I[i], J[i], Bseq[i], 0);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::extract_CB_sub_matrix_blocks
  (const std::vector<std::vector<std::size_t>>& I,
   const std::vector<std::vector<std::size_t>>& J,
   std::vector<DenseMW_t>& Bseq, int task_depth) const {
    for (std::size_t i=0; i<I.size(); i++)
      extract_CB_sub_matrix(I[i], J[i], Bseq[i], 0);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::partition_fronts
  (const Opts_t& opts, const SpMat_t& A, integer_t* sorder,
   bool is_root, int task_depth) {
    auto lch = lchild_.get();
    auto rch = rchild_.get();
    if (lch)
#pragma omp task default(shared) firstprivate(sorder,task_depth,lch)    \
  if(task_depth < params::task_recursion_cutoff_level)
      lch->partition_fronts(opts, A, sorder, false, task_depth+1);
    if (rch)
#pragma omp task default(shared) firstprivate(sorder,task_depth,rch)    \
  if(task_depth < params::task_recursion_cutoff_level)
      rch->partition_fronts(opts, A, sorder, false, task_depth+1);
#pragma omp taskwait
    partition(opts, A, sorder, is_root, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::partition
  (const Opts_t& opts, const SpMat_t& A, integer_t* sorder,
   bool is_root, int task_depth) {
    // default is to do nothing, see FrontalMatrixHSS for an actual
    // implementation
    std::iota(sorder+sep_begin_, sorder+sep_end_, sep_begin_);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::permute_CB
  (const integer_t* perm, int task_depth) {
    auto lch = lchild_.get();
    auto rch = rchild_.get();
    if (lch)
#pragma omp task default(shared) firstprivate(perm,task_depth,lch)      \
  if(task_depth < params::task_recursion_cutoff_level)
      lch->permute_CB(perm, task_depth+1);
    if (rch)
#pragma omp task default(shared) firstprivate(perm,task_depth,rch)      \
  if(task_depth < params::task_recursion_cutoff_level)
      rch->permute_CB(perm, task_depth+1);
    for (integer_t i=0; i<dim_upd(); i++)
      upd_[i] = perm[upd_[i]];
    std::sort(upd_.begin(), upd_.end());
#pragma omp taskwait
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::get_level_fronts
  (std::vector<const F_t*>& ldata, int elvl, int l) const {
    if (l < elvl) {
      if (lchild_) lchild_->get_level_fronts(ldata, elvl, l+1);
      if (rchild_) rchild_->get_level_fronts(ldata, elvl, l+1);
    } else ldata.push_back(this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::get_level_fronts
  (std::vector<F_t*>& ldata, int elvl, int l) {
    if (l < elvl) {
      if (lchild_) lchild_->get_level_fronts(ldata, elvl, l+1);
      if (rchild_) rchild_->get_level_fronts(ldata, elvl, l+1);
    } else ldata.push_back(this);
  }


  // explicit template instantiations
  template class FrontalMatrix<float,int>;
  template class FrontalMatrix<double,int>;
  template class FrontalMatrix<std::complex<float>,int>;
  template class FrontalMatrix<std::complex<double>,int>;

  template class FrontalMatrix<float,long int>;
  template class FrontalMatrix<double,long int>;
  template class FrontalMatrix<std::complex<float>,long int>;
  template class FrontalMatrix<std::complex<double>,long int>;

  template class FrontalMatrix<float,long long int>;
  template class FrontalMatrix<double,long long int>;
  template class FrontalMatrix<std::complex<float>,long long int>;
  template class FrontalMatrix<std::complex<double>,long long int>;

} // end namespace strumpack
