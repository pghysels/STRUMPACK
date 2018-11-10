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
#ifndef FRONTAL_MATRIX_HPP
#define FRONTAL_MATRIX_HPP

#include <iostream>
#include <algorithm>
#include <random>
#include <vector>

#include "misc/TaskTimer.hpp"
#include "StrumpackParameters.hpp"
#include "CompressedSparseMatrix.hpp"
#include "MatrixReordering.hpp"
#include "HSS/HSSMatrix.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#endif
#if defined(_OPENMP)
#include "omp.h"
#endif

namespace strumpack {

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> class ExtendAdd;
#endif

  template<typename scalar_t,typename integer_t> class FrontalMatrix {
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
#if defined(STRUMPACK_USE_MPI)
    using DistM_t = DistributedMatrix<scalar_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using ExtAdd = ExtendAdd<scalar_t,integer_t>;
#endif


  public:
    FrontalMatrix
    (F_t* lchild, F_t* rchild, integer_t sep, integer_t sep_begin,
     integer_t sep_end, std::vector<integer_t>& upd);
    virtual ~FrontalMatrix() = default;

    integer_t sep_begin() const { return sep_begin_; }
    integer_t sep_end() const { return sep_end_; }
    integer_t dim_sep() const { return sep_end_ - sep_begin_; }
    integer_t dim_upd() const { return upd_.size(); }
    integer_t dim_blk() const { return dim_sep() + dim_upd(); }
    const std::vector<integer_t>& upd() const { return upd_; }

    void draw(std::ostream& of, int etree_level=0) const;

    void find_upd_indices
    (const std::vector<std::size_t>& I, std::vector<std::size_t>& lI,
     std::vector<std::size_t>& oI) const;
    std::vector<std::size_t> upd_to_parent
    (const F_t* pa, std::size_t& upd2sep) const;
    std::vector<std::size_t> upd_to_parent(const F_t* pa) const;

    virtual void release_work_memory() = 0;

    virtual void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level=0, int task_depth=0) = 0;

    void multifrontal_solve(DenseM_t& b) const;
    virtual void forward_multifrontal_solve
    (DenseM_t& b, DenseM_t* work, int etree_level=0,
     int task_depth=0) const {};
    virtual void backward_multifrontal_solve
    (DenseM_t& y, DenseM_t* work, int etree_level=0,
     int task_depth=0) const {};

    void fwd_solve_phase1
    (DenseM_t& b, DenseM_t& bupd, DenseM_t* work,
     int etree_level, int task_depth) const;
    void bwd_solve_phase2
    (DenseM_t& y, DenseM_t& yupd, DenseM_t* work,
     int etree_level, int task_depth) const;

    virtual void extend_add_to_dense
    (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
     const FrontalMatrix<scalar_t,integer_t>* p, int task_depth) {}

    virtual int random_samples() const { return 0; }

    virtual void sample_CB
    (const SPOptions<scalar_t>& opts, const DenseM_t& R,
     DenseM_t& Sr, DenseM_t& Sc, F_t* parent, int task_depth=0) {};
    virtual void extract_CB_sub_matrix
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DenseM_t& B, int task_depth) const {};

    void extend_add_b
    (DenseM_t& b, DenseM_t& bupd, const DenseM_t& CB, const F_t* pa) const;
    void extract_b
    (const DenseM_t& y, const DenseM_t& yupd, DenseM_t& CB, const F_t* pa) const;

    virtual integer_t maximum_rank(int task_depth=0) const { return 0; }
    virtual long long factor_nonzeros(int task_depth=0) const;
    virtual long long dense_factor_nonzeros(int task_depth=0) const;
    virtual bool isHSS() const { return false; }
    virtual bool isMPI() const { return false; }
    virtual void print_rank_statistics(std::ostream &out) const {}
    virtual std::string type() const { return "FrontalMatrix"; }
    virtual void bisection_partitioning
    (const SPOptions<scalar_t>& opts, integer_t* sorder,
     bool isroot=true, int task_depth=0);
    void permute_upd_indices(const integer_t* perm, int task_depth=0);

    virtual void set_BLR_partitioning
    (const SPOptions<scalar_t>& opts, const HSS::HSSPartitionTree& sep_tree,
     const std::vector<bool>& adm, bool is_root) {}
    virtual void set_HSS_partitioning
    (const SPOptions<scalar_t>& opts, const HSS::HSSPartitionTree& sep_tree,
     bool is_root) {}
    virtual void set_HODLR_partitioning
    (const SPOptions<scalar_t>& opts, const HSS::HSSPartitionTree& sep_tree,
     bool is_root) {}

    int levels() const {
      int ll = 0, lr = 0;
      if (lchild_) ll = lchild_->levels();
      if (rchild_) lr = rchild_->levels();
      return std::max(ll, lr) + 1;
    }

    void set_lchild(std::unique_ptr<F_t> ch) { lchild_ = std::move(ch); }
    void set_rchild(std::unique_ptr<F_t> ch) { rchild_ = std::move(ch); }

    // TODO compute this (and levels) once, store it
    // maybe compute it when setting pointers to the children
    // create setters/getters for the children
    integer_t max_dim_upd() const {
      integer_t max_dupd = dim_upd();
      if (lchild_) max_dupd = std::max(max_dupd, lchild_->max_dim_upd());
      if (rchild_) max_dupd = std::max(max_dupd, rchild_->max_dim_upd());
      return max_dupd;
    }

    virtual int P() const { return 1; }


#if defined(STRUMPACK_USE_MPI)
    void multifrontal_solve(DenseM_t& bloc, DistM_t* bdist) const;
    virtual void forward_multifrontal_solve
    (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
     int etree_level=0) const;
    virtual void backward_multifrontal_solve
    (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
     int etree_level=0) const;

    virtual void extend_add_copy_to_buffers
    (std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
      assert(false); // TODO static assert?
    };
    virtual void extend_add_copy_from_buffers
    (DistM_t& F11, DistM_t& F12, DistM_t& F21, DistM_t& F22,
     scalar_t** pbuf, const FMPI_t* pa) const {
      ExtAdd::extend_add_seq_copy_from_buffers
        (F11, F12, F21, F22, *pbuf, pa, this);
    }
    virtual void extend_add_column_copy_to_buffers
    (const DistM_t& CB, const DenseM_t& seqCB,
     std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
      ExtAdd::extend_add_column_seq_copy_to_buffers(seqCB, sbuf, pa, this);
    }
    virtual void extend_add_column_copy_from_buffers
    (DistM_t& B, DistM_t& Bupd, scalar_t** pbuf, const FMPI_t* pa) const {
      ExtAdd::extend_add_column_seq_copy_from_buffers
        (B, Bupd, *pbuf, pa, this);
    }
    virtual void extract_column_copy_to_buffers
    (const DistM_t& b, const DistM_t& bupd, int ch_master,
     std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
      ExtAdd::extract_column_seq_copy_to_buffers
        (b, bupd, sbuf[ch_master], pa, this);
    }
    virtual void extract_column_copy_from_buffers
    (const DistM_t& b, DistM_t& CB, DenseM_t& seqCB,
     std::vector<scalar_t*>& pbuf, const FMPI_t* pa) const {
      seqCB = DenseM_t(dim_upd(), b.cols());
      ExtAdd::extract_column_seq_copy_from_buffers(seqCB, pbuf, pa, this);
    }

    virtual void get_submatrix_2d
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DistM_t& Bdist, DenseM_t& Bseq) const;
    virtual void get_submatrix_2d
    (const std::vector<std::vector<std::size_t>>& I,
     const std::vector<std::vector<std::size_t>>& J,
     std::vector<DistM_t>& Bdist, std::vector<DenseM_t>& Bseq) const;
#endif

  protected:
    integer_t sep_;
    integer_t sep_begin_;
    integer_t sep_end_;
    std::vector<integer_t> upd_;

    // TODO use a vector?
    //std::vector<std::unique_ptr<F_t>> ch_;
    std::unique_ptr<F_t> lchild_;
    std::unique_ptr<F_t> rchild_;

  private:
    FrontalMatrix(const FrontalMatrix&) = delete;
    FrontalMatrix& operator=(FrontalMatrix const&) = delete;

    virtual void draw_node(std::ostream& of, bool is_root) const;

    virtual long long node_factor_nonzeros() const {
      return dense_node_factor_nonzeros();
    }
    virtual long long dense_node_factor_nonzeros() const {
      long long dsep = dim_sep();
      long long dupd = dim_upd();
      return dsep * (dsep + 2 * dupd);
    }
  };

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

  template<typename scalar_t,typename integer_t> std::vector<std::size_t>
  FrontalMatrix<scalar_t,integer_t>::upd_to_parent
  (const F_t* pa) const {
    std::size_t upd2sep;
    return upd_to_parent(pa, upd2sep);
  }

  template<typename scalar_t,typename integer_t> std::vector<std::size_t>
  FrontalMatrix<scalar_t,integer_t>::upd_to_parent
  (const F_t* pa, std::size_t& upd2sep) const {
    integer_t r = 0;
    integer_t dupd = dim_upd();
    integer_t pa_dsep = pa->dim_sep();
    std::vector<std::size_t> I(dupd);
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
    // TODO adjust flops for multiple columns
    STRUMPACK_FLOPS
      ((is_complex<scalar_t>()?2:1)*
       static_cast<long long int>(CB.rows()));
    STRUMPACK_BYTES
      (sizeof(scalar_t)*static_cast<long long int>
       (3*CB.rows())+sizeof(integer_t)*(CB.rows()+bupd.rows()));
  }

  /**
   * Assemble CB=b(I^{upd}) from [b(I^{sep});b(I^{upd})] of the
   * parent.
   */
  template<typename scalar_t,typename integer_t> inline void
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
    STRUMPACK_FLOPS
      ((is_complex<scalar_t>()?2:1)*
       static_cast<long long int>(CB.rows()));
    STRUMPACK_BYTES
      (sizeof(scalar_t)*static_cast<long long int>
       (3*CB.rows())+sizeof(integer_t)*(CB.rows()+yupd.rows()));
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::multifrontal_solve(DenseM_t& b) const {
    auto max_dupd = max_dim_upd();
    auto lvls = levels();
    std::vector<DenseM_t> CB(lvls);
    for (auto& cb : CB)
      cb = DenseM_t(max_dupd, b.cols());
    TIMER_TIME(TaskType::FORWARD_SOLVE, 0, t_fwd);
    forward_multifrontal_solve(b, CB.data());
    TIMER_STOP(t_fwd);
    TIMER_TIME(TaskType::BACKWARD_SOLVE, 0, t_bwd);
    backward_multifrontal_solve(b, CB.data());
    TIMER_STOP(t_bwd);
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
          for (auto& cb : work2)
            cb = DenseM_t(rchild_->max_dim_upd(), b.cols());
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
      if (rchild_)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          std::vector<DenseM_t> work2(rchild_->levels());
          for (auto& cb : work2)
            cb = DenseM_t(rchild_->max_dim_upd(), y.cols());
          DenseMW_t CB(rchild_->dim_upd(), y.cols(), work2[0], 0, 0);
          rchild_->extract_b(y, yupd, CB, this);
          rchild_->backward_multifrontal_solve
            (y, work2.data(), etree_level+1, task_depth+1);
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

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::bisection_partitioning
  (const SPOptions<scalar_t>& opts, integer_t* sorder,
   bool isroot, int task_depth) {
    auto lch = lchild_.get();
    auto rch = rchild_.get();
    if (lch)
#pragma omp task default(shared) firstprivate(sorder,task_depth,lch)    \
  if(task_depth < params::task_recursion_cutoff_level)
      lch->bisection_partitioning(opts, sorder, false, task_depth+1);
    if (rch)
#pragma omp task default(shared) firstprivate(sorder,task_depth,rch)    \
  if(task_depth < params::task_recursion_cutoff_level)
      rch->bisection_partitioning(opts, sorder, false, task_depth+1);
#pragma omp taskwait

    // default is to do nothing, see FrontalMatrixHSS for an actual
    // implementation
    for (integer_t i=sep_begin_; i<sep_end_; i++)
      sorder[i] = -i;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::permute_upd_indices
  (const integer_t* perm, int task_depth) {
    auto lch = lchild_.get();
    auto rch = rchild_.get();
    if (lch)
#pragma omp task default(shared) firstprivate(perm,task_depth,lch)      \
  if(task_depth < params::task_recursion_cutoff_level)
      lch->permute_upd_indices(perm, task_depth+1);
    if (rch)
#pragma omp task default(shared) firstprivate(perm,task_depth,rch)      \
  if(task_depth < params::task_recursion_cutoff_level)
      rch->permute_upd_indices(perm, task_depth+1);

    for (integer_t i=0; i<dim_upd(); i++)
      upd_[i] = perm[upd_[i]];
    std::sort(upd_.begin(), upd_.end());
#pragma omp taskwait
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
    for (auto& cb : CB)
      cb = DenseM_t(max_dupd, bloc.cols());
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
    for (auto& cb : CB)
      cb = DenseM_t(max_dupd, yloc.cols());
    CB[0] = seqyupd;
    backward_multifrontal_solve(yloc, CB.data(), etree_level, 0);
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
    for (std::size_t i=0; i<I.size(); i++) {
      Bseq.emplace_back(I[i].size(), J[i].size());
      Bseq[i].zero();
      // for the threaded code, just extract block per block
      extract_CB_sub_matrix(I[i], J[i], Bseq[i], 0);
    }
  }
#endif

} // end namespace strumpack

#endif //FRONTAL_MATRIX_HPP
