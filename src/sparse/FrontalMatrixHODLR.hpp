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
#ifndef FRONTAL_MATRIX_HODLR_HPP
#define FRONTAL_MATRIX_HODLR_HPP

#include <iostream>
#include <algorithm>
#include <memory>

#include "misc/TaskTimer.hpp"
#include "FrontalMatrix.hpp"
#include "HODLR/HODLRMatrix.hpp"
#include "HODLR/LRBFMatrix.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixHODLR
    : public FrontalMatrix<scalar_t,integer_t> {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;

  public:
    FrontalMatrixHODLR
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd);
    FrontalMatrixHODLR(const FrontalMatrixHODLR&) = delete;
    FrontalMatrixHODLR& operator=(FrontalMatrixHODLR const&) = delete;

    void extend_add_to_dense
    (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
     const FrontalMatrix<scalar_t,integer_t>* p, int task_depth) override;

    void sample_CB
    (const SPOptions<scalar_t>& opts, const DenseM_t& R, DenseM_t& Sr,
     DenseM_t& Sc, F_t* pa, int task_depth) override;

    void release_work_memory() override;
    void random_sampling
    (const SpMat_t& A, const SPOptions<scalar_t>& opts, DenseM_t& Rr,
     DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc, int etree_level,
     int task_depth); // TODO const?

    void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level=0, int task_depth=0) override;

    void forward_multifrontal_solve
    (DenseM_t& b, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;
    void backward_multifrontal_solve
    (DenseM_t& y, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;

    integer_t maximum_rank(int task_depth=0) const override;
    void print_rank_statistics(std::ostream &out) const override;
    std::string type() const override { return "FrontalMatrixHODLR"; }

    void set_HODLR_partitioning
    (const SPOptions<scalar_t>& opts,
     const HSS::HSSPartitionTree& sep_tree, bool is_root) override;

  private:
    HODLR::HODLRMatrix<scalar_t> F11_;
    HODLR::LRBFMatrix<scalar_t> F12_, F21_;
    std::unique_ptr<HODLR::HODLRMatrix<scalar_t>> F22_;
    MPIComm commself_;

    void draw_node(std::ostream& of, bool is_root) const override;

    void multifrontal_factorization_node
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level, int task_depth);

    void fwd_solve_node
    (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const;
    void bwd_solve_node
    (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const;

    long long node_factor_nonzeros() const override;

    using FrontalMatrix<scalar_t,integer_t>::lchild_;
    using FrontalMatrix<scalar_t,integer_t>::rchild_;
    using FrontalMatrix<scalar_t,integer_t>::dim_sep;
    using FrontalMatrix<scalar_t,integer_t>::dim_upd;
    using FrontalMatrix<scalar_t,integer_t>::sep_begin_;
    using FrontalMatrix<scalar_t,integer_t>::sep_end_;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixHODLR<scalar_t,integer_t>::FrontalMatrixHODLR
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : FrontalMatrix<scalar_t,integer_t>
    (nullptr, nullptr, sep, sep_begin, sep_end, upd),
    commself_(MPI_COMM_SELF) { }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::release_work_memory() {
    F22_.reset(nullptr);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const FrontalMatrix<scalar_t,integer_t>* p, int task_depth) {
    const std::size_t pdsep = paF11.rows();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
    DenseM_t CB(dupd, dupd);
    {
      DenseM_t id(dupd, dupd);
      id.eye();
      F22_->mult('N', id, CB);
    }

#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared) grainsize(64)      \
  if(task_depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t c=0; c<dupd; c++) {
      auto pc = I[c];
      if (pc < pdsep) {
        for (std::size_t r=0; r<upd2sep; r++)
          paF11(I[r],pc) += CB(r,c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF21(I[r]-pdsep,pc) += CB(r,c);
      } else {
        for (std::size_t r=0; r<upd2sep; r++)
          paF12(I[r],pc-pdsep) += CB(r, c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF22(I[r]-pdsep,pc-pdsep) += CB(r,c);
      }
    }
    release_work_memory();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::sample_CB
  (const SPOptions<scalar_t>& opts, const DenseM_t& R,
   DenseM_t& Sr, DenseM_t& Sc, F_t* pa, int task_depth) {
    if (!dim_upd()) return;
    auto I = this->upd_to_parent(pa);
    auto cR = R.extract_rows(I);
    DenseM_t cSr(cR.rows(), cR.cols());
    DenseM_t cSc(cR.rows(), cR.cols());

    std::cout << "TODO sample with HODLR for CB" << std::endl;
    // _H.Schur_product_direct
    //   (_ULV, _Theta, _DUB01, _Phi,
    //    _ThetaVhatC_or_VhatCPhiC, cR, cSr, cSc);

    Sr.scatter_rows_add(I, cSr, task_depth);
    Sc.scatter_rows_add(I, cSc, task_depth);
    STRUMPACK_CB_SAMPLE_FLOPS(cSr.rows()*cSr.cols()*2);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (task_depth == 0)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      multifrontal_factorization_node(A, opts, etree_level, task_depth);
    else multifrontal_factorization_node(A, opts, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::random_sampling
  (const SpMat_t& A, const SPOptions<scalar_t>& opts, DenseM_t& Rr,
   DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc, int etree_level,
   int task_depth) {
    Sr.zero();
    Sc.zero();
    A.front_multiply
      (sep_begin_, sep_end_, this->upd_, Rr, Sr, Sc, task_depth);
    if (lchild_)
      lchild_->sample_CB(opts, Rr, Sr, Sc, this, task_depth);
    if (rchild_)
      rchild_->sample_CB(opts, Rr, Sr, Sc, this, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::multifrontal_factorization_node
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    bool tasked = task_depth < params::task_recursion_cutoff_level;
    if (tasked) {
      if (lchild_)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        lchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth+1);
      if (rchild_)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        rchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth+1);
#pragma omp taskwait
    } else {
      if (lchild_)
        lchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
      if (rchild_)
        rchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
    }

    const auto dsep = dim_sep();
    const auto dupd = dim_upd();

    //// ------ temporary test code ------------------------------
    DenseM_t dF11(dsep, dsep); dF11.zero();
    DenseM_t dF12(dsep, dupd); dF12.zero();
    DenseM_t dF21(dupd, dsep); dF21.zero();
    DenseM_t dF22(dupd, dupd); dF22.zero();
    A.extract_front
      (dF11, dF12, dF21, this->sep_begin_, this->sep_end_,
       this->upd_, task_depth);
    if (lchild_)
      lchild_->extend_add_to_dense
        (dF11, dF12, dF21, dF22, this, task_depth);
    if (rchild_)
      rchild_->extend_add_to_dense
        (dF11, dF12, dF21, dF22, this, task_depth);
    //// ---------------------------------------------------------

    auto sample_F11 = [&](char op, const DenseM_t& R, DenseM_t& S) {
      TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
      gemm(c2T(op), Trans::N, scalar_t(1.), dF11, R, scalar_t(0.), S, task_depth);
      TIMER_STOP(t_sampling);
    };
    auto sample_F12 = [&]
      (char op, scalar_t a, const DenseM_t& R, scalar_t b, DenseM_t& S) {
      TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
      gemm(c2T(op), Trans::N, a, dF12, R, b, S, task_depth);
      TIMER_STOP(t_sampling);
    };
    auto sample_F21 = [&]
      (char op, scalar_t a, const DenseM_t& R, scalar_t b, DenseM_t& S) {
      TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
      gemm(c2T(op), Trans::N, a, dF21, R, b, S, task_depth);
      TIMER_STOP(t_sampling);
    };

    F11_.compress(sample_F11);
    if (dupd) {
      F12_ = HODLR::LRBFMatrix<scalar_t>(F11_, *F22_, sample_F12);
      F21_ = HODLR::LRBFMatrix<scalar_t>(*F22_, F11_, sample_F21);
    }
    TIMER_TIME(TaskType::HSS_FACTOR, 0, t_fact);
    F11_.factor();
    TIMER_STOP(t_fact);

    //// ------ temporary test code -----------------------------
    auto piv = dF11.LU(task_depth);
    if (dupd) {
      dF12.laswp(piv, true);
      trsm(Side::L, UpLo::L, Trans::N, Diag::U,
           scalar_t(1.), dF11, dF12, task_depth);
      trsm(Side::R, UpLo::U, Trans::N, Diag::N,
           scalar_t(1.), dF11, dF21, task_depth);
      gemm(Trans::N, Trans::N, scalar_t(-1.), dF21, dF12,
           scalar_t(1.), dF22, task_depth);
      //// ---------------------------------------------------------

      auto sample_CB = [&](char op, const DenseM_t& R, DenseM_t& S) {
        TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
        gemm(c2T(op), Trans::N, scalar_t(1.), dF22, R, scalar_t(0.), S, task_depth);
        TIMER_STOP(t_sampling);
      };
      F22_->compress(sample_CB);
    }

    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    if (task_depth == 0)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      fwd_solve_node(b, work, etree_level, task_depth);
    else fwd_solve_node(b, work, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::fwd_solve_node
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t bupd(dim_upd(), b.cols(), work[0], 0, 0);
    bupd.zero();
    this->fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
      DenseM_t rhs(bloc);
      F11_.solve(rhs, bloc);
      if (dim_upd())
        F21_.mult('N', scalar_t(-1.), bloc, scalar_t(1.), bupd);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    if (task_depth == 0)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      bwd_solve_node(y, work, etree_level, task_depth);
    else bwd_solve_node(y, work, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::bwd_solve_node
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t yupd(dim_upd(), y.cols(), work[0], 0, 0);
    if (dim_sep() && dim_upd()) {
      DenseMW_t yloc(dim_sep(), y.cols(), y, this->sep_begin_, 0);
      F12_.mult('N', scalar_t(-1.), yupd, scalar_t(1.), yloc);
    }
    this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> integer_t
  FrontalMatrixHODLR<scalar_t,integer_t>::maximum_rank(int task_depth) const {
    integer_t r = /*_H.rank()*/ -1, rl = 0, rr = 0;
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
  FrontalMatrixHODLR<scalar_t,integer_t>::print_rank_statistics
  (std::ostream &out) const {
    if (lchild_) lchild_->print_rank_statistics(out);
    if (rchild_) rchild_->print_rank_statistics(out);
    out << "# HODLRMatrix rank info .... TODO" << std::endl;
    // _H.print_info(out);
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixHODLR<scalar_t,integer_t>::node_factor_nonzeros() const {
    return -1;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::draw_node
  (std::ostream& of, bool is_root) const {
    std::cout << "TODO draw" << std::endl;
    // if (is_root) _H.draw(of, sep_begin_, sep_begin_);
    // else _H.child(0)->draw(of, sep_begin_, sep_begin_);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::set_HODLR_partitioning
  (const SPOptions<scalar_t>& opts, const HSS::HSSPartitionTree& sep_tree,
   bool is_root) {
    assert(sep_tree.size == dim_sep());
    F11_ = std::move
      (HODLR::HODLRMatrix<scalar_t>
       (commself_, sep_tree, opts.HODLR_options()));
    if (!is_root && dim_upd()) {
      HSS::HSSPartitionTree CB_tree(dim_upd());
      CB_tree.refine(opts.HODLR_options().leaf_size());
      F22_ = std::unique_ptr<HODLR::HODLRMatrix<scalar_t>>
        (new HODLR::HODLRMatrix<scalar_t>
         (commself_, CB_tree, opts.HODLR_options()));
    }
  }

} // end namespace strumpack

#endif //FRONTAL_MATRIX_HODLR_HPP
