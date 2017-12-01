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
#ifndef FRONTAL_MATRIX_DENSE_HPP
#define FRONTAL_MATRIX_DENSE_HPP

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>

#include "misc/TaskTimer.hpp"
#include "dense/BLASLAPACKWrapper.hpp"
#include "CompressedSparseMatrix.hpp"
#include "MatrixReordering.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixDense
    : public FrontalMatrix<scalar_t,integer_t> {
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;

  public:
    DenseM_t F11, F12, F21, F22;
    std::vector<int> piv; // regular int because it is passed to BLAS

    FrontalMatrixDense
    (integer_t _sep, integer_t _sep_begin, integer_t _sep_end,
     std::vector<integer_t>& _upd);
    ~FrontalMatrixDense() {}
    void release_work_memory() { F22.clear(); }
    void extend_add_to_dense
    (FrontalMatrixDense<scalar_t,integer_t>* p, int task_depth) override;
    void sample_CB
    (const SPOptions<scalar_t>& opts, const DenseM_t& R, DenseM_t& Sr,
     DenseM_t& Sc, FrontalMatrix<scalar_t,integer_t>* pa,
     int task_depth) override;
    void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level=0, int task_depth=0) override;

    void forward_multifrontal_solve
    (DenseM_t& b, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;
    void backward_multifrontal_solve
    (DenseM_t& y, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;

    void extract_CB_sub_matrix
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DenseM_t& B, int task_depth) const override;
    std::string type() const override { return "FrontalMatrixDense"; }

  private:
    FrontalMatrixDense(const FrontalMatrixDense&) = delete;
    FrontalMatrixDense& operator=(FrontalMatrixDense const&) = delete;

    void factor_phase1
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level, int task_depth);
    void factor_phase2
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level, int task_depth);

    void fwd_solve_phase1
    (DenseM_t& b, DenseM_t& bupd, DenseM_t* work,
     int etree_level, int task_depth) const;
    void fwd_solve_phase2
    (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const;
    void bwd_solve_phase1
    (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const;
    void bwd_solve_phase2
    (DenseM_t& y, DenseM_t& yupd, DenseM_t* work,
     int etree_level, int task_depth) const;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixDense<scalar_t,integer_t>::FrontalMatrixDense
  (integer_t _sep, integer_t _sep_begin, integer_t _sep_end,
   std::vector<integer_t>& _upd)
    : FrontalMatrix<scalar_t,integer_t>
    (NULL, NULL, _sep, _sep_begin, _sep_end, _upd) {}

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::extend_add_to_dense
  (FrontalMatrixDense<scalar_t,integer_t>* p, int task_depth) {
    const std::size_t pdsep = p->dim_sep();
    const std::size_t dupd = this->dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
#pragma omp taskloop default(shared) grainsize(64)      \
  if(task_depth < params::task_recursion_cutoff_level)
    for (std::size_t c=0; c<dupd; c++) {
      auto pc = I[c];
      if (pc < pdsep) {
        for (std::size_t r=0; r<upd2sep; r++)
          p->F11(I[r],pc) += F22(r,c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          p->F21(I[r]-pdsep,pc) += F22(r,c);
      } else {
        for (std::size_t r=0; r<upd2sep; r++)
          p->F12(I[r],pc-pdsep) += F22(r, c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          p->F22(I[r]-pdsep,pc-pdsep) += F22(r,c);
      }
    }
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*
                    static_cast<long long int>(dupd*dupd));
    release_work_memory();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::sample_CB
  (const SPOptions<scalar_t>& opts, const DenseM_t& R, DenseM_t& Sr,
   DenseM_t& Sc, FrontalMatrix<scalar_t,integer_t>* pa, int task_depth) {
    auto I = this->upd_to_parent(pa);
    auto cR = R.extract_rows(I);
    DenseM_t cS(this->dim_upd(), R.cols());
    gemm(Trans::N, Trans::N, scalar_t(1.), F22, cR,
         scalar_t(0.), cS, task_depth);
    Sr.scatter_rows_add(I, cS);
    gemm(Trans::C, Trans::N, scalar_t(1.), F22, cR,
         scalar_t(0.), cS, task_depth);
    Sc.scatter_rows_add(I, cS);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (task_depth == 0) {
      // use tasking for children and for extend-add parallelism
#pragma omp parallel if(!omp_in_parallel()) default(shared)
#pragma omp single
      factor_phase1(A, opts, etree_level, task_depth);
      // do not use tasking for blas/lapack parallelism (use system
      // blas threading!)
      factor_phase2(A, opts, etree_level, params::task_recursion_cutoff_level);
    } else {
      factor_phase1(A, opts, etree_level, task_depth);
      factor_phase2(A, opts, etree_level, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::factor_phase1
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (task_depth < params::task_recursion_cutoff_level) {
      if (this->lchild)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        this->lchild->multifrontal_factorization
          (A, opts, etree_level+1, task_depth+1);
      if (this->rchild)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        this->rchild->multifrontal_factorization
          (A, opts, etree_level+1, task_depth+1);
#pragma omp taskwait
    } else {
      if (this->lchild)
        this->lchild->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
      if (this->rchild)
        this->rchild->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
    }
    auto f0 = params::flops;
    // TODO can we allocate the memory in one go??
    const auto dsep = this->dim_sep();
    const auto dupd = this->dim_upd();
    F11 = DenseM_t(dsep, dsep); F11.zero();
    F12 = DenseM_t(dsep, dupd); F12.zero();
    F21 = DenseM_t(dupd, dsep); F21.zero();
    A.extract_front
      (F11, F12, F21, this->sep_begin, this->sep_end, this->upd, task_depth);
    if (dupd) {
      F22 = DenseM_t(dupd, dupd);
      F22.zero();
    }
    if (this->lchild) this->lchild->extend_add_to_dense(this, task_depth);
    if (this->rchild) this->rchild->extend_add_to_dense(this, task_depth);
    params::full_rank_flops += params::flops - f0;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::factor_phase2
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    auto f0 = params::flops;
    if (this->dim_sep()) {
      piv = F11.LU(task_depth);
      if (opts.replace_tiny_pivots()) {
        // TODO consider other values for thresh
        //  - sqrt(eps)*|A|_1 as in SuperLU ?
        auto thresh = blas::lamch<real_t>('E') * A.size();
        for (std::size_t i=0; i<F11.rows(); i++)
          if (std::abs(F11(i,i)) < thresh)
            F11(i,i) = (std::real(F11(i,i)) < 0) ? -thresh : thresh;
      }
      if (this->dim_upd()) {
        F12.laswp(piv, true);
        trsm
          (Side::L, UpLo::L, Trans::N, Diag::U,
           scalar_t(1.), F11, F12, task_depth);
        trsm
          (Side::R, UpLo::U, Trans::N, Diag::N,
           scalar_t(1.), F11, F21, task_depth);
        gemm
          (Trans::N, Trans::N, scalar_t(-1.), F21, F12,
           scalar_t(1.), F22, task_depth);
      }
    }
    params::full_rank_flops += params::flops - f0;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t bupd(this->dim_upd(), b.cols(), work[0], 0, 0);
    bupd.zero();
    if (task_depth == 0) {
      // tasking when calling the children
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
      // no tasking for the root node computations, use system blas threading!
      return fwd_solve_phase2
        (b, bupd, etree_level, params::task_recursion_cutoff_level);
    } else {
      fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
      return fwd_solve_phase2(b, bupd, etree_level, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::fwd_solve_phase1
  (DenseM_t& b, DenseM_t& bupd, DenseM_t* work,
   int etree_level, int task_depth) const {
    if (task_depth < params::task_recursion_cutoff_level) {
      if (this->lchild)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        this->lchild->forward_multifrontal_solve
          (b, work+1, etree_level+1, task_depth+1);
      if (this->rchild)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          std::vector<DenseM_t> work2(this->rchild->levels());
          for (auto& cb : work2)
            cb = DenseM_t(this->rchild->max_dim_upd(), b.cols());
          this->rchild->forward_multifrontal_solve
            (b, work2.data(), etree_level+1, task_depth+1);
          DenseMW_t CBch
            (this->rchild->dim_upd(), b.cols(), work2[0], 0, 0);
          this->extend_add_b(this->rchild, b, bupd, CBch);
        }
#pragma omp taskwait
      if (this->lchild) {
        DenseMW_t CBch
          (this->lchild->dim_upd(), b.cols(), work[1], 0, 0);
        this->extend_add_b(this->lchild, b, bupd, CBch);
      }
    } else {
      if (this->lchild) {
        this->lchild->forward_multifrontal_solve
          (b, work+1, etree_level+1, task_depth);
        DenseMW_t CBch
          (this->lchild->dim_upd(), b.cols(), work[1], 0, 0);
        this->extend_add_b(this->lchild, b, bupd, CBch);
      }
      if (this->rchild) {
        this->rchild->forward_multifrontal_solve
          (b, work+1, etree_level+1, task_depth);
        DenseMW_t CBch
          (this->rchild->dim_upd(), b.cols(), work[1], 0, 0);
        this->extend_add_b(this->rchild, b, bupd, CBch);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    if (this->dim_sep()) {
      DenseMW_t bloc(this->dim_sep(), b.cols(), b, this->sep_begin, 0);
      bloc.laswp(piv, true);
      if (b.cols() == 1) {
        trsv(UpLo::L, Trans::N, Diag::U, F11, bloc, task_depth);
        if (this->dim_upd())
          gemv(Trans::N, scalar_t(-1.), F21, bloc,
               scalar_t(1.), bupd, task_depth);
      } else {
        trsm(Side::L, UpLo::L, Trans::N, Diag::U,
             scalar_t(1.), F11, bloc, task_depth);
        if (this->dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F21, bloc,
               scalar_t(1.), bupd, task_depth);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t yupd(this->dim_upd(), y.cols(), work[0], 0, 0);
    if (task_depth == 0) {
      // no tasking in blas routines, use system threaded blas instead
      bwd_solve_phase1
        (y, yupd, etree_level, params::task_recursion_cutoff_level);
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      // tasking when calling children
      bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
    } else {
      bwd_solve_phase1(y, yupd, etree_level, task_depth);
      bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::bwd_solve_phase1
  (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const {
    if (this->dim_sep()) {
      DenseMW_t yloc(this->dim_sep(), y.cols(), y, this->sep_begin, 0);
      if (y.cols() == 1) {
        if (this->dim_upd())
          gemv(Trans::N, scalar_t(-1.), F12, yupd,
               scalar_t(1.), yloc, task_depth);
        trsv(UpLo::U, Trans::N, Diag::N, F11, yloc, task_depth);
      } else {
        if (this->dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F12, yupd,
               scalar_t(1.), yloc, task_depth);
        trsm(Side::L, UpLo::U, Trans::N, Diag::N, scalar_t(1.),
             F11, yloc, task_depth);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::bwd_solve_phase2
  (DenseM_t& y, DenseM_t& yupd, DenseM_t* work,
   int etree_level, int task_depth) const {
    if (task_depth < params::task_recursion_cutoff_level) {
      if (this->lchild) {
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          DenseMW_t CB(this->lchild->dim_upd(), y.cols(), work[1], 0, 0);
          this->extract_b(this->lchild, y, yupd, CB);
          this->lchild->backward_multifrontal_solve
            (y, work+1, etree_level+1, task_depth+1);
        }
      }
      if (this->rchild)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          std::vector<DenseM_t> work2(this->rchild->levels());
          for (auto& cb : work2)
            cb = DenseM_t(this->rchild->max_dim_upd(), y.cols());
          DenseMW_t CB(this->rchild->dim_upd(), y.cols(), work2[0], 0, 0);
          this->extract_b(this->rchild, y, yupd, CB);
          this->rchild->backward_multifrontal_solve
            (y, work2.data(), etree_level+1, task_depth+1);
        }
#pragma omp taskwait
    } else {
      if (this->lchild) {
        DenseMW_t CB(this->lchild->dim_upd(), y.cols(), work[1], 0, 0);
        this->extract_b(this->lchild, y, yupd, CB);
        this->lchild->backward_multifrontal_solve
          (y, work+1, etree_level+1, task_depth);
      }
      if (this->rchild) {
        DenseMW_t CB(this->rchild->dim_upd(), y.cols(), work[1], 0, 0);
        this->extract_b(this->rchild, y, yupd, CB);
        this->rchild->backward_multifrontal_solve
          (y, work+1, etree_level+1, task_depth);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::extract_CB_sub_matrix
  (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
   DenseM_t& B, int task_depth) const {
    std::vector<std::size_t> lJ, oJ;
    this->find_upd_indices(J, lJ, oJ);
    if (lJ.empty()) return;
    std::vector<std::size_t> lI, oI;
    this->find_upd_indices(I, lI, oI);
    if (lI.empty()) return;
    for (std::size_t j=0; j<lJ.size(); j++)
      for (std::size_t i=0; i<lI.size(); i++)
        B(oI[i], oJ[j]) += F22(lI[i], lJ[j]);
    STRUMPACK_FLOPS((is_complex<scalar_t>() ? 2 : 1) * lJ.size() * lI.size());
  }

} // end namespace strumpack

#endif
