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
#ifndef HSS_MATRIX_FACTOR_HPP
#define HSS_MATRIX_FACTOR_HPP

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::factor() {
      WorkFactor<scalar_t> w;
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      factor_recursive(w, true, false, this->openmp_task_depth_);
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::partial_factor() {
      this->ULV_ = HSSFactors<scalar_t>();
      WorkFactor<scalar_t> w;
      child(0)->factor_recursive
        (w, true, true, this->openmp_task_depth_);
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::factor_recursive
    (WorkFactor<scalar_t>& w, bool isroot, bool partial, int depth) {
      this->ULV_ = HSSFactors<scalar_t>();
      DenseM_t Vh;
      if (!this->leaf()) {
        w.c.resize(2);
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        child(0)->factor_recursive
          (w.c[0], false, partial, depth+1);
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        child(1)->factor_recursive
          (w.c[1], false, partial, depth+1);
#pragma omp taskwait
        auto u_rows = child(0)->U_rank() + child(1)->U_rank();
        if (u_rows) {
          this->ULV_.D_ = DenseM_t(u_rows, u_rows);
          auto c0u = child(0)->U_rank();
          copy(w.c[0].Dt, this->ULV_.D_, 0, 0);
          copy(w.c[1].Dt, this->ULV_.D_, c0u, c0u);
          gemm(Trans::N, Trans::C, scalar_t(1.), B01_, w.c[1].Vt1,
               scalar_t(0.), this->ULV_.D_.ptr(0, c0u), this->ULV_.D_.ld(), depth);
          gemm(Trans::N, Trans::C, scalar_t(1.), B10_, w.c[0].Vt1,
               scalar_t(0.), this->ULV_.D_.ptr(c0u, 0), this->ULV_.D_.ld(), depth);
          STRUMPACK_ULV_FACTOR_FLOPS
            (gemm_flops(Trans::N, Trans::C, scalar_t(1.), B01_, w.c[1].Vt1, scalar_t(0.)) +
             gemm_flops(Trans::N, Trans::C, scalar_t(1.), B10_, w.c[0].Vt1, scalar_t(0.)));
        }
        if (!isroot || partial) {
          Vh = DenseM_t(U_.rows(), V_.cols());
          DenseMW_t Vh0(child(0)->U_rank(), V_.cols(), Vh, 0, 0);
          DenseMW_t Vh1(child(1)->U_rank(), V_.cols(), Vh,
                        child(0)->U_rank(), 0);
          auto V = V_.dense();
          gemm(Trans::N, Trans::N, scalar_t(1.),
               w.c[0].Vt1, V.ptr(0, 0), V.ld(), scalar_t(0.), Vh0, depth);
          gemm(Trans::N, Trans::N, scalar_t(1.),
               w.c[1].Vt1, V.ptr(child(0)->V_rank(), 0), V.ld(),
               scalar_t(0.), Vh1, depth);
          STRUMPACK_ULV_FACTOR_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(1.),
                        w.c[0].Vt1, scalar_t(0.), Vh0) +
             gemm_flops(Trans::N, Trans::N, scalar_t(1.),
                        w.c[1].Vt1, scalar_t(0.), Vh1));
        }
        w.c.clear();
      } else {
        this->ULV_.D_ = D_;
        Vh = V_.dense();
      }
      if (isroot) {
        this->ULV_.piv_ = this->ULV_.D_.LU(depth);
        STRUMPACK_ULV_FACTOR_FLOPS(LU_flops(this->ULV_.D_));
        if (partial) this->ULV_.Vt0_ = std::move(Vh);
      } else {
        this->ULV_.D_.laswp(U_.P(), true); // compute P^t D
        if (U_.rows() > U_.cols()) {
          // set W1 <- (P^t D)_0
          this->ULV_.W1_ = DenseM_t(U_.cols(), U_.rows(), this->ULV_.D_, 0, 0);
          // set W0 <- (P^t D)_1   (bottom part of P^t D)
          DenseM_t W0(U_.rows()-U_.cols(), U_.rows(), this->ULV_.D_, U_.cols(), 0);
          this->ULV_.D_.clear();
          // set W0 <- -E * (P^t D)_0 + W0 = -E * W1 + W0
          gemm(Trans::N, Trans::N, scalar_t(-1.), U_.E(), this->ULV_.W1_,
               scalar_t(1.), W0, depth);
          STRUMPACK_ULV_FACTOR_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(-1.), U_.E(), this->ULV_.W1_, scalar_t(1.)));

          W0.LQ(this->ULV_.L_, this->ULV_.Q_, depth);
          STRUMPACK_ULV_FACTOR_FLOPS(LQ_flops(W0));
          W0.clear();

          this->ULV_.Vt0_ = DenseM_t(U_.rows()-U_.cols(), V_.cols());
          w.Vt1 = DenseM_t(U_.cols(), V_.cols());
          DenseMW_t Q0(U_.rows()-U_.cols(), U_.rows(), this->ULV_.Q_, 0, 0);
          DenseMW_t Q1(U_.cols(), U_.rows(), this->ULV_.Q_, Q0.rows(), 0);
          gemm(Trans::N, Trans::N, scalar_t(1.), Q0, Vh,
               scalar_t(0.), this->ULV_.Vt0_, depth); // Q0 * Vh
          gemm(Trans::N, Trans::N, scalar_t(1.), Q1, Vh,
               scalar_t(0.), w.Vt1, depth); // Q1 * Vh

          w.Dt = DenseM_t(U_.cols(), U_.cols());
          gemm(Trans::N, Trans::C, scalar_t(1.), this->ULV_.W1_, Q1,
               scalar_t(0.), w.Dt, depth); // W1 * Q1^c
          STRUMPACK_ULV_FACTOR_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(1.), Q0, Vh, scalar_t(0.)) +
             gemm_flops(Trans::N, Trans::N, scalar_t(1.), Q0, Vh, scalar_t(0.)) +
             gemm_flops(Trans::N, Trans::C, scalar_t(1.), this->ULV_.W1_, Q1, scalar_t(0.)));
        } else {
          w.Vt1 = std::move(Vh);
          w.Dt = std::move(this->ULV_.D_);
        }
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_FACTOR_HPP
