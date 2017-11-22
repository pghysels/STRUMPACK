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

    template<typename scalar_t> HSSFactors<scalar_t>
    HSSMatrix<scalar_t>::factor() const {
      HSSFactors<scalar_t> f;
      WorkFactor<scalar_t> w;
      factor_recursive(f, w, true, false, this->_openmp_task_depth);
      return f;
    }

    template<typename scalar_t> HSSFactors<scalar_t>
    HSSMatrix<scalar_t>::partial_factor() const {
      HSSFactors<scalar_t> f;
      WorkFactor<scalar_t> w;
      this->_ch[0]->factor_recursive
        (f, w, true, true, this->_openmp_task_depth);
      return f;
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::factor_recursive
    (HSSFactors<scalar_t>& f, WorkFactor<scalar_t>& w, bool isroot,
     bool partial, int depth) const {
      DenseM_t Vh;
      if (!this->leaf()) {
        f._ch.resize(2);
        w.c.resize(2);
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        this->_ch[0]->factor_recursive
          (f._ch[0], w.c[0], false, partial, depth+1);
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        this->_ch[1]->factor_recursive
          (f._ch[1], w.c[1], false, partial, depth+1);
#pragma omp taskwait
        auto u_rows = this->_ch[0]->U_rank() + this->_ch[1]->U_rank();
        if (u_rows) {
          f._D = DenseM_t(u_rows, u_rows);
          auto c0u = this->_ch[0]->U_rank();
          copy(w.c[0].Dt, f._D, 0, 0);
          copy(w.c[1].Dt, f._D, c0u, c0u);
          gemm(Trans::N, Trans::C, scalar_t(1.), _B01, w.c[1].Vt1,
               scalar_t(0.), f._D.ptr(0, c0u), f._D.ld(), depth);
          gemm(Trans::N, Trans::C, scalar_t(1.), _B10, w.c[0].Vt1,
               scalar_t(0.), f._D.ptr(c0u, 0), f._D.ld(), depth);
        }
        if (!isroot || partial) {
          Vh = DenseM_t(_U.rows(), _V.cols());
          DenseMW_t Vh0(this->_ch[0]->U_rank(), _V.cols(), Vh, 0, 0);
          DenseMW_t Vh1
            (this->_ch[1]->U_rank(), _V.cols(), Vh,
             this->_ch[0]->U_rank(), 0);
          auto V = _V.dense();
          gemm(Trans::N, Trans::N, scalar_t(1.),
               w.c[0].Vt1, V.ptr(0, 0), V.ld(), scalar_t(0.), Vh0, depth);
          gemm(Trans::N, Trans::N, scalar_t(1.),
               w.c[1].Vt1, V.ptr(this->_ch[0]->V_rank(), 0), V.ld(),
               scalar_t(0.), Vh1, depth);
        }
        w.c.clear();
      } else {
        f._D = _D;
        Vh = _V.dense();
      }
      if (isroot) {
        f._piv = f._D.LU(depth);
        if (partial) f._Vt0 = std::move(Vh);
      } else {
        f._D.laswp(_U.P(), true); // compute P^t D
        if (_U.rows() > _U.cols()) {
          // set W1 <- (P^t D)_0
          f._W1 = DenseM_t(_U.cols(), _U.rows(), f._D, 0, 0);
          // set W0 <- (P^t D)_1   (bottom part of P^t D)
          DenseM_t W0(_U.rows()-_U.cols(), _U.rows(), f._D, _U.cols(), 0);
          f._D.clear();
          // set W0 <- -E * (P^t D)_0 + W0 = -E * W1 + W0
          gemm(Trans::N, Trans::N, scalar_t(-1.), _U.E(), f._W1,
               scalar_t(1.), W0, depth);

          W0.LQ(f._L, f._Q, depth);
          W0.clear();

          f._Vt0 = DenseM_t(_U.rows()-_U.cols(), _V.cols());
          w.Vt1 = DenseM_t(_U.cols(), _V.cols());
          DenseMW_t Q0(_U.rows()-_U.cols(), _U.rows(), f._Q, 0, 0);
          DenseMW_t Q1(_U.cols(), _U.rows(), f._Q, Q0.rows(), 0);
          gemm(Trans::N, Trans::N, scalar_t(1.), Q0, Vh,
               scalar_t(0.), f._Vt0, depth); // Q0 * Vh
          gemm(Trans::N, Trans::N, scalar_t(1.), Q1, Vh,
               scalar_t(0.), w.Vt1, depth); // Q1 * Vh

          w.Dt = DenseM_t(_U.cols(), _U.cols());
          gemm(Trans::N, Trans::C, scalar_t(1.), f._W1, Q1,
               scalar_t(0.), w.Dt, depth); // W1 * Q1^c
        } else {
          w.Vt1 = std::move(Vh);
          w.Dt = std::move(f._D);
        }
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_FACTOR_HPP
