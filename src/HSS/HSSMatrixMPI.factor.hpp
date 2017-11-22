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
#ifndef HSS_MATRIX_MPI_FACTOR_HPP
#define HSS_MATRIX_MPI_FACTOR_HPP

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> HSSFactorsMPI<scalar_t>
    HSSMatrixMPI<scalar_t>::factor() const {
      HSSFactorsMPI<scalar_t> f;
      WorkFactorMPI<scalar_t> w;
      factor_recursive(f, w, _ctxt_loc, true, false);
      return f;
    }

    template<typename scalar_t> HSSFactorsMPI<scalar_t>
    HSSMatrixMPI<scalar_t>::partial_factor() const {
      HSSFactorsMPI<scalar_t> f;
      WorkFactorMPI<scalar_t> w;
      this->_ch[0]->factor_recursive(f, w, _ctxt_loc, true, true);
      return f;
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::factor_recursive
    (HSSFactorsMPI<scalar_t>& f, WorkFactorMPI<scalar_t>& w,
     int local_ctxt, bool isroot, bool partial) const {
      if (!this->active()) return;
      DistM_t Vh;
      if (!this->leaf()) {
        f._ch.resize(2);
        w.c.resize(2);
        this->_ch[0]->factor_recursive
          (f._ch[0], w.c[0], local_ctxt, false, partial);
        this->_ch[1]->factor_recursive
          (f._ch[1], w.c[1], local_ctxt, false, partial);
        auto c0u = _B01.rows();
        auto c1u = _B10.rows();
        auto c0v = _B10.cols();
        auto c1v = _B01.cols();
        auto u_rows = c0u + c1u;
        // move w.c[x].Vt1 from the child ctxt to the current ctxt
        DistM_t c0Vt1(_ctxt, c0u, c0v, w.c[0].Vt1, _ctxt_all);
        DistM_t c1Vt1(_ctxt, c1u, c1v, w.c[1].Vt1, _ctxt_all);
        if (u_rows) {
          f._D = DistM_t(_ctxt, u_rows, u_rows);
          copy(c0u, c0u, w.c[0].Dt, 0, 0, f._D, 0, 0, _ctxt_all);
          copy(c1u, c1u, w.c[1].Dt, 0, 0, f._D, c0u, c0u, _ctxt_all);
          DistMW_t D01(c0u, c1u, f._D, 0, c0u);
          DistMW_t D10(c1u, c0u, f._D, c0u, 0);
          gemm(Trans::N, Trans::C, scalar_t(1.),
               _B01, c1Vt1, scalar_t(0.), D01);
          gemm(Trans::N, Trans::C, scalar_t(1.),
               _B10, c0Vt1, scalar_t(0.), D10);
        }
        if (!isroot || partial) {
          Vh = DistM_t(_ctxt, this->U_rows(), this->V_rank());
          DistMW_t Vh0(c0u, this->V_rank(), Vh, 0, 0);
          DistMW_t Vh1(c1u, this->V_rank(), Vh, c0u, 0);
          auto V = _V.dense();
          DistMW_t V0(c0v, this->V_rank(), V, 0, 0);
          DistMW_t V1(c1v, this->V_rank(), V, c0v, 0);
          gemm(Trans::N, Trans::N, scalar_t(1.),
               c0Vt1, V0, scalar_t(0.), Vh0);
          gemm(Trans::N, Trans::N, scalar_t(1.),
               c1Vt1, V1, scalar_t(0.), Vh1);
        }
        w.c.clear();
      } else {
        f._D = _D;
        Vh = _V.dense();
      }
      if (isroot) {
        f._piv = f._D.LU();
        if (partial) f._Vt0 = std::move(Vh);
      } else {
        f._D.laswp(_U.P(), true); // compute P^t D
        if (this->U_rows() > this->U_rank()) {
          f._W1 = DistM_t(_ctxt, this->U_rank(), this->U_rows());
          // set W1 <- (P^t D)_0
          copy(this->U_rank(), this->U_rows(), f._D, 0, 0,
               f._W1, 0, 0, _ctxt_all);
          DistM_t W0(_ctxt, this->U_rows()-this->U_rank(), this->U_rows());
          // set W0 <- (P^t D)_1   (bottom part of P^t D)
          copy(W0.rows(), W0.cols(), f._D, this->U_rank(), 0,
               W0, 0, 0, _ctxt_all);
          f._D.clear();
          // set W0 <- -E * (P^t D)_0 + W0 = -E * W1 + W0
          gemm(Trans::N, Trans::N, scalar_t(-1.),
               _U.E(), f._W1, scalar_t(1.), W0);

          W0.LQ(f._L, f._Q);
          W0.clear();

          f._Vt0 = DistM_t
            (_ctxt, this->U_rows()-this->U_rank(), this->V_rank());
          w.Vt1 = DistM_t(_ctxt, this->U_rank(), this->V_rank());
          DistMW_t Q0
            (this->U_rows()-this->U_rank(), this->U_rows(), f._Q, 0, 0);
          DistMW_t Q1(this->U_rank(), this->U_rows(), f._Q, Q0.rows(), 0);
          gemm(Trans::N, Trans::N, scalar_t(1.),
               Q0, Vh, scalar_t(0.), f._Vt0); // Q0 * Vh
          gemm(Trans::N, Trans::N, scalar_t(1.),
               Q1, Vh, scalar_t(0.), w.Vt1);  // Q1 * Vh

          w.Dt = DistM_t(_ctxt, this->U_rank(), this->U_rank());
          gemm(Trans::N, Trans::C, scalar_t(1.),
               f._W1, Q1, scalar_t(0.), w.Dt); // W1 * Q1^c
        } else {
          w.Vt1 = std::move(Vh);
          w.Dt = std::move(f._D);
        }
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_MPI_FACTOR_HPP
