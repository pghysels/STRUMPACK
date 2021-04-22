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

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::factor() {
      WorkFactorMPI<scalar_t> w;
      factor_recursive(w, grid_local(), true, false);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::partial_factor() {
      this->ULV_mpi_ = HSSFactorsMPI<scalar_t>();
      WorkFactorMPI<scalar_t> w;
      child(0)->factor_recursive(w, grid_local(), true, true);
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::factor_recursive
    (WorkFactorMPI<scalar_t>& w, const BLACSGrid* lg,
     bool isroot, bool partial) {
      this->ULV_mpi_ = HSSFactorsMPI<scalar_t>();
      if (!this->active()) return;
      DistM_t Vh;
      if (!this->leaf()) {
        w.c.resize(2);
        child(0)->factor_recursive(w.c[0], lg, false, partial);
        child(1)->factor_recursive(w.c[1], lg, false, partial);
        auto c0u = B01_.rows();
        auto c1u = B10_.rows();
        auto c0v = B10_.cols();
        auto c1v = B01_.cols();
        auto u_rows = c0u + c1u;
        // move w.c[x].Vt1 from the child grid to the current grid
        DistM_t c0Vt1(grid(), c0u, c0v, w.c[0].Vt1, grid()->ctxt_all());
        DistM_t c1Vt1(grid(), c1u, c1v, w.c[1].Vt1, grid()->ctxt_all());
        if (u_rows) {
          this->ULV_mpi_.D_ = DistM_t(grid(), u_rows, u_rows);
          copy(c0u, c0u, w.c[0].Dt, 0, 0, this->ULV_mpi_.D_, 0, 0, grid()->ctxt_all());
          copy(c1u, c1u, w.c[1].Dt, 0, 0, this->ULV_mpi_.D_, c0u, c0u, grid()->ctxt_all());
          DistMW_t D01(c0u, c1u, this->ULV_mpi_.D_, 0, c0u);
          DistMW_t D10(c1u, c0u, this->ULV_mpi_.D_, c0u, 0);
          gemm(Trans::N, Trans::C, scalar_t(1.), B01_, c1Vt1, scalar_t(0.), D01);
          gemm(Trans::N, Trans::C, scalar_t(1.), B10_, c0Vt1, scalar_t(0.), D10);
          STRUMPACK_ULV_FACTOR_FLOPS
            (gemm_flops(Trans::N, Trans::C, scalar_t(1.), B01_, c1Vt1, scalar_t(0.)) +
             gemm_flops(Trans::N, Trans::C, scalar_t(1.), B10_, c0Vt1, scalar_t(0.)));
        }
        if (!isroot || partial) {
          Vh = DistM_t(grid(), this->U_rows(), this->V_rank());
          DistMW_t Vh0(c0u, this->V_rank(), Vh, 0, 0);
          DistMW_t Vh1(c1u, this->V_rank(), Vh, c0u, 0);
          auto V = V_.dense();
          DistMW_t V0(c0v, this->V_rank(), V, 0, 0);
          DistMW_t V1(c1v, this->V_rank(), V, c0v, 0);
          gemm(Trans::N, Trans::N, scalar_t(1.), c0Vt1, V0, scalar_t(0.), Vh0);
          gemm(Trans::N, Trans::N, scalar_t(1.), c1Vt1, V1, scalar_t(0.), Vh1);
          STRUMPACK_ULV_FACTOR_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(1.), c0Vt1, V0, scalar_t(0.)) +
             gemm_flops(Trans::N, Trans::N, scalar_t(1.), c1Vt1, V1, scalar_t(0.)));
        }
        w.c.clear();
      } else {
        this->ULV_mpi_.D_ = D_;
        Vh = V_.dense();
      }
      if (isroot) {
        this->ULV_mpi_.piv_ = this->ULV_mpi_.D_.LU();
        STRUMPACK_ULV_FACTOR_FLOPS(LU_flops(this->ULV_mpi_.D_));
        if (partial) this->ULV_mpi_.Vt0_ = std::move(Vh);
      } else {
        this->ULV_mpi_.D_.laswp(U_.P(), true); // compute P^t D
        if (this->U_rows() > this->U_rank()) {
          this->ULV_mpi_.W1_ = DistM_t(grid(), this->U_rank(), this->U_rows());
          // set W1 <- (P^t D)_0
          copy(this->U_rank(), this->U_rows(), this->ULV_mpi_.D_, 0, 0,
               this->ULV_mpi_.W1_, 0, 0, grid()->ctxt_all());
          DistM_t W0(grid(), this->U_rows()-this->U_rank(), this->U_rows());
          // set W0 <- (P^t D)_1   (bottom part of P^t D)
          copy(W0.rows(), W0.cols(), this->ULV_mpi_.D_, this->U_rank(), 0, W0, 0, 0, grid()->ctxt_all());
          this->ULV_mpi_.D_.clear();
          // set W0 <- -E * (P^t D)_0 + W0 = -E * W1 + W0
          gemm(Trans::N, Trans::N, scalar_t(-1.), U_.E(), this->ULV_mpi_.W1_, scalar_t(1.), W0);
          STRUMPACK_ULV_FACTOR_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(-1.), U_.E(), this->ULV_mpi_.W1_, scalar_t(1.)));

          W0.LQ(this->ULV_mpi_.L_, this->ULV_mpi_.Q_);
          STRUMPACK_ULV_FACTOR_FLOPS(LQ_flops(W0));
          W0.clear();

          this->ULV_mpi_.Vt0_ = DistM_t
            (grid(), this->U_rows()-this->U_rank(), this->V_rank());
          w.Vt1 = DistM_t(grid(), this->U_rank(), this->V_rank());
          DistMW_t Q0
            (this->U_rows()-this->U_rank(), this->U_rows(), this->ULV_mpi_.Q_, 0, 0);
          DistMW_t Q1(this->U_rank(), this->U_rows(), this->ULV_mpi_.Q_, Q0.rows(), 0);
          gemm(Trans::N, Trans::N, scalar_t(1.),
               Q0, Vh, scalar_t(0.), this->ULV_mpi_.Vt0_); // Q0 * Vh
          gemm(Trans::N, Trans::N, scalar_t(1.),
               Q1, Vh, scalar_t(0.), w.Vt1);  // Q1 * Vh

          w.Dt = DistM_t(grid(), this->U_rank(), this->U_rank());
          gemm(Trans::N, Trans::C, scalar_t(1.),
               this->ULV_mpi_.W1_, Q1, scalar_t(0.), w.Dt); // W1 * Q1^c
          STRUMPACK_ULV_FACTOR_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(1.), Q0, Vh, scalar_t(0.)) +
             gemm_flops(Trans::N, Trans::N, scalar_t(1.), Q1, Vh, scalar_t(0.)) +
             gemm_flops(Trans::N, Trans::C, scalar_t(1.), this->ULV_mpi_.W1_, Q1, scalar_t(0.)));
        } else {
          w.Vt1 = std::move(Vh);
          w.Dt = std::move(this->ULV_mpi_.D_);
        }
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_MPI_FACTOR_HPP
