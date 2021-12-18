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
#ifndef HSS_MATRIX_MPI_SOLVE_HPP
#define HSS_MATRIX_MPI_SOLVE_HPP

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::solve(DistM_t& b) const {
      assert(std::size_t(b.rows()) == this->rows());
      // TODO assert that the ULV factorization has been performed and
      // is a valid one
      // assert(ULV.D_.rows() == U_.rows());
      WorkSolveMPI<scalar_t> w;
      DistSubLeaf<scalar_t> B(b.cols(), this, grid_local(), b);
      solve_fwd(B, w, false, true);
      solve_bwd(B, w, true);
      B.from_block_row(b);
    }

    // TODO do not pass work, just return the reduced_rhs, and w.x at the root
    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::forward_solve
    (WorkSolveMPI<scalar_t>& w, const DistM_t& b, bool partial) const {
      DistSubLeaf<scalar_t> B(b.cols(), this, grid_local(), b);
      solve_fwd(B, w, partial, true);
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::backward_solve
    (WorkSolveMPI<scalar_t>& w, DistM_t& x) const {
      DistSubLeaf<scalar_t> X(x.cols(), this, grid_local());
      solve_bwd(X, w, true);
      X.from_block_row(x);
    }

    // have this routine return ft1, or x at the root!!!
    // then ft1 and x do not need to be stored in WorkSolve!!
    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::solve_fwd
    (const DistSubLeaf<scalar_t>& b, WorkSolveMPI<scalar_t>& w,
     bool partial, bool isroot) const {
      if (!this->active()) return;
      auto n = b.cols();
      DistM_t f, z;
      if (this->leaf()) {
        f = DistM_t(grid(), this->rows(), n);
        copy(this->rows(), n, b.leaf, 0, 0, f, 0, 0, grid()->ctxt_all());
      } else {
        w.c.resize(2);
        child(0)->solve_fwd(b, w.c[0], partial, false);
        child(1)->solve_fwd(b, w.c[1], partial, false);
        auto c0urank = child(0)->U_rank();
        auto c1urank = child(1)->U_rank();
        auto c0vrank = child(0)->V_rank();
        auto c1vrank = child(1)->V_rank();
        auto c0urows = child(0)->U_rows();
        auto c1urows = child(1)->U_rows();
        auto urows = c0urank + c1urank;
        auto vrows = c0vrank + c1vrank;
        f = DistM_t(grid(), urows, n);
        DistMW_t f0(c0urank, n, f, 0, 0);
        DistMW_t f1(c1urank, n, f, c0urank, 0);
        copy(c0urank, n, w.c[0].ft1, 0, 0, f0, 0, 0, grid()->ctxt_all());
        copy(c1urank, n, w.c[1].ft1, 0, 0, f1, 0, 0, grid()->ctxt_all());
        w.c[0].ft1.clear();
        w.c[1].ft1.clear();
        z = DistM_t(grid(), vrows, n);
        DistMW_t z0(c0vrank, n, z, 0, 0);
        DistMW_t z1(c1vrank, n, z, c0vrank, 0);
        copy(c0vrank, n, w.c[0].z, 0, 0, z0, 0, 0, grid()->ctxt_all());
        copy(c1vrank, n, w.c[1].z, 0, 0, z1, 0, 0, grid()->ctxt_all());
        w.c[0].z.clear();
        w.c[1].z.clear();
        DistM_t y0(grid(), c0urows-c0urank, n);
        DistM_t y1(grid(), c1urows-c1urank, n);
        // TODO as extra optimization, combine these pgemr2d's?
        copy(y0.rows(), n, w.c[0].y, 0, 0, y0, 0, 0, grid()->ctxt_all());
        copy(y1.rows(), n, w.c[1].y, 0, 0, y1, 0, 0, grid()->ctxt_all());
        gemm(Trans::N, Trans::N, scalar_t(-1.), B01_, z1, scalar_t(1.), f0);
        gemm(Trans::N, Trans::N, scalar_t(-1.), B10_, z0, scalar_t(1.), f1);
        STRUMPACK_HSS_SOLVE_FLOPS
          (gemm_flops(Trans::N, Trans::N, scalar_t(-1.), B01_, z1, scalar_t(1.)) +
           gemm_flops(Trans::N, Trans::N, scalar_t(-1.), B10_, z0, scalar_t(1.)));
        if (c0urows > c0urank) {
          DistM_t Q00(grid(), c0urows - c0urank, c0urows);
          copy(Q00.rows(), Q00.cols(), child(0)->ULV_mpi_.Q_, 0, 0,
               Q00, 0, 0, grid()->ctxt_all());
          DistM_t tmp0(grid(), c0urows, n);
          DistM_t c0W1(grid(), c0urank, c0urows);
          copy(c0W1.rows(), c0W1.cols(), child(0)->ULV_mpi_.W1_, 0, 0,
               c0W1, 0, 0, grid()->ctxt_all());
          gemm(Trans::C, Trans::N, scalar_t(1.), Q00, y0, scalar_t(0.), tmp0);
          gemm(Trans::N, Trans::N, scalar_t(-1.),
               c0W1, tmp0, scalar_t(1.), f0);
          STRUMPACK_HSS_SOLVE_FLOPS
            (gemm_flops(Trans::C, Trans::N, scalar_t(1.), Q00, y0, scalar_t(0.)) +
             gemm_flops(Trans::N, Trans::N, scalar_t(-1.), c0W1, tmp0, scalar_t(1.)));
        }
        if (c1urows > c1urank) {
          DistM_t Q10(grid(), c1urows - c1urank, c1urows);
          copy(c1urows - c1urank, c1urows, child(1)->ULV_mpi_.Q_, 0, 0,
               Q10, 0, 0, grid()->ctxt_all());
          DistM_t tmp1(grid(), Q10.cols(), n);
          DistM_t c1W1(grid(), c1urank, c1urows);
          copy(c1W1.rows(), c1W1.cols(), child(1)->ULV_mpi_.W1_, 0, 0,
               c1W1, 0, 0, grid()->ctxt_all());
          gemm(Trans::C, Trans::N, scalar_t(1.), Q10, y1, scalar_t(0.), tmp1);
          gemm(Trans::N, Trans::N, scalar_t(-1.), c1W1, tmp1, scalar_t(1.), f1);
          STRUMPACK_HSS_SOLVE_FLOPS
            (gemm_flops(Trans::C, Trans::N, scalar_t(1.), Q10, y1, scalar_t(0.)) +
             gemm_flops(Trans::N, Trans::N, scalar_t(-1.), c1W1, tmp1, scalar_t(1.)));
        }
      }
      if (isroot) {
        w.x = this->ULV_mpi_.D_.solve(f, this->ULV_mpi_.piv_);
        STRUMPACK_HSS_SOLVE_FLOPS(solve_flops(f));
        if (partial) {
          // compute reduced_rhs = \hat{V}^* y_0 + V^* [z_0; z_1]
          w.reduced_rhs = DistM_t(grid(), this->V_rank(), w.x.cols());
          gemm(Trans::C, Trans::N, scalar_t(1.), this->ULV_mpi_.Vt0_, w.x, scalar_t(0.), w.reduced_rhs);
          STRUMPACK_HSS_SOLVE_FLOPS
            (gemm_flops(Trans::C, Trans::N, scalar_t(1.), this->ULV_mpi_.Vt0_, w.x, scalar_t(0.)));
          if (!this->leaf()) {
            w.reduced_rhs.add(V_.applyC(z));
            STRUMPACK_HSS_SOLVE_FLOPS
              (V_.applyC_flops(z.cols()) + w.reduced_rhs.lrows() * w.reduced_rhs.lcols());
          }
        }
      } else {
        f.laswp(U_.P(), true);
        if (this->U_rows() > this->U_rank()) {
          w.ft1 = DistM_t(grid(), this->U_rank(), n);
          copy(w.ft1.rows(), n, f, 0, 0, w.ft1, 0, 0, grid()->ctxt_all());
          w.y = DistM_t(grid(), this->U_rows()-this->U_rank(), n);
          // put ft0 in w.y
          copy(w.y.rows(), n, f, this->U_rank(), 0, w.y, 0, 0, grid()->ctxt_all());
          gemm(Trans::N, Trans::N, scalar_t(-1.), U_.E(), w.ft1, scalar_t(1.), w.y);
          trsm(Side::L, UpLo::L, Trans::N, Diag::N, scalar_t(1.), this->ULV_mpi_.L_, w.y);
          STRUMPACK_HSS_SOLVE_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(-1.),
                        U_.E(), w.ft1, scalar_t(1.)) +
             trsm_flops(Side::L, scalar_t(1.), this->ULV_mpi_.L_, w.y));
          if (!this->leaf()) {
            // TODO do the concat first, create wrappers for z0 and z1
            w.z = V_.applyC(z);
            gemm(Trans::C, Trans::N, scalar_t(1.), this->ULV_mpi_.Vt0_, w.y,
                 scalar_t(1.), w.z);
            STRUMPACK_HSS_SOLVE_FLOPS
              (V_.applyC_flops(z.cols()) +
               gemm_flops(Trans::C, Trans::N, scalar_t(1.),
                          this->ULV_mpi_.Vt0_, w.y, scalar_t(1.)));
          } else {
            w.z = DistM_t(grid(), this->V_rank(), n);
            gemm(Trans::C, Trans::N, scalar_t(1.), this->ULV_mpi_.Vt0_, w.y,
                 scalar_t(0.), w.z);
            STRUMPACK_HSS_SOLVE_FLOPS
              (gemm_flops(Trans::C, Trans::N, scalar_t(1.),
                          this->ULV_mpi_.Vt0_, w.y, scalar_t(0.)));
          }
        } else {
          w.ft1 = DistM_t(grid(), this->U_rank(), n);
          w.y = DistM_t(grid(), 0, n);
          copy(this->U_rank(), n, f, 0, 0, w.ft1, 0, 0, grid()->ctxt_all());
          if (!this->leaf()) {
            w.z = V_.applyC(z);
            STRUMPACK_HSS_SOLVE_FLOPS(V_.applyC_flops(z.cols()));
          } else {
            w.z = DistM_t(grid(), this->V_rank(), n);
            w.z.zero(); // TODO can this be avoided?
          }
        }
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::solve_bwd
    (DistSubLeaf<scalar_t>& x, WorkSolveMPI<scalar_t>& w, bool isroot) const {
      if (!this->active()) return;
      auto n = x.cols();
      if (this->leaf())
        copy(this->rows(), n, w.x, 0, 0, x.leaf, 0, 0, grid()->ctxt_all());
      else {
        auto c0urank = child(0)->U_rank();
        auto c1urank = child(1)->U_rank();
        auto c0urows = child(0)->U_rows();
        auto c1urows = child(1)->U_rows();
        {
          w.c[0].x = DistM_t(w.c[0].y.grid(), c0urows, n);
          if (c0urows > c0urank) {
            DistM_t tmp(w.c[0].y.grid(), c0urows, n);
            // TODO this first one is on the same context, should not
            // need communication!!
            copy(c0urows-c0urank, n, w.c[0].y, 0, 0, tmp, 0, 0, grid()->ctxt_all());
            copy(c0urank, n, w.x, 0, 0, tmp, c0urows-c0urank, 0, grid()->ctxt_all());
            gemm(Trans::C, Trans::N, scalar_t(1.), child(0)->ULV_mpi_.Q_, tmp,
                 scalar_t(0.), w.c[0].x);
            STRUMPACK_HSS_SOLVE_FLOPS
              (gemm_flops(Trans::C, Trans::N, scalar_t(1.),
                          child(0)->ULV_mpi_.Q_, tmp, scalar_t(0.)));
          } else copy(c0urank, n, w.x, 0, 0, w.c[0].x, 0, 0, grid()->ctxt_all());
        } {
          w.c[1].x = DistM_t(w.c[1].y.grid(), c1urows, n);
          if (c1urows > c1urank) {
            // TODO this first one is on the same context, should not
            // need communication!!
            DistM_t tmp(w.c[1].y.grid(), c1urows, n);
            copy(c1urows-c1urank, n, w.c[1].y, 0, 0, tmp, 0, 0, grid()->ctxt_all());
            copy(c1urank, n, w.x, c0urank, 0, tmp, c1urows-c1urank, 0, grid()->ctxt_all());
            gemm(Trans::C, Trans::N, scalar_t(1.), child(1)->ULV_mpi_.Q_,
                 tmp, scalar_t(0.), w.c[1].x);
            STRUMPACK_HSS_SOLVE_FLOPS
              (gemm_flops(Trans::C, Trans::N, scalar_t(1.),
                          child(1)->ULV_mpi_.Q_, tmp, scalar_t(0.)));
          } else copy(c1urank, n, w.x, c0urank, 0, w.c[1].x, 0, 0, grid()->ctxt_all());
        }
        w.x.clear();
        w.c[0].y.clear();
        w.c[1].y.clear();
        child(0)->solve_bwd(x, w.c[0], false);
        child(1)->solve_bwd(x, w.c[1], false);
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_MPI_SOLVE_HPP
