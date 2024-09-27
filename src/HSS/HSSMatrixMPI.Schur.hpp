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
 */
#ifndef HSS_MATRIXMPI_SCHUR_HPP
#define HSS_MATRIXMPI_SCHUR_HPP

namespace strumpack {
  namespace HSS {

    /**
     * The Schur complement is F22 - Theta * Vhat^C * Phi^C.  This
     * routine returns Theta, Phi, Vhat and (D0^{-1} * U0 * B01), all
     * on grid().
     *   Theta = U1big * B10
     *   Phi = (D0^{-1} * U0 * B01 * V1big^C)^C
     *       = V1big * (D0^{-1} * U0 * B01)^C
     */

    /**
     * Apply Schur complement the direct way:
     *   Sr = H.child(1) R - U1big B10 Vhat^* D00^{-1} U0 B01 V1big^* R
     *      = H.child(1) R - Theta Vhat^* D11^{-1} U0 B01 V1big^* R
     *      = H.child(1) R - Theta Vhat^* DUB01 (V1big^* R)
     *   Sc = (H.child(1))^* R - V1big B01^* (Vhat^* D00^{-1} U0)^* B10^* U1big^* R)
     *      = (H.child(1))^* R - Phi Vhat B10^* (U1big^* R)
     *
     * Here, application of U1big^* R is also the forward step of
     * H.child(1)^* R, so it can be reused. Likewise for H.child(1) R
     * and V1big^* R.
     */
    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::Schur_product_direct
    (const DistM_t& Theta, const DistM_t& Vhat, const DistM_t& DUB01,
     const DistM_t& Phi, const DistM_t& ThetaVhatC, const DistM_t& VhatCPhiC,
     const DistM_t& R, DistM_t& Sr, DistM_t& Sc) const {
      long long int flops = 0;
      auto ch1 = child(1);
      auto n = R.cols();
      DistM_t RS_ch1(ch1->grid(grid_local()), ch1->rows(), R.cols());
      copy(ch1->rows(), R.cols(), R, 0, 0, RS_ch1, 0, 0, grid()->ctxt_all());
      DistSubLeaf<scalar_t> R_br(n, ch1, grid_local(), RS_ch1),
        Sr_br(n, ch1, grid_local()), Sc_br(n, ch1, grid_local());
      WorkApplyMPI<scalar_t> wr, wc;
      ch1->apply_fwd(R_br, wr, false, flops);
      ch1->applyT_fwd(R_br, wc, false, flops);
      DistM_t wrtmp(grid(), ch1->V_rank(), n);
      DistM_t wctmp(grid(), ch1->U_rank(), n);
      copy(ch1->V_rank(), n, wr.tmp1, 0, 0, wrtmp, 0, 0, grid()->ctxt_all());
      copy(ch1->U_rank(), n, wc.tmp1, 0, 0, wctmp, 0, 0, grid()->ctxt_all());
      if (Theta.cols() < Phi.cols()) {
        DistM_t VtDUB01(grid(), Vhat.cols(), DUB01.cols());
        gemm(Trans::C, Trans::N, scalar_t(1.), Vhat, DUB01,
             scalar_t(0.), VtDUB01);
        DistM_t tmpr(grid(), child(0)->V_rank(), n);
        gemm(Trans::N, Trans::N, scalar_t(1.), VtDUB01, wrtmp,
             scalar_t(0.), tmpr);
        DistM_t tmpc(grid(), B10_.cols(), n);
        gemm(Trans::C, Trans::N, scalar_t(1.), B10_, wctmp,
             scalar_t(0.), tmpc);
        STRUMPACK_CB_SAMPLE_FLOPS
          (gemm_flops(Trans::C, Trans::N, scalar_t(1.), Vhat, DUB01, scalar_t(0.)) +
           gemm_flops(Trans::N, Trans::N, scalar_t(1.), VtDUB01, wrtmp, scalar_t(0.)) +
           gemm_flops(Trans::C, Trans::N, scalar_t(1.), B10_, wctmp, scalar_t(0.)));

        ch1->apply_bwd(R_br, scalar_t(0.), Sr_br, wr, true, flops);
        ch1->applyT_bwd(R_br, scalar_t(0.), Sc_br, wc, true, flops);

        Sr_br.from_block_row(RS_ch1);
        DistM_t Sloc(grid(), Sr.rows(), Sr.cols());
        copy(ch1->rows(), Sr.cols(), RS_ch1, 0, 0, Sloc, 0, 0, grid()->ctxt_all());
        gemm(Trans::N, Trans::N, scalar_t(-1.), Theta, tmpr,
             scalar_t(1.), Sloc);
        copy(ch1->rows(), Sr.cols(), Sloc, 0, 0, Sr, 0, 0, grid()->ctxt_all());

        Sc_br.from_block_row(RS_ch1);
        copy(ch1->rows(), Sc.cols(), RS_ch1, 0, 0, Sloc, 0, 0, grid()->ctxt_all());
        gemm(Trans::C, Trans::N, scalar_t(-1.), VhatCPhiC, tmpc,
             scalar_t(1.), Sloc);
        copy(ch1->rows(), Sc.cols(), Sloc, 0, 0, Sc, 0, 0, grid()->ctxt_all());
        STRUMPACK_CB_SAMPLE_FLOPS
          (gemm_flops(Trans::N, Trans::N, scalar_t(-1.), Theta, tmpr, scalar_t(1.)) +
           gemm_flops(Trans::C, Trans::N, scalar_t(-1.), VhatCPhiC, tmpc, scalar_t(1.)));
      } else {
        DistM_t tmpr(grid(), DUB01.rows(), n);
        gemm(Trans::N, Trans::N, scalar_t(1.), DUB01, wrtmp,
             scalar_t(0.), tmpr);
        DistM_t VB10t(grid(), Vhat.rows(), B10_.rows());
        gemm(Trans::N, Trans::C, scalar_t(1.), Vhat, B10_,
             scalar_t(0.), VB10t);
        DistM_t tmpc(grid(), Vhat.rows(), n);
        gemm(Trans::N, Trans::N, scalar_t(1.), VB10t, wctmp,
             scalar_t(0.), tmpc);
        STRUMPACK_CB_SAMPLE_FLOPS
          (gemm_flops(Trans::N, Trans::N, scalar_t(1.), DUB01, wrtmp, scalar_t(0.)) +
           gemm_flops(Trans::N, Trans::C, scalar_t(1.), Vhat, B10_, scalar_t(0.)) +
           gemm_flops(Trans::N, Trans::N, scalar_t(1.), VB10t, wctmp, scalar_t(0.)));

        ch1->apply_bwd(R_br, scalar_t(0.), Sr_br, wr, true, flops);
        ch1->applyT_bwd(R_br, scalar_t(0.), Sc_br, wc, true, flops);

        Sr_br.from_block_row(RS_ch1);
        DistM_t Sloc(grid(), Sr.rows(), Sr.cols());
        copy(ch1->rows(), Sr.cols(), RS_ch1, 0, 0, Sloc, 0, 0, grid()->ctxt_all());
        gemm(Trans::N, Trans::N, scalar_t(-1.), ThetaVhatC, tmpr,
             scalar_t(1.), Sloc);
        copy(ch1->rows(), Sr.cols(), Sloc, 0, 0, Sr, 0, 0, grid()->ctxt_all());

        Sc_br.from_block_row(RS_ch1);
        copy(ch1->rows(), Sc.cols(), RS_ch1, 0, 0, Sloc, 0, 0, grid()->ctxt_all());
        gemm(Trans::N, Trans::N, scalar_t(-1.), Phi, tmpc,
             scalar_t(1.), Sloc);
        copy(ch1->rows(), Sc.cols(), Sloc, 0, 0, Sc, 0, 0, grid()->ctxt_all());
        STRUMPACK_CB_SAMPLE_FLOPS
          (gemm_flops(Trans::N, Trans::N, scalar_t(-1.), ThetaVhatC, tmpr, scalar_t(1.)) +
           gemm_flops(Trans::N, Trans::N, scalar_t(-1.), Phi, tmpc, scalar_t(1.)));
      }
      STRUMPACK_CB_SAMPLE_FLOPS(flops);
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::apply_UV_big
    (DistSubLeaf<scalar_t>& Theta, DistM_t& Uop, DistSubLeaf<scalar_t>& Phi,
     DistM_t& Vop, long long int& flops) const {
      if (this->leaf()) {
        if (this->U_rank() && Uop.cols()) {
          U_.apply(Uop, Theta.leaf);
          flops += U_.apply_flops(Uop.cols());
        } else Theta.leaf.zero();
        if (this->V_rank() && Vop.cols()) {
          V_.apply(Vop, Phi.leaf);
          flops += V_.apply_flops(Vop.cols());
        } else Phi.leaf.zero();
      } else {
        DistM_t Uop0, Uop1, Vop0, Vop1;
        if (this->U_rank() && Uop.cols()) {
          auto tmp = U_.apply(Uop);
          flops += U_.apply_flops(Uop.cols());
          Uop0 = DistM_t(child(0)->grid(Theta.grid_local()),
                         child(0)->U_rank(), Uop.cols());
          Uop1 = DistM_t(child(1)->grid(Theta.grid_local()),
                         child(1)->U_rank(), Uop.cols());
          copy(child(0)->U_rank(), Uop.cols(), tmp, 0, 0, Uop0, 0, 0, grid()->ctxt_all());
          copy(child(1)->U_rank(), Uop.cols(), tmp,
               child(0)->U_rank(), 0, Uop1, 0, 0, grid()->ctxt_all());
          Uop.clear();
        }
        if (this->V_rank() && Vop.cols()) {
          auto tmp = V_.apply(Vop);
          flops += V_.apply_flops(Vop.cols());
          Vop0 = DistM_t(child(0)->grid(Phi.grid_local()),
                         child(0)->V_rank(), Vop.cols());
          Vop1 = DistM_t(child(1)->grid(Phi.grid_local()),
                         child(1)->V_rank(), Vop.cols());
          copy(child(0)->V_rank(), Vop.cols(), tmp, 0, 0, Vop0, 0, 0, grid()->ctxt_all());
          copy(child(1)->V_rank(), Vop.cols(), tmp,
               child(0)->V_rank(), 0, Vop1, 0, 0, grid()->ctxt_all());
          Vop.clear();
        }
        child(0)->apply_UV_big(Theta, Uop0, Phi, Vop0, flops);
        child(1)->apply_UV_big(Theta, Uop1, Phi, Vop1, flops);
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIXMPI_SCHUR_HPP
