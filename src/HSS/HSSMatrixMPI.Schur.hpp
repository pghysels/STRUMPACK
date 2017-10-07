#ifndef HSS_MATRIXMPI_SCHUR_HPP
#define HSS_MATRIXMPI_SCHUR_HPP

namespace strumpack {
  namespace HSS {

    /**
     * The Schur complement is F22 - Theta * Vhat^C * Phi^C.  This
     * routine returns Theta, Phi, Vhat and (D0^{-1} * U0 * B01), all
     * on ctxt().
     *   Theta = U1big * B10
     *   Phi = (D0^{-1} * U0 * B01 * V1big^C)^C
     *       = V1big * (D0^{-1} * U0 * B01)^C
     */
    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::Schur_update
    (const HSSFactorsMPI<scalar_t>& f, DistM_t& Theta, DistM_t& Vhat,
     DistM_t& DUB01, DistM_t& Phi) const {
      if (this->leaf()) return;
      auto ch0 = child(0);
      auto ch1 = child(1);
      DistM_t DU(ctxt(), ch0->U_rows(), ch0->U_rank());
      if (auto ch0mpi =
          dynamic_cast<const HSSMatrixMPI<scalar_t>*>(child(0))) {
        DistM_t chDU;
        if (ch0mpi->active()) chDU = f._D.solve(ch0mpi->_U.dense(), f._piv);
        copy(ch0->U_rows(), ch0->U_rank(), chDU, 0, 0, DU, 0, 0, _ctxt_all);
      } else {
        auto ch0seq = dynamic_cast<const HSSMatrix<scalar_t>*>(child(0));
        DenseM_t chDU;
        if (ch0seq->active())
          chDU = f._D.gather().solve
            (ch0seq->_U.dense(), f._piv, ch0seq->_openmp_task_depth);
        copy(ch0->U_rows(), ch0->U_rank(), chDU, 0/*rank ch0*/,
             DU, 0, 0, _ctxt_all);
      }
      DUB01 = DistM_t(ctxt(), ch0->U_rows(), ch1->V_rank());
      gemm(Trans::N, Trans::N, scalar_t(1.), DU, _B01, scalar_t(0.), DUB01);

      DistM_t _theta(ch1->ctxt(ctxt_loc()), _B10.rows(), _B10.cols());
      copy(_B10.rows(), _B10.cols(), _B10, 0, 0, _theta, 0, 0, ctxt_all());
      auto DUB01t = DUB01.transpose();
      DistM_t _phi(ch1->ctxt(ctxt_loc()), DUB01t.rows(), DUB01t.cols());
      copy(DUB01t.rows(), DUB01t.cols(), DUB01t, 0, 0, _phi, 0, 0, ctxt_all());
      DUB01t.clear();

      DistSubLeaf<scalar_t> Theta_br(_theta.cols(), ch1, ctxt_loc()),
        Phi_br(_phi.cols(), ch1, ctxt_loc());
      DistM_t Theta_ch(ch1->ctxt(ctxt_loc()), ch1->rows(), _theta.cols());
      DistM_t Phi_ch(ch1->ctxt(ctxt_loc()), ch1->cols(), _phi.cols());
      ch1->apply_UV_big(Theta_br, _theta, Phi_br, _phi);
      Theta_br.from_block_row(Theta_ch);
      Phi_br.from_block_row(Phi_ch);
      Theta = DistM_t(ctxt(), Theta_ch.rows(), Theta_ch.cols());
      Phi = DistM_t(ctxt(), Phi_ch.rows(), Phi_ch.cols());
      copy(Theta.rows(), Theta.cols(), Theta_ch, 0, 0,
           Theta, 0, 0, _ctxt_all);
      copy(Phi.rows(), Phi.cols(), Phi_ch, 0, 0, Phi, 0, 0, _ctxt_all);

      Vhat = DistM_t(ctxt(), Phi.cols(), Theta.cols());
      copy(Vhat.rows(), Vhat.cols(), f.Vhat(), 0, 0, Vhat, 0, 0, ctxt_all());
    }

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
      auto ch1 = child(1);
      auto n = R.cols();
      DistM_t RS_ch1(ch1->ctxt(ctxt_loc()), ch1->rows(), R.cols());
      copy(ch1->rows(), R.cols(), R, 0, 0, RS_ch1, 0, 0, _ctxt_all);
      DistSubLeaf<scalar_t> R_br(n, ch1, ctxt_loc(), RS_ch1),
        Sr_br(n, ch1, ctxt_loc()), Sc_br(n, ch1, ctxt_loc());
      WorkApplyMPI<scalar_t> wr, wc;
      ch1->apply_fwd(R_br, wr, false);
      ch1->applyT_fwd(R_br, wc, false);
      DistM_t wrtmp(ctxt(), ch1->V_rank(), n);
      DistM_t wctmp(ctxt(), ch1->U_rank(), n);
      copy(ch1->V_rank(), n, wr.tmp1, 0, 0, wrtmp, 0, 0, _ctxt_all);
      copy(ch1->U_rank(), n, wc.tmp1, 0, 0, wctmp, 0, 0, _ctxt_all);
      if (Theta.cols() < Phi.cols()) {
        DistM_t VtDUB01(ctxt(), Vhat.cols(), DUB01.cols());
        gemm(Trans::C, Trans::N, scalar_t(1.), Vhat, DUB01,
             scalar_t(0.), VtDUB01);
        DistM_t tmpr(ctxt(), child(0)->V_rank(), n);
        gemm(Trans::N, Trans::N, scalar_t(1.), VtDUB01, wrtmp,
             scalar_t(0.), tmpr);
        DistM_t tmpc(ctxt(), _B10.cols(), n);
        gemm(Trans::C, Trans::N, scalar_t(1.), _B10, wctmp,
             scalar_t(0.), tmpc);
        ch1->apply_bwd(R_br, scalar_t(0.), Sr_br, wr, true);
        ch1->applyT_bwd(R_br, scalar_t(0.), Sc_br, wc, true);

        Sr_br.from_block_row(RS_ch1);
        DistM_t Sloc(ctxt(), Sr.rows(), Sr.cols());
        copy(ch1->rows(), Sr.cols(), RS_ch1, 0, 0, Sloc, 0, 0, _ctxt_all);
        gemm(Trans::N, Trans::N, scalar_t(-1.), Theta, tmpr,
             scalar_t(1.), Sloc);
        copy(ch1->rows(), Sr.cols(), Sloc, 0, 0, Sr, 0, 0, _ctxt_all);

        Sc_br.from_block_row(RS_ch1);
        copy(ch1->rows(), Sc.cols(), RS_ch1, 0, 0, Sloc, 0, 0, _ctxt_all);
        gemm(Trans::C, Trans::N, scalar_t(-1.), VhatCPhiC, tmpc,
             scalar_t(1.), Sloc);
        copy(ch1->rows(), Sc.cols(), Sloc, 0, 0, Sc, 0, 0, _ctxt_all);
      } else {
        DistM_t tmpr(ctxt(), DUB01.rows(), n);
        gemm(Trans::N, Trans::N, scalar_t(1.), DUB01, wrtmp,
             scalar_t(0.), tmpr);
        DistM_t VB10t(ctxt(), Vhat.rows(), _B10.rows());
        gemm(Trans::N, Trans::C, scalar_t(1.), Vhat, _B10,
             scalar_t(0.), VB10t);
        DistM_t tmpc(ctxt(), Vhat.rows(), n);
        gemm(Trans::N, Trans::N, scalar_t(1.), VB10t, wctmp,
             scalar_t(0.), tmpc);
        ch1->apply_bwd(R_br, scalar_t(0.), Sr_br, wr, true);
        ch1->applyT_bwd(R_br, scalar_t(0.), Sc_br, wc, true);

        Sr_br.from_block_row(RS_ch1);
        DistM_t Sloc(ctxt(), Sr.rows(), Sr.cols());
        copy(ch1->rows(), Sr.cols(), RS_ch1, 0, 0, Sloc, 0, 0, _ctxt_all);
        gemm(Trans::N, Trans::N, scalar_t(-1.), ThetaVhatC, tmpr,
             scalar_t(1.), Sloc);
        copy(ch1->rows(), Sr.cols(), Sloc, 0, 0, Sr, 0, 0, _ctxt_all);

        Sc_br.from_block_row(RS_ch1);
        copy(ch1->rows(), Sc.cols(), RS_ch1, 0, 0, Sloc, 0, 0, _ctxt_all);
        gemm(Trans::N, Trans::N, scalar_t(-1.), Phi, tmpc,
             scalar_t(1.), Sloc);
        copy(ch1->rows(), Sc.cols(), Sloc, 0, 0, Sc, 0, 0, _ctxt_all);
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::apply_UV_big
    (DistSubLeaf<scalar_t>& Theta, DistM_t& Uop, DistSubLeaf<scalar_t>& Phi,
     DistM_t& Vop) const {
      if (this->leaf()) {
        if (this->U_rank() && Uop.cols()) _U.apply(Uop, Theta.leaf);
        else Theta.leaf.zero();
        if (this->V_rank() && Vop.cols()) _V.apply(Vop, Phi.leaf);
        else Phi.leaf.zero();
      } else {
        DistM_t Uop0, Uop1, Vop0, Vop1;
        if (this->U_rank() && Uop.cols()) {
          auto tmp = _U.apply(Uop);
          Uop0 = DistM_t(this->_ch[0]->ctxt(Theta.ctxt_loc()),
                         this->_ch[0]->U_rank(), Uop.cols());
          Uop1 = DistM_t(this->_ch[1]->ctxt(Theta.ctxt_loc()),
                         this->_ch[1]->U_rank(), Uop.cols());
          copy(this->_ch[0]->U_rank(), Uop.cols(), tmp, 0, 0,
               Uop0, 0, 0, _ctxt_all);
          copy(this->_ch[1]->U_rank(), Uop.cols(), tmp,
               this->_ch[0]->U_rank(), 0, Uop1, 0, 0, _ctxt_all);
          Uop.clear();
        }
        if (this->V_rank() && Vop.cols()) {
          auto tmp = _V.apply(Vop);
          Vop0 = DistM_t(this->_ch[0]->ctxt(Phi.ctxt_loc()),
                         this->_ch[0]->V_rank(), Vop.cols());
          Vop1 = DistM_t(this->_ch[1]->ctxt(Phi.ctxt_loc()),
                         this->_ch[1]->V_rank(), Vop.cols());
          copy(this->_ch[0]->V_rank(), Vop.cols(), tmp, 0, 0,
               Vop0, 0, 0, _ctxt_all);
          copy(this->_ch[1]->V_rank(), Vop.cols(), tmp,
               this->_ch[0]->V_rank(), 0, Vop1, 0, 0, _ctxt_all);
          Vop.clear();
        }
        this->_ch[0]->apply_UV_big(Theta, Uop0, Phi, Vop0);
        this->_ch[1]->apply_UV_big(Theta, Uop1, Phi, Vop1);
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIXMPI_SCHUR_HPP
