#ifndef HSS_MATRIX_SCHUR_HPP
#define HSS_MATRIX_SCHUR_HPP

namespace strumpack {
  namespace HSS {

    /**
     * The Schur complement is F22 - Theta * Vhat * Phi^C.
     * This routine returns Theta and Phi, Vhat is stored as f._Vt0.
     *   Theta = U1big * B10
     *   Phi = (D0^{-1} * U0 * B01 * V1big^C)^C
     */
    template<typename scalar_t> void HSSMatrix<scalar_t>::Schur_update
    (const HSSFactors<scalar_t>& f, DenseM_t& Theta, DenseM_t& DUB01, DenseM_t& Phi) const {
      if (this->leaf()) return;
      auto depth = this->_openmp_task_depth;
      DenseM_t _theta = _B10;
      DUB01 = f._D.solve(child(0)->_U.apply(_B01, depth), f._piv, depth);
      auto _phi = DUB01.transpose();
      auto ch1 = child(1);
      Theta = DenseM_t(ch1->rows(), _theta.cols());
      Phi = DenseM_t(ch1->cols(), _phi.cols());
      std::pair<std::size_t,std::size_t> offset;
      ch1->apply_UV_big(Theta, _theta, Phi, _phi, offset, depth);
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
    template<typename scalar_t> void HSSMatrix<scalar_t>::Schur_product_direct
    (const HSSFactors<scalar_t>& f, const DenseM_t& Theta, const DenseM_t& DUB01, const DenseM_t& Phi,
     const DenseM_t& ThetaVhatC_or_VhatCPhiC, const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc) const {
      auto depth = this->_openmp_task_depth;
      auto ch0 = child(0);
      auto ch1 = child(1);
      WorkApply<scalar_t> wr, wc;
      ch1->apply_fwd(R, wr, false, depth);
      ch1->applyT_fwd(R, wc, false, depth);

      if (Theta.cols() < Phi.cols()) {
	DenseM_t VtDUB01(f.Vhat().cols(), DUB01.cols());
	gemm(Trans::C, Trans::N, scalar_t(1.), f.Vhat(), DUB01, scalar_t(0.), VtDUB01, depth);
	DenseM_t tmpr(ch0->V_rank(), R.cols());
	gemm(Trans::N, Trans::N, scalar_t(1.), VtDUB01, wr.tmp1, scalar_t(0.), tmpr, depth);

	DenseM_t tmpc(_B10.cols(), R.cols());
	gemm(Trans::C, Trans::N, scalar_t(1.), _B10, wc.tmp1, scalar_t(0.), tmpc, depth);

	ch1->apply_bwd(R, scalar_t(0.), Sr, wr, true, depth);
	ch1->applyT_bwd(R, scalar_t(0.), Sc, wc, true, depth);

	gemm(Trans::N, Trans::N, scalar_t(-1.), Theta, tmpr, scalar_t(1.), Sr, depth);
	gemm(Trans::C, Trans::N, scalar_t(-1.), ThetaVhatC_or_VhatCPhiC, tmpc, scalar_t(1.), Sc, depth);
      } else {
	DenseM_t tmpr(DUB01.rows(), R.cols());
	gemm(Trans::N, Trans::N, scalar_t(1.), DUB01, wr.tmp1, scalar_t(0.), tmpr, depth);

	DenseM_t VB10t(f.Vhat().rows(), _B10.rows());
	gemm(Trans::N, Trans::C, scalar_t(1.), f.Vhat(), _B10, scalar_t(0.), VB10t, depth);
	DenseM_t tmpc(f.Vhat().rows(), R.cols());
	gemm(Trans::N, Trans::N, scalar_t(1.), VB10t, wc.tmp1, scalar_t(0.), tmpc, depth);

	ch1->apply_bwd(R, scalar_t(0.), Sr, wr, true, depth);
	ch1->applyT_bwd(R, scalar_t(0.), Sc, wc, true, depth);

	gemm(Trans::N, Trans::N, scalar_t(-1.), ThetaVhatC_or_VhatCPhiC, tmpr, scalar_t(1.), Sr, depth);
	gemm(Trans::N, Trans::N, scalar_t(-1.), Phi, tmpc, scalar_t(1.), Sc, depth);
      }
    }

    /**
     * Apply Schur complement the indirect way:
     *   Sr = Sr1 - U1big B10 (V0big^* R0 + (Vhat^* D00^{-1} U0) B01 V1big^* R1)
     *      = Sr1 - U1big (B10 V0big^* R0 + (B10 Vhat^* DUB01) (V1big^* R1))

     *   Sc = Sc1 - V1big B01^* (U0big^* R0 + (Vhat^* D00^{-1} U0)^* B10^* U1big^* R1)
     *      = Sc1 - V1big (B01^* (U0big^* R0) + B01^* (Vhat^* D00^{-1} U0)^* B10^* (U1big^* R1))
     *      = Sc1 - V1big (B01^* (U0big^* R0) + (B10 Vhat^* DU0B01)^* (U1big^* R1))
     */
    template<typename scalar_t> void HSSMatrix<scalar_t>::Schur_product_indirect
    (const HSSFactors<scalar_t>& f, const DenseM_t& DUB01, const DenseM_t& R0,
     const DenseM_t& R1, const DenseM_t& Sr1, const DenseM_t& Sc1,
     DenseM_t& Sr, DenseM_t& Sc) const {
      if (this->leaf()) return;
      auto depth = this->_openmp_task_depth;
      auto ch0 = child(0);
      auto ch1 = child(1);

      auto c = R1.cols();
      assert(R0.cols() == R1.cols());
      assert(Sr1.cols() == Sc1.cols());

      DenseM_t V0tR0(ch0->V_rank(), c);
      DenseM_t U0tR0(ch0->U_rank(), c);
      std::pair<std::size_t,std::size_t> off0;
      ch0->apply_UtVt_big(R0, U0tR0, V0tR0, off0, depth);

      DenseM_t V1tR1(ch1->V_rank(), c);
      DenseM_t U1tR1(ch1->U_rank(), c);
      std::pair<std::size_t,std::size_t> off1;
      ch1->apply_UtVt_big(R1, U1tR1, V1tR1, off1, depth);

      // TODO store this for the next application??!! NO will only be one application!!
      DenseM_t VtDUB01(f.Vhat().cols(), DUB01.cols());
      gemm(Trans::C, Trans::N, scalar_t(1.), f.Vhat(), DUB01, scalar_t(0.), VtDUB01, depth);
      DenseM_t B10VtDUB01(_B10.rows(), VtDUB01.cols());
      gemm(Trans::N, Trans::N, scalar_t(1.), _B10, VtDUB01, scalar_t(0.), B10VtDUB01, depth);
      VtDUB01.clear();

      DenseM_t B10V0tR0(_B10.rows(), c);
      DenseM_t B01tU0tR0(_B01.cols(), c);
      gemm(Trans::N, Trans::N, scalar_t(-1.), _B10, V0tR0, scalar_t(0.), B10V0tR0, depth);
      gemm(Trans::C, Trans::N, scalar_t(-1.), _B01, U0tR0, scalar_t(0.), B01tU0tR0, depth);
      V0tR0.clear();  U0tR0.clear();

      gemm(Trans::N, Trans::N, scalar_t(1.), B10VtDUB01, V1tR1, scalar_t(1.), B10V0tR0, depth);
      gemm(Trans::C, Trans::N, scalar_t(1.), B10VtDUB01, U1tR1, scalar_t(1.), B01tU0tR0, depth);
      B10VtDUB01.clear(); V1tR1.clear();
      B10VtDUB01.clear(); U1tR1.clear();

      Sr = DenseM_t(R1.rows(), R1.cols());
      Sc = DenseM_t(R1.rows(), R1.cols());
      std::pair<std::size_t,std::size_t> off;
      ch1->apply_UV_big(Sr, B10V0tR0, Sc, B01tU0tR0, off, depth);

      Sr.add(Sr1);
      Sc.add(Sc1);
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::apply_UtVt_big
    (const DenseM_t& A, DenseM_t& UtA, DenseM_t& VtA,
     const std::pair<std::size_t, std::size_t>& offset, int depth) const {
      bool tasked = depth < params::task_recursion_cutoff_level;
      bool isfinal = depth >= params::task_recursion_cutoff_level-1;
      if (this->leaf()) {
	auto Al = ConstDenseMatrixWrapperPtr(this->rows(), A.cols(), A, offset.first, 0);
	UtA = _U.applyC(*Al, depth);
	VtA = _V.applyC(*Al, depth);
      } else {
	DenseM_t UtA0, UtA1;
	DenseM_t VtA0, VtA1;
#pragma omp task default(shared) if(tasked) final(isfinal) mergeable
	this->_ch[0]->apply_UtVt_big(A, UtA0, VtA0, offset, depth+1);
#pragma omp task default(shared) if(tasked) final(isfinal) mergeable
	this->_ch[1]->apply_UtVt_big(A, UtA1, VtA1, offset+this->_ch[0]->dims(), depth+1);
#pragma omp taskwait
	UtA = _U.applyC(vconcat(UtA0, UtA1), depth);
	VtA = _V.applyC(vconcat(VtA0, VtA1), depth);
      }
    }


    // TODO symplify this??
    template<typename scalar_t> void HSSMatrix<scalar_t>::apply_UV_big
    (DenseM_t& Theta, DenseM_t& Uop, DenseM_t& Phi, DenseM_t& Vop,
     const std::pair<std::size_t, std::size_t>& offset, int depth) const {
      bool tasked = depth < params::task_recursion_cutoff_level;
      bool isfinal = depth >= params::task_recursion_cutoff_level-1;
      if (this->leaf()) {
	DenseMW_t ltheta(_U.rows(), Uop.cols(), Theta, offset.first, 0);
	DenseMW_t lphi(_V.rows(), Vop.cols(), Phi, offset.second, 0);
#pragma omp task default(shared) if(tasked) final(isfinal) mergeable
	if (_U.cols() && Uop.cols()) _U.apply(Uop, ltheta, depth);
	else ltheta.zero();
#pragma omp task default(shared) if(tasked) final(isfinal) mergeable
	if (_V.cols() && Vop.cols()) _V.apply(Vop, lphi, depth);
	else lphi.zero();
#pragma omp taskwait
      } else {
	DenseM_t Uop0, Uop1, Vop0, Vop1;
#pragma omp task default(shared) if(tasked) final(isfinal) mergeable
	if (_U.cols() && Uop.cols()) {
	  // if Uop.cols()==0, then don't do anything, leafs will set Theta/Phi to 0

	  // TODO what if U_rank is 0??? the dimension of Uop0/1 will be wrong??
	  auto tmp = _U.apply(Uop, depth);
	  Uop0 = DenseM_t(this->_ch[0]->U_rank(), Uop.cols());
	  Uop1 = DenseM_t(this->_ch[1]->U_rank(), Uop.cols());
	  Uop0.copy(tmp, 0, 0);
	  Uop1.copy(tmp, this->_ch[0]->U_rank(), 0);
	  Uop.clear();
	}
#pragma omp task default(shared) if(tasked) final(isfinal) mergeable
	if (_V.cols() && Vop.cols()) {
	  auto tmp = _V.apply(Vop, depth);
	  Vop0 = DenseM_t(this->_ch[0]->V_rank(), Vop.cols());
	  Vop1 = DenseM_t(this->_ch[1]->V_rank(), Vop.cols());
	  Vop0.copy(tmp, 0, 0);
	  Vop1.copy(tmp, this->_ch[0]->V_rank(), 0);
	  Vop.clear();
	}
#pragma omp taskwait
#pragma omp task default(shared) if(tasked) final(isfinal) mergeable
	this->_ch[0]->apply_UV_big(Theta, Uop0, Phi, Vop0, offset, depth+1);
#pragma omp task default(shared) if(tasked) final(isfinal) mergeable
	this->_ch[1]->apply_UV_big(Theta, Uop1, Phi, Vop1, offset+this->_ch[0]->dims(), depth+1);
#pragma omp taskwait
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_SCHUR_HPP
