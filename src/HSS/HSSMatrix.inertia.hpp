#ifndef HSS_MATRIX_INERTIA_HPP
#define HSS_MATRIX_INERTIA_HPP

namespace strumpack {
  namespace HSS {

    // assumes both A and HSS(A) compression are real symmetric
    template<typename scalar_t> HSSInertia<scalar_t> HSSMatrix<scalar_t>::inertia() const {
      HSSInertia<scalar_t> in;
      WorkInertia<scalar_t> w;
      inertia_recursive(in, w, true, 0);
      return in;
    }


    template<typename scalar_t> void HSSMatrix<scalar_t>::readInertiaOffBlockDiag(HSSInertia<scalar_t>& in, const DenseM_t D, const std::vector<int>& IPIV) const{
      unsigned int np = 0;
      unsigned int nn = 0;
      unsigned int nz = 0;
      unsigned int k  = 0;
      scalar_t a;
      scalar_t b;
      scalar_t c;
      auto nD = D.rows();
      if (nD == 0) {
        return;
      }
      while (k < nD-1) {
        if ((IPIV[k+1] < 0) && (IPIV[k]==IPIV[k+1])){
          // 2x2 block
          a = D(k,k);
          b = D(k+1,k);
          c = D(k+1,k+1);
          auto lam1 = scalar_t(0.5)*( (a+c) + std::sqrt((a-c)*(a-c) + scalar_t(4.)*b*b));
          auto lam2 = scalar_t(0.5)*( (a+c) - std::sqrt((a-c)*(a-c) + scalar_t(4.)*b*b));
          if (std::real(lam1) > 0.) {
            np += 1;
          } else if (std::real(lam1) < 0.) {
            nn += 1;
          } else {
            nz += 1;
          }
          if (std::real(lam2) > 0.) {
            np += 1;
          } else if (std::real(lam2) < 0.) {
            nn += 1;
          } else {
            nz += 1;
          }
          k += 2;

        } else {
          // 1x1 block
          a = D(k,k);
          if (std::real(a) > 0.) {
            np += 1;
          } else if (std::real(a) < 0.) {
            nn += 1;
          } else {
            nz += 1;
          }
          k += 1;
        }
      }
      if (k < nD) { // need one more 1x1 block to finish
        a = D(k,k);
        if (std::real(a) > 0.) {
          np += 1;
        } else if (std::real(a) < 0.) {
          nn += 1;
        } else {
          nz += 1;
        }
        k += 1;
      }
      in.np += np;
      in.nn += nn;
      in.nz += nz;
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::inertia_recursive(HSSInertia<scalar_t>& in, WorkInertia<scalar_t>& w, bool isroot, int depth) const {

      DenseM_t Dt;
      // Form Dt
      if (!this->leaf()){
        w.c.resize(2);
        in.ch.resize(2);
        this->_ch[0]->inertia_recursive(in.ch[0], w.c[0], false, depth+1);
        this->_ch[1]->inertia_recursive(in.ch[1], w.c[1], false, depth+1);
        in.np = in.ch[0].np + in.ch[1].np;
        in.nn = in.ch[0].nn + in.ch[1].nn;
        in.nz = in.ch[0].nz + in.ch[1].nz;
        // Form Dt = [S{ch1} B12{i}; B21{i} S{ch2}]
        auto u_size = this->_ch[0]->U_rank() + this->_ch[1]->U_rank();
        Dt = DenseM_t(u_size, u_size);
        auto c0u = this->_ch[0]->U_rank();
        copy(w.c[0].S, Dt, 0, 0);
        copy(w.c[1].S, Dt, c0u, c0u);
        copy(_B01, Dt, 0, c0u);
        copy(_B10, Dt, c0u, 0);
      } else {
        // Form Dt = D;
        Dt = _D;
        in.np = 0;
        in.nn = 0;
        in.nz = 0;
      }

      if (isroot) {
      // LDL(Dt) for what is remaining
      auto IPIV = Dt.sytrf();
      readInertiaOffBlockDiag(in, Dt, IPIV);

      } else {

      DenseM_t Om;
      auto E_cols = _U.cols();
      auto E_rows = _U.rows() - E_cols;
      // Dt <-- P' * Dt * P
      ////// Dt <-- P' * Dt
      Dt.laswp(_U.P(), true);
      ////// Dt <-- Dt * P = (P' * Dt')';
      Dt = Dt.transpose();
      Dt.laswp(_U.P(), true);
      Dt = Dt.transpose();
      // Dt.lapmt(_U.P, false); // I don't think the P permutation works with lapmt nor with lapmr
      // Dt.lapmr(_U.P, true);

      //      Dt        <--      Omega    *      Dt     *   Omega'         (permutation included in above step)
      // [ D11  D12 ]  ----   [ -E  Irr ] . [ C11 C12 ] . [ -E'  Icc ]
      // [ D21  D22 ]  ----   [ Icc 0cr ]   [ C21 C22 ]   [ Irr  0rc ]
      // D11 =  E*C11*E' - 2*E*C12 + C22   = E*(C11*E' - 2*C12) + C22
      // D12 = -E*C11  + C21
      // D21 = -C11*E' + C12   = D12'
      // D22 =  C11
      DenseMW_t C12(E_cols, E_rows, Dt, 0, E_cols);
      DenseMW_t C21(E_rows, E_cols, Dt, E_cols, 0);
      DenseM_t D22(E_cols, E_cols, Dt, 0, 0); // D22 = C11

      DenseMW_t D11(E_rows, E_rows, Dt, E_cols, E_cols); // D11 <-- C22
      gemm(Trans::N, Trans::C, scalar_t(1.), D22, _U.E(), scalar_t(-1.), C12);
      gemm(Trans::N, Trans::N, scalar_t(1.), _U.E(), C12, scalar_t(1.), D11);
      gemm(Trans::N, Trans::C, scalar_t(-1.), C21, _U.E(), scalar_t(1.), D11);

      DenseMW_t D12(E_rows, E_cols, Dt, E_cols, 0); // D12 <-- C21
      gemm(Trans::N, Trans::N, scalar_t(-1.), _U.E(), D22, scalar_t(1.), D12);

      auto D21 = D12.transpose();

      // LDL(Db(1:rtop, 1:rtop)), then form S = D22 - D21 inv(D11) D12 with inv(D11) using LDL
      auto IPIV = D11.sytrf();
      readInertiaOffBlockDiag(in, D11, IPIV);
      sytrs(UpLo::L, D11, IPIV, D12);
      gemm(Trans::N, Trans::N, scalar_t(-1.), D21, D12, scalar_t(1.), D22);
      w.S = D22;
      }

    }

  }
}
#endif
