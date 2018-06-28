#ifndef HSS_MATRIX_INERTIA_HPP
#define HSS_MATRIX_INERTIA_HPP

namespace strumpack {
  namespace HSS {

    // assumes both A and HSS(A) compression are real symmetric
    template<typename scalar_t> HSSFactors<scalar_t> HSSMatrix<scalar_t>::inertia() const {
      HSSInertia<scalar_t> in; // need to be implemented : in = [np, nn, nz] cumulative inertia of all nodes in tree below current node being processed
      WorkInertia<scalar_t> w; // need to be implemented : w.S matrix from leftover factors of LDL^T on current node
      inertia_recursive(in, w, true);
      return in;
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::inertia_recursive(HSSInertia<scalar_t>& in, WorkInertia<scalar_t>& w, bool isroot, int depth) const {

      DenseM_t Dt;

      if (!this->leaf()){
        this->_ch[0]->inertia_recursive(in, w.c[0], false);
        this->_ch[1]->inertia_recursive(in, w.c[1], false);
        // Dt = [S{ch1} B12{i}; B21{i} S{ch2}];
      } else {
        // Dt = D;
      }


      if (isroot) {
      // [Ltemp, Dh, Ptemp] = ldl(Db(1:rtop, 1:rtop));
      } else {
      // Omega = [ -U{i} eye(size(U{i},1)); eye(size(U{i},2)) zeros(size(U{i},2),size(U{i},1))] * Pr{i}';
      // rbottom = size(U{i},2);
      // rtop = b - rbottom;
      // Db = Omega * Dt * Omega';
      // [Ltemp, Dh, Ptemp] = ldl(Db(1:rtop, 1:rtop));
      // Lh11 = Ptemp' * Ltemp;
      // Lh21 = ( (Lh11*(Dh')) \ Db(rtop+1:b,1:rtop)' )';
      // S{i} = Db(rtop+1:b,rtop+1:b) - Lh21 * Dh * Lh21';
      }

      // in += readInertiaOffBlockDiag(Dh);
    }

  }
}
