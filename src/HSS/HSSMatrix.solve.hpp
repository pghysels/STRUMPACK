#ifndef HSS_MATRIX_SOLVE_HPP
#define HSS_MATRIX_SOLVE_HPP

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> void HSSMatrix<scalar_t>::solve
    (const HSSFactors<scalar_t>& ULV, DenseMatrix<scalar_t>& b) const {
      assert(b.rows() == this->rows());
      // TODO assert that the ULV factorization has been performed and
      // is a valid one
      // assert(ULV._D.rows() == _U.rows());
      WorkSolve<scalar_t> w;
      solve_fwd(ULV, b, w, false, true, this->_openmp_task_depth);
      solve_bwd(ULV, b, w, true, this->_openmp_task_depth);
    }

    // TODO do not pass work, just return the reduced_rhs, and w.x at the root
    template<typename scalar_t> void HSSMatrix<scalar_t>::forward_solve
    (const HSSFactors<scalar_t>& ULV, WorkSolve<scalar_t>& w,
     const DenseMatrix<scalar_t>& b, bool partial) const {
      solve_fwd(ULV, b, w, partial, true, this->_openmp_task_depth);
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::backward_solve
    (const HSSFactors<scalar_t>& ULV, WorkSolve<scalar_t>& w,
     DenseMatrix<scalar_t>& b) const {
      solve_bwd(ULV, b, w, true, this->_openmp_task_depth);
    }

    // have this routine return ft1, or x at the root!!!
    // then ft1 and x do not need to be stored in WorkSolve!!
    template<typename scalar_t> void HSSMatrix<scalar_t>::solve_fwd
    (const HSSFactors<scalar_t>& ULV, const DenseMatrix<scalar_t>& b,
     WorkSolve<scalar_t>& w, bool partial, bool isroot, int depth) const {
      DenseM_t f;
      if (this->leaf())
        f = DenseM_t(this->rows(), b.cols(), b, w.offset.second, 0);
      else {
        w.c.resize(2);
        w.c[0].offset = w.offset;
        w.c[1].offset = w.offset + this->_ch[0]->dims();
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        this->_ch[0]->solve_fwd
          (ULV._ch[0], b, w.c[0], partial, false, depth+1);
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        this->_ch[1]->solve_fwd
          (ULV._ch[1], b, w.c[1], partial, false, depth+1);
#pragma omp taskwait
        DenseM_t& f0 = w.c[0].ft1;
        DenseM_t& f1 = w.c[1].ft1;
        gemm
          (Trans::N, Trans::N, scalar_t(-1.),
           _B01, w.c[1].z, scalar_t(1.), f0, depth);
        gemm
          (Trans::N, Trans::N, scalar_t(-1.),
           _B10, w.c[0].z, scalar_t(1.), f1, depth);
        if (this->_ch[0]->U_rows() > this->_ch[0]->U_rank()) {
          auto Q00 = ConstDenseMatrixWrapperPtr
            (this->_ch[0]->U_rows()-this->_ch[0]->U_rank(),
             this->_ch[0]->U_rows(), ULV._ch[0]._Q, 0, 0);
          DenseM_t tmp0(Q00->cols(), b.cols());
          gemm
            (Trans::C, Trans::N, scalar_t(1.),
             *Q00, w.c[0].y, scalar_t(0.), tmp0, depth);
          gemm
            (Trans::N, Trans::N, scalar_t(-1.),
             ULV._ch[0]._W1, tmp0, scalar_t(1.), f0, depth);
        }
        if (this->_ch[1]->U_rows() > this->_ch[1]->U_rank()) {
          auto Q10 = ConstDenseMatrixWrapperPtr
            (this->_ch[1]->U_rows()-this->_ch[1]->U_rank(),
             this->_ch[1]->U_rows(), ULV._ch[1]._Q, 0, 0);
          DenseM_t tmp1(Q10->cols(), b.cols());
          gemm
            (Trans::C, Trans::N, scalar_t(1.),
             *Q10, w.c[1].y, scalar_t(0.), tmp1, depth);
          gemm
            (Trans::N, Trans::N, scalar_t(-1.),
             ULV._ch[1]._W1, tmp1, scalar_t(1.), f1, depth);
        }
        f = vconcat(f0, f1);
        f0.clear();
        f1.clear();
      }
      if (isroot) {
        w.x = ULV._D.solve(f, ULV._piv, depth);
        if (partial) {
          // compute reduced_rhs = \hat{V}^* y_0 + V^* [z_0; z_1]
          w.reduced_rhs = DenseM_t(this->V_rank(), w.x.cols());
          gemm
            (Trans::C, Trans::N, scalar_t(1.),
             ULV._Vt0, w.x, scalar_t(0.), w.reduced_rhs, depth);
          if (!this->leaf())
            w.reduced_rhs.add(_V.applyC(vconcat(w.c[0].z, w.c[1].z), depth));
        }
      } else {
        f.laswp(_U.P(), true);
        if (this->U_rows() > this->U_rank()) {
          w.ft1 = DenseM_t(this->U_rank(), f.cols(), f, 0, 0);
          w.y = DenseM_t    // put ft0 in w.y
            (this->U_rows()-this->U_rank(), f.cols(), f, this->U_rank(), 0);
          gemm
            (Trans::N, Trans::N, scalar_t(-1.),
             _U.E(), w.ft1, scalar_t(1.), w.y, depth);
          trsm
            (Side::L, UpLo::L, Trans::N, Diag::N,
             scalar_t(1.), ULV._L, w.y, depth);
          if (!this->leaf()) {
            w.z = _V.applyC(vconcat(w.c[0].z, w.c[1].z), depth);
            gemm
              (Trans::C, Trans::N, scalar_t(1.),
               ULV._Vt0, w.y, scalar_t(1.), w.z, depth);
          } else {
            w.z = DenseM_t(this->V_rank(), b.cols());
            gemm
              (Trans::C, Trans::N, scalar_t(1.),
               ULV._Vt0, w.y, scalar_t(0.), w.z, depth);
          }
        } else {
          w.ft1 = DenseM_t(this->U_rank(), f.cols(), f, 0, 0);
          if (!this->leaf())
            w.z = _V.applyC(vconcat(w.c[0].z, w.c[1].z), depth);
          else {
            w.z = DenseM_t(this->V_rank(), b.cols());
            w.z.zero(); // TODO can this be avoided?
          }
        }
      }
      if (!this->leaf()) {
        w.c[0].z.clear();
        w.c[1].z.clear();
      }
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::solve_bwd
    (const HSSFactors<scalar_t>& ULV, DenseMatrix<scalar_t>& x,
     WorkSolve<scalar_t>& w, bool isroot, int depth) const {
      if (this->leaf()) copy(w.x, x.ptr(w.offset.second, 0), x.ld());
      else {
        w.c[0].x = DenseM_t(this->_ch[0]->U_rows(), x.cols());
        w.c[1].x = DenseM_t(this->_ch[1]->U_rows(), x.cols());
        DenseMW_t x0(this->_ch[0]->U_rank(), x.cols(), w.x, 0, 0);
        DenseMW_t x1(this->_ch[1]->U_rank(), x.cols(), w.x, x0.rows(), 0);
        // TODO instead of concat, use 2 separate gemms!!
        if (this->_ch[0]->U_rows() > this->_ch[0]->U_rank())
          gemm
            (Trans::C, Trans::N, scalar_t(1.),
             ULV._ch[0]._Q, vconcat(w.c[0].y, x0),
             scalar_t(0.), w.c[0].x, depth);
        else w.c[0].x.copy(x0);
        if (this->_ch[1]->U_rows() > this->_ch[1]->U_rank())
          gemm
            (Trans::C, Trans::N, scalar_t(1.),
             ULV._ch[1]._Q, vconcat(w.c[1].y, x1),
             scalar_t(0.), w.c[1].x, depth);
        else w.c[1].x.copy(x1);
        w.x.clear();
        w.c[0].y.clear();
        w.c[1].y.clear();
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        this->_ch[0]->solve_bwd(ULV._ch[0], x, w.c[0], false, depth+1);
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        this->_ch[1]->solve_bwd(ULV._ch[1], x, w.c[1], false, depth+1);
#pragma omp taskwait
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_SOLVE_HPP
