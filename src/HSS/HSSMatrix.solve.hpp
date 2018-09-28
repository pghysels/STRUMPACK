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
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      {
        solve_fwd(ULV, b, w, false, true, this->_openmp_task_depth);
        solve_bwd(ULV, b, w, true, this->_openmp_task_depth);
      }
    }

    // TODO do not pass work, just return the reduced_rhs, and w.x at the root
    template<typename scalar_t> void HSSMatrix<scalar_t>::forward_solve
    (const HSSFactors<scalar_t>& ULV, WorkSolve<scalar_t>& w,
     const DenseMatrix<scalar_t>& b, bool partial) const {
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      solve_fwd(ULV, b, w, partial, true, this->_openmp_task_depth);
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::backward_solve
    (const HSSFactors<scalar_t>& ULV, WorkSolve<scalar_t>& w,
     DenseMatrix<scalar_t>& b) const {
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
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
        gemm(Trans::N, Trans::N, scalar_t(-1.),
             _B01, w.c[1].z, scalar_t(1.), f0, depth);
        gemm(Trans::N, Trans::N, scalar_t(-1.),
             _B10, w.c[0].z, scalar_t(1.), f1, depth);
        STRUMPACK_HSS_SOLVE_FLOPS
          (gemm_flops(Trans::N, Trans::N, scalar_t(-1.),
                      _B01, w.c[1].z, scalar_t(1.)) +
           gemm_flops(Trans::N, Trans::N, scalar_t(-1.),
                      _B10, w.c[0].z, scalar_t(1.)));
        if (this->_ch[0]->U_rows() > this->_ch[0]->U_rank()) {
          auto Q00 = ConstDenseMatrixWrapperPtr
            (this->_ch[0]->U_rows()-this->_ch[0]->U_rank(),
             this->_ch[0]->U_rows(), ULV._ch[0]._Q, 0, 0);
          DenseM_t tmp0(Q00->cols(), b.cols());
          gemm(Trans::C, Trans::N, scalar_t(1.),
               *Q00, w.c[0].y, scalar_t(0.), tmp0, depth);
          gemm(Trans::N, Trans::N, scalar_t(-1.),
               ULV._ch[0]._W1, tmp0, scalar_t(1.), f0, depth);
          STRUMPACK_HSS_SOLVE_FLOPS
            (gemm_flops(Trans::C, Trans::N, scalar_t(1.),
                        *Q00, w.c[0].y, scalar_t(0.)) +
             gemm_flops(Trans::N, Trans::N, scalar_t(-1.),
                        ULV._ch[0]._W1, tmp0, scalar_t(1.)));
        }
        if (this->_ch[1]->U_rows() > this->_ch[1]->U_rank()) {
          auto Q10 = ConstDenseMatrixWrapperPtr
            (this->_ch[1]->U_rows()-this->_ch[1]->U_rank(),
             this->_ch[1]->U_rows(), ULV._ch[1]._Q, 0, 0);
          DenseM_t tmp1(Q10->cols(), b.cols());
          gemm(Trans::C, Trans::N, scalar_t(1.),
               *Q10, w.c[1].y, scalar_t(0.), tmp1, depth);
          gemm(Trans::N, Trans::N, scalar_t(-1.),
               ULV._ch[1]._W1, tmp1, scalar_t(1.), f1, depth);
          STRUMPACK_HSS_SOLVE_FLOPS
            (gemm_flops(Trans::C, Trans::N, scalar_t(1.),
                        *Q10, w.c[1].y, scalar_t(0.)) +
             gemm_flops(Trans::N, Trans::N, scalar_t(-1.),
                        ULV._ch[1]._W1, tmp1, scalar_t(1.)));
        }
        f = vconcat(f0, f1);
        f0.clear();
        f1.clear();
      }
      if (isroot) {
        w.x = ULV._D.solve(f, ULV._piv, depth);
        STRUMPACK_HSS_SOLVE_FLOPS(solve_flops(f));
        if (partial) {
          // compute reduced_rhs = \hat{V}^* y_0 + V^* [z_0; z_1]
          w.reduced_rhs = DenseM_t(this->V_rank(), w.x.cols());
          gemm(Trans::C, Trans::N, scalar_t(1.),
               ULV._Vt0, w.x, scalar_t(0.), w.reduced_rhs, depth);
          STRUMPACK_HSS_SOLVE_FLOPS
            (gemm_flops(Trans::C, Trans::N, scalar_t(1.),
                        ULV._Vt0, w.x, scalar_t(0.)));
          if (!this->leaf()) {
            w.reduced_rhs.add
              (_V.applyC(vconcat(w.c[0].z, w.c[1].z), depth), depth);
            STRUMPACK_HSS_SOLVE_FLOPS
              (_V.applyC_flops(w.c[0].z.cols()) +
               w.reduced_rhs.rows() * w.reduced_rhs.cols());
          }
        }
      } else {
        f.laswp(_U.P(), true);
        if (this->U_rows() > this->U_rank()) {
          w.ft1 = DenseM_t(this->U_rank(), f.cols(), f, 0, 0);
          w.y = DenseM_t    // put ft0 in w.y
            (this->U_rows()-this->U_rank(), f.cols(), f, this->U_rank(), 0);
          gemm(Trans::N, Trans::N, scalar_t(-1.),
               _U.E(), w.ft1, scalar_t(1.), w.y, depth);
          trsm(Side::L, UpLo::L, Trans::N, Diag::N,
               scalar_t(1.), ULV._L, w.y, depth);
          STRUMPACK_HSS_SOLVE_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(-1.),
                        _U.E(), w.ft1, scalar_t(1.)) +
             trsm_flops(Side::L, scalar_t(1.), ULV._L, w.y));
          if (!this->leaf()) {
            w.z = _V.applyC(vconcat(w.c[0].z, w.c[1].z), depth);
            gemm(Trans::C, Trans::N, scalar_t(1.),
                 ULV._Vt0, w.y, scalar_t(1.), w.z, depth);
            STRUMPACK_HSS_SOLVE_FLOPS
              (_V.applyC_flops(w.c[0].z.cols()) +
               gemm_flops(Trans::C, Trans::N, scalar_t(1.),
                          ULV._Vt0, w.y, scalar_t(1.)));
          } else {
            w.z = DenseM_t(this->V_rank(), b.cols());
            gemm(Trans::C, Trans::N, scalar_t(1.),
                 ULV._Vt0, w.y, scalar_t(0.), w.z, depth);
            STRUMPACK_HSS_SOLVE_FLOPS
              (gemm_flops(Trans::C, Trans::N, scalar_t(1.),
                          ULV._Vt0, w.y, scalar_t(0.)));
          }
        } else {
          w.ft1 = DenseM_t(this->U_rank(), f.cols(), f, 0, 0);
          if (!this->leaf()) {
            w.z = _V.applyC(vconcat(w.c[0].z, w.c[1].z), depth);
            STRUMPACK_HSS_SOLVE_FLOPS(_V.applyC_flops(w.c[0].z.cols()));
          } else {
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
        if (this->_ch[0]->U_rows() > this->_ch[0]->U_rank()) {
          auto tmp = vconcat(w.c[0].y, x0);
          gemm(Trans::C, Trans::N, scalar_t(1.), ULV._ch[0]._Q,
               tmp, scalar_t(0.), w.c[0].x, depth);
          STRUMPACK_HSS_SOLVE_FLOPS
            (gemm_flops(Trans::C, Trans::N, scalar_t(1.),
                        ULV._ch[0]._Q, tmp, scalar_t(0.)));
        } else w.c[0].x.copy(x0);
        if (this->_ch[1]->U_rows() > this->_ch[1]->U_rank()) {
          auto tmp = vconcat(w.c[1].y, x1);
          gemm(Trans::C, Trans::N, scalar_t(1.),
               ULV._ch[1]._Q, tmp, scalar_t(0.), w.c[1].x, depth);
          STRUMPACK_HSS_SOLVE_FLOPS
            (gemm_flops(Trans::C, Trans::N, scalar_t(1.),
                        ULV._ch[1]._Q, tmp, scalar_t(0.)));
        } else w.c[1].x.copy(x1);
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
