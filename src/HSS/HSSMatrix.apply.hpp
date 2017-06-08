#ifndef HSS_MATRIX_APPLY_HPP
#define HSS_MATRIX_APPLY_HPP

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> DenseMatrix<scalar_t> HSSMatrix<scalar_t>::apply(const DenseM_t& b) const {
      assert(this->cols() == b.rows());
      DenseM_t c(this->rows(), b.cols());
      apply_HSS(Trans::N, *this, b, scalar_t(0.), c);
      return c;
    }

    template<typename scalar_t> DenseMatrix<scalar_t> HSSMatrix<scalar_t>::applyC(const DenseM_t& b) const {
      assert(this->rows() == b.rows());
      DenseM_t c(this->cols(), b.cols());
      apply_HSS(Trans::C, *this, b, scalar_t(0.), c);
      return c;
    }

    /** C = ta(A_HSS) * B + beta * C   */
    template<typename scalar_t> void apply_HSS
    (Trans ta, const HSSMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b,
     scalar_t beta, DenseMatrix<scalar_t>& c) {
      WorkApply<scalar_t> w;
      if (ta == Trans::N) {
	a.apply_fwd(b, w, true, a._openmp_task_depth);
	a.apply_bwd(b, beta, c, w, true, a._openmp_task_depth);
      } else {
	a.applyT_fwd(b, w, true, a._openmp_task_depth);
	a.applyT_bwd(b, beta, c, w, true, a._openmp_task_depth);
      }
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::apply_fwd
    (const DenseM_t& b, WorkApply<scalar_t>& w, bool isroot, int depth) const {
      if (this->leaf()) {  // TODO can w.tmp1 be stored in b??
	if (!isroot) w.tmp1 = _V.applyC(b.cols(), b.ptr(w.offset.second, 0), b.ld(), depth);
      } else {
	w.c.resize(2);
	w.c[0].offset = w.offset;
	w.c[1].offset = w.offset + this->_ch[0]->dims();
#pragma omp task default(shared) if(depth < params::task_recursion_cutoff_level) \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
	  this->_ch[0]->apply_fwd(b, w.c[0], false, depth+1);
#pragma omp task default(shared) if(depth < params::task_recursion_cutoff_level) \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
	  this->_ch[1]->apply_fwd(b, w.c[1], false, depth+1);
#pragma omp taskwait
	if (!isroot) w.tmp1 = _V.applyC(vconcat(w.c[0].tmp1, w.c[1].tmp1), depth);
      }
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::apply_bwd
    (const DenseM_t& b, scalar_t beta, DenseM_t& c, WorkApply<scalar_t>& w, bool isroot, int depth) const {
      if (this->leaf()) {
	DenseMW_t lc(this->rows(), c.cols(), c, w.offset.second, 0);
	if (_U.cols() && !isroot) { // c = D*b + beta*c + U*w.tmp2
	  gemm(Trans::N, Trans::N, scalar_t(1.), _D, b.ptr(w.offset.second, 0), b.ld(), beta, lc, depth);
	  lc.add(_U.apply(w.tmp2, depth));
	} else  // c = D*b + beta*c
	  gemm(Trans::N, Trans::N, scalar_t(1.), _D, b.ptr(w.offset.second, 0), b.ld(), beta, lc, depth);
      } else {
	w.c[0].tmp2 = DenseM_t(this->_ch[0]->U_rank(), b.cols());
	w.c[1].tmp2 = DenseM_t(this->_ch[1]->U_rank(), b.cols());
	if (isroot || !_U.cols()) { // TODO these can be done in parallel
	  gemm(Trans::N, Trans::N, scalar_t(1.), _B01, w.c[1].tmp1, scalar_t(0.), w.c[0].tmp2, depth);
	  gemm(Trans::N, Trans::N, scalar_t(1.), _B10, w.c[0].tmp1, scalar_t(0.), w.c[1].tmp2, depth);
	} else {
	  auto tmp = _U.apply(w.tmp2, depth);
	  copy(this->_ch[0]->U_rank(), b.cols(), tmp, 0, 0, w.c[0].tmp2, 0, 0);
	  copy(this->_ch[1]->U_rank(), b.cols(), tmp, this->_ch[0]->U_rank(), 0, w.c[1].tmp2, 0, 0);
	  gemm(Trans::N, Trans::N, scalar_t(1.), _B01, w.c[1].tmp1, scalar_t(1.), w.c[0].tmp2, depth);
	  gemm(Trans::N, Trans::N, scalar_t(1.), _B10, w.c[0].tmp1, scalar_t(1.), w.c[1].tmp2, depth);
	}
	// TODO clear tmp1, tmp2??
#pragma omp task default(shared) if(depth < params::task_recursion_cutoff_level) \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
	this->_ch[0]->apply_bwd(b, beta, c, w.c[0], false, depth+1);
#pragma omp task default(shared) if(depth < params::task_recursion_cutoff_level) \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
	this->_ch[1]->apply_bwd(b, beta, c, w.c[1], false, depth+1);
#pragma omp taskwait
      }
    }


    template<typename scalar_t> void HSSMatrix<scalar_t>::applyT_fwd
    (const DenseM_t& b, WorkApply<scalar_t>& w, bool isroot, int depth) const {
      if (this->leaf()) {
	if (!isroot) w.tmp1 = _U.applyC(b.cols(), b.ptr(w.offset.second, 0), b.ld(), depth);
      } else {
	w.c.resize(2);
	w.c[0].offset = w.offset;
	w.c[1].offset = w.offset + this->_ch[0]->dims();
#pragma omp task default(shared) if(depth < params::task_recursion_cutoff_level) \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
	this->_ch[0]->applyT_fwd(b, w.c[0], false, depth+1);
#pragma omp task default(shared) if(depth < params::task_recursion_cutoff_level) \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
	this->_ch[1]->applyT_fwd(b, w.c[1], false, depth+1);
#pragma omp taskwait
	if (!isroot) w.tmp1 = _U.applyC(vconcat(w.c[0].tmp1, w.c[1].tmp1), depth);
      }
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::applyT_bwd
    (const DenseM_t& b, scalar_t beta, DenseM_t& c, WorkApply<scalar_t>& w, bool isroot, int depth) const {
      if (this->leaf()) {
	DenseMW_t lc(this->rows(), c.cols(), c, w.offset.second, 0);
	if (_V.cols() && !isroot) { // c = D'*b + beta*c
	  gemm(Trans::C, Trans::N, scalar_t(1.), _D, b.ptr(w.offset.second, 0), b.ld(), beta, lc, depth);
	  // TODO this creates a temporary!!
	  lc.add(_V.apply(w.tmp2, depth)); // c += V*w.tmp2
	} else  // c = D'*b + beta*c
	  gemm(Trans::C, Trans::N, scalar_t(1.), _D, b.ptr(w.offset.second, 0), b.ld(), beta, lc, depth);
      } else {
	w.c[0].tmp2 = DenseM_t(this->_ch[0]->V_rank(), b.cols());
	w.c[1].tmp2 = DenseM_t(this->_ch[1]->V_rank(), b.cols());
	if (isroot || !_V.cols()) {
	  gemm(Trans::C, Trans::N, scalar_t(1.), _B10, w.c[1].tmp1, scalar_t(0.), w.c[0].tmp2, depth);
	  gemm(Trans::C, Trans::N, scalar_t(1.), _B01, w.c[0].tmp1, scalar_t(0.), w.c[1].tmp2, depth);
	} else {
	  auto tmp = _V.apply(w.tmp2, depth);
	  copy(this->_ch[0]->V_rank(), b.cols(), tmp, 0, 0, w.c[0].tmp2, 0, 0);
	  copy(this->_ch[1]->V_rank(), b.cols(), tmp, this->_ch[0]->V_rank(), 0, w.c[1].tmp2, 0, 0);
	  gemm(Trans::C, Trans::N, scalar_t(1.), _B10, w.c[1].tmp1, scalar_t(1.), w.c[0].tmp2, depth);
	  gemm(Trans::C, Trans::N, scalar_t(1.), _B01, w.c[0].tmp1, scalar_t(1.), w.c[1].tmp2, depth);
	}
#pragma omp task default(shared) if(depth < params::task_recursion_cutoff_level) \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
	this->_ch[0]->applyT_bwd(b, beta, c, w.c[0], false, depth+1);
#pragma omp task default(shared) if(depth < params::task_recursion_cutoff_level) \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
	this->_ch[1]->applyT_bwd(b, beta, c, w.c[1], false, depth+1);
#pragma omp taskwait
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_APPLY_HPP
