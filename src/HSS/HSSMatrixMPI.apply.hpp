#ifndef HSS_MATRIX_MPI_APPLY_HPP
#define HSS_MATRIX_MPI_APPLY_HPP

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> DistributedMatrix<scalar_t>
    HSSMatrixMPI<scalar_t>::apply(const DistM_t& b) const {
      assert(this->cols() == std::size_t(b.rows()));
      DistM_t c(b.ctxt(), this->rows(), b.cols());
#if !defined(NDEBUG)
      c.zero(); // silence valgrind
#endif
      apply_HSS(Trans::N, *this, b, scalar_t(0.), c);
      return c;
    }

    template<typename scalar_t> DistributedMatrix<scalar_t>
    HSSMatrixMPI<scalar_t>::applyC(const DistM_t& b) const {
      assert(this->rows() == std::size_t(b.rows()));
      DistM_t c(b.ctxt(), this->cols(), b.cols());
#if !defined(NDEBUG)
      c.zero(); // silence valgrind
#endif
      apply_HSS(Trans::C, *this, b, scalar_t(0.), c);
      return c;
    }

    template<typename scalar_t> void apply_HSS
    (Trans ta, const HSSMatrixMPI<scalar_t>& a, const DistributedMatrix<scalar_t>& b,
     scalar_t beta, DistributedMatrix<scalar_t>& c) {
      DistSubLeaf<scalar_t> B(b.cols(), &a, a.ctxt_loc(), b), C(b.cols(), &a, a.ctxt_loc());
      WorkApplyMPI<scalar_t> w;
      if (ta == Trans::N) {
	a.apply_fwd(B, w, true);
	a.apply_bwd(B, beta, C, w, true);
      } else {
	a.applyT_fwd(B, w, true);
	a.applyT_bwd(B, beta, C, w, true);
      }
      C.from_block_row(c);
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::apply_fwd
    (const DistSubLeaf<scalar_t>& B, WorkApplyMPI<scalar_t>& w, bool isroot) const {
      if (!this->active()) return;
      if (this->leaf()) {
	if (!isroot) w.tmp1 = _V.applyC(B.leaf);
      } else {
	w.c.resize(2);
	w.c[0].offset = w.offset;
	w.c[1].offset = w.offset + this->_ch[0]->dims();
	this->_ch[0]->apply_fwd(B, w.c[0], mpi_nprocs(_comm)==1);
	this->_ch[1]->apply_fwd(B, w.c[1], mpi_nprocs(_comm)==1);
	if (!isroot)
	  w.tmp1 = _V.applyC(vconcat(B.cols(), this->_B10.cols(), this->_B01.cols(),
				     w.c[0].tmp1, w.c[1].tmp1, _ctxt, _ctxt_all));
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::apply_bwd
    (const DistSubLeaf<scalar_t>& B, scalar_t beta, DistSubLeaf<scalar_t>& C,
     WorkApplyMPI<scalar_t>& w, bool isroot) const {
      if (!this->active()) return;
      if (this->leaf()) {
	if (this->U_rank() && !isroot) {  // c = D*b + beta*c + U*w.tmp2
	  gemm(Trans::N, Trans::N, scalar_t(1.), _D, B.leaf, beta, C.leaf);
	  C.leaf.add(_U.apply(w.tmp2));
	} else gemm(Trans::N, Trans::N, scalar_t(1.), _D, B.leaf, beta, C.leaf);
      } else {
	auto n = B.cols();
	DistM_t c0tmp1(_ctxt, _B10.cols(), n, w.c[0].tmp1, _ctxt_all);
	DistM_t c1tmp1(_ctxt, _B01.cols(), n, w.c[1].tmp1, _ctxt_all);
	DistM_t c0tmp2(_ctxt, _B01.rows(), n);
	DistM_t c1tmp2(_ctxt, _B10.rows(), n);
	if (isroot || !this->U_rank()) {
	  gemm(Trans::N, Trans::N, scalar_t(1.), _B01, c1tmp1, scalar_t(0.), c0tmp2);
	  gemm(Trans::N, Trans::N, scalar_t(1.), _B10, c0tmp1, scalar_t(0.), c1tmp2);
	} else {
	  auto tmp = _U.apply(w.tmp2);
	  copy(this->_B01.rows(), n, tmp, 0, 0, c0tmp2, 0, 0, _ctxt_all);
	  copy(this->_B10.rows(), n, tmp, this->_B01.rows(), 0, c1tmp2, 0, 0, _ctxt_all);
	  gemm(Trans::N, Trans::N, scalar_t(1.), _B01, c1tmp1, scalar_t(1.), c0tmp2);
	  gemm(Trans::N, Trans::N, scalar_t(1.), _B10, c0tmp1, scalar_t(1.), c1tmp2);
	}
	w.c[0].tmp2 = DistM_t(w.c[0].tmp1.ctxt(), _B01.rows(), n, c0tmp2, _ctxt_all);
	w.c[1].tmp2 = DistM_t(w.c[1].tmp1.ctxt(), _B10.rows(), n, c1tmp2, _ctxt_all);
	this->_ch[0]->apply_bwd(B, beta, C, w.c[0], false);
	this->_ch[1]->apply_bwd(B, beta, C, w.c[1], false);
      }
    }


    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::applyT_fwd
    (const DistSubLeaf<scalar_t>& B, WorkApplyMPI<scalar_t>& w, bool isroot) const {
      if (!this->active()) return;
      if (this->leaf()) {
	if (!isroot) w.tmp1 = _U.applyC(B.leaf);
      } else {
	w.c.resize(2);
	w.c[0].offset = w.offset;
	w.c[1].offset = w.offset + this->_ch[0]->dims();
	this->_ch[0]->applyT_fwd(B, w.c[0], mpi_nprocs(_comm)==1);
	this->_ch[1]->applyT_fwd(B, w.c[1], mpi_nprocs(_comm)==1);
	if (!isroot)
	  w.tmp1 = _U.applyC(vconcat(B.cols(), this->_B01.rows(), this->_B10.rows(),
				     w.c[0].tmp1, w.c[1].tmp1, _ctxt, _ctxt_all));
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::applyT_bwd
    (const DistSubLeaf<scalar_t>& B, scalar_t beta, DistSubLeaf<scalar_t>& C,
     WorkApplyMPI<scalar_t>& w, bool isroot) const {
      if (!this->active()) return;
      if (this->leaf()) {
	if (this->V_rank() && !isroot) {  // c = D*b + beta*c + U*w.tmp2
	  gemm(Trans::C, Trans::N, scalar_t(1.), _D, B.leaf, beta, C.leaf);
	  C.leaf.add(_V.apply(w.tmp2));
	} else gemm(Trans::C, Trans::N, scalar_t(1.), _D, B.leaf, beta, C.leaf);
      } else {
	auto n = B.cols();
	DistM_t c0tmp1(_ctxt, _B01.rows(), n, w.c[0].tmp1, _ctxt_all);
	DistM_t c1tmp1(_ctxt, _B10.rows(), n, w.c[1].tmp1, _ctxt_all);
	DistM_t c0tmp2(_ctxt, _B10.cols(), n);
	DistM_t c1tmp2(_ctxt, _B01.cols(), n);
	if (isroot || !this->V_rank()) {
	  gemm(Trans::C, Trans::N, scalar_t(1.), _B10, c1tmp1, scalar_t(0.), c0tmp2);
	  gemm(Trans::C, Trans::N, scalar_t(1.), _B01, c0tmp1, scalar_t(0.), c1tmp2);
	} else {
	  auto tmp = _V.apply(w.tmp2);
	  copy(c0tmp2.rows(), n, tmp, 0, 0, c0tmp2, 0, 0, _ctxt_all);
	  copy(c1tmp2.rows(), n, tmp, c0tmp2.rows(), 0, c1tmp2, 0, 0, _ctxt_all);
	  gemm(Trans::C, Trans::N, scalar_t(1.), _B10, c1tmp1, scalar_t(1.), c0tmp2);
	  gemm(Trans::C, Trans::N, scalar_t(1.), _B01, c0tmp1, scalar_t(1.), c1tmp2);
	}
	w.c[0].tmp2 = DistM_t(w.c[0].tmp1.ctxt(), c0tmp2.rows(), n, c0tmp2, _ctxt_all);
	w.c[1].tmp2 = DistM_t(w.c[1].tmp1.ctxt(), c1tmp2.rows(), n, c1tmp2, _ctxt_all);
	this->_ch[0]->applyT_bwd(B, beta, C, w.c[0], false);
	this->_ch[1]->applyT_bwd(B, beta, C, w.c[1], false);
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_MPI_APPLY_HPP
