#ifndef HSS_MATRIX_MPI_EXTRACT_HPP
#define HSS_MATRIX_MPI_EXTRACT_HPP

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> scalar_t HSSMatrixMPI<scalar_t>::get(std::size_t i, std::size_t j) const {
      if (this->leaf()) return _D.all_global(i, j);
      DistM_t e(_ctxt, this->cols(), 1);
      e.zero();
      e.global(j, 0, scalar_t(1.));
      return apply(e).all_global(i, 0);
    }

    template<typename scalar_t> DistributedMatrix<scalar_t> HSSMatrixMPI<scalar_t>::extract
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     int Bctxt, int Bprows, int Bpcols) const {
      DistM_t B(Bctxt, I.size(), J.size());
      B.zero();
      extract_add(I, J, B, Bprows, Bpcols);
      return B;
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_add
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DistM_t& B, int Bprows, int Bpcols) const {
      WorkExtractMPI<scalar_t> w;
      w.J = J;
      w.I = I;
      w.ycols.reserve(J.size());
      for (std::size_t c=0; c<J.size(); c++) w.ycols.push_back(c);
      extract_fwd(w, ctxt_loc(), false);
      w.rl2g.reserve(I.size());
      for (std::size_t r=0; r<I.size(); r++) w.rl2g.push_back(r);
      w.cl2g.reserve(J.size());
      for (std::size_t c=0; c<J.size(); c++) w.cl2g.push_back(c);

      // TODO is this necessary???
      w.z = DistM_t(_ctxt, this->U_rank(), w.ycols.size());
      w.z.zero();

      std::vector<Triplet<scalar_t>> triplets;
      extract_bwd(triplets, ctxt_loc(), w);
      triplets_to_DistM(triplets, B, Bprows, Bpcols);
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::triplets_to_DistM
    (std::vector<Triplet<scalar_t>>& triplets, DistM_t& B, int Bprows, int Bpcols) const {
      auto P = mpi_nprocs(_comm);
      std::vector<std::vector<Triplet<scalar_t>>> sbuf(P);
      const int MB = DistM_t::default_MB;
      for (auto& t : triplets) {
	auto dest = (t._r / MB) % Bprows + ((t._c / MB) % Bpcols) * Bprows;
	sbuf[dest].push_back(t);
      }
      MPI_Datatype triplet_type;
      create_triplet_mpi_type<scalar_t>(&triplet_type);
      Triplet<scalar_t>* rbuf;
      std::size_t totrsize;
      all_to_all_v(sbuf, rbuf, totrsize, _comm, triplet_type);
      if (B.active())
	for (auto t=rbuf; t!=rbuf+totrsize; t++)
	  B.global_fixed(t->_r, t->_c) += t->_v;
      delete[] rbuf;
    }

    // TODO lctxt is not used here
    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_fwd
    (WorkExtractMPI<scalar_t>& w, int lctxt, bool odiag) const {
      if (!this->active() || w.J.empty()) return;
      if (this->leaf()) {
	if (odiag) w.y = _V.extract_rows(w.J, _ctxt_all).transpose();
	else w.ycols.clear();
      }	else {
	w.split_extraction_sets(this->_ch[0]->dims());
	for (std::size_t c=0; c<w.J.size(); c++) {
	  if (w.J[c] < this->_ch[0]->cols()) w.c[0].ycols.push_back(w.ycols[c]);
	  else w.c[1].ycols.push_back(w.ycols[c]);
	}
	this->_ch[0]->extract_fwd(w.c[0], lctxt, odiag || !w.c[1].I.empty());
	this->_ch[1]->extract_fwd(w.c[1], lctxt, odiag || !w.c[0].I.empty());
	w.ycols.clear();
	w.communicate_child_ycols(_comm, Pl(_nprocs));
	if (!odiag) return;
	w.combine_child_ycols();
	if (this->V_rank()) {
	  DistM_t y01(_ctxt, this->V_rows(), w.ycols.size());
	  y01.zero();
	  copy(this->_ch[0]->V_rank(), w.c[0].ycols.size(), w.c[0].y, 0, 0, y01, 0, 0, _ctxt_all);
	  copy(this->_ch[1]->V_rank(), w.c[1].ycols.size(), w.c[1].y, 0, 0, y01,
	       this->_ch[0]->V_rank(), w.c[0].ycols.size(), _ctxt_all);
	  w.y = _V.applyC(y01);
	} else w.y = DistM_t(_ctxt, 0, w.J.size());
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_bwd
    (std::vector<Triplet<scalar_t>>& triplets,
     int lctxt, WorkExtractMPI<scalar_t>& w) const {
      if (!this->active() || w.I.empty()) return;
      if (this->leaf()) {
	if (_D.active())
	  for (std::size_t c=0; c<w.J.size(); c++)
	    for (std::size_t r=0; r<w.I.size(); r++)
	      if (_D.is_local(w.I[r], w.J[c]))
		triplets.emplace_back(w.rl2g[r], w.cl2g[c], _D.global(w.I[r],w.J[c]));
	if (w.z.cols() && _U.cols()) {
	  DistM_t tmp(_ctxt, w.I.size(), w.z.cols());
	  gemm(Trans::N, Trans::N, scalar_t(1), _U.extract_rows(w.I, _ctxt_all), w.z, scalar_t(0.), tmp);
	  if (tmp.active())
	    for (int c=0; c<w.z.cols(); c++)
	      for (std::size_t r=0; r<w.I.size(); r++)
		if (tmp.is_local(r, c))
		  triplets.emplace_back(w.rl2g[r], w.zcols[c], tmp.global(r,c));
	}
      } else {
	w.split_extraction_sets(this->_ch[0]->dims());
	w.c[0].rl2g.reserve(w.c[0].I.size());
	w.c[1].rl2g.reserve(w.c[1].I.size());
	for (std::size_t r=0; r<w.I.size(); r++) {
	  if (w.I[r] < this->_ch[0]->rows()) w.c[0].rl2g.push_back(w.rl2g[r]);
	  else w.c[1].rl2g.push_back(w.rl2g[r]);
	}
	w.c[0].cl2g.reserve(w.c[0].J.size());
	w.c[1].cl2g.reserve(w.c[1].J.size());
	for (std::size_t c=0; c<w.J.size(); c++) {
	  if (w.J[c] < this->_ch[0]->cols()) w.c[0].cl2g.push_back(w.cl2g[c]);
	  else w.c[1].cl2g.push_back(w.cl2g[c]);
	}
	auto U = _U.dense();
	if (!w.c[0].I.empty()) {
	  auto z0cols = w.c[1].ycols.size() + w.z.cols();
	  auto z0rows = _B01.rows();
	  w.c[0].z = DistM_t(this->_ch[0]->ctxt(lctxt), z0rows, z0cols);
	  if (!w.c[1].ycols.empty()) {
	    DistM_t z00(_ctxt, z0rows, w.c[1].ycols.size());
	    DistM_t wc1y(_ctxt, _B01.cols(), w.c[1].ycols.size());
	    copy(_B01.cols(), w.c[1].ycols.size(), w.c[1].y, 0, 0, wc1y, 0, 0, _ctxt_all);
	    gemm(Trans::N, Trans::N, scalar_t(1.), _B01, wc1y, scalar_t(0.), z00);
	    copy(z0rows, w.c[1].ycols.size(), z00, 0, 0, w.c[0].z, 0, 0, _ctxt_all);
	  }
	  if (this->U_rank()) {
	    DistM_t z01(_ctxt, z0rows, w.z.cols());
	    DistMW_t U0(z0rows, this->U_rank(), U, 0, 0);
	    gemm(Trans::N, Trans::N, scalar_t(1.), U0, w.z, scalar_t(0.), z01);
	    copy(z0rows, w.z.cols(), z01, 0, 0, w.c[0].z, 0, w.c[1].ycols.size(), _ctxt_all);
	  } else {
	    DistMW_t z01(z0rows, w.z.cols(), w.c[0].z, 0, w.c[1].ycols.size());
	    z01.zero();
	  }
	  w.c[0].zcols.reserve(z0cols);
	  for (auto c : w.c[1].ycols) w.c[0].zcols.push_back(c);
	  for (auto c : w.zcols) w.c[0].zcols.push_back(c);
	}
	if (!w.c[1].I.empty()) {
	  auto z1cols = w.c[0].ycols.size() + w.z.cols();
	  auto z1rows = _B10.rows();
	  w.c[1].z = DistM_t(this->_ch[1]->ctxt(lctxt), z1rows, z1cols);
	  if (!w.c[0].ycols.empty()) {
	    DistM_t z10(_ctxt, z1rows, w.c[0].ycols.size());
	    DistM_t wc0y(_ctxt, _B10.cols(), w.c[0].ycols.size());
	    copy(_B10.cols(), w.c[0].ycols.size(), w.c[0].y, 0, 0, wc0y, 0, 0, _ctxt_all);
	    gemm(Trans::N, Trans::N, scalar_t(1.), _B10, wc0y, scalar_t(0.), z10);
	    copy(z1rows, w.c[0].ycols.size(), z10, 0, 0, w.c[1].z, 0, 0, _ctxt_all);
	  }
	  if (this->U_rank()) {
	    DistM_t z11(_ctxt, z1rows, w.z.cols());
	    DistMW_t U1(z1rows, U.cols(), U, this->_ch[0]->U_rank(), 0);
	    gemm(Trans::N, Trans::N, scalar_t(1.), U1, w.z, scalar_t(0.), z11);
	    copy(z1rows, w.z.cols(), z11, 0, 0, w.c[1].z, 0, w.c[0].ycols.size(), _ctxt_all);
	  } else {
	    DistMW_t z11(z1rows, w.z.cols(), w.c[1].z, 0, w.c[0].y.cols());
	    z11.zero();
	  }
	  w.c[1].zcols.reserve(z1cols);
	  for (auto c : w.c[0].ycols) w.c[1].zcols.push_back(c);
	  for (auto c : w.zcols) w.c[1].zcols.push_back(c);
	}
	this->_ch[0]->extract_bwd(triplets, lctxt, w.c[0]);
	this->_ch[1]->extract_bwd(triplets, lctxt, w.c[1]);
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_MPI_EXTRACT_HPP
