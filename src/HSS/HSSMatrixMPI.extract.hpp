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
#ifndef HSS_MATRIX_MPI_EXTRACT_HPP
#define HSS_MATRIX_MPI_EXTRACT_HPP

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> scalar_t
    HSSMatrixMPI<scalar_t>::get(std::size_t i, std::size_t j) const {
      if (this->leaf()) return _D.all_global(i, j);
      DistM_t e(grid(), this->cols(), 1);
      e.zero();
      e.global(j, 0, scalar_t(1.));
      return apply(e).all_global(i, 0);
    }

    template<typename scalar_t> DistributedMatrix<scalar_t>
    HSSMatrixMPI<scalar_t>::extract
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     const BLACSGrid* g) const {
      DistM_t B(g, I.size(), J.size());
      B.zero();
      extract_add(I, J, B);
      return B;
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_add
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DistM_t& B) const {
      WorkExtractMPI<scalar_t> w;
      w.J = J;
      w.I = I;
      w.ycols.reserve(J.size());
      for (std::size_t c=0; c<J.size(); c++) w.ycols.push_back(c);
      extract_fwd(w, grid_local(), false);
      w.rl2g.reserve(I.size());
      for (std::size_t r=0; r<I.size(); r++) w.rl2g.push_back(r);
      w.cl2g.reserve(J.size());
      for (std::size_t c=0; c<J.size(); c++) w.cl2g.push_back(c);

      // TODO is this necessary???
      w.z = DistM_t(grid(), this->U_rank(), w.ycols.size());
      w.z.zero();

      std::vector<Triplet<scalar_t>> triplets;
      extract_bwd(triplets, grid_local(), w);
      triplets_to_DistM(triplets, B);
    }

    template<typename scalar_t>
    void HSSMatrixMPI<scalar_t>::triplets_to_DistM
    (std::vector<Triplet<scalar_t>>& triplets, DistM_t& B) const {
      const int MB = DistM_t::default_MB;
      const int Bprows = B.grid()->nprows();
      const int Bpcols = B.grid()->npcols();
      const int P = Comm().size();
      auto destr = new int[B.rows()+B.cols()+P];
      auto destc = destr + B.rows();
      auto ssize = destc + B.cols();
      std::fill(destr, destr+B.rows()+B.cols(), -1);
      std::fill(ssize, ssize+P, 0);
      for (auto& t : triplets) {
        assert(t._r >= 0);
        assert(t._c >= 0);
        assert(t._r < B.rows());
        assert(t._c < B.cols());
        auto dr = destr[t._r];
        if (dr == -1) dr = destr[t._r] = (t._r / MB) % Bprows;
        auto dc = destc[t._c];
        if (dc == -1) dc = destc[t._c] = ((t._c / MB) % Bpcols) * Bprows;
        assert(dr+dc >= 0 && dr+dc < P);
        ssize[dr+dc]++;
      }
      std::vector<std::vector<Triplet<scalar_t>>> sbuf(P);
      for (int p=0; p<P; p++)
        sbuf[p].reserve(ssize[p]);
      for (auto& t : triplets)
        sbuf[destr[t._r]+destc[t._c]].emplace_back(t);
      MPI_Datatype triplet_type;
      create_triplet_mpi_type<scalar_t>(&triplet_type);
      std::vector<Triplet<scalar_t>> rbuf;
      std::vector<Triplet<scalar_t>*> pbuf;
      Comm().all_to_all_v(sbuf, rbuf, pbuf, triplet_type);
      MPI_Type_free(&triplet_type);
      if (B.active()) {
        std::fill(destr, destr+B.rows()+B.cols(), -1);
        auto lr = destr;
        auto lc = destc;
        for (auto& t : rbuf) {
          int locr = lr[t._r];
          if (locr == -1) locr = lr[t._r] = B.rowg2l_fixed(t._r);
          int locc = lc[t._c];
          if (locc == -1) locc = lc[t._c] = B.colg2l_fixed(t._c);
          B(locr, locc) += t._v;
        }
      }
      delete[] destr;
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_fwd
    (WorkExtractMPI<scalar_t>& w, const BLACSGrid* lg, bool odiag) const {
      if (!this->active() || w.J.empty()) return;
      if (this->leaf()) {
        if (odiag) w.y = _V.extract_rows(w.J).transpose();
        else w.ycols.clear();
      } else {
        w.split_extraction_sets(this->_ch[0]->dims());
        for (std::size_t c=0; c<w.J.size(); c++) {
          if (w.J[c] < this->_ch[0]->cols())
            w.c[0].ycols.push_back(w.ycols[c]);
          else w.c[1].ycols.push_back(w.ycols[c]);
        }
        this->_ch[0]->extract_fwd(w.c[0], lg, odiag || !w.c[1].I.empty());
        this->_ch[1]->extract_fwd(w.c[1], lg, odiag || !w.c[0].I.empty());
        w.ycols.clear();
        w.communicate_child_ycols(comm(), Pl());
        if (!odiag) return;
        w.combine_child_ycols();
        if (this->V_rank()) {
          DistM_t y01(grid(), this->V_rows(), w.ycols.size());
          y01.zero();
          copy(this->_ch[0]->V_rank(), w.c[0].ycols.size(), w.c[0].y, 0, 0,
               y01, 0, 0, grid()->ctxt_all());
          copy(this->_ch[1]->V_rank(), w.c[1].ycols.size(), w.c[1].y, 0, 0,
               y01, this->_ch[0]->V_rank(), w.c[0].ycols.size(),
               grid()->ctxt_all());
          w.y = _V.applyC(y01);
          STRUMPACK_EXTRACTION_FLOPS
            (_V.applyC_flops(y01.cols()));
        } else w.y = DistM_t(grid(), 0, w.J.size());
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_bwd
    (std::vector<Triplet<scalar_t>>& triplets, const BLACSGrid* lg,
     WorkExtractMPI<scalar_t>& w) const {
      if (!this->active() || w.I.empty()) return;
      if (this->leaf()) {
        if (_D.active())
          for (std::size_t c=0; c<w.J.size(); c++)
            for (std::size_t r=0; r<w.I.size(); r++)
              if (_D.is_local(w.I[r], w.J[c]))
                triplets.emplace_back
                  (w.rl2g[r], w.cl2g[c], _D.global(w.I[r],w.J[c]));
        if (w.z.cols() && _U.cols()) {
          DistM_t tmp(grid(), w.I.size(), w.z.cols());
          {
            auto Uex = _U.extract_rows(w.I);
            gemm(Trans::N, Trans::N, scalar_t(1),
                 Uex, w.z, scalar_t(0.), tmp);
            STRUMPACK_EXTRACTION_FLOPS
              (gemm_flops(Trans::N, Trans::N, scalar_t(1),
                          Uex, w.z, scalar_t(0.)));
          }
          if (tmp.active())
            for (int c=0; c<w.z.cols(); c++)
              for (std::size_t r=0; r<w.I.size(); r++)
                if (tmp.is_local(r, c))
                  triplets.emplace_back
                    (w.rl2g[r], w.zcols[c], tmp.global(r,c));
        }
      } else {
        w.split_extraction_sets(this->_ch[0]->dims());
        w.c[0].rl2g.reserve(w.c[0].I.size());
        w.c[1].rl2g.reserve(w.c[1].I.size());
        for (std::size_t r=0; r<w.I.size(); r++) {
          if (w.I[r] < this->_ch[0]->rows())
            w.c[0].rl2g.push_back(w.rl2g[r]);
          else w.c[1].rl2g.push_back(w.rl2g[r]);
        }
        w.c[0].cl2g.reserve(w.c[0].J.size());
        w.c[1].cl2g.reserve(w.c[1].J.size());
        for (std::size_t c=0; c<w.J.size(); c++) {
          if (w.J[c] < this->_ch[0]->cols())
            w.c[0].cl2g.push_back(w.cl2g[c]);
          else w.c[1].cl2g.push_back(w.cl2g[c]);
        }
        auto U = _U.dense();
        if (!w.c[0].I.empty()) {
          auto z0cols = w.c[1].ycols.size() + w.z.cols();
          auto z0rows = _B01.rows();
          w.c[0].z = DistM_t(this->_ch[0]->grid(lg), z0rows, z0cols);
          if (!w.c[1].ycols.empty()) {
            DistM_t z00(grid(), z0rows, w.c[1].ycols.size());
            DistM_t wc1y(grid(), _B01.cols(), w.c[1].ycols.size());
            copy(_B01.cols(), w.c[1].ycols.size(),
                 w.c[1].y, 0, 0, wc1y, 0, 0, grid()->ctxt_all());
            gemm(Trans::N, Trans::N, scalar_t(1.), _B01, wc1y,
                 scalar_t(0.), z00);
            STRUMPACK_EXTRACTION_FLOPS
              (gemm_flops(Trans::N, Trans::N, scalar_t(1.), _B01, wc1y,
                          scalar_t(0.)));
            copy(z0rows, w.c[1].ycols.size(), z00, 0, 0,
                 w.c[0].z, 0, 0, grid()->ctxt_all());
          }
          if (this->U_rank()) {
            DistM_t z01(grid(), z0rows, w.z.cols());
            DistMW_t U0(z0rows, this->U_rank(), U, 0, 0);
            gemm(Trans::N, Trans::N, scalar_t(1.), U0, w.z,
                 scalar_t(0.), z01);
            STRUMPACK_EXTRACTION_FLOPS
              (gemm_flops(Trans::N, Trans::N, scalar_t(1.), U0, w.z,
                          scalar_t(0.)));
            copy(z0rows, w.z.cols(), z01, 0, 0,
                 w.c[0].z, 0, w.c[1].ycols.size(), grid()->ctxt_all());
          } else {
            DistMW_t z01(z0rows, w.z.cols(), w.c[0].z,
                         0, w.c[1].ycols.size());
            z01.zero();
          }
          w.c[0].zcols.reserve(z0cols);
          for (auto c : w.c[1].ycols) w.c[0].zcols.push_back(c);
          for (auto c : w.zcols) w.c[0].zcols.push_back(c);
        }
        if (!w.c[1].I.empty()) {
          auto z1cols = w.c[0].ycols.size() + w.z.cols();
          auto z1rows = _B10.rows();
          w.c[1].z = DistM_t(this->_ch[1]->grid(lg), z1rows, z1cols);
          if (!w.c[0].ycols.empty()) {
            DistM_t z10(grid(), z1rows, w.c[0].ycols.size());
            DistM_t wc0y(grid(), _B10.cols(), w.c[0].ycols.size());
            copy(_B10.cols(), w.c[0].ycols.size(),
                 w.c[0].y, 0, 0, wc0y, 0, 0, grid()->ctxt_all());
            gemm(Trans::N, Trans::N, scalar_t(1.), _B10, wc0y,
                 scalar_t(0.), z10);
            STRUMPACK_EXTRACTION_FLOPS
              (gemm_flops(Trans::N, Trans::N, scalar_t(1.), _B10, wc0y,
                          scalar_t(0.)));
            copy(z1rows, w.c[0].ycols.size(), z10, 0, 0,
                 w.c[1].z, 0, 0, grid()->ctxt_all());
          }
          if (this->U_rank()) {
            DistM_t z11(grid(), z1rows, w.z.cols());
            DistMW_t U1(z1rows, this->U_rank(), U, this->_ch[0]->U_rank(), 0);
            gemm(Trans::N, Trans::N, scalar_t(1.),
                 U1, w.z, scalar_t(0.), z11);
            STRUMPACK_EXTRACTION_FLOPS
              (gemm_flops(Trans::N, Trans::N, scalar_t(1.),
                          U1, w.z, scalar_t(0.)));
            copy(z1rows, w.z.cols(), z11, 0, 0, w.c[1].z, 0,
                 w.c[0].ycols.size(), grid()->ctxt_all());
          } else {
            DistMW_t z11(z1rows, w.z.cols(), w.c[1].z,
                         0, w.c[0].y.cols());
            z11.zero();
          }
          w.c[1].zcols.reserve(z1cols);
          for (auto c : w.c[0].ycols) w.c[1].zcols.push_back(c);
          for (auto c : w.zcols) w.c[1].zcols.push_back(c);
        }
        this->_ch[0]->extract_bwd(triplets, lg, w.c[0]);
        this->_ch[1]->extract_bwd(triplets, lg, w.c[1]);
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_MPI_EXTRACT_HPP
