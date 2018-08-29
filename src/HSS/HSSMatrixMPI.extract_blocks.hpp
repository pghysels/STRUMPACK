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
#ifndef HSS_MATRIX_MPI_EXTRACT_BLOCKS_HPP
#define HSS_MATRIX_MPI_EXTRACT_BLOCKS_HPP

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> std::vector<DistributedMatrix<scalar_t>>
    HSSMatrixMPI<scalar_t>::extract
    (const std::vector<std::vector<std::size_t>>& I,
     const std::vector<std::vector<std::size_t>>& J,
     const BLACSGrid* Bg) const {
      std::vector<DistributedMatrix<scalar_t>> B;
      B.reserve(I.size());
      for (std::size_t i=0; i<I.size(); i++) {
        B.emplace_back(Bg, I[i].size(), J[i].size());
        B[i].zero();
      }
      extract_add(I, J, B);
      return B;
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_add
    (const std::vector<std::vector<std::size_t>>& I,
     const std::vector<std::vector<std::size_t>>& J,
     std::vector<DistM_t>& B) const {
      assert(I.size() == J.size());
      auto nb = I.size();
      if (!nb) return;
      WorkExtractBlocksMPI<scalar_t> w(nb);
      w.I = I;
      w.J = J;
      for (std::size_t k=0; k<nb; k++) {
        w.ycols[k].reserve(J[k].size());
        for (std::size_t c=0; c<J[k].size(); c++)
          w.ycols[k].push_back(c);
      }
      std::vector<bool> odiag(nb, false);
      extract_fwd(w, grid_local(), odiag);
      for (std::size_t k=0; k<nb; k++) {
        w.rl2g[k].reserve(I[k].size());
        for (std::size_t r=0; r<I[k].size(); r++)
          w.rl2g[k].push_back(r);
      }
      for (std::size_t k=0; k<nb; k++) {
        w.cl2g[k].reserve(J[k].size());
        for (std::size_t c=0; c<J[k].size(); c++)
          w.cl2g[k].push_back(c);
      }
      w.z.reserve(nb);
      for (std::size_t k=0; k<nb; k++) {
        w.z.emplace_back(grid(), this->U_rank(), w.ycols[k].size());
        w.z[k].zero();
      }
      std::vector<std::vector<Triplet<scalar_t>>> triplets(nb);
      extract_bwd(triplets, grid_local(), w);
      triplets_to_DistM(triplets, B);
    }

    template<typename scalar_t>
    void HSSMatrixMPI<scalar_t>::triplets_to_DistM
    (std::vector<std::vector<Triplet<scalar_t>>>& triplets,
     std::vector<DistM_t>& B) const {
      if (B.empty()) return;
      const int P = Comm().size();
      const int MB = DistM_t::default_MB;
      const int Bprows = B[0].grid()->nprows();
      const int Bpcols = B[0].grid()->npcols();
      const auto nb = triplets.size();
      struct Quadlet { int r; int c; int k; scalar_t v;
        Quadlet() {}
        Quadlet(Triplet<scalar_t>& t, int k_)
          : r(t._r), c(t._c), k(k_), v(t._v) {}
      };
      std::vector<std::vector<Quadlet>> sbuf(P);
      int maxBrows = 0, maxBcols = 0;
      for (auto& Bi : B) maxBrows = std::max(maxBrows, Bi.rows());
      for (auto& Bi : B) maxBcols = std::max(maxBcols, Bi.cols());
      auto nb_destr = new int[nb*(maxBrows+maxBcols)+P];
      auto ssize = nb_destr + nb*(maxBrows+maxBcols);
      std::fill(nb_destr, nb_destr+nb*(maxBrows+maxBcols), -1);
      std::fill(ssize, ssize+P, 0);
      for (std::size_t k=0; k<nb; k++) {
        auto destr = nb_destr + k*(maxBrows+maxBcols);
        auto destc = destr + maxBrows;
        for (auto& t : triplets[k]) {
          assert(t._r >= 0);
          assert(t._c >= 0);
          assert(t._r < B[k].rows());
          assert(t._c < B[k].cols());
          auto dr = destr[t._r];
          if (dr == -1) dr = destr[t._r] = (t._r / MB) % Bprows;
          auto dc = destc[t._c];
          if (dc == -1) dc = destc[t._c] = ((t._c / MB) % Bpcols) * Bprows;
          assert(dr+dc >= 0 && dr+dc < P);
          ssize[dr+dc]++;
        }
      }
      for (int p=0; p<P; p++)
        sbuf[p].reserve(ssize[p]);
      for (std::size_t k=0; k<nb; k++) {
        auto destr = nb_destr + k*(maxBrows+maxBcols);
        auto destc = destr + maxBrows;
        for (auto& t : triplets[k])
          sbuf[destr[t._r]+destc[t._c]].emplace_back(t, k);
      }
      MPI_Datatype quadlet_type;
      MPI_Type_contiguous(sizeof(Quadlet), MPI_BYTE, &quadlet_type);
      MPI_Type_commit(&quadlet_type);
      std::vector<Quadlet> rbuf;
      std::vector<Quadlet*> pbuf;
      Comm().all_to_all_v(sbuf, rbuf, pbuf, quadlet_type);
      MPI_Type_free(&quadlet_type);
      std::fill(nb_destr, nb_destr+nb*(maxBrows+maxBcols), -1);
      for (auto& q : rbuf) {
        auto k = q.k;
        auto lr = nb_destr + k*(maxBrows+maxBcols);
        auto lc = lr + maxBrows;
        int locr = lr[q.r];
        if (locr == -1) locr = lr[q.r] = B[k].rowg2l_fixed(q.r);
        int locc = lc[q.c];
        if (locc == -1) locc = lc[q.c] = B[k].colg2l_fixed(q.c);
        B[k](locr, locc) += q.v;
      }
      delete[] nb_destr;
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_fwd
    (WorkExtractBlocksMPI<scalar_t>& w, const BLACSGrid* lg,
     std::vector<bool>& odiag) const {
      if (!this->active()) return;
      const auto nb = w.I.size();
      if (this->leaf()) {
        for (std::size_t k=0; k<nb; k++) {
          if (odiag[k])
            w.y[k] = _V.extract_rows(w.J[k]).transpose();
          else w.ycols[k].clear();
        }
      } else {
        w.split_extraction_sets(this->_ch[0]->dims());
        for (std::size_t k=0; k<nb; k++)
          for (std::size_t c=0; c<w.J[k].size(); c++)
            if (w.J[k][c] < this->_ch[0]->cols())
              w.c[0].ycols[k].push_back(w.ycols[k][c]);
            else w.c[1].ycols[k].push_back(w.ycols[k][c]);
        {
          std::vector<bool> odiag0(nb), odiag1(nb);
          for (std::size_t k=0; k<nb; k++) {
            odiag0[k] = odiag[k] || !w.c[1].I[k].empty();
            odiag1[k] = odiag[k] || !w.c[0].I[k].empty();
          }
          this->_ch[0]->extract_fwd(w.c[0], lg, odiag0);
          this->_ch[1]->extract_fwd(w.c[1], lg, odiag1);
        }
        w.communicate_child_ycols(comm(), Pl());
        w.combine_child_ycols(odiag);
        if (this->V_rank()) {
          for (std::size_t k=0; k<nb; k++) {
            if (!odiag[k]) continue;
            DistM_t y01(grid(), this->V_rows(), w.ycols[k].size());
            y01.zero();
            assert(w.c[0].ycols.size() > k);
            assert(w.c[0].y.size() > k);
            copy(this->_ch[0]->V_rank(), w.c[0].ycols[k].size(),
                 w.c[0].y[k], 0, 0, y01, 0, 0, grid()->ctxt_all());
            copy(this->_ch[1]->V_rank(), w.c[1].ycols[k].size(),
                 w.c[1].y[k], 0, 0, y01,
                 this->_ch[0]->V_rank(), w.c[0].ycols[k].size(),
                 grid()->ctxt_all());
            w.y[k] = _V.applyC(y01);
            STRUMPACK_EXTRACTION_FLOPS
              (_V.applyC_flops(y01.cols()));
          }
        } else {
          for (std::size_t k=0; k<nb; k++) {
            if (!odiag[k]) continue;
            w.y[k] = DistM_t(grid(), 0, w.J[k].size());
          }
        }
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_bwd
    (std::vector<std::vector<Triplet<scalar_t>>>& triplets,
     const BLACSGrid* lg, WorkExtractBlocksMPI<scalar_t>& w) const {
      if (!this->active()) return;
      const auto nb = w.I.size();
      if (this->leaf()) {
        for (std::size_t k=0; k<nb; k++) {
          if (_D.active())
            for (std::size_t c=0; c<w.J[k].size(); c++)
              for (std::size_t r=0; r<w.I[k].size(); r++)
                if (_D.is_local(w.I[k][r], w.J[k][c]))
                  triplets[k].emplace_back
                    (w.rl2g[k][r], w.cl2g[k][c],
                     _D.global(w.I[k][r],w.J[k][c]));
          if (w.z[k].cols() && _U.cols()) {
            DistM_t tmp(grid(), w.I[k].size(), w.z[k].cols());
            {
              auto Uex = _U.extract_rows(w.I[k]);
              gemm(Trans::N, Trans::N, scalar_t(1),
                   Uex, w.z[k], scalar_t(0.), tmp);
              STRUMPACK_EXTRACTION_FLOPS
                (gemm_flops(Trans::N, Trans::N, scalar_t(1),
                            Uex, w.z[k], scalar_t(0.)));
            }
            if (tmp.active())
              for (int c=0; c<w.z[k].cols(); c++)
                for (std::size_t r=0; r<w.I[k].size(); r++)
                  if (tmp.is_local(r, c))
                    triplets[k].emplace_back
                      (w.rl2g[k][r], w.zcols[k][c], tmp.global(r,c));
          }
        }
      } else {
        w.split_extraction_sets(this->_ch[0]->dims());
        for (std::size_t k=0; k<nb; k++) {
          w.c[0].rl2g[k].reserve(w.c[0].I[k].size());
          w.c[1].rl2g[k].reserve(w.c[1].I[k].size());
          for (std::size_t r=0; r<w.I[k].size(); r++) {
            if (w.I[k][r] < this->_ch[0]->rows())
              w.c[0].rl2g[k].push_back(w.rl2g[k][r]);
            else w.c[1].rl2g[k].push_back(w.rl2g[k][r]);
          }
          w.c[0].cl2g[k].reserve(w.c[0].J[k].size());
          w.c[1].cl2g[k].reserve(w.c[1].J[k].size());
          for (std::size_t c=0; c<w.J[k].size(); c++) {
            if (w.J[k][c] < this->_ch[0]->cols())
              w.c[0].cl2g[k].push_back(w.cl2g[k][c]);
            else w.c[1].cl2g[k].push_back(w.cl2g[k][c]);
          }
        }
        auto U = _U.dense();

        // TODO split this into comm - comp - comm phases
        for (std::size_t k=0; k<nb; k++) {
          if (!w.c[0].I[k].empty()) {
            auto z0cols = w.c[1].ycols[k].size() + w.z[k].cols();
            auto z0rows = _B01.rows();
            w.c[0].z[k] = DistM_t(this->_ch[0]->grid(lg), z0rows, z0cols);
            if (!w.c[1].ycols[k].empty()) {
              DistM_t z00(grid(), z0rows, w.c[1].ycols[k].size());
              DistM_t wc1y(grid(), _B01.cols(), w.c[1].ycols[k].size());
              copy(_B01.cols(), w.c[1].ycols[k].size(),
                   w.c[1].y[k], 0, 0, wc1y, 0, 0, grid()->ctxt_all());
              gemm(Trans::N, Trans::N, scalar_t(1.), _B01, wc1y,
                   scalar_t(0.), z00);
              STRUMPACK_EXTRACTION_FLOPS
                (gemm_flops(Trans::N, Trans::N, scalar_t(1.), _B01, wc1y,
                            scalar_t(0.)));
              copy(z0rows, w.c[1].ycols[k].size(),
                   z00, 0, 0, w.c[0].z[k], 0, 0, grid()->ctxt_all());
            }
            if (this->U_rank()) {
              DistM_t z01(grid(), z0rows, w.z[k].cols());
              DistMW_t U0(z0rows, this->U_rank(), U, 0, 0);
              gemm(Trans::N, Trans::N, scalar_t(1.), U0, w.z[k],
                   scalar_t(0.), z01);
              STRUMPACK_EXTRACTION_FLOPS
                (gemm_flops(Trans::N, Trans::N, scalar_t(1.), U0, w.z[k],
                            scalar_t(0.)));
              copy(z0rows, w.z[k].cols(), z01, 0, 0,
                   w.c[0].z[k], 0, w.c[1].ycols[k].size(), grid()->ctxt_all());
            } else {
              DistMW_t z01(z0rows, w.z[k].cols(), w.c[0].z[k],
                           0, w.c[1].ycols[k].size());
              z01.zero();
            }
            w.c[0].zcols[k].reserve(z0cols);
            for (auto c : w.c[1].ycols[k]) w.c[0].zcols[k].push_back(c);
            for (auto c : w.zcols[k]) w.c[0].zcols[k].push_back(c);
          }
          if (!w.c[1].I[k].empty()) {
            auto z1cols = w.c[0].ycols[k].size() + w.z[k].cols();
            auto z1rows = _B10.rows();
            w.c[1].z[k] = DistM_t(this->_ch[1]->grid(lg), z1rows, z1cols);
            if (!w.c[0].ycols[k].empty()) {
              DistM_t z10(grid(), z1rows, w.c[0].ycols[k].size());
              DistM_t wc0y(grid(), _B10.cols(), w.c[0].ycols[k].size());
              copy(_B10.cols(), w.c[0].ycols[k].size(),
                   w.c[0].y[k], 0, 0, wc0y, 0, 0, grid()->ctxt_all());
              gemm(Trans::N, Trans::N, scalar_t(1.), _B10, wc0y,
                   scalar_t(0.), z10);
              STRUMPACK_EXTRACTION_FLOPS
                (gemm_flops(Trans::N, Trans::N, scalar_t(1.), _B10, wc0y,
                            scalar_t(0.)));
              copy(z1rows, w.c[0].ycols[k].size(),
                   z10, 0, 0, w.c[1].z[k], 0, 0, grid()->ctxt_all());
            }
            if (this->U_rank()) {
              DistM_t z11(grid(), z1rows, w.z[k].cols());
              DistMW_t U1(z1rows, this->U_rank(), U, this->_ch[0]->U_rank(), 0);
              gemm(Trans::N, Trans::N, scalar_t(1.), U1, w.z[k], scalar_t(0.), z11);
              STRUMPACK_EXTRACTION_FLOPS
                (gemm_flops(Trans::N, Trans::N, scalar_t(1.),
                            U1, w.z[k], scalar_t(0.)));
              copy(z1rows, w.z[k].cols(), z11, 0, 0,
                   w.c[1].z[k], 0, w.c[0].ycols[k].size(), grid()->ctxt_all());
            } else {
              DistMW_t z11(z1rows, w.z[k].cols(), w.c[1].z[k],
                           0, w.c[0].y[k].cols());
              z11.zero();
            }
            w.c[1].zcols[k].reserve(z1cols);
            for (auto c : w.c[0].ycols[k]) w.c[1].zcols[k].push_back(c);
            for (auto c : w.zcols[k]) w.c[1].zcols[k].push_back(c);
          }
        }
        this->_ch[0]->extract_bwd(triplets, lg, w.c[0]);
        this->_ch[1]->extract_bwd(triplets, lg, w.c[1]);
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_MPI_EXTRACT_BLOCKS_HPP
