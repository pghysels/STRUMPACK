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
 * five (5) year renewals, the U.S. Government igs granted for itself
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
#ifndef EXTEND_ADD_HPP
#define EXTEND_ADD_HPP

#include "DistributedMatrix.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixMPI;
  template<typename scalar_t,typename integer_t> class FrontalMatrixHSSMPI;
  template<typename scalar_t,typename integer_t> class FrontalMatrixDenseMPI;

  template<typename scalar_t,typename integer_t> class ExtendAdd {
    using DistM_t = DistributedMatrix<scalar_t>;
  public:
    static void extend_add_copy_to_buffers
    (DistM_t& CB, DistM_t& F11, DistM_t& F12, DistM_t& F21, DistM_t& F22,
     std::vector<std::vector<scalar_t>>& sbuf,
     FrontalMatrixDenseMPI<scalar_t,integer_t>* pa,
     std::vector<std::size_t>& I) {
      auto pa_row = new int[CB.lrows()+CB.lcols()];
      auto pa_col = pa_row+CB.lrows();
      auto lrows = CB.lrows();
      auto lcols = CB.lcols();
      if (CB.fixed()) {
        for (int r=0; r<lrows; r++) pa_row[r] = I[CB.rowl2g_fixed(r)];
        for (int c=0; c<lcols; c++) pa_col[c] = I[CB.coll2g_fixed(c)];
      } else {
        for (int r=0; r<lrows; r++) pa_row[r] = I[CB.rowl2g(r)];
        for (int c=0; c<lcols; c++) pa_col[c] = I[CB.coll2g(c)];
      }
      int r_upd, c_upd;
      for (r_upd=0; r_upd<lrows; r_upd++)
        if (pa_row[r_upd] >= pa->dim_sep) break;
      for (c_upd=0; c_upd<lcols; c_upd++)
        if (pa_col[c_upd] >= pa->dim_sep) break;
      if (CB.fixed()) {
        for (int c=0; c<c_upd; c++) // F11
          for (int r=0; r<r_upd; r++)
            sbuf[pa->find_rank_fixed(pa_row[r], pa_col[c], F11)].
              push_back(CB(r,c));
        for (int c=c_upd; c<lcols; c++) // F12
          for (int r=0; r<r_upd; r++)
            sbuf[pa->find_rank_fixed(pa_row[r], pa_col[c]-pa->dim_sep, F12)].
              push_back(CB(r,c));
        for (int c=0; c<c_upd; c++) // F21
          for (int r=r_upd; r<lrows; r++)
            sbuf[pa->find_rank_fixed(pa_row[r]-pa->dim_sep, pa_col[c], F21)].
              push_back(CB(r,c));
        for (int c=c_upd; c<lcols; c++) // F22
          for (int r=r_upd; r<lrows; r++)
            sbuf[pa->find_rank_fixed(pa_row[r]-pa->dim_sep,
                                     pa_col[c]-pa->dim_sep, F22)].
              push_back(CB(r,c));
      } else {
        for (int c=0; c<c_upd; c++) // F11
          for (int r=0; r<r_upd; r++)
            sbuf[pa->find_rank(pa_row[r], pa_col[c], F11)].
              push_back(CB(r,c));
        for (int c=c_upd; c<lcols; c++) // F12
          for (int r=0; r<r_upd; r++)
            sbuf[pa->find_rank(pa_row[r], pa_col[c]-pa->dim_sep, F12)].
              push_back(CB(r,c));
        for (int c=0; c<c_upd; c++) // F21
          for (int r=r_upd; r<lrows; r++)
            sbuf[pa->find_rank(pa_row[r]-pa->dim_sep, pa_col[c], F21)].
              push_back(CB(r,c));
        for (int c=c_upd; c<lcols; c++) // F22
          for (int r=r_upd; r<lrows; r++)
            sbuf[pa->find_rank(pa_row[r]-pa->dim_sep,
                               pa_col[c]-pa->dim_sep, F22)].
              push_back(CB(r,c));
      }
      delete[] pa_row;
    }

    // TODO use skinny-extend-add
    static void extend_add_column_copy_to_buffers
    (DistM_t& Bchild, DistM_t& Bsep, DistM_t& Bupd,
     std::vector<std::vector<scalar_t>>& sbuf,
     FrontalMatrixMPI<scalar_t,integer_t>* pa, std::vector<std::size_t>& I) {
      if (Bchild.fixed()) {
        for (int r=0; r<Bchild.lrows(); r++) {
          integer_t pa_row = I[Bchild.rowl2g_fixed(r)];
          if (pa_row < pa->dim_sep)
            sbuf[pa->find_rank_fixed(pa_row, 0, Bsep)].
              push_back(Bchild(r,0));
          else
            sbuf[pa->find_rank_fixed(pa_row-pa->dim_sep, 0, Bupd)].
              push_back(Bchild(r,0));
        }
      } else {
        for (int r=0; r<Bchild.lrows(); r++) {
          integer_t pa_row = I[Bchild.rowl2g(r)];
          if (pa_row < pa->dim_sep)
            sbuf[pa->find_rank(pa_row, 0, Bsep)].
              push_back(Bchild(r,0));
          else
            sbuf[pa->find_rank(pa_row-pa->dim_sep, 0, Bupd)].
              push_back(Bchild(r,0));
        }
      }
    }

    static void extract_b_copy_to_buffers
    (DistM_t& Bsep, DistM_t& Bupd, std::vector<std::vector<scalar_t>>& sbuf,
     std::function<int(integer_t)> ch_rank, std::vector<std::size_t>& I,
     int ch_proc_rows) {
      std::size_t pa_dim_sep = Bsep.rows();
      integer_t ch_dim_upd = I.size();
      integer_t ind_ptr = 0;
      for (int r=0; r<Bsep.lrows(); r++) {
        std::size_t gr = Bsep.rowl2g(r);
        while (ind_ptr < ch_dim_upd && I[ind_ptr] < gr) ind_ptr++;
        if (ind_ptr == ch_dim_upd) return;
        if (I[ind_ptr] >= pa_dim_sep) break;
        if (I[ind_ptr] != gr) continue;
        sbuf[ch_rank(ind_ptr)].push_back(Bsep(r,0));
      }
      for (int r=0; r<Bupd.lrows(); r++) {
        std::size_t gr = Bupd.rowl2g(r) + pa_dim_sep;
        while (ind_ptr < ch_dim_upd && I[ind_ptr] < gr) ind_ptr++;
        if (ind_ptr == ch_dim_upd) break;
        if (I[ind_ptr] != gr) continue;
        sbuf[ch_rank(ind_ptr)].push_back(Bupd(r,0));
      }
    }

    static void extract_copy_to_buffers
    (const DistM_t& F, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J,
     const std::vector<std::size_t>& oI,
     const std::vector<std::size_t>& oJ,
     const DistM_t& B, std::vector<std::vector<scalar_t>>& sbuf) {
      if (!F.active()) return;
      if (F.fixed()) {
        for (std::size_t c=0; c<J.size(); c++) {
          auto gc = J[c];
          if (F.colg2p_fixed(gc) != F.pcol()) continue;
          auto lc = F.colg2l_fixed(gc);
          for (std::size_t r=0; r<I.size(); r++) {
            auto gr = I[r];
            if (F.rowg2p_fixed(gr) == F.prow())
              sbuf[B.rank_fixed(oI[r],oJ[c])].
                push_back(F(F.rowg2l_fixed(gr),lc));
          }
        }
      } else {
        for (std::size_t c=0; c<J.size(); c++) {
          auto gc = J[c];
          if (F.colg2p(gc) != F.pcol()) continue;
          auto lc = F.colg2l(gc);
          for (std::size_t r=0; r<I.size(); r++) {
            auto gr = I[r];
            if (F.rowg2p(gr) == F.prow())
              sbuf[B.rank(oI[r],oJ[c])].push_back(F(F.rowg2l(gr),lc));
          }
        }
      }
    }

    static void extend_copy_to_buffers
    (const DistM_t& F, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J, const DistM_t& B,
     std::vector<std::vector<scalar_t>>& sbuf) {
      if (!F.active()) return;
      if (F.fixed()) {
        for (int c=0; c<F.lcols(); c++) {
          auto pcol = J[F.coll2g_fixed(c)];
          for (int r=0; r<F.lrows(); r++)
            sbuf[B.rank_fixed(I[F.rowl2g_fixed(r)],pcol)].push_back(F(r,c));
        }
      } else {
        for (int c=0; c<F.lcols(); c++) {
          auto pcol = J[F.coll2g(c)];
          for (int r=0; r<F.lrows(); r++)
            sbuf[B.rank(I[F.rowl2g(r)],pcol)].push_back(F(r,c));
        }
      }
    }

    static void extend_add_copy_from_buffers
    (DistM_t& F11, DistM_t& F12, DistM_t& F21, DistM_t& F22, scalar_t** pbuf,
     integer_t sep_begin, integer_t* pa_upd,
     integer_t* ch_upd, integer_t ch_dim_upd,
     std::function<int(integer_t,integer_t)> F22rank) {
      std::function<integer_t(integer_t)> sep_map =
        [&](integer_t i) { return i + sep_begin; };
      std::function<integer_t(integer_t)> upd_map =
        [&](integer_t i) { return pa_upd[i]; };
      copy_from_buffer(F11, pbuf, ch_upd, ch_dim_upd, F22rank,
                       sep_map, sep_map);
      copy_from_buffer(F12, pbuf, ch_upd, ch_dim_upd, F22rank,
                       sep_map, upd_map);
      copy_from_buffer(F21, pbuf, ch_upd, ch_dim_upd, F22rank,
                       upd_map, sep_map);
      copy_from_buffer(F22, pbuf, ch_upd, ch_dim_upd, F22rank,
                       upd_map, upd_map);
    }


    static void skinny_extend_add_copy_to_buffers
    (DistM_t& Schild, DistM_t& S, std::vector<std::vector<scalar_t>>& sbuf,
     FrontalMatrixHSSMPI<scalar_t,integer_t>* pa,
     std::vector<std::size_t>& I) {
      if (Schild.fixed()) {
        for (int c=0; c<Schild.lcols(); c++) {
          auto gc = Schild.coll2g_fixed(c);
          for (int r=0; r<Schild.lrows(); r++)
            sbuf[pa->find_rank_fixed(I[Schild.rowl2g_fixed(r)],gc,S)].
              push_back(Schild(r,c));
        }
      } else {
        for (int c=0; c<Schild.lcols(); c++) {
          auto gc = Schild.coll2g(c);
          for (int r=0; r<Schild.lrows(); r++)
            sbuf[pa->find_rank(I[Schild.rowl2g(r)],gc,S)].
              push_back(Schild(r,c));
        }
      }
    }

    static void skinny_extend_add_copy_from_buffers
    (DistM_t& F, scalar_t** pbuf, integer_t sep_begin, integer_t dim_sep,
     integer_t* pa_upd, integer_t* ch_upd, integer_t ch_dim_upd,
     std::function<int(integer_t,integer_t)> chSrank) {
      if (!F.active()) return;
      auto rowf2g = [&](integer_t r) {
        return (r < dim_sep) ? r+sep_begin : pa_upd[r-dim_sep];
      };
      if (F.fixed()) {
        for (int c=0; c<F.lcols(); c++) {
          auto gc = F.coll2g_fixed(c);
          integer_t upd_r = 0;
          for (int r=0; r<F.lrows(); r++) {
            auto fgr = rowf2g(F.rowl2g_fixed(r));
            while (upd_r < ch_dim_upd && ch_upd[upd_r] < fgr) upd_r++;
            if (upd_r == ch_dim_upd) break;
            if (ch_upd[upd_r] != fgr) continue;
            F(r,c) += *(pbuf[chSrank(upd_r, gc)]++);
          }
        }
      } else {
        for (int c=0; c<F.lcols(); c++) {
          auto gc = F.coll2g(c);
          integer_t upd_r = 0;
          for (int r=0; r<F.lrows(); r++) {
            auto fgr = rowf2g(F.rowl2g(r));
            while (upd_r < ch_dim_upd && ch_upd[upd_r] < fgr) upd_r++;
            if (upd_r == ch_dim_upd) break;
            if (ch_upd[upd_r] != fgr) continue;
            F(r,c) += *(pbuf[chSrank(upd_r,gc)]++);
          }
        }
      }
    }

    // TODO use skinny-extend-add
    static void extend_add_column_copy_from_buffers
    (DistM_t& Bsep, DistM_t& Bupd, std::vector<std::vector<scalar_t>>& buf,
     integer_t sep_begin, integer_t* pa_upd, integer_t* ch_upd,
     integer_t ch_dim_upd, std::function<int(integer_t,integer_t)> b_rank) {
      std::vector<scalar_t*> pbuf(buf.size());
      for (size_t p=0; p<buf.size(); p++) pbuf[p] = buf[p].data();

      std::function<integer_t(integer_t)> sep_map =
        [&](integer_t i) { return i + sep_begin;
      };
      std::function<integer_t(integer_t)> upd_map =
        [&](integer_t i) { return pa_upd[i];
      };
      copy_column_from_buffer(Bsep, pbuf, ch_upd, ch_dim_upd,
                              b_rank, sep_map);
      copy_column_from_buffer(Bupd, pbuf, ch_upd, ch_dim_upd,
                              b_rank, upd_map);
    }

    // write a more general skinny_extract / extract_rows
    static void extract_b_copy_from_buffers
    (DistM_t& F, std::vector<std::vector<scalar_t>>& buf,
     std::vector<std::size_t>& I, std::function<int(integer_t)> src_rank) {
      std::vector<scalar_t*> pbuf(buf.size());
      for (size_t p=0; p<buf.size(); p++) pbuf[p] = buf[p].data();
      if (F.fixed())
        for (int r=0; r<F.lrows(); r++)
          F(r,0) = *(pbuf[src_rank(I[F.rowl2g_fixed(r)])]++);
      else
        for (int r=0; r<F.lrows(); r++)
          F(r,0) = *(pbuf[src_rank(I[F.rowl2g(r)])]++);
    }

    static void copy_from_buffer
    (DistM_t& F, scalar_t** pbuf, integer_t* ch_upd, integer_t ch_dim_upd,
     std::function<int(integer_t,integer_t)> F22rank,
     std::function<integer_t(integer_t)> fr2g,
     std::function<integer_t(integer_t)> fc2g) {
      if (!F.active()) return;
      // TODO precompute indices??
      //   list of local rows/col that have corresponding global index
      // in front and corresponding child row similar to
      // compute_parent_indices and pass as input because this
      // function is called 4 times
      if (F.fixed()) {
        integer_t upd_c = 0;
        for (int c=0; c<F.lcols(); c++) {
          auto fgc = fc2g(F.coll2g_fixed(c));
          while (upd_c < ch_dim_upd && ch_upd[upd_c] < fgc) upd_c++;
          if (upd_c == ch_dim_upd) break;
          if (ch_upd[upd_c] != fgc) continue;
          integer_t upd_r = 0;
          for (int r=0; r<F.lrows(); r++) {
            auto fgr = fr2g(F.rowl2g_fixed(r));
            while (upd_r < ch_dim_upd && ch_upd[upd_r] < fgr) upd_r++;
            if (upd_r == ch_dim_upd) break;
            if (ch_upd[upd_r] != fgr) continue;
            F(r,c) += *(pbuf[F22rank(upd_r,upd_c)]++);
          }
        }
      } else {
        integer_t upd_c = 0;
        for (int c=0; c<F.lcols(); c++) {
          auto fgc = fc2g(F.coll2g(c));
          while (upd_c < ch_dim_upd && ch_upd[upd_c] < fgc) upd_c++;
          if (upd_c == ch_dim_upd) break;
          if (ch_upd[upd_c] != fgc) continue;
          integer_t upd_r = 0;
          for (int r=0; r<F.lrows(); r++) {
            auto fgr = fr2g(F.rowl2g(r));
            while (upd_r < ch_dim_upd && ch_upd[upd_r] < fgr) upd_r++;
            if (upd_r == ch_dim_upd) break;
            if (ch_upd[upd_r] != fgr) continue;
            F(r,c) += *(pbuf[F22rank(upd_r,upd_c)]++);
          }
        }
      }
    }

    static void copy_column_from_buffer
    (DistM_t& F, std::vector<scalar_t*>& pbuf, integer_t* ch_upd,
     integer_t ch_dim_upd, std::function<int(integer_t,integer_t)>
     b_child_rank, std::function<integer_t(integer_t)> f2g) {
      if (!F.active()) return;
      integer_t upd_r = 0;
      for (int r=0; r<F.lrows(); r++) {
        auto fgr = f2g(F.rowl2g(r));
        while (upd_r < ch_dim_upd && ch_upd[upd_r] < fgr) upd_r++;
        if (upd_r == ch_dim_upd) break;
        if (ch_upd[upd_r] != fgr) continue;
        F(r,0) += *(pbuf[b_child_rank(upd_r,0)]++);
      }
    }

    static void extract_copy_from_buffers
    (DistM_t& F, std::vector<std::size_t>& I, std::vector<std::size_t>& J,
     std::vector<std::size_t>& oI, std::vector<std::size_t>& oJ,
     const DistM_t& B, scalar_t** pbuf) {
      if (!F.active()) return;
      if (F.fixed()) {
        for (std::size_t c=0; c<oJ.size(); c++) {
          auto gc = oJ[c];
          if (F.colg2p_fixed(gc) != F.pcol()) continue;
          auto lc = F.colg2l_fixed(gc);
          for (std::size_t r=0; r<oI.size(); r++) {
            auto gr = oI[r];
            if (F.rowg2p_fixed(gr) == F.prow())
              F(F.rowg2l_fixed(gr),lc) += *(pbuf[B.rank_fixed(I[r], J[c])]++);
          }
        }
      } else {
        for (std::size_t c=0; c<oJ.size(); c++) {
          auto gc = oJ[c];
          if (F.colg2p(gc) != F.pcol()) continue;
          auto lc = F.colg2l(gc);
          for (std::size_t r=0; r<oI.size(); r++) {
            auto gr = oI[r];
            if (F.rowg2p(gr) == F.prow())
              F(F.rowg2l(gr),lc) += *(pbuf[B.rank(I[r], J[c])]++);
          }
        }
      }
    }

    static void extend_copy_from_buffers
    (DistM_t& F, const std::vector<std::size_t>& oI,
     const std::vector<std::size_t>& oJ, const DistM_t& B, scalar_t** pbuf) {
      if (!F.active()) return;
      if (F.fixed()) {
        for (std::size_t c=0; c<oJ.size(); c++) {
          auto gc = oJ[c];
          if (F.colg2p_fixed(gc) != F.pcol()) continue;
          auto lc = F.colg2l_fixed(gc);
          for (std::size_t r=0; r<oI.size(); r++) {
            auto gr = oI[r];
            if (F.rowg2p_fixed(gr) == F.prow())
              F(F.rowg2l_fixed(gr),lc) += *(pbuf[B.rank_fixed(r,c)]++);
          }
        }
      } else {
        for (std::size_t c=0; c<oJ.size(); c++) {
          auto gc = oJ[c];
          if (F.colg2p(gc) != F.pcol()) continue;
          auto lc = F.colg2l(gc);
          for (std::size_t r=0; r<oI.size(); r++) {
            auto gr = oI[r];
            if (F.rowg2p(gr) == F.prow())
              F(F.rowg2l(gr),lc) += *(pbuf[B.rank(r,c)]++);
          }
        }
      }
    }
  };


  template<typename scalar_t,typename integer_t> class ExtractFront {
    using CSM = CompressedSparseMatrix<scalar_t,integer_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
  public:
    static void extract_F11(DistM_t& F, CSM* A, integer_t sep_begin,
                            integer_t dim_sep) {
      if (!F.active()) return;
      F.zero();
      const auto CB = F.colblocks();
      const auto RB = F.rowblocks();
#pragma omp parallel for collapse(2)
      for (int cb=0; cb<CB; cb++) {
        for (int rb=0; rb<RB; rb++) {
          auto col = (F.pcol()+cb*F.pcols())*F.NB();
          auto row = (F.prow()+rb*F.prows())*F.MB();
          auto block = F.data() + cb*F.NB()*F.ld() + rb*F.MB();
          auto nr_cols = std::min(F.NB(), F.cols()-col);
          A->extract_F11_block(block, F.ld(), row+sep_begin,
                               std::min(F.MB(),F.rows()-row),
                               col+sep_begin, nr_cols);
        }
      }
    }

    static void extract_F12(DistM_t& F, CSM* A, integer_t upd_row_begin,
                            integer_t upd_col_begin,
                            integer_t dim_upd, integer_t* upd) {
      if (!F.active()) return;
      F.zero();
      const auto CB = F.colblocks();
      const auto RB = F.rowblocks();
#pragma omp parallel for collapse(2)
      for (int cb=0; cb<CB; cb++) {
        for (int rb=0; rb<RB; rb++) {
          auto col = (F.pcol()+cb*F.pcols())*F.NB();
          auto row = (F.prow()+rb*F.prows())*F.MB();
          auto block_upd = upd + (F.pcol()+cb*F.pcols())*F.NB();
          auto nr_cols = std::min(F.NB(), F.cols()-col);
          auto block = F.data() + cb*F.NB()*F.ld() + rb*F.MB();
          A->extract_F12_block(block, F.ld(), row+upd_row_begin,
                               std::min(F.MB(), F.rows()-row),
                               col+upd_col_begin, nr_cols, block_upd);
        }
      }
    }

    static void extract_F21(DistM_t& F, CSM* A, integer_t upd_row_begin,
                            integer_t upd_col_begin,
                            integer_t dim_upd, integer_t* upd) {
      if (!F.active()) return;
      F.zero();
      const auto CB = F.colblocks();
      const auto RB = F.rowblocks();
#pragma omp parallel for collapse(2)
      for (int cb=0; cb<CB; cb++) {
        for (int rb=0; rb<RB; rb++) {
          auto col = (F.pcol()+cb*F.pcols())*F.NB();
          auto row = (F.prow()+rb*F.prows())*F.MB();
          auto nr_cols = std::min(F.NB(), F.cols()-col);
          auto block = F.data() + cb*F.NB()*F.ld() + rb*F.MB();
          auto block_upd = upd + F.prow()*F.MB() + rb*F.prows()*F.MB();
          A->extract_F21_block(block, F.ld(), row+upd_row_begin,
                               std::min(F.MB(), F.rows()-row),
                               col+upd_col_begin, nr_cols, block_upd);
        }
      }
    }
  };

} // end namespace strumpack

#endif // EXTEND_ADD_HPP
