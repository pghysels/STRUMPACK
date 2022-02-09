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

#include "BLRExtendAdd.hpp"
#include "sparse/fronts/FrontalMatrixBLRMPI.hpp"

namespace strumpack {
  namespace BLR {

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::copy_to_buffers
    (const DistM_t& CB, VVS_t& sbuf, const FBLRMPI_t* pa, const VI_t& I) {
      if (!CB.active()) return;
      assert(CB.fixed());
      const auto lrows = CB.lrows();
      const auto lcols = CB.lcols();
      const auto pa_sep = pa->dim_sep();
      const auto nprows = pa->grid2d().nprows();
      std::unique_ptr<int[]> work(new int[lrows+lcols]);
      auto pr = work.get();
      auto pc = pr + CB.lrows();
      int r_upd, c_upd;
      for (r_upd=0; r_upd<lrows; r_upd++) {
        auto t = I[CB.rowl2g_fixed(r_upd)];
        if (t >= std::size_t(pa_sep)) break;
        pr[r_upd] = pa->sep_rg2p(t);
      }
      for (int r=r_upd; r<lrows; r++)
        pr[r] = pa->upd_rg2p(I[CB.rowl2g_fixed(r)]-pa_sep);
      for (c_upd=0; c_upd<lcols; c_upd++) {
        auto t = I[CB.coll2g_fixed(c_upd)];
        if (t >= std::size_t(pa_sep)) break;
        pc[c_upd] = pa->sep_cg2p(t) * nprows;
      }
      for (int c=c_upd; c<lcols; c++)
        pc[c] = pa->upd_cg2p(I[CB.coll2g_fixed(c)]-pa_sep) * nprows;
      { // reserve space for the send buffers
        VI_t cnt(sbuf.size());
        for (int c=0; c<lcols; c++)
          for (int r=0; r<lrows; r++)
            cnt[pr[r]+pc[c]]++;
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      for (int c=0; c<c_upd; c++) // F11
        for (int r=0, pcc=pc[c]; r<r_upd; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
      for (int c=c_upd; c<lcols; c++) // F12
        for (int r=0, pcc=pc[c]; r<r_upd; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
      for (int c=0; c<c_upd; c++) // F21
        for (int r=r_upd, pcc=pc[c]; r<lrows; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
      for (int c=c_upd; c<lcols; c++) // F22
        for (int r=r_upd, pcc=pc[c]; r<lrows; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
    }

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::copy_to_buffers
    (const BLRMPI_t& CB, VVS_t& sbuf, const FBLRMPI_t* pa, const VI_t& I) {
      if (!CB.active()) return;
      // TODO expand CB to dense, per block column
      // CB.to_dense();
      const int lrows = CB.lrows();
      const int lcols = CB.lcols();
      const int pa_sep = pa->dim_sep();
      const int nprows = pa->grid2d().nprows();
      // destination rank is pr[r] + pc[c]
      std::unique_ptr<int[]> work(new int[lrows+lcols]);
      auto pr = work.get();
      auto pc = pr + CB.lrows();
      int r_upd, c_upd;
      for (r_upd=0; r_upd<lrows; r_upd++) {
        auto t = I[CB.rl2g(r_upd)];
        if (t >= std::size_t(pa_sep)) break;
        pr[r_upd] = pa->sep_rg2p(t);
      }
      for (int r=r_upd; r<lrows; r++)
        pr[r] = pa->upd_rg2p(I[CB.rl2g(r)]-pa_sep);
      for (c_upd=0; c_upd<lcols; c_upd++) {
        auto t = I[CB.cl2g(c_upd)];
        if (t >= std::size_t(pa_sep)) break;
        pc[c_upd] = pa->sep_cg2p(t) * nprows;
      }
      for (int c=c_upd; c<lcols; c++)
        pc[c] = pa->upd_cg2p(I[CB.cl2g(c)]-pa_sep) * nprows;
      { // reserve space for the send buffers
        VI_t cnt(sbuf.size());
        for (int c=0; c<lcols; c++)
          for (int r=0; r<lrows; r++)
            cnt[pr[r]+pc[c]]++;
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      for (int c=0; c<lcols; c++) // F11 and F12
        for (int r=0, pcc=pc[c]; r<r_upd; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
      for (int c=0; c<lcols; c++) // F21 and F22
        for (int r=r_upd, pcc=pc[c]; r<lrows; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
    }

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::copy_to_buffers_col
    (const DistM_t& CB, VVS_t& sbuf, const FBLRMPI_t* pa, const VI_t& I,
     integer_t begin_col, integer_t end_col) {
      if (!CB.active()) return;
      assert(CB.fixed());
      const auto lrows = CB.lrows();
      const auto lcols = CB.lcols();
      const auto pa_sep = pa->dim_sep();
      const auto nprows = pa->grid2d().nprows();
      std::unique_ptr<int[]> work(new int[lrows+lcols]);
      auto pr = work.get();
      auto pc = pr + CB.lrows();
      int r_upd, c_upd;
      for (r_upd=0; r_upd<lrows; r_upd++) {
        auto t = I[CB.rowl2g_fixed(r_upd)];
        if (t >= std::size_t(pa_sep)) break;
        pr[r_upd] = pa->sep_rg2p(t);
      }
      for (int r=r_upd; r<lrows; r++)
        pr[r] = pa->upd_rg2p(I[CB.rowl2g_fixed(r)]-pa_sep);
      for (c_upd=0; c_upd<lcols; c_upd++) {
        auto t = I[CB.coll2g_fixed(c_upd)];
        if (t >= std::size_t(pa_sep)) break;
        if (t < std::size_t(begin_col) || t >= std::size_t(end_col))
          pc[c_upd] = -1;
        else
          pc[c_upd] = pa->sep_cg2p(t) * nprows;
      }
      for (int c=c_upd; c<lcols; c++) {
        auto t = I[CB.coll2g_fixed(c)];
        if (t < std::size_t(begin_col) || t >= std::size_t(end_col))
          pc[c] = -1;
        else
          pc[c] = pa->upd_cg2p(I[CB.coll2g_fixed(c)]-pa_sep) * nprows;
      }
      { // reserve space for the send buffers
        VI_t cnt(sbuf.size());
        for (int c=0; c<lcols; c++) {
          if (pc[c] == -1) continue;
          for (int r=0; r<lrows; r++)
            cnt[pr[r]+pc[c]]++;
        }
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      for (int c=0; c<lcols; c++) { // F11 and F12
        if (pc[c] == -1) continue;
        for (int r=0, pcc=pc[c]; r<r_upd; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
      }
      for (int c=0; c<lcols; c++) { // F21 and F22
        if (pc[c] == -1) continue;
        for (int r=r_upd, pcc=pc[c]; r<lrows; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
      }
    }

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::copy_to_buffers_col
    (const BLRMPI_t& CB, VVS_t& sbuf, const FBLRMPI_t* pa, const VI_t& I,
     integer_t begin_col, integer_t end_col) {
      if (!CB.active()) return;
      const int lrows = CB.lrows();
      const int lcols = CB.lcols();
      const int pa_sep = pa->dim_sep();
      const int nprows = pa->grid2d().nprows();
      int c_min = 0, c_max = 0;
      // destination rank is pr[r] + pc[c]
      std::unique_ptr<int[]> work(new int[lrows+lcols]);
      auto pr = work.get();
      auto pc = pr + CB.lrows();
      int r_upd, c_upd;
      for (r_upd=0; r_upd<lrows; r_upd++) {
        auto t = I[CB.rl2g(r_upd)];
        if (t >= std::size_t(pa_sep)) break;
        pr[r_upd] = pa->sep_rg2p(t);
      }
      for (int r=r_upd; r<lrows; r++)
        pr[r] = pa->upd_rg2p(I[CB.rl2g(r)]-pa_sep);
      for (c_upd=0; c_upd<lcols; c_upd++) {
        auto t = I[CB.cl2g(c_upd)];
        if (c_min == 0 && t >= std::size_t(pa_sep)) break;
        if (t < std::size_t(begin_col)) {
          c_min = c_upd+1;
          continue;
        }
        if (t >= std::size_t(end_col) || t >= std::size_t(pa_sep)) {
          c_max = c_upd;
          break;
        }
        pc[c_upd] = pa->sep_cg2p(t) * nprows;
      }
      if (c_max == 0 && c_upd == lcols) c_max = lcols;
      for (int c=c_upd; c<lcols; c++) {
        auto t = I[CB.cl2g(c)];
        if (t < std::size_t(begin_col)) {
          c_min = c+1;
          continue;
        }
        if (t >= std::size_t(end_col)) {
          c_max = c;
          break;
        }
        pc[c] = pa->upd_cg2p(I[CB.cl2g(c)]-pa_sep) * nprows;
        if (c == lcols-1) c_max = lcols;
      }
      { // reserve space for the send buffers
        VI_t cnt(sbuf.size());
        for (int c=c_min; c<c_max; c++) {
          for (int r=0; r<lrows; r++)
            cnt[pr[r]+pc[c]]++;
        }
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      const_cast<BLRMPI_t&>(CB).decompress_local_columns(c_min, c_max);
      if (r_upd)
        for (int c=c_min; c<c_max; c++) { // F11 and F12
          auto pcc = pc[c];
          auto lc = CB.cl2l_[c];
          auto tc = CB.cl2t_[c];
          auto trmax = CB.rl2t_[r_upd-1];
          for (std::size_t tr=0, r=0; tr<=trmax; tr++) {
            auto& tD = CB.ltile_dense(tr, tc).D();
            auto lrmax = std::min(r_upd-r, tD.rows());
            for (std::size_t lr=0; lr<lrmax; lr++)
              sbuf[pr[r++]+pcc].push_back(tD(lr, lc));
          }
        }
      if (r_upd < lrows)
        for (int c=c_min; c<c_max; c++) { // F21 and F22
          auto pcc = pc[c];
          auto lc = CB.cl2l_[c];
          auto tc = CB.cl2t_[c];
          auto trmax = CB.rl2t_[lrows-1];
          for (std::size_t tr=CB.rl2t_[r_upd], r=r_upd; tr<=trmax; tr++) {
            auto& tD = CB.ltile_dense(tr, tc).D();
            auto lrmin = CB.rl2l_[r];
            auto lrmax = tD.rows();
            for (std::size_t lr=lrmin; lr<lrmax; lr++)
              sbuf[pr[r++]+pcc].push_back(tD(lr, lc));
          }
        }
      const_cast<BLRMPI_t&>(CB).remove_tiles_before_local_column(c_min, c_max);
    }

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::copy_from_buffers
    (BLRMPI_t& F11, BLRMPI_t& F12, BLRMPI_t& F21, BLRMPI_t& F22,
     scalar_t** pbuf, const FBLRMPI_t* pa, const FBLRMPI_t* ch) {
      assert(pa != nullptr);
      if (!pa->grid2d().active()) return;
      const auto ch_dim_upd = ch->dim_upd();
      const int chprows = ch->grid2d().nprows();
      const auto& ch_upd = ch->upd();
      const auto& pa_upd = pa->upd();
      const auto pa_sep = pa->sep_begin();
      std::unique_ptr<int[]> iwork
        (new int[2*F11.lrows()+2*F11.lcols()+
                 2*F22.lrows()+2*F22.lcols()]);
      auto upd_r_1 = iwork.get();
      auto upd_c_1 = upd_r_1 + F11.lrows();
      auto upd_r_2 = upd_c_1 + F11.lcols();
      auto upd_c_2 = upd_r_2 + F22.lrows();
      auto r_1 = upd_c_2 + F22.lcols();
      auto c_1 = r_1 + F11.lrows();
      auto r_2 = c_1 + F11.lcols();
      auto c_2 = r_2 + F22.lrows();
      int r_max_1 = 0, r_max_2 = 0,
        c_max_1 = 0, c_max_2 = 0;
      for (int r=0, ur=0; r<int(F11.lrows()); r++) {
        integer_t fgr = F11.rl2g(r) + pa_sep;
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_1[r_max_1] = r;
        upd_r_1[r_max_1++] = ch->upd_rg2p(ur);
      }
      for (int c=0, uc=0; c<int(F11.lcols()); c++) {
        integer_t fgc = F11.cl2g(c) + pa_sep;
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        c_1[c_max_1] = c;
        upd_c_1[c_max_1++] = ch->upd_cg2p(uc) * chprows;
      }
      for (int r=0, ur=0; r<int(F22.lrows()); r++) {
        auto fgr = pa_upd[F22.rl2g(r)];
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_2[r_max_2] = r;
        upd_r_2[r_max_2++] = ch->upd_rg2p(ur);
      }
      for (int c=0, uc=0; c<int(F22.lcols()); c++) {
        auto fgc = pa_upd[F22.cl2g(c)];
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        c_2[c_max_2] = c;
        upd_c_2[c_max_2++] = ch->upd_cg2p(uc) * chprows;
      }
      for (int c=0; c<c_max_1; c++)
        for (int r=0, cc=c_1[c], ucc=upd_c_1[c]; r<r_max_1; r++)
          F11(r_1[r],cc) += *(pbuf[upd_r_1[r]+ucc]++);
      for (int c=0; c<c_max_2; c++)
        for (int r=0, cc=c_2[c], ucc=upd_c_2[c]; r<r_max_1; r++)
          F12(r_1[r],cc) += *(pbuf[upd_r_1[r]+ucc]++);
      for (int c=0; c<c_max_1; c++)
        for (int r=0, cc=c_1[c], ucc=upd_c_1[c]; r<r_max_2; r++)
          F21(r_2[r],cc) += *(pbuf[upd_r_2[r]+ucc]++);
      for (int c=0; c<c_max_2; c++)
        for (int r=0, cc=c_2[c], ucc=upd_c_2[c]; r<r_max_2; r++)
          F22(r_2[r],cc) += *(pbuf[upd_r_2[r]+ucc]++);
    }

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::copy_from_buffers_col
    (BLRMPI_t& F11, BLRMPI_t& F12, BLRMPI_t& F21, BLRMPI_t& F22,
     scalar_t** pbuf, const FBLRMPI_t* pa, const FBLRMPI_t* ch,
     integer_t begin_col, integer_t end_col) {
      assert(pa != nullptr);
      if (!pa->grid2d().active()) return;
      const auto ch_dim_upd = ch->dim_upd();
      const int chprows = ch->grid2d().nprows();
      const auto& ch_upd = ch->upd();
      const auto& pa_upd = pa->upd();
      const auto pa_sep = pa->sep_begin();
      std::unique_ptr<int[]> iwork
        (new int[2*F11.lrows()+2*F11.lcols()+
                 2*F22.lrows()+2*F22.lcols()]);
      auto upd_r_1 = iwork.get();
      auto upd_c_1 = upd_r_1 + F11.lrows();
      auto upd_r_2 = upd_c_1 + F11.lcols();
      auto upd_c_2 = upd_r_2 + F22.lrows();
      auto r_1 = upd_c_2 + F22.lcols();
      auto c_1 = r_1 + F11.lrows();
      auto r_2 = c_1 + F11.lcols();
      auto c_2 = r_2 + F22.lrows();
      int r_max_1 = 0, r_max_2 = 0, c_max_1 = 0, c_max_2 = 0;
      for (int r=0, ur=0; r<int(F11.lrows()); r++) {
        integer_t fgr = F11.rl2g(r) + pa_sep;
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_1[r_max_1] = r;
        upd_r_1[r_max_1++] = ch->upd_rg2p(ur);
      }
      for (int c=0, uc=0; c<int(F11.lcols()); c++) {
        integer_t fgc = F11.cl2g(c) + pa_sep;
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        if (fgc - pa_sep < begin_col || fgc - pa_sep >= end_col) {
          c_1[c_max_1] = -1;
          upd_c_1[c_max_1++] = -1;
        } else {
          c_1[c_max_1] = c;
          upd_c_1[c_max_1++] = ch->upd_cg2p(uc) * chprows;
        }
      }
      for (int r=0, ur=0; r<int(F22.lrows()); r++) {
        auto fgr = pa_upd[F22.rl2g(r)];
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_2[r_max_2] = r;
        upd_r_2[r_max_2++] = ch->upd_rg2p(ur);
      }
      for (int c=0, uc=0; c<int(F22.lcols()); c++) {
        auto gc22 = F22.cl2g(c);
        auto fgc = pa_upd[gc22];
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        if (gc22 + F11.cols() < std::size_t(begin_col) ||
            gc22 + F11.cols() >= std::size_t(end_col)) {
          c_2[c_max_2] = -1;
          upd_c_2[c_max_2++] = -1;
        } else {
          c_2[c_max_2] = c;
          upd_c_2[c_max_2++] = ch->upd_cg2p(uc) * chprows;
        }
      }
      for (int c=0; c<c_max_1; c++) {
        if (c_1[c] == -1) continue;
        for (int r=0, cc=c_1[c], ucc=upd_c_1[c]; r<r_max_1; r++)
          F11(r_1[r],cc) += *(pbuf[upd_r_1[r]+ucc]++);
      }
      for (int c=0; c<c_max_2; c++) {
        if (c_2[c] == -1) continue;
        for (int r=0, cc=c_2[c], ucc=upd_c_2[c]; r<r_max_1; r++)
          F12(r_1[r],cc) += *(pbuf[upd_r_1[r]+ucc]++);
      }
      for (int c=0; c<c_max_1; c++) {
        if (c_1[c] == -1) continue;
        for (int r=0, cc=c_1[c], ucc=upd_c_1[c]; r<r_max_2; r++)
          F21(r_2[r],cc) += *(pbuf[upd_r_2[r]+ucc]++);
      }
      for (int c=0; c<c_max_2; c++) {
        if (c_2[c] == -1) continue;
        for (int r=0, cc=c_2[c], ucc=upd_c_2[c]; r<r_max_2; r++)
          F22(r_2[r],cc) += *(pbuf[upd_r_2[r]+ucc]++);
      }
    }

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::copy_from_buffers
    (BLRMPI_t& F11, BLRMPI_t& F12, BLRMPI_t& F21, BLRMPI_t& F22,
     scalar_t** pbuf, const FBLRMPI_t* pa, const FMPI_t* ch) {
      if (!pa->grid2d().active()) return;
      const auto ch_dim_upd = ch->dim_upd();
      const auto& ch_upd = ch->upd();
      const auto& pa_upd = pa->upd();
      const auto pa_sep = pa->sep_begin();
      const auto prows = ch->grid()->nprows();
      const auto pcols = ch->grid()->npcols();
      const auto B = DistM_t::default_MB;
      // source rank is
      //  ((r / B) % prows) + ((c / B) % pcols) * prows
      // where r,c is the coordinate in the F22 block of the child
      std::unique_ptr<int[]> iwork
        (new int[2*F11.lrows()+2*F11.lcols()+
                 2*F22.lrows()+2*F22.lcols()]);
      auto upd_r_1 = iwork.get();
      auto upd_c_1 = upd_r_1 + F11.lrows();
      auto upd_r_2 = upd_c_1 + F11.lcols();
      auto upd_c_2 = upd_r_2 + F22.lrows();
      auto r_1 = upd_c_2 + F22.lcols();
      auto c_1 = r_1 + F11.lrows();
      auto r_2 = c_1 + F11.lcols();
      auto c_2 = r_2 + F22.lrows();
      integer_t r_max_1 = 0, r_max_2 = 0,
        c_max_1 = 0, c_max_2 = 0;
      for (int r=0, ur=0; r<int(F11.lrows()); r++) {
        integer_t fgr = F11.rl2g(r) + pa_sep;
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_1[r_max_1] = r;
        upd_r_1[r_max_1++] = (ur / B) % prows;
      }
      for (int c=0, uc=0; c<int(F11.lcols()); c++) {
        integer_t fgc = F11.cl2g(c) + pa_sep;
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        c_1[c_max_1] = c;
        upd_c_1[c_max_1++] = ((uc / B) % pcols) * prows;
      }
      for (int r=0, ur=0; r<int(F22.lrows()); r++) {
        integer_t fgr = pa_upd[F22.rl2g(r)];
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_2[r_max_2] = r;
        upd_r_2[r_max_2++] = (ur / B) % prows;
      }
      for (int c=0, uc=0; c<int(F22.lcols()); c++) {
        auto fgc = pa_upd[F22.cl2g(c)];
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        c_2[c_max_2] = c;
        upd_c_2[c_max_2++] = ((uc / B) % pcols) * prows;
      }
      for (int c=0; c<c_max_1; c++)
        for (int r=0, cc=c_1[c], ucc=upd_c_1[c]; r<r_max_1; r++)
          F11(r_1[r],cc) += *(pbuf[upd_r_1[r]+ucc]++);
      for (int c=0; c<c_max_2; c++)
        for (int r=0, cc=c_2[c], ucc=upd_c_2[c]; r<r_max_1; r++)
          F12(r_1[r],cc) += *(pbuf[upd_r_1[r]+ucc]++);
      for (int c=0; c<c_max_1; c++)
        for (int r=0, cc=c_1[c], ucc=upd_c_1[c]; r<r_max_2; r++)
          F21(r_2[r],cc) += *(pbuf[upd_r_2[r]+ucc]++);
      for (int c=0; c<c_max_2; c++)
        for (int r=0, cc=c_2[c], ucc=upd_c_2[c]; r<r_max_2; r++)
          F22(r_2[r],cc) += *(pbuf[upd_r_2[r]+ucc]++);
    }

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::copy_from_buffers_col
    (BLRMPI_t& F11, BLRMPI_t& F12, BLRMPI_t& F21, BLRMPI_t& F22,
     scalar_t** pbuf, const FBLRMPI_t* pa, const FMPI_t* ch,
     integer_t begin_col, integer_t end_col) {
      if (!pa->grid2d().active()) return;
      const auto ch_dim_upd = ch->dim_upd();
      const auto& ch_upd = ch->upd();
      const auto& pa_upd = pa->upd();
      const auto pa_sep = pa->sep_begin();
      const auto prows = ch->grid()->nprows();
      const auto pcols = ch->grid()->npcols();
      const auto B = DistM_t::default_MB;
      // source rank is
      //  ((r / B) % prows) + ((c / B) % pcols) * prows
      // where r,c is the coordinate in the F22 block of the child
      std::unique_ptr<int[]> iwork
        (new int[2*F11.lrows()+2*F11.lcols()+
                 2*F22.lrows()+2*F22.lcols()]);
      auto upd_r_1 = iwork.get();
      auto upd_c_1 = upd_r_1 + F11.lrows();
      auto upd_r_2 = upd_c_1 + F11.lcols();
      auto upd_c_2 = upd_r_2 + F22.lrows();
      auto r_1 = upd_c_2 + F22.lcols();
      auto c_1 = r_1 + F11.lrows();
      auto r_2 = c_1 + F11.lcols();
      auto c_2 = r_2 + F22.lrows();
      integer_t r_max_1 = 0, r_max_2 = 0, c_max_1 = 0, c_max_2 = 0;
      for (int r=0, ur=0; r<int(F11.lrows()); r++) {
        integer_t fgr = F11.rl2g(r) + pa_sep;
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_1[r_max_1] = r;
        upd_r_1[r_max_1++] = (ur / B) % prows;
      }
      for (int c=0, uc=0; c<int(F11.lcols()); c++) {
        integer_t fgc = F11.cl2g(c) + pa_sep;
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        if (fgc - pa_sep < begin_col || fgc - pa_sep >= end_col) {
          c_1[c_max_1] = -1;
          upd_c_1[c_max_1++] = -1;
        } else {
          c_1[c_max_1] = c;
          upd_c_1[c_max_1++] = ((uc / B) % pcols) * prows;
        }
      }
      for (int r=0, ur=0; r<int(F22.lrows()); r++) {
        integer_t fgr = pa_upd[F22.rl2g(r)];
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_2[r_max_2] = r;
        upd_r_2[r_max_2++] = (ur / B) % prows;
      }
      for (int c=0, uc=0; c<int(F22.lcols()); c++) {
        auto gc22 = F22.cl2g(c);
        auto fgc = pa_upd[gc22];
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        if (gc22 + F11.cols() < std::size_t(begin_col) ||
            gc22 + F11.cols() >= std::size_t(end_col)) {
          c_2[c_max_2] = -1;
          upd_c_2[c_max_2++] = -1;
        } else {
          c_2[c_max_2] = c;
          upd_c_2[c_max_2++] = ((uc / B) % pcols) * prows;
        }
      }
      for (int c=0; c<c_max_1; c++) {
        if (c_1[c] == -1) continue;
        for (int r=0, cc=c_1[c], ucc=upd_c_1[c]; r<r_max_1; r++)
          F11(r_1[r],cc) += *(pbuf[upd_r_1[r]+ucc]++);
      }
      for (int c=0; c<c_max_2; c++) {
        if (c_2[c] == -1) continue;
        for (int r=0, cc=c_2[c], ucc=upd_c_2[c]; r<r_max_1; r++)
          F12(r_1[r],cc) += *(pbuf[upd_r_1[r]+ucc]++);
      }
      for (int c=0; c<c_max_1; c++) {
        if (c_1[c] == -1) continue;
        for (int r=0, cc=c_1[c], ucc=upd_c_1[c]; r<r_max_2; r++)
          F21(r_2[r],cc) += *(pbuf[upd_r_2[r]+ucc]++);
      }
      for (int c=0; c<c_max_2; c++) {
        if (c_2[c] == -1) continue;
        for (int r=0, cc=c_2[c], ucc=upd_c_2[c]; r<r_max_2; r++)
          F22(r_2[r],cc) += *(pbuf[upd_r_2[r]+ucc]++);
      }
    }

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::seq_copy_to_buffers
    (const DenseM_t& CB, VVS_t& sbuf, const FBLRMPI_t* pa, const F_t* ch) {
      std::size_t u2s;
      const auto I = ch->upd_to_parent(static_cast<const F_t*>(pa), u2s);
      const std::size_t du = ch->dim_upd();
      const std::size_t ds = pa->dim_sep();
      const int nprows = pa->grid2d().nprows();
      std::unique_ptr<int[]> work(new int[CB.rows()+CB.cols()]);
      const auto pr = work.get();
      const auto pc = pr + CB.rows();
      for (std::size_t i=0; i<u2s; i++) {
        auto Ii = I[i];
        pr[i] = pa->sep_rg2p(Ii); // can be optimized
        pc[i] = pa->sep_cg2p(Ii) * nprows;
      }
      for (std::size_t i=u2s; i<du; i++) {
        auto Ii = I[i] - ds;
        pr[i] = pa->upd_rg2p(Ii);
        pc[i] = pa->upd_cg2p(Ii) * nprows;
      }
      { // reserve space for the send buffers
        VI_t cnt(sbuf.size());
        for (std::size_t c=0; c<du; c++)
          for (std::size_t r=0; r<du; r++)
            cnt[pr[r]+pc[c]]++;
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      for (std::size_t c=0; c<u2s; c++) // F11
        for (std::size_t r=0, pcc=pc[c]; r<u2s; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
      for (std::size_t c=u2s; c<du; c++) // F12
        for (std::size_t r=0, pcc=pc[c]; r<u2s; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
      for (std::size_t c=0; c<u2s; c++) // F21
        for (std::size_t r=u2s, pcc=pc[c]; r<du; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
      for (std::size_t c=u2s; c<du; c++) // F22
        for (std::size_t r=u2s, pcc=pc[c]; r<du; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
    }

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::seq_copy_to_buffers
    (const BLR_t& CB, VVS_t& sbuf, const FBLRMPI_t* pa, const F_t* ch) {
      auto F = CB.dense();
      seq_copy_to_buffers(F, sbuf, pa, ch);
    }

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::seq_copy_to_buffers_col
    (const DenseM_t& CB, VVS_t& sbuf, const FBLRMPI_t* pa, const F_t* ch,
     integer_t begin_col, integer_t end_col) {
      std::size_t u2s;
      const auto I = ch->upd_to_parent(static_cast<const F_t*>(pa), u2s);
      const std::size_t du = ch->dim_upd();
      const std::size_t ds = pa->dim_sep();
      const int nprows = pa->grid2d().nprows();
      std::unique_ptr<int[]> work(new int[CB.rows()+CB.cols()]);
      const auto pr = work.get();
      const auto pc = pr + CB.rows();
      for (std::size_t i=0; i<u2s; i++) {
        auto Ii = I[i];
        pr[i] = pa->sep_rg2p(Ii); // can be optimized
        if (Ii < std::size_t(begin_col) || Ii >= std::size_t(end_col))
          pc[i] = -1;
        else
          pc[i] = pa->sep_cg2p(Ii) * nprows;
      }
      for (std::size_t i=u2s; i<du; i++) {
        auto Ii = I[i] - ds;
        pr[i] = pa->upd_rg2p(Ii);
        if (Ii + ds < std::size_t(begin_col) ||
            Ii + ds >= std::size_t(end_col))
          pc[i] = -1;
        else
          pc[i] = pa->upd_cg2p(Ii) * nprows;
      }
      { // reserve space for the send buffers
        VI_t cnt(sbuf.size());
        for (std::size_t c=0; c<du; c++) {
          if (pc[c] == -1) continue;
          for (std::size_t r=0; r<du; r++)
            cnt[pr[r]+pc[c]]++;
        }
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      for (std::size_t c=0; c<u2s; c++) { // F11
        if (pc[c] == -1) continue;
        for (std::size_t r=0, pcc=pc[c]; r<u2s; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
      }
      for (std::size_t c=u2s; c<du; c++) { // F12
        if (pc[c] == -1) continue;
        for (std::size_t r=0, pcc=pc[c]; r<u2s; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
      }
      for (std::size_t c=0; c<u2s; c++) { // F21
        if (pc[c] == -1) continue;
        for (std::size_t r=u2s, pcc=pc[c]; r<du; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
      }
      for (std::size_t c=u2s; c<du; c++) { // F22
        if (pc[c] == -1) continue;
        for (std::size_t r=u2s, pcc=pc[c]; r<du; r++)
          sbuf[pr[r]+pcc].push_back(CB(r,c));
      }
    }

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::blrseq_copy_to_buffers_col
    (const BLR_t& CB, VVS_t& sbuf, const FBLRMPI_t* pa, const F_t* ch,
     integer_t begin_col, integer_t end_col,
     const BLROptions<scalar_t>& opts) {
      std::size_t u2s;
      const auto I = ch->upd_to_parent(static_cast<const F_t*>(pa), u2s);
      const std::size_t du = ch->dim_upd();
      const std::size_t ds = pa->dim_sep();
      const int nprows = pa->grid2d().nprows();
      int c_min = 0, c_max = 0;
      std::unique_ptr<int[]> work(new int[CB.rows()+CB.cols()]);
      const auto pr = work.get();
      const auto pc = pr + CB.rows();
      for (std::size_t i=0; i<u2s; i++) {
        auto Ii = I[i];
        pr[i] = pa->sep_rg2p(Ii);
      }
      for (std::size_t i=u2s; i<du; i++) {
        auto Ii = I[i] - ds;
        pr[i] = pa->upd_rg2p(Ii);
      }
      for (std::size_t i=0; i<u2s; i++) {
        auto Ii = I[i];
        if (Ii < std::size_t(begin_col)) {
          c_min = i+1;
          continue;
        }
        if (Ii >= std::size_t(end_col)) {
          c_max = i;
          break;
        }
        pc[i] = pa->sep_cg2p(Ii) * nprows;
      }
      if (c_max == 0 && u2s == du) c_max = du;
      for (std::size_t i=u2s; i<du; i++) {
        auto Ii = I[i] - ds;
        if (Ii + ds < std::size_t(begin_col)) {
          c_min = i+1;
          continue;
        }
        if (Ii + ds >= std::size_t(end_col)) {
          if (c_max == 0) c_max = i;
          break;
        }
        pc[i] = pa->upd_cg2p(Ii) * nprows;
        if (i == du-1) c_max = du;
      }
      { // reserve space for the send buffers
        VI_t cnt(sbuf.size());
        for (int c=c_min; c<c_max; c++)
          for (std::size_t r=0; r<du; r++)
            cnt[pr[r]+pc[c]]++;
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      if (c_max > 0 && (opts.BLR_factor_algorithm() == BLR::BLRFactorAlgorithm::COLWISE))
        const_cast<BLR_t&>(CB).decompress_local_columns(c_min, c_max);
      if (u2s)
        for (int c=c_min; c<c_max; c++) { // F11 and F12
          auto pcc=pc[c];
          auto lc = CB.cl2l_[c];
          auto tc = CB.cg2t(c);
          auto trmax = CB.nbrows_;
          for (std::size_t tr=0, r=0; tr<trmax; tr++) {
            auto& tD = CB.tile_dense(tr, tc).D();
            auto lrmax = std::min(u2s-r, tD.rows());
            for (std::size_t lr=0; lr<lrmax; lr++)
              sbuf[pr[r++]+pcc].push_back(tD(lr,lc));
          }
        }
      if (u2s < du)
        for (int c=c_min; c<c_max; c++) { // F21 and F22
          auto pcc=pc[c];
          auto lc = CB.cl2l_[c];
          auto tc = CB.cg2t(c);
          auto trmax = CB.nbrows_;
          for (std::size_t tr=CB.rg2t(u2s), r=u2s; tr<trmax; tr++) {
            auto& tD = CB.tile_dense(tr, tc).D();
            auto lrmin = CB.rl2l_[r];
            auto lrmax = tD.rows();
            for (std::size_t lr=lrmin; lr<lrmax; lr++)
              sbuf[pr[r++]+pcc].push_back(tD(lr,lc));
          }
        }
      const_cast<BLR_t&>(CB).remove_tiles_before_local_column(c_min, c_max);
    }

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::seq_copy_from_buffers
    (BLRMPI_t& F11, BLRMPI_t& F12, BLRMPI_t& F21, BLRMPI_t& F22,
     scalar_t*& pbuf, const FBLRMPI_t* pa, const F_t* ch) {
      if (!(F11.active() || F22.active())) return;
      const int ch_dim_upd = ch->dim_upd();
      const auto ch_upd = ch->upd();
      const auto pa_upd = pa->upd();
      const int pa_sep = pa->sep_begin();
      std::unique_ptr<int[]> work
        (new int[F11.lrows()+F11.lcols()+F22.lrows()+F22.lcols()]);
      auto r_1 = work.get();
      auto c_1 = r_1 + F11.lrows();
      auto r_2 = c_1 + F11.lcols();
      auto c_2 = r_2 + F22.lrows();
      int r_max_1 = 0, r_max_2 = 0, c_max_1 = 0, c_max_2 = 0;
      for (int r=0, ur=0; r<int(F11.lrows()); r++) {
        int fgr = F11.rl2g(r) + pa_sep;
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_1[r_max_1++] = r;
      }
      for (int c=0, uc=0; c<int(F11.lcols()); c++) {
        int fgc = F11.cl2g(c) + pa_sep;
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        c_1[c_max_1++] = c;
      }
      for (int r=0, ur=0; r<int(F22.lrows()); r++) {
        int fgr = pa_upd[F22.rl2g(r)];
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_2[r_max_2++] = r;
      }
      for (int c=0, uc=0; c<int(F22.lcols()); c++) {
        int fgc = pa_upd[F22.cl2g(c)];
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        c_2[c_max_2++] = c;
      }
      for (int c=0; c<c_max_1; c++)
        for (int r=0, cc=c_1[c]; r<r_max_1; r++)
          F11(r_1[r],cc) += *(pbuf++);
      for (int c=0; c<c_max_2; c++)
        for (int r=0, cc=c_2[c]; r<r_max_1; r++)
          F12(r_1[r],cc) += *(pbuf++);
      for (int c=0; c<c_max_1; c++)
        for (int r=0, cc=c_1[c]; r<r_max_2; r++)
          F21(r_2[r],cc) += *(pbuf++);
      for (int c=0; c<c_max_2; c++)
        for (int r=0, cc=c_2[c]; r<r_max_2; r++)
          F22(r_2[r],cc) += *(pbuf++);
    }

    template<typename scalar_t,typename integer_t> void
    BLRExtendAdd<scalar_t,integer_t>::seq_copy_from_buffers_col
    (BLRMPI_t& F11, BLRMPI_t& F12, BLRMPI_t& F21, BLRMPI_t& F22,
     scalar_t*& pbuf, const FBLRMPI_t* pa, const F_t* ch,
     integer_t begin_col, integer_t end_col) {
      if (!(F11.active())) return;
      const int ch_dim_upd = ch->dim_upd();
      const auto ch_upd = ch->upd();
      const auto pa_upd = pa->upd();
      const int pa_sep = pa->sep_begin();
      std::unique_ptr<int[]> work
        (new int[F11.lrows()+F11.lcols()+F22.lrows()+F22.lcols()]);
      auto r_1 = work.get();
      auto c_1 = r_1 + F11.lrows();
      auto r_2 = c_1 + F11.lcols();
      auto c_2 = r_2 + F22.lrows();
      int r_max_1 = 0, r_max_2 = 0, c_max_1 = 0, c_max_2 = 0;
      for (int r=0, ur=0; r<int(F11.lrows()); r++) {
        int fgr = F11.rl2g(r) + pa_sep;
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_1[r_max_1++] = r;
      }
      for (int c=0, uc=0; c<int(F11.lcols()); c++) {
        int fgc = F11.cl2g(c) + pa_sep;
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        if (fgc - pa_sep < begin_col || fgc - pa_sep >= end_col)
          c_1[c_max_1++] = -1;
        else
          c_1[c_max_1++] = c;
      }
      for (int r=0, ur=0; r<int(F22.lrows()); r++) {
        int fgr = pa_upd[F22.rl2g(r)];
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_2[r_max_2++] = r;
      }
      for (int c=0, uc=0; c<int(F22.lcols()); c++) {
        auto gc22 = F22.cl2g(c);
        int fgc = pa_upd[gc22];
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        if (gc22 + F11.cols() < std::size_t(begin_col) ||
            gc22 + F11.cols() >= std::size_t(end_col))
          c_2[c_max_2++] = -1;
        else
          c_2[c_max_2++] = c;
      }
      for (int c=0; c<c_max_1; c++) {
        if (c_1[c] == -1) continue;
        for (int r=0, cc=c_1[c]; r<r_max_1; r++)
          F11(r_1[r],cc) += *(pbuf++);
      }
      for (int c=0; c<c_max_2; c++) {
        if (c_2[c] == -1) continue;
        for (int r=0, cc=c_2[c]; r<r_max_1; r++)
            F12(r_1[r],cc) += *(pbuf++);
      }
      for (int c=0; c<c_max_1; c++) {
        if (c_1[c] == -1) continue;
        for (int r=0, cc=c_1[c]; r<r_max_2; r++)
          F21(r_2[r],cc) += *(pbuf++);
      }
      for (int c=0; c<c_max_2; c++) {
        if (c_2[c] == -1) continue;
        for (int r=0, cc=c_2[c]; r<r_max_2; r++)
          F22(r_2[r],cc) += *(pbuf++);
      }
    }

    // explicit template instantiation
    template class BLRExtendAdd<float,int>;
    template class BLRExtendAdd<double,int>;
    template class BLRExtendAdd<std::complex<float>,int>;
    template class BLRExtendAdd<std::complex<double>,int>;

    template class BLRExtendAdd<float,long int>;
    template class BLRExtendAdd<double,long int>;
    template class BLRExtendAdd<std::complex<float>,long int>;
    template class BLRExtendAdd<std::complex<double>,long int>;

    template class BLRExtendAdd<float,long long int>;
    template class BLRExtendAdd<double,long long int>;
    template class BLRExtendAdd<std::complex<float>,long long int>;
    template class BLRExtendAdd<std::complex<double>,long long int>;

  } // end namespace BLR
} // end namespace strumpack
