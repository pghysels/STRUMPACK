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

#include "dense/DistributedMatrix.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixMPI;
  template<typename scalar_t,typename integer_t> class FrontalMatrixHSSMPI;
  template<typename scalar_t,typename integer_t> class FrontalMatrixDenseMPI;

  template<typename scalar_t,typename integer_t> class ExtendAdd {
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;

  public:

    static void extend_add_copy_to_buffers
    (const DistM_t& CB, const DistM_t& F11, const DistM_t& F12,
     const DistM_t& F21, const DistM_t& F22,
     std::vector<std::vector<scalar_t>>& sbuf,
     const FrontalMatrixDenseMPI<scalar_t,integer_t>* pa,
     const std::vector<std::size_t>& I) {
      if (!CB.active()) return;
      assert(CB.fixed());
      const auto lrows = CB.lrows();
      const auto lcols = CB.lcols();
      const auto pa_sep = pa->dim_sep;
      const auto prows = pa->proc_rows;
      const auto pcols = pa->proc_cols;
      const auto B = DistM_t::default_MB;
      // destination rank is:
      //  ((r / B) % prows) + ((c / B) % pcols) * prows
      //  = pr[r] + pc[c]
      auto pr = new int[CB.lrows()+CB.lcols()];
      auto pc = pr + CB.lrows();
      int r_upd, c_upd;
      for (r_upd=0; r_upd<lrows; r_upd++) {
        auto t = I[CB.rowl2g_fixed(r_upd)];
        if (t >= std::size_t(pa_sep)) break;
        pr[r_upd] = (t / B) % prows;
      }
      for (int r=r_upd; r<lrows; r++)
        pr[r] = ((I[CB.rowl2g_fixed(r)]-pa_sep) / B) % prows;
      for (c_upd=0; c_upd<lcols; c_upd++) {
        auto t = I[CB.coll2g_fixed(c_upd)];
        if (t >= std::size_t(pa_sep)) break;
        pc[c_upd] = ((t / B) % pcols) * prows;
      }
      for (int c=c_upd; c<lcols; c++)
        pc[c] = (((I[CB.coll2g_fixed(c)]-pa_sep) / B) % pcols) * prows;
      { // reserve space for the send buffers
        std::vector<std::size_t> cnt(sbuf.size());
        for (int c=0; c<lcols; c++)
          for (int r=0; r<lrows; r++)
            cnt[pr[r]+pc[c]]++;
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      for (int c=0; c<c_upd; c++) // F11
        for (int r=0; r<r_upd; r++)
          sbuf[pr[r]+pc[c]].push_back(CB(r,c));
      for (int c=c_upd; c<lcols; c++) // F12
        for (int r=0; r<r_upd; r++)
          sbuf[pr[r]+pc[c]].push_back(CB(r,c));
      for (int c=0; c<c_upd; c++) // F21
        for (int r=r_upd; r<lrows; r++)
          sbuf[pr[r]+pc[c]].push_back(CB(r,c));
      for (int c=c_upd; c<lcols; c++) // F22
        for (int r=r_upd; r<lrows; r++)
          sbuf[pr[r]+pc[c]].push_back(CB(r,c));
      delete[] pr;
    }

    static void extend_add_seq_copy_to_buffers
    (const DenseM_t& CB, const DistM_t& F11, const DistM_t& F12,
     const DistM_t& F21, const DistM_t& F22,
     std::vector<std::vector<scalar_t>>& sbuf,
     const FrontalMatrixDenseMPI<scalar_t,integer_t>* pa,
     const FrontalMatrixDense<scalar_t,integer_t>* ch) {
      std::size_t u2s;
      const auto I = ch->upd_to_parent(pa, u2s);
      const std::size_t du = ch->dim_upd;
      const std::size_t ds = pa->dim_sep;
      const auto pr = new int[CB.rows()+CB.cols()];
      const auto pc = pr + CB.rows();
      const auto prows = pa->proc_rows;
      const auto pcols = pa->proc_cols;
      const auto B = DistM_t::default_MB;
      // destination rank is:
      //  ((r / B) % prows) + ((c / B) % pcols) * prows
      //  = pr[r] + pc[c]
      for (std::size_t i=0; i<u2s; i++) {
        auto Ii = I[i];
        pr[i] = (Ii / B) % prows;
        pc[i] = ((Ii / B) % pcols) * prows;
      }
      for (std::size_t i=u2s; i<du; i++) {
        auto Ii = I[i] - ds;
        pr[i] = (Ii / B) % prows;
        pc[i] = ((Ii / B) % pcols) * prows;
      }
      { // reserve space for the send buffers
        std::vector<std::size_t> cnt(sbuf.size());
        for (std::size_t c=0; c<du; c++)
          for (std::size_t r=0; r<du; r++)
            cnt[pr[r]+pc[c]]++;
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      for (std::size_t c=0; c<u2s; c++) // F11
        for (std::size_t r=0; r<u2s; r++)
          sbuf[pr[r]+pc[c]].push_back(CB(r,c));
      for (std::size_t c=u2s; c<du; c++) // F12
        for (std::size_t r=0; r<u2s; r++)
          sbuf[pr[r]+pc[c]].push_back(CB(r,c));
      for (std::size_t c=0; c<u2s; c++) // F21
        for (std::size_t r=u2s; r<du; r++)
          sbuf[pr[r]+pc[c]].push_back(CB(r,c));
      for (std::size_t c=u2s; c<du; c++) // F22
        for (std::size_t r=u2s; r<du; r++)
          sbuf[pr[r]+pc[c]].push_back(CB(r,c));
      delete[] pr;
    }


    static void extend_add_seq_copy_from_buffers
    (DistM_t& F11, DistM_t& F12, DistM_t& F21, DistM_t& F22,
     scalar_t*& pbuf, const FrontalMatrixDenseMPI<scalar_t,integer_t>* pa,
     const FrontalMatrixDense<scalar_t,integer_t>* ch) {
      if (!(F11.active() || F22.active())) return;
      const auto ch_dim_upd = ch->dim_upd;
      const auto ch_upd = ch->upd;
      const auto pa_upd = pa->upd;
      const auto pa_sep = pa->sep_begin;
      auto r_1 =
        new int[F11.lrows()+F11.lcols()+F22.lrows()+F22.lcols()];
      auto c_1 = r_1 + F11.lrows();
      auto r_2 = c_1 + F11.lcols();
      auto c_2 = r_2 + F22.lrows();
      integer_t r_max_1 = 0, r_max_2 = 0;
      integer_t c_max_1 = 0, c_max_2 = 0;
      for (int r=0, ur=0; r<F11.lrows(); r++) {
        auto fgr = F11.rowl2g_fixed(r) + pa_sep;
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_1[r_max_1++] = r;
      }
      for (int c=0, uc=0; c<F11.lcols(); c++) {
        auto fgc = F11.coll2g_fixed(c) + pa_sep;
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        c_1[c_max_1++] = c;
      }
      for (int r=0, ur=0; r<F22.lrows(); r++) {
        auto fgr = pa_upd[F22.rowl2g_fixed(r)];
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_2[r_max_2++] = r;
      }
      for (int c=0, uc=0; c<F22.lcols(); c++) {
        auto fgc = pa_upd[F22.coll2g_fixed(c)];
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        c_2[c_max_2++] = c;
      }
      for (int c=0; c<c_max_1; c++)
        for (int r=0; r<r_max_1; r++)
          F11(r_1[r],c_1[c]) += *(pbuf++);
      for (int c=0; c<c_max_2; c++)
        for (int r=0; r<r_max_1; r++)
          F12(r_1[r],c_2[c]) += *(pbuf++);
      for (int c=0; c<c_max_1; c++)
        for (int r=0; r<r_max_2; r++)
          F21(r_2[r],c_1[c]) += *(pbuf++);
      for (int c=0; c<c_max_2; c++)
        for (int r=0; r<r_max_2; r++)
          F22(r_2[r],c_2[c]) += *(pbuf++);
      delete[] r_1;
    }

    static void extend_add_copy_from_buffers
    (DistM_t& F11, DistM_t& F12, DistM_t& F21, DistM_t& F22,
     scalar_t** pbuf, const FrontalMatrixDenseMPI<scalar_t,integer_t>* pa,
     const FrontalMatrixDenseMPI<scalar_t,integer_t>* ch) {
      if (!(F11.active() || F22.active())) return;
      const auto ch_dim_upd = ch->dim_upd;
      const auto ch_upd = ch->upd;
      const auto pa_upd = pa->upd;
      const auto pa_sep = pa->sep_begin;
      const auto prows = ch->proc_rows;
      const auto pcols = ch->proc_cols;
      const auto B = DistM_t::default_MB;
      // source rank is
      //  ((r / B) % prows) + ((c / B) % pcols) * prows
      // where r,c is the coordinate in the F22 block of the child
      auto upd_r_1 =
        new int[2*F11.lrows()+2*F11.lcols()+
                2*F22.lrows()+2*F22.lcols()];
      auto upd_c_1 = upd_r_1 + F11.lrows();
      auto upd_r_2 = upd_c_1 + F11.lcols();
      auto upd_c_2 = upd_r_2 + F22.lrows();
      auto r_1 = upd_c_2 + F22.lcols();
      auto c_1 = r_1 + F11.lrows();
      auto r_2 = c_1 + F11.lcols();
      auto c_2 = r_2 + F22.lrows();
      integer_t r_max_1 = 0, r_max_2 = 0;
      integer_t c_max_1 = 0, c_max_2 = 0;
      for (int r=0, ur=0; r<F11.lrows(); r++) {
        auto fgr = F11.rowl2g_fixed(r) + pa_sep;
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_1[r_max_1] = r;
        upd_r_1[r_max_1++] = (ur / B) % prows;
      }
      for (int c=0, uc=0; c<F11.lcols(); c++) {
        auto fgc = F11.coll2g_fixed(c) + pa_sep;
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        c_1[c_max_1] = c;
        upd_c_1[c_max_1++] = ((uc / B) % pcols) * prows;
      }
      for (int r=0, ur=0; r<F22.lrows(); r++) {
        auto fgr = pa_upd[F22.rowl2g_fixed(r)];
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_2[r_max_2] = r;
        upd_r_2[r_max_2++] = (ur / B) % prows;
      }
      for (int c=0, uc=0; c<F22.lcols(); c++) {
        auto fgc = pa_upd[F22.coll2g_fixed(c)];
        while (uc < ch_dim_upd && ch_upd[uc] < fgc) uc++;
        if (uc == ch_dim_upd) break;
        if (ch_upd[uc] != fgc) continue;
        c_2[c_max_2] = c;
        upd_c_2[c_max_2++] = ((uc / B) % pcols) * prows;
      }
      for (int c=0; c<c_max_1; c++)
        for (int r=0; r<r_max_1; r++)
          F11(r_1[r],c_1[c]) += *(pbuf[upd_r_1[r]+upd_c_1[c]]++);
      for (int c=0; c<c_max_2; c++)
        for (int r=0; r<r_max_1; r++)
          F12(r_1[r],c_2[c]) += *(pbuf[upd_r_1[r]+upd_c_2[c]]++);
      for (int c=0; c<c_max_1; c++)
        for (int r=0; r<r_max_2; r++)
          F21(r_2[r],c_1[c]) += *(pbuf[upd_r_2[r]+upd_c_1[c]]++);
      for (int c=0; c<c_max_2; c++)
        for (int r=0; r<r_max_2; r++)
          F22(r_2[r],c_2[c]) += *(pbuf[upd_r_2[r]+upd_c_2[c]]++);
      delete[] upd_r_1;
    }

    static void skinny_extend_add_copy_to_buffers
    (const DistM_t& cSr, const DistM_t& cSc,
     std::vector<std::vector<scalar_t>>& sbuf,
     const FrontalMatrixHSSMPI<scalar_t,integer_t>* pa,
     const std::vector<std::size_t>& I) {
      if (!cSr.active()) return;
      assert(cSr.fixed() && cSc.fixed());
      const auto prows = pa->proc_rows;
      const auto pcols = pa->proc_cols;
      const auto B = DistM_t::default_MB;
      const auto lrows = cSr.lrows();
      const auto lcols = cSr.lcols();
      auto destr = new int[lrows+lcols];
      auto destc = destr + lrows;
      for (int r=0; r<lrows; r++)
        destr[r] = (I[cSr.rowl2g_fixed(r)] / B) % prows;
      for (int c=0; c<lcols; c++)
        destc[c] = ((cSr.coll2g_fixed(c) / B) % pcols) * prows;
      {
        std::vector<std::size_t> cnt(sbuf.size());
        for (int c=0; c<lcols; c++)
          for (int r=0; r<lrows; r++)
            cnt[destr[r]+destc[c]]++;
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      for (int c=0; c<lcols; c++)
        for (int r=0; r<lrows; r++)
          sbuf[destr[r]+destc[c]].push_back(cSr(r,c));
      for (int c=0; c<lcols; c++)
        for (int r=0; r<lrows; r++)
          sbuf[destr[r]+destc[c]].push_back(cSc(r,c));
      delete[] destr;
    }

    static void skinny_extend_add_copy_from_buffers
    (DistM_t& Sr, DistM_t& Sc, scalar_t** pbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa,
     const FrontalMatrixMPI<scalar_t,integer_t>* ch) {
      if (!Sr.active()) return;
      assert(Sr.fixed());
      const auto lrows = Sr.lrows();
      const auto lcols = Sc.lcols();
      const auto sep_begin = pa->sep_begin;
      const auto dim_sep = pa->dim_sep;
      const auto pa_upd = pa->upd;
      const auto ch_upd = ch->upd;
      const auto ch_dim_upd = ch->dim_upd;
      const auto prows = ch->proc_rows;
      const auto pcols = ch->proc_cols;
      const auto B = DistM_t::default_MB;
      auto lr = new int[2*lrows+lcols];
      auto srcr = lr + Sr.lrows();
      auto srcc = srcr + Sr.lrows();
      for (int c=0; c<lcols; c++)
        srcc[c] = ((Sr.coll2g_fixed(c) / B) % pcols) * prows;
      integer_t rmax = 0;
      for (int r=0, ur=0; r<lrows; r++) {
        auto t = Sr.rowl2g_fixed(r);
        auto fgr = (t < dim_sep) ? t+sep_begin : pa_upd[t-dim_sep];
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        lr[rmax] = r;
        srcr[rmax++] = (ur / B) % prows;
      }
      for (int c=0; c<lcols; c++)
        for (int r=0; r<rmax; r++)
          Sr(lr[r],c) += *(pbuf[srcr[r]+srcc[c]]++);
      for (int c=0; c<lcols; c++)
        for (int r=0; r<rmax; r++)
          Sc(lr[r],c) += *(pbuf[srcr[r]+srcc[c]]++);
      delete[] lr;
    }

    static void extend_copy_to_buffers
    (const DistM_t& F, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J, const DistM_t& B,
     std::vector<std::vector<scalar_t>>& sbuf) {
      if (!F.active()) return;
      assert(F.fixed());
      const auto lcols = F.lcols();
      const auto lrows = F.lrows();
      const auto prows = B.prows();
      const auto pcols = B.pcols();
      const auto MB = DistM_t::default_MB;
      auto destr = new int[lrows+lcols];
      auto destc = destr + lrows;
      for (int r=0; r<lrows; r++)
        destr[r] = (I[F.rowl2g_fixed(r)] / MB) % prows;
      for (int c=0; c<lcols; c++)
        destc[c] = ((J[F.coll2g_fixed(c)] / MB) % pcols) * prows;
      {
        std::vector<std::size_t> cnt(sbuf.size());
        for (int c=0; c<lcols; c++)
          for (int r=0; r<lrows; r++)
            cnt[destr[r]+destc[c]]++;
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      for (int c=0; c<lcols; c++)
        for (int r=0; r<lrows; r++)
          sbuf[destr[r]+destc[c]].push_back(F(r,c));
      delete[] destr;
    }

    static void extend_copy_from_buffers
    (DistM_t& F, const std::vector<std::size_t>& oI,
     const std::vector<std::size_t>& oJ, const DistM_t& B, scalar_t** pbuf) {
      if (!F.active()) return;
      assert(F.fixed());
      const auto prows = B.prows();
      const auto pcols = B.pcols();
      const auto MB = DistM_t::default_MB;
      auto srcr = new int[2*oI.size()];
      auto lr = srcr + oI.size();
      for (std::size_t r=0; r<oI.size(); r++) {
        srcr[r] = (r / MB) % prows;
        auto gr = oI[r];
        if (F.rowg2p_fixed(gr) == F.prow())
          lr[r] = F.rowg2l_fixed(gr);
        else lr[r] = -1;
      }
      for (std::size_t c=0; c<oJ.size(); c++) {
        auto gc = oJ[c];
        if (F.colg2p_fixed(gc) != F.pcol()) continue;
        auto lc = F.colg2l_fixed(gc);
        auto srcc = ((c / MB) % pcols) * prows;
        for (std::size_t r=0; r<oI.size(); r++)
          if (lr[r] != -1)
            F(lr[r],lc) += *(pbuf[srcr[r]+srcc]++);
      }
      delete[] srcr;
    }

    static void extend_add_column_copy_to_buffers
    (const DistM_t& Bch, const DistM_t& Bsep, const DistM_t& Bupd,
     std::vector<std::vector<scalar_t>>& sbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa,
     const std::vector<std::size_t>& I) {
      assert(Bch.fixed());
      for (int r=0; r<Bch.lrows(); r++) {
        integer_t pa_row = I[Bch.rowl2g_fixed(r)];
        if (pa_row < pa->dim_sep)
          sbuf[pa->find_rank_fixed(pa_row, 0, Bsep)].
            push_back(Bch(r,0));
        else
          sbuf[pa->find_rank_fixed(pa_row-pa->dim_sep, 0, Bupd)].
            push_back(Bch(r,0));
      }
    }

    // TODO use skinny-extend-add
    static void extend_add_column_copy_from_buffers
    (DistM_t& Bsep, DistM_t& Bupd, std::vector<std::vector<scalar_t>>& buf,
     integer_t sep_begin, const std::vector<integer_t>& pa_upd,
     const std::vector<integer_t>& ch_upd,
     std::function<int(integer_t,integer_t)> b_rank) {
      std::vector<scalar_t*> pbuf(buf.size());
      for (size_t p=0; p<buf.size(); p++)
        pbuf[p] = buf[p].data();
      std::function<integer_t(integer_t)> sep_map =
        [&](integer_t i) { return i + sep_begin; };
      std::function<integer_t(integer_t)> upd_map =
        [&](integer_t i) { return pa_upd[i]; };
      copy_column_from_buffer
        (Bsep, pbuf, ch_upd, b_rank, sep_map);
      copy_column_from_buffer
        (Bupd, pbuf, ch_upd, b_rank, upd_map);
    }
    static void copy_column_from_buffer
    (DistM_t& F, std::vector<scalar_t*>& pbuf,
     const std::vector<integer_t>& ch_upd,
     std::function<int(integer_t,integer_t)>
     b_child_rank, std::function<integer_t(integer_t)> f2g) {
      if (!F.active()) return;
      integer_t ch_dim_upd = ch_upd.size();
      integer_t upd_r = 0;
      for (int r=0; r<F.lrows(); r++) {
        auto fgr = f2g(F.rowl2g(r));
        while (upd_r < ch_dim_upd && ch_upd[upd_r] < fgr) upd_r++;
        if (upd_r == ch_dim_upd) break;
        if (ch_upd[upd_r] != fgr) continue;
        F(r,0) += *(pbuf[b_child_rank(upd_r,0)]++);
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

    // TODO optimize loops!!
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

  };


  template<typename scalar_t,typename integer_t> class ExtractFront {
    using CSM = CompressedSparseMatrix<scalar_t,integer_t>;
    using DistM_t = DistributedMatrix<scalar_t>;

  public:
    static void extract_F11
    (DistM_t& F, const CSM& A, integer_t sep_begin, integer_t dim_sep) {
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
          A.extract_F11_block
            (block, F.ld(), row+sep_begin, std::min(F.MB(),F.rows()-row),
             col+sep_begin, nr_cols);
        }
      }
    }

    static void extract_F12
    (DistM_t& F, const CSM& A, integer_t upd_row_begin,
     integer_t upd_col_begin, const std::vector<integer_t>& upd) {
      if (!F.active()) return;
      F.zero();
      const auto CB = F.colblocks();
      const auto RB = F.rowblocks();
#pragma omp parallel for collapse(2)
      for (int cb=0; cb<CB; cb++) {
        for (int rb=0; rb<RB; rb++) {
          auto col = (F.pcol()+cb*F.pcols())*F.NB();
          auto row = (F.prow()+rb*F.prows())*F.MB();
          auto block_upd = upd.data() + (F.pcol()+cb*F.pcols())*F.NB();
          auto nr_cols = std::min(F.NB(), F.cols()-col);
          auto block = F.data() + cb*F.NB()*F.ld() + rb*F.MB();
          A.extract_F12_block
            (block, F.ld(), row+upd_row_begin, std::min(F.MB(), F.rows()-row),
             col+upd_col_begin, nr_cols, block_upd);
        }
      }
    }

    static void extract_F21
    (DistM_t& F, const CSM& A, integer_t upd_row_begin,
     integer_t upd_col_begin, const std::vector<integer_t>& upd) {
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
          auto block_upd = upd.data() + F.prow()*F.MB() + rb*F.prows()*F.MB();
          A.extract_F21_block
            (block, F.ld(), row+upd_row_begin, std::min(F.MB(), F.rows()-row),
             col+upd_col_begin, nr_cols, block_upd);
        }
      }
    }
  };

} // end namespace strumpack

#endif // EXTEND_ADD_HPP
