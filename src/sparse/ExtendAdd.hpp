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
    (const DistM_t& CB, std::vector<std::vector<scalar_t>>& sbuf,
     const FrontalMatrixDenseMPI<scalar_t,integer_t>* pa,
     const std::vector<std::size_t>& I) {
      if (!CB.active()) return;
      assert(CB.fixed());
      const auto lrows = CB.lrows();
      const auto lcols = CB.lcols();
      const auto pa_sep = pa->dim_sep();
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
    (const DenseM_t& CB, std::vector<std::vector<scalar_t>>& sbuf,
     const FrontalMatrixDenseMPI<scalar_t,integer_t>* pa,
     const FrontalMatrixDense<scalar_t,integer_t>* ch) {
      std::size_t u2s;
      const auto I = ch->upd_to_parent(pa, u2s);
      const std::size_t du = ch->dim_upd();
      const std::size_t ds = pa->dim_sep();
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
      const auto ch_dim_upd = ch->dim_upd();
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
      const auto ch_dim_upd = ch->dim_upd();
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




    //////////////////////////////////////////////////////////////
    ////// 1D extend-add for the right-hand side /////////////////
    //////////////////////////////////////////////////////////////
    static void extend_add_column_copy_to_buffers
    (const DistM_t& CB, std::vector<std::vector<scalar_t>>& sbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa,
     const std::vector<std::size_t>& I) {
      if (!CB.active()) return;
      assert(CB.fixed());
      const auto lrows = CB.lrows();
      const auto lcols = CB.lcols();
      const auto pa_sep = pa->dim_sep();
      const auto prows = pa->proc_rows;
      const auto pcols = pa->proc_cols;
      const auto B = DistM_t::default_MB;
      // destination rank is:
      //  ((r / B) % prows) + ((c / B) % pcols) * prows
      //  = pr[r] + pc[c]
      auto pr = new int[CB.lrows()+CB.lcols()];
      auto pc = pr + CB.lrows();
      int r_upd;
      for (r_upd=0; r_upd<lrows; r_upd++) {
        auto t = I[CB.rowl2g_fixed(r_upd)];
        if (t >= std::size_t(pa_sep)) break;
        pr[r_upd] = (t / B) % prows;
      }
      for (int r=r_upd; r<lrows; r++)
        pr[r] = ((I[CB.rowl2g_fixed(r)]-pa_sep) / B) % prows;
      for (int c=0; c<lcols; c++)
        pc[c] = ((CB.coll2g_fixed(c) / B) % pcols) * prows;
      { // reserve space for the send buffers
        std::vector<std::size_t> cnt(sbuf.size());
        for (int c=0; c<lcols; c++)
          for (int r=0; r<lrows; r++)
            cnt[pr[r]+pc[c]]++;
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      for (int c=0; c<lcols; c++) // b
        for (int r=0; r<r_upd; r++)
          sbuf[pr[r]+pc[c]].push_back(CB(r,c));
      for (int c=0; c<lcols; c++) // bupd
        for (int r=r_upd; r<lrows; r++)
          sbuf[pr[r]+pc[c]].push_back(CB(r,c));
      delete[] pr;
    }

    static void extend_add_column_seq_copy_to_buffers
    (const DenseM_t& CB, std::vector<std::vector<scalar_t>>& sbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa,
     const FrontalMatrix<scalar_t,integer_t>* ch) {
      std::size_t u2s;
      const auto I = ch->upd_to_parent(pa, u2s);
      const std::size_t du = ch->dim_upd();
      const std::size_t ds = pa->dim_sep();
      const auto cols = CB.cols();
      const auto pr = new int[CB.rows()+cols];
      const auto pc = pr + CB.rows();
      const auto prows = pa->proc_rows;
      const auto pcols = pa->proc_cols;
      const auto B = DistM_t::default_MB;
      // destination rank is:
      //  ((r / B) % prows) + ((c / B) % pcols) * prows
      //  = pr[r] + pc[c]
      for (std::size_t r=0; r<u2s; r++)
        pr[r] = (I[r] / B) % prows;
      for (std::size_t r=u2s; r<du; r++)
        pr[r] = ((I[r]-ds) / B) % prows;
      for (std::size_t c=0; c<cols; c++)
        pc[c] = ((c / B) % pcols) * prows;
      { // reserve space for the send buffers
        std::vector<std::size_t> cnt(sbuf.size());
        for (std::size_t c=0; c<cols; c++)
          for (std::size_t r=0; r<du; r++)
            cnt[pr[r]+pc[c]]++;
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      for (std::size_t c=0; c<cols; c++) // b
        for (std::size_t r=0; r<u2s; r++)
          sbuf[pr[r]+pc[c]].push_back(CB(r,c));
      for (std::size_t c=0; c<cols; c++) // bupd
        for (std::size_t r=u2s; r<du; r++)
          sbuf[pr[r]+pc[c]].push_back(CB(r,c));
      delete[] pr;
    }

    static void extend_add_column_seq_copy_from_buffers
    (DistM_t& b, DistM_t& bupd, scalar_t*& pbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa,
     const FrontalMatrix<scalar_t,integer_t>* ch) {
      if (!(b.active() || bupd.active())) return;
      const auto ch_dim_upd = ch->dim_upd();
      const auto ch_upd = ch->upd;
      const auto pa_upd = pa->upd;
      const auto pa_sep = pa->sep_begin;
      const auto lcols = b.lcols();
      auto r_1 = new int[b.lrows()+bupd.lrows()];
      auto r_2 = r_1 + b.lrows();
      integer_t r_max_1 = 0, r_max_2 = 0;
      for (int r=0, ur=0; r<b.lrows(); r++) {
        auto fgr = b.rowl2g_fixed(r) + pa_sep;
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_1[r_max_1++] = r;
      }
      for (int r=0, ur=0; r<bupd.lrows(); r++) {
        auto fgr = pa_upd[bupd.rowl2g_fixed(r)];
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_2[r_max_2++] = r;
      }
      for (int c=0; c<lcols; c++)
        for (int r=0; r<r_max_1; r++)
          b(r_1[r],c) += *(pbuf++);
      for (int c=0; c<lcols; c++)
        for (int r=0; r<r_max_2; r++)
          bupd(r_2[r],c) += *(pbuf++);
      delete[] r_1;
    }

    static void extend_add_column_copy_from_buffers
    (DistM_t& b, DistM_t& bupd, scalar_t** pbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa,
     const FrontalMatrixMPI<scalar_t,integer_t>* ch) {
      if (!(b.active() || bupd.active())) return;
      const auto ch_dim_upd = ch->dim_upd();
      const auto ch_upd = ch->upd;
      const auto pa_upd = pa->upd;
      const auto pa_sep = pa->sep_begin;
      const auto prows = ch->proc_rows;
      const auto pcols = ch->proc_cols;
      const auto B = DistM_t::default_MB;
      const auto lcols = b.lcols();
      // source rank is
      //  ((r / B) % prows) + ((c / B) % pcols) * prows
      // where r,c is the coordinate in the F22 block of the child
      auto upd_r_1 = new int[2*b.lrows()+2*bupd.lrows()+lcols];
      auto upd_r_2 = upd_r_1 + b.lrows();
      auto r_1 = upd_r_2 + bupd.lrows();
      auto r_2 = r_1 + b.lrows();
      auto upd_c_1 = r_2 + bupd.lrows();
      integer_t r_max_1 = 0, r_max_2 = 0;
      for (int r=0, ur=0; r<b.lrows(); r++) {
        auto fgr = b.rowl2g_fixed(r) + pa_sep;
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_1[r_max_1] = r;
        upd_r_1[r_max_1++] = (ur / B) % prows;
      }
      // TODO ur can just continue from before?
      for (int r=0, ur=0; r<bupd.lrows(); r++) {
        auto fgr = pa_upd[bupd.rowl2g_fixed(r)];
        while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
        if (ur == ch_dim_upd) break;
        if (ch_upd[ur] != fgr) continue;
        r_2[r_max_2] = r;
        upd_r_2[r_max_2++] = (ur / B) % prows;
      }
      for (int c=0; c<lcols; c++)
        upd_c_1[c] = ((b.coll2g_fixed(c) / B) % pcols) * prows;
      for (int c=0; c<lcols; c++)
        for (int r=0; r<r_max_1; r++)
          b(r_1[r],c) += *(pbuf[upd_r_1[r]+upd_c_1[c]]++);
      for (int c=0; c<lcols; c++)
        for (int r=0; r<r_max_2; r++)
          bupd(r_2[r],c) += *(pbuf[upd_r_2[r]+upd_c_1[c]]++);
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
      const auto dim_sep = pa->dim_sep();
      const auto pa_upd = pa->upd;
      const auto ch_upd = ch->upd;
      const auto ch_dim_upd = ch->dim_upd();
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

    static void extract_column_copy_to_buffers
    (const DistM_t& b, const DistM_t& bupd,
     std::vector<std::vector<scalar_t>>& sbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa,
     const FrontalMatrixMPI<scalar_t,integer_t>* ch) {
      const auto I = ch->upd_to_parent(pa);
      const std::size_t pa_dim_sep = b.rows();
      const std::size_t ch_dim_upd = ch->dim_upd();
      const auto ch_master = pa->child_master(ch);
      const auto prows = ch->proc_rows;
      const auto pcols = ch->proc_cols;
      const auto B = DistM_t::default_MB;
      const std::size_t blcols = b.lcols();
      const std::size_t blrows = b.lrows();
      const std::size_t ulrows = bupd.lrows();
      auto pb = new int[2*blrows+2*ulrows+blcols];
      auto rb = pb + blrows;
      auto pu = rb + blrows;
      auto ru = pu + ulrows;
      auto pc = ru + ulrows;
      std::size_t ur = 0, brmax = 0, urmax = 0;
      for (std::size_t r=0, ur=0; r<blrows; r++) {
        const std::size_t gr = b.rowl2g_fixed(r);
        while (ur < ch_dim_upd && I[ur] < gr) ur++;
        if (ur == ch_dim_upd) break;
        if (I[ur] >= pa_dim_sep) break;
        if (I[ur] != gr) continue;
        rb[brmax] = r;
        pb[brmax++] = ch_master + (ur / B) % prows;
      }
      for (std::size_t r=0; r<ulrows; r++) {
        const auto gr = bupd.rowl2g_fixed(r) + pa_dim_sep;
        while (ur < ch_dim_upd && I[ur] < gr) ur++;
        if (ur == ch_dim_upd) break;
        if (I[ur] != gr) continue;
        ru[urmax] = r;
        pu[urmax++] = ch_master + (ur / B) % prows;
      }
      for (std::size_t c=0; c<blcols; c++)
        pc[c] = ((b.coll2g_fixed(c) / B) % pcols) * prows;
      {
        std::vector<std::size_t> cnt(sbuf.size());
        for (std::size_t c=0; c<blcols; c++) {
          for (std::size_t r=0; r<brmax; r++)
            cnt[pb[r]+pc[c]]++;
          for (std::size_t r=0; r<urmax; r++)
            cnt[pu[r]+pc[c]]++;
        }
        for (std::size_t p=0; p<sbuf.size(); p++)
          sbuf[p].reserve(sbuf[p].size()+cnt[p]);
      }
      for (std::size_t c=0; c<blcols; c++) {
        for (std::size_t r=0; r<brmax; r++)
          sbuf[pb[r]+pc[c]].push_back(b(rb[r],c));
        for (std::size_t r=0; r<urmax; r++)
          sbuf[pu[r]+pc[c]].push_back(bupd(ru[r],c));
      }
      delete[] pb;
    }

    static void extract_column_seq_copy_to_buffers
    (const DistM_t& b, const DistM_t& bupd, std::vector<scalar_t>& sbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa,
     const FrontalMatrix<scalar_t,integer_t>* ch) {
      const auto I = ch->upd_to_parent(pa);
      const std::size_t pa_dim_sep = b.rows();
      const std::size_t ch_dim_upd = ch->dim_upd();
      const std::size_t blcols = b.lcols();
      const std::size_t blrows = b.lrows();
      const std::size_t ulrows = bupd.lrows();
      auto rb = new int[blrows+ulrows];
      auto ru = rb + blrows;
      std::size_t ur = 0, brmax = 0, urmax = 0;
      for (std::size_t r=0, ur=0; r<blrows; r++) {
        const std::size_t gr = b.rowl2g_fixed(r);
        while (ur < ch_dim_upd && I[ur] < gr) ur++;
        if (ur == ch_dim_upd) break;
        if (I[ur] >= pa_dim_sep) break;
        if (I[ur] != gr) continue;
        rb[brmax++] = r;
      }
      for (std::size_t r=0; r<ulrows; r++) {
        const auto gr = bupd.rowl2g_fixed(r) + pa_dim_sep;
        while (ur < ch_dim_upd && I[ur] < gr) ur++;
        if (ur == ch_dim_upd) break;
        if (I[ur] != gr) continue;
        ru[urmax++] = r;
      }
      sbuf.reserve(sbuf.size()+(brmax+urmax)*blcols);
      for (std::size_t c=0; c<blcols; c++) {
        for (std::size_t r=0; r<brmax; r++)
          sbuf.push_back(b(rb[r],c));
        for (std::size_t r=0; r<urmax; r++)
          sbuf.push_back(bupd(ru[r],c));
      }
      delete[] rb;
    }


    // TODO optimize loops
    static void extract_column_copy_from_buffers
    (DistM_t& CB, scalar_t** pbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa,
     const FrontalMatrix<scalar_t,integer_t>* ch) {
      const auto I = ch->upd_to_parent(pa);
      const auto prows = pa->proc_rows;
      const auto pcols = pa->proc_cols;
      const auto B = DistM_t::default_MB;
      const auto pa_dim_sep = pa->dim_sep();
      for (int c=0; c<CB.lcols(); c++) {
        const auto pc = ((CB.coll2g_fixed(c) / B) % pcols) * prows;
        for (int r=0; r<CB.lrows(); r++) {
          integer_t gr = I[CB.rowl2g_fixed(r)];
          if (gr >= pa_dim_sep) gr -= pa_dim_sep;
          auto pr = (gr / B) % prows;
          CB(r,c) = *(pbuf[pr+pc]++);
        }
      }
    }

    // TODO optimize loops
    static void extract_column_seq_copy_from_buffers
    (DenseM_t& CB, scalar_t** pbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa,
     const FrontalMatrix<scalar_t,integer_t>* ch) {
      const auto I = ch->upd_to_parent(pa);
      const auto prows = pa->proc_rows;
      const auto pcols = pa->proc_cols;
      const auto B = DistM_t::default_MB;
      const auto pa_dim_sep = pa->dim_sep();
      for (std::size_t c=0; c<CB.cols(); c++) {
        const auto pc = ((c / B) % pcols) * prows;
        for (std::size_t r=0; r<CB.rows(); r++) {
          integer_t gr = I[r];
          if (gr >= pa_dim_sep) gr -= pa_dim_sep;
          auto pr = (gr / B) % prows;
          CB(r,c) = *(pbuf[pr+pc]++);
        }
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
