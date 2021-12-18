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

#include "ExtendAdd.hpp"
#include "FrontalMatrix.hpp"
#include "FrontalMatrixMPI.hpp"
#include "BLR/BLRMatrixMPI.hpp"

#include "sparse/CompressedSparseMatrix.hpp"


namespace strumpack {

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extend_add_copy_to_buffers
  (const DistM_t& CB, VVS_t& sbuf, const FMPI_t* pa, const VI_t& I) {
    if (!CB.active()) return;
    assert(CB.fixed());
    const auto lrows = CB.lrows();
    const auto lcols = CB.lcols();
    const auto pa_sep = pa->dim_sep();
    const auto prows = pa->grid()->nprows();
    const auto pcols = pa->grid()->npcols();
    const auto B = DistM_t::default_MB;
    // destination rank is:
    //  ((r / B) % prows) + ((c / B) % pcols) * prows
    //  = pr[r] + pc[c]
    std::unique_ptr<int[]> pr(new int[lrows+lcols]);
    auto pc = pr.get() + CB.lrows();
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
      VI_t cnt(sbuf.size());
      for (int c=0; c<lcols; c++)
        for (int r=0; r<lrows; r++)
          cnt[pr[r]+pc[c]]++;
      for (std::size_t p=0; p<sbuf.size(); p++)
        sbuf[p].reserve(sbuf[p].size()+cnt[p]);
    }
    if (params::num_threads != 1) {
#pragma omp parallel for
      for (int t=0; t<int(sbuf.size()); t++) {
        for (int c=0; c<c_upd; c++) // F11
          for (int r=0, pcc=pc[c]; r<r_upd; r++) {
            auto d = pr[r]+pcc;
            if (d == t) sbuf[d].push_back(CB(r,c));
          }
        for (int c=c_upd; c<lcols; c++) // F12
          for (int r=0, pcc=pc[c]; r<r_upd; r++) {
            auto d = pr[r]+pcc;
            if (d == t) sbuf[d].push_back(CB(r,c));
          }
        for (int c=0; c<c_upd; c++) // F21
          for (int r=r_upd, pcc=pc[c]; r<lrows; r++) {
            auto d = pr[r]+pcc;
            if (d == t) sbuf[d].push_back(CB(r,c));
          }
        for (int c=c_upd; c<lcols; c++) // F22
          for (int r=r_upd, pcc=pc[c]; r<lrows; r++) {
            auto d = pr[r]+pcc;
            if (d == t) sbuf[d].push_back(CB(r,c));
          }
      }
    } else {
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
  }

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extend_add_seq_copy_to_buffers
  (const DenseM_t& CB, VVS_t& sbuf, const FMPI_t* pa, const F_t* ch) {
    std::size_t u2s;
    const auto I = ch->upd_to_parent
      (static_cast<const F_t*>(pa), u2s);
    const std::size_t du = ch->dim_upd();
    const std::size_t ds = pa->dim_sep();
    std::unique_ptr<int[]> pr(new int[CB.rows()+CB.cols()]);
    const auto pc = pr.get() + CB.rows();
    const auto prows = pa->grid()->nprows();
    const auto pcols = pa->grid()->npcols();
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
  ExtendAdd<scalar_t,integer_t>::extend_add_seq_copy_from_buffers
  (DistM_t& F11, DistM_t& F12, DistM_t& F21, DistM_t& F22,
   scalar_t*& pbuf, const FMPI_t* pa, const F_t* ch) {
    if (!(F11.active() || F22.active())) return;
    const auto ch_dim_upd = ch->dim_upd();
    const auto& ch_upd = ch->upd();
    const auto& pa_upd = pa->upd();
    const auto pa_sep = pa->sep_begin();
    std::unique_ptr<int[]> r_1
      (new int[F11.lrows()+F11.lcols()+F22.lrows()+F22.lcols()]);
    auto c_1 = r_1.get() + F11.lrows();
    auto r_2 = c_1 + F11.lcols();
    auto c_2 = r_2 + F22.lrows();
    integer_t r_max_1 = 0, r_max_2 = 0, c_max_1 = 0, c_max_2 = 0;
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
  ExtendAdd<scalar_t,integer_t>::extend_add_copy_from_buffers
  (DistM_t& F11, DistM_t& F12, DistM_t& F21, DistM_t& F22,
   scalar_t** pbuf, const FMPI_t* pa, const FMPI_t* ch) {
    if (!(F11.active() || F22.active())) return;
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
  ExtendAdd<scalar_t,integer_t>::extend_add_column_copy_to_buffers
  (const DistM_t& CB, VVS_t& sbuf, const FMPI_t* pa, const VI_t& I) {
    if (!CB.active()) return;
    assert(CB.fixed());
    const auto lrows = CB.lrows();
    const auto lcols = CB.lcols();
    const auto pa_sep = pa->dim_sep();
    const auto prows = pa->grid()->nprows();
    const auto pcols = pa->grid()->npcols();
    const auto B = DistM_t::default_MB;
    // destination rank is:
    //  ((r / B) % prows) + ((c / B) % pcols) * prows
    //  = pr[r] + pc[c]
    std::unique_ptr<int[]> iwork(new int[lrows+lcols]);
    auto pr = iwork.get();
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
      VI_t cnt(sbuf.size());
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
  }

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extend_add_column_seq_copy_to_buffers
  (const DenseM_t& CB, VVS_t& sbuf, const FMPI_t* pa, const F_t* ch) {
    std::size_t u2s;
    const auto I = ch->upd_to_parent(pa, u2s);
    const std::size_t du = ch->dim_upd();
    const std::size_t ds = pa->dim_sep();
    const auto cols = CB.cols();
    std::unique_ptr<int[]> iwork(new int[CB.rows()+cols]);
    const auto pr = iwork.get();
    const auto pc = pr + CB.rows();
    const auto prows = pa->grid()->nprows();
    const auto pcols = pa->grid()->npcols();
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
      VI_t cnt(sbuf.size());
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
  }

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extend_add_column_seq_copy_from_buffers
  (DistM_t& b, DistM_t& bupd, scalar_t*& pbuf,
   const FMPI_t* pa, const F_t* ch) {
    if (!(b.active() || bupd.active())) return;
    const auto ch_dim_upd = ch->dim_upd();
    const auto& ch_upd = ch->upd();
    const auto& pa_upd = pa->upd();
    const auto pa_sep = pa->sep_begin();
    const auto lcols = b.lcols();
    std::unique_ptr<int[]> r_1(new int[b.lrows()+bupd.lrows()]);
    auto r_2 = r_1.get() + b.lrows();
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
  }

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extend_add_column_copy_from_buffers
  (DistM_t& b, DistM_t& bupd, scalar_t** pbuf,
   const FMPI_t* pa, const FMPI_t* ch) {
    if (!(b.active() || bupd.active())) return;
    const auto ch_dim_upd = ch->dim_upd();
    const auto& ch_upd = ch->upd();
    const auto& pa_upd = pa->upd();
    const auto pa_sep = pa->sep_begin();
    const auto prows = ch->grid()->nprows();
    const auto pcols = ch->grid()->npcols();
    const auto B = DistM_t::default_MB;
    const auto lcols = b.lcols();
    // source rank is
    //  ((r / B) % prows) + ((c / B) % pcols) * prows
    // where r,c is the coordinate in the F22 block of the child
    std::unique_ptr<int[]> upd_r_1(new int[2*b.lrows()+2*bupd.lrows()+lcols]);
    auto upd_r_2 = upd_r_1.get() + b.lrows();
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
  }

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::skinny_extend_add_copy_to_buffers
  (const DistM_t& cS, VVS_t& sbuf, const FMPI_t* pa, const VI_t& I) {
    if (!cS.active()) return;
    assert(cS.fixed());
    const auto prows = pa->grid()->nprows();
    const auto pcols = pa->grid()->npcols();
    const auto B = DistM_t::default_MB;
    const auto lrows = cS.lrows();
    const auto lcols = cS.lcols();
    std::unique_ptr<int[]> destr(new int[lrows+lcols]);
    auto destc = destr.get() + lrows;
    for (int r=0; r<lrows; r++)
      destr[r] = (I[cS.rowl2g_fixed(r)] / B) % prows;
    for (int c=0; c<lcols; c++)
      destc[c] = ((cS.coll2g_fixed(c) / B) % pcols) * prows;
    {
      VI_t cnt(sbuf.size());
      for (int c=0; c<lcols; c++)
        for (int r=0, dc=destc[c]; r<lrows; r++)
          cnt[destr[r]+dc]++;
      for (std::size_t p=0; p<sbuf.size(); p++)
        sbuf[p].reserve(sbuf[p].size()+cnt[p]);
    }
    for (int c=0; c<lcols; c++)
      for (int r=0, dc=destc[c]; r<lrows; r++)
        sbuf[destr[r]+dc].push_back(cS(r,c));
  }


  /*
   * This does not do the 'extend' part, that was already done at
   * the child. Here, just send to the parent, so it can be added.
   */
  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::skinny_extend_add_seq_copy_to_buffers
  (const DenseM_t& cS, VVS_t& sbuf, const FMPI_t* pa) {
    const auto prows = pa->grid()->nprows();
    const auto pcols = pa->grid()->npcols();
    const auto B = DistM_t::default_MB;
    const int rows = cS.rows();
    const int cols = cS.cols();
    auto destr = new int[rows+cols];
    auto destc = destr + rows;
    for (int r=0; r<rows; r++)
      destr[r] = (r / B) % prows;
    for (int c=0; c<cols; c++)
      destc[c] = ((c / B) % pcols) * prows;
    {
      VI_t cnt(sbuf.size());
      for (int c=0; c<cols; c++)
        for (int r=0; r<rows; r++)
          cnt[destr[r]+destc[c]]++;
      for (std::size_t p=0; p<sbuf.size(); p++)
        sbuf[p].reserve(sbuf[p].size()+cnt[p]);
    }
    for (int c=0; c<cols; c++)
      for (int r=0; r<rows; r++)
        sbuf[destr[r]+destc[c]].push_back(cS(r,c));
    delete[] destr;
  }

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::skinny_extend_add_copy_from_buffers
  (DistM_t& S, scalar_t** pbuf, const FMPI_t* pa, const FMPI_t* ch) {
    if (!S.active()) return;
    assert(S.fixed());
    const auto lrows = S.lrows();
    const auto lcols = S.lcols();
    const auto sep_begin = pa->sep_begin();
    const auto dim_sep = pa->dim_sep();
    const auto& pa_upd = pa->upd();
    const auto& ch_upd = ch->upd();
    const auto ch_dim_upd = ch->dim_upd();
    const auto prows = ch->grid()->nprows();
    const auto pcols = ch->grid()->npcols();
    const auto B = DistM_t::default_MB;
    auto lr = new int[2*lrows+lcols];
    auto srcr = lr + lrows;
    auto srcc = srcr + lrows;
    for (int c=0; c<lcols; c++)
      srcc[c] = ((S.coll2g_fixed(c) / B) % pcols) * prows;
    integer_t rmax = 0;
    for (int r=0, ur=0; r<lrows; r++) {
      auto t = S.rowl2g_fixed(r);
      auto fgr = (t < dim_sep) ? t+sep_begin : pa_upd[t-dim_sep];
      while (ur < ch_dim_upd && ch_upd[ur] < fgr) ur++;
      if (ur == ch_dim_upd) break;
      if (ch_upd[ur] != fgr) continue;
      lr[rmax] = r;
      srcr[rmax++] = (ur / B) % prows;
    }
    for (int c=0; c<lcols; c++)
      for (int r=0; r<rmax; r++)
        S(lr[r],c) += *(pbuf[srcr[r]+srcc[c]]++);
    delete[] lr;
  }

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::skinny_extend_add_seq_copy_from_buffers
  (DistM_t& S, scalar_t*& pbuf, const FMPI_t* pa, const F_t* ch) {
    if (!S.active()) return;
    assert(S.fixed());
    const auto lrows = S.lrows();
    const auto lcols = S.lcols();
    for (int c=0; c<lcols; c++)
      for (int r=0; r<lrows; r++)
        S(r,c) += *(pbuf++);
  }


  // TODO what if B is not active?? Do we have the correct processor
  // grid info???
  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extend_copy_to_buffers
  (const DistM_t& F, const VI_t& I, const VI_t& J,
   const DistM_t& B, VVS_t& sbuf) {
    if (!F.active()) return;
    assert(F.fixed());
    const auto lcols = F.lcols();
    const auto lrows = F.lrows();
    const auto prows = B.nprows();
    const auto pcols = B.npcols();
    const auto MB = DistM_t::default_MB;
    auto destr = new int[lrows+lcols];
    auto destc = destr + lrows;
    for (int r=0; r<lrows; r++)
      destr[r] = (I[F.rowl2g_fixed(r)] / MB) % prows;
    for (int c=0; c<lcols; c++)
      destc[c] = ((J[F.coll2g_fixed(c)] / MB) % pcols) * prows;
    {
      VI_t cnt(sbuf.size());
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

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extend_copy_from_buffers
  (DistM_t& F, const VI_t& oI, const VI_t& oJ, const DistM_t& B,
   std::vector<scalar_t*>& pbuf) {
    if (!F.active()) return;
    assert(F.fixed());
    const auto prows = B.nprows();
    const auto pcols = B.npcols();
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

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extract_column_copy_to_buffers
  (const DistM_t& b, const DistM_t& bupd, VVS_t& sbuf,
   const FMPI_t* pa, const FMPI_t* ch) {
    const auto I = ch->upd_to_parent(pa);
    const std::size_t pa_dim_sep = b.rows();
    const std::size_t ch_dim_upd = ch->dim_upd();
    const auto ch_master = pa->master(ch);
    const auto prows = ch->grid()->nprows();
    const auto pcols = ch->grid()->npcols();
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
      VI_t cnt(sbuf.size());
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

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extract_column_seq_copy_to_buffers
  (const DistM_t& b, const DistM_t& bupd, std::vector<scalar_t>& sbuf,
   const FMPI_t* pa, const F_t* ch) {
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
  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extract_column_copy_from_buffers
  (DistM_t& CB, std::vector<scalar_t*>& pbuf,
   const FMPI_t* pa, const F_t* ch) {
    const auto I = ch->upd_to_parent(pa);
    const auto prows = pa->grid()->nprows();
    const auto pcols = pa->grid()->npcols();
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
  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extract_column_seq_copy_from_buffers
  (DenseM_t& CB, std::vector<scalar_t*>& pbuf,
   const FMPI_t* pa, const F_t* ch) {
    const auto I = ch->upd_to_parent(pa);
    const auto prows = pa->grid()->nprows();
    const auto pcols = pa->grid()->npcols();
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


  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extract_copy_to_buffers
  (const DistM_t& F, const VI_t& I, const VI_t& J, const VI_t& oI,
   const VI_t& oJ, const DistM_t& B, VVS_t& sbuf) {
    if (!F.active()) return;
    assert(F.fixed() && B.fixed());
    auto prow = F.prow();
    auto pcol = F.pcol();
    auto nprows = B.nprows();
    std::unique_ptr<int[]> pr(new int[2*I.size()]);
    auto lr = pr.get() + I.size();
    for (std::size_t r=0; r<I.size(); r++) {
      pr[r] = B.rowg2p_fixed(oI[r]);
      auto gr = I[r];
      lr[r] = (F.rowg2p_fixed(gr) == prow) ?
        F.rowg2l_fixed(gr) : -1;
    }
    for (std::size_t c=0; c<J.size(); c++) {
      auto gc = J[c];
      if (F.colg2p_fixed(gc) != pcol) continue;
      auto lc = F.colg2l_fixed(gc);
      auto pc = nprows * B.colg2p_fixed(oJ[c]);
      for (std::size_t r=0; r<I.size(); r++) {
        if (lr[r] == -1) continue;
        sbuf[pr[r]+pc].push_back(F(lr[r],lc));
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extract_copy_from_buffers
  (DistM_t& F, const VI_t& I, const VI_t& J, const VI_t& oI, const VI_t& oJ,
   const DistM_t& B, std::vector<scalar_t*>& pbuf) {
    if (!F.active()) return;
    assert(F.fixed() && B.fixed());
    auto prow = F.prow();
    auto pcol = F.pcol();
    auto nprows = B.nprows();
    std::unique_ptr<int[]> pr(new int[2*I.size()]);
    auto lr = pr.get() + I.size();
    for (std::size_t r=0; r<I.size(); r++) {
      pr[r] = B.rowg2p_fixed(I[r]);
      auto gr = oI[r];
      lr[r] = (F.rowg2p_fixed(gr) == prow) ?
        F.rowg2l_fixed(oI[r]) : -1;
    }
    for (std::size_t c=0; c<oJ.size(); c++) {
      auto gc = oJ[c];
      if (F.colg2p_fixed(gc) != pcol) continue;
      auto pc = nprows * B.colg2p_fixed(J[c]);
      auto lc = F.colg2l_fixed(gc);
      for (std::size_t r=0; r<oI.size(); r++) {
        if (lr[r] == -1) continue;
        F(lr[r],lc) += *(pbuf[pr[r]+pc]++);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extract_copy_to_buffers
  (const BLRMPI_t& F, const VI_t& I, const VI_t& J,
   const VI_t& oI, const VI_t& oJ, const DistM_t& B, VVS_t& sbuf) {
    if (!F.active()) return;
    assert(B.fixed());
    auto pcol = F.grid()->pcol();
    auto prow = F.grid()->prow();
    auto nprows = B.nprows();
    std::unique_ptr<int[]> tr(new int[3*I.size()]);
    auto lr = tr.get() + I.size();
    auto pr = lr + I.size();
    for (std::size_t r=0; r<I.size(); r++) {
      auto gr = I[r];
      if (F.rg2p(gr) == prow) {
        auto t = F.rg2t(gr);
        tr[r] = F.tilerg2l(t);
        lr[r] = gr - F.tileroff(t);
        pr[r] = B.rowg2p_fixed(oI[r]);
      } else lr[r] = -1;
    }
    for (std::size_t c=0; c<J.size(); c++) {
      auto gc = J[c];
      if (F.cg2p(gc) != pcol) continue;
      auto tc = F.cg2t(gc);
      auto lc = gc - F.tilecoff(tc);
      tc = F.tilecg2l(tc);
      auto pc = nprows * B.colg2p_fixed(oJ[c]);
      for (std::size_t r=0; r<I.size(); r++)
        if (lr[r] != -1)
          //sbuf[pr[r]+pc].push_back(F.ltile_dense(tr[r],tc).D()(lr[r],lc));
          sbuf[pr[r]+pc].push_back(
            const_cast<BLRMPI_t&>(F).get_element_and_decompress_HODBF(tr[r],tc,lr[r],lc));
    }
  }

  template<typename scalar_t,typename integer_t> void
  ExtendAdd<scalar_t,integer_t>::extract_copy_from_buffers
  (DistM_t& F, const VI_t& I, const VI_t& J, const VI_t& oI, const VI_t& oJ,
   const BLRMPI_t& B, std::vector<scalar_t*>& pbuf) {
    if (!F.active()) return;
    assert(F.fixed());
    auto pcol = F.pcol();
    auto prow = F.prow();
    auto g = B.grid();
    auto nprows = g->nprows();
    std::unique_ptr<int[]> lr(new int[2*I.size()]);
    auto pr = lr.get() + I.size();
    for (std::size_t r=0; r<I.size(); r++) {
      auto gr = oI[r];
      if (F.rowg2p_fixed(gr) == prow) {
        lr[r] = F.rowg2l_fixed(gr);
        pr[r] = g->rg2p(B.rg2t(I[r]));
      } else lr[r] = -1;
    }
    for (std::size_t c=0; c<oJ.size(); c++) {
      auto gc = oJ[c];
      if (F.colg2p_fixed(gc) != pcol) continue;
      auto lc = F.colg2l_fixed(gc);
      auto pc = nprows * g->cg2p(B.cg2t(J[c]));
      for (std::size_t r=0; r<oI.size(); r++)
        if (lr[r] != -1)
          F(lr[r], lc) += *(pbuf[pr[r]+pc]++);
    }
  }

  template<typename scalar_t,typename integer_t> void
  ExtractFront<scalar_t,integer_t>::extract_F11
  (DistM_t& F, const CSM& A, integer_t sep_begin, integer_t dim_sep) {
    if (!F.active()) return;
    F.zero();
    const auto CB = F.colblocks();
    const auto RB = F.rowblocks();
#pragma omp parallel for collapse(2)
    for (int cb=0; cb<CB; cb++) {
      for (int rb=0; rb<RB; rb++) {
        auto col = (F.pcol()+cb*F.npcols())*F.NB();
        auto row = (F.prow()+rb*F.nprows())*F.MB();
        auto block = F.data() + cb*F.NB()*F.ld() + rb*F.MB();
        auto nr_cols = std::min(F.NB(), F.cols()-col);
        A.extract_F11_block
          (block, F.ld(), row+sep_begin, std::min(F.MB(),F.rows()-row),
           col+sep_begin, nr_cols);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  ExtractFront<scalar_t,integer_t>::extract_F12
  (DistM_t& F, const CSM& A, integer_t upd_row_begin,
   integer_t upd_col_begin, const std::vector<integer_t>& upd) {
    if (!F.active()) return;
    F.zero();
    const auto CB = F.colblocks();
    const auto RB = F.rowblocks();
#pragma omp parallel for collapse(2)
    for (int cb=0; cb<CB; cb++) {
      for (int rb=0; rb<RB; rb++) {
        auto col = (F.pcol()+cb*F.npcols())*F.NB();
        auto row = (F.prow()+rb*F.nprows())*F.MB();
        auto block_upd = upd.data() + (F.pcol()+cb*F.npcols())*F.NB();
        auto nr_cols = std::min(F.NB(), F.cols()-col);
        auto block = F.data() + cb*F.NB()*F.ld() + rb*F.MB();
        A.extract_F12_block
          (block, F.ld(), row+upd_row_begin, std::min(F.MB(), F.rows()-row),
           col+upd_col_begin, nr_cols, block_upd);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  ExtractFront<scalar_t,integer_t>::extract_F21
  (DistM_t& F, const CSM& A, integer_t upd_row_begin,
   integer_t upd_col_begin, const std::vector<integer_t>& upd) {
    if (!F.active()) return;
    F.zero();
    const auto CB = F.colblocks();
    const auto RB = F.rowblocks();
#pragma omp parallel for collapse(2)
    for (int cb=0; cb<CB; cb++) {
      for (int rb=0; rb<RB; rb++) {
        auto col = (F.pcol()+cb*F.npcols())*F.NB();
        auto row = (F.prow()+rb*F.nprows())*F.MB();
        auto nr_cols = std::min(F.NB(), F.cols()-col);
        auto block = F.data() + cb*F.NB()*F.ld() + rb*F.MB();
        auto block_upd = upd.data() + F.prow()*F.MB() + rb*F.nprows()*F.MB();
        A.extract_F21_block
          (block, F.ld(), row+upd_row_begin, std::min(F.MB(), F.rows()-row),
           col+upd_col_begin, nr_cols, block_upd);
      }
    }
  }

  // explicit template instantiations
  template class ExtendAdd<float,int>;
  template class ExtendAdd<double,int>;
  template class ExtendAdd<std::complex<float>,int>;
  template class ExtendAdd<std::complex<double>,int>;

  template class ExtendAdd<float,long int>;
  template class ExtendAdd<double,long int>;
  template class ExtendAdd<std::complex<float>,long int>;
  template class ExtendAdd<std::complex<double>,long int>;

  template class ExtendAdd<float,long long int>;
  template class ExtendAdd<double,long long int>;
  template class ExtendAdd<std::complex<float>,long long int>;
  template class ExtendAdd<std::complex<double>,long long int>;


  template class ExtractFront<float,int>;
  template class ExtractFront<double,int>;
  template class ExtractFront<std::complex<float>,int>;
  template class ExtractFront<std::complex<double>,int>;

  template class ExtractFront<float,long int>;
  template class ExtractFront<double,long int>;
  template class ExtractFront<std::complex<float>,long int>;
  template class ExtractFront<std::complex<double>,long int>;

  template class ExtractFront<float,long long int>;
  template class ExtractFront<double,long long int>;
  template class ExtractFront<std::complex<float>,long long int>;
  template class ExtractFront<std::complex<double>,long long int>;

} // end namespace strumpack
