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
#ifndef HSS_MATRIX_MPI_COMPRESS_KERNEL_HPP
#define HSS_MATRIX_MPI_COMPRESS_KERNEL_HPP

#include "misc/RandomWrapper.hpp"
#include "DistSamples.hpp"
#include "DistElemMult.hpp"

namespace strumpack {
  namespace HSS {

    //NEW kernel routine: start
    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::compress_ann
    (DenseMatrix<std::size_t>& ann, DenseM_t& scores, const delem_t& Aelem,
     const opts_t& opts) {
      TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_compress);
      auto Aelemw = [&]
        (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
         DistM_t& B, const DistM_t& A, std::size_t rlo, std::size_t clo,
         MPI_Comm comm) {
        Aelem(I, J, B);
      };
      WorkCompressMPIANN<scalar_t> w;
      compress_recursive_ann(ann, scores, Aelemw, w, opts, grid_local());
    }
    //NEW kernel routine: end

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_recursive_ann
    (DenseMatrix<std::size_t>& ann, DenseM_t& scores, const delemw_t& Aelem,
     WorkCompressMPIANN<scalar_t>& w, const opts_t& opts,
     const BLACSGrid* lg) {
      if (!this->active()) return;
      if (this->leaf()) {
        if (this->is_untouched()) {
          std::vector<std::size_t> I, J;
          I.reserve(this->rows());
          J.reserve(this->cols());
          for (std::size_t i=0; i<this->rows(); i++)
            I.push_back(i+w.offset.first);
          for (std::size_t j=0; j<this->cols(); j++)
            J.push_back(j+w.offset.second);
          _D = DistM_t(grid(), this->rows(), this->cols());
          Aelem(I, J, _D, _A, 0, 0, comm());
        }
      } else {
        w.split(this->_ch[0]->dims());
        this->_ch[0]->compress_recursive_ann
          (ann, scores, Aelem, w.c[0], opts, lg);
        this->_ch[1]->compress_recursive_ann
          (ann, scores, Aelem, w.c[1], opts, lg);
        communicate_child_data_ann(w);
        if (!this->_ch[0]->is_compressed() ||
            !this->_ch[1]->is_compressed()) return;
        if (this->is_untouched()) {
          _B01 = DistM_t(grid(), w.c[0].Ir.size(), w.c[1].Ic.size());
          _B10 = DistM_t(grid(), w.c[1].Ir.size(), w.c[0].Ic.size());
          Aelem(w.c[0].Ir, w.c[1].Ic, _B01, _A01, 0, 0, comm());
          Aelem(w.c[1].Ir, w.c[0].Ic, _B10, _A10, 0, 0, comm());
        }
      }
      if (w.lvl == 0)
        this->_U_state = this->_V_state = State::COMPRESSED;
      else {
        compute_local_samples_ann(ann, scores, w, Aelem, opts);
        compute_U_V_bases_ann(w.S, opts, w);
        this->_U_state = this->_V_state = State::COMPRESSED;
      }
    }

    // Main differences here
    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compute_local_samples_ann
    (DenseMatrix<std::size_t>& ann, DenseM_t& scores,
     WorkCompressMPIANN<scalar_t>& w, const delemw_t& Aelem,
     const opts_t& opts) {
      std::size_t ann_number = ann.rows();
      std::vector<std::size_t> I;
      if (this->leaf()) {
        I.reserve(this->rows());
        for (std::size_t i=0; i<this->rows(); i++)
          I.push_back(i+w.offset.first);

        for (std::size_t i=w.offset.first;
             i<w.offset.first+this->rows(); i++)
          for (std::size_t j=0; j<ann_number; j++)
            if ((ann(j, i) < w.offset.first) ||
                (ann(j, i) >= w.offset.first + this->rows()))
              w.ids_scores.emplace_back(ann(j, i), scores(j, i));
      } else {
        I.reserve(w.c[0].Ir.size() + w.c[1].Ir.size());
        for (std::size_t i=0; i<w.c[0].Ir.size(); i++)
          I.push_back(w.c[0].Ir[i]);
        for (std::size_t i=0; i<w.c[1].Ir.size(); i++)
          I.push_back(w.c[1].Ir[i]);

        for (std::size_t i=0; i<w.c[0].ids_scores.size(); i++)
          if ((w.c[0].ids_scores[i].first < w.offset.first) ||
              (w.c[0].ids_scores[i].first >= w.offset.first + this->rows()))
            w.ids_scores.emplace_back(w.c[0].ids_scores[i]);
        for (std::size_t i=0; i<w.c[1].ids_scores.size(); i++)
          if ((w.c[1].ids_scores[i].first < w.offset.first) ||
              (w.c[1].ids_scores[i].first >= w.offset.first + this->rows()))
            w.ids_scores.emplace_back(w.c[1].ids_scores[i]);
      }

      // sort on column indices first, then on scores
      std::sort(w.ids_scores.begin(), w.ids_scores.end());

      // remove duplicate indices, keep only first entry of
      // duplicates, which is the one with the highest score, because
      // of the above sort
      w.ids_scores.erase
        (std::unique(w.ids_scores.begin(), w.ids_scores.end(),
                     [](const std::pair<std::size_t,double>& a,
                        const std::pair<std::size_t,double>& b) {
                       return a.first == b.first; }), w.ids_scores.end());

      // maximum number of samples
      std::size_t d_max = this->leaf() ?
        I.size() + opts.dd() :   // leaf size + some oversampling
        w.c[0].Ir.size() + w.c[1].Ir.size() + opts.dd();
      auto d = std::min(w.ids_scores.size(), d_max);

      std::vector<std::size_t> Scolids;
      if (d < w.ids_scores.size()) {
        // sort based on scores, keep only d_max closest
        std::sort(w.ids_scores.begin(), w.ids_scores.end(),
                  [](const std::pair<std::size_t,double>& a,
                     const std::pair<std::size_t,double>& b) {
                    return a.second < b.second; });
        w.ids_scores.resize(d);
      }
      Scolids.reserve(d);
      for (std::size_t j=0; j<d; j++)
        Scolids.push_back(w.ids_scores[j].first);
      w.S = DistM_t(grid(), I.size(), Scolids.size());
      Aelem(I, Scolids, w.S, _A, 0, 0, comm());
    }

    template<typename scalar_t> bool
    HSSMatrixMPI<scalar_t>::compute_U_V_bases_ann
    (DistM_t& S, const opts_t& opts, WorkCompressMPIANN<scalar_t>& w) {
      auto rtol = opts.rel_tol() / w.lvl;
      auto atol = opts.abs_tol() / w.lvl;
      auto gT = grid()->transpose();
      DistM_t wSr(S);
      wSr.ID_row(_U.E(), _U.P(), w.Jr, rtol, atol, &gT);
      _V.E() = _U.E();
      _V.P() = _U.P();
      w.Jc = w.Jr;
      STRUMPACK_ID_FLOPS(ID_row_flops(S, w.Jr.size()));
      STRUMPACK_ID_FLOPS(ID_row_flops(S, w.Jc.size()));
      notify_inactives_J(w);
      bool accurate = true;
      int d = S.cols();
      if (!(d - opts.p() >= opts.max_rank() ||
            (int(w.Jr.size()) <= d - opts.p() &&
             int(w.Jc.size()) <= d - opts.p()))) {
        accurate = false;
        std::cout << "WARNING: ID did not reach required accuracy:"
                  << "\t increase k (number of ANN's), or Delta_d."
                  << std::endl;
      }
      this->_U_rank = w.Jr.size();  this->_U_rows = S.rows();
      this->_V_rank = w.Jc.size();  this->_V_rows = S.rows();
      w.Ir.reserve(w.Jr.size());
      w.Ic.reserve(w.Jc.size());
      if (this->leaf()) {
        for (auto i : w.Jr) w.Ir.push_back(w.offset.first + i);
        for (auto j : w.Jc) w.Ic.push_back(w.offset.second + j);
      } else {
        auto r0 = w.c[0].Ir.size();
        for (auto i : w.Jr)
          w.Ir.push_back((i < r0) ? w.c[0].Ir[i] : w.c[1].Ir[i-r0]);
        r0 = w.c[0].Ic.size();
        for (auto j : w.Jc)
          w.Ic.push_back((j < r0) ? w.c[0].Ic[j] : w.c[1].Ic[j-r0]);
      }
      // TODO clear w.c[0].Ir, w.c[1].Ir, w.c[0].Ic, w.c[1].Ic
      return accurate;
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::communicate_child_data_ann
    (WorkCompressMPIANN<scalar_t>& w) {
      int rank = Comm().rank(), P = Ptotal(), root1 = Pl();
      int P0active = this->_ch[0]->Pactive();
      int P1active = this->_ch[1]->Pactive();
      std::vector<MPIRequest> sreq;
      std::vector<std::size_t> sbuf0, sbuf1;
      assert(sizeof(typename decltype(w.ids_scores)::value_type::second_type)
             == sizeof(std::size_t));
      if (rank < P0active) {
        if (rank < (P-P0active)) {
          // I'm one of the first P-P0active processes that are active
          // on child0, so I need to send to one or more others which
          // are not active on child0, ie the ones in [P0active,P)
          sbuf0.reserve(7+w.c[0].Ir.size()+w.c[0].Ic.size()+
                        w.c[0].Jr.size()+w.c[0].Jc.size()+
                        2*w.c[0].ids_scores.size());
          sbuf0.push_back(std::size_t(this->_ch[0]->_U_state));
          sbuf0.push_back(std::size_t(this->_ch[0]->_V_state));
          sbuf0.push_back(this->_ch[0]->_U_rank);
          sbuf0.push_back(this->_ch[0]->_V_rank);
          sbuf0.push_back(this->_ch[0]->_U_rows);
          sbuf0.push_back(this->_ch[0]->_V_rows);
          sbuf0.push_back(w.c[0].ids_scores.size());
          for (auto i : w.c[0].Ir) sbuf0.push_back(i);
          for (auto i : w.c[0].Ic) sbuf0.push_back(i);
          for (auto i : w.c[0].Jr) sbuf0.push_back(i);
          for (auto i : w.c[0].Jc) sbuf0.push_back(i);
          for (auto i : w.c[0].ids_scores) sbuf0.push_back(i.first);
          for (auto i : w.c[0].ids_scores)
            sbuf0.push_back(reinterpret_cast<std::size_t&>(i.second));
          for (int p=P0active; p<P; p++)
            if (rank == (p - P0active) % P0active)
              sreq.emplace_back(Comm().isend(sbuf0, p, 0));
        }
      }
      if (rank >= root1 && rank < root1+P1active) {
        if ((rank-root1) < (P-P1active)) {
          // I'm one of the first P-P1active processes that are active
          // on child1, so I need to send to one or more others which
          // are not active on child1, ie the ones in [0,root1) union
          // [root1+P1active,P)
          sbuf1.reserve(7+w.c[1].Ir.size()+w.c[1].Ic.size()+
                        w.c[1].Jr.size()+w.c[1].Jc.size()+
                        2*w.c[1].ids_scores.size());
          sbuf1.push_back(std::size_t(this->_ch[1]->_U_state));
          sbuf1.push_back(std::size_t(this->_ch[1]->_V_state));
          sbuf1.push_back(this->_ch[1]->_U_rank);
          sbuf1.push_back(this->_ch[1]->_V_rank);
          sbuf1.push_back(this->_ch[1]->_U_rows);
          sbuf1.push_back(this->_ch[1]->_V_rows);
          sbuf1.push_back(w.c[1].ids_scores.size());
          for (auto i : w.c[1].Ir) sbuf1.push_back(i);
          for (auto i : w.c[1].Ic) sbuf1.push_back(i);
          for (auto i : w.c[1].Jr) sbuf1.push_back(i);
          for (auto i : w.c[1].Jc) sbuf1.push_back(i);
          for (auto i : w.c[1].ids_scores) sbuf1.push_back(i.first);
          for (auto i : w.c[1].ids_scores)
            sbuf1.push_back(reinterpret_cast<std::size_t&>(i.second));
          for (int p=0; p<root1; p++)
            if (rank - root1 == p % P1active)
              sreq.emplace_back(Comm().isend(sbuf1, p, 1));
          for (int p=root1+P1active; p<P; p++)
            if (rank - root1 == (p - P1active) % P1active)
              sreq.emplace_back(Comm().isend(sbuf1, p, 1));
        }
      }
      if (!(rank < P0active)) {
        // I'm not active on child0, so I need to receive
        int dest = -1;
        for (int p=0; p<P0active; p++)
          if (p == (rank - P0active) % P0active) { dest = p; break; }
        assert(dest >= 0);
        auto buf = Comm().template recv<std::size_t>(dest, 0);
        auto ptr = buf.begin();
        this->_ch[0]->_U_state = State(*ptr++);
        this->_ch[0]->_V_state = State(*ptr++);
        this->_ch[0]->_U_rank = *ptr++;
        this->_ch[0]->_V_rank = *ptr++;
        this->_ch[0]->_U_rows = *ptr++;
        this->_ch[0]->_V_rows = *ptr++;
        auto d = *ptr++;
        w.c[0].Ir.resize(this->_ch[0]->_U_rank);
        w.c[0].Ic.resize(this->_ch[0]->_V_rank);
        w.c[0].Jr.resize(this->_ch[0]->_U_rank);
        w.c[0].Jc.resize(this->_ch[0]->_V_rank);
        w.c[0].ids_scores.resize(d);
        for (int i=0; i<this->_ch[0]->_U_rank; i++) w.c[0].Ir[i] = *ptr++;
        for (int i=0; i<this->_ch[0]->_V_rank; i++) w.c[0].Ic[i] = *ptr++;
        for (int i=0; i<this->_ch[0]->_U_rank; i++) w.c[0].Jr[i] = *ptr++;
        for (int i=0; i<this->_ch[0]->_V_rank; i++) w.c[0].Jc[i] = *ptr++;
        for (std::size_t i=0; i<d; i++) w.c[0].ids_scores[i].first = *ptr++;
        for (std::size_t i=0; i<d; i++)
          w.c[0].ids_scores[i].second = reinterpret_cast<double&>(*ptr++);
        //assert(msgsize == std::distance(buf, ptr));
      }
      if (!(rank >= root1 && rank < root1+P1active)) {
        // I'm not active on child1, so I need to receive
        int dest = -1;
        for (int p=root1; p<root1+P1active; p++) {
          if (rank < root1) {
            if (p - root1 == rank % P1active) { dest = p; break; }
          } else if (p - root1 == (rank - P1active) % P1active) {
            dest = p; break;
          }
        }
        assert(dest >= 0);
        auto buf = Comm().template recv<std::size_t>(dest, 1);
        auto ptr = buf.begin();
        this->_ch[1]->_U_state = State(*ptr++);
        this->_ch[1]->_V_state = State(*ptr++);
        this->_ch[1]->_U_rank = *ptr++;
        this->_ch[1]->_V_rank = *ptr++;
        this->_ch[1]->_U_rows = *ptr++;
        this->_ch[1]->_V_rows = *ptr++;
        auto d = *ptr++;
        w.c[1].Ir.resize(this->_ch[1]->_U_rank);
        w.c[1].Ic.resize(this->_ch[1]->_V_rank);
        w.c[1].Jr.resize(this->_ch[1]->_U_rank);
        w.c[1].Jc.resize(this->_ch[1]->_V_rank);
        w.c[1].ids_scores.resize(d);
        for (int i=0; i<this->_ch[1]->_U_rank; i++) w.c[1].Ir[i] = *ptr++;
        for (int i=0; i<this->_ch[1]->_V_rank; i++) w.c[1].Ic[i] = *ptr++;
        for (int i=0; i<this->_ch[1]->_U_rank; i++) w.c[1].Jr[i] = *ptr++;
        for (int i=0; i<this->_ch[1]->_V_rank; i++) w.c[1].Jc[i] = *ptr++;
        for (std::size_t i=0; i<d; i++) w.c[1].ids_scores[i].first = *ptr++;
        for (std::size_t i=0; i<d; i++)
          w.c[1].ids_scores[i].second = reinterpret_cast<double&>(*ptr++);
        //assert(msgsize == std::distance(buf, ptr));
      }
      wait_all(sreq);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::notify_inactives_J
    (WorkCompressMPIANN<scalar_t>& w) {
      // TODO reuse notify_inactives_J(WorkCompressMPI .. for this
      int rank = Comm().rank(), actives = Pactive();
      int inactives = grid()->P() - actives;
      if (rank < inactives) {
        std::vector<std::size_t> sbuf;
        sbuf.reserve(2+w.Jr.size()+w.Jc.size());
        sbuf.push_back(w.Jr.size());
        sbuf.push_back(w.Jc.size());
        for (auto i : w.Jr) sbuf.push_back(i);
        for (auto i : w.Jc) sbuf.push_back(i);
        Comm().send(sbuf, actives+rank, 0);
      }
      if (rank >= actives) {
        auto buf = Comm().template recv<std::size_t>(rank - actives, 0);
        auto ptr = buf.begin();
        w.Jr.resize(*ptr++);
        w.Jc.resize(*ptr++);
        for (std::size_t i=0; i<w.Jr.size(); i++) w.Jr[i] = *ptr++;
        for (std::size_t i=0; i<w.Jc.size(); i++) w.Jc[i] = *ptr++;
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_MPI_COMPRESS_kernel_HPP
