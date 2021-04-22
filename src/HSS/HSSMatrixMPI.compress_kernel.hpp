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

#include "clustering/NeighborSearch.hpp"

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::compress
    (const kernel::Kernel<real_t>& K, const opts_t& opts) {
      TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_compress);
      auto Aelemw = [&]
        (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
         DistM_t& B, const DistM_t& A, std::size_t rlo, std::size_t clo,
         MPI_Comm comm) {
        std::vector<std::size_t> lI, lJ;
        lI.reserve(B.lrows());
        lJ.reserve(B.lcols());
        for (size_t j=0; j<J.size(); j++)
          if (B.colg2p(j) == B.pcol())
            lJ.push_back(J[j]);
        for (size_t i=0; i<I.size(); i++)
          if (B.rowg2p(i) == B.prow())
            lI.push_back(I[i]);
        auto lB = B.dense_wrapper();
        K(lI, lJ, lB);
      };
      int ann_number = std::min(int(K.n()), opts.approximate_neighbors());
      while (!this->is_compressed()) {
        DenseMatrix<std::uint32_t> ann;
        DenseMatrix<real_t> scores;
        TaskTimer timer("approximate_neighbors");
        timer.start();
        find_approximate_neighbors
          (K.data(), opts.ann_iterations(), ann_number, ann, scores);
        if (opts.verbose() && Comm().is_root())
          std::cout << "# k-ANN=" << ann_number
                    << ", approximate neighbor search time = "
                    << timer.elapsed() << std::endl;
        WorkCompressMPIANN<scalar_t> w;
        compress_recursive_ann(ann, scores, Aelemw, w, opts, grid_local());
        ann_number = std::min(2*ann_number, int(K.n()));
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_recursive_ann
    (DenseMatrix<std::uint32_t>& ann, DenseMatrix<real_t>& scores,
     const delemw_t& Aelem, WorkCompressMPIANN<scalar_t>& w,
     const opts_t& opts, const BLACSGrid* lg) {
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
          D_ = DistM_t(grid(), this->rows(), this->cols());
          Aelem(I, J, D_, A_, 0, 0, comm());
        }
      } else {
        w.split(child(0)->dims());
        child(0)->compress_recursive_ann
          (ann, scores, Aelem, w.c[0], opts, lg);
        child(1)->compress_recursive_ann
          (ann, scores, Aelem, w.c[1], opts, lg);
        communicate_child_data_ann(w);
        if (!child(0)->is_compressed() ||
            !child(1)->is_compressed()) return;
        // TODO do not re-extract if children are not re-compressed
        // if (this->is_untouched()) {
        B01_ = DistM_t(grid(), w.c[0].Ir.size(), w.c[1].Ic.size());
        Aelem(w.c[0].Ir, w.c[1].Ic, B01_, A01_, 0, 0, comm());
        B10_ = B01_.transpose();
        //}
      }
      if (w.lvl == 0)
        this->U_state_ = this->V_state_ = State::COMPRESSED;
      else {
        // TODO only do this if not already compressed
        //if (!this->is_compressed()) {
        compute_local_samples_ann(ann, scores, w, Aelem, opts);
        if (compute_U_V_bases_ann(w.S, opts, w))
          this->U_state_ = this->V_state_ = State::COMPRESSED;
        w.c.clear();
      }
      w.c.clear();
      w.c.shrink_to_fit();
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compute_local_samples_ann
    (DenseMatrix<std::uint32_t>& ann, DenseMatrix<real_t>& scores,
     WorkCompressMPIANN<scalar_t>& w, const delemw_t& Aelem,
     const opts_t& opts) {
      std::size_t ann_number = ann.rows();
      std::vector<std::size_t> I;
      if (this->leaf()) {
        I.reserve(this->rows());
        for (std::size_t i=0; i<this->rows(); i++)
          I.push_back(i+w.offset.first);

        w.ids_scores.reserve(this->rows()*ann_number);
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

        w.ids_scores.reserve(w.c[0].ids_scores.size()+
                             w.c[1].ids_scores.size());
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
                     [](const std::pair<std::size_t,real_t>& a,
                        const std::pair<std::size_t,real_t>& b) {
                       return a.first == b.first; }), w.ids_scores.end());

#if 0 // drop some columns
      // maximum number of samples
      std::size_t d_max = this->leaf() ?
        I.size() + opts.dd() :   // leaf size + some oversampling
        w.c[0].Ir.size() + w.c[1].Ir.size() + opts.dd();
      auto d = std::min(w.ids_scores.size(), d_max);

      if (d < w.ids_scores.size()) {
        // sort based on scores, keep only d closest
        std::nth_element
          (w.ids_scores.begin(), w.ids_scores.begin()+d, w.ids_scores.end(),
           [](const std::pair<std::size_t,real_t>& a,
              const std::pair<std::size_t,real_t>& b) {
            return a.second < b.second; });
        w.ids_scores.resize(d);
      }
#else
      auto d = w.ids_scores.size();
#endif
      std::vector<std::size_t> Scolids;
      Scolids.reserve(d);
      for (std::size_t j=0; j<d; j++)
        Scolids.push_back(w.ids_scores[j].first);
      w.S = DistM_t(grid(), I.size(), Scolids.size());
      Aelem(I, Scolids, w.S, A_, 0, 0, comm());
    }

    template<typename scalar_t> bool
    HSSMatrixMPI<scalar_t>::compute_U_V_bases_ann
    (DistM_t& S, const opts_t& opts, WorkCompressMPIANN<scalar_t>& w) {
      auto rtol = opts.rel_tol() / w.lvl;
      auto atol = opts.abs_tol() / w.lvl;
      auto gT = grid()->transpose();
      DistM_t wSr(S);
      wSr.ID_row(U_.E(), U_.P(), w.Jr, rtol, atol, opts.max_rank(), &gT);
      V_.E() = U_.E();
      V_.P() = U_.P();
      w.Jc = w.Jr;
      notify_inactives_J(w);
      STRUMPACK_ID_FLOPS(ID_row_flops(S, w.Jr.size()));
      int d = S.cols();
      if (!(d >= int(this->cols()) || d >= opts.max_rank() ||
            (int(w.Jr.size()) + opts.p() < d  &&
             int(w.Jc.size()) + opts.p() < d))) {
        // std::cout << "WARNING: ID did not reach required accuracy:"
        //           << "\t increase k (number of ANN's), or Delta_d."
        //           << std::endl;
        return false;
      }
      this->U_rank_ = w.Jr.size();  this->U_rows_ = S.rows();
      this->V_rank_ = w.Jc.size();  this->V_rows_ = S.rows();
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
      return true;
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::communicate_child_data_ann
    (WorkCompressMPIANN<scalar_t>& w) {
      int rank = Comm().rank(), P = Ptotal(), root1 = Pl();
      int P0active = child(0)->Pactive();
      int P1active = child(1)->Pactive();
      std::vector<MPIRequest> sreq;
      std::vector<std::size_t> sbuf0, sbuf1;
      std::vector<real_t> sbuf0_scalar, sbuf1_scalar;
      if (rank < P0active) {
        if (rank < (P-P0active)) {
          // I'm one of the first P-P0active processes that are active
          // on child0, so I need to send to one or more others which
          // are not active on child0, ie the ones in [P0active,P)
          sbuf0.reserve(7+w.c[0].Ir.size()+w.c[0].Ic.size()+
                        w.c[0].Jr.size()+w.c[0].Jc.size()+
                        w.c[0].ids_scores.size());
          sbuf0_scalar.reserve(w.c[0].ids_scores.size());
          sbuf0.push_back(std::size_t(child(0)->U_state_));
          sbuf0.push_back(std::size_t(child(0)->V_state_));
          sbuf0.push_back(child(0)->U_rank_);
          sbuf0.push_back(child(0)->V_rank_);
          sbuf0.push_back(child(0)->U_rows_);
          sbuf0.push_back(child(0)->V_rows_);
          sbuf0.push_back(w.c[0].ids_scores.size());
          for (auto i : w.c[0].Ir) sbuf0.push_back(i);
          for (auto i : w.c[0].Ic) sbuf0.push_back(i);
          for (auto i : w.c[0].Jr) sbuf0.push_back(i);
          for (auto i : w.c[0].Jc) sbuf0.push_back(i);
          for (auto i : w.c[0].ids_scores) {
            sbuf0.push_back(i.first);
            sbuf0_scalar.push_back(i.second);
          }
          for (int p=P0active; p<P; p++)
            if (rank == (p - P0active) % P0active) {
              sreq.emplace_back(Comm().isend(sbuf0, p, 0));
              sreq.emplace_back(Comm().isend(sbuf0_scalar, p, 2));
            }
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
                        w.c[1].ids_scores.size());
          sbuf1_scalar.reserve(w.c[1].ids_scores.size());
          sbuf1.push_back(std::size_t(child(1)->U_state_));
          sbuf1.push_back(std::size_t(child(1)->V_state_));
          sbuf1.push_back(child(1)->U_rank_);
          sbuf1.push_back(child(1)->V_rank_);
          sbuf1.push_back(child(1)->U_rows_);
          sbuf1.push_back(child(1)->V_rows_);
          sbuf1.push_back(w.c[1].ids_scores.size());
          for (auto i : w.c[1].Ir) sbuf1.push_back(i);
          for (auto i : w.c[1].Ic) sbuf1.push_back(i);
          for (auto i : w.c[1].Jr) sbuf1.push_back(i);
          for (auto i : w.c[1].Jc) sbuf1.push_back(i);
          for (auto i : w.c[1].ids_scores) {
            sbuf1.push_back(i.first);
            sbuf1_scalar.push_back(i.second);
          }
          for (int p=0; p<root1; p++)
            if (rank - root1 == p % P1active) {
              sreq.emplace_back(Comm().isend(sbuf1, p, 1));
              sreq.emplace_back(Comm().isend(sbuf1_scalar, p, 3));
            }
          for (int p=root1+P1active; p<P; p++)
            if (rank - root1 == (p - P1active) % P1active) {
              sreq.emplace_back(Comm().isend(sbuf1, p, 1));
              sreq.emplace_back(Comm().isend(sbuf1_scalar, p, 3));
            }
        }
      }
      if (!(rank < P0active)) {
        // I'm not active on child0, so I need to receive
        int dest = -1;
        for (int p=0; p<P0active; p++)
          if (p == (rank - P0active) % P0active) { dest = p; break; }
        assert(dest >= 0);
        auto buf = Comm().template recv<std::size_t>(dest, 0);
        auto buf_scalar = Comm().template recv<real_t>(dest, 2);
        auto ptr = buf.begin();
        auto ptr_scalar = buf_scalar.begin();
        child(0)->U_state_ = State(*ptr++);
        child(0)->V_state_ = State(*ptr++);
        child(0)->U_rank_ = *ptr++;
        child(0)->V_rank_ = *ptr++;
        child(0)->U_rows_ = *ptr++;
        child(0)->V_rows_ = *ptr++;
        auto d = *ptr++;
        w.c[0].Ir.resize(child(0)->U_rank_);
        w.c[0].Ic.resize(child(0)->V_rank_);
        w.c[0].Jr.resize(child(0)->U_rank_);
        w.c[0].Jc.resize(child(0)->V_rank_);
        w.c[0].ids_scores.resize(d);
        for (int i=0; i<child(0)->U_rank_; i++) w.c[0].Ir[i] = *ptr++;
        for (int i=0; i<child(0)->V_rank_; i++) w.c[0].Ic[i] = *ptr++;
        for (int i=0; i<child(0)->U_rank_; i++) w.c[0].Jr[i] = *ptr++;
        for (int i=0; i<child(0)->V_rank_; i++) w.c[0].Jc[i] = *ptr++;
        for (std::size_t i=0; i<d; i++) {
          w.c[0].ids_scores[i].first = *ptr++;
          w.c[0].ids_scores[i].second = *ptr_scalar++;
        }
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
        auto buf_scalar = Comm().template recv<real_t>(dest, 3);
        auto ptr = buf.begin();
        auto ptr_scalar = buf_scalar.begin();
        child(1)->U_state_ = State(*ptr++);
        child(1)->V_state_ = State(*ptr++);
        child(1)->U_rank_ = *ptr++;
        child(1)->V_rank_ = *ptr++;
        child(1)->U_rows_ = *ptr++;
        child(1)->V_rows_ = *ptr++;
        auto d = *ptr++;
        w.c[1].Ir.resize(child(1)->U_rank_);
        w.c[1].Ic.resize(child(1)->V_rank_);
        w.c[1].Jr.resize(child(1)->U_rank_);
        w.c[1].Jc.resize(child(1)->V_rank_);
        w.c[1].ids_scores.resize(d);
        for (int i=0; i<child(1)->U_rank_; i++) w.c[1].Ir[i] = *ptr++;
        for (int i=0; i<child(1)->V_rank_; i++) w.c[1].Ic[i] = *ptr++;
        for (int i=0; i<child(1)->U_rank_; i++) w.c[1].Jr[i] = *ptr++;
        for (int i=0; i<child(1)->V_rank_; i++) w.c[1].Jc[i] = *ptr++;
        for (std::size_t i=0; i<d; i++) {
          w.c[1].ids_scores[i].first = *ptr++;
          w.c[1].ids_scores[i].second = *ptr_scalar++;
        }
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
