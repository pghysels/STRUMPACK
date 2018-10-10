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

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_kernel_nosync_MPI
    (DenseM_t& ann, DenseM_t& scores, const delem_t& Aelem,
    const opts_t& opts) {
      // std::cout << "compress_recursive_ann" << std::endl;
      auto d = opts.d0();
      auto dd = opts.dd();
      WorkCompressMPI_ANN<scalar_t> w_mpi;
      compress_recursive_ann(ann, scores, Aelem, w_mpi, d, dd, opts );
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_recursive_ann
    (DenseM_t& ann, DenseM_t& scores, const delem_t& Aelem,
    WorkCompressMPI_ANN<scalar_t>& w_mpi, int d, int dd, const opts_t& opts) {
      std::cout << "compress_recursive_ann_dist" << std::endl;
      if (!this->active()) return;
      if (this->leaf()) {
        std::cout << "LEAF" << std::endl;
        if (this->is_untouched()) {
          std::vector<std::size_t> I, J;
          I.reserve(this->rows());
          J.reserve(this->cols());
          for (std::size_t i=0; i<this->rows(); i++)
            I.push_back(i+w_mpi.offset.first);
          for (std::size_t j=0; j<this->cols(); j++)
            J.push_back(j+w_mpi.offset.second);
          _D = DistM_t(grid(), this->rows(), this->cols());
          Aelem(I, J, _D);
          // Aelem(I, J, _D, _A, w.offset.first, w.offset.second, comm());
        }
      }
      else {
        std::cout << "NLEF(" << this->rows() << "," << this->cols() << ")\n";
        w_mpi.split(this->_ch[0]->dims());
        this->_ch[0]->compress_recursive_ann
          (ann, scores, Aelem, w_mpi.c[0], d, dd, opts);
        this->_ch[1]->compress_recursive_ann
          (ann, scores, Aelem, w_mpi.c[1], d, dd, opts);
        // communicate_child_data_kernel(w); //   <-- Needs major modifications
        if (!this->_ch[0]->is_compressed() ||
            !this->_ch[1]->is_compressed()) return;

        if (this->is_untouched()) {
          _B01 = DistM_t(grid(), w_mpi.c[0].Ir.size(), w_mpi.c[1].Ic.size());
          _B10 = DistM_t(grid(), w_mpi.c[1].Ir.size(), w_mpi.c[0].Ic.size());
          Aelem(w_mpi.c[0].Ir, w_mpi.c[1].Ic, _B01);
          Aelem(w_mpi.c[1].Ir, w_mpi.c[0].Ic, _B10);
          // Aelem(w.c[0].Ir, w.c[1].Ic, _B01, _A01,
          //       w.offset.first, w.offset.second+this->_ch[0]->cols(), comm());
          // Aelem(w.c[1].Ir, w.c[0].Ic, _B10, _A10,
          //       w.offset.first+this->_ch[0]->rows(), w.offset.second, comm());
        }
      }

      if (w_mpi.lvl == 0)
        this->_U_state = this->_V_state = State::COMPRESSED;
      else {
        if (!this->is_compressed()) {
          compute_local_samples_kernel_MPI(ann, scores, w_mpi, Aelem, opts);
          // compute_U_basis_kernel(opts, w_mpi, d, dd);
          // compute_V_basis_kernel(opts, w_mpi, d, dd);
          // notify_inactives_states_kernel(w_mpi);
          // reduce_local_samples_kernel_MPI(RS, w_mpi, d+dd, false);
        }
      }
    }

    // Main differences here
    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compute_local_samples_kernel_MPI
    (DenseM_t &ann, DenseM_t &scores, WorkCompressMPI_ANN<scalar_t> &w_mpi,
    const delem_t &Aelem, const opts_t &opts) {
      std::cout << "compute_local_samples_kernel_MPI";

      // std::size_t ann_number = ann.rows();
      // if (this->leaf()) {
      //   std::vector<std::size_t> I;
      //   I.reserve(this->rows());
      //   for (std::size_t i=0; i<this->rows(); i++)
      //     I.push_back(i+w.offset.first);

      //   // combine non-diagonal neibs of all points in the leaf
      //   std::vector<std::size_t> leaf_neibs;
      //   std::vector<double> leaf_neib_scores;
      //   for (std::size_t i = w.offset.first; i < w.offset.first + this->rows(); i++) {
      //     for (std::size_t j = 0; j < ann_number; j++) {
      //       if ((ann(j, i) < w.offset.first) ||
      //           (ann(j, i) >= w.offset.first + this->rows())) {
      //         leaf_neibs.push_back(ann(j, i));
      //         leaf_neib_scores.push_back(scores(j, i));
      //       }
      //     }
      //   }

      //   // sort column indices and corresponding scores
      //   std::vector<std::size_t> order = find_sort_permutation(leaf_neibs);
      //   leaf_neibs = apply_permutation(leaf_neibs, order);
      //   leaf_neib_scores = apply_permutation(leaf_neib_scores, order);

      //   // remove duplicates
      //   std::size_t cur = 0;
      //   for (std::size_t i = 1; i < leaf_neibs.size(); i++) {
      //     if (leaf_neibs[i] > leaf_neibs[i-1]) {
      //       cur++;
      //       leaf_neibs[cur] = leaf_neibs[i];
      //       leaf_neib_scores[cur] = leaf_neib_scores[i];
      //     } else {
      //       // keep the smallest score
      //       if (leaf_neib_scores[cur] > leaf_neib_scores[i])
      //         leaf_neib_scores[cur] = leaf_neib_scores[i];
      //     }
      //   }
      //   leaf_neibs.resize(cur+1);
      //   leaf_neib_scores.resize(cur+1);

      //   // maximum number of samples is leaf size + some oversampling
      //   int d_max = I.size() + opts.dd();
      //   if (leaf_neibs.size() < d_max) {
      //     for (std::size_t j = 0; j < leaf_neibs.size(); j++) {
      //       w.Scolids.push_back(leaf_neibs[j]);
      //       w.Scolscs.push_back(leaf_neib_scores[j]);
      //     }
      //   } else {
      //     // sort based on scores
      //     std::vector<std::size_t> order = find_sort_permutation(leaf_neib_scores);
      //     leaf_neibs = apply_permutation(leaf_neibs, order);
      //     leaf_neib_scores = apply_permutation(leaf_neib_scores, order);
      //     // keep only d_max closest
      //     for (std::size_t j = 0; j < d_max; j++) {
      //       w.Scolids.push_back(leaf_neibs[j]);
      //       w.Scolscs.push_back(leaf_neib_scores[j]);
      //     }
      //   }
      //   w.S = DenseM_t(I.size(), w.Scolids.size());
      //   Aelem(I, w.Scolids, w.S);
      // }
      // else {
      //   std::vector<std::size_t> I;
      //   for (std::size_t i = 0; i < w.c[0].Ir.size(); i++)
      //     I.push_back(w.c[0].Ir[i]);
      //   for (std::size_t i = 0; i < w.c[1].Ir.size(); i++)
      //     I.push_back(w.c[1].Ir[i]);

      //   std::vector<std::size_t> colids;
      //   std::vector<double> colscs;
      //   for (std::size_t i = 0; i < w.c[0].Scolids.size(); i++) {
      //     if ((w.c[0].Scolids[i] < w.offset.first) ||
      //         (w.c[0].Scolids[i] >= w.offset.first + this->rows())) {
      //       colids.push_back(w.c[0].Scolids[i]);
      //       colscs.push_back(w.c[0].Scolscs[i]);
      //     }
      //   }
      //   for (std::size_t i = 0; i < w.c[1].Scolids.size(); i++) {
      //     if ((w.c[1].Scolids[i] < w.offset.first) ||
      //         (w.c[1].Scolids[i] >= w.offset.first + this->rows())) {
      //       colids.push_back(w.c[1].Scolids[i]);
      //       colscs.push_back(w.c[1].Scolscs[i]);
      //     }
      //   }

      //   // sort column indices and corresponding scores
      //   std::vector<std::size_t> order = find_sort_permutation(colids);
      //   colids = apply_permutation(colids, order);
      //   colscs = apply_permutation(colscs, order);

      //   // remove duplicate column indices
      //   std::size_t cur = 0;
      //   for (std::size_t i = 1; i < colids.size(); i++) {
      //     if (colids[i] > colids[i-1]) {
      //       cur++;
      //       colids[cur] = colids[i];
      //       colscs[cur] = colscs[i];
      //     } else {
      //       // keep the smallest score
      //       if (colscs[cur] > colscs[i])
      //         colscs[cur] = colscs[i];
      //     }
      //   }
      //   colids.resize(cur+1);
      //   colscs.resize(cur+1);

      //   int d_max = w.c[0].Ir.size() + w.c[1].Ir.size() + opts.dd();
      //   if (colids.size() < d_max) {
      //       for (std::size_t j = 0; j < colids.size(); j++) {
      //         // if we want to add more samples until d, it is here
      //         w.Scolids.push_back(colids[j]);
      //         w.Scolscs.push_back(colscs[j]);
      //       }
      //   } else {
      //     // sort based on scores
      //     std::vector<std::size_t> order = find_sort_permutation(colscs);
      //     colids = apply_permutation(colids, order);
      //     colscs = apply_permutation(colscs, order);
      //     // keep only d closest
      //     for (std::size_t j = 0; j < d_max; j++) {
      //       w.Scolids.push_back(colids[j]);
      //       w.Scolscs.push_back(colscs[j]);
      //     }
      //   }
      //   w.S = DenseM_t(I.size(), w.Scolids.size());
      //   Aelem(I, w.Scolids, w.S);
      // }

    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::reduce_local_samples_kernel_MPI
    (const DistSamples<scalar_t>& RS, WorkCompressMPI_ANN<scalar_t>& w_mpi,
    int dd, bool was_compressed) {
      std::cout << "reduce_local_samples_kernel_MPI";
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compute_U_basis_kernel_MPI
    (const opts_t& opts, WorkCompressMPI_ANN<scalar_t>& w_mpi, int d, int dd) {
      std::cout << "compute_U_basis_kernel";
      if (this->_U_state == State::COMPRESSED) return;
      // int u_rows = this->leaf() ? this->rows() :
      //   this->_ch[0]->U_rank()+this->_ch[1]->U_rank();
      // auto gT = grid()->transpose();
      // if (!w.Sr.active()) return;
      // if (d+dd >= opts.max_rank() || d+dd >= u_rows ||
      //     update_orthogonal_basis_kernel_MPI
      //     (opts, w.U_r_max, w.Sr, w.Qr, d, dd,
      //      this->_U_state == State::UNTOUCHED, w.lvl)) {
      //   w.Qr.clear();
      //   // TODO pass max_rank to ID in DistributedMatrix
      //   auto rtol = opts.rel_tol() / w.lvl;
      //   auto atol = opts.abs_tol() / w.lvl;
      //   w.Sr.ID_row(_U.E(), _U.P(), w.Jr, rtol, atol, &gT);
      //   STRUMPACK_ID_FLOPS(ID_row_flops(w.Sr, _U.cols()));
      //   this->_U_rank = _U.cols();
      //   this->_U_rows = _U.rows();
      //   w.Ir.reserve(_U.cols());
      //   if (this->leaf())
      //     for (auto i : w.Jr) w.Ir.push_back(w.offset.first + i);
      //   else {
      //     auto r0 = w.c[0].Ir.size();
      //     for (auto i : w.Jr)
      //       w.Ir.push_back((i < r0) ? w.c[0].Ir[i] : w.c[1].Ir[i-r0]);
      //   }
      //   this->_U_state = State::COMPRESSED;
      // } else this->_U_state = State::PARTIALLY_COMPRESSED;
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compute_V_basis_kernel_MPI
    (const opts_t& opts, WorkCompressMPI_ANN<scalar_t>& w_mpi, int d, int dd) {
      std::cout << "compute_V_basis_kernel";
      if (this->_V_state == State::COMPRESSED) return;
      // int v_rows = this->leaf() ? this->rows() :
      //   this->_ch[0]->V_rank()+this->_ch[1]->V_rank();
      // auto gT = grid()->transpose();
      // if (!w.Sc.active()) return;
      // if (d+dd >= opts.max_rank() || d+dd >= v_rows ||
      //     update_orthogonal_basis_kernel_MPI
      //     (opts, w.V_r_max, w.Sc, w.Qc, d, dd,
      //      this->_V_state == State::UNTOUCHED, w.lvl)) {
      //   w.Qc.clear();
      //   // TODO pass max_rank to ID in DistributedMatrix
      //   auto rtol = opts.rel_tol() / w.lvl;
      //   auto atol = opts.abs_tol() / w.lvl;
      //   w.Sc.ID_row(_V.E(), _V.P(), w.Jc, rtol, atol, &gT);
      //   STRUMPACK_ID_FLOPS(ID_row_flops(w.Sc, _V.cols()));
      //   this->_V_rank = _V.cols();
      //   this->_V_rows = _V.rows();
      //   w.Ic.reserve(_V.cols());
      //   if (this->leaf())
      //     for (auto j : w.Jc) w.Ic.push_back(w.offset.second + j);
      //   else {
      //     auto r0 = w.c[0].Ic.size();
      //     for (auto j : w.Jc)
      //       w.Ic.push_back((j < r0) ? w.c[0].Ic[j] : w.c[1].Ic[j-r0]);
      //   }
      //   this->_V_state = State::COMPRESSED;
      // } else this->_V_state = State::PARTIALLY_COMPRESSED;
    }

    template<typename scalar_t> bool
    HSSMatrixMPI<scalar_t>::update_orthogonal_basis_kernel_MPI() {
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::communicate_child_data_kernel_MPI
    (WorkCompressMPI_ANN<scalar_t>& w_mpi) {
      // w.c[0].dR = w.c[0].Rr.cols();
      // w.c[0].dS = w.c[0].Sr.cols();
      // w.c[1].dR = w.c[1].Rr.cols();
      // w.c[1].dS = w.c[1].Sr.cols();
      // int rank = Comm().rank(), P = Ptotal(), root1 = Pl();
      // int P0active = this->_ch[0]->Pactive();
      // int P1active = this->_ch[1]->Pactive();
      // std::vector<MPIRequest> sreq;
      // std::vector<std::size_t> sbuf0, sbuf1;
      // if (rank < P0active) {
      //   if (rank < (P-P0active)) {
      //     // I'm one of the first P-P0active processes that are active
      //     // on child0, so I need to send to one or more others which
      //     // are not active on child0, ie the ones in [P0active,P)
      //     sbuf0.reserve(8+w.c[0].Ir.size()+w.c[0].Ic.size()+
      //                   w.c[0].Jr.size()+w.c[0].Jc.size());
      //     sbuf0.push_back(std::size_t(this->_ch[0]->_U_state));
      //     sbuf0.push_back(std::size_t(this->_ch[0]->_V_state));
      //     sbuf0.push_back(this->_ch[0]->_U_rank);
      //     sbuf0.push_back(this->_ch[0]->_V_rank);
      //     sbuf0.push_back(this->_ch[0]->_U_rows);
      //     sbuf0.push_back(this->_ch[0]->_V_rows);
      //     sbuf0.push_back(w.c[0].dR);
      //     sbuf0.push_back(w.c[0].dS);
      //     for (auto i : w.c[0].Ir) sbuf0.push_back(i);
      //     for (auto i : w.c[0].Ic) sbuf0.push_back(i);
      //     for (auto i : w.c[0].Jr) sbuf0.push_back(i);
      //     for (auto i : w.c[0].Jc) sbuf0.push_back(i);
      //     for (int p=P0active; p<P; p++)
      //       if (rank == (p - P0active) % P0active)
      //         sreq.emplace_back(Comm().isend(sbuf0, p, 0));
      //   }
      // }
      // if (rank >= root1 && rank < root1+P1active) {
      //   if ((rank-root1) < (P-P1active)) {
      //     // I'm one of the first P-P1active processes that are active
      //     // on child1, so I need to send to one or more others which
      //     // are not active on child1, ie the ones in [0,root1) union
      //     // [root1+P1active,P)
      //     sbuf1.reserve(8+w.c[1].Ir.size()+w.c[1].Ic.size()+
      //                   w.c[1].Jr.size()+w.c[1].Jc.size());
      //     sbuf1.push_back(std::size_t(this->_ch[1]->_U_state));
      //     sbuf1.push_back(std::size_t(this->_ch[1]->_V_state));
      //     sbuf1.push_back(this->_ch[1]->_U_rank);
      //     sbuf1.push_back(this->_ch[1]->_V_rank);
      //     sbuf1.push_back(this->_ch[1]->_U_rows);
      //     sbuf1.push_back(this->_ch[1]->_V_rows);
      //     sbuf1.push_back(w.c[1].dR);
      //     sbuf1.push_back(w.c[1].dS);
      //     for (auto i : w.c[1].Ir) sbuf1.push_back(i);
      //     for (auto i : w.c[1].Ic) sbuf1.push_back(i);
      //     for (auto i : w.c[1].Jr) sbuf1.push_back(i);
      //     for (auto i : w.c[1].Jc) sbuf1.push_back(i);
      //     for (int p=0; p<root1; p++)
      //       if (rank - root1 == p % P1active)
      //         sreq.emplace_back(Comm().isend(sbuf1, p, 1));
      //     for (int p=root1+P1active; p<P; p++)
      //       if (rank - root1 == (p - P1active) % P1active)
      //         sreq.emplace_back(Comm().isend(sbuf1, p, 1));
      //   }
      // }
      // if (!(rank < P0active)) {
      //   // I'm not active on child0, so I need to receive
      //   int dest = -1;
      //   for (int p=0; p<P0active; p++)
      //     if (p == (rank - P0active) % P0active) { dest = p; break; }
      //   assert(dest >= 0);
      //   auto buf = Comm().template recv<std::size_t>(dest, 0);
      //   auto ptr = buf.begin();
      //   this->_ch[0]->_U_state = State(*ptr++);
      //   this->_ch[0]->_V_state = State(*ptr++);
      //   this->_ch[0]->_U_rank = *ptr++;
      //   this->_ch[0]->_V_rank = *ptr++;
      //   this->_ch[0]->_U_rows = *ptr++;
      //   this->_ch[0]->_V_rows = *ptr++;
      //   w.c[0].dR = *ptr++;
      //   w.c[0].dS = *ptr++;
      //   w.c[0].Ir.resize(this->_ch[0]->_U_rank);
      //   w.c[0].Ic.resize(this->_ch[0]->_V_rank);
      //   w.c[0].Jr.resize(this->_ch[0]->_U_rank);
      //   w.c[0].Jc.resize(this->_ch[0]->_V_rank);
      //   for (int i=0; i<this->_ch[0]->_U_rank; i++) w.c[0].Ir[i] = *ptr++;
      //   for (int i=0; i<this->_ch[0]->_V_rank; i++) w.c[0].Ic[i] = *ptr++;
      //   for (int i=0; i<this->_ch[0]->_U_rank; i++) w.c[0].Jr[i] = *ptr++;
      //   for (int i=0; i<this->_ch[0]->_V_rank; i++) w.c[0].Jc[i] = *ptr++;
      //   //assert(msgsize == std::distance(buf, ptr));
      // }
      // if (!(rank >= root1 && rank < root1+P1active)) {
      //   // I'm not active on child1, so I need to receive
      //   int dest = -1;
      //   for (int p=root1; p<root1+P1active; p++) {
      //     if (rank < root1) {
      //       if (p - root1 == rank % P1active) { dest = p; break; }
      //     } else if (p - root1 == (rank - P1active) % P1active) {
      //       dest = p; break;
      //     }
      //   }
      //   assert(dest >= 0);
      //   auto buf = Comm().template recv<std::size_t>(dest, 1);
      //   auto ptr = buf.begin();
      //   this->_ch[1]->_U_state = State(*ptr++);
      //   this->_ch[1]->_V_state = State(*ptr++);
      //   this->_ch[1]->_U_rank = *ptr++;
      //   this->_ch[1]->_V_rank = *ptr++;
      //   this->_ch[1]->_U_rows = *ptr++;
      //   this->_ch[1]->_V_rows = *ptr++;
      //   w.c[1].dR = *ptr++;
      //   w.c[1].dS = *ptr++;
      //   w.c[1].Ir.resize(this->_ch[1]->_U_rank);
      //   w.c[1].Ic.resize(this->_ch[1]->_V_rank);
      //   w.c[1].Jr.resize(this->_ch[1]->_U_rank);
      //   w.c[1].Jc.resize(this->_ch[1]->_V_rank);
      //   for (int i=0; i<this->_ch[1]->_U_rank; i++) w.c[1].Ir[i] = *ptr++;
      //   for (int i=0; i<this->_ch[1]->_V_rank; i++) w.c[1].Ic[i] = *ptr++;
      //   for (int i=0; i<this->_ch[1]->_U_rank; i++) w.c[1].Jr[i] = *ptr++;
      //   for (int i=0; i<this->_ch[1]->_V_rank; i++) w.c[1].Jc[i] = *ptr++;
      //   //assert(msgsize == std::distance(buf, ptr));
      // }
      // wait_all(sreq);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::notify_inactives_states_kernel_MPI
    (WorkCompressMPI_ANN<scalar_t>& w_mpi) {
      // int rank = Comm().rank(), actives = Pactive();
      // int inactives = Ptotal() - actives;
      // if (rank < inactives) {
      //   std::vector<std::size_t> sbuf;
      //   sbuf.reserve(8+w.Ir.size()+w.Ic.size()+w.Jr.size()+w.Jc.size());
      //   sbuf.push_back(std::size_t(this->_U_state));
      //   sbuf.push_back(std::size_t(this->_V_state));
      //   sbuf.push_back(this->_U_rank);
      //   sbuf.push_back(this->_V_rank);
      //   sbuf.push_back(this->_U_rows);
      //   sbuf.push_back(this->_V_rows);
      //   sbuf.push_back(w.Rr.cols());
      //   sbuf.push_back(w.Sr.cols());
      //   for (auto i : w.Ir) sbuf.push_back(i);
      //   for (auto i : w.Ic) sbuf.push_back(i);
      //   for (auto i : w.Jr) sbuf.push_back(i);
      //   for (auto i : w.Jc) sbuf.push_back(i);
      //   Comm().send(sbuf, actives+rank, 0);
      // }
      // if (rank >= actives) {
      //   auto buf = Comm().template recv<std::size_t>(rank-actives, 0);
      //   auto ptr = buf.begin();
      //   this->_U_state = State(*ptr++);
      //   this->_V_state = State(*ptr++);
      //   this->_U_rank = *ptr++;
      //   this->_V_rank = *ptr++;
      //   this->_U_rows = *ptr++;
      //   this->_V_rows = *ptr++;
      //   w.dR = *ptr++;
      //   w.dS = *ptr++;
      //   w.Ir.resize(this->_U_rank);
      //   w.Ic.resize(this->_V_rank);
      //   w.Jr.resize(this->_U_rank);
      //   w.Jc.resize(this->_V_rank);
      //   for (int i=0; i<this->_U_rank; i++) w.Ir[i] = *ptr++;
      //   for (int i=0; i<this->_V_rank; i++) w.Ic[i] = *ptr++;
      //   for (int i=0; i<this->_U_rank; i++) w.Jr[i] = *ptr++;
      //   for (int i=0; i<this->_V_rank; i++) w.Jc[i] = *ptr++;
      // }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_MPI_COMPRESS_kernel_HPP
