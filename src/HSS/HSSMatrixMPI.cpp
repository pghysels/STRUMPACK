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
#include <algorithm>

#include "HSSMatrixMPI.hpp"

#include "misc/RandomWrapper.hpp"
#include "misc/TaskTimer.hpp"
#include "misc/Tools.hpp"

#include "BlockCyclic2BlockRow.hpp"
#include "DistSamples.hpp"
#include "DistElemMult.hpp"

#include "HSSMatrixMPI.apply.hpp"
#include "HSSMatrixMPI.compress.hpp"
#include "HSSMatrixMPI.compress_stable.hpp"
#include "HSSMatrixMPI.compress_kernel.hpp"
#include "HSSMatrixMPI.factor.hpp"
#include "HSSMatrixMPI.solve.hpp"
#include "HSSMatrixMPI.extract.hpp"
#include "HSSMatrixMPI.extract_blocks.hpp"
#include "HSSMatrixMPI.Schur.hpp"

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const DistM_t& A, const opts_t& opts)
      : HSSMatrixBase<scalar_t>(A.rows(), A.cols(), true),
      blacs_grid_(A.grid()) {
      setup_hierarchy(opts, 0, 0);
      setup_local_context();
      setup_ranges(0, 0);
      compress(A, opts);
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const structured::ClusterTree& t, const BLACSGrid* g, const opts_t& opts)
      : HSSMatrixBase<scalar_t>(t.size, t.size, true), blacs_grid_(g) {
      setup_hierarchy(t, opts, 0, 0);
      setup_local_context();
      setup_ranges(0, 0);
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const structured::ClusterTree& t, const DistM_t& A, const opts_t& opts)
      : HSSMatrixBase<scalar_t>(A.rows(), A.cols(), true),
      blacs_grid_(A.grid()) {
      assert(t.size == A.rows() && t.size == A.cols());
      setup_hierarchy(t, opts, 0, 0);
      setup_local_context();
      setup_ranges(0, 0);
      compress(A, opts);
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (std::size_t m, std::size_t n, const BLACSGrid* Agrid,
     const dmult_t& Amult, const delem_t& Aelem, const opts_t& opts)
      : HSSMatrixBase<scalar_t>(m, n, true), blacs_grid_(Agrid) {
      setup_hierarchy(opts, 0, 0);
      setup_local_context();
      setup_ranges(0, 0);
      compress(Amult, Aelem, opts);
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const structured::ClusterTree& t, const BLACSGrid* Agrid,
     const dmult_t& Amult, const delem_t& Aelem, const opts_t& opts)
      : HSSMatrixBase<scalar_t>(t.size, t.size, true), blacs_grid_(Agrid) {
      setup_hierarchy(t, opts, 0, 0);
      setup_local_context();
      setup_ranges(0, 0);
      compress(Amult, Aelem, opts);
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (kernel::Kernel<real_t>& K, const BLACSGrid* Kgrid, const opts_t& opts)
      : HSSMatrixBase<scalar_t>(K.n(), K.n(), true), blacs_grid_(Kgrid) {
      TaskTimer timer("clustering");
      timer.start();
      auto t = binary_tree_clustering
        (opts.clustering_algorithm(), K.data(), K.permutation(), opts.leaf_size());
      if (opts.verbose() && Comm().is_root())
        std::cout << "# clustering (" << get_name(opts.clustering_algorithm())
                  << ") time = " << timer.elapsed() << std::endl;
      setup_hierarchy(t, opts, 0, 0);
      setup_local_context();
      setup_ranges(0, 0);
      compress(K, opts);
    }

    template<typename scalar_t>
    HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const HSSMatrixMPI<scalar_t>& other)
      : HSSMatrixBase<scalar_t>(other.rows(), other.cols(), other.active()) {
      *this = other;
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>&
    HSSMatrixMPI<scalar_t>::operator=(const HSSMatrixMPI<scalar_t>& other) {
      HSSMatrixBase<scalar_t>::operator=(other);
      blacs_grid_ = other.blacs_grid_;
      blacs_grid_local_ = other.blacs_grid_local_;
      auto& og = *other.owned_blacs_grid_.get();
      owned_blacs_grid_ = std::unique_ptr<BLACSGrid>(new BLACSGrid(og));
      auto& ogl = *other.owned_blacs_grid_.get();
      owned_blacs_grid_local_ = std::unique_ptr<BLACSGrid>(new BLACSGrid(ogl));
      ranges_ = other.ranges_;
      U_ = other.U_;
      V_ = other.V_;
      D_ = other.D_;
      B01_ = other.B01_;
      B10_ = other.B10_;
      return *this;
    }

    template<typename scalar_t> std::unique_ptr<HSSMatrixBase<scalar_t>>
    HSSMatrixMPI<scalar_t>::clone() const {
      return std::unique_ptr<HSSMatrixBase<scalar_t>>
        (new HSSMatrixMPI<scalar_t>(*this));
    }

    /** private constructor */
    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (std::size_t m, std::size_t n, const opts_t& opts,
     const MPIComm& c, int P, std::size_t roff, std::size_t coff)
      : HSSMatrixBase<scalar_t>(m, n, !c.is_null()) {
      owned_blacs_grid_ = std::unique_ptr<BLACSGrid>(new BLACSGrid(c, P));
      blacs_grid_ = owned_blacs_grid_.get();
      setup_hierarchy(opts, roff, coff);
      setup_local_context();
      setup_ranges(roff, coff);
    }

    /** private constructor */
    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const structured::ClusterTree& t, const opts_t& opts,
     const MPIComm& c, int P, std::size_t roff, std::size_t coff)
      : HSSMatrixBase<scalar_t>(t.size, t.size, !c.is_null()) {
      owned_blacs_grid_ = std::unique_ptr<const BLACSGrid>(new BLACSGrid(c, P));
      blacs_grid_ = owned_blacs_grid_.get();
      setup_hierarchy(t, opts, roff, coff);
      setup_local_context();
      setup_ranges(roff, coff);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::setup_local_context() {
      if (!this->leaf()) {
        if (Pl() <= 1) { // child 0 is sequential, create a local context
          if (!Comm().is_null()) {
            auto c = Comm().sub_self(0);
            if (Comm().rank() == 0) {
              owned_blacs_grid_local_ = std::unique_ptr<const BLACSGrid>
                (new BLACSGrid(std::move(c), 1));
              blacs_grid_local_ = owned_blacs_grid_local_.get();
            }
          }
        } else {
          if (child(0)->active())
            blacs_grid_local_ = child(0)->grid_local();
        }
        if (Pr() <= 1) { // child 1 is sequential, create a local context
          if (!Comm().is_null()) {
            auto c = Comm().sub_self(Pl());
            if (Comm().rank() == Pl()) {
              owned_blacs_grid_local_ = std::unique_ptr<const BLACSGrid>
                (new BLACSGrid(std::move(c), 1));
              blacs_grid_local_ = owned_blacs_grid_local_.get();
            }
          }
        } else {
          if (child(1)->active())
            blacs_grid_local_ = child(1)->grid_local();
        }
      } else blacs_grid_local_ = blacs_grid_;
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::setup_ranges
    (std::size_t roff, std::size_t coff) {
      const int P = Ptotal();
      ranges_ = TreeLocalRanges(P);
      if (this->leaf()) {
        for (int p=0; p<P; p++) {
          ranges_.rlo(p) = roff;
          ranges_.rhi(p) = roff + this->rows();
          ranges_.clo(p) = coff;
          ranges_.chi(p) = coff + this->cols();
          ranges_.leaf_procs(p) = P;
        }
      } else {
        auto pl = Pl();
        auto pr = Pr();
        if (pl <= 1) {
          ranges_.rlo(0) = roff;
          ranges_.clo(0) = coff;
          if (P > 1) {
            ranges_.rhi(0) = roff + child(0)->rows();
            ranges_.chi(0) = coff + child(0)->cols();
          }
          ranges_.leaf_procs(0) = 1;
        } else {
          auto ch0 = static_cast<HSSMatrixMPI<scalar_t>*>(child(0));
          for (int p=0; p<pl; p++) {
            ranges_.rlo(p) = ch0->ranges_.rlo(p);
            ranges_.rhi(p) = ch0->ranges_.rhi(p);
            ranges_.clo(p) = ch0->ranges_.clo(p);
            ranges_.chi(p) = ch0->ranges_.chi(p);
            ranges_.leaf_procs(p) = ch0->ranges_.leaf_procs(p);
          }
        }
        if (pr <= 1) {
          if (P > 1) {
            ranges_.rlo(pl) = roff + child(0)->rows();
            ranges_.clo(pl) = coff + child(0)->cols();
          }
          ranges_.rhi(pl) = roff + this->rows();
          ranges_.chi(pl) = coff + this->cols();
          ranges_.leaf_procs(pl) = 1;
        } else {
          auto ch1 = static_cast<HSSMatrixMPI<scalar_t>*>(child(1));
          for (int p=pl; p<P; p++) {
            ranges_.rlo(p) = ch1->ranges_.rlo(p-pl);
            ranges_.rhi(p) = ch1->ranges_.rhi(p-pl);
            ranges_.clo(p) = ch1->ranges_.clo(p-pl);
            ranges_.chi(p) = ch1->ranges_.chi(p-pl);
            ranges_.leaf_procs(p) = ch1->ranges_.leaf_procs(p-pl);
          }
        }
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::setup_hierarchy
    (const opts_t& opts, std::size_t roff, std::size_t coff) {
      auto m = this->rows();
      auto n = this->cols();
      if (m > std::size_t(opts.leaf_size()) ||
          n > std::size_t(opts.leaf_size())) {
        this->ch_.reserve(2);
        auto pl = Pl(m, m/2, m-m/2, Ptotal());
        auto pr = Pr(m, m/2, m-m/2, Ptotal());
        if (pl > 1) {
          this->ch_.emplace_back
            (new HSSMatrixMPI<scalar_t>
             (m/2, n/2, opts, Comm().sub(0, pl), pl, roff, coff));
        } else {
          bool act = !Comm().is_null() && Comm().rank() == 0;
          this->ch_.emplace_back
            (new HSSMatrix<scalar_t>(m/2, n/2, opts, act));
        }
        if (pr > 1) {
          this->ch_.emplace_back
            (new HSSMatrixMPI<scalar_t>
             (m-m/2, n-n/2, opts, Comm().sub(pl, pr), pr,
              roff+m/2, coff+n/2));
        } else {
          bool act = !Comm().is_null() && Comm().rank() == pl;
          this->ch_.emplace_back
            (new HSSMatrix<scalar_t>(m-m/2, n-n/2, opts, act));
        }
      }
    }

    // TODO this only works with 1 tree, so all blocks are square!!
    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::setup_hierarchy
    (const structured::ClusterTree& t, const opts_t& opts,
     std::size_t roff, std::size_t coff) {
      if (!t.c.empty()) {
        assert(t.size == t.c[0].size + t.c[1].size);
        const int P = grid()->P();
        auto pl = Pl(t.size, t.c[0].size, t.c[1].size, P);
        auto pr = Pr(t.size, t.c[0].size, t.c[1].size, P);
        this->ch_.reserve(2);
        if (pl > 1) {
          this->ch_.emplace_back
            (new HSSMatrixMPI<scalar_t>
             (t.c[0], opts, Comm().sub(0, pl), pl, roff, coff));
        } else {
          bool act = !Comm().is_null() && Comm().rank() == 0;
          this->ch_.emplace_back(new HSSMatrix<scalar_t>(t.c[0], opts, act));
        }
        if (pr > 1) {
          this->ch_.emplace_back
            (new HSSMatrixMPI<scalar_t>
             (t.c[1], opts, Comm().sub(pl, pr), pr,
              roff+t.c[0].size, coff+t.c[0].size));
        } else {
          bool act = !Comm().is_null() && Comm().rank() == pl;
          this->ch_.emplace_back(new HSSMatrix<scalar_t>(t.c[1], opts, act));
        }
      }
    }

    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::max_rank() const {
      return Comm().all_reduce(this->rank(), MPI_MAX);
    }
    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::rank() const {
      if (!this->active()) return 0;
      std::size_t rank = std::max(U_.cols(), V_.cols());
      for (auto& c : this->ch_) rank = std::max(rank, c->rank());
      return rank;
    }

    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::total_memory() const {
      return Comm().all_reduce(memory(), MPI_SUM);
    }
    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::memory() const {
      if (!this->active()) return 0;
      std::size_t memory = sizeof(*this) + U_.memory() + V_.memory()
        + D_.memory() + B01_.memory() + B10_.memory();
      for (auto& c : this->ch_) memory += c->memory();
      return memory;
    }

    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::total_nonzeros() const {
      return Comm().all_reduce(nonzeros(), MPI_SUM);
    }
    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::nonzeros() const {
      if (!this->active()) return 0;
      std::size_t nnz = sizeof(*this) + U_.nonzeros() + V_.nonzeros()
        + D_.nonzeros() + B01_.nonzeros() + B10_.nonzeros();
      for (auto& c : this->ch_) nnz += c->nonzeros();
      return nnz;
    }

    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::total_factor_nonzeros() const {
      return Comm().all_reduce(factor_nonzeros(), MPI_SUM);
    }
    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::factor_nonzeros() const {
      if (!this->active()) return 0;
      std::size_t fnnz = this->ULV_mpi_.nonzeros();
      for (auto& c : this->ch_) fnnz += c->factor_nonzeros();
      return fnnz;
    }

    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::max_levels() const {
      return Comm().all_reduce(levels(), MPI_MAX);
    }
    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::levels() const {
      if (!this->active()) return 0;
      std::size_t lvls = 0;
      for (auto& c : this->ch_) lvls = std::max(lvls, c->levels());
      return 1 + lvls;
    }

    template<typename scalar_t> DistributedMatrix<scalar_t>
    HSSMatrixMPI<scalar_t>::dense() const {
      // TODO faster implementation?  an implementation similar to the
      // sequential algorithm is difficult, as it will require a lot
      // of communication? Maybe just use the extraction routine??
      DistM_t identity(grid(), this->cols(), this->cols());
      identity.eye();
      return apply(identity);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::shift(scalar_t sigma) {
      if (!this->active()) return;
      if (this->leaf()) D_.shift(sigma);
      else for (auto& c : this->ch_) c->shift(sigma);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::communicate_child_data
    (WorkCompressMPI<scalar_t>& w) {
      w.c[0].dR = w.c[0].Rr.cols();
      w.c[0].dS = w.c[0].Sr.cols();
      w.c[1].dR = w.c[1].Rr.cols();
      w.c[1].dS = w.c[1].Sr.cols();
      int rank = Comm().rank(), P = Ptotal(), root1 = Pl();
      int P0active = child(0)->Pactive();
      int P1active = child(1)->Pactive();
      std::vector<MPIRequest> sreq;
      std::vector<std::size_t> sbuf0, sbuf1;
      if (rank < P0active) {
        if (rank < (P-P0active)) {
          // I'm one of the first P-P0active processes that are active
          // on child0, so I need to send to one or more others which
          // are not active on child0, ie the ones in [P0active,P)
          sbuf0.reserve(8+w.c[0].Ir.size()+w.c[0].Ic.size()+
                        w.c[0].Jr.size()+w.c[0].Jc.size());
          sbuf0.push_back(std::size_t(child(0)->U_state_));
          sbuf0.push_back(std::size_t(child(0)->V_state_));
          sbuf0.push_back(child(0)->U_rank_);
          sbuf0.push_back(child(0)->V_rank_);
          sbuf0.push_back(child(0)->U_rows_);
          sbuf0.push_back(child(0)->V_rows_);
          sbuf0.push_back(w.c[0].dR);
          sbuf0.push_back(w.c[0].dS);
          for (auto i : w.c[0].Ir) sbuf0.push_back(i);
          for (auto i : w.c[0].Ic) sbuf0.push_back(i);
          for (auto i : w.c[0].Jr) sbuf0.push_back(i);
          for (auto i : w.c[0].Jc) sbuf0.push_back(i);
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
          sbuf1.reserve(8+w.c[1].Ir.size()+w.c[1].Ic.size()+
                        w.c[1].Jr.size()+w.c[1].Jc.size());
          sbuf1.push_back(std::size_t(child(1)->U_state_));
          sbuf1.push_back(std::size_t(child(1)->V_state_));
          sbuf1.push_back(child(1)->U_rank_);
          sbuf1.push_back(child(1)->V_rank_);
          sbuf1.push_back(child(1)->U_rows_);
          sbuf1.push_back(child(1)->V_rows_);
          sbuf1.push_back(w.c[1].dR);
          sbuf1.push_back(w.c[1].dS);
          for (auto i : w.c[1].Ir) sbuf1.push_back(i);
          for (auto i : w.c[1].Ic) sbuf1.push_back(i);
          for (auto i : w.c[1].Jr) sbuf1.push_back(i);
          for (auto i : w.c[1].Jc) sbuf1.push_back(i);
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
        child(0)->U_state_ = State(*ptr++);
        child(0)->V_state_ = State(*ptr++);
        child(0)->U_rank_ = *ptr++;
        child(0)->V_rank_ = *ptr++;
        child(0)->U_rows_ = *ptr++;
        child(0)->V_rows_ = *ptr++;
        w.c[0].dR = *ptr++;
        w.c[0].dS = *ptr++;
        w.c[0].Ir.resize(child(0)->U_rank_);
        w.c[0].Ic.resize(child(0)->V_rank_);
        w.c[0].Jr.resize(child(0)->U_rank_);
        w.c[0].Jc.resize(child(0)->V_rank_);
        for (int i=0; i<child(0)->U_rank_; i++) w.c[0].Ir[i] = *ptr++;
        for (int i=0; i<child(0)->V_rank_; i++) w.c[0].Ic[i] = *ptr++;
        for (int i=0; i<child(0)->U_rank_; i++) w.c[0].Jr[i] = *ptr++;
        for (int i=0; i<child(0)->V_rank_; i++) w.c[0].Jc[i] = *ptr++;
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
        child(1)->U_state_ = State(*ptr++);
        child(1)->V_state_ = State(*ptr++);
        child(1)->U_rank_ = *ptr++;
        child(1)->V_rank_ = *ptr++;
        child(1)->U_rows_ = *ptr++;
        child(1)->V_rows_ = *ptr++;
        w.c[1].dR = *ptr++;
        w.c[1].dS = *ptr++;
        w.c[1].Ir.resize(child(1)->U_rank_);
        w.c[1].Ic.resize(child(1)->V_rank_);
        w.c[1].Jr.resize(child(1)->U_rank_);
        w.c[1].Jc.resize(child(1)->V_rank_);
        for (int i=0; i<child(1)->U_rank_; i++) w.c[1].Ir[i] = *ptr++;
        for (int i=0; i<child(1)->V_rank_; i++) w.c[1].Ic[i] = *ptr++;
        for (int i=0; i<child(1)->U_rank_; i++) w.c[1].Jr[i] = *ptr++;
        for (int i=0; i<child(1)->V_rank_; i++) w.c[1].Jc[i] = *ptr++;
        //assert(msgsize == std::distance(buf, ptr));
      }
      wait_all(sreq);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::notify_inactives_J(WorkCompressMPI<scalar_t>& w) {
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

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::notify_inactives_states
    (WorkCompressMPI<scalar_t>& w) {
      int rank = Comm().rank(), actives = Pactive();
      int inactives = Ptotal() - actives;
      if (rank < inactives) {
        std::vector<std::size_t> sbuf;
        sbuf.reserve(8+w.Ir.size()+w.Ic.size()+w.Jr.size()+w.Jc.size());
        sbuf.push_back(std::size_t(this->U_state_));
        sbuf.push_back(std::size_t(this->V_state_));
        sbuf.push_back(this->U_rank_);
        sbuf.push_back(this->V_rank_);
        sbuf.push_back(this->U_rows_);
        sbuf.push_back(this->V_rows_);
        sbuf.push_back(w.Rr.cols());
        sbuf.push_back(w.Sr.cols());
        for (auto i : w.Ir) sbuf.push_back(i);
        for (auto i : w.Ic) sbuf.push_back(i);
        for (auto i : w.Jr) sbuf.push_back(i);
        for (auto i : w.Jc) sbuf.push_back(i);
        Comm().send(sbuf, actives+rank, 0);
      }
      if (rank >= actives) {
        auto buf = Comm().template recv<std::size_t>(rank-actives, 0);
        auto ptr = buf.begin();
        this->U_state_ = State(*ptr++);
        this->V_state_ = State(*ptr++);
        this->U_rank_ = *ptr++;
        this->V_rank_ = *ptr++;
        this->U_rows_ = *ptr++;
        this->V_rows_ = *ptr++;
        w.dR = *ptr++;
        w.dS = *ptr++;
        w.Ir.resize(this->U_rank_);
        w.Ic.resize(this->V_rank_);
        w.Jr.resize(this->U_rank_);
        w.Jc.resize(this->V_rank_);
        for (int i=0; i<this->U_rank_; i++) w.Ir[i] = *ptr++;
        for (int i=0; i<this->V_rank_; i++) w.Ic[i] = *ptr++;
        for (int i=0; i<this->U_rank_; i++) w.Jr[i] = *ptr++;
        for (int i=0; i<this->V_rank_; i++) w.Jc[i] = *ptr++;
      }
    }

    /**
     * Redistribute a matrix according to the tree of this HSS
     * matrix. This redistribution is based on the column partitioning
     * of the HSS matrix. If this process has a local subtree, then
     * return the local part of A in sub. Else, if this process
     * belongs to a parallel leaf, return the matrix corresponding to
     * that parallel leaf in leaf.
     */
    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::to_block_row
    (const DistM_t& dist, DenseM_t& sub, DistM_t& leaf) const {
      if (!this->active()) return;
      assert(std::size_t(dist.rows())==this->cols());
      BC2BR::block_cyclic_to_block_row
        (ranges_, dist, sub, leaf, grid_local(), Comm());
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::allocate_block_row
    (int d, DenseM_t& sub, DistM_t& leaf) const {
      if (!this->active()) return;
      const int rank = Comm().rank();
      const int P = grid()->P();
      for (int p=0; p<P; p++) {
        auto m = ranges_.chi(p) - ranges_.clo(p);
        if (ranges_.leaf_procs(p) == 1) {
          if (p == rank) sub = DenseM_t(m, d);
        } else {
          if (p <= rank && rank < p+ranges_.leaf_procs(p))
            leaf = DistM_t(grid_local(), m, d);
          p += ranges_.leaf_procs(p)-1;
        }
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::from_block_row
    (DistM_t& dist, const DenseM_t& sub, const DistM_t& leaf,
     const BLACSGrid* lg) const {
      if (!this->active()) return;
      assert(std::size_t(dist.rows())==this->cols());
      BC2BR::block_row_to_block_cyclic(ranges_, dist, sub, leaf, Comm());
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::delete_trailing_block() {
      B01_.clear();
      B10_.clear();
      HSSMatrixBase<scalar_t>::delete_trailing_block();
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::reset() {
      U_.clear();
      V_.clear();
      D_.clear();
      B01_.clear();
      B10_.clear();
      HSSMatrixBase<scalar_t>::reset();
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::print_info
    (std::ostream &out, std::size_t roff, std::size_t coff) const {
      if (!this->active()) return;
      if (Comm().is_root()) {
        out << "rank = " << Comm().rank() << "/" << mpi_rank(MPI_COMM_WORLD)
            << " P=" << Comm().size()
            << " b = [" << roff << "," << roff+this->rows()
            << " x " << coff << "," << coff+this->cols() << "]  U = "
            << this->U_rows() << " x " << this->U_rank() << " V = "
            << this->V_rows() << " x " << this->V_rank();
        if (this->leaf()) std::cout << " leaf" << std::endl;
        else out << " non-leaf" << std::endl;
      }
      for (auto& c : this->ch_) {
        Comm().barrier();
        c->print_info(out, roff, coff);
        roff += c->rows();
        coff += c->cols();
      }
    }


    // explicit template instantiations
    template class HSSMatrixMPI<float>;
    template class HSSMatrixMPI<double>;
    template class HSSMatrixMPI<std::complex<float>>;
    template class HSSMatrixMPI<std::complex<double>>;

    template void
    apply_HSS(Trans ta, const HSSMatrixMPI<float>& a,
              const DistributedMatrix<float>& b, float beta,
              DistributedMatrix<float>& c);
    template void
    apply_HSS(Trans ta, const HSSMatrixMPI<double>& a,
              const DistributedMatrix<double>& b, double beta,
              DistributedMatrix<double>& c);
    template void
    apply_HSS(Trans ta, const HSSMatrixMPI<std::complex<float>>& a,
              const DistributedMatrix<std::complex<float>>& b,
              std::complex<float> beta,
              DistributedMatrix<std::complex<float>>& c);
    template void
    apply_HSS(Trans ta, const HSSMatrixMPI<std::complex<double>>& a,
              const DistributedMatrix<std::complex<double>>& b,
              std::complex<double> beta,
              DistributedMatrix<std::complex<double>>& c);

  } // end namespace HSS
} // end namespace strumpack


