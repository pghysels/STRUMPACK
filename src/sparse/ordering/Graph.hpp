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
 *
 */
#ifndef STRUMPACK_ORDERING_GRAPH_HPP
#define STRUMPACK_ORDERING_GRAPH_HPP

#include <memory>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif
#include "PPt.hpp"
#include "misc/TaskTimer.hpp"
#include "BLASLAPACKWrapper.hpp"

namespace strumpack {
  namespace ordering {

    enum class Interpolation { CONSTANT, AVERAGE };
    inline std::string name(const Interpolation& i) {
      switch (i) {
      case Interpolation::CONSTANT: return "constant";
      case Interpolation::AVERAGE: return "average";
      default: return "interpolation-not-recognized";
      }
    }

    enum class EdgeToVertex { GREEDY, ONESIDED };
    inline std::string name(const EdgeToVertex& i) {
      switch (i) {
      case EdgeToVertex::GREEDY: return "greedy";
      case EdgeToVertex::ONESIDED: return "onesided";
      default: return "edgetovertex-not-recognized";
      }
    }

    template<typename intt> class Graph {
    public:
      Graph() = default;
      Graph(intt n, intt nnz);
      Graph(std::vector<intt,NoInit<intt>>&& ptr,
            std::vector<intt,NoInit<intt>>&& ind);
      Graph(const std::string& filename);
      Graph(const std::vector<intt>& buf);
      Graph(const Graph<intt>&) = delete;
      Graph(Graph<intt>&&) = default;
      Graph<intt>& operator=(const Graph<intt>&) = delete;
      Graph<intt>& operator=(Graph<intt>&&) = default;
      virtual ~Graph() = default;

      intt n() const { return n_; }
      intt nnz() const { return nnz_; }

      const intt* ptr() const { return ptr_.data(); }
      const intt* ind() const { return ind_.data(); }
      intt* ptr() { return ptr_.data(); }
      intt* ind() { return ind_.data(); }
      const intt& ptr(std::size_t i) const
      { assert(i<=std::size_t(n())); return ptr_[i]; }
      const intt& ind(std::size_t i) const
      { assert(i<std::size_t(nnz())); return ind_[i]; }
      intt& ptr(std::size_t i)
      { assert(i<=std::size_t(n())); return ptr_[i]; }
      intt& ind(std::size_t i)
      { assert(i<std::size_t(nnz())); return ind_[i]; }

      intt degree(intt i) const
      { assert(i<n()); return ptr_[i+1] - ptr_[i]; }

      void print_info() const;
      void check() const;
      void check_rectangular() const;
      std::size_t fill(const PPt<intt>& p) const;
      Graph<intt> symperm(const PPt<intt>& p) const;

      void write_Laplacian(const std::string& filename) const;

      template<typename scalart>
      void dense_Laplacian(scalart* D, int ld) const;

      template<ThreadingModel par, typename scalart>
      void Laplacian(const scalart* x, scalart* y) const;

      template<ThreadingModel par, typename scalart>
      void Laplacian_add_off_diagonal
      (const scalart* xd, const scalart* xo, scalart* y) const;

      template<ThreadingModel par, typename scalart>
      void Dinv(scalart* y) const;

      std::pair<Graph<intt>,std::vector<intt>>
      coarsen(ThreadingModel par) const;

      template<typename scalart>
      std::vector<scalart,NoInit<scalart>>
      interpolate(ThreadingModel par,
                  const std::vector<scalart,NoInit<scalart>>& fc,
                  const std::vector<intt>& state,
                  const Interpolation& interp) const;

      template<typename scalart> void
      interpolate(ThreadingModel par, const scalart* fc, scalart* f,
                  const std::vector<intt>& state,
                  const Interpolation& interp) const;

      template<typename scalart>
      void vertex_separator_from_Fiedler
      (ThreadingModel par, const scalart* F, scalart cut,
       std::vector<intt>& part, EdgeToVertex e2v) const;

      void to_1_based();
      Graph<intt> get_1_based() const;
      void to_0_based();
      Graph<intt> get_0_based() const;

      bool connected() const;
      bool connected(std::vector<intt>& mark) const;

      std::pair<Graph<intt>,Graph<intt>>
      extract_domains(ThreadingModel par,
                      const std::vector<intt>& part) const;
      Graph<intt>
      extract_domain_1(ThreadingModel par,
                       const std::vector<intt>& part) const;

      bool unconnected_nodes(std::vector<intt>& mark) const;

      bool dense_nodes(std::vector<intt>& mark) const;

      std::vector<intt> serialize() const;

      template<typename scalart> scalart minimize_conductance
      (ThreadingModel par, const scalart* F, int imbalance) const;
      template<typename scalart> void print_conductance
      (ThreadingModel par, const scalart* F,
       std::ostream& log = std::cout) const;

      template<typename scalart> scalart minimize_approx_conductance
      (ThreadingModel par, const scalart* F, int imbalance) const;
      template<typename scalart> void print_approx_conductance
      (ThreadingModel par, const scalart* F,
       std::ostream& log = std::cout) const;

    private:
      intt n_ = 0, nnz_ = 0;
      std::vector<intt,NoInit<intt>> ptr_, ind_;

      void symperm(const PPt<intt>& p, Graph<intt>& gp) const;
      void read_mtx(const std::string& filename);
      void symmetrize();
      void etree(intt* parent/*[n]*/, intt* work/*[n]*/) const;
      static void post_order(intt n, const intt* parent,
                             intt* post, intt* work);
      static intt leaf(intt i, intt j, const intt* first, intt* maxfirst,
                       intt* prevleaf, intt* ancestor, intt* jleaf);
      static intt tdfs(intt j, intt k, intt* head, const intt* next,
                       intt* post, intt* stack);
      std::size_t fill_Cholesky(const intt* parent, const intt* post,
                                intt* work);

      auto coarsen_seq() const;
      auto coarsen_task() const;

      template<typename scalart> void
      interpolate_constant_seq
      (const scalart* fc, scalart* f, const std::vector<intt>& state) const;
      template<typename scalart> void
      interpolate_constant_task
      (const scalart* fc, scalart* f, const std::vector<intt>& state) const;
      template<typename scalart> void
      interpolate_average_seq
      (const scalart* fc, scalart* f, const std::vector<intt>& state) const;
      template<typename scalart> void
      interpolate_average_task
      (const scalart* fc, scalart* f, const std::vector<intt>& state) const;

      auto extract_domains_seq(const std::vector<intt>& part) const;
      auto extract_domains_task(const std::vector<intt>& part) const;

      auto extract_domain_1_seq(const std::vector<intt>& part) const;
      auto extract_domain_1_task(const std::vector<intt>& part) const;
    };

    template<typename intt> Graph<intt>::Graph(intt n, intt nnz)
      : n_(n), nnz_(nnz), ptr_(n_+1), ind_(nnz) {
      ptr_[0] = 0;
    }

    template<typename intt>
    Graph<intt>::Graph(std::vector<intt,NoInit<intt>>&& ptr,
                       std::vector<intt,NoInit<intt>>&& ind)
      : n_(ptr.size()-1), nnz_(ind.size()),
        ptr_(std::move(ptr)), ind_(std::move(ind)) {
      check();
    }

    template<typename intt> Graph<intt>::Graph(const std::string& filename) {
      // TODO check extension, call proper read function?
      read_mtx(filename);
    }

    /** deserialize */
    template<typename intt> Graph<intt>::Graph(const std::vector<intt>& buf)
      : n_(buf[0]), nnz_(buf[1]), ptr_(buf.data()+2, buf.data()+2+buf[0]+1),
        ind_(buf.data()+2+buf[0]+1, buf.data()+2+buf[0]+1+buf[1]) { }

    template<typename intt> void Graph<intt>::print_info() const {
      std::cout << "# Graph with " << n_ << " vertices and "
                << nnz_ << " edges" << std::endl;
    }

    template<typename intt> std::vector<intt>
    Graph<intt>::serialize() const {
      std::vector<intt> buf;
      buf.reserve(2+ptr_.size()+ind_.size());
      buf.push_back(n_);
      buf.push_back(nnz_);
      std::copy(ptr_.begin(), ptr_.end(), std::back_inserter(buf));
      std::copy(ind_.begin(), ind_.end(), std::back_inserter(buf));
      return buf;
    }

    template<typename intt> void Graph<intt>::check() const {
#if !defined(NDEBUG)
      assert(n_ >= 0);
      assert(nnz_ >= 0);
      assert(!n_ || ptr_[n_] == nnz_);
      for (intt i=0; i<n_; i++) {
        assert(ptr_[i] >= 0);
        assert(ptr_[i] <= nnz_);
        assert(ptr_[i] <= ptr_[i+1]);
        for (intt j=ptr_[i]; j<ptr_[i+1]; j++) {
          assert(ind_[j] >= 0);
          assert(ind_[j] < n_);
        }
      }
      // check symmetry
      for (intt i=0; i<n_; i++)
        for (intt jj=ptr_[i]; jj<ptr_[i+1]; jj++) {
          intt ke = ptr_[ind_[jj]+1];
          assert(std::find(ind()+ptr_[ind_[jj]], ind()+ke, i) != ind()+ke);
        }
#endif
    }

    template<typename intt> void Graph<intt>::check_rectangular() const {
      assert(n_ >= 0);
      assert(nnz_ >= 0);
      assert(!n_ || ptr_[n_] == nnz_);
      for (intt i=0; i<n_; i++) {
        assert(ptr_[i] >= 0);
        assert(ptr_[i] <= nnz_);
        assert(ptr_[i] <= ptr_[i+1]);
        for (intt j=ptr_[i]; j<ptr_[i+1]; j++) {
          assert(ind_[j] >= 0);
        }
      }
    }

    template<typename intt> std::size_t
    Graph<intt>::fill(const PPt<intt>& p) const {
      std::unique_ptr<intt[]> tmp(new intt[7*n_]);
      auto parent = tmp.get();
      auto post   = tmp.get() + n_;
      auto work   = tmp.get() + 2*n_;
      auto gp = symperm(p);
      gp.etree(parent, post);
      post_order(n_, parent, post, work);          // work[2n]
      return gp.fill_Cholesky(parent, post, work); // work[5n]
    }

    template<typename intt> std::size_t Graph<intt>::fill_Cholesky
    (const intt* parent, const intt* post, intt* work) {
      auto ancestor = work;
      auto maxfirst = work + n_;
      auto prevleaf = work + 2*n_;
      auto first = work + 3*n_;
      auto level = work + 4*n_;

      //firstdesc(n, parent, post, first, level); // find first and level
      for (intt i=0; i<n_; i++) first[i] = -1;
      for (intt k=0; k<n_; k++) {
        auto i = post[k];  // node i of etree is kth postordered node
        intt r=i, len=0;   // traverse from i towards the root
        for (; r!=-1 && first[r]==-1; r=parent[r], len++) first[r] = k;
        len += (r == -1) ? (-1) : level[r];  // root node or end of path
        for (intt s=i; s!=r; s=parent[s]) level[s] = len--;
      }

      for (intt i=0; i<n_; i++) {
        prevleaf[i] = -1;  // no previous leaf of the ith row subtree
        maxfirst[i] = -1;  // max first[j] for node j in ith subtree
        ancestor[i] = i;   // every node is in its own set, by itself
      }
      intt jleaf;
      std::size_t nnz = 0;
      for (intt k=0; k<n_; k++) {
        const auto j = post[k];   // j is the kth node in the postordered etree
        const auto hi = ptr_[j+1];
        for (intt p=ptr_[j]; p<hi; p++) {
          auto i = ind_[p];
          auto q = leaf(i, j, first, maxfirst, prevleaf, ancestor, &jleaf);
          if (jleaf) nnz += (level[j] - level[q]);
        }
        if (parent[j] != -1) ancestor[j] = parent[j];
      }
      return nnz+n_; // add diagonal
    }

    template<typename intt> Graph<intt>
    Graph<intt>::symperm(const PPt<intt>& p) const {
      Graph g(n(), nnz());
      symperm(p, g);
      return g;
    }

    template<typename intt> void
    Graph<intt>::symperm(const PPt<intt>& p, Graph<intt>& g) const {
      // assert(g.n() == n());
      // assert(g.nnz() == nnz());
      intt nnz = 0;
      g.ptr(0) = 0;
      for (intt i=0; i<n_; i++) {
        const auto hi = ptr_[p.Pt(i)+1];
        for (intt j=ptr_[p.Pt(i)]; j<hi; j++)
          g.ind(nnz++) = p.P(ind_[j]);
        g.ptr(i+1) = nnz;
      }
    }

    template<typename intt> template<typename scalart> void
    Graph<intt>::dense_Laplacian(scalart* D, int ld) const {
      for (intt j=0; j<n_; j++)
        for (intt i=0; i<n_; i++)
          D[i+j*ld] = 0.;
      for (intt i=0; i<n_; i++) {
        const auto lo = ptr_[i];
        const auto hi = ptr_[i+1];
        D[i+i*ld] = hi - lo;
        const auto hij = ind()+hi;
        for (auto pj=ind()+lo; pj!=hij; pj++)
          D[*pj+i*ld] = -1.;
      }
    }


    template<typename intt> template<ThreadingModel par, typename scalart>
    void Graph<intt>::Laplacian
    (const scalart* x, scalart* y) const {
      TIMER_TIME(TaskType::SPMV, 1, tlapl);
      if constexpr (par == ThreadingModel::SEQ) {
        for (intt i=0; i<n_; i++) {
          const auto lo = ptr_[i];
          const auto hi = ptr_[i+1];
          const auto hii = ind() + hi;
          scalart yi = (hi - lo) * x[i];
          for (auto pj=ind()+lo; pj!=hii; pj++)
            yi -= x[*pj];
          y[i] = yi;
        }
      }
      else if constexpr (par == ThreadingModel::PAR) {
#pragma omp parallel for
        for (intt i=0; i<n_; i++) {
          const auto lo = ptr_[i];
          const auto hi = ptr_[i+1];
          const auto hii = ind() + hi;
          scalart yi = (hi - lo) * x[i];
          for (auto pj=ind()+lo; pj!=hii; pj++)
            yi -= x[*pj];
          y[i] = yi;
        }
      }
      else if constexpr (par == ThreadingModel::LOOP) {
#pragma omp for
        for (intt i=0; i<n_; i++) {
          const auto lo = ptr_[i];
          const auto hi = ptr_[i+1];
          const auto hii = ind() + hi;
          scalart yi = (hi - lo) * x[i];
          for (auto pj=ind()+lo; pj!=hii; pj++)
            yi -= x[*pj];
          y[i] = yi;
        }
      }
      else if constexpr (par == ThreadingModel::TASK) {
#pragma omp taskloop default(shared)
        for (intt i=0; i<n_; i++) {
          const auto lo = ptr_[i];
          const auto hi = ptr_[i+1];
          const auto hii = ind() + hi;
          scalart yi = (hi - lo) * x[i];
          for (auto pj=ind()+lo; pj!=hii; pj++)
            yi -= x[*pj];
          y[i] = yi;
        }
      }
    }


    template<typename intt> template<ThreadingModel par, typename scalart>
    void Graph<intt>::Laplacian_add_off_diagonal
    (const scalart* xd, const scalart* xo, scalart* y) const {
      TIMER_TIME(TaskType::SPMV, 1, tlapl);
      if constexpr (par == ThreadingModel::SEQ) {
        for (intt i=0; i<n_; i++) {
          const auto lo = ptr_[i];
          const auto hi = ptr_[i+1];
          const auto hii = ind() + hi;
          scalart yi = (hi - lo) * xd[i];
          for (auto pj=ind()+lo; pj!=hii; pj++)
            yi -= xo[*pj];
          y[i] += yi;
        }
      }
      else if constexpr (par == ThreadingModel::PAR) {
#pragma omp parallel for
        for (intt i=0; i<n_; i++) {
          const auto lo = ptr_[i];
          const auto hi = ptr_[i+1];
          const auto hii = ind() + hi;
          scalart yi = (hi - lo) * xd[i];
          for (auto pj=ind()+lo; pj!=hii; pj++)
            yi -= xo[*pj];
          y[i] += yi;
        }
      }
      else if constexpr (par == ThreadingModel::LOOP) {
#pragma omp for
        for (intt i=0; i<n_; i++) {
          const auto lo = ptr_[i];
          const auto hi = ptr_[i+1];
          const auto hii = ind() + hi;
          scalart yi = (hi - lo) * xd[i];
          for (auto pj=ind()+lo; pj!=hii; pj++)
            yi -= xo[*pj];
          y[i] += yi;
        }
      }
      else if constexpr (par == ThreadingModel::TASK) {
#pragma omp taskloop default(shared)
        for (intt i=0; i<n_; i++) {
          const auto lo = ptr_[i];
          const auto hi = ptr_[i+1];
          const auto hii = ind() + hi;
          scalart yi = (hi - lo) * xd[i];
          for (auto pj=ind()+lo; pj!=hii; pj++)
            yi -= xo[*pj];
          y[i] += yi;
        }
      }
    }

    template<typename intt>
    template<ThreadingModel par, typename scalart>
    void Graph<intt>::Dinv(scalart* y) const {
      if constexpr (par == ThreadingModel::SEQ) {
        for (intt i=0; i<n_; i++)
          y[i] /= degree(i);
      }
      else if constexpr (par == ThreadingModel::PAR) {
#pragma omp parallel for simd
        for (intt i=0; i<n_; i++)
          y[i] /= degree(i);
      }
      else if constexpr (par == ThreadingModel::LOOP) {
#pragma omp for simd
        for (intt i=0; i<n_; i++)
          y[i] /= degree(i);
      }
      else if constexpr (par == ThreadingModel::TASK) {
#pragma omp taskloop simd default(shared)
        for (intt i=0; i<n_; i++)
          y[i] /= degree(i);
      }
    }

    template<typename intt> void Graph<intt>::write_Laplacian
    (const std::string& filename) const {
      std::fstream f(filename, std::fstream::out);
      f << "%%MatrixMarket matrix coordinate real general\n";
      f << n() << " " << n() << " " << nnz()+n() << "\n";
      for (intt i=0; i<n_; i++) {
        f << i+1 << " " << i+1 << " " << degree(i) << "\n";
        for (intt j=ptr_[i]; j<ptr_[i+1]; j++)
          f << i+1 << " " << ind_[j]+1 << " -1\n";
      }
    }

    template<typename intt> std::pair<Graph<intt>,std::vector<intt>>
    Graph<intt>::coarsen(ThreadingModel par) const {
      if (n_ < 5000) return coarsen_seq();
      switch (par) {
      case ThreadingModel::SEQ: return coarsen_seq();
      case ThreadingModel::TASK: return coarsen_task();
      case ThreadingModel::LOOP: {
        std::pair<Graph<intt>,std::vector<intt>> gs;
#pragma omp single
        gs = coarsen_task();
        return gs;
      }
      case ThreadingModel::PAR: {
        std::pair<Graph<intt>,std::vector<intt>> gs;
#pragma omp parallel
#pragma omp single nowait
        gs = coarsen_task();
        return gs;
      }
      }
      return {Graph<intt>(),std::vector<intt>()};
    }

    // this is the tasked version of coarsen_seq
    template<typename intt> auto Graph<intt>::coarsen_task() const {
      TIMER_TIME(TaskType::COARSEN, 1, tcoarse);
      std::vector<intt> state(n_);
#if defined(_OPENMP)
      int B = 5000; // TODO tune task minimum block size!!
      const int nb = std::min(int(std::ceil(float(n_) / B)),
                              omp_get_num_threads()*4);
      B = int(std::ceil(float(n_) / nb));
#else
      const int nb = 1;
      const int B = n_;
#endif
      std::unique_ptr<intt[]> iwork(new intt[2*nb]);
      auto tnc = iwork.get();
      auto tec = tnc + nb;
      // assert(std::min(nb*B, n_) == n_);
#pragma omp taskloop grainsize(1) default(shared)
      for (intt b=0; b<nb; b++) {
        const auto lb = b*B;
        const auto ub = std::min((b+1)*B, n_);
        intt nc = 0;
        for (intt i=lb; i<ub; i++) {
          if (state[i] != 0) continue;
          state[i] = nc;
          const auto hij = ind() + ptr_[i+1];
          for (auto pj=ind()+ptr_[i]; pj!=hij; pj++)
            if ((*pj >= lb && *pj < ub) && state[*pj] == 0)
              state[*pj] = -nc-1;
          nc++;
        }
        tnc[b] = nc;
      }
      for (intt b=1; b<nb; b++)
        tnc[b] += tnc[b-1];
      int nc = tnc[nb-1];
#pragma omp taskloop grainsize(1) default(shared)
      for (intt b=1; b<nb; b++) {
        const auto lb = b*B;
        const auto ub = std::min((b+1)*B, n_);
        const auto ncb = tnc[b-1];
        for (intt i=lb; i<ub; i++)
          if (state[i] >= 0) state[i] += ncb;
          else state[i] -= ncb;
      }

      std::vector<std::vector<std::pair<intt,intt>>> ec(nb);
#pragma omp taskloop grainsize(1) default(shared)
      for (intt b=0; b<nb; b++) {
        ec[b].reserve(nnz_ / nb);
        const auto lb = b*B;
        const auto ub = std::min((b+1)*B, n_);
        for (intt i=lb; i<ub; i++) {
          const auto ci = state[i];
          if (ci < 0) continue; // skip fine nodes
          const auto eco = ec[b].size();
          const auto hij = ind() + ptr_[i+1];
          for (auto pj=ind()+ptr_[i]; pj!=hij; pj++) {
            const auto indj = *pj;
            // neighbor can be a fine or coarse node
            const auto cj = (state[indj] >= 0) ? state[indj] : -state[indj]-1;
            if (cj == ci) {
              // neighbor is part of same coarse node
              // loop over its neighbors
              const auto hik = ind() + ptr_[indj+1];
              for (auto pk=ind()+ptr_[indj]; pk!=hik; pk++) {
                const auto indk = *pk;
                const auto ck = (state[indk] >= 0) ?
                  state[indk] : (-state[indk]-1);
                if (ci != ck) {
                  std::pair<intt,intt> cik(ci, ck);
                  if (std::find(ec[b].begin()+eco, ec[b].end(), cik)
                      == ec[b].end())
                    ec[b].emplace_back(std::move(cik));
                }
              }
            } else { // neighbor is other coarse node
              std::pair<intt,intt> cij(ci, cj);
              if (std::find(ec[b].begin()+eco, ec[b].end(), cij) == ec[b].end())
                ec[b].emplace_back(std::move(cij));
            }
          }
        }
      }
      tec[0] = ec[0].size();
      for (int b=1; b<nb; b++)
        tec[b] = ec[b].size() + tec[b-1];
      intt nnzc = tec[nb-1];

      Graph<intt> gc(nc, nnzc);
      const auto gcptr = gc.ptr();
      gcptr[0] = 0;
#pragma omp taskloop grainsize(1) default(shared)
      for (intt b=0; b<nb; b++) {
        const auto eoff = (b == 0) ? 0 : tec[b-1];
        const auto ilo = (b == 0) ? 0 : tnc[b-1];
        const auto ihi = tnc[b];
        for (intt i=ilo; i<ihi; i++)
          gcptr[i+1] = 0;
        const auto gcind = gc.ind() + eoff;
        const auto hie = ec[b].size();
        for (std::size_t e=0; e<hie; e++) {
          gcptr[ec[b][e].first+1]++;
          gcind[e] = ec[b][e].second;
        }
        gcptr[ilo+1] += eoff;
        for (intt i=ilo+1; i<ihi; i++)
          gcptr[i+1] += gcptr[i];
      }
      gc.check();
      return std::pair{std::move(gc), std::move(state)};
    }

    template<typename intt> auto Graph<intt>::coarsen_seq() const {
      TIMER_TIME(TaskType::COARSEN, 1, tcoarse);
      std::vector<intt> state(n_);
      intt nc = 0;
      for (intt i=0; i<n_; i++) {
        if (state[i] != 0) continue;
        state[i] = nc;
        const auto hij = ind() + ptr_[i+1];
        for (auto pj=ind()+ptr_[i]; pj!=hij; pj++)
          if (state[*pj] == 0)
            state[*pj] = -nc-1;
        nc++;
      }
      /* if node i is a coarse node, state[i] has the coarse index, if
         node i is a fine node, state[i] is -c-1, where c is the
         corresponding coarse node. */
      std::vector<std::pair<intt,intt>> ec;
      ec.reserve(nnz_); // too much?
      for (intt i=0; i<n_; i++) {
        auto ci = state[i];
        if (ci < 0) continue; // skip fine nodes
        const auto eco = ec.size();
        const auto hij = ind() + ptr_[i+1];
        for (auto pj=ind()+ptr_[i]; pj!=hij; pj++) {
          const auto indj = *pj;
          // neighbor must be a fine node, since coarse nodes are not
          // directly connected
          assert(state[indj] < 0);
          const auto cj = -state[indj]-1;
          if (cj == ci) {
            // neighbor is part of same coarse node
            // loop over its neighbors
            const auto hik = ind() + ptr_[indj+1];
            for (auto pk=ind()+ptr_[indj]; pk!=hik; pk++) {
              auto ck = state[*pk];
              ck = (ck >= 0) ? ck : -ck-1;
              if (ci != ck) {
                std::pair<intt,intt> cik(ci, ck);
                if (std::find(ec.begin()+eco, ec.end(), cik) == ec.end()) {
                  assert(cik.second >= 0);
                  ec.emplace_back(std::move(cik));
                }
              }
            }
          } else { // neighbor is part of other coarse node
            std::pair<intt,intt> cij(ci, cj);
            if (std::find(ec.begin()+eco, ec.end(), cij) == ec.end()) {
              assert(cij.second >= 0);
              ec.emplace_back(std::move(cij));
            }
          }
        }
      }
      intt nnzc = ec.size();
      Graph<intt> gc(nc, nnzc);
      std::fill(gc.ptr_.begin(), gc.ptr_.end(), 0);
      for (intt e=0, j=0; e<nnzc; e++) {
        gc.ptr_[ec[e].first+1]++;
        gc.ind_[j++] = ec[e].second;
      }
      for (intt i=1; i<=nc; i++)
        gc.ptr_[i] += gc.ptr_[i-1];
      gc.check();
      return std::pair{std::move(gc), std::move(state)};
    }


    template<typename intt> template<typename scalart>
    void Graph<intt>::interpolate
    (ThreadingModel par, const scalart* fc, scalart* f,
     const std::vector<intt>& state, const Interpolation& interp) const {
      TIMER_TIME(TaskType::INTERPOLATE, 1, tinterp);
      switch (interp) {
      case Interpolation::CONSTANT: {
        switch (par) {
        case ThreadingModel::SEQ:
          interpolate_constant_seq(fc, f, state); break;
        case ThreadingModel::TASK:
          interpolate_constant_task(fc, f, state); break;
        case ThreadingModel::LOOP:
#pragma omp single
          interpolate_constant_task(fc, f, state);
          break;
        case ThreadingModel::PAR:
#pragma omp parallel
#pragma omp single nowait
          interpolate_constant_task(fc, f, state);
          break;
        default: assert(false);
        }
      } break;
      case Interpolation::AVERAGE: {
        switch (par) {
        case ThreadingModel::SEQ:
          interpolate_average_seq(fc, f, state); break;
        case ThreadingModel::TASK:
          interpolate_average_task(fc, f, state); break;
        case ThreadingModel::LOOP:
#pragma omp single
          interpolate_average_task(fc, f, state);
          break;
        case ThreadingModel::PAR:
#pragma omp parallel
#pragma omp single nowait
          interpolate_average_task(fc, f, state);
          break;
        default: assert(false);
        }
      } break;
      default:
        std::cout << "# WARNING: Interpolation not recognized" << std::endl;
      }
    }

    template<typename intt> template<typename scalart>
    std::vector<scalart,NoInit<scalart>> Graph<intt>::interpolate
    (ThreadingModel par, const std::vector<scalart,NoInit<scalart>>& fc,
     const std::vector<intt>& state, const Interpolation& interp) const {
      std::vector<scalart,NoInit<scalart>> f(n_);
      interpolate(par, fc.data(), f.data(), state, interp);
      return f;
    }

    template<typename intt> template<typename scalart> void
    Graph<intt>::interpolate_constant_seq
    (const scalart* fc, scalart* f, const std::vector<intt>& state) const {
      for (intt i=0; i<n_; i++) {
        intt nc = state[i];
        if (nc >= 0) f[i] = fc[nc];
        else f[i] = fc[-nc-1];
      }
    }
    template<typename intt> template<typename scalart> void
    Graph<intt>::interpolate_constant_task
    (const scalart* fc, scalart* f, const std::vector<intt>& state) const {
#pragma omp taskloop default(shared)
      for (intt i=0; i<n_; i++) {
        intt nc = state[i];
        if (nc >= 0) f[i] = fc[nc];
        else f[i] = fc[-nc-1];
      }
    }
    template<typename intt> template<typename scalart> void
    Graph<intt>::interpolate_average_seq
    (const scalart* fc, scalart* f, const std::vector<intt>& state) const {
      for (intt i=0; i<n_; i++) {
        auto nc = state[i];
        if (nc >= 0) f[i] = fc[nc];
        else {
          scalart fi = 0.;
          intt nf = 0;
          const auto hi = ind() + ptr_[i+1];
          for (auto pj=ind()+ptr_[i]; pj!=hi; pj++) {
            auto nc = state[*pj];
            if (nc >= 0) {
              fi += fc[nc];
              nf++;
            }
          }
          // average of the neighboring (coarse) nodes
          f[i] = fi / nf;
        }
      }
    }
    template<typename intt> template<typename scalart> void
    Graph<intt>::interpolate_average_task
    (const scalart* fc, scalart* f, const std::vector<intt>& state) const {
#pragma omp taskloop default(shared)
      for (intt i=0; i<n_; i++) {
        auto nc = state[i];
        if (nc >= 0) f[i] = fc[nc];
        else {
          scalart fi = 0.;
          intt nf = 0;
          const auto hi = ind() + ptr_[i+1];
          for (auto pj=ind()+ptr_[i]; pj!=hi; pj++) {
            auto nc = state[*pj];
            if (nc >= 0) {
              fi += fc[nc];
              nf++;
            }
          }
          // average of the neighboring (coarse) nodes
          f[i] = fi / nf;
        }
      }
    }

    template<typename intt> template<typename scalart>
    void Graph<intt>::vertex_separator_from_Fiedler
    (ThreadingModel par, const scalart* F, scalart cut,
     std::vector<intt>& part, EdgeToVertex e2v) const {
      switch (e2v) {
      case EdgeToVertex::GREEDY: {
        // a greedy algorithm to find a minimum vertex separator
        // TODO threading
        intt se = 0, sv = 0;
        for (intt i=0; i<n_; i++) {
          const auto Fi = F[i];
          auto ose = se;
          const auto hij = ind() + ptr_[i+1];
          if (Fi <= cut) {
            part[i] = 0;
            for (auto pj=ind()+ptr_[i]; pj!=hij; pj++)
              if (F[*pj] > cut) se++;
          } else {
            part[i] = 1;
            for (auto pj=ind()+ptr_[i]; pj!=hij; pj++)
              if (F[*pj] <= cut) se++;
          }
          if (se != ose) part[i] = -(++sv);
        }
        Graph<intt> E(sv, se);
        std::vector<std::pair<intt,intt>> vd(sv);
        E.ptr(0) = se = sv = 0;
        for (intt i=0; i<n_; i++) {
          if (part[i] >= 0) continue;
          const auto hij = ind() + ptr_[i+1];
          intt d = 0;
          if (F[i] <= cut) {
            for (auto pj=ind()+ptr_[i]; pj!=hij; pj++) {
              const auto indj = *pj;
              if (F[indj] > cut) {
                d++;
                E.ind(se++) = -part[indj]-1;
              }
            }
          } else {
            for (auto pj=ind()+ptr_[i]; pj!=hij; pj++) {
              const auto indj = *pj;
              if (F[indj] <= cut) {
                d++;
                E.ind(se++) = -part[indj]-1;
              }
            }
          }
          vd[sv].first = i;
          vd[sv].second = d;
          E.ptr(sv+1) = E.ptr(sv) + d;
          sv++;
        }
        // greedy approach: find vertex, that is connected to the edge
        // separator, with the highest degree. Put that vertex in the
        // vertex separator, remove edges to/from that vertex and update
        // degrees.
        while (1) {
          intt imax = 0, dmax = 0;
          for (intt i=0; i<sv; i++)
            if (vd[i].second > dmax) {
              dmax = vd[i].second;
              imax = i;
            }
          if (dmax == 0) {
            for (intt i=0; i<sv; i++)
              if (vd[i].second == 0)
                part[vd[i].first] = (F[vd[i].first] <= cut) ? 0 : 1;
            break;
          }
          auto hij = E.ptr(imax+1);
          for (intt j=E.ptr(imax); j<hij; j++) {
            auto k = E.ind(j);
            if (k < 0) continue;
            auto hil = E.ptr(k+1);
            for (intt l=E.ptr(k); l<hil; l++) {
              if (E.ind(l) == imax) {
                E.ind(l) = -1;
                vd[k].second--;
              }
            }
            E.ind(j) = -1;
          }
          vd[imax].second = -1;
          part[vd[imax].first] = 2;
        };
      } break;
      case EdgeToVertex::ONESIDED: {
        // TODO PAR, LOOP, SEQ
        // this is cheaper, and parallel, but worse quality
        switch (par) {
        case ThreadingModel::SEQ: {
          for (intt i=0; i<n_; i++) {
            if (F[i] <= cut) {
              part[i] = 0;
              const auto hi = ind() + ptr(i+1);
              for (auto pj=ind()+ptr(i); pj!=hi; pj++)
                if (F[*pj] > cut) { part[i] = 2; break; }
            } else part[i] = 1;
          }
        } break;
        case ThreadingModel::PAR: {
#pragma omp parallel for
          for (intt i=0; i<n_; i++) {
            if (F[i] <= cut) {
              part[i] = 0;
              const auto hi = ind() + ptr(i+1);
              for (auto pj=ind()+ptr(i); pj!=hi; pj++)
                if (F[*pj] > cut) { part[i] = 2; break; }
            } else part[i] = 1;
          }
        } break;
        case ThreadingModel::LOOP: {
#pragma omp for
          for (intt i=0; i<n_; i++) {
            if (F[i] <= cut) {
              part[i] = 0;
              const auto hi = ind() + ptr(i+1);
              for (auto pj=ind()+ptr(i); pj!=hi; pj++)
                if (F[*pj] > cut) { part[i] = 2; break; }
            } else part[i] = 1;
          }
        } break;
        case ThreadingModel::TASK: {
#pragma omp taskloop default(shared)
          for (intt i=0; i<n_; i++) {
            if (F[i] <= cut) {
              part[i] = 0;
              const auto hi = ind() + ptr(i+1);
              for (auto pj=ind()+ptr(i); pj!=hi; pj++)
                if (F[*pj] > cut) { part[i] = 2; break; }
            } else part[i] = 1;
          }
        }}
      }}
    }

    template<typename intt> void Graph<intt>::to_1_based() {
      for (intt i=0; i<nnz_; i++) ind_[i]++;
      for (intt i=0; i<=n_; i++) ptr_[i]++;
    }

    template<typename intt> Graph<intt> Graph<intt>::get_1_based() const {
      Graph<intt> g(n_, nnz_);
      for (intt i=0; i<nnz_; i++) g.ind_[i] = ind_[i] + 1;
      for (intt i=0; i<=n_; i++) g.ptr_[i] = ptr_[i] + 1;
      return g;
    }

    template<typename intt> void Graph<intt>::to_0_based() {
      for (intt i=0; i<=n_; i++) ptr_[i]--;
      for (intt i=0; i<nnz_; i++) ind_[i]--;
    }

    template<typename intt> Graph<intt> Graph<intt>::get_0_based() const {
      Graph<intt> g(n_, nnz_);
      for (intt i=0; i<nnz_; i++) g.ind_[i] = ind_[i] - 1;
      for (intt i=0; i<=n_; i++) g.ptr_[i] = ptr_[i] - 1;
      return g;
    }

    template<typename intt> bool Graph<intt>::connected() const {
      std::vector<intt> mark;
      return connected(mark);
    }

    // Pass a vector in as work-storage. This vector will be set to 0
    // for the first connected component and to 1 for the rest. The
    // vector can be empty, it will be resized to size n.
    template<typename intt> bool
    Graph<intt>::connected(std::vector<intt>& mark) const {
      TIMER_TIME(TaskType::CONNECTED, 1, tex);
#if 1
      // this is a breadth-first search
      mark.assign(n_, 1);
      std::unique_ptr<intt[]> work(new intt[n_]);
      auto base = work.get();
      auto stack = base;
      *stack++ = 0;
      // TODO threading/tasking?
      while (stack != base) {
        auto i = *(--stack);
        mark[i] = 0;
        const auto hi = ind() + ptr_[i+1];
        for (auto pj=ind()+ptr_[i]; pj!=hi; pj++) {
          auto k = *pj;
          if (mark[k] == 1) {
            *stack++ = k;
            mark[k] = 0;
          }
        }
      }
      return std::find(mark.begin(), mark.end(), 1) == mark.end();
#else
      // this is a label propagation algorithm
      // it is more parallel than the BFS algorithm, but slower
      mark.resize(n_);
      std::iota(mark.begin(), mark.end(), 0);
      bool done;
      do {
        // TODO threading, should be easy
        done = true;
        for (intt i=0; i<n_; i++) {
          intt m = mark[i];
          const auto hi = ind() + ptr_[i+1];
          for (auto pj=ind()+ptr_[i]; pj!=hi; pj++)
            m = std::min(m, mark[*pj]);
          if (m < mark[i]) {
            done = false;
            mark[i] = m;
          }
        }
      } while (!done);
      bool connected = true;
      for (intt i=0; i<n_; i++)
        if (mark[i] != 0) {
          mark[i] = 1;
          connected = false;
        }
      return connected;
#endif
    }


    // Pass a vector in as work-storage. This vector will be set to 0
    // for all nodes that are not connected to anything, and to 1 for
    // the rest. The vector can be empty, it will be resized to size n.
    // Returns true if there are unconnected nodes.
    template<typename intt> bool
    Graph<intt>::unconnected_nodes(std::vector<intt>& mark) const {
      TIMER_TIME(TaskType::UNCONNECTED, 1, tex);
      mark.resize(n_);
      for (intt i=0; i<n_; i++)
        mark[i] = (ptr_[i+1] == ptr_[i]) ? 0 : 1;
      // return whether there are unconnected nodes
      return std::find(mark.begin(), mark.end(), 0) != mark.end();
    }

    // Pass a vector in as work-storage. This vector will be set to 0
    // for densely connected nodes, and to 1 for the rest. The vector
    // can be empty, it will be resized to size n.  Returns true if
    // there are unconnected nodes.
    template<typename intt> bool Graph<intt>::dense_nodes
    (std::vector<intt>& mark) const {
      mark.resize(n_);
      std::fill(mark.begin(), mark.end(), 1);
      const double max_degree = 10. * double(nnz_) / n_;
      bool has_dense = false;
      for (intt i=0; i<n_; i++)
        if (ptr_[i+1] - ptr_[i] > max_degree) {
          mark[i] = 0;
          has_dense = true;
        }
      return has_dense;
    }

    template<typename intt>
    std::pair<Graph<intt>,Graph<intt>> Graph<intt>::extract_domains
    (ThreadingModel par, const std::vector<intt>& part) const {
      TIMER_TIME(TaskType::EXTRACT_DOMAINS, 1, tex);
      switch (par) {
      case ThreadingModel::SEQ: return extract_domains_seq(part);
      case ThreadingModel::TASK: return extract_domains_task(part);
      case ThreadingModel::LOOP: {
        std::pair<Graph<intt>,Graph<intt>> AB;
#pragma omp single
        { AB = extract_domains_task(part); }
        return AB;
      }
      case ThreadingModel::PAR: {
        std::pair<Graph<intt>,Graph<intt>> AB;
#pragma omp parallel
#pragma omp single nowait
        { AB = extract_domains_task(part); }
        return AB;
      }
      default:
        return std::pair<Graph<intt>,Graph<intt>>();
        assert(false);
      }
    }

    template<typename intt> auto Graph<intt>::extract_domains_task
    (const std::vector<intt>& part) const {
#if defined(_OPENMP)
      int TB = 5000; // TODO tune task minimum block size!!
      const int nb =
        std::min(int(std::ceil(float(n_) / TB)), omp_get_num_threads()*4);
      TB = int(std::ceil(float(n_) / nb));
#else
      const int nb = 1, TB = n_;
#endif
      std::unique_ptr<intt[]> work(new intt[n_ + 4*nb]);
      auto g2s = work.get();
      auto tvA = g2s + n_;  auto tvB = tvA + nb;
      auto teA = tvB + nb;  auto teB = teA + nb;
#pragma omp taskloop grainsize(1) default(shared)
      for (intt b=0; b<nb; b++) {
        const auto lb = b*TB;
        const auto ub = std::min((b+1)*TB, n_);
        intt vA = 0, vB = 0, eA = 0, eB = 0;
        for (intt i=lb; i<ub; i++) {
          if (part[i] == 0) {
            vA++;
            const auto hi = ind() + ptr_[i+1];
            for (auto pj=ind()+ptr_[i]; pj!=hi; pj++)
              if (part[*pj] == 0) eA++;
          } else if (part[i] == 1) {
            vB++;
            const auto hi = ind() + ptr_[i+1];
            for (auto pj=ind()+ptr_[i]; pj!=hi; pj++)
              if (part[*pj] == 1) eB++;
          }
        }
        tvA[b] = vA;  tvB[b] = vB;
        teA[b] = eA;  teB[b] = eB;
      }
      for (intt b=1; b<nb; b++) {
        tvA[b] += tvA[b-1];  tvB[b] += tvB[b-1];
        teA[b] += teA[b-1];  teB[b] += teB[b-1];
      }
#pragma omp taskloop grainsize(1) default(shared)
      for (intt b=0; b<nb; b++) {
        const auto lb = b*TB;
        const auto ub = std::min((b+1)*TB, n_);
        intt vA = 0, vB = 0;
        if (b > 0) {
          vA = tvA[b-1];
          vB = tvB[b-1];
        }
        for (intt i=lb; i<ub; i++) {
          if (part[i] == 0) g2s[i] = vA++;
          else if (part[i] == 1) g2s[i] = vB++;
        }
      }
      Graph<intt> A(tvA[nb-1], teA[nb-1]), B(tvB[nb-1], teB[nb-1]);
      A.ptr(0) = B.ptr(0) = 0;

#pragma omp taskloop grainsize(1) default(shared)
      for (intt b=0; b<nb; b++) {
        const auto lb = b*TB;
        const auto ub = std::min((b+1)*TB, n_);
        intt eA = 0, eB = 0;
        auto Aptr = A.ptr();
        auto Bptr = B.ptr();
        if (b > 0) {
          Aptr += tvA[b-1];
          Bptr += tvB[b-1];
          eA = teA[b-1];
          eB = teB[b-1];
        }
        auto Aind = A.ind();
        auto Bind = B.ind();
        for (intt i=lb, vA=0, vB=0; i<ub; i++) {
          if (part[i] == 0) {
            const auto hi = ind() + ptr_[i+1];
            for (auto pj=ind()+ptr_[i]; pj!=hi; pj++) {
              auto ij = *pj;
              if (part[ij] == 0) Aind[eA++] = g2s[ij];
            }
            Aptr[++vA] = eA;
          } else if (part[i] == 1) {
            const auto hi = ind() + ptr_[i+1];
            for (auto pj=ind()+ptr_[i]; pj!=hi; pj++) {
              auto ij = *pj;
              if (part[ij] == 1) Bind[eB++] = g2s[ij];
            }
            Bptr[++vB] = eB;
          }
        }
      }
      A.check();
      B.check();
      return std::pair{std::move(A), std::move(B)};
    }

    template<typename intt> auto Graph<intt>::extract_domains_seq
    (const std::vector<intt>& part) const {
      intt vA = 0, vB = 0, eA = 0, eB = 0;
      std::unique_ptr<intt[]> work(new intt[n_]);
      auto g2s = work.get();
      for (intt i=0; i<n_; i++) {
        if (part[i] == 0) {
          g2s[i] = vA++;
          const auto hi = ind() + ptr_[i+1];
          for (auto pj=ind()+ptr_[i]; pj!=hi; pj++)
            if (part[*pj] == 0) eA++;
        }
        if (part[i] == 1) {
          g2s[i] = vB++;
          const auto hi = ind() + ptr_[i+1];
          for (auto pj=ind()+ptr_[i]; pj!=hi; pj++)
            if (part[*pj] == 1) eB++;
        }
      }
      Graph<intt> A(vA, eA), B(vB, eB);
      vA = vB = eA = eB = A.ptr(0) = B.ptr(0) = 0;
      for (intt i=0; i<n_; i++) {
        if (part[i] == 0) {
          const auto hi = ind() + ptr_[i+1];
          for (auto pj=ind()+ptr_[i]; pj!=hi; pj++) {
            auto ij = *pj;
            if (part[ij] == 0) A.ind(eA++) = g2s[ij];
          }
          A.ptr(++vA) = eA;
        }
        if (part[i] == 1) {
          const auto hi = ind() + ptr_[i+1];
          for (auto pj=ind()+ptr_[i]; pj!=hi; pj++) {
            auto ij = *pj;
            if (part[ij] == 1) B.ind(eB++) = g2s[ij];
          }
          B.ptr(++vB) = eB;
        }
      }
      A.check();
      B.check();
      return std::pair{std::move(A), std::move(B)};
    }

    template<typename intt> Graph<intt>
    Graph<intt>::extract_domain_1
    (ThreadingModel par, const std::vector<intt>& part) const {
      TIMER_TIME(TaskType::EXTRACT_DOMAINS, 1, tex);
      switch (par) {
      case ThreadingModel::SEQ:
        return extract_domain_1_seq(part); break;
      case ThreadingModel::TASK:
        return extract_domain_1_task(part); break;
      case ThreadingModel::LOOP: {
        Graph<intt> B;
#pragma omp single
        { B = extract_domain_1_task(part); }
        return B;
      }
      case ThreadingModel::PAR: {
        Graph<intt> B;
#pragma omp parallel
#pragma omp single nowait
        { B = extract_domain_1_task(part); }
        return B;
      }
      default:
        assert(false);
        return Graph<intt>();
      }
    }

    template<typename intt> auto Graph<intt>::extract_domain_1_task
    (const std::vector<intt>& part) const {
#if defined(_OPENMP)
      int TB = 5000; // TODO tune task minimum block size!!
      const int nb =
        std::min(int(std::ceil(float(n_) / TB)), omp_get_num_threads()*4);
      TB = int(std::ceil(float(n_) / nb));
#else
      const int nb = 1, TB = n_;
#endif
      std::unique_ptr<intt[]> work(new intt[n_ + 2*nb]);
      auto g2s = work.get();
      auto tvB = g2s + n_;
      auto teB = tvB + nb;
#pragma omp taskloop grainsize(1) default(shared)
      for (intt b=0; b<nb; b++) {
        const auto lb = b*TB;
        const auto ub = std::min((b+1)*TB, n_);
        intt vB = 0, eB = 0;
        for (intt i=lb; i<ub; i++) {
          if (part[i] == 1) {
            vB++;
            const auto hi = ind() + ptr_[i+1];
            for (auto pj=ind()+ptr_[i]; pj!=hi; pj++)
              if (part[*pj] == 1) eB++;
          }
        }
        tvB[b] = vB;
        teB[b] = eB;
      }
      for (intt b=1; b<nb; b++) {
        tvB[b] += tvB[b-1];
        teB[b] += teB[b-1];
      }
#pragma omp taskloop grainsize(1) default(shared)
      for (intt b=0; b<nb; b++) {
        const auto lb = b*TB;
        const auto ub = std::min((b+1)*TB, n_);
        intt vB = 0;
        if (b > 0) vB = tvB[b-1];
        for (intt i=lb; i<ub; i++)
          if (part[i] == 1) g2s[i] = vB++;
      }
      Graph B(tvB[nb-1], teB[nb-1]);
      B.ptr(0) = 0;

#pragma omp taskloop grainsize(1) default(shared)
      for (intt b=0; b<nb; b++) {
        const auto lb = b*TB;
        const auto ub = std::min((b+1)*TB, n_);
        intt eB = 0;
        auto Bptr = B.ptr();
        if (b > 0) {
          Bptr += tvB[b-1];
          eB = teB[b-1];
        }
        auto Bind = B.ind();
        for (intt i=lb, vB=0; i<ub; i++) {
          if (part[i] == 1) {
            const auto hi = ind() + ptr_[i+1];
            for (auto pj=ind()+ptr_[i]; pj!=hi; pj++) {
              auto ij = *pj;
              if (part[ij] == 1) Bind[eB++] = g2s[ij];
            }
            Bptr[++vB] = eB;
          }
        }
      }
      B.check();
      return B;
    }

    template<typename intt> auto Graph<intt>::extract_domain_1_seq
    (const std::vector<intt>& part) const {
      intt vB = 0, eB = 0;
      std::unique_ptr<intt[]> work(new intt[n_]);
      auto g2s = work.get();
      for (intt i=0; i<n_; i++) {
        if (part[i] == 1) {
          g2s[i] = vB++;
          const auto hi = ind() + ptr_[i+1];
          for (auto pj=ind()+ptr_[i]; pj!=hi; pj++)
            if (part[*pj] == 1) eB++;
        }
      }
      Graph B(vB, eB);
      vB = eB = B.ptr(0) = 0;
      for (intt i=0; i<n_; i++) {
        if (part[i] == 1) {
          const auto hi = ind() + ptr_[i+1];
          for (auto pj=ind()+ptr_[i]; pj!=hi; pj++) {
            auto ij = *pj;
            if (part[ij] == 1) B.ind(eB++) = g2s[ij];
          }
          B.ptr(++vB) = eB;
        }
      }
      B.check();
      return B;
    }

    template<typename intt> template<typename scalart> scalart
    Graph<intt>::minimize_conductance
    (ThreadingModel par, const scalart* F, int imbalance) const {
      std::unique_ptr<intt[]> work(new intt[2*n_]);
      auto Fi = work.get();
      auto Finv = Fi + n_;
      std::iota(Fi, Fi+n_, 0);
      sort(par, n_, Fi, // get permutation vector that sorts F
           [&F](const auto& i, const auto& j) { return F[i] < F[j]; });
      for (intt i=0; i<n_; i++)
        Finv[Fi[i]] = i;
      TIMER_TIME(TaskType::COUNT_CUTS, 1, tc);
      scalart VolV1 = 0, totV = nnz_,
        Fcut = 0, minQ = std::numeric_limits<scalart>::max();
      // Q = cut(V1, V2) / min(Vol(V1),Vol(V2))
      // cut(V1,V2) is sum of cut edges
      // Vol(V1) is sum of degrees in V1
      auto lo = n_ / (imbalance + 1);
      auto hi = n_ * imbalance / (imbalance + 1);
      std::size_t cuts = 0;
      for (intt i=0; i<hi; i++) {
        auto ii = Fi[i];
        VolV1 += degree(ii);
        auto hi = ind() + ptr_[ii+1];
        for (auto pj=ind()+ptr_[ii]; pj!=hi; pj++)
          if (Finv[*pj] < i) cuts--;
          else cuts++;
        auto Q = cuts / float(std::min(VolV1, totV - VolV1));
        if (Q < minQ && i >= lo) {
          minQ = Q;
          Fcut = F[ii];
        }
      }
      return Fcut;
    }

    template<typename intt> template<typename scalart>
    void Graph<intt>::print_conductance
    (ThreadingModel par, const scalart* F, std::ostream& log) const {
      std::unique_ptr<intt[]> work(new intt[2*n_]);
      auto Fi = work.get();
      auto Finv = Fi + n_;
      std::iota(Fi, Fi+n_, 0);
      sort(par, n_, Fi,
           [&F](const auto& i, const auto& j) { return F[i] < F[j]; });
      for (intt i=0; i<n_; i++)
        Finv[Fi[i]] = i;
      std::size_t cuts = 0;
      float VolV1 = 0, totV = nnz_;
      log << "# i  cuts  Q  F  Fsorted" << std::endl;
      for (intt i=0; i<n_; i++) {
        auto ii = Fi[i];
        VolV1 += degree(ii);
        auto hi = ind() + ptr_[ii+1];
        for (auto pj=ind()+ptr_[ii]; pj!=hi; pj++)
          if (Finv[*pj] < i) cuts--;
          else cuts++;
        auto Q = cuts / std::min(VolV1, totV - VolV1);
        log << cuts << " " << Q << " " << F[i] << " " << F[ii] << std::endl;
      }
    }

    template<typename intt> template<typename scalart> scalart
    Graph<intt>::minimize_approx_conductance
    (ThreadingModel par, const scalart* F, int imbalance) const {
      auto [min, max] = minmax(par, n_, F);
      scalart range = (n_ - 1.0) / (max - min);
      std::unique_ptr<intt[]> work(new intt[3*n_]);
      auto Fi = work.get();
      auto Finv = Fi + n_;
      auto mapped = Finv + n_;
      std::fill(Fi, Fi+n_, -1);
      std::fill(mapped, mapped+n_, -1);
      for (intt i=0; i<n_; i++) {
        Finv[i] = std::round((F[i] - min) * range);
        assert(Finv[i] >= 0);
        assert(Finv[i] < n_);
        auto& Fii = Fi[Finv[i]];
        if (Fii != -1) mapped[i] = Fii;
        Fii = i;
      }
      TIMER_TIME(TaskType::COUNT_CUTS, 1, tc);
      // Q = cut(V1, V2) / min(Vol(V1),Vol(V2))
      // cut(V1,V2) is sum of edges cut
      // Vol(V1) is number if vertices in V1
      auto lo = n_ / (imbalance + 1);
      auto hi = n_ * imbalance / (imbalance + 1);
      scalart Fcut = 0, minQ = std::numeric_limits<scalart>::max();
      std::size_t cuts = 0;
      for (intt i=0; i<hi; i++) {
        auto ii = Fi[i];
        while (ii != -1) {
          auto hi = ind() + ptr_[ii+1];
          for (auto pj=ind()+ptr_[ii]; pj!=hi; pj++) {
            auto Fij = Finv[*pj];
            if (Fij < i) cuts--;
            else if (Fij > i) cuts++;
          }
          ii = mapped[ii];
        }
        if (Fi[i] == -1) continue;
        auto Q = cuts / float(std::min(i, n_-i));
        if (Q < minQ && i >= lo) {
          minQ = Q;
          Fcut = F[Fi[i]];
        }
      }
      return Fcut;
    }


    template<typename intt> template<typename scalart> void
    Graph<intt>::print_approx_conductance
    (ThreadingModel par, const scalart* F,
     std::ostream& log) const {
      auto [min, max] = minmax(par, n_, F);
      scalart range = (n_ - 1.0) / (max - min);
      std::unique_ptr<intt[]> work(new intt[3*n_]);
      auto Fi = work.get();
      auto Finv = Fi + n_;
      auto mapped = Finv + n_;
      std::fill(Fi, Fi+n_, -1);
      std::fill(mapped, mapped+n_, -1);
      for (intt i=0; i<n_; i++) {
        Finv[i] = std::round((F[i] - min) * range);
        assert(Finv[i] >= 0);
        assert(Finv[i] < n_);
        auto& Fii = Fi[Finv[i]];
        if (Fii != -1) mapped[i] = Fii;
        Fii = i;
      }
      // Q = cut(V1, V2) / min(Vol(V1),Vol(V2))
      // cut(V1,V2) is sum of edges cut
      // Vol(V1) is number if vertices in V1
      scalart Fcut = 0, minQ = std::numeric_limits<scalart>::max();
      std::size_t cuts = 0;
      for (intt i=0; i<n_; i++) {
        auto ii = Fi[i];
        while (ii != -1) {
          auto hi = ind() + ptr_[ii+1];
          for (auto pj=ind()+ptr_[ii]; pj!=hi; pj++) {
            auto Fij = Finv[*pj];
            if (Fij < i) cuts--;
            else if (Fij > i) cuts++;
          }
          ii = mapped[ii];
        }
        auto Q = cuts / float(std::min(i, n_-i));
        log << cuts << " " << Q << " " << F[i] << " "
            << ((Fi[i] != -1) ? F[Fi[i]] : -1) << std::endl;
      }
    }

    template<typename intt> void
    Graph<intt>::etree(intt* parent, intt* ancestor) const {
      intt inext;
      for (intt k=0; k<n_; k++) {
        parent[k] = -1;                     // node k has no parent yet
        ancestor[k] = -1;                   // nor does k have an ancestor
        const auto hi = ptr_[k+1];
        for (intt p=ptr_[k]; p<hi; p++) {
          auto i = ind_[p];
          for (; i!=-1 && i<k; i=inext) {   // traverse from i to k
            inext = ancestor[i];            // inext = ancestor of i
            ancestor[i] = k;                // path compression
            if (inext == -1) parent[i] = k; // no anc., parent is k
          }
        }
      }
    }

    template<typename intt> void
    Graph<intt>::read_mtx(const std::string& filename) {
      FILE* fp = fopen(filename.c_str(), "r");
      if (!fp) {
        std::cerr << "ERROR: could not read file" << std::endl;
        exit(1);
      }
      const int max_cline = 256;
      char cline[max_cline];
      if (!fgets(cline, max_cline, fp)) {
        std::cerr << "ERROR: could not read from file" << std::endl;
        exit(1);
      }
      bool complex = strstr(cline, "complex");
      bool symm = strstr(cline, "skew-symmetric")
        || strstr(cline, "symmetric") || strstr(cline, "hermitian");
      while (fgets(cline, max_cline, fp)) {
        if (cline[0] != '%') { // first line should be: m n nnz
          intt m;
          sscanf(cline, "%d %d %d", &m, &n_, &nnz_);
          if (m != n_) {
            std::cerr << "ERROR: matrix is not square!" << std::endl;
            exit(1);
          }
          if (symm) nnz_ = 2 * nnz_;
          std::cout << "# reading " << m << " by " << n_ << " matrix with "
                    << nnz_ << " nnz's from " << filename << std::endl;
          break;
        }
      }
      std::unique_ptr<intt[]> rows(new intt[2*nnz_+n_]);
      auto cols = rows.get() + nnz_;
      auto rowsums = cols + nnz_;
      for (intt i=0; i<n_; i++) rowsums[i] = 0;
      intt ir, ic;
      double dv, dvi;
      nnz_ = 0;
      while ((complex ?
              fscanf(fp, "%d %d %lf %lf\n", &ir, &ic, &dv, &dvi) :
              fscanf(fp, "%d %d %lf\n", &ir, &ic, &dv)) != EOF) {
        ir--; ic--;
        if (ir != ic) { // do not store the diagonal elements
          rows[nnz_  ] = ir;
          cols[nnz_++] = ic;
          rowsums[ir]++;
          if (symm) {
            rows[nnz_  ] = ic;
            cols[nnz_++] = ir;
            rowsums[ic]++;
          }
        }
      }
      fclose(fp);
      ptr_.resize(n_+1);
      ind_.resize(nnz_);
      ptr_[0] = 0;
      for (intt i=0; i<n_; i++) {
        ptr_[i+1] = ptr_[i] + rowsums[i];
        rowsums[i] = 0;
      }
      for (intt i=0; i<nnz_; i++) {
        auto r = rows[i];
        ind_[ptr_[r]+rowsums[r]] = cols[i];
        rowsums[r]++;
      }
      print_info();
      if (!symm) symmetrize();
      std::cout << "# after symmetrization: ";
      print_info();
    }

    template<typename intt> void Graph<intt>::symmetrize() {
      std::unique_ptr<intt[]> rowsum(new intt[n_]);
      for (intt i=0; i<n_; i++) rowsum[i] = ptr_[i+1] - ptr_[i];
      bool update = false;
      for (intt i=0; i<n_; i++)
        for (intt jj=ptr_[i]; jj<ptr_[i+1]; jj++) {
          intt kb = ptr_[ind_[jj]], ke = ptr_[ind_[jj]+1];
          if (std::find(ind()+kb, ind()+ke, i) == ind()+ke) {
            rowsum[ind_[jj]]++;
            update = true;
          }
        }
      if (update) {
        std::vector<intt,NoInit<intt>> ptr_sym(n_+1);
        ptr_sym[0] = 0;
        for (intt i=0; i<n_; i++) ptr_sym[i+1] = ptr_sym[i] + rowsum[i];
        auto nnz_sym = ptr_sym[n_] - ptr_sym[0];
        std::vector<intt,NoInit<intt>> ind_sym(nnz_sym);
        for (intt i=0; i<n_; i++) {
          rowsum[i] = ptr_sym[i] + ptr_[i+1] - ptr_[i];
          for (intt jj=ptr_[i], k=ptr_sym[i]; jj<ptr_[i+1]; jj++)
            ind_sym[k++] = ind_[jj];
        }
        for (intt i=0; i<n_; i++)
          for (intt jj=ptr_[i]; jj<ptr_[i+1]; jj++) {
            intt kb = ptr_[ind_[jj]], ke = ptr_[ind_[jj]+1];
            if (std::find(ind()+kb,ind()+ke, i) == ind()+ke) {
              intt t = ind_[jj];
              ind_sym[rowsum[t]] = i;
              rowsum[t]++;
            }
          }
        ptr_ = std::move(ptr_sym);
        ind_ = std::move(ind_sym);
        nnz_ = nnz_sym;
      }
    }

    template<typename intt> void Graph<intt>::post_order
    (intt n, const intt* parent, intt* post, intt* work) {
      auto head = work;
      auto next = work + n;
      auto stack = work + 2*n;
      for (intt j=0; j<n; j++) head[j] = -1; // empty linked lists
      for (intt j=n-1; j>=0; j--) {          // traverse nodes in reverse order
        if (parent[j] == -1) continue;       // j is a root
        next[j] = head[parent[j]];           // add j to list of its parent
        head[parent[j]] = j;
      }
      for (intt j=0, k=0; j<n; j++) {
        if (parent[j] != -1) continue;       // skip j if it is not a root
        k = tdfs(j, k, head, next, post, stack);
      }
    }

    template<typename intt> intt Graph<intt>::tdfs
    (intt j, intt k, intt* head, const intt* next, intt* post, intt* stack) {
      //if (!head || !next || !post || !stack) return -1; // check inputs
      stack[0] = j;           // place j on the stack
      intt top = 0;
      while (top >= 0) {      // while (stack is not empty)
        auto p = stack[top];   // p = top of stack
        auto i = head[p];      // i = youngest child of p
        if (i == -1) {
          top--;              // p has no unordered children left
          post[k++] = p;      // node p is the kth postordered node
        } else {
          head[p] = next[i];  // remove i from children of p
          stack[++top] = i;   // start dfs on child node i
        }
      }
      return k;
    }

    template<typename intt> intt Graph<intt>::leaf
    (intt i, intt j, const intt* first, intt* maxfirst,
     intt* prevleaf, intt* ancestor, intt* jleaf) {
      //if (!first || !maxfirst || !prevleaf || !ancestor || !jleaf) return -1;
      *jleaf = 0;
      //if (i <= j || first[j] <= maxfirst[i]) return -1; // j not a leaf
      maxfirst[i] = first[j];   // update max first[j] seen so far
      intt jprev = prevleaf[i];  // jprev = previous leaf of ith subtree
      prevleaf[i] = j;
      *jleaf = (jprev == -1) ? 1 : 2;  // j is first or subsequent leaf
      if (*jleaf == 1) return i;  // if 1st leaf, q = root of ith subtree
      intt q = jprev;
      for (; q!=ancestor[q]; q=ancestor[q]);
      for (intt s=jprev, sparent; s!=q; s=sparent) {
        sparent = ancestor[s];  // path compression
        ancestor[s] = q;
      }
      return q;  // q = least common ancester (jprev,j)
    }

  } // end namespace ordering
} //end namespace strumpack

#endif // STRUMPACK_ORDERING_GRAPH_HPP
