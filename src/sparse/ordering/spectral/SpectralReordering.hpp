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
#ifndef STRUMPACK_ORDERING_SPECTRAL_HPP
#define STRUMPACK_ORDERING_SPECTRAL_HPP

#include "misc/TaskTimer.hpp"
#include "misc/Tools.hpp"
// #include "dense/BLASLAPACKWrapper.hpp"
#include "sparse/ordering/Graph.hpp"
//#include "sparse/ordering/spectral/KernighanLin.hpp"
#include "sparse/ordering/RCMReordering.hpp"
#include "sparse/ordering/minimum_degree/AMDReordering.hpp"
#include "sparse/ordering/minimum_degree/MMDReordering.hpp"
//#include "MLFReordering.hpp"
//#include "sparse/ordering/sparspak/ANDReordering.hpp"
#include "sparse/ordering/MetisReordering.hpp"
//#include "sparse/ordering/ScotchReordering.hpp"
#include "sparse/SeparatorTree.hpp"
// #if defined(PROJECTND_USE_SLEPC)
// #include "SLEPcFiedlerSolver.hpp"
// #endif
#include "SpectralStats.hpp"
#include "LanczosFiedlerSolver.hpp"
#include "NDOptions.hpp"

namespace strumpack {
  namespace ordering {

    template<ThreadingModel par, typename scalart, typename intt> void
    smooth_Fiedler(const Graph<intt>& g, scalart* f, const NDOptions& opts) {
      if (opts.smoothing_steps() == 0) return;
      TIMER_TIME(TaskType::SMOOTH, 1, tsmooth);
      // Smooth the interpolated vector f:
      // f_(i+1) <- f_i - D^(-1) A f_i + lambda D^(-1) f_i
      // TODO add 2/3 weighting factor?
      auto n = g.n();
      std::unique_ptr<scalart[]> work(new scalart[n]);
      auto h1 = work.get();
      auto s = opts.smoothing_steps();
      for (int i=0; i<s; i++) {
        if constexpr (par == ThreadingModel::TASK) {
          double tmp1 = 0, tmp2 = 0;
#pragma omp taskloop simd reduction(+:tmp1,tmp2) default(shared)
          for (int l=0; l<n; l++) {
            auto lo = g.ptr(l);
            auto hi = g.ptr(l+1);
            const auto hii = g.ind() + hi;
            auto fl = f[l];
            scalart yi = (hi - lo) * fl;
            for (auto pj=g.ind()+lo; pj!=hii; pj++)
              yi -= f[*pj];
            h1[l] = yi;
            tmp1 += fl * fl;
            tmp2 += fl * yi;
          }
          scalart lambda = tmp2 / tmp1;
#pragma omp taskloop simd default(shared)
          for (int l=0; l<n; l++)
            f[l] -= (h1[l] - lambda * f[l]) / (g.ptr(l+1) - g.ptr(l));
        }
        else if constexpr (par == ThreadingModel::SEQ) {
          double tmp1 = 0, tmp2 = 0;
          for (int l=0; l<n; l++) {
            auto lo = g.ptr(l);
            auto hi = g.ptr(l+1);
            const auto hii = g.ind() + hi;
            auto fl = f[l];
            scalart yi = (hi - lo) * fl;
            for (auto pj=g.ind()+lo; pj!=hii; pj++)
              yi -= f[*pj];
            h1[l] = yi;
            tmp1 += fl * fl;
            tmp2 += fl * yi;
          }
          scalart lambda = tmp2 / tmp1;
          for (int l=0; l<n; l++)
            f[l] -= (h1[l] - lambda * f[l]) / (g.ptr(l+1) - g.ptr(l));
        }
        else {
          g.template Laplacian<par>(f, h1);
          auto lambda = Rayleigh_quotient<par>(n, f, h1);
          axpy<par>(n, -lambda, f, h1);
          g.template Dinv<par>(h1);
          axpy<par>(n, scalart(-1.), h1, f);
        }
      }
    }

    template<typename scalart, typename intt> void
    compute_Fiedler(ThreadingModel par, const Graph<intt>& g,
                    scalart* f, SpectralStats& stats,
                    const NDOptions& opts, std::ostream& log = std::cout) {
      int its = 0;
      bool conv = false;
      TIMER_TIME(TaskType::LANCZOS, 1, tlanczos);
      //     switch (opts.Fiedler_solver()) {
      //     case FiedlerSolver::AUTO:
      //     case FiedlerSolver::PIPE_LANCZOS:
      //     case FiedlerSolver::CA_LANCZOS:
      //     case FiedlerSolver::LANCZOS:
      conv = Lanczos_Fiedler(par, g, f, its, opts, log);
      //       break;
      //     case FiedlerSolver::IMPLICIT_LANCZOS:
      //       conv = implicit_restarted_Lanczos_Fiedler(par, g, f, its, opts, log);
      //       break;
      //     case FiedlerSolver::SLEPC: {
      // #if defined(PROJECTND_USE_SLEPC)
      //       SLEPcFiedlerSolver slepc;
      //       conv = slepc.solve(g, f, its, opts, log);
      //       if (!conv) conv = Lanczos_Fiedler(par, g, f, its, opts, log);
      // #else
      //       std::cout << "ERROR: Not configured with SLEPc support!" << std::endl;
      //       conv = Lanczos_Fiedler(par, g, f, its, opts, log);
      // #endif
      //     } break;
      //     case FiedlerSolver::S_STEP:
      //     case FiedlerSolver::LOBPCG:
      //     case FiedlerSolver::LOBPCG_STABLE:
      //     default: std::cerr << "ERROR: Solver not recognized" << std::endl;
      //     }
      TIMER_STOP(tlanczos);
      stats.Fiedler_solves++;
      stats.Fiedler_its += its;
      stats.Fiedler_maxits = std::max(stats.Fiedler_maxits, its);
      if (!conv) stats.Fiedler_failed_solves++;
    }

    template<typename scalart, typename intt>
    std::vector<scalart,NoInit<scalart>>
    coarse_Fiedler(const Graph<intt>& g, SpectralStats& stats,
                   const NDOptions& opts) {
      TIMER_TIME(TaskType::COARSE_FIEDLER, 1, tc);
      stats.coarse_solves++;
      CoarseSolver cs = opts.coarse_solver();
      if (g.n() > 1000) { // TODO pick something better???
        if (opts.verbose())
          std::cerr << "# WARNING: encountered large coarse problem, n="
                    << g.n() << ", solving with Lanczos" << std::endl;
        cs = CoarseSolver::LANCZOS_RANDOM;
      }
      std::vector<scalart,NoInit<scalart>> f(g.n());
      switch (cs) {
      case CoarseSolver::SYEVX: {
        int n = g.n();
        std::unique_ptr<int[]> iwork_(new int[6*n]);
        auto iwork = iwork_.get();
        auto ifail = iwork + 5*n;
        int M, IL = std::min(2, n), lwork = -1;
        scalart swork;
        auto info = syevx
          ('V', 'I', 'U', n, nullptr, n, 0, 0,
           IL, IL, opts.Fiedler_tol(), M,
           nullptr, f.data(), n, &swork, lwork, iwork, ifail);
        lwork = int(swork);
        std::unique_ptr<scalart[]> work_(new scalart[lwork+n*n+n]);
        auto work = work_.get();
        auto L = work + lwork;
        auto W = L + n*n;
        g.dense_Laplacian(L, n);
        info = syevx
          ('V', 'I', 'U', n, L, n, 0, 0,
           IL, IL, opts.Fiedler_tol(), M,
           W, f.data(), n, work, lwork, iwork, ifail);
        if (info || M!=1)
          std::cerr << "ERROR: *syevx returned info=" << info << std::endl;
      } break;
      case CoarseSolver::LANCZOS_LINEAR: {
        for (std::size_t i=0; i<f.size(); i++)
          f[i] = scalart(i) - (g.n() + 1.) / 2.;
        compute_Fiedler(ThreadingModel::SEQ, g, f.data(), stats, opts);
      } break;
      case CoarseSolver::LANCZOS_RANDOM: {
        random_vector(g.n(), f.data());
        compute_Fiedler(ThreadingModel::SEQ, g, f.data(), stats, opts);
      } break;
      default:
        std::cerr << "WARNING: Coarse solver not recognized" << std::endl;
      }
      return f;
    }

    template<typename scalart, typename intt> void
    smooth_Fiedler(ThreadingModel par, const Graph<intt>& g,
                   scalart* f, const NDOptions& opts) {
      // TODO make this value 200 an option, tune this!!
      if (g.n() < 200) par = ThreadingModel::SEQ;
      switch (par) {
      case ThreadingModel::SEQ:
        smooth_Fiedler<ThreadingModel::SEQ>(g, f, opts); break;
      case ThreadingModel::PAR:
        smooth_Fiedler<ThreadingModel::PAR>(g, f, opts); break;
      case ThreadingModel::TASK:
        smooth_Fiedler<ThreadingModel::TASK>(g, f, opts); break;
      case ThreadingModel::LOOP:
      default:
        static_assert
          (true, "smooth_Fiedler does not work with LOOP parallelism,"
           " use SEQ, PAR or TASK instead.");
      }
    }

    template<typename scalart, typename intt>
    std::vector<scalart,NoInit<scalart>>
    multilevel_Fiedler(ThreadingModel par, const Graph<intt>& g,
                       SpectralStats& stats, const NDOptions& opts) {
      if (g.n() < opts.multilevel_cutoff())
        return coarse_Fiedler<scalart>(g, stats, opts);
      // if (opts.Fiedler_solver() == FiedlerSolver::SLEPC) {
      //   auto f = random_vector<scalart,NoInit<scalart>>(g.n());
      //   compute_Fiedler(par, g, f.data(), stats, opts);
      //   return f;
      // }
      auto [gc, state] = g.coarsen(par);
      //if (gc.n() == g.n() || gc.n() == 1)
      if (gc.n() == g.n())
        return coarse_Fiedler<scalart>(g, stats, opts);
      stats.coarsenings++;
      stats.coarsening_factor += (float(g.n()) / gc.n());
      if (opts.verbose())
        std::cout << "# coarsened graph n=" << gc.n()
                  << " nnz=" << gc.nnz() << std::endl;
      auto fc = multilevel_Fiedler<scalart>(par, gc, stats, opts);
      auto f = g.interpolate(par, fc, state, opts.interpolation());
      smooth_Fiedler(par, g, f.data(), opts);
      compute_Fiedler(par, g, f.data(), stats, opts);
      return f;
    }

    template<typename intt, typename scalart>
    scalart get_Fiedler_cut_value
    (ThreadingModel par, const Graph<intt>& g,
     const scalart* F, const NDOptions& opts) {
      TIMER_TIME(TaskType::CUT_VALUE, 1, tc);
      auto n = g.n();
      scalart cut = 0.;
      switch (opts.Fiedler_cut()) {
      case FiedlerCut::MEDIAN:
        cut = median(n, F); break;
      case FiedlerCut::MIDWAY: {
        auto [min, max] = minmax(par, n, F);
        return (min + max) / 2.;
      } break;
      case FiedlerCut::AVERAGE:
        cut = sum(par, n, F) / n; break;
      case FiedlerCut::ZERO:
        cut = 0.; break;
      case FiedlerCut::OPTIMAL:
        cut = g.minimize_conductance(par, F, opts.max_imbalance()); break;
      case FiedlerCut::FAST:
        cut = g.minimize_approx_conductance(par, F, opts.max_imbalance()); break;
      default: std::cerr << "WARNING: FiedlerCut not recognized" << std::endl;
      }
      return cut;
    }

    template<typename intt> PPt<intt>
    sub_ordering(const Graph<intt>& g, const Ordering& o) {
      switch (o) {
      case Ordering::NATURAL:
        return PPt<intt>(g.n()).identity();
        //case Ordering::RCM: return rcm(g);
      case Ordering::AMD: return amd(g);
      case Ordering::MMD: return mmd(g);
        //case Ordering::MLF: return mlf(g);
        //case Ordering::AND: return sparspak_and(g);
        //case Ordering::METIS: return metis_nd(g);
        //case Ordering::SCOTCH: return scotch_nd(g);
      default: std::cerr << "ERROR: sub ordering not supported!"
                         << std::endl;
      }
      return PPt<intt>(g.n()).identity();
    }

    template<typename intt> PPt<intt>
    sub_ordering(Graph<intt>&& g, const Ordering& o) {
      switch (o) {
      case Ordering::NATURAL:
        return PPt<intt>(g.n()).identity();
        //case Ordering::RCM: return rcm(std::move(g));
      case Ordering::AMD: return amd(std::move(g));
      case Ordering::MMD: return mmd(std::move(g));
        //case Ordering::MLF: return mlf(std::move(g));
        //case Ordering::AND: return sparspak_and(std::move(g));
        //case Ordering::METIS: return metis_nd(std::move(g));
        //case Ordering::SCOTCH: return scotch_nd(std::move(g));
      default: std::cerr << "ERROR: sub ordering not supported!"
                         << std::endl;
      }
      return PPt<intt>(g.n()).identity();
    }

    /**
     * gtmp should be a pointer to g if it may be modified, or nullptr.
     *
     * lvl is the current nested dissection level.
     *
     * connected denotes whether g is fully connected (true) or may not
     * be fully connected (false).
     *
     * no_singles denotes that there are no nodes in g with degree zero
     * (true), or that there might be nodes with degree zero.
     */
    template<typename scalart, typename intt>
    PPt<intt> spectral_nd_recursive
    (const Graph<intt>& g, Graph<intt>* gtmp, SpectralStats& stats,
     const NDOptions& opts, int lvl=0, bool connected=false,
     bool no_singles=false) {
      if (opts.verbose())
        std::cout << "## Spectral_ND on graph with n = "
                  << g.n() << " nnz = " << g.nnz() << std::endl;
      stats.levels = std::max(stats.levels, lvl);

      if (g.n() <= opts.dissection_cutoff()) {
        PPt<intt> p;
        if (gtmp) return sub_ordering(std::move(*gtmp), opts.sub_ordering());
        else return sub_ordering(g, opts.sub_ordering());
      }

      auto par = (lvl < opts.max_task_lvl()) ?
        ThreadingModel::TASK : ThreadingModel::SEQ;

      std::vector<intt> part;

      // if not specified that the graph is connected or has no
      // singles, then check for unconnected nodes
      if (!connected && !no_singles && g.unconnected_nodes(part)) {
        // there are a bunch of unconnected nodes, B is all the rest
        auto B = g.extract_domain_1(par, part);
        auto nA = g.n() - B.n();
        stats.unconnected_nodes += nA;
        if (opts.verbose())
          std::cout << "Graph with n = " << g.n() << " nnz = " << g.nnz()
                    << " has unconnected nodes!!" << std::endl
                    << "\tsplitting in " << nA
                    << " unconnected nodes, and a graph with "
                    << B.n() << " nodes" << std::endl;
        // B is not necessarily connected, but has no single nodes
        auto pB = spectral_nd_recursive<scalart>
          (B, &B, stats, opts, lvl+1, false, true);  // TODO should level be incremented?
        PPt<intt> p(g.n());
        // use identity for the first part (all the unconnected nodes),
        // and pB for the second part
        for (intt i=0, iA=0, iB=0; i<g.n(); i++)
          p[i] = (part[i] == 0) ? iA++ : nA + pB[iB++];
        return p;
      }

      if (g.dense_nodes(part)) {
        // B has a bunch of dense nodes, A is the rest
        auto A = g.extract_domain_1(par, part);
        auto nA = A.n();
        stats.dense_nodes += g.n() - nA;
        if (opts.verbose())
          std::cout << "Graph with n = " << g.n() << " nnz = " << g.nnz()
                    << " has " << (g.n() - nA) << " dense nodes"
                    << ", avg degree=" << double(g.nnz()) / g.n() << std::endl;
        // A might not be connected after removing the dense nodes, and
        // can have singles
        auto pA = spectral_nd_recursive<scalart>
          (A, &A, stats, opts, lvl+1, false, false);  // TODO should level be incremented?
        PPt<intt> p(g.n());
        // p = [pA I], identity for dense nodes
        for (intt i=0, iA=0, iB=0; i<g.n(); i++)
          p[i] = (part[i] == 1) ? pA[iA++] : nA + iB++;
        return p;
      }

      if (!connected && !g.connected(part)) {
        stats.unconnected_graphs++;
        Graph<intt> A, B;
        std::tie(A, B) = g.extract_domains(par, part);
        if (opts.verbose())
          std::cout << "Graph with n = " << g.n() << " nnz = " << g.nnz()
                    << " is not fully connected!!" << std::endl
                    << "\tsplitting in nA=" << A.n()
                    << " and nB=" << B.n() << std::endl;
        PPt<intt> pA, pB;
        if (par == ThreadingModel::TASK) {
#pragma omp task default(shared)
          { pA = spectral_nd_recursive<scalart>(A, &A, stats, opts, lvl+1, true); }
#pragma omp task default(shared)
          { pB = spectral_nd_recursive<scalart>(B, &B, stats, opts, lvl+1); }
#pragma omp taskwait
        } else {
          pA = spectral_nd_recursive<scalart>(A, &A, stats, opts, lvl+1, true);
          pB = spectral_nd_recursive<scalart>(B, &B, stats, opts, lvl+1);
        }
        PPt<intt> p(g.n());
        for (intt i=0, iA=0, iB=0; i<g.n(); i++)
          p[i] = (part[i] == 0) ? pA[iA++] : pA.n() + pB[iB++];
        return p;
      }

      auto F = multilevel_Fiedler<scalart>(par, g, stats, opts);
      scalart cut = get_Fiedler_cut_value(par, g, F.data(), opts);
      if (opts.verbose()) {
        auto mM = std::minmax_element(F.begin(), F.end());
        std::cout << "# Fiedler min = " << *(mM.first)
                  << ", median = " << median(g.n(), F.data())
                  << ", midway = " << (*(mM.first) + *(mM.second)) / 2.
                  << ", avg = " << sum(par, g.n(), F.data()) / g.n()
                  << ", optimal = "
                  << g.minimize_conductance(par, F.data(), opts.max_imbalance())
                  << ", fast = "
                  << g.minimize_approx_conductance(par, F.data(), opts.max_imbalance())
                  << ", max = " << *(mM.second) << std::endl;
      }
      part.resize(g.n());

      // Kernighan_Lin(g, F, cut, part, opts.verbose());

      g.vertex_separator_from_Fiedler
        (par, F.data(), cut, part, opts.edge_to_vertex());

      Graph<intt> A, B;
      std::tie(A, B) = g.extract_domains(par, part);
      if (A.n() == g.n() || B.n() == g.n()) {
        stats.empty_subdomains++;
        if (opts.verbose())
          std::cout << "WARNING: ending the recursion early"
                    << ", n=" << g.n() << " nnz=" << g.nnz()
                    << " A.n=" << A.n() << " A.nnz=" << A.nnz()
                    << " B.n=" << B.n() << " B.nnz=" << B.nnz()
                    << std::endl;
        if (gtmp) return sub_ordering(std::move(*gtmp), opts.sub_ordering());
        else return sub_ordering(g, opts.sub_ordering());
      }

      PPt<intt> pA, pB;
      if (par == ThreadingModel::TASK) {
#pragma omp task default(shared)
        { pA = spectral_nd_recursive<scalart>(A, &A, stats, opts, lvl+1); }
#pragma omp task default(shared)
        { pB = spectral_nd_recursive<scalart>(B, &B, stats, opts, lvl+1); }
#pragma omp taskwait
      } else {
        pA = spectral_nd_recursive<scalart>(A, &A, stats, opts, lvl+1);
        pB = spectral_nd_recursive<scalart>(B, &B, stats, opts, lvl+1);
      }

      // TODO threading
      PPt<intt> p(g.n());
      const auto nA = pA.n();
      const auto nAB = nA + pB.n();
      for (intt i=0, iA=0, iB=0, iS=0; i<g.n(); i++) {
        if (part[i] == 0) p[i] = pA[iA++];
        else if (part[i] == 1) p[i] = nA + pB[iB++];
        else p[i] = nAB + iS++;
      }
      return p;
    }

    template<typename intt> PPt<intt>
    spectral_nd(const Graph<intt>& g, SpectralStats& stats,
                const NDOptions& opts) {
      PPt<intt> p;
#pragma omp parallel default(shared)
#pragma omp single nowait
      {
        switch (opts.precision()) {
        case Precision::SINGLE: {
          p = spectral_nd_recursive<float,intt>(g, nullptr, stats, opts);
        } break;
        case Precision::DOUBLE: {
          p = spectral_nd_recursive<double,intt>(g, nullptr, stats, opts);
        }}
      }
      p.set_Pt();
      return p;
    }

    template<typename scalar_t,typename intt>
    std::unique_ptr<SeparatorTree<intt>>
    spectral_nd(const CSRMatrix<scalar_t,intt>& g,
                std::vector<intt>& perm, std::vector<intt>& iperm,
                const NDOptions& opts) {
      SpectralStats stats;
      std::vector<intt,NoInit<intt>> ptr(g.size()+1), ind(g.nnz());
      intt e = 0;
      for (intt j=0; j<g.size(); j++) {
        ptr[j] = e;
        for (intt t=g.ptr(j); t<g.ptr(j+1); t++)
          if (g.ind(t) != j) ind[e++] = g.ind(t);
      }
      ptr[g.size()] = e;
      if (!e)
        if (mpi_root())
          std::cerr << "# WARNING: matrix seems to be diagonal!" << std::endl;
      ind.resize(e);
      Graph<intt> graph(std::move(ptr), std::move(ind));
      auto p = spectral_nd(graph, stats, opts);
      perm.assign(p.P(), p.P()+g.size());
      iperm.assign(p.Pt(), p.Pt()+g.size());
      return build_sep_tree_from_perm(graph.ptr(), graph.ind(), perm, iperm);
    }

  } // end namespace ordering
} // end namespace strumpack

#endif // STRUMPACK_ORDERING_SPECTRAL_HPP
