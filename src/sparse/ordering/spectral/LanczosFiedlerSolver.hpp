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
#ifndef STRUMPACK_ORDERING_LANCZOSFIEDLER_HPP
#define STRUMPACK_ORDERING_LANCZOSFIEDLER_HPP

#include <random>
#include <iostream>

#include "sparse/ordering/Graph.hpp"
// #include "dense/BLASLAPACKWrapper.hpp"
#include "NDOptions.hpp"
// #include "LanczosFiedlerSolverGPU.hpp"

#define FUSE_LOOPS 1

namespace strumpack {
  namespace ordering {

    template<ThreadingModel par, typename intt, typename scalart> bool
    Lanczos_Fiedler(const Graph<intt>& g, scalart* f,
                    int& its, const NDOptions& opts,
                    std::ostream& log = std::cout) {
      static_assert(par != ThreadingModel::LOOP,
                    "Lanczos_Fiedler does not work with LOOP parallelism,"
                    " use SEQ, PAR or TASK instead.");
      const int m = opts.Fiedler_restart();
      const intt n = g.n();
      std::unique_ptr<scalart[]> work_(new scalart[13*m-1+n*(m+1)]);
      auto alpha = work_.get(); // new scalart[m];
      auto beta  = alpha + m;   // new scalart[m];
      auto Z     = beta + m;    // new scalart[m];
      auto V     = Z + m;       // new scalart[n*(m+1)];
      auto D     = V + n*(m+1); // new scalart[m];
      auto E     = D + m;       // new scalart[m-1];
      auto work  = E + m-1;     // new scalart[8*m];
      scalart res = 1., lambda;
      its = 0;
      bool converged = false;
      if (opts.verbose())
        log << "# running Lanczos on graph with "
            << n << " vertices" << std::endl;

      do {
        // TODO combine copy with add below
        // deflate the constant vector which is the eigenvector
        // belonging to the eigenvalue zero
        // compute V(0) = V(0) - dot([1 .. 1]/sqrt(n), V(0)) * [1 .. 1]'/sqrt(n)

#if defined(FUSE_LOOPS)
        if constexpr (par == ThreadingModel::TASK) {
          double tmp1 = 0, tmp2 = 0.;
#pragma omp taskloop simd reduction(+:tmp1) default(shared)
          for (int i=0; i<n; i++) tmp1 += f[i];
          tmp1 /= n;
#pragma omp taskloop simd reduction(+:tmp2) default(shared)
          for (int i=0; i<n; i++) {
            V[i] = f[i] - tmp1;
            tmp2 += double(V[i])*double(V[i]);
          }
          tmp2 = std::sqrt(tmp2);
          if (tmp2 > strumpack::blas::lamch<scalart>('P')) {
            tmp2 = 1.0 / tmp2;
#pragma omp taskloop simd default(shared)
            for (int i=0; i<n; i++) V[i] *= tmp2;
          } else {
            random_vector(n, V);
            add<par>(n, V, -sum<par>(n, V)/n);
            normalize<par>(n, V);
          }
        }
        else if constexpr (par == ThreadingModel::SEQ) {
          double tmp1 = 0, tmp2 = 0.;
          for (int i=0; i<n; i++) tmp1 += f[i];
          tmp1 /= n;
          for (int i=0; i<n; i++) {
            V[i] = f[i] - tmp1;
            tmp2 += double(V[i])*double(V[i]);
          }
          tmp2 = std::sqrt(tmp2);
          if (tmp2 > strumpack::blas::lamch<scalart>('P')) {
            tmp2 = 1.0 / tmp2;
            for (int i=0; i<n; i++) V[i] *= tmp2;
          } else {
            random_vector(n, V);
            add<par>(n, V, -sum<par>(n, V)/n);
            normalize<par>(n, V);
          }
        }
        else
#endif
          {
            std::copy(f, f+n, V);
            add<par>(n, V, -sum<par>(n, V)/n);
            auto tmp = norm<par>(n, V);
            if (tmp > strumpack::blas::lamch<scalart>('P'))
              scale<par>(n, V, scalart(1.)/tmp);
            else {
              random_vector(n, V);
              add<par>(n, V, -sum<par>(n, V)/n);
              normalize<par>(n, V);
            }
          }

        for (int i=0; i<m; i++) {
#if defined(FUSE_LOOPS)
          if constexpr (par == ThreadingModel::TASK) {
            auto x = V + i*n;
            auto y = V + (i+1)*n;
            auto z = V + (i-1)*n;
            double tmp2 = 0;
            if (i > 0) {
              double tmp1 = beta[i-1];
#pragma omp taskloop simd reduction(+:tmp2) default(shared)
              for (int l=0; l<n; l++) {
                auto lo = g.ptr(l);
                auto hi = g.ptr(l+1);
                const auto hii = g.ind() + hi;
                double yi = (hi - lo) * x[l] - tmp1 * z[l];
                for (auto pj=g.ind()+lo; pj!=hii; pj++)
                  yi -= x[*pj];
                y[l] = yi;
                tmp2 += x[l] * yi;
              }
            } else {
#pragma omp taskloop simd reduction(+:tmp2) default(shared)
              for (int l=0; l<n; l++) {
                auto lo = g.ptr(l);
                auto hi = g.ptr(l+1);
                const auto hii = g.ind() + hi;
                double yi = (hi - lo) * x[l];
                for (auto pj=g.ind()+lo; pj!=hii; pj++)
                  yi -= x[*pj];
                y[l] = yi;
                tmp2 += x[l] * yi;
              }
            }
            alpha[i] = tmp2;
          }
          if constexpr (par == ThreadingModel::SEQ) {
            auto x = V + i*n;
            auto y = V + (i+1)*n;
            auto z = V + (i-1)*n;
            int lo, hi = g.ptr(0);
            double tmp2 = 0;
            scalart yi;
            if (i > 0) {
              auto bi = beta[i-1];
              for (int l=0; l<n; l++) {
                lo = hi;
                hi = g.ptr(l+1);
                const auto hii = g.ind() + hi;
                yi = (hi - lo) * x[l] - bi * z[l];
                for (auto pj=g.ind()+lo; pj!=hii; pj++)
                  yi -= x[*pj];
                y[l] = yi;
                tmp2 += x[l] * yi;
              }
            } else {
              for (int l=0; l<n; l++) {
                lo = hi;
                hi = g.ptr(l+1);
                const auto hii = g.ind() + hi;
                yi = (hi - lo) * x[l];
                for (auto pj=g.ind()+lo; pj!=hii; pj++)
                  yi -= x[*pj];
                y[l] = yi;
                tmp2 += x[l] * yi;
              }
            }
            alpha[i] = tmp2;
          }
          else
#endif
            {
              g.template Laplacian<par>(V+i*n, V+(i+1)*n);
              if (i > 0) axpy<par>(n, -beta[i-1], V+(i-1)*n, V+(i+1)*n);
              alpha[i] = dot<par>(n, V+(i+1)*n, V+i*n);
            }
          if (opts.Fiedler_reorthogonalize()) {
            axpy<par>(n, -alpha[i], V+i*n, V+(i+1)*n);
            auto tmp = dot<par>(n, V+(i+1)*n, V+i*n);
            axpy<par>(n, -tmp, V+i*n, V+(i+1)*n);
            alpha[i] += tmp;
            for (int o=0; o<i; o++)
              axpy<par>(n, -dot<par>(n, V+(i+1)*n, V+o*n), V+o*n, V+(i+1)*n);
            add<par>(n, V+(i+1)*n, -sum<par>(n, V+(i+1)*n)/n);
            beta[i] = norm<par>(n, V+(i+1)*n);
            scale<par>(n, V+(i+1)*n, scalart(1.)/beta[i]);
          } else {
#if defined(FUSE_LOOPS)
            if constexpr (par == ThreadingModel::TASK) {
              double tmp1 = 0, tmp2 = 0;
              auto x = V + (i+1)*n;
              auto y = V + i*n;
              auto ai = alpha[i];
#pragma omp taskloop simd reduction(+:tmp1) default(shared)
              for (int l=0; l<n; l++) {
                x[l] -= ai * y[l];
                tmp1 += x[l];
              }
              tmp1 /= n;
#pragma omp taskloop simd reduction(+:tmp2) default(shared)
              for (int l=0; l<n; l++) {
                x[l] -= tmp1;
                tmp2 += double(x[l]) * x[l];
              }
              tmp2 = std::sqrt(tmp2);
              beta[i] = tmp2;
              tmp2 = 1. / tmp2;
#pragma omp taskloop simd default(shared)
              for (int l=0; l<n; l++)
                x[l] *= tmp2;
            }
            else if constexpr (par == ThreadingModel::SEQ) {
              double tmp1 = 0, tmp2 = 0;
              auto x = V + (i+1)*n;
              auto y = V + i*n;
              auto ai = alpha[i];
              for (int l=0; l<n; l++) {
                x[l] -= ai * y[l];
                tmp1 += x[l];
              }
              tmp1 /= n;
              for (int l=0; l<n; l++) {
                x[l] -= tmp1;
                tmp2 += double(x[l]) * x[l];
              }
              tmp2 = std::sqrt(tmp2);
              beta[i] = tmp2;
              tmp2 = 1. / tmp2;
              for (int l=0; l<n; l++)
                x[l] *= tmp2;
            }
            else
#endif
              {
                axpy<par>(n, -alpha[i], V+i*n, V+(i+1)*n);
                add<par>(n, V+(i+1)*n, -sum<par>(n, V+(i+1)*n)/n);
                beta[i] = norm<par>(n, V+(i+1)*n);
                scale<par>(n, V+(i+1)*n, scalart(1.)/beta[i]);
              }
          }

          // Cholesky factorization of T = spdiags(alpha, beta..)
          if (i == 0) D[i] = std::sqrt(alpha[i]);
          else {
            E[i-1] = beta[i-1] / D[i-1];
            D[i] = std::sqrt(alpha[i] - E[i-1]*E[i-1]);
          }
          // initial guess for Z, is [Z 1]
          Z[i] = scalart(1.);

          if (i % opts.Lanczos_skip() == 0 || i == (m-1)) {
            // inverse iteration for the smallest eigenpair of T
            int info = stev_Fiedler_inverse_iteration
              (i+1, m, opts.Fiedler_tol()/3.0, alpha, beta,
               D, E, Z, lambda, work);
            if (info)
              log << "*stevx finished with info = " << info << std::endl;
            res = std::abs(beta[i]*Z[i]);

            // if (opts.Fiedler_monitor_residual())
            //   log << "# Lanczos iteration " << its
            //       << " ||A f - " << lambda << " f||/||f|| = "
            //       << res << std::endl;
            converged = res < opts.Fiedler_tol();
            // if (opts.Fiedler_monitor_orthogonality()) {
            //   std::unique_ptr<scalart[]> orth_work
            //     (new scalart[(i+2)*n + (i+2)*(i+2)]);
            //   auto VdefV = orth_work.get(); // new scalart[(i+2)*n];
            //   auto VtV = VdefV + (i+2)*n;   // new scalart[(i+2)*(i+2)];
            //   // set deflation vector
            //   std::fill(VdefV, VdefV+n, scalart(1.) / std::sqrt(n));
            //   for (int c=0; c<i+1; c++)
            //     std::copy(V+c*n, V+(c+1)*n, VdefV+(c+1)*n);
            //   gemm('T', 'N', i+2, i+2, n, scalart(1.), VdefV, n, VdefV, n,
            //        scalart(0.), VtV, i+2);
            //   for (int c=0; c<i+2; c++) VtV[c+c*(i+2)] -= scalart(1.);
            //   auto orth = lange('F', i+2, i+2, VtV, i+2);
            //   log << "# Lanczos iteration " << its
            //     //<< " ||V'V-I||_F = " << orth << std::endl;
            //       << " orthogonality = " << orth << std::endl;
            // }
            // if (opts.Fiedler_monitor_true_residual()) {
            //   std::unique_ptr<scalart[]> res_work(new scalart[n]);
            //   auto resvec = res_work.get(); //new scalart[n];
            //   gemv<par>('N', n, i+1, scalart(1.), V, n, Z, 1,
            //             scalart(0.), f, 1);
            //   g.template Laplacian<par>(f, resvec);
            //   axpy<par>(n, -lambda, f, resvec);
            //   log << "# Lanczos iteration " << its
            //       << " trueres = " << norm<par>(n, resvec) << std::endl;
            // }
          }
          its++;
          // Compute approximate Fiedler vector
          if (converged || its >= opts.Fiedler_maxit() || i == (m-1) )
            gemv<par>('N', n, i+1, scalart(1.), V, n, Z, 1,
                      scalart(0.), f, 1);
          if (converged || its >= opts.Fiedler_maxit()) break;
        }
      } while (!converged && its < opts.Fiedler_maxit());
      if (!converged && opts.verbose())
        log << "WARNING!: Lanczos did not converge in "
            << opts.Fiedler_maxit() << " iterations"
            << ", n = " << g.n() << " nnz = " << g.nnz()
            << " lambda_0 = " << lambda << " res = " << res
            << std::endl;
      else if (opts.Fiedler_print_convergence() || opts.verbose())
        log << "# Lanczos iteration " << its
            << " ||A f - " << lambda << " f||/||f|| = "
            <<  res << std::endl;
      return converged;
    }


    template<typename intt, typename scalart> bool
    Lanczos_Fiedler(ThreadingModel par, const Graph<intt>& g, scalart* f,
                    int& its, const NDOptions& opts,
                    std::ostream& log = std::cout) {
      if (opts.use_gpu()) { // && g.n() >= opts.gpu_threshold())
        //std::cout << "ERROR: GPU not supported yet!!!" << std::endl;
        //   if (g.n() > 50000) // TODO tune, make options
        //     return Lanczos_Fiedler_GPU_large(g, f, its, opts, log);
        //   else if (g.n() > 1000)
        //     return Lanczos_Fiedler_GPU_small(g, f, its, opts, log);
      }
      // TODO make this value 200 an option, tune this!!

      if (g.n() < 200) par = ThreadingModel::SEQ;
      switch (par) {
      case ThreadingModel::SEQ:
        return Lanczos_Fiedler<ThreadingModel::SEQ>(g, f, its, opts, log);
      case ThreadingModel::PAR:
        return Lanczos_Fiedler<ThreadingModel::PAR>(g, f, its, opts, log);
      case ThreadingModel::TASK:
        return Lanczos_Fiedler<ThreadingModel::TASK>(g, f, its, opts, log);
      case ThreadingModel::LOOP:
      default:
        static_assert
          (true, "Lanczos_Fiedler does not work with LOOP parallelism,"
           " use SEQ, PAR or TASK instead.");
        return false;
      }
    }


    template<ThreadingModel par, typename intt, typename scalart> bool
    implicit_restarted_Lanczos_Fiedler
    (const Graph<intt>& g, scalart* f, int& its,
     const NDOptions& opts, std::ostream& log = std::cout) {
      // TODO
      // - allocate workspace for LAPACK calls once
      // - store only upper/lower part of T
      // - apply Givens manually, then stev, iso syev

      const int m = opts.Fiedler_restart();
      const int s = opts.Fiedler_subspace();
      const intt n = g.n();
      auto Z   =                  // new scalart[m*m];
        new scalart[m*m+n*(m+1)+(m+1)*(m+1)+m+m+n*s+n+s];
      auto V   = Z + m*m;         // new scalart[n*(m+1)];
      auto T   = V + n*(m+1);     // new scalart[(m+1)*(m+1)];
      auto W   = T + (m+1)*(m+1); // new scalart[m];
      auto z   = W + m;           // new scalart[m];
      auto Y   = z + m;           // new scalart[n*s];
      auto q   = Y + n*s;         // new scalart[n];
      auto sig = q + n;           // new scalart[s];
      int ldZ = m, ldT = m+1;
      scalart alpha, beta = 0, res = 1.;
      its = 0;
      bool converged = false;
      if (opts.verbose())
        log << "# running ImplicitRestartedLanczos on graph with "
            << n << " vertices" << std::endl;
      std::copy(f, f+n, V);
      // deflate the constant vector which is the eigenvector
      // belonging to the eigenvalue zero
      // compute V(0) = V(0) - dot([1 .. 1]/sqrt(n), V(0)) * [1 .. 1]'/sqrt(n)
      add<par>(n, V, -sum<par>(n, V)/n);
      normalize<par>(n, V);
      int i = 0;
      do {
        for ( ; i<m; i++) {
          g.template Laplacian<par>(V+i*n, V+(i+1)*n);
          if (i > 0) axpy<par>(n, -beta, V+(i-1)*n, V+(i+1)*n);
          alpha = dot<par>(n, V+(i+1)*n, V+i*n);
          axpy<par>(n, -alpha, V+i*n, V+(i+1)*n);
          beta = norm<par>(n, V+(i+1)*n);
          scale<par>(n, V+(i+1)*n, scalart(1.)/beta);
          T[i+i*ldT] = alpha;
          T[(i+1)+i*ldT] = T[i+(i+1)*ldT] = beta;
          if (i % opts.Lanczos_skip() == 0 || i == (m-1)) {
            if (its < m) {
              for (int j=0; j<=i; j++) Z[j] = T[j+j*ldT];
              for (int j=0; j< i; j++) Z[j+ldZ] = T[j+1+j*ldT];
              auto info = stevx_1(i+1, Z, Z+ldZ, z, m, W[0]);
              if (info)
                log << "*stevx finished with info = "
                    << info << std::endl;
            } else {
              for (int c=0; c<=i; c++)
                for (int r=0; r<=i; r++)
                  Z[r+c*ldZ] = T[r+c*ldT];
              // TODO first manually transform T/Z to a tridiagonal matrix
              // with Givens rotations? Then call stevx.
              auto info = syevx_i(1, opts.Fiedler_tol(), i+1, Z, ldZ, W, z, ldZ);
              if (info)
                log << "*syev finished with info = " << info << std::endl;
            }
            res = std::abs(beta*z[i]);
            if (opts.Fiedler_monitor_residual())
              log << "# ImplicitRestartedLanczos iteration " << its
                  << " mu_0 = " << W[0] << " res = " << res
                  << std::endl;
            converged = res < opts.Fiedler_tol();
            if (opts.Fiedler_monitor_orthogonality()) {
              auto VdefV = new scalart[(i+2)*n];
              auto VtV = new scalart[(i+2)*(i+2)];
              // set deflation vector
              std::fill(VdefV, VdefV+n, scalart(1.) / std::sqrt(n));
              for (int c=0; c<i+1; c++)
                std::copy(V+c*n, V+(c+1)*n, VdefV+(c+1)*n);
              gemm('T', 'N', i+2, i+2, n, scalart(1.), VdefV, n, VdefV, n,
                   scalart(0.), VtV, i+2);
              for (int c=0; c<i+2; c++) VtV[c+c*(i+2)] -= scalart(1.);
              auto orth = lange('F', i+2, i+2, VtV, i+2);
              log << "# ImplicitRestartedLanczos iteration " << its
                //<< " ||V'V-I||_F = " << orth << std::endl;
                  << " orthogonality = " << orth << std::endl;
              delete[] VtV;
              delete[] VdefV;
            }
            if (opts.Fiedler_monitor_true_residual()) {
              auto resvec = new scalart[n];
              gemv<par>('N', n, i+1, scalart(1.), V, n, z, 1,
                        scalart(0.), f, 1);
              g.template Laplacian<par>(f, resvec);
              axpy<par>(n, -W[0], f, resvec);
              log << "# ImplicitRestartedLanczos iteration " << its
                  << " trueres = " << norm<par>(n, resvec) << std::endl;
              delete[] resvec;
            }
          }
          its++;
          // Compute approximate Fiedler vector
          if (converged || its >= opts.Fiedler_maxit() || i == (m-1)) {
            gemv<par>('N', n, i+1, scalart(1.), V, n, z, 1,
                      scalart(0.), f, 1);
            break;
          }
        }
        if (converged || its >= opts.Fiedler_maxit()) break;
        if (its == m) {
          // TODO use divide-and-conquer algorithm
          for (int j=0; j<m; j++) W[j] = T[j+j*ldT];
          for (int j=0; j<m-1; j++) z[j] = T[j+1+j*ldT];
          auto info = stev('V', m, W, z, Z, ldZ);
          if (info)
            log << "*stev finished with info = " << info << std::endl;
        } else {
          for (int c=0; c<m; c++)
            for (int r=0; r<m; r++)
              Z[r+c*ldZ] = T[r+c*ldT];
          auto info = syevd('V', 'U', m, Z, ldZ, W);
          if (info)
            log << "*syev finished with info = " << info << std::endl;
        }
        gemm('N', 'N', n, s, m, scalart(1.), V, n, Z, m, scalart(0.), Y, n);
        auto v = V+m*n;
        g.template Laplacian<par>(v, q);
        alpha = dot<par>(n, v, q);
        for (int j=0; j<s; j++)
          sig[j] = T[m+(m-1)*ldT] * Z[(m-1)+j*ldZ];
        axpy<par>(n, -alpha, v, q);
        for (int j=0; j<s; j++)
          axpy<par>(n, -sig[j], Y+j*n, q);
        beta = norm<par>(n, q);
        scale<par>(n, q, scalart(1.)/beta);
        for (int j=0; j<s; j++)
          std::copy(Y+j*n, Y+(j+1)*n, V+j*n);
        std::copy(V+m*n, V+(m+1)*n, V+s*n);
        std::copy(q, q+n, V+(s+1)*n);
        for (int c=0; c<=m; c++)
          for (int r=0; r<=m; r++)
            T[r+c*ldT] = scalart(0.);
        for (int j=0; j<s; j++) {
          T[j+j*ldT] = W[j];
          T[j+s*ldT] = T[s+j*ldT] = sig[j];
        }
        T[s+s*ldT] = alpha;
        T[(s+1)+s*ldT] = T[s+(s+1)*ldT] = beta;

        // TODO check the residual here as well? see matlab code

        i = s+1;
      } while (!converged && its < opts.Fiedler_maxit());
      if (!converged)
        log << "WARNING!: ImplicitRestartedLanczos did not converge in "
            << opts.Fiedler_maxit() << " iterations"
            << ", n = " << g.n() << " nnz = " << g.nnz()
            << " lambda_0 = " << W[0] << " res = " << res
            << std::endl;
      else if (opts.Fiedler_print_convergence() || opts.verbose())
        log << "# ImplicitRestartedLanczos iteration " << its
            << " lambda_0 = " << W[0] << " res = " << res << std::endl;
      delete[] Z;
      return converged;
    }


    template<typename intt, typename scalart> bool
    implicit_restarted_Lanczos_Fiedler
    (ThreadingModel par, const Graph<intt>& g, scalart* f, int& its,
     const NDOptions& opts, std::ostream& log = std::cout) {
      // TODO make this value 200 an option, tune this!!
      if (g.n() < 200) par = ThreadingModel::SEQ;
      switch (par) {
      case ThreadingModel::SEQ:
        return implicit_restarted_Lanczos_Fiedler
          <ThreadingModel::SEQ>(g, f, its, opts, log);
      case ThreadingModel::PAR:
        return implicit_restarted_Lanczos_Fiedler
          <ThreadingModel::PAR>(g, f, its, opts, log);
      case ThreadingModel::TASK:
        return implicit_restarted_Lanczos_Fiedler
          <ThreadingModel::TASK>(g, f, its, opts, log);
      case ThreadingModel::LOOP:
      default:
        static_assert
          (true, "implicit_restarted_Lanczos_Fiedler does not work"
           " with LOOP parallelism, use SEQ, PAR or TASK instead.");
        return false;
      }
    }

  } // end namespace ordering
} // end namespace strumpack

#endif // STRUMPACK_ORDERING_LANCZOSFIEDLER_HPP
