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
#ifndef BICGSTAB_HPP
#define BICGSTAB_HPP
#include <iostream>
#include <iomanip>

#include "StrumpackParameters.hpp"
#include "CompressedSparseMatrix.hpp"

namespace strumpack {

  /**
   * http://www.netlib.org/templates/matlab/bicgstab.m
   */
  template<typename scalar_t,
           typename real_t = typename RealType<scalar_t>::value_type>
  real_t BiCGStab
  (const std::function<void(const scalar_t*,scalar_t*)>& spmv,
   const std::function<void(scalar_t*)>& preconditioner,
   std::size_t n, scalar_t* x, const scalar_t* b, real_t rtol, real_t atol,
   int& totit, int maxit, bool non_zero_guess, bool verbose) {
    real_t bnrm2 = blas::nrm2(n, b, 1);
    if (bnrm2 == 0.0) return real_t(0.0);
    std::unique_ptr<scalar_t[]> work(new scalar_t[8*n]);
    auto r = work.get();
    auto r_tld = r + n;
    auto p_hat = r + 2 * n;
    auto s_hat = r + 3 * n;
    auto p = r + 4 * n;
    auto v = r + 5 * n;
    auto s = r + 6 * n;
    auto t = r + 7 * n;
    if (non_zero_guess) {      // compute initial residual
      spmv(x, r);
      blas::axpby(n, scalar_t(1.), b, 1, scalar_t(-1.), r, 1);
    } else {
      std::copy(b, b+n, r);
      std::fill(x, x+n, scalar_t(0.));
    }
    real_t resid = blas::nrm2(n, r, 1);
    real_t error = resid / bnrm2;
    if (verbose)
      std::cout << "BiCGStab it. " << totit
                << "\tres = " << std::setw(12) << resid
                << "\trel.res = " << std::setw(12) << error << std::endl;
    if (error <= rtol || resid <= atol)
      return error;
    scalar_t alpha = scalar_t(0.), rho, rho_1 = scalar_t(0.), beta;
    scalar_t omega = scalar_t(1.);
    std::copy(r, r+n, r_tld);
    for (totit=1; totit<=maxit; totit++) {
      rho = blas::dotc(n, r_tld, 1, r, 1);
      if (rho == scalar_t(0.0)) break;
      if (totit > 1) {
        beta = (rho / rho_1) * (alpha / omega);
        // p = r + beta (p - omega v)
        blas::axpy(n, -omega, v, 1, p, 1);
        blas::axpby(n, scalar_t(1), r, 1, beta, p, 1);
      } else std::copy(r, r+n, p);
      std::copy(p, p+n, p_hat);               // p_hat = M \ p
      preconditioner(p_hat);
      spmv(p_hat, v);                         // v = A * p_hat
      alpha = rho / blas::dotc(n, r_tld, 1, v, 1);
      std::copy(r, r+n, s);                   // s = r_1 - alpha v
      blas::axpy(n, -alpha, v, 1, s, 1);
      if (blas::nrm2(n, s, 1) < atol) {       // early convergence check
        blas::axpy(n, alpha, p_hat, 1, x, 1);
        spmv(x, r);
        blas::axpby(n, scalar_t(1.), b, 1, scalar_t(-1.), r, 1);
        resid = blas::nrm2(n, r, 1);
        error = resid / bnrm2;
        if (verbose)
          std::cout << "BiCGStab it. " << totit
                    << "\tres = " << std::setw(12) << resid
                    << "\trel.res = " << std::setw(12) << error << std::endl;
        break;
      }
      std::copy(s, s+n, s_hat);               // s_hat = M \ s
      preconditioner(s_hat);
      spmv(s_hat, t);                         // t = A*s_hat
      omega = blas::dotc(n, t, 1, s, 1) / blas::dotc(n, t, 1, t, 1);    // omega = ( t'*s) / ( t'*t );
      blas::axpy(n, alpha, p_hat, 1, x, 1);   // x = x + alpha*p_hat + omega*s_hat
      blas::axpy(n, omega, s_hat, 1, x, 1);
      std::copy(s, s+n, r);                   // r = s - omega*t
      blas::axpy(n, -omega, t, 1, r, 1);
      resid = blas::nrm2(n, r, 1);
      error = resid / bnrm2;
      if (verbose)
        std::cout << "BiCGStab it. " << totit
                  << "\tres = " << std::setw(12) << resid
                  << "\trel.res = " << std::setw(12) << error << std::endl;
      if (error <= rtol || resid <= atol) break;
      if (omega == scalar_t(0.0)) break;
      rho_1 = rho;
    }
    error = blas::nrm2(n, r, 1) / bnrm2;
    return error;
  }

} // end namespace strumpack

#endif // BICGSTAB_HPP
