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
#ifndef GMRES_HPP
#define GMRES_HPP
#include <iostream>
#include <iomanip>

#include "StrumpackParameters.hpp"
#include "CompressedSparseMatrix.hpp"

namespace strumpack {

  /*
   * This is left preconditioned restarted GMRes.
   *
   *  Input vectors x and b have stride 1, length n
   */
  template<typename scalar_t,
           typename real_t = typename RealType<scalar_t>::value_type>
  real_t GMRes
  (const std::function<void(const scalar_t*,scalar_t*)>& spmv,
   const std::function<void(scalar_t*)>& preconditioner,
   std::size_t n, scalar_t* x, const scalar_t* b, real_t rtol, real_t atol,
   int& totit, int maxit, int restart, GramSchmidtType GStype,
   bool non_zero_guess, bool verbose) {
    if (restart > maxit) restart = maxit;
    std::unique_ptr<scalar_t[]> work
      (new scalar_t[restart + restart + restart+1 +
                    (restart+1)*restart + n*(restart+1) + n]);
    auto givens_c = work.get();
    auto givens_s = givens_c + restart;
    auto b_ = givens_s + restart;
    auto hess = b_ + restart+1;
    auto V = hess + (restart+1)*restart;
    auto b_prec = V + n*(restart+1);

    int ldh = restart+1;
    real_t rho;
    real_t rho0 = real_t(0.);
    blas::copy(n, b, 1, b_prec, 1);
    preconditioner(b_prec);

    bool no_conv = true;
    totit = 0;
    while (no_conv) {
      if (non_zero_guess || totit>0) {
        spmv(x, V);
        preconditioner(V);
        blas::axpby(n, scalar_t(1.), b_prec, 1, scalar_t(-1.), V, 1);
      } else {
        std::copy(b_prec, b_prec+n, V);
        std::fill(x, x+n, scalar_t(0.));
      }
      rho = blas::nrm2(n, V, 1);
      if (totit==0) rho0 = rho;
      if (rho/rho0 < rtol || rho < atol) { no_conv = false; break; }
      blas::scal(n, scalar_t(1./rho), V, 1);
      b_[0] = rho;
      for (int i=1; i<=restart; i++) b_[i] = scalar_t(0.);

      int nrit = restart-1;
      if (verbose)
        std::cout << "GMRES it. " << totit << "\tres = "
                  << std::setw(12) << rho
                  << "\trel.res = " << std::setw(12)
                  << rho/rho0 << "\t restart!" << std::endl;
      for (int it=0; it<restart; it++) {
        totit++;
        spmv(&V[it*n], &V[(it+1)*n]);
        preconditioner(&V[(it+1)*n]);

        if (GStype == GramSchmidtType::CLASSICAL) {
          blas::gemv
            ('C', n, it+1, scalar_t(1.), V, n, &V[(it+1)*n], 1,
             scalar_t(0.), &hess[it*ldh], 1);
          blas::gemv
            ('N', n, it+1, scalar_t(-1.), V, n, &hess[it*ldh], 1,
             scalar_t(1.), &V[(it+1)*n], 1);
        } else if (GStype == GramSchmidtType::MODIFIED) {
          for (int k=0; k<=it; k++) {
            hess[k+it*ldh] = blas::dotc(n, &V[k*n], 1, &V[(it+1)*n], 1);
            blas::axpy
              (n, scalar_t(-hess[k+it*ldh]), &V[k*n], 1, &V[(it+1)*n], 1);
          }
        }
        hess[it+1+it*ldh] = blas::nrm2(n, &V[(it+1)*n], 1);
        blas::scal(n, scalar_t(1.)/hess[it+1+it*ldh], &V[(it+1)*n], 1);

        for (int k=1; k<it+1; k++) {
          scalar_t gamma = blas::my_conj(givens_c[k-1])*hess[k-1+it*ldh]
            + blas::my_conj(givens_s[k-1])*hess[k+it*ldh];
          hess[k+it*ldh] = -givens_s[k-1]*hess[k-1+it*ldh]
            + givens_c[k-1]*hess[k+it*ldh];
          hess[k-1+it*ldh] = gamma;
        }
        scalar_t delta =
          std::sqrt(std::pow(std::abs(hess[it+it*ldh]),scalar_t(2))
                    + std::pow(hess[it+1+it*ldh],scalar_t(2)));
        givens_c[it] = hess[it+it*ldh] / delta;
        givens_s[it] = hess[it+1+it*ldh] / delta;
        hess[it+it*ldh] = blas::my_conj(givens_c[it])*hess[it+it*ldh]
          + blas::my_conj(givens_s[it])*hess[it+1+it*ldh];
        b_[it+1] = -givens_s[it]*b_[it];
        b_[it] = blas::my_conj(givens_c[it])*b_[it];
        rho = std::abs(b_[it+1]);
        if (verbose)
          std::cout << "GMRES it. " << totit << "\tres = "
                    << std::setw(12) << rho
                    << "\trel.res = " << std::setw(12)
                    << rho/rho0 << std::endl;
        if ((rho < atol) || (rho/rho0 < rtol) || (totit >= maxit)) {
          no_conv = false;
          nrit = it;
          break;
        }
      }
      blas::trsv('U', 'N', 'N', nrit+1, hess, ldh, b_, 1);
      blas::gemv
        ('N', n, nrit+1, scalar_t(1.), V, n, b_, 1, scalar_t(1.), x, 1);
    }
    return rho;
  }

} // end namespace strumpack

#endif // GMRES_HPP
