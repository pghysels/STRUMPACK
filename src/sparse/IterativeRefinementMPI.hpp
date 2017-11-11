/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The Regents of
 * the University of California, through Lawrence Berkeley National Laboratory
 * (subject to receipt of any required approvals from the U.S. Dept. of Energy).
 * All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Technology Transfer Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, prepare derivative works, and perform publicly and display publicly.
 * Beginning five (5) years after the date permission to assert copyright is
 * obtained from the U.S. Department of Energy, and subject to any subsequent five
 * (5) year renewals, the U.S. Government is granted for itself and others acting
 * on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, prepare derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 *
 */
#ifndef ITERATIVE_REFINEMENT_MPI_HPP
#define ITERATIVE_REFINEMENT_MPI_HPP
#include <iostream>
#include <iomanip>

#include "StrumpackParameters.hpp"

namespace strumpack {

  /*
   * This is iterative refinement
   *  Input vectors x and b have stride 1, length n
   */
  template<typename scalar_t,typename integer_t> void IterativeRefinementMPI
  (MPI_Comm comm, CSRMatrixMPI<scalar_t,integer_t>* A, std::function<void(scalar_t*)> direct_solve,
   integer_t n, scalar_t* x, scalar_t* b, real_t rtol, real_t atol, int& totit, int maxit,
   bool non_zero_guess, bool verbose) {
    using real_t = typename RealType<scalar_t>::value_type;
    auto r = new scalar_t[n];
    if (non_zero_guess) {
      A->omp_spmv(x, r);
      blas::axpby(n, scalar_t(1.), b, 1, scalar_t(-1.), r, 1);
    } else {
      // TODO these can be avoided?
      std::copy(b, b+n, r);
      std::fill(x, x+n, scalar_t(0.));
    }
    auto res_norm = nrm2_omp_mpi(n, r, 1, comm);
    auto res0 = res_norm;
    auto rel_res_norm = real_t(1.);
    auto bw_error = real_t(1.);
    totit = 0;
    if (verbose) std::cout << "REFINEMENT it. " << totit << "\tres = " << std::setw(12) << res_norm
			   << "\trel.res = " << std::setw(12) << rel_res_norm
			   << "\tbw.error = " << std::setw(12) << bw_error
			   << std::endl;
    while (res_norm > atol && rel_res_norm > rtol && totit++ < maxit && bw_error > atol) {
      direct_solve(r);
      blas::axpy(n, scalar_t(1.), r, 1, x, 1);
      A->omp_spmv(x, r);
      blas::axpby(n, scalar_t(1.), b, 1, scalar_t(-1.), r, 1);
      res_norm = nrm2_omp_mpi(n, r, 1, comm);
      rel_res_norm = res_norm / res0;
      bw_error = A->max_scaled_residual(x, b);
      if (verbose) std::cout << "REFINEMENT it. " << totit << "\tres = " << std::setw(12) << res_norm
			     << "\trel.res = " << std::setw(12) << rel_res_norm
			     << "\tbw.error = " << std::setw(12) << bw_error
			     << std::endl;
    }
    delete[] r;
  }

} // end namespace strumpack

#endif // ITERATIVE_REFINEMENT_MPI_HPP
