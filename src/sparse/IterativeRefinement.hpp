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
/*!
 * \file IterativeRefinement.hpp
 * \brief Contains the iterative refinement implementation.
 */
#ifndef ITERATIVE_REFINEMENT_HPP
#define ITERATIVE_REFINEMENT_HPP
#include <iostream>
#include <iomanip>

#include "StrumpackParameters.hpp"
#include "CompressedSparseMatrix.hpp"
#include "dense/DenseMatrix.hpp"

namespace strumpack {

  /**
   * Iterative refinement, with a sparse matrix, to solve a linear
   * system M^{-1}Ax=M^{-1}b.
   *
   * \tparam scalar_t scalar type
   * \tparam integer_t integer type used in A
   * \tparam real_t real type, can be derived from the scalar_t type
   *
   * \param A dense matrix A
   * \param direct_solve routine to apply M^{-1} to a matrix
   * \param x on output this contains the solution, on input this can
   * be the initial guess. This always has to be allocated to the
   * correct size (A.rows() x b.cols())
   * \param b the right hand side, should have A.rows() rows
   * \param rtol relative stopping tolerance
   * \param atol absolute stopping tolerance
   * \param totit on output this will contain the number of iterations
   * that were performed
   * \param maxit maximum number of iterations
   * \param non_zero_guess x use x as an initial guess
   */
  template<typename scalar_t,typename integer_t,
           typename real_t = typename RealType<scalar_t>::value_type>
  void IterativeRefinement
  (const CompressedSparseMatrix<scalar_t,integer_t>& A,
   const std::function<void(DenseMatrix<scalar_t>&)>& direct_solve,
   DenseMatrix<scalar_t>& x, const DenseMatrix<scalar_t>& b,
   real_t rtol, real_t atol, int& totit, int maxit,
   bool non_zero_guess, bool verbose) {
    DenseMatrix<scalar_t> r(x.rows(), x.cols());
    if (non_zero_guess) {
      A.spmv(x, r);
      r.scale_and_add(scalar_t(-1.), b);
    } else {
      r = b;
      x.zero();
    }
    auto res_norm = r.norm();
    auto res0 = res_norm;
    auto rel_res_norm = real_t(1.);
    auto bw_error = real_t(1.);
    totit = 0;
    if (verbose)
      std::cout << "REFINEMENT it. " << totit
                << "\tres = " << std::setw(12) << res_norm
                << "\trel.res = " << std::setw(12) << rel_res_norm
                << "\tbw.error = " << std::setw(12) << bw_error
                << std::endl;
    while (res_norm > atol && rel_res_norm > rtol &&
           totit++ < maxit && bw_error > atol) {
      direct_solve(r);
      x.add(r);
      bw_error = A.max_scaled_residual(x, b);
      A.spmv(x, r);
      r.scale_and_add(scalar_t(-1.), b);
      res_norm = r.norm();
      rel_res_norm = res_norm / res0;
      if (verbose)
        std::cout << "REFINEMENT it. " << totit << "\tres = "
                  << std::setw(12) << res_norm
                  << "\trel.res = " << std::setw(12) << rel_res_norm
                  << "\tbw.error = " << std::setw(12) << bw_error
                  << std::endl;
    }
  }


  /**
   * Iterative refinement, with a dense matrix, to solve a linear
   * system M^{-1}Ax=M^{-1}b.
   *
   * \tparam scalar_t scalar type
   * \tparam real_t real type, can be derived from the scalar_t type
   *
   * \param A dense matrix A
   * \param direct_solve routine to apply M^{-1} to a matrix
   * \param x on output this contains the solution, on input this can
   * be the initial guess. This always has to be allocated to the
   * correct size (A.rows() x b.cols())
   * \param b the right hand side, should have A.rows() rows
   * \param rtol relative stopping tolerance
   * \param atol absolute stopping tolerance
   * \param totit on output this will contain the number of iterations
   * that were performed
   * \param maxit maximum number of iterations
   * \param non_zero_guess x use x as an initial guess
   */
  template<typename scalar_t,
           typename real_t = typename RealType<scalar_t>::value_type>
  void IterativeRefinement
  (const DenseMatrix<scalar_t>& A,
   const std::function<void(DenseMatrix<scalar_t>&)>& direct_solve,
   DenseMatrix<scalar_t>& x, const DenseMatrix<scalar_t>& b,
   real_t rtol, real_t atol, int& totit, int maxit,
   bool non_zero_guess, bool verbose) {
    DenseMatrix<scalar_t> r(b);
    if (non_zero_guess)
      gemm(Trans::N, Trans::N, scalar_t(-1.), A, x, scalar_t(1.), r);
    else
      x.zero();
    auto res_norm = r.norm();
    auto res0 = res_norm;
    auto rel_res_norm = real_t(1.);
    totit = 0;
    if (verbose)
      std::cout << "REFINEMENT it. " << totit
                << "\tres = " << std::setw(12) << res_norm
                << "\trel.res = " << std::setw(12) << rel_res_norm
                << std::endl;
    while (res_norm > atol && rel_res_norm > rtol &&
           totit++ < maxit) {
      direct_solve(r);
      x.add(r);
      r.copy(b);
      gemm(Trans::N, Trans::N, scalar_t(-1.), A, x, scalar_t(1.), r);
      res_norm = r.norm();
      rel_res_norm = res_norm / res0;
      if (verbose)
        std::cout << "REFINEMENT it. " << totit << "\tres = "
                  << std::setw(12) << res_norm
                  << "\trel.res = " << std::setw(12) << rel_res_norm
                  << std::endl;
    }
  }

} // end namespace strumpack

#endif // ITERATIVE_REFINEMENT_HPP
