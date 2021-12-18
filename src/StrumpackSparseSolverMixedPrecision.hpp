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
/**
 * \file SparseSolverMixedPrecision.hpp
 * \brief Contains the mixed precision definition of the
 * sequential/multithreaded sparse solver class.
 */
#ifndef SPARSE_SOLVER_MIXED_PRECISION_HPP
#define SPARSE_SOLVER_MIXED_PRECISION_HPP

#include <new>
#include <memory>
#include <vector>
#include <string>

#include "iterative/IterativeSolvers.hpp"
#include "StrumpackOptions.hpp"
#include "StrumpackSparseSolver.hpp"


namespace strumpack {

  /**
   * \class SparseSolverMixedPrecision
   *
   * \brief SparseSolverMixedPrecision Allows to use lower precision
   * (float) for the factorization/preconditioner, and higher (double)
   * for the outer iterative solver. See also LAPACK's dsgesv.
   *
   * Mixed precision solver class. The input and output (sparse
   * matrix, right hand side, and the solution of the linear system)
   * are expected in refine_t precision, which should be double or
   * std::complex<double>, while factor_t is the type for the internal
   * factorization, which should be float or std::complex<float>.
   *
   * There are options associated with the outer solver and with the
   * inner solver (the lower-precision preconditioner). Make sure to
   * set the correct ones. By default the inner solver options, such
   * as solver tolerances, will be initialized for single precision,
   * while the outer solver options are initialized for double
   * precision. To change the final accuracy, access
   * this->options().set_rel_tol(..) and
   * this->options().set_abs_tol(..).  To set preconditioner specific
   * options, use this->solver().options()....  By default, this will
   * set the inner solver to be KrylovSolver::DIRECT (a single
   * preconditioner application), and the outer solver to be
   * KrylovSolver::AUTO (which will default to iterative refinement).
   *
   * \tparam factor_t can be: float or std::complex<float>
   * \tparam refine_t can be: double or std::complex<double>
   *
   * \tparam integer_t defaults to a regular int. If regular int
   * causes 32 bit integer overflows, you should switch to
   * integer_t=int64_t instead. This should be a __signed__ integer
   * type.
   *
   * \see SparseSolver
   */
  template<typename factor_t,typename refine_t,typename integer_t>
  class SparseSolverMixedPrecision {

  public:
    /**
     * Constructor for the mixed precision solver class.
     */
    SparseSolverMixedPrecision(bool verbose=true, bool root=true);

    /**
     * Constructor for the mixed precision solver class.
     *
     * The command line arguments will be passed to both the options
     * for the inner and outer solvers. Note that these options are
     * not parsed until the user explicitly calls
     * this->options().set_from_command_line(), and/or
     * this->solver().options().set_from_command_line() (same as
     * this->solver().set_from_options());
     */
    SparseSolverMixedPrecision(int argc, char* argv[],
                               bool verbose=true, bool root=true);

    /**
     * Destructor.
     */
    ~SparseSolverMixedPrecision();

    void set_matrix(const CSRMatrix<refine_t,integer_t>& A);
    void set_matrix(const CSRMatrix<factor_t,integer_t>& A);

    ReturnCode factor();
    ReturnCode reorder(int nx=1, int ny=1, int nz=1);

    ReturnCode solve(const DenseMatrix<refine_t>& b,
                     DenseMatrix<refine_t>& x,
                     bool use_initial_guess=false);
    ReturnCode solve(const refine_t* b, refine_t* x,
                     bool use_initial_guess=false);

    ReturnCode solve(const DenseMatrix<factor_t>& b,
                     DenseMatrix<factor_t>& x,
                     bool use_initial_guess=false);
    ReturnCode solve(const factor_t* b, factor_t* x,
                     bool use_initial_guess=false);

    SPOptions<refine_t>& options() { return opts_; }
    const SPOptions<refine_t>& options() const { return opts_; }

    SparseSolver<factor_t,integer_t>& solver() { return solver_; }
    const SparseSolver<factor_t,integer_t>& solver() const { return solver_; }

    /**
     * Return the number of iterations performed by the outer (Krylov)
     * iterative solver. Call this after calling the solve routine.
     */
    int Krylov_iterations() const { return Krylov_its_; }

  private:
    CSRMatrix<refine_t,integer_t> mat_;
    SparseSolver<factor_t,integer_t> solver_;
    SPOptions<refine_t> opts_;
    int Krylov_its_ = 0;
  };

  template<typename factor_t,typename refine_t,typename integer_t>
  using StrumpackSparseSolverMixedPrecision =
    SparseSolverMixedPrecision<factor_t,refine_t,integer_t>;

} //end namespace strumpack

#endif // STRUMPACK_SPARSE_SOLVER_MIXED_PRECISION_HPP
