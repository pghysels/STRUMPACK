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
#include <iostream>
#include <type_traits>
#include <random>
#include <cmath>

#include "StrumpackSparseSolver.hpp"
#include "StrumpackSparseSolverMixedPrecision.hpp"
#include "sparse/CSRMatrix.hpp"

using namespace strumpack;


/**
 * Test the STRUMPACK sparse solver, and the mixed precision sparse
 * solver.
 *
 * For working_t == float, the mixed precision solver will
 * compute the factorization in single precision, but do iterative
 * refinement in double precision to give a more accurate results than
 * the standard single precision solver.
 *
 * For working_t == double, the mixed precision solver will compute
 * the factorization in single precision and perform the iterative
 * refinement in double precision. If the problem is not too
 * ill-conditioned, this should be about as accurate, and about twice
 * as fast as the standard double precision solver. The speedup
 * depends on the relative cost of the sparse triangular solver phase
 * compared to the sparse LU factorization phase.
 *
 * TODO long double
 */
template<typename working_t>
void test(CSRMatrix<working_t,int>& A,
          DenseMatrix<working_t>& b, DenseMatrix<working_t>& x_exact,
          int argc, char* argv[]) {
  int m = b.cols();  // number of right-hand sides
  auto N = A.size();
  DenseMatrix<working_t> x(N, m);

  std::cout << std::endl;
  std::cout << "###############################################" << std::endl;
  std::cout << "### Working precision: " <<
    (std::is_same<float,working_t>::value ? "single" : "double")
            << " #################" << std::endl;
  std::cout << "###############################################" << std::endl;

  {
    std::cout << std::endl;
    std::cout << "### MIXED Precision Solver ####################" << std::endl;

    SparseSolverMixedPrecision<float,double,int> spss;
    /** options for the outer solver */
    spss.options().set_Krylov_solver(KrylovSolver::REFINE);
    // spss.options().set_Krylov_solver(KrylovSolver::PREC_BICGSTAB);
    // spss.options().set_Krylov_solver(KrylovSolver::PREC_GMRES);
    spss.options().set_rel_tol(1e-14);
    spss.options().set_from_command_line(argc, argv);

    /** options for the inner solver */
    spss.solver().options().set_Krylov_solver(KrylovSolver::DIRECT);
    // spss.solver().options().set_matching(MatchingJob::NONE);
    spss.solver().options().set_from_command_line(argc, argv);

    spss.set_matrix(A);
    spss.reorder();
    spss.factor();
    spss.solve(b, x);

    std::cout << "# COMPONENTWISE SCALED RESIDUAL = "
              << A.max_scaled_residual(x.data(), b.data()) << std::endl;
    strumpack::blas::axpy(N, -1., x_exact.data(), 1, x.data(), 1);
    auto nrm_error = strumpack::blas::nrm2(N, x.data(), 1);
    auto nrm_x_exact = strumpack::blas::nrm2(N, x_exact.data(), 1);
    std::cout << "# RELATIVE ERROR = " << (nrm_error/nrm_x_exact) << std::endl;
  }

  {
    std::cout << std::endl;
    std::cout << "### STANDARD solver ###########################" << std::endl;

    SparseSolver<working_t,int> spss;
    // spss.options().set_matching(MatchingJob::NONE);
    spss.options().set_from_command_line(argc, argv);

    spss.set_matrix(A);
    spss.reorder();
    spss.factor();
    spss.solve(b, x);

    std::cout << "# COMPONENTWISE SCALED RESIDUAL = "
              << A.max_scaled_residual(x.data(), b.data()) << std::endl;
    strumpack::blas::axpy(N, -1., x_exact.data(), 1, x.data(), 1);
    auto nrm_error = strumpack::blas::nrm2(N, x.data(), 1);
    auto nrm_x_exact = strumpack::blas::nrm2(N, x_exact.data(), 1);
    std::cout << "# RELATIVE ERROR = " << (nrm_error/nrm_x_exact) << std::endl;
  }

  std::cout << std::endl << std::endl;
}


int main(int argc, char* argv[]) {

  std::cout << "long double size in bytes: "
            << sizeof(long double) << " "
            << std::endl;

  std::string f;
  if (argc > 1) f = std::string(argv[1]);

  CSRMatrix<double,int> A_d;
  A_d.read_matrix_market(f);
  auto A_f = cast_matrix<double,int,float>(A_d);

  int N = A_d.size();
  int m = 1; // nr of RHSs
  DenseMatrix<double> b_d(N, m), x_true_d(N, m);


  // set the exact solution, see:
  //   http://www.netlib.org/lapack/lawnspdf/lawn165.pdf
  // page 20
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0., std::sqrt(24.));
  for (int j=0; j<m; j++) {
    // step 4, use a different tau for each RHS
    double tau = std::pow(dist(gen), 2.);
    for (int i=0; i<N; i++)
      // step 4c
      x_true_d(i, j) = std::pow(tau, -double(i)/(N-1));
  }

  // step 6, but in double, not double-double
  A_d.spmv(x_true_d, b_d);
  {
    // step 7, but in double, not double-double
    SparseSolver<double,int> spss;
    // SparseSolverMixedPrecision<double,long double,int> spss;
    spss.set_matrix(A_d);
    spss.solve(b_d, x_true_d);
  }

  // cast RHS and true solution to single precision
  DenseMatrix<float> b_f(N, m), x_true_f(N, m);
  copy(x_true_d, x_true_f);
  copy(b_d, b_f);

  test<double>(A_d, b_d, x_true_d, argc, argv);
  test<float >(A_f, b_f, x_true_f, argc, argv);

  return 0;
}
