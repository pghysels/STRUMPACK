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

#include "StrumpackSparseSolver.hpp"
#include "StrumpackSparseSolverMixedPrecision.hpp"
#include "sparse/CSRMatrix.hpp"

using namespace strumpack;


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
  std::string f;
  if (argc > 1) f = std::string(argv[1]);

  CSRMatrix<double,int> A_d;
  A_d.read_matrix_market(f);
  auto A_f = cast_matrix<double,int,float>(A_d);

  int N = A_d.size();
  int m = 1; // nr of RHSs
  DenseMatrix<double> b_d(N, m), x_exact_d(N, m);
  x_exact_d.random();
  A_d.spmv(x_exact_d, b_d);

  DenseMatrix<float> b_f(N, m), x_exact_f(N, m);
  copy(x_exact_d, x_exact_f);
  copy(b_d, b_f);

  test<double>(A_d, b_d, x_exact_d, argc, argv);
  test<float >(A_f, b_f, x_exact_f, argc, argv);

  return 0;
}
