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
#include <cstring>
#include <iostream>
using namespace std;

#include "StrumpackSparseSolver.hpp"
#include "StrumpackSparseSolverMixedPrecision.hpp"
#include "misc/RandomWrapper.hpp"
#include "sparse/CSRMatrix.hpp"

using namespace strumpack;

#define ERROR_TOLERANCE 1e2
#define SOLVE_TOLERANCE 1e-12

template <typename working_t, typename integer_t>
void test(CSRMatrix<working_t, integer_t> &A, DenseMatrix<working_t> &b,
          DenseMatrix<working_t> &x_exact, int argc, char *argv[]) {
  int m = b.cols(); // number of right-hand sides
  auto N = A.size();
  DenseMatrix<working_t> x(N, m);

  std::cout << std::endl;
  std::cout << "###############################################" << std::endl;
  std::cout << "### Working precision: "
            << (std::is_same<float, working_t>::value ? "single" : "double")
            << " #################" << std::endl;
  std::cout << "###############################################" << std::endl;

  {
    std::cout << std::endl;
    std::cout << "### MIXED Precision Solver ####################" << std::endl;

    SparseSolverMixedPrecision<float, double, int> spss;
    /** options for the outer solver */
    spss.options().set_Krylov_solver(KrylovSolver::REFINE);
    //     spss.options().set_Krylov_solver(KrylovSolver::PREC_BICGSTAB);
    //     spss.options().set_Krylov_solver(KrylovSolver::PREC_GMRES);
    spss.options().set_rel_tol(1e-14);
    spss.options().enable_symmetric();
    spss.options().enable_positive_definite();
    spss.options().set_matching(strumpack::MatchingJob::NONE);
    spss.options().set_from_command_line(argc, argv);

    /** options for the inner solver */
    spss.solver().options().set_Krylov_solver(KrylovSolver::DIRECT);
    spss.solver().options().set_from_command_line(argc, argv);
    spss.solver().options().set_matching(strumpack::MatchingJob::NONE);
    spss.solver().options().enable_symmetric();
    spss.solver().options().enable_positive_definite();

    spss.set_lower_triangle_matrix(A);
    spss.reorder();
    spss.factor();
    spss.solve(b, x);

    std::cout << "# COMPONENTWISE SCALED RESIDUAL = "
              << A.max_scaled_residual(x.data(), b.data()) << std::endl;
    strumpack::blas::axpy(N, -1., x_exact.data(), 1, x.data(), 1);
    auto nrm_error = strumpack::blas::nrm2(N, x.data(), 1);
    auto nrm_x_exact = strumpack::blas::nrm2(N, x_exact.data(), 1);
    std::cout << "# RELATIVE ERROR = " << (nrm_error / nrm_x_exact)
              << std::endl;
  }

  {
    std::cout << std::endl;
    std::cout << "### STANDARD solver ###########################" << std::endl;

    SparseSolver<working_t, int> spss;

    spss.options().enable_symmetric();
    spss.options().enable_positive_definite();
    spss.options().set_matching(strumpack::MatchingJob::NONE);
    spss.options().set_from_command_line(argc, argv);

    spss.set_lower_triangle_matrix(A);
    spss.reorder();
    spss.factor();
    spss.solve(b, x);

    std::cout << "# COMPONENTWISE SCALED RESIDUAL = "
              << A.max_scaled_residual(x.data(), b.data()) << std::endl;
    strumpack::blas::axpy(N, -1., x_exact.data(), 1, x.data(), 1);
    auto nrm_error = strumpack::blas::nrm2(N, x.data(), 1);
    auto nrm_x_exact = strumpack::blas::nrm2(N, x_exact.data(), 1);
    std::cout << "# RELATIVE ERROR = " << (nrm_error / nrm_x_exact)
              << std::endl;
  }

  std::cout << std::endl << std::endl;
}

template <typename integer_t>
int test_sparse_solver(int argc, char *argv[],
                       CSRMatrix<double, integer_t> &A_d) {
  // set the exact solution, see:
  //   http://www.netlib.org/lapack/lawnspdf/lawn165.pdf
  // page 20
  int N = A_d.size();
  int m = 1; // nr of RHSs
  DenseMatrix<double> b_d(N, m), x_true_d(N, m);
  auto A_f = cast_matrix<double, integer_t, float>(A_d);

  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0., std::sqrt(24.));
  for (int j = 0; j < m; j++) {
    // step 4, use a different tau for each RHS
    double tau = std::pow(dist(gen), 2.);
    for (int i = 0; i < N; i++)
      // step 4c
      x_true_d(i, j) = std::pow(tau, -double(i) / (N - 1));
  }

  // step 6, but in double, not double-double
  A_d.spmv(x_true_d, b_d);
  {
    // step 7, but in double, not double-double
    SparseSolver<double, integer_t> spss;
    // SparseSolverMixedPrecision<double,long double,int> spss;
//    spss.set_lower_triangle_matrix(A_d);
    spss.set_matrix(A_d);
    spss.solve(b_d, x_true_d);
  }

  // cast RHS and true solution to single precision
  DenseMatrix<float> b_f(N, m), x_true_f(N, m);
  copy(x_true_d, x_true_f);
  copy(b_d, b_f);

  test<double, integer_t>(A_d, b_d, x_true_d, argc, argv);
  test<float, integer_t>(A_f, b_f, x_true_f, argc, argv);

  return 0;
}

template <typename integer_t>
int read_matrix_and_run_tests(int argc, char *argv[]) {
  string f(argv[1]);
  CSRMatrix<double, integer_t> A;
  if (A.read_matrix_market(f) == 0)
    return test_sparse_solver(argc, argv, A);
  else {
    CSRMatrix<complex<double>, integer_t> Acomplex;
    if (Acomplex.read_matrix_market(f)) {
      std::cerr << "Could not read matrix from file." << std::endl;
      return 1;
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout
        << "Solve a linear system with a matrix given in matrix market format\n"
        << "using the sequential/multithreaded C++ STRUMPACK interface.\n\n"
        << "Usage: \n\t./testMMdouble pde900.mtx" << endl;
    return 1;
  }
  cout << "# Running with:\n# ";
#if defined(_OPENMP)
  cout << "OMP_NUM_THREADS=" << omp_get_max_threads() << " ";
#endif
  for (int i = 0; i < argc; i++)
    cout << argv[i] << " ";
  cout << endl;

  int ierr = 0;
  // ierr = read_matrix_and_run_tests<float,int>(argc, argv);
  // if (ierr) return ierr;
  ierr = read_matrix_and_run_tests<int>(argc, argv);
  return ierr;
}
