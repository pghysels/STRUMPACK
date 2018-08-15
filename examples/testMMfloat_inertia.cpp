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
#include <iostream>
#include "StrumpackSparseSolver.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/CSCMatrix.hpp"

using namespace strumpack;

template<typename scalar,typename integer> void
test(int argc, char* argv[], CSRMatrix<scalar,integer>& A) {
  StrumpackSparseSolver<scalar,integer> spss;
  // spss.options().set_reordering_method(ReorderingStrategy::NATURAL);
  spss.options().set_matching(MatchingJob::NONE);
  // spss.options().enable_HSS();
  spss.options().set_from_command_line(argc, argv);

  TaskTimer::t_begin = GET_TIME_NOW();

  int N = A.size();
  std::vector<scalar> b(N, scalar(1.)), x(N, scalar(0.));

  spss.set_matrix(A);

  if (spss.reorder() != ReturnCode::SUCCESS) {
    std::cout << "problem with reordering of the matrix." << std::endl;
    return;
  }
  if (spss.factor() != ReturnCode::SUCCESS) {
    std::cout << "problem during factorization of the matrix." << std::endl;
    return;
  }

  auto in = spss.inertia();
  std::cout << " " << std::endl;
  std::cout << "# Inertia = {" << in.np << ","
            << in.nn << "," << in.nz << "}" << std::endl;
  std::cout << " " << std::endl;

  spss.solve(b.data(), x.data());

  std::cout << "# COMPONENTWISE SCALED RESIDUAL = "
	    << A.max_scaled_residual(x.data(), b.data()) << std::endl;

}

int main(int argc, char* argv[]) {
  std::string f(argv[1]);
  typedef int64_t integer;
  //typedef int integer;

  CSRMatrix<float,integer> A;
  if (A.read_matrix_market(f) == 0)
    test<float,integer>(argc, argv, A);
  else {
    CSRMatrix<std::complex<float>,integer> A;
    A.read_matrix_market(f);
    test<std::complex<float>,integer>(argc, argv, A);
  }
}
