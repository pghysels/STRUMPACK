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
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li,.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
#include <iostream>
#include "StrumpackSparseSolverMPI.hpp"
#include "sparse/CSRMatrix.hpp"

using namespace strumpack;

template<typename scalar,typename integer> void
test(int argc, char* argv[], CSRMatrix<scalar,integer>& A) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  StrumpackSparseSolverMPI<scalar,integer> spss(MPI_COMM_WORLD);
  spss.options().set_from_command_line(argc, argv);

  TaskTimer::t_begin = GET_TIME_NOW();

  auto N = A.size();
  std::vector<scalar> b(N, scalar(1.)), x(N, scalar(0.));

  spss.set_matrix(A);
  if (spss.reorder() != ReturnCode::SUCCESS) {
    if (!rank)
      std::cout << "problem with reordering of the matrix." << std::endl;
    return;
  }
  if (spss.factor() != ReturnCode::SUCCESS) {
    if (!rank)
      std::cout << "problem during factorization of the matrix." << std::endl;
    return;
  }
  spss.solve(b.data(), x.data());

  if (!rank)
    std::cout << "# COMPONENTWISE SCALED RESIDUAL = "
              << A.max_scaled_residual(x.data(), b.data()) << std::endl;
}

int main(int argc, char* argv[]) {
  int thread_level;
  //MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (thread_level != MPI_THREAD_FUNNELED && rank == 0)
    std::cout << "MPI implementation does not support MPI_THREAD_FUNNELED" << std::endl;
  // if (thread_level != MPI_THREAD_MULTIPLE && rank == 0)
  //   std::cout << "MPI implementation does not support MPI_THREAD_MULTIPLE, "
  //     "which might be needed for pt-scotch!" << std::endl;

  if (argc < 2) {
    std::cout << "Solve a linear system with a matrix given in matrix market format" << std::endl
              << "using the MPI C++ STRUMPACK interface with replicated input/output."
              << std::endl << std::endl
              << "Usage: \n\tmpirun -n 4 ./testMMdoubleMPI pde900.mtx" << std::endl;
    return 1;
  }
  std::string f(argv[1]);

  CSRMatrix<double,int>* A = NULL;
  CSRMatrix<std::complex<double>,int>* A_c = NULL;
  int n, nnz;
  bool is_complex = false;
  if (!rank) {
    A = new CSRMatrix<double,int>();
    if (A->read_matrix_market(f)) {
      is_complex = true;
      delete A;
      A_c = new CSRMatrix<std::complex<double>,int>();
      A_c->read_matrix_market(f);
      n = A_c->size();
      nnz = A_c->nnz();
    } else {
      is_complex = false;
      n = A->size();
      nnz = A->nnz();
    }
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&is_complex, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);

  if (is_complex) {
    if (rank) A_c = new CSRMatrix<std::complex<double>,int>(n, nnz);
    MPI_Bcast(A_c->ptr(), n+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(A_c->ind(), nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(A_c->val(), nnz, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    test(argc, argv, *A_c);
    delete A_c;
  } else {
    if (rank) A = new CSRMatrix<double,int>(n, nnz);
    MPI_Bcast(A->ptr(), n+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(A->ind(), nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(A->val(), nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    test(argc, argv, *A);
    delete A;
  }
  TimerList::Finalize();
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}
