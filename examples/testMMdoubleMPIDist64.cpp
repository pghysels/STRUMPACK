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
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li,.
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 *
 */
#include <iostream>
#include "StrumpackSparseSolverMPIDist.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/CSRMatrixMPI.hpp"

using namespace strumpack;

template<typename scalar,typename integer> void
test(int argc, char* argv[], CSRMatrixMPI<scalar,integer>* Adist) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  StrumpackSparseSolverMPIDist<scalar,integer> spss(MPI_COMM_WORLD);
  spss.options().set_mc64job(0);
  spss.options().set_from_command_line(argc, argv);

  TaskTimer::t_begin = GET_TIME_NOW();

  auto n_local = Adist->local_rows();
  std::vector<scalar> b(n_local, scalar(1.)), x(n_local, scalar(0.));

  spss.set_distributed_csr_matrix(Adist->local_rows(), Adist->get_ptr(), Adist->get_ind(),
				  Adist->get_val(), Adist->get_dist().data(), Adist->has_symmetric_sparsity());
  if (spss.reorder() != ReturnCode::SUCCESS) {
    if (!rank) std::cout << "problem with reordering of the matrix." << std::endl;
    return;
  }
  if (spss.factor() != ReturnCode::SUCCESS) {
    if (!rank) std::cout << "problem during factorization of the matrix." << std::endl;
    return;
  }
  spss.solve(b.data(), x.data());

  auto scaled_res = Adist->max_scaled_residual(x.data(), b.data());
  if (!rank) std::cout << "# COMPONENTWISE SCALED RESIDUAL = " << scaled_res << std::endl;
}

int main(int argc, char* argv[]) {
  int thread_level, rank;
  //MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (thread_level != MPI_THREAD_FUNNELED && rank == 0)
    std::cout << "MPI implementation does not support MPI_THREAD_FUNNELED" << std::endl;
  // if (thread_level != MPI_THREAD_MULTIPLE && rank == 0)
  //   std::cout << "MPI implementation does not support MPI_THREAD_MULTIPLE, "
  //     "which might be needed for pt-scotch!" << std::endl;

  if (argc < 2) {
    std::cout << "Solve a linear system with a matrix given in matrix market format" << std::endl
	      << "using the MPI fully distributed C++ STRUMPACK interface with 64 bit indexing."
	      << std::endl << std::endl
	      << "Usage: \n\tmpirun -n 4 ./testMMdoubleMPIDist64 pde900.mtx" << std::endl;
    return 1;
  }
  std::string f(argv[1]);

  CSRMatrix<double,int64_t>* A = NULL;
  CSRMatrix<std::complex<double>,int64_t>* A_c = NULL;
  bool is_complex = false;

  if (rank == 0) {
    A = new CSRMatrix<double,int64_t>();
    if (!A->read_matrix_market(f)) {
      is_complex = false;
      std::vector<int64_t> perm;
      std::vector<double> Dr;
      std::vector<double> Dc;
      int ierr = A->permute_and_scale(5, perm, Dr, Dc);
      if (ierr) std::cout << "MC64 reordering failed!" << std::endl;
      A->symmetrize_sparsity();
    } else {
      delete A;
      is_complex = true;
      A_c = new CSRMatrix<std::complex<double>,int64_t>();
      A_c->read_matrix_market(f);
      std::vector<int64_t> perm;
      std::vector<std::complex<double>> Dr;
      std::vector<std::complex<double>> Dc;
      int ierr = A_c->permute_and_scale(5, perm, Dr, Dc);
      if (ierr) std::cout << "MC64 reordering failed!" << std::endl;
      A_c->symmetrize_sparsity();
    }
  }
  MPI_Bcast(&is_complex, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);

  if (!is_complex) {
    auto Adist = new CSRMatrixMPI<double,int64_t>(A, MPI_COMM_WORLD, true);
    delete A;
    test(argc, argv, Adist);
    delete Adist;
  } else {
    auto Adist_c = new CSRMatrixMPI<std::complex<double>,int64_t>(A_c, MPI_COMM_WORLD, true);
    delete A_c;
    test(argc, argv, Adist_c);
    delete Adist_c;
  }
  TimerList::Finalize();
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}
