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

#include "StrumpackSparseSolverMPIDist.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/CSRMatrixMPI.hpp"
// to create a random vector
#include "misc/RandomWrapper.hpp"

using namespace strumpack;

template<typename scalar,typename integer> void
test(int argc, char* argv[], CSRMatrix<scalar,integer>& A) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  StrumpackSparseSolverMPIDist<scalar,integer> spss(MPI_COMM_WORLD);
  spss.options().set_from_command_line(argc, argv);

  // distribute/scatter the matrix A from the root
  CSRMatrixMPI<scalar,integer> Adist(&A, MPI_COMM_WORLD, true);
  auto N = Adist.size();
  auto n_local = Adist.local_rows();
  std::vector<scalar> b(n_local), x(n_local), x_exact(n_local);
  {
    using real_t = typename RealType<scalar>::value_type;
    auto rgen = random::make_default_random_generator<real_t>();
    for (auto& xi : x_exact)
      xi = scalar(rgen->get());
  }
  Adist.spmv(x_exact.data(), b.data());

  spss.set_matrix(Adist);
  //alternative interface:
  // spss.set_distributed_csr_matrix
  //   (Adist.local_rows(), Adist.ptr(), Adist.ind(),
  //    Adist.val(), Adist.dist().data(), Adist.symm_sparse());

  if (spss.reorder() != ReturnCode::SUCCESS) {
    if (!rank)
      std::cout << "problem with reordering of the matrix." << std::endl;
    return;
  }
  if (spss.factor() != ReturnCode::SUCCESS) {
    if (!rank)
      std::cout << "problem during factorization of the matrix."
                << std::endl;
    return;
  }
  spss.solve(b.data(), x.data());

  auto scaled_res = Adist.max_scaled_residual(x.data(), b.data());
  if (!rank)
    std::cout << "# COMPONENTWISE SCALED RESIDUAL = "
              << scaled_res << std::endl;
  strumpack::blas::axpy(n_local, scalar(-1.), x_exact.data(), 1, x.data(), 1);
  auto nrm_error = norm2(x, MPIComm());
  auto nrm_x_exact = norm2(x_exact, MPIComm());
  if (!rank)
    std::cout << "# RELATIVE ERROR = " << (nrm_error/nrm_x_exact)
              << std::endl;
}

int main(int argc, char* argv[]) {
  int thread_level;
  //MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (thread_level != MPI_THREAD_FUNNELED && rank == 0)
    std::cout << "MPI implementation does not support MPI_THREAD_FUNNELED"
              << std::endl;
  // if (thread_level != MPI_THREAD_MULTIPLE && rank == 0)
  //   std::cout << "MPI implementation does not support MPI_THREAD_MULTIPLE,"
  //     " which might be needed for pt-scotch!" << std::endl;

  if (argc < 2) {
    if (!rank)
      std::cout
        << "Solve a linear system with a matrix in matrix market format\n"
        << "using the MPI fully distributed C++ STRUMPACK interface.\n\n"
        << "Usage: \n\tmpirun -n 4 ./testMMdoubleMPIDist pde900.mtx"
        << std::endl;
    return 1;
  }

  std::string f(argv[1]);

  CSRMatrix<double,int> A;
  CSRMatrix<std::complex<double>,int> Acomplex;
  //---- For 64bit integers use this instead: --------
  // CSRMatrix<double,int64_t> A;
  // CSRMatrix<std::complex<double>,int64_t> Acomplex;
  //--------------------------------------------------

  bool is_complex = false;
  if (rank == 0) {
    if (A.read_matrix_market(f)) {
      is_complex = true;
      if (Acomplex.read_matrix_market(f)) {
        std::cerr << "Could not read matrix from file." << std::endl;
        return 1;
      }
    }
  }
  MPI_Bcast(&is_complex, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);
  if (!is_complex)
    test(argc, argv, A);
  else
    test(argc, argv, Acomplex);

  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}
