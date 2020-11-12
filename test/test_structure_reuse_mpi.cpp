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
#include <vector>
#include <cstring>
#include <random>
using namespace std;

#define ERROR_TOLERANCE 1e2
#define SOLVE_TOLERANCE 1e-12

#include "StrumpackSparseSolverMPIDist.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/CSRMatrixMPI.hpp"
#include "dense/DistributedVector.hpp"
#include "misc/RandomWrapper.hpp"

using namespace strumpack;

void abort_MPI(MPI_Comm *c, int *error, ... ) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cout << "rank = " << rank << " ABORTING!!!!!" << endl;
  abort();
}

template<typename scalar_t,typename integer_t>
int test_sparse_solver(int argc, const char* const argv[],
                       const CSRMatrix<scalar_t,integer_t>& A) {
  using real_t = typename RealType<scalar_t>::value_type;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  StrumpackSparseSolverMPIDist<scalar_t,integer_t> spss(MPI_COMM_WORLD);
  // spss.options().set_matching(MatchingJob::NONE);
  spss.options().set_from_command_line(argc, argv);

  // distribute/scatter the matrix A from the root
  CSRMatrixMPI<scalar_t,integer_t> Adist(&A, MPI_COMM_WORLD, true);

  auto N = Adist.size();
  auto n_local = Adist.local_rows();
  vector<scalar_t> b(n_local), x(n_local), x_exact(n_local);
  {
    auto rgen = random::make_default_random_generator<real_t>();
    for (auto& xi : x_exact)
      xi = rgen->get();
  }
  Adist.spmv(x_exact.data(), b.data());

  spss.set_matrix(Adist);
  if (spss.reorder() != ReturnCode::SUCCESS) {
    if (!rank)
      cout << "problem with reordering of the matrix." << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (spss.factor() != ReturnCode::SUCCESS) {
    if (!rank)
      cout << "problem during factorization of the matrix." << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  spss.solve(b.data(), x.data());

  auto scaled_res = Adist.max_scaled_residual(x.data(), b.data());
  if (!rank)
    cout << "# COMPONENTWISE SCALED RESIDUAL = "
         << scaled_res << endl;
  blas::axpy(n_local, scalar_t(-1.), x_exact.data(), 1, x.data(), 1);
  auto nrm_error = norm2(x, MPIComm());
  auto nrm_x_exact = norm2(x_exact, MPIComm());
  if (!rank)
    cout << "# RELATIVE ERROR = " << (nrm_error/nrm_x_exact) << endl;
  if (scaled_res > ERROR_TOLERANCE*spss.options().rel_tol())
    MPI_Abort(MPI_COMM_WORLD, 1);


  // modify the matrix values, but not the sparsity pattern
  {
    std::default_random_engine generator;
    std::normal_distribution<real_t> distribution(1.0, .05);
    for (int i=0; i<Adist.local_nnz(); i++)
      Adist.val(i) = Adist.val(i) * distribution(generator);
  }

  spss.delete_factors();
  // update the values
  spss.update_matrix_values(Adist);
  //spss.set_matrix(Adist);

  // recompute right hand side
  Adist.spmv(x_exact.data(), b.data());

  // this new solve will reuse the permutation
  spss.solve(b.data(), x.data());

  scaled_res = Adist.max_scaled_residual(x.data(), b.data());
  if (!rank)
    cout << "# COMPONENTWISE SCALED RESIDUAL = "
         << scaled_res << endl;
  blas::axpy(n_local, scalar_t(-1.), x_exact.data(), 1, x.data(), 1);
  nrm_error = norm2(x, MPIComm());
  nrm_x_exact = norm2(x_exact, MPIComm());
  if (!rank)
    cout << "# RELATIVE ERROR = " << (nrm_error/nrm_x_exact) << endl;
  if (scaled_res > ERROR_TOLERANCE*spss.options().rel_tol())
    MPI_Abort(MPI_COMM_WORLD, 1);

  return 0;
}


template<typename real_t,typename integer_t>
int read_matrix_and_run_tests(int argc, const char* const argv[]) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  string f(argv[1]);
  CSRMatrix<real_t,integer_t> A;
  CSRMatrix<complex<real_t>,integer_t> Acomplex;
  bool is_complex = false;
  if (rank == 0) {
    if (A.read_matrix_market(f)) {
      is_complex = true;
      if (Acomplex.read_matrix_market(f)) {
        cerr << "Could not read matrix from file." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
  }
  MPI_Bcast(&is_complex, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);
  return is_complex ?
    test_sparse_solver(argc, argv, Acomplex) :
    test_sparse_solver(argc, argv, A);
}


int main(int argc, char* argv[]) {
  int thread_level, rank, P;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  if (!rank) {
    cout << "# Running with:\n# ";
#if defined(_OPENMP)
    cout << "OMP_NUM_THREADS=" << omp_get_max_threads() << " ";
#endif
    cout << "mpirun -n " << P << " ";
    for (int i=0; i<argc; i++)
      cout << argv[i] << " ";
    cout << endl;
  }
  if (thread_level != MPI_THREAD_MULTIPLE && !rank)
    cout << "MPI implementation does not support MPI_THREAD_MULTIPLE, "
      "which might be needed for pt-scotch!" << endl;

  if (argc < 2) {
    if (!rank)
      cout
        << "Solve a linear system with a matrix given"
        << " in matrix market format\n"
        << "using the MPI fully distributed C++ STRUMPACK interface.\n\n"
        << "Usage: \n\tmpirun -n 4 ./testMMdoubleMPIDist pde900.mtx"
        << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Errhandler eh;
  MPI_Comm_create_errhandler(abort_MPI, &eh);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, eh);

  int ierr = 0;
  // ierr = read_matrix_and_run_tests<float,int>(argc, argv);
  // if (ierr) MPI_Abort(MPI_COMM_WORLD, 1);
  ierr = read_matrix_and_run_tests<double,int>(argc, argv);
  if (ierr) MPI_Abort(MPI_COMM_WORLD, 1);
  // ierr = read_matrix_and_run_tests<float,long long int>(argc, argv);
  // if (ierr) MPI_Abort(MPI_COMM_WORLD, 1);
  ierr = read_matrix_and_run_tests<double,long long int>(argc, argv);
  if (ierr) MPI_Abort(MPI_COMM_WORLD, 1);

  MPI_Errhandler_free(&eh);
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return ierr;
}
