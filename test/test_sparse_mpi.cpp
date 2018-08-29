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
#include <sstream>
using namespace std;

#include <getopt.h>

#define STRUMPACK_PBLAS_BLOCKSIZE 3
#define ERROR_TOLERANCE 1e2
#define SOLVE_TOLERANCE 1e-12

#include "StrumpackSparseSolverMPIDist.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/CSRMatrixMPI.hpp"
#include "dense/DistributedVector.hpp"

using namespace strumpack;

void abort_MPI(MPI_Comm *c, int *error, ...) {
  cout << "rank = " << mpi_rank() << " ABORTING!!!!!" << endl;
  abort();
}

template<typename scalar,typename integer> int
test(int argc, char* argv[], const CSRMatrixMPI<scalar,integer>* Adist) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  StrumpackSparseSolverMPIDist<scalar,integer> spss(MPI_COMM_WORLD);
  spss.options().set_matching(MatchingJob::NONE);
  spss.options().set_from_command_line(argc, argv);

  TaskTimer::t_begin = GET_TIME_NOW();

  auto N = Adist->size();
  auto n_local = Adist->local_rows();
  vector<scalar> b(n_local), x(n_local),
    x_exact(n_local, scalar(1.)/sqrt(N));
  Adist->spmv(x_exact.data(), b.data());

  spss.set_distributed_csr_matrix
    (Adist->local_rows(), Adist->ptr(), Adist->ind(),
     Adist->val(), Adist->dist().data(),
     Adist->symm_sparse());
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

  auto scaled_res = Adist->max_scaled_residual(x.data(), b.data());
  if (!rank)
    cout << "# COMPONENTWISE SCALED RESIDUAL = "
         << scaled_res << endl;

  blas::axpy(n_local, scalar(-1.), x_exact.data(), 1, x.data(), 1);

  auto nrm_error = norm2(x, MPIComm());
  auto nrm_x_exact = norm2(x_exact, MPIComm());
  if (!rank)
    cout << "# RELATIVE ERROR = "
         << (nrm_error/nrm_x_exact) << endl;

  if (scaled_res > ERROR_TOLERANCE*spss.options().rel_tol())
    MPI_Abort(MPI_COMM_WORLD, 1);
  return 0;
}

int main(int argc, char* argv[]) {
  int thread_level;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);
  //MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
  int rank = mpi_rank();
  if (!rank) {
    cout << "# Running with:\n# ";
#if defined(_OPENMP)
    cout << "OMP_NUM_THREADS=" << omp_get_max_threads()
         << " mpirun -n " << mpi_nprocs() << " ";
#else
    cout << "mpirun -n " << mpi_nprocs() << " ";
#endif
    for (int i=0; i<argc; i++) cout << argv[i] << " ";
    cout << endl;
  }
  // if (thread_level != MPI_THREAD_FUNNELED && rank == 0)
  //   cout << "MPI implementation does not support MPI_THREAD_FUNNELED"
  //             << endl;
  if (thread_level != MPI_THREAD_MULTIPLE && rank == 0)
    cout << "MPI implementation does not support MPI_THREAD_MULTIPLE, "
      "which might be needed for pt-scotch!" << endl;

  if (argc < 3) {
    if (!rank)
      cout
        << "Solve a linear system with a matrix given"
        << " in matrix market format\n"
        << "using the MPI fully distributed C++ STRUMPACK interface.\n\n"
        << "Usage: \n\tmpirun -n 4 ./testMMdoubleMPIDist m pde900.mtx\n"
        << "or\n\tmpirun -n 4 ./testMMdoubleMPIDist b pde900.bin\n"
        << "Specify the matrix input file with 'm filename'"
        << " if the matrix is in matrix-market format,\n"
        << "or with 'b filename' if the matrix is in binary." << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Errhandler eh;
  MPI_Comm_create_errhandler(abort_MPI, &eh);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, eh);

  bool binary_input = false;
  string format(argv[1]);
  if (format.compare("b") == 0) binary_input = true;
  else if (format.compare("m") == 0) binary_input = false;
  else cerr << "Error format is eiter b (binary) or m (matrix market)."
            << endl;

  string f(argv[2]);

  CSRMatrix<double,int>* A = NULL;
  CSRMatrix<complex<double>,int>* A_c = NULL;
  bool is_complex = false;

  if (rank == 0) {
    A = new CSRMatrix<double,int>();
    int ierr;
    if (binary_input) ierr = A->read_binary(f);
    else ierr = A->read_matrix_market(f);
    if (!ierr) {
      is_complex = false;
      vector<int> perm;
      vector<double> Dr, Dc;
      ierr = A->permute_and_scale
        (MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING, perm, Dr, Dc);
      if (ierr) cerr << "MC64 reordering failed!" << endl;
      A->symmetrize_sparsity();
    } else {
      delete A;
      is_complex = true;
      A_c = new CSRMatrix<complex<double>,int>();
      if (binary_input) ierr = A_c->read_binary(f);
      else ierr = A_c->read_matrix_market(f);
      if (ierr) {
        cerr << "Could not read matrix from file." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      vector<int> perm;
      vector<complex<double>> Dr, Dc;
      ierr = A_c->permute_and_scale
        (MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING, perm, Dr, Dc);
      if (ierr) cerr << "MC64 reordering failed!" << endl;
      A_c->symmetrize_sparsity();
    }
  }
  MPI_Bcast(&is_complex, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);

  int ierr = 0;
  if (!is_complex) {
    auto Adist = new CSRMatrixMPI<double,int>
      (A, MPI_COMM_WORLD, true);
    delete A;
    ierr = test(argc, argv, Adist);
    delete Adist;
  } else {
    auto Adist_c = new CSRMatrixMPI<complex<double>,int>
      (A_c, MPI_COMM_WORLD, true);
    delete A_c;
    ierr = test(argc, argv, Adist_c);
    delete Adist_c;
  }

  MPI_Errhandler_free(&eh);
  TimerList::Finalize();
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return ierr;
}
