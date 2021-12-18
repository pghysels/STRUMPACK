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
#include <cmath>
#include <iostream>
using namespace std;

#include "dense/DistributedMatrix.hpp"
#include "HSS/HSSMatrixMPI.hpp"
using namespace strumpack;
using namespace strumpack::HSS;

#define ERROR_TOLERANCE 1e1
#define SOLVE_TOLERANCE 1e-12

int run(int argc, char* argv[]) {
  int m = 150;
  int n = 1;

  HSSOptions<double> hss_opts;
  hss_opts.set_verbose(false);

  auto usage = [&]() {
    if (!mpi_rank()) {
      cout << "# Usage:\n"
      << "#     OMP_NUM_THREADS=4 ./test_HSS_mpi"
      << " problem options [HSS Options]\n"
      << "# where:\n"
      << "#  - problem: a char that can be\n"
      << "#      'T': solve a Toeplitz problem\n"
      << "#            options: m (matrix dimension)\n"
      << "#      'f': read matrix from file (binary)\n"
      << "#            options: filename\n";
      hss_opts.describe_options();
    }
    exit(1);
  };

  BLACSGrid grid(MPI_COMM_WORLD);
  DistributedMatrix<double> A;

  char test_problem = 'T';
  if (argc > 1) test_problem = argv[1][0];
  else usage();
  switch (test_problem) {
  case 'T': { // Toeplitz
    if (argc > 2) m = stoi(argv[2]);
    if (argc <= 2 || m < 0) {
      cout << "# matrix dimension should be positive integer" << endl;
      usage();
    }
    A = DistributedMatrix<double>(&grid, m, m);
    // TODO only loop over local rows and columns, get the global coordinate..
    for (int j=0; j<m; j++)
      for (int i=0; i<m; i++)
        A.global(i, j, (i==j) ? 1. : 1./(1+abs(i-j)));
  } break;
  case 'U': { // upper triangular Toeplitz
    if (argc > 2) m = stoi(argv[2]);
    if (argc <= 2 || m < 0) {
      cout << "# matrix dimension should be positive integer" << endl;
      usage();
    }
    A = DistributedMatrix<double>(&grid, m, m);
    // TODO only loop over local rows and columns, get the global coordinate..
    for (int j=0; j<m; j++)
      for (int i=0; i<m; i++)
        if (i > j) A.global(i, j, 0.);
        else A.global(i, j, (i==j) ? 1. : 1./(1+abs(i-j)));
  } break;
  case 'L': {
    if (argc > 2) m = stoi(argv[2]);
    if (argc <= 2 || m < 0) {
      cout << "# matrix dimension should be positive integer" << endl;
      usage();
    }
    A = DistributedMatrix<double>(&grid, m, m);
    A.eye();
    DistributedMatrix<double> U(&grid, m, max(1, int(0.3*m)));
    DistributedMatrix<double> V(&grid, m, max(1, int(0.3*m)));
    U.random();
    V.random();
    gemm(Trans::N, Trans::C, 1./m, U, V, 1., A);
  } break;
  case 'f': { // matrix from a file
    DenseMatrix<double> Aseq;
    if (!mpi_rank()) {
      string filename;
      if (argc > 2) filename = argv[2];
      else {
        cout << "# specify a filename" << endl;
        usage();
      }
      cout << "Opening file " << filename << endl;
      ifstream file(filename, ifstream::binary);
      file.read(reinterpret_cast<char*>(&m), sizeof(int));
      Aseq = DenseMatrix<double>(m, m);
      file.read(reinterpret_cast<char*>(Aseq.data()), sizeof(double)*m*m);
    }
    MPI_Bcast(&m, 1, mpi_type<int>(), 0, MPI_COMM_WORLD);
    A = DistributedMatrix<double>(&grid, m, m);
    A.scatter(Aseq);
  } break;
  default:
    usage();
    exit(1);
  }
  hss_opts.set_from_command_line(argc, argv);

  if (hss_opts.verbose()) A.print("A");
  if (!mpi_rank()) cout << "# tol = " << hss_opts.rel_tol() << endl;

  HSSMatrixMPI<double> H(A, hss_opts);
  if (H.is_compressed()) {
    if (!mpi_rank()) {
      cout << "# created H matrix of dimension "
           << H.rows() << " x " << H.cols()
           << " with " << H.levels() << " levels" << endl;
      cout << "# compression succeeded!" << endl;
    }
  } else {
    if (!mpi_rank()) cout << "# compression failed!!!!!!!!" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  auto Hrank = H.max_rank();
  auto Hmem = H.total_memory();
  auto Amem = A.total_memory();
  if (!mpi_rank()) {
    cout << "# rank(H) = " << Hrank << endl;
    cout << "# memory(H) = " << Hmem/1e6 << " MB, "
         << 100. * Hmem / Amem << "% of dense" << endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  auto Hdense = H.dense();
  MPI_Barrier(MPI_COMM_WORLD);
  if (hss_opts.verbose()) Hdense.print("H");

  MPI_Barrier(MPI_COMM_WORLD);

  Hdense.scaled_add(-1., A);
  auto HnormF = Hdense.normF();
  auto AnormF = A.normF();
  if (!mpi_rank())
    cout << "# relative error = ||A-H*I||_F/||A||_F = "
         << HnormF / AnormF << endl;
  if (A.active() && HnormF / AnormF >
      ERROR_TOLERANCE * max(hss_opts.rel_tol(),hss_opts.abs_tol())) {
    if (!mpi_rank()) cout << "ERROR: compression error too big!!" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  {
    if (!mpi_rank()) cout << "# matrix-free compression!!" << endl;
    DistElemMultDuplicated<double> mat(A);
    hss_opts.set_synchronized_compression(false);
    HSSMatrixMPI<double> HMF(A.rows(), A.cols(), &grid, mat, mat, hss_opts);
    auto HMFdense = HMF.dense();
    HMFdense.scaled_add(-1., A);
    auto HMFnormF = HMFdense.normF();
    if (!mpi_rank())
      cout << "# relative error = ||A-H*I||_F/||A||_F = "
           << HMFnormF / AnormF << endl;
  }

  if (!H.leaf()) {
    double beta = 0.;
    HSSMatrixBase<double>* H0 = H.child(0);
    if (auto H0mpi = dynamic_cast<HSSMatrixMPI<double>*>(H0)) {
      DistributedMatrix<double>
        B0(H0mpi->grid(), H0mpi->cols(), H0mpi->cols()),
        C0check(H0mpi->grid(), H0mpi->rows(), B0.cols());
      B0.random();
      DistributedMatrix<double> A0
        (H0mpi->grid(), H0mpi->rows(), H0mpi->cols());
      copy(H0mpi->rows(), H0mpi->cols(), A, 0, 0, A0, 0, 0, grid.ctxt_all());
      if (H0mpi->active()) {
        auto C0 = H0mpi->apply(B0);
        gemm(Trans::N, Trans::N, 1., A0, B0, beta, C0check);
        C0.scaled_add(-1., C0check);
        auto C0norm = C0.normF();
        auto C0checknorm = C0check.normF();
        if (!mpi_rank())
          cout << "# relative error = ||H0*B0-A0*B0||_F/||A0*B0||_F = "
               << C0norm / C0checknorm << endl;
        apply_HSS(Trans::C, *H0mpi, B0, beta, C0);
        gemm(Trans::C, Trans::N, 1., A0, B0, beta, C0check);
        C0.scaled_add(-1., C0check);
        C0norm = C0.normF();
        C0checknorm = C0check.normF();
        if (!mpi_rank())
          cout << "# relative error = ||H0'*B0-A0'*B0||_F/||A0'*B0||_F = "
               << C0norm / C0checknorm << endl;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  default_random_engine gen;
  uniform_int_distribution<size_t> random_idx(0,m-1);
  if (!mpi_rank()) cout << "# extracting individual elements, avg error = ";
  double ex_err = 0;
  int iex = 5;
  for (int i=0; i<iex; i++) {
    auto r = random_idx(gen);
    auto c = random_idx(gen);
    if (r > c) continue;
    ex_err += abs(H.get(r, c) - A.all_global(r, c));
  }
  if (!mpi_rank()) cout << ex_err/iex << endl;
  if (A.active() && ex_err / iex >
      ERROR_TOLERANCE * max(hss_opts.rel_tol(),hss_opts.abs_tol())) {
    if (!mpi_rank()) cout << "ERROR: extraction error too big!!" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  vector<size_t> I, J;
  auto nI = 8; //random_idx(gen);
  auto nJ = 8; //random_idx(gen);
  for (int i=0; i<nI; i++) I.push_back(random_idx(gen));
  for (int j=0; j<nJ; j++) J.push_back(random_idx(gen));
  if (!mpi_rank() && hss_opts.verbose()) {
    cout << "# extracting I=[";
    for (auto i : I) { cout << i << " "; } cout << "];\n#            J=[";
    for (auto j : J) { cout << j << " "; } cout << "];" << endl;
  }
  auto sub = H.extract(I, J, &grid);
  auto sub_dense = A.extract(I, J);
  // sub.print("sub");
  // sub_dense.print("sub_dense");
  sub.scaled_add(-1., sub_dense);
  // sub.print("sub_error");
  auto relsubnorm = sub.normF() / sub_dense.normF();
  if (!mpi_rank())
    cout << "# sub-matrix extraction error = " << relsubnorm << endl;
  if (sub.active() && relsubnorm >
      ERROR_TOLERANCE * max(hss_opts.rel_tol(),hss_opts.abs_tol())) {
    if (!mpi_rank()) cout << "ERROR: extraction error too big!!" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (!mpi_rank()) cout << "# computing ULV factorization of HSS matrix .. ";
  H.factor();
  if (!mpi_rank()) cout << "Done!" << endl;

  if (!mpi_rank()) cout << "# solving linear system .." << endl;
  DistributedMatrix<double> B(&grid, m, n);
  B.random();
  DistributedMatrix<double> C(B);
  H.solve(C);

  DistributedMatrix<double> Bcheck(&grid, m, n);
  apply_HSS(Trans::N, H, C, 0., Bcheck);
  Bcheck.scaled_add(-1., B);
  auto Bchecknorm = Bcheck.normF();
  auto Bnorm = B.normF();
  if (!mpi_rank())
    cout << "# relative error = ||B-H*(H\\B)||_F/||B||_F = "
         << Bchecknorm / Bnorm << endl;
  if (B.active() && Bchecknorm / Bnorm > SOLVE_TOLERANCE) {
    if (!mpi_rank())
      cout << "ERROR: ULV solve relative error too big!!" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (!mpi_rank()) cout << "# test succeeded, exiting" << endl;
  return 0;
}


int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  if (!mpi_rank()) {
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

  int ierr;
#pragma omp parallel
#pragma omp single nowait
  ierr = run(argc, argv);

  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return ierr;
}
