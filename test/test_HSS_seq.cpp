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
#include <random>
using namespace std;

#include "dense/DenseMatrix.hpp"
#include "HSS/HSSMatrix.hpp"
using namespace strumpack;
using namespace strumpack::HSS;

#define ERROR_TOLERANCE 1e2
#define SOLVE_TOLERANCE 1e-12


int run(int argc, char* argv[]) {
  int m = 100, n = 1;

  HSSOptions<double> hss_opts;
  hss_opts.set_verbose(false);

  auto usage = [&]() {
    cout << "# Usage:\n"
    << "#     OMP_NUM_THREADS=4 ./test1 problem options [HSS Options]\n"
    << "# where:\n"
    << "#  - problem: a char that can be\n"
    << "#      'T': solve a Toeplitz problem\n"
    << "#            options: m (matrix dimension)\n"
    << "#      'U': solve an upper triangular Toeplitz problem\n"
    << "#            options: m (matrix dimension)\n"
    << "#      'f': read matrix from file (binary)\n"
    << "#            options: filename\n";
    hss_opts.describe_options();
    exit(1);
  };

  DenseMatrix<double> A;

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
    A = DenseMatrix<double>(m, m);
    for (int j=0; j<m; j++)
      for (int i=0; i<m; i++)
        A(i,j) = (i==j) ? 1. : 1./(1+abs(i-j));
  } break;
  case 'U': { // upper triangular Toeplitz
    if (argc > 2) m = stoi(argv[2]);
    if (argc <= 2 || m < 0) {
      cout << "# matrix dimension should be positive integer" << endl;
      usage();
    }
    A = DenseMatrix<double>(m, m);
    for (int j=0; j<m; j++)
      for (int i=0; i<m; i++)
        if (i > j) A(i,j) = 0.;
        else A(i,j) = (i==j) ? 1. : 1./(1+abs(i-j));
  } break;
  case 'L': {
    if (argc > 2) m = stoi(argv[2]);
    if (argc <= 2 || m < 0) {
      cout << "# matrix dimension should be positive integer" << endl;
      usage();
    }
    A = DenseMatrix<double>(m, m);
    A.eye();
    DenseMatrix<double> U(m, max(1, int(0.3*m)));
    DenseMatrix<double> V(m, max(1, int(0.3*m)));
    U.random();
    V.random();
    gemm(Trans::N, Trans::C, 1./m, U, V, 1., A);
  } break;
  case 'f': { // matrix from a file
    string filename;
    if (argc > 2) filename = argv[2];
    else {
      cout << "# specify a filename" << endl;
      usage();
    }
    cout << "Opening file " << filename << endl;
    ifstream file(filename, ifstream::binary);
    file.read(reinterpret_cast<char*>(&m), sizeof(int));
    A = DenseMatrix<double>(m, m);
    file.read(reinterpret_cast<char*>(A.data()), sizeof(double)*m*m);
  } break;
  default:
    usage();
    exit(1);
  }
  hss_opts.set_from_command_line(argc, argv);

  if (hss_opts.verbose()) A.print("A");
  cout << "# tol = " << hss_opts.rel_tol() << endl;

  HSSMatrix<double> H(A, hss_opts);
  if (H.is_compressed()) {
    cout << "# created H matrix of dimension "
         << H.rows() << " x " << H.cols()
         << " with " << H.levels() << " levels" << endl;
    cout << "# compression succeeded!" << endl;
  } else {
    cout << "# compression failed!!!!!!!!" << endl;
    return 1;
  }
  cout << "# rank(H) = " << H.rank() << endl;
  cout << "# memory(H) = " << H.memory()/1e6 << " MB, "
       << 100. * H.memory() / A.memory() << "% of dense" << endl;

  // H.print_info();
  auto Hdense = H.dense();
  Hdense.scaled_add(-1., A);
  cout << "# relative error = ||A-H*I||_F/||A||_F = "
       << Hdense.normF() / A.normF() << endl;
  cout << "# absolute error = ||A-H*I||_F = " << Hdense.normF() << endl;
  if (Hdense.normF() / A.normF() > ERROR_TOLERANCE
      * max(hss_opts.rel_tol(),hss_opts.abs_tol())) {
    cout << "ERROR: compression error too big!!" << endl;
    return 1;
  }

  if (!H.leaf()) {
    double beta = 0.;
    HSSMatrix<double>* H0 = H.child(0);
    DenseMatrix<double> B0(H0->cols(), H0->cols()),
      C0_check(H0->rows(), B0.cols());
    B0.random();
    DenseMatrixWrapper<double> A0(H0->rows(), H0->cols(), A, 0, 0);

    auto C0 = H0->apply(B0);
    gemm(Trans::N, Trans::N, 1., A0, B0, beta, C0_check);
    C0.scaled_add(-1., C0_check);
    cout << "# relative error = ||H0*B0-A0*B0||_F/||A0*B0||_F = "
         << C0.normF() / C0_check.normF() << endl;
    apply_HSS(Trans::C, *H0, B0, beta, C0);
    gemm(Trans::C, Trans::N, 1., A0, B0, beta, C0_check);
    C0.scaled_add(-1., C0_check);
    cout << "# relative error = ||H0'*B0-A0'*B0||_F/||A0'*B0||_F = "
         << C0.normF() / C0_check.normF() << endl;
  }
  if (!H.leaf()) {
    double beta = 0.;
    HSSMatrix<double>* H1 = H.child(1);
    DenseMatrix<double> B1(H1->cols(), H1->cols()),
      C1_check(H1->rows(), B1.cols());
    B1.random();
    DenseMatrixWrapper<double> A1(H1->rows(), H1->cols(), A, 0, 0);

    auto C1 = H1->apply(B1);
    gemm(Trans::N, Trans::N, 1., A1, B1, beta, C1_check);
    C1.scaled_add(-1., C1_check);
    cout << "# relative error = ||H1*B1-A1*B1||_F/||A1*B1||_F = "
         << C1.normF() / C1_check.normF() << endl;
    apply_HSS(Trans::C, *H1, B1, beta, C1);
    gemm(Trans::C, Trans::N, 1., A1, B1, beta, C1_check);
    C1.scaled_add(-1., C1_check);
    cout << "# relative error = ||H1'*B1-A1'*B1||_F/||A1'*B1||_F = "
         << C1.normF() / C1_check.normF() << endl;
  }

  default_random_engine gen;
  uniform_int_distribution<size_t> random_idx(0,m-1);
  cout << "# extracting individual elements, avg error = ";
  double ex_err = 0;
  int iex = 5;
  for (int i=0; i<iex; i++) {
    auto r = random_idx(gen);
    auto c = random_idx(gen);
    ex_err += abs(H.get(r,c) - A(r,c));
  }
  cout << ex_err/iex << endl;
  if (ex_err / iex > ERROR_TOLERANCE
      * max(hss_opts.rel_tol(),hss_opts.abs_tol())) {
    cout << "ERROR: extraction error too big!!" << endl;
    return 1;
  }

  vector<size_t> I, J;
  auto nI = 8; //random_idx(gen);
  auto nJ = 8; //random_idx(gen);
  for (int i=0; i<nI; i++) I.push_back(random_idx(gen));
  for (int j=0; j<nJ; j++) J.push_back(random_idx(gen));
  if (hss_opts.verbose()) {
    cout << "# extracting I=[";
    for (auto i : I) { cout << i << " "; } cout << "];\n#            J=[";
    for (auto j : J) { cout << j << " "; } cout << "];" << endl;
  }
  auto sub = H.extract(I, J);
  auto sub_dense = A.extract(I, J);
  // sub.print_to_file("sub", "sub.m");
  // sub_dense.print_to_file("sub_dense", "sub_dens.m");
  if (hss_opts.verbose()) sub.print("sub");
  sub.scaled_add(-1., sub_dense);
  // sub.print("sub_error");
  cout << "# sub-matrix extraction error = "
       << sub.normF() / sub_dense.normF() << endl;
  if (sub.normF() / sub_dense.normF() >
      1e2*max(hss_opts.rel_tol(),hss_opts.abs_tol())) {
    cout << "ERROR: extraction error too big!!" << endl;
    return 1;
  }

  cout << "# computing ULV factorization of HSS matrix .." << endl;
  H.factor();
  cout << "# solving linear system .." << endl;

  DenseMatrix<double> B(m, n);
  B.random();
  DenseMatrix<double> C(B);
  H.solve(C);
  auto Bcheck = H.apply(C);
  Bcheck.scaled_add(-1., B);
  cout << "# relative error = ||B-H*(H\\B)||_F/||B||_F = "
       << Bcheck.normF() / B.normF() << endl;
  if (Bcheck.normF() / B.normF() > SOLVE_TOLERANCE) {
    cout << "ERROR: ULV solve relative error too big!!" << endl;
    return 1;
  }

  if (!H.leaf()) {
    H.partial_factor();
    cout << "# Computing Schur update .." << endl;
    DenseMatrix<double> Theta, Phi, DUB01;
    H.Schur_update(Theta, DUB01, Phi);
    // Theta.print("Theta");
    // Phi.print("Phi");
    // TODO check the Schur update
  }

  cout << "# exiting" << endl;
  return 0;
}


int main(int argc, char* argv[]) {
  cout << "# Running with:\n# ";
#if defined(_OPENMP)
  cout << "OMP_NUM_THREADS=" << omp_get_max_threads() << " ";
#endif
  for (int i=0; i<argc; i++) cout << argv[i] << " ";
  cout << endl;

  int ierr;
#pragma omp parallel
#pragma omp single nowait
  ierr = run(argc, argv);
  return ierr;
}
