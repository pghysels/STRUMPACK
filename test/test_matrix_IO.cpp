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
    << "#     OMP_NUM_THREADS=4 ./test_matrix_IO problem options [HSS Options]\n"
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

  {
    A.write("A_hss.bin");
    auto B = DenseMatrix<double>::read("A_hss.bin");
    B.scaled_add(-1., A);
    std::cout << "||B-A||_F = " << B.norm() << std::endl;
    if (B.norm() > SOLVE_TOLERANCE) {
      cout << "ERROR: file IO error too big!!" << endl;
      return 1;
    }
  }

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

  {
    H.write("H.bin");
    auto H2 = HSSMatrix<double>::read("H.bin");
    Hdense = H.dense();
    auto H2dense = H2.dense();
    H2dense.scaled_add(-1., Hdense);
    std::cout << "||B-A||_F = " << H2dense.norm() << std::endl;
    if (H2dense.norm() > SOLVE_TOLERANCE) {
      cout << "ERROR: file IO error too big!!" << endl;
      return 1;
    }
    std::cout << "||H-H2||_F = " << H2dense.norm() << std::endl;
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
