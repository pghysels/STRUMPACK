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
#include "H/HMatrixBase.hpp"
using namespace strumpack;
using namespace strumpack::H;

#define ERROR_TOLERANCE 1e2
#define SOLVE_TOLERANCE 1e-12


int run(int argc, char* argv[]) {
  int m = 100, n = 5;

  HOptions<double> h_opts;
  h_opts.set_verbose(false);

  auto usage = [&]() {
    cout << "# Usage:\n"
    << "#     OMP_NUM_THREADS=4 ./test1 problem options [H Options]\n"
    << "# where:\n"
    << "#  - problem: a char that can be\n"
    << "#      'T': solve a Toeplitz problem\n"
    << "#            options: m (matrix dimension)\n"
    << "#      'U': solve an upper triangular Toeplitz problem\n"
    << "#            options: m (matrix dimension)\n"
    << "#      'f': read matrix from file (binary)\n"
    << "#            options: filename\n";
    h_opts.describe_options();
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
  h_opts.set_from_command_line(argc, argv);

  if (h_opts.verbose()) A.print("A");
  cout << "# tol = " << h_opts.rel_tol() << endl;

  HBlockPartition part(A.rows());
  part.refine(h_opts.leaf_size());

  {
    StrongAdmissibility adm_strong;
    auto Hstrong = HMatrixBase<double>::compress
      (A, part, part, adm_strong, h_opts);
    if (Hstrong) {
      cout << "# created STRONG admissible H matrix of dimension "
           << Hstrong->rows() << " x " << Hstrong->cols()
           << " with " << Hstrong->levels() << " levels" << endl;
      cout << "# compression succeeded!" << endl;
    } else {
      cout << "# compression failed!!!!!!!!" << endl;
      return 1;
    }
    draw(Hstrong, "Hstrong");
    cout << "# rank(H) = " << Hstrong->rank() << endl;
    cout << "# memory(H) = " << Hstrong->memory()/1e6 << " MB, "
         << 100. * Hstrong->memory() / A.memory() << "% of dense" << endl;

    auto Hstrong_dense = Hstrong->dense();
    Hstrong_dense.scaled_add(-1., A);
    cout << "# relative error = ||A-H*I||_F/||A||_F = "
         << Hstrong_dense.normF() / A.normF() << endl;
    cout << "# absolute error = ||A-H*I||_F = " << Hstrong_dense.normF() << endl;
    if (Hstrong_dense.normF() / A.normF() > ERROR_TOLERANCE
        * max(h_opts.rel_tol(), h_opts.abs_tol())) {
      cout << "ERROR: compression error too big!!" << endl;
      return 1;
    }

    auto piv_strong = LU(Hstrong);
  }

  {
    WeakAdmissibility adm_weak;
    auto Hweak = HMatrixBase<double>::compress
      (A, part, part, adm_weak, h_opts);
    if (Hweak) {
      cout << "# created WEAK admissible H matrix of dimension "
           << Hweak->rows() << " x " << Hweak->cols()
           << " with " << Hweak->levels() << " levels" << endl;
      cout << "# compression succeeded!" << endl;
    } else {
      cout << "# compression failed!!!!!!!!" << endl;
      return 1;
    }
    draw(Hweak, "Hweak");
    cout << "# rank(H) = " << Hweak->rank() << endl;
    cout << "# memory(H) = " << Hweak->memory()/1e6 << " MB, "
         << 100. * Hweak->memory() / A.memory() << "% of dense" << endl;

    auto Hweak_dense = Hweak->dense();
    Hweak_dense.scaled_add(-1., A);
    cout << "# relative error = ||A-H*I||_F/||A||_F = "
         << Hweak_dense.normF() / A.normF() << endl;
    cout << "# absolute error = ||A-H*I||_F = " << Hweak_dense.normF() << endl;
    if (Hweak_dense.normF() / A.normF() > ERROR_TOLERANCE
        * max(h_opts.rel_tol(), h_opts.abs_tol())) {
      cout << "ERROR: compression error too big!!" << endl;
      return 1;
    }

    auto piv_weak = LU(Hweak);
  }


  {
    BLRAdmissibility adm_blr;
    auto Hblr = HMatrixBase<double>::compress
      (A, part, part, adm_blr, h_opts);
    if (Hblr) {
      cout << "# created BLR admissible H matrix of dimension "
           << Hblr->rows() << " x " << Hblr->cols()
           << " with " << Hblr->levels() << " levels" << endl;
      cout << "# compression succeeded!" << endl;
    } else {
      cout << "# compression failed!!!!!!!!" << endl;
      return 1;
    }
    draw(Hblr, "Hblr");
    cout << "# rank(H) = " << Hblr->rank() << endl;
    cout << "# memory(H) = " << Hblr->memory()/1e6 << " MB, "
         << 100. * Hblr->memory() / A.memory() << "% of dense" << endl;

    auto Hblr_dense = Hblr->dense();
    Hblr_dense.scaled_add(-1., A);
    cout << "# relative error = ||A-H*I||_F/||A||_F = "
         << Hblr_dense.normF() / A.normF() << endl;
    cout << "# absolute error = ||A-H*I||_F = " << Hblr_dense.normF() << endl;
    if (Hblr_dense.normF() / A.normF() > ERROR_TOLERANCE
        * max(h_opts.rel_tol(), h_opts.abs_tol())) {
      cout << "ERROR: compression error too big!!" << endl;
      return 1;
    }
  }

  // DenseMatrix<double> X(m, n), Y(m, n);
  // X.random();
  // // Compute Y = H*X
  // gemm(Trans::N, Trans::N, 1., *Hstrong, X, 0., Y);
  // DenseMatrix<double> Ytest(m, n);
  // gemm(Trans::N, Trans::N, 1., Hstrong_dense, X, 0., Ytest);
  // Ytest.scaled_add(-1., Y);
  // cout << "# H*X relative error = ||Y-H*X||_F/||Y||_F = "
  //      << Ytest.normF() / Y.normF() << endl;

  //  auto piv_weak = LU(Hweak);
  // auto Xsolve = solve(Hstrong, piv, Y);
  // Xsolve.scaled_add(-1., X);
  // cout << "# LU relative error = ||X-(H\Y)||_F/||X||_F = "
  //      << Xsolve.normF() / X.normF() << endl;


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
