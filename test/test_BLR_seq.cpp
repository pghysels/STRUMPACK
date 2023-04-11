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
#include "BLR/BLRMatrix.hpp"
#include "structured/ClusterTree.hpp"
#include "misc/TaskTimer.hpp"
using namespace strumpack;
using namespace strumpack::BLR;

#define ERROR_TOLERANCE 1e2
#define SOLVE_TOLERANCE 1e-12


int run(int argc, char* argv[]) {
  int m = 100; //, n = 1;

  BLROptions<double> blr_opts;
  blr_opts.set_verbose(false);

  auto usage = [&]() {
    cout << "# Usage:\n"
         << "#     OMP_NUM_THREADS=4 ./test1 problem options [BLR Options]\n";
    /*<< "# where:\n"
      << "#  - problem: a char that can be\n"
      << "#      'T': solve a Toeplitz problem\n"
      << "#            options: m (matrix dimension)\n"
      << "#      'U': solve an upper triangular Toeplitz problem\n"
      << "#            options: m (matrix dimension)\n"
      << "#      'f': read matrix from file (binary)\n"
      << "#            options: filename\n";*/
    blr_opts.describe_options();
    exit(1);
  };

  DenseMatrix<double> A;

  /*char test_problem = 'T';
    if (argc > 1) test_problem = argv[1][0];
    else usage();
    switch (test_problem) {
    case 'T': { // Toeplitz
    if (argc > 2) m = stoi(argv[2]);
    if (argc <= 2 || m < 0) {
    cout << "# matrix dimension should be positive integer" << endl;
    usage();
    }*/
  if (argc > 1) m = stoi(argv[1]);
  if (argc <= 1 || m < 0) {
    cout << "# matrix dimension should be positive integer" << endl;
    usage();
  }
  A = DenseMatrix<double>(m, m);
  for (int j=0; j<m; j++)
    for (int i=0; i<m; i++)
      A(i,j) = (i==j) ? 1. : 1./(1+abs(i-j));
  /*} break;
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
    }*/
  blr_opts.set_from_command_line(argc, argv);

  if (blr_opts.verbose()) A.print("A");
  cout << "# tol = " << blr_opts.rel_tol() << endl;

  // define a partition tree for the BLR matrix
  structured::ClusterTree tree(m);
  tree.refine(blr_opts.leaf_size());
  //tree.print();
  auto tiles=tree.template leaf_sizes<std::size_t>();
  //ADMISSIBILITY -- weak
  std::size_t nt = tiles.size();
  DenseMatrix<bool> adm(nt, nt);
  adm.fill(true);
  for (std::size_t t=0; t<nt; t++)
    adm(t, t) = false;
  long long int f0 = 0, ftot = 0;
#if defined(STRUMPACK_COUNT_FLOPS)
  //std::cout << "flop_counter_start" << std::endl;
  f0 = params::flops;
  //std::cout << "# start flops       = " << double(f0) << double(params::flops) << std::endl;
#endif
  TaskTimer t3("Compression");
  t3.start();
  BLRMatrix<double> B(A, tiles, adm, blr_opts);
  t3.stop();
#if defined(STRUMPACK_COUNT_FLOPS)
  //std::cout << "flop_counter_stop" << std::endl;
  ftot = params::flops - f0;
  //std::cout << "# stop flops       = " << double(params::flops) << std::endl;
#endif
  cout << "# created BLR matrix of dimension "
       << B.rows() << " x " << B.cols() << endl;
  //B.print("B");
  //cout << "# compression succeeded!" << endl;
  cout << "# rank(B) = " << B.rank() << endl;
  cout << "# memory(B) = " << B.memory()/1e6 << " MB, "
       << 100. * B.memory() / A.memory() << "% of dense" << endl;
#if defined(STRUMPACK_COUNT_FLOPS)
  std::cout << "# flops       = " << double(ftot) << std::endl;
  std::cout << "# time = " << t3.elapsed() << std::endl;
  //std::cout << "# flop rate = " << ftot / t3.elapsed() / 1e9
  //              << " GFlop/s" << std::endl;
#endif

  //solve AX=Y, A Toeplitz
  A = DenseMatrix<double>(m, m);
  for (int j=0; j<m; j++)
    for (int i=0; i<m; i++)
      A(i,j) = (i==j) ? 1. : 1./(1+abs(i-j));
  DenseMatrix<double> Y(m, 10), X(m, 10);//, T1(m, 10);
  X.random();
  // compute Y <- AX
  gemm(Trans::N, Trans::N, 1., A, X, 0., Y);
  B.solve(Y);
  auto Xnorm = X.normF();
  //Y.scaled_add(-1., X);
  X.scaled_add(-1., Y);
  cout << "# relative error = ||X-B\\(A*X)||_F/||X||_F = "
       << X.normF() / Xnorm << endl;
  if (X.normF() / Xnorm > ERROR_TOLERANCE
      * max(blr_opts.rel_tol(),blr_opts.abs_tol())) {
    cout << "ERROR: compression error too big!!" << endl;
    return 1;
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
