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
/*
 * Regression test for the (multithreaded) BLR LU factorization with
 * partial pivoting.
 *
 * The matrix is random (not diagonally dominant), so the LU of the
 * diagonal tiles performs actual row interchanges. These have to be
 * applied consistently to the tiles left and right of the diagonal.
 * The factorization is repeated several times, since errors in the
 * OpenMP task dependencies only show up for some task schedules.
 *
 * Usage:
 *   OMP_NUM_THREADS=8 ./test_BLR_pivot m reps [BLR options]
 */
#include <iostream>
#include <random>
#include <string>
#include <algorithm>
using namespace std;

#include "dense/DenseMatrix.hpp"
#include "BLR/BLRMatrix.hpp"
#include "structured/ClusterTree.hpp"
using namespace strumpack;
using namespace strumpack::BLR;

#define SOLVE_TOLERANCE 1e-6

int run(int argc, char* argv[]) {
  int m = 1000, reps = 10;
  if (argc > 1) m = stoi(argv[1]);
  if (argc > 2) reps = stoi(argv[2]);
  if (m <= 0 || reps <= 0) {
    cout << "# Usage: OMP_NUM_THREADS=8 ./test_BLR_pivot m reps "
         << "[BLR options]" << endl;
    return 1;
  }

  BLROptions<double> opts;
  opts.set_verbose(false);
  opts.set_leaf_size(32);
  // tight compression tolerance, so the error measures the
  // correctness of the factorization, not the compression
  opts.set_rel_tol(1e-12);
  opts.set_abs_tol(1e-14);
  opts.set_from_command_line(argc, argv);

  // random matrix, partial pivoting will interchange rows
  mt19937 gen(1234);
  uniform_real_distribution<double> dist(-1., 1.);
  DenseMatrix<double> A(m, m);
  for (int j=0; j<m; j++)
    for (int i=0; i<m; i++)
      A(i, j) = dist(gen);

  structured::ClusterTree tree(m);
  tree.refine(opts.leaf_size());
  auto tiles = tree.template leaf_sizes<std::size_t>();
  std::size_t nt = tiles.size();
  // weak admissibility: all off-diagonal tiles are compressed (they
  // stay dense if compression does not pay off)
  DenseMatrix<bool> adm(nt, nt);
  adm.fill(true);
  for (std::size_t t=0; t<nt; t++) adm(t, t) = false;

  cout << "# m = " << m << ", tiles = " << nt
       << ", leaf_size = " << opts.leaf_size()
       << ", algorithm = "
       << get_name(opts.BLR_factor_algorithm()) << endl;

  DenseMatrix<double> X(m, 4), B(m, 4);
  X.random();
  auto Xnorm = X.normF();
  int failed = 0;
  double worst = 0.;
  for (int r=0; r<reps; r++) {
    BLRMatrix<double> F(m, tiles, m, tiles);
    F.compress_and_factor(A, adm, opts);
    gemm(Trans::N, Trans::N, 1., A, X, 0., B);
    F.solve(B);
    B.scaled_add(-1., X);
    auto err = B.normF() / Xnorm;
    worst = max(worst, err);
    if (!(err < SOLVE_TOLERANCE)) failed++;
  }
  cout << "# relative error ||X-F\\(A*X)||_F/||X||_F, worst of "
       << reps << " = " << worst << endl;
  if (failed) {
    cout << "ERROR: " << failed << " out of " << reps
         << " factorizations failed!!" << endl;
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
  int ierr = 0;
#pragma omp parallel
#pragma omp single nowait
  ierr = run(argc, argv);
  return ierr;
}
