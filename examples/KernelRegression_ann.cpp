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
 * reproduce, prepare derivative works, and perform publicly and display
 * publicly. Beginning five (5) years after the date permission to assert
 * copyright is obtained from the U.S. Department of Energy, and subject to any
 * subsequent five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive, irrevocable,
 * worldwide license in the Software to reproduce, prepare derivative works,
 * distribute copies to the public, perform publicly and display publicly, and
 * to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 * Division).
 *
 */
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>

#include "HSS/HSSMatrix.hpp"
#include "misc/TaskTimer.hpp"
#include "FileManipulation.h"
#include "preprocessing.h"
#include "find_ann.h"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

// random_device rd;
mt19937 generator(1);

class Kernel {
  using DenseM_t = DenseMatrix<double>;
  using DenseMW_t = DenseMatrixWrapper<double>;

public:
  vector<double> _data;
  int _d = 0;
  int _n = 0;
  double _h = 0.;
  double _l = 0.;

  Kernel() = default;

  Kernel(vector<double> data, int d, double h, double l)
  : _data(std::move(data)), _d(d), _n(_data.size() / _d), _h(h), _l(l) {
    assert(_n * _d == _data.size());
  }

  void operator()(const vector<size_t> &I, const vector<size_t> &J,
  DenseM_t &B) {
    assert(I.size() == B.rows() && J.size() == B.cols());
    for (size_t j = 0; j < J.size(); j++) {
      for (size_t i = 0; i < I.size(); i++) {
        B(i, j) = Gauss_kernel(&_data[I[i] * _d], &_data[J[j] * _d], _d, _h);
        if (I[i] == J[j]) {
          B(i, j) += _l;
        }
      }
    }
  }

  void times(DenseM_t &Rr, DenseM_t &Sr) {
    assert(Rr.rows() == _n);
    Sr.zero();
    const size_t B = 64;
    DenseM_t Asub(B, B);
    #pragma omp parallel for firstprivate(Asub) schedule(dynamic)
    for (size_t r = 0; r < _n; r += B) {
      // loop over blocks of A
      for (size_t c = 0; c < _n; c += B) {
        const int Br = std::min(B, _n - r);
        const int Bc = std::min(B, _n - c);
        // construct a block of A
        for (size_t j = 0; j < Bc; j++) {
          for (size_t i = 0; i < Br; i++) {
            Asub(i, j) = Gauss_kernel
              (&_data[(r + i) * _d], &_data[(c + j) * _d], _d, _h);
          }
          if (r==c) Asub(j, j) += _l;
        }
        DenseMW_t Ablock(Br, Bc, Asub, 0, 0);
        // Rblock is a subblock of Rr of dimension Bc x Rr.cols(),
        // starting at position c,0 in Rr
        DenseMW_t Rblock(Bc, Rr.cols(), Rr, c, 0);
        DenseMW_t Sblock(Br, Sr.cols(), Sr, r, 0);
        // multiply block of A with a row-block of Rr and add result to Sr
        gemm(Trans::N, Trans::N, 1., Ablock, Rblock, 1., Sblock);
      }
    }
  }

  void operator()(DenseM_t &Rr, DenseM_t &Rc, DenseM_t &Sr, DenseM_t &Sc) {
    times(Rr, Sr);
    Sc.copy(Sr);
  }
};

int main(int argc, char *argv[]) {
  string filename("smalltest.dat");
  int d = 2;
  string reorder("natural");
  double h = 3.;
  double lambda = 1.;
  int kernel = 1; // Gaussian=1, Laplace=2
  string mode("test");

  cout << endl << "# usage: ./KernelRegression_mf file d h lambda "
          "kern(1=Gau,2=Lapl) "
          "reorder(nat, 2means, kd, pca) mode(valid, test)"
       << endl;

  if (argc > 1)
    filename = string(argv[1]);
  if (argc > 2)
    d = stoi(argv[2]);
  if (argc > 3)
    h = stof(argv[3]);
  if (argc > 4)
    lambda = stof(argv[4]);
  if (argc > 5)
    kernel = stoi(argv[5]);
  if (argc > 6)
    reorder = string(argv[6]);
  if (argc > 7)
    mode = string(argv[7]);

  cout << endl;
  cout << "# data dimension    = "<<d << endl;
  cout << "# kernel h          = "<<h << endl;
  cout << "# lambda            = "<<lambda << endl;
  cout << "# kernel type       = "<<((kernel == 1) ? "Gauss":"Laplace") << endl;
  cout << "# reordering/clust  = "<<reorder << endl;
  cout << "# validation/test   = "<<mode << endl;

  TaskTimer::t_begin = GET_TIME_NOW();
  TaskTimer timer(string("compression"), 1);

  HSSOptions<double> hss_opts;
  hss_opts.set_verbose(true);
  hss_opts.set_from_command_line(argc, argv);

  cout << endl;
  cout << "# hss_opts.d0       = " << hss_opts.d0()    << endl;
  cout << "# hss_opts.dd       = " << hss_opts.dd()    << endl;
  cout << "# hss_opts.rel_t    = " << hss_opts.rel_tol()   << endl;
  cout << "# hss_opts.abs_t    = " << hss_opts.abs_tol()   << endl;
  cout << "# hss_opts.leaf     = " << hss_opts.leaf_size() << endl;
  cout << endl;

  cout << "# Reading data..." << endl;
  timer.start();
  string data_train_dat_FILE = filename + "_train.csv";
  string data_train_lab_FILE = filename + "_train_label.csv";
  string data_test_dat_FILE  = filename + "_" + mode + ".csv";
  string data_test_lab_FILE  = filename + "_" + mode + "_label.csv";

  // Read from csv file
  vector<double> data_train       = write_from_file(data_train_dat_FILE);
  vector<double> data_train_label = write_from_file(data_train_lab_FILE);
  vector<double> data_test        = write_from_file(data_test_dat_FILE);
  vector<double> data_test_label  = write_from_file(data_test_lab_FILE);
  cout << "# Reading took " << timer.elapsed() << endl;

  int n = data_train.size() / d;
  int m = data_test.size() / d;
  cout << "# matrix size = " << n << " x " << d << endl;

  HSSPartitionTree cluster_tree;
  cluster_tree.size = n;
  int cluster_size = hss_opts.leaf_size();

  if (reorder == "2means") {
    recursive_2_means(data_train.data(), n, d, cluster_size, cluster_tree,
                      data_train_label.data(), generator);
  } else if (reorder == "kd") {
    recursive_kd(data_train.data(), n, d, cluster_size, cluster_tree,
                 data_train_label.data());
  } else if (reorder == "pca") {
    recursive_pca(data_train.data(), n, d, cluster_size, cluster_tree,
                  data_train_label.data());
  }

  cout << endl << "# Starting HSS compression..." << endl;
  HSSMatrix<double> K;
  if (reorder != "natural")
    K = HSSMatrix<double>(cluster_tree, hss_opts);
  else{
    K = HSSMatrix<double>(n, n, hss_opts);
  }

  // Find ANN: start ------------------------------------------------
  int ann_number = 5 * d;
  vector<int> neighbors(n*ann_number, 0);
  vector<double> neighbor_scores(n*ann_number, 0.0);
  int num_iters = 5;

  timer.start();
    find_approximate_neighbors(data_train, n, d, num_iters, ann_number,
    neighbors, neighbor_scores, generator);
  cout << "# ANN time = " << timer.elapsed() << " sec" <<endl;

  DenseMatrix<std::size_t> ann(ann_number, n);
  for (int i=0; i<ann_number; i++)
    for (int j=0; j<n; j++)
      ann(i, j) = std::size_t(neighbors[i+j*ann_number]);

  // Distances to closest neighbors, sorted in ascending order
  DenseMatrixWrapper<double> scores(ann_number, n, &neighbor_scores[0],
                                    ann_number);
  // Find ANN: end ------------------------------------------------

  // Compression: start ------------------------------------------------
  Kernel kernel_matrix(data_train, d, h, lambda);

  timer.start();
  K.compress_ann(ann, scores, kernel_matrix, hss_opts);
  // K.compress(kernel_matrix, kernel_matrix, hss_opts);
  cout << "### compression time = " << timer.elapsed() << " ###" <<endl;

  if (K.is_compressed()) {
    cout << "# created K matrix of dimension " << K.rows() << " x " << K.cols()
         << " with " << K.levels() << " levels" << endl;
    cout << "# compression succeeded!" << endl;
  } else {
    cout << "# compression failed!!!!!!!!" << endl;
    return 1;
  }
  cout << "# rank(K) = " << K.rank() << endl;
  cout << "# HSS memory(K) = " << K.memory() / 1e6 << " MB " << endl;

  // Build dense matrix to test error
  DenseMatrix<double> Kdense(n, n);
  if (kernel == 1) {
    for (int c=0; c<n; c++)
      for (int r=0; r<n; r++){
        Kdense(r, c) = Gauss_kernel(&data_train[r*d], &data_train[c*d], d, h);
        if (r == c) {
          Kdense(r, c) = Kdense(r, c) + lambda;
        }
      }
  }
  else {
    for (int c=0; c<n; c++)
    for (int r=0; r<n; r++){
      Kdense(r, c) = Laplace_kernel(&data_train[r*d], &data_train[c*d], d, h);
      if (r == c) {
        Kdense(r, c) = Kdense(r, c) + lambda;
      }
    }
  }

  cout << "# HSS matrix is "<< 100. * K.memory() /  Kdense.memory()
       << "% of dense" << endl;

  auto Ktest = K.dense();
  Ktest.scaled_add(-1., Kdense);
  cout << "# compression error = ||Kdense-K*I||_F/||Kdense||_F = "
       << Ktest.normF() / Kdense.normF() << endl;
  // Compression: end ------------------------------------------------

  // Factorization and Solve: start-----------------------------------
  cout << endl << "# Factorization start" << endl;
  timer.start();
  auto ULV = K.factor();
  cout << "# factorization time = " << timer.elapsed() << endl;

  DenseMatrix<double> B(n, 1, &data_train_label[0], n);
  DenseMatrix<double> weights(B);

  cout << endl << "# Solution start..." << endl;
  timer.start();
  K.solve(ULV, weights);
  cout << "# solve time = " << timer.elapsed() << endl;

  vector<double> sample_vector(n); // Generate random vector
  normal_distribution<double> normal_distr(0.0,1.0);
  for (int i = 0; i < n; i++) {
    sample_vector[i] = normal_distr(generator);
  }
  double sample_norm = norm(&sample_vector[0], n);
  for (int i = 0; i < n; i++)
    sample_vector[i] /= sample_norm;

  DenseMatrixWrapper<double> sample_v(n, 1, &sample_vector[0], n);
  DenseMatrix<double> sample_rhs(n, 1);
  gemm(Trans::N, Trans::N, 1., Kdense, sample_v, 0., sample_rhs);
  K.solve(ULV, sample_rhs);
  sample_v.scaled_add(-1., sample_rhs);
  cout << "# solution error = "
       << sample_v.normF() << endl;
  // Factorization and Solve: end-----------------------------------

  // Prediction: start-----------------------------------
  cout << endl << "# Prediction start..." << endl;
  timer.start();
  std::vector<double> prediction(m);
  if (kernel == 1) {
    for (int c = 0; c < m; c++)
      for (int r = 0; r < n; r++)
        prediction[c] +=
            Gauss_kernel(&data_train[r * d], &data_test[c * d], d, h) *
            weights(r, 0);
  } else {
    for (int c = 0; c < m; c++)
      for (int r = 0; r < n; r++)
        prediction[c] +=
            Laplace_kernel(&data_train[r * d], &data_test[c * d], d, h) *
            weights(r, 0);
  }

  for (int i = 0; i < m; ++i)
    prediction[i] = ((prediction[i] > 0) ? 1. : -1.);

  // compute accuracy score of prediction
  double incorrect_quant = 0;
  for (int i = 0; i < m; ++i) {
    double a = (prediction[i] - data_test_label[i]) / 2;
    incorrect_quant += (a > 0 ? a : -a);
  }

  cout << "# Prediction took " << timer.elapsed() << endl;
  cout << "# prediction score: " << ((m - incorrect_quant) / m) * 100 << "%"
       << endl << endl;
  // Prediction: end-----------------------------------

  return 0;
}
