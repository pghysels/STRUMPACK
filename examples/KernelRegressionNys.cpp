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
 * works, and perform publicly and display publicly. Beginning five
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
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "kernel/KernelRegression.hpp"
#include "misc/TaskTimer.hpp"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;
using namespace strumpack::kernel;
using scalar_t = float;

template<typename scalar_t> vector<scalar_t>
read_from_file(string filename) {
  vector<scalar_t> data;
  ifstream f(filename);
  string l;
  while (getline(f, l)) {
    istringstream sl(l);
    string s;
    while (getline(sl, s, ','))
      data.push_back(stod(s));
  }
  data.shrink_to_fit();
  return data;
}

inline float dist2(float* x, float* y, int d) {
  float k = 0.;
  for (int i=0; i<d; i++) k += pow(x[i] - y[i], 2.);
  return k;
}

inline float Gauss_kernel(float* x, float* y, int d, float h) {
  return exp(-dist2(x, y, d)/(2*h*h));
}

void print_sample(float* x) {
  int d = 8;
  for (int i = 0; i < d; ++i)
    cout << x[i] << " ";
  cout << endl;
}

inline bool areSameScalarsToEps(scalar_t a, scalar_t b) {
    return fabs(a - b) < 1e-4;
}

int main(int argc, char *argv[]) {
  string filename("smalltest.dat");
  size_t d = 2;
  scalar_t h = 3.;
  scalar_t lambda = 1.;
  KernelType ktype = KernelType::GAUSS;
  string mode("test");
  int M;
  TaskTimer timer("compression");

  cout << "# usage: ./KernelRegression file d h lambda "
       << "kernel(Gauss, Laplace) mode(valid, test)" << std::endl;
  if (argc > 1) filename = string(argv[1]);
  if (argc > 2) d = stoi(argv[2]);
  if (argc > 3) h = stof(argv[3]);
  if (argc > 4) lambda = stof(argv[4]);
  if (argc > 5) ktype = kernel_type(string(argv[5]));
  if (argc > 6) mode = string(argv[6]);
  if (argc > 7) M = stoi(argv[7]);
  cout << std::endl;
  cout << "# data dimension   = " << d << std::endl;
  cout << "# kernel sigma (h) = " << h << std::endl;
  cout << "# lambda           = " << lambda << std::endl;
  cout << "# kernel type      = " << get_name(ktype) << std::endl;
  cout << "# validation/test  = " << mode << std::endl;

  HSSOptions<scalar_t> hss_opts;
  hss_opts.set_verbose(true);
  hss_opts.set_from_command_line(argc, argv);
  // hss_opts.describe_options();
  cout << "# hss_rel_tol      = " << hss_opts.rel_tol() << std::endl;
  cout << "# hss_leaf_size    = " << hss_opts.leaf_size() << std::endl ;
  cout << "# hss_d0           = " << hss_opts.d0() << std::endl ;
  cout << "# hss_dd           = " << hss_opts.dd() << std::endl;
  cout << "# M (Nystrom)      = " << M << std::endl << std::endl;

  cout << "# Reading data ..." << std::endl;
  timer.start();
  // Read from csv files
  auto training     = read_from_file<scalar_t>(filename + "_train.csv");
  auto testing      = read_from_file<scalar_t>(filename + "_" + mode + ".csv");
  auto train_labels = read_from_file<scalar_t>(filename + "_train_label.csv");
  auto test_labels  = read_from_file<scalar_t>(filename + "_" + mode + "_label.csv");
  cout << "## Reading took " << timer.elapsed() << std::endl;

  size_t n = training.size() / d;
  size_t m = testing.size() / d;
  cout << "# training dataset = " << n << " x " << d << std::endl;
  cout << "# testing dataset  = " << m << " x " << d << std::endl;

  // Wrap datasets into DenseMatrixWrapper objects
  DenseMatrixWrapper<scalar_t>
    training_points(d, n, training.data(), d),
    test_points(d, m, testing.data(), d);

  #if 0
    std::vector<std::size_t> Mid(M);
    std::uniform_int_distribution<std::size_t> dist(0, n-1);
    std::mt19937 gen(1); // reproducible
    std::generate(Mid.begin(), Mid.end(), [&]() { return dist(gen); });
  #else
    M = 10;
    cout << "# Nystrom centers  = " << M << std::endl;
    std::vector<std::size_t> Mid{8148-1, 9058-1, 1270-1, 9132-1, 6322-1, 975-1, 2784-1, 5465-1, 9568-1, 9641-1};
  #endif

  // Take a random subset Mid of size Mid.size() from the original dataset
  DenseMatrix<scalar_t> training_Nystrom(d, M);
  for (std::size_t j=0; j<M; j++)
    for (std::size_t i=0; i<d; i++)
      training_Nystrom(i, j) = training_points(i, Mid[j]);
  // training_Nystrom.print("training_Nystrom", true, 10);

  scalar_t l0(0.);
  // Kernel object setup for the complete dataset
  auto K = create_kernel<scalar_t>(ktype, training_points, h, l0);
  // Kernel object setup for Nystrom (reduced) dataset
  auto K_Nystrom = create_kernel<scalar_t>(ktype, training_Nystrom, h, l0);

  // Clustering
  std::cout << "Clustering ..." << std::endl;
  timer.start();
  std::vector<int> perm;
  auto t = binary_tree_clustering(hss_opts.clustering_algorithm(),
     K_Nystrom->data(), perm, hss_opts.leaf_size());

  // Permute columns of matrix training_Nystrom
  // training_Nystrom.lapmt(perm, true);
  // Get inverse permutation
  // std::vector<int> iperm;
  // iperm.resize(perm.size());
  // for (std::size_t i=0; i<M; i++)
  //   iperm[perm[i]-1] = i+1;
  cout << "## Clustering took: " << timer.elapsed() << std::endl;

  std::cout << "Forming Kmm ..." << std::endl;
  timer.start();
  DenseMatrix<scalar_t> Kmm(M, M);
  #pragma omp parallel for collapse(2)
  for (std::size_t j=0; j<M; j++)
    for (std::size_t i=0; i<M; i++)
      Kmm(i, j) = K_Nystrom->eval(i, j);
  cout << "## Elapsed: " << timer.elapsed() << std::endl;
  // Kmm.print("Kmm",true,10);

  std::cout << "Forming Knm ..." << std::endl;
  timer.start();
  DenseMatrix<scalar_t> Knm(n, M);
  #pragma omp parallel for collapse(2)
  for (std::size_t j=0; j<M; j++)
    for (std::size_t i=0; i<n; i++)
      Knm(i, j) = K->eval(i, Mid[j]);
      // Knm(i, j) = K->eval(i, Mid[iperm[j]-1]);
  cout << "## Elapsed: " << timer.elapsed() << std::endl;
  // Knm.print("Knm",true,10);

  std::cout << "Forming H dense ..."  << std::endl;
  std::cout << "H = (KnM^t)*(KnM) + n*lambda*(Kmm)" << std::endl;
  timer.start();
  DenseMatrix<scalar_t> Hdense(Kmm);
  gemm(Trans::T, Trans::N, scalar_t(1.), Knm, Knm,
       scalar_t(n*lambda), Hdense);
  cout << "## Elapsed: " << timer.elapsed() << std::endl;
  // Hdense.print("Hdense",true,10);

  // Compression to HSS
  std::cout << "# H compression to HSS ..." << std::endl;
  timer.start();
  HSSMatrix<scalar_t> H(t, hss_opts);
  H.compress(Hdense, hss_opts);

  if (H.is_compressed())
    std::cout << "# created HSS matrix of dimension "
              << H.rows() << " x " << H.cols()
              << " with " << H.levels() << " levels" << std::endl
              << "# compression succeeded!" << std::endl;
  else std::cout << "# compression failed!!!" << std::endl;

  cout << "# rank(H) = " << H.rank() << std::endl;
  cout << "# memory(H) = " << H.memory()/1e6 << " MB, "
       << 100. * H.memory() / Hdense.memory() << "% of dense" << std::endl;
  cout << "## Compression took: " << timer.elapsed() << std::endl;

  // Factorization and solve
  cout << "# Factorization ..." << std::endl;
  timer.start();
  auto ULV = H.factor();
  cout << "## Factorization took: " << timer.elapsed() << std::endl;

  cout << "# Solve ..." << std::endl;
  timer.start();
  // z = (Knm') * y;
  // weights = H\z;
  DenseMatrix<scalar_t> y(n, 1, &train_labels[0], n);
  DenseMatrix<scalar_t> z(M, 1);
  gemm(Trans::T, Trans::N, scalar_t(1.), Knm, y, scalar_t(0.), z);
  // z.print("z",true,10);

  DenseMatrix<scalar_t> weights(z);
  H.solve(ULV, weights);
  // weights.print("weights",true,10);
  cout << "## Solve took: " << timer.elapsed() << std::endl;

  // Prediction
  cout << "# Prediction ..." << endl;
  timer.start();

  DenseMatrix<scalar_t> Kr(m,M);
  for (int c = 0; c < m; c++)
    for (int r = 0; r < M; r++)
      Kr(c, r) = Gauss_kernel(&testing[c*d], &training[Mid[r]*d], d, h);
  // Kr.print("Kr",true,10);
  // assert( areSameScalarsToEps(Kr.norm(), 64.299353) );

  // Matrix vector product to compute predictions
  DenseMatrix<scalar_t> prediction(m, 1);
  gemm(Trans::N, Trans::N, scalar_t(1.), Kr, weights, scalar_t(0.), prediction);

  // Quantify threshold
  for (int i = 0; i < m; ++i)
    prediction(i,0) = ((prediction(i,0) > 0) ? 1. : -1.);

  // Compute accuracy score of the predictions
  scalar_t incorrect_quant = 0;
  for (int i = 0; i < m; ++i) {
    scalar_t a = (prediction(i,0) - test_labels[i]) / 2;
    incorrect_quant += (a > 0 ? a : -a);
  }

  scalar_t c_err = 100.0 - (((m - incorrect_quant) / m) * 100);
  cout << "## Prediction := " << timer.elapsed() << std::endl;
  cout << "## c-err: " << c_err << "%" << std::endl;
  // assert( areSameScalarsToEps(c_err, 22.6) );

  return 0;
}
