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
#include <fstream>

typedef float real_t;

#include "clustering/PCAPartitioning.hpp"
#include "clustering/KDTree.hpp"
#include "clustering/KMeans.hpp"
#include "clustering/NeighborSearch.hpp"
#include "HSS/HSSMatrix.hpp"
#include "misc/TaskTimer.hpp"
// #include "FileManipulation.h"
#include "preprocessing.h"
#include "dense/DistributedMatrix.hpp"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

// random_device rd;
mt19937 generator(1); // Fixed seed

class KernelMPI {
  using DenseM_t = DenseMatrix<real_t>;
  using DenseMW_t = DenseMatrixWrapper<real_t>;
  using DistM_t = DistributedMatrix<real_t>;
  using DistMW_t = DistributedMatrixWrapper<real_t>;

public:
  vector<real_t> data_;
  size_t d_ = 0;
  size_t n_ = 0;
  real_t h_ = 0.;
  real_t l_ = 0.;
  KernelMPI() = default;

  KernelMPI(vector<real_t> data, int d, real_t h, real_t l)
    : data_(std::move(data)), d_(d), n_(data_.size() / d_), h_(h), l_(l) {
    assert(n_ * d_ == data_.size());
  }

  /**
   * Compute selected elements {I,J} of the kernel matrix,
   * and put them in matrix B.
   */
  void operator()
  (const vector<size_t>& I, const vector<size_t>& J, DistM_t& B) {
    if (!B.active()) return;
    assert(I.size() == B.rows() && J.size() == B.cols());
    for (size_t j=0; j<J.size(); j++) {
      if (B.colg2p(j) != B.pcol()) continue;
      for (size_t i=0; i<I.size(); i++) {
        if (B.rowg2p(i) == B.prow()) {
          assert(B.is_local(i, j));
          B.global(i, j) = Gauss_kernel
            (&data_[I[i]*d_], &data_[J[j]*d_], d_, h_);
          if (I[i] == J[j]) B.global(i, j) += l_;
        }
      }
    }
  }
};

vector<real_t> read_from_file(string filename) {
  vector<real_t> data;
  ifstream f(filename);
  string l;
  while (getline(f, l)) {
    istringstream sl(l);
    string s;
    while (getline(sl, s, ','))
      data.push_back(stod(s));
  }
  return data;
}

int run(int argc, char *argv[]) {
  MPIComm c;
  BLACSGrid grid(c);

  string filename("smalltest.dat");
  int d = 2;
  string reorder("natural");
  real_t h = 3.;
  real_t lambda = 1.;
  int kernel = 1; // Gaussian=1, Laplace=2
  real_t total_time = 0.;
  string mode("test");

  if (c.is_root())
    cout << "# usage: ./KernelRegression file d h kernel(1=Gauss,2=Laplace) "
            "reorder(natural, 2means, kd, pca) lambda"
         << endl;
  if (argc > 1) filename = string(argv[1]);
  if (argc > 2) d = stoi(argv[2]);
  if (argc > 3) h = stof(argv[3]);
  if (argc > 4) lambda = stof(argv[4]);
  if (argc > 5) kernel = stoi(argv[5]);
  if (argc > 6) reorder = string(argv[6]);
  if (argc > 7) mode = string(argv[7]);

  if (c.is_root()) {
    cout << "# data dimension = " << d << endl;
    cout << "# kernel h = " << h << endl;
    cout << "# lambda = " << lambda << endl;
    cout << "# kernel type = "
         << ((kernel == 1) ? "Gauss" : "Laplace") << endl;
    cout << "# reordering/clustering = " << reorder << endl;
    cout << "# validation/test = " << mode << endl;
  }

  HSSOptions<real_t> hss_opts;
  hss_opts.set_verbose(true);
  hss_opts.set_from_command_line(argc, argv);

  vector<real_t> data_train       = read_from_file(filename + "_train.csv");
  vector<real_t> data_test        = read_from_file(filename + "_" + mode + ".csv");
  vector<real_t> data_train_label = read_from_file(filename + "_train_label.csv");
  vector<real_t> data_test_label  = read_from_file(filename + "_" + mode + "_label.csv");

  int n = data_train.size() / d;
  int m = data_test.size() / d;
  if (c.is_root()){
    cout << "# matrix size = " << n << " x " << d << endl;
    cout << "# test size = " << m << " x " << d << endl;
  }
  DenseMatrixWrapper<real_t> train_matrix(d, n, data_train.data(), d);
  DenseMatrixWrapper<real_t> label_matrix(1, n, data_train_label.data(), 1);

  HSSPartitionTree cluster_tree;
  cluster_tree.size = n;
  int cluster_size = hss_opts.leaf_size();

  if (reorder == "2means")
    recursive_2_means
      (train_matrix, cluster_size, cluster_tree, label_matrix, generator);
  else if (reorder == "kd")
    recursive_kd(train_matrix, cluster_size, cluster_tree, label_matrix);
  else if (reorder == "pca")
    recursive_pca(train_matrix, cluster_size, cluster_tree, label_matrix);

  TaskTimer::t_begin = GET_TIME_NOW();
  TaskTimer timer(string("compression"), 1);

  if (c.is_root())
    cout << "finding ANN.. " << endl;

  // Find ANN: start ------------------------------------------------
  int ann_number = 64;
  int num_iters = 5;
  DenseMatrix<uint32_t> neighbors;
  DenseMatrix<real_t> scores;
  timer.start();
  find_approximate_neighbors
    (train_matrix, num_iters, ann_number, neighbors, scores, generator);
  if (c.is_root()) {
    cout << "# ANN time = " << timer.elapsed() << " sec" <<endl;
    total_time += timer.elapsed();
  }
  // Find ANN: end ------------------------------------------------

  if (c.is_root())
    cout << "starting HSS compression .. " << endl;

  HSSMatrixMPI<real_t> K;
  timer.start();
  KernelMPI kernel_matrix(data_train, d, h, lambda);
  // Constructor for ANN compression
  if (reorder != "natural")
    K = HSSMatrixMPI<real_t>
      (cluster_tree, &grid, neighbors, scores, kernel_matrix, hss_opts);
  else
    K = HSSMatrixMPI<real_t>
      (n, n, &grid, neighbors, scores, kernel_matrix, hss_opts);
  if (c.is_root())
    cout << "# compression time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();

  if (K.is_compressed()) {
    // reduction over all processors
    const auto max_rank = K.max_rank();
    const auto total_memory = K.total_memory();
    if (c.is_root())
      cout << "# created K matrix of dimension "
           << K.rows() << " x " << K.cols()
           << " with " << K.levels() << " levels" << endl
           << "# compression succeeded!" << endl
           << "# rank(K) = " << max_rank << endl
           << "# memory(K) = " << total_memory / 1e6 << " MB " << endl;
  } else {
    if (c.is_root())
      cout << "# compression failed!!!!!!!!" << endl;
    return 1;
  }

  if (c.is_root())
    cout << "factorization start" << endl;

  timer.start();
  auto ULV = K.factor();
  if (c.is_root())
    cout << "# factorization time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();

  DenseMatrix<real_t> B;
  DistributedMatrix<real_t> wdist(&grid, n, 1);
  if (c.is_root())
    B = DenseMatrix<real_t>(n, 1, &data_train_label[0], n);
  wdist.scatter(B);

  if (c.is_root())
    cout << "solution start" << endl;
  timer.start();
  K.solve(ULV, wdist);
  if (c.is_root())
    cout << "# solve time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();
  if (c.is_root())
    cout << "# total time: " << total_time << endl;

  auto weights = wdist.gather();

  //// ----- Error checking with dense matrix: start ----- ////
  if (n <= 1000) {
    // Build dense matrix out of HSS matrix
    auto KtestD = K.dense();
    auto Ktest = KtestD.gather(); // Gather matrix to rank 0
    if (c.is_root()) {
      // Build dense matrix to test error
      DenseMatrix<real_t> Kdense(n, n);
      if (kernel == 1) {
        for (int c=0; c<n; c++) {
          for (int r=0; r<n; r++)
            Kdense(r, c) = Gauss_kernel
              (&data_train[r*d], &data_train[c*d], d, h);
          Kdense(c, c) += lambda;
        }
      } else {
        for (int c=0; c<n; c++) {
          for (int r=0; r<n; r++)
            Kdense(r, c) = Laplace_kernel
              (&data_train[r*d], &data_train[c*d], d, h);
          Kdense(c, c) += lambda;
        }
      }
      Ktest.scaled_add(-1., Kdense);
      cout << "# compression error = ||Kdense-K*I||_F/||Kdense||_F = "
           << Ktest.normF() / Kdense.normF() << endl;
    }
  }
  //// ----- Error checking with dense matrix: end ----- ////

  // // Computing prediction accuracy on root rank
  // if (c.is_root()) {
  //   cout << "# Starting prediction step" << endl;
  //   timer.start();
  //   std::vector<real_t> prediction(m);
  //   if (kernel == 1) {
  //     for (int c = 0; c < m; c++)
  //       for (int r = 0; r < n; r++) {
  //         prediction[c] +=
  //           Gauss_kernel(&data_train[r * d], &data_test[c * d], d, h) *
  //           weights(r, 0);
  //       }
  //   } else {
  //     for (int c = 0; c < m; c++)
  //       for (int r = 0; r < n; r++) {
  //         prediction[c] +=
  //           Laplace_kernel(&data_train[r * d], &data_test[c * d], d, h) *
  //           weights(r, 0);
  //       }
  //   }
  //   for (int i = 0; i < m; ++i)
  //     prediction[i] = ((prediction[i] > 0) ? 1. : -1.);

  //   // compute accuracy score of prediction
  //   real_t incorrect_quant = 0;
  //   for (int i = 0; i < m; ++i) {
  //     real_t a = (prediction[i] - data_test_label[i]) / 2;
  //     incorrect_quant += (a > 0 ? a : -a);
  //   }
  //   if (c.is_root()){
  //     cout << "# seq prediction took " << timer.elapsed() << endl;
  //     cout << "# prediction score: "
  //          << ((m - incorrect_quant) / m) * 100 << "%"
  //          << endl << endl;
  //   }
  // }

  // Computing prediction in parallel
  if (!mpi_rank())
    cout << "# Starting prediction step" << endl;
  timer.start();

  double* prediction = new double[m];
  std::fill(prediction, prediction+m, 0.);

  timer.start();
  if (kernel == 1) {
    if (wdist.active() && wdist.lcols() > 0)
      #pragma omp parallel for
      for (int c = 0; c < m; c++) {
        for (int r = 0; r < wdist.lrows(); r++) {
          prediction[c] +=
            Gauss_kernel
            (&data_train[wdist.rowl2g(r) * d], &data_test[c * d], d, h)
            * wdist(r, 0);
        }
      }
  } else {
    if (wdist.active() && wdist.lcols() > 0)
      #pragma omp parallel for
      for (int c = 0; c < m; c++) {
        for (int r = 0; r < wdist.lrows(); r++) {
          prediction[c] +=
            Laplace_kernel
            (&data_train[wdist.rowl2g(r) * d], &data_test[c * d], d, h)
            * wdist(r, 0);
        }
      }
  }
  MPI_Allreduce
    (MPI_IN_PLACE, prediction, m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  #pragma omp parallel for
  for (int i = 0; i < m; ++i)
    prediction[i] = ((prediction[i] > 0) ? 1. : -1.);

  // compute accuracy score of prediction
  double incorrect_quant = 0;
  #pragma omp parallel for reduction(+:incorrect_quant)
  for (int i = 0; i < m; ++i) {
    double a = (prediction[i] - data_test_label[i]) / 2;
    incorrect_quant += (a > 0 ? a : -a);
  }

  if (!mpi_rank())
    cout << "# par prediction took " << timer.elapsed() << endl;
  if (!mpi_rank())
    cout << "# prediction score: " << ((m - incorrect_quant) / m) * 100 << "%"
         << endl << endl;

  return 0;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int ierr = run(argc, argv);
  MPI_Finalize();
  return ierr;
}
