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
#include "dense/DistributedMatrix.hpp"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

// random_device rd;
mt19937 generator(1);

class KernelMPI {
  using DenseM_t = DenseMatrix<double>;
  using DenseMW_t = DenseMatrixWrapper<double>;
  using DistM_t = DistributedMatrix<double>;
  using DistMW_t = DistributedMatrixWrapper<double>;

public:
  vector<double> _data;
  int _d = 0;
  int _n = 0;
  double _h = 0.;
  double _l = 0.;
  KernelMPI() = default;
  KernelMPI(vector<double> data, int d, double h, double l)
    : _data(std::move(data)), _d(d), _n(_data.size() / _d), _h(h), _l(l) {
    assert(_n * _d == _data.size());
  }

  void operator()(const vector<size_t> &I, const vector<size_t> &J,
  DistM_t &B) {
    if (!B.active()) return;
    assert(I.size() == B.rows() && J.size() == B.cols());
    for (size_t j = 0; j < J.size(); j++) {
      if (B.colg2p(j) != B.pcol()) continue;
      for (size_t i = 0; i < I.size(); i++) {
        if (B.rowg2p(i) == B.prow()) {
          assert(B.is_local(i, j));
          B.global(i, j) = Gauss_kernel(&_data[I[i] * _d], &_data[J[j] * _d], _d, _h);
          if (I[i] == J[j])
            B.global(i, j) += _l;
        }
      }
    }
  }

  void times(DenseM_t &R, DistM_t &S, int Rprow) {
    const auto B = S.MB();
    const auto Bc = S.lcols();
    DenseM_t Asub(B, B);
    #pragma omp parallel for firstprivate(Asub) schedule(dynamic)
    for (int lr = 0; lr < S.lrows(); lr += B) {
      const int Br = std::min(B, S.lrows() - lr);
      const int Ar = S.rowl2g(lr);
      for (int k = 0, Ac = Rprow*B; Ac < _n; k += B) {
        const int Bk = std::min(B, _n - Ac);
        // construct a block of A
        for (size_t j = 0; j < Bk; j++) {
          for (size_t i = 0; i < Br; i++) {
            Asub(i, j) = Gauss_kernel
              (&_data[(Ar + i) * _d], &_data[(Ac + j) * _d], _d, _h);
          }
          if (Ar==Ac) Asub(j,j) += _l;
        }
        DenseMW_t Ablock(Br, Bk, Asub, 0, 0);
        DenseMW_t Sblock(Br, Bc, &S(lr, 0), S.ld());
        DenseMW_t Rblock(Bk, Bc, &R(k, 0), R.ld());
        // multiply block of A with a row-block of Rr and add result to Sr
        gemm(Trans::N, Trans::N, 1., Ablock, Rblock, 1., Sblock);
        Ac += S.nprows() * B;
      }
    }
  }

  void operator()(DistM_t &R, DistM_t &Sr, DistM_t &Sc) {
    Sr.zero();
    int maxlocrows = R.MB() * (R.rows() / R.MB());
    if (R.rows() % R.MB()) maxlocrows += R.MB();
    int maxloccols = R.MB() * (R.cols() / R.MB());
    if (R.cols() % R.MB()) maxloccols += R.MB();
    DenseM_t tmp(maxlocrows, maxloccols);
    // each processor broadcasts his/her local part of R to all
    // processes in the same column of the BLACS grid, one after the
    // other
    for (int p=0; p<R.nprows(); p++) {
      if (p == R.prow()) {
        strumpack::scalapack::gebs2d
          (R.ctxt(), 'C', ' ', R.lrows(), R.lcols(), R.data(), R.ld());
        DenseMW_t Rdense(R.lrows(), R.lcols(), R.data(), R.ld());
        strumpack::copy(Rdense, tmp, 0, 0);
      } else {
        int recvrows = strumpack::scalapack::numroc
          (R.rows(), R.MB(), p, 0, R.nprows());
        strumpack::scalapack::gebr2d
          (R.ctxt(), 'C', ' ', recvrows, R.lcols(),
           tmp.data(), tmp.ld(), p, R.pcol());
      }
      times(tmp, Sr, p);
    }
    Sc = Sr;
  }
};

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  auto P = mpi_nprocs(MPI_COMM_WORLD);
  // initialize the BLACS grid
  int npcol = floor(sqrt((float)P));
  int nprow = P / npcol;
  int ctxt, dummy, prow, pcol;
  scalapack::Cblacs_get(0, 0, &ctxt);
  scalapack::Cblacs_gridinit(&ctxt, "C", nprow, npcol);
  scalapack::Cblacs_gridinfo(ctxt, &dummy, &dummy, &prow, &pcol);
  int ctxt_all = scalapack::Csys2blacs_handle(MPI_COMM_WORLD);
  scalapack::Cblacs_gridinit(&ctxt_all, "R", 1, P);

  // Get BLACSGrid object
  BLACSGrid grid(MPI_COMM_WORLD);

  string filename("smalltest.dat");
  int d = 2;
  string reorder("natural");
  double h = 3.;
  double lambda = 1.;
  int kernel = 1; // Gaussian=1, Laplace=2
  double total_time = 0.;
  string mode("test");

  if (!mpi_rank())
    cout << "# usage: ./KernelRegression file d h kernel(1=Gauss,2=Laplace) "
      "reorder(natural, 2means, kd, pca) lambda"
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

  if (!mpi_rank()) {
    cout << "# data dimension = " << d << endl;
    cout << "# kernel h = " << h << endl;
    cout << "# lambda = " << lambda << endl;
    cout << "# kernel type = "
         << ((kernel == 1) ? "Gauss" : "Laplace") << endl;
    cout << "# reordering/clustering = " << reorder << endl;
    cout << "# validation/test = " << mode << endl;
  }

  HSSOptions<double> hss_opts;
  hss_opts.set_verbose(true);
  hss_opts.set_from_command_line(argc, argv);

  vector<double> data_train = write_from_file(filename + "_train.csv");
  vector<double> data_test = write_from_file(filename + "_" + mode + ".csv");
  vector<double> data_train_label =
    write_from_file(filename + "_train_label.csv");
  vector<double> data_test_label =
    write_from_file(filename + "_" + mode + "_label.csv");

  int n = data_train.size() / d;
  int m = data_test.size() / d;
  if (!mpi_rank())
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

  if (!mpi_rank())
    cout << "starting HSS compression .. " << endl;

  HSSMatrixMPI<double>* K = nullptr;

  TaskTimer::t_begin = GET_TIME_NOW();
  TaskTimer timer(string("compression"), 1);
  timer.start();
  KernelMPI kernel_matrix(data_train, d, h, lambda);

  // if (reorder != "natural")
  //   K = new HSSMatrixMPI<double>
  //     (cluster_tree, kernel_matrix, ctxt, kernel_matrix,
  //      hss_opts, MPI_COMM_WORLD);
  // else
  //   K = new HSSMatrixMPI<double>
  //     (n, n, kernel_matrix, ctxt, kernel_matrix,
  //      hss_opts, MPI_COMM_WORLD);


// HSSMatrixMPI(std::size_t m, std::size_t n, const BLACSGrid* Agrid,
// const dmult_t& Amult, const delem_t& Aelem, const opts_t& opts);

  K = new HSSMatrixMPI<double>(n, n, &grid, kernel_matrix, kernel_matrix,
  hss_opts);

  if (!mpi_rank())
    cout << "# compression time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();

  if (K->is_compressed()) {
    // reduction over all processors
    const auto max_rank = K->max_rank();
    const auto total_memory = K->total_memory();
    if (!mpi_rank())
      cout << "# created K matrix of dimension "
           << K->rows() << " x " << K->cols()
           << " with " << K->levels() << " levels" << endl
           << "# compression succeeded!" << endl
           << "# rank(K) = " << max_rank << endl
           << "# memory(K) = " << total_memory / 1e6 << " MB " << endl;
  } else {
    if (!mpi_rank())
      cout << "# compression failed!!!!!!!!" << endl;
    return 1;
  }

  if (!mpi_rank())
    cout << "factorization start" << endl;
  timer.start();
  auto ULV = K->factor();
  if (!mpi_rank())
    cout << "# factorization time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();

  DenseMatrix<double> B(n, 1, &data_train_label[0], n);
  DenseMatrix<double> weights(B);
  DistributedMatrix<double> Bdist(&grid, B);
  DistributedMatrix<double> wdist(&grid, weights);

  if (!mpi_rank())
    cout << "solution start" << endl;
  timer.start();
  K->solve(ULV, wdist);
  if (!mpi_rank())
    cout << "# solve time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();
  if (!mpi_rank())
    cout << "# total time: " << total_time << endl;

  auto Bcheck = K->apply(wdist);

  Bcheck.scaled_add(-1., Bdist);
  auto Bchecknorm = Bcheck.normF() / Bdist.normF();
  if (!mpi_rank())
    cout << "# relative error = ||B-H*(H\\B)||_F/||B||_F = "
         << Bchecknorm << endl;

  double* prediction = new double[m];
  std::fill(prediction, prediction+m, 0.);

  if (kernel == 1) {
    for (int c = 0; c < m; c++) {
      for (int r = 0; r < n; r++) {
        prediction[c] +=
          Gauss_kernel(&data_train[r * d], &data_test[c * d], d, h) *
          weights(r, 0);
      }
    }
  } else {
    for (int c = 0; c < m; c++) {
      for (int r = 0; r < n; r++) {
        prediction[c] +=
          Laplace_kernel(&data_train[r * d], &data_test[c * d], d, h) *
          wdist(r, 0);
      }
    }
  }

  for (int i = 0; i < m; ++i)
    prediction[i] = ((prediction[i] > 0) ? 1. : -1.);

  // compute accuracy score of prediction
  double incorrect_quant = 0;
  for (int i = 0; i < m; ++i) {
    double a = (prediction[i] - data_test_label[i]) / 2;
    incorrect_quant += (a > 0 ? a : -a);
  }
  if (!mpi_rank())
    cout << "# prediction score: " << ((m - incorrect_quant) / m) * 100 << "%"
         << endl << endl;

  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}