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

int main(int argc, char *argv[]) {
  // HODLR code does not support float (yet)
  using scalar_t = double;

  MPI_Init(&argc, &argv);
  MPIComm c;

  string filename("./data/susy_10Kn");
  int d = 8;
  scalar_t h = 1.3;
  scalar_t lambda = 3.11;
  int p = 1;  // kernel degree
  KernelType ktype = KernelType::GAUSS;
  string mode("test");
  TaskTimer timer("prediction");
  double rtol = 1e-2;

  if (c.is_root())
    cout << "# usage: ./KernelRegression file d h lambda degree "
         << "kernel(Gauss, Laplace, ANOVA) mode(valid, test)" << endl;
  if (argc > 1) filename = string(argv[1]);
  if (argc > 2) d = stoi(argv[2]);
  if (argc > 3) h = stof(argv[3]);
  if (argc > 4) lambda = stof(argv[4]);
  if (argc > 5) p = stoi(argv[5]);
  if (argc > 6) ktype = kernel_type(string(argv[6]));
  if (argc > 7) mode = string(argv[7]);
  if (c.is_root())
    cout << "# file            = " << filename << endl
         << "# data dimension  = " << d << endl
         << "# kernel h        = " << h << endl
         << "# lambda          = " << lambda << endl
         << "# p               = " << p << endl
         << "# kernel type     = " << get_name(ktype) << endl
         << "# validation/test = " << mode << endl;

  auto training     = read_from_file<scalar_t>(filename + "_train.csv");
  auto testing      = read_from_file<scalar_t>(filename + "_" + mode + ".csv");
  auto train_labels = read_from_file<scalar_t>(filename + "_train_label.csv");
  auto test_labels  = read_from_file<scalar_t>(filename + "_" + mode + "_label.csv");

  size_t n = training.size() / d;
  size_t m = testing.size() / d;
  if (c.is_root())
    cout << "# training dataset = " << n << " x " << d << endl
         << "# testing dataset  = " << m << " x " << d << endl << endl;

  DenseMatrixWrapper<scalar_t>
    training_points(d, n, training.data(), d),
    test_points(d, m, testing.data(), d);

  auto check = [&](const std::vector<scalar_t>& prediction) {
    // compute accuracy score of prediction
    if (c.is_root()) {
      size_t incorrect_quant = 0;
      for (size_t i=0; i<m; i++)
        if ((prediction[i] >= 0 && test_labels[i] < 0) ||
            (prediction[i] < 0 && test_labels[i] >= 0))
          incorrect_quant++;
      cout << "# prediction score: "
      << (float(m - incorrect_quant) / m) * 100. << "%" << endl
      << "# c-err: "
      << (float(incorrect_quant) / m) * 100. << "%"
      << endl << endl;
    }
  };

  auto K = create_kernel<scalar_t>(ktype, training_points, h, lambda, p);

  {
    HSSOptions<scalar_t> opts;
    opts.set_verbose(true);
    opts.set_rel_tol(rtol);
    opts.set_from_command_line(argc, argv);
    if (c.is_root()) opts.describe_options();

    BLACSGrid g(c);
    auto weights = K->fit_HSS(g, train_labels, opts);
    if (c.is_root()) cout << endl << "# HSS prediction start..." << endl;
    timer.start();
    auto prediction = K->predict(test_points, weights);
    if (c.is_root()) cout << "# prediction took " << timer.elapsed() << endl;
    check(prediction);
  }

#if defined(STRUMPACK_USE_BPACK)
  {
    HODLR::HODLROptions<scalar_t> opts;
    opts.set_verbose(true);
    opts.set_rel_tol(rtol);
    opts.set_from_command_line(argc, argv);
    if (c.is_root()) opts.describe_options();

    auto weights = K->fit_HODLR(c, train_labels, opts);
    if (c.is_root()) cout << endl << "# HODLR prediction start..." << endl;
    timer.start();
    auto prediction = K->predict(test_points, weights);
    if (c.is_root()) cout << "# prediction took " << timer.elapsed() << endl;
    check(prediction);
  }
#endif

  MPI_Finalize();
  return 0;
}
