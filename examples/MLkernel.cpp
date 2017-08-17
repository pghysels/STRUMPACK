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
 * reproduce, prepare derivative works, and perform publicly and display publicly.
 * Beginning five (5) years after the date permission to assert copyright is
 * obtained from the U.S. Department of Energy, and subject to any subsequent five
 * (5) year renewals, the U.S. Government is granted for itself and others acting
 * on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, prepare derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 *
 */
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include "HSS/HSSMatrix.hpp"
#include "TaskTimer.hpp"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

#define ERROR_TOLERANCE 1e2

const int kmeans_max_it = 20;
random_device rd;
mt19937 generator(rd());

inline double dist2(double* x, double* y, int d) {
  double k = 0.;
  for (int i=0; i<d; i++) k += pow(x[i] - y[i], 2.);
  return k;
}

inline double dist(double* x, double* y, int d) {
  return sqrt(dist2(x, y, d));
}

inline double norm1(double* x, double* y, int d) {
  double k = 0.;
  for (int i=0; i<d; i++) k += fabs(x[i] - y[i]);
  return k;
}

inline double Gauss_kernel(double* x, double* y, int d, double h) {
  return exp(-dist2(x, y, d)/(h*h));
}

inline double Laplace_kernel(double* x, double* y, int d, double h) {
  return exp(-norm1(x, y, d)/h);
}

void k_means(int k, double* p, int n, int d, int* nc) {
  // pick k random centers
  uniform_int_distribution<> uniform_random(0, n-1);
  double** center = new double*[k];
  for (int c=0; c<k; c++) {
    center[c] = new double[d];
    auto t = uniform_random(generator);
    for (int j=0; j<d; j++)
      center[c][j] = p[t*d+j];
  }
  int iter = 0;
  int* cluster = new int[n];
  while (iter < kmeans_max_it) {
    // for each point, find the closest cluster center
    for (int i=0; i<n; i++) {
      double min_dist = dist2(&p[i*d], center[0], d);
      cluster[i] = 0;
      for (int c=1; c<k; c++) {
	double dd = dist2(&p[i*d], center[c], d);
	if (dd < min_dist) {
	  min_dist = dd;
	  cluster[i] = c;
	}
      }
    }
    // update cluster centers
    for (int c=0; c<k; c++) {
      nc[c] = 0;
      for (int j=0; j<d; j++) center[c][j] = 0.;
    }
    for (int i=0; i<n; i++) {
      auto c = cluster[i];
      nc[c]++;
      for (int j=0; j<d; j++) center[c][j] += p[i*d+j];
    }
    for (int c=0; c<k; c++)
      for (int j=0; j<d; j++) center[c][j] /= nc[c];
    iter++;
  }

  int* ci = new int[k];
  for (int c=0; c<k; c++) ci[c] = 0;
  double* p_perm = new double[n*d];
  int row = 0;
  for (int c=0; c<k; c++) {
    for (int j=0; j<nc[c]; j++) {
      while (cluster[ci[c]] != c) ci[c]++;
      for (int l=0; l<d; l++)
	p_perm[l+row*d] = p[l+ci[c]*d];
      ci[c]++;
      row++;
    }
  }
  copy(p_perm, p_perm+n*d, p);
  delete[] p_perm;
  delete[] ci;

  for (int i=0; i<k; i++) delete[] center[i];
  delete[] center;
  delete[] cluster;
}

void recursive_2_means(double* p, int n, int d, int cluster_size, HSSPartitionTree& tree) {
  if (n < cluster_size) return;
  auto nc = new int[2];
  k_means(2, p, n, d, nc);
  if (nc[0] == 0 || nc[1] == 0) return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_2_means(p+d,       nc[0], d, cluster_size, tree.c[0]);
  recursive_2_means(p+nc[0]*d, nc[1], d, cluster_size, tree.c[1]);
  delete[] nc;
}

int main(int argc, char* argv[]) {

  string filename("smalltest.dat");
  int d = 2;
  bool reorder = true;
  double h = 3.;
  int kernel = 1; // Gaussian=1, Laplace=2

  cout << "# usage: ./ML_kernel file d h kernel(1=Gauss,2=Laplace) reorder(0=natural,1=recursive 2-means)" << endl;
  if (argc > 1) filename = string(argv[1]);
  if (argc > 2) d = stoi(argv[2]);
  if (argc > 3) h = stof(argv[3]);
  if (argc > 4) kernel = stoi(argv[4]);
  if (argc > 5) reorder = bool(stoi(argv[5]));
  cout << "# data dimension = " << d << endl;
  cout << "# kernel h = " << h << endl;
  cout << "# kernel type = " << ((kernel == 1) ? "Gauss" : "Laplace") << endl;
  cout << "# reordering/clustering = " << (reorder ? "recursive 2-means" : "natural") << endl;

  HSSOptions<double> hss_opts;
  hss_opts.set_verbose(true);
  hss_opts.set_from_command_line(argc, argv);

  vector<double> data;
  {
    ifstream f(filename);
    string l;
    while (getline(f, l)) {
      istringstream sl(l);
      string s;
      while (getline(sl, s, ',')) data.push_back(stod(s));
    }
    f.close();
  }

  int n = data.size() / d;
  cout << "# matrix size = " << n << " x " << d << endl;

  HSSPartitionTree cluster_tree;
  cluster_tree.size = n;
  int cluster_size = hss_opts.leaf_size();
  if (reorder)
    recursive_2_means(data.data(), n, d, cluster_size, cluster_tree);

  cout << "constructing Kdense .. " << endl;

  DenseMatrix<double> Kdense(n, n);
  if (kernel == 1) {
#pragma omp parallel for collapse(2)
    for (int c=0; c<n; c++)
      for (int r=0; r<n; r++)
	Kdense(r, c) = Gauss_kernel(&data[r*d], &data[c*d], d, h);
  } else {
#pragma omp parallel for collapse(2)
    for (int c=0; c<n; c++)
      for (int r=0; r<n; r++)
	Kdense(r, c) = Laplace_kernel(&data[r*d], &data[c*d], d, h);
  }

  vector<double>().swap(data); // clear data

  cout << "starting HSS compression .. " << endl;


  HSSMatrix<double> K;
  if (reorder) K = HSSMatrix<double>(cluster_tree, hss_opts);
  else K = HSSMatrix<double>(n, n, hss_opts);

  TaskTimer::t_begin = GET_TIME_NOW();
  TaskTimer timer(string("compression"), 1);
  timer.start();
  K.compress(Kdense, hss_opts);
  cout << "# compression time = " << timer.elapsed() << endl;

  if (K.is_compressed()) {
    cout << "# created K matrix of dimension " << K.rows() << " x " << K.cols()
	 << " with " << K.levels() << " levels" << endl;
    cout << "# compression succeeded!" << endl;
  } else {
    cout << "# compression failed!!!!!!!!" << endl;
    return 1;
  }
  cout << "# rank(K) = " << K.rank() << endl;
  cout << "# memory(K) = " << K.memory()/1e6 << " MB, "
       << 100. * K.memory() / Kdense.memory() << "% of dense" << endl;

  // K.print_info();
  auto Ktest = K.dense();
  Ktest.scaled_add(-1., Kdense);
  cout << "# relative error = ||Kdense-K*I||_F/||Kdense||_F = " << Ktest.normF() / Kdense.normF() << endl;
  if (Ktest.normF() / Kdense.normF() > ERROR_TOLERANCE * max(hss_opts.rel_tol(),hss_opts.abs_tol())) {
    cout << "ERROR: compression error too big!!" << endl;
    return 1;
  }
  return 0;
}
