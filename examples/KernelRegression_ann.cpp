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

#include "HSS/HSSMatrix.hpp"
#include "misc/TaskTimer.hpp"

#include "FileManipulation.h"
#include <fstream>

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

extern "C" {
#define SSYEVX_FC FC_GLOBAL(ssyevx, SSYEVX)
#define DSYEVX_FC FC_GLOBAL(dsyevx, DSYEVX)
void SSYEVX_FC(char *JOBZ, char *RANGE, char *UPLO, int *N, float *A, int *LDA,
               float *VL, float *VU, int *IL, int *IU, float *ABSTOL, int *M,
               float *W, float *Z, int *LDZ, float *WORK, int *LWORK,
               int *IWORK, int *IFAIL, int *INFO);
void DSYEVX_FC(char *JOBZ, char *RANGE, char *UPLO, int *N, double *A, int *LDA,
               double *VL, double *VU, int *IL, int *IU, double *ABSTOL, int *M,
               double *W, double *Z, int *LDZ, double *WORK, int *LWORK,
               int *IWORK, int *IFAIL, int *INFO);
}

inline int syevx(char JOBZ, char RANGE, char UPLO, int N, float *A, int LDA,
                 float VL, float VU, int IL, int IU, float ABSTOL, int &M,
                 float *W, float *Z, int LDZ) {
  int INFO;
  auto IWORK = new int[5 * N + N];
  auto IFAIL = IWORK + 5 * N;
  int LWORK = -1;
  float SWORK;
  SSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU, &ABSTOL, &M,
            W, Z, &LDZ, &SWORK, &LWORK, IWORK, IFAIL, &INFO);
  LWORK = int(SWORK);
  auto WORK = new float[LWORK];
  SSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU, &ABSTOL, &M,
            W, Z, &LDZ, WORK, &LWORK, IWORK, IFAIL, &INFO);
  delete[] WORK;
  delete[] IWORK;
  return INFO;
}
inline int dyevx(char JOBZ, char RANGE, char UPLO, int N, double *A, int LDA,
                 double VL, double VU, int IL, int IU, double ABSTOL, int &M,
                 double *W, double *Z, int LDZ) {
  int INFO;
  auto IWORK = new int[5 * N + N];
  auto IFAIL = IWORK + 5 * N;
  int LWORK = -1;
  double DWORK;
  DSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU, &ABSTOL, &M,
            W, Z, &LDZ, &DWORK, &LWORK, IWORK, IFAIL, &INFO);
  LWORK = int(DWORK);
  auto WORK = new double[LWORK];
  DSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU, &ABSTOL, &M,
            W, Z, &LDZ, WORK, &LWORK, IWORK, IFAIL, &INFO);
  delete[] WORK;
  delete[] IWORK;
  return INFO;
}

#define ERROR_TOLERANCE 1e2

const int kmeans_max_it = 100;
random_device rd;
double r;
mt19937 generator(rd());

inline double dist2(double *x, double *y, int d) {
  double k = 0.;
  for (int i = 0; i < d; i++)
    k += pow(x[i] - y[i], 2.);
  return k;
}

inline double dist(double *x, double *y, int d) { return sqrt(dist2(x, y, d)); }

inline double norm1(double *x, double *y, int d) {
  double k = 0.;
  for (int i = 0; i < d; i++)
    k += fabs(x[i] - y[i]);
  return k;
}

inline double Gauss_kernel(double *x, double *y, int d, double h) {
  return exp(-dist2(x, y, d) / (2. * h * h));
}

inline double Laplace_kernel(double *x, double *y, int d, double h) {
  return exp(-norm1(x, y, d) / h);
}

inline int *kmeans_start_random(int n, int k) {
  uniform_int_distribution<int> uniform_random(0, n - 1);
  int *ind_centers = new int[k];
  for (int i = 0; i < k; i++) {
    ind_centers[i] = uniform_random(generator);
  }
  return ind_centers;
}

// 3 more start sampling methods for the case k == 2

int *kmeans_start_random_dist_maximized(int n, double *p, int d) {
  constexpr size_t k = 2;

  uniform_int_distribution<int> uniform_random(0, n - 1);
  const auto t = uniform_random(generator);
  // compute probabilities
  double *cur_dist = new double[n];
  for (int i = 0; i < n; i++) {
    cur_dist[i] = dist2(&p[i * d], &p[t * d], d);
  }

  std::discrete_distribution<int> random_center(&cur_dist[0], &cur_dist[n]);

  delete[] cur_dist;

  int *ind_centers = new int[k];
  ind_centers[0] = t;
  ind_centers[1] = random_center(generator);
  return ind_centers;
}

// for k = 2 only
int *kmeans_start_dist_maximized(int n, double *p, int d) {
  constexpr size_t k = 2;

  // find centroid
  double centroid[d];

  for (int i = 0; i < d; i++) {
    centroid[i] = 0;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      centroid[j] += p[i * d + j];
    }
  }

  for (int j = 0; j < d; j++)
    centroid[j] /= n;

  // find farthest point from centroid
  int first_index = 0;
  double max_dist = -1;

  for (int i = 0; i < n; i++) {
    double dd = dist(&p[i * d], centroid, d);
    if (dd > max_dist) {
      max_dist = dd;
      first_index = i;
    }
  }
  // find fathest point from the firsth point
  int second_index = 0;
  max_dist = -1;
  for (int i = 0; i < n; i++) {
    double dd = dist(&p[i * d], &p[first_index * d], d);
    if (dd > max_dist) {
      max_dist = dd;
      second_index = i;
    }
  }
  int *ind_centers = new int[k];
  ind_centers[0] = first_index;
  ind_centers[1] = second_index;
  return ind_centers;
}

inline int *kmeans_start_fixed(int n, double *p, int d) {
  int *ind_centers = new int[2];
  ind_centers[0] = 0;
  ind_centers[1] = n - 1;
  return ind_centers;
}

void k_means(int k, double *p, int n, int d, int *nc, double *labels) {
  double **center = new double *[k];

  int *ind_centers = NULL;

  constexpr int kmeans_options = 2;
  switch (kmeans_options) {
  case 1:
    ind_centers = kmeans_start_random(n, k);
    break;
  case 2:
    ind_centers = kmeans_start_random_dist_maximized(n, p, d);
    break;
  case 3:
    ind_centers = kmeans_start_dist_maximized(n, p, d);
    break;
  case 4:
    ind_centers = kmeans_start_fixed(n, p, d);
    break;
  }

  for (int c = 0; c < k; c++) {
    center[c] = new double[d];
    for (int j = 0; j < d; j++)
      center[c][j] = p[ind_centers[c] * d + j];
  }

  int iter = 0;
  bool changes = true;
  int *cluster = new int[n];
  for (int i = 0; i < n; i++) {
    cluster[i] = 0;
  }

  while ((changes == true) and (iter < kmeans_max_it)) {
    // for each point, find the closest cluster center
    changes = false;
    for (int i = 0; i < n; i++) {
      double min_dist = dist(&p[i * d], center[0], d);
      cluster[i] = 0;
      for (int c = 1; c < k; c++) {
        double dd = dist(&p[i * d], center[c], d);
        if (dd <= min_dist) {
          min_dist = dd;
          if (c != cluster[i]) {
            changes = true;
          }
          cluster[i] = c;
        }
      }
    }

    for (int c = 0; c < k; c++) {
      nc[c] = 0;
      for (int j = 0; j < d; j++)
        center[c][j] = 0.;
    }
    for (int i = 0; i < n; i++) {
      auto c = cluster[i];
      nc[c]++;
      for (int j = 0; j < d; j++)
        center[c][j] += p[i * d + j];
    }
    for (int c = 0; c < k; c++)
      for (int j = 0; j < d; j++)
        center[c][j] /= nc[c];
    iter++;
  }

  int *ci = new int[k];
  for (int c = 0; c < k; c++)
    ci[c] = 0;
  double *p_perm = new double[n * d];
  double *labels_perm = new double[n];
  int row = 0;
  for (int c = 0; c < k; c++) {
    for (int j = 0; j < nc[c]; j++) {
      while (cluster[ci[c]] != c)
        ci[c]++;
      for (int l = 0; l < d; l++)
        p_perm[l + row * d] = p[l + ci[c] * d];
      labels_perm[row] = labels[ci[c]];
      ci[c]++;
      row++;
    }
  }
  copy(p_perm, p_perm + n * d, p);
  copy(labels_perm, labels_perm + n, labels);
  delete[] p_perm;
  delete[] labels_perm;
  delete[] ci;

  for (int i = 0; i < k; i++)
    delete[] center[i];
  delete[] center;
  delete[] cluster;
  delete[] ind_centers;
}

void recursive_2_means(double *p, int n, int d, int cluster_size,
                       HSSPartitionTree &tree, double *labels) {
  if (n < cluster_size)
    return;
  auto nc = new int[2];
  k_means(2, p, n, d, nc, labels);
  if (nc[0] == 0 || nc[1] == 0)
    return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_2_means(p, nc[0], d, cluster_size, tree.c[0], labels);
  recursive_2_means(p + nc[0] * d, nc[1], d, cluster_size, tree.c[1],
                    labels + nc[0]);
  delete[] nc;
}

void kd_partition(double *p, int n, int d, int *nc, double *labels,
                  int cluster_size) {
  // find coordinate of the most spread
  double *maxes = new double[d];
  double *mins = new double[d];

  for (int j = 0; j < d; ++j) {
    maxes[j] = p[j];
    mins[j] = p[j];
  }

  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      if (p[i * d + j] > maxes[j]) {
        maxes[j] = p[i * d + j];
      }
      if (p[i * d + j] > mins[j]) {
        mins[j] = p[i * d + j];
      }
    }
  }
  double max_var = maxes[0] - mins[0];
  int dim = 0;
  for (int j = 0; j < d; ++j) {
    if (maxes[j] - mins[j] > max_var) {
      max_var = maxes[j] - mins[j];
      dim = j;
    }
  }

  // find the mean
  double mean_value = 0.;
  for (int i = 0; i < n; ++i) {
    mean_value += p[i * d + dim];
  }
  mean_value /= n;

  // split the data
  int *cluster = new int[n];
  for (int i = 0; i < n; i++) {
    cluster[i] = 0;
  }
  nc[0] = 0;
  nc[1] = 0;
  for (int i = 0; i < n; ++i) {
    if (p[d * i + dim] > mean_value) {
      cluster[i] = 1;
      nc[1] += 1;
    } else {
      nc[0] += 1;
    }
  }

  // if clusters are too disbalanced, assign trivial clusters

  if ((nc[0] < cluster_size && nc[1] > 100 * cluster_size) ||
      (nc[1] < cluster_size && nc[0] > 100 * cluster_size)) {
    nc[0] = 0;
    nc[1] = 0;
    for (int i = 0; i < n; i++) {
      if (i <= n / 2) {
        cluster[i] = 0;
        nc[0] += 1;
      } else {
        cluster[i] = 1;
        nc[1] += 1;
      }
    }
  }
  // permute the data

  int *ci = new int[2];
  for (int c = 0; c < 2; c++)
    ci[c] = 0;
  double *p_perm = new double[n * d];
  double *labels_perm = new double[n];
  int row = 0;
  for (int c = 0; c < 2; c++) {
    for (int j = 0; j < nc[c]; j++) {
      while (cluster[ci[c]] != c)
        ci[c]++;
      for (int l = 0; l < d; l++)
        p_perm[l + row * d] = p[l + ci[c] * d];
      labels_perm[row] = labels[ci[c]];
      ci[c]++;
      row++;
    }
  }

  copy(p_perm, p_perm + n * d, p);
  copy(labels_perm, labels_perm + n, labels);
  delete[] p_perm;
  delete[] labels_perm;
  delete[] ci;
  delete[] maxes;
  delete[] mins;
  delete[] cluster;
}

void recursive_kd(double *p, int n, int d, int cluster_size,
                  HSSPartitionTree &tree, double *labels) {
  if (n < cluster_size)
    return;
  auto nc = new int[2];
  kd_partition(p, n, d, nc, labels, cluster_size);
  if (nc[0] == 0 || nc[1] == 0)
    return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_kd(p, nc[0], d, cluster_size, tree.c[0], labels);
  recursive_kd(p + nc[0] * d, nc[1], d, cluster_size, tree.c[1],
               labels + nc[0]);
  delete[] nc;
}

void pca_partition(double *p, int n, int d, int *nc, double *labels) {
  // find first pca direction
  int num = 0;
  double *W = new double[d];
  double *Z = new double[d * d];
  DenseMatrixWrapper<double> X(n, d, p, n);
  DenseMatrix<double> XtX(d, d);
  gemm(Trans::T, Trans::N, 1., X, X, 0., XtX);
  double *XtX_data = new double[d * d];
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      XtX_data[d * i + j] = XtX(i, j);
    }
  }
  dyevx('V', 'I', 'U', d, XtX_data, d, 1., 1., d, d, 1e-2, num, W, Z, d);
  // compute pca coordinates
  double *new_x_coord = new double[n];
  for (int i = 0; i < n; i++) {
    new_x_coord[i] = 0.;
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      new_x_coord[i] += p[i * d + j] * Z[j];
    }
  }

  // find the mean
  double mean_value = 0.;
  for (int i = 0; i < n; ++i) {
    mean_value += new_x_coord[i];
  }
  mean_value /= n;

  // split the data
  int *cluster = new int[n];
  for (int i = 0; i < n; i++) {
    cluster[i] = 0;
  }
  nc[0] = 0;
  nc[1] = 0;
  for (int i = 0; i < n; ++i) {
    if (new_x_coord[i] > mean_value) {
      cluster[i] = 1;
      nc[1] += 1;
    } else {
      nc[0] += 1;
    }
  }

  // permute the data

  int *ci = new int[2];
  for (int c = 0; c < 2; c++)
    ci[c] = 0;
  double *p_perm = new double[n * d];
  double *labels_perm = new double[n];
  int row = 0;
  for (int c = 0; c < 2; c++) {
    for (int j = 0; j < nc[c]; j++) {
      while (cluster[ci[c]] != c)
        ci[c]++;
      for (int l = 0; l < d; l++)
        p_perm[l + row * d] = p[l + ci[c] * d];
      labels_perm[row] = labels[ci[c]];
      ci[c]++;
      row++;
    }
  }

  copy(p_perm, p_perm + n * d, p);
  copy(labels_perm, labels_perm + n, labels);
  delete[] p_perm;
  delete[] labels_perm;
  delete[] ci;
  delete[] cluster;
  delete[] new_x_coord;
  delete[] W;
  delete[] Z;
}

void recursive_pca(double *p, int n, int d, int cluster_size,
                   HSSPartitionTree &tree, double *labels) {
  if (n < cluster_size)
    return;
  auto nc = new int[2];
  pca_partition(p, n, d, nc, labels);
  if (nc[0] == 0 || nc[1] == 0)
    return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_pca(p, nc[0], d, cluster_size, tree.c[0], labels);
  recursive_pca(p + nc[0] * d, nc[1], d, cluster_size, tree.c[1],
                labels + nc[0]);
  delete[] nc;
}

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

vector<double> write_from_file(string filename) {
  vector<double> data;
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

bool save_vector_to_binary_file( const vector<double> data, size_t length, const string& file_path )
{
    ofstream fileOut(file_path.c_str(), ios::binary | ios::out | ios::app);
    if ( !fileOut.is_open() )
        return false;
    fileOut.write((char*)&data[0], streamsize(length*sizeof(double)));
    fileOut.close();
    return true;
}

bool read_vector_from_binary_file( const vector<double> &data, size_t length, const string& file_path)
{
    ifstream is(file_path.c_str(), ios::binary | ios::in);
    if ( !is.is_open() )
        return false;
    is.read( (char*)&data[0], streamsize(length*sizeof(double)));
    is.close();
    return true;
}

void save_to_binary_file(string FILE, size_t n, int d){

  string bin_FILE = FILE + ".bin";
  cout << "Creating file: " << bin_FILE << endl;
  vector<double> data = write_from_file(FILE); // read

  // Debug: Print out data
  // cout << "To write:" << endl;
  // for (int i = 0; i < n*d; ++i){
  //   cout << data[i] << " ";
  //   if ((i+1)%d == 0)
  //     cout << endl;
  // }

  save_vector_to_binary_file(data, n*d, bin_FILE);

  // vector<double> read_data(n*d);
  // read_vector_from_binary_file(read_data, n*d, bin_FILE);

  // cout << "Read:" << endl;
  // for (int i = 0; i < 10*d; ++i){
  //   cout << read_data[i] << " ";
  //   if ((i+1)%d == 0)
  //     cout << endl;
  // }

}

int main(int argc, char *argv[]) {
  string filename("smalltest.dat");
  int d = 2;
  string reorder("natural");
  double h = 3.;
  double lambda = 1.;
  int kernel = 1; // Gaussian=1, Laplace=2
  double total_time;
  string mode("valid");

  cout << "# usage: ./KernelRegression_mf file d h lambda "
          "kern(1=Gau,2=Lapl) "
          "reorder(nat, 2mea, kd, pca) mode(valid, test)"
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

  cout << "# data dimension = " << d << endl;
  cout << "# kernel h = " << h << endl;
  cout << "# lambda = " << lambda << endl;
  cout << "# kernel type = " << ((kernel == 1) ? "Gauss" : "Laplace") << endl;
  cout << "# reordering/clustering = " << reorder << endl;
  cout << "# validation/test = " << mode << endl;
  
  TaskTimer::t_begin = GET_TIME_NOW();
  TaskTimer timer(string("compression"), 1);

  HSSOptions<double> hss_opts;
  hss_opts.set_verbose(true);
  hss_opts.set_from_command_line(argc, argv);

  cout << "# Reading data..." << endl;
  timer.start();

  string data_train_dat_FILE = filename + "_train.csv";
  string data_train_lab_FILE = filename + "_train_label.csv";
  string data_test_dat_FILE  = filename + "_" + mode + ".csv";
  string data_test_lab_FILE  = filename + "_" + mode + "_label.csv";

  // // // Read from csv file
  vector<double> data_train       = write_from_file(data_train_dat_FILE);
  vector<double> data_train_label = write_from_file(data_train_lab_FILE);
  vector<double> data_test        = write_from_file(data_test_dat_FILE);
  vector<double> data_test_label  = write_from_file(data_test_lab_FILE);

  // // // Save csv file to binary file
  // save_to_binary_file(data_train_dat_FILE, count_lines(data_train_dat_FILE), d); //data
  // save_to_binary_file(data_train_lab_FILE, count_lines(data_train_lab_FILE), 1); //labels
  // save_to_binary_file(data_test_dat_FILE, count_lines(data_test_dat_FILE), d); //data
  // save_to_binary_file(data_test_lab_FILE, count_lines(data_test_lab_FILE), 1); //labels

  // Size of training and testing datasets
  // size_t ntrain = count_lines(data_train_lab_FILE);
  // size_t ntest  = count_lines(data_test_lab_FILE);
  // cout << "ntrain = " << ntrain << endl;
  // cout << "ntest = " << ntest << endl;

  // vector<double> data_train(ntrain*d);
  // vector<double> data_train_label(ntrain*1);
  // vector<double> data_test(ntest*d);
  // vector<double> data_test_label(ntest*1);

  // // Read from binary file
  // read_vector_from_binary_file(data_train,        ntrain*d, data_train_dat_FILE+".bin");
  // read_vector_from_binary_file(data_train_label,  ntrain*1, data_train_lab_FILE+".bin");
  // read_vector_from_binary_file(data_test,         ntest*d,  data_test_dat_FILE+".bin");
  // read_vector_from_binary_file(data_test_label,   ntest*1,  data_test_lab_FILE+".bin");

  cout << "# Reading took " << timer.elapsed() << endl;

  // New normalization
  // size_t training   = 10000;
  // size_t testing    = 1000;
  // size_t validation = 0; 

  // vector<double> data_train; 
  // vector<double> data_train_label;
  
  // vector<double> data_test;
  // vector<double> data_test_label;
  
  // vector<double> data_valid;
  // vector<double> data_valid_label;

  // string filename_DATA   = "/Users/gichavez/Desktop/susy10k/susy_10Kn_train.csv";
  // string filename_LABELS = "/Users/gichavez/Desktop/susy10k/susy_10Kn_train_label.csv";

  // data_read_normalize_from_file(filename_DATA, d, training, validation, testing, data_train, data_valid, data_test);
  // labels_read_from_file(filename_LABELS, d, training, validation, testing, data_train_label, data_valid_label, data_test_label);

  // cout << "# data_train.size() = " << data_train.size()/d << endl;
  // cout << "# data_valid.size() = " << data_valid.size()/d << endl;
  // cout << "# data_test.size() = " << data_test.size()/d << endl;

  // cout << "# data_train_label.size() = " << data_train_label.size() << endl;
  // cout << "# data_valid_label.size() = " << data_valid_label.size() << endl;
  // cout << "# data_test_label.size() = " << data_test_label.size() << endl;
  // exit(0);
  // New normalization

  int n = data_train.size() / d;
  int m = data_test.size() / d;
  cout << "# matrix size = " << n << " x " << d << endl;

  HSSPartitionTree cluster_tree;
  cluster_tree.size = n;
  int cluster_size = hss_opts.leaf_size();

  if (reorder == "2means") {
    recursive_2_means(data_train.data(), n, d, cluster_size, cluster_tree,
                      data_train_label.data());
  } else if (reorder == "kd") {
    recursive_kd(data_train.data(), n, d, cluster_size, cluster_tree,
                 data_train_label.data());
  } else if (reorder == "pca") {
    recursive_pca(data_train.data(), n, d, cluster_size, cluster_tree,
                  data_train_label.data());
  }

  cout << "starting HSS compression .. " << endl;

  HSSMatrix<double> K;
  if (reorder != "natural")
    K = HSSMatrix<double>(cluster_tree, hss_opts);
  else{
    K = HSSMatrix<double>(n, n, hss_opts);
  }

  timer.start();
  //  K.compress(Kdense, hss_opts);
  Kernel kernel_matrix(data_train, d, h, lambda);


  // TODO get from ANN search
  DenseMatrix<double> ann, scores;
  K.compress_ann(ann, scores, kernel_matrix, hss_opts);
  //K.compress(kernel_matrix, kernel_matrix, hss_opts);


  cout << "# compression time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();

  if (K.is_compressed()) {
    cout << "# created K matrix of dimension " << K.rows() << " x " << K.cols()
         << " with " << K.levels() << " levels" << endl;
    cout << "# compression succeeded!" << endl;
  } else {
    cout << "# compression failed!!!!!!!!" << endl;
    return 1;
  }
  cout << "# rank(K) = " << K.rank() << endl;
  cout << "# memory(K) = " << K.memory() / 1e6 << " MB " << endl;

  cout << "factorization start" << endl;
  timer.start();
  auto ULV = K.factor();
  cout << "# factorization time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();

  DenseMatrix<double> B(n, 1, &data_train_label[0], n);
  DenseMatrix<double> weights(B);

  cout << "solution start" << endl;
  timer.start();
  K.solve(ULV, weights);
  cout << "# solve time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();
  cout << "# total time: " << total_time << endl;

  auto Bcheck = K.apply(weights);

  Bcheck.scaled_add(-1., B);
  cout << "# relative error = ||B-H*(H\\B)||_F/||B||_F = "
       << Bcheck.normF() / B.normF() << endl;

  cout << "# Starting prediction step " << endl;
  timer.start();

  double *prediction = new double[m];
  for (int i = 0; i < m; ++i) {
    prediction[i] = 0;
  }

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

  for (int i = 0; i < m; ++i) {
    prediction[i] = ((prediction[i] > 0) ? 1. : -1.);
  }

  // compute accuracy score of prediction
  double incorrect_quant = 0;
  for (int i = 0; i < m; ++i) {
    double a = (prediction[i] - data_test_label[i]) / 2;
    incorrect_quant += (a > 0 ? a : -a);
  }

  cout << "# Prediction took " << timer.elapsed() << endl;
  cout << "# prediction score: " << ((m - incorrect_quant) / m) * 100 << "%"
       << endl << endl;

  return 0;
}
