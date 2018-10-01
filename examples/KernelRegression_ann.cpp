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
#include "find_ann.hpp"

#include "FileManipulation.h"
#include <fstream>

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

//----------------------------------------------------------------------------
//--------PREPROCESSING-------------------------------------------------------
//----------------------------------------------------------------------------
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

const int kmeans_max_it = 100;
double r;
// random_device rd;
mt19937 generator(1);

inline double dist2(double *x, double *y, int d) {
  double k = 0.;
  for (int i = 0; i < d; i++)
    k += pow(x[i] - y[i], 2.);
  return k;
}

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

//----------------------------------------------------------------------------
//--------CLASS KERNEL--------------------------------------------------------
//----------------------------------------------------------------------------

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

//----------------------------------------------------------------------------
//--------FILE MANIPULATIONS--------------------------------------------------
//----------------------------------------------------------------------------

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

bool save_vector_to_binary_file( const vector<double> data, size_t length,
const string& file_path ) {
    ofstream fileOut(file_path.c_str(), ios::binary | ios::out | ios::app);
    if ( !fileOut.is_open() )
        return false;
    fileOut.write((char*)&data[0], streamsize(length*sizeof(double)));
    fileOut.close();
    return true;
}

bool read_vector_from_binary_file( const vector<double> &data, size_t length,
const string& file_path) {
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


//----------------------------------------------------------------------------
//--------ANN SEARCH----------------------------------------------------------
//----------------------------------------------------------------------------

// //-------FIND APPROXIMATE NEAREST NEIGHBORS FROM PROJECTION TREE---

// 1. CONSTRUCT THE TREE
// leaves - list of all indices such that 
// leaves[leaf_sizes[i]]...leaves[leaf_sizes[i+1]] 
// belong to i-th leaf of the projection tree
// gauss_id and gaussian samples - for the fixed samples option 
void construct_projection_tree (vector<double> &data, int n, int d,
int min_leaf_size, vector<int> &cur_indices, int start, int cur_node_size,
vector<int> &leaves, vector<int> &leaf_sizes, mt19937 &generator) {
    if (cur_node_size < min_leaf_size) 
    {
        int prev_size = leaf_sizes.back();
        leaf_sizes.push_back(cur_node_size + prev_size);  
        for (int i = 0; i < cur_node_size; i++)
        {
            leaves.push_back(cur_indices[start + i]);
        }
        return;
    }

    // choose random direction
    vector<double> direction_vector(d);
    normal_distribution<double> normal_distr(0.0,1.0);
    for (int i = 0; i < d; i++) 
    {
        direction_vector[i] = normal_distr(generator);
        //  option for fixed direction samples
        //  direction_vector[i] = gaussian_samples[gauss_id*d + i];
        //  cout << direction_vector[i] << ' ';
    }
    // gauss_id++;
    double dir_vector_norm = norm(&direction_vector[0], d);
    for (int i = 0; i < d; i++) 
    {
        direction_vector[i] /= dir_vector_norm;
    }

    // choose margin (MARGIN IS CURRENTLY NOT USED)
    //double diameter = estimate_set_diameter(data, n, d, cur_indices, cur_node_size, generator);
    //uniform_real_distribution<double> uniform_on_segment(-1, 1);
    //double delta = uniform_on_segment(generator);
    //double margin = delta*6*diameter/sqrt(d);


    // find relative coordinates
    vector<double> relative_coordinates(cur_node_size, 0.0);
    for (int i = 0; i < cur_node_size; i++)
    {
        relative_coordinates[i] = dot_product(&data[cur_indices[start + i] * d], &direction_vector[0], d);
    }
  
    // median split
    vector<int> idx(cur_node_size);
    iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return relative_coordinates[a] < relative_coordinates[b];   
    });
    vector<int> cur_indices_sorted(cur_node_size, 0);
    for (int i = 0; i < cur_node_size; i++)
    {
      cur_indices_sorted[i] = cur_indices[start + idx[i]];
    }
    for (int i = start; i < start + cur_node_size; i++)
    {
      cur_indices[i] = cur_indices_sorted[i - start];
    }
    
    int half_size =  (int)cur_node_size / 2;


    construct_projection_tree(data, n, d, min_leaf_size, 
                               cur_indices, start, half_size, 
                               leaves, leaf_sizes, generator);
                               //, gauss_id, gaussian_samples);

    construct_projection_tree(data, n, d, min_leaf_size, 
                               cur_indices, start + half_size, cur_node_size - half_size, 
                               leaves, leaf_sizes, generator);
                               //, gauss_id, gaussian_samples);

}

// 2. FIND CLOSEST POINTS INSIDE LEAVES
// find ann_number exact neighbors for every point among the points
// within its leaf (in randomized projection tree)
void find_neibs_in_tree(vector<double> &data, int n, int d, int ann_number,
vector<int> &leaves, vector<int> &leaf_sizes, vector<int> &neighbors,
vector<double> &neighbor_scores) {
    for (int leaf = 0; leaf < leaf_sizes.size() - 1; leaf++) 
    {
        // initialize size and content of the current leaf
        int cur_leaf_size = leaf_sizes[leaf+1] - leaf_sizes[leaf];
        vector<int> index_subset(cur_leaf_size, 0); // list of indices in the current leaf
        for (int i = 0; i < index_subset.size(); i++)
        {
            index_subset[i] = leaves[leaf_sizes[leaf] + i];
        }

        vector<vector<double>> leaf_dists;
        find_distance_matrix(data, d, index_subset, leaf_dists);
        
        // record ann_number closest points in each leaf to neighbors
        for (int i = 0; i < cur_leaf_size; i++) 
        {
            vector<int> idx(cur_leaf_size);
            iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int i1, int i2) {
                return leaf_dists[i][i1] < leaf_dists[i][i2];
             });
            for (int j = 0; j < ann_number; j++)
            {
                int neibid = leaves[leaf_sizes[leaf] + idx[j]];
                double neibscore = leaf_dists[i][idx[j]];
                neighbors[index_subset[i] * ann_number + j] = neibid;
                neighbor_scores[index_subset[i] * ann_number + j] = neibscore;
            }
        }
    }
}

// 3. FIND ANN IN ONE TREE SAMPLE
void find_ann_candidates(vector<double> &data, int n, int d, int ann_number, 
vector<int> &neighbors,vector<double> &neighbor_scores, mt19937 &generator) {

    int min_leaf_size = 6*ann_number;
    
    vector<int> leaves;
    vector<int> leaf_sizes;
    leaf_sizes.push_back(0);

    int cur_node_size = n;
    int start = 0;
   //  int gauss_id = 0;
    vector<int> cur_indices(cur_node_size);
    iota(cur_indices.begin(), cur_indices.end(), 0);

     construct_projection_tree(data, n, d, min_leaf_size, 
                               cur_indices, start, cur_node_size, 
                               leaves, leaf_sizes, generator);
                               // , gauss_id, gaussian_samples);

    find_neibs_in_tree(data, n, d, ann_number, leaves, leaf_sizes, neighbors, neighbor_scores);
}

//---------------CHOOSE BEST NEIGHBORS FROM TWO TREE SAMPLES------------------

// take closest neighbors from neighbors and new_neighbors,
// write them to neighbors and their scores to neighbor_scores
void choose_best_neighbors(vector<int> &neighbors,
vector<double> &neighbor_scores, vector<int> &new_neighbors,
vector<double> &new_neighbor_scores, int ann_number) {
    for (int vertex = 0; vertex < neighbors.size(); vertex = vertex + ann_number)
    {
        vector<int> cur_neighbors(ann_number, 0);
        vector<double> cur_neighbor_scores(ann_number, 0.0);
        int iter1 = 0;
        int iter2 = 0;
        int cur = 0;
        while ((iter1 < ann_number) && (iter2 < ann_number) && (cur < ann_number))
        {
            if (neighbor_scores[vertex+iter1] > new_neighbor_scores[vertex+iter2])
            {
                cur_neighbors[cur] = new_neighbors[vertex+iter2];
                cur_neighbor_scores[cur] = new_neighbor_scores[vertex+iter2];
                iter2++;
            } else {
                cur_neighbors[cur] = neighbors[vertex+iter1];
                cur_neighbor_scores[cur] = neighbor_scores[vertex+iter1];
                if (neighbors[vertex+iter1] == new_neighbors[vertex+iter2])
                {
                    iter2++;
                }
                iter1++;
            }
            cur++;
        }
        while (cur < ann_number)
        {
            if (iter1 == ann_number)
            {
                cur_neighbors[cur] = new_neighbors[vertex+iter2];
                cur_neighbor_scores[cur] = new_neighbor_scores[vertex+iter2];
                iter2++;
            }
            else 
            {
                cur_neighbors[cur] = neighbors[vertex+iter1];
                cur_neighbor_scores[cur] = neighbor_scores[vertex+iter1];
                iter1++;
            }
            cur++;
        }

        for(int i = 0; i < ann_number; i++)
        {
            neighbors[vertex+i] = cur_neighbors[i];
            neighbor_scores[vertex+i] = cur_neighbor_scores[i];
        }
    }

}

//----------------QUALITY CHECK WITH TRUE NEIGHBORS----------------------------
void find_true_nn(vector<double> &data, int n, int d, int ann_number,
vector<int> &n_neighbors, vector<double> &n_neighbor_scores) {
         // create full distance matrix
        vector<vector<double>> all_dists;
        vector<int> all_ids(n); // index subset = everything
        iota(all_ids.begin(), all_ids.end(), 0);
        find_distance_matrix(data, d, all_ids, all_dists);

       // record ann_number closest points in each leaf to neighbors
        for (int i = 0; i < n; i++) 
        {
            vector<int> idx(n);
            iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int i1, int i2) {
                return all_dists[i][i1] < all_dists[i][i2];
             });
            for (int j = 0; j < ann_number; j++)
            {
                n_neighbors[i * ann_number + j] = idx[j];
                n_neighbor_scores[i * ann_number + j] = all_dists[i][idx[j]];
            }
        }
}

// quality = average fraction of ann_number approximate neighbors (neighbors),
//  which are within the closest ann_number of true neighbors (n_neighbors);
// average is taken over all data points
double check_quality(vector<double> &data, int n, int d, int ann_number,
vector<int> &neighbors)
{
    vector<int> n_neighbors(n*ann_number, 0);
    vector<double> n_neighbor_scores(n*ann_number, 0.0);
    auto start_nn = chrono::system_clock::now();
    find_true_nn(data, n, d, ann_number, n_neighbors, n_neighbor_scores);
    auto end_nn = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds_nn = end_nn-start_nn;
    cout << "elapsed time for exact neighbor search: " 
         << elapsed_seconds_nn.count() << " sec" << endl;

    vector<double> quality_vec;
    for (int i = 0; i < n; i++) 
    {
        int iter1 = ann_number*i;
        int iter2 = ann_number*i;
        int num_nei_found = 0;
        while (iter2 < ann_number*(i+1))
        {
            if (neighbors[iter1] == n_neighbors[iter2])
            {
                iter1++;
                iter2++;
                num_nei_found++;
            }
            else
            {
                iter2++;
            }
        }
       quality_vec.push_back((double)num_nei_found/ann_number);
    }
    cout << endl;
  
    double ann_quality = 0.0;
    for (int i = 0; i < quality_vec.size(); i++)
    {
        ann_quality += quality_vec[i];
    }
    return (double)ann_quality/quality_vec.size();
}

//------------ITERATE OVER SEVERAL PROJECTION TREES TO FIND ANN----------------
void find_approximate_neighbors
(vector<double> &data, int n, int d, int num_iters, int ann_number, 
vector<int> &neighbors, vector<double> &neighbor_scores) {
    
  find_ann_candidates(data, n, d, ann_number, neighbors, neighbor_scores,
                      generator);

  for (int iter = 1; iter < num_iters; iter++)
  {
      vector<int> new_neighbors(n*ann_number, 0);
      vector<double> new_neighbor_scores(n*ann_number, 0.0);   
      find_ann_candidates(data, n, d, ann_number, new_neighbors, new_neighbor_scores, generator);
      choose_best_neighbors(neighbors, neighbor_scores, new_neighbors, new_neighbor_scores, ann_number);
      // cout << "iter " << iter << " done" << endl;       
  }
}


//----------------------------------------------------------------------------
//--------MAIN----------------------------------------------------------------
//----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  string filename("smalltest.dat");
  int d = 2;
  string reorder("natural");
  double h = 3.;
  double lambda = 1.;
  int kernel = 1; // Gaussian=1, Laplace=2
  double total_time;
  string mode("valid");

  cout << endl;
  cout << "# usage: ./KernelRegression_mf file d h lambda "
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

  // // // Read from csv file
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
                      data_train_label.data());
  } else if (reorder == "kd") {
    recursive_kd(data_train.data(), n, d, cluster_size, cluster_tree,
                 data_train_label.data());
  } else if (reorder == "pca") {
    recursive_pca(data_train.data(), n, d, cluster_size, cluster_tree,
                  data_train_label.data());
  }

  cout << endl;
  cout << "# Starting HSS compression... " << endl;

  HSSMatrix<double> K;
  if (reorder != "natural")
    K = HSSMatrix<double>(cluster_tree, hss_opts);
  else{
    K = HSSMatrix<double>(n, n, hss_opts);
  }


  // FIND ANN ------------------------------------------------
  int ann_number = 64;
  vector<int> neighbors(n*ann_number, 0);
  vector<double> neighbor_scores(n*ann_number, 0.0);

  int num_iters = 5;
  auto start_ann = chrono::system_clock::now();
  find_approximate_neighbors(data_train, n, d, num_iters, ann_number,
  neighbors, neighbor_scores);
  auto end_ann = chrono::system_clock::now();
  chrono::duration<double> elapsed_seconds_ann = end_ann-start_ann;
  cout << "# Time for approximate neighbor search: " 
       << elapsed_seconds_ann.count() << " sec" << endl;
  

  vector<double> neighbors_d;
  for (int i = 0; i < ann_number*n; i++)
  {
    neighbors_d.push_back((double)neighbors[i]);
  }
  DenseMatrixWrapper<double> ann(ann_number, n, &neighbors_d[0], ann_number);
  DenseMatrixWrapper<double> scores(ann_number, n, &neighbor_scores[0],
                                    ann_number);

  timer.start();

  // Print scores and ann
  // for (int i = 0; i < ann_number; i++) {
  //   cout << ann(i, 0) << ' ';
  // }
  // cout << endl;

  // for (int i = 0; i < ann_number; i++) {
  // cout << scores(i, 0) << ' ';
  // }
  // cout << endl;

  Kernel kernel_matrix(data_train, d, h, lambda);
  //cout << "# rank(K) = " << kernel_matrix.normF() << endl;

  // ---CHOOSE SEARCH OPTION ANN/STANDARD--------------
  K.compress_ann(ann, scores, kernel_matrix, hss_opts);
  //  K.compress(kernel_matrix, kernel_matrix, hss_opts);


  cout << "### compression time = " << timer.elapsed() << " ###" <<endl;
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
  // K.print_info(); // Multiple rank information

  auto Ktest = K.dense();
  Ktest.scaled_add(-1., Kdense);
  cout << "# compression error = ||Kdense-K*I||_F/||Kdense||_F = "
       << Ktest.normF() / Kdense.normF() << endl;

  cout << endl;
  cout << "# Factorization start" << endl;
  timer.start();
  auto ULV = K.factor();
  cout << "# factorization time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();

  DenseMatrix<double> B(n, 1, &data_train_label[0], n);
  DenseMatrix<double> weights(B);

  cout << endl;
  cout << "# Solution start..." << endl;
  timer.start();
  K.solve(ULV, weights);
  cout << "# solve time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();
  cout << "# total time: " << total_time << endl;

  //------generate random vector----
  vector<double> sample_vector(n);
  normal_distribution<double> normal_distr(0.0,1.0);
  for (int i = 0; i < n; i++) 
  {
    sample_vector[i] = normal_distr(generator);
  }
  double sample_norm = norm(&sample_vector[0], n);
  for (int i = 0; i < n; i++) 
  {
    sample_vector[i] /= sample_norm;
  }
  
  DenseMatrixWrapper<double> sample_v(n, 1, &sample_vector[0], n);

  // for (int i = 0; i < d; i++) 
  // {
  // cout << sample_v(i, 0) << endl;
  // }

  DenseMatrix<double> sample_rhs(n, 1);
  gemm(Trans::N, Trans::N, 1., Kdense, sample_v, 0., sample_rhs);
  
  K.solve(ULV, sample_rhs);

 // auto Bcheck = K.apply(weights);
  sample_v.scaled_add(-1., sample_rhs);

  //cout << "weights_F =  "
  //    << weights.normF() << endl;
  
  //cout << "# relative error = ||B-H*(H\\B)||_F = "
  //    << Bcheck.normF() << endl;

  // for (int i = 0; i < d; i++) 
  // {
  //   cout << sample_v(i, 0) << endl;
  // }
  cout << "# solution error = "
        << sample_v.normF() << endl;

  cout << endl;
  cout << "# Prediction start..." << endl;
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
