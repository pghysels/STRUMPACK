#ifndef preprocessing_hpp
#define preprocessing_hpp

#include <iostream>  // Input/Output streams
#include <vector>    // STD Dynamic vectors
#include <fstream>   // Open file
#include <sstream>   // Open file
#include <cmath>     // Common math, pow
#include <algorithm> // sort
#include <numeric>   // std::iota
#include <random>
#include <string>
#include <chrono>
#include <ctime>

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

//------STANDARD VECTOR ALGEBRA-----------------
inline double dist(const double* x, const double* y, int d)  {
  double k = 0.;
   for (int i = 0; i < d; i++) k += pow(x[i] - y[i], 2.);
   return sqrt(k);
}

inline double norm(const double* x, int d)  {
  double k = 0.;
  for (int i = 0; i < d; i++) k += pow(x[i], 2.);
  return sqrt(k);
}

inline double dot_product(const double* x, const double* y, int d)  {
  double k = 0.;
  for (int i = 0; i < d; i++) k += x[i]*y[i];
  return k;
}

inline double dist2(const double *x, const double *y, int d) {
  double k = 0.;
  for (int i = 0; i < d; i++)
    k += pow(x[i] - y[i], 2.);
  return k;
}

inline double norm1(const double *x, const double *y, int d) {
  double k = 0.;
  for (int i = 0; i < d; i++)
    k += fabs(x[i] - y[i]);
  return k;
}

inline double Gauss_kernel
(const double *x, const double *y, int d, double h) {
  return exp(-dist2(x, y, d) / (2. * h * h));
}

inline double Laplace_kernel
(const double *x, const double *y, int d, double h) {
  return exp(-norm1(x, y, d) / h);
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

#endif
