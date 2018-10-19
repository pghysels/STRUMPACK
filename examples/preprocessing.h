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



#endif
