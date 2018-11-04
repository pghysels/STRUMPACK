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
inline real_t dist(const real_t* x, const real_t* y, int d)  {
  real_t k = 0.;
   for (int i = 0; i < d; i++) k += pow(x[i] - y[i], 2.);
   return sqrt(k);
}

inline real_t norm(const real_t* x, int d)  {
  real_t k = 0.;
  for (int i = 0; i < d; i++) k += pow(x[i], 2.);
  return sqrt(k);
}

inline real_t dot_product(const real_t* x, const real_t* y, int d)  {
  real_t k = 0.;
  for (int i = 0; i < d; i++) k += x[i]*y[i];
  return k;
}

inline real_t dist2(const real_t *x, const real_t *y, int d) {
  real_t k = 0.;
  for (int i = 0; i < d; i++)
    k += pow(x[i] - y[i], 2.);
  return k;
}

inline real_t norm1(const real_t *x, const real_t *y, int d) {
  real_t k = 0.;
  for (int i = 0; i < d; i++)
    k += fabs(x[i] - y[i]);
  return k;
}

inline real_t Gauss_kernel
(const real_t *x, const real_t *y, int d, real_t h) {
  return exp(-dist2(x, y, d) / (2. * h * h));
}

inline real_t Laplace_kernel
(const real_t *x, const real_t *y, int d, real_t h) {
  return exp(-norm1(x, y, d) / h);
}



#endif
