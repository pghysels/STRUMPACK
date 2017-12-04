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
 * works, and perform publicly and display publicly.  Beginning five
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
#ifndef TOOLS_H
#define TOOLS_H

#include <vector>
#include <iomanip>
#include "StrumpackParameters.hpp"
#include "dense/BLASLAPACKWrapper.hpp"

namespace strumpack {

  // TODO implement these things in DenseMatrix??

  // compute z[i] = x[i] * y[i] for three vectors with n elements
  template<typename scalar_t,typename integer_t> inline void
  x_mult_y(integer_t n, const scalar_t* x, const scalar_t* y, scalar_t* z) {
#pragma omp parallel for
    for (int i=0; i<n; ++i)
      z[i] = x[i] * y[i];
    STRUMPACK_FLOPS((is_complex<scalar_t>()?6:1)*
                    static_cast<long long int>(n));
    STRUMPACK_BYTES(sizeof(scalar_t)*static_cast<long long int>(n)*4);
  }

  // compute x[i] *= y[i] for two vectors with n elements
  template<typename scalar_t,typename integer_t> inline void
  x_mult_y(integer_t n, scalar_t* x, const scalar_t* y) {
#pragma omp parallel for
    for (int i=0; i<n; ++i)
      x[i] *= y[i];
    STRUMPACK_FLOPS((is_complex<scalar_t>()?6:1)*
                    static_cast<long long int>(n));
    STRUMPACK_BYTES(sizeof(scalar_t)*static_cast<long long int>(n)*3);
  }

  /** compute x[i] /= y[i] for two vectors with n elements */
  template<typename scalar_t,typename integer_t> inline void
  x_div_y(integer_t n, scalar_t* x, const scalar_t* y) {
#pragma omp parallel for
    for (int i=0; i<n; ++i)
      x[i] /= y[i];
    STRUMPACK_FLOPS((is_complex<scalar_t>()?11:1)*
                    static_cast<long long int>(n));
    STRUMPACK_BYTES(sizeof(scalar_t)*static_cast<long long int>(n)*3);
  }

  template<typename scalar_t,typename integer_t> inline void mat_print
  (std::string name, const scalar_t* F,
   integer_t rows, integer_t cols, integer_t ld) {
    std::cout << name << " = [" << std::endl;
    for (integer_t r=0; r<rows; r++) {
      for (integer_t c=0; c<cols; c++)
        std::cout << F[r+c*ld] << " ";
      std::cout << std::endl;
    }
    std::cout << "];" << std::endl;
  }

  template<typename scalar_t,typename integer_t> inline void
  mat_zero(integer_t m, integer_t n, scalar_t* a, integer_t lda) {
    blas::laset('A', m, n, scalar_t(0.), scalar_t(0.), a, lda);
  }

  // b = a + beta * b
  template<typename scalar_t,typename integer_t> inline void mat_add
  (integer_t m, integer_t n, const scalar_t* a, integer_t lda,
   scalar_t beta, scalar_t* b, integer_t ldb) {
    if (beta == scalar_t(1.)) {
      for (integer_t i=0; i<n; i++)
        for (integer_t j=0; j<m; j++)
          b[j+i*ldb] += a[j+i*lda];
      STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*static_cast<long long int>(double(m)*double(n)));
    } else if (beta == scalar_t(0.)) blas::lacpy('A', m, n, a, lda, b, ldb);
    else {
      for (integer_t i=0; i<n; i++)
        for (integer_t j=0; j<m; j++)
          b[j+i*ldb] = a[j+i*lda] + beta * b[j+i*ldb];
      STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*static_cast<long long int>(double(m)*double(n)*2.));
    }
    STRUMPACK_BYTES(sizeof(scalar_t)*static_cast<long long int>(double(m)*double(n)*3.));
  }

  template<typename scalar_t,typename integer_t> inline void permute_vector
  (integer_t n, scalar_t* vect, const integer_t* order, bool forward) {
    // TODO use DenseMatrix member functions!!
    auto tmp = new scalar_t[n];
    if (forward) {
#pragma omp parallel for
      for (integer_t k=0; k<n; k++)
        tmp[k] = vect[order[k]];
    } else {
#pragma omp parallel for
      for (integer_t k=0; k<n; k++)
        tmp[order[k]] = vect[k]; // I don't think parallel for helps here
    }
    std::copy(tmp, tmp+n, vect);
    delete[] tmp;
    STRUMPACK_BYTES(sizeof(scalar_t)*static_cast<long long int>(n)*2+sizeof(int)*static_cast<long long int>(n));
  }

  template<typename scalar_t,typename integer_t> inline void permute_vector
  (integer_t n, scalar_t* vect,
   const std::vector<integer_t>& order, bool forward) {
    permute_vector(n, vect, order.data(), forward);
  }

  template<typename integer_t> inline void merge_indices
  (integer_t* v1, integer_t& l1, integer_t l1max,
   integer_t* v2, integer_t l2) {
    integer_t* result = v1;
    v1 += std::min(l1max,l1+l2) - l1;
    for (integer_t i=l1-1; i>=0; --i) v1[i] = result[i];
    integer_t* e1 = v1+l1; integer_t* e2 = v2+l2;
    l1 = 0;
    while (true) {
      if (v1 == e1) { for (;v2!=e2; v2++) result[l1++] = *v2; return; }
      if (v2 == e2) { for (;v1!=e1; v1++) result[l1++] = *v1; return; }
      if (*v1 < *v2) { result[l1] = *v1; ++v1; }
      else if (*v2 < *v1) { result[l1] = *v2; ++v2; }
      else { result[l1] = *v1; ++v1; ++v2; }
      l1++;
    }
  }

  // this sorts both indices and values at the same time
  template<typename scalar_t,typename integer_t> void
  sort_indices_values
  (integer_t *ind, scalar_t *val, integer_t begin, integer_t end) {
    if (end > begin) {
      integer_t left = begin + 1;
      integer_t right = end;
      integer_t pivot = (begin+(end-begin)/2);
      std::swap(ind[begin], ind[pivot]);
      std::swap(val[begin], val[pivot]);
      pivot = ind[begin];

      while (left < right) {
        if (ind[left] <= pivot)
          left++;
        else {
          while (left<--right && ind[right]>=pivot) {}
          std::swap(ind[left], ind[right]);
          std::swap(val[left], val[right]);
        }
      }
      left--;
      std::swap(ind[begin], ind[left]);
      std::swap(val[begin], val[left]);
      sort_indices_values<scalar_t>(ind, val, begin, left);
      sort_indices_values<scalar_t>(ind, val, right, end);
    }
  }

  template<class T> std::string number_format_with_commas(T value) {
    struct Numpunct: public std::numpunct<char>{
    protected:
      virtual char do_thousands_sep() const{return ',';}
      virtual std::string do_grouping() const{return "\03";}
    };
    std::stringstream ss;
    ss.imbue({std::locale(), new Numpunct});
    ss << std::setprecision(2) << std::fixed << value;
    return ss.str();
  }

} // end namespace strumpack

#endif // TOOLS_H
