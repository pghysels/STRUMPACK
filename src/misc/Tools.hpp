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
#include <numeric>
#include <algorithm>
#include "StrumpackParameters.hpp"
#include "dense/BLASLAPACKWrapper.hpp"

namespace strumpack {

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

//--------------SORT PAIRS---------------------------

  template <typename T>
  std::vector<std::size_t> find_sort_permutation(const std::vector<T>& vec)
  {
     std::vector<std::size_t> order(vec.size());
     std::iota(order.begin(), order.end(), 0);
     std::sort(order.begin(), order.end(),
        [&](std::size_t i, std::size_t j){ return vec[i] < vec[j]; });
     return order;
  }

  template <typename T>
  std::vector<T> apply_permutation(
    const std::vector<T>& vec,
    const std::vector<std::size_t>& p)
  {
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
        [&](std::size_t i){ return vec[i]; });
    return sorted_vec;
  }


} // end namespace strumpack

#endif // TOOLS_H
