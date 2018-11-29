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
/*! \file Metrics.hpp
 * \brief Definitions of distance metrics.
 */
#ifndef STRUMPACK_METRICS_HPP
#define STRUMPACK_METRICS_HPP

#include <cmath>
#include "dense/BLASLAPACKWrapper.hpp"

namespace strumpack {

  /**
   * Evaluate the Euclidean distance squared between two points x and
   * y.
   *
   * \tparam scalar_t datatype of the points
   * \tparam real_t real type corresponding to scalar_t
   * \param d dimension of the points
   * \param x pointer to first point (stored with stride 1)
   * \param y pointer to second point (stored with stride 1)
   */
  template<typename scalar_t,
           typename real_t=typename RealType<scalar_t>::value_type>
  real_t Euclidean_distance_squared
  (std::size_t d, const scalar_t* x, const scalar_t* y) {
    real_t k(0.);
    for (std::size_t i=0; i<d; i++) {
      auto xy = x[i]-y[i];
      k += xy * xy;
    }
    return k;
  }

  /**
   * Evaluate the Euclidean distance between two points x and y.
   *
   * \tparam scalar_t datatype of the points
   * \tparam real_t real type corresponding to scalar_t
   * \param d dimension of the points
   * \param x pointer to first point (stored with stride 1)
   * \param y pointer to second point (stored with stride 1)
   */
  template<typename scalar_t,
           typename real_t=typename RealType<scalar_t>::value_type>
  real_t Euclidean_distance
  (std::size_t d, const scalar_t* x, const scalar_t* y) {
    return std::sqrt(Euclidean_distance_squared(d, x, y));
  }


  /**
   * Evaluate the 1-norm distance between two points x and y.
   *
   * \tparam scalar_t datatype of the points
   * \tparam real_t real type corresponding to scalar_t
   * \param d dimension of the points
   * \param x pointer to first point (stored with stride 1)
   * \param x pointer to second point (stored with stride 1)
   */
  template<typename scalar_t,
           typename real_t=typename RealType<scalar_t>::value_type>
  real_t norm1_distance
  (std::size_t d, const scalar_t* x, const scalar_t* y) {
    real_t k(0.);
    for (std::size_t i=0; i<d; i++)
      k += std::abs(x[i]-y[i]);
    return k;
  }

} // end namespace strumpack

#endif // STRUMPACK_METRICS_HPP

