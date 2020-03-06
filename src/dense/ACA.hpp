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
#ifndef ADAPTIVE_CROSS_APPROXIMATION_HPP
#define ADAPTIVE_CROSS_APPROXIMATION_HPP

#include <functional>

#include "DenseMatrix.hpp"

namespace strumpack {

  /*
   * Compute U*V ~ A
   */
  template<typename scalar_t,
           typename real_t = typename RealType<scalar_t>::value_type>
  void adaptive_cross_approximation
  (DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V,
   std::size_t m, std::size_t n,
   const std::function<void(std::size_t,scalar_t*)>& Arow,
   const std::function<void(std::size_t,scalar_t*)>& Acol,
   real_t rtol, real_t atol, int max_rank,
   int task_depth=params::task_recursion_cutoff_level);

  /**
   * ACA with element extraction routine.
   * If possible use the ACA version with column/row extraction routines.
   */
  template<typename scalar_t,
           typename real_t = typename RealType<scalar_t>::value_type>
  void adaptive_cross_approximation
  (DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V,
   std::size_t m, std::size_t n,
   const std::function<scalar_t(std::size_t,std::size_t)>& Aelem,
   real_t rtol, real_t atol, int max_rank);

} // end namespace strumpack

#endif // ADAPTIVE_CROSS_APPROXIMATION_HPP
