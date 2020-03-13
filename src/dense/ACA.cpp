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
#include <iterator>
#include <algorithm>
#include <random>

#include "ACA.hpp"

namespace strumpack {

  /*
   * Compute U*V ~ A
   */
  template<typename scalar_t,typename real_t>
  void adaptive_cross_approximation
  (DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V,
   std::size_t m, std::size_t n,
   const std::function<void(std::size_t,scalar_t*)>& Arow,
   const std::function<void(std::size_t,scalar_t*)>& Acol,
   real_t rtol, real_t atol, int max_rank, int task_depth) {
    using D_t = DenseMatrix<scalar_t>;
    using DW_t = DenseMatrixWrapper<scalar_t>;
    auto sfmin = blas::lamch<real_t>('S');
    int rmax = std::min(std::min(m, n), std::size_t(max_rank));
    D_t U_(m, rmax), V_(n, rmax); // TODO store V iso V transpose
    std::vector<real_t> temp(std::max(m, n));
    std::vector<scalar_t> du(rmax), dv(rmax);
    std::mt19937 mt;
    std::uniform_int_distribution<int> rgen(0, m-1);
    int row = rgen(mt), col = 0, rank = 0;
    std::vector<int> rowids, colids;
    rowids.reserve(rmax);
    colids.reserve(rmax);
    real_t approx_norm(0.);
    while (rank < rmax) {
      rowids.push_back(row);
      DW_t Vr(n, 1, V_, 0, rank);
      Arow(row, Vr.data());
      gemv(Trans::N, scalar_t(-1.), DW_t(n, rank, V_, 0, 0),
           U_.ptr(row, 0), U_.ld(), scalar_t(1.), Vr.ptr(0, 0), 1,
           task_depth);
      for (std::size_t i=0; i<n; i++) temp[i] = std::abs(Vr(i, 0));
      // avoid already selected cols
      for (auto c : colids) temp[c] = real_t(-1);
      col = std::distance
        (temp.begin(), std::max_element(temp.begin(), temp.begin()+n));
      if (std::abs(Vr(col, 0)) < sfmin) break;
      colids.push_back(col);
      Vr.scale(scalar_t(1.) / Vr(col, 0));
      DW_t Ur(m, 1, U_, 0, rank);
      Acol(col, Ur.data());
      gemv(Trans::N, scalar_t(-1.), DW_t(m, rank, U_, 0, 0),
           V_.ptr(col, 0), V_.ld(), scalar_t(1.), Ur.ptr(0, 0), 1,
           task_depth);
      gemv(Trans::C, scalar_t(1.), DW_t(m, rank+1, U_, 0, 0), Ur,
           scalar_t(0.), du.data(), 1, task_depth);
      gemv(Trans::C, scalar_t(1.), DW_t(n, rank+1, V_, 0, 0), Vr,
           scalar_t(0.), dv.data(), 1, task_depth);
      scalar_t cross_products = blas::dotu(rank, du.data(), 1, dv.data(), 1);
      real_t normUV2 = std::real(du[rank] * dv[rank]);
      approx_norm = std::sqrt
        (std::real(approx_norm*approx_norm +
                   real_t(2.) * cross_products + normUV2));
      rank++;
      real_t nrmUV = std::sqrt(normUV2);
      if (nrmUV < approx_norm * rtol || nrmUV < atol) break;
      // select a new row
      for (std::size_t i=0; i<m; i++) temp[i] = std::abs(Ur(i, 0));
      // avoid already selected rows
      for (auto r : rowids) temp[r] = real_t(-1);
      row = std::distance
        (temp.begin(), std::max_element(temp.begin(), temp.begin()+m));
    }

    // // recompression TODO what tolerance to use here??
    // // TODO also compress V
    // D_t B;
    // DW_t(m, rank, U_, 0, 0).low_rank(U, B, rtol, atol, rank, 0);
    // //DW_t(m, rank, U_, 0, 0).low_rank(U, B, 1e-20, 1e-20, rank, 0);
    // V = D_t(U.cols(), n);
    // gemm(Trans::N, Trans::C, scalar_t(1.), B,
    //      DW_t(n, rank, V_, 0, 0), scalar_t(0.), V);

    U = D_t(m, rank); U.copy(U_, 0, 0);
    V = DW_t(n, rank, V_, 0, 0).transpose();
  }


  /**
   * ACA with element extraction routine.
   * If possible use the ACA version with column/row extraction routines.
   */
  template<typename scalar_t,typename real_t>
  void adaptive_cross_approximation
  (DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V,
   std::size_t m, std::size_t n,
   const std::function<scalar_t(std::size_t,std::size_t)>& Aelem,
   real_t rtol, real_t atol, int max_rank) {
    auto Arow =
      [&](std::size_t r, scalar_t* B) {
        for (std::size_t j=0; j<n; j++) B[j] = Aelem(r, j);
      };
    auto Acol =
      [&](std::size_t c, scalar_t* B) {
        for (std::size_t i=0; i<m; i++) B[i] = Aelem(i, c);
      };
    adaptive_cross_approximation<scalar_t>
      (U, V, m, n, Arow, Acol, rtol, atol, max_rank);
  }


  // explicit template instantiations
  template void adaptive_cross_approximation
  (DenseMatrix<float>& U, DenseMatrix<float>& V,
   std::size_t m, std::size_t n,
   const std::function<void(std::size_t,float*)>& Arow,
   const std::function<void(std::size_t,float*)>& Acol,
   float rtol, float atol, int max_rank, int task_depth);
  template void adaptive_cross_approximation
  (DenseMatrix<double>& U, DenseMatrix<double>& V,
   std::size_t m, std::size_t n,
   const std::function<void(std::size_t,double*)>& Arow,
   const std::function<void(std::size_t,double*)>& Acol,
   double rtol, double atol, int max_rank, int task_depth);
  template void adaptive_cross_approximation
  (DenseMatrix<std::complex<float>>& U, DenseMatrix<std::complex<float>>& V,
   std::size_t m, std::size_t n,
   const std::function<void(std::size_t,std::complex<float>*)>& Arow,
   const std::function<void(std::size_t,std::complex<float>*)>& Acol,
   float rtol, float atol, int max_rank, int task_depth);
  template void adaptive_cross_approximation
  (DenseMatrix<std::complex<double>>& U, DenseMatrix<std::complex<double>>& V,
   std::size_t m, std::size_t n,
   const std::function<void(std::size_t,std::complex<double>*)>& Arow,
   const std::function<void(std::size_t,std::complex<double>*)>& Acol,
   double rtol, double atol, int max_rank, int task_depth);

  template void adaptive_cross_approximation
  (DenseMatrix<float>& U, DenseMatrix<float>& V,
   std::size_t m, std::size_t n,
   const std::function<float(std::size_t,std::size_t)>& Aelem,
   float rtol, float atol, int max_rank);
  template void adaptive_cross_approximation
  (DenseMatrix<double>& U, DenseMatrix<double>& V,
   std::size_t m, std::size_t n,
   const std::function<double(std::size_t,std::size_t)>& Aelem,
   double rtol, double atol, int max_rank);
  template void adaptive_cross_approximation
  (DenseMatrix<std::complex<float>>& U, DenseMatrix<std::complex<float>>& V,
   std::size_t m, std::size_t n,
   const std::function<std::complex<float>(std::size_t,std::size_t)>& Aelem,
   float rtol, float atol, int max_rank);
  template void adaptive_cross_approximation
  (DenseMatrix<std::complex<double>>& U, DenseMatrix<std::complex<double>>& V,
   std::size_t m, std::size_t n,
   const std::function<std::complex<double>(std::size_t,std::size_t)>& Aelem,
   double rtol, double atol, int max_rank);

} // end namespace strumpack
