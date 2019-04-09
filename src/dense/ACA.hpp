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
   const std::function<void(std::size_t,scalar_t*,int)>& Arow,
   const std::function<void(std::size_t,scalar_t*,int)>& Acol,
   real_t rtol, real_t atol, int max_rank) {
    using D_t = DenseMatrix<scalar_t>;
    using DW_t = DenseMatrixWrapper<scalar_t>;
    auto sfmin = blas::lamch<real_t>('S');
    int minmn = std::min(m, n);
    int rmax = std::min(minmn, max_rank);
    D_t U_(m, rmax), V_(n, rmax);
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
      Arow(row, Vr.data(), 1);
      for (int l=0; l<rank; l++)
        Vr.scaled_add(-U_(row, l), DW_t(n, 1, V_, 0, l));
      if (Vr.norm() < sfmin) break;
      col = 0;
      auto Vrmax = std::abs(Vr(col, 0));
      for (std::size_t i=1; i<n; i++) {
        if (auto absVri = std::abs(Vr(i, 0)) > Vrmax &&
            std::find(colids.begin(), colids.end(), i) == colids.end()) {
          col = i; Vrmax = absVri;
        }
      }
      colids.push_back(col);
      Vr.scale(scalar_t(1.) / Vr(col, 0));
      DW_t Ur(m, 1, U_, 0, rank);
      Acol(col, Ur.data(), 1);
      for (int l=0; l<rank; l++)
        Ur.scaled_add(-V_(col, l), DW_t(m, 1, U_, 0, l));

      scalar_t cross_products(0.);
      for (int k=0; k<rank; k++) {
        scalar_t dot_u(0.), dot_v(0.);
        for (std::size_t i=0; i<m; i++)
          dot_u += Ur(i, 0) * U_(i, k);
        for (std::size_t i=0; i<n; i++)
          dot_v += Vr(i, 0) * V_(i, k);
        cross_products += dot_u * dot_v;
      }
      real_t normVr2(0.), normUr2(0.);
      for (std::size_t i=0; i<n; i++)
        normVr2 += std::real(Vr(i, 0) * Vr(i, 0));
      for (std::size_t i=0; i<m; i++)
        normUr2 += std::real(Ur(i, 0) * Ur(i, 0));
      approx_norm =
        std::sqrt(std::real(approx_norm*approx_norm +
                            real_t(2.) * cross_products
                            + normUr2 * normVr2));
      real_t normVr = std::sqrt(normVr2), normUr = std::sqrt(normUr2);
      rank++;

#if 0
      scalar_t FrobS(0.);
      for (std::size_t j=0; j<n; j++)
        for (std::size_t i=0; i<m; i++) {
          scalar_t Sij(0.);
          for (int k=0; k<rank; k++)
            Sij += U_(i, k) * V_(j, k);
          FrobS += Sij * Sij;
        }
      FrobS = std::sqrt(std::real(FrobS));
      std::cout << "rank=" << rank
                << ", ||Ur||=" << normUr
                << ", ||Vr||=" << normVr
                << ", ||UVt||=" << approx_norm
                << ", ||S||=" << FrobS << " / " << UVrank.normF()
                << ", | ||UVt|| - ||S|| |="
                << std::abs(approx_norm - FrobS)
                << std::endl;
#endif

      if (normUr * normVr < approx_norm * rtol ||
          normUr * normVr < atol)
        break;

      // select a new row
      row = 0;
      auto Urmax = std::abs(Ur(row, 0));
      for (std::size_t i=1; i<m; i++) {
        if (auto absUri = std::abs(Ur(i, 0)) > Urmax &&
            std::find(rowids.begin(), rowids.end(), i) == rowids.end()) {
          row = i; Urmax = absUri;
        }
      }
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

#if 0
    D_t A(m, n);
    for (std::size_t i=0; i<m; i++)
      for (std::size_t j=0; j<n; j++)
        A(i, j) = Aelem(i, j);
    auto Anorm = A.norm();
    gemm(Trans::N, Trans::N, scalar_t(-1.), U, V, scalar_t(1.), A);
    auto e = A.norm();
    std::cout << "ACA abs_error = " << e
              << " ACA rel_error = " << e/Anorm << std::endl;
#endif
  }


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
   real_t rtol, real_t atol, int max_rank) {
    auto Arow = [&](std::size_t r, scalar_t* B, int ldB) {
      for (std::size_t j=0; j<n; j++)
        B[j*ldB] = Aelem(r, j);
    };
    auto Acol = [&](std::size_t c, scalar_t* B, int ldB) {
      for (std::size_t i=0; i<m; i++)
        B[i*ldB] = Aelem(i, c);
    };
    adaptive_cross_approximation<scalar_t>
      (U, V, m, n, Arow, Acol, rtol, atol, max_rank);
  }

} // end namespace strumpack

#endif // ADAPTIVE_CROSS_APPROXIMATION_HPP
