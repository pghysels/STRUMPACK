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
   * Compute U*V^T ~ A
   */
  template<typename scalar_t>
  void AdaptiveCrossApproximation
  (DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V,
   std::size_t m, std::size_t n,
   const std::function<scalar_t(std::size_t,std::size_t)>& Aelem,
   typename RealType<scalar_t>::value_type rel_tol,
   typename RealType<scalar_t>::value_type abs_tol, int max_rank) {
    using D_t = DenseMatrix<scalar_t>;
    using DW_t = DenseMatrixWrapper<scalar_t>;
    using real_t = typename RealType<scalar_t>::value_type;
    auto sfmin = blas::lamch<real_t>('S');
    int minmn = std::min(m, n);
    int rmax = std::min(minmn, max_rank);
    D_t U_(m, rmax), V_(n, rmax);
    // select initial row somehow??
    // use row for point closest to center of cluster?
    int row = minmn / 2, col = 0, rank = 0;
    std::vector<int> Z;
    Z.reserve(rmax);
    scalar_t normUVt2(0.);
    real_t normVr(0.), normUr(0.);
    while (rank < rmax) {
      Z.push_back(row);
      DW_t Vr(n, 1, V_, 0, rank);
      for (std::size_t i=0; i<n; i++)
        Vr(i, 0) = Aelem(row, i);
      for (int l=0; l<rank; l++)
        Vr.scaled_add(-U_(row, l), DW_t(n, 1, V_, 0, l));
      if (Vr.norm() < sfmin)
        break;
      col = 0;
      auto Vrmax = std::fabs(Vr(col, 0));
      for (std::size_t i=1; i<n; i++) {
        if (auto absVri = std::fabs(Vr(i, 0)) > Vrmax) {
          col = i; Vrmax = absVri;
        }
      }
      Vr.scale(scalar_t(1.) / Vr(col, 0));
      DW_t Ur(m, 1, U_, 0, rank);
      for (std::size_t i=0; i<m; i++)
        Ur(i, 0) = Aelem(i, col);
      for (int l=0; l<rank; l++)
        Ur.scaled_add(-V_(col, l), DW_t(m, 1, U_, 0, l));

      normUVt2 += normUr * normUr * normVr * normVr;
      for (int k=0; k<rank-1; k++) {
        scalar_t UrdotUk(0.);
        for (std::size_t i=0; i<m; i++)
          UrdotUk += U_(i, rank-1) * U_(i, k);
        scalar_t VrdotVk(0.);
        for (std::size_t i=0; i<n; i++)
          VrdotVk += V_(i, rank-1) * V_(i, k);
        normUVt2 += scalar_t(2.) * UrdotUk * VrdotVk;
      }
      normVr = Vr.norm();
      normUr = Ur.norm();

#if 0
      scalar_t FrobS(0.);
      for (std::size_t j=0; j<n; j++)
        for (std::size_t i=0; i<m; i++) {
          scalar_t Sij(0.);
          for (int k=0; k<rank; k++)
            Sij += U_(i, k) * V_(j, k);
          FrobS += Sij * Sij;
        }
      //assert(std::fabs(normUVt2 - FrobS) < 10.0 * blas::lamch<real_t>('P'));
      if (std::fabs(normUVt2 - FrobS) > 10.0 * blas::lamch<real_t>('P'))
        std::cout << "rank=" << rank
                  << ", ||Ur||=" << normUr << ", ||Vr||=" << normVr
                  << ", ||UVt||=" << std::sqrt(normUVt2)
                  << ", | ||UVt|| - ||S|| |="
                  << std::fabs(normUVt2 - FrobS)
                  << std::endl;
#endif

      if (normUr * normVr < std::sqrt(normUVt2) * rel_tol ||
          normUr * normVr < abs_tol)
        break;

      // select a new row
      row = 0;
      auto Urmax = std::fabs(Ur(row, 0));
      for (std::size_t i=1; i<m; i++) {
        if (auto absUri = std::fabs(Ur(i, 0)) > Urmax &&
            std::find(Z.begin(), Z.end(), i) == Z.end()) {
          row = i; Urmax = absUri;
        }
      }
      rank++;
    }


    // TODO recompress!?
    // QR of U, SVD of R?
    if (!rank) {
      rank = 1;
      U = D_t(m, rank); U.zero();
      V = D_t(n, rank); V.zero();
    } else {
      U = D_t(m, rank); U.copy(U_, 0, 0);
      V = D_t(n, rank); V.copy(V_, 0, 0);
    }

    // A.print("A");
    // U.print("U");
    // V.print("V");
    // D_t UVt(m, n);
    // gemm(Trans::N, Trans::C, scalar_t(1.), U, V, scalar_t(0.), UVt);
    // UVt.print("UVt");

  }

} // end namespace strumpack

#endif // ADAPTIVE_CROSS_APPROXIMATION_HPP
