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
#ifndef BLOCKED_ADAPTIVE_CROSS_APPROXIMATION_HPP
#define BLOCKED_ADAPTIVE_CROSS_APPROXIMATION_HPP

#include "DenseMatrix.hpp"
#include <unordered_set>
#include <memory>
#include <iterator>

namespace strumpack {

  template<typename scalar_t,
           typename real_t = typename RealType<scalar_t>::value_type>
  int LRID(DenseMatrix<scalar_t>& C, DenseMatrix<scalar_t>& W,
           DenseMatrix<scalar_t>& R, real_t rtol, real_t atol,
           int* piv, scalar_t* tau) {
    assert(W.rows() == W.cols() && W.rows() == C.cols() &&
           W.cols() == R.rows());
    std::size_t d = W.rows();
    int rank = 0;
    std::fill(piv, piv+d, 0);
    int info = blas::geqp3tol
      (d, d, W.data(), W.ld(), piv, tau, rank, rtol, atol,
       params::task_recursion_cutoff_level);
    blas::lapmt(true, C.rows(), C.cols(), C.data(), C.ld(), piv);
    // R = T^{-1} Q^T R
    info = blas::xxmqr
      ('L', 'T', d, R.cols(), rank, W.data(), W.ld(), tau,
       R.data(), R.ld());
    blas::trsm
      ('L', 'U', 'N', 'N', rank, R.cols(), scalar_t(1.),
       W.data(), W.ld(), R.data(), R.ld());
    return rank;
  }

  template<typename scalar_t,
           typename real_t = typename RealType<scalar_t>::value_type>
  real_t LRnorm(DenseMatrix<scalar_t>& C, DenseMatrix<scalar_t>& R,
                scalar_t* tau) {
    using D_t = DenseMatrix<scalar_t>;
    using DW_t = DenseMatrixWrapper<scalar_t>;
    std::size_t k = C.cols();
    if (k == 0) return real_t(0.);
    std::size_t m = C.rows(), n = R.cols();
    D_t Ctemp(C), Rtemp(R), TT(k, k);
    int info = blas::geqrf
      (Ctemp.rows(), Ctemp.cols(), Ctemp.data(), Ctemp.ld(), tau);
    info = blas::gelqf
      (Rtemp.rows(), Rtemp.cols(), Rtemp.data(), Rtemp.ld(), tau);
    auto minmk = std::min(m, k), minnk = std::min(n, k);
    for (std::size_t j=0; j<minmk; j++)
      for (std::size_t i=j; i<minmk; i++)
        Ctemp(i, j) = scalar_t(0.);
    for (std::size_t j=0; j<minnk; j++)
      for (std::size_t i=0; i<j; i++)
        Rtemp(i, j) = scalar_t(0.);
    gemm(Trans::N, Trans::C, scalar_t(1.), DW_t(minmk, k, Ctemp, 0, 0),
         DW_t(minnk, k, Rtemp, 0, 0), scalar_t(0.), TT);
    return TT.normF();
  }


  // template<typename scalar_t,
  //          typename real_t = typename RealType<scalar_t>::value_type>
  // real_t LRnormUp
  // (DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V, real_t mu,
  //  DenseMatrix<scalar_t>& C, DenseMatrix<scalar_t>& R, real_t nu) {
  //   // TODO
  //   return mu;
  // }

  template<typename scalar_t> std::vector<std::size_t> ID
  (DenseMatrix<scalar_t>& X, int* piv, scalar_t* tau) {
    auto m = X.rows();
    auto n = X.cols();
    std::fill(piv, piv+n, 0);
    int info = blas::geqp3(m, n, X.data(), X.ld(), piv, tau);
    std::vector<std::size_t> I;
    std::transform
      (piv, piv+std::min(m, n), std::back_inserter(I),
       [](int pvt){ return pvt-1; });
    return I;
  }


  /*
   * Compute U*V ~ A
   */
  template<typename scalar_t,
           typename real_t = typename RealType<scalar_t>::value_type>
  void blocked_adaptive_cross_approximation
  (DenseMatrix<scalar_t>& Uout, DenseMatrix<scalar_t>& Vout,
   std::size_t m, std::size_t n,
   const std::function<void
   (const std::vector<std::size_t>&, DenseMatrix<scalar_t>&)>& Arow,
   const std::function<void
   (const std::vector<std::size_t>&, DenseMatrix<scalar_t>&)>& Acol,
   int d, real_t rtol, real_t atol, int max_rank, int task_depth=0) {
    using D_t = DenseMatrix<scalar_t>;
    using DW_t = DenseMatrixWrapper<scalar_t>;
    int rmax = std::min(m, n), maxmn = std::max(m, n);
    d = std::min(d, rmax);
    Uout.resize(m, rmax);
    Vout.resize(rmax, n);
    std::unique_ptr<int[]> piv_(new int[maxmn]); auto piv = piv_.get();
    std::unique_ptr<scalar_t[]> tau_(new scalar_t[maxmn]); auto tau = tau_.get();
    std::vector<std::size_t> J, I;
    {
      std::mt19937 mt;
      std::uniform_int_distribution<int> rgen(0, n-1);
      std::unordered_set<std::size_t> unique_ids;
      while (unique_ids.size() < d)
        unique_ids.insert(rgen(mt));
      J.assign(unique_ids.begin(), unique_ids.end());
    }
    int rank = 0, info;
    real_t mu(0.);
    while (rank < rmax) {
      if (rmax - rank < d) {
        d = rmax - rank;
        J.resize(d);
      }
      DW_t U(m, rank, Uout, 0, 0), V(rank, n, Vout, 0, 0),
        C(m, d, Uout, 0, rank), R(d, n, Vout, rank, 0);
      Acol(J, C);                                      // C = A(:,J)
      gemm(Trans::N, Trans::N, scalar_t(-1.), U,       // C -= U*V(:,J)
           V.extract_cols(J), scalar_t(1.), C, task_depth);
      {
        auto Ct = C.transpose();
        I = ID(Ct, piv, tau);
      }
      Arow(I, R);                                      // R = A(I,:)
      auto UI = U.extract_rows(I);                     // UI = U(I,:)
      gemm(Trans::N, Trans::N, scalar_t(-1.), UI, V,   // R -= U(I,:)*V
           scalar_t(1.), R, task_depth);
      {
        D_t Rcopy(R);
        J = ID(Rcopy, piv, tau);
      }
      Acol(J, C);                                      // C = A(:,J)
      auto W = C.extract_rows(I);                      // W = C(I,:) = A(I,J)
      auto VJ = V.extract_cols(J);                     // VJ = V(:,J)
      gemm(Trans::N, Trans::N, scalar_t(-1.), U, VJ,   // C -= U*V(:,J)
           scalar_t(1.), C, task_depth);
      gemm(Trans::N, Trans::N, scalar_t(-1.), UI, VJ,  // W -= U(I,:)*V(:,J)
           scalar_t(1.), W, task_depth);
      {
        D_t Rbar(R);            // Rbar(:,j) = R(:,j) for j not in J, else zero
        for (auto j : J)
          for (std::size_t i=0; i<d; i++)
            Rbar(i, j) = scalar_t(0.);
        J = ID(Rbar, piv, tau);
      }
      rank += LRID(C, W, R, rtol, atol, piv, tau);     // CR = C inv(W) R

      real_t nu = LRnorm(C, R, tau);
      // mu = LRnormUp(U, V, mu, C, R, nu);
      mu = LRnorm(U, V, tau);
      if (nu < rtol * mu || nu < atol) break;
    }

    // TODO recompress

    Uout.resize(m, rank);
    Vout.resize(rank, n);
  }

} // end namespace strumpack

#endif // BLOCKED_ADAPTIVE_CROSS_APPROXIMATION_HPP
