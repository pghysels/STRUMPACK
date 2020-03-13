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

#include <unordered_set>
#include <memory>
#include <iterator>
#include <algorithm>
#include <random>

#include "BACA.hpp"

namespace strumpack {

  template<typename scalar_t,
           typename real_t = typename RealType<scalar_t>::value_type>
  int LRID(DenseMatrix<scalar_t>& C, DenseMatrix<scalar_t>& W,
           DenseMatrix<scalar_t>& R, real_t rtol, real_t atol,
           int* piv, scalar_t* tau) {
    assert(W.rows() == W.cols() && W.rows() == C.cols() &&
           W.cols() == R.rows());
    int d = W.rows(), rank = 0;
    std::fill(piv, piv+d, 0);
#if 0
    blas::geqp3tol
      (d, d, W.data(), W.ld(), piv, tau, rank, rtol, atol,
       params::task_recursion_cutoff_level);
#else
    blas::geqp3(d, d, W.data(), W.ld(), piv, tau);
    auto sfmin = blas::lamch<real_t>('S');
    for (; rank<d; rank++) if (std::abs(W(rank,rank)) < sfmin) break;
#endif
    blas::lapmt(true, C.rows(), C.cols(), C.data(), C.ld(), piv);
    // R = T^{-1} Q^T R
    blas::xxmqr
      ('L', 'T', d, R.cols(), rank, W.data(), W.ld(), tau,
       R.data(), R.ld());
    blas::trsm
      ('L', 'U', 'N', 'N', rank, R.cols(), scalar_t(1.),
       W.data(), W.ld(), R.data(), R.ld());
    return rank;
  }

  template<typename scalar_t,
           typename real_t = typename RealType<scalar_t>::value_type>
  void LRnormUpCholQR
  (const DenseMatrix<scalar_t>& U, const DenseMatrix<scalar_t>& V, real_t& mu,
   const DenseMatrix<scalar_t>& C, const DenseMatrix<scalar_t>& R, real_t& nu,
   scalar_t* work) {
    real_t s = mu*mu;
    std::size_t r = U.cols(), rb = C.cols();
    DenseMatrixWrapper<scalar_t> VRt(r, rb, work, r),
      CtU(rb, r, work+r*rb, rb);
    gemm(Trans::N, Trans::C, scalar_t(1.), V, R, scalar_t(0.), VRt,
         params::task_recursion_cutoff_level);
    gemm(Trans::C, Trans::N, scalar_t(1.), C, U, scalar_t(0.), CtU,
         params::task_recursion_cutoff_level);
    for (std::size_t i=0; i<r-rb; i++)
      for (std::size_t j=0; j<rb; j++)
        s += real_t(2.) * std::real(VRt(i, j) * CtU(j, i));
    DenseMatrixWrapper<scalar_t> RRt(rb, rb, VRt, r-rb, 0),
      CtC(rb, rb, CtU, 0, r-rb);
    blas::potrf('U', CtC.rows(), CtC.data(), CtC.ld());
    blas::potrf('L', RRt.rows(), RRt.data(), RRt.ld());
    nu = real_t(0.);
    for (std::size_t j=0; j<rb; j++)
      for (std::size_t i=0; i<rb; i++) {
        scalar_t RLij(0.);
        for (std::size_t k=std::max(i,j); k<rb; k++)
          RLij += CtC(i, k) * RRt(k,j);
        nu += std::real(RLij*RLij);
      }
    mu = std::sqrt(s + nu);
    nu = std::sqrt(nu);
  }

  template<typename scalar_t> void ID
  (DenseMatrix<scalar_t>& X, int* piv, scalar_t* tau,
   std::vector<std::size_t>& I) {
    auto m = X.rows();
    auto n = X.cols();
    std::fill(piv, piv+n, 0);
    blas::geqp3(m, n, X.data(), X.ld(), piv, tau);
    I.resize(std::min(m, n));
    std::transform
      (piv, piv+std::min(m, n), I.begin(), [](int pvt){ return pvt-1; });
    std::sort(I.begin(), I.end());
  }


  /*
   * Compute U*V ~ A
   */
  template<typename scalar_t,typename real_t>
  void blocked_adaptive_cross_approximation
  (DenseMatrix<scalar_t>& Uout, DenseMatrix<scalar_t>& Vout,
   std::size_t m, std::size_t n,
   const std::function<void
   (const std::vector<std::size_t>&, DenseMatrix<scalar_t>&)>& Arow,
   const std::function<void
   (const std::vector<std::size_t>&, DenseMatrix<scalar_t>&)>& Acol,
   std::size_t d, real_t rtol, real_t atol, std::size_t max_rank,
   int task_depth) {
    using DW_t = DenseMatrixWrapper<scalar_t>;
    std::size_t rmax = std::min(max_rank, std::min(m, n)),
      maxmn = std::max(m, n);
    d = std::min(d, rmax);
    Uout.resize(m, rmax);
    Vout.resize(rmax, n);
    std::vector<std::size_t> J, I;
    {
      std::mt19937 mt;
      std::uniform_int_distribution<std::size_t> rgen(0, n-1);
      std::unordered_set<std::size_t> unique_ids;
      while (unique_ids.size() < d)
        unique_ids.insert(rgen(mt));
      J.assign(unique_ids.begin(), unique_ids.end());
      std::sort(J.begin(), J.end());
    }

    std::unique_ptr<int[]> piv_(new int[maxmn]); auto piv = piv_.get();
    std::size_t lwork_W_UI_VJ_Rtemp_Ctemp = d*d+2*rmax*d+maxmn*d;
    std::unique_ptr<scalar_t[]> w_
      (new scalar_t[maxmn + lwork_W_UI_VJ_Rtemp_Ctemp]);
    auto tau = w_.get();
    auto work = tau + maxmn;
    std::size_t rank = 0;
    real_t mu(0.);

    while (rank < rmax) {
      if (rmax - rank < d) {
        d = rmax - rank;
        J.resize(d);
      }
      DW_t U(m, rank, Uout, 0, 0), V(rank, n, Vout, 0, 0),
        C(m, d, Uout, 0, rank), R(d, n, Vout, rank, 0),
        W(d, d, work, d), UI(d, rank, W.end(), d),
        VJ(rank, d, UI.end(), rank), Rtemp(d, n, VJ.end(), d),
        Ct(d, m, VJ.end(), d); // Rtemp and Ct overlap
      Acol(J, C);                                      // C = A(:,J)
      V.extract_cols(J, VJ);                           // VJ = V(:,J)
      gemm(Trans::N, Trans::N, scalar_t(-1.), U,       // C -= U*V(:,J)
           VJ, scalar_t(1.), C, task_depth);
      C.transpose(Ct);
      ID(Ct, piv, tau, I);
      Arow(I, R);                                      // R = A(I,:)
      U.extract_rows(I, UI);                           // UI = U(I,:)
      gemm(Trans::N, Trans::N, scalar_t(-1.), UI, V,   // R -= U(I,:)*V
           scalar_t(1.), R, task_depth);
      Rtemp.copy(R);
      ID(Rtemp, piv, tau, J);
      Acol(J, C);                                      // C = A(:,J)
      C.extract_rows(I, W);                            // W = C(I,:) = A(I,J)
      V.extract_cols(J, VJ);                           // VJ = V(:,J)
      gemm(Trans::N, Trans::N, scalar_t(-1.), U, VJ,   // C -= U*V(:,J)
           scalar_t(1.), C, task_depth);
      gemm(Trans::N, Trans::N, scalar_t(-1.), UI, VJ,  // W -= U(I,:)*V(:,J)
           scalar_t(1.), W, task_depth);
      Rtemp.copy(R);
      for (auto j : J)
        for (std::size_t i=0; i<d; i++)
          Rtemp(i, j) = scalar_t(0.);
      ID(Rtemp, piv, tau, J);
      auto dr = LRID(C, W, R, rtol, atol, piv, tau);   // CR = C inv(W) R
      if (dr == 0) break;
      rank += dr;
      real_t nu;
      LRnormUpCholQR
        (DW_t(m, rank, Uout, 0, 0), DW_t(rank, n, Vout, 0, 0), mu,
         DW_t(m, dr, Uout, 0, rank-dr), DW_t(dr, n, Vout, rank-dr, 0),
         nu, work);
      if (nu < rtol * mu || nu < atol || dr == 0) break;
    }

    // TODO recompress

    Uout.resize(m, rank);
    Vout.resize(rank, n);
  }



  /*
   * Compute U*V ~ A
   */
  template<typename scalar_t,typename real_t>
  void blocked_adaptive_cross_approximation_nodups
  (DenseMatrix<scalar_t>& Uout, DenseMatrix<scalar_t>& Vout,
   std::size_t m, std::size_t n,
   const std::function<void
   (const std::vector<std::size_t>&, DenseMatrix<scalar_t>&)>& Arow,
   const std::function<void
   (const std::vector<std::size_t>&, DenseMatrix<scalar_t>&)>& Acol,
   std::size_t d, real_t rtol, real_t atol, std::size_t max_rank,
   int task_depth) {
    using DW_t = DenseMatrixWrapper<scalar_t>;
    std::size_t rmax = std::min(max_rank, std::min(m, n)),
      maxmn = std::max(m, n);
    d = std::min(d, rmax);
    Uout.resize(m, rmax);
    Vout.resize(rmax, n);
    std::vector<std::size_t> J, I, rowids, colids;
    rowids.reserve(rmax);
    colids.reserve(rmax);
    {
      std::mt19937 mt;
      std::uniform_int_distribution<std::size_t> rgen(0, n-1);
      std::unordered_set<std::size_t> unique_ids;
      while (unique_ids.size() < d)
        unique_ids.insert(rgen(mt));
      J.assign(unique_ids.begin(), unique_ids.end());
      std::sort(J.begin(), J.end());
    }
    std::unique_ptr<int[]> piv_(new int[maxmn]); auto piv = piv_.get();
    std::size_t lwork_W_UI_VJ_Rtemp_Ctemp = d*d+2*rmax*d+maxmn*d;
    std::unique_ptr<scalar_t[]> w_
      (new scalar_t[maxmn + lwork_W_UI_VJ_Rtemp_Ctemp]);
    auto tau = w_.get();
    auto work = tau + maxmn;
    std::size_t rank = 0;
    real_t mu(0.);

    while (rank < rmax) {
      if (rmax - rank < d) {
        d = rmax - rank;
        J.resize(d);
      }
      DW_t U(m, rank, Uout, 0, 0), V(rank, n, Vout, 0, 0),
        C(m, d, Uout, 0, rank), R(d, n, Vout, rank, 0),
        W(d, d, work, d), UI(d, rank, W.end(), d),
        VJ(rank, d, UI.end(), rank), Rtemp(d, n, VJ.end(), d),
        Ct(d, m, VJ.end(), d); // Rtemp and Ct overlap
      colids.insert(colids.end(), J.begin(), J.end());
      Acol(J, C);                                      // C = A(:,J)
      V.extract_cols(J, VJ);
      gemm(Trans::N, Trans::N, scalar_t(-1.), U,       // C -= U*V(:,J)
           VJ, scalar_t(1.), C, task_depth);
      C.transpose(Ct);
      for (auto j : rowids)
        for (std::size_t i=0; i<d; i++)
          Ct(i, j) = scalar_t(0.);
      ID(Ct, piv, tau, I);
      rowids.insert(rowids.end(), I.begin(), I.end());
      Arow(I, R);                                      // R = A(I,:)
      R.extract_cols(J, W);                            // W = C(I,:) = A(I,J)
      U.extract_rows(I, UI);                           // UI = U(I,:)
      gemm(Trans::N, Trans::N, scalar_t(-1.), UI, V,   // R -= U(I,:)*V
           scalar_t(1.), R, task_depth);
      gemm(Trans::N, Trans::N, scalar_t(-1.), UI, VJ,  // W -= U(I,:)*V(:,J)
           scalar_t(1.), W, task_depth);
      Rtemp.copy(R);
      for (auto j : colids)
        for (std::size_t i=0; i<d; i++)
          Rtemp(i, j) = scalar_t(0.);
      ID(Rtemp, piv, tau, J);
      auto dr = LRID(C, W, R, rtol, atol, piv, tau);   // CR = C inv(W) R
      if (dr == 0) break;
      rank += dr;
      rowids.resize(rank);
      colids.resize(rank);
      real_t nu;
      LRnormUpCholQR
        (DW_t(m, rank, Uout, 0, 0), DW_t(rank, n, Vout, 0, 0), mu,
         DW_t(m, dr, Uout, 0, rank-dr), DW_t(dr, n, Vout, rank-dr, 0),
         nu, work);
      if (nu < rtol * mu || nu < atol) break;
    }

    // TODO recompress

    Uout.resize(m, rank);
    Vout.resize(rank, n);
  }

  // explicit template instantiations
  template void blocked_adaptive_cross_approximation
  (DenseMatrix<float>& Uout, DenseMatrix<float>& Vout,
   std::size_t m, std::size_t n,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<float>&)>& Arow,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<float>&)>& Acol,
   std::size_t d, float rtol, float atol, std::size_t max_rank, int task_depth);
  template void blocked_adaptive_cross_approximation
  (DenseMatrix<double>& Uout, DenseMatrix<double>& Vout,
   std::size_t m, std::size_t n,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<double>&)>& Arow,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<double>&)>& Acol,
   std::size_t d, double rtol, double atol, std::size_t max_rank, int task_depth);
  template void blocked_adaptive_cross_approximation
  (DenseMatrix<std::complex<float>>& Uout, DenseMatrix<std::complex<float>>& Vout,
   std::size_t m, std::size_t n,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<std::complex<float>>&)>& Arow,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<std::complex<float>>&)>& Acol,
   std::size_t d, float rtol, float atol, std::size_t max_rank, int task_depth);
  template void blocked_adaptive_cross_approximation
  (DenseMatrix<std::complex<double>>& Uout, DenseMatrix<std::complex<double>>& Vout,
   std::size_t m, std::size_t n,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<std::complex<double>>&)>& Arow,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<std::complex<double>>&)>& Acol,
   std::size_t d, double rtol, double atol, std::size_t max_rank, int task_depth);


  template void blocked_adaptive_cross_approximation_nodups
  (DenseMatrix<float>& Uout, DenseMatrix<float>& Vout,
   std::size_t m, std::size_t n,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<float>&)>& Arow,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<float>&)>& Acol,
   std::size_t d, float rtol, float atol, std::size_t max_rank, int task_depth);
  template void blocked_adaptive_cross_approximation_nodups
  (DenseMatrix<double>& Uout, DenseMatrix<double>& Vout,
   std::size_t m, std::size_t n,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<double>&)>& Arow,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<double>&)>& Acol,
   std::size_t d, double rtol, double atol, std::size_t max_rank, int task_depth);
  template void blocked_adaptive_cross_approximation_nodups
  (DenseMatrix<std::complex<float>>& Uout, DenseMatrix<std::complex<float>>& Vout,
   std::size_t m, std::size_t n,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<std::complex<float>>&)>& Arow,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<std::complex<float>>&)>& Acol,
   std::size_t d, float rtol, float atol, std::size_t max_rank, int task_depth);
  template void blocked_adaptive_cross_approximation_nodups
  (DenseMatrix<std::complex<double>>& Uout, DenseMatrix<std::complex<double>>& Vout,
   std::size_t m, std::size_t n,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<std::complex<double>>&)>& Arow,
   const std::function<void(const std::vector<std::size_t>&, DenseMatrix<std::complex<double>>&)>& Acol,
   std::size_t d, double rtol, double atol, std::size_t max_rank, int task_depth);

} // end namespace strumpack
