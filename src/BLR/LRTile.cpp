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
#include <cassert>
#include <iostream>
#include <iomanip>

#include "LRTile.hpp"
#include "DenseTile.hpp"

#include "StrumpackParameters.hpp"
#include "dense/ACA.hpp"
#include "dense/BACA.hpp"

namespace strumpack {
  namespace BLR {

    template<typename scalar_t> LRTile<scalar_t>::LRTile
    (std::size_t m, std::size_t n, std::size_t r) {
      U_ = DenseM_t(m, r);
      V_ = DenseM_t(r, n);
    }

    template<typename scalar_t> LRTile<scalar_t>::LRTile
    (const DenseM_t& T, const Opts_t& opts) {
      if (opts.low_rank_algorithm() == LowRankAlgorithm::RRQR) {
        if (T.rows() == 0 || T.cols() == 0) {
          U_ = DenseM_t(T.rows(), 0);
          V_ = DenseM_t(0, T.cols());
        } else
          T.low_rank(U_, V_, opts.rel_tol(), opts.abs_tol(), opts.max_rank(),
                     params::task_recursion_cutoff_level);
      } else if (opts.low_rank_algorithm() == LowRankAlgorithm::ACA) {
        adaptive_cross_approximation<scalar_t>
          (U_, V_, T.rows(), T.cols(),
           [&](std::size_t i, std::size_t j) -> scalar_t {
             assert(i < T.rows());
             assert(j < T.cols());
             return T(i, j); },
           opts.rel_tol(), opts.abs_tol(), opts.max_rank());
      }
    }

    template<typename scalar_t> LRTile<scalar_t>
    LRTile<scalar_t>::multiply(const BLRTile<scalar_t>& a) const {
      return a.left_multiply(*this);
    }
    template<typename scalar_t> LRTile<scalar_t>
    LRTile<scalar_t>::left_multiply(const LRTile<scalar_t>& a) const {
      LRTile<scalar_t> t(a.rows(), cols(), std::min(a.rank(), rank()));
      left_multiply(a, t.U(), t.V());
      return t;
    }
    template<typename scalar_t> LRTile<scalar_t>
    LRTile<scalar_t>::left_multiply(const DenseTile<scalar_t>& a) const {
      // (a.D*U)*V
      LRTile<scalar_t> t(a.rows(), cols(), rank());
      left_multiply(a, t.U(), t.V());
      return t;
    }

    template<typename scalar_t> void LRTile<scalar_t>::multiply
    (const BLRTile<scalar_t>& a, DenseM_t& b, DenseM_t& c) const {
      a.left_multiply(*this, b, c);
    }
    template<typename scalar_t> void LRTile<scalar_t>::left_multiply
    (const LRTile<scalar_t>& a, DenseM_t& b, DenseM_t& c) const {
      DenseM_t VU(a.rank(), rank());
      gemm(Trans::N, Trans::N, scalar_t(1.), a.V(), U_, scalar_t(0.),
           VU, params::task_recursion_cutoff_level);
      if (a.rank() < rank()) {
        // a.U*((a.V * U_)*V_)
        gemm(Trans::N, Trans::N, scalar_t(1.), VU, V_, scalar_t(0.),
             c, params::task_recursion_cutoff_level);
        copy(a.U(), b, 0, 0);
      } else {
        // (a.U*(a.V * U_))*V_
        gemm(Trans::N, Trans::N, scalar_t(1.), a.U(), VU, scalar_t(0.),
             b, params::task_recursion_cutoff_level);
        copy(V(), c, 0, 0);
      }
    }

    template<typename scalar_t> void LRTile<scalar_t>::left_multiply
    (const DenseTile<scalar_t>& a, DenseM_t& b, DenseM_t& c) const {
      // (a.D*U)*V
      gemm(Trans::N, Trans::N, scalar_t(1.), a.D(), U(), scalar_t(0.),
           b, params::task_recursion_cutoff_level);
      copy(V(), c, 0, 0);
    }


    /**
     * .. by extracting individual elements
     */
    template<typename scalar_t> LRTile<scalar_t>::LRTile
    (std::size_t m, std::size_t n,
     const std::function<scalar_t(std::size_t,std::size_t)>& Telem,
     const Opts_t& opts) {
      adaptive_cross_approximation<scalar_t>
        (U_, V_, m, n, Telem, opts.rel_tol(), opts.abs_tol(),
         opts.max_rank());
    }

    /**
     * .. by extracting 1 column or 1 row at a time
     */
    template<typename scalar_t> LRTile<scalar_t>::LRTile
    (std::size_t m, std::size_t n,
     const std::function<void(std::size_t,scalar_t*)>& Trow,
     const std::function<void(std::size_t,scalar_t*)>& Tcol,
     const Opts_t& opts) {
      adaptive_cross_approximation<scalar_t>
        (U_, V_, m, n, Trow, Tcol, opts.rel_tol(), opts.abs_tol(),
         opts.max_rank());
    }


    /**
     * .. by extracting multiple columns or rows at a time
     */
    template<typename scalar_t> LRTile<scalar_t>::LRTile
    (std::size_t m, std::size_t n,
     const std::function<void(const std::vector<std::size_t>&,
                              DenseMatrix<scalar_t>&)>& Trow,
     const std::function<void(const std::vector<std::size_t>&,
                              DenseMatrix<scalar_t>&)>& Tcol,
     const Opts_t& opts) {
      //blocked_adaptive_cross_approximation_nodups<scalar_t>
      blocked_adaptive_cross_approximation<scalar_t>
        (U_, V_, m, n, Trow, Tcol, opts.BACA_blocksize(),
         opts.rel_tol(), opts.abs_tol(), opts.max_rank(),
         params::task_recursion_cutoff_level);
    }


    template<typename scalar_t> void
    LRTile<scalar_t>::dense(DenseM_t& A) const {
      assert(A.rows() == rows() && A.cols() == cols());
      gemm(Trans::N, Trans::N, scalar_t(1.), U_, V_, scalar_t(0.), A,
           params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    LRTile<scalar_t>::dense() const {
      DenseM_t A(rows(), cols());
      dense(A);
      return A;
    }

    template<typename scalar_t> std::unique_ptr<BLRTile<scalar_t>>
    LRTile<scalar_t>::clone() const {
      return std::unique_ptr<BLRTile<scalar_t>>(new LRTile(*this));
    }

    template<typename scalar_t> void LRTile<scalar_t>::draw
    (std::ostream& of, std::size_t roff, std::size_t coff) const {
      char prev = std::cout.fill('0');
      int maxrank = rows() * cols() / (rows() + cols());
      int red = std::floor(255.0 * rank() / maxrank);
      int blue = 255 - red;
      of << "set obj rect from "
         << roff << ", " << coff << " to "
         << roff+rows() << ", " << coff+cols()
         << " fc rgb '#"
         << std::hex << std::setw(2) << std::setfill('0') << red
         << "00" << std::setw(2)  << std::setfill('0') << blue
         << "'" << std::dec << std::endl;
      std::cout.fill(prev);
    }

    template<typename scalar_t> scalar_t
    LRTile<scalar_t>::operator()(std::size_t i, std::size_t j) const {
      return blas::dotu(rank(), U_.ptr(i, 0), U_.ld(), V_.ptr(0, j), 1);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::extract(const std::vector<std::size_t>& I,
                              const std::vector<std::size_t>& J,
                              DenseM_t& B) const {
      gemm(Trans::N, Trans::N, scalar_t(1.), U_.extract_rows(I),
           V_.extract_cols(J), scalar_t(0.), B,
           params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::laswp(const std::vector<int>& piv, bool fwd) {
      U_.laswp(piv, fwd);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::trsm_b(Side s, UpLo ul, Trans ta, Diag d,
                             scalar_t alpha, const DenseM_t& a) {
      strumpack::trsm
        (s, ul, ta, d, alpha, a, (s == Side::L) ? U_ : V_,
         params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::gemv_a(Trans ta, scalar_t alpha, const DenseM_t& x,
                             scalar_t beta, DenseM_t& y) const {
      DenseM_t tmp(rank(), x.cols());
      gemv(ta, scalar_t(1.), ta==Trans::N ? V() : U(), x, scalar_t(0.), tmp,
           params::task_recursion_cutoff_level);
      gemv(ta, alpha, ta==Trans::N ? U() : V(), tmp, beta, y,
           params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::gemm_a(Trans ta, Trans tb, scalar_t alpha,
                             const BLRTile<scalar_t>& b,
                             scalar_t beta, DenseM_t& c) const {
      b.gemm_b(ta, tb, alpha, *this, beta, c);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::gemm_a(Trans ta, Trans tb, scalar_t alpha,
                             const DenseM_t& b, scalar_t beta,
                             DenseM_t& c, int task_depth) const {
      DenseM_t tmp(rank(), c.cols());
      gemm(ta, tb, scalar_t(1.), ta==Trans::N ? V() : U(), b,
           scalar_t(0.), tmp, task_depth);
      gemm(ta, Trans::N, alpha, ta==Trans::N ? U() : V(), tmp,
           beta, c, task_depth);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::gemm_b(Trans ta, Trans tb, scalar_t alpha,
                             const LRTile<scalar_t>& a, scalar_t beta,
                             DenseM_t& c) const {
      DenseM_t tmp1(a.rank(), rank());
      gemm(ta, tb, scalar_t(1.), ta==Trans::N ? a.V() : a.U(),
           tb==Trans::N ? U() : V(), scalar_t(0.), tmp1,
           params::task_recursion_cutoff_level);
      DenseM_t tmp2(c.rows(), tmp1.cols());
      gemm(ta, Trans::N, scalar_t(1.), ta==Trans::N ? a.U() : a.V(), tmp1,
           scalar_t(0.), tmp2, params::task_recursion_cutoff_level);
      gemm(Trans::N, tb, alpha, tmp2, tb==Trans::N ? V() : U(),
           beta, c, params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::gemm_b(Trans ta, Trans tb, scalar_t alpha,
                             const DenseTile<scalar_t>& a, scalar_t beta,
                             DenseM_t& c) const {
      gemm_b(ta, tb, alpha, a.D(), beta, c,
             params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::gemm_b(Trans ta, Trans tb, scalar_t alpha,
                             const DenseM_t& a, scalar_t beta,
                             DenseM_t& c, int task_depth) const {
      DenseM_t tmp(c.rows(), rank());
      gemm(ta, tb, scalar_t(1.), a, tb==Trans::N ? U() : V(),
           scalar_t(0.), tmp, task_depth);
      gemm(Trans::N, tb, alpha, tmp, tb==Trans::N ? V() : U(),
           beta, c, task_depth);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::Schur_update_col_a
    (std::size_t i, const BLRTile<scalar_t>& b, scalar_t* c,
     scalar_t* work) const {
      b.Schur_update_col_b(i, *this, c, work);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::Schur_update_row_a
    (std::size_t i, const BLRTile<scalar_t>& b, scalar_t* c,
     scalar_t* work) const {
      b.Schur_update_row_b(i, *this, c, work);
    }

    /* work should be at least rank(a) + rows(b) */
    template<typename scalar_t> void
    LRTile<scalar_t>::Schur_update_col_b
    (std::size_t i, const LRTile<scalar_t>& a, scalar_t* c,
     scalar_t* work) const {
      DMW_t temp1(rows(), 1, work, rows()),
        temp2(a.rank(), 1, work+rows(), a.rank());
      gemv(Trans::N, scalar_t(1.), U_, V_.ptr(0, i), 1,
           scalar_t(0.), temp1, params::task_recursion_cutoff_level);
      gemv(Trans::N, scalar_t(1.), a.V(), temp1, scalar_t(0.), temp2,
           params::task_recursion_cutoff_level);
      gemv(Trans::N, scalar_t(-1.), a.U(), temp2, scalar_t(1.), c, 1,
           params::task_recursion_cutoff_level);
    }

    /* work should be at least rows(b) */
    template<typename scalar_t> void
    LRTile<scalar_t>::Schur_update_col_b
    (std::size_t i, const DenseTile<scalar_t>& a, scalar_t* c,
     scalar_t* work) const {
      DMW_t temp(rows(), 1, work, rows());
      gemv(Trans::N, scalar_t(1.), U_, V_.ptr(0, i), 1,
           scalar_t(0.), temp, params::task_recursion_cutoff_level);
      gemv(Trans::N, scalar_t(-1.), a.D(), temp, scalar_t(1.), c, 1,
           params::task_recursion_cutoff_level);
    }

    /* work should be at least cols(a) + rank(b) */
    template<typename scalar_t> void
    LRTile<scalar_t>::Schur_update_row_b
    (std::size_t i, const LRTile<scalar_t>& a, scalar_t* c,
     scalar_t* work) const {
      DMW_t temp1(1, a.cols(), work, 1),
        temp2(1, rank(), work+a.cols(), 1);
      gemv(Trans::C, scalar_t(1.), a.V(), a.U().ptr(i, 0), a.U().ld(),
           scalar_t(0.), temp1.data(), temp1.ld(),
           params::task_recursion_cutoff_level);
      gemv(Trans::C, scalar_t(1.), U_, temp1.data(), temp1.ld(),
           scalar_t(0.), temp2.data(), temp2.ld(),
           params::task_recursion_cutoff_level);
      gemv(Trans::C, scalar_t(-1.), V_, temp2.data(), temp2.ld(),
           scalar_t(1.), c, 1, params::task_recursion_cutoff_level);
    }

    /* work should be at least rank(b) */
    template<typename scalar_t> void
    LRTile<scalar_t>::Schur_update_row_b
    (std::size_t i, const DenseTile<scalar_t>& a, scalar_t* c,
     scalar_t* work) const {
      DMW_t temp(1, rank(), work, 1);
      gemv(Trans::C, scalar_t(1.), U(), a.D().ptr(i, 0), a.D().ld(),
           scalar_t(0.), temp.data(), temp.ld(),
           params::task_recursion_cutoff_level);
      gemv(Trans::C, scalar_t(-1.), V(), temp.data(), temp.ld(),
           scalar_t(1.), c, 1, params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::Schur_update_cols_a
    (const std::vector<std::size_t>& cols, const BLRTile<scalar_t>& b,
     DenseMatrix<scalar_t>& c, scalar_t* work) const {
      b.Schur_update_cols_b(cols, *this, c, work);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::Schur_update_rows_a
    (const std::vector<std::size_t>& rows, const BLRTile<scalar_t>& b,
     DenseMatrix<scalar_t>& c, scalar_t* work) const {
      b.Schur_update_rows_b(rows, *this, c, work);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::Schur_update_cols_b
    (const std::vector<std::size_t>& cols, const LRTile<scalar_t>& a,
     DenseMatrix<scalar_t>& c, scalar_t* work) const {
      auto d = cols.size();
      auto r = rank();
      auto m = rows();
      DMW_t Vc(r, d, work, r),
        temp1(m, d, Vc.end(), m),
        temp2(a.rank(), d, temp1.end(), a.rank());
      V_.extract_cols(cols, Vc);
      gemm(Trans::N, Trans::N, scalar_t(1.), U_, Vc,
           scalar_t(0.), temp1, params::task_recursion_cutoff_level);
      gemm(Trans::N, Trans::N, scalar_t(1.), a.V(), temp1, scalar_t(0.),
           temp2, params::task_recursion_cutoff_level);
      gemm(Trans::N, Trans::N, scalar_t(-1.), a.U(), temp2, scalar_t(1.),
           c, params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::Schur_update_cols_b
    (const std::vector<std::size_t>& cols, const DenseTile<scalar_t>& a,
     DenseMatrix<scalar_t>& c, scalar_t* work) const {
      auto r = rank();
      auto d = cols.size();
      auto m = rows();
      DMW_t Vc(r, d, work, r), temp(m, d, Vc.end(), m);
      V_.extract_cols(cols, Vc);
      gemm(Trans::N, Trans::N, scalar_t(1.), U_, Vc,
           scalar_t(0.), temp, params::task_recursion_cutoff_level);
      gemm(Trans::N, Trans::N, scalar_t(-1.), a.D(), temp, scalar_t(1.),
           c, params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::Schur_update_rows_b
    (const std::vector<std::size_t>& rows, const LRTile<scalar_t>& a,
     DenseMatrix<scalar_t>& c, scalar_t* work) const {
      auto d = rows.size();
      DMW_t aUr(d, a.rank(), work, d),
        temp1(d, a.cols(), aUr.end(), d),
        temp2(d, rank(), temp1.end(), d);
      a.U().extract_rows(rows, aUr);
      gemm(Trans::N, Trans::N, scalar_t(1.), aUr, a.V(),
           scalar_t(0.), temp1, params::task_recursion_cutoff_level);
      gemm(Trans::N, Trans::N, scalar_t(1.), temp1, U_, scalar_t(0.),
           temp2, params::task_recursion_cutoff_level);
      gemm(Trans::N, Trans::N, scalar_t(-1.), temp2, V_, scalar_t(1.),
           c, params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void
    LRTile<scalar_t>::Schur_update_rows_b
    (const std::vector<std::size_t>& rows, const DenseTile<scalar_t>& a,
     DenseMatrix<scalar_t>& c, scalar_t* work) const {
      auto d = rows.size();
      DMW_t aDr(d, a.cols(), work, d),
        temp(d, rank(), aDr.end(), rows.size());
      a.D().extract_rows(rows, aDr);
      gemm(Trans::N, Trans::N, scalar_t(1.), aDr, U(),
           scalar_t(0.), temp, params::task_recursion_cutoff_level);
      gemm(Trans::N, Trans::N, scalar_t(-1.), temp, V(),
           scalar_t(1.), c, params::task_recursion_cutoff_level);
    }

    // explicit template instantiations
    template class LRTile<float>;
    template class LRTile<double>;
    template class LRTile<std::complex<float>>;
    template class LRTile<std::complex<double>>;

  } // end namespace BLR
} // end namespace strumpack
