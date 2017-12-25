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
#ifndef HSS_MATRIX_LR_HPP
#define HSS_MATRIX_LR_HPP

#include <cassert>
#include <utility>
#include <fstream>
#include <string>

#include "dense/AdaptiveCrossApproximation.hpp"

namespace strumpack {
  namespace H {

    template<typename scalar_t> class HMatrixLR
      : public HMatrixBase<scalar_t> {
      using opts_t = HOptions<scalar_t>;
      using real_t = typename RealType<scalar_t>::value_type;
      using D_t = DenseMatrix<scalar_t>;
      using DW_t = DenseMatrixWrapper<scalar_t>;
      using HB_t = HMatrixBase<scalar_t>;
      using HLR_t = HMatrixLR<scalar_t>;
      using HD_t = HMatrixDense<scalar_t>;
      using H_t = HMatrix<scalar_t>;
      using elem_t = std::function<scalar_t(std::size_t,std::size_t)>;

    public:
      HMatrixLR(const D_t& A, const opts_t& opts);
      HMatrixLR(std::size_t m, std::size_t n,
                const elem_t& Aelem, const opts_t& opts);

      std::size_t rows() const { return _U.rows(); }
      std::size_t cols() const { return _V.rows(); }
      std::size_t rank() const { return _U.cols(); }
      std::size_t memory() const { return _U.memory() + _V.memory(); }
      std::size_t levels() const { return 1; }
      D_t dense() const;
      std::string name() const { return "HMatrixLR"; }

      D_t& U() { return _U; }
      D_t& V() { return _V; }
      const D_t& U() const { return _U; }
      const D_t& V() const { return _V; }

      const opts_t& options() const { return _opts; }

      void AGEMM(Trans ta, Trans tb, scalar_t alpha, const HB_t& b,
                 scalar_t beta, HB_t& c, int depth=0) const {
        b.ABGEMM(ta, tb, alpha, *this, beta, c, depth);
      }

      void ABGEMM(Trans ta, Trans tb, scalar_t alpha, const H_t& a,
                  scalar_t beta, HB_t& c, int depth=0) const {
        c.ABCGEMM(ta, tb, alpha, a, *this, beta, depth);
      }
      void ABGEMM(Trans ta, Trans tb, scalar_t alpha, const HLR_t& a,
                  scalar_t beta, HB_t& c, int depth=0) const {
        c.ABCGEMM(ta, tb, alpha, a, *this, beta, depth);
      }
      void ABGEMM(Trans ta, Trans tb, scalar_t alpha, const HD_t& a,
                  scalar_t beta, HB_t& c, int depth=0) const {
        c.ABCGEMM(ta, tb, alpha, a, *this, beta, depth);
      }
      void ABGEMM(Trans ta, Trans tb, scalar_t alpha, const D_t& a,
                  scalar_t beta, HB_t& c, int depth=0) const {
        c.ABCGEMM(ta, tb, alpha, a, *this, beta, depth);
      }

      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const H_t& a,
                   const H_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }
      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const H_t& a,
                   const HLR_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }
      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const H_t& a,
                   const HD_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }
      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const H_t& a,
                   const D_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }

      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HLR_t& a,
                   const H_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }
      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HLR_t& a,
                   const HLR_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }
      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HLR_t& a,
                   const HD_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }
      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HLR_t& a,
                   const D_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }

      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HD_t& a,
                   const H_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }
      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HD_t& a,
                   const HLR_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }
      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HD_t& a,
                   const HD_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }
      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HD_t& a,
                   const D_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }

      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const D_t& a,
                   const H_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }
      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const D_t& a,
                   const HLR_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }
      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const D_t& a,
                   const HD_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }
      void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const D_t& a,
                   const D_t& b, scalar_t beta, int depth=0) {
        gemm(ta, tb, alpha, a, b, beta, *this, depth);
      }

    protected:
      D_t _U;
      D_t _V;
      opts_t _opts;
      void draw(std::ostream& of, std::size_t rlo=0, std::size_t clo=0) const;
      std::vector<int> LU(int task_depth);
      void laswp(const std::vector<int>& piv, bool fwd);

      template<typename T> friend
      void draw(std::unique_ptr<HMatrixBase<T>> const&, const std::string&);
    };

    template<typename scalar_t>
    HMatrixLR<scalar_t>::HMatrixLR(const D_t& A, const opts_t& opts)
      : _opts(opts) {
      A.low_rank(_U, _V, opts.rel_tol(), opts.abs_tol(), opts.max_rank(), 0);
    }

    template<typename scalar_t>
    HMatrixLR<scalar_t>::HMatrixLR
    (std::size_t m, std::size_t n, const elem_t& Aelem, const opts_t& opts) {
      AdaptiveCrossApproximation
        (_U, _V, m, n, Aelem,
         opts.rel_tol(), opts.abs_tol(), opts.max_rank());
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    HMatrixLR<scalar_t>::dense() const {
      D_t out(rows(), cols());
      gemm(Trans::N, Trans::C, scalar_t(1.), _U, _V, scalar_t(0.), out);
      return out;
    }

    template<typename scalar_t> void
    HMatrixLR<scalar_t>::draw
    (std::ostream& of, std::size_t rlo, std::size_t clo) const {
      int minmn = std::min(rows(), cols());
      int red = std::floor(255.0 * rank() / minmn);
      assert(red < 256 && red >= 0);
      int blue = 255 - red;
      assert(blue < 256 && blue >= 0);
      char prev = std::cout.fill('0');
      of << "set obj rect from "
         << rlo << ", " << clo << " to "
         << rlo+rows() << ", " << clo+cols()
         << " fc rgb '#"
         << std::hex << std::setw(2) << std::setfill('0') << red
         << "00" << std::setw(2)  << std::setfill('0') << blue
         << "'" << std::dec << std::endl;
      std::cout.fill(prev);
    }
    template<typename scalar_t> std::vector<int>
    HMatrixLR<scalar_t>::LU(int task_depth) {
      assert(false);
      return std::vector<int>();
    }
    template<typename scalar_t> void
    HMatrixLR<scalar_t>::laswp(const std::vector<int>& piv, bool fwd) {
      assert(piv.size() == rows());
      _U.laswp(piv, fwd);
    }

  } // end namespace H
} // end namespace strumpack

#endif // H_MATRIX_LR_HPP
