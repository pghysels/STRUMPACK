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
#ifndef HSS_MATRIX_HPP
#define HSS_MATRIX_HPP

#include <cassert>
#include <utility>
#include <fstream>
#include <string>

namespace strumpack {
  namespace H {

    template<typename scalar_t> class HMatrix
      : public HMatrixBase<scalar_t> {
      using opts_t = HOptions<scalar_t>;
      using real_t = typename RealType<scalar_t>::value_type;
      using D_t = DenseMatrix<scalar_t>;
      using DW_t = DenseMatrixWrapper<scalar_t>;
      using HB_t = HMatrixBase<scalar_t>;
      using HLR_t = HMatrixLR<scalar_t>;
      using HD_t = HMatrixDense<scalar_t>;
      using H_t = HMatrix<scalar_t>;

    public:
      HMatrix(std::unique_ptr<HMatrixBase<scalar_t>> H00,
              std::unique_ptr<HMatrixBase<scalar_t>> H01,
              std::unique_ptr<HMatrixBase<scalar_t>> H10,
              std::unique_ptr<HMatrixBase<scalar_t>> H11)
        : _H00(std::move(H00)), _H01(std::move(H01)),
          _H10(std::move(H10)), _H11(std::move(H11)) {}

      std::size_t rows() const { return H00().rows() + H10().rows(); }
      std::size_t cols() const { return H00().cols() + H01().cols(); }
      std::size_t rank() const;
      std::size_t memory() const;
      std::size_t levels() const;
      D_t dense() const;
      std::string name() const { return "HMatrix"; }

      HMatrixBase<scalar_t>& H00() { return *_H00; }
      HMatrixBase<scalar_t>& H01() { return *_H01; }
      HMatrixBase<scalar_t>& H10() { return *_H10; }
      HMatrixBase<scalar_t>& H11() { return *_H11; }
      const HMatrixBase<scalar_t>& H00() const { return *_H00; }
      const HMatrixBase<scalar_t>& H01() const { return *_H01; }
      const HMatrixBase<scalar_t>& H10() const { return *_H10; }
      const HMatrixBase<scalar_t>& H11() const { return *_H11; }

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
      std::unique_ptr<HMatrixBase<scalar_t>> _H00;
      std::unique_ptr<HMatrixBase<scalar_t>> _H01;
      std::unique_ptr<HMatrixBase<scalar_t>> _H10;
      std::unique_ptr<HMatrixBase<scalar_t>> _H11;
      void draw(std::ostream& of, std::size_t rlo=0, std::size_t clo=0) const;
      std::vector<int> LU(int task_depth);
      void laswp(const std::vector<int>& piv, bool fwd);
      template<typename T>
      friend void draw(std::unique_ptr<HMatrixBase<T>> const&,
                       const std::string&);
    };
    template<typename scalar_t> std::size_t
    HMatrix<scalar_t>::rank() const {
      return std::max(std::max(H00().rank(), H01().rank()),
                      std::max(H10().rank(), H11().rank()));
    }
    template<typename scalar_t> std::size_t
    HMatrix<scalar_t>::memory() const {
      return H00().memory() + H01().memory() +
        H10().memory() + H11().memory();
      }
    template<typename scalar_t> std::size_t
    HMatrix<scalar_t>::levels() const {
      return 1 + std::max(std::max(H00().levels(), H01().levels()),
                          std::max(H10().levels(), H11().levels()));
    }
    template<typename scalar_t> DenseMatrix<scalar_t>
    HMatrix<scalar_t>::dense() const {
      D_t out(rows(), cols());
      DW_t out00(H00().rows(), H00().cols(), out, 0, 0);
      DW_t out01(H01().rows(), H01().cols(), out, 0, H00().cols());
      DW_t out10(H10().rows(), H10().cols(), out, H00().rows(), 0);
      DW_t out11(H11().rows(), H11().cols(), out,
                 H00().rows(), H00().cols());
      out00 = H00().dense();
      out01 = H01().dense();
      out10 = H10().dense();
      out11 = H11().dense();
      return out;
    }
    template<typename scalar_t> void
    HMatrix<scalar_t>::draw
    (std::ostream& of, std::size_t rlo, std::size_t clo) const {
      H00().draw(of, rlo, clo);
      H01().draw(of, rlo, clo+H00().cols());
      H10().draw(of, rlo+H00().rows(), clo);
      H11().draw(of, rlo+H00().rows(), clo+H00().cols());
    }
    template<typename scalar_t> std::vector<int>
    HMatrix<scalar_t>::LU(int task_depth) {
      auto piv0 = H00().LU(task_depth);
      H01().laswp(piv0, true);
      // trsm(Side::L, UpLo::L, Trans::N, Diag::U,
      //      scalar_t(1.), H00(), H01(), task_depth);
      // trsm(Side::R, UpLo::U, Trans::N, Diag::N,
      //      scalar_t(1.), H00(), H10(), task_depth);
      gemm(Trans::N, Trans::N, scalar_t(-1.), H10(), H01(),
           scalar_t(1.), H11(), task_depth);
      auto piv1 = H11().LU(task_depth);
      piv1.reserve(piv0.size() + piv1.size());
      auto m0 = H00().rows();
      for (auto i : piv1) piv0.push_back(i + m0);
      return piv0;
    }
    template<typename scalar_t> void
    HMatrix<scalar_t>::laswp(const std::vector<int>& piv, bool fwd) {
      assert(piv.size() == rows());
      std::vector<int> piv0(piv.begin(), piv.begin()+H00().rows());
      std::vector<int> piv1(piv.begin()+H00().rows(), piv.end());
      auto m0 = H00().rows();
      for (auto& i : piv1) i -= m0;
      H00().laswp(piv0, fwd);  H01().laswp(piv0, fwd);
      H10().laswp(piv1, fwd);  H11().laswp(piv1, fwd);
    }

  } // end namespace H
} // end namespace strumpack

#endif // H_MATRIX_HPP
