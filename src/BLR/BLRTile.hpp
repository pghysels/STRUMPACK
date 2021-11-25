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
/*! \file BLRTile.hpp
 * \brief Contains the pure virtual BLRTile class.
 */
#ifndef BLR_TILE_HPP
#define BLR_TILE_HPP

#include <cassert>

#include "dense/DenseMatrix.hpp"
#include "BLROptions.hpp"

namespace strumpack {
  namespace BLR {

    // forward declarations
    template<typename scalar_t> class LRTile;
    template<typename scalar_t> class DenseTile;

    template<typename scalar_t> class BLRTile {
      using DenseM_t = DenseMatrix<scalar_t>;
      using Opts_t = BLROptions<scalar_t>;
      using DMW_t = DenseMatrixWrapper<scalar_t>;

    public:
      virtual ~BLRTile() = default;

      virtual std::size_t rows() const = 0;
      virtual std::size_t cols() const = 0;
      virtual std::size_t rank() const = 0;

      virtual std::size_t memory() const = 0;
      virtual std::size_t nonzeros() const = 0;
      virtual std::size_t maximum_rank() const = 0;
      virtual bool is_low_rank() const = 0;
      virtual void dense(DenseM_t& A) const = 0;
      virtual DenseM_t dense() const = 0;

      virtual std::unique_ptr<BLRTile<scalar_t>> clone() const = 0;

      virtual std::unique_ptr<LRTile<scalar_t>>
      compress(const Opts_t& opts) const = 0;

      virtual void draw(std::ostream& of,
                        std::size_t roff, std::size_t coff) const = 0;

      virtual DenseM_t& D() = 0; //{ assert(false); }
      virtual DenseM_t& U() = 0; //{ assert(false); }
      virtual DenseM_t& V() = 0; //{ assert(false); }
      virtual const DenseM_t& D() const = 0; //{ assert(false); }
      virtual const DenseM_t& U() const = 0; //{ assert(false); }
      virtual const DenseM_t& V() const = 0; //{ assert(false); }

      virtual LRTile<scalar_t> multiply(const BLRTile<scalar_t>& a) const=0;
      virtual LRTile<scalar_t> left_multiply(const LRTile<scalar_t>& a) const=0;
      virtual LRTile<scalar_t> left_multiply(const DenseTile<scalar_t>& a) const=0;

      virtual void multiply(const BLRTile<scalar_t>& a,
                            DenseM_t& b, DenseM_t& c) const=0;
      virtual void left_multiply(const LRTile<scalar_t>& a,
                                 DenseM_t& b, DenseM_t& c) const=0;
      virtual void left_multiply(const DenseTile<scalar_t>& a,
                                 DenseM_t& b, DenseM_t& c) const=0;

      virtual scalar_t operator()(std::size_t i, std::size_t j) const = 0;
      virtual void extract(const std::vector<std::size_t>& I,
                           const std::vector<std::size_t>& J,
                           DenseM_t& B) const {
        assert(B.rows() == I.size() && B.cols() == J.size());
        for (std::size_t j=0; j<J.size(); j++)
          for (std::size_t i=0; i<I.size(); i++) {
            assert(I[i] < rows());
            assert(J[j] < cols());
            B(i, j) = operator()(I[i], J[j]);
          }
      }

      virtual std::vector<int> LU() { assert(false); return std::vector<int>(); };
      virtual void laswp(const std::vector<int>& piv, bool fwd) = 0;

      virtual void trsm_b(Side s, UpLo ul, Trans ta, Diag d,
                          scalar_t alpha, const DenseM_t& a) = 0;
      virtual void gemv_a(Trans ta, scalar_t alpha, const DenseM_t& x,
                          scalar_t beta, DenseM_t& y) const = 0;
      virtual void gemm_a(Trans ta, Trans tb, scalar_t alpha,
                          const BLRTile<scalar_t>& b, scalar_t beta,
                          DenseM_t& c) const = 0;
      virtual void gemm_a(Trans ta, Trans tb, scalar_t alpha,
                          const DenseM_t& b, scalar_t beta,
                          DenseM_t& c, int task_depth) const = 0;
      virtual void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                          const LRTile<scalar_t>& a, scalar_t beta,
                          DenseM_t& c) const = 0;
      virtual void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                          const DenseTile<scalar_t>& a, scalar_t beta,
                          DenseM_t& c) const = 0;
      virtual void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                          const DenseM_t& a, scalar_t beta,
                          DenseM_t& c, int task_depth) const = 0;

      virtual void Schur_update_col_a
      (std::size_t i, const BLRTile<scalar_t>& b, scalar_t* c,
       scalar_t* work) const = 0;
      virtual void Schur_update_row_a
      (std::size_t i, const BLRTile<scalar_t>& b, scalar_t* c,
       scalar_t* work) const = 0;
      virtual void Schur_update_col_b
      (std::size_t i, const LRTile<scalar_t>& a, scalar_t* c,
       scalar_t* work) const = 0;
      virtual void Schur_update_col_b
      (std::size_t i, const DenseTile<scalar_t>& a, scalar_t* c,
       scalar_t* work) const = 0;
      virtual void Schur_update_row_b
      (std::size_t i, const LRTile<scalar_t>& a, scalar_t* c,
       scalar_t* work) const = 0;
      virtual void Schur_update_row_b
      (std::size_t i, const DenseTile<scalar_t>& a, scalar_t* c,
       scalar_t* work) const = 0;

      virtual void Schur_update_cols_a
      (const std::vector<std::size_t>& cols, const BLRTile<scalar_t>& b,
       DenseMatrix<scalar_t>& c, scalar_t* work) const = 0;
      virtual void Schur_update_rows_a
      (const std::vector<std::size_t>& rows, const BLRTile<scalar_t>& b,
       DenseMatrix<scalar_t>& c, scalar_t* work) const = 0;
      virtual void Schur_update_cols_b
      (const std::vector<std::size_t>& cols, const LRTile<scalar_t>& a,
       DenseMatrix<scalar_t>& c, scalar_t* work) const = 0;
      virtual void Schur_update_cols_b
      (const std::vector<std::size_t>& cols, const DenseTile<scalar_t>& a,
       DenseMatrix<scalar_t>& c, scalar_t* work) const = 0;
      virtual void Schur_update_rows_b
      (const std::vector<std::size_t>& rows, const LRTile<scalar_t>& a,
       DenseMatrix<scalar_t>& c, scalar_t* work) const = 0;
      virtual void Schur_update_rows_b
      (const std::vector<std::size_t>& rows, const DenseTile<scalar_t>& a,
       DenseMatrix<scalar_t>& c, scalar_t* work) const = 0;
    };

  } // end namespace BLR
} // end namespace strumpack

#endif // BLR_TILE_HPP
