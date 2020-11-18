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
/*! \file LRTile.hpp
 * \brief Contains LRTile class, subclass of BLRTile.
 */
#ifndef LR_TILE_HPP
#define LR_TILE_HPP

#include <functional>

#include "BLRTile.hpp"
#include "BLROptions.hpp"
#include "dense/DenseMatrix.hpp"

namespace strumpack {
  namespace BLR {

    /**
     * Low rank U*V tile
     */
    template<typename scalar_t> class LRTile
      : public BLRTile<scalar_t> {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DMW_t = DenseMatrixWrapper<scalar_t>;
      using Opts_t = BLROptions<scalar_t>;

    public:
      LRTile(std::size_t m, std::size_t n, std::size_t r);

      LRTile(const DenseM_t& T, const Opts_t& opts);

      /**
       * .. by extracting individual elements
       */
      LRTile(std::size_t m, std::size_t n,
             const std::function<scalar_t(std::size_t,std::size_t)>& Telem,
             const Opts_t& opts);

      /**
       * .. by extracting 1 column or 1 row at a time
       */
      LRTile(std::size_t m, std::size_t n,
             const std::function<void(std::size_t,scalar_t*)>& Trow,
             const std::function<void(std::size_t,scalar_t*)>& Tcol,
             const Opts_t& opts);

      /**
       * .. by extracting multiple columns or rows at a time
       */
      LRTile(std::size_t m, std::size_t n,
             const std::function<void(const std::vector<std::size_t>&,
                                      DenseMatrix<scalar_t>&)>& Trow,
             const std::function<void(const std::vector<std::size_t>&,
                                      DenseMatrix<scalar_t>&)>& Tcol,
             const Opts_t& opts);

      std::size_t rows() const override { return U_.rows(); }
      std::size_t cols() const override { return V_.cols(); }
      std::size_t rank() const override { return U_.cols(); }
      bool is_low_rank() const override { return true; };

      std::size_t memory() const override { return U_.memory() + V_.memory(); }
      std::size_t nonzeros() const override { return U_.nonzeros() + V_.nonzeros(); }
      std::size_t maximum_rank() const override { return U_.cols(); }

      void dense(DenseM_t& A) const override;
      DenseM_t dense() const override;

      std::unique_ptr<BLRTile<scalar_t>> clone() const override;

      std::unique_ptr<LRTile<scalar_t>>
      compress(const Opts_t& opts) const override {
        assert(false);
        return nullptr;
      };

      void draw(std::ostream& of, std::size_t roff,
                std::size_t coff) const override;

      DenseM_t& D() override { assert(false); return U_; }
      DenseM_t& U() override { return U_; }
      DenseM_t& V() override { return V_; }
      const DenseM_t& D() const override { assert(false); return U_; }
      const DenseM_t& U() const override { return U_; }
      const DenseM_t& V() const override { return V_; }

      LRTile<scalar_t> multiply(const BLRTile<scalar_t>& a) const override;
      LRTile<scalar_t> left_multiply(const LRTile<scalar_t>& a) const override;
      LRTile<scalar_t> left_multiply(const DenseTile<scalar_t>& a) const override;

      void multiply(const BLRTile<scalar_t>& a,
                    DenseM_t& b, DenseM_t& c) const override;
      void left_multiply(const LRTile<scalar_t>& a,
                         DenseM_t& b, DenseM_t& c) const override;
      void left_multiply(const DenseTile<scalar_t>& a,
                         DenseM_t& b, DenseM_t& c) const override;

      scalar_t operator()(std::size_t i, std::size_t j) const override;

      void extract(const std::vector<std::size_t>& I,
                   const std::vector<std::size_t>& J,
                   DenseM_t& B) const override;

      void laswp(const std::vector<int>& piv, bool fwd) override;

      void trsm_b(Side s, UpLo ul, Trans ta, Diag d,
                  scalar_t alpha, const DenseM_t& a) override;

      void gemv_a(Trans ta, scalar_t alpha, const DenseM_t& x,
                  scalar_t beta, DenseM_t& y) const override;

      void gemm_a(Trans ta, Trans tb, scalar_t alpha,
                  const BLRTile<scalar_t>& b,
                  scalar_t beta, DenseM_t& c) const override;

      void gemm_a(Trans ta, Trans tb, scalar_t alpha,
                  const DenseM_t& b, scalar_t beta,
                  DenseM_t& c, int task_depth) const override;

      void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                  const LRTile<scalar_t>& a, scalar_t beta,
                  DenseM_t& c) const override;

      void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                  const DenseTile<scalar_t>& a, scalar_t beta,
                  DenseM_t& c) const override;

      void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                  const DenseM_t& a, scalar_t beta,
                  DenseM_t& c, int task_depth) const override;

      void Schur_update_col_a(std::size_t i, const BLRTile<scalar_t>& b,
                              scalar_t* c, scalar_t* work) const override;

      void Schur_update_row_a(std::size_t i, const BLRTile<scalar_t>& b,
                              scalar_t* c, scalar_t* work) const override;


      /* work should be at least rank(a) + rows(b) */
      void Schur_update_col_b(std::size_t i, const LRTile<scalar_t>& a,
                              scalar_t* c, scalar_t* work) const override;

      /* work should be at least rows(b) */
      void Schur_update_col_b(std::size_t i, const DenseTile<scalar_t>& a,
                              scalar_t* c, scalar_t* work) const override;

      /* work should be at least cols(a) + rank(b) */
      void Schur_update_row_b(std::size_t i, const LRTile<scalar_t>& a,
                              scalar_t* c, scalar_t* work) const override;

      /* work should be at least rank(b) */
      void Schur_update_row_b(std::size_t i, const DenseTile<scalar_t>& a,
                              scalar_t* c, scalar_t* work) const override;

      void Schur_update_cols_a(const std::vector<std::size_t>& cols,
                               const BLRTile<scalar_t>& b,
                               DenseMatrix<scalar_t>& c,
                               scalar_t* work) const override;

      void Schur_update_rows_a(const std::vector<std::size_t>& rows,
                               const BLRTile<scalar_t>& b,
                               DenseMatrix<scalar_t>& c,
                               scalar_t* work) const override;

      void Schur_update_cols_b(const std::vector<std::size_t>& cols,
                               const LRTile<scalar_t>& a,
                               DenseMatrix<scalar_t>& c,
                               scalar_t* work) const override;

      void Schur_update_cols_b(const std::vector<std::size_t>& cols,
                               const DenseTile<scalar_t>& a,
                               DenseMatrix<scalar_t>& c,
                               scalar_t* work) const override;

      void Schur_update_rows_b(const std::vector<std::size_t>& rows,
                               const LRTile<scalar_t>& a,
                               DenseMatrix<scalar_t>& c,
                               scalar_t* work) const override;

      void Schur_update_rows_b(const std::vector<std::size_t>& rows,
                               const DenseTile<scalar_t>& a,
                               DenseMatrix<scalar_t>& c,
                               scalar_t* work) const override;

    private:
      DenseM_t U_, V_;
    };


  } // end namespace BLR
} // end namespace strumpack

#endif // LR_TILE_HPP
