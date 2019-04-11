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
/*! \file DenseTile.hpp
 * \brief Contains the DenseTile class, subclass of BLRTile.
 */
#ifndef DENSE_TILE_HPP
#define DENSE_TILE_HPP

#include <cassert>

#include "BLRTile.hpp"

namespace strumpack {
  namespace BLR {

    template<typename scalar_t> class DenseTile
      : public BLRTile<scalar_t> {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using BLRT_t = BLRTile<scalar_t>;

    public:
      DenseTile(std::size_t m, std::size_t n) : D_(m, n) {}
      DenseTile(const DenseM_t& D) : D_(D) {}

      std::size_t rows() const override { return D_.rows(); }
      std::size_t cols() const override { return D_.cols(); }
      std::size_t rank() const override { return std::min(rows(), cols()); }

      std::size_t memory() const override { return D_.memory(); }
      std::size_t nonzeros() const override { return D_.nonzeros(); }
      std::size_t maximum_rank() const override { return 0; }
      bool is_low_rank() const override { return false; };

      void dense(DenseM_t& A) const override { A = D_; }

      void draw
      (std::ostream& of, std::size_t roff, std::size_t coff) const override {
        char prev = std::cout.fill('0');
        of << "set obj rect from "
           << roff << ", " << coff << " to "
           << roff+rows() << ", " << coff+cols()
           << " fc rgb '#FF0000'" << std::endl;
        std::cout.fill(prev);
      }

      DenseM_t& D() override { return D_; }
      const DenseM_t& D() const override { return D_; }

      DenseM_t& U() override { assert(false); return D_; }
      DenseM_t& V() override { assert(false); return D_; }
      const DenseM_t& U() const override { assert(false); return D_; }
      const DenseM_t& V() const override { assert(false); return D_; }

      scalar_t operator()(std::size_t i, std::size_t j) const override {
        return D_(i, j);
      }

      std::vector<int> LU() override {
        return D_.LU(params::task_recursion_cutoff_level);
      }
      void laswp(const std::vector<int>& piv, bool fwd) override {
        D_.laswp(piv, fwd);
      }

      void trsm_b(Side s, UpLo ul, Trans ta, Diag d,
                  scalar_t alpha, const DenseM_t& a) override {
        trsm(s, ul, ta, d, alpha, a, D_, params::task_recursion_cutoff_level);
      }
      void gemv_a(Trans ta, scalar_t alpha, const DenseM_t& x,
                  scalar_t beta, DenseM_t& y) const override {
        gemv(ta, alpha, D_, x, beta, y,
             params::task_recursion_cutoff_level);
      }
      void gemm_a(Trans ta, Trans tb, scalar_t alpha, const BLRT_t& b,
                  scalar_t beta, DenseM_t& c) const override {
        b.gemm_b(ta, tb, alpha, *this, beta, c);
      }
      void gemm_a(Trans ta, Trans tb, scalar_t alpha,
                  const DenseM_t& b, scalar_t beta,
                  DenseM_t& c, int task_depth) const override {
        gemm(ta, tb, alpha, D_, b, beta, c, task_depth);
      }
      void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                  const LRTile<scalar_t>& a, scalar_t beta,
                  DenseM_t& c) const override {
        DenseM_t tmp(a.rank(), tb==Trans::N ? cols() : rows());
        gemm(ta, tb, scalar_t(1.), ta==Trans::N ? a.V() : a.U(), D_,
             scalar_t(0.), tmp, params::task_recursion_cutoff_level);
        gemm(ta, Trans::N, alpha, ta==Trans::N ? a.U() : a.V(), tmp,
             beta, c, params::task_recursion_cutoff_level);
      }
      void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                  const DenseTile<scalar_t>& a, scalar_t beta,
                  DenseM_t& c) const override {
        gemm(ta, tb, alpha, a.D(), D(), beta, c);
      }
      void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                  const DenseM_t& a, scalar_t beta,
                  DenseM_t& c, int task_depth) const override {
        gemm(ta, tb, alpha, a, D_, beta, c, task_depth);
      }

      void Schur_update_col_a
      (std::size_t i, const BLRTile<scalar_t>& b, scalar_t* c, int incc)
        const override {
        b.Schur_update_col_b(i, *this, c, incc);
      }
      virtual void Schur_update_row_a
      (std::size_t i, const BLRTile<scalar_t>& b, scalar_t* c, int incc)
        const override {
        b.Schur_update_row_b(i, *this, c, incc);
      }

      virtual void Schur_update_col_b
      (std::size_t i, const LRTile<scalar_t>& a, scalar_t* c, int incc)
        const override {
        DenseM_t temp(a.rank(), 1);
        gemv(Trans::N, scalar_t(1.), a.V(), D_.ptr(0, i), 1,
             scalar_t(0.), temp, params::task_recursion_cutoff_level);
        gemv(Trans::N, scalar_t(-1.), a.U(), temp,
             scalar_t(1.), c, incc, params::task_recursion_cutoff_level);
      }
      virtual void Schur_update_col_b
      (std::size_t i, const DenseTile<scalar_t>& a, scalar_t* c, int incc)
        const override {
        gemv(Trans::N, scalar_t(-1.), a.D(), D_.ptr(0, i), 1,
             scalar_t(1.), c, incc, params::task_recursion_cutoff_level);
      }
      virtual void Schur_update_row_b
      (std::size_t i, const LRTile<scalar_t>& a, scalar_t* c, int incc)
        const override {
        DenseM_t temp(1, a.cols());
        gemv(Trans::C, scalar_t(1.), a.V(), a.U().ptr(i, 0), a.U().ld(),
             scalar_t(0.), temp.data(), temp.ld(),
             params::task_recursion_cutoff_level);
        gemv(Trans::C, scalar_t(-1.), D_, temp.data(), temp.ld(),
             scalar_t(1.), c, incc, params::task_recursion_cutoff_level);
      }
      virtual void Schur_update_row_b
      (std::size_t i, const DenseTile<scalar_t>& a, scalar_t* c, int incc)
        const override {
        gemv(Trans::C, scalar_t(-1.), D_, a.D().ptr(i, 0), a.D().ld(),
             scalar_t(1), c, incc, params::task_recursion_cutoff_level);
      }

    private:
      DenseM_t D_;
    };


  } // end namespace BLR
} // end namespace strumpack

#endif // DENSE_TILE_HPP
