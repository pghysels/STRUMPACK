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
/*! \file BLRTileBLAS.hpp
 * \brief Contains BLAS routines on BLRTiles.
 */
#ifndef BLR_TILE_BLAS_HPP
#define BLR_TILE_BLAS_HPP

#include <cassert>

#include "LRTile.hpp"
#include "DenseTile.hpp"

namespace strumpack {
  namespace BLR {

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const BLRTile<scalar_t>& a,
         const BLRTile<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c) {
      a.gemm_a(ta, tb, alpha, b, beta, c);
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const BLRTile<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c, int task_depth) {
      b.gemm_b(ta, tb, alpha, a, beta, c);
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const BLRTile<scalar_t>& a,
         const DenseMatrix<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c, int task_depth) {
      a.gemm_a(ta, tb, alpha, b, beta, c, task_depth);
    }

    template<typename scalar_t> void
    trsm(Side s, UpLo ul, Trans ta, Diag d, scalar_t alpha,
         const BLRTile<scalar_t>& a, BLRTile<scalar_t>& b) {
      b.trsm_b(s, ul, ta, d, alpha, a.D());
    }

    template<typename scalar_t> void
    trsm(Side s, UpLo ul, Trans ta, Diag d, scalar_t alpha,
         const BLRTile<scalar_t>& a, DenseMatrix<scalar_t>& b,
         int task_depth) {
      trsm(s, ul, ta, d, alpha, a.D(), b, task_depth);
    }

    template<typename scalar_t> void Schur_update_col
    (std::size_t j, const BLRTile<scalar_t>& a,
     const BLRTile<scalar_t>& b, scalar_t* c, scalar_t* work) {
      a.Schur_update_col_a(j, b, c, work);
    }

    template<typename scalar_t> void Schur_update_row
    (std::size_t i, const BLRTile<scalar_t>& a,
     const BLRTile<scalar_t>& b, scalar_t* c, scalar_t* work) {
      a.Schur_update_row_a(i, b, c, work);
    }

    template<typename scalar_t> void Schur_update_cols
    (const std::vector<std::size_t>& cols, const BLRTile<scalar_t>& a,
     const BLRTile<scalar_t>& b, DenseMatrix<scalar_t>& c, scalar_t* work) {
      a.Schur_update_cols_a(cols, b, c, work);
    }

    template<typename scalar_t> void Schur_update_rows
    (const std::vector<std::size_t>& rows, const BLRTile<scalar_t>& a,
     const BLRTile<scalar_t>& b, DenseMatrix<scalar_t>& c, scalar_t* work) {
      a.Schur_update_rows_a(rows, b, c, work);
    }

  } // end namespace BLR
} // end namespace strumpack

#endif // BLR_TILE_BLAS_HPP
