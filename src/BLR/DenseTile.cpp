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

#include "DenseTile.hpp"
#include "LRTile.hpp"

namespace strumpack {
  namespace BLR {

    template<typename scalar_t> std::unique_ptr<BLRTile<scalar_t>>
    DenseTile<scalar_t>::clone() const {
      return std::unique_ptr<BLRTile<scalar_t>>(new DenseTile(D()));
    }

    template<typename scalar_t> std::unique_ptr<LRTile<scalar_t>>
    DenseTile<scalar_t>::compress(const Opts_t& opts) const {
      return std::unique_ptr<LRTile<scalar_t>>
        (new LRTile<scalar_t>(D(), opts));
    }

    template<typename scalar_t> void
    DenseTile<scalar_t>::copy_to(scalar_t*& ptr) const {
      std::copy(D().data(), D().end(), ptr);
      ptr += rows()*cols();
    }

#if defined(STRUMPACK_USE_GPU)
    template<typename scalar_t> void
    DenseTile<scalar_t>::copy_from_device_to(scalar_t*& ptr) const {
      gpu::copy(ptr, D());
      ptr += rows()*cols();
    }
#endif

    template<typename scalar_t> LRTile<scalar_t>
    DenseTile<scalar_t>::multiply(const BLRTile<scalar_t>& a) const {
      return a.left_multiply(*this);
    }
    template<typename scalar_t> LRTile<scalar_t>
    DenseTile<scalar_t>::left_multiply(const LRTile<scalar_t>& a) const {
      // a.U* (a.V*D)
      LRTile<scalar_t> t(a.rows(), cols(), a.rank());
      left_multiply(a, t.U(), t.V());
      return t;
    }

    template<typename scalar_t> LRTile<scalar_t>
    DenseTile<scalar_t>::left_multiply(const DenseTile<scalar_t>& a) const {
      assert(false);
      return LRTile<scalar_t>(0,0,0);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::multiply
    (const BLRTile<scalar_t>& a, DenseM_t& b, DenseM_t& c) const {
      a.left_multiply(*this, b, c);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::left_multiply
    (const LRTile<scalar_t>& a, DenseM_t& b, DenseM_t& c) const {
      // a.U* (a.V*D)
      gemm(Trans::N, Trans::N, scalar_t(1.), a.V(), D(), scalar_t(0.),
           c, params::task_recursion_cutoff_level);
      copy(a.U(), b, 0, 0);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::left_multiply
    (const DenseTile<scalar_t>& a, DenseM_t& b, DenseM_t& c) const{
      assert(false);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::draw
    (std::ostream& of, std::size_t roff, std::size_t coff) const {
      char prev = std::cout.fill('0');
      of << "set obj rect from "
         << roff << ", " << coff << " to "
         << roff+rows() << ", " << coff+cols()
         << " fc rgb '#FF0000'" << std::endl;
      std::cout.fill(prev);
    }

    template<typename scalar_t> std::vector<int>
    DenseTile<scalar_t>::LU(real_t thresh) {
      auto& A = D();
      auto piv = A.LU(params::task_recursion_cutoff_level);
      if (thresh > 0.)
        for (std::size_t i=0; i<rows(); i++)
          if (std::abs(A(i,i)) < thresh)
            A(i,i) = (std::real(A(i,i)) < 0) ? -thresh : thresh;
      return piv;
    }

    template<typename scalar_t> void DenseTile<scalar_t>::laswp
    (const std::vector<int>& piv, bool fwd) {
      D().laswp(piv, fwd);
    }
#if defined(STRUMPACK_USE_GPU)
    template<typename scalar_t> void
    DenseTile<scalar_t>::laswp(gpu::Handle& h, int* dpiv, bool fwd) {
      gpu::laswp(h, D(), 1, D().rows(), dpiv, fwd ? 1 : -1);
    }

    template<typename scalar_t> void
    DenseTile<scalar_t>::move_to_cpu(gpu::Stream& s, scalar_t* pinned) {
      DenseM_t hD(rows(), cols());
      if (!pinned)
        gpu::copy(hD, D());
      else {
        DenseMW_t pD(rows(), cols(), pinned, rows());
        gpu::copy_async(pD, D(), s);
        s.synchronize();
        hD.copy(pD);
      }
      D_.reset(new DenseM_t(std::move(hD)));
    }

    template<typename scalar_t> void
    DenseTile<scalar_t>::move_to_gpu(gpu::Stream& s, scalar_t* dptr,
                                     scalar_t* pinned) {
      DenseMW_t dD(rows(), cols(), dptr, rows());
      if (!pinned)
        gpu::copy(dD, D());
      else {
        DenseMW_t hD(rows(), cols(), pinned, rows());
        hD.copy(D());
        gpu::copy_async(dD, hD, s);
        s.synchronize();
      }
      D_.reset(new DenseMW_t(std::move(dD)));
    }
#endif

    template<typename scalar_t> void DenseTile<scalar_t>::trsm_b
    (Side s, UpLo ul, Trans ta, Diag d,
     scalar_t alpha, const DenseM_t& a) {
      trsm(s, ul, ta, d, alpha, a, D(), params::task_recursion_cutoff_level);
    }

#if defined(STRUMPACK_USE_GPU)
    template<typename scalar_t> void DenseTile<scalar_t>::trsm_b
    (gpu::Handle& handle, Side s, UpLo ul, Trans ta,
     Diag d, scalar_t alpha, DenseM_t& a) {
      strumpack::gpu::trsm(handle, s, ul, ta, d, alpha, a, D());
    }
#endif

    template<typename scalar_t> void DenseTile<scalar_t>::gemv_a
    (Trans ta, scalar_t alpha, const DenseM_t& x,
     scalar_t beta, DenseM_t& y) const {
      gemv(ta, alpha, D(), x, beta, y,
           params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::gemm_a
    (Trans ta, Trans tb, scalar_t alpha, const BLRT_t& b,
     scalar_t beta, DenseM_t& c) const {
      b.gemm_b(ta, tb, alpha, *this, beta, c);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::gemm_a
    (Trans ta, Trans tb, scalar_t alpha,
     const DenseM_t& b, scalar_t beta,
     DenseM_t& c, int task_depth) const {
      gemm(ta, tb, alpha, D(), b, beta, c, task_depth);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::gemm_b
    (Trans ta, Trans tb, scalar_t alpha,
     const LRTile<scalar_t>& a, scalar_t beta,
     DenseM_t& c) const {
      DenseM_t tmp(a.rank(), tb==Trans::N ? cols() : rows());
      gemm(ta, tb, scalar_t(1.), ta==Trans::N ? a.V() : a.U(), D(),
           scalar_t(0.), tmp, params::task_recursion_cutoff_level);
      gemm(ta, Trans::N, alpha, ta==Trans::N ? a.U() : a.V(), tmp,
           beta, c, params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::gemm_b
    (Trans ta, Trans tb, scalar_t alpha,
     const DenseTile<scalar_t>& a, scalar_t beta,
     DenseM_t& c) const {
      gemm(ta, tb, alpha, a.D(), D(), beta, c);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::gemm_b
    (Trans ta, Trans tb, scalar_t alpha,
     const DenseM_t& a, scalar_t beta,
     DenseM_t& c, int task_depth) const {
      gemm(ta, tb, alpha, a, D(), beta, c, task_depth);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::Schur_update_col_a
    (std::size_t i, const BLRTile<scalar_t>& b, scalar_t* c,
     scalar_t* work) const {
      b.Schur_update_col_b(i, *this, c, work);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::Schur_update_row_a
    (std::size_t i, const BLRTile<scalar_t>& b, scalar_t* c,
     scalar_t* work) const {
      b.Schur_update_row_b(i, *this, c, work);
    }

    /* work should be at least rank(a) */
    template<typename scalar_t> void DenseTile<scalar_t>::Schur_update_col_b
    (std::size_t i, const LRTile<scalar_t>& a, scalar_t* c,
     scalar_t* work) const {
      DenseMW_t temp(a.rank(), 1, work, a.rank());
      gemv(Trans::N, scalar_t(1.), a.V(), D().ptr(0, i), 1,
           scalar_t(0.), temp, params::task_recursion_cutoff_level);
      gemv(Trans::N, scalar_t(-1.), a.U(), temp,
           scalar_t(1.), c, 1, params::task_recursion_cutoff_level);
    }

    /* work not used */
    template<typename scalar_t> void DenseTile<scalar_t>::Schur_update_col_b
    (std::size_t i, const DenseTile<scalar_t>& a, scalar_t* c,
     scalar_t* work) const {
      gemv(Trans::N, scalar_t(-1.), a.D(), D().ptr(0, i), 1,
           scalar_t(1.), c, 1, params::task_recursion_cutoff_level);
    }

    /* work should be at least cols(a) */
    template<typename scalar_t> void DenseTile<scalar_t>::Schur_update_row_b
    (std::size_t i, const LRTile<scalar_t>& a, scalar_t* c,
     scalar_t* work) const {
      DenseMW_t temp(1, a.cols(), work, 1);
      gemv(Trans::C, scalar_t(1.), a.V(), a.U().ptr(i, 0), a.U().ld(),
           scalar_t(0.), temp.data(), temp.ld(),
           params::task_recursion_cutoff_level);
      gemv(Trans::C, scalar_t(-1.), D(), temp.data(), temp.ld(),
           scalar_t(1.), c, 1, params::task_recursion_cutoff_level);
    }

    /* work not used */
    template<typename scalar_t> void DenseTile<scalar_t>::Schur_update_row_b
    (std::size_t i, const DenseTile<scalar_t>& a, scalar_t* c,
     scalar_t* work) const {
      gemv(Trans::C, scalar_t(-1.), D(), a.D().ptr(i, 0), a.D().ld(),
           scalar_t(1), c, 1, params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::Schur_update_cols_a
    (const std::vector<std::size_t>& cols, const BLRTile<scalar_t>& b,
     DenseMatrix<scalar_t>& c, scalar_t* work) const {
      b.Schur_update_cols_b(cols, *this, c, work);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::Schur_update_rows_a
    (const std::vector<std::size_t>& rows, const BLRTile<scalar_t>& b,
     DenseMatrix<scalar_t>& c, scalar_t* work) const {
      b.Schur_update_rows_b(rows, *this, c, work);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::Schur_update_cols_b
    (const std::vector<std::size_t>& cols, const LRTile<scalar_t>& a,
     DenseMatrix<scalar_t>& c, scalar_t* work) const {
      auto m = rows(); auto d = cols.size();
      DenseMW_t Dc(m, d, work, m), temp(a.rank(), d, Dc.end(), a.rank());
      D().extract_cols(cols, Dc);
      gemm(Trans::N, Trans::N, scalar_t(1.), a.V(), Dc,
           scalar_t(0.), temp, params::task_recursion_cutoff_level);
      gemm(Trans::N, Trans::N, scalar_t(-1.), a.U(), temp,
           scalar_t(1.), c, params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::Schur_update_cols_b
    (const std::vector<std::size_t>& cols, const DenseTile<scalar_t>& a,
     DenseMatrix<scalar_t>& c, scalar_t* work) const {
      auto d = cols.size(); auto m = rows();
      DenseMW_t Dc(m, d, work, m);
      D().extract_cols(cols, Dc);
      gemm(Trans::N, Trans::N, scalar_t(-1.), a.D(), Dc, scalar_t(1.), c,
           params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::Schur_update_rows_b
    (const std::vector<std::size_t>& rows, const LRTile<scalar_t>& a,
     DenseMatrix<scalar_t>& c, scalar_t* work) const {
      auto d = rows.size();
      DenseMW_t aUr(d, a.rank(), work, d), temp(d, a.cols(), aUr.end(), d);
      a.U().extract_rows(rows, aUr);
      gemm(Trans::N, Trans::N, scalar_t(1.), aUr,
           a.V(), scalar_t(0.), temp, params::task_recursion_cutoff_level);
      gemm(Trans::N, Trans::N, scalar_t(-1.), temp, D(),
           scalar_t(1.), c, params::task_recursion_cutoff_level);
    }

    template<typename scalar_t> void DenseTile<scalar_t>::Schur_update_rows_b
    (const std::vector<std::size_t>& rows, const DenseTile<scalar_t>& a,
     DenseMatrix<scalar_t>& c, scalar_t* work) const {
      auto d = rows.size();
      DenseMW_t aDr(d, a.cols(), work, d);
      a.D().extract_rows(rows, aDr);
      gemm(Trans::N, Trans::N, scalar_t(-1.), aDr, D(), scalar_t(1), c,
           params::task_recursion_cutoff_level);
    }

    // explicit template instantiations
    template class DenseTile<float>;
    template class DenseTile<double>;
    template class DenseTile<std::complex<float>>;
    template class DenseTile<std::complex<double>>;

  } // end namespace BLR
} // end namespace strumpack
