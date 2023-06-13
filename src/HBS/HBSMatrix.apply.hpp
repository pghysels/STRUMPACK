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
 */
#ifndef HBS_MATRIX_APPLY_HPP
#define HBS_MATRIX_APPLY_HPP

namespace strumpack {
  namespace HBS {

    template<typename scalar_t> class WorkApply {
    public:
      std::pair<std::size_t,std::size_t> offset;
      std::vector<WorkApply<scalar_t>> c;
      int lvl = 0;
      DenseMatrix<scalar_t> q, u;

      void split(const std::pair<std::size_t,std::size_t>& dim) {
        if (c.empty()) {
          c.resize(2);
          c[0].offset = offset;
          // why does this work in HSS, not here?
          // c[1].offset = offset + dim;
          c[1].offset.first = offset.first + dim.first;
          c[1].offset.second = offset.second + dim.second;
          c[0].lvl = c[1].lvl = lvl + 1;
        }
      }
    };

    template<typename scalar_t> void HBSMatrix<scalar_t>::mult
    (Trans op, const DenseM_t& x, DenseM_t& y) const {
      apply_HBS(op, *this, x, scalar_t(0.), y);
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    HBSMatrix<scalar_t>::apply(const DenseM_t& b) const {
      assert(this->cols() == b.rows());
      DenseM_t c(this->rows(), b.cols());
      apply_HBS(Trans::N, *this, b, scalar_t(0.), c);
      return c;
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    HBSMatrix<scalar_t>::applyC(const DenseM_t& b) const {
      assert(this->rows() == b.rows());
      DenseM_t c(this->cols(), b.cols());
      apply_HBS(Trans::C, *this, b, scalar_t(0.), c);
      return c;
    }

    template<typename scalar_t> void
    HBSMatrix<scalar_t>::apply_fwd(const DenseM_t& b, WorkApply<scalar_t>& w,
                                   int depth) const {
      if (this->leaf()) {
        auto bloc = ConstDenseMatrixWrapperPtr
          (rows(), b.cols(), b, w.offset.second, 0);
        w.q = gemm(Trans::C, Trans::N, scalar_t(1.), V_, *bloc, depth);
      } else {
        w.split(child(0)->dims());
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        child(0)->apply_fwd(b, w.c[0], depth+1);
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        child(1)->apply_fwd(b, w.c[1], depth+1);
#pragma omp taskwait
        if (w.lvl != 0) {
          auto qq = vconcat(w.c[0].q, w.c[1].q);
          w.q = gemm(Trans::C, Trans::N, scalar_t(1.), V_, qq);
        }
      }
    }

    template<typename scalar_t> void
    HBSMatrix<scalar_t>::apply_bwd(const DenseM_t& b, scalar_t beta,
                                   DenseM_t& c, WorkApply<scalar_t>& w,
                                   int depth) const {
      if (this->leaf()) {
        DenseMW_t cloc(rows(), c.cols(), c, w.offset.second, 0);
        if (w.lvl != 0)
          gemm(Trans::N, Trans::N, scalar_t(1.), U_, w.u, beta, cloc, depth);
        auto bloc = ConstDenseMatrixWrapperPtr
          (rows(), b.cols(), b, w.offset.second, 0);
        gemm(Trans::N, Trans::N, scalar_t(1.), D_, *bloc,
             scalar_t(1.), cloc, depth);
      } else {
        auto qq = vconcat(w.c[0].q, w.c[1].q);
        auto uu = gemm(Trans::N, Trans::N, scalar_t(1.), D_, qq);
        if (w.lvl != 0)
          gemm(Trans::N, Trans::N, scalar_t(1.), U_, w.u, beta, uu, depth);
        w.c[0].u = DenseM_t(child(0)->U_.rows(), uu.cols(), uu, 0, 0);
        w.c[1].u = DenseM_t(child(1)->U_.rows(), uu.cols(),
                            uu, child(0)->U_.rows(), 0);
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        child(0)->apply_bwd(b, beta, c, w.c[0], depth+1);
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        child(1)->apply_bwd(b, beta, c, w.c[1], depth+1);
#pragma omp taskwait
      }
    }

  } // end namespace HBS
} // end namespace strumpack

#endif // HBS_MATRIX_APPLY_HPP
