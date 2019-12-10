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
/*!
 * \file Kernel.hpp
 *
 * \brief Definitions of several kernel functions, and helper
 * routines. Also provides driver routines for kernel ridge
 * regression.
 */
#ifndef STRUMPACK_KERNEL_REGRESSION_HPP
#define STRUMPACK_KERNEL_REGRESSION_HPP

#include "Kernel.hpp"
#include "HSS/HSSMatrix.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "HSS/HSSMatrixMPI.hpp"
#if defined(STRUMPACK_USE_BPACK)
#include "HODLR/HODLRMatrix.hpp"
#endif
#endif

//#define ITERATIVE_REFINEMENT 1

namespace strumpack {

  namespace kernel {

    template<typename scalar_t>
    DenseMatrix<scalar_t> Kernel<scalar_t>::fit_HSS
    (std::vector<scalar_t>& labels, const HSS::HSSOptions<scalar_t>& opts) {
      TaskTimer timer("compression");
      if (opts.verbose())
        std::cout << "# starting HSS compression..." << std::endl;
      timer.start();
      HSS::HSSMatrix<scalar_t> H(*this, opts);
      DenseMW_t B(1, n(), labels.data(), 1);
      B.lapmt(perm_, true);
      if (opts.verbose()) {
        std::cout << "# HSS compression time = "
                  << timer.elapsed() << std::endl;
        if (H.is_compressed())
          std::cout << "# created HSS matrix of dimension "
                    << H.rows() << " x " << H.cols()
                    << " with " << H.levels() << " levels" << std::endl
                    << "# compression succeeded!" << std::endl;
        else std::cout << "# compression failed!!!" << std::endl;
        std::cout << "# rank(H) = " << H.rank() << std::endl
                  << "# HSS memory(H) = "
                  << H.memory() / 1e6 << " MB " << std::endl << std::endl
                  << "# factorization start" << std::endl;
      }
      timer.start();
      auto ULV = H.factor();
      if (opts.verbose())
        std::cout << "# factorization time = "
                  << timer.elapsed() << std::endl
                  << "# solution start..." << std::endl;
#if defined(ITERATIVE_REFINEMENT)
      DenseMW_t rhs(n(), 1, labels.data(), n());
      DenseM_t weights(rhs), residual(n(), 1);
      H.solve(ULV, weights);
      auto rhs_normF = rhs.normF();
      using real_t = typename RealType<scalar_t>::value_type;
      for (int ref=0; ref<3; ref++) {
        auto residual = H.apply(weights);
        residual.scaled_add(scalar_t(-1.), rhs);
        auto rres = residual.normF() / rhs_normF;
        if (opts.verbose())
          std::cout << "||H*weights - labels||_2/||labels||_2 = "
                    << rres << std::endl;
        if (rres < 10*blas::lamch<real_t>('E')) break;
        H.solve(ULV, residual);
        weights.scaled_add(scalar_t(-1.), residual);
      }
#else // no iterative refinement
      DenseM_t weights(n(), 1, labels.data(), n());
      H.solve(ULV, weights);
#endif
      if (opts.verbose())
        std::cout << "# solve time = " << timer.elapsed() << std::endl;
      return weights;
    }

    template<typename scalar_t>
    std::vector<scalar_t> Kernel<scalar_t>::predict
    (const DenseM_t& test, const DenseM_t& weights) const {
      assert(test.rows() == d());
      return predict
        (test.cols(),
         [this,&test](int r, int c) -> scalar_t {
           return eval_kernel_function(data_->ptr(0, r), test.ptr(0, c));
         }, weights);
    }

    template<typename scalar_t>
    std::vector<scalar_t> Kernel<scalar_t>::predict
    (int m, const std::function<scalar_t(int,int)>& Kxy,
     const DenseM_t& weights) const {
      std::vector<scalar_t> prediction(m);
#pragma omp parallel for
      for (std::size_t c=0; c<m; c++)
        for (std::size_t r=0; r<n(); r++)
          prediction[c] += weights(r, 0) * Kxy(r, c);
      return prediction;
    }

    template<typename scalar_t>
    std::vector<scalar_t> Kernel<scalar_t>::predict
    (int m, const std::function
     <void(const std::vector<std::size_t>&,
           const std::vector<std::size_t>&, DenseMatrix<scalar_t>&)>& Kxy,
     const DenseM_t& weights) const {
      std::vector<scalar_t> prediction(m);
      DenseMatrix<scalar_t> K(n(), m);
      std::vector<std::size_t> I(n()), J(m);
      for (std::size_t i=0; i<n(); i++)
        I[i] = perm_[i]-1;
      std::iota(J.begin(), J.end(), 0);
      Kxy(I, J, K);
#pragma omp parallel for
      for (std::size_t c=0; c<m; c++)
        for (std::size_t r=0; r<n(); r++)
          prediction[c] += weights(r, 0) * K(r, c);
      return prediction;
    }

#if defined(STRUMPACK_USE_MPI)
    template<typename scalar_t>
    DistributedMatrix<scalar_t> Kernel<scalar_t>::fit_HSS
    (const BLACSGrid& grid, std::vector<scalar_t>& labels,
     const HSS::HSSOptions<scalar_t>& opts) {
      TaskTimer timer("HSScompression");
      auto& c = grid.Comm();
      bool verb = opts.verbose() && c.is_root();
      if (verb) std::cout << "# starting HSS compression..." << std::endl;
      timer.start();
      HSS::HSSMatrixMPI<scalar_t> H(*this, &grid, opts);
      DenseMW_t B(1, n(), labels.data(), 1);
      B.lapmt(perm_, true);
      //perm.clear(); // TODO not needed anymore??
      if (opts.verbose()) {
        const auto lvls = H.max_levels();
        const auto rank = H.max_rank();
        const auto mem = H.total_memory();
        if (c.is_root()) {
          std::cout << "# HSS compression time = "
                    << timer.elapsed() << std::endl;
          if (H.is_compressed())
            std::cout << "# created HSS matrix of dimension "
                      << H.rows() << " x " << H.cols()
                      << " with " << lvls << " levels" << std::endl
                      << "# compression succeeded!" << std::endl;
          else std::cout << "# compression failed!!!" << std::endl;
          std::cout << "# rank(H) = " << rank << std::endl
                    << "# HSS memory(H) = " << mem / 1e6
                    << " MB " << std::endl << std::endl
                    << "# factorization start" << std::endl;
        }
      }
      timer.start();
      auto ULV = H.factor();
      if (verb)
        std::cout << "# factorization time = "
                  << timer.elapsed() << std::endl
                  << "# solution start..." << std::endl;
      DenseMW_t cB(n(), 1, labels.data(), n());
      DistM_t weights(&grid, n(), 1);
      weights.scatter(cB);
#if defined(ITERATIVE_REFINEMENT)
      DistM_t rhs(weights), residual(&grid, n(), 1);
      H.solve(ULV, weights);
      auto rhs_normF = rhs.normF();
      using real_t = typename RealType<scalar_t>::value_type;
      for (int ref=0; ref<3; ref++) {
        auto residual = H.apply(weights);
        residual.scaled_add(scalar_t(-1.), rhs);
        auto rres = residual.normF() / rhs_normF;
        if (verb)
          std::cout << "||H*weights - labels||_2/||labels||_2 = "
                    << rres << std::endl;
        if (rres < 10*blas::lamch<real_t>('E')) break;
        H.solve(ULV, residual);
        weights.scaled_add(scalar_t(-1.), residual);
      }
#else // no iterative refinement
      H.solve(ULV, weights);
#endif
      if (verb)
        std::cout << "# solve time = " << timer.elapsed() << std::endl;
      return weights;
    }

    template<typename scalar_t>
    std::vector<scalar_t> Kernel<scalar_t>::predict
    (const DenseM_t& test, const DistM_t& weights) const {
      return predict
        (test.cols(),
         [this,&test](int r, int c) -> scalar_t {
           return eval_kernel_function
             (data_->ptr(0, r), test.ptr(0, c));
         }, weights);
    }

    template<typename scalar_t>
    std::vector<scalar_t> Kernel<scalar_t>::predict
    (int m, const std::function<scalar_t(int,int)>& Kxy,
     const DistM_t& weights) const {
      std::vector<scalar_t> prediction(m);
      if (weights.active() && weights.lcols()) {
#pragma omp parallel for
        for (std::size_t c=0; c<m; c++)
          for (std::size_t r=0; r<weights.lrows(); r++)
            prediction[c] += weights(r, 0) * Kxy(weights.rowl2g(r), c);
      }
      // reduce the local sums to the global vector
      weights.Comm().all_reduce
        (prediction.data(), prediction.size(), MPI_SUM);
      return prediction;
    }

    template<typename scalar_t>
    std::vector<scalar_t> Kernel<scalar_t>::predict
    (int m, const std::function
     <void(const std::vector<std::size_t>&,
           const std::vector<std::size_t>&, DenseMatrix<scalar_t>&)>& Kxy,
     const DistM_t& weights) const {
      std::vector<scalar_t> prediction(m);
      DenseMatrix<scalar_t> K(weights.lrows(), m);
      std::vector<std::size_t> I(weights.lrows()), J(m);
      for (std::size_t r=0; r<weights.lrows(); r++)
        I[r] = weights.rowl2g(r);
      std::iota(J.begin(), J.end(), 0);
      Kxy(I, J, K);
      if (weights.active() && weights.lcols()) {
#pragma omp parallel for
        for (std::size_t c=0; c<m; c++)
          for (std::size_t r=0; r<weights.lrows(); r++)
            prediction[c] += weights(r, 0) * K(r, c);
      }
      // reduce the local sums to the global vector
      weights.Comm().all_reduce
        (prediction.data(), prediction.size(), MPI_SUM);
      return prediction;
    }

#if defined(STRUMPACK_USE_BPACK)
    template<typename scalar_t>
    DenseMatrix<scalar_t> Kernel<scalar_t>::fit_HODLR
    (const MPIComm& c, std::vector<scalar_t>& labels,
     const HODLR::HODLROptions<scalar_t>& opts) {
      TaskTimer timer("HODLRcompression");
      bool verb = opts.verbose() && c.is_root();
      if (verb) std::cout << "# starting HODLR compression..." << std::endl;
      timer.start();
      HODLR::HODLRMatrix<scalar_t> H(c, *this, opts);
      //H.print_stats();
      DenseMW_t B(1, n(), labels.data(), 1);
      B.lapmt(perm_, true);
      if (verb)
        std::cout << "# HODLR compression time = "
                  << timer.elapsed() << std::endl;
      timer.start();
      H.factor();
      if (verb)
        std::cout << "# factorization time = "
                  << timer.elapsed() << std::endl
                  << "# solution start..." << std::endl;
      int lrows = H.lrows();
      DenseMW_t lB(lrows, 1, &labels[H.begin_row()], lrows);
      DenseM_t lw(lrows, 1);
      H.solve(lB, lw);
      auto weights = H.all_gather_from_1D(lw);
      if (verb)
        std::cout << "# solve time = " << timer.elapsed() << std::endl;
      return weights;
    }
#endif
#endif

  } // end namespace kernel

} // end namespace strumpack


#endif // STRUMPACK_KERNEL_REGRESSION_HPP
