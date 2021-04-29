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
#include <iostream>
#include <iomanip>

#include "IterativeSolvers.hpp"

namespace strumpack {
  namespace iterative {

    template<typename scalar_t>
    using DMat = DenseMatrix<scalar_t>;

    template<typename scalar_t,typename integer_t>
    using SpMat = CompressedSparseMatrix<scalar_t,integer_t>;

    template<typename scalar_t>
    using Prec = std::function<void(DMat<scalar_t>&)>;


    template<typename scalar_t,typename integer_t,typename real_t> void
    IterativeRefinement(const SpMat<scalar_t,integer_t>& A,
                        const Prec<scalar_t>& M,
                        DMat<scalar_t>& x, const DMat<scalar_t>& b,
                        real_t rtol, real_t atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose) {
      DMat<scalar_t> r(x.rows(), x.cols());
      if (non_zero_guess) {
        A.spmv(x, r);
        r.scale_and_add(scalar_t(-1.), b);
      } else {
        r = b;
        x.zero();
      }
      auto res_norm = r.norm();
      auto res0 = res_norm;
      auto rel_res_norm = real_t(1.);
      auto bw_error = real_t(1.);
      totit = 0;
      if (verbose)
        std::cout << "REFINEMENT it. " << totit
                  << "\tres = " << std::setw(12) << res_norm
                  << "\trel.res = " << std::setw(12) << rel_res_norm
                  << "\tbw.error = " << std::setw(12) << bw_error
                  << std::endl;
      while (res_norm > atol && rel_res_norm > rtol &&
             totit++ < maxit && bw_error > atol) {
        M(r);
        x.add(r);
        bw_error = A.max_scaled_residual(x, b);
        A.spmv(x, r);
        r.scale_and_add(scalar_t(-1.), b);
        res_norm = r.norm();
        rel_res_norm = res_norm / res0;
        if (verbose)
          std::cout << "REFINEMENT it. " << totit << "\tres = "
                    << std::setw(12) << res_norm
                    << "\trel.res = " << std::setw(12) << rel_res_norm
                    << "\tbw.error = " << std::setw(12) << bw_error
                    << std::endl;
      }
    }


    // TODO avoid this duplication
    template<typename scalar_t,typename real_t> void
    IterativeRefinement(const DMat<scalar_t>& A, const Prec<scalar_t>& M,
                        DMat<scalar_t>& x, const DMat<scalar_t>& b,
                        real_t rtol, real_t atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose) {
      DMat<scalar_t> r(b);
      if (non_zero_guess)
        gemm(Trans::N, Trans::N, scalar_t(-1.), A, x, scalar_t(1.), r);
      else
        x.zero();
      auto res_norm = r.norm();
      auto res0 = res_norm;
      auto rel_res_norm = real_t(1.);
      totit = 0;
      if (verbose)
        std::cout << "REFINEMENT it. " << totit
                  << "\tres = " << std::setw(12) << res_norm
                  << "\trel.res = " << std::setw(12) << rel_res_norm
                  << std::endl;
      while (res_norm > atol && rel_res_norm > rtol &&
             totit++ < maxit) {
        M(r);
        x.add(r);
        r.copy(b);
        gemm(Trans::N, Trans::N, scalar_t(-1.), A, x, scalar_t(1.), r);
        res_norm = r.norm();
        rel_res_norm = res_norm / res0;
        if (verbose)
          std::cout << "REFINEMENT it. " << totit << "\tres = "
                    << std::setw(12) << res_norm
                    << "\trel.res = " << std::setw(12) << rel_res_norm
                    << std::endl;
      }
    }

    // explicit template instantiations
    template void
    IterativeRefinement(const SpMat<float,int>& A, const Prec<float>& M,
                        DMat<float>& x, const DMat<float>& b,
                        float rtol, float atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);
    template void
    IterativeRefinement(const SpMat<double,int>& A, const Prec<double>& M,
                        DMat<double>& x, const DMat<double>& b,
                        double rtol, double atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);
    template void
    IterativeRefinement(const SpMat<std::complex<float>,int>& A,
                        const Prec<std::complex<float>>& M,
                        DMat<std::complex<float>>& x,
                        const DMat<std::complex<float>>& b,
                        float rtol, float atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);
    template void
    IterativeRefinement(const SpMat<std::complex<double>,int>& A,
                        const Prec<std::complex<double>>& M,
                        DMat<std::complex<double>>& x,
                        const DMat<std::complex<double>>& b,
                        double rtol, double atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);

    template void
    IterativeRefinement(const SpMat<float,long int>& A, const Prec<float>& M,
                        DMat<float>& x, const DMat<float>& b,
                        float rtol, float atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);
    template void
    IterativeRefinement(const SpMat<double,long int>& A,
                        const Prec<double>& M,
                        DMat<double>& x, const DMat<double>& b,
                        double rtol, double atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);
    template void
    IterativeRefinement(const SpMat<std::complex<float>,long int>& A,
                        const Prec<std::complex<float>>& M,
                        DMat<std::complex<float>>& x,
                        const DMat<std::complex<float>>& b,
                        float rtol, float atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);
    template void
    IterativeRefinement(const SpMat<std::complex<double>,long int>& A,
                        const Prec<std::complex<double>>& M,
                        DMat<std::complex<double>>& x,
                        const DMat<std::complex<double>>& b,
                        double rtol, double atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);

    template void
    IterativeRefinement(const SpMat<float,long long int>& A,
                        const Prec<float>& M,
                        DMat<float>& x, const DMat<float>& b,
                        float rtol, float atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);
    template void
    IterativeRefinement(const SpMat<double,long long int>& A,
                        const Prec<double>& M,
                        DMat<double>& x, const DMat<double>& b,
                        double rtol, double atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);
    template void
    IterativeRefinement(const SpMat<std::complex<float>,long long int>& A,
                        const Prec<std::complex<float>>& M,
                        DMat<std::complex<float>>& x,
                        const DMat<std::complex<float>>& b,
                        float rtol, float atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);
    template void
    IterativeRefinement(const SpMat<std::complex<double>,long long int>& A,
                        const Prec<std::complex<double>>& M,
                        DMat<std::complex<double>>& x,
                        const DMat<std::complex<double>>& b,
                        double rtol, double atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);



    template void
    IterativeRefinement(const DMat<float>& A, const Prec<float>& M,
                        DMat<float>& x, const DMat<float>& b,
                        float rtol, float atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);
    template void
    IterativeRefinement(const DMat<double>& A, const Prec<double>& M,
                        DMat<double>& x, const DMat<double>& b,
                        double rtol, double atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);
    template void
    IterativeRefinement(const DMat<std::complex<float>>& A,
                        const Prec<std::complex<float>>& M,
                        DMat<std::complex<float>>& x,
                        const DMat<std::complex<float>>& b,
                        float rtol, float atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);
    template void
    IterativeRefinement(const DMat<std::complex<double>>& A,
                        const Prec<std::complex<double>>& M,
                        DMat<std::complex<double>>& x,
                        const DMat<std::complex<double>>& b,
                        double rtol, double atol, int& totit, int maxit,
                        bool non_zero_guess, bool verbose);

  } // end namespace iterative
} // end namespace strumpack

