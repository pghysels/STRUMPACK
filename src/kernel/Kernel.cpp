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
#include "KernelRegression.hpp"
#include "Kernel.h"

using namespace strumpack;
using namespace strumpack::kernel;
using namespace strumpack::HSS;

template<typename scalar_t>
class STRUMPACKKernelRegression {
public:
  STRUMPACKKernelRegression() {}
  std::unique_ptr<Kernel<scalar_t>> K_;
  DenseMatrix<scalar_t> training_, weights_;
};

#ifdef __cplusplus
extern "C" {
#endif

  //STRUMPACKKernel STRUMPACK_create_kernel_double
  STRUMPACKKernel STRUMPACK_create_kernel_double
  (int n, int d, double* train, double h, double lambda, int type) {
    std::cout << "# C++, creating kernel: n="
      << n << ", d=" << d << " h=" << h << " lambda=" << lambda
      << std::endl;
    auto kernel = new STRUMPACKKernelRegression<double>();
    kernel->training_ = DenseMatrix<double>(d, n, train, d);
    switch(type) {
    case 0:
      kernel->K_.reset(new GaussKernel<double>(kernel->training_, h, lambda));
      break;
    case 1:
      kernel->K_.reset(new LaplaceKernel<double>(kernel->training_, h, lambda));
      break;
    default: std::cout << "ERROR: Kernel type not recognized!" << std::endl;
    }
    return kernel;
  }

  void STRUMPACK_destroy_kernel_double(STRUMPACKKernel kernel) {
    delete static_cast<STRUMPACKKernelRegression<double>*>(kernel);
  }

  void STRUMPACK_kernel_fit_HSS_double
  (STRUMPACKKernel kernel, double* labels, int argc, char* argv[]) {
    auto KR = static_cast<STRUMPACKKernelRegression<double>*>(kernel);
    std::vector<double> vl(labels, labels+KR->K_->n());
    HSSOptions<double> opts;
    opts.set_verbose(false);
    opts.set_clustering_algorithm(ClusteringAlgorithm::COBBLE);
    opts.set_from_command_line(argc, argv);
    KR->weights_ = KR->K_->fit_HSS(vl, opts);
  }

  void STRUMPACK_kernel_fit_HODLR_double
  (STRUMPACKKernel kernel, double* labels, int argc, char* argv[]) {
#if defined(STRUMPACK_USE_HODLRBF)
    auto KR = static_cast<STRUMPACKKernelRegression<double>*>(kernel);
    std::vector<double> vl(labels, labels+KR->K_->n());
    HODLR::HODLROptions<double> opts;
    opts.set_verbose(false);
    opts.set_clustering_algorithm(ClusteringAlgorithm::COBBLE);
    opts.set_from_command_line(argc, argv);
    KR->weights_ = KR->K_->fit_HODLR(MPI_COMM_SELF, vl, opts);
#else
    std::cerr << "ERROR: STRUMPACK was not configured with HODLR support."
              << "       Using HSS compression as fallback!!" << std::endl;
    STRUMPACK_kernel_fit_HSS_double(kernel, labels, argc, argv);
#endif
  }

  void STRUMPACK_kernel_predict_double
  (STRUMPACKKernel kernel, int m, double* test, double* prediction) {
    auto KR = static_cast<STRUMPACKKernelRegression<double>*>(kernel);
    DenseMatrixWrapper<double> test_(KR->K_->d(), m, test, KR->K_->d());
    auto pred = KR->K_->predict(test_, KR->weights_);
    std::copy(pred.begin(), pred.end(), prediction);
  }

#ifdef __cplusplus
}
#endif
