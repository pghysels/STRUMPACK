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
#ifndef STRUMPACK_MAGMA_WRAPPER_HPP
#define STRUMPACK_MAGMA_WRAPPER_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <cassert>
#include <memory>

#include "DenseMatrix.hpp"
#include "CUDAWrapper.hpp"

#include <magma_v2.h>

namespace strumpack {
  namespace gpu {
    namespace magma {

      inline int getrf(DenseMatrix<float>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        //magma_sgetrf_native
        magma_sgetrf_gpu
          (A.rows(), A.cols(), A.data(), A.ld(), piv.data(), &info);
        gpu::copy_host_to_device(dpiv, piv.data(), A.rows());
        return info;
      }
      inline int getrf(DenseMatrix<double>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        //magma_dgetrf_native
        magma_dgetrf_gpu
          (A.rows(), A.cols(), A.data(), A.ld(), piv.data(), &info);
        gpu::copy_host_to_device(dpiv, piv.data(), A.rows());
        return info;
      }
      inline int getrf(DenseMatrix<std::complex<float>>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        //magma_cgetrf_native
        magma_cgetrf_gpu
          (A.rows(), A.cols(), reinterpret_cast<magmaFloatComplex*>(A.data()),
           A.ld(), piv.data(), &info);
        gpu::copy_host_to_device(dpiv, piv.data(), A.rows());
        return info;
      }
      inline int getrf(DenseMatrix<std::complex<double>>& A, int* dpiv) {
        std::vector<int> piv(A.rows());
        int info = 0;
        //magma_zgetrf_native
        magma_zgetrf_gpu
          (A.rows(), A.cols(), reinterpret_cast<magmaDoubleComplex*>(A.data()),
           A.ld(), piv.data(), &info);
        gpu::copy_host_to_device(dpiv, piv.data(), A.rows());
        return info;
      }

    } // end namespace magma
  } // end namespace gpu
} // end namespace strumpack

#endif // STRUMPACK_CUDA_WRAPPER_HPP
