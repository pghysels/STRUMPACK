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
#ifndef STRUMPACK_CUDA_WRAPPER_HPP
#define STRUMPACK_CUDA_WRAPPER_HPP

#include "GPUWrapper.hpp"
#if defined(STRUMPACK_USE_MAGMA)
#include <magma_v2.h>
#endif
#if defined(STRUMPACK_USE_KBLAS)
#include "kblas.h"
#endif

#include <cmath>
#include <complex>
#include <iostream>
#include <cassert>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace strumpack {
  namespace gpu {

    // this is valid for compute capability 3.5 -> 8.0 (and beyond?)
    //const unsigned int MAX_BLOCKS_X = 4294967295; // 2^31-1
    const unsigned int MAX_BLOCKS_Y = 65535;
    const unsigned int MAX_BLOCKS_Z = 65535;
    // const unsigned int MAX_BLOCKS_Y = 8; // for debugging large cases
    // const unsigned int MAX_BLOCKS_Z = 8;

#define gpu_check(err) {                                               \
      strumpack::gpu::cuda_assert((err), __FILE__, __LINE__);          \
    }
    void cuda_assert(cudaError_t code, const char *file, int line,
                     bool abort=true);
    void cuda_assert(cusolverStatus_t code, const char *file, int line,
                     bool abort=true);
    void cuda_assert(cublasStatus_t code, const char *file, int line,
                     bool abort=true);

    const cudaStream_t& get_cuda_stream(const Stream& s);
    cudaStream_t& get_cuda_stream(Stream& s);

    const cublasHandle_t& get_cublas_handle(const Handle& h);
    cublasHandle_t& get_cublas_handle(Handle& s);

    const cusolverDnHandle_t& get_cusolver_handle(const Handle& h);
    cusolverDnHandle_t& get_cusolver_handle(Handle& s);

#if defined(STRUMPACK_USE_MAGMA)
    const magma_queue_t& get_magma_queue(const Handle& h);
    magma_queue_t& get_magma_queue(Handle& s);
#endif

#if defined(STRUMPACK_USE_KBLAS)
    const kblasHandle_t& get_kblas_handle(const Handle& h);
    kblasHandle_t& get_kblas_handle(Handle& h);

    const kblasRandState_t& get_kblas_rand_state(const Handle& h);
    kblasRandState_t& get_kblas_rand_state(Handle& h);
#endif

  } // end namespace gpu
} // end namespace strumpack

#endif // STRUMPACK_CUDA_WRAPPER_HPP
