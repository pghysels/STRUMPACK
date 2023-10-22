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
#ifndef STRUMPACK_HIP_WRAPPER_HPP
#define STRUMPACK_HIP_WRAPPER_HPP

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

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocsolver/rocsolver.h>

namespace strumpack {
  namespace gpu {

    const unsigned int MAX_BLOCKS_Y = 65535;
    const unsigned int MAX_BLOCKS_Z = 65535;

#define gpu_check(err) {                                              \
      strumpack::gpu::hip_assert((err), __FILE__, __LINE__);          \
    }
    void hip_assert(hipError_t code, const char *file, int line,
                    bool abort=true);
    void hip_assert(rocblas_status code, const char *file, int line,
                    bool abort=true);
    void hip_assert(hipblasStatus_t code, const char *file, int line,
                    bool abort=true);

    const hipStream_t& get_hip_stream(const Stream& s);
    hipStream_t& get_hip_stream(Stream& s);

    const hipblasHandle_t& get_hipblas_handle(const Handle& h);
    hipblasHandle_t& get_hipblas_handle(Handle& s);

    const rocblas_handle& get_rocblas_handle(const Handle& h);
    rocblas_handle& get_rocblas_handle(Handle& s);

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

#endif // STRUMPACK_HIP_WRAPPER_HPP
