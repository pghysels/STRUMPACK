/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The
 * Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals
 * from the U.S. Dept. of Energy).  All rights reserved.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 */
// CUDA backend of the native batched ARA, see BatchedARAKernels.hpp
#define STRUMPACK_NO_TRIPLET_MPI
#include "KBLASWrapper.hpp"

#define ARA_STREAM_T cudaStream_t
#define ARA_GET_STREAM(h, s) \
  gpu_check(cublasGetStream(get_cublas_handle(h), &(s)))
#define ARA_SHFL_DOWN(v, o) __shfl_down_sync(0xffffffffu, (v), (o))

#include "BatchedARAKernels.hpp"
