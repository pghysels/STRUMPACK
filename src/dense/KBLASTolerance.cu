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
// Per-tile ARA thresholds using the same initial scale as CPU pivoted QR.
#define STRUMPACK_NO_TRIPLET_MPI
#include "KBLASWrapper.hpp"
#if defined(KBLAS_HAS_ARA_TOL_ARRAY) && KBLAS_HAS_ARA_TOL_ARRAY
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strumpack {
namespace gpu {
namespace kblas {
namespace {

__device__ float magnitude(float x) { return fabsf(x); }
__device__ double magnitude(double x) { return fabs(x); }
__device__ float magnitude(cuComplex x) { return hypotf(x.x, x.y); }
__device__ double magnitude(cuDoubleComplex x) { return hypot(x.x, x.y); }
__device__ float combine(float a, float b) { return hypotf(a, b); }
__device__ double combine(double a, double b) { return hypot(a, b); }

// One warp per column, eight warps per tile. Hypot accumulation avoids
// overflow/underflow in squared norms, including strongly scaled tiles.
template<typename T, typename R> __global__ void
norm_kernel(const int* rows, const int* cols, T* const* matrices,
            const int* ld, R rtol, R atol, R* result, bool frobenius) {
  constexpr int warps = 8;
  __shared__ R partial[warps];
  const int tile = blockIdx.x, lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  R accum = 0;
  for (int col = warp; col < cols[tile]; col += warps) {
    R norm = 0;
    for (int row = lane; row < rows[tile]; row += 32)
      norm = combine(norm, magnitude(matrices[tile]
                     [std::size_t(col) * ld[tile] + row]));
    for (int offset = 16; offset; offset /= 2)
      norm = combine(norm, __shfl_down_sync(0xffffffff, norm, offset));
    if (lane == 0)
      accum = frobenius ? combine(accum, norm) : (accum > norm ? accum : norm);
  }
  if (lane == 0) partial[warp] = accum;
  __syncthreads();
  if (threadIdx.x == 0) {
    R norm = 0;
    for (int i = 0; i < warps; i++)
      norm = frobenius ? combine(norm, partial[i])
        : (norm > partial[i] ? norm : partial[i]);
    const R relative = rtol * norm;
    result[tile] = frobenius ? norm : (atol > relative ? atol : relative);
  }
}

template<typename T> struct DeviceType { using type = T; };
template<> struct DeviceType<std::complex<float>> { using type = cuComplex; };
template<> struct DeviceType<std::complex<double>> { using type = cuDoubleComplex; };

template<typename T, typename R> cudaStream_t
launch_norms(cudaStream_t stream, const int* rows, const int* cols,
             T* const* matrices, const int* ld, R rtol, R atol,
             R* result, int count, bool frobenius) {
  using D = typename DeviceType<T>::type;
  norm_kernel<D, R><<<count, 256, 0, stream>>>
    (rows, cols, reinterpret_cast<D* const*>(matrices), ld,
     rtol, atol, result, frobenius);
  gpu_check(cudaPeekAtLastError());
  return stream;
}
} // namespace

template<typename T, typename R> void
ara_tolerances(Handle& handle, const int* rows, const int* cols,
               T* const* matrices, const int* ld, R rtol, R atol,
               R* tolerances, int count) {
  if (count < 0 || !std::isfinite(rtol) || !std::isfinite(atol) ||
      rtol < 0 || atol < 0)
    throw std::invalid_argument("Invalid ARA compression tolerance or batch size");
  if (!count) return;
  cudaStream_t stream;
  gpu_check(cublasGetStream(get_cublas_handle(handle), &stream));
  launch_norms(stream, rows, cols, matrices, ld, rtol, atol,
               tolerances, count, false);
}

template<typename T> typename RealType<T>::value_type
front_norm(const DenseMatrix<T>& F11,
           const DenseMatrix<T>& F12, const DenseMatrix<T>& F21) {
  using R = typename RealType<T>::value_type;
  const DenseMatrix<T>* fronts[] = {&F11, &F12, &F21};
  int dims[9];
  T* pointers[3];
  for (int i = 0; i < 3; i++) {
    const auto& F = *fronts[i];
    if (F.rows() > std::size_t(std::numeric_limits<int>::max()) ||
        F.cols() > std::size_t(std::numeric_limits<int>::max()) ||
        F.ld() > std::size_t(std::numeric_limits<int>::max()))
      throw std::overflow_error("ARA front dimensions exceed 32-bit indices");
    dims[i] = F.rows(); dims[i+3] = F.cols(); dims[i+6] = F.ld();
    pointers[i] = const_cast<T*>(F.data());
  }
  DeviceMemory<int> ddims(9);
  DeviceMemory<T*> dptrs(3);
  DeviceMemory<R> dnorms(3);
  copy_host_to_device<int>(ddims, dims, 9);
  copy_host_to_device<T*>(dptrs, pointers, 3);
  auto stream = launch_norms<T, R>
    (nullptr, ddims, ddims.template as<int>()+3, dptrs,
     ddims.template as<int>()+6, R(0), R(0), dnorms, 3, true);
  gpu_check(cudaStreamSynchronize(stream));
  R norms[3];
  copy_device_to_host<R>(norms, dnorms, 3);
  return std::hypot(std::hypot(norms[0], norms[1]), norms[2]);
}

#define INSTANTIATE(T, R) \
  template void ara_tolerances<T, R>(Handle&, const int*, const int*, \
    T* const*, const int*, R, R, R*, int); \
  template R front_norm<T>(const DenseMatrix<T>&, \
    const DenseMatrix<T>&, const DenseMatrix<T>&);
INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(std::complex<float>, float)
INSTANTIATE(std::complex<double>, double)
#undef INSTANTIATE

} // namespace kblas
} // namespace gpu
} // namespace strumpack
#endif
