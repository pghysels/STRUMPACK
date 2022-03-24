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
#define STRUMPACK_NO_TRIPLET_MPI
#include "FrontalMatrixGPUKernels.hpp"

#include <hip/hip_runtime.h>

#include <complex>
#include <iostream>

//#define STRUMPACK_HIP_HAVE_ROCTHRUST
#if defined(STRUMPACK_HIP_HAVE_ROCTHRUST)
#include <thrust/complex.h>
#endif

// this is valid for compute capability 3.5 -> 8.0 (and beyond?)
//const unsigned int MAX_BLOCKS_X = 4294967295; // 2^32-1
const unsigned int MAX_BLOCKS_Y = 65535;
const unsigned int MAX_BLOCKS_Z = 65535;


namespace strumpack {
  namespace gpu {

    /**
     * Get the real T type corresponding to a scalar, for instance T,
     * std::complex<T> or thrust::complex<T>, to be used for instance
     * to compute norms or absolute value.
     */
    template<class T> struct real_type { typedef T value_type; };
    template<class T> struct real_type<std::complex<T>> { typedef T value_type; };
#if defined(STRUMPACK_HIP_HAVE_ROCTHRUST)
    template<class T> struct real_type<thrust::complex<T>> { typedef T value_type; };
#endif

    /**
     * The types float2 and double2 are binary the same as
     * std::complex or thrust::complex, but they can be used as
     * __shared__ variables, whereas thrust::complex cannot because it
     * doesn't have a no-argument default constructor.
     */
    template<class T> struct primitive_type { typedef T value_type; };
    template<> struct primitive_type<std::complex<float>> { typedef float2 value_type; };
    template<> struct primitive_type<std::complex<double>> { typedef double2 value_type; };
#if defined(STRUMPACK_HIP_HAVE_ROCTHRUST)
    template<> struct primitive_type<thrust::complex<float>> { typedef float2 value_type; };
    template<> struct primitive_type<thrust::complex<double>> { typedef double2 value_type; };
#endif

    /**
     * Get the corresponding thrust::complex for std::complex
     */
    template<class T> struct cuda_type { typedef T value_type; };
#if defined(STRUMPACK_HIP_HAVE_ROCTHRUST)
    template<class T> struct cuda_type<std::complex<T>> { typedef thrust::complex<T> value_type; };
#endif

    __device__ float real(const float& a) { return a; }
    __device__ double real(const double& a) { return a; }
    __device__ float real(const std::complex<float>& a) { return a.real(); }
    __device__ double real(const std::complex<double>& a) { return a.real(); }
#if defined(STRUMPACK_HIP_HAVE_ROCTHRUST)
    __device__ float real(const thrust::complex<float>& a) { return a.real(); }
    __device__ double real(const thrust::complex<double>& a) { return a.real(); }
#endif

    __device__ float abs(const float& a) { return (a >= 0) ? a : -a; }
    __device__ double abs(const double& a) { return  (a >= 0) ? a : -a; }
    __device__ float abs(const std::complex<float>& a) {
      return sqrtf(a.real() * a.real() + a.imag() * a.imag());
    }
    __device__ double abs(const std::complex<double>& a) {
      return sqrt(a.real() * a.real() + a.imag() * a.imag());
    }
#if defined(STRUMPACK_HIP_HAVE_ROCTHRUST)
    __device__ float abs(const thrust::complex<float>& a) { return thrust::abs(a); }
    __device__ double abs(const thrust::complex<double>& a) { return thrust::abs(a); }
#endif

#if !defined(STRUMPACK_HIP_HAVE_ROCTHRUST)
    template<typename T> __device__ std::complex<T>
    operator+(const std::complex<T>& a, const std::complex<T>& b) {
      return std::complex<T>(a.real() + b.real(), a.imag() + b.imag());
    }
    template<typename T> __device__ std::complex<T>
    operator*(const std::complex<T>& a, const std::complex<T>& b) {
      return std::complex<T>
        (a.real() * b.real() - a.imag() * b.imag(),
         a.imag() * b.real() + a.real() * b.imag());
    }
    template<typename T> __device__ std::complex<T>
    operator+=(std::complex<T>& a, const std::complex<T>& b) {
      a = std::complex<T>(a.real() + b.real(), a.imag() + b.imag());
      return a;
    }
    template<typename T> __device__ std::complex<T>
    operator-=(std::complex<T>& a, const std::complex<T>& b) {
      a = std::complex<T>(a.real() - b.real(), a.imag() - b.imag());
      return a;
    }
    template<typename T> __device__ std::complex<T>
    operator/=(std::complex<T>& a, const std::complex<T>& b) {
      auto denom = b.real() * b.real() + b.imag() * b.imag();
      a = std::complex<T>
        ((a.real() * b.real() + a.imag() * b.imag()) / denom,
         (a.imag() * b.real() - a.real() * b.imag()) / denom);
      return a;
    }
#endif


    /**
     * Put elements of the sparse matrix in the F11, F12 and F21 parts
     * of the front.  The sparse elements are taken from F.e11, F.e12,
     * F.e21, which are lists of triplets {r,c,v}. The front is
     * assumed to be initialized to zero.
     *
     */
    template<typename T, int unroll> __global__ void
    assemble_kernel(unsigned int f0, unsigned int nf, AssembleData<T>* dat) {
      int idx = (hipBlockIdx_x * hipBlockDim_x) * unroll + hipThreadIdx_x,
        op = (hipBlockIdx_y + f0) * hipBlockDim_y + hipThreadIdx_y;
      if (op >= nf) return;
      auto& F = dat[op];
      for (int i=0, j=idx; i<unroll; i++, j+=hipBlockDim_x) {
        if (j >= F.n11) break;
        auto& t = F.e11[j];
        F.F11[t.r + t.c*F.d1] = t.v;
      }
      for (int i=0, j=idx; i<unroll; i++, j+=hipBlockDim_x) {
        if (j >= F.n12) break;
        auto& t = F.e12[j];
        F.F12[t.r + t.c*F.d1] = t.v;
      }
      for (int i=0, j=idx; i<unroll; i++, j+=hipBlockDim_x) {
        if (j >= F.n21) break;
        auto& t = F.e21[j];
        F.F21[t.r + t.c*F.d2] = t.v;
      }
    }


    /**
     * Single extend-add operation from one contribution block into
     * the parent front. d1 is the size of F11, d2 is the size of F22.
     */
    template<typename T, unsigned int unroll>
    __global__ void extend_add_kernel
    (unsigned int by0, unsigned int nf, AssembleData<T>* dat, bool left) {
      int y = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x,
        x0 = (hipBlockIdx_y + by0) * unroll,
        z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
      if (z >= nf) return;
      auto& f = dat[z];
      auto CB = left ? f.CB1 : f.CB2;
      if (!CB) return;
      auto dCB = left ? f.dCB1 : f.dCB2;
      if (y >= dCB) return;
      auto I = left ? f.I1 : f.I2;
      auto Iy = I[y];
      CB += y + x0*dCB;
      int d1 = f.d1, d2 = f.d2;
      int ld;
      T* F[2];
      if (Iy < d1) {
        ld = d1;
        F[0] = f.F11+Iy;
        F[1] = f.F12+Iy-d1*d1;
      } else {
        ld = d2;
        F[0] = f.F21+Iy-d1;
        F[1] = f.F22+Iy-d1-d1*d2;
      }
#pragma unroll
      for (int i=0; i<unroll; i++) {
        int x = x0 + i;
        if (x >= dCB) break;
        auto Ix = I[x];
        F[Ix >= d1][Ix*ld] += CB[i*dCB];
      }
    }

    template<typename T> void
    assemble(unsigned int nf, AssembleData<T>* dat,
             AssembleData<T>* ddat) {
      { // front assembly from sparse matrix
        std::size_t nnz = 0;
        for (unsigned int f=0; f<nf; f++)
          nnz = std::max
            (nnz, std::max(dat[f].n11, std::max(dat[f].n12, dat[f].n21)));
        if (nnz) {
          unsigned int nt = 512, ops = 1;
          const int unroll = 8;
          while (nt*unroll > nnz && nt > 8 && ops < 64) {
            nt /= 2;
            ops *= 2;
          }
          ops = std::min(ops, nf);
          unsigned int nb = (nnz + nt*unroll - 1) / (nt*unroll),
            nbf = (nf + ops - 1) / ops;
          dim3 block(nt, ops);
          for (unsigned int f=0; f<nbf; f+=MAX_BLOCKS_Y) {
            dim3 grid(nb, std::min(nbf-f, MAX_BLOCKS_Y));
            hipLaunchKernelGGL(HIP_KERNEL_NAME(assemble_kernel<T,unroll>),
                               grid, block, 0, 0, f, nf, ddat+f*ops);
          }
        }
      }
      { // extend-add
        int du = 0;
        for (unsigned int f=0; f<nf; f++)
          du = std::max(du, std::max(dat[f].dCB1, dat[f].dCB2));
        if (du) {
          const unsigned int unroll = 16;
          unsigned int nt = 512, ops = 1;
          while (nt > du && ops < 64) {
            nt /= 2;
            ops *= 2;
          }
          ops = std::min(ops, nf);
          unsigned int nbx = (du + nt - 1) / nt,
            nby = (du + unroll - 1) / unroll,
            nbf = (nf + ops - 1) / ops;
          dim3 block(nt, 1, ops);
          using T_ = typename cuda_type<T>::value_type;
          auto dat_ = reinterpret_cast<AssembleData<T_>*>(ddat);
          for (unsigned int y=0; y<nby; y+=MAX_BLOCKS_Y) {
            unsigned int ny = std::min(nby-y, MAX_BLOCKS_Y);
            for (unsigned int f=0; f<nbf; f+=MAX_BLOCKS_Z) {
              dim3 grid(nbx, ny, std::min(nbf-f, MAX_BLOCKS_Z));
              hipLaunchKernelGGL
                (HIP_KERNEL_NAME(extend_add_kernel<T_,unroll>),
                 grid, block, 0, 0, y, nf, dat_+f*ops, true);
              hipLaunchKernelGGL
                (HIP_KERNEL_NAME(extend_add_kernel<T_,unroll>),
                 grid, block, 0, 0, y, nf, dat_+f*ops, false);
            }
          }
        }
      }
    }

    // /**
    //  * This only works if value >= 0.
    //  * It's assuming two's complement for the int.
    //  * __float_as_int is like reinterpret_cast<int&>(value)
    //  */
    // __device__ __forceinline__ void atomicAbsMax(float* data, float value) {
    //   atomicMax((int *)data, __float_as_int(value));
    // }
    // __device__ __forceinline__ void atomicAbsMax(double* addr, double value) {
    //   // why does this not compile?
    //   atomicMax((long long int *)addr, __double_as_longlong(value));
    // }


    /**
     * LU with row pivoting, with a single NTxNT thread block. The
     * matrix size n must be less than NT.
     *
     * This is a naive implementation. The goal here is to reduce
     * kernel launch overhead by batching many small LU
     * factorizations.
     *
     * Use thrust::complex instead of std::complex.
     */
    template<typename T, int NT> __device__ int
    LU_block_kernel(int n, T* F, int* piv) {
      using cuda_primitive_t = typename primitive_type<T>::value_type;
      using real_t = typename real_type<T>::value_type;
      __shared__ int p;
      __shared__ cuda_primitive_t M_[NT*NT];
      T* M = reinterpret_cast<T*>(M_);
      __shared__ real_t Mmax, cabs[NT];
      int info = 0;
      int j = hipThreadIdx_x, i = hipThreadIdx_y;

      // copy F from global device storage into shared memory
      if (i < n && j < n)
        M[i+j*NT] = F[i+j*n];
      __syncthreads();

      for (int k=0; k<n; k++) {
        // only 1 thread looks for the pivot element
        // this should be optimized?
// #if 0
//         real_t pa = 0.;
//         Mmax = 0.;
//         piv[k] = k + 1;
//         __syncthreads();
//         if (j == k && i >= k) {
//           pa = abs(M[i+k*NT]);
//           // see above, not working for double?
//           atomicAbsMax(&Mmax, pa);
//         }
//         __syncthreads();
//         if (j == k && i > k)
//           if (Mmax == pa)
//             atomicMin(piv+k, i+1);
#if 0
        if (j == k && i == k) {
          p = k;
          Mmax = abs(M[k+k*NT]);
          for (int l=k+1; l<n; l++) {
            auto tmp = abs(M[l+k*NT]);
            if (tmp > Mmax) {
              Mmax = tmp;
              p = l;
            }
          }
          piv[k] = p + 1;
        }
#else
        if (j == k && i >= k)
          cabs[i] = abs(M[i+j*NT]);
        __syncthreads();
        if (j == k && i == k) {
          p = k;
          Mmax = cabs[k]; //abs(M[k+k*NT]);
          for (int l=k+1; l<n; l++) {
            auto tmp = cabs[l]; //abs(M[l+k*NT]);
            if (tmp > Mmax) {
              Mmax = tmp;
              p = l;
            }
          }
          piv[k] = p + 1;
        }
#endif
        __syncthreads();
        if (Mmax == T(0.)) {
          if (info == 0)
            info = k;
        } else {
          // swap row k with the pivot row
          if (j < n && i == k && p != k) {
            auto tmp = M[k+j*NT];
            M[k+j*NT] = M[p+j*NT];
            M[p+j*NT] = tmp;
          }
          __syncthreads();
          // divide by the pivot element
          if (j == k && i > k && i < n)
            M[i+k*NT] /= M[k+k*NT];
          __syncthreads();
          // Schur update
          if (j > k && i > k && j < n && i < n)
            M[i+j*NT] -= M[i+k*NT] * M[k+j*NT];
          __syncthreads();
        }
      }
      // write back from shared to global device memory
      if (i < n && j < n)
        F[i+j*n] = M[i+j*NT];
      return info;
    }

    template<typename T, int NT, typename real_t> __global__ void
    LU_block_kernel_batched(FrontData<T>* dat, bool replace, real_t thresh) {
      FrontData<T>& A = dat[hipBlockIdx_x];
      int info = LU_block_kernel<T,NT>(A.n1, A.F11, A.piv);
      if (info || replace) {
        int i = hipThreadIdx_x, j = hipThreadIdx_y;
        if (i == j && i < A.n1) {
          std::size_t k = i + i*A.n1;
          if (abs(A.F11[k]) < thresh)
            A.F11[k] = (gpu::real(A.F11[k]) < 0) ? -thresh : thresh;
        }
      }
    }


    template<typename T, typename real_t> __global__ void
    replace_pivots_kernel(int n, T* A, real_t thresh) {
      int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
      if (i < n) {
        std::size_t k = i + i*n;
        if (abs(A[k]) < thresh)
          A[k] = (gpu::real(A[k]) < 0) ? -thresh : thresh;
      }
    }

    template<typename T, typename real_t>
    void replace_pivots(int n, T* A, real_t thresh, gpu::Stream& s) {
      if (!n) return;
      using T_ = typename cuda_type<T>::value_type;
      int NT = 128;
      hipLaunchKernelGGL
        (HIP_KERNEL_NAME(replace_pivots_kernel<T_,real_t>),
         (n+NT)/NT, NT, 0, s, n, reinterpret_cast<T_*>(A), thresh);
    }

    /**
     * LU solve with matrix F factor in LU, with pivot vector piv. F
     * is n x n, and n <= NT. X is the right hand side, and is n x
     * m. Both F and X have leading dimension n.
     *
     * NTxNT is the dimension of the thread block.
     *
     * This doesn't work for T = std::complex<?>, use
     * T=thrust::complex<?> instead.
     */
    template<typename T, int NT> __device__ void
    solve_block_kernel(int n, int m, T* F, T* X, int* piv) {
      using primitive_t = typename primitive_type<T>::value_type;
      __shared__ int P[NT];
      __shared__ primitive_t A_[NT*NT], B_[NT*NT];
      T *B = reinterpret_cast<T*>(B_), *A = reinterpret_cast<T*>(A_);
      int j = hipThreadIdx_x, i = hipThreadIdx_y;
      if (j == 0)
        P[i] = i;
      __syncthreads();
      if (i == 0 && j == 0)
        for (int k=0; k<n; k++) {
          auto p = piv[k]-1;
          auto tmp = P[k];
          P[k] = P[p];
          P[p] = tmp;
        }
      // put matrix F in shared memory
      if (i < n && j < n)
        A[j+i*NT] = F[i+j*n];
      __syncthreads();

      // loop over blocks of NT columns of X
      for (int b=0; b<m; b+=NT) {
        int c = b + j;

        // put X in shared memory, while applying the permutation
        if (i < n && c < m)
          B[j+i*NT] = X[P[i]+c*n];
        __syncthreads();

        // solve with L (unit diagonal)
        for (int k=0; k<n; k++) {
          if (i > k && i < n && c < m)
            B[j+i*NT] -= A[k+i*NT] * B[j+k*NT];
          __syncthreads();
        }

        // solve with U
        for (int k=n-1; k>=0; k--) {
          if (i == k && c < m)
            B[j+i*NT] /= A[i+i*NT];
          __syncthreads();
          if (i < k && c < m)
            B[j+i*NT] -= A[k+i*NT] * B[j+k*NT];
          __syncthreads();
        }

        // write from shared back to global device memory
        if (i < n && c < m)
          X[i+c*n] = B[j+i*NT];
      }
    }

    template<typename T, int NT> __global__ void
    solve_block_kernel_batched(FrontData<T>* dat) {
      FrontData<T>& A = dat[hipBlockIdx_x];
      solve_block_kernel<T,NT>(A.n1, A.n2, A.F11, A.F12, A.piv);
    }


    /**
     * Compute F -= F21 * F12, where F is d2 x d2 and F12 is d1 x d2.
     * d1 is <= NT. This should be called with a single NT x NT thread
     * block.
     */
    template<typename T, int NT> __device__ void
    Schur_block_kernel(int d1, int d2, T* F12, T* F21, T* F22) {
      using cuda_primitive_t = typename primitive_type<T>::value_type;
      __shared__ cuda_primitive_t B_[NT*NT], A_[NT*NT];
      T *B = reinterpret_cast<T*>(B_), *A = reinterpret_cast<T*>(A_);
      int j = hipThreadIdx_x, i = hipThreadIdx_y;
      A[j+i*NT] = B[j+i*NT] = 0.;
      for (int cb=0; cb<d2; cb+=NT) {
        int c = cb + j;
        // put NT columns of F12 in shared memory B
        if (i < d1 && c < d2)
          B[j+i*NT] = F12[i+c*d1];
        __syncthreads();
        for (int rb=0; rb<d2; rb+=NT) {
          int r = rb + i;
          // put NT rows of F21 in shared memory A
          if (r < d2 && j < d1)
            A[j+i*NT] = F21[r+j*d2];
          __syncthreads(); // wait for A and B
          if (c < d2 && r < d2) {
            T tmp(0.);
            // k < d1 <= NT, by using k<NT this can be unrolled
            for (int k=0; k<NT; k++)
              tmp += A[k+i*NT] * B[j+k*NT];
            F22[r+c*d2] -= tmp;
          }
          __syncthreads(); // sync before reading new A/B
        }
      }
    }

    template<typename T, int NT> __global__ void
    Schur_block_kernel_batched(FrontData<T>* dat) {
      FrontData<T>& A = dat[hipBlockIdx_x];
      Schur_block_kernel<T,NT>(A.n1, A.n2, A.F12, A.F21, A.F22);
    }


    template<typename T, int NT, typename real_t> void
    factor_block_batch(unsigned int count, FrontData<T>* dat,
                       bool replace, real_t thresh) {
      if (!count) return;
      using T_ = typename cuda_type<T>::value_type;
      auto dat_ = reinterpret_cast<FrontData<T_>*>(dat);
      dim3 block(NT, NT); //, grid(count, 1, 1);
      hipLaunchKernelGGL
        (HIP_KERNEL_NAME(LU_block_kernel_batched<T_,NT,real_t>),
         dim3(count), block, 0, 0, dat_, replace, thresh);
      hipLaunchKernelGGL(HIP_KERNEL_NAME(solve_block_kernel_batched<T_,NT>),
                         dim3(count), block, 0, 0, dat_);
      hipLaunchKernelGGL(HIP_KERNEL_NAME(Schur_block_kernel_batched<T_,NT>),
                         dim3(count), block, 0, 0, dat_);
    }


    template<typename T, int NT> __global__ void
    solve_block_kernel_batched(int nrhs, FrontData<T>* dat) {
      FrontData<T>& A = dat[hipBlockIdx_x];
      solve_block_kernel<T,NT>(A.n1, nrhs, A.F11, A.F12, A.piv);
    }

    template<typename T, int NT, int alpha, int beta> __device__ void
    gemmNN_block_inner_kernel(int m, int n, int k,
                              T* Aglobal, T* Bglobal, T* Cglobal) {
      using cuda_primitive_t = typename primitive_type<T>::value_type;
      __shared__ cuda_primitive_t B_[NT*NT], A_[NT*NT];
      T *B = reinterpret_cast<T*>(B_), *A = reinterpret_cast<T*>(A_);
      int j = hipThreadIdx_x, i = hipThreadIdx_y;
      for (int cb=0; cb<n; cb+=NT) {
        int c = cb + j;
        // put NT columns of Bglobal in shared memory B
        if (i < k && c < n)
          B[j+i*NT] = Bglobal[i+c*k];
        __syncthreads();
        for (int rb=0; rb<m; rb+=NT) {
          int r = rb + i;
          // put NT rows of F21 in shared memory A
          if (r < m && j < k)
            A[j+i*NT] = Aglobal[r+j*m];
          __syncthreads(); // wait for A and B
          if (c < n && r < m) {
            T tmp(0.);
            // l < n <= NT, by using k<NT this can be unrolled
            for (int l=0; l<k; l++)
              tmp += A[l+i*NT] * B[j+l*NT];
            Cglobal[r+c*m] = T(alpha) * tmp + T(beta) * Cglobal[r+c*m];
          }
          __syncthreads(); // sync before reading new A/B
        }
      }
    }
    /**
     * Compute C = alpha*A*B + beta*C, with a single thread block,
     * with NT x NT threads. A is m x k, and k <= NT. B is m x n and C
     * is m x n.
     */
    template<typename T, int NT, int alpha, int beta> __global__ void
    gemmNN_block_inner_kernel_batched(int nrhs, FrontData<T>* dat) {
      FrontData<T>& A = dat[hipBlockIdx_x];
      gemmNN_block_inner_kernel<T,NT,alpha,beta>
        (A.n2, nrhs, A.n1, A.F21, A.F12, A.F22);
    }

    /**
     * Compute a matrix vector product C = alpha*A + beta*C with a
     * single 1 x NT thread block. A is m x k, with m <= NT. B is k x
     * 1, C is m x 1.
     */
    template<typename T, int NT, int alpha, int beta> __device__ void
    gemvN_block_inner_kernel(int m, int k,
                             T* Aglobal, T* Bglobal, T* Cglobal) {
      using cuda_primitive_t = typename primitive_type<T>::value_type;
      __shared__ cuda_primitive_t B_[NT];
      T *B = reinterpret_cast<T*>(B_);
      int i = hipThreadIdx_y;
      B[i] = Bglobal[i];
      __syncthreads();
      for (int r=i; r<m; r+=NT) {
        T tmp(0.);
        for (int j=0; j<k; j++) // j < k <= NT
          tmp += Aglobal[r+j*m] * B[j];
        Cglobal[r] = T(alpha) * tmp + T(beta) * Cglobal[r];
      }
    }
    template<typename T, int NT, int alpha, int beta> __global__ void
    gemvN_block_inner_kernel_batched(FrontData<T>* dat) {
      FrontData<T>& A = dat[hipBlockIdx_x];
      gemvN_block_inner_kernel<T,NT,alpha,beta>
        (A.n2, A.n1, A.F21, A.F12, A.F22);
    }

    /**
     * Single extend-add operation along the column dimension, for the
     * solve.  d1 is the size of F11, d2 is the size of F22.
     */
    template<typename T> __device__ void
    ea_rhs_kernel(int x, int y, int nrhs, int dsep, int dupd, int dCB,
                  T* b, T* bupd, T* CB, std::size_t* I) {
      if (x >= nrhs || y >= dCB) return;
      auto Iy = I[y];
      if (Iy < dsep) b[Iy+x*dsep] += CB[y+x*dCB];
      else bupd[Iy-dsep+x*dupd] += CB[y+x*dCB];
    }

    template<typename T> __global__ void
    extend_add_rhs_kernel_left
    (int nrhs, unsigned int by0, AssembleData<T>* dat) {
      int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x,
        y = (hipBlockIdx_y + by0) * hipBlockDim_y + hipThreadIdx_y;
      auto& F = dat[hipBlockIdx_z];
      if (F.CB1)
        ea_rhs_kernel(x, y, nrhs, F.d1, F.d2, F.dCB1,
                      F.F11, F.F21, F.CB1, F.I1);
    }
    template<typename T> __global__ void
    extend_add_rhs_kernel_right
    (int nrhs, unsigned int by0, AssembleData<T>* dat) {
      int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x,
        y = (hipBlockIdx_y + by0) * hipBlockDim_y + hipThreadIdx_y;
      auto& F = dat[hipBlockIdx_z];
      if (F.CB2)
        ea_rhs_kernel(x, y, nrhs, F.d1, F.d2, F.dCB2,
                      F.F11, F.F21, F.CB2, F.I2);
    }


    template<typename T> void
    extend_add_rhs(int nrhs, unsigned int nf,
                   AssembleData<T>* dat, AssembleData<T>* ddat) {
      unsigned int nty = 64, nby = 0;
      for (int f=0; f<nf; f++) {
        int b = dat[f].dCB1 / nty + (dat[f].dCB1 % nty != 0);
        if (b > nby) nby = b;
        b = dat[f].dCB2 / nty + (dat[f].dCB2 % nty != 0);
        if (b > nby) nby = b;
      }
      int ntx = (nrhs == 1) ? 1 : 16;
      int nbx = nrhs / ntx + (nrhs % ntx != 0);
      dim3 block(ntx, nty);
      using T_ = typename cuda_type<T>::value_type;
      auto dat_ = reinterpret_cast<AssembleData<T_>*>(ddat);
      for (unsigned int by=0; by<nby; by+=MAX_BLOCKS_Y) {
        int nbyy = std::min(nby-by, MAX_BLOCKS_Y);
        for (unsigned int f=0; f<nf; f+=MAX_BLOCKS_Z) {
          dim3 grid(nbx, nbyy, std::min(nf-f, MAX_BLOCKS_Z));
          hipLaunchKernelGGL(HIP_KERNEL_NAME(extend_add_rhs_kernel_left),
                             grid, block, 0, 0, nrhs, by, dat_+f);
        }
      }
      hipDeviceSynchronize();
      for (unsigned int by=0; by<nby; by+=MAX_BLOCKS_Y) {
        int nbyy = std::min(nby-by, MAX_BLOCKS_Y);
        for (unsigned int f=0; f<nf; f+=MAX_BLOCKS_Z) {
          dim3 grid(nbx, nbyy, std::min(nf-f, MAX_BLOCKS_Z));
          hipLaunchKernelGGL(HIP_KERNEL_NAME(extend_add_rhs_kernel_right),
                             grid, block, 0, 0, nrhs, by, dat_+f);
        }
      }
    }

    template<typename T, int NT> void
    fwd_block_batch(int nrhs, unsigned int count,
                    FrontData<T>* dat) {
      if (!count) return;
      using T_ = typename cuda_type<T>::value_type;
      auto dat_ = reinterpret_cast<FrontData<T_>*>(dat);
      dim3 block(NT, NT);
      hipLaunchKernelGGL(HIP_KERNEL_NAME(solve_block_kernel_batched<T_,NT>),
                         count, block, 0, 0, nrhs, dat_);
      if (nrhs == 1) {
        dim3 block1(1, NT);
        hipLaunchKernelGGL
          (HIP_KERNEL_NAME(gemvN_block_inner_kernel_batched<T_,NT,-1,1>),
           count, block1, 0, 0, dat_);
      } else {
        hipLaunchKernelGGL
          (HIP_KERNEL_NAME(gemmNN_block_inner_kernel_batched<T_,NT,-1,1>),
           count, block, 0, 0, nrhs, dat_);
      }
    }



    /**
     * Single extend-add operation along the column dimension, for the
     * solve.  d1 is the size of F11, d2 is the size of F22.
     */
    template<typename T> __device__ void
    extract_rhs_kernel(int x, int y, int nrhs, int dsep, int dupd, int dCB,
                       T* b, T* bupd, T* CB, std::size_t* I) {
      if (x >= nrhs || y >= dCB) return;
      auto Iy = I[y];
      if (Iy < dsep) CB[y+x*dCB] = b[Iy+x*dsep];
      else CB[y+x*dCB] = bupd[Iy-dsep+x*dupd];
    }

    template<typename T> __global__ void
    extract_rhs_kernel(int nrhs, unsigned int by0, AssembleData<T>* dat) {
      int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x,
        y = (hipBlockIdx_y + by0) * hipBlockDim_y + hipThreadIdx_y;
      auto& F = dat[hipBlockIdx_z];
      if (F.CB1)
        extract_rhs_kernel(x, y, nrhs, F.d1, F.d2, F.dCB1,
                           F.F11, F.F21, F.CB1, F.I1);
      if (F.CB2)
        extract_rhs_kernel(x, y, nrhs, F.d1, F.d2, F.dCB2,
                           F.F11, F.F21, F.CB2, F.I2);
    }

    template<typename T> void
    extract_rhs(int nrhs, unsigned int nf, AssembleData<T>* dat,
                AssembleData<T>* ddat) {
      unsigned int nty = 64, nby = 0;
      for (int f=0; f<nf; f++) {
        int b = dat[f].dCB1 / nty + (dat[f].dCB1 % nty != 0);
        if (b > nby) nby = b;
        b = dat[f].dCB2 / nty + (dat[f].dCB2 % nty != 0);
        if (b > nby) nby = b;
      }
      int ntx = (nrhs == 1) ? 1 : 16;
      int nbx = nrhs / ntx + (nrhs % ntx != 0);
      dim3 block(ntx, nty);
      using T_ = typename cuda_type<T>::value_type;
      auto dat_ = reinterpret_cast<AssembleData<T_>*>(ddat);
      for (unsigned int by=0; by<nby; by+=MAX_BLOCKS_Y) {
        int nbyy = std::min(nby-by, MAX_BLOCKS_Y);
        for (unsigned int f=0; f<nf; f+=MAX_BLOCKS_Z) {
          dim3 grid(nbx, nbyy, std::min(nf-f, MAX_BLOCKS_Z));
          hipLaunchKernelGGL
            (HIP_KERNEL_NAME(extract_rhs_kernel),
             grid, block, 0, 0, nrhs, by, dat_+f);
        }
      }
    }



    /**
     * Compute a matrix vector product C = alpha*A + beta*C with a
     * single 1 x NT thread block. A is m x k, with k <= NT. B is k x
     * 1, C is m x 1.
     */
    template<typename T, int NT, int alpha, int beta> __device__ void
    gemvN_block_outer_kernel(int m, int k,
                             T* Aglobal, T* Bglobal, T* Cglobal) {
      using cuda_primitive_t = typename primitive_type<T>::value_type;
      __shared__ cuda_primitive_t B_[NT], C_[NT];
      T *B = reinterpret_cast<T*>(B_), *C = reinterpret_cast<T*>(C_);
      int i = hipThreadIdx_y;
      C[i] = T(0.);
      for (int c=0; c<k; c+=NT) {
        B[i] = Bglobal[c+i];
        __syncthreads();
        if (i < m) {
          T tmp(0.);
          for (int j=0; j<min(NT, k-c); j++)
            tmp += Aglobal[i+(c+j)*m] * B[j];
          C[i] += tmp;
        }
      }
      if (i < m)
        Cglobal[i] = T(alpha) * C[i] + T(beta) * Cglobal[i];
    }
    template<typename T, int NT, int alpha, int beta> __global__ void
    gemvN_block_outer_kernel_batched(FrontData<T>* dat) {
      FrontData<T>& A = dat[hipBlockIdx_x];
      // F12 is F12, F21 holds yupd, F11 holds y
      gemvN_block_outer_kernel<T,NT,alpha,beta>
        (A.n1, A.n2, A.F12, A.F21, A.F11);
    }


    template<typename T, int NT, int alpha, int beta> __device__ void
    gemmNN_block_outer_kernel(int m, int n, int k,
                              T* Aglobal, T* Bglobal, T* Cglobal) {
      using cuda_primitive_t = typename primitive_type<T>::value_type;
      __shared__ cuda_primitive_t B_[NT*NT], A_[NT*NT], C_[NT*NT];
      T *A = reinterpret_cast<T*>(A_), *B = reinterpret_cast<T*>(B_),
        *C = reinterpret_cast<T*>(C_);
      int j = hipThreadIdx_x, i = hipThreadIdx_y;
      for (int nb=0; nb<n; nb+=NT) {
        int n_ = nb + j;
        C[i+j*NT] = T(0.);
        for (int kb=0; kb<k; kb+=NT) {
          int dk = min(NT, k-kb);
          if (j < dk && i < m)
            A[i+j*NT] = Aglobal[i+(kb+j)*m];
          if (i < dk && n_ < n)
            B[i+j*NT] = Bglobal[(kb+i)+n_*k];
          __syncthreads();
          if (i < m && n_ < n) {
            T tmp(0.);
            for (int l=0; l<dk; l++)
              tmp += A[i+l*NT] * B[l+j*NT];
            C[i+j*NT] += tmp;
          }
          __syncthreads();
        }
        if (i < m && n_ < n)
          Cglobal[i+n_*m] = T(alpha) * C[i+j*NT] + T(beta) * Cglobal[i+n_*m];
      }
    }
    /**
     * Compute C = alpha*A*B + beta*C, with a single thread block,
     * with NT x NT threads. A is m x k, and k <= NT. B is m x n and C
     * is m x n.
     */
    template<typename T, int NT, int alpha, int beta> __global__ void
    gemmNN_block_outer_kernel_batched(int nrhs, FrontData<T>* dat) {
      FrontData<T>& A = dat[hipBlockIdx_x];
      gemmNN_block_outer_kernel<T,NT,alpha,beta>
        (A.n1, nrhs, A.n2, A.F12, A.F21, A.F11);
    }


    template<typename T, int NT> void
    bwd_block_batch(int nrhs, unsigned int count,
                    FrontData<T>* dat) {
      if (!count) return;
      using T_ = typename cuda_type<T>::value_type;
      auto dat_ = reinterpret_cast<FrontData<T_>*>(dat);
      if (nrhs == 1) {
        dim3 block(1, NT);
        hipLaunchKernelGGL
          (HIP_KERNEL_NAME(gemvN_block_outer_kernel_batched<T_,NT,-1,1>),
           count, block, 0, 0, dat_);
      } else {
        dim3 block(NT, NT);
        hipLaunchKernelGGL
          (HIP_KERNEL_NAME(gemmNN_block_outer_kernel_batched<T_,NT,-1,1>),
           count, block, 0, 0, nrhs, dat_);
      }
    }


    // explicit template instantiations
    template void assemble(unsigned int, AssembleData<float>*, AssembleData<float>*);
    template void assemble(unsigned int, AssembleData<double>*, AssembleData<double>*);
    template void assemble(unsigned int, AssembleData<std::complex<float>>*, AssembleData<std::complex<float>>*);
    template void assemble(unsigned int, AssembleData<std::complex<double>>*, AssembleData<std::complex<double>>*);

    template void extend_add_rhs(int, unsigned int, AssembleData<float>*, AssembleData<float>*);
    template void extend_add_rhs(int, unsigned int, AssembleData<double>*, AssembleData<double>*);
    template void extend_add_rhs(int, unsigned int, AssembleData<std::complex<float>>*, AssembleData<std::complex<float>>*);
    template void extend_add_rhs(int, unsigned int, AssembleData<std::complex<double>>*, AssembleData<std::complex<double>>*);

    template void extract_rhs(int, unsigned int, AssembleData<float>*, AssembleData<float>*);
    template void extract_rhs(int, unsigned int, AssembleData<double>*, AssembleData<double>*);
    template void extract_rhs(int, unsigned int, AssembleData<std::complex<float>>*, AssembleData<std::complex<float>>*);
    template void extract_rhs(int, unsigned int, AssembleData<std::complex<double>>*, AssembleData<std::complex<double>>*);


    template void factor_block_batch<float,8,float>(unsigned int, FrontData<float>*, bool, float);
    template void factor_block_batch<double,8,double>(unsigned int, FrontData<double>*, bool, double);
    template void factor_block_batch<std::complex<float>,8,float>(unsigned int, FrontData<std::complex<float>>*, bool, float);
    template void factor_block_batch<std::complex<double>,8,double>(unsigned int, FrontData<std::complex<double>>*, bool, double);

    template void factor_block_batch<float,16,float>(unsigned int, FrontData<float>*, bool, float);
    template void factor_block_batch<double,16,double>(unsigned int, FrontData<double>*, bool, double);
    template void factor_block_batch<std::complex<float>,16,float>(unsigned int, FrontData<std::complex<float>>*, bool, float);
    template void factor_block_batch<std::complex<double>,16,double>(unsigned int, FrontData<std::complex<double>>*, bool, double);

    template void factor_block_batch<float,24,float>(unsigned int, FrontData<float>*, bool, float);
    template void factor_block_batch<double,24,double>(unsigned int, FrontData<double>*, bool, double);
    template void factor_block_batch<std::complex<float>,24,float>(unsigned int, FrontData<std::complex<float>>*, bool, float);
    template void factor_block_batch<std::complex<double>,24,double>(unsigned int, FrontData<std::complex<double>>*, bool, double);

    template void factor_block_batch<float,32,float>(unsigned int, FrontData<float>*, bool, float);
    template void factor_block_batch<double,32,double>(unsigned int, FrontData<double>*, bool, double);
    template void factor_block_batch<std::complex<float>,32,float>(unsigned int, FrontData<std::complex<float>>*, bool, float);
    template void factor_block_batch<std::complex<double>,32,double>(unsigned int, FrontData<std::complex<double>>*, bool, double);

    template void replace_pivots(int, float*, float, gpu::Stream&);
    template void replace_pivots(int, double*, double, gpu::Stream&);
    template void replace_pivots(int, std::complex<float>*, float, gpu::Stream&);
    template void replace_pivots(int, std::complex<double>*, double, gpu::Stream&);

    template void fwd_block_batch<float,8>(int, unsigned int, FrontData<float>*);
    template void fwd_block_batch<double,8>(int, unsigned int, FrontData<double>*);
    template void fwd_block_batch<std::complex<float>,8>(int, unsigned int, FrontData<std::complex<float>>*);
    template void fwd_block_batch<std::complex<double>,8>(int, unsigned int, FrontData<std::complex<double>>*);

    template void fwd_block_batch<float,16>(int, unsigned int, FrontData<float>*);
    template void fwd_block_batch<double,16>(int, unsigned int, FrontData<double>*);
    template void fwd_block_batch<std::complex<float>,16>(int, unsigned int, FrontData<std::complex<float>>*);
    template void fwd_block_batch<std::complex<double>,16>(int, unsigned int, FrontData<std::complex<double>>*);

    template void fwd_block_batch<float,24>(int, unsigned int, FrontData<float>*);
    template void fwd_block_batch<double,24>(int, unsigned int, FrontData<double>*);
    template void fwd_block_batch<std::complex<float>,24>(int, unsigned int, FrontData<std::complex<float>>*);
    template void fwd_block_batch<std::complex<double>,24>(int, unsigned int, FrontData<std::complex<double>>*);

    template void fwd_block_batch<float,32>(int, unsigned int, FrontData<float>*);
    template void fwd_block_batch<double,32>(int, unsigned int, FrontData<double>*);
    template void fwd_block_batch<std::complex<float>,32>(int, unsigned int, FrontData<std::complex<float>>*);
    template void fwd_block_batch<std::complex<double>,32>(int, unsigned int, FrontData<std::complex<double>>*);


    template void bwd_block_batch<float,8>(int, unsigned int, FrontData<float>*);
    template void bwd_block_batch<double,8>(int, unsigned int, FrontData<double>*);
    template void bwd_block_batch<std::complex<float>,8>(int, unsigned int, FrontData<std::complex<float>>*);
    template void bwd_block_batch<std::complex<double>,8>(int, unsigned int, FrontData<std::complex<double>>*);

    template void bwd_block_batch<float,16>(int, unsigned int, FrontData<float>*);
    template void bwd_block_batch<double,16>(int, unsigned int, FrontData<double>*);
    template void bwd_block_batch<std::complex<float>,16>(int, unsigned int, FrontData<std::complex<float>>*);
    template void bwd_block_batch<std::complex<double>,16>(int, unsigned int, FrontData<std::complex<double>>*);

    template void bwd_block_batch<float,24>(int, unsigned int, FrontData<float>*);
    template void bwd_block_batch<double,24>(int, unsigned int, FrontData<double>*);
    template void bwd_block_batch<std::complex<float>,24>(int, unsigned int, FrontData<std::complex<float>>*);
    template void bwd_block_batch<std::complex<double>,24>(int, unsigned int, FrontData<std::complex<double>>*);

    template void bwd_block_batch<float,32>(int, unsigned int, FrontData<float>*);
    template void bwd_block_batch<double,32>(int, unsigned int, FrontData<double>*);
    template void bwd_block_batch<std::complex<float>,32>(int, unsigned int, FrontData<std::complex<float>>*);
    template void bwd_block_batch<std::complex<double>,32>(int, unsigned int, FrontData<std::complex<double>>*);

  } // end namespace gpu
} // end namespace strumpack
