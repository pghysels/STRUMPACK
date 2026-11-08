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
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
/*
 * Native GPU (HIP and CUDA) implementation of the batched adaptive
 * randomized approximation (ARA) interface in gpu::kblas (see
 * KBLASWrapper.hpp), so that the BLR code can use batched compression
 * without the (private) KBLAS library.
 *
 * This file is included by HIPBatchedARA.hip and CUDABatchedARA.cu,
 * which define the ARA_* macros used below:
 *   ARA_STREAM_T, ARA_GET_STREAM(handle, stream), ARA_SHFL_DOWN(v, offset)
 * Everything else uses the backend independent routines in GPUWrapper.hpp.
 *
 * One thread block per tile. For every tile M (m x n) an orthonormal
 * basis Q (m x rank) is computed with an adaptive randomized range
 * finder: blocks of random vectors w are drawn on the fly, y = M w is
 * orthogonalized (twice, classical Gram-Schmidt) against the current
 * basis, and accepted if its norm is larger than the tolerance. The
 * iteration stops after r consecutive vectors were rejected. Then
 * V = M^H Q, such that M ~ Q V^H.
 */
#include <atomic>
#include <cstdint>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strumpack {
  namespace gpu {
    namespace kblas {
      namespace {

        constexpr int NT = 256;   // threads per block
        constexpr int NBLK = 16;  // max number of random vectors per round
        constexpr int JC = 64;    // chunk of columns for the sampling

        template<typename R> struct cplx { R x, y; };

        template<typename R> __device__ __forceinline__ cplx<R>
        operator+(cplx<R> a, cplx<R> b) { return {a.x+b.x, a.y+b.y}; }
        template<typename R> __device__ __forceinline__ cplx<R>
        operator-(cplx<R> a, cplx<R> b) { return {a.x-b.x, a.y-b.y}; }
        template<typename R> __device__ __forceinline__ cplx<R>
        operator*(cplx<R> a, cplx<R> b) {
          return {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
        }
        template<typename R> __device__ __forceinline__ cplx<R>&
        operator+=(cplx<R>& a, cplx<R> b) { a.x += b.x; a.y += b.y; return a; }
        template<typename R> __device__ __forceinline__ cplx<R>&
        operator-=(cplx<R>& a, cplx<R> b) { a.x -= b.x; a.y -= b.y; return a; }

        __device__ __forceinline__ float cj(float a) { return a; }
        __device__ __forceinline__ double cj(double a) { return a; }
        template<typename R> __device__ __forceinline__ cplx<R>
        cj(cplx<R> a) { return {a.x, -a.y}; }

        __device__ __forceinline__ float ab2(float a) { return a*a; }
        __device__ __forceinline__ double ab2(double a) { return a*a; }
        template<typename R> __device__ __forceinline__ R
        ab2(cplx<R> a) { return a.x*a.x + a.y*a.y; }

        __device__ __forceinline__ float scl(float a, float s) { return a*s; }
        __device__ __forceinline__ double scl(double a, double s) { return a*s; }
        template<typename R> __device__ __forceinline__ cplx<R>
        scl(cplx<R> a, R s) { return {a.x*s, a.y*s}; }

        __device__ __forceinline__ float magnitude(float x) { return fabsf(x); }
        __device__ __forceinline__ double magnitude(double x) { return fabs(x); }
        __device__ __forceinline__ float magnitude(cplx<float> x) {
          return hypotf(x.x, x.y);
        }
        __device__ __forceinline__ double magnitude(cplx<double> x) {
          return hypot(x.x, x.y);
        }
        __device__ __forceinline__ float combine(float a, float b) {
          return hypotf(a, b);
        }
        __device__ __forceinline__ double combine(double a, double b) {
          return hypot(a, b);
        }

        __device__ __forceinline__ float shfld(float v, int o) {
          return ARA_SHFL_DOWN(v, o);
        }
        __device__ __forceinline__ double shfld(double v, int o) {
          return ARA_SHFL_DOWN(v, o);
        }
        template<typename R> __device__ __forceinline__ cplx<R>
        shfld(cplx<R> v, int o) {
          return {ARA_SHFL_DOWN(v.x, o), ARA_SHFL_DOWN(v.y, o)};
        }

        template<typename S> __device__ __forceinline__ S warp_sum(S v) {
          for (int o=warpSize/2; o>0; o>>=1) v += shfld(v, o);
          return v;
        }

        template<typename R> __device__ R block_sum(R v, R* wred) {
          const int lane = threadIdx.x % warpSize;
          const int wid = threadIdx.x / warpSize;
          const int nw = blockDim.x / warpSize;
          for (int o=warpSize/2; o>0; o>>=1) v += shfld(v, o);
          if (lane == 0) wred[wid] = v;
          __syncthreads();
          R s = 0;
          for (int w=0; w<nw; w++) s += wred[w];
          __syncthreads();
          return s;
        }

        // stateless random numbers, unit variance
        __device__ __forceinline__ unsigned long long
        mix64(unsigned long long x) {
          x += 0x9E3779B97F4A7C15ULL;
          x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
          x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
          return x ^ (x >> 31);
        }
        __device__ __forceinline__ double g1(unsigned long long h) {
          double u = double(mix64(h) >> 11) * (1.0 / 9007199254740992.0);
          return 1.7320508075688772 * (2.0*u - 1.0);
        }
        template<typename S> struct Rnd;
        template<> struct Rnd<float> {
          __device__ static float get(unsigned long long h) {
            return float(g1(h));
          }
        };
        template<> struct Rnd<double> {
          __device__ static double get(unsigned long long h) { return g1(h); }
        };
        template<typename R> struct Rnd<cplx<R>> {
          __device__ static cplx<R> get(unsigned long long h) {
            const double s = 0.7071067811865476;
            return {R(s*g1(h)), R(s*g1(h ^ 0xD1B54A32D192ED03ULL))};
          }
        };

        template<typename T> struct Dev { using type = T; };
        template<typename R> struct Dev<std::complex<R>> {
          using type = cplx<R>;
        };

        template<typename S, typename R> __global__ void
        ara_kernel(const int* rows, const int* cols,
                   S* const* M, const int* ldm,
                   S* const* Qp, const int* ldq,
                   S* const* Vp, const int* ldv,
                   int* ranks, const int* maxrv, int* info,
                   const R* tolv, R tols, int bs, int rstop,
                   unsigned long long seed) {
          extern __shared__ __align__(16) unsigned char dsm[];
          __shared__ S om[JC*NBLK];
          __shared__ R wred[32];
          S* coef = reinterpret_cast<S*>(dsm);
          const int t = blockIdx.x, tid = threadIdx.x;
          const int m = rows[t], n = cols[t];
          const int lda = ldm[t], ldQ = ldq[t], ldV = ldv[t];
          const S* A = M[t];
          S* Q = Qp[t];
          S* V = Vp[t];
          const R tol = tolv ? tolv[t] : tols;
          const int maxr = maxrv[t];
          const int mn = m < n ? m : n;
          const int cap = mn > 16 ? mn : 16;
          const int lane = tid % warpSize, wid = tid / warpSize;
          const int nw = NT / warpSize;
          int rank = 0, small = 0, iter = 0;
          bool fail = false, done = (m == 0 || n == 0);

          while (!done) {
            const int rank0 = rank;
            int nb = cap - rank0;
            if (nb > bs) nb = bs;
            if (nb > NBLK) nb = NBLK;
            if (nb <= 0) { fail = true; break; }
            // Y = A * Omega, stored in columns rank0 .. rank0+nb of Q
            for (int idx=tid; idx<m*nb; idx+=NT)
              Q[std::size_t(rank0 + idx/m)*ldQ + idx%m] = S();
            __syncthreads();
            for (int j0=0; j0<n; j0+=JC) {
              const int jc = (n-j0 < JC) ? n-j0 : JC;
              for (int idx=tid; idx<jc*nb; idx+=NT) {
                const int jj = idx / nb, k = idx % nb;
                unsigned long long h = seed
                  ^ (std::uint64_t(t+1) * 0x9E3779B97F4A7C15ULL)
                  ^ (std::uint64_t(iter+1) * 0xC2B2AE3D27D4EB4FULL)
                  ^ (std::uint64_t(j0+jj+1) * 0x165667B19E3779F9ULL)
                  ^ (std::uint64_t(k+1) * 0x27D4EB2F165667C5ULL);
                om[jj*NBLK + k] = Rnd<S>::get(h);
              }
              __syncthreads();
              for (int row=tid; row<m; row+=NT) {
                S s[NBLK];
#pragma unroll
                for (int k=0; k<NBLK; k++)
                  s[k] = (k < nb) ? Q[std::size_t(rank0+k)*ldQ + row] : S();
                for (int jj=0; jj<jc; jj++) {
                  const S a = A[std::size_t(j0+jj)*lda + row];
#pragma unroll
                  for (int k=0; k<NBLK; k++) s[k] += a * om[jj*NBLK + k];
                }
                for (int k=0; k<nb; k++)
                  Q[std::size_t(rank0+k)*ldQ + row] = s[k];
              }
              __syncthreads();
            }

            // orthogonalize the new vectors one by one
            int acc = 0;
            for (int k=0; k<nb && !done && !fail; k++) {
              S* y = Q + std::size_t(rank0+k)*ldQ;
              const int q = rank0 + acc;
              for (int pass=0; pass<2; pass++) {
                for (int j=wid; j<q; j+=nw) {
                  const S* qj = Q + std::size_t(j)*ldQ;
                  S s = S();
                  for (int i=lane; i<m; i+=warpSize) s += cj(qj[i]) * y[i];
                  s = warp_sum(s);
                  if (lane == 0) coef[j] = s;
                }
                __syncthreads();
                for (int i=tid; i<m; i+=NT) {
                  S v = y[i];
                  for (int j=0; j<q; j++)
                    v -= Q[std::size_t(j)*ldQ + i] * coef[j];
                  y[i] = v;
                }
                __syncthreads();
              }
              R ss = 0;
              for (int i=tid; i<m; i+=NT) ss += ab2(y[i]);
              const R nrm = sqrt(block_sum(ss, wred));
              if (nrm > 0 && nrm >= tol) {
                if (rank >= maxr) fail = true;
                else {
                  const R inv = R(1) / nrm;
                  S* dst = Q + std::size_t(rank0+acc)*ldQ;
                  for (int i=tid; i<m; i+=NT) dst[i] = scl(y[i], inv);
                  __syncthreads();
                  acc++; rank++; small = 0;
                }
              } else if (++small >= rstop) done = true;
            }
            iter++;
          }

          if (!fail) {
            // V = A^H Q. Thread tid owns column j0+tid of A. Chunks of
            // Q (JC rows x NBLK vectors) are staged in shared memory
            // (reusing om), same register tiling as the sampling.
            __syncthreads();
            for (int j0=0; j0<n; j0+=NT) {
              const int j = j0 + tid;
              const S* aj = A + std::size_t(j) * lda;
              for (int k0=0; k0<rank; k0+=NBLK) {
                const int kb = (rank-k0 < NBLK) ? rank-k0 : NBLK;
                S s[NBLK];
#pragma unroll
                for (int k=0; k<NBLK; k++) s[k] = S();
                for (int i0=0; i0<m; i0+=JC) {
                  const int ic = (m-i0 < JC) ? m-i0 : JC;
                  for (int idx=tid; idx<ic*NBLK; idx+=NT) {
                    const int ii = idx % ic, k = idx / ic;
                    om[ii*NBLK + k] = (k < kb) ?
                      Q[std::size_t(k0+k)*ldQ + i0 + ii] : S();
                  }
                  __syncthreads();
                  if (j < n) {
                    for (int ii=0; ii<ic; ii++) {
                      const S a = cj(aj[i0+ii]);
#pragma unroll
                      for (int k=0; k<NBLK; k++) s[k] += a * om[ii*NBLK + k];
                    }
                  }
                  __syncthreads();
                }
                if (j < n)
                  for (int k=0; k<kb; k++)
                    V[std::size_t(k0+k)*ldV + j] = s[k];
              }
            }
          }
          if (tid == 0) {
            ranks[t] = fail ? 0 : rank;
            info[t] = fail ? 1 : 0;
          }
        }

        // Per tile norms: either max_j ||M(:,j)||_2 combined with
        // rtol/atol (result = max(atol, rtol*norm)), or the Frobenius
        // norm. hypot accumulation avoids overflow/underflow.
        template<typename T, typename R> __global__ void
        norm_kernel(const int* rows, const int* cols, T* const* matrices,
                    const int* ld, R rtol, R atol, R* result, bool frobenius) {
          __shared__ R partial[32];
          const int tile = blockIdx.x, lane = threadIdx.x % warpSize;
          const int warp = threadIdx.x / warpSize;
          const int nw = blockDim.x / warpSize;
          R accum = 0;
          for (int col=warp; col<cols[tile]; col+=nw) {
            R norm = 0;
            for (int row=lane; row<rows[tile]; row+=warpSize)
              norm = combine(norm, magnitude(matrices[tile]
                             [std::size_t(col) * ld[tile] + row]));
            for (int o=warpSize/2; o; o/=2)
              norm = combine(norm, shfld(norm, o));
            if (lane == 0)
              accum = frobenius ? combine(accum, norm)
                : (accum > norm ? accum : norm);
          }
          if (lane == 0) partial[warp] = accum;
          __syncthreads();
          if (threadIdx.x == 0) {
            R norm = 0;
            for (int i=0; i<nw; i++)
              norm = frobenius ? combine(norm, partial[i])
                : (norm > partial[i] ? norm : partial[i]);
            const R relative = rtol * norm;
            result[tile] = frobenius ? norm
              : (atol > relative ? atol : relative);
          }
        }

        // Frobenius norm of one matrix per blockIdx.y, spread over
        // gridDim.x blocks. Writes one partial result per block
        // (hypot accumulation, avoids overflow/underflow).
        template<typename T, typename R> __global__ void
        front_norm_kernel(const int* rows, const int* cols,
                          T* const* matrices, const int* ld, R* partial) {
          __shared__ R wpart[32];
          const int mat = blockIdx.y;
          const std::size_t m = rows[mat], n = cols[mat], l = ld[mat];
          const T* A = matrices[mat];
          const std::size_t total = m * n;
          const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
          R acc = 0;
          for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x
                 + threadIdx.x; idx < total; idx += stride) {
            const std::size_t c = idx / m, r = idx - c * m;
            acc = combine(acc, magnitude(A[c * l + r]));
          }
          const int lane = threadIdx.x % warpSize;
          const int warp = threadIdx.x / warpSize;
          const int nw = blockDim.x / warpSize;
          for (int o=warpSize/2; o; o/=2) acc = combine(acc, shfld(acc, o));
          if (lane == 0) wpart[warp] = acc;
          __syncthreads();
          if (threadIdx.x == 0) {
            R norm = 0;
            for (int i=0; i<nw; i++) norm = combine(norm, wpart[i]);
            partial[std::size_t(mat) * gridDim.x + blockIdx.x] = norm;
          }
        }

        // ---- kernels for the batched-GEMM (MAGMA) based ARA, the GEMMs
        // are done on the host side (BLRBatch.cpp), these kernels do
        // the bookkeeping, the random numbers and the in-block QR.
        template<typename S> __global__ void
        magma_init_kernel(const int* rows, const int* cols, S* const* Q,
                          int* ranks, int* info, int* ymrow, int* nbv,
                          int* rkm, int* ldc, int* state, int* small,
                          S** omega, S** Y, S** C, S* omega_data,
                          S* c_data, int maxn, int bs, int cap, int B) {
          const int t = blockIdx.x * blockDim.x + threadIdx.x;
          if (t >= B) return;
          ranks[t] = 0;  info[t] = 0;  small[t] = 0;
          nbv[t] = 0;  rkm[t] = 0;  ymrow[t] = 0;  ldc[t] = cap;
          state[t] = (rows[t] <= 0 || cols[t] <= 0) ? 1 : 0;
          omega[t] = omega_data + std::size_t(t) * maxn * bs;
          C[t] = c_data + std::size_t(t) * cap * bs;
          Y[t] = Q[t];
        }

        // per round: decide the block size, set the dimensions of the
        // batched GEMMs (zero for finished tiles), draw Omega
        template<typename S> __global__ void
        magma_round_kernel(const int* rows, const int* cols, S* const* Q,
                           const int* ldq, int* ranks, int* info,
                           int* state, int* ymrow, int* nbv, int* rkm,
                           S** Y, S* const* omega, int* active, int bs,
                           int iter, unsigned long long seed) {
          __shared__ int snb;
          const int t = blockIdx.x, tid = threadIdx.x;
          if (tid == 0) {
            if (t == 0) *active = 0;
            int nb = 0;
            if (state[t] == 0) {
              const int m = rows[t], n = cols[t];
              int cap = m < n ? m : n;
              if (cap < 16) cap = 16;
              nb = cap - ranks[t];
              if (nb > bs) nb = bs;
              if (nb <= 0) {
                nb = 0;  state[t] = 2;  ranks[t] = 0;  info[t] = 1;
              }
            }
            nbv[t] = nb;
            ymrow[t] = nb > 0 ? rows[t] : 0;
            rkm[t] = nb > 0 ? ranks[t] : 0;
            Y[t] = Q[t] + std::size_t(ranks[t]) * ldq[t];
            snb = nb;
          }
          __syncthreads();
          const int nb = snb;
          if (nb == 0) return;
          const int n = cols[t];
          S* om = omega[t];
          for (int idx=tid; idx<n*nb; idx+=NT) {
            const int j = idx % n, k = idx / n;
            unsigned long long h = seed
              ^ (std::uint64_t(t+1) * 0x9E3779B97F4A7C15ULL)
              ^ (std::uint64_t(iter+1) * 0xC2B2AE3D27D4EB4FULL)
              ^ (std::uint64_t(j+1) * 0x165667B19E3779F9ULL)
              ^ (std::uint64_t(k+1) * 0x27D4EB2F165667C5ULL);
            om[idx] = Rnd<S>::get(h);
          }
        }

        // Y (columns rank .. rank+nb of Q) is already orthogonal to the
        // previous basis vectors, orthogonalize within the block,
        // accept/reject, update the rank
        template<typename S, typename R> __global__ void
        magma_inblock_kernel(const int* rows, S* const* Qp, const int* ldq,
                             int* ranks, int* info, const int* maxrv,
                             const R* tolv, const int* nbv, int* state,
                             int* small, int* active, int rstop) {
          __shared__ S coef[NBLK];
          __shared__ R wred[32];
          const int t = blockIdx.x, tid = threadIdx.x;
          const int nb = nbv[t];
          if (state[t] != 0 || nb == 0) return;
          const int m = rows[t], ldQ = ldq[t];
          S* Q = Qp[t];
          const R tol = tolv[t];
          const int maxr = maxrv[t];
          const int lane = tid % warpSize, wid = tid / warpSize;
          const int nw = NT / warpSize;
          int rank = ranks[t], sm = small[t];
          const int rank0 = rank;
          int acc = 0;
          bool fail = false, done = false;
          for (int k=0; k<nb && !done && !fail; k++) {
            S* y = Q + std::size_t(rank0+k)*ldQ;
            for (int pass=0; pass<2; pass++) {
              for (int j=wid; j<acc; j+=nw) {
                const S* qj = Q + std::size_t(rank0+j)*ldQ;
                S s = S();
                for (int i=lane; i<m; i+=warpSize) s += cj(qj[i]) * y[i];
                s = warp_sum(s);
                if (lane == 0) coef[j] = s;
              }
              __syncthreads();
              for (int i=tid; i<m; i+=NT) {
                S v = y[i];
                for (int j=0; j<acc; j++)
                  v -= Q[std::size_t(rank0+j)*ldQ + i] * coef[j];
                y[i] = v;
              }
              __syncthreads();
            }
            R ss = 0;
            for (int i=tid; i<m; i+=NT) ss += ab2(y[i]);
            const R nrm = sqrt(block_sum(ss, wred));
            if (nrm > 0 && nrm >= tol) {
              if (rank >= maxr) fail = true;
              else {
                const R inv = R(1) / nrm;
                S* dst = Q + std::size_t(rank0+acc)*ldQ;
                for (int i=tid; i<m; i+=NT) dst[i] = scl(y[i], inv);
                __syncthreads();
                acc++;  rank++;  sm = 0;
              }
            } else if (++sm >= rstop) done = true;
          }
          if (tid == 0) {
            ranks[t] = fail ? 0 : rank;
            small[t] = sm;
            if (fail) { state[t] = 2;  info[t] = 1; }
            else if (done) state[t] = 1;
            else atomicAdd(active, 1);
          }
        }

        // dimensions of the final GEMM V = A^H Q
        __global__ void
        magma_final_kernel(const int* cols, const int* ranks,
                           const int* state, int* ymrow, int* rkm, int B) {
          const int t = blockIdx.x * blockDim.x + threadIdx.x;
          if (t >= B) return;
          const bool fail = state[t] == 2;
          ymrow[t] = fail ? 0 : cols[t];
          rkm[t] = fail ? 0 : ranks[t];
        }

        // Copy Q to the front of the tile storage (U, m x rank) and
        // store V^H (rank x n) behind it. One block per tile.
        template<typename S> __global__ void
        finalize_kernel(const int* rows, const int* cols, S* const* M,
                        S* const* Qp, const int* ldq,
                        S* const* Vp, const int* ldv,
                        const int* ranks, const int* info) {
          const int t = blockIdx.x;
          if (info[t]) return;
          const int m = rows[t], n = cols[t], r = ranks[t];
          S* U = M[t];
          S* VH = U + std::size_t(m) * r;
          const S* Q = Qp[t];
          const S* V = Vp[t];
          const int lq = ldq[t], lv = ldv[t];
          for (int idx=threadIdx.x; idx<m*r; idx+=blockDim.x)
            U[idx] = Q[std::size_t(idx / m) * lq + idx % m];
          for (int idx=threadIdx.x; idx<n*r; idx+=blockDim.x) {
            const int k = idx / n, j = idx % n;
            VH[std::size_t(j) * r + k] = cj(V[std::size_t(k) * lv + j]);
          }
        }

        template<typename T, typename R> ARA_STREAM_T
        launch_norms(ARA_STREAM_T stream, const int* rows, const int* cols,
                     T* const* matrices, const int* ld, R rtol, R atol,
                     R* result, int count, bool frobenius) {
          using D = typename Dev<T>::type;
          norm_kernel<D, R><<<count, NT, 0, stream>>>
            (rows, cols, reinterpret_cast<D* const*>(matrices), ld,
             rtol, atol, result, frobenius);
          peek_at_last_error();
          return stream;
        }

        ARA_STREAM_T get_stream(Handle& handle) {
          ARA_STREAM_T stream;
          ARA_GET_STREAM(handle, stream);
          return stream;
        }

      } // end anonymous namespace

      template<typename scalar_t> void
      ara_workspace(Handle&, int, int) {
        // no workspace needed
      }

      template<typename scalar_t, typename real_t> void
      ara_tolerances(Handle& handle, const int* rows, const int* cols,
                     scalar_t* const* matrices, const int* ld,
                     real_t rtol, real_t atol,
                     real_t* tolerances, int count) {
        if (count < 0 || !std::isfinite(rtol) || !std::isfinite(atol) ||
            rtol < 0 || atol < 0)
          throw std::invalid_argument
            ("Invalid ARA compression tolerance or batch size");
        if (!count) return;
        launch_norms(get_stream(handle), rows, cols, matrices, ld,
                     rtol, atol, tolerances, count, false);
      }

      template<typename scalar_t, typename real_t> void
      ara_tol(Handle& handle, int* rows_batch, int* cols_batch,
              scalar_t** M_batch, int* ldm_batch,
              scalar_t** A_batch, int* lda_batch,
              scalar_t** B_batch, int* ldb_batch, int* ranks_batch,
              const real_t* tol_batch, int max_rows, int max_cols,
              int* max_rank, int bs, int r, int* info, int num_ops) {
        using S = typename Dev<scalar_t>::type;
        static std::atomic<unsigned long long> counter{0};
        if (num_ops <= 0) return;
        if (bs < 1) bs = 1;
        if (bs > NBLK) bs = NBLK;
        if (r < 1) r = 1;
        const int mn = max_rows < max_cols ? max_rows : max_cols;
        const std::size_t shmem = (std::size_t(mn) + 1) * sizeof(S);
        // static shared memory used by the kernel (om, wred)
        const std::size_t sshmem =
          sizeof(S)*JC*NBLK + 32*sizeof(real_t);
        if (shmem + sshmem > 48*1024)
          throw std::runtime_error("HIP ARA: tile too large");
        const unsigned long long seed =
          0x5DEECE66DULL + 7919ULL * counter.fetch_add(1);
        ara_kernel<S, real_t><<<num_ops, NT, shmem, get_stream(handle)>>>
          (rows_batch, cols_batch,
           reinterpret_cast<S* const*>(M_batch), ldm_batch,
           reinterpret_cast<S* const*>(A_batch), lda_batch,
           reinterpret_cast<S* const*>(B_batch), ldb_batch,
           ranks_batch, max_rank, info, tol_batch, real_t(0),
           bs, r, seed);
        peek_at_last_error();
      }

      template<typename scalar_t> void
      ara_finalize(Handle& handle, int* rows_batch, int* cols_batch,
                   scalar_t** M_batch, scalar_t** Q_batch, int* ldq_batch,
                   scalar_t** V_batch, int* ldv_batch,
                   int* ranks_batch, int* info, int num_ops) {
        using S = typename Dev<scalar_t>::type;
        if (num_ops <= 0) return;
        finalize_kernel<S><<<num_ops, NT, 0, get_stream(handle)>>>
          (rows_batch, cols_batch, reinterpret_cast<S* const*>(M_batch),
           reinterpret_cast<S* const*>(Q_batch), ldq_batch,
           reinterpret_cast<S* const*>(V_batch), ldv_batch,
           ranks_batch, info);
        peek_at_last_error();
      }

      template<typename scalar_t, typename real_t> void
      ara_magma_init(Handle& handle, AraMagma<scalar_t,real_t>& a) {
        using S = typename Dev<scalar_t>::type;
        static std::atomic<unsigned long long> counter{0};
        a.seed = 0x5DEECE66DULL + 7919ULL * counter.fetch_add(1);
        if (a.B <= 0) return;
        magma_init_kernel<S><<<(a.B+255)/256, 256, 0, get_stream(handle)>>>
          (a.rows, a.cols, reinterpret_cast<S* const*>(a.Q), a.ranks,
           a.info, a.ymrow, a.nbv, a.rkm, a.ldc, a.state, a.small,
           reinterpret_cast<S**>(a.omega), reinterpret_cast<S**>(a.Y),
           reinterpret_cast<S**>(a.C), reinterpret_cast<S*>(a.omega_data),
           reinterpret_cast<S*>(a.c_data), a.maxn, a.bs, a.cap, a.B);
        peek_at_last_error();
      }

      template<typename scalar_t, typename real_t> void
      ara_magma_round(Handle& handle, const AraMagma<scalar_t,real_t>& a,
                      int iter) {
        using S = typename Dev<scalar_t>::type;
        if (a.B <= 0) return;
        magma_round_kernel<S><<<a.B, NT, 0, get_stream(handle)>>>
          (a.rows, a.cols, reinterpret_cast<S* const*>(a.Q), a.ldq,
           a.ranks, a.info, a.state, a.ymrow, a.nbv, a.rkm,
           reinterpret_cast<S**>(a.Y),
           reinterpret_cast<S* const*>(a.omega), a.active, a.bs, iter,
           a.seed);
        peek_at_last_error();
      }

      template<typename scalar_t, typename real_t> void
      ara_magma_inblock(Handle& handle, const AraMagma<scalar_t,real_t>& a) {
        using S = typename Dev<scalar_t>::type;
        if (a.B <= 0) return;
        magma_inblock_kernel<S, real_t><<<a.B, NT, 0, get_stream(handle)>>>
          (a.rows, reinterpret_cast<S* const*>(a.Q), a.ldq, a.ranks,
           a.info, a.maxr, a.tol, a.nbv, a.state, a.small, a.active,
           a.rstop);
        peek_at_last_error();
      }

      template<typename scalar_t, typename real_t> void
      ara_magma_final(Handle& handle, const AraMagma<scalar_t,real_t>& a) {
        if (a.B <= 0) return;
        magma_final_kernel<<<(a.B+255)/256, 256, 0, get_stream(handle)>>>
          (a.cols, a.ranks, a.state, a.ymrow, a.rkm, a.B);
        peek_at_last_error();
      }

      template<typename T> typename RealType<T>::value_type
      front_norm(const DenseMatrix<T>& F11, const DenseMatrix<T>& F12,
                 const DenseMatrix<T>& F21, void* dwork) {
        using R = typename RealType<T>::value_type;
        const DenseMatrix<T>* fronts[] = {&F11, &F12, &F21};
        int dims[9];
        T* pointers[3];
        for (int i = 0; i < 3; i++) {
          const auto& F = *fronts[i];
          if (F.rows() > std::size_t(std::numeric_limits<int>::max()) ||
              F.cols() > std::size_t(std::numeric_limits<int>::max()) ||
              F.ld() > std::size_t(std::numeric_limits<int>::max()))
            throw std::overflow_error
              ("ARA front dimensions exceed 32-bit indices");
          dims[i] = F.rows(); dims[i+3] = F.cols(); dims[i+6] = F.ld();
          pointers[i] = const_cast<T*>(F.data());
        }
        // workspace layout, see front_norm_worksize
        int* ddims = static_cast<int*>(dwork);
        T** dptrs = aligned_ptr<T*>(ddims + 9);
        R* dnorms = aligned_ptr<R>(dptrs + 3);
        copy_host_to_device<int>(ddims, dims, 9);
        copy_host_to_device<T*>(dptrs, pointers, 3);
        using D = typename Dev<T>::type;
        front_norm_kernel<D, R>
          <<<dim3(front_norm_blocks, 3), NT, 0, nullptr>>>
          (ddims, ddims+3, reinterpret_cast<D* const*>(dptrs), ddims+6,
           dnorms);
        peek_at_last_error();
        synchronize_default_stream(); // launched on the default stream
        std::vector<R> parts(3 * front_norm_blocks);
        copy_device_to_host<R>(parts.data(), dnorms, parts.size());
        R norms[3] = {0, 0, 0};
        for (int i=0; i<3; i++)
          for (int b=0; b<front_norm_blocks; b++)
            norms[i] = std::hypot(norms[i], parts[i*front_norm_blocks+b]);
        return std::hypot(std::hypot(norms[0], norms[1]), norms[2]);
      }

#define STRUMPACK_GPU_ARA_INST(T, R)                                    \
      template void ara_workspace<T>(Handle&, int, int);                \
      template void ara_tolerances<T, R>                                \
      (Handle&, const int*, const int*, T* const*, const int*,          \
       R, R, R*, int);                                                  \
      template void ara_tol<T, R>                                       \
      (Handle&, int*, int*, T**, int*, T**, int*, T**, int*, int*,      \
       const R*, int, int, int*, int, int, int*, int);                  \
      template void ara_finalize<T>                                     \
      (Handle&, int*, int*, T**, T**, int*, T**, int*, int*, int*, int); \
      template void ara_magma_init<T, R>(Handle&, AraMagma<T,R>&);       \
      template void ara_magma_round<T, R>                               \
      (Handle&, const AraMagma<T,R>&, int);                             \
      template void ara_magma_inblock<T, R>                             \
      (Handle&, const AraMagma<T,R>&);                                  \
      template void ara_magma_final<T, R>                               \
      (Handle&, const AraMagma<T,R>&);                                  \
      template R front_norm<T>                                          \
      (const DenseMatrix<T>&, const DenseMatrix<T>&,                    \
       const DenseMatrix<T>&, void*);
      STRUMPACK_GPU_ARA_INST(float, float)
      STRUMPACK_GPU_ARA_INST(double, double)
      STRUMPACK_GPU_ARA_INST(std::complex<float>, float)
      STRUMPACK_GPU_ARA_INST(std::complex<double>, double)
#undef STRUMPACK_GPU_ARA_INST

    } // end namespace kblas
  } // end namespace gpu
} // end namespace strumpack
