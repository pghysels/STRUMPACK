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
#include <sycl/sycl.hpp>
#include <complex>
#include <iostream>
#include <complex>

#define STRUMPACK_NO_TRIPLET_MPI
#include "FrontalMatrixGPUKernels.hpp"
#include "dense/SYCLWrapper.hpp"


namespace strumpack {
  namespace gpu {

    /**
     * Get the real T type corresponding to a scalar, for instance T,
     * std::complex<T> or thrust::complex<T>, to be used for instance
     * to compute norms or absolute value.
     */
    template<class T> struct real_type { typedef T value_type; };
    template<class T> struct real_type<std::complex<T>> { typedef T value_type; };

    float real_part(float& a) { return a; }
    double real_part(double& a) { return a; }
    float real_part(std::complex<float> &a) { return a.real(); }
    double real_part(std::complex<double> &a) { return a.real(); }

    float absolute_value(float &a) { return std::abs(a); }
    double absolute_value(double &a) { return std::abs(a); }
    float absolute_value(std::complex<float> &a) { return std::abs(a); }
    double absolute_value(std::complex<double> &a) { return std::abs(a); }
    // float absolute_value(std::complex<float> &a) { return thrust::abs(a); }
    // double absolute_value(std::complex<double> &a) { return thrust::abs(a); }

    /**
     * Put elements of the sparse matrix in the F11, F12 and F21 parts
     * of the front.  The sparse elements are taken from F.e11, F.e12,
     * F.e21, which are lists of triplets {r,c,v}. The front is
     * assumed to be initialized to zero.
     *
     */
    template<typename T> struct Assemble {
      AssembleData<T>* dat;
      std::size_t nf;
      Assemble(AssembleData<T>* d, std::size_t N) : dat(d), nf(N) {}
      void operator()(const sycl::nd_item<2>& it) const {
        std::size_t op = it.get_global_id(0);
        if (op >= nf) return;
        auto& F = dat[op];
        auto idx = it.get_global_id(1);
        if (idx < F.n11) {
          auto& t = F.e11[idx];
          F.F11[t.r + t.c*F.d1] = t.v;
        }
        if (idx < F.n12) {
          auto& t = F.e12[idx];
          F.F12[t.r + t.c*F.d1] = t.v;
        }
        if (idx < F.n21) {
          auto& t = F.e21[idx];
          F.F21[t.r + t.c*F.d2] = t.v;
        }
      }
    };


    /**
     * Single extend-add operation from one contribution block into
     * the parent front. d1 is the size of F11, d2 is the size of F22.
     */
    template<typename T, unsigned int unroll> struct EA {
      AssembleData<T>* dat;
      bool left;
      std::size_t nf;
      EA(AssembleData<T>* d, std::size_t N, bool l)
        : dat(d), nf(N), left(l) {}
      void operator()(const sycl::nd_item<3>& it) const {
        int y = it.get_global_id(2),
          x0 = it.get_group(1) * unroll,
          z = it.get_global_id(0);
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
    };

    template<typename T> T rnd(T a, T b) { return ((a + b - 1) / b) * b; }

    template <typename T>
    void assemble(unsigned int N, AssembleData<T> *dat,
                  AssembleData<T> *ddat) {
      sycl::queue &q = get_sycl_queue();
      { // front assembly from sparse matrix
        std::size_t nnz = 0;
        for (std::size_t f=0; f<N; f++)
          nnz = std::max
            (nnz, std::size_t(std::max(dat[f].n11,
                                       std::max(dat[f].n12, dat[f].n21))));
        if (nnz) {
          // TODO unroll
          std::size_t nt = 512, ops = 1;
          while (nt > nnz && ops < 64) {
            nt /= 2;
            ops *= 2;
          }
          // assert(rnd(std::size_t(N),ops) * rnd(nnz,nt) < std::numeric_limits<int>::max());
          sycl::range<2> global{rnd(std::size_t(N),ops),
              rnd(nnz,nt)}, local{ops, nt};
          q.parallel_for(sycl::nd_range<2>{global, local},
                         Assemble<T>(ddat, N));
        }
      }
      { // extend-add
        std::size_t gCB = 0;
        for (std::size_t f=0; f<N; f++)
          gCB = std::max(gCB, std::size_t(std::max(dat[f].dCB1, dat[f].dCB2)));
        if (gCB) {
          std::size_t nt = 256, ops = 1;
          const std::size_t unroll = 16;
          while (nt > gCB && ops < 64) {
            nt /= 2;
            ops *= 2;
          }
          std::size_t gx = (gCB + unroll - 1) / unroll;
          gCB = rnd(gCB, nt);
          // assert(gCB * gx * rnd(N,ops) < std::numeric_limits<int>::max());
          sycl::range<3> global{rnd(std::size_t(N), ops), gx, gCB},
            local{ops, 1, nt};
          q.parallel_for(sycl::nd_range<3>{global, local},
                         EA<T,unroll>(ddat, N, true));
          q.parallel_for(sycl::nd_range<3>{global, local},
                         EA<T,unroll>(ddat, N, false));
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
    template<typename T, int NT, typename real_t> void
    LU_block_kernel(int n, T* F, int* piv, int* info,
                    const sycl::nd_item<3> &item_ct1, int &p,
                    T* M, real_t &Mmax, real_t *cabs) {
      int j = item_ct1.get_local_id(2), i = item_ct1.get_local_id(1);
      if (i == 0 && j == 0)
        *info = 0;

      // copy F from global device storage into shared memory
      if (i < n && j < n)
        M[i+j*NT] = F[i+j*n];
      /*
        DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
        performance if there is no access to global memory.
      */
      item_ct1.barrier();

      for (int k=0; k<n; k++) {
        // only 1 thread looks for the pivot element
        // this should be optimized?
        if (j == k && i >= k)
          cabs[i] = absolute_value(M[i+j*NT]);
        /*
          DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
          sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
          better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (j == k && i == k) {
          p = k;
          Mmax = cabs[k];
          for (int l=k+1; l<n; l++) {
            auto tmp = cabs[l];
            if (tmp > Mmax) {
              Mmax = tmp;
              p = l;
            }
          }
          piv[k] = p + 1;
        }
        /*
          DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
          sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
          better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (Mmax == T(0.)) {
          if (j == k && i == k && *info == 0)
            *info = k;
        } else {
          // swap row k with the pivot row
          if (j < n && i == k && p != k) {
            auto tmp = M[k+j*NT];
            M[k+j*NT] = M[p+j*NT];
            M[p+j*NT] = tmp;
          }
          /*
            DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
          */
          item_ct1.barrier();
          // divide by the pivot element
          if (j == k && i > k && i < n)
            M[i+k*NT] /= M[k+k*NT];
          /*
            DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
          */
          item_ct1.barrier();
          // Schur update
          if (j > k && i > k && j < n && i < n)
            M[i+j*NT] -= M[i+k*NT] * M[k+j*NT];
          /*
            DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
          */
          item_ct1.barrier();
        }
      }
      // write back from shared to global device memory
      if (i < n && j < n)
        F[i+j*n] = M[i+j*NT];
    }

    template<typename T, int NT, typename real_t> void
    LU_block_kernel_batched(FrontData<T>* dat, bool replace,
                            real_t thresh, int* dinfo,
                            const sycl::nd_item<3> &item_ct1, int &p,
                            T* M, real_t &Mmax, real_t *cabs) {
      FrontData<T> &A = dat[item_ct1.get_group(2)];
      LU_block_kernel<T, NT>(A.n1, A.F11, A.piv, &dinfo[item_ct1.get_group(2)],
                             item_ct1, p, M, Mmax, cabs);
      if (replace) {
        int i = item_ct1.get_local_id(2), j = item_ct1.get_local_id(1);
        if (i == j && i < A.n1) {
          std::size_t k = i + i*A.n1;
          if (absolute_value(A.F11[k]) < thresh)
            A.F11[k] = (real_part(A.F11[k]) < 0) ? -thresh : thresh;
        }
      }
    }

    template<typename T, typename real_t> void
    replace_pivots_kernel(int n, T* A, real_t thresh,
                          const sycl::nd_item<3> &item_ct1) {
      int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
      if (i < n) {
        std::size_t k = i + i*n;
        if (absolute_value(A[k]) < thresh)
          A[k] = (real_part(A[k]) < 0) ? -thresh : thresh;
      }
    }

    template<typename T, typename real_t>
    void replace_pivots(int n, T* A, real_t thresh, gpu::Stream* s) {
      if (!n) return;
      int NT = 128;
      if (s)
        /*
          DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
          limit. To get the device limit, query info::device::max_work_group_size.
          Adjust the work-group size if needed.
        */
        get_sycl_queue(*s).parallel_for
          (sycl::nd_range<3>((n + NT - 1) / NT * sycl::range<3>(1, 1, NT),
                             sycl::range<3>(1, 1, NT)),
           [=](sycl::nd_item<3> item_ct1) {
            replace_pivots_kernel<T, real_t>(n, A, thresh, item_ct1);
          });
      else
        /*
          DPCT1049:10: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        get_sycl_queue().parallel_for
          (sycl::nd_range<3>(sycl::range<3>(1, 1, (n + NT - 1) / NT) *
                             sycl::range<3>(1, 1, NT),
                             sycl::range<3>(1, 1, NT)),
           [=](sycl::nd_item<3> item_ct1) {
            replace_pivots_kernel<T, real_t>(n, A, thresh, item_ct1);
          });
    }

    template<typename T, typename real_t> void
    replace_pivots_vbatch_kernel(int* dn, T** dA, int* lddA, real_t thresh,
                                 unsigned int batchCount,
                                 const sycl::nd_item<3> &item_ct1) {
      int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2),
        f = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
        item_ct1.get_local_id(1);
      if (f >= batchCount) return;
      if (i >= dn[f]) return;
      auto A = dA[f];
      auto ldA = lddA[f];
      std::size_t ii = i + i*ldA;
      if (absolute_value(A[ii]) < thresh)
        A[ii] = (real_part(A[ii]) < 0) ? -thresh : thresh;
    }

    template<typename T, typename real_t>
    void replace_pivots_vbatched(Handle& handle, int* dn, int max_n,
                                 T** dA, int* lddA, real_t thresh,
                                 unsigned int batchCount) {
      if (max_n <= 0 || !batchCount) return;
      unsigned int nt = 512, ops = 1;
      while (nt > max_n) {
        nt /= 2;
        ops *= 2;
      }
      ops = std::min(ops, batchCount);
      unsigned int nbx = (max_n + nt - 1) / nt,
        nbf = (batchCount + ops - 1) / ops;
      sycl::range<3> block(1, ops, nt);
      for (unsigned int f=0; f<nbf; f+=MAX_BLOCKS_Y) {
        std::cout << "TODO fix" << std::endl;
        sycl::range<3> grid(nbx, std::min(nbf - f, MAX_BLOCKS_Y), 1);
        auto f0 = f * ops;
        /*
          DPCT1049:11: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        get_sycl_queue(handle).parallel_for
          (sycl::nd_range<3>(grid * block, block),
           [=](sycl::nd_item<3> item_ct1) {
            replace_pivots_vbatch_kernel(dn + f0, dA + f0, lddA + f0,
                                         thresh, batchCount - f0, item_ct1);
          });
      }
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
    template<typename T, int NT> void
    solve_block_kernel(int n, int m, T* F, T* X, int* piv,
                       const sycl::nd_item<3> &item_ct1, int *P, T* A, T* B) {
      int j = item_ct1.get_local_id(2), i = item_ct1.get_local_id(1);
      if (j == 0)
        P[i] = i;
      /*
        DPCT1065:12: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
        performance if there is no access to global memory.
      */
      item_ct1.barrier();
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
      /*
        DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
        performance if there is no access to global memory.
      */
      item_ct1.barrier();

      // loop over blocks of NT columns of X
      for (int b=0; b<m; b+=NT) {
        int c = b + j;

        // put X in shared memory, while applying the permutation
        if (i < n && c < m)
          B[j+i*NT] = X[P[i]+c*n];
        /*
          DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
          sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
          better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // solve with L (unit diagonal)
        for (int k=0; k<n; k++) {
          if (i > k && i < n && c < m)
            B[j+i*NT] -= A[k+i*NT] * B[j+k*NT];
          /*
            DPCT1065:15: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
          */
          item_ct1.barrier();
        }

        // solve with U
        for (int k=n-1; k>=0; k--) {
          if (i == k && c < m)
            B[j+i*NT] /= A[i+i*NT];
          /*
            DPCT1065:16: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
          */
          item_ct1.barrier();
          if (i < k && c < m)
            B[j+i*NT] -= A[k+i*NT] * B[j+k*NT];
          /*
            DPCT1065:17: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
          */
          item_ct1.barrier();
        }

        // write from shared back to global device memory
        if (i < n && c < m)
          X[i+c*n] = B[j+i*NT];
      }
    }

    template<typename T, int NT> void
    solve_block_kernel_batched(FrontData<T>* dat,
                               const sycl::nd_item<3> &item_ct1, int *P,
                               T* A_, T* B_) {
      FrontData<T> &A = dat[item_ct1.get_group(2)];
      solve_block_kernel<T, NT>(A.n1, A.n2, A.F11, A.F12, A.piv, item_ct1, P, A_, B_);
    }


    /**
     * Compute F -= F21 * F12, where F is d2 x d2 and F12 is d1 x d2.
     * d1 is <= NT. This should be called with a single NT x NT thread
     * block.
     */
    template<typename T, int NT> void
    Schur_block_kernel(int d1, int d2, T* F12, T* F21, T* F22,
                       const sycl::nd_item<3> &item_ct1, T* B, T* A) {
      int j = item_ct1.get_local_id(2), i = item_ct1.get_local_id(1);
      A[j+i*NT] = B[j+i*NT] = 0.;
      for (int cb=0; cb<d2; cb+=NT) {
        int c = cb + j;
        // put NT columns of F12 in shared memory B
        if (i < d1 && c < d2)
          B[j+i*NT] = F12[i+c*d1];
        /*
          DPCT1065:18: Consider replacing sycl::nd_item::barrier() with
          sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
          better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        for (int rb=0; rb<d2; rb+=NT) {
          int r = rb + i;
          // put NT rows of F21 in shared memory A
          if (r < d2 && j < d1)
            A[j+i*NT] = F21[r+j*d2];
          /*
            DPCT1065:19: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
          */
          item_ct1.barrier(); // wait for A and B
          if (c < d2 && r < d2) {
            T tmp(0.);
            // k < d1 <= NT, by using k<NT this can be unrolled
            for (int k=0; k<NT; k++)
              tmp += A[k+i*NT] * B[j+k*NT];
            F22[r+c*d2] -= tmp;
          }
          /*
            DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
          */
          item_ct1.barrier(); // sync before reading new A/B
        }
      }
    }

    template<typename T, int NT> void
    Schur_block_kernel_batched(FrontData<T>* dat,
                               const sycl::nd_item<3> &item_ct1,
                               T* B_, T* A_) {
      FrontData<T> &A = dat[item_ct1.get_group(2)];
      Schur_block_kernel<T, NT>(A.n1, A.n2, A.F12, A.F21, A.F22, item_ct1, B_, A_);
    }

    template <typename T, int NT, typename real_t>
    void factor_block_batch(unsigned int count, FrontData<T> *dat, bool replace,
                            real_t thresh, int *dinfo) {
      sycl::queue &q_ct1 = get_sycl_queue();
      if (!count) return;
      sycl::range<3> block(1, NT, NT); //, grid(count, 1, 1);
      /*
        DPCT1049:21: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
      */
      q_ct1.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<int, 0> p_acc_ct1(cgh);
          sycl::local_accessor<T, 1> M__acc_ct1(sycl::range<1>(NT * NT), cgh);
          sycl::local_accessor<real_t, 0> Mmax_acc_ct1(cgh);
          sycl::local_accessor<real_t, 1> cabs_acc_ct1(sycl::range<1>(NT), cgh);
          cgh.parallel_for
            (sycl::nd_range<3>(sycl::range<3>(1, 1, count) * block, block),
             [=](sycl::nd_item<3> item_ct1) {
              LU_block_kernel_batched<T, NT, real_t>
                (dat, replace, thresh, dinfo, item_ct1, p_acc_ct1,
                 M__acc_ct1.template get_multi_ptr<sycl::access::decorated::yes>().get(),
		 Mmax_acc_ct1,
                 cabs_acc_ct1.template get_multi_ptr<sycl::access::decorated::yes>().get());
            });
        });
      /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
      */
      q_ct1.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<int, 1> P_acc_ct1(sycl::range<1>(NT), cgh);
          sycl::local_accessor<T, 1> A__acc_ct1(sycl::range<1>(NT * NT), cgh);
          sycl::local_accessor<T, 1> B__acc_ct1(sycl::range<1>(NT * NT), cgh);
          cgh.parallel_for
            (sycl::nd_range<3>(sycl::range<3>(1, 1, count) * block, block),
             [=](sycl::nd_item<3> item_ct1) {
              solve_block_kernel_batched<T, NT>
                (dat, item_ct1,
		 P_acc_ct1.template get_multi_ptr<sycl::access::decorated::yes>().get(),
                 A__acc_ct1.template get_multi_ptr<sycl::access::decorated::yes>().get(),
		 B__acc_ct1.template get_multi_ptr<sycl::access::decorated::yes>().get());
            });
        });
      /*
        DPCT1049:23: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
      */
      q_ct1.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<T, 1> B__acc_ct1(sycl::range<1>(NT * NT), cgh);
          sycl::local_accessor<T, 1> A__acc_ct1(sycl::range<1>(NT * NT), cgh);
          cgh.parallel_for
            (sycl::nd_range<3>(sycl::range<3>(1, 1, count) * block, block),
             [=](sycl::nd_item<3> item_ct1) {
              Schur_block_kernel_batched<T, NT>
                (dat, item_ct1,
		 B__acc_ct1.template get_multi_ptr<sycl::access::decorated::yes>().get(),
                 A__acc_ct1.template get_multi_ptr<sycl::access::decorated::yes>().get());
            });
        });
    }


    template<typename T, int NT> void
    solve_block_kernel_batched(int nrhs, FrontData<T>* dat,
                               const sycl::nd_item<3> &item_ct1, int *P,
                               T* A_, T* B_) {
      FrontData<T> &A = dat[item_ct1.get_group(2)];
      solve_block_kernel<T, NT>(A.n1, nrhs, A.F11, A.F12, A.piv, item_ct1, P,
                                A_, B_);
    }

    /**
     * Single extend-add operation along the column dimension, for the
     * solve.  d1 is the size of F11, d2 is the size of F22.
     */
    template<typename T> void
    ea_rhs_kernel(int r, int N, int nrhs,
                  int dsep, int dupd, int dCB,
                  T* b, T* bupd, T* CB, std::size_t* I) {
      if (r >= dCB) return;
      auto Ir = I[r];
      for (int c=0; c<nrhs; c++)
        if (Ir < dsep) b[Ir+c*N] += CB[r+c*dCB];
        else bupd[Ir-dsep+c*dupd] += CB[r+c*dCB];
    }

    template<typename T> void
    extend_add_rhs_kernel_left
    (int N, int nrhs, unsigned int nf, AssembleData<T>* dat,
     const sycl::nd_item<3> &item_ct1) {
      int r = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2),
        i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
        item_ct1.get_local_id(1);
      if (i >= nf) return;
      auto& f = dat[i];
      if (f.CB1)
        ea_rhs_kernel(r, N, nrhs, f.d1, f.d2, f.dCB1,
                      f.F11, f.F21, f.CB1, f.I1);
    }
    template<typename T> void
    extend_add_rhs_kernel_right
    (int N, int nrhs, unsigned int nf, AssembleData<T>* dat,
     const sycl::nd_item<3> &item_ct1) {
      int r = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2),
        i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
        item_ct1.get_local_id(1);
      if (i >= nf) return;
      auto& f = dat[i];
      if (f.CB2)
        ea_rhs_kernel(r, N, nrhs, f.d1, f.d2, f.dCB2,
                      f.F11, f.F21, f.CB2, f.I2);
    }

    template <typename T>
    void extend_add_rhs(int N, int nrhs, unsigned int nf, AssembleData<T> *dat,
                        AssembleData<T> *ddat) {
      sycl::queue &q_ct1 = get_sycl_queue();
      int du = 0;
      for (unsigned int f=0; f<nf; f++)
        du = std::max(du, std::max(dat[f].dCB1, dat[f].dCB2));
      if (!du) return;
      unsigned int nt = 512, ops = 1;
      while (nt > du && ops < 64) {
        nt /= 2;
        ops *= 2;
      }
      ops = std::min(ops, nf);
      unsigned int nb = (du + nt - 1) / nt, nbf = (nf + ops - 1) / ops;
      sycl::range<3> block(1, ops, nt);
      for (unsigned int f=0; f<nbf; f+=MAX_BLOCKS_Z) {
        std::cout << "TODO fix" << std::endl;
        sycl::range<3> grid(nb, std::min(nbf - f, MAX_BLOCKS_Z), 1);
        /*
          DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for
          (sycl::nd_range<3>(grid * block, block),
           [=](sycl::nd_item<3> item_ct1) {
            extend_add_rhs_kernel_left(N, nrhs, nf - f * ops, ddat + f * ops, item_ct1);
          });
        /*
          DPCT1049:25: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for
          (sycl::nd_range<3>(grid * block, block),
           [=](sycl::nd_item<3> item_ct1) {
            extend_add_rhs_kernel_right(N, nrhs, nf - f * ops, ddat + f * ops, item_ct1);
          });
      }
    }


    /**
     * Single extend-add operation along the column dimension, for the
     * solve.  d1 is the size of F11, d2 is the size of F22.
     */
    template<typename T> void
    extract_rhs_kernel(int r, int N, int nrhs,
                       int dsep, int dupd, int dCB,
                       T* b, T* bupd, T* CB, std::size_t* I) {
      //const sycl::nd_item<3> &item_ct1) {
      if (r >= dCB) return;
      auto Ir = I[r];
      for (int c=0; c<nrhs; c++)
        if (Ir < dsep) CB[r+c*dCB] = b[Ir+c*N];
        else CB[r+c*dCB] = bupd[Ir-dsep+c*dupd];
    }

    template<typename T> void
    extract_rhs_kernel(int N, int nrhs, unsigned int nf,
                       AssembleData<T>* dat, const sycl::nd_item<3> &item_ct1) {
      int r = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2),
        i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
        item_ct1.get_local_id(1);
      if (i >= nf) return;
      auto& f = dat[i];
      if (f.CB1)
        extract_rhs_kernel(r, N, nrhs, f.d1, f.d2, f.dCB1, f.F11,
                           f.F21, f.CB1, f.I1);
      if (f.CB2)
        extract_rhs_kernel(r, N, nrhs, f.d1, f.d2, f.dCB2, f.F11,
                           f.F21, f.CB2, f.I2);
    }

    template<typename T> void
    extract_rhs(int N, int nrhs, unsigned int nf, AssembleData<T>* dat,
                AssembleData<T>* ddat) {
      int du = 0;
      for (unsigned int f=0; f<nf; f++)
        du = std::max(du, std::max(dat[f].dCB1, dat[f].dCB2));
      if (!du) return;
      unsigned int nt = 512, ops = 1;
      while (nt > du && ops < 64) {
        nt /= 2;
        ops *= 2;
      }
      ops = std::min(ops, nf);
      unsigned int nb = (du + nt - 1) / nt, nbf = (nf + ops - 1) / ops;
      sycl::range<3> block(1, ops, nt);
      for (unsigned int f=0; f<nbf; f+=MAX_BLOCKS_Z) {
        std::cout << "TODO fix" << std::endl;
        sycl::range<3> grid(nb, std::min(nbf - f, MAX_BLOCKS_Z), 1);
        /*
          DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        get_sycl_queue().parallel_for
          (sycl::nd_range<3>(grid * block, block),
           [=](sycl::nd_item<3> item_ct1) {
            extract_rhs_kernel(N, nrhs, nf - f * ops, ddat + f * ops, item_ct1);
          });
      }
    }


    // explicit template instantiations
    template void assemble(unsigned int, AssembleData<float>*, AssembleData<float>*);
    template void assemble(unsigned int, AssembleData<double>*, AssembleData<double>*);
    template void assemble(unsigned int, AssembleData<std::complex<float>>*, AssembleData<std::complex<float>>*);
    template void assemble(unsigned int, AssembleData<std::complex<double>>*, AssembleData<std::complex<double>>*);

    template void extend_add_rhs(int, int, unsigned int, AssembleData<float>*, AssembleData<float>*);
    template void extend_add_rhs(int, int, unsigned int, AssembleData<double>*, AssembleData<double>*);
    template void extend_add_rhs(int, int, unsigned int, AssembleData<std::complex<float>>*, AssembleData<std::complex<float>>*);
    template void extend_add_rhs(int, int, unsigned int, AssembleData<std::complex<double>>*, AssembleData<std::complex<double>>*);

    template void extract_rhs(int, int, unsigned int, AssembleData<float>*, AssembleData<float>*);
    template void extract_rhs(int, int, unsigned int, AssembleData<double>*, AssembleData<double>*);
    template void extract_rhs(int, int, unsigned int, AssembleData<std::complex<float>>*, AssembleData<std::complex<float>>*);
    template void extract_rhs(int, int, unsigned int, AssembleData<std::complex<double>>*, AssembleData<std::complex<double>>*);


    template void factor_block_batch<float,8,float>(unsigned int, FrontData<float>*, bool, float, int*);
    template void factor_block_batch<double,8,double>(unsigned int, FrontData<double>*, bool, double, int*);
    template void factor_block_batch<std::complex<float>,8,float>(unsigned int, FrontData<std::complex<float>>*, bool, float, int*);
    template void factor_block_batch<std::complex<double>,8,double>(unsigned int, FrontData<std::complex<double>>*, bool, double, int*);

    template void factor_block_batch<float,16,float>(unsigned int, FrontData<float>*, bool, float, int*);
    template void factor_block_batch<double,16,double>(unsigned int, FrontData<double>*, bool, double, int*);
    template void factor_block_batch<std::complex<float>,16,float>(unsigned int, FrontData<std::complex<float>>*, bool, float, int*);
    template void factor_block_batch<std::complex<double>,16,double>(unsigned int, FrontData<std::complex<double>>*, bool, double, int*);

    template void factor_block_batch<float,24,float>(unsigned int, FrontData<float>*, bool, float, int*);
    template void factor_block_batch<double,24,double>(unsigned int, FrontData<double>*, bool, double, int*);
    template void factor_block_batch<std::complex<float>,24,float>(unsigned int, FrontData<std::complex<float>>*, bool, float, int*);
    template void factor_block_batch<std::complex<double>,24,double>(unsigned int, FrontData<std::complex<double>>*, bool, double, int*);

    template void factor_block_batch<float,32,float>(unsigned int, FrontData<float>*, bool, float, int*);
    template void factor_block_batch<double,32,double>(unsigned int, FrontData<double>*, bool, double, int*);
    template void factor_block_batch<std::complex<float>,32,float>(unsigned int, FrontData<std::complex<float>>*, bool, float, int*);
    template void factor_block_batch<std::complex<double>,32,double>(unsigned int, FrontData<std::complex<double>>*, bool, double, int*);

    template void replace_pivots(int, float*, float, gpu::Stream*);
    template void replace_pivots(int, double*, double, gpu::Stream*);
    template void replace_pivots(int, std::complex<float>*, float, gpu::Stream*);
    template void replace_pivots(int, std::complex<double>*, double, gpu::Stream*);

    template void replace_pivots_vbatched(Handle&, int*, int, float**, int*, float, unsigned int);
    template void replace_pivots_vbatched(Handle&, int*, int, double**, int*, double, unsigned int);
    template void replace_pivots_vbatched(Handle&, int*, int, std::complex<float>**, int*, float, unsigned int);
    template void replace_pivots_vbatched(Handle&, int*, int, std::complex<double>**, int*, double, unsigned int);

  } // end namespace gpu
} // end namespace strumpack
