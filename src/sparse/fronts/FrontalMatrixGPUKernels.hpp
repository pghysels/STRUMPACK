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
#ifndef FRONTAL_MATRIX_GPU_KERNELS_HPP
#define FRONTAL_MATRIX_GPU_KERNELS_HPP

#include "misc/Triplet.hpp"
#if defined(STRUMPACK_USE_CUDA)
#include "dense/CUDAWrapper.hpp"
#endif
#if defined(STRUMPACK_USE_HIP)
#include "dense/HIPWrapper.hpp"
#endif

namespace strumpack {
  namespace gpu {

    template<typename T> struct AssembleData {
      AssembleData(int d1_, int d2_, T* F11_, T* F12_, T* F21_, T* F22_,
                   std::size_t n11_, std::size_t n12_, std::size_t n21_,
                   Triplet<T>* e11_, Triplet<T>* e12_, Triplet<T>* e21_)
        : d1(d1_), d2(d2_), F11(F11_), F12(F12_), F21(F21_), F22(F22_),
          n11(n11_), n12(n12_), n21(n21_), e11(e11_), e12(e12_), e21(e21_) {}
      AssembleData(int d1_, int d2_, T* F11_, T* F21_)
        : d1(d1_), d2(d2_), F11(F11_), F21(F21_) {}

      // sizes and pointers for this front
      int d1 = 0, d2 = 0;
      T *F11 = nullptr, *F12 = nullptr, *F21 = nullptr, *F22 = nullptr;

      // info for extend add
      int dCB1 = 0, dCB2 = 0;
      T *CB1 = nullptr, *CB2 = nullptr;
      std::size_t *I1 = nullptr, *I2 = nullptr;

      // sparse matrix elements
      std::size_t n11 = 0, n12 = 0, n21 = 0;
      Triplet<T> *e11 = nullptr, *e12 = nullptr, *e21 = nullptr;

      void set_ext_add_left(int dCB, T* CB, std::size_t* I) {
        dCB1 = dCB;
        CB1 = CB;
        I1 = I;
      }
      void set_ext_add_right(int dCB, T* CB, std::size_t* I) {
        dCB2 = dCB;
        CB2 = CB;
        I2 = I;
      }
    };

    template<typename T> struct FrontData {
      FrontData() {}
      FrontData(int n1_, int n2_, T* F11_, T* F12_,
                T* F21_, T* F22_, int* piv_)
        : n1(n1_), n2(n2_), F11(F11_), F12(F12_),
          F21(F21_), F22(F22_), piv(piv_) {}
      int n1, n2;
      T *F11, *F12, *F21, *F22;
      int* piv;
    };

    template<typename T> void
    assemble(unsigned int, AssembleData<T>*, AssembleData<T>*);

    template<typename T, int NT=32,
             typename real_t = typename RealType<T>::value_type>
    void factor_block_batch(unsigned int, FrontData<T>*, bool, real_t, int*);

    template<typename T,
             typename real_t = typename RealType<T>::value_type>
    void replace_pivots(int, T*, real_t, gpu::Stream* = nullptr);

    template<typename T,
             typename real_t = typename RealType<T>::value_type>
    void replace_pivots_vbatched(BLASHandle& handle, int* dn, int max_n,
                                 T** dA, int* lddA, real_t thresh,
                                 unsigned int batchCount);

    template<typename T> void
    extend_add_rhs(int, int, unsigned int, AssembleData<T>*, AssembleData<T>*);

    template<typename T> void
    extract_rhs(int, int, unsigned int, AssembleData<T>*, AssembleData<T>*);

    // constexpr
    inline int align_max_struct() {
      auto m = sizeof(std::complex<double>);
      m = std::max(m, sizeof(gpu::FrontData<std::complex<double>>));
      m = std::max(m, sizeof(gpu::AssembleData<std::complex<double>>));
      m = std::max(m, sizeof(Triplet<std::complex<double>>));
      int k = 16;
      while (k < int(m)) k *= 2;
      return k;
    }

    inline std::size_t round_up(std::size_t n) {
      int k = align_max_struct();
      return std::size_t((n + k - 1) / k) * k;
    }
    template<typename T> T* aligned_ptr(void* p) {
      return (T*)(round_up(uintptr_t(p)));
    }

  } // end namespace gpu
} // end namespace strumpack

#endif // FRONTAL_MATRIX_GPU_KERNELS_HPP
