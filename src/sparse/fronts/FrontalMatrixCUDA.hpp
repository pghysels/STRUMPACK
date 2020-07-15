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
#ifndef FRONTAL_MATRIX_CUDA_HPP
#define FRONTAL_MATRIX_CUDA_HPP

#include "misc/Triplet.hpp"


namespace strumpack {

  namespace cuda {

    template<typename T> struct AssembleData {
      AssembleData(int d1_, int d2_, T* F11_, T* F12_, T* F21_, T* F22_,
                   int n11_, int n12_, int n21_,
                   Triplet<T>* e11_, Triplet<T>* e12_, Triplet<T>* e21_)
        : d1(d1_), d2(d2_), F11(F11_), F12(F12_), F21(F21_), F22(F22_),
          n11(n11_), n12(n12_), n21(n21_), e11(e11_), e12(e12_), e21(e21_) {}

      // sizes and pointers for this front
      int d1, d2;
      T *F11, *F12, *F21, *F22;

      // sparse matrix elements
      int n11, n12, n21;
      Triplet<T> *e11, *e12, *e21;

      // info for extend add
      T *CB1 = nullptr, *CB2 = nullptr;
      int dCB1 = 0, dCB2 = 0;
      std::size_t *I1 = nullptr, *I2 = nullptr;
    };

    template<typename T> struct FrontData {
      FrontData(int n1_, int n2_, T* F11_, T* F12_,
                T* F21_, T* F22_, int* piv_)
        : n1(n1_), n2(n2_), F11(F11_), F12(F12_),
          F21(F21_), F22(F22_), piv(piv_) {}
      int n1, n2;
      T *F11, *F12, *F21, *F22;
      int* piv;
    };


    template<typename T> void assemble(unsigned int, AssembleData<T>*);

    template<typename T> void extend_add(unsigned int, AssembleData<T>*);

    template<typename T, int NT=32> void
    factor_block_batch(unsigned int, FrontData<T>*);

  } // end namespace cuda
} // end namespace strumpack

#endif // FRONTAL_MATRIX_CUDA_HPP
