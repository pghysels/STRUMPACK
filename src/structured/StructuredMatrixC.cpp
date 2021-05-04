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
#include <complex.h>
#undef I

#include "StructuredMatrix.h"
#include "StructuredMatrix.hpp"
using namespace strumpack;
using namespace strumpack::structured;

extern "C" {

  void SP_s_struct_default_options(CSPOptions* opts) {
  }
  void SP_d_struct_default_options(CSPOptions* opts) {
  }
  void SP_c_struct_default_options(CSPOptions* opts) {
  }
  void SP_z_struct_default_options(CSPOptions* opts) {
  }

  void SP_s_struct_destroy(CSPStructMat* S) {
    auto H = reinterpret_cast<StructuredMatrix<float>*>(*S);
    delete H;
    H = NULL;
  }
  void SP_d_struct_destroy(CSPStructMat* S) {
    auto H = reinterpret_cast<StructuredMatrix<double>*>(*S);
    delete H;
    H = NULL;
  }
  void SP_c_struct_destroy(CSPStructMat* S) {
    auto H = reinterpret_cast<StructuredMatrix<float _Complex>*>(*S);
    delete H;
    H = NULL;
  }
  void SP_z_struct_destroy(CSPStructMat* S) {
    auto H = reinterpret_cast<StructuredMatrix<double _Complex>*>(*S);
    delete H;
    H = NULL;
  }

  int SP_s_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             float* A, int ldA, CSPOptions* opts) {
    // TODO set from opts
    StructuredOptions<float> o;
    try {
      auto H = construct_from_dense<float>(rows, cols, A, ldA, o);
      *S = H.release();
    } catch (std::exception& e) { return 1; }
    return 0;
  }
  int SP_d_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             double* A, int ldA, CSPOptions* opts) {
    // TODO set from opts
    StructuredOptions<double> o;
    o.set_type(Type::HSS);
    try {
      auto H = construct_from_dense<double>(rows, cols, A, ldA, o);
      *S = H.release();
    } catch (std::exception& e) { return 1; }
    return 0;
  }
  int SP_c_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             float _Complex* A, int ldA, CSPOptions* opts) {
    // TODO set from opts
    StructuredOptions<std::complex<float>> o;
    try {
      auto H = construct_from_dense<std::complex<float>>
        (rows, cols, reinterpret_cast<std::complex<float>*>(A), ldA, o);
      *S = H.release();
    } catch (std::exception& e) { return 1; }
    return 0;
  }
  int SP_z_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             double _Complex* A, int ldA, CSPOptions* opts) {
    // TODO set from opts
    StructuredOptions<std::complex<double>> o;
    try {
      auto H = construct_from_dense<std::complex<double>>
        (rows, cols, reinterpret_cast<std::complex<double>*>(A), ldA, o);
      *S = H.release();
    } catch (std::exception& e) { return 1; }
    return 0;
  }



  int SP_s_struct_mult(CSPStructMat S, char trans, int m, float* B, int ldB, float* C, int ldC) {
    auto H = reinterpret_cast<StructuredMatrix<float>*>(S);
    H->mult(c2T(trans), m, B, ldB, C, ldC);
    return 0;
  }
  int SP_d_struct_mult(CSPStructMat S, char trans, int m, double* B, int ldB, double* C, int ldC) {
    auto H = reinterpret_cast<StructuredMatrix<double>*>(S);
    H->mult(c2T(trans), m, B, ldB, C, ldC);
    return 0;
  }
  int SP_c_struct_mult(CSPStructMat S, char trans, int m, float _Complex* B, int ldB, float _Complex* C, int ldC) {
    auto H = reinterpret_cast<StructuredMatrix<std::complex<float>>*>(S);
    H->mult(c2T(trans), m, reinterpret_cast<std::complex<float>*>(B), ldB,
            reinterpret_cast<std::complex<float>*>(C), ldC);
    return 0;
  }
  int SP_z_struct_mult(CSPStructMat S, char trans, int m, double _Complex* B, int ldB, double _Complex* C, int ldC) {
    auto H = reinterpret_cast<StructuredMatrix<std::complex<double>>*>(S);
    H->mult(c2T(trans), m, reinterpret_cast<std::complex<double>*>(B), ldB,
            reinterpret_cast<std::complex<double>*>(C), ldC);
    return 0;
  }

  int SP_s_struct_factor(CSPStructMat S) { return 0; }
  int SP_d_struct_factor(CSPStructMat S) { return 0; }
  int SP_c_struct_factor(CSPStructMat S) { return 0; }
  int SP_z_struct_factor(CSPStructMat S) { return 0; }

  int SP_s_struct_solve(CSPStructMat S, int nrhs, float* B, int ldB) { return 0; }
  int SP_d_struct_solve(CSPStructMat S, int nrhs, double* B, int ldB) { return 0; }
  int SP_c_struct_solve(CSPStructMat S, int nrhs, float _Complex* B, int ldB) { return 0; }
  int SP_z_struct_solve(CSPStructMat S, int nrhs, double _Complex* B, int ldB) { return 0; }

}

