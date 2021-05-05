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
#ifndef STRUMPACK_STRUCTURED_C_H
#define STRUMPACK_STRUCTURED_C_H

#include <stdint.h>
#include <complex.h>
#undef I

#include "StrumpackConfig.hpp"

#if defined(STRUMPACK_USE_MPI)
#define OMPI_SKIP_MPICXX 1
#include "mpi.h"
#endif

/**
 * Enumeration of possible structured matrix types.
 * See structured::Type
 */
typedef enum {
              SP_TYPE_HSS = 0,
              SP_TYPE_BLR,
              SP_TYPE_HODLR,
              SP_TYPE_HODBF,
              SP_TYPE_BUTTERFLY,
              SP_TYPE_LR,
              SP_TYPE_LOSSY,
              SP_TYPE_LOSSLESS
} SP_STRUCTURED_TYPE;


typedef struct CSPOptions {
  SP_STRUCTURED_TYPE type;
  double rel_tol;
  double abs_tol;
  int leaf_size;
  int max_rank;
  int verbose;
} CSPOptions;


typedef void* CSPStructMat;


#ifdef __cplusplus
extern "C" {
#endif

  void SP_s_struct_default_options(CSPOptions* opts);
  void SP_d_struct_default_options(CSPOptions* opts);
  void SP_c_struct_default_options(CSPOptions* opts);
  void SP_z_struct_default_options(CSPOptions* opts);

  void SP_s_struct_destroy(CSPStructMat* S);
  void SP_d_struct_destroy(CSPStructMat* S);
  void SP_c_struct_destroy(CSPStructMat* S);
  void SP_z_struct_destroy(CSPStructMat* S);

  int SP_s_struct_from_dense(CSPStructMat* S, int rows, int cols, float* A, int ldA, CSPOptions* opts);
  int SP_d_struct_from_dense(CSPStructMat* S, int rows, int cols, double* A, int ldA, CSPOptions* opts);
  int SP_c_struct_from_dense(CSPStructMat* S, int rows, int cols, float _Complex* A, int ldA, CSPOptions* opts);
  int SP_z_struct_from_dense(CSPStructMat* S, int rows, int cols, double _Complex* A, int ldA, CSPOptions* opts);

  int SP_s_struct_mult(CSPStructMat S, char trans, int m, float* B, int ldB, float* C, int ldC);
  int SP_d_struct_mult(CSPStructMat S, char trans, int m, double* B, int ldB, double* C, int ldC);
  int SP_c_struct_mult(CSPStructMat S, char trans, int m, float _Complex* B, int ldB, float _Complex* C, int ldC);
  int SP_z_struct_mult(CSPStructMat S, char trans, int m, double _Complex* B, int ldB, double _Complex* C, int ldC);

  int SP_s_struct_factor(CSPStructMat S);
  int SP_d_struct_factor(CSPStructMat S);
  int SP_c_struct_factor(CSPStructMat S);
  int SP_z_struct_factor(CSPStructMat S);

  int SP_s_struct_solve(CSPStructMat S, int nrhs, float* B, int ldB);
  int SP_d_struct_solve(CSPStructMat S, int nrhs, double* B, int ldB);
  int SP_c_struct_solve(CSPStructMat S, int nrhs, float _Complex* B, int ldB);
  int SP_z_struct_solve(CSPStructMat S, int nrhs, double _Complex* B, int ldB);

#ifdef __cplusplus
}
#endif

#endif
