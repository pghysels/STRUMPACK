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
/*!
 * \file Kernel.h
 *
 * \brief C interface to the Kernel classes.
 */
#ifndef STRUMPACK_C_KERNEL_HPP
#define STRUMPACK_C_KERNEL_HPP

typedef void* STRUMPACKKernel;

#ifdef __cplusplus
extern "C" {
#endif

  STRUMPACKKernel STRUMPACK_create_kernel_double
  (int n, int d, double* train, double h, double lambda, int p, int type);
  STRUMPACKKernel STRUMPACK_create_kernel_float
  (int n, int d, float* train, float h, float lambda, int p, int type);

  void STRUMPACK_destroy_kernel_double(STRUMPACKKernel K);
  void STRUMPACK_destroy_kernel_float(STRUMPACKKernel K);

  void STRUMPACK_kernel_fit_HSS_double
  (STRUMPACKKernel K, double* labels, int argc, char* argv[]);
  void STRUMPACK_kernel_fit_HSS_float
  (STRUMPACKKernel K, float* labels, int argc, char* argv[]);

#if defined(STRUMPACK_USE_MPI)
  /**
   * TODO should pass an MPIComm object from python
   */
  void STRUMPACK_kernel_fit_HSS_MPI_double
  (STRUMPACKKernel K, double* labels, int argc, char* argv[]);
  void STRUMPACK_kernel_fit_HSS_MPI_float
  (STRUMPACKKernel K, float* labels, int argc, char* argv[]);

  /**
   * TODO should pass an MPIComm object from python
   */
  void STRUMPACK_kernel_fit_HODLR_MPI_double
  (STRUMPACKKernel K, double* labels, int argc, char* argv[]);
  void STRUMPACK_kernel_fit_HODLR_MPI_float
  (STRUMPACKKernel K, float* labels, int argc, char* argv[]);
#endif

  /**
   * This works for both sequential and MPI fits.
   */
  void STRUMPACK_kernel_predict_double
  (STRUMPACKKernel K, int m, double* test, double* prediction);
  void STRUMPACK_kernel_predict_float
  (STRUMPACKKernel K, int m, float* test, float* prediction);

#ifdef __cplusplus
}
#endif

#endif //STRUMPACK_C_KERNEL_HPP
