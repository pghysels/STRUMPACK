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
/*! \file StructuredMatrix.h
 * \brief Contains the MPI structured matrix C interfaces, see
 * StructuredMatrix.hpp for the C++ interface.
 */
#ifndef STRUMPACK_STRUCTURED_MPI_C_H
#define STRUMPACK_STRUCTURED_MPI_C_H

#include "StructuredMatrix.h"

#define OMPI_SKIP_MPICXX 1
#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(SWIG)
  int SP_s_struct_from_dense2d(CSPStructMat* S, const MPI_Comm comm,
                               int rows, int cols, const float* A,
                               int IA, int JA, int* DESCA,
                               const CSPOptions* opts);
  int SP_d_struct_from_dense2d(CSPStructMat* S, const MPI_Comm comm,
                               int rows, int cols, const double* A,
                               int IA, int JA, int* DESCA,
                               const CSPOptions* opts);
  int SP_c_struct_from_dense2d(CSPStructMat* S, const MPI_Comm comm,
                               int rows, int cols, const float _Complex* A,
                               int IA, int JA, int* DESCA,
                               const CSPOptions* opts);
  int SP_z_struct_from_dense2d(CSPStructMat* S, const MPI_Comm comm,
                               int rows, int cols, const double _Complex* A,
                               int IA, int JA, int* DESCA,
                               const CSPOptions* opts);

  /**
   * Construct a structured matrix using a routine to compute
   * individual elements, using MPI. Should be called by all ranks in
   * the communicator.
   *
   * \param S Pointer to CSPStructMat object, which will be constructed.
   * \param comm communicator
   * \param rows Number of rows in A
   * \param cols Number of columns in A
   * \param A Matrix element routine, returning A(i,j)
   * \param opts Options structure, needs to be initialized by the
   * user.
   * \return 0 if successful
   *
   * \see SP_s_struct_destroy, SP_s_struct_default_options
   */
  int SP_s_struct_from_elements_mpi(CSPStructMat* S, const MPI_Comm comm,
                                    int rows, int cols,
                                    float A(int i, int j),
                                    const CSPOptions* opts);
  int SP_d_struct_from_elements_mpi(CSPStructMat* S, const MPI_Comm comm,
                                    int rows, int cols,
                                    double A(int i, int j),
                                    const CSPOptions* opts);
  int SP_c_struct_from_elements_mpi(CSPStructMat* S, const MPI_Comm comm,
                                    int rows, int cols,
                                    float _Complex A(int i, int j),
                                    const CSPOptions* opts);
  int SP_z_struct_from_elements_mpi(CSPStructMat* S, const MPI_Comm comm,
                                    int rows, int cols,
                                    double _Complex A(int i, int j),
                                    const CSPOptions* opts);
#endif

  /**
   * For a 1d distributed structured matrix, return the local number
   * of rows assigned to this process.
   * \return number of local rows
   */
  int SP_s_struct_local_rows(const CSPStructMat S);
  int SP_d_struct_local_rows(const CSPStructMat S);
  int SP_c_struct_local_rows(const CSPStructMat S);
  int SP_z_struct_local_rows(const CSPStructMat S);


  /**
   * For a 1d distributed structured matrix, return the first row
   * assigned to this process.
   * \return first row in 1d block row distribution
   */
  int SP_s_struct_begin_row(const CSPStructMat S);
  int SP_d_struct_begin_row(const CSPStructMat S);
  int SP_c_struct_begin_row(const CSPStructMat S);
  int SP_z_struct_begin_row(const CSPStructMat S);

#ifdef __cplusplus
}
#endif

#endif
