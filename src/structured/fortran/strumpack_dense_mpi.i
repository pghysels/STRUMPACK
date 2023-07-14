%module strumpack_dense_mpi;

%insert(fbegin)
%{
! STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The
! Regents of the University of California, through Lawrence Berkeley
! National Laboratory (subject to receipt of any required approvals
! from the U.S. Dept. of Energy).  All rights reserved.
!
! If you have questions about your rights to use or distribute this
! software, please contact Berkeley Lab's Technology Transfer
! Department at TTD@lbl.gov.
!
! NOTICE. This software is owned by the U.S. Department of Energy. As
! such, the U.S. Government has been granted for itself and others
! acting on its behalf a paid-up, nonexclusive, irrevocable,
! worldwide license in the Software to reproduce, prepare derivative
! works, and perform publicly and display publicly.  Beginning five
! (5) years after the date permission to assert copyright is obtained
! from the U.S. Department of Energy, and subject to any subsequent
! five (5) year renewals, the U.S. Government is granted for itself
! and others acting on its behalf a paid-up, nonexclusive,
! irrevocable, worldwide license in the Software to reproduce,
! prepare derivative works, distribute copies to the public, perform
! publicly and display publicly, and to permit others to do so.
!
! Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
!             (Lawrence Berkeley National Lab, Computational Research
!             Division).
!
!> @file strumpack_dense_mpi.f90
!> @brief Fortran interface to the structured matrix functionality
!%}
%include <complex.i>

// Translate all enums and other compile-time constants into fortran parameters
%fortranconst;
// Provide interfaces to functions and `struct`s rather than wrapping them
%fortranbindc;

// Allow this struct to be passed natively between C and Fortran
%fortran_struct(CSPOptions)

%rename("SP_s_struct_from_dense2d") SP_s_struct_from_dense2d_f;
%rename("SP_d_struct_from_dense2d") SP_d_struct_from_dense2d_f;
%rename("SP_c_struct_from_dense2d") SP_c_struct_from_dense2d_f;
%rename("SP_z_struct_from_dense2d") SP_z_struct_from_dense2d_f;

%rename("SP_s_struct_from_elements_mpi") SP_s_struct_from_elements_mpi_f;
%rename("SP_d_struct_from_elements_mpi") SP_d_struct_from_elements_mpi_f;
%rename("SP_c_struct_from_elements_mpi") SP_c_struct_from_elements_mpi_f;
%rename("SP_z_struct_from_elements_mpi") SP_z_struct_from_elements_mpi_f;

%inline %{
typedef void* CSPStructMat;
#include "../StructuredMatrixMPI.h"

#ifdef __cplusplus
extern "C" {
#endif
int SP_s_struct_from_dense2d_f(CSPStructMat* S, int comm,
                               int rows, int cols, const float* A,
                               int IA, int JA, int* DESCA,
                               const CSPOptions* opts) {
  return SP_s_struct_from_dense2d
    (S, MPI_Comm_f2c(comm), rows, cols, A, IA, JA, DESCA, opts);
}
int SP_d_struct_from_dense2d_f(CSPStructMat* S, int comm,
                               int rows, int cols, const double* A,
                               int IA, int JA, int* DESCA,
                               const CSPOptions* opts) {
  return SP_d_struct_from_dense2d
    (S, MPI_Comm_f2c(comm), rows, cols, A, IA, JA, DESCA, opts);
}
int SP_c_struct_from_dense2d_f(CSPStructMat* S, int comm,
                               int rows, int cols, const float _Complex* A,
                               int IA, int JA, int* DESCA,
                               const CSPOptions* opts) {
  return SP_c_struct_from_dense2d
    (S, MPI_Comm_f2c(comm), rows, cols, A, IA, JA, DESCA, opts);
}
int SP_z_struct_from_dense2d_f(CSPStructMat* S, int comm,
                               int rows, int cols, const double _Complex* A,
                               int IA, int JA, int* DESCA,
                               const CSPOptions* opts) {
  return SP_z_struct_from_dense2d
    (S, MPI_Comm_f2c(comm), rows, cols, A, IA, JA, DESCA, opts);
}
int SP_s_struct_from_elements_mpi_f(CSPStructMat* S, int comm,
                                    int rows, int cols,
                                    float A(int i, int j),
                                    const CSPOptions* opts) {
  return SP_s_struct_from_elements_mpi
    (S, MPI_Comm_f2c(comm), rows, cols, A, opts);
}
int SP_d_struct_from_elements_mpi_f(CSPStructMat* S, int comm,
                                    int rows, int cols,
                                    double A(int i, int j),
                                    const CSPOptions* opts) {
  return SP_d_struct_from_elements_mpi
    (S, MPI_Comm_f2c(comm), rows, cols, A, opts);
}
int SP_c_struct_from_elements_mpi_f(CSPStructMat* S, int comm,
                                    int rows, int cols,
                                    float _Complex A(int i, int j),
                                    const CSPOptions* opts) {
  return SP_c_struct_from_elements_mpi
    (S, MPI_Comm_f2c(comm), rows, cols, A, opts);
}
int SP_z_struct_from_elements_mpi_f(CSPStructMat* S, int comm,
                                    int rows, int cols,
                                    double _Complex A(int i, int j),
                                    const CSPOptions* opts) {
  return SP_z_struct_from_elements_mpi
    (S, MPI_Comm_f2c(comm), rows, cols, A, opts);
}

#ifdef __cplusplus
}
#endif

%}

// Process and create wrappers for the following header file
%include "../StructuredMatrixMPI.h"
