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
 * \brief Contains the structured matrix C interfaces, see
 * StructuredMatrix.hpp for the C++ interface.
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


/**
 * \struct CSPOptions
 * \brief Structure containing options for structured matrix
 * compression, to be used with the C interface.
 *
 * This mimics the options found in the C++ class
 * structured::StructuredOptions.
 *
 * \see structured::StructuredOptions
 */
typedef struct CSPOptions {
  SP_STRUCTURED_TYPE type;
  double rel_tol;
  double abs_tol;
  int leaf_size;
  int max_rank;
  int verbose;
} CSPOptions;


/**
 * \brief Type representing a structured matrix in the C interface.
 *
 * See any of the SP_x_struct... routines. Internally this will be
 * represented as a structured::StructuredMatrix object (implemented
 * in C++).
 */
typedef void* CSPStructMat;


#ifdef __cplusplus
extern "C" {
#endif

  // TODO:
  // - query statistics: memory usage, levels, rank, ...
  // - more construction routines, matvec, ..
  // - pass clustertree?
  // - in HODLR/HODBF store the MPIComm instead of a pointer!!

  /**
   * Fill the options structure with default values. Use this for
   * single precision, real.
   * \param opts Pointer to CSPOptions structure
   */
  void SP_s_struct_default_options(CSPOptions* opts);
  /**
   * Fill the options structure with default values. Use this for
   * double precision, real.
   * \param opts Pointer to CSPOptions structure
   */
  void SP_d_struct_default_options(CSPOptions* opts);
  /**
   * Fill the options structure with default values. Use this for
   * single precision, complex.
   * \param opts Pointer to CSPOptions structure
   */
  void SP_c_struct_default_options(CSPOptions* opts);
  /**
   * Fill the options structure with default values. Use this for
   * double precision, complex.
   * \param opts Pointer to CSPOptions structure
   */
  void SP_z_struct_default_options(CSPOptions* opts);


  /**
   * Destroy the structured matrix. Use this for a structured matrix
   * created with one of the SP_s_struct_... routines, i.e., single
   * precision, real.
   * \param S Pointer to CSPStructMat object.
   */
  void SP_s_struct_destroy(CSPStructMat* S);
  /**
   * Destroy the structured matrix. Use this for a structured matrix
   * created with one of the SP_d_struct_... routines, i.e., double
   * precision, real.
   * \param S Pointer to CSPStructMat object.
   */
  void SP_d_struct_destroy(CSPStructMat* S);
  /**
   * Destroy the structured matrix. Use this for a structured matrix
   * created with one of the SP_c_struct_... routines, i.e., single
   * precision, complex.
   * \param S Pointer to CSPStructMat object.
   */
  void SP_c_struct_destroy(CSPStructMat* S);
  /**
   * Destroy the structured matrix. Use this for a structured matrix
   * created with one of the SP_z_struct_... routines, i.e., double
   * precision, complex.
   * \param S Pointer to CSPStructMat object.
   */
  void SP_z_struct_destroy(CSPStructMat* S);


  /**
   * Return number of rows in the structured matrix.
   * \return Number of rows of S.
   */
  int SP_s_struct_rows(const CSPStructMat S);
  /**
   * Return number of rows in the structured matrix.
   * \return Number of rows of S.
   */
  int SP_d_struct_rows(const CSPStructMat S);
  /**
   * Return number of rows in the structured matrix.
   * \return Number of rows of S.
   */
  int SP_c_struct_rows(const CSPStructMat S);
  /**
   * Return number of rows in the structured matrix.
   * \return Number of rows of S.
   */
  int SP_z_struct_rows(const CSPStructMat S);


  /**
   * Return number of cols in the structured matrix.
   * \return Number of cols of S.
   */
  int SP_s_struct_cols(const CSPStructMat S);
  /**
   * Return number of cols in the structured matrix.
   * \return Number of cols of S.
   */
  int SP_d_struct_cols(const CSPStructMat S);
  /**
   * Return number of cols in the structured matrix.
   * \return Number of cols of S.
   */
  int SP_c_struct_cols(const CSPStructMat S);
  /**
   * Return number of cols in the structured matrix.
   * \return Number of cols of S.
   */
  int SP_z_struct_cols(const CSPStructMat S);


  /**
   * Return the total amount of memory used by this matrix, in
   * bytes.
   * \return Memory usage in bytes.
   * \see nonzeros
   */
  long long int SP_s_struct_memory(const CSPStructMat S);
  /**
   * Return the total amount of memory used by this matrix, in
   * bytes.
   * \return Memory usage in bytes.
   * \see nonzeros
   */
  long long int SP_d_struct_memory(const CSPStructMat S);
  /**
   * Return the total amount of memory used by this matrix, in
   * bytes.
   * \return Memory usage in bytes.
   * \see nonzeros
   */
  long long int SP_c_struct_memory(const CSPStructMat S);
  /**
   * Return the total amount of memory used by this matrix, in
   * bytes.
   * \return Memory usage in bytes.
   * \see nonzeros
   */
  long long int SP_z_struct_memory(const CSPStructMat S);


  /**
   * Return the total number of nonzeros stored by this matrix.
   * \return Nonzeros in the matrix representation.
   * \see memory
   */
  long long int SP_s_struct_nonzeros(const CSPStructMat S);
  /**
   * Return the total number of nonzeros stored by this matrix.
   * \return Nonzeros in the matrix representation.
   * \see memory
   */
  long long int SP_d_struct_nonzeros(const CSPStructMat S);
  /**
   * Return the total number of nonzeros stored by this matrix.
   * \return Nonzeros in the matrix representation.
   * \see memory
   */
  long long int SP_c_struct_nonzeros(const CSPStructMat S);
  /**
   * Return the total number of nonzeros stored by this matrix.
   * \return Nonzeros in the matrix representation.
   * \see memory
   */
  long long int SP_z_struct_nonzeros(const CSPStructMat S);


  /**
   * Return the maximum rank of this matrix over all low-rank
   * compressed blocks.
   * \return Maximum rank.
   */
  int SP_s_struct_rank(const CSPStructMat S);
  /**
   * Return the maximum rank of this matrix over all low-rank
   * compressed blocks.
   * \return Maximum rank.
   */
  int SP_d_struct_rank(const CSPStructMat S);
  /**
   * Return the maximum rank of this matrix over all low-rank
   * compressed blocks.
   * \return Maximum rank.
   */
  int SP_c_struct_rank(const CSPStructMat S);
  /**
   * Return the maximum rank of this matrix over all low-rank
   * compressed blocks.
   * \return Maximum rank.
   */
  int SP_z_struct_rank(const CSPStructMat S);


  /**
   * Construct a structured matrix from a dense (column major)
   * matrix. Use this for single precision, real.
   *
   * \param S Pointer to CSPStructMat object, which will be constructed.
   * \param rows Number of rows in A
   * \param cols Number of columns in A
   * \param A Pointer to data of A, column major
   * \param ldA leading dimension of A
   * \param opts Options structure, needs to be initialized by the
   * user.
   * \return 0 if successful
   *
   * \see SP_s_struct_destroy, SP_s_struct_default_options
   */
  int SP_s_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             const float* A, int ldA,
                             const CSPOptions* opts);
  /**
   * Construct a structured matrix from a dense (column major)
   * matrix. Use this for double precision, real.
   *
   * \param S Pointer to CSPStructMat object, which will be constructed.
   * \param rows Number of rows in A
   * \param cols Number of columns in A
   * \param A Pointer to data of A, column major
   * \param ldA leading dimension of A
   * \param opts Options structure, needs to be initialized by the
   * user.
   * \return 0 if successful
   *
   * \see SP_d_struct_destroy, SP_d_struct_default_options
   */
  int SP_d_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             const double* A, int ldA,
                             const CSPOptions* opts);
  /**
   * Construct a structured matrix from a dense (column major)
   * matrix. Use this for single precision, complex.
   *
   * \param S Pointer to CSPStructMat object, which will be constructed.
   * \param rows Number of rows in A
   * \param cols Number of columns in A
   * \param A Pointer to data of A, column major
   * \param ldA leading dimension of A
   * \param opts Options structure, needs to be initialized by the
   * user.
   * \return 0 if successful
   *
   * \see SP_c_struct_destroy, SP_c_struct_default_options
   */
  int SP_c_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             const float _Complex* A, int ldA,
                             const CSPOptions* opts);
  /**
   * Construct a structured matrix from a dense (column major)
   * matrix. Use this for double precision, complex.
   *
   * \param S Pointer to CSPStructMat object, which will be constructed.
   * \param rows Number of rows in A
   * \param cols Number of columns in A
   * \param A Pointer to data of A, column major
   * \param ldA leading dimension of A
   * \param opts Options structure, needs to be initialized by the
   * user.
   * \return 0 if successful
   *
   * \see SP_z_struct_destroy, SP_z_struct_default_options
   */
  int SP_z_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             const double _Complex* A, int ldA,
                             const CSPOptions* opts);


  int SP_s_struct_from_elements(CSPStructMat* S, int rows, int cols,
                                float(int i, int j),
                                const CSPOptions* opts);
  int SP_d_struct_from_elements(CSPStructMat* S, int rows, int cols,
                                double(int i, int j),
                                const CSPOptions* opts);
  int SP_c_struct_from_elements(CSPStructMat* S, int rows, int cols,
                                float _Complex(int i, int j),
                                const CSPOptions* opts);
  int SP_z_struct_from_elements(CSPStructMat* S, int rows, int cols,
                                double _Complex(int i, int j),
                                const CSPOptions* opts);


#if defined(STRUMPACK_USE_MPI)
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
   * \see SP_d_struct_destroy, SP_d_struct_default_options
   */
  int SP_d_struct_from_elements_mpi(CSPStructMat* S, const MPI_Comm comm,
                                    int rows, int cols,
                                    double A(int i, int j),
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
   * \see SP_c_struct_destroy, SP_c_struct_default_options
   */
  int SP_c_struct_from_elements_mpi(CSPStructMat* S, const MPI_Comm comm,
                                    int rows, int cols,
                                    float _Complex A(int i, int j),
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
   * \see SP_z_struct_destroy, SP_z_struct_default_options
   */
  int SP_z_struct_from_elements_mpi(CSPStructMat* S, const MPI_Comm comm,
                                    int rows, int cols,
                                    double _Complex A(int i, int j),
                                    const CSPOptions* opts);


  /**
   * For a 1d distributed structured matrix, return the local number
   * of rows assigned to this process.
   * \return number of local rows
   */
  int SP_s_struct_local_rows(const CSPStructMat S);
  /**
   * For a 1d distributed structured matrix, return the local number
   * of rows assigned to this process.
   * \return number of local rows
   */
  int SP_d_struct_local_rows(const CSPStructMat S);
  /**
   * For a 1d distributed structured matrix, return the local number
   * of rows assigned to this process.
   * \return number of local rows
   */
  int SP_c_struct_local_rows(const CSPStructMat S);
  /**
   * For a 1d distributed structured matrix, return the local number
   * of rows assigned to this process.
   * \return number of local rows
   */
  int SP_z_struct_local_rows(const CSPStructMat S);


  /**
   * For a 1d distributed structured matrix, return the first row
   * assigned to this process.
   * \return first row in 1d block row distribution
   */
  int SP_s_struct_begin_row(const CSPStructMat S);
  /**
   * For a 1d distributed structured matrix, return the first row
   * assigned to this process.
   * \return first row in 1d block row distribution
   */
  int SP_d_struct_begin_row(const CSPStructMat S);
  /**
   * For a 1d distributed structured matrix, return the first row
   * assigned to this process.
   * \return first row in 1d block row distribution
   */
  int SP_c_struct_begin_row(const CSPStructMat S);
  /**
   * For a 1d distributed structured matrix, return the first row
   * assigned to this process.
   * \return first row in 1d block row distribution
   */
  int SP_z_struct_begin_row(const CSPStructMat S);

#endif


  /**
   * Multiply a structured matrix with a dense matrix (or vector):
    \verbatim
         C = S*B   if trans == 'N' or 'n'
         C = S^T*B if trans == 'T' or 't'
         C = S^C*B if trans == 'C' or 'c'
    \endverbatim
   * C should have rows(S) rows if trans == 'N'/'n', else cols(S).
   * B should have cols(S) rows if trans == 'N'/'n', else rows(S).
   *
   * \param S structured matrix
   * \param trans (conjugate-)transpose of S?
   * \param m number of columns in B and C
   * \param B pointer to data of B (column major)
   * \param ldB leading dimension for B
   * \param C pointer to matrix C (column major)
   * \param ldC leading dimension for C
   */
  int SP_s_struct_mult(const CSPStructMat S, char trans, int m,
                       const float* B, int ldB,
                       float* C, int ldC);
  /**
   * Multiply a structured matrix with a dense matrix (or vector):
    \verbatim
         C = S*B   if trans == 'N' or 'n'
         C = S^T*B if trans == 'T' or 't'
         C = S^C*B if trans == 'C' or 'c'
    \endverbatim
   * C should have rows(S) rows if trans == 'N'/'n', else cols(S).
   * B should have cols(S) rows if trans == 'N'/'n', else rows(S).
   *
   * \param S structured matrix
   * \param trans (conjugate-)transpose of S?
   * \param m number of columns in B and C
   * \param B pointer to data of B (column major)
   * \param ldB leading dimension for B
   * \param C pointer to matrix C (column major)
   * \param ldC leading dimension for C
   */
  int SP_d_struct_mult(const CSPStructMat S, char trans, int m,
                       const double* B, int ldB,
                       double* C, int ldC);
  /**
   * Multiply a structured matrix with a dense matrix (or vector):
    \verbatim
         C = S*B   if trans == 'N' or 'n'
         C = S^T*B if trans == 'T' or 't'
         C = S^C*B if trans == 'C' or 'c'
    \endverbatim
   * C should have rows(S) rows if trans == 'N'/'n', else cols(S).
   * B should have cols(S) rows if trans == 'N'/'n', else rows(S).
   *
   * \param S structured matrix
   * \param trans (conjugate-)transpose of S?
   * \param m number of columns in B and C
   * \param B pointer to data of B (column major)
   * \param ldB leading dimension for B
   * \param C pointer to matrix C (column major)
   * \param ldC leading dimension for C
   */
  int SP_c_struct_mult(const CSPStructMat S, char trans, int m,
                       const float _Complex* B, int ldB,
                       float _Complex* C, int ldC);
  /**
   * Multiply a structured matrix with a dense matrix (or vector):
    \verbatim
         C = S*B   if trans == 'N' or 'n'
         C = S^T*B if trans == 'T' or 't'
         C = S^C*B if trans == 'C' or 'c'
    \endverbatim
   * C should have rows(S) rows if trans == 'N'/'n', else cols(S).
   * B should have cols(S) rows if trans == 'N'/'n', else rows(S).
   *
   * \param S structured matrix
   * \param trans (conjugate-)transpose of S?
   * \param m number of columns in B and C
   * \param B pointer to data of B (column major)
   * \param ldB leading dimension for B
   * \param C pointer to matrix C (column major)
   * \param ldC leading dimension for C
   */
  int SP_z_struct_mult(const CSPStructMat S, char trans, int m,
                       const double _Complex* B, int ldB,
                       double _Complex* C, int ldC);


  /**
   * Compute a factorization of the structured matrix. Factors are
   * stored internally. This needs to be called before calling
   * SP_s_struct_solve, and after constructing the structured matrix.
   *
   * \param S structured matrix
   *
   * \see SP_s_struct_solve
   */
  int SP_s_struct_factor(CSPStructMat S);
  /**
   * Compute a factorization of the structured matrix. Factors are
   * stored internally. This needs to be called before calling
   * SP_d_struct_solve, and after constructing the structured matrix.
   *
   * \param S structured matrix
   *
   * \see SP_d_struct_solve
   */
  int SP_d_struct_factor(CSPStructMat S);
  /**
   * Compute a factorization of the structured matrix. Factors are
   * stored internally. This needs to be called before calling
   * SP_c_struct_solve, and after constructing the structured matrix.
   *
   * \param S structured matrix
   *
   * \see SP_c_struct_solve
   */
  int SP_c_struct_factor(CSPStructMat S);
  /**
   * Compute a factorization of the structured matrix. Factors are
   * stored internally. This needs to be called before calling
   * SP_z_struct_solve, and after constructing the structured matrix.
   *
   * \param S structured matrix
   *
   * \see SP_z_struct_solve
   */
  int SP_z_struct_factor(CSPStructMat S);


  /**
   * Solve a system of linear equations with a structured matrix, with
   * possibly multiple right-hand sides (column major). The solution
   * overwrites the right-hand side. This should be called after
   * SP_s_struct_factor. This can be called multiple times.
   *
   * \param S structured matrix
   * \param nrhs number of right-hand sides, columns in B
   * \param B right-hand side, will be overwritten by the solution
   * \param ldB leading dimension of B
   *
   * \see SP_s_struct_factor
   */
  int SP_s_struct_solve(const CSPStructMat S, int nrhs,
                        float* B, int ldB);
  /**
   * Solve a system of linear equations with a structured matrix, with
   * possibly multiple right-hand sides (column major). The solution
   * overwrites the right-hand side. This should be called after
   * SP_d_struct_factor. This can be called multiple times.
   *
   * \param S structured matrix
   * \param nrhs number of right-hand sides, columns in B
   * \param B right-hand side, will be overwritten by the solution
   * \param ldB leading dimension of B
   *
   * \see SP_d_struct_factor
   */
  int SP_d_struct_solve(const CSPStructMat S, int nrhs,
                        double* B, int ldB);
  /**
   * Solve a system of linear equations with a structured matrix, with
   * possibly multiple right-hand sides (column major). The solution
   * overwrites the right-hand side. This should be called after
   * SP_c_struct_factor. This can be called multiple times.
   *
   * \param S structured matrix
   * \param nrhs number of right-hand sides, columns in B
   * \param B right-hand side, will be overwritten by the solution
   * \param ldB leading dimension of B
   *
   * \see SP_c_struct_factor
   */
  int SP_c_struct_solve(const CSPStructMat S, int nrhs,
                        float _Complex* B, int ldB);
  /**
   * Solve a system of linear equations with a structured matrix, with
   * possibly multiple right-hand sides (column major). The solution
   * overwrites the right-hand side. This should be called after
   * SP_z_struct_factor. This can be called multiple times.
   *
   * \param S structured matrix
   * \param nrhs number of right-hand sides, columns in B
   * \param B right-hand side, will be overwritten by the solution
   * \param ldB leading dimension of B
   *
   * \see SP_z_struct_factor
   */
  int SP_z_struct_solve(const CSPStructMat S, int nrhs,
                        double _Complex* B, int ldB);


  /**
   * Apply a shift to the diagonal of this matrix. Ie, S +=
   * s*I, with I the identity matrix. If this is called after
   * calling factor, then the factors are not updated. To solve a
   * linear system with the shifted matrix, you need to call
   * factor again.
   *
   * \param S structured matrix
   * \param s Shift to be applied to the diagonal.
   */
  int SP_s_struct_shift(CSPStructMat S, float s);
  /**
   * Apply a shift to the diagonal of this matrix. Ie, S +=
   * s*I, with I the identity matrix. If this is called after
   * calling factor, then the factors are not updated. To solve a
   * linear system with the shifted matrix, you need to call
   * factor again.
   *
   * \param S structured matrix
   * \param s Shift to be applied to the diagonal.
   */
  int SP_d_struct_shift(CSPStructMat S, double s);
  /**
   * Apply a shift to the diagonal of this matrix. Ie, S +=
   * s*I, with I the identity matrix. If this is called after
   * calling factor, then the factors are not updated. To solve a
   * linear system with the shifted matrix, you need to call
   * factor again.
   *
   * \param S structured matrix
   * \param s Shift to be applied to the diagonal.
   */
  int SP_c_struct_shift(CSPStructMat S, float _Complex s);
  /**
   * Apply a shift to the diagonal of this matrix. Ie, S +=
   * s*I, with I the identity matrix. If this is called after
   * calling factor, then the factors are not updated. To solve a
   * linear system with the shifted matrix, you need to call
   * factor again.
   *
   * \param S structured matrix
   * \param s Shift to be applied to the diagonal.
   */
  int SP_z_struct_shift(CSPStructMat S, double _Complex s);

#ifdef __cplusplus
}
#endif

#endif
