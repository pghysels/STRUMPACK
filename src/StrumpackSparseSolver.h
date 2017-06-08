/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The Regents of
 * the University of California, through Lawrence Berkeley National Laboratory
 * (subject to receipt of any required approvals from the U.S. Dept. of Energy).
 * All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Technology Transfer Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, prepare derivative works, and perform publicly and display publicly.
 * Beginning five (5) years after the date permission to assert copyright is
 * obtained from the U.S. Department of Energy, and subject to any subsequent five
 * (5) year renewals, the U.S. Government is granted for itself and others acting
 * on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, prepare derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 *
 */
/*! \file StrumpackSparseSolver.h
 * \brief C interface to StrumpackSparseSolver.
 */
#ifndef STRUMPACK_SPARSE_SOLVER_H
#define STRUMPACK_SPARSE_SOLVER_H

#include <stdint.h>
#include "mpi.h"

/*! \brief The precision to be used by the STRUMPACK_SparseSolver. */
typedef enum {
  STRUMPACK_FLOAT,             /*!< Single precision, real, 32bit integers.    */
  STRUMPACK_DOUBLE,            /*!< Double precision, real, 32bit integers.    */
  STRUMPACK_FLOATCOMPLEX,      /*!< Single precision, complex, 32bit integers. */
  STRUMPACK_DOUBLECOMPLEX,     /*!< Double precision, complex, 32bit integers. */
  STRUMPACK_FLOAT_64,          /*!< Single precision, real, 64bit integers.    */
  STRUMPACK_DOUBLE_64,         /*!< Double precision, real, 64bit integers.    */
  STRUMPACK_FLOATCOMPLEX_64,   /*!< Single precision, complex, 64bit integers. */
  STRUMPACK_DOUBLECOMPLEX_64   /*!< Double precision, complex, 64bit integers. */
} STRUMPACK_PRECISION;

/*! \brief The precision to be used by the STRUMPACK_SparseSolver. */
typedef enum {
  STRUMPACK_MT,          /*!< The sequential/multithreaded solver interface. */
  STRUMPACK_MPI,         /*!< The distributed solver, with replicated input. */
  STRUMPACK_MPI_DIST     /*!< The fully distributed solver.                  */
} STRUMPACK_INTERFACE;

/*! \brief The main solver object through which one interacts with
 *         STRUMPACK-sparse (using the C interface).
 *
 * Only interact with this struct through the provided functions STRUMPACK_... .
 * \sa STRUMPACK_init, STRUMPACK_destroy, STRUMPACK_set_csr_matrix, ...
 */
typedef struct {
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  void* solver;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
  STRUMPACK_PRECISION precision;  /*!< Information about the scalar and integer type used by the solver. */
  STRUMPACK_INTERFACE interface;  /*!< Which interface the solver uses. See the manual for a description of the different interfaces. */
} STRUMPACK_SparseSolver;

/*! \brief Definition of a single precision complex number.
 *
 * This is compatible with a c++ std::complex<float>, or a fortran
 * single precision complex number.
 */
typedef struct {
  float r;  /*!< Real part.      */
  float i;  /*!< Imaginary part. */
} floatcomplex;

/*! \brief Definition of a double precision complex number.
 *
 * This is compatible with a c++ std::complex<double>, or a fortran
 * double precision complex number.
 */
typedef struct {
  double r;  /*!< Real part.      */
  double i;  /*!< Imaginary part. */
} doublecomplex;

/*! \brief Nested dissection code. */
typedef enum {
  STRUMPACK_METIS,     /*!< Use either Metis (StrumpackSparseSolver, StrumpackSparseSolverMPI) or ParMetis (StrumpackSparseSolverMPI)    */
  STRUMPACK_PARMETIS,
  STRUMPACK_SCOTCH,    /*!< Use either Scotch (StrumpackSparseSolver, StrumpackSparseSolverMPI) or PT-Scotch (StrumpackSparseSolverMPI)  */
  STRUMPACK_PTSCOTCH,
  RCM,
  STRUMPACK_GEOMETRIC  /*!< A simple geometric nested dissection code that only works for regular meshes. (see Sp::reorder)              */
} STRUMPACK_REORDERING_STRATEGY;

/*! \brief Type of Gram-Schmidt orthogonalization used in GMRes. */
typedef enum {
  STRUMPACK_CLASSICAL,  /*!< Classical Gram-Schmidt is faster, more scalable.   */
  STRUMPACK_MODIFIED    /*!< Modified Gram-Schmidt is slower, but stable.       */
} STRUMPACK_GRAM_SCHMIDT_TYPE;

/*! \brief The random number distribution. */
typedef enum {
  STRUMPACK_NORMAL,   /*!< Normal(0,1) distributed numbers (takes roughly 23 flops per random number).  */
  STRUMPACK_UNIFORM   /*!< Uniform [0,1] distributed numbers (takes about 7 flops per random number).   */
} STRUMPACK_RANDOM_DISTRIBUTION;

/*! \brief Random number engine. */
typedef enum {
  STRUMPACK_LINEAR,    /*!< The C++11 std::minstd_rand random number generator. */
  STRUMPACK_MERSENNE   /*!< The C++11 std::mt19937 random number generator.     */
} STRUMPACK_RANDOM_ENGINE;

/*! \brief Type of outer iterative (Krylov) solver. */
typedef enum {
  STRUMPACK_AUTO,           /*!< Use iterative refinement if no HSS compression is used, otherwise use GMRes.      */
  STRUMPACK_DIRECT,         /*!< No outer iterative solver, just a single application of the multifrontal solver.  */
  STRUMPACK_REFINE,         /*!< Iterative refinement.                                                             */
  STRUMPACK_PREC_GMRES,     /*!< Preconditioned GMRes. The preconditioner is the (approx) multifrontal solver.     */
  STRUMPACK_GMRES,          /*!< UN-preconditioned GMRes. (for testing mainly)                                     */
  STRUMPACK_PREC_BICGSTAB,  /*!< Preconditioned BiCGStab. The preconditioner is the (approx) multifrontal solver.  */
  STRUMPACK_BICGSTAB        /*!< UN-preconditioned BiCGStab. (for testing mainly)                                  */
 } STRUMPACK_KRYLOV_SOLVER;

/*! \brief The possible return codes. */
typedef enum {
  STRUMPACK_SUCCESS=0,           /*!< Operation completed successfully. */
  STRUMPACK_MATRIX_NOT_SET,      /*!< The input matrix was not set.     */
  STRUMPACK_REORDERING_ERROR     /*!< The matrix reordering failed.     */
} STRUMPACK_RETURN_CODE;

/*! \brief Rank patterns, see manual section 5. */
typedef enum {
  STRUMPACK_ADAPTIVE=0,           /*!< Adaptive rank determination (not supported when using MPI). */
  STRUMPACK_CONSTANT,             /*!< Constant number of random vectors.                          */
  STRUMPACK_SQRTN,                /*!< d = alpha * sqrt(N) + beta, N is the separator size.        */
  STRUMPACK_SQRTNLOGN,            /*!< d = alpha * sqrt(N) * log(sqrt(N)) + beta.                  */
  STRUMPACK_BISECTIONCUT          /*!< Not implemented yet.                                        */
} STRUMPACK_RANK_PATTERN;


#ifdef __cplusplus
extern "C" {
#endif

  /*! \brief Initialize a STRUMPACK_SparseSolver object.
   *
   * \param S         Solver object to initialize.
   * \param comm      MPI communicator. The communicator is duplicated internally.
   *                  This will be ignored when using the STRUMPACK_MT interface.
   * \param precision The required precision, and integer type.
   * \param interface The interface (sequential/multithreaded, MPI, fully distributed).
   * \param argc      Number of arguments, in argv.
   * \param argv      Command line arguments.
   * \param verbose   Enable/suppress output printing (by the root only).
   * \sa              STRUMPACK_destroy, STRUMPACK_set_from_options, ...
   */
  void STRUMPACK_init(STRUMPACK_SparseSolver* S, MPI_Comm comm, STRUMPACK_PRECISION precision, STRUMPACK_INTERFACE interface, int argc, char* argv[], int verbose);

  /*! \brief Destroy a solver object. Will deallocate all internally
   *         allocated memory.
   * \param S   The solver object, should be initialized.
   */
  void STRUMPACK_destroy(STRUMPACK_SparseSolver* S);

  /*! \brief Associate a (sequential) CSR matrix with the Strumpack
   *         sparse solver.
   *
   * Only use this after initializing the STRUMPACK_SparseSolver.
   * This is a general 'polymorphic' interface, which comes without
   * type checking.  Use this with {N,row_ptr,col_ind} int* or
   * int64_t* depending on how you initialized S (S.precision).
   * Likewise for values, it should be float*, double*, floatcomplex*
   * or doublecomplex*.
   *
   * \param S        The solver object, should be initialized.
   * \param N        Dimension of the CRS input matrix.
   * \param row_ptr  Indices in col_ind and values for the start of each row.
   *                 Nonzeros for row r are in [row_ptr[r],row_ptr[r+1]).
   * \param col_ind  Column indices of each nonzero.
   * \param values   Nonzero values.
   * \param symmetric_pattern Denotes whether the sparsity pattern of
   *                          the input matrix is symmetric.
   *                          Use 0 for non-symmetric.
   * \sa STRUMPACK_init, STRUMPACK_destroy, STRUMPACK_factor,
   *     STRUMPACK_solve
   */
  void STRUMPACK_set_csr_matrix(STRUMPACK_SparseSolver S, void* N, void* row_ptr, void* col_ind, void* values, int symmetric_pattern);
  /*! \brief Associate a block-row distributed CSR matrix with the solver object.
   *
   * This should only be used with a STRUMPACK_SparseSolver with a
   * STRUMPACK_MPI_DIST. This is a general 'polymorphic' interface,
   * which comes without type checking.  Use this with
   * {N,row_ptr,col_ind,dist} int* or int64_t* depending on how you
   * initialized S (S.precision).  Likewise, values should be float*,
   * double*, floatcomplex* or doublecomplex*.
   *
   * \param S          The solver object, should be initialized.
   * \param local_rows The number of rows of the input matrix assigned to this MPI process.
   *                   This should equal dist[rank+1]-dist[rank].
   * \param row_ptr    Indices in col_ind and values for the start of each row.
   *                   Nonzeros for row r+dist[rank] are in [row_ptr[r],row_ptr[r+1]).
   * \param col_ind    Column indices of each nonzero.
   * \param values     Nonzero values. Should have at least (row_ptr[dist[p+1]-dist[p]]-row_ptr[0]) elements.
   * \param dist       Specifies the block-row distribution.
   *                   A process with rank p owns rows [dist[p],dist[p+1]).
   * \param symmetric_pattern Denotes whether the sparsity pattern of the input matrix is symmetric.
   * \sa STRUMPACK_init, STRUMPACK_destroy, STRUMPACK_factor,
   *     STRUMPACK_solve
   */
  void STRUMPACK_set_distributed_csr_matrix(STRUMPACK_SparseSolver S, void* local_rows, void* row_ptr, void* col_ind, void* values, void* dist, int symmetric_pattern);
  /*! \brief Associate a (PETSc) MPIAIJ matrix with the solver object.
   *
   * This should only be used with a STRUMPACK_SparseSolver with a
   * STRUMPACK_MPI_DIST. This is a general 'polymorphic' interface,
   * which comes without type checking.  Use this with
   * {n,d_ptr,d_ind,o_ptr,o_ind,garray} int* or int64_t* depending on
   * how you initialized S (S.precision).  Likewise, d_val and o_val
   * should be float*, double*, floatcomplex* or doublecomplex*.
   *
   * \param S      The solver object, should be initialized.
   * \param n      The number of rows of the input matrix assigned to this MPI process.
   * \param d_ptr  Indices in d_ind and values for the start of each row in the diagonal block.
   * \param d_ind  Column indices of each nonzero in the diagonal block.
   * \param d_val  Nonzero values in the diagonal block.
   * \param o_ptr  Indices in d_ind and values for the start of each row in the off-diagonal block.
   * \param o_ind  Column indices of each nonzero in the off-diagonal block.
   * \param o_val  Nonzero values in the off-diagonal block.
   * \param garray Converts column indices in o_ind to global column indices.
   * \sa STRUMPACK_init, STRUMPACK_set_distributed_csr_matrix
   */
  void STRUMPACK_set_MPIAIJ_matrix(STRUMPACK_SparseSolver S, void* n, void* d_ptr, void* d_ind, void* d_val,
				   void* o_ptr, void* o_ind, void* o_val, void* garray);
  /*! \brief Solve a linear system with the solver object.
   *
   * Only use this after initializing the STRUMPACK_SparseSolver and
   * assigning a matrix to it.  This is a general 'polymorphic'
   * interface, which comes without type checking.  Use this with b
   * and x pointer to float*, double*, floatcomplex* or doublecomplex*
   * depending on how you initialized S.
   *
   * \param S   The solver object, should be initialized.
   * \param b   Input, will not be modified. Pointer to the right-hand side.
   *            Array should be lenght N, the dimension of the input matrix when
   *            using the STRUMPACK_MT or STRUMPACK_MPI interfaces. For the
   *            STRUMPACK_MPI_DIST interface, the length of b should be correspond
   *            the partitioning of the block-row distributed input matrix.
   * \param x   Output. Pointer to the solution vector. Array should be lenght N,
   *            the dimension of the input matrix  when using the STRUMPACK_MT
   *            or STRUMPACK_MPI interfaces. For the STRUMPACK_MPI_DIST interface,
   *            the length of b should be correspond the partitioning of the
   *            block-row distributed input matrix.
   * \param use_initial_guess Set to true if x contains an intial guess to the solution.
   *                          This is mainly useful when using an iterative solver.
   *                          If set to false, x should not be set (but should be allocated).
   * \return    Error code.
   * \sa STRUMPACK_init, STRUMPACK_set_csr_matrix, STRUMPACK_factor,
   *     STRUMPACK_solve
   */
  STRUMPACK_RETURN_CODE STRUMPACK_solve(STRUMPACK_SparseSolver S, void* b, void* x, int use_initial_guess);
  /*! \brief Parse the command line options passed to the
   *         constructor.
   *
   * The options are only parsed when this function is called, not
   * during construction of the class. This way, one can set options
   * on the object after construction, but before parsing the
   * command line options.
   *
   * \param S   The solver object, should be initialized.
   */
  void STRUMPACK_set_from_options(STRUMPACK_SparseSolver S);
  /*! \brief Compute matrix reorderings for numerical stability and
   *         to reduce fill-in.
   *
   * Start computation of the matrix reorderings. See the relevant
   * options to control the matrix reordering. A first reordering is
   * the MC64 column permutation for numerical stability. This can
   * be disabled if the matrix has large nonzero diagonal
   * entries. MC64 optionally also performs row and column
   * scaling. Next, a fill-reducing reordering is computed. This is
   * done with the nested dissection algortihms of either
   * (PT-)Scotch, (Par)Metis or a simple geometric nested dissection
   * code which only works on regular meshes. For
   * StrumpackSparseSolver and StrumpackSparseSolverMPI, setting the
   * reordering method to STRUMPACK_METIS or
   * STRUMPACK_SCOTCH will result in Metis or Scotch nested
   * dissection respectively, whereas StrumpackSparseSolverMPIDist
   * will use ParMetis or PT-Scotch.
   *
   * \param S The solver object, should be initialized.
   * \return  Error code.
   * \sa      STRUMPACK_set_mc64job, STRUMPACK_set_reorder_method, STRUMPACK_set_nd_param
   */
  STRUMPACK_RETURN_CODE STRUMPACK_reorder(STRUMPACK_SparseSolver S);
  /*! \brief Compute matrix reorderings for numerical stability and
   *         to reduce fill-in.
   *
   * See also STRUMPACK_reorder. This reordering function only works
   * for regular meshed with a simple stencil (3 points wide per
   * dimension), and only a single DoF per gridpoint.
   *
   * \param S   The solver object, should be initialized.
   * \param nx  This (optional) parameter is only meaningful when the matrix corresponds
   *            to a stencil on a regular mesh. The stecil is assumed to be at most 3 points
   *            wide in each dimension and only contain a single degree of freedom per grid point.
   *            The nx parameter denotes the number of grid points in the first spatial dimension.
   * \param ny  See parameters nx. Parameter ny denotes the number of gridpoints in the second
   *            spatial dimension. This should only be set if the mesh is 2 or 3 dimensional.
   * \param nz  See parameters nx. Parameter nz denotes the number of gridpoints in the third
   *            spatial dimension. This should only be set if the mesh is 3 dimensional.
   */
  STRUMPACK_RETURN_CODE STRUMPACK_reorder_regular(STRUMPACK_SparseSolver S, int nx, int ny, int nz);
  /*! \brief Perform numerical factorization of the sparse input matrix.
   *
   * \param S   The solver object, should be initialized.
   * \return    Error code.
   */
  STRUMPACK_RETURN_CODE STRUMPACK_factor(STRUMPACK_SparseSolver S);
  /*! \brief Set the maximum number of Krylov iterations.
   * \param S     The solver object, should be initialized.
   * \param maxit The maximum number of iterations, should be >= 1.
   */

  void STRUMPACK_set_maxit(STRUMPACK_SparseSolver S, int maxit);
  /*! \brief Set the GMRes restart length. A larger value most likely leads
   *         to faster convergence, but requires more memory.
   * \param S   The solver object, should be initialized.
   * \param m   The new GMRes restart length, should be >= 1.
   */
  void STRUMPACK_set_gmres_restart(STRUMPACK_SparseSolver S, int m);

  void STRUMPACK_set_rel_tol(STRUMPACK_SparseSolver S, double tol);

  void STRUMPACK_set_abs_tol(STRUMPACK_SparseSolver S, double tol);

  /*! \brief Set parameter controlling the leaf size in the nested dissection procedure.
   * \param S   The solver object, should be initialized.
   * \param nd_param  Stop the nested dissection recursion when a separator is smaller then nd_param.
   */
  void STRUMPACK_set_nd_param(STRUMPACK_SparseSolver S, int nd_param);
  /*! \brief Set the fill-reducing matrix reordering strategy
   * \param S   The solver object, should be initialized.
   * \param m   One of STRUMPACK_REORDERING_STRATEGY. \n
   *            - STRUMPACK_METIS: Use Metis or ParMetis (for STRUMPACK_MPI_DIST) \n
   *            - STRUMPACK_SCOTCH: Use Scotch or PT-Scotch (for STRUMPACK_MPI_DIST) \n
   *            - STRUMPACK_GEOMETRIC: Use a simple geometric nested dissection code for regular meshes.
   *              This only works when this->reorder(nx,ny,nz) was called with the correct mesh sizes.
   */
  void STRUMPACK_set_reordering_method(STRUMPACK_SparseSolver S, STRUMPACK_REORDERING_STRATEGY m);
  /*! \brief Set the type of Gram Schmidth orthogonalization used on GMRes.
   * \param S   The solver object, should be initialized.
   * \param t STRUMPACK_CLASSICAL is faster (more scalable) but STRUMPACK_MODIFIED is stable.
   */
  void STRUMPACK_set_GramSchmidt_type(STRUMPACK_SparseSolver S, STRUMPACK_GRAM_SCHMIDT_TYPE t);
  /*! \brief Set the MC64 job type.
   *
   * MC64 is used to permute the input matrix for numerical
   * stability. It can also apply row and column scaling for
   * stability.
   *
   * \param S   The solver object, should be initialized.
   * \param job The job type. \n
   *        - job=0: Disable MC64, for many matrices MC64 is not required. \n
   *        - job=1: This is not supported. \n
   *        - job=2: Maximize the smallest diagonal value. \n
   *        - job=3: Same as 2, but using a different algorithm. \n
   *        - job=4: Maximize the sum of the diagonal entries. \n
   *        - job=5: Maximize the product of the diagonal entries and perform row and column scaling.
   */
  void STRUMPACK_set_mc64job(STRUMPACK_SparseSolver S, int job);
  /*! \brief Set the outer iterative (Krylov) solver to be used.
   * \param S   The solver object, should be initialized.
   * \param solver_type No outer solver, iterative refinement, choice of Krylov solver, or auto.
   *                    Auto will select iterative refinement when the solver is used as a
   *                    direct solver (ie. no HSS approximations), and will use GMRes(m) when
   *                    the multifrontal solver is used as a preconditioner, with HSS compression.
   *                    To just do a single application of the multifrontal solver/preconditioner set
   *                    solver_type=STRUMPACK_DIRECT.
   */
  void STRUMPACK_set_Krylov_solver(STRUMPACK_SparseSolver S, STRUMPACK_KRYLOV_SOLVER solver_type);

  void STRUMPACK_enable_HSS(STRUMPACK_SparseSolver S);
  void STRUMPACK_disable_HSS(STRUMPACK_SparseSolver S);
  void STRUMPACK_set_HSS_min_front_size(STRUMPACK_SparseSolver S, int size);
  void STRUMPACK_set_HSS_min_sep_size(STRUMPACK_SparseSolver S, int size);

  /*! \brief Set the relative compression tolerance used in rank-revealing QR factorization during HSS construction. */
  void STRUMPACK_set_hss_rel_tol(STRUMPACK_SparseSolver S, double rctol);
  /*! \brief Set the absolute compression tolerance used in rank-revealing QR factorization during HSS construction. */
  void STRUMPACK_set_hss_abs_tol(STRUMPACK_SparseSolver S, double actol);
  /*! \brief Disable/enable output of statistics. */
  void STRUMPACK_set_verbose(STRUMPACK_SparseSolver S, int v);

  /*! \brief Get the maximum number of Krylov iterations. */
  int STRUMPACK_maxit(STRUMPACK_SparseSolver S);
  /*! \brief Get the GMRes restart parameter. */
  int STRUMPACK_get_gmres_restart(STRUMPACK_SparseSolver S);
  double STRUMPACK_rel_tol(STRUMPACK_SparseSolver S);
  double STRUMPACK_abs_tol(STRUMPACK_SparseSolver S);

  /*! \brief Get the nested dissection parameter being used. (maximum size of nested dissection leaf separator), */
  int STRUMPACK_nd_param(STRUMPACK_SparseSolver S);
  /*! \brief Get the matrix reordering (nested dissection) method being used. */
  STRUMPACK_REORDERING_STRATEGY STRUMPACK_reordering_method(STRUMPACK_SparseSolver S);
  /*! \brief Get the Gram-Schmidt type being used in GMRes. */
  STRUMPACK_GRAM_SCHMIDT_TYPE STRUMPACK_GramSchmidt_type(STRUMPACK_SparseSolver S);
  /*! \brief Get the maximum rank encountered in any of the HSS matrices.
   * Call this AFTER numerical factorization. */
  int STRUMPACK_max_rank(STRUMPACK_SparseSolver S);
  /*! \brief Get the number of nonzeros in the (sparse) factors. This is the fill-in.
   * Call this AFTER numerical factorization. */
  long long STRUMPACK_factor_nonzeros(STRUMPACK_SparseSolver S);
  /*! \brief Get the number of nonzeros in the (sparse) factors. This is the fill-in.
   * Call this AFTER numerical factorization. */
  long long STRUMPACK_factor_memory(STRUMPACK_SparseSolver S);
  /*! \brief Get the number of iterations performed by the outer (Krylov) iterative solver. */
  int STRUMPACK_its(STRUMPACK_SparseSolver S);
  /*! \brief Get the job number being used for MC64. */
  int STRUMPACK_mc64job(STRUMPACK_SparseSolver S);
  /*! \brief Get the relative compression tolerance used in rank-revealing QR during HSS construction. */
  double STRUMPACK_hss_rel_tol(STRUMPACK_SparseSolver S);
  /*! \brief Get the absolute compression tolerance used in rank-revealing QR during HSS construction. */
  double STRUMPACK_hss_abs_tol(STRUMPACK_SparseSolver S);
  /*! \brief Get the type of outer (Krylov) iterative solver being used. */
  STRUMPACK_KRYLOV_SOLVER STRUMPACK_Krylov_solver(STRUMPACK_SparseSolver S);

  int STRUMPACK_HSS_min_front_size(STRUMPACK_SparseSolver S);
  int STRUMPACK_HSS_min_sep_size(STRUMPACK_SparseSolver S);

#ifdef __cplusplus
}
#endif

#endif
