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
/*! \file StructuredMatrix.hpp
 * \brief Contains the structured matrix interfaces.
 */
#ifndef STRUCTURED_MATRIX_HPP
#define STRUCTURED_MATRIX_HPP

#include <vector>
#include <cassert>
#include <memory>
#include <functional>
#include <algorithm>

#include "StructuredOptions.hpp"
#include "ClusterTree.hpp"
#include "dense/DenseMatrix.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "dense/DistributedMatrix.hpp"
#endif

namespace strumpack {

  /**
   * Namespace containing the StructuredMatrix class, a general
   * abstract interface to several other rank-structured matrix
   * representations.
   */
  namespace structured {

    /**
     * Type for element extraction routine. This can be implemented
     * as a lambda function, a functor or a funtion pointer.
     *
     * \param i vector of row indices
     * \param j vector of column indices
     * \return element i,j of the matrix
     */
    template<typename scalar_t>
    using extract_t = std::function
      <scalar_t(std::size_t i, std::size_t j)>;

    /**
     * Type for block element extraction routine. This can be
     * implemented as a lambda function, a functor or a funtion
     * pointer.
     *
     * \param I vector of row indices
     * \param J vector of column indices
     * \param B submatrix (I, J) to be computed by user code
     */
    template<typename scalar_t>
    using extract_block_t = std::function
      <void(const std::vector<std::size_t>& I,
            const std::vector<std::size_t>& J,
            DenseMatrix<scalar_t>& B)>;

    /**
     * Type for block element extraction routine. This can be
     * implemented as a lambda function, a functor or a funtion
     * pointer. The sub-matrix should be put into a 2d block cyclicly
     * distributed matrix.
     *
     * \param I vector of row indices
     * \param J vector of column indices
     * \param B 2d block cyclic matrix, submatrix (I, J) to be
     * computed by user code
     */
    template<typename scalar_t>
    using extract_dist_block_t = std::function
      <void(const std::vector<std::size_t>& I,
            const std::vector<std::size_t>& J,
            DistributedMatrix<scalar_t>& B)>;

    /**
     * Type for matrix multiplication routine. This can be
     * implemented as a lambda function, a functor or a funtion
     * pointer.
     *
     * \param op whether to compute the transposed/conjugate product
     * \param R random matrix passed to user code
     * \param S sample matrix to be computed by user code: \n
     *            S = A*R if op is Trans::N \n
     *            S = A*T*R if op is Trans::T \n
     *            S = A^C*R if op is Trans::C
     */
    template<typename scalar_t>
    using mult_t = std::function
      <void(Trans op,
            const DenseMatrix<scalar_t>& R,
            DenseMatrix<scalar_t>& S)>;

    /**
     * Type to specify admissibility of individual sub-blocks.
     */
    using admissibility_t = DenseMatrix<bool>;



    /**
     * \class StructuredMatrix
     *
     * \brief Class to represent a structured matrix. This is the
     * abstract base class for several types of structured matrices.
     *
     * \tparam scalar_t Can be float, double, std:complex<float> or
     * std::complex<double>. Some formats do not support all
     * precisions, see strumpack::StructuredMatrix::Type.
     *
     * For construction, use one of:
     * - structured::construct_from_dense (DENSE)
     * - structured::construct_from_elements (ELEM)
     * - structured::construct_matrix_free (MF)
     * - structured::construct_partially_matrix_free (PMF)
     * - nearest neighbors (TODO)
     *
     * However, not all formats support all construction methods:
     *
     * |          | DENSE | ELEM | MF | PMF | NN |
     * |----------|-------|------|----|-----|----|
     * | HSS      | X     |  -   | -  | X   | X  |
     * | BLR      | X     |  X   | -  | -   | -  |
     * | HODLR    | X     |  X   | X  | -   | X  |
     * | HODBF    | X     |  X   | X  | -   | X  |
     * | LRBF     | X     |  X   | X  | -   | X  |
     * | LOSSY    | X     |  -   | -  | -   | -  |
     * | LOSSLESS | X     |  -   | -  | -   | -  |
     *
     * \see HSS::HSSMatrix, BLR::BLRMatrix, HODLR::HODLRMatrix,
     * HODLR::ButterflyMatrix, ...
     */
    template<typename scalar_t> class StructuredMatrix {
      using real_t = typename RealType<scalar_t>::value_type;

    public:

      /**
       * Virtual destructor.
       */
      virtual ~StructuredMatrix() = default;

      /**
       * Get number of rows in this matrix
       */
      virtual std::size_t rows() const = 0;

      /**
       * Get number of columns in this matrix
       */
      virtual std::size_t cols() const = 0;

      /**
       * Return the total amount of memory used by this matrix, in
       * bytes.
       *
       * \return Memory usage in bytes.
       * \see nonzeros
       */
      virtual std::size_t memory() const = 0;

      /**
       * Return the total number of nonzeros stored by this matrix.
       *
       * \return Nonzeros in the matrix representation.
       * \see memory
       */
      virtual std::size_t nonzeros() const = 0;

      /**
       * Return the maximum rank of this matrix over all low-rank
       * compressed blocks.
       *
       * \return Maximum rank.
       */
      virtual std::size_t rank() const = 0;

      /**
       * Multiply the StructuredMatrix (A) with a dense matrix: y =
       * op(A)*x.
       *
       * \param op take transpose/conjugate or not
       * \param x matrix, x.rows() == op(A).cols()
       * \param y matrix, y.cols() == x.cols(), y.rows() == A.rows()
       */
      virtual void mult(Trans op, const DenseMatrix<scalar_t>& x,
                        DenseMatrix<scalar_t>& y) const;

      /**
       * Compute a factorization (or the inverse) of this matrix, to
       * be used later for solving linear systems. The actual type of
       * factorization depends on the StructuredMatrix::Type of this
       * matrix.
       **/
      virtual void factorization() {} //= 0;

      /**
       * Solve a linear system A*x=b, with this StructuredMatrix (A).
       * If not already done by the user, this will first call
       * this->factor().
       *
       * \param b right-hand side, b.rows() == A.cols()
       * \param x solution, should be allocated by user, x.cols() ==
       * b.cols() and b.cols() == A.rows()
       */
      // virtual void solve(const DenseMatrix<scalar_t>& b,
      //                    DenseMatrix<scalar_t>& x) const = 0;

      /**
       * Solve a linear system A*x=b, with this StructuredMatrix
       * (A). This solve is done in-place.
       *
       * \param b right-hand side, b.rows() == A.cols(), will be
       * overwritten with the solution x.
       */
      virtual void solve(DenseMatrix<scalar_t>& b) const {}; //= 0;

    };


    /**
     * Construct a StructuredMatrix from a DenseMatrix<scalar_t>.
     * This construction, taking a dense matrix should be valid for
     * all StructuredMatrix types (see
     * StructuredMatrix::Type). However, faster and more memory
     * efficient methods of constructing StructuredMatrix
     * representations are available, see
     * StructuredMatrix::construct_from_elements,
     * StructuredMatrix::construct_matrix_free and
     * StructuredMatrix::construct_partially_matrix_free.
     *
     * \tparam scalar_t precision of input matrix, and of
     * constructed StructuredMatrix. Note that not all types support
     * every all precisions. See StructuredMatrix::Type.
     *
     * \param A Input dense matrix, will not be modified
     * \param opts Options object
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \throw std::invalid_argument Operation not implemented,
     * operation not supported for StructuredMatrix type, operation
     * requires MPI.
     *
     * \see construct_from_elements, construct_matrix_free and
     * construct_partially_matrix_free
     */
    template<typename scalar_t>
    std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_dense(const DenseMatrix<scalar_t>& A,
                         const StructuredOptions<scalar_t>& opts);

    /**
     * Construct a StructuredMatrix from a DenseMatrix<scalar_t>.
     * This construction, taking a dense matrix should be valid for
     * all StructuredMatrix types (see
     * StructuredMatrix::Type). However, faster and more memory
     * efficient methods of constructing StructuredMatrix
     * representations are available, see
     * StructuredMatrix::construct_from_elements,
     * StructuredMatrix::construct_matrix_free and
     * StructuredMatrix::construct_partially_matrix_free.
     *
     * \tparam scalar_t precision of input matrix, and of
     * constructed StructuredMatrix. Note that not all types support
     * every all precisions. See StructuredMatrix::Type.
     *
     * \param rows number of rows in matrix to be constructed
     * \param cols number of columnss in matrix to be constructed
     * \param A pointer to matrix A data
     * \param ldA leading dimension of A
     * \param opts Options object
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \see StructuredMatrix::construct_from_elements,
     * StructuredMatrix::construct_matrix_free and
     * StructuredMatrix::construct_partially_matrix_free
     */
    template<typename scalar_t>
    std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_dense(int rows, int cols, const scalar_t* A, int ldA,
                         const StructuredOptions<scalar_t>& opts);

    /**
     * Construct a StructuredMatrix using a routine, provided by the
     * user, to extract elements from the matrix.
     *
     * \tparam scalar_t precision of input matrix, and of
     * constructed StructuredMatrix. Note that not all types support
     * every all precisions. See StructuredMatrix::Type.
     *
     * \param rows number of rows in matrix to be constructed
     * \param cols number of columnss in matrix to be constructed
     * \param A element extraction routine
     * \param opts Options object
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \see StructuredMatrix::construct_from_dense,
     * StructuredMatrix::construct_matrix_free and
     * StructuredMatrix::construct_partially_matrix_free
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(int rows, int cols,
                            const extract_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts);

    /**
     * Construct a StructuredMatrix using a routine to extract a
     * sub-block from the matrix.
     *
     * TODO describe parameters
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(int rows, int cols,
                            const extract_block_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts);

    /**
     * Construct a StructuredMatrix using only a matrix-vector
     * multiplication routine.
     *
     * TODO desribe parameters
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_matrix_free(int rows, int cols,
                          const mult_t<scalar_t>& Amult,
                          const StructuredOptions<scalar_t>& opts);

    /**
     * Construct a StructuredMatrix using both a matrix-vector
     * multiplication routine and a routine to extract a matrix
     * sub-block.
     *
     * TODO desribe parameters
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<scalar_t>& Amult,
                                    const extract_block_t<scalar_t>& Aelem,
                                    const StructuredOptions<scalar_t>& opts);

    /**
     * Construct a StructuredMatrix using both a matrix-vector
     * multiplication routine and an element extraction routine.
     *
     * TODO desribe parameters
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<scalar_t>& Amult,
                                    const extract_t<scalar_t>& Aelem,
                                    const StructuredOptions<scalar_t>& opts);

#if defined(STRUMPACK_USE_MPI)
    /**
     * Construct a StructuredMatrix from a 2D block-cyclicly
     * distributed matrix (ScaLAPACK layout).
     *
     * TODO desribe parameters
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_dense(const DistributedMatrix<scalar_t>& A,
                         const StructuredOptions<scalar_t>& opts);

    /**
     * Construct a StructuredMatrix using an element extraction
     * routine.
     *
     * TODO desribe parameters
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(const MPIComm& comm,
                            int rows, int cols,
                            const extract_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts);

    /**
     * Construct a StructuredMatrix using a routine to extract a
     * matrix sub-block.
     *
     * TODO desribe parameters
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(const MPIComm& comm,
                            int rows, int cols,
                            const extract_dist_block_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts);
#endif

  } // end namespace structured
} // end namespace strumpack

#endif // STRUCTURED_MATRIX_HPP
