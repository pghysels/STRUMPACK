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

#if defined(STRUMPACK_USE_MPI)
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
     * Type for matrix multiplication routine with 2D block cyclic
     * distribution. This can be implemented as a lambda function, a
     * functor or a funtion pointer.
     *
     * \param op whether to compute the transposed/conjugate product
     * \param R random matrix passed to user code
     * \param S sample matrix to be computed by user code: \n
     *            S = A*R if op is Trans::N \n
     *            S = A*T*R if op is Trans::T \n
     *            S = A^C*R if op is Trans::C
     */
    template<typename scalar_t>
    using mult_2d_t = std::function
      <void(Trans op,
            const DistributedMatrix<scalar_t>& R,
            DistributedMatrix<scalar_t>& S)>;

    /**
     * Type for matrix multiplication routine with 1D block row
     * distribution.  The block row/column distribution of the matrix
     * is given by the rdist and cdist vectors, with processor p
     * owning rows [rdist[p],rdist[p+1]). cdist is only needed for
     * non-square matrices. S is distributed according to rdist if
     * t==Trans::N, else cdist. R is distributed according to cdist of
     * t==Trans::N, else rdist.
     *
     * \param op whether to compute the transposed/conjugate product
     * \param R random matrix passed to user code
     * \param S sample matrix to be computed by user code: \n
     *            S = A*R if op is Trans::N \n
     *            S = A*T*R if op is Trans::T \n
     *            S = A^C*R if op is Trans::C
     * \param rdist Matrix row distribution, same on all ranks
     * \param cdist Matrix column distribution, same on all ranks
     */
    template<typename scalar_t>
    using mult_1d_t = std::function
      <void(Trans op,
            const DenseMatrix<scalar_t>& R,
            DenseMatrix<scalar_t>& S,
            const std::vector<int>& rdist,
            const std::vector<int>& cdist)>;
#endif

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
     * \tparam scalar_t Can be float, double, std::complex<float> or
     * std::complex<double>. Some formats do not support all
     * precisions, see the table below.
     *
     * For construction, use one of:
     * - structured::construct_from_dense (DENSE)
     * - structured::construct_from_elements (ELEM)
     * - structured::construct_matrix_free (MF)
     * - structured::construct_partially_matrix_free (PMF)
     * - nearest neighbors (NN)
     *
     * However, not all formats support all construction methods,
     * matrix operations, or precisions (s: float, d: double, c:
     * std::complex<float>, z: std::complex<double>):
     *
     * |           |  parallel? || construct from ..        ||||| operation                  |||| precision  ||||
     * |-----------|------|------|-------|------|----|-----|----|------|--------|-------|-------|---|---|---|---|
     * |  ^        |  seq | MPI  | DENSE | ELEM | MF | PMF | NN | mult | factor | solve | shift | s | d | c | z |
     * | BLR       |  X   |  X   | X     |  X   |    |     |    | X    |   X    |  X    | ?     | X | X | X | X |
     * | HSS       |  X   |  X   | X     |      |    | X   | X  |  X   |   X    |  X    | X     | X | X | X | X |
     * | HODLR     |      |  X   | X     |  X   | X  |     | X  |  X   |   X    |  X    | ?     |   | X |   | X |
     * | HODBF     |      |  X   | X     |  X   | X  |     | X  |  X   |   X    |  X    | ?     |   | X |   | X |
     * | BUTTERFLY |      |  X   | X     |  X   | X  |     | X  |  X   |        |       |       |   | X |   | X |
     * | LR        |      |  X   | X     |  X   | X  |     | X  |  X   |        |       |       |   | X |   | X |
     * | LOSSY     |  X   |      | X     |      |    |     |    |      |        |       |       | X | X | X | X |
     * | LOSSLESS  |  X   |      | X     |      |    |     |    |      |        |       |       | X | X | X | X |
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
       * For a distributed matrix, which uses a block row
       * distribution, this gives the number of rows stored on this
       * process.
       *
       * \return local rows, should be dist()[p+1]-dist()[p] for rank
       * p
       */
      virtual std::size_t local_rows() const {
        throw std::invalid_argument
          ("1d block row distribution not supported for this format.");
      }

      /**
       * For a distributed matrix, which uses a block row
       * distribution, this gives the first rows stored on this
       * process.
       *
       * \return first rows, should be dist()[p] for rank p
       */
      virtual std::size_t begin_row() const {
        throw std::invalid_argument
          ("1d block row distribution not supported for this format.");
      }

      /**
       * For a distributed matrix, which uses a block row
       * distribution, this gives the final row (+1) stored on this
       * process.
       *
       * \return last row + 1, should be dist()[p+1] for rank p
       */
      virtual std::size_t end_row() const {
        throw std::invalid_argument
          ("1d block row distribution not supported for this format.");
      }

      /**
       * For a distributed matrix, return the 1D block row
       * distribution over processes. This is for square matrices, for
       * rectagular use rdist for the rows and cdist for the columns.
       *
       * \return vector with P+1 elements, process with rank p will
       * own rows [dist()[p],dist()[p+1])
       */
      virtual const std::vector<int>& dist() const {
        static std::vector<int> d = {0, int(rows())};
        return d;
      };
      /**
       * For a distributed rectangular matrix, return the 1D block row
       * distribution.
       *
       * \return vector with P+1 elements, process with rank p will
       * own rows [cdist()[p],cdist()[p+1])
       */
      virtual const std::vector<int>& rdist() const {
        return dist();
      };
      /**
       * For a distributed rectangular matrix, return the 1D block
       * columns distribution.
       *
       * \return vector with P+1 elements, process with rank p will
       * own columns [cdist()[p],cdist()[p+1])
       */
      virtual const std::vector<int>& cdist() const {
        return dist();
      };

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
       * Multiply the StructuredMatrix (A) with a dense matrix: y =
       * op(A)*x.
       *
       * \param op take transpose/conjugate or not
       * \param m columns in x and y
       * \param x matrix, x should have rows == op(A).cols()
       * \param ldx leading dimension of x
       * \param y matrix, y should have cols == x.cols(), and rows ==
       * A.rows()
       * \param ldy leading dimension of y
       */
      void mult(Trans op, int m, const scalar_t* x, int ldx,
                scalar_t* y, int ldy) const;

#if defined(STRUMPACK_USE_MPI)
      /**
       * Multiply the StructuredMatrix (A) with a dense matrix: y =
       * op(A)*x. x and y are 2d block cyclic.
       *
       * \param op take transpose/conjugate or not
       * \param x matrix, x.rows() == op(A).cols()
       * \param y matrix, y.cols() == x.cols(), y.rows() == A.rows()
       */
      virtual void mult(Trans op, const DistributedMatrix<scalar_t>& x,
                        DistributedMatrix<scalar_t>& y) const;
#endif

      /**
       * Compute a factorization (or the inverse) of this matrix, to
       * be used later for solving linear systems. The actual type of
       * factorization depends on the StructuredMatrix::Type of this
       * matrix.
       **/
      virtual void factor();

      /**
       * Solve a linear system A*x=b, with this StructuredMatrix
       * (A). This solve is done in-place.
       *
       * \param b right-hand side, b.rows() == A.cols(), will be
       * overwritten with the solution x.
       */
      virtual void solve(DenseMatrix<scalar_t>& b) const;

      /**
       * Solve a linear system A*x=b, with this StructuredMatrix
       * (A). This solve is done in-place.
       *
       * \param nrhs number of right-hand sides
       * \param b right-hand side, should have this->cols() rows, will
       * be overwritten with the solution x.
       * \param ldb leading dimension of b
       */
      virtual void solve(int nrhs, scalar_t* b, int ldb) const {
        int lr = rows();
        try { lr = local_rows(); }
        catch(...) {}
        DenseMatrixWrapper<scalar_t> B(lr, nrhs, b, ldb);
        solve(B);
      }

#if defined(STRUMPACK_USE_MPI)
      /**
       * Solve a linear system A*x=b, with this StructuredMatrix
       * (A). This solve is done in-place.
       *
       * \param b right-hand side, b.rows() == A.cols(), will be
       * overwritten with the solution x.
       */
      virtual void solve(DistributedMatrix<scalar_t>& b) const;
#endif

      /**
       * Apply a shift to the diagonal of this matrix. Ie, this +=
       * s*I, with I the identity matrix. If this is called after
       * calling factor, then the factors are not updated. To solve a
       * linear system with the shifted matrix, you need to call
       * factor again.
       *
       * \param s Shift to be applied to the diagonal.
       */
      virtual void shift(scalar_t s);

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
     * \param row_tree optional clustertree for the rows, see also
     * strumpack::binary_tree_clustering
     * \param col_tree optional clustertree for the columns. If the
     * matrix is square, this does not need to be specified.
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \throw std::invalid_argument If the operatation is not
     * supported for the type of structured::StructuredMatrix, if the
     * type requires a square matrix and the input is not square, if
     * the structured::StructuredMatrix type requires MPI.
     * \throw std::logic_error If the operation is not implemented yet
     * \throw std::runtime_error If the operation requires a third
     * party library which was not enabled when configuring STRUMPACK.
     *
     * \see strumpack::binary_tree_clustering, construct_from_dense
     * construct_from_elements, construct_matrix_free and
     * construct_partially_matrix_free
     */
    template<typename scalar_t>
    std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_dense(const DenseMatrix<scalar_t>& A,
                         const StructuredOptions<scalar_t>& opts,
                         const structured::ClusterTree* row_tree=nullptr,
                         const structured::ClusterTree* col_tree=nullptr);

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
     * \param row_tree optional clustertree for the rows, see also
     * strumpack::binary_tree_clustering
     * \param col_tree optional clustertree for the columns. If the
     * matrix is square, this does not need to be specified.
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \throw std::invalid_argument If the operatation is not
     * supported for the type of structured::StructuredMatrix, if the
     * type requires a square matrix and the input is not square, if
     * the structured::StructuredMatrix type requires MPI.
     * \throw std::logic_error If the operation is not implemented yet
     * \throw std::runtime_error If the operation requires a third
     * party library which was not enabled when configuring STRUMPACK.
     *
     * \see strumpack::binary_tree_clustering, construct_from_dense
     * construct_from_elements, construct_matrix_free and
     * construct_partially_matrix_free
     */
    template<typename scalar_t>
    std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_dense(int rows, int cols, const scalar_t* A, int ldA,
                         const StructuredOptions<scalar_t>& opts,
                         const structured::ClusterTree* row_tree=nullptr,
                         const structured::ClusterTree* col_tree=nullptr);

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
     * \param row_tree optional clustertree for the rows, see also
     * strumpack::binary_tree_clustering
     * \param col_tree optional clustertree for the columns. If the
     * matrix is square, this does not need to be specified.
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \throw std::invalid_argument If the operatation is not
     * supported for the type of structured::StructuredMatrix, if the
     * type requires a square matrix and the input is not square, if
     * the structured::StructuredMatrix type requires MPI.
     * \throw std::logic_error If the operation is not implemented yet
     * \throw std::runtime_error If the operation requires a third
     * party library which was not enabled when configuring STRUMPACK.
     *
     * \see strumpack::binary_tree_clustering, construct_from_dense
     * construct_from_elements, construct_matrix_free and
     * construct_partially_matrix_free
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(int rows, int cols,
                            const extract_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts,
                            const structured::ClusterTree* row_tree=nullptr,
                            const structured::ClusterTree* col_tree=nullptr);


    /**
     * Construct a StructuredMatrix using a routine to extract a
     * sub-block from the matrix.
     *
     * \tparam scalar_t precision of input matrix, and of
     * constructed StructuredMatrix. Note that not all types support
     * every all precisions. See StructuredMatrix::Type.
     *
     * \param rows Number of rows of matrix to be constructed.
     * \param cols Number of columns of matrix to be constructed.
     * \param A Matrix block extraction routine.
     * \param opts Options object
     * \param row_tree optional clustertree for the rows, see also
     * strumpack::binary_tree_clustering
     * \param col_tree optional clustertree for the columns. If the
     * matrix is square, this does not need to be specified.
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \throw std::invalid_argument If the operatation is not
     * supported for the type of structured::StructuredMatrix, if the
     * type requires a square matrix and the input is not square, if
     * the structured::StructuredMatrix type requires MPI.
     * \throw std::logic_error If the operation is not implemented yet
     * \throw std::runtime_error If the operation requires a third
     * party library which was not enabled when configuring STRUMPACK.
     *
     * \see strumpack::binary_tree_clustering, construct_from_dense
     * construct_from_elements, construct_matrix_free and
     * construct_partially_matrix_free
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(int rows, int cols,
                            const extract_block_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts,
                            const structured::ClusterTree* row_tree=nullptr,
                            const structured::ClusterTree* col_tree=nullptr);

    /**
     * Construct a StructuredMatrix using only a matrix-vector
     * multiplication routine.
     *
     * \tparam scalar_t precision of input matrix, and of
     * constructed StructuredMatrix. Note that not all types support
     * every all precisions. See StructuredMatrix::Type.
     *
     * \param rows Number of rows of matrix to be constructed.
     * \param cols Number of columns of matrix to be constructed.
     * \param Amult Matrix-(multi)vector multiplication routine.
     * \param opts Options object
     * \param row_tree optional clustertree for the rows, see also
     * strumpack::binary_tree_clustering
     * \param col_tree optional clustertree for the columns. If the
     * matrix is square, this does not need to be specified.
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \throw std::invalid_argument If the operatation is not
     * supported for the type of structured::StructuredMatrix, if the
     * type requires a square matrix and the input is not square, if
     * the structured::StructuredMatrix type requires MPI.
     * \throw std::logic_error If the operation is not implemented yet
     * \throw std::runtime_error If the operation requires a third
     * party library which was not enabled when configuring STRUMPACK.
     *
     * \see strumpack::binary_tree_clustering, construct_from_dense
     * construct_from_elements, construct_matrix_free and
     * construct_partially_matrix_free
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_matrix_free(int rows, int cols,
                          const mult_t<scalar_t>& Amult,
                          const StructuredOptions<scalar_t>& opts,
                          const structured::ClusterTree* row_tree=nullptr,
                          const structured::ClusterTree* col_tree=nullptr);

    /**
     * Construct a StructuredMatrix using both a matrix-vector
     * multiplication routine and a routine to extract a matrix
     * sub-block.
     *
     * \tparam scalar_t precision of input matrix, and of
     * constructed StructuredMatrix. Note that not all types support
     * every all precisions. See StructuredMatrix::Type.
     *
     * \param rows Number of rows of matrix to be constructed.
     * \param cols Number of columns of matrix to be constructed.
     * \param Amult Matrix-(multi)vector multiplication routine.
     * \param Aelem Matrix block extraction routine.
     * \param opts Options object
     * \param row_tree optional clustertree for the rows, see also
     * strumpack::binary_tree_clustering
     * \param col_tree optional clustertree for the columns. If the
     * matrix is square, this does not need to be specified.
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \throw std::invalid_argument If the operatation is not
     * supported for the type of structured::StructuredMatrix, if the
     * type requires a square matrix and the input is not square, if
     * the structured::StructuredMatrix type requires MPI.
     * \throw std::logic_error If the operation is not implemented yet
     * \throw std::runtime_error If the operation requires a third
     * party library which was not enabled when configuring STRUMPACK.
     *
     * \see strumpack::binary_tree_clustering, construct_from_dense
     * construct_from_elements, construct_matrix_free and
     * construct_partially_matrix_free
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<scalar_t>& Amult,
                                    const extract_block_t<scalar_t>& Aelem,
                                    const StructuredOptions<scalar_t>& opts,
                                    const structured::ClusterTree* row_tree=nullptr,
                                    const structured::ClusterTree* col_tree=nullptr);

    /**
     * Construct a StructuredMatrix using both a matrix-vector
     * multiplication routine and a routine to extract individual
     * matrix elements.
     *
     * \tparam scalar_t precision of input matrix, and of
     * constructed StructuredMatrix. Note that not all types support
     * every all precisions. See StructuredMatrix::Type.
     *
     * \param rows Number of rows of matrix to be constructed.
     * \param cols Number of columns of matrix to be constructed.
     * \param Amult Matrix-(multi)vector multiplication routine.
     * \param Aelem Matrix element extraction routine.
     * \param opts Options object
     * \param row_tree optional clustertree for the rows, see also
     * strumpack::binary_tree_clustering
     * \param col_tree optional clustertree for the columns. If the
     * matrix is square, this does not need to be specified.
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \throw std::invalid_argument If the operatation is not
     * supported for the type of structured::StructuredMatrix, if the
     * type requires a square matrix and the input is not square, if
     * the structured::StructuredMatrix type requires MPI.
     * \throw std::logic_error If the operation is not implemented yet
     * \throw std::runtime_error If the operation requires a third
     * party library which was not enabled when configuring STRUMPACK.
     *
     * \see strumpack::binary_tree_clustering, construct_from_dense
     * construct_from_elements, construct_matrix_free and
     * construct_partially_matrix_free
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<scalar_t>& Amult,
                                    const extract_t<scalar_t>& Aelem,
                                    const StructuredOptions<scalar_t>& opts,
                                    const structured::ClusterTree* row_tree=nullptr,
                                    const structured::ClusterTree* col_tree=nullptr);


#if defined(STRUMPACK_USE_MPI)
    /**
     * Construct a StructuredMatrix from a 2D block-cyclicly
     * distributed matrix (ScaLAPACK layout).
     *
     * \tparam scalar_t precision of input matrix, and of
     * constructed StructuredMatrix. Note that not all types support
     * every all precisions. See StructuredMatrix::Type.
     *
     * \param A 2D block cyclic distributed matrix (ScaLAPACK layout).
     * \param opts Options object
     * \param row_tree optional clustertree for the rows, see also
     * strumpack::binary_tree_clustering
     * \param col_tree optional clustertree for the columns. If the
     * matrix is square, this does not need to be specified.
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \throw std::invalid_argument If the operatation is not
     * supported for the type of structured::StructuredMatrix, if the
     * type requires a square matrix and the input is not square, if
     * the structured::StructuredMatrix type requires MPI.
     * \throw std::logic_error If the operation is not implemented yet
     * \throw std::runtime_error If the operation requires a third
     * party library which was not enabled when configuring STRUMPACK.
     *
     * \see strumpack::binary_tree_clustering, construct_from_dense
     * construct_from_elements, construct_matrix_free and
     * construct_partially_matrix_free
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_dense(const DistributedMatrix<scalar_t>& A,
                         const StructuredOptions<scalar_t>& opts,
                         const structured::ClusterTree* row_tree=nullptr,
                         const structured::ClusterTree* col_tree=nullptr);

    /**
     * Construct a StructuredMatrix using an element extraction
     * routine.
     *
     * \tparam scalar_t precision of input matrix, and of
     * constructed StructuredMatrix. Note that not all types support
     * every all precisions. See StructuredMatrix::Type.
     *
     * \param comm MPI communicator (wrapper class)
     * \param rows Number of rows of matrix to be constructed.
     * \param cols Number of columns of matrix to be constructed.
     * \param A Matrix element extraction routine.
     * \param opts Options object
     * \param row_tree optional clustertree for the rows, see also
     * strumpack::binary_tree_clustering
     * \param col_tree optional clustertree for the columns. If the
     * matrix is square, this does not need to be specified.
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \throw std::invalid_argument If the operatation is not
     * supported for the type of structured::StructuredMatrix, if the
     * type requires a square matrix and the input is not square, if
     * the structured::StructuredMatrix type requires MPI.
     * \throw std::logic_error If the operation is not implemented yet
     * \throw std::runtime_error If the operation requires a third
     * party library which was not enabled when configuring STRUMPACK.
     *
     * \see strumpack::binary_tree_clustering, construct_from_dense
     * construct_from_elements, construct_matrix_free and
     * construct_partially_matrix_free
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts,
                            const structured::ClusterTree* row_tree=nullptr,
                            const structured::ClusterTree* col_tree=nullptr);

    /**
     * Construct a StructuredMatrix using a routine to extract a
     * matrix sub-block.
     *
     * \tparam scalar_t precision of input matrix, and of
     * constructed StructuredMatrix. Note that not all types support
     * every all precisions. See StructuredMatrix::Type.
     *
     * \param comm MPI communicator (wrapper class)
     * \param rows Number of rows of matrix to be constructed.
     * \param cols Number of columns of matrix to be constructed.
     * \param A Matrix sub-block extraction routine.
     * \param opts Options object
     * \param row_tree optional clustertree for the rows, see also
     * strumpack::binary_tree_clustering
     * \param col_tree optional clustertree for the columns. If the
     * matrix is square, this does not need to be specified.
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \throw std::invalid_argument If the operatation is not
     * supported for the type of structured::StructuredMatrix, if the
     * type requires a square matrix and the input is not square, if
     * the structured::StructuredMatrix type requires MPI.
     * \throw std::logic_error If the operation is not implemented yet
     * \throw std::runtime_error If the operation requires a third
     * party library which was not enabled when configuring STRUMPACK.
     *
     * \see strumpack::binary_tree_clustering, construct_from_dense
     * construct_from_elements, construct_matrix_free and
     * construct_partially_matrix_free
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_dist_block_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts,
                            const structured::ClusterTree* row_tree=nullptr,
                            const structured::ClusterTree* col_tree=nullptr);

    /**
     * Construct a StructuredMatrix using only a matrix-vector
     * multiplication routine.
     *
     * \param comm MPI communicator (wrapper class)
     * \param rows Number of rows of matrix to be constructed.
     * \param cols Number of columns of matrix to be constructed.
     * \param A Matrix-(multi)vector multiplication routine (using 2d
     * block cyclic layout for vectors).
     * \param opts Options object
     * \param row_tree optional clustertree for the rows, see also
     * strumpack::binary_tree_clustering
     * \param col_tree optional clustertree for the columns. If the
     * matrix is square, this does not need to be specified.
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \throw std::invalid_argument If the operatation is not
     * supported for the type of structured::StructuredMatrix, if the
     * type requires a square matrix and the input is not square, if
     * the structured::StructuredMatrix type requires MPI.
     * \throw std::logic_error If the operation is not implemented yet
     * \throw std::runtime_error If the operation requires a third
     * party library which was not enabled when configuring STRUMPACK.
     *
     * \see strumpack::binary_tree_clustering, construct_from_dense
     * construct_from_elements, construct_matrix_free and
     * construct_partially_matrix_free
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_matrix_free(const MPIComm& comm, const BLACSGrid* g,
                          int rows, int cols,
                          const mult_2d_t<scalar_t>& Amult,
                          const StructuredOptions<scalar_t>& opts,
                          const structured::ClusterTree* row_tree=nullptr,
                          const structured::ClusterTree* col_tree=nullptr);

    /**
     * Construct a StructuredMatrix using only a matrix-vector
     * multiplication routine.
     *
     * \param comm MPI communicator (wrapper class)
     * \param rows Number of rows of matrix to be constructed.
     * \param cols Number of columns of matrix to be constructed.
     * \param A Matrix-(multi)vector multiplication routine (using 1d
     * block row distribution for vectors).
     * \param opts Options object
     * \param row_tree optional clustertree for the rows, see also
     * strumpack::binary_tree_clustering
     * \param col_tree optional clustertree for the columns. If the
     * matrix is square, this does not need to be specified.
     *
     * \return std::unique_ptr holding a pointer to a
     * StructuredMatrix of the requested StructuredMatrix::Type
     *
     * \throw std::invalid_argument If the operatation is not
     * supported for the type of structured::StructuredMatrix, if the
     * type requires a square matrix and the input is not square, if
     * the structured::StructuredMatrix type requires MPI.
     * \throw std::logic_error If the operation is not implemented yet
     * \throw std::runtime_error If the operation requires a third
     * party library which was not enabled when configuring STRUMPACK.
     *
     * \see strumpack::binary_tree_clustering, construct_from_dense
     * construct_from_elements, construct_matrix_free and
     * construct_partially_matrix_free
     */
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_matrix_free(const MPIComm& comm, int rows, int cols,
                          const mult_1d_t<scalar_t>& Amult,
                          const StructuredOptions<scalar_t>& opts,
                          const structured::ClusterTree* row_tree=nullptr,
                          const structured::ClusterTree* col_tree=nullptr);

#endif

  } // end namespace structured
} // end namespace strumpack

#endif // STRUCTURED_MATRIX_HPP
