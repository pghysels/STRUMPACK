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
 * \file DenseMatrix.hpp
 * \brief Contains the DenseMatrix and DenseMatrixWrapper classes,
 * simple wrappers around BLAS/LAPACK style dense matrices.
 */
#ifndef DENSE_MATRIX_HPP
#define DENSE_MATRIX_HPP

#include <string>
#include <iomanip>
#include <cassert>
#include <algorithm>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "misc/Tools.hpp"
#include "misc/RandomWrapper.hpp"
#include "misc/TaskTimer.hpp"
#include "BLASLAPACKWrapper.hpp"
#include "BLASLAPACKOpenMPTask.hpp"

namespace strumpack {

  /**
   * Operation to perform on the matrix, as used by several BLAS
   * routines.  \ingroup Enumerations
   */
  enum class Trans : char {
    N='N',  /*!< No transpose      */
    C='C',  /*!< Complex conjugate */
    T='T'   /*!< Transpose         */
  };

  inline Trans c2T(char op) {
    switch (op) {
    case 'n': case 'N': return Trans::N;
    case 't': case 'T': return Trans::T;
    case 'c': case 'C': return Trans::C;
    default:
      std::cerr << "ERROR: char " << op << " not recognized,"
                << " should be one of n/N, t/T or c/C" << std::endl;
      return Trans::N;
    }
  }


  /**
   * Which side to apply the operation on, as used by several BLAS
   * routines.  \ingroup Enumerations
   */
  enum class Side : char {
    L='L',  /*!< Left side         */
    R='R'   /*!< Right side        */
  };

  /**
   * Which triangular part of the matrix to consider, as used by
   * several BLAS routines.  \ingroup Enumerations
   */
  enum class UpLo : char {
    U='U',  /*!< Upper triangle    */
    L='L'   /*!< Lower triangle    */
  };

  /**
   * Whether the matrix in unit diagonal or not.  \ingroup
   * Enumerations
   */
  enum class Diag : char {
    U='U',  /*!< Unit diagonal     */
    N='N'   /*!< Non-unit diagonal */
  };

  /**
   * Job for eigenvalue/vector computations
   * \ingroup Enumerations
   */
  enum class Jobz : char {
    N='N', /*!< Compute eigenvalues only             */
    V='V'  /*!< Compute eigenvalues and eigenvectors */
  };


  /**
   * \class DenseMatrix
   * \brief This class represents a matrix, stored in column major
   * format, to allow direct use of BLAS/LAPACK routines.
   *
   * This class represents a (2D) matrix, stored in column major
   * format, to allow direct use of BLAS/LAPACK routines. A
   * DenseMatrix allocates, owns and deallocates its memory. If you
   * want to use pre-allocated memory to represent a dense matrix, use
   * the DenseMatrixWrapper<scalar_t> class.
   *
   * Several routines in this matrix perform some sort of bounds or
   * size checking using __assertions__. These assertions can be
   * removed by compiling with -DNDEBUG, which is added by default
   * when using a CMake Release build.
   *
   * Several routines in this class take a __depth__ parameter. This
   * refers to the depth of the nested OpenMP task spawning. No more
   * tasks will be generated once the depth reaches a certain maximum
   * level (params::task_recursion_cutoff_level), in order to limit
   * the overhead of task spawning.
   *
   * \tparam scalar_t Possible values for the scalar_t template type
   * are: float, double, std::complex<float> and std::complex<double>.
   *
   * Several BLAS-like interfaces are provided:
   * \see strumpack::gemm
   * \see strumpack::trsm
   * \see strumpack::trmm
   * \see strumpack::gemv
   */
  template<typename scalar_t> class DenseMatrix {
    using real_t = typename RealType<scalar_t>::value_type;

  protected:
    scalar_t* data_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::size_t ld_ = 1;

  public:

    /**
     * Default constructor, constucts 0x0 empty matrix, with leading
     * dimension 1.
     */
    DenseMatrix();

    /**
     * Constructs, and allocates, an m x n dense matrix, using column
     * major storage. The leading dimension will be max(1, m).
     *
     * \param m Number of rows in the constructed matrix.
     * \param n Number of columns in the constructed matrix.
     */
    DenseMatrix(std::size_t m, std::size_t n);

    /**
     * Construct/allocate a dense m x n matrix, and initialize it by
     * copying the data pointed to by D (with leading dimension
     * ld).
     *
     * \param m Number of rows in the constructed matrix.
     * \param n Number of columns in the constructed matrix.
     * \param D pointer to data to be copied in newly allocated
     * DenseMatrix. Cannot be null.
     * \param ld Leading dimension of logically 2D matrix pointed to
     * by D. Should be >= m.
     */
    DenseMatrix
    (std::size_t m, std::size_t n, const scalar_t* D, std::size_t ld);

    /**
     * Construct a dense m x n matrix by copying a submatrix from
     * matrix D. The copied submatrix has has top-left corner at
     * position i,j in D, i.e. is D(i:i+m,j:j+n). The submatrix should
     * be contained in D, i.e.: i+m <= D.rows() and j+n <= D.cols().
     *
     * \param m Number of rows in the constructed matrix.
     * \param n Number of columns in the constructed matrix.
     * \param D Matrix from which the newly constructed DenseMatrix is
     * a submatrix.
     * \param i Row-offset in D, denoting top side of submatrix to
     * copy to new matrix.
     * \param j Column-offset in D, denoting left side of submatrix to
     * copy to new matrix.
     */
    DenseMatrix
    (std::size_t m, std::size_t n, const DenseMatrix<scalar_t>& D,
     std::size_t i, std::size_t j);

    /** Copy constructor */
    DenseMatrix(const DenseMatrix<scalar_t>& D);

    /** Move constructor */
    DenseMatrix(DenseMatrix<scalar_t>&& D);

    /** Destructor */
    virtual ~DenseMatrix();

    /** Copy operator, expensive operation for large matrices. */
    virtual DenseMatrix<scalar_t>& operator=(const DenseMatrix<scalar_t>& D);

    /**
     * Move operator
     * \param D Matrix to be moved into this object, will be emptied.
     */
    virtual DenseMatrix<scalar_t>& operator=(DenseMatrix<scalar_t>&& D);

    /** Number of rows of the matrix */
    inline std::size_t rows() const { return rows_; }

    /** Number of columns of the matrix */
    inline std::size_t cols() const { return cols_; }

    /**
     * Leading dimension used to store the matrix, typically set to
     * max(1, rows())
     */
    inline std::size_t ld() const { return ld_; }

    /**
     * Const pointer to the raw data used to represent this matrix.
     */
    inline const scalar_t* data() const { return data_; }

    /**
     * Pointer to the raw data used to represent this matrix.
     */
    inline scalar_t* data() { return data_; }

    /**
     * Pointer to the element after the last one in this matrix.
     * end() == data() + size()
     */
    inline scalar_t* end() { return data_ + ld_ * cols_; }

    /**
     * Const reference to element (i,j) in the matrix. This will do a
     * bounds check with assertions, which are enabled in Debug mode,
     * disabled in Release mode.
     *
     * \param i Row index, i < rows()
     * \param j Column index, j < cols()
     */
    inline const scalar_t& operator()(std::size_t i, std::size_t j) const
    { assert(i>=0 && i<=rows() && j>=0 && j<=cols()); return data_[i+ld_*j]; }

    /**
     * Const pointer to element (i,j) in the matrix. This will do a
     * bounds check with assertions, which are enabled in Debug mode,
     * disabled in Release mode.
     *
     * \param i Row index, i < rows()
     * \param j Column index, j < cols()
     */
    inline const scalar_t* ptr(std::size_t i, std::size_t j) const
    { assert(i>=0 && i<=rows() && j>=0 && j<=cols()); return data_+i+ld_*j; }

    /**
     * Reference to element (i,j) in the matrix. This will do a bounds
     * check with assertions, which are enabled in Debug mode,
     * disabled in Release mode.
     *
     * \param i Row index, i < rows()
     * \param j Column index, j < cols()
     */
    inline scalar_t& operator()(std::size_t i, std::size_t j)
    { assert(i>=0 && i<=rows() && j>=0 && j<=cols()); return data_[i+ld_*j]; }

    /**
     * Pointer to element (i,j) in the matrix. This will do a bounds
     * check with assertions, which are enabled in Debug mode,
     * disabled in Release mode.
     *
     * \param i Row index, i < rows()
     * \param j Column index, j < cols()
     */
    inline scalar_t* ptr(std::size_t i, std::size_t j)
    { assert(i>=0 && i<=rows() && j>=0 && j<=cols()); return data_+i+ld_*j; }

    /**
     * Print the matrix to std::cout, in a format interpretable by
     * Matlab/Octave. The matrix will only be printed in full when not
     * too big, i.e., when rows() <= 20 and cols() <= 32. Otherwise
     * only its sizes and its norm are printed. Useful for debugging.
     */
    void print() const { print("A"); }

    /**
     * Print the matrix to std::cout, in a format interpretable by
     * Matlab/Octave. The matrix is printed in full when not too big,
     * i.e., when rows() <= 20 and cols() <= 32, or when all is set to
     * true. Otherwise only its sizes and its norm are printed. Useful
     * for debugging.
     *
     * \param name Name to use when printing.
     * \param all If true, print all values, if false, only print all
     * values when not too big. Defaults to false.
     * \param param width Specifies how many digits to use for
     * printing floating point values, defaults to 8.
     */
    void print(std::string name, bool all=false, int width=8) const;

    /**
     * Print the matrix to a file, in a format readable by
     * Matlab/Octave.
     *
     * \param name Name to use for the printed matrix.
     * \param filename Name of the file to write to
     * \param width Number of digits to use to represent floating
     * point values, defaults to 8.
     */
    void print_to_file
    (std::string name, std::string filename, int width=8) const;

    /**
     * Fill the matrix with random numbers, using random number
     * generator/distribution
     * random::make_default_random_generator<real_t>()
     */
    void random();

    /**
     * Fill the matrix with random numbers, using the specified random
     * number generator.
     */
    void random
    (random::RandomGeneratorBase<typename RealType<scalar_t>::
     value_type>& rgen);

    /** Fill matrix with a constant value */
    void fill(scalar_t v);

    /** Set all matrix elements to zero */
    void zero();

    /**
     * Set the matrix to the identity matrix. Also works for
     * rectangular matrices.
     */
    void eye();

    /**
     * Clear the matrix. Resets the number of rows and columns to 0.
     */
    virtual void clear();

    /**
     * Resize the matrix. The relevant parts of the original matrix
     * will be copied to the new matrix. The contents of new parts of
     * the matrix are undefined.
     *
     * \param m Number of rows after resizing.
     * \param m Number of columns after resizing.
     */
    void resize(std::size_t m, std::size_t n);

    /**
     * Horizontally concatenate a matrix to this matrix: [this b].
     * The resulting matrix will have the same number of rows as b and
     * as this original matrix, and will have cols()+b.cols() columns.
     *
     * \param b Matrix to concatenate to this matrix. The b matrix
     * should have to same number of rows, rows == b.rows().
     */
    void hconcat(const DenseMatrix<scalar_t>& b);

    /**
     * Copy a submatrix of size rows() x cols() from B, at position
     * (i,j) into this matrix. The following conditions should be
     * satisfied: i+rows() <= B.rows() and j+cols() <= B.cols().
     *
     * \param B Matrix from which to copy
     * \param i Row-offset denoting the top of the submatrix of B to
     * copy
     * \param i Column-offset denoting the top of the submatrix of B
     * to copy
     */
    void copy
    (const DenseMatrix<scalar_t>& B, std::size_t i=0, std::size_t j=0);

    /**
     * Copy a submatrix of size rows() x cols() from matrix B, with
     * leading dimension ld, into this matrix.
     *
     * \param B Pointer to the matrix data to copy
     * \param ld Leading dimension of matrix pointed to by B, should
     * be at least cols()
     */
    void copy(const scalar_t* B, std::size_t ldb);

    /** Return the transpose of this matrix */
    DenseMatrix<scalar_t> transpose() const;

    /**
     * Set X to the transpose of this matrix.
     */
    void transpose(DenseMatrix<scalar_t>& X) const;

    /**
     * Apply the LAPACK routine xLASWP to the matrix. xLASWP performs
     * a series of row interchanges on the matrix.  One row
     * interchange is initiated for each of rows in the vector P.
     *
     * \param P vector with row interchanges, this is assumed to be of
     * size rows()
     * \param fwd if fwd is false, the pivots are applied in reverse
     * order
     */
    void laswp(const std::vector<int>& P, bool fwd);

    /**
     * Apply the LAPACK routine xLASWP to the matrix. xLASWP performs
     * a series of row interchanges on the matrix.  One row
     * interchange is initiated for each of rows in the vector P.
     *
     * \param P vector with row interchanges, this is assumed to be of
     * size rows()
     * \param fwd if fwd is false, the pivots are applied in reverse
     * order
     */
    void laswp(const int* P, bool fwd);

    /**
     * Apply the LAPACK routine xLAPMR to the matrix. xLAPMR
     * rearranges the rows of the M by N matrix X as specified by the
     * permutation K(1),K(2),...,K(M) of the integers 1,...,M.
     * If fwd == true, forward permutation:
     *    X(K(I),*) is moved X(I,*) for I = 1,2,...,M.
     * If fwd == false, backward permutation:
     *    X(I,*) is moved to X(K(I),*) for I = 1,2,...,M.
     *
     * \param P permutation vector, should be of size rows()
     * \param fwd apply permutation, or inverse permutation, see above
     */
    void lapmr(const std::vector<int>& P, bool fwd);

    /**
     * Apply the LAPACK routines xLAPMT to the matrix. xLAPMT
     * rearranges the columns of the M by N matrix X as specified by
     * the permutation K(1),K(2),...,K(N) of the integers 1,...,N.
     * If fwd == true, forward permutation:
     *     X(*,K(J)) is moved X(*,J) for J = 1,2,...,N.
     * If fwd == false, backward permutation:
     *     X(*,J) is moved to X(*,K(J)) for J = 1,2,...,N.
     *
     * \param P permutation vector, should be of size cols()
     * \param fwd apply permutation, or inverse permutation, see above
     */
    void lapmt(const std::vector<int>& P, bool fwd);

    /**
     * Extract rows of this matrix to a specified matrix. Row I[i] in
     * this matrix will become row i in matrix B.
     *
     * \param I set of indices of rows to extract from this
     * matrix. The elements of I should not be sorted, but they should
     * be I[i] < rows().
     * \params B matrix to extraxt to. B should have the correct size,
     * ie. B.cols() == cols() and B.rows() == I.size()
     */
    void extract_rows
    (const std::vector<std::size_t>& I, DenseMatrix<scalar_t>& B) const;

    /**
     * Return a matrix with rows I of this matrix.
     *
     * \param I set of indices of rows to extract from this
     * matrix. The elements of I should not be sorted, but they should
     * be I[i] < rows().
     * \params B matrix to extraxt to. B should have the correct size,
     * ie. B.cols() == cols() and B.rows() == I.size()
     */
    DenseMatrix<scalar_t> extract_rows
    (const std::vector<std::size_t>& I) const;

    /**
     * Extract columns of this matrix to a specified matrix. Column
     * I[i] in this matrix will become column i in matrix B.
     *
     * \param I set of indices of columns to extract from this
     * matrix. The elements of I should not be sorted, but they should
     * be I[i] < cols().
     * \params B matrix to extraxt to. B should have the correct size,
     * ie. B.cols() == I.size() and B.rows() == rows()
     */
    void extract_cols
    (const std::vector<std::size_t>& I, DenseMatrix<scalar_t>& B) const;

    /**
     * Return a matrix with columns I of this matrix.
     *
     * \param I set of indices of columns to extract from this
     * matrix. The elements of I should not be sorted, but they should
     * be I[i] < cols().
     * \params B matrix to extraxt to. B should have the correct size,
     * ie. B.cols() == I.size() and B.rows() == rows()
     */
    DenseMatrix<scalar_t> extract_cols
    (const std::vector<std::size_t>& I) const;

    /**
     * Return a submatrix of this matrix defined by (I,J). The vectors
     * I and J define the row and column indices of the submatrix. The
     * extracted submatrix will be I.size() x J.size(). The extracted
     * submatrix, lets call it B, satisfies B(i,j) =
     * this->operator()(I[i],J[j]).
     *
     * \param I row indices of elements to extract, I[i] < rows()
     * \param J column indices of elements to extract, J[j] < cols()
     */
    DenseMatrix<scalar_t> extract
    (const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J) const;

    /**
     * Add the rows of matrix B into this matrix at the rows specified
     * by vector I, ie, add row i of matrix B to row I[i] of this
     * matrix. This is used in the sparse solver.
     *
     * \param I index set in this matrix, where to add the rows of B,
     * I[i] < rows()
     * \param B matrix with rows to scatter in this matrix
     * \param depth current OpenMP task recursion depth
     */
    DenseMatrix<scalar_t>& scatter_rows_add
    (const std::vector<std::size_t>& I,
     const DenseMatrix<scalar_t>& B, int depth);

    /**
     * Add matrix B to this matrix. Return a reference to this matrix.
     *
     * \param B matrix to add to this matrix.
     * \param depth current OpenMP task recursion depth
     */
    DenseMatrix<scalar_t>& add(const DenseMatrix<scalar_t>& B, int depth=0);

    /**
     * Subtract matrix B from this matrix. Return a reference to this
     * matrix.
     *
     * \param B matrix to subtract from this matrix
     * \param depth current OpenMP task recursion depth
     */
    DenseMatrix<scalar_t>& sub(const DenseMatrix<scalar_t>& B, int depth=0);

    /**
     * Scale this matrix by a constant factor.
     *
     * \param alpha scaling factor
     */
    DenseMatrix<scalar_t>& scale(scalar_t alpha, int depth=0);

    /**
     * Add a scalar multiple of a given matrix to this matrix, ie,
     * this += alpha * B.
     *
     * \param alpha scalar factor
     * \param B matrix to add, should be the same size of this matrix
     * \param depth current OpenMP task recursion depth
     */
    DenseMatrix<scalar_t>& scaled_add
    (scalar_t alpha, const DenseMatrix<scalar_t>& B, int depth=0);

    /**
     * Scale this matrix, and add a given matrix to this matrix, ie,
     * this = alpha * this + B.
     *
     * \param alpha scalar factor
     * \param B matrix to add, should be the same size of this matrix
     * \param depth current OpenMP task recursion depth
     */
    DenseMatrix<scalar_t>& scale_and_add
    (scalar_t alpha, const DenseMatrix<scalar_t>& B, int depth=0);

    /**
     * Scale the rows of this matrix with the scalar values from the
     * vector D. Row i in this matrix is scaled with D[i].
     *
     * \param D scaling vector, D.size() == rows()
     * \param depth current OpenMP task recursion depth
     */
    DenseMatrix<scalar_t>& scale_rows
    (const std::vector<scalar_t>& D, int depth=0);

    /**
     * Scale the rows of this matrix with the inverses of the scalar
     * values in vector D, ie, this->operator()(i,j) /= D[i].
     *
     * \param D scalar factors, D.size() == rows()
     * \param depth current OpenMP task recursion depth
     */
    DenseMatrix<scalar_t>& div_rows
    (const std::vector<scalar_t>& D, int depth=0);

    /**
     * Return default norm of this matrix. Currently the default is
     * set to the Frobenius norm.
     */
    real_t norm() const;

    /**
     * Return the Frobenius norm of this matrix.
     */
    real_t normF() const;

    /**
     * Return the 1-norm of this matrix.
     */
    real_t norm1() const;

    /**
     * Return the infinity norm of this matrix.
     */
    real_t normI() const;

    /**
     * Return the (approximate) amount memory taken by this
     * memory. Simply nonzeros()*sizeof(scalar_t). The matrix metadata
     * is not counted in this.
     */
    virtual std::size_t memory() const {
      return sizeof(scalar_t) * rows() * cols();
    }

    /**
     * Return the number of nonzeros in this matrix, ie, simply
     * rows()*cols().
     */
    virtual std::size_t nonzeros() const {
      return rows()*cols();
    }

    /**
     * Compute an LU factorization of this matrix using partial
     * pivoting with row interchanges. The factorization has the form
     * A = P * L * U, where P is a permutation matrix, L is lower
     * triangular with unit diagonal elements, and U is upper
     * triangular. This calls the LAPACK routine DGETRF. The L and U
     * factors are stored in place, the permutation is returned, and
     * can be applied with the laswp() routine.
     *
     * \param depth current OpenMP task recursion depth
     * \see laswp, solve
     */
    std::vector<int> LU(int depth=0);

    /**
     * Compute a Cholesky factorization of this matrix in-place. This
     * calls the LAPACK routine DPOTRF. Only the lower triangle is
     * written. Only the lower triangle is referenced/stored.
     *
     * \param depth current OpenMP task recursion depth
     * \return info from xpotrf
     * \see LU, LDLt
     */
    int Cholesky(int depth=0);

    /**
     * Compute an LDLt factorization of this matrix in-place. This
     * calls the LAPACK routine sytrf. Only the lower triangle is
     * referenced/stored.
     *
     * \param depth current OpenMP task recursion depth
     * \return info from xsytrf
     * \see LU, Cholesky, solve_LDLt
     */
    std::vector<int> LDLt(int depth=0);

    /**
     * Compute an LDLt factorization of this matrix in-place. This
     * calls the LAPACK routine sytrf_rook. Only the lower triangle is
     * referenced/stored.
     *
     * Disabled for now because not supported on some systems with
     * older LAPACK versions.
     *
     * \param depth current OpenMP task recursion depth
     * \return info from xsytrf_rook
     * \see LU, Cholesky, solve_LDLt
     */
    //std::vector<int> LDLt_rook(int depth=0);


    /**
     * Solve a linear system Ax=b with this matrix, factored in its LU
     * factors (in place), using a call to this->LU. There can be
     * multiple right hand side vectors. The solution is returned by
     * value.
     *
     * \param b input, right hand side vector/matrix
     * \param piv pivot vector returned by LU factorization
     * \param depth current OpenMP task recursion depth
     * \return the solution x
     * \see LU, solve_LU_in_place, solve_LDLt_in_place, solve_LDLt_rook_in_place
     */
    DenseMatrix<scalar_t> solve
    (const DenseMatrix<scalar_t>& b,
     const std::vector<int>& piv, int depth=0) const;

    /**
     * Solve a linear system Ax=b with this matrix, factored in its LU
     * factors (in place), using a call to this->LU. There can be
     * multiple right hand side vectors.
     *
     * \param b input, right hand side vector/matrix. On output this
     * will be the solution.
     * \param piv pivot vector returned by LU factorization
     * \param depth current OpenMP task recursion depth
     * \see LU, solve_LU_in_place, solve_LDLt_in_place, solve_LDLt_rook_in_place
     */
    void solve_LU_in_place
    (DenseMatrix<scalar_t>& b, const std::vector<int>& piv, int depth=0) const;

    /**
     * Solve a linear system Ax=b with this matrix, factored in its
     * LDLt factors (in place). There can be multiple right hand side
     * vectors. The solution is returned by value.
     *
     * \param b input, right hand side vector/matrix. On output this
     * will be the solution.
     * \param piv pivot vector returned by LU factorization
     * \param depth current OpenMP task recursion depth
     * \see LDLt, LDLt_rook, solve_LDLt_rook_in_place, LU, solve_LU_in_place
     */
    void solve_LDLt_in_place
    (DenseMatrix<scalar_t>& b, const std::vector<int>& piv, int depth=0) const;

    /**
     * Solve a linear system Ax=b with this matrix, factored in its
     * LDLt factors (in place), using LDLt_rook. There can be multiple
     * right hand side vectors. The solution is returned by value.
     *
     * Disabled for now because not supported on some systems with
     * older LAPACK versions.
     *
     * \param b input, right hand side vector/matrix. On output this
     * will be the solution.
     * \param piv pivot vector returned by LU factorization
     * \param depth current OpenMP task recursion depth
     * \see LDLt_rook, LDLt, solve_LDLt_in_place, LU, solve_LU_in_place
     */
    // void solve_LDLt_rook_in_place
    // (DenseMatrix<scalar_t>& b, const std::vector<int>& piv, int depth=0) const;

    /**
     * Compute an LQ (lower triangular, unitary) factorization of this
     * matrix. This matrix is not modified, the L and Q factors are
     * returned by reference in the L and Q arguments. L and Q do not
     * have to be allocated, they will be resized accordingly.
     *
     * \param L lower triangular matrix, output argument. Does not
     * have to be allocated.
     * \param Q unitary matrix, not necessarily square. Does not have
     * to be allocated.
     * \param depth current OpenMP task recursion depth
     */
    void LQ
    (DenseMatrix<scalar_t>& L, DenseMatrix<scalar_t>& Q, int depth) const;

    /**
     * Builds an orthonormal basis for the columns in this matrix,
     * using QR factorization. It return the maximum and minimum
     * elements on the diagonal of the upper triangular matrix in the
     * QR factorization. These values can be used to check whether the
     * matrix was rank deficient.
     *
     * \param r_max maximum value encountered on the diagonal of R (as
     * returned from QR factorization)
     * \param r_min minimum value encountered on the diagonal of R (as
     * returned from QR factorization)
     * \param depth current OpenMP task recursion depth
     */
    void orthogonalize(scalar_t& r_max, scalar_t& r_min, int depth);

    /**
     * Compute an interpolative decomposition on the columns, ie,
     * write this matrix as a linear combination of some of its
     * columns. This is computed using QR with column pivoting
     * (dgeqp3, modified to take tolerances), also refered to as a
     * rank-revealing QR (RRQR), followed by a triangular solve.
     *
     * TODO check this, also definition of ind and piv??!!
     *
     *   this ~= this(:,ind) * [eye(rank,rank) X] \\
     *        ~= (this(:,piv))(:,1:rank) * [eye(rank,rank) X]
     *
     * \param X output, see description above, will have rank rows ???
     * \param piv pivot vector resulting from the column pivoted QR
     * \param ind set of columns selected from this matrix by RRQR
     * \param rel_tol relative tolerance used in the RRQR (column
     * pivoted QR)
     * \param abs_tol absolute tolerance used in the RRQR (column
     * pivoted QR)
     * \param max_rank maximum rank for RRQR
     * \param depth current OpenMP task recursion depth
     * \see ID_row
     */
    void ID_column
    (DenseMatrix<scalar_t>& X, std::vector<int>& piv,
     std::vector<std::size_t>& ind, real_t rel_tol,
     real_t abs_tol, int max_rank, int depth);

    /**
     * Similar to ID_column, but transposed. This is implemented by
     * calling ID_column on the transpose of this matrix, the
     * resulting X is then again transposed.
     *
     * \param X output, see description in ID_column, will have rank
     * columns ???
     * \param piv pivot vector resulting from the column pivoted QR
     * \param ind set of columns selected from this matrix by RRQR
     * \param rel_tol relative tolerance used in the RRQR (column
     * pivoted QR)
     * \param abs_tol absolute tolerance used in the RRQR (column
     * pivoted QR)
     * \param max_rank maximum rank for RRQR
     * \param depth current OpenMP task recursion depth
     * \see ID_column
     */
    void ID_row
    (DenseMatrix<scalar_t>& X, std::vector<int>& piv,
     std::vector<std::size_t>& ind, real_t rel_tol, real_t abs_tol,
     int max_rank, int depth) const;

    /**
     * Computes a low-rank factorization of this matrix, with
     * specified relative and absolute tolerances, (used in column
     * pivoted (rank-revealing) QR).
     *
     *    this ~= U * V
     *
     * \param U matrix U, low-rank factor. Will be of size
     * this->rows() x rank. Does not need to be allcoated beforehand.
     * \param V matrix V, low-rank factor. Will be of size rank x
     * this->cols(). Does not need to be allcoated beforehand.
     * \param rel_tol relative tolerance used in the RRQR (column
     * pivoted QR)
     * \param abs_tol absolute tolerance used in the RRQR (column
     * pivoted QR)
     * \param max_rank maximum rank for RRQR
     * \param depth current OpenMP task recursion depth
     */
    void low_rank
    (DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V,
     real_t rel_tol, real_t abs_tol, int max_rank, int depth) const;

    /**
     * Return a vector with the singular values of this matrix. Used
     * only for debugging puposes.
     */
    std::vector<scalar_t> singular_values() const;

    /**
     * Shift the diagonal with a value sigma, ie. add a scaled
     * identity matrix.
     *
     * \param sigma scalar value to add to diagonal
     */
    void shift(scalar_t sigma);


    /**
     * SYEV computes all eigenvalues and, optionally, eigenvectors of
     * this matrix. If job is N, the matrix is destroyed. If job is V,
     * on exit the matrix will contain all eigenvectors.
     *
     * \param job
     *   N:  Compute eigenvalues only,
     *   V:  Compute eigenvalues and eigenvectors.
     * \param ul
     *   U:  Upper triangle is stored,
     *   L:  Lower triangle is stored.
     * \param lambda
     *   on exit this will contain all eigenvalues of this matrix.
     */
    int syev(Jobz job, UpLo ul, std::vector<scalar_t>& lambda);

  private:
    void ID_column_GEQP3
    (DenseMatrix<scalar_t>& X, std::vector<int>& piv,
     std::vector<std::size_t>& ind, real_t rel_tol,
     real_t abs_tol, int max_rank, int depth);

    template<typename T> friend class DistributedMatrix;
  };


  /**
   * \class DenseMatrixWrapper
   * \brief Like DenseMatrix, this class represents a matrix, stored
   * in column major format, to allow direct use of BLAS/LAPACK
   * routines. However, objects of the DenseMatrixWrapper class do not
   * allocate, own or free any data.
   *
   * The user has to make sure that the memory that was used to create
   * this matrix stays valid for as long as this matrix wrapper, or
   * any other wrappers derived from this wrapper (submatrices), is
   * required. The DenseMatrixWrapper class is a subclass of
   * DenseMatrix, so a DenseMatrixWrapper can be used as a
   * DenseMatrix. A DenseMatrixWrapper can be used to wrap already
   * allocated memory, or to create a submatrix of a DenseMatrix.
   *
   * \tparam scalar_t Possible values for the scalar_t template type
   * are: float, double, std::complex<float> and std::complex<double>.
   *
   * \see DenseMatrix, ConstDenseMatrixWrapperPtr
   */
  template<typename scalar_t>
  class DenseMatrixWrapper : public DenseMatrix<scalar_t> {
  public:
    /**
     * Default constructor. Creates an empty, 0x0 matrix.
     */
    DenseMatrixWrapper() : DenseMatrix<scalar_t>() {}

    /**
     * Constructor. Create an m x n matrix wrapper using already
     * allocated memory, pointed to by D, with leading dimension ld.
     *
     * \param m number of rows of the new (sub) matrix
     * \param n number of columns of the new matrix
     * \param D pointer to memory representing matrix, this should
     * point to at least ld*n bytes of allocated memory
     * \param ld leading dimension of matrix allocated at D. ld >= m
     */
    DenseMatrixWrapper
    (std::size_t m, std::size_t n, scalar_t* D, std::size_t ld) {
      this->data_ = D; this->rows_ = m; this->cols_ = n;
      this->ld_ = std::max(std::size_t(1), ld);
    }

    /**
     * Constructor. Create a DenseMatrixWrapper as a submatrix of size
     * m x n, of a DenseMatrix (or DenseMatrixWrapper) D, at position
     * i,j in D. The constructed DenseMatrixWrapper will be the
     * submatrix D(i:i+m,j:j+n).
     *
     * \param m number of rows of the new (sub) matrix
     * \param n number of columns of the new matrix
     * \param D matrix from which to take a submatrix
     * \param i row offset in D of the top left corner of the submatrix
     * \param j columns offset in D of the top left corner of the
     * submatrix
     */
    DenseMatrixWrapper
    (std::size_t m, std::size_t n, DenseMatrix<scalar_t>& D,
     std::size_t i, std::size_t j)
      : DenseMatrixWrapper<scalar_t>(m, n, &D(i, j), D.ld()) {
      assert(i+m <= D.rows());
      assert(j+n <= D.cols());
    }

    /**
     * Virtual destructor. Since a DenseMatrixWrapper does not
     * actually own it's memory, put just keeps a pointer, this will
     * not free any memory.
     */
    virtual ~DenseMatrixWrapper() { this->data_ = nullptr; }

    /**
     * Clear the DenseMatrixWrapper. Ie, set to an empty matrix. This
     * will not affect the original matrix, to which this is a
     * wrapper, only the wrapper itself is reset. No memory is
     * released.
     */
    void clear() override {
      this->rows_ = 0; this->cols_ = 0;
      this->ld_ = 1; this->data_ = nullptr;
    }

    /**
     * Return the amount of memory taken by this wrapper, ie,
     * 0. (since the wrapper itself does not own the memory). The
     * memory will likely be owned by a DenseMatrix, while this
     * DenseMatrixWrapper is just a submatrix of that existing
     * matrix. Returning 0 here avoids counting memory double.
     *
     * \see nonzeros
     */
    std::size_t memory() const override { return 0; }

    /**
     * Return the number of nonzeros taken by this wrapper, ie,
     * 0. (since the wrapper itself does not own the memory). The
     * memory will likely be owned by a DenseMatrix, while this
     * DenseMatrixWrapper is just a submatrix of that existing
     * matrix. Returning 0 here avoids counting nonzeros double.
     *
     * \see memory
     */
    std::size_t nonzeros() const override { return 0; }

    /**
     * Default copy constructor, from another DenseMatrixWrapper.
     */
    DenseMatrixWrapper(const DenseMatrixWrapper<scalar_t>&) = default;

    /**
     * Constructing a DenseMatrixWrapper from a DenseMatrixWrapper is
     * not allowed.
     * TODO Why not??!! just delegate to DenseMatrixWrapper(m, n, D, i, j)??
     */
    DenseMatrixWrapper(const DenseMatrix<scalar_t>&) = delete;

    /**
     * Default move constructor.
     */
    DenseMatrixWrapper(DenseMatrixWrapper<scalar_t>&&) = default;

    /**
     * Moving from a DenseMatrix is not allowed.
     */
    DenseMatrixWrapper(DenseMatrix<scalar_t>&&) = delete;

    /**
     * Assignment operator. Shallow copy only. This only copies the
     * wrapper object. Does not copy matrix elements.
     *
     * \param D matrix wrapper to copy from, this will be duplicated
     */
    DenseMatrixWrapper<scalar_t>&
    operator=(const DenseMatrixWrapper<scalar_t>& D)
    { this->data_ = D.data(); this->rows_ = D.rows();
      this->cols_ = D.cols(); this->ld_ = D.ld(); return *this; }

    /**
     * Move assignment. This moves only the wrapper.
     *
     * \param D matrix wrapper to move from. This will not be
     * modified.
     */
    DenseMatrixWrapper<scalar_t>&
    operator=(DenseMatrixWrapper<scalar_t>&& D) {
      this->data_ = D.data(); this->rows_ = D.rows();
      this->cols_ = D.cols(); this->ld_ = D.ld(); return *this; }

    /**
     * Assignment operator, from a DenseMatrix. Assign the memory of
     * the DenseMatrix to the matrix wrapped by this
     * DenseMatrixWrapper object.
     *
     * \param a matrix to copy from, should be a.rows() ==
     * this->rows() and a.cols() == this->cols()
     */
    DenseMatrix<scalar_t>&
    operator=(const DenseMatrix<scalar_t>& a) override {
      assert(a.rows()==this->rows() && a.cols()==this->cols());
      for (std::size_t j=0; j<this->cols(); j++)
        for (std::size_t i=0; i<this->rows(); i++)
          this->operator()(i, j) = a(i, j);
      return *this;
    }
  };


  /**
   * Create a DenseMatrixWrapper for a const dense matrix.
   * TODO: we need to find a better way to handle this.
   *
   * \param m number of rows of the submatrix (the created wrapper),
   * the view in the matrix D
   * \param n number of columns of the submatrix
   * \param D pointer to a dense matrix. This will not be modified,
   * will not be freed.
   * \param ld leading dimension of D, ld >= m
   * \return unique_ptr with a const DenseMatrixWrapper
   */
  template<typename scalar_t>
  std::unique_ptr<const DenseMatrixWrapper<scalar_t>>
  ConstDenseMatrixWrapperPtr
  (std::size_t m, std::size_t n, const scalar_t* D, std::size_t ld) {
    return std::unique_ptr<const DenseMatrixWrapper<scalar_t>>
      (new DenseMatrixWrapper<scalar_t>(m, n, const_cast<scalar_t*>(D), ld));
  }

  /**
   * Create a DenseMatrixWrapper for a const dense matrix.
   * TODO: we need to find a better way to handle this.
   * Should have i+m <= D.rows() and j+n <= D.cols().
   *
   *
   * \param m number of rows of the submatrix (the created wrapper),
   * the view in the matrix D
   * \param n number of columns of the submatrix
   * \param D pointer to a dense matrix. This will not be modified,
   * will not be freed.
   * \param ld leading dimension of D
   * \param i row offset in the matrix D, denoting the top left corner
   * of the submatrix to be created
   * \param j column offset in the matrix D, denoting the top left corner
   * of the submatrix to be created
   * \return unique_ptr with a const DenseMatrixWrapper
   */
  template<typename scalar_t>
  std::unique_ptr<const DenseMatrixWrapper<scalar_t>>
  ConstDenseMatrixWrapperPtr
  (std::size_t m, std::size_t n, const DenseMatrix<scalar_t>& D,
   std::size_t i, std::size_t j) {
    return std::unique_ptr<const DenseMatrixWrapper<scalar_t>>
      (new DenseMatrixWrapper<scalar_t>
       (m, n, const_cast<DenseMatrix<scalar_t>&>(D), i, j));
  }


  /**
   * Copy submatrix of a at ia,ja of size m,n into b at position
   * ib,jb.  Should have ia+m <= a.rows(), ja+n <= a.cols(), ib+m <=
   * b.rows() and jb.n <= b.cols().
   *
   * \param m number of rows to copy
   * \param n number of columns to copy
   * \param a DenseMatrix to copy from
   * \param ia row offset of top left corner of submatrix of a to copy
   * \param ja column offset of top left corner of submatrix of a to
   * copy
   * \param b matrix to copy to
   * \param ib row offset of top left corner of place in b to copy to
   * \param jb column offset of top left corner of place in b to copy
   * to
   */
  template<typename scalar_t> void
  copy(std::size_t m, std::size_t n, const DenseMatrix<scalar_t>& a,
       std::size_t ia, std::size_t ja, DenseMatrix<scalar_t>& b,
       std::size_t ib, std::size_t jb) {
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        b(ib+i, jb+j) = a(ia+i, ja+j);
  }

  /**
   * Copy matrix a into matrix b at position ib, jb.  Should have
   * ib+a.rows() <= b.row() and jb+a.cols() <= b.cols().
   *
   * \param a matrix to copy
   * \param b matrix to copy to
   * \param ib row offset of top left corner of place in b to copy to
   * \param jb column offset of top left corner of place in b to copy
   * to
   */
  template<typename scalar_t> void
  copy(const DenseMatrix<scalar_t>& a, DenseMatrix<scalar_t>& b,
       std::size_t ib, std::size_t jb) {
    copy(a.rows(), a.cols(), a, 0, 0, b, ib, jb);
  }

  /**
   * Copy matrix a into matrix b. Should have ldb >= a.rows(). Matrix
   * b should have been allocated.
   *
   * \param a matrix to copy
   * \param b dense matrix to copy to
   * \param ldb leading dimension of b
   */
  template<typename scalar_t> void
  copy(const DenseMatrix<scalar_t>& a, scalar_t* b, std::size_t ldb) {
    for (std::size_t j=0; j<a.cols(); j++)
      for (std::size_t i=0; i<a.rows(); i++)
        b[i+j*ldb] = a(i, j);
  }


  /**
   * Vertically concatenate 2 DenseMatrix objects a and b: [a; b]. Should have
   * a.cols() == b.cols()
   *
   * \param a dense matrix, will be placed on top
   * \param b dense matrix, will be below
   * \return [a; b]
   */
  template<typename scalar_t> DenseMatrix<scalar_t>
  vconcat(const DenseMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b) {
    assert(a.cols() == b.cols());
    DenseMatrix<scalar_t> tmp(a.rows()+b.rows(), a.cols());
    copy(a, tmp, 0, 0);
    copy(b, tmp, a.rows(), 0);
    return tmp;
  }

  /**
   * Horizontally concatenate 2 DenseMatrix objects a and b: [a;
   * b]. Should have a.rows() == b.rows()
   *
   * \param a dense matrix, will be placed left
   * \param b dense matrix, will be placed right
   * \return [a b]
   */
  template<typename scalar_t> DenseMatrix<scalar_t>
  hconcat(const DenseMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b) {
    assert(a.rows() == b.rows());
    DenseMatrix<scalar_t> tmp(a.rows(), a.cols()+b.cols());
    copy(a, tmp, 0, 0);
    copy(b, tmp, 0, a.cols());
    return tmp;
  }

  /**
   * Create an identity matrix of size m x n, ie, 1 on the main
   * diagonal, zero everywhere else.
   *
   * \return DenseMatrix with rows()==m, cols()==n, operator()(i,i)==1
   * and operator()(i,j)==0 for i!=j.
   */
  template<typename scalar_t> DenseMatrix<scalar_t>
  eye(std::size_t m, std::size_t n) {
    DenseMatrix<scalar_t> I(m, n);
    I.eye();
    return I;
  }


  template<typename scalar_t> DenseMatrix<scalar_t>::DenseMatrix()
    : data_(nullptr), rows_(0), cols_(0), ld_(1) { }

  template<typename scalar_t> DenseMatrix<scalar_t>::DenseMatrix
  (std::size_t m, std::size_t n)
    : data_(new scalar_t[m*n]), rows_(m),
      cols_(n), ld_(std::max(std::size_t(1), m)) { }

  template<typename scalar_t> DenseMatrix<scalar_t>::DenseMatrix
  (std::size_t m, std::size_t n, const scalar_t* D, std::size_t ld)
    : data_(new scalar_t[m*n]), rows_(m), cols_(n),
      ld_(std::max(std::size_t(1), m)) {
    assert(ld >= m);
    for (std::size_t j=0; j<cols_; j++)
      for (std::size_t i=0; i<rows_; i++)
        operator()(i, j) = D[i+j*ld];
  }

  template<typename scalar_t> DenseMatrix<scalar_t>::DenseMatrix
  (std::size_t m, std::size_t n, const DenseMatrix<scalar_t>& D,
   std::size_t i, std::size_t j)
    : data_(new scalar_t[m*n]), rows_(m), cols_(n),
      ld_(std::max(std::size_t(1), m)) {
    for (std::size_t _j=0; _j<std::min(cols_, D.cols()-j); _j++)
      for (std::size_t _i=0; _i<std::min(rows_, D.rows()-i); _i++)
        operator()(_i, _j) = D(_i+i, _j+j);
  }

  template<typename scalar_t>
  DenseMatrix<scalar_t>::DenseMatrix(const DenseMatrix<scalar_t>& D)
    : data_(new scalar_t[D.rows()*D.cols()]), rows_(D.rows()),
      cols_(D.cols()), ld_(std::max(std::size_t(1), D.rows())) {
    for (std::size_t j=0; j<cols_; j++)
      for (std::size_t i=0; i<rows_; i++)
        operator()(i, j) = D(i, j);
  }

  template<typename scalar_t>
  DenseMatrix<scalar_t>::DenseMatrix(DenseMatrix<scalar_t>&& D)
    : data_(D.data()), rows_(D.rows()), cols_(D.cols()), ld_(D.ld()) {
    D.data_ = nullptr;
    D.rows_ = 0;
    D.cols_ = 0;
    D.ld_ = 1;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>::~DenseMatrix() {
    delete[] data_;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::operator=(const DenseMatrix<scalar_t>& D) {
    if (this == &D) return *this;
    if (rows_ != D.rows() || cols_ != D.cols()) {
      rows_ = D.rows();
      cols_ = D.cols();
      delete[] data_;
      data_ = new scalar_t[rows_*cols_];
      ld_ = std::max(std::size_t(1), rows_);
    }
    for (std::size_t j=0; j<cols_; j++)
      for (std::size_t i=0; i<rows_; i++)
        operator()(i,j) = D(i,j);
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::operator=(DenseMatrix<scalar_t>&& D) {
    rows_ = D.rows();
    cols_ = D.cols();
    ld_ = D.ld();
    delete[] data_;
    data_ = D.data();
    D.data_ = nullptr;
    return *this;
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::print(std::string name, bool all, int width) const {
    std::cout << name << " = [  % " << rows() << "x" << cols()
              << ", ld=" << ld() << ", norm=" << norm() << std::endl;
    if (all || (rows() <= 20 && cols() <= 32)) {
      for (std::size_t i=0; i<rows(); i++) {
        for (std::size_t j=0; j<cols(); j++)
          std::cout << std::setw(width) << operator()(i,j) << "  ";
        std::cout << std::endl;
      }
    } else std::cout << " ..." << std::endl;
    std::cout << "];" << std::endl << std::endl;
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::print_to_file
  (std::string name, std::string filename, int width) const {
    std::fstream fs(filename, std::fstream::out);
    fs << name << " = [  % " << rows() << "x" << cols()
       << ", ld=" << ld() << ", norm=" << norm() << std::endl;
    std::setprecision(16);
    for (std::size_t i=0; i<rows(); i++) {
      for (std::size_t j=0; j<cols(); j++)
        fs << std::setw(width) << operator()(i,j) << "  ";
      fs << std::endl;
    }
    fs << "];" << std::endl << std::endl;
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::random
  (random::RandomGeneratorBase<typename RealType<scalar_t>::
   value_type>& rgen) {
    TIMER_TIME(TaskType::RANDOM_GENERATE, 1, t_gen);
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<rows(); i++)
        operator()(i,j) = rgen.get();
    STRUMPACK_FLOPS(rgen.flops_per_prng()*cols()*rows());
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::random() {
    TIMER_TIME(TaskType::RANDOM_GENERATE, 1, t_gen);
    auto rgen = random::make_default_random_generator<real_t>();
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<rows(); i++)
        operator()(i,j) = rgen->get();
    STRUMPACK_FLOPS(rgen->flops_per_prng()*cols()*rows());
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::zero() {
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<rows(); i++)
        operator()(i,j) = scalar_t(0.);
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::fill(scalar_t v) {
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<rows(); i++)
        operator()(i,j) = v;
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::eye() {
    auto minmn = std::min(cols(), rows());
    for (std::size_t j=0; j<minmn; j++)
      for (std::size_t i=0; i<minmn; i++)
        operator()(i,j) = (i == j) ? scalar_t(1.) : scalar_t(0.);
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::clear() {
    rows_ = 0;
    cols_ = 0;
    ld_ = 1;
    delete[] data_;
    data_ = nullptr;
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::resize(std::size_t m, std::size_t n) {
    auto tmp = new scalar_t[m*n];
    for (std::size_t j=0; j<std::min(cols(),n); j++)
      for (std::size_t i=0; i<std::min(rows(),m); i++)
        tmp[i+j*m] = operator()(i,j);
    delete[] data_;
    data_ = tmp;
    ld_ = std::max(std::size_t(1), m);
    rows_ = m;
    cols_ = n;
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::hconcat(const DenseMatrix<scalar_t>& b) {
    assert(rows() == b.rows());
    auto my_cols = cols();
    resize(rows(), my_cols + b.cols());
    strumpack::copy(rows(), b.cols(), b, 0, 0, *this, 0, my_cols);
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::copy(const DenseMatrix<scalar_t>& B,
                              std::size_t i, std::size_t j) {
    assert(B.rows() >= rows()+i);
    assert(B.cols() >= cols()+j);
    for (std::size_t _j=0; _j<cols(); _j++)
      for (std::size_t _i=0; _i<rows(); _i++)
        operator()(_i,_j) = B(_i+i,_j+j);
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::copy(const scalar_t* B, std::size_t ldb) {
    assert(ldb >= rows());
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<rows(); i++)
        operator()(i,j) = B[i+j*ldb];
  }

  template<typename scalar_t> DenseMatrix<scalar_t>
  DenseMatrix<scalar_t>::transpose() const {
    DenseMatrix<scalar_t> tmp(cols(), rows());
    blas::omatcopy
      ('C', rows(), cols(), data(), ld(), tmp.data(), tmp.ld());
    return tmp;
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::transpose(DenseMatrix<scalar_t>& X) const {
    assert(rows() == X.cols() && cols() == X.rows());
    blas::omatcopy
      ('C', rows(), cols(), data(), ld(), X.data(), X.ld());
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::laswp(const std::vector<int>& P, bool fwd) {
    if (cols() != 0 && rows() != 0)
      blas::laswp(cols(), data(), ld(), 1, rows(), P.data(), fwd ? 1 : -1);
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::laswp(const int* P, bool fwd) {
    if (cols() != 0 && rows() != 0)
      blas::laswp(cols(), data(), ld(), 1, rows(), P, fwd ? 1 : -1);
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::lapmr(const std::vector<int>& P, bool fwd) {
    if (cols() != 0 && rows() != 0)
      blas::lapmr(fwd, rows(), cols(), data(), ld(), P.data());
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::lapmt(const std::vector<int>& P, bool fwd) {
    if (cols() != 0 && rows() != 0)
      blas::lapmt(fwd, rows(), cols(), data(), ld(), P.data());
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::extract_rows
  (const std::vector<std::size_t>& I, DenseMatrix<scalar_t>& B) const {
    auto d = I.size();
    assert(B.rows() == d);
    assert(cols() == B.cols());
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<d; i++) {
        assert(I[i] < rows());
        B(i, j) = operator()(I[i], j);
      }
  }

  template<typename scalar_t> DenseMatrix<scalar_t>
  DenseMatrix<scalar_t>::extract_rows
  (const std::vector<std::size_t>& I) const {
    DenseMatrix<scalar_t> B(I.size(), cols());
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<I.size(); i++) {
        assert(I[i] >= 0 && I[i] < rows());
        B(i, j) = operator()(I[i], j);
      }
    return B;
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::extract_cols
  (const std::vector<std::size_t>& I, DenseMatrix<scalar_t>& B) const {
    auto d = I.size();
    auto m = rows();
    assert(B.cols() == I.size());
    assert(m == B.rows());
    for (std::size_t j=0; j<d; j++)
      for (std::size_t i=0; i<m; i++) {
        assert(I[j] < cols());
        B(i, j) = operator()(i, I[j]);
      }
  }

  template<typename scalar_t> DenseMatrix<scalar_t>
  DenseMatrix<scalar_t>::extract_cols
  (const std::vector<std::size_t>& I) const {
    DenseMatrix<scalar_t> B(rows(), I.size());
    for (std::size_t j=0; j<I.size(); j++)
      for (std::size_t i=0; i<rows(); i++) {
        assert(I[i] >= 0 && I[j] < cols());
        B(i, j) = operator()(i, I[j]);
      }
    return B;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>
  DenseMatrix<scalar_t>::extract
  (const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J) const {
    DenseMatrix<scalar_t> B(I.size(), J.size());
    for (std::size_t j=0; j<J.size(); j++)
      for (std::size_t i=0; i<I.size(); i++) {
        assert(I[i] >= 0 && I[i] < rows());
        assert(J[j] >= 0 && J[j] < cols());
        B(i, j) = operator()(I[i], J[j]);
      }
    return B;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::scatter_rows_add
  (const std::vector<std::size_t>& I, const DenseMatrix<scalar_t>& B,
   int depth) {
    assert(I.size() == B.rows());
    assert(B.cols() == cols());
    const auto m = I.size();
    const auto n = cols();
#if defined(_OPENMP) && defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2) if(depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++) {
        assert(I[i] < rows());
        operator()(I[i], j) += B(i, j);
      }
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*cols()*I.size());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::add
  (const DenseMatrix<scalar_t>& B, int depth) {
    const auto m = rows();
    const auto n = cols();
#if defined(_OPENMP) && defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2) if(depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        operator()(i, j) += B(i,j);
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::sub
  (const DenseMatrix<scalar_t>& B, int depth) {
    const auto m = rows();
    const auto n = cols();
#if defined(_OPENMP) && defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2) if(depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        operator()(i, j) -= B(i, j);
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::scale(scalar_t alpha, int depth) {
    const auto m = rows();
    const auto n = cols();
#if defined(_OPENMP) && defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2) if(depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        operator()(i, j) *= alpha;
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::scaled_add
  (scalar_t alpha, const DenseMatrix<scalar_t>& B, int depth) {
    const auto m = rows();
    const auto n = cols();
#if defined(_OPENMP) && defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2) if(depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        operator()(i, j) += alpha * B(i, j);
    STRUMPACK_FLOPS((is_complex<scalar_t>()?8:2)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::scale_and_add
  (scalar_t alpha, const DenseMatrix<scalar_t>& B, int depth) {
    const auto m = rows();
    const auto n = cols();
#if defined(_OPENMP) && defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2) if(depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        operator()(i, j) = alpha * operator()(i, j) + B(i, j);
    STRUMPACK_FLOPS((is_complex<scalar_t>()?8:2)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::scale_rows
  (const std::vector<scalar_t>& D, int depth) {
    const auto m = rows();
    const auto n = cols();
#if defined(_OPENMP) && defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2) if(depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        operator()(i, j) *= D[i];
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::div_rows
  (const std::vector<scalar_t>& D, int depth) {
    const auto m = rows();
    const auto n = cols();
#if defined(_OPENMP) && defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2) if(depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        operator()(i, j) /= D[i];
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> typename RealType<scalar_t>::value_type
  DenseMatrix<scalar_t>::norm() const {
    return normF();
  }

  template<typename scalar_t> typename RealType<scalar_t>::value_type
  DenseMatrix<scalar_t>::norm1() const {
    return blas::lange('1', rows(), cols(), data(), ld());
  }

  template<typename scalar_t> typename RealType<scalar_t>::value_type
  DenseMatrix<scalar_t>::normI() const {
    return blas::lange('I', rows(), cols(), data(), ld());
  }

  template<typename scalar_t> typename RealType<scalar_t>::value_type
  DenseMatrix<scalar_t>::normF() const {
    return blas::lange('F', rows(), cols(), data(), ld());
  }

  template<typename scalar_t> std::vector<int>
  DenseMatrix<scalar_t>::LU(int depth) {
    std::vector<int> piv(rows());
    int info = 0;
    getrf_omp_task(rows(), cols(), data(), ld(), piv.data(), &info, depth);
    if (info) {
      std::cerr << "ERROR: LU factorization failed with info="
                << info << std::endl;
      exit(1);
    }
    return piv;
  }

  template<typename scalar_t> int
  DenseMatrix<scalar_t>::Cholesky(int depth) {
    assert(rows() == cols());
    // TODO openmp tasking ?
    int info = blas::potrf('L', rows(), data(), ld());
    if (info)
      std::cerr << "ERROR: Cholesky factorization failed with info="
                << info << std::endl;
    return info;
  }

  template<typename scalar_t> std::vector<int>
  DenseMatrix<scalar_t>::LDLt(int depth) {
    assert(rows() == cols());
    std::vector<int> piv(rows());
    int info = blas::sytrf('L', rows(), data(), ld(), piv.data());
    if (info)
      std::cerr << "ERROR: LDLt factorization failed with info="
                << info << std::endl;
    return piv;
  }

  // template<typename scalar_t> std::vector<int>
  // DenseMatrix<scalar_t>::LDLt_rook(int depth) {
  //   assert(rows() == cols());
  //   std::vector<int> piv(rows());
  //   int info = blas::sytrf_rook('L', rows(), data(), ld(), piv.data());
  //   if (info)
  //     std::cerr << "ERROR: LDLt_rook factorization failed with info="
  //               << info << std::endl;
  //   return piv;
  // }

  template<typename scalar_t> DenseMatrix<scalar_t>
  DenseMatrix<scalar_t>::solve
  (const DenseMatrix<scalar_t>& b,
   const std::vector<int>& piv, int depth) const {
    assert(b.rows() == rows());
    assert(piv.size() >= rows());
    int info = 0;
    DenseMatrix<scalar_t> x(b);
    if (!rows()) return x;
    getrs_omp_task
      (char(Trans::N), rows(), b.cols(), data(), ld(), piv.data(),
       x.data(), x.ld(), &info, depth);
    if (info) {
      std::cerr << "ERROR: LU solve failed with info=" << info << std::endl;
      exit(1);
    }
    return x;
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::solve_LU_in_place
  (DenseMatrix<scalar_t>& b, const std::vector<int>& piv, int depth) const {
    assert(b.rows() == rows());
    assert(piv.size() >= rows());
    int info = 0;
    if (!rows()) return;
    getrs_omp_task
      (char(Trans::N), rows(), b.cols(), data(), ld(), piv.data(),
       b.data(), b.ld(), &info, depth);
    if (info) {
      std::cerr << "ERROR: LU solve failed with info=" << info << std::endl;
      exit(1);
    }
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::solve_LDLt_in_place
  (DenseMatrix<scalar_t>& b, const std::vector<int>& piv, int depth) const {
    assert(b.rows() == rows());
    assert(piv.size() >= rows());
    if (!rows()) return;
    int info = blas::sytrs
      ('L', rows(), b.cols(), data(), ld(), piv.data(), b.data(), b.ld());
    if (info) {
      std::cerr << "ERROR: LDLt solve failed with info=" << info << std::endl;
      exit(1);
    }
  }

  // template<typename scalar_t> void
  // DenseMatrix<scalar_t>::solve_LDLt_rook_in_place
  // (DenseMatrix<scalar_t>& b, const std::vector<int>& piv, int depth) const {
  //   assert(b.rows() == rows());
  //   assert(piv.size() >= rows());
  //   if (!rows()) return;
  //   int info = blas::sytrs_rook
  //     ('L', rows(), b.cols(), data(), ld(), piv.data(), b.data(), b.ld());
  //   if (info) {
  //     std::cerr << "ERROR: LDLt_rook solve failed with info=" << info << std::endl;
  //     exit(1);
  //   }
  // }


  template<typename scalar_t> void DenseMatrix<scalar_t>::LQ
  (DenseMatrix<scalar_t>& L, DenseMatrix<scalar_t>& Q, int depth) const {
    auto minmn = std::min(rows(), cols());
    std::unique_ptr<scalar_t[]> tau(new scalar_t[minmn]);
    int info;
    DenseMatrix<scalar_t> tmp(std::max(rows(), cols()), cols(), *this, 0, 0);
    blas::gelqfmod(rows(), cols(), tmp.data(), tmp.ld(), tau.get(), &info, depth);
    if (info) {
      std::cerr << "ERROR: LQ factorization failed with info="
                << info << std::endl;
      exit(1);
    }
    L = DenseMatrix<scalar_t>(rows(), rows(), tmp, 0, 0); // copy to L
    auto sfmin = blas::lamch<real_t>('S');
    for (std::size_t i=0; i<minmn; i++)
      if (std::abs(L(i, i)) < sfmin) {
        std::cerr << "WARNING: small diagonal on L from LQ" << std::endl;
        break;
      }
    blas::xxglqmod(cols(), cols(), std::min(rows(), cols()),
                   tmp.data(), tmp.ld(), tau.get(), &info, depth);
    Q = DenseMatrix<scalar_t>(cols(), cols(), tmp, 0, 0); // generate Q
    if (info) {
      std::cerr << "ERROR: generation of Q from LQ failed with info="
                << info << std::endl;
      exit(1);
    }
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::orthogonalize
  (scalar_t& r_max, scalar_t& r_min, int depth) {
    if (!cols() || !rows()) return;
    TIMER_TIME(TaskType::QR, 1, t_qr);
    int minmn = std::min(rows(), cols());
    std::unique_ptr<scalar_t[]> tau(new scalar_t[minmn]);
    blas::geqrfmod(rows(), minmn, data(), ld(), tau.get(), depth);
    real_t Rmax = std::abs(operator()(0, 0));
    real_t Rmin = Rmax;
    for (int i=0; i<minmn; i++) {
      auto Rii = std::abs(operator()(i, i));
      Rmax = std::max(Rmax, Rii);
      Rmin = std::min(Rmin, Rii);
    }
    r_max = Rmax;
    r_min = Rmin;
    // TODO threading!!
    blas::xxgqr(rows(), minmn, minmn, data(), ld(), tau.get());
    if (cols() > rows()) {
      DenseMatrixWrapper<scalar_t> tmp
        (rows(), cols()-rows(), *this, 0, rows());
      tmp.zero();
    }
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::ID_row
  (DenseMatrix<scalar_t>& X, std::vector<int>& piv,
   std::vector<std::size_t>& ind, real_t rel_tol, real_t abs_tol,
   int max_rank, int depth) const {
    // TODO optimize by implementing by row directly, avoiding transposes
    TIMER_TIME(TaskType::HSS_SEQHQRINTERPOL, 1, t_hss_seq_hqr);
    DenseMatrix<scalar_t> Xt;
    transpose().ID_column(Xt, piv, ind, rel_tol, abs_tol, max_rank, depth);
    X = Xt.transpose();
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::ID_column
  (DenseMatrix<scalar_t>& X, std::vector<int>& piv,
   std::vector<std::size_t>& ind, real_t rel_tol, real_t abs_tol,
   int max_rank, int depth) {
    ID_column_GEQP3(X, piv, ind, rel_tol, abs_tol, max_rank, depth);
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::ID_column_GEQP3
  (DenseMatrix<scalar_t>& X, std::vector<int>& piv,
   std::vector<std::size_t>& ind, real_t rel_tol, real_t abs_tol,
   int max_rank, int depth) {
    int m = rows(), n = cols();
    std::unique_ptr<scalar_t[]> tau(new scalar_t[std::max(1,std::min(m, n))]);
    piv.resize(n);
    std::vector<int> iind(n);
    int rank = 0;
    // TODO make geqp3tol stop at max_rank
    if (m && n)
      blas::geqp3tol
        (m, n, data(), ld(), iind.data(), tau.get(),
         rank, rel_tol, abs_tol, depth);
    else std::iota(iind.begin(), iind.end(), 1);
    rank = std::min(rank, max_rank);
    for (int i=1; i<=n; i++) {
      int j = iind[i-1];
      assert(j-1 >= 0 && j-1 < int(iind.size()));
      while (j < i) j = iind[j-1];
      piv[i-1] = j;
    }
    ind.resize(rank);
    for (int i=0; i<rank; i++) ind[i] = iind[i]-1;
    trsm_omp_task('L', 'U', 'N', 'N', rank, n-rank, scalar_t(1.),
                  data(), ld(), ptr(0, rank), ld(), depth);
    X = DenseMatrix<scalar_t>(rank, n-rank, ptr(0, rank), ld());
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::low_rank
  (DenseMatrix<scalar_t>& U, DenseMatrix<scalar_t>& V,
   real_t rel_tol, real_t abs_tol, int max_rank, int depth) const {
    DenseMatrix<scalar_t> tmp(*this);
    int m = rows(), n = cols(), minmn = std::min(m, n);
    std::unique_ptr<scalar_t[]> tau(new scalar_t[minmn]);
    std::vector<int> ind(n);
    int rank;
    blas::geqp3tol
      (m, n, tmp.data(), tmp.ld(), ind.data(),
       tau.get(), rank, rel_tol, abs_tol, depth);
    std::vector<int> piv(n);
    for (int i=1; i<=n; i++) {
      int j = ind[i-1];
      assert(j-1 >= 0 && j-1 < int(ind.size()));
      while (j < i) j = ind[j-1];
      piv[i-1] = j;
    }
    V = DenseMatrix<scalar_t>(rank, cols(), tmp.ptr(0, 0), tmp.ld());
    for (int c=0; c<rank; c++)
      for (int r=c+1; r<rank; r++)
        V(r, c) = scalar_t(0.);
    V.lapmt(ind, false);
    blas::xxgqr(rows(), rank, rank, tmp.data(), tmp.ld(), tau.get());
    U = DenseMatrix<scalar_t>(rows(), rank, tmp.ptr(0, 0), tmp.ld());
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::shift(scalar_t sigma) {
    for (std::size_t i=0; i<std::min(cols(),rows()); i++)
      operator()(i, i) += sigma;
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*std::min(rows(),cols()));
  }

  template<typename scalar_t> std::vector<scalar_t>
  DenseMatrix<scalar_t>::singular_values() const {
    DenseMatrix tmp(*this);
    auto minmn = std::min(rows(), cols());
    std::vector<scalar_t> S(minmn);
    int info = blas::gesvd('N', 'N', rows(), cols(), tmp.data(), tmp.ld(),
                           S.data(), NULL, 1, NULL, 1);
    if (info)
      std::cout << "ERROR in gesvd: info = " << info << std::endl;
    return S;
  }

  template<typename scalar_t> int DenseMatrix<scalar_t>::syev
  (Jobz job, UpLo ul, std::vector<scalar_t>& lambda) {
    assert(rows() == cols());
    lambda.resize(rows());
    return blas::syev
      (char(job), char(ul), rows(), data(), ld(), lambda.data());
  }


  /**
   * GEMM, defined for DenseMatrix objects (or DenseMatrixWrapper).
   *
   * DGEMM  performs one of the matrix-matrix operations
   *
   *    C := alpha*op( A )*op( B ) + beta*C,
   *
   * where  op( X ) is one of
   *
   *    op( X ) = X   or   op( X ) = X**T,
   *
   * alpha and beta are scalars, and A, B and C are matrices, with op( A )
   * an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
   *
   * \param depth current OpenMP task recursion depth
   */
  template<typename scalar_t> void
  gemm(Trans ta, Trans tb, scalar_t alpha, const DenseMatrix<scalar_t>& a,
       const DenseMatrix<scalar_t>& b, scalar_t beta,
       DenseMatrix<scalar_t>& c, int depth=0) {
    assert((ta==Trans::N && a.rows()==c.rows()) ||
           (ta!=Trans::N && a.cols()==c.rows()));
    assert((tb==Trans::N && b.cols()==c.cols()) ||
           (tb!=Trans::N && b.rows()==c.cols()));
    assert((ta==Trans::N && tb==Trans::N && a.cols()==b.rows()) ||
           (ta!=Trans::N && tb==Trans::N && a.rows()==b.rows()) ||
           (ta==Trans::N && tb!=Trans::N && a.cols()==b.cols()) ||
           (ta!=Trans::N && tb!=Trans::N && a.rows()==b.cols()));
#if defined(_OPENMP)
    bool in_par = omp_in_parallel();
#else
    bool in_par = false;
#endif
    if (in_par)
      gemm_omp_task
        (char(ta), char(tb), c.rows(), c.cols(),
         (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(), a.ld(),
         b.data(), b.ld(), beta, c.data(), c.ld(), depth);
    else
      blas::gemm(char(ta), char(tb), c.rows(), c.cols(),
                 (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(), a.ld(),
                 b.data(), b.ld(), beta, c.data(), c.ld());
  }

  template<typename scalar_t> void
  gemm(Trans ta, Trans tb, scalar_t alpha, const DenseMatrix<scalar_t>& a,
       const scalar_t* b, int ldb, scalar_t beta,
       DenseMatrix<scalar_t>& c, int depth=0) {
    assert((ta==Trans::N && a.rows()==c.rows()) ||
           (ta!=Trans::N && a.cols()==c.rows()));
#if defined(_OPENMP)
    bool in_par = omp_in_parallel();
#else
    bool in_par = false;
#endif
    if (in_par)
      gemm_omp_task
        (char(ta), char(tb), c.rows(), c.cols(),
         (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(),
         a.ld(), b, ldb, beta, c.data(), c.ld(), depth);
    else
      blas::gemm
        (char(ta), char(tb), c.rows(), c.cols(),
         (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(),
         a.ld(), b, ldb, beta, c.data(), c.ld());
  }

  template<typename scalar_t> void
  gemm(Trans ta, Trans tb, scalar_t alpha, const DenseMatrix<scalar_t>& a,
       const DenseMatrix<scalar_t>& b, scalar_t beta,
       scalar_t* c, int ldc, int depth=0) {
    assert((ta==Trans::N && tb==Trans::N && a.cols()==b.rows()) ||
           (ta!=Trans::N && tb==Trans::N && a.rows()==b.rows()) ||
           (ta==Trans::N && tb!=Trans::N && a.cols()==b.cols()) ||
           (ta!=Trans::N && tb!=Trans::N && a.rows()==b.cols()));
#if defined(_OPENMP)
    bool in_par = omp_in_parallel();
#else
    bool in_par = false;
#endif
    if (in_par)
      gemm_omp_task
        (char(ta), char(tb), (ta==Trans::N) ? a.rows() : a.cols(),
         (tb==Trans::N) ? b.cols() : b.rows(),
         (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(), a.ld(),
         b.data(), b.ld(), beta, c, ldc, depth);
    else
      blas::gemm
        (char(ta), char(tb), (ta==Trans::N) ? a.rows() : a.cols(),
         (tb==Trans::N) ? b.cols() : b.rows(),
         (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(), a.ld(),
         b.data(), b.ld(), beta, c, ldc);
  }

  /**
   * TRMM performs one of the matrix-matrix operations
   *
   * B := alpha*op(A)*B,   or   B := alpha*B*op(A),
   *
   *  where alpha is a scalar, B is an m by n matrix, A is a unit, or
   *  non-unit, upper or lower triangular matrix and op( A ) is one of
   *    op( A ) = A   or   op( A ) = A**T.
   */
  template<typename scalar_t> void
  trmm(Side s, UpLo ul, Trans ta, Diag d, scalar_t alpha,
       const DenseMatrix<scalar_t>& a, DenseMatrix<scalar_t>& b,
       int depth=0) {
    bool in_par = false;
#if defined(_OPENMP)
    in_par = omp_in_parallel();
#endif
    if (in_par)
      trmm_omp_task
        (char(s), char(ul), char(ta), char(d), b.rows(), b.cols(),
         alpha, a.data(), a.ld(), b.data(), b.ld(), depth);
    else
      blas::trmm
        (char(s), char(ul), char(ta), char(d), b.rows(), b.cols(),
         alpha, a.data(), a.ld(), b.data(), b.ld());
  }

  /**
   * DTRSM solves one of the matrix equations
   *
   * op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
   *
   *  where alpha is a scalar, X and B are m by n matrices, A is a
   *  unit, or non-unit, upper or lower triangular matrix and op( A )
   *  is one of
   *
   *    op( A ) = A   or   op( A ) = A**T.
   *
   * The matrix X is overwritten on B.
   */
  template<typename scalar_t> void
  trsm(Side s, UpLo ul, Trans ta, Diag d, scalar_t alpha,
       const DenseMatrix<scalar_t>& a, DenseMatrix<scalar_t>& b,
       int depth=0) {
#if defined(_OPENMP)
    bool in_par = omp_in_parallel();
#else
    bool in_par = false;
#endif
    if (in_par)
      trsm_omp_task
        (char(s), char(ul), char(ta), char(d), b.rows(), b.cols(),
         alpha, a.data(), a.ld(), b.data(), b.ld(), depth);
    else
      blas::trsm(char(s), char(ul), char(ta), char(d), b.rows(), b.cols(),
                 alpha, a.data(), a.ld(), b.data(), b.ld());
  }

  /**
   * DTRSV  solves one of the systems of equations
   *
   *    A*x = b,   or   A**T*x = b,
   *
   *  where b and x are n element vectors and A is an n by n unit, or
   *  non-unit, upper or lower triangular matrix.
   */
  template<typename scalar_t> void
  trsv(UpLo ul, Trans ta, Diag d, const DenseMatrix<scalar_t>& a,
       DenseMatrix<scalar_t>& b, int depth=0) {
    assert(b.cols() == 1);
    assert(a.rows() == a.cols() && a.cols() == b.rows());
#if defined(_OPENMP)
    bool in_par = omp_in_parallel();
#else
    bool in_par = false;
#endif
    if (in_par)
      trsv_omp_task(char(ul), char(ta), char(d), a.rows(),
                    a.data(), a.ld(), b.data(), 1, depth);
    else
      blas::trsv(char(ul), char(ta), char(d), a.rows(),
                 a.data(), a.ld(), b.data(), 1);
  }

  /**
   * DGEMV performs one of the matrix-vector operations
   *
   *    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
   *
   * where alpha and beta are scalars, x and y are vectors and A is an
   * m by n matrix.
   */
  template<typename scalar_t> void
  gemv(Trans ta, scalar_t alpha, const DenseMatrix<scalar_t>& a,
       const DenseMatrix<scalar_t>& x, scalar_t beta,
       DenseMatrix<scalar_t>& y, int depth=0) {
    assert(x.cols() == 1);
    assert(y.cols() == 1);
    assert(ta != Trans::N || (a.rows() == y.rows() && a.cols() == x.rows()));
    assert(ta == Trans::N || (a.cols() == y.rows() && a.rows() == x.rows()));
#if defined(_OPENMP)
    bool in_par = omp_in_parallel();
#else
    bool in_par = false;
#endif
    if (in_par)
      gemv_omp_task
        (char(ta), a.rows(), a.cols(), alpha, a.data(), a.ld(),
         x.data(), 1, beta, y.data(), 1, depth);
    else
      blas::gemv(char(ta), a.rows(), a.cols(), alpha, a.data(), a.ld(),
                 x.data(), 1, beta, y.data(), 1);
  }

  /**
   * DGEMV performs one of the matrix-vector operations
   *
   *    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
   *
   * where alpha and beta are scalars, x and y are vectors and A is an
   * m by n matrix.
   */
  template<typename scalar_t> void
  gemv(Trans ta, scalar_t alpha, const DenseMatrix<scalar_t>& a,
       const scalar_t* x, int incx, scalar_t beta,
       DenseMatrix<scalar_t>& y, int depth=0) {
    assert(y.cols() == 1);
    assert(ta != Trans::N || (a.rows() == y.rows()));
    assert(ta == Trans::N || (a.cols() == y.rows()));
#if defined(_OPENMP)
    bool in_par = omp_in_parallel();
#else
    bool in_par = false;
#endif
    if (in_par)
      gemv_omp_task
        (char(ta), a.rows(), a.cols(), alpha, a.data(), a.ld(),
         x, incx, beta, y.data(), 1, depth);
    else
      blas::gemv(char(ta), a.rows(), a.cols(), alpha, a.data(), a.ld(),
                 x, incx, beta, y.data(), 1);
 }

  /**
   * DGEMV performs one of the matrix-vector operations
   *
   *    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
   *
   * where alpha and beta are scalars, x and y are vectors and A is an
   * m by n matrix.
   */
  template<typename scalar_t> void
  gemv(Trans ta, scalar_t alpha, const DenseMatrix<scalar_t>& a,
       const DenseMatrix<scalar_t>& x, scalar_t beta,
       scalar_t* y, int incy, int depth=0) {
    assert(x.cols() == 1);
    assert(ta != Trans::N || (a.cols() == x.rows()));
    assert(ta == Trans::N || (a.rows() == x.rows()));
#if defined(_OPENMP)
    bool in_par = omp_in_parallel();
#else
    bool in_par = false;
#endif
    if (in_par)
      gemv_omp_task
        (char(ta), a.rows(), a.cols(), alpha, a.data(), a.ld(),
         x.data(), 1, beta, y, incy, depth);
    else
      blas::gemv(char(ta), a.rows(), a.cols(), alpha, a.data(), a.ld(),
                 x.data(), 1, beta, y, incy);
  }

  /**
   * DGEMV performs one of the matrix-vector operations
   *
   *    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
   *
   * where alpha and beta are scalars, x and y are vectors and A is an
   * m by n matrix.
   */
  template<typename scalar_t> void
  gemv(Trans ta, scalar_t alpha, const DenseMatrix<scalar_t>& a,
       const scalar_t* x, int incx, scalar_t beta,
       scalar_t* y, int incy, int depth=0) {
#if defined(_OPENMP)
    bool in_par = omp_in_parallel();
#else
    bool in_par = false;
#endif
    if (in_par)
      gemv_omp_task
        (char(ta), a.rows(), a.cols(), alpha, a.data(), a.ld(),
         x, incx, beta, y, incy, depth);
    else
      blas::gemv(char(ta), a.rows(), a.cols(), alpha, a.data(), a.ld(),
                 x, incx, beta, y, incy);
  }


  /** return number of flops for LU factorization */
  template<typename scalar_t> long long int
  LU_flops(const DenseMatrix<scalar_t>& a) {
    return (is_complex<scalar_t>() ? 4:1) *
      blas::getrf_flops(a.rows(), a.cols());
  }

  /** return number of flops for solve, using LU factorization */
  template<typename scalar_t> long long int
  solve_flops(const DenseMatrix<scalar_t>& b) {
    return (is_complex<scalar_t>() ? 4:1) *
      blas::getrs_flops(b.rows(), b.cols());
  }

  /** return number of flops for LQ factorization */
  template<typename scalar_t> long long int
  LQ_flops(const DenseMatrix<scalar_t>& a) {
    auto minrc = std::min(a.rows(), a.cols());
    return (is_complex<scalar_t>() ? 4:1) *
      (blas::gelqf_flops(a.rows(), a.cols()) +
       blas::xxglq_flops(a.cols(), a.cols(), minrc));
  }

  /** return number of flops for interpolative decomposition */
  template<typename scalar_t> long long int
  ID_row_flops(const DenseMatrix<scalar_t>& a, int rank) {
    return (is_complex<scalar_t>() ? 4:1) *
      (blas::geqp3_flops(a.cols(), a.rows()) +
       blas::trsm_flops(rank, a.cols() - rank, scalar_t(1.), 'L'));
  }

  /** return number of flops for a trsm */
  template<typename scalar_t> long long int
  trsm_flops(Side s, scalar_t alpha, const DenseMatrix<scalar_t>& a,
             const DenseMatrix<scalar_t>& b) {
    return (is_complex<scalar_t>() ? 4:1) *
      blas::trsm_flops(b.rows(), b.cols(), alpha, char(s));
  }

  /** return number of flops for a gemm, given a and b */
  template<typename scalar_t> long long int
  gemm_flops(Trans ta, Trans tb, scalar_t alpha,
             const DenseMatrix<scalar_t>& a,
             const DenseMatrix<scalar_t>& b, scalar_t beta) {
    return (is_complex<scalar_t>() ? 4:1) *
      blas::gemm_flops
      ((ta==Trans::N) ? a.rows() : a.cols(),
       (tb==Trans::N) ? b.cols() : b.rows(),
       (ta==Trans::N) ? a.cols() : a.rows(), alpha, beta);
  }

  /** return number of flops for a gemm, given a and c */
  template<typename scalar_t> long long int
  gemm_flops(Trans ta, Trans tb, scalar_t alpha,
             const DenseMatrix<scalar_t>& a, scalar_t beta,
             const DenseMatrix<scalar_t>& c) {
    return (is_complex<scalar_t>() ? 4:1) *
      blas::gemm_flops
      (c.rows(), c.cols(), (ta==Trans::N) ? a.cols() : a.rows(), alpha, beta);
  }

  /** return number of flops for orthogonalization */
  template<typename scalar_t> long long int
  orthogonalize_flops(const DenseMatrix<scalar_t>& a) {
    auto minrc = std::min(a.rows(), a.cols());
    return (is_complex<scalar_t>() ? 4:1) *
      (blas::geqrf_flops(a.rows(), minrc) +
       blas::xxgqr_flops(a.rows(), minrc, minrc));
  }

} // end namespace strumpack

#endif // DENSE_MATRIX_HPP
