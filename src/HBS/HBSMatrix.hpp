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
/**
 * \file HBSMatrix.hpp
 *
 * \brief This file contains the HBSMatrix class definition as well as
 * implementations for a number of it's member routines. Other member
 * routines are implemented in files such as HBSMatrix.apply.hpp,
 * HBSMatrix.factor.hpp etc.
 */
#ifndef HBS_MATRIX_HPP
#define HBS_MATRIX_HPP

#include <cassert>
#include <functional>
#include <string>

#include "HBSOptions.hpp"
#include "structured/StructuredMatrix.hpp"

namespace strumpack {
  namespace HBS {

    template<typename scalar_t> class WorkCompress;

    /**
     * Enumeration of possible states of an HSS matrix/node. This is
     * used in the adaptive HSS compression algorithms, where a node
     * can be untouched (it is not yet visited by the compression
     * algorithm), partially_compressed (a compression was attempted
     * but failed, so the adaptive algorithm will have to try again),
     * or can be successfully compressed.
     * \ingroup Enumerations
     */
    enum class State : char
      {UNTOUCHED='U',   /*!< Node was not yet visited by the
                           compression algorithm */
       PARTIALLY_COMPRESSED='P', /*!< Compression was attempted for
                                    this node, but failed. The
                                    adaptive compression should try
                                    again. */
       COMPRESSED='C'   /*!< This HSS node was succesfully
                           compressed. */
      };

    /**
     * \class HBSMatrix
     *
     * \brief Class to represent a sequential/threaded Hierarchically
     * Block-Separable matrix.
     *
     * This is for non-symmetric matrices, but can be used with
     * symmetric matrices as well. An HBS matrix is represented
     * recursively, using 2 children which are also HBSMatrix objects,
     * except at the lowest level (the leafs), where the HBSMatrix has
     * no children. Hence, the tree representing the HBS matrix is
     * always a binary tree, but not necessarily a complete binary
     * tree.
     *
     * \tparam scalar_t Can be float, double, std:complex<float> or
     * std::complex<double>.
     *
     */
    template<typename scalar_t> class HBSMatrix
      : public structured::StructuredMatrix<scalar_t> {
      using real_t = typename RealType<scalar_t>::value_type;
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      // using elem_t = typename std::function
      //   <void(const std::vector<std::size_t>& I,
      //         const std::vector<std::size_t>& J, DenseM_t& B)>;
      using mult_t = typename std::function
        <void(DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc)>;
      using opts_t = HBSOptions<scalar_t>;

    public:
      /**
       * Default constructor, constructs an empty 0 x 0 matrix.
       */
      HBSMatrix();

      /**
       * Construct an HBS representation for a dense matrix A. The HBS
       * tree will be constructed by splitting the row/column set
       * evenly and recursively until the leafs in the HBS tree are
       * smaller than opts.leaf_size(). Alternative constructors can
       * be used to specify a specific HBS partitioning
       * tree. Internally, this will call the appropriate compress
       * routine to construct the HBS representation, using the
       * options (such as compression tolerance, adaptive compression
       * scheme etc) as specified in the HBSOptions opts object.
       *
       * \param A dense matrix (unmodified) to compress as HBS
       * \param opts object containing a number of options for HBS
       * compression
       * \see DenseMatrix
       * \see HBSOptions
       */
      HBSMatrix(const DenseM_t& A, const opts_t& opts);

      /**
       * Construct an HBS representation for an m x n matrix. The HBS
       * tree will be constructed by splitting the row/column set
       * evenly and recursively until the leafs in the HBS tree are
       * smaller than opts.leaf_size(). After construction, the HBS
       * matrix will be empty, and can be filled by calling one of the
       * compress member routines. Alternative constructors can be
       * used to specify a specific HBS partitioning tree.
       *
       * \param m number of rows in the constructed HBS matrix
       * \param n number of rows in the constructed HBS matrix
       * \param opts object containing a number of options for HBS
       * compression
       * \see compress, HBSOptions
       */
      HBSMatrix(std::size_t m, std::size_t n, const opts_t& opts);

      /**
       * Construct an HBS matrix using a specified HBS tree. After
       * construction, the HBS matrix will be empty, and can be filled
       * by calling one of the compress member routines.
       *
       * \param t tree specifying the HBS matrix partitioning
       * \param opts object containing a number of options for HBS
       * compression
       * \see compress, HBSOptions
       */
      HBSMatrix(const structured::ClusterTree& t, const opts_t& opts);

      /**
       * Construct an HBS approximation for the kernel matrix K.
       *
       * \param K Kernel matrix object. The data associated with this
       * kernel will be permuted according to the clustering algorithm
       * selected by the HBSOptions objects. The permutation will be
       * stored in the kernel object.
       * \param opts object containing a number of HBS options
       */
      //HBSMatrix(kernel::Kernel<real_t>& K, const opts_t& opts);

      /**
       * Copy constructor. Copying an HBSMatrix can be an expensive
       * operation.
       * \param other HBS matrix to be copied
       */
      HBSMatrix(const HBSMatrix<scalar_t>& other);

      /**
       * Copy assignment operator. Copying an HBSMatrix can be an
       * expensive operation.
       * \param other HBS matrix to be copied
       */
      HBSMatrix<scalar_t>& operator=(const HBSMatrix<scalar_t>& other);

      /**
       * Move constructor.
       * \param other HBS matrix to be moved from, will be emptied
       */
      HBSMatrix(HBSMatrix<scalar_t>&& other) = default;

      /**
       * Move assignment operator.
       * \param other HBS matrix to be moved from, will be emptied
       */
      HBSMatrix<scalar_t>& operator=(HBSMatrix<scalar_t>&& other) = default;

      /**
       * Create a clone of this matrix.
       * TODO remove this!!???
       */
      //std::unique_ptr<HBSMatrixBase<scalar_t>> clone() const override;

      /**
       * Return a const raw (non-owning) pointer to child c of this
       * HBS matrix. A child of an HBS matrix is itself an HBS
       * matrix. The value of c should be 0 or 1, and this HBS matrix
       * should not be a leaf!
       */
      const HBSMatrix<scalar_t>* child(int c) const {
        return dynamic_cast<HBSMatrix<scalar_t>*>(this->ch_[c].get());
      }

      /**
       * Return a raw (non-owning) pointer to child c of this HBS
       * matrix. A child of an HBS matrix is itself an HBS matrix. The
       * value of c should be 0 or 1, and this HBS matrix should not
       * be a leaf!
       */
      HBSMatrix<scalar_t>* child(int c) {
        return dynamic_cast<HBSMatrix<scalar_t>*>(this->ch_[c].get());
      }
      // /**
      //  * Return a const reference to the child (0, or 1) of this HBS
      //  * matrix. This is only valid when !this->leaf(). It is assumed
      //  * that a non-leaf node always has exactly 2 children.
      //  *
      //  * \param c Number of the child, should be 0 or 1, for the left
      //  * or the right child.
      //  * \return Const reference to the child (HBSMatrix).
      //  */
      // const HBSMatrix<scalar_t>& child(int c) const {
      //   assert(c>=0 && c<int(ch_.size())); return *(ch_[c]);
      // }

      // /**
      //  * Return a reference to the child (0, or 1) of this HBS
      //  * matrix. This is only valid when !this->leaf(). It is assumed
      //  * that a non-leaf node always has exactly 2 children.
      //  *
      //  * \param c Number of the child, should be 0 or 1, for the left
      //  * or the right child.
      //  * \return Reference to the child (HBSMatrix).
      //  */
      // HBSMatrix<scalar_t>& child(int c) {
      //   assert(c>=0 && c<int(ch_.size())); return *(ch_[c]);
      // }

      /**
       * Returns the dimensions of this HBS matrix, as a pair.
       *
       * \return pair with number of rows and columns of this HBS
       * matrix.
       */
      std::pair<std::size_t,std::size_t> dims() const {
        return std::make_pair(rows_, cols_);
      }

      /**
       * Return the number of rows in this HBS matrix.
       * \return number of rows
       */
      std::size_t rows() const override { return rows_; }

      /**
       * Return the number of columns in this HBS matrix.
       * \return number of columns
       */
      std::size_t cols() const override { return cols_; }

      /**
       * Check whether this node of the HBS tree is a leaf.
       * \return true if this node is a leaf, false otherwise.
       */
      bool leaf() const { return ch_.empty(); }

      std::size_t factor_nonzeros() const;

      /**
       * Check whether the HBS matrix was compressed.
       *
       * \return True if this HBS matrix was succesfully compressed,
       * false otherwise.
       *
       * \see is_untouched
       */
      bool is_compressed() const {
        return U_state_ == State::COMPRESSED &&
          V_state_ == State::COMPRESSED;
      }

      /**
       * Check whether the HBS compression was started for this
       * matrix.
       *
       * \return True if HBS compression was not started yet, false
       * otherwise. False may mean that compression was started but
       * failed, or that compression succeeded.
       *
       * \see is_compressed
       */
      bool is_untouched() const {
        return U_state_ == State::UNTOUCHED &&
          V_state_ == State::UNTOUCHED;
      }

      /**
       * Check if this HBS matrix (or node in the HBS tree) is active
       * on this rank.
       *
       * \return True if this node is active, false otherwise.
       */
      bool active() const { return active_; }

      /**
       * Print info about this HBS matrix, such as tree info, ranks,
       * etc.
       *
       * \param out Stream to print to, defaults to std::cout
       * \param roff Row offset of top left corner, defaults to
       * 0. This is used to recursively print the tree, you can leave
       * this at the default.
       * \param coff Column offset of top left corner, defaults to
       * 0. This is used to recursively print the tree, you can leave
       * this at the default.
       */
      void print_info(std::ostream &out=std::cout,
                      std::size_t roff=0,
                      std::size_t coff=0) const;

      /**
       * Set the depth of openmp nested tasks. This can be used to
       * limit the number of tasks to spawn in the HBS routines, which
       * is can reduce task creation overhead.  This is used in the
       * sparse solver when multiple HBS matrices are created from
       * within multiple openmp tasks. The HBS routines all use openmp
       * tasking to traverse the HBS tree and for parallelism within
       * the HBS nodes as well.
       */
      void set_openmp_task_depth(int depth) { openmp_task_depth_ = depth; }


      /**
       * Initialize this HBS matrix as the compressed HBS
       * representation of a given dense matrix. The HBS matrix should
       * have been constructed with the proper sizes, i.e., rows() ==
       * A.rows() and cols() == A.cols().  Internaly, this will call
       * the appropriate compress routine to construct the HBS
       * representation, using the options (such as compression
       * tolerance, adaptive compression scheme etc) as specified in
       * the HBSOptions opts object.
       *
       * \param A dense matrix (unmodified) to compress as HBS
       * \param opts object containing a number of options for HBS
       * compression
       * \see DenseMatrix
       * \see HBSOptions
       */
      void compress(const DenseM_t& A, const opts_t& opts);

      /**
       * Initialize this HBS matrix as the compressed HBS
       * representation. The compression uses the
       * matrix-(multiple)vector multiplication routine Amult,
       * provided by the user. The HBS matrix should have been
       * constructed with the proper sizes, i.e., rows() == A.rows()
       * and cols() == A.cols().  Internaly, this will call the
       * appropriate compress routine to construct the HBS
       * representation, using the options (such as compression
       * tolerance, adaptive compression scheme etc) as specified in
       * the HBSOptions opts object.
       *
       * \param Amult matrix-(multiple)vector product routine. This
       * can be a functor, or a lambda function for instance.
       * \param Rr Parameter to the matvec routine. Random
       * matrix. This will be set by the compression routine.
       * \param Rc Parameter to the matvec routine. Random
       * matrix. This will be set by the compression routine.
       * \param Sr Parameter to the matvec routine. Random sample
       * matrix, to be computed by the matrix-(multiple)vector
       * multiplication routine as A*Rr. This will aready be allocated
       * by the compression routine and should be Sr.rows() ==
       * this->rows() and Sr.cols() == Rr.cols().
       * \param Sc random sample matrix, to be computed by the
       * matrix-(multiple)vector multiplication routine as A^T*Rc, or
       * A^C*Rc. This will aready be allocated by the compression
       * routine and should be Sc.rows() == this->cols() and Sc.cols()
       * == Rc.cols().
       *
       * \param opts object containing a number of options for HBS
       * compression
       * \see DenseMatrix
       * \see HBSOptions
       */
      void compress(const std::function<void(DenseM_t& Rr,
                                             DenseM_t& Rc,
                                             DenseM_t& Sr,
                                             DenseM_t& Sc)>& Amult,
                    const opts_t& opts);

      /**
       * Reset the matrix to an empty, 0 x 0 matrix, freeing up all
       * it's memory.
       */
      void reset();

      /**
       * Compute a ULV factorization of this matrix.
       */
      void factor() override {}

      /**
       * Compute a partial ULV factorization of this matrix. Only the
       * left child is factored. This is not similar to calling
       * child(0)->factor(), except that the HBSFactors resulting from
       * calling partial_factor can be used to compute the Schur
       * complement.
       *
       * \see Schur_update, Schur_product_direct and
       * Schur_product_indirect
       */
      void partial_factor() {}

      /**
       * Solve a linear system with the ULV factorization of this
       * HBSMatrix. The right hand side vector (or matrix) b is
       * overwritten with the solution x of Ax=b.
       *
       * \param ULV ULV factorization of this matrix
       * \param b on input, the right hand side vector, on output the
       * solution of A x = b (with A this HBS matrix). The vector b
       * should be b.rows() == cols().
       * \see factor
       */
      void solve(DenseM_t& b) const override {}

      /**
       * Perform only the forward phase of the ULV linear solve. This
       * is for advanced use only, typically to be used in combination
       * with partial_factor. You should really just use factor/solve
       * when possible.
       *
       * \param w temporary working storage, to pass information from
       * forward_solve to backward_solve
       * \param b on input, the right hand side vector, on output the
       * intermediate solution. The vector b
       * should be b.rows() == cols().
       * \param partial denotes wether the matrix was fully or
       * partially factored
       * \see factor, partial_factor, backward_solve
       */
      // void forward_solve(WorkSolve<scalar_t>& w, const DenseM_t& b,
      //                    bool partial) const override {}

      /**
       * Perform only the backward phase of the ULV linear solve. This
       * is for advanced use only, typically to be used in combination
       * with partial_factor. You should really just use factor/solve
       * when possible.
       *
       * \param w temporary working storage, to pass information from
       * forward_solve to backward_solve
       * \param b on input, the vector obtained from forward_solve, on
       * output the solution of A x = b (with A this HBS matrix). The
       * vector b should be b.rows() == cols().
       * \see factor, partial_factor, backward_solve
       */
      // void backward_solve(WorkSolve<scalar_t>& w, DenseM_t& x) const override {}

      /**
       * Multiply this HBS matrix with a dense matrix (vector), ie,
       * compute x = this * b.
       *
       * \param b Matrix to multiply with, from the left.
       * \return The result of this * b.
       * \see applyC, HBS::apply_HBS
       */
      DenseM_t apply(const DenseM_t& b) const { return DenseM_t(); }

      /**
       * Multiply this HBS matrix with a dense matrix (vector), ie,
       * compute y = op(this) * x. Overrides from the StructuredMatrix
       * class method.
       *
       * \param op Transpose or complex conjugate
       * \param x right hand side matrix to multiply with, from the
       * left, rows(x) == cols(op(this))
       * \param y result of op(this) * b, cols(y) == cols(x), rows(r)
       * = rows(op(this))
       * \see applyC, HBS::apply_HBS
       */
      void mult(Trans op, const DenseM_t& x, DenseM_t& y) const override {}

      /**
       * Multiply the transpose or complex conjugate of this HBS
       * matrix with a dense matrix (vector), ie, compute x = this^C *
       * b.
       *
       * \param b Matrix to multiply with, from the left.
       * \return The result of this^C * b.
       * \see apply, HBS::apply_HBS
       */
      DenseM_t applyC(const DenseM_t& b) const { return DenseM_t(); }

      /**
       * Extract a single element this(i,j) from this HBS matrix. This
       * is expensive and should not be used to compute multiple
       * elements.
       *
       * \param i Global row index of element to compute.
       * \param j Global column index of element to compute.
       * \return this(i,j)
       * \see extract
       */
      scalar_t get(std::size_t i, std::size_t j) const { return 0.; }

      /**
       * Compute a submatrix this(I,J) from this HBS matrix.
       *
       * \param I set of row indices of elements to compute from this
       * HBS matrix.
       * \param I set of column indices of elements to compute from
       * this HBS matrix.
       * \return Submatrix from this HBS matrix this(I,J)
       * \see get, extract_add
       */
      DenseM_t extract(const std::vector<std::size_t>& I,
                       const std::vector<std::size_t>& J) const { return DenseM_t(); }

      /**
       * Compute a submatrix this(I,J) from this HBS matrix and add it
       * to a given matrix B.
       *
       * \param I set of row indices of elements to compute from this
       * HBS matrix.
       * \param I set of column indices of elements to compute from
       * this HBS matrix.
       * \param The extracted submatrix this(I,J) will be added to
       * this matrix. Should satisfy B.rows() == I.size() and B.cols()
       * == J.size().
       * \see get, extract
       */
      void extract_add(const std::vector<std::size_t>& I,
                       const std::vector<std::size_t>& J,
                       DenseM_t& B) const;

// #ifndef DOXYGEN_SHOULD_SKIP_THIS
//       void Schur_update(DenseM_t& Theta,
//                         DenseM_t& DUB01,
//                         DenseM_t& Phi) const;
//       void Schur_product_direct(const DenseM_t& Theta,
//                                 const DenseM_t& DUB01,
//                                 const DenseM_t& Phi,
//                                 const DenseM_t&_ThetaVhatC_or_VhatCPhiC,
//                                 const DenseM_t& R,
//                                 DenseM_t& Sr, DenseM_t& Sc) const;
//       void Schur_product_indirect(const DenseM_t& DUB01,
//                                   const DenseM_t& R1,
//                                   const DenseM_t& R2, const DenseM_t& Sr2,
//                                   const DenseM_t& Sc2,
//                                   DenseM_t& Sr, DenseM_t& Sc) const;
//       void delete_trailing_block() override;
// #endif // DOXYGEN_SHOULD_SKIP_THIS

      std::size_t rank() const override;
      std::size_t memory() const override;
      std::size_t nonzeros() const override;
      std::size_t levels() const;

      /**
       * Return a full/dense representation of this HBS matrix.
       *
       * \return Dense matrix obtained from multiplying out the
       * low-rank representation.
       */
      DenseM_t dense() const;

      void shift(scalar_t sigma) override;

      // void draw(std::ostream& of,
      //           std::size_t rlo=0, std::size_t clo=0) const override;

      /**
       * Write this HBSMatrix<scalar_t> to a binary file, called
       * fname.
       *
       * \see read
       */
      // void write(const std::string& fname) const;

      /**
       * Read an HBSMatrix<scalar_t> from a binary file, called
       * fname.
       *
       * \see write
       */
      // static HBSMatrix<scalar_t> read(const std::string& fname);

      // const HBSFactors<scalar_t>& ULV() { return this->ULV_; }

    protected:
      HBSMatrix(std::size_t m, std::size_t n, bool active);
      HBSMatrix(std::size_t m, std::size_t n,
                const opts_t& opts, bool active);
      HBSMatrix(const structured::ClusterTree& t,
                const opts_t& opts, bool active);

      std::size_t rows_, cols_;
      std::vector<std::unique_ptr<HBSMatrix<scalar_t>>> ch_;
      DenseM_t U_, V_, D_, B01_, B10_;

      State U_state_, V_state_;
      int openmp_task_depth_;
      bool active_;

      void compress_recursive(DenseM_t& Rr, DenseM_t& Rc,
                              DenseM_t& Sr, DenseM_t& Sc,
                              const opts_t& opts,
                              WorkCompress<scalar_t>& w,
                              int r, int depth);
    };

    /**
     * Compute C = op(A) * B + beta * C, with HBS matrix A.
     *
     * \param op Transpose/complex conjugate or none to be applied to
     * the HBS matrix A.
     * \param A HBS matrix
     * \param B Dense matrix
     * \param beta Scalar
     * \param C Result, should already be allocated to the appropriate
     * size.
     */
    template<typename scalar_t> void
    apply_HBS(Trans op, const HBSMatrix<scalar_t>& A,
              const DenseMatrix<scalar_t>& B,
              scalar_t beta, DenseMatrix<scalar_t>& C);

  } // end namespace HBS
} // end namespace strumpack

#endif // HBS_MATRIX_HPP
