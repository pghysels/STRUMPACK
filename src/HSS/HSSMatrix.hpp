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
 * \file HSSMatrix.hpp
 *
 * \brief This file contains the HSSMatrix class definition as well as
 * implementations for a number of it's member routines. Other member
 * routines are implemented in files such as HSSMatrix.apply.hpp,
 * HSSMatrix.factor.hpp etc.
 */
#ifndef HSS_MATRIX_HPP
#define HSS_MATRIX_HPP

#include <cassert>
#include <functional>
#include <string>

#include "HSSBasisID.hpp"
#include "HSSOptions.hpp"
#include "HSSExtra.hpp"
#include "HSSMatrixBase.hpp"
#include "kernel/Kernel.hpp"
#include "HSSMatrix.sketch.hpp"

namespace strumpack {
  namespace HSS {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    // forward declaration
    template<typename scalar_t> class HSSMatrixMPI;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


    /**
     * \class HSSMatrix
     *
     * \brief Class to represent a sequential/threaded Hierarchically
     * Semi-Separable matrix.
     *
     * This is for non-symmetric matrices, but can be used with
     * symmetric matrices as well. This class inherits from
     * HSSMatrixBase.  An HSS matrix is represented recursively, using
     * 2 children which are also HSSMatrix objects, except at the
     * lowest level (the leafs), where the HSSMatrix has no
     * children. Hence, the tree representing the HSS matrix is always
     * a binary tree, but not necessarily a complete binary tree.
     *
     * \tparam scalar_t Can be float, double, std:complex<float> or
     * std::complex<double>.
     *
     * \see HSSMatrixMPI, HSSMatrixBase
     */
    template<typename scalar_t> class HSSMatrix
      : public HSSMatrixBase<scalar_t> {
      using real_t = typename RealType<scalar_t>::value_type;
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using elem_t = typename std::function
        <void(const std::vector<std::size_t>& I,
              const std::vector<std::size_t>& J, DenseM_t& B)>;
      using mult_t = typename std::function
        <void(DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc)>;
      using opts_t = HSSOptions<scalar_t>;

    public:
      /**
       * Default constructor, constructs an empty 0 x 0 matrix.
       */
      HSSMatrix();

      /**
       * Construct an HSS representation for a dense matrix A. The HSS
       * tree will be constructed by splitting the row/column set
       * evenly and recursively until the leafs in the HSS tree are
       * smaller than opts.leaf_size(). Alternative constructors can
       * be used to specify a specific HSS partitioning
       * tree. Internally, this will call the appropriate compress
       * routine to construct the HSS representation, using the
       * options (such as compression tolerance, adaptive compression
       * scheme etc) as specified in the HSSOptions opts object.
       *
       * \param A dense matrix (unmodified) to compress as HSS
       * \param opts object containing a number of options for HSS
       * compression
       * \see DenseMatrix
       * \see HSSOptions
       */
      HSSMatrix(const DenseM_t& A, const opts_t& opts);

      /**
       * Construct an HSS representation for an m x n matrix. The HSS
       * tree will be constructed by splitting the row/column set
       * evenly and recursively until the leafs in the HSS tree are
       * smaller than opts.leaf_size(). After construction, the HSS
       * matrix will be empty, and can be filled by calling one of the
       * compress member routines. Alternative constructors can be
       * used to specify a specific HSS partitioning tree.
       *
       * \param m number of rows in the constructed HSS matrix
       * \param n number of rows in the constructed HSS matrix
       * \param opts object containing a number of options for HSS
       * compression
       * \see compress, HSSOptions
       */
      HSSMatrix(std::size_t m, std::size_t n, const opts_t& opts);

      /**
       * Construct an HSS matrix using a specified HSS tree. After
       * construction, the HSS matrix will be empty, and can be filled
       * by calling one of the compress member routines.
       *
       * \param t tree specifying the HSS matrix partitioning
       * \param opts object containing a number of options for HSS
       * compression
       * \see compress, HSSOptions
       */
      HSSMatrix(const structured::ClusterTree& t, const opts_t& opts);

      /**
       * Construct an HSS approximation for the kernel matrix K.
       *
       * \param K Kernel matrix object. The data associated with this
       * kernel will be permuted according to the clustering algorithm
       * selected by the HSSOptions objects. The permutation will be
       * stored in the kernel object.
       * \param opts object containing a number of HSS options
       */
      HSSMatrix(kernel::Kernel<real_t>& K, const opts_t& opts);

      /**
       * Copy constructor. Copying an HSSMatrix can be an expensive
       * operation.
       * \param other HSS matrix to be copied
       */
      HSSMatrix(const HSSMatrix<scalar_t>& other);

      /**
       * Copy assignment operator. Copying an HSSMatrix can be an
       * expensive operation.
       * \param other HSS matrix to be copied
       */
      HSSMatrix<scalar_t>& operator=(const HSSMatrix<scalar_t>& other);

      /**
       * Move constructor.
       * \param other HSS matrix to be moved from, will be emptied
       */
      HSSMatrix(HSSMatrix<scalar_t>&& other) = default;

      /**
       * Move assignment operator.
       * \param other HSS matrix to be moved from, will be emptied
       */
      HSSMatrix<scalar_t>& operator=(HSSMatrix<scalar_t>&& other) = default;

      /**
       * Create a clone of this matrix.
       * TODO remove this!!???
       */
      std::unique_ptr<HSSMatrixBase<scalar_t>> clone() const override;

      /**
       * Return a const raw (non-owning) pointer to child c of this
       * HSS matrix. A child of an HSS matrix is itself an HSS
       * matrix. The value of c should be 0 or 1, and this HSS matrix
       * should not be a leaf!
       */
      const HSSMatrix<scalar_t>* child(int c) const {
        return dynamic_cast<HSSMatrix<scalar_t>*>(this->ch_[c].get());
      }

      /**
       * Return a raw (non-owning) pointer to child c of this HSS
       * matrix. A child of an HSS matrix is itself an HSS matrix. The
       * value of c should be 0 or 1, and this HSS matrix should not
       * be a leaf!
       */
      HSSMatrix<scalar_t>* child(int c) {
        return dynamic_cast<HSSMatrix<scalar_t>*>(this->ch_[c].get());
      }

      /**
       * Initialize this HSS matrix as the compressed HSS
       * representation of a given dense matrix. The HSS matrix should
       * have been constructed with the proper sizes, i.e., rows() ==
       * A.rows() and cols() == A.cols().  Internaly, this will call
       * the appropriate compress routine to construct the HSS
       * representation, using the options (such as compression
       * tolerance, adaptive compression scheme etc) as specified in
       * the HSSOptions opts object.
       *
       * \param A dense matrix (unmodified) to compress as HSS
       * \param opts object containing a number of options for HSS
       * compression
       * \see DenseMatrix
       * \see HSSOptions
       */
      void compress(const DenseM_t& A, const opts_t& opts);

      /**
       * Initialize this HSS matrix as the compressed HSS
       * representation. The compression uses the
       * matrix-(multiple)vector multiplication routine Amult, and the
       * element (sub-matrix) extraction routine Aelem, both provided
       * by the user. The HSS matrix should have been constructed with
       * the proper sizes, i.e., rows() == A.rows() and cols() ==
       * A.cols().  Internaly, this will call the appropriate compress
       * routine to construct the HSS representation, using the
       * options (such as compression tolerance, adaptive compression
       * scheme etc) as specified in the HSSOptions opts object.
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
       * \param Aelem element extraction routine. This can be a
       * functor, or a lambda function for instance.
       * \param I Parameter in the element extraction routine. Set of
       * row indices of elements to extract.
       * \param J Parameter in the element extraction routine. Set of
       * column indices of elements to extract.
       * \param B Parameter in the element extraction routine. Matrix
       * where to place extracted elements. This matrix will already
       * be allocated. It will have B.rows() == I.size() and B.cols()
       * == J.size().
       * \param opts object containing a number of options for HSS
       * compression
       * \see DenseMatrix
       * \see HSSOptions
       */
      void compress(const std::function<void(DenseM_t& Rr,
                                             DenseM_t& Rc,
                                             DenseM_t& Sr,
                                             DenseM_t& Sc)>& Amult,
                    const std::function<void(const std::vector<std::size_t>& I,
                                             const std::vector<std::size_t>& J,
                                             DenseM_t& B)>& Aelem,
                    const opts_t& opts);


      /**
       * Initialize this HSS matrix as the compressed HSS
       * representation. The compression uses nearest neighbor
       * information (coordinates provided by the user).  The HSS
       * matrix should have been constructed with the proper sizes,
       * i.e., rows() == A.rows() and cols() == A.cols().
       *
       * \param coords matrix with coordinates for the underlying
       * geometry that defined the HSS matrix. This should be a d x n
       * matrix (d rows, n columns), where d is the dimension of the
       * coordinates and n is rows() and cols() of this matrix.
       * \param Aelem element extraction routine. This can be a
       * functor, or a lambda function for instance.
       * \param I Parameter in the element extraction routine. Set of
       * row indices of elements to extract.
       * \param J Parameter in the element extraction routine. Set of
       * column indices of elements to extract.
       * \param B Parameter in the element extraction routine. Matrix
       * where to place extracted elements. This matrix will already
       * be allocated. It will have B.rows() == I.size() and B.cols()
       * == J.size().
       * \param opts object containing a number of options for HSS
       * compression
       * \see DenseMatrix
       * \see HSSOptions
       */
      void compress_with_coordinates(const DenseMatrix<real_t>& coords,
                                     const std::function
                                     <void(const std::vector<std::size_t>& I,
                                           const std::vector<std::size_t>& J,
                                           DenseM_t& B)>& Aelem,
                                     const opts_t& opts);

      /**
       * Reset the matrix to an empty, 0 x 0 matrix, freeing up all
       * it's memory.
       */
      void reset() override;

      /**
       * Compute a ULV factorization of this matrix.
       */
      void factor() override;

      /**
       * Compute a partial ULV factorization of this matrix. Only the
       * left child is factored. This is not similar to calling
       * child(0)->factor(), except that the HSSFactors resulting from
       * calling partial_factor can be used to compute the Schur
       * complement.
       *
       * \see Schur_update, Schur_product_direct and
       * Schur_product_indirect
       */
      void partial_factor();

      /**
       * Solve a linear system with the ULV factorization of this
       * HSSMatrix. The right hand side vector (or matrix) b is
       * overwritten with the solution x of Ax=b.
       *
       * \param ULV ULV factorization of this matrix
       * \param b on input, the right hand side vector, on output the
       * solution of A x = b (with A this HSS matrix). The vector b
       * should be b.rows() == cols().
       * \see factor
       */
      void solve(DenseM_t& b) const override;

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
      void forward_solve(WorkSolve<scalar_t>& w, const DenseM_t& b,
                         bool partial) const override;

      /**
       * Perform only the backward phase of the ULV linear solve. This
       * is for advanced use only, typically to be used in combination
       * with partial_factor. You should really just use factor/solve
       * when possible.
       *
       * \param w temporary working storage, to pass information from
       * forward_solve to backward_solve
       * \param b on input, the vector obtained from forward_solve, on
       * output the solution of A x = b (with A this HSS matrix). The
       * vector b should be b.rows() == cols().
       * \see factor, partial_factor, backward_solve
       */
      void backward_solve(WorkSolve<scalar_t>& w, DenseM_t& x) const override;

      /**
       * Multiply this HSS matrix with a dense matrix (vector), ie,
       * compute x = this * b.
       *
       * \param b Matrix to multiply with, from the left.
       * \return The result of this * b.
       * \see applyC, HSS::apply_HSS
       */
      DenseM_t apply(const DenseM_t& b) const;

      /**
       * Multiply this HSS matrix with a dense matrix (vector), ie,
       * compute y = op(this) * x. Overrides from the StructuredMatrix
       * class method.
       *
       * \param op Transpose or complex conjugate
       * \param x right hand side matrix to multiply with, from the
       * left, rows(x) == cols(op(this))
       * \param y result of op(this) * b, cols(y) == cols(x), rows(r)
       * = rows(op(this))
       * \see applyC, HSS::apply_HSS
       */
      void mult(Trans op, const DenseM_t& x, DenseM_t& y) const override;

      /**
       * Multiply the transpose or complex conjugate of this HSS
       * matrix with a dense matrix (vector), ie, compute x = this^C *
       * b.
       *
       * \param b Matrix to multiply with, from the left.
       * \return The result of this^C * b.
       * \see apply, HSS::apply_HSS
       */
      DenseM_t applyC(const DenseM_t& b) const;

      /**
       * Extract a single element this(i,j) from this HSS matrix. This
       * is expensive and should not be used to compute multiple
       * elements.
       *
       * \param i Global row index of element to compute.
       * \param j Global column index of element to compute.
       * \return this(i,j)
       * \see extract
       */
      scalar_t get(std::size_t i, std::size_t j) const;

      /**
       * Compute a submatrix this(I,J) from this HSS matrix.
       *
       * \param I set of row indices of elements to compute from this
       * HSS matrix.
       * \param I set of column indices of elements to compute from
       * this HSS matrix.
       * \return Submatrix from this HSS matrix this(I,J)
       * \see get, extract_add
       */
      DenseM_t extract(const std::vector<std::size_t>& I,
                       const std::vector<std::size_t>& J) const;

      /**
       * Compute a submatrix this(I,J) from this HSS matrix and add it
       * to a given matrix B.
       *
       * \param I set of row indices of elements to compute from this
       * HSS matrix.
       * \param I set of column indices of elements to compute from
       * this HSS matrix.
       * \param The extracted submatrix this(I,J) will be added to
       * this matrix. Should satisfy B.rows() == I.size() and B.cols()
       * == J.size().
       * \see get, extract
       */
      void extract_add(const std::vector<std::size_t>& I,
                       const std::vector<std::size_t>& J,
                       DenseM_t& B) const;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
      void Schur_update(DenseM_t& Theta,
                        DenseM_t& DUB01,
                        DenseM_t& Phi) const;
      void Schur_product_direct(const DenseM_t& Theta,
                                const DenseM_t& DUB01,
                                const DenseM_t& Phi,
                                const DenseM_t&_ThetaVhatC_or_VhatCPhiC,
                                const DenseM_t& R,
                                DenseM_t& Sr, DenseM_t& Sc) const;
      void Schur_product_indirect(const DenseM_t& DUB01,
                                  const DenseM_t& R1,
                                  const DenseM_t& R2, const DenseM_t& Sr2,
                                  const DenseM_t& Sc2,
                                  DenseM_t& Sr, DenseM_t& Sc) const;
      void delete_trailing_block() override;
#endif // DOXYGEN_SHOULD_SKIP_THIS

      std::size_t rank() const override;
      std::size_t memory() const override;
      std::size_t nonzeros() const override;
      std::size_t levels() const override;

      void print_info(std::ostream& out=std::cout,
                      std::size_t roff=0,
                      std::size_t coff=0) const override;

      /**
       * Return a full/dense representation of this HSS matrix.
       *
       * \return Dense matrix obtained from multiplying out the
       * low-rank representation.
       */
      DenseM_t dense() const;

      void shift(scalar_t sigma) override;

      void draw(std::ostream& of,
                std::size_t rlo=0, std::size_t clo=0) const override;

      /**
       * Write this HSSMatrix<scalar_t> to a binary file, called
       * fname.
       *
       * \see read
       */
      void write(const std::string& fname) const;

      /**
       * Read an HSSMatrix<scalar_t> from a binary file, called
       * fname.
       *
       * \see write
       */
      static HSSMatrix<scalar_t> read(const std::string& fname);

      const HSSFactors<scalar_t>& ULV() { return this->ULV_; }

    protected:
      HSSMatrix(std::size_t m, std::size_t n,
                const opts_t& opts, bool active);
      HSSMatrix(const structured::ClusterTree& t,
                const opts_t& opts, bool active);
      HSSMatrix(std::ifstream& is);

      HSSBasisID<scalar_t> U_, V_;
      DenseM_t D_, B01_, B10_;

      void compress_original(const DenseM_t& A,
                             const opts_t& opts);
      void compress_original(const mult_t& Amult,
                             const elem_t& Aelem,
                             const opts_t& opts);
      void compress_stable(const DenseM_t& A,
                           const opts_t& opts);
      void compress_stable(const mult_t& Amult,
                           const elem_t& Aelem,
                           const opts_t& opts);
      void compress_hard_restart(const DenseM_t& A,
                                 const opts_t& opts);
      void compress_hard_restart(const mult_t& Amult,
                                 const elem_t& Aelem,
                                 const opts_t& opts);

      void compress_recursive_original(DenseM_t& Rr, DenseM_t& Rc,
                                       DenseM_t& Sr, DenseM_t& Sc,
                                       const elem_t& Aelem,
                                       const opts_t& opts,
                                       WorkCompress<scalar_t>& w,
                                       int dd, int depth) override;
      void compress_recursive_stable(DenseM_t& Rr, DenseM_t& Rc,
                                     DenseM_t& Sr, DenseM_t& Sc,
                                     const elem_t& Aelem,
                                     const opts_t& opts,
                                     WorkCompress<scalar_t>& w,
                                     int d, int dd, int depth) override;
      // SJLT_Matrix<scalar_t,int>* S=nullptr
      void compute_local_samples(DenseM_t& Rr, DenseM_t& Rc,
                                 DenseM_t& Sr, DenseM_t& Sc,
                                 WorkCompress<scalar_t>& w,
                                 int d0, int d, int depth,
                                 SJLTMatrix<scalar_t, int>* S=nullptr);
      bool compute_U_V_bases(DenseM_t& Sr, DenseM_t& Sc, const opts_t& opts,
                             WorkCompress<scalar_t>& w, int d, int depth);
      void compute_U_basis_stable(DenseM_t& Sr, const opts_t& opts,
                                  WorkCompress<scalar_t>& w,
                                  int d, int dd, int depth);
      void compute_V_basis_stable(DenseM_t& Sc, const opts_t& opts,
                                  WorkCompress<scalar_t>& w,
                                  int d, int dd, int depth);
      void reduce_local_samples(DenseM_t& Rr, DenseM_t& Rc,
                                WorkCompress<scalar_t>& w,
                                int d0, int d, int depth);
      bool update_orthogonal_basis(const opts_t& opts, scalar_t& r_max_0,
                                   const DenseM_t& S, DenseM_t& Q,
                                   int d, int dd, bool untouched,
                                   int L, int depth);
      void set_U_full_rank(WorkCompress<scalar_t>& w);
      void set_V_full_rank(WorkCompress<scalar_t>& w);

      void compress_level_original(DenseM_t& Rr, DenseM_t& Rc,
                                   DenseM_t& Sr, DenseM_t& Sc,
                                   const opts_t& opts,
                                   WorkCompress<scalar_t>& w,
                                   int dd, int lvl, int depth) override;
      void compress_level_stable(DenseM_t& Rr, DenseM_t& Rc,
                                 DenseM_t& Sr, DenseM_t& Sc,
                                 const opts_t& opts,
                                 WorkCompress<scalar_t>& w,
                                 int d, int dd, int lvl, int depth) override;
      void get_extraction_indices(std::vector<std::vector<std::size_t>>& I,
                                  std::vector<std::vector<std::size_t>>& J,
                                  const std::pair<std::size_t,std::size_t>& off,
                                  WorkCompress<scalar_t>& w,
                                  int& self, int lvl) override;
      void get_extraction_indices(std::vector<std::vector<std::size_t>>& I,
                                  std::vector<std::vector<std::size_t>>& J,
                                  std::vector<DenseM_t*>& B,
                                  const std::pair<std::size_t,std::size_t>& off,
                                  WorkCompress<scalar_t>& w,
                                  int& self, int lvl) override;
      void extract_D_B(const elem_t& Aelem, const opts_t& opts,
                       WorkCompress<scalar_t>& w, int lvl) override;

      void compress(const kernel::Kernel<real_t>& K, const opts_t& opts);
      void compress_recursive_ann(DenseMatrix<std::uint32_t>& ann,
                                  DenseMatrix<real_t>&  scores,
                                  const elem_t& Aelem, const opts_t& opts,
                                  WorkCompressANN<scalar_t>& w,
                                  int depth) override;
      void compute_local_samples_ann(DenseMatrix<std::uint32_t>& ann,
                                     DenseMatrix<real_t>& scores,
                                     WorkCompressANN<scalar_t>& w,
                                     const elem_t& Aelem, const opts_t& opts);
      bool compute_U_V_bases_ann(DenseM_t& S, const opts_t& opts,
                                 WorkCompressANN<scalar_t>& w, int depth);

      void factor_recursive(WorkFactor<scalar_t>& w,
                            bool isroot, bool partial,
                            int depth) override;

      void apply_fwd(const DenseM_t& b, WorkApply<scalar_t>& w,
                     bool isroot, int depth,
                     std::atomic<long long int>& flops) const override;
      void apply_bwd(const DenseM_t& b, scalar_t beta, DenseM_t& c,
                     WorkApply<scalar_t>& w, bool isroot, int depth,
                     std::atomic<long long int>& flops) const override;
      void applyT_fwd(const DenseM_t& b, WorkApply<scalar_t>& w, bool isroot,
                      int depth, std::atomic<long long int>& flops) const override;
      void applyT_bwd(const DenseM_t& b, scalar_t beta, DenseM_t& c,
                      WorkApply<scalar_t>& w, bool isroot, int depth,
                      std::atomic<long long int>& flops) const override;

      void solve_fwd(const DenseM_t& b, WorkSolve<scalar_t>& w,
                     bool partial, bool isroot, int depth) const override;
      void solve_bwd(DenseM_t& x, WorkSolve<scalar_t>& w,
                     bool isroot, int depth) const override;

      void extract_fwd(WorkExtract<scalar_t>& w,
                       bool odiag, int depth) const override;
      void extract_bwd(DenseM_t& B, WorkExtract<scalar_t>& w,
                       int depth) const override;
      void extract_bwd(std::vector<Triplet<scalar_t>>& triplets,
                       WorkExtract<scalar_t>& w, int depth) const override;
      void extract_bwd_internal(WorkExtract<scalar_t>& w, int depth) const;

      void apply_UV_big(DenseM_t& Theta, DenseM_t& Uop, DenseM_t& Phi,
                        DenseM_t& Vop,
                        const std::pair<std::size_t, std::size_t>& offset,
                        int depth, std::atomic<long long int>& flops)
        const override;
      void apply_UtVt_big(const DenseM_t& A, DenseM_t& UtA, DenseM_t& VtA,
                          const std::pair<std::size_t, std::size_t>& offset,
                          int depth, std::atomic<long long int>& flops)
        const override;

      void dense_recursive(DenseM_t& A, WorkDense<scalar_t>& w,
                           bool isroot, int depth) const override;

      /**
       * \see HSS::apply_HSS
       */
      template<typename T> friend
      void apply_HSS(Trans ta, const HSSMatrix<T>& a, const DenseMatrix<T>& b,
                     T beta, DenseMatrix<T>& c);

      /**
       * \see HSS::draw
       */
      template<typename T> friend
      void draw(const HSSMatrix<T>& H, const std::string& name);

      void read(std::ifstream& is) override;
      void write(std::ofstream& os) const override;

      friend class HSSMatrixMPI<scalar_t>;

      using HSSMatrixBase<scalar_t>::child;
    };

    /**
     * Write a gnuplot script to draw this an matrix.
     *
     * \param H HSS matrix to draw.
     * \param name Name of the HSS matrix. The script will be created
     * in the file plotname.gnuplot.
     */
    template<typename scalar_t>
    void draw(const HSSMatrix<scalar_t>& H, const std::string& name);

    /**
     * Compute C = op(A) * B + beta * C, with HSS matrix A.
     *
     * \param op Transpose/complex conjugate or none to be applied to
     * the HSS matrix A.
     * \param A HSS matrix
     * \param B Dense matrix
     * \param beta Scalar
     * \param C Result, should already be allocated to the appropriate
     * size.
     */
    template<typename scalar_t> void
    apply_HSS(Trans op, const HSSMatrix<scalar_t>& A,
              const DenseMatrix<scalar_t>& B,
              scalar_t beta, DenseMatrix<scalar_t>& C);

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_HPP
