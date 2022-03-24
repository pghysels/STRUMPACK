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
 * \file CompressedSparseMatrix.hpp
 * \brief Contains the CompressedSparseMatrix class, a base class for
 * compressed sparse storage.
 */
#ifndef COMPRESSED_SPARSE_MATRIX_HPP
#define COMPRESSED_SPARSE_MATRIX_HPP

#include <vector>
#include <string>
#include <tuple>

#include "misc/Tools.hpp"
#include "misc/Triplet.hpp"
#include "dense/DenseMatrix.hpp"
#include "StrumpackOptions.hpp"

namespace strumpack {

  // forward declarations
  template<typename integer_t> class CSRGraph;
  template<typename scalar_t> class DenseMatrix;
  template<typename scalar_t> class DistributedMatrix;


  template<typename scalar_t, typename integer_t,
           typename real_t = typename RealType<scalar_t>::value_type>
  class MatchingData {
  public:
    MatchingData() {}
    MatchingData(MatchingJob j, std::size_t n) : job(j) {
      if (job != MatchingJob::NONE)
        Q.resize(n);
      if (job == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING) {
        R.resize(n);
        C.resize(n);
      }
    }

    MatchingJob job = MatchingJob::NONE;
    std::vector<integer_t> Q;
    std::vector<real_t> R, C;

    integer_t mc64_work_int(std::size_t n, std::size_t nnz) const {
      switch (job) {
      case MatchingJob::MAX_SMALLEST_DIAGONAL: return 4*n;
      case MatchingJob::MAX_SMALLEST_DIAGONAL_2: return 10*n + nnz;
      case MatchingJob::MAX_CARDINALITY:
      case MatchingJob::MAX_DIAGONAL_SUM:
      case MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING:
      default: return 5*n;
      }
    }

    integer_t mc64_work_double(std::size_t n, std::size_t nnz) const {
      switch (job) {
      case MatchingJob::MAX_CARDINALITY: return 0;
      case MatchingJob::MAX_SMALLEST_DIAGONAL: return n;
      case MatchingJob::MAX_SMALLEST_DIAGONAL_2: return nnz;
      case MatchingJob::MAX_DIAGONAL_SUM: return 2*n + nnz;
      case MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING:
      default: return 3*n + nnz;
      }
    }
  };

  template<typename scalar_t,
           typename real_t = typename RealType<scalar_t>::value_type>
  class Equilibration {
  public:
    Equilibration() {}
    Equilibration(std::size_t N) : R(N), C(N) {}
    // use this one for block-row distributed sparse matrix, with rows
    // local rows, and cols global columns
    Equilibration(std::size_t rows, std::size_t cols) : R(rows), C(cols) {}

    int info = 0;
    EquilibrationType type = EquilibrationType::NONE;
    real_t rcond = 1, ccond = 1, Amax = 0;
    std::vector<real_t> R, C;

    void set_type() {
      const real_t thres = 0.1;
      const real_t small = blas::lamch<real_t>('S') / blas::lamch<real_t>('P');
      const real_t large = 1. / small;
      if (rcond >= thres && Amax >= small && Amax <= large) {
        R.clear();
        if (ccond >= thres) {
          type = EquilibrationType::NONE;
          C.clear();
        } else type = EquilibrationType::COLUMN;
      } else {
        if (ccond >= thres) {
          type = EquilibrationType::ROW;
          C.clear();
        } else type = EquilibrationType::BOTH;
      }
    }
  };

  /**
   * \class CompressedSparseMatrix
   * \brief Abstract base class for compressed sparse matrix storage.
   *
   * This is an abstract (pure virtual) base class for compressed
   * sparse matrices.
   *
   * This is only for __square__ matrices!
   * __TODO make this work on non-square matrices__
   *
   * The rows and the columns should always be __sorted__!
   *
   * __TODO make the public interface non-virtual__
   *
   * \tparam scalar_t type used to store matrix values
   * \tparam integer_t type used for indices in the row/column pointer
   * arrays and the column/row indices
   *
   * \see CSRMatrix, CSRMatrixMPI
   */
  template<typename scalar_t,typename integer_t>
  class CompressedSparseMatrix {
    using DenseM_t = DenseMatrix<scalar_t>;
    using real_t = typename RealType<scalar_t>::value_type;
#if defined(STRUMPACK_USE_MPI)
    using DistM_t = DistributedMatrix<scalar_t>;
#endif
    using Match_t = MatchingData<scalar_t,integer_t>;
    using Equil_t = Equilibration<scalar_t>;

  public:
    /**
     * Virtual destructor.
     */
    virtual ~CompressedSparseMatrix() {}

    /**
     * Return the size of this __square__ matrix, ie, the number of
     * rows and columns. For distributed memory matrices, this refer
     * to the global size, not the local size.
     */
    integer_t size() const { return n_; }

    /**
     * Return the number of nonzeros in this matrix.  For distributed
     * memory matrices, this refer to the global number of nonzeros,
     * not the local number of nonzeros.
     */
    integer_t nnz() const { return nnz_; }

    /**
     * Return a (const) pointer to the (row/column) pointer. For a
     * compressed sparse row format, this will be a pointer to an
     * array of dimension size()+1, where element ptr[i] denotes the
     * start (in ind() or val()) of row i, and ptr[i+1] is the end of
     * row i.
     *
     * \see ptr(integer_t i)
     */
    const integer_t* ptr() const { return ptr_.data(); }

    /**
     * Return a (const) pointer to the (row/column) indices. For
     * compressed sparse row, this will return a pointer to an array
     * of column indices, 1 for each nonzero.
     *
     * \see ptr, ind(integer_t i)
     */
    const integer_t* ind() const { return ind_.data(); }

    /**
     * Return a (const) pointer to the nonzero values. For compressed
     * sparse row, this will return a pointer to an array of nonzero
     * values.
     *
     * \see ptr, ind
     */
    const scalar_t* val() const { return val_.data(); }

    /**
     * Return a pointer to the (row/column) pointer. For a compressed
     * sparse row format, this will be a pointer to an array of
     * dimension size()+1, where element ptr[i] denotes the start (in
     * ind() or val()) of row i, and ptr[i+1] is the end of row i.
     *
     * \see ptr(integer_t i)
     */
    integer_t* ptr() { return ptr_.data(); }

    /**
     * Return a pointer to the (row/column) indices. For compressed
     * sparse row, this will return a pointer to an array of column
     * indices, 1 for each nonzero.
     *
     * \see ptr, ind(integer_t i)
     */
    integer_t* ind() { return ind_.data(); }

    /**
     * Return a pointer to the nonzero values. For compressed sparse
     * row, this will return a pointer to an array of nonzero values.
     *
     * \see ptr, ind
     */
    scalar_t* val() { return val_.data(); }

    /**
     * Access ptr[i], ie, the start of row i (in ind, or in val). This
     * will assert that i >= 0 and i <= size(). These assertions are
     * removed when compiling in Release mode (adding -DNDEBUG).
     */
    const integer_t& ptr(integer_t i) const { assert(i >= 0 && i <= size()); return ptr_[i]; }

    /**
     * Access ind[i], ie, the row/column index of the i-th
     * nonzero. This will assert that i >= 0 and i < nnz(). These
     * assertions are removed when compiling in Release mode (adding
     * -DNDEBUG).
     */
    const integer_t& ind(integer_t i) const { assert(i >= 0 && i < nnz()); return ind_[i]; }

    /**
     * Access ind[i], ie, the value of the i-th nonzero. This will
     * assert that i >= 0 and i < nnz(). These assertions are removed
     * when compiling in Release mode (adding -DNDEBUG).
     */
    const scalar_t& val(integer_t i) const { assert(i >= 0 && i < nnz()); return val_[i]; }

    /**
     * Access ptr[i], ie, the start of row i (in ind, or in val). This
     * will assert that i >= 0 and i <= size(). These assertions are
     * removed when compiling in Release mode (adding -DNDEBUG).
     */
    integer_t& ptr(integer_t i) { assert(i <= size()); return ptr_[i]; }

    /**
     * Access ind[i], ie, the row/column index of the i-th
     * nonzero. This will assert that i >= 0 and i < nnz(). These
     * assertions are removed when compiling in Release mode (adding
     * -DNDEBUG).
     */
    integer_t& ind(integer_t i) { assert(i < nnz()); return ind_[i]; }

    /**
     * Access ind[i], ie, the value of the i-th nonzero. This will
     * assert that i >= 0 and i < nnz(). These assertions are removed
     * when compiling in Release mode (adding -DNDEBUG).
     */
    scalar_t& val(integer_t i) { assert(i < nnz()); return val_[i]; }

    virtual real_t norm1() const = 0; //{ assert(false); return -1.; };

    /**
     * Check whether the matrix has a symmetric sparsity pattern (as
     * specified in the constructor, will not actually check).
     */
    bool symm_sparse() const { return symm_sparse_; }

    /**
     * Specify that the sparsity pattern of this matrix is symmetric,
     * or not.
     *
     * \param symm_sparse bool, set this to true (default if not
     * provided) to specify that the sparsity pattern is symmetric
     */
    void set_symm_sparse(bool symm_sparse=true) { symm_sparse_ = symm_sparse; }


    /**
     * Sparse matrix times dense vector/matrix product
     *    y = this * x
     * x and y can have multiple columns.
     * y should be pre-allocated!
     *
     * TODO make the public interface non-virtual
     *
     * \param x input right hand-side vector/matrix, should satisfy
     * x.size() == this->size()
     * \param y output, result of y = this * x, should satisfy
     * y.size() == this->size()
     */
    virtual void spmv(const DenseM_t& x, DenseM_t& y) const = 0;

    /**
     * Sparse matrix times dense vector product
     *    y = this * x
     * x and y can have multiple columns.
     * y should be pre-allocated.
     *
     * TODO make the public interface non-virtual
     *
     * \param x input right hand-side vector/matrix, should be a
     * pointer to an array of size size()
     * \param y output, result of y = this * x, should be a pointer to
     * an array of size size(), already allocated
     */
    virtual void spmv(const scalar_t* x, scalar_t* y) const = 0;

    /**
     * TODO Obtain reordering Anew = A(iorder,iorder). In addition,
     * entries of IND, VAL are sorted in increasing order
     */
    virtual void permute(const integer_t* iorder, const integer_t* order);

    virtual void permute(const std::vector<integer_t>& iorder,
                         const std::vector<integer_t>& order) {
      permute(iorder.data(), order.data());
    }

    virtual void permute_columns(const std::vector<integer_t>& perm) = 0;

    virtual Equil_t equilibration() const { return Equil_t(this->size()); }

    virtual void equilibrate(const Equil_t&) {}

    virtual Match_t matching(MatchingJob, bool apply=true);

    virtual void apply_matching(const Match_t&);

    virtual void symmetrize_sparsity();

    virtual void print() const;
    virtual void print_dense(const std::string& name) const {
      std::cerr << "print_dense not implemented for this matrix type"
                << std::endl;
    }
    virtual void print_matrix_market(const std::string& filename) const {
      std::cerr << "print_matrix_market not implemented for this matrix type"
                << std::endl;
    }

    virtual int read_matrix_market(const std::string& filename) = 0;

    virtual real_t max_scaled_residual(const scalar_t* x,
                                       const scalar_t* b) const = 0;
    virtual real_t max_scaled_residual(const DenseM_t& x,
                                       const DenseM_t& b) const = 0;


#ifndef DOXYGEN_SHOULD_SKIP_THIS
    virtual CSRGraph<integer_t>
    extract_graph(int ordering_level, integer_t lo, integer_t hi) const = 0;
    virtual CSRGraph<integer_t>
    extract_graph_sep_CB(int ordering_level, integer_t lo, integer_t hi,
                         const std::vector<integer_t>& upd) const = 0;
    virtual CSRGraph<integer_t>
    extract_graph_CB_sep(int ordering_level, integer_t lo, integer_t hi,
                         const std::vector<integer_t>& upd) const = 0;
    virtual CSRGraph<integer_t>
    extract_graph_CB(int ordering_level,
                     const std::vector<integer_t>& upd) const = 0;

    virtual void
    extract_separator(integer_t sep_end, const std::vector<std::size_t>& I,
                      const std::vector<std::size_t>& J, DenseM_t& B,
                      int depth) const = 0;
    virtual void
    extract_front(DenseM_t& F11, DenseM_t& F12, DenseM_t& F21,
                  integer_t slo, integer_t shi,
                  const std::vector<integer_t>& upd,
                  int depth) const = 0;
    virtual void
    push_front_elements(integer_t, integer_t, const std::vector<integer_t>&,
                        std::vector<Triplet<scalar_t>>&,
                        std::vector<Triplet<scalar_t>>&,
                        std::vector<Triplet<scalar_t>>&) const = 0;
    virtual void
    set_front_elements(integer_t, integer_t, const std::vector<integer_t>&,
                       Triplet<scalar_t>*, Triplet<scalar_t>*,
                       Triplet<scalar_t>*) const = 0;
    virtual void
    count_front_elements(integer_t, integer_t, const std::vector<integer_t>&,
                         std::size_t&, std::size_t&, std::size_t&) const = 0;

    virtual void
    front_multiply(integer_t slo, integer_t shi,
                   const std::vector<integer_t>& upd,
                   const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc,
                   int depth) const = 0;

    virtual void
    front_multiply_F11(Trans op, integer_t slo, integer_t shi,
                       const DenseM_t& R, DenseM_t& S, int depth) const = 0;
    virtual void
    front_multiply_F12(Trans op, integer_t slo, integer_t shi,
                       const std::vector<integer_t>& upd,
                       const DenseM_t& R, DenseM_t& S, int depth) const = 0;
    virtual void
    front_multiply_F21(Trans op, integer_t slo, integer_t shi,
                       const std::vector<integer_t>& upd,
                       const DenseM_t& R, DenseM_t& S, int depth) const = 0;

#if defined(STRUMPACK_USE_MPI)
    virtual void
    extract_F11_block(scalar_t* F, integer_t ldF,
                      integer_t row, integer_t nr_rows,
                      integer_t col, integer_t nr_cols) const = 0;
    virtual void
    extract_F12_block(scalar_t* F, integer_t ldF,
                      integer_t row, integer_t nr_rows,
                      integer_t col, integer_t nr_cols,
                      const integer_t* upd) const = 0;
    virtual void
    extract_F21_block(scalar_t* F, integer_t ldF,
                      integer_t row, integer_t nr_rows,
                      integer_t col, integer_t nr_cols,
                      const integer_t* upd) const = 0;
    virtual void
    extract_separator_2d(integer_t sep_end,
                         const std::vector<std::size_t>& I,
                         const std::vector<std::size_t>& J,
                         DistM_t& B) const = 0;
    virtual void
    front_multiply_2d(integer_t sep_begin, integer_t sep_end,
                      const std::vector<integer_t>& upd,
                      const DistM_t& R, DistM_t& Srow, DistM_t& Scol,
                      int depth) const = 0;
    virtual void
    front_multiply_2d(Trans op, integer_t sep_begin, integer_t sep_end,
                      const std::vector<integer_t>& upd, const DistM_t& R,
                      DistM_t& S, int depth) const = 0;
#endif //STRUMPACK_USE_MPI
#endif //DOXYGEN_SHOULD_SKIP_THIS

  protected:
    integer_t n_, nnz_;
    std::vector<integer_t> ptr_, ind_;
    std::vector<scalar_t> val_;
    bool symm_sparse_;

    enum MMsym {GENERAL, SYMMETRIC, SKEWSYMMETRIC, HERMITIAN};

    CompressedSparseMatrix();
    CompressedSparseMatrix(integer_t n, integer_t nnz,
                           bool symm_sparse=false);
    CompressedSparseMatrix(integer_t n,
                           const integer_t* row_ptr,
                           const integer_t* col_ind,
                           const scalar_t* values, bool symm_sparsity);

    std::vector<std::tuple<integer_t,integer_t,scalar_t>>
    read_matrix_market_entries(const std::string& filename);

    virtual int strumpack_mc64(MatchingJob, Match_t&) { return 0; }

    virtual void scale(const std::vector<scalar_t>&,
                       const std::vector<scalar_t>&) = 0;
    virtual void scale_real(const std::vector<real_t>&,
                            const std::vector<real_t>&) = 0;

    long long spmv_flops() const;
    long long spmv_bytes() const;
  };

} //end namespace strumpack

#endif // COMPRESSED_SPARSE_MATRIX_HPP
