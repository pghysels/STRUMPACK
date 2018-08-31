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
#ifndef COMPRESSEDSPARSEMATRIX_HPP
#define COMPRESSEDSPARSEMATRIX_HPP

#include <vector>
#include <algorithm>
#include <tuple>
#include <stdio.h>
#include <string.h>
#include "StrumpackOptions.hpp"
#include "misc/Tools.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "../dense/DistributedMatrix.hpp"
#endif
#include "../dense/DenseMatrix.hpp"

// where is this used?? in MC64?
#ifdef _LONGINT
  typedef long long int int_t;
#else // Default
  typedef int int_t;
#endif

namespace strumpack {

  /**
   * \class CompressedSparseMatrix
   * \brief Abstract base class for compressed sparse matrix storage.
   *
   * This is an abstract (pure virtual) base class for compressed
   * sparse matrices.
   *
   * This is only for __square__ matrices!
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
    virtual void permute
    (const std::vector<integer_t>& iorder, const std::vector<integer_t>& order)
    { permute(iorder.data(), order.data()); }
    virtual int permute_and_scale
    (MatchingJob job, std::vector<integer_t>& perm, std::vector<scalar_t>& Dr,
     std::vector<scalar_t>& Dc, bool apply=true);
    virtual void apply_scaling
    (const std::vector<scalar_t>& Dr, const std::vector<scalar_t>& Dc) = 0;
    virtual void apply_column_permutation
    (const std::vector<integer_t>& perm) = 0;
    virtual void symmetrize_sparsity();
    virtual void print() const;
    virtual void print_dense(const std::string& name) const {
      std::cerr << "print_dense not implemented for this matrix type"
                << std::endl;
    }
    virtual void print_MM(const std::string& filename) const {
      std::cerr << "print_MM not implemented for this matrix type"
                << std::endl;
    }
    virtual int read_matrix_market(const std::string& filename) = 0;
    virtual real_t max_scaled_residual
    (const scalar_t* x, const scalar_t* b) const = 0;
    virtual real_t max_scaled_residual
    (const DenseM_t& x, const DenseM_t& b) const = 0;
    virtual void strumpack_mc64
    (int_t job, int_t* num, integer_t* perm, int_t liw, int_t* iw, int_t ldw,
     double* dw, int_t* icntl, int_t* info) {}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    // TODO implement these outside of this class
    virtual void extract_separator
    (integer_t sep_end, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J, DenseM_t& B, int depth) const = 0;
    virtual void extract_front
    (DenseM_t& F11, DenseM_t& F12, DenseM_t& F21,
     integer_t sep_begin, integer_t sep_end,
     const std::vector<integer_t>& upd, int depth) const = 0;
    virtual void front_multiply
    (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
     const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc, int depth) const = 0;

#if defined(STRUMPACK_USE_MPI)
    virtual void extract_F11_block
    (scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows,
     integer_t col, integer_t nr_cols) const = 0;
    virtual void extract_F12_block
    (scalar_t* F, integer_t ldF, integer_t row,
     integer_t nr_rows, integer_t col, integer_t nr_cols,
     const integer_t* upd) const = 0;
    virtual void extract_F21_block
    (scalar_t* F, integer_t ldF, integer_t row,
     integer_t nr_rows, integer_t col, integer_t nr_cols,
     const integer_t* upd) const = 0;
    virtual void extract_separator_2d
    (integer_t sep_end, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J, DistM_t& B) const = 0;
    virtual void front_multiply_2d
    (integer_t sep_begin, integer_t sep_end,
     const std::vector<integer_t>& upd, const DistM_t& R,
     DistM_t& Srow, DistM_t& Scol, int depth) const = 0;
#endif //STRUMPACK_USE_MPI
#endif //DOXYGEN_SHOULD_SKIP_THIS

  protected:
    integer_t n_;
    integer_t nnz_;
    std::vector<integer_t> ptr_;
    std::vector<integer_t> ind_;
    std::vector<scalar_t> val_;
    bool symm_sparse_;

    enum MMsym {GENERAL, SYMMETRIC, SKEWSYMMETRIC, HERMITIAN};

    CompressedSparseMatrix();
    CompressedSparseMatrix
    (integer_t n, integer_t nnz, bool symm_sparse=false);
    CompressedSparseMatrix
    (integer_t n, const integer_t* row_ptr, const integer_t* col_ind,
     const scalar_t* values, bool symm_sparsity);

    std::vector<std::tuple<integer_t,integer_t,scalar_t>>
    read_matrix_market_entries(const std::string& filename);

    long long spmv_flops() const {
      return (is_complex<scalar_t>() ? 4 : 1 ) * (2ll * nnz_ - n_);
    }
    long long spmv_bytes() const {
      // read   ind  nnz  integer_t
      //        val  nnz  scalar_t
      //        ptr  n    integer_t
      //        x    n    scalar_t
      //        y    n    scalar_t
      // write  y    n    scalar_t
      return (sizeof(scalar_t) * 3 + sizeof(integer_t)) * n_
        + (sizeof(scalar_t) + sizeof(integer_t)) * nnz_;
    }
  };

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>::CompressedSparseMatrix()
    : n_(0), nnz_(0), symm_sparse_(false) {
  }

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>::CompressedSparseMatrix
  (integer_t n, integer_t nnz, bool symm_sparse)
    : n_(n), nnz_(nnz), symm_sparse_(symm_sparse) {
    ptr_.resize(n_+1);
    ind_.resize(nnz_);
    val_.resize(nnz_);
    ptr_[0] = 0;
    ptr_[n_] = nnz_;
  }

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>::CompressedSparseMatrix
  (integer_t n, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, bool symm_sparsity)
    : n_(n), nnz_(row_ptr[n_]-row_ptr[0]), ptr_(n_+1), ind_(nnz_), val_(nnz_),
      symm_sparse_(symm_sparsity) {
    if (row_ptr) std::copy(row_ptr, row_ptr+n_+1, ptr());
    if (col_ind) std::copy(col_ind, col_ind+nnz_, ind());
    if (values) std::copy(values, values+nnz_, val());
  }

  template<typename scalar_t,typename integer_t> void
  CompressedSparseMatrix<scalar_t,integer_t>::print() const {
    std::cout << "size: " << size() << std::endl;
    std::cout << "nnz: " << nnz() << std::endl;
    std::cout << "ptr: " << std::endl << "\t";
    for (integer_t i=0; i<=size(); i++)
      std::cout << ptr_[i] << " ";
    std::cout << std::endl << "ind: ";
    for (integer_t i=0; i<nnz(); i++)
      std::cout << ind_[i] << " ";
    std::cout << std::endl << "val: ";
    for (integer_t i=0; i<nnz(); i++)
      std::cout << val_[i] << " ";
    std::cout << std::endl;
  }

  extern "C" {
    int_t strumpack_mc64id_(int_t*);
    int_t strumpack_mc64ad_
    (int_t*, int_t*, int_t*, int_t*, int_t*, double*, int_t*, int_t*, int_t*,
     int_t*, int_t*, double*, int_t*, int_t*);
  }

  template<typename scalar_t,typename integer_t> int
  CompressedSparseMatrix<scalar_t,integer_t>::permute_and_scale
  (MatchingJob job, std::vector<integer_t>& perm, std::vector<scalar_t>& Dr,
   std::vector<scalar_t>& Dc, bool apply) {
    if (job == MatchingJob::NONE) return 1;
    if (job == MatchingJob::COMBBLAS) {
      std::cerr << "# ERROR: CombBLAS matching only supported in parallel."
                << std::endl;
      return 1;
    }
    perm.resize(n_);
    int_t liw = 0;
    switch (job) {
    case MatchingJob::MAX_SMALLEST_DIAGONAL: liw = 4*n_; break;
    case MatchingJob::MAX_SMALLEST_DIAGONAL_2: liw = 10*n_ + nnz_; break;
    case MatchingJob::MAX_CARDINALITY:
    case MatchingJob::MAX_DIAGONAL_SUM:
    case MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING:
    default: liw = 5*n_;
    }
    auto iw = new int_t[liw];
    int_t ldw = 0;
    switch (job) {
    case MatchingJob::MAX_CARDINALITY: ldw = 0; break;
    case MatchingJob::MAX_SMALLEST_DIAGONAL: ldw = n_; break;
    case MatchingJob::MAX_SMALLEST_DIAGONAL_2: ldw = nnz_; break;
    case MatchingJob::MAX_DIAGONAL_SUM: ldw = 2*n_ + nnz_; break;
    case MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING:
    default: ldw = 3*n_ + nnz_; break;
    }
    auto dw = new double[ldw];
    int_t icntl[10], info[10];
    int_t num;
    strumpack_mc64id_(icntl);
    //icntl[2] = 6; // print diagnostics
    //icntl[3] = 1; // no checking of input should be (slightly) faster
    switch (job) {
    case MatchingJob::MAX_CARDINALITY:
      strumpack_mc64(1, &num, perm.data(), liw, iw, ldw, dw, icntl, info);
      break;
    case MatchingJob::MAX_SMALLEST_DIAGONAL:
      strumpack_mc64(2, &num, perm.data(), liw, iw, ldw, dw, icntl, info);
      break;
    case MatchingJob::MAX_SMALLEST_DIAGONAL_2:
      strumpack_mc64(3, &num, perm.data(), liw, iw, ldw, dw, icntl, info);
      break;
    case MatchingJob::MAX_DIAGONAL_SUM:
      strumpack_mc64(4, &num, perm.data(), liw, iw, ldw, dw, icntl, info);
      break;
    case MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING:
      strumpack_mc64(5, &num, perm.data(), liw, iw, ldw, dw, icntl, info);
      break;
    default: break;
    }
    switch (info[0]) {
    case  0: break;
    case  1:
      std::cerr << "# ERROR: matrix is structurally singular" << std::endl;
      delete[] dw;
      delete[] iw;
      return 1;
      break;
    case  2:
        std::cerr << "# WARNING: mc64 scaling produced"
                  << " large scaling factors which may cause overflow!"
                  << std::endl;
      break;
    default:
      std::cerr << "# ERROR: mc64 failed with info[0]=" << info[0]
                << std::endl;
      delete[] dw;
      delete[] iw;
      return 1;
      break;
    }
    if (job == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING) { // scaling
      Dr.resize(n_);
      Dc.resize(n_);
#pragma omp parallel for
      for (integer_t i=0; i<n_; i++) {
        Dr[i] = scalar_t(std::exp(dw[i]));
        Dc[i] = scalar_t(std::exp(dw[n_+i]));
      }
      if (apply) apply_scaling(Dr, Dc);
    }
    delete[] iw; delete[] dw;
    if (apply) apply_column_permutation(perm);
    if (apply) symm_sparse_ = false;
    return 0;
  }

  template<typename scalar_t,typename integer_t> void
  CompressedSparseMatrix<scalar_t,integer_t>::symmetrize_sparsity() {
    if (symm_sparse_) return;
    std::vector<integer_t> a2_ctr(n_);
#pragma omp parallel for
    for (integer_t i=0; i<n_; i++)
      a2_ctr[i] = ptr_[i+1]-ptr_[i];

    bool change = false;
#pragma omp parallel for
    for (integer_t i=0; i<n_; i++)
      for (integer_t jj=ptr_[i]; jj<ptr_[i+1]; jj++) {
        integer_t kb = ptr_[ind_[jj]], ke = ptr_[ind_[jj]+1];
        if (std::find(ind()+kb, ind()+ke, i) == ind()+ke) {
#pragma omp critical
          {
            a2_ctr[ind_[jj]]++;
            change = true;
          }
        }
      }
    if (change) {
      std::vector<integer_t> a2_ptr(n_+1);
      a2_ptr[0] = 0;
      for (integer_t i=0; i<n_; i++) a2_ptr[i+1] = a2_ptr[i] + a2_ctr[i];
      auto new_nnz = a2_ptr[n_] - a2_ptr[0];
      std::vector<integer_t> a2_ind(new_nnz);
      std::vector<scalar_t> a2_val(new_nnz);
      nnz_ = new_nnz;
#pragma omp parallel for
      for (integer_t i=0; i<n_; i++) {
        a2_ctr[i] = a2_ptr[i] + ptr_[i+1] - ptr_[i];
        for (integer_t jj=ptr_[i], k=a2_ptr[i]; jj<ptr_[i+1]; jj++) {
          a2_ind[k  ] = ind_[jj];
          a2_val[k++] = val_[jj];
        }
      }
#pragma omp parallel for
      for (integer_t i=0; i<n_; i++)
        for (integer_t jj=ptr_[i]; jj<ptr_[i+1]; jj++) {
          integer_t kb = ptr_[ind_[jj]], ke = ptr_[ind_[jj]+1];
          if (std::find(ind()+kb,ind()+ke, i) == ind()+ke) {
            integer_t t = ind_[jj];
#pragma omp critical
            {
              a2_ind[a2_ctr[t]] = i;
              a2_val[a2_ctr[t]] = scalar_t(0.);
              a2_ctr[t]++;
            }
          }
        }
      std::swap(ptr_, a2_ptr);
      std::swap(ind_, a2_ind);
      std::swap(val_, a2_val);
    }
    symm_sparse_ = true;
  }

  template<typename scalar_t> scalar_t
  get_scalar(double vr, double vi) {
    return scalar_t(vr);
  }
  template<> inline std::complex<double>
  get_scalar(double vr, double vi) {
    return std::complex<double>(vr, vi);
  }
  template<> inline std::complex<float>
  get_scalar(double vr, double vi) {
    return std::complex<float>(vr, vi);
  }

  template<typename scalar_t,typename integer_t>
  std::vector<std::tuple<integer_t,integer_t,scalar_t>>
  CompressedSparseMatrix<scalar_t,integer_t>::read_matrix_market_entries
  (const std::string& filename) {
    std::cout << "# opening file \'" << filename << "\'" << std::endl;
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp == NULL) {
      std::cerr << "ERROR: could not read file";
      exit(1);
    }
    const int max_cline = 256;
    char cline[max_cline];
    if (fgets(cline, max_cline, fp) == NULL) {
      std::cerr << "ERROR: could not read from file" << std::endl;
      exit(1);
    }
    printf("# %s", cline);
    if (strstr(cline, "pattern")) {
      std::cerr << "ERROR: This is not a matrix,"
                << " but just a sparsity pattern" << std::endl;
      exit(1);
    }
    else if (strstr(cline, "complex")) {
      if (!is_complex<scalar_t>()) {
        fclose(fp);
        throw "ERROR: Complex matrix";
      }
    }
    MMsym s = GENERAL;
    if (strstr(cline, "skew-symmetric")) {
      s = SKEWSYMMETRIC;
      symm_sparse_ = true;
    } else if (strstr(cline, "symmetric")) {
      s = SYMMETRIC;
      symm_sparse_ = true;
    } else if (strstr(cline, "hermitian")) {
      s = HERMITIAN;
      symm_sparse_ = true;
    }

    while (fgets(cline, max_cline, fp)) {
      if (cline[0] != '%') { // first line should be: m n nnz
        int m, in, innz;
        sscanf(cline, "%d %d %d", &m, &in, &innz);
        nnz_ = static_cast<integer_t>(innz);
        n_ = static_cast<integer_t>(in);
        std::cout << "# reading " << number_format_with_commas(m) << " by "
                  << number_format_with_commas(n_) << " matrix with "
                  << number_format_with_commas(nnz_) << " nnz's from "
                  << filename << std::endl;
        if (s != GENERAL) nnz_ = 2 * nnz_;
        if (m != n_) {
          std::cerr << "ERROR: matrix is not square!" << std::endl;
          exit(1);
        }
        break;
      }
    }
    std::vector<std::tuple<integer_t,integer_t,scalar_t>> A;
    A.reserve(nnz_);
    bool zero_based = false;
    if (!is_complex<scalar_t>()) {
      int ir, ic;
      double dv;
      while (fscanf(fp, "%d %d %lf\n", &ir, &ic, &dv) != EOF) {
        scalar_t v = static_cast<scalar_t>(dv);
        integer_t r = static_cast<integer_t>(ir);
        integer_t c = static_cast<integer_t>(ic);
        if (r==0 || c==0) zero_based = true;
        A.push_back(std::make_tuple(r, c, v));
        if (r != c) {
          switch (s) {
          case SKEWSYMMETRIC:
            A.push_back(std::make_tuple(c, r, -v));
            break;
          case SYMMETRIC:
            A.push_back(std::make_tuple(c, r, v));
            break;
          case HERMITIAN:
            A.push_back(std::make_tuple(c, r, blas::my_conj(v)));
            break;
          default: break;
          }
        }
      }
    } else {
      double vr=0, vi=0;
      int ir, ic;
      while (fscanf(fp, "%d %d %lf %lf\n", &ir, &ic, &vr, &vi) != EOF) {
        scalar_t v = get_scalar<scalar_t>(vr, vi);
        integer_t r = static_cast<integer_t>(ir);
        integer_t c = static_cast<integer_t>(ic);
        if (r==0 || c==0) zero_based = true;
        A.push_back(std::make_tuple(r, c, v));
        if (r != c) {
          switch (s) {
          case SKEWSYMMETRIC:
            A.push_back(std::make_tuple(c, r, -v));
            break;
          case SYMMETRIC:
            A.push_back(std::make_tuple(c, r, v));
            break;
          case HERMITIAN:
            A.push_back(std::make_tuple(c, r, blas::my_conj(v)));
            break;
          default: break;
          }
        }
      }
    }
    nnz_ = A.size();
    fclose(fp);
    if (!zero_based)
      for (auto& t : A) {
        std::get<0>(t)--;
        std::get<1>(t)--;
      }
    return A;
  }

  template<typename scalar_t,typename integer_t> void
  CompressedSparseMatrix<scalar_t,integer_t>::permute
  (const integer_t* iorder, const integer_t* order) {
    std::vector<integer_t> new_ptr(n_+1);
    std::vector<integer_t> new_ind(nnz_);
    std::vector<scalar_t> new_val(nnz_);
    integer_t nnz = 0;
    new_ptr[0] = 0;
    for (integer_t i=0; i<n_; i++) {
      auto lb = ptr_[iorder[i]];
      auto ub = ptr_[iorder[i]+1];
      for (integer_t j=lb; j<ub; j++) {
        new_ind[nnz] = order[ind_[j]];
        new_val[nnz++] = val_[j];
      }
      new_ptr[i+1] = nnz;
    }
#pragma omp parallel for
    for (integer_t i=0; i<n_; i++) {
      auto lb = new_ptr[i];
      auto ub = new_ptr[i+1];
      sort_indices_values
        (new_ind.data()+lb, new_val.data()+lb, integer_t(0), ub-lb);
    }
    std::swap(ptr_, new_ptr);
    std::swap(ind_, new_ind);
    std::swap(val_, new_val);
  }

} //end namespace strumpack

#endif
