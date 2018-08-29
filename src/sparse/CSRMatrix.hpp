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
 * \file CSRMatrix.hpp
 * \brief Contains the compressed sparse row matrix storage class.
 */
#ifndef CSRMATRIX_HPP
#define CSRMATRIX_HPP
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include "misc/Tools.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "dense/DistributedMatrix.hpp"
#endif
#include "dense/DenseMatrix.hpp"
#include "CompressedSparseMatrix.hpp"

namespace strumpack {

  /**
   * \class CSRMatrix
   * \brief Class for storing a compressed sparse row matrix (single
   * node).
   *
   * \tparam scalar_t
   * \tparam integer_t
   *
   * \see CompressedSparseMatrix, CSRMatrixMPI
   */
  template<typename scalar_t,typename integer_t> class CSRMatrix
    : public CompressedSparseMatrix<scalar_t,integer_t> {
#if defined(STRUMPACK_USE_MPI)
    using DistM_t = DistributedMatrix<scalar_t>;
#endif
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using real_t = typename RealType<scalar_t>::value_type;

  public:
    CSRMatrix();
    CSRMatrix(integer_t n, integer_t nnz);
    CSRMatrix
    (integer_t n, const integer_t* ptr, const integer_t* ind,
     const scalar_t* values, bool symm_sparsity=false);

    void spmv(const DenseM_t& x, DenseM_t& y) const override;
    void spmv(const scalar_t* x, scalar_t* y) const override;

    void apply_scaling
    (const std::vector<scalar_t>& Dr,
     const std::vector<scalar_t>& Dc) override;
    void apply_column_permutation
    (const std::vector<integer_t>& perm) override;
    real_t max_scaled_residual
    (const scalar_t* x, const scalar_t* b) const override;
    real_t max_scaled_residual
    (const DenseM_t& x, const DenseM_t& b) const override;
    void strumpack_mc64
    (int_t job, int_t* num, integer_t* perm, int_t liw, int_t* iw, int_t ldw,
     double* dw, int_t* icntl, int_t* info) override;
    int read_matrix_market(const std::string& filename) override;
    int read_binary(const std::string& filename);
    void print_dense(const std::string& name) const override;
    void print_MM(const std::string& filename) const override;
    void print_binary(const std::string& filename) const;


#ifndef DOXYGEN_SHOULD_SKIP_THIS
    // TODO implement these outside of this class
    void front_multiply
    (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
     const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc, int depth) const override;
    void extract_separator
    (integer_t sep_end, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J,
     DenseM_t& B, int depth) const override;
    void extract_front
    (DenseM_t& F11, DenseM_t& F12, DenseM_t& F21, integer_t sep_begin,
     integer_t sep_end, const std::vector<integer_t>& upd,
     int depth) const override;

#if defined(STRUMPACK_USE_MPI)
    void extract_F11_block
    (scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows,
     integer_t col, integer_t nr_cols) const override;
    void extract_F12_block
    (scalar_t* F, integer_t ldF, integer_t row,
     integer_t nr_rows, integer_t col, integer_t nr_cols,
     const integer_t* upd) const override;
    void extract_F21_block
    (scalar_t* F, integer_t ldF, integer_t row,
     integer_t nr_rows, integer_t col, integer_t nr_cols,
     const integer_t* upd) const;
    void extract_separator_2d
    (integer_t sep_end, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J, DistM_t& B) const override;
    void front_multiply_2d
    (integer_t sep_begin, integer_t sep_end,
     const std::vector<integer_t>& upd, const DistM_t& R,
     DistM_t& Srow, DistM_t& Scol, int depth) const override;
#endif //defined(STRUMPACK_USE_MPI)
#endif //DOXYGEN_SHOULD_SKIP_THIS

    using CompressedSparseMatrix<scalar_t,integer_t>::n_;
    using CompressedSparseMatrix<scalar_t,integer_t>::nnz_;
    using CompressedSparseMatrix<scalar_t,integer_t>::ptr_;
    using CompressedSparseMatrix<scalar_t,integer_t>::ind_;
    using CompressedSparseMatrix<scalar_t,integer_t>::val_;
    using CompressedSparseMatrix<scalar_t,integer_t>::symm_sparse_;
  };

  template<typename scalar_t,typename integer_t>
  CSRMatrix<scalar_t,integer_t>::CSRMatrix()
    : CompressedSparseMatrix<scalar_t,integer_t>() {}

  template<typename scalar_t,typename integer_t>
  CSRMatrix<scalar_t,integer_t>::CSRMatrix
  (integer_t n, const integer_t* ptr, const integer_t* ind,
   const scalar_t* values, bool symm_sparsity)
    : CompressedSparseMatrix<scalar_t,integer_t>
    (n, ptr, ind, values, symm_sparsity) { }

  template<typename scalar_t,typename integer_t>
  CSRMatrix<scalar_t,integer_t>::CSRMatrix(integer_t n, integer_t nnz)
    : CompressedSparseMatrix<scalar_t,integer_t>(n, nnz) {}


  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::print_dense(const std::string& name) const {
    auto M = new scalar_t[n_ * n_];
    std::fill(M, M+(n_*n_), scalar_t(0.));
    for (integer_t row=0; row<n_; row++)
      for (integer_t j=ptr_[row]; j<ptr_[row+1]; j++)
        M[row + ind_[j]*n_] = val_[j];
    std::cout << name << " = [";
    for (integer_t row=0; row<n_; row++) {
      for (integer_t col=0; col<n_; col++)
        std::cout << M[row + n_ * col] << " ";
      std::cout << ";" << std::endl;
    }
    std::cout << "];" << std::endl;
    delete[] M;
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::print_MM
  (const std::string& filename) const {
    std::fstream fs(filename, std::fstream::out);
    if (is_complex<scalar_t>())
      fs << "%%MatrixMarket matrix coordinate complex general\n";
    else
      fs << "%%MatrixMarket matrix coordinate real general\n";
    fs << n_ << " " << n_ << " " << nnz_ << "\n";
    fs.precision(17);
    if (is_complex<scalar_t>()) {
      for (integer_t row=0; row<n_; row++)
        for (integer_t j=ptr_[row]; j<ptr_[row+1]; j++)
          fs << row+1 << " " << ind_[j]+1 << " "
             << std::real(val_[j]) << " "
             << std::imag(val_[j]) << "\n";
    } else {
      for (integer_t row=0; row<n_; row++)
        for (integer_t j=ptr_[row]; j<ptr_[row+1]; j++)
          fs << row+1 << " " << ind_[j]+1 << " "
             << val_[j] << "\n";
    }
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::print_binary
  (const std::string& filename) const {
    std::ofstream fs(filename, std::ofstream::binary);
    char s = 'R';
    fs.write(&s, sizeof(char));
    if (std::is_same<integer_t,int>()) s = '4';
    else if (std::is_same<integer_t,int64_t>()) s = '8';
    fs.write(&s, sizeof(char));
    if (is_complex<scalar_t>()) {
      if (std::is_same<real_t,float>()) s = 'c';
      else if (std::is_same<real_t,double>()) s = 'z';
    } else {
      if (std::is_same<real_t,float>()) s = 's';
      else if (std::is_same<real_t,double>()) s = 'd';
    }
    fs.write(&s, sizeof(char));
    fs.write((char*)&n_, sizeof(integer_t));
    fs.write((char*)&n_, sizeof(integer_t));
    fs.write((char*)&nnz_, sizeof(integer_t));

    for (integer_t i=0; i<n_+1; i++)
      fs.write((char*)(&ptr_[i]), sizeof(integer_t));
    for (integer_t i=0; i<nnz_; i++)
      fs.write((char*)(&ind_[i]), sizeof(integer_t));
    for (integer_t i=0; i<nnz_; i++)
      fs.write((char*)(&(val_[i])), sizeof(scalar_t));

    if (!fs.good()) {
      std::cout << "Error writing to file !!" << std::endl;
      std::cout << "failbit = " << fs.fail() << std::endl;
      std::cout << "eofbit  = " << fs.eof() << std::endl;
      std::cout << "badbit  = " << fs.bad() << std::endl;
    }
    std::cout << "Wrote " << fs.tellp() << " bytes to file "
              << filename << std::endl;
    fs.close();
  }

  template<typename scalar_t,typename integer_t> int
  CSRMatrix<scalar_t,integer_t>::read_binary(const std::string& filename) {
    std::ifstream fs(filename, std::ifstream::in | std::ifstream::binary);
    char s;
    fs.read(&s, sizeof(s));
    if (s != 'R') {
      std::cerr << "Error: matrix is not in binary CSR format." << std::endl;
      return 1;//throw "Error: matrix is not in binary CSR format.";
    }
    fs.read(&s, sizeof(s));
    if (sizeof(integer_t) != s-'0') {
      std::cerr << "Error: matrix integer_t type does not match,"
        " input matrix uses " << (s-'0') << " bytes per integer."
                << std::endl;
      //throw "Error: matrix integer_t type does not match input matrix.";
      return 1;
    }
    fs.read(&s, sizeof(s));
    if ((!is_complex<scalar_t>() && std::is_same<real_t,float>() && s!='s') ||
        (!is_complex<scalar_t>() && std::is_same<real_t,double>() && s!='d') ||
        (is_complex<scalar_t>() && std::is_same<real_t,float>() && s!='c') ||
        (is_complex<scalar_t>() && std::is_same<real_t,double>() && s!='z')) {
      std::cerr << "Error: scalar type of input matrix does not match,"
        " input matrix is of type " << s << std::endl;
      return 1;//throw "Error: scalar type of input matrix does not match";
    }
    fs.read((char*)&n_, sizeof(integer_t));
    fs.read((char*)&n_, sizeof(integer_t));
    fs.read((char*)&nnz_, sizeof(integer_t));
    std::cout << "# Reading matrix with n="
              << number_format_with_commas(n_)
              << ", nnz=" << number_format_with_commas(nnz_)
              << std::endl;
    symm_sparse_ = false;
    ptr_.resize(n_+1);
    ind_.resize(nnz_);
    val_.resize(nnz_);
    for (integer_t i=0; i<n_+1; i++)
      fs.read((char*)(&ptr_[i]), sizeof(integer_t));
    for (integer_t i=0; i<nnz_; i++)
      fs.read((char*)(&ind_[i]), sizeof(integer_t));
    for (integer_t i=0; i<nnz_; i++)
      fs.read((char*)(&val_[i]), sizeof(scalar_t));
    fs.close();
    return 0;
  }

#if defined(__INTEL_MKL__)
  // TODO test this, enable from CMake
  template<> void CSRMatrix<float>::spmv(const float* x, float* y) const {
    char trans = 'N';
    mkl_cspblas_scsrgemv(&no, &n, val_, ptr_, ind_, x, y);
    STRUMPACK_FLOPS(this->spmv_flops());
    STRUMPACK_BYTES(this->spmv_bytes());
  }
  template<> void CSRMatrix<double>::spmv(const double* x, double* y) const {
    char trans = 'N';
    mkl_cspblas_dcsrgemv(&no, &n, val_, ptr_, ind_, x, y);
    STRUMPACK_FLOPS(this->spmv_flops());
    STRUMPACK_BYTES(this->spmv_bytes());
  }
  template<> void CSRMatrix<std::complex<float>>::spmv
  (const std::complex<float>* x, std::complex<float>* y) const {
    char trans = 'N';
    mkl_cspblas_ccsrgemv(&no, &n, val_, ptr_, ind_, x, y);
    STRUMPACK_FLOPS(this->spmv_flops());
    STRUMPACK_BYTES(this->spmv_bytes());
  }
  template<> void CSRMatrix<std::complex<double>>::spmv
  (const std::complex<double>* x, std::complex<double>* y) const {
    char trans = 'N';
    mkl_cspblas_zcsrgemv(&no, &n, val_, ptr_, ind_, x, y);
    STRUMPACK_FLOPS(this->spmv_flops());
    STRUMPACK_BYTES(this->spmv_bytes());
  }
#endif

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::spmv
  (const scalar_t* x, scalar_t* y) const {
#pragma omp parallel for
    for (integer_t r=0; r<n_; r++) {
      const auto hij = ptr_[r+1];
      scalar_t yr(0);
      for (integer_t j=ptr_[r]; j<hij; j++)
        yr += val_[j] * x[ind_[j]];
      y[r] = yr;
    }
    STRUMPACK_FLOPS(this->spmv_flops());
    STRUMPACK_BYTES(this->spmv_bytes());
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::spmv
  (const DenseM_t& x, DenseM_t& y) const {
    assert(x.cols() == y.cols());
    assert(x.rows() == std::size_t(n_));
    assert(y.rows() == std::size_t(n_));
    for (std::size_t c=0; c<x.cols(); c++)
      spmv(x.ptr(0,c), y.ptr(0,c));
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::strumpack_mc64
  (int_t job, int_t* num, integer_t* perm, int_t liw, int_t* iw, int_t ldw,
   double* dw, int_t* icntl, int_t* info) {
    int_t n = n_;
    int_t nnz = nnz_;
    double* dval = new double[nnz];
    int_t* col_ptr = new int_t[n+1+nnz+n+n];
    int_t* row_ind = col_ptr + n + 1;
    int_t* permutation = row_ind + nnz;
    int_t* rowsums = permutation + n;
    for (int_t i=0; i<n; i++) rowsums[i] = 0;
    for (int_t col=0; col<n; col++)
      for (int_t i=ptr_[col]; i<ptr_[col+1]; i++)
        rowsums[ind_[i]]++;
    col_ptr[0] = 1;  // start from 1, because mc64 is fortran!
    for (int_t r=0; r<n; r++) col_ptr[r+1] = col_ptr[r] + rowsums[r];
    for (int_t i=0; i<n; i++) rowsums[i] = 0;
    for (int_t col=0; col<n; col++) {
      for (int_t i=ptr_[col]; i<ptr_[col+1]; i++) {
        int_t row = ind_[i];
        int_t j = col_ptr[row] - 1 + rowsums[row]++;
        if (is_complex<scalar_t>())
          dval[j] = static_cast<double>(std::abs(val_[i]));
        else dval[j] = static_cast<double>(std::real(val_[i]));
        row_ind[j] = col + 1;
      }
    }
    strumpack_mc64ad_
      (&job, &n, &nnz, col_ptr, row_ind, dval, num,
       permutation, &liw, iw, &ldw, dw, icntl, info);
    for (int_t i=0; i<n; i++) perm[i] = permutation[i] - 1;
    delete[] col_ptr;
    delete[] dval;
  }

  // TODO tasked!!
  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::extract_separator
  (integer_t sep_end, const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J, DenseM_t& B, int depth) const {
    const integer_t m = I.size();
    const integer_t n = J.size();
    if (m == 0 || n == 0) return;
    for (integer_t i=0; i<m; i++) {
      integer_t r = I[i];
      // indices sorted in increasing order
      auto cmin = ind_[ptr_[r]];
      auto cmax = ind_[ptr_[r+1]-1];
      for (integer_t k=0; k<n; k++) {
        integer_t c = J[k];
        if (c >= cmin && c <= cmax && (r < sep_end || c < sep_end)) {
          auto a_pos = ptr_[r];
          auto a_max = ptr_[r+1];
          while (a_pos < a_max-1 && ind_[a_pos] < c) a_pos++;
          B(i,k) = (ind_[a_pos] == c) ?
            val_[a_pos] : scalar_t(0.);
        } else B(i,k) = scalar_t(0.);
      }
    }
  }

  // TODO parallel -> will be hard to do efficiently
  // assume F11, F12 and F21 are set to zero
  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::extract_front
  (DenseM_t& F11, DenseM_t& F12, DenseM_t& F21, integer_t sep_begin,
   integer_t sep_end, const std::vector<integer_t>& upd, int depth) const {
    integer_t dim_sep = sep_end - sep_begin;
    integer_t dim_upd = upd.size();
    for (integer_t row=0; row<dim_sep; row++) { // separator rows
      integer_t upd_ptr = 0;
      const auto hij = ptr_[row+sep_begin+1];
      for (integer_t j=ptr_[row+sep_begin];
           j<hij; j++) {
        integer_t col = ind_[j];
        if (col >= sep_begin) {
          if (col < sep_end)
            F11(row, col-sep_begin) = val_[j];
          else {
            while (upd_ptr<dim_upd && upd[upd_ptr]<col)
              upd_ptr++;
            if (upd_ptr == dim_upd) break;
            if (upd[upd_ptr] == col)
              F12(row, upd_ptr) = val_[j];
          }
        }
      }
    }
    for (integer_t i=0; i<dim_upd; i++) { // update rows
      auto row = upd[i];
      const auto hij = ptr_[row+1];
      for (integer_t j=ptr_[row]; j<hij; j++) {
        integer_t col = ind_[j];
        if (col >= sep_begin) {
          if (col < sep_end)
            F21(i, col-sep_begin) = val_[j];
          else break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::front_multiply
  (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc, int depth) const {
    integer_t dupd = upd.size();
    //long long int local_flops = 0;
    const integer_t nbvec = R.cols();
    const integer_t ds = shi - slo;

    const auto B = 4; // blocking parameter
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)                    \
  if(depth < params::task_recursion_cutoff_level)
#endif
    for (integer_t c=0; c<nbvec; c+=B) {
      for (auto row=slo; row<shi; row++) { // separator rows
        integer_t upd_ptr = 0;
        const auto hij = ptr_[row+1];
        for (auto j=ptr_[row]; j<hij; j++) {
          const auto col = ind_[j];
          if (col >= slo) {
            const auto hicc = std::min(c+B, nbvec);
            const auto vj = val_[j];
            if (col < shi) {
              for (integer_t cc=c; cc<hicc; cc++) {
                Sr(row-slo, cc) += vj * R(col-slo, cc);
                Sc(col-slo, cc) += vj * R(row-slo, cc);
              }
              //local_flops += 4 * B;
            } else {
              while (upd_ptr<dupd && upd[upd_ptr]<col) upd_ptr++;
              if (upd_ptr == dupd) break;
              if (upd[upd_ptr] == col) {
                for (integer_t cc=c; cc<hicc; cc++) {
                  Sr(row-slo, cc) += vj * R(ds+upd_ptr, cc);
                  Sc(ds+upd_ptr, cc) += vj * R(row-slo, cc);
                }
                //local_flops += 4 * B;
              }
            }
          }
        }
      }
    }

#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)                    \
  if(depth < params::task_recursion_cutoff_level)
#endif
    for (integer_t c=0; c<nbvec; c+=B) {
      for (integer_t i=0; i<dupd; i++) { // remaining rows
        auto row = upd[i];
        const auto hij = ptr_[row+1];
        for (auto j=ptr_[row]; j<hij; j++) {
          auto col = ind_[j];
          if (col >= slo) {
            if (col < shi) {
              const auto vj = val_[j];
              const auto hicc = std::min(c+B, nbvec);
              for (integer_t cc=c; cc<hicc; cc++) {
                Sr(ds+i, cc) += vj * R(col-slo, cc);
                Sc(col-slo, cc) += vj * R(ds+i, cc);
              }
              //#pragma omp atomic
              //local_flops += 4 * B;
            } else break;
          }
        }
      }
    }
    // STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    // STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::apply_scaling
  (const std::vector<scalar_t>& Dr, const std::vector<scalar_t>& Dc) {
#pragma omp parallel for
    for (integer_t j=0; j<n_; j++)
      for (integer_t i=ptr_[j]; i<ptr_[j+1]; i++)
        val_[i] = val_[i] * Dr[j] * Dc[ind_[i]];
    STRUMPACK_FLOPS
      ((is_complex<scalar_t>()?6:1)*
       static_cast<long long int>(2.*double(this->nnz())));
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::apply_column_permutation
  (const std::vector<integer_t>& perm) {
    integer_t* iperm = new integer_t[n_];
    for (integer_t i=0; i<n_; i++) iperm[perm[i]] = i;
#pragma omp parallel for
    for (integer_t row=0; row<n_; row++) {
      for (integer_t i=ptr_[row]; i<ptr_[row+1]; i++)
        ind_[i] = iperm[ind_[i]];
      sort_indices_values<scalar_t>
        (&ind_[ptr_[row]], &val_[ptr_[row]],
         integer_t(0), ptr_[row+1]-ptr_[row]);
    }
    delete[] iperm;
  }

  template<typename scalar_t,typename integer_t> int
  CSRMatrix<scalar_t,integer_t>::read_matrix_market
  (const std::string& filename) {
    std::vector<std::tuple<integer_t,integer_t,scalar_t>> A;
    try {
      A = this->read_matrix_market_entries(filename);
    } catch (...) { return 1; }
    std::sort
      (A.begin(), A.end(),
       [](const std::tuple<integer_t,integer_t,scalar_t>& a,
          const std::tuple<integer_t,integer_t,scalar_t>& b) -> bool {
        // sort based on the row,column indices
        return std::make_tuple(std::get<0>(a),std::get<1>(a)) <
          std::make_tuple(std::get<0>(b), std::get<1>(b));
      });

    ptr_.resize(n_+1);
    ind_.resize(nnz_);
    val_.resize(nnz_);
    integer_t row = -1;
    for (integer_t i=0; i<nnz_; i++) {
      val_[i] = std::get<2>(A[i]);
      ind_[i] = std::get<1>(A[i]);
      auto new_row = std::get<0>(A[i]);
      if (new_row != row) {
        for (int j=row+1; j<=new_row; j++) ptr_[j] = i;
        row = new_row;
      }
    }
    for (int j=row+1; j<=n_; j++) ptr_[j] = nnz_;
    return 0;
  }

  template<typename scalar_t,typename integer_t> typename
  RealType<scalar_t>::value_type
  CSRMatrix<scalar_t,integer_t>::max_scaled_residual
  (const DenseM_t& x, const DenseM_t& b) const {
    real_t res = real_t(0.);
    const integer_t m = n_;
    const integer_t n = x.cols();
    for (integer_t c=0; c<n; c++) {
#pragma omp parallel for reduction(max:res)
      for (integer_t r=0; r<m; r++) {
        auto true_res = b(r, c);
        auto abs_res = std::abs(b(r, c));
        const auto hij = ptr_[r+1];
        for (integer_t j=ptr_[r]; j<hij; ++j) {
          const auto v = val_[j];
          const auto rj = ind_[j];
          true_res -= v * x(rj, c);
          abs_res += std::abs(v) * std::abs(x(rj,c));
        }
        res = std::max(res, std::abs(true_res) / std::abs(abs_res));
      }
    }
    return res;
  }

  template<typename scalar_t,typename integer_t> typename
  RealType<scalar_t>::value_type
  CSRMatrix<scalar_t,integer_t>::max_scaled_residual
  (const scalar_t* x, const scalar_t* b) const {
    auto X = ConstDenseMatrixWrapperPtr(n_, 1, x, n_);
    auto B = ConstDenseMatrixWrapperPtr(n_, 1, b, n_);
    return max_scaled_residual(*X, *B);
  }


#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::extract_separator_2d
  (integer_t sep_end, const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J, DistM_t& B) const {
    if (!B.active()) return;
    const integer_t m = I.size();
    const integer_t n = J.size();
    if (m == 0 || n == 0) return;
    B.zero();
    for (integer_t i=0; i<m; i++) {
      integer_t r = I[i];
      // indices sorted in increasing order
      auto cmin = ind_[ptr_[r]];
      auto cmax = ind_[ptr_[r+1]-1];
      for (integer_t k=0; k<n; k++) {
        integer_t c = J[k];
        if (c >= cmin && c <= cmax && (r < sep_end || c < sep_end)) {
          auto a_pos = ptr_[r];
          auto a_max = ptr_[r+1]-1;
          while (a_pos<a_max && ind_[a_pos]<c) a_pos++;
          if (ind_[a_pos] == c) B.global(i, k, val_[a_pos]);
        }
      }
    }
  }

  /** TODO this can be done more cleverly */
  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::front_multiply_2d
  (integer_t sep_begin, integer_t sep_end, const std::vector<integer_t>& upd,
   const DistM_t& R, DistM_t& Srow, DistM_t& Scol, int depth) const {
    integer_t dim_upd = upd.size();
    // redistribute R and Srow and Scol to 1d block cyclic column
    // distribution to avoid communication below
    DistM_t R_bc(R.grid(), R.rows(), R.cols(), R.rows(), R.NB());
    DistM_t Srow_bc(R.grid(), Srow.rows(), Srow.cols(), Srow.rows(), Srow.NB());
    DistM_t Scol_bc(R.grid(), Scol.rows(), Scol.cols(), Scol.rows(), Scol.NB());
    copy(R.rows(), R.cols(), R, 0, 0, R_bc, 0, 0, R.ctxt_all());
    copy(Srow.rows(), Srow.cols(), Srow, 0, 0, Srow_bc, 0, 0, R.ctxt_all());
    copy(Scol.rows(), Scol.cols(), Scol, 0, 0, Scol_bc, 0, 0, R.ctxt_all());

    if (R.active()) {
      long long int local_flops = 0;
      auto dim_sep = sep_end - sep_begin;
      auto n = R_bc.cols();
      for (integer_t row=sep_begin; row<sep_end; row++) { // separator rows
        auto upd_ptr = 0;
        auto hij = ptr_[row+1];
        for (integer_t j=ptr_[row]; j<hij; j++) {
          auto col = ind_[j];
          if (col >= sep_begin) {
            if (col < sep_end) {
              scalapack::pgeadd
                ('N', 1, n, val_[j], R_bc.data(), col-sep_begin+1, 1,
                 R_bc.desc(), scalar_t(1.), Srow_bc.data(), row-sep_begin+1,
                 1, Srow_bc.desc());
              scalapack::pgeadd
                ('N', 1, n, val_[j], R_bc.data(), row-sep_begin+1, 1,
                 R_bc.desc(), scalar_t(1.), Scol_bc.data(), col-sep_begin+1,
                 1, Scol_bc.desc());
              local_flops += 4 * n;
            } else {
              while (upd_ptr<dim_upd && upd[upd_ptr]<col) upd_ptr++;
              if (upd_ptr == dim_upd) break;
              if (upd[upd_ptr] == col) {
                scalapack::pgeadd
                  ('N', 1, n, val_[j], R_bc.data(), dim_sep+upd_ptr+1,
                   1, R_bc.desc(), scalar_t(1.), Srow_bc.data(),
                   row-sep_begin+1, 1, Srow_bc.desc());
                scalapack::pgeadd
                             ('N', 1, n, val_[j], R_bc.data(),
                              row-sep_begin+1, 1, R_bc.desc(), scalar_t(1.),
                              Scol_bc.data(), dim_sep+upd_ptr+1, 1,
                              Scol_bc.desc());
                local_flops += 4 * n;
              }
            }
          }
        }
      }
      for (integer_t i=0; i<dim_upd; i++) { // remaining rows
        integer_t row = upd[i];
        auto hij = ptr_[row+1];
        for (integer_t j=ptr_[row]; j<hij; j++) {
          integer_t col = ind_[j];
          if (col >= sep_begin) {
            if (col < sep_end) {
              scalapack::pgeadd
                ('N', 1, n, val_[j], R_bc.data(), col-sep_begin+1, 1,
                 R_bc.desc(), scalar_t(1.), Srow_bc.data(), dim_sep+i+1,
                 1, Srow_bc.desc());
              scalapack::pgeadd
                ('N', 1, n, val_[j], R_bc.data(), dim_sep+i+1, 1,
                 R_bc.desc(), scalar_t(1.), Scol_bc.data(), col-sep_begin+1,
                 1, Scol_bc.desc());
              local_flops += 4 * n;
            } else break;
          }
        }
      }
      if (R.is_master()) {
        STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
        STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
      }
    }
    copy(Srow.rows(), Srow.cols(), Srow_bc, 0, 0, Srow, 0, 0, R.ctxt_all());
    copy(Scol.rows(), Scol.cols(), Scol_bc, 0, 0, Scol, 0, 0, R.ctxt_all());
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::extract_F11_block
  (scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows,
   integer_t col, integer_t nr_cols) const {
    const auto rhi = std::min(row+nr_rows, n_);
    for (integer_t r=row; r<rhi; r++) {
      auto j = ptr_[r];
      const auto hij = ptr_[r+1];
      while (j<hij && ind_[j] < col) j++;
      for ( ; j<hij; j++) {
        integer_t c = ind_[j];
        if (c < col+nr_cols)
          F[r-row + (c-col)*ldF] = val_[j];
        else break;
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::extract_F12_block
  (scalar_t* F, integer_t ldF, integer_t row,
   integer_t nr_rows, integer_t col, integer_t nr_cols,
   const integer_t* upd) const {
    for (integer_t r=row; r<std::min(row+nr_rows, n_); r++) {
      integer_t upd_pos = 0;
      const auto hij = ptr_[r+1];
      for (integer_t j=ptr_[r]; j<hij; j++) {
        auto c = ind_[j];
        if (c >= col) {
          while (upd_pos<nr_cols && upd[upd_pos]<c) upd_pos++;
          if (upd_pos == nr_cols) break;
          if (upd[upd_pos] == c)
            F[r-row + upd_pos*ldF] = val_[j];
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::extract_F21_block
  (scalar_t* F, integer_t ldF, integer_t row,
   integer_t nr_rows, integer_t col, integer_t nr_cols,
   const integer_t* upd) const {
    auto rhi = std::min(row+nr_rows, n_);
    for (integer_t i=row; i<rhi; i++) {
      const auto r = upd[i-row];
      auto j = ptr_[r];
      const auto hij = ptr_[r+1];
      while (j<hij && ind_[j] < col) j++;
      for ( ; j<hij; j++) {
        auto c = ind_[j];
        if (c < col+nr_cols)
          F[i-row + (c-col)*ldF] = val_[j];
        else break;
      }
    }
  }
#endif

} // end namespace strumpack

#endif //CSRMATRIX_H
