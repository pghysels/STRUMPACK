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
#ifndef COMPRESSEDSPARSEMATRIX_HPP
#define COMPRESSEDSPARSEMATRIX_HPP

#include <vector>
#include <algorithm>
#include <tuple>
#include <stdio.h>
#include <string.h>
#include "misc/Tools.hpp"
#include "misc/MPIWrapper.hpp"

// where is this used?? in MC64?
#ifdef _LONGINT
  typedef long long int int_t;
#else // Default
  typedef int int_t;
#endif

namespace strumpack {

  template<typename scalar_t> class DistributedMatrix;
  template<typename scalar_t> class DenseMatrix;

  /**
   * Abstract base class to represent either compressed sparse row or
   * compressed sparse column matrices.  The rows and the columns
   * should always be sorted.
   */
  template<typename scalar_t,typename integer_t>
  class CompressedSparseMatrix {
    using DistM_t = DistributedMatrix<scalar_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using real_t = typename RealType<scalar_t>::value_type;

  public:
    CompressedSparseMatrix();
    CompressedSparseMatrix
    (integer_t n, integer_t nnz, bool symm_sparse=false);
    CompressedSparseMatrix
    (integer_t n, const integer_t* ptr, const integer_t* ind,
     const scalar_t* val, bool symm_sparsity);
    CompressedSparseMatrix
    (const CompressedSparseMatrix<scalar_t,integer_t>& A);
    CompressedSparseMatrix
    (CompressedSparseMatrix<scalar_t,integer_t>&& A);

    virtual ~CompressedSparseMatrix();

    CompressedSparseMatrix<scalar_t,integer_t>& operator=
    (const CompressedSparseMatrix<scalar_t,integer_t>& A);
    CompressedSparseMatrix<scalar_t,integer_t>& operator=
    (CompressedSparseMatrix<scalar_t,integer_t>&& A);

    inline integer_t size() const { return _n; }
    inline integer_t nnz() const { return _nnz; }
    inline integer_t* get_ptr() const { return _ptr; }
    inline integer_t* get_ind() const { return _ind; }
    inline scalar_t* get_val() const { return _val; }
    inline bool symm_sparse() const { return _symm_sparse; }
    inline void set_symm_sparse(bool symm_sparse=true) {
      _symm_sparse = symm_sparse;
    }

    virtual void spmv(const DenseM_t& x, DenseM_t& y) const = 0;
    virtual void omp_spmv(const DenseM_t& x, DenseM_t& y) const = 0;
    virtual void spmv(const scalar_t* x, scalar_t* y) const = 0;
    virtual void omp_spmv(const scalar_t* x, scalar_t* y) const = 0;

    virtual void permute(const integer_t* iorder, const integer_t* order);
    virtual void permute
    (const std::vector<integer_t>& iorder, std::vector<integer_t>& order) {
      permute(iorder.data(), order.data());
    }
    virtual int permute_and_scale
    (int job, std::vector<integer_t>& perm, std::vector<scalar_t>& Dr,
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

    // TODO implement these outside of this class
    virtual void extract_separator
    (integer_t sep_end, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J, DenseM_t& B, int depth) const = 0;
    virtual void extract_front
    (DenseM_t& F11, DenseM_t& F12, DenseM_t& F21,
     integer_t sep_begin, integer_t sep_end,
     const std::vector<integer_t>& upd, int depth) const = 0;
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
     const std::vector<std::size_t>& J, DistM_t& B, MPI_Comm comm) const = 0;
    virtual void front_multiply
    (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
     const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc) const = 0;
    virtual void front_multiply_2d
    (integer_t sep_begin, integer_t sep_end,
     const std::vector<integer_t>& upd,
     const DistM_t& R, DistM_t& Srow, DistM_t& Scol,
     int ctx_all, MPI_Comm R_comm, int depth) const = 0;

  protected:
    integer_t _n;
    integer_t _nnz;
    integer_t* _ptr;
    integer_t* _ind;
    scalar_t* _val;
    bool _symm_sparse;

    enum MMsym {GENERAL, SYMMETRIC, SKEWSYMMETRIC, HERMITIAN};
    std::vector<std::tuple<integer_t,integer_t,scalar_t>>
    read_matrix_market_entries(const std::string& filename);
    // void clone_data
    // (const CompressedSparseMatrix<scalar_t,integer_t>& A) const;
    inline void set_ptr(integer_t* new_ptr) { delete[] _ptr; _ptr = new_ptr; }
    inline void set_ind(integer_t* new_ind) { delete[] _ind; _ind = new_ind; }
    inline void set_val(scalar_t* new_val) { delete[] _val; _val = new_val; }
    virtual bool is_mpi_root() const { return mpi_root(); }

    long long spmv_flops() const {
      return (is_complex<scalar_t>() ? 4 : 1 ) *
        (2ll * this->_nnz - this->_n);
    }
    long long spmv_bytes() const {
      // read   ind  nnz  integer_t
      //        val  nnz  scalar_t
      //        ptr  n    integer_t
      //        x    n    scalar_t
      //        y    n    scalar_t
      // write  y    n    scalar_t
      return (sizeof(scalar_t) * 3 + sizeof(integer_t)) * this->_n
        + (sizeof(scalar_t) + sizeof(integer_t)) * this->_nnz;
    }
  };

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>::CompressedSparseMatrix()
    : _n(0), _nnz(0), _ptr(NULL), _ind(NULL), _val(NULL),
      _symm_sparse(false) {
  }

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>::CompressedSparseMatrix
  (integer_t n, integer_t nnz, bool symm_sparse)
    : _n(n), _nnz(nnz), _symm_sparse(symm_sparse) {
    _ptr = new integer_t[_n+1];
    _ind = new integer_t[_nnz];
    _val = new scalar_t[_nnz];
    _ptr[0] = 0;
    _ptr[_n] = _nnz;
  }

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>::CompressedSparseMatrix
  (integer_t n, const integer_t* ptr, const integer_t* ind,
   const scalar_t* val, bool symm_sparsity)
    : _n(n), _nnz(ptr[_n]-ptr[0]), _ptr(new integer_t[_n+1]),
      _ind(new integer_t[_nnz]), _val(new scalar_t[_nnz]),
      _symm_sparse(symm_sparsity) {
    if (ptr) std::copy(ptr, ptr+_n+1, _ptr);
    if (ind) std::copy(ind, ind+_nnz, _ind);
    if (val) std::copy(val, val+_nnz, _val);
  }

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>::CompressedSparseMatrix
  (const CompressedSparseMatrix<scalar_t,integer_t>& A)
    : _n(A._n), _nnz(A._nnz), _ptr(new integer_t[_n+1]),
      _ind(new integer_t[_nnz]), _val(new scalar_t[_nnz]),
      _symm_sparse(A._symm_sparse) {
    if (A._ptr) std::copy(A._ptr, A._ptr+_n+1, _ptr);
    if (A._ind) std::copy(A._ind, A._ind+_nnz, _ind);
    if (A._val) std::copy(A._val, A._val+_nnz, _val);
  }

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>::CompressedSparseMatrix
  (CompressedSparseMatrix<scalar_t,integer_t>&& A)
    : _n(A._n), _nnz(A._nnz), _symm_sparse(A._symm_sparse) {
    _ptr = A._ptr; A._ptr = nullptr;
    _ind = A._ind; A._ind = nullptr;
    _val = A._val; A._val = nullptr;
  }

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>::~CompressedSparseMatrix() {
    delete[] _ptr;
    delete[] _ind;
    delete[] _val;
  }

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>&
  CompressedSparseMatrix<scalar_t,integer_t>::operator=
  (const CompressedSparseMatrix<scalar_t,integer_t>& A) {
    _n = A._n;
    _nnz = A._nnz;
    _symm_sparse = A._symm_sparse;
    _ptr = new integer_t[_n+1];
    _ind = new integer_t[_nnz];
    _val = new scalar_t[_nnz];
    if (A._ptr) std::copy(A._ptr, A._ptr+_n+1, _ptr);
    if (A._ind) std::copy(A._ind, A._ind+_nnz, _ind);
    if (A._val) std::copy(A._val, A._val+_nnz, _val);
    return *this;
  }

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>&
  CompressedSparseMatrix<scalar_t,integer_t>::operator=
  (CompressedSparseMatrix<scalar_t,integer_t>&& A) {
    _n = A._n;
    _nnz = A._nnz;
    _symm_sparse = A._symm_sparse;
    _ptr = A._ptr; A._ptr = nullptr;
    _ind = A._ind; A._ind = nullptr;
    _val = A._val; A._val = nullptr;
    return *this;
  }

  template<typename scalar_t,typename integer_t> void
  CompressedSparseMatrix<scalar_t,integer_t>::print() const {
    if (!is_mpi_root()) return;
    std::cout << "size: " << size() << std::endl;
    std::cout << "nnz: " << nnz() << std::endl;
    std::cout << "ptr: " << std::endl << "\t";
    for (integer_t i=0; i<=size(); i++)
      std::cout << _ptr[i] << " ";
    std::cout << std::endl << "ind: ";
    for (integer_t i=0; i<nnz(); i++)
      std::cout << _ind[i] << " ";
    std::cout << std::endl << "val: ";
    for (integer_t i=0; i<nnz(); i++)
      std::cout << _val[i] << " ";
    std::cout << std::endl;
  }

  extern "C" {
    int_t strumpack_mc64id_(int_t*);
    int_t strumpack_mc64ad_
    (int_t*, int_t*, int_t*, int_t*, int_t*, double*, int_t*, int_t*, int_t*,
     int_t*, int_t*, double*, int_t*, int_t*);
  }

  /*
   *  0: do nothing
   *  1: maximum cardinality ! Doesn't work
   *  2: maximum smallest diagonal value
   *  3: 2 with different algo
   *  4: maximum sum of diagonal values
   *  5: maximum product of diagonal values + scaling
   */
  template<typename scalar_t,typename integer_t> int
  CompressedSparseMatrix<scalar_t,integer_t>::permute_and_scale
  (int job, std::vector<integer_t>& perm, std::vector<scalar_t>& Dr,
   std::vector<scalar_t>& Dc, bool apply) {
    if (job == 0) return 1;
    if (job > 5 || job < 0) {
      if (is_mpi_root())
        std::cerr
          << "# WARNING: mc64 job " << job
          << " not supported, I'm not doing any column permutation"
          << " or matrix scaling!" << std::endl;
      return 1;
    }
    perm.resize(_n);
    int_t liw = 0;
    switch (job) {
    case 2: liw = 4*_n; break;
    case 3: liw = 10*_n + _nnz; break;
    case 1: case 4: case 5: default: liw = 5*_n;
    }
    auto iw = new int_t[liw];
    int_t ldw = 0;
    switch (job) {
    case 1: ldw = 0; break;
    case 2: ldw = _n; break;
    case 3: ldw = _nnz; break;
    case 4: ldw = 2*_n + _nnz; break;
    case 5: default: ldw = 3*_n + _nnz; break;
    }
    auto dw = new double[ldw];
    int_t icntl[10], info[10];
    int_t num;
    strumpack_mc64id_(icntl);
    //icntl[2] = 6; // print diagnostics
    //icntl[3] = 1; // no checking of input should be (slightly) faster
    strumpack_mc64(job, &num, perm.data(), liw, iw, ldw, dw, icntl, info);
    switch (info[0]) {
    case  0: break;
    case  1: if (is_mpi_root())
        std::cerr << "# ERROR: matrix is structurally singular" << std::endl;
      delete[] dw;
      delete[] iw;
      return 1;
      break;
    case  2: if (is_mpi_root())
        std::cerr << "# WARNING: mc64 scaling produced"
                  << " large scaling factors which may cause overflow!"
                  << std::endl;
      break;
    default: if (is_mpi_root())
        std::cerr << "# ERROR: mc64 failed with info[0]=" << info[0]
                  << std::endl;
      delete[] dw;
      delete[] iw;
      return 1;
      break;
    }
    if (job == 5) { // scaling
      Dr.resize(_n);
      Dc.resize(_n);
#pragma omp parallel for
      for (integer_t i=0; i<_n; i++) {
        Dr[i] = scalar_t(std::exp(dw[i]));
        Dc[i] = scalar_t(std::exp(dw[_n+i]));
      }
      if (apply) apply_scaling(Dr, Dc);
    }
    delete[] iw; delete[] dw;
    if (apply) apply_column_permutation(perm);
    if (apply) _symm_sparse = false;
    return 0;
  }

  template<typename scalar_t,typename integer_t> void
  CompressedSparseMatrix<scalar_t,integer_t>::symmetrize_sparsity() {
    if (_symm_sparse) return;
    auto a2_ctr = new integer_t[_n];
#pragma omp parallel for
    for (integer_t i=0; i<_n; i++) a2_ctr[i] = _ptr[i+1]-_ptr[i];

    bool change = false;
#pragma omp parallel for
    for (integer_t i=0; i<_n; i++)
      for (integer_t jj=_ptr[i]; jj<_ptr[i+1]; jj++) {
        integer_t kb = _ptr[_ind[jj]], ke = _ptr[_ind[jj]+1];
        if (std::find(_ind+kb, _ind+ke, i) == _ind+ke) {
#pragma omp critical
          {
            a2_ctr[_ind[jj]]++;
            change = true;
          }
        }
      }
    if (change) {
      auto a2_ptr = new integer_t[_n+1];
      a2_ptr[0] = 0;
      for (integer_t i=0; i<_n; i++) a2_ptr[i+1] = a2_ptr[i] + a2_ctr[i];
      auto new_nnz = a2_ptr[_n] - a2_ptr[0];
      auto a2_ind = new integer_t[new_nnz];
      auto a2_val = new scalar_t[new_nnz];
      _nnz = new_nnz;
#pragma omp parallel for
      for (integer_t i=0; i<_n; i++) {
        a2_ctr[i] = a2_ptr[i] + _ptr[i+1] - _ptr[i];
        for (integer_t jj=_ptr[i], k=a2_ptr[i]; jj<_ptr[i+1]; jj++) {
          a2_ind[k  ] = _ind[jj];
          a2_val[k++] = _val[jj];
        }
      }
#pragma omp parallel for
      for (integer_t i=0; i<_n; i++)
        for (integer_t jj=_ptr[i]; jj<_ptr[i+1]; jj++) {
          integer_t kb = _ptr[_ind[jj]], ke = _ptr[_ind[jj]+1];
          if (std::find(_ind+kb,_ind+ke, i) == _ind+ke) {
            integer_t t = _ind[jj];
#pragma omp critical
            {
              a2_ind[a2_ctr[t]] = i;
              a2_val[a2_ctr[t]] = scalar_t(0.);
              a2_ctr[t]++;
            }
          }
        }
      set_ptr(a2_ptr);
      set_ind(a2_ind);
      set_val(a2_val);
    }
    delete[] a2_ctr;
    _symm_sparse = true;
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
    if (is_mpi_root())
      std::cout << "# opening file \'" << filename << "\'" << std::endl;
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp == NULL) {
      if (is_mpi_root()) std::cerr << "ERROR: could not read file";
      exit(1);
    }
    const int max_cline = 256;
    char cline[max_cline];
    if (fgets(cline, max_cline, fp) == NULL) {
      if (is_mpi_root())
        std::cerr << "ERROR: could not read from file" << std::endl;
      exit(1);
    }
    if (is_mpi_root()) printf("# %s", cline);
    if (strstr(cline, "pattern")) {
      if (is_mpi_root())
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
      _symm_sparse = true;
    } else if (strstr(cline, "symmetric")) {
      s = SYMMETRIC;
      _symm_sparse = true;
    } else if (strstr(cline, "hermitian")) {
      s = HERMITIAN;
      _symm_sparse = true;
    }

    while (fgets(cline, max_cline, fp)) {
      if (cline[0] != '%') { // first line should be: m n nnz
        int m, in, innz;
        sscanf(cline, "%d %d %d", &m, &in, &innz);
        _nnz = static_cast<integer_t>(innz);
        _n = static_cast<integer_t>(in);
        if (s != GENERAL) _nnz = 2 * _nnz - _n;
        if (is_mpi_root())
          std::cout << "# reading " << number_format_with_commas(m) << " by "
                    << number_format_with_commas(_n) << " matrix with "
                    << number_format_with_commas(_nnz) << " nnz's from "
                    << filename << std::endl;
        if (m != _n) {
          if (is_mpi_root())
            std::cerr << "ERROR: matrix is not square!" << std::endl;
          exit(1);
        }
        break;
      }
    }
    std::vector<std::tuple<integer_t,integer_t,scalar_t>> A;
    A.reserve(_nnz);
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
    fclose(fp);
    if (!zero_based)
      for (auto& t : A) {
        std::get<0>(t)--;
        std::get<1>(t)--;
      }
    return A;
  }

  /**
   * Obtain reordering Anew = A(iorder,iorder). In addition, entries
   * of IND, VAL are sorted in increasing order
   */
  template<typename scalar_t,typename integer_t> void
  CompressedSparseMatrix<scalar_t,integer_t>::permute
  (const integer_t* iorder, const integer_t* order) {
    auto new_ptr = new integer_t[_n+1];
    auto new_ind = new integer_t[_nnz];
    auto new_val = new scalar_t[_nnz];
    integer_t nnz = 0;
    new_ptr[0] = 0;
    for (integer_t i=0; i<_n; i++) {
      auto lb = _ptr[iorder[i]];
      auto ub = _ptr[iorder[i]+1];
      for (integer_t j=lb; j<ub; j++) {
        new_ind[nnz] = order[_ind[j]];
        new_val[nnz++] = _val[j];
      }
      new_ptr[i+1] = nnz;
    }
#pragma omp parallel for
    for (integer_t i=0; i<_n; i++) {
      auto lb = new_ptr[i];
      auto ub = new_ptr[i+1];
      sort_indices_values
        (new_ind+lb, new_val+lb, integer_t(0), ub-lb);
    }
    set_ptr(new_ptr);
    set_ind(new_ind);
    set_val(new_val);
  }

} //end namespace strumpack

#endif
