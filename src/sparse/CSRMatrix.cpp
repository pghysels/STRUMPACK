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
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <string>

#include "CSRMatrix.hpp"
#include "MC64ad.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "dense/DistributedMatrix.hpp"
#endif

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  CSRMatrix<scalar_t,integer_t>::CSRMatrix() : CSM_t() {}

  template<typename scalar_t,typename integer_t>
  CSRMatrix<scalar_t,integer_t>::CSRMatrix
  (integer_t n, const integer_t* ptr, const integer_t* ind,
   const scalar_t* values, bool symm_sparsity)
    : CSM_t(n, ptr, ind, values, symm_sparsity) { }

  template<typename scalar_t,typename integer_t>
  CSRMatrix<scalar_t,integer_t>::CSRMatrix
  (integer_t n, integer_t nnz) : CSM_t(n, nnz) {}

  template<typename scalar_t,typename integer_t>
  typename RealType<scalar_t>::value_type
  CSRMatrix<scalar_t,integer_t>::norm1() const {
    std::vector<real_t> n1(n_);
    for (integer_t i=0; i<n_; i++)
      for (integer_t j=ptr_[i]; j<ptr_[i+1]; j++)
        n1[ind_[j]] += std::abs(val_[j]);
    return *std::max_element(n1.begin(), n1.end());
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::print_dense(const std::string& name) const {
    DenseM_t M(n_, n_);
    M.fill(scalar_t(0.));
    for (integer_t i=0; i<n_; i++)
      for (integer_t j=ptr_[i]; j<ptr_[i+1]; j++)
        M(i, ind_[j]) = val_[j];
    M.print(name);
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::print_matrix_market
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

// #if defined(__INTEL_MKL__)
//   // TODO test this, enable from CMake
//   template<> void CSRMatrix<float>::spmv(const float* x, float* y) const {
//     char trans = 'N';
//     mkl_cspblas_scsrgemv(&no, &n, val_, ptr_, ind_, x, y);
//     STRUMPACK_FLOPS(this->spmv_flops());
//     STRUMPACK_BYTES(this->spmv_bytes());
//   }
//   template<> void CSRMatrix<double>::spmv(const double* x, double* y) const {
//     char trans = 'N';
//     mkl_cspblas_dcsrgemv(&no, &n, val_, ptr_, ind_, x, y);
//     STRUMPACK_FLOPS(this->spmv_flops());
//     STRUMPACK_BYTES(this->spmv_bytes());
//   }
//   template<> void CSRMatrix<std::complex<float>>::spmv
//   (const std::complex<float>* x, std::complex<float>* y) const {
//     char trans = 'N';
//     mkl_cspblas_ccsrgemv(&no, &n, val_, ptr_, ind_, x, y);
//     STRUMPACK_FLOPS(this->spmv_flops());
//     STRUMPACK_BYTES(this->spmv_bytes());
//   }
//   template<> void CSRMatrix<std::complex<double>>::spmv
//   (const std::complex<double>* x, std::complex<double>* y) const {
//     char trans = 'N';
//     mkl_cspblas_zcsrgemv(&no, &n, val_, ptr_, ind_, x, y);
//     STRUMPACK_FLOPS(this->spmv_flops());
//     STRUMPACK_BYTES(this->spmv_bytes());
//   }
// #endif

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
    // assert(x.cols() == y.cols());
    // assert(x.rows() == std::size_t(n_));
    // assert(y.rows() == std::size_t(n_));
    for (std::size_t c=0; c<x.cols(); c++)
      spmv(x.ptr(0,c), y.ptr(0,c));
  }

  // TODO use MKL routines for better performance
  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::spmv
  (Trans op, const DenseM_t& x, DenseM_t& y) const {
    if (op != Trans::N) y.zero();
    for (std::size_t c=0; c<x.cols(); c++) {
      auto px = x.ptr(0, c);
      auto py = y.ptr(0, c);
      if (op == Trans::N) {
        for (integer_t r=0; r<n_; r++) {
          const auto hij = ptr_[r+1];
          scalar_t yr(0);
          for (integer_t j=ptr_[r]; j<hij; j++)
            yr += val_[j] * px[ind_[j]];
          py[r] = yr;
        }
      } else if (op == Trans::T) {
        for (integer_t r=0; r<n_; r++) {
          const auto hij = ptr_[r+1];
          for (integer_t j=ptr_[r]; j<hij; j++)
            py[ind_[j]] += val_[j] * px[r];
        }
      } else if (op == Trans::C) {
        for (integer_t r=0; r<n_; r++) {
          const auto hij = ptr_[r+1];
          for (integer_t j=ptr_[r]; j<hij; j++)
            py[ind_[j]] += blas::my_conj(val_[j]) * px[r];
        }
      }
    }
    STRUMPACK_FLOPS(x.cols()*this->spmv_flops());
    STRUMPACK_BYTES(x.cols()*this->spmv_bytes());
  }


  template<typename scalar_t,typename integer_t> Equilibration<scalar_t>
  CSRMatrix<scalar_t,integer_t>::equilibration() const {
    Equil_t eq(n_);
    if (!n_) return eq;
    real_t small = blas::lamch<real_t>('S');
    real_t big = 1. / small;
#pragma omp parallel for
    for (integer_t i=0; i<n_; i++)
      for (integer_t j=ptr_[i]; j<ptr_[i+1]; j++)
        eq.R[i] = std::max(eq.R[i], std::abs(val_[j]));
    auto mM = std::minmax_element(eq.R.begin(), eq.R.end());
    real_t rmin = *(mM.first), rmax = *(mM.second);
    eq.Amax = rmax;
    if (rmin == 0.) {
      for (integer_t i=0; i<n_; i++)
        if (eq.R[i] == 0.) {
          eq.info = i+1;
          return eq;
        }
    }
#pragma omp parallel for
    for (integer_t i=0; i<n_; i++)
      eq.R[i] = 1. / std::min(std::max(eq.R[i], small), big);
    eq.rcond = std::max(rmin, small) / std::min(rmax, big);
    // cannot use openmp here
    for (integer_t i=0; i<n_; i++) {
      for (integer_t j=ptr_[i]; j<ptr_[i+1]; j++) {
        auto indj = ind_[j];
        eq.C[indj] = std::max(eq.C[indj], std::abs(val_[j]) * eq.R[i]);
      }
    }
    mM = std::minmax_element(eq.C.begin(), eq.C.end());
    real_t cmin = *(mM.first), cmax = *(mM.second);
    if (cmin == 0.) {
      for (integer_t i=0; i<n_; i++)
        if (eq.C[i] == 0.) {
          eq.info = n_+i+1;
          return eq;
        }
    }
#pragma omp parallel for
    for (integer_t i=0; i<n_; i++)
      eq.C[i] = 1. / std::min(std::max(eq.C[i], small), big);
    eq.ccond = std::max(cmin, small) / std::min(cmax, big);
    eq.set_type();
    return eq;
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::equilibrate(const Equil_t& eq) {
    if (!n_) return;
    switch (eq.type) {
    case EquilibrationType::COLUMN: {
#pragma omp parallel for
      for (integer_t i=0; i<n_; i++)
        for (integer_t j=ptr_[i]; j<ptr_[i+1]; j++)
          val_[j] *= eq.C[ind_[j]];
      STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*
                      static_cast<long long int>(double(nnz_)));
    } break;
    case EquilibrationType::ROW: {
#pragma omp parallel for
      for (integer_t i=0; i<n_; i++)
        for (integer_t j=ptr_[i]; j<ptr_[i+1]; j++)
          val_[j] *= eq.R[i];
      STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*
                      static_cast<long long int>(double(nnz_)));
    } break;
    case EquilibrationType::BOTH: {
#pragma omp parallel for
      for (integer_t i=0; i<n_; i++)
        for (integer_t j=ptr_[i]; j<ptr_[i+1]; j++)
          val_[j] *= eq.R[i] * eq.C[ind_[j]];
      STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*
                      static_cast<long long int>(2.*double(nnz_)));
    } break;
    case EquilibrationType::NONE: {}
    }
  }


  template<typename scalar_t,typename integer_t> int
  CSRMatrix<scalar_t,integer_t>::strumpack_mc64
  (MatchingJob job, Match_t& M) {
    integer_t n = n_, nnz = nnz_, icntl[10], info[10], num,
      ijob = static_cast<int>(job), liw = M.mc64_work_int(n_, nnz_),
      ldw = M.mc64_work_double(n_, nnz_);
    std::unique_ptr<integer_t[]> iw(new integer_t[liw + n+1+nnz+n+n]);
    std::unique_ptr<double[]> dw(new double[ldw + nnz]);
    mc64id(icntl);
    auto dval = dw.get() + ldw;
    auto cptr = iw.get() + liw;
    auto rind = cptr + n + 1;
    auto permutation = rind + nnz;
    auto rsums = permutation + n;
    for (integer_t i=0; i<n; i++) rsums[i] = 0;
    for (integer_t col=0; col<n; col++)
      for (integer_t i=ptr_[col]; i<ptr_[col+1]; i++)
        rsums[ind_[i]]++;
    cptr[0] = 1;  // start from 1, because mc64 is fortran!
    for (integer_t r=0; r<n; r++) {
      cptr[r+1] = cptr[r] + rsums[r];
      rsums[r] = 0;
    }
    for (integer_t col=0; col<n; col++) {
      for (integer_t i=ptr_[col]; i<ptr_[col+1]; i++) {
        integer_t row = ind_[i], j = cptr[row] - 1 + rsums[row]++;
        if (is_complex<scalar_t>())
          dval[j] = static_cast<double>(std::abs(val_[i]));
        else dval[j] = static_cast<double>(std::real(val_[i]));
        rind[j] = col + 1;
      }
    }
    mc64ad(&ijob, &n, &nnz, cptr, rind, dval, &num,
           permutation, &liw, iw.get(), &ldw, dw.get(),
           icntl, info);
    for (integer_t i=0; i<n; i++)
      M.Q[i] = permutation[i] - 1;
    if (job == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING) {
#pragma omp parallel for
      for (integer_t i=0; i<n_; i++) {
        M.R[i] = real_t(std::exp(dw[i]));
        M.C[i] = real_t(std::exp(dw[n_+i]));
      }
    }
    return info[0];
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

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::push_front_elements
  (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   std::vector<Triplet<scalar_t>>& e11, std::vector<Triplet<scalar_t>>& e12,
   std::vector<Triplet<scalar_t>>& e21) const {
    integer_t ds = shi - slo, du = upd.size();
    for (integer_t row=0; row<ds; row++) { // separator rows
      integer_t upd_ptr = 0;
      const auto hij = ptr_[row+slo+1];
      for (integer_t j=ptr_[row+slo]; j<hij; j++) {
        integer_t col = ind_[j];
        if (col >= slo) {
          if (col < shi)
            e11.emplace_back(row, col-slo, val_[j]);
          else {
            while (upd_ptr<du && upd[upd_ptr]<col) upd_ptr++;
            if (upd_ptr == du) break;
            if (upd[upd_ptr] == col)
              e12.emplace_back(row, upd_ptr, val_[j]);
          }
        }
      }
    }
    for (integer_t i=0; i<du; i++) { // update rows
      auto row = upd[i];
      const auto hij = ptr_[row+1];
      for (integer_t j=ptr_[row]; j<hij; j++) {
        integer_t col = ind_[j];
        if (col >= slo) {
          if (col < shi)
            e21.emplace_back(i, col-slo, val_[j]);
          else break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::set_front_elements
  (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   Triplet<scalar_t>* e11, Triplet<scalar_t>* e12,
   Triplet<scalar_t>* e21) const {
    integer_t ds = shi - slo, du = upd.size();
    for (integer_t row=0; row<ds; row++) { // separator rows
      integer_t upd_ptr = 0;
      const auto hij = ptr_[row+slo+1];
      for (integer_t j=ptr_[row+slo]; j<hij; j++) {
        integer_t col = ind_[j];
        if (col >= slo) {
          if (col < shi)
            *e11++ = Triplet<scalar_t>(row, col-slo, val_[j]);
          else {
            while (upd_ptr<du && upd[upd_ptr]<col) upd_ptr++;
            if (upd_ptr == du) break;
            if (upd[upd_ptr] == col)
              *e12++ = Triplet<scalar_t>(row, upd_ptr, val_[j]);
          }
        }
      }
    }
    for (integer_t i=0; i<du; i++) { // update rows
      auto row = upd[i];
      const auto hij = ptr_[row+1];
      for (integer_t j=ptr_[row]; j<hij; j++) {
        integer_t col = ind_[j];
        if (col >= slo) {
          if (col < shi)
            *e21++ = Triplet<scalar_t>(i, col-slo, val_[j]);
          else break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::count_front_elements
  (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   std::size_t& e11, std::size_t& e12, std::size_t& e21) const {
    integer_t ds = shi - slo, du = upd.size();
    for (integer_t row=0; row<ds; row++) { // separator rows
      integer_t upd_ptr = 0;
      const auto hij = ptr_[row+slo+1];
      for (integer_t j=ptr_[row+slo]; j<hij; j++) {
        integer_t col = ind_[j];
        if (col >= slo) {
          if (col < shi) e11++;
          else {
            while (upd_ptr<du && upd[upd_ptr]<col) upd_ptr++;
            if (upd_ptr == du) break;
            if (upd[upd_ptr] == col) e12++;
          }
        }
      }
    }
    for (integer_t i=0; i<du; i++) { // update rows
      auto row = upd[i];
      const auto hij = ptr_[row+1];
      for (integer_t j=ptr_[row]; j<hij; j++) {
        integer_t col = ind_[j];
        if (col >= slo) {
          if (col < shi) e21++;
          else break;
        }
      }
    }
  }

  // TODO parallel -> will be hard to do efficiently
  // assume F11, F12 and F21 are set to zero
  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::extract_front
  (DenseM_t& F11, DenseM_t& F12, DenseM_t& F21, integer_t slo,
   integer_t shi, const std::vector<integer_t>& upd, int depth) const {
    integer_t ds = shi - slo, du = upd.size();
    for (integer_t row=0; row<ds; row++) { // separator rows
      integer_t upd_ptr = 0;
      const auto hij = ptr_[row+slo+1];
      for (integer_t j=ptr_[row+slo]; j<hij; j++) {
        integer_t col = ind_[j];
        if (col >= slo) {
          if (col < shi)
            F11(row, col-slo) = val_[j];
          else {
            while (upd_ptr<du && upd[upd_ptr]<col)
              upd_ptr++;
            if (upd_ptr == du) break;
            if (upd[upd_ptr] == col)
              F12(row, upd_ptr) = val_[j];
          }
        }
      }
    }
    for (integer_t i=0; i<du; i++) { // update rows
      auto row = upd[i];
      const auto hij = ptr_[row+1];
      for (integer_t j=ptr_[row]; j<hij; j++) {
        integer_t col = ind_[j];
        if (col >= slo) {
          if (col < shi)
            F21(i, col-slo) = val_[j];
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
    const integer_t nbvec = R.cols();
    const integer_t ds = shi - slo;
    const auto B = 4; // blocking parameter
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)                    \
  if(depth < params::task_recursion_cutoff_level)
#endif
    for (integer_t c=0; c<nbvec; c+=B) {
      long long int local_flops = 0;
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
                Sc(col-slo, cc) += blas::my_conj(vj) * R(row-slo, cc);
              }
              local_flops += 4*(hicc-c);
            } else {
              while (upd_ptr<dupd && upd[upd_ptr]<col) upd_ptr++;
              if (upd_ptr == dupd) break;
              if (upd[upd_ptr] == col) {
                for (integer_t cc=c; cc<hicc; cc++) {
                  Sr(row-slo, cc) += vj * R(ds+upd_ptr, cc);
                  Sc(ds+upd_ptr, cc) += blas::my_conj(vj) * R(row-slo, cc);
                }
                local_flops += 4*(hicc-c);
              }
            }
          }
        }
      }
      STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
      STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    }

#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)                    \
  if(depth < params::task_recursion_cutoff_level)
#endif
    for (integer_t c=0; c<nbvec; c+=B) {
      long long int local_flops = 0;
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
              local_flops += 4*(hicc-c);
            } else break;
          }
        }
      }
      STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
      STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    }
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::front_multiply_F11
  (Trans op, integer_t slo, integer_t shi,
   const DenseM_t& R, DenseM_t& S, int depth) const {
    const integer_t nbvec = R.cols();
    const auto B = 4; // blocking parameter
    long long int local_flops = 0;
    if (op == Trans::N) {
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)                    \
  if(depth < params::task_recursion_cutoff_level)
#endif
      for (integer_t c=0; c<nbvec; c+=B)
        for (auto row=slo; row<shi; row++) { // separator rows
          const auto hij = ptr_[row+1];
          for (auto j=ptr_[row]; j<hij; j++) {
            const auto col = ind_[j];
            if (col >= slo) {
              const auto hicc = std::min(c+B, nbvec);
              const auto vj = val_[j];
              if (col < shi) {
                for (integer_t cc=c; cc<hicc; cc++)
                  S(row-slo, cc) += vj * R(col-slo, cc);
                local_flops += 4*(hicc-c);
              }
            }
          }
        }
    } else {
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)                    \
  if(depth < params::task_recursion_cutoff_level)
#endif
      for (integer_t c=0; c<nbvec; c+=B) {
        long long int local_flops = 0;
        for (auto row=slo; row<shi; row++) { // separator rows
          const auto hij = ptr_[row+1];
          for (auto j=ptr_[row]; j<hij; j++) {
            const auto col = ind_[j];
            if (col >= slo) {
              const auto hicc = std::min(c+B, nbvec);
              const auto vj = val_[j];
              if (col < shi) {
                for (integer_t cc=c; cc<hicc; cc++)
                  S(col-slo, cc) += blas::my_conj(vj) * R(row-slo, cc);
                local_flops += 4*(hicc-c);
              }
            }
          }
        }
      }
    }
    STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::front_multiply_F12
  (Trans op, integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   const DenseM_t& R, DenseM_t& S, int depth) const {
    integer_t dupd = upd.size();
    const integer_t nbvec = R.cols();
    long long int local_flops = 0;
    const auto B = 4; // blocking parameter
    if (op == Trans::N) {
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
              if (col >= shi) {
                while (upd_ptr<dupd && upd[upd_ptr]<col) upd_ptr++;
                if (upd_ptr == dupd) break;
                if (upd[upd_ptr] == col) {
                  for (integer_t cc=c; cc<hicc; cc++)
                    S(row-slo, cc) += vj * R(upd_ptr, cc);
                  local_flops += 4*(hicc-c);
                }
              }
            }
          }
        }
      }
    } else {
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
              if (col >= shi) {
                while (upd_ptr<dupd && upd[upd_ptr]<col) upd_ptr++;
                if (upd_ptr == dupd) break;
                if (upd[upd_ptr] == col) {
                  for (integer_t cc=c; cc<hicc; cc++)
                    S(upd_ptr, cc) += blas::my_conj(vj) * R(row-slo, cc);
                  local_flops += 4*(hicc-c);
                }
              }
            }
          }
        }
      }
    }
    STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::front_multiply_F21
  (Trans op, integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   const DenseM_t& R, DenseM_t& S, int depth) const {
    integer_t dupd = upd.size();
    const integer_t nbvec = R.cols();
    long long int local_flops = 0;
    const auto B = 4; // blocking parameter
    if (op == Trans::N) {
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
                for (integer_t cc=c; cc<hicc; cc++)
                  S(i, cc) += vj * R(col-slo, cc);
                local_flops += 4*(hicc-c);
              } else break;
            }
          }
        }
      }
    } else {
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
                for (integer_t cc=c; cc<hicc; cc++)
                  S(col-slo, cc) += vj * R(i, cc);
                local_flops += 4*(hicc-c);
              } else break;
            }
          }
        }
      }
    }
    STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::scale
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
  CSRMatrix<scalar_t,integer_t>::scale_real
  (const std::vector<real_t>& Dr, const std::vector<real_t>& Dc) {
#pragma omp parallel for
    for (integer_t j=0; j<n_; j++)
      for (integer_t i=ptr_[j]; i<ptr_[j+1]; i++)
        val_[i] = val_[i] * Dr[j] * Dc[ind_[i]];
    STRUMPACK_FLOPS
      ((is_complex<scalar_t>()?2:1)*
       static_cast<long long int>(2.*double(this->nnz())));
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrix<scalar_t,integer_t>::permute_columns
  (const std::vector<integer_t>& perm) {
    std::unique_ptr<integer_t[]> iperm(new integer_t[n_]);
    for (integer_t i=0; i<n_; i++) iperm[perm[i]] = i;
#pragma omp parallel for
    for (integer_t row=0; row<n_; row++) {
      for (integer_t i=ptr_[row]; i<ptr_[row+1]; i++)
        ind_[i] = iperm[ind_[i]];
      sort_indices_values<scalar_t>
        (ind_.data(), val_.data(), ptr_[row], ptr_[row+1]);
    }
  }

  template<typename scalar_t,typename integer_t> int
  CSRMatrix<scalar_t,integer_t>::read_matrix_market
  (const std::string& filename) {
    using Triplet = std::tuple<integer_t,integer_t,scalar_t>;
    std::vector<Triplet> A;
    try {
      A = this->read_matrix_market_entries(filename);
    } catch (...) { return 1; }
    std::sort
      (A.begin(), A.end(),
       [](const Triplet& a, const Triplet& b) -> bool {
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
  (integer_t shi, const std::vector<std::size_t>& I,
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
        if (c >= cmin && c <= cmax && (r < shi || c < shi)) {
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
  (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   const DistM_t& R, DistM_t& Srow, DistM_t& Scol, int depth) const {
    integer_t du = upd.size();
    // redistribute R and Srow and Scol to 1d block cyclic column
    // distribution to avoid communication below
    DistM_t R_bc(R.grid(), R.rows(), R.cols(), R.rows(), R.NB()),
      Srow_bc(R.grid(), Srow.rows(), Srow.cols(), Srow.rows(), Srow.NB()),
      Scol_bc(R.grid(), Scol.rows(), Scol.cols(), Scol.rows(), Scol.NB());
    copy(R.rows(), R.cols(), R, 0, 0, R_bc, 0, 0, R.ctxt_all());
    copy(Srow.rows(), Srow.cols(), Srow, 0, 0, Srow_bc, 0, 0, R.ctxt_all());
    copy(Scol.rows(), Scol.cols(), Scol, 0, 0, Scol_bc, 0, 0, R.ctxt_all());

    if (R.active()) {
      long long int local_flops = 0;
      auto ds = shi - slo;
      auto n = R_bc.cols();
      for (integer_t row=slo; row<shi; row++) { // separator rows
        auto upd_ptr = 0;
        auto hij = ptr_[row+1];
        for (integer_t j=ptr_[row]; j<hij; j++) {
          auto col = ind_[j];
          if (col >= slo) {
            if (col < shi) {
              scalapack::pgeadd
                ('N', 1, n, val_[j], R_bc.data(), col-slo+1, 1,
                 R_bc.desc(), scalar_t(1.), Srow_bc.data(), row-slo+1,
                 1, Srow_bc.desc());
              scalapack::pgeadd
                ('N', 1, n, val_[j], R_bc.data(), row-slo+1, 1,
                 R_bc.desc(), scalar_t(1.), Scol_bc.data(), col-slo+1,
                 1, Scol_bc.desc());
              local_flops += 4 * n;
            } else {
              while (upd_ptr<du && upd[upd_ptr]<col) upd_ptr++;
              if (upd_ptr == du) break;
              if (upd[upd_ptr] == col) {
                scalapack::pgeadd
                  ('N', 1, n, val_[j], R_bc.data(), ds+upd_ptr+1,
                   1, R_bc.desc(), scalar_t(1.), Srow_bc.data(),
                   row-slo+1, 1, Srow_bc.desc());
                scalapack::pgeadd
                             ('N', 1, n, val_[j], R_bc.data(),
                              row-slo+1, 1, R_bc.desc(), scalar_t(1.),
                              Scol_bc.data(), ds+upd_ptr+1, 1,
                              Scol_bc.desc());
                local_flops += 4 * n;
              }
            }
          }
        }
      }
      for (integer_t i=0; i<du; i++) { // remaining rows
        integer_t row = upd[i];
        auto hij = ptr_[row+1];
        for (integer_t j=ptr_[row]; j<hij; j++) {
          integer_t col = ind_[j];
          if (col >= slo) {
            if (col < shi) {
              scalapack::pgeadd
                ('N', 1, n, val_[j], R_bc.data(), col-slo+1, 1,
                 R_bc.desc(), scalar_t(1.), Srow_bc.data(), ds+i+1,
                 1, Srow_bc.desc());
              scalapack::pgeadd
                ('N', 1, n, val_[j], R_bc.data(), ds+i+1, 1,
                 R_bc.desc(), scalar_t(1.), Scol_bc.data(), col-slo+1,
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
  CSRMatrix<scalar_t,integer_t>::front_multiply_2d
  (Trans op, integer_t slo, integer_t shi,
   const std::vector<integer_t>& upd, const DistM_t& R,
   DistM_t& S, int depth) const {
    // TODO optimize this!!
    DistM_t Sdummy(S.grid(), S.rows(), S.cols());
    if (op == Trans::N)
      front_multiply_2d(slo, shi, upd, R, S, Sdummy, 0);
    else front_multiply_2d(slo, shi, upd, R, Sdummy, S, 0);
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

  template<typename scalar_t,typename integer_t> CSRGraph<integer_t>
  CSRMatrix<scalar_t,integer_t>::extract_graph
  (int ordering_level, integer_t lo, integer_t hi) const {
    assert(ordering_level == 0 || ordering_level == 1);
    auto dim = hi - lo;
    std::vector<bool> mark(dim);
    std::vector<integer_t> xadj, adjncy;
    xadj.reserve(dim+1);
    adjncy.reserve(5*dim);
    for (integer_t i=lo, e=0; i<hi; i++) {
      xadj.push_back(e);
      std::fill(mark.begin(), mark.end(), false);
      for (integer_t j=ptr_[i]; j<ptr_[i+1]; j++) {
        auto c = ind_[j];
        if (c == i) continue;
        auto lc = c - lo;
        if (lc >= 0 && lc < dim) {
          if (!mark[lc]) {
            mark[lc] = true;
            adjncy.push_back(lc);
            e++;
          }
        } else {
          if (ordering_level > 0) {
            for (integer_t k=ptr_[c]; k<ptr_[c+1]; k++) {
              auto cc = ind_[k];
              auto lcc = cc - lo;
              if (cc != i && lcc >= 0 && lcc < dim && !mark[lcc]) {
                mark[lcc] = true;
                adjncy.push_back(lcc);
                e++;
              }
            }
          }
        }
      }
    }
    xadj.push_back(adjncy.size());
    return CSRGraph<integer_t>(std::move(xadj), std::move(adjncy));
  }

  template<typename scalar_t,typename integer_t> CSRGraph<integer_t>
  CSRMatrix<scalar_t,integer_t>::extract_graph_sep_CB
  (int ordering_level, integer_t lo, integer_t hi,
   const std::vector<integer_t>& upd) const {
    integer_t dsep = hi - lo;
    std::vector<integer_t> gptr, gind;
    gptr.reserve(dsep+1);
    gptr.push_back(0);
    for (integer_t i=lo; i<hi; i++) {
      gptr.push_back(gptr.back());
      for (integer_t j=ptr_[i]; j<ptr_[i+1]; j++) {
        integer_t uj = ind_[j];
        // TODO this search is not necessary? just check range?
        auto lb = std::lower_bound(upd.begin(), upd.end(), uj);
        if (lb != upd.end() && *lb == uj) {
          auto ij = std::distance(upd.begin(), lb);
          assert(ij < static_cast<integer_t>(upd.size()) && ij >= 0);
          gind.push_back(ij);
          gptr.back()++;
        }
      }
    }
    return CSRGraph<integer_t>(std::move(gptr), std::move(gind));
  }

  template<typename scalar_t,typename integer_t> CSRGraph<integer_t>
  CSRMatrix<scalar_t,integer_t>::extract_graph_CB_sep
  (int ordering_level, integer_t lo, integer_t hi,
   const std::vector<integer_t>& upd) const {
    integer_t dupd = upd.size(), dsep = hi - lo;
    std::vector<integer_t> gptr, gind;
    gptr.reserve(dsep+1);
    gptr.push_back(0);
    for (integer_t ii=0; ii<dupd; ii++) {
      gptr.push_back(gptr.back());
      integer_t i = upd[ii];
      for (integer_t j=ptr_[i]; j<ptr_[i+1]; j++) {
        integer_t uj = ind_[j];
        if (uj >= lo && uj < hi) {
          gind.push_back(uj-lo);
          gptr.back()++;
        }
      }
    }
    return CSRGraph<integer_t>(std::move(gptr), std::move(gind));
  }

  template<typename scalar_t,typename integer_t> CSRGraph<integer_t>
  CSRMatrix<scalar_t,integer_t>::extract_graph_CB
  (int ordering_level, const std::vector<integer_t>& upd) const {
    integer_t dupd = upd.size();
    std::vector<integer_t> gptr, gind;
    gptr.reserve(dupd+1);
    gptr.push_back(0);
    for (integer_t ii=0; ii<dupd; ii++) {
      gptr.push_back(gptr.back());
      integer_t i = upd[ii];
      for (integer_t j=ptr_[i]; j<ptr_[i+1]; j++) {
        integer_t uj = ind_[j];
        // TODO this search is not necessary? just check range?
        auto lb = std::lower_bound(upd.begin(), upd.end(), uj);
        if (lb != upd.end() && *lb == uj) {
          auto ij = std::distance(upd.begin(), lb);
          assert(ij < dupd && ij >= 0);
          gind.push_back(ij);
          gptr.back()++;
        }
      }
    }
    return CSRGraph<integer_t>(std::move(gptr), std::move(gind));
  }


  // explicit template instantiations
  template class CSRMatrix<float,int>;
  template class CSRMatrix<double,int>;
  template class CSRMatrix<std::complex<float>,int>;
  template class CSRMatrix<std::complex<double>,int>;

  template class CSRMatrix<float,long int>;
  template class CSRMatrix<double,long int>;
  template class CSRMatrix<std::complex<float>,long int>;
  template class CSRMatrix<std::complex<double>,long int>;

  template class CSRMatrix<float,long long int>;
  template class CSRMatrix<double,long long int>;
  template class CSRMatrix<std::complex<float>,long long int>;
  template class CSRMatrix<std::complex<double>,long long int>;



  template<typename scalar_t, typename integer_t, typename cast_t>
  CSRMatrix<cast_t,integer_t>
  cast_matrix(const CSRMatrix<scalar_t,integer_t>& mat) {
    std::vector<cast_t> new_val(mat.val(), mat.val()+mat.nnz());
    return CSRMatrix<cast_t,integer_t>
      (mat.size(), mat.ptr(), mat.ind(), new_val.data(), mat.symm_sparse());
  }

  // explicit template instantiations
  template CSRMatrix<float,int>
  cast_matrix<double,int,float>(const CSRMatrix<double,int>& mat);
  template CSRMatrix<double,int>
  cast_matrix<float,int,double>(const CSRMatrix<float,int>& mat);
  template CSRMatrix<std::complex<float>,int>
  cast_matrix<std::complex<double>,int,std::complex<float>>
  (const CSRMatrix<std::complex<double>,int>& mat);
  template CSRMatrix<std::complex<double>,int>
  cast_matrix<std::complex<float>,int,std::complex<double>>
  (const CSRMatrix<std::complex<float>,int>& mat);

  template CSRMatrix<float,long int>
  cast_matrix<double,long int,float>(const CSRMatrix<double,long int>& mat);
  template CSRMatrix<double,long int>
  cast_matrix<float,long int,double>(const CSRMatrix<float,long int>& mat);
  template CSRMatrix<std::complex<float>,long int>
  cast_matrix<std::complex<double>,long int,std::complex<float>>
  (const CSRMatrix<std::complex<double>,long int>& mat);
  template CSRMatrix<std::complex<double>,long int>
  cast_matrix<std::complex<float>,long int,std::complex<double>>
  (const CSRMatrix<std::complex<float>,long int>& mat);

  template CSRMatrix<float,long long int>
  cast_matrix<double,long long int,float>
  (const CSRMatrix<double,long long int>& mat);
  template CSRMatrix<double,long long int>
  cast_matrix<float,long long int,double>
  (const CSRMatrix<float,long long int>& mat);
  template CSRMatrix<std::complex<float>,long long int>
  cast_matrix<std::complex<double>,long long int,std::complex<float>>
  (const CSRMatrix<std::complex<double>,long long int>& mat);
  template CSRMatrix<std::complex<double>,long long int>
  cast_matrix<std::complex<float>,long long int,std::complex<double>>
  (const CSRMatrix<std::complex<float>,long long int>& mat);

} // end namespace strumpack
