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

#include <string>
#include <iomanip>
#include <cassert>
#include <algorithm>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "DenseMatrix.hpp"
#include "misc/Tools.hpp"
#include "misc/TaskTimer.hpp"
#include "BLASLAPACKOpenMPTask.hpp"

namespace strumpack {

  template<typename scalar_t> DenseMatrix<scalar_t>::DenseMatrix()
    : data_(nullptr), rows_(0), cols_(0), ld_(1) { }

  template<typename scalar_t> DenseMatrix<scalar_t>::DenseMatrix
  (std::size_t m, std::size_t n)
    : data_(new scalar_t[m*n]), rows_(m),
      cols_(n), ld_(std::max(std::size_t(1), m)) {
    STRUMPACK_ADD_MEMORY(rows_*cols_*sizeof(scalar_t));
  }

  template<typename scalar_t> DenseMatrix<scalar_t>::DenseMatrix
  (std::size_t m, std::size_t n,
   const std::function<scalar_t(std::size_t,std::size_t)>& A)
    : DenseMatrix<scalar_t>(m, n) {
    fill(A);
  }

  template<typename scalar_t> DenseMatrix<scalar_t>::DenseMatrix
  (std::size_t m, std::size_t n, const scalar_t* D, std::size_t ld)
    : data_(new scalar_t[m*n]), rows_(m), cols_(n),
      ld_(std::max(std::size_t(1), m)) {
    STRUMPACK_ADD_MEMORY(rows_*cols_*sizeof(scalar_t));
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
    STRUMPACK_ADD_MEMORY(rows_*cols_*sizeof(scalar_t));
    for (std::size_t _j=0; _j<std::min(cols_, D.cols()-j); _j++)
      for (std::size_t _i=0; _i<std::min(rows_, D.rows()-i); _i++)
        operator()(_i, _j) = D(_i+i, _j+j);
  }

  template<typename scalar_t>
  DenseMatrix<scalar_t>::DenseMatrix(const DenseMatrix<scalar_t>& D)
    : data_(new scalar_t[D.rows()*D.cols()]), rows_(D.rows()),
      cols_(D.cols()), ld_(std::max(std::size_t(1), D.rows())) {
    STRUMPACK_ADD_MEMORY(rows_*cols_*sizeof(scalar_t));
    for (std::size_t j=0; j<cols_; j++)
      for (std::size_t i=0; i<rows_; i++)
        operator()(i, j) = static_cast<scalar_t>(D(i, j));
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
    if (data_) {
      STRUMPACK_SUB_MEMORY(rows_*cols_*sizeof(scalar_t));
      delete[] data_;
    }
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::operator=(const DenseMatrix<scalar_t>& D) {
    if (this == &D) return *this;
    if (rows_ != D.rows() || cols_ != D.cols()) {
      if (data_) {
        STRUMPACK_SUB_MEMORY(rows_*cols_*sizeof(scalar_t));
        delete[] data_;
      }
      rows_ = D.rows();
      cols_ = D.cols();
      STRUMPACK_ADD_MEMORY(rows_*cols_*sizeof(scalar_t));
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
    if (data_) {
      STRUMPACK_SUB_MEMORY(rows_*cols_*sizeof(scalar_t));
      delete[] data_;
    }
    rows_ = D.rows();
    cols_ = D.cols();
    ld_ = D.ld();
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
    for (std::size_t i=0; i<rows(); i++) {
      for (std::size_t j=0; j<cols(); j++)
        fs << std::setprecision(16) << std::setw(width) << operator()(i,j) << "  ";
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

  template<typename scalar_t> void DenseMatrix<scalar_t>::fill
  (const std::function<scalar_t(std::size_t,std::size_t)>& A) {
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<rows(); i++)
        operator()(i,j) = A(i, j);
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::eye() {
    auto minmn = std::min(cols(), rows());
    for (std::size_t j=0; j<minmn; j++)
      for (std::size_t i=0; i<minmn; i++)
        operator()(i,j) = (i == j) ? scalar_t(1.) : scalar_t(0.);
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::clear() {
    if (data_) {
      STRUMPACK_SUB_MEMORY(rows_*cols_*sizeof(scalar_t));
      delete[] data_;
    }
    rows_ = 0;
    cols_ = 0;
    ld_ = 1;
    data_ = nullptr;
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::resize(std::size_t m, std::size_t n) {
    STRUMPACK_ADD_MEMORY(m*n*sizeof(scalar_t));
    auto tmp = new scalar_t[m*n];
    for (std::size_t j=0; j<std::min(cols(),n); j++)
      for (std::size_t i=0; i<std::min(rows(),m); i++)
        tmp[i+j*m] = operator()(i,j);
    if (data_) {
      STRUMPACK_SUB_MEMORY(rows_*cols_*sizeof(scalar_t));
      delete[] data_;
    }
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
#pragma omp parallel if(depth < params::task_recursion_cutoff_level && !omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2)
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
#pragma omp parallel if(depth < params::task_recursion_cutoff_level && !omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2)
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
#pragma omp parallel if(depth < params::task_recursion_cutoff_level && !omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2)
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
#pragma omp parallel if(depth < params::task_recursion_cutoff_level && !omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2)
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
#pragma omp parallel if(depth < params::task_recursion_cutoff_level && !omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2)
#endif
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        operator()(i, j) = alpha * operator()(i, j) + B(i, j);
    STRUMPACK_FLOPS((is_complex<scalar_t>()?8:2)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::scale_rows(const std::vector<scalar_t>& D, int depth) {
    assert(D.size() == rows());
    scale_rows(D.data(), depth);
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::scale_rows(const scalar_t* D, int depth) {
    const auto m = rows();
    const auto n = cols();
#if defined(_OPENMP) && defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp parallel if(depth < params::task_recursion_cutoff_level && !omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2)
#endif
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        operator()(i, j) *= D[i];
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::scale_rows_real
  (const std::vector<real_t>& D, int depth) {
    assert(D.size() == rows());
    scale_rows_real(D.data(), depth);
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::scale_rows_real(const real_t* D, int depth) {
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
#pragma omp parallel if(depth < params::task_recursion_cutoff_level && !omp_in_parallel())
#pragma omp single nowait
#pragma omp taskloop default(shared) collapse(2)
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
    std::vector<int> piv;
    int info = LU(piv, depth);
    if (info) {
      std::cerr << "ERROR: LU factorization failed with info="
                << info << std::endl;
      exit(1);
    }
    return piv;
  }

  template<typename scalar_t> int
  DenseMatrix<scalar_t>::LU(std::vector<int>& piv, int depth) {
    piv.resize(rows());
#if defined(_OPENMP)
    bool in_par = depth < params::task_recursion_cutoff_level
      && omp_in_parallel();
#else
    bool in_par = false;
#endif
    if (in_par)
      return getrf_omp_task(rows(), cols(), data(), ld(), piv.data(), depth);
    else
      return blas::getrf(rows(), cols(), data(), ld(), piv.data());
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
    DenseMatrix<scalar_t> x(b);
    if (!rows()) return x;
    int info = getrs_omp_task
      (char(Trans::N), rows(), b.cols(), data(), ld(),
       piv.data(), x.data(), x.ld(), depth);
    if (info) {
      std::cerr << "ERROR: LU solve failed with info=" << info << std::endl;
      exit(1);
    }
    return x;
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::solve_LU_in_place
  (DenseMatrix<scalar_t>& b, const std::vector<int>& piv, int depth) const {
    assert(piv.size() >= rows());
    solve_LU_in_place(b, piv.data(), depth);
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::solve_LU_in_place
  (DenseMatrix<scalar_t>& b, const int* piv, int depth) const {
    assert(b.rows() == rows());
    if (!rows()) return;
    int info = getrs_omp_task
      (char(Trans::N), rows(), b.cols(), data(), ld(),
       piv, b.data(), b.ld(), depth);
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
    DenseMatrix<scalar_t> tmp(std::max(rows(), cols()), cols(), *this, 0, 0);
    int info = blas::gelqf(rows(), cols(), tmp.data(), tmp.ld(), tau.get());
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
    info = blas::xxglq(cols(), cols(), std::min(rows(), cols()),
                       tmp.data(), tmp.ld(), tau.get());
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
    blas::geqrf(rows(), minmn, data(), ld(), tau.get());
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
      blas::geqp3tol(m, n, data(), ld(), iind.data(), tau.get(),
                     rank, rel_tol, abs_tol);
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
       tau.get(), rank, rel_tol, abs_tol);
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

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::write(const std::string& fname) const {
    std::ofstream f(fname, std::ios::out | std::ios::trunc);
    f << *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>
  DenseMatrix<scalar_t>::read(const std::string& fname) {
    std::ifstream f;
    try {
      f.open(fname);
    } catch (std::ios_base::failure& e) {
      std::cerr << e.what() << std::endl;
    }
    DenseMatrix<scalar_t> D;
    f >> D;
    return D;
  }

  template<typename scalar_t> std::ofstream&
  operator<<(std::ofstream& os, const DenseMatrix<scalar_t>& D) {
    int v[3];
    get_version(v, v+1, v+2);
    os.write((const char*)v, sizeof(v));
    os.write((const char*)(&D), sizeof(DenseMatrix<scalar_t>));
    os.write((const char*)(D.data()), sizeof(scalar_t)*D.rows()*D.cols());
    return os;
  }
  template std::ofstream& operator<<(std::ofstream& os, const DenseMatrix<float>& D);
  template std::ofstream& operator<<(std::ofstream& os, const DenseMatrix<double>& D);
  template std::ofstream& operator<<(std::ofstream& os, const DenseMatrix<std::complex<float>>& D);
  template std::ofstream& operator<<(std::ofstream& os, const DenseMatrix<std::complex<double>>& D);

  template<typename scalar_t> std::ifstream&
  operator>>(std::ifstream& is, DenseMatrix<scalar_t>& D) {
    int v[3], vf[3];
    get_version(v, v+1, v+2);
    is.read((char*)vf, sizeof(vf));
    if (v[0] != vf[0] || v[1] != vf[1] || v[2] != vf[2]) {
      std::cerr << "Warning, file was created with a different"
                << " strumpack version (v"
                << vf[0] << "." << vf[1] << "." << vf[2]
                << " instead of v"
                << v[0] << "." << v[1] << "." << v[2]
                << ")" << std::endl;
    }
    is.read((char*)&D, sizeof(DenseMatrix<scalar_t>));
    D.data_ = new scalar_t[D.rows()*D.cols()];
    is.read((char*)D.data(), sizeof(scalar_t)*D.rows()*D.cols());
    return is;
  }
  template std::ifstream& operator>>(std::ifstream& os, DenseMatrix<float>& D);
  template std::ifstream& operator>>(std::ifstream& os, DenseMatrix<double>& D);
  template std::ifstream& operator>>(std::ifstream& os, DenseMatrix<std::complex<float>>& D);
  template std::ifstream& operator>>(std::ifstream& os, DenseMatrix<std::complex<double>>& D);


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
       DenseMatrix<scalar_t>& c, int depth) {
    assert((ta==Trans::N && a.rows()==c.rows()) ||
           (ta!=Trans::N && a.cols()==c.rows()));
    assert((tb==Trans::N && b.cols()==c.cols()) ||
           (tb!=Trans::N && b.rows()==c.cols()));
    assert((ta==Trans::N && tb==Trans::N && a.cols()==b.rows()) ||
           (ta!=Trans::N && tb==Trans::N && a.rows()==b.rows()) ||
           (ta==Trans::N && tb!=Trans::N && a.cols()==b.cols()) ||
           (ta!=Trans::N && tb!=Trans::N && a.rows()==b.cols()));
#if defined(_OPENMP)
    bool in_par = depth < params::task_recursion_cutoff_level
      && omp_in_parallel();
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
       DenseMatrix<scalar_t>& c, int depth) {
    assert((ta==Trans::N && a.rows()==c.rows()) ||
           (ta!=Trans::N && a.cols()==c.rows()));
#if defined(_OPENMP)
    bool in_par = depth < params::task_recursion_cutoff_level
      && omp_in_parallel();
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
       scalar_t* c, int ldc, int depth) {
    assert((ta==Trans::N && tb==Trans::N && a.cols()==b.rows()) ||
           (ta!=Trans::N && tb==Trans::N && a.rows()==b.rows()) ||
           (ta==Trans::N && tb!=Trans::N && a.cols()==b.cols()) ||
           (ta!=Trans::N && tb!=Trans::N && a.rows()==b.cols()));
#if defined(_OPENMP)
    bool in_par = depth < params::task_recursion_cutoff_level
      && omp_in_parallel();
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
       int depth) {
#if defined(_OPENMP)
    bool in_par = depth < params::task_recursion_cutoff_level
      && omp_in_parallel();
#else
    bool in_par = false;
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
       int depth) {
#if defined(_OPENMP)
    bool in_par = depth < params::task_recursion_cutoff_level
      && omp_in_parallel();
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
       DenseMatrix<scalar_t>& b, int depth) {
    assert(b.cols() == 1);
    assert(a.rows() == a.cols() && a.cols() == b.rows());
#if defined(_OPENMP)
    bool in_par = depth < params::task_recursion_cutoff_level
      && omp_in_parallel();
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
       DenseMatrix<scalar_t>& y, int depth) {
    assert(x.cols() == 1);
    assert(y.cols() == 1);
    assert(ta != Trans::N || (a.rows() == y.rows() && a.cols() == x.rows()));
    assert(ta == Trans::N || (a.cols() == y.rows() && a.rows() == x.rows()));
#if defined(_OPENMP)
    bool in_par = depth < params::task_recursion_cutoff_level
      && omp_in_parallel();
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
       DenseMatrix<scalar_t>& y, int depth) {
    assert(y.cols() == 1);
    assert(ta != Trans::N || (a.rows() == y.rows()));
    assert(ta == Trans::N || (a.cols() == y.rows()));
#if defined(_OPENMP)
    bool in_par = depth < params::task_recursion_cutoff_level
      && omp_in_parallel();
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
       scalar_t* y, int incy, int depth) {
    assert(x.cols() == 1);
    assert(ta != Trans::N || (a.cols() == x.rows()));
    assert(ta == Trans::N || (a.rows() == x.rows()));
#if defined(_OPENMP)
    bool in_par = depth < params::task_recursion_cutoff_level
      && omp_in_parallel();
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
       scalar_t* y, int incy, int depth) {
#if defined(_OPENMP)
    bool in_par = depth < params::task_recursion_cutoff_level
      && omp_in_parallel();
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

  template<typename scalar_t,typename cast_t> DenseMatrix<cast_t>
  cast_matrix(const DenseMatrix<scalar_t>& mat) {
    auto m = mat.rows();
    auto n = mat.cols();
    DenseMatrix<cast_t> A(m, n);
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        A(i, j) = mat(i, j);
    return A;
  }

  // explicit template instantiations
  //template class DenseMatrix<int>;
  template class DenseMatrix<float>;
  template class DenseMatrix<double>;
  template class DenseMatrix<std::complex<float>>;
  template class DenseMatrix<std::complex<double>>;

  template class DenseMatrixWrapper<float>;
  template class DenseMatrixWrapper<double>;
  template class DenseMatrixWrapper<std::complex<float>>;
  template class DenseMatrixWrapper<std::complex<double>>;

  // DenseMatrix<int/bool> only supports a few operations
  template DenseMatrix<int>::DenseMatrix();
  template DenseMatrix<unsigned int>::DenseMatrix();
  template DenseMatrix<std::size_t>::DenseMatrix();
  template DenseMatrix<bool>::DenseMatrix();
  template DenseMatrix<int>::DenseMatrix(std::size_t, std::size_t);
  template DenseMatrix<unsigned int>::DenseMatrix(std::size_t, std::size_t);
  template DenseMatrix<std::size_t>::DenseMatrix(std::size_t, std::size_t);
  template DenseMatrix<bool>::DenseMatrix(std::size_t, std::size_t);
  template DenseMatrix<int>::~DenseMatrix();
  template DenseMatrix<unsigned int>::~DenseMatrix();
  template DenseMatrix<std::size_t>::~DenseMatrix();
  template DenseMatrix<bool>::~DenseMatrix();
  template void DenseMatrix<int>::fill(int v);
  template void DenseMatrix<unsigned int>::fill(unsigned int v);
  template void DenseMatrix<std::size_t>::fill(std::size_t v);
  template void DenseMatrix<bool>::fill(bool v);
  template void DenseMatrix<int>::zero();
  template void DenseMatrix<unsigned int>::zero();
  template void DenseMatrix<std::size_t>::zero();
  template void DenseMatrix<bool>::zero();
  template void DenseMatrix<int>::resize(std::size_t, std::size_t);
  template void DenseMatrix<unsigned int>::resize(std::size_t, std::size_t);
  template void DenseMatrix<std::size_t>::resize(std::size_t, std::size_t);
  template void DenseMatrix<bool>::resize(std::size_t, std::size_t);
  template void DenseMatrix<int>::print(std::string, bool, int) const;
  template void DenseMatrix<unsigned int>::print(std::string, bool, int) const;
  template void DenseMatrix<std::size_t>::print(std::string, bool, int) const;
  template void DenseMatrix<bool>::print(std::string, bool, int) const;

  // DenseMatrix<long double> only supports a few operations



  template void
  gemm(Trans ta, Trans tb, float alpha, const DenseMatrix<float>& a,
       const DenseMatrix<float>& b, float beta,
       DenseMatrix<float>& c, int depth);
  template void
  gemm(Trans ta, Trans tb, double alpha, const DenseMatrix<double>& a,
       const DenseMatrix<double>& b, double beta,
       DenseMatrix<double>& c, int depth);
  template void
  gemm(Trans ta, Trans tb, std::complex<float> alpha,
       const DenseMatrix<std::complex<float>>& a,
       const DenseMatrix<std::complex<float>>& b, std::complex<float> beta,
       DenseMatrix<std::complex<float>>& c, int depth);
  template void
  gemm(Trans ta, Trans tb, std::complex<double> alpha,
       const DenseMatrix<std::complex<double>>& a,
       const DenseMatrix<std::complex<double>>& b, std::complex<double> beta,
       DenseMatrix<std::complex<double>>& c, int depth);

  template void
  gemm(Trans ta, Trans tb, float alpha, const DenseMatrix<float>& a,
       const float* b, int ldb, float beta,
       DenseMatrix<float>& c, int depth);
  template void
  gemm(Trans ta, Trans tb, double alpha, const DenseMatrix<double>& a,
       const double* b, int ldb, double beta,
       DenseMatrix<double>& c, int depth);
  template void
  gemm(Trans ta, Trans tb, std::complex<float> alpha,
       const DenseMatrix<std::complex<float>>& a,
       const std::complex<float>* b, int ldb, std::complex<float> beta,
       DenseMatrix<std::complex<float>>& c, int depth);
  template void
  gemm(Trans ta, Trans tb, std::complex<double> alpha,
       const DenseMatrix<std::complex<double>>& a,
       const std::complex<double>* b, int ldb, std::complex<double> beta,
       DenseMatrix<std::complex<double>>& c, int depth);

  template void
  gemm(Trans ta, Trans tb, float alpha, const DenseMatrix<float>& a,
       const DenseMatrix<float>& b, float beta,
       float* c, int ldc, int depth);
  template void
  gemm(Trans ta, Trans tb, double alpha, const DenseMatrix<double>& a,
       const DenseMatrix<double>& b, double beta,
       double* c, int ldc, int depth);
  template void
  gemm(Trans ta, Trans tb, std::complex<float> alpha,
       const DenseMatrix<std::complex<float>>& a,
       const DenseMatrix<std::complex<float>>& b, std::complex<float> beta,
       std::complex<float>* c, int ldc, int depth);
  template void
  gemm(Trans ta, Trans tb, std::complex<double> alpha,
       const DenseMatrix<std::complex<double>>& a,
       const DenseMatrix<std::complex<double>>& b, std::complex<double> beta,
       std::complex<double>* c, int ldc, int depth);

  template void
  trmm(Side s, UpLo ul, Trans ta, Diag d, float alpha,
       const DenseMatrix<float>& a, DenseMatrix<float>& b,
       int depth);
  template void
  trmm(Side s, UpLo ul, Trans ta, Diag d, double alpha,
       const DenseMatrix<double>& a, DenseMatrix<double>& b,
       int depth);
  template void
  trmm(Side s, UpLo ul, Trans ta, Diag d, std::complex<float> alpha,
       const DenseMatrix<std::complex<float>>& a,
       DenseMatrix<std::complex<float>>& b, int depth);
  template void
  trmm(Side s, UpLo ul, Trans ta, Diag d, std::complex<double> alpha,
       const DenseMatrix<std::complex<double>>& a,
       DenseMatrix<std::complex<double>>& b, int depth);

  template void
  trsm(Side s, UpLo ul, Trans ta, Diag d, float alpha,
       const DenseMatrix<float>& a, DenseMatrix<float>& b,
       int depth);
  template void
  trsm(Side s, UpLo ul, Trans ta, Diag d, double alpha,
       const DenseMatrix<double>& a, DenseMatrix<double>& b,
       int depth);
  template void
  trsm(Side s, UpLo ul, Trans ta, Diag d, std::complex<float> alpha,
       const DenseMatrix<std::complex<float>>& a,
       DenseMatrix<std::complex<float>>& b, int depth);
  template void
  trsm(Side s, UpLo ul, Trans ta, Diag d, std::complex<double> alpha,
       const DenseMatrix<std::complex<double>>& a,
       DenseMatrix<std::complex<double>>& b, int depth);

  template void
  trsv(UpLo ul, Trans ta, Diag d, const DenseMatrix<float>& a,
       DenseMatrix<float>& b, int depth);
  template void
  trsv(UpLo ul, Trans ta, Diag d, const DenseMatrix<double>& a,
       DenseMatrix<double>& b, int depth);
  template void
  trsv(UpLo ul, Trans ta, Diag d, const DenseMatrix<std::complex<float>>& a,
       DenseMatrix<std::complex<float>>& b, int depth);
  template void
  trsv(UpLo ul, Trans ta, Diag d, const DenseMatrix<std::complex<double>>& a,
       DenseMatrix<std::complex<double>>& b, int depth);

  template void
  gemv(Trans ta, float alpha, const DenseMatrix<float>& a,
       const DenseMatrix<float>& x, float beta,
       DenseMatrix<float>& y, int depth);
  template void
  gemv(Trans ta, double alpha, const DenseMatrix<double>& a,
       const DenseMatrix<double>& x, double beta,
       DenseMatrix<double>& y, int depth);
  template void
  gemv(Trans ta, std::complex<float> alpha,
       const DenseMatrix<std::complex<float>>& a,
       const DenseMatrix<std::complex<float>>& x, std::complex<float> beta,
       DenseMatrix<std::complex<float>>& y, int depth);
  template void
  gemv(Trans ta, std::complex<double> alpha,
       const DenseMatrix<std::complex<double>>& a,
       const DenseMatrix<std::complex<double>>& x, std::complex<double> beta,
       DenseMatrix<std::complex<double>>& y, int depth);

  template void
  gemv(Trans ta, float alpha, const DenseMatrix<float>& a,
       const float* x, int incx, float beta,
       DenseMatrix<float>& y, int depth);
  template void
  gemv(Trans ta, double alpha, const DenseMatrix<double>& a,
       const double* x, int incx, double beta,
       DenseMatrix<double>& y, int depth);
  template void
  gemv(Trans ta, std::complex<float> alpha,
       const DenseMatrix<std::complex<float>>& a,
       const std::complex<float>* x, int incx, std::complex<float> beta,
       DenseMatrix<std::complex<float>>& y, int depth);
  template void
  gemv(Trans ta, std::complex<double> alpha,
       const DenseMatrix<std::complex<double>>& a,
       const std::complex<double>* x, int incx, std::complex<double> beta,
       DenseMatrix<std::complex<double>>& y, int depth);

  template void
  gemv(Trans ta, float alpha, const DenseMatrix<float>& a,
       const DenseMatrix<float>& x, float beta,
       float* y, int incy, int depth);
  template void
  gemv(Trans ta, double alpha, const DenseMatrix<double>& a,
       const DenseMatrix<double>& x, double beta,
       double* y, int incy, int depth);
  template void
  gemv(Trans ta, std::complex<float> alpha,
       const DenseMatrix<std::complex<float>>& a,
       const DenseMatrix<std::complex<float>>& x, std::complex<float> beta,
       std::complex<float>* y, int incy, int depth);
  template void
  gemv(Trans ta, std::complex<double> alpha,
       const DenseMatrix<std::complex<double>>& a,
       const DenseMatrix<std::complex<double>>& x, std::complex<double> beta,
       std::complex<double>* y, int incy, int depth);

  template void
  gemv(Trans ta, float alpha, const DenseMatrix<float>& a,
       const float* x, int incx, float beta,
       float* y, int incy, int depth);
  template void
  gemv(Trans ta, double alpha, const DenseMatrix<double>& a,
       const double* x, int incx, double beta,
       double* y, int incy, int depth);
  template void
  gemv(Trans ta, std::complex<float> alpha,
       const DenseMatrix<std::complex<float>>& a,
       const std::complex<float>* x, int incx, std::complex<float> beta,
       std::complex<float>* y, int incy, int depth);
  template void
  gemv(Trans ta, std::complex<double> alpha,
       const DenseMatrix<std::complex<double>>& a,
       const std::complex<double>* x, int incx, std::complex<double> beta,
       std::complex<double>* y, int incy, int depth);

  template DenseMatrix<float>
  cast_matrix<double,float>(const DenseMatrix<double>& mat);
  template DenseMatrix<double>
  cast_matrix<float,double>(const DenseMatrix<float>& mat);
  template DenseMatrix<std::complex<float>>
  cast_matrix<std::complex<double>,std::complex<float>>
  (const DenseMatrix<std::complex<double>>& mat);
  template DenseMatrix<std::complex<double>>
  cast_matrix<std::complex<float>,std::complex<double>>
  (const DenseMatrix<std::complex<float>>& mat);

} // end namespace strumpack
