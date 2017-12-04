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
#ifndef DENSE_MATRIX_HPP
#define DENSE_MATRIX_HPP

#include <string>
#include <iomanip>
#include <cassert>
#include <algorithm>

#include "misc/RandomWrapper.hpp"
#include "misc/TaskTimer.hpp"
#include "BLASLAPACKWrapper.hpp"
#include "BLASLAPACKOpenMPTask.hpp"

namespace strumpack {

  enum class Trans : char { N='N', C='C', T='T' };
  enum class Side : char { L='L', R='R' };
  enum class UpLo : char { U='U', L='L' };
  enum class Diag : char { U='U', N='N' };

  template<typename scalar_t> class DistributedMatrix;

  template<typename scalar_t> class DenseMatrix {
    using real_t = typename RealType<scalar_t>::value_type;

  protected:
    scalar_t* _data;
    std::size_t _rows;
    std::size_t _cols;
    std::size_t _ld;

  public:
    DenseMatrix();
    DenseMatrix(std::size_t m, std::size_t n);
    DenseMatrix
    (std::size_t m, std::size_t n, const scalar_t* D, std::size_t ld);
    DenseMatrix
    (std::size_t m, std::size_t n, const DenseMatrix<scalar_t>& D,
     std::size_t i, std::size_t j);
    DenseMatrix(const DenseMatrix<scalar_t>& D);
    DenseMatrix(DenseMatrix<scalar_t>&& D);
    virtual ~DenseMatrix();

    DenseMatrix<scalar_t>& operator=(const DenseMatrix<scalar_t>& D);
    DenseMatrix<scalar_t>& operator=(DenseMatrix<scalar_t>&& D);

    inline std::size_t rows() const { return _rows; }
    inline std::size_t cols() const { return _cols; }
    inline std::size_t ld() const { return _ld; }
    inline const scalar_t* data() const { return _data; }
    inline scalar_t* data() { return _data; }
    inline const scalar_t& operator()(std::size_t i, std::size_t j) const
    { assert(i>=0 && i<=rows() && j>=0 && j<=cols()); return _data[i+_ld*j]; }
    inline const scalar_t* ptr(std::size_t i, std::size_t j) const
    { assert(i>=0 && i<=rows() && j>=0 && j<=cols()); return _data+i+_ld*j; }
    inline scalar_t& operator()(std::size_t i, std::size_t j)
    { assert(i>=0 && i<=rows() && j>=0 && j<=cols()); return _data[i+_ld*j]; }
    inline scalar_t* ptr(std::size_t i, std::size_t j)
    { assert(i>=0 && i<=rows() && j>=0 && j<=cols()); return _data+i+_ld*j; }

    void print() const { print("A"); }
    void print(std::string name, bool all=false, int width=8) const;
    void print_to_file
    (std::string name, std::string filename, int width=8) const;
    void random();
    void random
    (random::RandomGeneratorBase<typename RealType<scalar_t>::
     value_type>& rgen);
    void fill(scalar_t v);
    void zero();
    void eye();
    virtual void clear();
    void resize(std::size_t m, std::size_t n);
    void hconcat(const DenseMatrix<scalar_t>& b);
    void copy
    (const DenseMatrix<scalar_t>& B, std::size_t i=0, std::size_t j=0);
    void copy(const scalar_t* B, std::size_t ldb);
    DenseMatrix<scalar_t> transpose() const;

    void laswp(const std::vector<int>& P, bool fwd);
    void lapmr(const std::vector<int>& P, bool fwd);

    void extract_rows
    (const std::vector<std::size_t>& I, const DenseMatrix<scalar_t>& B);
    DenseMatrix<scalar_t> extract_rows
    (const std::vector<std::size_t>& I) const;
    DenseMatrix<scalar_t> extract
    (const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J) const;
    DenseMatrix<scalar_t>& scatter_rows_add
    (const std::vector<std::size_t>& I, const DenseMatrix<scalar_t>& B);
    DenseMatrix<scalar_t>& add(const DenseMatrix<scalar_t>& B);
    DenseMatrix<scalar_t>& sub(const DenseMatrix<scalar_t>& B);
    DenseMatrix<scalar_t>& scaled_add
    (scalar_t alpha, const DenseMatrix<scalar_t>& x);
    DenseMatrix<scalar_t>& scale_and_add
    (scalar_t alpha, const DenseMatrix<scalar_t>& x);
    DenseMatrix<scalar_t>& scale_rows(const std::vector<scalar_t>& D);
    DenseMatrix<scalar_t>& div_rows(const std::vector<scalar_t>& D);
    real_t norm() const;
    real_t normF() const;
    real_t norm1() const;
    real_t normI() const;
    virtual std::size_t memory() const {
      return sizeof(scalar_t) * rows() * cols();
    }
    virtual std::size_t nonzeros() const {
      return rows()*cols();
    }

    std::vector<int> LU(int depth);
    DenseMatrix<scalar_t> solve
    (const DenseMatrix<scalar_t>& b,
     const std::vector<int>& piv, int depth) const;
    void LQ
    (DenseMatrix<scalar_t>& L, DenseMatrix<scalar_t>& Q, int depth) const;
    void orthogonalize(scalar_t& r_max, scalar_t& r_min, int depth);
    void ID_column
    (DenseMatrix<scalar_t>& X, std::vector<int>& piv,
     std::vector<std::size_t>& ind, real_t rel_tol,
     real_t abs_tol, int max_rank, int depth);
    void ID_row
    (DenseMatrix<scalar_t>& X, std::vector<int>& piv,
     std::vector<std::size_t>& ind, real_t rel_tol, real_t abs_tol,
     int max_rank, int depth) const;
    std::vector<scalar_t> singular_values() const;

  private:
    void ID_column_MGS
    (DenseMatrix<scalar_t>& X, std::vector<int>& piv,
     std::vector<std::size_t>& ind, real_t rel_tol,
     real_t abs_tol, int max_rank, int depth);
    void ID_column_GEQP3
    (DenseMatrix<scalar_t>& X, std::vector<int>& piv,
     std::vector<std::size_t>& ind, real_t rel_tol,
     real_t abs_tol, int max_rank, int depth);
    template<typename T> friend class DistributedMatrix;
  };

  template<typename scalar_t>
  class DenseMatrixWrapper : public DenseMatrix<scalar_t> {
  public:
    DenseMatrixWrapper() : DenseMatrix<scalar_t>() {}
    DenseMatrixWrapper
    (std::size_t m, std::size_t n, scalar_t* D, std::size_t ld) {
      this->_data = D; this->_rows = m; this->_cols = n;
      this->_ld = std::max(std::size_t(1), ld);
    }
    DenseMatrixWrapper
    (std::size_t m, std::size_t n, DenseMatrix<scalar_t>& D,
     std::size_t i, std::size_t j)
      : DenseMatrixWrapper<scalar_t>(m, n, &D(i, j), D.ld()) {}
    virtual ~DenseMatrixWrapper() { this->_data = nullptr; }

    void clear() {
      this->_rows = 0; this->_cols = 0;
      this->_ld = 1; this->_data = nullptr;
    }
    std::size_t memory() const { return 0; }
    std::size_t nonzeros() const { return 0; }

    DenseMatrixWrapper(const DenseMatrixWrapper<scalar_t>&) = delete;
    DenseMatrixWrapper(const DenseMatrix<scalar_t>&) = delete;
    DenseMatrixWrapper(DenseMatrixWrapper<scalar_t>&&) = delete;
    DenseMatrixWrapper(DenseMatrix<scalar_t>&&) = delete;
    DenseMatrixWrapper<scalar_t>&
    operator=(const DenseMatrixWrapper<scalar_t>& D)
    { this->_data = D.data(); this->_rows = D.rows();
      this->_cols = D.cols(); this->_ld = D.ld(); return *this; }
    DenseMatrixWrapper<scalar_t>&
    operator=(DenseMatrixWrapper<scalar_t>&& D)
    { this->_data = D.data(); this->_rows = D.rows();
      this->_cols = D.cols(); this->_ld = D.ld(); return *this; }
    DenseMatrixWrapper<scalar_t>& operator=(const DenseMatrix<scalar_t>& a) {
      assert(a.rows()==this->rows() && a.cols()==this->cols());
      for (std::size_t j=0; j<this->cols(); j++)
        for (std::size_t i=0; i<this->rows(); i++)
          this->operator()(i, j) = a(i, j);
      return *this;
    }
  };

  template<typename scalar_t>
  std::unique_ptr<const DenseMatrixWrapper<scalar_t>>
  ConstDenseMatrixWrapperPtr
  (std::size_t m, std::size_t n, const scalar_t* D, std::size_t ld) {
    return std::unique_ptr<const DenseMatrixWrapper<scalar_t>>
      (new DenseMatrixWrapper<scalar_t>(m, n, const_cast<scalar_t*>(D), ld));
  }
  template<typename scalar_t>
  std::unique_ptr<const DenseMatrixWrapper<scalar_t>>
  ConstDenseMatrixWrapperPtr
  (std::size_t m, std::size_t n, const DenseMatrix<scalar_t>& D,
   std::size_t i, std::size_t j) {
    return std::unique_ptr<const DenseMatrixWrapper<scalar_t>>
      (new DenseMatrixWrapper<scalar_t>
       (m, n, const_cast<DenseMatrix<scalar_t>&>(D), i, j));
  }


  /** copy submatrix of a at ia,ja of size m,n into b at position ib,jb */
  template<typename scalar_t> void
  copy(std::size_t m, std::size_t n, const DenseMatrix<scalar_t>& a,
       std::size_t ia, std::size_t ja,
   DenseMatrix<scalar_t>& b, std::size_t ib, std::size_t jb) {
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        b(ib+i, jb+j) = a(ia+i, ja+j);
  }

  /** copy matrix a into matrix b at position ib, jb */
  template<typename scalar_t> void
  copy(const DenseMatrix<scalar_t>& a, DenseMatrix<scalar_t>& b,
       std::size_t ib, std::size_t jb) {
    copy(a.rows(), a.cols(), a, 0, 0, b, ib, jb);
  }

  /** copy matrix a into matrix b at position ib, jb */
  template<typename scalar_t> void
  copy(const DenseMatrix<scalar_t>& a, scalar_t* b, std::size_t ldb) {
    for (std::size_t j=0; j<a.cols(); j++)
      for (std::size_t i=0; i<a.rows(); i++)
        b[i+j*ldb] = a(i, j);
  }

  template<typename scalar_t> DenseMatrix<scalar_t>
  vconcat(const DenseMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b) {
    assert(a.cols() == b.cols());
    DenseMatrix<scalar_t> tmp(a.rows()+b.rows(), a.cols());
    copy(a, tmp, 0, 0);
    copy(b, tmp, a.rows(), 0);
    return tmp;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>
  hconcat(const DenseMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b) {
    assert(a.rows() == b.rows());
    DenseMatrix<scalar_t> tmp(a.rows(), a.cols()+b.cols());
    copy(a, tmp, 0, 0);
    copy(b, tmp, 0, a.cols());
    return tmp;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>
  eye(std::size_t m, std::size_t n) {
    DenseMatrix<scalar_t> I(m, n);
    I.eye();
    return I;
  }


  template<typename scalar_t> DenseMatrix<scalar_t>::DenseMatrix()
    : _data(nullptr), _rows(0), _cols(0), _ld(1) { }

  template<typename scalar_t> DenseMatrix<scalar_t>::DenseMatrix
  (std::size_t m, std::size_t n)
    : _data(new scalar_t[m*n]), _rows(m),
      _cols(n), _ld(std::max(std::size_t(1), m)) { }

  template<typename scalar_t> DenseMatrix<scalar_t>::DenseMatrix
  (std::size_t m, std::size_t n, const scalar_t* D, std::size_t ld)
    : _data(new scalar_t[m*n]), _rows(m), _cols(n),
      _ld(std::max(std::size_t(1), m)) {
    assert(ld >= m);
    for (std::size_t j=0; j<_cols; j++)
      for (std::size_t i=0; i<_rows; i++)
        operator()(i, j) = D[i+j*ld];
  }

  template<typename scalar_t> DenseMatrix<scalar_t>::DenseMatrix
  (std::size_t m, std::size_t n, const DenseMatrix<scalar_t>& D,
   std::size_t i, std::size_t j)
    : _data(new scalar_t[m*n]), _rows(m), _cols(n),
      _ld(std::max(std::size_t(1), m)) {
    for (std::size_t _j=0; _j<std::min(_cols, D.cols()-j); _j++)
      for (std::size_t _i=0; _i<std::min(_rows, D.rows()-i); _i++)
        operator()(_i, _j) = D(_i+i, _j+j);
  }

  template<typename scalar_t>
  DenseMatrix<scalar_t>::DenseMatrix(const DenseMatrix<scalar_t>& D)
    : _data(new scalar_t[D.rows()*D.cols()]), _rows(D.rows()),
      _cols(D.cols()), _ld(std::max(std::size_t(1), D.rows())) {
    for (std::size_t j=0; j<_cols; j++)
      for (std::size_t i=0; i<_rows; i++)
        operator()(i, j) = D(i, j);
  }

  template<typename scalar_t>
  DenseMatrix<scalar_t>::DenseMatrix(DenseMatrix<scalar_t>&& D)
    : _data(D.data()), _rows(D.rows()), _cols(D.cols()), _ld(D.ld()) {
    D._data = nullptr;
    D._rows = 0;
    D._cols = 0;
    D._ld = 1;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>::~DenseMatrix() {
    delete[] _data;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::operator=(const DenseMatrix<scalar_t>& D) {
    if (this == &D) return *this;
    if (_rows != D.rows() || _cols != D.cols()) {
      _rows = D.rows();
      _cols = D.cols();
      delete[] _data;
      _data = new scalar_t[_rows*_cols];
      _ld = std::max(std::size_t(1), _rows);
    }
    for (std::size_t j=0; j<_cols; j++)
      for (std::size_t i=0; i<_rows; i++)
        operator()(i,j) = D(i,j);
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::operator=(DenseMatrix<scalar_t>&& D) {
    _rows = D.rows();
    _cols = D.cols();
    _ld = D.ld();
    delete[] _data;
    _data = D.data();
    D._data = nullptr;
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
    _rows = 0;
    _cols = 0;
    _ld = 1;
    delete[] _data;
    _data = nullptr;
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::resize(std::size_t m, std::size_t n) {
    auto tmp = new scalar_t[m*n];
    for (std::size_t j=0; j<std::min(cols(),n); j++)
      for (std::size_t i=0; i<std::min(rows(),m); i++)
        tmp[i+j*m] = operator()(i,j);
    delete[] _data;
    _data = tmp;
    _ld = std::max(std::size_t(1), m);
    _rows = m;
    _cols = n;
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
  DenseMatrix<scalar_t>::laswp(const std::vector<int>& P, bool fwd) {
    blas::laswp(cols(), data(), ld(), 1, rows(), P.data(), fwd ? 1 : -1);
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::lapmr(const std::vector<int>& P, bool fwd) {
    blas::lapmr(fwd, rows(), cols(), data(), ld(), P.data());
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::extract_rows
  (const std::vector<std::size_t>& I, const DenseMatrix<scalar_t>& B) {
    assert(rows() == I.size());
    assert(cols() == B.cols());
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<rows(); i++) {
        assert(I[i] >= 0 && I[i] < B.rows());
        operator()(i, j) = B(I[i], j);
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

  template<typename scalar_t> DenseMatrix<scalar_t>
  DenseMatrix<scalar_t>::extract(const std::vector<std::size_t>& I,
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
  (const std::vector<std::size_t>& I, const DenseMatrix<scalar_t>& B) {
    assert(I.size() == B.rows());
    assert(B.cols() == cols());
    //#pragma omp taskloop default(none) firstprivate(c,r,l,d) shared(I,B) //grainsize(128) //collapse(2)
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<I.size(); i++) {
        assert(I[i] < rows());
        operator()(I[i], j) += B(i, j);
      }
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::add(const DenseMatrix<scalar_t>& B) {
    //#pragma omp taskloop default(shared) //grainsize(128) //collapse(2)
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<rows(); i++)
        operator()(i, j) += B(i, j);
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::sub(const DenseMatrix<scalar_t>& B) {
    //#pragma omp taskloop default(shared) //grainsize(128) //collapse(2)
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<rows(); i++)
        operator()(i, j) -= B(i, j);
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::scaled_add
  (scalar_t alpha, const DenseMatrix<scalar_t>& x) {
    //#pragma omp taskloop default(shared) //grainsize(128) //collapse(2)
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<rows(); i++)
        operator()(i, j) += alpha * x(i, j);
    STRUMPACK_FLOPS((is_complex<scalar_t>()?8:2)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::scale_and_add
  (scalar_t alpha, const DenseMatrix<scalar_t>& x) {
    //#pragma omp taskloop default(shared) //grainsize(128) //collapse(2)
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<rows(); i++)
        operator()(i, j) = alpha * operator()(i, j) + x(i, j);
    STRUMPACK_FLOPS((is_complex<scalar_t>()?8:2)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::scale_rows(const std::vector<scalar_t>& D) {
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<rows(); i++)
        operator()(i, j) *= D[i];
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*cols()*rows());
    return *this;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>&
  DenseMatrix<scalar_t>::div_rows(const std::vector<scalar_t>& D) {
    for (std::size_t j=0; j<cols(); j++)
      for (std::size_t i=0; i<rows(); i++)
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

  // TODO do in-place??!! to avoid copy
  template<typename scalar_t> DenseMatrix<scalar_t>
  DenseMatrix<scalar_t>::solve(const DenseMatrix<scalar_t>& b,
                               const std::vector<int>& piv, int depth) const {
    int info = 0;
    DenseMatrix<scalar_t> x(b);
    getrs_omp_task
      (char(Trans::N), rows(), b.cols(), data(), ld(), piv.data(),
       x.data(), x.ld(), &info, depth);
    if (info) {
      std::cerr << "ERROR: LU solve failed with info=" << info << std::endl;
      exit(1);
    }
    return x;
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::LQ
  (DenseMatrix<scalar_t>& L, DenseMatrix<scalar_t>& Q, int depth) const {
    auto minmn = std::min(rows(), cols());
    auto tau = new scalar_t[minmn];
    int info;
    DenseMatrix<scalar_t> tmp(std::max(rows(), cols()), cols(), *this, 0, 0);
    blas::gelqfmod(rows(), cols(), tmp.data(), tmp.ld(), tau, &info, depth);
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
                   tmp.data(), tmp.ld(), tau, &info, depth);
    Q = DenseMatrix<scalar_t>(cols(), cols(), tmp, 0, 0); // generate Q
    if (info) {
      std::cerr << "ERROR: generation of Q from LQ failed with info="
                << info << std::endl;
      exit(1);
    }
    delete[] tau;
  }

  template<typename scalar_t> void
  DenseMatrix<scalar_t>::orthogonalize
  (scalar_t& r_max, scalar_t& r_min, int depth) {
    if (!cols() || !rows()) return;
    int info;
    int minmn = std::min(rows(), cols());
    auto tau = new scalar_t[minmn];
    blas::geqrfmod(rows(), minmn, data(), ld(), tau, &info, depth);
    // TODO threading!!
    real_t Rmax = std::abs(operator()(0, 0));
    real_t Rmin = Rmax;
    for (int i=0; i<minmn; i++) {
      auto Rii = std::abs(operator()(i, i));
      Rmax = std::max(Rmax, Rii);
      Rmin = std::min(Rmin, Rii);
    }
    r_max = Rmax;
    r_min = Rmin;
    blas::xxgqr(rows(), minmn, minmn, data(), ld(), tau, &info);
    if (cols() > rows()) {
      DenseMatrixWrapper<scalar_t> tmp
        (rows(), cols()-rows(), *this, 0, rows());
      tmp.zero();
    }
    delete[] tau;
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
    // GEQP3 is much more accurate!!
    ID_column_GEQP3(X, piv, ind, rel_tol, abs_tol, max_rank, depth);
    //ID_column_MGS(X, piv, ind, rel_tol, abs_tol, max_rank, depth);
  }

  template<typename scalar_t> void DenseMatrix<scalar_t>::ID_column_GEQP3
  (DenseMatrix<scalar_t>& X, std::vector<int>& piv,
   std::vector<std::size_t>& ind, real_t rel_tol, real_t abs_tol,
   int max_rank, int depth) {
    int m = rows(), n = cols();
    auto tau = new scalar_t[std::max(1,std::min(m, n))];
    piv.resize(n);
    std::vector<int> iind(n);
    int rank, info;
    // TODO make geqp3tol stop at max_rank
    blas::geqp3tol(m, n, data(), ld(), iind.data(), tau, &info,
                   rank, rel_tol, abs_tol, depth);
    rank = std::min(rank, max_rank);
    delete[] tau;
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

  template<typename scalar_t> void DenseMatrix<scalar_t>::ID_column_MGS
  (DenseMatrix<scalar_t>& X, std::vector<int>& piv,
   std::vector<std::size_t>& ind, real_t rel_tol, real_t abs_tol,
   int max_rank, int depth) {
    int m = rows(), n = cols();
    piv.resize(n);
    ind.resize(n);
    DenseMatrix<scalar_t> R(m, n);
    auto cnrms = new real_t[n];
    //#pragma omp taskloop grainsize(32) default(shared)
    //if(depth < params::task_recursion_cutoff_level)
    //final(depth >= params::task_recursion_cutoff_level-1) mergeable
    for (std::size_t i=0; i<n; i++) {
      cnrms[i] = blas::nrm2(m, ptr(0, i), 1);
      cnrms[i] *= cnrms[i];
      ind[i] = i;
      piv[i] = i+1;
    }
    auto R00 = real_t(1.);
    int i = 0, NB = 32, mn = std::min(m, n);
    bool conv = false;
    for (int j=0; j<mn && !conv; j+=NB) {
      for (i=j; i<std::min(mn,j+NB); i++) {
        auto cmax = std::max_element(cnrms+i, cnrms+n) - cnrms;
        if (cmax != i) {
          piv[i] = cmax+1;
          std::swap(ind[i], ind[cmax]);
          std::swap(cnrms[i], cnrms[cmax]);
          blas::swap(m, ptr(0, i), 1, ptr(0, cmax), 1);
          blas::swap(i, &R(0, i), 1, R.ptr(0, cmax), 1);
        }
        gemv_omp_task('N', m, i-j, scalar_t(-1.), ptr(0, j), ld(),
                      R.ptr(j, i), 1, scalar_t(1.), ptr(0, i), 1, depth);
        auto Rii = blas::nrm2(m, ptr(0, i), 1);
        if (i == 0) R00 = Rii;
        if (Rii/R00 <= rel_tol || Rii <= abs_tol || i == max_rank) {
          conv = true;
          break;
        }
        R(i, i) = Rii;
        blas::scal(m, scalar_t(1.)/Rii, ptr(0, i), 1);
        if (i < n-1) {
          gemv_omp_task('C', m, n-(i+1), scalar_t(1.), ptr(0, i+1), ld(),
                        ptr(0, i), 1, scalar_t(0.),
                        R.ptr(i, i+1), R.ld(), depth);
          blas::lacgv(n-(i+1), R.ptr(i, i+1), R.ld());
          for (int k=i+1; k<n; k++)
            cnrms[k] -= std::real(std::conj(R(i, k))*R(i, k));
        }
      }
      if (i < n-1)
        gemm_omp_task('N', 'N', m, n-(i+1), i-j, scalar_t(-1.),
                      ptr(0, j), ld(), R.ptr(j, i+1), R.ld(),
                      scalar_t(1.), ptr(0, i+1), ld(), depth);
    }
    delete[] cnrms;
    ind.resize(i);
    trsm_omp_task('L', 'U', 'N', 'N', i, n-i, scalar_t(1.),
                  R.ptr(0, 0), R.ld(), R.ptr(0, i), R.ld(), depth);
    X = DenseMatrix<scalar_t>(i, n-i, R.ptr(0, i), R.ld());
    std::cout << "TODO count flops for ID_column_MGS,"
              << " note that this routine is not stable!" << std::endl;
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
    gemm_omp_task
      (char(ta), char(tb), c.rows(), c.cols(),
       (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(), a.ld(),
       b.data(), b.ld(), beta, c.data(), c.ld(), depth);
  }

  template<typename scalar_t> void
  gemm(Trans ta, Trans tb, scalar_t alpha, const DenseMatrix<scalar_t>& a,
       const scalar_t* b, int ldb, scalar_t beta,
       DenseMatrix<scalar_t>& c, int depth=0) {
    assert((ta==Trans::N && a.rows()==c.rows()) ||
           (ta!=Trans::N && a.cols()==c.rows()));
    gemm_omp_task
      (char(ta), char(tb), c.rows(), c.cols(),
       (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(),
       a.ld(), b, ldb, beta, c.data(), c.ld(), depth);
  }

  template<typename scalar_t> void
  gemm(Trans ta, Trans tb, scalar_t alpha, const DenseMatrix<scalar_t>& a,
       const DenseMatrix<scalar_t>& b, scalar_t beta,
       scalar_t* c, int ldc, int depth=0) {
    assert((ta==Trans::N && tb==Trans::N && a.cols()==b.rows()) ||
           (ta!=Trans::N && tb==Trans::N && a.rows()==b.rows()) ||
           (ta==Trans::N && tb!=Trans::N && a.cols()==b.cols()) ||
           (ta!=Trans::N && tb!=Trans::N && a.rows()==b.cols()));
    gemm_omp_task
      (char(ta), char(tb), (ta==Trans::N) ? a.rows() : a.cols(),
       (tb==Trans::N) ? b.cols() : b.rows(),
       (ta==Trans::N) ? a.cols() : a.rows(), alpha, a.data(), a.ld(),
       b.data(), b.ld(), beta, c, ldc, depth);
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
       const DenseMatrix<scalar_t>& a, DenseMatrix<scalar_t>& b, int depth) {
    trmm_omp_task
      (char(s), char(ul), char(ta), char(d), b.rows(), b.cols(),
       alpha, a.data(), a.ld(), b.data(), b.ld(), depth);
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
       const DenseMatrix<scalar_t>& a, DenseMatrix<scalar_t>& b, int depth) {
    // TODO assertions
    trsm_omp_task
      (char(s), char(ul), char(ta), char(d), b.rows(), b.cols(),
       alpha, a.data(), a.ld(), b.data(), b.ld(), depth);
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
    trsv_omp_task
      (char(ul), char(ta), char(d), a.rows(),
       a.data(), a.ld(), b.data(), 1, depth);
  }

  template<typename scalar_t> void
  gemv(Trans ta, scalar_t alpha, const DenseMatrix<scalar_t>& a,
       const DenseMatrix<scalar_t>& x, scalar_t beta,
       DenseMatrix<scalar_t>& y, int depth) {
    assert(x.cols() == 1);
    assert(y.cols() == 1);
    assert(ta != Trans::N || (a.rows() == y.rows() && a.cols() == x.rows()));
    assert(ta == Trans::N || (a.cols() == y.rows() && a.rows() == x.rows()));
    gemv_omp_task
      (char(ta), a.rows(), a.cols(), alpha, a.data(), a.ld(),
       x.data(), 1, beta, y.data(), 1, depth);
  }

} // end namespace strumpack

#endif // DENSE_MATRIX_HPP
