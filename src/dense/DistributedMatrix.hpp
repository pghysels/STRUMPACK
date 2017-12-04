#ifndef DISTRIBUTED_MATRIX_HPP
#define DISTRIBUTED_MATRIX_HPP

#include <cstddef>
#include <random>
#include <functional>
#include <limits>

#include "misc/MPIWrapper.hpp"
#include "misc/RandomWrapper.hpp"
#include "misc/TaskTimer.hpp"
#include "DenseMatrix.hpp"
#include "ScaLAPACKWrapper.hpp"

namespace strumpack {

  inline int
  indxl2g(int INDXLOC, int NB, int IPROC, int ISRCPROC, int NPROCS)
  { return NPROCS*NB*((INDXLOC-1)/NB) + (INDXLOC-1) % NB +
      ((NPROCS+IPROC-ISRCPROC) % NPROCS)*NB + 1; }
  inline int
  indxg2l(int INDXGLOB, int NB, int IPROC, int ISRCPROC, int NPROCS)
  { return NB*((INDXGLOB-1)/(NB*NPROCS)) + (INDXGLOB-1) % NB + 1; }
  inline int
  indxg2p(int INDXGLOB, int NB, int IPROC, int ISRCPROC, int NPROCS)
  { return ( ISRCPROC + (INDXGLOB - 1) / NB ) % NPROCS; }

  template<typename scalar_t> class DistributedMatrix {
    using real_t = typename RealType<scalar_t>::value_type;

  protected:
    scalar_t* _data = nullptr;
    int _desc[9];
    int _lrows;
    int _lcols;
    int _prows;
    int _pcols;
    int _prow;
    int _pcol;

  public:
    DistributedMatrix();
    DistributedMatrix(int ctxt, const DenseMatrix<scalar_t>& m);
    DistributedMatrix
    (int ctxt, int M, int N,
     const DistributedMatrix<scalar_t>& m, int ctxt_all);
    DistributedMatrix(int ctxt, int M, int N);
    DistributedMatrix(int ctxt, int M, int N, int MB, int NB);
    DistributedMatrix(int desc[9]);
    DistributedMatrix(const DistributedMatrix<scalar_t>& m);
    DistributedMatrix(DistributedMatrix<scalar_t>&& m);
    virtual ~DistributedMatrix();

    DistributedMatrix<scalar_t>&
    operator=(const DistributedMatrix<scalar_t>& m);
    DistributedMatrix<scalar_t>&
    operator=(DistributedMatrix<scalar_t>&& m);

    virtual int rows() const { return _desc[2]; }
    virtual int cols() const { return _desc[3]; }
    inline int lrows() const { return _lrows; }
    inline int lcols() const { return _lcols; }
    inline int ld() const { return _lrows; }
    inline int MB() const { return _desc[4]; }
    inline int NB() const { return _desc[5]; }
    inline int rowblocks() const { return std::ceil(float(lrows()) / MB()); }
    inline int colblocks() const { return std::ceil(float(lcols()) / NB()); }

    virtual int I() const { return 1; }
    virtual int J() const { return 1; }
    virtual void lranges(int& rlo, int& rhi, int& clo, int& chi) const;

    inline const scalar_t* data() const { return _data; }
    inline scalar_t* data() { return _data; }
    inline const scalar_t& operator()(int r, int c) const
    { return _data[r+ld()*c]; }
    inline scalar_t& operator()(int r, int c) { return _data[r+ld()*c]; }

    inline int prow() const { return _prow; }
    inline int pcol() const { return _pcol; }
    inline int prows() const { return _prows; }
    inline int pcols() const { return _pcols; }
    inline int procs() const { return (prows() == -1) ? 0 : prows()*pcols(); }
    inline bool is_master() const { return prow() == 0 && pcol() == 0; }
    inline int rowl2g(int row) const { assert(_prow != -1);
      return indxl2g(row+1, MB(), prow(), 0, prows()) - I(); }
    inline int coll2g(int col) const { assert(_pcol != -1);
      return indxl2g(col+1, NB(), pcol(), 0, pcols()) - J(); }
    inline int rowg2l(int row) const { assert(_prow != -1);
      return indxg2l(row+I(), MB(), prow(), 0, prows()) - 1; }
    inline int colg2l(int col) const { assert(_pcol != -1);
      return indxg2l(col+J(), NB(), pcol(), 0, pcols()) - 1; }
    inline int rowg2p(int row) const { assert(_prow != -1);
      return indxg2p(row+I(), MB(), prow(), 0, prows()); }
    inline int colg2p(int col) const { assert(_pcol != -1);
      return indxg2p(col+J(), NB(), pcol(), 0, pcols()); }
    inline int rank(int r, int c) const {
      return rowg2p(r) + colg2p(c) * prows(); }
    inline bool is_local(int r, int c) const { assert(_prow != -1);
      return rowg2p(r) == prow() && colg2p(c) == pcol(); }

    inline bool fixed() const { return MB()==default_MB && NB()==default_NB; }
    inline int rowl2g_fixed(int row) const {
      assert(_prow != -1); assert(fixed());
      return indxl2g(row+1, default_MB, prow(), 0, prows()) - I(); }
    inline int coll2g_fixed(int col) const {
      assert(_pcol != -1); assert(fixed());
      return indxl2g(col+1, default_NB, pcol(), 0, pcols()) - J(); }
    inline int rowg2l_fixed(int row) const {
      assert(_prow != -1); assert(fixed());
      return indxg2l(row+I(), default_MB, prow(), 0, prows()) - 1; }
    inline int colg2l_fixed(int col) const {
      assert(_pcol != -1); assert(fixed());
      return indxg2l(col+J(), default_NB, pcol(), 0, pcols()) - 1; }
    inline int rowg2p_fixed(int row) const {
      assert(_prow != -1); assert(fixed());
      return indxg2p(row+I(), default_MB, prow(), 0, prows()); }
    inline int colg2p_fixed(int col) const {
      assert(_pcol != -1); assert(fixed());
      return indxg2p(col+J(), default_NB, pcol(), 0, pcols()); }
    inline int rank_fixed(int r, int c) const {
      assert(fixed()); return rowg2p_fixed(r) + colg2p_fixed(c) * prows(); }
    inline bool is_local_fixed(int r, int c) const {
      assert(_prow != -1); assert(fixed());
      return rowg2p_fixed(r) == prow() && colg2p_fixed(c) == pcol(); }

    inline const int* desc() const { return _desc; }
    inline int* desc() { return _desc; }
    inline bool active() const { return _prow != -1; }
    inline int ctxt() const { return _desc[1]; }

    // TODO fixed versions??
    inline const scalar_t& global(int r, int c) const
    { assert(is_local(r, c)); return operator()(rowg2l(r),colg2l(c)); }
    inline scalar_t& global(int r, int c)
    { assert(is_local(r, c)); return operator()(rowg2l(r),colg2l(c)); }
    inline scalar_t& global_fixed(int r, int c) {
      assert(is_local(r, c)); assert(fixed());
      return operator()(rowg2l_fixed(r),colg2l_fixed(c)); }
    inline void global(int r, int c, scalar_t v) {
      if (active() && is_local(r, c)) operator()(rowg2l(r),colg2l(c)) = v;  }
    inline scalar_t all_global(int r, int c) const;

    void print() const { print("A"); }
    void print(std::string name, int precision=15) const;
    void print_to_file
    (std::string name, std::string filename,
     int width=8) const;
    void random();
    void random
    (random::RandomGeneratorBase<typename RealType<scalar_t>::
     value_type>& rgen);
    void zero();
    void fill(scalar_t a);
    void eye();
    void clear();
    virtual void resize(std::size_t m, std::size_t n);
    virtual void hconcat(const DistributedMatrix<scalar_t>& b);
    DistributedMatrix<scalar_t> transpose() const;

    void laswp(const std::vector<int>& P, bool fwd);

    DistributedMatrix<scalar_t>
    extract_rows(const std::vector<std::size_t>& Ir, MPI_Comm comm) const;
    DistributedMatrix<scalar_t>
    extract_cols(const std::vector<std::size_t>& Ic, MPI_Comm comm) const;
    DistributedMatrix<scalar_t> extract
    (const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J) const;
    DistributedMatrix<scalar_t>& add(const DistributedMatrix<scalar_t>& B);
    DistributedMatrix<scalar_t>& scaled_add
    (scalar_t alpha, const DistributedMatrix<scalar_t>& B);
    typename RealType<scalar_t>::value_type norm() const;
    typename RealType<scalar_t>::value_type normF() const;
    typename RealType<scalar_t>::value_type norm1() const;
    typename RealType<scalar_t>::value_type normI() const;
    virtual std::size_t memory() const
    { return sizeof(scalar_t)*lrows()*lcols(); }
    virtual std::size_t total_memory() const
    { return sizeof(scalar_t)*rows()*cols(); }
    virtual std::size_t nonzeros() const { return lrows()*lcols(); }
    virtual std::size_t total_nonzeros() const { return rows()*cols(); }

    void scatter(const DenseMatrix<scalar_t>& a);
    DenseMatrix<scalar_t> gather() const;
    DenseMatrix<scalar_t> all_gather(int ctxt_all) const;

    DenseMatrix<scalar_t> dense_and_clear();
    DenseMatrixWrapper<scalar_t> dense_wrapper();

    std::vector<int> LU();
    DistributedMatrix<scalar_t> solve
    (const DistributedMatrix<scalar_t>& b, const std::vector<int>& piv) const;
    void LQ
    (DistributedMatrix<scalar_t>& L, DistributedMatrix<scalar_t>& Q) const;
    void orthogonalize(scalar_t& r_max, scalar_t& r_min);
    void ID_column
    (DistributedMatrix<scalar_t>& X, std::vector<int>& piv,
     std::vector<std::size_t>& ind, real_t rel_tol, real_t abs_tol);
    void ID_row
    (DistributedMatrix<scalar_t>& X, std::vector<int>& piv,
     std::vector<std::size_t>& ind, real_t rel_tol,
     real_t abs_tol, int ctxt_T);

#ifdef STRUMPACK_PBLAS_BLOCKSIZE
    static const int default_MB = STRUMPACK_PBLAS_BLOCKSIZE;
    static const int default_NB = STRUMPACK_PBLAS_BLOCKSIZE;
#else
    static const int default_MB = 32;
    static const int default_NB = 32;
#endif
  };

  /** copy submatrix of a DistM_t at ia,ja of size m,n into a DenseM_t
      b at proc dest in ctxt_all */
  // TODO rename these copy functions to gemr2d
  template<typename scalar_t> void copy
  (std::size_t m, std::size_t n, const DistributedMatrix<scalar_t>& a,
   std::size_t ia, std::size_t ja, DenseMatrix<scalar_t>& b,
   int dest, int ctxt_all) {
    if (!m || !n) return;
    int b_desc[9];
    scalapack::descset(b_desc, m, n, m, n, 0, dest, ctxt_all, m);
    scalapack::pgemr2d
      (m, n, a.data(), a.I()+ia, a.J()+ja, a.desc(),
       b.data(), 1, 1, b_desc, ctxt_all);
  }

  template<typename scalar_t> void copy
  (std::size_t m, std::size_t n, const DenseMatrix<scalar_t>& a,
   int src, DistributedMatrix<scalar_t>& b,
   std::size_t ib, std::size_t jb, int ctxt_all) {
    if (!m || !n) return;
    int a_desc[9];
    scalapack::descset
      (a_desc, m, n, m, n, 0, src, ctxt_all, std::max(m, a.ld()));
    scalapack::pgemr2d
      (m, n, a.data(), 1, 1, a_desc, b.data(), b.I()+ib, b.J()+jb, b.desc(),
       ctxt_all);
  }

  /** copy submatrix of a at ia,ja of size m,n into b at position ib,jb */
  template<typename scalar_t> void copy
  (std::size_t m, std::size_t n, const DistributedMatrix<scalar_t>& a,
   std::size_t ia, std::size_t ja, DistributedMatrix<scalar_t>& b,
   std::size_t ib, std::size_t jb, int ctxt_all) {
    if (!m || !n) return;
    assert(!a.active() ||
           (m+ia <= std::size_t(a.rows()) && n+ja <= std::size_t(a.cols())));
    assert(!b.active() ||
           (m+ib <= std::size_t(b.rows()) && n+jb <= std::size_t(b.cols())));
    scalapack::pgemr2d
      (m, n, a.data(), a.I()+ia, a.J()+ja, a.desc(),
       b.data(), b.I()+ib, b.J()+jb, b.desc(), ctxt_all);
  }

  /**
   * Wrapper class does exactly the same as a regular DistributedMatrix,
   * but it is initialized with existing memory, so it does not
   * allocate, own or delete the memory
   */
  template<typename scalar_t>
  class DistributedMatrixWrapper : public DistributedMatrix<scalar_t> {
  private:
    int _rows, _cols;
    int _i, _j;
  public:
    DistributedMatrixWrapper() : DistributedMatrix<scalar_t>(),
      _rows(0), _cols(0), _i(0), _j(0) {}
    DistributedMatrixWrapper
    (std::size_t m, std::size_t n, DistributedMatrix<scalar_t>& A,
     std::size_t i, std::size_t j);
    DistributedMatrixWrapper
    (int ctxt, std::size_t m, std::size_t n, scalar_t* A);
    DistributedMatrixWrapper
    (int ctxt, std::size_t m, std::size_t n, int MB, int NB, scalar_t* A);
    DistributedMatrixWrapper
    (int ctxt, int rsrc, int csrc, std::size_t m, std::size_t n,
     DenseMatrix<scalar_t>& A);
    virtual ~DistributedMatrixWrapper() { this->_data = nullptr; }

    int rows() const override { return _rows; }
    int cols() const override { return _cols; }
    int I() const { return _i+1; }
    int J() const { return _j+1; }
    void lranges(int& rlo, int& rhi, int& clo, int& chi) const;

    void resize(std::size_t m, std::size_t n) { assert(1); }
    void hconcat(const DistributedMatrix<scalar_t>& b) { assert(1); }
    void clear()
    { this->_data = nullptr; DistributedMatrix<scalar_t>::clear(); }
    std::size_t memory() const { return 0; }
    std::size_t total_memory() const { return 0; }
    std::size_t nonzeros() const { return 0; }
    std::size_t total_nonzeros() const { return 0; }

    DenseMatrix<scalar_t> dense_and_clear() = delete;
    DenseMatrixWrapper<scalar_t> dense_wrapper() = delete;

    DistributedMatrixWrapper
    (const DistributedMatrixWrapper<scalar_t>&) = delete;
    DistributedMatrixWrapper(const DistributedMatrix<scalar_t>&) = delete;
    DistributedMatrixWrapper
    (const DistributedMatrixWrapper<scalar_t>&&) = delete;
    DistributedMatrixWrapper(const DistributedMatrix<scalar_t>&&) = delete;
    DistributedMatrixWrapper<scalar_t>&
    operator=(const DistributedMatrixWrapper<scalar_t>&) = default;
    DistributedMatrixWrapper<scalar_t>&
    operator=(const DistributedMatrix<scalar_t>&) = delete;
    DistributedMatrixWrapper<scalar_t>&
    operator=(DistributedMatrixWrapper<scalar_t>&& A) = default;
    DistributedMatrixWrapper<scalar_t>&
    operator=(DistributedMatrix<scalar_t>&&) = delete;
  };

  template<typename scalar_t>
  std::unique_ptr<const DistributedMatrixWrapper<scalar_t>>
  ConstDistributedMatrixWrapperPtr
  (std::size_t m, std::size_t n, const DistributedMatrix<scalar_t>& D,
   std::size_t i, std::size_t j) {
    return std::unique_ptr<const DistributedMatrixWrapper<scalar_t>>
      (new DistributedMatrixWrapper<scalar_t>
       (m, n, const_cast<DistributedMatrix<scalar_t>&>(D), i, j));
  }

  template<typename scalar_t>
  DistributedMatrixWrapper<scalar_t>::DistributedMatrixWrapper
  (std::size_t m, std::size_t n, DistributedMatrix<scalar_t>& A,
   std::size_t i, std::size_t j) : _rows(m), _cols(n), _i(i), _j(j) {
    assert(!A.active() || m+i <= std::size_t(A.rows()));
    assert(!A.active() || n+j <= std::size_t(A.cols()));
    assert(m >= 0 && n >= 0 && i >=0 && j >= 0);
    this->_data = A.data();
    std::copy(A.desc(), A.desc()+9, this->_desc);
    this->_lrows = A.lrows();   this->_lcols = A.lcols();
    this->_prows = A.prows();   this->_pcols = A.pcols();
    this->_prow = A.prow();     this->_pcol = A.pcol();
  }

  template<typename scalar_t>
  DistributedMatrixWrapper<scalar_t>::DistributedMatrixWrapper
  (int ctxt, std::size_t m, std::size_t n, scalar_t* A)
    : DistributedMatrixWrapper<scalar_t>
    (ctxt, m, n, DistributedMatrix<scalar_t>::default_MB,
     DistributedMatrix<scalar_t>::default_NB, A) {}

  template<typename scalar_t>
  DistributedMatrixWrapper<scalar_t>::DistributedMatrixWrapper
  (int ctxt, std::size_t m, std::size_t n, int MB, int NB, scalar_t* A)
    : _rows(m), _cols(n), _i(0), _j(0) {
    scalapack::Cblacs_gridinfo
      (ctxt, &this->_prows, &this->_pcols,
       &this->_prow, &this->_pcol);
    if (this->active()) {
      this->_data = A;
      if (scalapack::descinit
          (this->_desc, _rows, _cols, MB, NB,
           0, 0, ctxt, std::max(_rows, 1))) {
        std::cerr << "ERROR: Could not create DistributedMatrixWrapper"
                  << " descriptor!" << std::endl;
        abort();
      }
      this->_lrows = scalapack::numroc
        (this->_desc[2], this->_desc[4],
         this->_prow, this->_desc[6], this->_prows);
      this->_lcols = scalapack::numroc
        (this->_desc[3], this->_desc[5], this->_pcol,
         this->_desc[7], this->_pcols);
    } else {
      this->_data = nullptr;
      scalapack::descset(this->_desc, _rows, _cols, MB, NB, 0, 0, ctxt, 1);
      this->_lrows = this->_lcols = 0;
    }
  }

  template<typename scalar_t>
  DistributedMatrixWrapper<scalar_t>::DistributedMatrixWrapper
  (int ctxt, int rsrc, int csrc, std::size_t m,
   std::size_t n, DenseMatrix<scalar_t>& A)
    : _rows(m), _cols(n), _i(0), _j(0) {
    int MB = std::max(1, _rows);
    int NB = std::max(1, _cols);
    scalapack::Cblacs_gridinfo
      (ctxt, &this->_prows, &this->_pcols,
       &this->_prow, &this->_pcol);
    if (this->_prow == rsrc && this->_pcol == csrc) {
      this->_lrows = _rows;
      this->_lcols = _cols;
      this->_data = A.data();
      if (scalapack::descinit
          (this->_desc, _rows, _cols, MB, NB,
           rsrc, csrc, ctxt, std::max(_rows, 1))) {
        std::cerr << "ERROR: Could not create DistributedMatrixWrapper"
                  << " descriptor!" << std::endl;
        abort();
      }
    } else {
      this->_lrows = this->_lcols = 0;
      this->_data = nullptr;
      scalapack::descset
        (this->_desc, _rows, _cols, MB, NB,
         rsrc, csrc, ctxt, 1);
    }
  }

  template<typename scalar_t> void DistributedMatrixWrapper<scalar_t>::lranges
  (int& rlo, int& rhi, int& clo, int& chi) const {
    scalapack::infog2l
      (I(), J(), this->desc(), this->prows(), this->pcols(),
       this->prow(), this->pcol(), rlo, clo);
    scalapack::infog2l
      (I()+this->rows(), J()+this->cols(), this->desc(),
       this->prows(), this->pcols(),
       this->prow(), this->pcol(), rhi, chi);
    rlo--; rhi--; clo--; chi--;
  }

  template<typename scalar_t> DistributedMatrix<scalar_t>::DistributedMatrix()
    : DistributedMatrix(-1, 0, 0, default_MB, default_NB) {
    // make sure active() returns false if not constructed properly
    _prow = _pcol = _prows = _pcols = -1;
  }

  template<typename scalar_t> DistributedMatrix<scalar_t>::DistributedMatrix
  (int ctxt, const DenseMatrix<scalar_t>& m)
    : DistributedMatrix(ctxt, m.rows(), m.cols(), default_MB, default_NB) {
    //    assert(_prows == 1 && _pcols == 1);
    // TODO just do a copy instead of a gemr2d, or steal the data if
    // we also provide a move constructor!!
    scatter(m);
  }

  template<typename scalar_t> DistributedMatrix<scalar_t>::DistributedMatrix
  (int ctxt, int M, int N, const DistributedMatrix<scalar_t>& m, int ctxt_all)
    : DistributedMatrix(ctxt, M, N, default_MB, default_NB) {
    strumpack::copy(M, N, m, 0, 0, *this, 0, 0, ctxt_all);
  }

  template<typename scalar_t> DistributedMatrix<scalar_t>::DistributedMatrix
  (const DistributedMatrix<scalar_t>& m)
    : _lrows(m._lrows), _lcols(m._lcols), _prows(m._prows), _pcols(m._pcols),
      _prow(m._prow), _pcol(m._pcol) {
    std::copy(m._desc, m._desc+9, _desc);
    delete[] _data;
    _data = new scalar_t[_lrows*_lcols];
    std::copy(m._data, m._data+_lrows*_lcols, _data);
  }

  template<typename scalar_t> DistributedMatrix<scalar_t>::DistributedMatrix
  (DistributedMatrix<scalar_t>&& m)
    : _lrows(m._lrows), _lcols(m._lcols), _prows(m._prows), _pcols(m._pcols),
      _prow(m._prow), _pcol(m._pcol) {
    std::copy(m._desc, m._desc+9, _desc);
    delete[] _data;
    _data = m._data;
    m._data = nullptr;
  }

  template<typename scalar_t>
  DistributedMatrix<scalar_t>::DistributedMatrix(int ctxt, int M, int N)
    : DistributedMatrix(ctxt, M, N, default_MB, default_NB) {
  }

  template<typename scalar_t> DistributedMatrix<scalar_t>::DistributedMatrix
  (int ctxt, int M, int N, int MB, int NB) {
    assert(M >= 0 && N >= 0 && MB >= 0 && NB >= 0);
    MB = std::max(1, MB);
    NB = std::max(1, NB);
    scalapack::Cblacs_gridinfo(ctxt, &_prows, &_pcols, &_prow, &_pcol);
    if (_prow == -1 || _pcol == -1) {
      _lrows = _lcols = 0;
      _data = nullptr;
      scalapack::descset(_desc, M, N, MB, NB, 0, 0, ctxt, std::max(_lrows,1));
    } else {
      _lrows = scalapack::numroc(M, MB, _prow, 0, _prows);
      _lcols = scalapack::numroc(N, NB, _pcol, 0, _pcols);
      _data = new scalar_t[_lrows*_lcols];
      if (scalapack::descinit
          (_desc, M, N, MB, NB, 0, 0,
           ctxt, std::max(_lrows,1))) {
        std::cerr << "ERROR: Could not create DistributedMatrix descriptor!"
                  << std::endl;
        abort();
      }
    }
  }

  template<typename scalar_t>
  DistributedMatrix<scalar_t>::DistributedMatrix(int desc[9]) {
    std::copy(desc, desc+9, _desc);
    scalapack::Cblacs_gridinfo(_desc[1], &_prows, &_pcols, &_prow, &_pcol);
    if (_prow == -1 || _pcol == -1) {
      _lrows = _lcols = 0;
      _data = nullptr;
    } else {
      _lrows = scalapack::numroc(_desc[2], _desc[4], _prow, _desc[6], _prows);
      _lcols = scalapack::numroc(_desc[3], _desc[5], _pcol, _desc[7], _pcols);
      assert(_lrows==_desc[8]);
      if (_lrows && _lcols) _data = new scalar_t[_lrows*_lcols];
      else _data = nullptr;
    }
  }

  template<typename scalar_t>
  DistributedMatrix<scalar_t>::~DistributedMatrix() {
    clear();
  }

  template<typename scalar_t> DistributedMatrix<scalar_t>&
  DistributedMatrix<scalar_t>::operator=
  (const DistributedMatrix<scalar_t>& m) {
    _prows = m._prows;  _pcols = m._pcols;
    _prow = m._prow;    _pcol = m._pcol;
    _lrows = m._lrows;  _lcols = m._lcols;
    std::copy(m._desc, m._desc+9, _desc);
    delete[] _data;
    _data = new scalar_t[_lrows*_lcols];
    std::copy(m._data, m._data+_lrows*_lcols, _data);
    return *this;
  }

  template<typename scalar_t> DistributedMatrix<scalar_t>&
  DistributedMatrix<scalar_t>::operator=(DistributedMatrix<scalar_t>&& m) {
    _prows = m._prows;  _pcols = m._pcols;
    _prow = m._prow;    _pcol = m._pcol;
    _lrows = m._lrows;  _lcols = m._lcols;
    std::copy(m._desc, m._desc+9, _desc);
    delete[] _data;
    _data = m._data;
    m._data = nullptr;
    return *this;
  }

  template<typename scalar_t> void DistributedMatrix<scalar_t>::lranges
  (int& rlo, int& rhi, int& clo, int& chi) const {
    rlo = clo = 0;
    rhi = lrows();
    chi = lcols();
  }

  template<typename scalar_t> void DistributedMatrix<scalar_t>::clear() {
    delete[] _data;
    _data = nullptr;
    _prow = _pcol = _prows = _pcols = -1;
    _lrows = _lcols = 0;
    scalapack::descset(_desc, 0, 0, MB(), NB(), 0, 0, ctxt(), 1);
  }

  template<typename scalar_t> void DistributedMatrix<scalar_t>::resize
  (std::size_t m, std::size_t n) {
    DistributedMatrix<scalar_t> tmp(ctxt(), m, n, MB(), NB());
    for (int c=0; c<std::min(lcols(), tmp.lcols()); c++)
      for (int r=0; r<std::min(lrows(), tmp.lrows()); r++)
        tmp(r, c) = operator()(r, c);
    *this = std::move(tmp);
  }

  template<typename scalar_t> void DistributedMatrix<scalar_t>::hconcat
  (const DistributedMatrix<scalar_t>& b) {
    assert(rows() == b.rows());
    assert(ctxt() == b.ctxt());
    auto my_cols = cols();
    resize(rows(), my_cols+b.cols());
    if (!active()) return;
    strumpack::copy(rows(), b.cols(), b, 0, 0, *this, 0, my_cols, ctxt());
  }

  template<typename scalar_t> void DistributedMatrix<scalar_t>::zero() {
    if (!active()) return;
    int rlo, rhi, clo, chi;
    lranges(rlo, rhi, clo, chi);
    for (int c=clo; c<chi; c++)
      for (int r=rlo; r<rhi; r++)
        operator()(r,c) = scalar_t(0.);
  }

  template<typename scalar_t> void
  DistributedMatrix<scalar_t>::fill(scalar_t a) {
    if (!active()) return;
    int rlo, rhi, clo, chi;
    lranges(rlo, rhi, clo, chi);
    for (int c=clo; c<chi; c++)
      for (int r=rlo; r<rhi; r++)
        operator()(r,c) = a;
  }

  template<typename scalar_t> void DistributedMatrix<scalar_t>::random() {
    if (!active()) return;
    TIMER_TIME(TaskType::RANDOM_GENERATE, 1, t_gen);
    auto rgen = random::make_default_random_generator<real_t>();
    rgen->seed(_prow, _pcol);
    int rlo, rhi, clo, chi;
    lranges(rlo, rhi, clo, chi);
    for (int c=clo; c<chi; ++c)
      for (int r=rlo; r<rhi; ++r)
        operator()(r,c) = rgen->get();
  }

  template<typename scalar_t> void DistributedMatrix<scalar_t>::random
  (random::RandomGeneratorBase<typename RealType<scalar_t>::
   value_type>& rgen) {
    if (!active()) return;
    TIMER_TIME(TaskType::RANDOM_GENERATE, 1, t_gen);
    int rlo, rhi, clo, chi;
    lranges(rlo, rhi, clo, chi);
    for (int c=clo; c<chi; ++c)
      for (int r=rlo; r<rhi; ++r)
        operator()(r,c) = rgen.get();
  }

  template<typename scalar_t> void DistributedMatrix<scalar_t>::eye() {
    if (!active()) return;
    int rlo, rhi, clo, chi;
    lranges(rlo, rhi, clo, chi);
    // TODO set to zero, then a single loop for the diagonal
    for (int c=clo; c<chi; ++c)
      for (int r=rlo; r<rhi; ++r)
        operator()(r,c) = (rowl2g(r)-I()+1 == coll2g(c)-J()+1) ?
          scalar_t(1.) : scalar_t(0.);
  }

  /** correct value only on the procs in the ctxt */
  template<typename scalar_t> scalar_t
  DistributedMatrix<scalar_t>::all_global(int r, int c) const {
    if (!active()) return scalar_t(0.);
    scalar_t v;
    if (is_local(r, c)) {
      v = operator()(rowg2l(r), colg2l(c));
      scalapack::gebs2d(ctxt(), 'A', ' ', 1, 1, &v, 1);
    } else
      scalapack::gebr2d(ctxt(), 'A', ' ', 1, 1, &v, 1, rowg2p(r), colg2p(c));
    return v;
  }

  template<typename scalar_t> void
  DistributedMatrix<scalar_t>::print(std::string name, int precision) const {
    if (!active()) return;
    auto tmp = gather();
    if (is_master()) tmp.print(name);
  }

  template<typename scalar_t> void
  DistributedMatrix<scalar_t>::print_to_file
  (std::string name, std::string filename, int width) const {
    if (!active()) return;
    auto tmp = gather();
    if (is_master()) tmp.print_to_file(name, filename, width);
  }

  template<typename scalar_t> DistributedMatrix<scalar_t>
  DistributedMatrix<scalar_t>::transpose() const {
    DistributedMatrix<scalar_t> tmp(ctxt(), cols(), rows());
    if (!active()) return tmp;
    scalapack::ptranc
      (cols(), rows(), scalar_t(1.), data(), I(), J(),
      desc(), scalar_t(0.), tmp.data(),
      tmp.I(), tmp.J(), tmp.desc());
    return tmp;
  }

  template<typename scalar_t> void
  DistributedMatrix<scalar_t>::laswp(const std::vector<int>& P, bool fwd) {
    if (!active()) return;
    int descip[9];
    scalapack::descset
      (descip, rows() + MB()*prows(), 1, MB(), 1, 0, pcol(),
       ctxt(), MB() + scalapack::numroc
       (rows(), MB(), prow(), 0, prows()));
    scalapack::plapiv
      (fwd ? 'F' : 'B', 'R', 'C', rows(), cols(), data(), I(), J(), desc(),
       P.data(), 1, 1, descip, NULL);
  }

  template<typename scalar_t> DistributedMatrix<scalar_t>
  DistributedMatrix<scalar_t>::extract_rows
  (const std::vector<std::size_t>& Ir, MPI_Comm comm) const {
    TIMER_TIME(TaskType::DISTMAT_EXTRACT_ROWS, 1, t_dist_mat_extract_rows);
    DistributedMatrix<scalar_t> tmp(ctxt(), Ir.size(), cols());
    if (!active()) return tmp;
    std::vector<std::vector<scalar_t>> sbuf(_prows);
    std::vector<std::vector<scalar_t>> rbuf(_prows);
    {
      std::vector<std::size_t> rsizes(_prows);
      std::vector<std::size_t> ssizes(_prows);
      for (std::size_t r=0; r<Ir.size(); r++) {
        auto gr = Ir[r];
        auto owner = rowg2p(gr);
        if (owner != _prow) rsizes[owner] += _lcols;
        else {
          auto dest = rowg2p(r);
          if (dest == _prow) // just copy to tmp
            for (int c=0, tmpr=tmp.rowg2l(r), lr=rowg2l(gr);
                 c<_lcols; c++)
              tmp(tmpr, c) = operator()(lr, c);
          else ssizes[dest] += _lcols;
        }
      }
      for (int p=0; p<_prows; p++) {
        rbuf[p].resize(rsizes[p]);
        sbuf[p].reserve(ssizes[p]);
      }
    }
    for (std::size_t r=0; r<Ir.size(); r++) {
      auto gr = Ir[r];
      auto owner = rowg2p(gr);
      if (owner == _prow) {
        auto lr = rowg2l(gr);
        auto dest = rowg2p(r);
        if (dest != _prow) {
          for (int c=0; c<_lcols; c++)
            sbuf[dest].push_back(operator()(lr, c));
        }
      }
    }
    auto sreq = new MPI_Request[2*(_prows-1)];
    auto rreq = sreq + _prows-1;
    for (int p=0; p<_prows; p++)
      if (p != _prow) {
        MPI_Isend(sbuf[p].data(), sbuf[p].size(), mpi_type<scalar_t>(),
                  p+_pcol*_prows, 0, comm, (p < _prow) ? sreq+p : sreq+p-1);
        MPI_Irecv(rbuf[p].data(), rbuf[p].size(), mpi_type<scalar_t>(),
                  p+_pcol*_prows, 0, comm, (p < _prow) ? rreq+p : rreq+p-1);
      }
    MPI_Waitall(_prows-1, rreq, MPI_STATUSES_IGNORE);
    std::vector<scalar_t*> prbuf(_prows);
    for (int p=0; p<_prows; p++) prbuf[p] = rbuf[p].data();
    for (std::size_t r=0; r<Ir.size(); r++) {
      auto gr = Ir[r];
      auto owner = rowg2p(gr);
      if (owner == _prow) continue;
      auto dest = rowg2p(r);
      if (dest != _prow) continue;
      auto tmpr = tmp.rowg2l(r);
      for (int c=0; c<_lcols; c++)
        tmp(tmpr, c) = *(prbuf[owner]++);
    }
    MPI_Waitall(_prows-1, sreq, MPI_STATUSES_IGNORE);
    delete[] sreq;
    return tmp;
  }

  template<typename scalar_t> DistributedMatrix<scalar_t>
  DistributedMatrix<scalar_t>::extract_cols
  (const std::vector<std::size_t>& Jc, MPI_Comm comm) const {
    TIMER_TIME(TaskType::DISTMAT_EXTRACT_COLS, 1, t_dist_mat_extract_cols);
    DistributedMatrix<scalar_t> tmp(ctxt(), rows(), Jc.size());
    if (!active()) return tmp;
    assert(I() == 1 && J() == 1);
    std::vector<std::vector<scalar_t>> sbuf(_pcols);
    std::vector<std::vector<scalar_t>> rbuf(_pcols);
    {
      std::vector<std::size_t> rsizes(_pcols);
      std::vector<std::size_t> ssizes(_pcols);
      for (std::size_t c=0; c<Jc.size(); c++) {
        auto gc = Jc[c];
        auto owner = colg2p(gc);
        if (owner != _pcol) rsizes[owner] += _lrows;
        else {
          auto lc = colg2l(gc);
          auto dest = colg2p(c);
          if (dest == _pcol) { // just copy to tmp
            auto tmpc = tmp.colg2l(c);
            for (int r=0; r<_lrows; r++)
              tmp(r, tmpc) = operator()(r, lc);
          } else ssizes[dest] += _lrows;
        }
      }
      for (int p=0; p<_pcols; p++) {
        rbuf[p].resize(rsizes[p]);
        sbuf[p].reserve(ssizes[p]);
      }
    }
    for (std::size_t c=0; c<Jc.size(); c++) {
      auto gc = Jc[c];
      auto owner = colg2p(gc);
      if (owner == _pcol) {
        auto dest = colg2p(c);
        if (dest != _pcol)
          for (int r=0, lc=colg2l(gc); r<_lrows; r++)
            sbuf[dest].push_back(operator()(r, lc));
      }
    }
    auto sreq = new MPI_Request[2*(_pcols-1)];
    auto rreq = sreq + _pcols-1;
    for (int p=0; p<_pcols; p++)
      if (p != _pcol) {
        MPI_Isend(sbuf[p].data(), sbuf[p].size(), mpi_type<scalar_t>(),
                  _prow+p*_prows, 0, comm, (p < _pcol) ? sreq+p : sreq+p-1);
        MPI_Irecv(rbuf[p].data(), rbuf[p].size(), mpi_type<scalar_t>(),
                  _prow+p*_prows, 0, comm, (p < _pcol) ? rreq+p : rreq+p-1);
      }
    MPI_Waitall(_pcols-1, rreq, MPI_STATUSES_IGNORE);
    std::vector<scalar_t*> prbuf(_pcols);
    for (int p=0; p<_pcols; p++) prbuf[p] = rbuf[p].data();
    for (std::size_t c=0; c<Jc.size(); c++) {
      auto gc = Jc[c];
      auto owner = colg2p(gc);
      if (owner == _pcol) continue;
      auto dest = colg2p(c);
      if (dest != _pcol) continue;
      auto tmpc = tmp.colg2l(c);
      for (int r=0; r<_lrows; r++)
        tmp(r, tmpc) = *(prbuf[owner]++);
    }
    MPI_Waitall(_pcols-1, sreq, MPI_STATUSES_IGNORE);
    delete[] sreq;
    return tmp;
  }

  // TODO optimize
  template<typename scalar_t> DistributedMatrix<scalar_t>
  DistributedMatrix<scalar_t>::extract
  (const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J) const {
    TIMER_TIME(TaskType::DISTMAT_EXTRACT, 1, t_dist_mat_extract);
    // TODO optimize this!??
    DistributedMatrix<scalar_t> B(ctxt(), I.size(), J.size());
    auto tmp = gather();
    DenseMatrix<scalar_t> sub;
    if (is_master()) sub = tmp.extract(I, J);
    B.scatter(sub);
    return B;
  }

  template<typename scalar_t> DistributedMatrix<scalar_t>&
  DistributedMatrix<scalar_t>::add
  (const DistributedMatrix<scalar_t>& B) {
    if (!active()) return *this;
    int rlo, rhi, clo, chi, Brlo, Brhi, Bclo, Bchi;
    lranges(rlo, rhi, clo, chi);
    B.lranges(Brlo, Brhi, Bclo, Bchi);
    int lc = chi - clo;
    int lr = rhi - rlo;
    //#pragma omp taskloop grainsize(64) //collapse(2)
    for (int c=0; c<lc; ++c)
      for (int r=0; r<lr; ++r)
        operator()(r+rlo,c+clo) += B(r+Brlo,c+Bclo);
    return *this;
  }

  template<typename scalar_t> DistributedMatrix<scalar_t>&
  DistributedMatrix<scalar_t>::scaled_add
  (scalar_t alpha, const DistributedMatrix<scalar_t>& B) {
    if (!active()) return *this;
    assert(ctxt() == B.ctxt());
    // TODO assert that the layout is the same??
    // I()==B.I() etc?? ctxt()==B.ctxt()?? MB, NB, rows, cols?
    int rlo, rhi, clo, chi, Brlo, Brhi, Bclo, Bchi;
    lranges(rlo, rhi, clo, chi);
    B.lranges(Brlo, Brhi, Bclo, Bchi);
    int lc = chi - clo;
    int lr = rhi - rlo;
    //#pragma omp taskloop grainsize(64) //collapse(2)
    for (int c=0; c<lc; ++c)
      for (int r=0; r<lr; ++r)
        operator()(r+rlo,c+clo) += alpha * B(r+Brlo,c+Bclo);
    return *this;
  }

  template<typename scalar_t> typename RealType<scalar_t>::value_type
  DistributedMatrix<scalar_t>::norm() const {
    return normF();
  }

  template<typename scalar_t> typename RealType<scalar_t>::value_type
  DistributedMatrix<scalar_t>::norm1() const {
    if (!active()) return real_t(-1.);
    int IACOL = indxg2p(J(), NB(), pcol(), 0, pcols());
    int Nq0 = scalapack::numroc
      (cols()+ ((J()-1)%NB()), NB(), pcol(), IACOL, pcols());
    real_t* work = new real_t[Nq0];
    auto norm = scalapack::plange
      ('1', rows(), cols(), data(), I(), J(), desc(), work);
    delete[] work;
    return norm;
  }

  template<typename scalar_t> typename RealType<scalar_t>::value_type
  DistributedMatrix<scalar_t>::normI() const {
    if (!active()) return real_t(-1.);
    int IAROW = indxg2p(I(), MB(), prow(), 0, prows());
    int Mp0 = scalapack::numroc
      (rows()+ ((I()-1)%MB()), MB(), prow(), IAROW, prows());
    real_t* work = new real_t[Mp0];
    auto norm = scalapack::plange
      ('I', rows(), cols(), data(), I(), J(), desc(), work);
    delete[] work;
    return norm;
  }

  template<typename scalar_t> typename RealType<scalar_t>::value_type
  DistributedMatrix<scalar_t>::normF() const {
    if (!active()) return real_t(-1.);
    real_t* work = nullptr;
    return scalapack::plange
      ('F', rows(), cols(), data(), I(), J(), desc(), work);
  }

  template<typename scalar_t> void
  DistributedMatrix<scalar_t>::scatter(const DenseMatrix<scalar_t>& a) {
    if (!active()) return;
    int a_desc[9];
    scalapack::descset
      (a_desc, rows(), cols(), rows(), cols(), 0, 0, ctxt(),
       std::max(std::size_t(rows()), a.ld()));
    scalapack::pgemr2d
      (rows(), cols(), a.data(), 1, 1,
       a_desc, data(), I(), J(), desc(), ctxt());
  }

  /** gather to proc 0,0 in ctxt */
  template<typename scalar_t> DenseMatrix<scalar_t>
  DistributedMatrix<scalar_t>::gather() const {
    DenseMatrix<scalar_t> a;
    if (!active()) return a;
    if (is_master()) a = DenseMatrix<scalar_t>(rows(), cols());
    int a_desc[9];
    scalapack::descset
      (a_desc, rows(), cols(), rows(), cols(),
       0, 0, ctxt(), rows());
    scalapack::pgemr2d
      (rows(), cols(), data(), I(), J(), desc(),
       a.data(), 1, 1, a_desc, ctxt());
    return a;
  }

  /** gather to all process in ctxt_all */
  template<typename scalar_t> DenseMatrix<scalar_t>
  DistributedMatrix<scalar_t>::all_gather(int ctxt_all) const {
    DenseMatrix<scalar_t> a(rows(), cols());
    int a_desc[9];
    scalapack::descset
      (a_desc, rows(), cols(), rows(), cols(),
       0, 0, ctxt_all, rows());
    scalapack::pgemr2d
      (rows(), cols(), data(), I(), J(), desc(),
       a.data(), 1, 1, a_desc, ctxt_all);
    int all_prows, all_pcols, all_prow, all_pcol;
    scalapack::Cblacs_gridinfo
      (ctxt_all, &all_prows, &all_pcols, &all_prow, &all_pcol);
    if (all_prow==0 && all_pcol==0)
      scalapack::gebs2d(ctxt_all, 'A', ' ', rows(), cols(), a.data(), a.ld());
    else scalapack::gebr2d
           (ctxt_all, 'A', ' ', rows(), cols(), a.data(), a.ld(), 0, 0);
    return a;
  }

  template<typename scalar_t> DenseMatrix<scalar_t>
  DistributedMatrix<scalar_t>::dense_and_clear() {
    DenseMatrix<scalar_t> tmp;
    tmp._data = data();
    tmp._rows = lrows();
    tmp._cols = lcols();
    tmp._ld = ld();
    this->_data = nullptr;
    clear();
    return tmp;
  }

  template<typename scalar_t> DenseMatrixWrapper<scalar_t>
  DistributedMatrix<scalar_t>::dense_wrapper() {
    return DenseMatrixWrapper<scalar_t>(lrows(), lcols(), data(), ld());
  }

  template<typename scalar_t> std::vector<int>
  DistributedMatrix<scalar_t>::LU() {
    if (!active()) return std::vector<int>();
    std::vector<int> ipiv(lrows()+MB());
    int info = scalapack::pgetrf
      (rows(), cols(), data(), I(), J(), desc(), ipiv.data());
    if (info) {
      std::cerr << "ERROR: LU factorization of DistributedMatrix failed"
                << " with info = " << info << std::endl;
      exit(1);
    }
    return ipiv;
  }

  // Solve a system of linear equations with B as right hand side.
  // assumption: the current matrix should have been factored using LU.
  template<typename scalar_t> DistributedMatrix<scalar_t>
  DistributedMatrix<scalar_t>::solve
  (const DistributedMatrix<scalar_t>& b, const std::vector<int>& piv) const {
    if (!active())
      return DistributedMatrix<scalar_t>(b.ctxt(), b.rows(), b.cols());
    DistributedMatrix<scalar_t> c(b);
    // TODO in place??, add assertions, check dimensions!!
    if (scalapack::pgetrs
        (char(Trans::N), c.rows(), c.cols(), data(),
         I(), J(), desc(), piv.data(),
         c.data(), c.I(), c.J(), c.desc())) {
      std::cerr << "# ERROR: Failure in PGETRS :(" << std::endl; abort();
    }
    return c;
  }

  template<typename scalar_t> void
  DistributedMatrix<scalar_t>::orthogonalize
  (scalar_t& r_max, scalar_t& r_min) {
    if (!active()) return;
    auto minmn = std::min(rows(), cols());
    auto N = J() + minmn - 1;
    auto ltau = scalapack::numroc(N, NB(), pcol(), 0, pcols());
    auto tau = new scalar_t[ltau];
    auto info = scalapack::pgeqrf
      (rows(), minmn, data(), I(), J(), desc(), tau);
    if (lrows() && lcols()) {
      real_t Rmax(std::numeric_limits<real_t>::min());
      real_t Rmin(std::numeric_limits<real_t>::max());
      if (fixed()) {
        for (int gi=0; gi<minmn; gi++) {
          if (is_local_fixed(gi, gi)) {
            auto Rii = std::abs(global_fixed(gi,gi));
            Rmax = std::max(Rmax, Rii);
            Rmin = std::min(Rmin, Rii);
          }
        }
      } else {
        for (int gi=0; gi<minmn; gi++) {
          if (is_local(gi, gi)) {
            auto Rii = std::abs(global(gi,gi));
            Rmax = std::max(Rmax, Rii);
            Rmin = std::min(Rmin, Rii);
          }
        }
      }
      r_max = Rmax;
      r_min = Rmin;
    }
    scalapack::gamx2d(ctxt(), 'A', ' ', 1, 1, &r_max, 1,
                      NULL, NULL, -1, -1, -1);
    scalapack::gamn2d(ctxt(), 'A', ' ', 1, 1, &r_min, 1,
                      NULL, NULL, -1, -1, -1);
    info = scalapack::pxxgqr
      (rows(), minmn, minmn, data(), I(), J(), desc(), tau);
    if (info) {
      std::cerr << "ERROR: Orthogonalization (pxxgqr) failed with info = "
                << info << std::endl;
      abort();
    }
    if (cols() > rows()) {
      DistributedMatrixWrapper<scalar_t>
        tmp(rows(), cols()-rows(), *this, 0, rows());
      tmp.zero();
    }
    delete[] tau;
  }

  template<typename scalar_t> void DistributedMatrix<scalar_t>::LQ
  (DistributedMatrix<scalar_t>& L, DistributedMatrix<scalar_t>& Q) const {
    if (!active()) return;
    assert(I()==1 && J()==1);
    DistributedMatrix<scalar_t> tmp(ctxt(), std::max(rows(), cols()), cols());
    // TODO this is not a pgemr2d, this does not require communication!!
    strumpack::copy(rows(), cols(), *this, 0, 0, tmp, 0, 0, ctxt());
    // TODO the last argument to numroc, should it be prows/pcols???
    auto tau = new scalar_t
      [scalapack::numroc(I()+std::min(rows(),cols())-1, MB(),
                         prow(), 0, prows())];
    scalapack::pgelqf
      (rows(), tmp.cols(), tmp.data(), tmp.I(), tmp.J(),
       tmp.desc(), tau);
    L = DistributedMatrix<scalar_t>(ctxt(), rows(), rows());
    // TODO this is not a pgemr2d, this does not require communication!!
    strumpack::copy(rows(), rows(), tmp, 0, 0, L, 0, 0, ctxt());
    // TODO check the diagonal elements
    // auto sfmin = blas::lamch<real_t>('S');
    // for (std::size_t i=0; i<std::min(rows(), cols()); i++)
    //   if (std::abs(L(i, i)) < sfmin) {
    //     std::cerr << "WARNING: small diagonal on L from LQ" << std::endl;
    //     break;
    //   }
    scalapack::pxxglq
      (cols(), cols(), std::min(rows(), cols()),
       tmp.data(), tmp.I(), tmp.J(), tmp.desc(), tau);
    delete[] tau;
    if (tmp.rows() == cols()) Q = std::move(tmp);
    else {
      Q = DistributedMatrix<scalar_t>(ctxt(), cols(), cols());
      // TODO this is not a pgemr2d, this does not require communication!!
      strumpack::copy(cols(), cols(), tmp, 0, 0, Q, 0, 0, ctxt());
    }
  }

  template<typename scalar_t> void
  DistributedMatrix<scalar_t>::ID_row
  (DistributedMatrix<scalar_t>& X, std::vector<int>& piv,
   std::vector<std::size_t>& ind, real_t rel_tol, real_t abs_tol,
   int ctxt_T) {
    // transpose the BLACS grid and do a local transpose, then call
    // ID_column, then do local transpose of output X_T to get back in
    // the original blacs grid
    if (!active()) return;
    TIMER_TIME(TaskType::HSS_PARHQRINTERPOL, 1, t_hss_par_hqr);
    assert(I()==1 && J()==1);
    DistributedMatrix<scalar_t> this_T(ctxt_T, cols(), rows());
    blas::omatcopy('T', lrows(), lcols(), data(), ld(),
                   this_T.data(), this_T.ld());
    DistributedMatrix<scalar_t> X_T;
    this_T.ID_column(X_T, piv, ind, rel_tol, abs_tol);
    X = DistributedMatrix<scalar_t>(ctxt(), X_T.cols(), X_T.rows());
    blas::omatcopy('T', X_T.lrows(), X_T.lcols(), X_T.data(), X_T.ld(),
                   X.data(), X.ld());
  }

  template<typename scalar_t> void
  DistributedMatrix<scalar_t>::ID_column
  (DistributedMatrix<scalar_t>& X, std::vector<int>& piv,
   std::vector<std::size_t>& ind, real_t rel_tol, real_t abs_tol) {
    if (!active()) return;
    // _J: indices of permuted colums (int iso size_t -> ind)
    std::vector<int> _J(cols());
    std::iota(_J.begin(), _J.end(), 1);
    std::vector<int> gpiv(cols()); // gpiv: column permutation
    std::iota(gpiv.begin(), gpiv.end(), 1);
    int rank = 0;
    // Step 1: RRQR
    // TODO also use abs_tol!!
    scalapack::pgeqpfmod
      (rows(), cols(), data(), I(), J(), desc(),
       _J.data(), gpiv.data(), &rank, rel_tol);
    piv.resize(lcols()+NB());
    ind.resize(rank);
    for (int c=0; c<lcols(); c++) piv[c] = gpiv[coll2g(c)];
    for (int c=0; c<rank; c++) ind[c] = _J[c]-1;
    // Step 2: TRSM and permutation:
    //   R1^-1 R = [I R1^-1 R2] = [I X] with R = [R1 R2], R1 r x r
    DistributedMatrixWrapper<scalar_t> R1(rank, rank, *this, 0, 0);
    X = DistributedMatrix<scalar_t>(ctxt(), rank, cols()-rank);
    copy(rank, cols()-rank, *this, 0, rank, X, 0, 0, ctxt());
    trsm(Side::L, UpLo::U, Trans::N, Diag::N, scalar_t(1.), R1, X);
  }

  template<typename scalar_t> void gemm
  (Trans ta, Trans tb, scalar_t alpha, const DistributedMatrix<scalar_t>& A,
   const DistributedMatrix<scalar_t>& B,
   scalar_t beta, DistributedMatrix<scalar_t>& C) {
    if (!A.active()) return;
    assert((ta==Trans::N && A.rows()==C.rows()) ||
           (ta!=Trans::N && A.cols()==C.rows()));
    assert((tb==Trans::N && B.cols()==C.cols()) ||
           (tb!=Trans::N && B.rows()==C.cols()));
    assert((ta==Trans::N && tb==Trans::N && A.cols()==B.rows()) ||
           (ta!=Trans::N && tb==Trans::N && A.rows()==B.rows()) ||
           (ta==Trans::N && tb!=Trans::N && A.cols()==B.cols()) ||
           (ta!=Trans::N && tb!=Trans::N && A.rows()==B.cols()));
    assert(A.I()>=1 && A.J()>=1 && B.I()>=1 &&
           B.J()>=1 && C.I()>=1 && C.J()>=1);
    assert(A.ctxt()==B.ctxt() && A.ctxt()==C.ctxt());
    scalapack::pgemm
      (char(ta), char(tb), C.rows(), C.cols(),
       (ta==Trans::N) ? A.cols() : A.rows(), alpha,
       A.data(), A.I(), A.J(), A.desc(),
       B.data(), B.I(), B.J(), B.desc(),
       beta, C.data(), C.I(), C.J(), C.desc());
  }

  template<typename scalar_t> void trsm
  (Side s, UpLo u, Trans ta, Diag d, scalar_t alpha,
   const DistributedMatrix<scalar_t>& A, DistributedMatrix<scalar_t>& B) {
    if (!A.active()) return;
    assert(A.rows()==A.cols());
    assert(s!=Side::L || ta!=Trans::N || A.cols()==B.rows());
    assert(s!=Side::L || ta==Trans::N || A.rows()==B.rows());
    assert(s!=Side::R || ta!=Trans::N || A.rows()==B.cols());
    assert(s!=Side::R || ta==Trans::N || A.cols()==B.cols());
    scalapack::ptrsm
      (char(s), char(u), char(ta), char(d), B.rows(), B.cols(),
       alpha, A.data(), A.I(), A.J(), A.desc(),
       B.data(), B.I(), B.J(), B.desc());
  }

  template<typename scalar_t> void trsv
  (UpLo ul, Trans ta, Diag d, const DistributedMatrix<scalar_t>& A,
   DistributedMatrix<scalar_t>& B) {
    if (!A.active()) return;
    assert(B.cols() == 1 && A.rows() == A.cols() && A.cols() == A.rows());
    scalapack::ptrsv
      (char(ul), char(ta), char(d), A.rows(),
       A.data(), A.I(), A.J(), A.desc(),
       B.data(), B.I(), B.J(), B.desc(), 1);
    // TODO also support row vectors by passing different incb?
  }

  template<typename scalar_t> void gemv
  (Trans ta, scalar_t alpha, const DistributedMatrix<scalar_t>& A,
   const DistributedMatrix<scalar_t>& X, scalar_t beta,
   DistributedMatrix<scalar_t>& Y) {
    if (!A.active()) return;
    assert(X.cols() == 1 && Y.cols() == 1);
    assert(ta != Trans::N || (A.rows() == Y.rows() && A.cols() == X.rows()));
    assert(ta == Trans::N || (A.cols() == Y.rows() && A.rows() == X.rows()));
    scalapack::pgemv
      (char(ta), A.rows(), A.cols(), alpha,
       A.data(), A.I(), A.J(), A.desc(),
       X.data(), X.I(), X.J(), X.desc(), 1,
       beta, Y.data(), Y.I(), Y.J(), Y.desc(), 1);
    // TODO also support row vectors by passing different incb?
  }

  template<typename scalar_t> DistributedMatrix<scalar_t> vconcat
  (int cols, int arows, int brows, const DistributedMatrix<scalar_t>& a,
   const DistributedMatrix<scalar_t>& b, int ctxt_new, int ctxt_all) {
    DistributedMatrix<scalar_t> tmp(ctxt_new, arows+brows, cols);
    copy(arows, cols, a, 0, 0, tmp, 0, 0, ctxt_all);
    copy(brows, cols, b, 0, 0, tmp, arows, 0, ctxt_all);
    return tmp;
  }

} // end namespace strumpack

#endif // DISTRIBUTED_MATRIX_HPP
