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
 */
#ifndef DISTRIBUTED_MATRIX_HPP
#define DISTRIBUTED_MATRIX_HPP

#include <vector>
#include <string>
#include <cmath>
#include <functional>

#include "misc/MPIWrapper.hpp"
#include "misc/RandomWrapper.hpp"
#include "DenseMatrix.hpp"
#include "BLACSGrid.hpp"

namespace strumpack {

  inline int indxl2g(int INDXLOC, int NB, int IPROC, int ISRCPROC, int NPROCS)
  { return NPROCS*NB*((INDXLOC-1)/NB) + (INDXLOC-1) % NB +
      ((NPROCS+IPROC-ISRCPROC) % NPROCS)*NB + 1; }
  inline int indxg2l(int INDXGLOB, int NB, int IPROC, int ISRCPROC, int NPROCS)
  { return NB*((INDXGLOB-1)/(NB*NPROCS)) + (INDXGLOB-1) % NB + 1; }
  inline int indxg2p(int INDXGLOB, int NB, int IPROC, int ISRCPROC, int NPROCS)
  { return ( ISRCPROC + (INDXGLOB - 1) / NB ) % NPROCS; }


  template<typename scalar_t> class DistributedMatrix {
    using real_t = typename RealType<scalar_t>::value_type;

  protected:
    const BLACSGrid* grid_ = nullptr;
    scalar_t* data_ = nullptr;
    int lrows_;
    int lcols_;
    int desc_[9];

  public:
    DistributedMatrix();
    DistributedMatrix(const BLACSGrid* g, int M, int N);
    DistributedMatrix(const BLACSGrid* g, int M, int N,
                      const std::function<scalar_t(std::size_t,
                                                   std::size_t)>& A);
    DistributedMatrix(const BLACSGrid* g, const DenseMatrix<scalar_t>& m);
    DistributedMatrix(const BLACSGrid* g, DenseMatrix<scalar_t>&& m);
    DistributedMatrix(const BLACSGrid* g, DenseMatrixWrapper<scalar_t>&& m);
    DistributedMatrix(const BLACSGrid* g, int M, int N,
                      const DistributedMatrix<scalar_t>& m,
                      int context_all);
    DistributedMatrix(const BLACSGrid* g, int M, int N, int MB, int NB);
    DistributedMatrix(const BLACSGrid* g, int desc[9]);

    DistributedMatrix(const DistributedMatrix<scalar_t>& m);
    DistributedMatrix(DistributedMatrix<scalar_t>&& m);
    virtual ~DistributedMatrix();

    DistributedMatrix<scalar_t>&
    operator=(const DistributedMatrix<scalar_t>& m);
    DistributedMatrix<scalar_t>&
    operator=(DistributedMatrix<scalar_t>&& m);


    const int* desc() const { return desc_; }
    int* desc() { return desc_; }
    bool active() const { return grid() && grid()->active(); }

    const BLACSGrid* grid() const { return grid_; }
    const MPIComm& Comm() const { return grid()->Comm(); }
    MPI_Comm comm() const { return Comm().comm(); }

    int ctxt() const { return grid() ? grid()->ctxt() : -1; }
    int ctxt_all() const { return grid() ? grid()->ctxt_all() : -1; }

    virtual int rows() const { return desc_[2]; }
    virtual int cols() const { return desc_[3]; }
    int lrows() const { return lrows_; }
    int lcols() const { return lcols_; }
    int ld() const { return lrows_; }
    int MB() const { return desc_[4]; }
    int NB() const { return desc_[5]; }
    int rowblocks() const { return std::ceil(float(lrows()) / MB()); }
    int colblocks() const { return std::ceil(float(lcols()) / NB()); }

    virtual int I() const { return 1; }
    virtual int J() const { return 1; }
    virtual void lranges(int& rlo, int& rhi, int& clo, int& chi) const;

    const scalar_t* data() const { return data_; }
    scalar_t* data() { return data_; }
    const scalar_t& operator()(int r, int c) const
    { return data_[r+ld()*c]; }
    scalar_t& operator()(int r, int c) { return data_[r+ld()*c]; }

    int prow() const { assert(grid()); return grid()->prow(); }
    int pcol() const { assert(grid()); return grid()->pcol(); }
    int nprows() const { assert(grid()); return grid()->nprows(); }
    int npcols() const { assert(grid()); return grid()->npcols(); }
    int npactives() const { assert(grid()); return grid()->npactives(); }

    bool is_master() const { return grid() && prow() == 0 && pcol() == 0; }
    int rowl2g(int row) const { assert(grid());
      return indxl2g(row+1, MB(), prow(), 0, nprows()) - I(); }
    int coll2g(int col) const { assert(grid());
      return indxl2g(col+1, NB(), pcol(), 0, npcols()) - J(); }
    int rowg2l(int row) const { assert(grid());
      return indxg2l(row+I(), MB(), prow(), 0, nprows()) - 1; }
    int colg2l(int col) const { assert(grid());
      return indxg2l(col+J(), NB(), pcol(), 0, npcols()) - 1; }
    int rowg2p(int row) const { assert(grid());
      return indxg2p(row+I(), MB(), prow(), 0, nprows()); }
    int colg2p(int col) const { assert(grid());
      return indxg2p(col+J(), NB(), pcol(), 0, npcols()); }
    int rank(int r, int c) const {
      return rowg2p(r) + colg2p(c) * nprows(); }
    bool is_local(int r, int c) const { assert(grid());
      return rowg2p(r) == prow() && colg2p(c) == pcol();
    }

    bool fixed() const { return MB()==default_MB && NB()==default_NB; }
    int rowl2g_fixed(int row) const {
      assert(grid() && fixed());
      return indxl2g(row+1, default_MB, prow(), 0, nprows()) - I(); }
    int coll2g_fixed(int col) const {
      assert(grid() && fixed());
      return indxl2g(col+1, default_NB, pcol(), 0, npcols()) - J(); }
    int rowg2l_fixed(int row) const {
      assert(grid() && fixed());
      return indxg2l(row+I(), default_MB, prow(), 0, nprows()) - 1; }
    int colg2l_fixed(int col) const {
      assert(grid() && fixed());
      return indxg2l(col+J(), default_NB, pcol(), 0, npcols()) - 1; }
    int rowg2p_fixed(int row) const {
      assert(grid() && fixed());
      return indxg2p(row+I(), default_MB, prow(), 0, nprows()); }
    int colg2p_fixed(int col) const {
      assert(grid() && fixed());
      return indxg2p(col+J(), default_NB, pcol(), 0, npcols()); }
    int rank_fixed(int r, int c) const {
      assert(grid() && fixed()); return rowg2p_fixed(r) + colg2p_fixed(c) * nprows(); }
    bool is_local_fixed(int r, int c) const {
      assert(grid() && fixed());
      return rowg2p_fixed(r) == prow() && colg2p_fixed(c) == pcol(); }

    // TODO fixed versions??
    const scalar_t& global(int r, int c) const
    { assert(is_local(r, c)); return operator()(rowg2l(r),colg2l(c)); }
    scalar_t& global(int r, int c)
    { assert(is_local(r, c)); return operator()(rowg2l(r),colg2l(c)); }
    scalar_t& global_fixed(int r, int c) {
      assert(is_local(r, c)); assert(fixed());
      return operator()(rowg2l_fixed(r),colg2l_fixed(c)); }
    void global(int r, int c, scalar_t v) {
      if (active() && is_local(r, c)) operator()(rowg2l(r),colg2l(c)) = v;  }
    scalar_t all_global(int r, int c) const;

    void print() const { print("A"); }
    void print(std::string name, int precision=15) const;
    void print_to_file(std::string name, std::string filename,
                       int width=8) const;
    void print_to_files(std::string name, int precision=16) const;
    void random();
    void random(random::RandomGeneratorBase<typename RealType<scalar_t>::
                value_type>& rgen);
    void zero();
    void fill(scalar_t a);
    void fill(const std::function<scalar_t(std::size_t,
                                           std::size_t)>& A);
    void eye();
    void shift(scalar_t sigma);
    void clear();
    virtual void resize(std::size_t m, std::size_t n);
    virtual void hconcat(const DistributedMatrix<scalar_t>& b);
    DistributedMatrix<scalar_t> transpose() const;

    void mult(Trans op, const DistributedMatrix<scalar_t>& X,
              DistributedMatrix<scalar_t>& Y) const;

    void laswp(const std::vector<int>& P, bool fwd);

    DistributedMatrix<scalar_t>
    extract_rows(const std::vector<std::size_t>& Ir) const;
    DistributedMatrix<scalar_t>
    extract_cols(const std::vector<std::size_t>& Ic) const;

    DistributedMatrix<scalar_t>
    extract(const std::vector<std::size_t>& I,
            const std::vector<std::size_t>& J) const;
    DistributedMatrix<scalar_t>& add(const DistributedMatrix<scalar_t>& B);
    DistributedMatrix<scalar_t>&
    scaled_add(scalar_t alpha, const DistributedMatrix<scalar_t>& B);
    DistributedMatrix<scalar_t>&
    scale_and_add(scalar_t alpha, const DistributedMatrix<scalar_t>& B);

    real_t norm() const;
    real_t normF() const;
    real_t norm1() const;
    real_t normI() const;

    virtual std::size_t memory() const
    { return sizeof(scalar_t)*std::size_t(lrows())*std::size_t(lcols()); }
    virtual std::size_t total_memory() const
    { return sizeof(scalar_t)*std::size_t(rows())*std::size_t(cols()); }
    virtual std::size_t nonzeros() const
    { return std::size_t(lrows())*std::size_t(lcols()); }
    virtual std::size_t total_nonzeros() const
    { return std::size_t(rows())*std::size_t(cols()); }

    void scatter(const DenseMatrix<scalar_t>& a);
    DenseMatrix<scalar_t> gather() const;
    DenseMatrix<scalar_t> all_gather() const;

    DenseMatrix<scalar_t> dense_and_clear();
    DenseMatrix<scalar_t> dense() const;
    DenseMatrixWrapper<scalar_t> dense_wrapper();

    std::vector<int> LU();
    int LU(std::vector<int>&);

    DistributedMatrix<scalar_t>
    solve(const DistributedMatrix<scalar_t>& b,
          const std::vector<int>& piv) const;

    void LQ(DistributedMatrix<scalar_t>& L,
            DistributedMatrix<scalar_t>& Q) const;

    void orthogonalize(scalar_t& r_max, scalar_t& r_min);

    void ID_column(DistributedMatrix<scalar_t>& X, std::vector<int>& piv,
                   std::vector<std::size_t>& ind,
                   real_t rel_tol, real_t abs_tol, int max_rank);
    void ID_row(DistributedMatrix<scalar_t>& X, std::vector<int>& piv,
                std::vector<std::size_t>& ind, real_t rel_tol, real_t abs_tol,
                int max_rank, const BLACSGrid* grid_T);

    static const int default_MB = STRUMPACK_PBLAS_BLOCKSIZE;
    static const int default_NB = STRUMPACK_PBLAS_BLOCKSIZE;
  };

  /**
   * copy submatrix of a DistM_t at ia,ja of size m,n into a DenseM_t
   * b at proc dest
   */
  template<typename scalar_t> void
  copy(std::size_t m, std::size_t n, const DistributedMatrix<scalar_t>& a,
       std::size_t ia, std::size_t ja, DenseMatrix<scalar_t>& b,
       int dest, int context_all);

  template<typename scalar_t> void
  copy(std::size_t m, std::size_t n, const DenseMatrix<scalar_t>& a, int src,
       DistributedMatrix<scalar_t>& b, std::size_t ib, std::size_t jb,
       int context_all);

  /** copy submatrix of a at ia,ja of size m,n into b at position ib,jb */
  template<typename scalar_t> void
  copy(std::size_t m, std::size_t n, const DistributedMatrix<scalar_t>& a,
       std::size_t ia, std::size_t ja, DistributedMatrix<scalar_t>& b,
       std::size_t ib, std::size_t jb, int context_all);

  /**
   * Wrapper class does exactly the same as a regular DistributedMatrix,
   * but it is initialized with existing memory, so it does not
   * allocate, own or delete the memory
   */
  template<typename scalar_t>
  class DistributedMatrixWrapper : public DistributedMatrix<scalar_t> {
  private:
    int _rows, _cols, _i, _j;
  public:
    DistributedMatrixWrapper() : DistributedMatrix<scalar_t>(),
      _rows(0), _cols(0), _i(0), _j(0) {}

    DistributedMatrixWrapper(DistributedMatrix<scalar_t>& A);
    DistributedMatrixWrapper(const DistributedMatrixWrapper<scalar_t>& A);
    DistributedMatrixWrapper(DistributedMatrixWrapper<scalar_t>&& A);
    DistributedMatrixWrapper(std::size_t m, std::size_t n,
                             DistributedMatrix<scalar_t>& A,
                             std::size_t i, std::size_t j);
    DistributedMatrixWrapper(const BLACSGrid* g, std::size_t m, std::size_t n,
                             scalar_t* A);
    DistributedMatrixWrapper(const BLACSGrid* g, std::size_t m, std::size_t n,
                             int MB, int NB, scalar_t* A);
    DistributedMatrixWrapper(const BLACSGrid* g, std::size_t m, std::size_t n,
                             DenseMatrix<scalar_t>& A);

    virtual ~DistributedMatrixWrapper() { this->data_ = nullptr; }

    DistributedMatrixWrapper<scalar_t>&
    operator=(const DistributedMatrixWrapper<scalar_t>& A);
    DistributedMatrixWrapper<scalar_t>&
    operator=(DistributedMatrixWrapper<scalar_t>&& A);

    int rows() const override { return _rows; }
    int cols() const override { return _cols; }
    int I() const override { return _i+1; }
    int J() const override { return _j+1; }
    void lranges(int& rlo, int& rhi, int& clo, int& chi) const override;

    void resize(std::size_t m, std::size_t n) override { assert(1); }
    void hconcat(const DistributedMatrix<scalar_t>& b) override { assert(1); }
    void clear()
    { this->data_ = nullptr; DistributedMatrix<scalar_t>::clear(); }
    std::size_t memory() const override { return 0; }
    std::size_t total_memory() const override { return 0; }
    std::size_t nonzeros() const override { return 0; }
    std::size_t total_nonzeros() const override { return 0; }

    DenseMatrix<scalar_t> dense_and_clear() = delete;
    DenseMatrixWrapper<scalar_t> dense_wrapper() = delete;
    DistributedMatrixWrapper<scalar_t>&
    operator=(const DistributedMatrix<scalar_t>&) = delete;
    DistributedMatrixWrapper<scalar_t>&
    operator=(DistributedMatrix<scalar_t>&&) = delete;
  };


  template<typename scalar_t> long long int
  LU_flops(const DistributedMatrix<scalar_t>& a) {
    if (!a.active()) return 0;
    return (is_complex<scalar_t>() ? 4:1) *
      blas::getrf_flops(a.rows(), a.cols()) /
      a.npactives();
  }

  template<typename scalar_t> long long int
  solve_flops(const DistributedMatrix<scalar_t>& b) {
    if (!b.active()) return 0;
    return (is_complex<scalar_t>() ? 4:1) *
      blas::getrs_flops(b.rows(), b.cols()) /
      b.npactives();
  }

  template<typename scalar_t> long long int
  LQ_flops(const DistributedMatrix<scalar_t>& a) {
    if (!a.active()) return 0;
    auto minrc = std::min(a.rows(), a.cols());
    return (is_complex<scalar_t>() ? 4:1) *
      (blas::gelqf_flops(a.rows(), a.cols()) +
       blas::xxglq_flops(a.cols(), a.cols(), minrc)) /
      a.npactives();
  }

  template<typename scalar_t> long long int
  ID_row_flops(const DistributedMatrix<scalar_t>& a, int rank) {
    if (!a.active()) return 0;
    return (is_complex<scalar_t>() ? 4:1) *
      (blas::geqp3_flops(a.cols(), a.rows())
       + blas::trsm_flops(rank, a.cols() - rank, scalar_t(1.), 'L')) /
      a.npactives();
  }

  template<typename scalar_t> long long int
  trsm_flops(Side s, scalar_t alpha, const DistributedMatrix<scalar_t>& a,
             const DistributedMatrix<scalar_t>& b) {
    if (!a.active()) return 0;
    return (is_complex<scalar_t>() ? 4:1) *
      blas::trsm_flops(b.rows(), b.cols(), alpha, char(s)) /
      a.npactives();
  }

  template<typename scalar_t> long long int
  gemm_flops(Trans ta, Trans tb, scalar_t alpha,
             const DistributedMatrix<scalar_t>& a,
             const DistributedMatrix<scalar_t>& b, scalar_t beta) {
    if (!a.active()) return 0;
    return (is_complex<scalar_t>() ? 4:1) *
      blas::gemm_flops
      ((ta==Trans::N) ? a.rows() : a.cols(),
       (tb==Trans::N) ? b.cols() : b.rows(),
       (ta==Trans::N) ? a.cols() : a.rows(), alpha, beta) /
      a.npactives();
  }

  template<typename scalar_t> long long int
  gemv_flops(Trans ta, const DistributedMatrix<scalar_t>& a,
             scalar_t alpha, scalar_t beta) {
    if (!a.active()) return 0;
    auto m = (ta==Trans::N) ? a.rows() : a.cols();
    auto n = (ta==Trans::N) ? a.cols() : a.rows();
    return (is_complex<scalar_t>() ? 4:1) *
      ((alpha != scalar_t(0.)) * m * (n * 2 - 1) +
       (alpha != scalar_t(1.) && alpha != scalar_t(0.)) * m +
       (beta != scalar_t(0.) && beta != scalar_t(1.)) * m +
       (alpha != scalar_t(0.) && beta != scalar_t(0.)) * m) /
      a.npactives();
  }

  template<typename scalar_t> long long int
  orthogonalize_flops(const DistributedMatrix<scalar_t>& a) {
    if (!a.active()) return 0;
    auto minrc = std::min(a.rows(), a.cols());
    return (is_complex<scalar_t>() ? 4:1) *
      (blas::geqrf_flops(a.rows(), minrc) +
       blas::xxgqr_flops(a.rows(), minrc, minrc)) /
      a.npactives();
  }


  template<typename scalar_t>
  std::unique_ptr<const DistributedMatrixWrapper<scalar_t>>
  ConstDistributedMatrixWrapperPtr
  (std::size_t m, std::size_t n, const DistributedMatrix<scalar_t>& D,
   std::size_t i, std::size_t j) {
    return std::unique_ptr<const DistributedMatrixWrapper<scalar_t>>
      (new DistributedMatrixWrapper<scalar_t>
       (m, n, const_cast<DistributedMatrix<scalar_t>&>(D), i, j));
  }


  template<typename scalar_t> void gemm
  (Trans ta, Trans tb, scalar_t alpha, const DistributedMatrix<scalar_t>& A,
   const DistributedMatrix<scalar_t>& B, scalar_t beta,
   DistributedMatrix<scalar_t>& C);

  template<typename scalar_t> void trsm
  (Side s, UpLo u, Trans ta, Diag d, scalar_t alpha,
   const DistributedMatrix<scalar_t>& A, DistributedMatrix<scalar_t>& B);

  template<typename scalar_t> void trsv
  (UpLo ul, Trans ta, Diag d, const DistributedMatrix<scalar_t>& A,
   DistributedMatrix<scalar_t>& B);

  template<typename scalar_t> void gemv
  (Trans ta, scalar_t alpha, const DistributedMatrix<scalar_t>& A,
   const DistributedMatrix<scalar_t>& X, scalar_t beta,
   DistributedMatrix<scalar_t>& Y);

  template<typename scalar_t> DistributedMatrix<scalar_t> vconcat
  (int cols, int arows, int brows, const DistributedMatrix<scalar_t>& a,
   const DistributedMatrix<scalar_t>& b, const BLACSGrid* gnew, int cxt_all);

  template<typename scalar_t> void subgrid_copy_to_buffers
  (const DistributedMatrix<scalar_t>& a, const DistributedMatrix<scalar_t>& b,
   int p0, int npr, int npc, std::vector<std::vector<scalar_t>>& sbuf);

  template<typename scalar_t> void subproc_copy_to_buffers
  (const DenseMatrix<scalar_t>& a, const DistributedMatrix<scalar_t>& b,
   int p0, int npr, int npc, std::vector<std::vector<scalar_t>>& sbuf);

  template<typename scalar_t> void subgrid_add_from_buffers
  (const BLACSGrid* subg, int master, DistributedMatrix<scalar_t>& b,
   std::vector<scalar_t*>& pbuf);

} // end namespace strumpack

#endif // DISTRIBUTED_MATRIX_HPP
