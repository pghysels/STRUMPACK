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
#ifndef PROPORTIONALLY_DISTRIBUTED_SPARSE_MATRIX_HPP
#define PROPORTIONALLY_DISTRIBUTED_SPARSE_MATRIX_HPP

#include "misc/MPIWrapper.hpp"
#include "dense/BLASLAPACKWrapper.hpp"
#include "CompressedSparseMatrix.hpp"
#include <cstddef>
#if 0 // TODO check for compiler support, use C++17?
      // std::experimental::parallel?
#include <parallel/algorithm>
#endif
namespace strumpack {

  template<typename scalar_t,typename integer_t> class EliminationTreeMPIDist;

  /**
   * Sparse matrix distributed based on the proportional mapping of
   * the elimination tree, only to be used during the multifrontal
   * factorization phase. It does not implement stuff like a general
   * spmv (use original matrix for that).
   */
  template<typename scalar_t,typename integer_t>
  class ProportionallyDistributedSparseMatrix :
    public CompressedSparseMatrix<scalar_t,integer_t> {
    using DistM_t = DistributedMatrix<scalar_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using real_t = typename RealType<scalar_t>::value_type;

  public:
    ProportionallyDistributedSparseMatrix();
    virtual ~ProportionallyDistributedSparseMatrix();

    void setup
    (const CSRMatrixMPI<scalar_t,integer_t>& Ampi,
     const MatrixReorderingMPI<scalar_t,integer_t>& nd,
     const EliminationTreeMPIDist<scalar_t,integer_t>& et,
     bool duplicate_fronts);

    void print_dense(const std::string& name) const override;

    void extract_separator
    (integer_t sep_end, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J,
     DenseM_t& B, int depth) const override;
    void extract_front
    (DenseM_t& F11, DenseM_t& F12, DenseM_t& F21,
     integer_t sep_begin, integer_t sep_end,
     const std::vector<integer_t>& upd, int depth) const override;
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
     const integer_t* upd) const override;
    void extract_separator_2d
    (integer_t sep_end, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J, DistM_t& B,
     MPI_Comm comm) const override ;

    void front_multiply
    (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
     const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc) const override;
    void front_multiply_2d
    (integer_t sep_begin, integer_t sep_end,
     const std::vector<integer_t>& upd, const DistM_t& R,
     DistM_t& Srow, DistM_t& Scol, int ctxt_all, MPI_Comm R_comm,
     int depth) const override;

    void spmv(const DenseM_t& x, DenseM_t& y) const override {};
    void omp_spmv(const DenseM_t& x, DenseM_t& y) const override {};
    void spmv(const scalar_t* x, scalar_t* y) const override {};
    void omp_spmv(const scalar_t* x, scalar_t* y) const override {};
    void apply_scaling
    (const std::vector<scalar_t>& Dr,
     const std::vector<scalar_t>& Dc) override {};
    void apply_column_permutation
    (const std::vector<integer_t>& perm) override {};
    int read_matrix_market(const std::string& filename) override {
      return 1;
    };
    real_t max_scaled_residual(const scalar_t* x, const scalar_t* b) const {
      return real_t(1.);
    };
    real_t max_scaled_residual(const DenseM_t& x, const DenseM_t& b) const {
      return real_t(1.);
    };

  protected:
    integer_t _local_cols;  // number of columns stored on this proces
    integer_t* _global_col; // for each local column, this gives the
                            // global column index
  };

  template<typename scalar_t,typename integer_t>
  ProportionallyDistributedSparseMatrix<scalar_t,integer_t>::
  ProportionallyDistributedSparseMatrix()
    : CompressedSparseMatrix<scalar_t,integer_t>(), _global_col(nullptr) { }

  template<typename scalar_t,typename integer_t>
  ProportionallyDistributedSparseMatrix<scalar_t,integer_t>::
  ~ProportionallyDistributedSparseMatrix() {
    delete[] _global_col;
  }

  template<typename scalar_t,typename integer_t> void
  ProportionallyDistributedSparseMatrix<scalar_t,integer_t>::setup
  (const CSRMatrixMPI<scalar_t,integer_t>& Ampi,
   const MatrixReorderingMPI<scalar_t,integer_t>& nd,
   const EliminationTreeMPIDist<scalar_t,integer_t>& et,
   bool duplicate_fronts) {
    this->_n = Ampi.size();
    this->_nnz = Ampi.nnz();
    auto comm = Ampi.comm();
    auto P = mpi_nprocs(comm);
    auto eps = blas::lamch<real_t>('E');

    auto dest = new std::tuple<int,int,int>[Ampi.local_nnz()];
    auto scnts = new int[4*P];
    auto rcnts = scnts + P;
    auto sdisp = scnts + 2*P;
    auto rdisp = scnts + 3*P;
    std::fill(scnts, scnts+P, 0);
#pragma omp parallel for
    for (integer_t r=0; r<Ampi.local_rows(); r++) {
      auto r_perm = nd.perm[r+Ampi.begin_row()];
      auto hij = Ampi.get_ptr()[r+1]-Ampi.get_ptr()[0];
      for (integer_t j=Ampi.get_ptr()[r]-Ampi.get_ptr()[0]; j<hij; j++) {
        if (std::abs(Ampi.get_val()[j]) > eps) {
          auto c_perm = nd.perm[Ampi.get_ind()[j]];
          auto d = dest[j] = et.get_sparse_mapped_destination
            (Ampi, r_perm, c_perm, duplicate_fronts);
          auto hip = std::get<0>(d)+std::get<1>(d);
          for (int p=std::get<0>(d); p<hip; p+=std::get<2>(d))
#pragma omp atomic
            scnts[p]++;
        }
      }
    }
    struct Triplet { integer_t r; integer_t c; scalar_t a; };
    auto sbuf = new Triplet[std::accumulate(scnts, scnts+P, 0)];
    auto pp = new Triplet*[P];
    pp[0] = sbuf;
    for (int p=1; p<P; p++) pp[p] = pp[p-1] + scnts[p-1];
    for (integer_t r=0; r<Ampi.local_rows(); r++) {
      auto r_perm = nd.perm[r+Ampi.begin_row()];
      auto hij = Ampi.get_ptr()[r+1]-Ampi.get_ptr()[0];
      for (integer_t j=Ampi.get_ptr()[r]-Ampi.get_ptr()[0]; j<hij; j++) {
        auto a = Ampi.get_val()[j];
        if (std::abs(a) > eps) {
          Triplet t = {r_perm, nd.perm[Ampi.get_ind()[j]], a};
          auto d = dest[j];
          auto hip = std::get<0>(d)+std::get<1>(d);
          for (int p=std::get<0>(d); p<hip; p+=std::get<2>(d)) {
            *(pp[p]) = t;
            pp[p]++;
          }
        }
      }
    }
    delete[] dest;
    delete[] pp;
    for (int p=0; p<P; p++) scnts[p] *= sizeof(Triplet);
    MPI_Alltoall(scnts, 1, MPI_INT, rcnts, 1, MPI_INT, comm);
    sdisp[0] = rdisp[0] = 0;
    for (int p=1; p<P; p++) {
      sdisp[p] = sdisp[p-1] + scnts[p-1];
      rdisp[p] = rdisp[p-1] + rcnts[p-1];
    }
    std::vector<Triplet> triplets
      (std::accumulate(rcnts, rcnts+P, 0) / sizeof(Triplet));
    MPI_Alltoallv
      (sbuf, scnts, sdisp, MPI_BYTE, triplets.data(),
       rcnts, rdisp, MPI_BYTE, comm);
    delete[] scnts;
    delete[] sbuf;

    // TODO this sort can be avoided: first make the CSR/CSC
    // representation, then sort that row per row (in openmp parallel
    // for)!!
#if 0 // TODO check for compiler support, use C++17?
      // std::experimental::parallel?
    __gnu_parallel::sort
      (triplets.begin(), triplets.end(),
       [](const Triplet& a, const Triplet& b) {
        // sort according to column, then rows
        if (a.c != b.c) return (a.c < b.c);
        return (a.r < b.r);
      });
#else
    std::sort
      (triplets.begin(), triplets.end(),
       [](const Triplet& a, const Triplet& b) {
        // sort according to column, then rows
        if (a.c != b.c) return (a.c < b.c);
        return (a.r < b.r);
      });
#endif
    integer_t _local_nnz = triplets.size();
    _local_cols = _local_nnz ? 1 : 0;
    for (integer_t t=1; t<_local_nnz; t++)
      if (triplets[t].c != triplets[t-1].c) _local_cols++;
    this->_ptr = new integer_t[_local_cols+1];
    _global_col = new integer_t[_local_cols];
    this->_ind = new integer_t[_local_nnz];
    this->_val = new scalar_t[_local_nnz];
    integer_t col = 0;
    this->_ptr[col] = 0;
    if (_local_cols) {
      this->_ptr[1] = 0;
      _global_col[col] = triplets[0].c;
    }
    for (integer_t j=0; j<_local_nnz; j++) {
      this->_ind[j] = triplets[j].r;
      this->_val[j] = triplets[j].a;
      if (j > 0 && (triplets[j].c != triplets[j-1].c)) {
        col++;
        this->_ptr[col+1] = this->_ptr[col];
        _global_col[col] = triplets[j].c;
      }
      this->_ptr[col+1]++;
    }
  }

  template<typename scalar_t,typename integer_t> void
  ProportionallyDistributedSparseMatrix<scalar_t,integer_t>::
  print_dense(const std::string& name) const {
    std::fstream fs(name, std::fstream::out);
    auto M = new scalar_t[this->_n * this->_n];
    std::fill(M, M+(this->_n*this->_n), scalar_t(0.));
    for (integer_t c=0; c<_local_cols; c++)
      for (integer_t j=this->_ptr[c]; j<this->_ptr[c+1]; j++)
        M[this->_ind[j] + _global_col[c]*this->_n] = this->_val[j];
    fs << "A = [\n";
    for (integer_t row=0; row<this->_n; row++) {
      for (integer_t col=0; col<this->_n; col++)
        fs << M[row + this->_n * col] << " ";
      fs << ";" << std::endl;
    }
    fs << "];" << std::endl;
    delete[] M;
  }

  template<typename scalar_t,typename integer_t> void
  ProportionallyDistributedSparseMatrix<scalar_t,integer_t>::extract_separator
  (integer_t sep_end, const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J, DenseM_t& B, int depth) const {
    integer_t m = I.size();
    integer_t n = J.size();
    if (m == 0 || n == 0) return;
    for (integer_t j=0; j<n; j++) {
      integer_t c = std::lower_bound
        (_global_col, _global_col+_local_cols, J[j]) - _global_col;
      if (c == _local_cols || _global_col[c] != integer_t(J[j])) {
        std::fill(B.ptr(0,j), B.ptr(m, j), scalar_t(0.));
        continue;
      }
      auto rmin = this->_ind[this->_ptr[c]];
      auto rmax = this->_ind[this->_ptr[c+1]-1];
      for (integer_t i=0; i<m; i++) {
        integer_t r = I[i];
        if (r >= rmin && r <= rmax &&
            (_global_col[c] < sep_end || r < sep_end)) {
          auto a_pos = this->_ptr[c];
          auto a_max = this->_ptr[c+1];
          while (a_pos < a_max-1 && this->_ind[a_pos] < r) a_pos++;
          B(i,j) = (this->_ind[a_pos] == r) ?
            this->_val[a_pos] : scalar_t(0.);
        } else B(i,j) = scalar_t(0.);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  ProportionallyDistributedSparseMatrix<scalar_t,integer_t>::front_multiply
  (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc) const {
    integer_t dupd = upd.size();
    auto ds = shi - slo;
    auto nbvec = R.cols();
    auto c = std::lower_bound
      (_global_col, _global_col+_local_cols, slo) - _global_col;
    auto chi = std::lower_bound
      (_global_col+c, _global_col+_local_cols, shi) - _global_col;
    for (; c<chi; c++) {
      auto col = _global_col[c];
      integer_t row_upd = 0;
      for (auto j=this->_ptr[c]; j<this->_ptr[c+1]; j++) {
        auto row = this->_ind[j];
        if (row >= slo) {
          if (row < shi) {
            auto a = this->_val[j];
            for (std::size_t k=0; k<nbvec; k++) {
              Sr(row-slo, k) += a * R(col-slo, k);
              Sc(col-slo, k) += a * R(row-slo, k);
            }
          } else {
            while (row_upd < dupd && upd[row_upd] < row) row_upd++;
            if (row_upd == dupd) break;
            if (upd[row_upd] == row) {
              auto a = this->_val[j];
              for (std::size_t k=0; k<nbvec; k++) {
                Sr(ds+row_upd, k) += a * R(col-slo, k);
                Sc(col-slo, k) += a * R(ds+row_upd, k);
              }
            }
          }
        }
      }
    }
    for (integer_t i=0; i<dupd; i++) { // update columns
      //while (c < _local_cols && _global_col[c] < upd[i]) c++;
      c = std::lower_bound
        (_global_col+c, _global_col+_local_cols, upd[i]) - _global_col;
      if (c == _local_cols || _global_col[c] != upd[i]) continue;
      for (auto j=this->_ptr[c]; j<this->_ptr[c+1]; j++) {
        auto row = this->_ind[j];
        if (row >= slo) {
          if (row < shi) {
            auto a = this->_val[j];
            for (std::size_t k=0; k<nbvec; k++) {
              Sr(row-slo, k) += a * R(ds+i, k);
              Sc(ds+i, k) += a * R(row-slo, k);
            }
          } else break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  ProportionallyDistributedSparseMatrix<scalar_t,integer_t>::extract_front
  (DenseM_t& F11, DenseM_t& F12, DenseM_t& F21, integer_t sep_begin,
   integer_t sep_end, const std::vector<integer_t>& upd, int depth) const {
    integer_t dim_upd = upd.size();
    auto c = std::lower_bound
      (_global_col, _global_col+_local_cols, sep_begin) - _global_col;
    auto chi = std::lower_bound
      (_global_col+c, _global_col+_local_cols, sep_end) - _global_col;
    for (; c<chi; c++) {
      auto col = _global_col[c];
      integer_t row_ptr = 0;
      auto hij = this->_ptr[c+1];
      for (integer_t j=this->_ptr[c]; j<hij; j++) {
        auto row = this->_ind[j];
        if (row >= sep_begin) {
          if (row < sep_end)
            F11(row-sep_begin, col-sep_begin) = this->_val[j];
          else {
            while (row_ptr<dim_upd && upd[row_ptr]<row)
              row_ptr++;
            if (row_ptr == dim_upd) break;
            if (upd[row_ptr] == row)
              F21(row_ptr, col-sep_begin) = this->_val[j];
          }
        }
      }
    }
    for (integer_t i=0; i<dim_upd; ++i) { // update columns
      //while (c < _local_cols && _global_col[c] < upd[i]) c++;
      c = std::lower_bound
        (_global_col+c, _global_col+_local_cols, upd[i]) - _global_col;
      if (c == _local_cols || _global_col[c] != upd[i]) continue;
      for (integer_t j=this->_ptr[c]; j<this->_ptr[c+1]; j++) {
        auto row = this->_ind[j];
        if (row >= sep_begin) {
          if (row < sep_end)
            F12(row-sep_begin, i) = this->_val[j];
          else break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  ProportionallyDistributedSparseMatrix<scalar_t,integer_t>::extract_F11_block
  (scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows,
   integer_t col, integer_t nr_cols) const {
    auto c = std::lower_bound
      (_global_col, _global_col+_local_cols, col) - _global_col;
    auto chi = std::lower_bound
      (_global_col+c, _global_col+_local_cols, col+nr_cols) - _global_col;
    for (; c<chi; c++) {
      for (integer_t j=this->_ptr[c]; j<this->_ptr[c+1]; j++) {
        auto r = this->_ind[j];
        if (r >= row) {
          if (r < row + nr_rows)
            F[r-row + (_global_col[c]-col)*ldF] = this->_val[j];
          else break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  ProportionallyDistributedSparseMatrix<scalar_t,integer_t>::extract_F12_block
  (scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows,
   integer_t col, integer_t nr_cols, const integer_t* upd) const {
    if (nr_cols == 0 || nr_rows == 0) return;
    auto c = std::lower_bound
      (_global_col, _global_col+_local_cols, upd[0]) - _global_col;
    for (integer_t i=0; i<nr_cols; i++) {
      //while (c < _local_cols && _global_col[c] < upd[i]) c++;
      if (i > 0)
        c = std::lower_bound
          (_global_col+c, _global_col+_local_cols, upd[i]) - _global_col;
      if (c == _local_cols || _global_col[c] != upd[i]) continue;
      for (integer_t j=this->_ptr[c]; j<this->_ptr[c+1]; j++) {
        auto r = this->_ind[j];
        if (r >= row) {
          if (r < row+nr_rows) F[r-row + i*ldF] = this->_val[j];
          else break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  ProportionallyDistributedSparseMatrix<scalar_t,integer_t>::extract_F21_block
  (scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows,
   integer_t col, integer_t nr_cols, const integer_t* upd) const{
    if (nr_rows == 0 || nr_cols == 0) return;
    auto c = std::lower_bound
      (_global_col, _global_col+_local_cols, col) - _global_col;
    auto chi = std::lower_bound
      (_global_col+c, _global_col+_local_cols, col+nr_cols) - _global_col;
    for (; c<chi; c++) {
      integer_t row_upd = 0;
      for (integer_t j=this->_ptr[c]; j<this->_ptr[c+1]; j++) {
        auto r = this->_ind[j];
        while (row_upd<nr_rows && upd[row_upd]<r) row_upd++;
        if (row_upd == nr_rows) break;
        if (upd[row_upd] == r)
          F[row_upd + (_global_col[c]-col)*ldF] = this->_val[j];
      }
    }
  }

  /** mat should be defined on the same communicator as F11 */
  template<typename scalar_t,typename integer_t> void
  ProportionallyDistributedSparseMatrix<scalar_t,integer_t>::
  extract_separator_2d
  (integer_t sep_end, const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J, DistM_t& B, MPI_Comm comm) const {
    if (!B.active()) return;
    integer_t m = B.rows();
    integer_t n = B.cols();
    if (m == 0 || n == 0) return;
    B.zero();
    for (integer_t j=0; j<n; j++) {
      integer_t jj = J[j];
      integer_t c = std::lower_bound
        (_global_col, _global_col+_local_cols, jj) - _global_col;
      if (c == _local_cols || _global_col[c] != jj) continue;
      for (integer_t i=0; i<m; i++) {
        integer_t ii = I[i];
        if (jj >= sep_end && ii >= sep_end) break;
        for (integer_t k=this->_ptr[c]; k<this->_ptr[c+1]; k++) {
          if (this->_ind[k] == ii) {
            B.global(i, j, this->_val[k]);
            break;
          }
        }
      }
    }
  }


  /**
   * Multiply the front F = [ A(I^{sep},I^{sep})  A(I^{sep},I^{upd}) ]
   *                        [ A(I^{upd},I^{sep})  0                  ]
   * with the random matrix R. Compute both Srow = F R and Scol = F^T R.
   *
   * R, Srow and Scol are distributed in 2D block cyclic format, on a
   * p_rows x p_cols grid. There is only communication within columns
   * of the processor grid.
   *
   * The algorithm is as follows:
   * There are 3 loops over all the nonzeros in F.
   *
   *  # first loop to count what needs to be communicated
   *  - for all nnz fij in F do
   *  -     if I own row j of R
   *  -         I will need to send row j of R to the owner of row i in the same processor column as me
   *  -     if I own row i of R  # for the transposed multiply
   *  -         I will need to send row i of R to the owner of row j in the same processor column as me
   *  -     if I own row i of Srow
   *  -         I will need to receive aij times row j of R from the owner of row j of R in the same processor column as me
   *  -     if I own row j of Scol
   *  -         I will need to receive aji times row i of R from the owner of row i of R in the same processor column as me
   *
   *  # second loop to do the actual multiplication and copy results to send buffers
   *  - for all nnz fij in F do
   *  -     if I own row j of R
   *  -         multiply row j of R with aij and copy the result in sendbuffer for the owner of row i in the same processor column as me
   *  -     if I own row i of R  # for the transposed multiply
   *  -         multiply row i of R with aji and copy the result in sendbuffer for the owner of row j in the same processor column as me
   *
   *  # communicate with all the other processes in the same processor column
   *  - for all p in my processor column: MPI_Isend()
   *  - for all p in my processor column: MPI_Irecv()
   *
   *  - wait for all incoming messages
   *  # final loop to receive data and add it to Srow and Scol
   *  - for all nnz fij in F do
   *  -     if I own row i of Srow
   *  -         copy aij times row j of R from the recvbuffer of the owner of row j of R in the same processor column as me to row i in Srow
   *  -     if I own row j of Scol
   *  -         copy aji times row i of R from the recvbuffer of the owner of row i of R in the same processor column as me to row j in Scol
   *
   *  - wait for all outgoing messages to complete
   */
  template<typename scalar_t,typename integer_t> void
  ProportionallyDistributedSparseMatrix<scalar_t,integer_t>::front_multiply_2d
  (integer_t sep_begin, integer_t sep_end, const std::vector<integer_t>& upd,
   const DistM_t& R, DistM_t& Srow, DistM_t& Scol, int ctxt_all,
   MPI_Comm R_comm, int depth) const {

    assert(R.fixed());
    assert(Srow.fixed());
    assert(Scol.fixed());

    if (!R.active()) return;
    integer_t dim_upd = upd.size();
    long long int local_flops = 0;
    auto dim_sep = sep_end - sep_begin;
    auto lcols = R.lcols();
    auto p_rows = R.prows();
    auto p_row  = R.prow();
    auto rows_to = new integer_t[2*p_rows];
    auto rows_from = rows_to + p_rows;
    std::fill(rows_to, rows_to+2*p_rows, 0);

    auto clo = std::lower_bound
      (_global_col, _global_col+_local_cols, sep_begin) - _global_col;
    auto chi = std::lower_bound
      (_global_col+clo, _global_col+_local_cols, sep_end) - _global_col;
    integer_t c = clo;
    for (; c<chi; c++) { // separator columns
      auto row_j_rank = R.rowg2p_fixed(_global_col[c] - sep_begin);
      integer_t row_upd = 0;
      auto hij = this->_ptr[c+1];
      for (integer_t j=this->_ptr[c]; j<hij; j++) {
        auto row = this->_ind[j];
        if (row >= sep_begin) {
          if (row < sep_end) {
            auto row_i_rank = R.rowg2p_fixed(row - sep_begin);
            if (row_j_rank == p_row) {
              rows_to[row_i_rank]++;
              rows_from[row_i_rank]++;
            }
            if (row_i_rank == p_row) {
              rows_to[row_j_rank]++;
              rows_from[row_j_rank]++;
            }
            local_flops += 4 * lcols;
          } else {
            while (row_upd < dim_upd && upd[row_upd] < row) row_upd++;
            if (row_upd == dim_upd) break;
            if (upd[row_upd] == row) {
              auto row_i_rank = R.rowg2p_fixed(dim_sep + row_upd);
              if (row_j_rank == p_row) {
                rows_to[row_i_rank]++;
                rows_from[row_i_rank]++;
              }
              if (row_i_rank == p_row) {
                rows_to[row_j_rank]++;
                rows_from[row_j_rank]++;
              }
              local_flops += 4 * lcols;
            }
          }
        }
      }
    }
    for (integer_t i=0; i<dim_upd; i++) { // update columns
      c = std::lower_bound
        (_global_col+c, _global_col+_local_cols, upd[i]) - _global_col;
      if (c == _local_cols || _global_col[c] != upd[i]) continue;
      auto row_j_rank = R.rowg2p_fixed(dim_sep + i);
      auto hij = this->_ptr[c+1];
      for (integer_t j=this->_ptr[c]; j<hij; j++) {
        integer_t row = this->_ind[j];
        if (row >= sep_begin) {
          if (row < sep_end) {
            auto row_i_rank = R.rowg2p_fixed(row - sep_begin);
            if (row_j_rank == p_row) {
              rows_to[row_i_rank]++;
              rows_from[row_i_rank]++;
            }
            if (row_i_rank == p_row) {
              rows_to[row_j_rank]++;
              rows_from[row_j_rank]++;
            }
            local_flops += 4 * lcols;
          } else break;
        }
      }
    }
    if (R.is_master()) {
      STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    }

    size_t ssize = std::accumulate(rows_to, rows_to+p_rows, 0);
    size_t rsize = std::accumulate(rows_from, rows_from+p_rows, 0);
    auto sbuf = new scalar_t[(ssize+rsize)*lcols];
    auto rbuf = sbuf+ssize*lcols;
    std::fill(sbuf, sbuf+ssize*lcols, scalar_t(0.));
    auto pp = new scalar_t*[p_rows];
    pp[0] = sbuf;
    for (integer_t p=1; p<p_rows; p++)
      pp[p] = pp[p-1] + rows_to[p-1]*lcols;

    for (c=clo; c<chi; c++) { // separator columns
      auto Aj = _global_col[c] - sep_begin;
      auto row_j_rank = R.rowg2p_fixed(Aj);
      integer_t row_upd = 0;
      for (integer_t j=this->_ptr[c]; j<this->_ptr[c+1]; j++) {
        auto row = this->_ind[j];
        if (row >= sep_begin) {
          if (row < sep_end) {
            auto a = this->_val[j];
            auto Ai = row - sep_begin;
            auto row_i_rank = R.rowg2p_fixed(Ai);
            if (row_j_rank == p_row) {
              for (integer_t k=0, r=R.rowg2l_fixed(Aj); k<lcols; k++)
                pp[row_i_rank][k] += a * R(r, k);
              pp[row_i_rank] += lcols;
            }
            if (row_i_rank == p_row) { // transpose
              for (integer_t k=0, r=R.rowg2l_fixed(Ai); k<lcols; k++)
                pp[row_j_rank][k] += a * R(r, k);
              pp[row_j_rank] += lcols;
            }
          } else {
            while (row_upd < dim_upd && upd[row_upd] < row)
              row_upd++;
            if (row_upd == dim_upd) break;
            if (upd[row_upd] == row) {
              auto a = this->_val[j];
              auto Ai = dim_sep + row_upd;
              auto row_i_rank = R.rowg2p_fixed(Ai);
              if (row_j_rank == p_row) {
                for (integer_t k=0, r=R.rowg2l_fixed(Aj); k<lcols; k++)
                  pp[row_i_rank][k] += a * R(r, k);
                pp[row_i_rank] += lcols;
              }
              if (row_i_rank == p_row) { // transpose
                for (integer_t k=0, r=R.rowg2l_fixed(Ai); k<lcols; k++)
                  pp[row_j_rank][k] += a * R(r, k);
                pp[row_j_rank] += lcols;
              }
            }
          }
        }
      }
    }
    for (integer_t i=0; i<dim_upd; i++) { // update columns
      c = std::lower_bound
        (_global_col+c, _global_col+_local_cols, upd[i]) - _global_col;
      if (c == _local_cols || _global_col[c] != upd[i]) continue;
      auto Aj = dim_sep + i;
      auto row_j_rank = R.rowg2p_fixed(Aj);
      auto hij = this->_ptr[c+1];
      for (integer_t j=this->_ptr[c]; j<hij; j++) {
        integer_t row = this->_ind[j];
        if (row >= sep_begin) {
          if (row < sep_end) {
            auto a = this->_val[j];
            auto Ai = row - sep_begin;
            auto row_i_rank = R.rowg2p_fixed(Ai);
            if (row_j_rank == p_row) {
              for (integer_t k=0, r=R.rowg2l_fixed(Aj); k<lcols; k++)
                pp[row_i_rank][k] += a * R(r, k);
              pp[row_i_rank] += lcols;
            }
            if (row_i_rank == p_row) { // transpose
              for (integer_t k=0, r=R.rowg2l_fixed(Ai); k<lcols; k++)
                pp[row_j_rank][k] += a * R(r, k);
              pp[row_j_rank] += lcols;
            }
          } else break;
        }
      }
    }

    // TODO instead of a send/receive loop, get the column
    // communicator from BLACS?

    pp[0] = sbuf;
    for (integer_t p=1; p<p_rows; p++)
      pp[p] = pp[p-1] + rows_to[p-1]*lcols;
    auto p_col  = R.pcol();
    auto sreq = new MPI_Request[p_rows*2];
    auto rreq = sreq + p_rows;
    for (integer_t p=0; p<p_rows; p++)
      MPI_Isend(pp[p], rows_to[p]*lcols, mpi_type<scalar_t>(),
                p+p_col*p_rows, 0, R_comm, sreq+p);

    pp[0] = rbuf;
    for (integer_t p=1; p<p_rows; p++)
      pp[p] = pp[p-1] + rows_from[p-1]*lcols;
    for (integer_t p=0; p<p_rows; p++)
      MPI_Irecv(pp[p], rows_from[p]*lcols, mpi_type<scalar_t>(),
                p+p_col*p_rows, 0, R_comm, rreq+p);

    // wait for all incoming messages
    MPI_Waitall(p_rows, rreq, MPI_STATUSES_IGNORE);
    for (c=clo; c<chi; c++) { // separator columns
      auto Aj = _global_col[c] - sep_begin;
      auto row_j_rank = R.rowg2p_fixed(Aj);
      integer_t row_upd = 0;
      auto hij = this->_ptr[c+1];
      for (integer_t j=this->_ptr[c]; j<hij; j++) {
        auto row = this->_ind[j];
        if (row >= sep_begin) {
          if (row < sep_end) {
            auto Ai = row - sep_begin;
            auto row_i_rank = R.rowg2p_fixed(Ai);
            if (row_i_rank == p_row) {
              for (integer_t k=0, r=Srow.rowg2l_fixed(Ai); k<lcols; k++)
                Srow(r, k) += pp[row_j_rank][k];
              pp[row_j_rank] += lcols;
            }
            if (row_j_rank == p_row) { // transpose
              for (integer_t k=0, r=Srow.rowg2l_fixed(Aj); k<lcols; k++)
                Scol(r, k) += pp[row_i_rank][k];
              pp[row_i_rank] += lcols;
            }
          } else {
            while (row_upd < dim_upd && upd[row_upd] < row)
              row_upd++;
            if (row_upd == dim_upd) break;
            if (upd[row_upd] == row) {
              auto Ai = dim_sep + row_upd;
              auto row_i_rank = R.rowg2p_fixed(Ai);
              if (row_i_rank == p_row) {
                for (integer_t k=0, r=Srow.rowg2l_fixed(Ai); k<lcols; k++)
                  Srow(r, k) += pp[row_j_rank][k];
                pp[row_j_rank] += lcols;
              }
              if (row_j_rank == p_row) { // transpose
                for (integer_t k=0, r=Scol.rowg2l_fixed(Aj); k<lcols; k++)
                  Scol(r, k) += pp[row_i_rank][k];
                pp[row_i_rank] += lcols;
              }
            }
          }
        }
      }
    }
    for (integer_t i=0; i<dim_upd; i++) { // update columns
      c = std::lower_bound
        (_global_col+c, _global_col+_local_cols, upd[i]) - _global_col;
      if (c == _local_cols || _global_col[c] != upd[i])
        continue;
      auto Aj = dim_sep + i;
      auto row_j_rank = R.rowg2p_fixed(Aj);
      auto hij = this->_ptr[c+1];
      for (integer_t j=this->_ptr[c]; j<hij; j++) {
        integer_t row = this->_ind[j];
        if (row >= sep_begin) {
          if (row < sep_end) {
            auto Ai = row - sep_begin;
            auto row_i_rank = R.rowg2p_fixed(Ai);
            if (row_i_rank == p_row) {
              for (integer_t k=0, r=Srow.rowg2l_fixed(Ai); k<lcols; k++)
                Srow(r, k) += pp[row_j_rank][k];
              pp[row_j_rank] += lcols;
            }
            if (row_j_rank == p_row) { // transpose
              for (integer_t k=0, r=Scol.rowg2l_fixed(Aj); k<lcols; k++)
                Scol(r, k) += pp[row_i_rank][k];
              pp[row_i_rank] += lcols;
            }
          } else break;
        }
      }
    }
    delete[] pp;
    delete[] rows_to;

    // wait for sends to finish
    MPI_Waitall(p_rows, sreq, MPI_STATUSES_IGNORE);
    delete[] sreq;
    delete[] sbuf;
  }

} // end namespace strumpack

#endif // PROPORTIONALLY_DISTRIBUTED_SPARSE_MATRIX_HPP
