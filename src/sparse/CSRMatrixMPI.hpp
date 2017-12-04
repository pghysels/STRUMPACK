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
#ifndef CSRMATRIXMPI_HPP
#define CSRMATRIXMPI_HPP
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <memory>

#include "misc/MPIWrapper.hpp"
#include "dense/BLASLAPACKWrapper.hpp"
#include "CSRGraph.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class SPMVBuffers {
  public:
    std::vector<integer_t> _spmv_sranks;
    // ranks of the process from which I need to receive something
    std::vector<integer_t> _spmv_rranks;
    std::vector<integer_t> _spmv_soff;
    std::vector<integer_t> _spmv_roffs;
    // indices to receive from each rank from which I need to receive
    std::vector<integer_t> _spmv_sind;
    // indices to receive from each rank from which I need to receive
    std::vector<scalar_t> _spmv_sbuf;
    std::vector<scalar_t> _spmv_rbuf;
    // for each off-diagonal entry _spmv_prbuf stores the
    // corresponding index in the receive buffer
    std::vector<integer_t> _spmv_prbuf;
  };

  template<typename scalar_t,typename integer_t>
  class CSRMatrixMPI : public CompressedSparseMatrix<scalar_t,integer_t> {
    using DistM_t = DistributedMatrix<scalar_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using real_t = typename RealType<scalar_t>::value_type;

  public:
    CSRMatrixMPI();
    CSRMatrixMPI
    (integer_t local_rows, const integer_t* row_ptr,
     const integer_t* col_ind, const scalar_t* values,
     const integer_t* dist, MPI_Comm comm, bool symm_sparse);
    CSRMatrixMPI
    (integer_t local_rows, const integer_t* d_ptr, const integer_t* d_ind,
     const scalar_t* d_val, const integer_t* o_ptr, const integer_t* o_ind,
     const scalar_t* o_val, const integer_t* garray, MPI_Comm comm,
     bool symm_sparse=false);
    CSRMatrixMPI(const CSRMatrixMPI<scalar_t,integer_t>& A);
    CSRMatrixMPI
    (const CSRMatrix<scalar_t,integer_t>* A, MPI_Comm c, bool only_at_root);

    inline const std::vector<integer_t>& get_dist() const { return _dist; }
    inline std::vector<integer_t>& get_dist() { return _dist; }
    inline integer_t local_rows() const { return _end_row - _begin_row; }
    inline integer_t begin_row() const { return _begin_row; }
    inline integer_t end_row() const { return _end_row; }
    inline MPI_Comm comm() const { return _comm; }
    inline integer_t local_nnz() const { return _local_nnz; }

    void spmv(const DenseM_t& x, DenseM_t& y) const override;
    void omp_spmv(const DenseM_t& x, DenseM_t& y) const override;
    void spmv(const scalar_t* x, scalar_t* y) const override;
    void omp_spmv(const scalar_t* x, scalar_t* y) const override;

    void apply_scaling
    (const std::vector<scalar_t>& Dr,
     const std::vector<scalar_t>& Dc) override;
    void permute(const integer_t* iorder, const integer_t* order) override;
    std::unique_ptr<CSRMatrix<scalar_t,integer_t>> gather() const;
    int permute_and_scale
    (int job, std::vector<integer_t>& perm,
     std::vector<scalar_t>& Dr,
     std::vector<scalar_t>& Dc, bool apply=true) override;
    void apply_column_permutation
    (const std::vector<integer_t>& perm) override;
    void symmetrize_sparsity() override;
    int read_matrix_market(const std::string& filename) override;

    real_t max_scaled_residual
    (const DenseM_t& x, const DenseM_t& b) const override;
    real_t max_scaled_residual
    (const scalar_t* x, const scalar_t* b) const override;

    // TODO return by value? return a unique_ptr?
    CSRGraph<integer_t>* get_sub_graph
    (const integer_t* perm,
     const std::pair<integer_t,integer_t>* graph_ranges) const;
    void print() const override;
    void print_dense(const std::string& name) const override;
    void print_MM(const std::string& filename) const override;
    void check() const;

    // implement outside of this class
    void extract_separator
    (integer_t, const std::vector<std::size_t>&,
     const std::vector<std::size_t>&, DenseM_t&, int) const override {}
    void extract_separator_2d
    (integer_t, const std::vector<std::size_t>&,
     const std::vector<std::size_t>&, DistM_t&, MPI_Comm) const override {}
    void extract_front
    (DenseM_t&, DenseM_t&, DenseM_t&, integer_t,
     integer_t, const std::vector<integer_t>&, int) const override {}
    void extract_F11_block
    (scalar_t*, integer_t, integer_t, integer_t,
     integer_t, integer_t) const override {}
    void extract_F12_block
    (scalar_t*, integer_t, integer_t, integer_t, integer_t,
     integer_t, const integer_t*) const override {}
    void extract_F21_block
    (scalar_t*, integer_t, integer_t, integer_t, integer_t,
     integer_t, const integer_t*) const override {}
    void front_multiply
    (integer_t, integer_t, const std::vector<integer_t>&,
     const DenseM_t&, DenseM_t&, DenseM_t&) const override {}
    void front_multiply_2d
    (integer_t, integer_t, const std::vector<integer_t>&, const DistM_t&,
     DistM_t&, DistM_t&, int, MPI_Comm, int) const override {}

  protected:
    void split_diag_offdiag();
    void setup_spmv_buffers() const;
    bool is_mpi_root() const override { return mpi_root(_comm); }

    MPI_Comm _comm;

    /**
     * _dist is the same as the vtxdist array defined by parmetis, it
     *  is the same for each process processor p holds rows
     *  [_dist[p],_dist[p+1]-1]
     */
    std::vector<integer_t> _dist;

    /**
     * _odiag_ptr points to the start of the off-(block)-diagonal
     *  elements.
     */
    std::vector<integer_t> _offdiag_start;

    integer_t _local_rows; // = _end_row - _begin_row
    integer_t _local_nnz;  // = _ptr[local_rows]
    integer_t _begin_row;  // = _dist[rank]
    integer_t _end_row;    // = _dist[rank+1]

    mutable std::unique_ptr<SPMVBuffers<scalar_t,integer_t>> _spmv_buffers;
  };

  template<typename scalar_t,typename integer_t>
  CSRMatrixMPI<scalar_t,integer_t>::CSRMatrixMPI()
    : CompressedSparseMatrix<scalar_t,integer_t>(), _comm(MPI_COMM_NULL),
    _local_rows(0), _local_nnz(0), _begin_row(0), _end_row(0) {}

  template<typename scalar_t,typename integer_t>
  CSRMatrixMPI<scalar_t,integer_t>::CSRMatrixMPI
  (integer_t local_rows, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, const integer_t* dist, MPI_Comm comm,
   bool symm_sparse) : CompressedSparseMatrix<scalar_t,integer_t>(),
    _comm(comm) {
    auto P = mpi_nprocs(_comm);
    auto rank = mpi_rank(_comm);
    _local_rows = local_rows;
    _local_nnz = row_ptr[local_rows]-row_ptr[0];
    _begin_row = dist[rank];
    _end_row = dist[rank+1];
    _dist.resize(P+1);
    std::copy(dist, dist+P+1, _dist.data());
    this->_ptr = new integer_t[_local_rows+1];
    this->_ind = new integer_t[_local_nnz];
    this->_val = new scalar_t[_local_nnz];
    std::copy(row_ptr, row_ptr+_local_rows+1, this->_ptr);
    std::copy(col_ind, col_ind+_local_nnz, this->_ind);
    std::copy(values, values+_local_nnz, this->_val);
    this->_n = dist[P];
    MPI_Allreduce
      (&_local_nnz, &this->_nnz, 1, mpi_type<integer_t>(), MPI_SUM, _comm);
    this->_symm_sparse = symm_sparse;
    for (integer_t r=_local_rows; r>=0; r--)
      this->_ptr[r] -= this->_ptr[0];
    split_diag_offdiag();
    check();
  }

  template<typename scalar_t,typename integer_t>
  CSRMatrixMPI<scalar_t,integer_t>::CSRMatrixMPI
  (integer_t local_rows, const integer_t* d_ptr, const integer_t* d_ind,
   const scalar_t* d_val, const integer_t* o_ptr, const integer_t* o_ind,
   const scalar_t* o_val, const integer_t* garray, MPI_Comm comm,
   bool symm_sparse)
    : CompressedSparseMatrix<scalar_t,integer_t>(), _comm(comm) {
    auto P = mpi_nprocs(_comm);
    auto rank = mpi_rank(_comm);
    _local_rows = local_rows;
    _local_nnz = 0;
    if (d_ptr) // diagonal block can be empty (NULL)
      _local_nnz += d_ptr[local_rows] - d_ptr[0];
    if (o_ptr) // off-diagonal block can be empty (NULL)
      _local_nnz += o_ptr[local_rows] - o_ptr[0];
    this->_symm_sparse = symm_sparse;
    _dist.resize(P+1);
    MPI_Allgather
      (&local_rows, 1, mpi_type<integer_t>(),
       &_dist[1], 1, mpi_type<integer_t>(), _comm);
    for (int p=1; p<=P; p++) _dist[p] = _dist[p-1] + _dist[p];
    _begin_row = _dist[rank];
    _end_row = _dist[rank+1];
    MPI_Allreduce
      (&_local_nnz, &this->_nnz, 1, mpi_type<integer_t>(), MPI_SUM, _comm);
    this->_n = _dist[P];
    this->_ptr = new integer_t[_local_rows+1];
    this->_ind = new integer_t[_local_nnz];
    this->_val = new scalar_t[_local_nnz];
    this->_ptr[0] = 0;
    _offdiag_start.resize(_local_rows);
    for (integer_t r=0, nz=0; r<local_rows; r++) {
      this->_ptr[r+1] = this->_ptr[r];
      if (d_ptr)
        for (integer_t j=d_ptr[r]-d_ptr[0]; j<d_ptr[r+1]-d_ptr[0]; j++) {
          this->_ind[nz] = d_ind[j] + _begin_row;
          this->_val[nz++] = d_val[j];
          this->_ptr[r+1]++;
        }
      _offdiag_start[r] = this->_ptr[r+1];
      if (o_ptr)
        for (integer_t j=o_ptr[r]-o_ptr[0]; j<o_ptr[r+1]-o_ptr[0]; j++) {
          this->_ind[nz] = garray[o_ind[j]];
          this->_val[nz++] = o_val[j];
          this->_ptr[r+1]++;
        }
    }
    check();
  }

  template<typename scalar_t,typename integer_t>
  CSRMatrixMPI<scalar_t,integer_t>::CSRMatrixMPI
  (const CSRMatrixMPI<scalar_t,integer_t>& A)
    : CompressedSparseMatrix<scalar_t,integer_t>() {
    this->_n = A._n;
    this->_nnz = A._nnz;
    this->_symm_sparse = A._symm_sparse;
    _comm = A._comm;
    _dist = A._dist;
    _local_rows = A._local_rows;
    _local_nnz = A._local_nnz;
    _begin_row = A._begin_row;
    _end_row = A._end_row;
    this->_ptr = new integer_t[_local_rows+1];
    this->_ind = new integer_t[_local_nnz];
    this->_val = new scalar_t[_local_nnz];
    std::copy(A._ptr, A._ptr+_local_rows+1, this->_ptr);
    std::copy(A._ind, A._ind+_local_nnz, this->_ind);
    std::copy(A._val, A._val+_local_nnz, this->_val);
    split_diag_offdiag();
    check();
  }

  /**
   * Create a distributed CSR matrix from a serial one: every process
   * picks his part.  Collective on all processes in c.  If
   * !only_at_root: all need to pass a copy of A, else only the
   * rank==0 has to pass A, the others pass NULL.
   */
  template<typename scalar_t,typename integer_t>
  CSRMatrixMPI<scalar_t,integer_t>::CSRMatrixMPI
  (const CSRMatrix<scalar_t,integer_t>* A,
   MPI_Comm c, bool only_at_root) {
    if (A) {
      this->_n = A->size();
      this->_nnz = A->nnz();
      this->_symm_sparse = A->symm_sparse();
    }
    if (only_at_root) {
      MPI_Bcast(&this->_n, 1, mpi_type<integer_t>(), 0, c);
      MPI_Bcast(&this->_nnz, 1, mpi_type<integer_t>(), 0, c);
      MPI_Bcast(&this->_symm_sparse, sizeof(bool), MPI_BYTE, 0, c);
    }
    auto rank = mpi_rank(c);
    auto P = mpi_nprocs(c);
    _dist.resize(P+1);
    _comm = c;
    if (!only_at_root || (only_at_root && rank==0)) {
      // divide rows over processes, try to give equal number of nnz
      // to each process
      _dist[0] = 0;
      for (int p=1; p<P; p++) {
        integer_t t = p * float(A->nnz()) / P;
        auto hi = std::distance
          (A->get_ptr(), std::upper_bound
           (A->get_ptr()+_dist[p-1], A->get_ptr()+A->size(), t));
        _dist[p] = ((hi-1 >= _dist[p-1]) &&
                    (t-A->get_ptr()[hi-1] < A->get_ptr()[hi]-t)) ? hi-1 : hi;
      }
      _dist[P] = this->_n;
    }
    if (only_at_root)
      MPI_Bcast(_dist.data(), _dist.size(), mpi_type<integer_t>(), 0, c);
    _begin_row = _dist[rank];
    _end_row = _dist[rank+1];
    _local_rows = _end_row - _begin_row;
    if (!only_at_root) {
      _local_nnz = A->get_ptr()[_end_row] - A->get_ptr()[_begin_row];
      this->_ptr = new integer_t[_local_rows+1];
      this->_ind = new integer_t[_local_nnz];
      this->_val = new scalar_t[_local_nnz];
      auto i0 = A->get_ptr()[_begin_row];
      auto i1 = A->get_ptr()[_end_row];
      std::copy(A->get_ptr() + _begin_row,
                A->get_ptr() + _end_row + 1, this->_ptr);
      std::copy(A->get_ind() + i0, A->get_ind() + i1, this->_ind);
      std::copy(A->get_val() + i0, A->get_val() + i1, this->_val);
    } else {
      auto scnts = new int[2*P];
      auto sdisp = scnts + P;
      if (rank == 0)
        for (int p=0; p<P; p++)
          scnts[p] = A->get_ptr()[_dist[p+1]] - A->get_ptr()[_dist[p]];
      int loc_nnz;
      MPI_Scatter
        (scnts, 1, mpi_type<int>(), &loc_nnz,
         1, mpi_type<int>(), 0, c);
      _local_nnz = loc_nnz;
      this->_ptr = new integer_t[_local_rows+1];
      this->_ind = new integer_t[_local_nnz];
      this->_val = new scalar_t[_local_nnz];
      if (rank == 0)
        for (int p=0; p<P; p++) {
          scnts[p] = _dist[p+1] - _dist[p] + 1;
          sdisp[p] = _dist[p];
        }
      MPI_Scatterv
        (rank ? NULL : A->get_ptr(), scnts, sdisp,
         mpi_type<integer_t>(), this->_ptr, _local_rows+1,
         mpi_type<integer_t>(), 0, c);
      if (rank == 0)
        for (int p=0; p<P; p++) {
          scnts[p] = A->get_ptr()[_dist[p+1]] - A->get_ptr()[_dist[p]];
          sdisp[p] = A->get_ptr()[_dist[p]];
        }
      MPI_Scatterv
        (rank ? NULL : A->get_ind(), scnts, sdisp,
         mpi_type<integer_t>(), this->_ind, _local_nnz,
         mpi_type<integer_t>(), 0, c);
      MPI_Scatterv
        (rank ? NULL : A->get_val(), scnts, sdisp,
         mpi_type<scalar_t>(),  this->_val, _local_nnz,
         mpi_type<scalar_t>(), 0, c);
      delete[] scnts;
    }
    for (integer_t r=_local_rows; r>=0; r--)
      this->_ptr[r] -= this->_ptr[0];
    split_diag_offdiag();
    check();
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::print() const {
    for (int p=0; p<mpi_rank(_comm); p++)
      MPI_Barrier(_comm);
    if (mpi_rank(_comm)==0) {
      std::cout << "dist=[";
      for (auto d : _dist) std::cout << d << " ";
      std::cout << "];" << std::endl;
    }
    std::cout << "rank=" << mpi_rank(_comm) << "\nptr=[";
    for (integer_t i=0; i<=local_rows(); i++)
      std::cout << this->_ptr[i] << " ";
    std::cout << "];" << std::endl;
    std::cout << "ind=[";
    for (integer_t i=0; i<local_rows(); i++) {
      for (integer_t j=this->_ptr[i]; j<_offdiag_start[i]; j++)
        std::cout << this->_ind[j] << " ";
      std::cout << "| ";
      for (integer_t j=_offdiag_start[i]; j<this->_ptr[i+1]; j++)
        std::cout << this->_ind[j] << " ";
      std::cout << ", ";
    }
    std::cout << "];" << std::endl << std::flush;
    std::cout << "val=[";
    for (integer_t i=0; i<local_rows(); i++) {
      for (integer_t j=this->_ptr[i]; j<_offdiag_start[i]; j++)
        std::cout << this->_val[j] << " ";
      std::cout << "| ";
      for (integer_t j=_offdiag_start[i]; j<this->_ptr[i+1]; j++)
        std::cout << this->_val[j] << " ";
      std::cout << ", ";
    }
    std::cout << "];" << std::endl << std::flush;
    for (int p=mpi_rank(_comm); p<=mpi_nprocs(_comm); p++)
      MPI_Barrier(_comm);
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::print_dense
  (const std::string& name) const {
    auto Aseq = gather();
    if (Aseq) Aseq->print_dense(name);
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::print_MM
  (const std::string& filename) const {
    auto Aseq = gather();
    if (Aseq) Aseq->print_MM(filename);
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::check() const {
#if !defined(NDEBUG)
    auto rank = mpi_rank(_comm);
    assert(_local_rows >= 0);
    assert(_end_row - _begin_row == _local_rows);
    integer_t total_rows = _local_rows;
    MPI_Allreduce(MPI_IN_PLACE, &total_rows, 1,
                  mpi_type<integer_t>(), MPI_SUM, _comm);
    assert(total_rows == this->_n);
    assert(_local_nnz == this->_ptr[_local_rows]);
    integer_t total_nnz = _local_nnz;
    MPI_Allreduce(MPI_IN_PLACE, &total_nnz, 1,
                  mpi_type<integer_t>(), MPI_SUM, _comm);
    assert(total_nnz == this->_nnz);
    assert(_end_row >= _begin_row);
    if (rank == mpi_nprocs(_comm)-1) {
      assert(_end_row == this->_n);
    }
    if (rank == 0) { assert(_begin_row == 0); }
    assert(_begin_row == _dist[rank]);
    assert(_end_row == _dist[rank+1]);
    assert(this->_ptr[0] == 0);
    for (integer_t r=1; r<=_local_rows; r++) {
      assert(this->_ptr[r] >= this->_ptr[r-1]);
    }
    for (integer_t r=0; r<_local_rows; r++) {
      assert(_offdiag_start[r] >= this->_ptr[r]);
      assert(this->_ptr[r+1] >= _offdiag_start[r]);
    }
    for (integer_t r=0; r<_local_rows; r++) {
      for (integer_t j=this->_ptr[r]; j<this->_ptr[r+1]; j++) {
        assert(this->_ind[j] >= 0);
        assert(this->_ind[j] < this->_n);
      }
    }
    for (integer_t r=0; r<_local_rows; r++) {
      for (integer_t j=this->_ptr[r]; j<_offdiag_start[r]; j++) {
        assert(this->_ind[j] >= _begin_row);
        assert(this->_ind[j] < _end_row);
      }
      for (integer_t j=_offdiag_start[r]; j<this->_ptr[r+1]; j++) {
        assert(this->_ind[j] < _begin_row || this->_ind[j] >= _end_row);
      }
    }
#endif
  }

  /**
   * Extract part [graph_begin, graph_end) from this sparse matrix,
   * after applying the symmetric permutation perm/iperm.
   */
  // TODO move this to CSRGraph, have it return a unique_ptr or value!
  template<typename scalar_t,typename integer_t> CSRGraph<integer_t>*
  CSRMatrixMPI<scalar_t,integer_t>::get_sub_graph
  (const integer_t* perm,
   const std::pair<integer_t,integer_t>* graph_ranges) const {
    auto rank = mpi_rank(_comm);
    auto P = mpi_nprocs(_comm);
    auto scnts = new int[4*P+local_rows()];
    auto rcnts = scnts + P;
    auto sdispls = rcnts + P;
    auto rdispls = sdispls + P;
    auto dest = rdispls + P;
    std::fill(scnts, scnts+P, 0);
    std::fill(dest, dest+local_rows(), -1);
    for (integer_t row=0; row<local_rows(); row++) {
      auto perm_row = perm[row+begin_row()];
      for (int p=0; p<P; p++)
        if (graph_ranges[p].first <= perm_row &&
            perm_row < graph_ranges[p].second) {
          dest[row] = p;
          scnts[p] += 2 + this->_ptr[row+1]-this->_ptr[row];
          break;
        }
    }
    MPI_Alltoall
      (scnts, 1, mpi_type<int>(), rcnts, 1, mpi_type<int>(), _comm);
    auto sbuf = new integer_t[std::accumulate(scnts, scnts+P, integer_t(0))];
    auto pp = new integer_t*[P];
    sdispls[0] = 0;
    rdispls[0] = 0;
    pp[0] = sbuf;
    for (int p=1; p<P; p++) {
      sdispls[p] = sdispls[p-1] + scnts[p-1];
      rdispls[p] = rdispls[p-1] + rcnts[p-1];
      pp[p] = sbuf + sdispls[p];
    }
    for (integer_t row=0; row<local_rows(); row++) {
      auto d = dest[row];
      if (d == -1) continue;
      // send the number of the permuted row (vertex)
      *pp[d] = perm[row+begin_row()];  pp[d]++;
      // send the number of edges for this vertex
      *pp[d] = this->_ptr[row+1] - this->_ptr[row];  pp[d]++;
      for (auto j=this->_ptr[row]; j<this->_ptr[row+1]; j++) {
        // send the actual edges
        *pp[d] = perm[this->_ind[j]];  pp[d]++;
      }
    }
    delete[] pp;
    auto rsize = std::accumulate(rcnts, rcnts+P, size_t(0));
    auto rbuf = new integer_t[rsize];
    MPI_Alltoallv(sbuf, scnts, sdispls, mpi_type<integer_t>(),
                  rbuf, rcnts, rdispls, mpi_type<integer_t>(), _comm);
    delete[] sbuf;
    delete[] scnts;

    auto n_vert = graph_ranges[rank].second - graph_ranges[rank].first;
    auto edge_count = new integer_t[n_vert];
    integer_t n_edges = 0;
    size_t prbuf = 0;
    while (prbuf < rsize) {
      auto my_row = rbuf[prbuf] - graph_ranges[rank].first;
      edge_count[my_row] = rbuf[prbuf+1];
      n_edges += rbuf[prbuf+1];
      prbuf += 2 + rbuf[prbuf+1];
    }
    auto g = new CSRGraph<integer_t>(n_vert, n_edges);
    g->get_ptr()[0] = 0;
    for (integer_t i=1; i<=n_vert; i++)
      g->get_ptr()[i] = g->get_ptr()[i-1] + edge_count[i-1];
    delete[] edge_count;
    prbuf = 0;
    while (prbuf < rsize) {
      auto my_row = rbuf[prbuf] - graph_ranges[rank].first;
      std::copy(rbuf+prbuf+2, rbuf+prbuf+2+rbuf[prbuf+1],
                g->get_ind()+g->get_ptr()[my_row]);
      prbuf += 2 + rbuf[prbuf+1];
    }
    delete[] rbuf;
    return g;
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::split_diag_offdiag() {
    _offdiag_start.resize(local_rows());
    auto is_diag = [this](const integer_t& e){
      return e >= begin_row() && e < end_row();
    };
    // partition in diagonal and off-diagonal blocks
#pragma omp parallel for
    for (integer_t row=0; row<local_rows(); row++) {
      // like std::partition but on ind and val arrays simultaneously
      auto lo = this->_ptr[row];
      auto hi = this->_ptr[row+1];
      auto first_ind = &this->_ind[lo];
      auto last_ind = &this->_ind[hi];
      auto first_val = &this->_val[lo];
      auto last_val = &this->_val[hi];
      while (1) {
        while ((first_ind != last_ind) && is_diag(*first_ind)) {
          ++first_ind; ++first_val;
        }
        if (first_ind == last_ind) { last_val--; last_ind--; break; }
        last_val--; last_ind--;
        while ((first_ind != last_ind) && !is_diag(*last_ind)) {
          --last_ind; --last_val;
        }
        if (first_ind == last_ind) break;
        std::iter_swap(first_ind++, last_ind);
        std::iter_swap(first_val++, last_val);
      }
      _offdiag_start[row] = lo + std::distance(&this->_ind[lo], first_ind);
    }
  }

  // figure out what to send/receive from/to who during spmv
  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::setup_spmv_buffers() const {
    _spmv_buffers = std::unique_ptr<SPMVBuffers<scalar_t,integer_t>>
      (new SPMVBuffers<scalar_t,integer_t>());
    //  _spmv_setup = true;
    integer_t nr_offdiag_nnz = 0;
#pragma omp parallel for reduction(+:nr_offdiag_nnz)
    for (integer_t r=0; r<local_rows(); r++)
      nr_offdiag_nnz += this->_ptr[r+1] - _offdiag_start[r];

    auto P = mpi_nprocs(_comm);
    auto rsizes = new int[2*P]();
    auto ssizes = rsizes+P;
    std::vector<integer_t> spmv_rind;
    spmv_rind.reserve(nr_offdiag_nnz);
    for (integer_t r=0; r<local_rows(); r++)
      for (auto j=_offdiag_start[r]; j<this->_ptr[r+1]; j++)
        spmv_rind.push_back(this->_ind[j]);
    std::sort(spmv_rind.begin(), spmv_rind.end());
    spmv_rind.erase
      (std::unique(spmv_rind.begin(), spmv_rind.end()), spmv_rind.end());

    _spmv_buffers->_spmv_prbuf.reserve(nr_offdiag_nnz);
    for (integer_t r=0; r<_local_rows; r++)
      for (integer_t j=_offdiag_start[r]; j<this->_ptr[r+1]; j++)
        _spmv_buffers->_spmv_prbuf.push_back
          (std::distance
           (spmv_rind.begin(), std::lower_bound
            (spmv_rind.begin(), spmv_rind.end(), this->_ind[j])));

    // how much to receive from each proc
    for (size_t p=0, j=0; p<size_t(P); p++)
      while (j < spmv_rind.size() && spmv_rind[j] < _dist[p+1]) {
        j++; rsizes[p]++;
      }
    MPI_Alltoall
      (rsizes, 1, mpi_type<int>(), ssizes, 1, mpi_type<int>(), _comm);

    auto nr_recv_procs = std::count_if
                 (rsizes, rsizes+P, [](int s){ return s > 0;});
    auto nr_send_procs =
      std::count_if(ssizes, ssizes+P, [](int s){ return s > 0;});
    _spmv_buffers->_spmv_sranks.reserve(nr_send_procs);
    _spmv_buffers->_spmv_soff.reserve(nr_send_procs+1);
    _spmv_buffers->_spmv_rranks.reserve(nr_recv_procs);
    _spmv_buffers->_spmv_roffs.reserve(nr_recv_procs+1);
    int oset_recv = 0, oset_send = 0;
    for (int p=0; p<P; p++) {
      if (ssizes[p] > 0) {
        _spmv_buffers->_spmv_sranks.push_back(p);
        _spmv_buffers->_spmv_soff.push_back(oset_send);
        oset_send += ssizes[p];
      }
      if (rsizes[p] > 0) {
        _spmv_buffers->_spmv_rranks.push_back(p);
        _spmv_buffers->_spmv_roffs.push_back(oset_recv);
        oset_recv += rsizes[p];
      }
    }
    delete[] rsizes;
    _spmv_buffers->_spmv_soff.push_back(oset_send);
    _spmv_buffers->_spmv_roffs.push_back(oset_recv);
    _spmv_buffers->_spmv_sind.resize(oset_send);

    std::vector<MPI_Request> req(nr_recv_procs + nr_send_procs);
    for (int p=0; p<nr_recv_procs; p++)
      MPI_Isend
        (spmv_rind.data() + _spmv_buffers->_spmv_roffs[p],
         _spmv_buffers->_spmv_roffs[p+1] - _spmv_buffers->_spmv_roffs[p],
         mpi_type<integer_t>(), _spmv_buffers->_spmv_rranks[p],
         0, _comm, &req[p]);
    for (int p=0; p<nr_send_procs; p++)
      MPI_Irecv
        (_spmv_buffers->_spmv_sind.data() + _spmv_buffers->_spmv_soff[p],
         _spmv_buffers->_spmv_soff[p+1] - _spmv_buffers->_spmv_soff[p],
         mpi_type<integer_t>(), _spmv_buffers->_spmv_sranks[p],
         0, _comm, &req[nr_recv_procs+p]);
    MPI_Waitall(req.size(), req.data(), MPI_STATUSES_IGNORE);

    _spmv_buffers->_spmv_rbuf.resize(spmv_rind.size());
    _spmv_buffers->_spmv_sbuf.resize(_spmv_buffers->_spmv_sind.size());
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::spmv
  (const scalar_t* x, scalar_t* y) const {
    omp_spmv(x, y);
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::spmv
  (const DenseM_t& x, DenseM_t& y) const {
    omp_spmv(x, y);
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::omp_spmv
  (const DenseM_t& x, DenseM_t& y) const {
    assert(x.cols() == y.cols());
    assert(x.rows() == std::size_t(this->local_rows()));
    assert(y.rows() == std::size_t(this->local_rows()));
    for (std::size_t c=0; c<x.cols(); c++)
      omp_spmv(x.ptr(0,c), y.ptr(0,c));
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::omp_spmv
  (const scalar_t* x, scalar_t* y) const {
    if (!_spmv_buffers) setup_spmv_buffers();

    for (size_t i=0; i<_spmv_buffers->_spmv_sind.size(); i++)
      _spmv_buffers->_spmv_sbuf[i] =
        x[_spmv_buffers->_spmv_sind[i]-_begin_row];

    MPI_Request* sreq = new MPI_Request
      [_spmv_buffers->_spmv_sranks.size() +
       _spmv_buffers->_spmv_rranks.size()];
    MPI_Request* rreq = sreq + _spmv_buffers->_spmv_sranks.size();
    for (size_t p=0; p<_spmv_buffers->_spmv_sranks.size(); p++)
      MPI_Isend
        (_spmv_buffers->_spmv_sbuf.data() + _spmv_buffers->_spmv_soff[p],
         _spmv_buffers->_spmv_soff[p+1] - _spmv_buffers->_spmv_soff[p],
         mpi_type<scalar_t>(), _spmv_buffers->_spmv_sranks[p],
         0, _comm, &sreq[p]);

    for (size_t p=0; p<_spmv_buffers->_spmv_rranks.size(); p++)
      MPI_Irecv
        (_spmv_buffers->_spmv_rbuf.data() + _spmv_buffers->_spmv_roffs[p],
         _spmv_buffers->_spmv_roffs[p+1] - _spmv_buffers->_spmv_roffs[p],
         mpi_type<scalar_t>(), _spmv_buffers->_spmv_rranks[p],
         0, _comm, &rreq[p]);

    // first do the block diagonal part, while the communication is going on
#pragma omp parallel for
    for (integer_t r=0; r<local_rows(); r++) {
      auto yrow = scalar_t(0.);
      for (auto j=this->_ptr[r]; j<_offdiag_start[r]; j++)
        yrow += this->_val[j] * x[this->_ind[j] - begin_row()];
      y[r] = yrow;
    }
    // wait for incoming messages
    MPI_Waitall
      (_spmv_buffers->_spmv_rranks.size(), rreq, MPI_STATUSES_IGNORE);

    // do the block off-diagonal part of the matrix
    // TODO some openmp here
    auto pbuf = _spmv_buffers->_spmv_prbuf.begin();
    for (integer_t r=0; r<local_rows(); r++)
      for (integer_t j=_offdiag_start[r]; j<this->_ptr[r+1]; j++)
        y[r] += this->_val[j] * _spmv_buffers->_spmv_rbuf[*pbuf++];

    // wait for all send messages to finish
    MPI_Waitall
      (_spmv_buffers->_spmv_sranks.size(), sreq, MPI_STATUSES_IGNORE);
    delete[] sreq;
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::permute
  (const integer_t* iorder, const integer_t* order) {
    // This is called, but it does not actually permute the matrix.
    // Instead, we permute the right hand side and solution vectors
  }

  template<typename scalar_t,typename integer_t>
  std::unique_ptr<CSRMatrix<scalar_t,integer_t>>
  CSRMatrixMPI<scalar_t,integer_t>::gather() const {
    auto rank = mpi_rank(_comm);
    auto P = mpi_nprocs(_comm);
    std::unique_ptr<CSRMatrix<scalar_t,integer_t>> Aseq;
    int* rcnts = NULL;
    int* displs = NULL;
    if (rank==0) {
      Aseq = std::unique_ptr<CSRMatrix<scalar_t,integer_t>>
        (new CSRMatrix<scalar_t,integer_t>(this->size(), this->nnz()));
      rcnts = new int[2*P];
      displs = rcnts + P;
      for (int p=0; p<P; p++) {
        rcnts[p] = _dist[p+1]-_dist[p];
        displs[p] = _dist[p]+1;
      }
    }
    MPI_Gatherv
      (this->_ptr+1, _local_rows, mpi_type<integer_t>(),
       rank ? NULL : Aseq->get_ptr(), rcnts, displs,
       mpi_type<integer_t>(), 0, _comm);
    if (rank==0) {
      Aseq->get_ptr()[0] = 0;
      for (int p=1; p<P; p++) {
        if (_dist[p] > 0) {
          integer_t p_start = Aseq->get_ptr()[_dist[p]];
          for (int r=_dist[p]; r<_dist[p+1]; r++)
            Aseq->get_ptr()[r+1] += p_start;
        }
      }
      for (int p=0; p<P; p++) {
        rcnts[p] = Aseq->get_ptr()[_dist[p+1]]-Aseq->get_ptr()[_dist[p]];
        displs[p] = Aseq->get_ptr()[_dist[p]];
      }
    }
    MPI_Gatherv
      (this->_ind, _local_nnz, mpi_type<integer_t>(),
       rank ? NULL : Aseq->get_ind(), rcnts, displs,
       mpi_type<integer_t>(), 0, _comm);
    MPI_Gatherv
      (this->_val, _local_nnz, mpi_type<scalar_t>(),
       rank ? NULL : Aseq->get_val(), rcnts, displs,
       mpi_type<scalar_t>(), 0, _comm);
    delete[] rcnts;
    return Aseq;
  }

  /**
   * This gathers the matrix to 1 process, then applies MC64
   * sequentially.
   *
   * On output, the perm vector contains the GLOBAL column
   * permutation, such that the column perm[j] of the original matrix
   * is column j in the permuted matrix.
   *
   * Dr and Dc contain the LOCAL scaling vectors.
   */
  template<typename scalar_t,typename integer_t> int
  CSRMatrixMPI<scalar_t,integer_t>::permute_and_scale
  (int job, std::vector<integer_t>& perm, std::vector<scalar_t>& Dr,
   std::vector<scalar_t>& Dc, bool apply) {
    if (job == 0) return 0;
    if (job > 5 || job < 0 || job == 1) {
      if (mpi_rank()==0)
        std::cout << "# WARNING mc64 job " << job
                  << " not supported, I'm not doing any column permutation"
                  << " or matrix scaling!" << std::endl;
      return 1;
    }
    auto Aseq = gather();
    std::vector<scalar_t> Dr_global;
    std::vector<scalar_t> Dc_global;
    int ierr = 0;
    if (Aseq) {
      ierr = Aseq->permute_and_scale
        (job, perm, Dr_global, Dc_global,
         false/*do not apply perm/scaling*/);
      Aseq.reset(nullptr);
    } else {
      perm.resize(this->size());
      if (job == 5) Dc_global.resize(this->size());
    }
    MPI_Bcast(&ierr, 1, MPI_INT, 0, _comm);
    if (ierr) return ierr;
    MPI_Bcast(perm.data(), perm.size(), mpi_type<integer_t>(), 0, _comm);
    if (job == 5) {
      auto P = mpi_nprocs(_comm);
      auto rank = mpi_rank(_comm);
      auto scnts = new int[2*P];
      auto sdispls = scnts+P;
      for (int p=0; p<P; p++) {
        scnts[p] = _dist[p+1]-_dist[p];
        sdispls[p] = _dist[p];
      }
      Dr.resize(_local_rows);
      MPI_Scatterv
        (rank ? NULL : Dr_global.data(), scnts, sdispls, mpi_type<scalar_t>(),
         Dr.data(), _local_rows, mpi_type<scalar_t>(), 0, _comm);
      delete[] scnts;
      Dr_global.clear();
      MPI_Bcast(Dc_global.data(), Dc_global.size(),
                mpi_type<scalar_t>(), 0, _comm);
      apply_scaling(Dr, Dc_global);
      Dc.resize(_local_rows);
      std::copy
        (Dc_global.data()+_begin_row, Dc_global.data()+_end_row, Dc.data());
      Dc_global.clear();
    }
    apply_column_permutation(perm);
    this->_symm_sparse = false;
    return 0;
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::apply_column_permutation
  (const std::vector<integer_t>& perm) {
    integer_t* iperm = new integer_t[this->size()];
    for (integer_t i=0; i<this->size(); i++) iperm[perm[i]] = i;
#pragma omp parallel for
    for (integer_t r=0; r<this->_local_rows; r++)
      for (integer_t j=this->_ptr[r]; j<this->_ptr[r+1]; j++)
        this->_ind[j] = iperm[this->_ind[j]];
    delete[] iperm;
    split_diag_offdiag();
  }

  // Apply row and column scaling. Dr is LOCAL, Dc is global!
  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::apply_scaling
  (const std::vector<scalar_t>& Dr, const std::vector<scalar_t>& Dc) {
#pragma omp parallel for
    for (integer_t r=0; r<_local_rows; r++)
      for (integer_t j=this->_ptr[r]; j<this->_ptr[r+1]; j++)
        this->_val[j] = this->_val[j] * Dr[r] * Dc[this->_ind[j]];
    STRUMPACK_FLOPS((is_complex<scalar_t>()?6:1)*
                    static_cast<long long int>(2.*double(_local_nnz)));
  }

  // Symmetrize the sparsity pattern.
  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::symmetrize_sparsity() {
    if (this->_symm_sparse) return;
    auto P = mpi_nprocs(_comm);
    struct Idxij { integer_t i; integer_t j; };
    std::vector<std::vector<Idxij>> sbuf(P);
    for (integer_t r=0; r<local_rows(); r++)
      for (integer_t j=_offdiag_start[r]; j<this->_ptr[r+1]; j++) {
        auto col = this->_ind[j];
        auto row = r+_begin_row;
        auto dest = std::upper_bound
          (_dist.begin(), _dist.end(), col) - _dist.begin() - 1;
        sbuf[dest].emplace_back(Idxij{col, row});
      }
    auto ssizes = new int[4*P];
    auto rsizes = ssizes + P;
    auto sdispl = ssizes + 2*P;
    auto rdispl = ssizes + 3*P;
    auto idxij_byte_size = sizeof(Idxij);
    for (int p=0; p<P; p++) ssizes[p] = sbuf[p].size()*idxij_byte_size;
    MPI_Alltoall
      (ssizes, 1, mpi_type<int>(), rsizes, 1, mpi_type<int>(), _comm);
    auto totssize = std::accumulate(ssizes, ssizes+P, 0);
    auto totrsize = std::accumulate(rsizes, rsizes+P, 0);
    auto sendbuf = new Idxij[totssize/idxij_byte_size];
    sdispl[0] = rdispl[0] = 0;
    for (int p=1; p<P; p++) {
      sdispl[p] = sdispl[p-1] + ssizes[p-1];
      rdispl[p] = rdispl[p-1] + rsizes[p-1];
    }
    for (int p=0; p<P; p++)
      std::copy(sbuf[p].begin(), sbuf[p].end(),
                sendbuf+sdispl[p]/idxij_byte_size);
    sbuf.clear();
    std::vector<Idxij> edges(totrsize/idxij_byte_size);
    MPI_Alltoallv(sendbuf, ssizes, sdispl, MPI_BYTE, edges.data(),
                  rsizes, rdispl, MPI_BYTE, _comm);
    delete[] ssizes;
    delete[] sendbuf;

    std::sort(edges.begin(), edges.end(), [](const Idxij& a, const Idxij& b) {
        // sort according to rows, then columns
        if (a.i != b.i) return (a.i < b.i);
        return (a.j < b.j);
      });
    // count how many of the received values are not already here
    auto row_sums = new integer_t[local_rows()];
    for (integer_t r=0; r<local_rows(); r++)
      row_sums[r] = this->_ptr[r+1]-this->_ptr[r];
    auto new_nnz = _local_nnz;
    auto ep = edges.begin();
    for (integer_t r=0; r<local_rows(); r++) {
      while (ep != edges.end() && ep->i < r+_begin_row) ep++;
      if (ep == edges.end()) break;
      while (ep != edges.end() && ep->i == r+_begin_row) {
        integer_t kb = _offdiag_start[r], ke = this->_ptr[r+1];
        if (std::find(this->_ind+kb, this->_ind+ke, ep->j) == this->_ind+ke) {
          new_nnz++;
          row_sums[r]++;
        }
        ep++;
      }
    }
    // same for the diagonal block
    for (integer_t r=0; r<local_rows(); r++)
      for (integer_t j=this->_ptr[r]; j<_offdiag_start[r]; j++) {
        auto lc = this->_ind[j]-_begin_row;
        integer_t kb = this->_ptr[lc], ke = _offdiag_start[lc];
        if (std::find(this->_ind+kb, this->_ind+ke, r+_begin_row)
            == this->_ind+ke) {
          row_sums[lc]++;
          new_nnz++;
        }
      }
    if (new_nnz != _local_nnz) {
      _local_nnz = new_nnz;
      // allocate new arrays
      auto new_ptr = new integer_t[local_rows()+1];
      new_ptr[0] = 0;
      for (integer_t r=0; r<local_rows(); r++)
        new_ptr[r+1] = new_ptr[r] + row_sums[r];
      auto new_ind = new integer_t[new_nnz];
      auto new_val = new scalar_t[new_nnz];
      // copy old nonzeros to new arrays
      for (integer_t r=0; r<local_rows(); r++) {
        row_sums[r] = new_ptr[r] + this->_ptr[r+1] - this->_ptr[r];
        for (integer_t j=this->_ptr[r], k=new_ptr[r];
             j<this->_ptr[r+1]; j++) {
          new_ind[k  ] = this->_ind[j];
          new_val[k++] = this->_val[j];
        }
      }
      // diagonal block
      for (integer_t r=0; r<local_rows(); r++)
        for (integer_t j=this->_ptr[r]; j<_offdiag_start[r]; j++) {
          auto lc = this->_ind[j]-_begin_row;
          integer_t kb = this->_ptr[lc], ke = _offdiag_start[lc];
          if (std::find(this->_ind+kb, this->_ind+ke, r+_begin_row)
              == this->_ind+ke) {
            new_ind[row_sums[lc]] = r+_begin_row;
            new_val[row_sums[lc]] = scalar_t(0.);
            row_sums[lc]++;
          }
        }
      // off-diagonal entries
      ep = edges.begin();
      for (integer_t r=0; r<local_rows(); r++) {
        while (ep != edges.end() && ep->i < r+_begin_row) ep++;
        if (ep == edges.end()) break;
        while (ep != edges.end() && ep->i == r+_begin_row) {
          integer_t kb = _offdiag_start[r], ke = this->_ptr[r+1];
          if (std::find(this->_ind+kb, this->_ind+ke, ep->j)
              == this->_ind+ke) {
            new_ind[row_sums[r]] = ep->j;
            new_val[row_sums[r]] = scalar_t(0.);
            row_sums[r]++;
          }
          ep++;
        }
      }
      this->set_ptr(new_ptr);
      this->set_ind(new_ind);
      this->set_val(new_val);
    }
    delete[] row_sums;

    integer_t total_new_nnz = 0;
    MPI_Allreduce(&new_nnz, &total_new_nnz, 1,
                  mpi_type<integer_t>(), MPI_SUM, _comm);
    if (total_new_nnz != this->nnz()) {
      split_diag_offdiag();
      this->_nnz = total_new_nnz;
    }
    this->_symm_sparse = true;
  }

  template<typename scalar_t,typename integer_t> int
  CSRMatrixMPI<scalar_t,integer_t>::read_matrix_market
  (const std::string& filename) {
    std::cout << "ERROR: reading a distributed matrix"
              << " from file is not supported." << std::endl;
    abort();
    // TODO: first read the matrix as a sequential matrix
    // then make a Distributed matrix from the sequential one
    // Aydin has code for MPI-IO input of matrix-market file
    return 0;
  }


  template<typename scalar_t,typename integer_t>
  typename RealType<scalar_t>::value_type
  CSRMatrixMPI<scalar_t,integer_t>::max_scaled_residual
  (const DenseM_t& x, const DenseM_t& b) const {
    real_t res(0.);
    for (std::size_t c=0; c<x.cols(); c++)
      res = std::max(res, max_scaled_residual(x.ptr(0,c), b.ptr(0,c)));
    return res;
  }

  template<typename scalar_t,typename integer_t>
  typename RealType<scalar_t>::value_type
  CSRMatrixMPI<scalar_t,integer_t>::max_scaled_residual
  (const scalar_t* x, const scalar_t* b) const {
    if (!_spmv_buffers) setup_spmv_buffers();

    for (size_t i=0; i<_spmv_buffers->_spmv_sind.size(); i++)
      _spmv_buffers->_spmv_sbuf[i] =
        x[_spmv_buffers->_spmv_sind[i]-_begin_row];

    std::vector<MPI_Request> sreq(_spmv_buffers->_spmv_sranks.size());
    for (size_t p=0; p<_spmv_buffers->_spmv_sranks.size(); p++)
      MPI_Isend
        (_spmv_buffers->_spmv_sbuf.data() + _spmv_buffers->_spmv_soff[p],
         _spmv_buffers->_spmv_soff[p+1] - _spmv_buffers->_spmv_soff[p],
         mpi_type<scalar_t>(), _spmv_buffers->_spmv_sranks[p],
         0, _comm, &sreq[p]);

    std::vector<MPI_Request> rreq(_spmv_buffers->_spmv_rranks.size());
    for (size_t p=0; p<_spmv_buffers->_spmv_rranks.size(); p++)
      MPI_Irecv
        (_spmv_buffers->_spmv_rbuf.data() + _spmv_buffers->_spmv_roffs[p],
         _spmv_buffers->_spmv_roffs[p+1] - _spmv_buffers->_spmv_roffs[p],
         mpi_type<scalar_t>(), _spmv_buffers->_spmv_rranks[p],
         0, _comm, &rreq[p]);

    MPI_Waitall(rreq.size(), rreq.data(), MPI_STATUSES_IGNORE);

    real_t m = real_t(0.);
    auto pbuf = _spmv_buffers->_spmv_prbuf.begin();
    //pragma omp parallel for reduction(max:m)
    for (integer_t r=0; r<_local_rows; r++) {
      auto true_res = b[r];
      auto abs_res = std::abs(b[r]);
      for (auto j=this->_ptr[r]; j<_offdiag_start[r]; j++) {
        auto c = this->_ind[j];
        true_res -= this->_val[j] * x[c-_begin_row];
        abs_res += std::abs(this->_val[j]) * std::abs(x[c-_begin_row]);
      }
      for (auto j=_offdiag_start[r]; j<this->_ptr[r+1]; j++) {
        true_res -= this->_val[j] * _spmv_buffers->_spmv_rbuf[*pbuf];
        abs_res += std::abs(this->_val[j]) *
          std::abs(_spmv_buffers->_spmv_rbuf[*pbuf]);
        pbuf++;
      }
      m = std::max(m, std::abs(true_res) / std::abs(abs_res));
    }
    // wait for all send messages to finish
    MPI_Waitall(sreq.size(), sreq.data(), MPI_STATUSES_IGNORE);
    MPI_Allreduce(MPI_IN_PLACE, &m, 1, mpi_type<real_t>(), MPI_MAX, _comm);
    return m;
  }

} // end namespace strumpack

#endif //CSRMATRIXMPI_HPP
