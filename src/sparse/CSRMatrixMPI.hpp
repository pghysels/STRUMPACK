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
#if defined(STRUMPACK_USE_COMBBLAS)
#include "AWPMCombBLAS.hpp"
#endif

namespace strumpack {

  template<typename scalar_t,typename integer_t> class SPMVBuffers {
  public:
    bool initialized = false;
    std::vector<integer_t> sranks;
    // ranks of the process from which I need to receive something
    std::vector<integer_t> rranks;
    std::vector<integer_t> soff;
    std::vector<integer_t> roffs;
    // indices to receive from each rank from which I need to receive
    std::vector<integer_t> sind;
    // indices to receive from each rank from which I need to receive
    std::vector<scalar_t> sbuf;
    std::vector<scalar_t> rbuf;
    // for each off-diagonal entry spmv_prbuf stores the
    // corresponding index in the receive buffer
    std::vector<integer_t> prbuf;
  };


  /**
   * \class CSRMatrixMPI
   * \brief Block-row distributed compressed sparse row storage.
   *
   * TODO: cleanup this class
   *  - use MPIComm
   *  - store the block diagonal as a CSRMatrix
   *  - ...
   */
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
    CSRMatrixMPI
    (const CSRMatrix<scalar_t,integer_t>* A, MPI_Comm c, bool only_at_root);

    inline const std::vector<integer_t>& dist() const { return dist_; }
    inline const integer_t& dist(std::size_t p) const { assert(p < dist_.size()); return dist_[p]; }
    inline integer_t local_rows() const { return end_row_ - begin_row_; }
    inline integer_t begin_row() const { return begin_row_; }
    inline integer_t end_row() const { return end_row_; }
    inline MPI_Comm comm() const { return _comm; }
    inline integer_t local_nnz() const { return local_nnz_; }

    void spmv(const DenseM_t& x, DenseM_t& y) const override;
    void spmv(const scalar_t* x, scalar_t* y) const override;

    void apply_scaling
    (const std::vector<scalar_t>& Dr,
     const std::vector<scalar_t>& Dc) override;
    void permute(const integer_t* iorder, const integer_t* order) override;
    std::unique_ptr<CSRMatrix<scalar_t,integer_t>> gather() const;
    int permute_and_scale
    (MatchingJob job, std::vector<integer_t>& perm,
     std::vector<scalar_t>& Dr,
     std::vector<scalar_t>& Dc, bool apply=true) override;
    int permute_and_scale_MC64
    (MatchingJob job, std::vector<integer_t>& perm,
     std::vector<scalar_t>& Dr,
     std::vector<scalar_t>& Dc, bool apply=true);
    void apply_column_permutation
    (const std::vector<integer_t>& perm) override;
    void symmetrize_sparsity() override;
    int read_matrix_market(const std::string& filename) override;

    real_t max_scaled_residual
    (const DenseM_t& x, const DenseM_t& b) const override;
    real_t max_scaled_residual
    (const scalar_t* x, const scalar_t* b) const override;

    CSRGraph<integer_t> get_sub_graph
    (const std::vector<integer_t>& perm,
     const std::vector<std::pair<integer_t,integer_t>>& graph_ranges) const;

    void print() const override;
    void print_dense(const std::string& name) const override;
    void print_MM(const std::string& filename) const override;
    void check() const;


#ifndef DOXYGEN_SHOULD_SKIP_THIS
    // implement outside of this class
    void extract_separator
    (integer_t, const std::vector<std::size_t>&,
     const std::vector<std::size_t>&, DenseM_t&, int) const override {}
    void extract_separator_2d
    (integer_t, const std::vector<std::size_t>&,
     const std::vector<std::size_t>&, DistM_t&) const override {}
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
     const DenseM_t&, DenseM_t&, DenseM_t&, int depth) const override {}
    void front_multiply_2d
    (integer_t, integer_t, const std::vector<integer_t>&, const DistM_t&,
     DistM_t&, DistM_t&, int) const override {}
#endif //DOXYGEN_SHOULD_SKIP_THIS

  protected:
    void split_diag_offdiag();
    void setup_spmv_buffers() const;

    // TODO use MPIComm
    MPI_Comm _comm;

    /**
     * dist_ is the same as the vtxdist array defined by parmetis, it
     *  is the same for each process processor p holds rows
     *  [dist_[p],dist_[p+1]-1]
     */
    std::vector<integer_t> dist_;

    /**
     * _odiag_ptr points to the start of the off-(block)-diagonal
     *  elements.
     */
    std::vector<integer_t> offdiag_start_;

    integer_t local_rows_; // = end_row_ - begin_row_
    integer_t local_nnz_;  // = ptr_[local_rows]
    integer_t begin_row_;  // = dist_[rank]
    integer_t end_row_;    // = dist_[rank+1]

    mutable SPMVBuffers<scalar_t,integer_t> spmv_bufs_;

    using CompressedSparseMatrix<scalar_t,integer_t>::n_;
    using CompressedSparseMatrix<scalar_t,integer_t>::nnz_;
    using CompressedSparseMatrix<scalar_t,integer_t>::ptr_;
    using CompressedSparseMatrix<scalar_t,integer_t>::ind_;
    using CompressedSparseMatrix<scalar_t,integer_t>::val_;
    using CompressedSparseMatrix<scalar_t,integer_t>::symm_sparse_;
  };

  template<typename scalar_t,typename integer_t>
  CSRMatrixMPI<scalar_t,integer_t>::CSRMatrixMPI()
    : CompressedSparseMatrix<scalar_t,integer_t>(), _comm(MPI_COMM_NULL),
    local_rows_(0), local_nnz_(0), begin_row_(0), end_row_(0) {}

  template<typename scalar_t,typename integer_t>
  CSRMatrixMPI<scalar_t,integer_t>::CSRMatrixMPI
  (integer_t local_rows, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, const integer_t* dist, MPI_Comm comm,
   bool symm_sparse) : CompressedSparseMatrix<scalar_t,integer_t>(),
    _comm(comm) {
    auto P = mpi_nprocs(_comm);
    auto rank = mpi_rank(_comm);
    local_rows_ = local_rows;
    local_nnz_ = row_ptr[local_rows]-row_ptr[0];
    begin_row_ = dist[rank];
    end_row_ = dist[rank+1];
    dist_.resize(P+1);
    std::copy(dist, dist+P+1, dist_.data());
    ptr_.resize(local_rows_+1);
    ind_.resize(local_nnz_);
    val_.resize(local_nnz_);
    std::copy(row_ptr, row_ptr+local_rows_+1, ptr_.data());
    std::copy(col_ind, col_ind+local_nnz_, ind_.data());
    std::copy(values, values+local_nnz_, val_.data());
    n_ = dist[P];
    MPI_Allreduce
      (&local_nnz_, &nnz_, 1, mpi_type<integer_t>(), MPI_SUM, _comm);
    symm_sparse_ = symm_sparse;
    for (integer_t r=local_rows_; r>=0; r--)
      ptr_[r] -= ptr_[0];
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
    local_rows_ = local_rows;
    local_nnz_ = 0;
    if (d_ptr) // diagonal block can be empty (NULL)
      local_nnz_ += d_ptr[local_rows] - d_ptr[0];
    if (o_ptr) // off-diagonal block can be empty (NULL)
      local_nnz_ += o_ptr[local_rows] - o_ptr[0];
    symm_sparse_ = symm_sparse;
    dist_.resize(P+1);
    MPI_Allgather
      (&local_rows, 1, mpi_type<integer_t>(),
       &dist_[1], 1, mpi_type<integer_t>(), _comm);
    for (int p=1; p<=P; p++) dist_[p] = dist_[p-1] + dist_[p];
    begin_row_ = dist_[rank];
    end_row_ = dist_[rank+1];
    MPI_Allreduce
      (&local_nnz_, &nnz_, 1, mpi_type<integer_t>(), MPI_SUM, _comm);
    n_ = dist_[P];
    ptr_.resize(local_rows_+1);
    ind_.resize(local_nnz_);
    val_.resize(local_nnz_);
    ptr_[0] = 0;
    offdiag_start_.resize(local_rows_);
    for (integer_t r=0, nz=0; r<local_rows; r++) {
      ptr_[r+1] = ptr_[r];
      if (d_ptr)
        for (integer_t j=d_ptr[r]-d_ptr[0]; j<d_ptr[r+1]-d_ptr[0]; j++) {
          ind_[nz] = d_ind[j] + begin_row_;
          val_[nz++] = d_val[j];
          ptr_[r+1]++;
        }
      offdiag_start_[r] = ptr_[r+1];
      if (o_ptr)
        for (integer_t j=o_ptr[r]-o_ptr[0]; j<o_ptr[r+1]-o_ptr[0]; j++) {
          ind_[nz] = garray[o_ind[j]];
          val_[nz++] = o_val[j];
          ptr_[r+1]++;
        }
    }
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
  (const CSRMatrix<scalar_t,integer_t>* A, MPI_Comm c, bool only_at_root) {
    if (A) {
      n_ = A->size();
      nnz_ = A->nnz();
      symm_sparse_ = A->symm_sparse();
    }
    if (only_at_root) {
      MPI_Bcast(&n_, 1, mpi_type<integer_t>(), 0, c);
      MPI_Bcast(&nnz_, 1, mpi_type<integer_t>(), 0, c);
      MPI_Bcast(&symm_sparse_, sizeof(bool), MPI_BYTE, 0, c);
    }
    auto rank = mpi_rank(c);
    auto P = mpi_nprocs(c);
    dist_.resize(P+1);
    _comm = c;
    if (!only_at_root || (only_at_root && rank==0)) {
      // divide rows over processes, try to give equal number of nnz
      // to each process
      dist_[0] = 0;
      for (int p=1; p<P; p++) {
        integer_t t = p * float(A->nnz()) / P;
        auto hi = std::distance
          (A->ptr(), std::upper_bound
           (A->ptr()+dist_[p-1], A->ptr()+A->size(), t));
        dist_[p] = ((hi-1 >= dist_[p-1]) &&
                    (t-A->ptr(hi-1) < A->ptr(hi)-t)) ? hi-1 : hi;
      }
      dist_[P] = n_;
    }
    if (only_at_root)
      MPI_Bcast(dist_.data(), dist_.size(), mpi_type<integer_t>(), 0, c);
    begin_row_ = dist_[rank];
    end_row_ = dist_[rank+1];
    local_rows_ = end_row_ - begin_row_;
    if (!only_at_root) {
      local_nnz_ = A->ptr(end_row_) - A->ptr(begin_row_);
      auto i0 = A->ptr(begin_row_);
      auto i1 = A->ptr(end_row_);
      ptr_.assign(A->ptr()+begin_row_, A->ptr()+end_row_+1);
      ind_.assign(A->ind() + i0, A->ind() + i1);
      val_.assign(A->val() + i0, A->val() + i1);
    } else {
      auto scnts = new int[2*P];
      auto sdisp = scnts + P;
      if (rank == 0)
        for (int p=0; p<P; p++)
          scnts[p] = A->ptr(dist_[p+1]) - A->ptr(dist_[p]);
      int loc_nnz;
      MPI_Scatter(scnts, 1, mpi_type<int>(), &loc_nnz,
                  1, mpi_type<int>(), 0, c);
      local_nnz_ = loc_nnz;
      ptr_.resize(local_rows_+1);
      ind_.resize(local_nnz_);
      val_.resize(local_nnz_);
      if (rank == 0)
        for (int p=0; p<P; p++) {
          scnts[p] = dist_[p+1] - dist_[p] + 1;
          sdisp[p] = dist_[p];
        }
      MPI_Scatterv
        (rank ? NULL : const_cast<integer_t*>(A->ptr()), scnts, sdisp,
         mpi_type<integer_t>(), ptr_.data(), local_rows_+1,
         mpi_type<integer_t>(), 0, c);
      if (rank == 0)
        for (int p=0; p<P; p++) {
          scnts[p] = A->ptr(dist_[p+1]) - A->ptr(dist_[p]);
          sdisp[p] = A->ptr(dist_[p]);
        }
      MPI_Scatterv
        (rank ? NULL : const_cast<integer_t*>(A->ind()), scnts, sdisp,
         mpi_type<integer_t>(), ind_.data(), local_nnz_,
         mpi_type<integer_t>(), 0, c);
      MPI_Scatterv
        (rank ? NULL : const_cast<scalar_t*>(A->val()), scnts, sdisp,
         mpi_type<scalar_t>(),  val_.data(), local_nnz_,
         mpi_type<scalar_t>(), 0, c);
      delete[] scnts;
    }
    for (integer_t r=local_rows_; r>=0; r--)
      ptr_[r] -= ptr_[0];
    split_diag_offdiag();
    check();
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::print() const {
    for (int p=0; p<mpi_rank(_comm); p++)
      MPI_Barrier(_comm);
    if (mpi_rank(_comm)==0) {
      std::cout << "dist=[";
      for (auto d : dist_) std::cout << d << " ";
      std::cout << "];" << std::endl;
    }
    std::cout << "rank=" << mpi_rank(_comm) << "\nptr=[";
    for (integer_t i=0; i<=local_rows(); i++)
      std::cout << ptr_[i] << " ";
    std::cout << "];" << std::endl;
    std::cout << "ind=[";
    for (integer_t i=0; i<local_rows(); i++) {
      for (integer_t j=ptr_[i]; j<offdiag_start_[i]; j++)
        std::cout << ind_[j] << " ";
      std::cout << "| ";
      for (integer_t j=offdiag_start_[i]; j<ptr_[i+1]; j++)
        std::cout << ind_[j] << " ";
      std::cout << ", ";
    }
    std::cout << "];" << std::endl << std::flush;
    std::cout << "val=[";
    for (integer_t i=0; i<local_rows(); i++) {
      for (integer_t j=ptr_[i]; j<offdiag_start_[i]; j++)
        std::cout << val_[j] << " ";
      std::cout << "| ";
      for (integer_t j=offdiag_start_[i]; j<ptr_[i+1]; j++)
        std::cout << val_[j] << " ";
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
    assert(local_rows_ >= 0);
    assert(end_row_ - begin_row_ == local_rows_);
    integer_t total_rows = local_rows_;
    MPI_Allreduce(MPI_IN_PLACE, &total_rows, 1,
                  mpi_type<integer_t>(), MPI_SUM, _comm);
    assert(total_rows == n_);
    assert(local_nnz_ == ptr_[local_rows_]);
    integer_t total_nnz = local_nnz_;
    MPI_Allreduce(MPI_IN_PLACE, &total_nnz, 1,
                  mpi_type<integer_t>(), MPI_SUM, _comm);
    assert(total_nnz == nnz_);
    assert(end_row_ >= begin_row_);
    if (rank == mpi_nprocs(_comm)-1) {
      assert(end_row_ == n_);
    }
    if (rank == 0) { assert(begin_row_ == 0); }
    assert(begin_row_ == dist_[rank]);
    assert(end_row_ == dist_[rank+1]);
    assert(ptr_[0] == 0);
    for (integer_t r=1; r<=local_rows_; r++) {
      assert(ptr_[r] >= ptr_[r-1]);
    }
    for (integer_t r=0; r<local_rows_; r++) {
      assert(offdiag_start_[r] >= ptr_[r]);
      assert(ptr_[r+1] >= offdiag_start_[r]);
    }
    for (integer_t r=0; r<local_rows_; r++) {
      for (integer_t j=ptr_[r]; j<ptr_[r+1]; j++) {
        assert(ind_[j] >= 0);
        assert(ind_[j] < n_);
      }
    }
    for (integer_t r=0; r<local_rows_; r++) {
      for (integer_t j=ptr_[r]; j<offdiag_start_[r]; j++) {
        assert(ind_[j] >= begin_row_);
        assert(ind_[j] < end_row_);
      }
      for (integer_t j=offdiag_start_[r]; j<ptr_[r+1]; j++) {
        assert(ind_[j] < begin_row_ || ind_[j] >= end_row_);
      }
    }
#endif
  }

  /**
   * Extract part [graph_begin, graph_end) from this sparse matrix,
   * after applying the symmetric permutation perm/iperm.
   */
  // TODO move this to CSRGraph
  template<typename scalar_t,typename integer_t> CSRGraph<integer_t>
  CSRMatrixMPI<scalar_t,integer_t>::get_sub_graph
  (const std::vector<integer_t>& perm,
   const std::vector<std::pair<integer_t,integer_t>>& graph_ranges) const {
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
          scnts[p] += 2 + ptr_[row+1]-ptr_[row];
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
      *pp[d] = ptr_[row+1] - ptr_[row];  pp[d]++;
      for (auto j=ptr_[row]; j<ptr_[row+1]; j++) {
        // send the actual edges
        *pp[d] = perm[ind_[j]];  pp[d]++;
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
    CSRGraph<integer_t> g(n_vert, n_edges);
    g.ptr(0) = 0;
    for (integer_t i=1; i<=n_vert; i++)
      g.ptr(i) = g.ptr(i-1) + edge_count[i-1];
    delete[] edge_count;
    prbuf = 0;
    while (prbuf < rsize) {
      auto my_row = rbuf[prbuf] - graph_ranges[rank].first;
      std::copy(rbuf+prbuf+2, rbuf+prbuf+2+rbuf[prbuf+1],
                g.ind()+g.ptr(my_row));
      prbuf += 2 + rbuf[prbuf+1];
    }
    delete[] rbuf;
    return g;
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::split_diag_offdiag() {
    offdiag_start_.resize(local_rows());
    auto is_diag = [this](const integer_t& e){
      return e >= begin_row() && e < end_row();
    };
    // partition in diagonal and off-diagonal blocks
#pragma omp parallel for
    for (integer_t row=0; row<local_rows(); row++) {
      // like std::partition but on ind and val arrays simultaneously
      auto lo = ptr_[row];
      auto hi = ptr_[row+1];
      auto first_ind = &ind_[lo];
      auto last_ind = &ind_[hi];
      auto first_val = &val_[lo];
      auto last_val = &val_[hi];
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
      offdiag_start_[row] = lo + std::distance(&ind_[lo], first_ind);
    }
  }

  // figure out what to send/receive from/to who during spmv
  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::setup_spmv_buffers() const {
    if (spmv_bufs_.initialized) return;
    spmv_bufs_.initialized = true;

    integer_t nr_offdiag_nnz = 0;
#pragma omp parallel for reduction(+:nr_offdiag_nnz)
    for (integer_t r=0; r<local_rows(); r++)
      nr_offdiag_nnz += ptr_[r+1] - offdiag_start_[r];

    auto P = mpi_nprocs(_comm);
    auto rsizes = new int[2*P]();
    auto ssizes = rsizes+P;
    std::vector<integer_t> spmv_rind;
    spmv_rind.reserve(nr_offdiag_nnz);
    for (integer_t r=0; r<local_rows(); r++)
      for (auto j=offdiag_start_[r]; j<ptr_[r+1]; j++)
        spmv_rind.push_back(ind_[j]);
    std::sort(spmv_rind.begin(), spmv_rind.end());
    spmv_rind.erase
      (std::unique(spmv_rind.begin(), spmv_rind.end()), spmv_rind.end());

    spmv_bufs_.prbuf.reserve(nr_offdiag_nnz);
    for (integer_t r=0; r<local_rows_; r++)
      for (integer_t j=offdiag_start_[r]; j<ptr_[r+1]; j++)
        spmv_bufs_.prbuf.push_back
          (std::distance
           (spmv_rind.begin(), std::lower_bound
            (spmv_rind.begin(), spmv_rind.end(), ind_[j])));

    // how much to receive from each proc
    for (size_t p=0, j=0; p<size_t(P); p++)
      while (j < spmv_rind.size() && spmv_rind[j] < dist_[p+1]) {
        j++; rsizes[p]++;
      }
    MPI_Alltoall
      (rsizes, 1, mpi_type<int>(), ssizes, 1, mpi_type<int>(), _comm);

    auto nr_recv_procs = std::count_if
                 (rsizes, rsizes+P, [](int s){ return s > 0;});
    auto nr_send_procs =
      std::count_if(ssizes, ssizes+P, [](int s){ return s > 0;});
    spmv_bufs_.sranks.reserve(nr_send_procs);
    spmv_bufs_.soff.reserve(nr_send_procs+1);
    spmv_bufs_.rranks.reserve(nr_recv_procs);
    spmv_bufs_.roffs.reserve(nr_recv_procs+1);
    int oset_recv = 0, oset_send = 0;
    for (int p=0; p<P; p++) {
      if (ssizes[p] > 0) {
        spmv_bufs_.sranks.push_back(p);
        spmv_bufs_.soff.push_back(oset_send);
        oset_send += ssizes[p];
      }
      if (rsizes[p] > 0) {
        spmv_bufs_.rranks.push_back(p);
        spmv_bufs_.roffs.push_back(oset_recv);
        oset_recv += rsizes[p];
      }
    }
    delete[] rsizes;
    spmv_bufs_.soff.push_back(oset_send);
    spmv_bufs_.roffs.push_back(oset_recv);
    spmv_bufs_.sind.resize(oset_send);

    std::vector<MPI_Request> req(nr_recv_procs + nr_send_procs);
    for (int p=0; p<nr_recv_procs; p++)
      MPI_Isend
        (spmv_rind.data() + spmv_bufs_.roffs[p],
         spmv_bufs_.roffs[p+1] - spmv_bufs_.roffs[p],
         mpi_type<integer_t>(), spmv_bufs_.rranks[p],
         0, _comm, &req[p]);
    for (int p=0; p<nr_send_procs; p++)
      MPI_Irecv
        (spmv_bufs_.sind.data() + spmv_bufs_.soff[p],
         spmv_bufs_.soff[p+1] - spmv_bufs_.soff[p],
         mpi_type<integer_t>(), spmv_bufs_.sranks[p],
         0, _comm, &req[nr_recv_procs+p]);
    MPI_Waitall(req.size(), req.data(), MPI_STATUSES_IGNORE);

    spmv_bufs_.rbuf.resize(spmv_rind.size());
    spmv_bufs_.sbuf.resize(spmv_bufs_.sind.size());
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::spmv
  (const DenseM_t& x, DenseM_t& y) const {
    assert(x.cols() == y.cols());
    assert(x.rows() == std::size_t(local_rows()));
    assert(y.rows() == std::size_t(local_rows()));
    for (std::size_t c=0; c<x.cols(); c++)
      spmv(x.ptr(0,c), y.ptr(0,c));
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::spmv
  (const scalar_t* x, scalar_t* y) const {
    setup_spmv_buffers();

    for (size_t i=0; i<spmv_bufs_.sind.size(); i++)
      spmv_bufs_.sbuf[i] =
        x[spmv_bufs_.sind[i]-begin_row_];

    MPI_Request* sreq = new MPI_Request
      [spmv_bufs_.sranks.size() +
       spmv_bufs_.rranks.size()];
    MPI_Request* rreq = sreq + spmv_bufs_.sranks.size();
    for (size_t p=0; p<spmv_bufs_.sranks.size(); p++)
      MPI_Isend
        (spmv_bufs_.sbuf.data() + spmv_bufs_.soff[p],
         spmv_bufs_.soff[p+1] - spmv_bufs_.soff[p],
         mpi_type<scalar_t>(), spmv_bufs_.sranks[p],
         0, _comm, &sreq[p]);

    for (size_t p=0; p<spmv_bufs_.rranks.size(); p++)
      MPI_Irecv
        (spmv_bufs_.rbuf.data() + spmv_bufs_.roffs[p],
         spmv_bufs_.roffs[p+1] - spmv_bufs_.roffs[p],
         mpi_type<scalar_t>(), spmv_bufs_.rranks[p],
         0, _comm, &rreq[p]);

    // first do the block diagonal part, while the communication is going on
#pragma omp parallel for
    for (integer_t r=0; r<local_rows(); r++) {
      auto yrow = scalar_t(0.);
      for (auto j=ptr_[r]; j<offdiag_start_[r]; j++)
        yrow += val_[j] * x[ind_[j] - begin_row()];
      y[r] = yrow;
    }
    // wait for incoming messages
    MPI_Waitall
      (spmv_bufs_.rranks.size(), rreq, MPI_STATUSES_IGNORE);

    // do the block off-diagonal part of the matrix
    // TODO some openmp here
    auto pbuf = spmv_bufs_.prbuf.begin();
    for (integer_t r=0; r<local_rows(); r++)
      for (integer_t j=offdiag_start_[r]; j<ptr_[r+1]; j++)
        y[r] += val_[j] * spmv_bufs_.rbuf[*pbuf++];

    // wait for all send messages to finish
    MPI_Waitall
      (spmv_bufs_.sranks.size(), sreq, MPI_STATUSES_IGNORE);
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
    int* rcnts = nullptr;
    int* displs = nullptr;
    if (rank==0) {
      Aseq = std::unique_ptr<CSRMatrix<scalar_t,integer_t>>
        (new CSRMatrix<scalar_t,integer_t>(n_, nnz_));
      rcnts = new int[2*P];
      displs = rcnts + P;
      for (int p=0; p<P; p++) {
        rcnts[p] = dist_[p+1]-dist_[p];
        displs[p] = dist_[p]+1;
      }
    }
    MPI_Gatherv
      (const_cast<integer_t*>(this->ptr())+1, local_rows_,
       mpi_type<integer_t>(), rank ? NULL : Aseq->ptr(), rcnts, displs,
       mpi_type<integer_t>(), 0, _comm);
    if (rank==0) {
      Aseq->ptr(0) = 0;
      for (int p=1; p<P; p++) {
        if (dist_[p] > 0) {
          integer_t p_start = Aseq->ptr(dist_[p]);
          for (int r=dist_[p]; r<dist_[p+1]; r++)
            Aseq->ptr(r+1) += p_start;
        }
      }
      for (int p=0; p<P; p++) {
        rcnts[p] = Aseq->ptr(dist_[p+1])-Aseq->ptr(dist_[p]);
        displs[p] = Aseq->ptr(dist_[p]);
      }
    }
    MPI_Gatherv
      (const_cast<integer_t*>(this->ind()), local_nnz_, mpi_type<integer_t>(),
       rank ? NULL : Aseq->ind(), rcnts, displs,
       mpi_type<integer_t>(), 0, _comm);
    MPI_Gatherv
      (const_cast<scalar_t*>(this->val()), local_nnz_, mpi_type<scalar_t>(),
       rank ? NULL : Aseq->val(), rcnts, displs,
       mpi_type<scalar_t>(), 0, _comm);
    delete[] rcnts;
    return Aseq;
  }

  template<typename scalar_t,typename integer_t> int
  CSRMatrixMPI<scalar_t,integer_t>::permute_and_scale
  (MatchingJob job, std::vector<integer_t>& perm, std::vector<scalar_t>& Dr,
   std::vector<scalar_t>& Dc, bool apply) {
    if (job == MatchingJob::COMBBLAS) {
#if defined(STRUMPACK_USE_COMBBLAS)
      perm.resize(this->size());
      GetAWPM(*this, perm.data());
      return 0;
#else
      if (mpi_rank()==0)
        std::cerr << "# WARNING Matching with CombBLAS not supported.\n"
                  << "# Reconfigure STRUMPACK with CombBLAS support."
                  << std::endl;
      return 1;
#endif
    } else return permute_and_scale_MC64(job, perm, Dr, Dc, apply);
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
  CSRMatrixMPI<scalar_t,integer_t>::permute_and_scale_MC64
  (MatchingJob job, std::vector<integer_t>& perm, std::vector<scalar_t>& Dr,
   std::vector<scalar_t>& Dc, bool apply) {
    if (job == MatchingJob::NONE) return 0;
    if (job == MatchingJob::COMBBLAS || job == MatchingJob::MAX_CARDINALITY) {
      if (!mpi_rank())
        std::cerr << "# WARNING mc64 job not supported,"
                  << " I'm not doing any column permutation"
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
      if (job == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
        Dc_global.resize(this->size());
    }
    MPI_Bcast(&ierr, 1, MPI_INT, 0, _comm);
    if (ierr) return ierr;
    MPI_Bcast(perm.data(), perm.size(), mpi_type<integer_t>(), 0, _comm);
    if (job == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING) {
      auto P = mpi_nprocs(_comm);
      auto rank = mpi_rank(_comm);
      auto scnts = new int[2*P];
      auto sdispls = scnts+P;
      for (int p=0; p<P; p++) {
        scnts[p] = dist_[p+1]-dist_[p];
        sdispls[p] = dist_[p];
      }
      Dr.resize(local_rows_);
      MPI_Scatterv
        (rank ? NULL : Dr_global.data(), scnts, sdispls, mpi_type<scalar_t>(),
         Dr.data(), local_rows_, mpi_type<scalar_t>(), 0, _comm);
      delete[] scnts;
      Dr_global.clear();
      MPI_Bcast(Dc_global.data(), Dc_global.size(),
                mpi_type<scalar_t>(), 0, _comm);
      apply_scaling(Dr, Dc_global);
      Dc.resize(local_rows_);
      std::copy
        (Dc_global.data()+begin_row_, Dc_global.data()+end_row_, Dc.data());
      Dc_global.clear();
    }
    apply_column_permutation(perm);
    symm_sparse_ = false;
    return 0;
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::apply_column_permutation
  (const std::vector<integer_t>& perm) {
    integer_t* iperm = new integer_t[this->size()];
    for (integer_t i=0; i<this->size(); i++) iperm[perm[i]] = i;
#pragma omp parallel for
    for (integer_t r=0; r<this->local_rows_; r++)
      for (integer_t j=ptr_[r]; j<ptr_[r+1]; j++)
        ind_[j] = iperm[ind_[j]];
    delete[] iperm;
    split_diag_offdiag();
  }

  // Apply row and column scaling. Dr is LOCAL, Dc is global!
  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::apply_scaling
  (const std::vector<scalar_t>& Dr, const std::vector<scalar_t>& Dc) {
#pragma omp parallel for
    for (integer_t r=0; r<local_rows_; r++)
      for (integer_t j=ptr_[r]; j<ptr_[r+1]; j++)
        val_[j] = val_[j] * Dr[r] * Dc[ind_[j]];
    STRUMPACK_FLOPS((is_complex<scalar_t>()?6:1)*
                    static_cast<long long int>(2.*double(local_nnz_)));
  }

  // Symmetrize the sparsity pattern.
  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::symmetrize_sparsity() {
    if (symm_sparse_) return;
    auto P = mpi_nprocs(_comm);
    struct Idxij { integer_t i; integer_t j; };
    std::vector<std::vector<Idxij>> sbuf(P);
    for (integer_t r=0; r<local_rows(); r++)
      for (integer_t j=offdiag_start_[r]; j<ptr_[r+1]; j++) {
        auto col = ind_[j];
        auto row = r+begin_row_;
        auto dest = std::upper_bound
          (dist_.begin(), dist_.end(), col) - dist_.begin() - 1;
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
      row_sums[r] = ptr_[r+1]-ptr_[r];
    auto new_nnz = local_nnz_;
    auto ep = edges.begin();
    for (integer_t r=0; r<local_rows(); r++) {
      while (ep != edges.end() && ep->i < r+begin_row_) ep++;
      if (ep == edges.end()) break;
      while (ep != edges.end() && ep->i == r+begin_row_) {
        integer_t kb = offdiag_start_[r], ke = ptr_[r+1];
        if (std::find(this->ind()+kb, this->ind()+ke, ep->j)
            == this->ind()+ke) {
          new_nnz++;
          row_sums[r]++;
        }
        ep++;
      }
    }
    // same for the diagonal block
    for (integer_t r=0; r<local_rows(); r++)
      for (integer_t j=ptr_[r]; j<offdiag_start_[r]; j++) {
        auto lc = ind_[j]-begin_row_;
        integer_t kb = ptr_[lc], ke = offdiag_start_[lc];
        if (std::find(this->ind()+kb, this->ind()+ke, r+begin_row_)
            == this->ind()+ke) {
          row_sums[lc]++;
          new_nnz++;
        }
      }
    if (new_nnz != local_nnz_) {
      local_nnz_ = new_nnz;
      // allocate new arrays
      std::vector<integer_t> new_ptr(local_rows()+1);
      new_ptr[0] = 0;
      for (integer_t r=0; r<local_rows(); r++)
        new_ptr[r+1] = new_ptr[r] + row_sums[r];
      std::vector<integer_t> new_ind(new_nnz);
      std::vector<scalar_t> new_val(new_nnz);
      // copy old nonzeros to new arrays
      for (integer_t r=0; r<local_rows(); r++) {
        row_sums[r] = new_ptr[r] + ptr_[r+1] - ptr_[r];
        for (integer_t j=ptr_[r], k=new_ptr[r]; j<ptr_[r+1]; j++) {
          new_ind[k  ] = ind_[j];
          new_val[k++] = val_[j];
        }
      }
      // diagonal block
      for (integer_t r=0; r<local_rows(); r++)
        for (integer_t j=ptr_[r]; j<offdiag_start_[r]; j++) {
          auto lc = ind_[j]-begin_row_;
          integer_t kb = ptr_[lc], ke = offdiag_start_[lc];
          if (std::find(this->ind()+kb, this->ind()+ke, r+begin_row_) ==
              this->ind()+ke) {
            new_ind[row_sums[lc]] = r+begin_row_;
            new_val[row_sums[lc]] = scalar_t(0.);
            row_sums[lc]++;
          }
        }
      // off-diagonal entries
      ep = edges.begin();
      for (integer_t r=0; r<local_rows(); r++) {
        while (ep != edges.end() && ep->i < r+begin_row_) ep++;
        if (ep == edges.end()) break;
        while (ep != edges.end() && ep->i == r+begin_row_) {
          integer_t kb = offdiag_start_[r], ke = ptr_[r+1];
          if (std::find(this->ind()+kb, this->ind()+ke, ep->j) ==
              this->ind()+ke) {
            new_ind[row_sums[r]] = ep->j;
            new_val[row_sums[r]] = scalar_t(0.);
            row_sums[r]++;
          }
          ep++;
        }
      }
      ptr_.swap(new_ptr);
      ind_.swap(new_ind);
      val_.swap(new_val);
    }
    delete[] row_sums;

    integer_t total_new_nnz = 0;
    MPI_Allreduce(&new_nnz, &total_new_nnz, 1,
                  mpi_type<integer_t>(), MPI_SUM, _comm);
    if (total_new_nnz != nnz_) {
      split_diag_offdiag();
      nnz_ = total_new_nnz;
    }
    symm_sparse_ = true;
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
    setup_spmv_buffers();

    for (size_t i=0; i<spmv_bufs_.sind.size(); i++)
      spmv_bufs_.sbuf[i] = x[spmv_bufs_.sind[i]-begin_row_];

    std::vector<MPI_Request> sreq(spmv_bufs_.sranks.size());
    for (size_t p=0; p<spmv_bufs_.sranks.size(); p++)
      MPI_Isend
        (spmv_bufs_.sbuf.data() + spmv_bufs_.soff[p],
         spmv_bufs_.soff[p+1] - spmv_bufs_.soff[p],
         mpi_type<scalar_t>(), spmv_bufs_.sranks[p],
         0, _comm, &sreq[p]);

    std::vector<MPI_Request> rreq(spmv_bufs_.rranks.size());
    for (size_t p=0; p<spmv_bufs_.rranks.size(); p++)
      MPI_Irecv
        (spmv_bufs_.rbuf.data() + spmv_bufs_.roffs[p],
         spmv_bufs_.roffs[p+1] - spmv_bufs_.roffs[p],
         mpi_type<scalar_t>(), spmv_bufs_.rranks[p],
         0, _comm, &rreq[p]);

    MPI_Waitall(rreq.size(), rreq.data(), MPI_STATUSES_IGNORE);

    real_t m = real_t(0.);
    auto pbuf = spmv_bufs_.prbuf.begin();
    //pragma omp parallel for reduction(max:m)
    for (integer_t r=0; r<local_rows_; r++) {
      auto true_res = b[r];
      auto abs_res = std::abs(b[r]);
      for (auto j=ptr_[r]; j<offdiag_start_[r]; j++) {
        auto c = ind_[j];
        true_res -= val_[j] * x[c-begin_row_];
        abs_res += std::abs(val_[j]) * std::abs(x[c-begin_row_]);
      }
      for (auto j=offdiag_start_[r]; j<ptr_[r+1]; j++) {
        true_res -= val_[j] * spmv_bufs_.rbuf[*pbuf];
        abs_res += std::abs(val_[j]) * std::abs(spmv_bufs_.rbuf[*pbuf]);
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
