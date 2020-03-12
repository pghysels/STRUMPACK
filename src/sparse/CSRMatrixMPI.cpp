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
#include <cstddef>
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <memory>
#include <algorithm>


#include "CSRMatrixMPI.hpp"
#if defined(STRUMPACK_USE_COMBBLAS)
#include "AWPMCombBLAS.hpp"
#endif

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  CSRMatrixMPI<scalar_t,integer_t>::CSRMatrixMPI()
    : CSM_t(), comm_(MPI_COMM_NULL),
      lrows_(0), lnnz_(0), brow_(0) {}

  template<typename scalar_t,typename integer_t>
  CSRMatrixMPI<scalar_t,integer_t>::CSRMatrixMPI
  (integer_t lrows, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, const integer_t* dist, MPIComm c,
   bool symm_sparse) : CSM_t(), comm_(std::move(c)) {
    auto P = comm_.size();
    auto rank = comm_.rank();
    assert(dist[rank+1] - dist[rank] == lrows);
    lrows_ = lrows;
    lnnz_ = row_ptr[lrows] - row_ptr[0];
    brow_ = dist[rank];
    dist_.resize(P+1);
    std::copy(dist, dist+P+1, dist_.data());
    ptr_.resize(lrows_+1);
    ind_.resize(lnnz_);
    val_.resize(lnnz_);
    std::copy(row_ptr, row_ptr+lrows_+1, ptr_.data());
    std::copy(col_ind, col_ind+lnnz_, ind_.data());
    std::copy(values, values+lnnz_, val_.data());
    n_ = dist[P];
    nnz_ = comm_.all_reduce(lnnz_, MPI_SUM);
    symm_sparse_ = symm_sparse;
    for (integer_t r=lrows_; r>=0; r--)
      ptr_[r] -= ptr_[0];
    split_diag_offdiag();
    check();
  }

  template<typename scalar_t,typename integer_t>
  CSRMatrixMPI<scalar_t,integer_t>::CSRMatrixMPI
  (integer_t lrows, const integer_t* d_ptr, const integer_t* d_ind,
   const scalar_t* d_val, const integer_t* o_ptr, const integer_t* o_ind,
   const scalar_t* o_val, const integer_t* garray, MPIComm c,
   bool symm_sparse) : CSM_t(), comm_(std::move(c)) {
    auto P = comm_.size();
    auto rank = comm_.rank();
    lrows_ = lrows;
    lnnz_ = 0;
    if (d_ptr) // diagonal block can be empty (NULL)
      lnnz_ += d_ptr[lrows] - d_ptr[0];
    if (o_ptr) // off-diagonal block can be empty (NULL)
      lnnz_ += o_ptr[lrows] - o_ptr[0];
    symm_sparse_ = symm_sparse;
    dist_.resize(P+1);
    MPI_Allgather
      (&lrows, 1, mpi_type<integer_t>(),
       &dist_[1], 1, mpi_type<integer_t>(), comm());
    for (int p=1; p<=P; p++) dist_[p] = dist_[p-1] + dist_[p];
    brow_ = dist_[rank];
    nnz_ = comm_.all_reduce(lnnz_, MPI_SUM);
    n_ = dist_[P];
    ptr_.resize(lrows_+1);
    ind_.resize(lnnz_);
    val_.resize(lnnz_);
    ptr_[0] = 0;
    offdiag_start_.resize(lrows_);
    for (integer_t r=0, nz=0; r<lrows; r++) {
      ptr_[r+1] = ptr_[r];
      if (d_ptr)
        for (integer_t j=d_ptr[r]-d_ptr[0]; j<d_ptr[r+1]-d_ptr[0]; j++) {
          ind_[nz] = d_ind[j] + brow_;
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
  (const CSRMatrix<scalar_t,integer_t>* A, MPIComm c, bool only_at_root)
    : comm_(std::move(c)) {
    auto rank = comm_.rank();
    auto P = comm_.size();
    if (A) {
      n_ = A->size();
      nnz_ = A->nnz();
      symm_sparse_ = A->symm_sparse();
    }
    if (only_at_root) {
      comm_.broadcast(n_);
      comm_.broadcast(nnz_);
      comm_.broadcast(symm_sparse_);
    }
    dist_.resize(P+1);
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
      comm_.broadcast(dist_);
    brow_ = dist_[rank];
    auto erow = dist_[rank+1];
    lrows_ = erow - brow_;
    if (!only_at_root) {
      lnnz_ = A->ptr(erow) - A->ptr(brow_);
      auto i0 = A->ptr(brow_);
      auto i1 = A->ptr(erow);
      ptr_.assign(A->ptr()+brow_, A->ptr()+erow+1);
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
                  1, mpi_type<int>(), 0, comm());
      lnnz_ = loc_nnz;
      ptr_.resize(lrows_+1);
      ind_.resize(lnnz_);
      val_.resize(lnnz_);
      if (rank == 0)
        for (int p=0; p<P; p++) {
          scnts[p] = dist_[p+1] - dist_[p] + 1;
          sdisp[p] = dist_[p];
        }
      MPI_Scatterv
        (rank ? nullptr : const_cast<integer_t*>(A->ptr()), scnts, sdisp,
         mpi_type<integer_t>(), ptr_.data(), lrows_+1,
         mpi_type<integer_t>(), 0, comm());
      if (rank == 0)
        for (int p=0; p<P; p++) {
          scnts[p] = A->ptr(dist_[p+1]) - A->ptr(dist_[p]);
          sdisp[p] = A->ptr(dist_[p]);
        }
      MPI_Scatterv
        (rank ? nullptr : const_cast<integer_t*>(A->ind()), scnts, sdisp,
         mpi_type<integer_t>(), ind_.data(), lnnz_,
         mpi_type<integer_t>(), 0, comm());
      MPI_Scatterv
        (rank ? nullptr : const_cast<scalar_t*>(A->val()), scnts, sdisp,
         mpi_type<scalar_t>(),  val_.data(), lnnz_,
         mpi_type<scalar_t>(), 0, comm());
      delete[] scnts;
    }
    for (integer_t r=lrows_; r>=0; r--)
      ptr_[r] -= ptr_[0];
    split_diag_offdiag();
    check();
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::print() const {
    int P = comm_.size();
    for (int p=0; p<P; p++)
      comm_.barrier();
    if (comm_.is_root()) {
      std::cout << "dist=[";
      for (auto d : dist_) std::cout << d << " ";
      std::cout << "];" << std::endl;
    }
    std::cout << "rank=" << comm_.rank() << "\nptr=[";
    for (integer_t i=0; i<=lrows_; i++)
      std::cout << ptr_[i] << " ";
    std::cout << "];" << std::endl;
    std::cout << "ind=[";
    for (integer_t i=0; i<lrows_; i++) {
      for (integer_t j=ptr_[i]; j<offdiag_start_[i]; j++)
        std::cout << ind_[j] << " ";
      std::cout << "| ";
      for (integer_t j=offdiag_start_[i]; j<ptr_[i+1]; j++)
        std::cout << ind_[j] << " ";
      std::cout << ", ";
    }
    std::cout << "];" << std::endl << std::flush;
    std::cout << "val=[";
    for (integer_t i=0; i<lrows_; i++) {
      for (integer_t j=ptr_[i]; j<offdiag_start_[i]; j++)
        std::cout << val_[j] << " ";
      std::cout << "| ";
      for (integer_t j=offdiag_start_[i]; j<ptr_[i+1]; j++)
        std::cout << val_[j] << " ";
      std::cout << ", ";
    }
    std::cout << "];" << std::endl << std::flush;
    for (int p=comm_.rank(); p<=comm_.size(); p++)
      comm_.barrier();
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::print_dense
  (const std::string& name) const {
    auto Aseq = gather();
    if (Aseq) Aseq->print_dense(name);
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::print_matrix_market
  (const std::string& filename) const {
    auto Aseq = gather();
    if (Aseq) Aseq->print_matrix_market(filename);
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::check() const {
#if !defined(NDEBUG)
    auto rank = comm_.rank();
    assert(lrows_ >= 0);
    auto total_rows = comm_.all_reduce(lrows_, MPI_SUM);
    assert(total_rows == n_);
    assert(lnnz_ == ptr_[lrows_]);
    integer_t total_nnz = comm_.all_reduce(lnnz_, MPI_SUM);
    assert(total_nnz == nnz_);
    if (rank == comm_.size()-1) {
      assert(brow_+lrows_ == n_);
    }
    if (rank == 0) { assert(brow_ == 0); }
    assert(brow_ == dist_[rank]);
    assert(brow_+lrows_ == dist_[rank+1]);
    assert(ptr_[0] == 0);
    for (integer_t r=1; r<=lrows_; r++) {
      assert(ptr_[r] >= ptr_[r-1]);
    }
    for (integer_t r=0; r<lrows_; r++) {
      assert(offdiag_start_[r] >= ptr_[r]);
      assert(ptr_[r+1] >= offdiag_start_[r]);
    }
    for (integer_t r=0; r<lrows_; r++) {
      for (integer_t j=ptr_[r]; j<ptr_[r+1]; j++) {
        assert(ind_[j] >= 0);
        assert(ind_[j] < n_);
      }
    }
    for (integer_t r=0; r<lrows_; r++) {
      for (integer_t j=ptr_[r]; j<offdiag_start_[r]; j++) {
        assert(ind_[j] >= brow_);
        assert(ind_[j] < brow_+lrows_);
      }
      for (integer_t j=offdiag_start_[r]; j<ptr_[r+1]; j++) {
        assert(ind_[j] < brow_ || ind_[j] >= brow_+lrows_);
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
    auto rank = comm_.rank();
    auto P = comm_.size();
    auto scnts = new int[4*P+lrows_];
    auto rcnts = scnts + P;
    auto sdispls = rcnts + P;
    auto rdispls = sdispls + P;
    auto dest = rdispls + P;
    std::fill(scnts, scnts+P, 0);
    std::fill(dest, dest+lrows_, -1);
    for (integer_t row=0; row<lrows_; row++) {
      auto perm_row = perm[row+brow_];
      for (int p=0; p<P; p++)
        if (graph_ranges[p].first <= perm_row &&
            perm_row < graph_ranges[p].second) {
          dest[row] = p;
          scnts[p] += 2 + ptr_[row+1] - ptr_[row];
          break;
        }
    }

    // TODO
    MPI_Alltoall
      (scnts, 1, mpi_type<int>(), rcnts, 1, mpi_type<int>(), comm());
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
    for (integer_t row=0; row<lrows_; row++) {
      auto d = dest[row];
      if (d == -1) continue;
      // send the number of the permuted row (vertex)
      *pp[d] = perm[row+brow_];  pp[d]++;
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

    // TODO
    MPI_Alltoallv(sbuf, scnts, sdispls, mpi_type<integer_t>(),
                  rbuf, rcnts, rdispls, mpi_type<integer_t>(), comm());
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
    offdiag_start_.resize(lrows_);
    auto is_diag = [this](const integer_t& e){
      return e >= brow_ && e < brow_ + lrows_;
    };
    // partition in diagonal and off-diagonal blocks
#pragma omp parallel for
    for (integer_t row=0; row<lrows_; row++) {
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
    for (integer_t r=0; r<lrows_; r++)
      nr_offdiag_nnz += ptr_[r+1] - offdiag_start_[r];

    auto P = comm_.size();
    std::vector<integer_t> spmv_rind;
    spmv_rind.reserve(nr_offdiag_nnz);
    for (integer_t r=0; r<lrows_; r++)
      for (auto j=offdiag_start_[r]; j<ptr_[r+1]; j++)
        spmv_rind.push_back(ind_[j]);
    std::sort(spmv_rind.begin(), spmv_rind.end());
    spmv_rind.erase
      (std::unique(spmv_rind.begin(), spmv_rind.end()), spmv_rind.end());

    spmv_bufs_.prbuf.reserve(nr_offdiag_nnz);
    for (integer_t r=0; r<lrows_; r++)
      for (integer_t j=offdiag_start_[r]; j<ptr_[r+1]; j++)
        spmv_bufs_.prbuf.push_back
          (std::distance
           (spmv_rind.begin(), std::lower_bound
            (spmv_rind.begin(), spmv_rind.end(), ind_[j])));

    // how much to receive from each proc
    std::vector<int> rsizes(P), ssizes(P);
    for (std::size_t p=0, j=0; p<std::size_t(P); p++)
      while (j < spmv_rind.size() && spmv_rind[j] < dist_[p+1]) {
        j++;
        rsizes[p]++;
      }
    comm_.all_to_all(rsizes.data(), 1, ssizes.data());

    auto npr = std::count_if(rsizes.begin(), rsizes.end(),
                             [](int s){ return s > 0;});
    auto nps = std::count_if(ssizes.begin(), ssizes.end(),
                             [](int s){ return s > 0;});
    spmv_bufs_.sranks.reserve(nps);
    spmv_bufs_.soff.reserve(nps+1);
    spmv_bufs_.rranks.reserve(npr);
    spmv_bufs_.roffs.reserve(npr+1);
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
    spmv_bufs_.soff.push_back(oset_send);
    spmv_bufs_.roffs.push_back(oset_recv);
    spmv_bufs_.sind.resize(oset_send);
    std::vector<MPI_Request> req(npr + nps);
    for (int p=0; p<npr; p++)
      comm_.isend(spmv_rind.data() + spmv_bufs_.roffs[p],
                  spmv_bufs_.roffs[p+1] - spmv_bufs_.roffs[p],
                  spmv_bufs_.rranks[p], 0, &req[p]);
    for (int p=0; p<nps; p++)
      comm_.irecv(spmv_bufs_.sind.data() + spmv_bufs_.soff[p],
                  spmv_bufs_.soff[p+1] - spmv_bufs_.soff[p],
                  spmv_bufs_.sranks[p], 0, &req[npr+p]);
    wait_all(req);
    spmv_bufs_.rbuf.resize(oset_recv);
    spmv_bufs_.sbuf.resize(oset_send);
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::spmv
  (const DenseM_t& x, DenseM_t& y) const {
    assert(x.cols() == y.cols());
    assert(x.rows() == std::size_t(lrows_));
    assert(y.rows() == std::size_t(lrows_));
    for (std::size_t c=0; c<x.cols(); c++)
      spmv(x.ptr(0,c), y.ptr(0,c));
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::spmv
  (const scalar_t* x, scalar_t* y) const {
    setup_spmv_buffers();

    for (std::size_t i=0; i<spmv_bufs_.sind.size(); i++)
      spmv_bufs_.sbuf[i] =
        x[spmv_bufs_.sind[i]-brow_];

    std::vector<MPI_Request> sreq(spmv_bufs_.sranks.size()),
      rreq(spmv_bufs_.rranks.size());
    for (std::size_t p=0; p<spmv_bufs_.sranks.size(); p++)
      comm_.isend(spmv_bufs_.sbuf.data() + spmv_bufs_.soff[p],
                  spmv_bufs_.soff[p+1] - spmv_bufs_.soff[p],
                  spmv_bufs_.sranks[p], 0, &sreq[p]);

    for (std::size_t p=0; p<spmv_bufs_.rranks.size(); p++)
      comm_.irecv(spmv_bufs_.rbuf.data() + spmv_bufs_.roffs[p],
                  spmv_bufs_.roffs[p+1] - spmv_bufs_.roffs[p],
                  spmv_bufs_.rranks[p], 0, &rreq[p]);

    // first do the block diagonal part, while the communication is going on
#pragma omp parallel for
    for (integer_t r=0; r<lrows_; r++) {
      auto yrow = scalar_t(0.);
      for (auto j=ptr_[r]; j<offdiag_start_[r]; j++)
        yrow += val_[j] * x[ind_[j] - brow_];
      y[r] = yrow;
    }
    // wait for incoming messages
    wait_all(rreq);

    // do the block off-diagonal part of the matrix
    // TODO some openmp here
    auto pbuf = spmv_bufs_.prbuf.begin();
    for (integer_t r=0; r<lrows_; r++)
      for (integer_t j=offdiag_start_[r]; j<ptr_[r+1]; j++)
        y[r] += val_[j] * spmv_bufs_.rbuf[*pbuf++];

    // wait for all send messages to finish
    wait_all(sreq);
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
    auto rank = comm_.rank();
    auto P = comm_.size();
    if (rank == 0) {
      std::unique_ptr<int[]> iwork(new int[2*P]);
      auto rcnts = iwork.get();
      auto displs = rcnts + P;
      for (int p=0; p<P; p++) {
        rcnts[p] = dist_[p+1]-dist_[p];
        displs[p] = dist_[p];
      }
      std::unique_ptr<CSRMatrix<scalar_t,integer_t>> Aseq
        (new CSRMatrix<scalar_t,integer_t>(n_, nnz_));
      MPI_Gatherv
        (const_cast<integer_t*>(this->ptr())+1, lrows_,
         mpi_type<integer_t>(), Aseq->ptr()+1, rcnts, displs,
         mpi_type<integer_t>(), 0, comm());
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
      MPI_Gatherv
        (const_cast<integer_t*>(this->ind()), lnnz_,
         mpi_type<integer_t>(), Aseq->ind(), rcnts, displs,
         mpi_type<integer_t>(), 0, comm());
      MPI_Gatherv
        (const_cast<scalar_t*>(this->val()), lnnz_,
         mpi_type<scalar_t>(), Aseq->val(), rcnts, displs,
         mpi_type<scalar_t>(), 0, comm());
      return Aseq;
    } else {
      MPI_Gatherv
        (const_cast<integer_t*>(this->ptr())+1, lrows_,
         mpi_type<integer_t>(), nullptr, nullptr, nullptr,
         mpi_type<integer_t>(), 0, comm());
      MPI_Gatherv
        (const_cast<integer_t*>(this->ind()), lnnz_,
         mpi_type<integer_t>(), nullptr, nullptr, nullptr,
         mpi_type<integer_t>(), 0, comm());
      MPI_Gatherv
        (const_cast<scalar_t*>(this->val()), lnnz_,
         mpi_type<scalar_t>(), nullptr, nullptr, nullptr,
         mpi_type<scalar_t>(), 0, comm());
      return std::unique_ptr<CSRMatrix<scalar_t,integer_t>>();
    }
  }

  template<typename scalar_t,typename integer_t>
  std::unique_ptr<CSRGraph<integer_t>>
  CSRMatrixMPI<scalar_t,integer_t>::gather_graph() const {
    auto rank = comm_.rank();
    auto P = comm_.size();
    if (rank == 0) {
      std::unique_ptr<int[]> iwork(new int[2*P]);
      auto rcnts = iwork.get();
      auto displs = rcnts + P;
      for (int p=0; p<P; p++) {
        rcnts[p] = dist_[p+1]-dist_[p];
        displs[p] = dist_[p];
      }
      std::unique_ptr<CSRGraph<integer_t>> Aseq
        (new CSRGraph<integer_t>(n_, nnz_));
      MPI_Gatherv
        (const_cast<integer_t*>(this->ptr())+1, lrows_,
         mpi_type<integer_t>(), Aseq->ptr()+1, rcnts, displs,
         mpi_type<integer_t>(), 0, comm());
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
      MPI_Gatherv
        (const_cast<integer_t*>(this->ind()), lnnz_,
         mpi_type<integer_t>(), Aseq->ind(), rcnts, displs,
         mpi_type<integer_t>(), 0, comm());
      return Aseq;
    } else {
      MPI_Gatherv
        (const_cast<integer_t*>(this->ptr())+1, lrows_,
         mpi_type<integer_t>(), nullptr, nullptr, nullptr,
         mpi_type<integer_t>(), 0, comm());
      MPI_Gatherv
        (const_cast<integer_t*>(this->ind()), lnnz_,
         mpi_type<integer_t>(), nullptr, nullptr, nullptr,
         mpi_type<integer_t>(), 0, comm());
      return std::unique_ptr<CSRGraph<integer_t>>();
    }
  }

  template<typename scalar_t,typename integer_t> int
  CSRMatrixMPI<scalar_t,integer_t>::permute_and_scale
  (MatchingJob job, std::vector<integer_t>& perm, std::vector<scalar_t>& lDr,
   std::vector<scalar_t>& gDc, bool apply) {
    if (job == MatchingJob::COMBBLAS) {
#if defined(STRUMPACK_USE_COMBBLAS)
      perm.resize(this->size());
      GetAWPM(*this, perm.data());
      return 0;
#else
      if (comm_.is_root())
        std::cerr << "# WARNING Matching with CombBLAS not supported.\n"
                  << "# Reconfigure STRUMPACK with CombBLAS support."
                  << std::endl;
      return 1;
#endif
    } else return permute_and_scale_MC64(job, perm, lDr, gDc, apply);
  }

  template<typename scalar_t,typename integer_t> int
  CSRMatrixMPI<scalar_t,integer_t>::permute_and_scale_MC64
  (MatchingJob job, std::vector<integer_t>& perm, std::vector<scalar_t>& lDr,
   std::vector<scalar_t>& gDc, bool apply) {
    if (job == MatchingJob::NONE) return 0;
    if (job == MatchingJob::COMBBLAS || job == MatchingJob::MAX_CARDINALITY) {
      if (!mpi_rank())
        std::cerr << "# WARNING mc64 job not supported,"
                  << " I'm not doing any column permutation"
                  << " or matrix scaling!" << std::endl;
      return 1;
    }
    auto Aseq = gather();
    std::vector<scalar_t> gDr;
    int ierr = 0;
    if (Aseq) {
      ierr = Aseq->permute_and_scale
        (job, perm, gDr, gDc, false/*do not apply perm/scaling*/);
      Aseq.reset(nullptr);
    } else {
      perm.resize(this->size());
      if (job == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
        gDc.resize(this->size());
    }
    comm_.broadcast(ierr);
    if (ierr) return ierr;
    comm_.broadcast(perm);
    if (job == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING) {
      auto P = comm_.size();
      auto rank = comm_.rank();
      std::unique_ptr<int[]> iwork(new int[2*P]);
      auto scnts = iwork.get();
      auto sdispls = scnts+P;
      for (int p=0; p<P; p++) {
        scnts[p] = dist_[p+1] - dist_[p];
        sdispls[p] = dist_[p];
      }
      lDr.resize(lrows_);
      MPI_Scatterv
        (rank ? nullptr : gDr.data(), scnts, sdispls, mpi_type<scalar_t>(),
         lDr.data(), lrows_, mpi_type<scalar_t>(), 0, comm());
      gDr.clear();
      comm_.broadcast(gDc);
      apply_scaling(lDr, gDc);
    }
    apply_column_permutation(perm);
    return 0;
  }

  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::apply_column_permutation
  (const std::vector<integer_t>& perm) {
    std::unique_ptr<integer_t[]> iperm(new integer_t[this->size()]);
    for (integer_t i=0; i<this->size(); i++) iperm[perm[i]] = i;
#pragma omp parallel for
    for (integer_t r=0; r<this->lrows_; r++)
      for (integer_t j=ptr_[r]; j<ptr_[r+1]; j++)
        ind_[j] = iperm[ind_[j]];
    split_diag_offdiag();
    symm_sparse_ = false;
    spmv_bufs_ = SPMVBuffers<scalar_t,integer_t>();
  }

  // Apply row and column scaling. Dr is LOCAL, Dc is global!
  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::apply_scaling
  (const std::vector<scalar_t>& lDr, const std::vector<scalar_t>& gDc) {
#pragma omp parallel for
    for (integer_t r=0; r<lrows_; r++)
      for (integer_t j=ptr_[r]; j<ptr_[r+1]; j++)
        val_[j] = val_[j] * lDr[r] * gDc[ind_[j]];
    STRUMPACK_FLOPS((is_complex<scalar_t>()?6:1)*
                    static_cast<long long int>(2.*double(lnnz_)));
  }

  // Symmetrize the sparsity pattern.
  template<typename scalar_t,typename integer_t> void
  CSRMatrixMPI<scalar_t,integer_t>::symmetrize_sparsity() {
    if (symm_sparse_) return;
    auto P = comm_.size();
    struct Idxij {
      integer_t i, j;
    };
    MPI_Datatype Idxij_mpi_t;
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {mpi_type<decltype(Idxij::i)>(),
                             mpi_type<decltype(Idxij::j)>()};
    MPI_Aint offsets[2] = {offsetof(Idxij, i), offsetof(Idxij, j)};
    MPI_Type_create_struct(2, blocklengths, offsets, types, &Idxij_mpi_t);
    MPI_Type_commit(&Idxij_mpi_t);

    std::vector<std::vector<Idxij>> sbuf(P);
    for (integer_t r=0; r<lrows_; r++)
      for (integer_t j=offdiag_start_[r]; j<ptr_[r+1]; j++) {
        auto col = ind_[j];
        auto row = r + brow_;
        auto dest = std::upper_bound
          (dist_.begin(), dist_.end(), col) - dist_.begin() - 1;
        sbuf[dest].emplace_back(Idxij{col, row});
      }

    std::vector<Idxij> edges;
    std::vector<Idxij*> pp;
    comm_.all_to_all_v(sbuf, edges, pp, Idxij_mpi_t);
    MPI_Type_free(&Idxij_mpi_t);

    std::sort(edges.begin(), edges.end(),
              [](const Idxij& a, const Idxij& b) {
                // sort according to rows, then columns
                if (a.i != b.i) return (a.i < b.i);
                return (a.j < b.j);
              });
    // count how many of the received values are not already here
    std::unique_ptr<integer_t[]> row_sums(new integer_t[lrows_]);
    for (integer_t r=0; r<lrows_; r++)
      row_sums[r] = ptr_[r+1]-ptr_[r];
    auto new_nnz = lnnz_;
    auto ep = edges.begin();
    for (integer_t r=0; r<lrows_; r++) {
      while (ep != edges.end() && ep->i < r+brow_) ep++;
      if (ep == edges.end()) break;
      while (ep != edges.end() && ep->i == r+brow_) {
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
    for (integer_t r=0; r<lrows_; r++)
      for (integer_t j=ptr_[r]; j<offdiag_start_[r]; j++) {
        auto lc = ind_[j] - brow_;
        integer_t kb = ptr_[lc], ke = offdiag_start_[lc];
        if (std::find(this->ind()+kb, this->ind()+ke, r+brow_)
            == this->ind()+ke) {
          row_sums[lc]++;
          new_nnz++;
        }
      }
    if (new_nnz != lnnz_) {
      lnnz_ = new_nnz;
      // allocate new arrays
      std::vector<integer_t> new_ptr(lrows_+1);
      new_ptr[0] = 0;
      for (integer_t r=0; r<lrows_; r++)
        new_ptr[r+1] = new_ptr[r] + row_sums[r];
      std::vector<integer_t> new_ind(new_nnz);
      std::vector<scalar_t> new_val(new_nnz);
      // copy old nonzeros to new arrays
      for (integer_t r=0; r<lrows_; r++) {
        row_sums[r] = new_ptr[r] + ptr_[r+1] - ptr_[r];
        for (integer_t j=ptr_[r], k=new_ptr[r]; j<ptr_[r+1]; j++) {
          new_ind[k  ] = ind_[j];
          new_val[k++] = val_[j];
        }
      }
      // diagonal block
      for (integer_t r=0; r<lrows_; r++)
        for (integer_t j=ptr_[r]; j<offdiag_start_[r]; j++) {
          auto lc = ind_[j] - brow_;
          integer_t kb = ptr_[lc], ke = offdiag_start_[lc];
          if (std::find(this->ind()+kb, this->ind()+ke, r+brow_) ==
              this->ind()+ke) {
            new_ind[row_sums[lc]] = r+brow_;
            new_val[row_sums[lc]] = scalar_t(0.);
            row_sums[lc]++;
          }
        }
      // off-diagonal entries
      ep = edges.begin();
      for (integer_t r=0; r<lrows_; r++) {
        while (ep != edges.end() && ep->i < r+brow_) ep++;
        if (ep == edges.end()) break;
        while (ep != edges.end() && ep->i == r+brow_) {
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

    auto total_new_nnz = comm_.all_reduce(new_nnz, MPI_SUM);
    if (total_new_nnz != nnz_) {
      split_diag_offdiag();
      nnz_ = total_new_nnz;
    }
    symm_sparse_ = true;
    spmv_bufs_ = SPMVBuffers<scalar_t,integer_t>();
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

    for (std::size_t i=0; i<spmv_bufs_.sind.size(); i++)
      spmv_bufs_.sbuf[i] = x[spmv_bufs_.sind[i]-brow_];

    std::vector<MPI_Request> sreq(spmv_bufs_.sranks.size());
    for (std::size_t p=0; p<spmv_bufs_.sranks.size(); p++)
      comm_.isend(spmv_bufs_.sbuf.data() + spmv_bufs_.soff[p],
                  spmv_bufs_.soff[p+1] - spmv_bufs_.soff[p],
                  spmv_bufs_.sranks[p], 0, &sreq[p]);

    std::vector<MPI_Request> rreq(spmv_bufs_.rranks.size());
    for (std::size_t p=0; p<spmv_bufs_.rranks.size(); p++)
      comm_.irecv(spmv_bufs_.rbuf.data() + spmv_bufs_.roffs[p],
                  spmv_bufs_.roffs[p+1] - spmv_bufs_.roffs[p],
                  spmv_bufs_.rranks[p], 0, &rreq[p]);

    //MPI_Waitall(rreq.size(), rreq.data(), MPI_STATUSES_IGNORE);
    wait_all(rreq);

    real_t m = real_t(0.);
    auto pbuf = spmv_bufs_.prbuf.begin();
    //pragma omp parallel for reduction(max:m)
    for (integer_t r=0; r<lrows_; r++) {
      auto true_res = b[r];
      auto abs_res = std::abs(b[r]);
      for (auto j=ptr_[r]; j<offdiag_start_[r]; j++) {
        auto c = ind_[j];
        true_res -= val_[j] * x[c-brow_];
        abs_res += std::abs(val_[j]) * std::abs(x[c-brow_]);
      }
      for (auto j=offdiag_start_[r]; j<ptr_[r+1]; j++) {
        true_res -= val_[j] * spmv_bufs_.rbuf[*pbuf];
        abs_res += std::abs(val_[j]) * std::abs(spmv_bufs_.rbuf[*pbuf]);
        pbuf++;
      }
      m = std::max(m, std::abs(true_res) / std::abs(abs_res));
    }
    // wait for all send messages to finish
    wait_all(sreq);
    return comm_.all_reduce(m, MPI_MAX);
  }

  // explicit template instantiations
  template class CSRMatrixMPI<float,int>;
  template class CSRMatrixMPI<double,int>;
  template class CSRMatrixMPI<std::complex<float>,int>;
  template class CSRMatrixMPI<std::complex<double>,int>;

  template class CSRMatrixMPI<float,long int>;
  template class CSRMatrixMPI<double,long int>;
  template class CSRMatrixMPI<std::complex<float>,long int>;
  template class CSRMatrixMPI<std::complex<double>,long int>;

  template class CSRMatrixMPI<float,long long int>;
  template class CSRMatrixMPI<double,long long int>;
  template class CSRMatrixMPI<std::complex<float>,long long int>;
  template class CSRMatrixMPI<std::complex<double>,long long int>;

} // end namespace strumpack
