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
#ifndef ELIMINATION_TREE_MPI_DIST_HPP
#define ELIMINATION_TREE_MPI_DIST_HPP

#include <iostream>
#include <algorithm>
#include <random>
#include <stddef.h>

#include "StrumpackParameters.hpp"
#include "CompressedSparseMatrix.hpp"
#include "PropMapSparseMatrix.hpp"
#include "CSRMatrixMPI.hpp"
#include "EliminationTreeMPI.hpp"
#include "Redistribute.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  class EliminationTreeMPIDist :
    public EliminationTreeMPI<scalar_t,integer_t> {
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using Opts_t = SPOptions<scalar_t>;
    using SepRange = std::pair<std::size_t,std::size_t>;

  public:
    EliminationTreeMPIDist
    (const Opts_t& opts, const CSRMatrixMPI<scalar_t,integer_t>& A,
     MatrixReorderingMPI<scalar_t,integer_t>& nd, const MPIComm& comm);

    void multifrontal_factorization
    (const CompressedSparseMatrix<scalar_t,integer_t>& A,
     const Opts_t& opts) override;

    void multifrontal_solve_dist
    (DenseM_t& x, const std::vector<integer_t>& dist) override;

    std::tuple<int,int,int> get_sparse_mapped_destination
    (const CSRMatrixMPI<scalar_t,integer_t>& A,
     std::size_t oi, std::size_t oj, std::size_t i, std::size_t j,
     bool duplicate_fronts) const;

    void separator_reordering
    (const Opts_t& opts, const CSRMatrixMPI<scalar_t,integer_t>& A);

  private:
    using EliminationTreeMPI<scalar_t,integer_t>::comm_;
    using EliminationTreeMPI<scalar_t,integer_t>::rank_;
    using EliminationTreeMPI<scalar_t,integer_t>::P_;
    using EliminationTreeMPI<scalar_t,integer_t>::local_range_;
    using EliminationTreeMPI<scalar_t,integer_t>::subtree_ranges_;

    MatrixReorderingMPI<scalar_t,integer_t>& nd_;
    PropMapSparseMatrix<scalar_t,integer_t> Aprop_;

    /**
     * vector with A.local_rows() elements, storing for each row
     * which process has the corresponding separator entry
     */
    std::vector<int> row_owner_;
    void get_all_pfronts();
    void find_row_owner(const CSRMatrixMPI<scalar_t,integer_t>& A);

    /**
     * vector of size _A.size(), storing for each row, to which front
     * it belongs.
     */
    std::vector<int> row_pfront_;
    void find_row_front(const CSRMatrixMPI<scalar_t,integer_t>& A);

    struct ParallelFront {
      ParallelFront() {}
      ParallelFront
      (std::size_t lo, std::size_t hi, int _P0, int _P, BLACSGrid* g)
        : sep_begin(lo), sep_end(hi), P0(_P0), P(_P),
          prows(g->nprows()), pcols(g->npcols()), grid(g) {}
      std::size_t dim_sep() const { return sep_end - sep_begin; }
      std::size_t sep_begin, sep_end;
      int P0, P, prows, pcols;
      const BLACSGrid* grid;
    };

    /** all parallel fronts */
    std::vector<ParallelFront> all_pfronts_;
    /** all parallel fronts on which this process is active. */
    std::vector<ParallelFront> local_pfronts_;

    void symbolic_factorization
    (std::vector<std::vector<integer_t>>& upd,
     std::vector<integer_t>& dist_upd,
     std::vector<float>& subtree_work, float& dsep_work);

    void symbolic_factorization_local
    (integer_t sep, std::vector<std::vector<integer_t>>& upd,
     std::vector<float>& subtree_work, int depth);

    std::unique_ptr<F_t> proportional_mapping
    (const Opts_t& opts, std::vector<std::vector<integer_t>>& upd,
     std::vector<integer_t>& dist_upd,
     std::vector<float>& subtree_work, std::vector<float>& dist_subtree_work,
     integer_t dsep, int P0, int P, int P0_sibling, int P_sibling,
     const MPIComm& fcomm, bool parent_compression, int level);

    std::unique_ptr<F_t> proportional_mapping_sub_graphs
    (const Opts_t& opts, RedistSubTree<integer_t>& tree,
     integer_t dsep, integer_t sep, int P0, int P, int P0_sibling,
     int P_sibling, const MPIComm& fcomm, bool parent_compression, int level);

    void communicate_distributed_separator
    (integer_t dsep, std::vector<integer_t>& dist_upd,
     integer_t& dsep_begin, integer_t& dsep_end,
     std::vector<integer_t>& dsep_upd, int P0, int P,
     int P0_sibling, int P_sibling, int owner, bool bcast_dim_sep);

    template<typename It> void merge_if_larger
    (const It u0, const It u1,
     std::vector<integer_t>& out, integer_t s) const;
  };


  template<typename scalar_t,typename integer_t>
  EliminationTreeMPIDist<scalar_t,integer_t>::EliminationTreeMPIDist
  (const Opts_t& opts, const CSRMatrixMPI<scalar_t,integer_t>& A,
   MatrixReorderingMPI<scalar_t,integer_t>& nd, const MPIComm& comm)
    : EliminationTreeMPI<scalar_t,integer_t>(comm), nd_(nd) {

    std::vector<std::vector<integer_t>> lupd(nd_.local_tree().separators());
    // every process is responsible for 1 distributed separator, so
    // store only 1 dist_upd
    std::vector<integer_t> dupd;
    std::vector<float> ltree_work(nd_.local_tree().separators()),
      dtree_work(nd_.tree().separators());

    float dsep_work;
    MPIComm::control_start("symbolic_factorization");
    symbolic_factorization(lupd, dupd, ltree_work, dsep_work);
    MPIComm::control_stop("symbolic_factorization");

    {
      // communicate dist_subtree_work to everyone
      std::vector<float> sbuf(2*P_);
      for (integer_t dsep=0; dsep<nd_.tree().separators(); dsep++)
        if (rank_ == nd_.proc_dist_sep[dsep]) {
          if (nd_.tree().lch(dsep) == -1)
            sbuf[2*rank_] = ltree_work[nd_.local_tree().root()];
          else sbuf[2*rank_+1] = dsep_work;
        }
      MPI_Allgather
        (MPI_IN_PLACE, 2, MPI_FLOAT, sbuf.data(), 2, MPI_FLOAT, comm_.comm());
      for (integer_t dsep=0; dsep<nd_.tree().separators(); dsep++)
        dtree_work[dsep] = (nd_.tree().lch(dsep) == -1) ?
          sbuf[2*nd_.proc_dist_sep[dsep]] : sbuf[2*nd_.proc_dist_sep[dsep]+1];
    }

    local_range_ = {A.size(), 0};
    MPIComm::control_start("proportional_mapping");
    this->root_ = proportional_mapping
      (opts, lupd, dupd, ltree_work, dtree_work,
       nd_.tree().root(), 0, P_, 0, 0, comm_, true, 0);
    MPIComm::control_stop("proportional_mapping");

    MPIComm::control_start("block_row_A_to_prop_A");
    if (local_range_.first > local_range_.second)
      local_range_.first = local_range_.second = 0;
    subtree_ranges_.resize(P_);
    MPI_Allgather
      (&local_range_, sizeof(SepRange), MPI_BYTE,
       subtree_ranges_.data(), sizeof(SepRange),
       MPI_BYTE, comm_.comm());
    get_all_pfronts();
    find_row_front(A);
    find_row_owner(A);
    Aprop_.setup(A, nd_, *this, opts.compression() != CompressionType::NONE);
    MPIComm::control_stop("block_row_A_to_prop_A");
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::separator_reordering
  (const Opts_t& opts, const CSRMatrixMPI<scalar_t,integer_t>& A) {
    nd_.separator_reordering(opts, Aprop_, this->root());
    // the permutation vector has changed, so reconstruct the
    // distributed sparse matrix
    // TODO avoid this, instead just locally permute Aprop_
    find_row_owner(A);
    Aprop_ = PropMapSparseMatrix<scalar_t,integer_t>();
    Aprop_.setup
      (A, nd_, *this, opts.compression() != CompressionType::NONE);
  }

  /**
   * Figure out on which processor element i,j of the sparse matrix
   * (after symmetric nested dissection permutation) is mapped.  Since
   * we do not have upd[] for all the fronts, it is impossible to
   * figure out the exact rank when the element is part of the F12 or
   * F21 block of a distributed front. So we do some duplication over
   * a column or row of processors in a blacs grid. The return tuple
   * contains <P0, P, dP>, so the value should be send to each (p=P0;
   * p<P0+P; p+=dP)
   */
  template<typename scalar_t,typename integer_t> std::tuple<int,int,int>
  EliminationTreeMPIDist<scalar_t,integer_t>::get_sparse_mapped_destination
  (const CSRMatrixMPI<scalar_t,integer_t>& A,
   std::size_t oi, std::size_t oj, std::size_t i, std::size_t j,
   bool duplicate_fronts) const {
    auto fi = row_pfront_[i];
    if (fi < 0) return std::make_tuple(-fi-1, 1, 1);
    auto fj = row_pfront_[j];
    if (fj < 0) return std::make_tuple(-fj-1, 1, 1);
    constexpr auto B = DistM_t::default_MB;
    int pfront =
      (all_pfronts_[fi].sep_begin < all_pfronts_[fj].sep_begin) ? fi : fj;
    auto& f = all_pfronts_[pfront];
    if (duplicate_fronts)
      return std::make_tuple(f.P0, f.P, 1);
    if (i < f.sep_end) {
      if (j < f.sep_end) // F11
        return std::make_tuple
          (row_owner_[oi] + (((j-f.sep_begin) / B) % f.pcols) * f.prows, 1, 1);
      else // F12
        return
          std::make_tuple(row_owner_[oi], f.prows * f.pcols, f.prows);
    } else {
      if (j < f.sep_end) // F21
        return std::make_tuple
          (f.P0 + (((j-f.sep_begin) / B) % f.pcols) * f.prows, f.prows, 1);
    }
    assert(false);
    return std::make_tuple(0, P_, 1);
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::get_all_pfronts() {
    std::unique_ptr<int[]> iwork(new int[3*P_]);
    auto nr_par_fronts = iwork.get();
    auto rcnts = nr_par_fronts + P_;
    auto rdispls = rcnts + P_;
    nr_par_fronts[rank_] = 0;
    for (auto& f : local_pfronts_)
      if (f.P0 == rank_) nr_par_fronts[rank_]++;
    MPI_Allgather
      (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, nr_par_fronts, 1,
       mpi_type<int>(), comm_.comm());
    int total_pfronts = std::accumulate(nr_par_fronts, nr_par_fronts+P_, 0);
    all_pfronts_.resize(total_pfronts);
    rdispls[0] = 0;
    auto fbytes = sizeof(ParallelFront);
    for (int p=0; p<P_; p++)
      rcnts[p] = nr_par_fronts[p] * fbytes;
    for (int p=1; p<P_; p++)
      rdispls[p] = rdispls[p-1] + rcnts[p-1];
    {
      int i = rdispls[rank_] / fbytes;
      for (auto& f : local_pfronts_)
        if (f.P0 == rank_)
          all_pfronts_[i++] = f;
    }
    MPI_Allgatherv
      (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
       all_pfronts_.data(), rcnts, rdispls, MPI_BYTE, comm_.comm());
  }

  /**
   * Every row of the matrix is mapped to one specific proces according
   * to the proportional mapping. This function finds out which process
   * and stores that info in a vector<integer_t> row_owner_ of size
   * _A.local_rows().
   *
   * First gather a list of ParFrontMaster structs for all parallel
   * fronts on every processor, by gathering the data stored in
   * local_pfront_master (which keeps only the fronts for which this
   * process is the master).  Then loop over all elements in
   * [dist[rank],dist[rank+1]), and figure out to which process that
   * element belongs, by looking for the rank using the ParFrontMaster
   * list.
   */
  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::find_row_owner
  (const CSRMatrixMPI<scalar_t,integer_t>& A) {
    std::size_t lo = A.begin_row(), hi = A.end_row();
    std::size_t n_loc = hi - lo;
    const std::size_t B = DistM_t::default_MB;
    row_owner_.assign(n_loc, -1);
#pragma omp parallel for
    for (std::size_t r=0; r<n_loc; r++) {
      std::size_t pr = nd_.perm()[r+lo];
      auto rf = row_pfront_[pr];
      if (rf < 0)
        row_owner_[r] = -rf-1;
      else {
        auto& f = all_pfronts_[rf];
        row_owner_[r] = f.P0 + (((pr - f.sep_begin) / B) % f.prows);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::find_row_front
  (const CSRMatrixMPI<scalar_t,integer_t>& A) {
    row_pfront_.resize(A.size());
    for (int p=0; p<P_; p++) // local separators
      for (std::size_t r=subtree_ranges_[p].first;
           r<subtree_ranges_[p].second; r++)
        row_pfront_[r] = -p-1;
    for (std::size_t i=0; i<all_pfronts_.size(); i++) {
      auto& f = all_pfronts_[i];
      for (std::size_t r=f.sep_begin; r<f.sep_end; r++)
        row_pfront_[r] = i;
    }
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::multifrontal_factorization
  (const CompressedSparseMatrix<scalar_t,integer_t>& A,
   const Opts_t& opts) {
    this->root_->multifrontal_factorization(Aprop_, opts);
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::multifrontal_solve_dist
  (DenseM_t& x, const std::vector<integer_t>& dist) {
    integer_t B = DistM_t::default_MB;
    integer_t lo = dist[rank_];
    integer_t m = dist[rank_+1] - lo;
    integer_t n = x.cols();
    std::unique_ptr<int[]> iwork(new int[4*P_]);
    auto ibuf = iwork.get();
    auto scnts = ibuf;
    auto rcnts = ibuf + P_;
    auto sdispls = ibuf + 2*P_;
    auto rdispls = ibuf + 3*P_;
    // TODO use Triplet / std::tuple
    struct RCVal { integer_t r, c; scalar_t v; };
    std::unique_ptr<RCVal[]> sbufwork(new RCVal[m*n]);
    auto sbuf = sbufwork.get();
    // since some C++ pad the struct, IdxVal must zero the array or
    // will get valgrind warnings about MPI sending uninitialized data
    //memset(sbuf,0,m*n*sizeof(RCVal));
    std::unique_ptr<RCVal*[]> pp(new RCVal*[P_]);
    std::fill(scnts, scnts+P_, 0);
    if (n == 1) {
      for (integer_t r=0; r<m; r++)
        scnts[row_owner_[r]]++;
    } else {
      for (integer_t r=0; r<m; r++) {
        auto permr = nd_.perm()[r+lo];
        int pf = row_pfront_[permr];
        if (pf < 0) scnts[row_owner_[r]] += n;
        else {
          auto& f = all_pfronts_[pf];
          for (integer_t c=0; c<n; c++)
            scnts[row_owner_[r] + ((c / B) % f.pcols) * f.prows]++;
        }
      }
    }
    sdispls[0] = 0;
    pp[0] = sbuf;
    for (int p=1; p<P_; p++) {
      sdispls[p] = sdispls[p-1] + scnts[p-1];
      pp[p] = sbuf + sdispls[p];
    }
    if (n == 1) {
      const auto perm = nd_.perm().data() + lo;
      for (integer_t r=0; r<m; r++) {
        auto dest = row_owner_[r];
        *pp[dest] = {perm[r], 0, x(r,0)};
        pp[dest]++;
      }
    } else {
      for (integer_t r=0; r<m; r++) {
        auto destr = row_owner_[r];
        auto permr = nd_.perm()[r+lo];
        int pf = row_pfront_[permr];
        if (pf < 0) {
          for (integer_t c=0; c<n; c++) {
            *pp[destr] = {permr, c, x(r,c)};
            pp[destr]++;
          }
        } else {
          auto& f = all_pfronts_[pf];
          for (integer_t c=0; c<n; c++) {
            auto dest = destr + ((c / B) % f.pcols) * f.prows;
            *pp[dest] = {permr, c, x(r,c)};
            pp[dest]++;
          }
        }
      }
    }
    comm_.all_to_all(scnts, 1, rcnts);
    rdispls[0] = 0;
    for (int p=1; p<P_; p++)
      rdispls[p] = rdispls[p-1] + rcnts[p-1];

    MPI_Datatype RCVal_mpi_t;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] =
      {mpi_type<integer_t>(), mpi_type<integer_t>(), mpi_type<scalar_t>()};
    MPI_Aint offsets[3] =
      {offsetof(RCVal, r), offsetof(RCVal, c), offsetof(RCVal, v)};
    MPI_Type_create_struct(3, blocklengths, offsets, types, &RCVal_mpi_t);
    MPI_Type_commit(&RCVal_mpi_t);
    // MPI_Type_contiguous(sizeof(RCVal), MPI_BYTE, &RCVal_mpi_t);
    // MPI_Type_commit(&RCVal_mpi_t);

    auto rbuf = comm_.all_to_allv
      (sbuf, scnts, sdispls, rcnts, rdispls, RCVal_mpi_t);

    DenseM_t xloc(local_range_.second - local_range_.first, n);
    DenseMW_t Xloc
      (Aprop_.size(), n, xloc.data()-local_range_.first, xloc.ld());
    std::vector<DistM_t> xdist(local_pfronts_.size());

    for (std::size_t f=0; f<local_pfronts_.size(); f++)
      xdist[f] = DistM_t
        (local_pfronts_[f].grid, local_pfronts_[f].dim_sep(), n);
    auto rsize = rbuf.size();
#pragma omp parallel for
    for (std::size_t i=0; i<rsize; i++) {
      integer_t r = rbuf[i].r, c = rbuf[i].c;
      if (r >= local_range_.first && r < local_range_.second)
        Xloc(r, c) = rbuf[i].v;
      else {
        for (std::size_t f=0; f<local_pfronts_.size(); f++)
          if (r >= local_pfronts_[f].sep_begin &&
              r < local_pfronts_[f].sep_end) {
            xdist[f].global(r - local_pfronts_[f].sep_begin, c) = rbuf[i].v;
            break;
          }
      }
    }

    this->root_->multifrontal_solve(Xloc, xdist.data());

    rcnts = ibuf;
    scnts = ibuf + P_;
    rdispls = ibuf + 2*P_;
    sdispls = ibuf + 3*P_;
    for (int p=0; p<P_; p++)
      pp[p] = rbuf.data() + sdispls[p];
    for (std::size_t r=local_range_.first; r<local_range_.second; r++) {
      auto dest = std::upper_bound
        (dist.begin(), dist.end(), nd_.iperm()[r])-dist.begin()-1;
      auto permgr = nd_.iperm()[r];
      for (integer_t c=0; c<n; c++) {
        *pp[dest] = {permgr, c, Xloc(r,c)};
        pp[dest]++;
      }
    }
    for (std::size_t i=0; i<local_pfronts_.size(); i++) {
      if (xdist[i].lcols() == 0) continue;
      auto slo = local_pfronts_[i].sep_begin;
      for (int r=0; r<xdist[i].lrows(); r++) {
        auto gr = xdist[i].rowl2g(r) + slo;
        auto permgr = nd_.iperm()[gr];
        auto dest = std::upper_bound
          (dist.begin(), dist.end(), permgr)-dist.begin()-1;
        for (int c=0; c<xdist[i].lcols(); c++) {
          *pp[dest] = {permgr, xdist[i].coll2g(c), xdist[i](r,c)};
          pp[dest]++;
        }
      }
    }
    comm_.all_to_allv
      (rbuf.data(), scnts, sdispls, sbuf, rcnts, rdispls, RCVal_mpi_t);
    MPI_Type_free(&RCVal_mpi_t);
#pragma omp parallel for
    for (std::size_t i=0; i<m*n; i++)
      x(sbuf[i].r-lo,sbuf[i].c) = sbuf[i].v;
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::symbolic_factorization_local
  (integer_t sep, std::vector<std::vector<integer_t>>& upd,
   std::vector<float>& subtree_work, int depth) {
    auto chl = nd_.local_tree().lch(sep);
    auto chr = nd_.local_tree().rch(sep);
    if (depth < params::task_recursion_cutoff_level) {
      if (chl != -1)
#pragma omp task untied default(shared)                                 \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        symbolic_factorization_local(chl, upd, subtree_work, depth+1);
      if (chr != -1)
#pragma omp task untied default(shared)                                 \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        symbolic_factorization_local(chr, upd, subtree_work, depth+1);
#pragma omp taskwait
    } else {
      if (chl != -1)
        symbolic_factorization_local(chl, upd, subtree_work, depth);
      if (chr != -1)
        symbolic_factorization_local(chr, upd, subtree_work, depth);
    }
    auto sep_begin = nd_.local_tree().sizes(sep) +
      nd_.sub_graph_range.first;
    auto sep_end = nd_.local_tree().sizes(sep+1) +
      nd_.sub_graph_range.first;
    {
      auto lor = nd_.local_tree().sizes(sep);
      auto hir = nd_.local_tree().sizes(sep+1);
      std::size_t maxest = nd_.my_sub_graph.ptr(hir) -
        nd_.my_sub_graph.ptr(lor);
      if (chl != -1) maxest += upd[chl].size();
      if (chr != -1) maxest += upd[chr].size();
      upd[sep].reserve(maxest);
      for (integer_t r=lor; r<hir; r++)
        merge_if_larger
          (nd_.my_sub_graph.ind() + nd_.my_sub_graph.ptr(r),
           nd_.my_sub_graph.ind() + nd_.my_sub_graph.ptr(r+1),
           upd[sep], sep_end);
    }
    if (chl != -1)
      merge_if_larger(upd[chl].begin(), upd[chl].end(), upd[sep], sep_end);
    if (chr != -1)
      merge_if_larger(upd[chr].begin(), upd[chr].end(), upd[sep], sep_end);
    upd[sep].shrink_to_fit();
    // assume amount of work per front is N^3, work per subtree is
    // work on front plus children
    float dim_blk = (sep_end - sep_begin) + upd[sep].size();
    subtree_work[sep] = std::pow(dim_blk, 3);
    if (chl != -1) subtree_work[sep] += subtree_work[chl];
    if (chr != -1) subtree_work[sep] += subtree_work[chr];
  }

  template<typename scalar_t,typename integer_t>
  template<typename It> void
  EliminationTreeMPIDist<scalar_t,integer_t>::merge_if_larger
  (const It u0, const It u1, std::vector<integer_t>& out,
   integer_t s) const {
    auto b = std::lower_bound(u0, u1, s);
    auto m = out.size();
    std::copy(b, u1, std::back_inserter(out));
    std::inplace_merge(out.begin(), out.begin() + m, out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
  }

  /**
   * Symbolic factorization:
   *   bottom-up merging of upd indices and work estimate for each
   *   subtree
   *     - first do the symbolic factorization for the local subgraph,
   *        this does not require communication
   *     - then symbolic factorization for the distributed separator
   *        assigned to this process receive upd from left and right
   *        childs, merge with upd for local distributed separator
   *        send upd to parent receive work estimate from left and
   *        right subtrees work estimate for distributed separator
   *        subtree is dim_blk^3 + left_tree + right_tree send work
   *        estimate for this distributed separator / subtree to
   *        parent
   */
  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::symbolic_factorization
  (std::vector<std::vector<integer_t>>& local_upd,
   std::vector<integer_t>& dist_upd, std::vector<float>& local_subtree_work,
   float& dsep_work) {
    nd_.my_sub_graph.sort_rows();
    if (nd_.local_tree().separators() > 0) {
#pragma omp parallel
#pragma omp single
      symbolic_factorization_local
        (nd_.local_tree().root(), local_upd, local_subtree_work, 0);
    }

    dsep_work = 0.0;
    nd_.my_dist_sep.sort_rows();
    std::vector<MPI_Request> sreq;
    for (integer_t dsep=0; dsep<nd_.tree().separators(); dsep++) {
      // only consider the distributed separator owned by this
      // process: 1 leaf and 1 non-leaf
      if (nd_.proc_dist_sep[dsep] != rank_) continue;
      auto pa = nd_.tree().pa(dsep);
      if (pa == -1) continue; // skip the root separator
      auto pa_rank = nd_.proc_dist_sep[pa];
      if (nd_.tree().lch(dsep) == -1) {
        // leaf of distributed tree is local subgraph for process
        // proc_dist_sep[dsep].  local_upd[dsep] was computed above,
        // send it to the parent process
        // proc_dist_sep[nd_.tree().pa(dsep)]. dist_upd is
        // local_upd of the root of the local tree, which is
        // local_upd[this->nbsep-1], or local_upd.back()
        if (nd_.tree().pa(pa) == -1)
          continue; // do not send to parent if parent is root
        sreq.emplace_back();
        comm_.isend
          (local_upd[nd_.local_tree().root()], pa_rank, 1, &sreq.back());
        dsep_work = local_subtree_work[nd_.local_tree().root()];
        sreq.emplace_back();
        comm_.isend(dsep_work, pa_rank, 2, &sreq.back());
      } else {
        auto sep_begin = nd_.dist_sep_range.first;
        auto sep_end = nd_.dist_sep_range.second;
        for (integer_t r=0; r<sep_end-sep_begin; r++)
          merge_if_larger
            (nd_.my_dist_sep.ind() + nd_.my_dist_sep.ptr(r),
             nd_.my_dist_sep.ind() + nd_.my_dist_sep.ptr(r+1),
             dist_upd, sep_end);

        for (int i=0; i<2; i++) {
          // receive dist_upd from left/right child,
          auto du = comm_.template recv_any_src<integer_t>(1);
          // then merge elements larger than sep_end
          merge_if_larger
            (du.second.begin(), du.second.end(), dist_upd, sep_end);
        }
        if (nd_.tree().pa(pa) != -1) { // do not send to root
          sreq.emplace_back();     // send dist_upd to parent
          comm_.isend(dist_upd, pa_rank, 1, &sreq.back());
        }

        float dim_blk = (sep_end - sep_begin) + dist_upd.size();
        dsep_work = std::pow(dim_blk, 3);
        for (int i=0; i<2; i++) {
          // receive work estimates for left and right subtrees
          auto w = comm_.template recv_any_src<float>(2);
          dsep_work += w.second[0];
        }
        if (nd_.tree().pa(pa) != -1) {
          sreq.emplace_back();    // send work estimate to parent
          comm_.isend(dsep_work, pa_rank, 2, &sreq.back());
        }
      }
    }
    wait_all(sreq);
  }

  /**
   * Send the distributed separator from the process responsible for
   * it, to all the processes working on the frontal matrix
   * corresponding to it, and to all processes working on the sibling
   * front (needed for extend-add).  Hence this routine only needs to
   * be called by those processes (or simply by everyone in comm_).
   *
   * If bcast_dim_sep is set to true, the sizes of the separator are
   * communicated to all processes in comm_.
   * TODO I don't see why this is necessary? A process is always
   * also part of its parent, so should always have that info?
   */
  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::
  communicate_distributed_separator
  (integer_t dsep, std::vector<integer_t>& dist_upd,
   integer_t& dsep_begin, integer_t& dsep_end,
   std::vector<integer_t>& dsep_upd, int P0, int P,
   int P0_sibling, int P_sibling, int owner, bool bcast_dim_sep) {
    std::vector<integer_t> sbuf;
    std::vector<MPI_Request> sreq;
    int dest0 = std::min(P0, P0_sibling),
      dest1 = std::max(P0+P, P0_sibling+P_sibling);
    if (rank_ == owner) {
      sreq.resize(dest1-dest0);
      int msg = 0;
      if (bcast_dim_sep) {
        for (int dest=dest0; dest<dest1; dest++)
          comm_.isend(dist_upd, dest, 0, &sreq[msg++]);
      } else {
        sbuf.reserve(2+dist_upd.size());
        sbuf.push_back(nd_.dist_sep_range.first);
        sbuf.push_back(nd_.dist_sep_range.second);
        sbuf.insert(sbuf.end(), dist_upd.begin(), dist_upd.end());
        for (int dest=dest0; dest<dest1; dest++)
          comm_.isend(sbuf, dest, 0, &sreq[msg++]);
      }
    }
    if (bcast_dim_sep) {
      std::vector<integer_t> buf
        ({nd_.dist_sep_range.first, nd_.dist_sep_range.second});
      comm_.broadcast(buf, owner);
      dsep_begin = buf[0];
      dsep_end = buf[1];
    }
    if (rank_ >= dest0 && rank_ < dest1) {
      if (bcast_dim_sep)
        dsep_upd = comm_.template recv<integer_t>(owner, 0);
      else {
        auto rbuf = comm_.template recv<integer_t>(owner, 0);
        dsep_begin = rbuf[0];
        dsep_end = rbuf[1];
        dsep_upd.assign(rbuf.begin()+2, rbuf.end());
      }
    }
    if (rank_ == owner) wait_all(sreq);
  }


  template<typename scalar_t,typename integer_t>
  std::unique_ptr<FrontalMatrix<scalar_t,integer_t>>
  EliminationTreeMPIDist<scalar_t,integer_t>::proportional_mapping
  (const Opts_t& opts, std::vector<std::vector<integer_t>>& local_upd,
   std::vector<integer_t>& dist_upd, std::vector<float>& local_subtree_work,
   std::vector<float>& dist_subtree_work, integer_t dsep,
   int P0, int P, int P0_sibling, int P_sibling,
   const MPIComm& fcomm, bool parent_compression, int level) {
    auto chl = nd_.tree().lch(dsep);
    auto chr = nd_.tree().rch(dsep);
    auto owner = nd_.proc_dist_sep[dsep];

    if (chl == -1 && chr == -1) {
      // leaf of the distributed separator tree -> local subgraph
      RedistSubTree<integer_t> sub_tree
        (nd_, local_upd, local_subtree_work, P0, P,
         P0_sibling, P_sibling, owner, comm_.comm());
      return proportional_mapping_sub_graphs
        (opts, sub_tree, dsep, sub_tree.root, P0, P, P0_sibling, P_sibling,
         fcomm, parent_compression, level);
    }

    integer_t dsep_begin = 0, dsep_end = 0;
    std::vector<integer_t> dsep_upd;
    communicate_distributed_separator
      (dsep, dist_upd, dsep_begin, dsep_end, dsep_upd,
       P0, P, P0_sibling, P_sibling, owner, parent_compression);

    auto dim_dsep = dsep_end - dsep_begin;

    bool use_compression = is_compressed
      (dim_dsep, dsep_upd.size(), parent_compression, opts);

    std::unique_ptr<F_t> front;
    // only store fronts you work on and their siblings (needed for
    // extend-add operation)
    if ((rank_ >= P0 && rank_ < P0+P) ||
        (rank_ >= P0_sibling && rank_ < P0_sibling+P_sibling)) {
      if (P == 1) {
        front = create_frontal_matrix<scalar_t,integer_t>
          (opts, dsep, dsep_begin, dsep_end, dsep_upd, parent_compression,
           level, this->nr_fronts_, rank_ == P0);
        if (P0 == rank_) this->update_local_ranges(dsep_begin, dsep_end);
      } else {
        front = create_frontal_matrix<scalar_t,integer_t>
          (opts, local_pfronts_.size(), dsep_begin, dsep_end, dsep_upd,
           parent_compression, level, this->nr_fronts_, fcomm, P, rank_ == P0);
        if (rank_ >= P0 && rank_ < P0+P) {
          auto fpar = static_cast<FMPI_t*>(front.get());
          local_pfronts_.emplace_back
            (front->sep_begin(), front->sep_end(), P0, P, fpar->grid());
        }
      }
    }
    // here we should still continue, to send the local subgraph
    auto wl = dist_subtree_work[chl];
    auto wr = dist_subtree_work[chr];
    int Pl = std::max(1, std::min(int(std::round(P * wl / (wl + wr))), P-1));
    int Pr = std::max(1, P - Pl);
    auto lch = proportional_mapping
      (opts, local_upd, dist_upd, local_subtree_work, dist_subtree_work,
       chl, P0, Pl, P0+P-Pr, Pr, fcomm.sub(0, Pl), use_compression, level+1);
    auto rch = proportional_mapping
      (opts, local_upd, dist_upd, local_subtree_work, dist_subtree_work,
       chr, P0+P-Pr, Pr, P0, Pl, fcomm.sub(P-Pr, Pr), use_compression, level+1);
    if (front) {
      front->set_lchild(std::move(lch));
      front->set_rchild(std::move(rch));
    }
    return front;
  }

  /**
   * This should only be called by [P0,P0+P) and
   * [P0_sibling,P0_sibling+P_sibling)
   */
  template<typename scalar_t,typename integer_t>
  std::unique_ptr<FrontalMatrix<scalar_t,integer_t>>
  EliminationTreeMPIDist<scalar_t,integer_t>::proportional_mapping_sub_graphs
  (const Opts_t& opts, RedistSubTree<integer_t>& tree,
   integer_t dsep, integer_t sep, int P0, int P,
   int P0_sibling, int P_sibling, const MPIComm& fcomm,
   bool parent_compression, int level) {
    if (!tree.nr_sep) return nullptr;
    auto sep_begin = tree.sep_ptr[sep];
    auto sep_end = tree.sep_ptr[sep+1];
    auto dim_sep = sep_end - sep_begin;
    auto dim_upd = tree.dim_upd[sep];
    std::unique_ptr<F_t> front;
    if ((rank_ >= P0 && rank_ < P0+P) ||
        (rank_ >= P0_sibling && rank_ < P0_sibling+P_sibling)) {
      std::vector<integer_t> upd(tree.upd[sep], tree.upd[sep]+dim_upd);
      if (P == 1) {
        front = create_frontal_matrix<scalar_t,integer_t>
          (opts, sep, sep_begin, sep_end, upd, parent_compression,
           level, this->nr_fronts_, rank_ == P0);
        if (P0 == rank_) this->update_local_ranges(sep_begin, sep_end);
      } else {
        front = create_frontal_matrix<scalar_t,integer_t>
          (opts, local_pfronts_.size(), sep_begin, sep_end, upd,
           parent_compression, level, this->nr_fronts_, fcomm, P, rank_ == P0);
        if (rank_ >= P0 && rank_ < P0+P) {
          auto fpar = static_cast<FMPI_t*>(front.get());
          local_pfronts_.emplace_back
            (front->sep_begin(), front->sep_end(), P0, P, fpar->grid());
        }
      }
    }
    if (rank_ < P0 || rank_ >= P0+P) return front;
    auto chl = tree.lchild[sep];
    auto chr = tree.rchild[sep];
    if (chl != -1 && chr != -1) {
      auto wl = tree.work[chl];
      auto wr = tree.work[chr];
      int Pl = std::max
        (1, std::min(int(std::round(P * wl / (wl + wr))), P-1));
      int Pr = std::max(1, P - Pl);
      bool use_compression = is_compressed
        (dim_sep, dim_upd, parent_compression, opts);
      front->set_lchild
        (proportional_mapping_sub_graphs
         (opts, tree, dsep, chl, P0, Pl, P0+P-Pr, Pr,
          fcomm.sub(0, Pl), use_compression, level+1));
      front->set_rchild
        (proportional_mapping_sub_graphs
         (opts, tree, dsep, chr, P0+P-Pr, Pr, P0, Pl,
          fcomm.sub(P-Pr, Pr), use_compression, level+1));
    }
    return front;
  }

} // end namespace strumpack

#endif
