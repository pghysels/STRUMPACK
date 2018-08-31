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

#include "StrumpackParameters.hpp"
#include "CompressedSparseMatrix.hpp"
#include "ProportionallyDistributedSparseMatrix.hpp"
#include "CSRMatrixMPI.hpp"
#include "MatrixReorderingMPI.hpp"
#include "FrontalMatrixHSS.hpp"
#include "FrontalMatrixDense.hpp"
#include "FrontalMatrixDenseMPI.hpp"
#include "FrontalMatrixHSSMPI.hpp"
#include "FrontalMatrixBLRMPI.hpp"
#include "Redistribute.hpp"
#include "HSS/HSSPartitionTree.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  class EliminationTreeMPIDist :
    public EliminationTreeMPI<scalar_t,integer_t> {
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FD_t = FrontalMatrixDense<scalar_t,integer_t>;
    using FHSS_t = FrontalMatrixHSS<scalar_t,integer_t>;
    using FBLR_t = FrontalMatrixBLR<scalar_t,integer_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using FDMPI_t = FrontalMatrixDenseMPI<scalar_t,integer_t>;
    using FHSSMPI_t = FrontalMatrixHSSMPI<scalar_t,integer_t>;
    using FBLRMPI_t = FrontalMatrixBLRMPI<scalar_t,integer_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using SepRange = std::pair<std::size_t,std::size_t>;

  public:
    EliminationTreeMPIDist
    (const SPOptions<scalar_t>& opts,
     const CSRMatrixMPI<scalar_t,integer_t>& A,
     MatrixReorderingMPI<scalar_t,integer_t>& nd,
     const MPIComm& comm);

    void multifrontal_factorization
    (const CompressedSparseMatrix<scalar_t,integer_t>& A,
     const SPOptions<scalar_t>& opts) override;

    void multifrontal_solve_dist
    (DenseM_t& x, const std::vector<integer_t>& dist) override;

    std::tuple<int,int,int> get_sparse_mapped_destination
    (const CSRMatrixMPI<scalar_t,integer_t>& A,
     std::size_t i, std::size_t j, bool duplicate_fronts) const;

  private:
    using EliminationTreeMPI<scalar_t,integer_t>::comm_;
    using EliminationTreeMPI<scalar_t,integer_t>::rank_;
    using EliminationTreeMPI<scalar_t,integer_t>::P_;

    MatrixReorderingMPI<scalar_t,integer_t>& nd_;
    ProportionallyDistributedSparseMatrix<scalar_t,integer_t> Aprop_;

    SepRange local_range_;

    /**
     * vector with _A.local_rows() elements, storing for each row
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
    (const SPOptions<scalar_t>& opts,
     std::vector<std::vector<integer_t>>& upd,
     std::vector<integer_t>& dist_upd,
     std::vector<float>& subtree_work, std::vector<float>& dist_subtree_work,
     integer_t dsep, int P0, int P, int P0_sibling, int P_sibling,
     const MPIComm& fcomm, bool hss_parent, int level);

    std::unique_ptr<F_t> proportional_mapping_sub_graphs
    (const SPOptions<scalar_t>& opts, RedistSubTree<integer_t>& tree,
     integer_t dsep, integer_t sep, int P0, int P, int P0_sibling,
     int P_sibling, const MPIComm& fcomm, bool hss_parent, int level);

    void communicate_distributed_separator
    (integer_t dsep, std::vector<integer_t>& dist_upd,
     integer_t& dsep_begin, integer_t& dsep_end,
     std::vector<integer_t>& dsep_upd, int P0, int P,
     int P0_sibling, int P_sibling, int owner, bool use_hss);

    void communicate_hss_tree
    (HSS::HSSPartitionTree& tree, integer_t dsep, int P0, int P, int owner);
    void communicate_admissibility
    (std::vector<bool>& adm, integer_t dsep, int P0, int P, int owner);
  };


  template<typename scalar_t,typename integer_t>
  EliminationTreeMPIDist<scalar_t,integer_t>::EliminationTreeMPIDist
  (const SPOptions<scalar_t>& opts, const CSRMatrixMPI<scalar_t,integer_t>& A,
   MatrixReorderingMPI<scalar_t,integer_t>& nd, const MPIComm& comm)
    : EliminationTreeMPI<scalar_t,integer_t>(comm), nd_(nd) {
    std::vector<std::vector<integer_t>> local_upd(nd_.local_tree().separators());

    // every process is responsible for 1 distributed separator, so
    // store only 1 dist_upd
    std::vector<integer_t> dist_upd;
    std::vector<float> local_subtree_work(nd_.local_tree().separators()),
      dist_subtree_work(nd_.tree().separators());

    float dsep_work;
    MPIComm::control_start("symbolic_factorization");
    symbolic_factorization
      (local_upd, dist_upd, local_subtree_work, dsep_work);
    MPIComm::control_stop("symbolic_factorization");


    {// communicate dist_subtree_work to everyone
      std::vector<float> sbuf(2*P_);
      // initialize buffer or valgrind will complain about MPI sending
      // uninitialized data
      sbuf[2*rank_] = sbuf[2*rank_+1] = 0.0;
      for (integer_t dsep=0; dsep<nd_.tree().separators(); dsep++)
        if (rank_ == nd_.proc_dist_sep[dsep]) {
          if (nd_.tree().lch(dsep) == -1)
            sbuf[2*rank_] = local_subtree_work[nd_.local_tree().root()];
          else sbuf[2*rank_+1] = dsep_work;
        }
      MPI_Allgather
        (MPI_IN_PLACE, 2, MPI_FLOAT, sbuf.data(), 2, MPI_FLOAT, comm_.comm());
      for (integer_t dsep=0; dsep<nd_.tree().separators(); dsep++)
        dist_subtree_work[dsep] = (nd_.tree().lch(dsep) == -1) ?
          sbuf[2*nd_.proc_dist_sep[dsep]] : sbuf[2*nd_.proc_dist_sep[dsep]+1];
    }

    local_range_ = std::make_pair(A.size(), 0);
    MPIComm::control_start("proportional_mapping");
    this->root_ = proportional_mapping
      (opts, local_upd, dist_upd, local_subtree_work, dist_subtree_work,
       nd_.tree().root(), 0, P_, 0, 0, comm_, true, 0);
    MPIComm::control_stop("proportional_mapping");

    MPI_Pcontrol(1, "block_row_A_to_prop_A");
    if (local_range_.first > local_range_.second)
      local_range_.first = local_range_.second = 0;
    this->subtree_ranges.resize(P_);
    MPI_Allgather
      (&local_range_, sizeof(SepRange), MPI_BYTE,
       this->subtree_ranges.data(), sizeof(SepRange),
       MPI_BYTE, comm_.comm());
    get_all_pfronts();
    find_row_owner(A);
    find_row_front(A);
    Aprop_.setup(A, nd_, *this, opts.use_HSS());
    MPI_Pcontrol(-1, "block_row_A_to_prop_A");
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
   std::size_t i, std::size_t j, bool duplicate_fronts) const {
    auto fi = row_pfront_[i];
    if (fi < 0) return std::make_tuple(-fi-1, 1, 1);
    auto fj = row_pfront_[j];
    if (fj < 0) return std::make_tuple(-fj-1, 1, 1);
    constexpr auto B = DistM_t::default_MB;
    int pfront =
      (all_pfronts_[fi].sep_begin < all_pfronts_[fj].sep_begin) ? fi : fj;
    auto& f = all_pfronts_[pfront];
    if (i < f.sep_end) {
      if (j < f.sep_end) { // F11
        if (duplicate_fronts)
          return std::make_tuple(f.P0, f.P, 1);
        else
          return std::make_tuple
            (f.P0 + (((i - f.sep_begin) / B) % f.prows)
             + (((j-f.sep_begin) / B) % f.pcols) * f.prows, 1, 1);
      } else { // F12
        if (duplicate_fronts)
          return std::make_tuple(f.P0, f.P, 1);
        else
          return std::make_tuple
            (f.P0 + (((i-f.sep_begin) / B) % f.prows),
             f.prows * f.pcols, f.prows);
      }
    } else {
      if (j < f.sep_end) { // F21
        if (duplicate_fronts)
          return std::make_tuple(f.P0, f.P, 1);
        else
          return std::make_tuple
            (f.P0 + (((j - f.sep_begin) / B) % f.pcols) * f.prows,
             f.prows, 1);
      }
    }
    assert(false);
    return std::make_tuple(0, P_, 1);
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::get_all_pfronts() {
    auto nr_par_fronts = new integer_t[P_];
    nr_par_fronts[rank_] = 0;
    for (auto& f : local_pfronts_)
      if (f.P0 == rank_) nr_par_fronts[rank_]++;
    MPI_Allgather
      (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, nr_par_fronts, 1,
       mpi_type<integer_t>(), comm_.comm());
    integer_t total_pfronts = std::accumulate
      (nr_par_fronts, nr_par_fronts+P_, 0);
    all_pfronts_.resize(total_pfronts);
    auto rcnts = new int[2*P_];
    auto rdispls = rcnts + P_;
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
    delete[] nr_par_fronts;
    MPI_Allgatherv
      (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
       all_pfronts_.data(), rcnts, rdispls, MPI_BYTE, comm_.comm());
    delete[] rcnts;
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
    std::size_t lo = A.begin_row();
    std::size_t hi = A.end_row();
    std::size_t n_loc = hi - lo;
    const std::size_t B = DistM_t::default_MB;
    row_owner_.assign(n_loc, -1);
#pragma omp parallel for
    for (std::size_t r=0; r<n_loc; r++) {
      std::size_t pr = nd_.perm()[r+lo];
      for (int p=0; p<P_; p++) // local separators
        if (pr >= this->subtree_ranges[p].first &&
            pr < this->subtree_ranges[p].second) {
          row_owner_[r] = p;
          break;
        }
      if (row_owner_[r] != -1) continue;
      // distributed separators
      for (std::size_t i=0; i<all_pfronts_.size(); i++) {
        auto& f = all_pfronts_[i];
        if (pr >= f.sep_begin && pr < f.sep_end) {
          row_owner_[r] = f.P0 + (((pr - f.sep_begin) / B) % f.prows);
          break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::find_row_front
  (const CSRMatrixMPI<scalar_t,integer_t>& A) {
    row_pfront_.resize(A.size());
    for (int p=0; p<P_; p++) // local separators
      for (std::size_t r=this->subtree_ranges[p].first;
           r<this->subtree_ranges[p].second; r++)
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
   const SPOptions<scalar_t>& opts) {
    this->root_->multifrontal_factorization(Aprop_, opts);
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::multifrontal_solve_dist
  (DenseM_t& x, const std::vector<integer_t>& dist) {
    const std::size_t B = DistM_t::default_MB;
    const std::size_t lo = dist[rank_];
    const std::size_t m = dist[rank_+1] - lo;
    const std::size_t n = x.cols();
    auto ibuf = new int[4*P_];
    auto scnts = ibuf;
    auto rcnts = ibuf + P_;
    auto sdispls = ibuf + 2*P_;
    auto rdispls = ibuf + 3*P_;
    // TODO use Triplet / std::tuple
    struct RCVal { int r, c; scalar_t v; };
    auto sbuf = new RCVal[m*n];
    // since some C++ pad the struct IdxVal must zero the array or
    // will get valgrind warnings about MPI sending uninitialized data
    memset(sbuf,0,m*n*sizeof(RCVal));
    auto pp = new RCVal*[P_];
    std::fill(scnts, scnts+P_, 0);
    if (n == 1) {
      for (std::size_t r=0; r<m; r++)
        scnts[row_owner_[r]]++;
    } else {
      for (std::size_t r=0; r<m; r++) {
        auto permr = nd_.perm()[r+lo];
        int pf = row_pfront_[permr];
        if (pf < 0)
          scnts[row_owner_[r]] += n;
        else {
          auto& f = all_pfronts_[pf];
          for (std::size_t c=0; c<n; c++)
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
      for (std::size_t r=0; r<m; r++) {
        auto dest = row_owner_[r];
        pp[dest]->r = nd_.perm()[r+lo];
        pp[dest]->c = 0;
        pp[dest]->v = x(r,0);
        pp[dest]++;
      }
    } else {
      for (std::size_t r=0; r<m; r++) {
        auto destr = row_owner_[r];
        auto permr = nd_.perm()[r+lo];
        int pf = row_pfront_[permr];
        if (pf < 0) {
          for (std::size_t c=0; c<n; c++) {
            pp[destr]->r = permr;
            pp[destr]->c = c;
            pp[destr]->v = x(r,c);
            pp[destr]++;
          }
        } else {
          auto& f = all_pfronts_[pf];
          for (std::size_t c=0; c<n; c++) {
            auto dest = destr + ((c / B) % f.pcols) * f.prows;
            pp[dest]->r = permr;
            pp[dest]->c = c;
            pp[dest]->v = x(r,c);
            pp[dest]++;
          }
        }
      }
    }
    MPI_Alltoall
      (scnts, 1, mpi_type<int>(), rcnts, 1, mpi_type<int>(), comm_.comm());
    rdispls[0] = 0;
    std::size_t rsize = rcnts[0];
    for (int p=1; p<P_; p++) {
      rdispls[p] = rdispls[p-1] + rcnts[p-1];
      rsize += rcnts[p];
    }
    auto rbuf = new RCVal[rsize];
    MPI_Datatype RCVal_mpi_type;
    MPI_Type_contiguous(sizeof(RCVal), MPI_BYTE, &RCVal_mpi_type);
    MPI_Type_commit(&RCVal_mpi_type);
    MPI_Alltoallv
      (sbuf, scnts, sdispls, RCVal_mpi_type, rbuf, rcnts, rdispls,
       RCVal_mpi_type, comm_.comm());
    DenseM_t xloc(local_range_.second - local_range_.first, n);
    DenseMW_t Xloc
      (Aprop_.size(), n, xloc.data()-local_range_.first, xloc.ld());
    auto xdist = new DistM_t[local_pfronts_.size()];

    for (std::size_t f=0; f<local_pfronts_.size(); f++)
      xdist[f] = DistM_t
        (local_pfronts_[f].grid, local_pfronts_[f].dim_sep(), n);
#pragma omp parallel for
    for (std::size_t i=0; i<rsize; i++) {
      std::size_t r = rbuf[i].r;
      std::size_t c = rbuf[i].c;
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

    this->root_->multifrontal_solve(Xloc, xdist);

    rcnts = ibuf;
    scnts = ibuf + P_;
    rdispls = ibuf + 2*P_;
    sdispls = ibuf + 3*P_;
    for (int p=0; p<P_; p++)
      pp[p] = rbuf + sdispls[p];
    for (std::size_t r=local_range_.first; r<local_range_.second; r++) {
      auto dest = std::upper_bound
        (dist.begin(), dist.end(), nd_.iperm()[r])-dist.begin()-1;
      auto permgr = nd_.iperm()[r];
      for (std::size_t c=0; c<n; c++) {
        pp[dest]->r = permgr;
        pp[dest]->c = c;
        pp[dest]->v = Xloc(r,c);
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
          pp[dest]->r = permgr;
          pp[dest]->c = xdist[i].coll2g(c);
          pp[dest]->v = xdist[i](r,c);
          pp[dest]++;
        }
      }
    }
    delete[] pp;
    MPI_Alltoallv
      (rbuf, scnts, sdispls, RCVal_mpi_type, sbuf, rcnts,
       rdispls, RCVal_mpi_type, comm_.comm());
    MPI_Type_free(&RCVal_mpi_type);
    delete[] rbuf;
    delete[] ibuf;
#pragma omp parallel for
    for (std::size_t i=0; i<m*n; i++)
      x(sbuf[i].r-lo,sbuf[i].c) = sbuf[i].v;
    delete[] sbuf;
    delete[] xdist;
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
    for (integer_t r=nd_.local_tree().sizes(sep);
         r<nd_.local_tree().sizes(sep+1); r++) {
      auto ice = nd_.my_sub_graph.ind() + nd_.my_sub_graph.ptr(r+1);
      auto icb = std::lower_bound
        (nd_.my_sub_graph.ind() + nd_.my_sub_graph.ptr(r), ice, sep_end);
      auto mid = upd[sep].size();
      std::copy(icb, ice, std::back_inserter(upd[sep]));
      std::inplace_merge
        (upd[sep].begin(), upd[sep].begin() + mid, upd[sep].end());
      upd[sep].erase
        (std::unique(upd[sep].begin(), upd[sep].end()), upd[sep].end());
    }
    if (chl != -1) {
      auto icb = std::lower_bound(upd[chl].begin(), upd[chl].end(), sep_end);
      auto mid = upd[sep].size();
      std::copy(icb, upd[chl].end(), std::back_inserter(upd[sep]));
      std::inplace_merge
        (upd[sep].begin(), upd[sep].begin() + mid, upd[sep].end());
      upd[sep].erase
        (std::unique(upd[sep].begin(), upd[sep].end()), upd[sep].end());
    }
    if (chr != -1) {
      auto icb = std::lower_bound(upd[chr].begin(), upd[chr].end(), sep_end);
      auto mid = upd[sep].size();
      std::copy(icb, upd[chr].end(), std::back_inserter(upd[sep]));
      std::inplace_merge
        (upd[sep].begin(), upd[sep].begin() + mid, upd[sep].end());
      upd[sep].erase
        (std::unique(upd[sep].begin(), upd[sep].end()), upd[sep].end());
    }
    upd[sep].shrink_to_fit();
    integer_t dim_blk = (sep_end - sep_begin) + upd[sep].size();
    // assume amount of work per front is N^3, work per subtree is
    // work on front plus children
    float wl = (chl != -1) ? subtree_work[chl] : 0.;
    float wr = (chr != -1) ? subtree_work[chr] : 0.;
    subtree_work[sep] = (float(dim_blk)*dim_blk*dim_blk) + wl + wr;
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

    /* initialize dsep_work so valgrind does not complain, as it is
       not always set below */
    dsep_work = 0.0;
    nd_.my_dist_sep.sort_rows();
    std::vector<MPI_Request> send_req;
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
        send_req.emplace_back();
        int tag = (dsep == nd_.tree().lch(pa)) ? 1 : 2;
        MPI_Isend
          (local_upd[nd_.local_tree().root()].data(),
           local_upd[nd_.local_tree().root()].size(),
           mpi_type<integer_t>(), pa_rank, tag, comm_.comm(),
           &send_req.back());
        dsep_work = local_subtree_work[nd_.local_tree().root()];
        send_req.emplace_back();
        MPI_Isend(&dsep_work, 1, MPI_FLOAT, pa_rank, tag+2,
                  comm_.comm(), &send_req.back());
      } else {
        auto sep_begin = nd_.dist_sep_range.first;
        auto sep_end = nd_.dist_sep_range.second;
        for (integer_t r=0; r<sep_end-sep_begin; r++) {
          auto ice = nd_.my_dist_sep.ind() + nd_.my_dist_sep.ptr(r+1);
          auto icb = std::lower_bound
            (nd_.my_dist_sep.ind() + nd_.my_dist_sep.ptr(r), ice, sep_end);
          auto mid = dist_upd.size();
          std::copy(icb, ice, std::back_inserter(dist_upd));
          std::inplace_merge
            (dist_upd.begin(), dist_upd.begin() + mid, dist_upd.end());
          dist_upd.erase
            (std::unique(dist_upd.begin(), dist_upd.end()), dist_upd.end());
        }

        auto chl = nd_.proc_dist_sep[nd_.tree().lch(dsep)];
        auto chr = nd_.proc_dist_sep[nd_.tree().rch(dsep)];
        // receive dist_upd from left child
        MPI_Status stat;
        int msg_size;
        // TODO probe both left and right, take the first
        MPI_Probe(chl, 1, comm_.comm(), &stat);
        MPI_Get_count(&stat, mpi_type<integer_t>(), &msg_size);
        std::vector<integer_t> dist_upd_lch(msg_size);
        MPI_Recv(dist_upd_lch.data(), msg_size, mpi_type<integer_t>(), chl, 1,
                 comm_.comm(), &stat);
        // merge dist_upd from left child into dist_upd
        auto icb = std::lower_bound
          (dist_upd_lch.begin(), dist_upd_lch.end(), sep_end);
        auto mid = dist_upd.size();
        std::copy(icb, dist_upd_lch.end(), std::back_inserter(dist_upd));
        std::inplace_merge
          (dist_upd.begin(), dist_upd.begin() + mid, dist_upd.end());
        dist_upd.erase
          (std::unique(dist_upd.begin(), dist_upd.end()), dist_upd.end());

        // receive dist_upd from right child
        MPI_Probe(chr, 2, comm_.comm(), &stat);
        MPI_Get_count(&stat, mpi_type<integer_t>(), &msg_size);
        std::vector<integer_t> dist_upd_rch(msg_size);
        MPI_Recv
          (dist_upd_rch.data(), msg_size, mpi_type<integer_t>(),
           chr, 2, comm_.comm(), &stat);
        // merge dist_upd from right child into dist_upd
        icb = std::lower_bound
          (dist_upd_rch.begin(), dist_upd_rch.end(), sep_end);
        mid = dist_upd.size();
        std::copy(icb, dist_upd_rch.end(), std::back_inserter(dist_upd));
        std::inplace_merge
          (dist_upd.begin(), dist_upd.begin() + mid, dist_upd.end());
        dist_upd.erase
          (std::unique(dist_upd.begin(), dist_upd.end()), dist_upd.end());

        // receive work estimates for left and right subtrees
        float dsep_left_work, dsep_right_work;
        MPI_Recv(&dsep_left_work,  1, MPI_FLOAT, chl, 3, comm_.comm(), &stat);
        MPI_Recv(&dsep_right_work, 1, MPI_FLOAT, chr, 4, comm_.comm(), &stat);
        integer_t dim_blk = (sep_end - sep_begin) + dist_upd.size();
        dsep_work = (float(dim_blk)*dim_blk*dim_blk) + dsep_left_work +
          dsep_right_work;

        // send dist_upd and work estimate to parent
        if (nd_.tree().pa(pa) != -1) {
          // do not send to parent if parent is root
          send_req.emplace_back();
          int tag = (dsep == nd_.tree().lch(pa)) ? 1 : 2;
          MPI_Isend
            (dist_upd.data(), dist_upd.size(), mpi_type<integer_t>(),
             pa_rank, tag, comm_.comm(), &send_req.back());
          send_req.emplace_back();
          MPI_Isend
            (&dsep_work, 1, MPI_FLOAT, pa_rank, tag+2,
             comm_.comm(), &send_req.back());
        }
      }
    }
    MPI_Waitall(send_req.size(), send_req.data(), MPI_STATUSES_IGNORE);
  }

  /**
   * Send the distributed separator from the process responsible for it,
   * to all the processes working on the frontal matrix corresponding to
   * it, and to all processes working on the sibling front (needed for
   * extend-add).  Hence this routine only needs to be called by those
   * processes (or simply by everyone in comm_).
   */
  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::
  communicate_distributed_separator
  (integer_t dsep, std::vector<integer_t>& dist_upd,
   integer_t& dsep_begin, integer_t& dsep_end,
   std::vector<integer_t>& dsep_upd, int P0, int P,
   int P0_sibling, int P_sibling, int owner, bool use_hss) {
    integer_t* sbuf = NULL;
    std::vector<MPI_Request> sreq;
    int dest0 = std::min(P0, P0_sibling);
    int dest1 = std::max(P0+P, P0_sibling+P_sibling);
    if (rank_ == owner) {
      sbuf = new integer_t[2+dist_upd.size()];
      sbuf[0] = nd_.dist_sep_range.first;
      sbuf[1] = nd_.dist_sep_range.second;
      std::copy(dist_upd.begin(), dist_upd.end(), sbuf+2);
      if (use_hss) sreq.resize(P_);
      else sreq.resize(dest1-dest0);
      int msg = 0;
      for (int dest=dest0; dest<dest1; dest++)
        MPI_Isend(sbuf, 2 + dist_upd.size(), mpi_type<integer_t>(),
                  dest, 0, comm_.comm(), &sreq[msg++]);
      if (use_hss) {
        // when using HSS compression, every process needs to know the
        // size of this front, because you need to know if your parent
        // is HSS
        for (int dest=0; dest<dest0; dest++)
          MPI_Isend(sbuf, 2, mpi_type<integer_t>(), dest, 0,
                    comm_.comm(), &sreq[msg++]);
        for (int dest=dest1; dest<P_; dest++)
          MPI_Isend(sbuf, 2, mpi_type<integer_t>(), dest, 0,
                    comm_.comm(), &sreq[msg++]);
      }
    }
    if (rank_ >= dest0 && rank_ < dest1) {
      MPI_Status stat;
      MPI_Probe(owner, 0, comm_.comm(), &stat);
      int msg_size;
      MPI_Get_count(&stat, mpi_type<integer_t>(), &msg_size);
      auto rbuf = new integer_t[msg_size];
      MPI_Recv(rbuf, msg_size, mpi_type<integer_t>(),
               owner, 0, comm_.comm(), &stat);
      dsep_begin = rbuf[0];
      dsep_end = rbuf[1];
      dsep_upd.assign(rbuf+2, rbuf+msg_size);
      delete[] rbuf;
    } else if (use_hss) {
      integer_t rbuf[2];
      MPI_Recv(rbuf, 2, mpi_type<integer_t>(), owner,
               0, comm_.comm(), MPI_STATUS_IGNORE);
      dsep_begin = rbuf[0];
      dsep_end = rbuf[1];
    }
    if (rank_ == owner) {
      MPI_Waitall(sreq.size(), sreq.data(), MPI_STATUSES_IGNORE);
      delete[] sbuf;
    }
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::communicate_hss_tree
  (HSS::HSSPartitionTree& hss_tree, integer_t dsep, int P0, int P, int owner) {
    std::vector<MPI_Request> sreq;
    std::vector<int> sbuf;
    if (rank_ == owner) {
      sbuf = nd_.tree().HSS_tree(dsep).serialize();
      sreq.resize(P);
      for (int dest=P0; dest<P0+P; dest++)
        MPI_Isend(sbuf.data(), sbuf.size(), MPI_INT, dest, 0,
                  comm_.comm(), &sreq[dest-P0]);
    }
    if (rank_ >= P0 && rank_ < P0+P) {
      MPI_Status stat;
      MPI_Probe(owner, 0, comm_.comm(), &stat);
      int msg_size;
      MPI_Get_count(&stat, MPI_INT, &msg_size);
      std::vector<int> rbuf(msg_size);
      MPI_Recv(rbuf.data(), rbuf.size(), MPI_INT, owner, 0,
               comm_.comm(), &stat);
      hss_tree = HSS::HSSPartitionTree(rbuf);
    }
    if (rank_ == owner)
      MPI_Waitall(sreq.size(), sreq.data(), MPI_STATUSES_IGNORE);
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::communicate_admissibility
  (std::vector<bool>& adm, integer_t dsep, int P0, int P, int owner) {
    std::vector<MPIRequest> sreq;
    std::vector<int> sbuf;
    if (rank_ == owner) {
      auto& a = nd_.tree().admissibility(dsep);
      sbuf.assign(a.begin(), a.end());
      for (int dest=P0; dest<P0+P; dest++)
        sreq.emplace_back(comm_.isend(sbuf, dest, 0));
    }
    if (rank_ >= P0 && rank_ < P0+P) {
      auto rbuf = comm_.template recv<int>(owner, 0);
      adm.assign(rbuf.begin(), rbuf.end());
    }
    if (rank_ == owner)
      wait_all(sreq);
  }

  template<typename scalar_t,typename integer_t>
  std::unique_ptr<FrontalMatrix<scalar_t,integer_t>>
  EliminationTreeMPIDist<scalar_t,integer_t>::proportional_mapping
  (const SPOptions<scalar_t>& opts,
   std::vector<std::vector<integer_t>>& local_upd,
   std::vector<integer_t>& dist_upd, std::vector<float>& local_subtree_work,
   std::vector<float>& dist_subtree_work, integer_t dsep,
   int P0, int P, int P0_sibling, int P_sibling,
   const MPIComm& fcomm, bool hss_parent, int level) {
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
         fcomm, hss_parent, level);
    }

    integer_t dsep_begin, dsep_end;
    std::vector<integer_t> dsep_upd;
    communicate_distributed_separator
      (dsep, dist_upd, dsep_begin, dsep_end, dsep_upd,
       P0, P, P0_sibling, P_sibling, owner, opts.use_HSS() && hss_parent);
    auto dim_dsep = dsep_end - dsep_begin;
    bool is_hss = opts.use_HSS() && hss_parent &&
      (dim_dsep >= opts.HSS_min_sep_size());
    bool is_blr = opts.use_BLR() && (dim_dsep >= opts.BLR_min_sep_size());
    HSS::HSSPartitionTree hss_tree(dim_dsep);
    if (is_hss || is_blr)
      communicate_hss_tree(hss_tree, dsep, P0, P, owner);
    std::vector<bool> adm;
    if (is_blr) communicate_admissibility(adm, dsep, P0, P, owner);

    // bool is_hss = opts.use_HSS() && (dim_dsep >= opts.HSS_min_sep_size()) &&
    //   (dim_dsep + dsep_upd.size() >= opts.HSS_min_front_size());
    // // HSS::HSSPartitionTree hss_tree(dim_dsep);
    // // if (is_hss) communicate_distributed_separator_HSS_tree(hss_tree, dsep, P0, P, owner);

    if (rank_ == P0) {
      if (is_hss) this->nr_HSS_fronts_++;
      else if (is_blr) this->nr_BLR_fronts_++;
      else this->nr_dense_fronts_++;
    }
    std::unique_ptr<F_t> front;
    // only store fronts you work on and their siblings (needed for
    // extend-add operation)
    if ((rank_ >= P0 && rank_ < P0+P) ||
        (rank_ >= P0_sibling && rank_ < P0_sibling+P_sibling)) {
      if (P == 1) {
        if (is_hss) {
          front = std::unique_ptr<F_t>
            (new FHSS_t(dsep, dsep_begin, dsep_end, dsep_upd));
          front->set_HSS_partitioning(opts, hss_tree, level == 0);
        } else if (is_blr) {
          front = std::unique_ptr<F_t>
            (new FBLR_t(dsep, dsep_begin, dsep_end, dsep_upd));
          front->set_BLR_partitioning(opts, hss_tree, adm, level == 0);
        } else
          front = std::unique_ptr<F_t>
            (new FD_t(dsep, dsep_begin, dsep_end, dsep_upd));
        if (P0 == rank_) {
          local_range_.first = std::min
            (local_range_.first, std::size_t(dsep_begin));
          local_range_.second = std::max
            (local_range_.second, std::size_t(dsep_end));
        }
      } else {
        if (is_hss) {
          front = std::unique_ptr<F_t>
            (new FHSSMPI_t(local_pfronts_.size(), dsep_begin, dsep_end,
                           dsep_upd, fcomm, P));
          front->set_HSS_partitioning(opts, hss_tree, level == 0);
        } else {
          if (is_blr) {
            front = std::unique_ptr<F_t>
              (new FBLRMPI_t(local_pfronts_.size(), dsep_begin, dsep_end,
                             dsep_upd, fcomm, P));
            front->set_BLR_partitioning(opts, hss_tree, adm, level == 0);
          } else
            front = std::unique_ptr<F_t>
              (new FDMPI_t(local_pfronts_.size(), dsep_begin, dsep_end,
                           dsep_upd, fcomm, P));
        }
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
       chl, P0, Pl, P0+P-Pr, Pr, fcomm.sub(0, Pl), is_hss, level+1);
    auto rch = proportional_mapping
      (opts, local_upd, dist_upd, local_subtree_work, dist_subtree_work,
       chr, P0+P-Pr, Pr, P0, Pl, fcomm.sub(P-Pr, Pr), is_hss, level+1);
    if (front) {
      front->set_lchild(std::move(lch));
      front->set_rchild(std::move(rch));
    }
    return front;
  }

  /** This should only be called by [P0,P0+P) and
      [P0_sibling,P0_sibling+P_sibling) */
  template<typename scalar_t,typename integer_t>
  std::unique_ptr<FrontalMatrix<scalar_t,integer_t>>
  EliminationTreeMPIDist<scalar_t,integer_t>::proportional_mapping_sub_graphs
  (const SPOptions<scalar_t>& opts, RedistSubTree<integer_t>& tree,
   integer_t dsep, integer_t sep, int P0, int P,
   int P0_sibling, int P_sibling, const MPIComm& fcomm,
   bool hss_parent, int level) {
    if (!tree.nr_sep) return nullptr;
    auto sep_begin = tree.sep_ptr[sep];
    auto sep_end = tree.sep_ptr[sep+1];
    auto dim_sep = sep_end - sep_begin;
    auto dim_upd = tree.dim_upd[sep];
    std::vector<integer_t> upd(tree.upd[sep], tree.upd[sep]+dim_upd);
    std::unique_ptr<F_t> front;

    bool is_hss = opts.use_HSS() && hss_parent &&
      (dim_sep >= opts.HSS_min_sep_size());
    // bool is_hss = opts.use_HSS() && (dim_sep >= opts.HSS_min_sep_size()) &&
    //   (dim_sep + dim_upd >= opts.HSS_min_front_size());
    bool is_blr = opts.use_BLR() && (dim_sep >= opts.BLR_min_sep_size());

    if (rank_ == P0) {
      if (is_hss) this->nr_HSS_fronts_++;
      else if (is_blr) this->nr_BLR_fronts_++;
      else this->nr_dense_fronts_++;
    }
    if ((rank_ >= P0 && rank_ < P0+P) ||
        (rank_ >= P0_sibling && rank_ < P0_sibling+P_sibling)) {
      if (P == 1) {
        if (is_hss) {
          front = std::unique_ptr<F_t>(new FHSS_t(sep, sep_begin, sep_end, upd));
          front->set_HSS_partitioning(opts, tree.HSS_tree[sep], level == 0);
        } else if (is_blr) {
          front = std::unique_ptr<F_t>(new FBLR_t(sep, sep_begin, sep_end, upd));
          front->set_BLR_partitioning
            (opts, tree.HSS_tree[sep], tree.admissibility[sep], level == 0);
        } else front = std::unique_ptr<F_t>(new FD_t(sep, sep_begin, sep_end, upd));
        if (P0 == rank_) {
          local_range_.first = std::min
            (local_range_.first, std::size_t(sep_begin));
          local_range_.second = std::max
            (local_range_.second, std::size_t(sep_end));
        }
      } else {
        if (is_hss) {
          front = std::unique_ptr<F_t>
            (new FHSSMPI_t
             (local_pfronts_.size(), sep_begin, sep_end, upd, fcomm, P));
          front->set_HSS_partitioning(opts, tree.HSS_tree[sep], level == 0);
        } else {
          if (is_blr) {
            front = std::unique_ptr<F_t>
              (new FBLRMPI_t
               (local_pfronts_.size(), sep_begin, sep_end, upd, fcomm, P));
            front->set_BLR_partitioning
              (opts, tree.HSS_tree[sep], tree.admissibility[sep], level == 0);
          } else
            front = std::unique_ptr<F_t>
              (new FDMPI_t
               (local_pfronts_.size(), sep_begin, sep_end, upd, fcomm, P));
        }
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
      front->set_lchild
        (proportional_mapping_sub_graphs
         (opts, tree, dsep, chl, P0, Pl, P0+P-Pr, Pr,
          fcomm.sub(0, Pl), is_hss, level+1));
      front->set_rchild
        (proportional_mapping_sub_graphs
         (opts, tree, dsep, chr, P0+P-Pr, Pr, P0, Pl,
          fcomm.sub(P-Pr, Pr), is_hss, level+1));
    }
    return front;
  }

} // end namespace strumpack

#endif
