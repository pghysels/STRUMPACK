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
#include <stack>

#include "EliminationTreeMPIDist.hpp"
#include "Redistribute.hpp"
#include "CSRMatrixMPI.hpp"
#include "PropMapSparseMatrix.hpp"
#include "ordering/MatrixReorderingMPI.hpp"
#include "fronts/FrontalMatrix.hpp"
#include "fronts/FrontalMatrixMPI.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> MPI_Datatype
  EliminationTreeMPIDist<scalar_t,integer_t>::ParallelFront::pf_mpi_type = MPI_DATATYPE_NULL;

  template<typename scalar_t,typename integer_t>
  EliminationTreeMPIDist<scalar_t,integer_t>::EliminationTreeMPIDist
  (const Opts_t& opts, const CSRMPI_t& A, Reord_t& nd, const MPIComm& comm)
    : EliminationTreeMPI<scalar_t,integer_t>(comm), nd_(nd) {

    std::vector<std::vector<integer_t>> lupd(nd_.ltree().separators());
    // every process is responsible for 1 distributed separator, so
    // store only 1 dist_upd
    std::vector<integer_t> dist_upd, dleaf_upd;
    std::vector<float> ltree_work(nd_.ltree().separators()),
      dtree_work(nd_.tree().separators());

    float dsep_work, dleaf_work;
    MPIComm::control_start("symbolic_factorization");
    prop_map_ = opts.proportional_mapping();
    symb_fact(lupd, ltree_work, dist_upd, dsep_work, dleaf_upd, dleaf_work);
    MPIComm::control_stop("symbolic_factorization");

    {
      auto ndseps = nd_.tree().separators();
      // communicate dtree_work to everyone
      std::vector<float> sbuf(2*P_);
      for (integer_t dsep=0; dsep<ndseps; dsep++)
        if (rank_ == nd_.proc_dist_sep[dsep]) {
          sbuf[2*rank_] = dleaf_work;
          sbuf[2*rank_+1] = dsep_work;
        }
      comm_.all_gather(sbuf.data(), 2);
      for (integer_t dsep=0; dsep<ndseps; dsep++) {
        auto i = 2 * nd_.proc_dist_sep[dsep];
        dtree_work[dsep] = nd_.tree().is_leaf(dsep) ? sbuf[i] : sbuf[i+1];
      }
    }

    local_range_ = {A.size(), 0};
    MPIComm::control_start("proportional_mapping");
    this->root_ = prop_map
      (opts, lupd, ltree_work, dist_upd, dleaf_upd, dtree_work,
       nd_.tree().root(), 0, P_, 0, 0, comm_, true, 0);
    MPIComm::control_stop("proportional_mapping");

    MPIComm::control_start("block_row_A_to_prop_A");
    if (local_range_.first > local_range_.second)
      local_range_.first = local_range_.second = 0;
    subtree_ranges_.resize(P_);
    subtree_ranges_[rank_] = local_range_;
    comm_.all_gather(subtree_ranges_.data(), 1);

    get_all_pfronts();
    find_row_front(A);
    find_row_owner(A);
    Aprop_.setup(A, nd_, *this, opts.compression() != CompressionType::NONE);
    MPIComm::control_stop("block_row_A_to_prop_A");
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::update_values
  (const Opts_t& opts, const CSRMPI_t& A, Reord_t& nd) {
    Aprop_.setup(A, nd_, *this, opts.compression() != CompressionType::NONE);
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::separator_reordering
  (const Opts_t& opts, const CSRMPI_t& A) {
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
  (const CSRMPI_t& A, integer_t oi, integer_t oj,
   integer_t i, integer_t j, bool duplicate_fronts) const {
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
      if (f.P0 == rank_)
        nr_par_fronts[rank_]++;
    comm_.all_gather(nr_par_fronts, 1);
    int total_pfronts = std::accumulate(nr_par_fronts, nr_par_fronts+P_, 0);
    all_pfronts_.resize(total_pfronts);
    rdispls[0] = 0;
    for (int p=0; p<P_; p++)
      rcnts[p] = nr_par_fronts[p];
    for (int p=1; p<P_; p++)
      rdispls[p] = rdispls[p-1] + rcnts[p-1];
    {
      int i = rdispls[rank_];
      for (auto& f : local_pfronts_)
        if (f.P0 == rank_)
          all_pfronts_[i++] = f;
    }
    comm_.all_gather_v(all_pfronts_.data(), rcnts, rdispls);
    ParallelFront::free_mpi_type();
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
  (const CSRMPI_t& A) {
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
  (const CSRMPI_t& A) {
    row_pfront_.resize(A.size());
    for (int p=0; p<P_; p++) // local separators
      for (integer_t r=subtree_ranges_[p].first;
           r<subtree_ranges_[p].second; r++)
        row_pfront_[r] = -p-1;
    for (std::size_t i=0; i<all_pfronts_.size(); i++) {
      auto& f = all_pfronts_[i];
      for (integer_t r=f.sep_begin; r<f.sep_end; r++)
        row_pfront_[r] = i;
    }
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  EliminationTreeMPIDist<scalar_t,integer_t>::multifrontal_factorization
  (const CompressedSparseMatrix<scalar_t,integer_t>& A,
   const Opts_t& opts) {
    return this->root_->multifrontal_factorization(Aprop_, opts);
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::multifrontal_solve_dist
  (DenseM_t& x, const std::vector<integer_t>& dist) {
    integer_t B = DistM_t::default_MB;
    integer_t lo = dist[rank_];
    integer_t m = dist[rank_+1] - lo, n = x.cols();

    std::unique_ptr<int[]> iwork(new int[4*P_]);
    auto ibuf = iwork.get();
    auto scnts = ibuf;
    auto rcnts = ibuf + P_;
    auto sdispls = ibuf + 2*P_;
    auto rdispls = ibuf + 3*P_;

    using Triplet = Triplet<scalar_t,integer_t>;
    std::vector<Triplet> sbuf(m*n);
    std::fill(scnts, scnts+P_, 0);
    if (n == 1)
      for (integer_t r=0; r<m; r++)
        scnts[row_owner_[r]]++;
    else {
      for (integer_t r=0; r<m; r++) {
        int pf = row_pfront_[nd_.perm()[r+lo]];
        if (pf < 0) scnts[row_owner_[r]] += n;
        else {
          auto& f = all_pfronts_[pf];
          for (integer_t c=0; c<n; c++)
            scnts[row_owner_[r]+((c/B)%f.pcols)*f.prows]++;
        }
      }
    }
    std::vector<std::size_t> pp(P_);
    sdispls[0] = 0;
    pp[0] = 0;
    for (int p=1; p<P_; p++)
      pp[p] = sdispls[p] = sdispls[p-1] + scnts[p-1];
    if (n == 1) {
      const auto perm = nd_.perm().data() + lo;
      for (integer_t r=0; r<m; r++)
        sbuf[pp[row_owner_[r]]++] = {perm[r], 0, x(r,0)};
    } else {
      for (integer_t r=0; r<m; r++) {
        auto destr = row_owner_[r];
        auto permr = nd_.perm()[r+lo];
        int pf = row_pfront_[permr];
        if (pf < 0)
          for (integer_t c=0; c<n; c++)
            sbuf[pp[destr]++] = {permr, c, x(r,c)};
        else {
          auto& f = all_pfronts_[pf];
          for (integer_t c=0; c<n; c++)
            sbuf[pp[destr+((c/B)%f.pcols)*f.prows]++] =
              {permr, c, x(r,c)};
        }
      }
    }
    comm_.all_to_all(scnts, 1, rcnts);
    rdispls[0] = 0;
    for (int p=1; p<P_; p++)
      rdispls[p] = rdispls[p-1] + rcnts[p-1];

    auto rbuf = comm_.all_to_allv
      (sbuf.data(), scnts, sdispls, rcnts, rdispls);

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
      pp[p] = sdispls[p];
    for (integer_t r=local_range_.first; r<local_range_.second; r++) {
      auto dest = std::upper_bound
        (dist.begin(), dist.end(), nd_.iperm()[r])-dist.begin()-1;
      auto permgr = nd_.iperm()[r];
      for (integer_t c=0; c<n; c++)
        rbuf[pp[dest]++] = {permgr, c, Xloc(r,c)};
    }
    for (std::size_t i=0; i<local_pfronts_.size(); i++) {
      if (xdist[i].lcols() == 0) continue;
      auto slo = local_pfronts_[i].sep_begin;
      for (int r=0; r<xdist[i].lrows(); r++) {
        auto permgr = nd_.iperm()[xdist[i].rowl2g(r) + slo];
        auto dest = std::upper_bound
          (dist.begin(), dist.end(), permgr)-dist.begin()-1;
        for (int c=0; c<xdist[i].lcols(); c++)
          rbuf[pp[dest]++] = {permgr, xdist[i].coll2g(c), xdist[i](r,c)};
      }
    }
    comm_.all_to_allv
      (rbuf.data(), scnts, sdispls, sbuf.data(), rcnts, rdispls);

#pragma omp parallel for
    for (std::size_t i=0; i<std::size_t(m)*n; i++)
      x(sbuf[i].r-lo,sbuf[i].c) = sbuf[i].v;

    Triplet::free_mpi_type();
  }

  template<typename integer_t, typename It>
  void merge_if_larger(const It u0, const It u1,
                       std::vector<integer_t>& out,
                       integer_t s) {
    auto b = std::lower_bound(u0, u1, s);
    auto m = out.size();
    std::copy(b, u1, std::back_inserter(out));
    std::inplace_merge(out.begin(), out.begin() + m, out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
  }

  template<typename scalar_t,typename integer_t> float
  EliminationTreeMPIDist<scalar_t,integer_t>::symb_fact_loc
  (integer_t sep, std::vector<std::vector<integer_t>>& upd,
   std::vector<float>& work, int depth) {
    auto chl = nd_.ltree().lch[sep];
    auto chr = nd_.ltree().rch[sep];
    float fs = 0.;
    if (depth < params::task_recursion_cutoff_level) {
      float fl = 0., fr = 0.;
      if (chl != -1)
#pragma omp task untied default(shared)                                 \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        fl = symb_fact_loc(chl, upd, work, depth+1);
      if (chr != -1)
#pragma omp task untied default(shared)                                 \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        fr = symb_fact_loc(chr, upd, work, depth+1);
#pragma omp taskwait
      fs += fl + fr;
    } else {
      if (chl != -1)
        fs += symb_fact_loc(chl, upd, work, depth);
      if (chr != -1)
        fs += symb_fact_loc(chr, upd, work, depth);
    }
    auto sep_begin = nd_.ltree().sizes[sep] +
      nd_.sub_graph_range.first;
    auto sep_end = nd_.ltree().sizes[sep+1] +
      nd_.sub_graph_range.first;
    {
      auto lor = nd_.ltree().sizes[sep];
      auto hir = nd_.ltree().sizes[sep+1];
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
    float d1 = sep_end - sep_begin, d2 = upd[sep].size();
    fs += d1*(d1 + 2.*d2); // factor size up to and including this node
    switch (prop_map_) {
    case ProportionalMapping::FLOPS: {
      //          getrf            + 2 * trsm     + gemm
      work[sep] = 2.0/3.0*d1*d1*d1 + 2.0*d1*d1*d2 + 2.0*d2*d2*d1;
      if (chl != -1) work[sep] += work[chl];
      if (chr != -1) work[sep] += work[chr];
    } break;
    case ProportionalMapping::FACTOR_MEMORY: {
      work[sep] = fs;
    } break;
    case ProportionalMapping::PEAK_MEMORY: {
      work[sep] = fs + d2*d2; // memory for F22
      if (chl != -1) {
        auto d2l = upd[chl].size();
        work[sep] += d2l*d2l; // memory for F22 of left child
      }
      if (chr != -1) {
        auto d2r = upd[chr].size();
        work[sep] += d2r*d2r; // memory for F22 of right child
      }
      if (chl != -1) work[sep] = std::max(work[sep], work[chl]);
      if (chr != -1) work[sep] = std::max(work[sep], work[chr]);
    } break;
    }
    return fs;
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
  EliminationTreeMPIDist<scalar_t,integer_t>::symb_fact
  (std::vector<std::vector<integer_t>>& local_upd,
   std::vector<float>& local_subtree_work,
   std::vector<integer_t>& dist_upd, float& dsep_work,
   std::vector<integer_t>& dleaf_upd, float& dleaf_work) {
    nd_.my_sub_graph.sort_rows();

    float fs = 0.;
    if (!nd_.ltree().is_empty()) {
#pragma omp parallel
#pragma omp single
      fs = symb_fact_loc
        (nd_.ltree().root(), local_upd, local_subtree_work, 0);
    }

    dsep_work = dleaf_work = 0.;
    nd_.my_dist_sep.sort_rows();
    std::vector<MPI_Request> sreq;
    for (integer_t dsep=0; dsep<nd_.tree().separators(); dsep++) {
      // only consider the distributed separator owned by this
      // process: 1 leaf and 1 non-leaf
      if (nd_.proc_dist_sep[dsep] != rank_) continue;
      if (nd_.tree().is_root(dsep)) continue; // skip the root separator
      auto pa = nd_.tree().parent[dsep];
      auto pa_rank = nd_.proc_dist_sep[pa];
      if (nd_.tree().is_leaf(dsep)) {
        // Leaf of distributed tree is local subgraph for process
        // proc_dist_sep[dsep], unless the ltree is empty.
        if (!nd_.ltree().is_empty()) {
          dleaf_work = local_subtree_work[nd_.ltree().root()];
          dleaf_upd = local_upd[nd_.ltree().root()];
        } else {
          auto sep_begin = nd_.sub_graph_range.first;
          auto sep_end = nd_.sub_graph_range.second;
          for (integer_t r=0; r<sep_end-sep_begin; r++)
            merge_if_larger
              (nd_.my_sub_graph.ind() + nd_.my_sub_graph.ptr(r),
               nd_.my_sub_graph.ind() + nd_.my_sub_graph.ptr(r+1),
               dleaf_upd, sep_end);
          float d1 = sep_end - sep_begin, d2 = dist_upd.size();
          fs = d1*(d1 + 2.0*d2);
          switch (prop_map_) {
          case ProportionalMapping::FLOPS: {
            dleaf_work = 2.0/3.0*d1*d1*d1 + 2.0*d1*d1*d2 + 2.0*d2*d2*d1;
          } break;
          case ProportionalMapping::FACTOR_MEMORY: {
            dleaf_work = fs;
          } break;
          case ProportionalMapping::PEAK_MEMORY: {
            dleaf_work = fs + d2*d2;
          } break;
          }
        }
        // do not send to parent if parent is root
        if (nd_.tree().is_root(pa)) continue;
        sreq.emplace_back();
        comm_.isend(dleaf_upd, pa_rank, 1, &sreq.back());
        sreq.emplace_back();
        comm_.isend(dleaf_work, pa_rank, 2, &sreq.back());
        sreq.emplace_back();
        comm_.isend(fs, pa_rank, 3, &sreq.back());
      } else {
        auto sep_begin = nd_.dist_sep_range.first;
        auto sep_end = nd_.dist_sep_range.second;
        for (integer_t r=0; r<sep_end-sep_begin; r++)
          merge_if_larger
            (nd_.my_dist_sep.ind() + nd_.my_dist_sep.ptr(r),
             nd_.my_dist_sep.ind() + nd_.my_dist_sep.ptr(r+1),
             dist_upd, sep_end);
        int nr_children = (nd_.tree().is_leaf(dsep) &&
                           nd_.ltree().is_empty()) ? 0 : 2;
        std::vector<integer_t> d2ch(nr_children);
        std::vector<float> wch(nr_children);
        float fs = 0.;
        for (int i=0; i<nr_children; i++) {
          // receive dist_upd from left/right child,
          auto du = comm_.template recv_any_src<integer_t>(1);
          d2ch[i] = du.second.size();
          // then merge elements larger than sep_end
          merge_if_larger
            (du.second.begin(), du.second.end(), dist_upd, sep_end);
          wch[i] = comm_.template recv_one<float>(du.first, 2);
          fs += comm_.template recv_one<float>(du.first, 3);
        }
        float d1 = sep_end - sep_begin, d2 = dist_upd.size();
        fs += d1*(d1 + 2.*d2);   // factor size up to and including this node
        switch (prop_map_) {
        case ProportionalMapping::FLOPS: {
          //          getrf            + 2 * trsm     + gemm
          dsep_work = 2.0/3.0*d1*d1*d1 + 2.0*d1*d1*d2 + 2.0*d2*d2*d1;
          for (int i=0; i<nr_children; i++)
            dsep_work += wch[i];
        } break;
        case ProportionalMapping::FACTOR_MEMORY: {
          dsep_work = fs;
        } break;
        case ProportionalMapping::PEAK_MEMORY: {
          dsep_work = fs + d2*d2;  // storage for F22
          for (int i=0; i<nr_children; i++)
            dsep_work += d2ch[i]*d2ch[i];  // memory for child F22
          for (int i=0; i<nr_children; i++)
            dsep_work = std::max(dsep_work, wch[i]); // mem peak at children
        } break;
        }
        if (!nd_.tree().is_root(pa)) {
          sreq.emplace_back();    // send dist_upd to parent
          comm_.isend(dist_upd, pa_rank, 1, &sreq.back());
          // TODO combine in a single message
          sreq.emplace_back();    // send work estimate to parent
          comm_.isend(dsep_work, pa_rank, 2, &sreq.back());
          sreq.emplace_back();    // send current factor size to parent
          comm_.isend(fs, pa_rank, 3, &sreq.back());
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
   */
  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPIDist<scalar_t,integer_t>::
  comm_dist_sep(integer_t dsep, const std::vector<integer_t>& dupd_send,
                integer_t& dsep_begin, integer_t& dsep_end,
                std::vector<integer_t>& dupd_recv,
                int P0, int P, int P0_sib, int P_sib, int owner) {
    std::vector<integer_t> sbuf;
    std::vector<MPIRequest> sreq;
    int dest0 = std::min(P0, P0_sib),
      dest1 = std::max(P0+P, P0_sib+P_sib);
    if (rank_ == owner) {
      sbuf.reserve(2+dupd_send.size());
      sbuf.push_back(dsep_begin);
      sbuf.push_back(dsep_end);
      sbuf.insert(sbuf.end(), dupd_send.begin(), dupd_send.end());
      sreq.reserve(dest1-dest0);
      for (int dest=dest0; dest<dest1; dest++)
        if (dest != rank_)
          sreq.emplace_back(comm_.isend(sbuf, dest, dsep));
    }
    if (rank_ >= dest0 && rank_ < dest1) {
      if (rank_ == owner)
        dupd_recv = dupd_send;
      else {
        auto rbuf = comm_.template recv<integer_t>(owner, dsep);
        dsep_begin = rbuf[0];
        dsep_end = rbuf[1];
        dupd_recv.assign(rbuf.begin()+2, rbuf.end());
      }
    }
    if (rank_ == owner) wait_all(sreq);
  }


  template<typename scalar_t,typename integer_t>
  std::unique_ptr<FrontalMatrix<scalar_t,integer_t>>
  EliminationTreeMPIDist<scalar_t,integer_t>::prop_map
  (const Opts_t& opts, std::vector<std::vector<integer_t>>& local_upd,
   std::vector<float>& local_subtree_work,
   std::vector<integer_t>& dist_upd, std::vector<integer_t>& dleaf_upd,
   std::vector<float>& dist_subtree_work, integer_t dsep,
   int P0, int P, int P0_sib, int P_sib,
   const MPIComm& fcomm, bool pa_comp, int level) {
    auto owner = nd_.proc_dist_sep[dsep];
    if (nd_.tree().is_leaf(dsep)) {
      RedistSubTree<integer_t> sub_tree
        (nd_.ltree(), dsep, nd_.sub_graph_range.first,
         local_upd, local_subtree_work,
         P0, P, P0_sib, P_sib, owner, comm_);
      if (!sub_tree.nr_sep) return nullptr;
      return prop_map_sub_graphs
        (opts, sub_tree, P0, P, P0_sib, P_sib,
         fcomm, pa_comp, level);
    }
    std::vector<integer_t> dsep_upd;
    integer_t dsep_begin = nd_.dist_sep_range.first,
      dsep_end = nd_.dist_sep_range.second;
    comm_dist_sep(dsep, dist_upd, dsep_begin, dsep_end, dsep_upd,
                  P0, P, P0_sib, P_sib, owner);
    auto dim_dsep = dsep_end - dsep_begin;
    bool use_compression = is_compressed
      (dim_dsep, dsep_upd.size(), pa_comp, opts);
    std::unique_ptr<F_t> front;
    // only store fronts you work on and their siblings (needed for
    // extend-add operation)
    if ((rank_ >= P0 && rank_ < P0+P) ||
        (rank_ >= P0_sib && rank_ < P0_sib+P_sib)) {
      if (P == 1) {
        front = create_frontal_matrix<scalar_t,integer_t>
          (opts, dsep, dsep_begin, dsep_end, dsep_upd, pa_comp,
           level, this->nr_fronts_, rank_ == P0);
        if (P0 == rank_) this->update_local_ranges(dsep_begin, dsep_end);
      } else {
        auto fmpi = create_frontal_matrix<scalar_t,integer_t>
          (opts, local_pfronts_.size(), dsep_begin, dsep_end, dsep_upd,
           pa_comp, level, this->nr_fronts_, fcomm, P, rank_ == P0);
        if (rank_ >= P0 && rank_ < P0+P)
          local_pfronts_.emplace_back
            (dsep_begin, dsep_end, P0, P, fmpi->grid());
        front = std::move(fmpi);
      }
    }

    // here we should still continue, to send the local subgraph
    auto chl = nd_.tree().lch[dsep];
    auto chr = nd_.tree().rch[dsep];
    auto wl = dist_subtree_work[chl];
    auto wr = dist_subtree_work[chr];
    int Pl = std::max(1, std::min(int(std::round(P * wl / (wl + wr))), P-1));
    int Pr = std::max(1, P - Pl);
    auto cl = fcomm.sub(0, Pl);
    auto cr = fcomm.sub(P-Pr, Pr);
    auto lch = prop_map
      (opts, local_upd, local_subtree_work, dist_upd, dleaf_upd,
       dist_subtree_work, chl, P0, Pl, P0+P-Pr, Pr, cl,
       use_compression, level+1);
    auto rch = prop_map
      (opts, local_upd, local_subtree_work, dist_upd, dleaf_upd,
       dist_subtree_work, chr, P0+P-Pr, Pr, P0, Pl, cr,
       use_compression, level+1);
    if (front) {
      front->set_lchild(std::move(lch));
      front->set_rchild(std::move(rch));
    }
    return front;
  }


  /**
   * This should only be called by [P0,P0+P) and
   * [P0_sib,P0_sib+P_sib)
   */
  template<typename scalar_t,typename integer_t>
  std::unique_ptr<FrontalMatrix<scalar_t,integer_t>>
  EliminationTreeMPIDist<scalar_t,integer_t>::prop_map_sub_graphs
  (const Opts_t& opts, const RedistSubTree<integer_t>& tree,
   int P0, int P, int P0_sib, int P_sib,
   const MPIComm& fcomm, bool pa_comp, int level) {
    struct MapData {
      integer_t sep;
      int P0, P, P0_sib, P_sib;
      const MPIComm *pfcomm;
      MPIComm fcomm;
      bool pa_comp;
      int level;
      F_t* parent;
      bool left;
    };
    std::stack<MapData> fstack;
    fstack.emplace
      (MapData{tree.root, P0, P, P0_sib, P_sib,
         &fcomm, MPI_COMM_NULL, pa_comp, level, nullptr, true});
    std::unique_ptr<F_t> froot;
    while (!fstack.empty()) {
      auto m = fstack.top();
      fstack.pop();
      auto sep_beg = tree.sep_ptr[m.sep];
      auto sep_end = tree.sep_ptr[m.sep+1];
      auto dim_sep = sep_end - sep_beg;
      auto dim_upd = tree.dim_upd[m.sep];
      std::unique_ptr<F_t> front;
      const MPIComm* pcomm = m.pfcomm ? m.pfcomm : &m.fcomm;
      if ((rank_ >= m.P0 && rank_ < m.P0+m.P) ||
          (rank_ >= m.P0_sib && rank_ < m.P0_sib+m.P_sib)) {
        std::vector<integer_t> upd(tree.upd[m.sep], tree.upd[m.sep]+dim_upd);
        if (m.P == 1) {
          front = create_frontal_matrix<scalar_t,integer_t>
            (opts, m.sep, sep_beg, sep_end, upd, m.pa_comp,
             m.level, this->nr_fronts_, rank_ == m.P0);
          if (m.P0 == rank_) this->update_local_ranges(sep_beg, sep_end);
        } else {
          auto fmpi = create_frontal_matrix<scalar_t,integer_t>
            (opts, local_pfronts_.size(), sep_beg, sep_end, upd,
             m.pa_comp, m.level, this->nr_fronts_,
             *pcomm, m.P, rank_ == m.P0);
          if (rank_ >= m.P0 && rank_ < m.P0+m.P)
            local_pfronts_.emplace_back
              (sep_beg, sep_end, m.P0, m.P, fmpi->grid());
          front = std::move(fmpi);
        }
      }
      if (m.P0 <= rank_ && rank_ < m.P0+m.P) {
        auto chl = tree.lchild[m.sep];
        auto chr = tree.rchild[m.sep];
        if (chl != -1 && chr != -1) {
          bool comp = is_compressed
            (dim_sep, dim_upd, m.pa_comp, opts);
          if (m.P == 1) {
            fstack.emplace
              (MapData{chl, m.P0, 1, m.P0, 1, pcomm, MPI_COMM_NULL,
                 comp, m.level+1, front.get(), true});
            fstack.emplace
              (MapData{chr, m.P0, 1, m.P0, 1, pcomm, MPI_COMM_NULL,
                 comp, m.level+1, front.get(), false});
          } else {
            auto wl = tree.work[chl];
            auto wr = tree.work[chr];
            int Pl = std::max
              (1, std::min(int(std::round(m.P * wl / (wl + wr))), m.P-1));
            int Pr = std::max(1, m.P - Pl);
            fstack.emplace
              (MapData{chl, m.P0, Pl, m.P0+m.P-Pr, Pr,
                 nullptr, pcomm->sub(0, Pl),
                 comp, m.level+1, front.get(), true});
            fstack.emplace
              (MapData{chr, m.P0+m.P-Pr, Pr, m.P0, Pl,
                 nullptr, pcomm->sub(m.P-Pr, Pr),
                 comp, m.level+1, front.get(), false});
          }
        }
      }
      if (!m.parent) froot = std::move(front);
      else {
        if (m.left) m.parent->set_lchild(std::move(front));
        else m.parent->set_rchild(std::move(front));
      }
    }
    return froot;
  }

  // explicit template specializations
  template class EliminationTreeMPIDist<float,int>;
  template class EliminationTreeMPIDist<double,int>;
  template class EliminationTreeMPIDist<std::complex<float>,int>;
  template class EliminationTreeMPIDist<std::complex<double>,int>;

  template class EliminationTreeMPIDist<float,long int>;
  template class EliminationTreeMPIDist<double,long int>;
  template class EliminationTreeMPIDist<std::complex<float>,long int>;
  template class EliminationTreeMPIDist<std::complex<double>,long int>;

  template class EliminationTreeMPIDist<float,long long int>;
  template class EliminationTreeMPIDist<double,long long int>;
  template class EliminationTreeMPIDist<std::complex<float>,long long int>;
  template class EliminationTreeMPIDist<std::complex<double>,long long int>;

} // end namespace strumpack
