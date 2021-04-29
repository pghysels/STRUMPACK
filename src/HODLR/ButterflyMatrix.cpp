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
#include <cassert>
#include <algorithm>

#include "misc/TaskTimer.hpp"
#include "misc/Tools.hpp"
#include "ButterflyMatrix.hpp"
#include "HODLRWrapper.hpp"

namespace strumpack {
  namespace HODLR {

    template<typename scalar_t> ButterflyMatrix<scalar_t>::ButterflyMatrix
    (const MPIComm& comm, const structured::ClusterTree& row_tree,
     const structured::ClusterTree& col_tree, const opts_t& opts)
      : ButterflyMatrix<scalar_t>
      (HODLRMatrix<scalar_t>(comm, row_tree, opts),
       HODLRMatrix<scalar_t>(comm, col_tree, opts)) {
    }

    template<typename scalar_t> ButterflyMatrix<scalar_t>::ButterflyMatrix
    (const HODLRMatrix<scalar_t>& A, const HODLRMatrix<scalar_t>& B)
      : c_(A.c_) {
      rows_ = A.rows();
      cols_ = B.cols();
      Fcomm_ = A.Fcomm_;
      int P = c_->size();
      std::vector<int> groups(P);
      std::iota(groups.begin(), groups.end(), 0);
      // create hodlr data structures
      HODLR_createptree<scalar_t>(P, groups.data(), Fcomm_, ptree_);
      HODLR_createstats<scalar_t>(stats_);
      F2Cptr Aoptions = const_cast<F2Cptr>(A.options_);
      HODLR_copyoptions<scalar_t>(Aoptions, options_);
      HODLR_set_I_option<scalar_t>(options_, "nogeo", 1);
      HODLR_set_I_option<scalar_t>(options_, "knn", 0);
      LRBF_construct_init<scalar_t>
        (rows_, cols_, lrows_, lcols_, nullptr, nullptr, A.msh_, B.msh_,
         lr_bf_, options_, stats_, msh_, kerquant_, ptree_,
         nullptr, nullptr, nullptr);
      set_dist();
    }

    template<typename scalar_t> double
    ButterflyMatrix<scalar_t>::get_stat(const std::string& name) const {
      if (!stats_) return 0;
      return BPACK_get_stat<scalar_t>(stats_, name);
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::print_stats() {
      if (!stats_) return;
      HODLR_printstats<scalar_t>(stats_, ptree_);
    }

    template<typename integer_t> struct AdmInfoButterfly {
      std::pair<std::vector<int>,std::vector<int>> rmaps, cmaps;
      const DenseMatrix<bool>* adm;
    };

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::options_init(const opts_t& opts) {
      auto P = c_->size();
      std::vector<int> groups(P);
      std::iota(groups.begin(), groups.end(), 0);

      // create hodlr data structures
      HODLR_createptree<scalar_t>(P, groups.data(), Fcomm_, ptree_);
      HODLR_createoptions<scalar_t>(options_);
      HODLR_createstats<scalar_t>(stats_);

      // set hodlr options
      HODLR_set_I_option<scalar_t>(options_, "verbosity", opts.verbose() ? 2 : -2);
      // HODLR_set_I_option<scalar_t>(options_, "Nbundle", 8);
      // HODLR_set_I_option<scalar_t>(options_, "rmax", 10000);
      HODLR_set_I_option<scalar_t>(options_, "nogeo", 1);
      HODLR_set_I_option<scalar_t>(options_, "Nmin_leaf", rows_);
      // set RecLR_leaf to 2 for RRQR at bottom level of Hierarchical BACA
      HODLR_set_I_option<scalar_t>(options_, "RecLR_leaf", opts.lr_leaf()); // 5 = new version of BACA
      HODLR_set_I_option<scalar_t>(options_, "BACA_Batch", opts.BACA_block_size());
      HODLR_set_I_option<scalar_t>(options_, "xyzsort", 0);
      HODLR_set_I_option<scalar_t>(options_, "elem_extract", 1); // block extraction
      // set ErrFillFull to 1 to check acc for extraction code
      //HODLR_set_I_option<scalar_t>(options_, "ErrFillFull", opts.verbose() ? 1 : 0);
      HODLR_set_I_option<scalar_t>(options_, "ErrFillFull", 0);
      HODLR_set_I_option<scalar_t>(options_, "rank0", opts.rank_guess());
      HODLR_set_I_option<scalar_t>(options_, "less_adapt", opts.less_adapt()); // 0 or 1
      HODLR_set_I_option<scalar_t>(options_, "forwardN15flag", opts.BF_entry_n15()); // 0 or 1
      HODLR_set_I_option<scalar_t>(options_, "cpp", 1);
      HODLR_set_D_option<scalar_t>(options_, "sample_para", opts.BF_sampling_parameter());
      HODLR_set_D_option<scalar_t>(options_, "sample_para_outer", opts.BF_sampling_parameter());
      HODLR_set_D_option<scalar_t>(options_, "rankrate", opts.rank_rate());
      if (opts.butterfly_levels() > 0)
        HODLR_set_I_option<scalar_t>(options_, "LRlevel", opts.butterfly_levels());
      HODLR_set_D_option<scalar_t>(options_, "tol_comp", opts.rel_tol());
      HODLR_set_D_option<scalar_t>(options_, "tol_rand", opts.rel_tol());
      HODLR_set_D_option<scalar_t>(options_, "tol_Rdetect", 0.1*opts.rel_tol());
    }

    /**
     * This is not used now, but could be useful for the H format.
     */
    template<typename integer_t> void Butterfly_admissibility_query
    (int* m, int* n, int* admissible, C2Fptr fdata) {
      auto& info = *static_cast<AdmInfoButterfly<integer_t>*>(fdata);
      auto& adm = *(info.adm);
      auto& rmap0 = info.rmaps.first;
      auto& rmap1 = info.rmaps.second;
      auto& cmap0 = info.cmaps.first;
      auto& cmap1 = info.cmaps.second;
      int r = LRBF_treeindex_merged2child(*m);
      int c = LRBF_treeindex_merged2child(*n);
      if (r < 0) {
        r = -r;
        std::swap(r, c);
      } else c = -c;
      r--;
      c--;
      assert(r < int(rmap0.size()) && r < int(rmap1.size()));
      assert(c < int(cmap0.size()) && c < int(cmap1.size()));
      bool a = true;
      for (int j=cmap0[c]; j<=cmap1[c] && a; j++)
        for (int i=rmap0[r]; i<=rmap1[r] && a; i++)
          a = a && adm(i, j);
      *admissible = a;
    }

    template<typename integer_t> DenseMatrix<int> get_odiag_neighbors
    (int knn, const CSRGraph<integer_t>& gAB,
     const CSRGraph<integer_t>& gA, const CSRGraph<integer_t>& gB) {
      TIMER_TIME(TaskType::NEIGHBOR_SEARCH, 0, t_knn);
      int rows = gA.size(), cols = gB.size();
      DenseMatrix<int> nns(knn, rows);
      nns.fill(0);
      int B = std::ceil(rows / params::num_threads);
#pragma omp parallel for schedule(static, 1)
      for (int lo=0; lo<rows; lo+=B) {
        std::vector<bool> rmark(rows), cmark(cols);
        std::vector<int> rq(rows), cq(cols);
        for (int i=lo; i<std::min(lo+B, rows); i++) {
          int rqfront = 0, rqback = 0, cqfront = 0, cqback = 0, nn = 0;
          rq[rqback++] = i;
          rmark[i] = true;
          while (nn < knn && (rqfront < rqback || cqfront < cqback)) {
            if (rqfront < rqback) {
              auto k = rq[rqfront++];
              auto hi = gAB.ind() + gAB.ptr(k+1);
              for (auto pl=gAB.ind()+gAB.ptr(k); pl!=hi; pl++) {
                auto l = *pl;
                if (!cmark[l]) {
                  nns(nn++, i) = l+1;
                  if (nn == knn) break;
                  cmark[l] = true;
                  cq[cqback++] = l;
                }
              }
              if (nn == knn) break;
              hi = gA.ind() + gA.ptr(k+1);
              for (auto pl=gA.ind()+gA.ptr(k); pl!=hi; pl++) {
                auto l = *pl;
                if (!rmark[l]) {
                  rmark[l] = true;
                  rq[rqback++] = l;
                }
              }
            }
            if (cqfront < cqback) {
              auto k = cq[cqfront++];
              auto hi = gB.ind() + gB.ptr(k+1);
              for (auto pl=gB.ind()+gB.ptr(k); pl!=hi; pl++) {
                auto l = *pl;
                if (!cmark[l]) {
                  nns(nn++, i) = l+1;
                  if (nn == knn) break;
                  cmark[l] = true;
                  cq[cqback++] = l;
                }
              }
            }
          }
          for (int l=0; l<cqback; l++) cmark[cq[l]] = false;
          for (int l=0; l<rqback; l++) rmark[rq[l]] = false;
        }
      }
      return nns;
    }

    template<typename scalar_t> ButterflyMatrix<scalar_t>::ButterflyMatrix
    (const HODLRMatrix<scalar_t>& A, const HODLRMatrix<scalar_t>& B,
     DenseMatrix<int>& neighbors_rows, DenseMatrix<int>& neighbors_cols,
     const opts_t& opts) : c_(A.c_) {
      rows_ = A.rows();
      cols_ = B.cols();
      Fcomm_ = A.Fcomm_;
      int P = c_->size();
      std::vector<int> groups(P);
      std::iota(groups.begin(), groups.end(), 0);
      // create hodlr data structures
      HODLR_createptree<scalar_t>(P, groups.data(), Fcomm_, ptree_);
      HODLR_createstats<scalar_t>(stats_);
      F2Cptr Aoptions = const_cast<F2Cptr>(A.options_);
      HODLR_copyoptions<scalar_t>(Aoptions, options_);
      assert(neighbors_rows.rows() == std::size_t(opts.knn_lrbf()) &&
             neighbors_rows.cols() == rows() &&
             neighbors_cols.rows() == std::size_t(opts.knn_lrbf()) &&
             neighbors_cols.cols() == cols());
      HODLR_set_D_option<scalar_t>(options_, "tol_comp", 0.1*opts.rel_tol());
      HODLR_set_D_option<scalar_t>(options_, "tol_rand", 0.1*opts.rel_tol());
      HODLR_set_D_option<scalar_t>(options_, "tol_Rdetect", 0.01*opts.rel_tol());
      HODLR_set_I_option<scalar_t>(options_, "nogeo", 3);
      HODLR_set_I_option<scalar_t>(options_, "knn", opts.knn_lrbf());
      HODLR_set_I_option<scalar_t>(options_, "forwardN15flag", opts.BF_entry_n15()); // 0 or 1
      HODLR_set_I_option<scalar_t>(options_, "RecLR_leaf", opts.lr_leaf());
      { TIMER_TIME(TaskType::CONSTRUCT_INIT, 0, t_construct_h);
        LRBF_construct_init<scalar_t>
          (rows_, cols_, lrows_, lcols_,
           neighbors_rows.data(), neighbors_cols.data(),
           A.msh_, B.msh_, lr_bf_, options_, stats_, msh_,
           kerquant_, ptree_, nullptr, nullptr, nullptr);
      }
      set_dist();
    }

    template<typename scalar_t> void ButterflyMatrix<scalar_t>::set_dist() {
      int P = c_->size(), rank = c_->rank();
      rdist_.resize(P+1);
      cdist_.resize(P+1);
      rdist_[rank+1] = lrows_;
      cdist_[rank+1] = lcols_;
      c_->all_gather(rdist_.data()+1, 1);
      c_->all_gather(cdist_.data()+1, 1);
      for (int p=0; p<P; p++) {
        rdist_[p+1] += rdist_[p];
        cdist_[p+1] += cdist_[p];
      }
    }

    template<typename scalar_t> ButterflyMatrix<scalar_t>::~ButterflyMatrix() {
      if (stats_) HODLR_deletestats<scalar_t>(stats_);
      if (ptree_) HODLR_deleteproctree<scalar_t>(ptree_);
      if (msh_) HODLR_deletemesh<scalar_t>(msh_);
      if (kerquant_) HODLR_deletekernelquant<scalar_t>(kerquant_);
      if (options_) HODLR_deleteoptions<scalar_t>(options_);
      if (lr_bf_) LRBF_deletebf<scalar_t>(lr_bf_);
    }

    template<typename scalar_t> ButterflyMatrix<scalar_t>&
    ButterflyMatrix<scalar_t>::operator=(ButterflyMatrix<scalar_t>&& h) {
      lr_bf_ = h.lr_bf_;       h.lr_bf_ = nullptr;
      options_ = h.options_;   h.options_ = nullptr;
      stats_ = h.stats_;       h.stats_ = nullptr;
      msh_ = h.msh_;           h.msh_ = nullptr;
      kerquant_ = h.kerquant_; h.kerquant_ = nullptr;
      ptree_ = h.ptree_;       h.ptree_ = nullptr;
      Fcomm_ = h.Fcomm_;
      c_ = h.c_;
      rows_ = h.rows_;
      cols_ = h.cols_;
      lrows_ = h.lrows_;
      lcols_ = h.lcols_;
      std::swap(rdist_, h.rdist_);
      std::swap(cdist_, h.cdist_);
      return *this;
    }

    template<typename scalar_t> void LRBF_matvec_routine
    (const char* op, int* nin, int* nout, int* nvec,
     const scalar_t* X, scalar_t* Y, C2Fptr func, scalar_t* a, scalar_t* b) {
      auto A = static_cast<typename ButterflyMatrix<scalar_t>::mult_t*>(func);
      DenseMatrixWrapper<scalar_t> Yw(*nout, *nvec, Y, *nout),
        Xw(*nin, *nvec, const_cast<scalar_t*>(X), *nin);
      (*A)(c2T(*op), *a, Xw, *b, Yw);
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::compress(const mult_t& Amult) {
      C2Fptr f = static_cast<void*>(const_cast<mult_t*>(&Amult));
      LRBF_construct_matvec_compute
        (lr_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(LRBF_matvec_routine<scalar_t>), f);
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::compress(const mult_t& Amult, int rank_guess) {
      HODLR_set_I_option<scalar_t>(options_, "rank0", rank_guess);
      compress(Amult);
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::compress(const delem_blocks_t& Aelem) {
      BLACSGrid gloc(MPIComm(MPI_COMM_SELF), 1),
        gnull(MPIComm(MPI_COMM_NULL), 1);
      AelemCommPtrs<scalar_t> AC{&Aelem, c_, &gloc, &gnull};
      LRBF_construct_element_compute<scalar_t>
        (lr_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(HODLR_block_evaluation<scalar_t>), &AC);
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::compress(const elem_blocks_t& Aelem) {
      C2Fptr f = static_cast<void*>(const_cast<elem_blocks_t*>(&Aelem));
      LRBF_construct_element_compute<scalar_t>
        (lr_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(HODLR_block_evaluation_seq<scalar_t>), f);
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::mult
    (Trans op, const DenseM_t& X, DenseM_t& Y) const {
      assert(Y.cols() == X.cols());
      if (op == Trans::N)
        LRBF_mult(char(op), X.data(), Y.data(), lcols_, lrows_, X.cols(),
                  lr_bf_, options_, stats_, ptree_);
      else
        LRBF_mult(char(op), X.data(), Y.data(), lrows_, lcols_, X.cols(),
                  lr_bf_, options_, stats_, ptree_);
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::mult
    (Trans op, const DistM_t& X, DistM_t& Y) const {
      DenseM_t Y1D(lrows_, X.cols());
      if (op == Trans::N) {
        auto X1D = redistribute_2D_to_1D(X, cdist_);
        LRBF_mult(char(op), X1D.data(), Y1D.data(), lcols_, lrows_,
                  X1D.cols(), lr_bf_, options_, stats_, ptree_);
        redistribute_1D_to_2D(Y1D, Y, rdist_);
      } else {
        auto X1D = redistribute_2D_to_1D(X, rdist_);
        LRBF_mult(char(op), X1D.data(), Y1D.data(), lrows_, lcols_,
                  X1D.cols(), lr_bf_, options_, stats_, ptree_);
        redistribute_1D_to_2D(Y1D, Y, cdist_);
      }
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::set_sampling_parameter(double sample_param) {
      HODLR_set_D_option<scalar_t>(options_, "sample_para", sample_param);
      HODLR_set_D_option<scalar_t>(options_, "sample_para_outer", sample_param);
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::set_BACA_block(int bsize) {
      HODLR_set_I_option<scalar_t>(options_, "BACA_Batch", bsize);
    }


    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::extract_add_elements
    (ExtractionMeta& e, std::vector<DistMW_t>& B) {
      std::unique_ptr<scalar_t[]> alldat_loc(new scalar_t[e.Nalldat_loc]);
      auto ptr = alldat_loc.get();
      LRBF_extract_elements<scalar_t>
        (lr_bf_, options_, msh_, stats_, ptree_, e.Ninter,
         e.Nallrows, e.Nallcols, e.Nalldat_loc, e.allrows, e.allcols,
         ptr, e.rowids, e.colids, e.pgids, e.Npmap, e.pmaps);
      for (auto& Bk : B) {
        auto m = Bk.lcols();
        auto n = Bk.lrows();
        auto Bdata = Bk.data();
        auto Bld = Bk.ld();
        for (int j=0; j<m; j++)
          for (int i=0; i<n; i++)
            Bdata[i+j*Bld] += *ptr++;
      }
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::extract_add_elements
    (const VecVec_t& I, const VecVec_t& J, std::vector<DenseMW_t>& B) {
      if (I.empty()) return;
      assert(I.size() == J.size() && I.size() == B.size());
      ExtractionMeta e;
      int pmaps[3] = {1, 1, 0};
      int Nalldat_loc = 0;
      for (auto& Bk : B) Nalldat_loc += Bk.rows() * Bk.cols();
      set_extraction_meta_1grid(I, J, e, Nalldat_loc, pmaps);
      // extract_add_elements(I, J, B, e);
      std::unique_ptr<scalar_t[]> alldat_loc(new scalar_t[e.Nalldat_loc]);
      auto ptr = alldat_loc.get();
      LRBF_extract_elements<scalar_t>
        (lr_bf_, options_, msh_, stats_, ptree_, e.Ninter,
         e.Nallrows, e.Nallcols, e.Nalldat_loc, e.allrows, e.allcols,
         ptr, e.rowids, e.colids, e.pgids, e.Npmap, e.pmaps);
      for (auto& Bk : B) {
        auto m = Bk.cols();
        auto n = Bk.rows();
        auto Bdata = Bk.data();
        auto Bld = Bk.ld();
        for (std::size_t j=0; j<m; j++)
          for (std::size_t i=0; i<n; i++)
            Bdata[i+j*Bld] += *ptr++;
      }
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::extract_add_elements
    (ExtractionMeta& e, std::vector<DenseMW_t>& B) {
      std::unique_ptr<scalar_t[]> alldat_loc(new scalar_t[e.Nalldat_loc]);
      auto ptr = alldat_loc.get();
      LRBF_extract_elements<scalar_t>
        (lr_bf_, options_, msh_, stats_, ptree_, e.Ninter,
         e.Nallrows, e.Nallcols, e.Nalldat_loc, e.allrows, e.allcols,
         ptr, e.rowids, e.colids, e.pgids, e.Npmap, e.pmaps);
      for (auto& Bk : B) {
        auto m = Bk.cols();
        auto n = Bk.rows();
        auto Bdata = Bk.data();
        auto Bld = Bk.ld();
        for (std::size_t j=0; j<m; j++)
          for (std::size_t i=0; i<n; i++)
            Bdata[i+j*Bld] += *ptr++;
      }
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::set_extraction_meta_1grid
    (const VecVec_t& I, const VecVec_t& J,
     ExtractionMeta& e, int Nalldat_loc, int* pmaps) const {
      e.Ninter = I.size();
      e.Npmap = 1;
      e.pmaps = pmaps;
      e.Nalldat_loc = Nalldat_loc;
      e.Nallrows = e.Nallcols = e.Nalldat_loc = 0;
      for (auto Ik : I) e.Nallrows += Ik.size();
      for (auto Jk : J) e.Nallcols += Jk.size();
      e.iwork.reset(new int[e.Nallrows + e.Nallcols + 3*e.Ninter]);
      e.allrows = e.iwork.get();
      e.allcols = e.allrows + e.Nallrows;
      e.rowids = e.allcols + e.Nallcols;
      e.colids = e.rowids + e.Ninter;
      e.pgids = e.colids + e.Ninter;
      for (int k=0, i=0, j=0; k<e.Ninter; k++) {
        e.rowids[k] = I[k].size();
        e.colids[k] = J[k].size();
        e.pgids[k] = 0;
        for (auto l : I[k]) { assert(l < rows()); e.allrows[i++] = l+1; }
        for (auto l : J[k]) { assert(l < cols()); e.allcols[j++] = l+1; }
      }
    }

    template<typename scalar_t> DistributedMatrix<scalar_t>
    ButterflyMatrix<scalar_t>::dense(const BLACSGrid* g) const {
      DistM_t A(g, rows_, cols_), I(g, rows_, cols_);
      I.eye();
      mult(Trans::N, I, A);
      return A;
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    ButterflyMatrix<scalar_t>::redistribute_2D_to_1D
    (const DistM_t& R2D, const std::vector<int>& dist) const {
      const auto rank = c_->rank();
      DenseM_t R1D(dist[rank+1] - dist[rank], R2D.cols());
      redistribute_2D_to_1D(scalar_t(1.), R2D, scalar_t(0.), R1D, dist);
      return R1D;
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::redistribute_2D_to_1D
    (scalar_t a, const DistM_t& R2D, scalar_t b, DenseM_t& R1D,
     const std::vector<int>& dist) const {
      TIMER_TIME(TaskType::REDIST_2D_TO_HSS, 0, t_redist);
      const auto P = c_->size();
      const auto rank = c_->rank();
      // for (int p=0; p<P; p++)
      //   copy(dist[rank+1]-dist[rank], R2D.cols(), R2D, dist[rank], 0,
      //        R1D, p, R2D.grid()->ctxt_all());
      // return;
      const auto Rcols = R2D.cols();
      int R2Drlo, R2Drhi, R2Dclo, R2Dchi;
      R2D.lranges(R2Drlo, R2Drhi, R2Dclo, R2Dchi);
      const int Rlcols = R2Dchi - R2Dclo;
      const int Rlrows = R2Drhi - R2Drlo;
      const int nprows = R2D.nprows();
      const int lrows = R1D.rows();
      assert(lrows == dist[rank+1] - dist[rank]);
      std::vector<std::vector<scalar_t>> sbuf(P);
      if (R2D.active()) {
        // global, local, proc
        std::vector<std::tuple<int,int,int>> glp(Rlrows);
        {
          std::vector<std::size_t> count(P);
          for (int r=R2Drlo; r<R2Drhi; r++) {
            auto gr = R2D.rowl2g(r);
            auto p = -1 + std::distance
              (dist.begin(), std::upper_bound
               (dist.begin(), dist.end(), gr));
            glp[r-R2Drlo] = std::tuple<int,int,int>{gr, r, p};
            assert(p >= 0 && p < P);
            count[p] += Rlcols;
          }
          std::sort(glp.begin(), glp.end());
          for (int p=0; p<P; p++)
            sbuf[p].reserve(count[p]);
        }
        for (int r=R2Drlo; r<R2Drhi; r++)
          for (int c=R2Dclo, lr=std::get<1>(glp[r-R2Drlo]),
                 p=std::get<2>(glp[r-R2Drlo]); c<R2Dchi; c++)
            sbuf[p].push_back(R2D(lr,c));
      }
      std::vector<scalar_t,NoInit<scalar_t>> rbuf;
      std::vector<scalar_t*> pbuf;
      c_->all_to_all_v(sbuf, rbuf, pbuf);
      if (lrows) {
        std::vector<int> src_c(Rcols);
        for (int c=0; c<Rcols; c++)
          src_c[c] = R2D.colg2p_fixed(c)*nprows;
        if (b == scalar_t(0.)) {
          // don't assume that 0 * Nan == 0
          for (int r=0; r<lrows; r++)
            for (int c=0, src_r=R2D.rowg2p_fixed(r+dist[rank]); c<Rcols; c++)
              R1D(r, c) = *(pbuf[src_r + src_c[c]]++) * a;
        } else {
          for (int r=0; r<lrows; r++)
            for (int c=0, src_r=R2D.rowg2p_fixed(r+dist[rank]); c<Rcols; c++)
              R1D(r, c) = *(pbuf[src_r + src_c[c]]++) * a + b * R1D(r, c);
        }
      }
    }

    template<typename scalar_t> void
    ButterflyMatrix<scalar_t>::redistribute_1D_to_2D
    (const DenseM_t& S1D, DistM_t& S2D, const std::vector<int>& dist) const {
      TIMER_TIME(TaskType::REDIST_2D_TO_HSS, 0, t_redist);
      const int rank = c_->rank();
      const int P = c_->size();
      const int cols = S1D.cols();
      int S2Drlo, S2Drhi, S2Dclo, S2Dchi;
      S2D.lranges(S2Drlo, S2Drhi, S2Dclo, S2Dchi);
      const int nprows = S2D.nprows();
      const int lrows = dist[rank+1] - dist[rank];
      std::vector<std::vector<scalar_t>> sbuf(P);
      if (lrows) {
        std::vector<std::tuple<int,int,int>> glp(lrows);
        for (int r=0; r<lrows; r++) {
          auto gr = r + dist[rank];
          glp[r] = std::tuple<int,int,int>{gr,r,S2D.rowg2p_fixed(gr)};
        }
        std::sort(glp.begin(), glp.end());
        std::vector<int> pc(cols);
        for (int c=0; c<cols; c++)
          pc[c] = S2D.colg2p_fixed(c)*nprows;
        {
          std::vector<std::size_t> count(P);
          for (int r=0; r<lrows; r++)
            for (int c=0, pr=std::get<2>(glp[r]); c<cols; c++)
              count[pr+pc[c]]++;
          for (int p=0; p<P; p++)
            sbuf[p].reserve(count[p]);
        }
        for (int r=0; r<lrows; r++)
          for (int c=0, lr=std::get<1>(glp[r]),
                 pr=std::get<2>(glp[r]); c<cols; c++)
            sbuf[pr+pc[c]].push_back(S1D(lr,c));
      }
      std::vector<scalar_t,NoInit<scalar_t>> rbuf;
      std::vector<scalar_t*> pbuf;
      c_->all_to_all_v(sbuf, rbuf, pbuf);
      if (S2D.active()) {
        for (int r=S2Drlo; r<S2Drhi; r++) {
          auto gr = S2D.rowl2g(r);
          auto p = -1 + std::distance
            (dist.begin(), std::upper_bound
             (dist.begin(), dist.end(), gr));
          for (int c=S2Dclo; c<S2Dchi; c++)
            S2D(r,c) = *(pbuf[p]++);
        }
      }
    }

    // explicit instantiations
    template class ButterflyMatrix<float>;
    template class ButterflyMatrix<double>;
    template class ButterflyMatrix<std::complex<float>>;
    template class ButterflyMatrix<std::complex<double>>;

    template DenseMatrix<int>
    get_odiag_neighbors(int knn, const CSRGraph<int>& gAB,
                        const CSRGraph<int>& gA,
                        const CSRGraph<int>& gB);
    template DenseMatrix<int>
    get_odiag_neighbors(int knn, const CSRGraph<long>& gAB,
                        const CSRGraph<long>& gA,
                        const CSRGraph<long>& gB);
    template DenseMatrix<int>
    get_odiag_neighbors(int knn, const CSRGraph<long long int>& gAB,
                        const CSRGraph<long long int>& gA,
                        const CSRGraph<long long int>& gB);

  } // end namespace HODLR
} // end namespace strumpack
