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
/*! \file LRBFMatrix.hpp
 * \brief Classes wrapping around Yang Liu's butterfly code.
 */
#ifndef STRUMPACK_LRBF_MATRIX_HPP
#define STRUMPACK_LRBF_MATRIX_HPP

#include <cassert>

#include "HSS/HSSPartitionTree.hpp"
#include "dense/DistributedMatrix.hpp"
#include "HODLROptions.hpp"
#include "HODLRWrapper.hpp"

namespace strumpack {

  /**
   * Code in this namespace is a wrapper aroung Yang Liu's Fortran
   * code: https://github.com/liuyangzhuan/hod-lr-bf
   */
  namespace HODLR {

    template<typename scalar_t> class LRBFMatrix {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using DistM_t = DistributedMatrix<scalar_t>;
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      using opts_t = HODLROptions<scalar_t>;
      using VecVec_t = std::vector<std::vector<std::size_t>>;
      using delem_blocks_t = typename HODLRMatrix<scalar_t>::delem_blocks_t;
      using elem_blocks_t = typename HODLRMatrix<scalar_t>::elem_blocks_t;

    public:
      using mult_t = typename std::function
        <void(Trans,scalar_t,const DenseM_t&,scalar_t,DenseM_t&)>;

      LRBFMatrix() {}
      /**
       * Construct the block X, subblock of the matrix [A X; Y B]
       * A and B should be defined on the same MPI communicator.
       */
      LRBFMatrix
      (const HODLRMatrix<scalar_t>& A, const HODLRMatrix<scalar_t>& B);

      template<typename integer_t> LRBFMatrix
      (const HODLRMatrix<scalar_t>& A, const HSS::HSSPartitionTree& Atree,
       const CSRGraph<integer_t>& Agraph,
       const HODLRMatrix<scalar_t>& B, const HSS::HSSPartitionTree& Btree,
       const CSRGraph<integer_t>& Bgraph,
       const DenseMatrix<bool>& admissibility,
       const CSRGraph<integer_t>& graph, const opts_t& opts);

      LRBFMatrix(const LRBFMatrix<scalar_t>& h) = delete;
      LRBFMatrix(LRBFMatrix<scalar_t>&& h) { *this = h; }
      virtual ~LRBFMatrix();
      LRBFMatrix<scalar_t>& operator=(const LRBFMatrix<scalar_t>& h) = delete;
      LRBFMatrix<scalar_t>& operator=(LRBFMatrix<scalar_t>&& h);

      std::size_t rows() const { return rows_; }
      std::size_t cols() const { return cols_; }
      std::size_t lrows() const { return lrows_; }
      std::size_t lcols() const { return lcols_; }
      std::size_t begin_row() const { return rdist_[c_.rank()]; }
      std::size_t end_row() const { return rdist_[c_.rank()+1]; }
      const std::vector<int>& rdist() const { return rdist_; }
      std::size_t begin_col() const { return cdist_[c_.rank()]; }
      std::size_t end_col() const { return cdist_[c_.rank()+1]; }
      const std::vector<int>& cdist() const { return cdist_; }
      const MPIComm& Comm() const { return c_; }

      void compress(const mult_t& Amult);
      void compress(const mult_t& Amult, int rank_guess);
      void compress(const delem_blocks_t& Aelem);
      void compress(const elem_blocks_t& Aelem);

      void mult(Trans op, const DenseM_t& X, DenseM_t& Y) const;

      /**
       * Multiply this low-rank (or butterfly) matrix with a dense
       * matrix: Y = op(A) * X, where op can be none,
       * transpose or complex conjugate. X and Y are in 2D block
       * cyclic distribution.
       *
       * \param op Transpose, conjugate, or none.
       * \param X Right-hand side matrix. Should be X.rows() ==
       * this.rows().
       * \param Y Result, should be Y.cols() == X.cols(), Y.rows() ==
       * this.rows()
       * \see mult
       */
      void mult(Trans op, const DistM_t& X, DistM_t& Y) const;

      void extract_add_elements
      (const VecVec_t& I, const VecVec_t& J, std::vector<DistMW_t>& B);
      void extract_add_elements
      (const VecVec_t& I, const VecVec_t& J, std::vector<DenseMW_t>& B);
      void extract_add_elements(ExtractionMeta& e, std::vector<DistMW_t>& B);
      void extract_add_elements(ExtractionMeta& e, std::vector<DenseMW_t>& B);

      double get_stat(const std::string& name) const {
        if (!stats_) return 0;
        return BPACK_get_stat<scalar_t>(stats_, name);
      }

      DistM_t dense(const BLACSGrid* g) const;

      DenseM_t redistribute_2D_to_1D
      (const DistM_t& R2D, const std::vector<int>& dist) const;
      void redistribute_2D_to_1D
      (scalar_t a, const DistM_t& R2D, scalar_t b, DenseM_t& R1D,
       const std::vector<int>& dist) const;
      void redistribute_1D_to_2D
      (const DenseM_t& S1D, DistM_t& S2D, const std::vector<int>& dist) const;

    private:
      F2Cptr lr_bf_ = nullptr;     // LRBF handle returned by Fortran code
      F2Cptr options_ = nullptr;   // options structure returned by Fortran code
      F2Cptr stats_ = nullptr;     // statistics structure returned by Fortran code
      F2Cptr msh_ = nullptr;       // mesh structure returned by Fortran code
      F2Cptr kerquant_ = nullptr;  // kernel quantities structure returned by Fortran code
      F2Cptr ptree_ = nullptr;     // process tree returned by Fortran code
      MPI_Fint Fcomm_;             // the fortran MPI communicator
      MPIComm c_;
      int rows_ = 0, cols_ = 0, lrows_ = 0, lcols_ = 0;
      std::vector<int> rdist_, cdist_;  // begin rows/cols of each rank

      void set_extraction_meta_1grid
      (const VecVec_t& I, const VecVec_t& J, ExtractionMeta& e,
       int Nalldat_loc, int* pmaps) const;
    };

    template<typename scalar_t> LRBFMatrix<scalar_t>::LRBFMatrix
    (const HODLRMatrix<scalar_t>& A, const HODLRMatrix<scalar_t>& B)
      : c_(A.c_) {
      rows_ = A.rows();
      cols_ = B.cols();
      Fcomm_ = A.Fcomm_;
      int P = c_.size();
      int rank = c_.rank();
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
        (rows_, cols_, lrows_, lcols_, A.msh_, B.msh_, lr_bf_, options_,
         stats_, msh_, kerquant_, ptree_, nullptr, nullptr, nullptr);
      rdist_.resize(P+1);
      cdist_.resize(P+1);
      rdist_[rank+1] = lrows_;
      cdist_[rank+1] = lcols_;
      MPI_Allgather
        (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
         rdist_.data()+1, 1, MPI_INT, c_.comm());
      MPI_Allgather
        (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
         cdist_.data()+1, 1, MPI_INT, c_.comm());
      for (int p=0; p<P; p++) {
        rdist_[p+1] += rdist_[p];
        cdist_[p+1] += cdist_[p];
      }
    }

    template<typename integer_t> struct AdmInfoLRBF {
      std::pair<std::vector<int>,std::vector<int>> rmaps, cmaps;
      const DenseMatrix<bool>* adm;
      const CSRGraph<integer_t> *graph, *row_graph, *col_graph;
      integer_t rows, cols;
    };

    template<typename scalar_t, typename integer_t,
             typename real_t = typename RealType<scalar_t>::value_type>
    void LRBF_distance_query(int* m, int* n, real_t* dist, C2Fptr fdata) {
      auto& info = *static_cast<AdmInfoLRBF<integer_t>*>(fdata);
      auto& g = *(info.graph);
      auto& gr = *(info.row_graph);
      auto& gc = *(info.col_graph);
      int i = *m, j = *n;
      if (i < 0) {
        i = -i;
        std::swap(i, j);
      } else j = -j;
      i--;
      j--;
      assert(i >= 0 && j >= 0 && i < info.rows && j < info.cols);
      *dist = real_t(1.);
      auto hik = g.ind() + g.ptr(i+1);
      for (auto pk=g.ind()+g.ptr(i); pk!=hik; pk++)
        if (*pk == j) return;
      *dist = real_t(2.);
      hik = gr.ind() + gr.ptr(i+1);
      for (auto pk=gr.ind()+gr.ptr(i); pk!=hik; pk++) {
        auto k = *pk;
        auto hil = g.ind() + g.ptr(k+1);
        for (auto pl=g.ind()+g.ptr(k); pl!=hil; pl++)
          if (*pl == j) return;
      }
      hik = gc.ind() + gc.ptr(j+1);
      for (auto pk=gc.ind()+gc.ptr(j); pk!=hik; pk++) {
        auto k = *pk;
        auto hil = g.ind() + g.ptr(i+1);
        for (auto pl=g.ind()+g.ptr(i); pl!=hil; pl++)
          if (*pl == k) return;
      }
      *dist = real_t(3.);
    }

    template<typename integer_t> void LRBF_admissibility_query
    (int* m, int* n, int* admissible, C2Fptr fdata) {
      auto& info = *static_cast<AdmInfoLRBF<integer_t>*>(fdata);
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

    template<typename scalar_t> template<typename integer_t>
    LRBFMatrix<scalar_t>::LRBFMatrix
    (const HODLRMatrix<scalar_t>& A, const HSS::HSSPartitionTree& Atree,
     const CSRGraph<integer_t>& Agraph,
     const HODLRMatrix<scalar_t>& B, const HSS::HSSPartitionTree& Btree,
     const CSRGraph<integer_t>& Bgraph,
     const DenseMatrix<bool>& adm, const CSRGraph<integer_t>& graph,
     const opts_t& opts) : c_(A.c_) {
      rows_ = A.rows();
      cols_ = B.cols();
      Fcomm_ = A.Fcomm_;
      int P = c_.size(), rank = c_.rank();
      std::vector<int> groups(P);
      std::iota(groups.begin(), groups.end(), 0);
      // create hodlr data structures
      HODLR_createptree<scalar_t>(P, groups.data(), Fcomm_, ptree_);
      HODLR_createstats<scalar_t>(stats_);
      F2Cptr Aoptions = const_cast<F2Cptr>(A.options_);
      HODLR_copyoptions<scalar_t>(Aoptions, options_);
      HODLR_set_D_option<scalar_t>
        (options_, "sample_para",
         std::min(2.0, opts.BF_sampling_parameter()));
      if (opts.geo() == 2) {
        AdmInfoLRBF<integer_t> info;
        int min_lvl = 2 + std::ceil(std::log2(c_.size()));
        info.rmaps = Atree.map_from_complete_to_leafs
          (std::max(min_lvl, Atree.levels()));
        info.cmaps = Btree.map_from_complete_to_leafs
          (std::max(min_lvl, Btree.levels()));
        info.adm = &adm;
        info.graph = &graph;
        info.row_graph = &Agraph;
        info.col_graph = &Bgraph;
        info.rows = rows_;
        info.cols = cols_;
        HODLR_set_I_option<scalar_t>(options_, "nogeo", 2);
        HODLR_set_I_option<scalar_t>
          (options_, "knn",
           10 * std::ceil(float(graph.edges()) / graph.vertices()));
        LRBF_construct_init<scalar_t>
          (rows_, cols_, lrows_, lcols_, A.msh_, B.msh_, lr_bf_, options_,
           stats_, msh_, kerquant_, ptree_,
           &(LRBF_distance_query<scalar_t,integer_t>),
           &(LRBF_admissibility_query<integer_t>), &info);
      } else {
        HODLR_set_I_option<scalar_t>(options_, "nogeo", 1);
        HODLR_set_I_option<scalar_t>(options_, "knn", 0);
        LRBF_construct_init<scalar_t>
          (rows_, cols_, lrows_, lcols_, A.msh_, B.msh_, lr_bf_, options_,
           stats_, msh_, kerquant_, ptree_, nullptr, nullptr, nullptr);
      }
      rdist_.resize(P+1);
      cdist_.resize(P+1);
      rdist_[rank+1] = lrows_;
      cdist_[rank+1] = lcols_;
      MPI_Allgather
        (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
         rdist_.data()+1, 1, MPI_INT, c_.comm());
      MPI_Allgather
        (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
         cdist_.data()+1, 1, MPI_INT, c_.comm());
      for (int p=0; p<P; p++) {
        rdist_[p+1] += rdist_[p];
        cdist_[p+1] += cdist_[p];
      }
    }

    template<typename scalar_t> LRBFMatrix<scalar_t>::~LRBFMatrix() {
      if (stats_) HODLR_deletestats<scalar_t>(stats_);
      if (ptree_) HODLR_deleteproctree<scalar_t>(ptree_);
      if (msh_) HODLR_deletemesh<scalar_t>(msh_);
      if (kerquant_) HODLR_deletekernelquant<scalar_t>(kerquant_);
      if (options_) HODLR_deleteoptions<scalar_t>(options_);
      if (lr_bf_) LRBF_deletebf<scalar_t>(lr_bf_);
    }

    template<typename scalar_t> LRBFMatrix<scalar_t>&
    LRBFMatrix<scalar_t>::operator=(LRBFMatrix<scalar_t>&& h) {
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
      auto A = static_cast<typename LRBFMatrix<scalar_t>::mult_t*>(func);
      DenseMatrixWrapper<scalar_t> Yw(*nout, *nvec, Y, *nout),
        Xw(*nin, *nvec, const_cast<scalar_t*>(X), *nin);
      (*A)(c2T(*op), *a, Xw, *b, Yw);
    }

    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::compress(const mult_t& Amult) {
      C2Fptr f = static_cast<void*>(const_cast<mult_t*>(&Amult));
      LRBF_construct_matvec_compute
        (lr_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(LRBF_matvec_routine<scalar_t>), f);
    }

    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::compress(const mult_t& Amult, int rank_guess) {
      HODLR_set_I_option<scalar_t>(options_, "rank0", rank_guess);
      compress(Amult);
    }

    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::compress(const delem_blocks_t& Aelem) {
      AelemCommPtrs<scalar_t> AC{&Aelem, &c_};
      LRBF_construct_element_compute<scalar_t>
        (lr_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(HODLR_block_evaluation<scalar_t>), &AC);
    }

    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::compress(const elem_blocks_t& Aelem) {
      C2Fptr f = static_cast<void*>(const_cast<elem_blocks_t*>(&Aelem));
      LRBF_construct_element_compute<scalar_t>
        (lr_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(HODLR_block_evaluation_seq<scalar_t>), f);
    }

    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::mult
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
    LRBFMatrix<scalar_t>::mult
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
    LRBFMatrix<scalar_t>::extract_add_elements
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

    /**
     * All the matrices in B should have the same BLACSGrid!!!
     */
    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::extract_add_elements
    (const VecVec_t& I, const VecVec_t& J, std::vector<DistMW_t>& B) {
      if (I.empty()) return;
      assert(I.size() == J.size() && I.size() == B.size());
      ExtractionMeta e;
      int pmaps[3] = {B[0].nprows(), B[0].npcols(), 0};
      int Nalldat_loc = 0;
      for (auto& Bk : B) {
        assert(Bk.grid() == B[0].grid());
        Nalldat_loc += Bk.rows() * Bk.cols();
      }
      set_extraction_meta_1grid(I, J, e, Nalldat_loc, pmaps);
      extract_add_elements(I, J, B, e);
    }

    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::extract_add_elements
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
    LRBFMatrix<scalar_t>::extract_add_elements
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
    LRBFMatrix<scalar_t>::set_extraction_meta_1grid
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
        for (auto l : I[k]) { assert(l < rows_); e.allrows[i++] = l+1; }
        for (auto l : J[k]) { assert(l < cols_); e.allcols[j++] = l+1; }
      }
    }

    template<typename scalar_t> DistributedMatrix<scalar_t>
    LRBFMatrix<scalar_t>::dense(const BLACSGrid* g) const {
      DistM_t A(g, rows_, cols_), I(g, rows_, cols_);
      I.eye();
      mult(Trans::N, I, A);
      return A;
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    LRBFMatrix<scalar_t>::redistribute_2D_to_1D
    (const DistM_t& R2D, const std::vector<int>& dist) const {
      const auto rank = c_.rank();
      DenseM_t R1D(dist[rank+1] - dist[rank], R2D.cols());
      redistribute_2D_to_1D(scalar_t(1.), R2D, scalar_t(0.), R1D, dist);
      return R1D;
    }

    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::redistribute_2D_to_1D
    (scalar_t a, const DistM_t& R2D, scalar_t b, DenseM_t& R1D,
     const std::vector<int>& dist) const {
      TIMER_TIME(TaskType::REDIST_2D_TO_HSS, 0, t_redist);
      const auto P = c_.size();
      const auto rank = c_.rank();
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
      std::vector<scalar_t> rbuf;
      std::vector<scalar_t*> pbuf;
      c_.all_to_all_v(sbuf, rbuf, pbuf);
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
    LRBFMatrix<scalar_t>::redistribute_1D_to_2D
    (const DenseM_t& S1D, DistM_t& S2D, const std::vector<int>& dist) const {
      TIMER_TIME(TaskType::REDIST_2D_TO_HSS, 0, t_redist);
      const int rank = c_.rank();
      const int P = c_.size();
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
      std::vector<scalar_t> rbuf;
      std::vector<scalar_t*> pbuf;
      c_.all_to_all_v(sbuf, rbuf, pbuf);
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

  } // end namespace HODLR
} // end namespace strumpack

#endif // STRUMPACK_LRBF_MATRIX_HPP
