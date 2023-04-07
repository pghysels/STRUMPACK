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
#include <algorithm>

#include "misc/Tools.hpp"
#include "misc/TaskTimer.hpp"
#include "clustering/Clustering.hpp"
#include "HODLRMatrix.hpp"
#include "HODLRWrapper.hpp"

namespace strumpack {

  namespace HODLR {

    template<typename T> struct KernelCommPtrs {
      const kernel::Kernel<T>* K;
      const MPIComm* c;
    };

    /**
     * Routine used to pass to the fortran code to compute a selected
     * element of a kernel. The kernel argument needs to be a pointer
     * to a strumpack::kernel object.
     *
     * \param i row coordinate of element to be computed from the
     * kernel
     * \param i column coordinate of element to be computed from the
     * kernel
     * \param v output, kernel value
     * \param kernel pointer to Kernel object
     */
    template<typename scalar_t> void HODLR_kernel_evaluation
    (int* i, int* j, scalar_t* v, C2Fptr KC) {
      const auto& K = *(static_cast<KernelCommPtrs<scalar_t>*>(KC)->K);
      *v = K.eval(*i-1, *j-1);
    }

    template<typename scalar_t> void HODLR_kernel_block_evaluation
    (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
     int* allrows, int* allcols, scalar_t* alldat_loc,
     int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
     C2Fptr KC) {
      auto temp = static_cast<KernelCommPtrs<scalar_t>*>(KC);
      const auto& K = *(temp->K);
      const auto& comm = *(temp->c);
      auto data = alldat_loc;
      for (int isec=0, r0=0, c0=0; isec<*Ninter; isec++) {
        auto m = rowids[isec];
        auto n = colids[isec];
        auto p0 = pmaps[2*(*Npmap)+pgids[isec]];
        assert(pmaps[pgids[isec]] == 1);          // prows == 1
        assert(pmaps[(*Npmap)+pgids[isec]] == 1); // pcols == 1
        if (comm.rank() == p0) {
          for (int c=0; c<n; c++) {
            auto col = std::abs(allcols[c0+c])-1;
            for (int r=0; r<m; r++)
              data[r+c*m] = K.eval(allrows[r0+r]-1, col);
          }
          data += m*n;
        }
        r0 += m;
        c0 += n;
      }
    }

    template<typename scalar_t> void HODLR_element_evaluation
    (int* i, int* j, scalar_t* v, C2Fptr elem) {
      *v = static_cast<std::function<scalar_t(int,int)>*>
        (elem)->operator()(*i-1, *j-1);
    }

    template<typename scalar_t> void HODLR_block_evaluation
    (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
     int* allrows, int* allcols, scalar_t* alldat_loc,
     int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
     C2Fptr AC) {
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      auto temp = static_cast<AelemCommPtrs<scalar_t>*>(AC);
      std::vector<std::vector<std::size_t>> I(*Ninter), J(*Ninter);
      std::vector<DistMW_t> B(*Ninter);
      auto& comm = *(temp->c);
      auto rank = comm.rank();
      auto data = alldat_loc;
      for (int isec=0, r0=0, c0=0; isec<*Ninter; isec++) {
        auto m = rowids[isec];
        auto n = colids[isec];
        I[isec].reserve(m);
        J[isec].reserve(n);
        for (int i=0; i<m; i++)
          I[isec].push_back(allrows[r0+i]-1);
        for (int i=0; i<n; i++)
          J[isec].push_back(std::abs(allcols[c0+i])-1);
        auto p0 = pmaps[2*(*Npmap)+pgids[isec]];
        assert(pmaps[pgids[isec]] == 1);          // prows == 1
        assert(pmaps[(*Npmap)+pgids[isec]] == 1); // pcols == 1
        B[isec] = DistMW_t(rank == p0 ? temp->gl : temp->g0, m, n, data);
        r0 += m;
        c0 += n;
        if (rank == p0) data += m*n;
      }
      ExtractionMeta e
        {nullptr, *Ninter, *Nallrows, *Nallcols, *Nalldat_loc,
            allrows, allcols, rowids, colids, pgids, *Npmap, pmaps};
      temp->Aelem->operator()(I, J, B, e);
    }

    template<typename scalar_t> void HODLR_block_evaluation_seq
    (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
     int* allrows, int* allcols, scalar_t* alldat_loc,
     int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
     C2Fptr f) {
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      std::vector<std::vector<std::size_t>> I(*Ninter), J(*Ninter);
      std::vector<DenseMW_t> B(*Ninter);
      auto data = alldat_loc;
      for (int isec=0, r0=0, c0=0; isec<*Ninter; isec++) {
        auto m = rowids[isec];
        auto n = colids[isec];
        I[isec].reserve(m);
        J[isec].reserve(n);
        for (int i=0; i<m; i++)
          I[isec].push_back(allrows[r0+i]-1);
        for (int i=0; i<n; i++)
          J[isec].push_back(std::abs(allcols[c0+i])-1);
        B[isec] = DenseMW_t(m, n, data, m);
        r0 += m;
        c0 += n;
        data += m*n;
      }
      ExtractionMeta e
        {nullptr, *Ninter, *Nallrows, *Nallcols, *Nalldat_loc,
            allrows, allcols, rowids, colids, pgids, *Npmap, pmaps};
      static_cast<typename HODLRMatrix<scalar_t>::elem_blocks_t*>
        (f)->operator()(I, J, B, e);
    }


    template<typename scalar_t> HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, kernel::Kernel<real_t>& K, const opts_t& opts) {
      rows_ = cols_ = K.n();
      structured::ClusterTree tree(rows_);
      if (opts.geo() == 1)
        tree = binary_tree_clustering
          (opts.clustering_algorithm(), K.data(),
           K.permutation(), opts.leaf_size());
      else tree.refine(opts.leaf_size());
      int min_lvl = 2 + std::ceil(std::log2(c.size()));
      lvls_ = std::max(min_lvl, tree.levels());
      tree.expand_complete_levels(lvls_);
      leafs_ = tree.template leaf_sizes<int>();
      c_ = &c;
      Fcomm_ = MPI_Comm_c2f(c_->comm());
      options_init(opts);
      perm_.resize(rows_);
      HODLR_set_I_option<scalar_t>(options_, "knn", opts.knn_hodlrbf());
      HODLR_set_I_option<scalar_t>(options_, "RecLR_leaf", opts.lr_leaf());
      if (opts.geo() == 1) {
        // do not pass any neighbor info to the HODLR code
        HODLR_set_I_option<scalar_t>(options_, "nogeo", 1);
        HODLR_construct_init<scalar_t,real_t>
          (rows_, 0, nullptr, nullptr, lvls_-1, leafs_.data(), perm_.data(),
           lrows_, ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
           nullptr, nullptr, nullptr);
      } else {
        HODLR_set_I_option<scalar_t>(options_, "nogeo", 0);
        // pass the data points to the HODLR code, let the HODLR code
        // figure out the permutation etc
        HODLR_construct_init<scalar_t,real_t>
          (rows_, K.d(), K.data().data(), nullptr, lvls_-1, leafs_.data(),
           perm_.data(), lrows_, ho_bf_, options_, stats_, msh_,
           kerquant_, ptree_, nullptr, nullptr, nullptr);
      }
      perm_init();
      dist_init();
      KernelCommPtrs<real_t> KC{&K, c_};
      HODLR_construct_element_compute<scalar_t>
        (ho_bf_, options_, stats_, msh_, kerquant_,
         ptree_, &(HODLR_kernel_evaluation<scalar_t>),
         &(HODLR_kernel_block_evaluation<scalar_t>), &KC);
      if (opts.geo() != 1) {
        K.permutation() = perm();
        K.data().lapmr(perm(), true);
      }
    }

    template<typename scalar_t> HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const std::function<scalar_t(int i, int j)>& Aelem,
     const opts_t& opts) : HODLRMatrix<scalar_t>(c, tree, opts) {
      // 1 = block extraction, 0 = element
      HODLR_set_I_option<scalar_t>(options_, "elem_extract", 0);
      HODLR_construct_element_compute<scalar_t>
        (ho_bf_, options_, stats_, msh_, kerquant_,
         ptree_, &(HODLR_element_evaluation<scalar_t>), nullptr,
         const_cast<std::function<scalar_t(int i, int j)>*>(&Aelem));
    }

    template<typename scalar_t> HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const elem_blocks_t& Aelem, const opts_t& opts)
      : HODLRMatrix<scalar_t>(c, tree, opts) {
      // 1 = block extraction, 0 = element
      HODLR_set_I_option<scalar_t>(options_, "elem_extract", 1);
      HODLR_construct_element_compute<scalar_t>
        (ho_bf_, options_, stats_, msh_, kerquant_,
         ptree_, nullptr, &(HODLR_block_evaluation_seq<scalar_t>),
         const_cast<elem_blocks_t*>(&Aelem));
    }

    template<typename scalar_t> HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const mult_t& Amult, const opts_t& opts)
      : HODLRMatrix<scalar_t>(c, tree, opts) {
      compress(Amult);
    }

    template<typename scalar_t> HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const opts_t& opts) {
      rows_ = cols_ = tree.size;
      structured::ClusterTree full_tree(tree);
      int min_lvl = 2 + std::ceil(std::log2(c.size()));
      lvls_ = std::max(min_lvl, full_tree.levels());
      full_tree.expand_complete_levels(lvls_);
      leafs_ = full_tree.template leaf_sizes<int>();
      c_ = &c;
      if (c_->is_null()) return;
      Fcomm_ = MPI_Comm_c2f(c_->comm());
      options_init(opts);
      perm_.resize(rows_);
      HODLR_construct_init<scalar_t,real_t>
        (rows_, 0, nullptr, nullptr, lvls_-1, leafs_.data(), perm_.data(),
         lrows_, ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
         nullptr, nullptr, nullptr);
      perm_init();
      dist_init();
    }

    template<typename integer_t> struct AdmInfo {
      std::pair<std::vector<int>,std::vector<int>> maps;
      const DenseMatrix<bool>* adm;
    };

    // template<typename integer_t> void HODLR_admissibility_query
    // (int* m, int* n, int* admissible, C2Fptr fdata) {
    //   auto& info = *static_cast<AdmInfo<integer_t>*>(fdata);
    //   auto& adm = *(info.adm);
    //   auto& map0 = info.maps.first;
    //   auto& map1 = info.maps.second;
    //   int r = *m - 1, c = *n - 1;
    //   bool a = true;
    //   for (int j=map0[c]; j<=map1[c] && a; j++)
    //     for (int i=map0[r]; i<=map1[r] && a; i++)
    //       a = a && adm(i, j);
    //   *admissible = a;
    // }

    template<typename scalar_t> template<typename integer_t>
    HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const CSRGraph<integer_t>& graph, const opts_t& opts) {
      rows_ = cols_ = tree.size;
      structured::ClusterTree full_tree(tree);
      int min_lvl = 2 + std::ceil(std::log2(c.size()));
      lvls_ = std::max(min_lvl, full_tree.levels());
      full_tree.expand_complete_levels(lvls_);
      leafs_ = full_tree.template leaf_sizes<int>();
      c_ = &c;
      if (c_->is_null()) return;
      Fcomm_ = MPI_Comm_c2f(c_->comm());
      options_init(opts);
      perm_.resize(rows_);
      HODLR_set_I_option<scalar_t>(options_, "nogeo", 3);
      int knn = opts.knn_hodlrbf();
      HODLR_set_I_option<scalar_t>(options_, "knn", knn);
      DenseMatrix<int> nns(knn, rows_);
      { TIMER_TIME(TaskType::NEIGHBOR_SEARCH, 0, t_knn);
        nns.fill(0);
        int B = std::ceil(rows_ / params::num_threads);
#pragma omp parallel for schedule(static, 1)
        for (int lo=0; lo<rows_; lo+=B) {
          std::vector<bool> mark(rows_, false);
          std::vector<int> q(knn+1);
          for (int i=lo; i<std::min(lo+B, rows_); i++) {
            int qfront = 0, qback = 0, nn = 0;
            q[qback++] = i;
            mark[i] = true;
            while (nn < knn && qfront < qback) {
              auto k = q[qfront++];
              const auto hi = graph.ind() + graph.ptr(k+1);
              for (auto pl=graph.ind()+graph.ptr(k); pl!=hi; pl++) {
                auto l = *pl;
                if (!mark[l]) {
                  nns(nn++, i) = l+1; // found a new neighbor
                  if (nn == knn) break;
                  mark[l] = true;
                  q[qback++] = l;
                }
              }
            }
            for (int l=0; l<qback; l++) mark[q[l]] = false;
          }
        }
      }
      { TIMER_TIME(TaskType::CONSTRUCT_INIT, 0, t_construct_h);
        HODLR_construct_init<scalar_t,real_t>
          (rows_, 0, nullptr, nns.data(), lvls_-1, leafs_.data(), perm_.data(),
           lrows_, ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
           nullptr, nullptr, nullptr);
      }
      perm_init();
      dist_init();
    }


    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::options_init(const opts_t& opts) {
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

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::set_sampling_parameter(double sample_param) {
      HODLR_set_D_option<scalar_t>(options_, "sample_para", sample_param);
      HODLR_set_D_option<scalar_t>(options_, "sample_para_outer", sample_param);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::set_BACA_block(int bsize) {
      HODLR_set_I_option<scalar_t>(options_, "BACA_Batch", bsize);
    }


    template<typename scalar_t> void HODLRMatrix<scalar_t>::perm_init() {
      iperm_.resize(rows_);
      c_->broadcast(perm_);
      for (int i=1; i<=rows_; i++)
        iperm_[perm_[i-1]-1] = i;
    }

    template<typename scalar_t> void HODLRMatrix<scalar_t>::dist_init() {
      auto P = c_->size();
      auto rank = c_->rank();
      dist_.resize(P+1);
      dist_[rank+1] = lrows_;
      c_->all_gather(dist_.data()+1, 1);
      for (int p=0; p<P; p++) dist_[p+1] += dist_[p];
    }

    template<typename scalar_t> HODLRMatrix<scalar_t>::~HODLRMatrix() {
      if (stats_) HODLR_deletestats<scalar_t>(stats_);
      if (ptree_) HODLR_deleteproctree<scalar_t>(ptree_);
      if (msh_) HODLR_deletemesh<scalar_t>(msh_);
      if (kerquant_) HODLR_deletekernelquant<scalar_t>(kerquant_);
      if (ho_bf_) HODLR_delete<scalar_t>(ho_bf_);
      if (options_) HODLR_deleteoptions<scalar_t>(options_);
    }

    template<typename scalar_t> HODLRMatrix<scalar_t>&
    HODLRMatrix<scalar_t>::operator=(HODLRMatrix<scalar_t>&& h) {
      ho_bf_ = h.ho_bf_;       h.ho_bf_ = nullptr;
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
      std::swap(perm_, h.perm_);
      std::swap(iperm_, h.iperm_);
      std::swap(dist_, h.dist_);
      return *this;
    }

    template<typename scalar_t> double
    HODLRMatrix<scalar_t>::get_stat(const std::string& name) const {
      if (!stats_) return 0;
      return BPACK_get_stat<scalar_t>(stats_, name);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::print_stats() {
      if (!stats_) return;
      HODLR_printstats<scalar_t>(stats_, ptree_);
    }

    template<typename scalar_t> void HODLR_matvec_routine
    (const char* op, int* nin, int* nout, int* nvec,
     const scalar_t* X, scalar_t* Y, C2Fptr func) {
      auto A = static_cast<typename HODLRMatrix<scalar_t>::mult_t*>(func);
      DenseMatrixWrapper<scalar_t> Yw(*nout, *nvec, Y, *nout),
        Xw(*nin, *nvec, const_cast<scalar_t*>(X), *nin);
      (*A)(c2T(*op), Xw, Yw);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::compress(const mult_t& Amult) {
      if (c_->is_null()) return;
      C2Fptr f = static_cast<void*>(const_cast<mult_t*>(&Amult));
      HODLR_construct_matvec_compute<scalar_t>
        (ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(HODLR_matvec_routine<scalar_t>), f);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::compress
    (const mult_t& Amult, int rank_guess) {
      HODLR_set_I_option<scalar_t>(options_, "rank0", rank_guess);
      compress(Amult);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::compress(const delem_blocks_t& Aelem) {
      BLACSGrid gloc(MPIComm(MPI_COMM_SELF), 1),
        gnull(MPIComm(MPI_COMM_NULL), 1);
      AelemCommPtrs<scalar_t> AC{&Aelem, c_, &gloc, &gnull};
      HODLR_construct_element_compute<scalar_t>
        (ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(HODLR_element_evaluation<scalar_t>),
         &(HODLR_block_evaluation<scalar_t>), &AC);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::compress(const elem_blocks_t& Aelem) {
      C2Fptr f = static_cast<void*>(const_cast<elem_blocks_t*>(&Aelem));
      HODLR_construct_element_compute<scalar_t>
        (ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(HODLR_element_evaluation<scalar_t>),
         &(HODLR_block_evaluation_seq<scalar_t>), f);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::mult
    (Trans op, const DenseM_t& X, DenseM_t& Y) const {
      if (c_->is_null()) return;
      HODLR_mult(char(op), X.data(), Y.data(), lrows_, lrows_, X.cols(),
                 ho_bf_, options_, stats_, ptree_);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::mult
    (Trans op, const DistM_t& X, DistM_t& Y) const {
      if (c_->is_null()) return;
      DenseM_t Y1D(lrows_, X.cols());
      {
        auto X1D = redistribute_2D_to_1D(X);
        HODLR_mult(char(op), X1D.data(), Y1D.data(), lrows_, lrows_,
                   X.cols(), ho_bf_, options_, stats_, ptree_);
      }
      redistribute_1D_to_2D(Y1D, Y);
    }

    template<typename scalar_t> long long int
    HODLRMatrix<scalar_t>::inv_mult
    (Trans op, const DenseM_t& X, DenseM_t& Y) const {
      if (c_->is_null()) return 0;
      HODLR_inv_mult
        (char(op), X.data(), Y.data(), lrows_, lrows_, X.cols(),
         ho_bf_, options_, stats_, ptree_);
      long long int flops = get_stat("Flop_C_Mult");
#if 0
      DenseM_t R(X.rows(), X.cols()), E(X.rows(), X.cols());
      mult(op, Y, R);                     // R = A*Y
      flops += get_stat("Flop_C_Mult");
      R.scale_and_add(scalar_t(-1.), X);  // R = X - A*Y, residual
      flops += R.rows() * R.cols();
      HODLR_inv_mult
        (char(op), R.data(), E.data(), lrows_, lrows_, X.cols(),
         ho_bf_, options_, stats_, ptree_);
      flops += get_stat("Flop_C_Mult");
#endif
      return flops;
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::factor() {
      if (c_->is_null()) return;
      HODLR_factor<scalar_t>(ho_bf_, options_, stats_, ptree_, msh_);
    }

    template<typename scalar_t> long long int
    HODLRMatrix<scalar_t>::solve(const DenseM_t& B, DenseM_t& X) const {
      if (c_->is_null()) return 0;
      return inv_mult(Trans::N, B, X);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::solve(DenseM_t& B) const {
      if (c_->is_null()) return;
      DenseM_t X(B.rows(), B.cols());
      inv_mult(Trans::N, B, X);
      B.copy(X);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::solve(DistM_t& B) const {
      if (c_->is_null()) return;
      auto B1D = redistribute_2D_to_1D(B);
      DenseM_t X1D(lrows_, B.cols());
      inv_mult(Trans::N, B1D, X1D);
      redistribute_1D_to_2D(X1D, B);
    }

    template<typename scalar_t> long long int
    HODLRMatrix<scalar_t>::solve(const DistM_t& B, DistM_t& X) const {
      if (c_->is_null()) return 0;
      DenseM_t X1D(lrows_, X.cols());
      long long int flops = 0;
      {
        auto B1D = redistribute_2D_to_1D(B);
        flops = inv_mult(Trans::N, B1D, X1D);
      }
      redistribute_1D_to_2D(X1D, X);
      return flops;
    }

    template<typename scalar_t> void HODLRMatrix<scalar_t>::extract_elements
    (const VecVec_t& I, const VecVec_t& J, std::vector<DistM_t>& B) {
      if (I.empty()) return;
      assert(I.size() == J.size() && I.size() == B.size());
      int Ninter = I.size(), total_rows = 0, total_cols = 0, total_dat = 0;
      int pmaps[3] = {B[0].nprows(), B[0].npcols(), 0};
      for (auto Ik : I) total_rows += Ik.size();
      for (auto Jk : J) total_cols += Jk.size();
      std::unique_ptr<int[]> iwork
        (new int[total_rows + total_cols + 3*Ninter]);
      auto allrows = iwork.get();
      auto allcols = allrows + total_rows;
      auto rowidx = allcols + total_cols;
      auto colidx = rowidx + Ninter;
      auto pgids = colidx + Ninter;
      for (int k=0, i=0, j=0; k<Ninter; k++) {
        assert(B[k].nprows() == pmaps[0]);
        assert(B[k].npcols() == pmaps[1]);
        total_dat += B[k].lrows()*B[k].lcols();
        rowidx[k] = I[k].size();
        colidx[k] = J[k].size();
        pgids[k] = 0;
        for (auto l : I[k]) { assert(int(l) < rows_); allrows[i++] = l+1; }
        for (auto l : J[k]) { assert(int(l) < cols_); allcols[j++] = l+1; }
      }
      std::unique_ptr<scalar_t[]> alldat_loc(new scalar_t[total_dat]);
      auto ptr = alldat_loc.get();
      HODLR_extract_elements<scalar_t>
        (ho_bf_, options_, msh_, stats_, ptree_, Ninter,
         total_rows, total_cols, total_dat, allrows, allcols,
         ptr, rowidx, colidx, pgids, 1, pmaps);
      for (auto& Bk : B) {
        auto m = Bk.lcols();
        auto n = Bk.lrows();
        auto Bdata = Bk.data();
        auto Bld = Bk.ld();
        for (int j=0; j<m; j++)
          for (int i=0; i<n; i++)
            Bdata[i+j*Bld] = *ptr++;
      }
    }

    template<typename scalar_t> void HODLRMatrix<scalar_t>::extract_elements
    (const VecVec_t& I, const VecVec_t& J, std::vector<DenseM_t>& B) {
      if (I.empty()) return;
      assert(I.size() == J.size() && I.size() == B.size());
      int Ninter = I.size(), total_rows = 0, total_cols = 0, total_dat = 0;
      int pmaps[3] = {1, 1, 0};
      for (auto Ik : I) total_rows += Ik.size();
      for (auto Jk : J) total_cols += Jk.size();
      std::unique_ptr<int[]> iwork
        (new int[total_rows + total_cols + 3*Ninter]);
      auto allrows = iwork.get();
      auto allcols = allrows + total_rows;
      auto rowidx = allcols + total_cols;
      auto colidx = rowidx + Ninter;
      auto pgids = colidx + Ninter;
      for (int k=0, i=0, j=0; k<Ninter; k++) {
        total_dat += B[k].rows()*B[k].cols();
        rowidx[k] = I[k].size();
        colidx[k] = J[k].size();
        pgids[k] = 0;
        for (auto l : I[k]) { assert(int(l) < rows_); allrows[i++] = l+1; }
        for (auto l : J[k]) { assert(int(l) < cols_); allcols[j++] = l+1; }
      }
      std::unique_ptr<scalar_t[]> alldat_loc(new scalar_t[total_dat]);
      auto ptr = alldat_loc.get();
      HODLR_extract_elements<scalar_t>
        (ho_bf_, options_, msh_, stats_, ptree_, Ninter,
         total_rows, total_cols, total_dat, allrows, allcols,
         ptr, rowidx, colidx, pgids, 1, pmaps);
      for (auto& Bk : B) {
        auto m = Bk.cols();
        auto n = Bk.rows();
        auto Bdata = Bk.data();
        auto Bld = Bk.ld();
        for (std::size_t j=0; j<m; j++)
          for (std::size_t i=0; i<n; i++)
            Bdata[i+j*Bld] = *ptr++;
      }
    }

    template<typename scalar_t> void HODLRMatrix<scalar_t>::extract_elements
    (const Vec_t& I, const Vec_t& J, DenseM_t& B) {
      int m = I.size(), n = J.size(), pgids = 0;
      if (m == 0 || n == 0) return;
      int pmaps[3] = {1, 1, 0};
      std::vector<int> Ii, Ji;
      Ii.assign(I.begin(), I.end());
      Ji.assign(J.begin(), J.end());
      for (auto& i : Ii) i++;
      for (auto& j : Ji) j++;
      HODLR_extract_elements<scalar_t>
        (ho_bf_, options_, msh_, stats_, ptree_, 1, m, n, m*n,
         Ii.data(), Ji.data(), B.data(), &m, &n, &pgids, 1, pmaps);
    }

    template<typename scalar_t> DistributedMatrix<scalar_t>
    HODLRMatrix<scalar_t>::dense(const BLACSGrid* g) const {
      DistM_t A(g, rows_, cols_), I(g, cols_, cols_);
      I.eye();
      mult(Trans::N, I, A);
      return A;
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    HODLRMatrix<scalar_t>::gather_from_1D(const DenseM_t& A) const {
      // TODO avoid going through 2D
      assert(A.rows() == lrows());
      BLACSGrid g(*c_);
      DistM_t A2D(&g, rows_, A.cols());
      redistribute_1D_to_2D(A, A2D);
      return A2D.gather();
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    HODLRMatrix<scalar_t>::all_gather_from_1D(const DenseM_t& A) const {
      // TODO avoid going through 2D
      assert(A.rows() == lrows());
      BLACSGrid g(*c_);
      DistM_t A2D(&g, rows_, A.cols());
      redistribute_1D_to_2D(A, A2D);
      return A2D.all_gather();
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    HODLRMatrix<scalar_t>::scatter_to_1D(const DenseM_t& A) const {
      // TODO avoid going through 2D
      assert(A.rows() == rows());
      BLACSGrid g(*c_);
      DistM_t A2D(&g, rows_, A.cols());
      copy(rows_, A.cols(), A, 0, A2D, 0, 0, A2D.ctxt_all());
      return redistribute_2D_to_1D(A2D);
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    HODLRMatrix<scalar_t>::redistribute_2D_to_1D(const DistM_t& R2D) const {
      DenseM_t R1D(lrows_, R2D.cols());
      redistribute_2D_to_1D(R2D, R1D);
      return R1D;
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::redistribute_2D_to_1D
    (const DistM_t& R2D, DenseM_t& R1D) const {
      TIMER_TIME(TaskType::REDIST_2D_TO_HSS, 0, t_redist);
      if (c_->is_null()) return;
      const auto P = c_->size();
      const auto rank = c_->rank();
      // for (int p=0; p<P; p++)
      //   copy(dist_[rank+1]-dist_[rank], R2D.cols(), R2D, dist_[rank], 0,
      //        R1D, p, R2D.grid()->ctxt_all());
      // return;
      const auto Rcols = R2D.cols();
      int R2Drlo, R2Drhi, R2Dclo, R2Dchi;
      R2D.lranges(R2Drlo, R2Drhi, R2Dclo, R2Dchi);
      const auto Rlcols = R2Dchi - R2Dclo;
      const auto Rlrows = R2Drhi - R2Drlo;
      const auto nprows = R2D.nprows();
      std::vector<std::vector<scalar_t>> sbuf(P);
      if (R2D.active()) {
        // global, local, proc
        std::vector<std::tuple<int,int,int>> glp(Rlrows);
        {
          std::vector<std::size_t> count(P);
          for (int r=R2Drlo; r<R2Drhi; r++) {
            auto gr = R2D.rowl2g(r);
            auto p = -1 + std::distance
              (dist_.begin(), std::upper_bound
               (dist_.begin(), dist_.end(), gr));
            glp[r-R2Drlo] = std::tuple<int,int,int>{gr, r, p};
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
      assert(int(R1D.rows()) == lrows_ && int(R1D.cols()) == Rcols);
      if (lrows_) {
        std::vector<int> src_c(Rcols);
        for (int c=0; c<Rcols; c++)
          src_c[c] = R2D.colg2p_fixed(c)*nprows;
        for (int r=0; r<lrows_; r++) {
          auto gr = r + dist_[rank];
          auto src_r = R2D.rowg2p_fixed(gr);
          for (int c=0; c<Rcols; c++)
            R1D(r, c) = *(pbuf[src_r + src_c[c]]++);
        }
      }
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::redistribute_1D_to_2D
    (const DenseM_t& S1D, DistM_t& S2D) const {
      TIMER_TIME(TaskType::REDIST_2D_TO_HSS, 0, t_redist);
      if (c_->is_null()) return;
      const int rank = c_->rank();
      const int P = c_->size();
      const int cols = S1D.cols();
      int S2Drlo, S2Drhi, S2Dclo, S2Dchi;
      S2D.lranges(S2Drlo, S2Drhi, S2Dclo, S2Dchi);
      const auto nprows = S2D.nprows();
      std::vector<std::vector<scalar_t>> sbuf(P);
      assert(int(S1D.rows()) == lrows_);
      assert(int(S1D.rows()) == dist_[rank+1] - dist_[rank]);
      if (lrows_) {
        std::vector<std::tuple<int,int,int>> glp(lrows_);
        for (int r=0; r<lrows_; r++) {
          auto gr = r + dist_[rank];
          assert(gr >= 0 && gr < S2D.rows());
          glp[r] = std::tuple<int,int,int>{gr,r,S2D.rowg2p_fixed(gr)};
        }
        std::sort(glp.begin(), glp.end());
        std::vector<int> pc(cols);
        for (int c=0; c<cols; c++)
          pc[c] = S2D.colg2p_fixed(c)*nprows;
        {
          std::vector<std::size_t> count(P);
          for (int r=0; r<lrows_; r++)
            for (int c=0, pr=std::get<2>(glp[r]); c<cols; c++)
              count[pr+pc[c]]++;
          for (int p=0; p<P; p++)
            sbuf[p].reserve(count[p]);
        }
        for (int r=0; r<lrows_; r++)
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
            (dist_.begin(), std::upper_bound(dist_.begin(), dist_.end(), gr));
          assert(p < P && p >= 0);
          for (int c=S2Dclo; c<S2Dchi; c++) {
            auto tmp = *(pbuf[p]++);
            S2D(r,c) = tmp;
          }
        }
      }
    }

    // explicit instantiations
    template class HODLRMatrix<float>;
    template class HODLRMatrix<double>;
    template class HODLRMatrix<std::complex<float>>;
    template class HODLRMatrix<std::complex<double>>;

    // explicit instantiations for the integer type in the CSRGraph
    template HODLRMatrix<float>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const CSRGraph<int>& graph, const opts_t& opts);
    template HODLRMatrix<double>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const CSRGraph<int>& graph, const opts_t& opts);
    template HODLRMatrix<std::complex<float>>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const CSRGraph<int>& graph, const opts_t& opts);
    template HODLRMatrix<std::complex<double>>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const CSRGraph<int>& graph, const opts_t& opts);

    template HODLRMatrix<float>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const CSRGraph<long>& graph, const opts_t& opts);
    template HODLRMatrix<double>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const CSRGraph<long>& graph, const opts_t& opts);
    template HODLRMatrix<std::complex<float>>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const CSRGraph<long>& graph, const opts_t& opts);
    template HODLRMatrix<std::complex<double>>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const CSRGraph<long>& graph, const opts_t& opts);

    template HODLRMatrix<float>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const CSRGraph<long long int>& graph, const opts_t& opts);
    template HODLRMatrix<double>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const CSRGraph<long long int>& graph, const opts_t& opts);
    template HODLRMatrix<std::complex<float>>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const CSRGraph<long long int>& graph, const opts_t& opts);
    template HODLRMatrix<std::complex<double>>::HODLRMatrix
    (const MPIComm& c, const structured::ClusterTree& tree,
     const CSRGraph<long long int>& graph, const opts_t& opts);

  } // end namespace HODLR
} // end namespace strumpack
