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

#include "FrontalMatrixHODLR.hpp"
#include "ExtendAdd.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  FrontalMatrixHODLR<scalar_t,integer_t>::FrontalMatrixHODLR
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd),
    commself_(MPI_COMM_SELF) { }

  template<typename scalar_t,typename integer_t>
  FrontalMatrixHODLR<scalar_t,integer_t>::~FrontalMatrixHODLR() {
#if defined(STRUMPACK_COUNT_FLOPS)
    auto HOD_mem =
      F11_.get_stat("Mem_Fill") + F11_.get_stat("Mem_Factor")
      + F12_.get_stat("Mem_Fill") + F21_.get_stat("Mem_Fill");
    if (F22_) HOD_mem += F22_->get_stat("Mem_Fill");
    STRUMPACK_SUB_MEMORY(HOD_mem*1.e6);
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::release_work_memory() {
    if (F22_) {
      TaskTimer t_traverse(TaskType::BF_EXTRACT_TRAVERSE, 3),
        t_bf(TaskType::BF_EXTRACT_ENTRY, 3),
        t_comm(TaskType::BF_EXTRACT_COMM, 3);
      t_traverse.set_elapsed(F22_->get_stat("Time_Entry_Traverse"));
      t_bf.set_elapsed(F22_->get_stat("Time_Entry_BF"));
      t_comm.set_elapsed(F22_->get_stat("Time_Entry_Comm"));
      STRUMPACK_SUB_MEMORY(F22_->get_stat("Mem_Fill")*1.e6);
      F22_.reset(nullptr);
    }
  }

  template<typename scalar_t,typename integer_t> DenseMatrix<scalar_t>
  FrontalMatrixHODLR<scalar_t,integer_t>::get_dense_CB() const {
    const std::size_t dupd = dim_upd();
    DenseM_t CB(dupd, dupd), id(dupd, dupd);
    TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
#if defined(STRUMPACK_PERMUTE_CB)
    if (CB_perm_.size() == dupd) {
      CB.eye();
      F22_->mult(Trans::N, CB, id);
      for (std::size_t c=0; c<dupd; c++) {
        auto pc = CB_perm_[c];
        for (std::size_t r=0; r<dupd; r++)
          CB(r, c) = id(CB_perm_[r], pc);
      }
    } else {
      id.eye();
      F22_->mult(Trans::N, id, CB);
    }
#else
    id.eye();
    F22_->mult(Trans::N, id, CB);
#endif
    TIMER_STOP(t_f22mult);
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f = F22_->get_stat("Flop_C_Mult");
    STRUMPACK_CB_SAMPLE_FLOPS(f);
    STRUMPACK_FLOPS(f);
#endif
    return CB;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const F_t* p, int task_depth) {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    const std::size_t pdsep = paF11.rows();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
    auto CB = get_dense_CB();
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared) grainsize(64)      \
  if(task_depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t c=0; c<dupd; c++) {
      auto pc = I[c];
      if (pc < pdsep) {
        for (std::size_t r=0; r<upd2sep; r++)
          paF11(I[r],pc) += CB(r,c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF21(I[r]-pdsep,pc) += CB(r,c);
      } else {
        for (std::size_t r=0; r<upd2sep; r++)
          paF12(I[r],pc-pdsep) += CB(r, c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF22(I[r]-pdsep,pc-pdsep) += CB(r,c);
      }
    }
    release_work_memory();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf,
   const FrontalMatrixMPI<scalar_t,integer_t>* pa) const {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    auto CB = get_dense_CB();
    ExtendAdd<scalar_t,integer_t>::extend_add_seq_copy_to_buffers
      (CB, sbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::sample_CB
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    auto I = this->upd_to_parent(pa);
    auto cR = R.extract_rows(I);
    DenseM_t cS(dim_upd(), R.cols());
    TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
    F22_->mult(op, cR, cS);
    TIMER_STOP(t_f22mult);
    S.scatter_rows_add(I, cS, task_depth);
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f = F22_->get_stat("Flop_C_Mult") + cS.rows()*cS.cols();
    STRUMPACK_CB_SAMPLE_FLOPS(f);
    STRUMPACK_FLOPS(f);
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::sample_CB_to_F11
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    std::size_t u2s;
    auto Ir = this->upd_to_parent(pa, u2s);
    auto Rcols = R.cols();
    DenseM_t cR(dupd, Rcols);
    for (std::size_t c=0; c<Rcols; c++) {
      for (std::size_t r=0; r<u2s; r++) cR(r,c) = R(Ir[r],c);
      for (std::size_t r=u2s; r<dupd; r++) cR(r,c) = scalar_t(0.);
    }
    DenseM_t cS(dupd, Rcols);
    TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
    F22_->mult(op, cR, cS);
    TIMER_STOP(t_f22mult);
    for (std::size_t c=0; c<Rcols; c++)
      for (std::size_t r=0; r<u2s; r++)
        S(Ir[r],c) += cS(r,c);
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f = F22_->get_stat("Flop_C_Mult") + Rcols*u2s;
    STRUMPACK_CB_SAMPLE_FLOPS(f);
    STRUMPACK_FLOPS(f);
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::sample_CB_to_F12
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    std::size_t u2s;
    auto Ir = this->upd_to_parent(pa, u2s);
    auto Rcols = R.cols();
    auto pds = pa->dim_sep();
    DenseM_t cR(dupd, Rcols), cS(dupd, Rcols);
    if (op == Trans::N) {
      for (std::size_t c=0; c<Rcols; c++) {
        for (std::size_t r=0; r<u2s; r++) cR(r,c) = scalar_t(0.);
        for (std::size_t r=u2s; r<dupd; r++)
          cR(r,c) = R(Ir[r]-pds,c);
      }
      TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
      F22_->mult(op, cR, cS);
      TIMER_STOP(t_f22mult);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=0; r<u2s; r++)
          S(Ir[r],c) += cS(r,c);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int f = F22_->get_stat("Flop_C_Mult") + Rcols*u2s;
      STRUMPACK_CB_SAMPLE_FLOPS(f);
      STRUMPACK_FLOPS(f);
#endif
    } else {
      for (std::size_t c=0; c<Rcols; c++) {
        for (std::size_t r=0; r<u2s; r++) cR(r,c) = R(Ir[r],c);
        for (std::size_t r=u2s; r<dupd; r++) cR(r,c) = scalar_t(0.);
      }
      TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
      F22_->mult(op, cR, cS);
      TIMER_STOP(t_f22mult);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=u2s; r<dupd; r++)
          S(Ir[r]-pds,c) += cS(r,c);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int f = F22_->get_stat("Flop_C_Mult") + Rcols*(dupd-u2s);
      STRUMPACK_CB_SAMPLE_FLOPS(f);
      STRUMPACK_FLOPS(f);
#endif
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::sample_CB_to_F21
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    std::size_t u2s;
    auto Ir = this->upd_to_parent(pa, u2s);
    auto pds = pa->dim_sep();
    auto Rcols = R.cols();
    DenseM_t cR(dupd, Rcols), cS(dupd, Rcols);
    if (op == Trans::N) {
      for (std::size_t c=0; c<Rcols; c++) {
        for (std::size_t r=0; r<u2s; r++) cR(r,c) = R(Ir[r],c);
        for (std::size_t r=u2s; r<dupd; r++) cR(r,c) = scalar_t(0.);
      }
      TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
      F22_->mult(op, cR, cS);
      TIMER_STOP(t_f22mult);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=u2s; r<dupd; r++)
          S(Ir[r]-pds,c) += cS(r,c);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int f = F22_->get_stat("Flop_C_Mult") + Rcols*(dupd-u2s);
      STRUMPACK_CB_SAMPLE_FLOPS(f);
      STRUMPACK_FLOPS(f);
#endif
    } else {
      for (std::size_t c=0; c<Rcols; c++) {
        for (std::size_t r=0; r<u2s; r++) cR(r,c) = scalar_t(0.);
        for (std::size_t r=u2s; r<dupd; r++)
          cR(r,c) = R(Ir[r]-pds,c);
      }
      TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
      F22_->mult(op, cR, cS);
      TIMER_STOP(t_f22mult);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=0; r<u2s; r++)
          S(Ir[r],c) += cS(r,c);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int f = F22_->get_stat("Flop_C_Mult") + Rcols*u2s;
      STRUMPACK_CB_SAMPLE_FLOPS(f);
      STRUMPACK_FLOPS(f);
#endif
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::sample_CB_to_F22
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    std::size_t u2s;
    auto Ir = this->upd_to_parent(pa, u2s);
    auto pds = pa->dim_sep();
    auto Rcols = R.cols();
    DenseM_t cR(dupd, Rcols);
    for (std::size_t c=0; c<Rcols; c++) {
      for (std::size_t r=0; r<u2s; r++) cR(r,c) = scalar_t(0.);
      for (std::size_t r=u2s; r<dupd; r++)
        cR(r,c) = R(Ir[r]-pds,c);
    }
    DenseM_t cS(dupd, Rcols);
    TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
    F22_->mult(op, cR, cS);
    TIMER_STOP(t_f22mult);
    for (std::size_t c=0; c<Rcols; c++)
      for (std::size_t r=u2s; r<dupd; r++)
        S(Ir[r]-pds,c) += cS(r,c);
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f = F22_->get_stat("Flop_C_Mult") + Rcols*(dupd-u2s);
    STRUMPACK_CB_SAMPLE_FLOPS(f);
    STRUMPACK_FLOPS(f);
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::element_extraction
  (const SpMat_t& A, const std::vector<std::vector<std::size_t>>& I,
   const std::vector<std::vector<std::size_t>>& J,
   std::vector<DenseMW_t>& B, int task_depth, bool skip_sparse) {
    if (skip_sparse) {
      for (std::size_t k=0; k<I.size(); k++)
        B[k].zero();
    } else {
      TIMER_TIME(TaskType::EXTRACT_SEP_2D, 2, t_ex_sep);
      for (std::size_t k=0; k<I.size(); k++)
        A.extract_separator(sep_end_, I[k], J[k], B[k], task_depth);
    }
    TIMER_TIME(TaskType::GET_SUBMATRIX_2D, 2, t_getsub);
    if (lchild_)
      lchild_->extract_CB_sub_matrix_blocks(I, J, B, task_depth);
    if (rchild_)
      rchild_->extract_CB_sub_matrix_blocks(I, J, B, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const Opts_t& opts, int etree_level, int task_depth) {
    if (lchild_)
      lchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    if (rchild_)
      rchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    if (!this->dim_blk()) return;
    TaskTimer t("");
    if (opts.print_compressed_front_stats()) t.start();
    construct_hierarchy(A, opts, task_depth);
    switch (opts.HODLR_options().compression_algorithm()) {
    case HODLR::CompressionAlgorithm::RANDOM_SAMPLING:
      compress_sampling(A, opts, task_depth); break;
    case HODLR::CompressionAlgorithm::ELEMENT_EXTRACTION: {
      auto lopts = opts;
      lopts.HODLR_options().set_rel_tol
        (lopts.HODLR_options().rel_tol() / (etree_level+1));
      compress_extraction(A, lopts, task_depth);
    } break;
    }
#if defined(STRUMPACK_COUNT_FLOPS)
    auto HOD_peak_mem = F11_.get_stat("Mem_Peak") +
      F12_.get_stat("Mem_Peak") + F21_.get_stat("Mem_Peak");
    if (F22_) HOD_peak_mem += F22_->get_stat("Mem_Peak");
    STRUMPACK_ADD_MEMORY(HOD_peak_mem*1.e6);
    STRUMPACK_SUB_MEMORY(HOD_peak_mem*1.e6);
    auto HOD_mem =
      F11_.get_stat("Mem_Fill") + F11_.get_stat("Mem_Factor") +
      F12_.get_stat("Mem_Fill") + F21_.get_stat("Mem_Fill");
    if (F22_) HOD_mem += F22_->get_stat("Mem_Fill");
    STRUMPACK_ADD_MEMORY(HOD_mem*1.e6);
#endif
    if (opts.print_compressed_front_stats()) {
      auto time = t.elapsed();
      float perbyte = 1.0e6 / sizeof(scalar_t);
      auto F11rank = F11_.get_stat("Rank_max");
      float F11nnzH = F11_.get_stat("Mem_Fill") * perbyte;
      float F11nnzFactors = F11_.get_stat("Mem_Factor") * perbyte;
      auto F12rank = F12_.get_stat("Rank_max");
      float F12nnzH = F12_.get_stat("Mem_Fill") * perbyte;
      auto F21rank = F21_.get_stat("Rank_max");
      float F21nnzH = F21_.get_stat("Mem_Fill") * perbyte;
      int F22rank = 0;
      float F22nnzH = 0;
      if (F22_) {
        F22rank = F22_->get_stat("Rank_max");
        F22nnzH = F22_->get_stat("Mem_Fill") * perbyte;
      }
      std::cout << "#   - HODLR front: Nsep= " << dim_sep()
                << " Nupd= " << dim_upd()
                << " level= " << etree_level << "\n#       "
                << " nnz(F11)= " << F11nnzH << " , nnz(factor(F11))= "
                << F11nnzFactors << " , rank(F11)= " << F11rank << " ,\n#       "
                << " nnz(F12)= " << F12nnzH << " , rank(F12)= " << F12rank << " , "
                << " nnz(F21)= " << F21nnzH << " , rank(F21)= " << F21rank << " ,\n#       "
                << " nnz(F22)= " << F22nnzH << " , rank(F22)= " << F22rank << " , "
                << (float(F11nnzH + F11nnzFactors + F12nnzH + F21nnzH + F22nnzH)
                    / (float(dim_blk())*dim_blk()) * 100.)
                << " %compression, time= " << time
                << " sec" << std::endl;
#if defined(STRUMPACK_COUNT_FLOPS)
      std::cout << "#        total memory: "
                << double(strumpack::params::memory) / 1.0e6 << " MB"
                << ",   peak memory: "
                << double(strumpack::params::peak_memory) / 1.0e6
                << " MB" << std::endl;
#endif
    }
    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::construct_hierarchy
  (const SpMat_t& A, const Opts_t& opts, int task_depth) {
    TIMER_TIME(TaskType::CONSTRUCT_HIERARCHY, 0, t_construct_h);
    TIMER_TIME(TaskType::EXTRACT_GRAPH, 0, t_graph_11);
    auto g = A.extract_graph
      (opts.separator_ordering_level(), sep_begin_, sep_end_);
    TIMER_STOP(t_graph_11);
    F11_ = HODLR::HODLRMatrix<scalar_t>
      (commself_, sep_tree_, g, opts.HODLR_options());
    if (dim_upd()) {
      if (opts.HODLR_options().compression_algorithm() ==
          HODLR::CompressionAlgorithm::ELEMENT_EXTRACTION) {
        TIMER_TIME(TaskType::EXTRACT_GRAPH, 0, t_graph_22);
        auto gCB = A.extract_graph_CB(opts.separator_ordering_level(), this->upd());
        auto g12 = A.extract_graph_sep_CB
          (opts.separator_ordering_level(), sep_begin_, sep_end_, this->upd());
        auto g21 = A.extract_graph_CB_sep
          (opts.separator_ordering_level(), sep_begin_, sep_end_, this->upd());
#if defined(STRUMPACK_PERMUTE_CB)
        CB_perm_.resize(dim_upd());
        CB_iperm_.resize(dim_upd());
        auto CB_tree = gCB.recursive_bisection
          (opts.HODLR_options().leaf_size(), 0,
           CB_perm_.data(), CB_iperm_.data(), 0, 0, gCB.size());
        gCB.permute(CB_perm_.data(), CB_iperm_.data());
        g12.permute_cols(CB_perm_.data());
        g21.permute_rows(CB_iperm_.data());
#else
        structured::ClusterTree CB_tree(dim_upd());
        CB_tree.refine(opts.HODLR_options().leaf_size());
#endif
        TIMER_STOP(t_graph_22);
        F22_ = std::unique_ptr<HODLR::HODLRMatrix<scalar_t>>
          (new HODLR::HODLRMatrix<scalar_t>
           (commself_, CB_tree, gCB, opts.HODLR_options()));
        auto knn = opts.HODLR_options().knn_lrbf();
        auto nns12 = HODLR::get_odiag_neighbors(knn, g12, g, gCB);
        auto nns21 = HODLR::get_odiag_neighbors(knn, g21, gCB, g);
        F12_ = HODLR::ButterflyMatrix<scalar_t>
          (F11_, *F22_, nns12, nns21, opts.HODLR_options());
        F21_ = HODLR::ButterflyMatrix<scalar_t>
          (*F22_, F11_, nns21, nns12, opts.HODLR_options());
      } else {
        structured::ClusterTree CB_tree(dim_upd());
        CB_tree.refine(opts.HODLR_options().leaf_size());
        F22_ = std::unique_ptr<HODLR::HODLRMatrix<scalar_t>>
          (new HODLR::HODLRMatrix<scalar_t>
           (commself_, CB_tree, opts.HODLR_options()));
        F12_ = HODLR::ButterflyMatrix<scalar_t>(F11_, *F22_);
        F21_ = HODLR::ButterflyMatrix<scalar_t>(*F22_, F11_);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::compress_extraction
  (const SpMat_t& A, const Opts_t& opts, int task_depth) {
    F11_.set_sampling_parameter(1.2);
    auto extract_F11 =
      [&](VecVec_t& I, VecVec_t& J, std::vector<DenseMW_t>& B,
          HODLR::ExtractionMeta& e) {
        for (auto& Ik : I) for (auto& i : Ik) i += this->sep_begin_;
        for (auto& Jk : J) for (auto& j : Jk) j += this->sep_begin_;
        element_extraction(A, I, J, B, task_depth);
      };
    { TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_f11_compress);
      F11_.compress(extract_F11); }
    { TIMER_TIME(TaskType::HSS_FACTOR, 0, t_fact);
      F11_.factor(); }
    compress_flops_F11();
    if (dim_upd()) {
      // F12_.set_sampling_parameter(2.0);
      // F21_.set_sampling_parameter(2.0);
      auto extract_F12 =
        [&](VecVec_t& I, VecVec_t& J, std::vector<DenseMW_t>& B,
            HODLR::ExtractionMeta& e) {
          for (auto& Ik : I) for (auto& i : Ik) i += this->sep_begin_;
#if defined(STRUMPACK_PERMUTE_CB)
          for (auto& Jk : J) for (auto& j : Jk) j = this->upd_[CB_iperm_[j]];
#else
          for (auto& Jk : J) for (auto& j : Jk) j = this->upd_[j];
#endif
          element_extraction(A, I, J, B, task_depth);
        };
      auto extract_F21 =
        [&](VecVec_t& I, VecVec_t& J, std::vector<DenseMW_t>& B,
            HODLR::ExtractionMeta& e) {
#if defined(STRUMPACK_PERMUTE_CB)
          for (auto& Ik : I) for (auto& i : Ik) i = this->upd_[CB_iperm_[i]];
#else
          for (auto& Ik : I) for (auto& i : Ik) i = this->upd_[i];
#endif
          for (auto& Jk : J) for (auto& j : Jk) j += this->sep_begin_;
          element_extraction(A, I, J, B, task_depth);
        };
      { TIMER_TIME(TaskType::LRBF_COMPRESS, 0, t_lrbf_compress);
        F12_.compress(extract_F12);
        F21_.compress(extract_F21); }
      compress_flops_F12_F21();

      // first construct S=-F21*inv(F11)*F12 using matvecs, then
      // construct F22+S using element extraction
      auto sample_Schur =
        [&](Trans op, scalar_t a, const DenseM_t& R, scalar_t b, DenseM_t& S) {
          TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
          TIMER_TIME(TaskType::HSS_SCHUR_PRODUCT, 2, t_sprod);
          DenseM_t F12R(F12_.rows(), R.cols()),
            invF11F12R(F11_.rows(), R.cols());
          auto& F1 = (op == Trans::N) ? F12_ : F21_;
          auto& F2 = (op == Trans::N) ? F21_ : F12_;
          a = -a; // construct S=b*S-a*F21*inv(F11)*F12
          F1.mult(op, R, F12R);
          if (a != scalar_t(1.))
            F12R.scale(a);
          long long int invf11_mult_flops = F11_.inv_mult(op, F12R, invF11F12R);
          if (b != scalar_t(0.)) {
            DenseM_t Stmp(S.rows(), S.cols());
            F2.mult(op, invF11F12R, Stmp);
            S.scale_and_add(b, Stmp);
          } else F2.mult(op, invF11F12R, S);
          compress_flops_Schur(invf11_mult_flops);
        };
      HODLR::ButterflyMatrix<scalar_t> Schur(*F22_, *F22_);
      Schur.compress(sample_Schur);
      auto extract_F22 =
        [&](VecVec_t& I, VecVec_t& J, std::vector<DenseMW_t>& B,
            HODLR::ExtractionMeta& e) {
#if defined(STRUMPACK_PERMUTE_CB)
          for (auto& Ik : I) for (auto& i : Ik) i = this->upd_[CB_iperm_[i]];
          for (auto& Jk : J) for (auto& j : Jk) j = this->upd_[CB_iperm_[j]];
#else
          for (auto& Ik : I) for (auto& i : Ik) i = this->upd_[i];
          for (auto& Jk : J) for (auto& j : Jk) j = this->upd_[j];
#endif
          element_extraction(A, I, J, B, task_depth, true);
          Schur.extract_add_elements(e, B);
        };
      TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_f22_compress);
      F22_->compress(extract_F22);
      compress_flops_F22();
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::compress_sampling
  (const SpMat_t& A, const Opts_t& opts, int task_depth) {
    const auto dsep = dim_sep();
    const auto dupd = dim_upd();
    auto sample_F11 = [&](Trans op, const DenseM_t& R, DenseM_t& S) {
      TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
      S.zero();
      { TIMER_TIME(TaskType::FRONT_MULTIPLY_2D, 1, t_fmult);
        A.front_multiply_F11(op, sep_begin_, sep_end_, R, S, 0); }
      TIMER_TIME(TaskType::UUTXR, 1, t_UUtxR);
      if (lchild_) lchild_->sample_CB_to_F11(op, R, S, this, task_depth);
      if (rchild_) rchild_->sample_CB_to_F11(op, R, S, this, task_depth);
    };
    { TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_f11_compress);
      F11_.compress(sample_F11); }
    { TIMER_TIME(TaskType::HSS_FACTOR, 0, t_fact);
      F11_.factor(); }
    compress_flops_F11();
    if (dupd) {
      auto sample_F12 = [&]
        (Trans op, scalar_t a, const DenseM_t& R, scalar_t b, DenseM_t& S) {
        TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
        DenseM_t lR(R);
        lR.scale(a);
        if (b == scalar_t(0.)) S.zero();
        else S.scale(b);
        { TIMER_TIME(TaskType::FRONT_MULTIPLY_2D, 1, t_fmult);
          A.front_multiply_F12(op, sep_begin_, sep_end_, this->upd_, R, S, 0); }
        TIMER_TIME(TaskType::UUTXR, 1, t_UUtxR);
        if (lchild_) lchild_->sample_CB_to_F12(op, lR, S, this, task_depth);
        if (rchild_) rchild_->sample_CB_to_F12(op, lR, S, this, task_depth);
      };
      auto sample_F21 = [&]
        (Trans op, scalar_t a, const DenseM_t& R, scalar_t b, DenseM_t& S) {
        TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
        DenseM_t lR(R), lS(S);
        lR.scale(a);
        if (b == scalar_t(0.)) S.zero();
        else S.scale(b);
        { TIMER_TIME(TaskType::FRONT_MULTIPLY_2D, 1, t_fmult);
          A.front_multiply_F21(op, sep_begin_, sep_end_, this->upd_, R, S, 0); }
        TIMER_TIME(TaskType::UUTXR, 1, t_UUtxR);
        if (lchild_) lchild_->sample_CB_to_F21(op, lR, S, this, task_depth);
        if (rchild_) rchild_->sample_CB_to_F21(op, lR, S, this, task_depth);
      };
      { TIMER_TIME(TaskType::LRBF_COMPRESS, 0, t_lrbf_compress);
        F12_.compress(sample_F12);
        F21_.compress(sample_F21, F12_.get_stat("Rank_max")+10); }
      compress_flops_F12_F21();
      long long int invf11_mult_flops = 0;
      auto sample_CB =
        [&](Trans op, const DenseM_t& R, DenseM_t& S) {
          TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
          TIMER_TIME(TaskType::UUTXR, 1, t_UUtxR);
          TIMER_TIME(TaskType::HSS_SCHUR_PRODUCT, 2, t_sprod);
          DenseM_t F12R(dsep, R.cols()), invF11F12R(dsep, R.cols());
          if (op == Trans::N) {
            F12_.mult(op, R, F12R);
            invf11_mult_flops += F11_.inv_mult(op, F12R, invF11F12R);
            F21_.mult(op, invF11F12R, S);
          } else {
            F21_.mult(op, R, F12R);
            invf11_mult_flops += F11_.inv_mult(op, F12R, invF11F12R);
            F12_.mult(op, invF11F12R, S);
          }
          TIMER_STOP(t_sprod);
          S.scale(-1.);
          compress_flops_Schur(invf11_mult_flops);
          if (lchild_) lchild_->sample_CB_to_F22(op, R, S, this, task_depth);
          if (rchild_) rchild_->sample_CB_to_F22(op, R, S, this, task_depth);
        };
      { TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_f22_compress);
        F22_->compress(sample_CB, std::max(F12_.get_stat("Rank_max"),
                                           F21_.get_stat("Rank_max"))); }
      compress_flops_F22();
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::compress_flops_F11() {
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f11_fill_flops = F11_.get_stat("Flop_Fill");
    long long int f11_factor_flops = F11_.get_stat("Flop_Factor");
    STRUMPACK_HODLR_F11_FILL_FLOPS(f11_fill_flops);
    STRUMPACK_ULV_FACTOR_FLOPS(f11_factor_flops);
    STRUMPACK_FLOPS(f11_fill_flops + f11_factor_flops);
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::compress_flops_F12_F21() {
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f12_fill_flops = F12_.get_stat("Flop_Fill");
    long long int f21_fill_flops = F21_.get_stat("Flop_Fill");
    STRUMPACK_HODLR_F12_FILL_FLOPS(f12_fill_flops);
    STRUMPACK_HODLR_F21_FILL_FLOPS(f21_fill_flops);
    STRUMPACK_FLOPS(f12_fill_flops + f21_fill_flops);
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::compress_flops_F22() {
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f22_fill_flops = F22_->get_stat("Flop_Fill");
    STRUMPACK_HODLR_F22_FILL_FLOPS(f22_fill_flops);
    STRUMPACK_FLOPS(f22_fill_flops);
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::compress_flops_Schur
  (long long int invf11_mult_flops) {
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f21_mult_flops = F21_.get_stat("Flop_C_Mult"),
      //invf11_mult_flops = F11_.get_stat("Flop_C_Mult"),
      f12_mult_flops = F12_.get_stat("Flop_C_Mult");
    long long int schur_flops = f21_mult_flops + invf11_mult_flops
      + f12_mult_flops; // + S.rows()*S.cols(); // extra scaling?
    STRUMPACK_SCHUR_FLOPS(schur_flops);
    STRUMPACK_FLOPS(schur_flops);
    STRUMPACK_HODLR_F21_MULT_FLOPS(f21_mult_flops);
    STRUMPACK_HODLR_F12_MULT_FLOPS(f12_mult_flops);
    STRUMPACK_HODLR_INVF11_MULT_FLOPS(invf11_mult_flops);
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::extract_CB_sub_matrix
  (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
   DenseM_t& B, int task_depth) const {
    TIMER_TIME(TaskType::HSS_EXTRACT_SCHUR, 3, t_ex_schur);
    std::vector<std::size_t> lJ, oJ;
    this->find_upd_indices(J, lJ, oJ);
    auto n = lJ.size();
    if (n == 0) return;
    std::vector<std::size_t> lI, oI;
    this->find_upd_indices(I, lI, oI);
    auto m = lI.size();
    if (m == 0) return;
#if defined(STRUMPACK_PERMUTE_CB)
    if (CB_perm_.size() == std::size_t(dim_upd())) {
      for (auto& i : lI) i = CB_perm_[i];
      for (auto& j : lJ) j = CB_perm_[j];
    }
#endif
    DenseM_t Bloc(m, n);
    F22_->extract_elements(lI, lJ, Bloc);
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        B(oI[i], oJ[j]) += Bloc(i, j);
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*m*n);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::extract_CB_sub_matrix_blocks
  (const std::vector<std::vector<std::size_t>>& I,
   const std::vector<std::vector<std::size_t>>& J,
   std::vector<DenseM_t>& Bseq, int task_depth) const {
    if (!dim_upd()) return;
    TIMER_TIME(TaskType::HSS_EXTRACT_SCHUR, 3, t_ex_schur);
    auto nB = I.size();
    std::vector<std::vector<std::size_t>> lI(nB), lJ(nB), oI(nB), oJ(nB);
    std::vector<DenseM_t> Bloc(nB);
    for (std::size_t b=0; b<nB; b++) {
      this->find_upd_indices(I[b], lI[b], oI[b]);
      this->find_upd_indices(J[b], lJ[b], oJ[b]);
#if defined(STRUMPACK_PERMUTE_CB)
      if (CB_perm_.size() == std::size_t(dim_upd())) {
        for (auto& li : lI[b]) li = CB_perm_[li];
        for (auto& lj : lJ[b]) lj = CB_perm_[lj];
      }
#endif
      Bloc[b] = DenseM_t(lI[b].size(), lJ[b].size());
    }
    F22_->extract_elements(lI, lJ, Bloc);
    for (std::size_t b=0; b<nB; b++) {
      Bseq.emplace_back(I[b].size(), J[b].size());
      Bseq[b].zero();
      for (std::size_t j=0; j<lJ[b].size(); j++)
        for (std::size_t i=0; i<lI[b].size(); i++)
          Bseq[b](oI[b][i], oJ[b][j]) = Bloc[b](i, j);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::extract_CB_sub_matrix_blocks
  (const std::vector<std::vector<std::size_t>>& I,
   const std::vector<std::vector<std::size_t>>& J,
   std::vector<DenseMW_t>& Bseq, int task_depth) const {
    if (!dim_upd()) return;
    TIMER_TIME(TaskType::HSS_EXTRACT_SCHUR, 3, t_ex_schur);
    auto nB = I.size();
    std::vector<std::vector<std::size_t>> lI(nB), lJ(nB), oI(nB), oJ(nB);
    std::vector<DenseM_t> Bloc(nB);
    for (std::size_t b=0; b<nB; b++) {
      this->find_upd_indices(I[b], lI[b], oI[b]);
      this->find_upd_indices(J[b], lJ[b], oJ[b]);
#if defined(STRUMPACK_PERMUTE_CB)
      if (CB_perm_.size() == std::size_t(dim_upd())) {
        for (auto& li : lI[b]) li = CB_perm_[li];
        for (auto& lj : lJ[b]) lj = CB_perm_[lj];
      }
#endif
      Bloc[b] = DenseM_t(lI[b].size(), lJ[b].size());
    }
    F22_->extract_elements(lI, lJ, Bloc);
    for (std::size_t b=0; b<nB; b++) {
      for (std::size_t j=0; j<lJ[b].size(); j++)
        for (std::size_t i=0; i<lI[b].size(); i++)
          Bseq[b](oI[b][i], oJ[b][j]) += Bloc[b](i, j);
      STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*lI[b].size()*lJ[b].size());
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t bupd(dim_upd(), b.cols(), work[0], 0, 0);
    bupd.zero();
    if (lchild_) {
      lchild_->forward_multifrontal_solve
        (b, work+1, etree_level+1, task_depth);
      DenseMW_t CBch(lchild_->dim_upd(), b.cols(), work[1], 0, 0);
      lchild_->extend_add_b(b, bupd, CBch, this);
    }
    if (rchild_) {
      rchild_->forward_multifrontal_solve
        (b, work+1, etree_level+1, task_depth);
      DenseMW_t CBch(rchild_->dim_upd(), b.cols(), work[1], 0, 0);
      rchild_->extend_add_b(b, bupd, CBch, this);
    }
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
      DenseM_t rhs(bloc);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int solve_flops = F11_.solve(rhs, bloc);
      STRUMPACK_FLOPS(solve_flops);
#else
      F11_.solve(rhs, bloc);
#endif
      if (dim_upd()) {
        DenseM_t tmp(bupd.rows(), bupd.cols());
        F21_.mult(Trans::N, bloc, tmp);
#if defined(STRUMPACK_PERMUTE_CB)
        DenseM_t ptmp(tmp.rows(), tmp.cols());
        for (std::size_t r=0; r<tmp.rows(); r++)
          for (std::size_t c=0; c<tmp.cols(); c++)
            ptmp(r, c) = tmp(CB_perm_[r], c);
        bupd.scaled_add(scalar_t(-1.), ptmp);
#else
        bupd.scaled_add(scalar_t(-1.), tmp);
#endif
        STRUMPACK_FLOPS(F21_.get_stat("Flop_C_Mult") +
                        2*bupd.rows()*bupd.cols());
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t yupd(dim_upd(), y.cols(), work[0], 0, 0);
    if (dim_sep() && dim_upd()) {
      DenseM_t tmp(dim_sep(), y.cols()), tmp2(dim_sep(), y.cols());
#if defined(STRUMPACK_PERMUTE_CB)
      DenseM_t pyupd(yupd.rows(), yupd.cols());
      for (std::size_t r=0; r<yupd.rows(); r++)
        for (std::size_t c=0; c<yupd.cols(); c++)
          pyupd(r, c) = yupd(CB_iperm_[r], c);
      F12_.mult(Trans::N, pyupd, tmp);
#else
      F12_.mult(Trans::N, yupd, tmp);
#endif
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int solve_flops = F11_.solve(tmp, tmp2);
#else
      F11_.solve(tmp, tmp2);
#endif
      DenseMW_t yloc(dim_sep(), y.cols(), y, this->sep_begin_, 0);
      yloc.scaled_add(scalar_t(-1.), tmp2);
      STRUMPACK_FLOPS(F12_.get_stat("Flop_C_Mult") +
                      solve_flops + 2*yloc.rows()*yloc.cols());
    }
    // this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
    if (lchild_) {
      DenseMW_t CB(lchild_->dim_upd(), y.cols(), work[1], 0, 0);
      lchild_->extract_b(y, yupd, CB, this);
      lchild_->backward_multifrontal_solve
        (y, work+1, etree_level+1, task_depth);
    }
    if (rchild_) {
      DenseMW_t CB(rchild_->dim_upd(), y.cols(), work[1], 0, 0);
      rchild_->extract_b(y, yupd, CB, this);
      rchild_->backward_multifrontal_solve
        (y, work+1, etree_level+1, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> integer_t
  FrontalMatrixHODLR<scalar_t,integer_t>::front_rank(int task_depth) const {
    return std::max(F11_.get_stat("Rank_max"),
                    std::max(F12_.get_stat("Rank_max"),
                             F21_.get_stat("Rank_max")));
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::print_rank_statistics
  (std::ostream &out) const {
    if (lchild_) lchild_->print_rank_statistics(out);
    if (rchild_) rchild_->print_rank_statistics(out);
    out << "# HODLRMatrix rank info .... TODO" << std::endl;
    // _H.print_info(out);
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixHODLR<scalar_t,integer_t>::node_factor_nonzeros() const {
    return (F11_.get_stat("Mem_Fill") + F11_.get_stat("Mem_Factor")
            + F12_.get_stat("Mem_Fill") + F21_.get_stat("Mem_Fill"))
      * 1.0e6 / sizeof(scalar_t);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::draw_node
  (std::ostream& of, bool is_root) const {
    std::cout << "TODO draw" << std::endl;
    // if (is_root) _H.draw(of, sep_begin_, sep_begin_);
    // else _H.child(0)->draw(of, sep_begin_, sep_begin_);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::partition
  (const Opts_t& opts, const SpMat_t& A, integer_t* sorder,
   bool is_root, int task_depth) {
    auto g = A.extract_graph
      (opts.separator_ordering_level(), sep_begin_, sep_end_);
    sep_tree_ = g.recursive_bisection
      (opts.HODLR_options().leaf_size(), 0,
       sorder+sep_begin_, nullptr, 0, 0, dim_sep());
    std::vector<integer_t> siorder(dim_sep());
    for (integer_t i=sep_begin_; i<sep_end_; i++)
      siorder[sorder[i]] = i - sep_begin_;
    for (integer_t i=sep_begin_; i<sep_end_; i++)
      sorder[i] += sep_begin_;
  }

  // explicit template instantiations
  template class FrontalMatrixHODLR<float,int>;
  template class FrontalMatrixHODLR<double,int>;
  template class FrontalMatrixHODLR<std::complex<float>,int>;
  template class FrontalMatrixHODLR<std::complex<double>,int>;

  template class FrontalMatrixHODLR<float,long int>;
  template class FrontalMatrixHODLR<double,long int>;
  template class FrontalMatrixHODLR<std::complex<float>,long int>;
  template class FrontalMatrixHODLR<std::complex<double>,long int>;

  template class FrontalMatrixHODLR<float,long long int>;
  template class FrontalMatrixHODLR<double,long long int>;
  template class FrontalMatrixHODLR<std::complex<float>,long long int>;
  template class FrontalMatrixHODLR<std::complex<double>,long long int>;

} // end namespace strumpack
