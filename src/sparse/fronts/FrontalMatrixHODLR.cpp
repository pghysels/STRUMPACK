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

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  FrontalMatrixHODLR<scalar_t,integer_t>::FrontalMatrixHODLR
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd),
    commself_(MPI_COMM_SELF) { }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::release_work_memory() {
    F22_.reset(nullptr);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const F_t* p, int task_depth) {
    const std::size_t pdsep = paF11.rows();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
    DenseM_t CB(dupd, dupd);
    {
      DenseM_t id(dupd, dupd);
      id.eye();
      TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
      F22_->mult(Trans::N, id, CB);
      TIMER_STOP(t_f22mult);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int f = F22_->get_stat("Flop_C_Mult");
      STRUMPACK_CB_SAMPLE_FLOPS(f);
      STRUMPACK_FLOPS(f);
#endif
    }
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
  (const SpMat_t& A, const std::vector<std::size_t>& gI,
   const std::vector<std::size_t>& gJ, DenseM_t& B, int task_depth) {
    { TIMER_TIME(TaskType::EXTRACT_SEP_2D, 2, t_ex_sep);
      A.extract_separator(sep_end_, gI, gJ, B, task_depth); }
    TIMER_TIME(TaskType::GET_SUBMATRIX_2D, 2, t_getsub);
    if (lchild_)
      lchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
    if (rchild_)
      rchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const Opts_t& opts, int etree_level, int task_depth) {
    if (task_depth == 0)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      multifrontal_factorization_node(A, opts, etree_level, task_depth);
    else multifrontal_factorization_node(A, opts, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::multifrontal_factorization_node
  (const SpMat_t& A, const Opts_t& opts, int etree_level, int task_depth) {
    bool tasked = task_depth < params::task_recursion_cutoff_level;
    if (tasked) {
      if (lchild_)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        lchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth+1);
      if (rchild_)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        rchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth+1);
#pragma omp taskwait
    } else {
      if (lchild_)
        lchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
      if (rchild_)
        rchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
    }
    if (!this->dim_blk()) return;
    TaskTimer t("");
    if (/*etree_level == 0 && */opts.print_root_front_stats()) t.start();
    construct_hierarchy(A, opts, task_depth);
    switch (opts.HODLR_options().compression_algorithm()) {
    case HODLR::CompressionAlgorithm::RANDOM_SAMPLING:
      compress_sampling(A, opts, task_depth); break;
    case HODLR::CompressionAlgorithm::ELEMENT_EXTRACTION:
      compress_extraction(A, opts, task_depth); break;
    }
    if (/*etree_level == 0 && */opts.print_root_front_stats()) {
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
                << " Nupd= " << dim_upd() << "\n#       "
                << " nnz(F11)= " << F11nnzH << " , nnz(factor(F11))= "
                << F11nnzFactors << " , rank(F11)= " << F11rank << " ,\n#       "
                << " nnz(F12)= " << F12nnzH << " , rank(F12)= " << F12rank << " , "
                << " nnz(F21)= " << F21nnzH << " , rank(F21)= " << F21rank << " ,\n#       "
                << " nnz(F22)= " << F22nnzH << " , rank(F22)= " << F22rank << " , "
                << (float(F11nnzH + F11nnzFactors + F12nnzH + F21nnzH + F22nnzH)
                    / (float(dim_blk())*dim_blk()) * 100.)
                << " %compression, time= " << time
                << " sec" << std::endl;
    }
    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::construct_hierarchy
  (const SpMat_t& A, const Opts_t& opts, int task_depth) {
    auto g = A.extract_graph
      (opts.separator_ordering_level(), sep_begin_, sep_end_);
    F11_ = HODLR::HODLRMatrix<scalar_t>
      (commself_, sep_tree_, g, opts.HODLR_options());
    if (dim_upd()) {
      HSS::HSSPartitionTree CB_tree(dim_upd());
      CB_tree.refine(opts.HODLR_options().leaf_size());
      if (opts.HODLR_options().compression_algorithm() ==
          HODLR::CompressionAlgorithm::ELEMENT_EXTRACTION) {
        auto gCB = A.extract_graph_CB(opts.separator_ordering_level(), this->upd());
        F22_ = std::unique_ptr<HODLR::HODLRMatrix<scalar_t>>
          (new HODLR::HODLRMatrix<scalar_t>
           (commself_, CB_tree, gCB, opts.HODLR_options()));
        auto g12 = A.extract_graph_sep_CB
          (opts.separator_ordering_level(), sep_begin_, sep_end_, this->upd());
        auto g21 = A.extract_graph_CB_sep
          (opts.separator_ordering_level(), sep_begin_, sep_end_, this->upd());
        auto knn = opts.HODLR_options().knn_lrbf();
        auto nns12 = HODLR::get_odiag_neighbors(knn, g12, g, gCB);
        auto nns21 = HODLR::get_odiag_neighbors(knn, g21, gCB, g);
        F12_ = HODLR::ButterflyMatrix<scalar_t>
          (F11_, *F22_, nns12, nns21, opts.HODLR_options());
        F21_ = HODLR::ButterflyMatrix<scalar_t>
          (*F22_, F11_, nns21, nns12, opts.HODLR_options());
      } else {
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
    auto extract_F11 =
      [&](VecVec_t& I, VecVec_t& J, std::vector<DenseMW_t>& B,
          HODLR::ExtractionMeta& e) {
        for (auto& Ik : I) for (auto& i : Ik) i += this->sep_begin_;
        for (auto& Jk : J) for (auto& j : Jk) j += this->sep_begin_;
        for (std::size_t k=0; k<I.size(); k++)
          element_extraction(A, I[k], J[k], B[k], task_depth);
      };
    { TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_f11_compress);
      F11_.compress(extract_F11); }
    { TIMER_TIME(TaskType::HSS_FACTOR, 0, t_fact);
      F11_.factor(); }
    compress_flops_F11();
    if (dim_upd()) {
      auto extract_F12 =
        [&](VecVec_t& I, VecVec_t& J, std::vector<DenseMW_t>& B,
            HODLR::ExtractionMeta& e) {
          for (auto& Ik : I) for (auto& i : Ik) i += this->sep_begin_;
          for (auto& Jk : J) for (auto& j : Jk) j = this->upd_[j];
          for (std::size_t k=0; k<I.size(); k++)
            element_extraction(A, I[k], J[k], B[k], task_depth);
        };
      auto extract_F21 =
        [&](VecVec_t& I, VecVec_t& J, std::vector<DenseMW_t>& B,
            HODLR::ExtractionMeta& e) {
          for (auto& Ik : I) for (auto& i : Ik) i = this->upd_[i];
          for (auto& Jk : J) for (auto& j : Jk) j += this->sep_begin_;
          for (std::size_t k=0; k<I.size(); k++)
            element_extraction(A, I[k], J[k], B[k], task_depth);
        };
      { TIMER_TIME(TaskType::LRBF_COMPRESS, 0, t_lrbf_compress);
        F12_.compress(extract_F12);
        F21_.compress(extract_F21); }
      compress_flops_F12_F21();

      // first construct S=-F21*inv(F11)*F12 using matvecs, then
      // construct F22+S using element extraction
      long long int invf11_mult_flops = 0;
      auto sample_Schur =
        [&](Trans op, scalar_t a, const DenseM_t& R, scalar_t b, DenseM_t& S) {
          TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
          TIMER_TIME(TaskType::HSS_SCHUR_PRODUCT, 2, t_sprod);
          DenseM_t F12R(F12_.rows(), R.cols()),
            invF11F12R(F11_.rows(), R.cols());
          auto& F1 = (op == Trans::N) ? F12_ : F21_;
          auto& F2 = (op == Trans::N) ? F21_ : F12_;
          a = -a; // construct S=-F21*inv(F11)*F12
          if (a != scalar_t(1.)) {
            DenseM_t Rtmp(R);
            Rtmp.scale(a);
            F1.mult(op, Rtmp, F12R);
          } else F1.mult(op, R, F12R);
          invf11_mult_flops += F11_.inv_mult(op, F12R, invF11F12R);
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
          for (auto& Ik : I) for (auto& i : Ik) i = this->upd_[i];
          for (auto& Jk : J) for (auto& j : Jk) j = this->upd_[j];
          for (std::size_t k=0; k<I.size(); k++)
            element_extraction(A, I[k], J[k], B[k], task_depth);
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
    DenseM_t Bloc(m, n);
    F22_->extract_elements(lI, lJ, Bloc);
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<m; i++)
        B(oI[i], oJ[j]) += Bloc(i, j);
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*m*n);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    if (task_depth == 0)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      fwd_solve_node(b, work, etree_level, task_depth);
    else fwd_solve_node(b, work, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::fwd_solve_node
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t bupd(dim_upd(), b.cols(), work[0], 0, 0);
    bupd.zero();
    this->fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
      DenseM_t rhs(bloc);
      long long int solve_flops = F11_.solve(rhs, bloc);
      STRUMPACK_FLOPS(solve_flops);
      if (dim_upd()) {
        DenseM_t tmp(bupd.rows(), bupd.cols());
        F21_.mult(Trans::N, bloc, tmp);
        bupd.scaled_add(scalar_t(-1.), tmp);
        STRUMPACK_FLOPS(F21_.get_stat("Flop_C_Mult") +
                        2*bupd.rows()*bupd.cols());
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    if (task_depth == 0)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      bwd_solve_node(y, work, etree_level, task_depth);
    else bwd_solve_node(y, work, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::bwd_solve_node
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t yupd(dim_upd(), y.cols(), work[0], 0, 0);
    if (dim_sep() && dim_upd()) {
      DenseM_t tmp(dim_sep(), y.cols()), tmp2(dim_sep(), y.cols());
      F12_.mult(Trans::N, yupd, tmp);
      long long int solve_flops = F11_.solve(tmp, tmp2);
      DenseMW_t yloc(dim_sep(), y.cols(), y, this->sep_begin_, 0);
      yloc.scaled_add(scalar_t(-1.), tmp2);
      STRUMPACK_FLOPS(F12_.get_stat("Flop_C_Mult") +
                      solve_flops + 2*yloc.rows()*yloc.cols());
    }
    this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> integer_t
  FrontalMatrixHODLR<scalar_t,integer_t>::maximum_rank(int task_depth) const {
    integer_t r = std::max(F11_.get_stat("Rank_max"),
                           std::max(F12_.get_stat("Rank_max"),
                                    F21_.get_stat("Rank_max")));
    integer_t rl = 0, rr = 0;
    if (lchild_)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
      rl = lchild_->maximum_rank(task_depth+1);
    if (rchild_)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
      rr = rchild_->maximum_rank(task_depth+1);
#pragma omp taskwait
    return std::max(r, std::max(rl, rr));
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
      (opts.compression_leaf_size(), 0,
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
