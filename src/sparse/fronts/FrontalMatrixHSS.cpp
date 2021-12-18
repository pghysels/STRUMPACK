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

#include "FrontalMatrixHSS.hpp"
#include "sparse/CSRGraph.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  FrontalMatrixHSS<scalar_t,integer_t>::FrontalMatrixHSS
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : FrontalMatrix<scalar_t,integer_t>
    (nullptr, nullptr, sep, sep_begin, sep_end, upd) {
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::release_work_memory() {
    ThetaVhatC_or_VhatCPhiC_.clear();
    H_.delete_trailing_block();
    R1.clear();
    Sr2.clear();
    Sc2.clear();
    DUB01_.clear();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const FrontalMatrix<scalar_t,integer_t>* p, int task_depth) {
    const std::size_t pdsep = p->dim_sep();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);

    auto F22 = H_.child(1)->dense();
    if (Theta_.cols() < Phi_.cols())
      // S = F22 - Theta_ * ThetaVhatC_or_VhatCPhiC_
      gemm(Trans::N, Trans::N, scalar_t(-1.), Theta_,
           ThetaVhatC_or_VhatCPhiC_,
           scalar_t(1.), F22, task_depth);
    else
      // S = F22 - ThetaVhatC_or_VhatCPhiC_ * Phi_'
      gemm(Trans::N, Trans::C, scalar_t(-1.),
           ThetaVhatC_or_VhatCPhiC_, Phi_,
           scalar_t(1.), F22, task_depth);
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)                            \
  if(task_depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t c=0; c<dupd; c++) {
      std::size_t pc = I[c];
      if (pc < pdsep) {
        for (std::size_t r=0; r<upd2sep; r++)
          paF11(I[r],pc) += F22(r,c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF21(I[r]-pdsep,pc) += F22(r,c);
      } else {
        for (std::size_t r=0; r<upd2sep; r++)
          paF12(I[r],pc-pdsep) += F22(r,c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF22(I[r]-pdsep,pc-pdsep) += F22(r,c);
      }
    }
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*static_cast<long long int>(dupd*dupd));
    release_work_memory();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::extract_CB_sub_matrix
  (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
   DenseM_t& B, int task_depth) const {
    std::vector<std::size_t> lJ, oJ;
    this->find_upd_indices(J, lJ, oJ);
    if (lJ.empty()) return;
    std::vector<std::size_t> lI, oI;
    this->find_upd_indices(I, lI, oI);
    if (lI.empty()) return;

#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
    {
      auto M = H_.child(1)->extract(lI, lJ);
      for (std::size_t j=0; j<lJ.size(); j++)
        for (std::size_t i=0; i<lI.size(); i++)
          B(oI[i], oJ[j]) += M(i, j);

      if (Theta_.cols() < Phi_.cols()) {
        // S = F22 - _Theta * _ThetaVhatC_or_VhatCPhiC
        auto r = Theta_.cols();
        DenseM_t r_theta(lI.size(), r), c_vhatphiC(r, lJ.size());
        for (std::size_t i=0; i<lI.size(); i++)
          copy(1, r, Theta_, lI[i], 0, r_theta, i, 0);
        for (std::size_t j=0; j<lJ.size(); j++)
          copy(r, 1, ThetaVhatC_or_VhatCPhiC_, 0, lJ[j], c_vhatphiC, 0, j);
        gemm(Trans::N, Trans::N, scalar_t(-1.), r_theta, c_vhatphiC,
             scalar_t(0.), M, task_depth);
        for (std::size_t j=0; j<lJ.size(); j++)
          for (std::size_t i=0; i<lI.size(); i++)
            B(oI[i], oJ[j]) += M(i, j);
        STRUMPACK_EXTRACTION_FLOPS
          (gemm_flops(Trans::N, Trans::N, scalar_t(-1.), r_theta, c_vhatphiC,
                      scalar_t(0.)) + lJ.size()*lI.size());
      } else {
        // S = F22 - ThetaVhatC_or_VhatCPhiC_ * Phi_'
        auto r = Phi_.cols();
        DenseM_t r_thetavhat(lI.size(), r), r_phi(lJ.size(), r);
        for (std::size_t i=0; i<lI.size(); i++)
          copy(1, r, ThetaVhatC_or_VhatCPhiC_, lI[i], 0, r_thetavhat, i, 0);
        for (std::size_t j=0; j<lJ.size(); j++)
          copy(1, r, Phi_, lJ[j], 0, r_phi, j, 0);
        gemm(Trans::N, Trans::C, scalar_t(-1.), r_thetavhat, r_phi,
             scalar_t(0.), M, task_depth);
        for (std::size_t j=0; j<lJ.size(); j++)
          for (std::size_t i=0; i<lI.size(); i++)
            B(oI[i], oJ[j]) += M(i, j);
        STRUMPACK_EXTRACTION_FLOPS
          (gemm_flops(Trans::N, Trans::C, scalar_t(-1.), r_thetavhat, r_phi,
                      scalar_t(0.)) + lJ.size()*lI.size());
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::sample_CB
  (const Opts_t& opts, const DenseM_t& R,
   DenseM_t& Sr, DenseM_t& Sc, F_t* pa, int task_depth) {
    if (!dim_upd()) return;
    auto I = this->upd_to_parent(pa);
    auto cR = R.extract_rows(I);
    auto dchild = R1.cols();
    auto dall = R.cols();
    if (dchild > 0 && opts.indirect_sampling()) {
      DenseM_t cSr, cSc;
      DenseMW_t cRd0(cR.rows(), dchild, cR, 0, 0);
      TIMER_TIME(TaskType::HSS_SCHUR_PRODUCT, 2, t_sprod);
      H_.Schur_product_indirect(DUB01_, R1, cRd0, Sr2, Sc2, cSr, cSc);
      TIMER_STOP(t_sprod);
      DenseMW_t(Sr.rows(), dchild, Sr, 0, 0)
        .scatter_rows_add(I, cSr, task_depth);
      DenseMW_t(Sc.rows(), dchild, Sc, 0, 0)
        .scatter_rows_add(I, cSc, task_depth);
      R1.clear();
      Sr2.clear();
      Sc2.clear();
      if (dall > dchild) {
        DenseMW_t Srdd(Sr.rows(), dall-dchild, Sr, 0, dchild);
        DenseMW_t Scdd(Sc.rows(), dall-dchild, Sc, 0, dchild);
        DenseMW_t cRdd(cR.rows(), dall-dchild, cR, 0, dchild);
        sample_CB_direct(cRdd, Srdd, Scdd, I, task_depth);
      }
    } else sample_CB_direct(cR, Sr, Sc, I, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::sample_CB_direct
  (const DenseM_t& cR, DenseM_t& Sr, DenseM_t& Sc,
   const std::vector<std::size_t>& I, int task_depth) {
#if 0
    // TODO count flops here
    auto cSr = H_.child(1)->apply(cR);
    auto cSc = H_.child(1)->applyC(cR);
    if (Theta_.cols() < Phi_.cols()) {
      // TODO 2 tasks?
      DenseM_t tmp(ThetaVhatC_or_VhatCPhiC_.rows(), cR.cols());
      gemm(Trans::N, Trans::N, scalar_t(1.), ThetaVhatC_or_VhatCPhiC_, cR,
           scalar_t(0.), tmp, task_depth);
      gemm(Trans::N, Trans::N, scalar_t(-1.), Theta_, tmp,
           scalar_t(1.), cSr, task_depth);
      tmp = DenseM_t(_Theta.cols(), cR.cols());
      gemm(Trans::C, Trans::N, scalar_t(1.), Theta_, cR,
           scalar_t(0.), tmp, task_depth);
      gemm(Trans::C, Trans::N, scalar_t(-1.), ThetaVhatC_or_VhatCPhiC_, tmp,
           scalar_t(1.), cSc, task_depth);
    } else {
      DenseM_t tmp(Phi.cols(), cR.cols());
      gemm(Trans::C, Trans::N, scalar_t(1.), Phi_, cR,
           scalar_t(0.), tmp, task_depth);
      gemm(Trans::N, Trans::N, scalar_t(-1.), ThetaVhatC_or_VhatCPhiC_, tmp,
           scalar_t(1.), cSr, task_depth);
      tmp = DenseM_t(_ThetaVhatC_or_VhatCPhiC.cols(), cR.cols());
      gemm(Trans::C, Trans::N, scalar_t(1.), ThetaVhatC_or_VhatCPhiC_, cR,
           scalar_t(0.), tmp, task_depth);
      gemm(Trans::N, Trans::N, scalar_t(-1.), Phi_, tmp,
           scalar_t(1.), cSc, task_depth);
    }
#else
    DenseM_t cSr(cR.rows(), cR.cols());
    DenseM_t cSc(cR.rows(), cR.cols());
    TIMER_TIME(TaskType::HSS_SCHUR_PRODUCT, 2, t_sprod);
    H_.Schur_product_direct
      (Theta_, DUB01_, Phi_, ThetaVhatC_or_VhatCPhiC_, cR, cSr, cSc);
    TIMER_STOP(t_sprod);
#endif
    Sr.scatter_rows_add(I, cSr, task_depth);
    Sc.scatter_rows_add(I, cSc, task_depth);
    STRUMPACK_CB_SAMPLE_FLOPS(cSr.rows()*cSr.cols()*2);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const Opts_t& opts,
   int etree_level, int task_depth) {
    if (task_depth == 0)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      multifrontal_factorization_node(A, opts, etree_level, task_depth);
    else multifrontal_factorization_node(A, opts, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::random_sampling
  (const SpMat_t& A, const Opts_t& opts, DenseM_t& Rr,
   DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc, int etree_level,
   int task_depth) {
    Sr.zero();
    Sc.zero();
    const auto dsep = dim_sep();
    const auto dupd = dim_upd();
    if (opts.indirect_sampling()) {
      using real_t = typename RealType<scalar_t>::value_type;
      auto rgen = random::make_random_generator<real_t>
        (opts.HSS_options().random_engine(),
         opts.HSS_options().random_distribution());
      auto dd = opts.HSS_options().dd();
      auto d0 = opts.HSS_options().d0();
      integer_t d = Rr.cols(), m = Rr.rows();
      if (d0 % dd == 0) {
        for (integer_t c=0; c<d; c+=dd) {
          integer_t r = 0, cs = c + sampled_columns_;
          for (; r<dsep; r++) {
            rgen->seed(std::uint32_t(r+sep_begin_), std::uint32_t(cs));
            for (integer_t cc=c; cc<c+dd; cc++)
              Rr(r,cc) = Rc(r,cc) = rgen->get();
          }
          for (; r<m; r++) {
            rgen->seed(std::uint32_t(this->upd_[r-dsep]),
                       std::uint32_t(cs));
            for (integer_t cc=c; cc<c+dd; cc++)
              Rr(r,cc) = Rc(r,cc) = rgen->get();
          }
        }
      } else {
        for (integer_t c=0; c<d; c++) {
          integer_t r = 0, cs = c + sampled_columns_;
          for (; r<dsep; r++)
            Rr(r,c) = Rc(r,c) = rgen->get(r+sep_begin_, cs);
          for (; r<m; r++)
            Rr(r,c) = Rc(r,c) = rgen->get(this->upd_[r-dsep], cs);
        }
      }
      STRUMPACK_FLOPS(rgen->flops_per_prng()*d*m);
      STRUMPACK_RANDOM_FLOPS(rgen->flops_per_prng()*d*m);
    }

    TIMER_TIME(TaskType::FRONT_MULTIPLY_2D, 1, t_fmult);
    A.front_multiply
      (sep_begin_, sep_end_, this->upd_, Rr, Sr, Sc, task_depth);
    TIMER_STOP(t_fmult);
    TIMER_TIME(TaskType::UUTXR, 1, t_UUtxR);
    if (lchild_)
      lchild_->sample_CB(opts, Rr, Sr, Sc, this, task_depth);
    if (rchild_)
      rchild_->sample_CB(opts, Rr, Sr, Sc, this, task_depth);
    TIMER_STOP(t_UUtxR);

    if (opts.indirect_sampling() && etree_level != 0) {
      auto dold = R1.cols();
      auto dd = Rr.cols();
      auto dnew = dold + dd;
      R1.resize(dsep, dnew);
      Sr2.resize(dupd, dnew);
      Sc2.resize(dupd, dnew);
      copy(dsep, dd, Rr, 0, 0, R1, 0, dold);
      copy(dupd, dd, Sr, dsep, 0, Sr2, 0, dold);
      copy(dupd, dd, Sc, dsep, 0, Sc2, 0, dold);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::element_extraction
  (const SpMat_t& A, const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J, DenseM_t& B, int task_depth) {
    std::vector<std::size_t> gI, gJ;
    gI.reserve(I.size());
    gJ.reserve(J.size());
    const auto dsep = dim_sep();
    for (auto i : I)
      gI.push_back
        ((integer_t(i) < dsep) ? i+sep_begin_ : this->upd_[i-dsep]);
    for (auto j : J)
      gJ.push_back
        ((integer_t(j) < dsep) ? j+sep_begin_ : this->upd_[j-dsep]);
    {
      TIMER_TIME(TaskType::EXTRACT_SEP_2D, 2, t_ex_sep);
      A.extract_separator(sep_end_, gI, gJ, B, task_depth);
    }
    TIMER_TIME(TaskType::GET_SUBMATRIX_2D, 2, t_getsub);
    if (lchild_)
      lchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
    if (rchild_)
      rchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::multifrontal_factorization_node
  (const SpMat_t& A, const Opts_t& opts,
   int etree_level, int task_depth) {
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
    TaskTimer t("FrontalMatrixHSS_factor");
    if (opts.print_compressed_front_stats()) t.start();
    H_.set_openmp_task_depth(task_depth);
    auto mult = [&](DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc) {
      TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
      random_sampling(A, opts, Rr, Rc, Sr, Sc, etree_level, task_depth);
      sampled_columns_ += Rr.cols();
    };
    auto elem = [&](const std::vector<std::size_t>& I,
                    const std::vector<std::size_t>& J, DenseM_t& B) {
      TIMER_TIME(TaskType::EXTRACT_2D, 1, t_ex);
      element_extraction(A, I, J, B, task_depth);
    };
    auto HSSopts = opts.HSS_options();
    int child_samples = 0;
    if (lchild_)
      child_samples = lchild_->random_samples();
    if (rchild_)
      child_samples = std::max(child_samples, rchild_->random_samples());
    HSSopts.set_d0(std::max(child_samples - HSSopts.dd(), HSSopts.d0()));
    if (opts.indirect_sampling())
      HSSopts.set_user_defined_random(true);
    H_.compress(mult, elem, HSSopts);
    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
    if (dim_sep()) {
      if (etree_level > 0) {
        TIMER_TIME(TaskType::HSS_PARTIALLY_FACTOR, 0, t_pfact);
        H_.partial_factor();
        TIMER_STOP(t_pfact);
        TIMER_TIME(TaskType::HSS_COMPUTE_SCHUR, 0, t_comp_schur);
        H_.Schur_update(Theta_, DUB01_, Phi_);
        const DenseM_t& Vhat = H_.child(0)->ULV().Vhat();
        if (Theta_.cols() < Phi_.cols()) {
          ThetaVhatC_or_VhatCPhiC_ = DenseM_t(Vhat.cols(), Phi_.rows());
          gemm(Trans::C, Trans::C, scalar_t(1.), Vhat, Phi_,
               scalar_t(0.), ThetaVhatC_or_VhatCPhiC_, task_depth);
          STRUMPACK_SCHUR_FLOPS
            (gemm_flops(Trans::C, Trans::C, scalar_t(1.), Vhat, Phi_, scalar_t(0.)));
        } else {
          ThetaVhatC_or_VhatCPhiC_ = DenseM_t(Theta_.rows(), Vhat.rows());
          gemm(Trans::N, Trans::C, scalar_t(1.), Theta_, Vhat,
               scalar_t(0.), ThetaVhatC_or_VhatCPhiC_, task_depth);
          STRUMPACK_SCHUR_FLOPS
            (gemm_flops(Trans::N, Trans::C, scalar_t(1.), Theta_, Vhat, scalar_t(0.)));
        }
      } else {
        TIMER_TIME(TaskType::HSS_FACTOR, 0, t_fact);
        H_.factor();
        TIMER_STOP(t_fact);
      }
    }
    if (opts.print_compressed_front_stats()) {
      auto time = t.elapsed();
      auto rank = H_.rank();
      std::size_t nnzH = H_.nonzeros();
      std::size_t nnzULV = H_.factor_nonzeros();
      std::size_t nnzSchur = Theta_.nonzeros()
        + Phi_.nonzeros() + ThetaVhatC_or_VhatCPhiC_.nonzeros();
      std::cout << "#   - HSSMPI front: Nsep= " << dim_sep()
                << " , Nupd= " << dim_upd()
                << " , nnz(H)= " << nnzH
                << " , nnz(ULV)= " << nnzULV
                << " , nnz(Schur)= " << nnzSchur
                << " , maxrank= " << rank
                << " , " << (float(nnzH + nnzULV + nnzSchur)
                             / (float(dim_blk())*dim_blk()) * 100.)
                << " %compression, time= " << time
                << " sec" << std::endl;
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    if (task_depth == 0)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      fwd_solve_node(b, work, etree_level, task_depth);
    else fwd_solve_node(b, work, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::fwd_solve_node
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t bupd(dim_upd(), b.cols(), work[0], 0, 0);
    bupd.zero();
    this->fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
    if (etree_level) {
      if (Theta_.cols() && Phi_.cols()) {
        DenseMW_t bloc(dim_sep(), b.cols(), b, sep_begin_, 0);
        // TODO get rid of this!!!
        ULVwork_ = std::unique_ptr<HSS::WorkSolve<scalar_t>>
          (new HSS::WorkSolve<scalar_t>());
        H_.child(0)->forward_solve(*ULVwork_, bloc, true);
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), Theta_,
               ULVwork_->reduced_rhs, scalar_t(1.), bupd, task_depth);
        ULVwork_->reduced_rhs.clear();
      }
    } else {
      DenseMW_t bloc(dim_sep(), b.cols(), b, sep_begin_, 0);
      ULVwork_ = std::unique_ptr<HSS::WorkSolve<scalar_t>>
        (new HSS::WorkSolve<scalar_t>());
      H_.forward_solve(*ULVwork_, bloc, false);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    if (task_depth == 0)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      bwd_solve_node(y, work, etree_level, task_depth);
    else bwd_solve_node(y, work, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::bwd_solve_node
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t yupd(dim_upd(), y.cols(), work[0], 0, 0);
    if (etree_level) {
      if (Phi_.cols() && Theta_.cols()) {
        if (dim_upd()) {
          gemm(Trans::C, Trans::N, scalar_t(-1.), Phi_, yupd,
               scalar_t(1.), ULVwork_->x, task_depth);
        }
        DenseMW_t yloc(dim_sep(), y.cols(), y, sep_begin_, 0);
        H_.child(0)->backward_solve(*ULVwork_, yloc);
        ULVwork_.reset();
      }
    } else {
      DenseMW_t yloc(dim_sep(), y.cols(), y, sep_begin_, 0);
      H_.backward_solve(*ULVwork_, yloc);
    }
    this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> integer_t
  FrontalMatrixHSS<scalar_t,integer_t>::front_rank(int task_depth) const {
    return H_.rank();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::print_rank_statistics
  (std::ostream &out) const {
    if (lchild_) lchild_->print_rank_statistics(out);
    if (rchild_) rchild_->print_rank_statistics(out);
    out << "# HSSMatrix " << H_.rows() << "x" << H_.cols() << std::endl;
    H_.print_info(out);
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixHSS<scalar_t,integer_t>::node_factor_nonzeros() const {
    return H_.nonzeros() + H_.factor_nonzeros() + Theta_.nonzeros()
      + Phi_.nonzeros() + ThetaVhatC_or_VhatCPhiC_.nonzeros();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::draw_node
  (std::ostream& of, bool is_root) const {
    if (is_root) H_.draw(of, sep_begin_, sep_begin_);
    else H_.child(0)->draw(of, sep_begin_, sep_begin_);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::partition
  (const Opts_t& opts, const SpMat_t& A,
   integer_t* sorder, bool is_root, int task_depth) {
    auto g = A.extract_graph
      (opts.separator_ordering_level(), sep_begin_, sep_end_);
    auto sep_tree = g.recursive_bisection
      (opts.compression_leaf_size(), 0,
       sorder+sep_begin_, nullptr, 0, 0, dim_sep());
    for (integer_t i=sep_begin_; i<sep_end_; i++)
      sorder[i] += sep_begin_;
    if (is_root)
      H_ = HSS::HSSMatrix<scalar_t>(sep_tree, opts.HSS_options());
    else {
      structured::ClusterTree tree(this->dim_blk());
      tree.c.reserve(2);
      tree.c.push_back(sep_tree);
      tree.c.emplace_back(dim_upd());
      tree.c.back().refine(opts.HSS_options().leaf_size());
      H_ = HSS::HSSMatrix<scalar_t>(tree, opts.HSS_options());
    }
  }

  // explicit template instantiations
  template class FrontalMatrixHSS<float,int>;
  template class FrontalMatrixHSS<double,int>;
  template class FrontalMatrixHSS<std::complex<float>,int>;
  template class FrontalMatrixHSS<std::complex<double>,int>;

  template class FrontalMatrixHSS<float,long int>;
  template class FrontalMatrixHSS<double,long int>;
  template class FrontalMatrixHSS<std::complex<float>,long int>;
  template class FrontalMatrixHSS<std::complex<double>,long int>;

  template class FrontalMatrixHSS<float,long long int>;
  template class FrontalMatrixHSS<double,long long int>;
  template class FrontalMatrixHSS<std::complex<float>,long long int>;
  template class FrontalMatrixHSS<std::complex<double>,long long int>;

} // end namespace strumpack
