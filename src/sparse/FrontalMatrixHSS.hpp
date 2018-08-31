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
#ifndef FRONTAL_MATRIX_HSS_HPP
#define FRONTAL_MATRIX_HSS_HPP

#include <iostream>
#include <algorithm>
#include <memory>

#include "misc/TaskTimer.hpp"
#include "FrontalMatrix.hpp"
#include "CompressedSparseMatrix.hpp"
#include "MatrixReordering.hpp"
#include "HSS/HSSMatrix.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixHSS
    : public FrontalMatrix<scalar_t,integer_t> {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;

  public:
    FrontalMatrixHSS
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd);

    void extend_add_to_dense
    (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
     const FrontalMatrix<scalar_t,integer_t>* p, int task_depth) override;

    void sample_CB
    (const SPOptions<scalar_t>& opts, const DenseM_t& R, DenseM_t& Sr,
     DenseM_t& Sc, F_t* pa, int task_depth) override;
    void sample_CB_direct
    (const DenseM_t& cR, DenseM_t& Sr, DenseM_t& Sc,
     const std::vector<std::size_t>& I, int task_depth);

    void release_work_memory() override;
    void random_sampling
    (const SpMat_t& A, const SPOptions<scalar_t>& opts, DenseM_t& Rr,
     DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc, int etree_level,
     int task_depth); // TODO const?
    void element_extraction
    (const SpMat_t& A, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J, DenseM_t& B, int task_depth);
    void extract_CB_sub_matrix
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DenseM_t& B, int task_depth) const;

    void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level=0, int task_depth=0) override;

    void forward_multifrontal_solve
    (DenseM_t& b, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;
    void backward_multifrontal_solve
    (DenseM_t& y, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;

    integer_t maximum_rank(int task_depth=0) const override;
    void print_rank_statistics(std::ostream &out) const override;
    bool isHSS() const override { return true; };
    std::string type() const override { return "FrontalMatrixHSS"; }

    int random_samples() const { return R1.cols(); };
    void bisection_partitioning
    (const SPOptions<scalar_t>& opts, integer_t* sorder,
     bool isroot=true, int task_depth=0) override;

    void set_HSS_partitioning
    (const SPOptions<scalar_t>& opts,
     const HSS::HSSPartitionTree& sep_tree, bool is_root) override;


    // TODO make private?
    HSS::HSSMatrix<scalar_t> _H;
    HSS::HSSFactors<scalar_t> _ULV;

    // TODO do not store this here: makes solve not thread safe!!
    mutable std::unique_ptr<HSS::WorkSolve<scalar_t>> _ULVwork;

    /** Schur complement update:
     *    S = F22 - _Theta * Vhat^C * _Phi^C
     **/
    DenseM_t _Theta;
    DenseM_t _Phi;
    DenseM_t _ThetaVhatC_or_VhatCPhiC;
    DenseM_t _DUB01;

    /** these are saved during/after randomized compression and are
        then later used to sample the Schur complement when
        compressing the parent front */
    DenseM_t R1;        /* top of the random matrix used to construct
                           HSS matrix of this front */
    DenseM_t Sr2, Sc2;  /* bottom of the sample matrix used to
                           construct HSS matrix of this front */
    std::uint32_t _sampled_columns = 0;

  private:
    FrontalMatrixHSS(const FrontalMatrixHSS&) = delete;
    FrontalMatrixHSS& operator=(FrontalMatrixHSS const&) = delete;

    void draw_node(std::ostream& of, bool is_root) const override;

    void multifrontal_factorization_node
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level, int task_depth);

    void fwd_solve_node
    (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const;
    void bwd_solve_node
    (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const;

    long long node_factor_nonzeros() const override;

    void split_separator
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     HSS::HSSPartitionTree& hss_tree, integer_t& nr_parts, integer_t part,
     integer_t count, integer_t* sorder);
    void extract_separator
    (const SpMat_t& A, const SPOptions<scalar_t>& opts, integer_t part,
     std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy,
     integer_t* sorder);

    using FrontalMatrix<scalar_t,integer_t>::lchild_;
    using FrontalMatrix<scalar_t,integer_t>::rchild_;
    using FrontalMatrix<scalar_t,integer_t>::dim_sep;
    using FrontalMatrix<scalar_t,integer_t>::dim_upd;
    using FrontalMatrix<scalar_t,integer_t>::sep_begin_;
    using FrontalMatrix<scalar_t,integer_t>::sep_end_;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixHSS<scalar_t,integer_t>::FrontalMatrixHSS
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : FrontalMatrix<scalar_t,integer_t>
    (nullptr, nullptr, sep, sep_begin, sep_end, upd) {
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::release_work_memory() {
    _ThetaVhatC_or_VhatCPhiC.clear();
    _H.delete_trailing_block();
    R1.clear();
    Sr2.clear();
    Sc2.clear();
    _DUB01.clear();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const FrontalMatrix<scalar_t,integer_t>* p, int task_depth) {
    const std::size_t pdsep = p->dim_sep();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);

    auto F22 = _H.child(1)->dense();
    if (_Theta.cols() < _Phi.cols())
      // S = F22 - _Theta * _ThetaVhatC_or_VhatCPhiC
      gemm(Trans::N, Trans::N, scalar_t(-1.), _Theta,
           _ThetaVhatC_or_VhatCPhiC,
           scalar_t(1.), F22, task_depth);
    else
      // S = F22 - _ThetaVhatC_or_VhatCPhiC * _Phi'
      gemm(Trans::N, Trans::C, scalar_t(-1.),
           _ThetaVhatC_or_VhatCPhiC, _Phi,
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
      auto M = _H.child(1)->extract(lI, lJ);
      for (std::size_t j=0; j<lJ.size(); j++)
        for (std::size_t i=0; i<lI.size(); i++)
          B(oI[i], oJ[j]) += M(i, j);

      if (_Theta.cols() < _Phi.cols()) {
        // S = F22 - _Theta * _ThetaVhatC_or_VhatCPhiC
        auto r = _Theta.cols();
        DenseM_t r_theta(lI.size(), r);
        DenseM_t c_vhatphiC(r, lJ.size());
        for (std::size_t i=0; i<lI.size(); i++)
          copy(1, r, _Theta, lI[i], 0, r_theta, i, 0);
        for (std::size_t j=0; j<lJ.size(); j++)
          copy(r, 1, _ThetaVhatC_or_VhatCPhiC, 0, lJ[j], c_vhatphiC, 0, j);
        gemm(Trans::N, Trans::N, scalar_t(-1.), r_theta, c_vhatphiC,
             scalar_t(0.), M, task_depth);
        for (std::size_t j=0; j<lJ.size(); j++)
          for (std::size_t i=0; i<lI.size(); i++)
            B(oI[i], oJ[j]) += M(i, j);
        STRUMPACK_EXTRACTION_FLOPS
          (gemm_flops(Trans::N, Trans::N, scalar_t(-1.), r_theta, c_vhatphiC,
                      scalar_t(0.)) + lJ.size()*lI.size());
      } else {
        // S = F22 - _ThetaVhatC_or_VhatCPhiC * _Phi'
        auto r = _Phi.cols();
        DenseM_t r_thetavhat(lI.size(), r);
        DenseM_t r_phi(lJ.size(), r);
        for (std::size_t i=0; i<lI.size(); i++)
          copy(1, r, _ThetaVhatC_or_VhatCPhiC, lI[i], 0, r_thetavhat, i, 0);
        for (std::size_t j=0; j<lJ.size(); j++)
          copy(1, r, _Phi, lJ[j], 0, r_phi, j, 0);
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
  (const SPOptions<scalar_t>& opts, const DenseM_t& R,
   DenseM_t& Sr, DenseM_t& Sc, F_t* pa, int task_depth) {
    if (!dim_upd()) return;
    auto I = this->upd_to_parent(pa);
    auto cR = R.extract_rows(I);
    auto dchild = R1.cols();
    auto dall = R.cols();
    if (dchild > 0 && opts.indirect_sampling()) {
      DenseM_t cSr, cSc;
      DenseMW_t cRd0(cR.rows(), dchild, cR, 0, 0);
      _H.Schur_product_indirect(_ULV, _DUB01, R1, cRd0, Sr2, Sc2, cSr, cSc);
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
    auto cSr = _H.child(1)->apply(cR);
    auto cSc = _H.child(1)->applyC(cR);
    if (_Theta.cols() < _Phi.cols()) {
      // TODO 2 tasks?
      DenseM_t tmp(_ThetaVhatC_or_VhatCPhiC.rows(), cR.cols());
      gemm(Trans::N, Trans::N, scalar_t(1.), _ThetaVhatC_or_VhatCPhiC, cR,
           scalar_t(0.), tmp, task_depth);
      gemm(Trans::N, Trans::N, scalar_t(-1.), _Theta, tmp,
           scalar_t(1.), cSr, task_depth);
      tmp = DenseM_t(_Theta.cols(), cR.cols());
      gemm(Trans::C, Trans::N, scalar_t(1.), _Theta, cR,
           scalar_t(0.), tmp, task_depth);
      gemm(Trans::C, Trans::N, scalar_t(-1.), _ThetaVhatC_or_VhatCPhiC, tmp,
           scalar_t(1.), cSc, task_depth);
    } else {
      DenseM_t tmp(_Phi.cols(), cR.cols());
      gemm(Trans::C, Trans::N, scalar_t(1.), _Phi, cR,
           scalar_t(0.), tmp, task_depth);
      gemm(Trans::N, Trans::N, scalar_t(-1.), _ThetaVhatC_or_VhatCPhiC, tmp,
           scalar_t(1.), cSr, task_depth);
      tmp = DenseM_t(_ThetaVhatC_or_VhatCPhiC.cols(), cR.cols());
      gemm(Trans::C, Trans::N, scalar_t(1.), _ThetaVhatC_or_VhatCPhiC, cR,
           scalar_t(0.), tmp, task_depth);
      gemm(Trans::N, Trans::N, scalar_t(-1.), _Phi, tmp,
           scalar_t(1.), cSc, task_depth);
    }
#else
    DenseM_t cSr(cR.rows(), cR.cols());
    DenseM_t cSc(cR.rows(), cR.cols());
    _H.Schur_product_direct
      (_ULV, _Theta, _DUB01, _Phi,
       _ThetaVhatC_or_VhatCPhiC, cR, cSr, cSc);
#endif
    Sr.scatter_rows_add(I, cSr, task_depth);
    Sc.scatter_rows_add(I, cSc, task_depth);
    STRUMPACK_CB_SAMPLE_FLOPS(cSr.rows()*cSr.cols()*2);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (task_depth == 0)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      multifrontal_factorization_node(A, opts, etree_level, task_depth);
    else multifrontal_factorization_node(A, opts, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::random_sampling
  (const SpMat_t& A, const SPOptions<scalar_t>& opts, DenseM_t& Rr,
   DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc, int etree_level,
   int task_depth) {
    Sr.zero();
    Sc.zero();
    const auto dsep = dim_sep();
    const auto dupd = dim_upd();
    if (opts.indirect_sampling()) {
      auto rgen = random::make_random_generator<real_t>
        (opts.HSS_options().random_engine(),
         opts.HSS_options().random_distribution());
      auto dd = opts.HSS_options().dd();
      auto d0 = opts.HSS_options().d0();
      integer_t d = Rr.cols();
      integer_t m = Rr.rows();
      if (d0 % dd == 0) {
        for (integer_t c=0; c<d; c+=dd) {
          integer_t r = 0, cs = c + _sampled_columns;
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
          integer_t r = 0, cs = c + _sampled_columns;
          for (; r<dsep; r++)
            Rr(r,c) = Rc(r,c) = rgen->get(r+sep_begin_, cs);
          for (; r<m; r++)
            Rr(r,c) = Rc(r,c) = rgen->get(this->upd_[r-dsep], cs);
        }
      }
      STRUMPACK_FLOPS(rgen->flops_per_prng()*d*m);
      STRUMPACK_RANDOM_FLOPS(rgen->flops_per_prng()*d*m);
    }

    A.front_multiply
      (sep_begin_, sep_end_, this->upd_, Rr, Sr, Sc, task_depth);
    if (lchild_)
      lchild_->sample_CB(opts, Rr, Sr, Sc, this, task_depth);
    if (rchild_)
      rchild_->sample_CB(opts, Rr, Sr, Sc, this, task_depth);

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
    A.extract_separator(sep_end_, gI, gJ, B, task_depth);
    if (lchild_)
      lchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
    if (rchild_)
      rchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::multifrontal_factorization_node
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
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
    _H.set_openmp_task_depth(task_depth);
    auto mult = [&](DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc) {
      random_sampling(A, opts, Rr, Rc, Sr, Sc, etree_level, task_depth);
      _sampled_columns += Rr.cols();
    };
    auto elem = [&](const std::vector<std::size_t>& I,
                    const std::vector<std::size_t>& J, DenseM_t& B) {
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
    _H.compress(mult, elem, HSSopts);
    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
    if (dim_sep()) {
      if (etree_level > 0) {
        _ULV = _H.partial_factor();
        _H.Schur_update(_ULV, _Theta, _DUB01, _Phi);
        DenseM_t& Vhat = _ULV.Vhat();
        if (_Theta.cols() < _Phi.cols()) {
          _ThetaVhatC_or_VhatCPhiC = DenseM_t(Vhat.cols(), _Phi.rows());
          gemm(Trans::C, Trans::C, scalar_t(1.), Vhat, _Phi,
               scalar_t(0.), _ThetaVhatC_or_VhatCPhiC, task_depth);
          STRUMPACK_SCHUR_FLOPS
            (gemm_flops(Trans::C, Trans::C, scalar_t(1.), Vhat, _Phi, scalar_t(0.)));
        } else {
          _ThetaVhatC_or_VhatCPhiC = DenseM_t(_Theta.rows(), Vhat.rows());
          gemm(Trans::N, Trans::C, scalar_t(1.), _Theta, Vhat,
               scalar_t(0.), _ThetaVhatC_or_VhatCPhiC, task_depth);
          STRUMPACK_SCHUR_FLOPS
            (gemm_flops(Trans::N, Trans::C, scalar_t(1.), _Theta, Vhat, scalar_t(0.)));
        }
      } else
        _ULV = _H.factor();
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
      if (_Theta.cols() && _Phi.cols()) {
        DenseMW_t bloc(dim_sep(), b.cols(), b, sep_begin_, 0);
        // TODO get rid of this!!!
        _ULVwork = std::unique_ptr<HSS::WorkSolve<scalar_t>>
          (new HSS::WorkSolve<scalar_t>());
        _H.child(0)->forward_solve(_ULV, *_ULVwork, bloc, true);
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), _Theta,
               _ULVwork->reduced_rhs, scalar_t(1.), bupd, task_depth);
        _ULVwork->reduced_rhs.clear();
      }
    } else {
      DenseMW_t bloc(dim_sep(), b.cols(), b, sep_begin_, 0);
      _ULVwork = std::unique_ptr<HSS::WorkSolve<scalar_t>>
        (new HSS::WorkSolve<scalar_t>());
      _H.forward_solve(_ULV, *_ULVwork, bloc, false);
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
      if (_Phi.cols() && _Theta.cols()) {
        if (dim_upd()) {
          gemm(Trans::C, Trans::N, scalar_t(-1.), _Phi, yupd,
               scalar_t(1.), _ULVwork->x, task_depth);
        }
        DenseMW_t yloc(dim_sep(), y.cols(), y, sep_begin_, 0);
        _H.child(0)->backward_solve(_ULV, *_ULVwork, yloc);
        _ULVwork.reset();
      }
    } else {
      DenseMW_t yloc(dim_sep(), y.cols(), y, sep_begin_, 0);
      _H.backward_solve(_ULV, *_ULVwork, yloc);
    }
    this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> integer_t
  FrontalMatrixHSS<scalar_t,integer_t>::maximum_rank(int task_depth) const {
    integer_t r = _H.rank(), rl = 0, rr = 0;
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
  FrontalMatrixHSS<scalar_t,integer_t>::print_rank_statistics
  (std::ostream &out) const {
    if (lchild_) lchild_->print_rank_statistics(out);
    if (rchild_) rchild_->print_rank_statistics(out);
    out << "# HSSMatrix " << _H.rows() << "x" << _H.cols() << std::endl;
    _H.print_info(out);
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixHSS<scalar_t,integer_t>::node_factor_nonzeros() const {
    return _H.nonzeros() + _ULV.nonzeros() + _Theta.nonzeros()
      + _Phi.nonzeros() + _ThetaVhatC_or_VhatCPhiC.nonzeros();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::draw_node
  (std::ostream& of, bool is_root) const {
    if (is_root) _H.draw(of, sep_begin_, sep_begin_);
    else _H.child(0)->draw(of, sep_begin_, sep_begin_);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::set_HSS_partitioning
  (const SPOptions<scalar_t>& opts, const HSS::HSSPartitionTree& sep_tree,
   bool is_root) {
    assert(sep_tree.size == dim_sep());
    if (is_root)
      _H = HSS::HSSMatrix<scalar_t>(sep_tree, opts.HSS_options());
    else {
      HSS::HSSPartitionTree hss_tree(this->dim_blk());
      hss_tree.c.reserve(2);
      hss_tree.c.push_back(sep_tree);
      hss_tree.c.emplace_back(dim_upd());
      hss_tree.c.back().refine(opts.HSS_options().leaf_size());
      _H = HSS::HSSMatrix<scalar_t>(hss_tree, opts.HSS_options());
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::bisection_partitioning
  (const SPOptions<scalar_t>& opts, integer_t* sorder,
   bool isroot, int task_depth) {
    if (lchild_)
#pragma omp task default(shared)                        \
  if(task_depth < params::task_recursion_cutoff_level)
      lchild_->bisection_partitioning(opts, sorder, false, task_depth+1);
    if (rchild_)
#pragma omp task default(shared)                        \
  if(task_depth < params::task_recursion_cutoff_level)
      rchild_->bisection_partitioning(opts, sorder, false, task_depth+1);
#pragma omp taskwait

    std::cout << "TODO FrontalMatrixHSS::bisection_partitioning" << std::endl;

    for (integer_t i=sep_begin_; i<sep_end_; i++)
      sorder[i] = -i;
    HSS::HSSPartitionTree sep_tree(dim_sep());
    sep_tree.refine(opts.HSS_options().leaf_size());

    // TODO this still needs to work when the sparse matrix is a
    // ProportionallyMappedSparseMatrix!!!
    // if (this->dim_sep >= 2 * opts.HSS_options().leaf_size()) {
    //   integer_t nrparts = 0;
    //   split_separator(opts, sep_tree, nrparts, 0, 1, sorder);
    //   auto count = sep_begin_;
    //   for (integer_t part=0; part<nrparts; part++)
    //          for (integer_t i=sep_begin_; i<sep_end_; i++)
    //            if (sorder[i] == part) sorder[i] = -count++;
    // } else for (integer_t i=sep_begin_; i<sep_end_; i++) sorder[i] = -i;

    if (isroot)
      _H = HSS::HSSMatrix<scalar_t>(sep_tree, opts.HSS_options());
    else {
      HSS::HSSPartitionTree hss_tree(this->dim_blk());
      hss_tree.c.reserve(2);
      hss_tree.c.push_back(sep_tree);
      hss_tree.c.emplace_back(dim_upd());
      hss_tree.c.back().refine(opts.HSS_options().leaf_size());
      _H = HSS::HSSMatrix<scalar_t>(hss_tree, opts.HSS_options());
    }
  }


  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::split_separator
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   HSS::HSSPartitionTree& hss_tree, integer_t& nr_parts, integer_t part,
   integer_t count, integer_t* sorder) {
    std::vector<idx_t> xadj, adjncy;
    extract_separator(A, opts, part, xadj, adjncy, sorder);
    idx_t ncon = 1, edge_cut = 0, two = 2, nvtxs=xadj.size()-1;
    auto partitioning = new idx_t[nvtxs];
    int info = METIS_PartGraphRecursive
      (&nvtxs, &ncon, xadj.data(), adjncy.data(), NULL, NULL, NULL,
       &two, NULL, NULL, NULL, &edge_cut, partitioning);
    if (info != METIS_OK) {
      std::cerr << "METIS_PartGraphRecursive for separator reordering"
                << " returned: " << info << std::endl;
      exit(1);
    }
    hss_tree.c.resize(2);
    for (integer_t i=sep_begin_, j=0; i<sep_end_; i++)
      if (sorder[i] == part) {
        auto p = partitioning[j++];
        sorder[i] = -count - p;
        hss_tree.c[p].size++;
      }
    delete[] partitioning;
    for (integer_t p=0; p<2; p++)
      if (hss_tree.c[p].size >= 2 * opts.HSS_options().leaf_size())
        split_separator
          (A, opts, hss_tree.c[p], nr_parts, -count-p, count+2, sorder);
      else std::replace
             (sorder+sep_begin_, sorder+sep_end_,
              -count-p, nr_parts++);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSS<scalar_t,integer_t>::extract_separator
  (const SpMat_t& A, const SPOptions<scalar_t>& opts, integer_t part,
   std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy, integer_t* sorder) {
    assert(opts.separator_ordering_level() == 0 ||
           opts.separator_ordering_level() == 1);
    auto dsep = dim_sep();
    auto mark = new bool[dsep];
    auto ind_to_part = new integer_t[dsep];
    integer_t nvtxs = 0;
    for (integer_t r=0; r<dsep; r++)
      ind_to_part[r] = (sorder[r+sep_begin_] == part) ? nvtxs++ : -1;
    xadj.reserve(nvtxs+1);
    adjncy.reserve(5*nvtxs);
    for (integer_t i=sep_begin_, e=0; i<sep_end_; i++) {
      if (sorder[i] == part) {
        xadj.push_back(e);
        std::fill(mark, mark+dsep, false);
        for (integer_t j=A.ptr(i);
             j<A.ptr(i+1); j++) {
          auto c = A.ind(j);
          if (c == i) continue;
          auto lc = c - sep_begin_;
          if (lc >= 0 && lc < dsep && sorder[c]==part && !mark[lc]) {
            mark[lc] = true;
            adjncy.push_back(ind_to_part[lc]);
            e++;
          } else {
            if (opts.separator_ordering_level() > 0) {
              for (integer_t k=A.ptr(c);
                   k<A.ptr(c+1); k++) {
                auto cc = A.ind(k);
                auto lcc = cc - sep_begin_;
                if (cc!=i && lcc >= 0 && lcc < dsep &&
                    sorder[cc]==part && !mark[lcc]) {
                  mark[lcc] = true;
                  adjncy.push_back(ind_to_part[lcc]);
                  e++;
                }
              }
            }
          }
        }
      }
    }
    xadj.push_back(adjncy.size());
    delete[] mark;
    delete[] ind_to_part;
  }

} // end namespace strumpack

#endif
