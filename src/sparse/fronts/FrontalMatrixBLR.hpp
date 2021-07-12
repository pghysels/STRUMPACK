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
#ifndef FRONTAL_MATRIX_BLR_HPP
#define FRONTAL_MATRIX_BLR_HPP

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>

#include "misc/TaskTimer.hpp"
#include "dense/BLASLAPACKWrapper.hpp"
#include "sparse/CompressedSparseMatrix.hpp"
#include "BLR/BLRMatrix.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#include "BLR/BLRExtendAdd.hpp"
#endif

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixBLRMPI;

  template<typename scalar_t,typename integer_t> class FrontalMatrixBLR
    : public FrontalMatrix<scalar_t,integer_t> {
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using Opts_t = SPOptions<scalar_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using BLRM_t = BLR::BLRMatrix<scalar_t>;
#if defined(STRUMPACK_USE_MPI)
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using FBLRMPI_t = FrontalMatrixBLRMPI<scalar_t,integer_t>;
#endif

  public:
    FrontalMatrixBLR
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd);
    ~FrontalMatrixBLR() {}

    void release_work_memory() override;

    void build_front(const SpMat_t& A);

    void build_front_cols(const SpMat_t& A, std::size_t i, 
      bool part, std::size_t CP, 
      const std::vector<Triplet<scalar_t>>& e11, 
      const std::vector<Triplet<scalar_t>>& e12, 
      const std::vector<Triplet<scalar_t>>& e21, int task_depth, const Opts_t& opts);

    void extend_add_to_dense
    (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
     const F_t* p, int task_depth) override;
    void extend_add_to_blr
    (BLRM_t& paF11, BLRM_t& paF12, BLRM_t& paF21, BLRM_t& paF22,
     const F_t* p, int task_depth, const Opts_t& opts) override;
    void extend_add_to_blr_col
    (BLRM_t& paF11, BLRM_t& paF12, BLRM_t& paF21, BLRM_t& paF22,
     const F_t* p, integer_t begin_col, integer_t end_col, int task_depth, const Opts_t& opts) override;
    void sample_CB
    (const Opts_t& opts, const DenseM_t& R, DenseM_t& Sr,
     DenseM_t& Sc, F_t* pa, int task_depth) override;
    void multifrontal_factorization
    (const SpMat_t& A, const Opts_t& opts,
     int etree_level=0, int task_depth=0) override;
    void factor_node
    (const SpMat_t& A, const Opts_t& opts,
     int etree_level=0, int task_depth=0);

    void forward_multifrontal_solve
    (DenseM_t& b, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;
    void backward_multifrontal_solve
    (DenseM_t& y, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;

    void extract_CB_sub_matrix
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DenseM_t& B, int task_depth) const override;

    std::string type() const override { return "FrontalMatrixBLR"; }

#if defined(STRUMPACK_USE_MPI)
    void extend_add_copy_to_buffers
    (std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa)
      const override {
      if (F22blr_.rows() == std::size_t(dim_upd()))
        abort(); //F22blr_.dense(F22_);
      ExtendAdd<scalar_t,integer_t>::
        extend_add_seq_copy_to_buffers(F22_, sbuf, pa, this);
    }
    void extadd_blr_copy_to_buffers
    (std::vector<std::vector<scalar_t>>& sbuf, const FBLRMPI_t* pa)
      const override {
      if (F22blr_.rows() == std::size_t(dim_upd()))
        abort(); //F22blr_.dense(F22_);
      BLR::BLRExtendAdd<scalar_t,integer_t>::
        seq_copy_to_buffers(F22_, sbuf, pa, this);
    }
    void extadd_blr_copy_to_buffers_col
    (std::vector<std::vector<scalar_t>>& sbuf, const FBLRMPI_t* pa, 
     integer_t begin_col, integer_t end_col, const Opts_t& opts)
      const override {
      if (opts.BLR_options().BLR_CB() == BLR::BLRCB::DENSE) {
        if (F22blr_.rows() == std::size_t(dim_upd()))
          abort(); //F22blr_.dense(F22_);
        BLR::BLRExtendAdd<scalar_t,integer_t>::
          seq_copy_to_buffers_col(F22_, sbuf, pa, this, begin_col, end_col);
      } else if(opts.BLR_options().BLR_CB() == BLR::BLRCB::COLWISE || opts.BLR_options().BLR_CB() == BLR::BLRCB::BLR) {
        BLR::BLRExtendAdd<scalar_t,integer_t>::
          blrseq_copy_to_buffers_col(F22blr_, sbuf, pa, this, begin_col, end_col, opts.BLR_options());
      }
    }
#endif

    void partition
    (const Opts_t& opts, const SpMat_t& A, integer_t* sorder,
     bool is_root=true, int task_depth=0) override;

  private:
    BLRM_t F11blr_, F12blr_, F21blr_, F22blr_;
    DenseM_t F22_;
    std::vector<int> piv_;
    std::vector<std::size_t> sep_tiles_, upd_tiles_;
    DenseMatrix<bool> admissibility_;

    FrontalMatrixBLR(const FrontalMatrixBLR&) = delete;
    FrontalMatrixBLR& operator=(FrontalMatrixBLR const&) = delete;

    void fwd_solve_phase2
    (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const;
    void bwd_solve_phase1
    (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const;

    void draw_node(std::ostream& of, bool is_root) const override;

    long long node_factor_nonzeros() const override;

    using F_t::lchild_;
    using F_t::rchild_;
    using F_t::dim_sep;
    using F_t::dim_upd;
    using F_t::sep_begin_;
    using F_t::sep_end_;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixBLR<scalar_t,integer_t>::FrontalMatrixBLR
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::release_work_memory() {
    F22_.clear();
    F22blr_.clear();
    admissibility_.clear();
    sep_tiles_.clear();
    upd_tiles_.clear();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::build_front(const SpMat_t& A){
    const auto dsep = dim_sep();
    const auto dupd = dim_upd();
    if (dsep) {
      F11blr_ = BLRM_t(dsep, sep_tiles_, dsep, sep_tiles_);
      F11blr_.fill(0.);
      if (dupd) {
        F12blr_ = BLRM_t(dsep, sep_tiles_, dupd, upd_tiles_);
        F21blr_ = BLRM_t(dupd, upd_tiles_, dsep, sep_tiles_);
        F12blr_.fill(0.);
        F21blr_.fill(0.);
      }
    }
    if (dupd) {
      F22blr_ = BLRM_t(dupd, upd_tiles_, dupd,  upd_tiles_);
      F22blr_.fill(0.);
    }
    using Trip_t = Triplet<scalar_t>;
    std::vector<Trip_t> e11, e12, e21;
    A.push_front_elements(sep_begin_, sep_end_, this->upd(), e11, e12, e21);
    for (auto& e : e11) F11blr_(e.r, e.c) = e.v;
    for (auto& e : e12) F12blr_(e.r, e.c) = e.v;
    for (auto& e : e21) F21blr_(e.r, e.c) = e.v;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::build_front_cols(const SpMat_t& A, std::size_t i, 
   bool part, std::size_t CP, const std::vector<Triplet<scalar_t>>& e11, 
   const std::vector<Triplet<scalar_t>>& e12, const std::vector<Triplet<scalar_t>>& e21, int task_depth, const Opts_t& opts){
    const auto dsep = dim_sep();
    const auto dupd = dim_upd();
    if (dsep) {
      if (part) F11blr_.fill_col(0., i, true, CP);
      if (dupd) {
        if (!part) F12blr_.fill_col(0., i, false, CP);
        else F21blr_.fill_col(0., i, true, CP);
      }
    }
    if (dupd) {
      if (!part) F22blr_.fill_col(0., i, false, CP);
    }
    if (part){
      if (dsep) {
        for (auto& e : e11) 
          if (F11blr_.cg2t(e.c) >= i && F11blr_.cg2t(e.c) < i+CP)
            F11blr_(e.r, e.c) = e.v;
      }
      if (dupd){
        for (auto& e : e21) 
          if (F21blr_.cg2t(e.c) >= i && F21blr_.cg2t(e.c) < i+CP)
            F21blr_(e.r, e.c) = e.v;
      }
    } else{
      if (dupd) {
        for (auto& e : e12) 
          if(F12blr_.cg2t(e.c) >= i && F12blr_.cg2t(e.c) < i+CP)
            F12blr_(e.r, e.c) = e.v;
      }
    }
    if (part){
      if (lchild_)
        lchild_->extend_add_to_blr_col(F11blr_, F12blr_, F21blr_, F22blr_, this, F11blr_.tilecoff(i), 
           F11blr_.tilecoff(std::min(i+CP,F11blr_.colblocks())), task_depth, opts);
      if (rchild_)
        rchild_->extend_add_to_blr_col(F11blr_, F12blr_, F21blr_, F22blr_, this, F11blr_.tilecoff(i), 
           F11blr_.tilecoff(std::min(i+CP,F11blr_.colblocks())), task_depth, opts);
    } else{
      if (lchild_)
        lchild_->extend_add_to_blr_col(F11blr_, F12blr_, F21blr_, F22blr_, this, F22blr_.tilecoff(i)+dim_sep(), 
           F22blr_.tilecoff(std::min(i+CP,F22blr_.colblocks()))+dim_sep(), task_depth, opts);
      if (rchild_)
        rchild_->extend_add_to_blr_col(F11blr_, F12blr_, F21blr_, F22blr_, this, F22blr_.tilecoff(i)+dim_sep(), 
           F22blr_.tilecoff(std::min(i+CP,F22blr_.colblocks()))+dim_sep(), task_depth, opts);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const F_t* p, int task_depth) {
    const std::size_t pdsep = paF11.rows();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
    // if ACA was used, a compressed version of the CB was constructed
    // in F22blr_, so we need to expand it first into F22_
    if (F22blr_.rows() == dupd)
      F22blr_.dense(F22_);
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared) grainsize(64)      \
  if(task_depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t c=0; c<dupd; c++) {
      auto pc = I[c];
      if (pc < pdsep) {
        for (std::size_t r=0; r<upd2sep; r++)
          paF11(I[r],pc) += F22_(r,c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF21(I[r]-pdsep,pc) += F22_(r,c);
      } else {
        for (std::size_t r=0; r<upd2sep; r++)
          paF12(I[r],pc-pdsep) += F22_(r, c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF22(I[r]-pdsep,pc-pdsep) += F22_(r,c);
      }
    }
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1) * dupd * dupd);
    STRUMPACK_FULL_RANK_FLOPS((is_complex<scalar_t>()?2:1) * dupd * dupd);
    release_work_memory();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::extend_add_to_blr
  (BLRM_t& paF11, BLRM_t& paF12, BLRM_t& paF21, BLRM_t& paF22,
   const F_t* p, int task_depth, const Opts_t& opts) {
    //extend_add from seq. BLR to seq. BLR
    const std::size_t pdsep = paF11.rows();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
    if (opts.BLR_options().BLRseq_CB_Compression()) F22blr_.decompress(); // change to colwise
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared) grainsize(64)      \
  if(task_depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t c=0; c<dupd; c++) {
      auto pc = I[c];
      if (pc < pdsep) {
        for (std::size_t r=0; r<upd2sep; r++)
          paF11(I[r],pc) += F22blr_(r,c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF21(I[r]-pdsep,pc) += F22blr_(r,c);
      } else {
        for (std::size_t r=0; r<upd2sep; r++)
          paF12(I[r],pc-pdsep) += F22blr_(r, c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF22(I[r]-pdsep,pc-pdsep) += F22blr_(r,c);
      }
    }
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1) * dupd * dupd);
    STRUMPACK_FULL_RANK_FLOPS((is_complex<scalar_t>()?2:1) * dupd * dupd);
    release_work_memory();
   }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::extend_add_to_blr_col
  (BLRM_t& paF11, BLRM_t& paF12, BLRM_t& paF21, BLRM_t& paF22,
   const F_t* p, integer_t begin_col, integer_t end_col, int task_depth, const Opts_t& opts) {
    //extend_add from seq. BLR to seq. BLR
    const std::size_t pdsep = paF11.rows();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
    int c_min = 0, c_max = 0;
    for (std::size_t c=0; c<dupd; c++) {
      auto pc = I[c];
      if (c == dupd-1) c_max = dupd;
      if (pc < std::size_t(begin_col)){
        c_min = c+1;
        continue;
      }
      if (pc >= std::size_t(end_col)){
        c_max = c;
        break;
      }
    }
    if (opts.BLR_options().BLRseq_CB_Compression()) F22blr_.decompress_local_columns(c_min, c_max);
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared) grainsize(64)      \
  if(task_depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t c=0; c<dupd; c++) { //use c_min and c_max
      auto pc = I[c];
      if (pc < std::size_t(begin_col) || pc >= std::size_t(end_col))
        continue;
      if (pc < pdsep) {
        for (std::size_t r=0; r<upd2sep; r++)
          paF11(I[r],pc) += F22blr_(r,c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF21(I[r]-pdsep,pc) += F22blr_(r,c);
      } else {
        for (std::size_t r=0; r<upd2sep; r++)
          paF12(I[r],pc-pdsep) += F22blr_(r, c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF22(I[r]-pdsep,pc-pdsep) += F22blr_(r,c);
      }
    }
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1) * dupd * dupd);
    STRUMPACK_FULL_RANK_FLOPS((is_complex<scalar_t>()?2:1) * dupd * dupd);
    F22blr_.remove_tiles_before_local_column(c_min, c_max);
   }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::sample_CB
  (const Opts_t& opts, const DenseM_t& R, DenseM_t& Sr,
   DenseM_t& Sc, F_t* pa, int task_depth) {
    auto I = this->upd_to_parent(pa);
    auto cR = R.extract_rows(I);
    DenseM_t cS(dim_upd(), R.cols());
    gemm(Trans::N, Trans::N, scalar_t(1.), F22_, cR,
         scalar_t(0.), cS, task_depth);
    Sr.scatter_rows_add(I, cS, task_depth);
    gemm(Trans::C, Trans::N, scalar_t(1.), F22_, cR,
         scalar_t(0.), cS, task_depth);
    Sc.scatter_rows_add(I, cS, task_depth);
    STRUMPACK_CB_SAMPLE_FLOPS
      (gemm_flops(Trans::N, Trans::N, scalar_t(1.), F22_, cR, scalar_t(0.)) +
       gemm_flops(Trans::C, Trans::N, scalar_t(1.), F22_, cR, scalar_t(0.)) +
       cS.rows()*cS.cols()*2); // for the skinny-extend add
  }


  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const Opts_t& opts, int etree_level, int task_depth) {
    if (task_depth == 0) {
      // use tasking for children and for extend-add parallelism
#pragma omp parallel if(!omp_in_parallel()) default(shared)
#pragma omp single nowait
      factor_node(A, opts, etree_level, task_depth);
    } else factor_node(A, opts, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::factor_node
  (const SpMat_t& A, const Opts_t& opts, int etree_level, int task_depth) {
    if (task_depth < params::task_recursion_cutoff_level) {
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
    TaskTimer t("");
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f0 = 0, ftot = 0;
#endif
    if (etree_level == 0 && opts.print_root_front_stats()){
#if defined(STRUMPACK_COUNT_FLOPS)
      f0 = params::flops;
#endif
      t.start();
    }
    const auto dsep = dim_sep();
    const auto dupd = dim_upd();
    if (opts.BLR_options().low_rank_algorithm() ==
        BLR::LowRankAlgorithm::RRQR) {
      if(opts.BLR_options().BLR_CB() == BLR::BLRCB::COLWISE) {
        /* factor column-block-wise for memory reduction*/
        if (dsep) {
          F11blr_ = BLRM_t(dsep, sep_tiles_, dsep, sep_tiles_);
          F12blr_ = BLRM_t(dsep, sep_tiles_, dupd, upd_tiles_);
          F21blr_ = BLRM_t(dupd, upd_tiles_, dsep, sep_tiles_);
          F22blr_ = BLRM_t(dupd, upd_tiles_, dupd, upd_tiles_);
          using Trip_t = Triplet<scalar_t>;
          std::vector<Trip_t> e11, e12, e21;
          A.push_front_elements(sep_begin_, sep_end_, this->upd(), e11, e12, e21);
          BLRM_t::construct_and_partial_factor_col
            (F11blr_, F12blr_, F21blr_, F22blr_, piv_, sep_tiles_, 
            upd_tiles_, admissibility_, opts.BLR_options(),
            [&](int i, bool part, std::size_t CP){this->build_front_cols(A, i, part, CP, e11, e12, e21, task_depth, opts);});
        }
        if (lchild_) lchild_->release_work_memory();
        if (rchild_) rchild_->release_work_memory();
      } else if(opts.BLR_options().BLR_CB() == BLR::BLRCB::BLR) {
        build_front(A);
        if (lchild_)
          lchild_->extend_add_to_blr(F11blr_, F12blr_, F21blr_, F22blr_, this, task_depth, opts);
        if (rchild_)
          rchild_->extend_add_to_blr(F11blr_, F12blr_, F21blr_, F22blr_, this, task_depth, opts);
        if (dsep) {
          BLRM_t::construct_and_partial_factor
            (F11blr_, F12blr_, F21blr_, F22blr_, piv_, sep_tiles_, 
            upd_tiles_, admissibility_, opts.BLR_options());
        }
      } else if(opts.BLR_options().BLR_CB() == BLR::BLRCB::DENSE) {
        DenseM_t F11(dsep, dsep), F12(dsep, dupd), F21(dupd, dsep);
        F11.zero(); F12.zero(); F21.zero();
        A.extract_front(F11, F12, F21, sep_begin_, sep_end_, this->upd_, task_depth);
        if (dupd) {
          F22_ = DenseM_t(dupd, dupd);
          F22_.zero();
        }
        if (lchild_)
          lchild_->extend_add_to_dense(F11, F12, F21, F22_, this, task_depth);
        if (rchild_)
          rchild_->extend_add_to_dense(F11, F12, F21, F22_, this, task_depth);
        if (dsep) {
          BLRM_t::construct_and_partial_factor
            (F11, F12, F21, F22_, F11blr_, piv_, F12blr_, F21blr_,
            sep_tiles_, upd_tiles_, admissibility_, opts.BLR_options());
        }
      }
#if 0
      DenseM_t F11(dsep, dsep), F12(dsep, dupd), F21(dupd, dsep);
      F11.zero(); F12.zero(); F21.zero();
      A.extract_front(F11, F12, F21, sep_begin_, sep_end_, this->upd_, task_depth);
      if (dupd) {
        F22_ = DenseM_t(dupd, dupd);
        F22_.zero();
      }
      if (lchild_)
        lchild_->extend_add_to_dense(F11, F12, F21, F22_, this, task_depth);
      if (rchild_)
        rchild_->extend_add_to_dense(F11, F12, F21, F22_, this, task_depth);
      if (dsep) {
        F11blr_ = BLRM_t
          (F11, sep_tiles_, admissibility_, piv_, opts.BLR_options());
        F11.clear();
        if (dupd) {
          F12.laswp(piv_, true);
          trsm(Side::L, UpLo::L, Trans::N, Diag::U,
              scalar_t(1.), F11blr_, F12, task_depth);
          F12blr_ = BLRM_t(F12, sep_tiles_, upd_tiles_, opts.BLR_options());
          F12.clear();
          trsm(Side::R, UpLo::U, Trans::N, Diag::N,
              scalar_t(1.), F11blr_, F21, task_depth);
          F21blr_ = BLRM_t(F21, upd_tiles_, sep_tiles_, opts.BLR_options());
          F21.clear();
          gemm(Trans::N, Trans::N, scalar_t(-1.), F21blr_, F12blr_,
              scalar_t(1.), F22_, task_depth);
        }
      }
#endif
    } else { // ACA or BACA
      auto F11elem = [&](const std::vector<std::size_t>& lI,
                         const std::vector<std::size_t>& lJ, DenseM_t& B) {
        auto gI = lI; auto gJ = lJ;
        for (auto& i : gI) i += sep_begin_;
        for (auto& j : gJ) j += sep_begin_;
        A.extract_separator(sep_end_, gI, gJ, B, task_depth);
        if (lchild_) lchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
        if (rchild_) rchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
      };
      auto F12elem = [&](const std::vector<std::size_t>& lI,
                         const std::vector<std::size_t>& lJ, DenseM_t& B) {
        auto gI = lI; auto gJ = lJ;
        for (auto& i : gI) i += sep_begin_;
        for (auto& j : gJ) j = this->upd_[j];
        A.extract_separator(sep_end_, gI, gJ, B, task_depth);
        if (lchild_) lchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
        if (rchild_) rchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
      };
      auto F21elem = [&](const std::vector<std::size_t>& lI,
                         const std::vector<std::size_t>& lJ, DenseM_t& B) {
        auto gI = lI; auto gJ = lJ;
        for (auto& i : gI) i = this->upd_[i];
        for (auto& j : gJ) j += sep_begin_;
        A.extract_separator(sep_end_, gI, gJ, B, task_depth);
        if (lchild_) lchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
        if (rchild_) rchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
      };
      auto F22elem = [&](const std::vector<std::size_t>& lI,
                         const std::vector<std::size_t>& lJ, DenseM_t& B) {
        B.zero();
        auto gI = lI; auto gJ = lJ;
        for (auto& i : gI) i = this->upd_[i];
        for (auto& j : gJ) j = this->upd_[j];
        if (lchild_) lchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
        if (rchild_) rchild_->extract_CB_sub_matrix(gI, gJ, B, task_depth);
      };
      BLRM_t::construct_and_partial_factor
        (dsep, dupd, F11elem, F12elem, F21elem, F22elem,
         F11blr_, piv_, F12blr_, F21blr_, F22blr_,
         sep_tiles_, upd_tiles_, admissibility_, opts.BLR_options());
      if (lchild_) lchild_->release_work_memory();
      if (rchild_) rchild_->release_work_memory();
    }
    // TODO flops
    /*if (opts.print_root_front_stats()) {
      std::cout << "#   - BLR front: N = " << dim_sep()
                << " , N^2 = " << dim_sep() * dim_sep()
                << ", peak memory= " << double(strumpack::params::peak_memory) / 1.0e6 << " MB" << std::endl;
    }*/
    if (etree_level == 0 && opts.print_root_front_stats()) {
      auto time = t.elapsed();
      auto nnz = F11blr_.nonzeros();
      std::cout << "#   - BLR root front: N = " << dim_sep()
                << " , N^2 = " << dim_sep() * dim_sep()
                << " , nnz = " << nnz
                << " , " << float(nnz) / (dim_sep()*dim_sep()) * 100.
                << " % compression, time = " << time
                << " sec, " 
                << " maxrank= " << F11blr_.maximum_rank() << std::endl;
#if defined(STRUMPACK_COUNT_FLOPS)
      ftot = params::flops - f0;
      std::cout << "#   - BLR root front: factor flops = " << double(ftot) << std::endl;
#endif 
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t bupd(dim_upd(), b.cols(), work[0], 0, 0);
    bupd.zero();
    if (task_depth == 0) {
      // tasking when calling the children
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      this->fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
      // no tasking for the root node computations, use system blas threading!
      fwd_solve_phase2(b, bupd, etree_level, params::task_recursion_cutoff_level);
    } else {
      this->fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
      fwd_solve_phase2(b, bupd, etree_level, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
      bloc.laswp(piv_, true);
#if 1
      BLRM_t::trsmLNU_gemm(F11blr_, F21blr_, bloc, bupd, task_depth);
#else
      if (b.cols() == 1) {
        trsv(UpLo::L, Trans::N, Diag::U, F11blr_, bloc, task_depth);
        if (dim_upd())
          gemv(Trans::N, scalar_t(-1.), F21blr_, bloc,
               scalar_t(1.), bupd, task_depth);
      } else {
        trsm(Side::L, UpLo::L, Trans::N, Diag::U,
             scalar_t(1.), F11blr_, bloc, task_depth);
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F21blr_, bloc,
               scalar_t(1.), bupd, task_depth);
      }
#endif
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t yupd(dim_upd(), y.cols(), work[0], 0, 0);
    if (task_depth == 0) {
      // no tasking in blas routines, use system threaded blas instead
      bwd_solve_phase1
        (y, yupd, etree_level, params::task_recursion_cutoff_level);
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      // tasking when calling children
      this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
    } else {
      bwd_solve_phase1(y, yupd, etree_level, task_depth);
      this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::bwd_solve_phase1
  (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t yloc(dim_sep(), y.cols(), y, this->sep_begin_, 0);
#if 1
      BLRM_t::gemm_trsmUNN(F11blr_, F12blr_, yloc, yupd, task_depth);
#else
      if (y.cols() == 1) {
        if (dim_upd())
          gemv(Trans::N, scalar_t(-1.), F12blr_, yupd,
               scalar_t(1.), yloc, task_depth);
        trsv(UpLo::U, Trans::N, Diag::N, F11blr_, yloc, task_depth);
      } else {
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F12blr_, yupd,
               scalar_t(1.), yloc, task_depth);
        trsm(Side::L, UpLo::U, Trans::N, Diag::N,
             scalar_t(1.), F11blr_, yloc, task_depth);
      }
#endif
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::extract_CB_sub_matrix
  (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
   DenseM_t& B, int task_depth) const {
    std::vector<std::size_t> lJ, oJ;
    this->find_upd_indices(J, lJ, oJ);
    if (lJ.empty()) return;
    std::vector<std::size_t> lI, oI;
    this->find_upd_indices(I, lI, oI);
    if (lI.empty()) return;
    const std::size_t dupd = dim_upd();
    if (F22blr_.rows() == dupd) {
      // extract requires the indices to be sorted
      // auto T = F22blr_.extract(lI, lJ);
      // for (std::size_t j=0; j<lJ.size(); j++)
      //   for (std::size_t i=0; i<lI.size(); i++)
      //     B(oI[i], oJ[j]) += T(i, j);
      for (std::size_t j=0; j<lJ.size(); j++)
        for (std::size_t i=0; i<lI.size(); i++)
          B(oI[i], oJ[j]) += F22blr_(lI[i], lJ[j]);
    } else {
      for (std::size_t j=0; j<lJ.size(); j++)
        for (std::size_t i=0; i<lI.size(); i++)
          B(oI[i], oJ[j]) += F22_(lI[i], lJ[j]);
    }
    STRUMPACK_FLOPS((is_complex<scalar_t>() ? 2 : 1) * lJ.size() * lI.size());
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixBLR<scalar_t,integer_t>::node_factor_nonzeros() const {
    return F11blr_.nonzeros() + F12blr_.nonzeros() + F21blr_.nonzeros();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::partition
  (const Opts_t& opts, const SpMat_t& A,
   integer_t* sorder, bool is_root, int task_depth) {
    if (dim_sep()) {
      auto g = A.extract_graph
        (opts.separator_ordering_level(), sep_begin_, sep_end_);
      auto sep_tree = g.recursive_bisection
        (opts.compression_leaf_size(1), 0,
         sorder+sep_begin_, nullptr, 0, 0, dim_sep());
      std::vector<integer_t> siorder(dim_sep());
      for (integer_t i=sep_begin_; i<sep_end_; i++)
        siorder[sorder[i]] = i - sep_begin_;
      g.permute(sorder+sep_begin_, siorder.data());
      for (integer_t i=sep_begin_; i<sep_end_; i++)
        sorder[i] = sorder[i] + sep_begin_;
      sep_tiles_ = sep_tree.template leaf_sizes<std::size_t>();
      admissibility_ = g.admissibility(sep_tiles_);
    }
    if (dim_upd()) {
      auto leaf = opts.BLR_options().leaf_size();
      auto nt = std::ceil(float(dim_upd()) / leaf);
      upd_tiles_.resize(nt, leaf);
      upd_tiles_.back() = dim_upd() - leaf*(nt-1);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::draw_node
  (std::ostream& of, bool is_root) const {
    F11blr_.draw(of, this->sep_begin(), this->sep_begin());
  }

} // end namespace strumpack

#endif
