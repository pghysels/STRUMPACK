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
#include "CompressedSparseMatrix.hpp"
#include "MatrixReordering.hpp"
#include "BLR/BLRMatrix.hpp"
#include "ExtendAdd.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixBLR
    : public FrontalMatrix<scalar_t,integer_t> {
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using ExtAdd = ExtendAdd<scalar_t,integer_t>;
    using BLRM_t = BLR::BLRMatrix<scalar_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;

  public:
    FrontalMatrixBLR
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd);

    void release_work_memory() { F22_.clear(); }
    void extend_add_to_dense
    (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
     const FrontalMatrix<scalar_t,integer_t>* p, int task_depth) override;
    void sample_CB
    (const SPOptions<scalar_t>& opts, const DenseM_t& R, DenseM_t& Sr,
     DenseM_t& Sc, FrontalMatrix<scalar_t,integer_t>* pa,
     int task_depth) override;
    void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level=0, int task_depth=0) override;
    void factor_node
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
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

    void extend_add_copy_to_buffers
    (std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const override;

    void set_BLR_partitioning
    (const SPOptions<scalar_t>& opts,
     const HSS::HSSPartitionTree& sep_tree,
     const std::vector<bool>& adm, bool is_root) override;
    void set_HSS_partitioning
    (const SPOptions<scalar_t>& opts,
     const HSS::HSSPartitionTree& sep_tree,
     bool is_root) override;

  private:
    DenseM_t F11_, F12_, F21_, F22_;
    BLRM_t F11blr_, F12blr_, F21blr_;
    std::vector<int> piv_;
    std::vector<std::size_t> sep_tiles_;
    std::vector<std::size_t> upd_tiles_;
    std::vector<bool> adm_;

    FrontalMatrixBLR(const FrontalMatrixBLR&) = delete;
    FrontalMatrixBLR& operator=(FrontalMatrixBLR const&) = delete;

    void fwd_solve_phase2
    (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const;
    void bwd_solve_phase1
    (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const;

    void draw_node(std::ostream& of, bool is_root) const override;

    long long node_factor_nonzeros() const override;

    using FrontalMatrix<scalar_t,integer_t>::lchild_;
    using FrontalMatrix<scalar_t,integer_t>::rchild_;
    using FrontalMatrix<scalar_t,integer_t>::dim_sep;
    using FrontalMatrix<scalar_t,integer_t>::dim_upd;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixBLR<scalar_t,integer_t>::FrontalMatrixBLR
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : FrontalMatrix<scalar_t,integer_t>
    (nullptr, nullptr, sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const FrontalMatrix<scalar_t,integer_t>* p, int task_depth) {
    const std::size_t pdsep = paF11.rows();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
#pragma omp taskloop default(shared) grainsize(64)      \
  if(task_depth < params::task_recursion_cutoff_level)
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
  FrontalMatrixBLR<scalar_t,integer_t>::sample_CB
  (const SPOptions<scalar_t>& opts, const DenseM_t& R, DenseM_t& Sr,
   DenseM_t& Sc, FrontalMatrix<scalar_t,integer_t>* pa, int task_depth) {
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
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (task_depth == 0) {
      // use tasking for children and for extend-add parallelism
#pragma omp parallel if(!omp_in_parallel()) default(shared)
#pragma omp single nowait
      factor_node(A, opts, etree_level, task_depth);
    } else factor_node(A, opts, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::factor_node
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
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
    const auto dsep = dim_sep();
    const auto dupd = dim_upd();
    F11_ = DenseM_t(dsep, dsep); F11_.zero();
    F12_ = DenseM_t(dsep, dupd); F12_.zero();
    F21_ = DenseM_t(dupd, dsep); F21_.zero();
    A.extract_front
      (F11_, F12_, F21_, this->sep_begin_, this->sep_end_,
       this->upd_, task_depth);
    if (dupd) {
      F22_ = DenseM_t(dupd, dupd);
      F22_.zero();
    }
    if (lchild_)
      lchild_->extend_add_to_dense
        (F11_, F12_, F21_, F22_, this, task_depth);
    if (rchild_)
      rchild_->extend_add_to_dense
        (F11_, F12_, F21_, F22_, this, task_depth);
    if (dim_sep()) {
#if 1
      BLR::BLR_construct_and_partial_factor
        (F11_, F12_, F21_, F22_, F11blr_, piv_, F12blr_, F21blr_,
         sep_tiles_, upd_tiles_,
         [&](std::size_t i, std::size_t j) -> bool {
          return adm_[i+j*sep_tiles_.size()]; },
         opts.BLR_options());
#else
      F11blr_ = BLRM_t(sep_tiles_,
                       [&](std::size_t i, std::size_t j) -> bool {
                         return adm_[i+j*sep_tiles_.size()]; },
                       F11_, piv_, opts.BLR_options());
      F11_.clear();
      if (dim_upd()) {
        F12_.laswp(piv_, true);
        trsm(Side::L, UpLo::L, Trans::N, Diag::U,
             scalar_t(1.), F11blr_, F12_, task_depth);
        F12blr_ = BLRM_t(F12_, sep_tiles_, upd_tiles_, opts.BLR_options());
        F12_.clear();
        trsm(Side::R, UpLo::U, Trans::N, Diag::N,
             scalar_t(1.), F11blr_, F21_, task_depth);
        F21blr_ = BLRM_t(F21_, upd_tiles_, sep_tiles_, opts.BLR_options());
        F21_.clear();
        gemm(Trans::N, Trans::N, scalar_t(-1.), F21blr_, F12blr_,
             scalar_t(1.), F22_, task_depth);
      }
#endif
      // TODO flops
    }
  }
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtAdd::extend_add_seq_copy_to_buffers(F22_, sbuf, pa, this);
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
#if 0
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
#else
      BLR::BLR_trsmLNU_gemm(F11blr_, F21blr_, bloc, bupd, task_depth);
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
#if 0
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
#else
      BLR::BLR_gemm_trsmUNN(F11blr_, F12blr_, yloc, yupd, task_depth);
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
    for (std::size_t j=0; j<lJ.size(); j++)
      for (std::size_t i=0; i<lI.size(); i++)
        B(oI[i], oJ[j]) += F22_(lI[i], lJ[j]);
    STRUMPACK_FLOPS((is_complex<scalar_t>() ? 2 : 1) * lJ.size() * lI.size());
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixBLR<scalar_t,integer_t>::node_factor_nonzeros() const {
    return F11blr_.nonzeros() + F12blr_.nonzeros() + F21blr_.nonzeros();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::set_HSS_partitioning
  (const SPOptions<scalar_t>& opts, const HSS::HSSPartitionTree& sep_tree,
   bool is_root) {
    std::cout << "set admissibility condition!!" << std::endl;
    if (dim_sep()) {
      assert(sep_tree.size == dim_sep());
      auto lf = sep_tree.leaf_sizes();
      sep_tiles_.assign(lf.begin(), lf.end());
      adm_.resize(sep_tiles_.size()*sep_tiles_.size(), true);
    }
    if (dim_upd()) {
      auto leaf = opts.BLR_options().leaf_size();
      auto nt = std::ceil(float(dim_upd()) / leaf);
      upd_tiles_.resize(nt, leaf);
      upd_tiles_.back() = dim_upd() - leaf*(nt-1);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::set_BLR_partitioning
  (const SPOptions<scalar_t>& opts, const HSS::HSSPartitionTree& sep_tree,
   const std::vector<bool>& adm, bool is_root) {
    if (dim_sep()) {
      assert(sep_tree.size == dim_sep());
      auto lf = sep_tree.leaf_sizes();
      sep_tiles_.assign(lf.begin(), lf.end());
      adm_ = adm;
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
