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
    (integer_t _sep, integer_t _sep_begin, integer_t _sep_end,
     std::vector<integer_t>& _upd);
    ~FrontalMatrixBLR() {}
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

    void fwd_solve_phase1
    (DenseM_t& b, DenseM_t& bupd, DenseM_t* work,
     int etree_level, int task_depth) const;
    void fwd_solve_phase2
    (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const;
    void bwd_solve_phase1
    (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const;
    void bwd_solve_phase2
    (DenseM_t& y, DenseM_t& yupd, DenseM_t* work,
     int etree_level, int task_depth) const;

    long long node_factor_nonzeros() const override;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixBLR<scalar_t,integer_t>::FrontalMatrixBLR
  (integer_t _sep, integer_t _sep_begin, integer_t _sep_end,
   std::vector<integer_t>& _upd)
    : FrontalMatrix<scalar_t,integer_t>
    (NULL, NULL, _sep, _sep_begin, _sep_end, _upd) {}

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const FrontalMatrix<scalar_t,integer_t>* p, int task_depth) {
    const std::size_t pdsep = paF11.rows();
    const std::size_t dupd = this->dim_upd();
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
    DenseM_t cS(this->dim_upd(), R.cols());
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
    if (task_depth < params::task_recursion_cutoff_level) {
      if (this->lchild)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        this->lchild->multifrontal_factorization
          (A, opts, etree_level+1, task_depth+1);
      if (this->rchild)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        this->rchild->multifrontal_factorization
          (A, opts, etree_level+1, task_depth+1);
#pragma omp taskwait
    } else {
      if (this->lchild)
        this->lchild->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
      if (this->rchild)
        this->rchild->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
    }
    // TODO can we allocate the memory in one go??
    const auto dsep = this->dim_sep();
    const auto dupd = this->dim_upd();
    F11_ = DenseM_t(dsep, dsep); F11_.zero();
    F12_ = DenseM_t(dsep, dupd); F12_.zero();
    F21_ = DenseM_t(dupd, dsep); F21_.zero();
    A.extract_front
      (F11_, F12_, F21_, this->sep_begin, this->sep_end,
       this->upd, task_depth);
    if (dupd) {
      F22_ = DenseM_t(dupd, dupd);
      F22_.zero();
    }
    if (this->lchild)
      this->lchild->extend_add_to_dense
        (F11_, F12_, F21_, F22_, this, task_depth);
    if (this->rchild)
      this->rchild->extend_add_to_dense
        (F11_, F12_, F21_, F22_, this, task_depth);
    if (this->dim_sep()) {
      F11blr_ = BLRM_t(sep_tiles_,
                       [&](std::size_t i, std::size_t j) -> bool {
                         return adm_[i+j*sep_tiles_.size()]; },
                       F11_, piv_, opts.BLR_options());
      F11_.clear();
      if (this->dim_upd()) {
        F12_.laswp(piv_, true);
        trsm(Side::L, UpLo::L, Trans::N, Diag::U,
             scalar_t(1.), F11blr_, F12_, task_depth);
        F12blr_ = BLRM_t(sep_tiles_, upd_tiles_, F12_, opts.BLR_options());
        F12_.clear();
        trsm(Side::R, UpLo::U, Trans::N, Diag::N,
             scalar_t(1.), F11blr_, F21_, task_depth);
        F21blr_ = BLRM_t(upd_tiles_, sep_tiles_, F21_, opts.BLR_options());
        F21_.clear();
        gemm(Trans::N, Trans::N, scalar_t(-1.), F21blr_, F12blr_,
             scalar_t(1.), F22_, task_depth);
      }
    }
    // TODO flops
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtAdd::extend_add_seq_copy_to_buffers(F22_, sbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t bupd(this->dim_upd(), b.cols(), work[0], 0, 0);
    bupd.zero();
    if (task_depth == 0) {
      // tasking when calling the children
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
      // no tasking for the root node computations, use system blas threading!
      return fwd_solve_phase2
        (b, bupd, etree_level, params::task_recursion_cutoff_level);
    } else {
      fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
      return fwd_solve_phase2(b, bupd, etree_level, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::fwd_solve_phase1
  (DenseM_t& b, DenseM_t& bupd, DenseM_t* work,
   int etree_level, int task_depth) const {
    if (task_depth < params::task_recursion_cutoff_level) {
      if (this->lchild)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        this->lchild->forward_multifrontal_solve
          (b, work+1, etree_level+1, task_depth+1);
      if (this->rchild)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          std::vector<DenseM_t> work2(this->rchild->levels());
          for (auto& cb : work2)
            cb = DenseM_t(this->rchild->max_dim_upd(), b.cols());
          this->rchild->forward_multifrontal_solve
            (b, work2.data(), etree_level+1, task_depth+1);
          DenseMW_t CBch
            (this->rchild->dim_upd(), b.cols(), work2[0], 0, 0);
          this->extend_add_b(this->rchild, b, bupd, CBch);
        }
#pragma omp taskwait
      if (this->lchild) {
        DenseMW_t CBch
          (this->lchild->dim_upd(), b.cols(), work[1], 0, 0);
        this->extend_add_b(this->lchild, b, bupd, CBch);
      }
    } else {
      if (this->lchild) {
        this->lchild->forward_multifrontal_solve
          (b, work+1, etree_level+1, task_depth);
        DenseMW_t CBch
          (this->lchild->dim_upd(), b.cols(), work[1], 0, 0);
        this->extend_add_b(this->lchild, b, bupd, CBch);
      }
      if (this->rchild) {
        this->rchild->forward_multifrontal_solve
          (b, work+1, etree_level+1, task_depth);
        DenseMW_t CBch
          (this->rchild->dim_upd(), b.cols(), work[1], 0, 0);
        this->extend_add_b(this->rchild, b, bupd, CBch);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    if (this->dim_sep()) {
      DenseMW_t bloc(this->dim_sep(), b.cols(), b, this->sep_begin, 0);
      bloc.laswp(piv_, true);
      if (b.cols() == 1) {
        trsv(UpLo::L, Trans::N, Diag::U, F11blr_, bloc, task_depth);
        if (this->dim_upd())
          gemv(Trans::N, scalar_t(-1.), F21blr_, bloc,
               scalar_t(1.), bupd, task_depth);
      } else {
        trsm(Side::L, UpLo::L, Trans::N, Diag::U,
             scalar_t(1.), F11blr_, bloc, task_depth);
        if (this->dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F21blr_, bloc,
               scalar_t(1.), bupd, task_depth);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t yupd(this->dim_upd(), y.cols(), work[0], 0, 0);
    if (task_depth == 0) {
      // no tasking in blas routines, use system threaded blas instead
      bwd_solve_phase1
        (y, yupd, etree_level, params::task_recursion_cutoff_level);
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      // tasking when calling children
      bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
    } else {
      bwd_solve_phase1(y, yupd, etree_level, task_depth);
      bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::bwd_solve_phase1
  (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const {
    if (this->dim_sep()) {
      DenseMW_t yloc(this->dim_sep(), y.cols(), y, this->sep_begin, 0);
      if (y.cols() == 1) {
        if (this->dim_upd())
          gemv(Trans::N, scalar_t(-1.), F12blr_, yupd,
               scalar_t(1.), yloc, task_depth);
        trsv(UpLo::U, Trans::N, Diag::N, F11blr_, yloc, task_depth);
      } else {
        if (this->dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F12blr_, yupd,
               scalar_t(1.), yloc, task_depth);
        trsm(Side::L, UpLo::U, Trans::N, Diag::N,
             scalar_t(1.), F11blr_, yloc, task_depth);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::bwd_solve_phase2
  (DenseM_t& y, DenseM_t& yupd, DenseM_t* work,
   int etree_level, int task_depth) const {
    if (task_depth < params::task_recursion_cutoff_level) {
      if (this->lchild) {
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          DenseMW_t CB(this->lchild->dim_upd(), y.cols(), work[1], 0, 0);
          this->extract_b(this->lchild, y, yupd, CB);
          this->lchild->backward_multifrontal_solve
            (y, work+1, etree_level+1, task_depth+1);
        }
      }
      if (this->rchild)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          std::vector<DenseM_t> work2(this->rchild->levels());
          for (auto& cb : work2)
            cb = DenseM_t(this->rchild->max_dim_upd(), y.cols());
          DenseMW_t CB(this->rchild->dim_upd(), y.cols(), work2[0], 0, 0);
          this->extract_b(this->rchild, y, yupd, CB);
          this->rchild->backward_multifrontal_solve
            (y, work2.data(), etree_level+1, task_depth+1);
        }
#pragma omp taskwait
    } else {
      if (this->lchild) {
        DenseMW_t CB(this->lchild->dim_upd(), y.cols(), work[1], 0, 0);
        this->extract_b(this->lchild, y, yupd, CB);
        this->lchild->backward_multifrontal_solve
          (y, work+1, etree_level+1, task_depth);
      }
      if (this->rchild) {
        DenseMW_t CB(this->rchild->dim_upd(), y.cols(), work[1], 0, 0);
        this->extract_b(this->rchild, y, yupd, CB);
        this->rchild->backward_multifrontal_solve
          (y, work+1, etree_level+1, task_depth);
      }
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
    if (this->dim_sep()) {
      assert(sep_tree.size == this->dim_sep());
      auto lf = sep_tree.leaf_sizes();
      sep_tiles_.assign(lf.begin(), lf.end());
      adm_.resize(sep_tiles_.size()*sep_tiles_.size(), true);
    }
    if (this->dim_upd()) {
      auto leaf = opts.BLR_options().leaf_size();
      auto nt = std::ceil(float(this->dim_upd()) / leaf);
      upd_tiles_.resize(nt, leaf);
      upd_tiles_.back() = this->dim_upd() - leaf*(nt-1);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixBLR<scalar_t,integer_t>::set_BLR_partitioning
  (const SPOptions<scalar_t>& opts, const HSS::HSSPartitionTree& sep_tree,
   const std::vector<bool>& adm, bool is_root) {
    if (this->dim_sep()) {
      assert(sep_tree.size == this->dim_sep());
      auto lf = sep_tree.leaf_sizes();
      sep_tiles_.assign(lf.begin(), lf.end());
      adm_ = adm;
    }
    if (this->dim_upd()) {
      auto leaf = opts.BLR_options().leaf_size();
      auto nt = std::ceil(float(this->dim_upd()) / leaf);
      upd_tiles_.resize(nt, leaf);
      upd_tiles_.back() = this->dim_upd() - leaf*(nt-1);
    }
  }

} // end namespace strumpack

#endif
