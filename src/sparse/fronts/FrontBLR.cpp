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

#include <iostream>
#include <fstream>

#include "FrontBLR.hpp"
#include "sparse/CSRGraph.hpp"
#include "misc/TaskTimer.hpp"
#include "dense/BLASLAPACKWrapper.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#include "BLR/BLRExtendAdd.hpp"
#endif
#if defined(STRUMPACK_USE_GPU)
#include "FrontalMatrixGPUKernels.hpp"
#endif

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  FrontBLR<scalar_t,integer_t>::FrontBLR
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t> void
  FrontBLR<scalar_t,integer_t>::release_work_memory
  (VectorPool<scalar_t>& workspace) {
#if defined(STRUMPACK_USE_GPU)
    // CBdev_.release();
    workspace.restore(CBdev_);
#endif
    workspace.restore(CBstorage_);
    F22_.clear();
    F22blr_.clear();
    admissibility_.clear();
    sep_tiles_.clear();
    upd_tiles_.clear();
  }

  template<typename scalar_t,typename integer_t> void
  FrontBLR<scalar_t,integer_t>::build_front_cols
  (const SpMat_t& A, std::size_t i, bool part, std::size_t CP,
   const std::vector<Triplet<scalar_t>>& e11,
   const std::vector<Triplet<scalar_t>>& e12,
   const std::vector<Triplet<scalar_t>>& e21,
   int task_depth, const Opts_t& opts) {
    const auto dsep = dim_sep();
    const auto dupd = dim_upd();
    if (dsep) {
      if (part)
        F11blr_.fill_col(0., i, CP);
      if (dupd) {
        if (!part) F12blr_.fill_col(0., i, CP);
        else F21blr_.fill_col(0., i, CP);
      }
    }
    if (dupd && !part)
      F22blr_.fill_col(0., i, CP);
    if (part) {
      if (dsep)
        for (auto& e : e11)
          if (F11blr_.cg2t(e.c) >= i && F11blr_.cg2t(e.c) < i+CP)
            F11blr_(e.r, e.c) = e.v;
      if (dupd)
        for (auto& e : e21)
          if (F21blr_.cg2t(e.c) >= i && F21blr_.cg2t(e.c) < i+CP)
            F21blr_(e.r, e.c) = e.v;
    } else {
      if (dupd)
        for (auto& e : e12)
          if (F12blr_.cg2t(e.c) >= i && F12blr_.cg2t(e.c) < i+CP)
            F12blr_(e.r, e.c) = e.v;
    }
    if (part) {
      if (lchild_)
        lchild_->extend_add_to_blr_col
          (F11blr_, F12blr_, F21blr_, F22blr_, this, F11blr_.tilecoff(i),
           F11blr_.tilecoff(std::min(i+CP, F11blr_.colblocks())),
           task_depth, opts);
      if (rchild_)
        rchild_->extend_add_to_blr_col
          (F11blr_, F12blr_, F21blr_, F22blr_, this, F11blr_.tilecoff(i),
           F11blr_.tilecoff(std::min(i+CP, F11blr_.colblocks())),
           task_depth, opts);
    } else {
      if (lchild_)
        lchild_->extend_add_to_blr_col
          (F11blr_, F12blr_, F21blr_, F22blr_, this,
           F22blr_.tilecoff(i) + dim_sep(),
           F22blr_.tilecoff(std::min(i+CP, F22blr_.colblocks())) + dim_sep(),
           task_depth, opts);
      if (rchild_)
        rchild_->extend_add_to_blr_col
          (F11blr_, F12blr_, F21blr_, F22blr_, this,
           F22blr_.tilecoff(i) + dim_sep(),
           F22blr_.tilecoff(std::min(i+CP,F22blr_.colblocks())) + dim_sep(),
           task_depth, opts);
    }
  }

  template<typename scalar_t,typename integer_t> scalar_t*
  FrontBLR<scalar_t,integer_t>::get_device_F22(scalar_t* dF22) {
    return F22_.data();
  }

  template<typename scalar_t,typename integer_t> void
  FrontBLR<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const F_t* p, int task_depth) {
    VectorPool<scalar_t> workspace;
    extend_add_to_dense(paF11, paF12, paF21, paF22, p, workspace, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontBLR<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const F_t* p, VectorPool<scalar_t>& workspace, int task_depth) {
    const std::size_t dupd = dim_upd();
#if defined(STRUMPACK_USE_GPU)
    if (CBdev_.size()) {
      DenseM_t F22(dupd, dupd);
      gpu::copy_device_to_host(F22, CBdev_.template as<scalar_t>());
      this->extend_add(paF11, paF12, paF21, paF22, F22, p);
    } else
#endif
      {
        if (F22blr_.rows() == dupd) {
          auto F22 = F22blr_.dense();
          this->extend_add(paF11, paF12, paF21, paF22, F22, p);
        } else
          this->extend_add(paF11, paF12, paF21, paF22, F22_, p);
      }
    release_work_memory(workspace);
  }

  template<typename scalar_t,typename integer_t> void
  FrontBLR<scalar_t,integer_t>::extend_add_to_blr
  (BLRM_t& paF11, BLRM_t& paF12, BLRM_t& paF21, BLRM_t& paF22,
   const F_t* p, VectorPool<scalar_t>& workspace,
   int task_depth, const Opts_t& opts) {
    // extend_add from seq. BLR to seq. BLR
    const std::size_t pdsep = paF11.rows();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
    if (opts.BLR_options().BLR_factor_algorithm() ==
        BLR::BLRFactorAlgorithm::COLWISE)
      F22blr_.decompress(); // change to colwise
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
    release_work_memory(workspace);
   }

  template<typename scalar_t,typename integer_t> void
  FrontBLR<scalar_t,integer_t>::extend_add_to_blr_col
  (BLRM_t& paF11, BLRM_t& paF12, BLRM_t& paF21, BLRM_t& paF22,
   const F_t* p, integer_t begin_col, integer_t end_col,
   int task_depth, const Opts_t& opts) {
    // extend_add from seq. BLR to seq. BLR
    const std::size_t pdsep = paF11.rows();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
    int c_min = 0, c_max = 0;
    for (std::size_t c=0; c<dupd; c++) {
      auto pc = I[c];
      if (c == dupd-1) c_max = dupd;
      if (pc < std::size_t(begin_col)) {
        c_min = c+1;
        continue;
      }
      if (pc >= std::size_t(end_col)) {
        c_max = c;
        break;
      }
    }
    if (opts.BLR_options().BLR_factor_algorithm() ==
        BLR::BLRFactorAlgorithm::COLWISE)
      F22blr_.decompress_local_columns(c_min, c_max);
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared) grainsize(64)      \
  if(task_depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t c=0; c<dupd; c++) { // TODO use c_min and c_max
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
  FrontBLR<scalar_t,integer_t>::sample_CB
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


  template<typename scalar_t,typename integer_t> ReturnCode
  FrontBLR<scalar_t,integer_t>::factor
  (const SpMat_t& A, const Opts_t& opts, VectorPool<scalar_t>& workspace,
   int etree_level, int task_depth) {
    ReturnCode e = ReturnCode::SUCCESS;
    if (task_depth == 0) {
#pragma omp parallel if(!omp_in_parallel()) default(shared)
#pragma omp single nowait
      e = factor_node(A, opts, workspace, etree_level, task_depth);
    } else e = factor_node(A, opts, workspace, etree_level, task_depth);
    return e;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontBLR<scalar_t,integer_t>::factor_node
  (const SpMat_t& A, const Opts_t& opts, VectorPool<scalar_t>& workspace,
   int etree_level, int task_depth) {
    ReturnCode el = ReturnCode::SUCCESS, er = ReturnCode::SUCCESS;
    if (opts.use_openmp_tree() &&
        !opts.use_gpu() && // do not create too many GPU streams, handles, etc
        task_depth < params::task_recursion_cutoff_level) {
      if (lchild_)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        el = lchild_->factor(A, opts, workspace, etree_level+1, task_depth+1);
      if (rchild_)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        er = rchild_->factor(A, opts, workspace, etree_level+1, task_depth+1);
#pragma omp taskwait
    } else {
      if (lchild_) {
        el = lchild_->factor(A, opts, workspace, etree_level+1, task_depth);
      }
      if (rchild_)
        er = rchild_->factor(A, opts, workspace, etree_level+1, task_depth);
    }
    ReturnCode err_code = (el == ReturnCode::SUCCESS) ? er : el;
    TaskTimer t("");
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f0 = 0, ftot = 0;
#endif
    if (opts.print_compressed_front_stats()) {
#if defined(STRUMPACK_COUNT_FLOPS)
      f0 = params::flops;
#endif
      t.start();
    }
    const auto dsep = dim_sep();
    const auto dupd = dim_upd();
    auto& blr_opts = opts.BLR_options();
    if (blr_opts.low_rank_algorithm() ==
        BLR::LowRankAlgorithm::RRQR) {
      if (blr_opts.BLR_factor_algorithm() ==
          BLR::BLRFactorAlgorithm::COLWISE) {
        // factor column-block-wise for memory reduction
        if (dsep) {
          F11blr_ = BLRM_t(dsep, sep_tiles_, dsep, sep_tiles_);
          F12blr_ = BLRM_t(dsep, sep_tiles_, dupd, upd_tiles_);
          F21blr_ = BLRM_t(dupd, upd_tiles_, dsep, sep_tiles_);
          F22blr_ = BLRM_t(dupd, upd_tiles_, dupd, upd_tiles_);
          using Trip_t = Triplet<scalar_t>;
          std::vector<Trip_t> e11, e12, e21;
          A.push_front_elements
            (sep_begin_, sep_end_, this->upd(), e11, e12, e21);
          BLRM_t::construct_and_partial_factor_col
            (F11blr_, F12blr_, F21blr_, F22blr_, sep_tiles_,
             upd_tiles_, admissibility_, blr_opts,
             [&](int i, bool part, std::size_t CP) {
               build_front_cols
                 (A, i, part, CP, e11, e12, e21, task_depth, opts);
             });
        }
      } else {
#if defined(STRUMPACK_USE_GPU)
        if (opts.use_gpu()) {
          using Trip_t = Triplet<scalar_t>;
          std::vector<Trip_t> e11, e12, e21;
          A.push_front_elements
            (sep_begin_, sep_end_, this->upd(), e11, e12, e21);
          std::vector<std::size_t> Il, Ir;
          if (lchild_) Il = lchild_->upd_to_parent(this);
          if (rchild_) Ir = rchild_->upd_to_parent(this);

          std::size_t
            CBl_size = lchild_ ? lchild_->get_device_F22_worksize() : 0,
            CBr_size = rchild_ ? rchild_->get_device_F22_worksize() : 0;
          std::size_t d_mem_bytes =
            gpu::round_up(sizeof(scalar_t)*
                          (dsep*(dsep+2*dupd) + CBl_size + CBr_size)) +
            gpu::round_up(sizeof(Trip_t)*(e11.size()+e12.size()+e21.size())) +
            gpu::round_up(sizeof(std::size_t)*(Il.size()+Ir.size())) +
            gpu::round_up(sizeof(gpu::AssembleData<scalar_t>));
          auto d_mem = workspace.get_device_bytes(d_mem_bytes);
          auto smem = d_mem.template as<scalar_t>();
          gpu::memset<scalar_t>(smem, 0, dsep*(dsep+2*dupd));
          DenseMW_t dF11(dsep, dsep, smem, dsep);  smem += dsep*dsep;
          DenseMW_t dF12(dsep, dupd, smem, dsep);  smem += dsep*dupd;
          DenseMW_t dF21(dupd, dsep, smem, dupd);  smem += dsep*dupd;
          auto CBl = smem;  smem += CBl_size;
          auto CBr = smem;  smem += CBr_size;

          auto de11 = gpu::aligned_ptr<Trip_t>(smem);
          auto de12 = de11 + e11.size();
          auto de21 = de12 + e12.size();
          auto dIl = gpu::aligned_ptr<std::size_t>(de21 + e21.size());
          auto dIr = dIl + Il.size();
          auto dasmbl = gpu::aligned_ptr<gpu::AssembleData<scalar_t>>
            (dIr + Ir.size());

          // TODO replace push_front_elements with set_front_elements
          // to pinned memory, then do a single copy here
          gpu::copy_host_to_device(de11, e11.data(), e11.size());
          gpu::copy_host_to_device(de12, e12.data(), e12.size());
          gpu::copy_host_to_device(de21, e21.data(), e21.size());
          // TODO can combine this to a single copy?
          gpu::copy_host_to_device(dIl, Il.data(), Il.size());
          gpu::copy_host_to_device(dIr, Ir.data(), Ir.size());

          CBdev_ = workspace.get_device_bytes(dupd*dupd*sizeof(scalar_t));
          scalar_t* dCB = CBdev_.template as<scalar_t>();
          gpu::memset<scalar_t>(dCB, 0, dupd*dupd);
          F22_ = DenseMW_t(dupd, dupd, dCB, dupd);

          gpu::AssembleData<scalar_t> hasmbl
            (dsep, dupd, dF11.data(), dF12.data(), dF21.data(), F22_.data(),
             e11.size(), e12.size(), e21.size(), de11, de12, de21);
          if (lchild_)
            hasmbl.set_ext_add_left
              (lchild_->dim_upd(), lchild_->get_device_F22(CBl), dIl);
          if (rchild_)
            hasmbl.set_ext_add_right
              (rchild_->dim_upd(), rchild_->get_device_F22(CBr), dIr);
          gpu::copy_host_to_device(dasmbl, &hasmbl, 1);
          gpu::assemble<scalar_t>(1, &hasmbl, dasmbl);
          if (dsep)
            BLRM_t::construct_and_partial_factor_gpu
              (dF11, dF12, dF21, F22_, F11blr_, F12blr_, F21blr_,
               sep_tiles_, upd_tiles_, admissibility_, workspace,
               opts.BLR_options());
          workspace.restore(d_mem);
        } else
#endif
          {
            DenseM_t F11(dsep, dsep), F12(dsep, dupd), F21(dupd, dsep);
            F11.zero(); F12.zero(); F21.zero();
            A.extract_front
              (F11, F12, F21, sep_begin_, sep_end_, this->upd_, task_depth);
            if (dupd) {
              CBstorage_ = workspace.get(dupd*dupd);
              F22_ = DenseMW_t(dupd, dupd, CBstorage_.data(), dupd);
              F22_.zero();
            }
            if (lchild_)
              lchild_->extend_add_to_dense
                (F11, F12, F21, F22_, this, task_depth);
            if (rchild_)
              rchild_->extend_add_to_dense
                (F11, F12, F21, F22_, this, task_depth);
            if (dsep) {
              auto nF11 = F11.normF();
              auto nF12 = F12.normF();
              auto nF21 = F21.normF();
              auto nF = std::sqrt(nF11*nF11 + nF12*nF12 + nF21*nF21);
              auto lopts = blr_opts;
              lopts.set_abs_tol(lopts.abs_tol() * nF);
              BLRM_t::construct_and_partial_factor
                (F11, F12, F21, F22_, F11blr_, F12blr_, F21blr_,
                 sep_tiles_, upd_tiles_, admissibility_, lopts);
            }
          }
      }
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
         F11blr_, F12blr_, F21blr_, F22blr_,
         sep_tiles_, upd_tiles_, admissibility_, blr_opts);
    }
    if (lchild_) lchild_->release_work_memory(workspace);
    if (rchild_) rchild_->release_work_memory(workspace);
    if (opts.print_compressed_front_stats()) {
      auto time = t.elapsed();
      auto nnz = F11blr_.nonzeros();
      auto rank11 = F11blr_.rank();
      std::cout << "#   - BLR front: Nsep= " << dim_sep()
                << " , Nupd= " << dim_upd()
                << " level= " << etree_level
                << "\n#       " << " nnz(F11)= " << nnz
                << " rank(F11)= " << rank11;
      if (dim_upd()) {
        auto nnz12 = F12blr_.nonzeros();
        auto nnz21 = F21blr_.nonzeros();
        auto nnz22blr = F22blr_.nonzeros();
        auto nnz22dense = F22_.nonzeros();
        nnz += nnz12 + nnz21 + nnz22blr + nnz22dense;
        std::cout << "        nnz(F12)= " << nnz12
                  << " rank(F12)= " << F12blr_.rank()
                  << "\n#       " << " nnz(F21)= " << nnz21
                  << " rank(F21)= " << F21blr_.rank()
                  << "        nnz(F22)= " << nnz22blr << " / " << nnz22dense
                  << " rank(F22)= " << F22blr_.rank();
      }
      std::cout << "\n#        " << (float(nnz)) /
        (float(this->dim_blk())*this->dim_blk()) * 100.
                << " %compression, time= " << time
                << " sec,   factor mem= "
                << nnz *sizeof(scalar_t) / 1.e6 << " MB";
#if defined(STRUMPACK_COUNT_FLOPS)
      ftot = params::flops - f0;
      std::cout << ", flops= " << double(ftot) << std::endl
                << "#        total memory: "
                << double(strumpack::params::memory) / 1.0e6 << " MB"
                << ",   peak memory: "
                << double(strumpack::params::peak_memory) / 1.0e6
                << " MB";
#endif
      std::cout << std::endl;
    }
    // if (etree_level == 0)
    //   BLR::draw(F11blr_, "F11root_"
    //             + std::to_string(opts.BLR_options().leaf_size()) + "_"
    //             + BLR::get_name(opts.BLR_options().admissibility()));
    return err_code;
  }

  template<typename scalar_t,typename integer_t> void
  FrontBLR<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
      bloc.laswp(F11blr_.piv(), true);
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
  FrontBLR<scalar_t,integer_t>::bwd_solve_phase1
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
  FrontBLR<scalar_t,integer_t>::extract_CB_sub_matrix
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
  FrontBLR<scalar_t,integer_t>::node_factor_nonzeros() const {
    return F11blr_.nonzeros() + F12blr_.nonzeros() + F21blr_.nonzeros();
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontBLR<scalar_t,integer_t>::node_subnormals
  (std::size_t& ns, std::size_t& nz) const {
    auto dns = F11blr_.subnormals() + F12blr_.subnormals() + F21blr_.subnormals();
    auto dnz = F11blr_.zeros() + F12blr_.zeros() + F21blr_.zeros();
    // if (dns || dnz)
    //   std::cout << "BLR front ds= " << this->dim_sep()
    //             << " du= " << this->dim_upd()
    //             << " subnormals= " << dns
    //             << " zeros= " << dnz << std::endl;
    ns += dns;
    nz += dnz;
    return ReturnCode::SUCCESS;
  }

  template<typename scalar_t,typename integer_t> void
  FrontBLR<scalar_t,integer_t>::partition
  (const Opts_t& opts, const SpMat_t& A,
   integer_t* sorder, bool is_root, int task_depth) {
    if (dim_sep()) {
      auto g = A.extract_graph
        (opts.separator_ordering_level(), sep_begin_, sep_end_);
#if 1
      auto sep_tree = g.recursive_bisection
        (opts.BLR_options().leaf_size(), 0,
         sorder+sep_begin_, nullptr, 0, 0, dim_sep());
      sep_tiles_ = sep_tree.template leaf_sizes<std::size_t>();
#else
      int K = std::round((1.* dim_sep()) / opts.BLR_options().leaf_size());
      if (K > 1)
        sep_tiles_ = g.partition_K_way
          (K, sorder+sep_begin_, nullptr, 0, 0, dim_sep());
      else {
        sep_tiles_ = {std::size_t(dim_sep())};
        for (integer_t i=sep_begin_; i<sep_end_; i++)
          sorder[i] = i - sep_begin_;
      }
#endif
      std::vector<integer_t> siorder(dim_sep());
      for (integer_t i=sep_begin_; i<sep_end_; i++)
        siorder[sorder[i]] = i - sep_begin_;
      if (opts.BLR_options().admissibility() == BLR::Admissibility::STRONG) {
        g.permute(sorder+sep_begin_, siorder.data());
        admissibility_ = g.admissibility(sep_tiles_);
      } else {
        auto nt = sep_tiles_.size();
        admissibility_ = DenseMatrix<bool>(nt, nt);
        admissibility_.fill(true);
        for (std::size_t t=0; t<nt; t++)
          admissibility_(t, t) = false;
      }
      for (integer_t i=sep_begin_; i<sep_end_; i++)
        sorder[i] += sep_begin_;
    }
    if (dim_upd()) {
      auto leaf = opts.BLR_options().leaf_size();
      auto nt = std::ceil(float(dim_upd()) / leaf);
      upd_tiles_.resize(nt, leaf);
      upd_tiles_.back() = dim_upd() - leaf*(nt-1);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontBLR<scalar_t,integer_t>::draw_node
  (std::ostream& of, bool is_root) const {
    F11blr_.draw(of, this->sep_begin(), this->sep_begin());
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> void
  FrontBLR<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    if (F22blr_.rows() == std::size_t(dim_upd())) {
      auto F22 = F22blr_.dense();
      ExtendAdd<scalar_t,integer_t>::
        extend_add_seq_copy_to_buffers(F22, sbuf, pa, this);
    } else {
#if defined(STRUMPACK_USE_GPU)
      if (CBdev_.size()) {
        DenseM_t F22(dim_upd(), dim_upd());
        gpu::copy_device_to_host(F22, CBdev_.template as<scalar_t>());
        ExtendAdd<scalar_t,integer_t>::
          extend_add_seq_copy_to_buffers(F22, sbuf, pa, this);
      } else
#endif
        ExtendAdd<scalar_t,integer_t>::
          extend_add_seq_copy_to_buffers(F22_, sbuf, pa, this);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontBLR<scalar_t,integer_t>::extadd_blr_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf, const FBLRMPI_t* pa) const {
    if (F22blr_.rows() == std::size_t(dim_upd())) {
      auto F22 = F22blr_.dense();
      BLR::BLRExtendAdd<scalar_t,integer_t>::
        seq_copy_to_buffers(F22, sbuf, pa, this);
    } else {
#if defined(STRUMPACK_USE_GPU)
      if (CBdev_.size()) {
        DenseM_t F22(dim_upd(), dim_upd());
        gpu::copy_device_to_host(F22, CBdev_.template as<scalar_t>());
        BLR::BLRExtendAdd<scalar_t,integer_t>::
          seq_copy_to_buffers(F22, sbuf, pa, this);
      } else
#endif
        BLR::BLRExtendAdd<scalar_t,integer_t>::
          seq_copy_to_buffers(F22_, sbuf, pa, this);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontBLR<scalar_t,integer_t>::extadd_blr_copy_to_buffers_col
  (std::vector<std::vector<scalar_t>>& sbuf, const FBLRMPI_t* pa,
   integer_t begin_col, integer_t end_col, const Opts_t& opts) const {
    // GPU code does not support COLWISE, so will not call this
    if (F22blr_.rows() == std::size_t(dim_upd())) {
      BLR::BLRExtendAdd<scalar_t,integer_t>::
        blrseq_copy_to_buffers_col
        (F22blr_, sbuf, pa, this, begin_col, end_col, opts.BLR_options());
    } else
      BLR::BLRExtendAdd<scalar_t,integer_t>::
        seq_copy_to_buffers_col(F22_, sbuf, pa, this, begin_col, end_col);
  }
#endif

  // explicit template instantiations
  template class FrontBLR<float,int>;
  template class FrontBLR<double,int>;
  template class FrontBLR<std::complex<float>,int>;
  template class FrontBLR<std::complex<double>,int>;

  template class FrontBLR<float,long int>;
  template class FrontBLR<double,long int>;
  template class FrontBLR<std::complex<float>,long int>;
  template class FrontBLR<std::complex<double>,long int>;

  template class FrontBLR<float,long long int>;
  template class FrontBLR<double,long long int>;
  template class FrontBLR<std::complex<float>,long long int>;
  template class FrontBLR<std::complex<double>,long long int>;

} // end namespace strumpack
