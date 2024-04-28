//
// Created by tingxuan on 23-6-19.
//
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
#include <array>
#include <cstring>

#include "FrontGPUSPD.hpp"
#include "FrontalMatrixGPUKernels.hpp"

#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#include "FrontalMatrixMPI.hpp"
#endif

namespace strumpack {

  template<typename scalar_t, typename integer_t> class LevelInfoUnified {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FG_t = FrontGPUSPD<scalar_t,integer_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;

  public:
    LevelInfoUnified() {}

    LevelInfoUnified(const std::vector<F_t*>& fronts, gpu::Handle& handle,
                     int max_streams, const SpMat_t* A) {
      if (!A->symm_sparse()) {
        f.reserve(fronts.size());
        for (auto& F : fronts)
          f.push_back(dynamic_cast<FG_t*>(F));
        std::size_t max_dsep = 0;
        // This pragma causes "internal error: null pointer" on the
        // intel compiler.  It seems to be because the variables are
        // class members.
        // #pragma omp parallel for reduction(+:L_size,U_size,Schur_size,piv_size,total_upd_size,N8,N16,N24,N32,factors_small) reduction(max:max_dsep)
        for (std::size_t i=0; i<f.size(); i++) {
          auto F = f[i];
          const std::size_t dsep = F->dim_sep();
          const std::size_t dupd = F->dim_upd();
          diagonal_size += dsep * dsep;
          off_diagonal_size += 2 * dsep * dupd;
          L_size += dsep*(dsep + dupd);
          U_size += dsep*dupd;
          Schur_size += dupd*dupd;
          piv_size += dsep;
          total_upd_size += dupd;
          if (dsep <= 32) {
            factors_diagonal_small += dsep * dsep;
            factors_off_diagonal_small += 2 * dsep * dupd;
            factors_small += dsep*(dsep + 2*dupd);
            if (dsep <= 8)       N8++;
            else if (dsep <= 16) N16++;
            else if (dsep <= 24) N24++;
            else N32++;
          }
          if (dsep > max_dsep) max_dsep = dsep;
        }
        small_fronts = N8 + N16 + N24 + N32;
        if (small_fronts && small_fronts != f.size())
          std::partition
            (f.begin(), f.end(), [](const FG_t* const& a) -> bool {
              return a->dim_sep() <= 32; });
        {
          auto N = f.size();
          elems11.resize(N+1);
          elems12.resize(N+1);
          elems21.resize(N+1);
          Isize.resize(N+1);
#pragma omp parallel for
          for (std::size_t i=0; i<N; i++) {
            auto& F = *(f[i]);
            A->count_front_elements
              (F.sep_begin(), F.sep_end(), F.upd(),
               elems11[i+1], elems12[i+1], elems21[i+1]);
            if (F.lchild_) Isize[i+1] += F.lchild_->dim_upd();
            if (F.rchild_) Isize[i+1] += F.rchild_->dim_upd();
          }
          for (std::size_t i=0; i<N; i++) {
            elems11[i+1] += elems11[i];
            elems12[i+1] += elems12[i];
            elems21[i+1] += elems21[i];
            Isize[i+1] += Isize[i];
          }
        }
        factor_size = L_size + U_size;
        // TODO(Jie): fix for un-symmetric
        //            getrf_work_size = gpu::getrf_buffersize<scalar_t>(handle, max_dsep);
        getrf_work_size = gpu::potrf_buffersize<scalar_t>(handle, UpLo::L, max_dsep);

        factor_bytes = sizeof(scalar_t) * factor_size;
        factor_bytes = gpu::round_up(factor_bytes);

        work_bytes = sizeof(scalar_t) * (Schur_size + getrf_work_size * max_streams);
        work_bytes = gpu::round_up(work_bytes);
        work_bytes += sizeof(int) * (piv_size + f.size());
        work_bytes = gpu::round_up(work_bytes);
        work_bytes += sizeof(gpu::FrontData<scalar_t>) * (N8 + N16 + N24 + N32);
        work_bytes = gpu::round_up(work_bytes);

        ea_bytes = sizeof(gpu::AssembleData<scalar_t>) * f.size();
        ea_bytes = gpu::round_up(ea_bytes);
        ea_bytes += sizeof(std::size_t) * Isize.back();
        ea_bytes = gpu::round_up(ea_bytes);
        ea_bytes += sizeof(Triplet<scalar_t>) * (elems11.back() + elems12.back() + elems21.back());
        ea_bytes = gpu::round_up(ea_bytes);
      } else {
        f.reserve(fronts.size());
        for (auto& F : fronts)
          f.push_back(dynamic_cast<FG_t*>(F));
        std::size_t max_dsep = 0;
        // This pragma causes "internal error: null pointer" on the
        // intel compiler.  It seems to be because the variables are
        // class members.
        // #pragma omp parallel for reduction(+:L_size,U_size,Schur_size,piv_size,total_upd_size,N8,N16,N24,N32,factors_small) reduction(max:max_dsep)
        for (std::size_t i=0; i<f.size(); i++) {
          auto F = f[i];
          const std::size_t dsep = F->dim_sep();
          const std::size_t dupd = F->dim_upd();
          diagonal_size += dsep * dsep;
          off_diagonal_size += dsep * dupd;
          L_size += dsep*dsep;
          U_size += dsep*dupd;
          Schur_size += dupd*dupd;
          piv_size += dsep;
          total_upd_size += dupd;
          if (dsep <= 32) {
            factors_diagonal_small += dsep * dsep;
            factors_off_diagonal_small += dsep * dupd;
            factors_small += dsep*(dsep + dupd);
            if (dsep <= 8)       N8++;
            else if (dsep <= 16) N16++;
            else if (dsep <= 24) N24++;
            else N32++;
          }
          if (dsep > max_dsep) max_dsep = dsep;
        }
        small_fronts = N8 + N16 + N24 + N32;
        if (small_fronts && small_fronts != f.size())
          std::partition
            (f.begin(), f.end(), [](const FG_t* const& a) -> bool {
              return a->dim_sep() <= 32; });
        {
          auto N = f.size();
          elems11.resize(N+1);
          elems21.resize(N+1);
          Isize.resize(N+1);
#pragma omp parallel for
          for (std::size_t i=0; i<N; i++) {
            auto& F = *(f[i]);
            A->count_front_elements_symmetric
              (F.sep_begin(), F.sep_end(), F.upd(),
               elems11[i+1], elems21[i+1]);
            if (F.lchild_) Isize[i+1] += F.lchild_->dim_upd();
            if (F.rchild_) Isize[i+1] += F.rchild_->dim_upd();
          }
          for (std::size_t i=0; i<N; i++) {
            elems11[i+1] += elems11[i];
            elems21[i+1] += elems21[i];
            Isize[i+1] += Isize[i];
          }
        }
        factor_size = L_size + U_size;
        //            getrf_work_size = gpu::getrf_buffersize<scalar_t>(handle, max_dsep);
        getrf_work_size = gpu::potrf_buffersize<scalar_t>(handle, UpLo::L, max_dsep);

        factor_bytes = sizeof(scalar_t) * factor_size;
        factor_bytes = gpu::round_up(factor_bytes);

        work_bytes = sizeof(scalar_t) * (Schur_size + getrf_work_size * max_streams);
        work_bytes = gpu::round_up(work_bytes);
        work_bytes += sizeof(int) * (piv_size + f.size());
        work_bytes = gpu::round_up(work_bytes);
        work_bytes += sizeof(gpu::FrontData<scalar_t>) * (N8 + N16 + N24 + N32);
        work_bytes = gpu::round_up(work_bytes);

        ea_bytes = sizeof(gpu::AssembleData<scalar_t>) * f.size();
        ea_bytes = gpu::round_up(ea_bytes);
        ea_bytes += sizeof(std::size_t) * Isize.back();
        ea_bytes = gpu::round_up(ea_bytes);
        ea_bytes += sizeof(Triplet<scalar_t>) * (elems11.back() + elems21.back());
        ea_bytes = gpu::round_up(ea_bytes);
      }
    }

    void print_info(int l, int lvls) {
      std::cout << "#  level " << l << " of " << lvls
                << " has " << f.size() << " nodes and "
                << N8 << " <=8, " << N16 << " <=16, "
                << N24 << " <=24, " << N32 << " <=32, needs "
                << factor_bytes / 1.e6
                << " MB for factors, "
                << Schur_size * sizeof(scalar_t) / 1.e6
                << " MB for Schur complements" << std::endl;
    }

    void flops(long long& level_flops, long long& small_flops) {
      level_flops = small_flops = 0;
      auto N = f.size();
#pragma omp parallel for reduction(+: level_flops, small_flops)
      for (std::size_t i=0; i<N; i++) {
        auto F = f[i];
        auto flops = LU_flops(F->F11_) +
          gemm_flops(Trans::N, Trans::N, scalar_t(-1.),
                     F->F21_, F->F12_, scalar_t(1.)) +
          trsm_flops(Side::L, scalar_t(1.), F->F11_, F->F12_) +
          trsm_flops(Side::R, scalar_t(1.), F->F11_, F->F21_);
        level_flops += flops;
        if (F->dim_sep() <= 32)
          small_flops += flops;
      }
    }

    /*
     * first store L factors, then U factors,
     *  F11, F21, F11, F21, ..., F12, F12, ...
     */
    void set_factor_pointers(scalar_t* factors) {
      for (auto F : f) {
        const std::size_t dsep = F->dim_sep();
        const std::size_t dupd = F->dim_upd();
        F->F11_ = DenseMW_t(dsep, dsep, factors, dsep); factors += dsep*dsep;
        F->F12_ = DenseMW_t(dsep, dupd, factors, dsep); factors += dsep*dupd;
        F->F21_ = DenseMW_t(dupd, dsep, factors, dupd); factors += dupd*dsep;
      }
    }

    void set_factor_pointers(scalar_t* factors_diagonal, scalar_t* factors_off_diagonal) {
      for (auto F : f) {
        const std::size_t dsep = F->dim_sep();
        const std::size_t dupd = F->dim_upd();
        F->F11_ = DenseMW_t(dsep, dsep, factors_diagonal, dsep); factors_diagonal += dsep*dsep;
        F->F21_ = DenseMW_t(dupd, dsep, factors_off_diagonal, dupd); factors_off_diagonal += dupd*dsep;
      }
    }

    void set_pivot_pointers(int* pmem) {
      for (auto F : f) {
        F->piv_ = pmem;
        pmem += F->dim_sep();
      }
    }

    void set_work_pointers(void* wmem, int max_streams) {
      auto schur = gpu::aligned_ptr<scalar_t>(wmem);
      for (auto F : f) {
        const int dupd = F->dim_upd();
        if (dupd) {
          F->F22_ = DenseMW_t(dupd, dupd, schur, dupd);
          schur += dupd*dupd;
        }
      }
      dev_getrf_work = schur;
      schur += max_streams * getrf_work_size;
      auto imem = gpu::aligned_ptr<int>(schur);
      for (auto F : f) {
        F->piv_ = imem;
        imem += F->dim_sep();
      }
      dev_getrf_err = imem;   imem += f.size();
      auto fdat = gpu::aligned_ptr<gpu::FrontData<scalar_t>>(imem);
      f8  = fdat;  fdat += N8;
      f16 = fdat;  fdat += N16;
      f24 = fdat;  fdat += N24;
      f32 = fdat;  fdat += N32;
    }

    int align = 0;
    std::vector<FG_t*> f;
    std::size_t diagonal_size = 0, off_diagonal_size = 0, L_size = 0, U_size = 0,
      factors_diagonal_small = 0, factors_off_diagonal_small = 0,
      factor_size = 0, factors_small = 0, Schur_size = 0, piv_size = 0,
      total_upd_size = 0;
    std::size_t N8 = 0, N16 = 0, N24 = 0, N32 = 0, small_fronts = 0;
    std::size_t work_bytes = 0, ea_bytes = 0, factor_bytes = 0;
    std::vector<std::size_t> elems11, elems12, elems21, Isize;
    scalar_t* dev_getrf_work = nullptr;
    int* dev_getrf_err = nullptr;
    int getrf_work_size = 0;
    gpu::FrontData<scalar_t> *f8 = nullptr, *f16 = nullptr,
      *f24 = nullptr, *f32 = nullptr;
  };


  template<typename scalar_t,typename integer_t>
  FrontGPUSPD<scalar_t,integer_t>::FrontGPUSPD
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t>
  FrontGPUSPD<scalar_t,integer_t>::~FrontGPUSPD() {
#if defined(STRUMPACK_COUNT_FLOPS)
    const std::size_t dupd = dim_upd();
    const std::size_t dsep = dim_sep();
    STRUMPACK_SUB_MEMORY(dsep*(dsep+2*dupd)*sizeof(scalar_t));
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontGPUSPD<scalar_t,integer_t>::release_work_memory() {
    F22_.clear();
    host_Schur_.reset(nullptr);
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> void
  FrontGPUSPD<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf,
   const FrontalMatrixMPI<scalar_t,integer_t>* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_seq_copy_to_buffers
      (F22_, sbuf, pa, this);
  }
#endif


  template<typename scalar_t,typename integer_t> void
  FrontGPUSPD<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF21, DenseM_t& paF22,
   const F_t* p, int task_depth) {
    const std::size_t pdsep = paF11.rows();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared) grainsize(64)      \
  if(task_depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t c=0; c<dupd; c++) {
      auto pc = I[c];
      if (pc < pdsep) {
        for (std::size_t r=c; r<upd2sep; r++)
          paF11(I[r],pc) += F22_(r,c);
        for (std::size_t r=std::max(c, upd2sep); r<dupd; r++)
          paF21(I[r]-pdsep,pc) += F22_(r,c);
      } else {
        for (std::size_t r=std::max(c, upd2sep); r<dupd; r++)
          paF22(I[r]-pdsep,pc-pdsep) += F22_(r,c);
      }
    }
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1) * dupd * dupd);
    STRUMPACK_FULL_RANK_FLOPS((is_complex<scalar_t>()?2:1) * dupd * dupd);
    release_work_memory();
  }

  template<typename scalar_t,typename integer_t>
  std::size_t peak_device_memory
  (const std::vector<LevelInfoUnified<scalar_t,integer_t>>& ldata) {
    std::size_t peak_dmem = 0;
    for (std::size_t l=0; l<ldata.size(); l++) {
      auto& L = ldata[l];
      // memory needed on this level: factors,
      // schur updates, pivot vectors, cuSOLVER work space,
      // assembly data (indices, sparse elements)
      std::size_t level_mem = L.factor_bytes + L.work_bytes + L.ea_bytes;
      // the contribution blocks of the previous level are still
      // needed for the extend-add
      if (l+1 < ldata.size())
        level_mem += ldata[l+1].work_bytes;
      peak_dmem = std::max(peak_dmem, level_mem);
    }
    return peak_dmem;
  }

  template<typename scalar_t, typename integer_t> void
  FrontGPUSPD<scalar_t,integer_t>::front_assembly
  (const SpMat_t& A, LInfo_t& L, char* hea_mem, char* dea_mem) {
    using Trip_t = Triplet<scalar_t>;
    auto N = L.f.size();
    auto hasmbl = gpu::aligned_ptr<gpu::AssembleData<scalar_t>>(hea_mem);
    auto Iptr   = gpu::aligned_ptr<std::size_t>(hasmbl + N);
    auto e11    = gpu::aligned_ptr<Trip_t>(Iptr + L.Isize.back());
    auto e21    = e11 + L.elems11.back();
    auto dasmbl = gpu::aligned_ptr<gpu::AssembleData<scalar_t>>(dea_mem);
    auto dIptr  = gpu::aligned_ptr<std::size_t>(dasmbl + N);
    auto de11   = gpu::aligned_ptr<Trip_t>(dIptr + L.Isize.back());
    auto de21   = de11 + L.elems11.back();

#pragma omp parallel for
    for (std::size_t n=0; n<N; n++) {
      auto& f = *(L.f[n]);
      A.set_front_elements_symmetric
        (f.sep_begin_, f.sep_end_, f.upd_,
         e11+L.elems11[n], e21+L.elems21[n]);
      hasmbl[n] = gpu::AssembleData<scalar_t>
        (f.dim_sep(), f.dim_upd(), f.F11_.data(), nullptr,
         f.F21_.data(), f.F22_.data(),
         L.elems11[n+1]-L.elems11[n], 0,
         L.elems21[n+1]-L.elems21[n],
         de11+L.elems11[n], nullptr, de21+L.elems21[n]);
      auto fIptr = Iptr + L.Isize[n];
      auto fdIptr = dIptr + L.Isize[n];
      if (f.lchild_) {
        auto c = dynamic_cast<FG_t*>(f.lchild_.get());
        hasmbl[n].set_ext_add_left(c->dim_upd(), c->F22_.data(), fdIptr);
        c->upd_to_parent(&f, fIptr);
        fIptr += c->dim_upd();
        fdIptr += c->dim_upd();
      }
      if (f.rchild_) {
        auto c = dynamic_cast<FG_t*>(f.rchild_.get());
        hasmbl[n].set_ext_add_right(c->dim_upd(), c->F22_.data(), fdIptr);
        c->upd_to_parent(&f, fIptr);
      }
    }
    gpu::copy_host_to_device<char>(dea_mem, hea_mem, L.ea_bytes);
    gpu::assemble_symmetric(N, hasmbl, dasmbl);
  }

  // TODO(Jie): fix for un-symmetric
  template<typename scalar_t, typename integer_t> long long
  FrontGPUSPD<scalar_t,integer_t>::dense_node_factor_nonzeros() const {
    long long dsep = dim_sep(), dupd = dim_upd();
    return dsep * (dsep + dupd);
  }

  template<typename scalar_t, typename integer_t> void
  FrontGPUSPD<scalar_t,integer_t>::factor_small_fronts
  (LInfo_t& L, gpu::FrontData<scalar_t>* fdata, int* dinfo,
   const SPOptions<scalar_t>& opts) {
    if (!L.small_fronts) return;
    for (std::size_t n=0, n8=0, n16=L.N8, n24=n16+L.N16, n32=n24+L.N24;
         n<L.small_fronts; n++) {
      auto& f = *(L.f[n]);
      const auto dsep = f.dim_sep();
      gpu::FrontData<scalar_t>
        t(dsep, f.dim_upd(), f.F11_.data(), nullptr,
          f.F21_.data(), f.F22_.data(), f.piv_);
      if (dsep <= 8)       fdata[n8++] = t;
      else if (dsep <= 16) fdata[n16++] = t;
      else if (dsep <= 24) fdata[n24++] = t;
      else                 fdata[n32++] = t;
    }
    gpu::copy_host_to_device(L.f8, fdata, L.small_fronts);
    auto replace = opts.replace_tiny_pivots();
    auto thresh = opts.pivot_threshold();
    gpu::factor_symmetric_block_batch<scalar_t,8>(L.N8, L.f8, replace, thresh, dinfo);
    gpu::factor_symmetric_block_batch<scalar_t,16>(L.N16, L.f16, replace, thresh, dinfo+L.N8);
    gpu::factor_symmetric_block_batch<scalar_t,24>(L.N24, L.f24, replace, thresh, dinfo+L.N8+L.N16);
    gpu::factor_symmetric_block_batch<scalar_t,32>(L.N32, L.f32, replace, thresh, dinfo+L.N8+L.N16+L.N24);
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontGPUSPD<scalar_t,integer_t>::split_smaller
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (opts.verbose())
      std::cout << "# Factorization does not fit in GPU memory, "
        "splitting in smaller traversals." << std::endl;
    const std::size_t dupd = dim_upd(), dsep = dim_sep();
    ReturnCode err_code = ReturnCode::SUCCESS;
    if (lchild_) {
      auto el = lchild_->multifrontal_factorization_symmetric
        (A, opts, etree_level+1, task_depth);
      if (el != ReturnCode::SUCCESS) err_code = el;
    }
    if (rchild_) {
      auto er = rchild_->multifrontal_factorization_symmetric
        (A, opts, etree_level+1, task_depth);
      if (er != ReturnCode::SUCCESS) err_code = er;
    }
    STRUMPACK_ADD_MEMORY(dsep*(dsep+dupd)*sizeof(scalar_t));
    STRUMPACK_ADD_MEMORY(dupd*dupd*sizeof(scalar_t));
    host_factors_diagonal_.reset(new scalar_t[dsep*dsep]);
    host_factors_off_diagonal_.reset(new scalar_t[dsep * dupd]);
    host_Schur_.reset(new scalar_t[dupd*dupd]);
    {
      auto fmem_diagonal = host_factors_diagonal_.get();
      auto fmem_off_diagonal = host_factors_off_diagonal_.get();
      F11_ = DenseMW_t(dsep, dsep, fmem_diagonal, dsep);
      F21_ = DenseMW_t(dupd, dsep, fmem_off_diagonal, dupd);
    }
    F22_ = DenseMW_t(dupd, dupd, host_Schur_.get(), dupd);
    F11_.zero();
    F21_.zero(); F22_.zero();
    A.extract_front
      (F11_, F12_, F21_, this->sep_begin_, this->sep_end_,
       this->upd_, task_depth);
    if (lchild_) {
#pragma omp parallel
#pragma omp single
      lchild_->extend_add_to_dense(F11_, F21_, F22_, this, 0);
    }
    if (rchild_) {
#pragma omp parallel
#pragma omp single
      rchild_->extend_add_to_dense(F11_, F21_, F22_, this, 0);
    }
    // TaskTimer tl("");
    // tl.start();
    if (dsep) {
      gpu::Handle sh;
      auto workSize = gpu::potrf_buffersize<scalar_t>(sh, UpLo::L, dsep);
      gpu::DeviceMemory<scalar_t> dm11(dsep*dsep + workSize);
      gpu::DeviceMemory<int> dpiv(dsep+1); // and ierr
      DenseMW_t dF11(dsep, dsep, dm11, dsep);
      gpu::copy_host_to_device(dF11, F11_);
      gpu::potrf(sh, UpLo::L, dF11, dm11 + dsep*dsep, workSize, dpiv+dsep);
      if (opts.replace_tiny_pivots())
        gpu::replace_pivots
          (F11_.rows(), dF11.data(), opts.pivot_threshold());
      int info;
      gpu::copy_device_to_host(&info, dpiv+dsep, 1);
      if (info) err_code = ReturnCode::ZERO_PIVOT;
      pivot_mem_.resize(dsep);
      piv_ = pivot_mem_.data();
      gpu::copy_device_to_host(piv_, dpiv.as<int>(), dsep);
      gpu::copy_device_to_host(F11_, dF11);
      if (dupd) {
        gpu::Handle bh;
        gpu::DeviceMemory<scalar_t> dm21(dsep*dupd);
        DenseMW_t dF21(dupd, dsep, dm21, dupd);
        gpu::copy_host_to_device(dF21, F21_);
        gpu::trsm(bh, Side::R, UpLo::L, Trans::T, Diag::N, scalar_t(1.), dF11, dF21);
        gpu::copy_device_to_host(F21_, dF21);
        dm11.release();
        gpu::DeviceMemory<scalar_t> dm22((dsep+dupd)*dupd);
        DenseMW_t dF22(dupd, dupd, dm22, dupd);
        gpu::copy_host_to_device(dF22, host_Schur_.get());
        gpu::syrk(bh, UpLo::L, Trans::N,
                  scalar_t(-1.), dF21, scalar_t(1.), dF22);
        gpu::copy_device_to_host(host_Schur_.get(), dF22);
      }
    }
    // count flops
    auto level_flops = LU_flops(F11_) +
      gemm_flops(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.)) +
      trsm_flops(Side::L, scalar_t(1.), F11_, F21_);
    (void)level_flops;
    STRUMPACK_FULL_RANK_FLOPS(level_flops);
    // if (opts.verbose()) {
    //   auto level_time = tl.elapsed();
    //   std::cout << "#   GPU Factorization complete, took: "
    //             << level_time << " seconds, "
    //             << level_flops / 1.e9 << " GFLOPS, "
    //             << (float(level_flops) / level_time) / 1.e9
    //             << " GFLOP/s" << std::endl;
    // }
    return err_code;
  }

  template<typename scalar_t,typename integer_t>
  ReturnCode
  FrontGPUSPD<scalar_t, integer_t>::multifrontal_factorization_symmetric
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    ReturnCode err_code = ReturnCode::SUCCESS;
    const int max_streams = opts.gpu_streams();
    std::vector<gpu::Handle> handles(max_streams);
    const int lvls = this->levels();
    std::vector<LInfo_t> ldata(lvls);
    {
      std::vector<std::vector<F_t*>> fp(lvls);
      try {
        this->get_level_fronts_gpu(fp);
      } catch (...) {
        return split_smaller(A, opts, etree_level, task_depth);
      }
      for (int l=lvls-1; l>=0; l--)
        ldata[l] = LInfo_t(fp[l], handles[0], max_streams, &A);
    }
    auto peak_dmem = peak_device_memory(ldata);
    if (peak_dmem >= 0.9 * gpu::available_memory())
      return split_smaller(A, opts, etree_level, task_depth);

    std::vector<gpu::Stream> streams(max_streams);
    gpu::Stream copy_stream;
    for (int i=0; i<max_streams; i++) {
      handles[i].set_stream(streams[i]);
    }
    std::size_t max_small_fronts = 0, max_pinned = 0;
    for (int l=lvls-1; l>=0; l--) {
      auto& L = ldata[l];
      max_small_fronts = std::max(max_small_fronts, L.N8+L.N16+L.N24+L.N32);
      for (auto& f : L.f) {
        const std::size_t dsep = f->dim_sep();
        const std::size_t dupd = f->dim_upd();
        std::size_t fs = dsep*(dsep + dupd);
        max_pinned = std::max(max_pinned, fs);
      }
      max_pinned = std::max(max_pinned, L.factors_small);
    }
    gpu::HostMemory<scalar_t> pinned(2*max_pinned);
    gpu::HostMemory<gpu::FrontData<scalar_t>> fdata(max_small_fronts);
    std::size_t peak_hea_mem = 0;
    for (int l=lvls-1; l>=0; l--)
      peak_hea_mem = std::max(peak_hea_mem, ldata[l].ea_bytes);
    gpu::HostMemory<char> hea_mem(peak_hea_mem);
    gpu::DeviceMemory<char> all_dmem(peak_dmem);
    if (opts.verbose()) {
      std::size_t factor_bytes = 0, ea_bytes = 0, work_bytes = 0;
      for (auto &l: ldata) {
        factor_bytes = std::max(l.factor_bytes, factor_bytes);
        ea_bytes = std::max(l.ea_bytes, ea_bytes);
        work_bytes = std::max(l.work_bytes, work_bytes);
      }

      std::cout << "#   - working space memory (host) = " << double(peak_hea_mem) / 1024 / 1024 << " MB"
                << std::endl;
      printf("#   - working space memory (device) = factor + ea + working = %f MB + %f MB + %f MB = %f MB\n",
             double(factor_bytes) / 1024 / 1024, double(ea_bytes) / 1024 / 1024, double(work_bytes) / 1024 / 1024,
             double(peak_dmem) / 1024 / 1024);
      std::cout << "#   - working space memory (pinned) = "
                << double(2 * max_pinned) * sizeof(scalar_t) / 1024 / 1024 << " MB" << std::endl;
    }
    char* old_work = nullptr;
    for (int l=lvls-1; l>=0; l--) {
      // TaskTimer tl("");
      // tl.start();
      LInfo_t& L = ldata[l];
      // if (opts.verbose()) L.print_info(l, lvls);
      try {
        char *work_mem = nullptr, *dea_mem = nullptr;
        scalar_t* dev_factors = nullptr;
        if (l % 2) {
          work_mem = all_dmem;
          dea_mem = work_mem + L.work_bytes;
          dev_factors = gpu::aligned_ptr<scalar_t>(dea_mem + L.ea_bytes);
        } else {
          work_mem = all_dmem + peak_dmem - L.work_bytes;
          dea_mem = work_mem - L.ea_bytes;
          dev_factors = gpu::aligned_ptr<scalar_t>(dea_mem - L.factor_bytes);
        }
        gpu::memset<scalar_t>(work_mem, 0, L.Schur_size);
        gpu::memset<scalar_t>(dev_factors, 0, L.factor_size);
        //                L.set_factor_pointers(dev_factors);
        L.set_factor_pointers(dev_factors, dev_factors + L.diagonal_size);
        L.set_work_pointers(work_mem, max_streams);
        old_work = work_mem;

        // default stream
        // gpu_check(cudaDeviceSynchronize());
	gpu::synchronize_default_stream();
        front_assembly(A, L, hea_mem, dea_mem);
        gpu::Event e_assemble;
        e_assemble.record();
	gpu::synchronize_default_stream();
	// gpu_check(cudaDeviceSynchronize());

        // default stream
        factor_small_fronts(L, fdata, L.dev_getrf_err, opts);
        gpu::Event e_small;
        e_small.record();

        for (auto& s : streams)
          e_assemble.wait(s);

        // larger fronts in multiple streams.  Copy back in nchunks
        // chunks, but a single chunk cannot be larger than the pinned
        // buffer
        const int nchunks = 16;
        std::size_t Bf = (L.f.size()-L.small_fronts + nchunks - 1) / nchunks;
        std::vector<std::size_t> chunks;
        //                std::vector<std::size_t> factors_chunk;
        std::vector<std::size_t> factors_diagonal_chunk, factors_off_diagonal_chunk;
        for (std::size_t n=L.small_fronts, fc=0, c=0, fdc = 0, fodc = 0; n<L.f.size(); n++) {
          auto& f = *(L.f[n]);
          const std::size_t dsep = f.dim_sep();
          const std::size_t dupd = f.dim_upd();
          std::size_t size_front = dsep * (dsep + dupd);
          std::size_t size_factors_diagonal = dsep * dsep;
          std::size_t size_factors_off_diagonal = dsep * dupd;
          if (c == Bf || fc + size_front > max_pinned) {
            chunks.push_back(c);
            //                        factors_chunk.push_back(fc);
            factors_diagonal_chunk.push_back(fdc);
            factors_off_diagonal_chunk.push_back(fodc);
            c = fc = 0;
            fdc = fodc = 0;
          }
          c++;
          fc += size_front;
          fdc += size_factors_diagonal;
          fodc += size_factors_off_diagonal;
          if (n == L.f.size()-1) { // final chunk
            chunks.push_back(c);
            //                        factors_chunk.push_back(fc);
            factors_diagonal_chunk.push_back(fdc);
            factors_off_diagonal_chunk.push_back(fodc);
          }
        }

        e_small.wait(copy_stream);
        //                gpu::copy_device_to_host_async<scalar_t>
        //                                  (pinned, dev_factors, L.factors_small, copy_stream));
        gpu::copy_device_to_host_async<scalar_t>
          (pinned, dev_factors, L.factors_diagonal_small, copy_stream);
        gpu::copy_device_to_host_async<scalar_t>
          (pinned + L.factors_diagonal_small,
           dev_factors + L.diagonal_size,
           L.factors_off_diagonal_small, copy_stream);

        STRUMPACK_ADD_MEMORY(L.factor_bytes);
        //                L.f[0]->host_factors_.reset(new scalar_t[L.factor_size]);
        //                scalar_t* host_factors = L.f[0]->host_factors_.get();
        L.f[0]->host_factors_diagonal_.reset(new scalar_t[L.diagonal_size]);
        L.f[0]->host_factors_off_diagonal_.reset(new scalar_t[L.off_diagonal_size]);
        scalar_t* host_factors_diagonal = L.f[0]->host_factors_diagonal_.get();
        scalar_t* host_factors_off_diagonal = L.f[0]->host_factors_off_diagonal_.get();
        copy_stream.synchronize();
        //#pragma omp parallel for
        //                for (std::size_t i=0; i<L.factors_small; i++)
        //                    host_factors[i] = pinned[i];
        //                host_factors += L.factors_small;
	std::memcpy(host_factors_diagonal, pinned, L.factors_diagonal_small * sizeof(scalar_t));
	std::memcpy(host_factors_off_diagonal, pinned + L.factors_diagonal_small, L.factors_off_diagonal_small * sizeof(scalar_t));
        host_factors_diagonal += L.factors_diagonal_small;
        host_factors_off_diagonal += L.factors_off_diagonal_small;

        if (!chunks.empty()) {
          scalar_t* pin[2] = {pinned.template as<scalar_t>(),
            pinned.template as<scalar_t>() + max_pinned};
          std::vector<gpu::Event> events(chunks.size());

          for (std::size_t c=0, n=L.small_fronts; c<chunks.size(); c++) {
            int s = c % streams.size(), n0 = n;
#pragma omp parallel
#pragma omp single
            {
              if (c) {
#pragma omp task
                {
                  copy_stream.synchronize();
                  //                                    auto fc = factors_chunk[c-1];
                  //#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
                  //#pragma omp taskloop //num_tasks(omp_get_num_threads()-1)
                  //#endif
                  //                                    for (std::size_t i=0; i<fc; i++)
                  //                                        host_factors[i] = pin[(c-1) % 2][i];
                  //                                    host_factors += fc;
                  auto fdc = factors_diagonal_chunk[c-1];
                  auto fodc = factors_off_diagonal_chunk[c-1];
		  std::memcpy(host_factors_diagonal,
			      pin[(c-1) % 2], fdc * sizeof(scalar_t));
		  std::memcpy(host_factors_off_diagonal,
			      pin[(c-1) % 2] + fdc, fodc * sizeof(scalar_t));
                  host_factors_diagonal += fdc;
                  host_factors_off_diagonal += fodc;
                }
              }
#pragma omp task
              {
                for (std::size_t ci=0; ci<chunks[c]; ci++, n++) {
                  auto& f = *(L.f[n]);
                  gpu::potrf(handles[s],
                             UpLo::L,
                             f.F11_,
                             L.dev_getrf_work + s * L.getrf_work_size,
                             L.getrf_work_size,
                             L.dev_getrf_err + n);
                  if (opts.replace_tiny_pivots())
                    gpu::replace_pivots
                      (f.dim_sep(), f.F11_.data(),
                       opts.pivot_threshold(), &streams[s]);
                  if (f.dim_upd()) {
                    gpu::trsm(handles[s],
                              Side::R, UpLo::L, Trans::T, Diag::N,
                              scalar_t(1.), f.F11_, f.F21_);
                    gpu::syrk(handles[s],
                              UpLo::L, Trans::N,
                              scalar_t(-1.), f.F21_, scalar_t(1.), f.F22_);
                  }
                }
                events[c].record(streams[s]);
                events[c].wait(copy_stream);
                auto& f = *(L.f[n0]);
                //                                gpu::copy_device_to_host_async<scalar_t>
                //                                                  (pin[c % 2], f.F11_.data(),
                //                                                   factors_chunk[c], copy_stream);
                auto fdc = factors_diagonal_chunk[c];
                auto fodc = factors_off_diagonal_chunk[c];
                gpu::copy_device_to_host_async<scalar_t>
                  (pin[c % 2], f.F11_.data(),
                   fdc, copy_stream);
                gpu::copy_device_to_host_async<scalar_t>
                  (pin[c % 2] + fdc, f.F21_.data(),
                   fodc, copy_stream);
              }
            }
          }
          copy_stream.synchronize();
          //                    auto fc = factors_chunk.back();
          //#pragma omp parallel for
          //                    for (std::size_t i=0; i<fc; i++)
          //                        host_factors[i] = pin[(chunks.size()-1) % 2][i];
          auto fdc = factors_diagonal_chunk.back();
          auto fodc = factors_off_diagonal_chunk.back();
	  std::memcpy(host_factors_diagonal, pin[(chunks.size()-1) % 2], fdc * sizeof(scalar_t));
	  std::memcpy(host_factors_off_diagonal, pin[(chunks.size()-1) % 2] + fdc, fodc * sizeof(scalar_t));
        }

        //                L.f[0]->pivot_mem_.resize(L.piv_size);
        copy_stream.synchronize();
        //                gpu::copy_device_to_host
        //                                  (L.f[0]->pivot_mem_.data(), L.f[0]->piv_, L.piv_size);
        //                L.set_factor_pointers(L.f[0]->host_factors_.get();
        L.set_factor_pointers(L.f[0]->host_factors_diagonal_.get(), L.f[0]->host_factors_off_diagonal_.get());
        //                L.set_pivot_pointers(L.f[0]->pivot_mem_.data());

        std::vector<int> getrf_infos(L.f.size());
        gpu::copy_device_to_host
          (getrf_infos.data(), L.dev_getrf_err, L.f.size());
        for (auto ierr : getrf_infos)
          if (ierr) {
            err_code = ReturnCode::ZERO_PIVOT;
            break;
          }
      } catch (const std::bad_alloc& e) {
        std::cerr << "Out of memory" << std::endl;
        abort();
      }
      long long level_flops, small_flops;
      L.flops(level_flops, small_flops);
      STRUMPACK_FULL_RANK_FLOPS(level_flops);
      STRUMPACK_FLOPS(small_flops);
      // if (opts.verbose()) {
      //   auto level_time = tl.elapsed();
      //   std::cout << "#   GPU Factorization complete, took: "
      //             << level_time << " seconds, "
      //             << level_flops / 1.e9 << " GFLOPS, "
      //             << (float(level_flops) / level_time) / 1.e9
      //             << " GFLOP/s" << std::endl;
      // }
    }
    const std::size_t dupd = dim_upd();
    if (dupd) { // get the contribution block from the device
      host_Schur_.reset(new scalar_t[dupd*dupd]);
      gpu::copy_device_to_host
        (host_Schur_.get(),
         reinterpret_cast<scalar_t*>(old_work), dupd*dupd);
      F22_ = DenseMW_t(dupd, dupd, host_Schur_.get(), dupd);
    }
    return err_code;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontGPUSPD<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (!A.symm_sparse()) {
      std::cerr << "The Matrix is not symmetric, please use enable_symmetric in option settings" << std::endl;
      // TODO return something?
      exit(EXIT_FAILURE);
    }
    return multifrontal_factorization_symmetric(A, opts, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontGPUSPD<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
      trsv(UpLo::L, Trans::N, Diag::N, F11_, bloc, task_depth);
      //            F11_.solve_LU_in_place(bloc, piv_, task_depth);
      if (dim_upd()) {
        if (b.cols() == 1)
          gemv(Trans::N, scalar_t(-1.), F21_, bloc,
               scalar_t(1.), bupd, task_depth);
        else
          gemm(Trans::N, Trans::N, scalar_t(-1.), F21_, bloc,
               scalar_t(1.), bupd, task_depth);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontGPUSPD<scalar_t,integer_t>::bwd_solve_phase1
  (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t yloc(dim_sep(), y.cols(), y, this->sep_begin_, 0);
      if (y.cols() == 1) {
        if (dim_upd())
          gemv(Trans::T, scalar_t(-1.), F21_, yupd,
               scalar_t(1.), yloc, task_depth);
      } else {
        if (dim_upd())
          gemm(Trans::T, Trans::N, scalar_t(-1.), F21_, yupd,
               scalar_t(1.), yloc, task_depth);
      }
      trsv(UpLo::L, Trans::T, Diag::N, F11_, yloc, params::task_recursion_cutoff_level);
    }
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontGPUSPD<scalar_t,integer_t>::node_inertia
  (integer_t& neg, integer_t& zero, integer_t& pos) const {
    using real_t = typename RealType<scalar_t>::value_type;
    for (std::size_t i=0; i<F11_.rows(); i++) {
      if (piv_[i] != int(i+1)) return ReturnCode::INACCURATE_INERTIA;
      auto absFii = std::abs(F11_(i, i));
      if (absFii > real_t(0.)) pos++;
      else if (absFii < real_t(0.)) neg++;
      else if (absFii == real_t(0.)) zero++;
      else std::cerr << "F(" << i << "," << i << ")=" << F11_(i,i) << std::endl;
    }
    return ReturnCode::SUCCESS;
  }

  template<typename scalar_t>
  class MatrixWrapperForSparseF11 : public DenseMatrix<scalar_t> {
  public:
    /**
     * Default constructor. Creates an empty, 0x0 matrix.
     */
    MatrixWrapperForSparseF11() : DenseMatrix<scalar_t>() {}
    struct {
      size_t rowTotal, colTotal, nnz;
      scalar_t *value{nullptr};
      size_t *innerIndex{nullptr}, *outerIndex{nullptr};
    } sparseF11;

    /**
     * Constructor. Create an m x n matrix wrapper using already
     * allocated memory, pointed to by D, with leading dimension ld.
     *
     * \param m number of rows of the new (sub) matrix
     * \param n number of columns of the new matrix
     * \param D pointer to memory representing matrix, this should
     * point to at least ld*n bytes of allocated memory
     * \param ld leading dimension of matrix allocated at D. ld >= m
     */
    MatrixWrapperForSparseF11(std::size_t m, std::size_t n,
                              scalar_t* D, std::size_t ld) {
      this->data_ = D; this->rows_ = m; this->cols_ = n;
      this->ld_ = std::max(std::size_t(1), ld);
    }

    /**
     * Constructor. Create a DenseMatrixWrapper as a submatrix of size
     * m x n, of a DenseMatrix (or DenseMatrixWrapper) D, at position
     * i,j in D. The constructed DenseMatrixWrapper will be the
     * submatrix D(i:i+m,j:j+n).
     *
     * \param m number of rows of the new (sub) matrix
     * \param n number of columns of the new matrix
     * \param D matrix from which to take a submatrix
     * \param i row offset in D of the top left corner of the submatrix
     * \param j columns offset in D of the top left corner of the
     * submatrix
     */
    MatrixWrapperForSparseF11(std::size_t m, std::size_t n, DenseMatrix<scalar_t>& D,
                              std::size_t i, std::size_t j)
      : MatrixWrapperForSparseF11<scalar_t>(m, n, &D(i, j), D.ld()) {
      assert(i+m <= D.rows());
      assert(j+n <= D.cols());
    }

    /**
     * Virtual destructor. Since a DenseMatrixWrapper does not
     * actually own it's memory, put just keeps a pointer, this will
     * not free any memory.
     */
    virtual ~MatrixWrapperForSparseF11() { this->data_ = nullptr; }

    /**
     * Clear the MatrixWrapperForSparseF11. Ie, set to an empty matrix. This
     * will not affect the original matrix, to which this is a
     * wrapper, only the wrapper itself is reset. No memory is
     * released.
     */
    void clear() override {
      this->rows_ = 0; this->cols_ = 0;
      this->ld_ = 1; this->data_ = nullptr;
    }

    /**
     * Return the amount of memory taken by this wrapper, ie,
     * 0. (since the wrapper itself does not own the memory). The
     * memory will likely be owned by a DenseMatrix, while this
     * MatrixWrapperForSparseF11 is just a submatrix of that existing
     * matrix. Returning 0 here avoids counting memory double.
     *
     * \see nonzeros
     */
    std::size_t memory() const override { return 0; }

    /**
     * Return the number of nonzeros taken by this wrapper, ie,
     * 0. (since the wrapper itself does not own the memory). The
     * memory will likely be owned by a DenseMatrix, while this
     * MatrixWrapperForSparseF11 is just a submatrix of that existing
     * matrix. Returning 0 here avoids counting nonzeros double.
     *
     * \see memory
     */
    std::size_t nonzeros() const override { return 0; }

    /**
     * Default copy constructor, from another DenseMatrixWrapper.
     */
    MatrixWrapperForSparseF11(const MatrixWrapperForSparseF11<scalar_t>&) = default;

    /**
     * Constructing a MatrixWrapperForSparseF11 from a MatrixWrapperForSparseF11 is
     * not allowed.
     * TODO Why not??!! just delegate to MatrixWrapperForSparseF11(m, n, D, i, j)??
     */
    MatrixWrapperForSparseF11(const DenseMatrix<scalar_t>&) = delete;

    /**
     * Default move constructor.
     */
    MatrixWrapperForSparseF11(MatrixWrapperForSparseF11<scalar_t>&&) = default;

    /**
     * Moving from a DenseMatrix is not allowed.
     */
    MatrixWrapperForSparseF11(DenseMatrix<scalar_t>&&) = delete;

    // /**
    //  * Assignment operator. Shallow copy only. This only copies the
    //  * wrapper object. Does not copy matrix elements.
    //  *
    //  * \param D matrix wrapper to copy from, this will be duplicated
    //  */
    // MatrixWrapperForSparseF11<scalar_t>&
    // operator=(const MatrixWrapperForSparseF11<scalar_t>& D) {
    //   this->data_ = D.data();
    //   this->rows_ = D.rows();
    //   this->cols_ = D.cols();
    //   this->ld_ = D.ld();
    //   return *this;
    // }

    /**
     * Move assignment. This moves only the wrapper.
     *
     * \param D matrix wrapper to move from. This will not be
     * modified.
     */
    MatrixWrapperForSparseF11<scalar_t>&
    operator=(MatrixWrapperForSparseF11<scalar_t>&& D) {
      this->data_ = D.data(); this->rows_ = D.rows();
      this->cols_ = D.cols(); this->ld_ = D.ld(); return *this; }

    /**
     * Assignment operator, from a DenseMatrix. Assign the memory of
     * the DenseMatrix to the matrix wrapped by this
     * MatrixWrapperForSparseF11 object.
     *
     * \param a matrix to copy from, should be a.rows() ==
     * this->rows() and a.cols() == this->cols()
     */
    MatrixWrapperForSparseF11<scalar_t>&
    operator=(const DenseMatrix<scalar_t>& a) override {
      assert(a.rows()==this->rows() && a.cols()==this->cols());
      for (std::size_t j=0; j<this->cols(); j++)
        for (std::size_t i=0; i<this->rows(); i++)
          this->operator()(i, j) = a(i, j);
      return *this;
    }
  };

  // explicit template instantiations
  template class FrontGPUSPD<float,int>;
  template class FrontGPUSPD<double,int>;
  template class FrontGPUSPD<std::complex<float>,int>;
  template class FrontGPUSPD<std::complex<double>,int>;

  template class FrontGPUSPD<float,long int>;
  template class FrontGPUSPD<double,long int>;
  template class FrontGPUSPD<std::complex<float>,long int>;
  template class FrontGPUSPD<std::complex<double>,long int>;

  template class FrontGPUSPD<float,long long int>;
  template class FrontGPUSPD<double,long long int>;
  template class FrontGPUSPD<std::complex<float>,long long int>;
  template class FrontGPUSPD<std::complex<double>,long long int>;

} // end namespace strumpack
