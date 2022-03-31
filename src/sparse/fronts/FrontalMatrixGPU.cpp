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
#include <fstream>

#include "FrontalMatrixGPU.hpp"
#include "FrontalMatrixGPUKernels.hpp"

#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#include "FrontalMatrixMPI.hpp"
#endif
#if defined(STRUMPACK_USE_KBLAS)
#include "kblas_operators.h"
#include "batch_getrf.h"
#endif


// #define VBATCHED_STATS


namespace strumpack {

  template<typename scalar_t> class GPUFactorsImpl {
  public:
    GPUFactorsImpl(int lvls) : dL_lvl(lvls), dU_lvl(lvls), dP_lvl(lvls) {}

    gpu::DeviceMemory<scalar_t> dL, dU;
    gpu::DeviceMemory<int> dP;
    std::vector<scalar_t*> dL_lvl, dU_lvl;
    std::vector<int*> dP_lvl;
  };

  template<typename scalar_t>
  GPUFactors<scalar_t>::GPUFactors(int lvls)
    : data_(new GPUFactorsImpl<scalar_t>(lvls)) {}
  template<typename scalar_t>
  GPUFactors<scalar_t>::~GPUFactors() = default;

  // explicit template specialization
  template class GPUFactors<float>;
  template class GPUFactors<double>;
  template class GPUFactors<std::complex<float>>;
  template class GPUFactors<std::complex<double>>;


  uintptr_t round_to_16(uintptr_t p) { return (p + 15) & ~15; }
  uintptr_t round_to_16(void* p) {
    return round_to_16(reinterpret_cast<uintptr_t>(p));
  }


  template<typename scalar_t, typename integer_t> class LevelInfo {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FG_t = FrontalMatrixGPU<scalar_t,integer_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
  public:
    LevelInfo() {}
    LevelInfo(const std::vector<F_t*>& fronts,
              gpu::SOLVERHandle& handle,
#if defined(STRUMPACK_USE_KBLAS)
              kblasHandle_t& kblas_handle,
#endif
              const SpMat_t* A=nullptr) {
      auto N = fronts.size();
      f.reserve(N);
      std::size_t max_dsep = 0;
      for (auto& F : fronts)
        f.push_back(dynamic_cast<FG_t*>(F));
#pragma omp parallel for                                                \
  reduction(max:max_dsep)                                               \
  reduction(+:factor_size,Schur_size,piv_size)                          \
  reduction(+:total_upd_size,Nsmall)
      for (std::size_t i=0; i<N; i++) {
        auto& F = *(f[i]);
        const std::size_t dsep = F.dim_sep();
        const std::size_t dupd = F.dim_upd();
        factor_size += dsep*(dsep + 2*dupd);
        Schur_size += dupd*dupd;
        piv_size += dsep;
        total_upd_size += dupd;
        if (dsep <= FRONT_SMALL) Nsmall++;
        if (dsep > max_dsep) max_dsep = dsep;
      }
      if (Nsmall && Nsmall != N)
        std::partition
          (f.begin(), f.end(), [](const FG_t* const& a) -> bool {
            return a->dim_sep() <= FRONT_SMALL; });
      if (A) {
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
      d1_batch.resize(Nsmall);
      d2_batch.resize(Nsmall);
      ld1_batch.resize(Nsmall);
      ld2_batch.resize(Nsmall);
      F_batch.resize(4*Nsmall);
      ipiv_batch.resize(Nsmall);
#pragma omp parallel for                        \
  reduction(max:max_d1_small, max_d2_small)
      for (std::size_t i=0; i<Nsmall; i++) {
        auto& F = *(f[i]);
        const int dsep = F.dim_sep();
        const int dupd = F.dim_upd();
        d1_batch[i] = dsep;
        d2_batch[i] = dupd;
        max_d1_small = std::max(max_d1_small, dsep);
        max_d2_small = std::max(max_d2_small, dupd);
        ld1_batch[i] = std::max(1, dsep);
        ld2_batch[i] = std::max(1, dupd);
      }

      std::size_t getrf_work_cusolver =
        gpu::getrf_buffersize<scalar_t>(handle, max_dsep);
#if defined(STRUMPACK_USE_KBLAS)
      typedef typename kblas_base_type<scalar_t>::device_type kblas_type;
      getrf_work_bytes = kblas_getrf_vbatch_workspace_host<kblas_type>
        (kblas_handle, d1_batch.data(), d1_batch.data(), Nsmall);
#else
      getrf_work_bytes = 0;
#endif
      getrf_work_bytes =
        std::max(getrf_work_bytes, sizeof(scalar_t) * getrf_work_cusolver);

      work_bytes =
        round_to_16(sizeof(scalar_t) * Schur_size) +
        round_to_16(getrf_work_bytes) +
        // 2*piv_size for magma_sgetrf_vbatched_max_nocheck dpivinfo_array
        round_to_16(sizeof(int) * (piv_size + max_d1_small*Nsmall + N + 4 * (Nsmall+1))) +
        round_to_16(sizeof(scalar_t*) * 4 * Nsmall) +
        // x2 for magma_sgetrf_vbatched_max_nocheck dpivinfo_array
        round_to_16(sizeof(int*) * 2*Nsmall);
      ea_bytes =
        round_to_16(sizeof(gpu::AssembleData<scalar_t>) * f.size()) +
        round_to_16(sizeof(std::size_t) * Isize.back()) +
        round_to_16(sizeof(Triplet<scalar_t>) *
                    (elems11.back() + elems12.back() + elems21.back()));
    }

    void print_info(int l, int lvls) {
      std::cout << "#  level " << l << " of " << lvls
                << " has " << f.size() << " nodes and "
                << Nsmall << " small fronts, needs "
                << factor_size * sizeof(scalar_t) / 1.e6
                << " MB for factors, "
                << Schur_size * sizeof(scalar_t) / 1.e6
                << " MB for Schur complements" << std::endl;
    }

    void flops(long long& level_flops, long long& small_flops) {
      level_flops = small_flops = 0;
#pragma omp parallel for reduction(+: level_flops, small_flops)
      for (std::size_t i=0; i<f.size(); i++) {
        auto F = f[i];
        auto flops = LU_flops(F->F11_) +
          gemm_flops(Trans::N, Trans::N, scalar_t(-1.),
                     F->F21_, F->F12_, scalar_t(1.)) +
          trsm_flops(Side::L, scalar_t(1.), F->F11_, F->F12_) +
          trsm_flops(Side::R, scalar_t(1.), F->F11_, F->F21_);
        level_flops += flops;
        if (F->dim_sep() <= FRONT_SMALL)
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

    void set_pivot_pointers(int* pmem) {
      for (auto F : f) {
        F->piv_ = pmem;
        pmem += F->dim_sep();
      }
    }

    void set_work_pointers(void* wmem) {
      auto schur = reinterpret_cast<scalar_t*>(wmem);
      for (auto F : f) {
        const int dupd = F->dim_upd();
        if (dupd) {
          F->F22_ = DenseMW_t(dupd, dupd, schur, dupd);
          schur += dupd*dupd;
        }
      }
      auto gmem = reinterpret_cast<char*>(round_to_16(schur));
      dev_getrf_work = gmem;   gmem += getrf_work_bytes;
      auto imem = reinterpret_cast<int*>(round_to_16(gmem));
      for (auto F : f) {
        F->piv_ = imem;
        imem += F->dim_sep();
      }
      auto max_d1_small = d1_batch.empty() ? 0 :
        *std::max_element(d1_batch.begin(), d1_batch.end());
      dpivinfo = imem;         imem += max_d1_small*Nsmall;
      auto N = f.size();
      dev_getrf_err = imem;    imem += N;
      dev_d1_batch  = imem;    imem += Nsmall+1;
      dev_d2_batch  = imem;    imem += Nsmall+1;
      dev_ld1_batch = imem;    imem += Nsmall+1;
      dev_ld2_batch = imem;    imem += Nsmall+1;
      auto fmem = reinterpret_cast<scalar_t**>(round_to_16(imem));
      dev_F_batch = fmem;      fmem += 4 * Nsmall;
      auto ipmem = reinterpret_cast<int**>(round_to_16(fmem));
      dev_ipiv_batch = ipmem;  ipmem += Nsmall;
      dpivinfo_array = ipmem;  ipmem += Nsmall;
    }

    static const int FRONT_SMALL = 1024*1024;
    std::vector<FG_t*> f;
    std::size_t factor_size = 0, Schur_size = 0, piv_size = 0,
      total_upd_size = 0, work_bytes = 0, ea_bytes = 0,
      Nsmall = 0;
    std::vector<std::size_t> elems11, elems12, elems21, Isize;

    char* dev_getrf_work = nullptr;
    std::size_t getrf_work_bytes = 0;
    int* dev_getrf_err = nullptr;

    // meta-data for batched call(s)
    std::vector<int> d1_batch, d2_batch, ld1_batch, ld2_batch;
    int max_d1_small = 0, max_d2_small = 0;
    std::vector<scalar_t*> F_batch;
    std::vector<int*> ipiv_batch;
    int *dev_d1_batch = nullptr, *dev_d2_batch = nullptr,
      *dev_ld1_batch = nullptr, *dev_ld2_batch = nullptr,
      *dpivinfo = nullptr;
    scalar_t **dev_F_batch = nullptr;
    int **dev_ipiv_batch = nullptr, **dpivinfo_array = nullptr;
  };


  template<typename scalar_t,typename integer_t>
  FrontalMatrixGPU<scalar_t,integer_t>::FrontalMatrixGPU
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t>
  FrontalMatrixGPU<scalar_t,integer_t>::~FrontalMatrixGPU() {
#if defined(STRUMPACK_COUNT_FLOPS)
    const std::size_t dupd = dim_upd();
    const std::size_t dsep = dim_sep();
    STRUMPACK_SUB_MEMORY(dsep*(dsep+2*dupd)*sizeof(scalar_t));
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::release_work_memory() {
    F22_.clear();
    host_Schur_.release();
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf,
   const FrontalMatrixMPI<scalar_t,integer_t>* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_seq_copy_to_buffers
      (F22_, sbuf, pa, this);
  }
#endif

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
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

  template<typename scalar_t,typename integer_t>
  std::size_t peak_device_memory
  (const std::vector<LevelInfo<scalar_t,integer_t>>& ldata) {
    std::size_t peak_dmem = 0;
    for (std::size_t l=0; l<ldata.size(); l++) {
      auto& L = ldata[l];
      // memory needed on this level: factors,
      // schur updates, pivot vectors, cuSOLVER work space,
      // assembly data (indices, sparse elements)
      std::size_t level_mem =
        round_to_16(L.factor_size*sizeof(scalar_t))
        + L.work_bytes + L.ea_bytes;
      // the contribution blocks of the previous level are still
      // needed for the extend-add
      if (l+1 < ldata.size())
        level_mem += ldata[l+1].work_bytes;
      peak_dmem = std::max(peak_dmem, level_mem);
    }
    return peak_dmem;
  }

  template<typename scalar_t, typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::front_assembly
  (const SpMat_t& A, LInfo_t& L, char* hea_mem, char* dea_mem) {
    using Trip_t = Triplet<scalar_t>;
    auto N = L.f.size();
    auto hasmbl = reinterpret_cast<gpu::AssembleData<scalar_t>*>(hea_mem);
    auto Iptr = reinterpret_cast<std::size_t*>(round_to_16(hasmbl + N));
    auto e11 = reinterpret_cast<Trip_t*>(round_to_16(Iptr + L.Isize.back()));
    auto e12 = e11 + L.elems11.back();
    auto e21 = e12 + L.elems12.back();
    auto dasmbl = reinterpret_cast<gpu::AssembleData<scalar_t>*>(dea_mem);
    auto dIptr = reinterpret_cast<std::size_t*>(round_to_16(dasmbl + N));
    auto de11 = reinterpret_cast<Trip_t*>(round_to_16(dIptr + L.Isize.back()));
    auto de12 = de11 + L.elems11.back();
    auto de21 = de12 + L.elems12.back();
#pragma omp parallel for
    for (std::size_t n=0; n<N; n++) {
      auto& f = *(L.f[n]);
      A.set_front_elements
        (f.sep_begin_, f.sep_end_, f.upd_,
         e11+L.elems11[n], e12+L.elems12[n], e21+L.elems21[n]);
      hasmbl[n] = gpu::AssembleData<scalar_t>
        (f.dim_sep(), f.dim_upd(), f.F11_.data(), f.F12_.data(),
         f.F21_.data(), f.F22_.data(),
         L.elems11[n+1]-L.elems11[n], L.elems12[n+1]-L.elems12[n],
         L.elems21[n+1]-L.elems21[n],
         de11+L.elems11[n], de12+L.elems12[n], de21+L.elems21[n]);
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
    gpu::assemble(N, hasmbl, dasmbl);
  }


  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::split_smaller
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (opts.verbose())
      std::cout << "# Factorization does not fit in GPU memory, "
        "splitting in smaller traversals." << std::endl;
    const std::size_t dupd = dim_upd(), dsep = dim_sep();
    if (lchild_)
      lchild_->multifrontal_factorization(A, opts, etree_level+1, task_depth);
    if (rchild_)
      rchild_->multifrontal_factorization(A, opts, etree_level+1, task_depth);

    STRUMPACK_ADD_MEMORY(dsep*(dsep+2*dupd)*sizeof(scalar_t));
    STRUMPACK_ADD_MEMORY(dupd*dupd*sizeof(scalar_t));
    host_factors_.reset(new scalar_t[dsep*(dsep+2*dupd)]);
    host_Schur_.reset(new scalar_t[dupd*dupd]);
    {
      auto fmem = host_factors_.get();
      F11_ = DenseMW_t(dsep, dsep, fmem, dsep); fmem += dsep*dsep;
      F12_ = DenseMW_t(dsep, dupd, fmem, dsep); fmem += dsep*dupd;
      F21_ = DenseMW_t(dupd, dsep, fmem, dupd);
    }
    F22_ = DenseMW_t(dupd, dupd, host_Schur_.get(), dupd);
    F11_.zero(); F12_.zero();
    F21_.zero(); F22_.zero();
    A.extract_front
      (F11_, F12_, F21_, this->sep_begin_, this->sep_end_,
       this->upd_, task_depth);
    if (lchild_) {
#pragma omp parallel
#pragma omp single
      lchild_->extend_add_to_dense(F11_, F12_, F21_, F22_, this, 0);
    }
    if (rchild_) {
#pragma omp parallel
#pragma omp single
      rchild_->extend_add_to_dense(F11_, F12_, F21_, F22_, this, 0);
    }
    TaskTimer tl("");
    tl.start();
    if (dsep) {
      gpu::SOLVERHandle sh;
      gpu::DeviceMemory<scalar_t> dm11
        (dsep*dsep + gpu::getrf_buffersize<scalar_t>(sh, dsep));
      gpu::DeviceMemory<int> dpiv(dsep+1); // and ierr
      DenseMW_t dF11(dsep, dsep, dm11, dsep);
      gpu::copy_host_to_device(dF11, F11_);
      gpu::getrf(sh, dF11, dm11 + dsep*dsep, dpiv, dpiv+dsep);
      pivot_mem_.resize(dsep);
      piv_ = pivot_mem_.data();
      gpu::copy_device_to_host(piv_, dpiv.as<int>(), dsep);
      gpu::copy_device_to_host(F11_, dF11);
      if (opts.replace_tiny_pivots()) {
        auto thresh = opts.pivot_threshold();
        for (std::size_t i=0; i<F11_.rows(); i++)
          if (std::abs(F11_(i,i)) < thresh)
            F11_(i,i) = (std::real(F11_(i,i)) < 0) ? -thresh : thresh;
      }
      if (dupd) {
        gpu::DeviceMemory<scalar_t> dm12(dsep*dupd);
        DenseMW_t dF12(dsep, dupd, dm12, dsep);
        gpu::copy_host_to_device(dF12, F12_);
        gpu::getrs(sh, Trans::N, dF11, dpiv, dF12, dpiv+dsep);
        gpu::copy_device_to_host(F12_, dF12);
        dm11.release();
        gpu::DeviceMemory<scalar_t> dm2122((dsep+dupd)*dupd);
        DenseMW_t dF21(dupd, dsep, dm2122, dupd);
        DenseMW_t dF22(dupd, dupd, dm2122+(dsep*dupd), dupd);
        gpu::copy_host_to_device(dF21, F21_);
        gpu::copy_host_to_device(dF22, host_Schur_.get());
        gpu::BLASHandle bh;
        gpu::gemm(bh, Trans::N, Trans::N, scalar_t(-1.),
                  dF21, dF12, scalar_t(1.), dF22);
        gpu::copy_device_to_host(host_Schur_.get(), dF22);
      }
    }
    // count flops
    auto level_flops = LU_flops(F11_) +
      gemm_flops(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.)) +
      trsm_flops(Side::L, scalar_t(1.), F11_, F12_) +
      trsm_flops(Side::R, scalar_t(1.), F11_, F21_);
    STRUMPACK_FULL_RANK_FLOPS(level_flops);
    if (opts.verbose()) {
      auto level_time = tl.elapsed();
      std::cout << "#   GPU Factorization complete, took: "
                << level_time << " seconds, "
                << level_flops / 1.e9 << " GFLOPS, "
                << (float(level_flops) / level_time) / 1.e9
                << " GFLOP/s" << std::endl;
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
#if defined(STRUMPACK_USE_KBLAS)
    kblasHandle_t kblas_handle;
    // TODO need to free this handle!!
    kblasCreate(&kblas_handle);
#endif

    gpu::SOLVERHandle solver_handle;
    const int lvls = this->levels();
    std::vector<LInfo_t> ldata(lvls);
    for (int l=lvls-1; l>=0; l--) {
      std::vector<F_t*> fp;
      this->get_level_fronts(fp, l);
      auto& L = ldata[l];
      L = LInfo_t(fp, solver_handle,
#if defined(STRUMPACK_USE_KBLAS)
                  kblas_handle,
#endif
                  &A);
    }
    auto peak_dmem = peak_device_memory(ldata);
    if (peak_dmem >= 0.9 * gpu::available_memory()) {
      split_smaller(A, opts, etree_level, task_depth);
      return;
    }
    gpu::Stream comp_stream, copy_stream;
    gpu::BLASHandle blas_handle;
    blas_handle.set_stream(comp_stream);
    solver_handle.set_stream(comp_stream);

    magma_init();
    magma_queue_t magma_q;

#if defined(STRUMPACK_USE_CUDA)
    magma_queue_create_from_cuda
      (0, comp_stream, blas_handle, NULL, &magma_q);
#elif defined(STRUMPACK_USE_HIP)
    magma_queue_create_from_hip
      (0, comp_stream, blas_handle, NULL, &magma_q);
    // magma_queue_create(0, &magma_q);
#endif

#if defined(STRUMPACK_USE_KBLAS)
    kblasSetStream(kblas_handle, comp_stream);
    kblasEnableMagma(kblas_handle);
    kblas_gemm_batch_nonuniform_wsquery(kblas_handle);
    kblasAllocateWorkspace(kblas_handle);
#endif

    std::size_t pinned_size = 0;
    for (int l=lvls-1; l>=0; l--)
      pinned_size = std::max(pinned_size, ldata[l].factor_size);
    gpu::HostMemory<scalar_t> pinned(pinned_size);

    std::size_t peak_hea_mem = 0;
    for (int l=lvls-1; l>=0; l--)
      peak_hea_mem = std::max(peak_hea_mem, ldata[l].ea_bytes);
    gpu::HostMemory<char> hea_mem(peak_hea_mem);
    gpu::DeviceMemory<char> all_dmem(peak_dmem);
    char* old_work = nullptr;

#if defined(VBATCHED_STATS)
    TaskTimer t_getrf_small("getrf_vbatched"),
      t_getrf_large("getrf_large"),
      t_laswp("laswp_vbatched"),
      t_getrs("getrs_large"),
      t_trsm_L("trsm_L_vbatched"),
      t_trsm_U("trsm_U_vbatched"),
      t_gemm_small("gemm_small"),
      t_gemm("gemm");
    {
      std::ofstream fls("level_sizes.log");
      for (int l=lvls-1; l>=0; l--) {
        auto& L = ldata[l];
        for (std::size_t i=0; i<L.f.size(); i++) {
          auto& F = *(L.f[i]);
          fls << F.dim_sep() << " " << F.dim_upd() << std::endl;
        }
        fls << std::endl << std::endl;
      }

      std::ofstream ld1("ld1.log"), ld2("ld2.log"), lmax("lmax.log");
      for (int l=lvls-1; l>=0; l--) {
        auto& L = ldata[l];
        std::vector<integer_t> d1sizes, d2sizes;
        auto N = L.f.size();
        for (std::size_t i=0; i<N; i++) {
          auto& F = *(L.f[i]);
          if (F.dim_sep() || F.dim_upd()) {
            d1sizes.push_back(F.dim_sep());
            d2sizes.push_back(F.dim_upd());
          }
        }
        std::sort(d1sizes.begin(), d1sizes.end());
        std::sort(d2sizes.begin(), d2sizes.end());
        auto N1 = d1sizes.size();
        auto N2 = d2sizes.size();
        ld1 << std::max(d1sizes[0], integer_t(1)) << " " << d1sizes[N1/4] << " " << d1sizes[N1/2] << " " << d1sizes[3*N1/4] << " " << d1sizes[N1-1];
        ld2 << std::max(d2sizes[0], integer_t(1)) << " " << d2sizes[N2/4] << " " << d2sizes[N2/2] << " " << d2sizes[3*N2/4] << " " << d2sizes[N2-1];
        lmax << N1;
        ld1 << std::endl;
        ld2 << std::endl;
        lmax << std::endl;
      }
    }
#endif

    for (int l=lvls-1; l>=0; l--) {
      TaskTimer tl("");
      tl.start();
      auto& L = ldata[l];
      if (opts.verbose()) L.print_info(l, lvls);
      try {
        char *work_mem = nullptr, *dea_mem = nullptr;
        scalar_t* dev_factors = nullptr;
        if (l % 2) {
          work_mem = all_dmem;
          dea_mem = work_mem + L.work_bytes;
          dev_factors = reinterpret_cast<scalar_t*>(dea_mem + L.ea_bytes);
        } else {
          work_mem = all_dmem + peak_dmem - L.work_bytes;
          dea_mem = work_mem - L.ea_bytes;
          dev_factors = reinterpret_cast<scalar_t*>
            (dea_mem - round_to_16(L.factor_size * sizeof(scalar_t)));
        }
        gpu::memset<scalar_t>(work_mem, 0, L.Schur_size);
        gpu::memset<scalar_t>(dev_factors, 0, L.factor_size);
        L.set_factor_pointers(dev_factors);
        L.set_work_pointers(work_mem);
        old_work = work_mem;

        // default stream
        front_assembly(A, L, hea_mem, dea_mem);

        std::size_t N = L.f.size();
        std::size_t Nsmall = L.Nsmall;
#pragma omp parallel for
        for (std::size_t i=0; i<Nsmall; i++) {
          auto& f = *(L.f[i]);
          L.F_batch[i           ] = f.F11_.data();
          L.F_batch[i +   Nsmall] = f.F12_.data();
          L.F_batch[i + 2*Nsmall] = f.F21_.data();
          L.F_batch[i + 3*Nsmall] = f.F22_.data();
          L.ipiv_batch[i]    = f.piv_;
        }
        auto d1 = L.dev_d1_batch;   auto d2 = L.dev_d2_batch;
        auto ld1 = L.dev_ld1_batch; auto ld2 = L.dev_ld2_batch;
        auto F11 = L.dev_F_batch;   auto F12 = F11 + Nsmall;
        auto F21 = F12 + Nsmall;    auto F22 = F21 + Nsmall;
        gpu::copy_host_to_device(d1,  L.d1_batch.data(),  Nsmall);
        gpu::copy_host_to_device(d2,  L.d2_batch.data(),  Nsmall);
        gpu::copy_host_to_device(ld1, L.ld1_batch.data(), Nsmall);
        gpu::copy_host_to_device(ld2, L.ld2_batch.data(), Nsmall);
        gpu::copy_host_to_device(F11, L.F_batch.data(),   4*Nsmall);
        gpu::copy_host_to_device(L.dev_ipiv_batch, L.ipiv_batch.data(), Nsmall);

#if defined(VBATCHED_STATS)
	t_getrf_small.start();
#endif
#if defined(STRUMPACK_USE_KBLAS)
        typedef typename kblas_base_type<scalar_t>::device_type kblas_type;
        kblas_getrf_vbatch
          (kblas_handle, d1, d1, L.max_d1_small, L.max_d1_small,
           (kblas_type**)F11, d1, L.dev_ipiv_batch,
           L.dev_getrf_work, L.getrf_work_bytes, L.dev_getrf_err, Nsmall);
        comp_stream.synchronize();
#else
        magma_iset_pointer
          (L.dpivinfo_array, L.dpivinfo, 1, 0, 0,
           L.max_d1_small, Nsmall, magma_q);
        gpu::magma::getrf_vbatched_max_nocheck
          (d1, d1, d1, L.max_d1_small, L.max_d1_small, L.max_d1_small,
           L.max_d1_small*L.max_d1_small, F11, ld1, L.dev_ipiv_batch,
           L.dpivinfo_array, L.dev_getrf_err, Nsmall, magma_q);
#endif
#if defined(VBATCHED_STATS)
	comp_stream.synchronize();
	t_getrf_small.stop();
	t_getrf_large.start();
#endif
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::getrf
            (solver_handle, L.f[i]->F11_, (scalar_t*)L.dev_getrf_work,
             L.f[i]->piv_, L.dev_getrf_err+i);
#if defined(VBATCHED_STATS)
	comp_stream.synchronize();
	t_getrf_large.stop();
#endif

        // TODO
        // if (opts.replace_tiny_pivots())
        //   gpu::replace_pivots
        //     (f.dim_sep(), f.F11_.data(),
        //      opts.pivot_threshold(), comp_stream);


#if defined(VBATCHED_STATS)
	t_laswp.start();
#endif
        gpu::laswp_fwd_vbatched
          (blas_handle, d2, L.max_d2_small, F12, ld1, L.dev_ipiv_batch,
           d1, Nsmall);
#if defined(VBATCHED_STATS)
	comp_stream.synchronize();
	t_laswp.stop();
	t_trsm_L.start();
#endif
        gpu::magma::trsm_vbatched
          (MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
           d1, d2, scalar_t(1.), F11, ld1, F12, ld1, Nsmall, magma_q);
#if defined(VBATCHED_STATS)
	comp_stream.synchronize();
	t_trsm_L.stop();
	t_trsm_U.start();
#endif
        gpu::magma::trsm_vbatched
          (MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
           d1, d2, scalar_t(1.), F11, ld1, F12, ld1, Nsmall, magma_q);
#if defined(VBATCHED_STATS)
	comp_stream.synchronize();
	t_trsm_U.stop();
	t_getrs.start();
#endif
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::getrs
            (solver_handle, Trans::N, L.f[i]->F11_, L.f[i]->piv_,
             L.f[i]->F12_, L.dev_getrf_err+i);
#if defined(VBATCHED_STATS)
	comp_stream.synchronize();
	t_getrs.stop();
#endif

        STRUMPACK_ADD_MEMORY(L.factor_size*sizeof(scalar_t));
        L.f[0]->host_factors_.reset(new scalar_t[L.factor_size]);
        L.f[0]->pivot_mem_.resize(L.piv_size);

        comp_stream.synchronize();
        gpu::copy_device_to_host_async<scalar_t>
          (pinned, dev_factors, L.factor_size, copy_stream);

        // use max_nocheck to overlap this with copy above
#if defined(VBATCHED_STATS)
	t_gemm_small.start();
#endif
        gpu::magma::gemm_vbatched_max_nocheck
          (MagmaNoTrans, MagmaNoTrans, d2, d2, d1, scalar_t(-1.),
           F21, ld2, F12, ld1, scalar_t(1.), F22, ld2, Nsmall,
           L.max_d2_small, L.max_d2_small, L.max_d1_small, magma_q);
#if defined(VBATCHED_STATS)
	comp_stream.synchronize();
	t_gemm_small.stop();
	t_gemm.start();
#endif
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::gemm
            (blas_handle, Trans::N, Trans::N, scalar_t(-1.),
             L.f[i]->F21_, L.f[i]->F12_, scalar_t(1.), L.f[i]->F22_);
#if defined(VBATCHED_STATS)
	comp_stream.synchronize();
	t_gemm.stop();
#endif
        copy_stream.synchronize();
        auto host_factors = L.f[0]->host_factors_.get();
#pragma omp parallel for
        for (std::size_t i=0; i<L.factor_size; i++)
          host_factors[i] = pinned[i];

        gpu::copy_device_to_host
          (L.f[0]->pivot_mem_.data(), L.f[0]->piv_, L.piv_size);
        L.set_factor_pointers(L.f[0]->host_factors_.get());
        L.set_pivot_pointers(L.f[0]->pivot_mem_.data());
        comp_stream.synchronize();
      } catch (const std::bad_alloc& e) {
        std::cerr << "Out of memory" << std::endl;
        abort();
      }
      long long level_flops, small_flops;
      L.flops(level_flops, small_flops);
      STRUMPACK_FULL_RANK_FLOPS(level_flops);
      STRUMPACK_FLOPS(small_flops);
      if (opts.verbose()) {

#if defined(VBATCHED_STATS)
        long long flops_LU_small = 0, flops_LU_large = 0,
          flops_trsm_L_small = 0, flops_trsm_L_large = 0,
          flops_trsm_U_small = 0, flops_trsm_U_large = 0,
          flops_gemm_small = 0, flops_gemm_large = 0;
#if defined(VBATCHED_STATS)
        for (std::size_t i=0; i<L.f.size(); i++) {
          auto F = L.f[i];
          if (F->dim_sep() <= L.FRONT_SMALL) {
            flops_LU_small += LU_flops(F->F11_);
            flops_gemm_small += gemm_flops(Trans::N, Trans::N, scalar_t(-1.), F->F21_, F->F12_, scalar_t(1.));
            flops_trsm_L_small += trsm_flops(Side::L, scalar_t(1.), F->F11_, F->F12_);
            flops_trsm_U_small += trsm_flops(Side::R, scalar_t(1.), F->F11_, F->F21_);
          } else {
            flops_LU_large += LU_flops(F->F11_);
            flops_gemm_large += gemm_flops(Trans::N, Trans::N, scalar_t(-1.), F->F21_, F->F12_, scalar_t(1.));
            flops_trsm_L_large += trsm_flops(Side::L, scalar_t(1.), F->F11_, F->F12_);
            flops_trsm_U_large += trsm_flops(Side::R, scalar_t(1.), F->F11_, F->F21_);
          }
        }
#endif
        auto flops_getrs_large = flops_trsm_L_large + flops_trsm_U_large;
#endif
        auto level_time = tl.elapsed();
        std::cout << "#   GPU Factorization complete, took: "
                  << level_time << " seconds, "
                  << level_flops / 1.e9 << " GFLOPS, "
                  << (float(level_flops) / level_time) / 1.e9
                  << " GFLOP/s" << std::endl;
#if defined(VBATCHED_STATS)
	std::cout << "#          LU_vb:       " << t_getrf_small.elapsed() << "  " << flops_LU_small/1.e9 << " GFlop " << (float(flops_LU_small)/t_getrf_small.elapsed())/1.e9 << " GFlop/s" << std::endl
		  << "#          LU_large:    " << t_getrf_large.elapsed() << "  " << flops_LU_large/1.e9 << " GFlop " << (float(flops_LU_large)/t_getrf_large.elapsed())/1.e9 << " GFlop/s" << std::endl
	   	  << "#          LU_total:    " << (t_getrf_small.elapsed()+t_getrf_large.elapsed()) << "  " << (flops_LU_small+flops_LU_large)/1.e9 << " GFlop " << ((float(flops_LU_small)+float(flops_LU_large))/(t_getrf_small.elapsed()+t_getrf_large.elapsed()))/1.e9 << " GFlop/s" << std::endl
		  << "#          laswp_vb:    " << t_laswp.elapsed() << std::endl
		  << "#          getrs_large: " << t_getrs.elapsed() << "  " << flops_getrs_large/1.e9 << " GFlop " << (float(flops_getrs_large)/t_getrs.elapsed())/1.e9 << " GFlop/s" << std::endl
		  << "#          trsm_L_vb    " << t_trsm_L.elapsed() << "  " << flops_trsm_L_small/1.e9 << " GFlop " << (float(flops_trsm_L_small)/t_trsm_L.elapsed())/1.e9 << " GFlop/s" << std::endl
		  << "#          trsm_U_vb:   " <<t_trsm_U.elapsed() << "  " << flops_trsm_U_small/1.e9 << " GFlop " << (float(flops_trsm_U_small)/t_trsm_U.elapsed())/1.e9 << " GFlop/s" << std::endl
	   	  << "#          getrs_total: " << (t_getrs.elapsed()+t_trsm_L.elapsed()+t_trsm_U.elapsed()) << "  " << (flops_getrs_large+flops_trsm_L_small+flops_trsm_U_small)/1.e9 << " GFlop " << ((float(flops_getrs_large)+float(flops_trsm_L_small)+float(flops_trsm_U_small))/(t_getrs.elapsed()+t_trsm_L.elapsed()+t_trsm_U.elapsed()))/1.e9 << " GFlop/s" << std::endl
		  << "#          gemm_vb:     " << t_gemm_small.elapsed() << "  " << flops_gemm_small/1.e9 << " GFlop " << (float(flops_gemm_small)/t_gemm_small.elapsed())/1.e9 << " GFlop/s" << std::endl
		  << "#          gemm_large:  " << t_gemm.elapsed() << "  " << flops_gemm_large/1.e9 << " GFlop " << (float(flops_gemm_large)/t_gemm.elapsed())/1.e9 << " GFlop/s" << std::endl
	   	  << "#          gemm_total:  " << (t_gemm_small.elapsed()+t_gemm.elapsed()) << "  " << (flops_gemm_small+flops_gemm_large)/1.e9 << " GFlop " << ((float(flops_gemm_small)+float(flops_gemm_large))/(t_gemm_small.elapsed()+t_gemm.elapsed()))/1.e9 << " GFlop/s" << std::endl;
#endif

      }
    }
    const std::size_t dupd = dim_upd();
    if (dupd) { // get the contribution block from the device
      host_Schur_.reset(new scalar_t[dupd*dupd]);
      gpu::copy_device_to_host
        (host_Schur_.get(), reinterpret_cast<scalar_t*>(old_work), dupd*dupd);
      F22_ = DenseMW_t(dupd, dupd, host_Schur_.get(), dupd);
    }
    magma_queue_destroy(magma_q);
    magma_finalize();
  }


  template<typename scalar_t,typename integer_t>
  std::unique_ptr<GPUFactors<scalar_t>>
  FrontalMatrixGPU<scalar_t,integer_t>::move_to_gpu() const {
    const int lvls = this->levels();
    std::unique_ptr<GPUFactors<scalar_t>> df(new GPUFactors<scalar_t>(lvls));
    // std::vector<LInfo_t> ldata(lvls);
    // std::vector<gpu::SOLVERHandle> solver_handles(1);
    // std::size_t Lsize = 0, Usize = 0, Psize = 0;
    // for (int l=lvls-1; l>=0; l--) {
    //   std::vector<F_t*> fp;
    //   const_cast<FG_t*>(this)->get_level_fronts(fp, l);
    //   auto& L = ldata[l];
    //   L = LInfo_t(fp, solver_handles[0], 1);
    //   Lsize += L.L_size;
    //   Usize += L.U_size;
    //   Psize += L.piv_size;
    // }
    // df->data_->dL = gpu::DeviceMemory<scalar_t>(Lsize);
    // df->data_->dU = gpu::DeviceMemory<scalar_t>(Usize);
    // df->data_->dP = gpu::DeviceMemory<int>(Psize);
    // scalar_t *Lptr = df->data_->dL, *Uptr = df->data_->dU;
    // int* Pptr = df->data_->dP;
    // for (int l=lvls-1; l>=0; l--) {
    //   auto& L = ldata[l];
    //   df->data_->dL_lvl[l] = Lptr;
    //   df->data_->dU_lvl[l] = Uptr;
    //   df->data_->dP_lvl[l] = Pptr;
    //   gpu::copy_host_to_device<scalar_t>
    //     (df->data_->dL_lvl[l], L.f[0]->host_factors_.get(), L.L_size);
    //   gpu::copy_host_to_device<scalar_t>
    //     (df->data_->dU_lvl[l], L.f[0]->host_factors_.get()+L.L_size, L.U_size);
    //   gpu::copy_host_to_device<int>
    //     (df->data_->dP_lvl[l], L.f[0]->piv_, L.piv_size);
    //   Lptr += L.L_size;
    //   Uptr += L.U_size;
    //   Pptr += L.piv_size;
    // }
    return df;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::multifrontal_solve
  (DenseM_t& b, const GPUFactors<scalar_t>* gpu_factors) const {
#if 0
    fwd_solve_gpu(b, nullptr, gpu_factors);
    bwd_solve_gpu(b, nullptr, gpu_factors);
#else
    FrontalMatrix<scalar_t,integer_t>::multifrontal_solve(b);
#endif
  }

  // template<typename scalar_t,typename integer_t> void
  // FrontalMatrixGPU<scalar_t,integer_t>::rhs_to_contig
  // (LInfo_t& L, const DenseM_t& b, scalar_t* bptr) const {
  //   int d = b.cols();
  //   for (std::size_t n=0; n<L.f.size(); n++) {
  //     auto& f = *(L.f[n]);
  //     for (int i=0; i<d; i++) {
  //       std::copy(b.ptr(f.sep_begin_, i),
  //                 b.ptr(f.sep_end_, i), bptr);
  //       bptr += f.dim_sep();
  //     }
  //   }
  // }

  // template<typename scalar_t,typename integer_t> void
  // FrontalMatrixGPU<scalar_t,integer_t>::rhs_from_contig
  // (LInfo_t& L, DenseM_t& b, const scalar_t* bptr) const {
  //   int d = b.cols();
  //   for (std::size_t n=0; n<L.f.size(); n++) {
  //     auto& f = *(L.f[n]);
  //     const auto dsep = f.dim_sep();
  //     for (int i=0; i<d; i++) {
  //       std::copy(bptr, bptr+dsep, b.ptr(f.sep_begin_, i));
  //       bptr += dsep;
  //     }
  //   }
  // }

  // template<typename scalar_t, typename integer_t> void
  // FrontalMatrixGPU<scalar_t,integer_t>::assemble_rhs
  // (int nrhs, LInfo_t& L, scalar_t* db, scalar_t* dbupd,
  //  scalar_t* old_dbupd, char* dea_mem, char* hea_mem,
  //  std::size_t mem_size) const {
  //   auto N = L.f.size();
  //   auto hasmbl = reinterpret_cast<gpu::AssembleData<scalar_t>*>(hea_mem);
  //   auto Iptr = reinterpret_cast<std::size_t*>(round_to_16(hasmbl + N));
  //   auto dasmbl = reinterpret_cast<gpu::AssembleData<scalar_t>*>(dea_mem);
  //   auto dIptr = reinterpret_cast<std::size_t*>(round_to_16(dasmbl + N));
  //   for (std::size_t n=0; n<N; n++) {
  //     auto& f = *(L.f[n]);
  //     const auto dsep = f.dim_sep();
  //     const auto dupd = f.dim_upd();
  //     hasmbl[n] = gpu::AssembleData<scalar_t>(dsep, dupd, db, dbupd);
  //     if (f.lchild_) {
  //       auto c = dynamic_cast<FG_t*>(f.lchild_.get());
  //       auto cdupd = c->dim_upd();
  //       hasmbl[n].set_ext_add_left(cdupd, old_dbupd, dIptr);
  //       auto u = c->upd_to_parent(&f);
  //       std::copy(u.begin(), u.end(), Iptr);
  //       Iptr += cdupd;
  //       dIptr += cdupd;
  //       old_dbupd += cdupd*nrhs;
  //     }
  //     if (f.rchild_) {
  //       auto c = dynamic_cast<FG_t*>(f.rchild_.get());
  //       auto cdupd = c->dim_upd();
  //       hasmbl[n].set_ext_add_right(cdupd, old_dbupd, dIptr);
  //       auto u = c->upd_to_parent(&f);
  //       std::copy(u.begin(), u.end(), Iptr);
  //       Iptr += cdupd;
  //       dIptr += cdupd;
  //       old_dbupd += cdupd*nrhs;
  //     }
  //     db += dsep*nrhs;
  //     dbupd += dupd*nrhs;
  //   }
  //   gpu::copy_host_to_device<char>(dea_mem, hea_mem, mem_size);
  //   gpu::extend_add_rhs(nrhs, N, hasmbl, dasmbl);
  //   gpu::synchronize();
  // }

  // template<typename scalar_t, typename integer_t> void
  // FrontalMatrixGPU<scalar_t,integer_t>::fwd_small_fronts
  // (int nrhs, LInfo_t& L, gpu::FrontData<scalar_t>* fdata,
  //  gpu::FrontData<scalar_t>* dfdata, scalar_t* dL, int* dpiv,
  //  scalar_t* db, scalar_t* dbupd) const {
  //   if (L.N8 || L.N16 || L.N24 || L.N32) {
  //     for (std::size_t n=0, n8=0, n16=L.N8, n24=n16+L.N16, n32=n24+L.N24;
  //          n<L.f.size(); n++) {
  //       auto& f = *(L.f[n]);
  //       const auto dsep = f.dim_sep();
  //       const auto dupd = f.dim_upd();
  //       if (dsep <= 32) {
  //         gpu::FrontData<scalar_t>
  //           t(dsep, dupd, dL, db, dL+dsep*dsep, dbupd, dpiv);
  //         if (dsep <= 8)       fdata[n8++] = t;
  //         else if (dsep <= 16) fdata[n16++] = t;
  //         else if (dsep <= 24) fdata[n24++] = t;
  //         else                 fdata[n32++] = t;
  //       }
  //       dpiv += dsep;
  //       dL += dsep*(dsep+dupd);
  //       db += dsep*nrhs;
  //       dbupd += dupd*nrhs;
  //     }
  //     gpu::copy_host_to_device(dfdata, fdata, L.N8+L.N16+L.N24+L.N32);
  //     gpu::fwd_block_batch<scalar_t,8>(nrhs, L.N8, dfdata);
  //     gpu::fwd_block_batch<scalar_t,16>(nrhs, L.N16, dfdata+L.N8);
  //     gpu::fwd_block_batch<scalar_t,24>(nrhs, L.N24, dfdata+L.N8+L.N16);
  //     gpu::fwd_block_batch<scalar_t,32>(nrhs, L.N32, dfdata+L.N8+L.N16+L.N24);
  //   }
  // }

  // template<typename scalar_t, typename integer_t> void
  // FrontalMatrixGPU<scalar_t,integer_t>::fwd_large_fronts
  // (int nrhs, LInfo_t& L, scalar_t* dL, int* dpiv,
  //  int* derr, scalar_t* db, scalar_t* dbupd,
  //  std::vector<gpu::BLASHandle>& blas_handles,
  //  std::vector<gpu::SOLVERHandle>& solver_handles) const {
  //   for (std::size_t n=0; n<L.f.size(); n++) {
  //     auto& f = *(L.f[n]);
  //     auto stream = n % solver_handles.size();
  //     const auto dsep = f.dim_sep();
  //     const auto dupd = f.dim_upd();
  //     if (dsep > 32) {
  //       DenseMW_t b(dsep, nrhs, db, dsep),
  //         bupd(dupd, nrhs, dbupd, dupd), F11(dsep, dsep, dL, dsep),
  //         F21(dupd, dsep, dL+dsep*dsep, dupd);
  //       gpu::getrs
  //         (solver_handles[stream], Trans::N, F11, dpiv,
  //          b, derr + stream);
  //       if (dupd) {
  //         if (nrhs == 1)
  //           gpu::gemv(blas_handles[stream], Trans::N,
  //                     scalar_t(-1.), F21, b, scalar_t(1.), bupd);
  //         else
  //           gpu::gemm(blas_handles[stream], Trans::N, Trans::N,
  //                     scalar_t(-1.), F21, b, scalar_t(1.), bupd);
  //       }
  //     }
  //     dpiv += dsep;
  //     dL += dsep*(dsep+dupd);
  //     db += dsep*nrhs;
  //     dbupd += dupd*nrhs;
  //   }
  // }

// #if 0
//   template<typename scalar_t,typename integer_t> void
//   FrontalMatrixGPU<scalar_t,integer_t>::forward_multifrontal_solve
//   (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
//     fwd_solve_gpu(b, work, nullptr);
//   }
// #else

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::forward_multifrontal_solve
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
// #endif

  // template<typename scalar_t,typename integer_t> void
  // FrontalMatrixGPU<scalar_t,integer_t>::fwd_solve_gpu
  // (DenseM_t& b, DenseM_t* work, const GPUFactors<scalar_t>* gpu_factors)
  //   const {
  //   const int lvls = this->levels();
  //   const int d = b.cols();
  //   const int max_streams = 1; //opts.gpu_streams();
  //   std::vector<gpu::SOLVERHandle> solver_handles(max_streams);
  //   std::vector<LInfo_t> ldata(lvls);
  //   std::size_t max_small_fronts = 0, maxLb = 0, maxupd = 0, maxpiv = 0;
  //   for (int l=lvls-1; l>=0; l--) {
  //     std::vector<F_t*> fp;
  //     const_cast<FG_t*>(this)->get_level_fronts(fp, l);
  //     auto& L = ldata[l];
  //     L = LInfo_t(fp, solver_handles[0], max_streams);
  //     max_small_fronts = std::max(max_small_fronts, L.N8+L.N16+L.N24+L.N32);
  //     maxupd = std::max(maxupd, L.total_upd_size);
  //     if (gpu_factors) maxLb = std::max(maxLb, d*L.piv_size);
  //     else maxLb = std::max(maxLb, L.L_size + d*L.piv_size);
  //     maxpiv = std::max(maxpiv, L.piv_size);
  //   }
  //   std::size_t max_ea_size = 0;
  //   for (int l=lvls-2; l>=0; l--)
  //     max_ea_size = std::max
  //       (max_ea_size,
  //        round_to_16(ldata[l].f.size() * sizeof(gpu::AssembleData<scalar_t>)) +
  //        round_to_16(ldata[l+1].total_upd_size * sizeof(std::size_t)));


  //   // TODO, if the factorization was split, then split here as well,
  //   // keep a bool???

  //   // still check if memory is sufficient, if not, then split rhs
  //   // along column dimension

  //   // if (!sufficient_device_memory_solve(ldata, true, b.cols())) {
  //   //   std::cerr << "ERROR: factors do not fit in GPU" << std::endl;
  //   //   abort();
  //   //   // split_smaller(A, opts, etree_level, task_depth);
  //   //   // return;
  //   // }

  //   std::vector<gpu::Stream> streams(max_streams);
  //   std::vector<gpu::BLASHandle> blas_handles(max_streams);
  //   for (int i=0; i<max_streams; i++) {
  //     blas_handles[i].set_stream(streams[i]);
  //     solver_handles[i].set_stream(streams[i]);
  //   }
  //   gpu::HostMemory<gpu::FrontData<scalar_t>> fdata(max_small_fronts);
  //   gpu::HostMemory<char> hea(max_ea_size);
  //   gpu::DeviceMemory<char> dmem
  //     (round_to_16(max_small_fronts * sizeof(gpu::FrontData<scalar_t>)) +
  //      round_to_16((maxLb + 2 * d * maxupd) * sizeof(scalar_t)) +
  //      round_to_16(((gpu_factors ? 0 : maxpiv) + max_streams) * sizeof(int)) +
  //      max_ea_size);
  //   auto dfdata = dmem.as<gpu::FrontData<scalar_t>>();
  //   scalar_t *dL = nullptr, *db = nullptr, *dbupd = nullptr;
  //   if (gpu_factors) {
  //     db = reinterpret_cast<scalar_t*>(round_to_16(dfdata + max_small_fronts));
  //     dbupd = db + maxLb;
  //   } else {
  //     dL = reinterpret_cast<scalar_t*>(round_to_16(dfdata + max_small_fronts));
  //     dbupd = dL + maxLb;
  //   }
  //   auto old_dbupd = dbupd + d * maxupd;
  //   auto derr = reinterpret_cast<int*>(round_to_16(old_dbupd + d * maxupd));
  //   int *dpiv = nullptr;
  //   char* dea = nullptr;
  //   if (gpu_factors)
  //     dea = reinterpret_cast<char*>(round_to_16(derr + max_streams));
  //   else {
  //     dpiv = derr + max_streams;
  //     dea = reinterpret_cast<char*>(round_to_16(dpiv + maxpiv));
  //   }
  //   try {
  //     for (int l=lvls-1; l>=0; l--) {
  //       // TaskTimer tl("");
  //       // tl.start();
  //       auto& L = ldata[l];
  //       if (gpu_factors) {
  //         dL = gpu_factors->data_->dL_lvl[l];
  //         dpiv = gpu_factors->data_->dP_lvl[l];
  //       } else {
  //         gpu::copy_host_to_device<scalar_t>
  //           (dL, L.f[0]->host_factors_.get(), L.L_size);
  //         gpu::copy_host_to_device<int>
  //           (dpiv, L.f[0]->pivot_mem_.data(), L.piv_size);
  //         db = dL + L.L_size;
  //       }
  //       gpu::memset<scalar_t>(dbupd, 0, d*L.total_upd_size);
  //       std::unique_ptr<scalar_t[]> allb(new scalar_t[d * L.piv_size]);
  //       rhs_to_contig(L, b, allb.get());
  //       gpu::copy_host_to_device<scalar_t>
  //         (db, allb.get(), d * L.piv_size);
  //       if (l != lvls-1) {
  //         std::size_t mem_size =
  //           round_to_16(L.f.size() * sizeof(gpu::AssembleData<scalar_t>)) +
  //           round_to_16(ldata[l+1].total_upd_size * sizeof(std::size_t));
  //         assemble_rhs(d, L, db, dbupd, old_dbupd, dea, hea, mem_size);
  //       }
  //       fwd_small_fronts(d, L, fdata, dfdata, dL, dpiv, db, dbupd);
  //       fwd_large_fronts(d, L, dL, dpiv, derr, db, dbupd,
  //                        blas_handles, solver_handles);
  //       gpu::synchronize();
  //       std::swap(dbupd, old_dbupd);
  //       gpu::copy_device_to_host<scalar_t>(allb.get(), db, d*L.piv_size);
  //       rhs_from_contig(L, b, allb.get());
  //       // auto level_time = tl.elapsed();
  //       // std::cout << "#   GPU Fwd Solve complete, took: "
  //       //           << level_time << " seconds" << std::endl;
  //     }
  //   } catch (const std::bad_alloc& e) {
  //     std::cerr << "Out of memory" << std::endl;
  //     abort();
  //   }
  //   const std::size_t dupd = dim_upd();
  //   if (dupd) { // get bupd from the device, needed by parent
  //     DenseMW_t bupd(dupd, d, work[0], 0, 0);
  //     gpu::copy_device_to_host(bupd, old_dbupd);
  //   }
  // }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
      F11_.solve_LU_in_place(bloc, piv_, task_depth);
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

  // template<typename scalar_t, typename integer_t> void
  // FrontalMatrixGPU<scalar_t,integer_t>::extract_rhs
  // (int nrhs, LInfo_t& L, scalar_t* dy, scalar_t* dyupd, scalar_t* old_dyupd,
  //  char* dea_mem, char* hea_mem, std::size_t mem_size) const {
  //   auto N = L.f.size();
  //   auto hasmbl = reinterpret_cast<gpu::AssembleData<scalar_t>*>(hea_mem);
  //   auto Iptr = reinterpret_cast<std::size_t*>(round_to_16(hasmbl + N));
  //   auto dasmbl = reinterpret_cast<gpu::AssembleData<scalar_t>*>(dea_mem);
  //   auto dIptr = reinterpret_cast<std::size_t*>(round_to_16(dasmbl + N));
  //   for (std::size_t n=0; n<N; n++) {
  //     auto& f = *(L.f[n]);
  //     const auto dsep = f.dim_sep();
  //     const auto dupd = f.dim_upd();
  //     hasmbl[n] = gpu::AssembleData<scalar_t>(dsep, dupd, dy, dyupd);
  //     if (f.lchild_) {
  //       auto c = dynamic_cast<FG_t*>(f.lchild_.get());
  //       auto cdupd = c->dim_upd();
  //       hasmbl[n].set_ext_add_left(cdupd, old_dyupd, dIptr);
  //       auto u = c->upd_to_parent(&f);
  //       std::copy(u.begin(), u.end(), Iptr);
  //       Iptr += cdupd;
  //       dIptr += cdupd;
  //       old_dyupd += cdupd*nrhs;
  //     }
  //     if (f.rchild_) {
  //       auto c = dynamic_cast<FG_t*>(f.rchild_.get());
  //       auto cdupd = c->dim_upd();
  //       hasmbl[n].set_ext_add_right(cdupd, old_dyupd, dIptr);
  //       auto u = c->upd_to_parent(&f);
  //       std::copy(u.begin(), u.end(), Iptr);
  //       Iptr += cdupd;
  //       dIptr += cdupd;
  //       old_dyupd += cdupd*nrhs;
  //     }
  //     dy += dsep*nrhs;
  //     dyupd += dupd*nrhs;
  //   }
  //   gpu::copy_host_to_device<char>(dea_mem, hea_mem, mem_size);
  //   gpu::extract_rhs(nrhs, N, hasmbl, dasmbl);
  //   gpu::synchronize();
  // }

//   template<typename scalar_t, typename integer_t> void
//   FrontalMatrixGPU<scalar_t,integer_t>::bwd_small_fronts
//   (int nrhs, LInfo_t& L, gpu::FrontData<scalar_t>* fdata,
//    gpu::FrontData<scalar_t>* dfdata, scalar_t* dU,
//    scalar_t* dy, scalar_t* dyupd) const {
//     if (L.N8 || L.N16 || L.N24 || L.N32) {
//       for (std::size_t n=0, n8=0, n16=L.N8, n24=n16+L.N16, n32=n24+L.N24;
//            n<L.f.size(); n++) {
//         auto& f = *(L.f[n]);
//         const auto dsep = f.dim_sep();
//         const auto dupd = f.dim_upd();
//         if (dsep <= 32) {
//           gpu::FrontData<scalar_t>
//             t(dsep, dupd, dy, dU, dyupd, nullptr, nullptr);
//           if (dsep <= 8)       fdata[n8++] = t;
//           else if (dsep <= 16) fdata[n16++] = t;
//           else if (dsep <= 24) fdata[n24++] = t;
//           else                 fdata[n32++] = t;
//         }
//         dU += dsep*dupd;
//         dy += dsep*nrhs;
//         dyupd += dupd*nrhs;
//       }
//       gpu::copy_host_to_device(dfdata, fdata, L.N8+L.N16+L.N24+L.N32);
//       gpu::bwd_block_batch<scalar_t,8>(nrhs, L.N8, dfdata);
//       gpu::bwd_block_batch<scalar_t,16>(nrhs, L.N16, dfdata+L.N8);
//       gpu::bwd_block_batch<scalar_t,24>(nrhs, L.N24, dfdata+L.N8+L.N16);
//       gpu::bwd_block_batch<scalar_t,32>(nrhs, L.N32, dfdata+L.N8+L.N16+L.N24);
//     }
//   }

//   template<typename scalar_t, typename integer_t> void
//   FrontalMatrixGPU<scalar_t,integer_t>::bwd_large_fronts
//   (int nrhs, LInfo_t& L, scalar_t* dU, scalar_t* dy, scalar_t* dyupd,
//    std::vector<gpu::BLASHandle>& blas_handles,
//    std::vector<gpu::SOLVERHandle>& solver_handles) const {
//     for (std::size_t n=0; n<L.f.size(); n++) {
//       auto& f = *(L.f[n]);
//       auto stream = n % solver_handles.size();
//       const auto dsep = f.dim_sep();
//       const auto dupd = f.dim_upd();
//       if (dsep > 32) {
//         DenseMW_t y(dsep, nrhs, dy, dsep),
//           yupd(dupd, nrhs, dyupd, dupd), F12(dsep, dupd, dU, dsep);
//         if (dupd) {
//           if (nrhs == 1)
//             gpu::gemv(blas_handles[stream], Trans::N,
//                       scalar_t(-1.), F12, yupd, scalar_t(1.), y);
//           else
//             gpu::gemm(blas_handles[stream], Trans::N, Trans::N,
//                       scalar_t(-1.), F12, yupd, scalar_t(1.), y);
//         }
//       }
//       dU += dsep*dupd;
//       dy += dsep*nrhs;
//       dyupd += dupd*nrhs;
//     }
//   }

// #if 0
//   template<typename scalar_t,typename integer_t> void
//   FrontalMatrixGPU<scalar_t,integer_t>::backward_multifrontal_solve
//   (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
//     bwd_solve_gpu(y, work, nullptr);
//   }
// #else

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::backward_multifrontal_solve
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
// #endif

  // template<typename scalar_t,typename integer_t> void
  // FrontalMatrixGPU<scalar_t,integer_t>::bwd_solve_gpu
  // (DenseM_t& y, DenseM_t* work,
  //  const GPUFactors<scalar_t>* gpu_factors) const {
  //   const int lvls = this->levels();
  //   const int d = y.cols();
  //   const int max_streams = 1; //opts.gpu_streams();
  //   std::vector<gpu::SOLVERHandle> solver_handles(max_streams);
  //   std::vector<LInfo_t> ldata(lvls);
  //   std::size_t max_small_fronts = 0, maxUy = 0, maxupd = 0;
  //   for (int l=lvls-1; l>=0; l--) {
  //     std::vector<F_t*> fp;
  //     const_cast<FG_t*>(this)->get_level_fronts(fp, l);
  //     auto& L = ldata[l];
  //     L = LInfo_t(fp, solver_handles[0], max_streams);
  //     max_small_fronts = std::max(max_small_fronts, L.N8+L.N16+L.N24+L.N32);
  //     maxupd = std::max(maxupd, L.total_upd_size);
  //     if (gpu_factors) maxUy = std::max(maxUy, d*L.piv_size);
  //     else maxUy = std::max(maxUy, L.U_size + d*L.piv_size);
  //   }
  //   std::size_t max_ea_size = 0;
  //   for (int l=0; l<lvls-1; l++)
  //     max_ea_size = std::max
  //       (max_ea_size,
  //        round_to_16(ldata[l].f.size() * sizeof(gpu::AssembleData<scalar_t>)) +
  //        round_to_16(ldata[l+1].total_upd_size * sizeof(std::size_t)));

  //   std::vector<gpu::Stream> streams(max_streams);
  //   std::vector<gpu::BLASHandle> blas_handles(max_streams);
  //   for (int i=0; i<max_streams; i++) {
  //     blas_handles[i].set_stream(streams[i]);
  //     solver_handles[i].set_stream(streams[i]);
  //   }
  //   try {
  //     // use front data to also store pointers to b, bupd
  //     gpu::HostMemory<gpu::FrontData<scalar_t>> fdata(max_small_fronts);
  //     gpu::HostMemory<char> hea(max_ea_size);
  //     gpu::DeviceMemory<char> dmem
  //       (round_to_16(max_small_fronts * sizeof(gpu::FrontData<scalar_t>)) +
  //        round_to_16((2*d*maxupd + maxUy) * sizeof(scalar_t)) +
  //        max_ea_size);
  //     auto dfdata = dmem.as<gpu::FrontData<scalar_t>>();
  //     auto dyupd = reinterpret_cast<scalar_t*>
  //       (round_to_16(dfdata + max_small_fronts));
  //     auto new_dyupd = dyupd + d*maxupd;
  //     scalar_t *dU = nullptr, *dy = nullptr;
  //     char *dea = nullptr;
  //     if (gpu_factors) {
  //       dy = new_dyupd + d*maxupd;
  //       dea = reinterpret_cast<char*>(round_to_16(dy + maxUy));
  //     } else {
  //       dU = new_dyupd + d*maxupd;
  //       dea = reinterpret_cast<char*>(round_to_16(dU + maxUy));
  //     }
  //     for (int l=0; l<lvls; l++) {
  //       // TaskTimer tl("");
  //       // tl.start();
  //       auto& L = ldata[l];
  //       if (gpu_factors)
  //         dU = gpu_factors->data_->dU_lvl[l];
  //       else {
  //         dy = dU + L.U_size;
  //         gpu::copy_host_to_device<scalar_t>
  //           (dU, L.f[0]->host_factors_.get()+L.L_size, L.U_size);
  //       }
  //       std::unique_ptr<scalar_t[]> ally(new scalar_t[d * L.piv_size]);
  //       rhs_to_contig(L, y, ally.get());
  //       gpu::copy_host_to_device<scalar_t>
  //         (dy, ally.get(), d * L.piv_size);
  //       bwd_small_fronts(d, L, fdata, dfdata, dU, dy, dyupd);
  //       bwd_large_fronts(d, L, dU, dy, dyupd,
  //                        blas_handles, solver_handles);
  //       if (l != lvls-1) {
  //         std::size_t mem_size =
  //           round_to_16(L.f.size() * sizeof(gpu::AssembleData<scalar_t>)) +
  //           round_to_16(ldata[l+1].total_upd_size * sizeof(std::size_t));
  //         extract_rhs(d, L, dy, dyupd, new_dyupd, dea, hea, mem_size);
  //         std::swap(dyupd, new_dyupd);
  //       }
  //       gpu::synchronize();
  //       gpu::copy_device_to_host<scalar_t>(ally.get(), dy, d*L.piv_size);
  //       rhs_from_contig(L, y, ally.get());
  //       // auto level_time = tl.elapsed();
  //       // std::cout << "#   GPU Bwd Solve complete, took: "
  //       //           << level_time << " seconds" << std::endl;
  //     }
  //   } catch (const std::bad_alloc& e) {
  //     std::cerr << "Out of memory" << std::endl;
  //     abort();
  //   }
  // }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::bwd_solve_phase1
  (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t yloc(dim_sep(), y.cols(), y, this->sep_begin_, 0);
      if (y.cols() == 1) {
        if (dim_upd())
          gemv(Trans::N, scalar_t(-1.), F12_, yupd,
               scalar_t(1.), yloc, task_depth);
      } else {
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F12_, yupd,
               scalar_t(1.), yloc, task_depth);
      }
    }
  }

  // explicit template instantiations
  template class FrontalMatrixGPU<float,int>;
  template class FrontalMatrixGPU<double,int>;
  template class FrontalMatrixGPU<std::complex<float>,int>;
  template class FrontalMatrixGPU<std::complex<double>,int>;

  template class FrontalMatrixGPU<float,long int>;
  template class FrontalMatrixGPU<double,long int>;
  template class FrontalMatrixGPU<std::complex<float>,long int>;
  template class FrontalMatrixGPU<std::complex<double>,long int>;

  template class FrontalMatrixGPU<float,long long int>;
  template class FrontalMatrixGPU<double,long long int>;
  template class FrontalMatrixGPU<std::complex<float>,long long int>;
  template class FrontalMatrixGPU<std::complex<double>,long long int>;

} // end namespace strumpack
