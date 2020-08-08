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

#include "FrontalMatrixGPU.hpp"
#include "FrontalMatrixGPUKernels.hpp"

#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#include "FrontalMatrixMPI.hpp"
#endif


namespace strumpack {

  uintptr_t round_to_8(uintptr_t p) { return (p + 7) & ~7; }
  uintptr_t round_to_8(void* p) {
    return round_to_8(reinterpret_cast<uintptr_t>(p));
  }

  template<typename scalar_t, typename integer_t> class LevelInfo {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FG_t = FrontalMatrixGPU<scalar_t,integer_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;

  public:
    LevelInfo() {}

    LevelInfo(const std::vector<F_t*>& fronts, gpu::SOLVERHandle& handle,
              int max_streams) {
      f.reserve(fronts.size());
      for (auto& F : fronts)
        f.push_back(dynamic_cast<FG_t*>(F));
      int max_dsep = 0;
      for (auto F : f) {
        auto dsep = F->dim_sep();
        auto dupd = F->dim_upd();
        factor_size += dsep*dsep + 2*dsep*dupd;
        schur_size += dupd*dupd;
        piv_size += dsep;
        if (dsep <= 8)       N8++;
        else if (dsep <= 16) N16++;
        else if (dsep <= 32) N32++;
        if (dsep > max_dsep) max_dsep = dsep;
      }
      getrf_work_size = gpu::getrf_buffersize<scalar_t>(handle, max_dsep);
      work_bytes =
        round_to_8(sizeof(scalar_t) *
                   (schur_size + getrf_work_size * max_streams)) +
        round_to_8(sizeof(int) * (piv_size + max_streams)) +
        round_to_8(sizeof(gpu::FrontData<scalar_t>) * (N8 + N16 + N32));
    }

    void print_info(int l, int lvls) {
      std::cout << "#  level " << l << " of " << lvls
                << " has " << f.size() << " nodes and "
                << N8 << " <=8, " << N16 << " <=16, "
                << N32 << " <=32, needs "
                << factor_size * sizeof(scalar_t) / 1.e6
                << " MB for factors, "
                << schur_size * sizeof(scalar_t) / 1.e6
                << " MB for Schur complements" << std::endl;
    }

    long long total_flops() {
      long long level_flops = 0;
      for (auto F : f) {
        level_flops += LU_flops(F->F11_) +
          gemm_flops(Trans::N, Trans::N, scalar_t(-1.),
                     F->F21_, F->F12_, scalar_t(1.)) +
          trsm_flops(Side::L, scalar_t(1.), F->F11_, F->F12_) +
          trsm_flops(Side::R, scalar_t(1.), F->F11_, F->F21_);
      }
      return level_flops;
    }

    void set_factor_pointers(scalar_t* factors) {
      for (auto F : f) {
        const int dsep = F->dim_sep();
        const int dupd = F->dim_upd();
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

    void set_work_pointers(void* wmem, int max_streams) {
      auto schur = reinterpret_cast<scalar_t*>(wmem);
      for (auto F : f) {
        const int dupd = F->dim_upd();
        if (dupd) {
          F->F22_ = DenseMW_t(dupd, dupd, schur, dupd);
          schur += dupd*dupd;
        }
      }
      dev_getrf_work = schur;
      schur += max_streams * getrf_work_size;
      auto imem = reinterpret_cast<int*>(round_to_8(schur));
      for (auto F : f) {
        F->piv_ = imem;
        imem += F->dim_sep();
      }
      dev_getrf_err = imem;
      imem += max_streams;
      auto fdat = reinterpret_cast<gpu::FrontData<scalar_t>*>
        (round_to_8(imem));
      f8 = fdat;   fdat += N8;
      f16 = fdat;  fdat += N16;
      f32 = fdat;  fdat += N32;
    }

    std::vector<FG_t*> f;
    std::size_t factor_size = 0, schur_size = 0, piv_size = 0,
      work_bytes = 0, N8 = 0, N16 = 0, N32 = 0;
    scalar_t* dev_getrf_work = nullptr;
    int* dev_getrf_err = nullptr;
    int getrf_work_size = 0;
    gpu::FrontData<scalar_t> *f8 = nullptr, *f16 = nullptr, *f32 = nullptr;
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


  /**
   * this doesn't count the memory used for the sparse matrix elements
   * needed in the front assembly.
   */
  template<typename scalar_t,typename integer_t>
  bool sufficient_device_memory
  (const std::vector<LevelInfo<scalar_t,integer_t>>& ldata) {
    std::size_t peak_device_mem = 0;
    for (std::size_t l=0; l<ldata.size(); l++) {
      auto& L = ldata[l];
      // memory needed on this level: factors,
      // schur updates, pivot vectors, cuSOLVER work space, ...
      std::size_t level_mem = L.factor_size*sizeof(scalar_t) + L.work_bytes;
      // the contribution blocks of the previous level are still
      // needed for the extend-add
      if (l+1 < ldata.size())
        level_mem += ldata[l+1].work_bytes;
      peak_device_mem = std::max(peak_device_mem, level_mem);
    }
    // only use 90% of available memory, since we're not counting the
    // sparse elements in the peak_device_mem
    return peak_device_mem < 0.9 * gpu::available_memory();
  }


  template<typename scalar_t, typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::front_assembly
  (const SpMat_t& A, LevelInfo<scalar_t,integer_t>& L) {
    using Trip_t = Triplet<scalar_t>;
    auto N = L.f.size();
    std::vector<Trip_t> e11, e12, e21;
    std::vector<std::array<std::size_t,3>> ne(N+1);
    std::size_t Isize = 0;
    for (std::size_t n=0; n<N; n++) {
      auto& f = *(L.f[n]);
      ne[n] = std::array<std::size_t,3>
        {e11.size(), e12.size(), e21.size()};
      A.push_front_elements
        (f.sep_begin_, f.sep_end_, f.upd_, e11, e12, e21);
      if (f.lchild_) Isize += f.lchild_->dim_upd();
      if (f.rchild_) Isize += f.rchild_->dim_upd();
    }
    ne[N] = std::array<std::size_t,3>
      {e11.size(), e12.size(), e21.size()};
    std::size_t ea_mem_size = N*sizeof(gpu::AssembleData<scalar_t>) +
      Isize*sizeof(std::size_t) +
      (e11.size()+e12.size()+e21.size())*sizeof(Trip_t);
    gpu::HostMemory<char> host_ea_mem(ea_mem_size);
    gpu::DeviceMemory<char> dev_ea_mem(ea_mem_size);
    auto asmbl = host_ea_mem.as<gpu::AssembleData<scalar_t>>();
    auto Imem = reinterpret_cast<std::size_t*>(asmbl + N);
    auto delems = reinterpret_cast<Trip_t*>(Imem + Isize);
    auto de11 = delems;
    auto de12 = de11 + e11.size();
    auto de21 = de12 + e12.size();
    std::copy(e11.begin(), e11.end(), de11);
    std::copy(e12.begin(), e12.end(), de12);
    std::copy(e21.begin(), e21.end(), de21);
    auto Iptr = Imem;
    for (std::size_t n=0; n<N; n++) {
      auto& f = *(L.f[n]);
      asmbl[n] = gpu::AssembleData<scalar_t>
        (f.dim_sep(), f.dim_upd(), f.F11_.data(), f.F12_.data(),
         f.F21_.data(), f.F22_.data(),
         ne[n+1][0]-ne[n][0], ne[n+1][1]-ne[n][1], ne[n+1][2]-ne[n][2],
         de11+ne[n][0], de12+ne[n][1], de21+ne[n][2]);
      if (f.lchild_) {
        auto c = dynamic_cast<FG_t*>(f.lchild_.get());
        asmbl[n].set_ext_add_left(c->dim_upd(), c->F22_.data(), Iptr);
        auto u = c->upd_to_parent(&f);
        std::copy(u.begin(), u.end(), Iptr);
        Iptr += u.size();
      }
      if (f.rchild_) {
        auto c = dynamic_cast<FG_t*>(f.rchild_.get());
        asmbl[n].set_ext_add_right(c->dim_upd(), c->F22_.data(), Iptr);
        auto u = c->upd_to_parent(&f);
        std::copy(u.begin(), u.end(), Iptr);
        Iptr += u.size();
      }
    }
    gpu::copy_host_to_device<char>(dev_ea_mem, host_ea_mem, ea_mem_size);
    gpu::assemble(N, asmbl);
    gpu::synchronize();
  }


  template<typename scalar_t, typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::factor_small_fronts
  (LInfo_t& L, gpu::FrontData<scalar_t>* front_data) {
    if (L.N8 || L.N16 || L.N32) {
      for (std::size_t n=0, n8=0, n16=L.N8, n32=L.N8+L.N16;
           n<L.f.size(); n++) {
        auto& f = *(L.f[n]);
        const auto dsep = f.dim_sep();
        if (dsep <= 32) {
          gpu::FrontData<scalar_t>
            t(dsep, f.dim_upd(), f.F11_.data(), f.F12_.data(),
              f.F21_.data(), f.F22_.data(), f.piv_);
          if (dsep <= 8)       front_data[n8++] = t;
          else if (dsep <= 16) front_data[n16++] = t;
          else                 front_data[n32++] = t;
        }
      }
      gpu::copy_host_to_device(L.f8, front_data, L.N8+L.N16+L.N32);
      gpu::factor_block_batch<scalar_t,8>(L.N8, L.f8);
      gpu::factor_block_batch<scalar_t,16>(L.N16, L.f16);
      gpu::factor_block_batch<scalar_t,32>(L.N32, L.f32);
    }
  }

  template<typename scalar_t, typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::factor_large_fronts
  (LInfo_t& L, std::vector<gpu::BLASHandle>& blas_handles,
   std::vector<gpu::SOLVERHandle>& solver_handles) {
    for (std::size_t n=0; n<L.f.size(); n++) {
      auto& f = *(L.f[n]);
      auto stream = n % solver_handles.size();
      const auto dsep = f.dim_sep();
      if (dsep > 32) {
        const auto dupd = f.dim_upd();
        gpu::getrf
          (solver_handles[stream], f.F11_,
           L.dev_getrf_work + stream * L.getrf_work_size,
           f.piv_, L.dev_getrf_err + stream);
        // TODO if (opts.replace_tiny_pivots()) { ...
        if (dupd) {
          gpu::getrs
            (solver_handles[stream], Trans::N, f.F11_, f.piv_,
             f.F12_, L.dev_getrf_err + stream);
          gpu::gemm
            (blas_handles[stream], Trans::N, Trans::N,
             scalar_t(-1.), f.F21_, f.F12_, scalar_t(1.), f.F22_);
        }
      }
    }
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

    factor_mem_.reset(new scalar_t[dsep*(dsep+2*dupd)]);
    host_Schur_.reset(new scalar_t[dupd*dupd]);
    {
      auto fmem = factor_mem_.get();
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

      // gpu::SOLVERHandle sh;
      // gpu::DeviceMemory<scalar_t> dmem
      //   ((dsep+dupd)*(dsep+dupd) + gpu::getrf_buffersize<scalar_t>(sh, dsep));
      // gpu::DeviceMemory<int> dpiv(dsep+1); // and ierr
      // // TODO more overlapping??
      // gpu::copy_host_to_device
      //   (dmem.template as<scalar_t>(), factor_mem_.get(), dsep*(dsep+2*dupd));
      // gpu::copy_host_to_device
      //   (dmem+dsep*(dsep+2*dupd), host_Schur_.get(), dupd*dupd);
      // DenseMW_t dF11(dsep, dsep, dmem, dsep);
      // DenseMW_t dF12(dsep, dupd, dmem+dsep*dsep, dsep);
      // DenseMW_t dF21(dupd, dsep, dmem+dsep*(dsep+dupd), dupd);
      // DenseMW_t dF22(dupd, dupd, dmem+dsep*(dsep+2*dupd), dupd);
      // scalar_t* getrf_work = dmem+(dsep+dupd)*(dsep+dupd);
      // gpu::getrf(sh, dF11, getrf_work, dpiv, dpiv+dsep);
      // // TODO if (opts.replace_tiny_pivots()) { ...
      // if (dupd) {
      //   gpu::getrs(sh, Trans::N, dF11, dpiv, dF12, dpiv+dsep);
      //   gpu::BLASHandle bh;
      //   gpu::gemm(bh, Trans::N, Trans::N, scalar_t(-1.),
      //             dF21, dF12, scalar_t(1.), dF22);
      // }
      // pivot_mem_.resize(dsep);
      // piv_ = pivot_mem_.data();
      // gpu::copy_device_to_host(piv_, dpiv.as<int>(), dsep);
      // gpu::copy_device_to_host
      //   (factor_mem_.get(), dmem.template as<scalar_t>(), dsep*(dsep+dupd));
      // gpu::copy_device_to_host
      //   (host_Schur_.get(), dmem+dsep*(dsep+2*dupd), dupd*dupd);
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
    using LevelInfo_t = LevelInfo<scalar_t,integer_t>;
    const int max_streams = opts.cuda_streams();
    std::vector<gpu::SOLVERHandle> solver_handles(max_streams);
    const int lvls = this->levels();
    std::vector<LevelInfo_t> ldata(lvls);
    std::size_t max_small_fronts = 0;
    for (int l=lvls-1; l>=0; l--) {
      std::vector<F_t*> fp;
      fp.reserve(std::pow(2, l));
      this->get_level_fronts(fp, l);
      auto& L = ldata[l];
      L = LevelInfo_t(fp, solver_handles[0], max_streams);
      max_small_fronts = std::max(max_small_fronts, L.N8+L.N16+L.N32);
    }
    if (!sufficient_device_memory(ldata)) {
      split_smaller(A, opts, etree_level, task_depth);
      return;
    }
    std::vector<gpu::Stream> streams(max_streams);
    std::vector<gpu::BLASHandle> blas_handles(max_streams);
    for (int i=0; i<max_streams; i++) {
      blas_handles[i].set_stream(streams[i]);
      solver_handles[i].set_stream(streams[i]);
    }
    gpu::HostMemory<gpu::FrontData<scalar_t>> front_data(max_small_fronts);
    gpu::DeviceMemory<char> old_work;
    for (int l=lvls-1; l>=0; l--) {
      TaskTimer tl("");
      tl.start();
      auto& L = ldata[l];
      if (opts.verbose()) L.print_info(l, lvls);
      try {
        gpu::DeviceMemory<scalar_t> dev_factors(L.factor_size);
        gpu::DeviceMemory<char> work_mem(L.work_bytes);
        gpu::memset<scalar_t>(dev_factors, 0, L.factor_size);
        gpu::memset<scalar_t>(work_mem.as<scalar_t>(), 0, L.schur_size);
        L.set_factor_pointers(dev_factors);
        L.set_work_pointers(work_mem, max_streams);
        front_assembly(A, L);
        old_work = std::move(work_mem);
        factor_small_fronts(L, front_data);
        factor_large_fronts(L, blas_handles, solver_handles);
        STRUMPACK_ADD_MEMORY(L.factor_size*sizeof(scalar_t));
        L.f[0]->factor_mem_.reset(new scalar_t[L.factor_size]);
        L.f[0]->pivot_mem_.resize(L.piv_size);
        gpu::synchronize();
        gpu::copy_device_to_host_async<scalar_t>
          (L.f[0]->factor_mem_.get(), dev_factors,
           L.factor_size, streams[1 % streams.size()]);
        gpu::copy_device_to_host_async
          (L.f[0]->pivot_mem_.data(), L.f[0]->piv_, L.piv_size, streams[0]);
        L.set_factor_pointers(L.f[0]->factor_mem_.get());
        L.set_pivot_pointers(L.f[0]->pivot_mem_.data());
        gpu::synchronize();
      } catch (const std::bad_alloc& e) {
        std::cerr << "Out of memory" << std::endl;
        abort();
      }
      auto level_flops = L.total_flops();
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
    const std::size_t dupd = dim_upd();
    if (dupd) { // get the contribution block from the device
      host_Schur_.reset(new scalar_t[dupd*dupd]);
      gpu::copy_device_to_host
        (host_Schur_.get(), old_work.as<scalar_t>(), dupd*dupd);
      F22_ = DenseMW_t(dupd, dupd, host_Schur_.get(), dupd);
    }
  }

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

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
      //bloc.laswp(piv, true);
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

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::extract_CB_sub_matrix
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

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::sample_CB
  (const SPOptions<scalar_t>& opts, const DenseM_t& R, DenseM_t& Sr,
   DenseM_t& Sc, F_t* pa, int task_depth) {
    auto I = this->upd_to_parent(pa);
    auto cR = R.extract_rows(I);
    DenseM_t cS(dim_upd(), R.cols());
    TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
    gemm(Trans::N, Trans::N, scalar_t(1.), F22_, cR,
         scalar_t(0.), cS, task_depth);
    TIMER_STOP(t_f22mult);
    Sr.scatter_rows_add(I, cS, task_depth);
    TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult2);
    gemm(Trans::C, Trans::N, scalar_t(1.), F22_, cR,
         scalar_t(0.), cS, task_depth);
    TIMER_STOP(t_f22mult2);
    Sc.scatter_rows_add(I, cS, task_depth);
    STRUMPACK_CB_SAMPLE_FLOPS
      (gemm_flops(Trans::N, Trans::N, scalar_t(1.), F22_, cR, scalar_t(0.)) +
       gemm_flops(Trans::C, Trans::N, scalar_t(1.), F22_, cR, scalar_t(0.)) +
       cS.rows()*cS.cols()*2); // for the skinny-extend add
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::sample_CB
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    auto I = this->upd_to_parent(pa);
    auto cR = R.extract_rows(I);
    DenseM_t cS(dim_upd(), R.cols());
    TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
    gemm(op, Trans::N, scalar_t(1.), F22_, cR,
         scalar_t(0.), cS, task_depth);
    TIMER_STOP(t_f22mult);
    S.scatter_rows_add(I, cS, task_depth);
    STRUMPACK_CB_SAMPLE_FLOPS
      (gemm_flops(op, Trans::N, scalar_t(1.), F22_, cR, scalar_t(0.)) +
       cS.rows()*cS.cols()); // for the skinny-extend add
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::sample_CB_to_F11
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    std::size_t u2s;
    auto Ir = this->upd_to_parent(pa, u2s);
    auto Rcols = R.cols();
    DenseM_t cR(u2s, Rcols);
    for (std::size_t c=0; c<Rcols; c++)
      for (std::size_t r=0; r<u2s; r++)
        cR(r,c) = R(Ir[r],c);
    DenseM_t cS(u2s, Rcols);
    DenseMW_t CB11(u2s, u2s, const_cast<DenseMW_t&>(F22_), 0, 0);
    gemm(op, Trans::N, scalar_t(1.), CB11, cR, scalar_t(0.), cS, task_depth);
    for (std::size_t c=0; c<Rcols; c++)
      for (std::size_t r=0; r<u2s; r++)
        S(Ir[r],c) += cS(r,c);
    STRUMPACK_CB_SAMPLE_FLOPS(u2s*Rcols);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::sample_CB_to_F12
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    std::size_t u2s;
    auto Ir = this->upd_to_parent(pa, u2s);
    auto pds = pa->dim_sep();
    auto Rcols = R.cols();
    DenseMW_t CB12(u2s, dupd-u2s, const_cast<DenseMW_t&>(F22_), 0, u2s);
    if (op == Trans::N) {
      DenseM_t cR(dupd-u2s, Rcols);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=u2s; r<dupd; r++)
          cR(r-u2s,c) = R(Ir[r]-pds,c);
      DenseM_t cS(u2s, Rcols);
      gemm(op, Trans::N, scalar_t(1.), CB12, cR, scalar_t(0.), cS, task_depth);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=0; r<u2s; r++)
          S(Ir[r],c) += cS(r,c);
      STRUMPACK_CB_SAMPLE_FLOPS(u2s*Rcols);
    } else {
      DenseM_t cR(u2s, Rcols);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=0; r<u2s; r++)
          cR(r,c) = R(Ir[r],c);
      DenseM_t cS(dupd-u2s, Rcols);
      gemm(op, Trans::N, scalar_t(1.), CB12, cR, scalar_t(0.), cS, task_depth);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=u2s; r<dupd; r++)
          S(Ir[r]-pds,c) += cS(r-u2s,c);
      STRUMPACK_CB_SAMPLE_FLOPS((dupd-u2s)*Rcols);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::sample_CB_to_F21
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    std::size_t u2s;
    auto Ir = this->upd_to_parent(pa, u2s);
    auto Rcols = R.cols();
    auto pds = pa->dim_sep();
    DenseMW_t CB21(dupd-u2s, u2s, const_cast<DenseMW_t&>(F22_), u2s, 0);
    if (op == Trans::N) {
      DenseM_t cR(u2s, Rcols);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=0; r<u2s; r++)
          cR(r,c) = R(Ir[r],c);
      DenseM_t cS(dupd-u2s, Rcols);
      gemm(op, Trans::N, scalar_t(1.), CB21, cR, scalar_t(0.), cS, task_depth);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=u2s; r<dupd; r++)
          S(Ir[r]-pds,c) += cS(r-u2s,c);
      STRUMPACK_CB_SAMPLE_FLOPS((dupd-u2s)*Rcols);
    } else {
      DenseM_t cR(dupd-u2s, Rcols);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=u2s; r<dupd; r++)
          cR(r-u2s,c) = R(Ir[r]-pds,c);
      DenseM_t cS(u2s, Rcols);
      gemm(op, Trans::N, scalar_t(1.), CB21, cR, scalar_t(0.), cS, task_depth);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=0; r<u2s; r++)
          S(Ir[r],c) += cS(r,c);
      STRUMPACK_CB_SAMPLE_FLOPS(u2s*Rcols);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixGPU<scalar_t,integer_t>::sample_CB_to_F22
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    std::size_t u2s;
    auto Ir = this->upd_to_parent(pa, u2s);
    auto pds = pa->dim_sep();
    auto Rcols = R.cols();
    DenseM_t cR(dupd-u2s, Rcols);
    for (std::size_t c=0; c<Rcols; c++)
      for (std::size_t r=u2s; r<dupd; r++)
        cR(r-u2s,c) = R(Ir[r]-pds,c);
    DenseM_t cS(dupd-u2s, Rcols);
    DenseMW_t CB22(dupd-u2s, dupd-u2s, const_cast<DenseMW_t&>(F22_), u2s, u2s);
    gemm(op, Trans::N, scalar_t(1.), CB22, cR, scalar_t(0.), cS, task_depth);
    for (std::size_t c=0; c<Rcols; c++)
      for (std::size_t r=u2s; r<dupd; r++)
        S(Ir[r]-pds,c) += cS(r-u2s,c);
    STRUMPACK_CB_SAMPLE_FLOPS((dupd-u2s)*Rcols);
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
