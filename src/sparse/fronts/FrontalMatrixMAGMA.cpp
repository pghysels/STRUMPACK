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

#include "FrontalMatrixMAGMA.hpp"
#include "FrontalMatrixGPUKernels.hpp"
#include "dense/MAGMAWrapper.hpp"

#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#include "FrontalMatrixMPI.hpp"
#endif


namespace strumpack {

  template<typename scalar_t, typename integer_t> class LevelInfoMAGMA {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FM_t = FrontalMatrixMAGMA<scalar_t,integer_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
  public:
    LevelInfoMAGMA() {}
    LevelInfoMAGMA(const std::vector<F_t*>& fronts,
                   gpu::SOLVERHandle& handle,
                   const SpMat_t* A=nullptr) {
      auto N = fronts.size();
      f.reserve(N);
      for (auto& F : fronts)
        f.push_back(dynamic_cast<FM_t*>(F));
      std::size_t max_dsep = 0;
#pragma omp parallel for                                                \
  reduction(max:max_dsep)                                               \
  reduction(+:factor_size,Schur_size,piv_size)                          \
  reduction(+:total_upd_size)
      for (std::size_t i=0; i<N; i++) {
        const std::size_t dsep = f[i]->dim_sep();
        const std::size_t dupd = f[i]->dim_upd();
        factor_size += dsep*(dsep + 2*dupd);
        Schur_size += dupd*dupd;
        piv_size += dsep;
        total_upd_size += dupd;
        if (dsep > max_dsep) max_dsep = dsep;
      }
      Nsmall = (N > MIN_BATCH_COUNT) ? N : 0;
      if (A) {
        elems11.resize(N+1);
        elems12.resize(N+1);
        elems21.resize(N+1);
#pragma omp parallel for
        for (std::size_t i=0; i<N; i++) {
          auto& F = *(f[i]);
          A->count_front_elements
            (F.sep_begin(), F.sep_end(), F.upd(),
             elems11[i+1], elems12[i+1], elems21[i+1]);
        }
        for (std::size_t i=0; i<N; i++) {
          elems11[i+1] += elems11[i];
          elems12[i+1] += elems12[i];
          elems21[i+1] += elems21[i];
        }
      }
      Isize.resize(N+1);
      for (std::size_t i=0; i<N; i++) {
        auto& F = *(f[i]);
        Isize[i+1] = Isize[i];
        if (F.lchild_) Isize[i+1] += F.lchild_->dim_upd();
        if (F.rchild_) Isize[i+1] += F.rchild_->dim_upd();
      }
      batch_meta_bytes = gpu::round_up(4*(Nsmall+1)*sizeof(int));
      batch_meta_bytes += gpu::round_up(4*Nsmall*sizeof(scalar_t*));
      batch_meta_bytes += gpu::round_up(Nsmall*sizeof(int*));
      dev_batch_meta = gpu::HostMemory<char>(batch_meta_bytes);
      d1_batch = dev_batch_meta.template as<int>();
      d2_batch = d1_batch + Nsmall + 1;
      ld1_batch = d2_batch + Nsmall + 1;
      ld2_batch = ld1_batch + Nsmall + 1;
      F_batch = gpu::aligned_ptr<scalar_t*>(d1_batch+4*(Nsmall+1));
      ipiv_batch = gpu::aligned_ptr<int*>(F_batch+4*Nsmall);
#pragma omp parallel for                        \
  reduction(max:max_d1_small, max_d2_small)
      for (std::size_t i=0; i<Nsmall; i++) {
        const int dsep = f[i]->dim_sep();
        const int dupd = f[i]->dim_upd();
        d1_batch[i] = dsep;
        d2_batch[i] = dupd;
        max_d1_small = std::max(max_d1_small, dsep);
        max_d2_small = std::max(max_d2_small, dupd);
        ld1_batch[i] = std::max(1, dsep);
        ld2_batch[i] = std::max(1, dupd);
      }

      if (A) { // not needed for the solve
        int getrf_work_cusolver =
          sizeof(scalar_t) * gpu::getrf_buffersize<scalar_t>(handle, max_dsep);
        getrf_work_bytes = -1;
        magma_queue_t magma_q = nullptr;
        gpu::magma::getrf_vbatched_max_nocheck_work
          (nullptr, nullptr, max_d1_small, max_d1_small,
           max_d1_small, max_d1_small*max_d1_small,
           (scalar_t**)nullptr, nullptr, nullptr, nullptr, nullptr,
           &getrf_work_bytes, Nsmall, magma_q);
        getrf_work_bytes = std::max(getrf_work_bytes, getrf_work_cusolver);
        factor_bytes = gpu::round_up(sizeof(scalar_t)*factor_size);
        factor_bytes += gpu::round_up(sizeof(int)*piv_size);
      }
      work_bytes = batch_meta_bytes;
      if (A) { // not needed for the solve
        work_bytes += gpu::round_up(sizeof(int)*N);
        work_bytes += gpu::round_up(sizeof(scalar_t)*Schur_size);
        work_bytes += gpu::round_up(getrf_work_bytes);
      }
      ea_bytes = gpu::round_up(sizeof(gpu::AssembleData<scalar_t>)*f.size());
      ea_bytes += gpu::round_up(sizeof(std::size_t)*Isize.back());
      if (A)
        ea_bytes += gpu::round_up
          (sizeof(Triplet<scalar_t>) *
           (elems11.back()+elems12.back()+elems21.back()));
    }

    void print_info(int l, int lvls) {
      std::cout << "#  level " << l << " of " << lvls
                << " has " << f.size() << " nodes and "
                << Nsmall << " small fronts, needs "
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
        if (f.size() > MIN_BATCH_COUNT)
          small_flops += flops;
      }
    }

    void set_factor_pointers(void* factors, void* pivots=nullptr) {
      auto fmem = static_cast<scalar_t*>(factors);
      for (auto F : f) {
        const std::size_t dsep = F->dim_sep();
        const std::size_t dupd = F->dim_upd();
        F->F11_ = DenseMW_t(dsep, dsep, fmem, dsep); fmem += dsep*dsep;
        F->F12_ = DenseMW_t(dsep, dupd, fmem, dsep); fmem += dsep*dupd;
        F->F21_ = DenseMW_t(dupd, dsep, fmem, dupd); fmem += dupd*dsep;
      }
      int* pmem = nullptr;
      if (pivots) pmem = static_cast<int*>(pivots);
      else pmem = gpu::aligned_ptr<int>(fmem);
      for (auto F : f) {
        F->piv_ = pmem;
        pmem += F->dim_sep();
      }
    }

    void set_work_pointers(void* wmem) {
      auto schur = gpu::aligned_ptr<scalar_t>(wmem);
      for (auto F : f) {
        auto dupd = F->dim_upd();
        if (dupd) {
          F->F22_ = DenseMW_t(dupd, dupd, schur, dupd);
          schur += dupd*dupd;
        }
      }
      auto gmem = gpu::aligned_ptr<char>(schur);
      dev_getrf_work = gmem;   gmem += getrf_work_bytes;
      auto N = f.size();
      auto imem = gpu::aligned_ptr<int>(gmem);
      dev_getrf_err = imem;    imem += N;
      set_batch_data(imem);
    }

    void set_batch_data(void* mem) {
      auto imem = gpu::aligned_ptr<int>(mem);
      dev_d1_batch  = imem;    imem += Nsmall+1;
      dev_d2_batch  = imem;    imem += Nsmall+1;
      dev_ld1_batch = imem;    imem += Nsmall+1;
      dev_ld2_batch = imem;    imem += Nsmall+1;
      auto fmem = gpu::aligned_ptr<scalar_t*>(imem);
      dev_F_batch = fmem;      fmem += 4 * Nsmall;
      auto ipmem = gpu::aligned_ptr<int*>(fmem);
      dev_ipiv_batch = ipmem;  ipmem += Nsmall;
#pragma omp parallel for
      for (std::size_t i=0; i<Nsmall; i++) {
        F_batch[i           ] = f[i]->F11_.data();
        F_batch[i +   Nsmall] = f[i]->F12_.data();
        F_batch[i + 2*Nsmall] = f[i]->F21_.data();
        F_batch[i + 3*Nsmall] = f[i]->F22_.data();
        ipiv_batch[i] = f[i]->piv_;
      }
      gpu_check(gpu::copy_host_to_device<char>
                (gpu::aligned_ptr<char>(mem), dev_batch_meta,
                 batch_meta_bytes));
    }

    // static const int FRONT_SMALL = 10000000;
    static const int MIN_BATCH_COUNT = 8;
    std::vector<FM_t*> f;
    std::size_t factor_size = 0, Schur_size = 0, piv_size = 0,
      total_upd_size = 0, work_bytes, ea_bytes,
      factor_bytes, Nsmall = 0;
    std::vector<std::size_t> elems11, elems12, elems21, Isize;
    int max_d1_small = 0, max_d2_small = 0;
    char* dev_getrf_work = nullptr;
    int getrf_work_bytes = 0;
    int* dev_getrf_err = nullptr;
    std::size_t batch_meta_bytes = 0;
    gpu::HostMemory<char> dev_batch_meta;
    scalar_t** F_batch = nullptr;
    int** ipiv_batch = nullptr;
    int *d1_batch = nullptr, *d2_batch = nullptr,
      *ld1_batch = nullptr, *ld2_batch = nullptr;
    scalar_t **dev_F_batch = nullptr;
    int **dev_ipiv_batch = nullptr;
    int *dev_d1_batch = nullptr, *dev_d2_batch = nullptr,
      *dev_ld1_batch = nullptr, *dev_ld2_batch = nullptr;
  };


  template<typename scalar_t,typename integer_t>
  FrontalMatrixMAGMA<scalar_t,integer_t>::FrontalMatrixMAGMA
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t>
  FrontalMatrixMAGMA<scalar_t,integer_t>::~FrontalMatrixMAGMA() {
#if defined(STRUMPACK_COUNT_FLOPS)
    const std::size_t dupd = dim_upd();
    const std::size_t dsep = dim_sep();
    STRUMPACK_SUB_MEMORY(dsep*(dsep+2*dupd)*sizeof(scalar_t));
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMAGMA<scalar_t,integer_t>::release_work_memory() {
    F22_.clear();
    host_Schur_.release();
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMAGMA<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf,
   const FrontalMatrixMPI<scalar_t,integer_t>* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_seq_copy_to_buffers
      (F22_, sbuf, pa, this);
  }
#endif

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMAGMA<scalar_t,integer_t>::extend_add_to_dense
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

  template<typename scalar_t, typename integer_t> void
  FrontalMatrixMAGMA<scalar_t,integer_t>::front_assembly
  (const SpMat_t& A, LInfo_t& L, char* hea_mem, char* dea_mem) {
    using Trip_t = Triplet<scalar_t>;
    auto N = L.f.size();
    auto hasmbl = gpu::aligned_ptr<gpu::AssembleData<scalar_t>>(hea_mem);
    auto Iptr   = gpu::aligned_ptr<std::size_t>(hasmbl + N);
    auto e11    = gpu::aligned_ptr<Trip_t>(Iptr + L.Isize.back());
    auto e12    = e11 + L.elems11.back();
    auto e21    = e12 + L.elems12.back();
    auto dasmbl = gpu::aligned_ptr<gpu::AssembleData<scalar_t>>(dea_mem);
    auto dIptr  = gpu::aligned_ptr<std::size_t>(dasmbl + N);
    auto de11   = gpu::aligned_ptr<Trip_t>(dIptr + L.Isize.back());
    auto de12   = de11 + L.elems11.back();
    auto de21   = de12 + L.elems12.back();

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
        auto c = dynamic_cast<FM_t*>(f.lchild_.get());
        hasmbl[n].set_ext_add_left(c->dim_upd(), c->F22_.data(), fdIptr);
        c->upd_to_parent(&f, fIptr);
        fIptr += c->dim_upd();
        fdIptr += c->dim_upd();
      }
      if (f.rchild_) {
        auto c = dynamic_cast<FM_t*>(f.rchild_.get());
        hasmbl[n].set_ext_add_right(c->dim_upd(), c->F22_.data(), fdIptr);
        c->upd_to_parent(&f, fIptr);
      }
    }
    gpu_check(gpu::copy_host_to_device<char>
              (dea_mem, hea_mem, L.ea_bytes));
    gpu::assemble(N, hasmbl, dasmbl);
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontalMatrixMAGMA<scalar_t,integer_t>::factors_on_device
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   std::vector<LevelInfoMAGMA<scalar_t,integer_t>>& ldata,
   std::size_t total_dmem) {
    ReturnCode err_code = ReturnCode::SUCCESS;
    int lvls = ldata.size();
    gpu::Stream comp_stream, copy_stream;
    gpu::BLASHandle blas_handle(comp_stream);
    gpu::SOLVERHandle solver_handle(comp_stream);
    gpu::magma::MAGMAQueue magma_q(comp_stream, blas_handle);
    std::size_t peak_ea_hmem = 0, total_factor_dmem = 0,
      peak_work_dmem = 0;
    for (int l=0; l<lvls; l++) {
      peak_ea_hmem = std::max(peak_ea_hmem, ldata[l].ea_bytes);
      std::size_t level_mem = ldata[l].work_bytes + ldata[l].ea_bytes;
      if (l+1 < lvls)
        level_mem += ldata[l+1].work_bytes;
      peak_work_dmem = std::max(peak_work_dmem, level_mem);
      total_factor_dmem += ldata[l].factor_bytes;
    }
    gpu::HostMemory<char> ea_hmem(peak_ea_hmem);
    gpu::DeviceMemory<char> work_dmem(peak_work_dmem);
    dev_factors_.reset(new gpu::DeviceMemory<char>(total_factor_dmem));
    char* factors_dptr = dev_factors_->as<char>();
    for (int l=lvls-1; l>=0; l--) {
      // TaskTimer tl("");
      // tl.start();
      auto& L = ldata[l];
      // if (opts.verbose()) L.print_info(l, lvls);
      try {
        char *work_dptr = nullptr, *ea_dptr = nullptr;
        if (l % 2) {
          work_dptr = work_dmem;
          ea_dptr = work_dptr + L.work_bytes;
        } else {
          work_dptr = work_dmem + peak_work_dmem - L.work_bytes;
          ea_dptr = work_dptr - L.ea_bytes;
        }
        gpu_check(gpu::memset<scalar_t>(work_dptr, 0, L.Schur_size));
        gpu_check(gpu::memset<scalar_t>(factors_dptr, 0, L.factor_size));
        L.set_factor_pointers(factors_dptr);
        factors_dptr += L.factor_bytes;
        L.set_work_pointers(work_dptr);

        // default stream
        front_assembly(A, L, ea_hmem, ea_dptr);

        std::size_t N = L.f.size(), Nsmall = L.Nsmall;
        auto d1 = L.dev_d1_batch;   auto d2 = L.dev_d2_batch;
        auto ld1 = L.dev_ld1_batch; auto ld2 = L.dev_ld2_batch;
        auto F11 = L.dev_F_batch;   auto F12 = F11 + Nsmall;
        auto F21 = F12 + Nsmall;    auto F22 = F21 + Nsmall;

        gpu::magma::getrf_vbatched_max_nocheck_work
          (d1, d1, L.max_d1_small, L.max_d1_small, L.max_d1_small,
           L.max_d1_small*L.max_d1_small, F11, ld1, L.dev_ipiv_batch,
           L.dev_getrf_err, L.dev_getrf_work, &L.getrf_work_bytes,
           Nsmall, magma_q);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::getrf
            (solver_handle, L.f[i]->F11_, (scalar_t*)L.dev_getrf_work,
             L.f[i]->piv_, L.dev_getrf_err+i);
        // TODO check error code
        if (opts.replace_tiny_pivots()) {
          gpu::replace_pivots_vbatched
            (blas_handle, d1, L.max_d1_small, F11, ld1,
             opts.pivot_threshold(), Nsmall);
          for (std::size_t i=Nsmall; i<N; i++)
            gpu::replace_pivots
              (L.f[i]->F11_.rows(), L.f[i]->F11_.data(),
               opts.pivot_threshold());
        }
        gpu::laswp_fwd_vbatched
          (blas_handle, d2, L.max_d2_small, F12, ld1, L.dev_ipiv_batch,
           d1, Nsmall);
        gpu::magma::trsm_vbatched_max_nocheck
          (MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
           L.max_d1_small, L.max_d2_small, d1, d2, scalar_t(1.),
           F11, ld1, F12, ld1, Nsmall, magma_q);
        gpu::magma::trsm_vbatched_max_nocheck
          (MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
           L.max_d1_small, L.max_d2_small, d1, d2, scalar_t(1.),
           F11, ld1, F12, ld1, Nsmall, magma_q);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::getrs
            (solver_handle, Trans::N, L.f[i]->F11_, L.f[i]->piv_,
             L.f[i]->F12_, L.dev_getrf_err+i);
        gpu::magma::gemm_vbatched_max_nocheck
          (MagmaNoTrans, MagmaNoTrans, d2, d2, d1, scalar_t(-1.),
           F21, ld2, F12, ld1, scalar_t(1.), F22, ld2, Nsmall,
           L.max_d2_small, L.max_d2_small, L.max_d1_small, magma_q);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::gemm
            (blas_handle, Trans::N, Trans::N, scalar_t(-1.),
             L.f[i]->F21_, L.f[i]->F12_, scalar_t(1.), L.f[i]->F22_);
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
    return err_code;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontalMatrixMAGMA<scalar_t,integer_t>::split_smaller
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (opts.verbose())
      std::cout << "# Factorization does not fit in GPU memory, "
        "splitting in smaller traversals." << std::endl;
    const std::size_t dupd = dim_upd(), dsep = dim_sep();
    ReturnCode err_code = ReturnCode::SUCCESS;
    if (lchild_) {
      auto el = lchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
      if (el != ReturnCode::SUCCESS) err_code = el;
    }
    if (rchild_) {
      auto er = rchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
      if (er != ReturnCode::SUCCESS) err_code = er;
    }
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
    // TaskTimer tl("");
    // tl.start();
    if (dsep) {
      gpu::SOLVERHandle sh;
      gpu::DeviceMemory<scalar_t> dm11
        (dsep*dsep + gpu::getrf_buffersize<scalar_t>(sh, dsep));
      gpu::DeviceMemory<int> dpiv(dsep+1); // and ierr
      DenseMW_t dF11(dsep, dsep, dm11, dsep);
      gpu_check(gpu::copy_host_to_device(dF11, F11_));
      gpu::getrf(sh, dF11, dm11 + dsep*dsep, dpiv, dpiv+dsep);
      // TODO check return code!
      if (opts.replace_tiny_pivots())
        gpu::replace_pivots
          (F11_.rows(), dF11.data(), opts.pivot_threshold());
      pivot_mem_.resize(dsep);
      piv_ = pivot_mem_.data();
      gpu_check(gpu::copy_device_to_host(piv_, dpiv.as<int>(), dsep));
      gpu_check(gpu::copy_device_to_host(F11_, dF11));
      if (dupd) {
        gpu::DeviceMemory<scalar_t> dm12(dsep*dupd);
        DenseMW_t dF12(dsep, dupd, dm12, dsep);
        gpu_check(gpu::copy_host_to_device(dF12, F12_));
        gpu::getrs(sh, Trans::N, dF11, dpiv, dF12, dpiv+dsep);
        gpu_check(gpu::copy_device_to_host(F12_, dF12));
        dm11.release();
        gpu::DeviceMemory<scalar_t> dm2122((dsep+dupd)*dupd);
        DenseMW_t dF21(dupd, dsep, dm2122, dupd);
        DenseMW_t dF22(dupd, dupd, dm2122+(dsep*dupd), dupd);
        gpu_check(gpu::copy_host_to_device(dF21, F21_));
        gpu_check(gpu::copy_host_to_device(dF22, host_Schur_.get()));
        gpu::BLASHandle bh;
        gpu::gemm(bh, Trans::N, Trans::N, scalar_t(-1.),
                  dF21, dF12, scalar_t(1.), dF22);
        gpu_check(gpu::copy_device_to_host(host_Schur_.get(), dF22));
      }
    }
    // count flops
    auto level_flops = LU_flops(F11_) +
      gemm_flops(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.)) +
      trsm_flops(Side::L, scalar_t(1.), F11_, F12_) +
      trsm_flops(Side::R, scalar_t(1.), F11_, F21_);
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


  // check the device memory required for doing it level by level when
  // moving the factors back to the host after each level
  template<typename scalar_t,typename integer_t>
  std::size_t level_peak_device_memory_MAGMA
  (const std::vector<LevelInfoMAGMA<scalar_t,integer_t>>& ldata) {
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

  // check if we can keep everything on the device, so the factors
  // reside on the device and can be used in the solve
  template<typename scalar_t,typename integer_t>
  std::size_t total_peak_device_memory_MAGMA
  (const std::vector<LevelInfoMAGMA<scalar_t,integer_t>>& ldata) {
    std::size_t peak_dmem = 0;
    for (std::size_t l=0; l<ldata.size(); l++) {
      std::size_t level_mem = ldata[l].work_bytes + ldata[l].ea_bytes;
      if (l+1 < ldata.size())
        level_mem += ldata[l+1].work_bytes;
      peak_dmem = std::max(peak_dmem, level_mem);
    }
    for (std::size_t l=0; l<ldata.size(); l++)
      peak_dmem += ldata[l].factor_bytes;
    return peak_dmem;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontalMatrixMAGMA<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    ReturnCode err_code = ReturnCode::SUCCESS;
    gpu::Stream comp_stream, copy_stream;
    gpu::SOLVERHandle solver_handle(comp_stream);
    const int lvls = this->levels();
    std::vector<LInfo_t> ldata(lvls);
    for (int l=lvls-1; l>=0; l--) {
      std::vector<F_t*> fp;
      this->get_level_fronts(fp, l);
      ldata[l] = LInfo_t(fp, solver_handle, &A);
    }

#if 1 // enable for solve on GPU (if factors fit on device)
    auto total_dmem = total_peak_device_memory_MAGMA(ldata);
    if (opts.verbose())
      std::cout << "# Need " << total_dmem / 1.e6 << " MB "
                << "of device mem, "
                << gpu::available_memory() / 1.e6
                << " MB available" << std::endl;
    if (etree_level == 0 &&
        total_dmem < 0.9 * gpu::available_memory())
      return factors_on_device(A, opts, ldata, total_dmem);
#endif

    auto peak_dmem = level_peak_device_memory_MAGMA(ldata);
    if (peak_dmem >= 0.9 * gpu::available_memory())
      return split_smaller(A, opts, etree_level, task_depth);

    gpu::BLASHandle blas_handle(comp_stream);
    gpu::magma::MAGMAQueue magma_q(comp_stream, blas_handle);

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

    for (int l=lvls-1; l>=0; l--) {
      // TaskTimer tl("");
      // tl.start();
      auto& L = ldata[l];
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
        gpu_check(gpu::memset<scalar_t>(work_mem, 0, L.Schur_size));
        gpu_check(gpu::memset<scalar_t>(dev_factors, 0, L.factor_size));
        L.set_factor_pointers(dev_factors);
        L.set_work_pointers(work_mem);
        old_work = work_mem;

        // default stream
        front_assembly(A, L, hea_mem, dea_mem);

        std::size_t N = L.f.size(), Nsmall = L.Nsmall;
        auto d1 = L.dev_d1_batch;   auto d2 = L.dev_d2_batch;
        auto ld1 = L.dev_ld1_batch; auto ld2 = L.dev_ld2_batch;
        auto F11 = L.dev_F_batch;   auto F12 = F11 + Nsmall;
        auto F21 = F12 + Nsmall;    auto F22 = F21 + Nsmall;

        gpu::magma::getrf_vbatched_max_nocheck_work
          (d1, d1, L.max_d1_small, L.max_d1_small, L.max_d1_small,
           L.max_d1_small*L.max_d1_small, F11, ld1, L.dev_ipiv_batch,
           L.dev_getrf_err, L.dev_getrf_work, &L.getrf_work_bytes,
           Nsmall, magma_q);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::getrf
            (solver_handle, L.f[i]->F11_, (scalar_t*)L.dev_getrf_work,
             L.f[i]->piv_, L.dev_getrf_err+i);
        // TODO check error code
        if (opts.replace_tiny_pivots()) {
          gpu::replace_pivots_vbatched
            (blas_handle, d1, L.max_d1_small, F11, ld1,
             opts.pivot_threshold(), Nsmall);
          for (std::size_t i=Nsmall; i<N; i++)
            gpu::replace_pivots
              (L.f[i]->F11_.rows(), L.f[i]->F11_.data(),
               opts.pivot_threshold());
        }
        gpu::laswp_fwd_vbatched
          (blas_handle, d2, L.max_d2_small, F12, ld1, L.dev_ipiv_batch,
           d1, Nsmall);
        gpu::magma::trsm_vbatched_max_nocheck
          (MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
           L.max_d1_small, L.max_d2_small, d1, d2, scalar_t(1.),
           F11, ld1, F12, ld1, Nsmall, magma_q);
        gpu::magma::trsm_vbatched_max_nocheck
          (MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
           L.max_d1_small, L.max_d2_small, d1, d2, scalar_t(1.),
           F11, ld1, F12, ld1, Nsmall, magma_q);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::getrs
            (solver_handle, Trans::N, L.f[i]->F11_, L.f[i]->piv_,
             L.f[i]->F12_, L.dev_getrf_err+i);

        STRUMPACK_ADD_MEMORY(L.factor_bytes);
        L.f[0]->host_factors_.reset(new scalar_t[L.factor_size]);
        L.f[0]->pivot_mem_.resize(L.piv_size);

        comp_stream.synchronize();
        gpu_check(gpu::copy_device_to_host_async<scalar_t>
                  (pinned, dev_factors, L.factor_size, copy_stream));

        // use max_nocheck to overlap this with copy above
        gpu::magma::gemm_vbatched_max_nocheck
          (MagmaNoTrans, MagmaNoTrans, d2, d2, d1, scalar_t(-1.),
           F21, ld2, F12, ld1, scalar_t(1.), F22, ld2, Nsmall,
           L.max_d2_small, L.max_d2_small, L.max_d1_small, magma_q);
        for (std::size_t i=Nsmall; i<N; i++)
          gpu::gemm
            (blas_handle, Trans::N, Trans::N, scalar_t(-1.),
             L.f[i]->F21_, L.f[i]->F12_, scalar_t(1.), L.f[i]->F22_);
        copy_stream.synchronize();
        auto host_factors = L.f[0]->host_factors_.get();
#pragma omp parallel for
        for (std::size_t i=0; i<L.factor_size; i++)
          host_factors[i] = pinned[i];

        gpu_check(gpu::copy_device_to_host<int>
                  (L.f[0]->pivot_mem_.data(), L.f[0]->piv_, L.piv_size));
        L.set_factor_pointers
          (L.f[0]->host_factors_.get(), L.f[0]->pivot_mem_.data());
        comp_stream.synchronize();
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
      gpu_check(gpu::copy_device_to_host
                (host_Schur_.get(), (scalar_t*)(old_work), dupd*dupd));
      F22_ = DenseMW_t(dupd, dupd, host_Schur_.get(), dupd);
    }
    return err_code;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMAGMA<scalar_t,integer_t>::multifrontal_solve
  (DenseM_t& b) const {
    if (dev_factors_) gpu_solve(b);
    else
      // factors are not on the device, so do the solve on the CPU
      FrontalMatrix<scalar_t,integer_t>::multifrontal_solve(b);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMAGMA<scalar_t,integer_t>::fwd_solve_phase2
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

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMAGMA<scalar_t,integer_t>::bwd_solve_phase1
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
  FrontalMatrixMAGMA<scalar_t,integer_t>::gpu_solve(DenseM_t& b) const {
    gpu::Stream comp_stream;
    gpu::SOLVERHandle solver_handle(comp_stream);
    gpu::BLASHandle blas_handle(comp_stream);
    gpu::magma::MAGMAQueue magma_q(comp_stream, blas_handle);
    const int lvls = this->levels();
    std::vector<LInfo_t> ldata(lvls);
    for (int l=lvls-1; l>=0; l--) {
      std::vector<F_t*> fp;
      const_cast<FM_t*>(this)->get_level_fronts(fp, l);
      ldata[l] = LInfo_t(fp, solver_handle);
    }
    int nrhs = b.cols();
    std::vector<std::size_t> l_off(lvls, 0);
    std::size_t Nmax = 0, Ntotal = 0, max_bupd_size = 0,
      max_work_mem = 0, Isize = 0;
    for (int l=0; l<lvls; l++) {
      auto& L = ldata[l];
      auto N = L.f.size();
      if (l != lvls-1) l_off[l+1] = l_off[l] + N;
      Nmax = std::max(Nmax, N);
      Ntotal += N;
      Isize += L.Isize.back();
      max_bupd_size = std::max(max_bupd_size, L.total_upd_size*nrhs);
      max_work_mem = std::max(max_work_mem, L.work_bytes);
    }
    std::size_t d_mem_bytes = 0;
    d_mem_bytes += gpu::round_up(Ntotal*sizeof(gpu::AssembleData<scalar_t>));
    d_mem_bytes += gpu::round_up(Isize*sizeof(std::size_t));
    d_mem_bytes += gpu::round_up((3*(Nmax+1)+1)*sizeof(int));
    d_mem_bytes += gpu::round_up(2*Ntotal*sizeof(scalar_t*));
    auto h_mem_bytes = d_mem_bytes;
    d_mem_bytes += gpu::round_up(max_work_mem);
    d_mem_bytes += gpu::round_up((nrhs*b.rows()+2*max_bupd_size)*sizeof(scalar_t));
    gpu::DeviceMemory<char> d_mem(d_mem_bytes);
    auto d_asmbl      = d_mem.template as<gpu::AssembleData<scalar_t>>();
    auto d_I          = gpu::aligned_ptr<std::size_t>(d_asmbl+Ntotal);
    auto d_batch_int  = gpu::aligned_ptr<int>(d_I+Isize);
    auto d_batch_ptrs = gpu::aligned_ptr<scalar_t*>(d_batch_int+3*(Nmax+1)+1);
    auto dwork_mem    = gpu::aligned_ptr<char>(d_batch_ptrs+2*Ntotal);
    auto d_rhs_mem    = gpu::aligned_ptr<scalar_t>(dwork_mem+max_work_mem);
    gpu::HostMemory<char> h_mem(h_mem_bytes);
    auto h_asmbl      = h_mem.template as<gpu::AssembleData<scalar_t>>();
    auto h_I          = gpu::aligned_ptr<std::size_t>(h_asmbl+Ntotal);
    auto h_batch_int  = gpu::aligned_ptr<int>(h_I+Isize);
    auto h_batch_ptrs = gpu::aligned_ptr<scalar_t*>(h_batch_int+3*(Nmax+1)+1);
    // scalar device data
    auto d_b = d_rhs_mem;
    auto d_bupd_odd = d_b + nrhs * b.rows();
    auto d_bupd_even = d_bupd_odd + max_bupd_size;
    // int device data
    auto nrhs_batch = d_batch_int;
    auto ldrhs_batch = nrhs_batch + Nmax+1;
    auto inc_batch = ldrhs_batch + Nmax+1;
    auto getrs_err = inc_batch + Nmax+1;
    for (std::size_t i=0; i<Nmax+1; i++) {
      h_batch_int[i           ] = nrhs;
      h_batch_int[i+  (Nmax+1)] = b.rows();
      h_batch_int[i+2*(Nmax+1)] = 1;
    }
    std::vector<scalar_t**>
      h_rhs_batch(lvls), h_bupd_batch(lvls),
      d_rhs_batch(lvls), d_bupd_batch(lvls);
#pragma omp parallel for schedule(static,1)
    for (std::size_t l=0; l<std::size_t(lvls); l++) {
      d_rhs_batch[l] = d_batch_ptrs + l_off[l];
      d_bupd_batch[l] = d_batch_ptrs + Ntotal + l_off[l];
      h_rhs_batch[l] = h_batch_ptrs + l_off[l];
      h_bupd_batch[l] = h_batch_ptrs + Ntotal + l_off[l];
      auto bu = (l % 2) ? d_bupd_odd : d_bupd_even;
      auto& L = ldata[l];
      for (std::size_t i=0, pos=l_off[l]; i<L.f.size(); i++, pos++) {
        h_batch_ptrs[pos] = d_b + L.f[i]->sep_begin();
        h_batch_ptrs[Ntotal+pos] = bu;
        bu += nrhs * L.f[i]->dim_upd();
      }
    }
    for (std::size_t l=0, Ipos=0; l<std::size_t(lvls); l++) {
      auto& L = ldata[l];
#pragma omp parallel for
      for (std::size_t i=0; i<L.f.size(); i++) {
        auto& f = *(L.f[i]);
        h_asmbl[l_off[l]+i] = gpu::AssembleData<scalar_t>
          (f.dim_sep(), f.dim_upd(), h_rhs_batch[l][i], h_bupd_batch[l][i]);
        auto hI = h_I+Ipos+L.Isize[i];
        if (f.lchild_) {
          f.lchild_->upd_to_parent(&f, hI);
          hI += f.lchild_->dim_upd();
        }
        if (f.rchild_)
          f.rchild_->upd_to_parent(&f, hI);
      }
      for (std::size_t i=0, ch=0; i<L.f.size(); i++) {
        auto& f = *(L.f[i]);
        auto dI = d_I+Ipos+L.Isize[i];
        if (f.lchild_) {
          auto dupd = f.lchild_->dim_upd();
          h_asmbl[l_off[l]+i].set_ext_add_left
            (dupd, h_bupd_batch[l+1][ch++], dI);
          dI += dupd;
        }
        if (f.rchild_)
          h_asmbl[l_off[l]+i].set_ext_add_right
            (f.rchild_->dim_upd(), h_bupd_batch[l+1][ch++], dI);
      }
      Ipos += L.Isize.back();
    }
    // copy all meta-data at once, from pinned memory
    gpu_check(gpu::copy_host_to_device<char>(d_mem, h_mem, h_mem_bytes));
    // copy rhs, from pageable memory (input)
    gpu_check(gpu::copy_host_to_device<scalar_t>(d_b, b));

    // forward solve
    for (int l=lvls-1; l>=0; l--) {
      gpu::synchronize();
      auto& L = ldata[l];
      std::size_t N = L.f.size(), Nsmall = L.Nsmall;
      L.set_batch_data(dwork_mem);
      auto d1 = L.dev_d1_batch;   auto d2 = L.dev_d2_batch;
      auto ld1 = L.dev_ld1_batch; auto ld2 = L.dev_ld2_batch;
      auto F11 = L.dev_F_batch;   auto F21 = F11 + 2*Nsmall;
      gpu_check(gpu::memset<scalar_t>
                ((l % 2) ? d_bupd_odd : d_bupd_even, 0,
                 L.total_upd_size*nrhs));
      if (l != lvls-1)
        gpu::extend_add_rhs
          (b.rows(), nrhs, N, &h_asmbl[l_off[l]], d_asmbl+l_off[l]);
      // TODO remove?
      gpu::synchronize();
      gpu::laswp_fwd_vbatched
        (blas_handle, nrhs_batch+Nmax-Nsmall, nrhs,
         d_rhs_batch[l], ldrhs_batch+Nmax-Nsmall,
         L.dev_ipiv_batch, d1, Nsmall);
      gpu::magma::trsm_vbatched_max_nocheck
        (MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
         L.max_d1_small, nrhs, d1, nrhs_batch+Nmax-Nsmall, scalar_t(1.),
         F11, ld1, d_rhs_batch[l], ldrhs_batch+Nmax-Nsmall, Nsmall, magma_q);
      gpu::magma::trsm_vbatched_max_nocheck
        (MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
         L.max_d1_small, nrhs, d1, nrhs_batch+Nmax-Nsmall, scalar_t(1.),
         F11, ld1, d_rhs_batch[l], ldrhs_batch+Nmax-Nsmall, Nsmall, magma_q);
      for (std::size_t i=Nsmall; i<N; i++) {
        DenseMW_t fb(L.f[i]->dim_sep(), nrhs, h_rhs_batch[l][i], b.rows());
        gpu::getrs(solver_handle, Trans::N, L.f[i]->F11_,
                   L.f[i]->piv_, fb, getrs_err);
      }
      if (l == 0) break;
      if (nrhs == 1)
        gpu::magma::gemv_vbatched_max_nocheck
          (MagmaNoTrans, d2, d1, scalar_t(-1.),
           F21, ld2, d_rhs_batch[l], inc_batch+Nmax-Nsmall, scalar_t(1.),
           d_bupd_batch[l], inc_batch+Nmax-Nsmall, Nsmall,
           L.max_d2_small, L.max_d1_small, magma_q);
      else
        gpu::magma::gemm_vbatched_max_nocheck
          (MagmaNoTrans, MagmaNoTrans, d2, nrhs_batch+Nmax-Nsmall, d1,
           scalar_t(-1.), F21, ld2, d_rhs_batch[l], ldrhs_batch+Nmax-Nsmall,
           scalar_t(1.), d_bupd_batch[l], ld2, Nsmall,
           L.max_d2_small, nrhs, L.max_d1_small, magma_q);
      for (std::size_t i=Nsmall; i<N; i++) {
        DenseMW_t fb(L.f[i]->dim_sep(), nrhs, h_rhs_batch[l][i], b.rows()),
          fbu(L.f[i]->dim_upd(), nrhs, h_bupd_batch[l][i], L.f[i]->dim_upd());
        if (nrhs == 1)
          gpu::gemv(blas_handle, Trans::N, scalar_t(-1.),
                    L.f[i]->F21_, fb, scalar_t(1.), fbu);
        else
          gpu::gemm(blas_handle, Trans::N, Trans::N, scalar_t(-1.),
                    L.f[i]->F21_, fb, scalar_t(1.), fbu);
      }
    }
    gpu::synchronize();

    // backward solve
    for (int l=0; l<lvls; l++) {
      auto& L = ldata[l];
      std::size_t N = L.f.size(), Nsmall = L.Nsmall;
      L.set_batch_data(dwork_mem);
      auto d1 = L.dev_d1_batch;   auto d2 = L.dev_d2_batch;
      auto ld1 = L.dev_ld1_batch; auto ld2 = L.dev_ld2_batch;
      auto F12 = L.dev_F_batch + Nsmall;
      gpu::synchronize();
      if (l != 0) {
        if (nrhs == 1)
          gpu::magma::gemv_vbatched_max_nocheck
            (MagmaNoTrans, d1, d2, scalar_t(-1.), F12, ld1,
             d_bupd_batch[l], inc_batch+Nmax-Nsmall, scalar_t(1.),
             d_rhs_batch[l], inc_batch+Nmax-Nsmall, Nsmall,
             L.max_d1_small, L.max_d2_small, magma_q);
        else
          gpu::magma::gemm_vbatched_max_nocheck
            (MagmaNoTrans, MagmaNoTrans, d1, nrhs_batch+Nmax-Nsmall, d2,
             scalar_t(-1.), F12, ld1, d_bupd_batch[l], ld2, scalar_t(1.),
             d_rhs_batch[l], ldrhs_batch+Nmax-Nsmall, Nsmall,
             L.max_d1_small, nrhs, L.max_d2_small, magma_q);
        for (std::size_t i=Nsmall; i<N; i++) {
          DenseMW_t fb(L.f[i]->dim_sep(), nrhs, h_rhs_batch[l][i], b.rows()),
            fbu(L.f[i]->dim_upd(), nrhs, h_bupd_batch[l][i], L.f[i]->dim_upd());
          if (nrhs == 1)
            gpu::gemv(blas_handle, Trans::N, scalar_t(-1.),
                      L.f[i]->F12_, fbu, scalar_t(1.), fb);
          else
            gpu::gemm(blas_handle, Trans::N, Trans::N, scalar_t(-1.),
                      L.f[i]->F12_, fbu, scalar_t(1.), fb);
        }
      }
      gpu::synchronize();
      if (l != lvls-1)
        gpu::extract_rhs
          (b.rows(), nrhs, N, &h_asmbl[l_off[l]], d_asmbl+l_off[l]);
    }
    gpu_check(gpu::copy_device_to_host<scalar_t>(b, d_b));
  }

  // explicit template instantiations
  template class FrontalMatrixMAGMA<float,int>;
  template class FrontalMatrixMAGMA<double,int>;
  template class FrontalMatrixMAGMA<std::complex<float>,int>;
  template class FrontalMatrixMAGMA<std::complex<double>,int>;

  template class FrontalMatrixMAGMA<float,long int>;
  template class FrontalMatrixMAGMA<double,long int>;
  template class FrontalMatrixMAGMA<std::complex<float>,long int>;
  template class FrontalMatrixMAGMA<std::complex<double>,long int>;

  template class FrontalMatrixMAGMA<float,long long int>;
  template class FrontalMatrixMAGMA<double,long long int>;
  template class FrontalMatrixMAGMA<std::complex<float>,long long int>;
  template class FrontalMatrixMAGMA<std::complex<double>,long long int>;

} // end namespace strumpack
