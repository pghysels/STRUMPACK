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

#include "FrontalMatrixCUBLAS.hpp"
#include "dense/CUDAWrapper.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#include "FrontalMatrixMPI.hpp"
#endif
#include <cuda_profiler_api.h>
#if defined(STRUMPACK_USE_CUDA)
#include "FrontalMatrixCUDA.hpp"
#endif


namespace strumpack {


  template<typename scalar_t, typename integer_t> class LevelInfo {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FC_t = FrontalMatrixCUBLAS<scalar_t,integer_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;

  public:
    LevelInfo() {}

    LevelInfo(const std::vector<F_t*>& fronts, cusolverDnHandle_t& handle,
              int max_streams) {
      f.reserve(fronts.size());
      for (auto& F : fronts)
        f.push_back(dynamic_cast<FC_t*>(F));
      int max_dsep = 0;
      for (auto F : f) {
        auto dsep = F->dim_sep();
        auto dupd = F->dim_upd();
        factor_size += dsep*dsep + 2*dsep*dupd;
        schur_size += dupd*dupd;
        piv_size += dsep;
        if (dsep <= 8)       N_8++;
        else if (dsep <= 16) N_16++;
        else if (dsep <= 32) N_32++;
        if (dsep > max_dsep) max_dsep = dsep;
      }
      cuda::cusolverDngetrf_bufferSize
        (handle, max_dsep, max_dsep, (scalar_t*)(nullptr),
         max_dsep, &getrf_work_size);
      work_bytes = sizeof(scalar_t) * schur_size + sizeof(int) * piv_size +
        sizeof(scalar_t) * getrf_work_size * max_streams +
        sizeof(int) * max_streams;
    }

    void print_info(int l, int lvls) {
      std::cout << "#  level " << l << " of " << lvls
                << " has " << f.size() << " nodes and "
                << N_8 << " <=8, "
                << N_16 << " <=16, "
                << N_32 << " <=32, needs "
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

    void set_front_pointers(scalar_t* factors) {
      for (auto F : f) {
        const int dsep = F->dim_sep();
        const int dupd = F->dim_upd();
        F->F11_ = DenseMW_t(dsep, dsep, factors, dsep); factors += dsep*dsep;
        F->F12_ = DenseMW_t(dsep, dupd, factors, dsep); factors += dsep*dupd;
        F->F21_ = DenseMW_t(dupd, dsep, factors, dupd); factors += dupd*dsep;
      }
    }
    void set_front_pointers
    (scalar_t* factors, scalar_t* schur, int max_streams) {
      set_front_pointers(factors);
      for (auto F : f) {
        const int dupd = F->dim_upd();
        if (dupd) {
          F->F22_ = DenseMW_t(dupd, dupd, schur, dupd);
          schur += dupd*dupd;
        }
      }
      dev_piv.resize(f.size());
      dev_getrf_work = schur;
      schur += max_streams * getrf_work_size;
      auto imem = reinterpret_cast<int*>(schur);
      for (std::size_t n=0; n<f.size(); n++) {
        dev_piv[n] = imem;
        imem += f[n]->dim_sep();
      }
      dev_getrf_err = imem;
      imem += max_streams;
    }

    std::vector<FC_t*> f;
    std::size_t factor_size = 0, schur_size = 0, piv_size = 0,
      work_bytes = 0, N_8 = 0, N_16 = 0, N_32 = 0;
    scalar_t* dev_getrf_work = nullptr;
    int* dev_getrf_err = nullptr;
    std::vector<int*> dev_piv;
    std::vector<DenseMatrixWrapper<scalar_t>> bloc;
    int getrf_work_size = 0;
  };


  template<typename scalar_t,typename integer_t>
  FrontalMatrixCUBLAS<scalar_t,integer_t>::FrontalMatrixCUBLAS
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixCUBLAS<scalar_t,integer_t>::release_work_memory() {
    F22_.clear();
    if (dev_work_mem_) cudaFree(dev_work_mem_);
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixCUBLAS<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf,
   const FrontalMatrixMPI<scalar_t,integer_t>* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_seq_copy_to_buffers
      (F22_, sbuf, pa, this);
  }
#endif

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixCUBLAS<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const F_t* p, int task_depth) {
    const std::size_t pdsep = paF11.rows();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
    DenseM_t F22(dupd, dupd);
    // get the contribution block from the device
    cudaMemcpy(F22.data(), dev_work_mem_, dupd*dupd*sizeof(scalar_t),
               cudaMemcpyDeviceToHost);
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared) grainsize(64)      \
  if(task_depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t c=0; c<dupd; c++) {
      auto pc = I[c];
      if (pc < pdsep) {
        for (std::size_t r=0; r<upd2sep; r++)
          paF11(I[r],pc) += F22(r,c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF21(I[r]-pdsep,pc) += F22(r,c);
      } else {
        for (std::size_t r=0; r<upd2sep; r++)
          paF12(I[r],pc-pdsep) += F22(r, c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF22(I[r]-pdsep,pc-pdsep) += F22(r,c);
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
  (const std::vector<LevelInfo<scalar_t,integer_t>>& ldata,
   std::size_t max_front_data_size) {
    std::size_t peak_device_mem = 0;
    for (std::size_t l=0; l<ldata.size(); l++) {
      auto& L = ldata[l];
      // memory needed on this level: pointers to fronts, factors,
      // schur updates, pivot vectors, cuSOLVER work space, ...
      std::size_t level_mem = max_front_data_size
        + L.factor_size*sizeof(scalar_t) + L.work_bytes;
      // the contribution blocks of the previous level are still
      // needed for the extend-add
      if (l+1 < ldata.size())
        level_mem += ldata[l+1].work_bytes;
      peak_device_mem = std::max(peak_device_mem, level_mem);
    }
    std::size_t free_device_mem, total_device_mem;
    cudaMemGetInfo(&free_device_mem, &total_device_mem);
    // only use 90% of available memory, since we're not counting the
    // sparse elements in the peak_device_mem
    return peak_device_mem < 0.9 * free_device_mem;
  }


  void create_handles(std::vector<cudaStream_t>& stream,
                      std::vector<cublasHandle_t>& blas_handle,
                      std::vector<cusolverDnHandle_t>& solver_handle) {
    for (std::size_t i=0; i<stream.size(); i++) {
      cublasCreate(&blas_handle[i]);
      cusolverDnCreate(&solver_handle[i]);
      cudaStreamCreate(&stream[i]);
      cublasSetStream(blas_handle[i], stream[i]);
      cusolverDnSetStream(solver_handle[i], stream[i]);
    }
    if (auto err = cudaGetLastError()) {
      std::cerr << "Error in CUDA setup: "
                << cudaGetErrorString(err) << std::endl;
      exit(-1);
    }
  }

  void destroy_handles(std::vector<cudaStream_t>& stream,
                       std::vector<cublasHandle_t>& blas_handle,
                       std::vector<cusolverDnHandle_t>& solver_handle) {
    for (std::size_t i=0; i<stream.size(); i++) {
      cudaStreamDestroy(stream[i]);
      cublasDestroy(blas_handle[i]);
      cusolverDnDestroy(solver_handle[i]);
    }
  }


  template<typename scalar_t,typename integer_t> void
  FrontalMatrixCUBLAS<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    using FC_t = FrontalMatrixCUBLAS<scalar_t,integer_t>;
    using LevelInfo_t = LevelInfo<scalar_t,integer_t>;

    const int max_streams = opts.cuda_streams();
    std::vector<cudaStream_t> stream(max_streams);
    std::vector<cublasHandle_t> blas_handle(max_streams);
    std::vector<cusolverDnHandle_t> solver_handle(max_streams);
    create_handles(stream, blas_handle, solver_handle);

    const int lvls = this->levels();
    std::vector<LevelInfo_t> ldata(lvls);
    std::size_t max_front_data_size = 0;
    for (int l=lvls-1; l>=0; l--) {
      std::vector<F_t*> fp;
      fp.reserve(std::pow(2, l));
      this->get_level_fronts(fp, l);
      auto& L = ldata[l];
      L = LevelInfo_t(fp, solver_handle[0], max_streams);
      max_front_data_size = std::max
        (max_front_data_size, (L.N_8+L.N_16+L.N_32) *
         sizeof(cuda::FrontData<scalar_t>));
    }

    int starting_level = lvls - 1;

    if (!sufficient_device_memory(ldata, max_front_data_size)) {
      if (opts.verbose())
        std::cout << "# Factorization does not fit in GPU memory, "
          "splitting in smaller traversals." << std::endl;
      if (lchild_)
        lchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
      if (rchild_)
        rchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
      starting_level = 0;
    }

    cuda::FrontData<scalar_t> *host_front_data = nullptr,
      *dev_front_data = nullptr;
    // TODO check whether this works
    cudaMallocHost(&host_front_data, max_front_data_size);
    cudaMalloc(&dev_front_data, max_front_data_size);

    for (int l=starting_level; l>=0; l--) {
      TaskTimer tl("");
      tl.start();
      auto& L = ldata[l];
      auto N = L.f.size();
      if (opts.verbose()) L.print_info(l, lvls);

      scalar_t* dev_factors = nullptr;
      void *work_mem = nullptr;
      // TODO check whether this works
      cudaMalloc(&dev_factors, L.factor_size*sizeof(scalar_t));
      cudaMalloc(&work_mem, L.work_bytes);
      // set all factor and schur memory to 0, on the device
      cudaMemset(dev_factors, 0, L.factor_size*sizeof(scalar_t));
      cudaMemset(work_mem, 0, L.schur_size*sizeof(scalar_t));

      L.set_front_pointers(dev_factors, (scalar_t*)(work_mem), max_streams);

      {
        // front assembly
        using Trip_t = Triplet<scalar_t>;
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
        std::size_t ea_mem_size = N*sizeof(cuda::AssembleData<scalar_t>) +
          Isize*sizeof(std::size_t) +
          (e11.size()+e12.size()+e21.size())*sizeof(Trip_t);
        void* host_ea_mem = nullptr;
        void* dev_ea_mem = nullptr;
        // TODO check whether this works
        cudaMallocHost(&host_ea_mem, ea_mem_size);
        cudaMalloc(&dev_ea_mem, ea_mem_size);
        auto asmbl =
          reinterpret_cast<cuda::AssembleData<scalar_t>*>(host_ea_mem);
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
          asmbl[n] = cuda::AssembleData<scalar_t>
            (f.dim_sep(), f.dim_upd(), f.F11_.data(), f.F12_.data(),
             f.F21_.data(), f.F22_.data(),
             ne[n+1][0]-ne[n][0], ne[n+1][1]-ne[n][1], ne[n+1][2]-ne[n][2],
             de11+ne[n][0], de12+ne[n][1], de21+ne[n][2]);
          if (f.lchild_) {
            auto c = dynamic_cast<FC_t*>(f.lchild_.get());
            asmbl[n].set_ext_add_left(c->dim_upd(), c->F22_.data(), Iptr);
            auto u = c->upd_to_parent(&f);
            std::copy(u.begin(), u.end(), Iptr);
            Iptr += u.size();
          }
          if (f.rchild_) {
            auto c = dynamic_cast<FC_t*>(f.rchild_.get());
            asmbl[n].set_ext_add_right(c->dim_upd(), c->F22_.data(), Iptr);
            auto u = c->upd_to_parent(&f);
            std::copy(u.begin(), u.end(), Iptr);
            Iptr += u.size();
          }
        }
        cudaMemcpy(dev_ea_mem, host_ea_mem, ea_mem_size,
                   cudaMemcpyHostToDevice);
        cuda::assemble(N, asmbl);
        cudaFreeHost(host_ea_mem);
        cudaDeviceSynchronize();
        cudaFree(dev_ea_mem);
      }

      if (dev_work_mem_) cudaFree(dev_work_mem_);
      dev_work_mem_ = work_mem;

      if (L.N_8 || L.N_16 || L.N_32) {
        auto f8 = host_front_data;
        auto f16 = f8 + L.N_8;
        auto f32 = f16 + L.N_16;
        for (std::size_t n=0, n8=0, n16=0, n32=0; n<N; n++) {
          auto& f = *(L.f[n]);
          const auto dsep = f.dim_sep();
          if (dsep <= 32) {
            cuda::FrontData<scalar_t>
              t(dsep, f.dim_upd(), f.F11_.data(), f.F12_.data(),
                f.F21_.data(), f.F22_.data(), L.dev_piv[n]);
            if (dsep <= 8) f8[n8++] = t;
            else if (dsep <= 16) f16[n16++] = t;
            else f32[n32++] = t;
          }
        }
        cudaMemcpy(dev_front_data, host_front_data,
                   (L.N_8+L.N_16+L.N_32) * sizeof(cuda::FrontData<scalar_t>),
                   cudaMemcpyHostToDevice);
        cuda::factor_block_batch<scalar_t,8>(L.N_8, f8);
        cuda::factor_block_batch<scalar_t,16>(L.N_16, f16);
        cuda::factor_block_batch<scalar_t,32>(L.N_32, f32);
      }

      for (std::size_t n=0; n<N; n++) {
        auto& f = *(L.f[n]);
        auto stream = n % max_streams;
        const auto dsep = f.dim_sep();
        if (dsep > 32) {
          const auto dupd = f.dim_upd();
          cuda::cusolverDngetrf
            (solver_handle[stream], dsep, dsep, f.F11_.data(), dsep,
             L.dev_getrf_work + stream * L.getrf_work_size,
             L.dev_piv[n], L.dev_getrf_err + stream);
          // TODO if (opts.replace_tiny_pivots()) { ...
          if (dupd) {
            cuda::cusolverDngetrs
              (solver_handle[stream], CUBLAS_OP_N, dsep, dupd,
               f.F11_.data(), dsep, L.dev_piv[n],
               f.F12_.data(), dsep, L.dev_getrf_err + stream);
            gemm_cuda(blas_handle[stream], CUBLAS_OP_N, CUBLAS_OP_N,
                      scalar_t(-1.), f.F21_, f.F12_, scalar_t(1.), f.F22_);
          }
        }
      }

      // allocate memory for the factors/pivots, on the host
      L.f[0]->factor_mem_ = std::unique_ptr<scalar_t>
        (new scalar_t[L.factor_size]);
      auto fmem = L.f[0]->factor_mem_.get();
      auto pmem_ = std::unique_ptr<int[]>(new int[L.piv_size]);
      auto pmem = pmem_.get();

      // wait for device computations to complete
      cudaDeviceSynchronize();
      // copy the factors from the device to the host
      cudaMemcpyAsync(fmem, dev_factors, L.factor_size*sizeof(scalar_t),
                      cudaMemcpyDeviceToHost, stream[1 % stream.size()]);
      // copy pivot vectors from the device to the host
      cudaMemcpyAsync(pmem, L.dev_piv[0], L.piv_size*sizeof(int),
                      cudaMemcpyDeviceToHost, stream[0]);
      // set front pointers to host memory
      L.set_front_pointers(fmem);

      // count flops
      long long level_flops = L.total_flops();
      STRUMPACK_FULL_RANK_FLOPS(level_flops);
      if (opts.verbose()) {
        auto level_time = tl.elapsed();
        std::cout << "#   GPU Factorization complete, took: "
                  << level_time << " seconds, "
                  << level_flops / 1.e9 << " GFLOPS, "
                  << (float(level_flops) / level_time) / 1.e9
                  << " GFLOP/s" << std::endl;
      }

      cudaDeviceSynchronize();

      // delete factors from device
      cudaFree(dev_factors);

#pragma omp parallel for
      for (std::size_t n=0; n<N; n++) {
        auto& f = *(L.f[n]);
        const auto dsep = f.dim_sep();
        int offset = std::distance(L.dev_piv[0], L.dev_piv[n]);
        f.piv.assign(pmem + offset, pmem + offset + dsep);
      }
    }

    // free front pointer data from device
    cudaFreeHost(host_front_data);
    cudaFree(dev_front_data);

    // if dim_upd() != 0, this is not the root node, and the
    // contribution block will still be needed by the parent
    if (!dim_upd()) cudaFree(dev_work_mem_);

    destroy_handles(stream, blas_handle, solver_handle);
  }



  template<typename scalar_t,typename integer_t> void
  FrontalMatrixCUBLAS<scalar_t,integer_t>::forward_multifrontal_solve
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
  FrontalMatrixCUBLAS<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
      //bloc.laswp(piv, true);
      F11_.solve_LU_in_place(bloc, piv, task_depth);
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
  FrontalMatrixCUBLAS<scalar_t,integer_t>::backward_multifrontal_solve
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
  FrontalMatrixCUBLAS<scalar_t,integer_t>::bwd_solve_phase1
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
  FrontalMatrixCUBLAS<scalar_t,integer_t>::extract_CB_sub_matrix
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
  FrontalMatrixCUBLAS<scalar_t,integer_t>::sample_CB
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
  FrontalMatrixCUBLAS<scalar_t,integer_t>::sample_CB
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
  FrontalMatrixCUBLAS<scalar_t,integer_t>::sample_CB_to_F11
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
  FrontalMatrixCUBLAS<scalar_t,integer_t>::sample_CB_to_F12
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
  FrontalMatrixCUBLAS<scalar_t,integer_t>::sample_CB_to_F21
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
  FrontalMatrixCUBLAS<scalar_t,integer_t>::sample_CB_to_F22
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
  template class FrontalMatrixCUBLAS<float,int>;
  template class FrontalMatrixCUBLAS<double,int>;
  template class FrontalMatrixCUBLAS<std::complex<float>,int>;
  template class FrontalMatrixCUBLAS<std::complex<double>,int>;

  template class FrontalMatrixCUBLAS<float,long int>;
  template class FrontalMatrixCUBLAS<double,long int>;
  template class FrontalMatrixCUBLAS<std::complex<float>,long int>;
  template class FrontalMatrixCUBLAS<std::complex<double>,long int>;

  template class FrontalMatrixCUBLAS<float,long long int>;
  template class FrontalMatrixCUBLAS<double,long long int>;
  template class FrontalMatrixCUBLAS<std::complex<float>,long long int>;
  template class FrontalMatrixCUBLAS<std::complex<double>,long long int>;

} // end namespace strumpack
