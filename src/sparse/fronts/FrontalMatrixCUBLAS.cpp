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

  //template<typename scalar_t,typename integer_t> class FrontalMatrixCUBLAS;

  template<typename scalar_t, typename integer_t> class LevelInfo {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FC_t = FrontalMatrixCUBLAS<scalar_t,integer_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;

  public:
    LevelInfo() {}

    LevelInfo(const std::vector<F_t*>& fronts, cusolverDnHandle_t& handle,
              int max_streams, int cutoff_size) {
      f.reserve(fronts.size());
      for (auto& F : fronts)
        f.push_back(dynamic_cast<FC_t*>(F));
      int max_dsep = 0;
      for (auto F : f) {
        auto dsep = F->dim_sep();
        auto dupd = F->dim_upd();
        auto size = dsep + dupd;
        factor_size += dsep*dsep + 2*dsep*dupd;
        schur_size += dupd*dupd;
        piv_size += dsep;
        if (size < cutoff_size) nnodes_small++;
        if (dsep <= 8)       nnodes_8++;
        else if (dsep <= 16) nnodes_16++;
        else if (dsep <= 32) nnodes_32++;
        if (dsep > max_dsep) max_dsep = dsep;
      }
      cuda::cusolverDngetrf_bufferSize
        (handle, max_dsep, max_dsep, (scalar_t*)(nullptr),
         max_dsep, &getrf_work_size);
      work_bytes =
        sizeof(scalar_t) * schur_size + sizeof(int) * piv_size +
        sizeof(scalar_t) * getrf_work_size * max_streams +
        sizeof(int) * max_streams;
    }

    void print_info(int l, int lvls) {
      std::cout << "#  level " << l << " of " << lvls
                << " has " << f.size() << " nodes and "
                << nnodes_small << " small nodes, "
                << nnodes_8 << " <=8, "
                << nnodes_16 << " <=16, "
                << nnodes_32 << " <=32, needs "
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
    void set_front_pointers(scalar_t* factors, scalar_t* schur, int max_streams) {
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
      work_bytes = 0, nnodes_small = 0,
      nnodes_8 = 0, nnodes_16 = 0, nnodes_32 = 0;

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
    cudaFree(all_work_mem_);
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
  FrontalMatrixCUBLAS<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    using FC_t = FrontalMatrixCUBLAS<scalar_t,integer_t>;
    using LevelInfo_t = LevelInfo<scalar_t,integer_t>;

    const int cutoff_size = opts.cuda_cutoff();
    const int max_streams = opts.cuda_streams();

    const int lvls = this->levels();
    std::vector<LevelInfo_t> ldata(lvls);

    // cudaProfilerStart();
    // int device_id;
    // cudaGetDevice(&device_id);
    std::vector<cudaStream_t> streams(max_streams);
    std::vector<cublasHandle_t> blas_handle(max_streams);
    std::vector<cusolverDnHandle_t> solver_handle(max_streams);
    for (int i=0; i<max_streams; i++) {
      cudaStreamCreate(&streams[i]);
      cublasCreate(&blas_handle[i]);
      cublasSetStream(blas_handle[i], streams[i]);
      cusolverDnCreate(&solver_handle[i]);
      cusolverDnSetStream(solver_handle[i], streams[i]);
    }
    if (auto err = cudaGetLastError()) {
      std::cerr << "Error in CUDA setup: "
                << cudaGetErrorString(err) << std::endl;
      exit(-1);
    }


    for (int lvl=0; lvl<lvls; lvl++) {
      int l = lvls - lvl - 1;
      std::vector<F_t*> fp;
      fp.reserve(std::pow(2, l));
      this->get_level_fronts(fp, l);
      ldata[l] = LevelInfo_t(fp, solver_handle[0], max_streams, cutoff_size);
    }

    // TODO: check if the factorization will fit in the GPU memory, if
    // not, recurse on children and then handle this front with cuBLAS


    void *work_mem_old = nullptr;
    for (int lvl=0; lvl<lvls; lvl++) {
      int l = lvls - lvl - 1;
      TaskTimer tl("");
      tl.start();
      auto nnodes = ldata[l].f.size();
      auto nnodes_small = ldata[l].nnodes_small;
      auto nnodes_big = nnodes - nnodes_small;
      if (opts.verbose()) ldata[l].print_info(l, lvls);

      scalar_t* factor_mem = nullptr;
      void *work_mem = nullptr;
      cudaMallocManaged(&factor_mem, ldata[l].factor_size*sizeof(scalar_t));
      cudaMallocManaged(&work_mem, ldata[l].work_bytes);

      if (nnodes_big) {
        // set all factor memory to 0, on the device
        cudaMemset(factor_mem, 0, ldata[l].factor_size*sizeof(scalar_t));
        // also need to set the Schur complements to 0
        cudaMemset(work_mem, 0, ldata[l].schur_size*sizeof(scalar_t));
      }
      ldata[l].set_front_pointers
        (factor_mem, static_cast<scalar_t*>(work_mem), max_streams);


      if (nnodes_small) {
        // build front from sparse matrix and extend-add
#pragma omp parallel for
        for (std::size_t n=0; n<nnodes; n++) {
          auto& f = *(ldata[l].f[n]);
          if (f.dim_sep() + f.dim_upd() < cutoff_size) {
            f.F11_.zero();
            f.F12_.zero();
            f.F21_.zero();
            f.F22_.zero();
            A.extract_front
              (f.F11_, f.F12_, f.F21_, f.sep_begin_, f.sep_end_, f.upd_, 0);
            if (f.lchild_)
              f.lchild_->extend_add_to_dense
                (f.F11_, f.F12_, f.F21_, f.F22_, &f, 0);
            if (f.rchild_)
              f.rchild_->extend_add_to_dense
                (f.F11_, f.F12_, f.F21_, f.F22_, &f, 0);
          }
        }
      }

      // // prefetch to GPU
      //// This generates an error: invalid argument
      // if (nnodes_big) {
      //   cudaMemPrefetchAsync
      //     (work_mem, ldata[l].work_bytes, device_id, 0);
      //   cudaMemPrefetchAsync
      //     (ldata[l].f[0]->factor_mem_.get(),
      //      ldata[l].factor_size*sizeof(scalar_t), device_id, 0);
      // if (auto err = cudaGetLastError()) {
      //   std::cerr << "Error in CUDA memory prefetching: "
      //             << cudaGetErrorString(err) << std::endl;
      //   exit(-1);
      // }
      // }

      if (nnodes_big) {
        using Trip_t = Triplet<scalar_t>;
        std::vector<Trip_t> e11, e12, e21;
        std::vector<int> n11, n12, n21;
        n11.reserve(nnodes_big);
        n12.reserve(nnodes_big);
        n21.reserve(nnodes_big);
        std::size_t Isize = 0;
        for (std::size_t n=0; n<nnodes; n++) {
          auto& f = *(ldata[l].f[n]);
          if (f.dim_sep() + f.dim_upd() >= cutoff_size) {
            n11.push_back(e11.size());
            n12.push_back(e12.size());
            n21.push_back(e21.size());
            A.push_front_elements
              (f.sep_begin_, f.sep_end_, f.upd_, e11, e12, e21);
            if (f.lchild_) Isize += f.lchild_->dim_upd();
            if (f.rchild_) Isize += f.rchild_->dim_upd();
          }
        }
        n11.push_back(e11.size());
        n12.push_back(e12.size());
        n21.push_back(e21.size());
        void* all_ea_mem = nullptr;
        cudaMallocManaged(&all_ea_mem,
                          nnodes_big*sizeof(cuda::AssembleData<scalar_t>) +
                          Isize*sizeof(std::size_t) +
                          (e11.size()+e12.size()+e21.size())*sizeof(Trip_t));
        auto assemble =
          reinterpret_cast<cuda::AssembleData<scalar_t>*>(all_ea_mem);
        auto Imem = reinterpret_cast<std::size_t*>(assemble + nnodes_big);
        auto delems = reinterpret_cast<Trip_t*>(Imem + Isize);
        auto de11 = delems;
        auto de12 = de11 + e11.size();
        auto de21 = de12 + e12.size();
        std::copy(e11.begin(), e11.end(), de11);
        std::copy(e12.begin(), e12.end(), de12);
        std::copy(e21.begin(), e21.end(), de21);
        auto Iptr = Imem;
        for (std::size_t n=0, nbig=0; n<nnodes; n++) {
          auto& f = *(ldata[l].f[n]);
          if (f.dim_sep() + f.dim_upd() >= cutoff_size) {
            assemble[nbig] = cuda::AssembleData<scalar_t>
              (f.dim_sep(), f.dim_upd(), f.F11_.data(), f.F12_.data(),
               f.F21_.data(), f.F22_.data(),
               n11[nbig+1]-n11[nbig], n12[nbig+1]-n12[nbig],
               n21[nbig+1]-n21[nbig],
               de11+n11[nbig], de12+n12[nbig], de21+n21[nbig]);
            if (f.lchild_) {
              auto fc = dynamic_cast<FC_t*>(f.lchild_.get());
              assemble[nbig].CB1 = fc->F22_.data();
              assemble[nbig].dCB1 = fc->dim_upd();
              assemble[nbig].I1 = Iptr;
              auto u = fc->upd_to_parent(&f);
              std::copy(u.begin(), u.end(), Iptr);
              Iptr += u.size();
            }
            if (f.rchild_) {
              auto fc = dynamic_cast<FC_t*>(f.rchild_.get());
              assemble[nbig].CB2 = fc->F22_.data();
              assemble[nbig].dCB2 = fc->dim_upd();
              assemble[nbig].I2 = Iptr;
              auto u = fc->upd_to_parent(&f);
              std::copy(u.begin(), u.end(), Iptr);
              Iptr += u.size();
            }
            nbig++;
          }
        }
        cuda::assemble(nnodes_big, assemble);
        cuda::extend_add(nnodes_big, assemble);

        cudaDeviceSynchronize();

        // free memory containing FrontData: sizes and pointers to
        // front, assembly info, extend-add info, ..
        cudaFree(all_ea_mem);
      }

      if (work_mem_old)
        cudaFree(work_mem_old);
      work_mem_old = work_mem;

      if (nnodes_small) {
#pragma omp parallel for
        for (std::size_t n=0; n<nnodes; n++) {
          auto& f = *(ldata[l].f[n]);
          const auto dsep = f.dim_sep();
          const auto dupd = f.dim_upd();
          const auto size = dsep + dupd;
          if (size < cutoff_size) {
            if (dsep) {
              f.piv = f.F11_.LU_seq();
              // TODO if (opts.replace_tiny_pivots()) { ...
              if (dupd) {
                f.F11_.solve_LU_in_place_seq(f.F12_, f.piv);
                gemm_seq(Trans::N, Trans::N, scalar_t(-1),
                         f.F21_, f.F12_, scalar_t(1.), f.F22_);
              }
            }
          }
        }
      }
      if (nnodes_big) {
        if (ldata[l].nnodes_8 || ldata[l].nnodes_16 || ldata[l].nnodes_32) {
          cuda::FrontData<scalar_t>* fdat = nullptr;
          cudaMallocManaged
            (&fdat, (ldata[l].nnodes_8+ldata[l].nnodes_16+ldata[l].nnodes_32)
             * sizeof(cuda::FrontData<scalar_t>));
          auto f8 = fdat;
          auto f16 = f8 + ldata[l].nnodes_8;
          auto f32 = f16 + ldata[l].nnodes_16;
          for (std::size_t n=0, n8=0, n16=0, n32=0; n<nnodes; n++) {
            auto& f = *(ldata[l].f[n]);
            const auto dsep = f.dim_sep();
            const auto dupd = f.dim_upd();
            const auto size = dsep + dupd;
            if (size >= cutoff_size) {
              cuda::FrontData<scalar_t>
                t(dsep, dupd, f.F11_.data(), f.F12_.data(),
                  f.F21_.data(), f.F22_.data(), ldata[l].dev_piv[n]);
              if (dsep <= 8)       f8[n8++] = t;
              else if (dsep <= 16) f16[n16++] = t;
              else if (dsep <= 32) f32[n32++] = t;
            }
          }
          cuda::factor_block_batch<scalar_t,8>(ldata[l].nnodes_8, f8);
          cuda::factor_block_batch<scalar_t,16>(ldata[l].nnodes_16, f16);
          cuda::factor_block_batch<scalar_t,32>(ldata[l].nnodes_32, f32);
          // TODO this can be removed (moved down), so that <= 32 and
          // > 32 can be done concurrently ? once all operations for
          // <= 32 are implemented in the manual kernel
          cudaDeviceSynchronize();
          cudaFree(fdat);
        }

        for (std::size_t n=0; n<nnodes; n++) {
          auto& f = *(ldata[l].f[n]);
          auto stream = n % max_streams;
          const auto dsep = f.dim_sep();
          const auto dupd = f.dim_upd();
          const auto size = dsep + dupd;
          if (size >= cutoff_size) {
            if (dsep > 32) {
              cuda::cusolverDngetrf
                (solver_handle[stream], dsep, dsep, f.F11_.data(), dsep,
                 ldata[l].dev_getrf_work + stream * ldata[l].getrf_work_size,
                 ldata[l].dev_piv[n], ldata[l].dev_getrf_err + stream);
              // TODO if (opts.replace_tiny_pivots()) { ...
              if (dupd) {
                cuda::cusolverDngetrs
                  (solver_handle[stream], CUBLAS_OP_N, dsep, dupd,
                   f.F11_.data(), dsep, ldata[l].dev_piv[n],
                   f.F12_.data(), dsep, ldata[l].dev_getrf_err + stream);
                gemm_cuda(blas_handle[stream], CUBLAS_OP_N, CUBLAS_OP_N,
                          scalar_t(-1.), f.F21_, f.F12_, scalar_t(1.), f.F22_);
              }
            }
          }
        }
      }
      cudaDeviceSynchronize();


      // // prefetch from device
      //// This generates an error: invalid argument
      // if (nnodes_big)
      //   cudaMemPrefetchAsync
      //     (ldata[l].f[0]->factor_mem_.get(),
      //      ldata[l].factor_size*sizeof(scalar_t), cudaCpuDeviceId, 0);

      // copy pivot vectors back from the device
#pragma omp parallel for
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        const auto size = dsep + dupd;
        if (size >= cutoff_size) {
          f.piv.resize(dsep);
          std::copy
            (ldata[l].dev_piv[n], ldata[l].dev_piv[n]+dsep, f.piv.data());
        }
      }

      // copy the factor memory from managed memory to regular memory,
      // free the managed memory.
      ldata[l].f[0]->factor_mem_ = std::unique_ptr<scalar_t>
        (new scalar_t[ldata[l].factor_size]);
      auto fmem = ldata[l].f[0]->factor_mem_.get();
      std::copy(factor_mem, factor_mem+ldata[l].factor_size, fmem);
      ldata[l].set_front_pointers(fmem);
      cudaFree(factor_mem);

      // count flops
      long long level_flops = ldata[l].total_flops();
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

    if (!dim_upd()) cudaFree(work_mem_old);
    else {
      // copy work_mem from device to host, because it will be needed
      // in the extend add to the parallel parent front
    }

    for (int i=0; i<max_streams; i++) {
      cudaStreamDestroy(streams[i]);
      cublasDestroy(blas_handle[i]);
      cusolverDnDestroy(solver_handle[i]);
    }

    // cudaProfilerStop();
  }

//   // using F22 for bupd makes this not const!!!!
//   template<typename scalar_t,typename integer_t> void
//   FrontalMatrixCUBLAS<scalar_t,integer_t>::multifrontal_solve
//   (DenseM_t& b) const {
//     // TODO get the options in here
//     const int cutoff_size = default_cuda_cutoff(); //opts.cuda_cutoff();
//     const int max_streams = default_cuda_streams(); //opts.cuda_streams();
//     int device_id;
//     cudaGetDevice(&device_id);
//     std::vector<cudaStream_t> streams(max_streams);
//     std::vector<cublasHandle_t> blas_handle(max_streams);
//     std::vector<cusolverDnHandle_t> solver_handle(max_streams);
//     for (int i=0; i<max_streams; i++) {
//       cudaStreamCreate(&streams[i]);
//       cublasCreate(&blas_handle[i]);
//       cusolverDnCreate(&solver_handle[i]);
//       cusolverDnSetStream(solver_handle[i], streams[i]);
//     }
//     int nrhs = b.cols();
//     int lvls = this->levels();
//     std::vector<LevelInfo_t> ldata(lvls);
//     std::size_t max_level_work_bytes = 0, max_level_factor_size = 0;
//     for (int l=lvls-1; l>=0; l--) {
//       std::vector<const F_t*> fp;
//       fp.reserve(2 << (l-1));
//       this->get_level_fronts(fp, l);
//       ldata[l].f.reserve(fp.size());
//       for (auto f : fp)
//         ldata[l].f.push_back(dynamic_cast<FC_t*>(const_cast<F_t*>(f)));
//       auto nnodes = ldata[l].f.size();
//       ldata[l].dev_piv.resize(nnodes);
//       ldata[l].bloc.resize(nnodes);
//       for (std::size_t n=0; n<nnodes; n++) {
//         auto& f = *(ldata[l].f[n]);
//         const auto dsep = f.dim_sep();
//         const auto dupd = f.dim_upd();
//         const auto size = dsep + dupd;
//         ldata[l].factor_size += dsep*dsep + 2*dsep*dupd;
//         ldata[l].bloc_size += dsep*nrhs;
//         ldata[l].bupd_size += dupd*nrhs;
//         ldata[l].piv_size += dsep;
//         if (size < cutoff_size)
//           ldata[l].nnodes_small++;
//       }
//       ldata[l].total_work_bytes =
//         sizeof(scalar_t) * (ldata[l].bloc_size + ldata[l].bupd_size) +
//         sizeof(int) * (ldata[l].piv_size) +
//         max_streams * sizeof(int);  // CUDA getrs error code
//       max_level_work_bytes = std::max
//         (max_level_work_bytes, ldata[l].total_work_bytes);
//       max_level_factor_size = std::max
//         (max_level_factor_size, ldata[l].factor_size);
//     }
//     // ensure alignment?
//     max_level_work_bytes += max_level_work_bytes % 8;

//     void* all_work_mem = nullptr;
//     cudaMallocManaged
//       (&all_work_mem, 2 * max_level_work_bytes
//        + sizeof(scalar_t) * max_level_factor_size);
//     void* work_mem[2] =
//       {all_work_mem, (char*)all_work_mem + max_level_work_bytes};
//     scalar_t* factor_mem =
//       static_cast<scalar_t*>
//       (static_cast<void*>((char*)all_work_mem + 2 * max_level_work_bytes));

//     ////////////////////////////////////////////////////////////////
//     //////////////     forward solve      //////////////////////////
//     ////////////////////////////////////////////////////////////////
//     for (int l=lvls-1; l>=0; l--) {
//       auto nnodes = ldata[l].f.size();
//       auto nnodes_small = ldata[l].nnodes_small;
//       auto wmem = work_mem[l % 2];
//       auto fmem = factor_mem;
//       if (nnodes_small != nnodes)
//         std::copy
//           (ldata[l].f[0]->factor_mem_.get(),
//            ldata[l].f[0]->factor_mem_.get()+ldata[l].factor_size, fmem);

//       // initialize pointers to data for frontal matrices
//       for (std::size_t n=0; n<nnodes; n++) {
//         auto& f = *(ldata[l].f[n]);
//         const int dsep = f.dim_sep();
//         const int dupd = f.dim_upd();
//         ldata[l].bloc[n] =
//           DenseMW_t(dsep, nrhs, static_cast<scalar_t*>(wmem), dsep);
//         wmem = static_cast<scalar_t*>(wmem) + dsep*nrhs;
//         if (nnodes_small != nnodes) {
//           f.F11_ = DenseMW_t(dsep, dsep, fmem, dsep); fmem += dsep*dsep;
//           f.F12_ = DenseMW_t(dsep, dupd, fmem, dsep); fmem += dsep*dupd;
//           f.F21_ = DenseMW_t(dupd, dsep, fmem, dupd); fmem += dupd*dsep;
//         }
//         if (dupd) {
//           f.F22_ = DenseMW_t(dupd, nrhs, static_cast<scalar_t*>(wmem), dupd);
//           wmem = static_cast<scalar_t*>(wmem) + dupd*nrhs;
//         }
//       }
//       for (std::size_t n=0; n<nnodes; n++) {
//         auto& f = *(ldata[l].f[n]);
//         ldata[l].dev_piv[n] = static_cast<int*>(wmem);
//         wmem = static_cast<int*>(wmem) + f.dim_sep();
//         std::copy(f.piv.begin(), f.piv.end(), ldata[l].dev_piv[n]);
//       }
//       ldata[l].dev_getrf_err = static_cast<int*>(wmem);
//       wmem = static_cast<int*>(wmem) + max_streams;

//       // extend from children fronts
// #pragma omp parallel for
//       for (std::size_t n=0; n<nnodes; n++) {
//         auto& f = *(ldata[l].f[n]);
//         f.F22_.zero();
//         if (f.lchild_)
//           f.lchild_->extend_add_b
//             (b, f.F22_, dynamic_cast<FC_t*>(f.lchild_.get())->F22_, &f);
//         if (f.rchild_)
//           f.rchild_->extend_add_b
//             (b, f.F22_, dynamic_cast<FC_t*>(f.rchild_.get())->F22_, &f);
//       }

//       // copy right hand side to (managed) device memory
// #pragma omp parallel for
//       for (std::size_t n=0; n<nnodes; n++) {
//         auto& f = *(ldata[l].f[n]);
//         copy(f.dim_sep(), nrhs, b, f.sep_begin_, 0, ldata[l].bloc[n], 0, 0);
//       }

//       if (nnodes_small) {
// #pragma omp parallel for
//         for (std::size_t n=0; n<nnodes; n++) {
//           auto& f = *(ldata[l].f[n]);
//           const auto dsep = f.dim_sep();
//           const auto dupd = f.dim_upd();
//           const auto size = dsep + dupd;
//           if (size < cutoff_size) {
//             // call the blas/lapack routines directly to avoid
//             // overhead of tasking/checking whether in parallel region
//             if (dsep) {
//               int flag;
//               blas::getrs
//                 ('N', dsep, nrhs, f.F11_.data(), dsep,
//                  f.piv.data(), ldata[l].bloc[n].data(), dsep, &flag);
//               if (dupd) {
//                 if (nrhs == 1)
//                   blas::gemv
//                     ('N', dupd, dsep, scalar_t(-1.), f.F21_.data(), dupd,
//                      ldata[l].bloc[n].data(), 1, scalar_t(1.),
//                      f.F22_.data(), 1);
//                 else
//                   blas::gemm
//                     ('N', 'N', dupd, nrhs, dsep, scalar_t(-1.),
//                      f.F21_.data(), dupd, ldata[l].bloc[n].data(), dsep,
//                      scalar_t(1.), f.F22_.data(), dupd);
//               }
//             }
//           }
//         }
//       }
//       for (std::size_t n=0; n<nnodes; n++) {
//         auto& f = *(ldata[l].f[n]);
//         auto stream = n % max_streams;
//         const auto dsep = f.dim_sep();
//         const auto dupd = f.dim_upd();
//         const auto size = dsep + dupd;
//         if (size >= cutoff_size) {
//           if (dsep) {
//             cuda::cusolverDngetrs
//               (solver_handle[stream], CUBLAS_OP_N, dsep, nrhs,
//                f.F11_.data(), dsep, ldata[l].dev_piv[n],
//                ldata[l].bloc[n].data(), dsep,
//                ldata[l].dev_getrf_err + stream);
//             if (dupd) {
//               if (nrhs == 1)
//                 cuda::cublasgemv
//                   (blas_handle[stream], CUBLAS_OP_N,
//                    dupd, dsep, scalar_t(-1.), f.F21_.data(), dupd,
//                    ldata[l].bloc[n].data(), 1,
//                    scalar_t(1.), f.F22_.data(), 1);
//               else
//                 cuda::cublasgemm
//                   (blas_handle[stream], CUBLAS_OP_N, CUBLAS_OP_N,
//                    dupd, nrhs, dsep, scalar_t(-1.), f.F21_.data(), dupd,
//                    ldata[l].bloc[n].data(), dsep,
//                    scalar_t(1.), f.F22_.data(), dupd);
//             }
//           }
//         }
//       }
//       cudaDeviceSynchronize();

//       // copy right hand side back from (managed) device memory
// #pragma omp parallel for
//       for (std::size_t n=0; n<nnodes; n++) {
//         auto& f = *(ldata[l].f[n]);
//         copy(f.dim_sep(), nrhs, ldata[l].bloc[n], 0, 0, b, f.sep_begin_, 0);
//       }
//     }

//     ////////////////////////////////////////////////////////////////
//     //////////////     backward solve     //////////////////////////
//     ////////////////////////////////////////////////////////////////
//     for (int l=0; l<lvls; l++) {
//       auto nnodes = ldata[l].f.size();
//       auto nnodes_small = ldata[l].nnodes_small;
//       auto wmem = work_mem[l % 2];
//       if (nnodes_small != nnodes)
//         std::copy
//           (ldata[l].f[0]->factor_mem_.get(),
//            ldata[l].f[0]->factor_mem_.get()+ldata[l].factor_size, factor_mem);


//       // initialize pointers to data for frontal matrices
//       ldata[l].bloc.resize(nnodes);
//       for (std::size_t n=0; n<nnodes; n++) {
//         auto& f = *(ldata[l].f[n]);
//         const int dsep = f.dim_sep();
//         const int dupd = f.dim_upd();
//         ldata[l].bloc[n] =
//           DenseMW_t(dsep, nrhs, static_cast<scalar_t*>(wmem), dsep);
//         wmem = static_cast<scalar_t*>(wmem) + dsep*nrhs;
//         if (dupd) {
//           f.F22_ = DenseMW_t(dupd, nrhs, static_cast<scalar_t*>(wmem), dupd);
//           wmem = static_cast<scalar_t*>(wmem) + dupd*nrhs;
//         }
//       }

//       // extract from parent nodes
//       if (l > 0) {
// #pragma omp parallel for
//         for (std::size_t n=0; n<ldata[l-1].f.size(); n++) {
//           auto& f = *(ldata[l-1].f[n]);
//           if (f.lchild_)
//             f.lchild_->extract_b
//               (b, f.F22_, dynamic_cast<FC_t*>(f.lchild_.get())->F22_, &f);
//           if (f.rchild_)
//             f.rchild_->extract_b
//               (b, f.F22_, dynamic_cast<FC_t*>(f.rchild_.get())->F22_, &f);
//         }
//       }

//       // copy right hand side to (managed) device memory
// #pragma omp parallel for
//       for (std::size_t n=0; n<nnodes; n++) {
//         auto& f = *(ldata[l].f[n]);
//         copy(f.dim_sep(), nrhs, b, f.sep_begin_, 0, ldata[l].bloc[n], 0, 0);
//       }

//       if (nnodes_small) {
// #pragma omp parallel for
//         for (std::size_t n=0; n<nnodes; n++) {
//           auto& f = *(ldata[l].f[n]);
//           const auto dsep = f.dim_sep();
//           const auto dupd = f.dim_upd();
//           const auto size = dsep + dupd;
//           if (size < cutoff_size) {
//             if (dsep && dupd) {
//               if (nrhs == 1)
//                 blas::gemv
//                   ('N', dsep, dupd, scalar_t(-1.), f.F12_.data(), dsep,
//                    f.F22_.data(), 1, scalar_t(1.),
//                    ldata[l].bloc[n].data(), 1);
//               else
//                 blas::gemm
//                   ('N', 'N', dsep, nrhs, dupd, scalar_t(-1.),
//                    f.F12_.data(), dsep, f.F22_.data(), dupd,
//                    scalar_t(1.), ldata[l].bloc[n].data(), dsep);
//             }
//           }
//         }
//       }
//       for (std::size_t n=0; n<nnodes; n++) {
//         auto& f = *(ldata[l].f[n]);
//         auto stream = n % max_streams;
//         const auto dsep = f.dim_sep();
//         const auto dupd = f.dim_upd();
//         const auto size = dsep + dupd;
//         if (size >= cutoff_size) {
//           if (dsep && dupd) {
//             if (nrhs == 1)
//               cuda::cublasgemv
//                 (blas_handle[stream], CUBLAS_OP_N,
//                  dsep, dupd, scalar_t(-1.), f.F12_.data(), dsep,
//                  f.F22_.data(), 1, scalar_t(1.),
//                  ldata[l].bloc[n].data(), 1);
//             else
//               cuda::cublasgemm
//                 (blas_handle[stream], CUBLAS_OP_N, CUBLAS_OP_N,
//                  dsep, nrhs, dupd, scalar_t(-1.), f.F12_.data(), dsep,
//                  f.F22_.data(), dupd, scalar_t(1.),
//                  ldata[l].bloc[n].data(), dsep);
//           }
//         }
//       }
//       cudaDeviceSynchronize();

//       // copy right hand side back from (managed) device memory
// #pragma omp parallel for
//       for (std::size_t n=0; n<nnodes; n++) {
//         auto& f = *(ldata[l].f[n]);
//         copy(f.dim_sep(), nrhs, ldata[l].bloc[n], 0, 0, b, f.sep_begin_, 0);
//       }

//       // revert pointers to the original (not managed)
//       // memory
//       if (nnodes_small != nnodes) {
//         auto fmem = ldata[l].f[0]->factor_mem_.get();
//         for (std::size_t n=0; n<nnodes; n++) {
//           auto& f = *(ldata[l].f[n]);
//           const int dsep = f.dim_sep();
//           const int dupd = f.dim_upd();
//           f.F11_ = DenseMW_t(dsep, dsep, fmem, dsep); fmem += dsep*dsep;
//           f.F12_ = DenseMW_t(dsep, dupd, fmem, dsep); fmem += dsep*dupd;
//           f.F21_ = DenseMW_t(dupd, dsep, fmem, dupd); fmem += dupd*dsep;
//         }
//       }
//     }

//     cudaFree(all_work_mem);
//     for (int i=0; i<max_streams; i++) {
//       cudaStreamDestroy(streams[i]);
//       cublasDestroy(blas_handle[i]);
//       cusolverDnDestroy(solver_handle[i]);
//     }
//   }


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
