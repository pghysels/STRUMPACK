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
#ifndef FRONTAL_MATRIX_CUBLAS_HPP
#define FRONTAL_MATRIX_CUBLAS_HPP

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>

#include "misc/TaskTimer.hpp"
#include "dense/BLASLAPACKWrapper.hpp"
#include "CompressedSparseMatrix.hpp"
#include "MatrixReordering.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#endif
#include "dense/CUDAWrapper.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixCUBLAS;
  template<typename scalar_t, typename integer_t> class LevelInfo {
  public:
    std::vector<FrontalMatrixCUBLAS<scalar_t,integer_t>*> f;
    std::size_t factor_mem_size = 0, schur_mem_size = 0, piv_mem_size = 0,
      total_work_mem_size = 0, nnodes_small = 0, 
      bloc_mem_size = 0, bupd_mem_size = 0;
    scalar_t* dev_getrf_work = nullptr;
    int* dev_getrf_err = nullptr;
    std::vector<int*> dev_piv;
    std::vector<DenseMatrixWrapper<scalar_t>> bloc;
    int getrf_work_size = 0;
  };

  template<typename scalar_t,typename integer_t> class FrontalMatrixCUBLAS
    : public FrontalMatrix<scalar_t,integer_t> {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FC_t = FrontalMatrixCUBLAS<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using LevelInfo_t = LevelInfo<scalar_t,integer_t>;
#if defined(STRUMPACK_USE_MPI)
    using ExtAdd = ExtendAdd<scalar_t,integer_t>;
#endif

  public:
    FrontalMatrixCUBLAS
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd);

    void release_work_memory() override;

    void extend_add_to_dense
    (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
     const F_t* p, int task_depth) override;

    void sample_CB
    (const SPOptions<scalar_t>& opts, const DenseM_t& R,
     DenseM_t& Sr, DenseM_t& Sc, F_t* pa, int task_depth) override;
    void sample_CB
    (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
     int task_depth=0) const override;

    void sample_CB_to_F11
    (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
     int task_depth=0) const override;
    void sample_CB_to_F12
    (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
     int task_depth=0) const override;
    void sample_CB_to_F21
    (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
     int task_depth=0) const override;
    void sample_CB_to_F22
    (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
     int task_depth=0) const override;

    void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level=0, int task_depth=0) override;

    void forward_multifrontal_solve
    (DenseM_t& b, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;
    void forward_multifrontal_solve_CPU
    (DenseM_t& b, DenseM_t* work, int etree_level=0,
     int task_depth=0) const;

    void backward_multifrontal_solve
    (DenseM_t& y, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;

    void extract_CB_sub_matrix
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DenseM_t& B, int task_depth) const override;

    std::string type() const override { return "FrontalMatrixCUBLAS"; }

#if defined(STRUMPACK_USE_MPI)
    void extend_add_copy_to_buffers
    (std::vector<std::vector<scalar_t>>& sbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa) const override {
      ExtAdd::extend_add_seq_copy_to_buffers(F22_, sbuf, pa, this);
    }
#endif

  private:
    std::unique_ptr<scalar_t[], std::function<void(scalar_t*)>> factor_mem_;
    DenseMW_t F11_, F12_, F21_, F22_;
    std::vector<int> piv; // regular int because it is passed to BLAS

    FrontalMatrixCUBLAS(const FrontalMatrixCUBLAS&) = delete;
    FrontalMatrixCUBLAS& operator=(FrontalMatrixCUBLAS const&) = delete;

    void factor_phase1
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level, int task_depth);
    void factor_phase2
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level, int task_depth);

    void fwd_solve_phase2
    (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const;
    void bwd_solve_phase1
    (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const;

    using F_t::lchild_;
    using FrontalMatrix<scalar_t,integer_t>::rchild_;
    using FrontalMatrix<scalar_t,integer_t>::dim_sep;
    using FrontalMatrix<scalar_t,integer_t>::dim_upd;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixCUBLAS<scalar_t,integer_t>::FrontalMatrixCUBLAS
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixCUBLAS<scalar_t,integer_t>::release_work_memory() {
    F22_.clear();
  }

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
    int cutoff_size = opts.cuda_cutoff();
    using uniq_scalar_t = std::unique_ptr<scalar_t[],std::function<void(scalar_t*)>>;
    auto cuda_deleter = [](void* ptr) { cudaFree(ptr); };
    int device_id;
    cudaGetDevice(&device_id);
    const int max_streams = opts.cuda_streams();
    std::vector<cudaStream_t> streams(max_streams);
    std::vector<cublasHandle_t> blas_handle(max_streams);
    std::vector<cusolverDnHandle_t> solver_handle(max_streams);
    for (int i=0; i<max_streams; i++) {
      cudaStreamCreate(&streams[i]);
      cublasCreate(&blas_handle[i]);
      cusolverDnCreate(&solver_handle[i]);
      cusolverDnSetStream(solver_handle[i], streams[i]);
    }
    int lvls = this->levels();
    std::vector<LevelInfo_t> ldata(lvls);
    std::size_t max_level_work_mem_size = 0;
    for (int lvl=0; lvl<lvls; lvl++) {
      int l = lvls - lvl - 1;
      std::vector<F_t*> fp;
      fp.reserve(2 << (l-1));
      this->get_level_fronts(fp, l);
      ldata[l].f.reserve(fp.size());
      using FC_t = FrontalMatrixCUBLAS<scalar_t,integer_t>;
      for (auto f : fp)
        ldata[l].f.push_back(dynamic_cast<FC_t*>(f));
      auto nnodes = ldata[l].f.size();
      int max_dsep = 0, node_max_dsep = 0;
      std::size_t avg_size_sep = 0, avg_size_upd = 0;
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        const auto size = dsep + dupd;
        avg_size_sep += dsep;
        avg_size_upd += dupd;
        ldata[l].factor_mem_size += dsep*dsep + 2*dsep*dupd;
        ldata[l].schur_mem_size += dupd*dupd;
        ldata[l].piv_mem_size += dsep;
        if (size < cutoff_size)
          ldata[l].nnodes_small++;
        if (dsep > max_dsep) {
          node_max_dsep = n;
          max_dsep = dsep;
        }
      }
      // if (opts.verbose())
      //        std::cout << "# level " << l << " avg size = "
      //                  << (avg_size_sep+avg_size_upd) / nnodes
      //                  << " = " << avg_size_sep / nnodes << " + "
      //                  << avg_size_upd / nnodes << std::endl;
      cuda::cusolverDngetrf_bufferSize
        (solver_handle[0], max_dsep, max_dsep,
         /*f.F11_.data()*/ static_cast<scalar_t*>(nullptr),
         max_dsep, &(ldata[l].getrf_work_size));
      ldata[l].total_work_mem_size =
        sizeof(scalar_t) * ldata[l].schur_mem_size +
	//sizeof(int*) * nnodes + // dev_piv pointers!!
        sizeof(int) * (ldata[l].piv_mem_size) +
        // CUDA getrf work size and CUDA getrf error code
        max_streams * (sizeof(scalar_t) * ldata[l].getrf_work_size + sizeof(int));
      max_level_work_mem_size = std::max
        (max_level_work_mem_size, ldata[l].total_work_mem_size);
    }
    // ensure alignment?
    max_level_work_mem_size += max_level_work_mem_size % 8;

    void* all_work_mem = nullptr;
    cudaMallocManaged(&all_work_mem, max_level_work_mem_size*2);
    void* work_mem[2] = {all_work_mem, (char*)all_work_mem + max_level_work_mem_size};

    for (int lvl=0; lvl<lvls; lvl++) {
      TaskTimer tl("");
      tl.start();
      int l = lvls - lvl - 1;
      auto nnodes = ldata[l].f.size();
      auto nnodes_small = ldata[l].nnodes_small;
      if (opts.verbose())
        std::cout << "#  level " << l << " of " << lvls
                  << " has " << ldata[l].f.size() << " nodes and "
                  << ldata[l].nnodes_small << " small nodes, needs "
                  << ldata[l].factor_mem_size * sizeof(scalar_t) / 1.e6
                  << " MB for factors, "
                  << ldata[l].schur_mem_size * sizeof(scalar_t) / 1.e6
                  << " MB for Schur complements" << std::endl;

      scalar_t* fmem = nullptr;
      cudaMallocManaged(&fmem, ldata[l].factor_mem_size*sizeof(scalar_t));
      ldata[l].f[0]->factor_mem_ = uniq_scalar_t(fmem, cuda_deleter);
      auto wmem = work_mem[l % 2];

      // initialize pointers to data for frontal matrices
      // ldata[l].dev_piv = static_cast<int**>(wmem);
      // wmem = static_cast<int**>(wmem) + nnodes;
      ldata[l].dev_piv.resize(nnodes);
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const int dsep = f.dim_sep();
        const int dupd = f.dim_upd();
        f.F11_ = DenseMW_t(dsep, dsep, fmem, dsep); fmem += dsep*dsep;
        f.F12_ = DenseMW_t(dsep, dupd, fmem, dsep); fmem += dsep*dupd;
        f.F21_ = DenseMW_t(dupd, dsep, fmem, dupd); fmem += dupd*dsep;
        if (dupd) {
          f.F22_ = DenseMW_t(dupd, dupd, static_cast<scalar_t*>(wmem), dupd);
          wmem = static_cast<scalar_t*>(wmem) + dupd*dupd;
        }
      }
      ldata[l].dev_getrf_work = static_cast<scalar_t*>(wmem);
      wmem = static_cast<scalar_t*>(wmem) + max_streams * ldata[l].getrf_work_size;
      for (std::size_t n=0; n<nnodes; n++) {
        ldata[l].dev_piv[n] = static_cast<int*>(wmem);
        wmem = static_cast<int*>(wmem) + ldata[l].f[n]->dim_sep();
      }
      ldata[l].dev_getrf_err = static_cast<int*>(wmem);
      wmem = static_cast<int*>(wmem) + max_streams;

      // build front from sparse matrix and extend-add
      // TODO threading?
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
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

      // prefetch to GPU
      if (nnodes_small < nnodes) {
	cudaMemPrefetchAsync
	  (work_mem[l % 2], max_level_work_mem_size, device_id, 0);
	cudaMemPrefetchAsync
	  (ldata[l].f[0]->factor_mem_.get(),
	   ldata[l].factor_mem_size*sizeof(scalar_t), device_id, 0);
      }

      // count flops
      long long level_flops = 0;
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        level_flops += LU_flops(f.F11_) +
          gemm_flops(Trans::N, Trans::N, scalar_t(-1.), f.F21_, f.F12_, scalar_t(1.)) +
          trsm_flops(Side::L, scalar_t(1.), f.F11_, f.F12_) +
          trsm_flops(Side::R, scalar_t(1.), f.F11_, f.F21_);
      }
      STRUMPACK_FULL_RANK_FLOPS(level_flops);

      if (nnodes_small) {
        for (std::size_t n=0; n<nnodes; n++) {
          auto& f = *(ldata[l].f[n]);
          auto stream = n % max_streams;
          const auto dsep = f.dim_sep();
          const auto dupd = f.dim_upd();
          const auto size = dsep + dupd;
          if (size < cutoff_size) {
            if (dsep) {
              f.piv = f.F11_.LU(0);
              // TODO if (opts.replace_tiny_pivots()) { ...
              if (dupd) {
                f.F11_.solve_LU_in_place(f.F12_, f.piv, 0);
                gemm(Trans::N, Trans::N, scalar_t(-1.), f.F21_, f.F12_,
                     scalar_t(1.), f.F22_, 0);
              }
            }
          }
        }
      }

      // TODO why the synchronize? after the prefetch??
      //cudaDeviceSynchronize();

      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        auto stream = n % max_streams;
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        const auto size = dsep + dupd;
        if (size >= cutoff_size) {
          if (dsep) {
            auto LU_stat = cuda::cusolverDngetrf
              (solver_handle[stream], dsep, dsep, f.F11_.data(), dsep,
               ldata[l].dev_getrf_work + stream * ldata[l].getrf_work_size,
               ldata[l].dev_piv[n], ldata[l].dev_getrf_err + stream);
            // TODO if (opts.replace_tiny_pivots()) { ...
            if (dupd) {
              auto LU_stat = cuda::cusolverDngetrs
                (solver_handle[stream], CUBLAS_OP_N, dsep, dupd,
                 f.F11_.data(), dsep, ldata[l].dev_piv[n], f.F12_.data(), dsep,
                 ldata[l].dev_getrf_err + stream);
              auto stat = cuda::cublasgemm
                (blas_handle[stream], CUBLAS_OP_N, CUBLAS_OP_N,
                 dupd, dupd, dsep, scalar_t(-1.), f.F21_.data(), dupd,
                 f.F12_.data(), dsep, scalar_t(1.), f.F22_.data(), dupd);
            }
          }
        }
      }
      cudaDeviceSynchronize();

      // prefetch from device
      if (nnodes_small < nnodes)
	cudaMemPrefetchAsync
	  (ldata[l].f[0]->factor_mem_.get(),
	   ldata[l].factor_mem_size*sizeof(scalar_t), cudaCpuDeviceId, 0);

      // copy pivot vectors back from the device
      for (std::size_t n=0, idx=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        const auto size = dsep + dupd;
        if (size >= cutoff_size) {
          f.piv.resize(dsep);
          std::copy(ldata[l].dev_piv[n], ldata[l].dev_piv[n]+dsep, f.piv.data());
        }
      }

      // copy the factor memory from managed memory to regular memory,
      // free the managed memory.
      auto def_deleter = [](scalar_t* ptr) { delete[] ptr; };
      scalar_t* fmem_reg = new scalar_t[ldata[l].factor_mem_size];
      fmem = ldata[l].f[0]->factor_mem_.get();
      std::copy(fmem, fmem+ldata[l].factor_mem_size, fmem_reg);
      ldata[l].f[0]->factor_mem_ = uniq_scalar_t(fmem_reg, def_deleter);
      fmem = ldata[l].f[0]->factor_mem_.get();
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const int dsep = f.dim_sep();
        const int dupd = f.dim_upd();
        f.F11_ = DenseMW_t(dsep, dsep, fmem, dsep); fmem += dsep*dsep;
        f.F12_ = DenseMW_t(dsep, dupd, fmem, dsep); fmem += dsep*dupd;
        f.F21_ = DenseMW_t(dupd, dsep, fmem, dupd); fmem += dupd*dsep;
      }

      if (opts.verbose()) {
        auto level_time = tl.elapsed();
        std::cout << "#   GPU Factorization complete, took: "
                  << level_time << " seconds, "
                  << level_flops / 1.e9 << " GFLOPS, "
                  << (float(level_flops) / level_time) / 1.e9
                  << " GFLOP/s" << std::endl;
      }
    }

    cudaFree(all_work_mem);
    for (int i=0; i<max_streams; i++) {
      cudaStreamDestroy(streams[i]);
      cublasDestroy(blas_handle[i]);
      cusolverDnDestroy(solver_handle[i]);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixCUBLAS<scalar_t,integer_t>::forward_multifrontal_solve_CPU
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



  // using F22 for bupd makes this not const!!!!
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixCUBLAS<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    // forward_multifrontal_solve_CPU(b, work, etree_level, task_depth);
    // return;

    // TODO get the options in here
    int cutoff_size = 200; //opts.cuda_cutoff();
    using uniq_scalar_t = std::unique_ptr<scalar_t[],std::function<void(scalar_t*)>>;
    auto cuda_deleter = [](void* ptr) { cudaFree(ptr); };
    int device_id;
    cudaGetDevice(&device_id);
    const int max_streams = 10; //opts.cuda_streams();
    std::vector<cudaStream_t> streams(max_streams);
    std::vector<cublasHandle_t> blas_handle(max_streams);
    std::vector<cusolverDnHandle_t> solver_handle(max_streams);
    for (int i=0; i<max_streams; i++) {
      cudaStreamCreate(&streams[i]);
      cublasCreate(&blas_handle[i]);
      cusolverDnCreate(&solver_handle[i]);
      cusolverDnSetStream(solver_handle[i], streams[i]);
    }
    int nrhs = b.cols();
    int lvls = this->levels();
    std::vector<LevelInfo_t> ldata(lvls);
    std::size_t max_level_work_mem_size = 0;
    for (int lvl=0; lvl<lvls; lvl++) {
      int l = lvls - lvl - 1;
      std::vector<const F_t*> fp;
      fp.reserve(2 << (l-1));
      this->get_level_fronts(fp, l);
      ldata[l].f.reserve(fp.size());
      for (auto f : fp)
        ldata[l].f.push_back(dynamic_cast<FC_t*>(const_cast<F_t*>(f)));
      auto nnodes = ldata[l].f.size();
      int max_dsep = 0, node_max_dsep = 0;
      std::size_t avg_size_sep = 0, avg_size_upd = 0;
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        const auto size = dsep + dupd;
        ldata[l].bloc_mem_size += dsep*nrhs;
        ldata[l].bupd_mem_size += dupd*nrhs;
        ldata[l].piv_mem_size += dsep;
        if (size < cutoff_size)
          ldata[l].nnodes_small++;
      }
      ldata[l].total_work_mem_size =
        sizeof(scalar_t) * (ldata[l].bloc_mem_size + ldata[l].bupd_mem_size) +
        sizeof(int) * (ldata[l].piv_mem_size) +
        max_streams * sizeof(int);  // CUDA getrs error code
      max_level_work_mem_size = std::max
        (max_level_work_mem_size, ldata[l].total_work_mem_size);
    }
    // ensure alignment?
    max_level_work_mem_size += max_level_work_mem_size % 8;

    void* all_work_mem = nullptr;
    cudaMallocManaged(&all_work_mem, max_level_work_mem_size*2);
    void* work_mem[2] = {all_work_mem, (char*)all_work_mem + max_level_work_mem_size};

    for (int lvl=0; lvl<lvls; lvl++) {
      int l = lvls - lvl - 1;

      // if (opts.verbose())
      // std::cout << "#  level " << l << " of " << lvls
      // 		<< " has " << ldata[l].f.size() << " nodes and "
      // 		<< ldata[l].nnodes_small << " small nodes" << std::endl;

      // TaskTimer tl("");
      // tl.start();
      auto nnodes = ldata[l].f.size();
      auto nnodes_small = ldata[l].nnodes_small;
      auto wmem = work_mem[l % 2];

      // initialize pointers to data for frontal matrices
      ldata[l].dev_piv.resize(nnodes);
      ldata[l].bloc.resize(nnodes);
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const int dsep = f.dim_sep();
        const int dupd = f.dim_upd();
	ldata[l].bloc[n] = DenseMW_t(dsep, nrhs, static_cast<scalar_t*>(wmem), dsep);
	wmem = static_cast<scalar_t*>(wmem) + dsep*nrhs;
	if (dupd) {
	  f.F22_ = DenseMW_t(dupd, nrhs, static_cast<scalar_t*>(wmem), dupd);
	  wmem = static_cast<scalar_t*>(wmem) + dupd*nrhs;
	}
      }
      for (std::size_t n=0, idx=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        ldata[l].dev_piv[n] = static_cast<int*>(wmem);
        wmem = static_cast<int*>(wmem) + f.dim_sep();
	std::copy(f.piv.begin(), f.piv.end(), ldata[l].dev_piv[n]);
      }
      ldata[l].dev_getrf_err = static_cast<int*>(wmem);
      wmem = static_cast<int*>(wmem) + max_streams;

      // auto f0 = params::flops.load();

      // extend from children fronts
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
	f.F22_.zero();
	if (f.lchild_)
	  f.lchild_->extend_add_b
	    (b, f.F22_, dynamic_cast<FC_t*>(f.lchild_.get())->F22_, &f);
	if (f.rchild_)
	  f.rchild_->extend_add_b
	    (b, f.F22_, dynamic_cast<FC_t*>(f.rchild_.get())->F22_, &f);
      }

      // copy right hand side to (managed) device memory
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
	copy(f.dim_sep(), nrhs, b, f.sep_begin_, 0, ldata[l].bloc[n], 0, 0);
      }

      // prefetch to GPU
      if (nnodes_small < nnodes)
	cudaMemPrefetchAsync
	  (work_mem[l % 2], max_level_work_mem_size, device_id, 0);

      if (nnodes_small) {
        for (std::size_t n=0; n<nnodes; n++) {
          auto& f = *(ldata[l].f[n]);
          auto stream = n % max_streams;
          const auto dsep = f.dim_sep();
          const auto dupd = f.dim_upd();
          const auto size = dsep + dupd;
          if (size < cutoff_size) {
	    f.F11_.solve_LU_in_place(ldata[l].bloc[n], f.piv, task_depth);
	    if (dupd) {
	      if (nrhs == 1)
		gemv(Trans::N, scalar_t(-1.), f.F21_, ldata[l].bloc[n],
		     scalar_t(1.), f.F22_, task_depth);
	      else
		gemm(Trans::N, Trans::N, scalar_t(-1.), f.F21_, 
		     ldata[l].bloc[n], scalar_t(1.), f.F22_, task_depth);
	    }
          }
        }
      }

      // TODO why the synchronize? after the prefetch??
      //cudaDeviceSynchronize();

      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        auto stream = n % max_streams;
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        const auto size = dsep + dupd;
        if (size >= cutoff_size) {
          if (dsep) {
	    auto LU_stat = cuda::cusolverDngetrs
	      (solver_handle[stream], CUBLAS_OP_N, dsep, nrhs,
	       f.F11_.data(), dsep, ldata[l].dev_piv[n], 
	       ldata[l].bloc[n].data(), dsep,
	       ldata[l].dev_getrf_err + stream);
            if (dupd) {
	      // if (nrhs == 1) {
	      // } else {
	      auto stat = cuda::cublasgemm
		(blas_handle[stream], CUBLAS_OP_N, CUBLAS_OP_N,
		 dupd, nrhs, dsep, scalar_t(-1.), f.F21_.data(), dupd,
		 ldata[l].bloc[n].data(), dsep, scalar_t(1.), f.F22_.data(), dupd);
	      // }
	    }
          }
        }
      }
      cudaDeviceSynchronize();

      // TODO prefetch from device

      // copy right hand side back from (managed) device memory
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const int dsep = f.dim_sep();
        const int dupd = f.dim_upd();
	copy(dsep, nrhs, ldata[l].bloc[n], 0, 0, b, f.sep_begin_, 0);
      }

      // if (opts.verbose()) {
      // auto level_flops = params::flops - f0;
      // auto level_time = tl.elapsed();
      // std::cout << "#   GPU forward solve complete, took: "
      // 		<< level_time << " seconds, "
      // 		<< level_flops / 1.e9 << " GFLOPS, "
      // 		<< (float(level_flops) / level_time) / 1.e9
      // 		<< " GFLOP/s" << std::endl;
      // }
    }

    cudaFree(all_work_mem);
    for (int i=0; i<max_streams; i++) {
      cudaStreamDestroy(streams[i]);
      cublasDestroy(blas_handle[i]);
      cusolverDnDestroy(solver_handle[i]);
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

} // end namespace strumpack

#endif
