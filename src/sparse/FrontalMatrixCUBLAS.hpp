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
    std::size_t factor_size = 0, schur_size = 0, piv_size = 0,
      total_work_bytes = 0, nnodes_small = 0,
      bloc_size = 0, bupd_size = 0;
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
    using uniq_scalar_t = std::unique_ptr
      <scalar_t[], std::function<void(scalar_t*)>>;
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

    void multifrontal_solve(DenseM_t& b) const override;

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
    uniq_scalar_t factor_mem_;
    DenseMW_t F11_, F12_, F21_, F22_;
    std::vector<int> piv; // regular int because it is passed to BLAS

    FrontalMatrixCUBLAS(const FrontalMatrixCUBLAS&) = delete;
    FrontalMatrixCUBLAS& operator=(FrontalMatrixCUBLAS const&) = delete;

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
    std::size_t max_level_work_bytes = 0;
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
        ldata[l].factor_size += dsep*dsep + 2*dsep*dupd;
        ldata[l].schur_size += dupd*dupd;
        ldata[l].piv_size += dsep;
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
      ldata[l].total_work_bytes =
        sizeof(scalar_t) * ldata[l].schur_size +
        sizeof(int) * (ldata[l].piv_size) +
        // CUDA getrf work size and CUDA getrf error code
        max_streams * (sizeof(scalar_t) * ldata[l].getrf_work_size + sizeof(int));
      max_level_work_bytes = std::max
        (max_level_work_bytes, ldata[l].total_work_bytes);
    }
    // ensure alignment?
    max_level_work_bytes += max_level_work_bytes % 8;

    void* all_work_mem = nullptr;
    cudaMallocManaged(&all_work_mem, 2 * max_level_work_bytes);
    void* work_mem[2] =
      {all_work_mem, (char*)all_work_mem + max_level_work_bytes};

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
                  << ldata[l].factor_size * sizeof(scalar_t) / 1.e6
                  << " MB for factors, "
                  << ldata[l].schur_size * sizeof(scalar_t) / 1.e6
                  << " MB for Schur complements" << std::endl;

      scalar_t* fmem = nullptr;
      cudaMallocManaged(&fmem, ldata[l].factor_size*sizeof(scalar_t));
      ldata[l].f[0]->factor_mem_ = uniq_scalar_t(fmem, cuda_deleter);
      auto wmem = work_mem[l % 2];

      // initialize pointers to data for frontal matrices
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
      wmem = static_cast<scalar_t*>(wmem) +
        max_streams * ldata[l].getrf_work_size;
      for (std::size_t n=0; n<nnodes; n++) {
        ldata[l].dev_piv[n] = static_cast<int*>(wmem);
        wmem = static_cast<int*>(wmem) + ldata[l].f[n]->dim_sep();
      }
      ldata[l].dev_getrf_err = static_cast<int*>(wmem);
      wmem = static_cast<int*>(wmem) + max_streams;

      // build front from sparse matrix and extend-add
#pragma omp parallel for
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
          (work_mem[l % 2], max_level_work_bytes, device_id, 0);
        cudaMemPrefetchAsync
          (ldata[l].f[0]->factor_mem_.get(),
           ldata[l].factor_size*sizeof(scalar_t), device_id, 0);
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
#pragma omp parallel for
        for (std::size_t n=0; n<nnodes; n++) {
          auto& f = *(ldata[l].f[n]);
          const auto dsep = f.dim_sep();
          const auto dupd = f.dim_upd();
          const auto size = dsep + dupd;
          if (size < cutoff_size) {
            if (dsep) {
              f.piv.resize(dsep);
              int flag;
              blas::getrf(dsep, dsep, f.F11_.data(), dsep, f.piv.data(), &flag);
              // TODO if (opts.replace_tiny_pivots()) { ...
              if (dupd) {
                blas::getrs
                  ('N', dsep, dupd, f.F11_.data(), dsep,
                   f.piv.data(), f.F12_.data(), dsep, &flag);
                blas::gemm
                  ('N', 'N', dupd, dupd, dsep, scalar_t(-1.), f.F21_.data(), dupd,
                   f.F12_.data(), dsep, scalar_t(1.), f.F22_.data(), dupd);
              }
            }
          }
        }
      }
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        auto stream = n % max_streams;
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        const auto size = dsep + dupd;
        if (size >= cutoff_size) {
          if (dsep) {
            cuda::cusolverDngetrf
              (solver_handle[stream], dsep, dsep, f.F11_.data(), dsep,
               ldata[l].dev_getrf_work + stream * ldata[l].getrf_work_size,
               ldata[l].dev_piv[n], ldata[l].dev_getrf_err + stream);
            // TODO if (opts.replace_tiny_pivots()) { ...
            if (dupd) {
              cuda::cusolverDngetrs
                (solver_handle[stream], CUBLAS_OP_N, dsep, dupd,
                 f.F11_.data(), dsep, ldata[l].dev_piv[n], f.F12_.data(), dsep,
                 ldata[l].dev_getrf_err + stream);
              cuda::cublasgemm
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
           ldata[l].factor_size*sizeof(scalar_t), cudaCpuDeviceId, 0);

      // copy pivot vectors back from the device
#pragma omp parallel for
      for (std::size_t n=0; n<nnodes; n++) {
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
      scalar_t* fmem_reg = new scalar_t[ldata[l].factor_size];
      fmem = ldata[l].f[0]->factor_mem_.get();
      std::copy(fmem, fmem+ldata[l].factor_size, fmem_reg);
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

  // using F22 for bupd makes this not const!!!!
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixCUBLAS<scalar_t,integer_t>::multifrontal_solve
  (DenseM_t& b) const {
    // TODO get the options in here
    const int cutoff_size = default_cuda_cutoff(); //opts.cuda_cutoff();
    const int max_streams = default_cuda_streams(); //opts.cuda_streams();
    int device_id;
    cudaGetDevice(&device_id);
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
    std::size_t max_level_work_bytes = 0, max_level_factor_size = 0;
    for (int l=lvls-1; l>=0; l--) {
      std::vector<const F_t*> fp;
      fp.reserve(2 << (l-1));
      this->get_level_fronts(fp, l);
      ldata[l].f.reserve(fp.size());
      for (auto f : fp)
        ldata[l].f.push_back(dynamic_cast<FC_t*>(const_cast<F_t*>(f)));
      auto nnodes = ldata[l].f.size();
      ldata[l].dev_piv.resize(nnodes);
      ldata[l].bloc.resize(nnodes);
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        const auto size = dsep + dupd;
        ldata[l].factor_size += dsep*dsep + 2*dsep*dupd;
        ldata[l].bloc_size += dsep*nrhs;
        ldata[l].bupd_size += dupd*nrhs;
        ldata[l].piv_size += dsep;
        if (size < cutoff_size)
          ldata[l].nnodes_small++;
      }
      ldata[l].total_work_bytes =
        sizeof(scalar_t) * (ldata[l].bloc_size + ldata[l].bupd_size) +
        sizeof(int) * (ldata[l].piv_size) +
        max_streams * sizeof(int);  // CUDA getrs error code
      max_level_work_bytes = std::max
        (max_level_work_bytes, ldata[l].total_work_bytes);
      max_level_factor_size = std::max
        (max_level_factor_size, ldata[l].factor_size);
    }
    // ensure alignment?
    max_level_work_bytes += max_level_work_bytes % 8;

    void* all_work_mem = nullptr;
    cudaMallocManaged
      (&all_work_mem, 2 * max_level_work_bytes
       + sizeof(scalar_t) * max_level_factor_size);
    void* work_mem[2] =
      {all_work_mem, (char*)all_work_mem + max_level_work_bytes};
    scalar_t* factor_mem =
      static_cast<scalar_t*>
      (static_cast<void*>((char*)all_work_mem + 2 * max_level_work_bytes));

    ////////////////////////////////////////////////////////////////
    //////////////     forward solve      //////////////////////////
    ////////////////////////////////////////////////////////////////
    for (int l=lvls-1; l>=0; l--) {
      auto nnodes = ldata[l].f.size();
      auto nnodes_small = ldata[l].nnodes_small;
      auto wmem = work_mem[l % 2];
      auto fmem = factor_mem;
      if (nnodes_small != nnodes)
        std::copy
          (ldata[l].f[0]->factor_mem_.get(),
           ldata[l].f[0]->factor_mem_.get()+ldata[l].factor_size, fmem);

      // initialize pointers to data for frontal matrices
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const int dsep = f.dim_sep();
        const int dupd = f.dim_upd();
        ldata[l].bloc[n] =
          DenseMW_t(dsep, nrhs, static_cast<scalar_t*>(wmem), dsep);
        wmem = static_cast<scalar_t*>(wmem) + dsep*nrhs;
        if (nnodes_small != nnodes) {
          f.F11_ = DenseMW_t(dsep, dsep, fmem, dsep); fmem += dsep*dsep;
          f.F12_ = DenseMW_t(dsep, dupd, fmem, dsep); fmem += dsep*dupd;
          f.F21_ = DenseMW_t(dupd, dsep, fmem, dupd); fmem += dupd*dsep;
        }
        if (dupd) {
          f.F22_ = DenseMW_t(dupd, nrhs, static_cast<scalar_t*>(wmem), dupd);
          wmem = static_cast<scalar_t*>(wmem) + dupd*nrhs;
        }
      }
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        ldata[l].dev_piv[n] = static_cast<int*>(wmem);
        wmem = static_cast<int*>(wmem) + f.dim_sep();
        std::copy(f.piv.begin(), f.piv.end(), ldata[l].dev_piv[n]);
      }
      ldata[l].dev_getrf_err = static_cast<int*>(wmem);
      wmem = static_cast<int*>(wmem) + max_streams;

      // extend from children fronts
#pragma omp parallel for
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
#pragma omp parallel for
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        copy(f.dim_sep(), nrhs, b, f.sep_begin_, 0, ldata[l].bloc[n], 0, 0);
      }

      if (nnodes_small) {
#pragma omp parallel for
        for (std::size_t n=0; n<nnodes; n++) {
          auto& f = *(ldata[l].f[n]);
          const auto dsep = f.dim_sep();
          const auto dupd = f.dim_upd();
          const auto size = dsep + dupd;
          if (size < cutoff_size) {
            // call the blas/lapack routines directly to avoid
            // overhead of tasking/checking whether in parallel region
            if (dsep) {
              int flag;
              blas::getrs
                ('N', dsep, nrhs, f.F11_.data(), dsep,
                 f.piv.data(), ldata[l].bloc[n].data(), dsep, &flag);
              if (dupd) {
                if (nrhs == 1)
                  blas::gemv
                    ('N', dupd, dsep, scalar_t(-1.), f.F21_.data(), dupd,
                     ldata[l].bloc[n].data(), 1, scalar_t(1.),
                     f.F22_.data(), 1);
                else
                  blas::gemm
                    ('N', 'N', dupd, nrhs, dsep, scalar_t(-1.),
                     f.F21_.data(), dupd, ldata[l].bloc[n].data(), dsep,
                     scalar_t(1.), f.F22_.data(), dupd);
              }
            }
          }
        }
      }
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        auto stream = n % max_streams;
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        const auto size = dsep + dupd;
        if (size >= cutoff_size) {
          if (dsep) {
            cuda::cusolverDngetrs
              (solver_handle[stream], CUBLAS_OP_N, dsep, nrhs,
               f.F11_.data(), dsep, ldata[l].dev_piv[n],
               ldata[l].bloc[n].data(), dsep,
               ldata[l].dev_getrf_err + stream);
            if (dupd) {
              if (nrhs == 1)
                cuda::cublasgemv
                  (blas_handle[stream], CUBLAS_OP_N,
                   dupd, dsep, scalar_t(-1.), f.F21_.data(), dupd,
                   ldata[l].bloc[n].data(), 1,
                   scalar_t(1.), f.F22_.data(), 1);
              else
                cuda::cublasgemm
                  (blas_handle[stream], CUBLAS_OP_N, CUBLAS_OP_N,
                   dupd, nrhs, dsep, scalar_t(-1.), f.F21_.data(), dupd,
                   ldata[l].bloc[n].data(), dsep,
                   scalar_t(1.), f.F22_.data(), dupd);
            }
          }
        }
      }
      cudaDeviceSynchronize();

      // copy right hand side back from (managed) device memory
#pragma omp parallel for
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        copy(f.dim_sep(), nrhs, ldata[l].bloc[n], 0, 0, b, f.sep_begin_, 0);
      }
    }

    ////////////////////////////////////////////////////////////////
    //////////////     backward solve     //////////////////////////
    ////////////////////////////////////////////////////////////////
    for (int l=0; l<lvls; l++) {
      auto nnodes = ldata[l].f.size();
      auto nnodes_small = ldata[l].nnodes_small;
      auto wmem = work_mem[l % 2];
      if (nnodes_small != nnodes)
        std::copy
          (ldata[l].f[0]->factor_mem_.get(),
           ldata[l].f[0]->factor_mem_.get()+ldata[l].factor_size, factor_mem);


      // initialize pointers to data for frontal matrices
      ldata[l].bloc.resize(nnodes);
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const int dsep = f.dim_sep();
        const int dupd = f.dim_upd();
        ldata[l].bloc[n] =
          DenseMW_t(dsep, nrhs, static_cast<scalar_t*>(wmem), dsep);
        wmem = static_cast<scalar_t*>(wmem) + dsep*nrhs;
        if (dupd) {
          f.F22_ = DenseMW_t(dupd, nrhs, static_cast<scalar_t*>(wmem), dupd);
          wmem = static_cast<scalar_t*>(wmem) + dupd*nrhs;
        }
      }

      // extract from parent nodes
      if (l > 0) {
#pragma omp parallel for
        for (std::size_t n=0; n<ldata[l-1].f.size(); n++) {
          auto& f = *(ldata[l-1].f[n]);
          if (f.lchild_)
            f.lchild_->extract_b
              (b, f.F22_, dynamic_cast<FC_t*>(f.lchild_.get())->F22_, &f);
          if (f.rchild_)
            f.rchild_->extract_b
              (b, f.F22_, dynamic_cast<FC_t*>(f.rchild_.get())->F22_, &f);
        }
      }

      // copy right hand side to (managed) device memory
#pragma omp parallel for
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        copy(f.dim_sep(), nrhs, b, f.sep_begin_, 0, ldata[l].bloc[n], 0, 0);
      }

      if (nnodes_small) {
#pragma omp parallel for
        for (std::size_t n=0; n<nnodes; n++) {
          auto& f = *(ldata[l].f[n]);
          const auto dsep = f.dim_sep();
          const auto dupd = f.dim_upd();
          const auto size = dsep + dupd;
          if (size < cutoff_size) {
            if (dsep && dupd) {
              if (nrhs == 1)
                blas::gemv
                  ('N', dsep, dupd, scalar_t(-1.), f.F12_.data(), dsep,
                   f.F22_.data(), 1, scalar_t(1.),
                   ldata[l].bloc[n].data(), 1);
              else
                blas::gemm
                  ('N', 'N', dsep, nrhs, dupd, scalar_t(-1.),
                   f.F12_.data(), dsep, f.F22_.data(), dupd,
                   scalar_t(1.), ldata[l].bloc[n].data(), dsep);
            }
          }
        }
      }
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        auto stream = n % max_streams;
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        const auto size = dsep + dupd;
        if (size >= cutoff_size) {
          if (dsep && dupd) {
            if (nrhs == 1)
              cuda::cublasgemv
                (blas_handle[stream], CUBLAS_OP_N,
                 dsep, dupd, scalar_t(-1.), f.F12_.data(), dsep,
                 f.F22_.data(), 1, scalar_t(1.),
                 ldata[l].bloc[n].data(), 1);
            else
              cuda::cublasgemm
                (blas_handle[stream], CUBLAS_OP_N, CUBLAS_OP_N,
                 dsep, nrhs, dupd, scalar_t(-1.), f.F12_.data(), dsep,
                 f.F22_.data(), dupd, scalar_t(1.),
                 ldata[l].bloc[n].data(), dsep);
          }
        }
      }
      cudaDeviceSynchronize();

      // copy right hand side back from (managed) device memory
#pragma omp parallel for
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        copy(f.dim_sep(), nrhs, ldata[l].bloc[n], 0, 0, b, f.sep_begin_, 0);
      }

      // revert pointers to the original (not managed)
      // memory
      if (nnodes_small != nnodes) {
        auto fmem = ldata[l].f[0]->factor_mem_.get();
        for (std::size_t n=0; n<nnodes; n++) {
          auto& f = *(ldata[l].f[n]);
          const int dsep = f.dim_sep();
          const int dupd = f.dim_upd();
          f.F11_ = DenseMW_t(dsep, dsep, fmem, dsep); fmem += dsep*dsep;
          f.F12_ = DenseMW_t(dsep, dupd, fmem, dsep); fmem += dsep*dupd;
          f.F21_ = DenseMW_t(dupd, dsep, fmem, dupd); fmem += dupd*dsep;
        }
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
