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
#include <cassert>
#include <memory>
#include <functional>
#include <algorithm>

#include "misc/TaskTimer.hpp"

#include "BLRMatrixMPI.hpp"
#include "BLRTileBLAS.hpp"
#include "BLRBatch.hpp"

#if defined(STRUMPACK_USE_CUDA)
#include "dense/CUDAWrapper.hpp"
#endif
#if defined(STRUMPACK_USE_HIP)
#include "dense/HIPWrapper.hpp"
#endif

#include "sparse/fronts/FrontalMatrixGPUKernels.hpp"

namespace strumpack {
  namespace BLR {

    template<typename scalar_t> void
    BLRMatrixMPI<scalar_t>::move_to_gpu(gpu::Stream& s, scalar_t* dptr,
                                        scalar_t* pinned) {
#pragma omp parallel
#pragma omp single nowait
      for (std::size_t j=0; j<colblockslocal(); j++) {
        auto pin = pinned;
        for (std::size_t i=0; i<rowblockslocal(); i++) {
#pragma omp task firstprivate(dptr)
          ltile(i, j).move_to_gpu(s, dptr, pin);
          auto nnz = ltile(i, j).nonzeros();
          dptr += nnz;
          pin += nnz;
        }
#pragma omp taskwait
      }
    }

    template<typename scalar_t> void
    BLRMatrixMPI<scalar_t>::move_to_cpu(gpu::Stream& s, scalar_t* pinned) {
#pragma omp parallel
#pragma omp single nowait
      for (std::size_t j=0; j<colblockslocal(); j++) {
        auto pin = pinned;
        for (std::size_t i=0; i<rowblockslocal(); i++) {
#pragma omp task firstprivate(pin)
          ltile(i, j).move_to_cpu(s, pin);
          pin += ltile(i, j).nonzeros();
        }
#pragma omp taskwait
      }
    }


    template<typename scalar_t>
    std::vector<std::unique_ptr<BLRTile<scalar_t>>>
    BLRMatrixMPI<scalar_t>::bcast_row_of_tiles_along_cols_gpu
    (std::size_t i, std::size_t j0, std::size_t j1,
     scalar_t* dptr, scalar_t* pinned, VectorPool<scalar_t>& workspace,
     const SPOptions<scalar_t>& spopts) const {
      if (!grid()) return {};
      int src = i % grid()->nprows();
      std::size_t msg_size = 0, nr_tiles = 0;
      std::vector<std::int64_t> ranks;
      if (grid()->is_local_row(i)) {
        for (std::size_t j=j0; j<j1; j++)
          if (grid()->is_local_col(j)) {
            auto& t = tile(i, j);
            msg_size += t.nonzeros();
            ranks.push_back(t.is_low_rank() ? t.rank() : -1);
            nr_tiles++;
          }
      } else {
        for (std::size_t j=j0; j<j1; j++)
          if (grid()->is_local_col(j))
            nr_tiles++;
        ranks.resize(nr_tiles);
      }
      std::vector<std::unique_ptr<BLRTile<scalar_t>>> Tij;
      if (ranks.empty()) return Tij;
      ranks.push_back(msg_size);
      grid()->col_comm().broadcast_from(ranks, src);
      msg_size = ranks.back();

      if (spopts.use_gpu_aware_mpi()) {
        // assert(pinned_size >= msg_size);
        // gpu::DeviceMemory<scalar_t> dpinptr(msg_size);
        auto dbytes = workspace.get_device_bytes(msg_size*sizeof(scalar_t));
        //scalar_t *ptr = dpinptr.template as<scalar_t>();
        scalar_t *ptr = dbytes.template as<scalar_t>();
        //auto ptr = pinned; // not working correctly
        if (grid()->is_local_row(i)) {
          for (std::size_t j=j0; j<j1; j++)
            if (grid()->is_local_col(j)) {
              auto& t = tile(i, j);
              auto m = t.rows(), n = t.cols();
              if (t.is_low_rank()) {
                auto r = t.rank();
                gpu_check(gpu::copy_device_to_device(ptr, t.U()));
                ptr += m*r;
                gpu_check(gpu::copy_device_to_device(ptr, t.V()));
                ptr += r*n;
              } else {
                gpu_check(gpu::copy_device_to_device(ptr, t.D()));
                ptr += m*n;
              }
            }
        }
        //ptr = dpinptr; //device mem
        ptr = dbytes.template as<scalar_t>();
        //ptr = pinned; // not working correctly
        grid()->col_comm().broadcast_from(ptr, msg_size, src);
        Tij.reserve(nr_tiles);
        auto m = tilerows(i);
        for (std::size_t j=j0; j<j1; j++)
          if (grid()->is_local_col(j)) {
            auto r = ranks[Tij.size()];
            auto n = tilecols(j);
            if (r != -1) {
              DenseMW_t dU(m, r, dptr, m);  dptr += m*r;
              DenseMW_t dV(r, n, dptr, r);  dptr += r*n;
              gpu_check(gpu::copy_device_to_device(dU, ptr));
              ptr += m*r;
              gpu_check(gpu::copy_device_to_device(dV, ptr));
              ptr += r*n;
              Tij.emplace_back(new LRTile<scalar_t>(dU, dV));
            } else {
              DenseMW_t dD(m, n, dptr, m);  dptr += m*n;
              gpu_check(gpu::copy_device_to_device(dD, ptr));
              ptr += m*n;
              Tij.emplace_back(new DenseTile<scalar_t>(dD));
            }
          }
        workspace.restore(dbytes);
      } else { //NOT gpu_aware_mpi
        auto ptr = pinned;
        if (grid()->is_local_row(i)) {
          for (std::size_t j=j0; j<j1; j++)
            if (grid()->is_local_col(j)) {
              auto& t = tile(i, j);
              auto m = t.rows(), n = t.cols();
              if (t.is_low_rank()) {
                auto r = t.rank();
                gpu_check(gpu::copy_device_to_host(ptr, t.U()));
                ptr += m*r;
                gpu_check(gpu::copy_device_to_host(ptr, t.V()));
                ptr += r*n;
              } else {
                gpu_check(gpu::copy_device_to_host(ptr, t.D()));
                ptr += m*n;
              }
            }
        }
        ptr = pinned;
        grid()->col_comm().broadcast_from(ptr, msg_size, src);
        Tij.reserve(nr_tiles);
        auto m = tilerows(i);
        for (std::size_t j=j0; j<j1; j++)
          if (grid()->is_local_col(j)) {
            auto r = ranks[Tij.size()];
            auto n = tilecols(j);
            if (r != -1) {
              DenseMW_t dU(m, r, dptr, m);  dptr += m*r;
              DenseMW_t dV(r, n, dptr, r);  dptr += r*n;
              gpu_check(gpu::copy_host_to_device(dU, ptr));
              ptr += m*r;
              gpu_check(gpu::copy_host_to_device(dV, ptr));
              ptr += r*n;
              Tij.emplace_back(new LRTile<scalar_t>(dU, dV));
            } else {
              DenseMW_t dD(m, n, dptr, m);  dptr += m*n;
              gpu_check(gpu::copy_host_to_device(dD, ptr));
              ptr += m*n;
              Tij.emplace_back(new DenseTile<scalar_t>(dD));
            }
          }
      }
      // stream.synchronize();
      return Tij;
    }

    template<typename scalar_t>
    std::vector<std::unique_ptr<BLRTile<scalar_t>>>
    BLRMatrixMPI<scalar_t>::bcast_col_of_tiles_along_rows_gpu
    (std::size_t i0, std::size_t i1, std::size_t j,
     scalar_t* dptr, scalar_t* pinned, const SPOptions<scalar_t>& spopts) const {
      if (!grid()) return {};
      int src = j % grid()->npcols();
      std::size_t msg_size = 0, nr_tiles = 0;
      std::vector<std::int64_t> ranks;
      if (grid()->is_local_col(j)) {
        for (std::size_t i=i0; i<i1; i++)
          if (grid()->is_local_row(i)) {
            auto& t = tile(i, j);
            msg_size += t.nonzeros();
            ranks.push_back(t.is_low_rank() ? t.rank() : -1);
            nr_tiles++;
          }
      } else {
        for (std::size_t i=i0; i<i1; i++)
          if (grid()->is_local_row(i))
            nr_tiles++;
        ranks.resize(nr_tiles);
      }
      std::vector<std::unique_ptr<BLRTile<scalar_t>>> Tij;
      if (ranks.empty()) return Tij;
      ranks.push_back(msg_size);
      grid()->row_comm().broadcast_from(ranks, src);
      msg_size = ranks.back();
      // assert(pinned.size() >= msg_size);
      if (spopts.use_gpu_aware_mpi()){
        gpu::DeviceMemory<scalar_t> dpinptr(msg_size);
        scalar_t *ptr = dpinptr.template as<scalar_t>();
        //auto ptr = pinned; // not working correctly
        if (grid()->is_local_col(j)) {
          for (std::size_t i=i0; i<i1; i++)
            if (grid()->is_local_row(i)) {
              auto& t = tile(i, j);
              auto m = t.rows(), n = t.cols();
              if (t.is_low_rank()) {
                auto r = t.rank();
                gpu_check(gpu::copy_device_to_device(ptr, t.U()));
                ptr += m*r;
                gpu_check(gpu::copy_device_to_device(ptr, t.V()));
                ptr += r*n;
              } else {
                gpu_check(gpu::copy_device_to_device(ptr, t.D()));
                ptr += m*n;
              }
            }
        }
        ptr = dpinptr; //device mem
        //ptr = pinned; // not working correctly
        grid()->row_comm().broadcast_from(ptr, msg_size, src);
        Tij.reserve(nr_tiles);
        auto n = tilecols(j); 
        for (std::size_t i=i0; i<i1; i++)
          if (grid()->is_local_row(i)) {
            auto r = ranks[Tij.size()];
            auto m = tilerows(i);
            if (r != -1) {
              DenseMW_t dU(m, r, dptr, m);  dptr += m*r;
              DenseMW_t dV(r, n, dptr, r);  dptr += r*n;
              gpu_check(gpu::copy_device_to_device(dU, ptr));
              ptr += m*r;
              gpu_check(gpu::copy_device_to_device(dV, ptr));
              ptr += r*n;
              Tij.emplace_back(new LRTile<scalar_t>(dU, dV));
            } else {
              DenseMW_t dD(m, n, dptr, m);  dptr += m*n;
              gpu_check(gpu::copy_device_to_device(dD, ptr));
              ptr += m*n;
              Tij.emplace_back(new DenseTile<scalar_t>(dD));
            }
          }
      } else {  //NOT gpu_aware_mpi
        auto ptr = pinned;
        if (grid()->is_local_col(j)) {
          for (std::size_t i=i0; i<i1; i++)
            if (grid()->is_local_row(i)) {
              auto& t = tile(i, j);
              auto m = t.rows(), n = t.cols();
              if (t.is_low_rank()) {
                auto r = t.rank();
                gpu_check(gpu::copy_device_to_host(ptr, t.U()));
                ptr += m*r;
                gpu_check(gpu::copy_device_to_host(ptr, t.V()));
                ptr += r*n;
              } else {
                gpu_check(gpu::copy_device_to_host(ptr, t.D()));
                ptr += m*n;
              }
            }
        }
        ptr = pinned;
        grid()->row_comm().broadcast_from(ptr, msg_size, src);
        Tij.reserve(nr_tiles);
        auto n = tilecols(j);
        for (std::size_t i=i0; i<i1; i++)
          if (grid()->is_local_row(i)) {
            auto r = ranks[Tij.size()];
            auto m = tilerows(i);
            if (r != -1) {
              DenseMW_t dU(m, r, dptr, m);  dptr += m*r;
              DenseMW_t dV(r, n, dptr, r);  dptr += r*n;
              gpu_check(gpu::copy_host_to_device(dU, ptr));
              ptr += m*r;
              gpu_check(gpu::copy_host_to_device(dV, ptr));
              ptr += r*n;
              Tij.emplace_back(new LRTile<scalar_t>(dU, dV));
            } else {
              DenseMW_t dD(m, n, dptr, m);  dptr += m*n;
              gpu_check(gpu::copy_host_to_device(dD, ptr));
              ptr += m*n;
              Tij.emplace_back(new DenseTile<scalar_t>(dD));
            }
          }
      }
      return Tij;
    }


    /*
     * Input matrices are on CPU.
     */
    template<typename scalar_t> std::vector<int>
    BLRMatrixMPI<scalar_t>::partial_factor_gpu
    (BLRMPI_t& A11, BLRMPI_t& A12, BLRMPI_t& A21, BLRMPI_t& A22,
     const adm_t& adm, const Opts_t& opts, const SPOptions<scalar_t>& spopts) {
      auto g = A11.grid();
      std::vector<int> piv, piv_tile;
      if (!g->active()) return piv;

      gpu::Stream copy_stream, comp_stream;
      gpu::BLASHandle handle(comp_stream);
      gpu::SOLVERHandle solve_handle(comp_stream);

      auto rb = A11.rowblocks();
      auto rb2 = A22.rowblocks();
      int max_batchcount =
        A11.blocks_.size() + A12.blocks_.size() +
        A21.blocks_.size() + A22.blocks_.size();
      auto max_m1 = A11.maxtilerows();

#if defined(STRUMPACK_USE_KBLAS)
      VBatchedARA<scalar_t>::kblas_wsquery(handle, max_batchcount);
#else
      // TODO no KBLAS
#endif

      VectorPool<scalar_t> workspace;

      // used to bcast a row/col
      std::size_t pinned_size =
        std::max(max_m1, A22.maxtilerows()) *
        std::max(std::max(A11.lcols(), A11.lrows()),
                 std::max(A22.lcols(), A22.lrows()));
      auto pinned = workspace.get_pinned(pinned_size);

      auto getrf_work_size =
        gpu::getrf_buffersize<scalar_t>(solve_handle, max_m1);
      auto d_batch_meta = VBatchedGEMM<scalar_t>::dwork_bytes(max_batchcount);
      std::size_t d_scalars = getrf_work_size +
        max_m1 * (A11.lcols() + A22.lcols() + A11.lrows() + A22.lrows()) +
        A11.lrows() * A11.lcols() + A12.lrows()*A12.lcols() +
        A21.lrows() * A12.lcols() + A22.lrows()*A22.lcols();
      gpu::DeviceMemory<scalar_t> d_scalar_mem(d_scalars);
      auto dwork = d_scalar_mem.template as<scalar_t>();
      auto drow1 = dwork + getrf_work_size;
      auto drow2 = drow1 + max_m1*A11.lcols();
      auto dcol1 = drow2 + max_m1*A22.lcols();
      auto dcol2 = dcol1 + max_m1*A11.lrows();
      auto dA11  = dcol2 + max_m1*A22.lrows();
      auto dA12  = dA11 + A11.lrows() * A11.lcols();
      auto dA21  = dA12 + A12.lrows() * A12.lcols();
      auto dA22  = dA21 + A21.lrows() * A21.lcols();

      if (spopts.use_gpu_aware_mpi()){ 
        //--- this part of cuda aware, not working correctly
        std::size_t dpinned_size =
          std::max(max_m1, A22.maxtilerows()) *
          std::max(std::max(A11.lcols(), A11.lrows()),
                  std::max(A22.lcols(), A22.lrows()));
        gpu::DeviceMemory<scalar_t> d_pinned(dpinned_size);
        //---
      //}
        gpu::DeviceMemory<int> dpiv(max_m1+1);
      //if (spopts.use_gpu_aware_mpi()){
        auto dpiv_ptr = dpiv.template as<int>();
        auto dinfo = dpiv_ptr + max_m1;
      //} else 
        //auto dinfo = dpiv + max_m1;
        gpu::DeviceMemory<char> d_batch_mem(3*d_batch_meta);
        gpu::DeviceMemory<char> d_batch_matrix_mem;

        // TODO do this column wise to overlap
        A11.move_to_gpu(copy_stream, dA11, pinned);
        A12.move_to_gpu(copy_stream, dA12, pinned);
        A21.move_to_gpu(copy_stream, dA21, pinned);
        A22.move_to_gpu(copy_stream, dA22, pinned);

      //if (spopts.use_gpu_aware_mpi()){
        DenseTile<scalar_t> Tii;
        for (std::size_t i=0; i<rb; i++) {
          auto mi = A11.tilerows(i);
          DenseMW_t test(mi, mi, dcol1, mi);
          Tii = DenseTile<scalar_t>(test);
          if (g->is_local_row(i)) {
            piv_tile.resize(mi);
            if (g->is_local_col(i)) {
              gpu::getrf<scalar_t>
                (solve_handle, A11.tile(i, i).D(), dwork, dpiv, dinfo);
              comp_stream.synchronize();
              gpu_check(gpu::copy_device_to_device
                        (Tii.D(), A11.tile(i, i).D()));
            }
            int src = i % g->npcols();
            g->row_comm().broadcast_from(dpiv_ptr, mi, src);
            g->row_comm().broadcast_from(Tii.D().data(), mi*mi, src);
            gpu_check(gpu::copy_device_to_host<int>
                        (piv_tile.data(), dpiv, mi));
            int r0 = A11.tileroff(i);
            std::transform
              (piv_tile.begin(), piv_tile.end(), std::back_inserter(piv),
              [r0](int p) -> int { return p + r0; });
          }
          if (g->is_local_col(i)) {
            g->col_comm().broadcast_from(Tii.D().data(), mi*mi, i % g->nprows());
          }
          copy_stream.synchronize();

#if defined(STRUMPACK_USE_KBLAS)
          VBatchedARA<scalar_t> ara;
          if (g->is_local_row(i)) {
            for (std::size_t j=i+1; j<rb; j++)
              if (g->is_local_col(j) && adm(i, j))
                ara.add(A11.block(i, j));
            for (std::size_t j=0; j<rb2; j++)
              if (g->is_local_col(j))
                ara.add(A12.block(i, j));
          }
          if (g->is_local_col(i)) {
            for (std::size_t j=i+1; j<rb; j++)
              if (g->is_local_row(j) && adm(j, i))
                ara.add(A11.block(j, i));
            for (std::size_t j=0; j<rb2; j++)
              if (g->is_local_row(j))
                ara.add(A21.block(j, i));
          }
          ara.run(handle, workspace, opts.rel_tol());
#else
          std::cout << "TODO BLR compression requires KBLAS for now" << std::endl;
#endif
          VBatchedTRSM<scalar_t> trsm_left, trsm_right;
          if (g->is_local_row(i)) {
            for (std::size_t j=i+1; j<rb; j++)
              if (g->is_local_col(j)) {
                A11.tile(i, j).laswp(handle, dpiv, true);
                trsm_left.add(Tii.D(), A11.tile(i, j).U());
              }
            for (std::size_t j=0; j<rb2; j++)
              if (g->is_local_col(j)) {
                A12.tile(i, j).laswp(handle, dpiv, true);
                trsm_left.add(Tii.D(), A12.tile(i, j).U());
              }
          }
          if (g->is_local_col(i)) {
            for (std::size_t j=i+1; j<rb; j++)
              if (g->is_local_row(j))
                trsm_right.add(Tii.D(), A11.tile(j, i).V());
            for (std::size_t j=0; j<rb2; j++)
              if (g->is_local_row(j))
                trsm_right.add(Tii.D(), A21.tile(j, i).V());
          }
          trsm_left.run(handle, workspace, true);
          trsm_right.run(handle, workspace, false);
          comp_stream.synchronize();
          // Schur complement update
          auto Tij = A11.bcast_row_of_tiles_along_cols_gpu
            (i, i+1, rb, drow1, d_pinned, workspace, spopts);
          auto Tij2 = A12.bcast_row_of_tiles_along_cols_gpu
            (i, 0, rb2, drow2, d_pinned, workspace, spopts);
          auto Tki = A11.bcast_col_of_tiles_along_rows_gpu
            (i+1, rb, i, dcol1, d_pinned, spopts);
          auto Tk2i = A21.bcast_col_of_tiles_along_rows_gpu
            (0, rb2, i, dcol2, d_pinned, spopts);

          int batchcount = 0;
          std::size_t sVU = 0, sUVU = 0;
          for (std::size_t k=i+1, lk=0; k<rb; k++) {
            if (!g->is_local_row(k)) continue;
            for (std::size_t j=i+1, lj=0; j<rb; j++)
              if (g->is_local_col(j)) {
                batchcount++;
                multiply_inc_work_size(*(Tki[lk]), *(Tij[lj++]), sVU, sUVU);
              }
            for (std::size_t j=0, lj=0; j<rb2; j++)
              if (g->is_local_col(j)) {
                batchcount++;
                multiply_inc_work_size(*(Tki[lk]), *(Tij2[lj++]), sVU, sUVU);
              }
            lk++;
          }
          for (std::size_t k=0, lk=0; k<rb2; k++) {
            if (!g->is_local_row(k)) continue;
            for (std::size_t j=i+1, lj=0; j<rb; j++)
              if (g->is_local_col(j)) {
                multiply_inc_work_size(*(Tk2i[lk]), *(Tij[lj++]), sVU, sUVU);
                batchcount++;
              }
            for (std::size_t j=0, lj=0; j<rb2; j++)
              if (g->is_local_col(j)) {
                batchcount++;
                multiply_inc_work_size(*(Tk2i[lk]), *(Tij2[lj++]), sVU, sUVU);
              }
            lk++;
          }

          assert(batchcount <= max_batchcount);
          if ((sVU+sUVU)*sizeof(scalar_t) > d_batch_matrix_mem.size()) {
            workspace.restore(d_batch_matrix_mem);
            d_batch_matrix_mem = workspace.get_device_bytes
              ((sVU+sUVU)*sizeof(scalar_t));
          }
          auto dVU = d_batch_matrix_mem.template as<scalar_t>();
          auto dUVU = dVU + sVU;
          VBatchedGEMM<scalar_t> b1(batchcount, d_batch_mem),
            b2(batchcount, d_batch_mem+gpu::round_up(d_batch_meta)),
            b3(batchcount, d_batch_mem+2*gpu::round_up(d_batch_meta));

          for (std::size_t k=i+1, lk=0; k<rb; k++) {
            if (!g->is_local_row(k)) continue;
            for (std::size_t j=i+1, lj=0; j<rb; j++)
              if (g->is_local_col(j))
                add_tile_mult
                  (*(Tki[lk]), *(Tij[lj++]), A11.tile_dense(k, j).D(),
                  b1, b2, b3, dVU, dUVU);
            for (std::size_t j=0, lj=0; j<rb2; j++)
              if (g->is_local_col(j))
                add_tile_mult
                  (*(Tki[lk]), *(Tij2[lj++]), A12.tile_dense(k, j).D(),
                  b1, b2, b3, dVU, dUVU);
            lk++;
          }
          for (std::size_t k=0, lk=0; k<rb2; k++) {
            if (!g->is_local_row(k)) continue;
            for (std::size_t j=i+1, lj=0; j<rb; j++)
              if (g->is_local_col(j))
                add_tile_mult
                  (*(Tk2i[lk]), *(Tij[lj++]), A21.tile_dense(k, j).D(),
                  b1, b2, b3, dVU, dUVU);
            for (std::size_t j=0, lj=0; j<rb2; j++)
              if (g->is_local_col(j))
                add_tile_mult
                  (*(Tk2i[lk]), *(Tij2[lj++]), A22.tile_dense(k, j).D(),
                  b1, b2, b3, dVU, dUVU);
            lk++;
          }
          b1.run(scalar_t(1.), scalar_t(0.), comp_stream, handle);
          b2.run(scalar_t(1.), scalar_t(0.), comp_stream, handle);
          b3.run(scalar_t(-1.), scalar_t(1.), comp_stream, handle);
          comp_stream.synchronize();
        }
      } else { //NOT gpu_aware_mpi
        gpu::DeviceMemory<int> dpiv(max_m1+1);
        auto dinfo = dpiv + max_m1;
        gpu::DeviceMemory<char> d_batch_mem(3*d_batch_meta);
        gpu::DeviceMemory<char> d_batch_matrix_mem;

        // TODO do this column wise to overlap
        A11.move_to_gpu(copy_stream, dA11, pinned);
        A12.move_to_gpu(copy_stream, dA12, pinned);
        A21.move_to_gpu(copy_stream, dA21, pinned);
        A22.move_to_gpu(copy_stream, dA22, pinned);

        DenseTile<scalar_t> Tii;
        for (std::size_t i=0; i<rb; i++) {
          auto mi = A11.tilerows(i);
          Tii = DenseTile<scalar_t>(DenseMW_t(mi, mi, pinned, mi));
          if (g->is_local_row(i)) {
            piv_tile.resize(mi);
            if (g->is_local_col(i)) {
              gpu::getrf<scalar_t>
                (solve_handle, A11.tile(i, i).D(), dwork, dpiv, dinfo);
              comp_stream.synchronize();
              gpu_check(gpu::copy_device_to_host
                          (Tii.D(), A11.tile(i, i).D()));
              gpu_check(gpu::copy_device_to_host<int>
                          (piv_tile.data(), dpiv, mi));
            }
            int src = i % g->npcols();
            g->row_comm().broadcast_from(piv_tile, src);
            g->row_comm().broadcast_from(Tii.D().data(), mi*mi, src);
            if (!g->is_local_col(i)) {
              gpu_check(gpu::copy_host_to_device<int>
                        (dpiv, piv_tile.data(), mi));
              Tii.move_to_gpu(copy_stream, dcol1);
            }
            int r0 = A11.tileroff(i);
            std::transform
              (piv_tile.begin(), piv_tile.end(), std::back_inserter(piv),
              [r0](int p) -> int { return p + r0; });
          }
          if (g->is_local_col(i)) {
            g->col_comm().broadcast_from(Tii.D().data(), mi*mi, i % g->nprows());
            Tii.move_to_gpu(copy_stream, drow1);
          }
          copy_stream.synchronize();

#if defined(STRUMPACK_USE_KBLAS)
          VBatchedARA<scalar_t> ara;
          if (g->is_local_row(i)) {
            for (std::size_t j=i+1; j<rb; j++)
              if (g->is_local_col(j) && adm(i, j))
                ara.add(A11.block(i, j));
            for (std::size_t j=0; j<rb2; j++)
              if (g->is_local_col(j))
                ara.add(A12.block(i, j));
          }
          if (g->is_local_col(i)) {
            for (std::size_t j=i+1; j<rb; j++)
              if (g->is_local_row(j) && adm(j, i))
                ara.add(A11.block(j, i));
            for (std::size_t j=0; j<rb2; j++)
              if (g->is_local_row(j))
                ara.add(A21.block(j, i));
          }
          ara.run(handle, workspace, opts.rel_tol());
#else
          std::cout << "TODO BLR compression requires KBLAS for now" << std::endl;
#endif
          VBatchedTRSM<scalar_t> trsm_left, trsm_right;
          if (g->is_local_row(i)) {
            for (std::size_t j=i+1; j<rb; j++)
              if (g->is_local_col(j)) {
                A11.tile(i, j).laswp(handle, dpiv, true);
                trsm_left.add(Tii.D(), A11.tile(i, j).U());
              }
            for (std::size_t j=0; j<rb2; j++)
              if (g->is_local_col(j)) {
                A12.tile(i, j).laswp(handle, dpiv, true);
                trsm_left.add(Tii.D(), A12.tile(i, j).U());
              }
          }
          if (g->is_local_col(i)) {
            for (std::size_t j=i+1; j<rb; j++)
              if (g->is_local_row(j))
                trsm_right.add(Tii.D(), A11.tile(j, i).V());
            for (std::size_t j=0; j<rb2; j++)
              if (g->is_local_row(j))
                trsm_right.add(Tii.D(), A21.tile(j, i).V());
          }
          trsm_left.run(handle, workspace, true);
          trsm_right.run(handle, workspace, false);
          comp_stream.synchronize();

          // Schur complement update
          auto Tij = A11.bcast_row_of_tiles_along_cols_gpu
            (i, i+1, rb, drow1, pinned, workspace, spopts);
          auto Tij2 = A12.bcast_row_of_tiles_along_cols_gpu
            (i, 0, rb2, drow2, pinned, workspace, spopts);
          auto Tki = A11.bcast_col_of_tiles_along_rows_gpu
            (i+1, rb, i, dcol1, pinned, spopts);
          auto Tk2i = A21.bcast_col_of_tiles_along_rows_gpu
            (0, rb2, i, dcol2, pinned, spopts);

          int batchcount = 0;
          std::size_t sVU = 0, sUVU = 0;
          for (std::size_t k=i+1, lk=0; k<rb; k++) {
            if (!g->is_local_row(k)) continue;
            for (std::size_t j=i+1, lj=0; j<rb; j++)
              if (g->is_local_col(j)) {
                batchcount++;
                multiply_inc_work_size(*(Tki[lk]), *(Tij[lj++]), sVU, sUVU);
              }
            for (std::size_t j=0, lj=0; j<rb2; j++)
              if (g->is_local_col(j)) {
                batchcount++;
                multiply_inc_work_size(*(Tki[lk]), *(Tij2[lj++]), sVU, sUVU);
              }
            lk++;
          }
          for (std::size_t k=0, lk=0; k<rb2; k++) {
            if (!g->is_local_row(k)) continue;
            for (std::size_t j=i+1, lj=0; j<rb; j++)
              if (g->is_local_col(j)) {
                multiply_inc_work_size(*(Tk2i[lk]), *(Tij[lj++]), sVU, sUVU);
                batchcount++;
              }
            for (std::size_t j=0, lj=0; j<rb2; j++)
              if (g->is_local_col(j)) {
                batchcount++;
                multiply_inc_work_size(*(Tk2i[lk]), *(Tij2[lj++]), sVU, sUVU);
              }
            lk++;
          }

          assert(batchcount <= max_batchcount);
          if ((sVU+sUVU)*sizeof(scalar_t) > d_batch_matrix_mem.size()) {
            workspace.restore(d_batch_matrix_mem);
            d_batch_matrix_mem = workspace.get_device_bytes
              ((sVU+sUVU)*sizeof(scalar_t));
          }
          auto dVU = d_batch_matrix_mem.template as<scalar_t>();
          auto dUVU = dVU + sVU;
          VBatchedGEMM<scalar_t> b1(batchcount, d_batch_mem),
            b2(batchcount, d_batch_mem+gpu::round_up(d_batch_meta)),
            b3(batchcount, d_batch_mem+2*gpu::round_up(d_batch_meta));

          for (std::size_t k=i+1, lk=0; k<rb; k++) {
            if (!g->is_local_row(k)) continue;
            for (std::size_t j=i+1, lj=0; j<rb; j++)
              if (g->is_local_col(j))
                add_tile_mult
                  (*(Tki[lk]), *(Tij[lj++]), A11.tile_dense(k, j).D(),
                  b1, b2, b3, dVU, dUVU);
            for (std::size_t j=0, lj=0; j<rb2; j++)
              if (g->is_local_col(j))
                add_tile_mult
                  (*(Tki[lk]), *(Tij2[lj++]), A12.tile_dense(k, j).D(),
                  b1, b2, b3, dVU, dUVU);
            lk++;
          }
          for (std::size_t k=0, lk=0; k<rb2; k++) {
            if (!g->is_local_row(k)) continue;
            for (std::size_t j=i+1, lj=0; j<rb; j++)
              if (g->is_local_col(j))
                add_tile_mult
                  (*(Tk2i[lk]), *(Tij[lj++]), A21.tile_dense(k, j).D(),
                  b1, b2, b3, dVU, dUVU);
            for (std::size_t j=0, lj=0; j<rb2; j++)
              if (g->is_local_col(j))
                add_tile_mult
                  (*(Tk2i[lk]), *(Tij2[lj++]), A22.tile_dense(k, j).D(),
                  b1, b2, b3, dVU, dUVU);
            lk++;
          }
          b1.run(scalar_t(1.), scalar_t(0.), comp_stream, handle);
          b2.run(scalar_t(1.), scalar_t(0.), comp_stream, handle);
          b3.run(scalar_t(-1.), scalar_t(1.), comp_stream, handle);
          comp_stream.synchronize();
        }
      }
      A11.move_to_cpu(copy_stream, pinned);
      A12.move_to_cpu(copy_stream, pinned);
      A21.move_to_cpu(copy_stream, pinned);
      A22.move_to_cpu(copy_stream, pinned);
      return piv;
    }

    // explicit template instantiations
    template class BLRMatrixMPI<float>;
    template class BLRMatrixMPI<double>;
    template class BLRMatrixMPI<std::complex<float>>;
    template class BLRMatrixMPI<std::complex<double>>;

  } // end namespace BLR
} // end namespace strumpack
