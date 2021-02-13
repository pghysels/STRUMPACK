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

#include "BLRMatrixMPI.hpp"
#include "BLRTileBLAS.hpp"
#include "misc/Tools.hpp"
#include "misc/TaskTimer.hpp"

namespace strumpack {
  namespace BLR {

    ProcessorGrid2D::ProcessorGrid2D(const MPIComm& comm)
      : ProcessorGrid2D(comm, comm.size()) {}

    ProcessorGrid2D::ProcessorGrid2D(const MPIComm& comm, int P)
      : comm_(comm) {
      npcols_ = std::floor(std::sqrt((float)P));
      nprows_ = P / npcols_;
      if (comm_.is_null()) {
        active_ = false;
        return;
      }
      auto rank = comm_.rank();
      active_ = rank < nprows_ * npcols_;
      if (active_) {
        prow_ = rank % nprows_;
        pcol_ = rank / nprows_;
      }
      for (int i=0; i<nprows_; i++)
        if (i == prow_) rowcomm_ = comm_.sub(i, npcols_, nprows_);
        else comm_.sub(i, npcols_, nprows_);
      for (int i=0; i<npcols_; i++)
        if (i == pcol_) colcomm_ = comm_.sub(i*nprows_, nprows_, 1);
        else comm_.sub(i*nprows_, nprows_, 1);
    }

    template<typename scalar_t> BLRMatrixMPI<scalar_t>::BLRMatrixMPI() {}

    template<typename scalar_t> void
    BLRMatrixMPI<scalar_t>::fill(scalar_t v) {
      for (std::size_t i=0; i<brows_; i++)
        for (std::size_t j=0; j<bcols_; j++)
          if (grid_->is_local(i, j)) {
            std::unique_ptr<DenseTile<scalar_t>> t
              (new DenseTile<scalar_t>(tilerows(i), tilecols(j)));
            t->D().fill(v);
            block(i, j) = std::move(t);
          }
    }

    template<typename scalar_t> std::size_t
    BLRMatrixMPI<scalar_t>::memory() const {
      std::size_t mem = 0;
      for (auto& b : blocks_) mem += b->memory();
      return mem;
    }

    template<typename scalar_t> std::size_t
    BLRMatrixMPI<scalar_t>::nonzeros() const {
      std::size_t nnz = 0;
      for (auto& b : blocks_) nnz += b->nonzeros();
      return nnz;
    }

    template<typename scalar_t> std::size_t
    BLRMatrixMPI<scalar_t>::rank() const {
      std::size_t mrank = 0;
      for (auto& b : blocks_) mrank = std::max(mrank, b->maximum_rank());
      return mrank;
    }

    template<typename scalar_t> std::size_t
    BLRMatrixMPI<scalar_t>::total_memory() const {
        return Comm().all_reduce(memory(), MPI_SUM);
    }
    template<typename scalar_t> std::size_t
    BLRMatrixMPI<scalar_t>::total_nonzeros() const {
      return Comm().all_reduce(nonzeros(), MPI_SUM);
    }
    template<typename scalar_t> std::size_t
    BLRMatrixMPI<scalar_t>::max_rank() const {
      return Comm().all_reduce(this->rank(), MPI_MAX);
    }

    template<typename scalar_t> void
    BLRMatrixMPI<scalar_t>::print(const std::string& name) {
      std::cout << "BLR(" << name << ")="
                << rows() << "x" << cols() << ", "
                << rowblocks() << "x" << colblocks() << ", "
                << (float(nonzeros()) / (rows()*cols()) * 100.) << "%"
                << " [" << std::endl;
      for (std::size_t i=0; i<brows_; i++) {
        for (std::size_t j=0; j<bcols_; j++) {
          auto& tij = tile(i, j);
          if (tij.is_low_rank())
            std::cout << "LR:" << tij.rows() << "x"
                      << tij.cols() << "/" << tij.rank() << " ";
          else std::cout << "D:" << tij.rows() << "x" << tij.cols() << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "];" << std::endl;
    }

    template<typename scalar_t> int
    BLRMatrixMPI<scalar_t>::rg2p(std::size_t i) const {
      return grid_->rg2p(rg2t(i));
    }
    template<typename scalar_t> int
    BLRMatrixMPI<scalar_t>::cg2p(std::size_t j) const {
      return grid_->cg2p(cg2t(j));
    }

    template<typename scalar_t> std::size_t
    BLRMatrixMPI<scalar_t>::rg2t(std::size_t i) const {
      return std::distance
        (roff_.begin(), std::upper_bound(roff_.begin(), roff_.end(), i)) - 1;
    }
    template<typename scalar_t> std::size_t
    BLRMatrixMPI<scalar_t>::cg2t(std::size_t j) const {
      return std::distance
        (coff_.begin(), std::upper_bound(coff_.begin(), coff_.end(), j)) - 1;
    }

    template<typename scalar_t> std::size_t
    BLRMatrixMPI<scalar_t>::rl2g(std::size_t i) const {
      assert(i < rl2g_.size());
      assert(rl2g_.size() == std::size_t(lrows()));
      return rl2g_[i];
    }
    template<typename scalar_t> std::size_t
    BLRMatrixMPI<scalar_t>::cl2g(std::size_t j) const {
      assert(j < cl2g_.size());
      assert(cl2g_.size() == std::size_t(lcols()));
      return cl2g_[j];
    }

    template<typename scalar_t> const scalar_t&
    BLRMatrixMPI<scalar_t>::operator()(std::size_t i, std::size_t j) const {
      return ltile_dense(rl2t_[i], cl2t_[j]).D()(rl2l_[i], cl2l_[j]);
    }
    template<typename scalar_t> scalar_t&
    BLRMatrixMPI<scalar_t>::operator()(std::size_t i, std::size_t j) {
      return ltile_dense(rl2t_[i], cl2t_[j]).D()(rl2l_[i], cl2l_[j]);
    }

    template<typename scalar_t> const scalar_t&
    BLRMatrixMPI<scalar_t>::global(std::size_t i, std::size_t j) const {
      std::size_t rt = rg2t(i), ct = cg2t(j);
      return tile_dense(rt, ct).D()(i - roff_[rt], j - coff_[ct]);
    }
    template<typename scalar_t> scalar_t&
    BLRMatrixMPI<scalar_t>::global(std::size_t i, std::size_t j) {
      std::size_t rt = rg2t(i), ct = cg2t(j);
      return tile_dense(rt, ct).D()(i - roff_[rt], j - coff_[ct]);
    }

    template<typename scalar_t> BLRMatrixMPI<scalar_t>::BLRMatrixMPI
    (const ProcessorGrid2D& grid, const vec_t& Rt, const vec_t& Ct)
      : grid_(&grid) {
      brows_ = Rt.size();
      bcols_ = Ct.size();
      roff_.resize(brows_+1);
      coff_.resize(bcols_+1);
      std::partial_sum(Rt.begin(), Rt.end(), roff_.begin()+1);
      std::partial_sum(Ct.begin(), Ct.end(), coff_.begin()+1);
      rows_ = roff_.back();
      cols_ = coff_.back();
      if (grid_->active()) {
        lbrows_ = brows_ / grid_->nprows() +
          (grid_->prow() < int(brows_ % grid_->nprows()));
        lbcols_ = bcols_ / grid_->npcols() +
          (grid_->pcol() < int(bcols_ % grid_->npcols()));
        blocks_.resize(lbrows_ * lbcols_);
      }
      lrows_ = lcols_ = 0;
      for (std::size_t b=grid_->prow(); b<brows_; b+=grid_->nprows())
        lrows_ += roff_[b+1] - roff_[b];
      for (std::size_t b=grid_->pcol(); b<bcols_; b+=grid_->npcols())
        lcols_ += coff_[b+1] - coff_[b];
      rl2t_.resize(lrows_);
      cl2t_.resize(lcols_);
      rl2l_.resize(lrows_);
      cl2l_.resize(lcols_);
      rl2g_.resize(lrows_);
      cl2g_.resize(lcols_);
      for (std::size_t b=grid_->prow(), l=0, lb=0;
           b<brows_; b+=grid_->nprows()) {
        for (std::size_t i=0; i<roff_[b+1]-roff_[b]; i++) {
          rl2t_[l] = lb;
          rl2l_[l] = i;
          rl2g_[l] = i + roff_[b];
          l++;
        }
        lb++;
      }
      for (std::size_t b=grid_->pcol(), l=0, lb=0;
           b<bcols_; b+=grid_->npcols()) {
        for (std::size_t i=0; i<coff_[b+1]-coff_[b]; i++) {
          cl2t_[l] = lb;
          cl2l_[l] = i;
          cl2g_[l] = i + coff_[b];
          l++;
        }
        lb++;
      }
    }

    template<typename scalar_t> DenseTile<scalar_t>
    BLRMatrixMPI<scalar_t>::bcast_dense_tile_along_row
    (std::size_t i, std::size_t j) const {
      DenseTile<scalar_t> t(tilerows(i), tilecols(j));
      int src = j % grid()->npcols();
      auto& c = grid()->row_comm();
      if (c.rank() == src) t = tile_dense(i, j);
      c.broadcast_from(t.D().data(), t.rows()*t.cols(), src);
      return t;
    }
    template<typename scalar_t> DenseTile<scalar_t>
    BLRMatrixMPI<scalar_t>::bcast_dense_tile_along_col
    (std::size_t i, std::size_t j) const {
      DenseTile<scalar_t> t(tilerows(i), tilecols(j));
      int src = i % grid()->nprows();
      auto& c = grid()->col_comm();
      if (c.rank() == src) t = tile_dense(i, j);
      c.broadcast_from(t.D().data(), t.rows()*t.cols(), src);
      return t;
    }

    template<typename scalar_t>
    std::vector<std::unique_ptr<BLRTile<scalar_t>>>
    BLRMatrixMPI<scalar_t>::bcast_row_of_tiles_along_cols
    (std::size_t i, std::size_t j0, std::size_t j1) const {
      int src = i % grid()->nprows();
      std::size_t msg_size = 0, nr_tiles = 0;
      std::vector<std::int64_t> ranks;
      if (grid()->is_local_row(i)) {
        for (std::size_t j=j0; j<j1; j++)
          if (grid()->is_local_col(j)) {
            msg_size += tile(i, j).nonzeros();
            ranks.push_back(tile(i, j).is_low_rank() ?
                            tile(i, j).rank() : -1);
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
      std::vector<scalar_t> buf(msg_size);
      auto ptr = buf.data();
      if (grid()->is_local_row(i)) {
        for (std::size_t j=j0; j<j1; j++)
          if (grid()->is_local_col(j)) {
            auto& t = tile(i, j);
            if (t.is_low_rank()) {
              std::copy(t.U().data(), t.U().end(), ptr);
              ptr += t.U().rows()*t.U().cols();
              std::copy(t.V().data(), t.V().end(), ptr);
              ptr += t.V().rows()*t.V().cols();
            } else {
              std::copy(t.D().data(), t.D().end(), ptr);
              ptr += t.D().rows()*t.D().cols();
            }
          }
      }
      grid()->col_comm().broadcast_from(buf, src);
      Tij.reserve(nr_tiles);
      ptr = buf.data();
      auto m = tilerows(i);
      for (std::size_t j=j0; j<j1; j++)
        if (grid()->is_local_col(j)) {
          auto r = ranks[Tij.size()];
          auto n = tilecols(j);
          if (r != -1) {
            auto t = new LRTile<scalar_t>(m, n, r);
            std::copy(ptr, ptr+m*r, t->U().data());  ptr += m*r;
            std::copy(ptr, ptr+r*n, t->V().data());  ptr += r*n;
            Tij.emplace_back(t);
          } else {
            auto t = new DenseTile<scalar_t>(m, n);
            std::copy(ptr, ptr+m*n, t->D().data());  ptr += m*n;
            Tij.emplace_back(t);
          }
        }
      return Tij;
    }

    template<typename scalar_t>
    std::vector<std::unique_ptr<BLRTile<scalar_t>>>
    BLRMatrixMPI<scalar_t>::bcast_col_of_tiles_along_rows
    (std::size_t i0, std::size_t i1, std::size_t j) const {
      int src = j % grid()->npcols();
      std::size_t msg_size = 0, nr_tiles = 0;
      std::vector<std::int64_t> ranks;
      if (grid()->is_local_col(j)) {
        for (std::size_t i=i0; i<i1; i++)
          if (grid()->is_local_row(i)) {
            msg_size += tile(i, j).nonzeros();
            ranks.push_back(tile(i, j).is_low_rank() ?
                            tile(i, j).rank() : -1);
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
      std::vector<scalar_t> buf(msg_size);
      auto ptr = buf.data();
      if (grid()->is_local_col(j)) {
        for (std::size_t i=i0; i<i1; i++)
          if (grid()->is_local_row(i)) {
            auto& t = tile(i, j);
            if (t.is_low_rank()) {
              std::copy(t.U().data(), t.U().end(), ptr);
              ptr += t.U().rows()*t.U().cols();
              std::copy(t.V().data(), t.V().end(), ptr);
              ptr += t.V().rows()*t.V().cols();
            } else {
              std::copy(t.D().data(), t.D().end(), ptr);
              ptr += t.D().rows()*t.D().cols();
            }
          }
      }
      grid()->row_comm().broadcast_from(buf, src);
      Tij.reserve(nr_tiles);
      ptr = buf.data();
      auto n = tilecols(j);
      for (std::size_t i=i0; i<i1; i++)
        if (grid()->is_local_row(i)) {
          auto r = ranks[Tij.size()];
          auto m = tilerows(i);
          if (r != -1) {
            auto t = new LRTile<scalar_t>(m, n, r);
            std::copy(ptr, ptr+m*r, t->U().data());  ptr += m*r;
            std::copy(ptr, ptr+r*n, t->V().data());  ptr += r*n;
            Tij.emplace_back(t);
          } else {
            auto t = new DenseTile<scalar_t>(m, n);
            std::copy(ptr, ptr+m*n, t->D().data());  ptr += m*n;
            Tij.emplace_back(t);
          }
        }
      return Tij;
    }

    template<typename scalar_t>
    std::vector<std::unique_ptr<BLRTile<scalar_t>>>
    BLRMatrixMPI<scalar_t>::gather_rows
    (std::size_t i0, std::size_t i1, std::size_t j0, std::size_t j1) const {
      //TODO: avoid resending, instead gather and forward tiles after update step
      if (j1 > j0 && i1 > i0){
        std::size_t msg_size = 0;
        std::vector<std::int64_t> ranks;
        //CASE 1: send tiles of row 0:i0-1 of col j0 to all processes in col j0
        if (grid()->is_local_col(j0)) {
          for (std::size_t i=0; i<i0; i++){
            if (grid()->is_local_row(i)) {
              msg_size += tile(i, j0).nonzeros();
              ranks.push_back(tile(i, j0).is_low_rank() ?
                              tile(i, j0).rank() : -1);
            } 
          }
        }
        std::vector<std::unique_ptr<BLRTile<scalar_t>>> Tij;
        //CASE 1: send tiles of row 0:i0-1 to all processes in col j0
        std::size_t nr_tiles=0;
        std::vector<scalar_t> buf;
        std::vector<std::int64_t> all_ranks;
        std::vector<int> rcnts;
        std::vector<int> tile_displs;
        std::vector<int> displs;
        if (grid()->is_local_col(j0)) {
          ranks.push_back(msg_size);
          rcnts.resize(grid()->nprows());
          rcnts[grid()->prow()]=ranks.size();
          grid()->col_comm().all_gather(rcnts.data(), 1);
          displs.resize(grid()->nprows());
          for (std::size_t i=1; i<rcnts.size(); i++){
            displs[i]=displs[i-1]+rcnts[i-1];
          }
          all_ranks.resize(std::accumulate(rcnts.begin(),rcnts.end(),0));
          nr_tiles=all_ranks.size()-rcnts.size();
          std::copy(ranks.begin(), ranks.end(), all_ranks.begin()+displs[grid()->prow()]);
          grid()->col_comm().all_gather_v(all_ranks.data(), rcnts.data(), displs.data());
          std::size_t total_msg_size = 0;
          for (std::size_t i=0; i<rcnts.size(); i++){
            total_msg_size += all_ranks[displs[i]+rcnts[i]-1];
          }
          buf.resize(total_msg_size);
          std::vector<int> tile_rcnts(grid()->nprows());
          for (int i=0; i<grid()->nprows(); i++)
            tile_rcnts[i] = all_ranks[displs[i]+rcnts[i]-1];
          tile_displs.resize(grid()->nprows());
          for (std::size_t i=1; i<tile_rcnts.size(); i++){
            tile_displs[i]=tile_displs[i-1]+tile_rcnts[i-1];
          }
          auto ptr = buf.data() + tile_displs[grid()->prow()];
          for (std::size_t i=0; i<i0; i++){
            if (grid()->is_local_row(i)) {
              auto& t = tile(i, j0);
              if (tile(i, j0).is_low_rank()) {
                std::copy(t.U().data(), t.U().end(), ptr);
                ptr += t.U().rows()*t.U().cols();
                std::copy(t.V().data(), t.V().end(), ptr);
                ptr += t.V().rows()*t.V().cols();
              } else {
                std::copy(t.D().data(), t.D().end(), ptr);
                ptr += t.D().rows()*t.D().cols();
              }
            }
          }
          grid()->col_comm().all_gather_v(buf.data(), tile_rcnts.data(), tile_displs.data());
        }
        //CASE 2: cols j0+1:end, gather in proc in row i0
        std::size_t msg_size2 = 0;
        std::vector<std::int64_t> ranks2;
        for (std::size_t j=j0+1; j<j1; j++){
          if (grid()->is_local_col(j)) {
            for (std::size_t i=0; i<i0; i++){
              if (grid()->is_local_row(i)) {
                msg_size2 += tile(i, j).nonzeros();
                ranks2.push_back(tile(i, j).is_low_rank() ?
                              tile(i, j).rank() : -1);
              }
            }
          }
        }
        std::vector<scalar_t> buf2;
        std::vector<std::int64_t> all_ranks2;
        std::vector<int> rcnts2, tile_displs2, displs2;
        std::size_t rcnts_empty=0;
        std::size_t col_cnt=0;
        //CASE 2: cols j0+1:end, gather in proc in row i0
        for (int j=0; j<grid()->npcols(); j++){
          if (grid()->pcol() == j % grid()->npcols()) {
            for (std::size_t k=j0+1; k<j1; k++){
              if (grid()->is_local_col(k)) col_cnt++;
            }
            if (col_cnt!=0){
              int src = j0 % grid()->nprows();
              if (!(ranks2.empty())) ranks2.push_back(msg_size2);
              if (grid()->prow() == src && ranks2.empty()) ranks2.push_back(msg_size2);
              int scnt=0;
              if (grid()->prow() == src){
                rcnts2.resize(grid()->nprows());
                rcnts2[grid()->prow()]=ranks2.size();
                scnt=ranks2.size();
              } else{
                scnt=ranks2.size();
              }
              grid()->col_comm().gather(&scnt, 1, rcnts2.data(), 1, src);
              if (grid()->prow() == src){
                displs2.resize(grid()->nprows());
                displs2[0]=0;
                for (std::size_t i=1; i<rcnts2.size(); i++){
                  displs2[i]=displs2[i-1]+rcnts2[i-1];
                }
                all_ranks2.resize(std::accumulate(rcnts2.begin(),rcnts2.end(),0));
                std::copy(ranks2.begin(), ranks2.end(), all_ranks2.begin()+displs2[grid()->prow()]);//?? works if ranks empty??
                for (std::size_t i=0; i<rcnts2.size(); i++){
                  if (rcnts2[i]==0) rcnts_empty++;
                }
                nr_tiles+=all_ranks2.size()-rcnts2.size()+rcnts_empty;
              }
              grid()->col_comm().gather_v(ranks2.data(), scnt, all_ranks2.data(), rcnts2.data(), displs2.data(), src);
              std::vector<int> tile_rcnts;
              if (grid()->prow() == src){
                std::size_t total_msg_size = 0;
                for (std::size_t i=0; i<rcnts2.size(); i++){
                  if (rcnts2[i] == 0){
                    total_msg_size += 0;
                  } else {
                    total_msg_size += all_ranks2[displs2[i]+rcnts2[i]-1];
                  }
                }
                buf2.resize(total_msg_size);
                tile_rcnts.resize(grid()->nprows());
                for (int i=0; i<grid()->nprows(); i++){
                  if (rcnts2[i] == 0){
                    tile_rcnts[i] = 0;
                  } else{
                    tile_rcnts[i] = all_ranks2[displs2[i]+rcnts2[i]-1];
                  }
                }
                tile_displs2.resize(grid()->nprows());
                tile_displs2[0]=0;
                for (std::size_t i=1; i<tile_rcnts.size(); i++){
                  tile_displs2[i]=tile_displs2[i-1]+tile_rcnts[i-1];
                }
              }
              std::vector<scalar_t> sbuf(msg_size2);
              auto ptr = sbuf.data();
              for (std::size_t k=j0+1; k<j1; k++){
                if (grid()->is_local_col(k)){
                  for (std::size_t i=0; i<i0; i++){
                    if (grid()->is_local_row(i)) {
                      auto& t = tile(i, k);
                      if (tile(i, k).is_low_rank()) {
                        std::copy(t.U().data(), t.U().end(), ptr);
                        ptr += t.U().rows()*t.U().cols();
                        std::copy(t.V().data(), t.V().end(), ptr);
                        ptr += t.V().rows()*t.V().cols();
                      } else {
                        std::copy(t.D().data(), t.D().end(), ptr);
                        ptr += t.D().rows()*t.D().cols();
                      }
                    }
                  }
                }
              }
              grid()->col_comm().gather_v(sbuf.data(), msg_size2, buf2.data(), tile_rcnts.data(), tile_displs2.data(), src);
            }
          }
        }
        if (nr_tiles==0) return Tij;
        Tij.reserve(nr_tiles);
        if (grid()->is_local_col(j0)) {
          std::vector<scalar_t*> ptr(grid()->col_comm().size());
          std::vector<std::int64_t> i_ranks(grid()->col_comm().size());
          for (std::size_t p=0; p<ptr.size(); p++){
            ptr[p] = buf.data() + tile_displs[p];
            i_ranks[p] = displs[p];
          }
          auto n = tilecols(j0);
          for (std::size_t i=0; i<i0; i++){
            int sender=grid()->rg2p(i); 
            auto m = tilerows(i);
            auto r = all_ranks[i_ranks[sender]];
            if (r != -1) {
              auto t = new LRTile<scalar_t>(m, n, r);
              std::copy(ptr[sender], ptr[sender]+m*r, t->U().data());  ptr[sender] += m*r;
              std::copy(ptr[sender], ptr[sender]+r*n, t->V().data());  ptr[sender] += r*n;
              Tij.emplace_back(t);
            } else {
              auto t = new DenseTile<scalar_t>(m, n);
              std::copy(ptr[sender], ptr[sender]+m*n, t->D().data());  ptr[sender] += m*n;
              Tij.emplace_back(t);
            }
            i_ranks[sender]++;
          }
        }
        if (grid()->is_local_row(i0)){
          if (col_cnt!=0){
            std::vector<scalar_t*> ptr(grid()->col_comm().size());
            std::vector<std::int64_t> i_ranks(grid()->col_comm().size());
            for (std::size_t p=0; p<ptr.size(); p++){
              ptr[p] = buf2.data() + tile_displs2[p];
              i_ranks[p] = displs2[p];
            }
            for (std::size_t j=j0+1; j<j1; j++){
              if (grid()->is_local_col(j)) {
                auto n = tilecols(j);
                for (std::size_t i=0; i<i0; i++){
                  int sender=grid()->rg2p(i); 
                  auto m = tilerows(i);
                  auto r = all_ranks2[i_ranks[sender]];
                  if (r != -1) {
                    auto t = new LRTile<scalar_t>(m, n, r);
                    std::copy(ptr[sender], ptr[sender]+m*r, t->U().data());  ptr[sender] += m*r;
                    std::copy(ptr[sender], ptr[sender]+r*n, t->V().data());  ptr[sender] += r*n;
                    Tij.emplace_back(t);
                  } else {
                    auto t = new DenseTile<scalar_t>(m, n);
                    std::copy(ptr[sender], ptr[sender]+m*n, t->D().data());  ptr[sender] += m*n;
                    Tij.emplace_back(t);
                  }
                  i_ranks[sender]++;
                }
              }
            }
          }
        }
        return Tij;
      } else{
        std::vector<std::unique_ptr<BLRTile<scalar_t>>> Tij;
        return Tij;
      }
    }

    template<typename scalar_t>
    std::vector<std::unique_ptr<BLRTile<scalar_t>>>
    BLRMatrixMPI<scalar_t>::gather_cols
    (std::size_t i0, std::size_t i1, std::size_t j0, std::size_t j1) const {
      if (i1 > i0 && j1 > j0){
        //TODO: avoid resending, instead gather and forward tiles after update step
        std::size_t msg_size = 0;
        std::vector<std::int64_t> ranks;
        //CASE 1: send tiles of col 0:j0-1 of row i0 to all processes in row i0
        if (grid()->is_local_row(i0)) {
          for (std::size_t j=0; j<j0; j++){
            if (grid()->is_local_col(j)) {
              msg_size += tile(i0, j).nonzeros();
              ranks.push_back(tile(i0, j).is_low_rank() ?
                              tile(i0, j).rank() : -1);
            } 
          }
        }
        //CASE 2: rows i0+1:end, gather in proc in col j0
        std::size_t msg_size2 = 0;
        std::vector<std::int64_t> ranks2;
        for (std::size_t i=i0+1; i<i1; i++){
          if (grid()->is_local_row(i)) {
            for (std::size_t j=0; j<j0; j++){
              if (grid()->is_local_col(j)) {
                msg_size2 += tile(i, j).nonzeros();
                ranks2.push_back(tile(i, j).is_low_rank() ?
                              tile(i, j).rank() : -1);
              }
            }
          }
        }
        std::vector<std::unique_ptr<BLRTile<scalar_t>>> Tij;
        //CASE 1: row i0, send to all processes in row
        std::size_t nr_tiles=0;
        std::vector<scalar_t> buf;
        std::vector<std::int64_t> all_ranks;
        std::vector<int> rcnts, tile_displs, displs;
        if (grid()->is_local_row(i0)) {
          ranks.push_back(msg_size);
          rcnts.resize(grid()->npcols());
          rcnts[grid()->pcol()]=ranks.size();
          grid()->row_comm().all_gather(rcnts.data(), 1);
          displs.resize(grid()->npcols());
          for (std::size_t j=1; j<rcnts.size(); j++){
            displs[j]=displs[j-1]+rcnts[j-1];
          }
          all_ranks.resize(std::accumulate(rcnts.begin(),rcnts.end(),0));
          nr_tiles=all_ranks.size()-rcnts.size();
          std::copy(ranks.begin(), ranks.end(), all_ranks.begin()+displs[grid()->pcol()]);
          grid()->row_comm().all_gather_v(all_ranks.data(), rcnts.data(), displs.data());
          std::size_t total_msg_size = 0;
          for (std::size_t j=0; j<rcnts.size(); j++){
            total_msg_size += all_ranks[displs[j]+rcnts[j]-1];
          }
          buf.resize(total_msg_size);
          std::vector<int> tile_rcnts(grid()->npcols());
          for (int j=0; j<grid()->npcols(); j++)
            tile_rcnts[j] = all_ranks[displs[j]+rcnts[j]-1];
          tile_displs.resize(grid()->npcols());
          for (std::size_t j=1; j<tile_rcnts.size(); j++){
            tile_displs[j]=tile_displs[j-1]+tile_rcnts[j-1];
          }
          auto ptr = buf.data() + tile_displs[grid()->pcol()];
          for (std::size_t j=0; j<j0; j++){
            if (grid()->is_local_col(j)) {
              auto& t = tile(i0, j);
              if (tile(i0, j).is_low_rank()) {
                std::copy(t.U().data(), t.U().end(), ptr);
                ptr += t.U().rows()*t.U().cols();
                std::copy(t.V().data(), t.V().end(), ptr);
                ptr += t.V().rows()*t.V().cols();
              } else {
                std::copy(t.D().data(), t.D().end(), ptr);
                ptr += t.D().rows()*t.D().cols();
              }
            }
          }
          grid()->row_comm().all_gather_v(buf.data(), tile_rcnts.data(), tile_displs.data());
        }
        //CASE 2: rows i0+1:end, gather in proc in col j0
        std::vector<scalar_t> buf2;
        std::vector<std::int64_t> all_ranks2;
        std::vector<int> rcnts2, displs2, tile_displs2;
        std::size_t rcnts_empty=0;
        std::size_t row_cnt=0;
        for (int i=0; i<grid()->nprows(); i++){
          if (grid()->prow() == i % grid()->nprows()) {
            //TODO: easier way to check if processor owns rows 
            for (std::size_t k=i0+1; k<i1; k++){
              if (grid()->is_local_row(k)) row_cnt++;
            }
            if (row_cnt!=0){
              int src = i0 % grid()->npcols(); //int sender = grid()->rg2p(i0);
              if (!(ranks2.empty())) ranks2.push_back(msg_size2);
              if (grid()->pcol() == src && ranks2.empty()) ranks2.push_back(msg_size2);
              int scnt=0;
              if (grid()->pcol() == src){
                rcnts2.resize(grid()->npcols());
                rcnts2[grid()->pcol()]=ranks2.size();
                scnt=ranks2.size();
              } else{
                scnt=ranks2.size();
              }
              grid()->row_comm().gather(&scnt, 1, rcnts2.data(), 1, src);
              if (grid()->pcol() == src){
                displs2.resize(grid()->npcols());
                displs2[0]=0;
                for (std::size_t j=1; j<rcnts2.size(); j++){
                  displs2[j]=displs2[j-1]+rcnts2[j-1];
                }
                all_ranks2.resize(std::accumulate(rcnts2.begin(),rcnts2.end(),0));
                std::copy(ranks2.begin(), ranks2.end(), all_ranks2.begin()+displs2[grid()->pcol()]);//?? works if ranks empty??
                for (std::size_t j=0; j<rcnts2.size(); j++){
                  if (rcnts2[j]==0) rcnts_empty++;
                }
                nr_tiles+=all_ranks2.size()-rcnts2.size()+rcnts_empty;
              }
              grid()->row_comm().gather_v(ranks2.data(), scnt, all_ranks2.data(), rcnts2.data(), displs2.data(), src);
              std::vector<int> tile_rcnts;
              if (grid()->pcol() == src){
                std::size_t total_msg_size = 0;
                for (std::size_t j=0; j<rcnts2.size(); j++){
                  if (rcnts2[j] == 0){
                    total_msg_size += 0;
                  } else {
                    total_msg_size += all_ranks2[displs2[j]+rcnts2[j]-1];
                  }
                }
                buf2.resize(total_msg_size);
                tile_rcnts.resize(grid()->npcols());
                for (int j=0; j<grid()->npcols(); j++){
                  if (rcnts2[j] == 0){
                    tile_rcnts[j] = 0;
                  } else{
                    tile_rcnts[j] = all_ranks2[displs2[j]+rcnts2[j]-1];
                  }
                }
                tile_displs2.resize(grid()->npcols());
                tile_displs2[0]=0;
                for (std::size_t j=1; j<tile_rcnts.size(); j++){
                  tile_displs2[j]=tile_displs2[j-1]+tile_rcnts[j-1];
                }
              }
              std::vector<scalar_t> sbuf(msg_size2);
              auto ptr = sbuf.data();
              for (std::size_t k=i0+1; k<i1; k++){
                if (grid()->is_local_row(k)){
                  for (std::size_t j=0; j<j0; j++){
                    if (grid()->is_local_col(j)) {
                      auto& t = tile(k, j);
                      if (tile(k, j).is_low_rank()) { 
                        std::copy(t.U().data(), t.U().end(), ptr);
                        ptr += t.U().rows()*t.U().cols();
                        std::copy(t.V().data(), t.V().end(), ptr);
                        ptr += t.V().rows()*t.V().cols();
                      } else {
                        std::copy(t.D().data(), t.D().end(), ptr);
                        ptr += t.D().rows()*t.D().cols();
                      }
                    }
                  }
                }
              }
              grid()->row_comm().gather_v(sbuf.data(), msg_size2, buf2.data(), tile_rcnts.data(), tile_displs2.data(), src);
            }
          }
        }
        if (nr_tiles==0) return Tij;
        Tij.reserve(nr_tiles);
        if (grid()->is_local_row(i0)) {
          std::vector<scalar_t*> ptr(grid()->row_comm().size());
          std::vector<std::int64_t> j_ranks(grid()->row_comm().size());
          for (std::size_t p=0; p<ptr.size(); p++){
            ptr[p] = buf.data() + tile_displs[p];
            j_ranks[p] = displs[p];
          }
          auto m = tilerows(i0);
          for (std::size_t j=0; j<j0; j++){
            int sender = grid()->cg2p(j);
            auto n = tilecols(j);
            auto r = all_ranks[j_ranks[sender]];
            if (r != -1) {
              auto t = new LRTile<scalar_t>(m, n, r);
              std::copy(ptr[sender], ptr[sender]+m*r, t->U().data());  ptr[sender] += m*r;
              std::copy(ptr[sender], ptr[sender]+r*n, t->V().data());  ptr[sender] += r*n;
              Tij.emplace_back(t);
            } else {
              auto t = new DenseTile<scalar_t>(m, n);
              std::copy(ptr[sender], ptr[sender]+m*n, t->D().data());  ptr[sender] += m*n;
              Tij.emplace_back(t);
            }
            j_ranks[sender]++;
          }
        }
        if (grid()->is_local_col(j0)){
          if (row_cnt!=0){
            std::vector<scalar_t*> ptr(grid()->row_comm().size());
            std::vector<std::int64_t> j_ranks(grid()->row_comm().size());
            for (std::size_t p=0; p<ptr.size(); p++){
              ptr[p] = buf2.data() + tile_displs2[p];
              j_ranks[p] = displs2[p];
            }
            for (std::size_t i=i0+1; i<i1; i++){
              if (grid()->is_local_row(i)) {
                auto m = tilerows(i);
                for (std::size_t j=0; j<j0; j++){
                  int sender = grid()->cg2p(j); 
                  auto n = tilecols(j);
                  auto r = all_ranks2[j_ranks[sender]];
                  if (r != -1) {
                    auto t = new LRTile<scalar_t>(m, n, r);
                    std::copy(ptr[sender], ptr[sender]+m*r, t->U().data());  ptr[sender] += m*r;
                    std::copy(ptr[sender], ptr[sender]+r*n, t->V().data());  ptr[sender] += r*n;
                    Tij.emplace_back(t);
                  } else {
                    auto t = new DenseTile<scalar_t>(m, n);
                    std::copy(ptr[sender], ptr[sender]+m*n, t->D().data());  ptr[sender] += m*n;
                    Tij.emplace_back(t);
                  }
                  j_ranks[sender]++;
                }
              }
            }
          }
        }
        return Tij;
      } else{
        std::vector<std::unique_ptr<BLRTile<scalar_t>>> Tij;
        return Tij;
      }
    }

    template<typename scalar_t> void
    BLRMatrixMPI<scalar_t>::compress_tile
    (std::size_t i, std::size_t j, const Opts_t& opts) {
      auto t = tile(i, j).compress(opts);
      if (t->rank()*(t->rows() + t->cols()) < t->rows()*t->cols())
        block(i, j) = std::move(t);
    }

    template<typename scalar_t> std::vector<int>
    BLRMatrixMPI<scalar_t>::factor(const Opts_t& opts) {
      adm_t adm(rowblocks(), colblocks());
      adm.fill(true);
      return factor(adm, opts);
    }

    template<typename scalar_t> std::vector<int>
    BLRMatrixMPI<scalar_t>::factor(const adm_t& adm, const Opts_t& opts) {
      std::vector<int> piv, piv_tile;
      if (!grid()->active()) return piv;
      DenseTile<scalar_t> Tii;
      for (std::size_t i=0; i<rowblocks(); i++) {
        if (grid()->is_local_row(i)) {
          // LU factorization of diagonal tile
          if (grid()->is_local_col(i))
            piv_tile = tile(i, i).LU();
          else piv_tile.resize(tilerows(i));
          grid()->row_comm().broadcast_from(piv_tile, i % grid()->npcols());
          int r0 = tileroff(i);
          std::transform
            (piv_tile.begin(), piv_tile.end(), std::back_inserter(piv),
             [r0](int p) -> int { return p + r0; });
          Tii = bcast_dense_tile_along_row(i, i);
        }
        if (grid()->is_local_col(i))
          Tii = bcast_dense_tile_along_col(i, i);
        if (grid()->is_local_row(i)) {
          for (std::size_t j=i+1; j<colblocks(); j++) {
            if (grid()->is_local_col(j)) {
              if (adm(i, j)) compress_tile(i, j, opts);
              tile(i, j).laswp(piv_tile, true);
              trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                   scalar_t(1.), Tii, tile(i, j));
            }
          }
        }
        if (grid()->is_local_col(i)) {
          for (std::size_t j=i+1; j<rowblocks(); j++) {
            if (grid()->is_local_row(j)) {
              if (adm(j, i)) compress_tile(j, i, opts);
              trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                   scalar_t(1.), Tii, tile(j, i));
            }
          }
        }
#if 0 //RL-Update
        auto Tij = bcast_row_of_tiles_along_cols(i, i+1, rowblocks());
        auto Tki = bcast_col_of_tiles_along_rows(i+1, rowblocks(), i);
        for (std::size_t k=i+1, lk=0; k<rowblocks(); k++) {
          if (grid()->is_local_row(k)) {
            for (std::size_t j=i+1, lj=0; j<colblocks(); j++) {
              if (grid()->is_local_col(j)) {
                // this uses .D, assuming tile(k, j) is dense
                gemm(Trans::N, Trans::N, scalar_t(-1.), *(Tki[lk]),
                     *(Tij[lj]), scalar_t(1.), tile_dense(k, j).D());
                lj++;
              }
            }
            lk++;
          }
        }
#else //LL-Update
        auto Tik = gather_rows(i+1, rowblocks(), i+1, colblocks());
        auto Tkj = gather_cols(i+1, rowblocks(), i+1, colblocks());
        //GEMM
        if (grid()->is_local_row(i+1)) {
          std::size_t lk=0;
          for (std::size_t j=i+1; j<rowblocks(); j++) {
            if (grid()->is_local_col(j)) {
              for (std::size_t k=0, lj=0; k<i+1; k++) {
                gemm(Trans::N, Trans::N, scalar_t(-1.), *(Tkj[lj]),
                  *(Tik[lk]), scalar_t(1.), tile_dense(i+1, j).D());
                lj++;
                lk++;
              }
            }
          }
        }
        if (grid()->is_local_col(i+1)) {
          std::size_t lj=0;
          if (grid()->is_local_row(i+1)) lj=i+1;
          for (std::size_t j=i+2; j<rowblocks(); j++) { //check rowblocks and colblocks
            if (grid()->is_local_row(j)) {
              for (std::size_t k=0, lk=0; k<i+1; k++) {
                gemm(Trans::N, Trans::N, scalar_t(-1.), *(Tkj[lj]),
                    *(Tik[lk]), scalar_t(1.), tile_dense(j, i+1).D());
                lj++;
                lk++;
              }
            }
          }
        }
#endif
      }
      return piv;
    }


    template<typename scalar_t> std::vector<int>
    BLRMatrixMPI<scalar_t>::partial_factor(BLRMPI_t& A11, BLRMPI_t& A12,
                                           BLRMPI_t& A21, BLRMPI_t& A22,
                                           const adm_t& adm,
                                           const Opts_t& opts) {
      assert(A11.rows() == A12.rows() && A11.cols() == A21.cols() &&
             A21.rows() == A22.rows() && A12.cols() == A22.cols());
      assert(A11.grid() == A12.grid() && A11.grid() == A21.grid() &&
             A11.grid() == A22.grid());
      auto B1 = A11.rowblocks();
      auto B2 = A22.rowblocks();
      auto g = A11.grid();
      std::vector<int> piv, piv_tile;
      if (!g->active()) return piv;
      DenseTile<scalar_t> Tii;
      for (std::size_t i=0; i<B1; i++) {
#pragma omp parallel
        {
#pragma omp master
          {
            if (g->is_local_row(i)) {
              if (g->is_local_col(i))
                // LU factorization of diagonal tile
                piv_tile = A11.tile(i, i).LU();
              else piv_tile.resize(A11.tilerows(i));
            }
            if (g->is_local_row(i)) {
              g->row_comm().broadcast_from(piv_tile, i % g->npcols());
              int r0 = A11.tileroff(i);
              std::transform
                (piv_tile.begin(), piv_tile.end(), std::back_inserter(piv),
                 [r0](int p) -> int { return p + r0; });
              Tii = A11.bcast_dense_tile_along_row(i, i);
            }
            if (g->is_local_col(i))
              Tii = A11.bcast_dense_tile_along_col(i, i);
          }
#pragma omp single
          {
            if (g->is_local_row(i)) {
              // update trailing columns of A11
              for (std::size_t j=i+1; j<B1; j++)
                if (g->is_local_col(j) && adm(i, j))
#pragma omp task default(shared) firstprivate(i,j)
                  A11.compress_tile(i, j, opts);
              for (std::size_t j=0; j<B2; j++)
                if (g->is_local_col(j))
#pragma omp task default(shared) firstprivate(i,j)
                  A12.compress_tile(i, j, opts);
            }
            if (g->is_local_col(i)) {
              // update trailing rows of A11
              for (std::size_t j=i+1; j<B1; j++)
                if (g->is_local_row(j) && adm(j, i))
#pragma omp task default(shared) firstprivate(i,j)
                  A11.compress_tile(j, i, opts);
              // update trailing rows of A21
              for (std::size_t j=0; j<B2; j++)
                if (g->is_local_row(j))
#pragma omp task default(shared) firstprivate(i,j)
                  A21.compress_tile(j, i, opts);
            }
          }
        }
#pragma omp parallel
#pragma omp single
        {
          if (g->is_local_row(i)) {
            for (std::size_t j=i+1; j<B1; j++)
              if (g->is_local_col(j))
#pragma omp task default(shared) firstprivate(i,j)
                {
                  A11.tile(i, j).laswp(piv_tile, true);
                  trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                       scalar_t(1.), Tii, A11.tile(i, j));
                }
            for (std::size_t j=0; j<B2; j++)
              if (g->is_local_col(j))
#pragma omp task default(shared) firstprivate(i,j)
                {
                  A12.tile(i, j).laswp(piv_tile, true);
                  trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                       scalar_t(1.), Tii, A12.tile(i, j));
                }
          }
          if (g->is_local_col(i)) {
            for (std::size_t j=i+1; j<B1; j++)
              if (g->is_local_row(j))
#pragma omp task default(shared) firstprivate(i,j)
                trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                     scalar_t(1.), Tii, A11.tile(j, i));
            for (std::size_t j=0; j<B2; j++)
              if (g->is_local_row(j))
#pragma omp task default(shared) firstprivate(i,j)
                trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                     scalar_t(1.), Tii, A21.tile(j, i));
          }
        }
#if 1 //RL-Update
        auto Tij = A11.bcast_row_of_tiles_along_cols(i, i+1, B1);
        auto Tij2 = A12.bcast_row_of_tiles_along_cols(i, 0, B2);
        auto Tki = A11.bcast_col_of_tiles_along_rows(i+1, B1, i);
        auto Tk2i = A21.bcast_col_of_tiles_along_rows(0, B2, i);
#pragma omp parallel
#pragma omp single
        {
          for (std::size_t k=i+1, lk=0; k<B1; k++) {
            if (g->is_local_row(k)) {
              for (std::size_t j=i+1, lj=0; j<B1; j++) {
                if (g->is_local_col(j)) {
#pragma omp task default(shared) firstprivate(i,j,k,lk,lj)
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       *(Tki[lk]), *(Tij[lj]), scalar_t(1.),
                       A11.tile_dense(k, j).D());
                  lj++;
                }
              }
              lk++;
            }
          }
          for (std::size_t k=0, lk=0; k<B2; k++) {
            if (g->is_local_row(k)) {
              for (std::size_t j=i+1, lj=0; j<B1; j++) {
                if (g->is_local_col(j)) {
#pragma omp task default(shared) firstprivate(i,j,k,lk,lj)
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       *(Tk2i[lk]), *(Tij[lj]), scalar_t(1.),
                       A21.tile_dense(k, j).D());
                  lj++;
                }
              }
              lk++;
            }
          }
          for (std::size_t k=i+1, lk=0; k<B1; k++) {
            if (g->is_local_row(k)) {
              for (std::size_t j=0, lj=0; j<B2; j++) {
                if (g->is_local_col(j)) {
#pragma omp task default(shared) firstprivate(i,j,k,lk,lj)
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       *(Tki[lk]), *(Tij2[lj]), scalar_t(1.),
                       A12.tile_dense(k, j).D());
                  lj++;
                }
              }
              lk++;
            }
          }
          for (std::size_t k=0, lk=0; k<B2; k++) {
            if (g->is_local_row(k)) {
              for (std::size_t j=0, lj=0; j<B2; j++) {
                if (g->is_local_col(j)) {
#pragma omp task default(shared) firstprivate(i,j,k,lk,lj)
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       *(Tk2i[lk]), *(Tij2[lj]), scalar_t(1.),
                       A22.tile_dense(k, j).D());
                  lj++;
                }
              }
              lk++;
            }
          }
        }
#else //LL-Update
        auto Tik = A11.gather_rows(i+1, B1, i+1, B1);
        auto Tkj = A11.gather_cols(i+1, B1, i+1, B1);
        auto Tik2 = A12.gather_rows(i+1, B1, 0, B1);//(i, 0, B2);
        auto Tk2j = A21.gather_cols(0, B1, i+1, B1);
#endif
      }
      return piv;
    }

    template<typename scalar_t> void
    BLRMatrixMPI<scalar_t>::laswp(const std::vector<int>& piv, bool fwd) {
      if (!fwd) {
        std::cerr << "BLRMPI::laswp not implemented for fwd == false"
                  << std::endl;
        abort();
      }
      auto p = piv.data();
      for (std::size_t i=0; i<rowblocks(); i++)
        if (grid()->is_local_row(i)) {
          std::vector<int> tpiv;
          auto r0 = tileroff(i);
          std::transform
            (p, p+tilerows(i), std::back_inserter(tpiv),
             [r0](int pi) -> int { return pi - r0; });
          for (std::size_t j=0; j<colblocks(); j++)
            if (grid()->is_local_col(j))
              tile(i, j).laswp(tpiv, true);
          p += tilerows(i);
        }
    }

    template<typename scalar_t> void
    BLRMatrixMPI<scalar_t>::compress(const Opts_t& opts) {
      for (auto& b : blocks_) {
        if (b->is_low_rank()) continue;
        auto t = b->compress(opts);
        if (t->rank()*(t->rows() + t->cols()) < t->rows()*t->cols())
          b = std::move(t);
      }
    }

    template<typename scalar_t> BLRMatrixMPI<scalar_t>
    BLRMatrixMPI<scalar_t>::from_ScaLAPACK
    (const DistM_t& A, const ProcessorGrid2D& g, const Opts_t& opts) {
      auto l = opts.leaf_size();
      std::size_t
        nrt = std::max(1, int(std::ceil(float(A.rows()) / l))),
        nct = std::max(1, int(std::ceil(float(A.cols()) / l)));
      vec_t Rt(nrt, l), Ct(nct, l);
      Rt.back() = A.rows() - (nrt-1) * l;
      Ct.back() = A.cols() - (nct-1) * l;
      return from_ScaLAPACK(A, g, Rt, Ct);
    }


#if 1
    template<typename scalar_t> BLRMatrixMPI<scalar_t>
    BLRMatrixMPI<scalar_t>::from_ScaLAPACK
    (const DistM_t& A, const ProcessorGrid2D& g,
     const vec_t& Rt, const vec_t& Ct) {
      BLRMPI_t B(g, Rt, Ct);
      auto P = B.Comm().size();
      std::vector<std::vector<scalar_t>> sbuf(P);
      if (A.active()) {
        assert(A.fixed());
        const auto lm = A.lrows();
        const auto ln = A.lcols();
        const auto nprows = A.nprows();
        std::unique_ptr<int[]> work(new int[lm+ln]);
        auto pr = work.get();
        auto pc = pr + lm;
        for (int r=0; r<lm; r++)
          pr[r] = B.rg2p(A.rowl2g_fixed(r));
        for (int c=0; c<ln; c++)
          pc[c] = B.cg2p(A.coll2g_fixed(c)) * nprows;
        { // reserve space for the send buffers
          std::vector<std::size_t> cnt(sbuf.size());
          for (int c=0; c<ln; c++)
            for (int r=0, pcc=pc[c]; r<lm; r++)
              cnt[pr[r]+pcc]++;
          for (std::size_t p=0; p<sbuf.size(); p++)
            sbuf[p].reserve(cnt[p]);
        }
        for (int c=0; c<ln; c++)
          for (int r=0, pcc=pc[c]; r<lm; r++)
            sbuf[pr[r]+pcc].push_back(A(r,c));
      }
      std::vector<scalar_t,NoInit<scalar_t>> rbuf;
      std::vector<scalar_t*> pbuf;
      B.Comm().all_to_all_v(sbuf, rbuf, pbuf);
      B.fill(0.);
      if (B.active()) {
        const auto lm = B.lrows();
        const auto ln = B.lcols();
        const auto nprows = A.nprows();
        std::unique_ptr<int[]> work(new int[lm+ln]);
        auto pr = work.get();
        auto pc = pr + lm;
        for (std::size_t r=0; r<lm; r++)
          pr[r] = A.rowg2p(B.rl2g(r));
        for (std::size_t c=0; c<ln; c++)
          pc[c] = A.colg2p(B.cl2g(c)) * nprows;
        for (std::size_t c=0; c<ln; c++)
          for (std::size_t r=0, pcc=pc[c]; r<lm; r++)
            B(r, c) = *(pbuf[pr[r]+pcc]++);
      }
      return B;
    }
#else
    template<typename scalar_t> BLRMatrixMPI<scalar_t>
    BLRMatrixMPI<scalar_t>::from_ScaLAPACK
    (const DistM_t& A, const ProcessorGrid2D& g,
     const vec_t& Rt, const vec_t& Ct) {
      BLRMPI_t B(g, Rt, Ct);
      for (std::size_t j=0; j<B.colblocks(); j++)
        for (std::size_t i=0; i<B.rowblocks(); i++) {
          int dest = g.g2p(i, j);
          if (g.is_local(i, j)) {
            auto t = std::unique_ptr<DenseTile<scalar_t>>
              (new DenseTile<scalar_t>(B.tilerows(i), B.tilecols(j)));
            copy(B.tilerows(i), B.tilecols(j), A,
                 B.tileroff(i), B.tilecoff(j), t->D(), dest, A.ctxt_all());
            B.block(i, j) = std::move(t);
          } else {
            DenseM_t dummy;
            copy(B.tilerows(i), B.tilecols(j), A,
                 B.tileroff(i), B.tilecoff(j), dummy, dest, A.ctxt_all());
          }
        }
      return B;
    }
#endif

    template<typename scalar_t> DistributedMatrix<scalar_t>
    BLRMatrixMPI<scalar_t>::to_ScaLAPACK(const BLACSGrid* g) const {
      DistM_t A(g, rows(), cols());
      to_ScaLAPACK(A);
      return A;
    }

#if 1
    template<typename scalar_t> void
    BLRMatrixMPI<scalar_t>::to_ScaLAPACK(DistM_t& A) const {
      if (A.rows() != int(rows()) || A.cols() != int(cols()))
        A.resize(rows(), cols());
      int P = A.Comm().size();
      std::vector<std::vector<scalar_t>> sbuf(P);
      if (active()) {
        assert(A.fixed());
        const auto lm = lrows();
        const auto ln = lcols();
        const auto nprows = A.nprows();
        std::unique_ptr<int[]> work(new int[lm+ln]);
        auto pr = work.get();
        auto pc = pr + lm;
        for (std::size_t r=0; r<lm; r++)
          pr[r] = A.rowg2p_fixed(rl2g(r));
        for (std::size_t c=0; c<ln; c++)
          pc[c] = A.colg2p_fixed(cl2g(c)) * nprows;
        { // reserve space for the send buffers
          std::vector<std::size_t> cnt(sbuf.size());
          for (std::size_t c=0; c<ln; c++)
            for (std::size_t r=0, pcc=pc[c]; r<lm; r++)
              cnt[pr[r]+pcc]++;
          for (std::size_t p=0; p<sbuf.size(); p++)
            sbuf[p].reserve(cnt[p]);
        }
        for (std::size_t cB=0, lcB=0, c=0; cB<colblocks(); cB++) {
          if (grid()->is_local_col(cB)) {
            // expand a column of blocks to dense tiles
            std::vector<DenseTile<scalar_t>> cT;
            cT.reserve(rowblockslocal());
            for (std::size_t rB=0, lrB=0; rB<rowblocks(); rB++)
              if (grid()->is_local_row(rB)) {
                cT.emplace_back(ltile(lrB, lcB).dense());
                lrB++;
              }
            for (std::size_t lc=0; lc<tilecols(cB); lc++, c++) {
              auto pcc = pc[c];
              for (std::size_t rB=0, lrB=0, r=0; rB<rowblocks(); rB++)
                if (grid()->is_local_row(rB)) {
                  for (std::size_t lr=0; lr<tilerows(rB); lr++, r++)
                    sbuf[pr[r]+pcc].push_back(cT[lrB](lr,lc));
                  lrB++;
                }
            }
            lcB++;
          }
        }
      }
      std::vector<scalar_t,NoInit<scalar_t>> rbuf;
      std::vector<scalar_t*> pbuf;
      A.Comm().all_to_all_v(sbuf, rbuf, pbuf);
      if (A.active()) {
        const auto lm = A.lrows();
        const auto ln = A.lcols();
        const auto nprows = A.nprows();
        std::unique_ptr<int[]> work(new int[lm+ln]);
        auto pr = work.get();
        auto pc = pr + lm;
        for (int r=0; r<lm; r++)
          pr[r] = rg2p(A.rowl2g_fixed(r));
        for (int c=0; c<ln; c++)
          pc[c] = cg2p(A.coll2g_fixed(c)) * nprows;
        for (int c=0; c<ln; c++)
          for (int r=0, pcc=pc[c]; r<lm; r++)
            A(r,c) = *(pbuf[pr[r]+pcc]++);
      }
    }
#else
    template<typename scalar_t> void
    BLRMatrixMPI<scalar_t>::to_ScaLAPACK
    (DistributedMatrix<scalar_t>& A) const {
      if (A.rows() != int(rows()) || A.cols() != int(cols()))
        A.resize(rows(), cols());
      DenseM_t B;
      for (std::size_t j=0; j<colblocks(); j++)
        for (std::size_t i=0; i<rowblocks(); i++) {
          if (grid()->is_local(i, j))
            B = block(i, j)->dense();
          copy(tilerows(i), tilecols(j), B, grid()->g2p(i, j),
               A, tileroff(i), tilecoff(j), A.ctxt_all());
        }
    }
#endif

    // explicit template instantiations
    template class BLRMatrixMPI<float>;
    template class BLRMatrixMPI<double>;
    template class BLRMatrixMPI<std::complex<float>>;
    template class BLRMatrixMPI<std::complex<double>>;

    template<typename scalar_t> void
    trsv(UpLo ul, Trans ta, Diag d, const BLRMatrixMPI<scalar_t>& a,
         BLRMatrixMPI<scalar_t>& b) {
#if 0
      trsm(Side::L, ul, ta, d, scalar_t(1.), a, b);
#else
      if (!a.active()) return;
      assert(a.rows() == a.cols() && a.cols() == b.rows());
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      auto nb = b.rowblocks();
      auto g = a.grid();
      std::size_t nloc = 0;
      for (std::size_t i=0; i<nb; i++)
        if (g->is_local_row(i))
          nloc += b.tilerows(i);
      DenseM_t Bloc(nloc, b.cols());
      if (g->is_local_col(0)) {
        for (std::size_t i=0, lm=0; i<nb; i++)
          if (g->is_local_row(i)) {
            copy(b.tile(i, 0).D(), Bloc, lm, 0);
            lm += b.tilerows(i);
          }
      } else Bloc.zero();
      if (ul == UpLo::L) {
        for (std::size_t i=0, ln=0; i<nb; i++) {
          if (!(g->is_local_row(i) || g->is_local_col(i))) continue;
          auto n = b.tilerows(i);
          DenseM_t Bi(n, b.cols());
          if (g->is_local_row(i)) {
            copy(n, b.cols(), Bloc, ln, 0, Bi, 0, 0);
            auto& c = g->row_comm();
            c.reduce(Bi.data(), n*b.cols(), MPI_SUM, i % g->npcols());
            if (g->is_local_col(i)) {
              trsm(Side::L, ul, ta, d, scalar_t(1.), a.tile(i, i).D(), Bi);
              copy(Bi, Bloc, ln, 0);
            }
            ln += n;
          }
          if (g->is_local_col(i)) {
            auto& c = g->col_comm();
            c.broadcast_from(Bi.data(), n*b.cols(), i % g->nprows());
#pragma omp parallel
#pragma omp single
            for (std::size_t j=i+1, lm=ln; j<nb; j++) {
              if (g->is_local_row(j)) {
                auto m = b.tilerows(j);
#pragma omp task default(shared) firstprivate(m,lm)
                {
                  DenseMW_t Bj(m, b.cols(), Bloc, lm, 0);
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       a.tile(j, i), Bi, scalar_t(1.), Bj,
                       params::task_recursion_cutoff_level);
                }
                lm += m;
              }
            }
          }
        }
      } else { // ul == UpLo::U
        for (int i=nb-1, ln=nloc; i>=0; i--) {
          if (!(g->is_local_row(i) || g->is_local_col(i))) continue;
          auto n = b.tilerows(i);
          DenseM_t Bi(n, b.cols());
          if (g->is_local_row(i)) {
            ln -= n;
            copy(n, b.cols(), Bloc, ln, 0, Bi, 0, 0);
            auto& c = g->row_comm();
            c.reduce(Bi.data(), n*b.cols(), MPI_SUM, i % g->npcols());
            if (g->is_local_col(i)) {
              trsm(Side::L, ul, ta, d, scalar_t(1.), a.tile(i, i).D(), Bi);
              copy(Bi, Bloc, ln, 0);
            }
          }
          if (g->is_local_col(i)) {
            auto& c = g->col_comm();
            c.broadcast_from(Bi.data(), n*b.cols(), i % g->nprows());
            // TODO threading
#pragma omp parallel
#pragma omp single
            for (int j=0, lm=0; j<i; j++) {
              if (g->is_local_row(j)) {
                auto m = b.tilerows(j);
#pragma omp task default(shared) firstprivate(m,lm)
                {
                  DenseMW_t Bj(m, b.cols(), Bloc, lm, 0);
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       a.tile(j, i), Bi, scalar_t(1.), Bj,
                       params::task_recursion_cutoff_level);
                }
                lm += m;
              }
            }
          }
        }
      }
      for (std::size_t i=0, ln=0; i<nb; i++)
        if (g->is_local_row(i)) {
          auto& c = g->row_comm();
          auto n = b.tilerows(i);
          int src = i % g->npcols();
          if (c.rank() == src) {
            if (c.is_root())
              copy(n, b.cols(), Bloc, ln, 0, b.tile(i, 0).D(), 0, 0);
            else
              c.send(DenseM_t(n, b.cols(), Bloc, ln, 0).data(),
                     n*b.cols(), 0, 0);
          } else if (c.is_root())
            c.recv(b.tile(i, 0).D().data(), n*b.cols(), src, 0);
          ln += n;
        }
#endif
    }

    template<typename scalar_t> void
    gemv(Trans ta, scalar_t alpha, const BLRMatrixMPI<scalar_t>& a,
         const BLRMatrixMPI<scalar_t>& x, scalar_t beta,
         BLRMatrixMPI<scalar_t>& y) {
#if 0
      gemm(ta, Trans::N, alpha, a, x, beta, y);
#else
      if (!a.active()) return;
      if (ta != Trans::N) {
        std::cerr << "gemv BLRMPI operations not implemented" << std::endl;
        abort();
      }
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      auto nbx = x.rowblocks();
      auto nby = y.rowblocks();
      auto X = x.bcast_col_of_tiles_along_rows(0, nbx, 0);
      auto g = x.grid();
      std::size_t mloc = 0;
      for (std::size_t i=0; i<nby; i++)
        if (g->is_local_row(i))
          mloc += y.tilerows(i);
      DenseM_t Yloc(mloc, y.cols());
      Yloc.zero();
      for (std::size_t j=0, lj=0; j<nbx; j++) {
        if (g->is_local_col(j)) {
          auto m = x.tilerows(j);
          DenseM_t Xj(m, 1);
          int src = j % g->nprows();
          auto& c = g->col_comm();
          if (c.rank() == src) copy(X[lj]->D(), Xj);
          c.broadcast_from(Xj.data(), m, src);
#pragma omp parallel
#pragma omp single
          for (std::size_t i=0, lm=0; i<nby; i++)
            if (g->is_local_row(i)) {
              auto m = y.tilerows(i);
#pragma omp task default(shared) firstprivate(m,lm,i,j)
              {
                DenseMW_t Yi(m, y.cols(), Yloc, lm, 0);
                // TODO use gemv
                gemm(Trans::N, Trans::N, scalar_t(alpha),
                     a.tile(i, j), Xj, scalar_t(1.), Yi,
                     params::task_recursion_cutoff_level);
              }
              lm += m;
            }
        }
        if (g->is_local_row(j)) lj++;
      }
      g->row_comm().reduce(Yloc.data(), Yloc.rows()*Yloc.cols(), MPI_SUM);
      if (g->is_local_col(0))
#pragma omp parallel
#pragma omp single
        for (std::size_t i=0, lm=0; i<nby; i++)
          if (g->is_local_row(i)) {
            auto m = y.tilerows(i);
#pragma omp task default(shared) firstprivate(m,lm,i)
            y.tile(i, 0).D().scale_and_add
              (beta, DenseMW_t(m, y.cols(), Yloc, lm, 0));
            lm += m;
          }
#endif
    }

    template<typename scalar_t> void
    trsm(Side s, UpLo ul, Trans ta, Diag d,
         scalar_t alpha, const BLRMatrixMPI<scalar_t>& a,
         BLRMatrixMPI<scalar_t>& b) {
      if (!a.active()) return;
      if (s != Side::L) {
        std::cerr << "trsm BLRMPI operation not implemented" << std::endl;
        abort();
      }
      assert(a.rows() == a.cols() && a.cols() == b.rows());
      if (ul == UpLo::L) {
        for (std::size_t i=0; i<a.rowblocks(); i++) {
          if (a.grid()->is_local_row(i)) {
            auto Aii = a.bcast_dense_tile_along_row(i, i);
#pragma omp parallel
#pragma omp single
            for (std::size_t k=0; k<b.colblocks(); k++)
              if (b.grid()->is_local_col(k))
#pragma omp task default(shared) firstprivate(i,k)
                trsm(s, ul, ta, d, alpha, Aii, b.tile(i, k));
          }
          auto Aji = a.bcast_col_of_tiles_along_rows(i+1, a.rowblocks(), i);
          auto Bik = b.bcast_row_of_tiles_along_cols(i, 0, b.colblocks());
#pragma omp parallel
#pragma omp single
          for (std::size_t j=i+1, lj=0; j<a.rowblocks(); j++) {
            if (b.grid()->is_local_row(j)) {
              for (std::size_t k=0, lk=0; k<b.colblocks(); k++) {
                if (b.grid()->is_local_col(k)) {
#pragma omp task default(shared) firstprivate(j,k,lj,lk)
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       *(Aji[lj]), *(Bik[lk]), alpha, b.tile_dense(j, k).D());
                  lk++;
                }
              }
              lj++;
            }
          }
        }
      } else { // ul == UpLo::U
        for (int i=a.rowblocks()-1; i>=0; i--) {
          if (a.grid()->is_local_row(i)) {
            auto Aii = a.bcast_dense_tile_along_row(i, i);
#pragma omp parallel
#pragma omp single
            for (std::size_t k=0; k<b.colblocks(); k++)
              if (b.grid()->is_local_col(k))
#pragma omp task default(shared) firstprivate(i,k)
                trsm(s, ul, ta, d, alpha, Aii, b.tile(i, k));
          }
          auto Aji = a.bcast_col_of_tiles_along_rows(0, i, i);
          auto Bik = b.bcast_row_of_tiles_along_cols(i, 0, b.colblocks());
#pragma omp parallel
#pragma omp single
          for (int j=0, lj=0; j<i; j++) {
            if (b.grid()->is_local_row(j)) {
              for (std::size_t k=0, lk=0; k<b.colblocks(); k++) {
                if (b.grid()->is_local_col(k)) {
#pragma omp task default(shared) firstprivate(j,k,lj,lk)
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       *(Aji[lj]), *(Bik[lk]), alpha, b.tile_dense(j, k).D());
                  lk++;
                }
              }
              lj++;
            }
          }
        }
      }
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const BLRMatrixMPI<scalar_t>& a,
         const BLRMatrixMPI<scalar_t>& b, scalar_t beta,
         BLRMatrixMPI<scalar_t>& c) {
      if (!a.active()) return;
      if (ta != Trans::N || tb != Trans::N) {
        std::cerr << "gemm BLRMPI operations not implemented" << std::endl;
        abort();
      }
      assert(a.rows() == c.rows() && a.cols() == b.rows() &&
             b.cols() == c.cols());
      for (std::size_t k=0; k<a.colblocks(); k++) {
        auto Aik = a.bcast_col_of_tiles_along_rows(0, a.rowblocks(), k);
        auto Bkj = b.bcast_row_of_tiles_along_cols(k, 0, b.colblocks());
#pragma omp parallel
#pragma omp single
        for (std::size_t j=0, lj=0; j<c.colblocks(); j++) {
          if (c.grid()->is_local_col(j)) {
            for (std::size_t i=0, li=0; i<c.rowblocks(); i++) {
              if (c.grid()->is_local_row(i)) {
#pragma omp task default(shared) firstprivate(i,j,li,lj)
                gemm(ta, tb, alpha, *(Aik[li]), *(Bkj[lj]),
                     beta, c.tile_dense(i, j).D());
                li++;
              }
            }
            lj++;
          }
        }
      }
    }

    template void trsv(UpLo, Trans, Diag, const BLRMatrixMPI<float>&, BLRMatrixMPI<float>&);
    template void trsv(UpLo, Trans, Diag, const BLRMatrixMPI<double>&, BLRMatrixMPI<double>&);
    template void trsv(UpLo, Trans, Diag, const BLRMatrixMPI<std::complex<float>>&, BLRMatrixMPI<std::complex<float>>&);
    template void trsv(UpLo, Trans, Diag, const BLRMatrixMPI<std::complex<double>>&, BLRMatrixMPI<std::complex<double>>&);

    template void gemv(Trans, float, const BLRMatrixMPI<float>&, const BLRMatrixMPI<float>&, float, BLRMatrixMPI<float>&);
    template void gemv(Trans, double, const BLRMatrixMPI<double>&, const BLRMatrixMPI<double>&, double, BLRMatrixMPI<double>&);
    template void gemv(Trans, std::complex<float>, const BLRMatrixMPI<std::complex<float>>&, const BLRMatrixMPI<std::complex<float>>&, std::complex<float>, BLRMatrixMPI<std::complex<float>>&);
    template void gemv(Trans, std::complex<double>, const BLRMatrixMPI<std::complex<double>>&, const BLRMatrixMPI<std::complex<double>>&, std::complex<double>, BLRMatrixMPI<std::complex<double>>&);

    template void trsm(Side, UpLo, Trans, Diag, float, const BLRMatrixMPI<float>&, BLRMatrixMPI<float>&);
    template void trsm(Side, UpLo, Trans, Diag, double, const BLRMatrixMPI<double>&, BLRMatrixMPI<double>&);
    template void trsm(Side, UpLo, Trans, Diag, std::complex<float>, const BLRMatrixMPI<std::complex<float>>&, BLRMatrixMPI<std::complex<float>>&);
    template void trsm(Side, UpLo, Trans, Diag, std::complex<double>, const BLRMatrixMPI<std::complex<double>>&, BLRMatrixMPI<std::complex<double>>&);

    template void gemm(Trans, Trans, float, const BLRMatrixMPI<float>&, const BLRMatrixMPI<float>&, float, BLRMatrixMPI<float>&);
    template void gemm(Trans, Trans, double, const BLRMatrixMPI<double>&, const BLRMatrixMPI<double>&, double, BLRMatrixMPI<double>&);
    template void gemm(Trans, Trans, std::complex<float>, const BLRMatrixMPI<std::complex<float>>&, const BLRMatrixMPI<std::complex<float>>&, std::complex<float>, BLRMatrixMPI<std::complex<float>>&);
    template void gemm(Trans, Trans, std::complex<double>, const BLRMatrixMPI<std::complex<double>>&, const BLRMatrixMPI<std::complex<double>>&, std::complex<double>, BLRMatrixMPI<std::complex<double>>&);

  } // end namespace BLR
} // end namespace strumpack

