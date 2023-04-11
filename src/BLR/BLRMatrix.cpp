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

#include "BLRMatrix.hpp"
#include "BLRTileBLAS.hpp"

namespace strumpack {
  namespace BLR {

    template<typename scalar_t> BLRMatrix<scalar_t>::BLRMatrix
    (DenseM_t& A, const std::vector<std::size_t>& rowtiles,
     const std::vector<std::size_t>& coltiles, const Opts_t& opts)
      : BLRMatrix<scalar_t>(A.rows(), rowtiles, A.cols(), coltiles) {
      for (std::size_t j=0; j<colblocks(); j++)
        for (std::size_t i=0; i<rowblocks(); i++)
          block(i, j) = std::unique_ptr<BLRTile<scalar_t>>
            (new LRTile<scalar_t>(tile(A, i, j), opts));
    }

    template<typename scalar_t> BLRMatrix<scalar_t>::BLRMatrix
    (DenseM_t& A, const std::vector<std::size_t>& tiles,
     const adm_t& admissible, const Opts_t& opts)
      : BLRMatrix<scalar_t>(A.rows(), tiles, A.cols(), tiles) {
      assert(rowblocks() == colblocks());
      piv_.resize(rows());
      auto rb = rowblocks();
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
      // dummy for task synchronization
      std::unique_ptr<int[]> B_(new int[rb*rb]); auto B = B_.get();
#pragma omp taskgroup
#else
      int* B = nullptr;
#endif
      {
        for (std::size_t i=0; i<rb; i++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
          std::size_t ii = i+rb*i;
#pragma omp task default(shared) firstprivate(i,ii) depend(inout:B[ii])
#endif
          {
            create_dense_tile(i, i, A);
            auto tpiv = tile(i, i).LU();
            std::copy(tpiv.begin(), tpiv.end(), piv_.begin()+tileroff(i));
          }
          // COMPRESS and SOLVE
          for (std::size_t j=i+1; j<rb; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
            std::size_t ij = i+rb*j;
#pragma omp task default(shared) firstprivate(i,j,ii,ij)        \
  depend(in:B[ii]) depend(inout:B[ij])
#endif
            {
              if (admissible(i, j))
                create_LR_tile(i, j, A, opts);
              else create_dense_tile(i, j, A);
              // permute and solve with L, blocks right from the
              // diagonal block
              std::vector<int> tpiv
                (piv_.begin()+tileroff(i), piv_.begin()+tileroff(i+1));
              tile(i, j).laswp(tpiv, true);
              trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                 scalar_t(1.), tile(i, i), tile(i, j));
            }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
            std::size_t ji = j+rb*i;
#pragma omp task default(shared) firstprivate(i,j,ji,ii)        \
  depend(in:B[ii]) depend(inout:B[ji])
#endif
            {
              if (admissible(j, i))
                create_LR_tile(j, i, A, opts);
              else create_dense_tile(j, i, A);
              // solve with U, the blocks under the diagonal block
              trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                  scalar_t(1.), tile(i, i), tile(j, i));
            }
          }
          if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::RL) {
            for (std::size_t j=i+1; j<rb; j++) {
              for (std::size_t k=i+1; k<rb; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ij = i+rb*j, ki = k+rb*i, kj = k+rb*j;
#pragma omp task default(shared) firstprivate(i,j,k,ij,ki,kj)   \
  depend(in:B[ij],B[ki]) depend(inout:B[kj])
#endif
                { // Schur complement updates, always into full rank
                  auto Akj = tile(A, k, j);
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                      tile(k, i), tile(i, j), scalar_t(1.), Akj);
                }
              }
            }
          } else if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::LL) {
            // LL-Update
            for (std::size_t j=i+1; j<rb; j++){
              for (std::size_t k=0; k<i+1; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t ij = (i+1)+rb*j, ik = (i+1)+rb*k, kj = k+rb*j;
#pragma omp task default(shared) firstprivate(i,j,k,ij,ik,kj)   \
  depend(in:B[ik],B[kj]) depend(inout:B[ij]) priority(rb-j)
#endif
                { // Schur complement updates, always into full rank
                  auto Aij = tile(A, i+1, j);
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                      tile(i+1, k), tile(k, j), scalar_t(1.), Aij);
                }
                if (j != i+1) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t ji = j+rb*(i+1), jk = j+rb*k, ki = k+rb*(i+1);
#pragma omp task default(shared) firstprivate(i,j,k,ji,jk,ki)   \
  depend(in:B[jk],B[ki]) depend(inout:B[ji]) priority(rb-j)
#endif
                  { // Schur complement updates, always into full rank
                    auto Aji = tile(A, j, i+1);
                    gemm(Trans::N, Trans::N, scalar_t(-1.),
                        tile(j, k), tile(k, i+1), scalar_t(1.), Aji);
                  }
                }
              }
            }
          } else { //Comb or Star
            for (std::size_t j=i+1; j<rb; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ij = (i+1)+rb*j, i1j=ij-rb*(j-i), ij1=ij-1;
#pragma omp task default(shared) firstprivate(i,j,ij,i1j,ij1)   \
  depend(in:B[i1j],B[ij1]) depend(inout:B[ij])
#endif
              LUAR_B11(i+1, j, i+1, A, opts, B);
              if (j != i+1) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t ji = j+rb*(i+1), i1j=ji-rb, ij1=ji-j+i;
#pragma omp task default(shared) firstprivate(i,j,ji,i1j,ij1)   \
  depend(in:B[i1j],B[ij1]) depend(inout:B[ji])
#endif
                LUAR_B11(j, i+1, i+1, A, opts, B);
              }
            }
          }
        }
      }
      for (std::size_t i=0; i<rowblocks(); i++)
        for (std::size_t l=tileroff(i); l<tileroff(i+1); l++)
          piv_[l] += tileroff(i);
    }

    template<typename scalar_t> BLRMatrix<scalar_t>::BLRMatrix
    (std::size_t m, const std::vector<std::size_t>& rowtiles,
     std::size_t n, const std::vector<std::size_t>& coltiles)
      : m_(m), n_(n) {
      nbrows_ = rowtiles.size();
      nbcols_ = coltiles.size();
      roff_.resize(nbrows_+1);
      coff_.resize(nbcols_+1);
      cl2l_.resize(n_);
      rl2l_.resize(m_);
      for (std::size_t i=1; i<=nbrows_; i++)
        roff_[i] = roff_[i-1] + rowtiles[i-1];
      for (std::size_t j=1; j<=nbcols_; j++)
        coff_[j] = coff_[j-1] + coltiles[j-1];
      for (std::size_t b=0, l=0; b<nbcols_; b++) {
        for (std::size_t i=0; i<coff_[b+1]-coff_[b]; i++) {
          cl2l_[l] = i;
          l++;
        }
      }
      for (std::size_t b=0, l=0; b<nbrows_; b++) {
        for (std::size_t i=0; i<roff_[b+1]-roff_[b]; i++) {
          rl2l_[l] = i;
          l++;
        }
      }
      assert(roff_[nbrows_] == m_);
      assert(coff_[nbcols_] == n_);
      blocks_.resize(nbrows_ * nbcols_);
    }

    template<typename scalar_t> std::size_t
    BLRMatrix<scalar_t>::memory() const {
      std::size_t mem = 0;
      for (auto& b : blocks_) mem += b->memory();
      return mem;
    }

    template<typename scalar_t> std::size_t
    BLRMatrix<scalar_t>::nonzeros() const {
      std::size_t nnz = 0;
      for (auto& b : blocks_) nnz += b->nonzeros();
      return nnz;
    }

    template<typename scalar_t> std::size_t
    BLRMatrix<scalar_t>::rank() const {
      std::size_t mrank = 0;
      for (auto& b : blocks_) mrank = std::max(mrank, b->maximum_rank());
      return mrank;
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    BLRMatrix<scalar_t>::dense() const {
      DenseM_t A(rows(), cols());
      dense(A);
      return A;
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::dense(DenseM_t& A) const {
      assert(A.rows() == rows() && A.cols() == cols());
      auto cb = colblocks();
      auto rb = rowblocks();
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop collapse(2) default(shared)
#endif
      for (std::size_t j=0; j<cb; j++)
        for (std::size_t i=0; i<rb; i++) {
          DenseMW_t Aij = tile(A, i, j);
          tile(i, j).dense(Aij);
        }
    }

    template<typename scalar_t> void BLRMatrix<scalar_t>::draw
    (std::ostream& of, std::size_t roff, std::size_t coff) const {
      auto cb = colblocks();
      auto rb = rowblocks();
      for (std::size_t j=0; j<cb; j++)
        for (std::size_t i=0; i<rb; i++)
          tile(i, j).draw(of, roff+tileroff(i), coff+tilecoff(j));
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::print(const std::string& name) const {
      std::cout << "BLR(" << name << ")="
                << rows() << "x" << cols() << ", "
                << rowblocks() << "x" << colblocks() << ", "
                << (float(nonzeros()) / (rows()*cols()) * 100.) << "%"
                << " [" << std::endl;
      for (std::size_t i=0; i<nbrows_; i++) {
        for (std::size_t j=0; j<nbcols_; j++) {
          std::cout << "i= " << i << ", j= " << j << std::endl;
          auto& tij = tile(i, j);
          if (tij.is_low_rank())
            std::cout << "LR:" << tij.rows() << "x"
                      << tij.cols() << "/" << tij.rank() << " ";
          else
            std::cout << "D:" << tij.rows() << "x"
                      << tij.cols() << " " << std::endl;
        }
        std::cout << std::endl;
      }
      std::cout << "];" << std::endl;
    }

    template<typename scalar_t> void BLRMatrix<scalar_t>::clear() {
      m_ = n_ = nbrows_ = nbcols_ = 0;
      roff_.clear(); roff_.shrink_to_fit();
      coff_.clear(); coff_.shrink_to_fit();
      blocks_.clear(); blocks_.shrink_to_fit();
    }

    template<typename scalar_t> std::size_t
    BLRMatrix<scalar_t>::rg2t(std::size_t i) const {
      return std::distance
        (roff_.begin(), std::upper_bound(roff_.begin(), roff_.end(), i)) - 1;
    }
    template<typename scalar_t> std::size_t
    BLRMatrix<scalar_t>::cg2t(std::size_t j) const {
      return std::distance
        (coff_.begin(), std::upper_bound(coff_.begin(), coff_.end(), j)) - 1;
    }

    template<typename scalar_t> scalar_t
    BLRMatrix<scalar_t>::operator()(std::size_t i, std::size_t j) const {
      auto ti = std::distance
        (roff_.begin(), std::upper_bound(roff_.begin(), roff_.end(), i)) - 1;
      auto tj = std::distance
        (coff_.begin(), std::upper_bound(coff_.begin(), coff_.end(), j)) - 1;
      return tile(ti, tj)(i - roff_[ti], j - coff_[tj]);
    }

    template<typename scalar_t> scalar_t&
    BLRMatrix<scalar_t>::operator()(std::size_t i, std::size_t j) {
      auto ti = std::distance
        (roff_.begin(), std::upper_bound(roff_.begin(), roff_.end(), i)) - 1;
      auto tj = std::distance
        (coff_.begin(), std::upper_bound(coff_.begin(), coff_.end(), j)) - 1;
      return tile_dense(ti, tj).D()(i - roff_[ti], j - coff_[tj]);
    }

    /**
     * I and J should be sorted!!
     */
    template<typename scalar_t> DenseMatrix<scalar_t>
    BLRMatrix<scalar_t>::extract
    (const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J) const {
      auto m = I.size(); auto n = J.size();
      DenseM_t B(m, n);
      auto clo = coff_.begin();
      std::vector<std::size_t> lI, lJ;
      lI.reserve(m);
      lJ.reserve(n);
      for (std::size_t j=0; j<n; ) {
        clo = std::upper_bound(clo, coff_.end(), J[j]);
        auto tcol = std::distance(coff_.begin(), clo) - 1;
        lJ.clear();
        do {
          lJ.push_back(J[j] - coff_[tcol]);
          j++;
        } while (j < n && J[j] < *clo);
        auto rlo = roff_.begin();
        for (std::size_t i=0; i<m; ) {
          rlo = std::upper_bound(rlo, roff_.end(), I[i]);
          auto trow = std::distance(roff_.begin(), rlo) - 1;
          lI.clear();
          do {
            lI.push_back(I[i] - roff_[trow]);
            i++;
          } while (i < m && I[i] < *rlo);
          DenseMW_t lB(lI.size(), lJ.size(), B, i-lI.size(), j-lJ.size());
          tile(trow, tcol).extract(lI, lJ, lB);
        }
      }
      return B;
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::decompress() {
      for (std::size_t i=0; i<nbrows_; i++)
        for (std::size_t j=0; j<nbcols_; j++){
          auto &b = block(i, j);
          if (b && b->is_low_rank())
            b.reset(new DenseTile<scalar_t>(b->dense()));
        }
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::decompress_local_columns(int c_min, int c_max) {
      if (!c_max) return;
      for (std::size_t c=cg2t(c_min);
           c<=std::min(cg2t(c_max-1), nbcols_-1); c++) {
        for (std::size_t r=0; r<nbrows_; r++) {
          auto& b = block(r, c);
          if (b && b->is_low_rank())
            b.reset(new DenseTile<scalar_t>(b->dense()));
        }
      }
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::remove_tiles_before_local_column
    (int c_min, int c_max) {
      if (!c_max) return;
      for (std::size_t c=cg2t(c_min); c<cg2t(c_max-1); c++)
        for (std::size_t r=0; r<nbrows_; r++)
          block(r, c) = nullptr;
    }

    template<typename scalar_t> BLRTile<scalar_t>&
    BLRMatrix<scalar_t>::tile(std::size_t i, std::size_t j) {
      return *blocks_[i+j*rowblocks()].get();
    }

    template<typename scalar_t> const BLRTile<scalar_t>&
    BLRMatrix<scalar_t>::tile(std::size_t i, std::size_t j) const {
      return *blocks_[i+j*rowblocks()].get();
    }

    template<typename scalar_t> std::unique_ptr<BLRTile<scalar_t>>&
    BLRMatrix<scalar_t>::block(std::size_t i, std::size_t j) {
      return blocks_[i+j*rowblocks()];
    }

    template<typename scalar_t> DenseMatrixWrapper<scalar_t>
    BLRMatrix<scalar_t>::tile(DenseM_t& A, std::size_t i, std::size_t j) const {
      return DenseMW_t
        (tilerows(i), tilecols(j), A, tileroff(i), tilecoff(j));
    }

    template<typename scalar_t> DenseTile<scalar_t>&
    BLRMatrix<scalar_t>::tile_dense(std::size_t i, std::size_t j) {
      assert(dynamic_cast<DenseTile<scalar_t>*>
             (blocks_[i+j*rowblocks()].get()));
      return *static_cast<DenseTile<scalar_t>*>
        (blocks_[i+j*rowblocks()].get());
    }

    template<typename scalar_t> const DenseTile<scalar_t>&
    BLRMatrix<scalar_t>::tile_dense(std::size_t i, std::size_t j) const {
      assert(dynamic_cast<DenseTile<scalar_t>*>
             (blocks_[i+j*rowblocks()].get()));
      return *static_cast<DenseTile<scalar_t>*>
        (blocks_[i+j*rowblocks()].get());
    }

    template<typename scalar_t> void BLRMatrix<scalar_t>::create_dense_tile
    (std::size_t i, std::size_t j, DenseM_t& A) {
      block(i, j) = std::unique_ptr<DenseTile<scalar_t>>
        (new DenseTile<scalar_t>(tile(A, i, j)));
    }

    template<typename scalar_t> void BLRMatrix<scalar_t>::create_dense_tile
    (std::size_t i, std::size_t j, const extract_t<scalar_t>& Aelem) {
      auto m = tilerows(i);
      auto n = tilecols(j);
      block(i, j) = std::unique_ptr<DenseTile<scalar_t>>
        (new DenseTile<scalar_t>(m, n));
      std::vector<std::size_t> ii(m), jj(n);
      std::iota(ii.begin(), ii.end(), tileroff(i));
      std::iota(jj.begin(), jj.end(), tilecoff(j));
      Aelem(ii, jj, tile(i, j).D());
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::create_dense_tile_left_looking
    (std::size_t i, std::size_t j, const extract_t<scalar_t>& Aelem) {
      create_dense_tile(i, j, Aelem);
      for (std::size_t k=0; k<std::min(i, j); k++)
        gemm(Trans::N, Trans::N, scalar_t(-1.), tile(i, k), tile(k, j),
             scalar_t(1.), tile(i, j).D());
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::create_dense_tile_left_looking
    (std::size_t i, std::size_t j, std::size_t k,
     const extract_t<scalar_t>& Aelem, const BLRMatrix<scalar_t>& B21,
     const BLRMatrix<scalar_t>& B12) {
      create_dense_tile(i, j, Aelem);
      for (std::size_t l=0; l<k; l++)
        gemm(Trans::N, Trans::N, scalar_t(-1.), B21.tile(i, l),
             B12.tile(l, j), scalar_t(1.), tile(i, j).D());
    }

    template<typename scalar_t> void BLRMatrix<scalar_t>::create_LR_tile
    (std::size_t i, std::size_t j, DenseM_t& A, const Opts_t& opts) {
      block(i, j) = std::unique_ptr<LRTile<scalar_t>>
        (new LRTile<scalar_t>(tile(A, i, j), opts));
      auto& t = tile(i, j);
      if (t.rank()*(t.rows() + t.cols()) > t.rows()*t.cols())
        create_dense_tile(i, j, A);
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::create_LR_tile_left_looking
    (std::size_t i, std::size_t j, const extract_t<scalar_t>& Aelem,
     const Opts_t& opts) {
      create_LR_tile_left_looking
        (i, j, std::min(i,j), Aelem, *this, *this, opts);
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::create_LR_tile_left_looking
    (std::size_t i, std::size_t j, std::size_t k,
     const extract_t<scalar_t>& Aelem, const BLRMatrix<scalar_t>& B21,
     const BLRMatrix<scalar_t>& B12, const Opts_t& opts) {
      auto m = tilerows(i);  auto n = tilecols(j);
      auto dm = tileroff(i); auto dn = tilecoff(j);
      std::vector<std::size_t> lI(m), lJ(n), idx(1);
      std::iota(lJ.begin(), lJ.end(), dn);
      std::iota(lI.begin(), lI.end(), dm);
      if (opts.low_rank_algorithm() == LowRankAlgorithm::BACA) {
        std::size_t lwork = 0;
        for (std::size_t l=0; l<k; l++) {
          auto kk = B21.tilecols(l); // == B12.tilerows(l)
          lwork = std::max
            (lwork, 2*std::max(std::min(m,kk), std::min(n,kk))) + kk;
        }
        auto d = opts.BACA_blocksize();
        std::unique_ptr<scalar_t[]> work_(new scalar_t[lwork*d]);
        auto work = work_.get();
        std::vector<std::size_t> idx;
        idx.reserve(d);
        auto Arow = [&](const std::vector<std::size_t>& rows,
                        DenseMatrix<scalar_t>& c) {
          assert(rows.size() == c.rows() && c.cols() == n);
          idx.resize(rows.size());
          std::transform(rows.begin(), rows.end(), idx.begin(),
                         [&dm](std::size_t rr){ return rr+dm; });
          Aelem(idx, lJ, c);
          for (std::size_t l=0; l<k; l++)
            Schur_update_rows(rows, B21.tile(i, l), B12.tile(l, j), c, work);
        };
        auto Acol = [&](const std::vector<std::size_t>& cols,
                        DenseMatrix<scalar_t>& c) {
          assert(cols.size() == c.cols() && c.rows() == m);
          idx.resize(cols.size());
          std::transform(cols.begin(), cols.end(), idx.begin(),
                         [&dn](std::size_t cc){ return cc+dn; });
          Aelem(lI, idx, c);
          for (std::size_t l=0; l<k; l++)
            Schur_update_cols(cols, B21.tile(i, l), B12.tile(l, j), c, work);
        };
        block(i, j) = std::unique_ptr<LRTile<scalar_t>>
          (new LRTile<scalar_t>(m, n, Arow, Acol, opts));
      } else {
        std::size_t lwork = 0;
        for (std::size_t l=0; l<k; l++) {
          auto kk = B21.tilecols(l); // == B12.tilerows(l)
          lwork = std::max
            (lwork, std::max(std::min(m,kk), std::min(n,kk))) + kk;
        }
        std::unique_ptr<scalar_t[]> work_(new scalar_t[lwork]);
        auto work = work_.get();
        std::vector<std::size_t> idx(1);
        auto Arow = [&](std::size_t row, scalar_t* c) {
          idx[0] = dm + row;
          DenseMW_t cr(1, n, c, 1);
          Aelem(idx, lJ, cr);
          for (std::size_t l=0; l<k; l++)
            Schur_update_row
              (row, B21.tile(i, l), B12.tile(l, j), c, work);
        };
        auto Acol = [&](std::size_t col, scalar_t* c) {
          idx[0] = dn + col;
          DenseMW_t cc(m, 1, c, m);
          Aelem(lI, idx, cc);
          for (std::size_t l=0; l<k; l++)
            Schur_update_col
              (col, B21.tile(i, l), B12.tile(l, j), c, work);
        };
        block(i, j) = std::unique_ptr<LRTile<scalar_t>>
          (new LRTile<scalar_t>(m, n, Arow, Acol, opts));
      }
      auto& t = tile(i, j);
      if (t.rank()*(m + n) > m*n)
        create_dense_tile_left_looking(i, j, k, Aelem, B21, B12);
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::compress_tile
    (std::size_t i, std::size_t j, const Opts_t& opts) {
      auto t = tile(i, j).compress(opts);
      if (t->rank()*(t->rows() + t->cols()) < t->rows()*t->cols())
        block(i, j) = std::move(t);
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::fill(scalar_t v) {
      for (std::size_t i=0; i<nbrows_; i++)
        for (std::size_t j=0; j<nbcols_; j++) {
          block(i, j).reset
            (new DenseTile<scalar_t>(tilerows(i), tilecols(j)));
          block(i, j)->D().fill(v);
        }
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::fill_col
    (scalar_t v, std::size_t k, std::size_t CP) {
      std::size_t j_end = std::min(k + CP, colblocks());
      for (std::size_t i=0; i<nbrows_; i++)
        for (std::size_t j=k; j<j_end; j++) {
          block(i, j).reset
            (new DenseTile<scalar_t>(tilerows(i), tilecols(j)));
          block(i, j)->D().fill(v);
        }
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::construct_and_partial_factor
    (DenseMatrix<scalar_t>& A11, DenseMatrix<scalar_t>& A12,
     DenseMatrix<scalar_t>& A21, DenseMatrix<scalar_t>& A22,
     BLRMatrix<scalar_t>& B11, BLRMatrix<scalar_t>& B12,
     BLRMatrix<scalar_t>& B21,
     const std::vector<std::size_t>& tiles1,
     const std::vector<std::size_t>& tiles2,
     const DenseMatrix<bool>& admissible,
     const Opts_t& opts) {
      B11 = BLRMatrix<scalar_t>(A11.rows(), tiles1, A11.cols(), tiles1);
      B12 = BLRMatrix<scalar_t>(A12.rows(), tiles1, A12.cols(), tiles2);
      B21 = BLRMatrix<scalar_t>(A21.rows(), tiles2, A21.cols(), tiles1);
      B11.piv_.resize(B11.rows());
      auto rb = B11.rowblocks();
      auto rb2 = B21.rowblocks();
      //#pragma omp parallel if(!omp_in_parallel())
      //#pragma omp single nowait
      {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
        auto lrb = rb+rb2;
        // dummy for task synchronization
        std::unique_ptr<int[]> B_(new int[lrb*lrb]()); auto B = B_.get();
#pragma omp taskgroup
#else
        int* B = nullptr;
#endif
        {
          for (std::size_t i=0; i<rb; i++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
            std::size_t ii = i+lrb*i;
#pragma omp task default(shared) firstprivate(i,ii) depend(inout:B[ii])
#endif
            {
              B11.create_dense_tile(i, i, A11);
              auto tpiv = B11.tile(i, i).LU();
              std::copy(tpiv.begin(), tpiv.end(),
                        B11.piv_.begin()+B11.tileroff(i));
            }
            for (std::size_t j=i+1; j<rb; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ij = i+lrb*j;
#pragma omp task default(shared) firstprivate(i,j,ij,ii)        \
  depend(in:B[ii]) depend(inout:B[ij]) priority(rb-j)
#endif
              { // these blocks have received all updates, compress now
                if (admissible(i, j)) B11.create_LR_tile(i, j, A11, opts);
                else B11.create_dense_tile(i, j, A11);
                // permute and solve with L, blocks right from the
                // diagonal block
                std::vector<int> tpiv
                  (B11.piv_.begin()+B11.tileroff(i),
                   B11.piv_.begin()+B11.tileroff(i+1));
                B11.tile(i, j).laswp(tpiv, true);
                trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                     scalar_t(1.), B11.tile(i, i), B11.tile(i, j));
              }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ji = j+lrb*i;
#pragma omp task default(shared) firstprivate(i,j,ji,ii)        \
  depend(in:B[ii]) depend(inout:B[ji]) priority(rb-j)
#endif
              {
                if (admissible(j, i)) B11.create_LR_tile(j, i, A11, opts);
                else B11.create_dense_tile(j, i, A11);
                // solve with U, the blocks under the diagonal block
                trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                     scalar_t(1.), B11.tile(i, i), B11.tile(j, i));
              }
            }
            for (std::size_t j=0; j<rb2; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ij2 = i+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(i,j,ij2,ii)       \
  depend(in:B[ii]) depend(inout:B[ij2])
#endif
              {
                B12.create_LR_tile(i, j, A12, opts);
                // permute and solve with L blocks right from the
                // diagonal block
                std::vector<int> tpiv
                  (B11.piv_.begin()+B11.tileroff(i),
                   B11.piv_.begin()+B11.tileroff(i+1));
                B12.tile(i, j).laswp(tpiv, true);
                trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                     scalar_t(1.), B11.tile(i, i), B12.tile(i, j));
              }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t j2i = (rb+j)+lrb*i;
#pragma omp task default(shared) firstprivate(i,j,j2i,ii)       \
  depend(in:B[ii]) depend(inout:B[j2i])
#endif
              {
                B21.create_LR_tile(j, i, A21, opts);
                // solve with U, the blocks under the diagonal block
                trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                     scalar_t(1.), B11.tile(i, i), B21.tile(j, i));
              }
            }
            if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::RL) {
              for (std::size_t j=i+1; j<rb; j++) {
                for (std::size_t k=i+1; k<rb; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t ij = i+lrb*j, ki = k+lrb*i, kj = k+lrb*j;
#pragma omp task default(shared) firstprivate(i,j,k,ij,ki,kj)   \
  depend(in:B[ij],B[ki]) depend(inout:B[kj]) priority(rb-j)
#endif
                  { // Schur complement updates, always into full rank
                    auto Akj = B11.tile(A11, k, j);
                    gemm(Trans::N, Trans::N, scalar_t(-1.),
                         B11.tile(k, i), B11.tile(i, j), scalar_t(1.), Akj);
                  }
                }
              }
              for (std::size_t k=i+1; k<rb; k++) {
                for (std::size_t j=0; j<rb2; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t ki = k+lrb*i, ij2 = i+lrb*(rb+j),
                    kj2 = k+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(i,k,j,ki,ij2,kj2) \
  depend(in:B[ki],B[ij2]) depend(inout:B[kj2])
#endif
                  { // Schur complement updates, always into full rank
                    auto Akj = B12.tile(A12, k, j);
                    gemm(Trans::N, Trans::N, scalar_t(-1.),
                         B11.tile(k, i), B12.tile(i, j), scalar_t(1.), Akj);
                  }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t ik = i+lrb*k, j2i = (j+rb)+lrb*i,
                    j2k = (rb+j)+lrb*k;
#pragma omp task default(shared) firstprivate(i,k,j,ik,j2i,j2k) \
  depend(in:B[ik],B[j2i]) depend(inout:B[j2k])
#endif
                  { // Schur complement updates, always into full rank
                    auto Ajk = B21.tile(A21, j, k);
                    gemm(Trans::N, Trans::N, scalar_t(-1.),
                         B21.tile(j, i), B11.tile(i, k), scalar_t(1.), Ajk);
                  }
                }
              }
              for (std::size_t j=0; j<rb2; j++) {
                for (std::size_t k=0; k<rb2; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t ij2 = i+lrb*(rb+j), k2i = (rb+k)+lrb*i,
                    k2j2 = (rb+k)+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(i,j,k,ij2,k2i,k2j2)       \
  depend(in:B[ij2],B[k2i]) depend(inout:B[k2j2])
#endif
                  { // Schur complement updates, always into full rank
                    DenseMatrixWrapper<scalar_t> Akj
                      (B21.tilerows(k), B12.tilecols(j), A22,
                       B21.tileroff(k), B12.tilecoff(j));
                    gemm(Trans::N, Trans::N, scalar_t(-1.),
                         B21.tile(k, i), B12.tile(i, j), scalar_t(1.), Akj);
                  }
                }
              }
            } else if (opts.BLR_factor_algorithm() ==
                       BLRFactorAlgorithm::LL) {
              for (std::size_t j=i+1; j<rb; j++) {
                for (std::size_t k=0; k<i+1; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t ij = (i+1)+lrb*j, ik = (i+1)+lrb*k,
                    kj = k+lrb*j;
#pragma omp task default(shared) firstprivate(i,j,k,ij,ik,kj)   \
  depend(in:B[ik],B[kj]) depend(inout:B[ij]) priority(rb-j)
#endif
                  { // Schur complement updates, always into full rank
                    auto Aij = B11.tile(A11, i+1, j);
                    gemm(Trans::N, Trans::N, scalar_t(-1.),
                         B11.tile(i+1, k), B11.tile(k, j), scalar_t(1.), Aij);
                  }
                  if (j != i+1) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                    std::size_t ji = j+lrb*(i+1), jk = j+lrb*k,
                      ki = k+lrb*(i+1);
#pragma omp task default(shared) firstprivate(i,j,k,ji,jk,ki)   \
  depend(in:B[jk],B[ki]) depend(inout:B[ji]) priority(rb-j)
#endif
                    { // Schur complement updates, always into full rank
                      auto Aji = B11.tile(A11, j, i+1);
                      gemm(Trans::N, Trans::N, scalar_t(-1.),
                           B11.tile(j, k), B11.tile(k, i+1),
                           scalar_t(1.), Aji);
                    }
                  }
                }
              }
              if (i+1 < rb) {
                for (std::size_t j=0; j<rb2; j++) {
                  for (std::size_t k=0; k<i+1; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                    std::size_t ik = (i+1)+lrb*k, ij2 = (i+1)+lrb*(rb+j),
                      kj2 = k+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(i,k,j,ik,ij2,kj2) \
  depend(in:B[ik],B[kj2]) depend(inout:B[ij2])
#endif
                    { // Schur complement updates, always into full rank
                      auto Aij = B12.tile(A12, i+1, j);
                      gemm(Trans::N, Trans::N, scalar_t(-1.),
                           B11.tile(i+1, k), B12.tile(k, j),
                           scalar_t(1.), Aij);
                    }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                    std::size_t ki = k+lrb*(i+1), j2i = (j+rb)+lrb*(i+1),
                      j2k = (rb+j)+lrb*k;
#pragma omp task default(shared) firstprivate(i,k,j,ki,j2i,j2k) \
  depend(in:B[j2k],B[ki]) depend(inout:B[j2i])
#endif
                    { // Schur complement updates, always into full rank
                      auto Aji = B21.tile(A21, j, i+1);
                      gemm(Trans::N, Trans::N, scalar_t(-1.),
                           B21.tile(j, k), B11.tile(k, i+1),
                           scalar_t(1.), Aji);
                    }
                  }
                }
              }
            } else { // Comb or Star (LUAR)
              for (std::size_t j=i+1; j<rb; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t ij = (i+1)+lrb*j, i1j=ij-lrb*(j-i), ij1=ij-1;
#pragma omp task default(shared) firstprivate(i,j,ij,i1j,ij1)   \
  depend(in:B[i1j],B[ij1]) depend(inout:B[ij])
#endif
                B11.LUAR_B11(i+1, j, i+1, A11, opts, B);
                if (j != i+1) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t ji = j+lrb*(i+1), i1j = ji-lrb, ij1 = ji-j+i;
#pragma omp task default(shared) firstprivate(i,j,ji,i1j,ij1)   \
  depend(in:B[i1j],B[ij1]) depend(inout:B[ji])
#endif
                  B11.LUAR_B11(j, i+1, i+1, A11, opts, B);
                }
              }
              if (i+1 < rb) {
                for (std::size_t j=0; j<rb2; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t ij2 = (i+1)+lrb*(rb+j), i1j = ij2-lrb*(rb+j-i),
                    ij1 = ij2-1;
#pragma omp task default(shared) firstprivate(i,j,ij2,i1j,ij1)  \
  depend(in:B[i1j],B[ij1]) depend(inout:B[ij2])
#endif
                  B12.LUAR_B12(i+1, j, i+1, B11, A12, opts, B);
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t j2i = (rb+j)+lrb*(i+1), ji1 = j2i-lrb,
                    j1i = j2i-(rb-i)-j;
#pragma omp task default(shared) firstprivate(i,j,j2i,ji1,j1i)  \
  depend(in:B[ji1],B[j1i]) depend(inout:B[j2i])
#endif
                  B21.LUAR_B21(i+1, j, i+1, B11, A21, opts, B);
                }
              }
            }
          }
          if (opts.BLR_factor_algorithm() != BLRFactorAlgorithm::RL) {
            for (std::size_t i=0; i<rb2; i++) {
              for (std::size_t j=0; j<rb2; j++) {
                if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::LL) {
                  for (std::size_t k=0; k<rb; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                    std::size_t i2j2 = (rb+i)+lrb*(rb+j), i2k = (rb+i)+lrb*k,
                      kj2 = k+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(i,j,k,i2k,kj2,i2j2)       \
  depend(in:B[i2k],B[kj2]) depend(inout:B[i2j2])
#endif
                    { // Schur complement updates, always into full rank
                      DenseMatrixWrapper<scalar_t> Aij
                        (B21.tilerows(i), B12.tilecols(j), A22,
                         B21.tileroff(i), B12.tilecoff(j));
                      gemm(Trans::N, Trans::N, scalar_t(-1.),
                           B21.tile(i, k), B12.tile(k, j), scalar_t(1.), Aij);
                    }
                  }
                } else { //Comb or Star (LUAR-Update)
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t i2j2 = (rb+i)+lrb*(rb+j), i1j = rb*lrb-(rb2-i),
                    ij1 = i2j2-(i+1);
#pragma omp task default(shared) firstprivate(i,j,i2j2,i1j,ij1) \
  depend(in:B[i1j],B[ij1]) depend(inout:B[i2j2])
#endif
                  LUAR_B22(i, j, rb, B12, B21, A22, opts, B);
                }
              }
            }
          }
        }
      }
      for (std::size_t i=0; i<rb; i++)
        for (std::size_t l=B11.tileroff(i); l<B11.tileroff(i+1); l++)
          B11.piv_[l] += B11.tileroff(i);
      A11.clear();
      A12.clear();
      A21.clear();
    }

    template<typename scalar_t> void
    LUAR(const std::vector<BLRTile<scalar_t>*>& Ti,
         const std::vector<BLRTile<scalar_t>*>& Tj,
         DenseMatrixWrapper<scalar_t>& tij,
         const BLROptions<scalar_t>& opts, int* B) {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      auto kmax = Ti.size();
      if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::STAR) {
        std::size_t rank_sum = 0;
        for (std::size_t k=0; k<kmax; k++) {
          if (!(Ti[k]->is_low_rank() || Tj[k]->is_low_rank()))
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                 *Ti[k], *Tj[k], scalar_t(1.), tij);
          else if (Ti[k]->is_low_rank() && Tj[k]->is_low_rank())
            rank_sum += std::min(Ti[k]->rank(), Tj[k]->rank());
          else if (Ti[k]->is_low_rank())
            rank_sum += Ti[k]->rank();
          else
            rank_sum += Tj[k]->rank();
        }
        if (rank_sum > 0) {
          DenseM_t Uall(tij.rows(), rank_sum),
            Vall(rank_sum, tij.cols());
          std::size_t rank_tmp = 0;
          for (std::size_t k=0; k<kmax; k++) {
            if (Ti[k]->is_low_rank() || Tj[k]->is_low_rank()) {
              std::size_t minrank = 0;
              if (Ti[k]->is_low_rank() && Tj[k]->is_low_rank())
                minrank = std::min(Ti[k]->rank(), Tj[k]->rank());
              else if (Ti[k]->is_low_rank())
                minrank = Ti[k]->rank();
              else if (Tj[k]->is_low_rank())
                minrank = Tj[k]->rank();
              DenseMW_t t1(tij.rows(), minrank, Uall, 0, rank_tmp),
                t2(minrank, tij.cols(), Vall, rank_tmp, 0);
              Ti[k]->multiply(*Tj[k], t1, t2);
              rank_tmp += minrank;
            }
          }
          if (opts.compression_kernel() == CompressionKernel::FULL) {
            // recompress Uall and Vall
            LRTile<scalar_t> Uall_lr(Uall, opts), Vall_lr(Vall, opts);
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                 Uall_lr, Vall_lr, scalar_t(1.), tij);
          } else { // recompress Uall OR Vall
            if (Uall.rows() > Vall.cols()) // (Uall * Vall_lr.U) *Vall_lr.V
              gemm(Trans::N, Trans::N, scalar_t(-1.), Uall,
                   LRTile<scalar_t>(Vall, opts), scalar_t(1.), tij,
                   params::task_recursion_cutoff_level);
            else // Uall_lr.U * (Uall_lr.V * Vall)
              gemm(Trans::N, Trans::N, scalar_t(-1.),
                   LRTile<scalar_t>(Uall, opts), Vall, scalar_t(1.), tij,
                   params::task_recursion_cutoff_level);
          }
        }
      } else if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::COMB) {
        std::vector<std::pair<std::size_t,std::size_t>> ranks_idx;
        std::size_t rank_sum = 0;
        for (std::size_t k=0; k<kmax; k++) {
          if (!(Ti[k]->is_low_rank() || Tj[k]->is_low_rank()))
            // both tiles are DenseTiles
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                 *Ti[k], *Tj[k], scalar_t(1.), tij);
          else if (Ti[k]->is_low_rank() && Tj[k]->is_low_rank()) {
            // both tiles are LR, collect size of LR matrices
            ranks_idx.emplace_back
              (std::min(Ti[k]->rank(), Tj[k]->rank()), k);
            rank_sum += std::min(Ti[k]->rank(), Tj[k]->rank());
          } else if (Ti[k]->is_low_rank()) { // collect size of LR matrix
            ranks_idx.emplace_back(Ti[k]->rank(), k);
            rank_sum += Ti[k]->rank();
          } else { // collect size of LR matrix
            ranks_idx.emplace_back(Tj[k]->rank(), k);
            rank_sum += Tj[k]->rank();
          }
        }
        if (rank_sum > 0) {
          if (ranks_idx.size() > 1) {
            std::sort(ranks_idx.begin(), ranks_idx.end());
            DenseM_t tmpU(tij.rows(), rank_sum), tmpV(rank_sum, tij.cols());
            auto rk = ranks_idx[0].first;
            auto ki = ranks_idx[0].second;
            auto rank_tmp = rk;
            DenseMW_t t1(tij.rows(), rank_tmp, tmpU, 0, 0),
              t2(rank_tmp, tij.cols(), tmpV, 0, 0);
            Ti[ki]->multiply(*Tj[ki], t1, t2);
            for (std::size_t k=1; k<ranks_idx.size(); k++) {
              rk = ranks_idx[k].first;
              ki = ranks_idx[k].second;
              DenseMW_t t1(tij.rows(), rk, tmpU, 0, rank_tmp),
                t2(rk, tij.cols(), tmpV, rank_tmp, 0);
              Ti[ki]->multiply(*Tj[ki], t1, t2);
              DenseMW_t Uall(tij.rows(), rank_tmp+rk, tmpU, 0, 0),
                Vall(rank_tmp+rk, tij.cols(), tmpV, 0, 0);
              if (opts.compression_kernel() == CompressionKernel::FULL) {
                if (!(rank_tmp+rk == 0)){
                  // recompress Uall and Vall
                  LRTile<scalar_t> Uall_lr(Uall, opts), Vall_lr(Vall, opts);
                  if (k == ranks_idx.size() - 1)
                    gemm(Trans::N, Trans::N, scalar_t(-1.), Uall_lr, Vall_lr,
                        scalar_t(1.), tij);
                  else {
                    rank_tmp = std::min(Uall_lr.rank(), Vall_lr.rank());
                    DenseMW_t t1(tmpU.rows(), rank_tmp, tmpU, 0, 0),
                      t2(rank_tmp, tmpV.cols(), tmpV, 0, 0);
                    Uall_lr.multiply(Vall_lr, t1, t2);
                  }
                } else
                  rank_tmp = 0;
              } else { // recompress Uall OR Vall
                if (tij.rows() > tij.cols()) { // (Uall * Vall_lr.U) *Vall_lr.V
                  LRTile<scalar_t> Vall_lr(Vall, opts);
                  if (k == ranks_idx.size() - 1)
                    gemm(Trans::N, Trans::N, scalar_t(-1.),
                         Uall, Vall_lr, scalar_t(1.), tij,
                         params::task_recursion_cutoff_level);
                  else {
                    rank_tmp = Vall_lr.rank();
                    DenseM_t t1(Uall.rows(), rank_tmp);
                    gemm(Trans::N, Trans::N, scalar_t(1.),
                         Uall, Vall_lr.U(), scalar_t(0.), t1);
                    copy(Vall_lr.V(), tmpV, 0, 0);
                    copy(t1, tmpU, 0, 0);
                  }
                } else { // Uall_lr.U * (Uall_lr.V * Vall)
                  LRTile<scalar_t> Uall_lr(Uall, opts);
                  if (k == ranks_idx.size() - 1)
                    gemm(Trans::N, Trans::N, scalar_t(-1.),
                         Uall_lr, Vall, scalar_t(1.), tij,
                         params::task_recursion_cutoff_level);
                  else {
                    rank_tmp = Uall_lr.rank();
                    DenseM_t t2(rank_tmp, tij.cols());
                    gemm(Trans::N, Trans::N, scalar_t(1.), Uall_lr.V(), Vall,
                         scalar_t(0.), t2);
                    copy(Uall_lr.U(), tmpU, 0, 0);
                    copy(t2, tmpV, 0, 0);
                  }
                }
              }
            }
          } else
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                 *Ti[ranks_idx[0].second], *Tj[ranks_idx[0].second],
                 scalar_t(1.), tij);
        }
      }
    }


    template<typename scalar_t> void
    BLRMatrix<scalar_t>::LUAR_B11(std::size_t i, std::size_t j,
                                  std::size_t kmax, DenseM_t& A11,
                                  const Opts_t& opts, int* B) {
      std::vector<BLRTile<scalar_t>*> Ti(kmax), Tj(kmax);
      for (std::size_t k=0; k<kmax; k++) {
        Ti[k] = &tile(i, k);
        Tj[k] = &tile(k, j);
      }
      auto Aij = tile(A11, i, j);
      LUAR(Ti, Tj, Aij, opts, B);
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::LUAR_B12(std::size_t i, std::size_t j,
                                  std::size_t kmax, BLRMatrix<scalar_t>& B11,
                                  DenseM_t& A12, const Opts_t& opts, int* B) {
      std::vector<BLRTile<scalar_t>*> Ti(kmax), Tj(kmax);
      for (std::size_t k=0; k<kmax; k++) {
        Ti[k] = &B11.tile(i, k);
        Tj[k] = &tile(k, j);
      }
      auto Aij = tile(A12, i, j);
      LUAR(Ti, Tj, Aij, opts, B);
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::LUAR_B21(std::size_t i, std::size_t j,
                                  std::size_t kmax, BLRMatrix<scalar_t>& B11,
                                  DenseM_t& A21, const Opts_t& opts, int* B) {
      std::vector<BLRTile<scalar_t>*> Ti(kmax), Tj(kmax);
      for (std::size_t k=0; k<kmax; k++) {
        Ti[k] = &tile(j, k);
        Tj[k] = &B11.tile(k, i);
      }
      auto Aij = tile(A21, j, i);
      LUAR(Ti, Tj, Aij, opts, B);
    }

    template<typename scalar_t> void
    LUAR_B22(std::size_t i, std::size_t j, std::size_t kmax,
             BLRMatrix<scalar_t>& B12, BLRMatrix<scalar_t>& B21,
             DenseMatrix<scalar_t>& A22, const BLROptions<scalar_t>& opts,
             int* B) {
      DenseMatrixWrapper<scalar_t> Aij
        (B21.tilerows(i), B12.tilecols(j), A22,
         B21.tileroff(i), B12.tilecoff(j));
      std::vector<BLRTile<scalar_t>*> Ti(kmax), Tj(kmax);
      for (std::size_t k=0; k<kmax; k++) {
        Ti[k] = &B21.tile(i, k);
        Tj[k] = &B12.tile(k, j);
      }
      LUAR(Ti, Tj, Aij, opts, B);
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::construct_and_partial_factor
    (BLRMatrix<scalar_t>& B11, BLRMatrix<scalar_t>& B12,
     BLRMatrix<scalar_t>& B21, BLRMatrix<scalar_t>& B22,
     const std::vector<std::size_t>& tiles1,
     const std::vector<std::size_t>& tiles2,
     const DenseMatrix<bool>& admissible,
     const Opts_t& opts) {
      B11.piv_.resize(B11.rows());
      auto rb = B11.rowblocks();
      auto rb2 = B21.rowblocks();
//#pragma omp parallel if(!omp_in_parallel())
//#pragma omp single nowait
      {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
        auto lrb = rb+rb2;
        // dummy for task synchronization
        std::unique_ptr<int[]> B_(new int[lrb*lrb]()); auto B = B_.get();
#pragma omp taskgroup
#endif
        {
          for (std::size_t i=0; i<rb; i++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
            std::size_t ii = i+lrb*i;
#pragma omp task default(shared) firstprivate(i,ii) depend(inout:B[ii])
#endif
            {
              auto tpiv = B11.tile(i, i).LU();
              std::copy(tpiv.begin(), tpiv.end(),
                        B11.piv_.begin()+B11.tileroff(i));
            }
            for (std::size_t j=i+1; j<rb; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ij = i+lrb*j;
#pragma omp task default(shared) firstprivate(i,j,ij,ii)        \
  depend(in:B[ii]) depend(inout:B[ij]) priority(rb-j)
#endif
              {
                if (admissible(i, j)) B11.compress_tile(i, j, opts);
                std::vector<int> tpiv
                  (B11.piv_.begin()+B11.tileroff(i),
                   B11.piv_.begin()+B11.tileroff(i+1));
                B11.tile(i, j).laswp(tpiv, true);
                trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                    scalar_t(1.), B11.tile(i, i), B11.tile(i, j));
              }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ji = j+lrb*i;
#pragma omp task default(shared) firstprivate(i,j,ji,ii)        \
  depend(in:B[ii]) depend(inout:B[ji]) priority(rb-j)
#endif
              {
                if (admissible(j, i)) B11.compress_tile(j, i, opts);
                trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                    scalar_t(1.), B11.tile(i, i), B11.tile(j, i));
              }
            }
            for (std::size_t j=0; j<rb2; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ij2 = i+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(i,j,ij2,ii)       \
  depend(in:B[ii]) depend(inout:B[ij2])
#endif
              {
                B12.compress_tile(i, j, opts);
                std::vector<int> tpiv
                  (B11.piv_.begin()+B11.tileroff(i),
                   B11.piv_.begin()+B11.tileroff(i+1));
                B12.tile(i, j).laswp(tpiv, true);
                trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                    scalar_t(1.), B11.tile(i, i), B12.tile(i, j));
              }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t j2i = (rb+j)+lrb*i;
#pragma omp task default(shared) firstprivate(i,j,j2i,ii)       \
  depend(in:B[ii]) depend(inout:B[j2i])
#endif
              B21.compress_tile(j, i, opts);
              trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                   scalar_t(1.), B11.tile(i, i), B21.tile(j, i));
            }
            for (std::size_t j=i+1; j<rb; j++) {
              for (std::size_t k=i+1; k<rb; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t ij = i+lrb*j, ki = k+lrb*i, kj = k+lrb*j;
#pragma omp task default(shared) firstprivate(i,j,k,ij,ki,kj)   \
  depend(in:B[ij],B[ki]) depend(inout:B[kj]) priority(rb-j)
#endif
                gemm(Trans::N, Trans::N, scalar_t(-1.),
                     B11.tile(k, i), B11.tile(i, j), scalar_t(1.),
                     B11.tile_dense(k,j).D());
              }
            }
            for (std::size_t k=i+1; k<rb; k++) {
              for (std::size_t j=0; j<rb2; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t ki = k+lrb*i, ij2 = i+lrb*(rb+j), kj2 = k+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(i,k,j,ki,ij2,kj2) \
  depend(in:B[ki],B[ij2]) depend(inout:B[kj2])
#endif
                gemm(Trans::N, Trans::N, scalar_t(-1.),
                     B11.tile(k, i), B12.tile(i, j), scalar_t(1.),
                     B12.tile_dense(k,j).D());
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t ik = i+lrb*k, j2i = (j+rb)+lrb*i, j2k = (rb+j)+lrb*k;
#pragma omp task default(shared) firstprivate(i,k,j,ik,j2i,j2k) \
  depend(in:B[ik],B[j2i]) depend(inout:B[j2k])
#endif
                gemm(Trans::N, Trans::N, scalar_t(-1.),
                     B21.tile(j, i), B11.tile(i, k), scalar_t(1.),
                     B21.tile_dense(j,k).D());
              }
            }
            for (std::size_t j=0; j<rb2; j++) {
              for (std::size_t k=0; k<rb2; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t ij2 = i+lrb*(rb+j), k2i = (rb+k)+lrb*i,
                  k2j2 = (rb+k)+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(i,j,k,ij2,k2i,k2j2)       \
  depend(in:B[ij2],B[k2i]) depend(inout:B[k2j2])
#endif
                gemm(Trans::N, Trans::N, scalar_t(-1.),
                     B21.tile(k, i), B12.tile(i, j), scalar_t(1.),
                     B22.tile_dense(k,j).D());
              }
            }
          }
          for (std::size_t j=0; j<rb2; j++) {
            for (std::size_t k=0; k<rb2; k++) {
              if(j!=k){
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t k2j2 = (rb+k)+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(j,k,k2j2) \
  depend(inout:B[k2j2])
#endif
                B22.compress_tile(k, j, opts);
              }
            }
          }
        }
      }
      for (std::size_t i=0; i<rb; i++)
        for (std::size_t l=B11.tileroff(i); l<B11.tileroff(i+1); l++)
          B11.piv_[l] += B11.tileroff(i);
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::construct_and_partial_factor_col
    (BLRMatrix<scalar_t>& B11, BLRMatrix<scalar_t>& B12,
     BLRMatrix<scalar_t>& B21, BLRMatrix<scalar_t>& B22,
     const std::vector<std::size_t>& tiles1,
     const std::vector<std::size_t>& tiles2,
     const DenseMatrix<bool>& admissible, const Opts_t& opts,
     const std::function<void(int, bool, std::size_t)>& blockcol) {
      B11.piv_.resize(B11.rows());
      auto rb = B11.rowblocks();
      auto rb2 = B21.rowblocks();
      std::size_t CP = 1; // ??
      //#pragma omp parallel if(!omp_in_parallel())
      //#pragma omp single nowait
      {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
        auto lrb = rb+rb2;
        // dummy for task synchronization
        std::unique_ptr<int[]> B_(new int[lrb*lrb]()); auto B = B_.get();
#pragma omp taskgroup
#endif
        {
          for (std::size_t i=0; i<rb; i+=CP) { // F11 and F21
#pragma omp taskwait
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
            std::size_t ifirst = lrb*i;
#pragma omp task default(shared) firstprivate(i,ifirst) \
  depend(out:B[ifirst:lrb])
#endif
            {
              B11.fill_col(0., i, CP);
              B21.fill_col(0., i, CP);
              blockcol(i, true, CP);
            }
            for (std::size_t k=0; k<i; k++) {
              for (std::size_t j=i; j<std::min(i+CP, rb); j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t kj = k+lrb*j;
#pragma omp task default(shared) firstprivate(i,j,k,kj) \
  depend(inout:B[kj]) priority(rb-k)
#endif
                {
                  if (admissible(k, j)) B11.compress_tile(k, j, opts);
                  std::vector<int> tpiv
                    (B11.piv_.begin()+B11.tileroff(k),
                     B11.piv_.begin()+B11.tileroff(k+1));
                  B11.tile(k, j).laswp(tpiv, true);
                  trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                       scalar_t(1.), B11.tile(k, k), B11.tile(k, j));
                }
              }
              for (std::size_t lk=k+1; lk<rb; lk++) {
                for (std::size_t lj=i; lj<std::min(i+CP, rb); lj++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t klj = k+lrb*lj, lkk = lk+lrb*k,
                    lklj = lk+lrb*lj;
#pragma omp task default(shared) firstprivate(i,k,lk,lj,klj,lkk,lklj)   \
  depend(in:B[klj],B[lkk]) depend(inout:B[lklj])
#endif
                  gemm(Trans::N, Trans::N, scalar_t(-1.), B11.tile(lk, k),
                       B11.tile(k, lj), scalar_t(1.),
                       B11.tile_dense(lk, lj).D());
                }
              }
              for (std::size_t lk=0; lk<rb2; lk++) {
                for (std::size_t lj=i; lj<std::min(i+CP,rb); lj++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t klj = k+lrb*lj, lk2k = (lk+rb)+lrb*k,
                    lk2lj = (lk+rb)+lrb*lj;
#pragma omp task default(shared) firstprivate(i,k,lk,lj,klj,lk2k,lk2lj) \
  depend(in:B[klj],B[lk2k]) depend(inout:B[lk2lj])
#endif
                  gemm(Trans::N, Trans::N, scalar_t(-1.), B21.tile(lk, k),
                       B11.tile(k, lj), scalar_t(1.),
                       B21.tile_dense(lk, lj).D());
                }
              }
            }
            for (std::size_t c=i; c<std::min(i+CP,rb); c++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t cc = c+lrb*c;
#pragma omp task default(shared) firstprivate(i,c,cc)   \
  depend(inout:B[cc])
#endif
              {
                auto tpiv = B11.tile(c, c).LU();
                std::copy(tpiv.begin(), tpiv.end(),
                          B11.piv_.begin()+B11.tileroff(c));
              }
              for (std::size_t j=c+1; j<std::min(i+CP,rb); j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t cj = c+lrb*j;
#pragma omp task default(shared) firstprivate(i,c,j,cj,cc)      \
  depend(in:B[cc]) depend(inout:B[cj]) priority(rb-j)
#endif
                {
                  if (admissible(c, j)) B11.compress_tile(c, j, opts);
                  std::vector<int> tpiv
                    (B11.piv_.begin()+B11.tileroff(c),
                     B11.piv_.begin()+B11.tileroff(c+1));
                  B11.tile(c, j).laswp(tpiv, true);
                  trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                       scalar_t(1.), B11.tile(c, c), B11.tile(c, j));
                }
              }
              for (std::size_t j=c+1; j<rb; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t jc = j+lrb*c;
#pragma omp task default(shared) firstprivate(i,c,j,jc,cc)      \
  depend(in:B[cc]) depend(inout:B[jc]) priority(rb-j)
#endif
                {
                  if (admissible(j, c)) B11.compress_tile(j, c, opts);
                  trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                       scalar_t(1.), B11.tile(c, c), B11.tile(j, c));
                }
              }
              for (std::size_t j=0; j<rb2; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t j2c = (rb+j)+lrb*c;
#pragma omp task default(shared) firstprivate(i,c,j,j2c,cc)     \
  depend(in:B[cc]) depend(inout:B[j2c])
#endif
                {
                  B21.compress_tile(j, c, opts);
                  trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                       scalar_t(1.), B11.tile(c, c), B21.tile(j, c));
                }
              }
              for (std::size_t j=c+1; j<std::min(i+CP,rb); j++) {
                for (std::size_t k=c+1; k<rb; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t kc = k+lrb*c, cj = c+lrb*j, kj = k+lrb*j;
#pragma omp task default(shared) firstprivate(i,c,j,k,kc,cj,kj) \
  depend(in:B[kc],B[cj]) depend(inout:B[kj])
#endif
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       B11.tile(k, c), B11.tile(c, j), scalar_t(1.),
                       B11.tile_dense(k, j).D());
                }
              }
              for (std::size_t j=c+1; j<std::min(i+CP,rb); j++) {
                for (std::size_t k=0; k<rb2; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t k2c = (k+rb)+lrb*c, cj = c+lrb*j,
                    k2j = (k+rb)+lrb*j;
#pragma omp task default(shared) firstprivate(i,c,j,k,k2c,cj,k2j)       \
  depend(in:B[k2c],B[cj]) depend(inout:B[k2j])
#endif
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       B21.tile(k, c), B11.tile(c, j), scalar_t(1.),
                       B21.tile_dense(k,j).D());
                }
              }
            }
          }
          for (std::size_t i=0; i<rb2; i+=CP) { // F12 and F22
#pragma omp taskwait
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
            std::size_t ifirst = lrb*(i+rb);
#pragma omp task default(shared) firstprivate(i,ifirst) \
  depend(out:B[ifirst:lrb])
#endif
            {
              B12.fill_col(0., i, CP);
              B22.fill_col(0., i, CP);
              blockcol(i, false, CP);
            }
            for (std::size_t k=0; k<rb; k++) {
              for (std::size_t j=i; j<std::min(i+CP, rb2); j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                std::size_t kj2 = k+lrb*(j+rb);
#pragma omp task default(shared) firstprivate(i,k,j,kj2)        \
  depend(inout:B[kj2]) priority(rb-k)
#endif
                {
                  B12.compress_tile(k, j, opts);
                  std::vector<int> tpiv
                    (B11.piv_.begin()+B11.tileroff(k),
                     B11.piv_.begin()+B11.tileroff(k+1));
                  B12.tile(k, j).laswp(tpiv, true);
                  trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                       scalar_t(1.), B11.tile(k, k), B12.tile(k, j));
                }
              }
              for (std::size_t lk=k+1; lk<rb; lk++) {
                for (std::size_t lj=i; lj<std::min(i+CP, rb2); lj++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t klj2 = k+lrb*(lj+rb), lkk = lk+lrb*k,
                    lklj2 = lk+lrb*(lj+rb);
#pragma omp task default(shared) firstprivate(i,k,lk,lj,klj2,lkk,lklj2) \
  depend(in:B[klj2],B[lkk]) depend(inout:B[lklj2])
#endif
                  gemm(Trans::N, Trans::N, scalar_t(-1.), B11.tile(lk, k),
                       B12.tile(k, lj), scalar_t(1.),
                       B12.tile_dense(lk, lj).D());
                }
              }
              for (std::size_t lk=0; lk<rb2; lk++) {
                for (std::size_t lj=i; lj<std::min(i+CP,rb2); lj++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t klj2 = k+lrb*(lj+rb), lk2k = (lk+rb)+lrb*k,
                    lk2lj2 = (lk+rb)+lrb*(lj+rb);
#pragma omp task default(shared) firstprivate(i,k,lk,lj,klj2,lk2k,lk2lj2) \
  depend(in:B[klj2],B[lk2k]) depend(inout:B[lk2lj2])
#endif
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       B21.tile(lk, k), B12.tile(k, lj), scalar_t(1.),
                       B22.tile_dense(lk,lj).D());
                }
              }
            }
            for (std::size_t k=0; k<rb2; k++) {
              for (std::size_t j=i; j<std::min(i+CP, rb2); j++) {
                if (j != k) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
                  std::size_t k2j2 = (rb+k)+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(i,k,j,k2j2)       \
  depend(inout:B[k2j2])
#endif
                  B22.compress_tile(k, j, opts);
                }
              }
            }
          }
        }
      }
      for (std::size_t i=0; i<rb; i++)
        for (std::size_t l=B11.tileroff(i); l<B11.tileroff(i+1); l++)
          B11.piv_[l] += B11.tileroff(i);
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::construct_and_partial_factor
    (std::size_t n1, std::size_t n2,
     const extract_t<scalar_t>& A11, const extract_t<scalar_t>& A12,
     const extract_t<scalar_t>& A21, const extract_t<scalar_t>& A22,
     BLRMatrix<scalar_t>& B11, BLRMatrix<scalar_t>& B12,
     BLRMatrix<scalar_t>& B21, BLRMatrix<scalar_t>& B22,
     const std::vector<std::size_t>& tiles1,
     const std::vector<std::size_t>& tiles2,
     const DenseMatrix<bool>& admissible,
     const BLROptions<scalar_t>& opts) {
      B11 = BLRMatrix<scalar_t>(n1, tiles1, n1, tiles1);
      B12 = BLRMatrix<scalar_t>(n1, tiles1, n2, tiles2);
      B21 = BLRMatrix<scalar_t>(n2, tiles2, n1, tiles1);
      B22 = BLRMatrix<scalar_t>(n2, tiles2, n2, tiles2);
      B11.piv_.resize(B11.rows());
      auto rb = B11.rowblocks();
      auto rb2 = B21.rowblocks();
      for (std::size_t i=0; i<rb; i++) {
        B11.create_dense_tile_left_looking(i, i, A11);
        auto tpiv = B11.tile(i, i).LU();
        std::copy(tpiv.begin(), tpiv.end(), B11.piv_.begin()+B11.tileroff(i));
        for (std::size_t j=i+1; j<rb; j++) {
          // these blocks have received all updates, compress now
          if (admissible(i, j))
            B11.create_LR_tile_left_looking(i, j, A11, opts);
          else B11.create_dense_tile_left_looking(i, j, A11);
          // permute and solve with L, blocks right from the diagonal block
          std::vector<int> tpiv
            (B11.piv_.begin()+B11.tileroff(i),
             B11.piv_.begin()+B11.tileroff(i+1));
          B11.tile(i, j).laswp(tpiv, true);
          trsm(Side::L, UpLo::L, Trans::N, Diag::U,
               scalar_t(1.), B11.tile(i, i), B11.tile(i, j));
          if (admissible(j, i))
            B11.create_LR_tile_left_looking(j, i, A11, opts);
          else B11.create_dense_tile_left_looking(j, i, A11);
          // solve with U, the blocks under the diagonal block
          trsm(Side::R, UpLo::U, Trans::N, Diag::N,
               scalar_t(1.), B11.tile(i, i), B11.tile(j, i));
        }
        for (std::size_t j=0; j<rb2; j++) {
          B12.create_LR_tile_left_looking(i, j, i, A12, B11, B12, opts);
          // permute and solve with L  blocks right from the diagonal block
          std::vector<int> tpiv
            (B11.piv_.begin()+B11.tileroff(i),
             B11.piv_.begin()+B11.tileroff(i+1));
          B12.tile(i, j).laswp(tpiv, true);
          trsm(Side::L, UpLo::L, Trans::N, Diag::U,
               scalar_t(1.), B11.tile(i, i), B12.tile(i, j));
          B21.create_LR_tile_left_looking(j, i, i, A21, B21, B11, opts);
          // solve with U, the blocks under the diagonal block
          trsm(Side::R, UpLo::U, Trans::N, Diag::N,
               scalar_t(1.), B11.tile(i, i), B21.tile(j, i));
        }
      }
      for (std::size_t i=0; i<rb2; i++)
        for (std::size_t j=0; j<rb2; j++)
          if (i==j)
            B22.create_dense_tile_left_looking(i, j, rb, A22, B21, B12);
          else B22.create_LR_tile_left_looking(i, j, rb, A22, B21, B12, opts);
      for (std::size_t i=0; i<rb; i++)
        for (std::size_t l=B11.tileroff(i); l<B11.tileroff(i+1); l++)
          B11.piv_[l] += B11.tileroff(i);
    }


    template<typename scalar_t> void
    BLRMatrix<scalar_t>::trsmLNU_gemm
    (const BLRMatrix<scalar_t>& F1, const BLRMatrix<scalar_t>& F2,
     DenseMatrix<scalar_t>& B1, DenseMatrix<scalar_t>& B2, int task_depth) {
      using DMW_t = DenseMatrixWrapper<scalar_t>;
      if (B1.cols() == 1) {
        auto rb = F1.rowblocks();
        auto rb2 = F2.rowblocks();
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
        auto lrb = rb+rb2;
        std::unique_ptr<int[]> B_(new int[lrb]()); auto B = B_.get();
#pragma omp taskgroup
#endif
        {
          for (std::size_t i=0; i<rb; i++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
#pragma omp task default(shared) firstprivate(i) depend(inout:B[i])
#endif
            {
              DMW_t Bi(F1.tilerows(i), B1.cols(), B1, F1.tileroff(i), 0);
              trsv(UpLo::L, Trans::N, Diag::U, F1.tile(i, i).D(), Bi,
                   params::task_recursion_cutoff_level);
            }
            for (std::size_t j=i+1; j<rb; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
#pragma omp task default(shared) firstprivate(i,j)      \
  depend(in:B[i]) depend(inout:B[j]) priority(rb-i)
#endif
              {
                DMW_t Bi(F1.tilerows(i), B1.cols(), B1, F1.tileroff(i), 0);
                DMW_t Bj(F1.tilerows(j), B1.cols(), B1, F1.tileroff(j), 0);
                F1.tile(j, i).gemv_a(Trans::N, scalar_t(-1.), Bi, scalar_t(1.), Bj);
              }
            }
            for (std::size_t j=0; j<rb2; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t j2 = rb+j;
#pragma omp task default(shared) firstprivate(i,j,j2)   \
  depend(in:B[i]) depend(inout:B[j2]) priority(0)
#endif
              {
                DMW_t Bi(F1.tilerows(i), B1.cols(), B1, F1.tileroff(i), 0);
                DMW_t Bj(F2.tilerows(j), B2.cols(), B2, F2.tileroff(j), 0);
                F2.tile(j, i).gemv_a(Trans::N, scalar_t(-1.), Bi, scalar_t(1.), Bj);
              }
            }
          }
        }
      } else {
        // TODO optimize by merging
        trsm(Side::L, UpLo::L, Trans::N, Diag::U,
             scalar_t(1.), F1, B1, task_depth);
        gemm(Trans::N, Trans::N, scalar_t(-1.), F2, B1,
             scalar_t(1.), B2, task_depth);
      }
    }

    template<typename scalar_t> void BLRMatrix<scalar_t>::gemm_trsmUNN
    (const BLRMatrix<scalar_t>& F1, const BLRMatrix<scalar_t>& F2,
     DenseMatrix<scalar_t>& B1, DenseMatrix<scalar_t>& B2, int task_depth) {
      using DMW_t = DenseMatrixWrapper<scalar_t>;
      if (B1.cols() == 1) {
        auto rb = F1.colblocks();
        auto rb2 = F2.colblocks();
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
        auto lrb = rb+rb2;
        std::unique_ptr<int[]> B_(new int[lrb]()); auto B = B_.get();
#pragma omp taskgroup
#endif
        {
          for (std::size_t i=rb; i --> 0; ) {
            assert(i < rb);
            for (std::size_t j=0; j<rb2; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t j2 = rb+j;
#pragma omp task default(shared) firstprivate(i,j,j2)   \
  depend(in:B[j2]) depend(inout:B[i]) priority(1)
#endif
              {
                DMW_t Bi(F1.tilerows(i), B1.cols(), B1, F1.tileroff(i), 0);
                DMW_t Bj(F2.tilecols(j), B2.cols(), B2, F2.tilecoff(j), 0);
                F2.tile(i, j).gemv_a(Trans::N, scalar_t(-1.), Bj, scalar_t(1.), Bi);
              }
            }
            for (std::size_t j=i+1; j<rb; j++)
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
#pragma omp task default(shared) firstprivate(i,j)      \
  depend(in:B[j]) depend(inout:B[i]) priority(1)
#endif
              {
                DMW_t Bi(F1.tilerows(i), B1.cols(), B1, F1.tileroff(i), 0);
                DMW_t Bj(F1.tilecols(j), B1.cols(), B1, F1.tilecoff(j), 0);
                F1.tile(i, j).gemv_a(Trans::N, scalar_t(-1.), Bj, scalar_t(1.), Bi);
              }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
#pragma omp task default(shared) firstprivate(i) depend(inout:B[i]) priority(0)
#endif
            {
              DMW_t Bi(F1.tilerows(i), B1.cols(), B1, F1.tileroff(i), 0);
              trsv(UpLo::U, Trans::N, Diag::N, F1.tile(i, i).D(), Bi,
                   params::task_recursion_cutoff_level);
            }
          }
        }
      } else {
        // TODO optimize by merging
        gemm(Trans::N, Trans::N, scalar_t(-1.), F2, B2,
             scalar_t(1.), B1, task_depth);
        trsm(Side::L, UpLo::U, Trans::N, Diag::N,
             scalar_t(1.), F1, B1, task_depth);
      }
    }

    template<typename scalar_t> void
    trsm(Side s, UpLo ul, Trans ta, Diag d,
         scalar_t alpha, const BLRMatrix<scalar_t>& a,
         DenseMatrix<scalar_t>& b, int task_depth) {
      // TODO threading? is this used?
      //b.print();
      using DMW_t = DenseMatrixWrapper<scalar_t>;
      if (s == Side::R && ul == UpLo::U) {
        for (std::size_t j=0; j<a.colblocks(); j++) {
          DMW_t bj(b.rows(), a.tilerows(j), b, 0, a.tileroff(j));
          for (std::size_t k=0; k<j; k++)
            gemm(Trans::N, ta, scalar_t(-1.),
                 DMW_t(b.rows(), a.tilerows(k), b, 0, a.tileroff(k)),
                 ta==Trans::N ? a.tile(k, j) : a.tile(j, k),
                 scalar_t(1.), bj, task_depth);
          trsm(s, ul, ta, d, alpha, a.tile(j, j), bj, task_depth);
        }
      } else if (s == Side::L) {
        if (ul == UpLo::L) {
          for (std::size_t j=0; j<a.colblocks(); j++) {
            DMW_t bj(a.tilecols(j), b.cols(), b, a.tilecoff(j), 0);\
            for (std::size_t k=0; k<j; k++){
              gemm(ta, Trans::N, scalar_t(-1.),
                   ta==Trans::N ? a.tile(j, k) : a.tile(k, j),
                   DMW_t(a.tilecols(k), b.cols(), b, a.tilecoff(k), 0),
                   scalar_t(1.), bj, task_depth);
            }
            trsm(s, ul, ta, d, alpha, a.tile(j, j), bj, task_depth);
          }
        } else {
          for (int j=a.colblocks()-1; j>=0; j--) {
            DMW_t bj(a.tilecols(j), b.cols(), b, a.tilecoff(j), 0);
            for (std::size_t k=j+1; k<a.colblocks(); k++){
              gemm(ta, Trans::N, scalar_t(-1.),
                   ta==Trans::N ? a.tile(j, k) : a.tile(k, j),
                   DMW_t(a.tilecols(k), b.cols(), b, a.tilecoff(k), 0),
                   scalar_t(1.), bj, task_depth);
            }
            trsm(s, ul, ta, d, alpha, a.tile(j, j), bj, task_depth);
          }
        }
      } else { assert(false); }
    }

    template<typename scalar_t> void
    trsv(UpLo ul, Trans ta, Diag d, const BLRMatrix<scalar_t>& a,
         DenseMatrix<scalar_t>& b, int task_depth) {
      // TODO threading
      assert(b.cols() == 1);
      using DMW_t = DenseMatrixWrapper<scalar_t>;
      if (ul == UpLo::L) {
        for (std::size_t i=0; i<a.rowblocks(); i++) {
          DMW_t bi(a.tilecols(i), b.cols(), b, a.tilecoff(i), 0);
          for (std::size_t j=0; j<i; j++)
            (ta==Trans::N ? a.tile(i, j) : a.tile(j, i)).gemv_a
              (ta, scalar_t(-1.),
               DMW_t(a.tilecols(j), b.cols(), b, a.tilecoff(j), 0),
               scalar_t(1.), bi);
          trsv(ul, ta, d, a.tile(i, i).D(), bi,
               params::task_recursion_cutoff_level);
        }
      } else {
        for (int i=int(a.rowblocks())-1; i>=0; i--) {
          DMW_t bi(a.tilecols(i), b.cols(), b, a.tilecoff(i), 0);
          for (int j=int(a.colblocks())-1; j>i; j--)
            (ta==Trans::N ? a.tile(i, j) : a.tile(j, i)).gemv_a
              (ta, scalar_t(-1.),
               DMW_t(a.tilecols(j), b.cols(), b, a.tilecoff(j), 0),
               scalar_t(1.), bi);
          trsv(ul, ta, d, a.tile(i, i).D(), bi,
               params::task_recursion_cutoff_level);
        }
      }
    }

    template<typename scalar_t> void
    gemv(Trans ta, scalar_t alpha, const BLRMatrix<scalar_t>& a,
         const DenseMatrix<scalar_t>& x, scalar_t beta,
         DenseMatrix<scalar_t>& y, int task_depth) {
      assert(x.cols() == 1 && y.cols() == 1);
      using DMW_t = DenseMatrixWrapper<scalar_t>;
      const auto imax = ta == Trans::N ? a.rowblocks() : a.colblocks();
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)
#endif
      for (std::size_t i=0; i<imax; i++) {
        DMW_t yi(ta==Trans::N ? a.tilerows(i) : a.tilecols(i), y.cols(), y,
                 ta==Trans::N ? a.tileroff(i) : a.tilecoff(i), 0);
        for (std::size_t j=0; j<a.colblocks(); j++) {
          DMW_t xj(ta==Trans::N ? a.tilecols(j) : a.tilerows(j), x.cols(),
                   const_cast<DenseMatrix<scalar_t>&>(x),
                   ta==Trans::N ? a.tilecoff(j) : a.tileroff(j), 0);
          a.tile(i, j).gemv_a
            (ta, alpha, xj, j==0 ? beta : scalar_t(1.), yi);
        }
      }
    }


    template<typename scalar_t> void
    BLRMatrix<scalar_t>::mult(Trans op, const DenseMatrix<scalar_t>& x,
                              DenseMatrix<scalar_t>& y) const {
      gemm(op, Trans::N, scalar_t(1.), *this, x, scalar_t(0.), y, 0);
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const BLRMatrix<scalar_t>& a,
         const BLRMatrix<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c, int task_depth) {
      const auto imax = ta == Trans::N ? a.rowblocks() : a.colblocks();
      const auto jmax = tb == Trans::N ? b.colblocks() : b.rowblocks();
      const auto kmax = ta == Trans::N ? a.colblocks() : a.rowblocks();
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop collapse(2) default(shared)
#endif
      for (std::size_t i=0; i<imax; i++)
        for (std::size_t j=0; j<jmax; j++) {
          DenseMatrixWrapper<scalar_t> cij
            (ta==Trans::N ? a.tilerows(i) : a.tilecols(i),
             tb==Trans::N ? b.tilecols(j) : b.tilerows(j), c,
             ta==Trans::N ? a.tileroff(i) : a.tilecoff(i),
             tb==Trans::N ? b.tilecoff(j) : b.tileroff(j));
          for (std::size_t k=0; k<kmax; k++)
            gemm(ta, tb, alpha, ta==Trans::N ? a.tile(i, k) : a.tile(k, i),
                 tb==Trans::N ? b.tile(k, j) : b.tile(j, k),
                 k==0 ? beta : scalar_t(1.), cij);
        }
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const BLRMatrix<scalar_t>& A,
         const DenseMatrix<scalar_t>& B, scalar_t beta,
         DenseMatrix<scalar_t>& C, int task_depth) {
      using DMW_t = DenseMatrixWrapper<scalar_t>;
      const auto imax = ta == Trans::N ? A.rowblocks() : A.colblocks();
      const auto jmax = ta == Trans::N ? A.colblocks() : A.rowblocks();
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)
#endif
      for (std::size_t i=0; i<imax; i++) {
        DMW_t Ci(A.tilerows(i), C.cols(), C, A.tileroff(i), 0);
        for (std::size_t j=0; j<jmax; j++) {
          DMW_t Bj = ta == Trans::N  ?
            DMW_t(ta==Trans::N ? A.tilecols(j) : A.tilerows(j), B.cols(),
                  const_cast<DenseMatrix<scalar_t>&>(B),
                  ta==Trans::N ? A.tilecoff(j) : A.tileroff(j), 0) :
            DMW_t(B.cols(), ta==Trans::N ? A.tilecols(j) : A.tilerows(j),
                  const_cast<DenseMatrix<scalar_t>&>(B),
                  0, ta==Trans::N ? A.tilecoff(j) : A.tileroff(j));
          A.tile(i, j).gemm_a
            (ta, tb, alpha, Bj, j==0 ? beta : scalar_t(1.), Ci, 0);
        }
      }
    }


    // explicit template instantiations
    template class BLRMatrix<float>;
    template class BLRMatrix<double>;
    template class BLRMatrix<std::complex<float>>;
    template class BLRMatrix<std::complex<double>>;

    template void trsm(Side, UpLo, Trans, Diag, float, const BLRMatrix<float>&, DenseMatrix<float>&, int);
    template void trsm(Side, UpLo, Trans, Diag, double, const BLRMatrix<double>&, DenseMatrix<double>&, int);
    template void trsm(Side, UpLo, Trans, Diag, std::complex<float>, const BLRMatrix<std::complex<float>>&, DenseMatrix<std::complex<float>>&, int);
    template void trsm(Side, UpLo, Trans, Diag, std::complex<double>, const BLRMatrix<std::complex<double>>&, DenseMatrix<std::complex<double>>&, int);

    template void trsv(UpLo, Trans, Diag, const BLRMatrix<float>&, DenseMatrix<float>&, int);
    template void trsv(UpLo, Trans, Diag, const BLRMatrix<double>&, DenseMatrix<double>&, int);
    template void trsv(UpLo, Trans, Diag, const BLRMatrix<std::complex<float>>&, DenseMatrix<std::complex<float>>&, int);
    template void trsv(UpLo, Trans, Diag, const BLRMatrix<std::complex<double>>&, DenseMatrix<std::complex<double>>&, int);

    template void gemv(Trans, float, const BLRMatrix<float>&, const DenseMatrix<float>&, float, DenseMatrix<float>&, int);
    template void gemv(Trans, double, const BLRMatrix<double>&, const DenseMatrix<double>&, double, DenseMatrix<double>&, int);
    template void gemv(Trans, std::complex<float>, const BLRMatrix<std::complex<float>>&, const DenseMatrix<std::complex<float>>&, std::complex<float>, DenseMatrix<std::complex<float>>&, int);
    template void gemv(Trans, std::complex<double>, const BLRMatrix<std::complex<double>>&, const DenseMatrix<std::complex<double>>&, std::complex<double>, DenseMatrix<std::complex<double>>&, int);

    template void gemm(Trans, Trans, float, const BLRMatrix<float>&, const BLRMatrix<float>&, float, DenseMatrix<float>&, int);
    template void gemm(Trans, Trans, double, const BLRMatrix<double>&, const BLRMatrix<double>&, double, DenseMatrix<double>&, int);
    template void gemm(Trans, Trans, std::complex<float>, const BLRMatrix<std::complex<float>>&, const BLRMatrix<std::complex<float>>&, std::complex<float>, DenseMatrix<std::complex<float>>&, int);
    template void gemm(Trans, Trans, std::complex<double>, const BLRMatrix<std::complex<double>>&, const BLRMatrix<std::complex<double>>&, std::complex<double>, DenseMatrix<std::complex<double>>&, int);

    template void gemm(Trans, Trans, float, const BLRMatrix<float>&, const DenseMatrix<float>&, float, DenseMatrix<float>&, int);
    template void gemm(Trans, Trans, double, const BLRMatrix<double>&, const DenseMatrix<double>&, double, DenseMatrix<double>&, int);
    template void gemm(Trans, Trans, std::complex<float>, const BLRMatrix<std::complex<float>>&, const DenseMatrix<std::complex<float>>&, std::complex<float>, DenseMatrix<std::complex<float>>&, int);
    template void gemm(Trans, Trans, std::complex<double>, const BLRMatrix<std::complex<double>>&, const DenseMatrix<std::complex<double>>&, std::complex<double>, DenseMatrix<std::complex<double>>&, int);

  } // end namespace BLR
} // end namespace strumpack
