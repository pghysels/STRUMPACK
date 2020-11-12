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

    // template<typename scalar_t> BLRMatrix<scalar_t>::BLRMatrix
    // (std::size_t n, const std::vector<std::size_t>& tiles,
    //  const adm_t& admissible, const elem_t& Aelem,
    //  std::vector<int>& piv, const Opts_t& opts)
    //   : BLRMatrix<scalar_t>(n, tiles, n, tiles) {
    //   assert(rowblocks() == colblocks());
    //   piv.resize(rows());
    //   auto rb = rowblocks();
    //   for (std::size_t i=0; i<rb; i++) {
    //     create_dense_tile_left_looking(i, i, Aelem);
    //     auto tpiv = tile(i, i).LU();
    //     std::copy(tpiv.begin(), tpiv.end(), piv.begin()+tileroff(i));
    //     for (std::size_t j=i+1; j<rb; j++) {
    //       if (admissible(i, j))
    //         create_LR_tile_left_looking(i, j, Aelem, opts);
    //       else create_dense_tile_left_looking(i, j, Aelem);
    //       // permute and solve with L, blocks right from the diagonal block
    //       std::vector<int> tpiv
    //         (piv.begin()+tileroff(i), piv.begin()+tileroff(i+1));
    //       tile(i, j).laswp(tpiv, true);
    //       trsm(Side::L, UpLo::L, Trans::N, Diag::U,
    //            scalar_t(1.), tile(i, i), tile(i, j));
    //       if (admissible(j, i))
    //         create_LR_tile_left_looking(j, i, Aelem, opts);
    //       else create_dense_tile_left_looking(j, i, Aelem);
    //       // solve with U, the blocks under the diagonal block
    //       trsm(Side::R, UpLo::U, Trans::N, Diag::N,
    //            scalar_t(1.), tile(i, i), tile(j, i));
    //     }
    //   }
    //   for (std::size_t i=0; i<rb; i++)
    //     for (std::size_t l=tileroff(i); l<tileroff(i+1); l++)
    //       piv[l] += tileroff(i);
    // }

    template<typename scalar_t> BLRMatrix<scalar_t>::BLRMatrix
    (DenseM_t& A, const std::vector<std::size_t>& tiles,
     const adm_t& admissible, std::vector<int>& piv, const Opts_t& opts)
      : BLRMatrix<scalar_t>(A.rows(), tiles, A.cols(), tiles) {
      assert(rowblocks() == colblocks());
      piv.resize(rows());
      auto rb = rowblocks();
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
      // dummy for task synchronization
      std::unique_ptr<int[]> B_(new int[rb*rb]); auto B = B_.get();
#pragma omp taskgroup
#else
      int* B=nullptr; 
#endif
      {
        if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::RL) {
          for (std::size_t i=0; i<rb; i++) {
  #if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
            std::size_t ii = i+rb*i;
  #pragma omp task default(shared) firstprivate(i,ii) depend(inout:B[ii])
  #endif
            {
              create_dense_tile(i, i, A);
              auto tpiv = tile(i, i).LU();
              std::copy(tpiv.begin(), tpiv.end(), piv.begin()+tileroff(i));
            }
            for (std::size_t j=i+1; j<rb; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
            std::size_t ij = i+rb*j;
#pragma omp task default(shared) firstprivate(i,j,ii,ij)        \
  depend(in:B[ii]) depend(inout:B[ij])
#endif
              { // these blocks have received all updates, compress now
                if (admissible(i, j)){
                  create_LR_tile(i, j, A, opts);
                }
                else create_dense_tile(i, j, A);
                // permute and solve with L, blocks right from the diagonal block
                std::vector<int> tpiv
                  (piv.begin()+tileroff(i), piv.begin()+tileroff(i+1));
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
                if (admissible(j, i)) {
                  create_LR_tile(j, i, A, opts);
                }
                else create_dense_tile(j, i, A);
                // solve with U, the blocks under the diagonal block
                trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                    scalar_t(1.), tile(i, i), tile(j, i));
              }
            }
            for (std::size_t j=i+1; j<rb; j++){
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
          }
        }
        else { // LL, Comb or Star
          //FACTOR
          for (std::size_t i=0; i<rb; i++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
          std::size_t ii = i+rb*i;
#pragma omp task default(shared) firstprivate(i,ii) depend(inout:B[ii])
#endif
            {
              create_dense_tile(i, i, A);
              auto tpiv = tile(i, i).LU();
              std::copy(tpiv.begin(), tpiv.end(), piv.begin()+tileroff(i));
            }
            //COMPRESS and SOLVE
            for (std::size_t j=i+1; j<rb; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
            std::size_t ij = i+rb*j;
#pragma omp task default(shared) firstprivate(i,j,ij,ii)  \
  depend(in:B[ii]) depend(inout:B[ij]) priority(rb-j)
#endif
              { 
                // these blocks have received all updates, compress now
                if (admissible(i, j)) create_LR_tile(i, j, A, opts);
                else create_dense_tile(i, j, A);
                // permute and solve with L, blocks right from the diagonal block
                std::vector<int> tpiv
                  (piv.begin()+tileroff(i), piv.begin()+tileroff(i+1));
                tile(i, j).laswp(tpiv, true);
                trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                    scalar_t(1.), tile(i, i), tile(i, j));
              }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
            std::size_t ji = j+rb*i;
#pragma omp task default(shared) firstprivate(i,j,ji,ii)        \
  depend(in:B[ii]) depend(inout:B[ji]) priority(rb-j)
#endif
              {
                if (admissible(j, i)) create_LR_tile(j, i, A, opts);
                else create_dense_tile(j, i, A);
                // solve with U, the blocks under the diagonal block
                trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                    scalar_t(1.), tile(i, i), tile(j, i));
              }
            }
            if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::LL){ // LL-Update
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
                  if(j!=i+1){
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
            }
            else{ //Comb or Star
              for (std::size_t j=i+1; j<rb; j++){
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ij = (i+1)+rb*j, i1j=ij-rb*(j-i), ij1=ij-1;
#pragma omp task default(shared) firstprivate(i,j,ij,i1j,ij1)   \
  depend(in:B[i1j],B[ij1]) depend(inout:B[ij]) 
#endif
                {
                  this->LUAR_B11(i+1, j, i+1, A, opts, B);
                }
                if(j!=i+1){
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ji = j+rb*(i+1), i1j=ji-rb, ij1=ji-j+i;
#pragma omp task default(shared) firstprivate(i,j,ji,i1j,ij1)   \
  depend(in:B[i1j],B[ij1]) depend(inout:B[ji]) 
#endif
                  {
                    this->LUAR_B11(j, i+1, i+1, A, opts, B);
                  }
                }
              }
            }
          }
        }
      }
      for (std::size_t i=0; i<rowblocks(); i++)
        for (std::size_t l=tileroff(i); l<tileroff(i+1); l++)
          piv[l] += tileroff(i);
    }

    
    // private constructor
    template<typename scalar_t> BLRMatrix<scalar_t>::BLRMatrix
    (std::size_t m, const std::vector<std::size_t>& rowtiles,
     std::size_t n, const std::vector<std::size_t>& coltiles)
      : m_(m), n_(n) {
      nbrows_ = rowtiles.size();
      nbcols_ = coltiles.size();
      roff_.resize(nbrows_+1);
      coff_.resize(nbcols_+1);
      for (std::size_t i=1; i<=nbrows_; i++)
        roff_[i] = roff_[i-1] + rowtiles[i-1];
      for (std::size_t j=1; j<=nbcols_; j++)
        coff_[j] = coff_[j-1] + coltiles[j-1];
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
    BLRMatrix<scalar_t>::maximum_rank() const {
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
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop collapse(2) default(shared)
#endif
      for (std::size_t j=0; j<cb; j++)
        for (std::size_t i=0; i<rb; i++) {
          tile(i, j).draw(of, roff+tileroff(i), coff+tilecoff(j));
        }
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
          else {
            std::cout << "D:" << tij.rows() << "x" << tij.cols() << " " << std::endl;
            //tij.D().print();
          }   
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

    template<typename scalar_t> scalar_t
    BLRMatrix<scalar_t>::operator()(std::size_t i, std::size_t j) const {
      auto ti = std::distance
        (roff_.begin(), std::upper_bound(roff_.begin(), roff_.end(), i)) - 1;
      auto tj = std::distance
        (coff_.begin(), std::upper_bound(coff_.begin(), coff_.end(), j)) - 1;
      return tile(ti, tj)(i - roff_[ti], j - coff_[tj]);
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
    BLRMatrix<scalar_t>::construct_and_partial_factor
    (DenseMatrix<scalar_t>& A11, DenseMatrix<scalar_t>& A12,
     DenseMatrix<scalar_t>& A21, DenseMatrix<scalar_t>& A22,
     BLRMatrix<scalar_t>& B11, std::vector<int>& piv,
     BLRMatrix<scalar_t>& B12, BLRMatrix<scalar_t>& B21,
     const std::vector<std::size_t>& tiles1,
     const std::vector<std::size_t>& tiles2,
     const DenseMatrix<bool>& admissible,
     const BLROptions<scalar_t>& opts) {
      B11 = BLRMatrix<scalar_t>(A11.rows(), tiles1, A11.cols(), tiles1);
      B12 = BLRMatrix<scalar_t>(A12.rows(), tiles1, A12.cols(), tiles2);
      B21 = BLRMatrix<scalar_t>(A21.rows(), tiles2, A21.cols(), tiles1);
      piv.resize(B11.rows());
      auto rb = B11.rowblocks();
      auto rb2 = B21.rowblocks();
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
      auto lrb = rb+rb2;
      // dummy for task synchronization
      std::unique_ptr<int[]> B_(new int[lrb*lrb]()); auto B = B_.get();
#pragma omp taskgroup
#else
      int* B=nullptr; 
#endif
      {
        // RL
        if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::RL){
          for (std::size_t i=0; i<rb; i++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
          std::size_t ii = i+lrb*i;
#pragma omp task default(shared) firstprivate(i,ii) depend(inout:B[ii])
#endif
            {
              B11.create_dense_tile(i, i, A11);
              auto tpiv = B11.tile(i, i).LU();
              std::copy(tpiv.begin(), tpiv.end(), piv.begin()+B11.tileroff(i));
            }
            for (std::size_t j=i+1; j<rb; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
            std::size_t ij = i+lrb*j;
#pragma omp task default(shared) firstprivate(i,j,ij,ii)  \
  depend(in:B[ii]) depend(inout:B[ij]) priority(rb-j)
#endif
              { // these blocks have received all updates, compress now
                if (admissible(i, j)) B11.create_LR_tile(i, j, A11, opts);
                else B11.create_dense_tile(i, j, A11);
                // permute and solve with L, blocks right from the diagonal block
                std::vector<int> tpiv
                  (piv.begin()+B11.tileroff(i), piv.begin()+B11.tileroff(i+1));
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
                // permute and solve with L  blocks right from the diagonal block
                std::vector<int> tpiv
                  (piv.begin()+B11.tileroff(i), piv.begin()+B11.tileroff(i+1));
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
            for (std::size_t j=i+1; j<rb; j++)
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
            for (std::size_t k=i+1; k<rb; k++)
              for (std::size_t j=0; j<rb2; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ki = k+lrb*i, ij2 = i+lrb*(rb+j), kj2 = k+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(i,k,j,ki,ij2,kj2) \
  depend(in:B[ki],B[ij2]) depend(inout:B[kj2])
#endif
                { // Schur complement updates, always into full rank
                  auto Akj = B12.tile(A12, k, j);
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                      B11.tile(k, i), B12.tile(i, j), scalar_t(1.), Akj);
                }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ik = i+lrb*k, j2i = (j+rb)+lrb*i, j2k = (rb+j)+lrb*k;
#pragma omp task default(shared) firstprivate(i,k,j,ik,j2i,j2k)      \
  depend(in:B[ik],B[j2i]) depend(inout:B[j2k])
#endif
                { // Schur complement updates, always into full rank
                  auto Ajk = B21.tile(A21, j, k);
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                      B21.tile(j, i), B11.tile(i, k), scalar_t(1.), Ajk);
                }
              }
            for (std::size_t j=0; j<rb2; j++)
              for (std::size_t k=0; k<rb2; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ij2 = i+lrb*(rb+j), k2i = (rb+k)+lrb*i, k2j2 = (rb+k)+lrb*(rb+j);
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
        }
        else { //LL, Comb or Star
          //FACTOR
          for (std::size_t i=0; i<rb; i++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
          std::size_t ii = i+lrb*i;
#pragma omp task default(shared) firstprivate(i,ii) depend(inout:B[ii])
#endif
            {
              B11.create_dense_tile(i, i, A11);
              auto tpiv = B11.tile(i, i).LU();
              std::copy(tpiv.begin(), tpiv.end(), piv.begin()+B11.tileroff(i));
            }
            //COMPRESS and SOLVE
            for (std::size_t j=i+1; j<rb; j++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
            std::size_t ij = i+lrb*j;
#pragma omp task default(shared) firstprivate(i,j,ij,ii)  \
  depend(in:B[ii]) depend(inout:B[ij]) priority(rb-j)
#endif
              { // these blocks have received all updates, compress now
                if (admissible(i, j)) B11.create_LR_tile(i, j, A11, opts);
                else B11.create_dense_tile(i, j, A11);
                // permute and solve with L, blocks right from the diagonal block
                std::vector<int> tpiv
                  (piv.begin()+B11.tileroff(i), piv.begin()+B11.tileroff(i+1));
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
                // permute and solve with L  blocks right from the diagonal block
                std::vector<int> tpiv
                  (piv.begin()+B11.tileroff(i), piv.begin()+B11.tileroff(i+1));
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
            if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::LL){ //LL-Update
              for (std::size_t j=i+1; j<rb; j++){
                for (std::size_t k=0; k<i+1; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ij = (i+1)+lrb*j, ik = (i+1)+lrb*k, kj = k+lrb*j;
#pragma omp task default(shared) firstprivate(i,j,k,ij,ik,kj)   \
  depend(in:B[ik],B[kj]) depend(inout:B[ij]) priority(rb-j)
#endif 
                  { // Schur complement updates, always into full rank
                    auto Aij = B11.tile(A11, i+1, j);
                    gemm(Trans::N, Trans::N, scalar_t(-1.),
                        B11.tile(i+1, k), B11.tile(k, j), scalar_t(1.), Aij);
                  }
                  if(j!=i+1){
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ji = j+lrb*(i+1), jk = j+lrb*k, ki = k+lrb*(i+1);
#pragma omp task default(shared) firstprivate(i,j,k,ji,jk,ki)   \
  depend(in:B[jk],B[ki]) depend(inout:B[ji]) priority(rb-j)
#endif 
                    { // Schur complement updates, always into full rank
                      auto Aji = B11.tile(A11, j, i+1);
                      gemm(Trans::N, Trans::N, scalar_t(-1.),
                          B11.tile(j, k), B11.tile(k, i+1), scalar_t(1.), Aji);
                    }
                  }
                }
              }
              if(i+1<rb){
                for (std::size_t j=0; j<rb2; j++){
                  for (std::size_t k=0; k<i+1; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ik = (i+1)+lrb*k, ij2 = (i+1)+lrb*(rb+j), kj2 = k+lrb*(rb+j);
#pragma omp task default(shared) firstprivate(i,k,j,ik,ij2,kj2) \
  depend(in:B[ik],B[kj2]) depend(inout:B[ij2])
#endif 
                    { // Schur complement updates, always into full rank
                      auto Aij = B12.tile(A12, i+1, j);
                      gemm(Trans::N, Trans::N, scalar_t(-1.),
                          B11.tile(i+1, k), B12.tile(k, j), scalar_t(1.), Aij);
                    }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ki = k+lrb*(i+1), j2i = (j+rb)+lrb*(i+1), j2k = (rb+j)+lrb*k;
#pragma omp task default(shared) firstprivate(i,k,j,ki,j2i,j2k)      \
  depend(in:B[j2k],B[ki]) depend(inout:B[j2i])
#endif 
                    { // Schur complement updates, always into full rank
                      auto Aji = B21.tile(A21, j, i+1);
                      gemm(Trans::N, Trans::N, scalar_t(-1.),
                          B21.tile(j, k), B11.tile(k, i+1), scalar_t(1.), Aji);
                    }
                  }
                }
              }
            }
            else{ //Comb or Star (LUAR)
              for (std::size_t j=i+1; j<rb; j++){
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ij = (i+1)+lrb*j, i1j=ij-lrb*(j-i), ij1=ij-1;
#pragma omp task default(shared) firstprivate(i,j,ij,i1j,ij1)   \
  depend(in:B[i1j],B[ij1]) depend(inout:B[ij]) 
#endif
                {
                  B11.LUAR_B11(i+1, j, i+1, A11, opts, B);
                }
                if(j!=i+1){
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ji = j+lrb*(i+1), i1j=ji-lrb, ij1=ji-j+i;
#pragma omp task default(shared) firstprivate(i,j,ji,i1j,ij1)   \
  depend(in:B[i1j],B[ij1]) depend(inout:B[ji]) 
#endif
                  {
                      B11.LUAR_B11(j, i+1, i+1, A11, opts, B);
                  }
                }
              }
              if(i+1<rb){
                for (std::size_t j=0; j<rb2; j++){
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t ij2 = (i+1)+lrb*(rb+j), i1j=ij2-lrb*(rb+j-i), ij1=ij2-1;
#pragma omp task default(shared) firstprivate(i,j,ij2,i1j,ij1)   \
  depend(in:B[i1j],B[ij1]) depend(inout:B[ij2]) 
#endif
                  {
                      B12.LUAR_B12(i+1, j, i+1, B11, A12, opts, B);
                  }
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t j2i = (rb+j)+lrb*(i+1), ji1=j2i-lrb, j1i=j2i-(rb-i)-j;
#pragma omp task default(shared) firstprivate(i,j,j2i,ji1,j1i)   \
  depend(in:B[ji1],B[j1i]) depend(inout:B[j2i]) 
#endif
                  {
                      B21.LUAR_B21(i+1, j, i+1, B11, A21, opts, B);
                  }
                }
              }
            }
          }
          for(std::size_t i=0; i<rb2; i++) {
            for (std::size_t j=0; j<rb2; j++) {
              if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::LL){ //LL-Update
                for (std::size_t k=0; k<rb; k++) {
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t i2j2 = (rb+i)+lrb*(rb+j), i2k = (rb+i)+lrb*k, kj2 = k+lrb*(rb+j);
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
              }
              else{ //Comb or Star (LUAR-Update)
#if defined(STRUMPACK_USE_OPENMP_TASK_DEPEND)
              std::size_t i2j2 = (rb+i)+lrb*(rb+j), i1j=rb*lrb-(rb2-i), ij1=i2j2-(i+1);
#pragma omp task default(shared) firstprivate(i,j,i2j2,i1j,ij1)   \
  depend(in:B[i1j],B[ij1]) depend(inout:B[i2j2]) 
#endif
                {
                  LUAR_B22(i, j, rb, B12, B21, A22, opts, B);
                }
              }
            }
          }
        }
      }
      for (std::size_t i=0; i<rb; i++)
        for (std::size_t l=B11.tileroff(i); l<B11.tileroff(i+1); l++)
          piv[l] += B11.tileroff(i);
      A11.clear();
      A12.clear();
      A21.clear();
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::LUAR_B11
    (std::size_t i, std::size_t j,
     std::size_t kmax, DenseMatrix<scalar_t>&A11, const BLROptions<scalar_t>& opts, int* B){
      if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::STAR){
        std::size_t rank_sum=0;
        auto Aij = tile(A11, i, j);
        for (std::size_t k=0; k<kmax; k++) {
          if(!(tile(i, k).is_low_rank() || tile(k, j).is_low_rank())){
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                tile(i, k), tile(k, j), scalar_t(1.), Aij);
          } 
          else if(tile(i, k).is_low_rank() && tile(k, j).is_low_rank()){
            rank_sum+=std::min(tile(i, k).rank(), tile(k, j).rank());
          }
          else if(tile(i, k).is_low_rank()){
            rank_sum+=tile(i, k).rank();
          }
          else{
            rank_sum+=tile(k, j).rank();
          }
        }
        if(rank_sum>0){
          DenseMatrix<scalar_t> Uall(Aij.rows(), rank_sum);
          DenseMatrix<scalar_t> Vall(rank_sum, Aij.cols());
          std::size_t rank_tmp=0;
          for (std::size_t k=0; k<kmax; k++) {
            if(tile(i, k).is_low_rank() && tile(k, j).is_low_rank()){
              std::size_t minrank=std::min(tile(i,k).rank(), tile(k,j).rank());
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), minrank, Uall, 0, rank_tmp), 
              t2(minrank, Aij.cols(), Vall, rank_tmp, 0);
              tile(i,k).multiply(tile(k,j), t1, t2);
              rank_tmp+=minrank;
            }
            else if (tile(i, k).is_low_rank()){
              std::size_t minrank=tile(i,k).rank();
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), minrank, Uall, 0, rank_tmp), 
              t2(minrank, Aij.cols(), Vall, rank_tmp, 0);
              tile(i,k).multiply(tile(k,j), t1, t2);
              rank_tmp+=minrank;
            }
            else if (tile(k, j).is_low_rank()){
              std::size_t minrank=tile(k,j).rank();
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), minrank, Uall, 0, rank_tmp), 
              t2(minrank, Aij.cols(), Vall, rank_tmp, 0);
              tile(i,k).multiply(tile(k,j), t1, t2);
              rank_tmp+=minrank;
            }
          }
          if (opts.compression_kernel() == CompressionKernel::FULL){
            //Recompress Uall and Vall
            DenseMatrix<scalar_t> UU, UV;
            Uall.low_rank(UU, UV, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
            DenseMatrix<scalar_t> VU, VV;
            Vall.low_rank(VU, VV, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
            DenseMatrix<scalar_t> tmp1(UV.rows(), VU.cols());
            gemm(Trans::N, Trans::N, scalar_t(1.), UV, VU, scalar_t(0.), tmp1);
            if (UU.cols() > VU.cols()) {
              //(UU*(UV * VU)) *VV
              DenseMatrix<scalar_t> tmp2(UU.rows(), tmp1.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), UU, tmp1, scalar_t(0.), tmp2);
              gemm(Trans::N, Trans::N, scalar_t(-1.), tmp2, VV, scalar_t(1.), Aij);
            }
            else{
              // UU* ((UV * VU)*VV)
              DenseMatrix<scalar_t> tmp2(tmp1.rows(), VV.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), tmp1, VV, scalar_t(0.), tmp2);
              gemm(Trans::N, Trans::N, scalar_t(-1.), UU, tmp2, scalar_t(1.), Aij);
            }
          }
          else{ //Recompress Uall OR Vall
            if (Uall.rows() > Vall.cols()){
              // (Uall * U1) *V1
              DenseMatrix<scalar_t> U1, V1;
              Vall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
              DenseMatrix<scalar_t> tmp(Uall.rows(), U1.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), Uall, U1, scalar_t(0.), tmp);
              gemm(Trans::N, Trans::N, scalar_t(-1.), tmp, V1, scalar_t(1.), Aij);
            }
            else{
              // U1 * (V1 * Vall)
              DenseMatrix<scalar_t> U1, V1;
              Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
              DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
              gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij);
            }
            /* //Recompress Uall only
            DenseMatrix<scalar_t> U1, V1;
            Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
            DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
            gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
            gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij); */
          }
        }
      }
      else if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::COMB){
        std::vector<std::pair<size_t,size_t>> ranks_idx;
        std::size_t rank_sum=0;
        auto Aij = tile(A11, i, j);
        for (std::size_t k=0; k<kmax; k++) {
          if(!(tile(i, k).is_low_rank() || tile(k, j).is_low_rank())){ // both tiles are DenseTiles
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                     tile(i, k), tile(k, j), scalar_t(1.), Aij);
          } 
          else if(tile(i, k).is_low_rank() && tile(k, j).is_low_rank()){ // both tiles are LR, collect size of LR matrices
            ranks_idx.emplace_back(std::min(tile(i, k).rank(), tile(k, j).rank()),k);
            rank_sum += std::min(tile(i, k).rank(), tile(k, j).rank());
          } 
          else if(tile(i, k).is_low_rank()){ // collect size of LR matrix
            ranks_idx.emplace_back(tile(i, k).rank(),k);
            rank_sum += tile(i, k).rank();
          } 
          else{ // collect size of LR matrix
            ranks_idx.emplace_back(tile(k, j).rank(),k);
            rank_sum += tile(k, j).rank();
          } 
        }
        if(rank_sum>0){
          if(ranks_idx.size()>1){
            std::sort(ranks_idx.begin(),ranks_idx.end());
            DenseMatrix<scalar_t> tmpU(Aij.rows(), rank_sum);
            DenseMatrix<scalar_t> tmpV(rank_sum, Aij.cols());
            std::size_t rank_tmp=ranks_idx[0].first;
            DenseMatrixWrapper<scalar_t> t1(Aij.rows(), rank_tmp, tmpU, 0, 0), 
            t2(rank_tmp, Aij.cols(), tmpV, 0, 0);
            tile(i,ranks_idx[0].second).multiply(tile(ranks_idx[0].second,j), t1, t2);
            for (std::size_t k=1; k<ranks_idx.size(); k++) {
              DenseMatrix<scalar_t> Uall(Aij.rows(), rank_tmp+ranks_idx[k].first);
              DenseMatrix<scalar_t> Vall(rank_tmp+ranks_idx[k].first, Aij.cols());
              Uall.copy_tillpos(tmpU, Aij.rows(), rank_tmp);  
              Vall.copy_tillpos(tmpV, rank_tmp, Aij.cols());
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), ranks_idx[k].first, Uall, 0, rank_tmp), 
              t2(ranks_idx[k].first, Aij.cols(), Vall, rank_tmp, 0);
              tile(i,ranks_idx[k].second).multiply(tile(ranks_idx[k].second,j), t1, t2);
              if (opts.compression_kernel() == CompressionKernel::FULL){
                //Recompress Uall and Vall
                DenseMatrix<scalar_t> UU, UV;
                Uall.low_rank(UU, UV, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
                DenseMatrix<scalar_t> VU, VV;
                Vall.low_rank(VU, VV, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
                DenseMatrix<scalar_t> tmp1(UV.rows(), VU.cols());
                gemm(Trans::N, Trans::N, scalar_t(1.), UV, VU, scalar_t(0.), tmp1);
                if (UU.cols() > VU.cols()) {
                  //(UU*(UV * VU)) *VV
                  DenseMatrix<scalar_t> tmp2(UU.rows(), tmp1.cols());
                  gemm(Trans::N, Trans::N, scalar_t(1.), UU, tmp1, scalar_t(0.), tmp2);
                  if (k==ranks_idx.size()-1){
                    gemm(Trans::N, Trans::N, scalar_t(-1.), tmp2, VV, scalar_t(1.), Aij);
                  }
                  else{
                    tmpU.copy_topos(tmp2,0,0);
                    tmpV.copy_topos(VV,0,0);
                    rank_tmp = tmp2.cols();
                  }
                }
                else{
                  // UU* ((UV * VU)*VV)
                  DenseMatrix<scalar_t> tmp2(tmp1.rows(), VV.cols());
                  gemm(Trans::N, Trans::N, scalar_t(1.), tmp1, VV, scalar_t(0.), tmp2);
                  if (k==ranks_idx.size()-1){
                    gemm(Trans::N, Trans::N, scalar_t(-1.), UU, tmp2, scalar_t(1.), Aij);
                  }
                  else{
                    tmpU.copy_topos(UU,0,0);
                    tmpV.copy_topos(tmp2,0,0);
                    rank_tmp = UU.cols();
                  }
                }
              }
              else{ //Recompress Uall OR Vall
                if (Uall.rows() > Vall.cols()){
                  // (Uall * U1) *V1
                  DenseMatrix<scalar_t> U1, V1;
                  Vall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
                  if (k==ranks_idx.size()-1){
                    DenseMatrix<scalar_t> tmp(Uall.rows(), U1.cols());
                    gemm(Trans::N, Trans::N, scalar_t(1.), Uall, U1, scalar_t(0.), tmp);
                    gemm(Trans::N, Trans::N, scalar_t(-1.), tmp, V1, scalar_t(1.), Aij);
                  }
                  else{
                    DenseMatrixWrapper<scalar_t> t1(Uall.rows(), U1.cols(), tmpU, 0, 0), 
                    t2(V1.rows(), V1.cols(), tmpV, 0, 0);
                    gemm(Trans::N, Trans::N, scalar_t(1.), Uall, U1, scalar_t(0.), t1);
                    t2=V1;
                    rank_tmp = U1.cols();
                  }
                }
                else{
                  // U1 * (V1 * Vall)
                  DenseMatrix<scalar_t> U1, V1;
                  Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
                  if (k==ranks_idx.size()-1){
                    DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
                    gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
                    gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij); 
                  }
                  else{
                    DenseMatrixWrapper<scalar_t> t1(U1.rows(), U1.cols(), tmpU, 0, 0), 
                    t2(V1.rows(), Vall.cols(), tmpV, 0, 0);
                    t1=U1;
                    gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), t2);
                    rank_tmp = U1.cols();
                  }
                }
                /* //Recompress Uall only
                DenseMatrix<scalar_t> U1, V1;
                Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
                DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
                gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
                if (k==ranks_idx.size()-1){
                  gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij); 
                }
                else{
                  tmpU.copy_topos(U1,0,0);
                  tmpV.copy_topos(tmp,0,0);
                  rank_tmp = U1.cols();
                }*/
              }
            }
          }
          else{
            LRTile<scalar_t> tmp=tile(i,ranks_idx[0].second).multiply(tile(ranks_idx[0].second,j));
            gemm(Trans::N, Trans::N, scalar_t(-1.), tmp.U(), tmp.V(), scalar_t(1.), Aij);
          }
        }
      }
    }


    template<typename scalar_t> void
    BLRMatrix<scalar_t>::LUAR_B12
    (std::size_t i, std::size_t j,
     std::size_t kmax, BLRMatrix<scalar_t>& B11, DenseMatrix<scalar_t>&A12, const BLROptions<scalar_t>& opts, int* B)
    {
      if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::STAR){
        std::size_t rank_sum=0;
        auto Aij = tile(A12, i, j);
        for (std::size_t k=0; k<kmax; k++) {
          if(!(B11.tile(i, k).is_low_rank() || tile(k, j).is_low_rank())){ //both tiles are dense, then gemm directly
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                      B11.tile(i, k), tile(k, j), scalar_t(1.), Aij);
          } 
          else if(B11.tile(i, k).is_low_rank() && tile(k, j).is_low_rank()){
            rank_sum+=std::min(B11.tile(i, k).rank(), tile(k, j).rank());
          }
          else if(B11.tile(i, k).is_low_rank()){
            rank_sum+=B11.tile(i, k).rank();
          }
          else{
            rank_sum+=tile(k, j).rank();
          }
        }
        if(rank_sum>0){
          DenseMatrix<scalar_t> Uall(Aij.rows(), rank_sum);
          DenseMatrix<scalar_t> Vall(rank_sum, Aij.cols());
          std::size_t rank_tmp=0;
          for (std::size_t k=0; k<kmax; k++) {
            if(B11.tile(i, k).is_low_rank() && tile(k, j).is_low_rank()){
              std::size_t minrank=std::min(B11.tile(i,k).rank(), tile(k,j).rank());
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), minrank, Uall, 0, rank_tmp), 
              t2(minrank, Aij.cols(), Vall, rank_tmp, 0);
              B11.tile(i,k).multiply(tile(k,j), t1, t2);
              rank_tmp+=minrank;
            }
            else if (B11.tile(i, k).is_low_rank()){
              std::size_t minrank=B11.tile(i,k).rank();
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), minrank, Uall, 0, rank_tmp), 
              t2(minrank, Aij.cols(), Vall, rank_tmp, 0);
              B11.tile(i,k).multiply(tile(k,j), t1, t2);
              rank_tmp+=minrank;
            }
            else if(tile(k, j).is_low_rank()){
              std::size_t minrank=tile(k,j).rank();
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), minrank, Uall, 0, rank_tmp), 
              t2(minrank, Aij.cols(), Vall, rank_tmp, 0);
              B11.tile(i,k).multiply(tile(k,j), t1, t2);
              rank_tmp+=minrank;
            }
          }
          if (opts.compression_kernel() == CompressionKernel::FULL){
            //Recompress Uall and Vall
            DenseMatrix<scalar_t> UU, UV;
            Uall.low_rank(UU, UV, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
            DenseMatrix<scalar_t> VU, VV;
            Vall.low_rank(VU, VV, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
            DenseMatrix<scalar_t> tmp1(UV.rows(), VU.cols());
            gemm(Trans::N, Trans::N, scalar_t(1.), UV, VU, scalar_t(0.), tmp1);
            if (UU.cols() > VU.cols()) {
              //(UU*(UV * VU)) *VV
              DenseMatrix<scalar_t> tmp2(UU.rows(), tmp1.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), UU, tmp1, scalar_t(0.), tmp2);
              gemm(Trans::N, Trans::N, scalar_t(-1.), tmp2, VV, scalar_t(1.), Aij);
            }
            else{
              // UU* ((UV * VU)*VV)
              DenseMatrix<scalar_t> tmp2(tmp1.rows(), VV.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), tmp1, VV, scalar_t(0.), tmp2);
              gemm(Trans::N, Trans::N, scalar_t(-1.), UU, tmp2, scalar_t(1.), Aij);
            }
          }
          else{ //Recompress Uall OR Vall
            if (Uall.rows() > Vall.cols()){
              // (Uall * U1) *V1
              DenseMatrix<scalar_t> U1, V1;
              Vall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
              DenseMatrix<scalar_t> tmp(Uall.rows(), U1.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), Uall, U1, scalar_t(0.), tmp);
              gemm(Trans::N, Trans::N, scalar_t(-1.), tmp, V1, scalar_t(1.), Aij);
            }
            else{
              // U1 * (V1 * Vall)
              DenseMatrix<scalar_t> U1, V1;
              Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
              DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
              gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij);
            }
            /* //Recompress Uall only
            DenseMatrix<scalar_t> U1, V1;
            Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
            DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
            gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
            gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij); */
          }
        }
      }
      else if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::COMB){
        std::vector<std::pair<size_t,size_t>> ranks_idx;
        std::size_t rank_sum=0;
        auto Aij = tile(A12, i, j);
        for (std::size_t k=0; k<kmax; k++) {
          if(!(B11.tile(i, k).is_low_rank() || tile(k, j).is_low_rank())){ //both tiles are dense, then gemm directly
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                      B11.tile(i, k), tile(k, j), scalar_t(1.), Aij);
          } 
          else if(B11.tile(i, k).is_low_rank() && tile(k, j).is_low_rank()){ //both tiles are LR, collect size of LR matrices
            ranks_idx.emplace_back(std::min(B11.tile(i, k).rank(), tile(k, j).rank()),k);
            rank_sum += std::min(B11.tile(i, k).rank(), tile(k, j).rank());
          } 
          else if(B11.tile(i, k).is_low_rank()){ // collect size of LR matrix
            ranks_idx.emplace_back(B11.tile(i, k).rank(),k);
            rank_sum += B11.tile(i, k).rank();
          } 
          else{ // collect size of LR matrix
            ranks_idx.emplace_back(tile(k, j).rank(),k);
            rank_sum += tile(k, j).rank();
          } 
        }
        if(rank_sum>0){
          if(ranks_idx.size()>1){
            //sort ranks in increasing order
            std::sort(ranks_idx.begin(),ranks_idx.end());
            DenseMatrix<scalar_t> tmpU(Aij.rows(), rank_sum);
            DenseMatrix<scalar_t> tmpV(rank_sum, Aij.cols());
            std::size_t rank_tmp=ranks_idx[0].first;
            DenseMatrixWrapper<scalar_t> t1(Aij.rows(), rank_tmp, tmpU, 0, 0), 
            t2(rank_tmp, Aij.cols(), tmpV, 0, 0);
            B11.tile(i,ranks_idx[0].second).multiply(tile(ranks_idx[0].second,j), t1, t2);
            for (std::size_t k=1; k<ranks_idx.size(); k++) {
              DenseMatrix<scalar_t> Uall(Aij.rows(), rank_tmp+ranks_idx[k].first);
              DenseMatrix<scalar_t> Vall(rank_tmp+ranks_idx[k].first, Aij.cols());
              Uall.copy_tillpos(tmpU, tmpU.rows(), rank_tmp);
              Vall.copy_tillpos(tmpV, rank_tmp, tmpV.cols());
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), ranks_idx[k].first, Uall, 0, rank_tmp), 
              t2(ranks_idx[k].first, Aij.cols(), Vall, rank_tmp, 0);
              B11.tile(i,ranks_idx[k].second).multiply(tile(ranks_idx[k].second,j), t1, t2);
              if (opts.compression_kernel() == CompressionKernel::FULL){
                //Recompress Uall and Vall
                DenseMatrix<scalar_t> UU, UV;
                Uall.low_rank(UU, UV, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
                DenseMatrix<scalar_t> VU, VV;
                Vall.low_rank(VU, VV, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
                DenseMatrix<scalar_t> tmp1(UV.rows(), VU.cols());
                gemm(Trans::N, Trans::N, scalar_t(1.), UV, VU, scalar_t(0.), tmp1);
                if (UU.cols() > VU.cols()) {
                  //(UU*(UV * VU)) *VV
                  DenseMatrix<scalar_t> tmp2(UU.rows(), tmp1.cols());
                  gemm(Trans::N, Trans::N, scalar_t(1.), UU, tmp1, scalar_t(0.), tmp2);
                  if (k==ranks_idx.size()-1){
                    gemm(Trans::N, Trans::N, scalar_t(-1.), tmp2, VV, scalar_t(1.), Aij);
                  }
                  else{
                    tmpU.copy_topos(tmp2,0,0);
                    tmpV.copy_topos(VV,0,0);
                    rank_tmp = tmp2.cols();
                  }
                }
                else{
                // UU* ((UV * VU)*VV)
                  DenseMatrix<scalar_t> tmp2(tmp1.rows(), VV.cols());
                  gemm(Trans::N, Trans::N, scalar_t(1.), tmp1, VV, scalar_t(0.), tmp2);
                  if (k==ranks_idx.size()-1){
                    gemm(Trans::N, Trans::N, scalar_t(-1.), UU, tmp2, scalar_t(1.), Aij);
                  }
                  else{
                    tmpU.copy_topos(UU,0,0);
                    tmpV.copy_topos(tmp2,0,0);
                    rank_tmp = UU.cols();
                  }
                }
              }
              else{ //Recompress Uall OR Vall
                if (Uall.rows() > Vall.cols()){
                  // (Uall * U1) *V1
                  DenseMatrix<scalar_t> U1, V1;
                  Vall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(),
                                std::max(Vall.rows(), Vall.cols()),
                                params::task_recursion_cutoff_level);
                  if (k==ranks_idx.size()-1){
                    DenseMatrix<scalar_t> tmp(Uall.rows(), U1.cols());
                    gemm(Trans::N, Trans::N, scalar_t(1.), Uall, U1, scalar_t(0.), tmp);
                    gemm(Trans::N, Trans::N, scalar_t(-1.), tmp, V1, scalar_t(1.), Aij);
                  }
                  else{
                    DenseMatrixWrapper<scalar_t> t1(Uall.rows(), U1.cols(), tmpU, 0, 0), 
                    t2(V1.rows(), V1.cols(), tmpV, 0, 0);
                    gemm(Trans::N, Trans::N, scalar_t(1.), Uall, U1, scalar_t(0.), t1);
                    t2=V1;
                    rank_tmp = U1.cols();
                  }
                }
                else{
                  // U1 * (V1 * Vall)
                  DenseMatrix<scalar_t> U1, V1;
                  Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
                  if (k==ranks_idx.size()-1){
                    DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
                    gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
                    gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij); 
                  }
                  else{
                    DenseMatrixWrapper<scalar_t> t1(U1.rows(), U1.cols(), tmpU, 0, 0), 
                    t2(V1.rows(), Vall.cols(), tmpV, 0, 0);
                    t1=U1;
                    gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), t2);
                    rank_tmp = U1.cols();
                  }
                }
                /* //Recompress Uall only
                DenseMatrix<scalar_t> U1, V1;
                Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
                DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
                gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
                if (k==ranks_idx.size()-1){
                  gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij); 
                }
                else{
                  tmpU.copy_topos(U1,0,0);
                  tmpV.copy_topos(tmp,0,0);
                  rank_tmp = U1.cols();
                }*/
              }
            }
          }
          else{
            LRTile<scalar_t> tmp=B11.tile(i,ranks_idx[0].second).multiply(tile(ranks_idx[0].second,j));
            gemm(Trans::N, Trans::N, scalar_t(-1.), tmp.U(), tmp.V(), scalar_t(1.), Aij);
          }
        }
      }
    }

    template<typename scalar_t> void
    BLRMatrix<scalar_t>::LUAR_B21
    (std::size_t i, std::size_t j,
     std::size_t kmax, BLRMatrix<scalar_t>& B11, DenseMatrix<scalar_t>&A21, const BLROptions<scalar_t>& opts, int* B)
    {
      if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::STAR){
        std::size_t rank_sum=0;
        auto Aij = tile(A21, j, i);
        for (std::size_t k=0; k<kmax; k++) {
          if(!(tile(j, k).is_low_rank() || B11.tile(k, i).is_low_rank())){ //both tiles are dense, then gemm directly
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                      tile(j, k), B11.tile(k, i), scalar_t(1.), Aij);
          } 
          else if(tile(j, k).is_low_rank() && B11.tile(k, i).is_low_rank()){
            rank_sum+=std::min(tile(j, k).rank(), B11.tile(k, i).rank());
          }
          else if(tile(j, k).is_low_rank()){
            rank_sum+=tile(j, k).rank();
          }
          else{
            rank_sum+=B11.tile(k, i).rank();
          }
        }
        if(rank_sum>0){
          DenseMatrix<scalar_t> Uall(Aij.rows(), rank_sum);
          DenseMatrix<scalar_t> Vall(rank_sum, Aij.cols());
          std::size_t rank_tmp=0;
          for (std::size_t k=0; k<kmax; k++) {
            if(tile(j, k).is_low_rank() && B11.tile(k, i).is_low_rank()){
              std::size_t minrank=std::min(tile(j,k).rank(), B11.tile(k,i).rank());
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), minrank, Uall, 0, rank_tmp), 
              t2(minrank, Aij.cols(), Vall, rank_tmp, 0);
              tile(j,k).multiply(B11.tile(k,i), t1, t2);
              rank_tmp+=minrank;
            }
            else if(tile(j, k).is_low_rank()){
              std::size_t minrank=tile(j,k).rank();
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), minrank, Uall, 0, rank_tmp), 
              t2(minrank, Aij.cols(), Vall, rank_tmp, 0);
              tile(j,k).multiply(B11.tile(k,i), t1, t2);
              rank_tmp+=minrank;
            }
            else if(B11.tile(k, i).is_low_rank()){
              std::size_t minrank=B11.tile(k,i).rank();
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), minrank, Uall, 0, rank_tmp), 
              t2(minrank, Aij.cols(), Vall, rank_tmp, 0);
              tile(j,k).multiply(B11.tile(k,i), t1, t2);
              rank_tmp+=minrank;
            }
          }
          if (opts.compression_kernel() == CompressionKernel::FULL){
            //Recompress Uall and Vall
            DenseMatrix<scalar_t> UU, UV;
            Uall.low_rank(UU, UV, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
            DenseMatrix<scalar_t> VU, VV;
            Vall.low_rank(VU, VV, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
            DenseMatrix<scalar_t> tmp1(UV.rows(), VU.cols());
            gemm(Trans::N, Trans::N, scalar_t(1.), UV, VU, scalar_t(0.), tmp1);
            if (UU.cols() > VU.cols()) {
              //(UU*(UV * VU)) *VV
              DenseMatrix<scalar_t> tmp2(UU.rows(), tmp1.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), UU, tmp1, scalar_t(0.), tmp2);
              gemm(Trans::N, Trans::N, scalar_t(-1.), tmp2, VV, scalar_t(1.), Aij);
            }
            else{
              // UU* ((UV * VU)*VV)
              DenseMatrix<scalar_t> tmp2(tmp1.rows(), VV.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), tmp1, VV, scalar_t(0.), tmp2);
              gemm(Trans::N, Trans::N, scalar_t(-1.), UU, tmp2, scalar_t(1.), Aij);
            }
          }
          else{ //Recompress Uall OR Vall
            if (Uall.rows() > Vall.cols()){
              // (Uall * U1) *V1
              DenseMatrix<scalar_t> U1, V1;
              Vall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
              DenseMatrix<scalar_t> tmp(Uall.rows(), U1.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), Uall, U1, scalar_t(0.), tmp);
              gemm(Trans::N, Trans::N, scalar_t(-1.), tmp, V1, scalar_t(1.), Aij);
            }
            else{
              // U1 * (V1 * Vall)
              DenseMatrix<scalar_t> U1, V1;
              Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
              DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
              gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij);
            }
            /* //Recompress Uall only
            DenseMatrix<scalar_t> U1, V1;
            Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
            DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
            gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
            gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij); */
          }
        }
      }
      else if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::COMB){
        std::vector<std::pair<size_t,size_t>> ranks_idx;
        std::size_t rank_sum=0;
        auto Aij = tile(A21, j, i);
        for (std::size_t k=0; k<kmax; k++) {
          if(!(tile(j, k).is_low_rank() || B11.tile(k, i).is_low_rank())){ //both tiles are dense, then gemm directly
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                      tile(j, k), B11.tile(k, i), scalar_t(1.), Aij);
          } 
          else if(tile(j, k).is_low_rank() && B11.tile(k, i).is_low_rank()){ // both tiles are LR, collect size of LR matrices
            ranks_idx.emplace_back(std::min(tile(j, k).rank(), B11.tile(k, i).rank()),k);
            rank_sum += std::min(tile(j, k).rank(), B11.tile(k, i).rank());
          } 
          else if(tile(j, k).is_low_rank()){ // collect size of LR matrix
            ranks_idx.emplace_back(tile(j, k).rank(),k);
            rank_sum += tile(j, k).rank();
          } 
          else{ // collect size of LR matrix
            ranks_idx.emplace_back(B11.tile(k, i).rank(),k);
            rank_sum += B11.tile(k, i).rank();
          } 
        }
        if(rank_sum>0){
          if(ranks_idx.size()>1){
            //sort ranks in increasing order
            std::sort(ranks_idx.begin(),ranks_idx.end());
            DenseMatrix<scalar_t> tmpU(Aij.rows(), rank_sum);
            DenseMatrix<scalar_t> tmpV(rank_sum, Aij.cols());
            std::size_t rank_tmp=ranks_idx[0].first;
            DenseMatrixWrapper<scalar_t> t1(Aij.rows(), rank_tmp, tmpU, 0, 0), 
            t2(rank_tmp, Aij.cols(), tmpV, 0, 0);
            tile(j,ranks_idx[0].second).multiply(B11.tile(ranks_idx[0].second,i), t1, t2);
            for (std::size_t k=1; k<ranks_idx.size(); k++) {
              DenseMatrix<scalar_t> Uall(tmpU.rows(), rank_tmp+ranks_idx[k].first);
              DenseMatrix<scalar_t> Vall(rank_tmp+ranks_idx[k].first, tmpV.cols());
              Uall.copy_tillpos(tmpU, tmpU.rows(), rank_tmp);
              Vall.copy_tillpos(tmpV, rank_tmp, tmpV.cols());
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), ranks_idx[k].first, Uall, 0, rank_tmp), 
              t2(ranks_idx[k].first, Aij.cols(), Vall, rank_tmp, 0);
              tile(j,ranks_idx[k].second).multiply(B11.tile(ranks_idx[k].second,i), t1, t2);
              if (opts.compression_kernel() == CompressionKernel::FULL){
                //Recompress Uall and Vall
                DenseMatrix<scalar_t> UU, UV;
                Uall.low_rank(UU, UV, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
                DenseMatrix<scalar_t> VU, VV;
                Vall.low_rank(VU, VV, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
                DenseMatrix<scalar_t> tmp1(UV.rows(), VU.cols());
                gemm(Trans::N, Trans::N, scalar_t(1.), UV, VU, scalar_t(0.), tmp1);
                if (UU.cols() > VU.cols()) {
                  //(UU*(UV * VU)) *VV
                  DenseMatrix<scalar_t> tmp2(UU.rows(), tmp1.cols());
                  gemm(Trans::N, Trans::N, scalar_t(1.), UU, tmp1, scalar_t(0.), tmp2);
                  if (k==ranks_idx.size()-1){
                    gemm(Trans::N, Trans::N, scalar_t(-1.), tmp2, VV, scalar_t(1.), Aij);
                  }
                  else{
                    tmpU.copy_topos(tmp2,0,0);
                    tmpV.copy_topos(VV,0,0);
                    rank_tmp = tmp2.cols();
                  }
                }
                else{
                // UU* ((UV * VU)*VV)
                  DenseMatrix<scalar_t> tmp2(tmp1.rows(), VV.cols());
                  gemm(Trans::N, Trans::N, scalar_t(1.), tmp1, VV, scalar_t(0.), tmp2);
                  if (k==ranks_idx.size()-1){
                    gemm(Trans::N, Trans::N, scalar_t(-1.), UU, tmp2, scalar_t(1.), Aij);
                  }
                  else{
                    tmpU.copy_topos(UU,0,0);
                    tmpV.copy_topos(tmp2,0,0);
                    rank_tmp = UU.cols();
                  }
                }
              }
              else{ //Recompress Uall OR Vall
                if (Uall.rows() > Vall.cols()){
                  // (Uall * U1) *V1
                  DenseMatrix<scalar_t> U1, V1;
                  Vall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
                  if (k==ranks_idx.size()-1){
                    DenseMatrix<scalar_t> tmp(Uall.rows(), U1.cols());
                    gemm(Trans::N, Trans::N, scalar_t(1.), Uall, U1, scalar_t(0.), tmp);
                    gemm(Trans::N, Trans::N, scalar_t(-1.), tmp, V1, scalar_t(1.), Aij);
                  }
                  else{
                    DenseMatrixWrapper<scalar_t> t1(Uall.rows(), U1.cols(), tmpU, 0, 0), 
                    t2(V1.rows(), V1.cols(), tmpV, 0, 0);
                    gemm(Trans::N, Trans::N, scalar_t(1.), Uall, U1, scalar_t(0.), t1);
                    t2=V1;
                    rank_tmp = U1.cols();
                  }
                }
                else{
                  // U1 * (V1 * Vall)
                  DenseMatrix<scalar_t> U1, V1;
                  Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
                  if (k==ranks_idx.size()-1){
                    DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
                    gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
                    gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij); 
                  }
                  else{
                    DenseMatrixWrapper<scalar_t> t1(U1.rows(), U1.cols(), tmpU, 0, 0), 
                    t2(V1.rows(), Vall.cols(), tmpV, 0, 0);
                    t1=U1;
                    gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), t2);
                    rank_tmp = U1.cols();
                  }
                }
                /* //Recompress Uall only
                DenseMatrix<scalar_t> U1, V1;
                Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
                DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
                gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
                if (k==ranks_idx.size()-1){
                  gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij); 
                }
                else{
                  tmpU.copy_topos(U1,0,0);
                  tmpV.copy_topos(tmp,0,0);
                  rank_tmp = U1.cols();
                }*/
              }
            }
          }
          else{
            LRTile<scalar_t> tmp=tile(j,ranks_idx[0].second).multiply(B11.tile(ranks_idx[0].second,i));
            gemm(Trans::N, Trans::N, scalar_t(-1.), tmp.U(), tmp.V(), scalar_t(1.), Aij);
          }
        }
      }
    }
    
    template<typename scalar_t> void
    BLRMatrix<scalar_t>::construct_and_partial_factor
    (std::size_t n1, std::size_t n2,
     const extract_t<scalar_t>& A11, const extract_t<scalar_t>& A12,
     const extract_t<scalar_t>& A21, const extract_t<scalar_t>& A22,
     BLRMatrix<scalar_t>& B11, std::vector<int>& piv,
     BLRMatrix<scalar_t>& B12, BLRMatrix<scalar_t>& B21,
     BLRMatrix<scalar_t>& B22,
     const std::vector<std::size_t>& tiles1,
     const std::vector<std::size_t>& tiles2,
     const DenseMatrix<bool>& admissible,
     const BLROptions<scalar_t>& opts) {
      B11 = BLRMatrix<scalar_t>(n1, tiles1, n1, tiles1);
      B12 = BLRMatrix<scalar_t>(n1, tiles1, n2, tiles2);
      B21 = BLRMatrix<scalar_t>(n2, tiles2, n1, tiles1);
      B22 = BLRMatrix<scalar_t>(n2, tiles2, n2, tiles2);
      piv.resize(B11.rows());
      auto rb = B11.rowblocks();
      auto rb2 = B21.rowblocks();
      for (std::size_t i=0; i<rb; i++) {
        B11.create_dense_tile_left_looking(i, i, A11);
        auto tpiv = B11.tile(i, i).LU();
        std::copy(tpiv.begin(), tpiv.end(), piv.begin()+B11.tileroff(i));
        for (std::size_t j=i+1; j<rb; j++) {
          // these blocks have received all updates, compress now
          if (admissible(i, j))
            B11.create_LR_tile_left_looking(i, j, A11, opts);
          else B11.create_dense_tile_left_looking(i, j, A11);
          // permute and solve with L, blocks right from the diagonal block
          std::vector<int> tpiv
            (piv.begin()+B11.tileroff(i), piv.begin()+B11.tileroff(i+1));
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
            (piv.begin()+B11.tileroff(i), piv.begin()+B11.tileroff(i+1));
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
          piv[l] += B11.tileroff(i);
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

    template<typename scalar_t> void LUAR_B22
    (std::size_t i, std::size_t j, std::size_t kmax, BLRMatrix<scalar_t>& B12, 
     BLRMatrix<scalar_t>& B21, DenseMatrix<scalar_t>&A22, const BLROptions<scalar_t>& opts, int* B)
    {
      DenseMatrixWrapper<scalar_t> Aij
        (B21.tilerows(i), B12.tilecols(j), A22,
         B21.tileroff(i), B12.tilecoff(j));
      if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::STAR){
        std::size_t rank_sum=0;
        for (std::size_t k=0; k<kmax; k++) {
          if(!(B21.tile(i, k).is_low_rank() || B12.tile(k, j).is_low_rank())){ //both tiles are dense, then gemm directly
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                 B21.tile(i, k), B12.tile(k, j), scalar_t(1.), Aij);  
          }
          else if(B21.tile(i, k).is_low_rank() && B12.tile(k, j).is_low_rank()){
            rank_sum+=std::min(B21.tile(i, k).rank(), B12.tile(k, j).rank());
          }
          else if(B21.tile(i, k).is_low_rank()){
            rank_sum+=B21.tile(i, k).rank();
          }
          else{
            rank_sum+=B12.tile(k, j).rank();
          }
        }
        if(rank_sum>0){
          DenseMatrix<scalar_t> Uall(Aij.rows(), rank_sum);
          DenseMatrix<scalar_t> Vall(rank_sum, Aij.cols());
          std::size_t rank_tmp=0;
          for (std::size_t k=0; k<kmax; k++) {
            if(B21.tile(i, k).is_low_rank() && B12.tile(k, j).is_low_rank()){
              std::size_t minrank=std::min(B21.tile(i,k).rank(), B12.tile(k,j).rank());
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), minrank, Uall, 0, rank_tmp), 
              t2(minrank, Aij.cols(), Vall, rank_tmp, 0);
              B21.tile(i,k).multiply(B12.tile(k,j), t1, t2);
              rank_tmp+=minrank;
            }
            else if (B21.tile(i, k).is_low_rank()){
              std::size_t minrank=B21.tile(i,k).rank();
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), minrank, Uall, 0, rank_tmp), 
              t2(minrank, Aij.cols(), Vall, rank_tmp, 0);
              B21.tile(i,k).multiply(B12.tile(k,j), t1, t2);
              rank_tmp+=minrank;
            }
            else if(B12.tile(k, j).is_low_rank()){
              std::size_t minrank=B12.tile(k,j).rank();
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), minrank, Uall, 0, rank_tmp), 
              t2(minrank, Aij.cols(), Vall, rank_tmp, 0);
              B21.tile(i,k).multiply(B12.tile(k,j), t1, t2);
              rank_tmp+=minrank;
            }
          }
          if (opts.compression_kernel() == CompressionKernel::FULL){
            //Recompress Uall and Vall
            DenseMatrix<scalar_t> UU, UV;
            Uall.low_rank(UU, UV, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
            DenseMatrix<scalar_t> VU, VV;
            Vall.low_rank(VU, VV, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
            DenseMatrix<scalar_t> tmp1(UV.rows(), VU.cols());
            gemm(Trans::N, Trans::N, scalar_t(1.), UV, VU, scalar_t(0.), tmp1);
            if (UU.cols() > VU.cols()) {
              //(UU*(UV * VU)) *VV
              DenseMatrix<scalar_t> tmp2(UU.rows(), tmp1.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), UU, tmp1, scalar_t(0.), tmp2);
              gemm(Trans::N, Trans::N, scalar_t(-1.), tmp2, VV, scalar_t(1.), Aij);
            }
            else{
              // UU* ((UV * VU)*VV)
              DenseMatrix<scalar_t> tmp2(tmp1.rows(), VV.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), tmp1, VV, scalar_t(0.), tmp2);
              gemm(Trans::N, Trans::N, scalar_t(-1.), UU, tmp2, scalar_t(1.), Aij);
            }
          }
          else{ //Recompress Uall OR Vall
            if (Uall.rows() > Vall.cols()){
              // (Uall * U1) *V1
              DenseMatrix<scalar_t> U1, V1;
              Vall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
              DenseMatrix<scalar_t> tmp(Uall.rows(), U1.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), Uall, U1, scalar_t(0.), tmp);
              gemm(Trans::N, Trans::N, scalar_t(-1.), tmp, V1, scalar_t(1.), Aij);
            }
            else{
              // U1 * (V1 * Vall)
              DenseMatrix<scalar_t> U1, V1;
              Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
              DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
              gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
              gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij);
            }
            /* //Recompress Uall only
            DenseMatrix<scalar_t> U1, V1;
            Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
            DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
            gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
            gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij); */
          }
        }
      }
      else if (opts.BLR_factor_algorithm() == BLRFactorAlgorithm::COMB){
        std::vector<std::pair<size_t,size_t>> ranks_idx;
        std::size_t rank_sum=0;
        for (std::size_t k=0; k<kmax; k++) {
          if(!(B21.tile(i, k).is_low_rank() || B12.tile(k, j).is_low_rank())){ //both tiles are dense, then gemm directly
            gemm(Trans::N, Trans::N, scalar_t(-1.),
                 B21.tile(i, k), B12.tile(k, j), scalar_t(1.), Aij);
          }
          else if(B21.tile(i, k).is_low_rank() && B12.tile(k, j).is_low_rank()){ // both tiles are LR, collect size of LR matrices
            ranks_idx.emplace_back(std::min(B21.tile(i, k).rank(), B12.tile(k, j).rank()),k);
            rank_sum += std::min(B21.tile(i, k).rank(), B12.tile(k, j).rank());
          }
          else if(B21.tile(i, k).is_low_rank()){
            ranks_idx.emplace_back(B21.tile(i, k).rank(),k);
            rank_sum += B21.tile(i, k).rank();
          }
          else{
            ranks_idx.emplace_back(B12.tile(k, j).rank(),k);
            rank_sum += B12.tile(k, j).rank();
          }
        }
        if(rank_sum>0){
          if(ranks_idx.size()>1){
            //sort ranks in increasing order
            std::sort(ranks_idx.begin(),ranks_idx.end());
            DenseMatrix<scalar_t> tmpU(Aij.rows(), rank_sum);
            DenseMatrix<scalar_t> tmpV(rank_sum, Aij.cols());
            std::size_t rank_tmp=ranks_idx[0].first;
            DenseMatrixWrapper<scalar_t> t1(Aij.rows(), rank_tmp, tmpU, 0, 0), 
            t2(rank_tmp, Aij.cols(), tmpV, 0, 0);
            B21.tile(i,ranks_idx[0].second).multiply(B12.tile(ranks_idx[0].second,j), t1, t2);
            for (std::size_t k=1; k<ranks_idx.size(); k++) {
              DenseMatrix<scalar_t> Uall(Aij.rows(), rank_tmp+ranks_idx[k].first);
              DenseMatrix<scalar_t> Vall(rank_tmp+ranks_idx[k].first, Aij.cols());
              Uall.copy_tillpos(tmpU, tmpU.rows(), rank_tmp);
              Vall.copy_tillpos(tmpV, rank_tmp, tmpV.cols());
              DenseMatrixWrapper<scalar_t> t1(Aij.rows(), ranks_idx[k].first, Uall, 0, rank_tmp), 
              t2(ranks_idx[k].first, Aij.cols(), Vall, rank_tmp, 0);
              B21.tile(i,ranks_idx[k].second).multiply(B12.tile(ranks_idx[k].second,j), t1, t2);
              //if (ranks_idx.size()>2){
              if (opts.compression_kernel() == CompressionKernel::FULL){
                //Recompress Uall and Vall
                DenseMatrix<scalar_t> UU, UV;
                Uall.low_rank(UU, UV, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
                DenseMatrix<scalar_t> VU, VV;
                Vall.low_rank(VU, VV, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
                DenseMatrix<scalar_t> tmp1(UV.rows(), VU.cols());
                gemm(Trans::N, Trans::N, scalar_t(1.), UV, VU, scalar_t(0.), tmp1);
                if (UU.cols() > VU.cols()) {
                  //(UU*(UV * VU)) *VV
                  DenseMatrix<scalar_t> tmp2(UU.rows(), tmp1.cols());
                  gemm(Trans::N, Trans::N, scalar_t(1.), UU, tmp1, scalar_t(0.), tmp2);
                  if (k==ranks_idx.size()-1){
                    gemm(Trans::N, Trans::N, scalar_t(-1.), tmp2, VV, scalar_t(1.), Aij);
                  }
                  else{
                    tmpU.copy_topos(tmp2,0,0);
                    tmpV.copy_topos(VV,0,0);
                    rank_tmp = tmp2.cols();
                  }
                }
                else{
                  // UU* ((UV * VU)*VV)
                  DenseMatrix<scalar_t> tmp2(tmp1.rows(), VV.cols());
                  gemm(Trans::N, Trans::N, scalar_t(1.), tmp1, VV, scalar_t(0.), tmp2);
                  if (k==ranks_idx.size()-1){
                    gemm(Trans::N, Trans::N, scalar_t(-1.), UU, tmp2, scalar_t(1.), Aij);
                  }
                  else{
                    tmpU.copy_topos(UU,0,0);
                    tmpV.copy_topos(tmp2,0,0);
                    rank_tmp = UU.cols();
                  }
                }
              }
              else{ //Recompress Uall OR Vall
                if (Uall.rows() > Vall.cols()){
                  // (Uall * U1) *V1
                  DenseMatrix<scalar_t> U1, V1;
                  Vall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Vall.rows(), Vall.cols()), params::task_recursion_cutoff_level);
                  if (k==ranks_idx.size()-1){
                    DenseMatrix<scalar_t> tmp(Uall.rows(), U1.cols());
                    gemm(Trans::N, Trans::N, scalar_t(1.), Uall, U1, scalar_t(0.), tmp);
                    gemm(Trans::N, Trans::N, scalar_t(-1.), tmp, V1, scalar_t(1.), Aij);
                  }
                  else{
                    DenseMatrixWrapper<scalar_t> t1(Uall.rows(), U1.cols(), tmpU, 0, 0), 
                    t2(V1.rows(), V1.cols(), tmpV, 0, 0);
                    gemm(Trans::N, Trans::N, scalar_t(1.), Uall, U1, scalar_t(0.), t1);
                    t2=V1;
                    rank_tmp = U1.cols();
                  }
                }
                else{
                  // U1 * (V1 * Vall)
                  DenseMatrix<scalar_t> U1, V1;
                  Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
                  if (k==ranks_idx.size()-1){
                    DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
                    gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
                    gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij); 
                  }
                  else{
                    DenseMatrixWrapper<scalar_t> t1(U1.rows(), U1.cols(), tmpU, 0, 0), 
                    t2(V1.rows(), Vall.cols(), tmpV, 0, 0);
                    t1=U1;
                    gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), t2);
                    rank_tmp = U1.cols();
                  }
                }
                /* //Recompress Uall only
                DenseMatrix<scalar_t> U1, V1;
                Uall.low_rank(U1, V1, opts.rel_tol(), opts.abs_tol(), std::max(Uall.rows(), Uall.cols()), params::task_recursion_cutoff_level);
                DenseMatrix<scalar_t> tmp(V1.rows(), Vall.cols());
                gemm(Trans::N, Trans::N, scalar_t(1.), V1, Vall, scalar_t(0.), tmp);
                if (k==ranks_idx.size()-1){
                  gemm(Trans::N, Trans::N, scalar_t(-1.), U1, tmp, scalar_t(1.), Aij); 
                }
                else{
                  tmpU.copy_topos(U1,0,0);
                  tmpV.copy_topos(tmp,0,0);
                  rank_tmp = U1.cols();
                }*/
              }
            }
          }
          else{
            LRTile<scalar_t> tmp=B21.tile(i,ranks_idx[0].second).multiply(B12.tile(ranks_idx[0].second,j));
            gemm(Trans::N, Trans::N, scalar_t(-1.), tmp.U(), tmp.V(), scalar_t(1.), Aij);
          }
        }
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
      // TODO threading
      assert(x.cols() == 1);
      assert(y.cols() == 1);
      using DMW_t = DenseMatrixWrapper<scalar_t>;
      if (ta == Trans::N) {
        for (std::size_t i=0; i<a.rowblocks(); i++) {
          DMW_t yi(a.tilerows(i), y.cols(), y, a.tileroff(i), 0);
          for (std::size_t j=0; j<a.colblocks(); j++) {
            DMW_t xj(a.tilecols(j), x.cols(),
                     const_cast<DenseMatrix<scalar_t>&>(x),
                     a.tilecoff(j), 0);
            a.tile(i, j).gemv_a
              (ta, alpha, xj, j==0 ? beta : scalar_t(1.), yi);
          }
        }
      } else {
        for (std::size_t i=0; i<a.colblocks(); i++) {
          DMW_t yi(a.tilecols(i), y.cols(), y, a.tilecoff(i), 0);
          for (std::size_t j=0; j<a.rowblocks(); j++) {
            DMW_t xj(a.tilerows(j), x.cols(),
                     const_cast<DenseMatrix<scalar_t>&>(x),
                     a.tileroff(j), 0);
            a.tile(j, i).gemv_a
              (ta, alpha, xj, j==0 ? beta : scalar_t(1.), yi);
          }
        }
      }
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
      std::cout << "TODO gemm BLR*Dense+Dense" << std::endl;
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
