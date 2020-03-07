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
/*! \file BLRMatrixMPI.hpp
 * \brief Distributed memory block-low rank matrix format.
 */
#ifndef BLR_MATRIX_MPI_HPP
#define BLR_MATRIX_MPI_HPP

#include <cassert>

#include "dense/DistributedMatrix.hpp"
#include "BLRMatrix.hpp"
#include "BLRTile.hpp" // TODO remove

namespace strumpack {
  namespace BLR {

    class ProcessorGrid2D {
    public:
      ProcessorGrid2D(const MPIComm& comm) : comm_(comm) {
        auto P = comm_.size();
        auto rank = comm_.rank();
        npcols_ = std::floor(std::sqrt((float)P));
        nprows_ = P / npcols_;
        prow_ = rank % nprows_;
        pcol_ = rank / nprows_;

        for (int i=0; i<nprows_; i++)
          if (i == prow_) rowcomm_ = comm_.sub(i, npcols_, nprows_);
          else comm_.sub(i, npcols_, nprows_);
        for (int i=0; i<npcols_; i++)
          if (i == pcol_) colcomm_ = comm_.sub(i*nprows_, nprows_, 1);
          else comm_.sub(i*nprows_, nprows_, 1);
      }

      const MPIComm& Comm() const { return comm_; }
      int nprows() const { return nprows_; }
      int npcols() const { return npcols_; }
      int prow() const { return prow_; }
      int pcol() const { return pcol_; }
      int npactives() const { return nprows()*npcols(); }
      int rank() const { return Comm().rank(); }

      const MPIComm& row_comm() const { return rowcomm_; }
      const MPIComm& col_comm() const { return colcomm_; }

      int rowbg2p(int i) const { return i % nprows(); }
      int colbg2p(int j) const { return j / nprows(); }
      int bg2p(int i, int j) const { return rowbg2p(i) + colbg2p(j) * nprows(); }

      int rowbg2l(int i) const { return i / nprows() + i % nprows(); }
      int colbg2l(int j) const { return j / npcols() + j % npcols(); }

    private:
      int prow_ = 0;
      int pcol_ = 0;
      int nprows_ = 0;
      int npcols_ = 0;
      const MPIComm& comm_;
      MPIComm rowcomm_;
      MPIComm colcomm_;
    };


    template<typename scalar_t> class BLRMatrixMPI {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using DistM_t = DistributedMatrix<scalar_t>;
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      using Opts_t = BLROptions<scalar_t>;

    public:
      BLRMatrixMPI() {}

      BLRMatrixMPI(const ProcessorGrid2D& grid,
                   const std::vector<std::size_t>& rowtiles,
                   const std::vector<std::size_t>& coltiles,
                   DistM_t& A, const Opts_t& opts)
        : BLRMatrixMPI<scalar_t>
        (grid, A.rows(), rowtiles, A.cols(), coltiles, opts) {
        // for (std::size_t j=0; j<colblocks(); j++)
        //   for (std::size_t i=0; i<rowblocks(); i++)
        //     block(i, j) = std::unique_ptr<BLRTile<scalar_t>>
        //       (new LRTile<scalar_t>(tile(A, i, j), opts));

        // TODO only create local tiles
      }

      BLRMatrixMPI
      (const ProcessorGrid2D& grid,
       const std::vector<std::size_t>& tiles,
       //const std::function<bool(std::size_t,std::size_t)>& admissible,
       const DenseMatrix<bool>& admissible,
       DistM_t& A, std::vector<int>& piv, const Opts_t& opts)
        : BLRMatrixMPI<scalar_t>
        (grid, A.rows(), tiles, A.cols(), tiles, opts) {
        assert(rowblocks() == colblocks());

        // TODO
      }

      std::size_t rows() const { return m_; }
      std::size_t cols() const { return n_; }

      std::size_t memory() const {
        std::size_t mem = 0;
        for (auto& b : blocks_) mem += b->memory();
        return mem;
      }
      std::size_t nonzeros() const {
        std::size_t nnz = 0;
        for (auto& b : blocks_) nnz += b->nonzeros();
        return nnz;
      }
      std::size_t maximum_rank() const {
        std::size_t mrank = 0;
        for (auto& b : blocks_) mrank = std::max(mrank, b->maximum_rank());
        return mrank;
      }
      // TODO add total_memory, etc

      const ProcessorGrid2D* grid() const { return grid_; }

      void print(const std::string& name) {
        std::cout << "BLR(" << name << ")="
                  << rows() << "x" << cols() << ", "
                  << rowblocks() << "x" << colblocks() << ", "
                  << (float(nonzeros()) / (rows()*cols()) * 100.) << "%"
                  << " [" << std::endl;
        for (std::size_t i=0; i<nbrows_; i++) {
          for (std::size_t j=0; j<nbcols_; j++) {
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

    private:
      std::size_t m_;
      std::size_t n_;
      std::size_t nbrows_;
      std::size_t nbcols_;
      std::size_t nbrowslocal_;
      std::size_t nbcolslocal_;
      std::vector<std::size_t> roff_;
      std::vector<std::size_t> coff_;
      std::vector<std::size_t> rofflocal_;
      std::vector<std::size_t> cofflocal_;
      std::vector<std::unique_ptr<BLRTile<scalar_t>>> blocks_;
      const ProcessorGrid2D* grid_ = nullptr;

      BLRMatrixMPI(const ProcessorGrid2D& grid,
                   std::size_t m, const std::vector<std::size_t>& rowtiles,
                   std::size_t n, const std::vector<std::size_t>& coltiles,
                   const Opts_t& opts) : m_(m), n_(n), grid_(&grid) {
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

        // TODO store local tiles/offsets
      }

      // local???
      inline std::size_t rowblocks() const { return nbrows_; }
      inline std::size_t colblocks() const { return nbcols_; }
      inline std::size_t rowblockslocal() const { return nbrowslocal_; }
      inline std::size_t colblockslocal() const { return nbcolslocal_; }
      inline std::size_t tilerows(std::size_t i) const { return roff_[i+1] - roff_[i]; }
      inline std::size_t tilecols(std::size_t j) const { return coff_[j+1] - coff_[j]; }
      inline std::size_t tilerowoff(std::size_t i) const { return roff_[i]; }
      inline std::size_t tilecoloff(std::size_t j) const { return coff_[j]; }

      inline BLRTile<scalar_t>& tile(std::size_t i, std::size_t j) {
        return *blocks_[i+j*rowblocks()].get();
      }
      inline const BLRTile<scalar_t>& tile(std::size_t i, std::size_t j) const {
        return *blocks_[i+j*rowblocks()].get();
      }
      inline std::unique_ptr<BLRTile<scalar_t>>& block(std::size_t i, std::size_t j) {
        return blocks_[i+j*rowblocks()];
      }
      inline DenseMW_t tile(DenseM_t& A, std::size_t i, std::size_t j) const {
        return DenseMW_t
          (tilerows(i), tilecols(j), A, tilerowoff(i), tilecoloff(j));
      }

      // void create_dense_tile
      // (std::size_t i, std::size_t j, DenseM_t& A) {
      //   block(i, j) = std::unique_ptr<DenseTile<scalar_t>>
      //     (new DenseTile<scalar_t>(tile(A, i, j)));
      // }

      void create_LR_tile
      (std::size_t i, std::size_t j, DenseM_t& A, const Opts_t& opts) {
        block(i, j) = std::unique_ptr<LRTile<scalar_t>>
          (new LRTile<scalar_t>(tile(A, i, j), opts));
        auto& t = tile(i, j);
        if (t.rank()*(t.rows() + t.cols()) > t.rows()*t.cols())
          create_dense_tile(i, j, A);
      }

      // TODO optimize/avoid using this
      void from_block_cyclic(const DistM_t& A) {
        blocks_.resize(rowblockslocal()*colblockslocal());
        for (std::size_t j=0; j<colblocks(); j++)
          for (std::size_t i=0; i<rowblocks(); i++) {
            int dest = grid()->b2p(i, j);
            if (dest == grid()->rank()) {
              auto B = std::unique_ptr<DenseTile<scalar_t>>
                (new DenseTile<scalar_t>(tilerows(i), tilecols(j)));
              copy(tilerows(i), tilecols(j), A, tilerowoff(i), tilecoloff(j),
                   B.D(), dest, A.grid()->ctxt_all());
              block(grid()->rowbg2l(i), grid()->colbg2l(j)) = B;
            } else {
              DenseM_t dummy;
              copy(tilerows(i), tilecols(j), A, tilerowoff(i), tilecoloff(j),
                   dummy, dest, A.grid()->ctxt_all());
            }
          }
      }

      void to_block_cyclic(const DistM_t& A) {
        for (std::size_t j=0; j<colblocks(); j++)
          for (std::size_t i=0; i<rowblocks(); i++) {
            int dest = grid()->b2p(i, j);
            if (dest == grid()->rank()) {
              auto B = block(grid()->rowbg2l(i), grid()->colbg2l(j)).dense();
              copy(tilerows(i), tilecols(j), B, dest,
                   A, tilerowoff(i), tilecoloff(j), A.grid()->ctxt_all());
            } else {
              DenseM_t dummy;
              copy(tilerows(i), tilecols(j), dummy, dest,
                   A, tilerowoff(i), tilecoloff(j), A.grid()->ctxt_all());
            }
          }
      }

      // template<typename T> friend void
      // trsm(Side s, UpLo ul, Trans ta, Diag d, T alpha,
      //      const BLRMatrix<T>& a, BLRMatrix<T>& b, int task_depth);
      // template<typename T> friend void
      // trsm(Side s, UpLo ul, Trans ta, Diag d, T alpha,
      //      const BLRMatrix<T>& a, DenseMatrix<T>& b, int task_depth);
      // template<typename T> friend void
      // gemm(Trans ta, Trans tb, T alpha, const BLRMatrix<T>& a,
      //      const BLRMatrix<T>& b, T beta, DenseMatrix<T>& c, int task_depth);
      // template<typename T> friend void
      // trsv(UpLo ul, Trans ta, Diag d, const BLRMatrix<T>& a,
      //      DenseMatrix<T>& b, int task_depth);
      // template<typename T> friend void
      // gemv(Trans ta, T alpha, const BLRMatrix<T>& a, const DenseMatrix<T>& x,
      //      T beta, DenseMatrix<T>& y, int task_depth);
    };

    template<typename scalar_t> void
    trsm(Side s, UpLo ul, Trans ta, Diag d,
         scalar_t alpha, const BLRMatrixMPI<scalar_t>& a,
         DistributedMatrix<scalar_t>& b) {
      std::cout << "trsm" << std::endl;
    }

    template<typename scalar_t> void
    trsv(UpLo ul, Trans ta, Diag d, const BLRMatrixMPI<scalar_t>& a,
         DistributedMatrix<scalar_t>& b) {
      std::cout << "trsv" << std::endl;
    }

    template<typename scalar_t> void
    gemv(Trans ta, scalar_t alpha, const BLRMatrixMPI<scalar_t>& a,
         const DistributedMatrix<scalar_t>& x, scalar_t beta,
         DistributedMatrix<scalar_t>& y) {
      std::cout << "gemv" << std::endl;
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const BLRMatrixMPI<scalar_t>& a,
         const BLRMatrixMPI<scalar_t>& b, scalar_t beta,
         DistributedMatrix<scalar_t>& c) {
      std::cout << "gemm" << std::endl;
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const BLRMatrixMPI<scalar_t>& A,
         const DistributedMatrix<scalar_t>& B, scalar_t beta,
         DistributedMatrix<scalar_t>& C) {
      std::cout << "TODO gemm BLR*DistM+DistM" << std::endl;
    }


  } // end namespace BLR
} // end namespace strumpack

#endif // BLR_MATRIX_MPI_HPP
