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
/*! \file BLRMatrix.hpp
 * \brief For Pieter to complete
 */
#ifndef BLR_MATRIX_HPP
#define BLR_MATRIX_HPP

#include <cassert>

#include "../dense/DenseMatrix.hpp"
#include "BLROptions.hpp"

namespace strumpack {
  namespace BLR {

    // forward declarations
    template<typename scalar_t> class LRTile;
    template<typename scalar_t> class DenseTile;


    template<typename scalar_t> class BLRTile {
      using DenseM_t = DenseMatrix<scalar_t>;

    public:
      virtual std::size_t rows() const = 0;
      virtual std::size_t cols() const = 0;
      virtual std::size_t rank() const = 0;

      virtual std::size_t memory() const = 0;
      virtual std::size_t nonzeros() const = 0;
      virtual std::size_t maximum_rank() const = 0;
      virtual bool is_low_rank() const = 0;
      virtual void dense(DenseM_t& A) const = 0;

      virtual void draw
      (std::ostream& of, std::size_t roff, std::size_t coff) const = 0;

      virtual DenseM_t& D() = 0; //{ assert(false); }
      virtual DenseM_t& U() = 0; //{ assert(false); }
      virtual DenseM_t& V() = 0; //{ assert(false); }
      virtual const DenseM_t& D() const = 0; //{ assert(false); }
      virtual const DenseM_t& U() const = 0; //{ assert(false); }
      virtual const DenseM_t& V() const = 0; //{ assert(false); }

      virtual std::vector<int> LU() { assert(false); return std::vector<int>(); };
      virtual void laswp(const std::vector<int>& piv, bool fwd) = 0;

      virtual void trsm_b(Side s, UpLo ul, Trans ta, Diag d,
                          scalar_t alpha, const DenseM_t& a) = 0;
      virtual void gemv_a(Trans ta, scalar_t alpha, const DenseM_t& x,
                          scalar_t beta, DenseM_t& y) const = 0;
      virtual void gemm_a(Trans ta, Trans tb, scalar_t alpha,
                          const BLRTile<scalar_t>& b, scalar_t beta,
                          DenseM_t& c) const = 0;
      virtual void gemm_a(Trans ta, Trans tb, scalar_t alpha,
                          const DenseM_t& b, scalar_t beta,
                          DenseM_t& c, int task_depth) const = 0;
      virtual void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                          const LRTile<scalar_t>& a, scalar_t beta,
                          DenseM_t& c) const = 0;
      virtual void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                          const DenseTile<scalar_t>& a, scalar_t beta,
                          DenseM_t& c) const = 0;
      virtual void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                          const DenseM_t& a, scalar_t beta,
                          DenseM_t& c, int task_depth) const = 0;
    };


    template<typename scalar_t> class DenseTile
      : public BLRTile<scalar_t> {
      using DenseM_t = DenseMatrix<scalar_t>;
      using BLRT_t = BLRTile<scalar_t>;

    public:
      DenseTile(std::size_t m, std::size_t n) : D_(m, n) {}
      DenseTile(const DenseM_t& D) : D_(D) {}

      std::size_t rows() const override { return D_.rows(); }
      std::size_t cols() const override { return D_.cols(); }
      std::size_t rank() const override { return std::min(rows(), cols()); }

      std::size_t memory() const override { return D_.memory(); }
      std::size_t nonzeros() const override { return D_.nonzeros(); }
      std::size_t maximum_rank() const override { return 0; }
      bool is_low_rank() const override { return false; };

      void dense(DenseM_t& A) const override { A = D_; }

      void draw
      (std::ostream& of, std::size_t roff, std::size_t coff) const override {
        char prev = std::cout.fill('0');
        of << "set obj rect from "
           << roff << ", " << coff << " to "
           << roff+rows() << ", " << coff+cols()
           << " fc rgb '#FF0000'" << std::endl;
        std::cout.fill(prev);
      }

      DenseM_t& D() override { return D_; }
      const DenseM_t& D() const override { return D_; }

      DenseM_t& U() override { assert(false); return D_; }
      DenseM_t& V() override { assert(false); return D_; }
      const DenseM_t& U() const override { assert(false); return D_; }
      const DenseM_t& V() const override { assert(false); return D_; }

      std::vector<int> LU() override {
        return D_.LU(params::task_recursion_cutoff_level);
      }
      void laswp(const std::vector<int>& piv, bool fwd) override {
        D_.laswp(piv, fwd);
      }

      void trsm_b(Side s, UpLo ul, Trans ta, Diag d,
                  scalar_t alpha, const DenseM_t& a) override {
        trsm(s, ul, ta, d, alpha, a, D_, params::task_recursion_cutoff_level);
      }
      void gemv_a(Trans ta, scalar_t alpha, const DenseM_t& x,
                  scalar_t beta, DenseM_t& y) const override {
        gemv(ta, alpha, D_, x, beta, y,
             params::task_recursion_cutoff_level);
      }
      void gemm_a(Trans ta, Trans tb, scalar_t alpha, const BLRT_t& b,
                  scalar_t beta, DenseM_t& c) const override {
        b.gemm_b(ta, tb, alpha, *this, beta, c);
      }
      void gemm_a(Trans ta, Trans tb, scalar_t alpha,
                  const DenseM_t& b, scalar_t beta,
                  DenseM_t& c, int task_depth) const override {
        gemm(ta, tb, alpha, D_, b, beta, c, task_depth);
      }
      void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                  const LRTile<scalar_t>& a, scalar_t beta,
                  DenseM_t& c) const override {
        DenseM_t tmp(a.rank(), tb==Trans::N ? cols() : rows());
        gemm(ta, tb, scalar_t(1.), ta==Trans::N ? a.V() : a.U(), D_,
             scalar_t(0.), tmp, params::task_recursion_cutoff_level);
        gemm(ta, Trans::N, alpha, ta==Trans::N ? a.U() : a.V(), tmp,
             beta, c, params::task_recursion_cutoff_level);
      }
      void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                  const DenseTile<scalar_t>& a, scalar_t beta,
                  DenseM_t& c) const override {
        gemm(ta, tb, alpha, a.D(), D(), beta, c);
      }
      void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                  const DenseM_t& a, scalar_t beta,
                  DenseM_t& c, int task_depth) const override {
        gemm(ta, tb, alpha, a, D_, beta, c, task_depth);
      }

    private:
      DenseM_t D_;
    };


    /**
     * Low rank U*V tile
     */
    template<typename scalar_t> class LRTile
      : public BLRTile<scalar_t> {
      using DenseM_t = DenseMatrix<scalar_t>;
      using Opts_t = BLROptions<scalar_t>;

    public:
      LRTile(const DenseM_t& T, const Opts_t& opts) {
        if (opts.low_rank_algorithm() == LowRankAlgorithm::RRQR) {
          T.low_rank(U_, V_, opts.rel_tol(), opts.abs_tol(), opts.max_rank(),
                     params::task_recursion_cutoff_level);
        } else if (opts.low_rank_algorithm() == LowRankAlgorithm::ACA) {
          std::cerr << "ERROR: ACA is currently not supported." << std::endl;
          // DenseM_t Vt;
          // adaptive_cross_approximation<scalar_t>
          //   (U_, Vt, T.rows(), T.cols(),
          //    [&](std::size_t i, std::size_t j)->scalar_t { return T(i, j); },
          //    opts.rel_tol(), opts.abs_tol(), opts.max_rank());
          // V_ = Vt.transpose();
        }
      }

      std::size_t rows() const { return U_.rows(); }
      std::size_t cols() const { return V_.cols(); }
      std::size_t rank() const { return U_.cols(); }
      bool is_low_rank() const override { return true; };

      std::size_t memory() const override { return U_.memory() + V_.memory(); }
      std::size_t nonzeros() const override { return U_.nonzeros() + V_.nonzeros(); }
      std::size_t maximum_rank() const override { return U_.cols(); }

      void dense(DenseM_t& A) const override {
        gemm(Trans::N, Trans::N, scalar_t(1.), U_, V_, scalar_t(0.), A,
             params::task_recursion_cutoff_level);
      }

      void draw
      (std::ostream& of, std::size_t roff, std::size_t coff) const override {
        char prev = std::cout.fill('0');
        int minmn = std::min(rows(), cols());
        int red = std::floor(255.0 * rank() / minmn);
        int blue = 255 - red;
        of << "set obj rect from "
           << roff << ", " << coff << " to "
           << roff+rows() << ", " << coff+cols()
           << " fc rgb '#"
           << std::hex << std::setw(2) << std::setfill('0') << red
           << "00" << std::setw(2)  << std::setfill('0') << blue
           << "'" << std::dec << std::endl;
        std::cout.fill(prev);
      }

      DenseM_t& D() override { assert(false); return U_; }
      DenseM_t& U() override { return U_; }
      DenseM_t& V() override { return V_; }
      const DenseM_t& D() const override { assert(false); return U_; }
      const DenseM_t& U() const override { return U_; }
      const DenseM_t& V() const override { return V_; }

      void laswp(const std::vector<int>& piv, bool fwd) override {
        U_.laswp(piv, fwd);
      }

      void trsm_b(Side s, UpLo ul, Trans ta, Diag d,
                  scalar_t alpha, const DenseM_t& a) override {
        strumpack::trsm
          (s, ul, ta, d, alpha, a, (s == Side::L) ? U_ : V_,
           params::task_recursion_cutoff_level);
      }
      void gemv_a(Trans ta, scalar_t alpha, const DenseM_t& x,
                  scalar_t beta, DenseM_t& y) const override {
        DenseM_t tmp(rank(), x.cols());
        gemv(ta, scalar_t(1.), ta==Trans::N ? V() : U(), x, scalar_t(0.), tmp,
             params::task_recursion_cutoff_level);
        gemv(ta, alpha, ta==Trans::N ? U() : V(), tmp, beta, y,
             params::task_recursion_cutoff_level);
      }
      void gemm_a(Trans ta, Trans tb, scalar_t alpha,
                  const BLRTile<scalar_t>& b,
                  scalar_t beta, DenseM_t& c) const override {
        b.gemm_b(ta, tb, alpha, *this, beta, c);
      }
      void gemm_a(Trans ta, Trans tb, scalar_t alpha,
                  const DenseM_t& b, scalar_t beta,
                  DenseM_t& c, int task_depth) const override {
        DenseM_t tmp(rank(), c.cols());
        gemm(ta, tb, scalar_t(1.), ta==Trans::N ? V() : U(), b,
             scalar_t(0.), tmp, task_depth);
        gemm(ta, Trans::N, alpha, ta==Trans::N ? U() : V(), tmp,
             beta, c, task_depth);
      }
      void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                  const LRTile<scalar_t>& a, scalar_t beta,
                  DenseM_t& c) const override {
        DenseM_t tmp1(a.rank(), rank());
        gemm(ta, tb, scalar_t(1.), ta==Trans::N ? a.V() : a.U(),
             tb==Trans::N ? U() : V(), scalar_t(0.), tmp1,
             params::task_recursion_cutoff_level);
        DenseM_t tmp2(c.rows(), tmp1.cols());
        gemm(ta, Trans::N, scalar_t(1.), ta==Trans::N ? a.U() : a.V(), tmp1,
             scalar_t(0.), tmp2, params::task_recursion_cutoff_level);
        gemm(Trans::N, tb, alpha, tmp2, tb==Trans::N ? V() : U(),
             beta, c, params::task_recursion_cutoff_level);
      }
      void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                  const DenseTile<scalar_t>& a, scalar_t beta,
                  DenseM_t& c) const override {
        gemm_b(ta, tb, alpha, a.D(), beta, c,
               params::task_recursion_cutoff_level);
      }
      void gemm_b(Trans ta, Trans tb, scalar_t alpha,
                  const DenseM_t& a, scalar_t beta,
                  DenseM_t& c, int task_depth) const override {
        DenseM_t tmp(c.rows(), rank());
        gemm(ta, tb, scalar_t(1.), a, tb==Trans::N ? U() : V(),
             scalar_t(0.), tmp, task_depth);
        gemm(Trans::N, tb, alpha, tmp, tb==Trans::N ? V() : U(),
             beta, c, task_depth);
      }

    private:
      DenseM_t U_;
      DenseM_t V_;
    };


    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const BLRTile<scalar_t>& a,
         const BLRTile<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c) {
      a.gemm_a(ta, tb, alpha, b, beta, c);
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const DenseMatrix<scalar_t>& a,
         const BLRTile<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c, int task_depth) {
      b.gemm_b(ta, tb, alpha, a, beta, c);
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha, const BLRTile<scalar_t>& a,
         const DenseMatrix<scalar_t>& b, scalar_t beta,
         DenseMatrix<scalar_t>& c, int task_depth) {
      a.gemm_a(ta, tb, alpha, b, beta, c, task_depth);
    }

    template<typename scalar_t> void
    trsm(Side s, UpLo ul, Trans ta, Diag d, scalar_t alpha,
         const BLRTile<scalar_t>& a, BLRTile<scalar_t>& b) {
      b.trsm_b(s, ul, ta, d, alpha, a.D());
    }

    template<typename scalar_t> void
    trsm(Side s, UpLo ul, Trans ta, Diag d, scalar_t alpha,
         const BLRTile<scalar_t>& a, DenseMatrix<scalar_t>& b,
         int task_depth) {
      trsm(s, ul, ta, d, alpha, a.D(), b, task_depth);
    }


    template<typename scalar_t> class BLRMatrix {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using Opts_t = BLROptions<scalar_t>;

    public:
      BLRMatrix() {}

      BLRMatrix(DenseM_t& A,
                const std::vector<std::size_t>& rowtiles,
                const std::vector<std::size_t>& coltiles,
                const Opts_t& opts)
        : BLRMatrix<scalar_t>(A.rows(), rowtiles, A.cols(), coltiles) {
        for (std::size_t j=0; j<colblocks(); j++)
          for (std::size_t i=0; i<rowblocks(); i++)
            block(i, j) = std::unique_ptr<BLRTile<scalar_t>>
              (new LRTile<scalar_t>(tile(A, i, j), opts));
      }

      BLRMatrix(const std::vector<std::size_t>& tiles,
                const std::function<bool(std::size_t,std::size_t)>& admissible,
                DenseM_t& A, std::vector<int>& piv, const Opts_t& opts)
        : BLRMatrix<scalar_t>(A.rows(), tiles, A.cols(), tiles) {
        assert(rowblocks() == colblocks());
        piv.resize(rows());
        auto rb = rowblocks();
        auto B = new int[rb*rb](); // dummy for task synchronization
#pragma omp taskgroup
        {
          for (std::size_t i=0; i<rowblocks(); i++) {
#pragma omp task default(shared) depend(inout:B[i+rb*i])
            {
              create_dense_tile(i, i, A);
              auto tpiv = tile(i, i).LU();
              std::copy(tpiv.begin(), tpiv.end(), piv.begin()+tileroff(i));
            }
            for (std::size_t j=i+1; j<rowblocks(); j++) {
#pragma omp task default(shared) depend(in:B[i+rb*i]) depend(inout:B[i+rb*j])
              { // these blocks have received all updates, compress now
                if (admissible(i, j)) create_LR_tile(i, j, A, opts);
                else create_dense_tile(i, j, A);
                // permute and solve with L, blocks right from the diagonal block
                std::vector<int> tpiv
                  (piv.begin()+tileroff(i), piv.begin()+tileroff(i+1));
                tile(i, j).laswp(tpiv, true);
                trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                     scalar_t(1.), tile(i, i), tile(i, j));
              }
#pragma omp task default(shared) depend(in:B[i+rb*i]) depend(inout:B[j+rb*i])
              {
                if (admissible(j, i)) create_LR_tile(j, i, A, opts);
                else create_dense_tile(j, i, A);
                // solve with U, the blocks under the diagonal block
                trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                     scalar_t(1.), tile(i, i), tile(j, i));
              }
            }
            for (std::size_t j=i+1; j<colblocks(); j++)
              for (std::size_t k=i+1; k<rowblocks(); k++) {
#pragma omp task default(shared) depend(in:B[i+rb*j],B[k+rb*i]) depend(inout:B[k+rb*j])
                { // Schur complement updates, always into full rank
                  DenseMW_t Akj = tile(A, k, j);
                  gemm(Trans::N, Trans::N, scalar_t(-1.),
                       tile(k, i), tile(i, j), scalar_t(1.), Akj);
                }
              }
          }
        }
        for (std::size_t i=0; i<rowblocks(); i++)
          for (std::size_t l=tileroff(i); l<tileroff(i+1); l++)
            piv[l] += tileroff(i);
        delete[] B;
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

      DenseM_t dense() const {
        DenseM_t A(rows(), cols());
#pragma omp taskloop collapse(2) default(shared)
        for (std::size_t j=0; j<colblocks(); j++)
          for (std::size_t i=0; i<rowblocks(); i++) {
            DenseMW_t Aij = tile(A, i, j);
            tile(i, j).dense(Aij);
          }
        return A;
      }

      void draw(std::ostream& of, std::size_t roff, std::size_t coff) const {
#pragma omp taskloop collapse(2) default(shared)
        for (std::size_t j=0; j<colblocks(); j++)
          for (std::size_t i=0; i<rowblocks(); i++) {
            tile(i, j).draw(of, roff+tileroff(i), coff+tilecoff(j));
          }
      }

      void solve(std::vector<int>& piv, const DenseM_t& b) const {
        // TODO test this
        // b.laswp(piv, true);
        // trsm();
        // trsm();
      }

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
      std::vector<std::size_t> roff_;
      std::vector<std::size_t> coff_;
      std::vector<std::unique_ptr<BLRTile<scalar_t>>> blocks_;

      BLRMatrix(std::size_t m, const std::vector<std::size_t>& rowtiles,
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

      inline std::size_t rowblocks() const { return nbrows_; }
      inline std::size_t colblocks() const { return nbcols_; }
      inline std::size_t tilerows(std::size_t i) const { return roff_[i+1] - roff_[i]; }
      inline std::size_t tilecols(std::size_t j) const { return coff_[j+1] - coff_[j]; }
      inline std::size_t tileroff(std::size_t i) const { return roff_[i]; }
      inline std::size_t tilecoff(std::size_t j) const { return coff_[j]; }

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
          (tilerows(i), tilecols(j), A, tileroff(i), tilecoff(j));
      }

      void create_dense_tile
      (std::size_t i, std::size_t j, DenseM_t& A) {
        block(i, j) = std::unique_ptr<DenseTile<scalar_t>>
          (new DenseTile<scalar_t>(tile(A, i, j)));
      }

      void create_LR_tile
      (std::size_t i, std::size_t j, DenseM_t& A, const Opts_t& opts) {
        block(i, j) = std::unique_ptr<LRTile<scalar_t>>
          (new LRTile<scalar_t>(tile(A, i, j), opts));
        auto& t = tile(i, j);
        if (t.rank()*(t.rows() + t.cols()) > t.rows()*t.cols())
          create_dense_tile(i, j, A);
      }

      template<typename T> friend void
      trsm(Side s, UpLo ul, Trans ta, Diag d, T alpha,
           const BLRMatrix<T>& a, BLRMatrix<T>& b, int task_depth);
      template<typename T> friend void
      trsm(Side s, UpLo ul, Trans ta, Diag d, T alpha,
           const BLRMatrix<T>& a, DenseMatrix<T>& b, int task_depth);
      template<typename T> friend void
      gemm(Trans ta, Trans tb, T alpha, const BLRMatrix<T>& a,
           const BLRMatrix<T>& b, T beta, DenseMatrix<T>& c, int task_depth);
      template<typename T> friend void
      trsv(UpLo ul, Trans ta, Diag d, const BLRMatrix<T>& a,
           DenseMatrix<T>& b, int task_depth);
      template<typename T> friend void
      gemv(Trans ta, T alpha, const BLRMatrix<T>& a, const DenseMatrix<T>& x,
           T beta, DenseMatrix<T>& y, int task_depth);

      template<typename T> friend void
      BLR_construct_and_partial_factor
      (DenseMatrix<T>& A11, DenseMatrix<T>& A12, DenseMatrix<T>& A21, DenseMatrix<T>& A22,
       BLRMatrix<T>& B11, std::vector<int>& piv, BLRMatrix<T>& B12, BLRMatrix<T>& B21,
       const std::vector<std::size_t>& tiles1, const std::vector<std::size_t>& tiles2,
       const std::function<bool(std::size_t,std::size_t)>& admissible,
       const BLROptions<T>& opts);

      template<typename T> friend void
      BLR_trsmLNU_gemm
      (const BLRMatrix<T>& F1, const BLRMatrix<T>& F2,
       DenseMatrix<T>& B1, DenseMatrix<T>& B2, int task_depth);
      template<typename T> friend void
      BLR_gemm_trsmUNN
      (const BLRMatrix<T>& F1, const BLRMatrix<T>& F2,
       DenseMatrix<T>& B1, DenseMatrix<T>& B2, int task_depth);
    };

    template<typename scalar_t> inline void
    BLR_construct_and_partial_factor
    (DenseMatrix<scalar_t>& A11, DenseMatrix<scalar_t>& A12,
     DenseMatrix<scalar_t>& A21, DenseMatrix<scalar_t>& A22,
     BLRMatrix<scalar_t>& B11, std::vector<int>& piv,
     BLRMatrix<scalar_t>& B12, BLRMatrix<scalar_t>& B21,
     const std::vector<std::size_t>& tiles1,
     const std::vector<std::size_t>& tiles2,
     const std::function<bool(std::size_t,std::size_t)>& admissible,
     const BLROptions<scalar_t>& opts) {
      B11 = BLRMatrix<scalar_t>(A11.rows(), tiles1, A11.cols(), tiles1);
      B12 = BLRMatrix<scalar_t>(A12.rows(), tiles1, A12.cols(), tiles2);
      B21 = BLRMatrix<scalar_t>(A21.rows(), tiles2, A21.cols(), tiles1);

      piv.resize(B11.rows());
      auto rb = B11.rowblocks();
      auto rb2 = B21.rowblocks();
      auto lrb = rb+rb2;
      auto B = new int[lrb*lrb](); // dummy for task synchronization
#pragma omp taskgroup
      {
        for (std::size_t i=0; i<rb; i++) {
#pragma omp task default(shared) depend(inout:B[i+lrb*i])
          {
            B11.create_dense_tile(i, i, A11);
            auto tpiv = B11.tile(i, i).LU();
            std::copy(tpiv.begin(), tpiv.end(), piv.begin()+B11.tileroff(i));
          }
          for (std::size_t j=i+1; j<rb; j++) {
#pragma omp task default(shared) depend(in:B[i+lrb*i])  \
  depend(inout:B[i+lrb*j]) priority(rb-j)
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
#pragma omp task default(shared) depend(in:B[i+lrb*i])  \
  depend(inout:B[j+lrb*i]) priority(rb-j)
            {
              if (admissible(j, i)) B11.create_LR_tile(j, i, A11, opts);
              else B11.create_dense_tile(j, i, A11);
              // solve with U, the blocks under the diagonal block
              trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                   scalar_t(1.), B11.tile(i, i), B11.tile(j, i));
            }
          }
          for (std::size_t j=0; j<rb2; j++) {
#pragma omp task default(shared) depend(in:B[i+lrb*i])  \
  depend(inout:B[i+lrb*(rb+j)])
            {
              B12.create_LR_tile(i, j, A12, opts);
              // permute and solve with L  blocks right from the diagonal block
              std::vector<int> tpiv
                (piv.begin()+B11.tileroff(i), piv.begin()+B11.tileroff(i+1));
              B12.tile(i, j).laswp(tpiv, true);
              trsm(Side::L, UpLo::L, Trans::N, Diag::U,
                   scalar_t(1.), B11.tile(i, i), B12.tile(i, j));
            }
#pragma omp task default(shared) depend(in:B[i+lrb*i])  \
  depend(inout:B[(rb+j)+lrb*i])
            {
              B21.create_LR_tile(j, i, A21, opts);
              // solve with U, the blocks under the diagonal block
              trsm(Side::R, UpLo::U, Trans::N, Diag::N,
                   scalar_t(1.), B11.tile(i, i), B21.tile(j, i));
            }
          }
          for (std::size_t j=i+1; j<rb; j++)
            for (std::size_t k=i+1; k<rb; k++) {
              //std::size_t jk = std::min(j,k);
#pragma omp task default(shared) depend(in:B[i+lrb*j],B[k+lrb*i])       \
  depend(inout:B[k+lrb*j]) priority(rb-j)
              { // Schur complement updates, always into full rank
                auto Akj = B11.tile(A11, k, j);
                gemm(Trans::N, Trans::N, scalar_t(-1.),
                     B11.tile(k, i), B11.tile(i, j), scalar_t(1.), Akj);
              }
            }
          for (std::size_t k=i+1; k<rb; k++)
            for (std::size_t j=0; j<rb2; j++) {
#pragma omp task default(shared) depend(in:B[k+lrb*i],B[i+lrb*(rb+j)])  \
  depend(inout:B[k+lrb*(rb+j)])
              { // Schur complement updates, always into full rank
                auto Akj = B12.tile(A12, k, j);
                gemm(Trans::N, Trans::N, scalar_t(-1.),
                     B11.tile(k, i), B12.tile(i, j), scalar_t(1.), Akj);
              }
#pragma omp task default(shared) depend(in:B[i+lrb*k],B[(j+rb)+lrb*i]) \
  depend(inout:B[(rb+j)+lrb*k])
              { // Schur complement updates, always into full rank
                auto Ajk = B21.tile(A21, j, k);
                gemm(Trans::N, Trans::N, scalar_t(-1.),
                     B21.tile(j, i), B11.tile(i, k), scalar_t(1.), Ajk);
              }
            }

          for (std::size_t j=0; j<rb2; j++)
            for (std::size_t k=0; k<rb2; k++) {
#pragma omp task default(shared) depend(in:B[i+lrb*(rb+j)],B[(rb+k)+lrb*i]) \
  depend(inout:B[(rb+k)+lrb*(rb+j)])
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
      for (std::size_t i=0; i<rb; i++)
        for (std::size_t l=B11.tileroff(i); l<B11.tileroff(i+1); l++)
          piv[l] += B11.tileroff(i);
      A11.clear();
      A12.clear();
      A21.clear();
      delete[] B;
    }

    template<typename scalar_t> void BLR_trsmLNU_gemm
    (const BLRMatrix<scalar_t>& F1, const BLRMatrix<scalar_t>& F2,
     DenseMatrix<scalar_t>& B1, DenseMatrix<scalar_t>& B2, int task_depth) {
      using DMW_t = DenseMatrixWrapper<scalar_t>;
      if (B1.cols() == 1) {
        auto rb = F1.rowblocks();
        auto rb2 = F2.rowblocks();
        auto lrb = rb+rb2;
        auto B = new int[lrb];
#pragma omp taskgroup
        {
          for (std::size_t i=0; i<rb; i++) {
#pragma omp task default(shared) depend(inout:B[i])
            {
              DMW_t Bi(F1.tilerows(i), B1.cols(), B1, F1.tileroff(i), 0);
              trsv(UpLo::L, Trans::N, Diag::U, F1.tile(i, i).D(), Bi,
                   params::task_recursion_cutoff_level);
            }
            for (std::size_t j=i+1; j<rb; j++) {
#pragma omp task default(shared) depend(in:B[i]) depend(inout:B[j]) priority(rb-i)
              {
                DMW_t Bi(F1.tilerows(i), B1.cols(), B1, F1.tileroff(i), 0);
                DMW_t Bj(F1.tilerows(j), B1.cols(), B1, F1.tileroff(j), 0);
                F1.tile(j, i).gemv_a(Trans::N, scalar_t(-1.), Bi, scalar_t(1.), Bj);
              }
            }
            for (std::size_t j=0; j<rb2; j++) {
#pragma omp task default(shared) depend(in:B[i]) depend(inout:B[rb+j]) priority(0)
              {
                DMW_t Bi(F1.tilerows(i), B1.cols(), B1, F1.tileroff(i), 0);
                DMW_t Bj(F2.tilerows(j), B2.cols(), B2, F2.tileroff(j), 0);
                F2.tile(j, i).gemv_a(Trans::N, scalar_t(-1.), Bi, scalar_t(1.), Bj);
              }
            }
          }
        }
        delete[] B;
      } else {
        // TODO optimize by merging
        trsm(Side::L, UpLo::L, Trans::N, Diag::U,
             scalar_t(1.), F1, B1, task_depth);
        gemm(Trans::N, Trans::N, scalar_t(-1.), F2, B1,
             scalar_t(1.), B2, task_depth);
      }
    }

    template<typename scalar_t> void BLR_gemm_trsmUNN
    (const BLRMatrix<scalar_t>& F1, const BLRMatrix<scalar_t>& F2,
     DenseMatrix<scalar_t>& B1, DenseMatrix<scalar_t>& B2, int task_depth) {
      using DMW_t = DenseMatrixWrapper<scalar_t>;
      if (B1.cols() == 1) {
        auto rb = F1.colblocks();
        auto rb2 = F2.colblocks();
        auto lrb = rb+rb2;
        auto B = new int[lrb];
#pragma omp taskgroup
        {
          for (std::size_t i=rb; i --> 0; ) {
            assert(i < rb);
            for (std::size_t j=0; j<rb2; j++)
#pragma omp task default(shared) depend(in:B[rb+j]) depend(inout:B[i]) priority(1)
              {
                DMW_t Bi(F1.tilerows(i), B1.cols(), B1, F1.tileroff(i), 0);
                DMW_t Bj(F2.tilecols(j), B2.cols(), B2, F2.tilecoff(j), 0);
                F2.tile(i, j).gemv_a(Trans::N, scalar_t(-1.), Bj, scalar_t(1.), Bi);
              }
            for (std::size_t j=i+1; j<rb; j++)
#pragma omp task default(shared) depend(in:B[j]) depend(inout:B[i]) priority(1)
              {
                DMW_t Bi(F1.tilerows(i), B1.cols(), B1, F1.tileroff(i), 0);
                DMW_t Bj(F1.tilecols(j), B1.cols(), B1, F1.tilecoff(j), 0);
                F1.tile(i, j).gemv_a(Trans::N, scalar_t(-1.), Bj, scalar_t(1.), Bi);
              }
#pragma omp task default(shared) depend(inout:B[i]) priority(0)
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
      } else if (s == Side::L && ul == UpLo::L) {
        for (std::size_t j=0; j<a.colblocks(); j++) {
          DMW_t bj(a.tilecols(j), b.cols(), b, a.tilecoff(j), 0);
          for (std::size_t k=0; k<j; k++)
            gemm(ta, Trans::N, scalar_t(-1.),
                 ta==Trans::N ? a.tile(j, k) : a.tile(k, j),
                 DMW_t(a.tilecols(k), b.cols(), b, a.tilecoff(k), 0),
                 scalar_t(1.), bj, task_depth);
          trsm(s, ul, ta, d, alpha, a.tile(j, j), bj, task_depth);
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
#pragma omp taskloop collapse(2) default(shared)
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

  } // end namespace BLR
} // end namespace strumpack

#endif // BLR_MATRIX_HPP
