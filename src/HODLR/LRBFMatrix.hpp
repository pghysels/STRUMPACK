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
/*! \file LRBFMatrix.hpp
 * \brief Classes wrapping around Yang Liu's butterfly code.
 */
#ifndef STRUMPACK_LRBF_MATRIX_HPP
#define STRUMPACK_LRBF_MATRIX_HPP

#include <cassert>

#include "HSS/HSSPartitionTree.hpp"
#include "dense/DistributedMatrix.hpp"
#include "HODLROptions.hpp"
#include "HODLRWrapper.hpp"

namespace strumpack {

  /**
   * Code in this namespace is a wrapper aroung Yang Liu's Fortran
   * code: https://github.com/liuyangzhuan/hod-lr-bf
   */
  namespace HODLR {

    template<typename scalar_t> class LRBFMatrix {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DistM_t = DistributedMatrix<scalar_t>;
      using opts_t = HODLROptions<scalar_t>;
      using mult_t = typename std::function<
        void(Trans,scalar_t,const DenseM_t&,scalar_t,DenseM_t&)>;

    public:
      LRBFMatrix() {}
      /**
       * Construct the block X, subblock of the matrix [A X; Y B]
       * A and B should be defined on the same MPI communicator.
       */
      LRBFMatrix
      (const HODLRMatrix<scalar_t>& A, const HODLRMatrix<scalar_t>& B);

      LRBFMatrix(const LRBFMatrix<scalar_t>& h) = delete;
      LRBFMatrix(LRBFMatrix<scalar_t>&& h) { *this = h; }
      virtual ~LRBFMatrix();
      LRBFMatrix<scalar_t>& operator=(const LRBFMatrix<scalar_t>& h) = delete;
      LRBFMatrix<scalar_t>& operator=(LRBFMatrix<scalar_t>&& h);

      std::size_t rows() const { return rows_; }
      std::size_t cols() const { return cols_; }
      std::size_t lrows() const { return lrows_; }
      std::size_t lcols() const { return lcols_; }
      std::size_t begin_row() const { return rdist_[c_->rank()]; }
      std::size_t end_row() const { return rdist_[c_->rank()+1]; }
      const std::vector<int>& rdist() const { return rdist_; }
      std::size_t begin_col() const { return cdist_[c_->rank()]; }
      std::size_t end_col() const { return cdist_[c_->rank()+1]; }
      const std::vector<int>& cdist() const { return cdist_; }
      const MPIComm& Comm() const { return *c_; }

      void compress(const mult_t& Amult);

      void mult(Trans op, const DenseM_t& X, DenseM_t& Y) const;

      /**
       * Multiply this low-rank (or butterfly) matrix with a dense
       * matrix: Y = op(A) * X, where op can be none,
       * transpose or complex conjugate. X and Y are in 2D block
       * cyclic distribution.
       *
       * \param op Transpose, conjugate, or none.
       * \param X Right-hand side matrix. Should be X.rows() ==
       * this.rows().
       * \param Y Result, should be Y.cols() == X.cols(), Y.rows() ==
       * this.rows()
       * \see mult
       */
      void mult(Trans op, const DistM_t& X, DistM_t& Y) const;

      DenseM_t redistribute_2D_to_1D
      (const DistM_t& R2D, const std::vector<int>& dist) const;
      void redistribute_2D_to_1D
      (scalar_t a, const DistM_t& R2D, scalar_t b, DenseM_t& R1D,
       const std::vector<int>& dist) const;
      void redistribute_1D_to_2D
      (const DenseM_t& S1D, DistM_t& S2D, const std::vector<int>& dist) const;

    private:
      F2Cptr lr_bf_ = nullptr;     // LRBF handle returned by Fortran code
      F2Cptr options_ = nullptr;   // options structure returned by Fortran code
      F2Cptr stats_ = nullptr;     // statistics structure returned by Fortran code
      F2Cptr msh_ = nullptr;       // mesh structure returned by Fortran code
      F2Cptr kerquant_ = nullptr;  // kernel quantities structure returned by Fortran code
      F2Cptr ptree_ = nullptr;     // process tree returned by Fortran code
      MPI_Fint Fcomm_;             // the fortran MPI communicator
      const MPIComm* c_ = nullptr;
      int rows_, cols_, lrows_, lcols_;
      std::vector<int> rdist_, cdist_;  // begin rows/cols of each rank
    };

    template<typename scalar_t> LRBFMatrix<scalar_t>::LRBFMatrix
    (const HODLRMatrix<scalar_t>& A, const HODLRMatrix<scalar_t>& B)
      : c_(&A.c_) {
      rows_ = A.rows();
      cols_ = B.cols();
      Fcomm_ = A.Fcomm_;
      int P = c_->size();
      int rank = c_->rank();
      std::vector<int> groups(P);
      std::iota(groups.begin(), groups.end(), 0);

      // create hodlr data structures
      HODLR_createptree<scalar_t>(P, groups.data(), Fcomm_, ptree_);
      HODLR_createstats<scalar_t>(stats_);

      F2Cptr Aoptions = const_cast<F2Cptr>(A.options_);
      HODLR_copyoptions<scalar_t>(Aoptions, options_);

      LRBF_construct_matvec_init<scalar_t>
        (rows_, cols_, lrows_, lcols_, A.msh_, B.msh_, lr_bf_, options_,
         stats_, msh_, kerquant_, ptree_);

      rdist_.resize(P+1);
      cdist_.resize(P+1);
      rdist_[rank+1] = lrows_;
      cdist_[rank+1] = lcols_;
      MPI_Allgather
        (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
         rdist_.data()+1, 1, MPI_INT, c_->comm());
      MPI_Allgather
        (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
         cdist_.data()+1, 1, MPI_INT, c_->comm());
      for (int p=0; p<P; p++) {
        rdist_[p+1] += rdist_[p];
        cdist_[p+1] += cdist_[p];
      }
    }

    template<typename scalar_t> LRBFMatrix<scalar_t>::~LRBFMatrix() {
      if (stats_) HODLR_deletestats<scalar_t>(stats_);
      if (ptree_) HODLR_deleteproctree<scalar_t>(ptree_);
      if (msh_) HODLR_deletemesh<scalar_t>(msh_);
      if (kerquant_) HODLR_deletekernelquant<scalar_t>(kerquant_);
      if (options_) HODLR_deleteoptions<scalar_t>(options_);
      if (lr_bf_) LRBF_deletebf<scalar_t>(lr_bf_);
    }

    template<typename scalar_t> LRBFMatrix<scalar_t>&
    LRBFMatrix<scalar_t>::operator=(LRBFMatrix<scalar_t>&& h) {
      lr_bf_ = h.lr_bf_;       h.lr_bf_ = nullptr;
      options_ = h.options_;   h.options_ = nullptr;
      stats_ = h.stats_;       h.stats_ = nullptr;
      msh_ = h.msh_;           h.msh_ = nullptr;
      kerquant_ = h.kerquant_; h.kerquant_ = nullptr;
      ptree_ = h.ptree_;       h.ptree_ = nullptr;
      Fcomm_ = h.Fcomm_;
      c_ = h.c_;
      rows_ = h.rows_;
      cols_ = h.cols_;
      lrows_ = h.lrows_;
      lcols_ = h.lcols_;
      std::swap(rdist_, h.rdist_);
      std::swap(cdist_, h.cdist_);
      return *this;
    }

    template<typename scalar_t> void LRBF_matvec_routine
    (const char* op, int* nin, int* nout, int* nvec,
     const scalar_t* X, scalar_t* Y, C2Fptr func, scalar_t* a, scalar_t* b) {
      auto A = static_cast<std::function<
        void(Trans,scalar_t,const DenseMatrix<scalar_t>&,
             scalar_t,DenseMatrix<scalar_t>&)>*>(func);
      DenseMatrixWrapper<scalar_t> Yw(*nout, *nvec, Y, *nout),
        Xw(*nin, *nvec, const_cast<scalar_t*>(X), *nin);
      (*A)(c2T(*op), *a, Xw, *b, Yw);
    }

    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::compress(const mult_t& Amult) {
      C2Fptr f = static_cast<void*>(const_cast<mult_t*>(&Amult));
      LRBF_construct_matvec_compute
        (lr_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(LRBF_matvec_routine<scalar_t>), f);
    }

    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::mult
    (Trans op, const DenseM_t& X, DenseM_t& Y) const {
      assert(Y.cols() == X.cols());
      if (op == Trans::N)
        LRBF_mult(char(op), X.data(), Y.data(), lcols_, lrows_, X.cols(),
                  lr_bf_, options_, stats_, ptree_);
      else
        LRBF_mult(char(op), X.data(), Y.data(), lrows_, lcols_, X.cols(),
                  lr_bf_, options_, stats_, ptree_);
    }

    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::mult
    (Trans op, const DistM_t& X, DistM_t& Y) const {
      DenseM_t Y1D(lrows_, X.cols());
      {
        auto X1D = redistribute_2D_to_1D(X, cdist_);
        if (op == Trans::N)
          LRBF_mult(char(op), X1D.data(), Y1D.data(), lcols_, lrows_,
                    X1D.cols(), lr_bf_, options_, stats_, ptree_);
        else
          LRBF_mult(char(op), X1D.data(), Y1D.data(), lrows_, lcols_,
                    X1D.cols(), lr_bf_, options_, stats_, ptree_);
      }
      redistribute_1D_to_2D(Y1D, Y, rdist_);
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    LRBFMatrix<scalar_t>::redistribute_2D_to_1D
    (const DistM_t& R2D, const std::vector<int>& dist) const {
      const auto rank = c_->rank();
      DenseM_t R1D(dist[rank+1] - dist[rank], R2D.cols());
      redistribute_2D_to_1D(scalar_t(1.), R2D, scalar_t(0.), R1D, dist);
      return R1D;
    }

    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::redistribute_2D_to_1D
    (scalar_t a, const DistM_t& R2D, scalar_t b, DenseM_t& R1D,
     const std::vector<int>& dist) const {
      const auto P = c_->size();
      const auto rank = c_->rank();
      const auto Rcols = R2D.cols();
      int R2Drlo, R2Drhi, R2Dclo, R2Dchi;
      R2D.lranges(R2Drlo, R2Drhi, R2Dclo, R2Dchi);
      const auto Rlcols = R2Dchi - R2Dclo;
      const auto Rlrows = R2Drhi - R2Drlo;
      const auto nprows = R2D.nprows();
      const auto B = DistM_t::default_MB;
      const auto lrows = R1D.rows();
      assert(lrows == dist[rank+1] - dist[rank]);
      std::vector<std::vector<scalar_t>> sbuf(P);
      if (R2D.active()) {
        // global, local, proc
        std::vector<std::tuple<int,int,int>> glp(Rlrows);
        {
          std::vector<std::size_t> count(P);
          for (int r=R2Drlo; r<R2Drhi; r++) {
            auto gr = R2D.rowl2g(r);
            auto p = -1 + std::distance
              (dist.begin(), std::upper_bound
               (dist.begin(), dist.end(), gr));
            glp[r] = std::tuple<int,int,int>{gr, r, p};
            count[p] += Rlcols;
          }
          sort(glp.begin(), glp.end());
          for (int p=0; p<P; p++)
            sbuf[p].reserve(count[p]);
        }
        for (int r=R2Drlo; r<R2Drhi; r++)
          for (int c=R2Dclo, lr=std::get<1>(glp[r]),
                 p=std::get<2>(glp[r]); c<R2Dchi; c++)
            sbuf[p].push_back(R2D(lr,c));
      }
      std::vector<scalar_t> rbuf;
      std::vector<scalar_t*> pbuf;
      c_->all_to_all_v(sbuf, rbuf, pbuf);
      if (lrows) {
        std::vector<int> src_c(Rcols);
        for (int c=0; c<Rcols; c++)
          src_c[c] = R2D.colg2p_fixed(c)*nprows;
        if (a == scalar_t(1.) && b == scalar_t(0.)) {
          for (int r=0; r<lrows; r++) {
            auto gr = r + dist[rank];
            auto src_r = R2D.rowg2p_fixed(gr);
            for (int c=0; c<Rcols; c++)
              R1D(r, c) = *(pbuf[src_r + src_c[c]]++);
          }
        } else {
          for (int r=0; r<lrows; r++) {
            auto gr = r + dist[rank];
            auto src_r = R2D.rowg2p_fixed(gr);
            for (int c=0; c<Rcols; c++)
              R1D(r, c) = *(pbuf[src_r + src_c[c]]++) * a + b * R1D(r, c);
          }
        }
      }
    }

    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::redistribute_1D_to_2D
    (const DenseM_t& S1D, DistM_t& S2D, const std::vector<int>& dist) const {
      const auto rank = c_->rank();
      const auto P = c_->size();
      const auto B = DistM_t::default_MB;
      const auto cols = S1D.cols();
      int S2Drlo, S2Drhi, S2Dclo, S2Dchi;
      S2D.lranges(S2Drlo, S2Drhi, S2Dclo, S2Dchi);
      const auto nprows = S2D.nprows();
      const auto lrows = dist[rank+1] - dist[rank];
      std::vector<std::vector<scalar_t>> sbuf(P);
      if (lrows) {
        std::vector<std::tuple<int,int,int>> glp(lrows);
        for (int r=0; r<lrows; r++) {
          auto gr = r + dist[rank];
          glp[r] = std::tuple<int,int,int>{gr,r,S2D.rowg2p_fixed(gr)};
        }
        std::sort(glp.begin(), glp.end());
        std::vector<int> pc(cols);
        for (int c=0; c<cols; c++)
          pc[c] = S2D.colg2p_fixed(c)*nprows;
        {
          std::vector<std::size_t> count(P);
          for (int r=0; r<lrows; r++)
            for (int c=0, pr=std::get<2>(glp[r]); c<cols; c++)
              count[pr+pc[c]]++;
          for (int p=0; p<P; p++)
            sbuf[p].reserve(count[p]);
        }
        for (int r=0; r<lrows; r++)
          for (int c=0, lr=std::get<1>(glp[r]),
                 pr=std::get<2>(glp[r]); c<cols; c++)
            sbuf[pr+pc[c]].push_back(S1D(lr,c));
      }
      std::vector<scalar_t> rbuf;
      std::vector<scalar_t*> pbuf;
      c_->all_to_all_v(sbuf, rbuf, pbuf);
      if (S2D.active()) {
        for (int r=S2Drlo; r<S2Drhi; r++) {
          auto gr = S2D.rowl2g(r);
          auto p = -1 + std::distance
            (dist.begin(), std::upper_bound
             (dist.begin(), dist.end(), gr));
          for (int c=S2Dclo; c<S2Dchi; c++)
            S2D(r,c) = *(pbuf[p]++);
        }
      }
    }

  } // end namespace HODLR
} // end namespace strumpack

#endif // STRUMPACK_LRBF_MATRIX_HPP
