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

    template<typename scalar_t> void LRBF_matvec_routine
    (const char* op, int* nin, int* nout, int* nvec,
     const scalar_t* X, scalar_t* Y, C2Fptr func, scalar_t* a, scalar_t* b) {
      auto A = static_cast<std::function<
        void(char,scalar_t,const DenseMatrix<scalar_t>&,
             scalar_t,DenseMatrix<scalar_t>&)>*>(func);
      DenseMatrixWrapper<scalar_t> Yw(*nout, *nvec, Y, *nout),
        Xw(*nin, *nvec, const_cast<scalar_t*>(X), *nin);
      //(*A)(*op, *a, Xw, *b, Yw);
      Yw.zero();
    }


    template<typename scalar_t> class LRBFMatrix {
      using DenseM_t = DenseMatrix<scalar_t>;
      using opts_t = HODLROptions<scalar_t>;
      using mult_t = typename std::function<
        void(char,scalar_t,const DenseM_t&,scalar_t,DenseM_t&)>;

    public:
      LRBFMatrix() {}
      /**
       * Construct the block X, subblock of the matrix [A X; Y B]
       * A and B should be defined on the same MPI communicator.
       */
      LRBFMatrix
      (const HODLRMatrix<scalar_t>& A, const HODLRMatrix<scalar_t>& B,
       const mult_t& Amult);

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
      std::size_t begin_col() const { return rdist_[c_->rank()]; }
      std::size_t end_col() const { return rdist_[c_->rank()+1]; }
      const MPIComm& Comm() const { return *c_; }

      void mult(char op, scalar_t alpha, const DenseM_t& X,
                scalar_t beta, DenseM_t& Y) const;

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
    (const HODLRMatrix<scalar_t>& A, const HODLRMatrix<scalar_t>& B,
     const mult_t& Amult) : c_(A.c_) {
      rows_ = A.rows();
      cols_ = B.cols();
      Fcomm_ = A.Fcomm_; //MPI_Comm_c2f(c_->comm());
      int P = c_->size();
      int rank = c_->rank();
      std::vector<int> groups(P);
      std::iota(groups.begin(), groups.end(), 0);

      // create hodlr data structures
      HODLR_createptree<scalar_t>(P, groups.data(), Fcomm_, ptree_);
      HODLR_createstats<scalar_t>(stats_);

      // F2Cptr Aoptions = const_cast<F2Cptr>(A.options_);
      // HODLR_copyoptions<scalar_t>(Aoptions, options_);

      HODLR_createoptions<scalar_t>(options_);

      // set hodlr options
      int com_opt = 2;   // compression option 1:SVD 2:RRQR 3:ACA 4:BACA
      int sort_opt = 0;  // 0:natural order, 1:kd-tree, 2:cobble-like ordering
                         // 3:gram distance-based cobble-like ordering
      HODLR_set_D_option<scalar_t>(options_, "tol_comp", 1e-2);
      HODLR_set_I_option<scalar_t>(options_, "nogeo", 1);
      HODLR_set_I_option<scalar_t>(options_, "Nmin_leaf", rows_);
      //HODLR_set_I_option<scalar_t>(options_, "RecLR_leaf", com_opt);
      HODLR_set_I_option<scalar_t>(options_, "xyzsort", sort_opt);
      HODLR_set_I_option<scalar_t>(options_, "ErrFillFull", 0);
      HODLR_set_I_option<scalar_t>(options_, "BACA_Batch", 100);


      LRBF_construct_matvec_init<scalar_t>
        (rows_, cols_, lrows_, lcols_, A.msh_, B.msh_, lr_bf_, options_,
         stats_, msh_, kerquant_, ptree_);

      C2Fptr f = static_cast<void*>(const_cast<mult_t*>(&Amult));
      LRBF_construct_matvec_compute
        (lr_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(LRBF_matvec_routine<scalar_t>), f);

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

    template<typename scalar_t> void
    LRBFMatrix<scalar_t>::mult
    (char op, scalar_t alpha, const DenseM_t& X,
     scalar_t beta, DenseM_t& Y) const {
      LRBF_mult(op, X.data(), Y.data(), lrows_, lrows_, X.cols(),
                lr_bf_, options_, stats_, ptree_, alpha, beta);
    }

  } // end namespace HODLR
} // end namespace strumpack

#endif // STRUMPACK_LRBF_MATRIX_HPP
