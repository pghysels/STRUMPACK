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
/*! \file HODLRMatrix.hpp
 * \brief Class wrapping around Yang Liu's HODLR code.
 */
#ifndef STRUMPACK_HODLR_MATRIX_HPP
#define STRUMPACK_HODLR_MATRIX_HPP

#include <cassert>

#include "HSS/HSSPartitionTree.hpp"
#include "kernel/Kernel.hpp"
#include "clustering/Clustering.hpp"
#include "HODLROptions.hpp"
#include "HODLRWrapper.hpp"

namespace strumpack {

  /**
   * Code in this namespace is a wrapper aroung Yang Liu's Fortran
   * code: https://github.com/liuyangzhuan/hod-lr-bf
   */
  namespace HODLR {

    template<typename scalar_t> class HODLRMatrix {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using DistM_t = DistributedMatrix<scalar_t>;
      using opts_t = HODLROptions<scalar_t>;
      using mult_t = typename std::function<
        void(char,const DenseM_t&,DenseM_t&)>;

    public:
      HODLRMatrix() {}
      HODLRMatrix
      (const MPIComm& c, kernel::Kernel<scalar_t>& K,
       DenseM_t& labels, const opts_t& opts);
      HODLRMatrix
      (const MPIComm& c, const HSS::HSSPartitionTree& tree,
       const opts_t& opts);

      virtual ~HODLRMatrix();

      std::size_t rows() const { return rows_; }
      std::size_t cols() const { return cols_; }
      std::size_t lrows() const { return lrows_; }
      std::size_t begin_row() const { return dist_[c_->rank()]; }
      std::size_t end_row() const { return dist_[c_->rank()+1]; }
      const MPIComm& Comm() const { return *c_; }

#if defined(STRUMPACK_USE_HODLRBF)
      void compress(const mult_t& Amult);
      void mult(char op, const DenseM_t& X, DenseM_t& Y) /*const*/;
      void mult(char op, const DistM_t& X, DistM_t& Y) /*const*/;
      void factor();
      void solve(const DenseM_t& B, DenseM_t& X);
      void solve(const DistM_t& B, DistM_t& X);
#else
      void compress(const mult_t& Amult) {}
      void mult(char op, const DenseM_t& X, DenseM_t& Y) /*const*/ {}
      void mult(char op, const DistM_t& X, DistM_t& Y) /*const*/ {}
      void factor() {}
      void solve(const DenseM_t& B, DenseM_t& X) {}
      void solve(const DistM_t& B, DistM_t& X) {}
#endif

    private:
#if defined(STRUMPACK_USE_HODLRBF)
      F2Cptr ho_bf_ = nullptr;     // HODLR handle returned by Fortran code
      F2Cptr options_ = nullptr;   // options structure returned by Fortran code
      F2Cptr stats_ = nullptr;     // statistics structure returned by Fortran code
      F2Cptr msh_ = nullptr;       // mesh structure returned by Fortran code
      F2Cptr kerquant_ = nullptr;  // kernel quantities structure returned by Fortran code
      F2Cptr ptree_ = nullptr;     // process tree returned by Fortran code
      MPI_Fint Fcomm_;             // the fortran MPI communicator
#endif
      const MPIComm* c_;
      int rows_, cols_, lrows_;
      std::vector<int> perm_, iperm_; // permutation used by the HODLR code
      std::vector<int> dist_;         // begin rows of each rank

      DenseM_t redistribute_2D_to_1D(const DistM_t& R) const;
      void redistribute_1D_to_2D(const DenseM_t& S1D, DistM_t& S2D) const;
      template<typename S> friend class LRBFMatrix;
    };

#if defined(STRUMPACK_USE_HODLRBF)
    /**
     * Routine used to pass to the fortran code to compute a selected
     * element of a kernel. The kernel argument needs to be a pointer
     * to a strumpack::kernel object.
     *
     * \param i row coordinate of element to be computed from the
     * kernel
     * \param i column coordinate of element to be computed from the
     * kernel
     * \param v output, kernel value
     * \param kernel pointer to Kernel object
     */
    template<typename scalar_t> void HODLR_kernel_evaluation
    (int* i, int* j, scalar_t* v, C2Fptr kernel) {
      *v = static_cast<kernel::Kernel<scalar_t>*>(kernel)->eval(*i, *j);
    }
#endif

    template<typename scalar_t> HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, kernel::Kernel<scalar_t>& K,
     DenseM_t& labels, const opts_t& opts) : c_(&c) {
#if defined(STRUMPACK_USE_HODLRBF)
      int d = K.d();
      rows_ = cols_ = K.n();

      auto tree = binary_tree_clustering
        (opts.clustering_algorithm(), K.data(), labels, opts.leaf_size());

      Fcomm_ = MPI_Comm_c2f(c_->comm());
      int P = c_->size();
      int rank = c_->rank();
      std::vector<int> groups(P);
      std::iota(groups.begin(), groups.end(), 0);

      // create hodlr data structures
      HODLR_createptree<scalar_t>(P, groups.data(), Fcomm_, ptree_);
      HODLR_createoptions<scalar_t>(options_);
      HODLR_createstats<scalar_t>(stats_);

      // set hodlr options
      int com_opt = 2;   // compression option 1:SVD 2:RRQR 3:ACA 4:BACA
      int sort_opt = 0;  // 0:natural order, 1:kd-tree, 2:cobble-like ordering
                         // 3:gram distance-based cobble-like ordering
      HODLR_set_D_option<scalar_t>(options_, "tol_comp", opts.rel_tol());
      HODLR_set_I_option<scalar_t>(options_, "nogeo", 1);
      //HODLR_set_I_option<scalar_t>(options_, "Nmin_leaf", opts.leaf_size());
      HODLR_set_I_option<scalar_t>(options_, "Nmin_leaf", rows_);
      HODLR_set_I_option<scalar_t>(options_, "RecLR_leaf", com_opt);
      HODLR_set_I_option<scalar_t>(options_, "xyzsort", sort_opt);
      HODLR_set_I_option<scalar_t>(options_, "ErrFillFull", 0);
      HODLR_set_I_option<scalar_t>(options_, "BACA_Batch", 100);

      // TODO does the tree need to be complete?
      HSS::HSSPartitionTree full_tree(tree);
      full_tree.expand_complete(false); // no empty nodes!
      int lvls = full_tree.levels();
      std::vector<int> leafs = full_tree.leaf_sizes();

      perm_.resize(rows_);
      // construct HODLR with geometrical points
      HODLR_construct
        (rows_, d, K.data().data(), lvls-1, leafs.data(),
         perm_.data(), lrows_, ho_bf_, options_, stats_, msh_, kerquant_,
         ptree_, &(HODLR_kernel_evaluation<scalar_t>), &K, Fcomm_);

      //---- NOT NECESSARY ?? --------------------------------------------
      iperm_.resize(rows_);
      for (auto& i : perm_) i--; // Fortran to C
      MPI_Bcast(perm_.data(), perm_.size(), MPI_INT, 0, c_->comm());
      for (int i=0; i<rows_; i++)
        iperm_[perm_[i]] = i;
      //------------------------------------------------------------------

      dist_.resize(P+1);
      dist_[rank+1] = lrows_;
      MPI_Allgather
        (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
         dist_.data()+1, 1, MPI_INT, c_->comm());
      for (int p=0; p<P; p++)
        dist_[p+1] += dist_[p];
#else
      std::cerr << "ERROR: STRUMPACK was not configured with HODLRBF support."
                << std::endl;
#endif
    }


    template<typename scalar_t> HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, const HSS::HSSPartitionTree& tree,
     const opts_t& opts) : c_(&c) {
#if defined(STRUMPACK_USE_HODLRBF)
      rows_ = cols_ = tree.size;

      Fcomm_ = MPI_Comm_c2f(c_->comm());
      int P = c_->size();
      int rank = c_->rank();
      std::vector<int> groups(P);
      std::iota(groups.begin(), groups.end(), 0);

      // create hodlr data structures
      HODLR_createptree<scalar_t>(P, groups.data(), Fcomm_, ptree_);
      HODLR_createoptions<scalar_t>(options_);
      HODLR_createstats<scalar_t>(stats_);

      // set hodlr options
      int com_opt = 2;   // compression option 1:SVD 2:RRQR 3:ACA 4:BACA
      int sort_opt = 0;  // 0:natural order, 1:kd-tree, 2:cobble-like ordering
                         // 3:gram distance-based cobble-like ordering
      HODLR_set_D_option<scalar_t>(options_, "tol_comp", opts.rel_tol());
      HODLR_set_I_option<scalar_t>(options_, "nogeo", 1);
      HODLR_set_I_option<scalar_t>(options_, "Nmin_leaf", rows_);
      //HODLR_set_I_option<scalar_t>(options_, "RecLR_leaf", com_opt);
      HODLR_set_I_option<scalar_t>(options_, "xyzsort", sort_opt);
      HODLR_set_I_option<scalar_t>(options_, "ErrFillFull", 0);
      HODLR_set_I_option<scalar_t>(options_, "BACA_Batch", 100);

      // TODO does the tree need to be complete?
      HSS::HSSPartitionTree full_tree(tree);
      full_tree.expand_complete(false); // no empty nodes!
      int lvls = full_tree.levels();
      std::vector<int> leafs = full_tree.leaf_sizes();

      perm_.resize(rows_);
      HODLR_construct_matvec_init<scalar_t>
        (rows_, lvls-1, leafs.data(), perm_.data(), lrows_, ho_bf_, options_,
         stats_, msh_, kerquant_, ptree_);

      //---- NOT NECESSARY ?? --------------------------------------------
      iperm_.resize(rows_);
      for (auto& i : perm_) i--; // Fortran to C
      MPI_Bcast(perm_.data(), perm_.size(), MPI_INT, 0, c_->comm());
      for (int i=0; i<rows_; i++)
        iperm_[perm_[i]] = i;
      //------------------------------------------------------------------

      dist_.resize(P+1);
      dist_[rank+1] = lrows_;
      MPI_Allgather
        (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
         dist_.data()+1, 1, MPI_INT, c_->comm());
      for (int p=0; p<P; p++)
        dist_[p+1] += dist_[p];
#else
      std::cerr << "ERROR: STRUMPACK was not configured with HODLRBF support."
                << std::endl;
#endif
    }

    template<typename scalar_t> HODLRMatrix<scalar_t>::~HODLRMatrix() {
#if defined(STRUMPACK_USE_HODLRBF)
      HODLR_deletestats<scalar_t>(stats_);
      HODLR_deleteproctree<scalar_t>(ptree_);
      HODLR_deletemesh<scalar_t>(msh_);
      HODLR_deletekernelquant<scalar_t>(kerquant_);
      HODLR_deletehobf<scalar_t>(ho_bf_);
      HODLR_deleteoptions<scalar_t>(options_);
#endif
    }


#if defined(STRUMPACK_USE_HODLRBF)
    template<typename scalar_t> void HODLR_matvec_routine
    (const char* op, int* nin, int* nout, int* nvec,
     const scalar_t* X, scalar_t* Y, C2Fptr func) {
      auto A = static_cast<std::function<
        void(char,const DenseMatrix<scalar_t>&,
             DenseMatrix<scalar_t>&)>*>(func);
      DenseMatrixWrapper<scalar_t> Yw(*nout, *nvec, Y, *nout),
        Xw(*nin, *nvec, const_cast<scalar_t*>(X), *nin);
      (*A)(*op, Xw, Yw);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::compress
    (const mult_t& Amult) {
      C2Fptr f = static_cast<void*>(const_cast<mult_t*>(&Amult));
      HODLR_construct_matvec_compute
        (ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(HODLR_matvec_routine<scalar_t>), f);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::mult
    (char op, const DenseM_t& X, DenseM_t& Y) /*const*/ {
      HODLR_mult(op, X.data(), Y.data(), lrows_, lrows_, X.cols(),
                 ho_bf_, options_, stats_, ptree_);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::mult
    (char op, const DistM_t& X, DistM_t& Y) /*const*/ {
      DenseM_t Y1D(lrows_, X.cols());
      {
        auto X1D = redistribute_2D_to_1D(X);
        HODLR_mult(op, X1D.data(), Y1D.data(), lrows_, lrows_,
                   X.cols(), ho_bf_, options_, stats_, ptree_);
      }
      redistribute_1D_to_2D(Y1D, Y);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::factor() {
      HODLR_factor<scalar_t>(ho_bf_, options_, stats_, ptree_, msh_);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::solve(const DenseM_t& B, DenseM_t& X) /*const*/ {
      HODLR_solve(X.data(), B.data(), lrows_, X.cols(),
                  ho_bf_, options_, stats_, ptree_);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::solve(const DistM_t& B, DistM_t& X) /*const*/ {
      DenseM_t X1D(lrows_, X.cols());
      {
        auto B1D = redistribute_2D_to_1D(B);
        HODLR_solve(X1D.data(), B1D.data(), lrows_, X.cols(),
                    ho_bf_, options_, stats_, ptree_);
      }
      redistribute_1D_to_2D(X1D, X);
    }
#endif

    template<typename scalar_t> DenseMatrix<scalar_t>
    HODLRMatrix<scalar_t>::redistribute_2D_to_1D(const DistM_t& R) const {
      const auto P = c_->size();
      const auto rank = c_->rank();
      const auto Rcols = R.cols();
      const auto Rlcols = R.lcols();
      const auto Rlrows = R.lrows();
      const auto B = DistM_t::default_MB;
      int nprows = R.nprows(), npcols = R.npcols();
      std::vector<std::vector<scalar_t>> sbuf(P);
      if (R.active()) {
        // global, local, proc
        std::vector<std::tuple<int,int,int>> glp(Rlrows);
        {
          std::vector<std::size_t> count(P);
          for (int r=0; r<Rlrows; r++) {
            auto gr = perm_[R.rowl2g(r)];
            auto p = -1 + std::distance
              (dist_.begin(), std::upper_bound
               (dist_.begin(), dist_.end(), gr));
            glp[r] = std::tuple<int,int,int>{gr, r, p};
            count[p] += Rlcols;
          }
          sort(glp.begin(), glp.end());
          for (int p=0; p<P; p++)
            sbuf[p].reserve(count[p]);
        }
        for (int r=0; r<Rlrows; r++)
          for (int c=0, lr=std::get<1>(glp[r]),
                 p=std::get<2>(glp[r]); c<Rlcols; c++)
            sbuf[p].push_back(R(lr,c));
      }
      std::vector<scalar_t> rbuf;
      std::vector<scalar_t*> pbuf;
      c_->all_to_all_v(sbuf, rbuf, pbuf);
      DenseM_t R1D(lrows_, Rcols);
      if (lrows_) {
        std::vector<int> src_c(Rcols);
        for (int c=0; c<Rcols; c++)
          src_c[c] = ((c / B) % npcols) * nprows;
        for (int r=0; r<lrows_; r++) {
          auto gr = perm_[r + dist_[rank]];
          auto src_r = (gr / B) % nprows;
          for (int c=0; c<Rcols; c++)
            R1D(r, c) = *(pbuf[src_r + src_c[c]]++);
        }
      }
      return R1D;
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::redistribute_1D_to_2D
    (const DenseM_t& S1D, DistM_t& S2D) const {
      const auto rank = c_->rank();
      const auto P = c_->size();
      const auto B = DistM_t::default_MB;
      const auto cols = S1D.cols();
      const auto Slcols = S2D.lcols();
      const auto Slrows = S2D.lrows();
      int nprows = S2D.nprows(), npcols = S2D.npcols();
      std::vector<std::vector<scalar_t>> sbuf(P);
      if (lrows_) {
        std::vector<std::tuple<int,int,int>> glp(lrows_);
        for (int r=0; r<lrows_; r++) {
          auto gr = iperm_[r + dist_[rank]];
          glp[r] = std::tuple<int,int,int>{gr,r,(gr / B) % nprows};
        }
        sort(glp.begin(), glp.end());
        std::vector<int> pc(cols);
        for (int c=0; c<cols; c++)
          pc[c] = ((c / B) % npcols) * nprows;
        {
          std::vector<std::size_t> count(P);
          for (int r=0; r<lrows_; r++)
            for (int c=0, pr=std::get<2>(glp[r]); c<cols; c++)
              count[pr+pc[c]]++;
          for (int p=0; p<P; p++)
            sbuf[p].reserve(count[p]);
        }
        for (int r=0; r<lrows_; r++)
          for (int c=0, lr=std::get<1>(glp[r]),
                 pr=std::get<2>(glp[r]); c<cols; c++)
            sbuf[pr+pc[c]].push_back(S1D(lr,c));
      }
      std::vector<scalar_t> rbuf;
      std::vector<scalar_t*> pbuf;
      c_->all_to_all_v(sbuf, rbuf, pbuf);
      if (S2D.active()) {
        for (int r=0; r<Slrows; r++) {
          auto gr = perm_[S2D.rowl2g(r)];
          auto p = -1 + std::distance
            (dist_.begin(), std::upper_bound(dist_.begin(), dist_.end(), gr));
          for (int c=0; c<Slcols; c++)
            S2D(r,c) = *(pbuf[p]++);
        }
      }
    }

  } // end namespace HODLR
} // end namespace strumpack

#endif // STRUMPACK_HODLR_MATRIX_HPP
