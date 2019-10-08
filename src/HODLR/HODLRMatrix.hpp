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
/**
 * \file HODLRMatrix.hpp
 * \brief Class wrapping around Yang Liu's HODLR code.
 */
#ifndef STRUMPACK_HODLR_MATRIX_HPP
#define STRUMPACK_HODLR_MATRIX_HPP

#include <cassert>
#include <algorithm>

#include "HSS/HSSPartitionTree.hpp"
#include "kernel/Kernel.hpp"
#include "clustering/Clustering.hpp"
#include "dense/DistributedMatrix.hpp"
#include "HODLROptions.hpp"
#include "HODLRWrapper.hpp"
#include "sparse/CSRGraph.hpp"

namespace strumpack {

  /**
   * Code in this namespace is a wrapper around Yang Liu's Fortran
   * code:
   *    https://github.com/liuyangzhuan/ButterflyPACK
   */
  namespace HODLR {

    struct ExtractionMeta {
      std::unique_ptr<int[]> iwork;
      int Ninter, Nallrows, Nallcols, Nalldat_loc,
        *allrows, *allcols, *rowids, *colids, *pgids, Npmap, *pmaps;
    };

    /**
     * \class HODLRMatrix
     *
     * \brief Hierarchically low-rank matrix representation.
     *
     * This requires MPI support.
     *
     * There are 3 different ways to create an HODLRMatrix
     *  - By specifying a matrix-(multiple)vector multiplication
     *    routine.
     *  - By specifying an element extraction routine.
     *  - By specifying a strumpack::kernel::Kernel matrix, defined by
     *    a collection of points and a kernel function.
     *
     * \tparam scalar_t Can be float, double, std:complex<float> or
     * std::complex<double>.
     *
     * \see HSS::HSSMatrix
     */
    template<typename scalar_t> class HODLRMatrix {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using DistM_t = DistributedMatrix<scalar_t>;
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      using opts_t = HODLROptions<scalar_t>;
      using Vec_t = std::vector<std::size_t>;
      using VecVec_t = std::vector<std::vector<std::size_t>>;

    public:
      using mult_t = typename std::function
        <void(Trans, const DenseM_t&, DenseM_t&)>;
      using elem_t = typename std::function
        <scalar_t(std::size_t i, std::size_t j)>;
      using delem_blocks_t = typename std::function
        <void(VecVec_t& I, VecVec_t& J, std::vector<DistMW_t>& B,
              ExtractionMeta&)>;
      using elem_blocks_t = typename std::function
        <void(VecVec_t& I, VecVec_t& J, std::vector<DenseMW_t>& B,
              ExtractionMeta&)>;

      /**
       * Default constructor, makes an empty 0 x 0 matrix.
       */
      HODLRMatrix() {}

      /**
       * Construct an HODLR approximation for the kernel matrix K.
       *
       * \param c MPI communicator, this communicator is copied
       * internally.
       * \param K Kernel matrix object. The data associated with this
       * kernel will be permuted according to the clustering algorithm
       * selected by the HODLROptions objects. The permutation will be
       * stored in the kernel object.
       * \param opts object containing a number of HODLR options
       */
      HODLRMatrix
      (const MPIComm& c, kernel::Kernel<scalar_t>& K, const opts_t& opts);

	  
      /**
       * Construct an HODLR approximation for the kernel matrix K using geometries.
       *
       * \param c MPI communicator, this communicator is copied
       * internally.
       * \param K Kernel matrix object. The data associated with this
       * kernel will be permuted according to the clustering algorithm
       * selected by the HODLROptions objects. The permutation will be
       * stored in the kernel object.
       * \param opts object containing a number of HODLR options
       */
      HODLRMatrix
	  (const MPIComm& c, kernel::Kernel<scalar_t>& K, int dim, std::vector<scalar_t>& geos, const opts_t& opts);
	  
      /**
       * Construct an HODLR approximation using a routine to evaluate
       * individual matrix elements.
       *
       * \param c MPI communicator, this communicator is copied
       * internally.
       * \param t tree specifying the HODLR matrix partitioning
       * \param Aelem Routine, std::function, which can also be a
       * lambda function or a functor (class object implementing the
       * member "scalar_t operator()(int i, int j)"), that
       * evaluates/returns the matrix element A(i,j)
       * \param opts object containing a number of HODLR options
       */
      HODLRMatrix
      (const MPIComm& c, const HSS::HSSPartitionTree& tree,
       const std::function<scalar_t(int i, int j)>& Aelem,
       const opts_t& opts);

      /**
       * Construct an HODLR matrix using a specified HODLR tree and
       * matrix-vector multiplication routine.
       *
       * \param c MPI communicator, this communicator is copied
       * internally.
       * \param t tree specifying the HODLR matrix partitioning
       * \param Amult Routine for the matrix-vector product. Trans op
       * argument will be N, T or C for none, transpose or complex
       * conjugate. The const DenseM_t& argument is the the random
       * matrix R, and the final DenseM_t& argument S is what the user
       * routine should compute as A*R, A^t*R or A^c*R. S will already
       * be allocated.
       * \param opts object containing a number of options for HODLR
       * compression
       * \see compress, HODLROptions
       */
      HODLRMatrix
      (const MPIComm& c, const HSS::HSSPartitionTree& tree,
       const std::function<void(Trans op,const DenseM_t& R,DenseM_t& S)>& Amult,
       const opts_t& opts);

      /**
       * Construct an HODLR matrix using a specified HODLR tree. After
       * construction, the HODLR matrix will be empty, and can be filled
       * by calling one of the compress member routines.
       *
       * \param c MPI communicator, this communicator is copied
       * internally.
       * \param t tree specifying the HODLR matrix partitioning
       * \param opts object containing a number of options for HODLR
       * compression
       * \see compress, HODLROptions
       */
      HODLRMatrix
      (const MPIComm& c, const HSS::HSSPartitionTree& tree,
       const opts_t& opts);

      /**
       * Construct an HODLR matrix using a specified HODLR tree. After
       * construction, the HODLR matrix will be empty, and can be filled
       * by calling one of the compress member routines.
       *
       * \param c MPI communicator, this communicator is copied
       * internally.
       * \param t tree specifying the HODLR matrix partitioning
       * \param admissibility matrix of size tree.leaf_sizes().size()
       * ^2 with admissibility info
       * \param graph connectivity info for the dofs
       * \param opts object containing a number of options for HODLR
       * compression
       * \see compress, HODLROptions
       */
      template<typename integer_t> HODLRMatrix
      (const MPIComm& c, const HSS::HSSPartitionTree& tree,
       const DenseMatrix<bool>& admissibility,
       const CSRGraph<integer_t>& graph, const opts_t& opts);

      /**
       * Copy constructor is not supported.
       */
      HODLRMatrix(const HODLRMatrix<scalar_t>& h) = delete;

      /**
       * Move constructor.
       * \param h HODLRMatrix to move from, will be emptied.
       */
      HODLRMatrix(HODLRMatrix<scalar_t>&& h) { *this = h; }

      /**
       * Virtual destructor.
       */
      virtual ~HODLRMatrix();

      /**
       * Copy assignement operator is not supported.
       */
      HODLRMatrix<scalar_t>& operator=(const HODLRMatrix<scalar_t>& h) = delete;

      /**
       * Move assignment operator.
       * \param h HODLRMatrix to move from, will be emptied.
       */
      HODLRMatrix<scalar_t>& operator=(HODLRMatrix<scalar_t>&& h);

      /**
       * Return the number of rows in the matrix.
       * \return Global number of rows in the matrix.
       */
      std::size_t rows() const { return rows_; }
      /**
       * Return the number of columns in the matrix.
       * \return Global number of columns in the matrix.
       */
      std::size_t cols() const { return cols_; }
      /**
       * Return the number of local rows, owned by this process.
       * \return Number of local rows.
       */
      std::size_t lrows() const { return lrows_; }
      /**
       * Return the first row of the local rows owned by this process.
       * \return Return first local row
       */
      std::size_t begin_row() const { return dist_[c_.rank()]; }
      /**
       * Return last row (+1) of the local rows (begin_rows()+lrows())
       * \return Final local row (+1).
       */
      std::size_t end_row() const { return dist_[c_.rank()+1]; }
      /**
       * Return MPI communicator wrapper object.
       */
      const MPIComm& Comm() const { return c_; }

      double get_stat(const std::string& name) const {
        if (!stats_) return 0;
        return BPACK_get_stat<scalar_t>(stats_, name);
      }

      /**
       * Construct the compressed HODLR representation of the matrix,
       * using only a matrix-(multiple)vector multiplication routine.
       *
       * \param Amult Routine for the matrix-vector product. Trans op
       * argument will be N, T or C for none, transpose or complex
       * conjugate. The const DenseM_t& argument is the the random
       * matrix R, and the final DenseM_t& argument S is what the user
       * routine should compute as A*R, A^t*R or A^c*R. S will already
       * be allocated.
       */
      void compress
      (const std::function<void(Trans op,const DenseM_t& R,DenseM_t& S)>& Amult);

      /**
       * Construct the compressed HODLR representation of the matrix,
       * using only a matrix-(multiple)vector multiplication routine.
       *
       * \param Amult Routine for the matrix-vector product. Trans op
       * argument will be N, T or C for none, transpose or complex
       * conjugate. The const DenseM_t& argument is the the random
       * matrix R, and the final DenseM_t& argument S is what the user
       * routine should compute as A*R, A^t*R or A^c*R. S will already
       * be allocated.
       * \param rank_guess Initial guess for the rank
       */
      void compress
      (const std::function<void(Trans op,const DenseM_t& R,DenseM_t& S)>& Amult,
       int rank_guess);

      /**
       * Construct the compressed HODLR representation of the matrix,
       * using only a element evaluation (multiple blocks at once).
       *
       * \param Aelem element extraction routine, extracting multiple
       * blocks at once.
       */
      void compress(const delem_blocks_t& Aelem);

      /**
       * Construct the compressed HODLR representation of the matrix,
       * using only a element evaluation (multiple blocks at once).
       *
       * \param Aelem element extraction routine, extracting multiple
       * blocks at once.
       */
      void compress(const elem_blocks_t& Aelem);

      /**
       * Multiply this HODLR matrix with a dense matrix: Y =
       * op(this)*X, where op can be none, transpose or complex
       * conjugate. X and Y are the local parts of block-row
       * distributed matrices. The number of rows in X and Y should
       * correspond to the distribution of this HODLR matrix.
       *
       * \param op Transpose, conjugate, or none.
       * \param X Right-hand side matrix. This is the local part of
       * the distributed matrix X. Should be X.rows() == this.lrows().
       * \param Y Result, should be Y.cols() == X.cols(), Y.rows() ==
       * this.lrows()
       * \see lrows, begin_row, end_row, mult
       */
      void mult(Trans op, const DenseM_t& X, DenseM_t& Y) const;

      /**
       * Multiply this HODLR matrix with a dense matrix: Y =
       * op(this)*X, where op can be none, transpose or complex
       * conjugate. X and Y are in 2D block cyclic distribution.
       *
       * \param op Transpose, conjugate, or none.
       * \param X Right-hand side matrix. Should be X.rows() ==
       * this.rows().
       * \param Y Result, should be Y.cols() == X.cols(), Y.rows() ==
       * this.rows()
       * \see mult
       */
      void mult(Trans op, const DistM_t& X, DistM_t& Y) const;

      /**
       * Compute the factorization of this HODLR matrix. The matrix
       * can still be used for multiplication.
       *
       * \see solve, inv_mult
       */
      void factor();

      /**
       * Solve a system of linear equations A*X=B, with possibly
       * multiple right-hand sides.
       *
       * \param B Right hand side. This is the local part of
       * the distributed matrix B. Should be B.rows() == this.lrows().
       * \param X Result, should be X.cols() == B.cols(), X.rows() ==
       * this.lrows(). X should be allocated.
       * \see factor, lrows, begin_row, end_row, inv_mult
       */
      void solve(const DenseM_t& B, DenseM_t& X) const;

      /**
       * Solve a system of linear equations A*X=B, with possibly
       * multiple right-hand sides. X and B are in 2D block cyclic
       * distribution.
       *
       * \param B Right hand side. This is the local part of
       * the distributed matrix B. Should be B.rows() == this.rows().
       * \param X Result, should be X.cols() == B.cols(), X.rows() ==
       * this.rows(). X should be allocated.
       * \see factor, lrows, begin_row, end_row, inv_mult
       */
      void solve(const DistM_t& B, DistM_t& X) const;

      /**
       * Solve a system of linear equations op(A)*X=B, with possibly
       * multiple right-hand sides, where op can be none, transpose or
       * complex conjugate.
       *
       * \param B Right hand side. This is the local part of
       * the distributed matrix B. Should be B.rows() == this.lrows().
       * \param X Result, should be X.cols() == B.cols(), X.rows() ==
       * this.lrows(). X should be allocated.
       * \see factor, solve, lrows, begin_row, end_row
       */
      void inv_mult(Trans op, const DenseM_t& B, DenseM_t& X) const;

      void extract_elements
      (const VecVec_t& I, const VecVec_t& J, std::vector<DistM_t>& B);
      void extract_elements(const Vec_t& I, const Vec_t& J, DenseM_t& B);

      DistM_t dense(const BLACSGrid* g) const;

      DenseM_t redistribute_2D_to_1D(const DistM_t& R) const;
      void redistribute_2D_to_1D(const DistM_t& R2D, DenseM_t& R1D) const;
      void redistribute_1D_to_2D(const DenseM_t& S1D, DistM_t& S2D) const;

    private:
      F2Cptr ho_bf_ = nullptr;     // HODLR handle returned by Fortran code
      F2Cptr options_ = nullptr;   // options structure returned by Fortran code
      F2Cptr stats_ = nullptr;     // statistics structure returned by Fortran code
      F2Cptr msh_ = nullptr;       // mesh structure returned by Fortran code
      F2Cptr kerquant_ = nullptr;  // kernel quantities structure returned by Fortran code
      F2Cptr ptree_ = nullptr;     // process tree returned by Fortran code
      MPI_Fint Fcomm_;             // the fortran MPI communicator
      MPIComm c_;
      int rows_ = 0, cols_ = 0, lrows_ = 0, lvls_ = 0;
      std::vector<int> perm_, iperm_; // permutation used by the HODLR code
      std::vector<int> dist_;         // begin rows of each rank
      std::vector<int> leafs_;        // leaf sizes of the tree

      void options_init(const opts_t& opts);
      void perm_init();
      void dist_init();

      template<typename S> friend class LRBFMatrix;
    };


    template<typename scalar_t> struct KernelCommPtrs {
      const kernel::Kernel<scalar_t>* K;
      const MPIComm* c;
    };

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
    (int* i, int* j, scalar_t* v, C2Fptr KC) {
      const auto& K = *(static_cast<KernelCommPtrs<scalar_t>*>(KC)->K);
      *v = K.eval(*i-1, *j-1);
    }

    template<typename scalar_t> void HODLR_kernel_block_evaluation
    (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
     int* allrows, int* allcols, scalar_t* alldat_loc,
     int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
     C2Fptr KC) {
      auto temp = static_cast<KernelCommPtrs<scalar_t>*>(KC);
      const auto& K = *(temp->K);
      const auto& comm = *(temp->c);
      auto data = alldat_loc;
      for (int isec=0, r0=0, c0=0; isec<*Ninter; isec++) {
        auto m = rowids[isec];
        auto n = colids[isec];
        auto p0 = pmaps[2*(*Npmap)+pgids[isec]];
        assert(pmaps[pgids[isec]] == 1);          // prows == 1
        assert(pmaps[(*Npmap)+pgids[isec]] == 1); // pcols == 1
        if (comm.rank() == p0) {
          for (int c=0; c<n; c++)
            for (int r=0; r<m; r++)
              data[r+c*m] = K.eval(allrows[r0+r]-1, allcols[c0+c]-1);
          data += m*n;
        }
        r0 += m;
        c0 += n;
      }
    }

    template<typename scalar_t> void HODLR_element_evaluation
    (int* i, int* j, scalar_t* v, C2Fptr elem) {
      *v = static_cast<std::function<scalar_t(int,int)>*>
        (elem)->operator()(*i-1, *j-1);
    }

    template<typename scalar_t> struct AelemCommPtrs {
      const typename HODLRMatrix<scalar_t>::delem_blocks_t* Aelem;
      const MPIComm* c;
    };

    template<typename scalar_t> void HODLR_block_evaluation
    (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
     int* allrows, int* allcols, scalar_t* alldat_loc,
     int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
     C2Fptr AC) {
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      auto temp = static_cast<AelemCommPtrs<scalar_t>*>(AC);
      std::vector<std::vector<std::size_t>> I(*Ninter), J(*Ninter);
      // grid should still exist when calling B destructors
      std::vector<BLACSGrid> grids;
      {
        std::vector<DistMW_t> B(*Ninter);
        auto& comm = *(temp->c);
        auto rank = comm.rank();
        grids.reserve(*Ninter);
        auto data = alldat_loc;
        for (int isec=0, r0=0, c0=0; isec<*Ninter; isec++) {
          auto m = rowids[isec];
          auto n = colids[isec];
          for (int i=0; i<m; i++)
            I[isec].push_back(allrows[r0+i]-1);
          for (int i=0; i<n; i++)
            J[isec].push_back(std::abs(allcols[c0+i])-1);
          auto p0 = pmaps[2*(*Npmap)+pgids[isec]];
          assert(pmaps[pgids[isec]] == 1);          // prows == 1
          assert(pmaps[(*Npmap)+pgids[isec]] == 1); // pcols == 1
          grids.emplace_back(BLACSGrid(comm.sub(p0, 1), 1));
          B[isec] = DistMW_t(&grids[isec], m, n, data);
          r0 += m;
          c0 += n;
          if (rank == p0) data += m*n;
        }
        ExtractionMeta e
          {nullptr, *Ninter, *Nallrows, *Nallcols, *Nalldat_loc,
              allrows, allcols, rowids, colids, pgids, *Npmap, pmaps};
        temp->Aelem->operator()(I, J, B, e);
      }
    }

    template<typename scalar_t> void HODLR_block_evaluation_seq
    (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
     int* allrows, int* allcols, scalar_t* alldat_loc,
     int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
     C2Fptr f) {
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      std::vector<std::vector<std::size_t>> I(*Ninter), J(*Ninter);
      std::vector<DenseMW_t> B(*Ninter);
      auto data = alldat_loc;
      for (int isec=0, r0=0, c0=0; isec<*Ninter; isec++) {
        auto m = rowids[isec];
        auto n = colids[isec];
        I[isec].reserve(m);
        J[isec].reserve(n);
        for (int i=0; i<m; i++)
          I[isec].push_back(allrows[r0+i]-1);
        for (int i=0; i<n; i++)
          J[isec].push_back(std::abs(allcols[c0+i])-1);
        B[isec] = DenseMW_t(m, n, data, m);
        r0 += m;
        c0 += n;
        data += m*n;
      }
      ExtractionMeta e
        {nullptr, *Ninter, *Nallrows, *Nallcols, *Nalldat_loc,
            allrows, allcols, rowids, colids, pgids, *Npmap, pmaps};
      static_cast<typename HODLRMatrix<scalar_t>::elem_blocks_t*>
        (f)->operator()(I, J, B, e);
    }

	
    template<typename scalar_t> HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, kernel::Kernel<scalar_t>& K, int dim, std::vector<scalar_t>& geos, const opts_t& opts) {
      rows_ = cols_ = K.n();
      auto tree = binary_tree_clustering
        (opts.clustering_algorithm(), K.data(), K.permutation(), opts.leaf_size());
      int min_lvl = 2 + std::ceil(std::log2(c.size()));
      lvls_ = std::max(min_lvl, tree.levels());
      tree.expand_complete_levels(lvls_);
      leafs_ = tree.template leaf_sizes<int>();
      c_ = c;
      Fcomm_ = MPI_Comm_c2f(c_.comm());
      options_init(opts);
      perm_.resize(rows_);
      KernelCommPtrs<scalar_t> KC{&K, &c_};
	  HODLR_set_I_option<scalar_t>(options_, "nogeo", 0);
	  // HODLR_set_I_option<scalar_t>(options_, "xyzsort", 2);
	  HODLR_set_I_option<scalar_t>(options_, "knn", 10);
      HODLR_construct_init<scalar_t>
        (rows_, dim, geos.data(), lvls_-1, leafs_.data(), perm_.data(),
         lrows_, ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
         nullptr, nullptr, nullptr);
      HODLR_construct_element_compute<scalar_t>
        (ho_bf_, options_, stats_, msh_, kerquant_,
         ptree_, &(HODLR_kernel_evaluation<scalar_t>),
         &(HODLR_kernel_block_evaluation<scalar_t>), &KC);
      perm_init();
      dist_init();
    }	
	
	
    template<typename scalar_t> HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, kernel::Kernel<scalar_t>& K, const opts_t& opts) {
      rows_ = cols_ = K.n();
      auto tree = binary_tree_clustering
        (opts.clustering_algorithm(), K.data(), K.permutation(), opts.leaf_size());
      int min_lvl = 2 + std::ceil(std::log2(c.size()));
      lvls_ = std::max(min_lvl, tree.levels());
      tree.expand_complete_levels(lvls_);
      leafs_ = tree.template leaf_sizes<int>();
      c_ = c;
      Fcomm_ = MPI_Comm_c2f(c_.comm());
      options_init(opts);
      perm_.resize(rows_);
      KernelCommPtrs<scalar_t> KC{&K, &c_};
      HODLR_construct_init<scalar_t>
        (rows_, 0, nullptr, lvls_-1, leafs_.data(), perm_.data(),
         lrows_, ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
         nullptr, nullptr, nullptr);
      HODLR_construct_element_compute<scalar_t>
        (ho_bf_, options_, stats_, msh_, kerquant_,
         ptree_, &(HODLR_kernel_evaluation<scalar_t>),
         &(HODLR_kernel_block_evaluation<scalar_t>), &KC);
      perm_init();
      dist_init();
    }

    template<typename scalar_t> HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, const HSS::HSSPartitionTree& tree,
     const std::function<scalar_t(int i, int j)>& Aelem,
     const opts_t& opts) {
      rows_ = cols_ = tree.size;
      HSS::HSSPartitionTree full_tree(tree);
      int min_lvl = 2 + std::ceil(std::log2(c.size()));
      lvls_ = std::max(min_lvl, full_tree.levels());
      full_tree.expand_complete_levels(lvls_);
      leafs_ = full_tree.template leaf_sizes<int>();
      c_ = c;
      Fcomm_ = MPI_Comm_c2f(c_.comm());
      options_init(opts);
      perm_.resize(rows_);
      //KernelCommPtrs<scalar_t> KC{&K, &c_};
      HODLR_construct_init<scalar_t>
        (rows_, 0, nullptr, lvls_-1, leafs_.data(), perm_.data(),
         lrows_, ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
         nullptr, nullptr, nullptr);
      HODLR_set_I_option<scalar_t>(options_, "elem_extract", 0); // block extraction
      HODLR_construct_element_compute<scalar_t>
        (ho_bf_, options_, stats_, msh_, kerquant_,
         ptree_, &(HODLR_element_evaluation<scalar_t>),
         &(HODLR_block_evaluation<scalar_t>),
         const_cast<std::function<scalar_t(int i, int j)>*>(&Aelem));
      perm_init();
      dist_init();
    }

    template<typename scalar_t> HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, const HSS::HSSPartitionTree& tree,
     const mult_t& Amult, const opts_t& opts)
      : HODLRMatrix<scalar_t>(c, tree, opts) {
      compress(Amult);
    }

    template<typename scalar_t> HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, const HSS::HSSPartitionTree& tree,
     const opts_t& opts) {
      rows_ = cols_ = tree.size;
      HSS::HSSPartitionTree full_tree(tree);
      int min_lvl = 2 + std::ceil(std::log2(c.size()));
      lvls_ = std::max(min_lvl, full_tree.levels());
      full_tree.expand_complete_levels(lvls_);
      leafs_ = full_tree.template leaf_sizes<int>();
      c_ = c;
      if (c_.is_null()) return;
      Fcomm_ = MPI_Comm_c2f(c_.comm());
      options_init(opts);
      perm_.resize(rows_);
      HODLR_construct_init<scalar_t>
        (rows_, 0, nullptr, lvls_-1, leafs_.data(), perm_.data(),
         lrows_, ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
         nullptr, nullptr, nullptr);
      perm_init();
      dist_init();
    }

    template<typename integer_t> struct AdmInfo {
      std::pair<std::vector<int>,std::vector<int>> maps;
      const DenseMatrix<bool>* adm;
      const CSRGraph<integer_t>* graph;
    };

    template<typename scalar_t, typename integer_t,
             typename real_t = typename RealType<scalar_t>::value_type>
    void HODLR_distance_query(int* m, int* n, real_t* dist, C2Fptr fdata) {
      int i = *m - 1, j = *n - 1;
      if (i == j) { *dist = real_t(0.); return; }
      auto& info = *static_cast<AdmInfo<integer_t>*>(fdata);
      auto& g = *(info.graph);
      *dist = real_t(1.);
      auto pkhi = g.ind() + g.ptr(i+1);
      for (auto pk=g.ind() + g.ptr(i); pk!=pkhi; pk++)
        if (*pk == j) return;
      *dist = real_t(2.);
      for (auto pk=g.ind() + g.ptr(i); pk!=pkhi; pk++) {
        auto plhi = g.ind() + g.ptr(*pk+1);
        for (auto pl=g.ind() + g.ptr(*pk); pl!=plhi; pl++)
          if (*pl == j) return;
      }
      *dist = real_t(3.);
    }

    template<typename integer_t> void HODLR_admissibility_query
    (int* m, int* n, int* admissible, C2Fptr fdata) {
      auto& info = *static_cast<AdmInfo<integer_t>*>(fdata);
      auto& adm = *(info.adm);
      auto& map0 = info.maps.first;
      auto& map1 = info.maps.second;
      int r = *m - 1, c = *n - 1;
      bool a = true;
      for (int j=map0[c]; j<=map1[c] && a; j++)
        for (int i=map0[r]; i<=map1[r] && a; i++)
          a = a && adm(i, j);
      *admissible = a;
    }

    template<typename scalar_t> template<typename integer_t>
    HODLRMatrix<scalar_t>::HODLRMatrix
    (const MPIComm& c, const HSS::HSSPartitionTree& tree,
     const DenseMatrix<bool>& adm, const CSRGraph<integer_t>& graph,
     const opts_t& opts) {
      rows_ = cols_ = tree.size;
      HSS::HSSPartitionTree full_tree(tree);
      int min_lvl = 2 + std::ceil(std::log2(c.size()));
      lvls_ = std::max(min_lvl, full_tree.levels());
      full_tree.expand_complete_levels(lvls_);
      leafs_ = full_tree.template leaf_sizes<int>();
      c_ = c;
      if (c_.is_null()) return;
      Fcomm_ = MPI_Comm_c2f(c_.comm());
      options_init(opts);
      perm_.resize(rows_);
      if (opts.geo() == 2) {
        AdmInfo<integer_t> info;
        info.maps = tree.map_from_complete_to_leafs(lvls_);
        info.adm = &adm;
        info.graph = &graph;
        // use the distance and admissibility functions
        HODLR_set_I_option<scalar_t>(options_, "nogeo", 2);
        // nedge/nvert is the average degree, but since we also consider
        // length 2 connections, we need to consider more than that, 5
        // is just a heuristic.  For instance for a 2d 9-point stencil,
        // there are 3^2=9 points in the stencil and 5^2=25 points in
        // the extended (length 2 connections) stencil.
        HODLR_set_I_option<scalar_t>
          (options_, "knn", 5 * graph.edges() / graph.vertices());
        HODLR_construct_init<scalar_t>
          (rows_, 0, nullptr, lvls_-1, leafs_.data(), perm_.data(),
           lrows_, ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
           &(HODLR_distance_query<scalar_t,integer_t>),
           &(HODLR_admissibility_query<integer_t>), &info);
      } else
        HODLR_construct_init<scalar_t>
          (rows_, 0, nullptr, lvls_-1, leafs_.data(), perm_.data(),
           lrows_, ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
           nullptr, nullptr, nullptr);
      perm_init();
      dist_init();
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::options_init(const opts_t& opts) {
      auto P = c_.size();
      std::vector<int> groups(P);
      std::iota(groups.begin(), groups.end(), 0);

      // create hodlr data structures
      HODLR_createptree<scalar_t>(P, groups.data(), Fcomm_, ptree_);
      HODLR_createoptions<scalar_t>(options_);
      HODLR_createstats<scalar_t>(stats_);

      // set hodlr options
      HODLR_set_I_option<scalar_t>(options_, "verbosity", opts.verbose() ? 2 : -1);
      HODLR_set_I_option<scalar_t>(options_, "nogeo", 1);
      HODLR_set_I_option<scalar_t>(options_, "Nmin_leaf", rows_);
      // set RecLR_leaf to 2 for RRQR at bottom level of Hierarchical BACA
      HODLR_set_I_option<scalar_t>(options_, "RecLR_leaf", 5); // 5 = new version of BACA
      HODLR_set_I_option<scalar_t>(options_, "BACA_Batch", opts.BACA_block_size());
      HODLR_set_I_option<scalar_t>(options_, "xyzsort", 0);
      HODLR_set_I_option<scalar_t>(options_, "elem_extract", 1); // block extraction
      // set ErrFillFull to 1 to check acc for extraction code
      //HODLR_set_I_option<scalar_t>(options_, "ErrFillFull", opts.verbose() ? 1 : 0);
      HODLR_set_I_option<scalar_t>(options_, "ErrFillFull", 0);
      HODLR_set_I_option<scalar_t>(options_, "rank0", opts.rank_guess());
      HODLR_set_I_option<scalar_t>(options_, "cpp", 1);
      HODLR_set_I_option<scalar_t>(options_, "forwardN15flag", 0);
      HODLR_set_D_option<scalar_t>(options_, "sample_para", opts.BF_sampling_parameter());
      HODLR_set_D_option<scalar_t>(options_, "rankrate", opts.rank_rate());
      if (opts.butterfly_levels() > 0)
        HODLR_set_I_option<scalar_t>(options_, "LRlevel", opts.butterfly_levels());
      HODLR_set_D_option<scalar_t>(options_, "tol_comp", opts.rel_tol());
      HODLR_set_D_option<scalar_t>(options_, "tol_rand", opts.rel_tol());
      HODLR_set_D_option<scalar_t>(options_, "tol_Rdetect", 0.1*opts.rel_tol());
    }

    template<typename scalar_t> void HODLRMatrix<scalar_t>::perm_init() {
      iperm_.resize(rows_);
      MPI_Bcast(perm_.data(), perm_.size(), MPI_INT, 0, c_.comm());
      for (int i=0; i<rows_; i++) {
        perm_[i]--;
        iperm_[perm_[i]] = i;
      }
    }

    template<typename scalar_t> void HODLRMatrix<scalar_t>::dist_init() {
      auto P = c_.size();
      auto rank = c_.rank();
      dist_.resize(P+1);
      dist_[rank+1] = lrows_;
      MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                    dist_.data()+1, 1, MPI_INT, c_.comm());
      for (int p=0; p<P; p++) dist_[p+1] += dist_[p];
    }

    template<typename scalar_t> HODLRMatrix<scalar_t>::~HODLRMatrix() {
      if (stats_) HODLR_deletestats<scalar_t>(stats_);
      if (ptree_) HODLR_deleteproctree<scalar_t>(ptree_);
      if (msh_) HODLR_deletemesh<scalar_t>(msh_);
      if (kerquant_) HODLR_deletekernelquant<scalar_t>(kerquant_);
      if (ho_bf_) HODLR_delete<scalar_t>(ho_bf_);
      if (options_) HODLR_deleteoptions<scalar_t>(options_);
    }

    template<typename scalar_t> HODLRMatrix<scalar_t>&
    HODLRMatrix<scalar_t>::operator=(HODLRMatrix<scalar_t>&& h) {
      ho_bf_ = h.ho_bf_;       h.ho_bf_ = nullptr;
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
      std::swap(perm_, h.perm_);
      std::swap(iperm_, h.iperm_);
      std::swap(dist_, h.dist_);
      return *this;
    }

    template<typename scalar_t> void HODLR_matvec_routine
    (const char* op, int* nin, int* nout, int* nvec,
     const scalar_t* X, scalar_t* Y, C2Fptr func) {
      auto A = static_cast<typename HODLRMatrix<scalar_t>::mult_t*>(func);
      DenseMatrixWrapper<scalar_t> Yw(*nout, *nvec, Y, *nout),
        Xw(*nin, *nvec, const_cast<scalar_t*>(X), *nin);
      (*A)(c2T(*op), Xw, Yw);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::compress(const mult_t& Amult) {
      if (c_.is_null()) return;
      C2Fptr f = static_cast<void*>(const_cast<mult_t*>(&Amult));
      HODLR_construct_matvec_compute<scalar_t>
        (ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(HODLR_matvec_routine<scalar_t>), f);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::compress
    (const mult_t& Amult, int rank_guess) {
      HODLR_set_I_option<scalar_t>(options_, "rank0", rank_guess);
      compress(Amult);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::compress(const delem_blocks_t& Aelem) {
      AelemCommPtrs<scalar_t> AC{&Aelem, &c_};
      HODLR_construct_element_compute<scalar_t>
        (ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(HODLR_element_evaluation<scalar_t>),
         &(HODLR_block_evaluation<scalar_t>), &AC);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::compress(const elem_blocks_t& Aelem) {
      C2Fptr f = static_cast<void*>(const_cast<elem_blocks_t*>(&Aelem));
      HODLR_construct_element_compute<scalar_t>
        (ho_bf_, options_, stats_, msh_, kerquant_, ptree_,
         &(HODLR_element_evaluation<scalar_t>),
         &(HODLR_block_evaluation_seq<scalar_t>), f);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::mult
    (Trans op, const DenseM_t& X, DenseM_t& Y) const {
      if (c_.is_null()) return;
      HODLR_mult(char(op), X.data(), Y.data(), lrows_, lrows_, X.cols(),
                 ho_bf_, options_, stats_, ptree_);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::mult
    (Trans op, const DistM_t& X, DistM_t& Y) const {
      if (c_.is_null()) return;
      DenseM_t Y1D(lrows_, X.cols());
      {
        auto X1D = redistribute_2D_to_1D(X);
        HODLR_mult(char(op), X1D.data(), Y1D.data(), lrows_, lrows_,
                   X.cols(), ho_bf_, options_, stats_, ptree_);
      }
      redistribute_1D_to_2D(Y1D, Y);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::inv_mult
    (Trans op, const DenseM_t& X, DenseM_t& Y) const {
      if (c_.is_null()) return;
      HODLR_inv_mult
        (char(op), X.data(), Y.data(), lrows_, lrows_, X.cols(),
         ho_bf_, options_, stats_, ptree_);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::factor() {
      if (c_.is_null()) return;
      HODLR_factor<scalar_t>(ho_bf_, options_, stats_, ptree_, msh_);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::solve(const DenseM_t& B, DenseM_t& X) const {
      if (c_.is_null()) return;
      HODLR_solve(X.data(), B.data(), lrows_, X.cols(),
                  ho_bf_, options_, stats_, ptree_);
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::solve(const DistM_t& B, DistM_t& X) const {
      if (c_.is_null()) return;
      DenseM_t X1D(lrows_, X.cols());
      {
        auto B1D = redistribute_2D_to_1D(B);
        HODLR_solve(X1D.data(), B1D.data(), lrows_, X.cols(),
                    ho_bf_, options_, stats_, ptree_);
      }
      redistribute_1D_to_2D(X1D, X);
    }

    template<typename scalar_t> void HODLRMatrix<scalar_t>::extract_elements
    (const VecVec_t& I, const VecVec_t& J, std::vector<DistM_t>& B) {
      if (I.empty()) return;
      assert(I.size() == J.size() && I.size() == B.size());
      int Ninter = I.size(), total_rows = 0, total_cols = 0, total_dat = 0;
      int pmaps[3] = {B[0].nprows(), B[0].npcols(), 0};
      for (auto Ik : I) total_rows += Ik.size();
      for (auto Jk : J) total_cols += Jk.size();
      std::unique_ptr<int[]> iwork
        (new int[total_rows + total_cols + 3*Ninter]);
      auto allrows = iwork.get();
      auto allcols = allrows + total_rows;
      auto rowidx = allcols + total_cols;
      auto colidx = rowidx + Ninter;
      auto pgids = colidx + Ninter;
      for (int k=0, i=0, j=0; k<Ninter; k++) {
        assert(B[k].nprows() == pmaps[0]);
        assert(B[k].npcols() == pmaps[1]);
        total_dat += B[k].lrows()*B[k].lcols();
        rowidx[k] = I[k].size();
        colidx[k] = J[k].size();
        pgids[k] = 0;
        for (auto l : I[k]) { assert(int(l) < rows_); allrows[i++] = l+1; }
        for (auto l : J[k]) { assert(int(l) < cols_); allcols[j++] = l+1; }
      }
      for (int k=0; k<Ninter; k++)
        total_dat += B[k].lrows()*B[k].lcols();
      std::unique_ptr<scalar_t[]> alldat_loc(new scalar_t[total_dat]);
      auto ptr = alldat_loc.get();
      HODLR_extract_elements<scalar_t>
        (ho_bf_, options_, msh_, stats_, ptree_, Ninter,
         total_rows, total_cols, total_dat, allrows, allcols,
         ptr, rowidx, colidx, pgids, 1, pmaps);
      for (auto& Bk : B) {
        auto m = Bk.lcols();
        auto n = Bk.lrows();
        auto Bdata = Bk.data();
        auto Bld = Bk.ld();
        for (int j=0; j<m; j++)
          for (int i=0; i<n; i++)
            Bdata[i+j*Bld] = *ptr++;
      }
    }

    template<typename scalar_t> void HODLRMatrix<scalar_t>::extract_elements
    (const Vec_t& I, const Vec_t& J, DenseM_t& B) {
      int m = I.size(), n = J.size(), pgids = 0;
      if (m == 0 || n == 0) return;
      int pmaps[3] = {1, 1, 0};
      std::vector<int> Ii, Ji;
      Ii.assign(I.begin(), I.end());
      Ji.assign(J.begin(), J.end());
      for (auto& i : Ii) i++;
      for (auto& j : Ji) j++;
      HODLR_extract_elements<scalar_t>
        (ho_bf_, options_, msh_, stats_, ptree_, 1, m, n, m*n,
         Ii.data(), Ji.data(), B.data(), &m, &n, &pgids, 1, pmaps);
    }

    template<typename scalar_t> DistributedMatrix<scalar_t>
    HODLRMatrix<scalar_t>::dense(const BLACSGrid* g) const {
      DistM_t A(g, rows_, cols_), I(g, rows_, cols_);
      I.eye();
      mult(Trans::N, I, A);
      return A;
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    HODLRMatrix<scalar_t>::redistribute_2D_to_1D(const DistM_t& R2D) const {
      DenseM_t R1D(lrows_, R2D.cols());
      redistribute_2D_to_1D(R2D, R1D);
      return R1D;
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::redistribute_2D_to_1D
    (const DistM_t& R2D, DenseM_t& R1D) const {
      TIMER_TIME(TaskType::REDIST_2D_TO_HSS, 0, t_redist);
      if (c_.is_null()) return;
      const auto P = c_.size();
      const auto rank = c_.rank();
      // for (int p=0; p<P; p++)
      //   copy(dist_[rank+1]-dist_[rank], R2D.cols(), R2D, dist_[rank], 0,
      //        R1D, p, R2D.grid()->ctxt_all());
      // return;
      const auto Rcols = R2D.cols();
      int R2Drlo, R2Drhi, R2Dclo, R2Dchi;
      R2D.lranges(R2Drlo, R2Drhi, R2Dclo, R2Dchi);
      const auto Rlcols = R2Dchi - R2Dclo;
      const auto Rlrows = R2Drhi - R2Drlo;
      const auto nprows = R2D.nprows();
      std::vector<std::vector<scalar_t>> sbuf(P);
      if (R2D.active()) {
        // global, local, proc
        std::vector<std::tuple<int,int,int>> glp(Rlrows);
        {
          std::vector<std::size_t> count(P);
          for (int r=R2Drlo; r<R2Drhi; r++) {
            auto gr = perm_[R2D.rowl2g(r)];
            auto p = -1 + std::distance
              (dist_.begin(), std::upper_bound
               (dist_.begin(), dist_.end(), gr));
            glp[r-R2Drlo] = std::tuple<int,int,int>{gr, r, p};
            count[p] += Rlcols;
          }
          std::sort(glp.begin(), glp.end());
          for (int p=0; p<P; p++)
            sbuf[p].reserve(count[p]);
        }
        for (int r=R2Drlo; r<R2Drhi; r++)
          for (int c=R2Dclo, lr=std::get<1>(glp[r-R2Drlo]),
                 p=std::get<2>(glp[r-R2Drlo]); c<R2Dchi; c++)
            sbuf[p].push_back(R2D(lr,c));
      }
      std::vector<scalar_t> rbuf;
      std::vector<scalar_t*> pbuf;
      c_.all_to_all_v(sbuf, rbuf, pbuf);
      assert(int(R1D.rows()) == lrows_ && int(R1D.cols()) == Rcols);
      if (lrows_) {
        std::vector<int> src_c(Rcols);
        for (int c=0; c<Rcols; c++)
          src_c[c] = R2D.colg2p_fixed(c)*nprows;
        for (int r=0; r<lrows_; r++) {
          auto gr = perm_[r + dist_[rank]];
          auto src_r = R2D.rowg2p_fixed(gr);
          for (int c=0; c<Rcols; c++)
            R1D(r, c) = *(pbuf[src_r + src_c[c]]++);
        }
      }
    }

    template<typename scalar_t> void
    HODLRMatrix<scalar_t>::redistribute_1D_to_2D
    (const DenseM_t& S1D, DistM_t& S2D) const {
      TIMER_TIME(TaskType::REDIST_2D_TO_HSS, 0, t_redist);
      if (c_.is_null()) return;
      const int rank = c_.rank();
      const int P = c_.size();
      const int cols = S1D.cols();
      int S2Drlo, S2Drhi, S2Dclo, S2Dchi;
      S2D.lranges(S2Drlo, S2Drhi, S2Dclo, S2Dchi);
      const auto nprows = S2D.nprows();
      std::vector<std::vector<scalar_t>> sbuf(P);
      assert(int(S1D.rows()) == lrows_);
      assert(int(S1D.rows()) == dist_[rank+1] - dist_[rank]);
      if (lrows_) {
        std::vector<std::tuple<int,int,int>> glp(lrows_);
        for (int r=0; r<lrows_; r++) {
          auto gr = iperm_[r + dist_[rank]];
          // assert(gr == r + dist_[rank]);
          assert(gr >= 0 && gr < S2D.rows());
          glp[r] = std::tuple<int,int,int>{gr,r,S2D.rowg2p_fixed(gr)};
        }
        std::sort(glp.begin(), glp.end());
        std::vector<int> pc(cols);
        for (int c=0; c<cols; c++)
          pc[c] = S2D.colg2p_fixed(c)*nprows;
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
      c_.all_to_all_v(sbuf, rbuf, pbuf);
      if (S2D.active()) {
        for (int r=S2Drlo; r<S2Drhi; r++) {
          auto gr = perm_[S2D.rowl2g(r)];
          assert(gr == S2D.rowl2g(r));
          auto p = -1 + std::distance
            (dist_.begin(), std::upper_bound(dist_.begin(), dist_.end(), gr));
          assert(p < P && p >= 0);
          for (int c=S2Dclo; c<S2Dchi; c++) {
            auto tmp = *(pbuf[p]++);
            S2D(r,c) = tmp;
          }
        }
      }
    }

  } // end namespace HODLR
} // end namespace strumpack

#endif // STRUMPACK_HODLR_MATRIX_HPP
