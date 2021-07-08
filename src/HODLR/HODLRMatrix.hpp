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
#include <functional>

#include "kernel/Kernel.hpp"
#include "dense/DistributedMatrix.hpp"
#include "HODLROptions.hpp"
#include "sparse/CSRGraph.hpp"
#include "structured/StructuredMatrix.hpp"

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
     * \tparam scalar_t Can be double, or std::complex<double>.
     *
     * \see HSS::HSSMatrix
     */
    template<typename scalar_t> class HODLRMatrix
      : public structured::StructuredMatrix<scalar_t> {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using DistM_t = DistributedMatrix<scalar_t>;
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      using opts_t = HODLROptions<scalar_t>;
      using Vec_t = std::vector<std::size_t>;
      using VecVec_t = std::vector<std::vector<std::size_t>>;
      using F2Cptr = void*;

    public:
      using real_t = typename RealType<scalar_t>::value_type;
      using mult_t = typename std::function
        <void(Trans, const DenseM_t&, DenseM_t&)>;
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
      HODLRMatrix(const MPIComm& c, kernel::Kernel<real_t>& K,
                  const opts_t& opts);

      /**
       * Construct an HODLR approximation using a routine to evaluate
       * individual matrix elements. This will construct an
       * approximation of a permuted matrix, with the permutation
       * available through this->perm() (and it's inverse
       * this->iperm()). This permutation is applied symmetrically to
       * rows and columns.
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
      HODLRMatrix(const MPIComm& c, const structured::ClusterTree& tree,
                  const std::function<scalar_t(int i, int j)>& Aelem,
                  const opts_t& opts);


      /**
       * Construct an HODLR approximation using a routine to evaluate
       * multiple sub-blocks of the matrix at once. This will
       * construct an approximation of a permuted matrix, with the
       * permutation available through this->perm() (and it's inverse
       * this->iperm()). This permutation is applied symmetrically to
       * rows and columns.
       *
       * \param c MPI communicator, this communicator is copied
       * internally.
       * \param t tree specifying the HODLR matrix partitioning
       * \param Aelem Routine, std::function, which can also be a
       * lambda function or a functor. This should have the signature:
       * void(std::vector<std::vector<std::size_t>>& I,
       *       std::vector<std::vector<std::size_t>>& J,
       *       std::vector<DenseMatrixWrapper<scalar_t>>& B,
       *       ExtractionMeta&);
       * The ExtractionMeta object can be ignored.
       * \param opts object containing a number of HODLR options
       */
      HODLRMatrix(const MPIComm& c, const structured::ClusterTree& tree,
                  const elem_blocks_t& Aelem, const opts_t& opts);


      /**
       * Construct an HODLR matrix using a specified HODLR tree and
       * matrix-vector multiplication routine.
       *
       * \param c MPI communicator, this communicator is copied
       * internally.
       * \param t tree specifying the HODLR matrix partitioning
       * \param Amult Routine for the matrix-vector product. Trans op
       * argument will be N, T or C for none, transpose or complex
       * conjugate. The const DenseM_t& argument is the random
       * matrix R, and the final DenseM_t& argument S is what the user
       * routine should compute as A*R, A^t*R or A^c*R. S will already
       * be allocated.
       * \param opts object containing a number of options for HODLR
       * compression
       * \see compress, HODLROptions
       */
      HODLRMatrix(const MPIComm& c, const structured::ClusterTree& tree,
                  const std::function<
                  void(Trans op, const DenseM_t& R, DenseM_t& S)>& Amult,
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
      HODLRMatrix(const MPIComm& c, const structured::ClusterTree& tree,
                  const opts_t& opts);

      /**
       * Construct an HODLR matrix using a specified HODLR tree. After
       * construction, the HODLR matrix will be empty, and can be filled
       * by calling one of the compress member routines.
       *
       * \param c MPI communicator, this communicator is copied
       * internally.
       * \param t tree specifying the HODLR matrix partitioning
       * \param graph connectivity info for the dofs
       * \param opts object containing a number of options for HODLR
       * compression
       * \see compress, HODLROptions
       */
      template<typename integer_t>
      HODLRMatrix(const MPIComm& c, const structured::ClusterTree& tree,
                  const CSRGraph<integer_t>& graph, const opts_t& opts);

      /**
       * Copy constructor is not supported.
       */
      HODLRMatrix(const HODLRMatrix<scalar_t>& h) = delete;

      /**
       * Move constructor.
       * \param h HODLRMatrix to move from, will be emptied.
       */
      HODLRMatrix(HODLRMatrix<scalar_t>&& h) { *this = std::move(h); }

      /**
       * Virtual destructor.
       */
      virtual ~HODLRMatrix();

      /**
       * Copy assignment operator is not supported.
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
      std::size_t rows() const override { return rows_; }
      /**
       * Return the number of columns in the matrix.
       * \return Global number of columns in the matrix.
       */
      std::size_t cols() const override { return cols_; }
      /**
       * Return the number of local rows, owned by this process.
       * \return Number of local rows.
       */
      std::size_t lrows() const { return lrows_; }
      /**
       * Return the number of local rows, owned by this process.
       * \return Number of local rows.
       */
      std::size_t local_rows() const override { return lrows(); }
      /**
       * Return the first row of the local rows owned by this process.
       * \return Return first local row
       */
      std::size_t begin_row() const override { return dist_[c_->rank()]; }
      /**
       * Return last row (+1) of the local rows (begin_rows()+lrows())
       * \return Final local row (+1).
       */
      std::size_t end_row() const override { return dist_[c_->rank()+1]; }
      /**
       * Return vector describing the 1d block row
       * distribution. dist()[rank]==begin_row() and
       * dist()[rank+1]==end_row()
       * \return 1D block row distribution
       */
      const std::vector<int>& dist() const override { return dist_; }

      /**
       * Return MPI communicator wrapper object.
       */
      const MPIComm& Comm() const { return *c_; }

      /**
       * Return the memory for this HODLR matrix, on this rank, in
       * bytes.
       */
      std::size_t memory() const override {
        return get_stat("Mem_Fill") * 1024 * 1024;
      }

      std::size_t nonzeros() const override {
        return memory() / sizeof(scalar_t);
      }

      /**
       * Return the total memory for this HODLR matrix, summed over
       * all ranks, in bytes.
       */
      std::size_t total_memory() const {
        return c_->all_reduce(memory(), MPI_SUM);
      }

      /**
       * Return the additional memory for the factorization of this
       * HODLR matrix, in bytes. This memory is only for the
       * additional storage for the factorization, not the HODLR
       * matrix itself.
       */
      std::size_t factor_memory() const {
        return get_stat("Mem_Factor") * 1.e6;
      }

      /**
       * Return the total additional memory for the factorization of
       * this HODLR matrix, in bytes, summed over all processes in
       * Comm(). This call is collective on Comm(). This memory is
       * only for the additional storage for the factorization, not
       * the HODLR matrix itself.
       */
      std::size_t total_factor_memory() const {
        return c_->all_reduce(factor_memory(), MPI_SUM);
      }

      /**
       * Return the maximal rank encountered in this HODLR matrix.
       */
      std::size_t rank() const override { return get_stat("Rank_max"); }

      /**
       * Return the maximal rank encountered in this HODLR matrix,
       * taking the maximum over all processes. This is collective on
       * Comm().
       */
      std::size_t max_rank() const {
        return c_->all_reduce(rank(), MPI_MAX);
      }


      /**
       * Get certain statistics about the HODLR matrix.  See the HODLR
       * code at https://github.com/liuyangzhuan/ButterflyPACK for
       * more info.
       *
       * \param name fi : Rank_max, Mem_Factor, Mem_Fill, Flop_Fill,
       * Flop_Factor, Flop_C_Mult
       */
      double get_stat(const std::string& name) const;

      void print_stats();

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
      void compress(const std::function
                    <void(Trans op, const DenseM_t& R,
                          DenseM_t& S)>& Amult);

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
      void compress(const std::function
                    <void(Trans op, const DenseM_t& R,
                          DenseM_t& S)>& Amult,
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
      void mult(Trans op, const DenseM_t& X, DenseM_t& Y) const override;

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
      void mult(Trans op, const DistM_t& X, DistM_t& Y) const override;

      /**
       * Compute the factorization of this HODLR matrix. The matrix
       * can still be used for multiplication.
       *
       * \see solve, inv_mult
       */
      void factor() override;

      /**
       * Solve a system of linear equations A*X=B, with possibly
       * multiple right-hand sides. X and B are distributed using 1D
       * block row distribution.
       *
       * \param B Right hand side. This is the local part of
       * the distributed matrix B. Should be B.rows() == this.lrows().
       * \param X Result, should be X.cols() == B.cols(), X.rows() ==
       * this.lrows(). X should be allocated.
       * \return number of flops
       * \see factor, lrows, begin_row, end_row, inv_mult
       */
      long long int solve(const DenseM_t& B, DenseM_t& X) const;

      /**
       * Solve a system of linear equations A*X=B, with possibly
       * multiple right-hand sides. X and B are distributed using 1D
       * block row distribution. The solution X will overwrite the
       * right-hand side vector B.
       *
       * \param B Right hand side. This is the local part of the
       * distributed matrix B. Should be B.rows() == this.lrows(). Wil
       * lbe overwritten with the solution.
       *
       * \see factor, lrows, begin_row, end_row, inv_mult
       */
      void solve(DenseM_t& B) const override;

      /**
       * Solve a system of linear equations A*X=B, with possibly
       * multiple right-hand sides. X and B are in 2D block cyclic
       * distribution.
       *
       * \param B Right hand side. This is the local part of
       * the distributed matrix B. Should be B.rows() == this.rows().
       * \param X Result, should be X.cols() == B.cols(), X.rows() ==
       * this.rows(). X should be allocated.
       * \return number of flops
       *
       * \see factor, inv_mult
       */
      long long int solve(const DistM_t& B, DistM_t& X) const;

      /**
       * Solve a system of linear equations A*X=B, with possibly
       * multiple right-hand sides. X and B are in 2D block cyclic
       * distribution. The solution X will overwrite the righ-hand
       * side B.
       *
       * \param B Right hand side. This is the local part of the
       * distributed matrix B. Should be B.rows() == this.rows(). Will
       * be overwritten by the solution.
       *
       * \see factor, inv_mult, solve
       */
      void solve(DistM_t& B) const override;

      /**
       * Solve a system of linear equations op(A)*X=B, with possibly
       * multiple right-hand sides, where op can be none, transpose or
       * complex conjugate.
       *
       * \param B Right hand side. This is the local part of
       * the distributed matrix B. Should be B.rows() == this.lrows().
       * \param X Result, should be X.cols() == B.cols(), X.rows() ==
       * this.lrows(). X should be allocated.
       * \return number of flops
       * \see factor, solve, lrows, begin_row, end_row
       */
      long long int inv_mult(Trans op, const DenseM_t& B, DenseM_t& X) const;

      void extract_elements(const VecVec_t& I, const VecVec_t& J,
                            std::vector<DistM_t>& B);
      void extract_elements(const VecVec_t& I, const VecVec_t& J,
                            std::vector<DenseM_t>& B);

      /**
       * Extract a submatrix defined by index sets I (rows) and J
       * (columns), and put the result in matrix B (only on the
       * root!).
       *
       * \param I set of row indices, needs to be specified on all
       * ranks!
       * \param J set of column indices, needs to be specified on all
       * ranks!
       * \param B output, the extracted elements. B should have the
       * correct size B.rows() == I.size() and B.cols() == J.size()
       */
      void extract_elements(const Vec_t& I, const Vec_t& J, DenseM_t& B);

      /**
       * Create a dense matrix from this HODLR compressed matrix. This
       * is mainly for debugging.
       *
       * \param g BLACSFGrid to be used for the output
       * \return dense matrix representation of *this, in 2D bloc
       * cyclic format.
       */
      DistM_t dense(const BLACSGrid* g) const;

      void set_sampling_parameter(double sample_param);
      void set_BACA_block(int bsize);

      DenseM_t redistribute_2D_to_1D(const DistM_t& R) const;
      void redistribute_2D_to_1D(const DistM_t& R2D, DenseM_t& R1D) const;
      void redistribute_1D_to_2D(const DenseM_t& S1D, DistM_t& S2D) const;

      DenseM_t gather_from_1D(const DenseM_t& A) const;
      DenseM_t all_gather_from_1D(const DenseM_t& A) const;
      DenseM_t scatter_to_1D(const DenseM_t& A) const;

      /**
       * The permutation for the matrix, which is applied to both rows
       * and columns. This is 1-based!.
       */
      const std::vector<int>& perm() const { return perm_; }
      /**
       * The inverse permutation for the matrix, which is applied to
       * both rows and columns. This is 1-based!.
       */
      const std::vector<int>& iperm() const { return iperm_; }

    private:
      F2Cptr ho_bf_ = nullptr;     // HODLR handle returned by Fortran code
      F2Cptr options_ = nullptr;   // options structure returned by Fortran code
      F2Cptr stats_ = nullptr;     // statistics structure returned by Fortran code
      F2Cptr msh_ = nullptr;       // mesh structure returned by Fortran code
      F2Cptr kerquant_ = nullptr;  // kernel quantities structure returned by Fortran code
      F2Cptr ptree_ = nullptr;     // process tree returned by Fortran code
      MPI_Fint Fcomm_;             // the fortran MPI communicator
      const MPIComm* c_;
      int rows_ = 0, cols_ = 0, lrows_ = 0, lvls_ = 0;
      std::vector<int> perm_, iperm_; // permutation used by the HODLR code
      std::vector<int> dist_;         // begin rows of each rank
      std::vector<int> leafs_;        // leaf sizes of the tree

      void options_init(const opts_t& opts);
      void perm_init();
      void dist_init();

      template<typename S> friend class ButterflyMatrix;
    };


    template<typename scalar_t> struct AelemCommPtrs {
      const typename HODLRMatrix<scalar_t>::delem_blocks_t* Aelem;
      const MPIComm* c;
      const BLACSGrid* gl;
      const BLACSGrid* g0;
    };

    template<typename scalar_t> void HODLR_block_evaluation
    (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
     int* allrows, int* allcols, scalar_t* alldat_loc,
     int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
     void* AC);

    template<typename scalar_t> void HODLR_block_evaluation_seq
    (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
     int* allrows, int* allcols, scalar_t* alldat_loc,
     int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
     void* f);

  } // end namespace HODLR
} // end namespace strumpack

#endif // STRUMPACK_HODLR_MATRIX_HPP
