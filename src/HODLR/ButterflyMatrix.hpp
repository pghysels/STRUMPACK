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
/*! \file ButterflyMatrix.hpp
 * \brief Classes wrapping around Yang Liu's butterfly code.
 */
#ifndef STRUMPACK_BUTTERFLY_MATRIX_HPP
#define STRUMPACK_BUTTERFLY_MATRIX_HPP

#include "HODLRMatrix.hpp"

namespace strumpack {

  /**
   * Code in this namespace is a wrapper aroung Yang Liu's Fortran
   * code: https://github.com/liuyangzhuan/hod-lr-bf
   */
  namespace HODLR {

    /**
     * \class ButterflyMatrix
     *
     * \brief Butterfly matrix representation, this includes low-rank
     * matrix representation as a special case.
     *
     * This requires MPI support.
     *
     * There are 2 different ways to create a ButterflyMatrix
     *  - By specifying a matrix-(multiple)vector multiplication
     *    routine.
     *  - By specifying an element extraction routine.
     *
     * \tparam scalar_t Can be double, or std::complex<double>.
     *
     * \see HSS::HSSMatrix, HODLR::HODLRMatrix
     */
    template<typename scalar_t> class ButterflyMatrix
      : public structured::StructuredMatrix<scalar_t> {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using DistM_t = DistributedMatrix<scalar_t>;
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      using opts_t = HODLROptions<scalar_t>;
      using VecVec_t = std::vector<std::vector<std::size_t>>;
      using F2Cptr = void*;

    public:
      using mult_t = typename std::function
        <void(Trans,scalar_t,const DenseM_t&,scalar_t,DenseM_t&)>;
      using delem_blocks_t = typename HODLRMatrix<scalar_t>::delem_blocks_t;
      using elem_blocks_t = typename HODLRMatrix<scalar_t>::elem_blocks_t;

      ButterflyMatrix() {}

      ButterflyMatrix(const MPIComm& comm,
                      const structured::ClusterTree& row_tree,
                      const structured::ClusterTree& col_tree,
                      const opts_t& opts);

      /**
       * Construct the block X, subblock of the matrix [A X; Y B]
       * A and B should be defined on the same MPI communicator.
       */
      ButterflyMatrix(const HODLRMatrix<scalar_t>& A,
                      const HODLRMatrix<scalar_t>& B);

      ButterflyMatrix(const HODLRMatrix<scalar_t>& A,
                      const HODLRMatrix<scalar_t>& B,
                      DenseMatrix<int>& neighbors_rows,
                      DenseMatrix<int>& neighbors_cols,
                      const opts_t& opts);

      ButterflyMatrix(const ButterflyMatrix<scalar_t>& h) = delete;
      ButterflyMatrix(ButterflyMatrix<scalar_t>&& h) { *this = std::move(h); }
      virtual ~ButterflyMatrix();
      ButterflyMatrix<scalar_t>& operator=(const ButterflyMatrix<scalar_t>& h) = delete;
      ButterflyMatrix<scalar_t>& operator=(ButterflyMatrix<scalar_t>&& h);

      std::size_t rows() const override { return rows_; }
      std::size_t cols() const override { return cols_; }
      std::size_t lrows() const { return lrows_; }
      std::size_t local_rows() const override { return lrows_; }
      std::size_t lcols() const { return lcols_; }
      std::size_t begin_row() const override { return rdist_[c_->rank()]; }
      std::size_t end_row() const override { return rdist_[c_->rank()+1]; }
      const std::vector<int>& rdist() const override { return rdist_; }
      std::size_t begin_col() const { return cdist_[c_->rank()]; }
      std::size_t end_col() const { return cdist_[c_->rank()+1]; }
      const std::vector<int>& cdist() const override { return cdist_; }
      const MPIComm& Comm() const { return *c_; }

      std::size_t memory() const override { return get_stat("Mem_Fill") * 1e6; }
      std::size_t nonzeros() const override { return memory() / sizeof(scalar_t); }
      std::size_t rank() const override { return get_stat("Rank_max"); }

      void compress(const mult_t& Amult);
      void compress(const mult_t& Amult, int rank_guess);
      void compress(const delem_blocks_t& Aelem);
      void compress(const elem_blocks_t& Aelem);

      void mult(Trans op, const DenseM_t& X, DenseM_t& Y) const override;

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
      void mult(Trans op, const DistM_t& X, DistM_t& Y) const override;

      void extract_add_elements(const VecVec_t& I, const VecVec_t& J,
                                std::vector<DenseMW_t>& B);
      void extract_add_elements(ExtractionMeta& e, std::vector<DistMW_t>& B);
      void extract_add_elements(ExtractionMeta& e, std::vector<DenseMW_t>& B);

      double get_stat(const std::string& name) const;

      void print_stats();

      void set_sampling_parameter(double sample_param);
      void set_BACA_block(int bsize);

      DistM_t dense(const BLACSGrid* g) const;

      DenseM_t redistribute_2D_to_1D(const DistM_t& R2D,
                                     const std::vector<int>& dist) const;
      void redistribute_2D_to_1D(scalar_t a, const DistM_t& R2D,
                                 scalar_t b, DenseM_t& R1D,
                                 const std::vector<int>& dist) const;
      void redistribute_1D_to_2D(const DenseM_t& S1D, DistM_t& S2D,
                                 const std::vector<int>& dist) const;

    private:
      F2Cptr lr_bf_ = nullptr;     // Butterfly handle returned by Fortran code
      F2Cptr options_ = nullptr;   // options structure returned by Fortran code
      F2Cptr stats_ = nullptr;     // statistics structure returned by Fortran code
      F2Cptr msh_ = nullptr;       // mesh structure returned by Fortran code
      F2Cptr kerquant_ = nullptr;  // kernel quantities structure returned by Fortran code
      F2Cptr ptree_ = nullptr;     // process tree returned by Fortran code
      MPI_Fint Fcomm_;             // the fortran MPI communicator
      const MPIComm* c_;
      int rows_ = 0, cols_ = 0, lrows_ = 0, lcols_ = 0;
      std::vector<int> rdist_, cdist_;  // begin rows/cols of each rank

      void set_dist();

      void options_init(const opts_t& opts);

      void set_extraction_meta_1grid(const VecVec_t& I, const VecVec_t& J,
                                     ExtractionMeta& e,
                                     int Nalldat_loc, int* pmaps) const;
    };

    template<typename integer_t> DenseMatrix<int>
    get_odiag_neighbors(int knn, const CSRGraph<integer_t>& gAB,
                        const CSRGraph<integer_t>& gA,
                        const CSRGraph<integer_t>& gB);

  } // end namespace HODLR
} // end namespace strumpack

#endif // STRUMPACK_BUTTERFLY_MATRIX_HPP
