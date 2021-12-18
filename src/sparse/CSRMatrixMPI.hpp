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
 */
/**
 * \file CSRMatrixMPI.hpp
 * \brief Contains the CSRMatrixMPI class definition, a class
 * representing a block-row distributed compressed sparse row matrix.
 */
#ifndef STRUMPACK_CSRMATRIX_MPI_HPP
#define STRUMPACK_CSRMATRIX_MPI_HPP

#include <vector>
#include <tuple>
#include <memory>

#include "CSRMatrix.hpp"
#include "misc/MPIWrapper.hpp"
#include "dense/BLASLAPACKWrapper.hpp"
#include "CSRGraph.hpp"


namespace strumpack {

  template<typename scalar_t,typename integer_t> class SPMVBuffers {
  public:
    bool initialized = false;
    std::vector<integer_t> sranks, rranks, soff, roffs, sind;
    std::vector<scalar_t> sbuf, rbuf;
    // for each off-diagonal entry spmv_prbuf stores the
    // corresponding index in the receive buffer
    std::vector<integer_t> prbuf;
  };


  /**
   * \class CSRMatrixMPI
   * \brief Block-row distributed compressed sparse row storage.
   *
   * TODO: cleanup this class
   *  - use MPIComm
   *  - store the block diagonal as a CSRMatrix
   *  - ...
   *
   * \tparam scalar_t ...
   * \tparam integer_t  TODO set a default for this?
   */
  template<typename scalar_t,typename integer_t>
  class CSRMatrixMPI : public CompressedSparseMatrix<scalar_t,integer_t> {
    using DistM_t = DistributedMatrix<scalar_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using CSM_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using real_t = typename RealType<scalar_t>::value_type;
    using Match_t = MatchingData<scalar_t,integer_t>;
    using Equil_t = Equilibration<scalar_t>;

  public:
    CSRMatrixMPI();
    CSRMatrixMPI(integer_t local_rows, const integer_t* row_ptr,
                 const integer_t* col_ind, const scalar_t* values,
                 const integer_t* dist, MPIComm comm, bool symm_sparse);
    CSRMatrixMPI(integer_t lrows, const integer_t* d_ptr,
                 const integer_t* d_ind, const scalar_t* d_val,
                 const integer_t* o_ptr, const integer_t* o_ind,
                 const scalar_t* o_val, const integer_t* garray,
                 MPIComm comm, bool symm_sparse=false);
    CSRMatrixMPI(const CSRMatrix<scalar_t,integer_t>* A, MPIComm c,
                 bool only_at_root);

    integer_t local_nnz() const { return lnnz_; }
    integer_t local_rows() const { return lrows_; }
    integer_t begin_row() const { return brow_; }
    integer_t end_row() const { return brow_ + lrows_; }
    MPIComm Comm() const { return comm_; }
    MPI_Comm comm() const { return comm_.comm(); }

    const std::vector<integer_t>& dist() const { return dist_; }
    const integer_t& dist(std::size_t p) const {
      assert(p < dist_.size());
      return dist_[p];
    }

    real_t norm1() const override;

    void spmv(const DenseM_t& x, DenseM_t& y) const override;
    void spmv(const scalar_t* x, scalar_t* y) const override;

    void permute(const integer_t* iorder, const integer_t* order) override;

    std::unique_ptr<CSRMatrix<scalar_t,integer_t>> gather() const;
    std::unique_ptr<CSRGraph<integer_t>> gather_graph() const;


    /**
     * This gathers the matrix to 1 process, then applies MC64
     * sequentially. lDr and gDc are only set when job ==
     * MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING.
     *
     * \param job The job type.
     * \param perm Output, column permutation vector containing the
     * GLOBAL column permutation, such that the column perm[j] of the
     * original matrix is column j in the permuted matrix.
     * \param lDr Row scaling factors, this is local, ie, Dr.size() ==
     * this->local_rows().
     * \param gDc Col scaling factors, this is global, ie, Dc.size()
     * == this->size()
     */
    Match_t matching(MatchingJob job, bool apply=true) override;

    Equil_t equilibration() const override;

    void equilibrate(const Equil_t&) override;

    void permute_columns(const std::vector<integer_t>& perm) override;

    void symmetrize_sparsity() override;

    int read_matrix_market(const std::string& filename) override;

    real_t max_scaled_residual(const DenseM_t& x, const DenseM_t& b)
      const override;

    real_t max_scaled_residual(const scalar_t* x, const scalar_t* b)
      const override;

    CSRGraph<integer_t>
    get_sub_graph(const std::vector<integer_t>& perm,
                  const std::vector<std::pair<integer_t,integer_t>>&
                  graph_ranges) const;

    CSRGraph<integer_t>
    extract_graph(int ordering_level,
                  integer_t lo, integer_t hi) const override {
      assert(false);
      return CSRGraph<integer_t>();
    }

    void print() const override;
    void print_dense(const std::string& name) const override;
    void print_matrix_market(const std::string& filename) const override;
    void check() const;


#ifndef DOXYGEN_SHOULD_SKIP_THIS
    // implement outside of this class
    void extract_separator
    (integer_t, const std::vector<std::size_t>&,
     const std::vector<std::size_t>&, DenseM_t&, int) const override {}
    void extract_separator_2d
    (integer_t, const std::vector<std::size_t>&,
     const std::vector<std::size_t>&, DistM_t&) const override {}
    void extract_front
    (DenseM_t&, DenseM_t&, DenseM_t&, integer_t,
     integer_t, const std::vector<integer_t>&, int) const override {}
    void push_front_elements
    (integer_t, integer_t, const std::vector<integer_t>&,
     std::vector<Triplet<scalar_t>>&, std::vector<Triplet<scalar_t>>&,
     std::vector<Triplet<scalar_t>>&) const override {}
    void set_front_elements
    (integer_t, integer_t, const std::vector<integer_t>&,
     Triplet<scalar_t>*, Triplet<scalar_t>*,
     Triplet<scalar_t>*) const override {}
    void count_front_elements
    (integer_t, integer_t, const std::vector<integer_t>&,
     std::size_t&, std::size_t&, std::size_t&) const override {}

    void extract_F11_block
    (scalar_t*, integer_t, integer_t, integer_t,
     integer_t, integer_t) const override {}
    void extract_F12_block
    (scalar_t*, integer_t, integer_t, integer_t, integer_t,
     integer_t, const integer_t*) const override {}
    void extract_F21_block
    (scalar_t*, integer_t, integer_t, integer_t, integer_t,
     integer_t, const integer_t*) const override {}
    void front_multiply
    (integer_t, integer_t, const std::vector<integer_t>&,
     const DenseM_t&, DenseM_t&, DenseM_t&, int depth) const override {}
    void front_multiply_2d
    (integer_t, integer_t, const std::vector<integer_t>&, const DistM_t&,
     DistM_t&, DistM_t&, int) const override {}
    void front_multiply_2d
    (Trans op, integer_t, integer_t, const std::vector<integer_t>&,
     const DistM_t&, DistM_t&, int) const override {}

    CSRGraph<integer_t> extract_graph_sep_CB
    (int ordering_level, integer_t lo, integer_t hi,
     const std::vector<integer_t>& upd) const override {
      return CSRGraph<integer_t>(); };
    CSRGraph<integer_t> extract_graph_CB_sep
    (int ordering_level, integer_t lo, integer_t hi,
     const std::vector<integer_t>& upd) const override {
      return CSRGraph<integer_t>(); };
    CSRGraph<integer_t> extract_graph_CB
    (int ordering_level, const std::vector<integer_t>& upd) const override {
      return CSRGraph<integer_t>(); };

    void front_multiply_F11
    (Trans op, integer_t slo, integer_t shi,
     const DenseM_t& R, DenseM_t& S, int depth) const override {};
    void front_multiply_F12
    (Trans op, integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
     const DenseM_t& R, DenseM_t& S, int depth) const override {};
    void front_multiply_F21
    (Trans op, integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
     const DenseM_t& R, DenseM_t& S, int depth) const override {};
#endif //DOXYGEN_SHOULD_SKIP_THIS

  protected:
    void split_diag_offdiag();
    void setup_spmv_buffers() const;

    // TODO use MPIComm
    MPIComm comm_;

    /**
     * dist_ is the same as the vtxdist array defined by parmetis, it
     *  is the same for each process processor p holds rows
     *  [dist_[p],dist_[p+1]-1]
     */
    std::vector<integer_t> dist_;

    /**
     * points to the start of the off-(block)-diagonal
     * elements.
     */
    std::vector<integer_t> offdiag_start_;

    integer_t brow_;    // = dist_[rank], dist_[rank+1]
    integer_t lrows_;   // = erow_ - brow_
    integer_t lnnz_;    // = ptr_[local_rows]

    mutable SPMVBuffers<scalar_t,integer_t> spmv_bufs_;


    /**
     * Apply row and column scaling. lDr is LOCAL, gDc is global!
     */
    void scale(const std::vector<scalar_t>& lDr,
               const std::vector<scalar_t>& gDc) override;
    void scale_real(const std::vector<real_t>& lDr,
                    const std::vector<real_t>& gDc) override;

    using CSM_t::n_;
    using CSM_t::nnz_;
    using CSM_t::ptr_;
    using CSM_t::ind_;
    using CSM_t::val_;
    using CSM_t::symm_sparse_;
  };

  /**
   * Creates a copy of a matrix templated on cast_t and
   * integer_t. Original matrix is unmodified.
   *
   * \tparam scalar_t value type of original matrix
   * \tparam integer_t integer type of original matrix
   * \tparam cast_t value type of returned matrix
   *
   * \param mat const CSRMatrix<scalar_t,integer_t>&, const ref. of
   * input matrix.
   */
  template<typename scalar_t, typename integer_t, typename cast_t>
  CSRMatrixMPI<cast_t,integer_t>
  cast_matrix(const CSRMatrixMPI<scalar_t,integer_t>& mat);

} // end namespace strumpack

#endif // STRUMPACK_CSRMATRIX_MPI_HPP
