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
/*!
 * \file CSRMatrix.hpp
 * \brief Contains the compressed sparse row matrix storage class.
 */
#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <vector>

#include "CompressedSparseMatrix.hpp"
#include "CSRGraph.hpp"

namespace strumpack {

  /**
   * \class CSRMatrix
   * \brief Class for storing a compressed sparse row matrix (single
   * node).
   *
   * \tparam scalar_t
   * \tparam integer_t
   *
   * \see CompressedSparseMatrix, CSRMatrixMPI
   */
  template<typename scalar_t,typename integer_t> class CSRMatrix
    : public CompressedSparseMatrix<scalar_t,integer_t> {
#if defined(STRUMPACK_USE_MPI)
    using DistM_t = DistributedMatrix<scalar_t>;
#endif
    using CSM_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using real_t = typename RealType<scalar_t>::value_type;

  public:
    CSRMatrix();
    CSRMatrix(integer_t n, integer_t nnz);
    CSRMatrix
    (integer_t n, const integer_t* ptr, const integer_t* ind,
     const scalar_t* values, bool symm_sparsity=false);

    void spmv(const DenseM_t& x, DenseM_t& y) const override;
    void spmv(const scalar_t* x, scalar_t* y) const override;

    void spmv(Trans op, const DenseM_t& x, DenseM_t& y) const;

    void apply_scaling
    (const std::vector<scalar_t>& Dr,
     const std::vector<scalar_t>& Dc) override;
    void apply_column_permutation
    (const std::vector<integer_t>& perm) override;
    real_t max_scaled_residual
    (const scalar_t* x, const scalar_t* b) const override;
    real_t max_scaled_residual
    (const DenseM_t& x, const DenseM_t& b) const override;
    void strumpack_mc64
    (int_t job, int_t* num, integer_t* perm, int_t liw, int_t* iw, int_t ldw,
     double* dw, int_t* icntl, int_t* info) override;
    int read_matrix_market(const std::string& filename) override;
    int read_binary(const std::string& filename);
    void print_dense(const std::string& name) const override;
    void print_matrix_market(const std::string& filename) const override;
    void print_binary(const std::string& filename) const;

    CSRGraph<integer_t> extract_graph
    (int ordering_level, integer_t lo, integer_t hi) const override;
    CSRGraph<integer_t> extract_graph_sep_CB
    (int ordering_level, integer_t lo, integer_t hi,
     const std::vector<integer_t>& upd) const override;
    CSRGraph<integer_t> extract_graph_CB_sep
    (int ordering_level, integer_t lo, integer_t hi,
     const std::vector<integer_t>& upd) const override;
    CSRGraph<integer_t> extract_graph_CB
    (int ordering_level, const std::vector<integer_t>& upd) const override;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    // TODO implement these outside of this class
    void front_multiply
    (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
     const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc, int depth) const override;
    void extract_separator
    (integer_t sep_end, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J,
     DenseM_t& B, int depth) const override;
    void extract_front
    (DenseM_t& F11, DenseM_t& F12, DenseM_t& F21, integer_t sep_begin,
     integer_t sep_end, const std::vector<integer_t>& upd,
     int depth) const override;

    void front_multiply_F11
    (Trans op, integer_t slo, integer_t shi,
     const DenseM_t& R, DenseM_t& S, int depth) const override;
    void front_multiply_F12
    (Trans op, integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
     const DenseM_t& R, DenseM_t& S, int depth) const override;
    void front_multiply_F21
    (Trans op, integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
     const DenseM_t& R, DenseM_t& S, int depth) const override;

#if defined(STRUMPACK_USE_MPI)
    void extract_F11_block
    (scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows,
     integer_t col, integer_t nr_cols) const override;
    void extract_F12_block
    (scalar_t* F, integer_t ldF, integer_t row,
     integer_t nr_rows, integer_t col, integer_t nr_cols,
     const integer_t* upd) const override;
    void extract_F21_block
    (scalar_t* F, integer_t ldF, integer_t row,
     integer_t nr_rows, integer_t col, integer_t nr_cols,
     const integer_t* upd) const override;
    void extract_separator_2d
    (integer_t sep_end, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J, DistM_t& B) const override;
    void front_multiply_2d
    (integer_t sep_begin, integer_t sep_end,
     const std::vector<integer_t>& upd, const DistM_t& R,
     DistM_t& Srow, DistM_t& Scol, int depth) const override;
    void front_multiply_2d
    (Trans op, integer_t sep_begin, integer_t sep_end,
     const std::vector<integer_t>& upd, const DistM_t& R,
     DistM_t& S, int depth) const override;
#endif //defined(STRUMPACK_USE_MPI)
#endif //DOXYGEN_SHOULD_SKIP_THIS

  private:
    using CSM_t::n_;
    using CSM_t::nnz_;
    using CSM_t::ptr_;
    using CSM_t::ind_;
    using CSM_t::val_;
    using CSM_t::symm_sparse_;
  };

} // end namespace strumpack

#endif // CSR_MATRIX_HPP
