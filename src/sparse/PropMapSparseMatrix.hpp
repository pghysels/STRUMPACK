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
#ifndef PROP_MAP_SPARSE_MATRIX_HPP
#define PROP_MAP_SPARSE_MATRIX_HPP

#include <vector>

#include "CSRGraph.hpp"
#include "CompressedSparseMatrix.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class EliminationTreeMPIDist;
  template<typename scalar_t,typename integer_t> class MatrixReorderingMPI;
  template<typename scalar_t,typename integer_t> class CSRMatrixMPI;
  template<typename scalar_t> class DistributedMatrix;

  /**
   * \class ProportionallyDistributedSparseMatrix
   *
   * \brief Sparse matrix distributed based on the proportional
   * mapping of the elimination tree, only to be used during the
   * multifrontal factorization phase. It does not implement stuff
   * like a general spmv (use original matrix for that).
   *
   * __TODO__ How is this stored?
   *
   * __TODO__ This needs some performance improvements! Try to use
   * optimized (MKL) routines for the spmv (in the random sampling
   * code).
   */
  template<typename scalar_t,typename integer_t>
  class PropMapSparseMatrix :
    public CompressedSparseMatrix<scalar_t,integer_t> {
    using DistM_t = DistributedMatrix<scalar_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using CSM_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using real_t = typename RealType<scalar_t>::value_type;

  public:
    PropMapSparseMatrix();

    /**
     * duplicate_fronts should be set to true when sampling with the
     * front is required using 2d block cyclic vectors.
     */
    void setup(const CSRMatrixMPI<scalar_t,integer_t>& Ampi,
               const MatrixReorderingMPI<scalar_t,integer_t>& nd,
               const EliminationTreeMPIDist<scalar_t,integer_t>& et,
               bool duplicate_fronts);

    void print_dense(const std::string& name) const override;

    void extract_separator(integer_t shi, const std::vector<std::size_t>& I,
                           const std::vector<std::size_t>& J,
                           DenseM_t& B, int depth) const override;
    void extract_front(DenseM_t& F11, DenseM_t& F12, DenseM_t& F21,
                       integer_t slo, integer_t shi,
                       const std::vector<integer_t>& upd,
                       int depth) const override;

    void push_front_elements(integer_t, integer_t,
                             const std::vector<integer_t>&,
                             std::vector<Triplet<scalar_t>>&,
                             std::vector<Triplet<scalar_t>>&,
                             std::vector<Triplet<scalar_t>>&) const override;
    void set_front_elements(integer_t, integer_t,
                            const std::vector<integer_t>&,
                            Triplet<scalar_t>*, Triplet<scalar_t>*,
                            Triplet<scalar_t>*) const override;
    void count_front_elements(integer_t, integer_t,
                              const std::vector<integer_t>&,
                              std::size_t&, std::size_t&, std::size_t&)
      const override;

    void extract_F11_block(scalar_t* F, integer_t ldF,
                           integer_t row, integer_t nr_rows,
                           integer_t col, integer_t nr_cols) const override;
    void extract_F12_block(scalar_t* F, integer_t ldF,
                           integer_t row, integer_t nr_rows,
                           integer_t col, integer_t nr_cols,
                           const integer_t* upd) const override;
    void extract_F21_block(scalar_t* F, integer_t ldF,
                           integer_t row, integer_t nr_rows,
                           integer_t col, integer_t nr_cols,
                           const integer_t* upd) const override;
    void extract_separator_2d(integer_t shi,
                              const std::vector<std::size_t>& I,
                              const std::vector<std::size_t>& J,
                              DistM_t& B) const override ;

    void front_multiply(integer_t slo, integer_t shi,
                        const std::vector<integer_t>& upd,
                        const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc,
                        int depth) const override;
    void front_multiply_F11(Trans op, integer_t slo, integer_t shi,
                            const DenseM_t& R, DenseM_t& S,
                            int depth) const override;
    void front_multiply_F12(Trans op, integer_t slo, integer_t shi,
                            const std::vector<integer_t>& upd,
                            const DenseM_t& R, DenseM_t& S,
                            int depth) const override;
    void front_multiply_F21(Trans op, integer_t slo, integer_t shi,
                            const std::vector<integer_t>& upd,
                            const DenseM_t& R, DenseM_t& S,
                            int depth) const override;

    void front_multiply_2d(integer_t slo, integer_t shi,
                           const std::vector<integer_t>& upd,
                           const DistM_t& R, DistM_t& Srow, DistM_t& Scol,
                           int depth) const override;
    void front_multiply_2d(Trans op, integer_t slo, integer_t shi,
                           const std::vector<integer_t>& upd,
                           const DistM_t& R, DistM_t& S,
                           int depth) const override {
      if (op == Trans::N)
        front_multiply_2d_N(slo, shi, upd, R, S, depth);
      else front_multiply_2d_TC(slo, shi, upd, R, S, depth);
    }

    CSRGraph<integer_t>
    extract_graph(int ordering_level, integer_t lo, integer_t hi)
      const override;
    CSRGraph<integer_t>
    extract_graph_sep_CB(int ordering_level, integer_t lo, integer_t hi,
                         const std::vector<integer_t>& upd) const override;
    CSRGraph<integer_t>
    extract_graph_CB_sep(int ordering_level, integer_t lo, integer_t hi,
                         const std::vector<integer_t>& upd) const override;
    CSRGraph<integer_t>
    extract_graph_CB(int ordering_level,
                     const std::vector<integer_t>& upd) const override;

    void spmv(const DenseM_t& x, DenseM_t& y) const override {};
    void spmv(const scalar_t* x, scalar_t* y) const override {};

    real_t norm1() const override { return real_t(1.); }

    void permute_columns(const std::vector<integer_t>& perm) override {};

    int read_matrix_market(const std::string& filename) override { return 1; };

    real_t max_scaled_residual(const scalar_t* x,
                               const scalar_t* b) const override {
      return real_t(1.);
    };
    real_t max_scaled_residual(const DenseM_t& x,
                               const DenseM_t& b) const override {
      return real_t(1.);
    };

    void permute(const integer_t* iorder, const integer_t* order) override;

  protected:
    integer_t local_cols_;  // number of columns stored on this proces
    std::vector<integer_t> global_col_; // for each local column, this
                                        // gives the global column
                                        // index

    integer_t find_global(integer_t c, integer_t clo=0) const {
      // TODO create a loopkup vector
      return std::distance
        (global_col_.begin(),
         std::lower_bound(global_col_.begin()+clo, global_col_.end(), c));
    }

    void scale(const std::vector<scalar_t>& Dr,
               const std::vector<scalar_t>& Dc) override {};
    void scale_real(const std::vector<real_t>& Dr,
                    const std::vector<real_t>& Dc) override {};

    void front_multiply_2d_N(integer_t slo, integer_t shi,
                             const std::vector<integer_t>& upd,
                             const DistM_t& R, DistM_t& S, int depth) const;
    void front_multiply_2d_TC(integer_t slo, integer_t shi,
                              const std::vector<integer_t>& upd,
                              const DistM_t& R, DistM_t& S, int depth) const;

    using CSM_t::n_;
    using CSM_t::nnz_;
    using CSM_t::ptr_;
    using CSM_t::ind_;
    using CSM_t::val_;
  };

} // end namespace strumpack

#endif // PRO_MAP_SPARSE_MATRIX_HPP
