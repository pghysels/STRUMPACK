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
 * \file SparseSolver.hpp
 * \brief Contains the definition of the sequential/multithreaded
 * sparse solver class.
 */
#ifndef STRUMPACK_SPARSE_SOLVER_HPP
#define STRUMPACK_SPARSE_SOLVER_HPP

#include <new>
#include <memory>
#include <vector>
#include <string>

#include "SparseSolverBase.hpp"

/**
 * All of STRUMPACK is contained in the strumpack namespace.
 */
namespace strumpack {

  // forward declarations
  template<typename scalar_t,typename integer_t> class MatrixReordering;
  template<typename scalar_t,typename integer_t> class EliminationTree;
  class TaskTimer;

  /**
   * \class SparseSolver
   *
   * \brief SparseSolver is the main sequential or
   * multithreaded sparse solver class.
   *
   * This is the main interface to STRUMPACK's sparse solver. Use this
   * for a sequential or multithreaded sparse solver. For the fully
   * distributed solver, see SparseSolverMPIDist.
   *
   * \tparam scalar_t can be: float, double, std::complex<float> or
   * std::complex<double>.
   *
   * \tparam integer_t defaults to a regular int. If regular int
   * causes 32 bit integer overflows, you should switch to
   * integer_t=int64_t instead. This should be a __signed__ integer
   * type.
   *
   * \see SparseSolverMPIDist
   */
  template<typename scalar_t,typename integer_t=int>
  class SparseSolver :
    public SparseSolverBase<scalar_t,integer_t> {

    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using Tree_t = EliminationTree<scalar_t,integer_t>;
    using Reord_t = MatrixReordering<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;

  public:

    /**
     * Constructor of the SparseSolver class, taking command line
     * arguments.
     *
     * \param argc number of arguments, i.e, number of elements in
     * the argv array
     * \param argv command line arguments. Add -h or --help to have a
     * description printed
     * \param verbose flag to enable/disable output to cout
     * \param root flag to denote whether this process is the root MPI
     * process, only the root will print certain messages to cout
     */
    SparseSolver(int argc, char* argv[], bool verbose=true, bool root=true);

    /**
     * Constructor of the SparseSolver class.
     *
     * \param verbose flag to enable/disable output to cout
     * \param root flag to denote whether this process is the root MPI
     * process. Only the root will print certain messages
     * \see set_from_options
     */
    SparseSolver(bool verbose=true, bool root=true);

    /**
     * (Virtual) destructor of the SparseSolver class.
     */
    ~SparseSolver();

    /**
     * Associate a (sequential) CSRMatrix with this solver.
     *
     * This matrix will not be modified. An internal copy will be
     * made, so it is safe to delete the data immediately after
     * calling this function.
     *
     * \param A A CSRMatrix<scalar_t,integer_t> object, will
     * internally be duplicated
     *
     * \see set_csr_matrix
     */
    void set_matrix(const CSRMatrix<scalar_t,integer_t>& A);

    /**
     * Associate a (sequential) NxN CSR matrix with this solver.
     *
     * This matrix will not be modified. An internal copy will be
     * made, so it is safe to delete the data immediately after
     * calling this function. See the manual for a description of the
     * CSR format. You can also use the CSRMatrix class.
     *
     * \param N number of rows and columns of the CSR input matrix.
     * \param row_ptr indices in col_ind and values for the start of
     * each row. Nonzeros for row r are in [row_ptr[r],row_ptr[r+1])
     * \param col_ind column indices of each nonzero
     * \param values nonzero values
     * \param symmetric_pattern denotes whether the sparsity
     * __pattern__ of the input matrix is symmetric, does not require
     * the matrix __values__ to be symmetric
     *
     * \see set_matrix
     */
    void set_csr_matrix(integer_t N,
                        const integer_t* row_ptr, const integer_t* col_ind,
                        const scalar_t* values, bool symmetric_pattern=false);

    /**
     * This can only be used to UPDATE the nonzero values of the
     * matrix. So it should be called with exactly the same sparsity
     * pattern (row_ptr and col_ind) as used to set the initial matrix
     * (using set_matrix or set_csr_matrix). This routine can be
     * called after having performed a factorization of a different
     * matrix with the same sparsity pattern. In that case, when this
     * solver is used for another solve, with the updated matrix
     * values, the permutation vector previously computed will be
     * reused to permute the updated matrix values, instead of
     * recomputing the permutation. The numerical factorization will
     * automatically be redone.
     *
     * \param N Number of rows in the matrix.
     * \param row_ptr Row pointer array in the typical compressed
     * sparse row representation. This should be the same as used in
     * an earlier call to set_csr_matrix.
     * \param col_ind Column index array in the typical compressed
     * sparse row representation. This should be the same as used in
     * an earlier call to set_csr_matrix.
     * \param values Array with numerical nonzero values for the
     * matrix, corresponding to the row_ptr and col_ind compressed
     * sparse row representation.
     * \param symmetric_pattern Denotes whether the sparsity
     * __pattern__ of the input matrix is symmetric, does not require
     * the matrix __values__ to be symmetric
     *
     * \see set_csr_matrix, set_matrix
     */
    void update_matrix_values(integer_t N,
                              const integer_t* row_ptr,
                              const integer_t* col_ind,
                              const scalar_t* values,
                              bool symmetric_pattern=false);

    /**
     * This can only be used to UPDATE the nonzero values of the
     * matrix. So it should be called with exactly the same sparsity
     * pattern as used to set the initial matrix (using set_matrix or
     * set_csr_matrix). This routine can be called after having
     * performed a factorization of a different matrix with the same
     * sparsity pattern. In that case, when this solver is used for
     * another solve, with the updated matrix values, the permutation
     * vector previously computed will be reused to permute the
     * updated matrix values, instead of recomputing the
     * permutation. The numerical factorization will automatically be
     * redone.
     *
     * \param A Sparse matrix, should have the same sparsity pattern
     * as the matrix associated with this solver earlier.
     *
     * \see set_csr_matrix, set_matrix
     */
    void update_matrix_values(const CSRMatrix<scalar_t,integer_t>& A);

  private:
    void setup_tree() override;
    void setup_reordering() override;
    int compute_reordering(const int* p, int base,
                           int nx, int ny, int nz,
                           int components, int width) override;
    void separator_reordering() override;

    SpMat_t* matrix() override { return mat_.get(); }
    std::unique_ptr<SpMat_t> matrix_nonzero_diag() override {
      return mat_->add_missing_diagonal(opts_.pivot_threshold());
    }
    Reord_t* reordering() override { return nd_.get(); }
    Tree_t* tree() override { return tree_.get(); }
    const SpMat_t* matrix() const override { return mat_.get(); }
    const Reord_t* reordering() const override { return nd_.get(); }
    const Tree_t* tree() const override { return tree_.get(); }

    void permute_matrix_values();

    ReturnCode solve_internal(const scalar_t* b, scalar_t* x,
                              bool use_initial_guess=false) override;
    ReturnCode solve_internal(const DenseM_t& b, DenseM_t& x,
                              bool use_initial_guess=false) override;

    void delete_factors_internal() override;

    void transform_x0(DenseM_t& x, DenseM_t& xtmp);
    void transform_b(const DenseM_t& b, DenseM_t& bloc);
    void transform_x(DenseM_t& x, DenseM_t& xtmp);

    std::unique_ptr<CSRMatrix<scalar_t,integer_t>> mat_;
    std::unique_ptr<MatrixReordering<scalar_t,integer_t>> nd_;
    std::unique_ptr<EliminationTree<scalar_t,integer_t>> tree_;

    using SPBase_t = SparseSolverBase<scalar_t,integer_t>;
    using SPBase_t::opts_;
    using SPBase_t::is_root_;
    using SPBase_t::matching_;
    using SPBase_t::equil_;
    using SPBase_t::factored_;
    using SPBase_t::reordered_;
    using SPBase_t::Krylov_its_;
  };

  template<typename scalar_t,typename integer_t>
  using StrumpackSparseSolver = SparseSolver<scalar_t,integer_t>;

} //end namespace strumpack

#endif // STRUMPACK_SPARSE_SOLVER_HPP
