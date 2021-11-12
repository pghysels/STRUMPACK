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
 * \file StrumpackSparseSolverMPIDist.hpp
 * \brief Contains the definition of fully distributed sparse solver class.
 */
#ifndef STRUMPACK_SPARSE_SOLVER_MPI_DIST_HPP
#define STRUMPACK_SPARSE_SOLVER_MPI_DIST_HPP

#include "SparseSolverBase.hpp"
#include "dense/ScaLAPACKWrapper.hpp"
#include "dense/DistributedVector.hpp"
#include "sparse/CSRMatrixMPI.hpp"

namespace strumpack {

  // forward declarations
  template<typename scalar_t,typename integer_t> class EliminationTreeMPIDist;
  template<typename scalar_t,typename integer_t> class MatrixReorderingMPI;

  /**
   * \class SparseSolverMPIDist
   *
   * \brief This is the fully distributed solver.
   *
   * All steps are distributed: the symbolic factorization, the
   * factorization and the solve. The sparse factors (fill-in) are
   * distributed over the MPI processes. Only the MC64 phase is not
   * distributed. If MC64 reordering is enabled, the sparse input
   * matrix will be gathered to a single process, where then MC64 is
   * called. This can be a bottleneck for large matrices. Try to
   * disable MC64, since for many matrices, this reordering is not
   * required.
   *
   * \tparam scalar_t can be: float, double, std::complex<float> or
   * std::complex<double>.
   *
   * \tparam integer_t defaults to a regular int. If regular int
   * causes 32 bit integer overflows, you should switch to
   * integer_t=int64_t instead. This should be a __signed__ integer
   * type.
   *
   * \see SparseSolver
   */
  template<typename scalar_t,typename integer_t>
  class SparseSolverMPIDist :
    public SparseSolverBase<scalar_t,integer_t> {

    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using Tree_t = EliminationTree<scalar_t,integer_t>;
    using Reord_t = MatrixReordering<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;

  public:
    /**
     * Constructor taking an MPI communicator and command line
     * arguments. This routine is collective on all ranks in the MPI
     * communicator comm.
     *
     * \param mpi_comm MPI communicator.  Can be MPI_COMM_WORLD or a
     * subcommunicator.
     * \param argc The number of arguments, i.e, number of elements in
     * the argv array.
     * \param argv Command line arguments. Add -h or --help to have a
     * description printed.
     * \param verb Flag to suppres/enable output.  Only the root of
     * comm will print to stdout.
     */
    SparseSolverMPIDist(MPI_Comm comm, int argc, char* argv[],
                        bool verbose=true);

    /**
     * Constructor of the SparseSolver class. This routine is
     * collective on all ranks in the MPI communicator comm.
     *
     * \param verbose flag to enable/disable output to cout
     * \param root flag to denote whether this process is the root MPI
     * process. Only the root will print certain messages
     * \see set_from_options
     */
    SparseSolverMPIDist(MPI_Comm comm, bool verbose=true);

    /**
     * Destructor, virtual.
     */
    ~SparseSolverMPIDist();

    /**
     * Set a matrix for this sparse solver. __Only the matrix provided
     * by the root process (in comm()) will be referenced.__ The input
     * matrix will immediately be distributed over all the processes
     * in the communicator associated with the solver. This routine is
     * collective on the MPI communicator from this solver.
     *
     * \param A input sparse matrix, should only be provided on the
     * root process. The matrix will be copied internally, so it can
     * be safely modified/deleted after calling this function.
     * \see set_csr_matrix
     */
    void broadcast_matrix(const CSRMatrix<scalar_t,integer_t>& A);

    /**
     * Set a matrix for this sparse solver. __Only the matrix provided
     * by the root process (in comm()) will be referenced.__ The input
     * matrix will immediately be distributed over all the processes
     * in the communicator associated with the solver. This routine is
     * collective on the MPI communicator from this solver.
     *
     * \param N number of rows/column in the input matrix
     * \param row_ptr row pointers, points to the start of each row in
     * the col_ind and values arrays. Array of size N+1, with
     * row_ptr[N+1] the total number of nonzeros
     * \param col_ind for each nonzeros, the column index
     * \param values nonzero values
     * \param symmetric_pattern denote whether the sparsity pattern
     * (not the numerical values) of the sparse inut matrix is
     * symmetric.
     *
     * \see set_matrix, broadcast_matrix
     */
    void broadcast_csr_matrix
    (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
     const scalar_t* values, bool symmetric_pattern=false);

    /**
     * Set a (distributed) matrix for this sparse solver. This matrix
     * will not be modified. An internal copy will be made, so it is
     * safe to delete the data immediately after calling this
     * function. This routine is collective on the MPI communicator
     * associated with the solver.
     *
     * \param A input sparse matrix, should be provided on all
     * ranks. The matrix will be copied internally, so it can be
     * safely modified/deleted after calling this function.
     * \see set_csr_matrix
     */
    void set_matrix(const CSRMatrixMPI<scalar_t,integer_t>& A);

    /**
     * Associate a block-row distributed CSR matrix with the solver
     * object.
     *
     * \param local_rows the number of rows of the input matrix assigned
     * to this MPI process.  This should equal to
     * dist[rank+1]-dist[rank].
     * \param row_ptr indices in col_ind and values for the start
     * of each row. Nonzeros for row r+dist[rank] are
     * in [row_ptr[r],row_ptr[r+1]).
     * \param col_ind column indices of each nonzero.
     * \param values nonzero values. Should have at least
     * (row_ptr[dist[p+1]-dist[p]]-row_ptr[0]) elements.
     * \param dist specifies the block-row distribution. A process
     * with rank p owns rows [dist[p],dist[p+1]).
     * \param symmetric_pattern denotes whether the sparsity pattern
     * (not the numerical values) of the input matrix is symmetric.
     *
     * \see set_matrix, set_csr_matrix
     */
    void set_distributed_csr_matrix
    (integer_t local_rows, const integer_t* row_ptr,
     const integer_t* col_ind, const scalar_t* values,
     const integer_t* dist, bool symmetric_pattern=false);

    /**
     * Associate a (PETSc) MPIAIJ block-row distributed CSR matrix
     * with the solver object. See the PETSc manual for a description
     * of MPIAIJ matrix format.
     */
    void set_MPIAIJ_matrix
    (integer_t local_rows,
     const integer_t* d_ptr, const integer_t* d_ind, const scalar_t* d_val,
     const integer_t* o_ptr, const integer_t* o_ind, const scalar_t* o_val,
     const integer_t* garray);


    /**
     * This can only be used to UPDATE the nonzero values of the
     * matrix. So it should be called with exactly the same sparsity
     * pattern (row_ptr and col_ind) and distribution as used to set
     * the initial matrix (using set_matrix or
     * set_distributed_csr_matrix). This routine can be called after
     * having performed a factorization of a different matrix with the
     * same sparsity pattern. In that case, when this solver is used
     * for another solve, with the updated matrix values, the
     * permutation vector previously computed will be reused to
     * permute the updated matrix values, instead of recomputing the
     * permutation. The numerical factorization will automatically be
     * redone.
     *
     * \param local_rows Number of rows of the matrix assigned to this
     * process.
     * \param row_ptr Row pointer array in the typical compressed
     * sparse row representation. This should be the same as used in
     * an earlier call to set_csr_matrix.
     * \param col_ind Column index array in the typical compressed
     * sparse row representation. This should be the same as used in
     * an earlier call to set_csr_matrix.
     * \param values Array with numerical nonzero values for the
     * matrix, corresponding to the row_ptr and col_ind compressed
     * sparse row representation.
     * \param dist Describes the processor distribution, see also
     * set_distributed_csr_matrix
     * \param symmetric_pattern Denotes whether the sparsity
     * __pattern__ of the input matrix is symmetric, does not require
     * the matrix __values__ to be symmetric
     *
     * \see set_csr_matrix, set_matrix
     */
    void update_matrix_values
    (integer_t local_rows, const integer_t* row_ptr,
     const integer_t* col_ind, const scalar_t* values,
     const integer_t* dist, bool symmetric_pattern=false);

    /**
     * This can only be used to UPDATE the nonzero values of the
     * matrix. So it should be called with exactly the same sparsity
     * pattern (row_ptr and col_ind) and distribution as used to set
     * the initial matrix (using set_matrix or
     * set_distributed_csr_matrix). This routine can be called after
     * having performed a factorization of a different matrix with the
     * same sparsity pattern. In that case, when this solver is used
     * for another solve, with the updated matrix values, the
     * permutation vector previously computed will be reused to
     * permute the updated matrix values, instead of recomputing the
     * permutation. The numerical factorization will automatically be
     * redone.
     *
     * \param A Sparse matrix, should have the same sparsity pattern
     * as the matrix associated with this solver earlier.
     *
     * \see set_csr_matrix, set_matrix
     */
    void update_matrix_values(const CSRMatrixMPI<scalar_t,integer_t>& A);

    /**
     * This can only be used to UPDATE the nonzero values of the
     * matrix. So it should be called with exactly the same sparsity
     * pattern (d_ptr, d_ind, o_ptr and o_ind) and distribution
     * (garray) as used to set the initial matrix (using set_matrix or
     * set_distributed_csr_matrix). This routine can be called after
     * having performed a factorization of a different matrix with the
     * same sparsity pattern. In that case, when this solver is used
     * for another solve, with the updated matrix values, the
     * permutation vector previously computed will be reused to
     * permute the updated matrix values, instead of recomputing the
     * permutation. The numerical factorization will automatically be
     * redone.
     *
     * \see set_MPIAIJ_matrix
     */
    void update_MPIAIJ_matrix_values
    (integer_t local_rows, const integer_t* d_ptr, const integer_t* d_ind,
     const scalar_t* d_val, const integer_t* o_ptr, const integer_t* o_ind,
     const scalar_t* o_val, const integer_t* garray);

    /**
     * Return the MPI_Comm object associated with this solver.
     * \return MPI_Comm object for this solver.
     */
    MPI_Comm comm() const;

    /**
     * Return the const MPIComm object associated with this solver.
     * \return MPIComm object for this solver.
     */
    const MPIComm& Comm() const { return comm_; }

  private:
    using SparseSolverBase<scalar_t,integer_t>::is_root_;
    using SparseSolverBase<scalar_t,integer_t>::opts_;
    MPIComm comm_;

    SpMat_t* matrix() override { return mat_mpi_.get(); }
    Reord_t* reordering() override;
    Tree_t* tree() override { return tree_mpi_dist_.get(); }
    const SpMat_t* matrix() const override { return mat_mpi_.get(); }
    const Reord_t* reordering() const override;
    const Tree_t* tree() const override { return tree_mpi_dist_.get(); }

    void setup_tree() override;
    void setup_reordering() override;
    int compute_reordering(const int* p, int base, int nx, int ny, int nz,
                           int components, int width) override;
    void separator_reordering() override;

    void perf_counters_stop(const std::string& s) override;
    void synchronize() override { comm_.barrier(); }
    void reduce_flop_counters() const override;

    double max_peak_memory() const override {
      return comm_.reduce(double(params::peak_memory), MPI_MAX);
    }
    double min_peak_memory() const override {
      return comm_.reduce(double(params::peak_memory), MPI_MIN);
    }

    void redistribute_values();

    void delete_factors_internal() override;

    ReturnCode
    solve_internal(const scalar_t* b, scalar_t* x,
                   bool use_initial_guess=false) override;
    ReturnCode
    solve_internal(const DenseM_t& b, DenseM_t& x,
                   bool use_initial_guess=false) override;
    ReturnCode
    solve_internal(int nrhs, const scalar_t* b, int ldb,
                   scalar_t* x, int ldx,
                   bool use_initial_guess=false) override;

    std::unique_ptr<CSRMatrixMPI<scalar_t,integer_t>> mat_mpi_;
    std::unique_ptr<MatrixReorderingMPI<scalar_t,integer_t>> nd_mpi_;
    std::unique_ptr<EliminationTreeMPIDist<scalar_t,integer_t>> tree_mpi_dist_;
  };

  template<typename scalar_t,typename integer_t>
  using StrumpackSparseSolverMPIDist =
    SparseSolverMPIDist<scalar_t,integer_t>;

} // end namespace strumpack

#endif // STRUMPACK_SPARSE_SOLVER_MPI_DIST_HPP
