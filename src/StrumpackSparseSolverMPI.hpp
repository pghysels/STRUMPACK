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
 * \file StrumpackSparseSolverMPI.hpp
 * \brief Contains the definition of MPI distributed sparse solver
 * class with REPLICATED input.
 */
#ifndef STRUMPACK_SPARSE_SOLVER_MPI_HPP
#define STRUMPACK_SPARSE_SOLVER_MPI_HPP

#include "StrumpackSparseSolver.hpp"
#include "misc/MPIWrapper.hpp"


namespace strumpack {

  // forward declatations
  template<typename scalar_t,typename integer_t> class MatrixReorderingMPI;
  template<typename scalar_t,typename integer_t> class EliminationTreeMPI;

  /**
   * \class StrumpackSparseSolverMPI
   *
   * \brief This is the interface for the distributed memory solver
   * with REPLICATED input.
   *
   * This solver class inherits from StrumpackSparseSolver. Unlike
   * StrumpackSparseSolver, the factorization and solve are
   * distributed using MPI. The sparse factors (fill-in) are
   * distributed over the MPI processes, but each MPI process needs a
   * copy of the entire input sparse matrix, the entire right-hand
   * side and the entire solution vector. The matrix reordering and
   * the symbolic factorization are not distributed (but they are
   * threaded).
   *
   * __We recommend not to use this interface, but to use
   * StrumpackSparseSolverMPIDist instead.__
   *
   * \tparam scalar_t can be: float, double, std::complex<float> or
   * std::complex<double>.
   *
   * \tparam integer_t defaults to a regular int. If regular int
   * causes 32 bit integer overflows, you should switch to
   * integer_t=int64_t instead. This should be a __signed__ integer
   * type.
   *
   * \see StrumpackSparseSolverMPIDist, StrumpackSparseSolver
   */
  template<typename scalar_t,typename integer_t=int>
  class StrumpackSparseSolverMPI :
    public StrumpackSparseSolver<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using Tree_t = EliminationTree<scalar_t,integer_t>;
    using Reord_t = MatrixReordering<scalar_t,integer_t>;

  public:
    /**
     * Constructor taking an MPI communicator and command line
     * arguments. This routine is collective on all ranks in the MPI
     * communicator comm.
     *
     * \param mpi_comm MPI communicator. Can be MPI_COMM_WORLD or a
     * subcommunicator. This will internally be duplicated.
     * \param argc The number of arguments, i.e, number of elements in
     * the argv array.
     * \param argv Command line arguments. Add -h or --help to
     * have a description printed.
     * \param verb Flag to suppres/enable output.  Only the root of
     * comm will print output to stdout.
     */
    StrumpackSparseSolverMPI
    (MPI_Comm comm, int argc, char* argv[], bool verbose=true);

    /**
     * Constructor of the StrumpackSparseSolver class. This routine is
     * collective on all ranks in the MPI communicator comm.
     *
     * \param verbose flag to enable/disable output to cout
     * \param root flag to denote whether this process is the root MPI
     * process. Only the root will print certain messages
     * \see set_from_options
     */
    StrumpackSparseSolverMPI(MPI_Comm comm, bool verbose=true);

    /**
     * Destructor, virtual.
     */
    virtual ~StrumpackSparseSolverMPI();

    /**
     * Set a matrix for this sparse solver. This method overwrites the
     * corresponding routine from the base class
     * StrumpackSparseSolver. For this interface, every rank needs to
     * provide a copy of the entire matrix. This routine is collective
     * on the MPI communicator from this solver.
     *
     * \param A input sparse matrix, should be provided on all
     * ranks. The matrix will be copied internally, so it can be
     * safely modified/deleted after calling this function.
     * \see set_csr_matrix
     */
    virtual void set_matrix(const CSRMatrix<scalar_t,integer_t>& A) override;


    /**
     * Set a matrix, in compressed sparse row storage, for this sparse
     * solver. Indices in the CSR storage are 0-based. This method
     * overwrites the corresponding routine from the base class
     * StrumpackSparseSolver. For this interface, every rank needs to
     * provide a copy of the entire matrix. This routine is collective
     * on the MPI communicator from this solver. The matrix will be
     * copied internally, so it can be safely modified/deleted after
     * calling this function.
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
     * \see set_matrix
     */
    virtual void set_csr_matrix
    (integer_t N, const integer_t* row_ptr,
     const integer_t* col_ind, const scalar_t* values,
     bool symmetric_pattern) override;

    /**
     * Return the MPI_Comm object associated with this solver.
     * \return MPI_Comm object for this solver.
     */
    MPI_Comm comm() const;

  protected:
    using StrumpackSparseSolver<scalar_t,integer_t>::is_root_;
    using StrumpackSparseSolver<scalar_t,integer_t>::opts_;
    MPIComm comm_;

    virtual SpMat_t* matrix() override { return mat_.get(); }
    virtual Reord_t* reordering() override { return nd_.get(); }
    virtual Tree_t* tree() override { return tree_mpi_.get(); }
    virtual const SpMat_t* matrix() const override { return mat_.get(); }
    virtual const Reord_t* reordering() const override { return nd_.get(); }
    virtual const Tree_t* tree() const override { return tree_mpi_.get(); }

    virtual void setup_tree() override;
    virtual void setup_reordering() override;
    virtual int compute_reordering
    (const int* p, int base, int nx, int ny, int nz,
     int components, int width) override;
    virtual void separator_reordering() override;
    void perf_counters_stop(const std::string& s) override;
    virtual void synchronize() override { comm_.barrier(); }
    virtual void reduce_flop_counters() const override;

  private:
    std::unique_ptr<CSRMatrix<scalar_t,integer_t>> mat_;
    std::unique_ptr<MatrixReordering<scalar_t,integer_t>> nd_;
    std::unique_ptr<EliminationTreeMPI<scalar_t,integer_t>> tree_mpi_;
  };

} // end namespace strumpack

#endif // STRUMPACK_SPARSE_SOLVER_MPI_HPP
