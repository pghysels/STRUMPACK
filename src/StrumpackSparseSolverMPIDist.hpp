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
/*! \file StrumpackSparseSolverMPIDist.hpp
 * \brief Contains the definition of fully distributed sparse solver class.
 */
#ifndef STRUMPACK_SPARSE_SOLVER_MPI_DIST_H
#define STRUMPACK_SPARSE_SOLVER_MPI_DIST_H

#include "StrumpackSparseSolverMPI.hpp"
#include "sparse/EliminationTreeMPIDist.hpp"
#include "sparse/MatrixReorderingMPI.hpp"
#include "sparse/GMResMPI.hpp"
#include "sparse/IterativeRefinementMPI.hpp"
#include "sparse/BiCGStabMPI.hpp"
#include "mpi.h"

namespace strumpack {

  /*! \brief This is the fully distributed solver.
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
   * The StrumpackSparseSolverMPIDist class depends on 3 template
   * paramaters <scalar_t,real_t,integer_t>. Supported combinations
   * for scalar_t,real_t are: <float,float>, <double,double>,
   * <std::complex<float>,float> and
   * <std::complex<double>,double>. The integer_t type defaults to a
   * regular int. If regular int causes 32 bit integer overflows, you
   * should switch to integer_t=int64_t instead.
   */
  template<typename scalar_t,typename integer_t>
  class StrumpackSparseSolverMPIDist :
    public StrumpackSparseSolverMPI<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using Tree_t = EliminationTree<scalar_t,integer_t>;
    using Reord_t = MatrixReordering<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;

  public:
    /*! \brief Constructor taking an MPI communicator and command line
     *         arguments.
     *
     * \param mpi_comm   MPI communicator.
     *                   Can be MPI_COMM_WORLD or a subcommunicator.
     * \param argc       The number of arguments, i.e,
     *                   number of elements in the argv array.
     * \param argv       Command line arguments. Add -h or --help to
     *                   have a description printed.
     * \param verb       Flag to suppres/enable output.
     *                   Only the root of mpi_comm will print to stdout.
     */
    StrumpackSparseSolverMPIDist
    (MPI_Comm mpi_comm, int argc, char* argv[], bool verbose=true);
    StrumpackSparseSolverMPIDist(MPI_Comm mpi_comm, bool verbose=true);
    virtual ~StrumpackSparseSolverMPIDist();

    void set_matrix(const CSRMatrix<scalar_t,integer_t>& A);
    // void set_matrix(CSCMatrix<scalar_t,integer_t>& A);
    /*! \brief Associate a (distributed) CSRMatrix with this solver.
     *
     * This matrix will not be modified. An internal copy will be
     * made, so it is safe to delete the data immediately after
     * calling this function. When this is called on a
     * StrumpackSparseSolverMPIDist object, the input matrix will
     * immediately be distributed over all the processes in the
     * communicator associated with the solver. This method is thus
     * collective on the MPI communicator associated with the solver.
     *
     * \param A  A CSRMatrixMPI<scalar_t,integer_t> object.
     */
    virtual void set_matrix
    (const CSRMatrixMPI<scalar_t,integer_t>& A);

    virtual void set_csr_matrix
    (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
     const scalar_t* values, bool symmetric_pattern=false);

    /*! \brief Associate a block-row distributed CSR matrix
     *         with the solver object.
     *
     * \param local_rows   The number of rows of the input matrix assigned
     *                     to this MPI process.  This should equal to
     *                     dist[rank+1]-dist[rank].
     * \param row_ptr      Indices in col_ind and values for the start
     *                     of each row. Nonzeros for row r+dist[rank] are
     *                     in [row_ptr[r],row_ptr[r+1]).
     * \param col_ind      Column indices of each nonzero.
     * \param values       Nonzero values. Should have at least
     *                     (row_ptr[dist[p+1]-dist[p]]-row_ptr[0]) elements.
     * \param dist         Specifies the block-row distribution. A process
     *                     with rank p owns rows [dist[p],dist[p+1]).
     * \param symmetric_pattern
     *                     Denotes whether the sparsity pattern of the
     *                     input matrix is symmetric.
     */
    void set_distributed_csr_matrix
    (integer_t local_rows, const integer_t* row_ptr,
     const integer_t* col_ind, const scalar_t* values,
     const integer_t* dist, bool symmetric_pattern=false);
    /*! \brief Associate a (PETSc) MPIAIJ block-row distributed CSR
        matrix with the solver object. */

    void set_MPIAIJ_matrix
    (integer_t local_rows, const integer_t* d_ptr, const integer_t* d_ind,
     const scalar_t* d_val, const integer_t* o_ptr, const integer_t* o_ind,
     const scalar_t* o_val, const integer_t* garray);

    ReturnCode solve
    (const scalar_t* b, scalar_t* x, bool use_initial_guess=false);
    ReturnCode solve
    (const DenseM_t& b, DenseM_t& x, bool use_initial_guess=false);

  protected:
    virtual SpMat_t* matrix() override { return _mat_mpi.get(); }
    virtual Reord_t* reordering() override { return _nd_mpi.get(); }
    virtual Tree_t* tree() override { return _tree_mpi_dist.get(); }
    virtual const SpMat_t* matrix() const override { return _mat_mpi.get(); }
    virtual const Reord_t* reordering() const override { return _nd_mpi.get(); }
    virtual const Tree_t* tree() const override { return _tree_mpi_dist.get(); }

    virtual void setup_tree() override;
    virtual void setup_reordering() override;
    virtual int compute_reordering(int nx, int ny, int nz) override;
    virtual void compute_separator_reordering() override;

  private:
    std::unique_ptr<CSRMatrixMPI<scalar_t,integer_t>> _mat_mpi;
    std::unique_ptr<MatrixReorderingMPI<scalar_t,integer_t>> _nd_mpi;
    std::unique_ptr<EliminationTreeMPIDist<scalar_t,integer_t>> _tree_mpi_dist;
  };

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::
  StrumpackSparseSolverMPIDist(MPI_Comm mpi_comm, bool verbose) :
    StrumpackSparseSolverMPIDist<scalar_t,integer_t>
    (mpi_comm, 0, nullptr, verbose) {
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::
  StrumpackSparseSolverMPIDist(MPI_Comm mpi_comm, int argc, char* argv[],
                               bool verbose) :
    StrumpackSparseSolverMPI<scalar_t,integer_t>
    (mpi_comm, argc, argv, verbose) {
    // Set the default reordering to PARMETIS?
    //this->_opts.set_reordering_method(ReorderingStrategy::PARMETIS);
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::
  ~StrumpackSparseSolverMPIDist() {
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::set_matrix
  (const CSRMatrix<scalar_t,integer_t>& A) {
    if (_mat_mpi) _mat_mpi.reset();
    _mat_mpi = std::unique_ptr<CSRMatrixMPI<scalar_t,integer_t>>
      (new CSRMatrixMPI<scalar_t,integer_t>(&A, this->_comm, true));
    this->_factored = this->_reordered = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::set_matrix
  (const CSRMatrixMPI<scalar_t,integer_t>& A) {
    if (_mat_mpi) _mat_mpi.reset();
    _mat_mpi = std::unique_ptr<CSRMatrixMPI<scalar_t,integer_t>>
      (new CSRMatrixMPI<scalar_t,integer_t>(A));
    this->_factored = this->_reordered = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::set_csr_matrix
  (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, bool symmetric_pattern) {
    CSRMatrix<scalar_t,integer_t> mat_seq
      (N, row_ptr, col_ind, values, symmetric_pattern);
    if (_mat_mpi) _mat_mpi.reset();
    _mat_mpi = std::unique_ptr<CSRMatrixMPI<scalar_t,integer_t>>
      (new CSRMatrixMPI<scalar_t,integer_t>
       (&mat_seq, this->_comm, true));
    this->_factored = this->_reordered = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::set_distributed_csr_matrix
  (integer_t local_rows, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, const integer_t* dist, bool symmetric_pattern) {
    if (_mat_mpi) _mat_mpi.reset();
    _mat_mpi = std::unique_ptr<CSRMatrixMPI<scalar_t,integer_t>>
      (new CSRMatrixMPI<scalar_t,integer_t>
       (local_rows, row_ptr, col_ind, values, dist,
        this->_comm, symmetric_pattern));
    this->_factored = this->_reordered = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::set_MPIAIJ_matrix
  (integer_t local_rows, const integer_t* d_ptr, const integer_t* d_ind,
   const scalar_t* d_val, const integer_t* o_ptr, const integer_t* o_ind,
   const scalar_t* o_val, const integer_t* garray) {
    if (_mat_mpi) _mat_mpi.reset();
    _mat_mpi = std::unique_ptr<CSRMatrixMPI<scalar_t,integer_t>>
      (new CSRMatrixMPI<scalar_t,integer_t>
       (local_rows, d_ptr, d_ind, d_val, o_ptr, o_ind, o_val,
        garray, this->_comm));
    this->_factored = this->_reordered = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::
  setup_reordering() {
    if (_nd_mpi) _nd_mpi.reset();
    _nd_mpi = std::unique_ptr<MatrixReorderingMPI<scalar_t,integer_t>>
      (new MatrixReorderingMPI<scalar_t,integer_t>
       (matrix()->size(), this->_comm));
  }

  template<typename scalar_t,typename integer_t> int
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::compute_reordering
  (int nx, int ny, int nz) {
    return _nd_mpi->nested_dissection
      (this->_opts, _mat_mpi.get(), nx, ny, nz);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::
  compute_separator_reordering() {
    return _nd_mpi->separator_reordering
      (this->_opts, _mat_mpi.get(), this->_opts.verbose() && this->_is_root);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::setup_tree() {
    if (_tree_mpi_dist) _tree_mpi_dist.reset();
    _tree_mpi_dist =
      std::unique_ptr<EliminationTreeMPIDist<scalar_t,integer_t>>
      (new EliminationTreeMPIDist<scalar_t,integer_t>
       (this->_opts, *_mat_mpi, *_nd_mpi, this->_comm));
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::solve
  (const DenseM_t& b, DenseM_t& x, bool use_initial_guess) {
    if (!this->_factored) {
      ReturnCode ierr = this->factor();
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }
    assert(std::size_t(_mat_mpi->local_rows()) == b.rows());
    assert(b.rows() == x.rows());
    assert(b.cols() == x.cols());
    TaskTimer t("solve");
    this->perf_counters_start();
    t.start();
    auto n_local = x.rows();
    this->_Krylov_its = 0;

    auto bloc = b;
    if (this->_opts.mc64job() == 5)
      bloc.scale_rows(this->_mc64_Dr);

    auto gmres = [&](std::function<void(scalar_t*)> prec) {
      GMResMPI<scalar_t,integer_t>
      (this->_comm, _mat_mpi.get(), prec, n_local, x.data(), bloc.data(),
       this->_opts.rel_tol(), this->_opts.abs_tol(),
       this->_Krylov_its, this->_opts.maxit(),
       this->_opts.gmres_restart(), this->_opts.GramSchmidt_type(),
       use_initial_guess, this->_opts.verbose() && this->_is_root);
    };
    auto bicgstab = [&](std::function<void(scalar_t*)> prec) {
      BiCGStabMPI<scalar_t,integer_t>
      (this->_comm, _mat_mpi.get(), prec, n_local, x.data(), bloc.data(),
       this->_opts.rel_tol(), this->_opts.abs_tol(),
       this->_Krylov_its, this->_opts.maxit(),
       use_initial_guess, this->_opts.verbose() && this->_is_root);
    };
    auto MFsolve = [&](scalar_t* w) {
      //MPI_Pcontrol(1, "multifrontal_solve_dist");
      DenseMW_t X(n_local, x.cols(), w, x.ld());
      tree()->multifrontal_solve_dist(X, _mat_mpi->get_dist());
      //MPI_Pcontrol(-1, "multifrontal_solve_dist");
    };
    auto refine = [&]() {
      IterativeRefinementMPI<scalar_t,integer_t>
      (this->_comm, *_mat_mpi.get(),
       //MFsolve, n_local, x.data(), bloc.data(),
       [&](DenseM_t& w) {
        tree()->multifrontal_solve_dist(w, _mat_mpi->get_dist()); },
       x, bloc,
       this->_opts.rel_tol(), this->_opts.abs_tol(),
       this->_Krylov_its, this->_opts.maxit(),
       use_initial_guess, this->_opts.verbose() && this->_is_root);
    };

    switch (this->_opts.Krylov_solver()) {
    case KrylovSolver::AUTO: {
      if (this->_opts.use_HSS() && x.cols() == 1)
        gmres(MFsolve);
      else refine();
    }; break;
    case KrylovSolver::REFINE: {
      refine();
    }; break;
    case KrylovSolver::GMRES: {
      assert(x.cols() == 1);
      gmres([](scalar_t*){});
    }; break;
    case KrylovSolver::PREC_GMRES: {
      assert(x.cols() == 1);
      gmres(MFsolve);
    }; break;
    case KrylovSolver::BICGSTAB: {
      assert(x.cols() == 1);
      bicgstab([](scalar_t*){});
    }; break;
    case KrylovSolver::PREC_BICGSTAB: {
      assert(x.cols() == 1);
      bicgstab(MFsolve);
    }; break;
    case KrylovSolver::DIRECT: {
      // TODO bloc is already a copy, avoid extra copy?
      x = bloc;
      tree()->multifrontal_solve_dist(x, _mat_mpi->get_dist());
    }; break;
    }

    if (this->_opts.mc64job() != 0)
      // TODO do this in a single routine/comm phase
      for (std::size_t c=0; c<x.cols(); c++)
        permute_vector
          (x.ptr(0,c), this->_mc64_cperm, _mat_mpi->get_dist(), this->_comm);
    if (this->_opts.mc64job() == 5)
      x.scale_rows(this->_mc64_Dc);

    t.stop();
    this->perf_counters_stop("DIRECT/GMRES solve");
    this->print_solve_stats(t);
    return ReturnCode::SUCCESS;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolverMPIDist<scalar_t,integer_t>::solve
  (const scalar_t* b, scalar_t* x, bool use_initial_guess) {
    auto N = _mat_mpi->local_rows();
    auto B = ConstDenseMatrixWrapperPtr(N, 1, b, N);
    DenseMW_t X(N, 1, x, N);
    return solve(*B, X, use_initial_guess);
  }

} // end namespace strumpack

#endif
