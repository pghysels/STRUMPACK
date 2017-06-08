/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The Regents of
 * the University of California, through Lawrence Berkeley National Laboratory
 * (subject to receipt of any required approvals from the U.S. Dept. of Energy).
 * All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Technology Transfer Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, prepare derivative works, and perform publicly and display publicly.
 * Beginning five (5) years after the date permission to assert copyright is
 * obtained from the U.S. Department of Energy, and subject to any subsequent five
 * (5) year renewals, the U.S. Government is granted for itself and others acting
 * on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, prepare derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 */
/*! \file StrumpackSparseSolverMPI.hpp
 * \brief Contains the definition of MPI distributed sparse solver class with REPLICATED input.
 */
#ifndef STRUMPACK_SPARSE_SOLVER_MPI_H
#define STRUMPACK_SPARSE_SOLVER_MPI_H

#include "StrumpackSparseSolver.hpp"
#include "EliminationTreeMPI.hpp"
#include "MatrixReorderingMPI.hpp"
#include "mpi.h"

namespace strumpack {

  /*! \brief This is the interface for the distributed memory solver
   *         with REPLICATED input.
   *
   * The factorization and solve are distributed using MPI. The sparse
   * factors (fill-in) are distributed over the MPI processes, but
   * each MPI process needs a copy of the entire input sparse matrix,
   * the entire right-hand side and the entire solution vector. The
   * matrix reordering and the symbolic factorization are not
   * distributed (but they are threaded).
   *
   * The StrumpackSparseSolverMPI class depends on 3 template
   * paramaters <scalar_t,real_t,integer_t>. Supported combinations
   * for scalar_t,real_t are: <float,float>, <double,double>,
   * <std::complex<float>,float> and
   * <std::complex<double>,double>. The integer_t type defaults to a
   * regular int. If regular int causes 32 bit integer overflows, you
   * should switch to integer_t=int64_t instead.
   */
  template<typename scalar_t,typename integer_t>
  class StrumpackSparseSolverMPI : public StrumpackSparseSolver<scalar_t,integer_t> {
  public:
    /*! \brief Constructor taking an MPI communicator and command line
     *         arguments.
     *
     * \param mpi_comm MPI communicator. Can be MPI_COMM_WORLD or a subcommunicator.
     * \param argc  The number of arguments, i.e, number of elements in the argv array.
     * \param argv  Command line arguments. Add -h or --help to have a description printed.
     * \param verb  Flag to suppres/enable output. Only the root of mpi_comm will print output to stdout.
     */
    StrumpackSparseSolverMPI(MPI_Comm mpi_comm, int argc, char* argv[], bool verbose=true);

    StrumpackSparseSolverMPI(MPI_Comm mpi_comm, bool verbose=true);
    /*! \brief Destructor. */
    virtual ~StrumpackSparseSolverMPI();

    virtual void set_matrix(CSRMatrix<scalar_t,integer_t>& A);
    virtual void set_csr_matrix(integer_t N, integer_t* row_ptr, integer_t* col_ind,
				scalar_t* values, bool symmetric_pattern);

  protected:
    MPI_Comm _comm;
    virtual CompressedSparseMatrix<scalar_t,integer_t>* matrix() { return _mat; }
    virtual MatrixReordering<scalar_t,integer_t>* reordering() { return _nd; }
    virtual EliminationTree<scalar_t,integer_t>* elimination_tree() { return _et_mpi; }
    virtual void setup_elimination_tree();
    virtual void setup_matrix_reordering();
    virtual int compute_reordering(int nx, int ny, int nz);
    virtual void compute_separator_reordering();
    void perf_counters_stop(std::string s);
    virtual void synchronize() { MPI_Barrier(_comm); }

  private:
    CSRMatrix<scalar_t,integer_t>* _mat = nullptr;
    MatrixReordering<scalar_t,integer_t>* _nd = nullptr;
    EliminationTreeMPI<scalar_t,integer_t>* _et_mpi = nullptr;
  };

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolverMPI<scalar_t,integer_t>::StrumpackSparseSolverMPI
  (MPI_Comm mpi_comm, bool verbose)
    : StrumpackSparseSolverMPI<scalar_t,integer_t>(mpi_comm, 0, nullptr, verbose) {
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolverMPI<scalar_t,integer_t>::StrumpackSparseSolverMPI
  (MPI_Comm mpi_comm, int argc, char* argv[], bool verbose) :
    StrumpackSparseSolver<scalar_t,integer_t>(argc, argv, verbose, mpi_rank(mpi_comm) == 0) {
    MPI_Comm_dup(mpi_comm, &_comm);
    if (this->_opts.verbose() && this->_is_root)
      std::cout << "# using " << mpi_nprocs(_comm) << " MPI processes" << std::endl;
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolverMPI<scalar_t,integer_t>::~StrumpackSparseSolverMPI() {
    mpi_free_comm(&_comm);
    delete _nd;
    delete _et_mpi;
    delete _mat;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::set_matrix
  (CSRMatrix<scalar_t,integer_t>& A) {
    if (_mat) delete _mat;
    _mat = A.clone();
    this->_factored = this->_reordered = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::set_csr_matrix
  (integer_t N, integer_t* row_ptr, integer_t* col_ind, scalar_t* values, bool symmetric_pattern) {
    if (_mat) delete _mat;
    _mat = new CSRMatrix<scalar_t,integer_t>(N, row_ptr, col_ind, values, symmetric_pattern);
    this->_factored = this->_reordered = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::setup_elimination_tree() {
    if (_et_mpi) delete _et_mpi;
    _et_mpi = new EliminationTreeMPI<scalar_t,integer_t>
      (this->_opts, matrix(), reordering()->sep_tree.get(), _comm);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::setup_matrix_reordering() {
    if (_nd) delete _nd;
    _nd = new MatrixReordering<scalar_t,integer_t>(matrix()->size());
  }

  template<typename scalar_t,typename integer_t> int
  StrumpackSparseSolverMPI<scalar_t,integer_t>::compute_reordering(int nx, int ny, int nz) {
    return _nd->nested_dissection(this->_opts, _mat, _comm, nx, ny, nz);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::compute_separator_reordering() {
    //_nd->separator_reordering(this->_opts, _mat, _et_mpi->root_front());
    _nd->separator_reordering(this->_opts, _mat, this->_opts.verbose() && this->_is_root);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::perf_counters_stop(std::string s) {
    if (this->_opts.verbose()) {
#if defined(HAVE_PAPI)
      float rtime1=0., ptime1=0., mflops=0.;
      long_long flpops1=0;
#pragma omp parallel reduction(+:flpops1) reduction(max:rtime1) reduction(max:ptime1)
      PAPI_flops(&rtime1, &ptime1, &flpops1, &mflops);
      float papi_total_flops = flpops1 - this->_flpops;
      MPI_Allreduce(MPI_IN_PLACE, &papi_total_flops, 1, MPI_FLOAT, MPI_SUM, _comm);
      if (this->is_root)
	std::cout << "# " << s << " PAPI stats:" << std::endl
		  << "#   - total flops = " << papi_total_flops << std::endl
		  << "#   - flop rate = " <<  papi_total_flops/(rtime1-this->_rtime)/1e9 << " GFlops/s" << std::endl
		  << "#   - real time = " << rtime1-this->_rtime << " sec" << std::endl
		  << "#   - processor time = " << ptime1-this->_ptime << " sec" << std::endl;
#endif
#if defined(COUNT_FLOPS)
      long long int flopsbytes[2] = { params::flops - this->_f0, params::bytes - this->_b0 };
      MPI_Allreduce(MPI_IN_PLACE, flopsbytes, 2, MPI_LONG_LONG_INT, MPI_SUM, _comm);
      this->_ftot = flopsbytes[0];
      this->_btot = flopsbytes[1];
      flopsbytes[0] = params::flops - this->_f0;
      MPI_Allreduce(MPI_IN_PLACE, flopsbytes, 2, MPI_LONG_LONG_INT, MPI_MIN, _comm);
      this->_fmin = flopsbytes[0];
      flopsbytes[0] = params::flops - this->_f0;
      MPI_Allreduce(MPI_IN_PLACE, flopsbytes, 2, MPI_LONG_LONG_INT, MPI_MAX, _comm);
      this->_fmax = flopsbytes[0];
#endif
    }
  }

} // end namespace strumpack

#endif
