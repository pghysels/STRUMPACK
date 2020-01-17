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
#ifndef STRUMPACK_SPARSE_SOLVER_MPI_H
#define STRUMPACK_SPARSE_SOLVER_MPI_H

#include "StrumpackSparseSolver.hpp"
#include "sparse/EliminationTreeMPI.hpp"
#include "sparse/MatrixReorderingMPI.hpp"
#include "mpi.h"

namespace strumpack {

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
    virtual ~StrumpackSparseSolverMPI() {}

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
    MPI_Comm comm() const { return comm_.comm(); }

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

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolverMPI<scalar_t,integer_t>::StrumpackSparseSolverMPI
  (MPI_Comm comm, bool verbose)
    : StrumpackSparseSolverMPI<scalar_t,integer_t>
    (comm, 0, nullptr, verbose) {
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolverMPI<scalar_t,integer_t>::StrumpackSparseSolverMPI
  (MPI_Comm comm, int argc, char* argv[], bool verbose) :
    StrumpackSparseSolver<scalar_t,integer_t>
    (argc, argv, verbose, mpi_rank(comm) == 0), comm_(comm) {
    if (opts_.verbose() && is_root_)
      std::cout << "# using " << comm_.size()
                << " MPI processes" << std::endl;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::set_matrix
  (const CSRMatrix<scalar_t,integer_t>& A) {
    mat_ = std::unique_ptr<CSRMatrix<scalar_t,integer_t>>
      (new CSRMatrix<scalar_t,integer_t>(A));
    this->factored_ = this->reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::set_csr_matrix
  (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, bool symmetric_pattern) {
    mat_ = std::unique_ptr<CSRMatrix<scalar_t,integer_t>>
      (new CSRMatrix<scalar_t,integer_t>
       (N, row_ptr, col_ind, values, symmetric_pattern));
    this->factored_ = this->reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::setup_tree() {
    tree_mpi_ = std::unique_ptr<EliminationTreeMPI<scalar_t,integer_t>>
      (new EliminationTreeMPI<scalar_t,integer_t>(opts_, *mat_, *nd_, comm_));
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::setup_reordering() {
    nd_ = std::unique_ptr<MatrixReordering<scalar_t,integer_t>>
      (new MatrixReordering<scalar_t,integer_t>(mat_->size()));
  }

  template<typename scalar_t,typename integer_t> int
  StrumpackSparseSolverMPI<scalar_t,integer_t>::compute_reordering
  (const int* p, int base, int nx, int ny, int nz,
   int components, int width) {
    if (p) return nd_->set_permutation(opts_, *mat_, comm_, p, base);
    return nd_->nested_dissection
      (opts_, *mat_, comm_, nx, ny, nz, components, width);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::separator_reordering() {
    nd_->separator_reordering(opts_, *mat_, tree_mpi_->root());
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::perf_counters_stop
  (const std::string& s) {
    if (opts_.verbose()) {
#if defined(STRUMPACK_USE_PAPI)
      float rtime1=0., ptime1=0., mflops=0.;
      long_long flpops1=0;
#pragma omp parallel reduction(+:flpops1) reduction(max:rtime1) \
  reduction(max:ptime1)
      PAPI_flops(&rtime1, &ptime1, &flpops1, &mflops);
      float papi_total_flops = flpops1 - this->flpops_;
      papi_total_flops = comm_.all_reduce(papi_total_flops, MPI_SUM);

      // TODO memory usage with PAPI

      if (is_root_) {
        std::cout << "# " << s << " PAPI stats:" << std::endl;
        std::cout << "#   - total flops = " << papi_total_flops << std::endl;
        std::cout << "#   - flop rate = "
                  <<  papi_total_flops/(rtime1-this->rtime_)/1e9
                  << " GFlops/s" << std::endl;
        std::cout << "#   - real time = " << rtime1-this->rtime_
                  << " sec" << std::endl;
        std::cout << "#   - processor time = " << ptime1-this->ptime_
                  << " sec" << std::endl;
      }
#endif
#if defined(STRUMPACK_COUNT_FLOPS)
      auto df = params::flops - this->f0_;
      long long int flopsbytes[2] = {df, params::bytes - this->b0_};
      comm_.all_reduce(flopsbytes, 2, MPI_SUM);
      this->ftot_ = flopsbytes[0];
      this->btot_ = flopsbytes[1];
      this->fmin_ = comm_.all_reduce(df, MPI_MIN);
      this->fmax_ = comm_.all_reduce(df, MPI_MAX);
#endif
    }
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::reduce_flop_counters() const {
#if defined(STRUMPACK_COUNT_FLOPS)
    std::array<long long int,19> flops = {
      params::random_flops.load(),
      params::ID_flops.load(),
      params::QR_flops.load(),
      params::ortho_flops.load(),
      params::reduce_sample_flops.load(),
      params::update_sample_flops.load(),
      params::extraction_flops.load(),
      params::CB_sample_flops.load(),
      params::sparse_sample_flops.load(),
      params::ULV_factor_flops.load(),
      params::schur_flops.load(),
      params::full_rank_flops.load(),
      params::f11_fill_flops.load(),
      params::f12_fill_flops.load(),
      params::f21_fill_flops.load(),
      params::f22_fill_flops.load(),
      params::f21_mult_flops.load(),
      params::invf11_mult_flops.load(),
      params::f12_mult_flops.load()
    };
    comm_.reduce(flops.data(), flops.size(), MPI_SUM);
    params::random_flops = flops[0];
    params::ID_flops = flops[1];
    params::QR_flops = flops[2];
    params::ortho_flops = flops[3];
    params::reduce_sample_flops = flops[4];
    params::update_sample_flops = flops[5];
    params::extraction_flops = flops[6];
    params::CB_sample_flops = flops[7];
    params::sparse_sample_flops = flops[8];
    params::ULV_factor_flops = flops[9];
    params::schur_flops = flops[10];
    params::full_rank_flops = flops[11];
    params::f11_fill_flops = flops[12];
    params::f12_fill_flops = flops[13];
    params::f21_fill_flops = flops[14];
    params::f22_fill_flops = flops[15];
    params::f21_mult_flops = flops[16];
    params::invf11_mult_flops = flops[17];
    params::f12_mult_flops = flops[18];
#endif
  }

} // end namespace strumpack

#endif
