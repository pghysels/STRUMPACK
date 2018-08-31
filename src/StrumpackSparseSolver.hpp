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
 * \file StrumpackSparseSolver.hpp
 * \brief Contains the definition of the sequential/multithreaded
 * sparse solver class.
 */
#ifndef STRUMPACK_SOLVER_H
#define STRUMPACK_SOLVER_H

#include <sstream>
#include <getopt.h>
#include <new>
#include <cmath>
#include "StrumpackConfig.hpp"
#if defined(STRUMPACK_USE_TBB_MALLOC)
#include <tbb/scalable_allocator.h>
#endif
#if defined(STRUMPACK_USE_PAPI)
#include <papi.h>
#endif
#include "misc/Tools.hpp"
#include "StrumpackOptions.hpp"
#include "sparse/CompressedSparseMatrix.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/MatrixReordering.hpp"
#include "sparse/EliminationTree.hpp"
#include "sparse/GMRes.hpp"
#include "sparse/BiCGStab.hpp"
#include "sparse/IterativeRefinement.hpp"

#if defined(STRUMPACK_USE_TBB_MALLOC)
void* operator new(std::size_t sz) throw(std::bad_alloc) {
  return scalable_malloc(sz);
}
void operator delete(void* ptr) throw() {
  if (!ptr) return;
  scalable_free(ptr);
}
#endif

/**
 * All of STRUMPACK is contained in the strumpack namespace.
 */
namespace strumpack {

  /**
   * \class StrumpackSparseSolver
   *
   * \brief StrumpackSparseSolver is the main sequential or
   * multithreaded sparse solver class.
   *
   * This is the main interface to STRUMPACK's sparse solver. Use this
   * for a sequential or multithreaded sparse solver. For the fully
   * distributed solver, see StrumpackSparseSolverMPIDist.
   *
   * \tparam scalar_t can be: float, double, std::complex<float> or
   * std::complex<double>.
   *
   * \tparam integer_t defaults to a regular int. If regular int
   * causes 32 bit integer overflows, you should switch to
   * integer_t=int64_t instead. This should be a __signed__ integer
   * type.
   *
   * \see StrumpackSparseSolverMPIDist, StrumpackSparseSolverMPI
   */
  template<typename scalar_t,typename integer_t=int>
  class StrumpackSparseSolver {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using Tree_t = EliminationTree<scalar_t,integer_t>;
    using Reord_t = MatrixReordering<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;

  public:

    /**
     * Constructor of the StrumpackSparseSolver class, taking command
     * line arguments.
     *
     * \param argc number of arguments, i.e, number of elements in
     * the argv array
     * \param argv command line arguments. Add -h or --help to have a
     * description printed
     * \param verbose flag to enable/disable output to cout
     * \param root flag to denote whether this process is the root MPI
     * process, only the root will print certain messages to cout
     */
    StrumpackSparseSolver
    (int argc, char* argv[], bool verbose=true, bool root=true);

    /**
     * Constructor of the StrumpackSparseSolver class.
     *
     * \param verbose flag to enable/disable output to cout
     * \param root flag to denote whether this process is the root MPI
     * process. Only the root will print certain messages
     */
    StrumpackSparseSolver(bool verbose=true, bool root=true);

    /**
     * (Virtual) destructor of the StrumpackSparseSolver class.
     */
    virtual ~StrumpackSparseSolver();

    /**
     * Associate a (sequential) CSRMatrix with this solver.
     *
     * This matrix will not be modified. An internal copy will be
     * made, so it is safe to delete the data immediately after
     * calling this function. When this is called on a
     * StrumpackSparseSolverMPIDist object, the input matrix will
     * immediately be distributed over all the processes in the
     * communicator associated with the solver. This method is thus
     * collective on the MPI communicator associated with the solver.
     *
     * \param A A CSRMatrix<scalar_t,integer_t> object, will
     * internally be duplicated
     */
    virtual void set_matrix(const CSRMatrix<scalar_t,integer_t>& A);

    /**
     * Associate a (sequential) NxN CSR matrix with this solver.
     *
     * This matrix will not be modified. An internal copy will be
     * made, so it is safe to delete the data immediately after
     * calling this function. When this is called on a
     * StrumpackSparseSolverMPIDist object, the input matrix will
     * immediately be distributed over all the processes in the
     * communicator associated with the solver. This method is thus
     * collective on the MPI communicator associated with the solver.
     * See the manual for a description of the CSR format. You can
     * also use the CSRMatrix class.
     *
     * \param N number of rows and columns of the CSR input matrix.
     * \param row_ptr indices in col_ind and values for the start of
     * each row. Nonzeros for row r are in [row_ptr[r],row_ptr[r+1])
     * \param col_ind column indices of each nonzero
     * \param values nonzero values
     * \param symmetric_pattern denotes whether the sparsity
     * __pattern__ of the input matrix is symmetric, does not require
     * the matrix __values__ to be symmetric
     */
    virtual void set_csr_matrix
    (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
     const scalar_t* values, bool symmetric_pattern=false);

    /**
     * Compute matrix reorderings for numerical stability and to
     * reduce fill-in.
     *
     * Start computation of the matrix reorderings. See the relevant
     * options to control the matrix reordering in the manual. A first
     * reordering is the MC64 column permutation for numerical
     * stability. This can be disabled if the matrix has large nonzero
     * diagonal entries. MC64 optionally also performs row and column
     * scaling. Next, a fill-reducing reordering is computed. This is
     * done with the nested dissection algortihms of either
     * (PT-)Scotch, (Par)Metis or a simple geometric nested dissection
     * code which only works on regular meshes.
     *
     * \param nx this (optional) parameter is only meaningful when the
     * matrix corresponds to a stencil on a regular mesh. The stencil
     * is assumed to be at most 3 points wide in each dimension and
     * only contain a single degree of freedom per grid point. The nx
     * parameter denotes the number of grid points in the first
     * spatial dimension.
     * \param ny see parameters nx. Parameter ny denotes the number of
     * gridpoints in the second spatial dimension.
     * This should only be set if the mesh is 2 or 3 dimensional.
     * \param nz See parameters nx. Parameter nz denotes the number of
     * gridpoints in the third spatial dimension.
     * This should only be set if the mesh is 3 dimensional.
     * \return error code
     * \see SPOptions
     */
    virtual ReturnCode reorder
    (int nx=1, int ny=1, int nz=1, int components=1, int width=1);

    /**
     * Perform numerical factorization of the sparse input matrix.
     *
     * This is the computationally expensive part of the
     * code. However, once the factorization is performed, it can be
     * reused for multiple solves. This requires that a valid matrix
     * has been assigned to this solver. internally this check whether
     * the matrix has been reordered (with a call to reorder), and if
     * not, it will call reorder.
     *
     * \return error code
     * \see set_matrix, set_csr_matrix
     */
    ReturnCode factor();

    /**
     * Solve a linear system with a single right-hand side.
     *
     * \param b input, will not be modified. Pointer to the right-hand
     * side. Array should be lenght N, the dimension of the input
     * matrix for StrumpackSparseSolver and
     * StrumpackSparseSolverMPI. For StrumpackSparseSolverMPIDist, the
     * length of b should be correspond the partitioning of the
     * block-row distributed input matrix.
     * \param x Output, pointer to the solution vector.  Array should
     * be lenght N, the dimension of the input matrix for
     * StrumpackSparseSolver and StrumpackSparseSolverMPI. For
     * StrumpackSparseSolverMPIDist, the length of b should be
     * correspond the partitioning of the block-row distributed input
     * matrix.
     * \param use_initial_guess set to true if x contains an intial
     * guess to the solution. This is mainly useful when using an
     * iterative solver. If set to false, x should not be set (but
     * should be allocated).
     * \return error code
     */
    virtual ReturnCode solve
    (const scalar_t* b, scalar_t* x, bool use_initial_guess=false);

    /**
     * Solve a linear system with a single or multiple right-hand
     * sides.
     *
     * \param b input, will not be modified. DenseMatrix containgin
     * the right-hand side vector/matrix. Should have N rows, with N
     * the dimension of the input matrix for StrumpackSparseSolver and
     * StrumpackSparseSolverMPI. For StrumpackSparseSolverMPIDist, the
     * number or rows of b should be correspond to the partitioning of
     * the block-row distributed input matrix.
     * \param x Output, pointer to the solution vector.  Array should
     * be lenght N, the dimension of the input matrix for
     * StrumpackSparseSolver and StrumpackSparseSolverMPI. For
     * StrumpackSparseSolverMPIDist, the length of b should be
     * correspond the partitioning of the block-row distributed input
     * matrix.
     * \param use_initial_guess set to true if x contains an intial
     * guess to the solution.  This is mainly useful when using an
     * iterative solver.  If set to false, x should not be set (but
     * should be allocated).
     * \return error code
     * \see DenseMatrix, solve()
     */
    virtual ReturnCode solve
    (const DenseM_t& b, DenseM_t& x, bool use_initial_guess=false);

    /**
     * Return the object holding the options for this sparse solver.
     */
    SPOptions<scalar_t>& options() { return opts_; }

    /**
     * Return the object holding the options for this sparse solver.
     */
    const SPOptions<scalar_t>& options() const { return opts_; }

    /**
     * Parse the command line options passed in the constructor, and
     * modify the options object accordingly. Run with option -h or
     * --help to see a list of supported options. Or check the
     * SPOptions documentation.
     */
    void set_from_options() { opts_.set_from_command_line(); }

    /**
     * Parse the command line options, and modify the options object
     * accordingly. Run with option -h or --help to see a list of
     * supported options. Or check the SPOptions documentation.
     *
     * \param argc number of options in argv
     * \param argv list of options
     */
    void set_from_options(int argc, char* argv[])
    { opts_.set_from_command_line(argc, argv); }

    /**
     * Return the maximum rank encountered in any of the HSS matrices
     * used to compress the sparse triangular factors. This should be
     * called after the factorization phase. For the
     * StrumpackSparseSolverMPI and StrumpackSparseSolverMPIDist
     * distributed memory solvers, this routine is collective on the
     * MPI communicator.
     */
    int maximum_rank() const { return tree()->maximum_rank(); }

    /**
     * Return the number of nonzeros in the (sparse) factors. This is
     * known as the fill-in. This should be called after computing the
     * numerical factorization. For the StrumpackSparseSolverMPI and
     * StrumpackSparseSolverMPIDist distributed memory solvers, this
     * routine is collective on the MPI communicator.
     */
    std::size_t factor_nonzeros() const { return tree()->factor_nonzeros(); }

    /**
     * Return the amount of memory taken by the sparse factorization
     * factors. This is the fill-in. It is simply computed as
     * factor_nonzeros() * sizeof(scalar_t), so it does not include
     * any overhead from the metadata for the datastructures. This
     * should be called after the factorization. For the
     * StrumpackSparseSolverMPI and StrumpackSparseSolverMPIDist
     * distributed memory solvers, this routine is collective on the
     * MPI communicator.
     */
    std::size_t factor_memory() const
    { return tree()->factor_nonzeros() * sizeof(scalar_t); }

    /**
     * Return the number of iterations performed by the outer (Krylov)
     * iterative solver. Call this after calling the solve routine.
     */
    int Krylov_iterations() const { return Krylov_its_; }

    /**
     * Create a gnuplot script to draw/plot the sparse factors. Only
     * do this for small matrices! It is very slow!
     *
     * \param name filename of the generated gnuplot script. Running
     * \verbatim gnuplot plotname.gnuplot \endverbatim will generate a
     * pdf file.
     */
    void draw(const std::string& name) const { tree()->draw(*matrix(), name); }

  protected:
    virtual void setup_tree();
    virtual void setup_reordering();
    virtual int compute_reordering
    (int nx, int ny, int nz, int components, int width);
    virtual void compute_separator_reordering();

    virtual SpMat_t* matrix() { return mat_.get(); }
    virtual Reord_t* reordering() { return nd_.get(); }
    virtual Tree_t* tree() { return tree_.get(); }
    virtual const SpMat_t* matrix() const { return mat_.get(); }
    virtual const Reord_t* reordering() const { return nd_.get(); }
    virtual const Tree_t* tree() const { return tree_.get(); }

    void papi_initialize();
    inline long long dense_factor_nonzeros() const {
      return tree()->dense_factor_nonzeros();
    }
    inline long long dense_factor_memory() const {
      return tree()->dense_factor_nonzeros() * sizeof(scalar_t);
    }
    void print_solve_stats(TaskTimer& t) const;
    virtual void perf_counters_start();
    virtual void perf_counters_stop(const std::string& s);
    virtual void synchronize() {}
    virtual void communicate_ordering() {}
    virtual void flop_breakdown() const;
    virtual void print_flop_breakdown
    (float random_flops, float ID_flops, float QR_flops, float ortho_flops,
     float reduce_sample_flops, float update_sample_flops,
     float extraction_flops, float CB_sample_flops, float sparse_sample_flops,
     float ULV_factor_flops, float schur_flops, float full_rank_flops) const;
    virtual void flop_breakdown_reset() const;

    SPOptions<scalar_t> opts_;
    bool is_root_;
    std::vector<integer_t> matching_cperm_;
    std::vector<scalar_t> matching_Dr_; // row scaling
    std::vector<scalar_t> matching_Dc_; // column scaling
    std::new_handler old_handler_;
    std::ostream* rank_out_ = nullptr;
    bool factored_ = false;
    bool reordered_ = false;
    int Krylov_its_ = 0;

#if defined(STRUMPACK_USE_PAPI)
    float rtime_ = 0., ptime_ = 0.;
    long_long _flpops = 0;
#endif
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f0_ = 0, ftot_ = 0, fmin_ = 0, fmax_ = 0;
    long long int b0_ = 0, btot_ = 0, bmin_ = 0, bmax_ = 0;
#endif
  private:
    std::unique_ptr<CSRMatrix<scalar_t,integer_t>> mat_;
    std::unique_ptr<MatrixReordering<scalar_t,integer_t>> nd_;
    std::unique_ptr<EliminationTree<scalar_t,integer_t>> tree_;
  };

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolver<scalar_t,integer_t>::StrumpackSparseSolver
  (bool verbose, bool root)
    : StrumpackSparseSolver<scalar_t,integer_t>(0, nullptr, verbose, root) {
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolver<scalar_t,integer_t>::StrumpackSparseSolver
  (int argc, char* argv[], bool verbose, bool root)
    : opts_(argc, argv), is_root_(root) {
    opts_.set_verbose(verbose);
    old_handler_ = std::set_new_handler
      ([]{ std::cerr << "STRUMPACK: out of memory!" << std::endl; abort(); });
    papi_initialize();
    if (opts_.verbose() && is_root_) {
      std::cout << "# Initializing STRUMPACK" << std::endl;
#if defined(_OPENMP)
      if (params::num_threads == 1)
        std::cout << "# using " << params::num_threads
                  << " OpenMP thread" << std::endl;
      else
        std::cout << "# using " << params::num_threads
                  << " OpenMP threads" << std::endl;
#else
      std::cout << "# running serially, no OpenMP support!" << std::endl;
#endif
      // a heuristic to set the recursion task cutoff level based on
      // the number of threads
      if (params::num_threads == 1) params::task_recursion_cutoff_level = 0;
      else {
        params::task_recursion_cutoff_level =
          std::log2(params::num_threads) + 3;
        std::cout << "# number of tasking levels = "
                  << params::task_recursion_cutoff_level
                  << " = log_2(#threads) + 3"<< std::endl;
      }
    }
#if defined(STRUMPACK_COUNT_FLOPS)
    if (!params::flops.is_lock_free())
      std::cerr << "# WARNING: the flop counter is not lock free"
                << std::endl;
#endif
    opts_.HSS_options().set_synchronized_compression(true);
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolver<scalar_t,integer_t>::~StrumpackSparseSolver() {
    std::set_new_handler(old_handler_);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::setup_tree() {
    tree_ = std::unique_ptr<EliminationTree<scalar_t,integer_t>>
      (new EliminationTree<scalar_t,integer_t>(opts_, *mat_, nd_->tree()));
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::setup_reordering() {
    nd_ = std::unique_ptr<MatrixReordering<scalar_t,integer_t>>
      (new MatrixReordering<scalar_t,integer_t>(matrix()->size()));
  }

  template<typename scalar_t,typename integer_t> int
  StrumpackSparseSolver<scalar_t,integer_t>::compute_reordering
  (int nx, int ny, int nz, int components, int width) {
    return nd_->nested_dissection
      (opts_, *mat_, nx, ny, nz, components, width);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::compute_separator_reordering() {
    nd_->separator_reordering
      (opts_, *mat_, opts_.verbose() && is_root_);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::papi_initialize() {
#if defined(STRUMPACK_USE_PAPI)
    // TODO call PAPI_library_init???
    float mflops = 0.;
    int retval = PAPI_flops(&rtime_, &ptime_, &_flpops, &mflops);
    if (retval != PAPI_OK) {
      std::cerr << "# WARNING: problem starting PAPI performance counters:"
                << std::endl;
      switch (retval) {
      case PAPI_EINVAL:
        std::cerr << "#   - the counters were already started by"
          << " something other than: PAPI_flips() or PAPI_flops()."
          << std::endl; break;
      case PAPI_ENOEVNT:
        std::cerr << "#   - the floating point operations, floating point"
                  << " instructions or total cycles event does not exist."
                  << std::endl; break;
      case PAPI_ENOMEM:
        std::cerr << "#   - insufficient memory to complete the operation."
                  << std::endl; break;
      default:
        std::cerr << "#   - some other error: " << retval << std::endl;
      }
    }
#endif
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::perf_counters_start() {
#if defined(STRUMPACK_USE_PAPI)
    float mflops = 0., rtime = 0., ptime = 0.;
    long_long flpops = 0; // cannot use class variables in openmp clause
#pragma omp parallel reduction(+:flpops) reduction(max:rtime) \
  reduction(max:ptime)
    PAPI_flops(&rtime, &ptime, &flpops, &mflops);
    _flpops = flpops; rtime_ = rtime; ptime_ = ptime;
#endif
#if defined(STRUMPACK_COUNT_FLOPS)
    f0_ = params::flops;
    b0_ = params::bytes;
#endif
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::perf_counters_stop
  (const std::string& s) {
#if defined(STRUMPACK_USE_PAPI)
    float mflops = 0., rtime = 0., ptime = 0.;
    long_long flpops = 0;
#pragma omp parallel reduction(+:flpops) reduction(max:rtime)  \
  reduction(max:ptime)
    PAPI_flops(&rtime, &ptime, &flpops, &mflops);
    PAPI_dmem_info_t dmem;
    PAPI_get_dmem_info(&dmem);
    if (opts_.verbose() && is_root_) {
      std::cout << "# " << s << " PAPI stats:" << std::endl;
      std::cout << "#   - total flops = "
                << double(flpops-_flpops) << std::endl;
      std::cout << "#   - flop rate = "
                << double(flpops-_flpops)/(rtime-rtime_)/1e9
                << " GFlops/s" << std::endl;
      std::cout << "#   - real time = " << rtime-rtime_
                << " sec" << std::endl;
      std::cout << "#   - processor time = " << ptime-ptime_
                << " sec" << std::endl;
      std::cout << "# mem size:\t\t" << dmem.size << std::endl;
      std::cout << "# mem resident:\t\t" << dmem.resident << std::endl;
      std::cout << "# mem high water mark:\t" << dmem.high_water_mark
                << std::endl;
      std::cout << "# mem shared:\t\t" << dmem.shared << std::endl;
      std::cout << "# mem text:\t\t" << dmem.text << std::endl;
      std::cout << "# mem library:\t\t" << dmem.library << std::endl;
      std::cout << "# mem heap:\t\t" << dmem.heap << std::endl;
      std::cout << "# mem locked:\t\t" << dmem.locked << std::endl;
      std::cout << "# mem stack:\t\t" << dmem.stack << std::endl;
      std::cout << "# mem pagesize:\t\t" << dmem.pagesize << std::endl;
    }
#endif
#if defined(STRUMPACK_COUNT_FLOPS)
    fmin_ = fmax_ = ftot_ = params::flops - f0_;
    bmin_ = bmax_ = btot_ = params::bytes - b0_;
#endif
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::print_solve_stats
  (TaskTimer& t) const {
    double tel = t.elapsed();
    if (opts_.verbose() && is_root_) {
      std::cout << "# DIRECT/GMRES solve:" << std::endl;
      std::cout << "#   - abs_tol = " << opts_.abs_tol()
                << ", rel_tol = " << opts_.rel_tol()
                << ", restart = " << opts_.gmres_restart()
                << ", maxit = " << opts_.maxit() << std::endl;
      std::cout << "#   - number of Krylov iterations = "
                << Krylov_its_ << std::endl;
      std::cout << "#   - solve time = " << tel << std::endl;
#if defined(STRUMPACK_COUNT_FLOPS)
      std::cout << "#   - total flops = " << double(ftot_) << ", min = "
                << double(fmin_) << ", max = " << double(fmax_) << std::endl;
      std::cout << "#   - flop rate = " << ftot_ / tel / 1e9 << " GFlop/s"
                << std::endl;
      std::cout << "#   - bytes moved = " << double(btot_) / 1e6
                << " MB, min = "<< double(bmin_) / 1e6
                << " MB, max = " << double(bmax_) / 1e6 << " MB" << std::endl;
      std::cout << "#   - byte rate = " << btot_ / tel / 1e9 << " GByte/s"
                << std::endl;
      std::cout << "#   - solve arithmetic intensity = "
                << double(ftot_) / btot_
                << " flop/byte" << std::endl;
#endif
    }
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::set_matrix
  (const CSRMatrix<scalar_t,integer_t>& A) {
    mat_ = std::unique_ptr<CSRMatrix<scalar_t,integer_t>>
      (new CSRMatrix<scalar_t,integer_t>(A));
    factored_ = reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::set_csr_matrix
  (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, bool symmetric_pattern) {
    mat_ = std::unique_ptr<CSRMatrix<scalar_t,integer_t>>
      (new CSRMatrix<scalar_t,integer_t>
       (N, row_ptr, col_ind, values, symmetric_pattern));
    factored_ = reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolver<scalar_t,integer_t>::reorder
  (int nx, int ny, int nz, int components, int width) {
    if (!matrix()) return ReturnCode::MATRIX_NOT_SET;
    TaskTimer t1("permute-scale");
    int ierr;
    if (opts_.matching() != MatchingJob::NONE) {
      if (opts_.verbose() && is_root_)
        std::cout << "# matching job: "
                  << get_description(opts_.matching())
                  << std::endl;
      t1.time([&](){
          ierr = matrix()->permute_and_scale
            (opts_.matching(), matching_cperm_, matching_Dr_, matching_Dc_);
        });
      if (ierr) {
        std::cerr << "ERROR: matching failed" << std::endl;
        return ReturnCode::REORDERING_ERROR;
      }
    }
    auto old_nnz = matrix()->nnz();
    TaskTimer t2("sparsity-symmetrization",
                 [&](){ matrix()->symmetrize_sparsity(); });
    if (matrix()->nnz() != old_nnz && opts_.verbose() && is_root_) {
      std::cout << "# Matrix padded with zeros to get symmetric pattern."
                << std::endl;
      std::cout << "# Number of nonzeros increased from "
                << number_format_with_commas(old_nnz) << " to "
                << number_format_with_commas(matrix()->nnz()) << "."
                << std::endl;
    }

    TaskTimer t3("nested-dissection");
    perf_counters_start();
    t3.start();
    setup_reordering();
    ierr = compute_reordering(nx, ny, nz, components, width);
    if (ierr) {
      std::cerr << "ERROR: nested dissection went wrong, ierr="
                << ierr << std::endl;
      return ReturnCode::REORDERING_ERROR;
    }
    matrix()->permute(reordering()->iperm(), reordering()->perm());
    t3.stop();
    if (opts_.verbose() && is_root_) {
      std::cout << "#   - nd time = " << t3.elapsed() << std::endl;
      if (opts_.matching() != MatchingJob::NONE)
        std::cout << "#   - matching time = " << t1.elapsed() << std::endl;
      std::cout << "#   - symmetrization time = " << t2.elapsed()
                << std::endl;
    }
    perf_counters_stop("nested dissection");

    if (opts_.use_HSS() || opts_.use_BLR()) {
      perf_counters_start();
      TaskTimer t4("separator-reordering", [&](){
          compute_separator_reordering();
          // TODO also broadcast this?? is computed with scotch
        });
      if (opts_.verbose() && is_root_)
        std::cout << "#   - sep-reorder time = " << t4.elapsed() << std::endl;
      perf_counters_stop("separator reordering");
    }

    perf_counters_start();
    TaskTimer t0("symbolic-factorization", [&](){ setup_tree(); });
    reordering()->clear_tree_data();
    if (opts_.verbose()) {
      // this might require a reduction
      auto nr_dense = tree()->nr_dense_fronts();
      auto nr_HSS = tree()->nr_HSS_fronts();
      auto nr_BLR = tree()->nr_BLR_fronts();
      if (is_root_) {
        std::cout << "# symbolic factorization:" << std::endl;
        std::cout << "#   - nr of dense Frontal matrices = "
                  << number_format_with_commas(nr_dense) << std::endl;
        std::cout << "#   - nr of HSS Frontal matrices = "
                  << number_format_with_commas(nr_HSS) << std::endl;
        std::cout << "#   - nr of BLR Frontal matrices = "
                  << number_format_with_commas(nr_BLR) << std::endl;
        std::cout << "#   - symb-factor time = " << t0.elapsed() << std::endl;
      }
    }
    perf_counters_stop("symbolic factorization");

    // if (opts_.use_HSS()) {
    //   perf_counters_start();
    //   TaskTimer t4("separator-reordering", [&](){
    //            compute_separator_reordering();
    //            // TODO also broadcast this?? is computed with scotch
    //          });
    // if (opts_.verbose() && is_root_)
    //   std::cout << "#   - sep-reorder time = "
    //             << t4.elapsed() << std::endl;
    //   perf_counters_stop("separator reordering");
    // }

    reordered_ = true;
    return ReturnCode::SUCCESS;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::flop_breakdown() const {
#if defined(STRUMPACK_COUNT_FLOPS)
    print_flop_breakdown
      (params::random_flops, params::ID_flops, params::QR_flops,
       params::ortho_flops, params::reduce_sample_flops,
       params::update_sample_flops, params::extraction_flops,
       params::CB_sample_flops, params::sparse_sample_flops,
       params::ULV_factor_flops, params::schur_flops,
       params::full_rank_flops);
#endif
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::flop_breakdown_reset() const {
#if defined(STRUMPACK_COUNT_FLOPS)
    params::random_flops = 0;
    params::ID_flops = 0;
    params::QR_flops = 0;
    params::ortho_flops = 0;
    params::reduce_sample_flops = 0;
    params::update_sample_flops = 0;
    params::extraction_flops = 0;
    params::CB_sample_flops = 0;
    params::sparse_sample_flops = 0;
    params::ULV_factor_flops = 0;
    params::schur_flops = 0;
    params::full_rank_flops = 0;
#endif
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::print_flop_breakdown
  (float random_flops, float ID_flops, float QR_flops, float ortho_flops,
   float reduce_sample_flops, float update_sample_flops,
   float extraction_flops, float CB_sample_flops, float sparse_sample_flops,
   float ULV_factor_flops, float schur_flops, float full_rank_flops) const {
    if (!is_root_) return;
    float sample_flops = CB_sample_flops
      + sparse_sample_flops;
    float compression_flops = random_flops
      + ID_flops + QR_flops + ortho_flops
      + reduce_sample_flops + update_sample_flops
      + extraction_flops + sample_flops;
    std::cout << std::endl;
    std::cout << "# ----- FLOP BREAKDOWN ---------------------"
              << std::endl;
    std::cout << "# compression           = "
              << compression_flops << std::endl;
    std::cout << "#    random             = "
              << random_flops << std::endl;
    std::cout << "#    ID                 = "
              << ID_flops << std::endl;
    std::cout << "#    QR                 = "
              << QR_flops << std::endl;
    std::cout << "#    ortho              = "
              << ortho_flops << std::endl;
    std::cout << "#    reduce_samples     = "
              << reduce_sample_flops << std::endl;
    std::cout << "#    update_samples     = "
              << update_sample_flops << std::endl;
    std::cout << "#    extraction         = "
              << extraction_flops << std::endl;
    std::cout << "#    sampling           = "
              << sample_flops << std::endl;
    std::cout << "#       CB_sample       = "
              << CB_sample_flops << std::endl;
    std::cout << "#       sparse_sampling = "
              << sparse_sample_flops << std::endl;
    std::cout << "# ULV_factor            = "
              << ULV_factor_flops << std::endl;
    std::cout << "# Schur                 = "
              << schur_flops << std::endl;
    std::cout << "# full_rank             = "
              << full_rank_flops << std::endl;
    std::cout << "# --------------------------------------------"
              << std::endl;
    std::cout << "# total                 = "
              << (compression_flops + ULV_factor_flops +
                  schur_flops + full_rank_flops) << std::endl;
    std::cout << "# --------------------------------------------"
              << std::endl;
    std::cout << std::endl;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolver<scalar_t,integer_t>::factor() {
    if (!matrix()) return ReturnCode::MATRIX_NOT_SET;
    if (factored_) return ReturnCode::SUCCESS;
    if (!reordered_) {
      ReturnCode ierr = reorder();
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }
    float dfnnz = 0.;
    if (opts_.verbose()) {
      dfnnz = dense_factor_memory();
      if (is_root_) {
        std::cout << "# multifrontal factorization:" << std::endl;
        std::cout << "#   - estimated memory usage (exact solver) = "
                  << dfnnz / 1e6 << " MB" << std::endl;
      }
    }
    perf_counters_start();
    flop_breakdown_reset();
    TaskTimer t1("factorization", [&]() {
        tree()->multifrontal_factorization(*matrix(), opts_);
      });
    perf_counters_stop("numerical factorization");
    if (opts_.verbose()) {
      auto fnnz = factor_nonzeros();
      auto max_rank = maximum_rank();
      if (is_root_) {
        std::cout << "#   - factor time = " << t1.elapsed() << std::endl;
        std::cout << "#   - factor nonzeros = "
                  << number_format_with_commas(fnnz) << std::endl;
        std::cout << "#   - factor memory = "
                  << fnnz * sizeof(scalar_t) / 1e6 << " MB" << std::endl;
#if defined(STRUMPACK_COUNT_FLOPS)
        std::cout << "#   - total flops = " << double(ftot_) << ", min = "
                  << double(fmin_) << ", max = " << double(fmax_)
                  << std::endl;
        std::cout << "#   - flop rate = " << ftot_ / t1.elapsed() / 1e9
                  << " GFlop/s" << std::endl;
#endif
        std::cout << "#   - factor memory/nonzeros = "
                  << float(fnnz * sizeof(scalar_t)) / dfnnz * 100.0
                  << " % of multifrontal" << std::endl;
        std::cout << "#   - maximum HSS rank = " << max_rank << std::endl;
        std::cout << "#   - HSS compression = " << std::boolalpha
                  << opts_.use_HSS() << std::endl;
        if (opts_.use_HSS()) {
          std::cout << "#   - relative compression tolerance = "
                    << opts_.HSS_options().rel_tol() << std::endl;
          std::cout << "#   - absolute compression tolerance = "
                    << opts_.HSS_options().abs_tol() << std::endl;
          std::cout << "#   - "
                    << get_name(opts_.HSS_options().random_distribution())
                    << " distribution with "
                    << get_name(opts_.HSS_options().random_engine())
                    << " engine" << std::endl;
        }
        std::cout << "#   - BLR compression = " << std::boolalpha
                  << opts_.use_BLR() << std::endl;
        if (opts_.use_BLR()) {
          std::cout << "#   - relative compression tolerance = "
                    << opts_.BLR_options().rel_tol() << std::endl;
          std::cout << "#   - absolute compression tolerance = "
                    << opts_.BLR_options().abs_tol() << std::endl;
        }

      }
      if (opts_.use_HSS())
        flop_breakdown();
    }
    if (rank_out_) tree()->print_rank_statistics(*rank_out_);
    factored_ = true;
    return ReturnCode::SUCCESS;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolver<scalar_t,integer_t>::solve
  (const scalar_t* b, scalar_t* x, bool use_initial_guess) {
    auto N = matrix()->size();
    auto B = ConstDenseMatrixWrapperPtr(N, 1, b, N);
    DenseMW_t X(N, 1, x, N);
    return solve(*B, X, use_initial_guess);
  }

  // TODO make this const
  //  Krylov its and flops, bytes, time are modified!!
  // pass those as a pointer to a struct ??
  // this can also call factor if not already factored!!
  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolver<scalar_t,integer_t>::solve
  (const DenseM_t& b, DenseM_t& x, bool use_initial_guess) {
    if (!this->factored_ &&
        opts_.Krylov_solver() != KrylovSolver::GMRES &&
        opts_.Krylov_solver() != KrylovSolver::BICGSTAB) {
      ReturnCode ierr = factor();
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }
    TaskTimer t("solve");
    perf_counters_start();
    t.start();

    integer_t N = matrix()->size();
    assert(N < std::numeric_limits<int>::max());
    std::vector<int> intIP(N);
    for (integer_t i=0; i<N; i++)
      intIP[i] = reordering()->iperm()[i] + 1;
    std::vector<int> int_matching_cperm;
    if (opts_.matching() != MatchingJob::NONE) {
      int_matching_cperm.resize(N);
      for (integer_t i=0; i<N; i++)
        int_matching_cperm[i] = matching_cperm_[i] + 1;
    }

    auto bloc = b;
    if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
      bloc.scale_rows(matching_Dr_);
    bloc.lapmr(intIP, true);

    if (use_initial_guess &&
        opts_.Krylov_solver() != KrylovSolver::DIRECT) {
      if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
        x.div_rows(matching_Dc_);
      if (opts_.matching() != MatchingJob::NONE)
        x.lapmr(int_matching_cperm, true);
      x.lapmr(intIP, true);
    }

    Krylov_its_ = 0;

    auto spmv = [&](const scalar_t* x, scalar_t* y) {
      matrix()->spmv(x, y);
    };

    auto gmres_solve = [&](const std::function<void(scalar_t*)>& prec) {
      GMRes<scalar_t>
      (spmv, prec, x.rows(), x.data(), bloc.data(),
       opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
       opts_.gmres_restart(), opts_.GramSchmidt_type(),
       use_initial_guess, opts_.verbose() && is_root_);
    };
    auto bicgstab_solve = [&](const std::function<void(scalar_t*)>& prec) {
      BiCGStab<scalar_t>
      (spmv, prec, x.rows(), x.data(), bloc.data(),
       opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
       use_initial_guess, opts_.verbose() && is_root_);
    };
    auto MFsolve = [&](scalar_t* w) {
      DenseMW_t X(x.rows(), 1, w, x.ld());
      tree()->multifrontal_solve(X);
    };
    auto refine = [&]() {
      IterativeRefinement<scalar_t,integer_t>
      (*matrix(), [&](DenseM_t& w) { tree()->multifrontal_solve(w); },
       x, bloc, opts_.rel_tol(), opts_.abs_tol(),
       Krylov_its_, opts_.maxit(), use_initial_guess,
       opts_.verbose() && is_root_);
    };

    switch (opts_.Krylov_solver()) {
    case KrylovSolver::AUTO: {
      if ((opts_.use_HSS() || opts_.use_BLR()) && x.cols() == 1)
        gmres_solve(MFsolve);
      else refine();
    }; break;
    case KrylovSolver::DIRECT: {
      x = bloc;
      tree()->multifrontal_solve(x);
    }; break;
    case KrylovSolver::REFINE: {
      refine();
    }; break;
    case KrylovSolver::PREC_GMRES: {
      assert(x.cols() == 1);
      gmres_solve(MFsolve);
    }; break;
    case KrylovSolver::GMRES: {
      assert(x.cols() == 1);
      gmres_solve([](scalar_t* x){});
    }; break;
    case KrylovSolver::PREC_BICGSTAB: {
      assert(x.cols() == 1);
      bicgstab_solve(MFsolve);
    }; break;
    case KrylovSolver::BICGSTAB: {
      assert(x.cols() == 1);
      bicgstab_solve([](scalar_t* x){});
    }; break;
    }

    x.lapmr(intIP, false);
    if (opts_.matching() != MatchingJob::NONE)
      x.lapmr(int_matching_cperm, false);
    if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
      x.scale_rows(matching_Dc_);

    t.stop();
    perf_counters_stop("DIRECT/GMRES solve");
    print_solve_stats(t);
    return ReturnCode::SUCCESS;
  }

} //end namespace strumpack

#endif
