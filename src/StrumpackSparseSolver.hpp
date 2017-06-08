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
/*! \file StrumpackSparseSolver.hpp
 * \brief Contains the definition of the sequential/multithreaded sparse solver class.
 */
#ifndef STRUMPACK_SOLVER_H
#define STRUMPACK_SOLVER_H

#include <sstream>
#include <getopt.h>
#include <new>
#include <cmath>
#ifdef USE_TBB_MALLOC
#include <tbb/scalable_allocator.h>
#endif
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#ifdef HAVE_PAPI
#include <papi.h>
#endif
#include "SPOptions.hpp"
#include "CompressedSparseMatrix.hpp"
#include "CSRMatrix.hpp"
#include "MatrixReordering.hpp"
#include "EliminationTree.hpp"
#include "tools.hpp"
#include "GMRes.hpp"
#include "BiCGStab.hpp"
#include "IterativeRefinement.hpp"

#ifdef USE_TBB_MALLOC
void* operator new(std::size_t sz) throw(std::bad_alloc) {
  return scalable_malloc(sz);
}
void operator delete(void* ptr) throw() {
  if (!ptr) return;
  scalable_free(ptr);
}
#endif

/*! All of STRUMPACK-sparse is contained in the strumpack namespace. */
namespace strumpack {

  /*! \brief StrumpackSparseSolver<scalar_t,integer_t> is the
   *         main sequential or multithreaded sparse solver class.
   *
   * The StrumpackSparseSolver<scalar_t,integer_t> class depends on 3
   * template paramaters. Scalar_t can be: float, double,
   * std::complex<float>, std::complex<double>. The integer_t type
   * defaults to a regular int. If regular int causes 32 bit integer
   * overflows, you should switch to integer_t=int64_t instead.
   */
  template<typename scalar_t,typename integer_t=int> class StrumpackSparseSolver {
  public:
    /*! \brief Constructor of the StrumpackSparseSolver class,
     *         taking command line arguments.
     *
     * \param argc  The number of arguments, i.e, number of elements in the argv array.
     * \param argv  Command line arguments. Add -h or --help to have a description printed.
     * \param verb  Flag to enable output.
     * \param root  Flag to denote whether this process is the root MPI process.
     *              Only the root will print certain messages.
     */
    StrumpackSparseSolver(int argc, char* argv[], bool verbose=true, bool root=true);

    StrumpackSparseSolver(bool verbose=true, bool root=true);
    /*! \brief Destructor of the StrumpackSparseSolver class. */
    virtual ~StrumpackSparseSolver();

    /*! \brief Associate a (sequential) CSRMatrix with this solver.
     *
     * This matrix will not be modified. An internal copy will be
     * made, so it is safe to delete the data immediately after
     * calling this function. When this is called on a
     * StrumpackSparseSolverMPIDist object, the input matrix will
     * immediately be distributed over all the processes in the
     * communicator associated with the solver. This method is thus
     * collective on the MPI communicator associated with the solver.
     *
     * \param A  A CSRMatrix<scalar_t,integer_t> object.
     */
    virtual void set_matrix(CSRMatrix<scalar_t,integer_t>& A);
    /*! \brief Associate a (sequential) NxN CSR matrix with this
     *         solver.
     *
     * This matrix will not be modified. An internal copy will be
     * made, so it is safe to delete the data immediately after
     * calling this function. When this is called on a
     * StrumpackSparseSolverMPIDist object, the input matrix will
     * immediately be distributed over all the processes in the
     * communicator associated with the solver. This method is thus
     * collective on the MPI communicator associated with the solver.
     *
     * \param N                 The number of rows and columns of the CSR input matrix.
     * \param row_ptr           Indices in col_ind and values for the start of each row.
     *                          Nonzeros for row r are in [row_ptr[r],row_ptr[r+1]).
     * \param col_ind           Column indices of each nonzero.
     * \param values            Nonzero values.
     * \param symmetric_pattern Denotes whether the sparsity pattern of the input matrix is symmetric.
     */
    virtual void set_csr_matrix(integer_t N, integer_t* row_ptr, integer_t* col_ind, scalar_t* values, bool symmetric_pattern=false);
    /*! \brief Compute matrix reorderings for numerical stability and
     *         to reduce fill-in.
     *
     * Start computation of the matrix reorderings. See the relevant
     * options to control the matrix reordering. A first reordering is
     * the MC64 column permutation for numerical stability. This can
     * be disabled if the matrix has large nonzero diagonal
     * entries. MC64 optionally also performs row and column
     * scaling. Next, a fill-reducing reordering is computed. This is
     * done with the nested dissection algortihms of either
     * (PT-)Scotch, (Par)Metis or a simple geometric nested dissection
     * code which only works on regular meshes.
     *
     * \param nx  This (optional) parameter is only meaningful when the matrix corresponds
     *            to a stencil on a regular mesh. The stecil is assumed to be at most 3 points
     *            wide in each dimension and only contain a single degree of freedom per grid point.
     *            The nx parameter denotes the number of grid points in the first spatial dimension.
     * \param ny  See parameters nx. Parameter ny denotes the number of gridpoints in the second
     *            spatial dimension. This should only be set if the mesh is 2 or 3 dimensional.
     * \param nz  See parameters nx. Parameter nz denotes the number of gridpoints in the third
     *            spatial dimension. This should only be set if the mesh is 3 dimensional.
     * \return    Error code.
     * \sa        set_mc64job, set_matrix_reordering_method, set_nested_dissection_parameter,
     *            set_scotch_strategy
     */
    virtual ReturnCode reorder(int nx=1, int ny=1, int nz=1);
    /*! \brief Perform numerical factorization of the sparse input matrix.
     *
     * This is the computationally expensive part of the code.
     * \return    Error code.
     */
    ReturnCode factor();
    /*! \brief  Solve a linear system with a single right-hand side.
     *
     * \param b  Input, will not be modified. Pointer to the right-hand side.
     *           Array should be lenght N, the dimension of the input matrix for
     *           StrumpackSparseSolver and StrumpackSparseSolverMPI. For
     *           StrumpackSparseSolverMPIDist, the length of b should be correspond
     *           the partitioning of the block-row distributed input matrix.
     * \param x  Output. Pointer to the solution vector.
     *           Array should be lenght N, the dimension of the input matrix for
     *           StrumpackSparseSolver and StrumpackSparseSolverMPI. For
     *           StrumpackSparseSolverMPIDist, the length of b should be correspond
     *           the partitioning of the block-row distributed input matrix.
     * \param use_initial_guess Set to true if x contains an intial guess to the solution.
     *                          This is mainly useful when using an iterative solver.
     *                          If set to false, x should not be set (but should be allocated).
     * \return    Error code.
     */
    virtual ReturnCode solve(scalar_t* b, scalar_t* x, bool use_initial_guess=false);

    SPOptions<scalar_t>& options() { return _opts; }
    void set_from_options() { _opts.set_from_command_line(); }
    void set_from_options(int argc, char* argv[]) { _opts.set_from_command_line(argc, argv); }

    /*! \brief Get the maximum rank encountered in any of the HSS matrices.
     * Call this AFTER numerical factorization. */
    int maximum_rank() { return elimination_tree()->maximum_rank(); }
    /*! \brief Get the number of nonzeros in the (sparse) factors. This is the fill-in.
     * Call this AFTER numerical factorization. */
    std::size_t factor_nonzeros() { return elimination_tree()->factor_nonzeros(); }
    /*! \brief Get the number of nonzeros in the (sparse) factors. This is the fill-in.
     * Call this AFTER numerical factorization. */
    std::size_t factor_memory() { return elimination_tree()->factor_nonzeros() * sizeof(scalar_t); }
    /*! \brief Get the number of iterations performed by the outer (Krylov) iterative solver. */
    int Krylov_iterations() { return _Krylov_its; }

  protected:
    virtual void setup_elimination_tree();
    virtual void setup_matrix_reordering();
    virtual int compute_reordering(int nx, int ny, int nz);
    virtual void compute_separator_reordering();
    virtual CompressedSparseMatrix<scalar_t,integer_t>* matrix() { return _mat; }
    virtual MatrixReordering<scalar_t,integer_t>* reordering() { return _nd; }
    virtual EliminationTree<scalar_t,integer_t>* elimination_tree() { return _et; }
    void papi_initialize();
    inline long long dense_factor_nonzeros() { return elimination_tree()->dense_factor_nonzeros(); }
    inline long long dense_factor_memory() { return elimination_tree()->dense_factor_nonzeros() * sizeof(scalar_t); }
    void print_solve_stats(TaskTimer& t);
    virtual void perf_counters_start();
    virtual void perf_counters_stop(std::string s);
    virtual void synchronize() {}
    virtual void communicate_ordering() {}

    SPOptions<scalar_t> _opts;
    bool _is_root;
    std::vector<integer_t> _mc64_cperm;
    std::vector<scalar_t> _mc64_Dr; // row scaling
    std::vector<scalar_t> _mc64_Dc; // column scaling
    std::new_handler _old_handler;
    std::ostream* _rank_out = nullptr;
    bool _factored = false;
    bool _reordered = false;
    int _Krylov_its = 0;

#if defined(HAVE_PAPI)
    float _rtime = 0., _ptime = 0.;
    long_long _flpops = 0;
#endif
#if defined(COUNT_FLOPS)
    long long int _f0 = 0, _ftot = 0, _fmin = 0, _fmax = 0;
    long long int _b0 = 0, _btot = 0, _bmin = 0, _bmax = 0;
#endif
  private:
    CSRMatrix<scalar_t,integer_t>* _mat = nullptr;
    MatrixReordering<scalar_t,integer_t>* _nd = nullptr;
    EliminationTree<scalar_t,integer_t>* _et = nullptr;
  };

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolver<scalar_t,integer_t>::StrumpackSparseSolver(bool verbose, bool root)
    : StrumpackSparseSolver<scalar_t,integer_t>(0, nullptr, verbose, root) {
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolver<scalar_t,integer_t>::StrumpackSparseSolver
  (int argc, char* argv[], bool verbose, bool root) : _opts(argc, argv), _is_root(root) {
    _opts.set_verbose(verbose);
    _old_handler = std::set_new_handler([]{ std::cerr << "STRUMPACK: out of memory!" << std::endl; abort(); });
    papi_initialize();
    if (_opts.verbose() && _is_root) {
      std::cout << "# Initializing STRUMPACK" << std::endl;
#if defined(_OPENMP)
      if (params::num_threads == 1)
	std::cout << "# using " << params::num_threads << " OpenMP thread" << std::endl;
      else std::cout << "# using " << params::num_threads << " OpenMP threads" << std::endl;
#else
      std::cout << "# running serially, no OpenMP support!" << std::endl;
#endif
      // a heuristic to set the recursion task cutoff level based on the number of threads
      if (params::num_threads == 1) params::task_recursion_cutoff_level = 0;
      else {
	params::task_recursion_cutoff_level = std::log2(params::num_threads) + 3;
	std::cout << "# number of tasking levels = " << params::task_recursion_cutoff_level
		  << " = log_2(#threads) + 3"<< std::endl;
      }
    }
#ifdef COUNT_FLOPS
    // TODO why does this not compile on my GCC 5.0.1??
    //if (!params::flops.is_lock_free()) std::cerr << "# WARNING: the flop counter is not lock free" << std::endl;
#endif
    _opts.HSS_options().set_synchronized_compression(true);
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolver<scalar_t,integer_t>::~StrumpackSparseSolver() {
    std::set_new_handler(_old_handler);
    delete _nd;
    delete _et;
    delete _mat;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::setup_elimination_tree() {
    if (_et) delete _et;
    _et = new EliminationTree<scalar_t,integer_t>
      (_opts, matrix(), reordering()->sep_tree.get());
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::setup_matrix_reordering() {
    if (_nd) delete _nd;
    _nd = new MatrixReordering<scalar_t,integer_t>(matrix()->size());
  }

  template<typename scalar_t,typename integer_t> int
  StrumpackSparseSolver<scalar_t,integer_t>::compute_reordering(int nx, int ny, int nz) {
    return _nd->nested_dissection(_opts, _mat, nx, ny, nz);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::compute_separator_reordering() {
    //_nd->separator_reordering(_opts, _mat, _et->root_front());
    _nd->separator_reordering(_opts, _mat, _opts.verbose() && _is_root);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::papi_initialize() {
#if defined(HAVE_PAPI)
    float mflops = 0.;
    int retval = PAPI_flops(&_rtime, &_ptime, &_flpops, &mflops);
    if (retval!=PAPI_OK) {
      std::cerr << "# WARNING: problem starting the PAPI performance counters:" << std::endl;
      switch (retval) {
      case PAPI_EINVAL: std::cerr << "#   - the counters were already started by something other than: PAPI_flips() or PAPI_flops()." << std::endl; break;
      case PAPI_ENOEVNT: std::cerr << "#   - the floating point operations, floating point instructions or total cycles event does not exist." << std::endl; break;
      case PAPI_ENOMEM: std::cerr << "#   - insufficient memory to complete the operation." << std::endl; break;
      default: std::cerr << "#   - some other error: " << retval << std::endl;
      }
    }
#endif
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::perf_counters_start() {
#if defined(HAVE_PAPI)
    float mflops = 0., rtime = 0., ptime = 0.;
    long_long flpops = 0; // cannot use class variables in openmp clause
#pragma omp parallel reduction(+:flpops) reduction(max:rtime) reduction(max:ptime)
    PAPI_flops(&rtime, &ptime, &flpops, &mflops);
    _flpops0 = flpops; _rtime = rtime; _ptime = ptime;
#endif
#ifdef COUNT_FLOPS
    _f0 = _b0 = 0;
#pragma omp parallel reduction(+:_f0) reduction(+:_b0)
    {
      _f0 = params::flops;
      _b0 = params::bytes;
    }
#endif
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::perf_counters_stop(std::string s) {
#if defined(HAVE_PAPI)
    float mflops = 0., rtime1 = 0., ptime1 = 0.;
    long_long flpops1 = 0;
#pragma omp parallel reduction(+:flpops1) reduction(max:rtime) reduction(max:ptime1)
    PAPI_flops(&rtime1, &ptime1, &flpops1, &mflops);
    if (_opts.verbose() && _is_root)
      std::cout << "# " << s << " PAPI stats:" << std::endl
		<< "#   - total flops = " << double(flpops1-_flpops) << std::endl
		<< "#   - flop rate = " <<  double(flpops1-_flpops)/(rtime1-_rtime)/1e9 << " GFlops/s" << std::endl
		<< "#   - real time = " << rtime1-_rtime << " sec" << std::endl
		<< "#   - processor time = " << ptime1-_ptime << " sec" << std::endl;
#endif
#ifdef COUNT_FLOPS
    _ftot = -_f0;
    _btot = -_b0;
#pragma omp parallel reduction(+:_ftot) reduction(+:_btot)
    {
      _ftot = params::flops;
      _btot = params::bytes;
    }
    _fmin = _fmax = _ftot;
    _bmin = _bmax = _btot;
#endif
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::print_solve_stats(TaskTimer& t) {
    double tel = t.elapsed();
    if (_opts.verbose() && _is_root) {
      std::cout << "# DIRECT/GMRES solve:" << std::endl
		<< "#   - abs_tol = " << _opts.abs_tol()
		<< ", rel_tol = " << _opts.rel_tol()
		<< ", restart = " << _opts.gmres_restart()
		<< ", maxit = " << _opts.maxit() << std::endl
		<< "#   - number of Krylov iterations = " << _Krylov_its << std::endl
		<< "#   - solve time = " << tel << std::endl;
#ifdef COUNT_FLOPS
      std::cout << "#   - total flops = " << double(_ftot) << ", min = " << double(_fmin) << ", max = " << double(_fmax) << std::endl
		<< "#   - flop rate = " << _ftot / tel / 1e9 << " GFlop/s" << std::endl
		<< "#   - bytes moved = " << double(_btot) / 1e6 << " MB, min = "<< double(_bmin) / 1e6
		<< " MB, max = " << double(_bmax) / 1e6 << " MB" << std::endl
		<< "#   - byte rate = " << _btot / tel / 1e9 << " GByte/s" << std::endl
		<< "#   - solve arithmetic intensity = " << double(_ftot) / _btot
		<< " flop/byte" << std::endl;
#endif
    }
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::set_matrix(CSRMatrix<scalar_t,integer_t>& A) {
    if (_mat) delete _mat;
    _mat = A.clone();
    _factored = _reordered = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::set_csr_matrix
  (integer_t N, integer_t* row_ptr, integer_t* col_ind, scalar_t* values, bool symmetric_pattern) {
    if (_mat) delete _mat;
    _mat = new CSRMatrix<scalar_t,integer_t>(N, row_ptr, col_ind, values, symmetric_pattern);
    _factored = _reordered = false;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolver<scalar_t,integer_t>::reorder(int nx, int ny, int nz) {
    if (!matrix()) return ReturnCode::MATRIX_NOT_SET;
    TaskTimer t1("permute-scale");
    int ierr;
    if (_opts.mc64job() != 0) {
      if (_opts.verbose() && _is_root) {
	std::cout << "# mc64ad job " << _opts.mc64job() << ": ";
	switch (_opts.mc64job()) {
	case 1: std::cout << "maximum cardinality ! Doesn't work" << std::endl; return ReturnCode::REORDERING_ERROR; break;
	case 2: std::cout << "maximum smallest diagonal value, version 1" << std::endl; break;
	case 3: std::cout << "maximum smallest diagonal value, version 2" << std::endl; break;
	case 4: std::cout << "maximum sum of diagonal values" << std::endl; break;
	case 5: std::cout << "maximum matching + row and column scaling" << std::endl; break;
	}
      }
      t1.time([&](){ ierr = matrix()->permute_and_scale(_opts.mc64job(), _mc64_cperm, _mc64_Dr, _mc64_Dc); });
      if (ierr) {
	std::cerr << "ERROR: mc64 failed" << std::endl;
	return ReturnCode::REORDERING_ERROR;
      }
    }
    auto old_nnz = matrix()->nnz();
    TaskTimer t2("sparsity-symmetrization", [&](){ matrix()->symmetrize_sparsity(); });
    if (matrix()->nnz() != old_nnz && _opts.verbose() && _is_root)
      std::cout << "# Matrix padded with zeros to get symmetric pattern." << std::endl
		<< "# Number of nonzeros increased from "
		<< number_format_with_commas(old_nnz) << " to "
		<< number_format_with_commas(matrix()->nnz()) << "." << std::endl;

    TaskTimer t3("nested-dissection");
    perf_counters_start();
    t3.start();
    setup_matrix_reordering();
    ierr = compute_reordering(nx, ny, nz);
    if (ierr) { std::cerr << "ERROR: nested dissection went wrong, ierr=" << ierr << std::endl; return ReturnCode::REORDERING_ERROR; }
    matrix()->permute(reordering()->iperm, reordering()->perm);
    t3.stop();
    if (_opts.verbose() && _is_root) {
      std::cout << "#   - nd time = " << t3.elapsed() << std::endl;
      if (_opts.mc64job() != 0) std::cout << "#   - mc64 time = " << t1.elapsed() << std::endl;
      std::cout << "#   - symmetrization time = " << t2.elapsed() << std::endl;
    }
    perf_counters_stop("nested dissection");

    if (_opts.use_HSS()) {
      perf_counters_start();
      TaskTimer t4("separator-reordering", [&](){
	  compute_separator_reordering();
	  // TODO also broadcast this?? is computed with scotch
	});
      if (_opts.verbose() && _is_root) std::cout << "#   - sep-reorder time = " << t4.elapsed() << std::endl;
      perf_counters_stop("separator reordering");
    }

    perf_counters_start();
    TaskTimer t0("symbolic-factorization", [&](){ setup_elimination_tree(); });
    reordering()->clear_tree_data();
    if (_opts.verbose()) {
      auto nr_dense = elimination_tree()->nr_dense_fronts(); // this might require a reduction
      auto nr_HSS = elimination_tree()->nr_HSS_fronts();
      if (_is_root)
	std::cout << "# symbolic factorization:" << std::endl
		  << "#   - nr of dense Frontal matrices = " << number_format_with_commas(nr_dense) << std::endl
		  << "#   - nr of HSS Frontal matrices = " << number_format_with_commas(nr_HSS) << std::endl
		  << "#   - symb-factor time = " << t0.elapsed() << std::endl;
    }
    perf_counters_stop("symbolic factorization");

    // if (_opts.use_HSS()) {
    //   perf_counters_start();
    //   TaskTimer t4("separator-reordering", [&](){
    // 	  compute_separator_reordering();
    // 	  // TODO also broadcast this?? is computed with scotch
    // 	});
    //   if (_opts.verbose() && _is_root) std::cout << "#   - sep-reorder time = " << t4.elapsed() << std::endl;
    //   perf_counters_stop("separator reordering");
    // }

    _reordered = true;
    return ReturnCode::SUCCESS;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolver<scalar_t,integer_t>::factor() {
    if (!matrix()) return ReturnCode::MATRIX_NOT_SET;
    if (_factored) return ReturnCode::SUCCESS;
    if (!_reordered) {
      ReturnCode ierr = reorder();
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }
    perf_counters_start();
    TaskTimer t1("factorization", [&]() { elimination_tree()->multifrontal_factorization(_opts); });
    perf_counters_stop("numerical factorization");
    if (_opts.verbose()) {
      auto fnnz = factor_nonzeros();
      auto dfnnz = dense_factor_memory();
      auto max_rank = maximum_rank();
      if (_is_root) {
	std::cout << "# multifrontal factorization:" << std::endl
		  << "#   - factor time = " << t1.elapsed() << std::endl
		  << "#   - factor nonzeros = " << number_format_with_commas(fnnz) << std::endl
		  << "#   - factor memory = " << fnnz * sizeof(scalar_t) / 1e6 << " MB" << std::endl;
#ifdef COUNT_FLOPS
	std::cout << "#   - total flops = " << double(_ftot) << ", min = " << double(_fmin) << ", max = " << double(_fmax) << std::endl
		  << "#   - flop rate = " << _ftot / t1.elapsed() / 1e9 << " GFlop/s" << std::endl;
#endif
	std::cout << "#   - factor memory/nonzeros = " << float(fnnz * sizeof(scalar_t)) / dfnnz * 100.0 << " % of multifrontal" << std::endl
		  << "#   - maximum HSS rank = " << max_rank << std::endl
		  << "#   - HSS compression = " << std::boolalpha << _opts.use_HSS() << std::endl
		  << "#   - relative compression tolerance = " << _opts.HSS_options().rel_tol() << std::endl
		  << "#   - absolute compression tolerance = " << _opts.HSS_options().abs_tol() << std::endl
		  << "#   - " << get_name(_opts.HSS_options().random_distribution()) << " distribution with "
		  << get_name(_opts.HSS_options().random_engine()) << " engine" << std::endl;
#if defined(COUNT_FLOPS)
	std::cout << std::endl;
	std::cout << "# ----- FLOP BREAKDOWN ---------------------" << std::endl;
	std::cout << "# compression           = " << float(params::compression_flops.load()) << std::endl;
	std::cout << "#    random             = " << float(params::random_flops.load()) << std::endl;
	std::cout << "#    ID                 = " << float(params::ID_flops.load()) << std::endl;
	std::cout << "#    QR                 = " << float(params::QR_flops.load()) << std::endl;
	std::cout << "#    ortho              = " << float(params::ortho_flops.load()) << std::endl;
	std::cout << "#    reduce_samples     = " << float(params::reduce_sample_flops.load()) << std::endl;
	std::cout << "#    update_samples     = " << float(params::update_sample_flops.load()) << std::endl;
	std::cout << "#    extraction         = " << float(params::extraction_flops.load()) << std::endl;
	std::cout << "#    sampling           = " << float(params::sample_flops.load()) << std::endl;
	std::cout << "#       CB_sample       = " << float(params::CB_sample_flops.load()) << std::endl;
	std::cout << "#       sparse_sampling = " << float(params::sparse_sample_flops.load()) << std::endl;
	std::cout << "#       intial_sampling = " << float(params::initial_sample_flops.load()) << std::endl;
	std::cout << "# ULV_factor            = " << float(params::ULV_factor_flops.load()) << std::endl;
	std::cout << "# Schur                 = " << float(params::schur_flops.load()) << std::endl;
	std::cout << "# full_rank             = " << float(params::full_rank_flops.load()) << std::endl;
	std::cout << "# --------------------------------------------" << std::endl;
	std::cout << "# total                 = " << float(params::compression_flops.load() +
							   params::ULV_factor_flops.load() +
							   params::schur_flops.load() +
							   params::full_rank_flops.load()) << std::endl;
        std::cout << "# --------------------------------------------" << std::endl;
	std::cout << std::endl;
#endif
      }
    }
    if (_rank_out) elimination_tree()->print_rank_statistics(*_rank_out);
    _factored = true;
    return ReturnCode::SUCCESS;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolver<scalar_t,integer_t>::solve
  (scalar_t* b, scalar_t* x, bool use_initial_guess) {
    if (!_factored) {
      ReturnCode ierr = factor();
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }
    TaskTimer t("solve");
    perf_counters_start();
    t.start();
    auto N = matrix()->size();

    auto b_loc = new scalar_t[N];
    if (_opts.mc64job() == 5) x_mult_y(N, b, _mc64_Dr.data(), b_loc);
    else { std::copy(b, b+N, b_loc); STRUMPACK_BYTES(sizeof(scalar_t)*static_cast<long long int>(N)*3); }
    permute_vector(N, b_loc, reordering()->iperm, 1);

    if (!use_initial_guess) {
      std::fill(x, x+N, scalar_t(0.)); STRUMPACK_BYTES(sizeof(scalar_t)*static_cast<long long int>(N)*2);
    } else {
      if (_opts.mc64job() == 5) x_div_y(N, x, _mc64_Dc.data());
      if (_opts.mc64job() != 0) permute_vector(N, x, _mc64_cperm, 1);
      permute_vector(N, x, reordering()->iperm, 1);
    }
    _Krylov_its = 0;
    elimination_tree()->allocate_solve_work_memory(); // for 1 vector

    auto gmres_solve = [&](std::function<void(scalar_t*)> preconditioner) {
      GMRes<scalar_t,integer_t>
      (matrix(), preconditioner, N, x, b_loc, _opts.rel_tol(), _opts.abs_tol(),
       _Krylov_its, _opts.maxit(), _opts.gmres_restart(), _opts.GramSchmidt_type(),
       use_initial_guess, _opts.verbose() && _is_root);
    };
    auto bicgstab_solve = [&](std::function<void(scalar_t*)> preconditioner) {
      BiCGStab<scalar_t,integer_t>
      (matrix(), preconditioner, N, x, b_loc, _opts.rel_tol(), _opts.abs_tol(),
       _Krylov_its, _opts.maxit(), use_initial_guess, _opts.verbose() && _is_root);
    };
    auto loc_et = elimination_tree();
    auto refine = [&]() {
      IterativeRefinement<scalar_t,integer_t>
      (matrix(), [loc_et](scalar_t* x){ loc_et->multifrontal_solve(x); },
       N, x, b_loc, _opts.rel_tol(), _opts.abs_tol(), _Krylov_its, _opts.maxit(),
       use_initial_guess, _opts.verbose() && _is_root);
    };
    switch (_opts.Krylov_solver()) {
    case KrylovSolver::AUTO: {
      if (_opts.use_HSS()) gmres_solve([loc_et,N](scalar_t* x){ loc_et->multifrontal_solve(x); });
      else refine();
    }; break;
    case KrylovSolver::DIRECT: {
      std::copy(b_loc, b_loc+N, x);
      elimination_tree()->multifrontal_solve(x);
      STRUMPACK_BYTES(sizeof(scalar_t)*static_cast<long long int>(N)*3);
    }; break;
    case KrylovSolver::REFINE:     { refine(); }; break;
    case KrylovSolver::PREC_GMRES: { gmres_solve([loc_et](scalar_t* x){ loc_et->multifrontal_solve(x); }); }; break;
    case KrylovSolver::GMRES:      { gmres_solve([](scalar_t* x){}); }; break;
    case KrylovSolver::PREC_BICGSTAB: { bicgstab_solve([loc_et](scalar_t* x){ loc_et->multifrontal_solve(x); }); }; break;
    case KrylovSolver::BICGSTAB:      { bicgstab_solve([](scalar_t* x){}); }; break;
    }

    delete[] b_loc;
    elimination_tree()->delete_solve_work_memory();

    permute_vector(N, x, reordering()->perm, 1);
    if (_opts.mc64job() != 0) permute_vector(N, x, _mc64_cperm, 0);
    if (_opts.mc64job() == 5) x_mult_y(N, x, _mc64_Dc.data());

    t.stop();
    perf_counters_stop("DIRECT/GMRES solve");
    print_solve_stats(t);
    return ReturnCode::SUCCESS;
  }

} //end namespace strumpack

#endif
