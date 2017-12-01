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
/*! \file StrumpackSparseSolver.hpp
 *
 * \brief Contains the definition of the sequential/multithreaded
 * sparse solver class.
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
#include "misc/Tools.hpp"
#include "StrumpackOptions.hpp"
#include "sparse/CompressedSparseMatrix.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/MatrixReordering.hpp"
#include "sparse/EliminationTree.hpp"
#include "sparse/GMRes.hpp"
#include "sparse/BiCGStab.hpp"
#include "sparse/IterativeRefinement.hpp"

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
  template<typename scalar_t,typename integer_t=int>
  class StrumpackSparseSolver {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using Tree_t = EliminationTree<scalar_t,integer_t>;
    using Reord_t = MatrixReordering<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;

  public:
    /*! \brief Constructor of the StrumpackSparseSolver class,
     *         taking command line arguments.
     *
     * \param argc    The number of arguments, i.e,
     *                number of elements in the argv array.
     * \param argv    Command line arguments. Add -h or --help to
     *                have a description printed.
     * \param verb    Flag to enable output.
     * \param root    Flag to denote whether this process is the root MPI
     *                process. Only the root will print certain messages.
     */
    StrumpackSparseSolver
    (int argc, char* argv[], bool verbose=true, bool root=true);

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
    virtual void set_matrix(const CSRMatrix<scalar_t,integer_t>& A);
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
     * \param N           The number of rows and columns of the CSR
     *                    input matrix.
     * \param row_ptr     Indices in col_ind and values for the start
     *                    of each row. Nonzeros for row r are in
     *                    [row_ptr[r],row_ptr[r+1]).
     * \param col_ind     Column indices of each nonzero.
     * \param values      Nonzero values.
     * \param symmetric_pattern
     *                    Denotes whether the sparsity pattern of the input
     *                    matrix is symmetric.
     */
    virtual void set_csr_matrix
    (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
     const scalar_t* values, bool symmetric_pattern=false);
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
     * \param nx  This (optional) parameter is only meaningful when the
     *            matrix corresponds to a stencil on a regular mesh.
     *            The stencil is assumed to be at most 3 points wide in each
     *            dimension and only contain a single degree of freedom per
     *            grid point. The nx parameter denotes the number of grid
     *            points in the first spatial dimension.
     * \param ny  See parameters nx. Parameter ny denotes the number of
     *            gridpoints in the second spatial dimension.
     *            This should only be set if the mesh is 2 or 3 dimensional.
     * \param nz  See parameters nx. Parameter nz denotes the number of
     *            gridpoints in the third spatial dimension.
     *            This should only be set if the mesh is 3 dimensional.
     * \return    Error code.
     * \sa        set_mc64job, set_matrix_reordering_method,
     *            set_nested_dissection_parameter,
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
     *           Array should be lenght N, the dimension of the input matrix
     *           for StrumpackSparseSolver and StrumpackSparseSolverMPI. For
     *           StrumpackSparseSolverMPIDist, the length of b should be
     *           correspond the partitioning of the block-row distributed
     *           input matrix.
     * \param x  Output. Pointer to the solution vector.
     *           Array should be lenght N, the dimension of the input matrix
     *           for StrumpackSparseSolver and StrumpackSparseSolverMPI. For
     *           StrumpackSparseSolverMPIDist, the length of b should be
     *           correspond the partitioning of the block-row distributed
     *           input matrix.
     * \param use_initial_guess
     *           Set to true if x contains an intial guess to the solution.
     *           This is mainly useful when using an iterative solver.
     *           If set to false, x should not be set
     *           (but should be allocated).
     * \return Error code.
     */
    virtual ReturnCode solve
    (const scalar_t* b, scalar_t* x, bool use_initial_guess=false);

    virtual ReturnCode solve
    (const DenseM_t& b, DenseM_t& x, bool use_initial_guess=false);

    SPOptions<scalar_t>& options() { return _opts; }
    const SPOptions<scalar_t>& options() const { return _opts; }
    void set_from_options() { _opts.set_from_command_line(); }
    void set_from_options(int argc, char* argv[])
    { _opts.set_from_command_line(argc, argv); }

    /*! \brief Get the maximum rank encountered in any of the HSS
     * matrices.  Call this AFTER numerical factorization. */
    int maximum_rank() const { return tree()->maximum_rank(); }
    /*! \brief Get the number of nonzeros in the (sparse)
        factors. This is the fill-in.  * Call this AFTER numerical
        factorization. */
    std::size_t factor_nonzeros() const
    { return tree()->factor_nonzeros(); }
    /*! \brief Get the number of nonzeros in the (sparse)
     * factors. This is the fill-in.  Call this AFTER numerical
     * factorization. */
    std::size_t factor_memory() const
    { return tree()->factor_nonzeros() * sizeof(scalar_t); }
    /*! \brief Get the number of iterations performed by the outer
        (Krylov) iterative solver. */
    int Krylov_iterations() const { return _Krylov_its; }

  protected:
    virtual void setup_tree();
    virtual void setup_reordering();
    virtual int compute_reordering(int nx, int ny, int nz);
    virtual void compute_separator_reordering();

    virtual SpMat_t* matrix() { return _mat.get(); }
    virtual const SpMat_t* matrix() const { return _mat.get(); }
    virtual Reord_t* reordering() { return _nd.get(); }
    virtual const Reord_t* reordering() const { return _nd.get(); }
    virtual Tree_t* tree() { return _tree.get(); }
    virtual const Tree_t* tree() const { return _tree.get(); }

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
    std::unique_ptr<CSRMatrix<scalar_t,integer_t>> _mat;
    std::unique_ptr<MatrixReordering<scalar_t,integer_t>> _nd;
    std::unique_ptr<EliminationTree<scalar_t,integer_t>> _tree;
  };

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolver<scalar_t,integer_t>::StrumpackSparseSolver
  (bool verbose, bool root)
    : StrumpackSparseSolver<scalar_t,integer_t>(0, nullptr, verbose, root) {
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolver<scalar_t,integer_t>::StrumpackSparseSolver
  (int argc, char* argv[], bool verbose, bool root)
    : _opts(argc, argv), _is_root(root) {
    _opts.set_verbose(verbose);
    _old_handler = std::set_new_handler
      ([]{ std::cerr << "STRUMPACK: out of memory!" << std::endl; abort(); });
    papi_initialize();
    if (_opts.verbose() && _is_root) {
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
#ifdef COUNT_FLOPS
    // TODO why does this not compile on my GCC 5.0.1??
    // if (!params::flops.is_lock_free())
    //   std::cerr << "# WARNING: the flop counter is not lock free"
    //             << std::endl;
#endif
    _opts.HSS_options().set_synchronized_compression(true);
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolver<scalar_t,integer_t>::~StrumpackSparseSolver() {
    std::set_new_handler(_old_handler);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::setup_tree() {
    if (_tree) _tree.reset();
    _tree = std::unique_ptr<EliminationTree<scalar_t,integer_t>>
      (new EliminationTree<scalar_t,integer_t>
       (_opts, *matrix(), *(reordering()->sep_tree)));
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::setup_reordering() {
    if (_nd) _nd.reset();
    _nd = std::unique_ptr<MatrixReordering<scalar_t,integer_t>>
      (new MatrixReordering<scalar_t,integer_t>(matrix()->size()));
  }

  template<typename scalar_t,typename integer_t> int
  StrumpackSparseSolver<scalar_t,integer_t>::compute_reordering
  (int nx, int ny, int nz) {
    return _nd->nested_dissection(_opts, _mat.get(), nx, ny, nz);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::compute_separator_reordering() {
    _nd->separator_reordering
      (_opts, _mat.get(), _opts.verbose() && _is_root);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::papi_initialize() {
#if defined(HAVE_PAPI)
    // TODO call PAPI_library_init???
    float mflops = 0.;
    int retval = PAPI_flops(&_rtime, &_ptime, &_flpops, &mflops);
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
#if defined(HAVE_PAPI)
    float mflops = 0., rtime = 0., ptime = 0.;
    long_long flpops = 0; // cannot use class variables in openmp clause
#pragma omp parallel reduction(+:flpops) reduction(max:rtime) \
  reduction(max:ptime)
    PAPI_flops(&rtime, &ptime, &flpops, &mflops);
    _flpops = flpops; _rtime = rtime; _ptime = ptime;
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
  StrumpackSparseSolver<scalar_t,integer_t>::perf_counters_stop
  (const std::string& s) {
#if defined(HAVE_PAPI)
    float mflops = 0., rtime = 0., ptime = 0.;
    long_long flpops = 0;
#pragma omp parallel reduction(+:flpops) reduction(max:rtime)  \
  reduction(max:ptime)
    PAPI_flops(&rtime, &ptime, &flpops, &mflops);
    PAPI_dmem_info_t dmem;
    PAPI_get_dmem_info(&dmem);
    if (_opts.verbose() && _is_root) {
      std::cout << "# " << s << " PAPI stats:" << std::endl;
      std::cout << "#   - total flops = "
                << double(flpops-_flpops) << std::endl;
      std::cout << "#   - flop rate = "
                << double(flpops-_flpops)/(rtime-_rtime)/1e9
                << " GFlops/s" << std::endl;
      std::cout << "#   - real time = " << rtime-_rtime
                << " sec" << std::endl;
      std::cout << "#   - processor time = " << ptime-_ptime
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
  StrumpackSparseSolver<scalar_t,integer_t>::print_solve_stats
  (TaskTimer& t) const {
    double tel = t.elapsed();
    if (_opts.verbose() && _is_root) {
      std::cout << "# DIRECT/GMRES solve:" << std::endl;
      std::cout << "#   - abs_tol = " << _opts.abs_tol()
                << ", rel_tol = " << _opts.rel_tol()
                << ", restart = " << _opts.gmres_restart()
                << ", maxit = " << _opts.maxit() << std::endl;
      std::cout << "#   - number of Krylov iterations = "
                << _Krylov_its << std::endl;
      std::cout << "#   - solve time = " << tel << std::endl;
#ifdef COUNT_FLOPS
      std::cout << "#   - total flops = " << double(_ftot) << ", min = "
                << double(_fmin) << ", max = " << double(_fmax) << std::endl;
      std::cout << "#   - flop rate = " << _ftot / tel / 1e9 << " GFlop/s"
                << std::endl;
      std::cout << "#   - bytes moved = " << double(_btot) / 1e6
                << " MB, min = "<< double(_bmin) / 1e6
                << " MB, max = " << double(_bmax) / 1e6 << " MB" << std::endl;
      std::cout << "#   - byte rate = " << _btot / tel / 1e9 << " GByte/s"
                << std::endl;
      std::cout << "#   - solve arithmetic intensity = "
                << double(_ftot) / _btot
                << " flop/byte" << std::endl;
#endif
    }
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::set_matrix
  (const CSRMatrix<scalar_t,integer_t>& A) {
    if (_mat) _mat.reset();
    _mat = std::unique_ptr<CSRMatrix<scalar_t,integer_t>>
      (new CSRMatrix<scalar_t,integer_t>(A));
    _factored = _reordered = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::set_csr_matrix
  (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, bool symmetric_pattern) {
    if (_mat) _mat.reset();
    _mat = std::unique_ptr<CSRMatrix<scalar_t,integer_t>>
      (new CSRMatrix<scalar_t,integer_t>
       (N, row_ptr, col_ind, values, symmetric_pattern));
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
        case 1:
          std::cout << "maximum cardinality ! Doesn't work" << std::endl;
          return ReturnCode::REORDERING_ERROR;
          break;
        case 2:
          std::cout << "maximum smallest diagonal value, version 1"
                    << std::endl; break;
        case 3:
          std::cout << "maximum smallest diagonal value, version 2"
                    << std::endl; break;
        case 4:
          std::cout << "maximum sum of diagonal values" << std::endl;
          break;
        case 5:
          std::cout << "maximum matching + row and column scaling"
                    << std::endl;
          break;
        }
      }
      t1.time([&](){
          ierr = matrix()->permute_and_scale
            (_opts.mc64job(), _mc64_cperm, _mc64_Dr, _mc64_Dc);
        });
      if (ierr) {
        std::cerr << "ERROR: mc64 failed" << std::endl;
        return ReturnCode::REORDERING_ERROR;
      }
    }
    auto old_nnz = matrix()->nnz();
    TaskTimer t2("sparsity-symmetrization",
                 [&](){ matrix()->symmetrize_sparsity(); });
    if (matrix()->nnz() != old_nnz && _opts.verbose() && _is_root) {
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
    ierr = compute_reordering(nx, ny, nz);
    if (ierr) {
      std::cerr << "ERROR: nested dissection went wrong, ierr="
                << ierr << std::endl;
      return ReturnCode::REORDERING_ERROR;
    }
    matrix()->permute(reordering()->iperm, reordering()->perm);
    t3.stop();
    if (_opts.verbose() && _is_root) {
      std::cout << "#   - nd time = " << t3.elapsed() << std::endl;
      if (_opts.mc64job() != 0)
        std::cout << "#   - mc64 time = " << t1.elapsed() << std::endl;
      std::cout << "#   - symmetrization time = " << t2.elapsed()
                << std::endl;
    }
    perf_counters_stop("nested dissection");

    if (_opts.use_HSS()) {
      perf_counters_start();
      TaskTimer t4("separator-reordering", [&](){
          compute_separator_reordering();
          // TODO also broadcast this?? is computed with scotch
        });
      if (_opts.verbose() && _is_root)
        std::cout << "#   - sep-reorder time = " << t4.elapsed() << std::endl;
      perf_counters_stop("separator reordering");
    }

    perf_counters_start();
    TaskTimer t0("symbolic-factorization", [&](){ setup_tree(); });
    reordering()->clear_tree_data();
    if (_opts.verbose()) {
      // this might require a reduction
      auto nr_dense = tree()->nr_dense_fronts();
      auto nr_HSS = tree()->nr_HSS_fronts();
      if (_is_root) {
        std::cout << "# symbolic factorization:" << std::endl;
        std::cout << "#   - nr of dense Frontal matrices = "
                  << number_format_with_commas(nr_dense) << std::endl;
        std::cout << "#   - nr of HSS Frontal matrices = "
                  << number_format_with_commas(nr_HSS) << std::endl;
        std::cout << "#   - symb-factor time = " << t0.elapsed() << std::endl;
      }
    }
    perf_counters_stop("symbolic factorization");

    // if (_opts.use_HSS()) {
    //   perf_counters_start();
    //   TaskTimer t4("separator-reordering", [&](){
    //            compute_separator_reordering();
    //            // TODO also broadcast this?? is computed with scotch
    //          });
    // if (_opts.verbose() && _is_root)
    //   std::cout << "#   - sep-reorder time = "
    //             << t4.elapsed() << std::endl;
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
    TaskTimer t1("factorization", [&]() {
        tree()->multifrontal_factorization(*matrix(), _opts);
      });
    perf_counters_stop("numerical factorization");
    if (_opts.verbose()) {
      auto fnnz = factor_nonzeros();
      auto dfnnz = dense_factor_memory();
      auto max_rank = maximum_rank();
      if (_is_root) {
        std::cout << "# multifrontal factorization:" << std::endl;
        std::cout << "#   - factor time = " << t1.elapsed() << std::endl;
        std::cout << "#   - factor nonzeros = "
                  << number_format_with_commas(fnnz) << std::endl;
        std::cout << "#   - factor memory = "
                  << fnnz * sizeof(scalar_t) / 1e6 << " MB" << std::endl;
#ifdef COUNT_FLOPS
        std::cout << "#   - total flops = " << double(_ftot) << ", min = "
                  << double(_fmin) << ", max = " << double(_fmax)
                  << std::endl;
        std::cout << "#   - flop rate = " << _ftot / t1.elapsed() / 1e9
                  << " GFlop/s" << std::endl;
#endif
        std::cout << "#   - factor memory/nonzeros = "
                  << float(fnnz * sizeof(scalar_t)) / dfnnz * 100.0
                  << " % of multifrontal" << std::endl;
        std::cout << "#   - maximum HSS rank = " << max_rank << std::endl;
        std::cout << "#   - HSS compression = " << std::boolalpha
                  << _opts.use_HSS() << std::endl;
        std::cout << "#   - relative compression tolerance = "
                  << _opts.HSS_options().rel_tol() << std::endl;
        std::cout << "#   - absolute compression tolerance = "
                  << _opts.HSS_options().abs_tol() << std::endl;
        std::cout << "#   - "
                  << get_name(_opts.HSS_options().random_distribution())
                  << " distribution with "
                  << get_name(_opts.HSS_options().random_engine())
                  << " engine" << std::endl;
#if defined(COUNT_FLOPS)
        std::cout << std::endl;
        std::cout << "# ----- FLOP BREAKDOWN ---------------------"
                  << std::endl;
        std::cout << "# compression           = "
                  << float(params::compression_flops.load()) << std::endl;
        std::cout << "#    random             = "
                  << float(params::random_flops.load()) << std::endl;
        std::cout << "#    ID                 = "
                  << float(params::ID_flops.load()) << std::endl;
        std::cout << "#    QR                 = "
                  << float(params::QR_flops.load()) << std::endl;
        std::cout << "#    ortho              = "
                  << float(params::ortho_flops.load()) << std::endl;
        std::cout << "#    reduce_samples     = "
                  << float(params::reduce_sample_flops.load()) << std::endl;
        std::cout << "#    update_samples     = "
                  << float(params::update_sample_flops.load()) << std::endl;
        std::cout << "#    extraction         = "
                  << float(params::extraction_flops.load()) << std::endl;
        std::cout << "#    sampling           = "
                  << float(params::sample_flops.load()) << std::endl;
        std::cout << "#       CB_sample       = "
                  << float(params::CB_sample_flops.load()) << std::endl;
        std::cout << "#       sparse_sampling = "
                  << float(params::sparse_sample_flops.load()) << std::endl;
        std::cout << "#       intial_sampling = "
                  << float(params::initial_sample_flops.load()) << std::endl;
        std::cout << "# ULV_factor            = "
                  << float(params::ULV_factor_flops.load()) << std::endl;
        std::cout << "# Schur                 = "
                  << float(params::schur_flops.load()) << std::endl;
        std::cout << "# full_rank             = "
                  << float(params::full_rank_flops.load()) << std::endl;
        std::cout << "# --------------------------------------------"
                  << std::endl;
        std::cout << "# total                 = "
                  << float(params::compression_flops.load() +
                           params::ULV_factor_flops.load() +
                           params::schur_flops.load() +
                           params::full_rank_flops.load()) << std::endl;
        std::cout << "# --------------------------------------------"
                  << std::endl;
        std::cout << std::endl;
#endif
      }
    }
    if (_rank_out) tree()->print_rank_statistics(*_rank_out);
    _factored = true;
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
    if (!_factored) {
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
      intIP[i] = reordering()->iperm[i] + 1;
    std::vector<int> int_mc64_cperm;
    if (_opts.mc64job() != 0) {
      int_mc64_cperm.resize(N);
      for (integer_t i=0; i<N; i++)
        int_mc64_cperm[i] = _mc64_cperm[i] + 1;
    }

    auto bloc = b;
    if (_opts.mc64job() == 5)
      bloc.scale_rows(_mc64_Dr);
    bloc.lapmr(intIP, true);

    if (use_initial_guess &&
        _opts.Krylov_solver() != KrylovSolver::DIRECT) {
      if (_opts.mc64job() == 5)
        x.div_rows(_mc64_Dc);
      if (_opts.mc64job() != 0)
        x.lapmr(int_mc64_cperm, true);
      x.lapmr(intIP, true);
    }

    _Krylov_its = 0;

    auto gmres_solve = [&](std::function<void(scalar_t*)> preconditioner) {
      GMRes<scalar_t,integer_t>
      (*matrix(), preconditioner, x.rows(), x.data(), bloc.data(),
       _opts.rel_tol(), _opts.abs_tol(), _Krylov_its, _opts.maxit(),
       _opts.gmres_restart(), _opts.GramSchmidt_type(),
       use_initial_guess, _opts.verbose() && _is_root);
    };
    auto bicgstab_solve = [&](std::function<void(scalar_t*)> preconditioner) {
      BiCGStab<scalar_t,integer_t>
      (*matrix(), preconditioner, x.rows(), x.data(), bloc.data(),
       _opts.rel_tol(), _opts.abs_tol(), _Krylov_its, _opts.maxit(),
       use_initial_guess, _opts.verbose() && _is_root);
    };
    auto MFsolve = [&](scalar_t* w) {
      DenseMW_t X(x.rows(), 1, w, x.ld());
      tree()->multifrontal_solve(X);
    };
    auto refine = [&]() {
      IterativeRefinement<scalar_t,integer_t>
      (*matrix(), [&](DenseM_t& w) { tree()->multifrontal_solve(w); },
       x, bloc, _opts.rel_tol(), _opts.abs_tol(),
       _Krylov_its, _opts.maxit(), use_initial_guess,
       _opts.verbose() && _is_root);
    };

    switch (_opts.Krylov_solver()) {
    case KrylovSolver::AUTO: {
      if (_opts.use_HSS() && x.cols() == 1)
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
    if (_opts.mc64job() != 0)
      x.lapmr(int_mc64_cperm, false);
    if (_opts.mc64job() == 5)
      x.scale_rows(_mc64_Dc);

    t.stop();
    perf_counters_stop("DIRECT/GMRES solve");
    print_solve_stats(t);
    return ReturnCode::SUCCESS;
  }

} //end namespace strumpack

#endif
