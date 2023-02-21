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
 * \file SparseSolverBase.hpp
 * \brief Contains the definition of the base (abstract/pure virtual)
 * sparse solver class.
 */
#ifndef STRUMPACK_SPARSE_SOLVER_BASE_HPP
#define STRUMPACK_SPARSE_SOLVER_BASE_HPP

#include <new>
#include <memory>
#include <vector>
#include <string>

#include "StrumpackConfig.hpp"
#include "StrumpackOptions.hpp"
#include "sparse/CSRMatrix.hpp"
#include "dense/DenseMatrix.hpp"

/**
 * All of STRUMPACK is contained in the strumpack namespace.
 */
namespace strumpack {

  // forward declarations
  template<typename scalar_t,typename integer_t> class MatrixReordering;
  template<typename scalar_t,typename integer_t> class EliminationTree;
  class TaskTimer;

  /**
   * \class SparseSolverBase
   *
   * \brief SparseSolverBase is the virtual base for both the
   * sequential/multithreaded and distributed sparse solver classes.
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
   * \see SparseSolverMPIDist,
   * SparseSolverMixedPrecision
   */
  template<typename scalar_t,typename integer_t=int>
  class SparseSolverBase {

    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using Tree_t = EliminationTree<scalar_t,integer_t>;
    using Reord_t = MatrixReordering<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;

  public:

    /**
     * Constructor of the SparseSolver class, taking command
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
    SparseSolverBase(int argc, char* argv[],
                     bool verbose=true, bool root=true);

    /**
     * Constructor of the SparseSolver class.
     *
     * \param verbose flag to enable/disable output to cout
     * \param root flag to denote whether this process is the root MPI
     * process. Only the root will print certain messages
     * \see set_from_options
     */
    SparseSolverBase(bool verbose=true, bool root=true);

    /**
     * (Virtual) destructor of the SparseSolver class.
     */
    virtual ~SparseSolverBase();

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
     * All parameters here are optional and only used with the
     * geometric ordering, which only makes sense on a regular 1D, 2D
     * or 3D mesh, in the natural ordering.
     *
     * \param nx this (optional) parameter is only meaningful when the
     * matrix corresponds to a stencil on a regular mesh. The nx
     * parameter denotes the number of grid points in the first
     * spatial dimension.
     * \param ny see parameters nx. Parameter ny denotes the number of
     * gridpoints in the second spatial dimension.  This should only
     * be set if the mesh is 2 or 3 dimensional, otherwise it can be
     * 1 (default).
     * \param nz See parameters nx. Parameter nz denotes the number of
     * gridpoints in the third spatial dimension.  This should only be
     * set if the mesh is 3 dimensional, otherwise it can be 1
     * (default).
     * \param components Number of degrees of freedom per grid point
     * (default 1)
     * \param width Width of the stencil, a 1D 3-point stencil needs a
     * separator of width 1, a 1D 5-point wide stencil needs a
     * separator of width 2 (default 1).
     * \return error code
     * \see SPOptions
     */
    ReturnCode reorder(int nx=1, int ny=1, int nz=1,
                       int components=1, int width=1);

    /**
     * Perform sparse matrix reordering, with a user-supplied
     * permutation vector. Using this will ignore the reordering
     * method selected in the options struct.
     *
     * \param p permutation vector, should be of size N, the size of
     * the sparse matrix associated with this solver
     * \param base is the permutation 0 or 1 based?
     */
    ReturnCode reorder(const int* p, int base=0);

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
     * OBSOLETE if the factors fit in device memory they will already
     * be on the device
     */
    void move_to_gpu();
    /**
     * TODO implement this to clear the device memory, but still keep
     * the factors in host memory
     */
    void remove_from_gpu();

    /**
     * Solve a linear system with a single right-hand side. Before
     * being able to solve a linear system, the matrix needs to be
     * factored. One can call factor() explicitly, or if this was not
     * yet done, this routine will call factor() internally.
     *
     * \param b input, will not be modified. Pointer to the right-hand
     * side. Array should be lenght N, the dimension of the input
     * matrix for SparseSolver and SparseSolverMPI. For
     * SparseSolverMPIDist, the length of b should be correspond the
     * partitioning of the block-row distributed input matrix.
     * \param x Output, pointer to the solution vector.  Array should
     * be lenght N, the dimension of the input matrix for SparseSolver
     * and SparseSolverMPI. For SparseSolverMPIDist, the length of b
     * should be correspond the partitioning of the block-row
     * distributed input matrix.
     * \param use_initial_guess set to true if x contains an intial
     * guess to the solution. This is mainly useful when using an
     * iterative solver. If set to false, x should not be set (but
     * should be allocated).
     * \return error code
     */
    ReturnCode solve(const scalar_t* b, scalar_t* x,
                     bool use_initial_guess=false);

    /**
     * Solve a linear system with a single or multiple right-hand
     * sides. Before being able to solve a linear system, the matrix
     * needs to be factored. One can call factor() explicitly, or if
     * this was not yet done, this routine will call factor()
     * internally.
     *
     * \param b input, will not be modified. DenseMatrix containgin
     * the right-hand side vector/matrix. Should have N rows, with N
     * the dimension of the input matrix for SparseSolver and
     * SparseSolverMPI. For SparseSolverMPIDist, the number or rows of
     * b should be correspond to the partitioning of the block-row
     * distributed input matrix.
     * \param x Output, pointer to the solution vector.  Array should
     * be lenght N, the dimension of the input matrix for SparseSolver
     * and SparseSolverMPI. For SparseSolverMPIDist, the length of b
     * should be correspond the partitioning of the block-row
     * distributed input matrix.
     * \param use_initial_guess set to true if x contains an intial
     * guess to the solution.  This is mainly useful when using an
     * iterative solver.  If set to false, x should not be set (but
     * should be allocated).
     * \return error code
     * \see DenseMatrix, solve(), factor()
     */
    ReturnCode solve(const DenseM_t& b, DenseM_t& x,
                     bool use_initial_guess=false);

    /**
     * Solve a linear system with a single or multiple right-hand
     * sides. Before being able to solve a linear system, the matrix
     * needs to be factored. One can call factor() explicitly, or if
     * this was not yet done, this routine will call factor()
     * internally.
     *
     * \param nrhs Number of right hand sides.
     * \param b input, will not be modified. DenseMatrix containgin
     * the right-hand side vector/matrix. Should have N rows, with N
     * the dimension of the input matrix for SparseSolver and
     * SparseSolverMPI. For SparseSolverMPIDist, the number or rows of
     * b should be correspond to the partitioning of the block-row
     * distributed input matrix.
     * \param ldb leading dimension of b
     * \param x Output, pointer to the solution vector.  Array should
     * be lenght N, the dimension of the input matrix for SparseSolver
     * and SparseSolverMPI. For SparseSolverMPIDist, the length of b
     * should be correspond the partitioning of the block-row
     * distributed input matrix.
     * \param ldx leading dimension of x
     * \param use_initial_guess set to true if x contains an intial
     * guess to the solution.  This is mainly useful when using an
     * iterative solver.  If set to false, x should not be set (but
     * should be allocated).
     * \return error code
     * \see DenseMatrix, solve(), factor()
     */
    ReturnCode solve(int nrhs, const scalar_t* b, int ldb,
                     scalar_t* x, int ldx,
                     bool use_initial_guess=false);

    /**
     * Return the object holding the options for this sparse solver.
     */
    SPOptions<scalar_t>& options();

    /**
     * Return the object holding the options for this sparse solver.
     */
    const SPOptions<scalar_t>& options() const;

    /**
     * Parse the command line options passed in the constructor, and
     * modify the options object accordingly. Run with option -h or
     * --help to see a list of supported options. Or check the
     * SPOptions documentation.
     */
    void set_from_options();

    /**
     * Parse the command line options, and modify the options object
     * accordingly. Run with option -h or --help to see a list of
     * supported options. Or check the SPOptions documentation.
     *
     * \param argc number of options in argv
     * \param argv list of options
     */
    void set_from_options(int argc, char* argv[]);

    /**
     * Return the maximum rank encountered in any of the HSS matrices
     * used to compress the sparse triangular factors. This should be
     * called after the factorization phase. For the SparseSolverMPI
     * and SparseSolverMPIDist distributed memory solvers, this
     * routine is collective on the MPI communicator.
     */
    int maximum_rank() const;

    /**
     * Return the number of nonzeros in the (sparse) factors. This is
     * known as the fill-in. This should be called after computing the
     * numerical factorization. For the SparseSolverMPI and
     * SparseSolverMPIDist distributed memory solvers, this routine is
     * collective on the MPI communicator.
     */
    std::size_t factor_nonzeros() const;

    /**
     * Return the amount of memory taken by the sparse factorization
     * factors. This is the fill-in. It is simply computed as
     * factor_nonzeros() * sizeof(scalar_t), so it does not include
     * any overhead from the metadata for the datastructures. This
     * should be called after the factorization. For the
     * SparseSolverMPI and SparseSolverMPIDist distributed memory
     * solvers, this routine is collective on the MPI communicator.
     */
    std::size_t factor_memory() const;

    /**
     * Return the number of iterations performed by the outer (Krylov)
     * iterative solver. Call this after calling the solve routine.
     */
    int Krylov_iterations() const;


    /**
     * Return the inertia of the matrix. A sparse matrix needs to be
     * set before inertia can be computed. The matrix needs to be
     * factored. If this->factor() was not called already, then it is
     * called inside the inertia routine.
     *
     * To get accurate inertia the matching needs to be disabled,
     * because the matching applies a non-symmetric permutation.
     * Matching can be disabled using
     * this->options().set_matching(strumpack::MatchingJob::NONE);
     *
     * The inertia will not be correct if pivoting was performed, in
     * which case the return value will be
     * ReturnCode::INACCURATE_INERTIA.  Inertia also cannot be
     * computed when compression is applied (fi, HSS, HODLR, ...).
     *
     * \param neg number of negative eigenvalues (if return value is
     * ReturnCode::SUCCESS)
     * \param zero number of zero eigenvalues (if return value is
     * ReturnCode::SUCCESS)
     * \param pos number of positive eigenvalues (if return value is
     * ReturnCode::SUCCESS)
     */
    ReturnCode inertia(integer_t& neg, integer_t& zero, integer_t& pos);

    /**
     * Create a gnuplot script to draw/plot the sparse factors. Only
     * do this for small matrices! It is very slow!
     *
     * \param name filename of the generated gnuplot script. Running
     * \verbatim gnuplot plotname.gnuplot \endverbatim will generate a
     * pdf file.
     */
    void draw(const std::string& name) const;

    /**
     * Free memory held by the factors.  This does not delete the
     * symbolic analysis information. So after calling this routine,
     * one can still update the sparse matrix values using for
     * instance update_matrix_values.
     */
    void delete_factors();

  protected:
    virtual void setup_tree() = 0;
    virtual void setup_reordering() = 0;
    virtual
    int compute_reordering(const int* p, int base,
                           int nx, int ny, int nz,
                           int components, int width) = 0;
    virtual void separator_reordering() = 0;

    virtual SpMat_t* matrix() = 0;
    virtual std::unique_ptr<SpMat_t> matrix_nonzero_diag() = 0;
    virtual Reord_t* reordering() = 0;
    virtual Tree_t* tree() = 0;
    virtual const SpMat_t* matrix() const = 0;
    virtual const Reord_t* reordering() const = 0;
    virtual const Tree_t* tree() const = 0;

    virtual void perf_counters_start();
    virtual void perf_counters_stop(const std::string& s);

    virtual void synchronize() {}
    virtual void communicate_ordering() {}
    virtual double max_peak_memory() const
    { return double(params::peak_memory); }
    virtual double min_peak_memory() const
    { return double(params::peak_memory); }

    void papi_initialize();
    long long dense_factor_nonzeros() const;
    void print_solve_stats(TaskTimer& t) const;

    virtual void reduce_flop_counters() const {}
    void print_flop_breakdown_HSS() const;
    void print_flop_breakdown_HODLR() const;
    void flop_breakdown_reset() const;

    void print_wrong_sparsity_error();

    SPOptions<scalar_t> opts_;
    bool is_root_;

    MatchingData<scalar_t,integer_t> matching_;
    Equilibration<scalar_t> equil_;

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
    long long int m0_ = 0, mtot_ = 0, mmin_ = 0, mmax_ = 0;
    long long int ptot_ = 0, pmin_ = 0, pmax_ = 0;
    long long int dm0_ = 0, dmtot_ = 0, dmmin_ = 0, dmmax_ = 0;
    long long int dptot_ = 0, dpmin_ = 0, dpmax_ = 0;
#endif

  private:
    ReturnCode reorder_internal(const int* p, int base,
                                int nx, int ny, int nz,
                                int components, int width);

    virtual
    ReturnCode solve_internal(const scalar_t* b, scalar_t* x,
                              bool use_initial_guess=false) = 0;
    virtual
    ReturnCode solve_internal(const DenseM_t& b, DenseM_t& x,
                              bool use_initial_guess=false) = 0;

    virtual
    ReturnCode solve_internal(int nrhs, const scalar_t* b, int ldb,
                              scalar_t* x, int ldx,
                              bool use_initial_guess=false);

    virtual void delete_factors_internal() = 0;
  };

  template<typename scalar_t,typename integer_t>
  using StrumpackSparseSolverBase =
    SparseSolverBase<scalar_t,integer_t>;

} //end namespace strumpack

#endif // STRUMPACK_SPARSE_SOLVER_BASE_HPP
