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
 * \file StrumpackOptions.hpp
 * \brief Holds options for the sparse solver.
 */
#ifndef SPOPTIONS_HPP
#define SPOPTIONS_HPP

#include <limits>

#include "dense/BLASLAPACKWrapper.hpp"
#include "HSS/HSSOptions.hpp"
#include "BLR/BLROptions.hpp"
#include "HODLR/HODLROptions.hpp"

namespace strumpack {

  /**
   * Enumeration of possible sparse fill-reducing orderings.
   * \ingroup Enumerations
   */
  enum class ReorderingStrategy {
    NATURAL,    /*!< Do not reorder the system                      */
    METIS,      /*!< Use Metis nested-dissection reordering         */
    PARMETIS,   /*!< Use ParMetis nested-dissection reordering      */
    SCOTCH,     /*!< Use Scotch nested-dissection reordering        */
    PTSCOTCH,   /*!< Use PT-Scotch nested-dissection reordering     */
    RCM,        /*!< Use RCM reordering                             */
    GEOMETRIC   /*!< A simple geometric nested dissection code that
                  only works for regular meshes. (see Sp::reorder)  */
  };

  /**
   * Return a string with the name of the reordering method.
   */
  std::string get_name(ReorderingStrategy method);

  /**
   * Check whether or not the reordering needs to be run in parallel.
   */
  bool is_parallel(ReorderingStrategy method);

  /**
   * Enumeration of rank-structured data formats, which can be used
   * for compression within the sparse solver.
   * \ingroup Enumerations
   */
  enum class CompressionType {
    NONE,     /*!< No compression, purely direct solver  */
    HSS,      /*!< HSS compression of frontal matrices   */
    BLR,      /*!< Block low-rank compression of fronts  */
    HODLR,    /*!< Hierarchically Off-diagonal Low-Rank
                   compression of frontal matrices       */
    LOSSLESS, /*!< Lossless cmpresssion                  */
    LOSSY     /*!< Lossy cmpresssion                     */
  };

  /**
   * Return a name/string for the CompressionType.
   */
  std::string get_name(CompressionType comp);


  /**
   * Enumeration of possible matching algorithms, used for permutation
   * of the sparse matrix to improve stability.
   * \ingroup Enumerations
   */
  enum class MatchingJob {
    NONE,                         /*!< Don't do anything                   */
    MAX_CARDINALITY,              /*!< Maximum cardinality                 */
    MAX_SMALLEST_DIAGONAL,        /*!< Maximum smallest diagonal value     */
    MAX_SMALLEST_DIAGONAL_2,      /*!< Same as MAX_SMALLEST_DIAGONAL,
                                    but different algorithm                */
    MAX_DIAGONAL_SUM,             /*!< Maximum sum of diagonal values      */
    MAX_DIAGONAL_PRODUCT_SCALING, /*!< Maximum product of diagonal values
                                    and row and column scaling             */
    COMBBLAS                      /*!< Use AWPM from CombBLAS              */
  };

  /**
   * Convert a job number to a MatchingJob enum type.
   */
  MatchingJob get_matching(int job);

  /**
   * Convert a MatchingJob enum type to a job number. Prefer to use
   * the MachingJob enum instead of the job number.
   */
  int get_matching(MatchingJob job);

  /**
   * Return a string describing the matching algorithm.
   */
  std::string get_description(MatchingJob job);


  /**
   * Type of Gram-Schmidt orthogonalization used in GMRes.
   * \ingroup Enumerations
   */
  enum class GramSchmidtType {
    CLASSICAL,   /*!< Classical Gram-Schmidt is faster, more scalable.   */
    MODIFIED     /*!< Modified Gram-Schmidt is slower, but stable.       */
  };

  /**
   * Type of outer iterative (Krylov) solver.
   * \ingroup Enumerations
   */
  enum class KrylovSolver {
    AUTO,           /*!< Use iterative refinement if no compression is
                      used, otherwise use GMRes.                            */
    DIRECT,         /*!< No outer iterative solver, just a single
                      application of the multifrontal solver.               */
    REFINE,         /*!< Iterative refinement.                              */
    PREC_GMRES,     /*!< Preconditioned GMRes. The preconditioner is the
                      (approx) multifrontal solver.                         */
    GMRES,          /*!< UN-preconditioned GMRes. (for testing mainly)      */
    PREC_BICGSTAB,  /*!< Preconditioned BiCGStab. The preconditioner is the
                      (approx) multifrontal solver.                         */
    BICGSTAB        /*!< UN-preconditioned BiCGStab. (for testing mainly)   */
  };

  /**
   * Default relative tolerance used when solving a linear system. For
   * iterative solvers such as GMRES and BiCGStab, this is the
   * relative residual tolerance. Exact value depends on the floating
   * point type.
   */
  template<typename real_t> inline real_t default_rel_tol()
  { return real_t(1.e-6); }
  /**
   * Default absolute tolerance used when solving a linear system. For
   * iterative solvers such as GMRES and BiCGStab, this is the
   * residual tolerance. Exact value depends on the floating point
   * type.
   */
  template<typename real_t> inline real_t default_abs_tol()
  { return real_t(1.e-10); }
  template<> inline float default_rel_tol() { return 1.e-4; }
  template<> inline float default_abs_tol() { return 1.e-6; }

  inline int default_cuda_cutoff() { return 500; }
  inline int default_cuda_streams() { return 10; }

  /**
   * \class SPOptions
   * \brief Options for the sparse solver.
   *
   * This sparse solver object also stores an object with HSS options
   * (HSS_options), one with BLR options (BLR_options) and one with
   * HODLR options (HODLR_options), since HSS, BLR and HODLR
   * compression can be used in the sparse solver.
   *
   * Running with -h or --help will print a list of options when the
   * set_from_command_line() routine is called.
   *
   * \tparam scalar_t can be float, double, std::complex<float> or
   * std::complex<double>, should be the same as used for the
   * StrumpackSparseSolver object
   */
  template<typename scalar_t> class SPOptions {
  public:
    /**
     * real_t is the real type corresponding to the (possibly
     * complex) scalar_t template parameter
     */
    using real_t = typename RealType<scalar_t>::value_type;

    /**
     * Default constructor, initializing all options to their default
     * values.
     */
    SPOptions() { _hss_opts.set_verbose(false); }

    /**
     * Constructor, initializing all options to their default
     * values. This will store a copy of *argv[]. It will not yet
     * parse the options!! The options are only parsed when calling
     * set_from_command_line(). This allows the user of this class to
     * set certain options using the member functions of this class,
     * and then overwrite those with command line arguments later.
     *
     * \param argc number of arguments (number of char* in argv)
     * \param argv inout from the command line with list of options
     */
    SPOptions(int argc, char* argv[]) : _argc(argc), _argv(argv) {
      _hss_opts.set_verbose(false);
      _blr_opts.set_verbose(false);
      _hodlr_opts.set_verbose(false);
    }

    /**
     * Set verbose to true/false, ie, allow the sparse solver to print
     * out progress information, statistics on time, flops, memory
     * usage etc. to cout. Warnings will go to cerr regardless of the
     * verbose options set here.
     */
    void set_verbose(bool verbose) { _verbose = verbose; }

    /**
     * Set the maximum number of iterations to use in any of the
     * iterative solvers.
     */
    void set_maxit(int maxit) { assert(maxit >= 1); _maxit = maxit; }

    /**
     * Set the relative tolerance to use in the iterative
     * solvers. This will typically be the relative residual decrease.
     *
     * \param rel_tol relative tolerance
     */
    void set_rel_tol(real_t rel_tol) {
      assert(rel_tol <= real_t(1.) && rel_tol >= real_t(0.));
      _rel_tol = rel_tol;
    }

    /**
     * Set the absolute tolerance to use in the iterative
     * solvers.
     *
     * \param abs_tol absolute tolerance
     */
    void set_abs_tol(real_t abs_tol) {
      assert(abs_tol >= real_t(0.));
      _abs_tol = abs_tol;
    }

    /**
     * Select a Krylov outer solver. Note that the versions GMRES and
     * BICGSTAB are not preconditioned! Use PREC_GMRES and
     * PREC_BICGSTAB instead, as they will use STRUMPACK's sparse
     * solver (possibly with compression) as a preconditioner. Setting
     * this to DIRECT will not use any iterative solver, and instead
     * only perform a solve with the (incomplete/compressed) LU
     * factorization.
     *
     * \param s outer, iterative solver to use
     * \see set_compression()
     */
    void set_Krylov_solver(KrylovSolver s) { _Krylov_solver = s; }

    /**
     * Set the GMRES restart length
     *
     * \param m GMRES restart, should be > 0 and not too crazy big
     */
    void set_gmres_restart(int m) { assert(m >= 1); _gmres_restart = m; }

    /**
     * Set the type of Gram-Schmidt orthogonalization to use in GMRES
     *
     * \param t Gram-Schmidt type to use in GMRES
     */
    void set_GramSchmidt_type(GramSchmidtType t) { _Gram_Schmidt_type = t; }

    /**
     * Set the sparse fill-reducing reordering. This can greatly
     * affect the memory usage and factorization time. However, note
     * that most reordering routines are provided by third party
     * libraries, such as Metis and Scotch. In order to use those,
     * STRUMPACK needs to be configured with support for those
     * libraries. Some reorderings only work when using the
     * distributed memory solvers (StrumpackSparseSolverMPI or
     * StrumpackSparseSolverMPIDist).
     *
     * \param m fill reducing reordering
     */
    void set_reordering_method(ReorderingStrategy m) { _reordering_method = m; }

    /**
     * Set the parameter used in nested dissection to determine when
     * to stop the nested-dissection recursion. Larger values may lead
     * to less nodes in the separator tree, which eliminates some
     * overhead, but typically leads to more fill.
     */
    void set_nd_param(int nd_param)
    { assert(nd_param>=0); _nd_param = nd_param; }

    /**
     * Set the mesh dimensions. This is only useful when the sparse
     * matrix was generated by a stencil on a regular 1d, 2d or 3d
     * mesh. The stencil can be multiple gridpoints wide, and there
     * can be multiple degrees of freedom per gridpoint. The degrees
     * of freedom for a single gridpoint should be ordered
     * consecutively. When the dimensions of the grid are specified,
     * along with the width of the stencil and the number of degrees
     * of freedom per gridpoint, one can use the GEOMETRIC option for
     * the fill-reducing reordering.
     *
     * \param nx dimension of the grid along the first axis
     * \param ny dimension of the grid along the second axis, should
     * not be specified for a 1d mesh
     * \param nz dimension of the grid along the third axis, should
     * not be specified for a 2d mesh
     * \see set_components(), set_separator_width(),
     *  ReorderingStrategy::GEOMETRIC, set_reordering_method()
     */
    void set_dimensions(int nx, int ny=1, int nz=1) {
      assert(nx>=1 && ny>=1 && nz>=1);
      _nx = nx; _ny = ny; _nz = nz;
    }

    /**
     * Set the mesh dimension along the x-axis. This is only useful
     * when the sparse matrix was generated by a stencil on a regular
     * 1d, 2d or 3d mesh.
     *
     * \see set_dimensions(), set_ny(), set_nz(), set_components(),
     *  set_separator_width(), ReorderingStrategy::GEOMETRIC,
     *  set_reordering_method()
     */
    void set_nx(int nx) {assert(nx>=1); _nx = nx; }

    /**
     * Set the mesh dimension along the y-axis. This is only useful
     * when the sparse matrix was generated by a stencil on a regular
     * 1d, 2d or 3d mesh. For a 1d mesh this should not be specified,
     * defaults to 1.
     *
     * \see set_dimensions(), set_nx(), set_nz(), set_components(),
     *  set_separator_width(), ReorderingStrategy::GEOMETRIC,
     *  set_reordering_method()
     */
    void set_ny(int ny) {assert(ny>=1); _ny = ny; }

    /**
     * Set the mesh dimension along the z-axis. This is only useful
     * when the sparse matrix was generated by a stencil on a regular
     * 1d, 2d or 3d mesh. For a 2d mesh this should not be specified,
     * defaults to 1.
     *
     * \see set_dimensions(), set_nx(), set_ny(), set_components(),
     *  set_separator_width(), ReorderingStrategy::GEOMETRIC,
     *  set_reordering_method()
     */
    void set_nz(int nz) {assert(nz>=1); _nz = nz; }

    /**
     * Set the number of components per gridpoint. This is only useful
     * when the sparse matrix was generated by a stencil on a regular
     * 1d, 2d or 3d mesh. The degrees of freedom for a single
     * gridpoint should be ordered consecutively.
     *
     * \param components number of components per gridpoint
     * \see set_dimensions(), set_separator_width(),
     * ReorderingStrategy::GEOMETRIC, set_reordering_method()
     */
    void set_components(int components)
    { assert(components>=1); _components = components; }

    /**
     * Set the width of the separator, which depends on the width of
     * the stencil. For instance for a 1d 3-point stencil, the
     * separator is just a single point, for a 2d 5-point stencil the
     * separator is a single line and for a 3d stencil, the separator
     * is a single plane, and hence, for all these cases, the
     * separator width is 1. Wider stencils need a wider separator,
     * for instance a stencil with 5 points in each dimension needs a
     * separator width of 3. This is only useful when the sparse
     * matrix was generated by a stencil on a regular 1d, 2d or 3d
     * mesh. The degrees of freedom for a single gridpoint should be
     * ordered consecutively.
     *
     * \param width width of the separator
     * \see set_dimensions(), set_components(),
     * ReorderingStrategy::GEOMETRIC, set_reordering_method()
     */
    void set_separator_width(int width)
    { assert(width>=1); _separator_width = width; }

    /**
     * Enable use of the routine METIS_NodeNDP, instead of
     * METIS_NodeND. METIS_NodeNDP is a non-documented Metis routine
     * to perform nested dissection which (unlike METIS_NodeND) also
     * return the separator tree.
     *
     * \see disable_METIS_NodeNDP(), enable_METIS_NodeND(),
     * disable_METIS_NodeND()
     */
    void enable_METIS_NodeNDP() { _use_METIS_NodeNDP = true; }

    /**
     * Disable use of the routine METIS_NodeNDP, and instead use
     * METIS_NodeND. METIS_NodeNDP is a non-documented Metis routine
     * to perform nested dissection which (unlike METIS_NodeND) also
     * return the separator tree.
     *
     * \see enable_METIS_NodeNDP(), enable_METIS_NodeND(),
     * disable_METIS_NodeND()
     */
    void disable_METIS_NodeNDP() { _use_METIS_NodeNDP = false; }


    /**
     * Use the routine METIS_NodeND instead of the undocumented
     * routine METIS_NodeNDP.
     *
     * \see enable_METIS_NodeNDP(), disable_METIS_NodeNDP(),
     * disable_METIS_NodeND()
     */
    void enable_METIS_NodeND() { _use_METIS_NodeNDP = false; }

    /**
     * Do not use the routine METIS_NodeND, but instead use the
     * undocumented routine METIS_NodeNDP.
     *
     * \see enable_METIS_NodeNDP(), disable_METIS_NodeNDP(),
     * enable_METIS_NodeND()
     */
    void disable_METIS_NodeND() { _use_METIS_NodeNDP = true; }

    /**
     * Use the SYMQAMD routine (provided by the MUMPS folks) to
     * construct the supernodal tree from the elimination tree. In
     * some cases this can reduce the amount of fill.
     *
     * \see disable_MUMPS_SYMQAMD(), enable_agg_amalg(), disable_agg_amalg()
     */
    void enable_MUMPS_SYMQAMD() { _use_MUMPS_SYMQAMD = true; }

    /**
     * Do not use the SYMQAMD routine (provided by the MUMPS folks) to
     * construct the supernodal tree from the elimination tree. In
     * some cases SYMQAMD can reduce the amount of fill. Use
     * STRUMPACK's own routine instead.
     *
     * \see enable_MUMPS_SYMQAMD(), enable_agg_amalg(), disable_agg_amalg()
     */
    void disable_MUMPS_SYMQAMD() { _use_MUMPS_SYMQAMD = false; }

    /**
     * When using MUMPS_SYMQAMD, enable aggressive amalgamation of
     * nodes into supernodes. This is only used when MUMPS_SYMQAMD is
     * enabled.
     *
     * \see disable_agg_amalg(), enable_MUMPS_SYMQAMD(),
     * disable_MUMPS_SYMQAMD()
     */
    void enable_agg_amalg() { _use_agg_amalg = true; }

    /**
     * Disbale aggressive amalgamation of nodes into supernodes inside
     * MUMPS_SYMQMAD. This is only relevant when MUMPS_SYMQAMD is
     * enabled.
     *
     * \see enable_agg_amalg(), enable_MUMPS_SYMQAMD(),
     * disable_MUMPS_SYMQAMD()
     */
    void disable_agg_amalg() { _use_agg_amalg = false; }

    /**
     * Specify the job type for the column ordering for
     * stability. This ordering is computed using a maximum matching
     * algorithm, to try to get nonzero (as large as possible) values
     * on the diagonal to improve numerica stability. Possibly, this
     * can also scale the rows and columns of the matrix.
     *
     * \param job type of matching to perform
     */
    void set_matching(MatchingJob job) { _matching_job = job; }

    /**
     * Log the assembly tree to a file. __Currently not supported.__
     */
    void enable_assembly_tree_log() { _log_assembly_tree = true; }

    /**
     * Do not log the assembly tree to a file. __Logging of the tree
     * is currently not supported.__
     */
    void disable_assembly_tree_log() { _log_assembly_tree = false; }

    /**
     * Set the type of rank-structured compression to use.
     *
     * \param c compression type
     *
     * \see set_compression_min_sep_size(),
     * set_compression_min_front_size()
     */
    void set_compression(CompressionType c) { _comp = c; }

    /**
     * Set the minimum size of the top left part of frontal matrices
     * (dense submatrices of the sparse triangular factors), ie, the
     * part corresponding to the separators, for which to use
     * compression (when HSS/BLR/HODLR/LOSSY compression has been
     * enabled by the user. Most fronts will be quite small, and will
     * not have benefit from compression.
     *
     * \see set_compression(), set_compression_min_front_size()
     */
    void set_compression_min_sep_size(int s) {
      assert(s >= 0);
      _hss_min_sep_size = s;
      _blr_min_sep_size = s;
      _hodlr_min_sep_size = s;
      _lossy_min_sep_size = s;
    }

    /**
     * Set the minimum size of frontal matrices (dense submatrices of
     * the sparse triangular factors), for which to use compression
     * (when HSS/BLR/HODLR/LOSSY compression has been enabled by the
     * user. Most fronts will be quite small, and will not have
     * benefit from compression.
     *
     * \see set_compression(), set_compression_min_front_size()
     */
    void set_compression_min_front_size(int s) {
      assert(s >= 0);
      _hss_min_front_size = s;
      _blr_min_front_size = s;
      _hodlr_min_front_size = s;
      _lossy_min_front_size = s;
    }

    /**
     * Set the leaf size used by any of the rank-structured formats.
     *
     * \see HSS_options(), BLR_options(), HODLR_options()
     */
    void set_compression_leaf_size(int s) {
      _hss_opts.set_leaf_size(s);
      _blr_opts.set_leaf_size(s);
      _hodlr_opts.set_leaf_size(s);
    }

    /**
     * This is used in the reordering of the separators, which is
     * useful when using compression (HSS, BLR) to reduce the ranks
     * and to define the block-sizes in the rank structured matrix
     * format. A value of 1 means that additional edges are introduced
     * between nodes in the separators, edges corresponding to 2-step
     * connections in the graph of the original matrix, before passing
     * the graph to a graph partitioner. These additional edges are
     * useful beause sometimes the separator graph can become
     * unconnected (although the points are connected in the original
     * graph). A value of 0 means not to add these additional edges.
     *
     * You should probably leave this at 1, unless separator
     * reordering takes a long time, in which case you can try 0.
     */
    void set_separator_ordering_level(int l)
    { assert(l >= 0); _sep_order_level = l; }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    void enable_indirect_sampling() { _indirect_sampling = true; }
    void disable_indirect_sampling() { _indirect_sampling = false; }
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /**
     * Enable replacing of small pivot values with a larger
     * value. This can prevent to numerical factorization to fail
     * completely, but the resulting solve might be inaccurate and
     * hence require more iterations. If the outer iterative solver
     * fails to converge, you can switch from iterative refinement
     * (which is used with the AUTO solver strategy if compression is
     * disabled), to PGMRES (preconditioned!).
     *
     * If you encounter numerical issues during the factorization
     * (such as small pivots, failure in LU factorization), you can
     * also try a different matching
     * (MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING is the most robust
     * option).
     *
     * \see disable_replace_tiny_pivots(), set_matching()
     */
    void enable_replace_tiny_pivots() { _replace_tiny_pivots = true; }

    /**
     * Disable replacement of tiny pivots.
     *
     * \see enable_replace_tiny_pivots()
     */
    void disable_replace_tiny_pivots() { _replace_tiny_pivots = false; }

    /**
     * Dump the root front to a set of files, one for each rank. This
     * will only have affect when running with more than one MPI rank,
     * and without compression.
     */
    void set_write_root_front(bool b)  { _write_root_front = b; }

    /**
     * Enable off-loading to the GPU. This only works when STRUMPACK
     * was configured with GPU support (through CUDA, MAGMA or SLATE)
     */
    void enable_gpu()  { use_gpu_ = true; }

    /**
     * Disable GPU off-loading.
     */
    void disable_gpu()  { use_gpu_ = false; }

    /**
     * Set the minimum dense matrix size for dense matrix operations
     * to be off-loaded to the GPU.
     */
    void set_cuda_cutoff(int c) { cuda_cutoff_ = c; }

    /**
     * Set the number of (CUDA) streams to be used in the code.
     */
    void set_cuda_streams(int s) { cuda_streams_ = s; }

    /**
     * Set the precision for lossy compression.
     */
    void set_lossy_precision(int p) { _lossy_precision = p; }

    /**
     * Print statistics, about ranks, memory etc, for the root front
     * only.
     */
    void set_print_root_front_stats(bool b)  { _print_root_front_stats = b; }

    /**
     * Check if verbose output is enabled.
     * \see set_verbose()
     */
    bool verbose() const { return _verbose; }

    /**
     * Get the maximum number of allowed iterative solver iterations.
     * \see set_maxit()
     */
    int maxit() const { return _maxit; }

    /**
     * Get the relative tolerance to be used in the iterative solver.
     * \see set_rel_tol()
     */
    real_t rel_tol() const { return _rel_tol; }

    /**
     * Get the absolute tolerance to be used in the iterative solver.
     * \see set_abs_tol()
     */
    real_t abs_tol() const { return _abs_tol; }

    /**
     * Get the type of iterative solver to be used as outer solver.
     * \see set_Krylov_solver()
     */
    KrylovSolver Krylov_solver() const { return _Krylov_solver; }

    /**
     * Get the GMRES restart length.
     * \see set_gmres_restart()
     */
    int gmres_restart() const { return _gmres_restart; }

    /**
     * Get the Gram-Schmidth orthogonalization type used in GMRES.
     * \see set_GramSchmidth_type()
     */
    GramSchmidtType GramSchmidt_type() const { return _Gram_Schmidt_type; }

    /**
     * Get the currently set fill reducing reordering method.
     * \see set_reordering_method()
     */
    ReorderingStrategy reordering_method() const { return _reordering_method; }

    /**
     * Return the nested-dissection recursion ending parameter.
     * \see set_nd_param()
     */
    int nd_param() const { return _nd_param; }

    /**
     * Get the specified nx mesh dimension.
     * \see set_nx()
     */
    int nx() const { return _nx; }

    /**
     * Get the specified ny mesh dimension.
     * \see set_ny()
     */
    int ny() const { return _ny; }

    /**
     * Get the specified nz mesh dimension.
     * \see set_nz()
     */
    int nz() const { return _nz; }

    /**
     * Get the currently specified number of components (DoF's) per
     * mesh point.
     * \see set_components()
     */
    int components() const { return _components; }

    /**
     * Get the currently specified width of a separator.
     * \see set_separator_width()
     */
    int separator_width() const { return _separator_width; }

    /**
     * Is use of METIS_NodeNDP enabled? (instead of METIS_NodeND)
     * \see enable_METIS_NodeNDP()
     */
    bool use_METIS_NodeNDP() const { return _use_METIS_NodeNDP; }

    /**
     * Is use of METIS_NodeND enabled? (instead of METIS_NodeNDP)
     * \see enable_METIS_NodeND()
     */
    bool use_METIS_NodeND() const { return !_use_METIS_NodeNDP; }

    /**
     * Is MUMPS_SYMQAMD enabled?
     * \see enable_MUMPS_SYMQAMD()
     */
    bool use_MUMPS_SYMQAMD() const { return _use_MUMPS_SYMQAMD; }

    /**
     * Is aggressive amalgamation enabled? (only used when
     * MUMPS_SYMQAMD is enabled)
     * \see enable_agg_amalg(), enable_MUMPS_SYMQAMD()
     */
    bool use_agg_amalg() const { return _use_agg_amalg; }

    /**
     * Get the matching job to use for numerical stability reordering.
     * \see set_matching()
     */
    MatchingJob matching() const { return _matching_job; }

    /**
     * Should we log the assembly tree?
     * __Currently not supported.__
     */
    bool log_assembly_tree() const { return _log_assembly_tree; }

    /**
     * Get the type of compression to use.
     */
    CompressionType compression() const { return _comp; }

    /**
     * Get the minimum size of a separator to enable compression. This
     * will depend on which type of compression is selected.
     *
     * \see set_compression(), set_compression_min_sep_size(),
     * compression_min_front_size()
     */
    int compression_min_sep_size() const {
      switch (_comp) {
      case CompressionType::HSS:
        return _hss_min_sep_size;
      case CompressionType::BLR:
        return _blr_min_sep_size;
      case CompressionType::HODLR:
        return _hodlr_min_sep_size;
      case CompressionType::LOSSY:
      case CompressionType::LOSSLESS:
        return _lossy_min_sep_size;
      case CompressionType::NONE:
      default:
        return std::numeric_limits<int>::max();
      }
    }

    /**
     * Get the minimum size of a front to enable compression. This
     * will depend on which type of compression is selected.
     *
     * \see set_compression(), set_compression_min_sep_size(),
     * compression_min_front_size()
     */
    int compression_min_front_size() const {
      switch (_comp) {
      case CompressionType::HSS:
        return _hss_min_front_size;
      case CompressionType::BLR:
        return _blr_min_front_size;
      case CompressionType::HODLR:
        return _hodlr_min_front_size;
      case CompressionType::LOSSY:
      case CompressionType::LOSSLESS:
        return _lossy_min_front_size;
      case CompressionType::NONE:
      default:
        return std::numeric_limits<int>::max();
      }
    }

    /**
     * Get the leaf size used in the rank-structured format used for
     * compression. This will depend on which type of compression is
     * selected.
     *
     * \see set_compression(), set_compression_leaf_size(),
     * compression_min_sep_size()
     */
    int compression_leaf_size() const {
      switch (_comp) {
      case CompressionType::HSS:
        return _hss_opts.leaf_size();
      case CompressionType::BLR:
        return _blr_opts.leaf_size();
      case CompressionType::HODLR:
        return _hodlr_opts.leaf_size();
      case CompressionType::LOSSY:
      case CompressionType::LOSSLESS:
        return 4;
      case CompressionType::NONE:
      default:
        return std::numeric_limits<int>::max();
      }
    }

    /**
     * Get the current value of the number of additional links to
     * include in the separator before reordering.
     * \see set_separator_ordering_level()
     */
    int separator_ordering_level() const { return _sep_order_level; }

    /**
     * Is indirect sampling for HSS construction enabled?
     */
    bool indirect_sampling() const { return _indirect_sampling; }

    /**
     * Check whether replacement of tiny pivots is enabled.
     */
    bool replace_tiny_pivots() const { return _replace_tiny_pivots; }

    /**
     * The root front will be written to a file.
     */
    bool write_root_front() const { return _write_root_front; }

    /**
     * Check wheter or not to use GPU off-loading.
     */
    bool use_gpu() const { return use_gpu_; }

    /**
     * Returns the minimum size of a dense matrix for GPU off-loading.
     */
    int cuda_cutoff() const { return cuda_cutoff_; }

    /**
     * Returns the number of CUDA streams to use.
     */
    int cuda_streams() const { return cuda_streams_; }

    /**
     * Returns the precision for lossy compression.
     */
    int lossy_precision() const { return _lossy_precision; }

    /**
     * Info about the stats of the root front will be printed to
     * std::cout
     */
    bool print_root_front_stats() const { return _print_root_front_stats; }

    /**
     * Get a (const) reference to an object holding various options
     * pertaining to the HSS code, and data structures.
     */
    const HSS::HSSOptions<scalar_t>& HSS_options() const { return _hss_opts; }

    /**
     * Get a reference to an object holding various options pertaining
     * to the HSS code, and data structures.
     */
    HSS::HSSOptions<scalar_t>& HSS_options() { return _hss_opts; }

    /**
     * Get a (const) reference to an object holding various options
     * pertaining to the BLR code, and data structures.
     */
    const BLR::BLROptions<scalar_t>& BLR_options() const { return _blr_opts; }

    /**
     * Get a reference to an object holding various options pertaining
     * to the BLR code, and data structures.
     */
    BLR::BLROptions<scalar_t>& BLR_options() { return _blr_opts; }

    /**
     * Get a (const) reference to an object holding various options
     * pertaining to the HODLR code, and data structures.
     */
    const HODLR::HODLROptions<scalar_t>& HODLR_options() const { return _hodlr_opts; }

    /**
     * Get a reference to an object holding various options pertaining
     * to the HODLR code, and data structures.
     */
    HODLR::HODLROptions<scalar_t>& HODLR_options() { return _hodlr_opts; }

    /**
     * Parse the command line options that were passed to this object
     * in the constructor. Run the code with -h or --help and call
     * this routine to see a list of supported options.
     */
    void set_from_command_line() { set_from_command_line(_argc, _argv); }

    /**
     * Parse command line options. These options will not be
     * modified. The options can also contain HSS, BLR or HODLR
     * specific options, they will be parsed by the HSS::HSSOptions
     * and BLR::BLROptions, HODLR::HODLROptions objects returned by
     * HSS_options(), BLR_options() and HODLR_options()
     * respectively. Run the code with -h or --help and call this
     * routine to see a list of supported options.
     *
     * \param argc number of arguments in the argv array
     * \param argv list of options
     */
    void set_from_command_line(int argc, const char* const* argv);

    /**
     * Print an overview of all supported options. Not including any
     * HSS/BLR specific options.
     */
    void describe_options() const;

  private:
    bool _verbose = true;
    /** Krylov solver options */
    int _maxit = 5000;
    real_t _rel_tol = default_rel_tol<real_t>();
    real_t _abs_tol = default_abs_tol<real_t>();
    KrylovSolver _Krylov_solver = KrylovSolver::AUTO;
    int _gmres_restart = 30;
    GramSchmidtType _Gram_Schmidt_type = GramSchmidtType::MODIFIED;
    /** Reordering options */
    ReorderingStrategy _reordering_method = ReorderingStrategy::METIS;
    int _nd_param = 8;
    int _nx = 1;
    int _ny = 1;
    int _nz = 1;
    int _components = 1;
    int _separator_width = 1;
    bool _use_METIS_NodeNDP = false;
    bool _use_MUMPS_SYMQAMD = false;
    bool _use_agg_amalg = false;
    MatchingJob _matching_job = MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING;
    bool _log_assembly_tree = false;
    bool _replace_tiny_pivots = false;
    bool _write_root_front = false;
    bool _print_root_front_stats = false;

    /** GPU options */
    bool use_gpu_ = true;
    int cuda_cutoff_ = default_cuda_cutoff();
    int cuda_streams_ = default_cuda_streams();

    /** compression options */
    CompressionType _comp = CompressionType::NONE;

    /** HSS options */
    int _hss_min_front_size = 1000;
    int _hss_min_sep_size = 256;
    int _sep_order_level = 1;
    bool _indirect_sampling = false;
    HSS::HSSOptions<scalar_t> _hss_opts;

    /** BLR options */
    BLR::BLROptions<scalar_t> _blr_opts;
    int _blr_min_front_size = 1000;
    int _blr_min_sep_size = 256;

    /** HODLR options */
    HODLR::HODLROptions<scalar_t> _hodlr_opts;
    int _hodlr_min_front_size = 1000;
    int _hodlr_min_sep_size = 256;

    /** LOSSY/LOSSLESS options */
    int _lossy_min_front_size = 16;
    int _lossy_min_sep_size = 8;
    int _lossy_precision = 16;

    int _argc = 0;
    char** _argv = nullptr;
  };

} // end namespace strumpack

#endif // SPOPTIONS_HPP
