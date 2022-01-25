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
   * Enumeration of strategies for proportional mapping of the
   * multifrontal tree.
   * \ingroup Enumerations
   */
  enum class ProportionalMapping {
    FLOPS,          /*!< Balance flops, optimze runtime                 */
    FACTOR_MEMORY,  /*!< Balance final memory for LU factors            */
    PEAK_MEMORY     /*!< Balance peak memory usage during factorization */
  };

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
    NONE,      /*!< No compression, purely direct solver  */
    HSS,       /*!< HSS compression of frontal matrices   */
    BLR,       /*!< Block low-rank compression of fronts  */
    HODLR,     /*!< Hierarchically Off-diagonal Low-Rank
                    compression of frontal matrices       */
    BLR_HODLR, /*!< Block low-rank compression of medium
                    fronts and Hierarchically Off-diagonal
                    Low-Rank compression of large fronts  */
    ZFP_BLR_HODLR, /*!< ZFP compression for small fronts,
                    Block low-rank compression of medium
                    fronts and Hierarchically Off-diagonal
                    Low-Rank compression of large fronts  */
    LOSSLESS,  /*!< Lossless cmpresssion                  */
    LOSSY      /*!< Lossy cmpresssion                     */
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

  enum class EquilibrationType : char
    { NONE='N', ROW='R', COLUMN='C', BOTH='B' };


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

  inline int default_gpu_streams() { return 4; }

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
    SPOptions() { hss_opts_.set_verbose(false); }

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
    SPOptions(int argc, const char* const argv[]) : argc_(argc), argv_(argv) {
      hss_opts_.set_verbose(false);
      blr_opts_.set_verbose(false);
      hodlr_opts_.set_verbose(false);
    }

    /**
     * Set verbose to true/false, ie, allow the sparse solver to print
     * out progress information, statistics on time, flops, memory
     * usage etc. to cout. Warnings will go to cerr regardless of the
     * verbose options set here.
     */
    void set_verbose(bool verbose) { verbose_ = verbose; }

    /**
     * Set the maximum number of iterations to use in any of the
     * iterative solvers.
     */
    void set_maxit(int maxit) { assert(maxit >= 1); maxit_ = maxit; }

    /**
     * Set the relative tolerance to use in the iterative
     * solvers. This will typically be the relative residual decrease.
     *
     * \param rel_tol relative tolerance
     */
    void set_rel_tol(real_t rtol) {
      assert(rtol <= real_t(1.) && rtol >= real_t(0.));
      rel_tol_ = rtol;
    }

    /**
     * Set the absolute tolerance to use in the iterative
     * solvers.
     *
     * \param abs_tol absolute tolerance
     */
    void set_abs_tol(real_t atol) {
      assert(atol >= real_t(0.));
      abs_tol_ = atol;
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
    void set_Krylov_solver(KrylovSolver s) { Krylov_solver_ = s; }

    /**
     * Set the GMRES restart length
     *
     * \param m GMRES restart, should be > 0 and not too crazy big
     */
    void set_gmres_restart(int m) { assert(m >= 1); gmres_restart_ = m; }

    /**
     * Set the type of Gram-Schmidt orthogonalization to use in GMRES
     *
     * \param t Gram-Schmidt type to use in GMRES
     */
    void set_GramSchmidt_type(GramSchmidtType t) { Gram_Schmidt_type_ = t; }

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
    void set_reordering_method(ReorderingStrategy m) { reordering_method_ = m; }

    /**
     * Set the parameter used in nested dissection to determine when
     * to stop the nested-dissection recursion. Larger values may lead
     * to less nodes in the separator tree, which eliminates some
     * overhead, but typically leads to more fill.
     */
    void set_nd_param(int nd_param)
    { assert(nd_param>=0); nd_param_ = nd_param; }

    /**
     * Set the number of levels in nested-dissection for which the
     * separators are parallel instead of perpedicular to the longest
     * direction. This is to reduce the ranks in the F12/F21 blocks
     * when using compression.
     */
    void set_nd_planar_levels(int nd_planar_levels)
    { assert(nd_planar_levels>=0); nd_planar_levels_ = nd_planar_levels; }

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
      nx_ = nx; ny_ = ny; nz_ = nz;
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
    void set_nx(int nx) {assert(nx>=1); nx_ = nx; }

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
    void set_ny(int ny) {assert(ny>=1); ny_ = ny; }

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
    void set_nz(int nz) { assert(nz>=1); nz_ = nz; }

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
    { assert(components>=1); components_ = components; }

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
    { assert(width>=1); separator_width_ = width; }

    /**
     * Enable use of the routine METIS_NodeNDP, instead of
     * METIS_NodeND. METIS_NodeNDP is a non-documented Metis routine
     * to perform nested dissection which (unlike METIS_NodeND) also
     * return the separator tree.
     *
     * \see disable_METIS_NodeNDP(), enable_METIS_NodeND(),
     * disable_METIS_NodeND()
     */
    void enable_METIS_NodeNDP() { use_METIS_NodeNDP_ = true; }

    /**
     * Disable use of the routine METIS_NodeNDP, and instead use
     * METIS_NodeND. METIS_NodeNDP is a non-documented Metis routine
     * to perform nested dissection which (unlike METIS_NodeND) also
     * return the separator tree.
     *
     * \see enable_METIS_NodeNDP(), enable_METIS_NodeND(),
     * disable_METIS_NodeND()
     */
    void disable_METIS_NodeNDP() { use_METIS_NodeNDP_ = false; }


    /**
     * Use the routine METIS_NodeND instead of the undocumented
     * routine METIS_NodeNDP.
     *
     * \see enable_METIS_NodeNDP(), disable_METIS_NodeNDP(),
     * disable_METIS_NodeND()
     */
    void enable_METIS_NodeND() { use_METIS_NodeNDP_ = false; }

    /**
     * Do not use the routine METIS_NodeND, but instead use the
     * undocumented routine METIS_NodeNDP.
     *
     * \see enable_METIS_NodeNDP(), disable_METIS_NodeNDP(),
     * enable_METIS_NodeND()
     */
    void disable_METIS_NodeND() { use_METIS_NodeNDP_ = true; }

    /**
     * Use the SYMQAMD routine (provided by the MUMPS folks) to
     * construct the supernodal tree from the elimination tree. In
     * some cases this can reduce the amount of fill.
     *
     * \see disable_MUMPS_SYMQAMD(), enable_agg_amalg(), disable_agg_amalg()
     */
    void enable_MUMPS_SYMQAMD() { use_MUMPS_SYMQAMD_ = true; }

    /**
     * Do not use the SYMQAMD routine (provided by the MUMPS folks) to
     * construct the supernodal tree from the elimination tree. In
     * some cases SYMQAMD can reduce the amount of fill. Use
     * STRUMPACK's own routine instead.
     *
     * \see enable_MUMPS_SYMQAMD(), enable_agg_amalg(), disable_agg_amalg()
     */
    void disable_MUMPS_SYMQAMD() { use_MUMPS_SYMQAMD_ = false; }

    /**
     * When using MUMPS_SYMQAMD, enable aggressive amalgamation of
     * nodes into supernodes. This is only used when MUMPS_SYMQAMD is
     * enabled.
     *
     * \see disable_agg_amalg(), enable_MUMPS_SYMQAMD(),
     * disable_MUMPS_SYMQAMD()
     */
    void enable_agg_amalg() { use_agg_amalg_ = true; }

    /**
     * Disbale aggressive amalgamation of nodes into supernodes inside
     * MUMPS_SYMQMAD. This is only relevant when MUMPS_SYMQAMD is
     * enabled.
     *
     * \see enable_agg_amalg(), enable_MUMPS_SYMQAMD(),
     * disable_MUMPS_SYMQAMD()
     */
    void disable_agg_amalg() { use_agg_amalg_ = false; }

    /**
     * Specify the job type for the column ordering for
     * stability. This ordering is computed using a maximum matching
     * algorithm, to try to get nonzero (as large as possible) values
     * on the diagonal to improve numerica stability. Possibly, this
     * can also scale the rows and columns of the matrix.
     *
     * \param job type of matching to perform
     */
    void set_matching(MatchingJob job) { matching_job_ = job; }

    /**
     * Log the assembly tree to a file. __Currently not supported.__
     */
    void enable_assembly_tree_log() { log_assembly_tree_ = true; }

    /**
     * Do not log the assembly tree to a file. __Logging of the tree
     * is currently not supported.__
     */
    void disable_assembly_tree_log() { log_assembly_tree_ = false; }

    /**
     * Set the type of rank-structured compression to use.
     *
     * \param c compression type
     *
     * \see set_compression_min_sep_size(),
     * set_compression_min_front_size()
     */
    void set_compression(CompressionType c) { comp_ = c; }

    /**
     * Set the relative compression tolerance to be used for low-rank
     * compression. This currently affects BLR, HSS, HODLR, HODBF,
     * Butterfly. It does not affect the lossy compression, see
     * set_lossy_precision. The same tolerances can also be set
     * individually by calling set_rel_tol on the returned objects
     * from either HODLR_options(), BLR_options() or HSS_options().
     *
     * \param rtol the relative low-rank compression tolerance
     *
     * \see set_compression_abs_tol, set_lossy_precision,
     * compression_rel_tol
     */
    void set_compression_rel_tol(real_t rtol) {
      hss_opts_.set_rel_tol(rtol);
      blr_opts_.set_rel_tol(rtol);
      hodlr_opts_.set_rel_tol(rtol);
    }

    /**
     * Set the absolute compression tolerance to be used for low-rank
     * compression. This currently affects BLR, HSS, HODLR, HODBF,
     * Butterfly. It does not affect the lossy compression, see
     * set_lossy_precision. The same tolerances can also be set
     * individually by calling set_rel_tol on the returned objects
     * from either HODLR_options(), BLR_options() or HSS_options().
     *
     * \param rtol the relative low-rank compression tolerance
     *
     * \see set_compression_rel_tol, set_lossy_precision,
     * compression_abs_tol
     */
    void set_compression_abs_tol(real_t atol) {
      hss_opts_.set_abs_tol(atol);
      blr_opts_.set_abs_tol(atol);
      hodlr_opts_.set_abs_tol(atol);
    }

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
      hss_min_sep_size_ = s;
      blr_min_sep_size_ = s;
      hodlr_min_sep_size_ = s;
      lossy_min_sep_size_ = s;
    }
    void set_hss_min_sep_size(int s) {
      assert(s >= 0);
      hss_min_sep_size_ = s;
    }
    void set_hodlr_min_sep_size(int s) {
      assert(s >= 0);
      hodlr_min_sep_size_ = s;
    }
    void set_blr_min_sep_size(int s) {
      assert(s >= 0);
      blr_min_sep_size_ = s;
    }
    void set_lossy_min_sep_size(int s) {
      assert(s >= 0);
      lossy_min_sep_size_ = s;
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
      hss_min_front_size_ = s;
      blr_min_front_size_ = s;
      hodlr_min_front_size_ = s;
      lossy_min_front_size_ = s;
    }
    void set_hss_min_front_size(int s) {
      assert(s >= 0);
      hss_min_front_size_ = s;
    }
    void set_hodlr_min_front_size(int s) {
      assert(s >= 0);
      hodlr_min_front_size_ = s;
    }
    void set_blr_min_front_size(int s) {
      assert(s >= 0);
      blr_min_front_size_ = s;
    }
    void set_lossy_min_front_size(int s) {
      assert(s >= 0);
      lossy_min_front_size_ = s;
    }

    /**
     * Set the leaf size used by any of the rank-structured formats.
     *
     * \see HSS_options(), BLR_options(), HODLR_options()
     */
    void set_compression_leaf_size(int s) {
      hss_opts_.set_leaf_size(s);
      blr_opts_.set_leaf_size(s);
      hodlr_opts_.set_leaf_size(s);
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
    { assert(l >= 0); sep_order_level_ = l; }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    void enable_indirect_sampling() { indirect_sampling_ = true; }
    void disable_indirect_sampling() { indirect_sampling_ = false; }
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
    void enable_replace_tiny_pivots() { replace_tiny_pivots_ = true; }

    /**
     * Disable replacement of tiny pivots.
     *
     * \see enable_replace_tiny_pivots()
     */
    void disable_replace_tiny_pivots() { replace_tiny_pivots_ = false; }

    /**
     * Set the minimum pivot value. If the option replace_tiny_pivots
     * is enabled (using enable_replace_tiny_pivots()), the all pivots
     * smaller than this threshold value will be replaced by this
     * threshold value. This should not be set by the user directly,
     * but is set in the solver.
     *
     * \see enable_replace_tiny_pivots()
     */
    void set_pivot_threshold(real_t thresh) { pivot_ = thresh; }

    /**
     * Dump the root front to a set of files, one for each rank. This
     * will only have affect when running with more than one MPI rank,
     * and without compression.
     */
    void set_write_root_front(bool b) { write_root_front_ = b; }

    /**
     * Enable off-loading to the GPU. This only works when STRUMPACK
     * was configured with GPU support (through CUDA, MAGMA or SLATE)
     */
    void enable_gpu() { use_gpu_ = true; }

    /**
     * Disable GPU off-loading.
     */
    void disable_gpu() { use_gpu_ = false; }

    /**
     * Set the number of (CUDA) streams to be used in the code.
     */
    void set_gpu_streams(int s) { gpu_streams_ = s; }

    /**
     * Set the precision for lossy compression.
     */
    void set_lossy_precision(int p) { lossy_precision_ = p; }

    /**
     * Print statistics, about ranks, memory etc, for the root front
     * only.
     */
    void set_print_compressed_front_stats(bool b) { print_comp_front_stats_ = b; }

    /**
     * Set the type of proportional mapping.
     */
    void set_proportional_mapping(ProportionalMapping pmap) { prop_map_ = pmap; }

    /**
     * Check if verbose output is enabled.
     * \see set_verbose()
     */
    bool verbose() const { return verbose_; }

    /**
     * Get the maximum number of allowed iterative solver iterations.
     * \see set_maxit()
     */
    int maxit() const { return maxit_; }

    /**
     * Get the relative tolerance to be used in the iterative solver.
     * \see set_rel_tol()
     */
    real_t rel_tol() const { return rel_tol_; }

    /**
     * Get the absolute tolerance to be used in the iterative solver.
     * \see set_abs_tol()
     */
    real_t abs_tol() const { return abs_tol_; }

    /**
     * Get the type of iterative solver to be used as outer solver.
     * \see set_Krylov_solver()
     */
    KrylovSolver Krylov_solver() const { return Krylov_solver_; }

    /**
     * Get the GMRES restart length.
     * \see set_gmres_restart()
     */
    int gmres_restart() const { return gmres_restart_; }

    /**
     * Get the Gram-Schmidth orthogonalization type used in GMRES.
     * \see set_GramSchmidth_type()
     */
    GramSchmidtType GramSchmidt_type() const { return Gram_Schmidt_type_; }

    /**
     * Get the currently set fill reducing reordering method.
     * \see set_reordering_method()
     */
    ReorderingStrategy reordering_method() const { return reordering_method_; }

    /**
     * Return the nested-dissection recursion ending parameter.
     * \see set_nd_param()
     */
    int nd_param() const { return nd_param_; }

    /**
     * Return the number of levels in nested-dissection for which the
     * separators are parallel instead of split perpedicular to the
     * longest direction. This is to reduce the ranks in the F12/F21
     * blocks when using compression.
     */
    int nd_planar_levels() const { return nd_planar_levels_; }

    /**
     * Get the specified nx mesh dimension.
     * \see set_nx()
     */
    int nx() const { return nx_; }

    /**
     * Get the specified ny mesh dimension.
     * \see set_ny()
     */
    int ny() const { return ny_; }

    /**
     * Get the specified nz mesh dimension.
     * \see set_nz()
     */
    int nz() const { return nz_; }

    /**
     * Get the currently specified number of components (DoF's) per
     * mesh point.
     * \see set_components()
     */
    int components() const { return components_; }

    /**
     * Get the currently specified width of a separator.
     * \see set_separator_width()
     */
    int separator_width() const { return separator_width_; }

    /**
     * Is use of METIS_NodeNDP enabled? (instead of METIS_NodeND)
     * \see enable_METIS_NodeNDP()
     */
    bool use_METIS_NodeNDP() const { return use_METIS_NodeNDP_; }

    /**
     * Is use of METIS_NodeND enabled? (instead of METIS_NodeNDP)
     * \see enable_METIS_NodeND()
     */
    bool use_METIS_NodeND() const { return !use_METIS_NodeNDP_; }

    /**
     * Is MUMPS_SYMQAMD enabled?
     * \see enable_MUMPS_SYMQAMD()
     */
    bool use_MUMPS_SYMQAMD() const { return use_MUMPS_SYMQAMD_; }

    /**
     * Is aggressive amalgamation enabled? (only used when
     * MUMPS_SYMQAMD is enabled)
     * \see enable_agg_amalg(), enable_MUMPS_SYMQAMD()
     */
    bool use_agg_amalg() const { return use_agg_amalg_; }

    /**
     * Get the matching job to use for numerical stability reordering.
     * \see set_matching()
     */
    MatchingJob matching() const { return matching_job_; }

    /**
     * Should we log the assembly tree?
     * __Currently not supported.__
     */
    bool log_assembly_tree() const { return log_assembly_tree_; }

    /**
     * Get the type of compression to use.
     */
    CompressionType compression() const { return comp_; }


    /**
     * Return the relative compression tolerance used for the
     * currently selected low-rank method, either HODLR, HSS or BLR.
     * If NONE, LOSSY or LOSSLESS compression are selected, then this
     * returns 0.
     *
     * \see set_compression_rel_tol, set_lossy_compression
     */
    real_t compression_rel_tol(int l=0) const {
      switch (comp_) {
      case CompressionType::HSS:
        return hss_opts_.rel_tol();
      case CompressionType::BLR:
        return blr_opts_.rel_tol();
      case CompressionType::HODLR:
        return hodlr_opts_.rel_tol();
      case CompressionType::BLR_HODLR:
        if (l==0) return hodlr_opts_.rel_tol();
        else return blr_opts_.rel_tol();
      case CompressionType::ZFP_BLR_HODLR:
        if (l==0) return hodlr_opts_.rel_tol();
        else return blr_opts_.rel_tol();
      case CompressionType::LOSSY:
      case CompressionType::LOSSLESS:
      case CompressionType::NONE:
      default: return 0.;
      }
    }

    /**
     * Return the absolute compression tolerance used for the
     * currently selected low-rank method, either HODLR, HSS or BLR.
     * If NONE, LOSSY or LOSSLESS compression are selected, then this
     * returns 0.
     *
     * \see set_compression_rel_tol, set_lossy_compression
     */
    real_t compression_abs_tol(int l=0) const {
      switch (comp_) {
      case CompressionType::HSS:
        return hss_opts_.abs_tol();
      case CompressionType::BLR:
        return blr_opts_.abs_tol();
      case CompressionType::HODLR:
        return hodlr_opts_.abs_tol();
      case CompressionType::BLR_HODLR:
        if (l==0) return hodlr_opts_.abs_tol();
        else return blr_opts_.abs_tol();
      case CompressionType::ZFP_BLR_HODLR:
        if (l==0) return hodlr_opts_.abs_tol();
        else return blr_opts_.abs_tol();
      case CompressionType::LOSSY:
      case CompressionType::LOSSLESS:
      case CompressionType::NONE:
      default: return 0.;
      }
    }

    /**
     * Get the minimum size of a separator to enable compression. This
     * will depend on which type of compression is selected.
     *
     * \see set_compression(), set_compression_min_sep_size(),
     * compression_min_front_size()
     */
    int compression_min_sep_size(int l=0) const {
      switch (comp_) {
      case CompressionType::HSS:
        return hss_min_sep_size_;
      case CompressionType::BLR:
        return blr_min_sep_size_;
      case CompressionType::HODLR:
        return hodlr_min_sep_size_;
      case CompressionType::BLR_HODLR:
        if (l==0) return hodlr_min_sep_size_;
        else return blr_min_sep_size_;
      case CompressionType::ZFP_BLR_HODLR:
        if (l==0) return hodlr_min_sep_size_;
        else if (l==1) return blr_min_sep_size_;
        else return lossy_min_sep_size_;
      case CompressionType::LOSSY:
      case CompressionType::LOSSLESS:
        return lossy_min_sep_size_;
      case CompressionType::NONE:
      default:
        return std::numeric_limits<int>::max();
      }
    }
    int hss_min_sep_size() const {
      return hss_min_sep_size_;
    }
    int hodlr_min_sep_size() const {
      return hodlr_min_sep_size_;
    }
    int blr_min_sep_size() const {
      return blr_min_sep_size_;
    }
    int lossy_min_sep_size() const {
      return lossy_min_sep_size_;
    }

    /**
     * Get the minimum size of a front to enable compression. This
     * will depend on which type of compression is selected.
     *
     * \see set_compression(), set_compression_min_sep_size(),
     * compression_min_front_size()
     */
    int compression_min_front_size(int l=0) const {
      switch (comp_) {
      case CompressionType::HSS:
        return hss_min_front_size_;
      case CompressionType::BLR:
        return blr_min_front_size_;
      case CompressionType::HODLR:
        return hodlr_min_front_size_;
      case CompressionType::BLR_HODLR:
        if (l==0) return hodlr_min_front_size_;
        else return blr_min_front_size_;
      case CompressionType::ZFP_BLR_HODLR:
        if (l==0) return hodlr_min_front_size_;
        else if (l==1) return blr_min_front_size_;
        else return lossy_min_front_size_;
      case CompressionType::LOSSY:
      case CompressionType::LOSSLESS:
        return lossy_min_front_size_;
      case CompressionType::NONE:
      default:
        return std::numeric_limits<int>::max();
      }
    }
    int hss_min_front_size() const {
      return hss_min_front_size_;
    }
    int hodlr_min_front_size() const {
      return hodlr_min_front_size_;
    }
    int blr_min_front_size() const {
      return blr_min_front_size_;
    }
    int lossy_min_front_size() const {
      return lossy_min_front_size_;
    }

    /**
     * Get the leaf size used in the rank-structured format used for
     * compression. This will depend on which type of compression is
     * selected.
     *
     * \see set_compression(), set_compression_leaf_size(),
     * compression_min_sep_size()
     */
    int compression_leaf_size(int l=0) const {
      switch (comp_) {
      case CompressionType::HSS:
        return hss_opts_.leaf_size();
      case CompressionType::BLR:
        return blr_opts_.leaf_size();
      case CompressionType::HODLR:
        return hodlr_opts_.leaf_size();
      case CompressionType::BLR_HODLR:
        if (l==0) return hodlr_opts_.leaf_size();
        else return blr_opts_.leaf_size();
      case CompressionType::ZFP_BLR_HODLR:
        if (l==0) return hodlr_opts_.leaf_size();
        else if (l==1) return blr_opts_.leaf_size();
        else return 4;
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
    int separator_ordering_level() const { return sep_order_level_; }

    /**
     * Is indirect sampling for HSS construction enabled?
     */
    bool indirect_sampling() const { return indirect_sampling_; }

    /**
     * Check whether replacement of tiny pivots is enabled.
     *
     * \see enable_replace_tiny_pivots()
     */
    bool replace_tiny_pivots() const { return replace_tiny_pivots_; }

    /**
     * Get the minimum pivot value. If the option replace_tiny_pivots
     * is enabled (using enable_replace_tiny_pivots()), the all pivots
     * smaller than this threshold value will be replaced by this
     * threshold value.
     *
     * \see enable_replace_tiny_pivots() set_pivot_threshold()
     */
    real_t pivot_threshold() const { return pivot_; }

    /**
     * The root front will be written to a file.
     */
    bool write_root_front() const { return write_root_front_; }

    /**
     * Check wheter or not to use GPU off-loading.
     */
    bool use_gpu() const { return use_gpu_; }

    /**
     * Returns the number of GPU streams to use.
     */
    int gpu_streams() const { return gpu_streams_; }

    /**
     * Returns the precision for lossy compression.
     */
    int lossy_precision() const {
      return (compression() == CompressionType::LOSSLESS) ?
        -1 : lossy_precision_;
    }

    /**
     * Info about the stats of the root front will be printed to
     * std::cout
     */
    bool print_compressed_front_stats() const { return print_comp_front_stats_; }

    /**
     * Get the type of proportional mapping to be used.
     */
    ProportionalMapping proportional_mapping() const { return prop_map_; }

    /**
     * Get a (const) reference to an object holding various options
     * pertaining to the HSS code, and data structures.
     */
    const HSS::HSSOptions<scalar_t>& HSS_options() const { return hss_opts_; }

    /**
     * Get a reference to an object holding various options pertaining
     * to the HSS code, and data structures.
     */
    HSS::HSSOptions<scalar_t>& HSS_options() { return hss_opts_; }

    /**
     * Get a (const) reference to an object holding various options
     * pertaining to the BLR code, and data structures.
     */
    const BLR::BLROptions<scalar_t>& BLR_options() const { return blr_opts_; }

    /**
     * Get a reference to an object holding various options pertaining
     * to the BLR code, and data structures.
     */
    BLR::BLROptions<scalar_t>& BLR_options() { return blr_opts_; }

    /**
     * Get a (const) reference to an object holding various options
     * pertaining to the HODLR code, and data structures.
     */
    const HODLR::HODLROptions<scalar_t>& HODLR_options() const { return hodlr_opts_; }

    /**
     * Get a reference to an object holding various options pertaining
     * to the HODLR code, and data structures.
     */
    HODLR::HODLROptions<scalar_t>& HODLR_options() { return hodlr_opts_; }

    /**
     * Parse the command line options that were passed to this object
     * in the constructor. Run the code with -h or --help and call
     * this routine to see a list of supported options.
     */
    void set_from_command_line() { set_from_command_line(argc_, argv_); }

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
    void set_from_command_line(int argc, const char* const* cargv);

    /**
     * Print an overview of all supported options. Not including any
     * HSS/BLR specific options.
     */
    void describe_options() const;

  private:
    bool verbose_ = true;
    /** Krylov solver options */
    int maxit_ = 5000;
    real_t rel_tol_ = default_rel_tol<real_t>();
    real_t abs_tol_ = default_abs_tol<real_t>();
    KrylovSolver Krylov_solver_ = KrylovSolver::AUTO;
    int gmres_restart_ = 30;
    GramSchmidtType Gram_Schmidt_type_ = GramSchmidtType::MODIFIED;
    /** Reordering options */
    ReorderingStrategy reordering_method_ = ReorderingStrategy::METIS;
    int nd_planar_levels_ = 0;
    int nd_param_ = 8;
    int nx_ = 1;
    int ny_ = 1;
    int nz_ = 1;
    int components_ = 1;
    int separator_width_ = 1;
    bool use_METIS_NodeNDP_ = true;
    bool use_MUMPS_SYMQAMD_ = false;
    bool use_agg_amalg_ = false;
    MatchingJob matching_job_ = MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING;
    bool log_assembly_tree_ = false;
    bool replace_tiny_pivots_ = false;
    real_t pivot_ = std::sqrt(blas::lamch<real_t>('E'));
    bool write_root_front_ = false;
    bool print_comp_front_stats_ = false;
    ProportionalMapping prop_map_ = ProportionalMapping::FLOPS;

    /** GPU options */
#if defined(STRUMPACK_USE_CUDA) || defined(STRUMPACK_USE_HIP)
    bool use_gpu_ = true;
#else
    bool use_gpu_ = false;
#endif
    int gpu_streams_ = default_gpu_streams();

    /** compression options */
    CompressionType comp_ = CompressionType::NONE;

    /** HSS options */
    int hss_min_front_size_ = 100000;
    int hss_min_sep_size_ = 1000;
    int sep_order_level_ = 1;
    bool indirect_sampling_ = false;
    HSS::HSSOptions<scalar_t> hss_opts_;

    /** BLR options */
    BLR::BLROptions<scalar_t> blr_opts_;
    int blr_min_front_size_ = 100000;
    int blr_min_sep_size_ = 512;

    /** HODLR options */
    HODLR::HODLROptions<scalar_t> hodlr_opts_;
    int hodlr_min_front_size_ = 100000;
    int hodlr_min_sep_size_ = 5000;

    /** LOSSY/LOSSLESS options */
    int lossy_min_front_size_ = 100000;
    int lossy_min_sep_size_ = 8;
    int lossy_precision_ = 16;

    int argc_ = 0;
    const char* const* argv_ = nullptr;
  };

} // end namespace strumpack

#endif // SPOPTIONS_HPP
