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

#include <cstring>

// this is needed for RealType, put that somewhere else?
#include "dense/BLASLAPACKWrapper.hpp"
#include "misc/RandomWrapper.hpp"
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
  inline std::string get_name(ReorderingStrategy method) {
    switch (method) {
    case ReorderingStrategy::NATURAL: return "Natural"; break;
    case ReorderingStrategy::METIS: return "Metis"; break;
    case ReorderingStrategy::SCOTCH: return "Scotch"; break;
    case ReorderingStrategy::GEOMETRIC: return "Geometric"; break;
    case ReorderingStrategy::PARMETIS: return "ParMetis"; break;
    case ReorderingStrategy::PTSCOTCH: return "PTScotch"; break;
    case ReorderingStrategy::RCM: return "RCM"; break;
    }
    return "UNKNOWN";
  }

  /**
   * Check whether or not the reordering needs to be run in parallel.
   */
  inline bool is_parallel(ReorderingStrategy method) {
    switch (method) {
    case ReorderingStrategy::NATURAL: return false; break;
    case ReorderingStrategy::METIS: return false; break;
    case ReorderingStrategy::SCOTCH: return false; break;
    case ReorderingStrategy::GEOMETRIC: return true; break;
    case ReorderingStrategy::PARMETIS: return true; break;
    case ReorderingStrategy::PTSCOTCH: return true; break;
    case ReorderingStrategy::RCM: return false; break;
    }
    return false;
  }

  /**
   * Enumeration of rank-structured data formats, which can be used
   * for compression within the sparse solver.
   * \ingroup Enumerations
   */
  enum class CompressionType {
    NONE,   /*!< No compression, purely direct solver  */
    HSS,    /*!< HSS compression of frontal matrices   */
    BLR,    /*!< Block low-rank compression of fronts  */
    HODLR   /*!< Hierarchically Off-diagonal Low-Rank
                 compression of frontal matrices,      */
  };

  /**
   * Return a name/string for the CompressionType.
   */
  inline std::string get_name(CompressionType comp) {
    switch (comp) {
    case CompressionType::NONE: return "none";
    case CompressionType::HSS: return "HSS";
    case CompressionType::BLR: return "BLR";
    case CompressionType::HODLR: return "HODLR";
    }
  }


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
  inline MatchingJob get_matching(int job) {
    if (job < 0 || job > 6)
      std::cerr << "ERROR: Matching job not recognized!!" << std::endl;
    switch (job) {
    case 0: return MatchingJob::NONE;
    case 1: return MatchingJob::MAX_CARDINALITY;
    case 2: return MatchingJob::MAX_SMALLEST_DIAGONAL;
    case 3: return MatchingJob::MAX_SMALLEST_DIAGONAL_2;
    case 4: return MatchingJob::MAX_DIAGONAL_SUM;
    case 5: return MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING;
    case 6: return MatchingJob::COMBBLAS;
    }
    return MatchingJob::NONE;
  }

  /**
   * Convert a MatchingJob enum type to a job number. Prefer to use
   * the MachingJob enum instead of the job number.
   */
  inline int get_matching(MatchingJob job) {
    switch (job) {
    case MatchingJob::NONE: return 0;
    case MatchingJob::MAX_CARDINALITY: return 1;
    case MatchingJob::MAX_SMALLEST_DIAGONAL: return 2;
    case MatchingJob::MAX_SMALLEST_DIAGONAL_2: return 3;
    case MatchingJob::MAX_DIAGONAL_SUM: return 4;
    case MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING: return 5;
    case MatchingJob::COMBBLAS: return 6;
    }
    return -1;
  }

  /**
   * Return a string describing the matching algorithm.
   */
  inline std::string get_description(MatchingJob job) {
    switch (job) {
    case MatchingJob::NONE: return "none";
    case MatchingJob::MAX_CARDINALITY:
      return "maximum cardinality ! Doesn't work";
    case MatchingJob::MAX_SMALLEST_DIAGONAL:
      return "maximum smallest diagonal value, version 1";
    case MatchingJob::MAX_SMALLEST_DIAGONAL_2:
      return "maximum smallest diagonal value, version 2";
    case MatchingJob::MAX_DIAGONAL_SUM:
      return "maximum sum of diagonal values";
    case MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING:
      return "maximum matching with row and column scaling";
    case MatchingJob::COMBBLAS:
      return "approximate weigthed perfect matching, from CombBLAS";
    }
    return "UNKNOWN";
  }


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
    SPOptions(int argc, char* argv[]) :
      _argc(argc), _argv(argv) {
      _hss_opts.set_verbose(false);
      _blr_opts.set_verbose(false);
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
    void set_rel_tol(real_t rel_tol)
    { assert(rel_tol <= real_t(1.) && rel_tol >= real_t(0.)); _rel_tol = rel_tol; }

    /**
     * Set the absolute tolerance to use in the iterative
     * solvers.
     *
     * \param abs_tol absolute tolerance
     */
    void set_abs_tol(real_t abs_tol)
    { assert(abs_tol >= real_t(0.));  _abs_tol = abs_tol; }

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
     * \see enable_HSS(), enable_BLR(), enable_HODLR()
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
    void set_dimensions(int nx, int ny=1, int nz=1)
    { assert(nx>=1 && ny>=1 && nz>=1);
      _nx = nx; _ny = ny; _nz = nz; }

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
     * Enable HSS compression in the sparse solver. See the relevant
     * section in the online manual for tuning of the HSS enabled
     * preconditioner. Enabling HSS compression will also change the
     * iterative solver from iterative refinement to GMRES (this is
     * the AUTO solver setting, but can be manually overwritten).
     *
     * \see disable_HSS(), HSS_options(), set_HSS_min_sep_size(),
     * enable_BLR(), enable_HODLR()
     */
    void enable_HSS() { _comp = CompressionType::HSS; }

    /**
     * Do not use HSS compression in the sparse solver. The type of
     * compression will be reset to NONE, and the solver will act as a
     * direct solver without compression.
     *
     * \see enable_HSS(), enable_BLR(), enable_HODLR()
     */
    void disable_HSS() { _comp = CompressionType::NONE; }

    /**
     * Set the minimum size of frontal matrices (dense submatrices of
     * the sparse triangular factors) for which to use HSS compression
     * (when HSS compression has been enabled by the user using
     * enable_HSS()). Most fronts will be quite small, and will not
     * have benefit from HSS compression.
     *
     * __This is currently ignored!! Use set_HSS_min_sep_size
     * instead__
     *
     * \see HSS_options(), set_HSS_min_sep_size()
     */
    void set_HSS_min_front_size(int s)
    { assert(_hss_min_front_size >= 0); _hss_min_front_size = s; }

    /**
     * Set the minimum size of the top left part of frontal matrices
     * (dense submatrices of the sparse triangular factors), ie, the
     * part corresponding to the separators, for which to use HSS
     * compression (when HSS compression has been enabled by the user
     * using enable_HSS()). Most fronts will be quite small, and will
     * not have benefit from HSS compression.
     *
     * \see HSS_options(), set_HSS_min_front_size()
     */
    void set_HSS_min_sep_size(int s)
    { assert(_hss_min_sep_size >= 0); _hss_min_sep_size = s; }

    /**
     * Enable Block Low-Rank compression inside the sparse solver. If
     * HSS compression was previously enabled, this will be disabled,
     * and changed to BLR compression. We refer to the manual for a
     * description, and tips on tuning the BLR enabled
     * preconditioner. Enabling BLR compression will change the
     * default iterative solver from iterative refinement to GMRES
     * (this is the AUTO solver strategy).
     *
     * \see disable_BLR(), set_BLR_min_sep_size(), BLR_options()
     */
    void enable_BLR() { _comp = CompressionType::BLR; }

    /**
     * Disable BLR compression in the sparse solver.
     *
     * \see enable_BLR()
     */
    void disable_BLR() { _comp = CompressionType::NONE; }

    /**
     * Set the minimum size of frontal matrices (dense submatrices of
     * the sparse triangular factors) for which to use BLR compression
     * (when BLR compression has been enabled by the user using
     * enable_BLR()). Most fronts will be quite small, and will not
     * have benefit from BLR compression.
     *
     * __This is currently ignored!! Use set_BLR_min_sep_size
     * instead__
     *
     * \see BLR_options(), set_BLR_min_sep_size()
     */
    void set_BLR_min_front_size(int s)
    { assert(_blr_min_front_size >= 0); _blr_min_front_size = s; }

    /**
     * Set the minimum size of the top left part of frontal matrices
     * (dense submatrices of the sparse triangular factors), ie, the
     * part corresponding to the separators, for which to use BLR
     * compression (when BLR compression has been enabled by the user
     * using enable_BLR()). Most fronts will be quite small, and will
     * not have benefit from BLR compression.
     *
     * \see BLR_options(), set_BLR_min_front_size()
     */
    void set_BLR_min_sep_size(int s)
    { assert(_blr_min_sep_size >= 0); _blr_min_sep_size = s; }


    /**
     * Enable HODLR compression inside the sparse solver. If HSS/BLR
     * compression was previously enabled, this will be disabled, and
     * changed to HODLR compression. We refer to the manual for a
     * description, and tips on tuning the HODLR enabled
     * preconditioner. Enabling HODLR compression will change the
     * default iterative solver from iterative refinement to GMRES
     * (this is the AUTO solver strategy).
     *
     * \see disable_HODLR(), set_HODLR_min_sep_size(), HODLR_options()
     */
    void enable_HODLR() { _comp = CompressionType::HODLR; }

    /**
     * Disable HODLR compression in the sparse solver.
     *
     * \see enable_HODLR()
     */
    void disable_HODLR() { _comp = CompressionType::NONE; }

    /**
     * Set the minimum size of frontal matrices (dense submatrices of
     * the sparse triangular factors) for which to use HODLR compression
     * (when HODLR compression has been enabled by the user using
     * enable_HODLR()). Most fronts will be quite small, and will not
     * have benefit from HODLR compression.
     *
     * __This is currently ignored!! Use set_HODLR_min_sep_size
     * instead__
     *
     * \see HODLR_options(), set_HODLR_min_sep_size()
     */
    void set_HODLR_min_front_size(int s)
    { assert(_hodlr_min_front_size >= 0); _hodlr_min_front_size = s; }

    /**
     * Set the minimum size of the top left part of frontal matrices
     * (dense submatrices of the sparse triangular factors), ie, the
     * part corresponding to the separators, for which to use HODLR
     * compression (when HODLR compression has been enabled by the user
     * using enable_HODLR()). Most fronts will be quite small, and will
     * not have benefit from HODLR compression.
     *
     * \see HODLR_options(), set_HODLR_min_front_size()
     */
    void set_HODLR_min_sep_size(int s)
    { assert(_hodlr_min_sep_size >= 0); _hodlr_min_sep_size = s; }

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
     * Is HSS compression enabled?
     * \see enable_HSS()
     */
    bool use_HSS() const { return _comp == CompressionType::HSS; }

    /**
     * Get the minimum front size for HSS compression.
     *
     *__Not used at the moment!!, Use set_HSS_min_sep_size()
     * instead.__
     *
     * \see set_HSS_min_sep_size(), set_HSS_min_front_size()
     */
    int HSS_min_front_size() const { return _hss_min_front_size; }

    /**
     * Get the minimum size of a separator to enable HSS compression.
     * \see set_HSS_min_sep_size()
     */
    int HSS_min_sep_size() const { return _hss_min_sep_size; }

    /**
     * Is BLR compression enabled?
     * \see enable_BLR()
     */
    bool use_BLR() const { return _comp == CompressionType::BLR; }

    /**
     * Get the minimum front size for BLR compression.
     *
     *__Not used at the moment!!, Use set_BLR_min_sep_size()
     * instead.__
     *
     * \see set_BLR_min_sep_size(), set_BLR_min_front_size()
     */
    int BLR_min_front_size() const { return _blr_min_front_size; }

    /**
     * Get the minimum size of a separator to enable BLR compression.
     * \see set_BLR_min_sep_size()
     */
    int BLR_min_sep_size() const { return _blr_min_sep_size; }

    /**
     * Is HODLR compression enabled?
     * \see enable_BLR()
     */
    bool use_HODLR() const { return _comp == CompressionType::HODLR; }

    /**
     * Get the minimum front size for HODLR compression.
     *
     *__Not used at the moment!!, Use set_HODLR_min_sep_size()
     * instead.__
     *
     * \see set_HODLR_min_sep_size(), set_HODLR_min_front_size()
     */
    int HODLR_min_front_size() const { return _hodlr_min_front_size; }

    /**
     * Get the minimum size of a separator to enable HODLR compression.
     * \see set_HODLR_min_sep_size()
     */
    int HODLR_min_sep_size() const { return _hodlr_min_sep_size; }

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
    void set_from_command_line(int argc, const char* const* argv) {
      std::vector<char*> argv_local(argc);
      for (int i=0; i<argc; i++) {
        argv_local[i] = new char[strlen(argv[i])+1];
        strcpy(argv_local[i], argv[i]);
      }
      option long_options[] = {
        {"sp_maxit",                     required_argument, 0, 1},
        {"sp_rel_tol",                   required_argument, 0, 2},
        {"sp_abs_tol",                   required_argument, 0, 3},
        {"sp_Krylov_solver",             required_argument, 0, 4},
        {"sp_gmres_restart",             required_argument, 0, 5},
        {"sp_GramSchmidt_type",          required_argument, 0, 6},
        {"sp_reordering_method",         required_argument, 0, 7},
        {"sp_nd_param",                  required_argument, 0, 8},
        {"sp_enable_METIS_NodeNDP",      no_argument, 0, 9},
        {"sp_disable_METIS_NodeNDP",     no_argument, 0, 10},
        {"sp_enable_METIS_NodeND",       no_argument, 0, 11},
        {"sp_disable_METIS_NodeND",      no_argument, 0, 12},
        {"sp_enable_MUMPS_SYMQAMD",      no_argument, 0, 13},
        {"sp_disable_MUMPS_SYMQAMD",     no_argument, 0, 14},
        {"sp_enable_aggamal",            no_argument, 0, 15},
        {"sp_disable_aggamal",           no_argument, 0, 16},
        {"sp_matching",                  required_argument, 0, 17},
        {"sp_enable_assembly_tree_log",  no_argument, 0, 18},
        {"sp_disable_assembly_tree_log", no_argument, 0, 19},
        {"sp_enable_hss",                no_argument, 0, 20},
        {"sp_disable_hss",               no_argument, 0, 21},
        {"sp_hss_min_front_size",        required_argument, 0, 22},
        {"sp_hss_min_sep_size",          required_argument, 0, 23},
        {"sp_enable_blr",                no_argument, 0, 24},
        {"sp_disable_blr",               no_argument, 0, 25},
        {"sp_blr_min_front_size",        required_argument, 0, 26},
        {"sp_blr_min_sep_size",          required_argument, 0, 27},
        {"sp_enable_hodlr",              no_argument, 0, 28},
        {"sp_disable_hodlr",             no_argument, 0, 29},
        {"sp_hodlr_min_front_size",      required_argument, 0, 30},
        {"sp_hodlr_min_sep_size",        required_argument, 0, 31},
        {"sp_separator_ordering_level",  required_argument, 0, 32},
        {"sp_enable_indirect_sampling",  no_argument, 0, 33},
        {"sp_disable_indirect_sampling", no_argument, 0, 34},
        {"sp_enable_replace_tiny_pivots", no_argument, 0, 35},
        {"sp_disable_replace_tiny_pivots", no_argument, 0, 36},
        {"sp_nx",                        required_argument, 0, 37},
        {"sp_ny",                        required_argument, 0, 38},
        {"sp_nz",                        required_argument, 0, 39},
        {"sp_components",                required_argument, 0, 40},
        {"sp_separator_width",           required_argument, 0, 41},
        {"sp_verbose",                   no_argument, 0, 'v'},
        {"sp_quiet",                     no_argument, 0, 'q'},
        {"help",                         no_argument, 0, 'h'},
        {NULL, 0, NULL, 0}
      };
      int c, option_index = 0;
      // bool unrecognized_options = false;
      opterr = 0;
      while ((c = getopt_long_only
              (argc, argv_local.data(),
               "hvq", long_options, &option_index)) != -1) {
        switch (c) {
        case 1: {
          std::istringstream iss(optarg);
          iss >> _maxit;
          set_maxit(_maxit);
        } break;
        case 2: {
          std::istringstream iss(optarg);
          iss >> _rel_tol;
          set_rel_tol(_rel_tol);
        } break;
        case 3: {
          std::istringstream iss(optarg);
          iss >> _abs_tol;
          set_abs_tol(_abs_tol);
        } break;
        case 4: {
          std::string s; std::istringstream iss(optarg); iss >> s;
          if (s.compare("auto") == 0)
            set_Krylov_solver(KrylovSolver::AUTO);
          else if (s.compare("direct") == 0)
            set_Krylov_solver(KrylovSolver::DIRECT);
          else if (s.compare("refinement") == 0)
            set_Krylov_solver(KrylovSolver::REFINE);
          else if (s.compare("pgmres") == 0)
            set_Krylov_solver(KrylovSolver::PREC_GMRES);
          else if (s.compare("gmres") == 0)
            set_Krylov_solver(KrylovSolver::GMRES);
          else if (s.compare("pbicgstab") == 0)
            set_Krylov_solver(KrylovSolver::PREC_BICGSTAB);
          else if (s.compare("bicgstab") == 0)
            set_Krylov_solver(KrylovSolver::BICGSTAB);
          else
            std::cerr << "# WARNING: Krylov solver not recognized,"
              " using default" << std::endl;
        } break;
        case 5: {
          std::istringstream iss(optarg);
          iss >> _gmres_restart;
          set_gmres_restart(_gmres_restart); } break;
        case 6: {
          std::string s;
          std::istringstream iss(optarg); iss >> s;
          if (s.compare("modified") == 0)
            set_GramSchmidt_type(GramSchmidtType::MODIFIED);
          else if (s.compare("classical") == 0)
            set_GramSchmidt_type(GramSchmidtType::CLASSICAL);
          else
            std::cerr << "# WARNING: Gram-Schmidt type not recognized,"
              " use 'modified' or classical" << std::endl;
        } break;
        case 7: {
          std::string s; std::istringstream iss(optarg); iss >> s;
          if (s.compare("natural") == 0)
            set_reordering_method(ReorderingStrategy::NATURAL);
          else if (s.compare("metis") == 0)
            set_reordering_method(ReorderingStrategy::METIS);
          else if (s.compare("parmetis") == 0)
            set_reordering_method(ReorderingStrategy::PARMETIS);
          else if (s.compare("scotch") == 0)
            set_reordering_method(ReorderingStrategy::SCOTCH);
          else if (s.compare("ptscotch") == 0)
            set_reordering_method(ReorderingStrategy::PTSCOTCH);
          else if (s.compare("geometric") == 0)
            set_reordering_method(ReorderingStrategy::GEOMETRIC);
          else if (s.compare("rcm") == 0)
            set_reordering_method(ReorderingStrategy::RCM);
          else
            std::cerr << "# WARNING: matrix reordering strategy not"
              " recognized, use 'metis', 'parmetis', 'scotch', 'ptscotch',"
              " 'geometric' or 'rcm'" << std::endl;
        } break;
        case 8: {
          std::istringstream iss(optarg);
          iss >> _nd_param;
          set_nd_param(_nd_param);
        } break;
        case 9:  { enable_METIS_NodeNDP(); } break;
        case 10: { disable_METIS_NodeNDP(); } break;
        case 11: { enable_METIS_NodeND(); } break;
        case 12: { disable_METIS_NodeND(); } break;
        case 13: { enable_MUMPS_SYMQAMD(); } break;
        case 14: { disable_MUMPS_SYMQAMD(); } break;
        case 15: { enable_agg_amalg(); } break;
        case 16: { disable_agg_amalg(); } break;
        case 17: {
          std::istringstream iss(optarg);
          int job; iss >> job;
          set_matching(get_matching(job));
        } break;
        case 18: { enable_assembly_tree_log(); } break;
        case 19: { disable_assembly_tree_log(); } break;
        case 20: { enable_HSS(); } break;
        case 21: { disable_HSS(); } break;
        case 22: {
          std::istringstream iss(optarg);
          iss >> _hss_min_front_size;
          set_HSS_min_front_size(_hss_min_front_size);
        } break;
        case 23: {
          std::istringstream iss(optarg);
          iss >> _hss_min_sep_size;
          set_HSS_min_sep_size(_hss_min_sep_size);
        } break;
        case 24: { enable_BLR(); } break;
        case 25: { disable_BLR(); } break;
        case 26: {
          std::istringstream iss(optarg);
          iss >> _blr_min_front_size;
          set_BLR_min_front_size(_blr_min_front_size);
        } break;
        case 27: {
          std::istringstream iss(optarg);
          iss >> _blr_min_sep_size;
          set_BLR_min_sep_size(_blr_min_sep_size);
        } break;
        case 28: { enable_HODLR(); } break;
        case 29: { disable_HODLR(); } break;
        case 30: {
          std::istringstream iss(optarg);
          iss >> _hodlr_min_front_size;
          set_HODLR_min_front_size(_hodlr_min_front_size);
        } break;
        case 31: {
          std::istringstream iss(optarg);
          iss >> _hodlr_min_sep_size;
          set_HODLR_min_sep_size(_hodlr_min_sep_size);
        } break;
        case 32: {
          std::istringstream iss(optarg);
          iss >> _sep_order_level;
          set_separator_ordering_level(_sep_order_level);
        } break;
        case 33: { enable_indirect_sampling(); } break;
        case 34: { disable_indirect_sampling(); } break;
        case 35: { enable_replace_tiny_pivots(); } break;
        case 36: { disable_replace_tiny_pivots(); } break;
        case 37: {
          std::istringstream iss(optarg);
          iss >> _nx;
          set_nx(_nx);
        } break;
        case 38: {
          std::istringstream iss(optarg);
          iss >> _ny;
          set_ny(_ny);
        } break;
        case 39: {
          std::istringstream iss(optarg);
          iss >> _nz;
          set_nz(_nz);
        } break;
        case 40: {
          std::istringstream iss(optarg);
          iss >> _components;
          set_components(_components);
        } break;
        case 41: {
          std::istringstream iss(optarg);
          iss >> _separator_width;
          set_separator_width(_separator_width);
        } break;
        case 'h': { describe_options(); } break;
        case 'v': set_verbose(true); break;
        case 'q': set_verbose(false); break;
        // case '?': unrecognized_options = true; break;
        default: break;
        }
      }
      for (auto s : argv_local) delete[] s;

      // if (unrecognized_options/* && is_root*/)
      //   std::cerr << "# WARNING STRUMPACK: Unrecognized options."
      //             << std::endl;
      HSS_options().set_from_command_line(argc, argv);
      BLR_options().set_from_command_line(argc, argv);
      HODLR_options().set_from_command_line(argc, argv);
    }

    /**
     * Print an overview of all supported options. Not including any
     * HSS/BLR specific options.
     */
    void describe_options() const {
      std::cout << "# STRUMPACK options:" << std::endl;
      std::cout << "#   --sp_maxit int (default " << maxit() << ")" << std::endl;
      std::cout << "#          maximum Krylov iterations" << std::endl;
      std::cout << "#   --sp_rel_tol real_t (default " << rel_tol() << ")"
                << std::endl;
      std::cout << "#          Krylov relative (preconditioned) residual"
                << " stopping tolerance" << std::endl;
      std::cout << "#   --sp_abs_tol real_t (default " << abs_tol() << ")"
                << std::endl;
      std::cout << "#          Krylov absolute (preconditioned) residual"
                << " stopping tolerance" << std::endl;
      std::cout << "#   --sp_Krylov_solver auto|direct|refinement|pgmres|"
                << "gmres|pbicgstab|bicgstab" << std::endl;
      std::cout << "#          default: auto (refinement when no HSS, pgmres"
                << " (preconditioned) with HSS compression)" << std::endl;
      std::cout << "#   --sp_gmres_restart int (default " << gmres_restart()
                << ")" << std::endl;
      std::cout << "#          gmres restart length" << std::endl;
      std::cout << "#   --sp_GramSchmidt_type [modified|classical]"
                << std::endl;
      std::cout << "#          Gram-Schmidt type for GMRES" << std::endl;
      std::cout << "#   --sp_reordering_method natural|metis|scotch|parmetis|"
                << "ptscotch|rcm|geometric" << std::endl;
      std::cout << "#          Code for nested dissection." << std::endl;
      std::cout << "#          Geometric only works on regular meshes and you"
                << " need to provide the sizes." << std::endl;
      std::cout << "#   --sp_nd_param int (default " << _nd_param << ")"
                << std::endl;
      std::cout << "#   --sp_nx int (default " << _nx << ")"
                << std::endl;
      std::cout << "#   --sp_ny int (default " << _ny << ")"
                << std::endl;
      std::cout << "#   --sp_nz int (default " << _nz << ")"
                << std::endl;
      std::cout << "#   --sp_components int (default " << _components << ")"
                << std::endl;
      std::cout << "#   --sp_separator_width int (default "
                << _separator_width << ")" << std::endl;
      std::cout << "#   --sp_enable_METIS_NodeNDP (default "
                << std::boolalpha << use_METIS_NodeNDP() << ")" << std::endl;
      std::cout << "#          use undocumented Metis routine NodeNDP"
                << " instead of NodeND" << std::endl;
      std::cout << "#   --sp_disable_METIS_NodeNDP (default "
                << std::boolalpha << (!use_METIS_NodeNDP()) << ")"
                << std::endl;
      std::cout << "#          use Metis routine NodeND instead of the"
                << " undocumented NodeNDP" << std::endl;
      std::cout << "#   --sp_enable_METIS_NodeND (default " << std::boolalpha
                << use_METIS_NodeND() << ")" << std::endl;
      std::cout << "#          use Metis routine NodeND instead of the"
                << " undocumented NodeNDP" << std::endl;
      std::cout << "#   --sp_disable_METIS_NodeND (default "
                << std::boolalpha << !use_METIS_NodeND() << ")" << std::endl;
      std::cout << "#          use undocumented Metis routine NodeNDP"
                << " instead of NodeND" << std::endl;
      std::cout << "#   --sp_enable_MUMPS_SYMQAMD (default "
                << std::boolalpha << use_MUMPS_SYMQAMD() << ")" << std::endl;
      std::cout << "#   --sp_disable_MUMPS_SYMQAMD (default "
                << std::boolalpha << !use_MUMPS_SYMQAMD() << ")" << std::endl;
      std::cout << "#   --sp_enable_agg_amalg (default "
                << std::boolalpha << use_agg_amalg() << ")" << std::endl;
      std::cout << "#   --sp_disable_agg_amalg (default "
                << std::boolalpha << !use_agg_amalg() << ")" << std::endl;
      std::cout << "#   --sp_matching int [0-6] (default "
                << get_matching(matching()) << ")" << std::endl;
      for (int i=0; i<7; i++)
        std::cout << "#      " << i << " " <<
          get_description(get_matching(i)) << std::endl;
      std::cout << "#   --sp_enable_hss (default " << std::boolalpha
                << use_HSS() << ")" << std::endl;
      std::cout << "#   --sp_disable_hss (default " << std::boolalpha
                << !use_HSS() << ")" << std::endl;
      // std::cout << "#   --sp_hss_min_front_size int (default "
      //           << HSS_min_front_size() << ")" << std::endl;
      // std::cout << "#          minimum size of front for HSS compression"
      //           << std::endl;
      std::cout << "#   --sp_hss_min_sep_size int (default "
                << HSS_min_sep_size() << ")" << std::endl;
      std::cout << "#          minimum size of the separator for HSS"
                << " compression of the front" << std::endl;
      std::cout << "#   --sp_enable_blr (default " << std::boolalpha
                << use_BLR() << ")" << std::endl;
      std::cout << "#   --sp_disable_blr (default " << std::boolalpha
                << !use_BLR() << ")" << std::endl;
      std::cout << "#   --sp_blr_min_sep_size int (default "
                << BLR_min_sep_size() << ")" << std::endl;
      std::cout << "#          minimum size of the separator for BLR"
                << " compression of the front" << std::endl;
      std::cout << "#   --sp_enable_hodlr (default " << std::boolalpha
                << use_HODLR() << ")" << std::endl;
      std::cout << "#   --sp_disable_hodlr (default " << std::boolalpha
                << !use_HODLR() << ")" << std::endl;
      std::cout << "#   --sp_hodlr_min_sep_size int (default "
                << HODLR_min_sep_size() << ")" << std::endl;
      std::cout << "#          minimum size of the separator for HODLR"
                << " compression of the front" << std::endl;
      std::cout << "#   --sp_separator_ordering_level (default "
                << separator_ordering_level() << ")" << std::endl;
      std::cout << "#   --sp_enable_indirect_sampling" << std::endl;
      std::cout << "#   --sp_disable_indirect_sampling" << std::endl;
      std::cout << "#   --sp_enable_replace_tiny_pivots" << std::endl;
      std::cout << "#   --sp_disable_replace_tiny_pivots" << std::endl;
      std::cout << "#   --sp_verbose or -v (default " << verbose() << ")"
                << std::endl;
      std::cout << "#   --sp_quiet or -q (default " << !verbose() << ")"
                << std::endl;
      std::cout << "#   --help or -h" << std::endl;
      //synchronize();
      //HSS_options().describe_options();
    }

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

    /** compression options */
    CompressionType _comp = CompressionType::NONE;

    /** HSS options */
    int _hss_min_front_size = 1000;
    int _hss_min_sep_size = 256;
    int _sep_order_level = 1;
    bool _indirect_sampling = false;
    bool _replace_tiny_pivots = false;
    HSS::HSSOptions<scalar_t> _hss_opts;

    /** BLR options */
    BLR::BLROptions<scalar_t> _blr_opts;
    int _blr_min_front_size = 1000;
    int _blr_min_sep_size = 256;

    /** HODLR options */
    HODLR::HODLROptions<scalar_t> _hodlr_opts;
    int _hodlr_min_front_size = 1000;
    int _hodlr_min_sep_size = 256;

    int _argc = 0;
    char** _argv = nullptr;
  };

} // end namespace strumpack

#endif // SPOPTIONS_HPP
