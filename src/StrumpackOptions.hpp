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
/*! \file StrumpackSparseOptions.hpp
 * \brief Holds options for the sparse solver.
 */
#ifndef SPOPTIONS_HPP
#define SPOPTIONS_HPP

#include <string.h>

// this is needed for RealType, put that somewhere else?
#include "dense/BLASLAPACKWrapper.hpp"
#include "misc/RandomWrapper.hpp"
#include "misc/MPIWrapper.hpp"
#include "HSS/HSSOptions.hpp"

namespace strumpack {

  // TODO move this to MatrixReordering class
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

  enum class MC64Job {
    NONE,                         /*!< Don't do anything                   */
    MAX_CARDINALITY,              /*!< Maximum cardinality                 */
    MAX_SMALLEST_DIAGONAL,        /*!< Maximum smallest diagonal value     */
    MAX_SMALLEST_DIAGONAL_2,      /*!< Same as MAX_SMALLEST_DIAGONAL,
                                    but different algorithm                */
    MAX_DIAGONAL_SUM,             /*!< Maximum sum of diagonal values      */
    MAX_DIAGONAL_PRODUCT_SCALING  /*!< Maximum product of diagonal values
                                    and row and column scaling             */
  };

  inline int MC64_job_number(MC64Job job) {
    switch (job) {
    case MC64Job::NONE: return 0;
    case MC64Job::MAX_CARDINALITY: return 1;
    case MC64Job::MAX_SMALLEST_DIAGONAL: return 2;
    case MC64Job::MAX_SMALLEST_DIAGONAL_2: return 3;
    case MC64Job::MAX_DIAGONAL_SUM: return 4;
    case MC64Job::MAX_DIAGONAL_PRODUCT_SCALING: return 5;
    }
    return -1;
  }

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
  /*! \brief Type of Gram-Schmidt orthogonalization used in GMRes.
   * \ingroup Enumerations */
  enum class GramSchmidtType {
    CLASSICAL,   /*!< Classical Gram-Schmidt is faster, more scalable.   */
    MODIFIED     /*!< Modified Gram-Schmidt is slower, but stable.       */
  };

  /*! \brief Type of outer iterative (Krylov) solver.
   * \ingroup Enumerations */
  enum class KrylovSolver {
    AUTO,           /*!< Use iterative refinement if no HSS compression is
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

  template<typename real_t> inline real_t default_rel_tol()
  { return real_t(1.e-6); }
  template<typename real_t> inline real_t default_abs_tol()
  { return real_t(1.e-10); }
  template<> inline float default_rel_tol() { return 1.e-4; }
  template<> inline float default_abs_tol() { return 1.e-6; }

  template<typename scalar_t> class SPOptions {
    using real_t = typename RealType<scalar_t>::value_type;

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
    bool _use_METIS_NodeNDP = true;
    bool _use_MUMPS_SYMQAMD = false;
    bool _use_agg_amalg = false;
    int _mc64job = 5;
    bool _log_assembly_tree = false;
    /** HSS options */
    bool _use_hss = false;
    int _hss_min_front_size = 1000;
    int _hss_min_sep_size = 256;
    int _sep_order_level = 1;
    bool _indirect_sampling = false;
    bool _replace_tiny_pivots = false;
    HSS::HSSOptions<scalar_t> _hss_opts;

    int _argc = 0;
    char** _argv = nullptr;

  public:
    SPOptions() { _hss_opts.set_verbose(false); }
    SPOptions(int argc, char* argv[]) :
      _argc(argc), _argv(argv) { _hss_opts.set_verbose(false); }

    void set_verbose(bool verbose) { _verbose = verbose; }
    void set_maxit(int maxit) { assert(maxit >= 1); _maxit = maxit; }
    void set_rel_tol(real_t rel_tol)
    { assert(rel_tol <= real_t(1.) && rel_tol >= real_t(0.));
      _rel_tol = rel_tol; }
    void set_abs_tol(real_t abs_tol)
    { assert(abs_tol >= real_t(0.));  _abs_tol = abs_tol; }
    void set_Krylov_solver(KrylovSolver s) { _Krylov_solver = s; }
    void set_gmres_restart(int m) { assert(m >= 1); _gmres_restart = m; }
    void set_GramSchmidt_type(GramSchmidtType t) { _Gram_Schmidt_type = t; }
    void set_reordering_method(ReorderingStrategy m)
    { _reordering_method = m; }
    void set_nd_param(int nd_param)
    { assert(nd_param>=0); _nd_param = nd_param; }
    void enable_METIS_NodeNDP() { _use_METIS_NodeNDP = true; }
    void disable_METIS_NodeNDP() { _use_METIS_NodeNDP = false; }
    void enable_METIS_NodeND() { _use_METIS_NodeNDP = false; }
    void disable_METIS_NodeND() { _use_METIS_NodeNDP = true; }
    void enable_MUMPS_SYMQAMD() { _use_MUMPS_SYMQAMD = true; }
    void disable_MUMPS_SYMQAMD() { _use_MUMPS_SYMQAMD = false; }
    void enable_agg_amalg() { _use_agg_amalg = true; }
    void disable_agg_amalg() { _use_agg_amalg = false; }
    void set_mc64job(int job)
    { if (job < 0 || job > 5)
        std::cerr << "# WARNING: invalid mc64 job number" << std::endl;
      _mc64job = job; }
    void set_mc64job(MC64Job job) { _mc64job = MC64_job_number(job); }
    void enable_assembly_tree_log() { _log_assembly_tree = true; }
    void disable_assembly_tree_log() { _log_assembly_tree = false; }
    void enable_HSS() { _use_hss = true; }
    void disable_HSS() { _use_hss = false; }
    void set_HSS_min_front_size(int s)
    { assert(_hss_min_front_size >= 0); _hss_min_front_size = s; }
    void set_HSS_min_sep_size(int s)
    { assert(_hss_min_sep_size >= 0); _hss_min_sep_size = s; }
    void set_separator_ordering_level(int l)
    { assert(l >= 0); _sep_order_level = l; }
    void enable_indirect_sampling() { _indirect_sampling = true; }
    void disable_indirect_sampling() { _indirect_sampling = false; }
    void enable_replace_tiny_pivots() { _replace_tiny_pivots = true; }
    void disable_replace_tiny_pivots() { _replace_tiny_pivots = false; }

    bool verbose() const { return _verbose; }
    int maxit() const { return _maxit; }
    real_t rel_tol() const { return _rel_tol; }
    real_t abs_tol() const { return _abs_tol; }
    KrylovSolver Krylov_solver() const { return _Krylov_solver; }
    int gmres_restart() const { return _gmres_restart; }
    GramSchmidtType GramSchmidt_type() const { return _Gram_Schmidt_type; }
    ReorderingStrategy reordering_method() const
    { return _reordering_method; }
    int nd_param() const { return _nd_param; }
    bool use_METIS_NodeNDP() const { return _use_METIS_NodeNDP; }
    bool use_METIS_NodeND() const { return !_use_METIS_NodeNDP; }
    bool use_MUMPS_SYMQAMD() const { return _use_MUMPS_SYMQAMD; }
    bool use_agg_amalg() const { return _use_agg_amalg; }
    int mc64job() const { return _mc64job; }
    bool log_assembly_tree() const { return _log_assembly_tree; }
    bool use_HSS() const { return _use_hss; }
    int HSS_min_front_size() const { return _hss_min_front_size; }
    int HSS_min_sep_size() const { return _hss_min_sep_size; }
    int separator_ordering_level() const { return _sep_order_level; }
    bool indirect_sampling() const { return _indirect_sampling; }
    bool replace_tiny_pivots() const { return _replace_tiny_pivots; }

    const HSS::HSSOptions<scalar_t>& HSS_options() const { return _hss_opts; }
    HSS::HSSOptions<scalar_t>& HSS_options() { return _hss_opts; }

    void set_from_command_line() { set_from_command_line(_argc, _argv); }
    void set_from_command_line(int argc, char* argv[]) {
      std::vector<char*> argv_initial(argc);
      for (int i=0; i<argc; i++) {
        argv_initial[i] = new char[strlen(argv[i])+1];
        strcpy(argv_initial[i], argv[i]);
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
        {"sp_mc64job",                   required_argument, 0, 17},
        {"sp_enable_assembly_tree_log",  no_argument, 0, 18},
        {"sp_disable_assembly_tree_log", no_argument, 0, 19},
        {"sp_enable_hss",                no_argument, 0, 20},
        {"sp_disable_hss",               no_argument, 0, 21},
        {"sp_hss_min_front_size",        required_argument, 0, 22},
        {"sp_hss_min_sep_size",          required_argument, 0, 23},
        {"sp_separator_ordering_level",  required_argument, 0, 24},
        {"sp_enable_indirect_sampling",  no_argument, 0, 25},
        {"sp_disable_indirect_sampling", no_argument, 0, 26},
        {"sp_enable_replace_tiny_pivots", no_argument, 0, 27},
        {"sp_disable_replace_tiny_pivots", no_argument, 0, 28},
        {"sp_verbose",                   no_argument, 0, 'v'},
        {"sp_quiet",                     no_argument, 0, 'q'},
        {"help",                         no_argument, 0, 'h'},
        {NULL, 0, NULL, 0}
      };
      int c, option_index = 0;
      // bool unrecognized_options = false;
      opterr = 0;
      while ((c = getopt_long_only(argc, argv, "hvq",
                                   long_options, &option_index)) != -1) {
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
          iss >> _mc64job;
          set_mc64job(_mc64job);
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
        case 24: {
          std::istringstream iss(optarg);
          iss >> _sep_order_level;
          set_separator_ordering_level(_sep_order_level);
        } break;
        case 25: { enable_indirect_sampling(); } break;
        case 26: { disable_indirect_sampling(); } break;
        case 27: { enable_replace_tiny_pivots(); } break;
        case 28: { disable_replace_tiny_pivots(); } break;
        case 'h': { describe_options(); } break;
        case 'v': set_verbose(true); break;
        case 'q': set_verbose(false); break;
        // case '?': unrecognized_options = true; break;
        default: break;
        }
      }
      // if (unrecognized_options/* && is_root*/)
      //   std::cerr << "# WARNING STRUMPACK: Unrecognized options."
      //             << std::endl;
      HSS_options().set_from_command_line(argc, argv_initial.data());
      for (auto s : argv_initial) delete[] s;
    }

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
      std::cout << "#   --sp_mc64job int [0-5] (default "
                << mc64job() << ")" << std::endl;
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
  };

} // end namespace strumpack

#endif // SPOPTIONS_HPP
