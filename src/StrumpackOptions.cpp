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
#include "StrumpackOptions.hpp"
#include "StrumpackConfig.hpp"
#if defined(STRUMPACK_USE_GETOPT)
#include <vector>
#include <sstream>
#include <cstring>
#include <getopt.h>
#endif
#if defined(STRUMPACK_USE_MPI)
#include "misc/MPIWrapper.hpp"
#endif


namespace strumpack {

  std::string get_name(ReorderingStrategy method) {
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

  bool is_parallel(ReorderingStrategy method) {
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

  std::string get_name(CompressionType comp) {
    switch (comp) {
    case CompressionType::NONE: return "none";
    case CompressionType::HSS: return "hss";
    case CompressionType::BLR: return "blr";
    case CompressionType::HODLR: return "hodlr";
    case CompressionType::LOSSY: return "lossy";
    case CompressionType::LOSSLESS: return "lossless";
    }
    return "UNKNOWN";
  }

  MatchingJob get_matching(int job) {
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

  int get_matching(MatchingJob job) {
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

  std::string get_description(MatchingJob job) {
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


  template<typename scalar_t> void SPOptions<scalar_t>::set_from_command_line
  (int argc, const char* const* argv) {
#if defined(STRUMPACK_USE_GETOPT)
    std::vector<char*> argv_local(argc);
    for (int i=0; i<argc; i++) {
      argv_local[i] = new char[strlen(argv[i])+1];
      strcpy(argv_local[i], argv[i]);
    }
    option long_options[] =
      {{"sp_maxit",                     required_argument, 0, 1},
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
       {"sp_compression",               required_argument, 0, 20},
       {"sp_compression_min_sep_size",  required_argument, 0, 21},
       {"sp_compression_min_front_size", required_argument, 0, 22},
       {"sp_compression_leaf_size",     required_argument, 0, 23},
       {"sp_separator_ordering_level",  required_argument, 0, 24},
       {"sp_enable_indirect_sampling",  no_argument, 0, 25},
       {"sp_disable_indirect_sampling", no_argument, 0, 26},
       {"sp_enable_replace_tiny_pivots", no_argument, 0, 27},
       {"sp_disable_replace_tiny_pivots", no_argument, 0, 28},
       {"sp_nx",                        required_argument, 0, 29},
       {"sp_ny",                        required_argument, 0, 30},
       {"sp_nz",                        required_argument, 0, 31},
       {"sp_components",                required_argument, 0, 32},
       {"sp_separator_width",           required_argument, 0, 33},
       {"sp_write_root_front",          no_argument, 0, 34},
       {"sp_print_root_front_stats",    no_argument, 0, 35},
       {"sp_enable_gpu",                no_argument, 0, 36},
       {"sp_disable_gpu",               no_argument, 0, 37},
       {"sp_cuda_cutoff",               required_argument, 0, 38},
       {"sp_cuda_streams",              required_argument, 0, 39},
       {"sp_lossy_precision",           required_argument, 0, 40},
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
        if (s == "auto") set_Krylov_solver(KrylovSolver::AUTO);
        else if (s == "direct") set_Krylov_solver(KrylovSolver::DIRECT);
        else if (s == "refinement") set_Krylov_solver(KrylovSolver::REFINE);
        else if (s == "pgmres") set_Krylov_solver(KrylovSolver::PREC_GMRES);
        else if (s == "gmres") set_Krylov_solver(KrylovSolver::GMRES);
        else if (s == "pbicgstab") set_Krylov_solver(KrylovSolver::PREC_BICGSTAB);
        else if (s == "bicgstab") set_Krylov_solver(KrylovSolver::BICGSTAB);
        else std::cerr << "# WARNING: Krylov solver not recognized,"
               " using default" << std::endl;
      } break;
      case 5: {
        std::istringstream iss(optarg);
        iss >> _gmres_restart;
        set_gmres_restart(_gmres_restart); } break;
      case 6: {
        std::string s;
        std::istringstream iss(optarg); iss >> s;
        if (s == "modified")
          set_GramSchmidt_type(GramSchmidtType::MODIFIED);
        else if (s == "classical")
          set_GramSchmidt_type(GramSchmidtType::CLASSICAL);
        else std::cerr << "# WARNING: Gram-Schmidt type not recognized,"
               " use 'modified' or classical" << std::endl;
      } break;
      case 7: {
        std::string s; std::istringstream iss(optarg); iss >> s;
        if (s == "natural") set_reordering_method(ReorderingStrategy::NATURAL);
        else if (s == "metis") set_reordering_method(ReorderingStrategy::METIS);
        else if (s == "parmetis") set_reordering_method(ReorderingStrategy::PARMETIS);
        else if (s == "scotch") set_reordering_method(ReorderingStrategy::SCOTCH);
        else if (s == "ptscotch") set_reordering_method(ReorderingStrategy::PTSCOTCH);
        else if (s == "geometric") set_reordering_method(ReorderingStrategy::GEOMETRIC);
        else if (s == "rcm") set_reordering_method(ReorderingStrategy::RCM);
        else std::cerr << "# WARNING: matrix reordering strategy not"
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
      case 20: {
        std::string s; std::istringstream iss(optarg); iss >> s;
        for (auto& c : s) c = std::toupper(c);
        if (s == "NONE") set_compression(CompressionType::NONE);
        else if (s == "HSS") set_compression(CompressionType::HSS);
        else if (s == "BLR") set_compression(CompressionType::BLR);
        else if (s == "HODLR") set_compression(CompressionType::HODLR);
        else if (s == "LOSSY") set_compression(CompressionType::LOSSY);
        else if (s == "LOSSLESS") set_compression(CompressionType::LOSSLESS);
        else std::cerr << "# WARNING: compression type not"
               " recognized, use 'none', 'hss', 'blr', 'hodlr',"
               " 'lossy' or 'lossless'" << std::endl;
      } break;
      case 21: {
        std::istringstream iss(optarg);
        int min_sep;
        iss >> min_sep;
        set_compression_min_sep_size(min_sep);
      } break;
      case 22: {
        std::istringstream iss(optarg);
        int min_front;
        iss >> min_front;
        set_compression_min_front_size(min_front);
      } break;
      case 23: {
        std::istringstream iss(optarg);
        int ls;
        iss >> ls;
        set_compression_leaf_size(ls);
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
      case 29: {
        std::istringstream iss(optarg);
        iss >> _nx;
        set_nx(_nx);
      } break;
      case 30: {
        std::istringstream iss(optarg);
        iss >> _ny;
        set_ny(_ny);
      } break;
      case 31: {
        std::istringstream iss(optarg);
        iss >> _nz;
        set_nz(_nz);
      } break;
      case 32: {
        std::istringstream iss(optarg);
        iss >> _components;
        set_components(_components);
      } break;
      case 33: {
        std::istringstream iss(optarg);
        iss >> _separator_width;
        set_separator_width(_separator_width);
      } break;
      case 34: set_write_root_front(true); break;
      case 35: set_print_root_front_stats(true); break;
      case 36: enable_gpu(); break;
      case 37: disable_gpu(); break;
      case 38: {
        std::istringstream iss(optarg);
        iss >> cuda_cutoff_;
        set_cuda_cutoff(cuda_cutoff_);
      } break;
      case 39: {
        std::istringstream iss(optarg);
        iss >> cuda_streams_;
        set_cuda_streams(cuda_streams_);
      } break;
      case 40: {
        std::istringstream iss(optarg);
        iss >> _lossy_precision;
        set_lossy_precision(_lossy_precision);
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
#if defined(STRUMPACK_USE_BPACK)
    HODLR_options().set_from_command_line(argc, argv);
#endif
#else
    std::cerr << "WARNING: no support for getopt.h, "
      "not parsing command line options." << std::endl;
#endif
  }


  template<typename scalar_t> void
  SPOptions<scalar_t>::describe_options() const {
#if defined(STRUMPACK_USE_GETOPT)
#if defined(STRUMPACK_USE_MPI)
    MPIComm c;
    if (MPIComm::initialized() && !c.is_root()) return;
#endif
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
    std::cout << "#   --sp_Krylov_solver [auto|direct|refinement|pgmres|"
              << "gmres|pbicgstab|bicgstab]" << std::endl;
    std::cout << "#          default: auto (refinement when no HSS, pgmres"
              << " (preconditioned) with HSS compression)" << std::endl;
    std::cout << "#   --sp_gmres_restart int (default " << gmres_restart()
              << ")" << std::endl;
    std::cout << "#          gmres restart length" << std::endl;
    std::cout << "#   --sp_GramSchmidt_type [modified|classical]"
              << std::endl;
    std::cout << "#          Gram-Schmidt type for GMRES" << std::endl;
    std::cout << "#   --sp_reordering_method [natural|metis|scotch|parmetis|"
              << "ptscotch|rcm|geometric]" << std::endl;
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
    std::cout << "#   --sp_compression [none|hss|blr|hodlr|lossy]" << std::endl
              << "#          type of rank-structured compression to use"
              << std::endl;
    std::cout << "#   --sp_compression_min_sep_size (default "
              << compression_min_sep_size() << ")" << std::endl
              << "#          minimum separator size for compression"
              << std::endl;
    std::cout << "#   --sp_compression_leaf_size (default "
              << compression_leaf_size() << ")" << std::endl
              << "#          leaf size for rank-structured representation"
              << std::endl;
    std::cout << "#   --sp_separator_ordering_level (default "
              << separator_ordering_level() << ")" << std::endl;
    std::cout << "#   --sp_enable_indirect_sampling" << std::endl;
    std::cout << "#   --sp_disable_indirect_sampling" << std::endl;
    std::cout << "#   --sp_enable_replace_tiny_pivots" << std::endl;
    std::cout << "#   --sp_disable_replace_tiny_pivots" << std::endl;
    std::cout << "#   --sp_write_root_front" << std::endl;
    std::cout << "#   --sp_print_root_front_stats" << std::endl;
    std::cout << "#   --sp_enable_gpu" << std::endl;
    std::cout << "#   --sp_disable_gpu" << std::endl;
    std::cout << "#   --sp_cuda_cutoff (default "
              << cuda_cutoff() << ")" << std::endl
              << "#          CUDA kernel/CUBLAS cutoff size" << std::endl;
    std::cout << "#   --sp_cuda_streams (default "
              << cuda_streams() << ")" << std::endl
              << "#          number of CUDA streams" << std::endl;
    std::cout << "#   --sp_lossy_precision [1-64] (default "
              << lossy_precision() << ")" << std::endl
              << "#          lossy compression precicion" << std::endl;
    std::cout << "#   --sp_verbose or -v (default " << verbose() << ")"
              << std::endl;
    std::cout << "#   --sp_quiet or -q (default " << !verbose() << ")"
              << std::endl;
    std::cout << "#   --help or -h" << std::endl << std::endl;
#endif
  }

  // explicit template instantiations
  template class SPOptions<float>;
  template class SPOptions<double>;
  template class SPOptions<std::complex<float>>;
  template class SPOptions<std::complex<double>>;

} // end namespace strumpack
