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
#include "misc/Tools.hpp"


namespace strumpack {

  std::string get_name(ReorderingStrategy method) {
    switch (method) {
    case ReorderingStrategy::NATURAL: return "Natural";
    case ReorderingStrategy::METIS: return "Metis";
    case ReorderingStrategy::SCOTCH: return "Scotch";
    case ReorderingStrategy::GEOMETRIC: return "Geometric";
    case ReorderingStrategy::PARMETIS: return "ParMetis";
    case ReorderingStrategy::PTSCOTCH: return "PTScotch";
    case ReorderingStrategy::RCM: return "RCM";
    }
    return "UNKNOWN";
  }

  bool is_parallel(ReorderingStrategy method) {
    switch (method) {
    case ReorderingStrategy::NATURAL: return false;
    case ReorderingStrategy::METIS: return false;
    case ReorderingStrategy::SCOTCH: return false;
    case ReorderingStrategy::GEOMETRIC: return true;
    case ReorderingStrategy::PARMETIS: return true;
    case ReorderingStrategy::PTSCOTCH: return true;
    case ReorderingStrategy::RCM: return false;
    }
    return false;
  }

  std::string get_name(CompressionType comp) {
    switch (comp) {
    case CompressionType::NONE: return "none";
    case CompressionType::HSS: return "hss";
    case CompressionType::BLR: return "blr";
    case CompressionType::HODLR: return "hodlr";
    case CompressionType::BLR_HODLR: return "blr_hodlr";
    case CompressionType::ZFP_BLR_HODLR: return "zfp_blr_hodlr";
    case CompressionType::LOSSY: return "lossy";
    case CompressionType::LOSSLESS: return "lossless";
    }
    return "UNKNOWN";
  }

  MatchingJob get_matching(int job) {
    if (job < 0 || job > 6)
      std::cerr << "ERROR: Matching job not recognized!!" << std::endl;
    return static_cast<MatchingJob>(job);
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
      return "approximate weighted perfect matching, from CombBLAS";
    }
    return "UNKNOWN";
  }

  std::string get_name(ProportionalMapping pmap) {
    switch (pmap) {
    case ProportionalMapping::FLOPS: return "FLOPS";
    case ProportionalMapping::FACTOR_MEMORY: return "FACTOR_MEMORY";
    case ProportionalMapping::PEAK_MEMORY: return "PEAK_MEMORY";
    }
    return "UNKNOWN";
  }

  template<typename scalar_t> void SPOptions<scalar_t>::set_from_command_line
  (int argc, const char* const* cargv) {
#if defined(STRUMPACK_USE_GETOPT)
    std::vector<std::unique_ptr<char[]>> argv_data(argc);
    std::vector<char*> argv(argc);
    for (int i=0; i<argc; i++) {
      argv_data[i].reset(new char[strlen(cargv[i])+1]);
      argv[i] = argv_data[i].get();
      strcpy(argv[i], cargv[i]);
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
       {"sp_print_compressed_front_stats", no_argument, 0, 35},
       {"sp_enable_gpu",                no_argument, 0, 36},
       {"sp_disable_gpu",               no_argument, 0, 37},
       {"sp_gpu_streams",               required_argument, 0, 38},
       {"sp_lossy_precision",           required_argument, 0, 39},
       {"sp_hss_min_sep_size",          required_argument, 0, 40},
       {"sp_hss_min_front_size",        required_argument, 0, 41},
       {"sp_hodlr_min_sep_size",        required_argument, 0, 42},
       {"sp_hodlr_min_front_size",      required_argument, 0, 43},
       {"sp_blr_min_sep_size",          required_argument, 0, 44},
       {"sp_blr_min_front_size",        required_argument, 0, 45},
       {"sp_lossy_min_sep_size",        required_argument, 0, 46},
       {"sp_lossy_min_front_size",      required_argument, 0, 47},
       {"sp_nd_planar_levels",          required_argument, 0, 48},
       {"sp_proportional_mapping",      required_argument, 0, 49},
       {"sp_verbose",                   no_argument, 0, 'v'},
       {"sp_quiet",                     no_argument, 0, 'q'},
       {"help",                         no_argument, 0, 'h'},
       {NULL, 0, NULL, 0}
      };
    int c, option_index = 0;
    // bool unrecognized_options = false;
    opterr = optind = 0;
    while ((c = getopt_long_only
            (argc, argv.data(), "hvq",
             long_options, &option_index)) != -1) {
      switch (c) {
      case 1: {
        std::istringstream iss(optarg);
        iss >> maxit_;
        set_maxit(maxit_);
      } break;
      case 2: {
        std::istringstream iss(optarg);
        iss >> rel_tol_;
        set_rel_tol(rel_tol_);
      } break;
      case 3: {
        std::istringstream iss(optarg);
        iss >> abs_tol_;
        set_abs_tol(abs_tol_);
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
        iss >> gmres_restart_;
        set_gmres_restart(gmres_restart_); } break;
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
        iss >> nd_param_;
        set_nd_param(nd_param_);
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
        else if (s == "BLR_HODLR") set_compression(CompressionType::BLR_HODLR);
        else if (s == "ZFP_BLR_HODLR") set_compression(CompressionType::ZFP_BLR_HODLR);
        else if (s == "LOSSY") set_compression(CompressionType::LOSSY);
        else if (s == "LOSSLESS") set_compression(CompressionType::LOSSLESS);
        else std::cerr << "# WARNING: compression type not"
               " recognized, use 'none', 'hss', 'blr', 'hodlr',"
               " 'blr_hodlr', 'zfp_blr_hodlr', 'lossy' or 'lossless'" << std::endl;
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
        iss >> sep_order_level_;
        set_separator_ordering_level(sep_order_level_);
      } break;
      case 25: { enable_indirect_sampling(); } break;
      case 26: { disable_indirect_sampling(); } break;
      case 27: { enable_replace_tiny_pivots(); } break;
      case 28: { disable_replace_tiny_pivots(); } break;
      case 29: {
        std::istringstream iss(optarg);
        iss >> nx_;
        set_nx(nx_);
      } break;
      case 30: {
        std::istringstream iss(optarg);
        iss >> ny_;
        set_ny(ny_);
      } break;
      case 31: {
        std::istringstream iss(optarg);
        iss >> nz_;
        set_nz(nz_);
      } break;
      case 32: {
        std::istringstream iss(optarg);
        iss >> components_;
        set_components(components_);
      } break;
      case 33: {
        std::istringstream iss(optarg);
        iss >> separator_width_;
        set_separator_width(separator_width_);
      } break;
      case 34: set_write_root_front(true); break;
      case 35: set_print_compressed_front_stats(true); break;
      case 36: enable_gpu(); break;
      case 37: disable_gpu(); break;
      case 38: {
        std::istringstream iss(optarg);
        iss >> gpu_streams_;
        set_gpu_streams(gpu_streams_);
      } break;
      case 39: {
        std::istringstream iss(optarg);
        iss >> lossy_precision_;
        set_lossy_precision(lossy_precision_);
      } break;
      case 40: {
        std::istringstream iss(optarg);
        int min_sep;
        iss >> min_sep;
        set_hss_min_sep_size(min_sep);
      } break;
      case 41: {
        std::istringstream iss(optarg);
        int min_front;
        iss >> min_front;
        set_hss_min_front_size(min_front);
      } break;
      case 42: {
        std::istringstream iss(optarg);
        int min_sep;
        iss >> min_sep;
        set_hodlr_min_sep_size(min_sep);
      } break;
      case 43: {
        std::istringstream iss(optarg);
        int min_front;
        iss >> min_front;
        set_hodlr_min_front_size(min_front);
      } break;
      case 44: {
        std::istringstream iss(optarg);
        int min_sep;
        iss >> min_sep;
        set_blr_min_sep_size(min_sep);
      } break;
      case 45: {
        std::istringstream iss(optarg);
        int min_front;
        iss >> min_front;
        set_blr_min_front_size(min_front);
      } break;
      case 46: {
        std::istringstream iss(optarg);
        int min_sep;
        iss >> min_sep;
        set_lossy_min_sep_size(min_sep);
      } break;
      case 47: {
        std::istringstream iss(optarg);
        int min_front;
        iss >> min_front;
        set_lossy_min_front_size(min_front);
      } break;
      case 48: {
        std::istringstream iss(optarg);
        iss >> nd_planar_levels_;
        set_nd_planar_levels(nd_planar_levels_);
      } break;
      case 49: {
        std::string s; std::istringstream iss(optarg); iss >> s;
        for (auto& c : s) c = std::toupper(c);
        if (s == "FLOPS") set_proportional_mapping(ProportionalMapping::FLOPS);
        else if (s == "FACTOR_MEMORY") set_proportional_mapping(ProportionalMapping::FACTOR_MEMORY);
        else if (s == "PEAK_MEMORY") set_proportional_mapping(ProportionalMapping::PEAK_MEMORY);
        else std::cerr << "# WARNING: proportional-mapping type not"
               " recognized, use 'FLOPS', 'FACTOR_MEMORY', 'PEAK_MEMORY'"
                       << std::endl;
      } break;
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
    HSS_options().set_from_command_line(argc, cargv);
    BLR_options().set_from_command_line(argc, cargv);
#if defined(STRUMPACK_USE_BPACK)
    HODLR_options().set_from_command_line(argc, cargv);
#endif
#else
    std::cerr << "WARNING: no support for getopt.h, "
      "not parsing command line options." << std::endl;
#endif
  }


  template<typename scalar_t> void
  SPOptions<scalar_t>::describe_options() const {
#if defined(STRUMPACK_USE_GETOPT)
    if (!mpi_root()) return;
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
    std::cout << "#          default: auto (refinement when using compression, pgmres"
              << " (preconditioned) with compression)" << std::endl;
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
    std::cout << "#   --sp_nd_param int (default " << nd_param() << ")"
              << std::endl;
    std::cout << "#   --sp_nd_planar_levels int (default "
              << nd_planar_levels() << ")" << std::endl;
    std::cout << "#   --sp_nx int (default " << nx() << ")"
              << std::endl;
    std::cout << "#   --sp_ny int (default " << ny() << ")"
              << std::endl;
    std::cout << "#   --sp_nz int (default " << nz() << ")"
              << std::endl;
    std::cout << "#   --sp_components int (default " << components() << ")"
              << std::endl;
    std::cout << "#   --sp_separator_width int (default "
              << separator_width() << ")" << std::endl;
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
              << static_cast<int>(matching()) << ")" << std::endl;
    for (int i=0; i<7; i++)
      std::cout << "#      " << i << " " <<
        get_description(get_matching(i)) << std::endl;
    std::cout << "#   --sp_compression (default "
              << get_name(comp_) << ")" << std::endl 
              << "#          should be [none|hss|blr|hodlr|lossy|blr_hodlr|zfp_blr_hodlr]" << std::endl
              << "#          type of rank-structured compression to use"
              << std::endl;
    std::cout << "#   --sp_compression_min_sep_size (default "
              << compression_min_sep_size() << ")" << std::endl
              << "#          minimum separator size for compression"
              << std::endl;
    std::cout << "#   --sp_compression_min_front_size (default "
              << compression_min_front_size() << ")" << std::endl
              << "#          minimum front size for compression"
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
    std::cout << "#   --sp_print_compressed_front_stats" << std::endl;
    std::cout << "#   --sp_proportional_mapping (default "
              << get_name(prop_map_) << ")" << std::endl
              << "#          should be [FLOPS|FACTOR_MEMORY|PEAK_MEMORY]" << std::endl
              << "#          type of proportional mapping"
              << std::endl;
    std::cout << "#   --sp_enable_gpu" << std::endl;
    std::cout << "#   --sp_disable_gpu" << std::endl;
    std::cout << "#   --sp_gpu_streams (default "
              << gpu_streams() << ")" << std::endl
              << "#          number of GPU streams" << std::endl;
    std::cout << "#   --sp_lossy_precision [1-64] (default "
              << lossy_precision() << ")" << std::endl
              << "#          lossy compression precision" << std::endl
              << "#          (for lossless use <= 0)" << std::endl;
    std::cout << "#   --sp_hss_min_sep_size (default "
              << hss_min_sep_size() << ")" << std::endl
              << "#          minimum separator size for hss compression"
              << std::endl;
    std::cout << "#   --sp_hss_min_front_size (default "
              << hss_min_front_size() << ")" << std::endl
              << "#          minimum front size for hss compression"
              << std::endl;
    std::cout << "#   --sp_hodlr_min_sep_size (default "
              << hodlr_min_sep_size() << ")" << std::endl
              << "#          minimum separator size for hodlr compression"
              << std::endl;
    std::cout << "#   --sp_hodlr_min_front_size (default "
              << hodlr_min_front_size() << ")" << std::endl
              << "#          minimum front size for hodlr compression"
              << std::endl;
    std::cout << "#   --sp_blr_min_sep_size (default "
              << blr_min_sep_size() << ")" << std::endl
              << "#          minimum separator size for blr compression"
              << std::endl;
    std::cout << "#   --sp_blr_min_front_size (default "
              << blr_min_front_size() << ")" << std::endl
              << "#          minimum front size for blr compression"
              << std::endl;
    std::cout << "#   --sp_lossy_min_sep_size (default "
              << lossy_min_sep_size() << ")" << std::endl
              << "#          minimum separator size for lossy compression"
              << std::endl;
    std::cout << "#   --sp_lossy_min_front_size (default "
              << lossy_min_front_size() << ")" << std::endl
              << "#          minimum front size for lossy compression"
              << std::endl;
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
