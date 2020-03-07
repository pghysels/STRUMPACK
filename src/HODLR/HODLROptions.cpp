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
#include "HODLROptions.hpp"
#include "StrumpackConfig.hpp"
#if defined(STRUMPACK_USE_GETOPT)
#include <vector>
#include <sstream>
#include <cstring>
#include <getopt.h>
#endif
#include "misc/Tools.hpp"

namespace strumpack {

  namespace HODLR {

    std::string get_name(CompressionAlgorithm a) {
      switch (a) {
      case CompressionAlgorithm::RANDOM_SAMPLING: return "sampling"; break;
      case CompressionAlgorithm::ELEMENT_EXTRACTION: return "extraction"; break;
      default: return "unknown";
      }
    }

    CompressionAlgorithm get_compression_algorithm(const std::string& c) {
      if (c == "sampling")
        return CompressionAlgorithm::RANDOM_SAMPLING;
      else if (c == "extraction")
        return CompressionAlgorithm::ELEMENT_EXTRACTION;
      else {
        std::cerr << "WARNING: Compression algorithm not recognized,"
                  << " setting to 'sampling'."
                  << std::endl;
        return CompressionAlgorithm::RANDOM_SAMPLING;
      }
    }

    template<typename scalar_t> void
    HODLROptions<scalar_t>::set_from_command_line
    (int argc, const char* const* argv) {
#if defined(STRUMPACK_USE_GETOPT)
      std::vector<char*> argv_local(argc);
      for (int i=0; i<argc; i++) {
        argv_local[i] = new char[strlen(argv[i])+1];
        strcpy(argv_local[i], argv[i]);
      }
      option long_options[] =
        {{"hodlr_rel_tol",               required_argument, 0, 1},
         {"hodlr_abs_tol",               required_argument, 0, 2},
         {"hodlr_leaf_size",             required_argument, 0, 3},
         {"hodlr_max_rank",              required_argument, 0, 4},
         {"hodlr_rank_guess",            required_argument, 0, 5},
         {"hodlr_rank_rate",             required_argument, 0, 6},
         {"hodlr_clustering_algorithm",  required_argument, 0, 7},
         {"hodlr_compression_algorithm", required_argument, 0, 8},
         {"hodlr_butterfly_levels",      required_argument, 0, 9},
         {"hodlr_BACA_block_size",       required_argument, 0, 10},
         {"hodlr_BF_sampling_parameter", required_argument, 0, 11},
         {"hodlr_geo",                   required_argument, 0, 12},
         {"hodlr_knn_hodlrbf",           required_argument, 0, 13},
         {"hodlr_knn_lrbf",              required_argument, 0, 14},
         {"hodlr_lr_leaf",               required_argument, 0, 15},
         {"hodlr_verbose",               no_argument, 0, 'v'},
         {"hodlr_quiet",                 no_argument, 0, 'q'},
         {"help",                        no_argument, 0, 'h'},
         {NULL, 0, NULL, 0}
        };
      int c, option_index = 0;
      opterr = optind = 0;
      while ((c = getopt_long_only
              (argc, argv_local.data(),
               "hvq", long_options, &option_index)) != -1) {
        switch (c) {
        case 1: {
          std::istringstream iss(optarg);
          iss >> rel_tol_; set_rel_tol(rel_tol_);
        } break;
        case 2: {
          std::istringstream iss(optarg);
          iss >> abs_tol_; set_abs_tol(abs_tol_);
        } break;
        case 3: {
          std::istringstream iss(optarg);
          iss >> leaf_size_;
          set_leaf_size(leaf_size_);
        } break;
        case 4: {
          std::istringstream iss(optarg);
          iss >> max_rank_;
          set_max_rank(max_rank_);
        } break;
        case 5: {
          std::istringstream iss(optarg);
          iss >> rank_guess_;
          set_rank_guess(rank_guess_);
        } break;
        case 6: {
          std::istringstream iss(optarg);
          iss >> rank_rate_;
          set_rank_rate(rank_rate_);
        } break;
        case 7: {
          std::istringstream iss(optarg);
          std::string s; iss >> s;
          set_clustering_algorithm(get_clustering_algorithm(s));
        } break;
        case 8: {
          std::istringstream iss(optarg);
          std::string s; iss >> s;
          set_compression_algorithm(get_compression_algorithm(s));
        } break;
        case 9: {
          std::istringstream iss(optarg);
          iss >> butterfly_levels_;
          set_butterfly_levels(butterfly_levels_);
        } break;
        case 10: {
          std::istringstream iss(optarg);
          iss >> BACA_block_size_;
          set_BACA_block_size(BACA_block_size_);
        } break;
        case 11: {
          std::istringstream iss(optarg);
          iss >> BF_sampling_parameter_;
          set_BF_sampling_parameter(BF_sampling_parameter_);
        } break;
        case 12: {
          std::istringstream iss(optarg);
          iss >> geo_;
          set_geo(geo_);
        } break;
        case 13: {
          std::istringstream iss(optarg);
          iss >> knn_hodlrbf_;
          set_knn_hodlrbf(knn_hodlrbf_);
        } break;
        case 14: {
          std::istringstream iss(optarg);
          iss >> knn_lrbf_;
          set_knn_lrbf(knn_lrbf_);
        } break;
        case 15: {
          std::istringstream iss(optarg);
          iss >> lr_leaf_;
          set_lr_leaf(lr_leaf_);
        } break;
        case 'v': set_verbose(true); break;
        case 'q': set_verbose(false); break;
        case 'h': describe_options(); break;
        }
      }
      for (auto s : argv_local) delete[] s;
#else
      std::cerr << "WARNING: no support for getopt.h, "
        "not parsing command line options." << std::endl;
#endif
    }

    template<typename scalar_t> void
    HODLROptions<scalar_t>::describe_options() const {
#if defined(STRUMPACK_USE_GETOPT)
      if (!mpi_root()) return;
      std::cout << "# HODLR Options:" << std::endl
                << "#   --hodlr_rel_tol real_t (default "
                << rel_tol() << ")" << std::endl
                << "#   --hodlr_abs_tol real_t (default "
                << abs_tol() << ")" << std::endl
                << "#   --hodlr_leaf_size int (default "
                << leaf_size() << ")" << std::endl
                << "#   --hodlr_max_rank int (default "
                << max_rank() << ")" << std::endl
                << "#   --hodlr_rank_guess int (default "
                << rank_guess() << ")" << std::endl
                << "#   --hodlr_rank_rate double (default "
                << rank_rate() << ")" << std::endl
                << "#   --hodlr_clustering_algorithm natural|2means|kdtree|pca|cobble (default "
                << get_name(clustering_algorithm()) << ")" << std::endl
                << "#   --hodlr_butterfly_levels int (default "
                << butterfly_levels() << ")" << std::endl
                << "#   --hodlr_compression sampling|extraction (default "
                << get_name(compression_algorithm()) << ")" << std::endl
                << "#   --hodlr_BACA_block_size int (default "
                << BACA_block_size() << ")" << std::endl
                << "#   --hodlr_BF_sampling_parameter (default "
                << BF_sampling_parameter() << ")" << std::endl
                << "#   --hodlr_geo 1|2 (1: no neighbor info, 2: use neighbor info) (default "
                << geo() << ")" << std::endl
                << "#   --hodlr_knn_hodlrbf (default "
                << knn_hodlrbf() << ")" << std::endl
                << "#   --hodlr_knn_lrbf (default "
                << knn_lrbf() << ")" << std::endl
                << "#   --hodlr_verbose or -v (default "
                << verbose() << ")" << std::endl
                << "#   --hodlr_quiet or -q (default "
                << !verbose() << ")" << std::endl
                << "#   --help or -h" << std::endl << std::endl;
#endif
    }

    // explicit template instantiations
    template class HODLROptions<float>;
    template class HODLROptions<double>;
    template class HODLROptions<std::complex<float>>;
    template class HODLROptions<std::complex<double>>;

  } // end namespace HODLR
} // end namespace strumpack
