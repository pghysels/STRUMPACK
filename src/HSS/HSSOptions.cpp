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
#include "HSSOptions.hpp"
#include "StrumpackConfig.hpp"
#if defined(STRUMPACK_USE_GETOPT)
#include <vector>
#include <sstream>
#include <cstring>
#include <getopt.h>
#endif
#include "misc/Tools.hpp"

namespace strumpack {

  namespace HSS {

    std::string get_name(CompressionAlgorithm a) {
      switch (a) {
      case CompressionAlgorithm::ORIGINAL: return "original";
      case CompressionAlgorithm::STABLE: return "stable";
      case CompressionAlgorithm::HARD_RESTART: return "hard_restart";
      default: return "unknown";
      }
    }

    std::string get_name(CompressionSketch a) {
      switch (a) {
      case CompressionSketch::GAUSSIAN: return "Gaussian";
      case CompressionSketch::SJLT: return "SJLT";
      default: return "unknown";
      }
    }

    std::string get_name(SJLTAlgo a) {
      switch (a) {
      case SJLTAlgo::CHUNK: return "chunking SJLT sampling ";
      case SJLTAlgo::PERM: return "permutation SJLT sampling ";
      default: return "unknown";
      }
    }

    template<typename scalar_t> void
    HSSOptions<scalar_t>::set_from_command_line
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
        {{"hss_rel_tol",               required_argument, 0, 1},
         {"hss_abs_tol",               required_argument, 0, 2},
         {"hss_leaf_size",             required_argument, 0, 3},
         {"hss_d0",                    required_argument, 0, 4},
         {"hss_dd",                    required_argument, 0, 5},
         {"hss_p",                     required_argument, 0, 6},
         {"hss_nnz0",                  required_argument, 0, 7},
         {"hss_nnz",                   required_argument, 0, 8},
         {"hss_max_rank",              required_argument, 0, 9},
         {"hss_random_distribution",   required_argument, 0, 10},
         {"hss_random_engine",         required_argument, 0, 11},
         {"hss_compression_algorithm", required_argument, 0, 12},
         {"hss_compression_sketch",    required_argument, 0, 13},
         {"hss_SJLT_algo",             required_argument, 0, 14},
         {"hss_clustering_algorithm",  required_argument, 0, 15},
         {"hss_approximate_neighbors", required_argument, 0, 16},
         {"hss_ann_iterations",        required_argument, 0, 17},
         {"hss_user_defined_random",   no_argument, 0, 18},
         {"hss_enable_sync",           no_argument, 0, 19},
         {"hss_disable_sync",          no_argument, 0, 20},
         {"hss_log_ranks",             no_argument, 0, 21},
         {"hss_verbose",               no_argument, 0, 'v'},
         {"hss_quiet",                 no_argument, 0, 'q'},
         {"help",                      no_argument, 0, 'h'},
         {NULL, 0, NULL, 0}
      };
      int c, option_index = 0;
      opterr = optind = 0;
      while ((c = getopt_long_only
              (argc, argv.data(), "hvq",
               long_options, &option_index)) != -1) {
        switch (c) {
        case 1: {
          std::istringstream iss(optarg);
          iss >> this->rel_tol_;
          this->set_rel_tol(this->rel_tol_);
        } break;
        case 2: {
          std::istringstream iss(optarg);
          iss >> this->abs_tol_;
          this->set_abs_tol(this->abs_tol_);
        } break;
        case 3: {
          std::istringstream iss(optarg);
          iss >> this->leaf_size_;
          this->set_leaf_size(this->leaf_size_);
        } break;
        case 4: {
          std::istringstream iss(optarg);
          iss >> d0_; set_d0(d0_);
        } break;
        case 5: {
          std::istringstream iss(optarg);
          iss >> dd_; set_dd(dd_);
        } break;
        case 6: {
          std::istringstream iss(optarg);
          iss >> p_; set_p(p_);
        } break;
        case 7: {
          std::istringstream iss(optarg);
          iss >> nnz0_; set_nnz0(nnz0_);
        } break;
        case 8: {
          std::istringstream iss(optarg);
          iss >> nnz_; set_nnz(nnz_);
        } break;
        case 9: {
          std::istringstream iss(optarg);
          iss >> this->max_rank_;
          this->set_max_rank(this->max_rank_);
        } break;
        case 10: {
          std::istringstream iss(optarg);
          std::string s; iss >> s;
          if (s.compare("normal") == 0)
            set_random_distribution(random::RandomDistribution::NORMAL);
          else if (s.compare("uniform") == 0)
            set_random_distribution(random::RandomDistribution::UNIFORM);
          else
            std::cerr << "# WARNING: random number distribution not"
                      << " recognized, use 'normal' or 'uniform'."
                      << std::endl;
        } break;
        case 11: {
          std::istringstream iss(optarg);
          std::string s; iss >> s;
          if (s.compare("linear") == 0)
            set_random_engine(random::RandomEngine::LINEAR);
          else if (s.compare("mersenne") == 0)
            set_random_engine(random::RandomEngine::MERSENNE);
          else
            std::cerr << "# WARNING: random number engine not recognized,"
                      << " use 'linear' or 'mersenne'." << std::endl;
        } break;
        case 12: {
          std::istringstream iss(optarg);
          std::string s; iss >> s;
          if (s.compare("original") == 0)
            set_compression_algorithm(CompressionAlgorithm::ORIGINAL);
          else if (s.compare("stable") == 0)
            set_compression_algorithm(CompressionAlgorithm::STABLE);
          else if (s.compare("hard_restart") == 0)
            set_compression_algorithm(CompressionAlgorithm::HARD_RESTART);
          else
            std::cerr << "# WARNING: compression algorithm not recognized,"
                      << " use 'original', 'stable' or 'hard_restart'."
                      << std::endl;
        } break;
        case 13: {
          std::istringstream iss(optarg);
          std::string s; iss >> s;
          if (s.compare("Gaussian") == 0)
            set_compression_sketch(CompressionSketch::GAUSSIAN);
          else if (s.compare("SJLT") == 0)
            set_compression_sketch(CompressionSketch::SJLT);
          else
            std::cerr << "# WARNING: compression sketch not recognized,"
                      << " use 'Gaussian', or 'SJLT'."
                      << std::endl;
        } break;
        case 14: {
          std::istringstream iss(optarg);
          std::string s; iss >> s;
          if (s.compare("chunk") == 0)
            set_SJLT_algo(SJLTAlgo::CHUNK);
          else if (s.compare("perm") == 0)
            set_SJLT_algo(SJLTAlgo::PERM);
          else
            std::cerr << "# WARNING: compression sketch not recognized,"
                      << " use 'chunk', or 'perm'."
                      << std::endl;
        } break;
        case 15: {
          std::istringstream iss(optarg);
          std::string s; iss >> s;
          set_clustering_algorithm(get_clustering_algorithm(s));
        } break;
        case 16: {
          std::istringstream iss(optarg);
          iss >> approximate_neighbors_;
          set_approximate_neighbors(approximate_neighbors_);
        } break;
        case 17: {
          std::istringstream iss(optarg);
          iss >> ann_iterations_;
          set_ann_iterations(ann_iterations_);
        } break;
        case 18: { set_user_defined_random(true); } break;
        case 19: { set_synchronized_compression(true); } break;
        case 20: { set_synchronized_compression(false); } break;
        case 21: { set_log_ranks(true); } break;
        case 'v': this->set_verbose(true); break;
        case 'q': this->set_verbose(false); break;
        case 'h': describe_options(); break;
        }
      }
#else
      std::cerr << "WARNING: no support for getopt.h, "
        "not parsing command line options." << std::endl;
#endif
    }

    template<typename scalar_t> void
    HSSOptions<scalar_t>::describe_options() const {
#if defined(STRUMPACK_USE_GETOPT)
      if (!mpi_root()) return;
      std::cout << "# HSS Options:" << std::endl
                << "#   --hss_rel_tol real_t (default "
                << this->rel_tol() << ")" << std::endl
                << "#   --hss_abs_tol real_t (default "
                << this->abs_tol() << ")" << std::endl
                << "#   --hss_leaf_size int (default "
                << this->leaf_size() << ")" << std::endl
                << "#   --hss_d0 int (default " << d0() << ")" << std::endl
                << "#   --hss_dd int (default " << dd() << ")" << std::endl
                << "#   --hss_p int (default " << p() << ")" << std::endl
                << "#   --hss_nnz0 int (default " << nnz0() << ")" << std::endl
                << "#   --hss_nnz int (default " << nnz() << ")" << std::endl
                << "#   --hss_max_rank int (default "
                << this->max_rank() << ")" << std::endl
                << "#   --hss_random_distribution normal|uniform (default "
                << get_name(random_distribution()) << ")" << std::endl
                << "#   --hss_random_engine linear|mersenne (default "
                << get_name(random_engine()) << ")" << std::endl
                << "#   --hss_compression_algorithm original|stable|hard_restart (default "
                << get_name(compression_algorithm()) << ")" << std::endl
                << "#   --hss_compression_sketch Gaussian|SJLT (default "
                << get_name(compression_sketch()) << ")" << std::endl
                << "#   --hss_SJLT_algo chunk|perm (default "
                << get_name(SJLT_algo()) << ")" << std::endl
                << "#   --hss_clustering_algorithm natural|2means|kdtree|pca|cobble (default "
                << get_name(clustering_algorithm()) << ")" << std::endl
                << "#   --hss_user_defined_random (default "
                << user_defined_random() << ")" << std::endl
                << "#   --hss_approximate_neighbors int (default "
                << approximate_neighbors() << ")" << std::endl
                << "#   --hss_ann_iterations int (default "
                << ann_iterations() << ")" << std::endl
                << "#   --hss_enable_sync (default "
                << synchronized_compression() << ")" << std::endl
                << "#   --hss_disable_sync (default "
                << (!synchronized_compression()) << ")" << std::endl
                << "#   --hss_log_ranks (default "
                << log_ranks() << ")" << std::endl
                << "#   --hss_verbose or -v (default "
                << this->verbose() << ")" << std::endl
                << "#   --hss_quiet or -q (default "
                << !this->verbose() << ")" << std::endl
                << "#   --help or -h" << std::endl << std::endl;
#endif
    }

    // explicit template instantiations
    template class HSSOptions<float>;
    template class HSSOptions<double>;
    template class HSSOptions<std::complex<float>>;
    template class HSSOptions<std::complex<double>>;

  } // end namespace HSS
} // end namespace strumpack
