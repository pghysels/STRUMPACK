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
#include "BLROptions.hpp"
#include "StrumpackConfig.hpp"
#if defined(STRUMPACK_USE_GETOPT)
#include <vector>
#include <sstream>
#include <cstring>
#include <getopt.h>
#endif
#include "misc/Tools.hpp"

namespace strumpack {

  namespace BLR {

    std::string get_name(LowRankAlgorithm a) {
      switch (a) {
      case LowRankAlgorithm::RRQR: return "RRQR";
      case LowRankAlgorithm::ACA: return "ACA";
      case LowRankAlgorithm::BACA: return "BACA";
      default: return "unknown";
      }
    }

    std::string get_name(Admissibility a) {
      switch (a) {
      case Admissibility::STRONG: return "strong";
      case Admissibility::WEAK: return "weak";
      default: return "unknown";
      }
    }

    std::string get_name(BLRFactorAlgorithm a) {
      switch (a) {
      case BLRFactorAlgorithm::COLWISE: return "COLWISE";
      case BLRFactorAlgorithm::RL: return "RL";
      case BLRFactorAlgorithm::LL: return "LL";
      case BLRFactorAlgorithm::COMB: return "Comb";
      case BLRFactorAlgorithm::STAR: return "Star";
      default: return "unknown";
      }
    }

    std::string get_name(CompressionKernel a) {
      switch (a) {
      case CompressionKernel::FULL: return "full";
      case CompressionKernel::HALF: return "half";
      default: return "unknown";
      }
    }

    template<typename scalar_t> void
    BLROptions<scalar_t>::set_from_command_line
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
        {{"blr_rel_tol",               required_argument, 0, 1},
         {"blr_abs_tol",               required_argument, 0, 2},
         {"blr_leaf_size",             required_argument, 0, 3},
         {"blr_max_rank",              required_argument, 0, 4},
         {"blr_low_rank_algorithm",    required_argument, 0, 5},
         {"blr_admissibility",         required_argument, 0, 6},
         {"blr_BACA_blocksize",        required_argument, 0, 7},
         {"blr_factor_algorithm",      required_argument, 0, 8},
         {"blr_compression_kernel",    required_argument, 0, 9},
         {"blr_verbose",               no_argument, 0, 'v'},
         {"blr_quiet",                 no_argument, 0, 'q'},
         {"help",                      no_argument, 0, 'h'},
         {NULL, 0, NULL, 0}};
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
          iss >> this->max_rank_;
          this->set_max_rank(this->max_rank_);
        } break;
        case 5: {
          std::istringstream iss(optarg);
          std::string s; iss >> s;
          if (s == "RRQR")
            set_low_rank_algorithm(LowRankAlgorithm::RRQR);
          else if (s == "ACA")
            set_low_rank_algorithm(LowRankAlgorithm::ACA);
          else if (s == "BACA")
            set_low_rank_algorithm(LowRankAlgorithm::BACA);
          else
            std::cerr << "# WARNING: low-rank algorithm not"
                      << " recognized, use 'RRQR', 'ACA' or 'BACA'."
                      << std::endl;
        } break;
        case 6: {
          std::istringstream iss(optarg);
          std::string s; iss >> s;
          if (s == "weak")
            set_admissibility(Admissibility::WEAK);
          else if (s == "strong")
            set_admissibility(Admissibility::STRONG);
          else
            std::cerr << "# WARNING: admisibility not recognized"
                      << ", use 'weak' or 'strong'."
                      << std::endl;
        } break;
        case 7: {
          std::istringstream iss(optarg);
          iss >> BACA_blocksize_;
          set_BACA_blocksize(BACA_blocksize_);
        } break;
        case 8: {
          std::istringstream iss(optarg);
          std::string s; iss >> s;
          if (s == "COLWISE")
            set_BLR_factor_algorithm(BLRFactorAlgorithm::COLWISE);
          else if (s == "RL")
            set_BLR_factor_algorithm(BLRFactorAlgorithm::RL);
          else if (s == "LL")
            set_BLR_factor_algorithm(BLRFactorAlgorithm::LL);
          else if (s == "Comb")
            set_BLR_factor_algorithm(BLRFactorAlgorithm::COMB);
          else if (s == "Star")
            set_BLR_factor_algorithm(BLRFactorAlgorithm::STAR);
          else
            std::cerr << "# WARNING: BLR algorithm not"
                      << " recognized, use 'COLWISE', 'RL', 'LL', 'Comb' or 'Star'."
                      << std::endl;
        } break;
        case 9: {
          std::istringstream iss(optarg);
          std::string s; iss >> s;
          if (s == "full")
            set_compression_kernel(CompressionKernel::FULL);
          else if (s == "half")
            set_compression_kernel(CompressionKernel::HALF);
          else
            std::cerr << "# WARNING: compression kernel not"
                      << " recognized, use 'full' or 'half'."
                      << std::endl;
        } break;
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
    BLROptions<scalar_t>::describe_options() const {
#if defined(STRUMPACK_USE_GETOPT)
      if (!mpi_root()) return;
      std::cout << "# BLR Options:" << std::endl
                << "#   --blr_rel_tol real_t (default "
                << this->rel_tol() << ")" << std::endl
                << "#   --blr_abs_tol real_t (default "
                << this->abs_tol() << ")" << std::endl
                << "#   --blr_leaf_size int (default "
                << this->leaf_size() << ")" << std::endl
                << "#   --blr_max_rank int (default "
                << this->max_rank() << ")" << std::endl
                << "#   --blr_low_rank_algorithm (default "
                << get_name(lr_algo_) << ")" << std::endl
                << "#      should be [RRQR|ACA|BACA]" << std::endl
                << "#   --blr_admissibility (default "
                << get_name(adm_) << ")" << std::endl
                << "#      should be one of [weak|strong]" << std::endl
                << "#   --blr_factor_algorithm (default "
                << get_name(blr_algo_) << ")" << std::endl
                << "#      should be [COLWISE|RL|LL|Comb|Star]" << std::endl
                << "#   --blr_compression_kernel (default "
                << get_name(crn_krnl_) << ")" << std::endl
                << "#      should be [full|half]" << std::endl
                << "#   --blr_BACA_blocksize int (default "
                << BACA_blocksize() << ")" << std::endl
                << "#   --blr_verbose or -v (default "
                << this->verbose() << ")" << std::endl
                << "#   --blr_quiet or -q (default "
                << !this->verbose() << ")" << std::endl
                << "#   --help or -h" << std::endl << std::endl;
#endif
    }

    // explicit template instantiations
    template class BLROptions<float>;
    template class BLROptions<double>;
    template class BLROptions<std::complex<float>>;
    template class BLROptions<std::complex<double>>;

  } // end namespace BLR
} // end namespace strumpack
