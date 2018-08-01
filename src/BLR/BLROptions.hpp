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
/*! \file BLROptions.hpp
 * \brief For Pieter to complete
 */
#ifndef BLR_OPTIONS_HPP
#define BLR_OPTIONS_HPP

#include <cstring>
#include <getopt.h>

namespace strumpack {

  /*! BLR namespace. */
  namespace BLR {

    template<typename real_t> inline real_t default_BLR_rel_tol() {
      return real_t(1e-4);
    }
    template<typename real_t> inline real_t default_BLR_abs_tol() {
      return real_t(1e-10);
    }
    template<> inline float default_BLR_rel_tol() {
      return 1e-2;
    }
    template<> inline float default_BLR_abs_tol() {
      return 1e-5;
    }

    enum class LowRankAlgorithm { RRQR, ACA };
    inline std::string get_name(LowRankAlgorithm a) {
      switch (a) {
      case LowRankAlgorithm::RRQR: return "RRQR"; break;
      case LowRankAlgorithm::ACA: return "ACA"; break;
      default: return "unknown";
      }
    }

    template<typename scalar_t> class BLROptions {
      using real_t = typename RealType<scalar_t>::value_type;

    private:
      real_t _rel_tol = default_BLR_rel_tol<real_t>();
      real_t _abs_tol = default_BLR_abs_tol<real_t>();
      int _leaf_size = 128;
      int _max_rank = 5000;
      bool _verbose = true;
      LowRankAlgorithm _lr_algo = LowRankAlgorithm::RRQR;

    public:
      /*! \brief For Pieter to complete
       * \param rel_tol
       */
      void set_rel_tol(real_t rel_tol) {
        assert(rel_tol <= real_t(1.) && rel_tol >= real_t(0.));
        _rel_tol = rel_tol;
      }
      void set_abs_tol(real_t abs_tol) {
        assert(abs_tol >= real_t(0.));
        _abs_tol = abs_tol;
      }
      /*! \brief For Pieter to complete
       * \param leaf_size
       */
      void set_leaf_size(int leaf_size) {
        assert(_leaf_size > 0);
        _leaf_size = leaf_size;
      }
      void set_max_rank(int max_rank) {
        assert(max_rank > 0);
        _max_rank = max_rank;
      }
      void set_low_rank_algorithm(LowRankAlgorithm a) {
        _lr_algo = a;
      }
      void set_verbose(bool verbose) { _verbose = verbose; }

      real_t rel_tol() const { return _rel_tol; }
      real_t abs_tol() const { return _abs_tol; }
      int leaf_size() const { return _leaf_size; }
      int max_rank() const { return _max_rank; }
      LowRankAlgorithm low_rank_algorithm() const { return _lr_algo; }
      bool verbose() const { return _verbose; }

      void set_from_command_line(int argc, const char* const* argv) {
        std::vector<char*> argv_local(argc);
        for (int i=0; i<argc; i++) {
          argv_local[i] = new char[strlen(argv[i])+1];
          strcpy(argv_local[i], argv[i]);
        }
        option long_options[] = {
          {"blr_rel_tol",               required_argument, 0, 1},
          {"blr_abs_tol",               required_argument, 0, 2},
          {"blr_leaf_size",             required_argument, 0, 3},
          {"blr_max_rank",              required_argument, 0, 4},
          {"blr_low_rank_algorithm",    required_argument, 0, 5},
          {"blr_verbose",               no_argument, 0, 'v'},
          {"blr_quiet",                 no_argument, 0, 'q'},
          {"help",                      no_argument, 0, 'h'},
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
            iss >> _rel_tol; set_rel_tol(_rel_tol);
          } break;
          case 2: {
            std::istringstream iss(optarg);
            iss >> _abs_tol; set_abs_tol(_abs_tol);
          } break;
          case 3: {
            std::istringstream iss(optarg);
            iss >> _leaf_size;
            set_leaf_size(_leaf_size);
          } break;
          case 4: {
            std::istringstream iss(optarg);
            iss >> _max_rank;
            set_max_rank(_max_rank);
          } break;
          case 5: {
            std::istringstream iss(optarg);
            std::string s; iss >> s;
            if (s.compare("RRQR") == 0)
              set_low_rank_algorithm(LowRankAlgorithm::RRQR);
            else if (s.compare("ACA") == 0)
              set_low_rank_algorithm(LowRankAlgorithm::ACA);
            else
              std::cerr << "# WARNING: low-rank algorithm not"
                        << " recognized, use 'RRQR' or 'ACA'."
                        << std::endl;
          } break;

          case 'v': set_verbose(true); break;
          case 'q': set_verbose(false); break;
          case 'h': describe_options(); break;
          }
        }
        for (auto s : argv_local) delete[] s;
      }

      void describe_options() const {
        std::cout << "# BLR Options:" << std::endl
                  << "#   --blr_rel_tol real_t (default "
                  << rel_tol() << ")" << std::endl
                  << "#   --blr_abs_tol real_t (default "
                  << abs_tol() << ")" << std::endl
                  << "#   --blr_leaf_size int (default "
                  << leaf_size() << ")" << std::endl
                  << "#   --blr_max_rank int (default "
                  << max_rank() << ")" << std::endl
                  << "#   --blr_low_rank_algorithm (default "
                  << get_name(_lr_algo) << ")" << std::endl
                  << "       should be one of [RRQR|ACA]" << std::endl
                  << "#   --blr_verbose or -v (default "
                  << verbose() << ")" << std::endl
                  << "#   --blr_quiet or -q (default "
                  << !verbose() << ")" << std::endl
                  << "#   --help or -h" << std::endl;
      }
    };

  } // end namespace BLR
} // end namespace strumpack


#endif // BLR_OPTIONS_HPP
