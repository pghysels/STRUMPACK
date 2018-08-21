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
/*! \file HSSOptions.hpp
 * \brief For Pieter to complete
 */
#ifndef HSS_OPTIONS_HPP
#define HSS_OPTIONS_HPP

#include <cstring>
#include <getopt.h>

namespace strumpack {

  /*! HSS namespace. */
  namespace HSS {

    template<typename real_t> inline real_t default_HSS_rel_tol() {
      return real_t(1e-2);
    }
    template<typename real_t> inline real_t default_HSS_abs_tol() {
      return real_t(1e-8);
    }
    template<> inline float default_HSS_rel_tol() {
      return 1e-1;
    }
    template<> inline float default_HSS_abs_tol() {
      return 1e-5;
    }

    enum class CompressionAlgorithm { ORIGINAL, STABLE, HARD_RESTART };
    inline std::string get_name(CompressionAlgorithm a) {
      switch (a) {
      case CompressionAlgorithm::ORIGINAL: return "original"; break;
      case CompressionAlgorithm::STABLE: return "stable"; break;
      case CompressionAlgorithm::HARD_RESTART: return "hard_restart"; break;
      default: return "unknown";
      }
    }

    template<typename scalar_t> class HSSOptions {
      using real_t = typename RealType<scalar_t>::value_type;

    private:
      real_t _rel_tol = default_HSS_rel_tol<real_t>();
      real_t _abs_tol = default_HSS_abs_tol<real_t>();
      int _leaf_size = 128;
      int _d0 = 128;
      int _dd = 64;
      int _p = 10;
      int _max_rank = 5000;
      random::RandomEngine _random_engine =
        random::RandomEngine::LINEAR;
      random::RandomDistribution _random_distribution =
        random::RandomDistribution::NORMAL;
      bool _user_defined_random = false;
      bool _log_ranks = false;
      CompressionAlgorithm _compress_algo = CompressionAlgorithm::STABLE;
      bool _sync = false;
      bool _verbose = true;

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
      void set_d0(int d0) { assert(d0 > 0); _d0 = d0; }
      void set_dd(int dd) { assert(dd > 0); _dd = dd; }
      void set_p(int p) { assert(p >= 0); _p = p; }
      void set_max_rank(int max_rank) {
        assert(max_rank > 0);
        _max_rank = max_rank;
      }
      void set_random_engine(random::RandomEngine random_engine) {
        _random_engine = random_engine;
      }
      void set_random_distribution
      (random::RandomDistribution random_distribution) {
        _random_distribution = random_distribution;
      }
      void set_compression_algorithm(CompressionAlgorithm a) {
        _compress_algo = a;
      }
      void set_user_defined_random(bool user_defined_random) {
        _user_defined_random = user_defined_random;
      }
      void set_synchronized_compression(bool sync) {
        _sync = sync;
      }
      void set_log_ranks(bool log_ranks) { _log_ranks = log_ranks; }
      void set_verbose(bool verbose) { _verbose = verbose; }

      real_t rel_tol() const { return _rel_tol; }
      real_t abs_tol() const { return _abs_tol; }
      int leaf_size() const { return _leaf_size; }
      int d0() const { return _d0; }
      int dd() const { return _dd; }
      int p() const { return _p; }
      int max_rank() const { return _max_rank; }
      random::RandomEngine random_engine() const { return _random_engine; }
      random::RandomDistribution random_distribution() const {
        return _random_distribution;
      }
      CompressionAlgorithm compression_algorithm() const {
        return _compress_algo;
      }
      bool user_defined_random() const { return _user_defined_random; }
      bool synchronized_compression() const { return _sync; }
      bool log_ranks() const { return _log_ranks; }
      bool verbose() const { return _verbose; }

      void set_from_command_line(int argc, const char* const* argv) {
        std::vector<char*> argv_local(argc);
        for (int i=0; i<argc; i++) {
          argv_local[i] = new char[strlen(argv[i])+1];
          strcpy(argv_local[i], argv[i]);
        }
        option long_options[] = {
          {"hss_rel_tol",               required_argument, 0, 1},
          {"hss_abs_tol",               required_argument, 0, 2},
          {"hss_leaf_size",             required_argument, 0, 3},
          {"hss_d0",                    required_argument, 0, 4},
          {"hss_dd",                    required_argument, 0, 5},
          {"hss_p",                     required_argument, 0, 6},
          {"hss_max_rank",              required_argument, 0, 7},
          {"hss_random_distribution",   required_argument, 0, 8},
          {"hss_random_engine",         required_argument, 0, 9},
          {"hss_compression_algorithm", required_argument, 0, 10},
          {"hss_user_defined_random",   no_argument, 0, 11},
          {"hss_enable_sync",           no_argument, 0, 12},
          {"hss_disable_sync",          no_argument, 0, 13},
          {"hss_log_ranks",             no_argument, 0, 14},
          {"hss_verbose",               no_argument, 0, 'v'},
          {"hss_quiet",                 no_argument, 0, 'q'},
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
            iss >> _d0;
            set_d0(_d0);
          } break;
          case 5: {
            std::istringstream iss(optarg);
            iss >> _dd; set_dd(_dd);
          } break;
          case 6: {
            std::istringstream iss(optarg);
            iss >> _p;
            set_p(_p);
          } break;
          case 7: {
            std::istringstream iss(optarg);
            iss >> _max_rank;
            set_max_rank(_max_rank);
          } break;
          case 8: {
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
          case 9: {
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
          case 10: {
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
          case 11: { set_user_defined_random(true); } break;
          case 12: { set_synchronized_compression(true); } break;
          case 13: { set_synchronized_compression(false); } break;
          case 14: { set_log_ranks(true); } break;
          case 'v': set_verbose(true); break;
          case 'q': set_verbose(false); break;
          case 'h': describe_options(); break;
          }
        }
        for (auto s : argv_local) delete[] s;
      }

      void describe_options() const {
        std::cout << "# HSS Options:" << std::endl
                  << "#   --hss_rel_tol real_t (default "
                  << rel_tol() << ")" << std::endl
                  << "#   --hss_abs_tol real_t (default "
                  << abs_tol() << ")" << std::endl
                  << "#   --hss_leaf_size int (default "
                  << leaf_size() << ")" << std::endl
                  << "#   --hss_d0 int (default " << d0() << ")" << std::endl
                  << "#   --hss_dd int (default " << dd() << ")" << std::endl
                  << "#   --hss_p int (default " << p() << ")" << std::endl
                  << "#   --hss_max_rank int (default "
                  << max_rank() << ")" << std::endl
                  << "#   --hss_random_distribution normal|uniform (default "
                  << get_name(random_distribution()) << ")" << std::endl
                  << "#   --hss_random_engine linear|mersenne (default "
                  << get_name(random_engine()) << ")" << std::endl
                  << "#   --hss_compression_algorithm original|stable|hard_restart (default "
                  << get_name(compression_algorithm())<< ")" << std::endl
                  << "#   --hss_user_defined_random (default "
                  << user_defined_random() << ")" << std::endl
                  << "#   --hss_enable_sync (default "
                  << synchronized_compression() << ")" << std::endl
                  << "#   --hss_disable_sync (default "
                  << (!synchronized_compression()) << ")" << std::endl
                  << "#   --hss_log_ranks (default "
                  << log_ranks() << ")" << std::endl
                  << "#   --hss_verbose or -v (default "
                  << verbose() << ")" << std::endl
                  << "#   --hss_quiet or -q (default "
                  << !verbose() << ")" << std::endl
                  << "#   --help or -h" << std::endl;
      }
    };

  } // end namespace HSS
} // end namespace strumpack


#endif // HSS_OPTIONS_HPP
