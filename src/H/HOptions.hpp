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
 *
 */
#ifndef H_OPTIONS_HPP
#define H_OPTIONS_HPP
#include <getopt.h>

namespace strumpack {
  namespace H {

    template<typename real_t> inline real_t default_H_rel_tol() {
      return real_t(1e-4);
    }
    template<typename real_t> inline real_t default_H_abs_tol() {
      return real_t(1e-12);
    }
    template<> inline float default_H_rel_tol() {
      return 1e-2;
    }
    template<> inline float default_H_abs_tol() {
      return 1e-5;
    }

    template<typename scalar_t> class HOptions {
      using real_t = typename RealType<scalar_t>::value_type;

    private:
      real_t _rel_tol = default_H_rel_tol<real_t>();
      real_t _abs_tol = default_H_abs_tol<real_t>();
      int _leaf_size = 128;
      int _max_rank = 5000;
      bool _log_ranks = false;
      bool _verbose = true;

    public:
      void set_rel_tol(real_t rel_tol) {
        assert(rel_tol <= real_t(1.) && rel_tol >= real_t(0.));
        _rel_tol = rel_tol;
      }
      void set_abs_tol(real_t abs_tol) {
        assert(abs_tol >= real_t(0.));
        _abs_tol = abs_tol;
      }
      void set_leaf_size(int leaf_size) {
        assert(_leaf_size > 0);
        _leaf_size = leaf_size;
      }
      void set_max_rank(int max_rank) {
        assert(max_rank > 0);
        _max_rank = max_rank;
      }
      void set_log_ranks(bool log_ranks) { _log_ranks = log_ranks; }
      void set_verbose(bool verbose) { _verbose = verbose; }

      real_t rel_tol() const { return _rel_tol; }
      real_t abs_tol() const { return _abs_tol; }
      int leaf_size() const { return _leaf_size; }
      int max_rank() const { return _max_rank; }
      bool log_ranks() const { return _log_ranks; }
      bool verbose() const { return _verbose; }

      void set_from_command_line(int argc, char* argv[]) {
        option long_options[] = {
          {"h_rel_tol",               required_argument, 0, 1},
          {"h_abs_tol",               required_argument, 0, 2},
          {"h_leaf_size",             required_argument, 0, 3},
          {"h_max_rank",              required_argument, 0, 4},
          {"h_log_ranks",             no_argument, 0, 5},
          {"h_verbose",               no_argument, 0, 'v'},
          {"h_quiet",                 no_argument, 0, 'q'},
          {"help",                      no_argument, 0, 'h'},
          {NULL, 0, NULL, 0}
        };
        int c, option_index = 0;
        opterr = optind = 0;
        while ((c = getopt_long_only(argc, argv, "hvq", long_options,
                                     &option_index)) != -1) {
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
          case 5: { set_log_ranks(true); } break;
          case 'v': set_verbose(true); break;
          case 'q': set_verbose(false); break;
          case 'h': describe_options(); break;
          }
        }
      }
      void describe_options() const {
        std::cout << "# H Options:" << std::endl
                  << "#   --h_rel_tol real_t (default "
                  << rel_tol() << ")" << std::endl
                  << "#   --h_abs_tol real_t (default "
                  << abs_tol() << ")" << std::endl
                  << "#   --h_leaf_size int (default "
                  << leaf_size() << ")" << std::endl
                  << "#   --h_max_rank int (default "
                  << max_rank() << ")" << std::endl
                  << "#   --h_log_ranks (default "
                  << log_ranks() << ")" << std::endl
                  << "#   --h_verbose or -v (default "
                  << verbose() << ")" << std::endl
                  << "#   --h_quiet or -q (default "
                  << !verbose() << ")" << std::endl
                  << "#   --help or -h" << std::endl;
      }
    };

  } // end namespace H
} // end namespace strumpack


#endif // H_OPTIONS_HPP
