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
/*! \file HODLROptions.hpp
 * \brief For Pieter to complete
 */
#ifndef HODLR_OPTIONS_HPP
#define HODLR_OPTIONS_HPP

#include <cstring>
#include <getopt.h>
#include "clustering/Clustering.hpp"

namespace strumpack {

  /*! HODLR namespace. */
  namespace HODLR {

    template<typename real_t> inline real_t default_HODLR_rel_tol() {
      return real_t(1e-4);
    }
    template<typename real_t> inline real_t default_HODLR_abs_tol() {
      return real_t(1e-10);
    }
    template<> inline float default_HODLR_rel_tol() {
      return 1e-2;
    }
    template<> inline float default_HODLR_abs_tol() {
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

    enum class Admissibility { STRONG, WEAK };
    inline std::string get_name(Admissibility a) {
      switch (a) {
      case Admissibility::STRONG: return "strong"; break;
      case Admissibility::WEAK: return "weak"; break;
      default: return "unknown";
      }
    }

    template<typename scalar_t> class HODLROptions {
      using real_t = typename RealType<scalar_t>::value_type;

    public:
      /*! \brief For Pieter to complete
       * \param rel_tol
       */
      void set_rel_tol(real_t rel_tol) {
        assert(rel_tol <= real_t(1.) && rel_tol >= real_t(0.));
        rel_tol_ = rel_tol;
      }
      void set_abs_tol(real_t abs_tol) {
        assert(abs_tol >= real_t(0.));
        abs_tol_ = abs_tol;
      }
      /*! \brief For Pieter to complete
       * \param leaf_size
       */
      void set_leaf_size(int leaf_size) {
        assert(leaf_size_ > 0);
        leaf_size_ = leaf_size;
      }
      void set_max_rank(int max_rank) {
        assert(max_rank > 0);
        max_rank_ = max_rank;
      }
      /**
       * Specify the clustering algorithm. This is used when
       * constructing a kernel matrix approximation.
       */
      void set_clustering_algorithm(ClusteringAlgorithm a) {
        _clustering_algo = a;
      }
      void set_verbose(bool verbose) { verbose_ = verbose; }

      real_t rel_tol() const { return rel_tol_; }
      real_t abs_tol() const { return abs_tol_; }
      int leaf_size() const { return leaf_size_; }
      int max_rank() const { return max_rank_; }
      ClusteringAlgorithm clustering_algorithm() const {
        return _clustering_algo;
      }
      bool verbose() const { return verbose_; }

      void set_from_command_line(int argc, const char* const* argv) {
        std::vector<char*> argv_local(argc);
        for (int i=0; i<argc; i++) {
          argv_local[i] = new char[strlen(argv[i])+1];
          strcpy(argv_local[i], argv[i]);
        }
        option long_options[] = {
          {"hodlr_rel_tol",               required_argument, 0, 1},
          {"hodlr_abs_tol",               required_argument, 0, 2},
          {"hodlr_leaf_size",             required_argument, 0, 3},
          {"hodlr_max_rank",              required_argument, 0, 4},
          {"hodlr_clustering_algorithm",  required_argument, 0, 5},
          {"hodlr_verbose",               no_argument, 0, 'v'},
          {"hodlr_quiet",                 no_argument, 0, 'q'},
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
            std::string s; iss >> s;
            set_clustering_algorithm(get_clustering_algorithm(s));
          } break;
          case 'v': set_verbose(true); break;
          case 'q': set_verbose(false); break;
          case 'h': describe_options(); break;
          }
        }
        for (auto s : argv_local) delete[] s;
      }

      void describe_options() const {
        std::cout << "# HODLR Options:" << std::endl
                  << "#   --hodlr_rel_tol real_t (default "
                  << rel_tol() << ")" << std::endl
                  << "#   --hodlr_abs_tol real_t (default "
                  << abs_tol() << ")" << std::endl
                  << "#   --hodlr_leaf_size int (default "
                  << leaf_size() << ")" << std::endl
                  << "#   --hodlr_max_rank int (default "
                  << max_rank() << ")" << std::endl
                  << "#   --hodlr_clustering_algorithm natural|2means|kdtree|pca|cobble (default "
                  << get_name(clustering_algorithm())<< ")" << std::endl
                  << "#   --hodlr_verbose or -v (default "
                  << verbose() << ")" << std::endl
                  << "#   --hodlr_quiet or -q (default "
                  << !verbose() << ")" << std::endl
                  << "#   --help or -h" << std::endl << std::endl;
      }

    private:
      real_t rel_tol_ = default_HODLR_rel_tol<real_t>();
      real_t abs_tol_ = default_HODLR_abs_tol<real_t>();
      int leaf_size_ = 128;
      int max_rank_ = 5000;
      ClusteringAlgorithm _clustering_algo = ClusteringAlgorithm::TWO_MEANS;
      bool verbose_ = true;
    };

  } // end namespace HODLR
} // end namespace strumpack


#endif // HODLR_OPTIONS_HPP
