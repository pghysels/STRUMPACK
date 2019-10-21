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
/**
 * \file HODLROptions.hpp
 * \brief Contains the class holding HODLR matrix options.
 */
#ifndef HODLR_OPTIONS_HPP
#define HODLR_OPTIONS_HPP

#include <cstring>
#include <getopt.h>
#include "clustering/Clustering.hpp"

namespace strumpack {

  namespace HODLR {

    /**
     * Get the default relative HODLR compression tolerance (this is
     * for double precision, might be overloaded depending on floating
     * point precision). This can be changed using the HODLROptions
     * object. Tuning this parameter (in the HODLROptions object) is
     * crucial for performance of the HODLR algorithms.
     */
    template<typename real_t> inline real_t default_HODLR_rel_tol() {
      return real_t(1e-4);
    }
    /**
     * Get the default absolute HODLR compression tolerance (this is
     * for double precision, might be overloaded for single
     * precision). This can be changed using the HODLROptions object.
     */
    template<typename real_t> inline real_t default_HODLR_abs_tol() {
      return real_t(1e-10);
    }

    /**
     * Get the default relative HODLR compression tolerance for single
     * precision computations. This can be changed using the
     * HODLROptions object. Tuning this parameter (in the
     * HODLROptions<float> object) is crucial for performance of the
     * HODLR algorithms.
     */
    template<> inline float default_HODLR_rel_tol() {
      return 1e-2;
    }
    /**
     * Get the default absolute HODLR compression tolerance for single
     * precision computations. This can be changed using the
     * HODLROptions object.
     */
    template<> inline float default_HODLR_abs_tol() {
      return 1e-5;
    }

    /**
     * Enumeration of possible compressions, through randomized
     * sampling or via element extraction.
     * \ingroup Enumerations
     */
    enum class CompressionAlgorithm {
      RANDOM_SAMPLING,     /*!< Random sampling. */
      ELEMENT_EXTRACTION   /*!< Element extraction. */
    };

    /**
     * Return a string with the name of the compression algorithm.
     * \param a type of the compression algorihtm
     * \return name, string with a short description
     */
    inline std::string get_name(CompressionAlgorithm a) {
      switch (a) {
      case CompressionAlgorithm::RANDOM_SAMPLING: return "sampling"; break;
      case CompressionAlgorithm::ELEMENT_EXTRACTION: return "extraction"; break;
      default: return "unknown";
      }
    }

    /**
     * Return a CompressionAlgorithm enum based on the input string.
     *
     * \param c String, possible values are 'natural', '2means',
     * 'kdtree', 'pca' and 'cobble'. This is case sensitive.
     */
    inline CompressionAlgorithm
    get_compression_algorithm(const std::string& c) {
      if (c == "sampling") return CompressionAlgorithm::RANDOM_SAMPLING;
      else if (c == "extraction") return CompressionAlgorithm::ELEMENT_EXTRACTION;
      else {
        std::cerr << "WARNING: Compression algorithm not recognized,"
                  << " setting to 'sampling'."
                  << std::endl;
        return CompressionAlgorithm::RANDOM_SAMPLING;
      }
    }

    /**
     * \class HODLROptions
     * \brief Class containing several options for the HODLR code and
     * data-structures
     *
     * \tparam scalar_t scalar type, can be float, double,
     * std::complex<float> or std::complex<double>. This is used here
     * mainly because tolerances might depend on the precision.
     */
    template<typename scalar_t> class HODLROptions {

    public:
      /**
       * real_t is the real type corresponding to the (possibly
       * complex) scalar_t template parameter
       */
      using real_t = typename RealType<scalar_t>::value_type;

      /**
       * Default constructor, sets all options to their default
       * values.
       */
      HODLROptions() {}

      /**
       * Set the relative tolerance to be used for HODLR
       * compression. Tuning this parameter is very important for
       * performance.
       *
       * \param rel_tol relative compression tolerance
       */
      void set_rel_tol(real_t rel_tol) {
        assert(rel_tol <= real_t(1.) && rel_tol >= real_t(0.));
        rel_tol_ = rel_tol;
      }

      /**
       * Set the absolute compression tolerance.
       *
       * \param abs_tol absolute compression tolerance
       */
      void set_abs_tol(real_t abs_tol) {
        assert(abs_tol >= real_t(0.));
        abs_tol_ = abs_tol;
      }

      /**
       * Set the HODLR leaf size. The smallest diagonal blocks in the
       * HODLR hierarchy will have size approximately the leaf size
       * (within a factor 2).
       *
       * \param leaf_size
       */
      void set_leaf_size(int leaf_size) {
        assert(leaf_size_ > 0);
        leaf_size_ = leaf_size;
      }

      /**
       * Set the maximum rank allowed in HODLR compression.
       */
      void set_max_rank(int max_rank) {
        assert(max_rank > 0);
        max_rank_ = max_rank;
      }

      /**
       * Set the initial guess for the rank.
       */
      void set_rank_guess(int rank_guess) {
        assert(rank_guess > 0);
        rank_guess_ = rank_guess;
      }

      /**
       * Set the rate of increment for adaptively determining the
       * rank.
       */
      void set_rank_rate(double rank_rate) {
        assert(rank_rate > 0);
        rank_rate_ = rank_rate;
      }

      /**
       * Specify the clustering algorithm. This is used when
       * constructing a kernel matrix approximation.
       */
      void set_clustering_algorithm(ClusteringAlgorithm a) {
        clustering_algo_ = a;
      }

      /**
       * Specify the compression algorithm to be used.
       */
      void set_compression_algorithm(CompressionAlgorithm a) {
        compression_algo_ = a;
      }

      /**
       * Set the number of butterfly levels to use for each HODLR
       * matrix.
       */
      void set_butterfly_levels(int bfl) {
        assert(bfl >= 0);
        butterfly_levels_ = bfl;
      }

      /**
       * Set the BACA block size.
       */
      void set_BACA_block_size(int BACA) {
        assert(BACA > 0);
        BACA_block_size_ = BACA;
      }

      /**
       * Set sampling parameter for use in linear complexity butterfly
       * compression, higher for more robust sampling.
       */
      void set_BF_sampling_parameter(double param) {
        assert(param > 0);
        BF_sampling_parameter_ = param;
      }

      /**
       * geo should be 0, 1, 2 or 3. 0 means use point geometry, 1
       * means do not use any geometry. 2 means use the graph
       * connectivity for the distance and admissibility info, 3 means
       * use the graph to directly find closest neighbors.
       */
      void set_geo(int geo) {
        assert(geo == 0 || geo == 1 || geo == 2 || geo == 3);
        geo_ = geo;
      }

      /**
       * lr_leaf should be 1, 2, 3, 4, 5. 1 means svd, 2 means rrqr, 3
       * means baseline aca, 4 means baca original version, 5 means
       * baca improved version.
       */
      void set_lr_leaf(int lr_leaf) {
        assert(lr_leaf == 1 || lr_leaf == 2 || lr_leaf == 3 ||
               lr_leaf == 4 || lr_leaf == 5);
        lr_leaf_ = lr_leaf;
      }

      /**
       * Set the number of neighbors to use in construction of the
       * HODLR or HODBF matrices.
       */
      void set_knn_hodlrbf(int k) {
        assert(k >= 0);
        knn_hodlrbf_ = k;
      }

      /**
       * Set the number of neighbors to use in construction of the
       * LR or Butterfly blocks.
       */
      void set_knn_lrbf(int k) {
        assert(k >= 0);
        knn_lrbf_ = k;
      }

      /**
       * Enable or disable verbose output (only by the root process)
       * to stdout.
       */
      void set_verbose(bool verbose) { verbose_ = verbose; }

      /**
       * Get the relative compression tolerance.
       * \return the relative compression tolerance
       * \see set_rel_tol(), set_abs_tol(), get_abs_tol()
       */
      real_t rel_tol() const { return rel_tol_; }

      /**
       * Get the absolute compression tolerance.
       * \return the absolute compression tolerance
       * \see set_abs_tol(), set_rel_tol(), get_rel_tol()
       */
      real_t abs_tol() const { return abs_tol_; }

      /**
       * Get the HODLR leaf size.
       * \return the (approximate) HODLR leaf size
       * \see set_leaf_size()
       */
      int leaf_size() const { return leaf_size_; }

      /**
       * Get the maximum allowable rank (note, this is not the actual
       * maximum computed rank).
       * \return maximum allowable rank
       * \see set_max_rank()
       */
      int max_rank() const { return max_rank_; }

      /**
       * Get the initial guess for the rank.
       */
      int rank_guess() const { return rank_guess_; }

      /**
       * Get the rate of increment for adaptively determining the
       * rank.
       */
      double rank_rate() const { return rank_rate_; }

      /**
       * Get the clustering algorithm to be used. This is used when
       * constructing an HODLR approximation of a kernel matrix.
       * \return clustering algorithm
       * \see set_clustering_algorithm
       */
      ClusteringAlgorithm clustering_algorithm() const {
        return clustering_algo_;
      }

      /**
       * Get the compression algorithm to be used.
       * \return compression algorithm
       * \see set_compression_algorithm
       */
      CompressionAlgorithm compression_algorithm() const {
        return compression_algo_;
      }

      /**
       * get the number of butterfly levels to use for each HODLR
       * matrix.
       */
      int butterfly_levels() const { return butterfly_levels_; }

      /**
       * Get the BACA block size.
       */
      int BACA_block_size() const { return BACA_block_size_; }

      /**
       * Get butterfly sampling parameter.
       */
      double BF_sampling_parameter() const { return BF_sampling_parameter_; }

      /**
       * Use geometry information? 0, 1 or 2
       */
      int geo() const { return geo_; }

      /**
       * Bottom level compression algorithms in H-BACA 1, 2, 3, 4, or 5
       */
      int lr_leaf() const { return lr_leaf_; }

      /**
       * The number of neighbors to use in the HODLR or HODBF
       * construction.
       */
      int knn_hodlrbf() const { return knn_hodlrbf_; }

      /**
       * The number of neighbors to use in the LR or Butterfly block
       * construction.
       */
      int knn_lrbf() const { return knn_lrbf_; }

      /**
       * Verbose or quiet?
       * \return True if we want output from the HODLR algorithms,
       * else False.
       * \see set_verbose
       */
      bool verbose() const { return verbose_; }

      /**
       * Parse the command line options given by argc and argv.  The
       * options will not be modified. Run with --help to see an
       * overview of available options, or call describe_options().
       *
       * \param argc Number of elements in argv
       * \param argv Array with options
       */
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
      }

      /**
       * Print an overview of the available command line options and
       * their current values.
       */
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
      }

    private:
      real_t rel_tol_ = default_HODLR_rel_tol<real_t>();
      real_t abs_tol_ = default_HODLR_abs_tol<real_t>();
      int leaf_size_ = 128;
      int rank_guess_ = 128;
      double rank_rate_ = 2.;
      int max_rank_ = 5000;
      ClusteringAlgorithm clustering_algo_ = ClusteringAlgorithm::COBBLE;
      int butterfly_levels_ = 0;
      CompressionAlgorithm compression_algo_ = CompressionAlgorithm::ELEMENT_EXTRACTION;
      int BACA_block_size_ = 16;
      double BF_sampling_parameter_ = 2.0;
      int geo_ = 2;
      int lr_leaf_ = 5;
      int knn_hodlrbf_ = 64;
      int knn_lrbf_ = 256;
      bool verbose_ = true;
    };

  } // end namespace HODLR
} // end namespace strumpack


#endif // HODLR_OPTIONS_HPP
