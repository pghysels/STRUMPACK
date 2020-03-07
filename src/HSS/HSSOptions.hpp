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
 * \file HSSOptions.hpp
 * \brief Contains the HSSOptions class as well as general routines
 * for HSS options.
 */
#ifndef HSS_OPTIONS_HPP
#define HSS_OPTIONS_HPP

#include "clustering/Clustering.hpp"


namespace strumpack {

  /*! HSS namespace. */
  namespace HSS {

    /**
     * Get the default relative HSS compression tolerance (this is for
     * double precision, might be overloaded depending on floating
     * point precision). This can be changed using the HSSOptions
     * object. Tuning this parameter (in the HSSOptions object) is
     * crucial for performance of the HSS algorithms.
     */
    template<typename real_t> inline real_t default_HSS_rel_tol() {
      return real_t(1e-2);
    }
    /**
     * Get the default absolute HSS compression tolerance (this is for
     * double precision, might be overloaded for single
     * precision). This can be changed using the HSSOptions object.
     */
    template<typename real_t> inline real_t default_HSS_abs_tol() {
      return real_t(1e-8);
    }

    /**
     * Get the default relative HSS compression tolerance for single
     * precision computations. This can be changed using the
     * HSSOptions object. Tuning this parameter (in the
     * HSSOptions<float> object) is crucial for performance of the HSS
     * algorithms.
     */
    template<> inline float default_HSS_rel_tol() {
      return 1e-1;
    }
    /**
     * Get the default absolute HSS compression tolerance for single
     * precision computations. This can be changed using the
     * HSSOptions object.
     */
    template<> inline float default_HSS_abs_tol() {
      return 1e-5;
    }

    /**
     * Enumeration of possible versions of the randomized sampling HSS
     * compression algorithms.
     * \ingroup Enumerations
     */
    enum class CompressionAlgorithm {
      ORIGINAL,    /*!< Start with an initial guess of the rank,
                      double the number of random samples until
                      desired accuracy. */
      STABLE,      /*!< Start with an initial guess of the rank, add a
                     fix amount of random vectors, until desired
                     accuracy is reached. */
      HARD_RESTART /*!< Start with an initial guess of the rank, add a
                     fix amount of random vectors, if not enough,
                     start again from scratch using twice as many
                     random sample vectors. */
    };

    /**
     * Return a string with the name of the compression algorithm.
     * \param a type of the randomized compression algorihtm
     * \return name, string with a short description
     */
    std::string get_name(CompressionAlgorithm a);

    /**
     * \class HSSOptions
     * \brief Class containing several options for the HSS code and
     * data-structures
     *
     * \tparam scalar_t scalar type, can be float, double,
     * std::complex<float> or std::complex<double>. This is used here
     * mainly because tolerances might depend on the precision.
     */
    template<typename scalar_t> class HSSOptions {

    public:
      /**
       * real_t is the real type corresponding to the (possibly
       * complex) scalar_t template parameter
       */
      using real_t = typename RealType<scalar_t>::value_type;

      /**
       * Set the relative tolerance to be used for HSS
       * compression. Tuning this parameter is very important for
       * performance.
       *
       * \param rel_tol relative compression tolerance
       */
      void set_rel_tol(real_t rel_tol) {
        assert(rel_tol <= real_t(1.) && rel_tol >= real_t(0.));
        _rel_tol = rel_tol;
      }

      /**
       * Set the absolute compression tolerance.
       *
       * \param abs_tol absolute compression tolerance
       */
      void set_abs_tol(real_t abs_tol) {
        assert(abs_tol >= real_t(0.));
        _abs_tol = abs_tol;
      }

      /**
       * Set the HSS leaf size. The smallest diagonal blocks in the
       * HSS hierarchy will have size approximately the leaf size
       * (within a factor 2).
       *
       * \param leaf_size
       */
      void set_leaf_size(int leaf_size) {
        assert(_leaf_size > 0);
        _leaf_size = leaf_size;
      }

      /**
       * Set the initial number of random samples to be used in the
       * random sampling HSS construction algorithm. See the manual
       * for more information on the randomized compression algorithm.
       */
      void set_d0(int d0) { assert(d0 > 0); _d0 = d0; }

      /**
       * Set the number of random to be used to increment the random
       * samples vectors in the adaptive randomized HSS compression
       * algorithm. This is only used when compression_algorithm() ==
       * CompressionAlgorithm::STABLE. See the manual for more
       * information on the randomized compression algorithm.
       */
      void set_dd(int dd) { assert(dd > 0); _dd = dd; }

      /**
       * Oversampling parameter. Used in adaptive compression, to
       * check stopping criterion.
       */
      void set_p(int p) { assert(p >= 0); _p = p; }

      /**
       * Set the maximum rank allowed in HSS compression.
       */
      void set_max_rank(int max_rank) {
        assert(max_rank > 0);
        _max_rank = max_rank;
      }

      /**
       * Set the random engine, used in randomized compression.
       * \see RandomEngine, RandomDistribution, set_random_distribution()
       */
      void set_random_engine(random::RandomEngine random_engine) {
        _random_engine = random_engine;
      }

      /**
       * Set the random distribution, used in randomized compression.
       * \see RandomEngine, RandomDistribution, set_random_engine()
       */
      void set_random_distribution
      (random::RandomDistribution random_distribution) {
        _random_distribution = random_distribution;
      }

      /**
       * Specify the variant of the adaptive compression
       * algorithm. See the manual for more information.
       *
       * \param a Type of (adaptive) compression scheme
       */
      void set_compression_algorithm(CompressionAlgorithm a) {
        _compress_algo = a;
      }

      /**
       * Specify the clustering algorithm. This is used when
       * constructing a kernel matrix approximation.
       *
       * \param a Clustering algorithm.
       */
      void set_clustering_algorithm(ClusteringAlgorithm a) {
        _clustering_algo = a;
      }

      /**
       * Set the number of approximate nearest neighbors used in the
       * HSS compression algorithm for kernel matrices.
       *
       * \param neighbors Number of approximate neighbors
       */
      void set_approximate_neighbors(int neighbors) {
        _approximate_neighbors = neighbors;
      }

      /**
       * Set the number of iterations used in the approximate nearest
       * neighbors search algorithm used in the HSS compression
       * algorithm for kernel matrices. The algorithm will build iters
       * number of random trees.
       *
       * \param iters Number of random trees to be build
       */
      void set_ann_iterations(int iters) {
        assert(iters > 0);
        _ann_iterations = iters;
      }

      /**
       * Set this to true if you want to manually fill the random
       * sample vectors with random values.
       */
      void set_user_defined_random(bool user_defined_random) {
        _user_defined_random = user_defined_random;
      }

      /**
       * Set this to true if you require communication in the element
       * extraction routine, since in that case the element extraction
       * is collective, and has to be synchronized.
       */
      void set_synchronized_compression(bool sync) {
        _sync = sync;
      }

      /**
       * Log the HSS ranks to a file. TODO is this currently
       * supported??
       */
      void set_log_ranks(bool log_ranks) { _log_ranks = log_ranks; }

      /**
       * Enable or disable verbose output (only by the root process)
       * to stdout.
       */
      void set_verbose(bool verbose) { _verbose = verbose; }

      /**
       * Get the relative compression tolerance.
       * \return the relative compression tolerance
       * \see set_rel_tol(), set_abs_tol(), get_abs_tol()
       */
      real_t rel_tol() const { return _rel_tol; }

      /**
       * Get the absolute compression tolerance.
       * \return the absolute compression tolerance
       * \see set_abs_tol(), set_rel_tol(), rel_tol()
       */
      real_t abs_tol() const { return _abs_tol; }

      /**
       * Get the HSS leaf size.
       * \return the (approximate) HSS leaf size
       * \see set_leaf_size()
       */
      int leaf_size() const { return _leaf_size; }

      /**
       * Get the initial number of random vector that will be used in
       * adaptive randomized HSS compression. See the manual for more
       * info.
       *
       * \return initial guess for the rank, used in adaptive
       * compression
       * \see set_d0(), set_dd(), set_p()
       */
      int d0() const { return _d0; }

      /**
       * Increment for the number of random vectors during adaptive
       * compression.
       *
       * \return amount with which the number of random vectors will
       * be incremented
       * \see set_d0(), set_dd()
       */
      int dd() const { return _dd; }

      /**
       * Get the current value of the oversampling parameter.
       * \return the oversampling parameter.
       * \see set_p()
       */
      int p() const { return _p; }

      /**
       * Get the maximum allowable rank (note, this is not the actual
       * maximum computed rank).
       * \return maximum allowable rank
       * \see set_max_rank()
       */
      int max_rank() const { return _max_rank; }

      /**
       * Return the type of random engine to use.
       * \return random engine
       * \see set_random_engine
       */
      random::RandomEngine random_engine() const { return _random_engine; }

      /**
       * Return the type of random distribution to use in the random
       * sampling HSS construction.
       * \return random distribution
       * \see set_random_distribution
       */
      random::RandomDistribution random_distribution() const {
        return _random_distribution;
      }

      /**
       * Return which variant of the compression algorithm to use.
       * \return Variant of HSS compression algorithm
       * \see set_compression_algorithm
       */
      CompressionAlgorithm compression_algorithm() const {
        return _compress_algo;
      }

      /**
       * Get the clustering algorithm to be used. This is used when
       * constructing an HSS approximation of a kernel matrix.
       * \return clustering algorithm
       * \see set_clustering_algorithm
       */
      ClusteringAlgorithm clustering_algorithm() const {
        return _clustering_algo;
      }

      /**
       * Get the number of approximate nearest neighbors used in the
       * HSS compression algorithm for kernel matrices.
       *
       * \return Number of approximate neighbors
       */
      int approximate_neighbors() const {
        return _approximate_neighbors;
      }

      /**
       * Get the number of iterations used in the approximate nearest
       * neighbors search algorithm used in the HSS compression
       * algorithm for kernel matrices. The algorithm will build iters
       * number of random trees.
       *
       * \return Number of random trees to be build
       */
      int ann_iterations() const {
        return _ann_iterations;
      }

      /**
       * Will the user define its own random matrices?
       *
       * \return True if the user will fill up the random matrices,
       * used in the random sampling based HSS construcion,
       * him/her-self.
       * \see set_user_defined_random
       */
      bool user_defined_random() const { return _user_defined_random; }

      /**
       * Whether or not the synchronize the element extraction
       * routine.
       * \return True if synchronization is required in the element
       * extraction routine, else False.
       * \see set_synchromized_compression
       */
      bool synchronized_compression() const { return _sync; }

      /**
       * Check if the ranks should be printed to a log file.  __NOT
       * supported currently__
       *
       * \return True is the ranks should be printed to a log file,
       * else False.
       * \see set_log_ranks
       */
      bool log_ranks() const { return _log_ranks; }

      /**
       * Verbose or quiet?
       * \return True if we want output from the HSS algorithms,
       * else False.
       * \see set_verbose
       */
      bool verbose() const { return _verbose; }

      /**
       * Parse the command line options given by argc and argv.  The
       * options will not be modified. Run with --help to see an
       * overview of available options, or call describe_options().
       *
       * \param argc Number of elements in argv
       * \param argv Array with options
       */
      void set_from_command_line(int argc, const char* const* argv);

      /**
       * Print an overview of the available command line options and
       * their current values.
       */
      void describe_options() const;

    private:
      real_t _rel_tol = default_HSS_rel_tol<real_t>();
      real_t _abs_tol = default_HSS_abs_tol<real_t>();
      int _leaf_size = 512;
      int _d0 = 128;
      int _dd = 64;
      int _p = 10;
      int _max_rank = 50000;
      random::RandomEngine _random_engine =
        random::RandomEngine::LINEAR;
      random::RandomDistribution _random_distribution =
        random::RandomDistribution::NORMAL;
      bool _user_defined_random = false;
      bool _log_ranks = false;
      CompressionAlgorithm _compress_algo = CompressionAlgorithm::STABLE;
      bool _sync = false;
      ClusteringAlgorithm _clustering_algo = ClusteringAlgorithm::TWO_MEANS;
      int _approximate_neighbors = 64;
      int _ann_iterations = 5;
      bool _verbose = true;
    };

  } // end namespace HSS
} // end namespace strumpack


#endif // HSS_OPTIONS_HPP
