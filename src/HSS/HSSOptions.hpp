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
#include "structured/StructuredOptions.hpp"

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
     * Enumeration of possible versions of the randomized sampling HSS
     * sketching matrices.
     * \ingroup Enumerations
     */
    enum class CompressionSketch {
      GAUSSIAN,  /*!< Sketch using iid gaussian entries mean 0
                   variance 1. */
      SJLT       /*!< Sketch using the sparse Johnson-Lindenstrauss
                   transform, with nnz entries per row randomly selected each
                   with a value of 1/sqrt(nnz) with pr = 0.5 or -1/sqrt(nnz)
                   with pr = 0.5. */
    };

    /**
     * Return a string with the name of the compression algorithm.
     * \param a type of the randomized compression algorihtm
     * \return name, string with a short description
     */
    std::string get_name(CompressionSketch a);

    enum class SJLTAlgo {
      CHUNK,    /*!< puts nnz nonzeros in each row by spliting the row into
                  col/nnz chunks and putting one nonzero in each */
      PERM      /*!< puts nnz nonzeros on each row by permuting all
                  col indices and taking the first nnz*/
    };

    /**
     * Return a string with the name of the sjlt algorithm.
     * \param a type of SJLT sampling algorithm
     * \return name, string with a short description
     */
    std::string get_name(SJLTAlgo a);

    /**
     * \class HSSOptions
     * \brief Class containing several options for the HSS code and
     * data-structures
     *
     * \tparam scalar_t scalar type, can be float, double,
     * std::complex<float> or std::complex<double>. This is used here
     * mainly because tolerances might depend on the precision.
     */
    template<typename scalar_t> class HSSOptions
      : public structured::StructuredOptions<scalar_t> {

    public:

      /**
       * real_t is the real type corresponding to the (possibly
       * complex) scalar_t template parameter
       */
      using real_t = typename RealType<scalar_t>::value_type;

      HSSOptions() :
        structured::StructuredOptions<scalar_t>(structured::Type::HSS) {
        set_defaults();
      }

      HSSOptions(const structured::StructuredOptions<scalar_t>& sopts)
        : structured::StructuredOptions<scalar_t>(sopts) {
        this->type_ = structured::Type::HSS;
      }

      /**
       * Set the initial number of random samples to be used in the
       * random sampling HSS construction algorithm. See the manual
       * for more information on the randomized compression algorithm.
       */
      void set_d0(int d0) { assert(d0 > 0); d0_ = d0; }

      /**
       * Set the number of random to be used to increment the random
       * samples vectors in the adaptive randomized HSS compression
       * algorithm. This is only used when compression_algorithm() ==
       * CompressionAlgorithm::STABLE. See the manual for more
       * information on the randomized compression algorithm.
       */
      void set_dd(int dd) { assert(dd > 0); dd_ = dd; }

      /**
       * Oversampling parameter. Used in adaptive compression, to
       * check stopping criterion.
       */
      void set_p(int p) { assert(p >= 0); p_ = p; }

      /**
       * Set the initial number of nonzeros per row to be used in SJLT
       * sketch.
       */
      void set_nnz0(int nnz0) { assert(nnz0 > 0); nnz0_ = nnz0; }

      /**
       * Set the number of nonzeros per row added each time dd columns are
       * added to the SJLT sketch.
       */
      void set_nnz(int nnz) { assert(nnz > 0); nnz_ = nnz; }

      /**
       * Set the random engine, used in randomized compression.
       * \see RandomEngine, RandomDistribution, set_random_distribution()
       */
      void set_random_engine(random::RandomEngine random_engine) {
        random_engine_ = random_engine;
      }

      /**
       * Set the random distribution, used in randomized compression.
       * \see RandomEngine, RandomDistribution, set_random_engine()
       */
      void set_random_distribution
      (random::RandomDistribution random_distribution) {
        random_distribution_ = random_distribution;
      }

      /**
       * Specify the variant of the adaptive compression
       * algorithm. See the manual for more information.
       *
       * \param a Type of (adaptive) compression scheme
       */
      void set_compression_algorithm(CompressionAlgorithm a) {
        compress_algo_ = a;
      }

      /**
       * Specify the variant of the sketching matrix.
       *
       * \param a Type of sketching matrix
       */
      void set_compression_sketch(CompressionSketch a) {
        compress_sketch_ = a;
      }

      /**
       * Specify the variant of the sketching matrix.
       *
       * \param a Type of sketching matrix
       */
      void set_SJLT_algo(SJLTAlgo a) {
        sjlt_algo_ = a;
      }


      /**
       * Specify the clustering algorithm. This is used when
       * constructing a kernel matrix approximation.
       *
       * \param a Clustering algorithm.
       */
      void set_clustering_algorithm(ClusteringAlgorithm a) {
        clustering_algo_ = a;
      }

      /**
       * Set the number of approximate nearest neighbors used in the
       * HSS compression algorithm for kernel matrices.
       *
       * \param neighbors Number of approximate neighbors
       */
      void set_approximate_neighbors(int neighbors) {
        approximate_neighbors_ = neighbors;
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
        ann_iterations_ = iters;
      }

      /**
       * Set this to true if you want to manually fill the random
       * sample vectors with random values.
       */
      void set_user_defined_random(bool user_defined_random) {
        user_defined_random_ = user_defined_random;
      }

      /**
       * Set this to true if you require communication in the element
       * extraction routine, since in that case the element extraction
       * is collective, and has to be synchronized.
       */
      void set_synchronized_compression(bool sync) {
        sync_ = sync;
      }

      /**
       * Log the HSS ranks to a file. TODO is this currently
       * supported??
       */
      void set_log_ranks(bool log_ranks) { log_ranks_ = log_ranks; }


      /**
       * Get the initial number of random vector that will be used in
       * adaptive randomized HSS compression. See the manual for more
       * info.
       *
       * \return initial guess for the rank, used in adaptive
       * compression
       * \see set_d0(), set_dd(), set_p()
       */
      int d0() const { return d0_; }

      /**
       * Get the initial number of nonzero entries per row in the SJLT sketch.
       *
       * \return initial number of nonzeros per row, used in adaptive
       * compression with SJLT
       * \see set_nnz0(), set_nnz()
       */
      int nnz0() const { return nnz0_; }

      /**
       * Get the increment number of nonzero entries per row in the SJLT sketch
       * when incrementing by size dd
       *
       * \return additional number of nonzeros per row, added in adaptive
       * compression step with SJLT
       * \see set_nnz0(), set_nnz()
       */
       int nnz() const { return nnz_; }

      /**
       * Increment for the number of random vectors during adaptive
       * compression.
       *
       * \return amount with which the number of random vectors will
       * be incremented
       * \see set_d0(), set_dd()
       */
      int dd() const { return dd_; }

      /**
       * Get the current value of the oversampling parameter.
       * \return the oversampling parameter.
       * \see set_p()
       */
      int p() const { return p_; }

      /**
       * Return the type of random engine to use.
       * \return random engine
       * \see set_random_engine
       */
      random::RandomEngine random_engine() const { return random_engine_; }

      /**
       * Return the type of random distribution to use in the random
       * sampling HSS construction.
       * \return random distribution
       * \see set_random_distribution
       */
      random::RandomDistribution random_distribution() const {
        return random_distribution_;
      }

      /**
       * Return which variant of the compression algorithm to use.
       * \return Variant of HSS compression algorithm
       * \see set_compression_algorithm
       */
      CompressionAlgorithm compression_algorithm() const {
        return compress_algo_;
      }

      CompressionSketch compression_sketch() const {
          return compress_sketch_;
      }

      SJLTAlgo SJLT_algo() const{
          return sjlt_algo_;
      }
      /**
       * Get the clustering algorithm to be used. This is used when
       * constructing an HSS approximation of a kernel matrix.
       * \return clustering algorithm
       * \see set_clustering_algorithm
       */
      ClusteringAlgorithm clustering_algorithm() const {
        return clustering_algo_;
      }

      /**
       * Get the number of approximate nearest neighbors used in the
       * HSS compression algorithm for kernel matrices.
       *
       * \return Number of approximate neighbors
       */
      int approximate_neighbors() const {
        return approximate_neighbors_;
      }

      /**
       * Get the number of iterations used in the approximate nearest
       * neighbors search algorithm used in the HSS compression
       * algorithm for kernel matrices. The algorithm will build iters
       * number of random trees.
       *
       * \return Number of random trees to be build
       */
      int ann_iterations() const { return ann_iterations_; }

      /**
       * Will the user define its own random matrices?
       *
       * \return True if the user will fill up the random matrices,
       * used in the random sampling based HSS construcion,
       * him/her-self.
       * \see set_user_defined_random
       */
      bool user_defined_random() const { return user_defined_random_; }

      /**
       * Whether or not the synchronize the element extraction
       * routine.
       * \return True if synchronization is required in the element
       * extraction routine, else False.
       * \see set_synchromized_compression
       */
      bool synchronized_compression() const { return sync_; }

      /**
       * Check if the ranks should be printed to a log file.  __NOT
       * supported currently__
       *
       * \return True is the ranks should be printed to a log file,
       * else False.
       * \see set_log_ranks
       */
      bool log_ranks() const { return log_ranks_; }

      /**
       * Parse the command line options given by argc and argv.  The
       * options will not be modified. Run with --help to see an
       * overview of available options, or call describe_options().
       *
       * \param argc Number of elements in argv
       * \param argv Array with options
       */
      void set_from_command_line(int argc, const char* const* cargv) override;

      /**
       * Print an overview of the available command line options and
       * their current values.
       */
      void describe_options() const override;

    private:
      int d0_ = 128;
      int dd_ = 64;
      int nnz0_ = 4;
      int nnz_ = 4;
      int p_ = 10;
      random::RandomEngine random_engine_ =
        random::RandomEngine::LINEAR;
      random::RandomDistribution random_distribution_ =
        random::RandomDistribution::NORMAL;
      bool user_defined_random_ = false;
      bool log_ranks_ = false;
      CompressionAlgorithm compress_algo_ = CompressionAlgorithm::STABLE;
      CompressionSketch compress_sketch_ = CompressionSketch::GAUSSIAN;
      SJLTAlgo sjlt_algo_ = SJLTAlgo::CHUNK;
      bool sync_ = false;
      ClusteringAlgorithm clustering_algo_ = ClusteringAlgorithm::TWO_MEANS;
      int approximate_neighbors_ = 64;
      int ann_iterations_ = 5;

      void set_defaults() {
        this->type_ = structured::Type::HSS;
        this->rel_tol_ = default_HSS_rel_tol<real_t>();
        this->abs_tol_ = default_HSS_abs_tol<real_t>();
        this->leaf_size_ = 512;
        this->max_rank_ = 50000;
      }
    };

  } // end namespace HSS
} // end namespace strumpack


#endif // HSS_OPTIONS_HPP
