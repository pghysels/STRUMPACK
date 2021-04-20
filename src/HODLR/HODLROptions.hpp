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

#include "clustering/Clustering.hpp"
#include "structured/StructuredOptions.hpp"

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
    std::string get_name(CompressionAlgorithm a);

    /**
     * Return a CompressionAlgorithm enum based on the input string.
     *
     * \param c String, possible values are 'natural', '2means',
     * 'kdtree', 'pca' and 'cobble'. This is case sensitive.
     */
    CompressionAlgorithm get_compression_algorithm(const std::string& c);



    /**
     * \class HODLROptions
     * \brief Class containing several options for the HODLR code and
     * data-structures
     *
     * \tparam scalar_t scalar type, can be float, double,
     * std::complex<float> or std::complex<double>. This is used here
     * mainly because tolerances might depend on the precision.
     */
    template<typename scalar_t> class HODLROptions
      : public structured::StructuredOptions<scalar_t> {

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
      HODLROptions() :
        structured::StructuredOptions<scalar_t>(structured::Type::HODLR) {
        set_defaults();
      }

      HODLROptions(const structured::StructuredOptions<scalar_t>& sopts)
        : structured::StructuredOptions<scalar_t>(sopts) {
        this->type_ = structured::Type::HODLR;
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
       * Enable/disable the less_adapt algorithm in randomized
       * construction.
       */
      void set_less_adapt(bool l) { less_adapt_ = l; }

      /**
       * Enable/disable the N^1.5 entry evaluation-based construction.
       */
      void set_BF_entry_n15(bool l) { BF_entry_n15_ = l; }

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
       * Returns whether or not to use less_adapt in the randomized
       * construction algorithm.
       */
      bool less_adapt() const { return less_adapt_; }

      /**
       * Returns whether or not to use N^1.5 entry-evaluation-based algorithm.
       */
      bool BF_entry_n15() const { return BF_entry_n15_; }

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
      int rank_guess_ = 128;
      double rank_rate_ = 2.;
      ClusteringAlgorithm clustering_algo_ = ClusteringAlgorithm::COBBLE;
      int butterfly_levels_ = 0;
      CompressionAlgorithm compression_algo_ = CompressionAlgorithm::ELEMENT_EXTRACTION;
      int BACA_block_size_ = 16;
      double BF_sampling_parameter_ = 1.2;
      int geo_ = 2;
      int lr_leaf_ = 5;
      int knn_hodlrbf_ = 64;
      int knn_lrbf_ = 128;
      bool less_adapt_ = true;
      bool BF_entry_n15_ = false;

      void set_defaults() {
        this->type_ = structured::Type::HODLR;
        this->rel_tol_ = default_HODLR_rel_tol<real_t>();
        this->abs_tol_ = default_HODLR_abs_tol<real_t>();
        this->leaf_size_ = 256;
        this->max_rank_ = 50000;
      }

    };

  } // end namespace HODLR
} // end namespace strumpack


#endif // HODLR_OPTIONS_HPP
