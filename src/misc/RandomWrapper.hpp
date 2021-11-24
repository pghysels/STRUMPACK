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
 * five (5) year renewals, the U.S. Government igs granted for itself
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
 * \file RandomWrapper
 * \brief Contains convenience wrappers around the C++11 (pseudo)
 * random number generators and random distributions.
 */
#ifndef RANDOM_WRAPPER_HPP
#define RANDOM_WRAPPER_HPP

#include <memory>
#include <random>
#include <iostream>

namespace strumpack {

  /**
   * Namespace containing simple wrappers around the C++11 random
   * number generators and distributions.
   */
  namespace random {

    /**
     * \brief Random number engine.
     * \ingroup Enumerations
     */
    enum class RandomEngine {
      LINEAR,   /*!< The C++11 std::minstd_rand random number generator. */
      MERSENNE  /*!< The C++11 std::mt19937 random number generator.     */
    };

    /**
     * Return a short string (name) for the random engine.
     * \param e random engine
     * \return name of the random engine
     */
    inline std::string get_name(RandomEngine e) {
      switch (e) {
      case RandomEngine::LINEAR: return "minstd_rand";
      case RandomEngine::MERSENNE: return "mt19937";
      }
      return "unknown";
    }

    /**
     * The random number distribution.
     * \ingroup Enumerations
     */
    enum class RandomDistribution {
      NORMAL,   /*!< Normal(0,1) distributed numbers,
                  takes roughly 23 flops per random number.  */
      UNIFORM   /*!< Uniform [0,1] distributed numbers
                  takes about 7 flops per random number.   */
    };

    /**
     * Return a short string (name) for the random number distribution.
     * \param d random distribution
     * \return name of the random distribution
     */
    inline std::string get_name(RandomDistribution d) {
      switch (d) {
      case RandomDistribution::NORMAL: return "normal(0,1)";
      case RandomDistribution::UNIFORM: return "uniform[0,1]";
      }
      return "unknown";
    }

    /**
     * \class RandomGeneratorBase
     * \brief class to wrap the C++11 random number
     * generator/distribution
     *
     * This is a pure virtual base class.
     *
     * \tparam real_t can be float or double
     *
     * \see RandomGenerator
     */
    template<typename real_t> class RandomGeneratorBase {
    public:
      virtual ~RandomGeneratorBase() {}
      virtual void seed(std::size_t s) = 0;
      virtual void seed(std::seed_seq& s) = 0;
      virtual void seed(std::uint32_t i, std::uint32_t j) = 0;
      virtual real_t get() = 0;
      virtual real_t get(std::uint32_t i, std::uint32_t j) = 0;
      virtual int flops_per_prng() = 0;
    };

    /**
     * \class RandomGenerator
     * \brief Class for a random number generator
     *
     * \tparam real_t float or double
     * \tparam E the random number engine: std::mt19937 or
     * std::minstd_rand
     * \tparam D the random number distribution:
     * std::uniform_real_distribution or std::normal_distribution
     *
     * \see RandomGeneratorBase
     */
    template<typename real_t, typename E, typename D>
    class RandomGenerator : public RandomGeneratorBase<real_t> {
    public:
      /**
       * Default constructor, using seed 0.
       */
      RandomGenerator() : e(0) {}

      /**
       * Constructor using seed s.
       */
      RandomGenerator(std::size_t s) : e(s) {}

      /**
       * Seed with value s.
       */
      void seed(std::size_t s) { e.seed(s); d.reset(); }

      /**
       * Seed with a seed sequence.
       */
      void seed(std::seed_seq& s) { e.seed(s); d.reset(); }

      /**
       * Seed with two values (for instance 2 coordinates, point in a
       * matrix). Can be used to get reproducible (pseudo-random)
       * matrix elements.
       */
      void seed(std::uint32_t i, std::uint32_t j) {
        std::seed_seq seq{i,j};
        seed(seq);
      }

      /**
       * get the next random element.
       */
      real_t get() { return d(e); }

      /**
       * Get a (reproducible) element for a specific 2d point.
       */
      real_t get(std::uint32_t i, std::uint32_t j) {
        std::seed_seq seq{i,j};
        seed(seq);
        return get();
      };

      /**
       * Return the (approximate) number of flops required to generate
       * a random number.
       */
      int flops_per_prng() {
        if (std::is_same<D,std::normal_distribution<real_t>>())
          return 23;
        if (std::is_same<D,std::uniform_real_distribution<real_t>>())
          return 7;
        std::cout << "ERROR: random distribution not recognized" << std::endl;
        return 0;
      }

    private:
      E e;
      D d;
    };

    /**
     * Factory method to construct a RandomGeneratorBase with a
     * specified random engine and random distribution, with seed s.
     *
     * \tparam real_t float or double
     *
     * \param seed seed for the gererator
     * \param e random engine
     * \param d random distribution
     */
    template<typename real_t> std::unique_ptr<RandomGeneratorBase<real_t>>
    make_random_generator(std::size_t seed, RandomEngine e,
                          RandomDistribution d) {
      if (e == RandomEngine::LINEAR) {
        if (d == RandomDistribution::NORMAL)
          return std::unique_ptr<RandomGeneratorBase<real_t>>
            (new RandomGenerator<real_t,std::minstd_rand,
             std::normal_distribution<real_t>>(seed));
        else if (d == RandomDistribution::UNIFORM)
          return std::unique_ptr<RandomGeneratorBase<real_t>>
            (new RandomGenerator<real_t,std::minstd_rand,
             std::uniform_real_distribution<real_t>>(seed));
      } else if (e == RandomEngine::MERSENNE) {
        if (d == RandomDistribution::NORMAL)
          return std::unique_ptr<RandomGeneratorBase<real_t>>
            (new RandomGenerator<real_t,std::mt19937,
             std::normal_distribution<real_t>>(seed));
        else if (d == RandomDistribution::UNIFORM)
          return std::unique_ptr<RandomGeneratorBase<real_t>>
            (new RandomGenerator<real_t,std::mt19937,
             std::uniform_real_distribution<real_t>>(seed));
      }
      return NULL;
    }

    /**
     * Factory method to construct a RandomGeneratorBase with a
     * specified random engine and random distribution, using seed 0.
     *
     * \tparam real_t float or double
     *
     * \param seed seed for the gererator
     * \param e random engine
     * \param d random distribution
     */
    template<typename real_t> std::unique_ptr<RandomGeneratorBase<real_t>>
    make_random_generator(RandomEngine e, RandomDistribution d) {
      return make_random_generator<real_t>(0, e, d);
    }

    /**
     * Factory method to construct a RandomGeneratorBase with the
     * default engine and distribution.
     *
     * \tparam real_t float or double
     *
     * \param seed seed for the gererator
     */
    template<typename real_t> std::unique_ptr<RandomGeneratorBase<real_t>>
    make_default_random_generator(std::size_t seed=0) {
      return make_random_generator<real_t>
        (0, RandomEngine::LINEAR, RandomDistribution::NORMAL);
    }

  } // end namespace random
} // end namespace strumpack

#endif // RANDOM_WRAPPER
