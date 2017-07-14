/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The Regents of
 * the University of California, through Lawrence Berkeley National Laboratory
 * (subject to receipt of any required approvals from the U.S. Dept. of Energy).
 * All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Technology Transfer Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, prepare derivative works, and perform publicly and display publicly.
 * Beginning five (5) years after the date permission to assert copyright is
 * obtained from the U.S. Department of Energy, and subject to any subsequent five
 * (5) year renewals, the U.S. Government igs granted for itself and others acting
 * on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, prepare derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 *
 */
#ifndef RANDOM_WRAPPER_HPP
#define RANDOM_WRAPPER_HPP

#include <memory>
#include <random>

namespace strumpack {
  namespace random {

    /*! \brief Random number engine.
     * \ingroup Enumerations */
    enum class RandomEngine {
      LINEAR,   /*!< The C++11 std::minstd_rand random number generator. */
      MERSENNE  /*!< The C++11 std::mt19937 random number generator.     */
    };

    inline std::string get_name(RandomEngine e) {
      switch (e) {
      case RandomEngine::LINEAR: return "minstd_rand"; break;
      case RandomEngine::MERSENNE: return "mt19937"; break;
      }
      return "unknown";
    }

    /*! \brief The random number distribution.
     * \ingroup Enumerations */
    enum class RandomDistribution {
      NORMAL,   /*!< Normal(0,1) distributed numbers (takes roughly 23 flops per random number).  */
      UNIFORM   /*!< Uniform [0,1] distributed numbers (takes about 7 flops per random number).   */
    };

    inline std::string get_name(RandomDistribution d) {
      switch (d) {
      case RandomDistribution::NORMAL: return "normal(0,1)"; break;
      case RandomDistribution::UNIFORM: return "uniform[0,1]"; break;
      }
      return "unknown";
    }

    template<typename real_t> class RandomGeneratorBase {
    public:
      virtual void seed(std::size_t s) = 0;
      virtual void seed(std::seed_seq& s) = 0;
      virtual void seed(std::uint32_t i, std::uint32_t j) = 0;
      virtual real_t get() = 0;
      virtual real_t get(std::uint32_t i, std::uint32_t j) = 0;
      virtual int flops_per_prng() = 0;
    };

    template<typename real_t, typename E, typename D>
    class RandomGenerator : public RandomGeneratorBase<real_t> {
    public:
      RandomGenerator() : e(0) {}
      RandomGenerator(std::size_t s) : e(s) {}
      void seed(std::size_t s) { e.seed(s); d.reset(); }
      void seed(std::seed_seq& s) { e.seed(s); d.reset(); }
      void seed(std::uint32_t i, std::uint32_t j) {
	std::seed_seq seq{i,j};
	seed(seq);
      }
      real_t get() { return d(e); }
      real_t get(std::uint32_t i, std::uint32_t j) {
	std::seed_seq seq{i,j};
	seed(seq);
	return get();
      };
      int flops_per_prng() {
	if (std::is_same<D,std::normal_distribution<real_t>>()) return 23;
	if (std::is_same<D,std::uniform_real_distribution<real_t>>()) return 7;
	std::cout << "ERROR: random distribution not recognized" << std::endl;
	return 0;
      }
    private:
      E e;
      D d;
    };

    template<typename real_t> std::unique_ptr<RandomGeneratorBase<real_t>>
    make_random_generator(std::size_t seed, RandomEngine e, RandomDistribution d) {
      if (e == RandomEngine::LINEAR) {
	if (d == RandomDistribution::NORMAL)
	  return std::unique_ptr<RandomGeneratorBase<real_t>>
	    (new RandomGenerator<real_t,std::minstd_rand,std::normal_distribution<real_t>>(seed));
	else if (d == RandomDistribution::UNIFORM)
	  return std::unique_ptr<RandomGeneratorBase<real_t>>
	    (new RandomGenerator<real_t,std::minstd_rand,std::uniform_real_distribution<real_t>>(seed));
      } else if (e == RandomEngine::MERSENNE) {
	if (d == RandomDistribution::NORMAL)
	  return std::unique_ptr<RandomGeneratorBase<real_t>>
	    (new RandomGenerator<real_t,std::mt19937,std::normal_distribution<real_t>>(seed));
	else if (d == RandomDistribution::UNIFORM)
	  return std::unique_ptr<RandomGeneratorBase<real_t>>
	    (new RandomGenerator<real_t,std::mt19937,std::uniform_real_distribution<real_t>>(seed));
      }
      return NULL;
    }

    template<typename real_t> std::unique_ptr<RandomGeneratorBase<real_t>>
    make_random_generator(RandomEngine e, RandomDistribution d) {
      return make_random_generator<real_t>(0, e, d);
    }

    template<typename real_t> std::unique_ptr<RandomGeneratorBase<real_t>>
    make_default_random_generator(std::size_t seed=0) {
      return make_random_generator<real_t>(0, RandomEngine::LINEAR, RandomDistribution::NORMAL);
    }

  } // end namespace random
} // end namespace strumpack

#endif // RANDOM_WRAPPER
