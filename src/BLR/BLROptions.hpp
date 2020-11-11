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

#include <string>
#include <cassert>

#include "dense/BLASLAPACKWrapper.hpp"

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

    enum class LowRankAlgorithm { RRQR, ACA, BACA };
    std::string get_name(LowRankAlgorithm a);

    enum class Admissibility { STRONG, WEAK };
    std::string get_name(Admissibility a);

    enum class BLRFactorAlgorithm { RL, LL, COMB, STAR };
    std::string get_name(BLRFactorAlgorithm a);

    enum class CompressionKernel { HALF, FULL };
    std::string get_name(CompressionKernel a);

    template<typename scalar_t> class BLROptions {
      using real_t = typename RealType<scalar_t>::value_type;

    private:
      real_t rel_tol_ = default_BLR_rel_tol<real_t>();
      real_t abs_tol_ = default_BLR_abs_tol<real_t>();
      int leaf_size_ = 128;
      int max_rank_ = 5000;
      bool verbose_ = true;
      LowRankAlgorithm lr_algo_ = LowRankAlgorithm::RRQR;
      int BACA_blocksize_ = 4;
      Admissibility adm_ = Admissibility::STRONG;
      BLRFactorAlgorithm blr_algo_ = BLRFactorAlgorithm::STAR;
      CompressionKernel crn_krnl_ = CompressionKernel::HALF;


    public:
      void set_rel_tol(real_t rel_tol) {
        assert(rel_tol <= real_t(1.) && rel_tol >= real_t(0.));
        rel_tol_ = rel_tol;
      }
      void set_abs_tol(real_t abs_tol) {
        assert(abs_tol >= real_t(0.));
        abs_tol_ = abs_tol;
      }
      void set_leaf_size(int leaf_size) {
        assert(leaf_size_ > 0);
        leaf_size_ = leaf_size;
      }
      void set_max_rank(int max_rank) {
        assert(max_rank > 0);
        max_rank_ = max_rank;
      }
      void set_low_rank_algorithm(LowRankAlgorithm a) {
        lr_algo_ = a;
      }
      void set_admissibility(Admissibility adm) { adm_ = adm; }
      void set_verbose(bool verbose) { verbose_ = verbose; }
      void set_BACA_blocksize(int B) {
        assert(B > 0);
        BACA_blocksize_ = B;
      }
      void set_BLR_factor_algorithm(BLRFactorAlgorithm a) {
        blr_algo_ = a;
      }
      void set_compression_kernel(CompressionKernel a) {
        crn_krnl_ = a;
      }

      real_t rel_tol() const { return rel_tol_; }
      real_t abs_tol() const { return abs_tol_; }
      int leaf_size() const { return leaf_size_; }
      int max_rank() const { return max_rank_; }
      LowRankAlgorithm low_rank_algorithm() const { return lr_algo_; }
      Admissibility admissibility() const { return adm_; }
      bool verbose() const { return verbose_; }
      int BACA_blocksize() const { return BACA_blocksize_; }
      BLRFactorAlgorithm BLR_factor_algorithm() const { return blr_algo_; }
      CompressionKernel compression_kernel() const { return crn_krnl_; }

      void set_from_command_line(int argc, const char* const* cargv);

      void describe_options() const;
    };

  } // end namespace BLR
} // end namespace strumpack


#endif // BLR_OPTIONS_HPP
