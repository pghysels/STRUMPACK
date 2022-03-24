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
 * \brief Contains class holding BLROptions
 */
#ifndef BLR_OPTIONS_HPP
#define BLR_OPTIONS_HPP

#include <string>
#include <cassert>

#include "dense/BLASLAPACKWrapper.hpp"
#include "structured/StructuredOptions.hpp"

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

    enum class BLRFactorAlgorithm { COLWISE, RL, LL, COMB, STAR };
    std::string get_name(BLRFactorAlgorithm a);

    enum class CompressionKernel { HALF, FULL };
    std::string get_name(CompressionKernel a);


    /**
     * \class BLROptions
     * \brief Class containing several options for the BLR code and
     * data-structures
     *
     * \tparam scalar_t scalar type, can be float, double,
     * std::complex<float> or std::complex<double>. This is used here
     * mainly because tolerances might depend on the precision.
     */
    template<typename scalar_t> class BLROptions
      : public structured::StructuredOptions<scalar_t> {

    public:

      /**
       * real_t is the real type corresponding to the (possibly
       * complex) scalar_t template parameter
       */
      using real_t = typename RealType<scalar_t>::value_type;

      BLROptions() :
        structured::StructuredOptions<scalar_t>(structured::Type::BLR) {
        set_defaults();
      }

      BLROptions(const structured::StructuredOptions<scalar_t>& sopts)
        : structured::StructuredOptions<scalar_t>(sopts) {
        this->type_ = structured::Type::BLR;
      }

      void set_low_rank_algorithm(LowRankAlgorithm a) {
        lr_algo_ = a;
      }
      void set_admissibility(Admissibility adm) { adm_ = adm; }
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

      LowRankAlgorithm low_rank_algorithm() const { return lr_algo_; }
      Admissibility admissibility() const { return adm_; }
      int BACA_blocksize() const { return BACA_blocksize_; }
      BLRFactorAlgorithm BLR_factor_algorithm() const { return blr_algo_; }
      CompressionKernel compression_kernel() const { return crn_krnl_; }

      void set_from_command_line(int argc, const char* const* cargv) override;

      void describe_options() const override;

    private:
      bool verbose_ = true;
      LowRankAlgorithm lr_algo_ = LowRankAlgorithm::RRQR;
      int BACA_blocksize_ = 4;
      Admissibility adm_ = Admissibility::WEAK;
      BLRFactorAlgorithm blr_algo_ = BLRFactorAlgorithm::RL;
      CompressionKernel crn_krnl_ = CompressionKernel::HALF;

      void set_defaults() {
        this->rel_tol_ = default_BLR_rel_tol<real_t>();
        this->abs_tol_ = default_BLR_abs_tol<real_t>();
        this->leaf_size_ = 256;
        this->max_rank_ = 5000;
      }

    };

  } // end namespace BLR
} // end namespace strumpack


#endif // BLR_OPTIONS_HPP
