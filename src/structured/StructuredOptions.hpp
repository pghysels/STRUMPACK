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
/*! \file StructuredOptions.hpp
 * \brief Contains the class definition for StructuredOptions, as well
 * as some routines to get default options, and some enumerations,
 */
#ifndef STRUCTURED_OPTIONS_HPP
#define STRUCTURED_OPTIONS_HPP

#include <string>
#include <cassert>

#include "dense/BLASLAPACKWrapper.hpp"

namespace strumpack {
  namespace structured {

    template<typename real_t> inline real_t default_structured_rel_tol() {
      return real_t(1e-4);
    }
    template<typename real_t> inline real_t default_structured_abs_tol() {
      return real_t(1e-10);
    }
    template<> inline float default_structured_rel_tol() {
      return 1e-2;
    }
    template<> inline float default_structured_abs_tol() {
      return 1e-5;
    }

    /**
     * Enumeration of possible structured matrix types.
     * \ingroup Enumerations
     */
    enum class Type : int
      {
       HSS = 0,   /*!< Hierarchically Semi-Separable, see
                    HSS::HSSMatrix and HSS::HSSMatrixMPI */
       BLR,       /*!< Block Low Rank, see BLR::BLRMatrix
                    and BLR::BLRMatrixMPI */
       HODLR,     /*!< Hierarchically Off-Diagonal Low Rank,
                    see HODLR::HODLRMatrix. Does not support
                    float or std::complex<float>. */
       HODBF,     /*!< Hierarchically Off-Diagonal
                    Butterfly, implemented as
                    HODLR::HODLRMatrix. Does not support
                    float or std::complex<float>. */
       BUTTERFLY, /*!< Butterfly matrix, implemented as
                    HODLR::ButterflyMatrix. Does not support
                    float or std::complex<float>. */
       LR,        /*!< Low rank matrix, implemented as
                    HODLR::ButterflyMatrix. Does not support
                    float or std::complex<float>. */
       LOSSY,     /*!< Lossy compression matrix */
       LOSSLESS   /*!< Lossless compressed matrix */
      };

    inline std::string get_name(Type a) {
      switch (a) {
      case Type::HSS: return "HSS";
      case Type::BLR: return "BLR";
      case Type::HODLR: return "HODLR";
      case Type::HODBF: return "HODBF";
      case Type::BUTTERFLY: return "BUTTERFLY";
      case Type::LR: return "LR";
      case Type::LOSSY: return "LOSSY";
      case Type::LOSSLESS: return "LOSSLESS";
      default: return "unknown";
      }
    }

    /**
     * \class StructuredOptions
     * \brief Class containing several options for the
     * StructuredMatrix code and data-structures
     *
     * \tparam scalar_t scalar type, can be float, double,
     * std::complex<float> or std::complex<double>. This is used here
     * mainly because tolerances might depend on the precision.
     */
    template<typename scalar_t> class StructuredOptions {
      using real_t = typename RealType<scalar_t>::value_type;

    public:
      StructuredOptions() {}
      StructuredOptions(Type type) : type_(type) {}

      virtual ~StructuredOptions() {}

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
      void set_type(Type a) {
        type_ = a;
      }
      void set_verbose(bool verbose) { verbose_ = verbose; }

      real_t rel_tol() const { return rel_tol_; }
      real_t abs_tol() const { return abs_tol_; }
      int leaf_size() const { return leaf_size_; }
      int max_rank() const { return max_rank_; }
      Type type() const { return type_; }
      bool verbose() const { return verbose_; }

      virtual void set_from_command_line(int argc, const char* const* cargv);

      virtual void describe_options() const;

    protected:
      Type type_ = Type::BLR;
      real_t rel_tol_ = default_structured_rel_tol<real_t>();
      real_t abs_tol_ = default_structured_abs_tol<real_t>();
      int leaf_size_ = 128;
      int max_rank_ = 5000;
      bool verbose_ = true;
    };

  } // end namespace structured
} // end namespace strumpack


#endif // STRUCTURED_OPTIONS_HPP
