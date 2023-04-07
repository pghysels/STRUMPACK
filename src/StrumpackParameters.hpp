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
 * \file StrumpackParameters.hpp
 * \brief Contains the definition of some useful (global) variables.
 */
#ifndef STRUMPACK_PARAMETERS_HPP
#define STRUMPACK_PARAMETERS_HPP
#include <atomic>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "StrumpackConfig.hpp"

namespace strumpack { // these are all global variables

  /**
   * \brief Enumeration for the possible return codes.
   * \ingroup Enumerations
   */
  enum class ReturnCode {
    SUCCESS,            /*!< Operation completed successfully.      */
    MATRIX_NOT_SET,     /*!< The input matrix was not set.          */
    REORDERING_ERROR,   /*!< The matrix reordering failed.          */
    ZERO_PIVOT,         /*!< A zero pivot was encountered.          */
    NO_CONVERGENCE,     /*!< The iterative solver did not converge. */
    INACCURATE_INERTIA  /*!< Inertia could not be computed.         */
  };

  inline std::ostream& operator<<(std::ostream& os, ReturnCode& e) {
    switch (e) {
    case ReturnCode::SUCCESS:            os << "SUCCESS"; break;
    case ReturnCode::MATRIX_NOT_SET:     os << "MATRIX_NOT_SET"; break;
    case ReturnCode::REORDERING_ERROR:   os << "REORDERING_ERROR"; break;
    case ReturnCode::ZERO_PIVOT:         os << "ZERO_PIVOT"; break;
    case ReturnCode::NO_CONVERGENCE:     os << "NO_CONVERGENCE"; break;
    case ReturnCode::INACCURATE_INERTIA: os << "INACCURATE_INERTIA"; break;
    }
    return os;
  }

  namespace params {

#ifndef DOXYGEN_SHOULD_SKIP_THIS

    extern int num_threads;
    extern int task_recursion_cutoff_level;

    extern std::atomic<long long int> flops;
    extern std::atomic<long long int> bytes_moved;
    extern std::atomic<long long int> memory;
    extern std::atomic<long long int> peak_memory;
    extern std::atomic<long long int> device_memory;
    extern std::atomic<long long int> peak_device_memory;

    extern std::atomic<long long int> CB_sample_flops;
    extern std::atomic<long long int> sparse_sample_flops;
    extern std::atomic<long long int> extraction_flops;
    extern std::atomic<long long int> ULV_factor_flops;
    extern std::atomic<long long int> schur_flops;
    extern std::atomic<long long int> full_rank_flops;
    extern std::atomic<long long int> random_flops;
    extern std::atomic<long long int> ID_flops;
    extern std::atomic<long long int> ortho_flops;
    extern std::atomic<long long int> QR_flops;
    extern std::atomic<long long int> reduce_sample_flops;
    extern std::atomic<long long int> update_sample_flops;
    extern std::atomic<long long int> hss_solve_flops;

    extern std::atomic<long long int> f11_fill_flops;
    extern std::atomic<long long int> f12_fill_flops;
    extern std::atomic<long long int> f21_fill_flops;
    extern std::atomic<long long int> f22_fill_flops;

    extern std::atomic<long long int> f21_mult_flops;
    extern std::atomic<long long int> invf11_mult_flops;
    extern std::atomic<long long int> f12_mult_flops;

#endif //DOXYGEN_SHOULD_SKIP_THIS

  } //end namespace params

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#if defined(STRUMPACK_COUNT_FLOPS)
#define STRUMPACK_FLOPS(n)                      \
  strumpack::params::flops += n;
#define STRUMPACK_BYTES(n)                      \
  strumpack::params::bytes_moved += n;
#define STRUMPACK_ID_FLOPS(n)                   \
  strumpack::params::ID_flops += n;
#define STRUMPACK_QR_FLOPS(n)                   \
  strumpack::params::QR_flops += n;
#define STRUMPACK_ORTHO_FLOPS(n)                \
  strumpack::params::ortho_flops += n;
#define STRUMPACK_REDUCE_SAMPLE_FLOPS(n)        \
  strumpack::params::reduce_sample_flops += n;
#define STRUMPACK_UPDATE_SAMPLE_FLOPS(n)        \
  strumpack::params::update_sample_flops += n;
#define STRUMPACK_RANDOM_FLOPS(n)               \
  strumpack::params::random_flops += n;
#define STRUMPACK_SPARSE_SAMPLE_FLOPS(n)        \
  strumpack::params::sparse_sample_flops += n;
#define STRUMPACK_FULL_RANK_FLOPS(n)            \
  strumpack::params::full_rank_flops += n;
#define STRUMPACK_EXTRACTION_FLOPS(n)           \
  strumpack::params::extraction_flops += n;
#define STRUMPACK_ULV_FACTOR_FLOPS(n)           \
  strumpack::params::ULV_factor_flops += n;
#define STRUMPACK_SCHUR_FLOPS(n)                \
  strumpack::params::schur_flops += n;
#define STRUMPACK_CB_SAMPLE_FLOPS(n)            \
  strumpack::params::CB_sample_flops += n;
#define STRUMPACK_HSS_SOLVE_FLOPS(n)            \
  strumpack::params::hss_solve_flops += n;

#define STRUMPACK_HODLR_F11_FILL_FLOPS(n)       \
  strumpack::params::f11_fill_flops += n;
#define STRUMPACK_HODLR_F12_FILL_FLOPS(n)       \
  strumpack::params::f12_fill_flops += n
#define STRUMPACK_HODLR_F21_FILL_FLOPS(n)       \
  strumpack::params::f21_fill_flops += n
#define STRUMPACK_HODLR_F22_FILL_FLOPS(n)       \
  strumpack::params::f22_fill_flops += n

#define STRUMPACK_HODLR_F21_MULT_FLOPS(n)       \
  strumpack::params::f21_mult_flops += n
#define STRUMPACK_HODLR_INVF11_MULT_FLOPS(n)    \
  strumpack::params::invf11_mult_flops += n
#define STRUMPACK_HODLR_F12_MULT_FLOPS(n)       \
  strumpack::params::f12_mult_flops += n

#define STRUMPACK_ADD_MEMORY(n) {                                       \
    strumpack::params::memory += n;                                     \
    auto new_peak_ = std::max(strumpack::params::memory.load(),         \
                              strumpack::params::peak_memory.load());   \
    auto old_peak_ = strumpack::params::peak_memory.load();             \
    while (new_peak_ > old_peak_ &&                                     \
           !strumpack::params::peak_memory.compare_exchange_weak        \
           (old_peak_, new_peak_)) { }                                  \
  }
#define STRUMPACK_ADD_DEVICE_MEMORY(n) {                                \
    strumpack::params::device_memory += n;                              \
    auto new_peak_ = std::max(strumpack::params::device_memory.load(),  \
                              strumpack::params::peak_device_memory.load()); \
    auto old_peak_ = strumpack::params::peak_device_memory.load();      \
    while (new_peak_ > old_peak_ &&                                     \
           !strumpack::params::peak_device_memory.compare_exchange_weak \
           (old_peak_, new_peak_)) { }                                  \
  }

#define STRUMPACK_SUB_MEMORY(n)                 \
  strumpack::params::memory -= n;
#define STRUMPACK_SUB_DEVICE_MEMORY(n)          \
  strumpack::params::device_memory -= n;

#else

#define STRUMPACK_FLOPS(n) void(0);
#define STRUMPACK_BYTES(n) void(0);
#define STRUMPACK_ID_FLOPS(n) void(0);
#define STRUMPACK_QR_FLOPS(n) void(0);
#define STRUMPACK_ORTHO_FLOPS(n) void(0);
#define STRUMPACK_REDUCE_SAMPLE_FLOPS(n) void(0);
#define STRUMPACK_UPDATE_SAMPLE_FLOPS(n) void(0);
#define STRUMPACK_RANDOM_FLOPS(n) void(0);
#define STRUMPACK_SPARSE_SAMPLE_FLOPS(n) void(0);
#define STRUMPACK_FULL_RANK_FLOPS(n) void(0);
#define STRUMPACK_EXTRACTION_FLOPS(n) void(0);
#define STRUMPACK_ULV_FACTOR_FLOPS(n) void(0);
#define STRUMPACK_SCHUR_FLOPS(n) void(0);
#define STRUMPACK_CB_SAMPLE_FLOPS(n) void(0);
#define STRUMPACK_HSS_SOLVE_FLOPS(n) void(0);

#define STRUMPACK_HODLR_F11_FILL_FLOPS(n) void(0);
#define STRUMPACK_HODLR_F12_FILL_FLOPS(n) void(0);
#define STRUMPACK_HODLR_F21_FILL_FLOPS(n) void(0);
#define STRUMPACK_HODLR_F22_FILL_FLOPS(n) void(0);

#define STRUMPACK_HODLR_F21_MULT_FLOPS(n) void(0);
#define STRUMPACK_HODLR_INVF11_MULT_FLOPS(n) void(0);
#define STRUMPACK_HODLR_F12_FILL_FLOPS(n) void(0);

#define STRUMPACK_ADD_MEMORY(n) void(0);
#define STRUMPACK_SUB_MEMORY(n) void(0);
#define STRUMPACK_ADD_DEVICE_MEMORY(n) void(0);
#define STRUMPACK_SUB_DEVICE_MEMORY(n) void(0);

#endif
#endif // DOXYGEN_SHOULD_SKIP_THIS

} //end namespace strumpack

#endif // STRUMPACK_PARAMETERS_HPP
