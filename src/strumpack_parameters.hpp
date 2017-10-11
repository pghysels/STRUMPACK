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

namespace strumpack { // these are all global variables

  /*! \brief Enumeration for the possible return codes.
   * \ingroup Enumerations */
  enum class ReturnCode {
    SUCCESS,          /*!< Operation completed successfully. */
    MATRIX_NOT_SET,   /*!< The input matrix was not set.     */
    REORDERING_ERROR  /*!< The matrix reordering failed.     */
  };

  namespace params {

#ifndef DOXYGEN_SHOULD_SKIP_THIS

    extern int num_threads;
    extern int task_recursion_cutoff_level;

    extern long long int flops;
    extern long long int bytes;
    //#pragma omp threadprivate(flops, bytes)

    extern std::atomic<long long int> compression_flops;
    extern std::atomic<long long int> sample_flops;
    extern std::atomic<long long int> initial_sample_flops;
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

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  } //end namespace params

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#ifdef COUNT_FLOPS
#define STRUMPACK_FLOPS(n) strumpack::params::flops += n;
#define STRUMPACK_BYTES(n) strumpack::params::bytes += n;
#else
#define STRUMPACK_FLOPS(n) void(0);
#define STRUMPACK_BYTES(n) void(0);
#endif
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

} //end namespace strumpack

#endif // STRUMPACK_PARAMETERS_HPP
