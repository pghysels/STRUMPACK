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
 *
 */
#include "StrumpackParameters.hpp"
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace strumpack {
  namespace params {

#if defined(_OPENMP)
    int num_threads = omp_get_max_threads();
    int task_recursion_cutoff_level = (omp_get_max_threads() == 1) ? 0
      : std::log2(omp_get_max_threads()) + 3;
#else
    int num_threads = 1;
    int task_recursion_cutoff_level = 0;
#endif

    std::atomic<long long int> flops(0);
    std::atomic<long long int> bytes_moved(0);
    std::atomic<long long int> memory(0);
    std::atomic<long long int> peak_memory(0);
    std::atomic<long long int> device_memory(0);
    std::atomic<long long int> peak_device_memory(0);

    std::atomic<long long int> CB_sample_flops(0);
    std::atomic<long long int> sparse_sample_flops(0);
    std::atomic<long long int> extraction_flops(0);
    std::atomic<long long int> ULV_factor_flops(0);
    std::atomic<long long int> schur_flops(0);
    std::atomic<long long int> full_rank_flops(0);
    std::atomic<long long int> random_flops(0);
    std::atomic<long long int> ID_flops(0);
    std::atomic<long long int> QR_flops(0);
    std::atomic<long long int> ortho_flops(0);
    std::atomic<long long int> reduce_sample_flops(0);
    std::atomic<long long int> update_sample_flops(0);
    std::atomic<long long int> hss_solve_flops(0);

    std::atomic<long long int> f11_fill_flops(0);
    std::atomic<long long int> f12_fill_flops(0);
    std::atomic<long long int> f21_fill_flops(0);
    std::atomic<long long int> f22_fill_flops(0);

    std::atomic<long long int> f21_mult_flops(0);
    std::atomic<long long int> invf11_mult_flops(0);
    std::atomic<long long int> f12_mult_flops(0);

  } // end namespace params
} // end namespace strumpack
