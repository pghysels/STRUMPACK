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
#ifndef STRUMPACK_SYCL_WRAPPER_HPP
#define STRUMPACK_SYCL_WRAPPER_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <cassert>
#include <memory>
#include <map>

#if __has_include(<sycl/sycl.hpp>)
 #include <sycl/sycl.hpp>
#else
 #include <CL/sycl.hpp>
 namespace sycl = cl::sycl;
#endif
// #include <CL/sycl.hpp>
// #include "mkl.h"
#include "oneapi/mkl.hpp"

#include "GPUWrapper.hpp"


namespace strumpack {
  namespace gpu {

    // TODO get SYCL limits?
    const unsigned int MAX_BLOCKS_Y = 65535;
    const unsigned int MAX_BLOCKS_Z = 65535;

    /// Util function to get the current device (in int).
    int get_sycl_device();

    /// Util function to get the current queue
    sycl::queue& get_sycl_queue();

    const sycl::queue& get_sycl_queue(const Stream& s);
    sycl::queue& get_sycl_queue(Stream& s);

    const sycl::queue& get_sycl_queue(const Handle& s);
    sycl::queue& get_sycl_queue(Handle& s);

  } // end namespace sycl
} // end namespace strumpack

#endif // STRUMPACK_SYCL_WRAPPER_HPP
