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
#ifndef FRONT_FACTORY_HPP
#define FRONT_FACTORY_HPP

#include <iostream>
#include <algorithm>
#include "CSRGraph.hpp"
#include "FrontalMatrixDense.hpp"
#include "FrontalMatrixHSS.hpp"
#include "FrontalMatrixBLR.hpp"
#if defined(STRUMPACK_USE_BPACK)
#include "FrontalMatrixHODLR.hpp"
#endif
#if defined(STRUMPACK_USE_MPI)
#include "FrontalMatrixDenseMPI.hpp"
#include "FrontalMatrixHSSMPI.hpp"
#include "FrontalMatrixBLRMPI.hpp"
#if defined(STRUMPACK_USE_BPACK)
#include "FrontalMatrixHODLRMPI.hpp"
#endif
#endif
#if defined(STRUMPACK_USE_CUBLAS)
#include "FrontalMatrixCUBLAS.hpp"
#endif

namespace strumpack {

  struct FrontCounter {
    int dense, HSS, BLR, HODLR;
    FrontCounter() : dense(0), HSS(0), BLR(0), HODLR(0) {}
    FrontCounter(int* c) : dense(c[0]), HSS(c[1]), BLR(c[2]), HODLR(c[3]) {}
#if defined(STRUMPACK_USE_MPI)
    FrontCounter reduce(const MPIComm& comm) const {
      std::array<int,4> w = {dense, HSS, BLR, HODLR};
      comm.reduce(w.data(), w.size(), MPI_SUM);
      return FrontCounter(w.data());
    }
#endif
  };

  template<typename scalar_t> bool is_CUBLAS
  (const SPOptions<scalar_t>& opts) {
#if defined(STRUMPACK_USE_CUBLAS)
    return opts.use_gpu();
#else
    return false;
#endif
  }
  template<typename scalar_t, typename integer_t> bool is_HSS
  (integer_t dim_sep, bool compressed_parent,
   const SPOptions<scalar_t>& opts) {
    return opts.use_HSS() && compressed_parent &&
      (dim_sep >= opts.HSS_min_sep_size());
  }
  template<typename scalar_t, typename integer_t> bool is_BLR
  (integer_t dim_sep, bool compressed_parent,
   const SPOptions<scalar_t>& opts) {
    return opts.use_BLR() && (dim_sep >= opts.BLR_min_sep_size());
  }
  template<typename scalar_t, typename integer_t> bool is_HODLR
  (integer_t dim_sep, bool compressed_parent,
   const SPOptions<scalar_t>& opts) {
#if defined(STRUMPACK_USE_BPACK)
    return opts.use_HODLR() && (dim_sep >= opts.HODLR_min_sep_size());
#else
    return false;
#endif
  }
  template<typename scalar_t, typename integer_t> bool is_compressed
  (integer_t dim_sep, bool compressed_parent,
   const SPOptions<scalar_t>& opts) {
    return is_HSS(dim_sep, compressed_parent, opts) ||
      is_BLR(dim_sep, compressed_parent, opts) ||
      is_HODLR(dim_sep, compressed_parent, opts);
  }

  template<typename scalar_t, typename integer_t>
  std::unique_ptr<FrontalMatrix<scalar_t,integer_t>> create_frontal_matrix
  (const SPOptions<scalar_t>& opts, integer_t sep, integer_t sep_begin,
   integer_t sep_end, std::vector<integer_t>& upd, bool compressed_parent,
   int level, FrontCounter& fc, bool root=true) {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    auto dim_sep = sep_end - sep_begin;
    std::unique_ptr<F_t> front;
    if (is_CUBLAS(opts)) {
#if defined(STRUMPACK_USE_CUBLAS)
      using FCUBLAS_t = FrontalMatrixCUBLAS<scalar_t,integer_t>;
      front.reset(new FCUBLAS_t(sep, sep_begin, sep_end, upd));
      if (root) fc.dense++;
#endif
    } else if (is_HSS(dim_sep, compressed_parent, opts)) {
      using FHSS_t = FrontalMatrixHSS<scalar_t,integer_t>;
      front.reset(new FHSS_t(sep, sep_begin, sep_end, upd));
      if (root) fc.HSS++;
    } else if (is_BLR(dim_sep, compressed_parent, opts)) {
      using FBLR_t = FrontalMatrixBLR<scalar_t,integer_t>;
      front.reset(new FBLR_t(sep, sep_begin, sep_end, upd));
      if (root) fc.BLR++;
    } else if (is_HODLR(dim_sep, compressed_parent, opts)) {
#if defined(STRUMPACK_USE_BPACK)
      using FHODLR_t = FrontalMatrixHODLR<scalar_t,integer_t>;
      front.reset(new FHODLR_t(sep, sep_begin, sep_end, upd));
      if (root) fc.HODLR++;
#endif
    } else {
      using FD_t = FrontalMatrixDense<scalar_t,integer_t>;
      front.reset(new FD_t(sep, sep_begin, sep_end, upd));
      if (root) fc.dense++;
    }
    return front;
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t, typename integer_t>
  std::unique_ptr<FrontalMatrix<scalar_t,integer_t>> create_frontal_matrix
  (const SPOptions<scalar_t>& opts, integer_t dsep,
   integer_t sep_begin, integer_t sep_end, std::vector<integer_t>& upd,
   bool compressed_parent, int level, FrontCounter& fc,
   const MPIComm& comm, int P, bool root) {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FDMPI_t = FrontalMatrixDenseMPI<scalar_t,integer_t>;
    using FHSSMPI_t = FrontalMatrixHSSMPI<scalar_t,integer_t>;
    using FBLRMPI_t = FrontalMatrixBLRMPI<scalar_t,integer_t>;
#if defined(STRUMPACK_USE_BPACK)
    using FHODLRMPI_t = FrontalMatrixHODLRMPI<scalar_t,integer_t>;
#endif
    auto dim_sep = sep_end - sep_begin;
    std::unique_ptr<F_t> front;
    if (is_HSS(dim_sep, compressed_parent, opts)) {
      front.reset(new FHSSMPI_t(dsep, sep_begin, sep_end, upd, comm, P));
      if (root) fc.HSS++;
    } else if (is_BLR(dim_sep, compressed_parent, opts)) {
      front.reset(new FBLRMPI_t(dsep, sep_begin, sep_end, upd, comm, P));
      if (root) fc.BLR++;
    } else if (is_HODLR(dim_sep, compressed_parent, opts)) {
#if defined(STRUMPACK_USE_BPACK)
      front.reset(new FHODLRMPI_t(dsep, sep_begin, sep_end, upd, comm, P));
      if (root) fc.HODLR++;
#endif
    } else {
      front.reset(new FDMPI_t(dsep, sep_begin, sep_end, upd, comm, P));
      if (root) fc.dense++;
    }
    return front;
  }
#endif

} // end namespace strumpack

#endif // FRONT_FACTORY_HPP
