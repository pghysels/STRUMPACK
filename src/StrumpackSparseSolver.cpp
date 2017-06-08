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
 * (5) year renewals, the U.S. Government is granted for itself and others acting
 * on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, prepare derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 *
 */
#include "StrumpackSparseSolver.h"
#include "StrumpackSparseSolver.hpp"
#include "StrumpackSparseSolverMPI.hpp"
#include "StrumpackSparseSolverMPIDist.hpp"
#include "CSRMatrix.hpp"

using namespace strumpack;

// TODO update C bindings. For instance: use_hss and minimum_front_size iso levels

#define CASTS(x) (static_cast<StrumpackSparseSolver<float,int>*>(x))
#define CASTD(x) (static_cast<StrumpackSparseSolver<double,int>*>(x))
#define CASTC(x) (static_cast<StrumpackSparseSolver<std::complex<float>,int>*>(x))
#define CASTZ(x) (static_cast<StrumpackSparseSolver<std::complex<double>,int>*>(x))
#define CASTS64(x) (static_cast<StrumpackSparseSolver<float,int64_t>*>(x))
#define CASTD64(x) (static_cast<StrumpackSparseSolver<double,int64_t>*>(x))
#define CASTC64(x) (static_cast<StrumpackSparseSolver<std::complex<float>,int64_t>*>(x))
#define CASTZ64(x) (static_cast<StrumpackSparseSolver<std::complex<double>,int64_t>*>(x))

#define CASTSMPI(x) (static_cast<StrumpackSparseSolverMPI<float,int>*>(x))
#define CASTDMPI(x) (static_cast<StrumpackSparseSolverMPI<double,int>*>(x))
#define CASTCMPI(x) (static_cast<StrumpackSparseSolverMPI<std::complex<float>,int>*>(x))
#define CASTZMPI(x) (static_cast<StrumpackSparseSolverMPI<std::complex<double>,int>*>(x))
#define CASTS64MPI(x) (static_cast<StrumpackSparseSolverMPI<float,int64_t>*>(x))
#define CASTD64MPI(x) (static_cast<StrumpackSparseSolverMPI<double,int64_t>*>(x))
#define CASTC64MPI(x) (static_cast<StrumpackSparseSolverMPI<std::complex<float>,int64_t>*>(x))
#define CASTZ64MPI(x) (static_cast<StrumpackSparseSolverMPI<std::complex<double>,int64_t>*>(x))

#define CASTSMPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<float,int>*>(x))
#define CASTDMPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<double,int>*>(x))
#define CASTCMPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<std::complex<float>,int>*>(x))
#define CASTZMPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<std::complex<double>,int>*>(x))
#define CASTS64MPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<float,int64_t>*>(x))
#define CASTD64MPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<double,int64_t>*>(x))
#define CASTC64MPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<std::complex<float>,int64_t>*>(x))
#define CASTZ64MPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<std::complex<double>,int64_t>*>(x))

#define switch_precision(m)						\
  switch (S.interface) {						\
  case STRUMPACK_MT: {							\
    switch (S.precision) {						\
    case STRUMPACK_FLOAT:            CASTS(S.solver)->m;   break;	\
    case STRUMPACK_DOUBLE:           CASTD(S.solver)->m;   break;	\
    case STRUMPACK_FLOATCOMPLEX:     CASTC(S.solver)->m;   break;	\
    case STRUMPACK_DOUBLECOMPLEX:    CASTZ(S.solver)->m;   break;	\
    case STRUMPACK_FLOAT_64:         CASTS64(S.solver)->m; break;	\
    case STRUMPACK_DOUBLE_64:        CASTD64(S.solver)->m; break;	\
    case STRUMPACK_FLOATCOMPLEX_64:  CASTC64(S.solver)->m; break;	\
    case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64(S.solver)->m; break;	\
    }									\
  } break;								\
  case STRUMPACK_MPI: {							\
    switch (S.precision) {						\
    case STRUMPACK_FLOAT:            CASTSMPI(S.solver)->m;   break;	\
    case STRUMPACK_DOUBLE:           CASTDMPI(S.solver)->m;   break;	\
    case STRUMPACK_FLOATCOMPLEX:     CASTCMPI(S.solver)->m;   break;	\
    case STRUMPACK_DOUBLECOMPLEX:    CASTZMPI(S.solver)->m;   break;	\
    case STRUMPACK_FLOAT_64:         CASTS64MPI(S.solver)->m; break;	\
    case STRUMPACK_DOUBLE_64:        CASTD64MPI(S.solver)->m; break;	\
    case STRUMPACK_FLOATCOMPLEX_64:  CASTC64MPI(S.solver)->m; break;	\
    case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64MPI(S.solver)->m; break;	\
    }									\
  } break;								\
  case STRUMPACK_MPI_DIST: {						\
    switch (S.precision) {						\
    case STRUMPACK_FLOAT:            CASTSMPIDIST(S.solver)->m;   break; \
    case STRUMPACK_DOUBLE:           CASTDMPIDIST(S.solver)->m;   break; \
    case STRUMPACK_FLOATCOMPLEX:     CASTCMPIDIST(S.solver)->m;   break; \
    case STRUMPACK_DOUBLECOMPLEX:    CASTZMPIDIST(S.solver)->m;   break; \
    case STRUMPACK_FLOAT_64:         CASTS64MPIDIST(S.solver)->m; break; \
    case STRUMPACK_DOUBLE_64:        CASTD64MPIDIST(S.solver)->m; break; \
    case STRUMPACK_FLOATCOMPLEX_64:  CASTC64MPIDIST(S.solver)->m; break; \
    case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64MPIDIST(S.solver)->m; break; \
    }									\
  } break;								\
  }

#define switch_precision_return(m,r)					\
  switch (S.interface) {						\
  case STRUMPACK_MT: {							\
    switch (S.precision) {						\
    case STRUMPACK_FLOAT:            r = CASTS(S.solver)->m;   break;	\
    case STRUMPACK_DOUBLE:           r = CASTD(S.solver)->m;   break;	\
    case STRUMPACK_FLOATCOMPLEX:     r = CASTC(S.solver)->m;   break;	\
    case STRUMPACK_DOUBLECOMPLEX:    r = CASTZ(S.solver)->m;   break;	\
    case STRUMPACK_FLOAT_64:         r = CASTS64(S.solver)->m; break;	\
    case STRUMPACK_DOUBLE_64:        r = CASTD64(S.solver)->m; break;	\
    case STRUMPACK_FLOATCOMPLEX_64:  r = CASTC64(S.solver)->m; break;	\
    case STRUMPACK_DOUBLECOMPLEX_64: r = CASTZ64(S.solver)->m; break;	\
    }									\
  } break;								\
  case STRUMPACK_MPI: {							\
    switch (S.precision) {						\
    case STRUMPACK_FLOAT:            r = CASTSMPI(S.solver)->m;   break; \
    case STRUMPACK_DOUBLE:           r = CASTDMPI(S.solver)->m;   break; \
    case STRUMPACK_FLOATCOMPLEX:     r = CASTCMPI(S.solver)->m;   break; \
    case STRUMPACK_DOUBLECOMPLEX:    r = CASTZMPI(S.solver)->m;   break; \
    case STRUMPACK_FLOAT_64:         r = CASTS64MPI(S.solver)->m; break; \
    case STRUMPACK_DOUBLE_64:        r = CASTD64MPI(S.solver)->m; break; \
    case STRUMPACK_FLOATCOMPLEX_64:  r = CASTC64MPI(S.solver)->m; break; \
    case STRUMPACK_DOUBLECOMPLEX_64: r = CASTZ64MPI(S.solver)->m; break; \
    }									\
  } break;								\
  case STRUMPACK_MPI_DIST: {						\
    switch (S.precision) {						\
    case STRUMPACK_FLOAT:            r = CASTSMPIDIST(S.solver)->m;   break; \
    case STRUMPACK_DOUBLE:           r = CASTDMPIDIST(S.solver)->m;   break; \
    case STRUMPACK_FLOATCOMPLEX:     r = CASTCMPIDIST(S.solver)->m;   break; \
    case STRUMPACK_DOUBLECOMPLEX:    r = CASTZMPIDIST(S.solver)->m;   break; \
    case STRUMPACK_FLOAT_64:         r = CASTS64MPIDIST(S.solver)->m; break; \
    case STRUMPACK_DOUBLE_64:        r = CASTD64MPIDIST(S.solver)->m; break; \
    case STRUMPACK_FLOATCOMPLEX_64:  r = CASTC64MPIDIST(S.solver)->m; break; \
    case STRUMPACK_DOUBLECOMPLEX_64: r = CASTZ64MPIDIST(S.solver)->m; break; \
    }									\
  } break;								\
  }

extern "C" {
  void STRUMPACK_init(STRUMPACK_SparseSolver* S, MPI_Comm comm, STRUMPACK_PRECISION precision,
		      STRUMPACK_INTERFACE interface, int argc, char* argv[], int verbose) {
    S->precision = precision;
    S->interface = interface;
    bool v = static_cast<bool>(verbose);
    switch (interface) {
    case STRUMPACK_MT: {
      switch (precision) {
      case STRUMPACK_FLOAT:            S->solver = static_cast<void*>(new StrumpackSparseSolver<float,int>(argc, argv, v)); break;
      case STRUMPACK_DOUBLE:           S->solver = static_cast<void*>(new StrumpackSparseSolver<double,int>(argc, argv, v)); break;
      case STRUMPACK_FLOATCOMPLEX:     S->solver = static_cast<void*>(new StrumpackSparseSolver<std::complex<float>,int>(argc, argv, v)); break;
      case STRUMPACK_DOUBLECOMPLEX:    S->solver = static_cast<void*>(new StrumpackSparseSolver<std::complex<double>,int>(argc, argv, v)); break;
      case STRUMPACK_FLOAT_64:         S->solver = static_cast<void*>(new StrumpackSparseSolver<float,int64_t>(argc, argv, v)); break;
      case STRUMPACK_DOUBLE_64:        S->solver = static_cast<void*>(new StrumpackSparseSolver<double,int64_t>(argc, argv, v)); break;
      case STRUMPACK_FLOATCOMPLEX_64:  S->solver = static_cast<void*>(new StrumpackSparseSolver<std::complex<float>,int64_t>(argc, argv, v)); break;
      case STRUMPACK_DOUBLECOMPLEX_64: S->solver = static_cast<void*>(new StrumpackSparseSolver<std::complex<double>,int64_t>(argc, argv, v)); break;
      default: std::cerr << "ERROR: wrong precision!" << std::endl;
      }
    } break;
    case STRUMPACK_MPI: {
      switch (precision) {
      case STRUMPACK_FLOAT:            S->solver = static_cast<void*>(new StrumpackSparseSolverMPI<float,int>(comm, argc, argv, v)); break;
      case STRUMPACK_DOUBLE:           S->solver = static_cast<void*>(new StrumpackSparseSolverMPI<double,int>(comm, argc, argv, v)); break;
      case STRUMPACK_FLOATCOMPLEX:     S->solver = static_cast<void*>(new StrumpackSparseSolverMPI<std::complex<float>,int>(comm, argc, argv, v)); break;
      case STRUMPACK_DOUBLECOMPLEX:    S->solver = static_cast<void*>(new StrumpackSparseSolverMPI<std::complex<double>,int>(comm, argc, argv, v)); break;
      case STRUMPACK_FLOAT_64:         S->solver = static_cast<void*>(new StrumpackSparseSolverMPI<float,int64_t>(comm, argc, argv, v)); break;
      case STRUMPACK_DOUBLE_64:        S->solver = static_cast<void*>(new StrumpackSparseSolverMPI<double,int64_t>(comm, argc, argv, v)); break;
      case STRUMPACK_FLOATCOMPLEX_64:  S->solver = static_cast<void*>(new StrumpackSparseSolverMPI<std::complex<float>,int64_t>(comm, argc, argv, v)); break;
      case STRUMPACK_DOUBLECOMPLEX_64: S->solver = static_cast<void*>(new StrumpackSparseSolverMPI<std::complex<double>,int64_t>(comm, argc, argv, v)); break;
      default: std::cerr << "ERROR: wrong precision!" << std::endl;
      }
    } break;
    case STRUMPACK_MPI_DIST: {
      switch (precision) {
      case STRUMPACK_FLOAT:            S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<float,int>(comm, argc, argv, v)); break;
      case STRUMPACK_DOUBLE:           S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<double,int>(comm, argc, argv, v)); break;
      case STRUMPACK_FLOATCOMPLEX:     S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<std::complex<float>,int>(comm, argc, argv, v)); break;
      case STRUMPACK_DOUBLECOMPLEX:    S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<std::complex<double>,int>(comm, argc, argv, v)); break;
      case STRUMPACK_FLOAT_64:         S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<float,int64_t>(comm, argc, argv, v)); break;
      case STRUMPACK_DOUBLE_64:        S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<double,int64_t>(comm, argc, argv, v)); break;
      case STRUMPACK_FLOATCOMPLEX_64:  S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<std::complex<float>,int64_t>(comm, argc, argv, v)); break;
      case STRUMPACK_DOUBLECOMPLEX_64: S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<std::complex<double>,int64_t>(comm, argc, argv, v)); break;
      default: std::cerr << "ERROR: wrong precision!" << std::endl;
      }
    } break;
    default: std::cerr << "ERROR: wrong interface!" << std::endl;
    }
  }

  void STRUMPACK_destroy(STRUMPACK_SparseSolver* S) {
    switch (S->interface) {
    case STRUMPACK_MT: {
      switch (S->precision) {
      case STRUMPACK_FLOAT:            delete CASTS(S->solver);   break;
      case STRUMPACK_DOUBLE:           delete CASTD(S->solver);   break;
      case STRUMPACK_FLOATCOMPLEX:     delete CASTC(S->solver);   break;
      case STRUMPACK_DOUBLECOMPLEX:    delete CASTZ(S->solver);   break;
      case STRUMPACK_FLOAT_64:         delete CASTS64(S->solver); break;
      case STRUMPACK_DOUBLE_64:        delete CASTD64(S->solver); break;
      case STRUMPACK_FLOATCOMPLEX_64:  delete CASTC64(S->solver); break;
      case STRUMPACK_DOUBLECOMPLEX_64: delete CASTZ64(S->solver); break;
      }
    } break;
    case STRUMPACK_MPI: {
      switch (S->precision) {
      case STRUMPACK_FLOAT:            delete CASTSMPI(S->solver);   break;
      case STRUMPACK_DOUBLE:           delete CASTDMPI(S->solver);   break;
      case STRUMPACK_FLOATCOMPLEX:     delete CASTCMPI(S->solver);   break;
      case STRUMPACK_DOUBLECOMPLEX:    delete CASTZMPI(S->solver);   break;
      case STRUMPACK_FLOAT_64:         delete CASTS64MPI(S->solver); break;
      case STRUMPACK_DOUBLE_64:        delete CASTD64MPI(S->solver); break;
      case STRUMPACK_FLOATCOMPLEX_64:  delete CASTC64MPI(S->solver); break;
      case STRUMPACK_DOUBLECOMPLEX_64: delete CASTZ64MPI(S->solver); break;
      }
    } break;
    case STRUMPACK_MPI_DIST: {
      switch (S->precision) {
      case STRUMPACK_FLOAT:            delete CASTSMPIDIST(S->solver);   break;
      case STRUMPACK_DOUBLE:           delete CASTDMPIDIST(S->solver);   break;
      case STRUMPACK_FLOATCOMPLEX:     delete CASTCMPIDIST(S->solver);   break;
      case STRUMPACK_DOUBLECOMPLEX:    delete CASTZMPIDIST(S->solver);   break;
      case STRUMPACK_FLOAT_64:         delete CASTS64MPIDIST(S->solver); break;
      case STRUMPACK_DOUBLE_64:        delete CASTD64MPIDIST(S->solver); break;
      case STRUMPACK_FLOATCOMPLEX_64:  delete CASTC64MPIDIST(S->solver); break;
      case STRUMPACK_DOUBLECOMPLEX_64: delete CASTZ64MPIDIST(S->solver); break;
      }
    } break;
    }
    S->solver = NULL;
  }

  void STRUMPACK_set_csr_matrix(STRUMPACK_SparseSolver S, void* N, void* row_ptr, void* col_ind, void* values, int symmetric_pattern) {
    switch (S.interface) {
    case STRUMPACK_MT: {
      switch (S.precision) {
      case STRUMPACK_FLOAT:            CASTS(S.solver)->set_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<float*>(values), symmetric_pattern); break;
      case STRUMPACK_DOUBLE:           CASTD(S.solver)->set_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<double*>(values), symmetric_pattern); break;
      case STRUMPACK_FLOATCOMPLEX:     CASTC(S.solver)->set_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<std::complex<float>*>(values), symmetric_pattern); break;
      case STRUMPACK_DOUBLECOMPLEX:    CASTZ(S.solver)->set_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<std::complex<double>*>(values), symmetric_pattern); break;
      case STRUMPACK_FLOAT_64:         CASTS64(S.solver)->set_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<float*>(values), symmetric_pattern); break;
      case STRUMPACK_DOUBLE_64:        CASTD64(S.solver)->set_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<double*>(values), symmetric_pattern); break;
      case STRUMPACK_FLOATCOMPLEX_64:  CASTC64(S.solver)->set_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<std::complex<float>*>(values), symmetric_pattern); break;
      case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64(S.solver)->set_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<std::complex<double>*>(values), symmetric_pattern); break;
      }
    } break;
    case STRUMPACK_MPI: {
      switch (S.precision) {
      case STRUMPACK_FLOAT:            CASTSMPI(S.solver)->set_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<float*>(values), symmetric_pattern); break;
      case STRUMPACK_DOUBLE:           CASTDMPI(S.solver)->set_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<double*>(values), symmetric_pattern); break;
      case STRUMPACK_FLOATCOMPLEX:     CASTCMPI(S.solver)->set_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<std::complex<float>*>(values), symmetric_pattern); break;
      case STRUMPACK_DOUBLECOMPLEX:    CASTZMPI(S.solver)->set_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<std::complex<double>*>(values), symmetric_pattern); break;
      case STRUMPACK_FLOAT_64:         CASTS64MPI(S.solver)->set_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<float*>(values), symmetric_pattern); break;
      case STRUMPACK_DOUBLE_64:        CASTD64MPI(S.solver)->set_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<double*>(values), symmetric_pattern); break;
      case STRUMPACK_FLOATCOMPLEX_64:  CASTC64MPI(S.solver)->set_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<std::complex<float>*>(values), symmetric_pattern); break;
      case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64MPI(S.solver)->set_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<std::complex<double>*>(values), symmetric_pattern); break;
      }
    } break;
    case STRUMPACK_MPI_DIST: {
      switch (S.precision) {
      case STRUMPACK_FLOAT:            CASTSMPIDIST(S.solver)->set_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<float*>(values), symmetric_pattern); break;
      case STRUMPACK_DOUBLE:           CASTDMPIDIST(S.solver)->set_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<double*>(values), symmetric_pattern); break;
      case STRUMPACK_FLOATCOMPLEX:     CASTCMPIDIST(S.solver)->set_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<std::complex<float>*>(values), symmetric_pattern); break;
      case STRUMPACK_DOUBLECOMPLEX:    CASTZMPIDIST(S.solver)->set_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<std::complex<double>*>(values), symmetric_pattern); break;
      case STRUMPACK_FLOAT_64:         CASTS64MPIDIST(S.solver)->set_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<float*>(values), symmetric_pattern); break;
      case STRUMPACK_DOUBLE_64:        CASTD64MPIDIST(S.solver)->set_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<double*>(values), symmetric_pattern); break;
      case STRUMPACK_FLOATCOMPLEX_64:  CASTC64MPIDIST(S.solver)->set_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<std::complex<float>*>(values), symmetric_pattern); break;
      case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64MPIDIST(S.solver)->set_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<std::complex<double>*>(values), symmetric_pattern); break;
      }
    } break;
    }
  }

  void STRUMPACK_set_distributed_csr_matrix(STRUMPACK_SparseSolver S, void* N, void* row_ptr, void* col_ind, void* values, void* dist, int symmetric_pattern) {
    if (S.interface != STRUMPACK_MPI_DIST) { std::cerr << "ERROR: interface != STRUMPACK_MPI_DIST" << std::endl; return; }
    switch (S.precision) {
    case STRUMPACK_FLOAT:            CASTSMPIDIST(S.solver)->set_distributed_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<float*>(values), reinterpret_cast<int*>(dist), symmetric_pattern); break;
    case STRUMPACK_DOUBLE:           CASTDMPIDIST(S.solver)->set_distributed_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<double*>(values), reinterpret_cast<int*>(dist), symmetric_pattern); break;
    case STRUMPACK_FLOATCOMPLEX:     CASTCMPIDIST(S.solver)->set_distributed_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<std::complex<float>*>(values), reinterpret_cast<int*>(dist), symmetric_pattern); break;
    case STRUMPACK_DOUBLECOMPLEX:    CASTZMPIDIST(S.solver)->set_distributed_csr_matrix(*reinterpret_cast<int*>(N), reinterpret_cast<int*>(row_ptr), reinterpret_cast<int*>(col_ind), reinterpret_cast<std::complex<double>*>(values), reinterpret_cast<int*>(dist), symmetric_pattern); break;
    case STRUMPACK_FLOAT_64:         CASTS64MPIDIST(S.solver)->set_distributed_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<float*>(values), reinterpret_cast<int64_t*>(dist), symmetric_pattern); break;
    case STRUMPACK_DOUBLE_64:        CASTD64MPIDIST(S.solver)->set_distributed_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<double*>(values), reinterpret_cast<int64_t*>(dist), symmetric_pattern); break;
    case STRUMPACK_FLOATCOMPLEX_64:  CASTC64MPIDIST(S.solver)->set_distributed_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<std::complex<float>*>(values), reinterpret_cast<int64_t*>(dist), symmetric_pattern); break;
    case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64MPIDIST(S.solver)->set_distributed_csr_matrix(*reinterpret_cast<int64_t*>(N), reinterpret_cast<int64_t*>(row_ptr), reinterpret_cast<int64_t*>(col_ind), reinterpret_cast<std::complex<double>*>(values), reinterpret_cast<int64_t*>(dist), symmetric_pattern); break;
    }
  }

  void STRUMPACK_set_MPIAIJ_matrix(STRUMPACK_SparseSolver S, void* n, void* d_ptr, void* d_ind, void* d_val,
				   void* o_ptr, void* o_ind, void* o_val, void* garray) {
    if (S.interface != STRUMPACK_MPI_DIST) { std::cerr << "ERROR: interface != STRUMPACK_MPI_DIST" << std::endl; return; }
    switch (S.precision) {
    case STRUMPACK_FLOAT: CASTSMPIDIST(S.solver)->set_MPIAIJ_matrix(*reinterpret_cast<int*>(n), reinterpret_cast<int*>(d_ptr), reinterpret_cast<int*>(d_ind), reinterpret_cast<float*>(d_val),
								    reinterpret_cast<int*>(o_ptr), reinterpret_cast<int*>(o_ind), reinterpret_cast<float*>(o_val), reinterpret_cast<int*>(garray)); break;
    case STRUMPACK_DOUBLE: CASTDMPIDIST(S.solver)->set_MPIAIJ_matrix(*reinterpret_cast<int*>(n), reinterpret_cast<int*>(d_ptr), reinterpret_cast<int*>(d_ind), reinterpret_cast<double*>(d_val),
								     reinterpret_cast<int*>(o_ptr), reinterpret_cast<int*>(o_ind), reinterpret_cast<double*>(o_val), reinterpret_cast<int*>(garray)); break;
    case STRUMPACK_FLOATCOMPLEX: CASTCMPIDIST(S.solver)->set_MPIAIJ_matrix(*reinterpret_cast<int*>(n), reinterpret_cast<int*>(d_ptr), reinterpret_cast<int*>(d_ind), reinterpret_cast<std::complex<float>*>(d_val),
									   reinterpret_cast<int*>(o_ptr), reinterpret_cast<int*>(o_ind), reinterpret_cast<std::complex<float>*>(o_val), reinterpret_cast<int*>(garray)); break;
    case STRUMPACK_DOUBLECOMPLEX: CASTZMPIDIST(S.solver)->set_MPIAIJ_matrix(*reinterpret_cast<int*>(n), reinterpret_cast<int*>(d_ptr), reinterpret_cast<int*>(d_ind), reinterpret_cast<std::complex<double>*>(d_val),
									    reinterpret_cast<int*>(o_ptr), reinterpret_cast<int*>(o_ind), reinterpret_cast<std::complex<double>*>(o_val), reinterpret_cast<int*>(garray)); break;
    case STRUMPACK_FLOAT_64: CASTS64MPIDIST(S.solver)->set_MPIAIJ_matrix(*reinterpret_cast<int64_t*>(n), reinterpret_cast<int64_t*>(d_ptr), reinterpret_cast<int64_t*>(d_ind), reinterpret_cast<float*>(d_val),
									 reinterpret_cast<int64_t*>(o_ptr), reinterpret_cast<int64_t*>(o_ind), reinterpret_cast<float*>(o_val), reinterpret_cast<int64_t*>(garray)); break;
    case STRUMPACK_DOUBLE_64: CASTD64MPIDIST(S.solver)->set_MPIAIJ_matrix(*reinterpret_cast<int64_t*>(n), reinterpret_cast<int64_t*>(d_ptr), reinterpret_cast<int64_t*>(d_ind), reinterpret_cast<double*>(d_val),
									  reinterpret_cast<int64_t*>(o_ptr), reinterpret_cast<int64_t*>(o_ind), reinterpret_cast<double*>(o_val), reinterpret_cast<int64_t*>(garray)); break;
    case STRUMPACK_FLOATCOMPLEX_64: CASTC64MPIDIST(S.solver)->set_MPIAIJ_matrix(*reinterpret_cast<int64_t*>(n), reinterpret_cast<int64_t*>(d_ptr), reinterpret_cast<int64_t*>(d_ind), reinterpret_cast<std::complex<float>*>(d_val),
										reinterpret_cast<int64_t*>(o_ptr), reinterpret_cast<int64_t*>(o_ind), reinterpret_cast<std::complex<float>*>(o_val), reinterpret_cast<int64_t*>(garray)); break;
    case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64MPIDIST(S.solver)->set_MPIAIJ_matrix(*reinterpret_cast<int64_t*>(n), reinterpret_cast<int64_t*>(d_ptr), reinterpret_cast<int64_t*>(d_ind), reinterpret_cast<std::complex<double>*>(d_val),
										 reinterpret_cast<int64_t*>(o_ptr), reinterpret_cast<int64_t*>(o_ind), reinterpret_cast<std::complex<double>*>(o_val), reinterpret_cast<int64_t*>(garray)); break;
    }
  }

  STRUMPACK_RETURN_CODE STRUMPACK_solve(STRUMPACK_SparseSolver S, void* b, void* x, int use_initial_guess) {
    switch (S.interface) {
    case STRUMPACK_MT: {
      switch (S.precision) {
      case STRUMPACK_FLOAT:            return static_cast<STRUMPACK_RETURN_CODE>(CASTS(S.solver)->solve(reinterpret_cast<float*>(b), reinterpret_cast<float*>(x), use_initial_guess)); break;
      case STRUMPACK_DOUBLE:           return static_cast<STRUMPACK_RETURN_CODE>(CASTD(S.solver)->solve(reinterpret_cast<double*>(b), reinterpret_cast<double*>(x), use_initial_guess)); break;
      case STRUMPACK_FLOATCOMPLEX:     return static_cast<STRUMPACK_RETURN_CODE>(CASTC(S.solver)->solve(reinterpret_cast<std::complex<float>*>(b), reinterpret_cast<std::complex<float>*>(x), use_initial_guess)); break;
      case STRUMPACK_DOUBLECOMPLEX:    return static_cast<STRUMPACK_RETURN_CODE>(CASTZ(S.solver)->solve(reinterpret_cast<std::complex<double>*>(b), reinterpret_cast<std::complex<double>*>(x), use_initial_guess)); break;
      case STRUMPACK_FLOAT_64:         return static_cast<STRUMPACK_RETURN_CODE>(CASTS64(S.solver)->solve(reinterpret_cast<float*>(b), reinterpret_cast<float*>(x), use_initial_guess)); break;
      case STRUMPACK_DOUBLE_64:        return static_cast<STRUMPACK_RETURN_CODE>(CASTD64(S.solver)->solve(reinterpret_cast<double*>(b), reinterpret_cast<double*>(x), use_initial_guess)); break;
      case STRUMPACK_FLOATCOMPLEX_64:  return static_cast<STRUMPACK_RETURN_CODE>(CASTC64(S.solver)->solve(reinterpret_cast<std::complex<float>*>(b), reinterpret_cast<std::complex<float>*>(x), use_initial_guess)); break;
      case STRUMPACK_DOUBLECOMPLEX_64: return static_cast<STRUMPACK_RETURN_CODE>(CASTZ64(S.solver)->solve(reinterpret_cast<std::complex<double>*>(b), reinterpret_cast<std::complex<double>*>(x), use_initial_guess)); break;
      }
    } break;
    case STRUMPACK_MPI: {
      switch (S.precision) {
      case STRUMPACK_FLOAT:            return static_cast<STRUMPACK_RETURN_CODE>(CASTSMPI(S.solver)->solve(reinterpret_cast<float*>(b), reinterpret_cast<float*>(x), use_initial_guess)); break;
      case STRUMPACK_DOUBLE:           return static_cast<STRUMPACK_RETURN_CODE>(CASTDMPI(S.solver)->solve(reinterpret_cast<double*>(b), reinterpret_cast<double*>(x), use_initial_guess)); break;
      case STRUMPACK_FLOATCOMPLEX:     return static_cast<STRUMPACK_RETURN_CODE>(CASTCMPI(S.solver)->solve(reinterpret_cast<std::complex<float>*>(b), reinterpret_cast<std::complex<float>*>(x), use_initial_guess)); break;
      case STRUMPACK_DOUBLECOMPLEX:    return static_cast<STRUMPACK_RETURN_CODE>(CASTZMPI(S.solver)->solve(reinterpret_cast<std::complex<double>*>(b), reinterpret_cast<std::complex<double>*>(x), use_initial_guess)); break;
      case STRUMPACK_FLOAT_64:         return static_cast<STRUMPACK_RETURN_CODE>(CASTS64MPI(S.solver)->solve(reinterpret_cast<float*>(b), reinterpret_cast<float*>(x), use_initial_guess)); break;
      case STRUMPACK_DOUBLE_64:        return static_cast<STRUMPACK_RETURN_CODE>(CASTD64MPI(S.solver)->solve(reinterpret_cast<double*>(b), reinterpret_cast<double*>(x), use_initial_guess)); break;
      case STRUMPACK_FLOATCOMPLEX_64:  return static_cast<STRUMPACK_RETURN_CODE>(CASTC64MPI(S.solver)->solve(reinterpret_cast<std::complex<float>*>(b), reinterpret_cast<std::complex<float>*>(x), use_initial_guess)); break;
      case STRUMPACK_DOUBLECOMPLEX_64: return static_cast<STRUMPACK_RETURN_CODE>(CASTZ64MPI(S.solver)->solve(reinterpret_cast<std::complex<double>*>(b), reinterpret_cast<std::complex<double>*>(x), use_initial_guess)); break;
      }
    } break;
    case STRUMPACK_MPI_DIST: {
      switch (S.precision) {
      case STRUMPACK_FLOAT:            return static_cast<STRUMPACK_RETURN_CODE>(CASTSMPIDIST(S.solver)->solve(reinterpret_cast<float*>(b), reinterpret_cast<float*>(x), use_initial_guess)); break;
      case STRUMPACK_DOUBLE:           return static_cast<STRUMPACK_RETURN_CODE>(CASTDMPIDIST(S.solver)->solve(reinterpret_cast<double*>(b), reinterpret_cast<double*>(x), use_initial_guess)); break;
      case STRUMPACK_FLOATCOMPLEX:     return static_cast<STRUMPACK_RETURN_CODE>(CASTCMPIDIST(S.solver)->solve(reinterpret_cast<std::complex<float>*>(b), reinterpret_cast<std::complex<float>*>(x), use_initial_guess)); break;
      case STRUMPACK_DOUBLECOMPLEX:    return static_cast<STRUMPACK_RETURN_CODE>(CASTZMPIDIST(S.solver)->solve(reinterpret_cast<std::complex<double>*>(b), reinterpret_cast<std::complex<double>*>(x), use_initial_guess)); break;
      case STRUMPACK_FLOAT_64:         return static_cast<STRUMPACK_RETURN_CODE>(CASTS64MPIDIST(S.solver)->solve(reinterpret_cast<float*>(b), reinterpret_cast<float*>(x), use_initial_guess)); break;
      case STRUMPACK_DOUBLE_64:        return static_cast<STRUMPACK_RETURN_CODE>(CASTD64MPIDIST(S.solver)->solve(reinterpret_cast<double*>(b), reinterpret_cast<double*>(x), use_initial_guess)); break;
      case STRUMPACK_FLOATCOMPLEX_64:  return static_cast<STRUMPACK_RETURN_CODE>(CASTC64MPIDIST(S.solver)->solve(reinterpret_cast<std::complex<float>*>(b), reinterpret_cast<std::complex<float>*>(x), use_initial_guess)); break;
      case STRUMPACK_DOUBLECOMPLEX_64: return static_cast<STRUMPACK_RETURN_CODE>(CASTZ64MPIDIST(S.solver)->solve(reinterpret_cast<std::complex<double>*>(b), reinterpret_cast<std::complex<double>*>(x), use_initial_guess)); break;
      }
    } break;
    }
    return STRUMPACK_SUCCESS;
  }

  STRUMPACK_RETURN_CODE STRUMPACK_reorder(STRUMPACK_SparseSolver S)
  { ReturnCode c; switch_precision_return(reorder(), c);
    return static_cast<STRUMPACK_RETURN_CODE>(c); }
  STRUMPACK_RETURN_CODE STRUMPACK_reorder_regular(STRUMPACK_SparseSolver S, int nx, int ny, int nz)
  { ReturnCode c; switch_precision_return(reorder(nx, ny, nz), c);
    return static_cast<STRUMPACK_RETURN_CODE>(c); }
  STRUMPACK_RETURN_CODE STRUMPACK_factor(STRUMPACK_SparseSolver S)
  { ReturnCode c; switch_precision_return(factor(), c);
    return static_cast<STRUMPACK_RETURN_CODE>(c); }

  void STRUMPACK_set_from_options(STRUMPACK_SparseSolver S)
  { switch_precision(set_from_options()); }
  void STRUMPACK_set_maxit(STRUMPACK_SparseSolver S, int maxit)
  { switch_precision(options().set_maxit(maxit)); }
  void STRUMPACK_set_gmres_restart(STRUMPACK_SparseSolver S, int m)
  { switch_precision(options().set_gmres_restart(m)); }
  void STRUMPACK_set_rel_tol(STRUMPACK_SparseSolver S, double tol)
  { switch_precision(options().set_rel_tol(tol)); }
  void STRUMPACK_set_abs_tol(STRUMPACK_SparseSolver S, double tol)
  { switch_precision(options().set_abs_tol(tol)); }
  void STRUMPACK_set_nd_param(STRUMPACK_SparseSolver S, int nd_param)
  { switch_precision(options().set_nd_param(nd_param)); }
  void STRUMPACK_set_reordering_method(STRUMPACK_SparseSolver S, STRUMPACK_REORDERING_STRATEGY m)
  { switch_precision(options().set_reordering_method(static_cast<ReorderingStrategy>(m))); }
  void STRUMPACK_set_GramSchmidt_type(STRUMPACK_SparseSolver S, STRUMPACK_GRAM_SCHMIDT_TYPE t)
  { switch_precision(options().set_GramSchmidt_type(static_cast<GramSchmidtType>(t))); }
  void STRUMPACK_set_mc64job(STRUMPACK_SparseSolver S, int job)
  { switch_precision(options().set_mc64job(job)); }
  void STRUMPACK_set_Krylov_solver(STRUMPACK_SparseSolver S, STRUMPACK_KRYLOV_SOLVER solver_type)
  { switch_precision(options().set_Krylov_solver(static_cast<KrylovSolver>(solver_type))); }
  void STRUMPACK_enable_HSS(STRUMPACK_SparseSolver S)
  { switch_precision(options().enable_HSS()); }
  void STRUMPACK_disable_HSS(STRUMPACK_SparseSolver S)
  { switch_precision(options().disable_HSS()); }
  void STRUMPACK_set_HSS_min_front_size(STRUMPACK_SparseSolver S, int size)
  { switch_precision(options().set_HSS_min_front_size(size)); }
  void STRUMPACK_set_HSS_min_sep_size(STRUMPACK_SparseSolver S, int size)
  { switch_precision(options().set_HSS_min_sep_size(size)); }
  void STRUMPACK_set_hss_rel_tol(STRUMPACK_SparseSolver S, double rctol)
  { switch_precision(options().HSS_options().set_rel_tol(rctol)); }
  void STRUMPACK_set_hss_abs_tol(STRUMPACK_SparseSolver S, double actol)
  { switch_precision(options().HSS_options().set_abs_tol(actol)); }
  void STRUMPACK_set_verbose(STRUMPACK_SparseSolver S, int v)
  { switch_precision(options().set_verbose(static_cast<bool>(v))); }
  int STRUMPACK_maxit(STRUMPACK_SparseSolver S)
  { int maxit; switch_precision_return(options().maxit(), maxit); return maxit; }
  int STRUMPACK_gmres_restart(STRUMPACK_SparseSolver S)
  { int restart; switch_precision_return(options().gmres_restart(), restart); return restart; }
  double STRUMPACK_rel_tol(STRUMPACK_SparseSolver S)
  { double rtol; switch_precision_return(options().rel_tol(), rtol); return rtol; }
  double STRUMPACK_abs_tol(STRUMPACK_SparseSolver S)
  { double atol; switch_precision_return(options().abs_tol(), atol); return atol; }
  int STRUMPACK_nd_param(STRUMPACK_SparseSolver S)
  { int nd_param; switch_precision_return(options().nd_param(), nd_param); return nd_param; }
  STRUMPACK_REORDERING_STRATEGY STRUMPACK_reordering_method(STRUMPACK_SparseSolver S)
  { ReorderingStrategy r; switch_precision_return(options().reordering_method(), r);
    return static_cast<STRUMPACK_REORDERING_STRATEGY>(r); }
  STRUMPACK_GRAM_SCHMIDT_TYPE STRUMPACK_GramSchmidt_type(STRUMPACK_SparseSolver S)
  { GramSchmidtType gs; switch_precision_return(options().GramSchmidt_type(), gs);
    return static_cast<STRUMPACK_GRAM_SCHMIDT_TYPE>(gs); }
  int STRUMPACK_max_rank(STRUMPACK_SparseSolver S)
  { int rank; switch_precision_return(options().HSS_options().max_rank(), rank); return rank; }

  int STRUMPACK_maximum_rank(STRUMPACK_SparseSolver S)
  { int rank; switch_precision_return(maximum_rank(), rank); return rank; }
  long long STRUMPACK_factor_nonzeros(STRUMPACK_SparseSolver S)
  { long long nz; switch_precision_return(factor_nonzeros(), nz); return nz; }
  long long STRUMPACK_factor_memory(STRUMPACK_SparseSolver S)
  { long long mem; switch_precision_return(factor_memory(), mem); return mem; }
  int STRUMPACK_its(STRUMPACK_SparseSolver S)
  { int its; switch_precision_return(Krylov_iterations(), its); return its; }
  int STRUMPACK_mc64job(STRUMPACK_SparseSolver S)
  { int job; switch_precision_return(options().mc64job(), job); return job;}
  double STRUMPACK_hss_rel_tol(STRUMPACK_SparseSolver S)
  { double rctol; switch_precision_return(options().HSS_options().rel_tol(), rctol); return rctol; }
  double STRUMPACK_hss_abs_tol(STRUMPACK_SparseSolver S)
  { double actol; switch_precision_return(options().HSS_options().abs_tol(), actol); return actol; }
  STRUMPACK_KRYLOV_SOLVER STRUMPACK_Krylov_solver(STRUMPACK_SparseSolver S)
  { KrylovSolver s; switch_precision_return(options().Krylov_solver(), s);
    return static_cast<STRUMPACK_KRYLOV_SOLVER>(s); }
  int STRUMPACK_use_HSS(STRUMPACK_SparseSolver S)
  { bool u; switch_precision_return(options().use_HSS(), u); return u; }
  int STRUMPACK_HSS_min_front_size(STRUMPACK_SparseSolver S)
  { int size; switch_precision_return(options().HSS_min_front_size(), size); return size; }
  int STRUMPACK_HSS_min_sep_size(STRUMPACK_SparseSolver S)
  { int size; switch_precision_return(options().HSS_min_sep_size(), size); return size; }

}
