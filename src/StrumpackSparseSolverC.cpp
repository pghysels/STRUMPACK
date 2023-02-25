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
#include "StrumpackSparseSolver.h"
#include "StrumpackSparseSolver.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "StrumpackSparseSolverMPIDist.hpp"
#endif
#include "sparse/CSRMatrix.hpp"

using namespace strumpack;

#define CASTS(x) (static_cast<StrumpackSparseSolver<float,int>*>(x))
#define CASTD(x) (static_cast<StrumpackSparseSolver<double,int>*>(x))
#define CASTC(x) (static_cast<StrumpackSparseSolver<std::complex<float>,int>*>(x))
#define CASTZ(x) (static_cast<StrumpackSparseSolver<std::complex<double>,int>*>(x))
#define CASTS64(x) (static_cast<StrumpackSparseSolver<float,int64_t>*>(x))
#define CASTD64(x) (static_cast<StrumpackSparseSolver<double,int64_t>*>(x))
#define CASTC64(x) (static_cast<StrumpackSparseSolver<std::complex<float>,int64_t>*>(x))
#define CASTZ64(x) (static_cast<StrumpackSparseSolver<std::complex<double>,int64_t>*>(x))

#if defined(STRUMPACK_USE_MPI)
#define CASTSMPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<float,int>*>(x))
#define CASTDMPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<double,int>*>(x))
#define CASTCMPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<std::complex<float>,int>*>(x))
#define CASTZMPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<std::complex<double>,int>*>(x))
#define CASTS64MPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<float,int64_t>*>(x))
#define CASTD64MPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<double,int64_t>*>(x))
#define CASTC64MPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<std::complex<float>,int64_t>*>(x))
#define CASTZ64MPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<std::complex<double>,int64_t>*>(x))
#endif

#define switch_precision(m)                                             \
  switch (S.precision) {                                                \
  case STRUMPACK_FLOAT:            CASTS(S.solver)->m;   break;         \
  case STRUMPACK_DOUBLE:           CASTD(S.solver)->m;   break;         \
  case STRUMPACK_FLOATCOMPLEX:     CASTC(S.solver)->m;   break;         \
  case STRUMPACK_DOUBLECOMPLEX:    CASTZ(S.solver)->m;   break;         \
  case STRUMPACK_FLOAT_64:         CASTS64(S.solver)->m; break;         \
  case STRUMPACK_DOUBLE_64:        CASTD64(S.solver)->m; break;         \
  case STRUMPACK_FLOATCOMPLEX_64:  CASTC64(S.solver)->m; break;         \
  case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64(S.solver)->m; break;         \
  }                                                                     \

#define switch_precision_return_as(m,t)                                 \
  switch (S.precision) {                                                \
  case STRUMPACK_FLOAT:            return static_cast<t>(CASTS(S.solver)->m); \
  case STRUMPACK_DOUBLE:           return static_cast<t>(CASTD(S.solver)->m); \
  case STRUMPACK_FLOATCOMPLEX:     return static_cast<t>(CASTC(S.solver)->m); \
  case STRUMPACK_DOUBLECOMPLEX:    return static_cast<t>(CASTZ(S.solver)->m); \
  case STRUMPACK_FLOAT_64:         return static_cast<t>(CASTS64(S.solver)->m); \
  case STRUMPACK_DOUBLE_64:        return static_cast<t>(CASTD64(S.solver)->m); \
  case STRUMPACK_FLOATCOMPLEX_64:  return static_cast<t>(CASTC64(S.solver)->m); \
  case STRUMPACK_DOUBLECOMPLEX_64: return static_cast<t>(CASTZ64(S.solver)->m); \
  default: return t{};                                                  \
  }                                                                     \

#define REI(x) reinterpret_cast<int*>(x)
#define CREI(x) reinterpret_cast<const int*>(x)
#define RE64(x) reinterpret_cast<int64_t*>(x)
#define CRE64(x) reinterpret_cast<const int64_t*>(x)
#define RES(x) reinterpret_cast<float*>(x)
#define CRES(x) reinterpret_cast<const float*>(x)
#define RED(x) reinterpret_cast<double*>(x)
#define CRED(x) reinterpret_cast<const double*>(x)
#define REC(x) reinterpret_cast<std::complex<float>*>(x)
#define CREC(x) reinterpret_cast<const std::complex<float>*>(x)
#define REZ(x) reinterpret_cast<std::complex<double>*>(x)
#define CREZ(x) reinterpret_cast<const std::complex<double>*>(x)

extern "C" {

  void STRUMPACK_init_mt(STRUMPACK_SparseSolver* S,
                         STRUMPACK_PRECISION precision,
                         STRUMPACK_INTERFACE interface,
                         int argc, char* argv[], int verbose) {
    S->precision = precision;
    S->interface = interface;
    bool v = static_cast<bool>(verbose);
    switch (interface) {
    case STRUMPACK_MT: {
      switch (precision) {
      case STRUMPACK_FLOAT:            S->solver = static_cast<void*>(new StrumpackSparseSolver<float,int>(argc, argv, v));                    break;
      case STRUMPACK_DOUBLE:           S->solver = static_cast<void*>(new StrumpackSparseSolver<double,int>(argc, argv, v));                   break;
      case STRUMPACK_FLOATCOMPLEX:     S->solver = static_cast<void*>(new StrumpackSparseSolver<std::complex<float>,int>(argc, argv, v));      break;
      case STRUMPACK_DOUBLECOMPLEX:    S->solver = static_cast<void*>(new StrumpackSparseSolver<std::complex<double>,int>(argc, argv, v));     break;
      case STRUMPACK_FLOAT_64:         S->solver = static_cast<void*>(new StrumpackSparseSolver<float,int64_t>(argc, argv, v));                break;
      case STRUMPACK_DOUBLE_64:        S->solver = static_cast<void*>(new StrumpackSparseSolver<double,int64_t>(argc, argv, v));               break;
      case STRUMPACK_FLOATCOMPLEX_64:  S->solver = static_cast<void*>(new StrumpackSparseSolver<std::complex<float>,int64_t>(argc, argv, v));  break;
      case STRUMPACK_DOUBLECOMPLEX_64: S->solver = static_cast<void*>(new StrumpackSparseSolver<std::complex<double>,int64_t>(argc, argv, v)); break;
      default: std::cerr << "ERROR: wrong precision!" << std::endl;
      }
    } break;
    default: std::cerr << "ERROR: wrong interface!" << std::endl;
    }
  }


#if defined(STRUMPACK_USE_MPI)
  void STRUMPACK_init(STRUMPACK_SparseSolver* S, MPI_Comm comm,
                      STRUMPACK_PRECISION precision,
                      STRUMPACK_INTERFACE interface,
                      int argc, char* argv[], int verbose) {
    S->precision = precision;
    S->interface = interface;
    bool v = static_cast<bool>(verbose);
    switch (interface) {
    case STRUMPACK_MT: {
      switch (precision) {
      case STRUMPACK_FLOAT:            S->solver = static_cast<void*>(new StrumpackSparseSolver<float,int>(argc, argv, v));                    break;
      case STRUMPACK_DOUBLE:           S->solver = static_cast<void*>(new StrumpackSparseSolver<double,int>(argc, argv, v));                   break;
      case STRUMPACK_FLOATCOMPLEX:     S->solver = static_cast<void*>(new StrumpackSparseSolver<std::complex<float>,int>(argc, argv, v));      break;
      case STRUMPACK_DOUBLECOMPLEX:    S->solver = static_cast<void*>(new StrumpackSparseSolver<std::complex<double>,int>(argc, argv, v));     break;
      case STRUMPACK_FLOAT_64:         S->solver = static_cast<void*>(new StrumpackSparseSolver<float,int64_t>(argc, argv, v));                break;
      case STRUMPACK_DOUBLE_64:        S->solver = static_cast<void*>(new StrumpackSparseSolver<double,int64_t>(argc, argv, v));               break;
      case STRUMPACK_FLOATCOMPLEX_64:  S->solver = static_cast<void*>(new StrumpackSparseSolver<std::complex<float>,int64_t>(argc, argv, v));  break;
      case STRUMPACK_DOUBLECOMPLEX_64: S->solver = static_cast<void*>(new StrumpackSparseSolver<std::complex<double>,int64_t>(argc, argv, v)); break;
      default: std::cerr << "ERROR: wrong precision!" << std::endl;
      }
    } break;
    case STRUMPACK_MPI_DIST: {
      switch (precision) {
      case STRUMPACK_FLOAT:            S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<float,int>(comm, argc, argv, v));                    break;
      case STRUMPACK_DOUBLE:           S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<double,int>(comm, argc, argv, v));                   break;
      case STRUMPACK_FLOATCOMPLEX:     S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<std::complex<float>,int>(comm, argc, argv, v));      break;
      case STRUMPACK_DOUBLECOMPLEX:    S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<std::complex<double>,int>(comm, argc, argv, v));     break;
      case STRUMPACK_FLOAT_64:         S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<float,int64_t>(comm, argc, argv, v));                break;
      case STRUMPACK_DOUBLE_64:        S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<double,int64_t>(comm, argc, argv, v));               break;
      case STRUMPACK_FLOATCOMPLEX_64:  S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<std::complex<float>,int64_t>(comm, argc, argv, v));  break;
      case STRUMPACK_DOUBLECOMPLEX_64: S->solver = static_cast<void*>(new StrumpackSparseSolverMPIDist<std::complex<double>,int64_t>(comm, argc, argv, v)); break;
      default: std::cerr << "ERROR: wrong precision!" << std::endl;
      }
    } break;
    default: std::cerr << "ERROR: wrong interface!" << std::endl;
    }
  }
#endif

  void STRUMPACK_destroy(STRUMPACK_SparseSolver* S) {
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
    S->solver = NULL;
  }

  void STRUMPACK_set_csr_matrix(STRUMPACK_SparseSolver S, const void* N,
                                const void* row_ptr, const void* col_ind,
                                const void* values, int symm) {
    switch (S.precision) {
    case STRUMPACK_FLOAT:            CASTS(S.solver)->set_csr_matrix(*CREI(N), CREI(row_ptr), CREI(col_ind), CRES(values), symm);      break;
    case STRUMPACK_DOUBLE:           CASTD(S.solver)->set_csr_matrix(*CREI(N), CREI(row_ptr), CREI(col_ind), CRED(values), symm);      break;
    case STRUMPACK_FLOATCOMPLEX:     CASTC(S.solver)->set_csr_matrix(*CREI(N), CREI(row_ptr), CREI(col_ind), CREC(values), symm);      break;
    case STRUMPACK_DOUBLECOMPLEX:    CASTZ(S.solver)->set_csr_matrix(*CREI(N), CREI(row_ptr), CREI(col_ind), CREZ(values), symm);      break;
    case STRUMPACK_FLOAT_64:         CASTS64(S.solver)->set_csr_matrix(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CRES(values), symm); break;
    case STRUMPACK_DOUBLE_64:        CASTD64(S.solver)->set_csr_matrix(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CRED(values), symm); break;
    case STRUMPACK_FLOATCOMPLEX_64:  CASTC64(S.solver)->set_csr_matrix(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CREC(values), symm); break;
    case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64(S.solver)->set_csr_matrix(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CREZ(values), symm); break;
    }
  }

  void STRUMPACK_update_csr_matrix_values
  (STRUMPACK_SparseSolver S, const void* N, const void* row_ptr,
   const void* col_ind, const void* values, int symm) {
    switch (S.precision) {
    case STRUMPACK_FLOAT:            CASTS(S.solver)->update_matrix_values(*CREI(N), CREI(row_ptr), CREI(col_ind), CRES(values), symm);      break;
    case STRUMPACK_DOUBLE:           CASTD(S.solver)->update_matrix_values(*CREI(N), CREI(row_ptr), CREI(col_ind), CRED(values), symm);      break;
    case STRUMPACK_FLOATCOMPLEX:     CASTC(S.solver)->update_matrix_values(*CREI(N), CREI(row_ptr), CREI(col_ind), CREC(values), symm);      break;
    case STRUMPACK_DOUBLECOMPLEX:    CASTZ(S.solver)->update_matrix_values(*CREI(N), CREI(row_ptr), CREI(col_ind), CREZ(values), symm);      break;
    case STRUMPACK_FLOAT_64:         CASTS64(S.solver)->update_matrix_values(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CRES(values), symm); break;
    case STRUMPACK_DOUBLE_64:        CASTD64(S.solver)->update_matrix_values(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CRED(values), symm); break;
    case STRUMPACK_FLOATCOMPLEX_64:  CASTC64(S.solver)->update_matrix_values(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CREC(values), symm); break;
    case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64(S.solver)->update_matrix_values(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CREZ(values), symm); break;
    }
  }

#if defined(STRUMPACK_USE_MPI)
  void STRUMPACK_set_distributed_csr_matrix
  (STRUMPACK_SparseSolver S, const void* N, const void* row_ptr,
   const void* col_ind, const void* values, const void* dist, int symm) {
    if (S.interface != STRUMPACK_MPI_DIST) {
      std::cerr << "ERROR: interface != STRUMPACK_MPI_DIST" << std::endl;
      return;
    }
    switch (S.precision) {
    case STRUMPACK_FLOAT:            CASTSMPIDIST(S.solver)->set_distributed_csr_matrix(*CREI(N), CREI(row_ptr), CREI(col_ind), CRES(values), CREI(dist), symm);       break;
    case STRUMPACK_DOUBLE:           CASTDMPIDIST(S.solver)->set_distributed_csr_matrix(*CREI(N), CREI(row_ptr), CREI(col_ind), CRED(values), CREI(dist), symm);       break;
    case STRUMPACK_FLOATCOMPLEX:     CASTCMPIDIST(S.solver)->set_distributed_csr_matrix(*CREI(N), CREI(row_ptr), CREI(col_ind), CREC(values), CREI(dist), symm);       break;
    case STRUMPACK_DOUBLECOMPLEX:    CASTZMPIDIST(S.solver)->set_distributed_csr_matrix(*CREI(N), CREI(row_ptr), CREI(col_ind), CREZ(values), CREI(dist), symm);       break;
    case STRUMPACK_FLOAT_64:         CASTS64MPIDIST(S.solver)->set_distributed_csr_matrix(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CRES(values), CRE64(dist), symm); break;
    case STRUMPACK_DOUBLE_64:        CASTD64MPIDIST(S.solver)->set_distributed_csr_matrix(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CRED(values), CRE64(dist), symm); break;
    case STRUMPACK_FLOATCOMPLEX_64:  CASTC64MPIDIST(S.solver)->set_distributed_csr_matrix(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CREC(values), CRE64(dist), symm); break;
    case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64MPIDIST(S.solver)->set_distributed_csr_matrix(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CREZ(values), CRE64(dist), symm); break;
    }
  }

  void STRUMPACK_update_distributed_csr_matrix_values
  (STRUMPACK_SparseSolver S, const void* N, const void* row_ptr,
   const void* col_ind, const void* values, const void* dist, int symm) {
    if (S.interface != STRUMPACK_MPI_DIST) {
      std::cerr << "ERROR: interface != STRUMPACK_MPI_DIST" << std::endl;
      return;
    }
    switch (S.precision) {
    case STRUMPACK_FLOAT:            CASTSMPIDIST(S.solver)->update_matrix_values(*CREI(N), CREI(row_ptr), CREI(col_ind), CRES(values), CREI(dist), symm);       break;
    case STRUMPACK_DOUBLE:           CASTDMPIDIST(S.solver)->update_matrix_values(*CREI(N), CREI(row_ptr), CREI(col_ind), CRED(values), CREI(dist), symm);       break;
    case STRUMPACK_FLOATCOMPLEX:     CASTCMPIDIST(S.solver)->update_matrix_values(*CREI(N), CREI(row_ptr), CREI(col_ind), CREC(values), CREI(dist), symm);       break;
    case STRUMPACK_DOUBLECOMPLEX:    CASTZMPIDIST(S.solver)->update_matrix_values(*CREI(N), CREI(row_ptr), CREI(col_ind), CREZ(values), CREI(dist), symm);       break;
    case STRUMPACK_FLOAT_64:         CASTS64MPIDIST(S.solver)->update_matrix_values(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CRES(values), CRE64(dist), symm); break;
    case STRUMPACK_DOUBLE_64:        CASTD64MPIDIST(S.solver)->update_matrix_values(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CRED(values), CRE64(dist), symm); break;
    case STRUMPACK_FLOATCOMPLEX_64:  CASTC64MPIDIST(S.solver)->update_matrix_values(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CREC(values), CRE64(dist), symm); break;
    case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64MPIDIST(S.solver)->update_matrix_values(*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CREZ(values), CRE64(dist), symm); break;
    }
  }


  void STRUMPACK_set_MPIAIJ_matrix
  (STRUMPACK_SparseSolver S, const void* n,
   const void* d_ptr, const void* d_ind, const void* d_val,
   const void* o_ptr, const void* o_ind, const void* o_val,
   const void* garray) {
    if (S.interface != STRUMPACK_MPI_DIST) {
      std::cerr << "ERROR: interface != STRUMPACK_MPI_DIST" << std::endl;
      return;
    }
    switch (S.precision) {
    case STRUMPACK_FLOAT:            CASTSMPIDIST(S.solver)->set_MPIAIJ_matrix(*CREI(n), CREI(d_ptr), CREI(d_ind), CRES(d_val), CREI(o_ptr), CREI(o_ind), CRES(o_val), CREI(garray));         break;
    case STRUMPACK_DOUBLE:           CASTDMPIDIST(S.solver)->set_MPIAIJ_matrix(*CREI(n), CREI(d_ptr), CREI(d_ind), CRED(d_val), CREI(o_ptr), CREI(o_ind), CRED(o_val), CREI(garray));         break;
    case STRUMPACK_FLOATCOMPLEX:     CASTCMPIDIST(S.solver)->set_MPIAIJ_matrix(*CREI(n), CREI(d_ptr), CREI(d_ind), CREC(d_val), CREI(o_ptr), CREI(o_ind), CREC(o_val), CREI(garray));         break;
    case STRUMPACK_DOUBLECOMPLEX:    CASTZMPIDIST(S.solver)->set_MPIAIJ_matrix(*CREI(n), CREI(d_ptr), CREI(d_ind), CREZ(d_val), CREI(o_ptr), CREI(o_ind), CREZ(o_val), CREI(garray));         break;
    case STRUMPACK_FLOAT_64:         CASTS64MPIDIST(S.solver)->set_MPIAIJ_matrix(*CRE64(n), CRE64(d_ptr), CRE64(d_ind), CRES(d_val), CRE64(o_ptr), CRE64(o_ind), CRES(o_val), CRE64(garray)); break;
    case STRUMPACK_DOUBLE_64:        CASTD64MPIDIST(S.solver)->set_MPIAIJ_matrix(*CRE64(n), CRE64(d_ptr), CRE64(d_ind), CRED(d_val), CRE64(o_ptr), CRE64(o_ind), CRED(o_val), CRE64(garray)); break;
    case STRUMPACK_FLOATCOMPLEX_64:  CASTC64MPIDIST(S.solver)->set_MPIAIJ_matrix(*CRE64(n), CRE64(d_ptr), CRE64(d_ind), CREC(d_val), CRE64(o_ptr), CRE64(o_ind), CREC(o_val), CRE64(garray)); break;
    case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64MPIDIST(S.solver)->set_MPIAIJ_matrix(*CRE64(n), CRE64(d_ptr), CRE64(d_ind), CREZ(d_val), CRE64(o_ptr), CRE64(o_ind), CREZ(o_val), CRE64(garray)); break;
    }
  }

  void STRUMPACK_update_MPIAIJ_matrix_values
  (STRUMPACK_SparseSolver S, const void* n, const void* d_ptr,
   const void* d_ind, const void* d_val, const void* o_ptr, const void* o_ind,
   const void* o_val, const void* garray) {
    if (S.interface != STRUMPACK_MPI_DIST) {
      std::cerr << "ERROR: interface != STRUMPACK_MPI_DIST" << std::endl;
      return;
    }
    switch (S.precision) {
    case STRUMPACK_FLOAT:            CASTSMPIDIST(S.solver)->update_MPIAIJ_matrix_values(*CREI(n), CREI(d_ptr), CREI(d_ind), CRES(d_val), CREI(o_ptr), CREI(o_ind), CRES(o_val), CREI(garray));         break;
    case STRUMPACK_DOUBLE:           CASTDMPIDIST(S.solver)->update_MPIAIJ_matrix_values(*CREI(n), CREI(d_ptr), CREI(d_ind), CRED(d_val), CREI(o_ptr), CREI(o_ind), CRED(o_val), CREI(garray));         break;
    case STRUMPACK_FLOATCOMPLEX:     CASTCMPIDIST(S.solver)->update_MPIAIJ_matrix_values(*CREI(n), CREI(d_ptr), CREI(d_ind), CREC(d_val), CREI(o_ptr), CREI(o_ind), CREC(o_val), CREI(garray));         break;
    case STRUMPACK_DOUBLECOMPLEX:    CASTZMPIDIST(S.solver)->update_MPIAIJ_matrix_values(*CREI(n), CREI(d_ptr), CREI(d_ind), CREZ(d_val), CREI(o_ptr), CREI(o_ind), CREZ(o_val), CREI(garray));         break;
    case STRUMPACK_FLOAT_64:         CASTS64MPIDIST(S.solver)->update_MPIAIJ_matrix_values(*CRE64(n), CRE64(d_ptr), CRE64(d_ind), CRES(d_val), CRE64(o_ptr), CRE64(o_ind), CRES(o_val), CRE64(garray)); break;
    case STRUMPACK_DOUBLE_64:        CASTD64MPIDIST(S.solver)->update_MPIAIJ_matrix_values(*CRE64(n), CRE64(d_ptr), CRE64(d_ind), CRED(d_val), CRE64(o_ptr), CRE64(o_ind), CRED(o_val), CRE64(garray)); break;
    case STRUMPACK_FLOATCOMPLEX_64:  CASTC64MPIDIST(S.solver)->update_MPIAIJ_matrix_values(*CRE64(n), CRE64(d_ptr), CRE64(d_ind), CREC(d_val), CRE64(o_ptr), CRE64(o_ind), CREC(o_val), CRE64(garray)); break;
    case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64MPIDIST(S.solver)->update_MPIAIJ_matrix_values(*CRE64(n), CRE64(d_ptr), CRE64(d_ind), CREZ(d_val), CRE64(o_ptr), CRE64(o_ind), CREZ(o_val), CRE64(garray)); break;
    }
  }
#endif

  STRUMPACK_RETURN_CODE
  STRUMPACK_solve(STRUMPACK_SparseSolver S, const void* b, void* x,
                  int use_initial_guess) {
    switch (S.precision) {
    case STRUMPACK_FLOAT:            return static_cast<STRUMPACK_RETURN_CODE>(CASTS(S.solver)->solve(CRES(b), RES(x), use_initial_guess));
    case STRUMPACK_DOUBLE:           return static_cast<STRUMPACK_RETURN_CODE>(CASTD(S.solver)->solve(CRED(b), RED(x), use_initial_guess));
    case STRUMPACK_FLOATCOMPLEX:     return static_cast<STRUMPACK_RETURN_CODE>(CASTC(S.solver)->solve(CREC(b), REC(x), use_initial_guess));
    case STRUMPACK_DOUBLECOMPLEX:    return static_cast<STRUMPACK_RETURN_CODE>(CASTZ(S.solver)->solve(CREZ(b), REZ(x), use_initial_guess));
    case STRUMPACK_FLOAT_64:         return static_cast<STRUMPACK_RETURN_CODE>(CASTS64(S.solver)->solve(CRES(b), RES(x), use_initial_guess));
    case STRUMPACK_DOUBLE_64:        return static_cast<STRUMPACK_RETURN_CODE>(CASTD64(S.solver)->solve(CRED(b), RED(x), use_initial_guess));
    case STRUMPACK_FLOATCOMPLEX_64:  return static_cast<STRUMPACK_RETURN_CODE>(CASTC64(S.solver)->solve(CREC(b), REC(x), use_initial_guess));
    case STRUMPACK_DOUBLECOMPLEX_64: return static_cast<STRUMPACK_RETURN_CODE>(CASTZ64(S.solver)->solve(CREZ(b), REZ(x), use_initial_guess));
    }
    return STRUMPACK_SUCCESS;
  }

  STRUMPACK_RETURN_CODE
  STRUMPACK_matsolve(STRUMPACK_SparseSolver S, int nrhs,
                     const void* b, int ldb, void* x, int ldx,
                     int use_initial_guess) {
    switch (S.precision) {
    case STRUMPACK_FLOAT:            return static_cast<STRUMPACK_RETURN_CODE>(CASTS(S.solver)->solve(nrhs, CRES(b), ldb, RES(x), ldx, use_initial_guess));
    case STRUMPACK_DOUBLE:           return static_cast<STRUMPACK_RETURN_CODE>(CASTD(S.solver)->solve(nrhs, CRED(b), ldb, RED(x), ldx, use_initial_guess));
    case STRUMPACK_FLOATCOMPLEX:     return static_cast<STRUMPACK_RETURN_CODE>(CASTC(S.solver)->solve(nrhs, CREC(b), ldb, REC(x), ldx, use_initial_guess));
    case STRUMPACK_DOUBLECOMPLEX:    return static_cast<STRUMPACK_RETURN_CODE>(CASTZ(S.solver)->solve(nrhs, CREZ(b), ldb, REZ(x), ldx, use_initial_guess));
    case STRUMPACK_FLOAT_64:         return static_cast<STRUMPACK_RETURN_CODE>(CASTS64(S.solver)->solve(nrhs, CRES(b), ldb, RES(x), ldx, use_initial_guess));
    case STRUMPACK_DOUBLE_64:        return static_cast<STRUMPACK_RETURN_CODE>(CASTD64(S.solver)->solve(nrhs, CRED(b), ldb, RED(x), ldx, use_initial_guess));
    case STRUMPACK_FLOATCOMPLEX_64:  return static_cast<STRUMPACK_RETURN_CODE>(CASTC64(S.solver)->solve(nrhs, CREC(b), ldb, REC(x), ldx, use_initial_guess));
    case STRUMPACK_DOUBLECOMPLEX_64: return static_cast<STRUMPACK_RETURN_CODE>(CASTZ64(S.solver)->solve(nrhs, CREZ(b), ldb, REZ(x), ldx, use_initial_guess));
    }
    return STRUMPACK_SUCCESS;
  }

  void STRUMPACK_set_from_options(STRUMPACK_SparseSolver S) {
    switch_precision(set_from_options());
  }

  STRUMPACK_RETURN_CODE STRUMPACK_reorder(STRUMPACK_SparseSolver S) {
    switch_precision_return_as(reorder(), STRUMPACK_RETURN_CODE);
  }

  STRUMPACK_RETURN_CODE STRUMPACK_reorder_regular(STRUMPACK_SparseSolver S,
                                                  int nx, int ny, int nz,
                                                  int components, int width) {
    switch_precision_return_as(reorder(nx, ny, nz, components, width), STRUMPACK_RETURN_CODE);
  }

  STRUMPACK_RETURN_CODE STRUMPACK_factor(STRUMPACK_SparseSolver S) {
    switch_precision_return_as(factor(), STRUMPACK_RETURN_CODE);
  }

  STRUMPACK_RETURN_CODE STRUMPACK_inertia(STRUMPACK_SparseSolver S,
                                          int* neg, int* zero, int* pos) {
    auto ierr = strumpack::ReturnCode::SUCCESS;
    int64_t neg64, zero64, pos64;
    switch (S.precision) {
    case STRUMPACK_FLOAT:            ierr = CASTS(S.solver)->inertia(*neg, *zero, *pos); break;
    case STRUMPACK_DOUBLE:           ierr = CASTD(S.solver)->inertia(*neg, *zero, *pos); break;
    case STRUMPACK_FLOATCOMPLEX:     ierr = CASTC(S.solver)->inertia(*neg, *zero, *pos); break;
    case STRUMPACK_DOUBLECOMPLEX:    ierr = CASTZ(S.solver)->inertia(*neg, *zero, *pos); break;
    case STRUMPACK_FLOAT_64:         ierr = CASTS64(S.solver)->inertia(neg64, zero64, pos64); break;
    case STRUMPACK_DOUBLE_64:        ierr = CASTD64(S.solver)->inertia(neg64, zero64, pos64); break;
    case STRUMPACK_FLOATCOMPLEX_64:  ierr = CASTC64(S.solver)->inertia(neg64, zero64, pos64); break;
    case STRUMPACK_DOUBLECOMPLEX_64: ierr = CASTZ64(S.solver)->inertia(neg64, zero64, pos64); break;
    }
    switch (S.precision) {
    case STRUMPACK_FLOAT:
    case STRUMPACK_DOUBLE:
    case STRUMPACK_FLOATCOMPLEX:
    case STRUMPACK_DOUBLECOMPLEX:
      break;
    case STRUMPACK_FLOAT_64:
    case STRUMPACK_DOUBLE_64:
    case STRUMPACK_FLOATCOMPLEX_64:
    case STRUMPACK_DOUBLECOMPLEX_64:
      *neg = neg64; *zero = zero64; *pos = pos64;
      break;
    }
    return static_cast<STRUMPACK_RETURN_CODE>(ierr);
  }

  void STRUMPACK_move_to_gpu(STRUMPACK_SparseSolver S) {
    // switch_precision(move_to_gpu());
  }

  void STRUMPACK_remove_from_gpu(STRUMPACK_SparseSolver S) {
    // switch_precision(remove_from_gpu());
  }

  void STRUMPACK_delete_factors(STRUMPACK_SparseSolver S) {
    switch_precision(delete_factors());
  }


  /*************************************************************
   ** Set options **********************************************
   ************************************************************/
  void STRUMPACK_set_verbose(STRUMPACK_SparseSolver S, int v) { switch_precision(options().set_verbose(static_cast<bool>(v))); }
  void STRUMPACK_set_maxit(STRUMPACK_SparseSolver S, int maxit) { switch_precision(options().set_maxit(maxit)); }
  void STRUMPACK_set_gmres_restart(STRUMPACK_SparseSolver S, int m) { switch_precision(options().set_gmres_restart(m)); }
  void STRUMPACK_set_rel_tol(STRUMPACK_SparseSolver S, double tol) { switch_precision(options().set_rel_tol(tol)); }
  void STRUMPACK_set_abs_tol(STRUMPACK_SparseSolver S, double tol) { switch_precision(options().set_abs_tol(tol)); }
  void STRUMPACK_set_nd_param(STRUMPACK_SparseSolver S, int nd_param) { switch_precision(options().set_nd_param(nd_param)); }
  void STRUMPACK_set_reordering_method(STRUMPACK_SparseSolver S, STRUMPACK_REORDERING_STRATEGY m) { switch_precision(options().set_reordering_method(static_cast<ReorderingStrategy>(m))); }
  void STRUMPACK_enable_METIS_NodeNDP(STRUMPACK_SparseSolver S) { switch_precision(options().enable_METIS_NodeNDP()); }
  void STRUMPACK_disable_METIS_NodeNDP(STRUMPACK_SparseSolver S) { switch_precision(options().disable_METIS_NodeNDP()); }
  void STRUMPACK_set_nx(STRUMPACK_SparseSolver S, int nx) { switch_precision(options().set_nx(nx)); }
  void STRUMPACK_set_ny(STRUMPACK_SparseSolver S, int ny) { switch_precision(options().set_ny(ny)); }
  void STRUMPACK_set_nz(STRUMPACK_SparseSolver S, int nz) { switch_precision(options().set_nz(nz)); }
  void STRUMPACK_set_components(STRUMPACK_SparseSolver S, int nc) { switch_precision(options().set_components(nc)); }
  void STRUMPACK_set_separator_width(STRUMPACK_SparseSolver S, int w) { switch_precision(options().set_separator_width(w)); }
  void STRUMPACK_set_GramSchmidt_type(STRUMPACK_SparseSolver S, STRUMPACK_GRAM_SCHMIDT_TYPE t) { switch_precision(options().set_GramSchmidt_type(static_cast<GramSchmidtType>(t))); }
  void STRUMPACK_set_matching(STRUMPACK_SparseSolver S, STRUMPACK_MATCHING_JOB job) { switch_precision(options().set_matching(static_cast<MatchingJob>(job))); }
  void STRUMPACK_set_Krylov_solver(STRUMPACK_SparseSolver S, STRUMPACK_KRYLOV_SOLVER solver_type) { switch_precision(options().set_Krylov_solver(static_cast<KrylovSolver>(solver_type))); }
  void STRUMPACK_enable_gpu(STRUMPACK_SparseSolver S) { switch_precision(options().enable_gpu()); }
  void STRUMPACK_disable_gpu(STRUMPACK_SparseSolver S) { switch_precision(options().disable_gpu()); }
  void STRUMPACK_set_compression(STRUMPACK_SparseSolver S, STRUMPACK_COMPRESSION_TYPE t) { switch_precision(options().set_compression(static_cast<CompressionType>(t))); }
  void STRUMPACK_set_compression_min_front_size(STRUMPACK_SparseSolver S, int size) { switch_precision(options().set_compression_min_front_size(size)); }
  void STRUMPACK_set_compression_min_sep_size(STRUMPACK_SparseSolver S, int size) { switch_precision(options().set_compression_min_sep_size(size)); }
  void STRUMPACK_set_compression_leaf_size(STRUMPACK_SparseSolver S, int leaf_size) { switch_precision(options().set_compression_leaf_size(leaf_size)); }
  void STRUMPACK_set_compression_rel_tol(STRUMPACK_SparseSolver S, double rctol) { switch_precision(options().set_compression_rel_tol(rctol)); }
  void STRUMPACK_set_compression_abs_tol(STRUMPACK_SparseSolver S, double actol) { switch_precision(options().set_compression_abs_tol(actol)); }
  void STRUMPACK_set_compression_butterfly_levels(STRUMPACK_SparseSolver S, int l) { switch_precision(options().HODLR_options().set_butterfly_levels(l)); }
  void STRUMPACK_set_compression_lossy_precision(STRUMPACK_SparseSolver S, int p) { switch_precision(options().set_lossy_precision(p)); }


  /*************************************************************
   ** Get options **********************************************
   ************************************************************/
  int STRUMPACK_verbose(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().verbose(), int); }
  int STRUMPACK_maxit(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().maxit(), int); }
  int STRUMPACK_gmres_restart(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().gmres_restart(), int); }
  double STRUMPACK_rel_tol(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().rel_tol(), double); }
  double STRUMPACK_abs_tol(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().abs_tol(), double); }
  int STRUMPACK_nd_param(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().nd_param(), int); }
  STRUMPACK_REORDERING_STRATEGY STRUMPACK_reordering_method(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().reordering_method(), STRUMPACK_REORDERING_STRATEGY); }
  int STRUMPACK_use_METIS_NodeNDP(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().use_METIS_NodeNDP(), int); }
  STRUMPACK_MATCHING_JOB STRUMPACK_matching(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().matching(), STRUMPACK_MATCHING_JOB); }
  STRUMPACK_GRAM_SCHMIDT_TYPE STRUMPACK_GramSchmidt_type(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().GramSchmidt_type(), STRUMPACK_GRAM_SCHMIDT_TYPE); }
  STRUMPACK_KRYLOV_SOLVER STRUMPACK_Krylov_solver(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().Krylov_solver(), STRUMPACK_KRYLOV_SOLVER); }
  int STRUMPACK_use_gpu(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().use_gpu(), int); }
  STRUMPACK_COMPRESSION_TYPE STRUMPACK_compression(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().compression(), STRUMPACK_COMPRESSION_TYPE); }
  int STRUMPACK_compression_min_front_size(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().compression_min_front_size(), int); }
  int STRUMPACK_compression_min_sep_size(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().compression_min_sep_size(), int); }
  int STRUMPACK_compression_leaf_size(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().compression_leaf_size(), int); }
  double STRUMPACK_compression_rel_tol(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().compression_rel_tol(), double); }
  double STRUMPACK_compression_abs_tol(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().compression_abs_tol(), double); }
  int STRUMPACK_compression_butterfly_levels(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().HODLR_options().butterfly_levels(), int); }
  int STRUMPACK_compression_lossy_precision(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().lossy_precision(), int); }


  /*************************************************************
   ** Get solve statistics *************************************
   ************************************************************/
  int STRUMPACK_its(STRUMPACK_SparseSolver S) { switch_precision_return_as(Krylov_iterations(), int); }
  int STRUMPACK_rank(STRUMPACK_SparseSolver S) { switch_precision_return_as(maximum_rank(), int); }
  long long STRUMPACK_factor_nonzeros(STRUMPACK_SparseSolver S) { switch_precision_return_as(factor_nonzeros(), int64_t); }
  long long STRUMPACK_factor_memory(STRUMPACK_SparseSolver S) { switch_precision_return_as(factor_memory(), int64_t); }





  /*************************************************************
   ** Deprecated routines **************************************
   ************************************************************/
  void STRUMPACK_set_mc64job(STRUMPACK_SparseSolver S, int job) { switch_precision(options().set_matching(static_cast<MatchingJob>(job))); }
  int STRUMPACK_mc64job(STRUMPACK_SparseSolver S) { return STRUMPACK_matching(S); }

  void STRUMPACK_enable_HSS(STRUMPACK_SparseSolver S) { switch_precision(options().set_compression(CompressionType::HSS)); }
  void STRUMPACK_disable_HSS(STRUMPACK_SparseSolver S) { switch_precision(options().set_compression(CompressionType::NONE)); }
  void STRUMPACK_set_HSS_min_front_size(STRUMPACK_SparseSolver S, int size) { switch_precision(options().set_compression_min_front_size(size)); }
  void STRUMPACK_set_HSS_min_sep_size(STRUMPACK_SparseSolver S, int size) { switch_precision(options().set_compression_min_sep_size(size)); }
  void STRUMPACK_set_HSS_max_rank(STRUMPACK_SparseSolver S, int max_rank) { switch_precision(options().HSS_options().set_max_rank(max_rank)); }
  void STRUMPACK_set_HSS_leaf_size(STRUMPACK_SparseSolver S, int leaf_size) { switch_precision(options().HSS_options().set_leaf_size(leaf_size)); }
  void STRUMPACK_set_HSS_rel_tol(STRUMPACK_SparseSolver S, double rctol) { switch_precision(options().HSS_options().set_rel_tol(rctol)); }
  void STRUMPACK_set_HSS_abs_tol(STRUMPACK_SparseSolver S, double actol) { switch_precision(options().HSS_options().set_abs_tol(actol)); }

  int STRUMPACK_use_HSS(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().compression(), int); }
  int STRUMPACK_HSS_min_front_size(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().compression_min_front_size(), int); }
  int STRUMPACK_HSS_min_sep_size(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().compression_min_sep_size(), int); }
  int STRUMPACK_HSS_max_rank(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().HSS_options().max_rank(), int); }
  int STRUMPACK_HSS_leaf_size(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().HSS_options().leaf_size(), int); }
  double STRUMPACK_HSS_rel_tol(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().HSS_options().rel_tol(), double); }
  double STRUMPACK_HSS_abs_tol(STRUMPACK_SparseSolver S) { switch_precision_return_as(options().HSS_options().abs_tol(), double); }
}
