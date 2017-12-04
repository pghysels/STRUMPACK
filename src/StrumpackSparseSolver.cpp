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
#include "StrumpackSparseSolverMPI.hpp"
#include "StrumpackSparseSolverMPIDist.hpp"
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

#define CASTSMPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<float,int>*>(x))
#define CASTDMPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<double,int>*>(x))
#define CASTCMPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<std::complex<float>,int>*>(x))
#define CASTZMPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<std::complex<double>,int>*>(x))
#define CASTS64MPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<float,int64_t>*>(x))
#define CASTD64MPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<double,int64_t>*>(x))
#define CASTC64MPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<std::complex<float>,int64_t>*>(x))
#define CASTZ64MPIDIST(x) (static_cast<StrumpackSparseSolverMPIDist<std::complex<double>,int64_t>*>(x))

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

#define switch_precision_arg(m,a)                                            \
  switch (S.precision) {                                                \
  case STRUMPACK_FLOAT:            CASTS(S.solver)->m(a);   break;       \
  case STRUMPACK_DOUBLE:           CASTD(S.solver)->m(a);   break;       \
  case STRUMPACK_FLOATCOMPLEX:     CASTC(S.solver)->m(a);   break;       \
  case STRUMPACK_DOUBLECOMPLEX:    CASTZ(S.solver)->m(a);   break;       \
  case STRUMPACK_FLOAT_64:         CASTS64(S.solver)->m(a); break;       \
  case STRUMPACK_DOUBLE_64:        CASTD64(S.solver)->m(a); break;       \
  case STRUMPACK_FLOATCOMPLEX_64:  CASTC64(S.solver)->m(a); break;       \
  case STRUMPACK_DOUBLECOMPLEX_64: CASTZ64(S.solver)->m(a); break;       \
  }                                                                     \

#define switch_precision_return(m,r)                                    \
  switch (S.precision) {                                                \
  case STRUMPACK_FLOAT:            r = CASTS(S.solver)->m;   break;     \
  case STRUMPACK_DOUBLE:           r = CASTD(S.solver)->m;   break;     \
  case STRUMPACK_FLOATCOMPLEX:     r = CASTC(S.solver)->m;   break;     \
  case STRUMPACK_DOUBLECOMPLEX:    r = CASTZ(S.solver)->m;   break;     \
  case STRUMPACK_FLOAT_64:         r = CASTS64(S.solver)->m; break;     \
  case STRUMPACK_DOUBLE_64:        r = CASTD64(S.solver)->m; break;     \
  case STRUMPACK_FLOATCOMPLEX_64:  r = CASTC64(S.solver)->m; break;     \
  case STRUMPACK_DOUBLECOMPLEX_64: r = CASTZ64(S.solver)->m; break;     \
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
  void STRUMPACK_init
  (STRUMPACK_SparseSolver* S, MPI_Comm comm, STRUMPACK_PRECISION precision,
   STRUMPACK_INTERFACE interface, int argc, char* argv[], int verbose) {
    S->precision = precision;
    S->interface = interface;
    bool v = static_cast<bool>(verbose);
    switch (interface) {
    case STRUMPACK_MT: {
      switch (precision) {
      case STRUMPACK_FLOAT:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolver<float,int>(argc, argv, v));
        break;
      case STRUMPACK_DOUBLE:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolver<double,int>(argc, argv, v));
        break;
      case STRUMPACK_FLOATCOMPLEX:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolver<std::complex<float>,int>(argc, argv, v));
        break;
      case STRUMPACK_DOUBLECOMPLEX:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolver<std::complex<double>,int>
           (argc, argv, v));
        break;
      case STRUMPACK_FLOAT_64:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolver<float,int64_t>(argc, argv, v));
        break;
      case STRUMPACK_DOUBLE_64:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolver<double,int64_t>(argc, argv, v));
        break;
      case STRUMPACK_FLOATCOMPLEX_64:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolver<std::complex<float>,int64_t>
           (argc, argv, v));
        break;
      case STRUMPACK_DOUBLECOMPLEX_64:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolver<std::complex<double>,int64_t>
           (argc, argv, v));
        break;
      default: std::cerr << "ERROR: wrong precision!" << std::endl;
      }
    } break;
    case STRUMPACK_MPI_DIST: {
      switch (precision) {
      case STRUMPACK_FLOAT:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolverMPIDist<float,int>(comm, argc, argv, v));
        break;
      case STRUMPACK_DOUBLE:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolverMPIDist<double,int>(comm, argc, argv, v));
        break;
      case STRUMPACK_FLOATCOMPLEX:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolverMPIDist<std::complex<float>,int>
           (comm, argc, argv, v));
        break;
      case STRUMPACK_DOUBLECOMPLEX:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolverMPIDist<std::complex<double>,int>
           (comm, argc, argv, v));
        break;
      case STRUMPACK_FLOAT_64:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolverMPIDist<float,int64_t>
           (comm, argc, argv, v));
        break;
      case STRUMPACK_DOUBLE_64:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolverMPIDist<double,int64_t>
           (comm, argc, argv, v));
        break;
      case STRUMPACK_FLOATCOMPLEX_64:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolverMPIDist<std::complex<float>,int64_t>
           (comm, argc, argv, v));
        break;
      case STRUMPACK_DOUBLECOMPLEX_64:
        S->solver = static_cast<void*>
          (new StrumpackSparseSolverMPIDist<std::complex<double>,int64_t>
           (comm, argc, argv, v));
        break;
      default: std::cerr << "ERROR: wrong precision!" << std::endl;
      }
    } break;
    default: std::cerr << "ERROR: wrong interface!" << std::endl;
    }
  }

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

  void STRUMPACK_set_csr_matrix
  (STRUMPACK_SparseSolver S, const void* N, const void* row_ptr,
   const void* col_ind, const void* values, int symm) {
    switch (S.precision) {
    case STRUMPACK_FLOAT:
      CASTS(S.solver)->set_csr_matrix
        (*CREI(N), CREI(row_ptr), CREI(col_ind), CRES(values), symm);
      break;
    case STRUMPACK_DOUBLE:
      CASTD(S.solver)->set_csr_matrix
        (*CREI(N), CREI(row_ptr), CREI(col_ind), CRED(values), symm);
      break;
    case STRUMPACK_FLOATCOMPLEX:
      CASTC(S.solver)->set_csr_matrix
        (*CREI(N), CREI(row_ptr), CREI(col_ind), CREC(values), symm);
      break;
    case STRUMPACK_DOUBLECOMPLEX:
      CASTZ(S.solver)->set_csr_matrix
        (*CREI(N), CREI(row_ptr), CREI(col_ind), CREZ(values), symm);
      break;
    case STRUMPACK_FLOAT_64:
      CASTS64(S.solver)->set_csr_matrix
        (*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CRES(values), symm);
      break;
    case STRUMPACK_DOUBLE_64:
      CASTD64(S.solver)->set_csr_matrix
        (*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CRED(values), symm);
      break;
    case STRUMPACK_FLOATCOMPLEX_64:
      CASTC64(S.solver)->set_csr_matrix
        (*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CREC(values), symm);
      break;
    case STRUMPACK_DOUBLECOMPLEX_64:
      CASTZ64(S.solver)->set_csr_matrix
        (*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CREZ(values), symm);
      break;
    }
  }

  void STRUMPACK_set_distributed_csr_matrix
  (STRUMPACK_SparseSolver S, const void* N, const void* row_ptr,
   const void* col_ind, const void* values, const void* dist, int symm) {
    if (S.interface != STRUMPACK_MPI_DIST) {
      std::cerr << "ERROR: interface != STRUMPACK_MPI_DIST" << std::endl;
      return;
    }
    switch (S.precision) {
    case STRUMPACK_FLOAT:
      CASTSMPIDIST(S.solver)->set_distributed_csr_matrix
        (*CREI(N), CREI(row_ptr), CREI(col_ind), CRES(values), CREI(dist), symm);
      break;
    case STRUMPACK_DOUBLE:
      CASTDMPIDIST(S.solver)->set_distributed_csr_matrix
        (*CREI(N), CREI(row_ptr), CREI(col_ind), CRED(values), CREI(dist), symm);
      break;
    case STRUMPACK_FLOATCOMPLEX:
      CASTCMPIDIST(S.solver)->set_distributed_csr_matrix
        (*CREI(N), CREI(row_ptr), CREI(col_ind), CREC(values), CREI(dist), symm);
      break;
    case STRUMPACK_DOUBLECOMPLEX:
      CASTZMPIDIST(S.solver)->set_distributed_csr_matrix
        (*CREI(N), CREI(row_ptr), CREI(col_ind), CREZ(values), CREI(dist), symm);
      break;
    case STRUMPACK_FLOAT_64:
      CASTS64MPIDIST(S.solver)->set_distributed_csr_matrix
        (*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CRES(values), CRE64(dist), symm);
      break;
    case STRUMPACK_DOUBLE_64:
      CASTD64MPIDIST(S.solver)->set_distributed_csr_matrix
        (*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CRED(values), CRE64(dist), symm);
      break;
    case STRUMPACK_FLOATCOMPLEX_64:
      CASTC64MPIDIST(S.solver)->set_distributed_csr_matrix
        (*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CREC(values), CRE64(dist), symm);
      break;
    case STRUMPACK_DOUBLECOMPLEX_64:
      CASTZ64MPIDIST(S.solver)->set_distributed_csr_matrix
        (*CRE64(N), CRE64(row_ptr), CRE64(col_ind), CREZ(values), CRE64(dist), symm);
      break;
    }
  }

  void STRUMPACK_set_MPIAIJ_matrix
  (STRUMPACK_SparseSolver S, const void* n, const void* d_ptr,
   const void* d_ind, const void* d_val, const void* o_ptr, const void* o_ind,
   const void* o_val, const void* garray) {
    if (S.interface != STRUMPACK_MPI_DIST) {
      std::cerr << "ERROR: interface != STRUMPACK_MPI_DIST" << std::endl;
      return;
    }
    switch (S.precision) {
    case STRUMPACK_FLOAT:
      CASTSMPIDIST(S.solver)->set_MPIAIJ_matrix
        (*CREI(n), CREI(d_ptr), CREI(d_ind), CRES(d_val),
         CREI(o_ptr), CREI(o_ind), CRES(o_val), CREI(garray));
      break;
    case STRUMPACK_DOUBLE:
      CASTDMPIDIST(S.solver)->set_MPIAIJ_matrix
        (*CREI(n), CREI(d_ptr), CREI(d_ind), CRED(d_val),
         CREI(o_ptr), CREI(o_ind), CRED(o_val), CREI(garray));
      break;
    case STRUMPACK_FLOATCOMPLEX:
      CASTCMPIDIST(S.solver)->set_MPIAIJ_matrix
        (*CREI(n), CREI(d_ptr), CREI(d_ind), CREC(d_val),
         CREI(o_ptr), CREI(o_ind), CREC(o_val), CREI(garray));
      break;
    case STRUMPACK_DOUBLECOMPLEX:
      CASTZMPIDIST(S.solver)->set_MPIAIJ_matrix
        (*CREI(n), CREI(d_ptr), CREI(d_ind), CREZ(d_val),
         CREI(o_ptr), CREI(o_ind), CREZ(o_val), CREI(garray));
      break;
    case STRUMPACK_FLOAT_64:
      CASTS64MPIDIST(S.solver)->set_MPIAIJ_matrix
        (*CRE64(n), CRE64(d_ptr), CRE64(d_ind), CRES(d_val),
         CRE64(o_ptr), CRE64(o_ind), CRES(o_val), CRE64(garray));
      break;
    case STRUMPACK_DOUBLE_64:
      CASTD64MPIDIST(S.solver)->set_MPIAIJ_matrix
        (*CRE64(n), CRE64(d_ptr), CRE64(d_ind), CRED(d_val),
         CRE64(o_ptr), CRE64(o_ind), CRED(o_val), CRE64(garray));
      break;
    case STRUMPACK_FLOATCOMPLEX_64:
      CASTC64MPIDIST(S.solver)->set_MPIAIJ_matrix
        (*CRE64(n), CRE64(d_ptr), CRE64(d_ind), CREC(d_val),
         CRE64(o_ptr), CRE64(o_ind), CREC(o_val), CRE64(garray));
      break;
    case STRUMPACK_DOUBLECOMPLEX_64:
      CASTZ64MPIDIST(S.solver)->set_MPIAIJ_matrix
        (*CRE64(n), CRE64(d_ptr), CRE64(d_ind), CREZ(d_val),
         CRE64(o_ptr), CRE64(o_ind), CREZ(o_val), CRE64(garray));
      break;
    }
  }

  STRUMPACK_RETURN_CODE STRUMPACK_solve
  (STRUMPACK_SparseSolver S, const void* b, void* x,
   int use_initial_guess) {
    switch (S.precision) {
    case STRUMPACK_FLOAT:
      return static_cast<STRUMPACK_RETURN_CODE>
        (CASTS(S.solver)->solve(CRES(b), RES(x), use_initial_guess));
      break;
    case STRUMPACK_DOUBLE:
      return static_cast<STRUMPACK_RETURN_CODE>
        (CASTD(S.solver)->solve(CRED(b), RED(x), use_initial_guess));
      break;
    case STRUMPACK_FLOATCOMPLEX:
      return static_cast<STRUMPACK_RETURN_CODE>
        (CASTC(S.solver)->solve(CREC(b), REC(x), use_initial_guess));
      break;
    case STRUMPACK_DOUBLECOMPLEX:
      return static_cast<STRUMPACK_RETURN_CODE>
        (CASTZ(S.solver)->solve(CREZ(b), REZ(x), use_initial_guess));
      break;
    case STRUMPACK_FLOAT_64:
      return static_cast<STRUMPACK_RETURN_CODE>
        (CASTS64(S.solver)->solve(CRES(b), RES(x), use_initial_guess));
      break;
    case STRUMPACK_DOUBLE_64:
      return static_cast<STRUMPACK_RETURN_CODE>
        (CASTD64(S.solver)->solve(CRED(b), RED(x), use_initial_guess));
      break;
    case STRUMPACK_FLOATCOMPLEX_64:
      return static_cast<STRUMPACK_RETURN_CODE>
        (CASTC64(S.solver)->solve(CREC(b), REC(x), use_initial_guess));
      break;
    case STRUMPACK_DOUBLECOMPLEX_64:
      return static_cast<STRUMPACK_RETURN_CODE>
        (CASTZ64(S.solver)->solve(CREZ(b), REZ(x), use_initial_guess));
      break;
    }
    return STRUMPACK_SUCCESS;
  }

  void STRUMPACK_set_from_options(STRUMPACK_SparseSolver S) {
    switch_precision(set_from_options());
  }

  STRUMPACK_RETURN_CODE STRUMPACK_reorder(STRUMPACK_SparseSolver S) {
    ReturnCode c = ReturnCode::SUCCESS;;
    switch_precision_return(reorder(), c);
    return static_cast<STRUMPACK_RETURN_CODE>(c);
  }

  STRUMPACK_RETURN_CODE STRUMPACK_reorder_regular
  (STRUMPACK_SparseSolver S, int nx, int ny, int nz) {
    ReturnCode c = ReturnCode::SUCCESS;;
    switch_precision_return(reorder(nx, ny, nz), c);
    return static_cast<STRUMPACK_RETURN_CODE>(c);
  }
  STRUMPACK_RETURN_CODE STRUMPACK_factor(STRUMPACK_SparseSolver S) {
    ReturnCode c = ReturnCode::SUCCESS;
    switch_precision_return(factor(), c);
    return static_cast<STRUMPACK_RETURN_CODE>(c);
  }



  /*************************************************************
   ** Set options **********************************************
   ************************************************************/
  void STRUMPACK_set_verbose(STRUMPACK_SparseSolver S, int v)
  { switch_precision(options().set_verbose(static_cast<bool>(v))); }
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
  void STRUMPACK_set_reordering_method
  (STRUMPACK_SparseSolver S, STRUMPACK_REORDERING_STRATEGY m) {
    switch_precision
      (options().set_reordering_method
       (static_cast<ReorderingStrategy>(m)));
  }
  void STRUMPACK_set_GramSchmidt_type
  (STRUMPACK_SparseSolver S, STRUMPACK_GRAM_SCHMIDT_TYPE t) {
    switch_precision
      (options().set_GramSchmidt_type
       (static_cast<GramSchmidtType>(t)));
  }
  void STRUMPACK_set_mc64job(STRUMPACK_SparseSolver S, int job)
  { switch_precision(options().set_mc64job(job)); }
  void STRUMPACK_set_Krylov_solver
  (STRUMPACK_SparseSolver S, STRUMPACK_KRYLOV_SOLVER solver_type) {
    switch_precision
      (options().set_Krylov_solver
       (static_cast<KrylovSolver>(solver_type)));
  }

  /* set HSS specific options */
  void STRUMPACK_enable_HSS(STRUMPACK_SparseSolver S)
  { switch_precision(options().enable_HSS()); }
  void STRUMPACK_disable_HSS(STRUMPACK_SparseSolver S)
  { switch_precision(options().disable_HSS()); }
  void STRUMPACK_set_HSS_min_front_size(STRUMPACK_SparseSolver S, int size)
  { switch_precision(options().set_HSS_min_front_size(size)); }
  void STRUMPACK_set_HSS_min_sep_size(STRUMPACK_SparseSolver S, int size)
  { switch_precision(options().set_HSS_min_sep_size(size)); }
  void STRUMPACK_set_HSS_max_rank(STRUMPACK_SparseSolver S, int max_rank)
  { switch_precision(options().HSS_options().set_max_rank(max_rank)); }
  void STRUMPACK_set_HSS_leaf_size(STRUMPACK_SparseSolver S, int leaf_size)
  { switch_precision(options().HSS_options().set_leaf_size(leaf_size)); }
  void STRUMPACK_set_HSS_rel_tol(STRUMPACK_SparseSolver S, double rctol)
  { switch_precision(options().HSS_options().set_rel_tol(rctol)); }
  void STRUMPACK_set_HSS_abs_tol(STRUMPACK_SparseSolver S, double actol)
  { switch_precision(options().HSS_options().set_abs_tol(actol)); }


  /*************************************************************
   ** Get options **********************************************
   ************************************************************/
  int STRUMPACK_verbose(STRUMPACK_SparseSolver S) {
    int v = 0;
    switch_precision_return(options().verbose(), v);
    return v;
  }
  int STRUMPACK_maxit(STRUMPACK_SparseSolver S) {
    int maxit = 0;
    switch_precision_return(options().maxit(), maxit);
    return maxit;
  }
  int STRUMPACK_gmres_restart(STRUMPACK_SparseSolver S) {
    int restart = 0;
    switch_precision_return(options().gmres_restart(), restart);
    return restart;
  }
  double STRUMPACK_rel_tol(STRUMPACK_SparseSolver S) {
    double rtol = 0.;
    switch_precision_return(options().rel_tol(), rtol);
    return rtol;
  }
  double STRUMPACK_abs_tol(STRUMPACK_SparseSolver S) {
    double atol = 0.;
    switch_precision_return(options().abs_tol(), atol);
    return atol;
  }
  int STRUMPACK_nd_param(STRUMPACK_SparseSolver S) {
    int nd_param = 0;
    switch_precision_return(options().nd_param(), nd_param);
    return nd_param;
  }
  STRUMPACK_REORDERING_STRATEGY
  STRUMPACK_reordering_method(STRUMPACK_SparseSolver S) {
    ReorderingStrategy r = ReorderingStrategy::METIS;
    switch_precision_return(options().reordering_method(), r);
    return static_cast<STRUMPACK_REORDERING_STRATEGY>(r);
  }
  STRUMPACK_GRAM_SCHMIDT_TYPE
  STRUMPACK_GramSchmidt_type(STRUMPACK_SparseSolver S) {
    GramSchmidtType gs = GramSchmidtType::CLASSICAL;
    switch_precision_return(options().GramSchmidt_type(), gs);
    return static_cast<STRUMPACK_GRAM_SCHMIDT_TYPE>(gs);
  }
  int STRUMPACK_mc64job(STRUMPACK_SparseSolver S) {
    int job = 0;
    switch_precision_return(options().mc64job(), job);
    return job;
  }
  STRUMPACK_KRYLOV_SOLVER STRUMPACK_Krylov_solver(STRUMPACK_SparseSolver S) {
    KrylovSolver s = KrylovSolver::AUTO;
    switch_precision_return(options().Krylov_solver(), s);
    return static_cast<STRUMPACK_KRYLOV_SOLVER>(s);
  }

  /* get HSS specific options */
  int STRUMPACK_use_HSS(STRUMPACK_SparseSolver S) {
    int u = 0;
    switch_precision_return(options().use_HSS(), u);
    return u;
  }
  int STRUMPACK_HSS_min_front_size(STRUMPACK_SparseSolver S) {
    int size = 0;
    switch_precision_return(options().HSS_min_front_size(), size);
    return size;
  }
  int STRUMPACK_HSS_min_sep_size(STRUMPACK_SparseSolver S) {
    int size = 0;
    switch_precision_return(options().HSS_min_sep_size(), size);
    return size;
  }
  int STRUMPACK_HSS_max_rank(STRUMPACK_SparseSolver S) {
    int rank = 0;
    switch_precision_return(options().HSS_options().max_rank(), rank);
    return rank;
  }
  int STRUMPACK_HSS_leaf_size(STRUMPACK_SparseSolver S) {
    int l = 0;
    switch_precision_return(options().HSS_options().leaf_size(), l);
    return l;
  }
  double STRUMPACK_HSS_rel_tol(STRUMPACK_SparseSolver S) {
    double rctol = 0;
    switch_precision_return(options().HSS_options().rel_tol(), rctol);
    return rctol;
  }
  double STRUMPACK_HSS_abs_tol(STRUMPACK_SparseSolver S) {
    double actol = 0.;
    switch_precision_return(options().HSS_options().abs_tol(), actol);
    return actol;
  }

  /*************************************************************
   ** Get solve statistics *************************************
   ************************************************************/
  int STRUMPACK_its(STRUMPACK_SparseSolver S) {
    int its = 0;
    switch_precision_return(Krylov_iterations(), its);
    return its;
  }
  int STRUMPACK_rank(STRUMPACK_SparseSolver S) {
    int rank = 0;
    switch_precision_return(maximum_rank(), rank);
    return rank;
  }
  long long STRUMPACK_factor_nonzeros(STRUMPACK_SparseSolver S) {
    long long nz = 0;
    switch_precision_return(factor_nonzeros(), nz);
    return nz;
  }
  long long STRUMPACK_factor_memory(STRUMPACK_SparseSolver S) {
    long long mem = 0;
    switch_precision_return(factor_memory(), mem);
    return mem;
  }

}
