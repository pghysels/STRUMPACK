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

#include "StrumpackSparseSolverMixedPrecision.hpp"

#if defined(STRUMPACK_USE_PAPI)
#include <papi.h>
#endif

#include "misc/Tools.hpp"
#include "misc/TaskTimer.hpp"
#include "StrumpackOptions.hpp"
#include "sparse/ordering/MatrixReordering.hpp"
#include "sparse/EliminationTree.hpp"
#include "sparse/iterative/IterativeSolvers.hpp"

namespace strumpack {

  template<typename factor_t,typename refine_t,typename integer_t>
  StrumpackSparseSolverMixedPrecision<factor_t,refine_t,integer_t>::
  StrumpackSparseSolverMixedPrecision(bool verbose, bool root)
    : solver_(verbose, root) {
      solver_.options().set_Krylov_solver(KrylovSolver::DIRECT);
    }

  template<typename factor_t,typename refine_t,typename integer_t>
  StrumpackSparseSolverMixedPrecision<factor_t,refine_t,integer_t>::
  ~StrumpackSparseSolverMixedPrecision() = default;

  template<typename factor_t,typename refine_t,typename integer_t> void
  StrumpackSparseSolverMixedPrecision<factor_t,refine_t,integer_t>::
  solve(const DenseMatrix<refine_t>& b, DenseMatrix<refine_t>& x) {
  auto solve_func = [&](DenseMatrix<refine_t>& w) {
    DenseMatrix<factor_t> new_x(w.rows(), w.cols());
    DenseMatrix<factor_t> cast_b = cast_matrix<refine_t,factor_t>(w);
    solver_.solve(cast_b, new_x);
    w = cast_matrix<factor_t,refine_t>(new_x);
  };
  int totit = 0;
  iterative::IterativeRefinement<refine_t,integer_t>(
    mat_, solve_func, x, b, opts_.rel_tol(), opts_.abs_tol(), totit,
    opts_.maxit(), /*use_initial_guess=*/false, opts_.verbose());
  }

  template<typename factor_t,typename refine_t,typename integer_t> void
  StrumpackSparseSolverMixedPrecision<factor_t,refine_t,integer_t>::
  factor() {
    solver_.factor();
  }

  template<typename factor_t,typename refine_t,typename integer_t> ReturnCode
  StrumpackSparseSolverMixedPrecision<factor_t,refine_t,integer_t>::
  reorder(int nx, int ny, int nz) {
    return solver_.reorder(nx, ny, nz);
  }

  template<typename factor_t,typename refine_t,typename integer_t> void
  StrumpackSparseSolverMixedPrecision<factor_t,refine_t,integer_t>::
  set_matrix(const CSRMatrix<refine_t,integer_t>& A) {
      mat_ = A;
      CSRMatrix<factor_t,integer_t> cast_mat = 
          cast_matrix<refine_t,integer_t,factor_t>(A);
      cast_mat.set_symm_sparse(A.symm_sparse());
      solver_.set_matrix(cast_mat);
  }

  // explicit template instantiations
  template class StrumpackSparseSolverMixedPrecision<float,double,int>;

} //end namespace strumpack
