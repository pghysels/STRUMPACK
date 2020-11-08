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
    : StrumpackSparseSolver<factor_t,integer_t>(verbose, root) { }

  template<typename factor_t,typename refine_t,typename integer_t>
  StrumpackSparseSolverMixedPrecision<factor_t,refine_t,integer_t>::
  StrumpackSparseSolverMixedPrecision
  (int argc, char* argv[], bool verbose, bool root)
    : StrumpackSparseSolver<factor_t,integer_t>
    (argc, argv, verbose, root) { }

  template<typename factor_t,typename refine_t,typename integer_t>
  StrumpackSparseSolverMixedPrecision<factor_t,refine_t,integer_t>::
  ~StrumpackSparseSolverMixedPrecision() = default;

  template<typename factor_t,typename refine_t,typename integer_t>
  void StrumpackSparseSolverMixedPrecision<factor_t,refine_t,integer_t>::
  refine(const DenseMatrix<refine_t>& b, DenseMatrix<refine_t>& x) {
    integer_t d = b.cols();
    DenseMatrix<refine_t> bloc(b.rows(), d);
    iterative::IterativeRefinement<refine_t,integer_t>
        (*refine_mat_, [&](DenseMatrix<refine_t>& w) { 
           refine_tree_->multifrontal_solve(w); },
        x, bloc, this->opts_.rel_tol(), this->opts_.abs_tol(),
        this->Krylov_its_, this->opts_.maxit(), /*use_initial_guess=*/false,
        this->opts_.verbose() && this->is_root_);
  }

  template<typename factor_t,typename refine_t,typename integer_t>
  void StrumpackSparseSolverMixedPrecision<factor_t,refine_t,integer_t>::
  create_refine_matrix() {

  }

  template<typename factor_t,typename refine_t,typename integer_t>
  void StrumpackSparseSolverMixedPrecision<factor_t,refine_t,integer_t>::
  create_refine_tree() {

  }

  // explicit template instantiations
  template class StrumpackSparseSolverMixedPrecision<float,double,int>;
  template class StrumpackSparseSolverMixedPrecision
      <std::complex<float>,std::complex<double>,int>;
} //end namespace strumpack
