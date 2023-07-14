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

#include "StrumpackSparseSolver.hpp"

#if defined(STRUMPACK_USE_PAPI)
#include <papi.h>
#endif

#include "misc/Tools.hpp"
#include "misc/TaskTimer.hpp"
#include "StrumpackOptions.hpp"
#include "sparse/ordering/MatrixReordering.hpp"
#include "sparse/EliminationTree.hpp"
#include "iterative/IterativeSolvers.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  SparseSolver<scalar_t,integer_t>::SparseSolver
  (bool verbose, bool root) : SparseSolverBase<scalar_t,integer_t>
    (verbose, root) { }

  template<typename scalar_t,typename integer_t>
  SparseSolver<scalar_t,integer_t>::SparseSolver
  (int argc, char* argv[], bool verbose, bool root)
    : SparseSolverBase<scalar_t,integer_t>
    (argc, argv, verbose, root) { }

  template<typename scalar_t,typename integer_t>
  SparseSolver<scalar_t,integer_t>::~SparseSolver() = default;


  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::setup_tree() {
    tree_.reset(new EliminationTree<scalar_t,integer_t>
                (opts_, *mat_, nd_->tree()));
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::setup_reordering() {
    nd_.reset(new MatrixReordering<scalar_t,integer_t>(matrix()->size()));
  }

  template<typename scalar_t,typename integer_t> int
  SparseSolver<scalar_t,integer_t>::compute_reordering
  (const int* p, int base, int nx, int ny, int nz,
   int components, int width) {
    if (p) return nd_->set_permutation(opts_, *mat_, p, base);
    return nd_->nested_dissection
      (opts_, *mat_, nx, ny, nz, components, width);
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::separator_reordering() {
    nd_->separator_reordering(opts_, *mat_, tree_->root());
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::set_matrix
  (const CSRMatrix<scalar_t,integer_t>& A) {
    mat_.reset(new CSRMatrix<scalar_t,integer_t>(A));
    factored_ = reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::update_matrix_values
  (const CSRMatrix<scalar_t,integer_t>& A) {
    if (!(mat_ && A.size() == mat_->size() && A.nnz() <= mat_->nnz())) {
      // matrix() has been made symmetric, can have more nonzeros
      this->print_wrong_sparsity_error();
      return;
    }
    mat_.reset(new CSRMatrix<scalar_t,integer_t>(A));
    permute_matrix_values();
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::set_csr_matrix
  (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, bool symmetric_pattern) {
    mat_.reset(new CSRMatrix<scalar_t,integer_t>
               (N, row_ptr, col_ind, values, symmetric_pattern));
    factored_ = reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::update_matrix_values
  (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, bool symmetric_pattern) {
    if (!(mat_ && N == mat_->size() && row_ptr[N] <= mat_->nnz())) {
      // matrix() has been made symmetric, can have more nonzeros
      this->print_wrong_sparsity_error();
      return;
    }
    mat_.reset(new CSRMatrix<scalar_t,integer_t>
               (N, row_ptr, col_ind, values, symmetric_pattern));
    permute_matrix_values();
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::permute_matrix_values() {
    if (reordered_) {
      matrix()->apply_matching(matching_);
      matrix()->equilibrate(equil_);
      matrix()->symmetrize_sparsity();
      matrix()->permute(reordering()->iperm(), reordering()->perm());
      if (opts_.compression() != CompressionType::NONE)
        separator_reordering();
    }
    factored_ = false;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolver<scalar_t,integer_t>::solve_internal
  (const scalar_t* b, scalar_t* x, bool use_initial_guess) {
    auto N = matrix()->size();
    auto B = ConstDenseMatrixWrapperPtr(N, 1, b, N);
    DenseMW_t X(N, 1, x, N);
    return this->solve(*B, X, use_initial_guess);
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::transform_x0
  (DenseM_t& x, DenseM_t& xtmp) {
    integer_t N = matrix()->size(), d = x.cols();
    auto& P = reordering()->iperm();
    if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
      for (integer_t j=0; j<d; j++)
#pragma omp parallel for
        for (integer_t i=0; i<N; i++)
          x(i, j) = x(i, j) / matching_.C[i];
    if (opts_.matching() == MatchingJob::NONE)
      xtmp.copy(x);
    else
      for (integer_t j=0; j<d; j++)
#pragma omp parallel for
        for (integer_t i=0; i<N; i++)
          xtmp(i, j) = x(matching_.Q[i], j);
    if (this->equil_.type == EquilibrationType::COLUMN ||
        this->equil_.type == EquilibrationType::BOTH)
      for (integer_t j=0; j<d; j++)
#pragma omp parallel for
        for (integer_t i=0; i<N; i++)
          xtmp(i, j) = xtmp(i, j) / equil_.C[i];
    for (integer_t j=0; j<d; j++)
#pragma omp parallel for
      for (integer_t i=0; i<N; i++)
        x(i, j) = xtmp(P[i], j);
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::transform_x
  (DenseM_t& x, DenseM_t& xtmp) {
    integer_t N = matrix()->size(), d = x.cols();
    auto& Pi = reordering()->perm();
    for (integer_t j=0; j<d; j++)
#pragma omp parallel for
      for (integer_t i=0; i<N; i++)
        xtmp(i, j) = x(Pi[i], j);
    if (this->equil_.type == EquilibrationType::COLUMN ||
        this->equil_.type == EquilibrationType::BOTH)
      for (integer_t j=0; j<d; j++)
#pragma omp parallel for
        for (integer_t i=0; i<N; i++)
          xtmp(i, j) = equil_.C[i] * xtmp(i, j);
    if (opts_.matching() == MatchingJob::NONE)
      x.copy(xtmp);
    else {
      for (integer_t j=0; j<d; j++)
#pragma omp parallel for
        for (integer_t i=0; i<N; i++)
          x(matching_.Q[i], j) = xtmp(i, j);
      if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
        for (integer_t j=0; j<d; j++)
#pragma omp parallel for
          for (integer_t i=0; i<N; i++)
            x(i, j) = matching_.C[i] * x(i, j);
    }
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::transform_b
  (const DenseM_t& b, DenseM_t& bloc) {
    using real_t = typename RealType<scalar_t>::value_type;
    integer_t N = matrix()->size(), d = b.cols();
    auto& P = reordering()->iperm();
    std::vector<real_t> R(N, 1.);
    if (equil_.type == EquilibrationType::ROW ||
        equil_.type == EquilibrationType::BOTH)
      for (integer_t i=0; i<N; i++)
        R[i] *= equil_.R[i];
    if (this->reordered_ &&
        opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
      for (integer_t i=0; i<N; i++)
        R[i] *= matching_.R[i];
    for (integer_t j=0; j<d; j++)
#pragma omp parallel for
      for (integer_t i=0; i<N; i++) {
        auto p = P[i];
        bloc(i, j) = R[p] * b(p, j);
      }
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolver<scalar_t,integer_t>::solve_internal
  (const DenseM_t& b, DenseM_t& x, bool use_initial_guess) {
    TaskTimer t("solve");
    this->perf_counters_start();
    t.start();
    assert(b.cols() == x.cols());

    // reordering has to be called, even for the iterative solvers
    if (!this->reordered_) {
      ReturnCode ierr = this->reorder();
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }
    // factor needs to be called, except for the non-preconditioned
    // solvers
    if (!this->factored_ &&
        (opts_.Krylov_solver() != KrylovSolver::GMRES) &&
        (opts_.Krylov_solver() != KrylovSolver::BICGSTAB)) {
      ReturnCode ierr = this->factor();
      // TODO there could be zero pivots, but replaced, and this
      // should still continue!!
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }

    integer_t d = b.cols();
    assert(matrix()->size() < std::numeric_limits<int>::max());
    DenseM_t bloc(b.rows(), d);

    auto spmv = [&](const scalar_t* x, scalar_t* y)
                { matrix()->spmv(x, y); };
    Krylov_its_ = 0;

    if (use_initial_guess &&
        opts_.Krylov_solver() != KrylovSolver::DIRECT)
      transform_x0(x, bloc);
    transform_b(b, bloc);

    auto MFsolve =
      [&](scalar_t* w) {
        DenseMW_t X(x.rows(), 1, w, x.ld());
        tree()->multifrontal_solve(X);
      };

    switch (opts_.Krylov_solver()) {
    case KrylovSolver::AUTO: {
      if (opts_.compression() != CompressionType::NONE && x.cols() == 1)
        iterative::GMRes<scalar_t>
          (spmv, MFsolve, x.rows(), x.data(), bloc.data(),
           opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
           opts_.gmres_restart(), opts_.GramSchmidt_type(),
           use_initial_guess, opts_.verbose() && is_root_);
      else
        iterative::IterativeRefinement<scalar_t,integer_t>
          (*matrix(), [&](DenseM_t& w) { tree()->multifrontal_solve(w); },
           x, bloc, opts_.rel_tol(), opts_.abs_tol(),
           Krylov_its_, opts_.maxit(), use_initial_guess,
           opts_.verbose() && is_root_);
    }; break;
    case KrylovSolver::DIRECT: {
      x = bloc;
      tree()->multifrontal_solve(x);
    }; break;
    case KrylovSolver::REFINE: {
      iterative::IterativeRefinement<scalar_t,integer_t>
        (*matrix(), [&](DenseM_t& w) { tree()->multifrontal_solve(w); },
         x, bloc, opts_.rel_tol(), opts_.abs_tol(),
         Krylov_its_, opts_.maxit(), use_initial_guess,
         opts_.verbose() && is_root_);
    }; break;
    case KrylovSolver::PREC_GMRES: {
      assert(x.cols() == 1);
      iterative::GMRes<scalar_t>
        (spmv, MFsolve, x.rows(), x.data(), bloc.data(),
         opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
         opts_.gmres_restart(), opts_.GramSchmidt_type(),
         use_initial_guess, opts_.verbose() && is_root_);
    }; break;
    case KrylovSolver::PREC_BICGSTAB: {
      assert(x.cols() == 1);
      iterative::BiCGStab<scalar_t>
        (spmv, MFsolve, x.rows(), x.data(), bloc.data(),
         opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
         use_initial_guess, opts_.verbose() && is_root_);
    }; break;
    case KrylovSolver::GMRES: { // see above
      assert(x.cols() == 1);
      iterative::GMRes<scalar_t>
        (spmv, [](scalar_t* x) {}, x.rows(), x.data(), bloc.data(),
         opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
         opts_.gmres_restart(), opts_.GramSchmidt_type(),
         use_initial_guess, opts_.verbose() && is_root_);
    }; break;
    case KrylovSolver::BICGSTAB: {
      assert(x.cols() == 1);
      iterative::BiCGStab<scalar_t>
        (spmv, [](scalar_t* x) {}, x.rows(), x.data(), bloc.data(),
         opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
         use_initial_guess, opts_.verbose() && is_root_);
    }
    }
    transform_x(x, bloc);

    t.stop();
    this->perf_counters_stop("DIRECT/GMRES solve");
    this->print_solve_stats(t);
    return ReturnCode::SUCCESS;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolver<scalar_t,integer_t>::delete_factors_internal() {
    tree_.reset(nullptr);
  }

  // explicit template instantiations
  template class SparseSolver<float,int>;
  template class SparseSolver<double,int>;
  template class SparseSolver<std::complex<float>,int>;
  template class SparseSolver<std::complex<double>,int>;

  template class SparseSolver<float,long int>;
  template class SparseSolver<double,long int>;
  template class SparseSolver<std::complex<float>,long int>;
  template class SparseSolver<std::complex<double>,long int>;

  template class SparseSolver<float,long long int>;
  template class SparseSolver<double,long long int>;
  template class SparseSolver<std::complex<float>,long long int>;
  template class SparseSolver<std::complex<double>,long long int>;

} //end namespace strumpack
