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

#include "StrumpackSparseSolverMixedPrecisionMPIDist.hpp"

#if defined(STRUMPACK_USE_PAPI)
#include <papi.h>
#endif

#include "misc/Tools.hpp"
#include "misc/TaskTimer.hpp"
#include "StrumpackOptions.hpp"
#include "sparse/ordering/MatrixReordering.hpp"
#include "sparse/EliminationTree.hpp"
#include "iterative/IterativeSolversMPI.hpp"

namespace strumpack {

  template<typename factor_t,typename refine_t,typename integer_t>
  SparseSolverMixedPrecisionMPIDist<factor_t,refine_t,integer_t>::
  SparseSolverMixedPrecisionMPIDist(MPI_Comm comm, bool verbose)
    : solver_(comm, verbose) {
    solver_.options().set_Krylov_solver(KrylovSolver::DIRECT);
  }

  template<typename factor_t,typename refine_t,typename integer_t>
  SparseSolverMixedPrecisionMPIDist<factor_t,refine_t,integer_t>::
  SparseSolverMixedPrecisionMPIDist(MPI_Comm comm, int argc, char* argv[],
                                    bool verbose)
    : solver_(comm, argc, argv, verbose), opts_(argc, argv) {
    solver_.options().set_Krylov_solver(KrylovSolver::DIRECT);
  }

  template<typename factor_t,typename refine_t,typename integer_t>
  SparseSolverMixedPrecisionMPIDist<factor_t,refine_t,integer_t>::
  ~SparseSolverMixedPrecisionMPIDist() = default;

  template<typename factor_t,typename refine_t,typename integer_t> ReturnCode
  SparseSolverMixedPrecisionMPIDist<factor_t,refine_t,integer_t>::
  solve(const DenseMatrix<refine_t>& b, DenseMatrix<refine_t>& x,
        bool use_initial_guess) {
    auto solve_func =
      [&](DenseMatrix<refine_t>& w) {
        DenseMatrix<factor_t> new_x(w.rows(), w.cols()),
          cast_b = cast_matrix<refine_t,factor_t>(w);
        solver_.solve(cast_b, new_x);
        copy(new_x, w);
      };
    auto solve_func_ptr =
      [&](refine_t* w) {
        DenseMatrixWrapper<refine_t> wx(x.rows(), 1, w, x.rows());
        solve_func(wx);
      };
    auto spmv = [&](const refine_t* x, refine_t* y) { mat_.spmv(x, y); };

    auto old_verbose = solver_.options().verbose();
    solver_.options().set_verbose(false);
    Krylov_its_ = 0;
    bool verbose = opts_.verbose() && solver_.Comm().is_root();
    switch (opts_.Krylov_solver()) {
    case KrylovSolver::AUTO: {
      if (opts_.compression() != CompressionType::NONE && x.cols() == 1)
        iterative::GMResMPI<refine_t>
          (solver_.Comm(), spmv, solve_func_ptr, x.rows(), x.data(), b.data(),
           opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
           opts_.gmres_restart(), opts_.GramSchmidt_type(),
           use_initial_guess, verbose);
      else
        iterative::IterativeRefinementMPI<refine_t,integer_t>
          (solver_.Comm(), mat_, solve_func, x, b,
           opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
           use_initial_guess, verbose);
    }; break;
    case KrylovSolver::DIRECT: {
      copy(b, x, 0, 0);
      solve_func(x);
    }; break;
    case KrylovSolver::REFINE: {
      iterative::IterativeRefinementMPI<refine_t,integer_t>
        (solver_.Comm(), mat_, solve_func, x, b,
         opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
         use_initial_guess, verbose);
    }; break;
    case KrylovSolver::PREC_GMRES: {
      assert(x.cols() == 1);
      iterative::GMResMPI<refine_t>
        (solver_.Comm(), spmv, solve_func_ptr, x.rows(), x.data(), b.data(),
         opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
         opts_.gmres_restart(), opts_.GramSchmidt_type(),
         use_initial_guess, verbose);
    }; break;
    case KrylovSolver::PREC_BICGSTAB: {
      assert(x.cols() == 1);
      iterative::BiCGStabMPI<refine_t>
        (solver_.Comm(), spmv, solve_func_ptr, x.rows(), x.data(), b.data(),
         opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
         use_initial_guess, verbose);
    }; break;
    case KrylovSolver::GMRES:
    case KrylovSolver::BICGSTAB: {
      std::cerr << "ERROR: non-preconditioned solvers not supported "
        "as outer solver in mixed-precision solver." << std::endl;
    }
    }
    // TODO check convergence, return whether or not this converged
    solver_.options().set_verbose(old_verbose);
    return ReturnCode::SUCCESS;
  }

  template<typename factor_t,typename refine_t,typename integer_t> ReturnCode
  SparseSolverMixedPrecisionMPIDist<factor_t,refine_t,integer_t>::
  solve(const refine_t* b, refine_t* x, bool use_initial_guess) {
    auto N = mat_.local_rows();
    auto B = ConstDenseMatrixWrapperPtr(N, 1, b, N);
    DenseMatrixWrapper<refine_t> X(N, 1, x, N);
    return solve(*B, X, use_initial_guess);
  }

  template<typename factor_t,typename refine_t,typename integer_t> ReturnCode
  SparseSolverMixedPrecisionMPIDist<factor_t,refine_t,integer_t>::
  factor() {
    return solver_.factor();
  }

  template<typename factor_t,typename refine_t,typename integer_t> ReturnCode
  SparseSolverMixedPrecisionMPIDist<factor_t,refine_t,integer_t>::
  reorder(int nx, int ny, int nz) {
    return solver_.reorder(nx, ny, nz);
  }

  template<typename factor_t,typename refine_t,typename integer_t> void
  SparseSolverMixedPrecisionMPIDist<factor_t,refine_t,integer_t>::
  set_matrix(const CSRMatrixMPI<refine_t,integer_t>& A) {
    mat_ = A;
    solver_.set_matrix(cast_matrix<refine_t,integer_t,factor_t>(A));
  }

  // explicit template instantiations
  template class SparseSolverMixedPrecisionMPIDist<float,double,int>;
  template class SparseSolverMixedPrecisionMPIDist<std::complex<float>,std::complex<double>,int>;

} //end namespace strumpack
