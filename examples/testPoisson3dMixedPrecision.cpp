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
#include <iostream>
#include <chrono>
#include "StrumpackSparseSolver.hpp"
#include "StrumpackSparseSolverMixedPrecision.hpp"
#include "sparse/CSRMatrix.hpp"

using namespace strumpack;

struct timing_results {
    double factor_time = 0;
    double solve_time = 0;
    double total_time = 0;
};


/* 
*  Solves Ax=b using a StrumpackSparseSolver and returns the duration elapsed
*  during the factor and the solve.
*/
timing_results time_non_mixed(int n, int m, const CSRMatrix<double, int>& A,
                      const DenseMatrix<double>& b, DenseMatrix<double>& x) {
    // Create sparse solver.
    StrumpackSparseSolver<double,int> spss;
    spss.options().set_matching(MatchingJob::NONE);
    spss.options().set_reordering_method(ReorderingStrategy::GEOMETRIC);
    spss.options().set_rel_tol(1e-15);
    spss.options().set_abs_tol(1e-15);
    // spss.options().set_from_command_line(argc, argv);

    // Solver setup.
    spss.set_matrix(A);
    spss.reorder(n, n, n);

    // Factor and solve. We time this.
    auto start = std::chrono::steady_clock::now();
    spss.factor();
    auto factor = std::chrono::steady_clock::now();
    spss.solve(b, x);
    auto end = std::chrono::steady_clock::now();

    timing_results res;
    res.factor_time = std::chrono::duration<double>(factor - start).count();
    res.solve_time = std::chrono::duration<double>(end - factor).count();
    return res;
}

/* 
*  Solves Ax=b using a StrumpackSparseSolverMixedPrecision and returns the 
*  duration elapsed during the solve.
*/
timing_results time_mixed(int n, int m, const CSRMatrix<double, int>& A,
                      const DenseMatrix<double>& b, DenseMatrix<double>& x) {
    // Create sparse solver.
    StrumpackSparseSolverMixedPrecision<float,double,int> spss_mixed;
    spss_mixed.solver_options().set_rel_tol(1e-15);
    spss_mixed.solver_options().set_abs_tol(1e-15);
    
    spss_mixed.solver_options().set_reordering_method(
        ReorderingStrategy::GEOMETRIC);
    spss_mixed.solver_options().set_matching(MatchingJob::NONE);
    // spss.options().set_from_command_line(argc, argv);

    // Solver setup.
    spss_mixed.set_matrix(A);
    spss_mixed.reorder(n, n, n);

    // Factor and solve. We time this.
    auto start = std::chrono::steady_clock::now();
    spss_mixed.factor();
    auto factor = std::chrono::steady_clock::now();
    spss_mixed.solve(b, x);
    auto end = std::chrono::steady_clock::now();
    timing_results res;
    res.factor_time = std::chrono::duration<double>(factor - start).count();
    res.solve_time = std::chrono::duration<double>(end - factor).count();
    return res;
}

template <typename scalar_t>
CSRMatrix<scalar_t,int> generate_matrix(int n) {
  int n2 = n * n;
  int N = n * n2;
  int nnz = 7 * N - 6 * n2;

  CSRMatrix<scalar_t,int> A(N, nnz);
  auto cptr = A.ptr();
  auto rind = A.ind();
  auto val = A.val();

  nnz = 0;
  cptr[0] = 0;
  for (int xdim=0; xdim<n; xdim++)
    for (int ydim=0; ydim<n; ydim++)
      for (int zdim=0; zdim<n; zdim++) {
        int ind = zdim+ydim*n+xdim*n2;
        val[nnz] = 6.0;
        rind[nnz++] = ind;
        if (zdim > 0)  { val[nnz] = -1.0; rind[nnz++] = ind-1; } // left
        if (zdim < n-1){ val[nnz] = -1.0; rind[nnz++] = ind+1; } // right
        if (ydim > 0)  { val[nnz] = -1.0; rind[nnz++] = ind-n; } // front
        if (ydim < n-1){ val[nnz] = -1.0; rind[nnz++] = ind+n; } // back
        if (xdim > 0)  { val[nnz] = -1.0; rind[nnz++] = ind-n2; } // up
        if (xdim < n-1){ val[nnz] = -1.0; rind[nnz++] = ind+n2; } // down
        cptr[ind+1] = nnz;
      }
  A.set_symm_sparse();
  return A;
}

void run_trial(int n, std::vector<timing_results>& nm_results,
               std::vector<timing_results>& mixed_results) {
  // Creating x1, x2, x_exact, and b.
  int m = 1;
  int n2 = n * n;
  int N = n * n2;
  DenseMatrix<double> b(N, m), x1(N, m), x2(N, m), x_exact(N, m);
  x_exact.random();
  
  CSRMatrix<double,int> A = generate_matrix<double>(n);
  A.spmv(x_exact, b);

  auto start = std::chrono::steady_clock::now();
  timing_results timing_mixed = time_mixed(n, m, A, b, x2);
  auto end = std::chrono::steady_clock::now();
  timing_mixed.total_time = std::chrono::duration<double>(end - start).count();
  start = std::chrono::steady_clock::now();
  timing_results timing_non_mixed = time_non_mixed(n, m, A, b, x1);
  end = std::chrono::steady_clock::now();
  timing_non_mixed.total_time = std::chrono::duration<double>(end - start).count();

  double residual_mixed = A.max_scaled_residual(x2.data(), b.data());
  double residual_non_mixed = A.max_scaled_residual(x1.data(), b.data());
  
  nm_results.emplace_back(timing_non_mixed);
  mixed_results.emplace_back(timing_mixed);

  const std::string marker = std::string(50, '-') + '\n';
  std::cout << marker << marker;
  std::cout << "solver, factor time (s), solve time (s), total time(s), residual\n";
  std::cout << marker;
  printf("non mixed, %.5f, %.5f, %.5f, %.10e\n", timing_non_mixed.factor_time, 
        timing_non_mixed.solve_time, timing_non_mixed.solve_time + timing_non_mixed.factor_time,
         residual_non_mixed);
  printf("mixed, %.5f, %.5f, %.5f, %.10e\n", timing_mixed.factor_time, 
        timing_mixed.solve_time, timing_mixed.solve_time + timing_mixed.factor_time, 
        residual_mixed);
  std::cout << marker << marker; 
}

void print_results(const std::vector<timing_results>& nm_results,
        const std::vector<timing_results>& mix_results, int start, int reps, int stepsize) {
    for (int i = 0; i < nm_results.size(); ++i ) {
        timing_results nm_curr = nm_results[i];
        timing_results mix_curr = mix_results[i];
        printf("%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", 
            start+(i*stepsize), 
            nm_curr.factor_time, nm_curr.solve_time, nm_curr.total_time,
            mix_curr.factor_time, mix_curr.solve_time, mix_curr.total_time);
    }
}

int main(int argc, char* argv[]) {
  std::vector<timing_results> nm_results_;
  std::vector<timing_results> mix_results_;
    
  int start = 30;
  int end = 81;
  int reps = 5;
  for (int n = start; n < end; ++n) {
    for (int t = 0; t < reps; ++t) {
      run_trial(n, nm_results_, mix_results_);
      std::cout << "n,non_factor,non_solve,non_total,mix_factor,mix_solve,mix_total\n";
      print_results(nm_results_, mix_results_, start, reps, 10);
    }
  }
  return 0;
}
