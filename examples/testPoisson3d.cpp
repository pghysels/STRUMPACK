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
#include "StrumpackSparseSolver.hpp"
#include "sparse/CSRMatrix.hpp"

typedef double scalar;
typedef int64_t integer;

using namespace strumpack;

int main(int argc, char* argv[]) {
  int n = 30;
  if (argc > 1) n = atoi(argv[1]); // get grid size
  else std::cout << "# please provide grid size" << std::endl;

  StrumpackSparseSolver<scalar,integer> spss;
  spss.options().set_matching(MatchingJob::NONE);
  spss.options().set_reordering_method(ReorderingStrategy::GEOMETRIC);
  spss.options().set_from_command_line(argc, argv);

  int n2 = n * n;
  int N = n * n2;
  int nnz = 7 * N - 6 * n2;
  CSRMatrix<scalar,integer> A(N, nnz);
  auto cptr = A.ptr();
  auto rind = A.ind();
  auto val = A.val();

  nnz = 0;
  cptr[0] = 0;
  for (integer xdim=0; xdim<n; xdim++)
    for (integer ydim=0; ydim<n; ydim++)
      for (integer zdim=0; zdim<n; zdim++) {
        integer ind = zdim+ydim*n+xdim*n2;
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

  std::vector<scalar> b(N, scalar(0.)), x(N, scalar(0.)),
    x_exact(N, scalar(1.));

  A.spmv(x_exact.data(), b.data());

  spss.set_matrix(A);
  spss.reorder(n, n, n);
  spss.factor();
  spss.solve(b.data(), x.data());

  // spss.draw("P3D" + std::to_string(n));

  std::cout << "# COMPONENTWISE SCALED RESIDUAL = "
            << A.max_scaled_residual(x.data(), b.data()) << std::endl;

  return 0;
}
