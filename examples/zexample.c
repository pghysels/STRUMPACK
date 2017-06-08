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
#include <stdio.h>
#include <stdlib.h>
#include "StrumpackSparseSolver.h"

int main(int argc, char* argv[]) {
  int n = 30;
  if (argc > 1) n = atoi(argv[1]); // get grid size
  else printf("# please provide grid size\n");

  STRUMPACK_SparseSolver spss;
  STRUMPACK_init(&spss, 0, STRUMPACK_DOUBLECOMPLEX, STRUMPACK_MT, argc, argv, 1);
  STRUMPACK_set_mc64job(spss, 0);
  STRUMPACK_set_reordering_method(spss, STRUMPACK_GEOMETRIC);
  STRUMPACK_set_from_options(spss);

  int N = n * n;
  int nnz = 5 * N - 4 * n;
  int* row_ptr = malloc((N+1)*sizeof(int));
  int* col_ind = malloc(nnz*sizeof(int));
  doublecomplex* val = malloc(nnz*sizeof(doublecomplex));

  nnz = 0;
  row_ptr[0] = 0;
  int row, col, ind;
  for (row=0; row<n; row++) {
    for (col=0; col<n; col++) {
      ind = col+n*row;
      val[nnz].r = 4.; val[nnz].i = 0.;
      col_ind[nnz++] = ind;
      if (col > 0)  { val[nnz].r = -1.; val[nnz].i = 0.; col_ind[nnz++] = ind-1; } // left
      if (col < n-1){ val[nnz].r = -1.; val[nnz].i = 0.; col_ind[nnz++] = ind+1; } // right
      if (row > 0)  { val[nnz].r = -1.; val[nnz].i = 0.; col_ind[nnz++] = ind-n; } // up
      if (row < n-1){ val[nnz].r = -1.; val[nnz].i = 0.; col_ind[nnz++] = ind+n; } // down
      row_ptr[ind+1] = nnz;
    }
  }
  doublecomplex* b = malloc(N*sizeof(doublecomplex));
  doublecomplex* x = malloc(N*sizeof(doublecomplex));
  int i;
  for (i=0; i<N; i++) {
    b[i].r = 1.; b[i].i = 0.;
    x[i].r = 0.; x[i].i = 0.;
  }
  STRUMPACK_set_csr_matrix(spss, &N, row_ptr, col_ind, val, 1);
  STRUMPACK_reorder_regular(spss, n, n, 1);
  STRUMPACK_solve(spss, b, x, 0);

  free(row_ptr);
  free(col_ind);
  free(val);
  free(b);
  free(x);
  STRUMPACK_destroy(&spss);
  return 0;
}
