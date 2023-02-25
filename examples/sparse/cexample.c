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
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "StrumpackSparseSolver.h"

int main(int argc, char* argv[]) {
  int n = 400;
  if (argc > 1) n = atoi(argv[1]); // get grid size
  else printf("# no grid size provided\n");
  printf("# solving %d^2 Poisson problem with a single right hand side\n", n);

  STRUMPACK_SparseSolver S;
  STRUMPACK_init_mt(&S, STRUMPACK_FLOATCOMPLEX, STRUMPACK_MT, argc, argv, 1);
  STRUMPACK_set_matching(S, STRUMPACK_MATCHING_NONE);
  STRUMPACK_set_reordering_method(S, STRUMPACK_GEOMETRIC);

  /*
    Set compression method. Other options include NONE, HSS, HODLR,
    LOSSY, LOSSLESS. HODLR is only supported in parallel, and only
    supports double precision (including complex double).
  */
  STRUMPACK_set_compression(S, STRUMPACK_BLR);

  /*
    Set the block size and relative compression tolerances for BLR
    compression.
  */
  STRUMPACK_set_compression_leaf_size(S, 64);
  STRUMPACK_set_compression_rel_tol(S, 1.e-2);

  /*
    Only sub-blocks in the sparse triangular factors corresponing to
    separators larger than this minimum separator size will be
    compressed. For performance, this value should probably be larger
    than 128. This value should be larger for HODLR/HODBF, than for
    BLR, since HODLR/HODBF have larger constants in the complexity.
    For an n x n 2D domain, the largest separator will correspond to
    an n x n sub-block in the sparse factors.
  */
  STRUMPACK_set_compression_min_sep_size(S, 300);

  /*
    Parse any command line options.
  */
  STRUMPACK_set_from_options(S);

  int N = n * n;
  int nnz = 5 * N - 4 * n;
  int* row_ptr = malloc((N+1)*sizeof(int));
  int* col_ind = malloc(nnz*sizeof(int));
  float complex* val = malloc(nnz*sizeof(float complex));

  nnz = 0;
  row_ptr[0] = 0;
  int row, col, ind;
  for (row=0; row<n; row++) {
    for (col=0; col<n; col++) {
      ind = col+n*row;
      val[nnz] = 4.0 + 0. * I;
      col_ind[nnz++] = ind;
      if (col > 0)  { val[nnz] = -1. + 0. * I; col_ind[nnz++] = ind-1; } // left
      if (col < n-1){ val[nnz] = -1. + 0. * I; col_ind[nnz++] = ind+1; } // right
      if (row > 0)  { val[nnz] = -1. + 0. * I; col_ind[nnz++] = ind-n; } // up
      if (row < n-1){ val[nnz] = -1. + 0. * I; col_ind[nnz++] = ind+n; } // down
      row_ptr[ind+1] = nnz;
    }
  }
  float complex* b = malloc(N*sizeof(float complex));
  float complex* x = malloc(N*sizeof(float complex));
  int i;
  for (i=0; i<N; i++) {
    b[i] = 1. + 1. * I;
    x[i] = 0. + 0. * I;
  }

  STRUMPACK_set_csr_matrix(S, &N, row_ptr, col_ind, val, 1);
  /* n x n x 1 mesh, 1 component per grid-point, separator of width 1 */
  STRUMPACK_reorder_regular(S, n, n, 1, 1, 1);

  /*
    Solve will internally call factor (and reorder if necessary).
   */
  STRUMPACK_solve(S, b, x, 0);

  free(row_ptr);
  free(col_ind);
  free(val);
  free(b);
  free(x);
  STRUMPACK_destroy(&S);
  return 0;
}
