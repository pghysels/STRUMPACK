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
#include <math.h>
#include "structured/StructuredMatrix.h"
#include "StrumpackFortranCInterface.h"


void Cblacs_get(int, int, int *);
void Cblacs_gridinit(int *, const char *, int, int);
void Cblacs_gridmap(int *, int *, int, int, int);
void Cblacs_gridinfo(int, int *, int *, int *, int *);
void Cblacs_gridexit(int);
void Cblacs_exit(int);
int Csys2blacs_handle(MPI_Comm);
MPI_Comm Cblacs2sys_handle(int);

#define numroc STRUMPACK_FC_GLOBAL(numroc,NUMROC)
int numroc(int*, int*, int* , int *, int *);
#define descinit STRUMPACK_FC_GLOBAL(descinit,DESCINIT)
void descinit(int *, int *, int *, int *, int *, int *,
              int *, int *, int *, int *);

/* zero based indexing */
int indxl2g(int INDXLOC, int NB, int IPROC, int ISRCPROC, int NPROCS) {
  return NPROCS*NB*((INDXLOC)/NB) + (INDXLOC) % NB +
    ((NPROCS+IPROC-ISRCPROC) % NPROCS)*NB;
}


double Toeplitz(int i, int j) {
  return 1. / (1. + abs(i-j));
}


void test(CSPStructMat H, int n, int nrhs, int rank);


int main(int argc, char* argv[]) {
  int thread_level;
  int rank;
  int P;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  if (thread_level != MPI_THREAD_FUNNELED && rank == 0)
    printf("MPI implementation does not support MPI_THREAD_FUNNELED");

  int n = 1000;
  int nrhs = 10;
  if (argc > 1) n = atoi(argv[1]); /* get matrix size */
  else if (rank == 0)
    printf("# no matrix size provided\n");

  CSPOptions opts;
  SP_d_struct_default_options(&opts);

  opts.verbose = 0;
  opts.rel_tol = 1e-8;
  opts.type = SP_TYPE_HODBF;
  /* SP_TYPE_BLR, SP_TYPE_HSS, SP_TYPE_HODLR, SP_TYPE_BUTTERFLY, */
  /* SP_TYPE_LR */

  {
    /*
     * Construct a structured matrix H using a routine to compute
     * individual elements, see function Toeplitz above.
     */
    if (rank == 0)
      printf("# Construction from element routine\n");
    CSPStructMat H;
    SP_d_struct_from_elements_mpi
      (&H, MPI_COMM_WORLD, n, n, Toeplitz, &opts);
    test(H, n, nrhs, rank);
    SP_d_struct_destroy(&H);
  }

  {
    /*
     * Construct a structured matrix H from a 2d block cyclic matrix.
     */
    /*
     * First create a 2d processor grid, try to make it square, some
     * processes might be idle.
    */
    if (rank == 0)
      printf("# Construction from 2d block cyclic\n");
    int npcols = (int)floor(sqrt((double)P));
    int nprows = P / npcols;
    MPI_Comm active_comm;
    int in_grid = rank < nprows * npcols;
    MPI_Comm_split(MPI_COMM_WORLD, in_grid, rank, &active_comm);
    if (in_grid) {
      int ctxt = Csys2blacs_handle(active_comm);
      Cblacs_gridinit(&ctxt, "C", nprows, npcols);
      int NB = 32, rproc, cproc, rsrc = 0, csrc = 0;
      Cblacs_gridinfo(ctxt, &nprows, &npcols, &rproc, &cproc);
      int lrows = numroc(&n, &NB, &rproc, &rsrc, &nprows);
      int lcols = numroc(&n, &NB, &cproc, &csrc, &npcols);
      double* T2d = malloc(lrows*lcols*sizeof(double));
      /* fill the 2d block cyclic matrix, Toeplitz example  */
      for (int c=0; c<lcols; c++) {
        int cg = indxl2g(c, NB, cproc, csrc, npcols);
        for (int r=0; r<lrows; r++) {
          int rg = indxl2g(r, NB, rproc, rsrc, nprows);
          T2d[r+c*lrows] = Toeplitz(rg, cg);
        }
      }
      int desc[9], info;
      descinit(desc, &n, &n, &NB, &NB, &rsrc, &csrc, &ctxt, &lrows, &info);
      CSPStructMat H;
      SP_d_struct_from_dense2d
        (&H, active_comm, n, n, T2d, 1, 1, desc, &opts);
      test(H, n, nrhs, rank);
      SP_d_struct_destroy(&H);
    }
  }

  MPI_Finalize();
  return 0;
}


void test(CSPStructMat H, int n, int nrhs, int rank) {
  /* for 1d block row distributed H matrix, get the number of local */
  /* rows, and the first row assigned to this process. */
  int nloc = SP_d_struct_local_rows(H);
  int begin_row = SP_d_struct_begin_row(H);
  /* printf("# local_rows: %d, begin_row: %d\n", nloc, begin_row); */

  /*
   * Allocate 1d block row distributed matrices to check accuracy,
   * local blocks have nloc rows, using nloc as the leading dimension
   */
  double* id = malloc(nloc*n*sizeof(double));
  double* Hdense = malloc(nloc*n*sizeof(double));

  /* set id to identity matrix */
  for (int j=0; j<n; j++)
    for (int i=0; i<nloc; i++)
      id[i+j*nloc] = (i+begin_row == j) ? 1. : 0.;

  if (rank == 0)
    printf("# getting H to dense, Hdense = H*I ...\n");
  SP_d_struct_mult(H, 'N', n, id, nloc, Hdense, nloc);

  if (rank == 0)
    printf("# computing error ...\n");
  double err = 0.;
  double nrm = 0.;
  for (int j=0; j<n; j++)
    for (int i=0; i<nloc; i++) {
      double Tij = Toeplitz(i + begin_row, j);
      err += (Hdense[i+j*nloc] - Tij) *
        (Hdense[i+j*nloc] - Tij);
      nrm += Tij * Tij;
    }

  if (rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &nrm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    printf("||T-H||_F/||T||_F = %e\n", sqrt(err)/sqrt(nrm));
  } else {
    MPI_Reduce(&err, &err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nrm, &nrm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }


  double* B = malloc(nrhs*nloc*sizeof(double));
  double* X = malloc(nrhs*nloc*sizeof(double));
  for (int j=0; j<nrhs; j++)
    for (int i=0; i<nloc; i++)
      X[i + j*nloc] = (double)(rand()) / RAND_MAX;

  /* compute B = H*X */
  SP_d_struct_mult(H, 'N', nrhs, X, nloc, B, nloc);
  /* factor H */
  SP_d_struct_factor(H);
  /* solve H*X=B, overwrite B with X */
  SP_d_struct_solve(H, nrhs, B, nloc);

  err = 0.;
  nrm = 0.;
  for (int j=0; j<nrhs; j++)
    for (int i=0; i<nloc; i++) {
      err += (X[i+j*nloc] - B[i+j*nloc])*(X[i+j*nloc] - B[i+j*nloc]);
      nrm += X[i+j*nloc] * X[i+j*nloc];
    }
  if (rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &nrm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&err, &err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nrm, &nrm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  if (rank == 0)
    printf("||X-H\\(H*X)||_F/||X||_F = %e\n", sqrt(err)/sqrt(nrm));

  free(Hdense);
  free(id);
  /* free(T); */
  free(X);
  free(B);
}
