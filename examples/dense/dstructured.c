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

int main(int argc, char* argv[]) {
  int n = 1000;
  int nrhs = 10;
  if (argc > 1) n = atoi(argv[1]); // get matrix size
  else printf("# no matrix size provided\n");

  double* T = malloc(n*n*sizeof(double));
  for (int j=0; j<n; j++)
    for (int i=0; i<n; i++)
      T[i+j*n] = 1. / (1. + abs(i-j));

  CSPOptions opts;
  SP_d_struct_default_options(&opts);

  CSPStructMat H;
  SP_d_struct_from_dense(&H, n, n, T, n, &opts);


  double* id = malloc(n*n*sizeof(double));
  double* Hdense = malloc(n*n*sizeof(double));

  SP_d_struct_mult(H, 'N', n, id, n, Hdense, n);

  double err = 0.;
  double nrmT = 0.;
  for (int j=0; j<n; j++)
    for (int i=0; i<n; i++) {
      err += (Hdense[i+j*n] - T[i+j*n])*(Hdense[i+j*n] - T[i+j*n]);
      nrmT += T[i+j*n] * T[i+j*n];
    }
  err = sqrt(err);
  nrmT = sqrt(nrmT);
  printf("||T-H||_F/||T||_F = %f\n", err/nrmT);


  SP_d_struct_destroy(&H);
  free(Hdense);
  free(id);
  free(T);

  return 0;
}
