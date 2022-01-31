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
#include "mex.h"
#include "matrix.h"

#include "StrumpackSparseSolver.hpp"

// Aliases for input and output arguments
#define A_in  prhs[0]
#define b_in  prhs[1]
#define x_out plhs[0]

#define verbose (SPUMONI>0)
#define babble  (SPUMONI>1)
#define burble  (SPUMONI>2)

void mexFunction
(int nlhs,               // number of expected outputs
 mxArray *plhs[],        // matrix pointer array returning outputs
 int nrhs,               // number of inputs
 const mxArray *prhs[]   // matrix pointer array for inputs
 ) {
  // Check number of arguments passed from Matlab
  if (nrhs != 2)
    mexErrMsgTxt("STRUMPACK SOLVE requires 2 input arguments.");
  else if (nlhs != 1)
    mexErrMsgTxt("STRUMPACK SOLVE requires 1 output argument.");

  // Read the Sparse Monitor Flag
  mxArray *Y, *X = mxCreateString("spumoni");
  int mexerr = mexCallMATLAB(1, &Y, 1, &X, "sparsfun");
  int SPUMONI = mxGetScalar(Y);
  mxDestroyArray(Y);
  mxDestroyArray(X);

  int m = mxGetM(A_in), n = mxGetN(A_in), numrhs = mxGetN(b_in);
  if (babble)
    std::cout << "m= " << m << ", n=" << n
	      << ", numrhs=" << numrhs << std::endl;

  x_out = mxCreateDoubleMatrix(m, numrhs, mxREAL);

  if (verbose)
    mexPrintf("Call STRUMPACK SOLVE, use STRUMPACK to factor first ...\n");
  // strumpack::StrumpackSparseSolver<mxDouble,mwIndex> sp;
  // sp.set_csr_matrix(m, mxGetJc(A_in), mxGetIr(A_in), mxGetDoubles(A_in));
  // if (sp.solve(mxGetDoubles(b_in), mxGetDoubles(x_out))
  strumpack::StrumpackSparseSolver<double,mwIndex> sp(verbose);
  sp.set_csr_matrix(m, mxGetJc(A_in), mxGetIr(A_in), mxGetPr(A_in));
  if (sp.solve(mxGetPr(b_in), mxGetPr(x_out))
      != strumpack::ReturnCode::SUCCESS) {
    std::cout << "Error during matrix factorization." << std::endl;
    mexErrMsgTxt("Error during matrix factorization.\n");
  }
}
