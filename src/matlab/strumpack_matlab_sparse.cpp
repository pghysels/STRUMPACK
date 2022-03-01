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

#define verbose (SPUMONI>0)
#define babble  (SPUMONI>1)
#define burble  (SPUMONI>2)

void mexFunction
(int nlhs,               // number of expected outputs
 mxArray *plhs[],        // matrix pointer array returning outputs
 int nrhs,               // number of inputs
 const mxArray *prhs[]   // matrix pointer array for inputs
 ) {
  static std::unique_ptr<strumpack::StrumpackSparseSolver<double,std::int64_t>> sp;
  if (!sp) {
    if (nrhs != 2)
      mexErrMsgTxt("STRUMPACK SOLVE requires 2 input arguments on first call: sparse matrix and rhs vector.");
  } else if (nrhs != 1 && nrhs != 2)
    mexErrMsgTxt("STRUMPACK SOLVE requires 1 or 2 input arguments: sparse matrix and rhs vector, or rhs vector (for subsequent solve with same matrix)");
  if (nlhs != 1)
    mexErrMsgTxt("STRUMPACK SOLVE requires 1 output argument.");

  // Read the Sparse Monitor Flag
  mxArray *Y, *X = mxCreateString("spumoni");
  int mexerr = mexCallMATLAB(1, &Y, 1, &X, "spparms");
  int SPUMONI = mxGetScalar(Y);
  mxDestroyArray(Y);
  mxDestroyArray(X);

  const mxArray* b_in = (nrhs == 1) ? prhs[0] : prhs[1];
  int m = mxGetM(b_in), numrhs = mxGetN(b_in);

  mxArray*& x_out = plhs[0];
  x_out = mxCreateDoubleMatrix(m, numrhs, mxREAL);
  if (babble)
    std::cout << "m= " << m
              << ", numrhs=" << numrhs << std::endl;

  if (verbose)
    mexPrintf("Call STRUMPACK SOLVE, use STRUMPACK to factor first ...\n");

  if (nrhs == 2) {
    const mxArray* A_in = prhs[0];

    // the input matrix uses UNSIGNED integers,
    // strumpack needs SIGNED integers
    mwIndex *m_rowind = mxGetIr(A_in);
    mwIndex *m_colptr = mxGetJc(A_in);
    auto nnz = m_colptr[m];

    // the input matrix is compressed sparse COLUMN,
    // strumpack needs compressed sparse ROW
    std::int64_t *rowptr = (std::int64_t*)mxMalloc((m+1)*sizeof(std::int64_t));
    std::int64_t *colind = (std::int64_t*)mxMalloc(nnz*sizeof(std::int64_t));
    std::int64_t *rowsums = (std::int64_t*)mxMalloc(m*sizeof(std::int64_t));
    std::fill(rowsums, rowsums+m, 0);
    for (mwIndex c=0; c<m; c++)
      for (mwIndex i=m_colptr[c]; i<m_colptr[c+1]; i++)
        rowsums[m_rowind[i]]++;
    rowptr[0] = 0;
    for (std::int64_t r=0; r<m; r++)
      rowptr[r+1] = rowptr[r] + rowsums[r];
    std::fill(rowsums, rowsums+m, 0);
    for (mwIndex c=0; c<m; c++)
      for (mwIndex i=m_colptr[c]; i<m_colptr[c+1]; i++) {
        mwIndex r = m_rowind[i];
        colind[rowptr[r]+rowsums[r]] = c;
        rowsums[r]++;
      }
    sp.reset(new strumpack::StrumpackSparseSolver<double,std::int64_t>(verbose));
    sp->options().set_reordering_method(strumpack::ReorderingStrategy::SCOTCH);
    sp->set_csr_matrix(m, rowptr, colind, mxGetPr(A_in));
    sp->options().set_compression(strumpack::CompressionType::LOSSY);
    sp->options().set_lossy_precision(10);
    sp->options().set_Krylov_solver(strumpack::KrylovSolver::DIRECT);
    // sp->options().set_matching(strumpack::MatchingJob::NONE);
    if (sp->factor() != strumpack::ReturnCode::SUCCESS) {
      std::cout << "Error during matrix factorization." << std::endl;
      mexErrMsgTxt("Error during matrix factorization.\n");
    }
  }

  if (sp->solve(numrhs, mxGetPr(b_in), m, mxGetPr(x_out), m)
      != strumpack::ReturnCode::SUCCESS) {
    std::cout << "Error during triangular solve." << std::endl;
    mexErrMsgTxt("Error during triangular solve.\n");
  }
}
