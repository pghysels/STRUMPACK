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
#include <cstring>
using namespace std;

#include "sparse/CSRMatrix.hpp"

using namespace strumpack;

#define ERROR_TOLERANCE 1e2
#define SOLVE_TOLERANCE 1e-12


template<typename real_t,typename integer_t, typename cast_t> 
int check_matrix_equality
(CSRMatrix<real_t,integer_t> orig, CSRMatrix<cast_t,integer_t> cast) {
    cout << "# Casting succeeded, testing matrix equality.\n";
    assert(orig.size() == cast.size());
    assert(orig.nnz() == cast.nnz());
    assert(orig.symm_sparse() == cast.symm_sparse());

    const integer_t nnz = orig.nnz();
    const integer_t size = orig.size();
    for (int i = 0; i <= size; ++i) {
      assert(orig.ptr(i) == cast.ptr(i));
    }
    for (int i = 0; i < nnz; ++i) {
      assert(orig.ind(i) == cast.ind(i));
      assert(cast.val(i) == static_cast<cast_t>(orig.val(i)));
    }
    cout << "# Matrix equality passed.\n";
    return 0;
}

template<typename real_t,typename integer_t, typename cast_t>
int read_matrix_and_run_tests(int argc, const char* const argv[]) {
  string f(argv[1]);
  CSRMatrix<real_t,integer_t> A;
  if (A.read_matrix_market(f) != 0)
    return 1;
  CSRMatrix<cast_t,integer_t> new_mat = cast_matrix<real_t,integer_t,cast_t>(A);
  return check_matrix_equality<real_t,integer_t,cast_t>(A, new_mat);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cout
      << "Solve a linear system with a matrix given in matrix market format\n"
      << "using the sequential/multithreaded C++ STRUMPACK interface.\n\n"
      << "Usage: \n\t./test_sparse_seq_mixed pde900.mtx" << endl;
    return 1;
  }
  cout << "# Running with:\n# ";
  for (int i=0; i<argc; i++)
    cout << argv[i] << " ";
  cout << endl;

  int ierr = 0;
  ierr = read_matrix_and_run_tests<double,int,float>(argc, argv);
  if (ierr) return ierr;
}
