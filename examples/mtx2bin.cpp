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
#include <iostream>
#include "sparse/CSRMatrix.hpp"

using namespace strumpack;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Read a matrix in matrix market format, and print out binary file csr matrix." << std::endl
	      << "The output format is as follows: [order int_t scalar_t rows cols nnz ptr ind val]." << std::endl
	      << "order is a char:" << std::endl
	      << "\t- 'R' for compressed sparse row" << std::endl
	      << "\t- 'C' for compressed sparse column" << std::endl
	      << "int_t is a char denoting the type of the integers: " << std::endl
	      << "\t- '4' denotes 4 byte integers (signed 32 bit)" << std::endl
	      << "\t- '8' denoted 8 byte integers (signed 64 bit)" << std::endl
	      << "and scalar_t is a char denoting the scalar type" << std::endl
	      << "\t- 's' denotes single precision real" << std::endl
	      << "\t- 'd' denotes double precision complex" << std::endl
	      << "\t- 'c' denotes single precision real" << std::endl
	      << "\t- 'z' denotes double precision complex" << std::endl
	      << "ptr is an int_t array of size rows+1." << std::endl
	      << "ind is an int_t array of size nnz." << std::endl
	      << "val is a scalar_t array of nnz." << std::endl
	      << "Usage: \n\t./mtx2bin pde900.mtx pde900.bin" << std::endl;
    return 1;
  }
  std::string f(argv[1]);

  CSRMatrix<double,int> A;
  if (A.read_matrix_market(f) == 0)
    A.print_binary(argv[2]);
  else {
    CSRMatrix<std::complex<double>,int> A;
    A.read_matrix_market(f);
    A.print_binary(argv[2]);
  }

}
