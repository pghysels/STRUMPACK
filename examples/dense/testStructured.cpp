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
#include <cmath>
using namespace std;

#include "structured/StructuredMatrix.hpp"
using namespace strumpack;


template<typename scalar_t> void
print_info(const structured::StructuredMatrix<scalar_t>* H,
           const structured::StructuredOptions<scalar_t>& opts) {
  cout << get_name(opts.type()) << endl;
  cout << "  - nonzeros(H) = " << H->nonzeros() << endl;
  cout << "  - memory(H) = " << H->memory() / 1e6 << " MByte" << endl;
  cout << "  - rank(H) = " << H->rank() << endl;
}

template<typename scalar_t> void
check_accuracy(const DenseMatrix<scalar_t>& A,
               const structured::StructuredMatrix<scalar_t>* H) {
  DenseMatrix<scalar_t> id(A.rows(), A.cols()), Hdense(A.rows(), A.cols());
  id.eye();
  H->mult(Trans::N, id, Hdense);
  cout << "  - ||A-H||_F/||A||_F = "
       << Hdense.sub(A).normF() / A.normF() << endl;
}

template<typename scalar_t> void
factor_and_solve(int nrhs,
                 structured::StructuredMatrix<scalar_t>* H) {
  DenseMatrix<scalar_t> B(H->rows(), nrhs), X(H->rows(), nrhs);
  X.random();
  H->mult(Trans::N, X, B);
  H->factor();
  H->solve(B);
  cout << "  - ||X-H\\(H*X)||_F/||X||_F = "
       << B.sub(X).normF() / X.normF() << endl;
}


int main(int argc, char* argv[]) {
  int n = 1000, nrhs = 10;
  if (argc > 1) n = stoi(argv[1]);

  structured::StructuredOptions<double> options;
  options.set_verbose(false);
  options.set_from_command_line(argc, argv);


  // Routine to compute individual elements of the matrix.
  auto Toeplitz =
    [](int i, int j) {
      return 1. / (1. + abs(i-j));
    };

  // routine to compute sub-block of the matrix. Often this could be
  // implemented more efficiently than computing element per element.
  auto Toeplitz_block =
    [&Toeplitz](const std::vector<std::size_t>& I,
                const std::vector<std::size_t>& J,
                DenseMatrix<double>& B) {
      for (std::size_t j=0; j<J.size(); j++)
        for (std::size_t i=0; i<I.size(); i++)
          B(i, j) = Toeplitz(I[i], J[j]);
    };

  // construct a dense representation of the Toeplitz matrix (for
  // illustration purpose, this can be avoided to save memory)
  DenseMatrix<double> A(n, n, Toeplitz);

  // Matrix-vector multiplication routine. Ideally, user can provide a
  // faster implementation.
  auto Amult =
    [&A](Trans t, const DenseMatrix<double>& R,
         DenseMatrix<double>& S) {
      gemm(t, Trans::N, double(1.), A, R, double(0.), S);
    };

  // construct a balanced ClusterTree. If the user has for instance
  // spatial coordinates, then a better tree can be constructed. See
  // the clustering algorithms in clustering/Clustering.hpp
  structured::ClusterTree tree(n);
  tree.refine(options.leaf_size());


  std::vector<structured::Type> types =
    {structured::Type::BLR,
     structured::Type::HSS,
     structured::Type::LOSSY,
     structured::Type::LOSSLESS
    };
  // the HODLR, HODBF, Butterfly and LR types require MPI support, see
  // testStructuredMPI


  cout << "dense " << A.rows() << " x " << A.cols() << " matrix" << endl;
  cout << "  - memory(A) = " << A.memory() / 1e6 << " MByte"
       << endl << endl;

  cout << "===============================" << endl;
  cout << " Compression from dense matrix" << endl;
  cout << "===============================" << endl;
  for (auto type : types) {
    options.set_type(type);
    try {
      {
        // construct a StructuredMatrix from a dense matrix
        auto H = structured::construct_from_dense(A, options);
        print_info(H.get(), options);
        check_accuracy(A, H.get());
        factor_and_solve(nrhs, H.get());
      }
      {
        // user can define the ClusterTree
        auto H = structured::construct_from_dense(A, options, &tree);
        print_info(H.get(), options);
        check_accuracy(A, H.get());
        factor_and_solve(nrhs, H.get());
      }
    } catch (std::exception& e) {
      cout << get_name(type) << " failed: "
           << e.what() << endl;
    }
  }


  cout << endl << endl;
  cout << "==================================" << endl;
  cout << " Compression from matrix elements" << endl;
  cout << "==================================" << endl;
  for (auto type : types) {
    options.set_type(type);
    try {
      {
        // construct from individual elements
        auto H = structured::construct_from_elements<double>
          (n, n, Toeplitz, options);
        print_info(H.get(), options);
        check_accuracy(A, H.get());
        factor_and_solve(nrhs, H.get());
      }
      {
        // constrcut from sub-blocks (could have better performance)
        auto H = structured::construct_from_elements<double>
          (n, n, Toeplitz_block, options);
        print_info(H.get(), options);
        check_accuracy(A, H.get());
        factor_and_solve(nrhs, H.get());
      }
    } catch (std::exception& e) {
      cout << get_name(type) << " failed: "
           << e.what() << endl;
    }
  }


  cout << endl << endl;
  cout << "====================================" << endl;
  cout << " Compression, partially matrix-free " << endl;
  cout << "====================================" << endl;
  for (auto type : types) {
    options.set_type(type);
    try {

      auto H = structured::construct_partially_matrix_free<double>
        (n, n, Amult, Toeplitz, options);
      print_info(H.get(), options);
      check_accuracy(A, H.get());
      factor_and_solve(nrhs, H.get());

    } catch (std::exception& e) {
      cout << get_name(type) << " failed: "
           << e.what() << endl;
    }
  }

  return 0;
}
