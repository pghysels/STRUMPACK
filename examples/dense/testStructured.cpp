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
#include "iterative/IterativeSolvers.hpp"
using namespace strumpack;


/**
 * This takes a pointer to a structured::StructureMatrix and prints
 * out the memory usage, and the maximum rank.
 */
template<typename scalar_t> void
print_info(const structured::StructuredMatrix<scalar_t>* H,
           const structured::StructuredOptions<scalar_t>& opts) {
  cout << get_name(opts.type()) << endl;
  cout << "  - nonzeros(H) = " << H->nonzeros() << endl;
  cout << "  - memory(H) = " << H->memory() / 1e6 << " MByte" << endl;
  cout << "  - rank(H) = " << H->rank() << endl;
}


/**
 * Check the accuracy of a structured::StructuredMatrix by comparing
 * it with a DenseMatrix. This should only be used on a relatively
 * small matrix, as the storage of the DenseMatrix will become a
 * bottleneck for larger problems.
 */
template<typename scalar_t> void
check_accuracy(const DenseMatrix<scalar_t>& A,
               const structured::StructuredMatrix<scalar_t>* H) {
  // allocate 2 A.rows x A.cols matrices
  DenseMatrix<scalar_t> id(A.rows(), A.cols()),
    Hdense(A.rows(), A.cols());
  // set id to the identity matrix
  id.eye();
  // compute dense representation of H as H*I
  H->mult(Trans::N, id, Hdense);
  // compute relative Frobenius norm of the compression error
  cout << "  - ||A-H||_F/||A||_F = "
       << Hdense.sub(A).normF() / A.normF() << endl;
}


/**
 * Factor a structured::StructuredMatrix and solve a linear system
 * with nrhs right-hand side vectors.
 */
template<typename scalar_t> void
factor_and_solve(int nrhs,
                 const DenseMatrix<scalar_t>& A,
                 structured::StructuredMatrix<scalar_t>* H) {
  // Allocate memory for rhs and solution vectors (matrices)
  DenseMatrix<scalar_t> B(H->rows(), nrhs), X(H->rows(), nrhs);
  // Pick a random exact solution
  X.random();
  // Compute the right-hand side B as B=H*X
  H->mult(Trans::N, X, B);
  // Compute a factorization of H. The factors are stored in H.
  H->factor();
  // Solve a linear system H*X=B, input is the right-hand side B,
  // which will be overwritten with the solution X.
  H->solve(B);
  // Compute the relative error on the solution.
  cout << "  - ||X-H\\(H*X)||_F/||X||_F = "
       << B.sub(X).normF() / X.normF() << endl;


  // Same as above, but now we compute the right-hand side as B=A*X
  // (instead of B=H*X)
  gemm(Trans::N, Trans::N, scalar_t(1.), A, X, scalar_t(0.), B);
  // H->factor(); // already called
  // solve a linear system H*X=B, input is the right-hand side B,
  // which will be overwritten with the solution X.
  H->solve(B);
  // Compute the relative error on the solution. This now includes the
  // compression error.
  cout << "  - ||X-H\\(A*X)||_F/||X||_F = "
       << B.sub(X).normF() / X.normF() << endl;
}


/**
 * Use the structured::StructuredMatrix as a preconditioner in an
 * iterative solver. Here we use the dense matrix for the exact
 * matrix-vector product. However, the user can provide a faster
 * algorithm (f.i. using a sparse matrix, a fast transformation, ..)
 * This works only for a single right-hans side.
 */
template<typename scalar_t> void
preconditioned_solve(const DenseMatrix<scalar_t>& A,
                     structured::StructuredMatrix<scalar_t>* H) {

  // Preconditioned solves only work for a single right-hand side

  int nrhs = 1, n = A.rows();
  // Allocate memory for rhs and solution vectors
  DenseMatrix<scalar_t> B(n, nrhs), X(n, nrhs);
  // Pick a random exact solution
  X.random();
  // Keep a copy of the exact solution X
  DenseMatrix<scalar_t> Xexact(X);
  // compute the right-hand side vector B as B = A*X
  gemm(Trans::N, Trans::N, scalar_t(1.), A, X, scalar_t(0.), B);

  // factor the structured matrix, so it can be used as a preconditioner
  // H->factor();  // was already called

  int iterations = 0, maxit = 50, restart = 50;
  iterative::GMRes<scalar_t>
    ([&A](const scalar_t* v, scalar_t* w) {
       // Compute a matrix-vector product with the exact
       // (non-compressed) matrix.  The user might have a fast
       // matrix-vector product instead of this dense matrix
       // multiplication.
       gemv(Trans::N, scalar_t(1.), A, v, 1, scalar_t(0.), w, 1);
     },
      [&H, &n](scalar_t* v) {
        // Apply the preconditioner H: solve a linear system H*w=v,
        // with 1 right-hand side vector v, with leading dimension n
        H->solve(1, v, n);
      },
      n, X.data(), B.data(),  // matrix size, solution, right-hand side
      1e-10, 1e-14,           // rtol, atol
      iterations, maxit,      // iterations (output), maximum nr of iterations
      restart,                // GMRes restart
      GramSchmidtType::CLASSICAL,
      false, true);           // use initial guess, verbose

  // Compute the relative error
  cout << "  - ||X-A\\(A*X)||_F/||X||_F = "
       << X.sub(Xexact).normF() / Xexact.normF() << endl;


  iterative::BiCGStab<scalar_t>
    ([&A](const scalar_t* v, scalar_t* w) {
       // Compute a matrix-vector product with the exact
       // (non-compressed) matrix.  The user might have a fast
       // matrix-vector product instead of this dense matrix
       // multiplication.
       gemv(Trans::N, scalar_t(1.), A, v, 1, scalar_t(0.), w, 1);
     },
      [&H, &n](scalar_t* v) {
        // Apply the preconditioner H: solve a linear system H*w=v,
        // with 1 right-hand side vector v, with leading dimension n
        H->solve(1, v, n);
      },
      n, X.data(), B.data(),  // matrix size, solution, right-hand side
      1e-10, 1e-14,           // rtol, atol
      iterations, maxit,      // iterations (output), maximum nr of iterations
      false, true);           // use initial guess, verbose

  // Compute the relative error
  cout << "  - ||X-A\\(A*X)||_F/||X||_F = "
       << X.sub(Xexact).normF() / Xexact.normF() << endl;
}


template<typename scalar_t> void
test_shift(int nrhs,
           const DenseMatrix<scalar_t>& A,
           structured::StructuredMatrix<scalar_t>* H) {
  cout << "  - Adding diagonal shift" << endl;
  auto As = A;
  scalar_t sigma(10.);
  As.shift(sigma);
  H->shift(sigma);
  // check the shifted matrix
  check_accuracy(As, H);
  // after applying the shift, H->factor needs to be called again!
  factor_and_solve(nrhs, As, H);
}


int main(int argc, char* argv[]) {
  // matrix size and number of right-hand sides
  int n = 1000, nrhs = 10;
  if (argc > 1) n = stoi(argv[1]);

  // Define an options object, set to the default options.
  structured::StructuredOptions<double> options;
  // Suppress some output
  options.set_verbose(false);
  // Parse options passed on the command line, run with --help to see
  // more.
  options.set_from_command_line(argc, argv);


  // Define the matrix through a routine to compute individual
  // elements of the matrix. This is compatible with the
  // structured::extract_t type, which is:
  // std::function<scalar_t(std::size_t,std::size_t)>
  auto Toeplitz =
    [](int i, int j) {
      return 1. / (1. + abs(i-j));
    };


  // Construct a dense representation of the Toeplitz matrix. This is
  // mainly for illustration purposes, and to check the accuracy of
  // the final compressed matrix. In practice it can be avoided to
  // build the entire matrix as dense.
  DenseMatrix<double> A(n, n, Toeplitz);

  // Define a matrix-vector multiplication routine. In this case we
  // simply call the optimized gemm (dense matrix-matrix
  // multiplication) algorithm. Ideally, the user would provide a
  // faster algorithm.
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


  // In the tests below, we try all the following StructuredMatrix
  // Types. In practice, you pick one by for instance setting:
  //    options.set_type(structured::Type::BLR)
  std::vector<structured::Type> types =
    {structured::Type::BLR,
     structured::Type::HSS,
     structured::Type::LOSSY,
     structured::Type::LOSSLESS
    };
  // the HODLR, HODBF, Butterfly and LR types require MPI support, see
  // testStructuredMPI


  // Print how much memory the dense matrix representation takes.
  cout << "dense " << A.rows() << " x " << A.cols() << " matrix" << endl;
  cout << "  - memory(A) = " << A.memory() / 1e6 << " MByte"
       << endl << endl;

  cout << "===============================" << endl;
  cout << " Compression from dense matrix" << endl;
  cout << "===============================" << endl;
  // loop over all structured::Type
  for (auto type : types) {
    options.set_type(type);
    try {
      {
        // Construct a StructuredMatrix from a dense matrix and given
        // options.
        auto H = structured::construct_from_dense(A, options);
        // Print the memory usage etc for H
        print_info(H.get(), options);
        // Check the compression accuracy by comparing with the dense
        // matrix
        check_accuracy(A, H.get());
        // Factor H and (approximately) solve a linear system
        factor_and_solve(nrhs, A, H.get());
        // Solve a linear system using an iterative solver with H as
        // the preconditioner and using A as the exact matrix vector
        // product.
        preconditioned_solve(A, H.get());
        // add a diagonal shift to the structured matrix, this does
        // not require recompression. Then check the accuracy again,
        // and solve a linear system with the shifted matrix.
        test_shift(nrhs, A, H.get());
      }
      {
        // exactly the same as above, but now with an additional
        // optional argument tree, the ClusterTree, see also
        // strumpack::binary_tree_clustering
        auto H = structured::construct_from_dense(A, options, &tree);
        print_info(H.get(), options);
        check_accuracy(A, H.get());
        factor_and_solve(nrhs, A, H.get());
        preconditioned_solve(A, H.get());
        test_shift(nrhs, A, H.get());
      }
    } catch (std::exception& e) {
      cout << get_name(type) << " failed: " << e.what() << endl;
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
        // Construct a structured::StructuredMatrix from individual
        // elements. A ClusterTree for the rows (and columns) can also
        // be provided.
        auto H = structured::construct_from_elements<double>
          (n, n, Toeplitz, options);
        print_info(H.get(), options);
        check_accuracy(A, H.get());
        factor_and_solve(nrhs, A, H.get());
        preconditioned_solve(A, H.get());
        test_shift(nrhs, A, H.get());
      }
      {
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

        // Construct a StructuredMatrix from matrix sub-blocks (could
        // have better performance) instead of individual elements.
        auto H = structured::construct_from_elements<double>
          (n, n, Toeplitz_block, options);
        print_info(H.get(), options);
        check_accuracy(A, H.get());
        factor_and_solve(nrhs, A, H.get());
        preconditioned_solve(A, H.get());
        test_shift(nrhs, A, H.get());
      }
    } catch (std::exception& e) {
      cout << get_name(type) << " failed: " << e.what() << endl;
    }
  }


  cout << endl << endl;
  cout << "====================================" << endl;
  cout << " Compression, partially matrix-free " << endl;
  cout << "====================================" << endl;
  for (auto type : types) {
    options.set_type(type);
    try {
      // Construct a StructuredMatrix using both a (fast)
      // matrix-vector multiplication and an element extraction
      // routine. This is mainly usefull for HSS construction, which
      // requires element extraction for the diagonal blocks and
      // random projection with the matrix-vector multiplication for
      // the off-diagonal compression.
      auto H = structured::construct_partially_matrix_free<double>
        (n, n, Amult, Toeplitz, options);
      print_info(H.get(), options);
      check_accuracy(A, H.get());
      factor_and_solve(nrhs, A, H.get());
      preconditioned_solve(A, H.get());
      test_shift(nrhs, A, H.get());
    } catch (std::exception& e) {
      cout << get_name(type) << " failed: " << e.what() << endl;
    }
  }

  return 0;
}
