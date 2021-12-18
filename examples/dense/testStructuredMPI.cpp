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
#include "iterative/IterativeSolversMPI.hpp"
using namespace strumpack;


/**
 * This takes a pointer to a structured::StructureMatrix and prints
 * out the memory usage, and the maximum rank.
 *
 * MPIComm is a c++ wrapper around an MPI_Comm
 */
template<typename scalar_t> void
print_info(const MPIComm& comm,
           const structured::StructuredMatrix<scalar_t>* H,
           const structured::StructuredOptions<scalar_t>& opts) {
  if (comm.is_root())
    cout << get_name(opts.type()) << endl
         << "  - total_nonzeros(H) = " << H->nonzeros() << endl
         << "  - total_memory(H) = " << H->memory() / 1e6 << " MByte" << endl
         << "  - maximum_rank(H) = " << H->rank() << endl;
}

/**
 * Check the accuracy of a structured::StructuredMatrix by comparing
 * it with a DenseMatrix. This should only be used on a relatively
 * small matrix, as the storage of the DenseMatrix will become a
 * bottleneck for larger problems.
 */
template<typename scalar_t> void
check_accuracy(const DistributedMatrix<scalar_t>& A,
               const structured::StructuredMatrix<scalar_t>* H) {
  // Allocate 2 A.rows x A.cols matrices, use the same 2D processor
  // grid as A
  DistributedMatrix<scalar_t> id(A.grid(), A.rows(), A.cols()),
    Hdense(A.grid(), A.rows(), A.cols());
  // set id to the identity matrix
  id.eye();
  // compute dense representation of H as H*I
  H->mult(Trans::N, id, Hdense);
  // compute relative Frobenius norm of the compression error
  auto err = Hdense.scaled_add(scalar_t(-1.), A).normF() / A.normF();
  if (A.Comm().is_root())
    cout << "  - ||A-H||_F/||A||_F = " << err << endl;
}


/**
 * Factor a structured::StructuredMatrix and solve a linear system
 * with nrhs right-hand side vectors.
 */
template<typename scalar_t> void
factor_and_solve(int nrhs,
                 const DistributedMatrix<scalar_t>& A,
                 structured::StructuredMatrix<scalar_t>* H) {
  // Allocate memory for rhs and solution vectors (matrices). Use same
  // 2D processor grid as A.
  DistributedMatrix<scalar_t> B(A.grid(), H->rows(), nrhs),
    X(A.grid(), H->rows(), nrhs);
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
  auto err = B.scaled_add(scalar_t(-1.), X).normF() / X.normF();
  if (A.Comm().is_root())
    cout << "  - ||X-H\\(H*X)||_F/||X||_F = "
         << err << endl;

  // Same as above, but now we compute the right-hand side as B=A*X
  // (instead of B=H*X)
  gemm(Trans::N, Trans::N, scalar_t(1.), A, X, scalar_t(0.), B);
  // H->factor(); // already called
  // solve a linear system H*X=B, input is the right-hand side B,
  // which will be overwritten with the solution X.
  H->solve(B);
  // Compute the relative error on the solution. This now includes the
  // compression error.
  err = B.scaled_add(scalar_t(-1.), X).normF() / X.normF();
  if (A.Comm().is_root())
    cout << "  - ||X-A\\(A*X)||_F/||X||_F = "
         << err << endl;
}


/**
 * Use the structured::StructuredMatrix as a preconditioner in an
 * iterative solver. The preconditioned iterative solvers use a 1d
 * block row distribution, so here we use the user provided
 * matrix-vector product using the 1d block row distribution. Note
 * that not all structured::StructuredMatrix types use the 2d block
 * row distribution. This works only for a single right-hand side.
 */
template<typename scalar_t> void
preconditioned_solve(const MPIComm& comm,
                     const structured::mult_1d_t<scalar_t>& Amult1d,
                     structured::StructuredMatrix<scalar_t>* H) {
  // Preconditioned solves only work for a single right-hand side
  int nrhs = 1, n = H->rows();
  // Preconditioned solves work using a 1d block row matrix and vector
  // distribution. nloc is the number of rows owned by this process.
  int nloc = H->local_rows();

  // Allocate memory for local part of the rhs and solution vectors.
  DenseMatrix<scalar_t> B(nloc, nrhs), X(nloc, nrhs);

  // Pick a random exact solution
  X.random();
  // Keep a copy of the exact solution X
  auto Xexact = X;

  // compute the right-hand side vector B as B = A*X, using the 1d
  // block row matrix-vector product Amult1d, specified by the user.
  Amult1d(Trans::N, X, B, H->rdist(), H->cdist());

  // factor the structured matrix, so it can be used as a preconditioner
  // H->factor();  // was already called

  int iterations = 0, maxit = 50, restart = 50;
  iterative::GMResMPI<scalar_t>
    (comm,
     [&Amult1d, &H](const DenseMatrix<scalar_t>& v,
                    DenseMatrix<scalar_t>& w) {
       // matrix-vector product using 1d block row distrribution
       Amult1d(Trans::N, v, w, H->rdist(), H->cdist());
     },
     [&H](DenseMatrix<scalar_t>& v) {
       // Apply the preconditioner H: solve a linear system H*w=v.
       H->solve(v);
     },
     X, B,                    // solution (output), right-hand side
     1e-10, 1e-14,            // rtol, atol
     iterations, maxit,       // iterations (output), maximum iterations
     restart,                 // GMRes restart
     GramSchmidtType::CLASSICAL,
     false, comm.is_root());  // initial guess, verbose

  // Compute the relative error (needs global reduction with all
  // processes)
  auto err = comm.reduce(X.sub(Xexact).normF(), MPI_SUM)
    / comm.reduce(Xexact.normF(), MPI_SUM);
  if (comm.is_root())
    cout << "  - ||X-A\\(A*X)||_F/||X||_F = " << err << endl;


  iterative::BiCGStabMPI<scalar_t>
    (comm,
     [&Amult1d, &H](const DenseMatrix<scalar_t>& v,
                    DenseMatrix<scalar_t>& w) {
       // matrix-vector product with exact matrix
       Amult1d(Trans::N, v, w, H->rdist(), H->cdist());
     },
     [&H](DenseMatrix<scalar_t>& v) {
       // preconditioning with structured matrix
       H->solve(v);
     },
     X, B,                    // solution (output), right-hand side
     1e-10, 1e-14,            // rtol, atol
     iterations, maxit,       // iterations (output), maximum iterations
     false, comm.is_root());  // initial guess, verbose

  // Compute the relative error (needs global reduction with all
  // processes)
  err = comm.reduce(X.sub(Xexact).normF(), MPI_SUM)
    / comm.reduce(Xexact.normF(), MPI_SUM);
  if (comm.is_root())
    cout << "  - ||X-A\\(A*X)||_F/||X||_F = " << err << endl;
}



/**
 * Use the structured::StructuredMatrix with an iterative refinement
 * solver.  his version of the iterative refinement solver uses a 2d
 * block cyclic distribution for the vectors, so here we use the user
 * provided matrix-vector product using the 2d block cyclic
 * distribution. This works for multiple right-hand sides.
 */
template<typename scalar_t> void
iterative_refinement_2d(const MPIComm& comm, const BLACSGrid* g,
                        const structured::mult_2d_t<scalar_t>& Amult2d,
                        int nrhs, structured::StructuredMatrix<scalar_t>* H) {
  // Allocate memory for the rhs and solution vectors, in 2d block
  // cyclic format.
  DistributedMatrix<scalar_t> B(g, H->rows(), nrhs),
    X(g, H->rows(), nrhs);

  // Pick a random exact solution
  X.random();
  // Keep a copy of the exact solution X
  auto Xexact = X;

  // compute the right-hand side vector B as B = A*X, using the 2d
  // block cyclic matrix-vector product Amult2d, specified by the
  // user.
  Amult2d(Trans::N, X, B);

  // factor the structured matrix, so it can be used as a preconditioner
  // H->factor();  // was already called

  int iterations = 0, maxit = 50;
  iterative::IterativeRefinementMPI<scalar_t>
    (comm,
     [&Amult2d](const DistributedMatrix<scalar_t>& V,
                DistributedMatrix<scalar_t>& W) -> void {
       // matrix-vector product using 1d block row distrribution
       Amult2d(Trans::N, V, W);
     },
     [&H](DistributedMatrix<scalar_t>& V) {
       // Apply the preconditioner H: solve a linear system H*W=V,
       // overwriting W in V, using 2d block cyclic layout for V and
       // W.
       H->solve(V);
     },
     X, B,                    // solution (output), right-hand side
     1e-10, 1e-14,            // rtol, atol
     iterations, maxit,       // iterations (output), maximum iterations
     false, comm.is_root());  // initial guess, verbose

  // Compute the relative error (needs global reduction with all
  // processes)
  auto err = X.scaled_add(-1., Xexact).norm() / Xexact.norm();
  if (comm.is_root())
    cout << "  - ||X-A\\(A*X)||_F/||X||_F = " << err << endl;
}


template<typename scalar_t> void
test_shift(int nrhs, const DistributedMatrix<scalar_t>& A,
           structured::StructuredMatrix<scalar_t>* H) {
  if (A.Comm().is_root())
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
  int thread_level;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
  {
    // C++ wrapper around an MPI_Comm, defaults to MPI_COMM_WORLD
    MPIComm world;
    // we need at least MPI_THREADS_FUNNELED support
    if (thread_level != MPI_THREAD_FUNNELED && world.is_root())
      cout << "MPI implementation does not support MPI_THREAD_FUNNELED" << endl;

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

    // Create a 2D processor grid, as used in ScaLAPACK. This is only
    // needed if you will use 2d block-cyclic input/output.
    BLACSGrid grid(world);


    // Define the matrix through a routine to compute individual
    // elements of the matrix. This is compatible with the
    // structured::extract_t type, which is:
    // std::function<scalar_t(std::size_t,std::size_t)>
    auto Toeplitz =
      [](int i, int j) {
        return 1. / (1. + abs(i-j));
      };


    // Create 2d block cyclicly distributed matrix, and initialize it
    // as a Toeplitz matrix. This step can be avoided. Construction of
    // the StructuredMatrix can be done using a routine to compute
    // individual elements and/or using a black box matrix-vector
    // multiplication routine. Here we construct the 2D block cyclic
    // representation explicitly mainly for illustration and to test
    // the compression accuracy after compression.
    DistributedMatrix<double> A2d(&grid, n, n, Toeplitz);


    // Define a matrix-vector multiplication routine, using the 2d
    // block cyclic distribution. Ideally, the user can provide a
    // faster implementation. Instead of a 2D block cyclic
    // distribution, one can also use a 1d block row distribution, see
    // below.
    auto Tmult2d =
      [&A2d](Trans t,
             const DistributedMatrix<double>& R,
             DistributedMatrix<double>& S) {
        // simply call PxGEMM using A2d
        // gemm(t, Trans::N, double(1.), A2d, R, double(0.), S);
        A2d.mult(t, R, S); // same as gemm above
      };


    // Matrix-vector multiplication routine using 1d block row
    // distribution. Ideally, user can provide a faster
    // implementation. The block row/column distribution of the matrix
    // is given by the rdist and cdist vectors, with processor p
    // owning rows [rdist[p],rdist[p+1]). cdist is only needed for
    // non-square matrices. S is distributed according to rdist if
    // t==Trans::N, else cdist. R is distributed according to cdist of
    // t==Trans::N, else rdist.
    structured::mult_1d_t<double> Tmult1d =
      [&n, &Toeplitz, &world](Trans t,
                              const DenseMatrix<double>& R,
                              DenseMatrix<double>& S,
                              const std::vector<int>& rdist,
                              const std::vector<int>& cdist) {
        // this broadcasts each piece of the P (number of processes)
        // pieces of R to all processes. Then every process forms a
        // block of the Toeplitz matrix and multiplies that with the
        // current piece of R.
        int rank = world.rank(), P = world.size(),
          r0 = rdist[rank], r1 = rdist[rank+1];
        S.zero();
        for (int p=0; p<P; p++) {
          DenseMatrix<double> Rp(cdist[p+1]-cdist[p], R.cols()),
            Tp(r1 - r0, cdist[p+1]-cdist[p]);
          if (rank == p) Rp.copy(R);
          world.broadcast_from(Rp.data(), Rp.rows()*Rp.cols(), p);
          Tp.fill([&](std::size_t i, std::size_t j) -> double {
                    return Toeplitz(r0+i, cdist[p]+j);
                  });
          gemm(Trans::N, Trans::N, 1., Tp, Rp, 1., S);
        }
      };


    // In the tests below, we try all the following StructuredMatrix
    // Types. In practice, you pick one by for instance setting:
    //    options.set_type(structured::Type::BLR)
    std::vector<structured::Type> types =
      {structured::Type::BLR,
       structured::Type::HSS,
       structured::Type::HODLR,
       structured::Type::HODBF,
       structured::Type::BUTTERFLY,
       structured::Type::LR};
    // The LOSSY and LOSSLESS types do not support MPI
    // structured::Type::LOSSY, structured::Type::LOSSLESS};


    // Print how much memory the dense (2D block cyclic) matrix
    // representation takes (for comparison).
    if (world.is_root())
      cout << endl << endl
           << "dense (2DBC) " << A2d.rows() << " x " << A2d.cols()
           << " matrix" << endl
           << "  - memory(A2d) = " << A2d.memory() / 1e6 << " MByte"
           << endl << endl;

    if (world.is_root())
      cout << "===============================" << endl
           << " Compression from dense matrix" << endl
           << "===============================" << endl;
    for (auto type : types) {
      options.set_type(type);
      try {
        // Construct a structured::StructuredMatrix from a dense
        // matrix and given options.
        auto H = structured::construct_from_dense(A2d, options);
        // Print the memory usage etc for H
        print_info(world, H.get(), options);
        // Check the compression accuracy by comparing with the dense
        // matrix
        check_accuracy(A2d, H.get());
        // Factor H and (approximately) solve a linear system
        factor_and_solve(nrhs, A2d, H.get());
        // Solve a linear system using an iterative solver with H as
        // the preconditioner and using A as the exact matrix vector
        // product.
        preconditioned_solve(world, Tmult1d, H.get());
        // add a diagonal shift to the structured matrix, this does
        // not require recompression. Then check the accuracy again,
        // and solve a linear system with the shifted matrix.
        test_shift(nrhs, A2d, H.get());
      } catch (std::exception& e) {
        if (world.is_root())
          cout << get_name(type) << " failed: " << e.what() << endl;
      }
    }

    if (world.is_root())
      cout << endl << endl
           << "==================================" << endl
           << " Compression from matrix elements" << endl
           << "==================================" << endl;
    for (auto type : types) {
      options.set_type(type);
      try {

        {
          // Construct a structured::StructuredMatrix from individual
          // elements. A ClusterTree for the rows (and columns) can
          // also be provided.
          auto H = structured::construct_from_elements<double>
            (world, n, n, Toeplitz, options);
          print_info(world, H.get(), options);
          check_accuracy(A2d, H.get());
          factor_and_solve(nrhs, A2d, H.get());
          preconditioned_solve(world, Tmult1d, H.get());
          test_shift(nrhs, A2d, H.get());
        }

        {
          // Define a routine to compute sub-block of the
          // matrix. Often this could be implemented more efficiently
          // than computing element per element.
          auto Toeplitz_block =
            [&Toeplitz](const std::vector<std::size_t>& I,
                        const std::vector<std::size_t>& J,
                        DenseMatrix<double>& B) {
              for (std::size_t j=0; j<J.size(); j++)
                for (std::size_t i=0; i<I.size(); i++)
                  B(i, j) = Toeplitz(I[i], J[j]);
            };
          // TODO construction using a sub-block instead of individual
          // elements.
        }

      } catch (std::exception& e) {
        if (world.is_root())
          cout << get_name(type) << " failed: " << e.what() << endl;
      }
    }

    if (world.is_root())
      cout << endl << endl
           << "========================================" << endl
           << " Compression from matrix-vector product" << endl
           << "========================================" << endl;
    for (auto type : types) {
      options.set_type(type);
      try {
        {
          // Construct a structured::StructuredMatrix using a
          // matrix-vector product, where the (multi-)vectors are
          // stored using a 2d block cyclic distribution.
          auto H = structured::construct_matrix_free<double>
            (world, &grid, n, n, Tmult2d, options);
          print_info(world, H.get(), options);
          check_accuracy(A2d, H.get());
          factor_and_solve(nrhs, A2d, H.get());
          iterative_refinement_2d<double>
            (world, &grid, Tmult2d, nrhs, H.get());
          test_shift(nrhs, A2d, H.get());
        }
        { // 1d block row distribution for the product
          auto H = structured::construct_matrix_free<double>
            (world, n, n, Tmult1d, options);
          print_info(world, H.get(), options);
          check_accuracy(A2d, H.get());
          factor_and_solve(nrhs, A2d, H.get());
          preconditioned_solve(world, Tmult1d, H.get());
          test_shift(nrhs, A2d, H.get());
        }
      } catch (std::exception& e) {
        if (world.is_root())
          cout << get_name(type) << " failed: " << e.what() << endl;
      }
    }

  } // close this scope to destroy everything before calling
    // MPI_Finalize and Cblacs_exit
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}
