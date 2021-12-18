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

    options.set_leaf_size(64);
    options.set_rel_tol(1e-5);

    // Parse options passed on the command line, run with --help to see
    // more.
    options.set_from_command_line(argc, argv);


    // Define the matrix through a routine to compute individual
    // elements of the matrix. This is compatible with the
    // structured::extract_t type, which is:
    // std::function<double(std::size_t,std::size_t)>
    auto Toeplitz =
      [](int i, int j) {
        return 1. / (1. + abs(i-j));
      };


    // Create a 2D processor grid, as used in ScaLAPACK. This is only
    // needed if you will use 2d block-cyclic input/output.
    BLACSGrid grid(world);
    // Create 2d block cyclicly distributed matrix, and initialize it
    // as a Toeplitz matrix. This step can be avoided. Construction of
    // the StructuredMatrix can be done using a routine to compute
    // individual elements and/or using a black box matrix-vector
    // multiplication routine. Here we construct the 2D block cyclic
    // representation explicitly mainly for illustration and to test
    // the compression accuracy after compression.
    DistributedMatrix<double> T2d(&grid, n, n, Toeplitz);


    // Define a matrix-vector multiplication routine, using the 2d
    // block cyclic distribution. Ideally, the user can provide a
    // faster implementation. Instead of a 2D block cyclic
    // distribution, one can also use a 1d block row distribution, see
    // below.
    auto Tmult2d =
      [&T2d](Trans t,
             const DistributedMatrix<double>& R,
             DistributedMatrix<double>& S) {
        // simply call PxGEMM using T2d
        // gemm(t, Trans::N, 1., T2d, R, 0., S);
        T2d.mult(t, R, S); // same as gemm above
      };


    // Matrix-vector multiplication routine using 1d block row
    // distribution. Ideally, user can provide a faster
    // implementation. The block row/column distribution of the matrix
    // is given by the rdist and cdist vectors, with processor p
    // owning rows [rdist[p],rdist[p+1]). cdist is only needed for
    // non-square matrices. S is distributed according to rdist if
    // t==Trans::N, else cdist. R is distributed according to cdist of
    // t==Trans::N, else rdist.  auto
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


    // Set the structured::Type to HODLR (Hierarchically Off-Diagonal
    // Low Rank) or HODBF (Hierarchically Off-Diagonal Butterfly)
    options.set_type(structured::Type::HODLR);
    // options.set_type(structured::Type::HODBF);


    // Print how much memory the dense (2D block cyclic) matrix
    // representation takes (for comparison).
    if (world.is_root())
      cout << "dense (2DBC) " << T2d.rows() << " x " << T2d.cols()
           << " matrix" << endl
           << "  - memory(T2d) = " << T2d.memory() / 1e6 << " MByte"
           << endl << endl;


    if (world.is_root())
      cout << " Compression from matrix elements" << endl;


    // Construct a structured::StructuredMatrix from individual
    // elements. A ClusterTree for the rows (and columns) can
    // also be provided.
    auto H = structured::construct_from_elements<double>
      (world, n, n, Toeplitz, options);

    if (world.is_root())
      cout << get_name(options.type()) << endl
           << "  - total_nonzeros(H) = " << H->nonzeros() << endl
           << "  - total_memory(H) = " << H->memory() / 1e6 << " MByte" << endl
           << "  - maximum_rank(H) = " << H->rank() << endl;


    // check compression accuracy
    {
      // Allocate 2 T2d.rows x T2d.cols matrices, use the same 2D
      // processor grid as T2d
      DistributedMatrix<double> id(T2d.grid(), T2d.rows(), T2d.cols()),
        Hdense(T2d.grid(), T2d.rows(), T2d.cols());
      // set id to the identity matrix
      id.eye();
      // compute dense representation of H as H*I
      H->mult(Trans::N, id, Hdense);
      // compute relative Frobenius norm of the compression error
      auto err = Hdense.scaled_add(-1., T2d).normF() / T2d.normF();
      if (world.is_root())
        cout << "  - ||T-H||_F/||T||_F = " << err << endl;
    }

    // test factor and solve using 2d block cyclic right-hand side and
    // solution
    {
      // Allocate memory for rhs and solution vectors (matrices). Use
      // same 2D processor grid as T2d.
      DistributedMatrix<double> B(T2d.grid(), H->rows(), nrhs),
        X(T2d.grid(), H->rows(), nrhs);
      // Pick a random exact solution
      X.random();
      // Compute the right-hand side as B=T2d*X (calls ScaLAPACK PDGEMM)
      gemm(Trans::N, Trans::N, 1., T2d, X, 0., B);
      // Compute a factorization of H. The factors are stored in H.
      H->factor();
      // solve a linear system H*X=B, input is the right-hand side B,
      // which will be overwritten with the solution X.
      H->solve(B);
      // Compute the relative error on the solution. This now includes the
      // compression error.
      auto err = B.scaled_add(-1., X).normF() / X.normF();
      if (world.is_root())
        cout << "  - ||X-T\\(T*X)||_F/||X||_F = "
             << err << endl;
    }

    // test preconditioned solve, using 1d block row distribution
    {
      // Preconditioned solves only work for a single right-hand side
      nrhs = 1;

      // Preconditioned solves work using a 1d block row matrix and vector
      // distribution. nloc is the number of rows owned by this process.
      int nloc = H->local_rows();

      // Allocate memory for local part of the rhs and solution vectors.
      DenseMatrix<double> B(nloc, nrhs), X(nloc, nrhs);

      // Pick a random exact solution
      X.random();
      // Keep a copy of the exact solution X
      DenseMatrix<double> Xexact(X);

      // compute the right-hand side vector B as B = A*X, using the 1d
      // block row matrix-vector product Tmult1d, specified by the user.
      Tmult1d(Trans::N, X, B, H->rdist(), H->cdist());

      // factor the structured matrix, so it can be used as a preconditioner
      // H->factor();  // was already called

      int iterations = 0, maxit = 50, restart = 50;
      iterative::GMResMPI<double>
        (world,
         [&Tmult1d, &H](const DenseMatrix<double>& v,
                        DenseMatrix<double>& w) {
           // matrix-vector product using 1d block row distrribution
           Tmult1d(Trans::N, v, w, H->rdist(), H->cdist());
         },
         [&H](DenseMatrix<double>& v) {
           // Apply the preconditioner H: solve a linear system H*w=v.
           H->solve(v);
         },
         X, B,                     // solution (output), right-hand side
         1e-10, 1e-14,             // rtol, atol
         iterations, maxit,        // iterations (output), maximum iterations
         restart,                  // GMRes restart
         GramSchmidtType::CLASSICAL,
         false, world.is_root());  // initial guess, verbose

      // Compute the relative error (needs global reduction with all
      // processes)
      auto err = world.reduce(X.sub(Xexact).normF(), MPI_SUM)
        / world.reduce(Xexact.normF(), MPI_SUM);
      if (world.is_root())
        cout << "  - ||X-A\\(A*X)||_F/||X||_F = " << err << endl;


      iterative::BiCGStabMPI<double>
        (world,
         [&Tmult1d, &H](const DenseMatrix<double>& v,
                        DenseMatrix<double>& w) {
           // matrix-vector product with exact matrix
           Tmult1d(Trans::N, v, w, H->rdist(), H->cdist());
         },
         [&H](DenseMatrix<double>& v) {
           // preconditioning with structured matrix
           H->solve(v);
         },
         X, B,                     // solution (output), right-hand side
         1e-10, 1e-14,             // rtol, atol
         iterations, maxit,        // iterations (output), maximum iterations
         false, world.is_root());  // initial guess, verbose

      // Compute the relative error (needs global reduction with all
      // processes)
      err = world.reduce(X.sub(Xexact).normF(), MPI_SUM)
        / world.reduce(Xexact.normF(), MPI_SUM);
      if (world.is_root())
        cout << "  - ||X-A\\(A*X)||_F/||X||_F = " << err << endl;

    }

  } // close this scope to destroy everything before calling
    // MPI_Finalize and Cblacs_exit
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}
