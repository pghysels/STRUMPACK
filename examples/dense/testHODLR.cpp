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
#include <random>
#include "HODLR/HODLRMatrix.hpp"

using namespace std;
using namespace strumpack;

int main(int argc, char* argv[]) {
  // the HODLR interfaces require MPI
  MPI_Init(&argc, &argv);

  {
    // c will be a wrapper to MPI_COMM_WORLD
    MPIComm c;

    // the matrix size
    int N = 1000;
    if (argc > 1) N = stoi(argv[1]);
    int nrhs = 1;

    HODLR::HODLROptions<double> opts;
    opts.set_from_command_line(argc, argv);


    // define a partition tree for the HODLR matrix
    structured::ClusterTree t(N);
    t.refine(opts.leaf_size());

    auto Toeplitz = [](int i, int j) { return 1./(1.+abs(i-j)); };

    if (c.is_root())
      cout << "# compressing " << N << " x " << N << " Toeplitz matrix,"
           << " with relative tolerance " << opts.rel_tol() << endl;

    // construct a HODLR representation for a Toeplitz matrix, using
    // only a routine to evaluate individual elements
    HODLR::HODLRMatrix<double> H(c, t, Toeplitz, opts);

    auto memfill = H.total_memory() / 1.e6;
    auto maxrank = H.max_rank();
    if (c.is_root())
      cout << "# H has max rank " << maxrank << " and takes "
           << memfill << " MByte (compared to "
           << (N*N / 1.e6) << " MByte for dense storage)"
           << endl;

    //////////////////////////////////////////////////////////////////
    // extract a random submatrix from H to check the accuracy
    vector<std::size_t> I(20), J(20);
    if (c.is_root()) {
      default_random_engine gen;
      uniform_int_distribution<size_t> random_idx(0, N-1);
      for (int i=0; i<I.size(); i++) I[i] = random_idx(gen);
      for (int j=0; j<J.size(); j++) J[j] = random_idx(gen);
    }
    c.broadcast(I);
    c.broadcast(J);
    DenseMatrix<double> B(I.size(), J.size());
    // extract H(I,J) and put in B (only on root)
    H.extract_elements(I, J, B);
    if (c.is_root()) {
      for (int j=0; j<J.size(); j++)
        for (int i=0; i<I.size(); i++)
          B(i, j) -= Toeplitz(I[i], J[j]);
      cout << "# extracting a random " << I.size() << " x " << J.size()
           << " submatrix to check accuracy" << endl
           << "# ||H(I,J) - T(I,J)||_F = " << B.normF() << endl;
    }
    //////////////////////////////////////////////////////////////////


    // compute a factorization of the HODLR matrix. H can still be used
    // for regular matrix vector multiplication
    H.factor();

    auto memfactor = H.total_factor_memory() / 1.e6;
    if (c.is_root())
      cout << "# computed a factorization of H, which takes an additional "
           << memfactor << " MByte" << endl;

    {
      //////////////////////////////////////////////////////////////////
      // solve with 2D block cyclic distributed vectors X and B
      BLACSGrid grid(c);
      // vector of unknowns and right-hand side
      DistributedMatrix<double> X(&grid, N, nrhs),
        Xexact(&grid, N, nrhs), B(&grid, N, nrhs);
      // H*X = B
      Xexact.random();               // set a random X
      H.mult(Trans::N, Xexact, B);   // compute B = H*X
      auto normXexact = Xexact.normF();
      H.solve(B, X);
      X.scaled_add(-1., Xexact);
      auto normE = X.normF();
      if (c.is_root())
        cout << "# solving H*X=B with compressed HODLR matrix (2D block cyclic)" << endl
             << "# relative error = ||H\\(H*Xexact) - Xexact||_F/||Xexact||_F = "
             << normE / normXexact << endl;
      //////////////////////////////////////////////////////////////////
    }

    {
      //////////////////////////////////////////////////////////////////
      // solve with 1D block-row distributed vectors X and B
      int lrows = H.lrows();
      DenseMatrix<double> Xloc(lrows, nrhs),
        Bloc(lrows, nrhs), Xexactloc(lrows, nrhs);
      Xexactloc.random();
      H.mult(Trans::N, Xexactloc, Bloc);   // compute B = H*X
      auto normXexact = Xexactloc.normF();
      normXexact = sqrt(c.all_reduce(normXexact*normXexact, MPI_SUM));
      H.solve(Bloc, Xloc);
      Xloc.scaled_add(-1., Xexactloc);
      auto normE = Xloc.normF();
      normE = sqrt(c.all_reduce(normE*normE, MPI_SUM));
      if (c.is_root())
        cout << "# solving H*X=B with compressed HODLR matrix (1D block row)" << endl
             << "# relative error = ||H\\(H*Xexact) - Xexact||_F/||Xexact||_F = "
             << normE / normXexact << endl;
      //////////////////////////////////////////////////////////////////
    }

  }

  MPI_Finalize();
  return 0;
}
