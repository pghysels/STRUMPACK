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
#include "structured/ClusterTree.hpp"
#include "BLR/BLRMatrixMPI.hpp"
#include "../src/misc/TaskTimer.hpp"

using namespace std;
using namespace strumpack;

#define ERROR_TOLERANCE 1e2

void abort_MPI(MPI_Comm *c, int *error, ... ) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cout << "rank = " << rank << " ABORTING!!!!!" << endl;
  abort();
}

int main(int argc, char* argv[]) {
  // the HODLR interfaces require MPI
  MPI_Init(&argc, &argv);

  MPI_Errhandler eh;
  MPI_Comm_create_errhandler(abort_MPI, &eh);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, eh);

  {
    // c will be a wrapper to MPI_COMM_WORLD
    MPIComm c;

    // the matrix size
    int N = 1000;
    if (argc > 1) N = stoi(argv[1]);
    int nrhs = 1;

    BLR::BLROptions<double> opts;
    opts.set_from_command_line(argc, argv);


    // define a partition tree for the HODLR matrix
    structured::ClusterTree t(N);
    t.refine(opts.leaf_size());

    auto Toeplitz = [](int i, int j) { return 1./(1.+abs(i-j)); };

    BLACSGrid grid(MPI_COMM_WORLD);
    DistributedMatrix<double> A(&grid, N, N);
    for (int j=0; j<N; j++)
      for (int i=0; i<N; i++)
        A.global(i, j, Toeplitz(i, j));

    if (c.is_root()){
      cout << "# compressing " << N << " x " << N << " Toeplitz matrix,"
           << " with relative tolerance " << opts.rel_tol() << endl;
      cout << "# Running with:\n# ";
#if defined(_OPENMP)
      cout << "OMP_NUM_THREADS=" << omp_get_max_threads() << endl;
#endif
    }

    // construct a HODLR representation for a Toeplitz matrix, using
    // only a routine to evaluate individual elements

    BLR::ProcessorGrid2D g(MPI_COMM_WORLD);
    g.print();
    auto B = BLR::BLRMatrixMPI<double>::from_ScaLAPACK(A, g, opts);
    if (c.is_root()) std::cout << "# from_ScaLAPACK done!" << std::endl;
    long long int f0 = 0, ftot = 0;
#if defined(STRUMPACK_COUNT_FLOPS)
    f0 = params::flops;
#endif
    TaskTimer t3("Compression");
    t3.start();
    auto Bpiv = B.factor(opts);
    t3.stop();
#if defined(STRUMPACK_COUNT_FLOPS)
    ftot = params::flops - f0;
#endif
    if (c.is_root()) std::cout << "# factor done!" << std::endl;
    auto BLU = B.to_ScaLAPACK(&grid);
    if (c.is_root()) std::cout << "# to_ScaLAPACK done!" << std::endl;

    auto Apiv = A.LU();
    BLU.scaled_add(-1., A);

    auto memfill = B.total_memory() / 1.e6;
    auto maxrank = B.max_rank();
    auto err = BLU.normF() / A.normF();
    if (c.is_root()){
      cout << "# rank(B)   = " << maxrank << endl;
      cout << "# memory(B) = " << memfill << " MB, "
        << 100. * memfill / (N*N*sizeof(double) / 1.e6) << "% of dense" << endl;
#if defined(STRUMPACK_COUNT_FLOPS)
      std::cout << "# flops     = " << double(ftot) << std::endl;
      std::cout << "# time      = " << t3.elapsed() << std::endl;
#endif
      cout << "# error = || LU(A) - LU(B) ||_F / || LU(A) ||_F = "
       << err << endl;
      if (err > ERROR_TOLERANCE
          * max(opts.rel_tol(),opts.abs_tol())) {
        cout << "ERROR: compression error too big!!" << endl;
        return 1;
      }
    }
  }

  MPI_Finalize();
  return 0;
}
