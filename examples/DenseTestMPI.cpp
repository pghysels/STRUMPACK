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
 * works, and perform publicly and display publicly. Beginning five
 * (5) years after the date permission to assert copyright is obtained
 * from the U.S. Department of Energy, and subject to any subsequent
 * five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li,
               Gustavo Chávez.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <atomic>

#include "HSS/HSSMatrixMPI.hpp"
#include "misc/TaskTimer.hpp"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

#define ERROR_TOLERANCE 1e1
#define SOLVE_TOLERANCE 1e-12
#define myscalar double

int run(int argc, char *argv[]) {

  int n = 8;

  // Setting command line arguments
  if (argc > 1) n = stoi(argv[1]);

  HSSOptions<double> hss_opts;
  hss_opts.set_verbose(false);

  auto np = mpi_nprocs(MPI_COMM_WORLD);
  if (!mpi_rank())
    cout << "# usage: ./DenseTestMPI n (problem size)" << endl;

  if (!mpi_rank()){
    cout << "## Building distributed matrix" << endl;
    cout << "# matrix size: n = " << n << endl;
  }

  // BLACS variables
  int ctxt, dummy, myrow, mycol;
  int nprow=floor(sqrt((float)np));
  int npcol=np/nprow;

  scalapack::Cblacs_get(0 /*ctx number*/, 0 /*default number*/, &ctxt);
  scalapack::Cblacs_gridinit(&ctxt,"C",nprow,npcol);
  scalapack::Cblacs_gridinfo(ctxt,&nprow,&npcol,&myrow,&mycol);

  DistributedMatrix<double> A = DistributedMatrix<double>(ctxt, n, n);
  for (int c=0; c<n; c++)
  {
    for (int r=0; r<n; r++)
    {
      // Toeplitz matrix from Quantum Chemistry.
      myscalar pi=3.1416, d=0.1;
      A.global(r, c, (r==c) ? pow(pi,2)/6.0/pow(d,2) : pow(-1.0,r-c)/pow((myscalar)r-c,2)/pow(d,2) );
    }
  }

  hss_opts.set_from_command_line(argc, argv);
  

  if ( hss_opts.verbose() == 1 && n < 8 ){
    cout << "n = " << n << endl;
    A.print("A");
  }

  if (!mpi_rank())
    cout << "## Starting compression" << endl;

  if (!mpi_rank()) cout << "# rel_tol = " << hss_opts.rel_tol() << endl;
  HSSMatrixMPI<double> H(A, hss_opts, MPI_COMM_WORLD);
  if (H.is_compressed()) {
    if (!mpi_rank()) {
      cout << "# created H matrix of dimension "
           << H.rows() << " x " << H.cols()
           << " with " << H.levels() << " levels" << endl;
      cout << "# compression succeeded!" << endl;
    }
  } else {
    if (!mpi_rank()) cout << "# compression failed!!!!!!!!" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  auto Hrank = H.max_rank();
  auto Hmem = H.total_memory();
  auto Amem = A.total_memory();
  if (!mpi_rank()) {
    cout << "# rank(H) = " << Hrank << endl;
    cout << "# memory(H) = " << Hmem/1e6 << " MB, "
         << 100. * Hmem / Amem << "% of dense" << endl;
  }

  // Checking error against dense matrix
  if ( hss_opts.verbose() == 1 && n <= 1024) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto Hdense = H.dense(A.ctxt());
    MPI_Barrier(MPI_COMM_WORLD);

    Hdense.scaled_add(-1., A);
    auto HnormF = Hdense.normF();
    auto AnormF = A.normF();
    if (!mpi_rank())
      cout << "# relative error = ||A-H*I||_F/||A||_F = "
           << HnormF / AnormF << endl;
    if (A.active() && HnormF / AnormF >
        ERROR_TOLERANCE * max(hss_opts.rel_tol(),hss_opts.abs_tol())) {
      if (!mpi_rank()) cout << "ERROR: compression error too big!!" << endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }


  if (!mpi_rank())
    cout << "## Starting factorization and solve" << endl;

  MPI_Barrier(MPI_COMM_WORLD);
  if (!mpi_rank()) cout << "# computing ULV factorization of HSS matrix .. ";
  auto ULV = H.factor();
  if (!mpi_rank()) cout << "Done!" << endl;

  if (!mpi_rank()) cout << "# solving linear system .." << endl;
  DistributedMatrix<double> B(ctxt, n, 1);
  B.random();
  DistributedMatrix<double> C(B);
  H.solve(ULV, C);

  DistributedMatrix<double> Bcheck(ctxt, n, 1);
  apply_HSS(Trans::N, H, C, 0., Bcheck);
  Bcheck.scaled_add(-1., B);
  auto Bchecknorm = Bcheck.normF();
  auto Bnorm = B.normF();
  if (!mpi_rank())
    cout << "# relative error = ||B-H*(H\\B)||_F/||B||_F = "
         << Bchecknorm / Bnorm << endl;
  if (B.active() && Bchecknorm / Bnorm > SOLVE_TOLERANCE) {
    if (!mpi_rank())
      cout << "ERROR: ULV solve relative error too big!!" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (!mpi_rank())
    cout << "## Test succeeded, exiting" << endl;

  return 0;
}


int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int ierr = run(argc, argv);
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return ierr;
}
