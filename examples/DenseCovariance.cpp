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
               Gustavo Chavez.
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

#include "generateMatrix.hpp"
#include "generatePermutation.hpp"
#include "applyPermutation.hpp"

#define ENABLE_FLOP_COUNTER 0
#define ERROR_TOLERANCE 1e1
#define SOLVE_TOLERANCE 1e-11

#define myscalar double
#define myreal double

using namespace std;
// using namespace strumpack;
// using namespace strumpack::HSS;

int run(int argc, char *argv[]) {
  // myscalar lambda;
  // myreal tol, err, nrm;
  // int max_steps, steps, go;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int myid = mpi_rank();
  int np = mpi_nprocs();

  // initialize the BLACS grid
  int npcol = floor(sqrt((float)np));
  int nprow = np / npcol;
  int ctxt, dummy, prow, pcol;
  strumpack::scalapack::Cblacs_get(0, 0, &ctxt);
  strumpack::scalapack::Cblacs_gridinit(&ctxt, "C", nprow, npcol);
  strumpack::scalapack::Cblacs_gridinfo(ctxt, &dummy, &dummy, &prow, &pcol);
  int ctxt_all = strumpack::scalapack::Csys2blacs_handle(MPI_COMM_WORLD);
  strumpack::scalapack::Cblacs_gridinit(&ctxt_all, "R", 1, np);

  // A is a dense covariance matrix
  // It is centralized at first, then distributed.
  strumpack::DenseMatrix<myscalar> Acent;
  int n;
  int *perm = nullptr;
  if (!myid) {
    n = 0;
    myscalar* A = generateMatrix(argc, argv, &n);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n == 0) return -1;
    perm = generatePermutation(argc,argv);
    Acent = strumpack::DenseMatrix<myscalar>(n, n);
    applyPermutation(A, n, perm, Acent.data());
    delete[] A;
    free(perm);
  }
  else {
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n == 0) return -1;
  }

  strumpack::DistributedMatrix<myscalar> A(ctxt, n, n);
  strumpack::copy(n, n, Acent, 0, A, 0, 0, ctxt_all);

  //===================================================================
  //==== Compression to HSS ===========================================
  //===================================================================
  strumpack::HSS::HSSOptions<myscalar> hss_opts;
  hss_opts.set_from_command_line(argc, argv);

  if (!myid) cout << "# Creating HSS matrix H..." << endl;
  if (!myid) cout << "# rel_tol = " << hss_opts.rel_tol() << endl;
  MPI_Barrier(MPI_COMM_WORLD);

  // Simple compression
  double tstart = MPI_Wtime();
  strumpack::HSS::HSSMatrixMPI<myscalar> H(A, hss_opts, MPI_COMM_WORLD);
  double tend = MPI_Wtime();
  if (!myid) std::cout << "## Compression time = " << tend-tstart << "s" << std::endl;

  auto mlvl = H.max_levels();
  if (!myid) {
    if (H.is_compressed())
      cout << "# created H matrix of dimension "
           << H.rows() << " x " << H.cols()
           << " with " << mlvl << " levels" << endl
           << "# compression succeeded!" << endl;
    else cout << "# compression failed!!!!!!!!" << endl;
  }

  auto Hrank = H.max_rank();
  auto Hmem  = H.total_memory();
  auto Amem  = A.total_memory();
  if (!myid)
    cout << "## rank(H) = " << Hrank << endl
         << "# memory(H) = " << Hmem/1e6 << " MB, " << endl
         << "# mem percentage = " << 100. * Hmem / Amem
         << "% (of dense)" << endl;

  // Checking error against dense matrix
  if ( hss_opts.verbose() == 1 && n <= 1024) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto Hdense = H.dense(A.ctxt());
    MPI_Barrier(MPI_COMM_WORLD);

    Hdense.scaled_add(-1., A);
    auto HnormF = Hdense.normF();
    auto AnormF = A.normF();
    if (!myid)
      cout << "# relative error = ||A-H*I||_F/||A||_F = "
           << HnormF / AnormF << endl;
    if (A.active() && HnormF / AnormF >
        ERROR_TOLERANCE * max(hss_opts.rel_tol(),hss_opts.abs_tol())) {
      if (!myid) cout << "ERROR: compression error too big!!" << endl;
      // MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  //=======================================================================
  //=== Factorization ===
  //=======================================================================
  if (!myid)
    cout << "# Factorization..." << endl;
  TaskTimer timer(string("fact"), 1);
  timer.start();

  MPI_Barrier(MPI_COMM_WORLD);
  if (!myid)
    cout << "# computing ULV factorization of HSS matrix .. " << endl;

  auto ULV = H.factor();
  if (!myid)
    cout << "## Factorization time = " << timer.elapsed() << endl;

  //=======================================================================
  //=== Solve ===
  //=======================================================================
  if (!myid) cout << "# Solve..." << endl;
  MPI_Barrier(MPI_COMM_WORLD);


  if (!mpi_rank()) cout << "# solving linear system .." << endl;
  strumpack::DistributedMatrix<double> Xexact(ctxt, n, 1);
  strumpack::DistributedMatrix<double> B(ctxt, n, 1), Bh(ctxt, n, 1), R(ctxt, n, 1);
  Xexact.random();
  strumpack::HSS::apply_HSS(strumpack::Trans::N, H, Xexact, 0., Bh);
  strumpack::gemm(strumpack::Trans::N, strumpack::Trans::N, 1., A, Xexact, 0., B);

  strumpack::DistributedMatrix<double> X(B);
  H.solve(ULV, X);

  strumpack::gemm(strumpack::Trans::N, strumpack::Trans::N, 1., A, X, 0., R);
  R.scaled_add(-1., B);
  auto resnorm = R.normF();

  strumpack::DistributedMatrix<double> Xh(Bh);
  H.solve(ULV, Xh);
  Xh.scaled_add(-1., Xexact);
  auto errorh = Xh.normF();

  auto Xexactnorm = Xexact.normF();
  Xexact.scaled_add(-1., X);
  auto error = Xexact.normF();

  strumpack::DistributedMatrix<double> Bcheck(ctxt, n, 1);
  strumpack::HSS::apply_HSS(strumpack::Trans::N, H, X, 0., Bcheck);
  Bcheck.scaled_add(-1., B);
  auto Bchecknorm = Bcheck.normF();
  auto Bnorm = B.normF();

  if (!mpi_rank()) {
    cout << "# relative error = ||(H\\B)-Xexact||_F/||Xexact||_F = "
         << error / Xexactnorm << ",   A*Xexact=B" << endl;
    cout << "# relative error = ||(H\\B)-Xexact||_F/||Xexact||_F = "
         << errorh / Xexactnorm << ",   H*Xexact=B" << endl;
    cout << "# relative error = ||B-H*(H\\B)||_F/||B||_F = "
         << Bchecknorm / Bnorm << endl;
    cout << "# relative residual = ||A*(H\\B)-B||_F/||B||_F = "
         << resnorm / Bnorm << endl;
  }
  if (B.active() && Bchecknorm / Bnorm > SOLVE_TOLERANCE) {
    if (!mpi_rank())
      cout << "ERROR: ULV solve relative error too big!!" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  return 0;
}

void print_flop_breakdown
(float random_flops, float ID_flops, float QR_flops, float ortho_flops,
 float reduce_sample_flops, float update_sample_flops,
 float extraction_flops, float CB_sample_flops, float sparse_sample_flops,
 float ULV_factor_flops, float schur_flops, float full_rank_flops) {

  // Just root process continues
  if (mpi_rank() != 0) return;

  float sample_flops = CB_sample_flops
    + sparse_sample_flops;
  float compression_flops = random_flops
    + ID_flops + QR_flops + ortho_flops
    + reduce_sample_flops + update_sample_flops
    + extraction_flops + sample_flops;
  cout << endl;
  cout << "# ----- FLOP BREAKDOWN ---------------------"
       << endl;
  cout << "# compression           = "
       << compression_flops << endl;
  cout << "#    random             = "
       << random_flops << endl;
  cout << "#    ID                 = "
       << ID_flops << endl;
  cout << "#    QR                 = "
       << QR_flops << endl;
  cout << "#    ortho              = "
       << ortho_flops << endl;
  cout << "#    reduce_samples     = "
       << reduce_sample_flops << endl;
  cout << "#    update_samples     = "
       << update_sample_flops << endl;
  cout << "#    extraction         = "
       << extraction_flops << endl;
  cout << "#    sampling           = "
       << sample_flops << endl;
  cout << "#       CB_sample       = "
       << CB_sample_flops << endl;
  cout << "#       sparse_sampling = "
       << sparse_sample_flops << endl;
  cout << "# ULV_factor            = "
       << ULV_factor_flops << endl;
  cout << "# Schur                 = "
       << schur_flops << endl;
  cout << "# full_rank             = "
       << full_rank_flops << endl;
  cout << "# --------------------------------------------"
       << endl;
  cout << "# total                 = "
       << (compression_flops + ULV_factor_flops +
           schur_flops + full_rank_flops) << endl;
  cout << "# --------------------------------------------";
  cout << endl;
}


int main(int argc, char *argv[]) {
  // MPI_Init(&argc, &argv);

  // Main program execution
  int ierr = run(argc, argv);

  if (ENABLE_FLOP_COUNTER) {
    // Reducing flop counters
    float flops[12] = {
      float(strumpack::params::random_flops.load()),
      float(strumpack::params::ID_flops.load()),
      float(strumpack::params::QR_flops.load()),
      float(strumpack::params::ortho_flops.load()),
      float(strumpack::params::reduce_sample_flops.load()),
      float(strumpack::params::update_sample_flops.load()),
      float(strumpack::params::extraction_flops.load()),
      float(strumpack::params::CB_sample_flops.load()),
      float(strumpack::params::sparse_sample_flops.load()),
      float(strumpack::params::ULV_factor_flops.load()),
      float(strumpack::params::schur_flops.load()),
      float(strumpack::params::full_rank_flops.load())
    };
    float rflops[12];
    MPI_Reduce(flops, rflops, 12, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    print_flop_breakdown(rflops[0], rflops[1], rflops[2], rflops[3],
                         rflops[4], rflops[5], rflops[6], rflops[7],
                         rflops[8], rflops[9], rflops[10], rflops[11]);
  }

  strumpack::scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return ierr;
}
