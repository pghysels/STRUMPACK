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

#define ENABLE_FLOP_COUNTER 1
#define ERROR_TOLERANCE 1e1
#define SOLVE_TOLERANCE 1e-11

typedef std::complex<float> scomplex;
#define myscalar scomplex
#define SSTR(x) dynamic_cast<std::ostringstream&>(std::ostringstream() << std::dec << x).str()

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

int run(int argc, char *argv[]) {
  // Initialize timer
  TaskTimer::t_begin = GET_TIME_NOW();
  TaskTimer timer(string("compression"), 1);

  HSSOptions<myscalar> hss_opts;
  hss_opts.set_from_command_line(argc, argv);

  // Example 3
#define nprowA 8
#define npcolA 8
  int n = 27648; // In 8x8 blocks of 3456x3456
  int nrows[nprowA] = {3456, 3456, 3456, 3456, 3456, 3456, 3456, 3456};
  int ncols[npcolA] = {3456, 3456, 3456, 3456, 3456, 3456, 3456, 3456};
  // string prefix = "/global/cscratch1/sd/pghysels/BEM/mats/example3/";
  string prefix = "/global/cscratch1/sd/gichavez/intel17/paper2_tests/mats/example3/";

  int myid = mpi_rank();
  int np = mpi_nprocs();
  if (!myid)
    cout << "# matrix size: n = " << n << endl;

  int rowoffset[nprowA];
  int coloffset[npcolA];
  rowoffset[0] = 0;
  for (int i=1; i<nprowA; i++)
    rowoffset[i] = rowoffset[i-1] + nrows[i-1];
  coloffset[0] = 0;
  for (int i=1; i<npcolA; i++)
    coloffset[i] = coloffset[i-1] + ncols[i-1];

  // initialize the BLACS grid
  int npcol = floor(sqrt((float)np));
  int nprow = np / npcol;
  int ctxt, dummy, prow, pcol;
  scalapack::Cblacs_get(0, 0, &ctxt);
  scalapack::Cblacs_gridinit(&ctxt, "C", nprow, npcol);
  scalapack::Cblacs_gridinfo(ctxt, &dummy, &dummy, &prow, &pcol);
  int ctxt_all = scalapack::Csys2blacs_handle(MPI_COMM_WORLD);
  scalapack::Cblacs_gridinit(&ctxt_all, "R", 1, np);

  if (!myid)
    cout << "Reading and redistributing..." << endl;
  double tstart = MPI_Wtime();
  // read and Redistribute each piece
  DistributedMatrix<myscalar> A(ctxt, n, n);
  for (int i=0; i<nprowA; i++)
    for (int j=0; j<npcolA; j++) {
      DenseMatrix<myscalar> Atmp;
      if (!myid) {
        string locfile = "ZZ_" + SSTR(i) + "_" + SSTR(j) + "_" +
          SSTR(nrows[i]) + "_" + SSTR(ncols[j]);
        string filename = prefix + locfile;
        // cout << "Process " << myid << " reading from file "
        //      << locfile << endl;
        std::ifstream fp(filename.c_str(), ios::binary);
        if (!fp.is_open()) {
          cout << "Could not open file " << filename << endl;
          return -1;
        }
        // First 4 bytes are an integer
        int ierr;
        fp.read((char*)&ierr, 4);
        if (fp.fail() || ierr != nrows[i]*ncols[j]*8) {
          cout << "First 8 bytes should be an integer equal to nrows*ncols*8; "
               << "instead, " << ierr << endl;
          return -2;
        }
        // Read 8-byte fields
        Atmp = DenseMatrix<myscalar>(nrows[i], ncols[j]);
        fp.read((char*)Atmp.data(), 8*Atmp.rows()*Atmp.cols());
        if (fp.fail())
          cout << "Something went wrong while reading..." << endl;
        // Last 4 bytes are an integer
        fp.read((char*)&ierr,4);
        if (fp.fail() || ierr!=nrows[i]*ncols[j]*8) {
          cout << "First 8 bytes should be an integer equal to nrows*ncols*8; "
               << "instead, " << ierr << endl;
          return -2;
        }
        fp.close();
      }
      copy(nrows[i], ncols[j], Atmp, 0,
           A, rowoffset[i], coloffset[j], ctxt_all);
      // if (!myid)
      //   cout << myid << " working: (" << i << "," << j << "): "
      //        << nrows[i] << " x " << ncols[j] << endl;
    }
  MPI_Barrier(MPI_COMM_WORLD);
  double tend = MPI_Wtime();
  if (!myid)
    cout << "Read and redistribution done in " << tend - tstart << "s" << endl;
  if (!myid)
    cout << "# A.total_memory() = "
         << A.total_memory()/(1000.0*1000.0) << "MB" << endl;

  //===================================================================
  //==== Compression to HSS ===========================================
  //===================================================================
  if (!myid) cout << "# Creating HSS matrix H..." << endl;
  if (!myid) cout << "# rel_tol = " << hss_opts.rel_tol() << endl;
  MPI_Barrier(MPI_COMM_WORLD);

  // Simple compression
  timer.start();
  HSSMatrixMPI<myscalar> H(A, hss_opts, MPI_COMM_WORLD);
  if (!myid)
    cout << "## Compression time = " << timer.elapsed() << endl;

  if (!myid) {
    if (H.is_compressed())
      cout << "# created H matrix of dimension "
           << H.rows() << " x " << H.cols()
           << " with " << H.levels() << " levels" << endl
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

  DistributedMatrix<myscalar> B(ctxt, n, 1);
  B.random();
  DistributedMatrix<myscalar> C(B);

  timer.start();
    H.solve(ULV, C);
  
  if (!myid) cout << "## Solve time = " << timer.elapsed() << endl;

  //=======================================================================
  //=== Error checking ===
  //=======================================================================
  DistributedMatrix<myscalar> Bcheck(ctxt, n, 1);

// #if false

	myscalar c_m_one(-1.,0.);
	myscalar c_zero(0.,0.);

  apply_HSS(Trans::N, H, C, c_zero, Bcheck);
  Bcheck.scaled_add(c_m_one, B);
  auto Bchecknorm = Bcheck.normF();
  auto Bnorm = B.normF();
  if (!myid)
    cout << "# relative error = ||B-H*(H\\B)||_F/||B||_F = "
         << Bchecknorm / Bnorm << endl;
  if (B.active() && Bchecknorm / Bnorm > SOLVE_TOLERANCE) {
    if (!myid)
      cout << "ERROR: ULV solve relative error too big!!" << endl;
    // MPI_Abort(MPI_COMM_WORLD, 1);
  }
// #endif

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
  MPI_Init(&argc, &argv);

  // Main program execution
  int ierr = run(argc, argv);

  if (ENABLE_FLOP_COUNTER) {
    // Reducing flop counters
    float flops[12] = {
      float(params::random_flops.load()),
      float(params::ID_flops.load()),
      float(params::QR_flops.load()),
      float(params::ortho_flops.load()),
      float(params::reduce_sample_flops.load()),
      float(params::update_sample_flops.load()),
      float(params::extraction_flops.load()),
      float(params::CB_sample_flops.load()),
      float(params::sparse_sample_flops.load()),
      float(params::ULV_factor_flops.load()),
      float(params::schur_flops.load()),
      float(params::full_rank_flops.load())
    };
    float rflops[12];
    MPI_Reduce(flops, rflops, 12, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    print_flop_breakdown (rflops[0], rflops[1], rflops[2], rflops[3],
                          rflops[4], rflops[5], rflops[6], rflops[7],
                          rflops[8], rflops[9], rflops[10], rflops[11]);
  }

  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return ierr;
}
