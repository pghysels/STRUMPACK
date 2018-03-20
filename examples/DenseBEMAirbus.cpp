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

typedef std::complex<double> dcomplex;
#define myscalar dcomplex
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


  /* A simple driver that reads a BEM matrix
   * from Airbus. The matrix is stored in
   * a single file.
   *
   * Parameters:
   *  - argv[1] matrix number (1 or 2)
   *  - argv[2] use HSS? true of false.
   *  - argv[3] number of levels for HSS tree.
   *  - argv[4] number of random vectors.
   *  - argv[5] tolerance for compression.
   *
   */

  myscalar *A=NULL, *X=NULL, *B=NULL;
  int descA[BLACSCTXTSIZE], descAcent[BLACSCTXTSIZE], descXB[BLACSCTXTSIZE];
  int n;
  int nb=64;
  int locr, locc;
  int i, j;
  int ierr;
  int dummy;
  int myid, np, id;
  int myrow, mycol, nprow, npcol;
  int ctxt, ctxtcent, ctxtglob;
  double tstart, tend;
  std::string filename, prefix, locfile;
  std::ifstream fp;

  n=10002;
  locfile="sphereAcous10002.res";
  prefix="/global/cscratch1/sd/gichavez/intel17/paper2_tests/mats/";
  filename=prefix+locfile;

  myid=-1;
  if((ierr=MPI_Comm_rank(MPI_COMM_WORLD,&myid)))
    return 1;

  np=-1;
  if((ierr=MPI_Comm_size(MPI_COMM_WORLD,&np)))
    return 1;

  // /* Read arguments */
  // if(argc<6) {
  //   if(!myid)
  //     std::cout << "Usage: ./main matrix_number use_HSS levels random_vectors tolerance" << std::endl;
  //   MPI_Finalize();
  //   return 0;
  // }

  /* Process 0 reads the matrix */
  tstart=MPI_Wtime();
  if(!myid) {
    std::cout << "Process " << myid << " reading from file " << filename << std::endl;
    fp.open(filename.c_str(),std::ios::binary);
    if(!fp.is_open()) {
      std::cout << "Could not open file " << filename << std::endl;
      return -1;
    }

    if(!myid) {
      cout << "n = " << n << endl;
    }

    // Acent=new myscalar[n*n];
    DenseMatrix<myscalar> Acent = DenseMatrix<myscalar>(n, n);
    fp.read((char*)Acent.data(), 16*Acent.rows()*Acent.cols());
    // for(i=0;i<n*n;i++) {
      // fp.read((char *)&Acent[i],16); // Binary file with double complex number
      // if(fp.fail()) {
      //   std::cout << "Something went wrong while reading..." << std::endl;
      //   if(fp.eof())
      //     std :: cout << "Only " << i << " instead of " << n*n << std::endl;
      //   return 2;
      // }
    // }
    fp.close();
    Acent.print();
  }
  tend=MPI_Wtime();
  if(!myid) std::cout << "Reading file done in: " << tend-tstart << "s" << std::endl;

  /* We initialize a context with only id 0 */
  scalapack::Cblacs_get(0,0,&ctxtcent);
  scalapack::Cblacs_gridinit(&ctxtcent,"R",1,1);

  /* We initialize a context with all the processes */
  scalapack::Cblacs_get(0,0,&ctxtglob);
  scalapack::Cblacs_gridinit(&ctxtglob,"R",1,np);

  /* We initialize a 2D grid */
  nprow=floor(sqrt((float)np));
  npcol=np/nprow;
  scalapack::Cblacs_get(0,0,&ctxt);
  scalapack::Cblacs_gridinit(&ctxt,"R",nprow,npcol);
  scalapack::Cblacs_gridinfo(ctxt,&nprow,&npcol,&myrow,&mycol);

  /* Distribute A into 2D block-cyclic form */
  // if(myid==0)
  //   descinit_(descAcent,&n,&n,&nb,&nb,&IZERO,&IZERO,&ctxtcent,&n,&ierr);
  // else
  //   descset_(descAcent,&n,&n,&nb,&nb,&IZERO,&IZERO,&INONE,&IONE);
  // if(myid<nprow*npcol) {
  //   locr=numroc_(&n,&nb,&myrow,&IZERO,&nprow);
  //   locc=numroc_(&n,&nb,&mycol,&IZERO,&npcol);
  //   dummy=std::max(1,locr);
  //   A=new myscalar[locr*locc];
  //   descinit_(descA,&n,&n,&nb,&nb,&IZERO,&IZERO,&ctxt,&dummy,&ierr);
  // } else {
  //   descset_(descA,&n,&n,&nb,&nb,&IZERO,&IZERO,&INONE,&IONE);
  // }
  // pgemr2d(n,n,Acent,IONE,IONE,descAcent,A,IONE,IONE,descA,ctxtglob);
  // delete[] Acent;

  // /* Set the RHS (random vector) and the solution space */
  // if(myid<nprow*npcol) {
  //   locr=numroc_(&n,&nb,&myrow,&IZERO,&nprow);
  //   locc=numroc_(&IONE,&nb,&mycol,&IZERO,&npcol);
  //   B=new myscalar[locr*locc];
  //   dummy=std::max(1,locr);
  //   descinit_(descXB,&n,&IONE,&nb,&nb,&IZERO,&IZERO,&ctxt,&dummy,&ierr);
  //   for(i=0;i<locr*locc;i++)
  //     B[i]=1.0;
  //   X=new myscalar[locr*locc];
  // } else {
  //   descset_(descXB,&n,&IONE,&nb,&nb,&IZERO,&IZERO,&INONE,&IONE);
  // }

  // //===================================================================
  // //==== Compression to HSS ===========================================
  // //===================================================================
  // if (!myid) cout << "# Creating HSS matrix H..." << endl;
  // if (!myid) cout << "# rel_tol = " << hss_opts.rel_tol() << endl;
  // MPI_Barrier(MPI_COMM_WORLD);

  // // Simple compression
  // timer.start();
  // HSSMatrixMPI<myscalar> H(A, hss_opts, MPI_COMM_WORLD);
  // if (!myid)
  //   cout << "## Compression time = " << timer.elapsed() << endl;

  // if (!myid) {
  //   if (H.is_compressed())
  //     cout << "# created H matrix of dimension "
  //          << H.rows() << " x " << H.cols()
  //          << " with " << H.levels() << " levels" << endl
  //          << "# compression succeeded!" << endl;
  //   else cout << "# compression failed!!!!!!!!" << endl;
  // }

  // auto Hrank = H.max_rank();
  // auto Hmem  = H.total_memory();
  // auto Amem  = A.total_memory();
  // if (!myid)
  //   cout << "# rank(H) = " << Hrank << endl
  //        << "# memory(H) = " << Hmem/1e6 << " MB, " << endl
  //        << "# mem percentage = " << 100. * Hmem / Amem
  //        << "% (of dense)" << endl;


#if false

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

  DistributedMatrix<double> B(ctxt, n, 1);
  B.random();
  DistributedMatrix<double> C(B);

  timer.start();
    H.solve(ULV, C);
  if (!myid)
    cout << "## Solve time = " << timer.elapsed() << endl;

  //=======================================================================
  //=== Error checking ===
  //=======================================================================
  DistributedMatrix<double> Bcheck(ctxt, n, 1);
  apply_HSS(Trans::N, H, C, 0., Bcheck);
  Bcheck.scaled_add(-1., B);
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

#endif

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
