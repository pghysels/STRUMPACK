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
// #define myreal float
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

// # ==========================================================================
// # === Reading and BLACS variables ===
// # ==========================================================================

  int i,j;
  int ctxtA, ctxt, ctxttmp, ctxtglob;
  int ierr;
  int myid, np, id;
  int myrow, mycol, nprow, npcol;
  int myrowA, mycolA;
  std::string filename, prefix, locfile;
  std::ifstream fp;
  scomplex stmp;
  double tstart, tend;

  // Example 3
  #define nprowA 8
  #define npcolA 8
  int n = 27648; // In 8x8 blocks of 3456x3456
  int nrows[nprowA]={3456,3456,3456,3456,3456,3456,3456,3456};
  int ncols[npcolA]={3456,3456,3456,3456,3456,3456,3456,3456};
  prefix="/global/cscratch1/sd/gichavez/intel17/paper2_tests/mats/example3/";

  if (!mpi_rank()){
    cout << "# matrix size: n = " << n << endl;
  }

  int rowoffset[nprowA];
  int coloffset[npcolA];

  rowoffset[0]=1;
  for(i=1;i<nprowA;i++)
    rowoffset[i]=rowoffset[i-1]+nrows[i-1];
  coloffset[0]=1;
  for(i=1;i<npcolA;i++)
    coloffset[i]=coloffset[i-1]+ncols[i-1];

  if((ierr=MPI_Comm_rank(MPI_COMM_WORLD,&myid))) return 1;
  
  np=-1;
  if((ierr=MPI_Comm_size(MPI_COMM_WORLD,&np))) return 1;

  if(np<nprowA*npcolA) {
    std::cout << "This requires " << nprowA*npcolA << " processes or more." << std::endl;
    std::cout << "Aborting." << std::endl;
    MPI_Abort(MPI_COMM_WORLD,-1);
  }

/* Initialize a BLACS grid with nprowA*npcolA processes */
  myscalar *Atmp=NULL, *A=NULL, *Btmp=NULL, *X=NULL, *B=NULL;
  int descA[BLACSCTXTSIZE], descAtmp[BLACSCTXTSIZE], descXB[BLACSCTXTSIZE], descBtmp[BLACSCTXTSIZE];

  nprow=nprowA;
  npcol=npcolA;

  // scalapack::Cblacs_get(0,0,&ctxtA);
  // scalapack::Cblacs_gridinit(&ctxtA,"R",nprow,npcol); // ctxtA(R,8,8)
  // scalapack::Cblacs_gridinfo(ctxtA,&nprow,&npcol,&myrowA,&mycolA);
  // if(!myid) {
  //   cout << "ctxtA(R," << nprow << "," << npcol << ")" <<endl;
  // }

  /* Processes 0..nprow*npcolA read their piece of the matrix */
  MPI_Barrier(MPI_COMM_WORLD);
  tstart=MPI_Wtime();
  if(myid<nprowA*npcolA) {
    locfile="ZZ_"+SSTR(myrowA)+"_"+SSTR(mycolA)+"_"+SSTR(nrows[myrowA])+"_"+SSTR(ncols[mycolA]);
    filename=prefix+locfile;
    std::cout << "Process " << myid << " reading from file " << locfile << std::endl;
    fp.open(filename.c_str(),std::ios::binary);
    if(!fp.is_open()) {
      std::cout << "Could not open file " << filename << std::endl;
      return -1;
    }

    /* First 4 bytes are an integer */
    fp.read((char *)&ierr,4);
    if(fp.fail() || ierr!=nrows[myrowA]*ncols[mycolA]*8) {
      std::cout << "First 8 bytes should be an integer equal to nrows*ncols*8; instead, " << ierr  << std::endl;
      return -2;
    }

    /* Read 8-byte fields */
    Atmp=new myscalar[nrows[myrowA]*ncols[mycolA]];
    for(i=0;i<nrows[myrowA]*ncols[mycolA];i++) {
      fp.read((char *)&stmp,8);
      Atmp[i]=static_cast<myscalar>(stmp);
      if(fp.fail()) {
        std::cout << "Something went wrong while reading..." << std::endl;
        if(fp.eof())
          std :: cout << "Only " << i << " instead of " << nrows[myrowA]*ncols[mycolA] << std::endl;
        return 2;
      }
    }

    /* Last 4 bytes are an integer */
    fp.read((char *)&ierr,4);
    if(fp.fail() || ierr!=nrows[myrowA]*ncols[mycolA]*8) {
      std::cout << "First 8 bytes should be an integer equal to nrows*ncols*8; instead, " << ierr  << std::endl;
      return -2;
    }
    fp.close();
  }
  MPI_Barrier(MPI_COMM_WORLD);

  tend=MPI_Wtime();
  if(!myid) std::cout << "Reading done in: " << tend-tstart << "s" << std::endl;

  // Part 2

  /* Initialize a context with all the processes */
  scalapack::Cblacs_get(0,0,&ctxtglob);
  scalapack::Cblacs_gridinit(&ctxtglob,"R",1,np); // ctxtglob(R,1,64)

  if(!myid) {
    cout << "ctxtglob(R," << 1 << "," << np << ")" <<endl;
  }

  /* Initialize the BLACS grid */
  nprow=floor(sqrt((float)np));
  npcol=np/nprow;

  scalapack::Cblacs_get(0,0,&ctxt);
  scalapack::Cblacs_gridinit(&ctxt,"R",nprow,npcol); // ctxt(R,8,8)
  scalapack::Cblacs_gridinfo(ctxt,&nprow,&npcol,&myrow,&mycol);
  if(!myid) {
    cout << "ctxt(R," << nprow << "," << npcol << ")" <<endl;
  }

  /* Create A in 2D block-cyclic form by redistributing each piece */
  int nb = 64;
  int locr, locc;
  int dummy;

  if(myid<nprow*npcol) {
    locr=scalapack::numroc(n, nb, myrow, 0, nprow);
    locc=scalapack::numroc(n, nb, mycol, 0, npcol);
    dummy=std::max(1,locr);
    A=new myscalar[locr*locc];
    scalapack::descinit(descA, n, n, nb, nb, 0, 0, ctxt, dummy);
  } 
  else {
    scalapack::descset(descA, n, n, nb, nb, 0, 0, 0, 1);
  }

  // /* Redistribute each piece */

  if(!myid) std::cout << "Redistributing..." << std::endl;
  tstart=MPI_Wtime();
  
  for(i=0;i<nprowA;i++)
    for(j=0;j<npcolA;j++) {
      /* Initialize a grid that contains only the process that owns piece (i,j) */
      id=i*npcolA+j;
      scalapack::Cblacs_get(0, 0, &ctxttmp);
      scalapack::Cblacs_gridmap(&ctxttmp, &id, 1, 1, 1);
      if(myid==id) {
        printf("%d working: (%d,%d): %d x %d\n",myid,i,j,nrows[i],ncols[j]);
        /* myid owns the piece of A to be distributed */
        scalapack::descinit(descAtmp,nrows[i],ncols[j],nb,nb,0,0,ctxttmp,nrows[i]);
      } else
        scalapack::descset(descAtmp, nrows[i], ncols[j], nb, nb, 0, 0, -1, 1);

      scalapack::pgemr2d(nrows[i], ncols[j], Atmp, 1, 1, descAtmp, A, rowoffset[i], coloffset[j], descA, ctxtglob);
    }

  // if(myid<nprowA*npcolA)
  //   delete[] Atmp;

  MPI_Barrier(MPI_COMM_WORLD);
  tend=MPI_Wtime();
  if(!myid) std::cout << "Redistribution done in " << tend-tstart << "s" << std::endl;

// # ==========================================================================
// # === Build dense DistributedMatrixWrapper ===
// # ==========================================================================
  if (!mpi_rank()) cout << "# DistributedMatrixWrapper..." << endl;
  timer.start();

  DistributedMatrixWrapper<myscalar> AA(ctxt, n, n, nb, nb, A);

  if (!mpi_rank()){
    cout << "## DistributedMatrixWrapper time = " << timer.elapsed() << endl;
    cout << "# AA.total_memory() = " << (myscalar)AA.total_memory()/(1000.0*1000.0) << "MB" << endl;
  }

// # ==========================================================================
// # === Compression to HSS ===
// # ==========================================================================
  if (!mpi_rank()) cout << "# Creating HSS matrix H..." << endl;
  if (!mpi_rank()) cout << "# rel_tol = " << hss_opts.rel_tol() << endl;
  MPI_Barrier(MPI_COMM_WORLD);
  
  // Simple compression
  timer.start();
    HSSMatrixMPI<myscalar> H(static_cast<DistributedMatrix<myscalar>>(AA), hss_opts, MPI_COMM_WORLD);
  if (!mpi_rank()) cout << "## Compression time = " << timer.elapsed() << endl;

  if (H.is_compressed()) {
    if (!mpi_rank()) {
      cout << "# created H matrix of dimension "
           << H.rows() << " x " << H.cols()
           << " with " << H.levels() << " levels" << endl;
      cout << "# compression succeeded!" << endl;
    }
  } else {
    if (!mpi_rank()) cout << "# compression failed!!!!!!!!" << endl;
    // MPI_Abort(MPI_COMM_WORLD, 1);
  }

  auto Hrank = H.max_rank();
  auto Hmem  = H.total_memory();
  auto Amem  = AA.total_memory();

  if (!mpi_rank()) {
    cout << "# rank(H) = " << Hrank << endl;
    cout << "# memory(H) = " << Hmem/1e6 << " MB, " << endl;
    cout << "# mem percentage = " << 100. * Hmem / Amem << "% (of dense)" << endl;
  }



  #if false

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
      // MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

// # ==========================================================================
// # === Factorization ===
// # ==========================================================================
  if (!mpi_rank())
    cout << "# Factorization..." << endl;
  timer.start();

  MPI_Barrier(MPI_COMM_WORLD);
  if (!mpi_rank()) cout << "# computing ULV factorization of HSS matrix .. " << endl;
  
    auto ULV = H.factor();
  if (!mpi_rank())
    cout << "## Factorization time = " << timer.elapsed() << endl;

// # ==========================================================================
// # === Solve ===
// # ==========================================================================
  if (!mpi_rank()) cout << "# Solve..." << endl;

  DistributedMatrix<double> B(ctxt, n, 1);
  B.random();
  DistributedMatrix<double> C(B);
  
  timer.start();
    H.solve(ULV, C);
  if (!mpi_rank())
    cout << "## Solve time = " << timer.elapsed() << endl;

// # ==========================================================================
// # === Error checking ===
// # ==========================================================================
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
    std::cout << std::endl;
    std::cout << "# ----- FLOP BREAKDOWN ---------------------"
              << std::endl;
    std::cout << "# compression           = "
              << compression_flops << std::endl;
    std::cout << "#    random             = "
              << random_flops << std::endl;
    std::cout << "#    ID                 = "
              << ID_flops << std::endl;
    std::cout << "#    QR                 = "
              << QR_flops << std::endl;
    std::cout << "#    ortho              = "
              << ortho_flops << std::endl;
    std::cout << "#    reduce_samples     = "
              << reduce_sample_flops << std::endl;
    std::cout << "#    update_samples     = "
              << update_sample_flops << std::endl;
    std::cout << "#    extraction         = "
              << extraction_flops << std::endl;
    std::cout << "#    sampling           = "
              << sample_flops << std::endl;
    std::cout << "#       CB_sample       = "
              << CB_sample_flops << std::endl;
    std::cout << "#       sparse_sampling = "
              << sparse_sample_flops << std::endl;
    std::cout << "# ULV_factor            = "
              << ULV_factor_flops << std::endl;
    std::cout << "# Schur                 = "
              << schur_flops << std::endl;
    std::cout << "# full_rank             = "
              << full_rank_flops << std::endl;
    std::cout << "# --------------------------------------------"
              << std::endl;
    std::cout << "# total                 = "
              << (compression_flops + ULV_factor_flops +
                  schur_flops + full_rank_flops) << std::endl;
    std::cout << "# --------------------------------------------";
    std::cout << std::endl;
}


int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // Main program execution  
  int ierr = run(argc, argv);

  if (ENABLE_FLOP_COUNTER){
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
