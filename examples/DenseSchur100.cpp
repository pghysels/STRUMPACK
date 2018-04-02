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

#define ENABLE_FLOP_COUNTER 0
#define ERROR_TOLERANCE 1e1
#define SOLVE_TOLERANCE 1e-11

typedef std::complex<float> scomplex;
#define myscalar scomplex

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

void pseudoND(int, int, int, int, int, int, int, int, int *);

int run(int argc, char *argv[]) {

  /* A simple driver that reads a Schur complement written by Hsolver
   * and runs HSS compression.
   * Hsolver outputs a 2D distributed matrix using MPI I/O. Thus:
   *  ** the same number of MPI processes must be used (np=64, nb=64)**
   *  ** maybe the same system/architecture should be used to read the binary files... **
   * The matrix file is complex single precision (8 bytes).
   */

  int nb=64; // Hard-coded in Hsolver
  int locr, locc;

  // const char *file = "/global/cscratch1/sd/gichavez/intel17/paper2_tests/mats/Hsolver/front_3d_10000";
  const char *file = "/Users/gichavez/Desktop/front_3d_10000";

  int ctxt;
  // int descA[BLACSCTXTSIZE], descXB[BLACSCTXTSIZE];
  int np, ierr, myid;
  int myrow, mycol, nprow, npcol;
  int dummy;
  int *invusermap, *usermap;

  double tstart, tend;
  int i;
  HSSOptions<myscalar> hss_opts;
  hss_opts.set_from_command_line(argc, argv);

  // Initialize MPI
  myid=-1;
  if((ierr=MPI_Comm_rank(MPI_COMM_WORLD,&myid))) return 1;

  np=-1;
  if((ierr=MPI_Comm_size(MPI_COMM_WORLD,&np))) return 1;

  int n = 10000;

  /* Initialize the BLACS grid */
  nprow=floor(sqrt((float)np));
  npcol=np/nprow;
  scalapack::Cblacs_get(0,0,&ctxt);
  invusermap=new int[np];

  if (!myid){
    cout << "nprow=" << nprow << endl;
    cout << "npcol=" << npcol << endl;
    cout << "ctxt=" << ctxt << endl;
    cout <<  "np=" << np << endl;
  }

  /* Generate usermap with a pseudo ND of the processes */
  pseudoND(0,nprow-1,0,npcol-1,nprow,npcol,0,1,invusermap);
  
  usermap=new int[nprow*npcol];
  for(i=0;i<nprow*npcol;i++)
    usermap[invusermap[i]]=i;
  delete[] invusermap;

  // scalapack::Cblacs_gridmap(&ctxt,usermap,nprow,nprow,npcol);
  // delete[] usermap;
  // scalapack::Cblacs_gridinfo(ctxt,&nprow,&npcol,&myrow,&mycol);

  if(!myid){
    std::cout << "Processor grid for A: " << nprow << "x" << npcol << std::endl << std::endl;
  }

  locr=scalapack::numroc(n,nb,myrow,0,nprow);
  locc=scalapack::numroc(n,nb,mycol,0,npcol);
  
  if (!myid){
    cout << "locr=" << locr << endl;
    cout << "locc=" << locc << endl;
  }

  DenseMatrix<myscalar> Asingle = DenseMatrix<myscalar>(locr, locc);

  // dummy=std::max(1,locr);
  // scalapack::descinit(descA,n,n,nb,nb,0,0,ctxt,dummy);

  // Read matrix from file using MPI I/O.
  // The file is binary, and the same number of processes
  // and block sizes nb/nb that were used to generate the file
  // must be used here.

  tstart = MPI_Wtime();

  if(!myid) std::cout << "Reading from file " << file << "..." << std::endl;

  MPI_File fp;
  long long disp, bufsize, *allbufsize;

  allbufsize = new long long[np];
  bufsize=locr*locc;
  MPI_Allgather(&bufsize,1,MPI_INTEGER8,allbufsize,1,MPI_INTEGER8,MPI_COMM_WORLD);

  disp=0;
  for(i=0;i<myid;i++)
    disp+=allbufsize[i];
  disp*=8; /* Offset in bytes assuming 8-byte single complex */

  // if (myid==1){
  //   cout << "disp=" << disp << endl;
  // }

  MPI_File_open(MPI_COMM_WORLD,file,MPI_MODE_RDONLY,MPI_INFO_NULL,&fp);
  MPI_File_set_view(fp,disp,MPI_COMPLEX,MPI_COMPLEX,"native",MPI_INFO_NULL);
  MPI_File_read(fp,Asingle.data(),bufsize,MPI_COMPLEX,MPI_STATUS_IGNORE);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_close(&fp);

  delete[] allbufsize;

  tend=MPI_Wtime();
  if(!myid) std::cout << "Reading file done in: " << tend-tstart << "s" << std::endl;

  std::cout << myid << Asingle(0,0) << std::endl;

  // TODO: Distribute blocks into 2D block cyclic in STRUMPACK V2 format -> A

#if false

  //===================================================================
  //==== Compression to HSS ===========================================
  //===================================================================
  if (!myid) cout << "# Creating HSS matrix H..." << endl;
  if (!myid) cout << "# rel_tol = " << hss_opts.rel_tol() << endl;
  MPI_Barrier(MPI_COMM_WORLD);

  // Simple compression
  tstart = MPI_Wtime();

  HSSMatrixMPI<myscalar> H(A, hss_opts, MPI_COMM_WORLD);

  tend=MPI_Wtime();
  if(!myid) std::cout << "## Compression time = " << tend-tstart << "s" << std::endl;

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

void pseudoND(int bx, int ex, int by, int ey, int nx, int ny, int offset, int cut, int *order) {
  /* Pseudo-ND auxiliary routine for computation of usermap.
   * Ordering written in order starting at position "offset".
   * cut: 0 x-wise, 1 y-wise.
   *
   * Only works for nx=ny=2k (even).
   *
   * E.g, 4x4 grid:
   *   0 | 2 || 8 | 10
   *  ---|---||---|---
   *   1 | 3 || 9 | 11
   *  =======||=======
   *   4 | 6 || 12| 14
   *  ---|---||---|---
   *   5 | 7 || 13| 15
   *
   */

  int sx, sy, hx, hy;

  sx=ex-bx+1;
  sy=ey-by+1;

  if(sx==2 && sy==2) {
    /*  0 2
     *  1 3
     *  0: (0,0)=(bx,by)=(bx)*ny+(by) in nat ordering
     *  1: (1,0)=(bx+1,by)...
     *  2: ...
     */
    order[offset]  =(bx)  *ny+(by);
    order[offset+1]=(bx+1)*ny+(by);
    order[offset+2]=(bx)  *ny+(by+1);
    order[offset+3]=(bx+1)*ny+(by+1);
    return;
  }

  if(cut==0) {
    hx=bx+(ex-bx+1)/2-1;
    pseudoND(bx  ,hx,by,ey,nx,ny,offset           ,1,order);
    pseudoND(hx+1,ex,by,ey,nx,ny,offset+(hx-bx+1)*sy,1,order);
  } else {
    hy=by+(ey-by+1)/2-1;
    pseudoND(bx,ex,by  ,hy,nx,ny,offset           ,0,order);
    pseudoND(bx,ex,hy+1,ey,nx,ny,offset+(hy-by+1)*sx,0,order);
  }

}