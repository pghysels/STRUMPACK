#include <cmath>
#include <iostream>
using namespace std;

#include "dense/DistributedMatrix.hpp"
#include "HSS/HSSMatrixMPI.hpp"
using namespace strumpack;
using namespace strumpack::HSS;
using DistM_t = DistributedMatrix<double>;
using DenseM_t = DenseMatrix<double>;

int run(int argc, char* argv[]) {

  if (!mpi_rank()) cout << "## Running" << endl;

  auto P = mpi_nprocs(MPI_COMM_WORLD);

  int m  = 10;
  int s  = 10;
  int NB = 64;

  // Setting command line arguments
  if (argc > 3)  m = stoi(argv[1]);
  if (argc > 3)  s = stoi(argv[2]);
  if (argc > 3) NB = stoi(argv[3]);

  // Initialize the BLACS grid
  if (!mpi_rank()) cout << "## m  = " << m  << endl;
  if (!mpi_rank()) cout << "## s  = " << s  << endl;
  if (!mpi_rank()) cout << "## NB = " << NB << endl;

  int npcol = floor(sqrt((float)P));
  int nprow = P / npcol;
  int ctxt, dummy, prow, pcol;

  scalapack::Cblacs_get(0, 0, &ctxt);
  scalapack::Cblacs_gridinit(&ctxt, "C", nprow, npcol);
  scalapack::Cblacs_gridinfo(ctxt, &dummy, &dummy, &prow, &pcol);
  int ctxt_all = scalapack::Csys2blacs_handle(MPI_COMM_WORLD);
  scalapack::Cblacs_gridinit(&ctxt_all, "R", 1, P);

  // Setting up distributed matrices
  
  DistM_t A, B, C;

  A = DistM_t(ctxt, m, s, NB, NB);
  A.random();

  B = DistM_t(ctxt, m, s, NB, NB);
  B.random();

  C = DistM_t(ctxt, m, s, NB, NB);
  C.zero();

  auto start = std::chrono::system_clock::now();
  gemm(Trans::N, Trans::N, 1., A, B, 0., C);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;
  double d_elapsed_seconds = elapsed_seconds.count();
  double d_elapsed_seconds_max;
  MPI_Reduce(&d_elapsed_seconds, &d_elapsed_seconds_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  
  // STRUMPACK_CB_SAMPLE_FLOPS(gemm_flops(Trans::N, Trans::N, 1., A, B, 0.));
  // A.print();
  // B.print();
  // C.print();

  float flops = m*m*(2.*s-1.);

  // % Cori peak
  // Flops / MaxTime / (Cores*Peak) = % of peak

  if(!mpi_rank()){
    cout << "PDGEMM_time    = " << d_elapsed_seconds_max << endl;
    cout << "PDGEMM_flops   = " << flops << endl;
    cout << "PDGEMM_Gflop/s = " << flops / d_elapsed_seconds_max << endl;
    cout << "PDGEMM_perPeak = " << flops / d_elapsed_seconds_max / ( P * 36.8e9 ) << endl;
  }

  return 0;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int ierr = run(argc, argv);

  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return ierr;
}
