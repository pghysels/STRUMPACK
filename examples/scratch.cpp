#include <iostream>

#include "HSS/HSSMatrix.hpp"
// #include "misc/TaskTimer.hpp"
// #include "FileManipulation.h"
// #include "preprocessing.h"
// #include "find_ann.h"
#include "dense/DenseMatrix.hpp"
#include "dense/BLACSGrid.hpp"
#include "dense/DistributedMatrix.hpp"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

int n = 8;

int run(const BLACSGrid grid)
{
  // DistributedMatrix<double> Adist = DistributedMatrix<double>(&grid, n, n);

  if (!mpi_rank())
  {
    DistributedMatrix<double> Adist;
    // Dense matrix on master rank
    DenseMatrix<double> Adense(n, n);
    int cnt = 0;
    for (int j = 0; j < n; j++)
      for (int i = 0; i < n; i++)
        Adense(i, j) = cnt++;
    // Adense.print("A_rank1", true, 4);

    Adist = DistributedMatrix<double>(&grid, Adense);
    // cout << Adist.npcols();

    cout << "Done building" << endl;
  }
  // Adist.print("Adist", 4);

  // DistributedMatrixWrapper<double> DMW;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  auto P = mpi_nprocs(MPI_COMM_WORLD);
  // initialize the BLACS grid
  int npcol = floor(sqrt((float)P));
  int nprow = P / npcol;
  int ctxt, dummy, prow, pcol;
  scalapack::Cblacs_get(0, 0, &ctxt);
  scalapack::Cblacs_gridinit(&ctxt, "C", nprow, npcol);
  scalapack::Cblacs_gridinfo(ctxt, &dummy, &dummy, &prow, &pcol);
  int ctxt_all = scalapack::Csys2blacs_handle(MPI_COMM_WORLD);
  scalapack::Cblacs_gridinit(&ctxt_all, "R", 1, P);

  // Get BLACSGrid object
  BLACSGrid grid(MPI_COMM_WORLD);

  run(grid);

  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}