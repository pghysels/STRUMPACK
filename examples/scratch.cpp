#include <iostream>

#include "HSS/HSSMatrix.hpp"
#include "dense/DenseMatrix.hpp"
#include "dense/DistributedMatrix.hpp"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

int n = 8;

void run()
{
  auto P = mpi_nprocs(MPI_COMM_WORLD);
  cout << mpi_nprocs() << "/" << mpi_rank() << endl;

  // Initialize BLACSGrid
  BLACSGrid grid(MPI_COMM_WORLD);

  DenseMatrix<double> Adense;
  DistributedMatrix<double> Adist;
  DistributedMatrixWrapper<double> Adist;

  if (!mpi_rank())
  {
    // Dense matrix only on master rank
    Adense = DenseMatrix(n,n);
    int cnt = 0;
    for (int j = 0; j < n; j++)
      for (int i = 0; i < n; i++)
        Adense(i, j) = cnt++;
  }

}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  run();

  MPI_Finalize();
  return 0;
}