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
  MPIComm c;
  auto P = c.size();
  cout << P << "/" << c.rank() << endl;

  // Initialize BLACSGrid
  BLACSGrid grid(c);
  DistributedMatrix<double> Adist(&grid, n, n);
  DenseMatrix<double> Adense;

  if (!mpi_rank())
{
   Adense = DenseMatrix<double>(n, n);

    // Dense matrix only on master rank
    int cnt = 0;
    for (int j = 0; j < n; j++)
      for (int i = 0; i < n; i++)
        Adense(i, j) = cnt++;
  }

    // scatter Adense (from the root) to the distributed matrix Adist
  Adist.scatter(Adense);
  Adist.print("Adist");

}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  run();

  MPI_Finalize();
  return 0;
}