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
  cout << "Rank " << P << "/" << c.rank() << endl;

  auto cA = c.sub(0, 2); // Sub-communicator. ranks (0,1)
  if (!cA.is_null()) {
    BLACSGrid gridA(cA);
    DenseMatrix<double> Adense;
    DistributedMatrix<double> Adist(&gridA, n, n);
    if (mpi_rank() == 0){
      int cnt = 0;
      Adense = DenseMatrix<double>(n, n);
      cnt = 0;
      for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
          Adense(i, j) = cnt++;
    }
    Adist.scatter(Adense);
    Adist.print("Adist");
  }

  auto cB = c.sub(2, 2); // Sub-communicator. ranks (2,3)
  if (!cB.is_null()) {
    BLACSGrid gridB(cB);
    DenseMatrix<double> Bdense;
    DistributedMatrix<double> Bdist(&gridB, n, n);
    if (mpi_rank() == 2){
      Bdense = DenseMatrix<double>(n, n);
      int cnt = 100;
      for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
          Bdense(i, j) = cnt++;
    }
    Bdist.scatter(Bdense);
    Bdist.print("Bdist");
  }

}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  run();
  MPI_Finalize();
  return 0;
}