#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>

#include "HSS/HSSMatrixMPI.hpp"

int main(int argc, char* argv[]) {
MPI_Init(&argc, &argv);

{
  strumpack::MPIComm c;
  strumpack::BLACSGrid grid(c);
  strumpack::DenseMatrix<double> Aseq;
  strumpack::DistributedMatrix<double> A;
  int m = 0;

  if (!strumpack::mpi_rank()) {
    Aseq = strumpack::DenseMatrix<double>::read(argv[1]);
    m = Aseq.rows();
    std::cout << "# Matrix dimension read from file: " << m << std::endl;
  }

  MPI_Bcast(&m, 1, strumpack::mpi_type<int>(), 0, MPI_COMM_WORLD);
  
  A = strumpack::DistributedMatrix<double>(&grid, m, m);
  // A.scatter(Aseq);
  std::size_t B = 10000;
  for (std::size_t r=0; r<m; r+=B) {
    auto nr = std::min(B, m-r);
    strumpack::DenseMatrixWrapper<double> Ar(nr, m, Aseq, r, 0);
    strumpack::copy(nr, m, Ar, 0, A, r, 0, grid.ctxt_all());
  }
  
  std::cout << "# scatter success" << std::endl;
  Aseq.clear();

  

   strumpack::HSS::HSSOptions<double> hss_opts;
    hss_opts.set_from_command_line(argc, argv);


    std::vector<int> ranks;
    std::vector<double> times, errors;

    for (int r=0; r<10; r++) {
      auto begin = std::chrono::steady_clock::now();
      strumpack::HSS::HSSMatrixMPI<double> H(A, hss_opts);
      auto end = std::chrono::steady_clock::now();
      if (H.is_compressed()) {
        auto max_levels = H.max_levels();
        if (c.is_root())
          std::cout << "# created M matrix of dimension "
                    << H.rows() << " x " << H.cols()
                    << " with " << max_levels << " levels" << std::endl;
        auto T = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        times.push_back(T);
        if (c.is_root())
          std::cout << "# total compression time = " << T << " [10e-3s]" << std::endl
                    << "# compression succeeded!" << std::endl;
      } else {
        if (c.is_root())
          std::cout << "# compression failed!!!!!!!!" << std::endl;
        return 1;
      }
      auto rk = H.max_rank();
      if (c.is_root())
        std::cout << "# rank(H) = " << rk << std::endl;
      ranks.push_back(rk);

      auto tot_mem_H = H.total_memory();
      if (c.is_root())
        std::cout << "# memory(H) = " << tot_mem_H/1e6 << " MB, "
                  << 100. * tot_mem_H / A.total_memory() << "% of dense" << std::endl;

      // H.print_info();
      auto Hdense = H.dense();
      Hdense.scaled_add(-1., A);
      auto rel_err = Hdense.normF() / A.normF();
      errors.push_back(rel_err);
      auto Hdnorm = Hdense.normF();
      if (c.is_root())
        std::cout << "# relative error = ||A-H*I||_F/||A||_F = "
                  << rel_err << std::endl
                  << "# absolute error = ||A-H*I||_F = "
                  << Hdnorm << std::endl;
    }

    std::sort(ranks.begin(), ranks.end());
    std::sort(times.begin(), times.end());
    std::sort(errors.begin(), errors.end());

    if (c.is_root())
      std::cout << "min, median, max" << std::endl
                << "ranks: " << ranks[0] << " "
                << ranks[ranks.size()/2] << " "
                << ranks[ranks.size()-1] << std::endl
                << "times: " << times[0] << " "
                << times[times.size()/2] << " "
                << times[times.size()-1] << std::endl
                << "errors: " << errors[0] << " "
                << errors[errors.size()/2] << " "
                << errors[errors.size()-1] << std::endl;

    // strumpack::TimerList::Finalize();
  }

  return 0;
}
