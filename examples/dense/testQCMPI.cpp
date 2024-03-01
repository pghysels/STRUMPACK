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

  int n = std::atoi(argv[1]);

  {
    strumpack::MPIComm c;

    // strumpack::DenseMatrix<double> M(n, n);
    double d2 = 0.1 * 0.1;
    double d = std::pow(M_PI, 2.) / (6. * d2);
    auto QC = [&](int i, int j) {
      return i == j ? d : std::pow(-1.0, i-j) / (std::pow(i-j, 2.) * d2);
    };

    strumpack::BLACSGrid grid(c);
    strumpack::DistributedMatrix<double> A(&grid, n, n);
    A.fill(QC);
    // for (int j=0; j<n; j++)
    //   for (int i=0; i<n; i++)
    //     A.global(i, j, QC(i, j));

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
  MPI_Finalize();
  return 0;
}
