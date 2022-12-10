#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>

#include "HSS/HSSMatrix.hpp"
#define ERROR_TOLERANCE 1e2
#define SOLVE_TOLERANCE 1e-12


int main(int argc, char* argv[]) {

  auto M = strumpack::DenseMatrix<double>::read(argv[1]);
  int n = M.rows();

  strumpack::HSS::HSSOptions<double> hss_opts;
  hss_opts.set_from_command_line(argc, argv);

  std::vector<int> ranks;
  std::vector<double> times, errors;

  for (int r=0; r<5; r++) {
    auto begin = std::chrono::steady_clock::now();
    strumpack::HSS::HSSMatrix<double> H(M, hss_opts);
    auto end = std::chrono::steady_clock::now();
    if (H.is_compressed()) {
      std::cout << "# created M matrix of dimension "
                << H.rows() << " x " << H.cols()
                << " with " << H.levels() << " levels" << std::endl;
      auto T = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
      times.push_back(T);
      std::cout << "# total compression time = " << T << " [10e-3s]" << std::endl;
      std::cout << "# compression succeeded!" << std::endl;
    } else {
      std::cout << "# compression failed!!!!!!!!" << std::endl;
      return 1;
    }
    std::cout << "# rank(H) = " << H.rank() << std::endl;
    ranks.push_back(H.rank());
    std::cout << "# memory(H) = " << H.memory()/1e6 << " MB, "
              << 100. * H.memory() / M.memory() << "% of dense" << std::endl;

    // H.print_info();
    auto Hdense = H.dense();
    Hdense.scaled_add(-1., M);
    auto rel_err = Hdense.normF() / M.normF();
    errors.push_back(rel_err);
    std::cout << "# relative error = ||A-H*I||_F/||A||_F = "
              << rel_err << std::endl;
    std::cout << "# absolute error = ||A-H*I||_F = " << Hdense.normF() << std::endl;
    // if (Hdense.normF() / M.normF() > ERROR_TOLERANCE
    //     * std::max(hss_opts.rel_tol(),hss_opts.abs_tol())) {
    //   std::cout << "ERROR: compression error too big!!" << std::endl;
    //   return 1;
    // }
  }

  std::sort(ranks.begin(), ranks.end());
  std::sort(times.begin(), times.end());
  std::sort(errors.begin(), errors.end());

  std::cout << "min, median, max" << std::endl;
  std::cout << "ranks: " << ranks[ranks.size()/2] << " "
            << ranks[ranks.size()/2]-ranks[0] << " "
            << ranks.back() - ranks[ranks.size()/2] << std::endl;
  std::cout << "times: " << times[times.size()/2] << " "
            << times[times.size()/2]-times[0] << " "
            << times.back() - times[times.size()/2] << std::endl;
  std::cout << "errors: " << errors[errors.size()/2] << " "
            << errors[errors.size()/2]-errors[0] << " "
            << errors.back() - errors[errors.size()/2] << std::endl;

  // strumpack::TimerList::Finalize();

  return 0;
}
