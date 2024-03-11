#include <iostream>
#include <cmath>
#include <complex>
#include <vector>


#include "HSS/HSSMatrixMPI.hpp"
#include "dense/DenseMatrix.hpp"


template<typename T> void cross(const T* a, const T* b, T* c) {
  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
}

template<typename T> T norm(const T* v, int N) {
  T nrm(0.);
  for (int i=0; i<N; i++) nrm += v[i]*v[i];
  return std::sqrt(nrm);
}

template<typename T> std::complex<T> BesselH0(T x) {
  return std::cyl_bessel_j(0, x) +
    std::complex<T>(0., 1.) * std::cyl_neumann(0, x);
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);
  int N = std::atoi(argv[1]);

  {

  strumpack::MPIComm c;

  HSSOptions<std::complex<double>> hss_opts;
  hss_opts.set_from_command_line(argc, argv);

  int shape = 1;
  double pos_src[] = {1.8, 1.8};
  int order = 2;


  
  int kk = 16; 
  if (N == 5000) kk = 64;
  if (N == 10000) kk = 128;
  if (N == 20000) kk = 256;
  double w = M_PI * kk;

  int center[] = {1, 1};
  int nquad = 4;
  double gamma = 1.781072418;
  auto n_num = [](double x, double y) { return 2.; };
  auto g = [&n_num](double x[2], double x0[2], double w) {
    double d[] = {x[0]-x0[0], x[1]-x0[1]};
    return std::complex<double>(0, 1./4.) *
      BesselH0(w * n_num(x[0], x[1]) * norm(d, 2));
  };

  double a = 0.5, b = 0.5, dt = M_PI * 2 / (N - 1);
  std::vector<double> dl(N);
  DenseMatrix<double> pn0(2, N), pn1(2, N), xyz(2, N), pnrms(2, N);
  double z[] = {0., 0., 1.}, tmp1[3];
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    auto t = i * dt;
    pn0(0, i) = a * std::cos(t - dt / 2) + center[0];
    pn0(1, i) = b * std::sin(t - dt / 2) + center[1];
    pn1(0, i) = a * std::cos(t + dt / 2) + center[0];
    pn1(1, i) = b * std::sin(t + dt / 2) + center[1];
    xyz(0, i) = a * std::cos(t) + center[0];
    xyz(1, i) = b * std::sin(t) + center[1];
    double tmp[] = {pn1(0,i) - pn0(0,i), pn1(1,i) - pn0(1,i), 0.};
    cross(tmp, z, tmp1);
    double nrmtmp = norm(tmp1, 2);
    pnrms(0, i) = tmp1[0] / nrmtmp;
    pnrms(1, i) = tmp1[1] / nrmtmp;
    dl[i] = norm(tmp, 2);
  }

  if (c.is_root()) {
    int nmax = 2;
    auto lambda = 2. * M_PI / w / nmax;
    auto ppw = lambda / *std::max_element(dl.begin(), dl.end());
    std::cout << "ppw: " << ppw << std::endl;
  }

  DenseMatrix<std::complex<double>> B(N, 1);
  // #pragma omp parallel for
  for (int i=0; i<N; i++) {
    double p[] = {xyz(0,i), xyz(1,i)};
    double rvec[] = {p[0] - pos_src[0], p[1] - pos_src[1]};
    B(i, 0) = - std::complex<double>(0, 1./4.) *
      BesselH0(w * n_num(p[0], p[1]) * norm(rvec, 3));
  }


  TaskTimer tassmbly("");
  tassmbly.start();
  auto Lelem = [&](int i, int j) {
      double p[] = {xyz(0,i), xyz(1,i)};
      double k = w * n_num(p[0], p[1]);
        std::complex<double> Lij(0., 0.);
        if (i == j)
          Lij = 1. / (2. * M_PI) *
            (dl[j] - dl[j] * std::log(dl[j] / 2.));
        for (int aa=0; aa<nquad; aa++) {
          auto nq = (aa - 0.5) / nquad;
          double q[] = {pn0(0,j) + nq * (pn1(0,j)-pn0(0,j)),
            pn0(1,j) + nq * (pn1(1,j)-pn0(1,j))};
          double rvec[] = {p[0]-q[0], p[1]-q[1]};
          auto r = norm(rvec, 2);
          auto G = std::complex<double>(0, 1./4.) * BesselH0(k * r);
          if (std::abs(i-j) > 0)
            Lij += dl[j] / nquad * G;
          else {
            auto G0 = -1. / (2. * M_PI) * std::log(r);
            Lij += dl[j] / nquad * (G - G0);
          }
        }
      }
      return Lij;
  }
  strumpack::BLACSGrid grid(c);
  strumpack::DistributedMatrix<std::complex<double>> A(&grid, N, N);
  A.fill(Lelem);
  if (c.is_root())
  std::cout << "# SIE assembly time: " << tassmbly.elapsed() << std::endl;






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

}

MPI_Finalize();
  return 0;

}
