#include <iostream>
#include <cmath>
#include <complex>
#include <vector>



#include <random>
using namespace std;

#include "HSS/HSSMatrix.hpp"
using namespace strumpack::HSS;

#define ERROR_TOLERANCE 1e2
#define SOLVE_TOLERANCE 1e-12


#include "dense/DenseMatrix.hpp"
#include "misc/TaskTimer.hpp"
using namespace strumpack;

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
  HSSOptions<std::complex<double>> hss_opts;
  hss_opts.set_from_command_line(argc, argv);

  int shape = 1;
  double pos_src[] = {1.8, 1.8};
  int order = 2;


  int N = std::atoi(argv[1]);
  int kk = 16; // std::atoi(argv[2]);
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

  {
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
  DenseMatrix<std::complex<double>> Lop(N, N);

  std::string fname("Lop_" + std::to_string(N) + ".bin");
  std::ifstream f(fname.c_str());
  if (f.good())
    Lop = DenseMatrix<std::complex<double>>::read(fname);
  else {
#pragma omp parallel for
    for (int i=0; i<N; i++) {
      double p[] = {xyz(0,i), xyz(1,i)};
      double k = w * n_num(p[0], p[1]);
      for (int j=0; j<N; j++) {
	Lop(i, j) = 0.;
	if (i == j)
	  Lop(i, j) = 1. / (2. * M_PI) *
	    (dl[j] - dl[j] * std::log(dl[j] / 2.));
	for (int aa=0; aa<nquad; aa++) {
	  auto nq = (aa - 0.5) / nquad;
	  double q[] = {pn0(0,j) + nq * (pn1(0,j)-pn0(0,j)),
	    pn0(1,j) + nq * (pn1(1,j)-pn0(1,j))};
	  double rvec[] = {p[0]-q[0], p[1]-q[1]};
	  auto r = norm(rvec, 2);
	  auto G = std::complex<double>(0, 1./4.) * BesselH0(k * r);
	  if (std::abs(i-j) > 0)
	    Lop(i, j) += dl[j] / nquad * G;
	  else {
	    auto G0 = -1. / (2. * M_PI) * std::log(r);
	    Lop(i, j) += dl[j] / nquad * (G - G0);
	  }
	}
      }
    }
    std::cout << "# SIE assembly time: " << tassmbly.elapsed() << std::endl;
    Lop.write(fname);
  }

  // strumpack::HSS::HSSOptions<std::complex<double>> hss_opts;
  // hss_opts.set_from_command_line(argc, argv);

  std::vector<int> ranks;
  std::vector<double> times, errors;

  for (int r=0; r<3; r++) {
    auto begin = std::chrono::steady_clock::now();
    strumpack::HSS::HSSMatrix<std::complex<double>> H(Lop, hss_opts);
    auto end = std::chrono::steady_clock::now();
    if (H.is_compressed()) {
      std::cout << "# created Lop matrix of dimension "
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
              << 100. * H.memory() / Lop.memory() << "% of dense" << std::endl;

    // H.print_info();
    auto Hdense = H.dense();
    Hdense.scaled_add(-1., Lop);
    auto rel_err = Hdense.normF() / Lop.normF();
    errors.push_back(rel_err);
    std::cout << "# relative error = ||A-H*I||_F/||A||_F = "
              << rel_err << std::endl;
    std::cout << "# absolute error = ||A-H*I||_F = " << Hdense.normF() << std::endl;
    // if (Hdense.normF() / Lop.normF() > ERROR_TOLERANCE
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


//   HSSMatrix<std::complex<double>> H(Lop, hss_opts);
//   if (H.is_compressed()) {
//     cout << "# created Lop matrix of dimension "
//          << H.rows() << " x " << H.cols()
//          << " with " << H.levels() << " levels" << endl;
//     cout << "# compression succeeded!" << endl;
//   } else {
//     cout << "# compression failed!!!!!!!!" << endl;
//     return 1;
//   }
//   cout << "# rank(H) = " << H.rank() << endl;
//   cout << "# memory(H) = " << H.memory()/1e6 << " MB, "
//        << 100. * H.memory() / Lop.memory() << "% of dense" << endl;

//   // H.print_info();
//   auto Hdense = H.dense();
//   Hdense.scaled_add(-1., Lop);
//   cout << "# relative error = ||A-H*I||_F/||A||_F = "
//        << Hdense.normF() / Lop.normF() << endl;
//   cout << "# absolute error = ||A-H*I||_F = " << Hdense.normF() << endl;
//   if (Hdense.normF() / Lop.normF() > ERROR_TOLERANCE
//       * max(hss_opts.rel_tol(),hss_opts.abs_tol())) {
//     cout << "ERROR: compression error too big!!" << endl;
//     return 1;
//   }


//   TaskTimer tfactor("");
//   tfactor.start();
//   auto piv = Lop.LU();
//   std::cout << "# SIE factor time: " << tfactor.elapsed() << std::endl;

//   TaskTimer tsolve("");
//   tsolve.start();
//   auto I = Lop.solve(B, piv);
//   std::cout << "# SIE solve time: " << tsolve.elapsed() << std::endl;

//   TaskTimer tscatter("");
//   tscatter.start();
//   double xmin = 0, xmax = 2;
//   double ymin = 0, ymax = 2;
//   int Nx = 100, Ny = 100;
//   double dx = (xmax - xmin) / (Nx - 1),
//     dy = (ymax - ymin) / (Ny - 1);
//   // DenseMatrix<std::complex<double>> Fsca(Nx, Ny);
//   DenseMatrix<double> Fsca(Nx, Ny);
//   Fsca.zero();
// #pragma omp parallel for
//   for (int xi=0; xi<Nx; xi++) {
//     double x = xmin + xi * dx;
//     for (int yi=0; yi<Ny; yi++) {
//       double y = ymin + yi * dy;
//       double ob[] = {x, y};
//       for (int ss=0; ss<N; ss++) {
//         double p[] = {xyz(0, ss), xyz(1, ss)};
//         double dob[] = {ob[0]-p[0], ob[1]-p[1]};
//         if (norm(dob, 2) / norm(p, 2) < 1e-14)
//           Fsca(yi,xi) +=
//             std::real(I(ss, 0) * std::complex<double>(0., 1.) * dl[ss] / 4. *
//                       (1. + std::complex<double>(0., 2./M_PI) *
//                        (std::log(gamma * w * n_num(p[0], p[1]) * dl[ss] / 4.) - 1.)));
//         else
//           Fsca(yi,xi) += std::real(I(ss, 0) * dl[ss] * g(ob, p, w));
//       }
//     }
//   }
//   std::cout << "# SIE scatter time: " << tscatter.elapsed() << std::endl;

//   std::cout << "# printing scattered field to Fsca.m" << std::endl;
//   Fsca.print_to_file("Fsca", "Fsca.m");
}
