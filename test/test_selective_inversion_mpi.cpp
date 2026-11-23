/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The
 * Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals
 * from the U.S. Dept. of Energy).  All rights reserved.
 *
 * If you have questions about your rights to use or distribute this
 * software, please contact Berkeley Lab's Technology Transfer
 * Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As
 * such, the U.S. Government has been granted for itself and others
 * acting on its behalf a paid-up, nonexclusive, irrevocable,
 * worldwide license in the Software to reproduce, prepare derivative
 * works, and perform publicly and display publicly.  Beginning five
 * (5) years after the date permission to assert copyright is obtained
 * from the U.S. Department of Energy, and subject to any subsequent
 * five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
// Selective inversion of the distributed fronts, on a nearly singular
// matrix: A = C^T C - eps I (real) or C^T C - eps (1 + i) I (complex),
// with C the discrete curl on the interior edges of an m^3 cell grid (a
// large near null space of discrete gradients, like a low frequency
// Maxwell operator). One solver is factored several times (with
// update_matrix_values) with different options, and each factorization
// is used for a solve with 1 and with 2 right-hand sides. The checks do
// not depend on the separator tree:
//  - with the default limit, the residual is within a factor 10 of the
//    one without inversion,
//  - with a limit of 1e300 every distributed front is inverted, so the
//    solution differs from the one without inversion (more than one
//    process),
//  - with a limit of 0 (no inversion attempted) and of 1 (the inverses
//    are computed but rejected, the condition number is larger than 1)
//    the solution is bitwise identical to the one without inversion,
//  - after a factorization with inverted fronts, a refactorization with
//    a limit of 0, or with the option disabled, is bitwise identical to
//    no inversion, and a refactorization with 1e300 repeats the first
//    one with 1e300.
// Whether the default limit inverts some fronts and keeps others
// depends on the tree, and is only printed.
#include <cmath>
#include <complex>
#include <cstring>
#include <map>
#include <vector>
#include <iostream>
using namespace std;

#include "StrumpackSparseSolverMPIDist.hpp"
using namespace strumpack;

// the curl-curl matrix in CSR (real part, the diagonal includes -eps)
struct CurlCurl {
  int n = 0;
  std::vector<int> ptr, ind;
  std::vector<double> val;
  CurlCurl(int m, double eps) {
    auto nid = [&](int i, int j, int k) {
      return i + (m+1) * (j + (m+1) * k);
    };
    auto interior = [&](int d, const int* c) {
      for (int e=0; e<3; e++) {
        if (c[e] < 0 || c[e] > m) return false;
        if (e == d) { if (c[e] >= m) return false; }
        else if (c[e] == 0 || c[e] == m) return false;
      }
      return true;
    };
    // interior edges: direction d, origin node (i,j,k)
    std::map<std::pair<int,int>,int> eid;
    for (int k=0; k<=m; k++)
      for (int j=0; j<=m; j++)
        for (int i=0; i<=m; i++)
          for (int d=0; d<3; d++) {
            int c[3] = {i, j, k};
            if (interior(d, c)) {
              int id = eid.size();
              eid[{d, nid(i, j, k)}] = id;
            }
          }
    n = eid.size();
    // C has one row per face: the circulation of its edges
    std::vector<std::map<int,double>> A(n);
    for (int k=0; k<=m; k++)
      for (int j=0; j<=m; j++)
        for (int i=0; i<=m; i++)
          for (int d=0; d<3; d++) {
            int a = (d + 1) % 3, b = (d + 2) % 3, c0[3] = {i, j, k};
            if (c0[a] >= m || c0[b] >= m) continue;
            std::vector<std::pair<int,double>> row;
            auto edge = [&](int dir, int shift, double s) {
              int c[3] = {i, j, k};
              if (shift >= 0) c[shift]++;
              if (interior(dir, c))
                row.push_back({eid[{dir, nid(c[0], c[1], c[2])}], s});
            };
            edge(a, -1, 1.); edge(b, a, 1.); edge(a, b, -1.); edge(b, -1, -1.);
            for (auto& x : row)
              for (auto& y : row) A[x.first][y.first] += x.second * y.second;
          }
    for (int i=0; i<n; i++) A[i][i] -= eps;
    ptr.assign(1, 0);
    for (int i=0; i<n; i++) {
      for (auto& e : A[i]) { ind.push_back(e.first); val.push_back(e.second); }
      ptr.push_back(ind.size());
    }
  }
};

template<typename T> T value(double re, double) { return re; }
template<> std::complex<double> value(double re, double im) { return {re, im}; }

template<typename T> int run(const CurlCurl& cc, double eps, const char* name) {
  int rank, P;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  const int n = cc.n;
  std::vector<int> dist(P+1);
  for (int p=0; p<=P; p++) dist[p] = (long long)n * p / P;
  const int lo = dist[rank], nloc = dist[rank+1] - lo;
  // local rows of A, with -eps i on the diagonal for complex T
  std::vector<int> ptr(1, 0), ind;
  std::vector<T> val;
  for (int i=lo; i<lo+nloc; i++) {
    for (int e=cc.ptr[i]; e<cc.ptr[i+1]; e++) {
      ind.push_back(cc.ind[e]);
      val.push_back(value<T>(cc.val[e], cc.ind[e] == i ? -eps : 0.));
    }
    ptr.push_back(ind.size());
  }
  // b = A [x1, x2], column major with leading dimension nloc
  const int nrhs = 2;
  std::vector<T> xt(std::size_t(n) * nrhs), b(std::size_t(nloc) * nrhs);
  for (int c=0; c<nrhs; c++)
    for (int i=0; i<n; i++)
      xt[i+std::size_t(c)*n] = value<T>
        (std::sin(0.37 * i + c) + std::cos(1.3 * i * i), std::cos(0.71 * i * c));
  for (int c=0; c<nrhs; c++)
    for (int i=0; i<nloc; i++)
      for (int e=ptr[i]; e<ptr[i+1]; e++)
        b[i+std::size_t(c)*nloc] += val[e] * xt[ind[e]+std::size_t(c)*n];

  // relative residual ||b - A x||_F / ||b||_F of nc columns
  const int w = sizeof(T) / sizeof(double);
  std::vector<int> cnt(P), dsp(P);
  for (int p=0; p<P; p++) {
    cnt[p] = w * (dist[p+1] - dist[p]);
    dsp[p] = w * dist[p];
  }
  auto residual = [&](const std::vector<T>& x, int nc) {
    double rb[2] = {0., 0.};
    std::vector<T> xg(n);
    for (int c=0; c<nc; c++) {
      MPI_Allgatherv(x.data() + std::size_t(c)*nloc, w * nloc, MPI_DOUBLE,
                     xg.data(), cnt.data(), dsp.data(), MPI_DOUBLE,
                     MPI_COMM_WORLD);
      for (int i=0; i<nloc; i++) {
        T bi = b[i+std::size_t(c)*nloc], r = bi;
        for (int e=ptr[i]; e<ptr[i+1]; e++) r -= val[e] * xg[ind[e]];
        rb[0] += std::norm(r);
        rb[1] += std::norm(bi);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, rb, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return std::sqrt(rb[0] / rb[1]);
  };
  auto identical = [&](const std::vector<T>& x, const std::vector<T>& y) {
    int same = x.size() == y.size() &&
      !std::memcmp(x.data(), y.data(), x.size() * sizeof(T));
    MPI_Allreduce(MPI_IN_PLACE, &same, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    return bool(same);
  };

  struct Result { std::vector<T> x1, x2; double r1, r2; };
  SparseSolverMPIDist<T,int> sp(MPI_COMM_WORLD, false);
  sp.options().set_matching(MatchingJob::NONE);
  sp.options().set_Krylov_solver(KrylovSolver::DIRECT);
  sp.options().set_reordering_method(ReorderingStrategy::METIS);
  sp.set_distributed_csr_matrix
    (nloc, ptr.data(), ind.data(), val.data(), dist.data(), false);
  int nfail = 0, nfact = 0;
  // limit < 0: selective inversion disabled
  auto factor_solve = [&](double limit) {
    if (nfact++)
      sp.update_matrix_values
        (nloc, ptr.data(), ind.data(), val.data(), dist.data(), false);
    if (limit < 0) sp.options().disable_selective_inversion();
    else {
      sp.options().enable_selective_inversion();
      sp.options().set_selective_inversion_max_cond(limit);
    }
    Result r{std::vector<T>(nloc), std::vector<T>(std::size_t(nloc) * nrhs),
             0., 0.};
    if (sp.factor() != ReturnCode::SUCCESS ||
        sp.solve(b.data(), r.x1.data()) != ReturnCode::SUCCESS ||
        sp.solve(nrhs, b.data(), nloc, r.x2.data(), nloc) !=
        ReturnCode::SUCCESS)
      nfail++;
    r.r1 = residual(r.x1, 1);
    r.r2 = residual(r.x2, nrhs);
    return r;
  };
  auto check = [&](bool ok, const char* what) {
    if (!ok) {
      nfail++;
      if (!rank) cout << "FAILED (" << name << "): " << what << endl;
    }
  };
  auto same_as = [&](const Result& a, const Result& b) {
    return identical(a.x1, b.x1) && identical(a.x2, b.x2);
  };
  const double inf = 1e300;
  auto off = factor_solve(-1), def = factor_solve(1e3),
    all = factor_solve(inf), lim0 = factor_solve(0.),
    all2 = factor_solve(inf), dis = factor_solve(-1),
    lim1 = factor_solve(1.);
  if (!rank)
    cout << "# " << name << ", " << P << " processes, relative residual"
         << " (1 and " << nrhs << " right-hand sides): no inversion "
         << off.r1 << " " << off.r2 << ", default limit " << def.r1 << " "
         << def.r2 << ", limit 1e300 " << all.r1 << " " << all.r2 << endl;
  check(def.r1 <= 10. * off.r1 + 1e-15 && def.r2 <= 10. * off.r2 + 1e-15,
        "selective inversion less accurate");
  check(std::isfinite(all.r1) && std::isfinite(all.r2),
        "residual with every front inverted");
  if (P > 1)
    check(!identical(all.x1, off.x1) && !identical(all.x2, off.x2),
          "limit 1e300 did not invert a front");
  check(same_as(lim0, off), "limit 0 differs from no inversion");
  check(same_as(lim1, off), "limit 1 differs from no inversion");
  check(same_as(all2, all), "refactorization with 1e300 differs");
  check(same_as(dis, off), "refactorization with the option disabled "
        "differs from no inversion");
  bool some_inv = !identical(def.x1, off.x1),
    some_kept = !identical(def.x1, all.x1);
  if (!rank)
    cout << "# " << name << ", default limit: "
         << (some_inv ? "some" : "no") << " front inverted, "
         << (some_kept ? "some" : "no") << " front kept (depends on the "
         << "separator tree, not checked)" << endl;
  return nfail;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const double eps = 1e-9;
  CurlCurl cc(12, eps);
  int nfail = run<double>(cc, eps, "double") +
    run<std::complex<double>>(cc, eps, "complex<double>");
  MPI_Allreduce(MPI_IN_PLACE, &nfail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (!rank && !nfail) cout << "# passed" << endl;
  MPI_Finalize();
  return nfail ? 1 : 0;
}
