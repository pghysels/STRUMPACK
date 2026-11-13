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

// Compares the equilibration of a block-row distributed matrix
// (CSRMatrixMPI::equilibration) with the equilibration of the same
// matrix on one process (CSRMatrix::equilibration): the type, the
// row and column scalings that the type uses, the condition numbers,
// Amax, and the index of a zero row or column must be identical. The
// matrices are nonsymmetric 5-point stencils plus random long-range
// couplings, symmetrically permuted at random (many columns of other
// processes), in real and complex arithmetic, with 32 and 64 bit
// indices. Row and/or column scalings by powers of two give the four
// equilibration types. The rows are distributed in equal blocks, and
// in unequal blocks with an empty process. Each matrix is then solved
// with SparseSolverMPIDist.
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
using namespace std;

#include "StrumpackSparseSolverMPIDist.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/CSRMatrixMPI.hpp"
using namespace strumpack;

#define RESIDUAL_TOLERANCE 1e-10

enum class Scaling { NONE, ROW, COLUMN, BOTH, ZERO_ROW, ZERO_COLUMN };

std::string name(Scaling s) {
  switch (s) {
  case Scaling::NONE: return "no scaling";
  case Scaling::ROW: return "row scaling";
  case Scaling::COLUMN: return "column scaling";
  case Scaling::BOTH: return "row and column scaling";
  case Scaling::ZERO_ROW: return "a zero row";
  case Scaling::ZERO_COLUMN: return "a zero column";
  }
  return "";
}

template<typename scalar_t> scalar_t random_value(std::mt19937& g) {
  std::uniform_real_distribution<double> u(0.5, 1.);
  return scalar_t(u(g));
}
template<> std::complex<double> random_value(std::mt19937& g) {
  std::uniform_real_distribution<double> u(0.5, 1.);
  auto a = u(g);
  return std::polar(a, 6.28 * u(g));
}

// The same global matrix on every process.
template<typename scalar_t,typename integer_t>
struct Matrix {
  integer_t n;
  std::vector<integer_t> ptr, ind;
  std::vector<scalar_t> val;

  Matrix(int k, Scaling s, unsigned seed) : n(k*k) {
    std::mt19937 g(seed);
    std::vector<std::vector<std::pair<integer_t,scalar_t>>> rows(n);
    auto add = [&](integer_t i, integer_t j, scalar_t v) {
      for (auto& e : rows[i])
        if (e.first == j) { e.second += v; return; }
      rows[i].emplace_back(j, v);
    };
    for (int y=0; y<k; y++)
      for (int x=0; x<k; x++) {
        integer_t i = x + k*y;
        add(i, i, scalar_t(2. + std::abs(random_value<scalar_t>(g))));
        if (x > 0) add(i, i-1, -random_value<scalar_t>(g));
        if (x < k-1) add(i, i+1, -random_value<scalar_t>(g));
        if (y > 0) add(i, i-k, -random_value<scalar_t>(g));
        if (y < k-1) add(i, i+k, -random_value<scalar_t>(g));
      }
    std::uniform_int_distribution<integer_t> ri(0, n-1);
    for (integer_t c=0; c<n/4; c++) {
      auto i = ri(g), j = ri(g);
      if (i == j) continue;
      add(i, j, scalar_t(.1) * random_value<scalar_t>(g));
      add(j, i, scalar_t(.1) * random_value<scalar_t>(g));
    }
    // row and column scalings by powers of two
    std::uniform_int_distribution<int> re(-8, 8);
    std::vector<double> dr(n, 1.), dc(n, 1.);
    if (s == Scaling::ROW || s == Scaling::BOTH)
      for (auto& d : dr) d = std::ldexp(1., re(g));
    if (s == Scaling::COLUMN || s == Scaling::BOTH)
      for (integer_t j=0; j<n; j+=7) dc[j] = std::ldexp(1., -10);
    // random symmetric permutation
    std::vector<integer_t> p(n), ip(n);
    std::iota(p.begin(), p.end(), 0);
    std::shuffle(p.begin(), p.end(), g);
    for (integer_t i=0; i<n; i++) ip[p[i]] = i;
    ptr.assign(1, 0);
    for (integer_t i=0; i<n; i++) {
      for (auto& e : rows[p[i]]) {
        ind.push_back(ip[e.first]);
        val.push_back(e.second * scalar_t(dr[p[i]] * dc[e.first]));
      }
      ptr.push_back(ind.size());
    }
    if (s == Scaling::ZERO_ROW)
      for (auto j=ptr[n/3]; j<ptr[n/3+1]; j++) val[j] = 0.;
    if (s == Scaling::ZERO_COLUMN) {
      for (integer_t i=0; i<n; i++)
        for (auto j=ptr[i]; j<ptr[i+1]; j++)
          if (ind[j] == n/2) val[j] = 0.;
    }
  }
};

template<typename scalar_t,typename integer_t> int
test(const MPIComm& comm, int k, Scaling s, bool empty, unsigned seed,
     const std::string& type) {
  using real_t = typename RealType<scalar_t>::value_type;
  int P = comm.size(), rank = comm.rank();
  Matrix<scalar_t,integer_t> A(k, s, seed);
  auto n = A.n;
  std::vector<integer_t> dist(P+1);
  for (int p=0; p<=P; p++) dist[p] = (long long)(n) * p / P;
  if (empty && P > 1) {
    // process e has no rows, the following ones share its rows
    int e = (P-1) / 2;
    dist[e+1] = dist[e];
    for (int p=e+2; p<P; p++)
      dist[p] = dist[e] + (long long)(n - dist[e]) * (p-e-1) / (P-e-1);
  }
  integer_t lo = dist[rank], m = dist[rank+1] - lo;
  std::vector<integer_t> lptr(m+1);
  for (integer_t r=0; r<=m; r++) lptr[r] = A.ptr[lo+r] - A.ptr[lo];
  std::vector<integer_t> lind(A.ind.begin() + A.ptr[lo],
                              A.ind.begin() + A.ptr[lo+m]);
  std::vector<scalar_t> lval(A.val.begin() + A.ptr[lo],
                             A.val.begin() + A.ptr[lo+m]);

  CSRMatrix<scalar_t,integer_t> Aseq
    (n, A.ptr.data(), A.ind.data(), A.val.data(), true);
  CSRMatrixMPI<scalar_t,integer_t> Adist
    (m, lptr.data(), lind.data(), lval.data(), dist.data(), comm, true);
  auto eseq = Aseq.equilibration();
  auto edist = Adist.equilibration();

  std::string what = type + ", " + std::to_string(n) + " rows, " + name(s) +
    (empty ? ", a process without rows" : "");
  // For a zero row, the distributed info is the MIN over the
  // processes, 0 on those without a zero row: only compared on one
  // process (the equilibration is not applied in either case).
  int ok = edist.info == eseq.info ||
    (s == Scaling::ZERO_ROW && P > 1 && edist.info == 0);
  if (!eseq.info) {
    ok = ok && edist.type == eseq.type && edist.rcond == eseq.rcond &&
      edist.ccond == eseq.ccond && edist.Amax == eseq.Amax;
    bool row = eseq.type == EquilibrationType::ROW ||
      eseq.type == EquilibrationType::BOTH;
    bool col = eseq.type == EquilibrationType::COLUMN ||
      eseq.type == EquilibrationType::BOTH;
    // R is local, C global; both are only kept when used
    ok = ok && edist.R.size() == std::size_t(row ? m : 0) &&
      edist.C.size() == std::size_t(col ? n : 0);
    if (ok && row)
      ok = std::equal(edist.R.begin(), edist.R.end(), eseq.R.begin() + lo);
    if (ok && col) ok = edist.C == eseq.C;
  }
  ok = comm.all_reduce(ok, MPI_MIN);
  EquilibrationType expected = EquilibrationType::NONE;
  if (s == Scaling::ROW) expected = EquilibrationType::ROW;
  if (s == Scaling::COLUMN) expected = EquilibrationType::COLUMN;
  if (s == Scaling::BOTH) expected = EquilibrationType::BOTH;
  if (s == Scaling::ZERO_ROW) ok = ok && eseq.info == n/3 + 1;
  else if (s == Scaling::ZERO_COLUMN) ok = ok && eseq.info == n + n/2 + 1;
  else ok = ok && !eseq.info && eseq.type == expected;

  real_t res = 0.;
  if (!eseq.info) {
    SparseSolverMPIDist<scalar_t,integer_t> sp(comm.comm(), false);
    sp.options().set_matching(MatchingJob::NONE);
    sp.set_distributed_csr_matrix
      (m, lptr.data(), lind.data(), lval.data(), dist.data(), true);
    std::vector<scalar_t> b(m, scalar_t(1.)), x(m);
    if (sp.solve(b.data(), x.data()) != ReturnCode::SUCCESS) ok = 0;
    res = Adist.max_scaled_residual(x.data(), b.data());
    if (!(res <= RESIDUAL_TOLERANCE)) ok = 0;
    ok = comm.all_reduce(ok, MPI_MIN);
  }
  if (comm.is_root()) {
    cout << "# " << (ok ? "OK" : "FAILED") << ": " << what << ": type "
         << char(edist.type) << ", info " << edist.info;
    if (!eseq.info) cout << ", residual " << res;
    cout << endl;
  }
  return ok ? 0 : 1;
}

template<typename scalar_t,typename integer_t> int
run(const MPIComm& comm, const std::string& type) {
  int nfail = 0;
  unsigned seed = 1;
  for (auto s : {Scaling::NONE, Scaling::ROW, Scaling::COLUMN, Scaling::BOTH,
                 Scaling::ZERO_ROW, Scaling::ZERO_COLUMN})
    for (bool empty : {false, true})
      nfail += test<scalar_t,integer_t>(comm, 20, s, empty, seed++, type);
  return nfail;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int nfail = 0;
  {
    MPIComm world;
    nfail += run<double,int>(world, "real");
    nfail += run<std::complex<double>,int>(world, "complex");
    nfail += run<double,long long int>(world, "real, 64 bit indices");
    if (world.is_root())
      cout << "# distributed equilibration: " << nfail << " failed" << endl;
  }
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return nfail ? 1 : 0;
}
