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

// Compares the Schur complement onto given variables, the
// condensation of right-hand sides onto them and the expansion of
// their values to the other variables (set_schur_variables,
// get_schur_complement, schur_condense and schur_expand) with dense
// computations. The matrices are nonsymmetric 7-point stencils on a
// grid, with random row and column scalings that the equilibration
// undoes, in real and complex arithmetic. The Schur variables are a
// plane of the grid, which separates it, or random points of it, in
// random order. Each case is factored, then refactored with updated
// values. SparseSolver runs on the root process, SparseSolverMPIDist
// on all of them.
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>
using namespace std;

#include "StrumpackSparseSolver.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "StrumpackSparseSolverMPIDist.hpp"
#endif
using namespace strumpack;

#define TOLERANCE 1e-10

template<typename scalar_t> scalar_t random_value(std::mt19937& g);
template<> double random_value<double>(std::mt19937& g) {
  return std::uniform_real_distribution<double>(-1., 1.)(g);
}
template<> std::complex<double>
random_value<std::complex<double>>(std::mt19937& g) {
  double re = random_value<double>(g);
  return {re, random_value<double>(g)};
}

template<typename scalar_t> DenseMatrix<scalar_t>
random_matrix(int m, int n, std::mt19937& g) {
  DenseMatrix<scalar_t> A(m, n);
  for (int j=0; j<n; j++)
    for (int i=0; i<m; i++)
      A(i, j) = random_value<scalar_t>(g);
  return A;
}

template<typename scalar_t> double
max_abs(const DenseMatrix<scalar_t>& A) {
  double m = 0.;
  for (std::size_t j=0; j<A.cols(); j++)
    for (std::size_t i=0; i<A.rows(); i++)
      m = std::max(m, double(std::abs(A(i, j))));
  return m;
}

// A nonsymmetric 7-point stencil on a k x k x k grid, diagonally
// dominant before the row and column scalings (by 10^[-1,1] and
// 10^[-3,3]), in CSR format.
template<typename scalar_t> struct Stencil {
  int n;
  std::vector<int> ptr, ind;
  std::vector<scalar_t> val;
  Stencil(int k, std::mt19937& g) : n(k*k*k), ptr(n+1, 0) {
    std::uniform_real_distribution<double> e(-1., 1.);
    std::vector<double> r(n), c(n);
    for (auto& v : r) v = std::pow(10., e(g));
    // the wider column scalings also make the equilibration scale the
    // columns
    for (auto& v : c) v = std::pow(10., 3. * e(g));
    for (int z=0; z<k; z++)
      for (int y=0; y<k; y++)
        for (int x=0; x<k; x++) {
          int i = x + k*(y + k*z);
          for (auto d : {std::array<int,3>{0,0,-1}, {0,-1,0}, {-1,0,0},
                         {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}}) {
            int X = x+d[0], Y = y+d[1], Z = z+d[2];
            if (X < 0 || Y < 0 || Z < 0 || X >= k || Y >= k || Z >= k)
              continue;
            int j = X + k*(Y + k*Z);
            scalar_t v = random_value<scalar_t>(g);
            if (i == j) v += scalar_t(12.);
            ind.push_back(j);
            val.push_back(r[i] * v * c[j]);
          }
          ptr[i+1] = ind.size();
        }
  }
  // perturb the values by up to 10%, for a refactorization
  void update(std::mt19937& g) {
    for (auto& v : val) v *= scalar_t(1.) + 0.1 * random_value<scalar_t>(g);
  }
  DenseMatrix<scalar_t> dense() const {
    DenseMatrix<scalar_t> A(n, n);
    A.zero();
    for (int i=0; i<n; i++)
      for (int p=ptr[i]; p<ptr[i+1]; p++) A(i, ind[p]) = val[p];
    return A;
  }
};

// The dense Schur complement onto the variables s, the condensation
// g_s = b_s - A_si A_ii^-1 b_i, and the solution x, with x_s given in
// x0 and x_i = A_ii^-1 (b_i - A_is x_s).
template<typename scalar_t> struct Reference {
  DenseMatrix<scalar_t> S, g, x;
  Reference(DenseMatrix<scalar_t> A, const std::vector<int>& s,
            const DenseMatrix<scalar_t>& b, const DenseMatrix<scalar_t>& x0) {
    int n = A.rows(), ns = s.size(), ni = n - ns, d = b.cols();
    std::vector<bool> is_s(n, false);
    for (auto v : s) is_s[v] = true;
    std::vector<int> I;
    for (int v=0; v<n; v++)
      if (!is_s[v]) I.push_back(v);
    DenseMatrix<scalar_t> Aii(ni, ni), Ais(ni, ns), Asi(ns, ni),
      bi(ni, d), xs(ns, d);
    S = DenseMatrix<scalar_t>(ns, ns);
    g = DenseMatrix<scalar_t>(ns, d);
    for (int q=0; q<ni; q++) {
      for (int p=0; p<ni; p++) Aii(p, q) = A(I[p], I[q]);
      for (int p=0; p<ns; p++) Asi(p, q) = A(s[p], I[q]);
    }
    for (int q=0; q<ns; q++) {
      for (int p=0; p<ni; p++) Ais(p, q) = A(I[p], s[q]);
      for (int p=0; p<ns; p++) S(p, q) = A(s[p], s[q]);
    }
    for (int c=0; c<d; c++) {
      for (int p=0; p<ni; p++) bi(p, c) = b(I[p], c);
      for (int p=0; p<ns; p++) {
        g(p, c) = b(s[p], c);
        xs(p, c) = x0(s[p], c);
      }
    }
    auto piv = Aii.LU();
    gemm(Trans::N, Trans::N, scalar_t(-1.), Asi, Aii.solve(Ais, piv),
         scalar_t(1.), S);
    gemm(Trans::N, Trans::N, scalar_t(-1.), Asi, Aii.solve(bi, piv),
         scalar_t(1.), g);
    gemm(Trans::N, Trans::N, scalar_t(-1.), Ais, xs, scalar_t(1.), bi);
    auto xi = Aii.solve(bi, piv);
    x = DenseMatrix<scalar_t>(n, d);
    for (int c=0; c<d; c++) {
      for (int p=0; p<ni; p++) x(I[p], c) = xi(p, c);
      for (int p=0; p<ns; p++) x(s[p], c) = xs(p, c);
    }
  }
};

// Relative errors of S (on the root), g in the Schur rows, and x, for
// rows [lo, lo + g.rows()) of the problem, with g zero in the others.
template<typename scalar_t> std::array<double,3>
errors(const Reference<scalar_t>& ref, const std::vector<int>& s, int lo,
       const DenseMatrix<scalar_t>* S, const DenseMatrix<scalar_t>& g,
       const DenseMatrix<scalar_t>& x) {
  std::array<double,3> err = {0., 0., 0.};
  if (S) {
    auto D = *S;
    D.scaled_add(scalar_t(-1.), ref.S);
    err[0] = max_abs(D) / max_abs(ref.S);
  }
  std::vector<int> row(g.rows(), -1);
  for (std::size_t p=0; p<s.size(); p++)
    if (s[p] >= lo && s[p] < lo + int(g.rows())) row[s[p]-lo] = p;
  double gmax = max_abs(ref.g), xmax = max_abs(ref.x);
  for (std::size_t c=0; c<g.cols(); c++)
    for (std::size_t r=0; r<g.rows(); r++) {
      double eg = (row[r] < 0) ? std::abs(g(r, c)) :
        std::abs(g(r, c) - ref.g(row[r], c));
      err[1] = std::max(err[1], eg / gmax);
      err[2] = std::max(err[2], std::abs(x(r, c) - ref.x(lo + r, c)) / xmax);
    }
  return err;
}

int report(const std::string& name, std::array<double,3> err, bool root) {
  bool fail = !(err[0] <= TOLERANCE && err[1] <= TOLERANCE &&
                err[2] <= TOLERANCE);
  if (root)
    cout << (fail ? "# FAILED: " : "# ") << name << ": S " << err[0]
         << ", condensation " << err[1] << ", expansion " << err[2] << endl;
  return fail ? 1 : 0;
}

template<typename scalar_t> struct Case {
  std::mt19937 g;
  Stencil<scalar_t> A;
  std::vector<int> s;
  DenseMatrix<scalar_t> b, x0;
  Case(int k, bool plane, int nrhs, unsigned seed) : g(seed), A(k, g) {
    if (plane)
      for (int i=0; i<k*k; i++) s.push_back(i + k*k*(k/2));
    else {
      std::vector<int> all(A.n);
      std::iota(all.begin(), all.end(), 0);
      std::shuffle(all.begin(), all.end(), g);
      s.assign(all.begin(), all.begin() + 4*k);
    }
    b = random_matrix<scalar_t>(A.n, nrhs, g);
    x0 = random_matrix<scalar_t>(A.n, nrhs, g);
  }
};

template<typename scalar_t> int
test_sequential(Case<scalar_t> c, const std::string& name, bool pointers) {
  SparseSolver<scalar_t,int> sp(false);
  sp.options().set_matching(MatchingJob::NONE);
  sp.set_csr_matrix(c.A.n, c.A.ptr.data(), c.A.ind.data(), c.A.val.data(),
                    true);
  sp.set_schur_variables(c.s.size(), c.s.data());
  std::mt19937 g(7);
  int nfail = 0;
  for (int pass=0; pass<2; pass++) {
    if (pass) {
      c.A.update(g);
      sp.update_matrix_values(c.A.n, c.A.ptr.data(), c.A.ind.data(),
                              c.A.val.data(), true);
    }
    DenseMatrix<scalar_t> S, gs(c.b.rows(), c.b.cols()), x(c.x0);
    auto e0 = sp.get_schur_complement(S);
    auto e1 = pointers ?
      sp.schur_condense(c.b.cols(), c.b.data(), c.b.ld(), gs.data(), gs.ld()) :
      sp.schur_condense(c.b, gs);
    auto e2 = pointers ?
      sp.schur_expand(c.b.cols(), c.b.data(), c.b.ld(), x.data(), x.ld()) :
      sp.schur_expand(c.b, x);
    Reference<scalar_t> ref(c.A.dense(), c.s, c.b, c.x0);
    auto err = errors(ref, c.s, 0, &S, gs, x);
    if (e0 != ReturnCode::SUCCESS || e1 != ReturnCode::SUCCESS ||
        e2 != ReturnCode::SUCCESS) err[0] = 1.;
    nfail += report("SparseSolver, " + name + (pass ? ", refactored" : ""),
                    err, true);
  }
  return nfail;
}

#if defined(STRUMPACK_USE_MPI)
template<typename scalar_t> int
test_distributed(Case<scalar_t> c, const std::string& name,
                 const MPIComm& comm) {
  int P = comm.size(), rank = comm.rank(), n = c.A.n, d = c.b.cols();
  std::vector<int> dist(P+1);
  for (int p=0; p<=P; p++) dist[p] = (long long)(n) * p / P;
  int lo = dist[rank], m = dist[rank+1] - lo;
  SparseSolverMPIDist<scalar_t,int> sp(comm.comm(), false);
  sp.options().set_matching(MatchingJob::NONE);
  auto local = [&](std::vector<int>& lptr, std::vector<int>& lind,
                   std::vector<scalar_t>& lval) {
    auto p0 = c.A.ptr[lo], p1 = c.A.ptr[lo+m];
    lptr.resize(m+1);
    for (int r=0; r<=m; r++) lptr[r] = c.A.ptr[lo+r] - p0;
    lind.assign(c.A.ind.begin() + p0, c.A.ind.begin() + p1);
    lval.assign(c.A.val.begin() + p0, c.A.val.begin() + p1);
  };
  std::vector<int> lptr, lind;
  std::vector<scalar_t> lval;
  local(lptr, lind, lval);
  sp.set_distributed_csr_matrix(m, lptr.data(), lind.data(), lval.data(),
                                dist.data(), true);
  sp.set_schur_variables(c.s.size(), c.s.data());
  DenseMatrix<scalar_t> b(m, d), x0(m, d);
  b.copy(c.b, lo, 0);
  x0.copy(c.x0, lo, 0);
  std::mt19937 g(7);
  int nfail = 0;
  for (int pass=0; pass<2; pass++) {
    if (pass) {
      c.A.update(g);
      local(lptr, lind, lval);
      sp.update_matrix_values(m, lptr.data(), lind.data(), lval.data(),
                              dist.data(), true);
    }
    DenseMatrix<scalar_t> S, gs(m, d), x(x0);
    auto e0 = sp.get_schur_complement(S);
    auto e1 = sp.schur_condense(b, gs);
    auto e2 = sp.schur_expand(b, x);
    Reference<scalar_t> ref(c.A.dense(), c.s, c.b, c.x0);
    auto err = errors(ref, c.s, lo, comm.is_root() ? &S : nullptr, gs, x);
    if (e0 != ReturnCode::SUCCESS || e1 != ReturnCode::SUCCESS ||
        e2 != ReturnCode::SUCCESS) err[0] = 1.;
    comm.all_reduce(err.data(), 3, MPI_MAX);
    nfail += report("SparseSolverMPIDist on " + std::to_string(P) +
                    " processes, " + name + (pass ? ", refactored" : ""),
                    err, comm.is_root());
  }
  return nfail;
}
#endif

// Uses of Schur variables that are rejected.
int test_errors() {
  Case<double> c(6, true, 1, 3);
  auto solver = [&]() {
    auto sp = std::make_unique<SparseSolver<double,int>>(false, false);
    sp->options().set_matching(MatchingJob::NONE);
    sp->set_csr_matrix(c.A.n, c.A.ptr.data(), c.A.ind.data(),
                       c.A.val.data(), true);
    return sp;
  };
  int nfail = 0;
  auto expect = [&](const std::string& what, ReturnCode e) {
    if (e != ReturnCode::SCHUR_ERROR) {
      nfail++;
      cout << "# FAILED: " << what << " returned " << e << endl;
    }
  };
  {
    auto sp = solver();
    sp->set_schur_variables(c.s.size(), c.s.data());
    DenseMatrix<double> x(c.b.rows(), 1);
    expect("solve with Schur variables", sp->solve(c.b, x));
    expect("solve with Schur variables", sp->solve(c.b.data(), x.data()));
    expect("solve with Schur variables",
           sp->solve(1, c.b.data(), c.b.ld(), x.data(), x.ld()));
    int neg, zero, pos;
    expect("inertia with Schur variables", sp->inertia(neg, zero, pos));
  }
  {
    auto sp = solver();
    sp->factor();
    expect("Schur variables set after the factorization",
           sp->set_schur_variables(c.s.size(), c.s.data()));
  }
  {
    auto sp = solver();
    sp->options().set_matching(MatchingJob::MAX_CARDINALITY);
    sp->set_schur_variables(c.s.size(), c.s.data());
    expect("Schur variables with matching", sp->factor());
  }
  {
    auto sp = solver();
    sp->options().set_reordering_method(ReorderingStrategy::GEOMETRIC);
    sp->set_schur_variables(c.s.size(), c.s.data());
    expect("Schur variables with the geometric reordering",
           sp->reorder(6, 6, 6));
  }
  {
    auto sp = solver();
    sp->options().set_compression(CompressionType::BLR);
    sp->set_schur_variables(c.s.size(), c.s.data());
    expect("Schur variables with compression", sp->factor());
  }
  {
    auto sp = solver();
    std::vector<int> p(c.A.n);
    std::iota(p.begin(), p.end(), 0);
    sp->set_schur_variables(c.s.size(), c.s.data());
    expect("Schur variables with a given permutation", sp->reorder(p.data()));
  }
  for (auto s : {std::vector<int>{1, 2, 1}, std::vector<int>{0, c.A.n}}) {
    auto sp = solver();
    sp->set_schur_variables(s.size(), s.data());
    expect("repeated or out-of-range Schur variables", sp->factor());
  }
  {
    auto sp = solver();
    std::vector<int> s(c.A.n);
    std::iota(s.begin(), s.end(), 0);
    sp->set_schur_variables(s.size(), s.data());
    expect("all variables as Schur variables", sp->factor());
  }
#if defined(STRUMPACK_USE_MPI)
  {
    SparseSolverMPIDist<double,int> sp(MPI_COMM_SELF, false);
    std::vector<int> dist = {0, c.A.n};
    sp.options().set_matching(MatchingJob::NONE);
    sp.options().set_reordering_method(ReorderingStrategy::PARMETIS);
    sp.set_distributed_csr_matrix(c.A.n, c.A.ptr.data(), c.A.ind.data(),
                                  c.A.val.data(), dist.data(), true);
    sp.set_schur_variables(c.s.size(), c.s.data());
    expect("Schur variables with a parallel reordering", sp.factor());
  }
#endif
  {
    auto sp = solver();
    DenseMatrix<double> S;
    expect("Schur complement without Schur variables",
           sp->get_schur_complement(S));
  }
  cout << "# rejected uses of Schur variables: " << nfail << " failed"
       << endl;
  return nfail;
}

template<typename scalar_t> int
run(const std::string& type, bool root
#if defined(STRUMPACK_USE_MPI)
    , const MPIComm& comm
#endif
    ) {
  int nfail = 0;
  unsigned seed = 1;
  for (bool plane : {true, false})
    for (int nrhs : {1, 3}) {
      Case<scalar_t> c(10, plane, nrhs, seed++);
      std::string name = type + ", " + (plane ? "a plane" : "random points") +
        ", " + std::to_string(nrhs) + " right-hand side" +
        (nrhs > 1 ? "s" : "");
      if (root) nfail += test_sequential(c, name, nrhs > 1);
#if defined(STRUMPACK_USE_MPI)
      nfail += test_distributed(c, name, comm);
#endif
    }
  return nfail;
}

int main(int argc, char* argv[]) {
  int nfail = 0;
#if defined(STRUMPACK_USE_MPI)
  MPI_Init(&argc, &argv);
  {
    MPIComm world;
    bool root = world.is_root();
    if (root) nfail += test_errors();
    nfail += run<double>("real", root, world);
    nfail += run<std::complex<double>>("complex", root, world);
    nfail = world.all_reduce(nfail, MPI_MAX);
  }
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
#else
  nfail += test_errors();
  nfail += run<double>("real", true);
  nfail += run<std::complex<double>>("complex", true);
#endif
  return nfail ? 1 : 0;
}
