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

// Tests SPOptions::enable_shared_permutation(). MatrixReorderingMPI
// must give the same permutation, inverse permutation and local
// graphs with the permutation stored once per node as with a copy
// per process, for repeated nested_dissection and set_permutation
// calls on the same object (METIS and AMD). SparseSolverMPIDist must
// give bitwise the same solutions with and without the option: first
// factorization, refactorization after update_matrix_values, a new
// matrix of another size, a given permutation, and compression enabled
// after the reordering. Real and complex, 32 and 64 bit indices.
//
// Usage: test_shared_permutation_mpi [shared|fallback]
//   shared (default): the shared memory window must be used,
//   fallback: it must not be (e.g. with OMPI_MCA_osc=^sm, where the
//   window cannot be allocated).
// With STRUMPACK_SHARED_PERM_GROUP=k, groups of k consecutive ranks
// act as nodes (several node leaders).
#include <complex>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
using namespace std;

#include "StrumpackSparseSolverMPIDist.hpp"
#include "sparse/CSRMatrixMPI.hpp"
#include "sparse/ordering/MatrixReorderingMPI.hpp"
using namespace strumpack;

#define RESIDUAL_TOLERANCE 1e-10

template<typename scalar_t> scalar_t value(std::mt19937& g) {
  std::uniform_real_distribution<double> u(0.5, 1.);
  return scalar_t(u(g));
}
template<> std::complex<double> value(std::mt19937& g) {
  std::uniform_real_distribution<double> u(0.5, 1.);
  auto a = u(g);
  return std::complex<double>(a, u(g));
}

// Nonsymmetric 5-point stencil on a k x k grid, the same on every
// process, and its block rows.
template<typename scalar_t,typename integer_t> struct Matrix {
  integer_t n;
  std::vector<integer_t> ptr, ind, dist, lptr, lind;
  std::vector<scalar_t> val, lval;
  integer_t lo, m;

  Matrix(int k, unsigned seed, const MPIComm& comm) : n(k*k) {
    std::mt19937 g(seed);
    ptr.assign(1, 0);
    for (int y=0; y<k; y++)
      for (int x=0; x<k; x++) {
        integer_t i = x + k*y;
        for (integer_t j : {i-k, i-1, i, i+1, i+k}) {
          if (j < 0 || j >= n || (j == i-1 && x == 0) ||
              (j == i+1 && x == k-1)) continue;
          ind.push_back(j);
          val.push_back(j == i ? scalar_t(4.) + value<scalar_t>(g) :
                        -value<scalar_t>(g));
        }
        ptr.push_back(ind.size());
      }
    int P = comm.size(), rank = comm.rank();
    dist.resize(P+1);
    for (int p=0; p<=P; p++) dist[p] = (long long)(n) * p / P;
    lo = dist[rank];
    m = dist[rank+1] - lo;
    local();
  }
  void local() {
    lptr.resize(m+1);
    for (integer_t r=0; r<=m; r++) lptr[r] = ptr[lo+r] - ptr[lo];
    lind.assign(ind.begin() + ptr[lo], ind.begin() + ptr[lo+m]);
    lval.assign(val.begin() + ptr[lo], val.begin() + ptr[lo+m]);
  }
  void update(std::mt19937& g) {
    for (auto& v : val) v *= scalar_t(.5) + value<scalar_t>(g);
    local();
  }
};

int nfail = 0;

void check(const MPIComm& comm, bool ok, const std::string& what) {
  ok = comm.all_reduce(int(ok), MPI_MIN);
  if (!ok) nfail++;
  if (comm.is_root())
    cout << "# " << (ok ? "OK" : "FAILED") << ": " << what << endl;
}

template<typename integer_t> bool
same_graph(const CSRGraph<integer_t>& a, const CSRGraph<integer_t>& b) {
  return a.size() == b.size() && a.edges() == b.edges() &&
    std::equal(a.ptr(), a.ptr()+a.size()+1, b.ptr()) &&
    std::equal(a.ind(), a.ind()+a.edges(), b.ind());
}

// MatrixReorderingMPI with and without the shared permutation.
template<typename scalar_t,typename integer_t> void
test_reordering(const MPIComm& comm, bool expect_shared,
                const std::string& type) {
  using MR_t = MatrixReorderingMPI<scalar_t,integer_t>;
  Matrix<scalar_t,integer_t> M(30, 1, comm);
  CSRMatrixMPI<scalar_t,integer_t> A
    (M.m, M.lptr.data(), M.lind.data(), M.lval.data(), M.dist.data(),
     comm, true);
  auto n = A.size();
  for (auto order : {ReorderingStrategy::METIS, ReorderingStrategy::AMD}) {
    SPOptions<scalar_t> opts, sopts;
    for (auto o : {&opts, &sopts}) {
      o->set_verbose(false);
      o->set_reordering_method(order);
    }
    sopts.enable_shared_permutation();
    std::string name = type + ", " + get_name(order);
    MR_t ref(n, comm);
    ref.nested_dissection(opts, A, 1, 1, 1, 1, 1);
    std::vector<int> p(ref.perm_data(), ref.perm_data()+n);
    MR_t refp(n, comm); // set_permutation builds another tree
    refp.set_permutation(opts, A, p.data(), 0);
    auto compare = [&](const MR_t& nd, const MR_t& r, bool shared,
                       const std::string& w) {
      bool ok = nd.perm().empty() == shared &&
        std::equal(r.perm_data(), r.perm_data()+n, nd.perm_data()) &&
        std::equal(r.iperm_data(), r.iperm_data()+n, nd.iperm_data()) &&
        nd.sub_graph_range == r.sub_graph_range &&
        nd.dist_sep_range == r.dist_sep_range &&
        same_graph(nd.my_sub_graph, r.my_sub_graph) &&
        same_graph(nd.my_dist_sep, r.my_dist_sep);
      check(comm, ok, name + ", " + w);
    };
    MR_t nd(n, comm);
    nd.nested_dissection(sopts, A, 1, 1, 1, 1, 1);
    compare(nd, ref, expect_shared, std::string("shared permutation") +
            (expect_shared ? "" : " (fallback)"));
    nd.nested_dissection(sopts, A, 1, 1, 1, 1, 1);
    compare(nd, ref, expect_shared, "nested_dissection again, same object");
    nd.set_permutation(sopts, A, p.data(), 0);
    compare(nd, refp, false, "then set_permutation (a copy per process)");
    nd.nested_dissection(sopts, A, 1, 1, 1, 1, 1);
    compare(nd, ref, expect_shared, "then nested_dissection");
    nd.nested_dissection(opts, A, 1, 1, 1, 1, 1);
    compare(nd, ref, false, "then nested_dissection without the option");
  }
}

template<typename scalar_t,typename integer_t> bool
solve_and_compare(const MPIComm& comm,
                  SparseSolverMPIDist<scalar_t,integer_t>& ref,
                  SparseSolverMPIDist<scalar_t,integer_t>& sp,
                  const Matrix<scalar_t,integer_t>& M,
                  double tol=RESIDUAL_TOLERANCE) {
  std::vector<scalar_t> b(M.m), x(M.m), y(M.m);
  for (integer_t r=0; r<M.m; r++) b[r] = scalar_t(1. + (M.lo+r) % 7);
  bool ok = ref.solve(b.data(), x.data()) == ReturnCode::SUCCESS &&
    sp.solve(b.data(), y.data()) == ReturnCode::SUCCESS &&
    !std::memcmp(x.data(), y.data(), M.m*sizeof(scalar_t));
  CSRMatrixMPI<scalar_t,integer_t> A
    (M.m, M.lptr.data(), M.lind.data(), M.lval.data(), M.dist.data(),
     comm, true);
  return ok && A.max_scaled_residual(y.data(), b.data()) < tol;
}

// SparseSolverMPIDist with and without the shared permutation.
template<typename scalar_t,typename integer_t> void
test_solver(const MPIComm& comm, const std::string& type) {
  using SP_t = SparseSolverMPIDist<scalar_t,integer_t>;
  SP_t ref(comm.comm(), false), sp(comm.comm(), false);
  sp.options().enable_shared_permutation();
  std::mt19937 g(3);
  Matrix<scalar_t,integer_t> M(30, 2, comm);
  auto set = [&](SP_t& s) {
    s.set_distributed_csr_matrix
      (M.m, M.lptr.data(), M.lind.data(), M.lval.data(), M.dist.data(), true);
  };
  auto update = [&](SP_t& s) {
    s.update_matrix_values
      (M.m, M.lptr.data(), M.lind.data(), M.lval.data(), M.dist.data(), true);
  };
  set(ref); set(sp);
  check(comm, solve_and_compare(comm, ref, sp, M),
        type + ", solver, bitwise identical solution");
  M.update(g);
  update(ref); update(sp);
  check(comm, solve_and_compare(comm, ref, sp, M),
        type + ", solver, update_matrix_values");
  M = Matrix<scalar_t,integer_t>(25, 4, comm);
  set(ref); set(sp);
  check(comm, solve_and_compare(comm, ref, sp, M),
        type + ", solver, new matrix of another size");
  std::vector<int> p(M.n);
  std::iota(p.begin(), p.end(), 0);
  std::reverse(p.begin(), p.end());
  bool ok = ref.reorder(p.data()) == ReturnCode::SUCCESS &&
    sp.reorder(p.data()) == ReturnCode::SUCCESS;
  check(comm, ok && solve_and_compare(comm, ref, sp, M),
        type + ", solver, given permutation");
  ok = ref.reorder() == ReturnCode::SUCCESS &&
    sp.reorder() == ReturnCode::SUCCESS;
  ref.options().set_compression(CompressionType::BLR);
  sp.options().set_compression(CompressionType::BLR);
  ref.options().set_compression_min_sep_size(16);
  sp.options().set_compression_min_sep_size(16);
  M.update(g);
  update(ref); update(sp);
  // the solve iterates to the default tolerance of the solver
  check(comm, ok && solve_and_compare(comm, ref, sp, M, 1e-4),
        type + ", solver, compression enabled after the reordering");
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  {
    MPIComm world;
    bool expect_shared = !(argc > 1 && std::string(argv[1]) == "fallback");
    test_reordering<double,int>(world, expect_shared, "real");
    test_reordering<std::complex<double>,long long int>
      (world, expect_shared, "complex, 64 bit indices");
    test_solver<double,int>(world, "real");
    test_solver<std::complex<double>,int>(world, "complex");
    test_solver<double,long long int>(world, "real, 64 bit indices");
    if (world.is_root())
      cout << "# shared permutation: " << nfail << " failed" << endl;
  }
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return nfail ? 1 : 0;
}
