/* Shared ButterflyPACK process-tree lifetime and compatibility regression. */
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "HODLR/ButterflyMatrix.hpp"

using namespace strumpack;

void stage(const char* message) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (!rank) std::cerr << message << std::endl;
}

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template<typename scalar_t> void lifetime_test(const MPIComm& comm) {
  using Matrix = HODLR::HODLRMatrix<scalar_t>;
  using Butterfly = HODLR::ButterflyMatrix<scalar_t>;
  HODLR::HODLROptions<scalar_t> opts;
  opts.set_verbose(false);
  opts.set_leaf_size(8);
  opts.set_rel_tol(1.e-9);
  structured::ClusterTree tree(64), other_tree(96);
  tree.refine(8);
  other_tree.refine(8);
  Matrix survivor, replacement(comm, tree, opts);
  Butterfly offdiag, offdiag_replacement;
  {
    Matrix owner(comm, tree, opts);
    Matrix child(comm, other_tree, opts, &owner);
    // Move assignment must release the old tree while keeping the shared one.
    replacement = std::move(child);
    survivor = std::move(replacement);
    offdiag = Butterfly(owner, survivor);
    offdiag_replacement = Butterfly(owner, survivor);
    offdiag = std::move(offdiag_replacement);
  }
  // The original owner is gone. Both surviving matrices still need its grids.
  typename Matrix::delem_blocks_t entries =
    [](auto& I, auto& J, auto& blocks, auto&) {
      for (std::size_t k=0; k<blocks.size(); k++)
        for (int j=0; j<blocks[k].lcols(); j++)
          for (int i=0; i<blocks[k].lrows(); i++)
            blocks[k](i,j) = scalar_t
              (I[k][blocks[k].rowl2g(i)] == J[k][blocks[k].coll2g(j)] ? 2. : 0.);
    };
  survivor.compress(entries);
  survivor.factor();
  DenseMatrix<scalar_t> rhs(survivor.lrows(), 1), solution(rhs.rows(), 1);
  rhs.fill(scalar_t(2.));
  survivor.solve(rhs, solution);
  double error = 0.;
  for (std::size_t i=0; i<solution.rows(); i++) {
    double entry_error = std::abs(solution(i,0)-scalar_t(1.));
    require(std::isfinite(entry_error), "nonfinite shared-tree solve result");
    error = std::max(error, entry_error);
  }
  error = comm.all_reduce(error, MPI_MAX);
  require(std::isfinite(error) && error < 1.e-8,
          "shared process tree did not survive the original HODLR owner");

  typename Matrix::delem_blocks_t ones =
    [](auto&, auto&, auto& blocks, auto&) {
      for (auto& block : blocks) block.fill(scalar_t(1.));
    };
  offdiag.compress(ones);
  DenseMatrix<scalar_t> x(offdiag.lcols(), 1), y(offdiag.lrows(), 1);
  x.fill(scalar_t(1.));
  offdiag.mult(Trans::N, x, y);
  error = 0.;
  for (std::size_t i=0; i<y.rows(); i++) {
    double entry_error = std::abs(y(i,0)-scalar_t(offdiag.cols()));
    require(std::isfinite(entry_error), "nonfinite shared-tree butterfly result");
    error = std::max(error, entry_error);
  }
  error = comm.all_reduce(error, MPI_MAX);
  require(std::isfinite(error) && error < 1.e-7,
          "shared process tree did not survive the original butterfly parents");
}

void compatibility_test(const MPIComm& comm) {
  using Matrix = HODLR::HODLRMatrix<double>;
  HODLR::HODLROptions<double> opts;
  opts.set_verbose(false);
  structured::ClusterTree tree(64);
  tree.refine(8);
  Matrix owner(comm, tree, opts);
  // Congruent communicators with identical rank order are compatible.
  MPIComm duplicate(comm);
  Matrix compatible(duplicate, tree, opts, &owner);
  if (comm.size() < 2) return;
  MPI_Comm reversed_raw;
  MPI_Comm_split(comm.comm(), 0, comm.size()-comm.rank(), &reversed_raw);
  {
    MPIComm reversed(reversed_raw);
    bool rejected = false;
    try {
      Matrix incompatible(reversed, tree, opts, &owner);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "accepted a process tree with reversed rank order");
    Matrix reversed_matrix(reversed, tree, opts);
    rejected = false;
    try {
      HODLR::ButterflyMatrix<double> incompatible(owner, reversed_matrix);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "accepted butterfly parents with reversed rank order");
  }
  MPI_Comm_free(&reversed_raw);
}

int main(int argc, char** argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  int result = 0;
  try {
    MPIComm world(MPI_COMM_WORLD), self(MPI_COMM_SELF);
    stage("real self lifetime test");
    lifetime_test<double>(self);
    stage("complex self lifetime test");
    lifetime_test<std::complex<double>>(self);
    stage("real distributed lifetime test");
    lifetime_test<double>(world);
    stage("compatibility test");
    compatibility_test(world);
    if (world.is_root()) std::cout << "HODLR process-tree lifetime and compatibility: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    result = 1;
    MPI_Abort(MPI_COMM_WORLD, result);
  }
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return result;
}
