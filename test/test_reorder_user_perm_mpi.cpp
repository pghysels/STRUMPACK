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

// Factors and solves a 5-point Laplacian with SparseSolverMPIDist
// after reorder(p), i.e. with a permutation given by the user (the
// same one on every process), for an identity, a reversed and a
// random permutation. Checks first that set_permutation gives the
// same permutation on all processes, then the residual of the solve.
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

#include "StrumpackSparseSolverMPIDist.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/CSRMatrixMPI.hpp"
#include "sparse/ordering/MatrixReorderingMPI.hpp"
using namespace strumpack;

#define RESIDUAL_TOLERANCE 1e-12

int main(int argc, char* argv[]) {
  int thread_level;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
  int nfail = 0;
  {
    MPIComm world;
    const int k = 40, n = k*k;
    CSRMatrix<double,int> A;
    if (world.is_root()) {
      std::vector<int> ptr(n+1), ind;
      std::vector<double> val;
      for (int y=0, r=0; y<k; y++)
        for (int x=0; x<k; x++, r++) {
          auto add = [&](int xx, int yy, double v) {
            if (xx >= 0 && xx < k && yy >= 0 && yy < k) {
              ind.push_back(xx + k*yy);
              val.push_back(v);
            }
          };
          add(x, y-1, -1.); add(x-1, y, -1.); add(x, y, 4.);
          add(x+1, y, -1.); add(x, y+1, -1.);
          ptr[r+1] = ind.size();
        }
      A = CSRMatrix<double,int>(n, ptr.data(), ind.data(), val.data());
    }
    CSRMatrixMPI<double,int> Adist(&A, world.comm(), true);
    auto nloc = Adist.local_rows();
    std::vector<double> b(nloc, 1.), x(nloc);
    for (int t=0; t<3; t++) {
      std::vector<int> p(n);
      std::iota(p.begin(), p.end(), 0);
      if (t == 1) std::reverse(p.begin(), p.end());
      if (t == 2) {
        // Fisher-Yates on the mt19937 output, the same permutation
        // with every standard library (std::shuffle is not)
        std::mt19937 gen(7);
        for (int i=n-1; i>0; i--) std::swap(p[i], p[gen() % (i+1)]);
      }
      const char* name = t == 0 ? "identity" : t == 1 ? "reversed" : "random";
      {
        // set_permutation changes the permutation (postorder of the
        // elimination tree), it must do so on every process
        SPOptions<double> opts;
        opts.set_verbose(false);
        MatrixReorderingMPI<double,int> nd(n, world);
        nd.set_permutation(opts, Adist, p.data(), 0);
        auto perm0 = nd.perm();
        world.broadcast(perm0);
        if (world.all_reduce(int(perm0 != nd.perm()), MPI_MAX)) {
          if (world.is_root())
            cout << "# permutation " << name << ": set_permutation gives"
                 << " different permutations on the processes  FAILED"
                 << endl;
          nfail++;
          continue; // the factorization would use inconsistent orderings
        }
      }
      StrumpackSparseSolverMPIDist<double,int> sp(world.comm(), false);
      sp.options().set_matching(MatchingJob::NONE);
      sp.set_matrix(Adist);
      auto ret = sp.reorder(p.data());
      if (ret == ReturnCode::SUCCESS) ret = sp.factor();
      if (ret == ReturnCode::SUCCESS) ret = sp.solve(b.data(), x.data());
      auto res = Adist.max_scaled_residual(x.data(), b.data());
      bool ok = ret == ReturnCode::SUCCESS && res < RESIDUAL_TOLERANCE;
      if (world.is_root())
        cout << "# permutation " << name << ": ret " << int(ret)
             << ", componentwise scaled residual " << res
             << (ok ? "" : "  FAILED") << endl;
      if (!ok) nfail++;
    }
  }
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return nfail ? 1 : 0;
}
