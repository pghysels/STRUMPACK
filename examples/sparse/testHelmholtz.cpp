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

#include <iostream>
#include <complex>
#include <algorithm>

#include "StrumpackParameters.hpp"
#include "StrumpackFortranCInterface.h"
#include "StrumpackSparseSolverMPIDist.hpp"
#include "sparse/CSRMatrix.hpp"

#include "misc/TaskTimer.hpp"

using namespace strumpack;

extern "C" {
  void STRUMPACK_FC_GLOBAL_(genmatrix3d_anal,GENMATRIX3D_ANAL)
    (void*,void*,void*,void*,void*,void*,void*,void*,void*,void*);
  void STRUMPACK_FC_GLOBAL(genmatrix3d,GENMATRIX3D)
    (void*,void*,void*,void*,void*,void*,void*,void*,
     void*,void*,void*,void*,void*);
}

/**
 * Create 3D nx x nx x nx Helmholtz problem.
 */
template<typename realt>
CSRMatrixMPI<std::complex<realt>,std::int64_t> Helmholtz3D(std::int64_t nx) {
  char datafile[] = "void";
  std::int64_t fromfile = 0, npml = 8, nnz, n;
  std::int64_t nx_ex = nx;
  nx = std::max(std::int64_t(1), nx - 2 * npml);
  MPIComm c;
  std::int64_t rank = c.rank(), P = c.size();
  std::int64_t n_local = std::round(std::floor(float(nx_ex) / P));
  std::int64_t remainder = nx_ex%P, low_f, high_f;
  if (rank+1 <= remainder) {
    high_f = (rank+1)*(n_local+1);
    low_f = high_f - (n_local+1) + 1;
  } else {
    high_f = remainder*(n_local+1)+(rank+1-remainder)*n_local;
    low_f = high_f - (n_local) + 1;
  }
  n_local = high_f - low_f + 1;
  STRUMPACK_FC_GLOBAL(genmatrix3d_anal,GENMATRIX3D_ANAL)
    (&nx, &nx, &nx, &n_local, &npml, &n, &nnz, &fromfile, datafile, &rank);

  std::vector<std::int64_t> rowind(nnz), colind(nnz);
  std::vector<std::complex<float>> val(nnz);
  STRUMPACK_FC_GLOBAL(genmatrix3d,GENMATRIX3D)
    (rowind.data(), colind.data(), val.data(), &nx, &nx, &nx,
     &low_f, &high_f, &npml, &nnz, &fromfile, datafile, &rank);

  std::int64_t is = nnz ? rowind[0] : 0, ie = nnz ? rowind[nnz-1] : 0;
  // long int lrows = (nx+2*npml)*(nx+2*npml)*(nx+2*npml);
  long int lrows = nnz ? ie-is+1 : 0;
  std::vector<std::int64_t> ind_loc(nnz), ptr_loc(lrows+1), dist(P+1);
  std::vector<std::complex<realt>> val_loc(nnz);

  for (std::int64_t i=0; i<nnz; i++) {
    ind_loc[i] = colind[i] - 1;
    val_loc[i] = val[i];
  }
  ptr_loc[0] = 0;
  for (std::int64_t i=0; i<nnz-1; i++)
    if (rowind[i] != rowind[i+1])
      ptr_loc[rowind[i]-is+1] = i + 1;
  ptr_loc[lrows] = nnz;

  dist[0] = 0;
  dist[rank+1] = lrows;
  c.all_gather(dist.data()+1, 1);
  for (int p=0; p<P; p++) dist[p+1] += dist[p];

  return CSRMatrixMPI<std::complex<realt>,std::int64_t>
    (lrows, ptr_loc.data(), ind_loc.data(), val_loc.data(),
     dist.data(), MPI_COMM_WORLD, false);
}

int main(int argc, char* argv[]) {
  using realt = double;
  using scalart = std::complex<realt>;

  int thread_level;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (thread_level != MPI_THREAD_MULTIPLE && rank == 0)
    std::cout << "MPI implementation does not support MPI_THREAD_MULTIPLE"
              << std::endl;
  {
    std::int64_t nx = 20;
    if (argc < 2) {
      std::cout << "Solve a 3D nx^3 Helmholtz problem." << std::endl
                << " usage: ./main nx" << std::endl
                << std::endl;
      return 1;
    }
    nx = std::stof(argv[1]);

    SparseSolverMPIDist<scalart,std::int64_t> spss(MPI_COMM_WORLD);
    spss.options().set_matching(MatchingJob::NONE);
    spss.options().set_reordering_method(ReorderingStrategy::GEOMETRIC);
    spss.options().set_from_command_line(argc, argv);

    auto A = Helmholtz3D<realt>(nx);
    spss.set_matrix(A);
    if (spss.reorder(nx, nx, nx) != ReturnCode::SUCCESS) {
      std::cout << "problem with reordering of the matrix." << std::endl;
      return 1;
    }
    if (spss.factor() != ReturnCode::SUCCESS) {
      std::cout << "problem during factorization of the matrix." << std::endl;
      return 1;
    }

    auto N = A.size();
    auto n_local = A.local_rows();
    std::vector<scalart> b(n_local), x(n_local);

#if 1
    std::vector<scalart> x_exact(n_local, scalart(1.));
    A.spmv(x_exact.data(), b.data());
#else
    MPIComm c;
    auto rank = c.rank();
    // pick 2 sources in the domain
    std::int64_t sources[2] =
      {nx/2 + nx * (nx/2) + nx*nx * (nx/3),
       nx/2 + nx * (2*nx/3) + nx*nx * (nx/2)};
    auto begin_row = A.dist()[rank];
    auto end_row = A.dist()[rank+1];
    for (auto i : sources)
      if (i >= begin_row && i < end_row)
        b[i - begin_row] = 1.;
#endif

    spss.solve(b.data(), x.data());

    auto scaled_res = A.max_scaled_residual(x.data(), b.data());
    if (!rank)
      std::cout << "# COMPONENTWISE SCALED RESIDUAL = "
                << scaled_res << std::endl;

    strumpack::blas::axpy
      (n_local, std::complex<realt>(-1.), x_exact.data(), 1, x.data(), 1);
    auto nrm_error = norm2(x, MPIComm());
    auto nrm_x_exact = norm2(x_exact, MPIComm());
    if (!rank)
      std::cout << "# RELATIVE ERROR = "
                << (nrm_error/nrm_x_exact) << std::endl;
  }
  TimerList::Finalize();
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}

