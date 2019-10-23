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
#include "StrumpackParameters.hpp"
#include "StrumpackFortranCInterface.h"
#include "StrumpackSparseSolverMPIDist.hpp"
#include "sparse/CSRMatrix.hpp"
#include <iostream>
#include <complex>

using namespace strumpack;

extern "C" {
  void FC_GLOBAL_(genmatrix3d_anal,GENMATRIX3D_ANAL)
    (void*,void*,void*,void*,void*,void*,void*,void*);
  void FC_GLOBAL(genmatrix3d,GENMATRIX3D)
    (void*,void*,void*,void*,void*,void*,void*,void*,void*,void*);
}

/**
 * Create 3D nx x nx x nx Helmholtz problem.
 */
template<typename realt>
CSRMatrix<std::complex<realt>,int> Helmholtz3D(int nx) {
  char datafile[] = "void";
  int fromfile = 0, npml = 8, nnz, n;
  nx = std::max(1, nx - 2 * npml);
  FC_GLOBAL(genmatrix3d_anal,GENMATRIX3D_ANAL)
    (&nx, &nx, &nx, &npml, &n, &nnz, &fromfile, datafile);
  std::vector<std::tuple<int,int,std::complex<realt>>> rc;
  {
    std::vector<int> rowind(nnz), colind(nnz);
    std::vector<std::complex<float>> val(nnz);
    FC_GLOBAL(genmatrix3d,GENMATRIX3D)
      (colind.data(), rowind.data(), val.data(), &nx, &nx, &nx, &npml, &nnz,
       &fromfile, datafile);
    rc.resize(nnz);
    for (int i=0; i<nnz; i++)
      rc[i] = {rowind[i], colind[i], val[i]};
  }
  std::sort(rc.begin(), rc.end(),
            [](const std::tuple<int,int,std::complex<realt>>& a,
               const std::tuple<int,int,std::complex<realt>>& b) {
              return std::get<0>(a) < std::get<0>(b); });
  CSRMatrix<std::complex<realt>,int> A(n, nnz);
  for (int i=0; i<nnz; i++) {
    A.ind(i) = std::get<1>(rc[i]) - 1;
    A.val(i) = std::get<2>(rc[i]);
  }
  A.ptr(0) = 0;
  for (int i=0; i<nnz-1; i++)
    if (std::get<0>(rc[i]) != std::get<0>(rc[i+1]))
      A.ptr(std::get<0>(rc[i])) = i + 1;
  A.ptr(n) = nnz;
  return A;
}

int main(int argc, char* argv[]) {
  using realt = double;
  using scalart = std::complex<realt>;

  int thread_level;
  //MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (thread_level != MPI_THREAD_FUNNELED && rank == 0)
    std::cout << "MPI implementation does not support MPI_THREAD_FUNNELED"
              << std::endl;
  {
    int nx = 20;
    if (argc < 2) {
      std::cout << "Solve a 3D nx^3 Helmholtz problem." << std::endl
                << " usage: ./main nx" << std::endl
                << std::endl;
      return 1;
    }
    nx = std::stof(argv[1]);

    StrumpackSparseSolverMPIDist<scalart,int> spss(MPI_COMM_WORLD);
    spss.options().set_matching(MatchingJob::NONE);
    spss.options().set_reordering_method(ReorderingStrategy::GEOMETRIC);
    spss.options().set_from_command_line(argc, argv);

    CSRMatrix<scalart,int> Aseq;
    if (!rank) Aseq = Helmholtz3D<realt>(nx);
    CSRMatrixMPI<scalart,int> A(&Aseq, MPI_COMM_WORLD, true);
    Aseq = CSRMatrix<scalart,int>();

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
    std::vector<scalart> b(n_local), x(n_local),
      x_exact(n_local, scalart(1.)/std::sqrt(N));
    A.spmv(x_exact.data(), b.data());

    spss.solve(b.data(), x.data());

    auto scaled_res = A.max_scaled_residual(x.data(), b.data());
    if (!rank)
      std::cout << "# COMPONENTWISE SCALED RESIDUAL = "
                << scaled_res << std::endl;

    blas::axpy
      (n_local, std::complex<realt>(-1.), x_exact.data(), 1, x.data(), 1);
    auto nrm_error = norm2(x, MPIComm());
    auto nrm_x_exact = norm2(x_exact, MPIComm());
    if (!rank)
      std::cout << "# RELATIVE ERROR = "
                << (nrm_error/nrm_x_exact) << std::endl;
  }
  MPI_Finalize();
  return 0;
}

