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
#include <cmath>
using namespace std;

#include "structured/StructuredMatrix.hpp"
using namespace strumpack;


template<typename scalar_t> void
print_info(const MPIComm& comm,
           const structured::StructuredMatrix<scalar_t>* H,
           const structured::StructuredOptions<scalar_t>& opts) {
  if (comm.is_root())
    cout << get_name(opts.type()) << endl
         << "  - total_nonzeros(H) = " << H->nonzeros() << endl
         << "  - total_memory(H) = " << H->memory() / 1e6 << " MByte" << endl
         << "  - maximum_rank(H) = " << H->rank() << endl;
}

template<typename scalar_t> void
check_accuracy(const DistributedMatrix<scalar_t>& A,
               const structured::StructuredMatrix<scalar_t>* H) {
  DistributedMatrix<scalar_t> id(A.grid(), A.rows(), A.cols()),
    Hdense(A.grid(), A.rows(), A.cols());
  id.eye();
  H->mult(Trans::N, id, Hdense);
  auto err = Hdense.scaled_add(scalar_t(-1.), A).normF() / A.normF();
  if (A.Comm().is_root())
    cout << "  - ||A-H||_F/||A||_F = " << err << endl;
}

template<typename scalar_t> void
factor_and_solve(const MPIComm& comm, const BLACSGrid* g, int nrhs,
                 structured::StructuredMatrix<scalar_t>* H) {
  DistributedMatrix<scalar_t> B(g, H->rows(), nrhs), X(g, H->rows(), nrhs);
  X.random();
  H->mult(Trans::N, X, B);
  H->factor();
  H->solve(B);
  auto err = B.scaled_add(scalar_t(-1.), X).normF() / X.normF();
  if (comm.is_root())
    cout << "  - ||X-H\\(H*X)||_F/||X||_F = "
         << err << endl;
}


int main(int argc, char* argv[]) {
  int thread_level;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
  {
    MPIComm world;
    if (thread_level != MPI_THREAD_FUNNELED && world.is_root())
      cout << "MPI implementation does not support MPI_THREAD_FUNNELED" << endl;

    int n = 1000, nrhs = 10;
    if (argc > 1) n = stoi(argv[1]);

    structured::StructuredOptions<double> options;
    options.set_verbose(false);
    options.set_from_command_line(argc, argv);

    BLACSGrid grid(world);


    // Routine to compute individual elements of the matrix.
    auto Toeplitz =
      [](int i, int j) {
        return 1. / (1. + abs(i-j));
      };

    // routine to compute sub-block of the matrix. Often this could be
    // implemented more efficiently than computing element per element.
    auto Toeplitz_block =
      [&Toeplitz](const std::vector<std::size_t>& I,
                  const std::vector<std::size_t>& J,
                  DenseMatrix<double>& B) {
        for (std::size_t j=0; j<J.size(); j++)
          for (std::size_t i=0; i<I.size(); i++)
            B(i, j) = Toeplitz(I[i], J[j]);
      };

    // Matrix-vector multiplication routine. Ideally, user can provide a
    // faster implementation.
    // auto Amult =
    //   [&A](Trans t, const DenseMatrix<double>& R,
    //        DenseMatrix<double>& S) {
    //     gemm(t, Trans::N, double(1.), A, R, double(0.), S);
    //   };


    std::vector<structured::Type> types =
      {structured::Type::BLR,
       structured::Type::HSS,
       structured::Type::HODLR,
       structured::Type::HODBF,
       structured::Type::BUTTERFLY,
       structured::Type::LR,
       structured::Type::LOSSY,
       structured::Type::LOSSLESS};


    // create 2d block cyclicly distributed matrix, and initialize it as
    // a Toeplitz matrix
    DistributedMatrix<double> A2d(&grid, n, n, Toeplitz);

    if (world.is_root())
      cout << endl << endl
           << "dense (2DBC) " << A2d.rows() << " x " << A2d.cols()
           << " matrix" << endl
           << "  - memory(A2d) = " << A2d.memory() / 1e6 << " MByte"
           << endl << endl;

    if (world.is_root())
      cout << "===============================" << endl
           << " Compression from dense matrix" << endl
           << "===============================" << endl;
    for (auto type : types) {
      options.set_type(type);
      try {
        auto H = structured::construct_from_dense(A2d, options);
        print_info(world, H.get(), options);
        check_accuracy(A2d, H.get());
        factor_and_solve(world, &grid, nrhs, H.get());
      } catch (std::exception& e) {
        if (world.is_root())
          cout << get_name(type) << " compression failed: "
               << e.what() << endl;
      }
    }

    if (world.is_root())
      cout << endl << endl
           << "==================================" << endl
           << " Compression from matrix elements" << endl
           << "==================================" << endl;
    for (auto type : types) {
      options.set_type(type);
      try {
        // auto H = structured::StructuredMatrix<double>::
        //   construct_from_elements(world, n, n, Toeplitz, options);
        // print_info(world, H, options);
      } catch (std::exception& e) {
        if (world.is_root())
          cout << get_name(type) << " compression failed: "
               << e.what() << endl;
      }
    }

  }
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}
