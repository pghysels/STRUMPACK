/* Focused hybrid GPU interface, stride and startup regression. */
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include "HODLR/HODLRWrapper.hpp"
#include "HODLR/HODLRMatrix.hpp"
using namespace strumpack;

template<typename T> void check(const MPIComm& comm, int argc, char** argv) {
  HODLR::HODLROptions<T> opts;
  opts.set_from_command_line(argc, argv);
  opts.set_leaf_size(8);
  opts.set_rel_tol(1.e-9);
  structured::ClusterTree tree(64);
  tree.refine(8);
  HODLR::HODLRMatrix<T> A(comm, tree, opts);
  typename HODLR::HODLRMatrix<T>::delem_blocks_t entries =
    [](auto& I, auto& J, auto& blocks, auto&) {
      for (std::size_t k=0; k<blocks.size(); k++)
        for (int j=0; j<blocks[k].lcols(); j++)
          for (int i=0; i<blocks[k].lrows(); i++)
            blocks[k](i,j) = T(I[k][blocks[k].rowl2g(i)] ==
                               J[k][blocks[k].coll2g(j)] ? 2. : 0.);
    };
  A.compress(entries);
  A.factor();
  for (int nrhs : {1,3}) {
    DenseMatrix<T> parent(A.lrows()+7, nrhs), answer(A.lrows()+9, nrhs);
    parent.fill(T(-999)); answer.fill(T(-999));
    DenseMatrixWrapper<T> X(A.lrows(),nrhs,parent,2,0), Y(A.lrows(),nrhs,answer,3,0);
    for (std::size_t j=0; j<X.cols(); j++)
      for (std::size_t i=0; i<X.rows(); i++) X(i,j)=T(2.*(j+1));
    A.solve(X,Y);
    for (std::size_t j=0; j<Y.cols(); j++) {
      for (std::size_t i=0; i<Y.rows(); i++)
        if (!std::isfinite(std::abs(Y(i,j))) || std::abs(Y(i,j)-T(j+1))>1.e-9)
          throw std::runtime_error("wrong strided inverse application");
      for (std::size_t i=0; i<answer.rows(); i++)
        if ((i<3 || i>=3+Y.rows()) && answer(i,j)!=T(-999))
          throw std::runtime_error("inverse application overwrote padding");
    }
  }
  auto& stats=A.backend_statistics();
  if (stats.construction_seconds<=0 || stats.factor_seconds<=0 || stats.inverse_seconds<=0)
    throw std::runtime_error("missing execution timings");
  if (opts.use_gpu() && stats.rhs_boundary_bytes==0)
    throw std::runtime_error("missing RHS boundary accounting");
}
int main(int argc, char** argv) {
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);
  try {
    MPIComm world(MPI_COMM_WORLD), self(MPI_COMM_SELF);
    bool cold=false;
    for (int i=1;i<argc;i++) if (std::string(argv[i])=="--cold-subcomm") cold=true;
    HODLR::HODLROptions<double> opts;
    opts.set_from_command_line(argc,argv);
    if (cold) {
      if (!opts.use_gpu() || world.size()<2) throw std::runtime_error("cold mode needs GPU and multiple ranks");
      if (world.is_root()) {
        HODLR::HODLR_initialize_gpu<double>(self);
        check<double>(self,argc,argv);
      }
      MPI_Barrier(world.comm());
    } else {
      if (opts.use_gpu()) HODLR::HODLR_initialize_gpu<double>(world);
      check<double>(world,argc,argv);
      check<std::complex<double>>(world,argc,argv);
    }
    if (world.is_root()) std::cout << "HODLR hybrid interface: PASS\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
}
