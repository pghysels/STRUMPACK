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
 * works, and perform publicly and display publicly. Beginning five
 * (5) years after the date permission to assert copyright is obtained
 * from the U.S. Department of Energy, and subject to any subsequent
 * five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li,
               Gustavo Chávez.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <atomic>

#include "HSS/HSSMatrixMPI.hpp"
#include "misc/TaskTimer.hpp"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

#define myscalar double
#define myreal double
#define BLACSCTXTSIZE 9

int main(int argc, char *argv[]) {

  int n = 8;
  myscalar *A=NULL;
  int descA[BLACSCTXTSIZE];

  MPI_Init(&argc, &argv);
  auto np = mpi_nprocs(MPI_COMM_WORLD);
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (!mpi_rank())
    cout << "# usage: ./DenseTestMPI n (problem size)" << endl;

  if (!mpi_rank()) {
    cout << "# Building distributed matrix" << endl;
  }

  // initialize the BLACS grid
  int INONE=-1, IZERO=0, IONE=1;
  int nb=32;
  int ctxt, dummy, myrow, mycol;
  int i, j, ii, jj;
  int locr, locc;

  int nprow=floor(sqrt((float)np));
  int npcol=np/nprow;

  scalapack::Cblacs_get(0, 0, &ctxt);
  scalapack::Cblacs_gridinit(&ctxt,"C",nprow,npcol);
  scalapack::Cblacs_gridinfo(ctxt,&nprow,&npcol,&myrow,&mycol);
  
  // cout << "[" << myid << "] ctxt    = " << ctxt    << endl;
  // cout << "[" << myid << "] np    = " << np    << endl;
  // cout << "[" << myid << "] nprow = " << nprow << endl;
  // cout << "[" << myid << "] npcol = " << npcol << endl;
  // cout << "[" << myid << "] myrow = " << myrow << endl;
  // cout << "[" << myid << "] mycol = " << mycol << endl;

  /* A is a dense n x n distributed Toeplitz matrix */
  if(myid<nprow*npcol) {
    locr=strumpack::scalapack::numroc(n,nb,myrow,IZERO,nprow);
    locc=strumpack::scalapack::numroc(n,nb,mycol,IZERO,npcol);
    A=new myscalar[locr*locc];
    dummy=std::max(1,locr);
    scalapack::descinit(descA, n, n, nb, nb, IZERO, IZERO, ctxt, dummy);

    for(i=1;i<=locr;i++)
      for(j=1;j<=locc;j++) {
        ii = indxl2g( i, nb, myrow,IZERO, nprow ) ;
        jj = indxl2g( j, nb, mycol,IZERO, npcol ) ;
        // Toeplitz matrix from Quantum Chemistry.
        myreal pi=3.1416, d=0.1;
        A[locr*(j-1)+(i-1)]=ii==jj?pow(pi,2)/6.0/pow(d,2):pow(-1.0,ii-jj)/pow((myreal)ii-jj,2)/pow(d,2);

      }
  } else {
    scalapack::descset(descA, n, n, nb, nb, IZERO, IZERO, INONE, IONE);
  }

  // TaskTimer::t_begin = GET_TIME_NOW();
  // TaskTimer timer(string("compression"), 1);

  // HSSOptions<double> hss_opts;
  // hss_opts.set_verbose(true);
  // hss_opts.set_from_command_line(argc, argv);

  // vector<double> data_train = write_from_file(filename + "_train.csv");
  // vector<double> data_test = write_from_file(filename + "_" + mode + ".csv");
  // vector<double> data_train_label =
  //     write_from_file(filename + "_train_label.csv");
  // vector<double> data_test_label =
  //     write_from_file(filename + "_" + mode + "_label.csv");

  // int n = data_train.size() / d;
  // int m = data_test.size() / d;

  // if (!mpi_rank())
  //   cout << "# matrix size = " << n << " x " << d << endl;

  // if (!mpi_rank())
  //   cout << "# Preprocessing data..." << endl;
  // timer.start();

  // HSSPartitionTree cluster_tree;
  // cluster_tree.size = n;
  // int cluster_size = hss_opts.leaf_size();

  // if (reorder == "2means") {
  //   recursive_2_means(data_train.data(), n, d, cluster_size, cluster_tree,
  //                     data_train_label.data());
  // } else if (reorder == "kd") {
  //   recursive_kd(data_train.data(), n, d, cluster_size, cluster_tree,
  //                data_train_label.data());
  // } else if (reorder == "pca") {
  //   recursive_pca(data_train.data(), n, d, cluster_size, cluster_tree,
  //                 data_train_label.data());
  // }

  // if (!mpi_rank())
  //   cout << "# Preprocessing took " << timer.elapsed() << endl;

  // if (!mpi_rank())
  //   cout << "# HSS compression .. " << endl;
  // timer.start();

  // HSSMatrixMPI<double>* K = nullptr;

  // KernelMPI kernel_matrix
  //   (data_train, d, h, lambda, hss_opts,
  //    ctxt_all, nprow, npcol, nmpi, ninc, ACA);

  // auto f0_compress = strumpack::params::flops;
  // if (reorder != "natural")
  //   K = new HSSMatrixMPI<double>
  //     (cluster_tree, kernel_matrix, ctxt, kernel_matrix,
  //      hss_opts, MPI_COMM_WORLD);
  // else
  //   K = new HSSMatrixMPI<double>
  //     (n, n, kernel_matrix, ctxt, kernel_matrix,
  //      hss_opts, MPI_COMM_WORLD);


  // if (K->is_compressed()) {
  //   // reduction over all processors
  //   const auto max_rank = K->max_rank();
  //   const auto total_memory = K->total_memory();
  //   if (!mpi_rank())
  //     cout << "# created K matrix of dimension "
  //          << K->rows() << " x " << K->cols()
  //          << " with " << K->levels() << " levels" << endl
  //          << "# compression succeeded!" << endl
  //          << "# rank(K) = " << max_rank << endl
  //          << "# memory(K) = " << total_memory / 1e6 << " MB " << endl;
  // } else {
  //   if (!mpi_rank())
  //     cout << "# compression failed!!!!!!!!" << endl;
  //   return 1;
  // }

  // if (!mpi_rank())
  //   cout << "#HSS compression took " << timer.elapsed() << endl;
  // total_time += timer.elapsed();

  // auto total_flops_compress = Allreduce
  //   (strumpack::params::flops - f0_compress, MPI_SUM, MPI_COMM_WORLD);
  // if (!mpi_rank())
  //   cout << "# compression flops = " << total_flops_compress << endl;

  // // Starting factorization
  // if (!mpi_rank())
  //   cout << "factorization start" << endl;
  // timer.start();
  // auto f0_factor = strumpack::params::flops;
  // auto ULV = K->factor();
  // if (!mpi_rank())
  //   cout << "# factorization time = " << timer.elapsed() << endl;
  // total_time += timer.elapsed();
  // auto total_flops_factor = Allreduce
  //   (strumpack::params::flops - f0_factor, MPI_SUM, MPI_COMM_WORLD);
  // if (!mpi_rank())
  //   cout << "# factorization flops = " << total_flops_factor << endl;

  // DenseMatrix<double> B(n, 1, &data_train_label[0], n);
  // DenseMatrix<double> weights(B);
  // DistributedMatrix<double> Bdist(ctxt, B);
  // DistributedMatrix<double> wdist(ctxt, weights);

  // if (!mpi_rank())
  //   cout << "solve start" << endl;
  // timer.start();
  // K->solve(ULV, wdist);
  // if (!mpi_rank())
  //   cout << "# solve time = " << timer.elapsed() << endl;
  // total_time += timer.elapsed();
  // if (!mpi_rank())
  //   cout << "# total time (comp + fact): " << total_time << endl;

  // auto Bcheck = K->apply(wdist);

  // Bcheck.scaled_add(-1., Bdist);
  // auto Bchecknorm = Bcheck.normF() / Bdist.normF();
  // if (!mpi_rank())
  //   cout << "# relative error = ||B-H*(H\\B)||_F/||B||_F = "
  //        << Bchecknorm << endl;

  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}
