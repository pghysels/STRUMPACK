// #define CHECK_ERROR 0
#define CHECK_ERROR_RANDOMIZED 1

#include <cmath>
#include <iostream>
using namespace std;

#include "DistributedMatrix.hpp"
#include "HSS/HSSMatrixMPI.hpp"
using namespace strumpack;
using namespace strumpack::HSS;
using DistM_t = DistributedMatrix<double>;
using DenseM_t = DenseMatrix<double>;

class IUV {
public:
  double _alpha = 0.;
  double _beta = 0.;
  DistM_t _U, _V;
  DenseM_t _U1d,_V1d;
  HSSMatrixMPI<double>* _H;
  IUV() {}
  IUV(int ctxt, double alpha, double beta, int m, int rank,
      int decay_val) : _alpha(alpha), _beta(beta) {
    _U = DistM_t(ctxt, m, rank);
    _V = DistM_t(ctxt, m, rank);
    _U.random();
    _V.random();
    for (int c=0; c<_V.lcols(); c++) {
      auto gc = _V.coll2g(c);
      auto tmpD = _beta * exp2(-decay_val*double(gc)/rank);
      for (int r=0; r<_V.lrows(); r++)
        _V(r, c) = _V(r, c) * tmpD;
    }
    _U1d = _U.all_gather(_U.ctxt());
    _V1d = _V.all_gather(_V.ctxt());
  }

  void operator()(DistM_t& R, DistM_t& Sr, DistM_t& Sc) {
    DistM_t tmp(_U.ctxt(), _U.cols(), R.cols());
    gemm(Trans::C, Trans::N, 1., _V, R, 0., tmp);
    gemm(Trans::N, Trans::N, 1., _U, tmp, 0., Sr);
    Sr.scaled_add(_alpha, R);

    gemm(Trans::C, Trans::N, 1., _U, R, 0., tmp);
    gemm(Trans::N, Trans::N, 1., _V, tmp, 0., Sc);
    Sc.scaled_add(_alpha, R);
  }

  DistM_t dense() const {
    DistM_t D(_U.ctxt(), _U.rows(), _U.rows());
    D.eye();
    gemm(Trans::N, Trans::C, 1., _U, _V, _alpha, D);
    return D;
  }

  void operator()(const vector<size_t>& I,
                  const vector<size_t>& J, DistM_t& B) {
    if (!B.active()) return;
    for (size_t j=0; j<J.size(); j++)
      for (size_t i=0; i<I.size(); i++) {
        if (B.is_local(i,j)) {
          if (I[i] == J[j]) B.global(i,j) = _alpha;
          else B.global(i,j) = 0.;
          for (int k=0; k<_U.cols(); k++)
            B.global(i,j) += _U1d(I[i],k) * _V1d(J[j], k);
        }
      }
    return;
  }
};


int run(int argc, char* argv[]) {
  int m = 150;
  int rk = 10;
  double alpha;
  double beta;
  int decay_val;

  auto P = mpi_nprocs(MPI_COMM_WORLD);

  HSSOptions<double> hss_opts;
  auto usage = [&]() {
    if (!mpi_rank()) {
      cout << "# Usage:\n"
      << "#     OMP_NUM_THREADS=4 ./test1 options [HSS Options]\n";
      hss_opts.describe_options();
    }
    exit(1);
  };

  if (argc > 5) {
    m = stoi(argv[1]);
    rk = stoi(argv[2]);
    alpha = stod(argv[3]);
    beta = stod(argv[4]);
    decay_val = stoi(argv[5]);
  }
  if ((argc <= 5) || (m <= 0) || (rk <= 0)) {
    if (!mpi_rank()) {
      cout << "# matrix dimension should be a positive integer\n";
      cout << "# matrix-free rank should be a positive integer\n";
      usage();
    }
  }

  // initialize the BLACS grid
  int npcol = floor(sqrt((float)P));
  int nprow = P / npcol;
  int ctxt, dummy, prow, pcol;
  Cblacs_get(0, 0, &ctxt);
  Cblacs_gridinit(&ctxt, "C", nprow, npcol);
  Cblacs_gridinfo(ctxt, &dummy, &dummy, &prow, &pcol);
  int ctxt_all = Csys2blacs_handle(MPI_COMM_WORLD);
  Cblacs_gridinit(&ctxt_all, "R", 1, P);

  hss_opts.set_from_command_line(argc, argv);

  if (!mpi_rank()) {
    cout << endl;
    cout << "# rank  = " << rk << endl;
    cout << "# size  = " << m << endl;
    cout << "# alpha = " << alpha << endl;
    cout << "# beta = " << beta << endl;
    cout << "# decay_val = " << decay_val << endl;

    cout << "# procs = " << P << endl;
    cout << "# nprow = " << nprow << endl;
    cout << "# npcol = " << npcol << endl;

    cout << "# rel tol = " << hss_opts.rel_tol() << endl;
    cout << "# abs tol = " << hss_opts.abs_tol() << endl;
    cout << "# d0 = " << hss_opts.d0() << endl;
    cout << "# dd = " << hss_opts.dd() << endl;
  }

  IUV Amf(ctxt, alpha, beta, m, rk, decay_val);


  auto start = std::chrono::system_clock::now();
  HSSMatrixMPI<double> H(m, m, Amf, ctxt, Amf, hss_opts, MPI_COMM_WORLD);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;
  // std::chrono::duration<double> elapsed_seconds_max;
  // auto durationMax = std::chrono::system_clock::now();
  double d_elapsed_seconds = elapsed_seconds.count();
  double d_elapsed_seconds_max;
  MPI_Reduce(&d_elapsed_seconds, &d_elapsed_seconds_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // std::chrono::duration<double> elapsed_seconds = end-start;
  // std::cout << "## Compression elapsed time: " << elapsed_seconds.count() << "s\n";

  if(!mpi_rank()){
    std::cout << "## Compression elapsed time(max): " << d_elapsed_seconds_max << "s\n";
  }


  if (H.is_compressed()) {
    if (!mpi_rank()) {
      cout << "# created H matrix of dimension "
           << H.rows() << " x " << H.cols()
           << " with " << H.levels() << " levels" << endl;
      cout << "# compression succeeded!" << endl;
    }
  } else {
    if (!mpi_rank()) cout << "# compression failed!!!!!!!!" << endl;
    return 1;
  }

  auto Hrank = H.max_rank();
  auto Hmem = H.total_memory();
#if defined(CHECK_ERROR)
  DistributedMatrix<double> I(ctxt, m, m),
    A(ctxt, m, m), At(ctxt, m, m);
  I.eye();
  A.zero();
  At.zero();
  Amf(I, A, At);
  auto Amem = A.total_memory();
  auto Hdense = H.dense(ctxt);
  Hdense.scaled_add(-1., A);
  auto HnormF = Hdense.normF();
  auto AnormF = A.normF();
#else
  double Amem = m*m*sizeof(double);
#endif

#if defined(CHECK_ERROR_RANDOMIZED)
  int r = 100; // Size of matrix to test compression
  DistributedMatrix<double> norm_check(ctxt, m, r),
    A_norm_est(ctxt, m, r), At_norm_est(ctxt, m, r);
  norm_check.random();
  Amf(norm_check, A_norm_est, At_norm_est);
  At_norm_est.clear();
  auto H_norm_check = H.apply(norm_check);
  H_norm_check.scaled_add(-1., A_norm_est);
  auto Hnormcheck = H_norm_check.normF();
  auto Anormest = A_norm_est.normF();
  A_norm_est.clear();
#endif

  if (!mpi_rank()) {
    cout << "# rank(H) = " << Hrank << endl;
    cout << "# memory(A) = " << Amem/1e6 << " MB" << endl;
    cout << "# memory(H) = " << Hmem/1e6 << " MB, "
         << 100. * Hmem / Amem << "% of dense" << endl << endl;
#if defined(CHECK_ERROR)
    cout << "# relative error = ||A-H*I||_F/||A||_F = "
         << HnormF / AnormF << endl;
#endif
#if defined(CHECK_ERROR_RANDOMIZED)
    cout << "# relative error est = ||(A*R)-(H*R)||_F/||A*R||_F = "
         << Hnormcheck / Anormest << endl;
#endif
  }
  return 0;
}


int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  run(argc, argv);
  Cblacs_exit(1);
  MPI_Finalize();
}
