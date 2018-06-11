// #define CHECK_ERROR 1
#define CHECK_ERROR_RANDOMIZED 1

#include <cmath>
#include <iostream>
using namespace std;

#include "dense/DistributedMatrix.hpp"
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
  IUV(int ctxt, int ctxt_all, double alpha, double beta,
      int m, int rank, int decay_val) : _alpha(alpha), _beta(beta) {
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
    _U1d = _U.all_gather(ctxt_all);
    _V1d = _V.all_gather(ctxt_all);
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

  if (!mpi_rank()) cout << "##Starting" << endl;

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
      << "#     OMP_NUM_THREADS=1 ./test_HSS_mpi_2 m rk alpha beta decay_val [STRUMPACK_Options]\n";
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
      cout << "# Usage:\n"
      << "#     Remember: m > 0 and rk > 0\n";
      usage();
    }
  }

  // initialize the BLACS grid
  int npcol = floor(sqrt((float)P));
  int nprow = P / npcol;
  int ctxt, dummy, prow, pcol;

  scalapack::Cblacs_get(0, 0, &ctxt);
  scalapack::Cblacs_gridinit(&ctxt, "C", nprow, npcol);
  scalapack::Cblacs_gridinfo(ctxt, &dummy, &dummy, &prow, &pcol);
  int ctxt_all = scalapack::Csys2blacs_handle(MPI_COMM_WORLD);
  scalapack::Cblacs_gridinit(&ctxt_all, "R", 1, P);

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

  // Initialize timer
  TaskTimer::t_begin = GET_TIME_NOW();
  TaskTimer timer(string("compression"), 1);

// # =====================
// # === Compression =====
// # =====================

  IUV Amf(ctxt, ctxt_all, alpha, beta, m, rk, decay_val);

  auto start = std::chrono::system_clock::now();
  HSSMatrixMPI<double> H(m, m, Amf, ctxt, Amf, hss_opts, MPI_COMM_WORLD);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;
  double d_elapsed_seconds = elapsed_seconds.count();
  double d_elapsed_seconds_max;
  MPI_Reduce(&d_elapsed_seconds, &d_elapsed_seconds_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
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
  double Amem = 0.0, HnormF = 0.0, AnormF = 0.0;

  // Check error by computing dense matrix of A
  if (m < 20000) {
    DistributedMatrix<double> I(ctxt, m, m),
    A(ctxt, m, m), At(ctxt, m, m);
    I.eye();
    A.zero();
    At.zero();
    Amf(I, A, At);
    Amem = A.total_memory();
    auto Hdense = H.dense(ctxt);
    Hdense.scaled_add(-1., A);
    HnormF = Hdense.normF();
    AnormF = A.normF();
    }
  else {
    Amem = m*m*sizeof(double);
  }

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
  if (m < 20000)
    cout << "# relative error = ||A-H*I||_F/||A||_F = "
         << HnormF / AnormF << endl;

#if defined(CHECK_ERROR_RANDOMIZED)
    cout << "# relative error est = ||(A*R)-(H*R)||_F/||A*R||_F = "
         << Hnormcheck / Anormest << endl;
#endif
  }

// # =====================
// # === Factorization ===
// # =====================

  if (!mpi_rank())
    cout << endl << "# Factorization..." << endl;
  timer.start();

  MPI_Barrier(MPI_COMM_WORLD);
  if (!mpi_rank()) cout << "# computing ULV factorization of HSS matrix .. " << endl;

  auto ULV = H.factor();
  if (!mpi_rank()){
    cout << "## Factorization time = " << timer.elapsed() << endl;
    cout << "# ULV.memory() = " << ULV.memory()/(1000.0*1000.0) << "MB" << endl;
  }

// # =============
// # === Solve ===
// # =============
  if (!mpi_rank()) cout << "# Solve..." << endl;

  DistributedMatrix<double> B(ctxt, m, 1);
  B.random();
  DistributedMatrix<double> X(B);

  timer.start();
    H.solve(ULV, X);
  if (!mpi_rank())
    cout << "## Solve time = " << timer.elapsed() << endl;

// // # ==================================
// // # === Checking relative residual ===
// // # ==================================

//   auto Bnorm = B.normF();

//   DistributedMatrix<double> R(B);
//   gemm(Trans::N, Trans::N, 1., A, X, 0., R);
//   DistributedMatrix<double> R2 = H.apply(X);
//   R.scaled_add(-1., B);
//   R2.scaled_add(-1., B);
//   double resnorm = R.normF();
//   double resnorm2 = R2.normF();

//   if (!mpi_rank()){
//       cout << "# relative residual = ||A*X-B||_F/||B||_F = "
//            << resnorm / Bnorm << endl;
//       cout << "# relative residual = ||H*X-B||_F/||B||_F = "
//            << resnorm2 / Bnorm << endl;
//   }

  if (!mpi_rank()) cout << "##Ending" << endl;
  return 0;


}


void print_flop_breakdown
  (float random_flops, float ID_flops, float QR_flops, float ortho_flops,
   float reduce_sample_flops, float update_sample_flops,
   float extraction_flops, float CB_sample_flops, float sparse_sample_flops,
   float ULV_factor_flops, float schur_flops, float full_rank_flops,
   float hss_solve_flops) {

    // Just root process continues
    if (mpi_rank() != 0) return;

    float sample_flops = CB_sample_flops
      + sparse_sample_flops;
    float compression_flops = random_flops
      + ID_flops + QR_flops + ortho_flops
      + reduce_sample_flops + update_sample_flops
      + extraction_flops + sample_flops;
    std::cout << std::endl;
    std::cout << "# ----- FLOP BREAKDOWN ---------------------"
              << std::endl;
    std::cout << "# compression           = "
              << compression_flops << std::endl;
    std::cout << "#    random             = "
              << random_flops << std::endl;
    std::cout << "#    ID                 = "
              << ID_flops << std::endl;
    std::cout << "#    QR                 = "
              << QR_flops << std::endl;
    std::cout << "#    ortho              = "
              << ortho_flops << std::endl;
    std::cout << "#    reduce_samples     = "
              << reduce_sample_flops << std::endl;
    std::cout << "#    update_samples     = "
              << update_sample_flops << std::endl;
    std::cout << "#    extraction         = "
              << extraction_flops << std::endl;
    std::cout << "#    sampling           = "
              << sample_flops << std::endl;
    std::cout << "#       CB_sample       = "
              << CB_sample_flops << std::endl;
    std::cout << "#       sparse_sampling = "
              << sparse_sample_flops << std::endl;
    std::cout << "# ULV_factor            = "
              << ULV_factor_flops << std::endl;
    std::cout << "# Schur                 = "
              << schur_flops << std::endl;
    std::cout << "# full_rank             = "
              << full_rank_flops << std::endl;
    std::cout << "# HSS_solve             = "
              << hss_solve_flops << std::endl;
    std::cout << "# --------------------------------------------"
              << std::endl;
    std::cout << "# total                 = "
              << (compression_flops + ULV_factor_flops +
                  schur_flops + full_rank_flops + hss_solve_flops) << std::endl;
    std::cout << "# --------------------------------------------";
    std::cout << std::endl;
}


int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // Main program execution
  int ierr = run(argc, argv);

  // Reducing flop counters
  float flops[13] = {
    float(params::random_flops.load()),
    float(params::ID_flops.load()),
    float(params::QR_flops.load()),
    float(params::ortho_flops.load()),
    float(params::reduce_sample_flops.load()),
    float(params::update_sample_flops.load()),
    float(params::extraction_flops.load()),
    float(params::CB_sample_flops.load()),
    float(params::sparse_sample_flops.load()),
    float(params::ULV_factor_flops.load()),
    float(params::schur_flops.load()),
    float(params::full_rank_flops.load()),
    float(params::hss_solve_flops.load())
  };

  float rflops[13];
  MPI_Reduce(flops, rflops, 13, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  print_flop_breakdown (rflops[0], rflops[1], rflops[2], rflops[3],
                        rflops[4], rflops[5], rflops[6], rflops[7],
                        rflops[8], rflops[9], rflops[10], rflops[11],
                        rflops[12]);
  TimerList::Finalize();
  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return ierr;
}
