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
 */
#ifndef HBS_MATRIX_COMPRESS_HPP
#define HBS_MATRIX_COMPRESS_HPP

#include "misc/RandomWrapper.hpp"
// #include "HBS/HBSMatrix.sketch.hpp"

namespace strumpack {
  namespace HBS {

    template<typename scalar_t> class WorkCompress {
    public:
      std::vector<WorkCompress<scalar_t>> c;
      // std::vector<std::size_t> Ir, Ic, Jr, Jc;
      std::pair<std::size_t,std::size_t> offset;
      int lvl = 0;
      // scalar_t U_r_max, V_r_max;

      // only needed in the new compression algorithm
      DenseMatrix<scalar_t> Rr, Rc, Sr, Sc;
      void split(const std::pair<std::size_t,std::size_t>& dim) {
        if (c.empty()) {
          c.resize(2);
          c[0].offset = offset;
          // why does this work in HSS, not here?
          // c[1].offset = offset + dim;
          c[1].offset.first = offset.first + dim.first;
          c[1].offset.second = offset.second + dim.second;
          c[0].lvl = c[1].lvl = lvl + 1;
        }
      }
    };


    template<typename scalar_t> void HBSMatrix<scalar_t>::compress
    (const DenseM_t& A, const opts_t& opts) {
      // AFunctor<scalar_t> afunc(A);

      int d_old = 0, d = opts.d0() + opts.p(), total_nnz = 0;
      auto n = this->cols();
      auto rgen = random::make_random_generator<real_t>
        (opts.random_engine(), opts.random_distribution());

      DenseM_t Rr, Rc, Sr, Sc;
      WorkCompress<scalar_t> w;
      while (!this->is_compressed()) {
        Rr.resize(n, d);   Rc.resize(n, d);
        Sr.resize(n, d);   Sc.resize(n, d);
        DenseMW_t Rr_new(n, d-d_old, Rr, 0, d_old),
          Rc_new(n, d-d_old, Rc, 0, d_old),
          Sr_new(n, d-d_old, Sr, 0, d_old),
          Sc_new(n, d-d_old, Sc, 0, d_old);
        Rr_new.random(*rgen);
        Rc_new.copy(Rr_new);
        STRUMPACK_RANDOM_FLOPS
          (rgen->flops_per_prng() * Rr_new.rows() * Rr_new.cols());
        gemm(Trans::N, Trans::N, scalar_t(1.), A, Rr_new, scalar_t(0.), Sr_new);
        gemm(Trans::C, Trans::N, scalar_t(1.), A, Rc_new, scalar_t(0.), Sc_new);

        if (opts.verbose()) {
          std::cout << "# compressing with d = " << d-opts.p()
                    << " + " << opts.p() << " (original)" << std::endl;
          if (opts.compression_sketch() == CompressionSketch::SJLT)
            std::cout << "# nnz total = " << total_nnz << std::endl;
        }
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
        compress_recursive
          (Rr, Rc, Sr, Sc, opts, w, d-d_old, this->openmp_task_depth_);
        if (!this->is_compressed()) {
          d_old = d;
          d = 2 * (d_old - opts.p()) + opts.p();
        }
      }
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    nullspace(DenseMatrix<scalar_t>& A, int k) {
      // return last r columns of a full QR factorization of A^T
      auto At = A.transpose();
      k = std::max(std::min(k, (int)(At.cols())), 0);
      int minmn = std::min(At.rows(), At.cols());
      std::unique_ptr<scalar_t[]> tau(new scalar_t[minmn]);
      auto info = blas::geqrf
        (At.rows(), At.cols(), At.data(), At.ld(), tau.get());
      if (info)
        std::cerr << "Computation of nullspace failed in QR factorization, "
                  << "info= " << info << std::endl;
      info = blas::xxgqr
        (At.rows(), minmn, minmn, At.data(), At.ld(), tau.get());
      if (info)
        std::cerr << "Computation of nullspace failed in xxGQR, "
                  << "info= " << info << std::endl;
      return DenseMatrix<scalar_t>(At.rows(), k, At, 0, At.cols()-k);
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    colspace(DenseMatrix<scalar_t>& A, int k) {
      int minmn = std::min(A.rows(), A.cols());
      std::unique_ptr<scalar_t[]> tau(new scalar_t[minmn]);
      auto info = blas::geqrf
        (A.rows(), A.cols(), A.data(), A.ld(), tau.get());
      if (info)
        std::cerr << "Computation of colspace failed in QR factorization, "
                  << "info= " << info << std::endl;
      // TODO check k
      info = blas::xxgqr(A.rows(), k, minmn, A.data(), A.ld(), tau.get());
      if (info)
        std::cerr << "Computation of colspace failed in xxGQR, "
                  << "info= " << info << std::endl;
      return DenseMatrix<scalar_t>(A.rows(), k, A, 0, 0);
    }

    template<typename scalar_t> void
    HBSMatrix<scalar_t>::compress_recursive
    (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
     const opts_t& opts, WorkCompress<scalar_t>& w, int r, int depth) {

      std::cout << "in HBSMatrix::compress_recursive" << std::endl;

      int d = Rr.cols();
      if (this->leaf()) {
        int m = rows();
        w.Rr = DenseM_t(m, d, Rr, w.offset.first, 0);
        w.Rc = DenseM_t(m, d, Rc, w.offset.first, 0);
        w.Sr = DenseM_t(m, d, Sr, w.offset.first, 0);
        w.Sc = DenseM_t(m, d, Sc, w.offset.first, 0);
      } else {
        w.split(child(0)->dims());
        bool tasked = depth < params::task_recursion_cutoff_level;
        if (tasked) {
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          child(0)->compress_recursive
            (Rr, Rc, Sr, Sc, opts, w.c[0], r, depth+1);
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          child(1)->compress_recursive
            (Rr, Rc, Sr, Sc, opts, w.c[1], r, depth+1);
#pragma omp taskwait
        } else {
          child(0)->compress_recursive(Rr, Rc, Sr, Sc, opts, w.c[0], r, depth+1);
          child(1)->compress_recursive(Rr, Rc, Sr, Sc, opts, w.c[1], r, depth+1);
        }
        if (!child(0)->is_compressed() ||
            !child(1)->is_compressed()) return;
        auto rU0 = child(0)->U_.cols();   auto rV0 = child(0)->V_.cols();
        auto rU1 = child(1)->U_.cols();   auto rV1 = child(1)->V_.cols();
        w.Rr = DenseM_t(rV0 + rV1, d);
        w.Rc = DenseM_t(rU0 + rU1, d);
        DenseMW_t
          Rr0(rV0, d, w.Rr, 0, 0),   Rc0(rU0, d, w.Rc, 0, 0),
          Rr1(rV1, d, w.Rr, rV0, 0), Rc1(rU1, d, w.Rc, rU0, 0);
        gemm(Trans::C, Trans::N, scalar_t(1.), child(0)->V_, w.c[0].Rr, scalar_t(0.), Rr0);
        gemm(Trans::C, Trans::N, scalar_t(1.), child(1)->V_, w.c[1].Rr, scalar_t(0.), Rr1);
        gemm(Trans::C, Trans::N, scalar_t(1.), child(0)->U_, w.c[0].Rc, scalar_t(0.), Rc0);
        gemm(Trans::C, Trans::N, scalar_t(1.), child(1)->U_, w.c[1].Rc, scalar_t(0.), Rc1);

        w.Sr = DenseM_t(rU0 + rU1, d);
        w.Sc = DenseM_t(rV0 + rV1, d);
        DenseMW_t
          Sr0(rU0, d, w.Sr, 0, 0),   Sc0(rV0, d, w.Sc, 0, 0),
          Sr1(rU1, d, w.Sr, rU0, 0), Sc1(rV1, d, w.Sc, rV0, 0);
        gemm(Trans::N, Trans::C, scalar_t(-1.), child(0)->D_, w.c[0].Rr, scalar_t(0.), w.c[0].Sr);
        gemm(Trans::N, Trans::C, scalar_t(-1.), child(1)->D_, w.c[1].Rr, scalar_t(0.), w.c[1].Sr);
        gemm(Trans::C, Trans::N, scalar_t(1.),  child(0)->U_, w.c[0].Sr, scalar_t(0.), Sr0);
        gemm(Trans::C, Trans::N, scalar_t(1.),  child(1)->U_, w.c[1].Sr, scalar_t(0.), Sr1);
        gemm(Trans::N, Trans::C, scalar_t(-1.), child(0)->D_, w.c[0].Rc, scalar_t(0.), w.c[0].Sc);
        gemm(Trans::N, Trans::C, scalar_t(-1.), child(1)->D_, w.c[1].Rc, scalar_t(0.), w.c[1].Sc);
        gemm(Trans::C, Trans::N, scalar_t(1.),  child(0)->V_, w.c[0].Sc, scalar_t(0.), Sc0);
        gemm(Trans::C, Trans::N, scalar_t(1.),  child(1)->V_, w.c[1].Sc, scalar_t(0.), Sc1);
      }
      if (w.lvl == 0) {
        // D = Sr pseudoinv(Rr)
        // D Rr = Sr     or     Rr* D* = Sr*
        auto Srt = w.Sr.transpose();
        blas::gels('T', w.Rr.rows(), w.Rr.cols(), Srt.cols(),
                   w.Rr.data(), w.Rr.ld(), Srt.data(), Srt.cols());
        DenseMW_t Dt(w.Rr.rows(), Srt.cols(), Srt.ptr(0, 0), w.Rr.rows());
        D_ = Dt.transpose();
      } else {
        auto P = nullspace(w.Rr, r);
        auto Q = nullspace(w.Rc, r);
        auto SrP = gemm(Trans::N, Trans::N, scalar_t(1.), w.Sr, P);
        U_ = colspace(SrP, r);
        auto ScQ = gemm(Trans::N, Trans::N, scalar_t(1.), w.Sc, Q);
        V_ = colspace(ScQ, r);

        // D = (I - U U*) Sr pseudoinv(Rr) + U U* ((I - V V*) Sc pseudoinv(Rc))*
        // D = D1 + U U* D2*
        // solve D1 and D2 from the least squares systems
        //    D1 Rr = (I - U U*) Sr    or    Rr* D1* = Sr* (I - U U*)
        //    D2 Rc = (I - V V*) Sc    or    Rc* D2* = Sc* (I - V V*)
        auto SrIUU = w.Sr.transpose();
        auto SrU = gemm(Trans::C, Trans::N, scalar_t(-1.), w.Sr, U_);
        gemm(Trans::N, Trans::C, scalar_t(1.), SrU, U_, scalar_t(1.), SrIUU);
        w.Rr.print("w.Rr");
        SrIUU.print("SrIUU");
        int info = blas::gels
          ('T', w.Rr.rows(), w.Rr.cols(), SrIUU.cols(),
           w.Rr.data(), w.Rr.ld(), SrIUU.data(), SrIUU.ld());
        if (info)
          std::cerr << "xGELS failed with info= " << info << std::endl;
        DenseMW_t D1t(w.Rr.rows(), SrIUU.cols(), SrIUU.ptr(0, 0), w.Rr.rows());
        D_ = D1t.transpose();

        auto ScIVV = w.Sc.transpose();
        auto ScV = gemm(Trans::C, Trans::N, scalar_t(-1.), w.Sc, V_);
        gemm(Trans::N, Trans::C, scalar_t(1.), V_, ScV, scalar_t(1.), ScIVV);
        info = blas::gels
          ('T', w.Rc.rows(), w.Rc.cols(), ScIVV.cols(),
           w.Rc.data(), w.Rc.ld(), ScIVV.data(), ScIVV.ld());
        if (info)
          std::cerr << "xGELS failed with info= " << info << std::endl;
        DenseMW_t D2t(w.Rc.rows(), ScIVV.cols(), ScIVV.ptr(0, 0), w.Rc.rows());
        auto UtD2t = gemm(Trans::C, Trans::C, scalar_t(1.), U_, D2t);
        gemm(Trans::N, Trans::N, scalar_t(1.), U_, UtD2t, scalar_t(1.), D_);
      }
      this->U_state_ = this->V_state_ = State::COMPRESSED;
    }

  } // end namespace HBS
} // end namespace strumpack

#endif // HBS_MATRIX_COMPRESS_HPP
