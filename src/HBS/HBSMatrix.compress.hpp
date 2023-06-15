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
      std::pair<std::size_t,std::size_t> offset;
      int lvl = 0;
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


    template<typename scalar_t> void
    HBSMatrix<scalar_t>::compress(const DenseM_t& A,
                                  const opts_t& opts) {
      auto rgen = random::make_random_generator<real_t>
        (opts.random_engine(), opts.random_distribution());
      DenseM_t Rr, Rc, Sr, Sc;
      int s_old = 0;
      int r = opts.leaf_size();
      // int r = opts.d0();
      auto n = this->cols();
      while (!is_compressed()) {
        int s = 3 * r + opts.p();
        WorkCompress<scalar_t> w;
        Rr.resize(n, s);  DenseMW_t Rr_new(n, s-s_old, Rr, 0, s_old);
        Rc.resize(n, s);  DenseMW_t Rc_new(n, s-s_old, Rc, 0, s_old);
        Sr.resize(n, s);  DenseMW_t Sr_new(n, s-s_old, Sr, 0, s_old);
        Sc.resize(n, s);  DenseMW_t Sc_new(n, s-s_old, Sc, 0, s_old);
        Rr_new.random(*rgen);
        Rc_new.copy(Rr_new);
        STRUMPACK_RANDOM_FLOPS
          (rgen->flops_per_prng() * Rr_new.rows() * Rr_new.cols());
        gemm(Trans::N, Trans::N, scalar_t(1.), A, Rr_new, scalar_t(0.), Sr_new);
        gemm(Trans::C, Trans::N, scalar_t(1.), A, Rc_new, scalar_t(0.), Sc_new);
        if (opts.verbose()) {
          std::cout << "# compressing with s = " << s
                    << " = 3 * " << r << " + " << opts.p() << std::endl;
          // if (opts.compression_sketch() == CompressionSketch::SJLT)
          //   std::cout << "# nnz total = " << total_nnz << std::endl;
        }
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
        compress_recursive(Rr, Rc, Sr, Sc, opts, w, r);
        r *= 2;
      }
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    nullspace(const DenseMatrix<scalar_t>& A, int k) {
      // return last r columns of a full QR factorization of A^T
      auto At = A.transpose();
      int minmn = std::min(At.rows(), At.cols());
      std::unique_ptr<scalar_t[]> tau(new scalar_t[minmn]);
      auto info = blas::geqrf
        (At.rows(), At.cols(), At.data(), At.ld(), tau.get());
      if (info)
        std::cerr << "Computation of nullspace failed in QR factorization, "
                  << "info= " << info << std::endl;
      DenseMatrix<scalar_t> Q(At.rows(), At.rows());
      // TODO this can fail if At.cols > At.rows?
      copy(At.rows(), At.cols(), At, 0, 0, Q, 0, 0);
      info = blas::xxgqr
        (Q.rows(), Q.cols(), minmn, Q.data(), Q.ld(), tau.get());
      if (info)
        std::cerr << "Computation of nullspace failed in xxGQR, "
                  << "info= " << info << std::endl;
      k = std::min(k, (int)(At.rows()));
      return DenseMatrix<scalar_t>(Q.rows(), k, Q, 0, Q.cols()-k);
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    colspace(DenseMatrix<scalar_t>& A, int k,
             const HBSOptions<scalar_t>& opts) {
      int minmn = std::min(A.rows(), A.cols());
      k = std::min(minmn, k);
      std::unique_ptr<scalar_t[]> tau(new scalar_t[minmn]);
#if 0
      auto info = blas::geqrf
        (A.rows(), A.cols(), A.data(), A.ld(), tau.get());
      if (info)
        std::cerr << "Computation of colspace failed in QR factorization, "
                  << "info= " << info << std::endl;
      info = blas::xxgqr(A.rows(), k, minmn, A.data(), A.ld(), tau.get());
#else
      int r = 0;
      std::vector<int> ind(A.cols());
      auto info = blas::geqp3tol
        (A.rows(), A.cols(), A.data(), A.ld(), ind.data(), tau.get(),
         r, opts.rel_tol(), opts.abs_tol());
      if (info)
        std::cerr << "Computation of colspace failed in QR factorization, "
                  << "info= " << info << std::endl;
      k = std::min(r, k);
      info = blas::xxgqr(A.rows(), k, k, A.data(), A.ld(), tau.get());
#endif
      if (info)
        std::cerr << "Computation of colspace failed in xxGQR, "
                  << "info= " << info << std::endl;
      return DenseMatrix<scalar_t>(A.rows(), k, A, 0, 0);
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    pinv(const DenseMatrix<scalar_t>& a) {
      auto A = a;
      int minmn = std::min(A.rows(), A.cols());
      using real_t = typename RealType<scalar_t>::value_type;
      std::vector<real_t> s(minmn);
      DenseMatrix<scalar_t> U(A.rows(), minmn), Vt(minmn, A.cols());
      int info = blas::gesvd
        ('S', 'S', A.rows(), A.cols(), A.data(), A.ld(),
         s.data(), U.data(), U.ld(), Vt.data(), Vt.ld());
      if (info)
        std::cerr << "SVD failed to converge" << std::endl;
      for (int i=0; i<minmn; i++)
        // TODO check for small singular values?
        blas::scal(A.rows(), scalar_t(1./s[i]), U.ptr(0, i), 1);
      return gemm(Trans::C, Trans::C, scalar_t(1.), Vt, U);
    }


    /**
     * Based on: "LINEAR-COMPLEXITY BLACK-BOX RANDOMIZED COMPRESSION
     * OF RANK-STRUCTURED MATRICES" by JAMES LEVITT AND PER-GUNNAR
     * MARTINSSON. Algorithm 4.1.
     *
     * See https://arxiv.org/abs/2205.02990
     */
    template<typename scalar_t> void
    HBSMatrix<scalar_t>::compress_recursive(DenseM_t& Rr, DenseM_t& Rc,
                                            DenseM_t& Sr, DenseM_t& Sc,
                                            const opts_t& opts,
                                            WorkCompress<scalar_t>& w,
                                            int r) {
      // if (is_compressed()) return;
      int s = Rr.cols();
      if (this->leaf()) {
        int m = rows();
        w.Rr = DenseM_t(m, s, Rr, w.offset.first, 0);
        w.Rc = DenseM_t(m, s, Rc, w.offset.first, 0);
        w.Sr = DenseM_t(m, s, Sr, w.offset.first, 0);
        w.Sc = DenseM_t(m, s, Sc, w.offset.first, 0);
      } else {
        w.split(child(0)->dims());
#pragma omp task default(shared)
        child(0)->compress_recursive(Rr, Rc, Sr, Sc, opts, w.c[0], r);
#pragma omp task default(shared)
        child(1)->compress_recursive(Rr, Rc, Sr, Sc, opts, w.c[1], r);
#pragma omp taskwait
        if (!child(0)->is_compressed() ||
            !child(1)->is_compressed()) return;
        auto rU0 = child(0)->U_.cols();   auto rV0 = child(0)->V_.cols();
        auto rU1 = child(1)->U_.cols();   auto rV1 = child(1)->V_.cols();
        w.Rr = DenseM_t(rV0 + rV1, s);
        w.Rc = DenseM_t(rU0 + rU1, s);
        DenseMW_t
          Rr0(rV0, s, w.Rr, 0, 0),   Rc0(rU0, s, w.Rc, 0, 0),
          Rr1(rV1, s, w.Rr, rV0, 0), Rc1(rU1, s, w.Rc, rU0, 0);
        gemm(Trans::C, Trans::N, scalar_t(1.), child(0)->V_, w.c[0].Rr, scalar_t(0.), Rr0);
        gemm(Trans::C, Trans::N, scalar_t(1.), child(1)->V_, w.c[1].Rr, scalar_t(0.), Rr1);
        gemm(Trans::C, Trans::N, scalar_t(1.), child(0)->U_, w.c[0].Rc, scalar_t(0.), Rc0);
        gemm(Trans::C, Trans::N, scalar_t(1.), child(1)->U_, w.c[1].Rc, scalar_t(0.), Rc1);

        w.Sr = DenseM_t(rU0 + rU1, s);
        w.Sc = DenseM_t(rV0 + rV1, s);
        DenseMW_t
          Sr0(rU0, s, w.Sr, 0, 0),   Sc0(rV0, s, w.Sc, 0, 0),
          Sr1(rU1, s, w.Sr, rU0, 0), Sc1(rV1, s, w.Sc, rV0, 0);
        gemm(Trans::N, Trans::N, scalar_t(-1.), child(0)->D_, w.c[0].Rr, scalar_t(1.), w.c[0].Sr);
        gemm(Trans::N, Trans::N, scalar_t(-1.), child(1)->D_, w.c[1].Rr, scalar_t(1.), w.c[1].Sr);
        gemm(Trans::C, Trans::N, scalar_t(1.),  child(0)->U_, w.c[0].Sr, scalar_t(0.), Sr0);
        gemm(Trans::C, Trans::N, scalar_t(1.),  child(1)->U_, w.c[1].Sr, scalar_t(0.), Sr1);
        gemm(Trans::C, Trans::N, scalar_t(-1.), child(0)->D_, w.c[0].Rc, scalar_t(1.), w.c[0].Sc);
        gemm(Trans::C, Trans::N, scalar_t(-1.), child(1)->D_, w.c[1].Rc, scalar_t(1.), w.c[1].Sc);
        gemm(Trans::C, Trans::N, scalar_t(1.),  child(0)->V_, w.c[0].Sc, scalar_t(0.), Sc0);
        gemm(Trans::C, Trans::N, scalar_t(1.),  child(1)->V_, w.c[1].Sc, scalar_t(0.), Sc1);
      }
      if (w.lvl == 0) {
        // D = Sr pinv(Rr)
        D_ = gemm(Trans::N, Trans::N, scalar_t(1.), w.Sr, pinv(w.Rr));
        U_state_ = V_state_ = State::COMPRESSED;
      } else {
        {
          auto P = nullspace(w.Rr, r);
          auto SrP = gemm(Trans::N, Trans::N, scalar_t(1.), w.Sr, P);
          U_ = colspace(SrP, r, opts);
          if (int(U_.cols()) >= r - opts.p()) return;
          U_state_ = State::COMPRESSED;
        } {
          auto Q = nullspace(w.Rc, r);
          auto ScQ = gemm(Trans::N, Trans::N, scalar_t(1.), w.Sc, Q);
          V_ = colspace(ScQ, r, opts);
          if (int(V_.cols()) >= r - opts.p()) return;
          V_state_ = State::COMPRESSED;
        }
        // D = (I - U U*) Sr pinv(Rr) + U U* ((I - V V*) Sc pinv(Rc))*
        D_ = DenseM_t(U_.rows(), V_.rows());
        {
          gemm(Trans::N, Trans::N, scalar_t(1.), w.Sr, pinv(w.Rr), scalar_t(0.), D_);
          auto USrRr = gemm(Trans::C, Trans::N, scalar_t(1.), U_, D_);
          gemm(Trans::N, Trans::N, scalar_t(-1.), U_, USrRr, scalar_t(1.), D_);
        } {
          auto IVVScRc = gemm(Trans::N, Trans::N, scalar_t(1.), w.Sc, pinv(w.Rc));
          auto VScRc = gemm(Trans::C, Trans::N, scalar_t(1.), V_, IVVScRc);
          gemm(Trans::N, Trans::N, scalar_t(-1.), V_, VScRc, scalar_t(1.), IVVScRc);
          auto UIVVScRc = gemm(Trans::C, Trans::C, scalar_t(1.), U_, IVVScRc);
          gemm(Trans::N, Trans::N, scalar_t(1.), U_, UIVVScRc, scalar_t(1.), D_);
        }
      }
    }

  } // end namespace HBS
} // end namespace strumpack

#endif // HBS_MATRIX_COMPRESS_HPP
