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
#ifndef HSS_MATRIX_COMPRESS_STABLE_HPP
#define HSS_MATRIX_COMPRESS_STABLE_HPP

#include "misc/RandomWrapper.hpp"

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> void HSSMatrix<scalar_t>::compress_stable
    (const DenseM_t& A, const opts_t& opts) {
      AFunctor<scalar_t> afunc(A);
      if (opts.compression_sketch() == CompressionSketch::SJLT) {
        auto d = opts.d0();
        auto dd = opts.dd();
        auto total_nnz = opts.nnz0();
        // assert(dd <= d);
        auto n = this->cols();
        DenseM_t Rr, Rc, Sr, Sc;
        SJLTGenerator<scalar_t,int> g;
        bool chunk = opts.SJLT_algo() == SJLTAlgo::CHUNK;
        SJLTMatrix<scalar_t,int> S(g, 0, n, 0, chunk);
        if (opts.verbose())
          std::cout<< "# compressing with SJLT \n";
        WorkCompress<scalar_t> w;
        while (!this->is_compressed()) {
          Rr.resize(n, d+dd);
          Rc.resize(n, d+dd);
          Sr.resize(n, d+dd);
          Sc.resize(n, d+dd);
          int c = (d == opts.d0()) ? 0 : d;
          int dnew = (d == opts.d0()) ? d+dd : dd;
          DenseMW_t Rr_new(n, dnew, Rr, 0, c);
          DenseMW_t Rc_new(n, dnew, Rc, 0, c);
          DenseMW_t Sr_new(n, dnew, Sr, 0, c);
          DenseMW_t Sc_new(n, dnew, Sc, 0, c);
          if (c == 0) {
            S.add_columns(dnew,opts.nnz0());
            Rr_new.copy(S.SJLT_to_dense());
            matrix_times_SJLT(A, S, Sr_new);
            matrixT_times_SJLT(A, S, Sc_new);
          } else {
            SJLTMatrix<scalar_t,int> temp
              (S.get_g(), opts.nnz(), n, dnew, chunk);
            S.append_sjlt_matrix(temp);
            Rr_new.copy(temp.SJLT_to_dense());
            matrix_times_SJLT(A, temp, Sr_new);
            matrixT_times_SJLT(A, temp, Sc_new);
            total_nnz += opts.nnz();
          }
          Rc_new.copy(Rr_new);
          if (opts.verbose())
            std::cout << "# compressing with d+dd = " << d << "+" << dd
                      << " (stable)" << std::endl
                      << "# nnz total = " << total_nnz << std::endl;
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
          compress_recursive_stable
            (Rr, Rc, Sr, Sc, afunc, opts, w,
             d, dd, this->openmp_task_depth_/*,&S*/);
          if (!this->is_compressed()) {
            d += dd;
            dd = std::min(dd, opts.max_rank()-d);
          }
        }
        if (opts.verbose())
          std::cout << "# Final length of row: " << d+dd << std::endl
                    << "# Total nnz in each row: "
                    << total_nnz << std::endl;
      } else
        compress_stable(afunc, afunc, opts);
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::compress_stable
    (const mult_t& Amult, const elem_t& Aelem, const opts_t& opts) {
      auto d = opts.d0();
      auto dd = opts.dd();
      auto total_nnz = opts.nnz0();
      // assert(dd <= d);
      auto n = this->cols();
      DenseM_t Rr, Rc, Sr, Sc;
      std::unique_ptr<random::RandomGeneratorBase<real_t>> rgen;
      SJLTGenerator<scalar_t,int> g;
      if (!opts.user_defined_random() &&
          opts.compression_sketch() == CompressionSketch::GAUSSIAN)
        rgen = random::make_random_generator<real_t>
          (opts.random_engine(), opts.random_distribution());
      WorkCompress<scalar_t> w;
      while (!this->is_compressed()) {
        Rr.resize(n, d+dd);
        Rc.resize(n, d+dd);
        Sr.resize(n, d+dd);
        Sc.resize(n, d+dd);
        int c = (d == opts.d0()) ? 0 : d;
        int dnew = (d == opts.d0()) ? d+dd : dd;
        DenseMW_t Rr_new(n, dnew, Rr, 0, c);
        DenseMW_t Rc_new(n, dnew, Rc, 0, c);
        DenseMW_t Sr_new(n, dnew, Sr, 0, c);
        DenseMW_t Sc_new(n, dnew, Sc, 0, c);
        if (!opts.user_defined_random()) {
          if (opts.compression_sketch() == CompressionSketch::GAUSSIAN) {
            Rr_new.random(*rgen);
            STRUMPACK_RANDOM_FLOPS
              (rgen->flops_per_prng() * Rr_new.rows() * Rr_new.cols());
          } else if(opts.compression_sketch() == CompressionSketch::SJLT) {
            if (c == 0)
              g.SJLTDenseSketch(Rr_new, total_nnz);
            else {
              g.SJLTDenseSketch(Rr_new, opts.nnz());
              total_nnz += opts.nnz();
            }
          }
          Rc_new.copy(Rr_new);
        }
        Amult(Rr_new, Rc_new, Sr_new, Sc_new);
        if (opts.verbose()) {
          std::cout << "# compressing with d+dd = " << d << "+" << dd
                    << " (stable)" << std::endl;
          if (opts.compression_sketch() == CompressionSketch::SJLT)
            std::cout << "# nnz total = " << total_nnz << std::endl;
        }
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
        compress_recursive_stable
          (Rr, Rc, Sr, Sc, Aelem, opts, w,
           d, dd, this->openmp_task_depth_);
        if (!this->is_compressed()) {
          d += dd;
          dd = std::min(dd, opts.max_rank()-d);
        }
      }
      if (opts.verbose() && opts.compression_sketch() ==
          CompressionSketch::SJLT)
        std::cout << "# Final length of row: " << d << std::endl
                  << "total nnz in each row: "
                  << total_nnz << std::endl;
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress_recursive_stable
    (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
     const elem_t& Aelem, const opts_t& opts,
     WorkCompress<scalar_t>& w, int d, int dd, int depth) {
      // SJLTMatrix<scalar_t,int>* S) {
      if (this->leaf()) {
        if (this->is_untouched()) {
          std::vector<std::size_t> I, J;
          I.reserve(this->rows());
          J.reserve(this->cols());
          for (std::size_t i=0; i<this->rows(); i++)
            I.push_back(i+w.offset.first);
          for (std::size_t j=0; j<this->cols(); j++)
            J.push_back(j+w.offset.second);
          D_ = DenseM_t(this->rows(), this->cols());
          Aelem(I, J, D_);
        }
      } else {
        w.split(child(0)->dims());
        bool tasked = depth < params::task_recursion_cutoff_level;
        if (tasked) {
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          child(0)->compress_recursive_stable
            (Rr, Rc, Sr, Sc, Aelem, opts, w.c[0], d, dd, depth+1/*,S*/);
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          child(1)->compress_recursive_stable
            (Rr, Rc, Sr, Sc, Aelem, opts, w.c[1], d, dd, depth+1/*,S*/);
#pragma omp taskwait
        } else {
          child(0)->compress_recursive_stable
            (Rr, Rc, Sr, Sc, Aelem, opts, w.c[0], d, dd, depth+1/*,S*/);
          child(1)->compress_recursive_stable
            (Rr, Rc, Sr, Sc, Aelem, opts, w.c[1], d, dd, depth+1/*,S*/);
        }
        if (!child(0)->is_compressed() ||
            !child(1)->is_compressed()) return;
        if (this->is_untouched()) {
#pragma omp task default(shared) if(tasked)                             \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          {
            B01_ = DenseM_t(child(0)->U_rank(), child(1)->V_rank());
            Aelem(w.c[0].Ir, w.c[1].Ic, B01_);
          }
#pragma omp task default(shared) if(tasked)                             \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          {
            B10_ = DenseM_t(child(1)->U_rank(), child(0)->V_rank());
            Aelem(w.c[1].Ir, w.c[0].Ic, B10_);
          }
#pragma omp taskwait
        }
      }
      if (w.lvl == 0) this->U_state_ = this->V_state_ = State::COMPRESSED;
      else {
        if (this->is_untouched())
          compute_local_samples(Rr, Rc, Sr, Sc, w, 0, d+dd, depth/*,S*/);
        else compute_local_samples(Rr, Rc, Sr, Sc, w, d, dd, depth/*,S*/);
        if (!this->is_compressed()) {
          compute_U_basis_stable(Sr, opts, w, d, dd, depth);
          compute_V_basis_stable(Sc, opts, w, d, dd, depth);
          if (this->is_compressed())
            reduce_local_samples(Rr, Rc, w, 0, d+dd, depth);
        } else reduce_local_samples(Rr, Rc, w, d, dd, depth);
      }
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress_level_stable
    (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
     const opts_t& opts, WorkCompress<scalar_t>& w,
     int d, int dd, int lvl, int depth) {
      if (this->leaf()) {
        if (w.lvl < lvl) return;
      } else {
        if (w.lvl < lvl) {
          bool tasked = depth < params::task_recursion_cutoff_level;
          if (tasked) {
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
            child(0)->compress_level_stable
              (Rr, Rc, Sr, Sc, opts, w.c[0], d, dd, lvl, depth);
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
            child(1)->compress_level_stable
              (Rr, Rc, Sr, Sc, opts, w.c[1], d, dd, lvl, depth);
#pragma omp taskwait
          } else {
            child(0)->compress_level_stable
              (Rr, Rc, Sr, Sc, opts, w.c[0], d, dd, lvl, depth);
            child(1)->compress_level_stable
              (Rr, Rc, Sr, Sc, opts, w.c[1], d, dd, lvl, depth);
          }
          return;
        }
        if (!child(0)->is_compressed() ||
            !child(1)->is_compressed()) return;
      }
      if (w.lvl==0) this->U_state_ = this->V_state_ = State::COMPRESSED;
      else {
        if (this->is_untouched())
          compute_local_samples(Rr, Rc, Sr, Sc, w, 0, d+dd, depth);
        else compute_local_samples(Rr, Rc, Sr, Sc, w, d, dd, depth);
        if (!this->is_compressed()) {
          compute_U_basis_stable(Sr, opts, w, d, dd, depth);
          compute_V_basis_stable(Sc, opts, w, d, dd, depth);
          if (this->is_compressed())
            reduce_local_samples(Rr, Rc, w, 0, d+dd, depth);
        } else reduce_local_samples(Rr, Rc, w, d, dd, depth);
      }
    }


    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compute_U_basis_stable
    (DenseM_t& Sr, const opts_t& opts, WorkCompress<scalar_t>& w,
     int d, int dd, int depth) {
      if (this->U_state_ == State::COMPRESSED) return;
      int u_rows = this->leaf() ? this->rows() :
        child(0)->U_rank()+child(1)->U_rank();
      DenseMW_t lSr(u_rows, d+dd, Sr, w.offset.second, 0);
      if (d+dd >= opts.max_rank() || d+dd >= int(u_rows) ||
          update_orthogonal_basis
          (opts, w.U_r_max, lSr, w.Qr, d, dd,
           this->U_state_ == State::UNTOUCHED, w.lvl, depth)) {
        w.Qr.clear();
        auto rtol = opts.rel_tol() / w.lvl;
        auto atol = opts.abs_tol() / w.lvl;
        lSr.ID_row(U_.E(), U_.P(), w.Jr, rtol, atol, opts.max_rank(), depth);
        STRUMPACK_ID_FLOPS(ID_row_flops(lSr, U_.cols()));
        this->U_rank_ = U_.cols();
        this->U_rows_ = U_.rows();
        w.Ir.reserve(U_.cols());
        if (this->leaf())
          for (auto i : w.Jr) w.Ir.push_back(w.offset.first + i);
        else {
          auto r0 = w.c[0].Ir.size();
          for (auto i : w.Jr)
            w.Ir.push_back((i < r0) ? w.c[0].Ir[i] : w.c[1].Ir[i-r0]);
        }
        this->U_state_ = State::COMPRESSED;
      } else {
        // if (2*d > u_rows) set_U_full_rank(w);
        // else
        this->U_state_ = State::PARTIALLY_COMPRESSED;
      }
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compute_V_basis_stable
    (DenseM_t& Sc, const opts_t& opts, WorkCompress<scalar_t>& w,
     int d, int dd, int depth) {
      if (this->V_state_ == State::COMPRESSED) return;
      int v_rows = this->leaf() ? this->rows() :
        child(0)->V_rank()+child(1)->V_rank();
      DenseMW_t lSc(v_rows, d+dd, Sc, w.offset.second, 0);
      if (d+dd >= opts.max_rank() || d+dd >= v_rows ||
          update_orthogonal_basis
          (opts, w.V_r_max, lSc, w.Qc, d, dd,
           this->V_state_ == State::UNTOUCHED, w.lvl, depth)) {
        w.Qc.clear();
        auto rtol = opts.rel_tol() / w.lvl;
        auto atol = opts.abs_tol() / w.lvl;
        lSc.ID_row(V_.E(), V_.P(), w.Jc, rtol, atol, opts.max_rank(), depth);
        STRUMPACK_ID_FLOPS(ID_row_flops(lSc, V_.cols()));
        this->V_rank_ = V_.cols();
        this->V_rows_ = V_.rows();
        w.Ic.reserve(V_.cols());
        if (this->leaf())
          for (auto j : w.Jc) w.Ic.push_back(w.offset.second + j);
        else {
          auto r0 = w.c[0].Ic.size();
          for (auto j : w.Jc)
            w.Ic.push_back((j < r0) ? w.c[0].Ic[j] : w.c[1].Ic[j-r0]);
        }
        this->V_state_ = State::COMPRESSED;
      } else {
        // if (2*d > v_rows) set_V_full_rank(w);
        // else
        this->V_state_ = State::PARTIALLY_COMPRESSED;
      }
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::set_U_full_rank(WorkCompress<scalar_t>& w) {
      auto u_rows = this->leaf() ? this->rows() :
        child(0)->U_rank()+child(1)->U_rank();
      U_ = HSSBasisID<scalar_t>(u_rows);
      w.Jr.reserve(u_rows);
      for (std::size_t i=0; i<u_rows; i++) w.Jr.push_back(i);
      w.Ir.reserve(u_rows);
      if (this->leaf())
        for (std::size_t i=0; i<u_rows; i++)
          w.Ir.push_back(w.offset.first + i);
      else {
        for (std::size_t i=0; i<child(0)->U_rank(); i++)
          w.Ir.push_back(w.c[0].Ir[i]);
        for (std::size_t i=0; i<child(1)->U_rank(); i++)
          w.Ir.push_back(w.c[1].Ir[i]);
      }
      this->U_state_ = State::COMPRESSED;
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::set_V_full_rank(WorkCompress<scalar_t>& w) {
      auto v_rows = this->leaf() ? this->rows() :
        child(0)->V_rank()+child(1)->V_rank();
      V_ = HSSBasisID<scalar_t>(v_rows);
      w.Jc.reserve(v_rows);
      for (std::size_t j=0; j<v_rows; j++) w.Jc.push_back(j);
      w.Ic.reserve(v_rows);
      if (this->leaf())
        for (std::size_t j=0; j<v_rows; j++)
          w.Ic.push_back(w.offset.second + j);
      else {
        for (std::size_t j=0; j<child(0)->V_rank(); j++)
          w.Ic.push_back(w.c[0].Ic[j]);
        for (std::size_t j=0; j<child(1)->V_rank(); j++)
          w.Ic.push_back(w.c[1].Ic[j]);
      }
      this->V_state_ = State::COMPRESSED;
    }

    template<typename scalar_t> bool
    HSSMatrix<scalar_t>::update_orthogonal_basis
    (const opts_t& opts, scalar_t& r_max_0,
     const DenseM_t& S, DenseM_t& Q, int d, int dd,
     bool untouched, int L, int depth) {
      int m = S.rows();
      if (d >= m) return true;
      Q.resize(m, d+dd);
      copy(m, dd, S, 0, d, Q, 0, d);
      DenseMW_t Q2, Q12;
      if (untouched) {
        Q2 = DenseMW_t(m, std::min(d, m), Q, 0, 0);
        Q12 = DenseMW_t(m, std::min(d, m), Q, 0, 0);
        copy(m, d, S, 0, 0, Q, 0, 0);
      } else {
        Q2 = DenseMW_t(m, std::min(dd, m-(d-dd)), Q, 0, d-dd);
        Q12 = DenseMW_t(m, std::min(d, m), Q, 0, 0);
      }
      scalar_t r_min, r_max;
      Q2.orthogonalize(r_max, r_min, depth);
      STRUMPACK_QR_FLOPS(orthogonalize_flops(Q2));
      if (untouched) r_max_0 = r_max;
      auto atol = opts.abs_tol() / L;
      auto rtol = opts.rel_tol() / L;
      if (std::abs(r_min) < atol || std::abs(r_min / r_max_0) < rtol)
        return true;
      DenseMW_t Q3(m, dd, Q, 0, d);
      // only use p columns of Q3 to check the stopping criterion
      DenseMW_t Q3p(m, std::min(dd, opts.p()), Q, 0, d);
      DenseM_t Q12tQ3(Q12.cols(), Q3.cols());
      auto S3norm = Q3p.norm();
      TIMER_TIME(TaskType::ORTHO, 1, t_ortho);
      gemm(Trans::C, Trans::N, scalar_t(1.), Q12, Q3,
           scalar_t(0.), Q12tQ3, depth);
      gemm(Trans::N, Trans::N, scalar_t(-1.), Q12, Q12tQ3,
           scalar_t(1.), Q3, depth);
      // iterated classical Gram-Schmidt
      gemm(Trans::C, Trans::N, scalar_t(1.), Q12, Q3,
           scalar_t(0.), Q12tQ3, depth);
      gemm(Trans::N, Trans::N, scalar_t(-1.), Q12, Q12tQ3,
           scalar_t(1.), Q3, depth);
      TIMER_STOP(t_ortho);
      STRUMPACK_ORTHO_FLOPS((gemm_flops(Trans::C, Trans::N, scalar_t(1.), Q12, Q3, scalar_t(0.)) +
                             gemm_flops(Trans::N, Trans::N, scalar_t(-1.), Q12, Q12tQ3, scalar_t(1.)) +
                             gemm_flops(Trans::C, Trans::N, scalar_t(1.), Q12, Q3, scalar_t(0.)) +
                             gemm_flops(Trans::N, Trans::N, scalar_t(-1.), Q12, Q12tQ3, scalar_t(1.))));
      auto Q3norm = Q3p.norm(); // TODO norm flops ?
      if (opts.compression_sketch() == CompressionSketch::SJLT)
        return (Q3norm / std::sqrt(double(opts.nnz())) < atol)
          || (Q3norm / S3norm < rtol);
      return (Q3norm / std::sqrt(double(dd)) < atol)
        || (Q3norm / S3norm < rtol);
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_COMPRESS_HPP
