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
#ifndef HSS_MATRIX_COMPRESS_HPP
#define HSS_MATRIX_COMPRESS_HPP

#include "misc/RandomWrapper.hpp"
#include "HSS/HSSMatrix.sketch.hpp"

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> void HSSMatrix<scalar_t>::compress_original
    (const DenseM_t& A, const opts_t& opts) {
      AFunctor<scalar_t> afunc(A);
      if (opts.compression_sketch() == CompressionSketch::SJLT) {
        bool chunk = opts.SJLT_algo() == SJLTAlgo::CHUNK;
        if (opts.verbose())
          std::cout << "# compressing with SJLT" << std::endl;
        int d_old = 0, d = opts.d0() + opts.p(),
          total_nnz = 0, nnz_cur = opts.nnz();
        auto n = this->cols();
        DenseM_t Rr, Rc, Sr, Sc;
        SJLTGenerator<scalar_t,int> g;
        SJLTMatrix<scalar_t,int> S(g, 0, n, 0, chunk);
        WorkCompress<scalar_t> w;
        while (!this->is_compressed()) {
          Rr.resize(n, d);
          Rc.resize(n, d);
          Sr.resize(n, d);
          Sc.resize(n, d);
          DenseMW_t Rr_new(n, d-d_old, Rr, 0, d_old);
          DenseMW_t Rc_new(n, d-d_old, Rc, 0, d_old);
          DenseMW_t Sr_new(n, d-d_old, Sr, 0, d_old);
          DenseMW_t Sc_new(n, d-d_old, Sc, 0, d_old);
          if (d_old == 0) {
            S.add_columns(d,opts.nnz0());
            Rr_new.copy(S.SJLT_to_dense());
            matrix_times_SJLT(A, S, Sr_new);
            matrixT_times_SJLT(A, S, Sc_new);
            total_nnz += opts.nnz0();
          } else{
            SJLTMatrix<scalar_t,int> temp
              (S.get_g(), nnz_cur, n, d-d_old, chunk);
            total_nnz += nnz_cur;
            nnz_cur *= 2;
            S.append_sjlt_matrix(temp);
            Rr_new.copy(temp.SJLT_to_dense());
            matrix_times_SJLT(A, temp, Sr_new);
            matrixT_times_SJLT(A, temp, Sc_new);
          }
          Rc_new.copy(Rr_new);
          if (opts.verbose())
            std::cout << "# compressing with d = " << d-opts.p()
                      << " + " << opts.p() << " (original)" << std::endl
                      << "# nnz total = " << total_nnz << std::endl;
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
          compress_recursive_original
            (Rr, Rc, Sr, Sc, afunc, opts, w, d-d_old,
             this->openmp_task_depth_);
          if (!this->is_compressed()) {
            d_old = d;
            d = 2 * (d_old - opts.p()) + opts.p();
          }
        }
        if (opts.verbose() &&
            opts.compression_sketch() == CompressionSketch::SJLT)
          std::cout << "# final length of row: " << d << std::endl
                    << "# total nnz in each row: "
                    << total_nnz << std::endl;
      } else
        compress_original(afunc, afunc, opts);
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::compress_original
    (const mult_t& Amult, const elem_t& Aelem, const opts_t& opts) {
      int d_old = 0, d = opts.d0() + opts.p(), total_nnz = 0;
      auto n = this->cols();
      DenseM_t Rr, Rc, Sr, Sc;
      std::unique_ptr<random::RandomGeneratorBase<real_t>> rgen;
      SJLTGenerator<scalar_t,int> g;
      bool chunk = opts.SJLT_algo() == SJLTAlgo::CHUNK;
      SJLTMatrix<scalar_t,int> S(g, 0, n, 0, chunk);
      if (!opts.user_defined_random() &&
          opts.compression_sketch() == CompressionSketch::GAUSSIAN)
        rgen = random::make_random_generator<real_t>
          (opts.random_engine(), opts.random_distribution());
      WorkCompress<scalar_t> w;
      while (!this->is_compressed()) {
        Rr.resize(n, d);
        Rc.resize(n, d);
        Sr.resize(n, d);
        Sc.resize(n, d);
        DenseMW_t Rr_new(n, d-d_old, Rr, 0, d_old);
        DenseMW_t Rc_new(n, d-d_old, Rc, 0, d_old);
        if (!opts.user_defined_random()) {
          if (opts.compression_sketch() == CompressionSketch::GAUSSIAN) {
            Rr_new.random(*rgen);
            STRUMPACK_RANDOM_FLOPS
              (rgen->flops_per_prng() * Rr_new.rows() * Rr_new.cols());
          } else if (opts.compression_sketch() == CompressionSketch::SJLT) {
            if (d_old == 0) {
              S.add_columns(d, opts.nnz0());
              Rr_new.copy(S.SJLT_to_dense());
              total_nnz += opts.nnz0();
            } else{
              SJLTMatrix<scalar_t,int> temp
                (S.get_g(), opts.nnz(), n, d-d_old, chunk);
              S.append_sjlt_matrix(temp);
              Rr_new.copy(temp.SJLT_to_dense());
              total_nnz += opts.nnz();
            }
          }
          Rc_new.copy(Rr_new);
        }
        DenseMW_t Sr_new(n, d-d_old, Sr, 0, d_old);
        DenseMW_t Sc_new(n, d-d_old, Sc, 0, d_old);
        Amult(Rr_new, Rc_new, Sr_new, Sc_new);
        if (opts.verbose()) {
          std::cout << "# compressing with d = " << d-opts.p()
                    << " + " << opts.p() << " (original)" << std::endl;
          if (opts.compression_sketch() == CompressionSketch::SJLT)
            std::cout << "# nnz total = " << total_nnz << std::endl;
        }
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
        compress_recursive_original
          (Rr, Rc, Sr, Sc, Aelem, opts, w, d-d_old,
           this->openmp_task_depth_);
        if (!this->is_compressed()) {
          d_old = d;
          d = 2 * (d_old - opts.p()) + opts.p();
        }
      }
      if (opts.verbose() &&
          opts.compression_sketch() == CompressionSketch::SJLT)
        std::cout << "# final length of row: " << d << std::endl
                  << "# total nnz in each row: "
                  << total_nnz << std::endl;
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress_hard_restart
    (const DenseM_t& A, const opts_t& opts) {
      AFunctor<scalar_t> afunc(A);
      if (opts.compression_sketch() == CompressionSketch::SJLT) {
        bool chunk = opts.SJLT_algo() == SJLTAlgo::CHUNK;
        int d_old = 0, d = opts.d0() + opts.p(), total_nnz = opts.nnz0();
        auto n = this->cols();
        DenseM_t Rr, Rc, Sr, Sc, R2, Sr2, Sc2;
        SJLTGenerator<scalar_t,int> g;
        SJLTMatrix<scalar_t,int> S(g, 0, n, 0, chunk);
        if (opts.verbose())
          std::cout<< "# compressing with SJLT" << std::endl;
        while (!this->is_compressed()) {
          WorkCompress<scalar_t> w;
          Rr = DenseM_t(n, d);
          Rc = DenseM_t(n, d);
          Sr = DenseM_t(n, d);
          Sc = DenseM_t(n, d);
          strumpack::copy(R2,  Rr, 0, 0);
          strumpack::copy(R2,  Rc, 0, 0);
          strumpack::copy(Sr2, Sr, 0, 0);
          strumpack::copy(Sc2, Sc, 0, 0);
          DenseMW_t Rr_new(n, d-d_old, Rr, 0, d_old);
          DenseMW_t Rc_new(n, d-d_old, Rc, 0, d_old);
          DenseMW_t Sr_new(n, d-d_old, Sr, 0, d_old);
          DenseMW_t Sc_new(n, d-d_old, Sc, 0, d_old);
          if (d_old == 0) {
            S.add_columns(d, opts.nnz0());
            Rr_new.copy(S.SJLT_to_dense());
            matrix_times_SJLT(A, S, Sr_new);
            matrixT_times_SJLT(A, S, Sc_new);
          } else {
            SJLTMatrix<scalar_t, int> temp
              (S.get_g(), opts.nnz(), n, d-d_old, chunk);
            S.append_sjlt_matrix(temp);
            Rr_new.copy(temp.SJLT_to_dense());
            total_nnz += opts.nnz();
            matrix_times_SJLT(A, temp, Sr_new);
            matrixT_times_SJLT(A, temp, Sc_new);
          }
          Rc_new.copy(Rr_new);
          R2 = Rr; Sr2 = Sr; Sc2 = Sc;
          if (opts.verbose())
            std::cout << "# compressing with d = " << d-opts.p()
                      << " + " << opts.p() << " (original, hard restart)"
                      << std::endl
                      << "# compressing with nnz = " << total_nnz << std::endl;
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
          compress_recursive_original
            (Rr, Rc, Sr, Sc, afunc, opts, w, d,
             this->openmp_task_depth_);
          if (!this->is_compressed()) {
            d_old = d;
            d = 2 * (d_old - opts.p()) + opts.p();
            total_nnz += opts.nnz();
            reset();
          }
        }
        if (opts.verbose())
          std::cout << "# Final length of row: " << d << std::endl
                    << "total nnz in each row: "
                    << total_nnz << std::endl;
      } else
        compress_hard_restart(afunc, afunc, opts);
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress_hard_restart
    (const mult_t& Amult, const elem_t& Aelem, const opts_t& opts) {
      int d_old = 0, d = opts.d0() + opts.p(), total_nnz = opts.nnz0();
      auto n = this->cols();
      DenseM_t Rr, Rc, Sr, Sc, R2, Sr2, Sc2;
      std::unique_ptr<random::RandomGeneratorBase<real_t>> rgen;
      SJLTGenerator<scalar_t,int> g;
      if (!opts.user_defined_random() &&
          opts.compression_sketch() == CompressionSketch::GAUSSIAN)
        rgen = random::make_random_generator<real_t>
          (opts.random_engine(), opts.random_distribution());
      while (!this->is_compressed()) {
        WorkCompress<scalar_t> w;
        Rr = DenseM_t(n, d);
        Rc = DenseM_t(n, d);
        Sr = DenseM_t(n, d);
        Sc = DenseM_t(n, d);
        strumpack::copy(R2,  Rr, 0, 0);
        strumpack::copy(R2,  Rc, 0, 0);
        strumpack::copy(Sr2, Sr, 0, 0);
        strumpack::copy(Sc2, Sc, 0, 0);
        DenseMW_t Rr_new(n, d-d_old, Rr, 0, d_old);
        DenseMW_t Rc_new(n, d-d_old, Rc, 0, d_old);
        if (!opts.user_defined_random()) {
          if (opts.compression_sketch() == CompressionSketch::GAUSSIAN) {
            Rr_new.random(*rgen);
            STRUMPACK_RANDOM_FLOPS
              (rgen->flops_per_prng() * Rr_new.rows() * Rr_new.cols());
          }
          if (opts.compression_sketch() == CompressionSketch::SJLT)
            g.SJLTDenseSketch(Rr_new,  total_nnz);
          Rc_new.copy(Rr_new);
        }
        DenseMW_t Sr_new(n, d-d_old, Sr, 0, d_old);
        DenseMW_t Sc_new(n, d-d_old, Sc, 0, d_old);
        Amult(Rr_new, Rc_new, Sr_new, Sc_new);
        R2 = Rr; Sr2 = Sr; Sc2 = Sc;
        if (opts.verbose()) {
          std::cout << "# compressing with d = " << d-opts.p()
                    << " + " << opts.p() << " (original, hard restart)"
                    << std::endl;
          if (opts.compression_sketch() == CompressionSketch::SJLT)
            std::cout << "# compressing with nnz = "
                      << total_nnz << std::endl;
        }
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
        compress_recursive_original
          (Rr, Rc, Sr, Sc, Aelem, opts, w, d,
           this->openmp_task_depth_);
        if (!this->is_compressed()) {
          d_old = d;
          d = 2 * (d_old - opts.p()) + opts.p();
          total_nnz += opts.nnz();
          reset();
        }
      }
      if (opts.verbose() &&
          opts.compression_sketch() == CompressionSketch::SJLT)
          std::cout << "# Final length of row: " << d << std::endl
                    << "total nnz in each row: "
                    << total_nnz << std::endl;
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress_recursive_original
    (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
     const elem_t& Aelem, const opts_t& opts,
     WorkCompress<scalar_t>& w, int dd, int depth) {
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
          child(0)->compress_recursive_original
            (Rr, Rc, Sr, Sc, Aelem, opts, w.c[0], dd, depth+1);
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          child(1)->compress_recursive_original
            (Rr, Rc, Sr, Sc, Aelem, opts, w.c[1], dd, depth+1);
#pragma omp taskwait
        } else {
          child(0)->compress_recursive_original
            (Rr, Rc, Sr, Sc, Aelem, opts, w.c[0], dd, depth+1);
          child(1)->compress_recursive_original
            (Rr, Rc, Sr, Sc, Aelem, opts, w.c[1], dd, depth+1);
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
        auto d = Rr.cols();
        if (this->is_untouched())
          compute_local_samples(Rr, Rc, Sr, Sc, w, 0, d, depth);
        else compute_local_samples(Rr, Rc, Sr, Sc, w, d-dd, dd, depth);
        if (!this->is_compressed()) {
          if (compute_U_V_bases(Sr, Sc, opts, w, d, depth)) {
            reduce_local_samples(Rr, Rc, w, 0, d, depth);
            this->U_state_ = this->V_state_ = State::COMPRESSED;
          } else
            this->U_state_ = this->V_state_ = State::PARTIALLY_COMPRESSED;
        } else reduce_local_samples(Rr, Rc, w, d-dd, dd, depth);
      }
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress_level_original
    (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
     const opts_t& opts, WorkCompress<scalar_t>& w,
     int dd, int lvl, int depth) {
      if (this->leaf()) {
        if (w.lvl < lvl) return;
      } else {
        if (w.lvl < lvl) {
          bool tasked = depth < params::task_recursion_cutoff_level;
          if (tasked) {
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
            child(0)->compress_level_original
              (Rr, Rc, Sr, Sc, opts, w.c[0], dd, lvl, depth+1);
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
            child(1)->compress_level_original
              (Rr, Rc, Sr, Sc, opts, w.c[1], dd, lvl, depth+1);
#pragma omp taskwait
          } else {
            child(0)->compress_level_original
              (Rr, Rc, Sr, Sc, opts, w.c[0], dd, lvl, depth+1);
            child(1)->compress_level_original
              (Rr, Rc, Sr, Sc, opts, w.c[1], dd, lvl, depth+1);
          }
          return;
        }
        if (!child(0)->is_compressed() ||
            !child(1)->is_compressed()) return;
      }
      if (w.lvl==0) this->U_state_ = this->V_state_ = State::COMPRESSED;
      else {
        auto d = Rr.cols();
        if (this->is_untouched())
          compute_local_samples(Rr, Rc, Sr, Sc, w, 0, d, depth);
        else compute_local_samples(Rr, Rc, Sr, Sc, w, d-dd, dd, depth);
        if (!this->is_compressed()) {
          if (compute_U_V_bases(Sr, Sc, opts, w, d, depth)) {
            reduce_local_samples(Rr, Rc, w, 0, d, depth);
            this->U_state_ = this->V_state_ = State::COMPRESSED;
          } else
            this->U_state_ = this->V_state_ = State::PARTIALLY_COMPRESSED;
        } else reduce_local_samples(Rr, Rc, w, d-dd, dd, depth);
      }
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::get_extraction_indices
    (std::vector<std::vector<std::size_t>>& I,
     std::vector<std::vector<std::size_t>>& J,
     const std::pair<std::size_t,std::size_t>& off,
     WorkCompress<scalar_t>& w, int& self, int lvl) {
      if (this->leaf()) {
        if (w.lvl == lvl && this->is_untouched()) {
          self++;
          I.emplace_back();  J.emplace_back();
          I.back().reserve(this->rows());  J.back().reserve(this->cols());
          for (std::size_t i=0; i<this->rows(); i++)
            I.back().push_back(i+w.offset.first+off.first);
          for (std::size_t j=0; j<this->cols(); j++)
            J.back().push_back(j+w.offset.second+off.second);
        }
      } else {
        w.split(child(0)->dims());
        if (w.lvl < lvl) {
          child(0)->get_extraction_indices(I, J, off, w.c[0], self, lvl);
          child(1)->get_extraction_indices(I, J, off, w.c[1], self, lvl);
          return;
        }
        if (this->is_untouched()) {
          self += 2;
          I.push_back(w.c[0].Ir);  J.push_back(w.c[1].Ic);
          for (auto& i : I.back()) i += off.first;
          for (auto& j : J.back()) j += off.second;
          I.push_back(w.c[1].Ir);  J.push_back(w.c[0].Ic);
          for (auto& i : I.back()) i += off.first;
          for (auto& j : J.back()) j += off.second;
        }
      }
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::get_extraction_indices
    (std::vector<std::vector<std::size_t>>& I,
     std::vector<std::vector<std::size_t>>& J, std::vector<DenseM_t*>& B,
     const std::pair<std::size_t,std::size_t>& off,
     WorkCompress<scalar_t>& w, int& self, int lvl) {
      if (this->leaf()) {
        if (w.lvl == lvl && this->is_untouched()) {
          self++;
          I.emplace_back();  J.emplace_back();
          I.back().reserve(this->rows());  J.back().reserve(this->cols());
          for (std::size_t i=0; i<this->rows(); i++)
            I.back().push_back(i+w.offset.first+off.first);
          for (std::size_t j=0; j<this->cols(); j++)
            J.back().push_back(j+w.offset.second+off.second);
          D_ = DenseM_t(this->rows(), this->cols());
          B.push_back(&D_);
        }
      } else {
        w.split(child(0)->dims());
        if (w.lvl < lvl) {
          child(0)->get_extraction_indices(I, J, B, off, w.c[0], self, lvl);
          child(1)->get_extraction_indices(I, J, B, off, w.c[1], self, lvl);
          return;
        }
        if (this->is_untouched()) {
          self += 2;
          I.push_back(w.c[0].Ir);  J.push_back(w.c[1].Ic);
          for (auto& i : I.back()) i += off.first;
          for (auto& j : J.back()) j += off.second;
          I.push_back(w.c[1].Ir);  J.push_back(w.c[0].Ic);
          for (auto& i : I.back()) i += off.first;
          for (auto& j : J.back()) j += off.second;
          B01_ = DenseM_t(child(0)->U_rank(), child(1)->V_rank());
          B10_ = DenseM_t(child(1)->U_rank(), child(0)->V_rank());
          B.push_back(&B01_);
          B.push_back(&B10_);
        }
      }
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::extract_D_B
    (const elem_t& Aelem, const opts_t& opts,
     WorkCompress<scalar_t>& w, int lvl) {
      if (this->leaf()) {
        if (w.lvl < lvl) return;
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
        if (w.lvl < lvl) {
          child(0)->extract_D_B(Aelem, opts, w.c[0], lvl);
          child(1)->extract_D_B(Aelem, opts, w.c[1], lvl);
          return;
        }
        if (this->is_untouched()) {
          B01_ = DenseM_t(child(0)->U_rank(), child(1)->V_rank());
          B10_ = DenseM_t(child(1)->U_rank(), child(0)->V_rank());
          Aelem(w.c[0].Ir, w.c[1].Ic, B01_);
          Aelem(w.c[1].Ir, w.c[0].Ic, B10_);
        }
      }
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compute_local_samples
    (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
     WorkCompress<scalar_t>& w, int d0, int d, int depth,
     SJLTMatrix<scalar_t,int>* S) {
      TIMER_TIME(TaskType::COMPUTE_SAMPLES, 1, t_compute);
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
      {
        if (this->leaf()) {
          DenseMW_t wSr(this->rows(), d, Sr, w.offset.second, d0);
          //wSr = -D_*wRr + wSr
          if (true) { // TODO fix performance issue with SJLT
            //if (S == nullptr) {
            DenseMW_t wRr(this->rows(), d, Rr, w.offset.second, d0);
            TIMER_TIME(TaskType::RANDOM_SAMPLING, 1, t_compute);
            gemm(Trans::N, Trans::N, scalar_t(-1), D_, wRr,
                 scalar_t(1.), wSr, depth);
            STRUMPACK_UPDATE_SAMPLE_FLOPS
              (gemm_flops
               (Trans::N, Trans::N, scalar_t(-1), D_, wRr, scalar_t(1.)));
          } else {
            // doing SJLT case here
            // wSr = -D_ S(i:i+m,j:j+n) + wSr
            matrix_times_SJLT
              (D_, *S, wSr, this->rows(), d,w.offset.second, d0,
               scalar_t(-1.), scalar_t(1.));
          }
        } else {
          DenseMW_t wSr0(child(0)->U_rank(), d, Sr,
                         w.offset.second, d0);
          DenseMW_t wSr1(child(1)->U_rank(), d, Sr,
                         w.offset.second+child(0)->U_rank(), d0);
          DenseMW_t wSr_ch0(child(0)->U_rows(), d, Sr,
                            w.c[0].offset.second, d0);
          DenseMW_t wSr_ch1(child(1)->U_rows(), d, Sr,
                            w.c[1].offset.second, d0);
          auto tmp0 = wSr_ch0.extract_rows(w.c[0].Jr);
          auto tmp1 = wSr_ch1.extract_rows(w.c[1].Jr);
          wSr0.copy(tmp0);
          wSr1.copy(tmp1);
          DenseMW_t wRr1(child(1)->V_rank(), d, Rr,
                         w.c[1].offset.second, d0);
          DenseMW_t wRr0(child(0)->V_rank(), d, Rr,
                         w.c[0].offset.second, d0);
          gemm(Trans::N, Trans::N, scalar_t(-1.), B01_, wRr1,
               scalar_t(1.), wSr0, depth);
          gemm(Trans::N, Trans::N, scalar_t(-1.), B10_, wRr0,
               scalar_t(1.), wSr1, depth);
          STRUMPACK_UPDATE_SAMPLE_FLOPS
            (gemm_flops
             (Trans::N, Trans::N, scalar_t(-1.), B01_, wRr1, scalar_t(1.)) +
             gemm_flops
             (Trans::N, Trans::N, scalar_t(-1.), B10_, wRr0, scalar_t(1.)));
        }
      }
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
      {
        if (this->leaf()) {
          DenseMW_t wSc(this->rows(), d, Sc, w.offset.second, d0);
          if (true) { // TODO fix performance issue with SJLT
            //S == nullptr) {
            DenseMW_t wRc(this->rows(), d, Rc, w.offset.second, d0);
            TIMER_TIME(TaskType::RANDOM_SAMPLING, 1, t_compute);
            gemm(Trans::C, Trans::N, scalar_t(-1), D_, wRc,
                 scalar_t(1.), wSc, depth);
            STRUMPACK_UPDATE_SAMPLE_FLOPS
              (gemm_flops(Trans::C, Trans::N, scalar_t(-1), D_, wRc, scalar_t(1.)));
          } else {
            //wSr = -D_^* S(i:i+m,j:j+n) + wSr
            matrixT_times_SJLT
              (D_, *S, wSc, this->rows(), d, w.offset.second, d0,
               scalar_t(-1.), scalar_t(1.));
          }
        } else {
          DenseMW_t wSc0(child(0)->V_rank(), d, Sc, w.offset.second, d0);
          DenseMW_t wSc1(child(1)->V_rank(), d, Sc,
                         w.offset.second+child(0)->V_rank(), d0);
          DenseMW_t wSc_ch0(child(0)->V_rows(), d, Sc,
                            w.c[0].offset.second, d0);
          DenseMW_t wSc_ch1(child(1)->V_rows(), d, Sc,
                            w.c[1].offset.second, d0);
          auto tmp1 = wSc_ch1.extract_rows(w.c[1].Jc);
          auto tmp0 = wSc_ch0.extract_rows(w.c[0].Jc);
          wSc0.copy(tmp0);
          wSc1.copy(tmp1);
          DenseMW_t wRc1(child(1)->U_rank(), d, Rc,
                         w.c[1].offset.second, d0);
          DenseMW_t wRc0(child(0)->U_rank(), d, Rc,
                         w.c[0].offset.second, d0);
          gemm(Trans::C, Trans::N, scalar_t(-1.), B10_, wRc1,
               scalar_t(1.), wSc0, depth);
          gemm(Trans::C, Trans::N, scalar_t(-1.), B01_, wRc0,
               scalar_t(1.), wSc1, depth);
          STRUMPACK_UPDATE_SAMPLE_FLOPS
            (gemm_flops
             (Trans::C, Trans::N, scalar_t(-1.), B10_, wRc1, scalar_t(1.)) +
             gemm_flops
             (Trans::C, Trans::N, scalar_t(-1.), B01_, wRc0, scalar_t(1.)));
        }
      }
#pragma omp taskwait
    }

    // TODO split in U and V compression
    template<typename scalar_t> bool HSSMatrix<scalar_t>::compute_U_V_bases
    (DenseM_t& Sr, DenseM_t& Sc, const opts_t& opts,
     WorkCompress<scalar_t>& w, int d, int depth) {
      auto rtol = opts.rel_tol() / w.lvl;
      auto atol = opts.abs_tol() / w.lvl;
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
      {
        auto u_rows = this->leaf() ? this->rows() :
          child(0)->U_rank()+child(1)->U_rank();
        DenseM_t wSr(u_rows, d, Sr, w.offset.second, 0);
        wSr.ID_row(U_.E(), U_.P(), w.Jr, rtol, atol,
                   opts.max_rank(), depth);
        STRUMPACK_ID_FLOPS(ID_row_flops(wSr, U_.cols()));
      }
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
      {
        auto v_rows = this->leaf() ? this->rows() :
          child(0)->V_rank()+child(1)->V_rank();
        DenseM_t wSc(v_rows, d, Sc, w.offset.second, 0);
        wSc.ID_row(V_.E(), V_.P(), w.Jc, rtol, atol,
                   opts.max_rank(), depth);
        STRUMPACK_ID_FLOPS(ID_row_flops(wSc, V_.cols()));
      }
#pragma omp taskwait

      U_.check();  assert(U_.cols() == w.Jr.size());
      V_.check();  assert(V_.cols() == w.Jc.size());
      if (d-opts.p() >= opts.max_rank() ||
          (int(U_.cols()) < d - opts.p() &&
           int(V_.cols()) < d - opts.p())) {
        this->U_rank_ = U_.cols();  this->U_rows_ = U_.rows();
        this->V_rank_ = V_.cols();  this->V_rows_ = V_.rows();
        w.Ir.reserve(U_.cols());
        w.Ic.reserve(V_.cols());
        if (this->leaf()) {
          for (auto i : w.Jr) w.Ir.push_back(w.offset.first + i);
          for (auto j : w.Jc) w.Ic.push_back(w.offset.second + j);
        } else {
          auto r0 = w.c[0].Ir.size();
          for (auto i : w.Jr)
            w.Ir.push_back((i < r0) ? w.c[0].Ir[i] : w.c[1].Ir[i-r0]);
          r0 = w.c[0].Ic.size();
          for (auto j : w.Jc)
            w.Ic.push_back((j < r0) ? w.c[0].Ic[j] : w.c[1].Ic[j-r0]);
        }
        return true;
      } else {
        w.Jr.clear();
        w.Jc.clear();
        return false;
      }
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::reduce_local_samples
    (DenseM_t& Rr, DenseM_t& Rc, WorkCompress<scalar_t>& w,
     int d0, int d, int depth) {
      TIMER_TIME(TaskType::REDUCE_SAMPLES, 1, t_reduce);
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
      {
        DenseMW_t wRr(V_.rows(), d, Rr, w.offset.second, d0);
        if (this->leaf()) copy(V_.applyC(wRr, depth), wRr, 0, 0);
        else {
          DenseMW_t wRr0(child(0)->V_rank(), d, Rr,
                         w.c[0].offset.second, d0);
          DenseMW_t wRr1(child(1)->V_rank(), d, Rr,
                         w.c[1].offset.second, d0);
          copy(V_.applyC(vconcat(wRr0, wRr1), depth), wRr, 0, 0);
        }
      }
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
      {
        DenseMW_t wRc(U_.rows(), d, Rc, w.offset.second, d0);
        if (this->leaf()) copy(U_.applyC(wRc, depth), wRc, 0, 0);
        else {
          DenseMW_t wRc0(child(0)->U_rank(), d, Rc,
                         w.c[0].offset.second, d0);
          DenseMW_t wRc1(child(1)->U_rank(), d, Rc,
                         w.c[1].offset.second, d0);
          copy(U_.applyC(vconcat(wRc0, wRc1), depth), wRc, 0, 0);
        }
      }
#pragma omp taskwait
      STRUMPACK_REDUCE_SAMPLE_FLOPS
        (V_.applyC_flops(d) + U_.applyC_flops(d));
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_COMPRESS_HPP
