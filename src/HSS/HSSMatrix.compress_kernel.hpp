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
#ifndef HSS_MATRIX_COMPRESS_KERNEL_HPP
#define HSS_MATRIX_COMPRESS_KERNEL_HPP

#include <algorithm>

#include "misc/Tools.hpp"
#include "clustering/NeighborSearch.hpp"

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress
    (const kernel::Kernel<real_t>& K, const opts_t& opts) {
      auto Aelem = [&K]
        (const std::vector<std::size_t>& I,
         const std::vector<std::size_t>& J, DenseM_t& B){
        K(I,J,B);
      };
      compress_with_coordinates(K.data(), Aelem, opts);
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress_with_coordinates
    (const DenseMatrix<real_t>& coords,
     const std::function
     <void(const std::vector<std::size_t>& I,
           const std::vector<std::size_t>& J, DenseM_t& B)>& Aelem,
     const opts_t& opts) {
      int n = coords.cols();
      int ann_number = std::min(n, opts.approximate_neighbors());
      while (!this->is_compressed()) {
        DenseMatrix<std::uint32_t> ann;
        DenseMatrix<real_t> scores;
        TaskTimer timer("approximate_neighbors");
        timer.start();
        find_approximate_neighbors
          (coords, opts.ann_iterations(), ann_number, ann, scores);
        if (opts.verbose())
          std::cout << "# k-ANN=" << ann_number
                    << ", approximate neighbor search time = "
                    << timer.elapsed() << std::endl;
        WorkCompressANN<scalar_t> w;
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
        compress_recursive_ann
          (ann, scores, Aelem, opts, w, this->openmp_task_depth_);
        ann_number = std::min(2*ann_number, n);
      }
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress_recursive_ann
    (DenseMatrix<std::uint32_t>& ann, DenseMatrix<real_t>& scores,
     const elem_t& Aelem, const opts_t& opts, WorkCompressANN<scalar_t>& w,
     int depth) {
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
          child(0)->compress_recursive_ann
            (ann, scores, Aelem, opts, w.c[0], depth+1);
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          child(1)->compress_recursive_ann
            (ann, scores, Aelem, opts, w.c[1], depth+1);
#pragma omp taskwait
        } else {
          child(0)->compress_recursive_ann
            (ann, scores, Aelem, opts, w.c[0], depth);
          child(1)->compress_recursive_ann
            (ann, scores, Aelem, opts, w.c[1], depth);
        }
        if (!child(0)->is_compressed() ||
            !child(1)->is_compressed())
          return;
        // TODO do not re-extract if children are not re-compressed
        //if (this->is_untouched()) {
        B01_ = DenseM_t(child(0)->U_rank(), child(1)->V_rank());
        Aelem(w.c[0].Ir, w.c[1].Ic, B01_);
        B10_ = B01_.transpose();
        //}
      }
      if (w.lvl == 0)
        this->U_state_ = this->V_state_ = State::COMPRESSED;
      else {
        // TODO only do this if not already compressed
        //if (!this->is_compressed()) {
        compute_local_samples_ann(ann, scores, w, Aelem, opts);
        if (compute_U_V_bases_ann(w.S, opts, w, depth))
          this->U_state_ = this->V_state_ = State::COMPRESSED;
        // TODO
        // else
        //     this->_U_state = this->_V_state = State::PARTIALLY_COMPRESSED;
        //}
        w.c.clear();
      }
      w.c.clear();
      w.c.shrink_to_fit();
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compute_local_samples_ann
    (DenseMatrix<std::uint32_t>& ann, DenseMatrix<real_t>& scores,
     WorkCompressANN<scalar_t>& w, const elem_t& Aelem, const opts_t& opts) {
      std::size_t ann_number = ann.rows();
      std::vector<std::size_t> I;
      if (this->leaf()) {
        I.reserve(this->rows());
        for (std::size_t i=0; i<this->rows(); i++)
          I.push_back(i+w.offset.first);

        w.ids_scores.reserve(this->rows()*ann_number);
        for (std::size_t i=w.offset.first;
             i<w.offset.first+this->rows(); i++)
          for (std::size_t j=0; j<ann_number; j++)
            if ((ann(j, i) < w.offset.first) ||
                (ann(j, i) >= w.offset.first + this->rows()))
              w.ids_scores.emplace_back(ann(j, i), scores(j, i));
      } else {
        I.reserve(w.c[0].Ir.size() + w.c[1].Ir.size());
        for (std::size_t i=0; i<w.c[0].Ir.size(); i++)
          I.push_back(w.c[0].Ir[i]);
        for (std::size_t i=0; i<w.c[1].Ir.size(); i++)
          I.push_back(w.c[1].Ir[i]);

        w.ids_scores.reserve(w.c[0].ids_scores.size()+
                             w.c[1].ids_scores.size());
        for (std::size_t i=0; i<w.c[0].ids_scores.size(); i++)
          if ((w.c[0].ids_scores[i].first < w.offset.first) ||
              (w.c[0].ids_scores[i].first >= w.offset.first + this->rows()))
            w.ids_scores.emplace_back(w.c[0].ids_scores[i]);
        for (std::size_t i=0; i<w.c[1].ids_scores.size(); i++)
          if ((w.c[1].ids_scores[i].first < w.offset.first) ||
              (w.c[1].ids_scores[i].first >= w.offset.first + this->rows()))
            w.ids_scores.emplace_back(w.c[1].ids_scores[i]);
      }

      // sort on column indices first, then on scores
      std::sort(w.ids_scores.begin(), w.ids_scores.end());

      // remove duplicate indices, keep only first entry of
      // duplicates, which is the one with the highest score, because
      // of the above sort
      w.ids_scores.erase
        (std::unique(w.ids_scores.begin(), w.ids_scores.end(),
                     [](const std::pair<std::size_t,real_t>& a,
                        const std::pair<std::size_t,real_t>& b) {
                       return a.first == b.first; }), w.ids_scores.end());

#if 0 // drop some columns
      // maximum number of samples
      std::size_t d_max = this->leaf() ?
        I.size() + opts.dd() :   // leaf size + some oversampling
        w.c[0].Ir.size() + w.c[1].Ir.size() + opts.dd();
      auto d = std::min(w.ids_scores.size(), d_max);

      if (d < w.ids_scores.size()) {
        // sort based on scores, keep only d closest
        std::nth_element
          (w.ids_scores.begin(), w.ids_scores.begin()+d, w.ids_scores.end(),
           [](const std::pair<std::size_t,real_t>& a,
              const std::pair<std::size_t,real_t>& b) {
            return a.second < b.second; });
        w.ids_scores.resize(d);
      }
#else
      auto d = w.ids_scores.size();
#endif

      // sort based on ids; this is needed in the parent, see below
      std::sort(w.ids_scores.begin(), w.ids_scores.end());

      if (this->leaf()) {
        std::vector<std::size_t> Scolids(d);
        for (std::size_t j=0; j<d; j++)
          Scolids[j] = w.ids_scores[j].first;
        w.S = DenseM_t(I.size(), Scolids.size());
        Aelem(I, Scolids, w.S);
      } else {
        w.S = DenseM_t(I.size(), d);
        for (int c=0; c<2; c++) {
          std::size_t m = w.c[c].Ir.size();
          auto it_lo = w.c[c].ids_scores.begin();
          auto it_end = w.c[c].ids_scores.end();
          auto dm = (c == 0) ? 0 : w.c[0].Ir.size();
          std::vector<std::size_t> idj(1);
          for (std::size_t j=0; j<d; j++) {
            idj[0] = w.ids_scores[j].first;
            // this assumes the ids_scores of the child is sorted on ids
            // find the first element that has an id not less than idj[0]
            auto l = std::lower_bound
              (it_lo, it_end, idj[0],
               [](const std::pair<std::size_t,real_t>& a,
                  const std::size_t& b) { return a.first < b; });
            it_lo = l;
            if (l != it_end && l->first == idj[0]) {
              // found the idj[0] in the child, hence, the column was
              // already computed by the child
              auto li = std::distance(w.c[c].ids_scores.begin(), l);
              for (std::size_t i=0; i<m; i++)
                w.S(dm+i, j) = w.c[c].S(w.c[c].Jr[i], li);
            } else {
              DenseMW_t colj(m, 1, w.S, dm, j);
              Aelem(w.c[c].Ir, idj, colj);
            }
          }
        }
      }
    }

    template<typename scalar_t> bool
    HSSMatrix<scalar_t>::compute_U_V_bases_ann
    (DenseM_t& S, const opts_t& opts,
     WorkCompressANN<scalar_t>& w, int depth) {
      auto rtol = opts.rel_tol() / w.lvl;
      auto atol = opts.abs_tol() / w.lvl;
      DenseM_t wSr(S);
      wSr.ID_row(U_.E(), U_.P(), w.Jr, rtol, atol, opts.max_rank(), depth);
      STRUMPACK_ID_FLOPS(ID_row_flops(wSr, U_.cols()));
      // exploit symmetry, set V = U
      V_.E() = U_.E();
      V_.P() = U_.P();
      w.Jc = w.Jr;
      U_.check();  assert(U_.cols() == w.Jr.size());
      V_.check();  assert(V_.cols() == w.Jc.size());
      auto d = S.cols();
      if (!(d >= this->cols() || int(d) >= opts.max_rank() ||
          (U_.cols() + opts.p() < d  &&
           V_.cols() + opts.p() < d))) {
        // std::cout << "WARNING: ID did not reach required accuracy:"
        //           << "\t increase k (number of ANN's), or Delta_d."
        //           << std::endl;
        return false;
      }
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
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_COMPRESS_KERNEL_HPP
