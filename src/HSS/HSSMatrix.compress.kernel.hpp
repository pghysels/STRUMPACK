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

#include "misc/RandomWrapper.hpp"
#include "misc/Tools.hpp"

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress_ann
    (DenseMatrix<std::uint32_t>& ann, DenseM_t& scores,
     const elem_t& Aelem, const opts_t& opts) {
      std::cout << "---> USING COMPRESS_ANN <---" << std::endl;
      WorkCompressANN<scalar_t> w;
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      compress_recursive_ann
        (ann, scores, Aelem, opts, w, this->_openmp_task_depth);
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress_recursive_ann
    (DenseMatrix<std::uint32_t>& ann, DenseM_t& scores, const elem_t& Aelem,
     const opts_t& opts, WorkCompressANN<scalar_t>& w, int depth) {
      if (this->leaf()) {
        if (this->is_untouched()) {
          std::vector<std::size_t> I, J;
          I.reserve(this->rows());
          J.reserve(this->cols());
          for (std::size_t i=0; i<this->rows(); i++)
            I.push_back(i+w.offset.first);
          for (std::size_t j=0; j<this->cols(); j++)
            J.push_back(j+w.offset.second);
          _D = DenseM_t(this->rows(), this->cols());
          Aelem(I, J, _D);
        }
      } else {
        w.split(this->_ch[0]->dims());
        bool tasked = depth < params::task_recursion_cutoff_level;
        if (tasked) {
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          this->_ch[0]->compress_recursive_ann
            (ann, scores, Aelem, opts, w.c[0], depth+1);
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          this->_ch[1]->compress_recursive_ann
            (ann, scores, Aelem, opts, w.c[1], depth+1);
#pragma omp taskwait
        } else {
          this->_ch[0]->compress_recursive_ann
            (ann, scores, Aelem, opts, w.c[0], depth);
          this->_ch[1]->compress_recursive_ann
            (ann, scores, Aelem, opts, w.c[1], depth);
        }
        if (!this->_ch[0]->is_compressed() ||
            !this->_ch[1]->is_compressed())
          return;
        if (this->is_untouched()) {
          _B01 = DenseM_t(this->_ch[0]->U_rank(), this->_ch[1]->V_rank());
          Aelem(w.c[0].Ir, w.c[1].Ic, _B01);
          _B10 = _B01.transpose();
        }
      }
      if (w.lvl == 0)
        this->_U_state = this->_V_state = State::COMPRESSED;
      else {
        compute_local_samples_ann(ann, scores, w, Aelem, opts);
        compute_U_V_bases_ann(w.S, opts, w, depth);
        this->_U_state = this->_V_state = State::COMPRESSED;
        w.c.clear();
      }
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compute_local_samples_ann
    (DenseMatrix<std::uint32_t>& ann, DenseM_t& scores,
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
                     [](const std::pair<std::size_t,double>& a,
                        const std::pair<std::size_t,double>& b) {
                       return a.first == b.first; }), w.ids_scores.end());

      // maximum number of samples
      std::size_t d_max = this->leaf() ?
        I.size() + opts.dd() :   // leaf size + some oversampling
        w.c[0].Ir.size() + w.c[1].Ir.size() + opts.dd();
      auto d = std::min(w.ids_scores.size(), d_max);

      if (d < w.ids_scores.size()) {
        // sort based on scores, keep only d closest
        std::nth_element
          (w.ids_scores.begin(), w.ids_scores.begin()+d, w.ids_scores.end(),
           [](const std::pair<std::size_t,double>& a,
              const std::pair<std::size_t,double>& b) {
            return a.second < b.second; });
        w.ids_scores.resize(d);
      }

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
               [](const std::pair<std::size_t,double>& a,
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
      wSr.ID_row(_U.E(), _U.P(), w.Jr, rtol, atol, opts.max_rank(), depth);
      STRUMPACK_ID_FLOPS(ID_row_flops(wSr, _U.cols()));
      // exploit symmetry, set V = U
      _V.E() = _U.E();
      _V.P() = _U.P();
      w.Jc = w.Jr;
      _U.check();  assert(_U.cols() == w.Jr.size());
      _V.check();  assert(_V.cols() == w.Jc.size());
      bool accurate = true;
      int d = S.cols();
      if (!(d - opts.p() >= opts.max_rank() ||
            (int(_U.cols()) < d - opts.p() &&
             int(_V.cols()) < d - opts.p()))) {
        accurate = false;
        std::cout << "WARNING: ID did not reach required accuracy:"
                  << "\t increase k (number of ANN's), or Delta_d."
                  << std::endl;
      }
      this->_U_rank = _U.cols();  this->_U_rows = _U.rows();
      this->_V_rank = _V.cols();  this->_V_rows = _V.rows();
      w.Ir.reserve(_U.cols());
      w.Ic.reserve(_V.cols());
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
      return accurate;
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_COMPRESS_KERNEL_HPP
