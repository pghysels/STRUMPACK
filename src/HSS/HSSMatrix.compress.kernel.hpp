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

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> void HSSMatrix<scalar_t>::compress_ann
    (DenseM_t& ann, DenseM_t& scores, const elem_t& Aelem, const opts_t& opts) {

      //int d_old = 0, d = opts.d0() + opts.p();


      int d = opts.d0();
      auto n = this->cols();

      WorkCompressANN<scalar_t> w;

      compress_recursive_ann(ann, scores, Aelem, opts, w, d);


      // if (!this->is_compressed()) {
      //   // d_old = d;
      //   // d = 2 * (d_old - opts.p()) + opts.p();
      // }
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress_recursive_ann
    (DenseM_t& ann, DenseM_t& scores,
     const elem_t& Aelem, const opts_t& opts,
     WorkCompressANN<scalar_t>& w, int d) {
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
        this->_ch[0]->compress_recursive_ann
          (ann, scores, Aelem, opts, w.c[0], d);
        this->_ch[1]->compress_recursive_ann
          (ann, scores, Aelem, opts, w.c[1], d);
        if (!this->_ch[0]->is_compressed() ||
            !this->_ch[1]->is_compressed()) return;
        if (this->is_untouched()) {
          _B01 = DenseM_t(this->_ch[0]->U_rank(), this->_ch[1]->V_rank());
          Aelem(w.c[0].Ir, w.c[1].Ic, _B01);
          _B10 = DenseM_t(this->_ch[1]->U_rank(), this->_ch[0]->V_rank());
          Aelem(w.c[1].Ir, w.c[0].Ic, _B10);
        }
      }
      if (w.lvl == 0) this->_U_state = this->_V_state = State::COMPRESSED;
      else {
        compute_local_samples_ann(ann, scores, w, Aelem, d);

        // TODO get this working
        //compute_U_V_bases(w.S, w.S, opts, w, d);

        this->_U_state = this->_V_state = State::COMPRESSED;
      }
    }

    //NEW Main routine to change
    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compute_local_samples_ann
    (DenseM_t& ann, DenseM_t& scores, WorkCompressANN<scalar_t>& w,
     const elem_t& Aelem, int d) {
      if (this->leaf()) {

        std::vector<std::size_t> I;
        I.reserve(this->rows());
        for (std::size_t i=0; i<this->rows(); i++)
          I.push_back(i+w.offset.first);

        // TODO construct w.Scolids

        Aelem(I, w.Scolids, w.S);

      } else {

        // TODO



      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_COMPRESS_KERNEL_HPP
