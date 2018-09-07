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

    template<typename scalar_t> void HSSMatrix<scalar_t>::compress_ann
    (DenseM_t& ann, DenseM_t& scores, const elem_t& Aelem, const opts_t& opts) {

      std::cout << "---> USING COMPRESS_ANN <---" << std::endl;
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
        compute_U_V_bases_ANN(w.S, w.S, opts, w, d, 0);
       // std::cout << "computed id " << std::endl;
        this->_U_state = this->_V_state = State::COMPRESSED;
      }
    }
  

    //NEW Main routine to change
    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compute_local_samples_ann
    (DenseM_t& ann, DenseM_t& scores, WorkCompressANN<scalar_t>& w,
     const elem_t& Aelem, int d) {
      std::size_t ann_number = ann.rows();
      if (this->leaf()) {

        //std::cout << this->rows() << std::endl;
        std::vector<std::size_t> I;
        I.reserve(this->rows());
        for (std::size_t i=0; i<this->rows(); i++)
          I.push_back(i+w.offset.first);

        // TODO construct w.Scolids
        // combine non-diagonal neibs of all points in the leaf

        std::vector<std::size_t> leaf_neibs;
        std::vector<double> leaf_neib_scores;

        for (std::size_t i = w.offset.first; i < w.offset.first + this->rows(); i++)
        {
          for (std::size_t j = 0; j < ann_number; j++)
          {
            if ((ann(j, i) < w.offset.first) || (ann(j, i) >= w.offset.first + this->rows()))
            {
              leaf_neibs.push_back(ann(j, i));
              leaf_neib_scores.push_back(scores(j, i));
            }
          }
        }


        // remove duplicates 
        std::vector<std::size_t> order = find_sort_permutation(leaf_neibs);
        leaf_neibs = apply_permutation(leaf_neibs, order);
        leaf_neib_scores = apply_permutation(leaf_neib_scores, order);

        std::size_t cur = 0;
        for (std::size_t i = 1; i < leaf_neibs.size(); i++)
        {
          if (leaf_neibs[i] > leaf_neibs[i-1])
          {
            cur++;
            leaf_neibs[cur] = leaf_neibs[i];
            leaf_neib_scores[cur] = leaf_neib_scores[i];
          }
          else
          {
            if (leaf_neib_scores[cur] > leaf_neib_scores[i])
            {
             leaf_neib_scores[cur] = leaf_neib_scores[i];
            }
          }
        }
        
        leaf_neibs.resize(cur+1);
        leaf_neib_scores.resize(cur+1);

        //  if (w.offset.first == 0)
        //  {
        //      for (std::size_t j = 0; j < leaf_neibs.size(); j++)
        //    {
        //      std::cout << leaf_neibs[j] << ' ';
        //    }

        //    for (std::size_t j = 0; j < leaf_neibs.size(); j++)
        //    {
        //      std::cout << leaf_neib_scores[j] << ' ';
        //    }
        //  std::cout << std::endl;
        //  }

         // make at most d column samples
         if (leaf_neibs.size() < d)
         {
           for (std::size_t j = 0; j < leaf_neibs.size(); j++)
          {
            w.Scolids.push_back(leaf_neibs[j]);
            w.Scolscs.push_back(leaf_neib_scores[j]);
          }
         } 
         else
         {
          std::vector<std::size_t> order = find_sort_permutation(leaf_neib_scores);
          leaf_neibs = apply_permutation(leaf_neibs, order);
          leaf_neib_scores = apply_permutation(leaf_neib_scores, order);
          for (std::size_t j = 0; j < d; j++)
          {
            w.Scolids.push_back(leaf_neibs[j]);
            w.Scolscs.push_back(leaf_neib_scores[j]);
          }
           
         }
        w.S = DenseM_t(I.size(), w.Scolids.size());
        Aelem(I, w.Scolids, w.S);

      } else {

        // TODO
        std::vector<std::size_t> I;
        for (std::size_t i = 0; i < w.c[0].Ir.size(); i++)
        {
          I.push_back(w.c[0].Ir[i]);
        }
        for (std::size_t i = 0; i < w.c[1].Ir.size(); i++)
        {
          I.push_back(w.c[1].Ir[i]);
        }
        std::vector<std::size_t> colids;
        std::vector<double> colscs;

      //  std::cout << "ch1 node offsets: " << w.c[0].offset.first << "; " << w.c[0].offset.second << std::endl;
      //  std::cout << "ch2 node offsets: " << w.c[1].offset.first << "; " << w.c[1].offset.second << std::endl;
      //  std::cout << "current node " << w.offset.first << " - " << w.offset.first + this->rows() << std::endl;
         
       for (std::size_t i = 0; i < w.c[0].Scolids.size(); i++)
       {
        if ((w.c[0].Scolids[i] < w.offset.first) ||  (w.c[0].Scolids[i] >= w.offset.first + this->rows()))
        {
          colids.push_back(w.c[0].Scolids[i]);
          colscs.push_back(w.c[0].Scolscs[i]);
        }
       }
        for (std::size_t i = 0; i < w.c[1].Scolids.size(); i++)
       {
           if ((w.c[1].Scolids[i] < w.offset.first) ||  (w.c[1].Scolids[i] >= w.offset.first + this->rows()))
        {
          colids.push_back(w.c[1].Scolids[i]);
          colscs.push_back(w.c[1].Scolscs[i]);
        }
       }

        std::vector<std::size_t> order = find_sort_permutation(colids);
        colids = apply_permutation(colids, order);
        colscs = apply_permutation(colscs, order);

        std::size_t cur = 0;
        for (std::size_t i = 1; i < colids.size(); i++)
        {
          if (colids[i] > colids[i-1])
          {
            cur++;
            colids[cur] = colids[i];
            colscs[cur] = colscs[i];
          }
          else
          {
            if (colscs[cur] > colscs[i])
            {
             colscs[cur] = colscs[i];
            }
          }
        }
        
        colids.resize(cur+1);
        colscs.resize(cur+1);

       // std::cout << "non-leaf: " << colids.size() << " samples" << std::endl;
       
        if (colids.size() < d)
         {
            for (std::size_t j = 0; j < colids.size(); j++)
           {
             // if we want to add more samples until d, it is here
            w.Scolids.push_back(colids[j]);
            w.Scolscs.push_back(colscs[j]);
          }


         } else
         {
          std::vector<std::size_t> order = find_sort_permutation(colscs);
          colids = apply_permutation(colids, order);
          colscs = apply_permutation(colscs, order);
          for (std::size_t j = 0; j < d; j++)
          {
            w.Scolids.push_back(colids[j]);
            w.Scolscs.push_back(colscs[j]);
          }
         //std::cout << "non-leaf: " << w.Scolids.size() << std::endl;
          }
        //     std::cout << w.Scolids.size() << " by " << I.size() << std::endl;
        //   for (std::size_t j = 0; j < w.Scolids.size(); j++)
        //     {
        //      std::cout << w.Scolids[j] << ' ';
        //    }
        //  std::cout << std::endl;
        //    for (std::size_t j = 0; j < w.Scolids.size(); j++)
        //     {
        //      std::cout << w.Scolscs[j] << ' ';
        //    }
        //  std::cout << std::endl;
        w.S = DenseM_t(I.size(), w.Scolids.size());
        Aelem(I, w.Scolids, w.S);


      }
    }

 template<typename scalar_t> bool HSSMatrix<scalar_t>::compute_U_V_bases_ANN
    (DenseM_t& Sr, DenseM_t& Sc, const opts_t& opts,
     WorkCompressANN<scalar_t>& w, int d, int depth) {
       
      auto rtol = opts.rel_tol() / w.lvl;
      auto atol = opts.abs_tol() / w.lvl;
      
      auto u_rows = this->leaf() ? this->rows() :
        this->_ch[0]->U_rank()+this->_ch[1]->U_rank();
      

      //DenseM_t wSr(u_rows, d, Sr, w.offset.second, 0);
      DenseM_t wSr(u_rows, Sr.cols(), Sr, 0, 0);

      wSr.ID_row(_U.E(), _U.P(), w.Jr, rtol, atol, opts.max_rank(), depth);
      //  STRUMPACK_ID_FLOPS(ID_row_flops(wSr, _U.cols()));
      
      // std::cout << "ID_u found " << std::endl;
      {
        auto v_rows = this->leaf() ? this->rows() :
          this->_ch[0]->V_rank()+this->_ch[1]->V_rank();
        DenseM_t wSc(v_rows, Sc.cols(), Sc, 0, 0);
      //  DenseM_t wSc(v_rows, d, Sc, w.offset.second, 0);
        wSc.ID_row(_V.E(), _V.P(), w.Jc, rtol, atol,
                   opts.max_rank(), depth);
        STRUMPACK_ID_FLOPS(ID_row_flops(wSc, _V.cols()));
      }
    
      _U.check();  assert(_U.cols() == w.Jr.size());
      _V.check();  assert(_V.cols() == w.Jc.size());
      if (d-opts.p() >= opts.max_rank() ||
          (int(_U.cols()) < d - opts.p() &&
           int(_V.cols()) < d - opts.p())) {
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
        return true;
      } else {
        w.Jr.clear();
        w.Jc.clear();
        return false;
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_COMPRESS_KERNEL_HPP
