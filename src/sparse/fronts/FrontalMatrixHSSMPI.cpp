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
 * five (5) year renewals, the U.S. Government igs granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
#include "FrontalMatrixHSSMPI.hpp"
#include "ExtendAdd.hpp"
#include "sparse/CSRGraph.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  FrontalMatrixHSSMPI<scalar_t,integer_t>::FrontalMatrixHSSMPI
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd, const MPIComm& comm, int total_procs)
    : FrontalMatrixMPI<scalar_t,integer_t>
    (sep, sep_begin, sep_end, upd, comm, total_procs) {
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::release_work_memory() {
    thetaVhatC_.clear();
    VhatCPhiC_.clear();
    Vhat_.clear();
    if (H_) H_->delete_trailing_block();
    DUB01_.clear();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::random_sampling
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   const DistM_t& R, DistM_t& Sr, DistM_t& Sc) {
    Sr.zero();
    Sc.zero();
    {
      TIMER_TIME(TaskType::FRONT_MULTIPLY_2D, 1, t_fmult);
      A.front_multiply_2d(sep_begin_, sep_end_, this->upd_, R, Sr, Sc, 0);
    }
    TIMER_TIME(TaskType::UUTXR, 1, t_UUtxR);
    sample_children_CB(opts, R, Sr, Sc);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::sample_CB
  (const DistM_t& R, DistM_t& Sr, DistM_t& Sc, F_t* pa) const {
    if (!dim_upd()) return;
    if (Comm().is_null()) return;
    auto b = R.cols();
    Sr = DistM_t(grid(), dim_upd(), b);
    Sc = DistM_t(grid(), dim_upd(), b);
    TIMER_TIME(TaskType::HSS_SCHUR_PRODUCT, 2, t_sprod);
    H_->Schur_product_direct
      (theta_, Vhat_, DUB01_, phi_, thetaVhatC_, VhatCPhiC_, R, Sr, Sc);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::sample_children_CB
  (const SPOptions<scalar_t>& opts, const DistM_t& R,
   DistM_t& Sr, DistM_t& Sc) {
    if (!lchild_ && !rchild_) return;
    DistM_t cSrl, cSrr, cScl, cScr, Rl, Rr;
    DenseM_t seqSrl, seqSrr, seqScl, seqScr, seqRl, seqRr;
    if (lchild_) {
      lchild_->extract_from_R2D(R, Rl, seqRl, this, visit(lchild_));
      seqSrl = DenseM_t(seqRl.rows(), seqRl.cols());
      seqScl = DenseM_t(seqRl.rows(), seqRl.cols());
      seqSrl.zero();
      seqScl.zero();
    }
    if (rchild_) {
      rchild_->extract_from_R2D(R, Rr, seqRr, this, visit(rchild_));
      seqSrr = DenseM_t(seqRr.rows(), seqRr.cols());
      seqScr = DenseM_t(seqRr.rows(), seqRr.cols());
      seqSrr.zero();
      seqScr.zero();
    }
    if (visit(lchild_))
      lchild_->sample_CB(opts, Rl, cSrl, cScl, seqRl, seqSrl, seqScl, this);
    if (visit(rchild_))
      rchild_->sample_CB(opts, Rr, cSrr, cScr, seqRr, seqSrr, seqScr, this);
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    if (visit(lchild_)) {
      lchild_->skinny_ea_to_buffers(cSrl, seqSrl, sbuf, this);
      lchild_->skinny_ea_to_buffers(cScl, seqScl, sbuf, this);
    }
    if (visit(rchild_)) {
      rchild_->skinny_ea_to_buffers(cSrr, seqSrr, sbuf, this);
      rchild_->skinny_ea_to_buffers(cScr, seqScr, sbuf, this);
    }
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    if (lchild_) {
      lchild_->skinny_ea_from_buffers(Sr, pbuf.data(), this);
      lchild_->skinny_ea_from_buffers(Sc, pbuf.data(), this);
    }
    if (rchild_) {
      rchild_->skinny_ea_from_buffers
        (Sr, pbuf.data()+this->master(rchild_), this);
      rchild_->skinny_ea_from_buffers
        (Sc, pbuf.data()+this->master(rchild_), this);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const Opts_t& opts, int etree_level, int task_depth) {
    if (visit(lchild_))
      lchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    if (visit(rchild_))
      rchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    if (!dim_blk()) return;

    TaskTimer t("FrontalMatrixHSSMPI_factor");
    if (opts.print_compressed_front_stats()) t.start();
    auto mult = [&](DistM_t& R, DistM_t& Sr, DistM_t& Sc) {
      TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
      random_sampling(A, opts, R, Sr, Sc);
    };
    auto elem_blocks = [&]
      (const std::vector<std::vector<std::size_t>>& I,
       const std::vector<std::vector<std::size_t>>& J,
       std::vector<DistMW_t>& B) {
      element_extraction(A, I, J, B);
    };
    H_->compress(mult, elem_blocks, opts.HSS_options());

    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();

    if (dim_sep()) {
      if (etree_level > 0) {
        {
          TIMER_TIME(TaskType::HSS_PARTIALLY_FACTOR, 0, t_pfact);
          H_->partial_factor();
        }
        TIMER_TIME(TaskType::HSS_COMPUTE_SCHUR, 0, t_comp_schur);
        H_->Schur_update(theta_, Vhat_, DUB01_, phi_);
        if (theta_.cols() < phi_.cols()) {
          VhatCPhiC_ = DistM_t(phi_.grid(), Vhat_.cols(), phi_.rows());
          gemm(Trans::C, Trans::C, scalar_t(1.), Vhat_, phi_,
               scalar_t(0.), VhatCPhiC_);
          STRUMPACK_SCHUR_FLOPS
            (gemm_flops(Trans::C, Trans::C, scalar_t(1.), Vhat_, phi_,
                        scalar_t(0.)));
        } else {
          thetaVhatC_ = DistM_t(theta_.grid(), theta_.rows(), Vhat_.rows());
          gemm(Trans::N, Trans::C, scalar_t(1.), theta_, Vhat_,
               scalar_t(0.), thetaVhatC_);
          STRUMPACK_SCHUR_FLOPS
            (gemm_flops(Trans::N, Trans::C, scalar_t(1.), theta_, Vhat_,
                        scalar_t(0.)));
        }
      } else {
        TIMER_TIME(TaskType::HSS_FACTOR, 0, t_fact);
        H_->factor();
      }
    }
    if (opts.print_compressed_front_stats()) {
      auto time = t.elapsed();
      auto rank = H_->max_rank();
      std::size_t nnzH = (H_ ? H_->total_nonzeros() : 0);
      std::size_t nnzULV = H_->total_factor_nonzeros();
      std::size_t nnzT = theta_.total_nonzeros();
      std::size_t nnzP = phi_.total_nonzeros();
      std::size_t nnzV = Vhat_.total_nonzeros();
      if (Comm().is_root())
        std::cout << "#   - HSSMPI front: Nsep= " << dim_sep()
                  << " , Nupd= " << dim_upd()
                  << " , nnz(H)= " << nnzH
                  << " , nnz(ULV)= " << nnzULV
                  << " , nnz(Schur)= " << (nnzT + nnzP + nnzV)
                  << " , maxrank= " << rank
                  << " , " << (float(nnzH + nnzULV + nnzT + nnzP + nnzV)
                               / (float(dim_blk())*dim_blk()) * 100.)
                  << " %compression, time= " << time
                  << " sec" << std::endl;
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
   int etree_level) const {
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    if (visit(lchild_))
      lchild_->forward_multifrontal_solve
        (bloc, bdist, CBl, seqCBl, etree_level+1);
    if (visit(rchild_))
      rchild_->forward_multifrontal_solve
        (bloc, bdist, CBr, seqCBr, etree_level+1);
    DistM_t& b = bdist[this->sep_];
    bupd = DistM_t(H_->grid(), dim_upd(), b.cols());
    bupd.zero();
    this->extend_add_b(b, bupd, CBl, CBr, seqCBl, seqCBr);
    if (etree_level) {
      if (theta_.cols() && phi_.cols()) {
        TIMER_TIME(TaskType::SOLVE_LOWER, 0, t_reduce);
        ULVwork_ = std::unique_ptr<HSS::WorkSolveMPI<scalar_t>>
          (new HSS::WorkSolveMPI<scalar_t>());
        DistM_t lb(H_->child(0)->grid(H_->grid_local()), b.rows(), b.cols());
        copy(b.rows(), b.cols(), b, 0, 0, lb, 0, 0, grid()->ctxt_all());
        H_->child(0)->forward_solve(*ULVwork_, lb, true);
        DistM_t rhs(H_->grid(), theta_.cols(), bupd.cols());
        copy(rhs.rows(), rhs.cols(), ULVwork_->reduced_rhs, 0, 0,
             rhs, 0, 0, grid()->ctxt_all());
        ULVwork_->reduced_rhs.clear();
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), theta_, rhs,
               scalar_t(1.), bupd);
      }
    } else {
      TIMER_TIME(TaskType::SOLVE_LOWER_ROOT, 0, t_solve);
      DistM_t lb(H_->grid(), b.rows(), b.cols());
      copy(b.rows(), b.cols(), b, 0, 0, lb, 0, 0, grid()->ctxt_all());
      ULVwork_ = std::unique_ptr<HSS::WorkSolveMPI<scalar_t>>
        (new HSS::WorkSolveMPI<scalar_t>());
      H_->forward_solve(*ULVwork_, lb, false);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
   int etree_level) const {
    DistM_t& y = ydist[this->sep_];
    {
      TIMER_TIME(TaskType::SOLVE_UPPER, 0, t_expand);
      if (etree_level) {
        if (phi_.cols() && theta_.cols()) {
          DistM_t ly
            (H_->child(0)->grid(H_->grid_local()), y.rows(), y.cols());
          copy(y.rows(), y.cols(), y, 0, 0, ly, 0, 0, grid()->ctxt_all());
          if (dim_upd()) {
            // TODO can these copies be avoided??
            DistM_t wx
              (H_->grid(), phi_.cols(), yupd.cols(), ULVwork_->x, grid()->ctxt_all());
            DistM_t yupdHctxt
              (H_->grid(), dim_upd(), y.cols(), yupd, grid()->ctxt_all());
            gemm(Trans::C, Trans::N, scalar_t(-1.), phi_, yupdHctxt,
                 scalar_t(1.), wx);
            copy(wx.rows(), wx.cols(), wx, 0, 0, ULVwork_->x, 0, 0, grid()->ctxt_all());
          }
          H_->child(0)->backward_solve(*ULVwork_, ly);
          copy(y.rows(), y.cols(), ly, 0, 0, y, 0, 0, grid()->ctxt_all());
          ULVwork_.reset();
        }
      } else {
        DistM_t ly(H_->grid(), y.rows(), y.cols());
        copy(y.rows(), y.cols(), y, 0, 0, ly, 0, 0, grid()->ctxt_all());
        H_->backward_solve(*ULVwork_, ly);
        copy(y.rows(), y.cols(), ly, 0, 0, y, 0, 0, grid()->ctxt_all());
      }
    }
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    this->extract_b(y, yupd, CBl, CBr, seqCBl, seqCBr);
    if (visit(lchild_))
      lchild_->backward_multifrontal_solve
        (yloc, ydist, CBl, seqCBl, etree_level+1);
    if (visit(rchild_))
      rchild_->backward_multifrontal_solve
        (yloc, ydist, CBr, seqCBr, etree_level+1);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::element_extraction
  (const SpMat_t& A, const std::vector<std::vector<std::size_t>>& I,
   const std::vector<std::vector<std::size_t>>& J,
   std::vector<DistMW_t>& B) const {
    //if (I.empty() || J.empty()) return;
    std::vector<std::vector<std::size_t>> gI(I.size()), gJ(J.size());
    const std::size_t dsep = dim_sep();
    for (std::size_t j=0; j<I.size(); j++) {
      gI[j].reserve(I[j].size());
      gJ[j].reserve(J[j].size());
      for (auto i : I[j]) {
        assert(i < std::size_t(dim_blk()));
        gI[j].push_back((i < dsep) ? i+sep_begin_ : this->upd_[i-dsep]);
      }
      for (auto i : J[j]) {
        assert(i < std::size_t(dim_blk()));
        gJ[j].push_back((i < dsep) ? i+sep_begin_ : this->upd_[i-dsep]);
      }
    }
    this->extract_2d(A, gI, gJ, B);
  }

  /**
   * Extract from (HSS - theta Vhat^* phi*^).
   *
   * Note that B has the same context as this front, otherwise the
   * communication pattern would be hard to figure out.
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::extract_CB_sub_matrix_2d
  (const std::vector<std::vector<std::size_t>>& I,
   const std::vector<std::vector<std::size_t>>& J,
   std::vector<DistM_t>& B) const {
    if (Comm().is_null() || !dim_upd()) return;
    TIMER_TIME(TaskType::HSS_EXTRACT_SCHUR, 3, t_ex_schur);
    auto nB = I.size();
    std::vector<std::vector<std::size_t>>
      lI(nB), lJ(nB), oI(nB), oJ(nB), gI(nB), gJ(nB);
    for (std::size_t i=0; i<nB; i++) {
      this->find_upd_indices(I[i], lI[i], oI[i]);
      this->find_upd_indices(J[i], lJ[i], oJ[i]);
      gI[i] = lI[i];  for (auto& idx : gI[i]) idx += dim_sep();
      gJ[i] = lJ[i];  for (auto& idx : gJ[i]) idx += dim_sep();
    }
    std::vector<DistM_t> e_vec = H_->extract(gI, gJ, grid());

    std::vector<std::vector<scalar_t>> sbuf(this->P());
    // TODO extract all rows at once?????
    for (std::size_t i=0; i<nB; i++) {
      if (theta_.cols() < phi_.cols()) {
        DistM_t tr(grid(), lI[i].size(), theta_.cols(),
                   theta_.extract_rows(lI[i]), grid()->ctxt_all());
        DistM_t tc(grid(), VhatCPhiC_.rows(), lJ[i].size(),
                   VhatCPhiC_.extract_cols(lJ[i]), grid()->ctxt_all());
        gemm(Trans::N, Trans::N, scalar_t(-1), tr, tc, scalar_t(1.), e_vec[i]);
        STRUMPACK_EXTRACTION_FLOPS
          (gemm_flops(Trans::N, Trans::N, scalar_t(-1), tr, tc, scalar_t(1.)));
      } else {
        DistM_t tr(grid(), lI[i].size(), thetaVhatC_.cols(),
                   thetaVhatC_.extract_rows(lI[i]), grid()->ctxt_all());
        DistM_t tc(grid(), lJ[i].size(), phi_.cols(),
                   phi_.extract_rows(lJ[i]), grid()->ctxt_all());
        gemm(Trans::N, Trans::C, scalar_t(-1), tr, tc, scalar_t(1.), e_vec[i]);
        STRUMPACK_EXTRACTION_FLOPS
          (gemm_flops(Trans::N, Trans::C, scalar_t(-1), tr, tc, scalar_t(1.)));
      }
      ExtendAdd<scalar_t,integer_t>::extend_copy_to_buffers
        (e_vec[i], oI[i], oJ[i], B[i], sbuf);
    }
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    for (std::size_t i=0; i<nB; i++)
      ExtendAdd<scalar_t,integer_t>::extend_copy_from_buffers
        (B[i], oI[i], oJ[i], e_vec[i], pbuf);
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixHSSMPI<scalar_t,integer_t>::node_factor_nonzeros() const {
    // TODO does this return only when root??
    return (H_ ? H_->nonzeros() : 0) + H_->factor_nonzeros()
      + theta_.nonzeros() + phi_.nonzeros();
  }

  template<typename scalar_t,typename integer_t> integer_t
  FrontalMatrixHSSMPI<scalar_t,integer_t>::front_rank(int task_depth) const {
    return H_->max_rank();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHSSMPI<scalar_t,integer_t>::partition
  (const Opts_t& opts, const SpMat_t& A,
   integer_t* sorder, bool is_root, int task_depth) {
    if (Comm().is_null()) return;
    auto g = A.extract_graph
      (opts.separator_ordering_level(), sep_begin_, sep_end_);
    auto sep_tree = g.recursive_bisection
      (opts.compression_leaf_size(), 0,
       sorder+sep_begin_, nullptr, 0, 0, dim_sep());
    for (integer_t i=sep_begin_; i<sep_end_; i++)
      sorder[i] = sorder[i] + sep_begin_;
    if (is_root)
      H_ = std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>>
        (new HSS::HSSMatrixMPI<scalar_t>
         (sep_tree, grid(), opts.HSS_options()));
    else {
      structured::ClusterTree hss_tree(dim_blk());
      hss_tree.c.reserve(2);
      hss_tree.c.push_back(sep_tree);
      hss_tree.c.emplace_back(dim_upd());
      hss_tree.c.back().refine(opts.HSS_options().leaf_size());
      H_ = std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>>
        (new HSS::HSSMatrixMPI<scalar_t>
         (hss_tree, grid(), opts.HSS_options()));
    }
  }

  // explicit template instantiations
  template class FrontalMatrixHSSMPI<float,int>;
  template class FrontalMatrixHSSMPI<double,int>;
  template class FrontalMatrixHSSMPI<std::complex<float>,int>;
  template class FrontalMatrixHSSMPI<std::complex<double>,int>;

  template class FrontalMatrixHSSMPI<float,long int>;
  template class FrontalMatrixHSSMPI<double,long int>;
  template class FrontalMatrixHSSMPI<std::complex<float>,long int>;
  template class FrontalMatrixHSSMPI<std::complex<double>,long int>;

  template class FrontalMatrixHSSMPI<float,long long int>;
  template class FrontalMatrixHSSMPI<double,long long int>;
  template class FrontalMatrixHSSMPI<std::complex<float>,long long int>;
  template class FrontalMatrixHSSMPI<std::complex<double>,long long int>;

} // end namespace strumpack
