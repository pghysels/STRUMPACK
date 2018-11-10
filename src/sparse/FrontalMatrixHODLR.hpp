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
 *
 */
#ifndef FRONTAL_MATRIX_HODLR_HPP
#define FRONTAL_MATRIX_HODLR_HPP

#include <iostream>
#include <algorithm>
#include <memory>

#include "misc/TaskTimer.hpp"
#include "FrontalMatrix.hpp"
#include "CompressedSparseMatrix.hpp"
#include "MatrixReordering.hpp"
#include "HODLR/HODLRMatrix.hpp"
#include "HSS/HSSPartitionTree.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixHODLR
    : public FrontalMatrix<scalar_t,integer_t> {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;

  public:
    FrontalMatrixHODLR
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd);

    void extend_add_to_dense
    (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
     const FrontalMatrix<scalar_t,integer_t>* p, int task_depth) override;

    void sample_CB
    (const SPOptions<scalar_t>& opts, const DenseM_t& R, DenseM_t& Sr,
     DenseM_t& Sc, F_t* pa, int task_depth) override;

    void release_work_memory() override;
    void random_sampling
    (const SpMat_t& A, const SPOptions<scalar_t>& opts, DenseM_t& Rr,
     DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc, int etree_level,
     int task_depth); // TODO const?

    void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level=0, int task_depth=0) override;

    void forward_multifrontal_solve
    (DenseM_t& b, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;
    void backward_multifrontal_solve
    (DenseM_t& y, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;

    integer_t maximum_rank(int task_depth=0) const override;
    void print_rank_statistics(std::ostream &out) const override;
    std::string type() const override { return "FrontalMatrixHODLR"; }

    void set_HODLR_partitioning
    (const SPOptions<scalar_t>& opts,
     const HSS::HSSPartitionTree& sep_tree, bool is_root) override;

  private:
    FrontalMatrixHODLR(const FrontalMatrixHODLR&) = delete;
    FrontalMatrixHODLR& operator=(FrontalMatrixHODLR const&) = delete;

    void draw_node(std::ostream& of, bool is_root) const override;

    void multifrontal_factorization_node
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level, int task_depth);

    void fwd_solve_node
    (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const;
    void bwd_solve_node
    (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const;

    long long node_factor_nonzeros() const override;

    using FrontalMatrix<scalar_t,integer_t>::lchild_;
    using FrontalMatrix<scalar_t,integer_t>::rchild_;
    using FrontalMatrix<scalar_t,integer_t>::dim_sep;
    using FrontalMatrix<scalar_t,integer_t>::dim_upd;
    using FrontalMatrix<scalar_t,integer_t>::sep_begin_;
    using FrontalMatrix<scalar_t,integer_t>::sep_end_;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixHODLR<scalar_t,integer_t>::FrontalMatrixHODLR
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : FrontalMatrix<scalar_t,integer_t>
    (nullptr, nullptr, sep, sep_begin, sep_end, upd) {
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::release_work_memory() {
    std::cout << "TODO clean memory" << std::endl;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const FrontalMatrix<scalar_t,integer_t>* p, int task_depth) {
    std::cout << "extend add to dense" << std::endl;
    release_work_memory();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::sample_CB
  (const SPOptions<scalar_t>& opts, const DenseM_t& R,
   DenseM_t& Sr, DenseM_t& Sc, F_t* pa, int task_depth) {
    if (!dim_upd()) return;
    auto I = this->upd_to_parent(pa);
    auto cR = R.extract_rows(I);
    DenseM_t cSr(cR.rows(), cR.cols());
    DenseM_t cSc(cR.rows(), cR.cols());

    std::cout << "TODO sample with HODLR for CB" << std::endl;
    // _H.Schur_product_direct
    //   (_ULV, _Theta, _DUB01, _Phi,
    //    _ThetaVhatC_or_VhatCPhiC, cR, cSr, cSc);

    Sr.scatter_rows_add(I, cSr, task_depth);
    Sc.scatter_rows_add(I, cSc, task_depth);
    STRUMPACK_CB_SAMPLE_FLOPS(cSr.rows()*cSr.cols()*2);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (task_depth == 0)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      multifrontal_factorization_node(A, opts, etree_level, task_depth);
    else multifrontal_factorization_node(A, opts, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::random_sampling
  (const SpMat_t& A, const SPOptions<scalar_t>& opts, DenseM_t& Rr,
   DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc, int etree_level,
   int task_depth) {
    Sr.zero();
    Sc.zero();
    A.front_multiply
      (sep_begin_, sep_end_, this->upd_, Rr, Sr, Sc, task_depth);
    if (lchild_)
      lchild_->sample_CB(opts, Rr, Sr, Sc, this, task_depth);
    if (rchild_)
      rchild_->sample_CB(opts, Rr, Sr, Sc, this, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::multifrontal_factorization_node
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    bool tasked = task_depth < params::task_recursion_cutoff_level;
    if (tasked) {
      if (lchild_)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        lchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth+1);
      if (rchild_)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        rchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth+1);
#pragma omp taskwait
    } else {
      if (lchild_)
        lchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
      if (rchild_)
        rchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
    }

    auto mult = [&](DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc) {
      random_sampling(A, opts, Rr, Rc, Sr, Sc, etree_level, task_depth);
    };
    auto HODLRopts = opts.HODLR_options();

    std::cout << "TODO build HODLR matrix for this front" << std::endl;
    //_H.compress(mult, elem, HODLRopts);

    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
    if (dim_sep()) {
      if (etree_level > 0) {
        // _ULV = _H.partial_factor();
        // _H.Schur_update(_ULV, _Theta, _DUB01, _Phi);
        std::cout << "TODO factor FS part of front" << std::endl;
        std::cout << "TODO compute Schur complement update" << std::endl;
      } else
        //_ULV = _H.factor();
        std::cout << "TODO factor root front" << std::endl;
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    if (task_depth == 0)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      fwd_solve_node(b, work, etree_level, task_depth);
    else fwd_solve_node(b, work, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::fwd_solve_node
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t bupd(dim_upd(), b.cols(), work[0], 0, 0);
    bupd.zero();
    this->fwd_solve_phase1(b, bupd, work, etree_level, task_depth);

    std::cout << "TODO fwd solve" << std::endl;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    if (task_depth == 0)
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single
      bwd_solve_node(y, work, etree_level, task_depth);
    else bwd_solve_node(y, work, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::bwd_solve_node
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t yupd(dim_upd(), y.cols(), work[0], 0, 0);
    std::cout << "TODO bwd solve" << std::endl;

    this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
  }

  template<typename scalar_t,typename integer_t> integer_t
  FrontalMatrixHODLR<scalar_t,integer_t>::maximum_rank(int task_depth) const {
    integer_t r = /*_H.rank()*/ -1, rl = 0, rr = 0;
    if (lchild_)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
      rl = lchild_->maximum_rank(task_depth+1);
    if (rchild_)
#pragma omp task untied default(shared)                                 \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
      rr = rchild_->maximum_rank(task_depth+1);
#pragma omp taskwait
    return std::max(r, std::max(rl, rr));
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::print_rank_statistics
  (std::ostream &out) const {
    if (lchild_) lchild_->print_rank_statistics(out);
    if (rchild_) rchild_->print_rank_statistics(out);
    out << "# HODLRMatrix .... " << std::endl;
    // _H.print_info(out);
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixHODLR<scalar_t,integer_t>::node_factor_nonzeros() const {
    return -1;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::draw_node
  (std::ostream& of, bool is_root) const {
    std::cout << "TODO draw" << std::endl;
    // if (is_root) _H.draw(of, sep_begin_, sep_begin_);
    // else _H.child(0)->draw(of, sep_begin_, sep_begin_);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLR<scalar_t,integer_t>::set_HODLR_partitioning
  (const SPOptions<scalar_t>& opts, const HSS::HSSPartitionTree& sep_tree,
   bool is_root) {
    assert(sep_tree.size == dim_sep());

    std::cout << "create HODLR matrix hierarchy" << std::endl;

    if (is_root) {
      //_H = HODLR::HODLRMatrix<scalar_t>(sep_tree, opts.HODLR_options());
    } else {
      HSS::HSSPartitionTree hss_tree(this->dim_blk());
      hss_tree.c.reserve(2);
      hss_tree.c.push_back(sep_tree);
      hss_tree.c.emplace_back(dim_upd());
      hss_tree.c.back().refine(opts.HODLR_options().leaf_size());
      //_H = HODLR::HODLRMatrix<scalar_t>(hss_tree, opts.HODLR_options());
    }
  }

} // end namespace strumpack

#endif //FRONTAL_MATRIX_HODLR_HPP
