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
#ifndef FRONTAL_MATRIX_HPP
#define FRONTAL_MATRIX_HPP

#include <iostream>
#include <algorithm>
#include <random>
#include <vector>

#include "misc/TaskTimer.hpp"
#include "StrumpackParameters.hpp"
#include "CompressedSparseMatrix.hpp"
#include "MatrixReordering.hpp"
#include "HSS/HSSMatrix.hpp"
#if defined(_OPENMP)
#include "omp.h"
#endif

namespace strumpack {

  // forward declaration
  template<typename scalar_t> class DistributedMatrix;
  template<typename scalar_t,typename integer_t> class FrontalMatrixDense;

  template<typename scalar_t,typename integer_t> class FrontalMatrix {
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;

  public:
    FrontalMatrix
    (F_t* _lchild, F_t* _rchild, integer_t _sep, integer_t _sep_begin,
     integer_t _sep_end, std::vector<integer_t>& _upd);
    virtual ~FrontalMatrix() {
      delete lchild;
      delete rchild;
    }

    inline integer_t dim_sep() const { return sep_end - sep_begin; }
    inline integer_t dim_upd() const { return upd.size(); }
    inline integer_t dim_blk() const { return dim_sep() + dim_upd(); }

    void find_upd_indices
    (const std::vector<std::size_t>& I, std::vector<std::size_t>& lI,
     std::vector<std::size_t>& oI) const;
    std::vector<std::size_t> upd_to_parent
    (const F_t* pa, std::size_t& upd2sep) const;
    std::vector<std::size_t> upd_to_parent(const F_t* pa) const;

    virtual void release_work_memory() = 0;

    virtual void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level=0, int task_depth=0) = 0;

    virtual void extend_add_to_dense
    (FrontalMatrixDense<scalar_t,integer_t>* p, int task_depth) {}

    virtual void sample_CB
    (const SPOptions<scalar_t>& opts, const DenseM_t& R,
     DenseM_t& Sr, DenseM_t& Sc, F_t* parent, int task_depth=0) = 0;
    virtual void extract_CB_sub_matrix
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DenseM_t& B, int task_depth) const = 0;

    void multifrontal_solve(DenseM_t& b) const;
    virtual void forward_multifrontal_solve
    (DenseM_t& b, DenseM_t* work, int etree_level=0,
     int task_depth=0) const {};
    virtual void backward_multifrontal_solve
    (DenseM_t& y, DenseM_t* work, int etree_level=0,
     int task_depth=0) const {};
    void extend_add_b
    (const F_t* ch, DenseM_t& b, DenseM_t& bupd, const DenseM_t& CB) const;
    void extract_b
    (const F_t* ch, const DenseM_t& y, const DenseM_t& yupd,
     DenseM_t& CB) const;

    void multifrontal_solve(DenseM_t& bloc, DistM_t* bdist) const;
    virtual void forward_multifrontal_solve
    (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
     int etree_level=0) const;
    virtual void backward_multifrontal_solve
    (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
     int etree_level=0) const;

    void look_left(DistM_t& b_sep, scalar_t* wmem);
    void look_right(DistM_t& y_sep, scalar_t* wmem);

    virtual integer_t maximum_rank(int task_depth=0) const { return 0; }
    virtual long long factor_nonzeros(int task_depth=0) const;
    virtual long long dense_factor_nonzeros(int task_depth=0) const;
    virtual bool isHSS() const { return false; }
    virtual bool isMPI() const { return false; }
    virtual void print_rank_statistics(std::ostream &out) const {}
    virtual std::string type() const { return "FrontalMatrix"; }
    virtual int random_samples() const { return 0; }
    virtual void bisection_partitioning
    (const SPOptions<scalar_t>& opts, integer_t* sorder,
     bool isroot=true, int task_depth=0);
    void permute_upd_indices(const integer_t* perm, int task_depth=0);

    virtual void set_HSS_partitioning
    (const SPOptions<scalar_t>& opts, const HSS::HSSPartitionTree& sep_tree,
     bool is_root) {}


    // TODO compute this (and levels) once, store it
    // maybe compute it when setting pointers to the children
    // create setters/getters for the children
    integer_t max_dim_upd() const {
      integer_t max_dupd = dim_upd();
      if (lchild) max_dupd = std::max(max_dupd, lchild->max_dim_upd());
      if (rchild) max_dupd = std::max(max_dupd, rchild->max_dim_upd());
      return max_dupd;
    }
    int levels() const {
      int ll = 0, lr = 0;
      if (lchild) ll = lchild->levels();
      if (rchild) lr = rchild->levels();
      return std::max(ll, lr) + 1;
    }

    integer_t sep;
    integer_t sep_begin;
    integer_t sep_end;
    std::vector<integer_t> upd;

    F_t* lchild;
    F_t* rchild;

  protected:
    FrontalMatrix(const FrontalMatrix&) = delete;
    FrontalMatrix& operator=(FrontalMatrix const&) = delete;

    virtual long long node_factor_nonzeros() const {
      auto dsep = dim_sep();
      auto dupd = dim_upd();
      return dsep * (dsep + 2 * dupd);
    }
    virtual long long dense_node_factor_nonzeros() const {
      return node_factor_nonzeros();
    }
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrix<scalar_t,integer_t>::FrontalMatrix
  (F_t* _lchild, F_t* _rchild, integer_t _sep, integer_t _sep_begin,
   integer_t _sep_end, std::vector<integer_t>& _upd)
    : sep(_sep), sep_begin(_sep_begin), sep_end(_sep_end),
      upd(std::move(_upd)), lchild(_lchild), rchild(_rchild) {
  }

  /**
   * I[0:n-1] contains a list of unsorted global indices. This routine
   * finds the corresponding indices into the upd part of this front
   * and puts those indices in lI. Not all indices from I need to be
   * in lI, so oI stores for each element in lI, to which element in I
   * that element corresponds.
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::find_upd_indices
  (const std::vector<std::size_t>& I, std::vector<std::size_t>& lI,
   std::vector<std::size_t>& oI) const {
    auto n = I.size();
    lI.reserve(n);
    oI.reserve(n);
    for (std::size_t i=0; i<n; i++) {
      auto l = std::lower_bound(upd.begin(), upd.end(), I[i]);
      if (l != upd.end() && *l == int(I[i])) {
        lI.push_back(std::distance(upd.begin(), l));
        oI.push_back(i);
      }
    }
  }

  template<typename scalar_t,typename integer_t> std::vector<std::size_t>
  FrontalMatrix<scalar_t,integer_t>::upd_to_parent
  (const F_t* pa) const {
    std::size_t upd2sep;
    return upd_to_parent(pa, upd2sep);
  }

  template<typename scalar_t,typename integer_t> std::vector<std::size_t>
  FrontalMatrix<scalar_t,integer_t>::upd_to_parent
  (const F_t* pa, std::size_t& upd2sep) const {
    integer_t r = 0;
    integer_t dupd = dim_upd();
    integer_t pa_dsep = pa->dim_sep();
    std::vector<std::size_t> I(dupd);
    for (; r<dupd; r++) {
      auto up = upd[r];
      if (up >= pa->sep_end) break;
      I[r] = up - pa->sep_begin;
    }
    upd2sep = r;
    for (integer_t t=0; r<dupd; r++) {
      auto up = upd[r];
      while (pa->upd[t] < up) t++;
      I[r] = t + pa_dsep;
    }
    return I;
  }


  // TODO implement in DenseMatrix?? or use upd_to_parent??
  template<typename scalar_t,typename integer_t> inline void
  FrontalMatrix<scalar_t,integer_t>::extend_add_b
  (const F_t* ch, DenseM_t& b, DenseM_t& bupd, const DenseM_t& CB) const {
    integer_t r = 0, upd_pos = 0,
      CBrows = CB.rows(), cols = b.cols();
    for (; r<CBrows; r++) { // to parent separator
      upd_pos = ch->upd[r];
      if (upd_pos >= sep_end) break;
      for (integer_t c=0; c<cols; c++)
        b(upd_pos,c) += CB(r,c);
    }
    upd_pos = std::distance
      (upd.begin(), std::lower_bound(upd.begin(), upd.end(), upd_pos));
    for (; r<CBrows; r++) { // to parent update matrix
      integer_t t = ch->upd[r];
      while (upd[upd_pos] < t) upd_pos++;
      for (integer_t c=0; c<cols; c++)
        bupd(upd_pos,c) += CB(r,c);
    }
    STRUMPACK_FLOPS
      ((is_complex<scalar_t>()?2:1)*
       static_cast<long long int>(CBrows));
    STRUMPACK_BYTES
      (sizeof(scalar_t)*static_cast<long long int>
       (3*CBrows)+sizeof(integer_t)*(CBrows+dim_upd()));
  }


  /**
   * Assemble CB=b(I^{upd}) from [b(I^{sep});b(I^{upd})] of the parent.
   */
  template<typename scalar_t,typename integer_t> inline void
  FrontalMatrix<scalar_t,integer_t>::extract_b
  (const F_t* ch, const DenseM_t& y, const DenseM_t& yupd,
   DenseM_t& CB) const {
    integer_t r = 0, upd_pos = 0,
      cols = y.cols(), CBrows = CB.rows();
    for (; r<CBrows; r++) { // to parent separator
      upd_pos = ch->upd[r];
      if (upd_pos >= sep_end) break;
      for (integer_t c=0; c<cols; c++)
        CB(r,c) = y(upd_pos,c);
    }
    upd_pos = std::distance
      (upd.begin(), std::lower_bound(upd.begin(), upd.end(), upd_pos));
    for (; r<CBrows; r++) { // to parent update matrix
      integer_t t = ch->upd[r];
      while (upd[upd_pos] < t) upd_pos++;
      for (integer_t c=0; c<cols; c++)
        CB(r,c) = yupd(upd_pos,c);
    }
    STRUMPACK_FLOPS
      ((is_complex<scalar_t>()?2:1)*
       static_cast<long long int>(CBrows));
    STRUMPACK_BYTES
      (sizeof(scalar_t)*static_cast<long long int>
       (3*CBrows)+sizeof(integer_t)*(CBrows+dim_upd()));
  }


  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::look_left
  (DistM_t& b_sep, scalar_t* wmem) {
    TIMER_TIME(TaskType::LOOK_LEFT, 0, t_look);
    if (lchild) extend_add_b(lchild, b_sep, wmem, 0);
    if (rchild) extend_add_b(rchild, b_sep, wmem, 1);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::look_right
  (DistM_t& y_sep, scalar_t* wmem) {
    TIMER_TIME(TaskType::LOOK_RIGHT, 0, t_look);
    if (lchild) extract_b(lchild, y_sep, wmem);
    if (rchild) extract_b(rchild, y_sep, wmem);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::multifrontal_solve(DenseM_t& b) const {
    auto max_dupd = max_dim_upd();
    auto lvls = levels();
    std::vector<DenseM_t> CB(lvls);
    for (auto& cb : CB)
      cb = DenseM_t(max_dupd, b.cols());
    TIMER_TIME(TaskType::FORWARD_SOLVE, 0, t_fwd);
    forward_multifrontal_solve(b, CB.data());
    TIMER_STOP(t_fwd);
    TIMER_TIME(TaskType::BACKWARD_SOLVE, 0, t_bwd);
    backward_multifrontal_solve(b, CB.data());
    TIMER_STOP(t_bwd);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::multifrontal_solve
  (DenseM_t& bloc, DistM_t* bdist) const {
    DistM_t CB;
    DenseM_t seqCB;
    TIMER_TIME(TaskType::FORWARD_SOLVE, 0, t_fwd);
    forward_multifrontal_solve(bloc, bdist, CB, seqCB);
    TIMER_STOP(t_fwd);
    TIMER_TIME(TaskType::BACKWARD_SOLVE, 0, t_bwd);
    backward_multifrontal_solve(bloc, bdist, CB, seqCB);
    TIMER_STOP(t_bwd);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
   int etree_level) const {
    auto max_dupd = max_dim_upd();
    auto lvls = levels();
    std::vector<DenseM_t> CB(lvls);
    for (auto& cb : CB)
      cb = DenseM_t(max_dupd, bloc.cols());
    forward_multifrontal_solve(bloc, CB.data(), etree_level, 0);
    seqbupd = CB[0];
  }
  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
   int etree_level) const {
    auto max_dupd = max_dim_upd();
    auto lvls = levels();
    std::vector<DenseM_t> CB(lvls);
    for (auto& cb : CB)
      cb = DenseM_t(max_dupd, yloc.cols());
    CB[0] = seqyupd;
    backward_multifrontal_solve(yloc, CB.data(), etree_level, 0);
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrix<scalar_t,integer_t>::factor_nonzeros(int task_depth) const {
    long long nnz = node_factor_nonzeros(), nnzl = 0, nnzr = 0;
    if (lchild)
#pragma omp task default(shared)                        \
  if(task_depth < params::task_recursion_cutoff_level)
      nnzl = lchild->factor_nonzeros(task_depth+1);
    if (rchild)
#pragma omp task default(shared)                        \
  if(task_depth < params::task_recursion_cutoff_level)
      nnzr = rchild->factor_nonzeros(task_depth+1);
#pragma omp taskwait
    return nnz + nnzl + nnzr;
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrix<scalar_t,integer_t>::dense_factor_nonzeros
  (int task_depth) const {
    long long nnz = dense_node_factor_nonzeros(), nnzl = 0, nnzr = 0;
    if (lchild)
#pragma omp task default(shared)                        \
  if(task_depth < params::task_recursion_cutoff_level)
      nnzl = lchild->dense_factor_nonzeros(task_depth+1);
    if (rchild)
#pragma omp task default(shared)                        \
  if(task_depth < params::task_recursion_cutoff_level)
      nnzr = rchild->dense_factor_nonzeros(task_depth+1);
#pragma omp taskwait
    return nnz + nnzl + nnzr;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::bisection_partitioning
  (const SPOptions<scalar_t>& opts, integer_t* sorder,
   bool isroot, int task_depth) {
    auto lch = lchild;
    auto rch = rchild;
    if (lch)
#pragma omp task default(shared) firstprivate(sorder,task_depth,lch)    \
  if(task_depth < params::task_recursion_cutoff_level)
      lch->bisection_partitioning(opts, sorder, false, task_depth+1);
    if (rch)
#pragma omp task default(shared) firstprivate(sorder,task_depth,rch)    \
  if(task_depth < params::task_recursion_cutoff_level)
      rch->bisection_partitioning(opts, sorder, false, task_depth+1);

    // default is to do nothing, see FrontalMatrixHSS for an actual
    // implementation
    for (integer_t i=sep_begin; i<sep_end; i++)
      sorder[i] = -i;
#pragma omp taskwait
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrix<scalar_t,integer_t>::permute_upd_indices
  (const integer_t* perm, int task_depth) {
    auto lch = lchild;
    auto rch = rchild;
    if (lch)
#pragma omp task default(shared) firstprivate(perm,task_depth,lch)      \
  if(task_depth < params::task_recursion_cutoff_level)
      lch->permute_upd_indices(perm, task_depth+1);
    if (rch)
#pragma omp task default(shared) firstprivate(perm,task_depth,rch)      \
  if(task_depth < params::task_recursion_cutoff_level)
      rch->permute_upd_indices(perm, task_depth+1);

    for (integer_t i=0; i<dim_upd(); i++)
      upd[i] = perm[upd[i]];
    std::sort(upd.begin(), upd.end());
#pragma omp taskwait
  }

} // end namespace strumpack

#endif //FRONTAL_MATRIX_HPP
