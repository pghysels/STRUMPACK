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
#ifndef ELIMINATION_TREE_HPP
#define ELIMINATION_TREE_HPP

#include <iostream>
#include <algorithm>
#include "StrumpackParameters.hpp"
#include "CompressedSparseMatrix.hpp"
#include "FrontalMatrixHSS.hpp"
#include "FrontalMatrixDense.hpp"

namespace strumpack {

  // TODO rename this to SuperNodalTree?
  template<typename scalar_t,typename integer_t>
  class EliminationTree {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;

  public:
    EliminationTree();
    EliminationTree
    (const SPOptions<scalar_t>& opts, const SpMat_t& A,
     const SeparatorTree<integer_t>& sep_tree);
    virtual ~EliminationTree() {}
    virtual void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts);
    virtual void multifrontal_solve(DenseM_t& x) const;
    virtual void multifrontal_solve_dist
    (DenseM_t& x, const std::vector<integer_t>& dist) {} // TODO const
    virtual integer_t maximum_rank() const;
    virtual long long factor_nonzeros() const;
    virtual long long dense_factor_nonzeros() const;
    void print_rank_statistics(std::ostream &out) const {
      _root->print_rank_statistics(out);
    }
    virtual int nr_HSS_fronts() const { return _nr_HSS_fronts; }
    virtual int nr_dense_fronts() const { return _nr_dense_fronts; }

  protected:
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FD_t = FrontalMatrixDense<scalar_t,integer_t>;
    using FHSS_t = FrontalMatrixHSS<scalar_t,integer_t>;

    int _nr_HSS_fronts = 0;
    int _nr_dense_fronts = 0;
    std::unique_ptr<F_t> _root;

  private:
    F_t* setup_tree
    (const SPOptions<scalar_t>& opts, const SpMat_t& A,
     const SeparatorTree<integer_t>& sep_tree,
     std::vector<integer_t>* upd, integer_t sep,
     bool hss_parent, int level);

    void symbolic_factorization
    (const SpMat_t& A, const SeparatorTree<integer_t>& sep_tree,
     integer_t sep, std::vector<integer_t>* upd, int depth=0) const;
  };

  template<typename scalar_t,typename integer_t>
  EliminationTree<scalar_t,integer_t>::EliminationTree() { }

  template<typename scalar_t,typename integer_t>
  EliminationTree<scalar_t,integer_t>::EliminationTree
  (const SPOptions<scalar_t>& opts, const SpMat_t& A,
   const SeparatorTree<integer_t>& sep_tree) {
    auto upd = new std::vector<integer_t>[sep_tree.separators()];

#pragma omp parallel default(shared)
#pragma omp single
    symbolic_factorization(A, sep_tree, sep_tree.root(), upd);

    _root = std::unique_ptr<F_t>
      (setup_tree(opts, A, sep_tree, upd, sep_tree.root(), true, 0));
    delete[] upd;
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTree<scalar_t,integer_t>::symbolic_factorization
  (const SpMat_t& A, const SeparatorTree<integer_t>& sep_tree,
   integer_t sep, std::vector<integer_t>* upd, int depth) const {
    auto chl = sep_tree.lch()[sep];
    auto chr = sep_tree.rch()[sep];
    if (depth < params::task_recursion_cutoff_level) {
      if (chl != -1)
#pragma omp task untied default(shared)                                 \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        symbolic_factorization(A, sep_tree, chl, upd, depth+1);
      if (chr != -1)
#pragma omp task untied default(shared)                                 \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        symbolic_factorization(A, sep_tree, chr, upd, depth+1);
#pragma omp taskwait
    } else {
      if (chl != -1) symbolic_factorization(A, sep_tree, chl, upd, depth);
      if (chr != -1) symbolic_factorization(A, sep_tree, chr, upd, depth);
    }
    auto sep_begin = sep_tree.sizes()[sep];
    auto sep_end = sep_tree.sizes()[sep+1];
    if (sep != sep_tree.root()) { // not necessary for the root
      for (integer_t c=sep_begin; c<sep_end; c++) {
        auto ice = A.get_ind()+A.get_ptr()[c+1];
        auto icb = std::lower_bound
          (A.get_ind()+A.get_ptr()[c], ice, sep_end);
        auto mid = upd[sep].size();
        std::copy(icb, ice, std::back_inserter(upd[sep]));
        std::inplace_merge
          (upd[sep].begin(), upd[sep].begin() + mid, upd[sep].end());
        upd[sep].erase
          (std::unique(upd[sep].begin(), upd[sep].end()), upd[sep].end());
      }
      auto dim_sep = sep_end-sep_begin;
      if (chl != -1) {
        auto icb = dim_sep ?
          std::lower_bound(upd[chl].begin(), upd[chl].end(), sep_end) :
          upd[chl].begin();
        auto mid = upd[sep].size();
        std::copy(icb, upd[chl].end(), std::back_inserter(upd[sep]));
        std::inplace_merge
          (upd[sep].begin(), upd[sep].begin() + mid, upd[sep].end());
        upd[sep].erase
          (std::unique(upd[sep].begin(), upd[sep].end()), upd[sep].end());
      }
      if (chr != -1) {
        auto icb = dim_sep ?
          std::lower_bound(upd[chr].begin(), upd[chr].end(), sep_end) :
          upd[chr].begin();
        auto mid = upd[sep].size();
        std::copy(icb, upd[chr].end(), std::back_inserter(upd[sep]));
        std::inplace_merge
          (upd[sep].begin(), upd[sep].begin() + mid, upd[sep].end());
        upd[sep].erase
          (std::unique(upd[sep].begin(), upd[sep].end()), upd[sep].end());
      }
    }
  }

  template<typename scalar_t,typename integer_t>
  FrontalMatrix<scalar_t,integer_t>*
  EliminationTree<scalar_t,integer_t>::setup_tree
  (const SPOptions<scalar_t>& opts, const SpMat_t& A,
   const SeparatorTree<integer_t>& sep_tree,
   std::vector<integer_t>* upd, integer_t sep,
   bool hss_parent, int level) {
    auto sep_begin = sep_tree.sizes()[sep];
    auto sep_end = sep_tree.sizes()[sep+1];
    auto dim_sep = sep_end - sep_begin;
    // dummy nodes added at the end of the separator tree have
    // dim_sep==0, but they have sep_begin=sep_end=N, which is wrong
    // So fix this here!
    if (dim_sep==0 && sep_tree.lch()[sep] != -1)
      sep_begin = sep_end = sep_tree.sizes()[sep_tree.rch()[sep]+1];
    F_t* front = nullptr;
    // bool is_hss = opts.use_HSS() && (dim_sep >= opts.HSS_min_sep_size()) &&
    //   (dim_sep + upd[sep].size() >= opts.HSS_min_front_size());
    bool is_hss = opts.use_HSS() && hss_parent &&
      (dim_sep >= opts.HSS_min_sep_size());
    if (is_hss) {
      front = new FHSS_t
        (sep, sep_begin, sep_end, upd[sep]);
      front->set_HSS_partitioning
        (opts, sep_tree.HSS_trees().at(sep), level == 0);
      this->_nr_HSS_fronts++;
    } else {
      front = new FD_t
        (sep, sep_begin, sep_end, upd[sep]);
      this->_nr_dense_fronts++;
    }
    if (sep_tree.lch()[sep] != -1)
      front->lchild = setup_tree
        (opts, A, sep_tree, upd, sep_tree.lch()[sep], is_hss, level+1);
    if (sep_tree.rch()[sep] != -1)
      front->rchild = setup_tree
        (opts, A, sep_tree, upd, sep_tree.rch()[sep], is_hss, level+1);
    return front;
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTree<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts) {
    _root->multifrontal_factorization(A, opts);
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTree<scalar_t,integer_t>::multifrontal_solve
  (DenseM_t& x) const {
    _root->multifrontal_solve(x);
  }

  template<typename scalar_t,typename integer_t> integer_t
  EliminationTree<scalar_t,integer_t>::maximum_rank() const {
    integer_t max_rank;
#pragma omp parallel
#pragma omp single nowait
    max_rank = _root->maximum_rank();
    return max_rank;
  }

  template<typename scalar_t,typename integer_t> long long
  EliminationTree<scalar_t,integer_t>::factor_nonzeros() const {
    long long nonzeros;
#pragma omp parallel
#pragma omp single nowait
    nonzeros = _root->factor_nonzeros();
    return nonzeros;
  }

  template<typename scalar_t,typename integer_t> long long
  EliminationTree<scalar_t,integer_t>::dense_factor_nonzeros() const {
    long long nonzeros;
#pragma omp parallel
#pragma omp single nowait
    nonzeros = _root->dense_factor_nonzeros();
    return nonzeros;
  }

} // end namespace strumpack

#endif
