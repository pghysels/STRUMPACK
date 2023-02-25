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
#include <iostream>
#include <algorithm>

#include "EliminationTree.hpp"
#include "fronts/FrontFactory.hpp"
#include "fronts/FrontalMatrix.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  EliminationTree<scalar_t,integer_t>::EliminationTree
  (const SPOptions<scalar_t>& opts, const SpMat_t& A,
   SeparatorTree<integer_t>& sep_tree) {
    std::vector<std::vector<integer_t>> upd(sep_tree.separators());
#pragma omp parallel default(shared)
#pragma omp single
    symbolic_factorization(A, sep_tree, sep_tree.root(), upd);
    root_ = setup_tree(opts, A, sep_tree, upd, sep_tree.root(), true, 0);
  }

  template<typename scalar_t,typename integer_t>
  EliminationTree<scalar_t,integer_t>::~EliminationTree() {}

  template<typename scalar_t,typename integer_t>
  FrontalMatrix<scalar_t,integer_t>*
  EliminationTree<scalar_t,integer_t>::root() const {
    return root_.get();
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTree<scalar_t,integer_t>::print_rank_statistics
  (std::ostream &out) const {
    root_->print_rank_statistics(out);
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTree<scalar_t,integer_t>::symbolic_factorization
  (const SpMat_t& A, const SeparatorTree<integer_t>& sep_tree,
   integer_t sep, std::vector<std::vector<integer_t>>& upd, int depth) const {
    auto chl = sep_tree.lch[sep];
    auto chr = sep_tree.rch[sep];
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
    auto sep_begin = sep_tree.sizes[sep];
    auto sep_end = sep_tree.sizes[sep+1];
    if (sep != sep_tree.root()) { // not necessary for the root
      for (integer_t c=sep_begin; c<sep_end; c++) {
        auto ice = A.ind()+A.ptr(c+1);
        auto icb = std::lower_bound
          (A.ind()+A.ptr(c), ice, sep_end);
        auto mid = upd[sep].size();
        std::copy(icb, ice, std::back_inserter(upd[sep]));
        std::inplace_merge
          (upd[sep].begin(), upd[sep].begin() + mid, upd[sep].end());
        upd[sep].erase
          (std::unique(upd[sep].begin(), upd[sep].end()), upd[sep].end());
      }
      auto dim_sep = sep_end - sep_begin;
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
  std::unique_ptr<FrontalMatrix<scalar_t,integer_t>>
  EliminationTree<scalar_t,integer_t>::setup_tree
  (const SPOptions<scalar_t>& opts, const SpMat_t& A,
   SeparatorTree<integer_t>& sep_tree,
   std::vector<std::vector<integer_t>>& upd, integer_t sep,
   bool hss_parent, int level) {
    auto sep_begin = sep_tree.sizes[sep];
    auto sep_end = sep_tree.sizes[sep+1];
    auto dim_sep = sep_end - sep_begin;
    // dummy nodes added at the end of the separator tree have
    // dim_sep==0, but they have sep_begin=sep_end=N, which is wrong
    // So fix this here!
    if (dim_sep == 0 && sep_tree.lch[sep] != -1)
      sep_begin = sep_end = sep_tree.sizes[sep_tree.rch[sep]+1];
    auto front = create_frontal_matrix<scalar_t,integer_t>
      (opts, sep, sep_begin, sep_end, upd[sep],
       hss_parent, level, nr_fronts_);
    bool compressed =
      is_compressed(dim_sep, upd[sep].size(), hss_parent, opts);
    if (sep_tree.lch[sep] != -1)
      front->set_lchild
        (setup_tree(opts, A, sep_tree, upd, sep_tree.lch[sep],
                    compressed, level+1));
    if (sep_tree.rch[sep] != -1)
      front->set_rchild
        (setup_tree(opts, A, sep_tree, upd, sep_tree.rch[sep],
                    compressed, level+1));
    return front;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  EliminationTree<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts) {
    return root_->multifrontal_factorization(A, opts);
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTree<scalar_t,integer_t>::delete_factors() {
    root_->delete_factors();
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTree<scalar_t,integer_t>::multifrontal_solve
  (DenseM_t& x) const {
    root_->multifrontal_solve(x);
  }

  template<typename scalar_t,typename integer_t> integer_t
  EliminationTree<scalar_t,integer_t>::maximum_rank() const {
    integer_t max_rank;
#pragma omp parallel
#pragma omp single nowait
    max_rank = root_->maximum_rank();
    return max_rank;
  }

  template<typename scalar_t,typename integer_t> long long
  EliminationTree<scalar_t,integer_t>::factor_nonzeros() const {
    long long nonzeros;
#pragma omp parallel
#pragma omp single nowait
    nonzeros = root_->factor_nonzeros();
    return nonzeros;
  }

  template<typename scalar_t,typename integer_t> long long
  EliminationTree<scalar_t,integer_t>::dense_factor_nonzeros() const {
    long long nonzeros;
#pragma omp parallel
#pragma omp single nowait
    nonzeros = root_->dense_factor_nonzeros();
    return nonzeros;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  EliminationTree<scalar_t,integer_t>::inertia
  (integer_t& neg, integer_t& zero, integer_t& pos) const {
    return root_->inertia(neg, zero, pos);
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTree<scalar_t,integer_t>::draw
  (const SpMat_t& A, const std::string& name) const {
    std::ofstream of("plot" + name + ".gnuplot");
    of << "set terminal pdf enhanced color size 5,4" << std::endl;
    of << "set output '" << name << ".pdf'" << std::endl;
    of << "set style rectangle fillstyle noborder" << std::endl;
    root_->draw(of);
    of << "set xrange [0:" << A.size() << "]" << std::endl;
    of << "set yrange [" << A.size() << ":0]" << std::endl;
    of << "plot x lt -1 notitle" << std::endl;
    of.close();
  }

  // explicit template specializations
  template class EliminationTree<float,int>;
  template class EliminationTree<double,int>;
  template class EliminationTree<std::complex<float>,int>;
  template class EliminationTree<std::complex<double>,int>;

  template class EliminationTree<float,long int>;
  template class EliminationTree<double,long int>;
  template class EliminationTree<std::complex<float>,long int>;
  template class EliminationTree<std::complex<double>,long int>;

  template class EliminationTree<float,long long int>;
  template class EliminationTree<double,long long int>;
  template class EliminationTree<std::complex<float>,long long int>;
  template class EliminationTree<std::complex<double>,long long int>;

} // end namespace strumpack
