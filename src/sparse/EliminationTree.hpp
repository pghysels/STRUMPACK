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

#include <vector>
#include <memory>

#include "dense/DenseMatrix.hpp"
#include "CompressedSparseMatrix.hpp"
#include "fronts/FrontFactory.hpp"
#include "fronts/FrontalMatrix.hpp"
#include "StrumpackOptions.hpp"
#include "SeparatorTree.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrix;

  // TODO rename this to SuperNodalTree?
  template<typename scalar_t,typename integer_t>
  class EliminationTree {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;

  public:
    EliminationTree() {}

    EliminationTree(const SPOptions<scalar_t>& opts,
                    const SpMat_t& A,
                    SeparatorTree<integer_t>& sep_tree);
    virtual ~EliminationTree();

    virtual ReturnCode
    multifrontal_factorization(const SpMat_t& A,
                               const SPOptions<scalar_t>& opts);

    virtual void delete_factors();

    virtual void multifrontal_solve(DenseM_t& x) const;

    virtual void
    multifrontal_solve_dist(DenseM_t& x,
                            const std::vector<integer_t>& dist) {} // TODO const

    virtual integer_t maximum_rank() const;
    virtual long long factor_nonzeros() const;
    virtual long long dense_factor_nonzeros() const;

    virtual ReturnCode inertia(integer_t& neg,
                               integer_t& zero,
                               integer_t& pos) const;

    void print_rank_statistics(std::ostream &out) const;

    virtual FrontCounter front_counter() const { return nr_fronts_; }

    void draw(const SpMat_t& A, const std::string& name) const;

    F_t* root() const;

  protected:
    FrontCounter nr_fronts_;
    std::unique_ptr<F_t> root_;

  private:
    std::unique_ptr<F_t>
    setup_tree(const SPOptions<scalar_t>& opts, const SpMat_t& A,
               SeparatorTree<integer_t>& sep_tree,
               std::vector<std::vector<integer_t>>& upd, integer_t sep,
               bool hss_parent, int level);

    void
    symbolic_factorization(const SpMat_t& A,
                           const SeparatorTree<integer_t>& sep_tree,
                           integer_t sep,
                           std::vector<std::vector<integer_t>>& upd,
                           int depth=0) const;
  };

} // end namespace strumpack

#endif // ELIMINATION_TREE_HPP
