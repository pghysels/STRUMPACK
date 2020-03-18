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
#ifndef SEPARATOR_TREE_HPP
#define SEPARATOR_TREE_HPP

#include <vector>
#include <memory>
#if defined(STRUMPACK_USE_MPI)
#include "misc/MPIWrapper.hpp"
#endif

namespace strumpack {

  /**
   * Helper class to construct a SeparatorTree.
   */
  template<typename integer_t> class Separator {
  public:
    Separator(integer_t separator_end, integer_t parent,
              integer_t left_child, integer_t right_child)
      : sep_end(separator_end), pa(parent),
        lch(left_child), rch(right_child) {}
    integer_t sep_end, pa, lch, rch;
  };


  /**
   * Simple class to store a separator tree. A node in this tree
   * should always have 0 or 2 children.
   */
  template<typename integer_t> class SeparatorTree {
  public:

    /**
     * Construct based on number of separators.
     */
    SeparatorTree(integer_t nr_nodes);

    /**
     * Construct from on a vector of Separators.
     */
    SeparatorTree(std::vector<Separator<integer_t>>& seps);

    /**
     * Construct from an elimination tree.
     */
    SeparatorTree(std::vector<integer_t>& etree);

    integer_t levels() const;
    integer_t level(integer_t i) const;
    integer_t root() const;
    void print() const;
    void printm(const std::string& name) const;
    void check() const;

    std::unique_ptr<SeparatorTree<integer_t>> subtree(integer_t p, integer_t P) const;
    std::unique_ptr<SeparatorTree<integer_t>> toptree(integer_t P) const;

    integer_t separators() const { return nr_seps_; }

    const integer_t* pa() const { return parent_; }
    const integer_t* lch() const { return lchild_; }
    const integer_t* rch() const { return rchild_; }

    integer_t* pa() { return parent_; }
    integer_t* lch() { return lchild_; }
    integer_t* rch() { return rchild_; }

    integer_t sizes(integer_t sep) const { return sep_sizes_[sep]; }
    integer_t pa(integer_t sep) const { return parent_[sep]; }
    integer_t lch(integer_t sep) const { return lchild_[sep]; }
    integer_t rch(integer_t sep) const { return rchild_[sep]; }

    integer_t& sizes(integer_t sep) { return sep_sizes_[sep]; }
    integer_t& pa(integer_t sep) { return parent_[sep]; }
    integer_t& lch(integer_t sep) { return lchild_[sep]; }
    integer_t& rch(integer_t sep) { return rchild_[sep]; }

    bool is_leaf(integer_t sep) const {
      return lchild_[sep] == -1;
    }
    bool is_root(integer_t sep) const {
      return parent_[sep] == -1;
    }
    bool is_empty() const {
      return nr_seps_ == 0;
    }

#if defined(STRUMPACK_USE_MPI)
    void broadcast(const MPIComm& c);
#endif

  protected:
    integer_t nr_seps_ = 0;
    std::unique_ptr<integer_t[]> iwork_ = nullptr;
    integer_t *sep_sizes_ = nullptr, *parent_ = nullptr,
      *lchild_ = nullptr, *rchild_ = nullptr;

    integer_t size() const { return 4*nr_seps_+1; }

    void allocate_nr_seps(integer_t nseps) {
      nr_seps_ = nseps;
      iwork_.reset(new integer_t[4*nseps+1]);
      sep_sizes_ = iwork_.get();
      parent_ = sep_sizes_ + nr_seps_ + 1;
      lchild_ = parent_ + nr_seps_;
      rchild_ = lchild_ + nr_seps_;
    }

  private:
    mutable integer_t root_ = -1;
  };


  /**
   * Create a separator tree based on a matrix and a
   * permutation. First build the elimination tree, then postorder the
   * elimination tree. Then combine the postordering of the
   * elimination tree with the permutation. Build a separator tree
   * from the elimination tree. Set the inverse permutation.
   */
  template<typename integer_t>
  std::unique_ptr<SeparatorTree<integer_t>> build_sep_tree_from_perm
  (const integer_t* ptr, const integer_t* ind,
   std::vector<integer_t>& perm, std::vector<integer_t>& iperm);

} // end namespace strumpack

#endif
