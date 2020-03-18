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
#include <stack>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <functional>

#include "StrumpackConfig.hpp"
#include "SeparatorTree.hpp"
#include "ETree.hpp"

namespace strumpack {

  template<typename integer_t>
  SeparatorTree<integer_t>::SeparatorTree(integer_t nr_nodes) {
    allocate_nr_seps(nr_nodes);
  }

  template<typename integer_t> SeparatorTree<integer_t>::SeparatorTree
  (std::vector<Separator<integer_t>>& seps)
    : SeparatorTree<integer_t>(seps.size()) {
    sep_sizes_[0] = 0;
    integer_t i = 0;
    for (const auto& s : seps) {
      sep_sizes_[i+1] = s.sep_end;
      parent_[i] = s.pa;
      lchild_[i] = s.lch;
      rchild_[i] = s.rch;
      i++;
    }
    root_ = -1;
    check();
  }

  // TODO combine small leafs
  template<typename integer_t>
  SeparatorTree<integer_t>::SeparatorTree(std::vector<integer_t>& etree) {
    integer_t n = etree.size();
    if (n == 0) return;
    // in etree a root has parent n, replace n with -1
    std::replace(etree.begin(), etree.end(), n, integer_t(-1));
    integer_t nr_roots =
      std::count(etree.begin(), etree.end(), integer_t(-1));
    // deal with multiple roots
    if (nr_roots > 1) {
      for (integer_t r=0; r<nr_roots-1; r++) {
        auto hi = etree.size() - 1;
        while (etree[hi] != -1) hi--;
        auto root_right = hi;
        hi--;
        while (etree[hi] != -1) hi--;
        integer_t max_p = etree.size();
        etree.push_back(-1);
        etree[root_right] = max_p;
        etree[hi] = max_p;
      }
    }

    // TODO the number of dummies can be computed as sum_node{max(0,
    // node.nr_children-2))

    // TODO this way of adding dummies creates chains!!

    auto new_n = etree.size();
    std::vector<integer_t> count(new_n, 0),
      etree_lchild(new_n, integer_t(-1)),
      etree_rchild(new_n, integer_t(-1));
    for (size_t i=0; i<new_n; i++) {
      integer_t p = etree[i];
      if (p != -1) {
        count[p]++;
        switch (count[p]) {
        case 1:
          // node i is the first child of node p
          etree_lchild[p] = i;
          break;
        case 2:
          // node i is the second child of node p
          etree_rchild[p] = i;
          break;
        case 3:
          // node i is the third child of node p
          // make a new node with children the first two children of p,
          //     set this new node as the left child of p
          // make node i the right child of p
          integer_t max_p = etree.size();
          etree.push_back(max_p);
          etree_lchild.push_back(etree_lchild[p]);
          etree_rchild.push_back(etree_rchild[p]);
          etree_lchild[p] = max_p;
          etree_rchild[p] = i;
          count[p]--;
          break;
        }
      }
    }
    std::vector<Separator<integer_t>> seps;
    std::stack<integer_t,std::vector<integer_t>> s, l;
    s.push(std::distance(etree.begin(),
                         std::find(etree.begin(), etree.end(),
                                   integer_t(-1))));
    integer_t prev=-1;
    while (!s.empty()) {
      integer_t i = s.top();
      if (prev == -1 || etree_lchild[prev] == i || etree_rchild[prev] == i) {
        // moving down
        if (etree_lchild[i] != -1) // go down left
          s.push(etree_lchild[i]);
        else if (etree_rchild[i] != -1)
          s.push(etree_rchild[i]); // if no left, then go down right
      } else if (etree_lchild[i] == prev) {
        // moving up from the left,
        if (etree_rchild[i] != -1) {
          l.push(seps.size() - 1);
          s.push(etree_rchild[i]); // go down right
        }
      } else {
        // skip nodes that have only one child, this will group nodes
        // in fronts
        if ((etree_rchild[i] == -1 && etree_lchild[i] == -1) ||
            (etree_rchild[i] != -1 && etree_lchild[i] != -1)) {
          // two children, or no children
          auto pid = seps.size();
          seps.emplace_back((seps.empty()) ? 0 : seps.back().sep_end, -1,
                            (etree_lchild[i]!=-1) ? l.top() : -1,  // lch
                            (etree_rchild[i]!=-1) ? pid-1 : -1);   // rch
          if (etree_lchild[i] != -1) {
            seps[l.top()].pa = pid;
            l.pop();
          }
          if (etree_rchild[i] != -1) seps[pid-1].pa = pid;
        }
        // nodes >= n are empty separators introduced to avoid nodes
        // with three children, so do not count those when computing
        // separator size!
        if (i < n) seps.back().sep_end++;
        s.pop();
      }
      prev = i;
    }
    allocate_nr_seps(seps.size());
    sep_sizes_[0] = 0;
    integer_t i = 0;
    for (const auto& s : seps) {
      sep_sizes_[i+1] = s.sep_end;
      parent_[i] = s.pa;
      lchild_[i] = s.lch;
      rchild_[i] = s.rch;
      i++;
    }
    root_ = -1;
    check();
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename integer_t> void
  SeparatorTree<integer_t>::broadcast(const MPIComm& c) {
    c.broadcast<integer_t>(sep_sizes_, size());
  }
#endif

  template<typename integer_t> integer_t
  SeparatorTree<integer_t>::levels() const {
    if (nr_seps_) {
      return level(root());
    } else return 0;
  }

  template<typename integer_t> integer_t
  SeparatorTree<integer_t>::level(integer_t i) const {
    assert(0 <= i && i <= nr_seps_);
    integer_t lvl = 0;
    if (lchild_[i] != -1) lvl = level(lchild_[i]);
    if (rchild_[i] != -1) lvl = std::max(lvl, level(rchild_[i]));
    return lvl+1;
  }

  template<typename integer_t> integer_t
  SeparatorTree<integer_t>::root() const {
    if (root_ == -1)
      root_ = std::find(parent_, parent_+nr_seps_, -1) - parent_;
    return root_;
  }

  template<typename integer_t> void
  SeparatorTree<integer_t>::print() const {
    std::cout << "i\tpa\tlch\trch\tsep" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    for (integer_t i=0; i<nr_seps_; i++)
      std::cout << i << "\t" << parent_[i] << "\t"
                << lchild_[i] << "\t" << rchild_[i] << "\t"
                << sep_sizes_[i] << "/" << sep_sizes_[i+1] << std::endl;
    std::cout << std::endl;
  }

  template<typename integer_t> void
  SeparatorTree<integer_t>::printm(const std::string& name) const {
    check();
    float avg = 0;
    for (integer_t i=0; i<nr_seps_; i++) avg += sep_sizes_[i+1]-sep_sizes_[i];
    avg /= nr_seps_;
    integer_t empty = 0;
    for (integer_t i=0; i<nr_seps_; i++)
      if (sep_sizes_[i+1]-sep_sizes_[i] == 0) empty++;
    std::vector<int> subtree(nr_seps_);
    std::vector<float> inbalance(nr_seps_);
    std::function<void(integer_t)> compute_subtree_size =
      [&](integer_t node) {
      subtree[node] = sep_sizes_[node+1]-sep_sizes_[node];
      if (lchild_[node] != -1) {
        compute_subtree_size(lchild_[node]);
        subtree[node] += subtree[lchild_[node]];
      }
      if (rchild_[node] != -1) {
        compute_subtree_size(rchild_[node]);
        subtree[node] += subtree[rchild_[node]];
      }
      inbalance[node] = 1.;
      if (lchild_[node] != -1 && rchild_[node] != -1)
        inbalance[node] =
          float(std::max(subtree[rchild_[node]], subtree[lchild_[node]])) /
          float(std::min(subtree[rchild_[node]], subtree[lchild_[node]]));
    };
    compute_subtree_size(root());
    float avg_inbalance = 0, max_inbalance = 0;
    for (integer_t i=0; i<nr_seps_; i++) {
      avg_inbalance += inbalance[i];
      max_inbalance = std::max(max_inbalance, inbalance[i]);
    }
    avg_inbalance /= nr_seps_;
    std::ofstream file(name + ".m");
    file << "% Separator tree " << name << std::endl
         << "%   - nr nodes = " << nr_seps_ << std::endl
         << "%   - levels = " << levels() << std::endl
         << "%   - average node size = " << avg << std::endl
         << "%   - empty nodes = " << empty << std::endl
         << "%   - average inbalance = " << avg_inbalance << std::endl
         << "%   - max inbalance = " << max_inbalance << std::endl
         << std::endl;
    file << name << "parent = [";
    for (integer_t i=0; i<nr_seps_; i++) file << parent_[i]+1 << " ";
    file << "];" << std::endl;
    file << name << "sep_sizes = [";
    for (integer_t i=0; i<nr_seps_; i++) file << sep_sizes_[i+1]-sep_sizes_[i] << " ";
    file << "];" << std::endl;
    file.close();
  }

  template<typename integer_t> void
  SeparatorTree<integer_t>::check() const {
#if !defined(NDEBUG)
    if (nr_seps_ == 0) return;
    assert(std::count(parent_, parent_+nr_seps_, -1) == 1); // 1 root
    auto mark = new bool[nr_seps_];
    std::fill(mark, mark+nr_seps_, false);
    std::function<void(integer_t)> traverse = [&](integer_t node) {
      mark[node] = true;
      if (lchild_[node] != -1) traverse(lchild_[node]);
      if (rchild_[node] != -1) traverse(rchild_[node]);
    };
    traverse(root());
    assert(std::count(mark, mark+nr_seps_, false) == 0);
    delete[] mark;
    integer_t nr_leafs = 0;
    for (integer_t i=0; i<nr_seps_; i++) {
      assert(parent_[i]==-1 || parent_[i] >= 0);
      assert(parent_[i] < nr_seps_);
      assert(lchild_[i] < nr_seps_);
      assert(rchild_[i] < nr_seps_);
      assert(lchild_[i]>=0 || lchild_[i]==-1);
      assert(rchild_[i]>=0 || rchild_[i]==-1);
      if (lchild_[i]==-1) { assert(rchild_[i]==-1); }
      if (rchild_[i]==-1) { assert(lchild_[i]==-1); }
      if (parent_[i]!=-1) { assert(lchild_[parent_[i]]==i || rchild_[parent_[i]]==i); }
      if (lchild_[i]==-1 && rchild_[i]==-1) nr_leafs++;
    }
    assert(2*nr_leafs - 1 == nr_seps_);
    for (integer_t i=0; i<nr_seps_; i++) {
      if (sep_sizes_[i+1] < sep_sizes_[i]) {
        std::cout << "sep_sizes_[" << i+1 << "]=" << sep_sizes_[i+1]
                  << " >= sep_sizes_[" << i << "]=" << sep_sizes_[i] << std::endl;
        assert(false);
      };
    }
#endif
  }

  /**
   * Extract subtree p of P.
   */
  template<typename integer_t> std::unique_ptr<SeparatorTree<integer_t>>
  SeparatorTree<integer_t>::subtree(integer_t p, integer_t P) const {
    if (!nr_seps_)
      return std::unique_ptr<SeparatorTree<integer_t>>
        (new SeparatorTree<integer_t>(0));
    std::vector<bool> mark(nr_seps_);
    mark[root()] = true;
    integer_t nr_subtrees = 1;
    std::function<void(integer_t)> find_roots = [&](integer_t i) {
      if (mark[i]) {
        if (nr_subtrees < P && lchild_[i]!=-1 && rchild_[i]!=-1) {
          mark[lchild_[i]] = true;
          mark[rchild_[i]] = true;
          mark[i] = false;
          nr_subtrees++;
        }
      } else {
        if (lchild_[i]!=-1) find_roots(lchild_[i]);
        if (rchild_[i]!=-1) find_roots(rchild_[i]);
      }
    };
    // TODO this can get in an infinite loop
    while (nr_subtrees < P && nr_subtrees < nr_seps_)
      find_roots(root());

    integer_t sub_root = -1;
    std::function<void(integer_t,integer_t&)> find_my_root =
                         [&](integer_t i, integer_t& r) {
      if (mark[i]) {
        if (r++ == p) sub_root = i;
        return;
      }
      if (lchild_[i]!=-1 && rchild_[i]!=-1) {
        find_my_root(lchild_[i], r);
        find_my_root(rchild_[i], r);
      }
    };
    integer_t temp = 0;
    find_my_root(root(), temp);

    if (sub_root == -1)
      return std::unique_ptr<SeparatorTree<integer_t>>
        (new SeparatorTree<integer_t>(0));
    std::function<integer_t(integer_t)> count = [&](integer_t node) {
      integer_t c = 1;
      if (lchild_[node] != -1) c += count(lchild_[node]);
      if (rchild_[node] != -1) c += count(rchild_[node]);
      return c;
    };
    auto sub_size = count(sub_root);
    auto sub = std::unique_ptr<SeparatorTree<integer_t>>
      (new SeparatorTree<integer_t>(sub_size));
    if (sub_size == 0) return sub;
    std::function<void(integer_t,integer_t&)> fill_sub =
      [&](integer_t node, integer_t& id) {
      integer_t left_root = 0;
      if (lchild_[node] != -1) {
        fill_sub(lchild_[node], id);
        left_root = id-1;
      } else sub->lchild_[id] = -1;
      if (rchild_[node] != -1) {
        fill_sub(rchild_[node], id);
        sub->rchild_[id] = id-1;
        sub->parent_[id-1] = id;
      } else sub->rchild_[id] = -1;
      if (lchild_[node] != -1) {
        sub->lchild_[id] = left_root;
        sub->parent_[left_root] = id;
      }
      sub->sep_sizes_[id+1] = sub->sep_sizes_[id] + sep_sizes_[node+1] - sep_sizes_[node];
      id++;
    };
    integer_t id = 0;
    sub->sep_sizes_[0] = 0;
    fill_sub(sub_root, id);
    sub->parent_[sub_size-1] = -1;
    return sub;
  }

  /** extract the tree with the top 2*P-1 nodes, ie a tree with P leafs */
  template<typename integer_t> std::unique_ptr<SeparatorTree<integer_t>>
  SeparatorTree<integer_t>::toptree(integer_t P) const {
    integer_t top_nodes = std::min(std::max(integer_t(0), 2*P-1), nr_seps_);
    auto top = std::unique_ptr<SeparatorTree<integer_t>>
      (new SeparatorTree<integer_t>(top_nodes));
    std::vector<bool> mark(nr_seps_);
    mark[root()] = true;
    integer_t nr_leafs = 1;
    std::function<void(integer_t)> mark_top_tree = [&](integer_t node) {
      if (nr_leafs < P) {
        if (lchild_[node]!=-1 && rchild_[node]!=-1 &&
            !mark[lchild_[node]] && !mark[rchild_[node]]) {
          mark[lchild_[node]] = true;
          mark[rchild_[node]] = true;
          nr_leafs++;
        } else {
          if (lchild_[node]!=-1) mark_top_tree(lchild_[node]);
          if (rchild_[node]!=-1) mark_top_tree(rchild_[node]);
        }
      }
    };
    while (nr_leafs < P && nr_leafs < nr_seps_)
      mark_top_tree(root());

    std::function<integer_t(integer_t)> subtree_size = [&](integer_t i) {
      integer_t s = sep_sizes_[i+1] - sep_sizes_[i];
      if (lchild_[i] != -1) s += subtree_size(lchild_[i]);
      if (rchild_[i] != -1) s += subtree_size(rchild_[i]);
      return s;
    };

    // traverse the tree in reverse postorder!!
    std::function<void(integer_t,integer_t&)> fill_top =
      [&](integer_t node, integer_t& tid) {
      auto mytid = tid;
      tid--;
      if (rchild_[node]!=-1 && mark[rchild_[node]]) {
        top->rchild_[mytid] = tid;
        top->parent_[top->rchild_[mytid]] = mytid;
        fill_top(rchild_[node], tid);
      } else top->rchild_[mytid] = -1;
      if (lchild_[node]!=-1 && mark[lchild_[node]]) {
        top->lchild_[mytid] = tid;
        top->parent_[top->lchild_[mytid]] = mytid;
        fill_top(lchild_[node], tid);
      } else top->lchild_[mytid] = -1;
      if (top->rchild_[mytid] == -1)
        top->sep_sizes_[mytid+1] = subtree_size(node);
      else
        top->sep_sizes_[mytid+1] = sep_sizes_[node+1] - sep_sizes_[node];
    };
    integer_t tid = top_nodes-1;
    top->sep_sizes_[0] = 0;
    fill_top(root(), tid);
    top->parent_[top_nodes-1] = -1;
    for (integer_t i=0; i<top_nodes; i++)
      top->sep_sizes_[i+1] = top->sep_sizes_[i] + top->sep_sizes_[i+1];
    return top;
  }

  template<typename integer_t>
  std::unique_ptr<SeparatorTree<integer_t>> build_sep_tree_from_perm
  (const integer_t* ptr, const integer_t* ind,
   std::vector<integer_t>& perm, std::vector<integer_t>& iperm) {
    integer_t n = perm.size();
    std::vector<integer_t> rlo(n), rhi(n), pind(ptr[n]);
    for (integer_t i=0; i<n; i++) {
      rlo[perm[i]] = ptr[i];
      rhi[perm[i]] = ptr[i+1];
    }
    for (integer_t j=0; j<n; j++)
      for (integer_t i=rlo[j]; i<rhi[j]; i++)
        pind[i] = perm[ind[i]];
    auto etree = spsymetree(rlo.data(), rhi.data(), pind.data(), n);
    auto post = etree_postorder<integer_t>(etree);
    auto& iwork = iperm;
    for (integer_t i=0; i<n; i++) iwork[post[i]] = post[etree[i]];
    for (integer_t i=0; i<n; i++) etree[i] = iwork[i];
    // product of perm and post
    for (integer_t i=0; i<n; i++) iwork[i] = post[perm[i]];
    for (integer_t i=0; i<n; i++) perm[i] = iwork[i];
    for (integer_t i=0; i<n; i++) iperm[perm[i]] = i;
    return std::unique_ptr<SeparatorTree<integer_t>>
      (new SeparatorTree<integer_t>(etree));  // build separator tree
  }

  // explicit template instantiations
  template class SeparatorTree<int>;
  template class SeparatorTree<long int>;
  template class SeparatorTree<long long int>;

  template std::unique_ptr<SeparatorTree<int>>
  build_sep_tree_from_perm(const int* ptr, const int* ind,
                           std::vector<int>& perm,
                           std::vector<int>& iperm);
  template std::unique_ptr<SeparatorTree<long int>>
  build_sep_tree_from_perm(const long int* ptr, const long int* ind,
                           std::vector<long int>& perm,
                           std::vector<long int>& iperm);
  template std::unique_ptr<SeparatorTree<long long int>>
  build_sep_tree_from_perm(const long long int* ptr, const long long int* ind,
                           std::vector<long long int>& perm,
                           std::vector<long long int>& iperm);

} // end namespace strumpack
