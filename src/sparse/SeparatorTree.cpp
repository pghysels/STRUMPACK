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
#include <cassert>

#include "StrumpackConfig.hpp"
#include "SeparatorTree.hpp"

namespace strumpack {

  template<typename integer_t>
  SeparatorTree<integer_t>::SeparatorTree(integer_t nr_nodes) {
    allocate(nr_nodes);
  }

  template<typename integer_t> SeparatorTree<integer_t>::SeparatorTree
  (const std::vector<Separator<integer_t>>& seps)
    : SeparatorTree<integer_t>(seps.size()) {
    sizes[0] = 0;
    integer_t i = 0;
    for (const auto& s : seps) {
      sizes[i+1] = s.sep_end;
      parent[i] = s.pa;
      lch[i] = s.lch;
      rch[i] = s.rch;
      i++;
    }
    root_ = -1;
    check();
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename integer_t> void
  SeparatorTree<integer_t>::broadcast(const MPIComm& c) {
    c.broadcast<integer_t>(sizes, size());
  }
#endif

  template<typename integer_t> integer_t
  SeparatorTree<integer_t>::levels() const {
    if (nr_seps_) return level(root());
    else return 0;
  }

  template<typename integer_t> integer_t
  SeparatorTree<integer_t>::level(integer_t i) const {
    assert(0 <= i && i <= nr_seps_);
    integer_t lvl = 0;
    if (lch[i] != -1) lvl = level(lch[i]);
    if (rch[i] != -1) lvl = std::max(lvl, level(rch[i]));
    return lvl+1;
  }

  template<typename integer_t> integer_t
  SeparatorTree<integer_t>::root() const {
    if (root_ == -1)
      root_ = std::find(parent, parent+nr_seps_, -1) - parent;
    return root_;
  }

  template<typename integer_t> void
  SeparatorTree<integer_t>::print() const {
    std::cout << "i\tpa\tlch\trch\tsep" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    for (integer_t i=0; i<nr_seps_; i++)
      std::cout << i << "\t" << parent[i] << "\t"
                << lch[i] << "\t" << rch[i] << "\t"
                << sizes[i] << "/" << sizes[i+1] << std::endl;
    std::cout << std::endl;
  }

  template<typename integer_t> void
  SeparatorTree<integer_t>::printm(const std::string& name) const {
    check();
    float avg = 0;
    for (integer_t i=0; i<nr_seps_; i++) avg += sizes[i+1]-sizes[i];
    avg /= nr_seps_;
    integer_t empty = 0;
    for (integer_t i=0; i<nr_seps_; i++)
      if (sizes[i+1]-sizes[i] == 0) empty++;
    std::vector<int> subtree(nr_seps_);
    std::vector<float> inbalance(nr_seps_);
    std::function<void(integer_t)> compute_subtree_size =
      [&](integer_t node) {
      subtree[node] = sizes[node+1]-sizes[node];
      if (lch[node] != -1) {
        compute_subtree_size(lch[node]);
        subtree[node] += subtree[lch[node]];
      }
      if (rch[node] != -1) {
        compute_subtree_size(rch[node]);
        subtree[node] += subtree[rch[node]];
      }
      inbalance[node] = 1.;
      if (lch[node] != -1 && rch[node] != -1)
        inbalance[node] =
          float(std::max(subtree[rch[node]], subtree[lch[node]])) /
          float(std::min(subtree[rch[node]], subtree[lch[node]]));
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
    for (integer_t i=0; i<nr_seps_; i++) file << parent[i]+1 << " ";
    file << "];" << std::endl;
    file << name << "sep_sizes = [";
    for (integer_t i=0; i<nr_seps_; i++) file << sizes[i+1]-sizes[i] << " ";
    file << "];" << std::endl;
    file.close();
  }

  template<typename integer_t> void
  SeparatorTree<integer_t>::check() const {
#if !defined(NDEBUG)
    if (nr_seps_ == 0) return;
    assert(std::count(parent, parent+nr_seps_, -1) == 1); // 1 root
    auto mark = new bool[nr_seps_];
    std::fill(mark, mark+nr_seps_, false);
    std::function<void(integer_t)> traverse = [&](integer_t node) {
      mark[node] = true;
      if (lch[node] != -1) traverse(lch[node]);
      if (rch[node] != -1) traverse(rch[node]);
    };
    traverse(root());
    assert(std::count(mark, mark+nr_seps_, false) == 0);
    delete[] mark;
    integer_t nr_leafs = 0;
    for (integer_t i=0; i<nr_seps_; i++) {
      assert(parent[i]==-1 || parent[i] >= 0);
      assert(parent[i] < nr_seps_);
      assert(lch[i] < nr_seps_);
      assert(rch[i] < nr_seps_);
      assert(lch[i]>=0 || lch[i]==-1);
      assert(rch[i]>=0 || rch[i]==-1);
      if (lch[i]==-1) { assert(rch[i]==-1); }
      if (rch[i]==-1) { assert(lch[i]==-1); }
      if (parent[i]!=-1) { assert(lch[parent[i]]==i || rch[parent[i]]==i); }
      if (lch[i]==-1 && rch[i]==-1) nr_leafs++;
    }
    assert(2*nr_leafs - 1 == nr_seps_);
    for (integer_t i=0; i<nr_seps_; i++) {
      if (sizes[i+1] < sizes[i]) {
        std::cout << "sizes[" << i+1 << "]=" << sizes[i+1]
                  << " >= sizes[" << i << "]=" << sizes[i] << std::endl;
        assert(false);
      };
    }
#endif
  }

  /**
   * Extract subtree p of P.
   */
  template<typename integer_t> SeparatorTree<integer_t>
  SeparatorTree<integer_t>::subtree(integer_t p, integer_t P) const {
    if (!nr_seps_) return SeparatorTree<integer_t>(0);
    std::vector<bool> mark(nr_seps_);
    mark[root()] = true;
    integer_t nr_subtrees = 1, marked = 1;
    std::function<void(integer_t)> find_roots = [&](integer_t i) {
      if (mark[i]) {
        if (nr_subtrees < P && lch[i] != -1 && rch[i] != -1) {
          mark[lch[i]] = true;
          mark[rch[i]] = true;
          mark[i] = false;
          marked += 2;
          nr_subtrees++;
        }
      } else {
        if (lch[i] != -1) find_roots(lch[i]);
        if (rch[i] != -1) find_roots(rch[i]);
      }
    };
    while (nr_subtrees < P && marked < nr_seps_)
      find_roots(root());

    integer_t sub_root = -1;
    std::function<void(integer_t,integer_t&)> find_my_root =
      [&](integer_t i, integer_t& r) {
        if (mark[i]) {
          if (r++ == p) sub_root = i;
          return;
        }
        if (lch[i] != -1 && rch[i] != -1) {
          find_my_root(lch[i], r);
          find_my_root(rch[i], r);
        }
      };
    integer_t temp = 0;
    find_my_root(root(), temp);

    if (sub_root == -1)
      return SeparatorTree<integer_t>(0);
    std::function<integer_t(integer_t)> count = [&](integer_t node) {
      integer_t c = 1;
      if (lch[node] != -1) c += count(lch[node]);
      if (rch[node] != -1) c += count(rch[node]);
      return c;
    };
    auto sub_size = count(sub_root);
    SeparatorTree<integer_t> sub(sub_size);
    if (sub_size == 0) return sub;
    std::function<void(integer_t,integer_t&)> fill_sub =
      [&](integer_t node, integer_t& id) {
      integer_t left_root = 0;
      if (lch[node] != -1) {
        fill_sub(lch[node], id);
        left_root = id-1;
      } else sub.lch[id] = -1;
      if (rch[node] != -1) {
        fill_sub(rch[node], id);
        sub.rch[id] = id-1;
        sub.parent[id-1] = id;
      } else sub.rch[id] = -1;
      if (lch[node] != -1) {
        sub.lch[id] = left_root;
        sub.parent[left_root] = id;
      }
      sub.sizes[id+1] = sub.sizes[id] + sizes[node+1] - sizes[node];
      id++;
    };
    integer_t id = 0;
    sub.sizes[0] = 0;
    fill_sub(sub_root, id);
    sub.parent[sub_size-1] = -1;
    return sub;
  }

  /** extract the tree with the top 2*P-1 nodes, ie a tree with P leafs */
  template<typename integer_t> SeparatorTree<integer_t>
  SeparatorTree<integer_t>::toptree(integer_t P) const {
    integer_t top_nodes = std::min(std::max(integer_t(0), 2*P-1), nr_seps_);
    SeparatorTree<integer_t> top(top_nodes);
    std::vector<bool> mark(nr_seps_);
    mark[root()] = true;
    integer_t nr_leafs = 1, marked = 1;
    std::function<void(integer_t)> mark_top_tree = [&](integer_t node) {
      if (nr_leafs < P) {
        if (lch[node] != -1 && rch[node] != -1 &&
            !mark[lch[node]] && !mark[rch[node]]) {
          mark[lch[node]] = true;
          mark[rch[node]] = true;
          nr_leafs++;
          marked += 2;
        } else {
          if (lch[node] != -1) mark_top_tree(lch[node]);
          if (rch[node] != -1) mark_top_tree(rch[node]);
        }
      }
    };
    while (nr_leafs < P && marked < nr_seps_)
      mark_top_tree(root());

    std::function<integer_t(integer_t)> subtree_size = [&](integer_t i) {
      integer_t s = sizes[i+1] - sizes[i];
      if (lch[i] != -1) s += subtree_size(lch[i]);
      if (rch[i] != -1) s += subtree_size(rch[i]);
      return s;
    };

    // traverse the tree in reverse postorder!!
    std::function<void(integer_t,integer_t&)> fill_top =
      [&](integer_t node, integer_t& tid) {
      auto mytid = tid;
      tid--;
      if (rch[node]!=-1 && mark[rch[node]]) {
        top.rch[mytid] = tid;
        top.parent[top.rch[mytid]] = mytid;
        fill_top(rch[node], tid);
      } else top.rch[mytid] = -1;
      if (lch[node]!=-1 && mark[lch[node]]) {
        top.lch[mytid] = tid;
        top.parent[top.lch[mytid]] = mytid;
        fill_top(lch[node], tid);
      } else top.lch[mytid] = -1;
      if (top.rch[mytid] == -1)
        top.sizes[mytid+1] = subtree_size(node);
      else
        top.sizes[mytid+1] = sizes[node+1] - sizes[node];
    };
    integer_t tid = top_nodes-1;
    top.sizes[0] = 0;
    fill_top(root(), tid);
    top.parent[top_nodes-1] = -1;
    for (integer_t i=0; i<top_nodes; i++)
      top.sizes[i+1] = top.sizes[i] + top.sizes[i+1];
    return top;
  }

  template<typename integer_t> std::vector<integer_t>
  etree_postorder(const std::vector<integer_t>& etree) {
    integer_t n = etree.size();
    std::vector<integer_t> kid(n+1, -1), sib(n+1);
    // set up structure describing children
    for (integer_t v=n-1; v>=0; v--) {
      auto dad = etree[v];
      sib[v] = kid[dad];
      kid[dad] = v;
    }
    // depth-first search from dummy root vertex #n
    std::vector<integer_t> post(n+1);
    integer_t current = n, postnum = 0;
    while (postnum < n) {
      auto first = kid[current];
      if (first == -1) {              // no first kid for the current node
        post[current] = postnum++;    // numbering this node because it has no kid
        auto next = sib[current];     // looking for the next kid
        while (next == -1) {
          current = etree[current];   // no more kids : back to the parent node
          post[current] = postnum++;  // numbering the parent node
          next = sib[current];        // get the next kid
        }
        current = next;               // updating current node
      } else current = first;         // updating current node
    }
    return post;
  }

  template<typename integer_t> std::vector<integer_t>
  etree_from_perm(const integer_t* ptr, const integer_t* ind,
                  std::vector<integer_t>& perm) {
    integer_t n = perm.size();
    std::vector<integer_t> rlo(n), rhi(n), pind(ptr[n]);
    for (integer_t i=0; i<n; i++) {
      rlo[perm[i]] = ptr[i];
      rhi[perm[i]] = ptr[i+1];
    }
    for (integer_t j=0; j<n; j++)
      for (integer_t i=rlo[j]; i<rhi[j]; i++)
        pind[i] = perm[ind[i]];
    return spsymetree(rlo.data(), rhi.data(), pind.data(), n);
  }


  template<typename integer_t>
  std::vector<Separator<integer_t>>
  separators_from_etree(std::vector<integer_t>& etree,
                        std::vector<integer_t>& post) {
    integer_t dofs = etree.size(), n = dofs;
    // count number of root nodes (parent in etree == n)
    if (std::count(etree.begin(), etree.end(), dofs) != 1)
      etree.push_back(++n);
    // use -1 to denote root node, instead of n
    std::replace(etree.begin(), etree.end(), n, integer_t(-1));
    integer_t root = n - 1;
    std::vector<integer_t> kid0(n), kids(n), nch(n), w(n, 1);
    // count nodes in subtree (w), and number of children (nch)
    for (integer_t i=0; i<n; i++) {
      auto p = etree[i];
      if (p == -1) continue;
      nch[p]++;
      w[p] += w[i];
    }
    for (integer_t i=1; i<n; i++)
      kid0[i] = kid0[i-1] + nch[i-1];
    std::fill(nch.begin(), nch.end(), 0);
    for (integer_t i=0; i<n; i++) {
      auto p = etree[i];
      if (p == -1) continue;
      kids[kid0[p]+nch[p]++] = i;
    }
    // convert to binary tree, as balanced as possible
    integer_t node = 0;
    while (node < n) {
      auto nc = nch[node];
      if (nc > 2) {
        nch[node] = 2;
        etree.push_back(node);
        etree.push_back(node);
        auto k0 = kid0[node];
        auto kbeg = kids.begin() + k0;
        auto kend = kbeg + nc;
        // Node-count-balanced split.
        std::sort(kbeg, kend, [&w](integer_t a, integer_t b) {
          return w[a] > w[b]; });
        std::vector<integer_t> sk(kbeg, kend);
        integer_t wl = 0, wr = 0, ncl = 0, ncr = 0;
        for (auto ki : sk) {
          if (wl < wr) {
            *kbeg = ki;
            kbeg++;
            wl += w[ki];
            ncl++;
          } else {
            kend--;
            *kend = ki;
            wr += w[ki];
            ncr++;
          }
        }
        kid0.push_back(k0);
        kid0.push_back(k0+ncl);
        nch.push_back(ncl);
        nch.push_back(ncr);
        for (integer_t c=0; c<ncl; c++)
          etree[kids[k0+c]] = n;
        for (integer_t c=ncl; c<nc; c++)
          etree[kids[k0+c]] = n + 1;
        n += 2;
      }
      node++;
    }
    std::vector<integer_t> kid(n, -1), sib(n);
    for (integer_t v=n-1; v>=0; v--) {
      auto dad = etree[v];
      if (dad == -1) continue;
      sib[v] = kid[dad];
      kid[dad] = v;
    }
    // construct postordering
    integer_t current = root, postnum = 0;
    while (postnum < dofs) {
      auto first = kid[current];
      if (first == -1) {
        if (current < dofs)
        post[current] = postnum++;
        auto next = sib[current];
        while (next == -1) {
          current = etree[current];
          if (current < dofs)
            post[current] = postnum++;
          next = sib[current];
        }
        current = next;
      } else current = first;
    }
    std::vector<integer_t> elc(n, -1), erc(n, -1);
    for (integer_t i=0; i<n; i++) {
      auto p = etree[i];
      if (p == -1) continue;
      if (elc[p] == -1) elc[p] = i;
      else erc[p] = i;
    }
    std::vector<Separator<integer_t>> seps;
    std::stack<integer_t,std::vector<integer_t>> s, l;
    s.push(root);
    integer_t prev = -1;

    // find supernodes

    // TODO merge this traversal with the traversal to get the
    // postordering
    while (!s.empty()) {
      auto i = s.top();
      if (prev == -1 || elc[prev] == i || erc[prev] == i) {
        if (elc[i] != -1) s.push(elc[i]);
        else if (erc[i] != -1) s.push(erc[i]);
      } else if (elc[i] == prev) {
        if (erc[i] != -1) {
          l.push(seps.size() - 1);
          s.push(erc[i]);
        }
      } else {
        // skip nodes that have only one child, this will group nodes
        // in fronts/supernodes
        if ((erc[i] == -1 && elc[i] == -1) ||
            (erc[i] != -1 && elc[i] != -1)) {
          integer_t pid = seps.size();
          seps.emplace_back
            ((seps.empty()) ? 0 : seps.back().sep_end, -1,
             (elc[i] != -1) ? l.top() : -1, (erc[i] != -1) ? pid-1 : -1);
          if (elc[i] != -1) {
            seps[l.top()].pa = pid;
            l.pop();
          }
          if (erc[i] != -1) seps[pid-1].pa = pid;
        }
        // nodes >= dofs are empty separators introduced to avoid
        // nodes with three children, so do not count those when
        // computing separator size!
        if (i < dofs)
          seps.back().sep_end++;
        s.pop();
      }
      prev = i;
    }
    return seps;
  }


  /**
   * Relaxed supernode amalgamation on the supernodal separator tree.
   *
   * This pass performs two complementary, fill-safe transformations,
   * driven by a single absolute threshold `k` (in unknowns):
   *
   *   (1) SUBTREE COLLAPSE.  Whenever the TOTAL number of unknowns in the
   *       subtree rooted at a node is below `k`, the whole subtree is
   *       collapsed into that node, which becomes a leaf.  This removes
   *       the mass of tiny fronts near the bottom of the tree.
   *
   *   (2) CHAIN CONTRACTION.  The residual DEPTH is dominated by long,
   *       near-linear chains high in the tree: separators where one child
   *       carries almost the whole subtree and the sibling is small.  For
   *       every node whose SMALLER child subtree is below `k`, that small
   *       child is absorbed into the node and the node is then merged with
   *       its remaining (heavy) child -- the node inherits the heavy
   *       child's two children.  Each such contraction removes one level
   *       along the heavy path, directly shortening the deep chains.  This
   *       is what actually reduces the number of levels that governs GPU
   *       factorization performance.
   *
   * Both transformations only group columns that are already contiguous in
   * the postordering into a single, larger separator, so the fill-reducing
   * permutation stays valid and unchanged.  The symbolic factorization
   * recomputes the update sets from the enlarged separators, so the fill
   * is handled correctly; the only cost is a modest fill increase inside
   * the amalgamated fronts.
   */
  template<typename integer_t>
  std::vector<Separator<integer_t>>
  amalgamate_separators(const std::vector<Separator<integer_t>>& seps,
                        integer_t k, std::vector<integer_t>* colperm=nullptr) {
    integer_t ns = seps.size();
    if (ns == 0 || k <= 0) return seps;

    std::vector<integer_t> lch(ns), rch(ns), sz(ns), off(ns);
    integer_t prev_end = 0, root = -1;
    std::vector<integer_t> pa(ns);
    for (integer_t i=0; i<ns; i++) {
      lch[i] = seps[i].lch;
      rch[i] = seps[i].rch;
      pa[i] = seps[i].pa;
      off[i] = prev_end;                          // first column of sep i
      sz[i] = seps[i].sep_end - prev_end;
      prev_end = seps[i].sep_end;
      if (pa[i] == -1) root = i;
    }
    integer_t ncols = prev_end;

    {
      // Tree depth (root at level 0) via a parent walk with memoization.
      // For each node we walk up to the root or to an already-computed
      // ancestor, collecting the path, then assign increasing levels back
      // down the path.  Each node is finalized once, so this is O(ns).
      std::vector<integer_t> lvl(ns, -1);
      integer_t maxlev = 0;
      std::vector<integer_t> path;
      for (integer_t s=0; s<ns; s++) {
        if (lvl[s] != -1) continue;
        path.clear();
        integer_t c = s;
        while (c != -1 && lvl[c] == -1) { path.push_back(c); c = pa[c]; }
        integer_t base = (c == -1) ? -1 : lvl[c];  // ancestor's level (or -1)
        // path is from s (front) up to just below the memoized ancestor;
        // the deepest path node (last pushed) is closest to the ancestor.
        for (integer_t j = (integer_t)path.size()-1, d = base+1; j >= 0; j--, d++) {
          lvl[path[j]] = d;
          maxlev = std::max(maxlev, d);
        }
      }
      integer_t depth = maxlev + 1;
      integer_t minlevels = 48;
      if (depth < minlevels)
        return seps;
    }

    // Subtree size (unknowns) rooted at each node (postorder: children
    // precede parent, so a forward accumulation sweep suffices).
    std::vector<integer_t> subsize(sz);
    for (integer_t i=0; i<ns; i++)
      if (pa[i] != -1) subsize[pa[i]] += subsize[i];

    // We rebuild the tree with a top-down DFS.  Two fill-safe reshapings,
    // driven by the single threshold k:
    //   (1) SUBTREE COLLAPSE: a subtree with < k unknowns becomes a single
    //       leaf front.
    //   (2) SYMMETRIC CHAIN CONTRACTION: at each node we may contract the
    //       child whose SEPARATOR is small (< k) -- EITHER the left or the
    //       right child -- and continue the chain into it.  To keep every
    //       front's columns contiguous we control the emission order: the
    //       contracted child's subtree is emitted so that its separator
    //       ends up ADJACENT to the growing front's separator.  Because
    //       this may reorder sibling subtrees relative to the original
    //       postorder, we also emit a COLUMN PERMUTATION (colperm: old
    //       postorder position -> new position) that the caller composes
    //       into the final permutation, so the factorization stays exact.
    //
    // We prefer to contract toward the HEAVIER child so the chain follows
    // the heavy path and maximally reduces depth (this is the part that
    // now also "merges left", not just right).
    std::vector<Separator<integer_t>> out;
    out.reserve(ns);
    std::vector<integer_t> cperm; cperm.reserve(ncols);
    integer_t cum = 0;

    // Append all columns [off[i], off[i]+sz[i]) of ORIGINAL separator i to
    // the new column order.  This records, in new-position order, which
    // old column each new column comes from.
    auto append_cols = [&](integer_t i) {
      for (integer_t c=0; c<sz[i]; c++) cperm.push_back(off[i] + c);
      cum += sz[i];
    };
    // Append all original columns of an ENTIRE subtree, in the subtree's
    // own postorder (used when collapsing a small subtree into one leaf).
    std::function<void(integer_t)> append_subtree = [&](integer_t i) {
      if (lch[i] != -1) append_subtree(lch[i]);
      if (rch[i] != -1) append_subtree(rch[i]);
      append_cols(i);
    };

    // Emit the subtree rooted at `node`; returns the output front index
    // that roots it.  `pa_out` is the parent output front (-1 for root).
    std::function<integer_t(integer_t,integer_t)> emit =
      [&](integer_t node, integer_t pa_out) -> integer_t {
      // (1) small subtree -> single leaf front, columns in postorder.
      if (subsize[node] < k) {
        integer_t id = out.size();
        out.emplace_back(0, pa_out, -1, -1);      // patched below
        append_subtree(node);
        out[id].sep_end = cum;
        return id;
      }
      // Reserve this front; we fill its children and separator after we
      // know the full contracted chain.
      integer_t id = out.size();
      out.emplace_back(0, pa_out, -1, -1);
      // Walk the contraction chain.  At each step we keep the front's
      // NON-contracted child as a pending child subtree to emit later, and
      // continue into the contracted child.  We collect, in order, the
      // side subtrees whose columns precede this front's separator columns.
      std::vector<integer_t> side_children;       // roots to emit as children
      std::vector<integer_t> chain_nodes;         // nodes whose sep merges in
      integer_t cur = node;
      for (;;) {
        chain_nodes.push_back(cur);
        integer_t cl = lch[cur], cr = rch[cur];
        if (cl == -1 && cr == -1) break;          // leaf: chain ends
        // choose heavier child as the one to descend into (heavy path)
        integer_t heavy = (subsize[cl] >= subsize[cr]) ? cl : cr;
        integer_t light = (heavy == cl) ? cr : cl;
        // Contract into heavy iff heavy's SEPARATOR is small (< k); this
        // is the "merge either side" step.  Otherwise stop the chain and
        // recurse into both children as separate child fronts.
        if (sz[heavy] < k && subsize[heavy] >= k) {
          side_children.push_back(light);         // light stays a child
          cur = heavy;                            // continue chain
          continue;
        }
        // stop: both children become child fronts of this front.
        side_children.push_back(cl);
        side_children.push_back(cr);
        break;
      }
      // Emit side children first (their columns precede this front's
      // separator columns), in increasing original offset so the emitted
      // column order is a valid postorder-compatible layout.
      std::sort(side_children.begin(), side_children.end(),
                [&off](integer_t a, integer_t b){ return off[a] < off[b]; });
      std::vector<integer_t> kid_out;
      for (integer_t c : side_children) kid_out.push_back(emit(c, id));
      // Now append this front's merged separator columns: the separators
      // of every node on the contracted chain, in chain order (deepest
      // first so columns stay increasing/contiguous with the children).
      std::sort(chain_nodes.begin(), chain_nodes.end(),
                [&off](integer_t a, integer_t b){ return off[a] < off[b]; });
      for (integer_t cn : chain_nodes) append_cols(cn);
      // Wire children: a front has 0 or 2 children; if >2, join with empty
      // separators (left-deep) to preserve the invariant.
      integer_t nl = -1, nr = -1;
      if (kid_out.size() == 1) {
        // single child: pair it with an empty leaf to keep 0-or-2.  Insert
        // an empty leaf child before it in column order.
        integer_t eid = out.size();
        out.emplace_back(cum, id, -1, -1);        // empty leaf (size 0)
        nl = eid; nr = kid_out[0];
      } else if (kid_out.size() == 2) {
        nl = kid_out[0]; nr = kid_out[1];
      } else if (kid_out.size() > 2) {
        integer_t acc = kid_out[0];
        for (std::size_t j=1; j+1<kid_out.size(); j++) {
          integer_t jid = out.size();
          out.emplace_back(cum, id, acc, kid_out[j]); // empty joiner
          out[acc].pa = jid; out[kid_out[j]].pa = jid;
          acc = jid;
        }
        nl = acc; nr = kid_out.back();
      }
      out[id].lch = nl; out[id].rch = nr;
      if (nl != -1) out[nl].pa = id;
      if (nr != -1) out[nr].pa = id;
      out[id].sep_end = cum;
      return id;
    };
    integer_t root_out = emit(root, -1);
    out[root_out].pa = -1;

    // The DFS reserves a parent front BEFORE its children, so front IDs are
    // in pre-order while columns/sep_end are already in postorder.  Renumber
    // fronts into postorder (children before parents) by sorting on sep_end.
    integer_t no = out.size();
    std::vector<integer_t> order(no);
    for (integer_t i=0; i<no; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&out](integer_t a, integer_t b){
                       return out[a].sep_end < out[b].sep_end; });
    std::vector<integer_t> nid(no);
    for (integer_t i=0; i<no; i++) nid[order[i]] = i;
    std::vector<Separator<integer_t>> res; res.reserve(no);
    for (integer_t i=0; i<no; i++) {
      auto& s = out[order[i]];
      res.emplace_back(s.sep_end,
                       s.pa  == -1 ? -1 : nid[s.pa],
                       s.lch == -1 ? -1 : nid[s.lch],
                       s.rch == -1 ? -1 : nid[s.rch]);
    }
    if (colperm) *colperm = std::move(cperm);
    return res;
  }

  template<typename integer_t> SeparatorTree<integer_t>
  build_sep_tree_from_perm(const integer_t* ptr, const integer_t* ind,
                           std::vector<integer_t>& perm,
                           std::vector<integer_t>& iperm) {
    auto etree = etree_from_perm(ptr, ind, perm);
    auto n = perm.size();
    std::vector<integer_t> post(n);
    auto seps = separators_from_etree(etree, post);
    // Relaxed supernode amalgamation: collapse small subtrees into single
    // leaf fronts to reduce the number of (tiny) fronts.
    std::vector<integer_t> colperm;
    // Default amalgamation threshold 32.
    integer_t amalg = 32;
    seps = amalgamate_separators<integer_t>(seps, amalg, &colperm);
    // Compose the amalgamation column reordering (if any) into the final
    // permutation.  colperm[new_pos] = old_pos (old postorder position);
    // its inverse maps old postorder position -> new position.
    if (!colperm.empty()) {
      std::vector<integer_t> colinv(colperm.size());
      for (std::size_t p=0; p<colperm.size(); p++) colinv[colperm[p]] = p;
      for (std::size_t i=0; i<n; i++)
        iperm[i] = colinv[post[perm[i]]];
    } else {
      for (std::size_t i=0; i<n; i++)
        iperm[i] = post[perm[i]];
    }
    for (std::size_t i=0; i<n; i++)
      perm[iperm[i]] = i;
    std::swap(perm, iperm);
    return SeparatorTree<integer_t>(seps);
  }

  /** path halving */
  template<typename integer_t> inline integer_t
  find(integer_t i, std::vector<integer_t>& pp) {
    auto p = pp[i];
    auto gp = pp[p];
    while (gp != p) {
      pp[i] = gp;
      i = gp;
      p = pp[i];
      gp = pp[p];
    }
    return p;
  }

  template<typename integer_t> std::vector<integer_t>
  spsymetree(const integer_t *acolst,    // column starts
             const integer_t *acolend,   //   and ends past 1
             const integer_t *arow,      // row indices of A
             integer_t n,                // dimension of A
             integer_t subgraph_begin) { // first row/column of subgraph
    // if working on subgraph, acolst/end only for subgraph and n is
    // number of vertices in the subgraph
    std::vector<integer_t> root(n, 0), pp(n, 0), parent(n);
    for (integer_t col=0; col<n; col++) {
      auto cset = pp[col] = col;
      root[cset] = col;
      parent[col] = n;
      for (integer_t p=acolst[col]; p<acolend[col]; p++) {
        auto row = arow[p] - subgraph_begin;
        if (row >= col) continue;
        auto rset = find(row, pp);
        auto rroot = root[rset];
        if (rroot != col) {
          parent[rroot] = col;
          cset = pp[cset] = rset;
          root[cset] = col;
        }
      }
    }
    return parent;
  }


  // explicit template instantiations
  template class SeparatorTree<int>;
  template class SeparatorTree<long int>;
  template class SeparatorTree<long long int>;

  template SeparatorTree<int>
  build_sep_tree_from_perm(const int* ptr, const int* ind,
                           std::vector<int>& perm,
                           std::vector<int>& iperm);
  template SeparatorTree<long int>
  build_sep_tree_from_perm(const long int* ptr, const long int* ind,
                           std::vector<long int>& perm,
                           std::vector<long int>& iperm);
  template SeparatorTree<long long int>
  build_sep_tree_from_perm(const long long int* ptr, const long long int* ind,
                           std::vector<long long int>& perm,
                           std::vector<long long int>& iperm);

  template std::vector<int>
  spsymetree(const int* acolst, const int* acolend,
             const int* arow, int n, int subgraph_begin);
  template std::vector<long int>
  spsymetree(const long int* acolst, const long int* acolend,
             const long int* arow, long int n, long int subgraph_begin);
  template std::vector<long long int>
  spsymetree(const long long int* acolst, const long long int* acolend,
             const long long int* arow, long long int n,
             long long int subgraph_begin);

  template std::vector<int>
  etree_postorder(const std::vector<int>& etree);
  template std::vector<long int>
  etree_postorder(const std::vector<long int>& etree);
  template std::vector<long long int>
  etree_postorder(const std::vector<long long int>& etree);

  template std::vector<Separator<int>>
  separators_from_etree(std::vector<int>& etree,
                        std::vector<int>& post);
  template std::vector<Separator<long int>>
  separators_from_etree(std::vector<long int>& etree,
                        std::vector<long int>& post);
  template std::vector<Separator<long long int>>
  separators_from_etree(std::vector<long long int>& etree,
                        std::vector<long long int>& post);

} // end namespace strumpack
