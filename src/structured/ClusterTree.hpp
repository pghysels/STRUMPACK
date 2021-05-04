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
/**
 * \file ClusterTree.hpp
 * \brief This file contains the ClusterTree class definition.
 */
#ifndef CLUSTER_TREE_HPP
#define CLUSTER_TREE_HPP

#include <vector>
#include <unordered_map>
#include <cassert>
#include <iostream>

namespace strumpack {
  namespace structured {

    /**
     * \class ClusterTree
     *
     * \brief The cluster tree, or partition tree that represents the
     * partitioning of the rows or columns of a hierarchical matrix.
     *
     * This uses a recursive representation, a child in the tree is
     * also an ClusterTree. A node in this tree should always have 0
     * or 2 children, and the size of the tree should always be the
     * sum of the sizes of its children.
     *
     * To build a whole tree use the constructor ClusterTree(int n)
     * and then refine it, either evenly using refine(int leaf_size),
     * or manualy by adding nodes to the child vector ClusterTree::c.
     *
     * Or you can use one of the clustering algorithms, see
     * binary_tree_clustering()
     */
    class ClusterTree {
    public:
      /**
       * Size of the corresponing matrix.  This should always be >= 0
       * and should be equal to the sum of the sizes of the children.
       */
      int size;

      /**
       * Vector with children. This should always have c.size() == 0
       * or c.size() == 2, for a leaf and an internal node of the tree
       * respectively.
       */
      std::vector<ClusterTree> c;

      /**
       * Constructor, initializes to an empty tree, with size 0.
       */
      ClusterTree() : size(0) {}

      /**
       * Constructor, initializes to a matrix with size n, only 1 node
       * (1 level), so no refinement.
       *
       * \param n Size of the corresponding matrix.
       * \see n, c, refine
       */
      ClusterTree(int n) : size(n) {}

      /**
       * Refine the tree to a given leaf size. This just splits the
       * nodes into equal parts (+/- 1) as long as a node is larger
       * than 2*leaf_size.
       *
       * \param leaf_size Leaf size, leafs in the resulting tree will
       * be smaller or equal to this leaf_size.
       */
      const ClusterTree& refine(int leaf_size) {
        assert(c.empty());
        if (size >= 2*leaf_size) {
          c.resize(2);
          c[0].size = size/2;
          c[0].refine(leaf_size);
          c[1].size = size - size/2;
          c[1].refine(leaf_size);
        }
        return *this;
      }

      /**
       * Print the sizes of the nodes in the tree, in postorder.
       */
      void print() const {
        for (auto& ch : c) ch.print();
        std::cout << size << " ";
      }

      /**
       * Return the total number of nodes in this tree.
       *
       * \return Total number of nodes, 1 for a tree with only 1
       * level, ..
       */
      int nodes() const {
        int nr_nodes = 1;
        for (auto& ch : c)
          nr_nodes += ch.nodes();
        return nr_nodes;
      }

      /**
       * Return the number of levels in this tree.
       *
       * \return Number of levels in the tree (>= 1).
       */
      int levels() const {
        int lvls = 0;
        for (auto& ch : c)
          lvls = std::max(lvls, ch.levels());
        return lvls + 1;
      }

      /**
       * Truncate this tree to a complete tree. This routine will
       * first find the leaf node with the least number of ancestors,
       * then remove all nodes which have more ancestors. The
       * resulting tree will be complete and binary.
       *
       * \see expand_complete
       */
      void truncate_complete() {
        truncate_complete_rec(1, min_levels());
      }

      /**
       * Further refine the tree to make it complete. This will add
       * zero or size 1 leaf nodes at the lowest level.
       *
       * \param allow_zero_nodes If True, then the leaf nodes which
       * are not at the lowest level, are split into two nodes with
       * sizes (size,0), recursively until the tree is complete. If
       * False, a leaf node which is not at the lowest level, will be
       * split into 2 children with sizes (size-x,x) where x is such
       * that all leafs in the final tree have at least size 1 (this
       * might fail in certain cases, if too many levels are
       * required).
       *
       * \see truncate_complete
       */
      void expand_complete(bool allow_zero_nodes) {
        expand_complete_rec(1, levels(), allow_zero_nodes);
      }

      /**
       * Further refine the tree to make it a complete tree with a
       * specified number of levels.
       *
       * \param lvls The requested number of levels. Should be >=
       * this->levels.
       */
      void expand_complete_levels(int lvls) {
        expand_complete_levels_rec(1, lvls);
      }

      /**
       * Constructor, taking a vector desribing an entire tree. This
       * is used for deserialization. The constructor argument buf
       * should be one obtained from calling serialize on an
       * ClusterTree object.
       *
       * \param buf Serialized ClusterTree
       * \see serialize
       */
      template<typename integer_t> static ClusterTree
      deserialize(const std::vector<integer_t>& buf) {
        return deserialize(buf.data());
      }

      /**
       * Constructor, taking a vector desribing an entire tree. This
       * is used for deserialization. The constructor argument buf
       * should be one obtained from calling serialize on an
       * ClusterTree object.
       *
       * \param buf Serialized ClusterTree
       * \see serialize
       */
      template<typename integer_t> static ClusterTree
      deserialize(const integer_t* buf) {
        ClusterTree t;
        int n = buf[0];
        int pid = n-1;
        t.de_serialize_rec(buf+1, buf+n+1, buf+2*n+1, pid);
        return t;
      }

      /**
       * Serialize the entire tree to a single vector storing size and
       * child/parent info. This can be used to communicate the tree
       * with MPI for instance.  A new tree can be constructed at the
       * receiving side with the apporiate constructor.
       *
       * \return Serialized tree. Do not rely on the meaning of the
       * elements in the vector, only use with the corresponding
       * constructor.
       */
      std::vector<int> serialize() const {
        int n = nodes(), pid = 0;
        std::vector<int> buf(3*n+1);
        buf[0] = n;
        serialize_rec(buf.data()+1, buf.data()+n+1, buf.data()+2*n+1, pid);
        return buf;
      }

      /**
       * Check whether this tree is complete.
       *
       * \return True if this tree is complete, False otherwise.
       * \see expand_complete, truncate_complete
       */
      bool is_complete() const {
        if (c.empty()) return true;
        else return c[0].levels() == c[1].levels();
      }

      /**
       * Return a vector with leaf sizes.
       *
       * \return vector with the sizes of the leafs.
       */
      template<typename integer_t>
      std::vector<integer_t> leaf_sizes() const {
        std::vector<integer_t> lf;
        leaf_sizes_rec(lf);
        return lf;
      }

      /**
       * Return a map from nodes in a complete tree, with lvls levels,
       * numbered by level, with the root being 1, to leafs of the
       * original tree before it was made complete.  Note that in the
       * level by level ordering, a node with number id has children
       * 2*id and 2*id+1.
       */
      std::pair<std::vector<int>,std::vector<int>>
      map_from_complete_to_leafs(int lvls) const {
        int n = (1 << lvls) - 1;
        std::vector<int> map0(n, -1), map1(n, -1);
        int leaf = 0;
        complete_to_orig_rec(1, map0, map1, leaf);
        for (int i=0; i<n; i++) {
          if (map0[i] == -1) map0[i] = map0[(i+1)/2-1];
          if (map1[i] == -1) map1[i] = map1[(i+1)/2-1];
        }
        // std::cout << "nodes=" << nodes() << " levels()=" << levels()
        //           << " lvls=" << lvls << " map0/map1 = [";
        // for (int i=0; i<n; i++)
        //   std::cout << map0[i] << "/" << map1[i] << " ";
        // std::cout << std::endl;
        return {map0, map1};
      }

    private:
      int min_levels() const {
        int lvls = levels();
        for (auto& ch : c)
          lvls = std::min(lvls, 1 + ch.min_levels());
        return lvls;
      }
      void truncate_complete_rec(int lvl, int lvls) {
        if (lvl == lvls) c.clear();
        else
          for (auto& ch : c)
            ch.truncate_complete_rec(lvl+1, lvls);
      }
      void expand_complete_rec(int lvl, int lvls, bool allow_zero_nodes) {
        if (c.empty()) {
          if (lvl != lvls) {
            c.resize(2);
            if (allow_zero_nodes) {
              c[0].size = size;
              c[1].size = 0;
            } else {
              int l1 = 1 << (lvls - lvl - 1);
              c[0].size = size - l1;
              c[1].size = l1;
            }
            c[0].expand_complete_rec(lvl+1, lvls, allow_zero_nodes);
            c[1].expand_complete_rec(lvl+1, lvls, allow_zero_nodes);
          }
        } else
          for (auto& ch : c)
            ch.expand_complete_rec(lvl+1, lvls, allow_zero_nodes);
      }

      void complete_to_orig_rec
      (int id, std::vector<int>& map0, std::vector<int>& map1,
       int& leaf) const {
        if (c.empty()) map0[id-1] = map1[id-1] = leaf++;
        else {
          c[0].complete_to_orig_rec(id*2, map0, map1, leaf);
          c[1].complete_to_orig_rec(id*2+1, map0, map1, leaf);
          map0[id-1] = map0[id*2-1];
          map1[id-1] = map1[id*2];
        }
      }

      void expand_complete_levels_rec(int lvl, int lvls) {
        if (c.empty()) {
          if (lvl != lvls) {
            c.resize(2);
            c[0].size = size / 2;
            c[1].size = size - size / 2;
            c[0].expand_complete_levels_rec(lvl+1, lvls);
            c[1].expand_complete_levels_rec(lvl+1, lvls);
          }
        } else
          for (auto& ch : c)
            ch.expand_complete_levels_rec(lvl+1, lvls);
      }
      template<typename integer_t>
      void leaf_sizes_rec(std::vector<integer_t>& lf) const {
        for (auto& ch : c)
          ch.leaf_sizes_rec(lf);
        if (c.empty())
          lf.push_back(size);
      }
      void serialize_rec(int* sizes, int* lchild, int* rchild, int& pid) const {
        if (!c.empty()) {
          c[0].serialize_rec(sizes, lchild, rchild, pid);
          auto lroot = pid;
          c[1].serialize_rec(sizes, lchild, rchild, pid);
          lchild[pid] = lroot-1;
          rchild[pid] = pid-1;
        } else lchild[pid] = rchild[pid] = -1;
        sizes[pid++] = size;
      }

      template<typename integer_t> void de_serialize_rec
      (const integer_t* sizes, const integer_t* lchild,
       const integer_t* rchild, int& pid) {
        size = sizes[pid--];
        if (rchild[pid+1] != -1) {
          c.resize(2);
          c[1].de_serialize_rec(sizes, lchild, rchild, pid);
          c[0].de_serialize_rec(sizes, lchild, rchild, pid);
        }
      }
    };

  } // end namespace structured
} // end namespace strumpack

#endif
