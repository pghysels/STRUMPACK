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
#ifndef HSS_PARTITION_TREE_HPP
#define HSS_PARTITION_TREE_HPP

#include <vector>
#include <unordered_map>
#include <cassert>
#include <iostream>

namespace strumpack {
  namespace HSS {

    /** a node in this tree should always have 0 or 2 children */
    class HSSPartitionTree {
    public:
      int size;
      std::vector<HSSPartitionTree> c;
      HSSPartitionTree() : size(0) {}
      HSSPartitionTree(int n) : size(n) {}
      HSSPartitionTree(std::vector<int>& buf) {
        int n = buf.size() / 3;
        int pid = n-1;
        de_serialize_rec(buf.data(), buf.data()+n, buf.data()+2*n, pid);
      }
      HSSPartitionTree(int buf_size, int* buf) {
        int n = buf_size / 3;
        int pid = n-1;
        de_serialize_rec(buf, buf+n, buf+2*n, pid);
      }
      void refine(int leaf_size) {
        assert(c.empty());
        if (size > 2*leaf_size) {
          c.resize(2);
          c[0].size = size/2;
          c[0].refine(leaf_size);
          c[1].size = size - size/2;
          c[1].refine(leaf_size);
        }
      }
      void print() {
        for (auto& ch : c) ch.print();
        std::cout << size << " ";
      }
      int nodes() {
        int nr_nodes = 1;
        for (auto& ch : c) nr_nodes += ch.nodes();
        return nr_nodes;
      }
      std::vector<int> serialize() {
        int n = nodes(), pid = 0;
        std::vector<int> buf(3*n);
        serialize_rec(buf.data(), buf.data()+n, buf.data()+2*n, pid);
        return buf;
      }
    private:
      void serialize_rec(int* sizes, int* lchild, int* rchild, int& pid) {
        if (!c.empty()) {
          c[0].serialize_rec(sizes, lchild, rchild, pid);
          auto lroot = pid;
          c[1].serialize_rec(sizes, lchild, rchild, pid);
          lchild[pid] = lroot-1;
          rchild[pid] = pid-1;
        } else lchild[pid] = rchild[pid] = -1;
        sizes[pid++] = size;
      }
      void de_serialize_rec(int* sizes, int* lchild, int* rchild, int& pid) {
        size = sizes[pid--];
        if (rchild[pid+1] != -1) {
          c.resize(2);
          c[1].de_serialize_rec(sizes, lchild, rchild, pid);
          c[0].de_serialize_rec(sizes, lchild, rchild, pid);
        }
      }
    };

    template<typename integer_t> std::vector<int>
    serialize(std::unordered_map<integer_t, HSSPartitionTree>& hss_tree_map) {
      std::vector<int> buf;
      buf.push_back(hss_tree_map.size());
      for (auto& ht : hss_tree_map) {
        buf.push_back(ht.first);
        auto ht_buf = ht.second.serialize();
        buf.push_back(ht_buf.size());
        buf.insert(buf.end(), ht_buf.begin(), ht_buf.end());
      }
      return buf;
    }

  } // end namespace HSS
} // end namespace strumpack

#endif
