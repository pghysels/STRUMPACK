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
#ifndef CSRGRAPH_HPP
#define CSRGRAPH_HPP
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include "misc/MPIWrapper.hpp"

namespace strumpack {

  template<class integer_t>
  using Length2Edges = std::unordered_map<integer_t, std::vector<integer_t>>;

  /**
   * Compressed sparse row representation of a graph.
   *
   * This is also used as to store the local part of a 1-D block-row
   * distributed graph. So the column indices might not map to valid
   * nodes in this graph, ie, this graph contains outgoing edges to
   * nodes not part of this graph.
   */
  template<typename integer_t> class CSRGraph {
  public:
    CSRGraph() = default;
    CSRGraph(integer_t nr_vert, integer_t nr_edge);

    static CSRGraph<integer_t> deserialize(const std::vector<integer_t>& buf);
    static CSRGraph<integer_t> deserialize(const integer_t* buf);

    std::vector<integer_t> serialize() const;

    void print();
    integer_t size() const { return ptr_.size()-1; }
    integer_t nvert() const { return ptr_.size()-1; }
    integer_t nedge() const { return ind_.size(); }

    const integer_t* ptr() const { return ptr_.data(); }
    const integer_t* ind() const { return ind_.data(); }
    integer_t* ptr() { return ptr_.data(); }
    integer_t* ind() { return ind_.data(); }
    const integer_t& ptr(integer_t i) const { assert(i <= nvert()); return ptr_[i]; }
    const integer_t& ind(integer_t i) const { assert(i < nedge()); return ind_[i]; }
    integer_t& ptr(integer_t i) { assert(i <= nvert()); return ptr_[i]; }
    integer_t& ind(integer_t i) { assert(i < nedge()); return ind_[i]; }

    void sort_rows();

    void permute
    (const integer_t* order, const integer_t* iorder);

    void permute_local
    (const std::vector<integer_t>& order,
     const std::vector<integer_t>& iorder,
     integer_t clo, integer_t chi);

    void permute_columns(const std::vector<integer_t>& order);

    void permute_rows_local_cols_global
    (const std::vector<integer_t>& order,
     const std::vector<integer_t>& iorder,
     integer_t clo, integer_t chi);

    /**
     * Extract the separator from sep_begin to sep_end. Also add extra
     * length-2 edges if sep_order_level > 0.
     *
     * This only extracts the nodes i for which order[i] == part.
     *
     * This stores some work info for the lenght 2 edges, so it is not
     * thread safe!!
     */
    CSRGraph<integer_t> extract_subgraph
    (int order_level, integer_t lo, integer_t begin, integer_t end,
     integer_t part, const integer_t* order,
     const Length2Edges<integer_t>& o) const;

    CSRGraph<integer_t> extract_subgraph
    (integer_t lo, integer_t begin, integer_t end) const;

    Length2Edges<integer_t> length_2_edges(integer_t lo) const;

  private:
    std::vector<integer_t> ptr_, ind_;
  };

  template<typename integer_t>
  CSRGraph<integer_t>::CSRGraph(integer_t nr_vert, integer_t nr_edge)
    : ptr_(nr_vert+1), ind_(nr_edge) {
  }

  template<typename integer_t> CSRGraph<integer_t>
  CSRGraph<integer_t>::deserialize(const std::vector<integer_t>& buf) {
    return CSRGraph<integer_t>::deserialize(buf.data());
  }

  template<typename integer_t> CSRGraph<integer_t>
  CSRGraph<integer_t>::deserialize(const integer_t* buf) {
    CSRGraph<integer_t> g(buf[0]-1, buf[1]);
    std::copy(buf+2, buf+2+buf[0], g.ptr_.begin());
    std::copy(buf+2+buf[0], buf+2+buf[0]+buf[1], g.ind_.begin());
    return g;
  }

  template<typename integer_t> std::vector<integer_t>
  CSRGraph<integer_t>::serialize() const {
    std::vector<integer_t> buf(2+ptr_.size()+ind_.size());
    buf[0] = ptr_.size();
    buf[1] = ind_.size();
    std::copy(ptr_.begin(), ptr_.end(), buf.begin()+2);
    std::copy(ind_.begin(), ind_.end(), buf.begin()+2+ptr_.size());
    return buf;
  }

  template<typename integer_t> void
  CSRGraph<integer_t>::sort_rows() {
    auto n = size();
#pragma omp parallel for
    for (integer_t r=0; r<n; r++)
      std::sort(&ind_[ptr_[r]], &ind_[ptr_[r+1]]);
  }

  template<typename integer_t> void
  CSRGraph<integer_t>::print() {
    for (integer_t i=0; i<size(); i++) {
      std::cout << "r=" << i << ", ";
      for (integer_t j=ptr_[i]; j<ptr_[i+1]; j++)
        std::cout << ind_[j] << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  template<typename integer_t> void
  CSRGraph<integer_t>::permute
  (const integer_t* iorder, const integer_t* order) {
    auto n = nvert();
    std::vector<integer_t> ptr(ptr_.size()), ind(ind_.size());
    integer_t nnz = 0;
    for (integer_t i=0; i<n; i++) {
      auto lb = ptr_[iorder[i]];
      auto ub = ptr_[iorder[i]+1];
      for (integer_t j=lb; j<ub; j++)
        ind[nnz] = order[ind_[j]];
      ptr[i+1] = nnz;
    }
    std::swap(ptr_, ptr);
    std::swap(ind_, ind);
  }


  /**
   * order and iorder are of size this->size() == this->nvert() == chi-clo.
   * order has elements in the range [clo, chi).
   * iorder had elements in the range [0, chi-clo).
   *
   * This applies permutation only in the diagonal block. This is
   * useful for the local subgraphs. The off-diagonal elements
   * represent the connections to the distributed separators, which
   * are not permuted, so this permutation can be done completely
   * local!
   */
  template<typename integer_t> void
  CSRGraph<integer_t>::permute_local
  (const std::vector<integer_t>& order, const std::vector<integer_t>& iorder,
   integer_t clo, integer_t chi) {
    std::vector<integer_t> ptr(ptr_.size()), ind(ind_.size());
    integer_t nnz = 0;
    auto n = size();
    ptr[0] = 0;
    for (integer_t i=0; i<n; i++) {
      auto lb = ptr_[iorder[i]];
      auto ub = ptr_[iorder[i]+1];
      for (integer_t j=lb; j<ub; j++) {
        auto c = ind_[j];
        ind[nnz++] = (c >= clo && c < chi) ? order[c-clo] : c;
      }
      ptr[i+1] = nnz;
    }
    std::swap(ptr_, ptr);
    std::swap(ind_, ind);
  }

  /**
   * iorder is of size this->size() == this->nvert().
   * iorder had elements in the range [0, chi-clo).
   * order is of the global size.
   */
  template<typename integer_t> void
  CSRGraph<integer_t>::permute_rows_local_cols_global
  (const std::vector<integer_t>& order,
   const std::vector<integer_t>& iorder,
   integer_t clo, integer_t chi) {
    std::vector<integer_t> ptr(ptr_.size()), ind(ind_.size());
    integer_t nnz = 0;
    auto n = size();
    ptr[0] = 0;
    for (integer_t i=0; i<n; i++) {
      auto lb = ptr_[iorder[i]];
      auto ub = ptr_[iorder[i]+1];
      for (integer_t j=lb; j<ub; j++)
        ind[nnz++] = order[ind_[j]];
      ptr[i+1] = nnz;
    }
    std::swap(ptr_, ptr);
    std::swap(ind_, ind);
  }

  template<typename integer_t> void
  CSRGraph<integer_t>::permute_columns
  (const std::vector<integer_t>& order) {
    auto n = size();
    for (integer_t i=0; i<n; i++)
      for (integer_t j=ptr_[i]; j<ptr_[i+1]; j++)
        ind_[j] = order[ind_[j]];
  }

  template<typename integer_t> CSRGraph<integer_t>
  CSRGraph<integer_t>::extract_subgraph
  (integer_t lo, integer_t begin, integer_t end) const {
    auto n = size();
    auto dim = end - begin;
    integer_t e = 0;
    for (integer_t r=begin; r<end; r++)
      for (integer_t j=ptr_[r]; j<ptr_[r+1]; j++) {
        auto c = ind_[j] - lo;
        if (c != r && c >= begin && c < end)
          e++;
      }
    CSRGraph<integer_t> g(dim, e);
    e = 0;
    for (integer_t r=begin; r<end; r++) {
      for (integer_t j=ptr_[r]; j<ptr_[r+1]; j++) {
        auto c = ind_[j] - lo;
        if (c != r) {
          auto lc = c - begin;
          if (lc >= 0 && lc < dim)
            g.ind(e++) = c - begin;
        }
      }
      g.ptr(r-begin+1) = e;
    }
    for (integer_t i=0; i<g.nvert(); i++) {
      for (integer_t j=g.ptr(i); j<g.ptr(i+1); j++) {
        assert(j >= 0);
        assert(j < g.nedge());
        assert(g.ind(j) >= 0);
        assert(g.ind(j) < g.nvert());
      }
    }
    return g;
  }


  template<typename integer_t> Length2Edges<integer_t>
  CSRGraph<integer_t>::length_2_edges(integer_t lo) const {
    // for all nodes not in this graph to which there is an outgoing
    //   edge, we keep a list of edges to/from that node
    Length2Edges<integer_t> o;
    auto n = size();
    auto hi = lo + n;
    for (integer_t r=0; r<n; r++)
      for (integer_t j=ptr_[r]; j<ptr_[r+1]; j++) {
        auto c = ind_[j];
        if (c < lo || c >= hi) o[c].push_back(r);
      }
    return o;
  }

  template<typename integer_t> CSRGraph<integer_t>
  CSRGraph<integer_t>::extract_subgraph
  (int order_level, integer_t lo, integer_t begin, integer_t end,
   integer_t part, const integer_t* order,
   const Length2Edges<integer_t>& o) const {
    CSRGraph<integer_t> g;
    assert(order_level == 0 || order_level == 1);
    auto n = size();
    auto dim = end - begin;
    std::vector<bool> mark(dim);
    std::unique_ptr<integer_t[]> ind_to_part(new integer_t[dim]);
    integer_t count = 0;
    for (integer_t r=0; r<dim; r++)
      ind_to_part[r] = (order[r] == part) ? count++ : -1;
    g.ptr_.reserve(count+1);
    for (integer_t r=begin, edges=0; r<end; r++) {
      if (order[r-begin] == part) {
        g.ptr_.push_back(edges);
        std::fill(mark.begin(), mark.end(), false);
        for (integer_t j=ptr_[r]; j<ptr_[r+1]; j++) {
          auto c = ind_[j] - lo;
          if (c == r) continue;
          if (c >= 0 && c < n) {
            auto lc = c - begin;
            if (lc >= 0 && lc < dim && order[lc] == part && !mark[lc]) {
              mark[lc] = true;
              g.ind_.push_back(ind_to_part[lc]);
              edges++;
            } else {
              if (order_level > 0) {
                for (integer_t k=ptr_[c]; k<ptr_[c+1]; k++) {
                  auto cc = ind_[k] - lo;
                  auto lcc = cc - begin;
                  if (cc != r && lcc >= 0 && lcc < dim &&
                      order[lcc] == part && !mark[lcc]) {
                    mark[lcc] = true;
                    g.ind_.push_back(ind_to_part[lcc]);
                    edges++;
                  }
                }
              }
            }
          } else {
            if (order_level > 0) {
              for (auto cc : o.at(c+lo)) {
                auto lcc = cc - begin;
                if (cc != r && lcc >= 0 &&
                    lcc < dim && order[lcc] == part && !mark[lcc]) {
                  mark[lcc] = true;
                  g.ind_.push_back(ind_to_part[lcc]);
                  edges++;
                }
              }
            }
          }
        }
      }
    }
    g.ptr_.push_back(g.ind_.size());
    return g;
  }

} // end namespace strumpack

#endif //CSRGRAPH_HPP
