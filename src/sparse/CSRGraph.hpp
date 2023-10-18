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

#include <vector>
#include <unordered_map>

#include "structured/ClusterTree.hpp"
#include "dense/DenseMatrix.hpp"

#if defined(STRUMPACK_USE_MPI)
#include "misc/MPIWrapper.hpp"
#endif

namespace strumpack {


  /**
   * Compressed sparse row representation of a graph.
   *
   * This is also used as to store the local part of a 1-D block-row
   * distributed graph. So the column indices might not map to valid
   * nodes in this graph, ie, this graph contains outgoing edges to
   * nodes not part of this graph.
   */
  template<typename integer_t> class CSRGraph {
    using Length2Edges = std::unordered_map<integer_t, std::vector<integer_t>>;

  public:
    CSRGraph() = default;
    CSRGraph(integer_t nr_vert, integer_t nr_edge);
    CSRGraph(std::vector<integer_t>&& ptr, std::vector<integer_t>&& ind);

    static CSRGraph<integer_t> deserialize(const std::vector<integer_t>& buf);
    static CSRGraph<integer_t> deserialize(const integer_t* buf);

    std::vector<integer_t> serialize() const;

    void print();
    integer_t size() const { return ptr_.size()-1; }
    integer_t vertices() const { return ptr_.size()-1; }
    integer_t edges() const { return ind_.size(); }

    const integer_t* ptr() const { return ptr_.data(); }
    const integer_t* ind() const { return ind_.data(); }
    integer_t* ptr() { return ptr_.data(); }
    integer_t* ind() { return ind_.data(); }
    const integer_t& ptr(integer_t i) const { assert(i <= vertices()); return ptr_[i]; }
    const integer_t& ind(integer_t i) const { assert(i < edges()); return ind_[i]; }
    integer_t& ptr(integer_t i) { assert(i <= vertices()); return ptr_[i]; }
    integer_t& ind(integer_t i) { assert(i < edges()); return ind_[i]; }

    void sort_rows();

    void permute(const integer_t* order, const integer_t* iorder);
    void permute_rows(const integer_t* iorder);
    void permute_cols(const integer_t* order);

    void permute_local(const std::vector<integer_t>& order,
                       const std::vector<integer_t>& iorder,
                       integer_t clo, integer_t chi);

    void permute_rows_local_cols_global(const std::vector<integer_t>& order,
                                        const std::vector<integer_t>& iorder,
                                        integer_t clo, integer_t chi);

    structured::ClusterTree
    recursive_bisection(int leaf, int conn_level, integer_t* order,
                        integer_t* iorder, integer_t lo, integer_t sep_begin,
                        integer_t sep_end) const;

    std::vector<std::size_t>
    partition_K_way(int K, integer_t* order, integer_t* iorder, integer_t lo,
                    integer_t sep_begin, integer_t sep_end) const;

    template<typename int_t> DenseMatrix<bool>
    admissibility(const std::vector<int_t>& tiles) const;

    void print_dense(const std::string& name, integer_t cols=-1) const;

#if defined(STRUMPACK_USE_MPI)
    void broadcast(const MPIComm& comm);
#endif

  private:
    std::vector<integer_t> ptr_, ind_;

    Length2Edges length_2_edges(integer_t lo) const;

    void split_recursive(int leaf, int conn_level, integer_t lo,
                         integer_t sep_begin, integer_t sep_end,
                         integer_t* order, structured::ClusterTree& tree,
                         integer_t& parts, integer_t part,
                         integer_t count, const Length2Edges& l2) const;

    /**
     * Extract the separator from sep_begin to sep_end. Also add extra
     * length-2 edges if sep_order_level > 0.
     *
     * This only extracts the nodes i for which order[i] == part.
     */
    CSRGraph<integer_t>
    extract_subgraph(int order_level, integer_t lo, integer_t begin,
                     integer_t end, integer_t part, const integer_t* order,
                     const Length2Edges& o) const;
  };

  // template<typename integer_t, typename int_t>
  // DenseMatrix<bool> admissibility
  // (const CSRGraph<integer_t>& g11, const CSRGraph<integer_t>& g12,
  //  const CSRGraph<integer_t>& g22, const std::vector<int_t>& rtiles,
  //  const std::vector<int_t>& ctiles, int knn);

} // end namespace strumpack

#endif // CSRGRAPH_HPP
