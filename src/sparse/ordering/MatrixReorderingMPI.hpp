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
#ifndef MATRIX_REORDERING_MPI_HPP
#define MATRIX_REORDERING_MPI_HPP

#include <memory>
#include <vector>

#include "MatrixReordering.hpp"
#include "sparse/CSRMatrixMPI.hpp"

namespace strumpack {

  /**
   * A MatrixReorderingMPI has a distributed separator tree. This tree
   * is stored using 2 trees, one in the base class MatrixReordering
   * and one here in MatrixReorderingMPI. The tree in MatrixReordering
   * (sep_tree_) is the top of the tree, corresponding to the
   * distributed separators. The tree stored here (local_tree_)
   * corresponds to the local subtree.
   *
   * The distributed tree should have P leafs.  Lets number the
   * distributed separators level by level, root=1.  For instance: for
   * P=5, the distributed tree will look like:
   *                   1
   *                  / \
   *                 /   \
   *                2     3
   *               / \   / \
   *              4   5 6   7
   *             / \
   *            8   9
   *
   * The number of the node is given by dist_sep_id, left child has
   * id 2*dist_sep_id, right child 2*dist_sep_id+1.  Nodes for which
   * P<=dist_sep_id<2*P, are leafs of the distributed separator tree,
   * and they form the roots of the local subtree for process
   * dist_sep_id-P.
   *
   * Furthermore, each proces has a local tree rooted at one of the
   * leafs of the distributed tree.  To have the same convention as used
   * for PTScotch, the leafs are assigned to procs as in a postordering
   * of the distributed tree, hence for this example, proc_dist_sep:
   *                   3
   *                  / \
   *                 /   \
   *                1     2
   *               / \   / \
   *              0   2 3   4
   *             / \
   *            0   1
   */
  template<typename scalar_t,typename integer_t>
  class MatrixReorderingMPI : public MatrixReordering<scalar_t,integer_t> {
    using Opts_t = SPOptions<scalar_t>;
    using CSRMPI_t = CSRMatrixMPI<scalar_t,integer_t>;
    using CSM_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;

  public:
    MatrixReorderingMPI(integer_t n, const MPIComm& c);
    virtual ~MatrixReorderingMPI();

    int nested_dissection
    (const Opts_t& opts, const CSRMPI_t& A,
     int nx, int ny, int nz, int components, int width);

    int set_permutation
    (const Opts_t& opts, const CSRMPI_t& A, const int* p, int base);

    void separator_reordering(const Opts_t& opts, CSM_t& A, F_t* F);

    void clear_tree_data() override;

    /**
     * proc_dist_sep[sep] holds the rank of the process responsible
     * for distributed separator sep
     * - if sep is a leaf of the distributed tree, proc_dist_sep
     *    points to the process holding that subgraph as my_sub_graph
     * - if sep is a non-leaf, proc_dist_sep points to the process
     *    holding the graph of the distributed separator sep as
     *    my_dist_sep
     */
    std::vector<integer_t> proc_dist_sep;

    /**
     * Every process is responsible for one local subgraph of A.  The
     * distributed nested dissection will create a separator tree with
     * exactly P leafs.  Each process takes one of those leafs and
     * stores the corresponding part of the permuted matrix A in
     * sub_graph_A.
     */
    CSRGraph<integer_t> my_sub_graph;

    /**
     * Every process is responsible for one separator from the
     * distributed part of nested dissection.  The graph of the
     * distributed separator for which this rank is responsible is
     * stored as dist_sep_A.
     */
    CSRGraph<integer_t> my_dist_sep;

    std::vector<std::pair<integer_t,integer_t>> sub_graph_ranges;
    std::vector<std::pair<integer_t,integer_t>> dist_sep_ranges;

    std::pair<integer_t,integer_t> sub_graph_range;
    std::pair<integer_t,integer_t> dist_sep_range;

    /**
     * Number of the node in sep_tree corresponding to the root of the
     * local subtree.
     */
    integer_t dsep_leaf;

    const SeparatorTree<integer_t>& local_tree() const { return *local_tree_; }

  private:
    const MPIComm* comm_;
    std::unique_ptr<SeparatorTree<integer_t>> local_tree_;

    void get_local_graphs(const CSRMPI_t& Ampi);

    void build_local_tree(const CSRMPI_t& Ampi);

    void nested_dissection_print
    (const SPOptions<scalar_t>& opts, integer_t nnz) const;

    using MatrixReordering<scalar_t,integer_t>::perm_;
    using MatrixReordering<scalar_t,integer_t>::iperm_;
    using MatrixReordering<scalar_t,integer_t>::sep_tree_;
  };

} // end namespace strumpack

#endif // MATRIX_REORDERING_MPI_HPP
