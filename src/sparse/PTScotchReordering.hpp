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
#ifndef PT_SCOTCH_REORDERING_HPP
#define PT_SCOTCH_REORDERING_HPP

#include <ptscotch.h>
#include <cassert>
#include <type_traits>
#include "CSRMatrixMPI.hpp"

namespace strumpack {

  template<typename integer_t> inline int WRAPPER_SCOTCH_dgraphOrderPerm
  (SCOTCH_Dgraph* graph, SCOTCH_Dordering* ordeptr,
   std::vector<integer_t>& order) {
    std::vector<SCOTCH_Num> permloctab(order.size());
    int ierr = SCOTCH_dgraphOrderPerm(graph, ordeptr, permloctab.data());
    if (ierr) return ierr;
    order.assign(permloctab.begin(), permloctab.end());
    return 0;
  }
  template<> inline int WRAPPER_SCOTCH_dgraphOrderPerm
  (SCOTCH_Dgraph* graph, SCOTCH_Dordering* ordeptr,
   std::vector<SCOTCH_Num>& order) {
    return SCOTCH_dgraphOrderPerm(graph, ordeptr, order.data());
  }

  inline std::string get_ptscotch_strategy_string(int stratpar) {
    std::stringstream strategy_string;
    // switch (stratnum) {
    // case 1:  // based on a string from MUMPS, only performs nested-dissection for log(P) levels
    //   strategy_string <<
    //  "n{"           // nested dissection
    //  "ole=s,"    // "simple" parallel ordering strategy for each distributed leaf of parallel separator tree
    //  "ose=s,"    // "simple" parallel ordering strategy for each distributed separator of the separator tree
    //  "osq=s,"    // "simple" sequential ordering on centralized subgraphs, after n-d has gone far enough that subgraph is on 1 process
    //  "sep=m{"    // use parallel vertex multi-level method to find new separators
    //  "asc=b{width=3,strat=q{strat=f}},"   // use band method to refine distr vert seps after uncoarsening:
    //  "low=q{strat=h},"                    // use multi-sequential method to compute sep of coarsest graph
    //  "vert=1000,"                         // threshold minimum size under which graph is no longer coarsened
    //  "dvert=100,"                         // avg number of vert for proc under which the folding process is performed during coarsening
    //  "dlevl=0"                            // minimum level after which duplication is allowed in folding process
    //  "}"
    //  "}"; break;
    // case 2:

    // default:  // perform nested dissection all the way down
    //   // TODO stop when separator smaller than stratpar!
    strategy_string <<
      "n{"           // nested dissection
      "ole=s,"    // "simple" parallel ordering strategy for each distributed leaf of parallel separator tree
      "ose=s,"    // "simple" parallel ordering strategy for each distributed separator of the separator tree
      "osq=n{ole=s,ose=s,sep=g},"    // nested-dissection ordering on centralized subgraphs with Gibbs-Poole-Stockmeyer to find seps
      "sep=m{"                       // use parallel vertex multi-level method to find new separators
      "asc=b{width=3,strat=q{strat=f}}," // use band method to refine distr vert seps after uncoarsening:
      "low=q{strat=h},"                  // use multi-sequential method to compute sep of coarsest graph
      "vert=1000,"                       // threshold minimum size under which graph is no longer coarsened
      "dvert=100,"                       // avg number of vert for proc under which the folding process is performed during coarsening
      "dlevl=0"                          // minimum level after which duplication is allowed in folding process
      "}"
      "}";
    // break;
    // }
    return strategy_string.str();
  }

  /**
   * Take a ptscotch tree for the distributed part of nested-dissection
   * and make a regular binary tree.
   *
   * The ptscotch_tree layout:
   *   - ptscotch_tree[i] is parent of node i
   *   - the nodes with count 3 are nested-dissection nodes
   *     children are: (left_child, right_child, separator)
   *   - nodes with 2 children are nested dissection nodes with empty separator
   *   - ptscotch_sizes denote size of part plus all descendants
   *
   * return: the separator tree
   */
  template<typename integer_t>
  std::unique_ptr<SeparatorTree<integer_t>> sep_tree_from_ptscotch_nd_tree
  (std::vector<SCOTCH_Num>& ptscotch_tree,
   std::vector<SCOTCH_Num>& ptscotch_sizes) {
    assert(std::count(ptscotch_tree.begin(), ptscotch_tree.end(), -1) == 1);
    std::vector<integer_t> count(ptscotch_tree.size(), 0);
    std::vector<integer_t> ptscotch_lchild
      (ptscotch_tree.size(), integer_t(-1));
    std::vector<integer_t> ptscotch_rchild
      (ptscotch_tree.size(), integer_t(-1));
    integer_t nr_sep = 0;
    for (size_t i=0; i<ptscotch_tree.size(); i++) {
      integer_t p = ptscotch_tree[i];
      nr_sep++;
      if (p != -1) {
        count[p]++;
        switch (count[p]) {
        case 1: ptscotch_lchild[p] = i; break;
        case 2: ptscotch_rchild[p] = i; break;
        default: nr_sep--;
        }
      }
    }
    integer_t root_id = std::distance
      (ptscotch_tree.begin(),
       std::find(ptscotch_tree.begin(),
                 ptscotch_tree.end(), integer_t(-1)));
    auto sep_tree = std::unique_ptr<SeparatorTree<integer_t>>
      (new SeparatorTree<integer_t>(nr_sep));
    sep_tree->sizes(0) = 0;
    for (integer_t i=0; i<nr_sep; i++) {
      sep_tree->pa(i) = -1;
      sep_tree->lch(i) = -1;
      sep_tree->rch(i) = -1;
    }
    std::function<void(integer_t,integer_t&)> f =
      [&](integer_t i, integer_t& pid) {
      if (ptscotch_lchild[i] != -1 && ptscotch_rchild[i] != -1) {
        f(ptscotch_lchild[i], pid);
        integer_t left_root_id = pid - 1;
        f(ptscotch_rchild[i], pid);
        sep_tree->rch(pid) = pid - 1;
        sep_tree->pa(sep_tree->rch(pid)) = pid;
        sep_tree->lch(pid) = left_root_id;
        sep_tree->pa(sep_tree->lch(pid)) = pid;
      }
      auto size_pid = ptscotch_sizes[i];
      if (ptscotch_lchild[i] != -1 && ptscotch_rchild[i] != -1)
        size_pid -= ptscotch_sizes[ptscotch_lchild[i]]
          + ptscotch_sizes[ptscotch_rchild[i]];
      sep_tree->sizes(pid+1) = sep_tree->sizes(pid) + size_pid;
      pid++;
    };
    nr_sep = 0;
    f(root_id, nr_sep);
    return sep_tree;
  }

  template<typename scalar_t,typename integer_t>
  std::unique_ptr<SeparatorTree<integer_t>> ptscotch_nested_dissection
  (const CSRMatrixMPI<scalar_t,integer_t>& A, MPI_Comm comm, bool build_tree,
   std::vector<integer_t>& perm, const SPOptions<scalar_t>& opts) {
    auto local_rows = A.local_rows();
    auto ptr = A.ptr();
    auto ind = A.ind();
    std::vector<SCOTCH_Num> ptr_nodiag(local_rows+1),
      ind_nodiag(ptr[local_rows]);
    integer_t nnz_nodiag = 0;
    ptr_nodiag[0] = 0;
    for (integer_t i=0; i<local_rows; i++) { // remove diagonal elements
      for (integer_t j=ptr[i]; j<ptr[i+1]; j++)
        if (ind[j]!=i+A.begin_row()) ind_nodiag[nnz_nodiag++] = ind[j];
      ptr_nodiag[i+1] = nnz_nodiag;
    } // ptr_nodiag will now point locally, ind_nodiag still has
      // global column/row indices

    SCOTCH_Dgraph graph;
    int ierr = SCOTCH_dgraphInit(&graph, comm);
    if (ierr)
      std::cerr << "# ERROR: PTScotch failed to initialize the graph."
                << std::endl;
    ierr = SCOTCH_dgraphBuild
      (&graph, 0, local_rows, local_rows, ptr_nodiag.data(),
       &ptr_nodiag[1], NULL, NULL, nnz_nodiag, nnz_nodiag,
       ind_nodiag.data(), NULL, NULL);
    if (ierr)
      std::cerr << "# ERROR: PTScotch failed to build the graph."
                << std::endl;
    assert(SCOTCH_dgraphCheck(&graph) == 0);
    SCOTCH_Strat strategy;
    ierr = SCOTCH_stratInit(&strategy);
    if (ierr)
      std::cerr << "# ERROR: PTScotch failed to initialize the strategy."
                << std::endl;
    ierr = SCOTCH_stratDgraphOrder
      (&strategy, get_ptscotch_strategy_string(opts.nd_param()).c_str());
    if (ierr)
      std::cerr << "# ERROR: PTScotch failed to create the reordering strategy."
                << std::endl;
    SCOTCH_Dordering ordeptr;
    ierr = SCOTCH_dgraphOrderInit(&graph, &ordeptr);
    if (ierr)
      std::cerr << "# ERROR: PTScotch failed to initialize the reordering."
                << std::endl;
    ierr = SCOTCH_dgraphOrderCompute(&graph, &ordeptr, &strategy);
    if (ierr)
      std::cerr << "# ERROR: PTScotch failed to compute the reordering."
                << std::endl;
    std::vector<integer_t> local_order(local_rows);
    ierr = WRAPPER_SCOTCH_dgraphOrderPerm
      (&graph, &ordeptr, local_order);
    if (ierr)
      std::cerr << "# ERROR: PTScotch failed to retrieve the reordering."
                << std::endl;

    auto P = mpi_nprocs(comm);
    {
      std::unique_ptr<int[]> rcnts(new int[2*P]);
      auto displs = rcnts.get() + P;
      for (int p=0; p<P; p++) {
        rcnts[p] = A.dist(p+1) - A.dist(p);
        // need to copy because displs needs to be 'int'
        displs[p] = A.dist(p);
      }
      MPI_Allgatherv
        (local_order.data(), local_rows, mpi_type<integer_t>(),
         perm.data(), rcnts.get(), displs, mpi_type<integer_t>(), comm);
    }

    std::unique_ptr<SeparatorTree<integer_t>> sep_tree;
    if (build_tree) {
      // get info about the distributed levels of nested-dissection
      integer_t dist_nr_sep = SCOTCH_dgraphOrderCblkDist(&graph, &ordeptr);
      if (ierr)
        std::cerr << "# ERROR: PTScotch failed to build the separator tree."
                  << std::endl;
      std::vector<SCOTCH_Num> ptscotch_dist_tree(dist_nr_sep);
      std::vector<SCOTCH_Num> ptscotch_dist_sizes(dist_nr_sep);
      ierr = SCOTCH_dgraphOrderTreeDist
        (&graph, &ordeptr, ptscotch_dist_tree.data(),
         ptscotch_dist_sizes.data());
      if (ierr)
        std::cerr << "# ERROR: PTScotch failed to build the separator tree."
                  << std::endl;
      // build separator tree for distributed nested dissection part
      sep_tree = sep_tree_from_ptscotch_nd_tree<integer_t>
        (ptscotch_dist_tree, ptscotch_dist_sizes);
    }
    SCOTCH_dgraphOrderExit(&graph, &ordeptr);
    SCOTCH_dgraphExit(&graph);
    SCOTCH_stratExit(&strategy);
    return sep_tree;
  }

} // end namespace strumpack

#endif
