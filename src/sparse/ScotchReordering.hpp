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
#ifndef SCOTCH_REORDERING_HPP
#define SCOTCH_REORDERING_HPP

#include <memory>
#include <scotch.h>
#include "SeparatorTree.hpp"
#include "ETree.hpp"

namespace strumpack {

  // base template, integer_t != SCOTCH_Num, so need to copy everything
  template<typename integer_t> inline int WRAPPER_SCOTCH_graphOrder
  (integer_t n, SCOTCH_Graph* graph, SCOTCH_Strat* strat,
   std::vector<integer_t>& permtab, std::vector<integer_t>& peritab,
   integer_t& cblkptr, std::vector<SCOTCH_Num>& rangtab,
   std::vector<SCOTCH_Num>& treetab) {
    SCOTCH_Num scotch_cblkptr = cblkptr;
    std::vector<SCOTCH_Num> scotch_permtab(n), scotch_peritab(n);
    int ierr = SCOTCH_graphOrder
      (graph, strat, scotch_permtab.data(), scotch_peritab.data(),
       &scotch_cblkptr, rangtab.data(), treetab.data());
    if (ierr) return ierr;
    permtab.assign(scotch_permtab.begin(), scotch_permtab.end());
    peritab.assign(scotch_peritab.begin(), scotch_peritab.end());
    cblkptr = scotch_cblkptr;
    return 0;
  }
  // if integer_t == SCOTCH_Num, no need to copy
  template<> inline int WRAPPER_SCOTCH_graphOrder
  (SCOTCH_Num n, SCOTCH_Graph* graph, SCOTCH_Strat* strat,
   std::vector<SCOTCH_Num>& permtab, std::vector<SCOTCH_Num>& peritab,
   SCOTCH_Num& cblkptr, std::vector<SCOTCH_Num>& rangtab,
   std::vector<SCOTCH_Num>& treetab) {
    return SCOTCH_graphOrder
      (graph, strat, permtab.data(), peritab.data(), &cblkptr,
       rangtab.data(), treetab.data());
  }

  inline std::string get_scotch_strategy_string(int stratpar) {
    std::stringstream strategy_string;
    // switch (stratnum) {
    // case 1: // nested dissection with STRATPAR levels
    //   strategy_string << "n{ole=s,ose=s,sep=(/levl>"
    //                   << int(stratpar-1) << "?z:"
    //     "(m{asc=b{bnd=f{move=200,pass=1000,bal=0.9},"
    //     "org=(|h{pass=10})f{move=200,pass=1000,bal=0.9},width=3},"
    //     "low=h{pass=10},type=h,vert=100,rat=0.7});)}";
    //   break;
    // case 2: // nested dissection with separators <= STRATPAR
    strategy_string << "n{ole=s,ose=s,sep=(/vert<" << int(stratpar) << "?z:"
      "(m{asc=b{bnd=f{move=200,pass=1000,bal=0.9},org=(|h{pass=10})"
      "f{move=200,pass=1000,bal=0.9},width=3},"
      "low=h{pass=10},type=h,vert=100,rat=0.7});)}";
    // break;
    // default: // Pure nested dissection
    // strategy_string
    //   << "n{ole=s,ose=s,sep=m{asc=b{bnd=f{move=200,pass=1000,bal=0.9},"
    //   "org=(|h{pass=10})f{move=200,pass=1000,bal=0.9},width=3},"
    //   "low=h{pass=10},type=h,vert=100,rat=0.7}}";
    // }
    return strategy_string.str();
  }

  template<typename integer_t> std::unique_ptr<SeparatorTree<integer_t>>
  sep_tree_from_scotch_tree(std::vector<SCOTCH_Num>& scotch_tree,
                            std::vector<SCOTCH_Num>& scotch_sizes) {
    // if the graph is disconnected, add one root to connect all
    if (std::count(scotch_tree.begin(), scotch_tree.end(), -1) > 1) {
      SCOTCH_Num root_id =
        *std::max_element(scotch_tree.begin(), scotch_tree.end()) + 1;
      scotch_sizes.push_back(scotch_sizes.back());
      std::replace(scotch_tree.begin(), scotch_tree.end(),
                   SCOTCH_Num(-1), root_id);
      scotch_tree.push_back(SCOTCH_Num(-1));
    }
    integer_t nbsep = scotch_tree.size();
    std::vector<integer_t> count(nbsep, 0),
      scotch_lch(nbsep, integer_t(-1)),
      scotch_rch(nbsep, integer_t(-1));
    for (integer_t i=0; i<nbsep; i++) {
      integer_t p = scotch_tree[i];
      if (p != -1) {
        count[p]++;
        switch (count[p]) {
        case 1: // node i is the first child of node p
          scotch_lch[p] = i;
          break;
        case 2: // node i is the second child of node p
          scotch_rch[p] = i;
          break;
        case 3:
          // node i is the third child of node p
          // make a new (empty) node with children the first two children of p
          //   set this new node as the left child of p
          // make node i the right child of p
          integer_t max_p = scotch_tree.size();
          scotch_tree.push_back(max_p);
          scotch_lch.push_back(scotch_lch[p]);
          scotch_rch.push_back(scotch_rch[p]);
          scotch_lch[p] = max_p;
          scotch_rch[p] = i;
          count[p]--;
          break;
        }
      }
    }
    integer_t nbsep_non_empty = nbsep;
    nbsep = scotch_tree.size();
    std::unique_ptr<SeparatorTree<integer_t>>
      sep_tree(new SeparatorTree<integer_t>(nbsep));
    for (integer_t i=0; i<nbsep+1; i++) {
      if (i <= nbsep_non_empty) sep_tree->sizes(i) = scotch_sizes[i];
      else sep_tree->sizes(i) = sep_tree->sizes(i-1);
    }
    for (integer_t i=0; i<nbsep; i++) {
      sep_tree->lch(i) = scotch_lch[i];
      sep_tree->rch(i) = scotch_rch[i];
      if (sep_tree->lch(i)!=-1) sep_tree->pa(sep_tree->lch(i)) = i;
      if (sep_tree->rch(i)!=-1) sep_tree->pa(sep_tree->rch(i)) = i;
    }
    integer_t root =
      std::distance(scotch_tree.begin(),
                    std::find(scotch_tree.begin(),
                              scotch_tree.end()+nbsep, integer_t(-1)));
    sep_tree->pa(root) = -1;
    return sep_tree;
  }

  // TODO throw exception on error
  template<typename scalar_t,typename integer_t>
  std::unique_ptr<SeparatorTree<integer_t>>
  scotch_nested_dissection
  (const CompressedSparseMatrix<scalar_t,integer_t>& A,
   std::vector<integer_t>& perm, std::vector<integer_t>& iperm,
   const SPOptions<scalar_t>& opts) {
    auto n = A.size();
    auto ptr = A.ptr();
    auto ind = A.ind();
    SCOTCH_Graph graph;
    SCOTCH_graphInit(&graph);
    std::vector<SCOTCH_Num> ptr_nodiag(n+1), ind_nodiag(ptr[n]-ptr[0]);
    integer_t nnz_nodiag = 0;
    ptr_nodiag[0] = 0;
    for (integer_t i=0; i<n; i++) { // remove diagonal elements
      for (integer_t j=ptr[i]; j<ptr[i+1]; j++)
        if (ind[j]!=i) ind_nodiag[nnz_nodiag++] = ind[j];
      ptr_nodiag[i+1] = nnz_nodiag;
    }
    if (nnz_nodiag==0)
      std::cerr << "# WARNING: matrix seems to be diagonal!" << std::endl;
    int ierr = SCOTCH_graphBuild
      (&graph, 0, n, ptr_nodiag.data(), NULL, NULL, NULL,
       ptr_nodiag[n], ind_nodiag.data(), NULL);
    if (ierr)
      std::cerr << "# ERROR: Scotch failed to build graph." << std::endl;
    assert(SCOTCH_graphCheck(&graph) == 0);

    SCOTCH_Strat strategy;
    ierr = SCOTCH_stratInit(&strategy);
    if (ierr)
      std::cerr << "# ERROR: Scotch failed to initialize strategy."
                << std::endl;
    ierr = SCOTCH_stratGraphOrder
      (&strategy, get_scotch_strategy_string(opts.nd_param()).c_str());
    if (ierr)
      std::cerr << "# ERROR: Scotch failed to compute ordering." << std::endl;

    std::vector<SCOTCH_Num> scotch_sizes(n+1), scotch_tree(n);
    integer_t nbsep = 0;
    ierr = WRAPPER_SCOTCH_graphOrder<integer_t>
      (n, &graph, &strategy, perm, iperm, nbsep, scotch_sizes, scotch_tree);
    if (ierr)
      std::cerr << "# ERROR: Scotch failed to compute ordering." << std::endl;

    SCOTCH_graphExit(&graph);
    SCOTCH_stratExit(&strategy);

    scotch_tree.resize(nbsep);
    scotch_sizes.resize(nbsep+1);
    return sep_tree_from_scotch_tree<integer_t>(scotch_tree, scotch_sizes);
  }

} // end namespace strumpack

#endif
