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
#ifndef PARMETIS_REORDERING_HPP
#define PARMETIS_REORDERING_HPP

#include <typeinfo>
#include "parmetis.h"

namespace strumpack {

  template<typename integer_t> inline int WRAPPER_ParMETIS_V32_NodeND
  (const std::vector<integer_t>& dist, std::vector<idx_t>& xadj,
   std::vector<idx_t>& adjncy, idx_t numflag,
   std::vector<integer_t>& local_order, std::vector<idx_t>& sizes,
   MPI_Comm c) {
    std::vector<idx_t> vtxdist(dist.size()), loc_order(local_order.size());
    vtxdist.assign(dist.begin(), dist.end());
    int ierr = ParMETIS_V32_NodeND
      (vtxdist.data(), xadj.data(), adjncy.data(), NULL,
       &numflag, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
       loc_order.data(), sizes.data(), &c);
    local_order.assign(loc_order.begin(), loc_order.end());
    return ierr;
  }
  template<> inline int WRAPPER_ParMETIS_V32_NodeND
  (const std::vector<idx_t>& dist, std::vector<idx_t>& xadj,
   std::vector<idx_t>& adjncy, idx_t numflag,
   std::vector<idx_t>& local_order, std::vector<idx_t>& sizes,
   MPI_Comm c) {
    return ParMETIS_V32_NodeND
      (const_cast<idx_t*>(dist.data()), xadj.data(), adjncy.data(),
       NULL, &numflag, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
       local_order.data(), sizes.data(), &c);
  }

  template<typename scalar_t,typename integer_t>
  std::unique_ptr<SeparatorTree<integer_t>>
  parmetis_nested_dissection
  (const CSRMatrixMPI<scalar_t,integer_t>& A, MPI_Comm comm,
   bool build_tree, std::vector<integer_t>& perm,
   const SPOptions<scalar_t>& opts) {
    auto P = mpi_nprocs(comm);
    auto local_rows = A.local_rows();
    auto ptr = A.ptr();
    auto ind = A.ind();
    std::vector<idx_t> xadj(local_rows+1), adjncy(A.local_nnz());
    xadj[0] = 0;
    integer_t e = 0;
    for (integer_t i=0; i<local_rows; i++) { // remove diagonal elements
      for (integer_t j=ptr[i]; j<ptr[i+1]; j++)
        if (ind[j]!=i+A.begin_row()) adjncy[e++] = ind[j];
      xadj[i+1] = e;
    } // xadj will now point locally, adjncy still has global
      // column/row indices
    idx_t numflag = 0;
    int p2 = std::pow(2, std::floor(std::log2(P)));
    std::vector<idx_t> sizes(2*p2-1);

    {
      std::vector<integer_t> local_order(local_rows);
      int ierr = WRAPPER_ParMETIS_V32_NodeND
        (A.dist(), xadj, adjncy, numflag,
         local_order, sizes, comm);
      if (ierr != METIS_OK) {
        if (!mpi_rank(comm))
          std::cerr << "ParMETIS_V32_NodeND failed with ierr="
                    << ierr << std::endl;
        return nullptr; // TODO throw an exception
      }

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
      // p2 subdomains, p2-1 distributed separators
      integer_t dist_nr_sep = 2*p2 - 1;
      sep_tree = std::unique_ptr<SeparatorTree<integer_t>>
        (new SeparatorTree<integer_t>(dist_nr_sep));
      sep_tree->sizes(0) = 0;
      for (integer_t i=0; i<dist_nr_sep; i++) {
        sep_tree->pa(i) = -1;
        sep_tree->lch(i) = -1;
        sep_tree->rch(i) = -1;
      }
      int nr_dist_levels = std::log2(p2);
      std::function<void(integer_t,integer_t&,integer_t)>
        build_dist_binary_separator_tree =
        [&](integer_t dsep, integer_t& pid, integer_t level) {
        if (level > 0) {
          build_dist_binary_separator_tree(2*dsep+2, pid, level-1);
          integer_t lch = pid - 1;
          build_dist_binary_separator_tree(2*dsep+1, pid, level-1);
          sep_tree->lch(pid) = lch;
          sep_tree->rch(pid) = pid-1;
          sep_tree->pa(lch) = pid;
          sep_tree->pa(pid-1) = pid;
        }
        sep_tree->sizes(pid+1) = sizes[dist_nr_sep-1-dsep]
        + sep_tree->sizes(pid);
        pid++;
      };
      integer_t pid = 0;
      build_dist_binary_separator_tree(0, pid, nr_dist_levels);
    }
    return sep_tree;
  }

} // end namespace strumpack

#endif
