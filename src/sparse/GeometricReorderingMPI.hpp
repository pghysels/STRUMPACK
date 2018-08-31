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
#ifndef GEOMETRIC_REORDERING_MPI_HPP
#define GEOMETRIC_REORDERING_MPI_HPP

#include <array>
#include "misc/MPIWrapper.hpp"
#include "MatrixReordering.hpp"

namespace strumpack {

  template<typename integer_t>
  std::pair<std::unique_ptr<SeparatorTree<integer_t>>,
            std::unique_ptr<SeparatorTree<integer_t>>>
  geometric_nested_dissection_dist
  (int nx, int ny, int nz, int components, int width,
   integer_t lo, integer_t hi, MPI_Comm comm,
   std::vector<integer_t>& perm, std::vector<integer_t>& iperm,
   int nd_param, int HSS_leaf, int min_HSS) {
    assert(components == 1);
    assert(width == 1);
    auto P = mpi_nprocs(comm);
    auto rank = mpi_rank(comm);
    integer_t dist_nbsep = 0, local_nbsep = 0;
    std::vector<Separator<integer_t>> dist_tree, local_tree;
    integer_t perm_begin = 0, dsep_leaf_id = 0;
    std::array<integer_t,3> ld = {{nx, ny, nz}};
    std::unordered_map<integer_t,HSS::HSSPartitionTree> dist_HSS_trees;
    std::unordered_map<integer_t,HSS::HSSPartitionTree> local_HSS_trees;

    std::function<
      void(std::array<integer_t,3>,std::array<integer_t,3>,integer_t)>
      rec_nd = [&](std::array<integer_t,3> n0, std::array<integer_t,3> dims,
                   integer_t dsep_id) {
      auto N = components * dims[0]*dims[1]*dims[2];
      // d: dimension along which to split
      int d = std::distance
      (dims.begin(), std::max_element(dims.begin(), dims.end()));
      bool dsep = dsep_id < 2*P;
      bool dsep_leaf = dsep && dsep_id >= P;
      bool is_local = dsep_id >= P && dsep_leaf_id == rank;

      if (dims[d] < 2+width || N <= nd_param) {
        for (integer_t z=n0[2]; z<n0[2]+dims[2]; z++)
          for (integer_t y=n0[1]; y<n0[1]+dims[1]; y++)
            for (integer_t x=n0[0]; x<n0[0]+dims[0]; x++) {
              auto ind = components * (x + y*ld[0] + z*ld[0]*ld[1]);
              for (int c=0; c<components; c++) {
                perm[ind] = perm_begin;
                iperm[perm_begin] = ind;
                perm_begin++;
                ind++;
              }
            }
        if (dsep) {
          if (dist_nbsep)
            dist_tree.emplace_back(dist_tree.back().sep_end + N, -1, -1, -1);
          else dist_tree.emplace_back(N, -1, -1, -1);
          dist_nbsep++;
        }
        if (is_local) {
          if (local_nbsep)
            local_tree.emplace_back
              (local_tree.back().sep_end + N, -1, -1, -1);
          else local_tree.emplace_back(N, -1, -1, -1);
          local_nbsep++;
        }
      } else {
        // part 1/left
        std::array<integer_t,3> part_begin(n0);
        std::array<integer_t,3> part_size(dims);
        part_size[d] = dims[d]/2 - (width/2);
        rec_nd(part_begin, part_size, 2*dsep_id);
        auto dist_left_root_id = dist_nbsep - 1;
        auto local_left_root_id = local_nbsep - 1;

        // part 2/right
        part_begin[d] = n0[d] + dims[d]/2 + 1;
        part_size[d] = dims[d] - width - dims[d]/2;
        rec_nd(part_begin, part_size, 2*dsep_id+1);
        if (dsep && !dsep_leaf) {
          dist_tree[dist_left_root_id].pa = dist_nbsep;
          dist_tree[dist_nbsep-1].pa = dist_nbsep;
        }
        if (is_local) {
          local_tree[local_left_root_id].pa = local_nbsep;
          local_tree[local_nbsep-1].pa = local_nbsep;
        }

        // separator
        part_begin[d] = n0[d] + dims[d]/2  - (width/2);
        part_size[d] = width;
        auto sep_size = components * part_size[0]*part_size[1]*part_size[2];
        if (sep_size >= min_HSS) {
          HSS::HSSPartitionTree t;
          recursive_bisection
            (perm.data(), iperm.data(), perm_begin,
             part_begin, part_size, ld, components, HSS_leaf, t);
          if (is_local) local_HSS_trees[local_nbsep] = t; // Not thread safe!!
          else dist_HSS_trees[dist_nbsep] = t; // Not thread safe!!
        } else {
          for (integer_t z=part_begin[2]; z<part_begin[2]+part_size[2]; z++)
            for (integer_t y=part_begin[1]; y<part_begin[1]+part_size[1]; y++)
              for (integer_t x=part_begin[0]; x<part_begin[0]+part_size[0]; x++) {
                auto ind = components * (x + y*ld[0] + z*ld[0]*ld[1]);
                for (int c=0; c<components; c++) {
                  perm[ind] = perm_begin;
                  iperm[perm_begin] = ind;
                  perm_begin++;
                  ind++;
                }
              }
        }

        if (dsep) {
          if (dsep_leaf) {
            if (dist_nbsep)
              dist_tree.emplace_back
                (dist_tree.back().sep_end + N, -1, -1, -1);
            else dist_tree.emplace_back
                   (N, -1, dist_left_root_id, dist_nbsep-1);
          } else {
            if (dist_nbsep)
              dist_tree.emplace_back
                (dist_tree.back().sep_end + sep_size, -1,
                 dist_left_root_id, dist_nbsep-1);
            else
              dist_tree.emplace_back
                (sep_size, -1, dist_left_root_id, dist_nbsep-1);
          }
          dist_nbsep++;
        }
        if (is_local) {
          if (local_nbsep)
            local_tree.emplace_back
              (local_tree.back().sep_end + sep_size, -1,
               local_left_root_id, local_nbsep-1);
          else local_tree.emplace_back
                 (sep_size, -1, local_left_root_id, local_nbsep-1);
          local_nbsep++;
        }
      }
      if (dsep && dsep_leaf) dsep_leaf_id++;
    };


    // TODO tree only when HSS enabled .. cf seq geometric ordering
    // use blr leaf size if BLR is chosen iso HSS

    rec_nd({{0, 0, 0}}, {{nx, ny, nz}}, 1);
    std::unique_ptr<SeparatorTree<integer_t>> local_stree
      (new SeparatorTree<integer_t>(local_tree));
    std::unique_ptr<SeparatorTree<integer_t>> dist_stree
      (new SeparatorTree<integer_t>(dist_tree));
    local_stree->HSS_trees() = local_HSS_trees;
    dist_stree->HSS_trees() = dist_HSS_trees;
    return std::make_pair
      <std::unique_ptr<SeparatorTree<integer_t>>,
       std::unique_ptr<SeparatorTree<integer_t>>>
      (std::move(dist_stree), std::move(local_stree));
  }

} // end namespace strumpack

#endif
