/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The Regents of
 * the University of California, through Lawrence Berkeley National Laboratory
 * (subject to receipt of any required approvals from the U.S. Dept. of Energy).
 * All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Technology Transfer Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, prepare derivative works, and perform publicly and display publicly.
 * Beginning five (5) years after the date permission to assert copyright is
 * obtained from the U.S. Department of Energy, and subject to any subsequent five
 * (5) year renewals, the U.S. Government is granted for itself and others acting
 * on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, prepare derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 *
 */
#ifndef GEOMETRIC_REORDERING_HPP
#define GEOMETRIC_REORDERING_HPP

#include "MatrixReordering.hpp"
#include <array>

namespace strumpack {

  template<typename integer_t> void recursive_nested_dissection
  (integer_t* perm, integer_t* iperm, integer_t& perm_begin, integer_t& nbsep,
   std::array<integer_t,3> n0, std::array<integer_t,3> dims, std::array<integer_t,3> ld,
   std::vector<Separator<integer_t>>& tree, int stratpar) {
    integer_t N = dims[0]*dims[1]*dims[2];
    // d: dimension along which to split
    int d = std::distance(dims.begin(), std::max_element(dims.begin(), dims.end()));

    if (dims[d] < 3 || N <= stratpar) {
      for (integer_t z=n0[2]; z<n0[2]+dims[2]; z++)
	for (integer_t y=n0[1]; y<n0[1]+dims[1]; y++)
	  for (integer_t x=n0[0]; x<n0[0]+dims[0]; x++) {
	    auto ind = x + y*ld[0] + z*ld[0]*ld[1];
	    perm[ind] = perm_begin;
	    iperm[perm_begin] = ind;
	    perm_begin++;
	  }
      if (nbsep) tree.emplace_back(tree.back().sep_end + N, -1, -1, -1);
      else tree.emplace_back(N, -1, -1, -1);
      nbsep++;
    } else {
      // part 1
      std::array<integer_t,3> part_begin(n0);
      std::array<integer_t,3> part_size(dims);
      part_size[d] = dims[d]/2;
      recursive_nested_dissection(perm, iperm, perm_begin, nbsep, part_begin, part_size, ld, tree, stratpar);
      auto left_root_id = nbsep - 1;

      // part 2
      part_begin[d] = n0[d] + dims[d]/2 + 1;
      part_size[d] = dims[d] - 1 - dims[d]/2;
      recursive_nested_dissection(perm, iperm, perm_begin, nbsep, part_begin, part_size, ld, tree, stratpar);
      tree[left_root_id].pa = nbsep;
      tree[nbsep-1].pa = nbsep;

      // separator
      part_begin[d] = n0[d] + dims[d]/2;
      part_size[d] = 1;
      for (integer_t z=part_begin[2]; z<part_begin[2]+part_size[2]; z++)
	for (integer_t y=part_begin[1]; y<part_begin[1]+part_size[1]; y++)
	  for (integer_t x=part_begin[0]; x<part_begin[0]+part_size[0]; x++) {
	    auto ind = x + y*ld[0] + z*ld[0]*ld[1];
	    perm[ind] = perm_begin;
	    iperm[perm_begin] = ind;
	    perm_begin++;
	  }
      if (nbsep) tree.emplace_back(tree.back().sep_end + N/dims[d], -1, left_root_id, nbsep-1);
      else tree.emplace_back(N/dims[d], -1, left_root_id, nbsep-1);
      nbsep++;
    }
  }

  template<typename integer_t> std::unique_ptr<SeparatorTree<integer_t>>
  geometric_nested_dissection(int nx, int ny, int nz, integer_t* perm, integer_t* iperm, int nd_param) {
    std::vector<Separator<integer_t>> tree;
    integer_t nbsep = 0, perm_begin = 0;
    recursive_nested_dissection(perm, iperm, perm_begin, nbsep,
				{{0, 0, 0}}, {{nx, ny, nz}}, {{nx, ny, nz}}, tree, nd_param);
    return std::unique_ptr<SeparatorTree<integer_t>>(new SeparatorTree<integer_t>(tree));
  }

} // end namespace strumpack

#endif
