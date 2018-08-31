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
#ifndef GEOMETRIC_REORDERING_HPP
#define GEOMETRIC_REORDERING_HPP

#include "HSS/HSSPartitionTree.hpp"
#include "MatrixReordering.hpp"
#include <array>

namespace strumpack {

  template<typename integer_t> class GeomOrderData {
  public:
    integer_t* perm;
    integer_t* iperm;
    int components;
    int width;
    int stratpar;
    int leaf;
    int min_sep;
    bool separator_reordering;
    std::unordered_map<integer_t,HSS::HSSPartitionTree> trees;
  };

  template<typename integer_t>
  void recursive_bisection
  (integer_t* perm, integer_t* iperm, integer_t& pbegin,
   std::array<integer_t,3> n0, std::array<integer_t,3> dims,
   std::array<integer_t,3> ld, //const GeomOrderData<integer_t>& gd,
   int components, int leaf, HSS::HSSPartitionTree& hss_tree) {
    std::size_t sep_size = components * dims[0]*dims[1]*dims[2];
    hss_tree.size = sep_size;
    if (sep_size <= std::size_t(leaf)) {
      for (integer_t z=n0[2]; z<n0[2]+dims[2]; z++)
        for (integer_t y=n0[1]; y<n0[1]+dims[1]; y++)
          for (integer_t x=n0[0]; x<n0[0]+dims[0]; x++) {
            auto ind = components * (x + y*ld[0] + z*ld[0]*ld[1]);
            for (int c=0; c<components; c++) {
              perm[ind] = pbegin;
              iperm[pbegin++] = ind++;
            }
          }
    } else {
      hss_tree.c.resize(2);
      int d = std::distance
        (dims.begin(), std::max_element(dims.begin(), dims.end()));
      std::array<integer_t,3> part_begin(n0);
      std::array<integer_t,3> part_size(dims);
      part_size[d] = dims[d]/2;
      recursive_bisection
        (perm, iperm, pbegin, part_begin, part_size, ld,
         components, leaf, hss_tree.c[0]);
      part_begin[d] = n0[d] + dims[d]/2;
      part_size[d] = dims[d] - dims[d]/2;
      recursive_bisection
        (perm, iperm, pbegin, part_begin, part_size, ld,
         components, leaf, hss_tree.c[1]);
    }
  }

  template<typename integer_t>
  void recursive_nested_dissection
  (integer_t& pbegin, integer_t& nbsep,
   std::array<integer_t,3> n0, std::array<integer_t,3> dims,
   std::array<integer_t,3> ld, std::vector<Separator<integer_t>>& tree,
   GeomOrderData<integer_t>& gd) {
    int comps = gd.components;
    int width = gd.width;
    int stratpar = gd.stratpar;
    integer_t N = comps * (dims[0]*dims[1]*dims[2]);
    // d: dimension along which to split
    int d = std::distance
      (dims.begin(), std::max_element(dims.begin(), dims.end()));

    if (dims[d] < 2+width || N <= stratpar) {
      if (gd.separator_reordering && N >= gd.min_sep) {
        HSS::HSSPartitionTree hss_tree;
        recursive_bisection
          (gd.perm, gd.iperm, pbegin, n0, dims, ld,
           gd.components, gd.leaf, hss_tree);
        gd.trees[nbsep] = hss_tree; // Not thread safe!!
      } else {
        for (integer_t z=n0[2]; z<n0[2]+dims[2]; z++)
          for (integer_t y=n0[1]; y<n0[1]+dims[1]; y++)
            for (integer_t x=n0[0]; x<n0[0]+dims[0]; x++) {
              auto ind = comps * (x + y*ld[0] + z*ld[0]*ld[1]);
              for (int c=0; c<comps; c++) {
                gd.perm[ind] = pbegin;
                gd.iperm[pbegin++] = ind++;
              }
            }
      }
      if (nbsep) tree.emplace_back(tree.back().sep_end + N, -1, -1, -1);
      else tree.emplace_back(N, -1, -1, -1);
      nbsep++;
    } else {
      // part 1
      std::array<integer_t,3> part_begin(n0);
      std::array<integer_t,3> part_size(dims);
      part_size[d] = dims[d]/2 - (width/2);
      recursive_nested_dissection
        (pbegin, nbsep, part_begin, part_size, ld, tree, gd);
      auto left_root_id = nbsep - 1;

      // part 2
      part_begin[d] = n0[d] + dims[d]/2 + width;
      part_size[d] = dims[d] - width - dims[d]/2;
      recursive_nested_dissection
        (pbegin, nbsep, part_begin, part_size, ld, tree, gd);
      tree[left_root_id].pa = nbsep;
      tree[nbsep-1].pa = nbsep;

      // separator
      part_begin[d] = n0[d] + dims[d]/2  - (width/2);
      part_size[d] = width;
      integer_t sep_size = comps * part_size[0]*part_size[1]*part_size[2];
      if (gd.separator_reordering && sep_size >= gd.min_sep) {
        HSS::HSSPartitionTree hss_tree;
        recursive_bisection
          (gd.perm, gd.iperm, pbegin, part_begin, part_size, ld,
           gd.components, gd.leaf, hss_tree);
        gd.trees[nbsep] = hss_tree; // Not thread safe!!
      } else {
        for (integer_t z=part_begin[2]; z<part_begin[2]+part_size[2]; z++)
          for (integer_t y=part_begin[1]; y<part_begin[1]+part_size[1]; y++)
            for (integer_t x=part_begin[0]; x<part_begin[0]+part_size[0]; x++) {
              auto ind = comps * (x + y*ld[0] + z*ld[0]*ld[1]);
              for (int c=0; c<comps; c++) {
                gd.perm[ind] = pbegin;
                gd.iperm[pbegin++] = ind++;
              }
            }
      }
      if (nbsep)
        tree.emplace_back
          (tree.back().sep_end + sep_size, -1, left_root_id, nbsep-1);
      else tree.emplace_back(sep_size, -1, left_root_id, nbsep-1);
      nbsep++;
    }
  }

  template<typename integer_t,typename scalar_t>
  std::unique_ptr<SeparatorTree<integer_t>> geometric_nested_dissection
  (const CSRMatrix<scalar_t,integer_t>& A, int nx, int ny, int nz,
   int components, int width, std::vector<integer_t>& perm,
   std::vector<integer_t>& iperm, const SPOptions<scalar_t>& opts) {
    GeomOrderData<integer_t> gd;
    gd.perm = perm.data();
    gd.iperm = iperm.data();
    gd.components = components;
    gd.width = width;
    gd.stratpar = opts.nd_param();
    if (nx*ny*nz*components != A.size()) {
      nx = opts.nx();
      ny = opts.nz();
      nz = opts.nz();
      gd.components = opts.components();
      gd.width = opts.separator_width();
      if (nx*ny*nz*gd.components != A.size()) {
        std::cerr << "# ERROR: Geometric reordering failed. \n"
          "# Geometric reordering only works on"
          " a simple 3 point wide stencil\n"
          "# on a regular grid and you need to provide the mesh sizes."
                  << std::endl;
        return nullptr;
      }
    }
    gd.separator_reordering = opts.use_HSS() || opts.use_BLR();
    if (gd.separator_reordering) {
      gd.min_sep = A.size();
      gd.leaf = A.size();
      if (opts.use_HSS()) {
        gd.min_sep = opts.HSS_min_sep_size();
        gd.leaf = opts.HSS_options().leaf_size();
      } else if (opts.use_BLR()) {
        gd.min_sep = opts.BLR_min_sep_size();
        gd.leaf = opts.BLR_options().leaf_size();
      }
    }
    std::vector<Separator<integer_t>> tree;
    integer_t nbsep = 0, pbegin = 0;
    recursive_nested_dissection
      (pbegin, nbsep, {{0, 0, 0}}, {{nx, ny, nz}}, {{nx, ny, nz}}, tree, gd);
    std::unique_ptr<SeparatorTree<integer_t>> stree
      (new SeparatorTree<integer_t>(tree));
    if (gd.separator_reordering)
      stree->HSS_trees() = std::move(gd.trees);
    return stree;
  }

} // end namespace strumpack

#endif
