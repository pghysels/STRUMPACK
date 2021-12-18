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
#include <unordered_map>
#include <algorithm>

#include "GeometricReordering.hpp"

namespace strumpack {

  template<typename integer_t> class GeomOrderData {
  public:
    integer_t *perm, *iperm;
    int components, width, stratpar;
  };


  template<typename integer_t> void recursive_nested_dissection
  (integer_t& pbegin, integer_t& nbsep,
   std::array<integer_t,3> n0, std::array<integer_t,3> dims,
   std::array<integer_t,3> ld, std::vector<Separator<integer_t>>& tree,
   GeomOrderData<integer_t>& gd) {
    int comps = gd.components, width = gd.width, stratpar = gd.stratpar;
    integer_t N = comps * (dims[0]*dims[1]*dims[2]);
    // d: dimension along which to split
    int d = std::distance
      (dims.begin(), std::max_element(dims.begin(), dims.end()));

    if (dims[d] < 2+width || N <= stratpar) {
      for (integer_t z=n0[2]; z<n0[2]+dims[2]; z++)
        for (integer_t y=n0[1]; y<n0[1]+dims[1]; y++)
          for (integer_t x=n0[0]; x<n0[0]+dims[0]; x++) {
            auto ind = comps * (x + y*ld[0] + z*ld[0]*ld[1]);
            for (int c=0; c<comps; c++) {
              gd.perm[ind] = pbegin;
              gd.iperm[pbegin++] = ind++;
            }
          }
      if (nbsep) tree.emplace_back(tree.back().sep_end + N, -1, -1, -1);
      else tree.emplace_back(N, -1, -1, -1);
      nbsep++;
    } else {
      int dhalf = (n0[d] < ld[d]/2) ?
        std::ceil(dims[d]/2.) : std::floor(dims[d]/2.);

      // part 1
      std::array<integer_t,3> part_begin(n0), part_size(dims);
      part_size[d] = dhalf - (width/2);
      recursive_nested_dissection
        (pbegin, nbsep, part_begin, part_size, ld, tree, gd);
      auto left_root_id = nbsep - 1;

      // part 2
      part_begin[d] = n0[d] + dhalf + width;
      part_size[d] = dims[d] - width - dhalf;
      recursive_nested_dissection
        (pbegin, nbsep, part_begin, part_size, ld, tree, gd);
      tree[left_root_id].pa = nbsep;
      tree[nbsep-1].pa = nbsep;

      // separator
      part_begin[d] = n0[d] + dhalf - (width/2);
      part_size[d] = width;
      integer_t sep_size = comps * part_size[0]*part_size[1]*part_size[2];
      for (integer_t z=part_begin[2]; z<part_begin[2]+part_size[2]; z++)
        for (integer_t y=part_begin[1]; y<part_begin[1]+part_size[1]; y++)
          for (integer_t x=part_begin[0]; x<part_begin[0]+part_size[0]; x++) {
            auto ind = comps * (x + y*ld[0] + z*ld[0]*ld[1]);
            for (int c=0; c<comps; c++) {
              gd.perm[ind] = pbegin;
              gd.iperm[pbegin++] = ind++;
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
  (const CompressedSparseMatrix<scalar_t,integer_t>& A,
   int nx, int ny, int nz, int components, int width,
   std::vector<integer_t>& perm, std::vector<integer_t>& iperm,
   const SPOptions<scalar_t>& opts) {
    GeomOrderData<integer_t> gd;
    gd.perm = perm.data();
    gd.iperm = iperm.data();
    gd.components = components;
    gd.width = width;
    gd.stratpar = opts.nd_param();
    if (nx*ny*nz*components != A.size()) {
      nx = opts.nx();
      ny = opts.ny();
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
    std::vector<Separator<integer_t>> tree;
    integer_t nbsep = 0, pbegin = 0;
    recursive_nested_dissection
      (pbegin, nbsep, {{0, 0, 0}}, {{nx, ny, nz}}, {{nx, ny, nz}}, tree, gd);
    std::unique_ptr<SeparatorTree<integer_t>> stree
      (new SeparatorTree<integer_t>(tree));
    return stree;
  }

  template std::unique_ptr<SeparatorTree<int>>
  geometric_nested_dissection
  (const CompressedSparseMatrix<float,int>& A, int nx, int ny, int nz,
   int components, int width, std::vector<int>& perm,
   std::vector<int>& iperm, const SPOptions<float>& opts);
  template std::unique_ptr<SeparatorTree<int>>
  geometric_nested_dissection
  (const CompressedSparseMatrix<double,int>& A, int nx, int ny, int nz,
   int components, int width, std::vector<int>& perm,
   std::vector<int>& iperm, const SPOptions<double>& opts);
  template std::unique_ptr<SeparatorTree<int>>
  geometric_nested_dissection
  (const CompressedSparseMatrix<std::complex<float>,int>& A, int nx, int ny, int nz,
   int components, int width, std::vector<int>& perm,
   std::vector<int>& iperm, const SPOptions<std::complex<float>>& opts);
  template std::unique_ptr<SeparatorTree<int>>
  geometric_nested_dissection
  (const CompressedSparseMatrix<std::complex<double>,int>& A, int nx, int ny, int nz,
   int components, int width, std::vector<int>& perm,
   std::vector<int>& iperm, const SPOptions<std::complex<double>>& opts);

  template std::unique_ptr<SeparatorTree<long int>>
  geometric_nested_dissection
  (const CompressedSparseMatrix<float,long int>& A, int nx, int ny, int nz,
   int components, int width, std::vector<long int>& perm,
   std::vector<long int>& iperm, const SPOptions<float>& opts);
  template std::unique_ptr<SeparatorTree<long int>>
  geometric_nested_dissection
  (const CompressedSparseMatrix<double,long int>& A, int nx, int ny, int nz,
   int components, int width, std::vector<long int>& perm,
   std::vector<long int>& iperm, const SPOptions<double>& opts);
  template std::unique_ptr<SeparatorTree<long int>>
  geometric_nested_dissection
  (const CompressedSparseMatrix<std::complex<float>,long int>& A, int nx, int ny, int nz,
   int components, int width, std::vector<long int>& perm,
   std::vector<long int>& iperm, const SPOptions<std::complex<float>>& opts);
  template std::unique_ptr<SeparatorTree<long int>>
  geometric_nested_dissection
  (const CompressedSparseMatrix<std::complex<double>,long int>& A, int nx, int ny, int nz,
   int components, int width, std::vector<long int>& perm,
   std::vector<long int>& iperm, const SPOptions<std::complex<double>>& opts);

  template std::unique_ptr<SeparatorTree<long long int>>
  geometric_nested_dissection
  (const CompressedSparseMatrix<float,long long int>& A, int nx, int ny, int nz,
   int components, int width, std::vector<long long int>& perm,
   std::vector<long long int>& iperm, const SPOptions<float>& opts);
  template std::unique_ptr<SeparatorTree<long long int>>
  geometric_nested_dissection
  (const CompressedSparseMatrix<double,long long int>& A, int nx, int ny, int nz,
   int components, int width, std::vector<long long int>& perm,
   std::vector<long long int>& iperm, const SPOptions<double>& opts);
  template std::unique_ptr<SeparatorTree<long long int>>
  geometric_nested_dissection
  (const CompressedSparseMatrix<std::complex<float>,long long int>& A,
   int nx, int ny, int nz, int components, int width,
   std::vector<long long int>& perm, std::vector<long long int>& iperm,
   const SPOptions<std::complex<float>>& opts);
  template std::unique_ptr<SeparatorTree<long long int>>
  geometric_nested_dissection
  (const CompressedSparseMatrix<std::complex<double>,long long int>& A,
   int nx, int ny, int nz, int components, int width,
   std::vector<long long int>& perm, std::vector<long long int>& iperm,
   const SPOptions<std::complex<double>>& opts);

} // end namespace strumpack
