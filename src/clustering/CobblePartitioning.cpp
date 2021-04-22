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
#include <algorithm>

#include "Clustering.hpp"
#include "kernel/Metrics.hpp"

namespace strumpack {

  template<typename scalar_t> void cobble_partition
  (DenseMatrix<scalar_t>& p, std::vector<std::size_t>& nc, int* perm) {
    using real_t = scalar_t;
    auto d = p.rows();
    auto n = p.cols();
    // find centroid
    std::vector<scalar_t> centroid(d);
    for (std::size_t i=0; i<n; i++)
      for (std::size_t j=0; j<d; j++)
        centroid[j] += p(j, i);
    for (std::size_t j=0; j<d; j++)
      centroid[j] /= n;

    // find farthest point from centroid
    std::size_t first_index = 0;
    real_t max_dist(-1);
    for (std::size_t i=0; i<n; i++) {
      auto dd = Euclidean_distance(d, p.ptr(0, i), centroid.data());
      if (dd > max_dist) {
        max_dist = dd;
        first_index = i;
      }
    }

    // compute and sort distance from the firsth point
    std::vector<real_t> dists(n);
    for (std::size_t i=0; i<n; i++)
      dists[i] = Euclidean_distance(d, p.ptr(0, i), p.ptr(0, first_index));

    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element
      (idx.begin(), idx.begin() + n/2, idx.end(),
       [&](const std::size_t& a, const std::size_t& b) {
        return dists[a] < dists[b];
      });

    // split the data
    nc[0] = n/2;
    nc[1] = n - n/2;
    std::vector<int> cluster(n);
    for (std::size_t i=0; i<n/2; i++)
      cluster[idx[i]] = 0;
    for (std::size_t i=n/2; i<n; i++)
      cluster[idx[i]] = 1;

    // permute the data
    std::size_t ct = 0;
    for (std::size_t j=0, cj=ct; j<nc[0]; j++) {
      while (cluster[cj] != 0) cj++;
      if (cj != ct) {
        blas::swap(d, p.ptr(0,cj), 1, p.ptr(0,ct), 1);
        std::swap(perm[cj], perm[ct]);
        cluster[cj] = cluster[ct];
        cluster[ct] = 0;
      }
      cj++;
      ct++;
    }
  }

  template<typename scalar_t> structured::ClusterTree recursive_cobble
  (DenseMatrix<scalar_t>& p, std::size_t cluster_size, int* perm) {
    auto n = p.cols();
    structured::ClusterTree tree(n);
    if (n < cluster_size) return tree;
    std::vector<std::size_t> nc(2);
    cobble_partition(p, nc, perm);
    if (!nc[0] || !nc[1]) return tree;
    tree.c.resize(2);
    tree.c[0].size = nc[0];
    tree.c[1].size = nc[1];
    DenseMatrixWrapper<scalar_t> p0(p.rows(), nc[0], p, 0, 0);
    tree.c[0] = recursive_cobble(p0, cluster_size, perm);
    DenseMatrixWrapper<scalar_t> p1(p.rows(), nc[1], p, 0, nc[0]);
    tree.c[1] = recursive_cobble(p1, cluster_size, perm+nc[0]);
    return tree;
  }


  // explicit template instantiation (only for real types!)
  template void cobble_partition
  (DenseMatrix<float>& p, std::vector<std::size_t>& nc, int* perm);
  template void cobble_partition
  (DenseMatrix<double>& p, std::vector<std::size_t>& nc, int* perm);

  template structured::ClusterTree
  recursive_cobble(DenseMatrix<float>& p, std::size_t cluster_size,
                   int* perm);
  template structured::ClusterTree
  recursive_cobble(DenseMatrix<double>& p, std::size_t cluster_size,
                   int* perm);

} // end namespace strumpack
