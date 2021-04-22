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

namespace strumpack {

  template<typename scalar_t> void kd_partition
  (DenseMatrix<scalar_t>& p, std::vector<std::size_t>& nc,
   std::size_t cluster_size, int* perm) {
    auto n = p.cols();
    auto d = p.rows();
    // find coordinate of the most spread
    std::vector<scalar_t> maxs(d), mins(d);
    for (std::size_t j=0; j<d; ++j)
      maxs[j] = mins[j] = p(j, 0);
    for (std::size_t i=1; i<n; ++i)
      for (std::size_t j=0; j<d; ++j) {
        maxs[j] = std::max(p(j, i), maxs[j]);
        mins[j] = std::min(p(j, i), mins[j]);
      }
    scalar_t max_var = maxs[0] - mins[0];
    std::size_t dim = 0;
    for (std::size_t j=1; j<d; ++j) {
      auto t = maxs[j] - mins[j];
      if (t > max_var) {
        max_var = t;
        dim = j;
      }
    }

    std::vector<int> cluster(n);
    nc.resize(2);
    nc[0] = nc[1] = 0;

#if 0 // mean
    // find the mean
    scalar_t mean_value(0.);
    for (std::size_t i=0; i<n; ++i)
      mean_value += p(dim, i);
    mean_value /= n;
    // split the data
    for (std::size_t i=0; i<n; ++i)
      if (p(dim, i) > mean_value) {
        cluster[i] = 1;
        nc[1]++;
      } else nc[0]++;
    // if clusters are too disbalanced, assign trivial clusters
    if ((nc[0] < cluster_size && nc[1] > 100 * cluster_size) ||
        (nc[1] < cluster_size && nc[0] > 100 * cluster_size)) {
      // TODO should we still sort the data??
      nc[0] = nc[1] = 0;
      for (std::size_t i=0; i<n; i++) {
        if (i <= n / 2) {
          cluster[i] = 0;
          nc[0]++;
        } else {
          cluster[i] = 1;
          nc[1]++;
        }
      }
    }
#else // median
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element
      (idx.begin(), idx.begin() + n/2, idx.end(),
       [&](const std::size_t& a, const std::size_t& b) {
         return p(dim, a) < p(dim, b);
       });
    // split the data
    nc[0] = n/2;
    nc[1] = n - n/2;
    for (std::size_t i=0; i<n/2; i++)
      cluster[idx[i]] = 0;
    for (std::size_t i=n/2; i<n; i++)
      cluster[idx[i]] = 1;
#endif

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


  template<typename scalar_t> structured::ClusterTree recursive_kd
  (DenseMatrix<scalar_t>& p, std::size_t cluster_size, int* perm) {
    auto n = p.cols();
    structured::ClusterTree tree(n);
    if (n < cluster_size) return tree;
    std::vector<std::size_t> nc(2);
    kd_partition(p, nc, cluster_size, perm);
    if (!nc[0] || !nc[1]) return tree;
    tree.c.resize(2);
    tree.c[0].size = nc[0];
    tree.c[1].size = nc[1];
    DenseMatrixWrapper<scalar_t> p0(p.rows(), nc[0], p, 0, 0);
    tree.c[0] = recursive_kd(p0, cluster_size, perm);
    DenseMatrixWrapper<scalar_t> p1(p.rows(), nc[1], p, 0, nc[0]);
    tree.c[1] = recursive_kd(p1, cluster_size, perm+nc[0]);
    return tree;
  }

  // explicit template instantiations (only for real!)
  template structured::ClusterTree recursive_kd
  (DenseMatrix<float>& p, std::size_t cluster_size, int* perm);
  template structured::ClusterTree recursive_kd
  (DenseMatrix<double>& p, std::size_t cluster_size, int* perm);


} // end namespace strumpack

