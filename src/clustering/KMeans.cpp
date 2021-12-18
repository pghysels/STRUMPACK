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
#include "Clustering.hpp"
#include "kernel/Metrics.hpp"

namespace strumpack {

  inline std::vector<std::size_t> kmeans_start_random
  (std::size_t n, int k, std::mt19937& generator) {
    std::uniform_int_distribution<std::size_t> uniform_random(0, n-1);
    std::vector<std::size_t> ind_centers(k);
    for (int i=0; i<k; i++)
      ind_centers[i] = uniform_random(generator);
    return ind_centers;
  }

  /** only works for k == 2 */
  template<typename scalar_t>
  std::vector<std::size_t> kmeans_start_random_dist_maximized
  (const DenseMatrix<scalar_t>& p, std::mt19937& generator) {
    constexpr std::size_t k = 2;
    const auto n = p.cols();
    const auto d = p.rows();
    std::uniform_int_distribution<std::size_t> uniform_random(0, n-1);
    const auto t = uniform_random(generator);
    // compute probabilities
    std::vector<scalar_t> cur_dist(n);
    for (std::size_t i=0; i<n; i++)
      cur_dist[i] = Euclidean_distance_squared(d, &p(0, i), &p(0, t));
    std::discrete_distribution<int> random_center
      (cur_dist.begin(), cur_dist.end());
    std::vector<std::size_t> ind_centers(k);
    ind_centers[0] = t;
    ind_centers[1] = random_center(generator);
    return ind_centers;
  }

  /** only works for k == 2 */
  template<typename scalar_t,
           typename real_t=typename RealType<scalar_t>::value_type>
  std::vector<std::size_t> kmeans_start_dist_maximized
  (const DenseMatrix<scalar_t>& p) {
    constexpr std::size_t k = 2;
    const auto n = p.cols();
    const auto d = p.rows();
    // find centroid
    std::vector<scalar_t> centroid(d);
    for (std::size_t i=0; i<n; i++)
      for (std::size_t j=0; j<d; j++)
        centroid[j] += p(j, i);
    for (std::size_t j=0; j<d; j++)
      centroid[j] /= n;
    // find farthest point from centroid
    std::size_t first_index = 0, second_index = 0;
    real_t max_dist(-1);
    for (std::size_t i=0; i<n; i++) {
      real_t dd = Euclidean_distance(d, &p(0, i), centroid.data());
      if (dd > max_dist) {
        max_dist = dd;
        first_index = i;
      }
    }
    // find farthest point from the first point
    max_dist = -1;
    for (std::size_t i=0; i<n; i++) {
      real_t dd = Euclidean_distance(d, &p(0, i), &p(0, first_index));
      if (dd > max_dist) {
        max_dist = dd;
        second_index = i;
      }
    }
    std::vector<std::size_t> ind_centers(k);
    ind_centers[0] = first_index;
    ind_centers[1] = second_index;
    return ind_centers;
  }

  /** only works for k == 2 */
  template<typename scalar_t>
  std::vector<std::size_t> kmeans_start_fixed
  (const DenseMatrix<scalar_t>& p) {
    std::vector<std::size_t> ind_centers(2);
    ind_centers[0] = 0;
    ind_centers[1] = p.cols() - 1;
    return ind_centers;
  }


  template<typename scalar_t,
           typename real_t=typename RealType<scalar_t>::value_type>
  void k_means
  (int k, DenseMatrix<scalar_t>& p, std::vector<std::size_t>& nc,
   int* perm, std::mt19937& generator) {
    const auto d = p.rows();
    const auto n = p.cols();
    DenseMatrix<scalar_t> center(d, k);
    const int kmeans_max_it = 100;
    std::vector<std::size_t> ind_centers;
    // TODO make this an option
    constexpr int kmeans_options = 2;
    switch (kmeans_options) {
    case 1: ind_centers = kmeans_start_random(n, k, generator); break;
    case 2: ind_centers = kmeans_start_random_dist_maximized(p, generator);
      break;
    case 3: ind_centers = kmeans_start_dist_maximized(p); break;
    case 4: ind_centers = kmeans_start_fixed(p); break;
    }
    for (int c=0; c<k; c++)
      for (std::size_t j=0; j<d; j++)
        center(j, c) = p(j, ind_centers[c]);
    int iter = 0;
    bool changes = true;
    std::vector<int> cluster(n);
    while ((changes == true) && (iter < kmeans_max_it)) {
      // for each point, find the closest cluster center
      changes = false;
      for (std::size_t i=0; i<n; i++) {
        auto min_dist = Euclidean_distance(d, &p(0, i), &center(0, 0));
        int ci = 0;
        for (int c=1; c<k; c++) {
          auto dd = Euclidean_distance(d, &p(0, i), &center(0, c));
          if (dd < min_dist) {
            min_dist = dd;
            ci = c;
          }
        }
        if (ci != cluster[i]) changes = true;
        cluster[i] = ci;
      }
      std::fill(nc.begin(), nc.end(), 0);
      center.zero();
      for (std::size_t i=0; i<n; i++) {
        auto c = cluster[i];
        nc[c]++;
        for (std::size_t j=0; j<d; j++)
          center(j, c) += p(j, i);
      }
      for (int c=0; c<k; c++)
        for (std::size_t j=0; j<d; j++)
          center(j, c) /= nc[c];
      iter++;
    }
    // permute the data
    std::size_t ct = 0;
    for (int c=0; c<k-1; c++)
      for (std::size_t j=0, cj=ct; j<nc[c]; j++) {
        while (cluster[cj] != c) cj++;
        if (cj != ct) {
          blas::swap(d, p.ptr(0,cj), 1, p.ptr(0,ct), 1);
          std::swap(perm[cj], perm[ct]);
          cluster[cj] = cluster[ct];
          cluster[ct] = c;
        }
        cj++;
        ct++;
      }
  }


  template<typename scalar_t>
  structured::ClusterTree recursive_2_means
  (DenseMatrix<scalar_t>& p, std::size_t cluster_size,
   int* perm, std::mt19937& generator) {
    const auto n = p.cols();
    structured::ClusterTree tree(n);
    if (n < cluster_size) return tree;
    std::vector<std::size_t> nc(2);
    k_means(2, p, nc, perm, generator);
    if (!nc[0] || !nc[1]) return tree;
    tree.c.resize(2);
    tree.c[0].size = nc[0];
    tree.c[1].size = nc[1];
    // TODO threading
    DenseMatrixWrapper<scalar_t> p0(p.rows(), nc[0], p, 0, 0);
    tree.c[0] = recursive_2_means(p0, cluster_size, perm, generator);
    DenseMatrixWrapper<scalar_t> p1(p.rows(), nc[1], p, 0, nc[0]);
    tree.c[1] = recursive_2_means(p1, cluster_size, perm+nc[0], generator);
    return tree;
  }


  // explicit template instantiations (only for real types!)
  template structured::ClusterTree
  recursive_2_means(DenseMatrix<float>& p, std::size_t cluster_size,
                    int* perm, std::mt19937& generator);
  template structured::ClusterTree
  recursive_2_means(DenseMatrix<double>& p, std::size_t cluster_size,
                    int* perm, std::mt19937& generator);

} // end namespace strumpack

