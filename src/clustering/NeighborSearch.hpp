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
#ifndef NEIGHBOR_SEARCH_HPP
#define NEIGHBOR_SEARCH_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>

#include "dense/DenseMatrix.hpp"

namespace strumpack {

  // TODO move somewhere else
  template<typename scalar_t,
           typename real_t=typename RealType<scalar_t>::value_type>
  real_t Euclidean_distance_squared
  (std::size_t d, const scalar_t* x, const scalar_t* y) {
    real_t k(0.);
    for (std::size_t i=0; i<d; i++)
      k += (x[i]-y[i])*(x[i]-y[i]);
    return std::sqrt(k);
  }
  // TODO move somewhere else
  template<typename scalar_t,
           typename real_t=typename RealType<scalar_t>::value_type>
  real_t Euclidean_distance
  (std::size_t d, const scalar_t* x, const scalar_t* y) {
    return std::sqrt(Euclidean_distance_squared(d, x, y));
  }


  //--------------DISTANCE MATRIX------------------
  // finds distances between all data points with indices from
  // index_subset
  template<typename scalar_t=double, typename int_t=int,
           typename real_t=typename RealType<scalar_t>::value_type>
  DenseMatrix<real_t> find_distance_matrix
  (const DenseMatrix<scalar_t>& data,
   const std::vector<int_t>& index_subset) {
    auto subset_size = index_subset.size();
    DenseMatrix<real_t> distances(subset_size, subset_size);
    auto d = data.rows();
    for (std::size_t i=0; i<subset_size; i++) {
      distances(i, i) = real_t(0);
      for (std::size_t j=i+1; j<subset_size; j++) {
        distances(j, i) = Euclidean_distance
          (d, &data(0, index_subset[i]), &data(0, index_subset[j]));
        distances(i, j) = distances(j, i);
      }
    }
    return distances;
  }
  //-------FIND APPROXIMATE NEAREST NEIGHBORS FROM PROJECTION TREE---

  // 1. CONSTRUCT THE TREE
  // leaves - list of all indices such that
  // leaves[leaf_sizes[i]]...leaves[leaf_sizes[i+1]]
  // belong to i-th leaf of the projection tree
  // gauss_id and gaussian samples - for the fixed samples option
  template<typename scalar_t=double, typename int_t=int,
           typename real_t=typename RealType<scalar_t>::value_type>
  void construct_projection_tree
  (const DenseMatrix<scalar_t>& data, std::size_t min_leaf_size,
   std::vector<int_t>& cur_indices, std::size_t start,
   std::size_t cur_node_size, std::vector<std::size_t>& leaves,
   std::vector<std::size_t>& leaf_sizes, std::mt19937& generator) {
    auto d = data.rows();
    if (cur_node_size < min_leaf_size) {
      auto prev_size = leaf_sizes.back();
      leaf_sizes.push_back(cur_node_size + prev_size);
      for (std::size_t i=0; i<cur_node_size; i++)
        leaves.push_back(cur_indices[start+i]);
      return;
    }

    // choose random direction
    std::vector<scalar_t> direction_vector(d);
    std::normal_distribution<real_t> normal_distr(0.0, 1.0);
    for (std::size_t i=0; i<d; i++)
      direction_vector[i] = normal_distr(generator);
    real_t dir_vector_norm = blas::nrm2(d, &direction_vector[0], 1);
    for (std::size_t i=0; i<d; i++)
      direction_vector[i] /= dir_vector_norm;

    // find relative coordinates
    std::vector<scalar_t> relative_coordinates(cur_node_size, 0.0);
    for (std::size_t i=0; i<cur_node_size; i++)
      relative_coordinates[i] = blas::dotc
        (d, &data(0, cur_indices[start+i]), 1, &direction_vector[0], 1);

    // median split
    std::vector<int_t> idx(cur_node_size);
    iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](const int_t& a, const int_t& b) {
        return relative_coordinates[a] < relative_coordinates[b]; });
    std::vector<int_t> cur_indices_sorted(cur_node_size, 0);
    for (std::size_t i=0; i<cur_node_size; i++)
      cur_indices_sorted[i] = cur_indices[start+idx[i]];
    for (std::size_t i=start; i<start+cur_node_size; i++)
      cur_indices[i] = cur_indices_sorted[i-start];

    int_t half_size = (int_t)cur_node_size / 2;
    construct_projection_tree
      (data, min_leaf_size, cur_indices, start,
       half_size, leaves, leaf_sizes, generator);
    construct_projection_tree
      (data, min_leaf_size, cur_indices, start + half_size,
       cur_node_size - half_size, leaves, leaf_sizes, generator);
  }

  // 2. FIND CLOSEST POINTS INSIDE LEAVES
  // find ann_number exact neighbors for every point among the points
  // within its leaf (in randomized projection tree)
  template<typename scalar_t=double, typename int_t=int,
           typename real_t=typename RealType<scalar_t>::value_type>
  void find_neighbors_in_tree
  (const DenseMatrix<scalar_t>& data,
   std::size_t ann_number, std::vector<std::size_t>& leaves,
   std::vector<std::size_t>& leaf_sizes, DenseMatrix<int_t>& neighbors,
   DenseMatrix<real_t>& scores) {
    for (std::size_t leaf=0; leaf<leaf_sizes.size()-1; leaf++) {
      // initialize size and content of the current leaf
      auto cur_leaf_size = leaf_sizes[leaf+1] - leaf_sizes[leaf];
      // list of indices in the current leaf
      std::vector<int_t> index_subset(cur_leaf_size, 0);
      for (std::size_t i=0; i<index_subset.size(); i++)
        index_subset[i] = leaves[leaf_sizes[leaf] + i];
      auto leaf_dists = find_distance_matrix(data, index_subset);

      // record ann_number closest points in each leaf to neighbors
      for (std::size_t i=0; i<cur_leaf_size; i++) {
        std::vector<int_t> idx(cur_leaf_size);
        iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](const int_t& i1, const int_t& i2) {
                    return leaf_dists(i,i1) < leaf_dists(i,i2); });
        for (std::size_t j=0; j<ann_number; j++) {
          neighbors(j, index_subset[i]) = leaves[leaf_sizes[leaf] + idx[j]];
          scores(j, index_subset[i]) = leaf_dists(i, idx[j]);
        }
      }
    }
  }

  // 3. FIND ANN IN ONE TREE SAMPLE
  template<typename scalar_t=double, typename int_t=int,
           typename real_t=typename RealType<scalar_t>::value_type>
  void find_ann_candidates
  (const DenseMatrix<scalar_t>& data, std::size_t ann_number,
   DenseMatrix<int_t>& neighbors, DenseMatrix<real_t>& scores,
   std::mt19937& generator) {
    auto n = data.cols();
    std::size_t min_leaf_size = 6 * ann_number;
    std::vector<std::size_t> leaves, leaf_sizes;
    leaf_sizes.push_back(0);
    std::size_t cur_node_size = n;
    std::size_t start = 0;
    std::vector<int_t> cur_indices(cur_node_size);
    std::iota(cur_indices.begin(), cur_indices.end(), 0);
    construct_projection_tree
      (data, min_leaf_size, cur_indices, start,
       cur_node_size, leaves, leaf_sizes, generator);
    find_neighbors_in_tree
      (data, ann_number, leaves, leaf_sizes, neighbors, scores);
  }

  //---------------CHOOSE BEST NEIGHBORS FROM TWO TREE SAMPLES----------------

  // take closest neighbors from neighbors and new_neighbors, write them
  // to neighbors and scores
  template<typename scalar_t=double, typename int_t=int,
           typename real_t=typename RealType<scalar_t>::value_type>
  void choose_best_neighbors
  (DenseMatrix<int_t>& neighbors, DenseMatrix<real_t>& scores,
   DenseMatrix<int_t>& new_neighbors, DenseMatrix<real_t>& new_scores,
   std::size_t ann_number) {
    for (std::size_t c=0; c<neighbors.cols(); c++) {
      std::vector<int_t> cur_neighbors(ann_number);
      std::vector<real_t> cur_scores(ann_number);
      std::size_t r1 = 0, r2 = 0, cur = 0;
      while ((r1 < ann_number) && (r2 < ann_number) &&
             (cur < ann_number)) {
        if (scores(r1, c) > new_scores(r2, c)) {
          cur_neighbors[cur] = new_neighbors(r2, c);
          cur_scores[cur] = new_scores(r2, c);
          r2++;
        } else {
          cur_neighbors[cur] = neighbors(r1, c);
          cur_scores[cur] = scores(r1, c);
          if (neighbors(r1, c) == new_neighbors(r2, c)) r2++;
          r1++;
        }
        cur++;
      }
      while (cur < ann_number) {
        if (r1 == ann_number) {
          cur_neighbors[cur] = new_neighbors(r2, c);
          cur_scores[cur] = new_scores(r2, c);
          r2++;
        } else {
          cur_neighbors[cur] = neighbors(r1, c);
          cur_scores[cur] = scores(r1, c);
          r1++;
        }
        cur++;
      }
      for (std::size_t i=0; i<ann_number; i++) {
        neighbors(i, c) = cur_neighbors[i];
        scores(i, c) = cur_scores[i];
      }
    }
  }

  //----------------QUALITY CHECK WITH TRUE NEIGHBORS-------------------------
  template<typename scalar_t=double, typename int_t=int,
           typename real_t=typename RealType<scalar_t>::value_type>
  void find_true_nn
  (std::vector<scalar_t>& data, std::size_t n, std::size_t d,
   int ann_number, DenseMatrix<int_t>& n_neighbors,
   DenseMatrix<real_t>& n_scores) {
    std::vector<int_t> all_ids(n); // index subset = everything
    std::iota(all_ids.begin(), all_ids.end(), 0);
    // create full distance matrix
    auto all_dists = find_distance_matrix(data, d, all_ids);

    // record ann_number closest points in each leaf to neighbors
    for (std::size_t i=0; i<n; i++) {
      std::vector<int_t> idx(n);
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(), [&](const int_t& i1, const int& i2) {
          return all_dists(i, i1) < all_dists(i, i2); });
      for (int j=0; j<ann_number; j++) {
        n_neighbors(j, i) = idx[j];
        n_scores(j, i) = all_dists(i, idx[j]);
      }
    }
  }

  // quality = average fraction of ann_number approximate neighbors
  //  (neighbors), which are within the closest ann_number of true
  //  neighbors (n_neighbors); average is taken over all data points
  template<typename scalar_t=double, typename int_t=int,
           typename real_t=typename RealType<scalar_t>::value_type>
  double check_quality
  (const DenseMatrix<scalar_t>& data, std::size_t ann_number,
   std::vector<int_t>& neighbors) {
    auto n = data.cols();
    DenseMatrix<int_t> n_neighbors(ann_number, n);
    DenseMatrix<real_t> n_scores(ann_number, n);
    n_neighbors.zero();
    n_scores.zero();
    auto start_nn = std::chrono::system_clock::now();
    find_true_nn(data, ann_number, n_neighbors, n_scores);
    auto end_nn = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds_nn = end_nn-start_nn;
    std::cout << "elapsed time for exact neighbor search: "
              << elapsed_seconds_nn.count() << " sec" << std::endl;
    std::vector<double> quality_vec(n);
    for (std::size_t i=0; i<n; i++) {
      std::size_t r1 = 0, r2 = 0;
      int num_nei_found = 0;
      while (r2 < ann_number)
        if (neighbors(r1, i) == n_neighbors(r2, i)) {
          r1++;
          r2++;
          num_nei_found++;
        } else r2++;
      quality_vec[i] = (double)num_nei_found / ann_number;
    }
    std::cout << std::endl;
    double ann_quality = 0.0;
    for (std::size_t i=0; i<quality_vec.size(); i++)
      ann_quality += quality_vec[i];
    return (double)ann_quality/quality_vec.size();
  }

  //------------ Main function call----------------
  template<typename scalar_t=double, typename int_t=int,
           typename real_t=typename RealType<scalar_t>::value_type>
  void find_approximate_neighbors
  (DenseMatrix<scalar_t>& data, std::size_t num_iters, std::size_t ann_number,
   DenseMatrix<int_t>& neighbors, DenseMatrix<real_t>& scores,
   std::mt19937& generator) {
    auto n = data.cols();
    neighbors.resize(ann_number, n);
    scores.resize(ann_number, n);
    neighbors.zero();
    scores.zero();
    find_ann_candidates
      (data, ann_number, neighbors, scores, generator);
    // construct several random projection trees to find approximate
    // nearest neighbors
    for (std::size_t iter=1; iter<num_iters; iter++) {
      DenseMatrix<int_t> new_neighbors(ann_number, n);
      DenseMatrix<real_t> new_scores(ann_number, n);
      new_neighbors.zero();
      new_scores.zero();
      find_ann_candidates
        (data, ann_number, new_neighbors,
         new_scores, generator);
      choose_best_neighbors
        (neighbors, scores, new_neighbors, new_scores, ann_number);
    }
  }

} // end namespace strumpack

#endif // NEIGHBOR_SEARCH_HPP
