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
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>

#include "NeighborSearch.hpp"
#include "kernel/Metrics.hpp"

namespace strumpack {

  //--------------DISTANCE MATRIX------------------
  // finds distances between all data points with indices from
  // index_subset
  template<typename real_t, typename int_t>
  DenseMatrix<real_t> find_distance_matrix
  (const DenseMatrix<real_t>& data,
   const std::vector<int_t>& index_subset) {
    auto subset_size = index_subset.size();
    DenseMatrix<real_t> distances(subset_size, subset_size);
    auto d = data.rows();
    for (std::size_t i=0; i<subset_size; i++) {
      distances(i, i) = real_t(0);
      for (std::size_t j=i+1; j<subset_size; j++) {
        distances(j, i) = Euclidean_distance_squared
          (d, &data(0, index_subset[i]), &data(0, index_subset[j]));
        distances(i, j) = distances(j, i);
      }
    }
    return distances;
  }

  // finds distances between all data points with indices from
  // index_subset to all points in the data set
  template<typename real_t, typename int_t>
  DenseMatrix<real_t> find_distance_matrix_from_subset
  (const DenseMatrix<real_t>& data,
   const std::vector<int_t>& index_subset) {
    auto n = data.cols();
    auto d = data.rows();
    auto subset_size = index_subset.size();
    DenseMatrix<real_t> distances(subset_size, n);
    for (std::size_t j=0; j<n; j++)
      for (std::size_t i=0; i<subset_size; i++)
        distances(i, j) = Euclidean_distance_squared
          (d, &data(0, index_subset[i]), &data(0, j));
    return distances;
  }

  //-------FIND APPROXIMATE NEAREST NEIGHBORS FROM PROJECTION TREE---

  // 1. CONSTRUCT THE TREE
  // leaves - list of all indices such that
  // leaves[leaf_sizes[i]]...leaves[leaf_sizes[i+1]]
  // belong to i-th leaf of the projection tree
  // gauss_id and gaussian samples - for the fixed samples option
  template<typename real_t, typename int_t>
  void construct_projection_tree
  (const DenseMatrix<real_t>& data, std::size_t min_leaf_size,
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
    std::vector<real_t> direction_vector(d);
    std::normal_distribution<real_t> normal_distr(0.0, 1.0);
    for (std::size_t i=0; i<d; i++)
      direction_vector[i] = normal_distr(generator);
    real_t dir_vector_norm = blas::nrm2(d, &direction_vector[0], 1);
    for (std::size_t i=0; i<d; i++)
      direction_vector[i] /= dir_vector_norm;

    // find relative coordinates
    std::vector<real_t> relative_coordinates(cur_node_size, 0.0);
    //#pragma omp parallel for if(cur_node_size > 1000)
    for (std::size_t i=0; i<cur_node_size; i++)
      relative_coordinates[i] = blas::dotc
        (d, &data(0, cur_indices[start+i]), 1, &direction_vector[0], 1);

    // median split
    std::vector<int_t> idx(cur_node_size);
    std::iota(idx.begin(), idx.end(), 0);
    int_t half_size = (int_t)cur_node_size / 2;
    // std::nth_element
    //   (idx.begin(), idx.begin()+half_size, idx.end(),
    //    [&](const int_t& a, const int_t& b) {
    //      return (relative_coordinates[a] < relative_coordinates[b]) ||
    //        ((relative_coordinates[a] == relative_coordinates[b])
    //         && (a < b)); });
    std::sort
      (idx.begin(), idx.end(),
       [&](const int_t& a, const int_t& b) {
         return (relative_coordinates[a] < relative_coordinates[b]) ||
           ((relative_coordinates[a] == relative_coordinates[b]) && (a < b)); });
    // std::stable_sort
    //   (idx.begin(), idx.end(),
    //    [&](const int_t& a, const int_t& b) {
    //      return relative_coordinates[a] < relative_coordinates[b]; });
    std::vector<int_t> cur_indices_sorted(cur_node_size, 0);
    for (std::size_t i=0; i<cur_node_size; i++)
      cur_indices_sorted[i] = cur_indices[start+idx[i]];
    for (std::size_t i=start; i<start+cur_node_size; i++)
      cur_indices[i] = cur_indices_sorted[i-start];

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
  template<typename real_t, typename int_t>
  void find_neighbors_in_tree
  (const DenseMatrix<real_t>& data, std::vector<std::size_t>& leaves,
   std::vector<std::size_t>& leaf_sizes, DenseMatrix<int_t>& neighbors,
   DenseMatrix<real_t>& scores) {
    auto ann_number = neighbors.rows();
#pragma omp parallel for default(shared) schedule(dynamic)
    for (std::size_t leaf=0; leaf<leaf_sizes.size()-1; leaf++) {
      // initialize size and content of the current leaf
      auto cur_leaf_size = leaf_sizes[leaf+1] - leaf_sizes[leaf];
      // list of indices in the current leaf
      std::vector<int_t> index_subset(cur_leaf_size, 0);
      for (std::size_t i=0; i<index_subset.size(); i++)
        index_subset[i] = leaves[leaf_sizes[leaf] + i];
      auto leaf_dists = find_distance_matrix(data, index_subset);

      // record ann_number closest points in each leaf to neighbors
      std::vector<int_t> idx(cur_leaf_size);
      for (std::size_t i=0; i<cur_leaf_size; i++) {
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort
          (idx.begin(), idx.begin()+ann_number, idx.end(),
           [&](const int_t& i1, const int_t& i2) {
             return (leaf_dists(i,i1) < leaf_dists(i,i2)) ||
               ((leaf_dists(i,i1) == leaf_dists(i,i2)) && (i1 < i2)); });
        // std::stable_sort
        //   (idx.begin(), idx.end(),
        //    [&](const int_t& i1, const int_t& i2) {
        //      return leaf_dists(i,i1) < leaf_dists(i,i2); });
        for (std::size_t j=0; j<ann_number; j++) {
          neighbors(j, index_subset[i]) = leaves[leaf_sizes[leaf] + idx[j]];
          scores(j, index_subset[i]) = leaf_dists(i, idx[j]);
        }
      }
    }
  }

  // 3. FIND ANN IN ONE TREE SAMPLE
  template<typename real_t, typename int_t>
  void find_ann_candidates
  (const DenseMatrix<real_t>& data, DenseMatrix<int_t>& neighbors,
   DenseMatrix<real_t>& scores, std::mt19937& generator) {
    auto n = data.cols();
    auto ann_number = neighbors.rows();
    std::size_t min_leaf_size = 6 * ann_number;
    std::vector<std::size_t> leaves, leaf_sizes;
    leaves.reserve(n);
    leaf_sizes.reserve(2*n / min_leaf_size);
    leaf_sizes.push_back(0);
    std::size_t cur_node_size = n;
    std::size_t start = 0;
    std::vector<int_t> cur_indices(cur_node_size);
    std::iota(cur_indices.begin(), cur_indices.end(), 0);
    construct_projection_tree
      (data, min_leaf_size, cur_indices, start,
       cur_node_size, leaves, leaf_sizes, generator);
    find_neighbors_in_tree(data, leaves, leaf_sizes, neighbors, scores);
  }

  //---------------CHOOSE BEST NEIGHBORS FROM TWO TREE SAMPLES----------------

  // take closest neighbors from neighbors and new_neighbors, write them
  // to neighbors and scores
  template<typename real_t, typename int_t>
  void choose_best_neighbors
  (DenseMatrix<int_t>& neighbors, DenseMatrix<real_t>& scores,
   DenseMatrix<int_t>& new_neighbors, DenseMatrix<real_t>& new_scores) {
    auto ann_number = neighbors.rows();
    std::vector<int_t> cur_neighbors(ann_number);
    std::vector<real_t> cur_scores(ann_number);
    for (std::size_t c=0; c<neighbors.cols(); c++) {
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
  template<typename real_t, typename int_t> void find_true_nn
  (const DenseMatrix<real_t>& data, const std::vector<std::size_t>& samples,
   DenseMatrix<int_t>& neighbors, DenseMatrix<real_t>& scores) {
    auto n = data.cols();
    auto ann_number = neighbors.rows();
    auto sample_dists = find_distance_matrix_from_subset(data, samples);
    // record ann_number closest points in each leaf to neighbors
    std::vector<int_t> idx(n);
    for (int_t i=0; i<samples.size(); i++) {
      std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort
        (idx.begin(), idx.begin()+ann_number, idx.end(),
         [&](const int_t& i1, const int_t& i2) {
           return (sample_dists(i, i1) < sample_dists(i, i2)) ||
             ((sample_dists(i, i1) == sample_dists(i, i2)) && (i1 < i2)); });
      // std::stable_sort
      //   (idx.begin(), idx.end(),
      //    [&](const int_t& i1, const int& i2) {
      //      return sample_dists(i, i1) < sample_dists(i, i2); });
      for (std::size_t j=0; j<ann_number; j++) {
        neighbors(j, i) = idx[j];
        scores(j, i) = sample_dists(i, idx[j]);
      }
    }
  }

  // quality = average fraction of ann_number approximate neighbors
  //  (neighbors), which are within the closest ann_number of true
  //  neighbors (n_neighbors); average is taken over a subset
  //  (nr_samples)
  template<typename real_t, typename int_t> real_t check_quality
  (const DenseMatrix<real_t>& data, const DenseMatrix<int_t>& neighbors,
   std::mt19937& generator) {
    auto n = data.cols();
    auto ann_number = neighbors.rows();
    std::size_t nr_samples = 100;
    std::vector<std::size_t> samples(nr_samples);
    {
      std::uniform_int_distribution<std::size_t> uni_int(0, n-1);
      for (std::size_t i=0; i<samples.size(); i++)
        samples[i] = uni_int(generator);
    }
    DenseMatrix<int_t> n_neighbors(ann_number, nr_samples);
    DenseMatrix<real_t> n_scores(ann_number, nr_samples);
    n_neighbors.zero();
    n_scores.zero();
    find_true_nn(data, samples, n_neighbors, n_scores);
    real_t ann_quality = 0.0;
    for (std::size_t j=0; j<nr_samples; j++) {
      auto i = samples[j];
      std::size_t r1 = 0, r2 = 0;
      int num_nei_found = 0;
      while (r2 < ann_number)
        if (neighbors(r1, i) == n_neighbors(r2, j)) {
          r1++;
          r2++;
          num_nei_found++;
        } else r2++;
      ann_quality += (real_t)num_nei_found / ann_number;
    }
    return ann_quality / nr_samples;
  }

  //------------ Main function call----------------
  template<typename real_t, typename int_t> void find_approximate_neighbors
  (const DenseMatrix<real_t>& data, std::size_t num_iters,
   std::size_t ann_number, DenseMatrix<int_t>& neighbors,
   DenseMatrix<real_t>& scores) {
    auto n = data.cols();
    neighbors.resize(ann_number, n);
    scores.resize(ann_number, n);
    neighbors.zero();
    scores.zero();
    std::mt19937 generator(1); // reproducible
    find_ann_candidates(data, neighbors, scores, generator);
    real_t quality = check_quality(data, neighbors, generator);
    // construct several random projection trees to find approximate
    // nearest neighbors
    std::size_t iter = 0;
    for (; iter<num_iters && quality<0.99; iter++) {
      DenseMatrix<int_t> new_neighbors(ann_number, n);
      DenseMatrix<real_t> new_scores(ann_number, n);
      find_ann_candidates(data, new_neighbors, new_scores, generator);
      choose_best_neighbors(neighbors, scores, new_neighbors, new_scores);
      quality = check_quality(data, neighbors, generator);
    }
    std::cout << "# ANN search quality = " << quality
              << " after " << iter << " iterations" << std::endl;
  }

  // explicit template instantiations
  template void find_approximate_neighbors
  (const DenseMatrix<float>& data, std::size_t num_iters,
   std::size_t ann_number, DenseMatrix<unsigned int>& neighbors,
   DenseMatrix<float>& scores);
  template void find_approximate_neighbors
  (const DenseMatrix<double>& data, std::size_t num_iters,
   std::size_t ann_number, DenseMatrix<unsigned int>& neighbors,
   DenseMatrix<double>& scores);

} // end namespace strumpack
