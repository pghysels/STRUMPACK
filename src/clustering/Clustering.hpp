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
/**
 * \file Clustering.hpp
 * \brief Main include file for the different clustering/ordering
 * codes. These ordering codes can be used to define a (binary)
 * cluster tree.
 */
#ifndef STRUMPACK_CLUSTERING_HPP
#define STRUMPACK_CLUSTERING_HPP

#include "KDTree.hpp"
#include "KMeans.hpp"
#include "PCAPartitioning.hpp"
#include "CobblePartitioning.hpp"

namespace strumpack {

  /**
   * Enumeration of clustering codes to order input data and create a
   * binary cluster tree.
   * \ingroup Enumerations
   */
  enum class ClusteringAlgorithm {
    NATURAL,    /*!< No reordering, split evenly              */
    TWO_MEANS,  /*!< Defines a binary tree using k(=2)-means  */
    KD_TREE,    /*!< Simple kd-tree clustering                */
    PCA,        /*!< Cluster based on the principal component */
    COBBLE      /*!< Cobble partitioning                      */
  };

  /**
   * Return a short string with the name of the clustering algorithm.
   *
   * \param c Clustering algorithm.
   * \return String with the name of the clustering algorithm.
   */
  inline std::string get_name(ClusteringAlgorithm c) {
    switch (c) {
    case ClusteringAlgorithm::NATURAL: return "natural"; break;
    case ClusteringAlgorithm::TWO_MEANS: return "2means"; break;
    case ClusteringAlgorithm::KD_TREE: return "kdtree"; break;
    case ClusteringAlgorithm::PCA: return "PCA"; break;
    case ClusteringAlgorithm::COBBLE: return "cobble"; break;
    default: return "unknown";
    }
  }

  /**
   * Return a ClusteringAlgorithm enum based on the input string.
   *
   * \param c String, possible values are 'natural', '2means',
   * 'kdtree', 'pca' and 'cobble'. This is case sensitive.
   */
  inline ClusteringAlgorithm
  get_clustering_algorithm(const std::string& c) {
    if (c == "natural")     return ClusteringAlgorithm::NATURAL;
    else if (c == "2means") return ClusteringAlgorithm::TWO_MEANS;
    else if (c == "kdtree") return ClusteringAlgorithm::KD_TREE;
    else if (c == "pca")    return ClusteringAlgorithm::PCA;
    else if (c == "cobble") return ClusteringAlgorithm::COBBLE;
    else {
      std::cerr << "WARNING: binary tree clustering not recognized,"
                << " setting to recursive 2 means (2means)."
                << std::endl;
      return ClusteringAlgorithm::TWO_MEANS;
    }
  }


  /**
   * Reorder the input data and define a (binary) cluster tree.
   *
   * \param algo ClusteringAlgorithm to use
   *
   * \param p Input data set. This is a dxn matrix (column major). d
   * is the number of features, n is the number of datapoints. Hence
   * the data is stored point after point (column major). This will be
   * reordered according to perm.
   *
   * \param perm The permutation. The permutation uses 1-based
   * indexing, so it can be used with lapack permutation routines
   * (such as DenseMatrix::lapmt). This will be resized to the correct
   * size, ie., n == p.cols().
   *
   * \param cluster_size Stop partitioning when this cluster_size is
   * reached. This corresponds to the HSS/HODLR leaf size.
   *
   * \return This is output, a HSS::HSSPartitionTree defined by the
   * (recursive) clustering.
   *
   * \see strumpack::DenseMatrix::lapmt, get_clustering_algorithm,
   * get_name(ClusteringAlgorithm)
   */
  template<typename scalar_t>
  HSS::HSSPartitionTree binary_tree_clustering
  (ClusteringAlgorithm algo, DenseMatrix<scalar_t>& p,
   std::vector<int>& perm, std::size_t cluster_size) {
    HSS::HSSPartitionTree tree;
    perm.resize(p.cols());
    std::iota(perm.begin(), perm.end(), 1);
    switch (algo) {
    case ClusteringAlgorithm::NATURAL: {
      tree.size = p.cols();
      tree.refine(cluster_size);
    } break;
    case ClusteringAlgorithm::TWO_MEANS: {
      std::mt19937 gen(1); // reproducible
      tree = recursive_2_means(p, cluster_size, perm.data(), gen);
    } break;
    case ClusteringAlgorithm::KD_TREE:
      tree = recursive_kd(p, cluster_size, perm.data()); break;
    case ClusteringAlgorithm::PCA:
      tree = recursive_pca(p, cluster_size, perm.data()); break;
    case ClusteringAlgorithm::COBBLE:
      tree = recursive_cobble(p, cluster_size, perm.data()); break;
    default:
      std::cerr << "ERROR: clustering type not recognized." << std::endl;
    }
    return tree;
  }


} // end namespace strumpack

#endif // STRUMPACK_CLUSTERING_HPP
