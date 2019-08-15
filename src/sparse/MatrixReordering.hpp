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
#ifndef MATRIX_REORDERING_HPP
#define MATRIX_REORDERING_HPP

#include <vector>
#include <string>
#include <algorithm>
#include <memory>

#include "HSS/HSSPartitionTree.hpp"
#include "StrumpackOptions.hpp"
#include "StrumpackConfig.hpp"
#include "CSRMatrix.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "misc/MPIWrapper.hpp"
#include "CSRMatrixMPI.hpp"
#endif
#if defined(STRUMPACK_USE_SCOTCH)
#include "ScotchReordering.hpp"
#endif
#include "MetisReordering.hpp"
#if defined(STRUMPACK_USE_PARMETIS)
#include "ParMetisReordering.hpp"
#endif
#include "GeometricReordering.hpp"
#include "RCMReordering.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrix;

  template<typename scalar_t,typename integer_t> class MatrixReordering {
  public:
    MatrixReordering(integer_t  n);

    virtual ~MatrixReordering() {}

    int nested_dissection
    (const SPOptions<scalar_t>& opts,
     const CSRMatrix<scalar_t,integer_t>& A,
     int nx, int ny, int nz, int components, int width);

#if defined(STRUMPACK_USE_MPI)
    int nested_dissection
    (SPOptions<scalar_t>& opts, const CSRMatrix<scalar_t,integer_t>& A,
     const MPIComm& comm, int nx, int ny, int nz, int components, int width);
#endif

    void separator_reordering
    (const SPOptions<scalar_t>& opts, CSRMatrix<scalar_t,integer_t>& A,
     FrontalMatrix<scalar_t,integer_t>* F);

    virtual void clear_tree_data();

    const std::vector<integer_t>& perm() const { return perm_; }
    const std::vector<integer_t>& iperm() const { return iperm_; }

    const SeparatorTree<integer_t>& tree() const { return *sep_tree_; }
    SeparatorTree<integer_t>& tree() { return *sep_tree_; }

  protected:
    virtual void separator_reordering_print
    (integer_t max_nr_neighbours, integer_t max_dim_sep);

    void nested_dissection_print
    (const SPOptions<scalar_t>& opts, integer_t nnz, int max_level,
     int total_separators, bool verbose) const;

    std::vector<integer_t> perm_, iperm_;

    std::unique_ptr<SeparatorTree<integer_t>> sep_tree_;

  private:
    void nested_dissection_print
    (const SPOptions<scalar_t>& opts, integer_t nnz, bool verbose) const;
  };


  template<typename scalar_t,typename integer_t>
  MatrixReordering<scalar_t,integer_t>::MatrixReordering(integer_t n)
    : perm_(n), iperm_(n) {
  }

  // if running in parallel, only root should call this
  template<typename scalar_t,typename integer_t> int
  MatrixReordering<scalar_t,integer_t>::nested_dissection
  (const SPOptions<scalar_t>& opts,
   const CSRMatrix<scalar_t,integer_t>& A,
   int nx, int ny, int nz, int components, int width) {
    switch (opts.reordering_method()) {
    case ReorderingStrategy::NATURAL: {
      for (integer_t i=0; i<A.size(); i++) perm_[i] = i;
      sep_tree_ = build_sep_tree_from_perm(A.ptr(), A.ind(), perm_, iperm_);
      break;
    }
    case ReorderingStrategy::METIS: {
      sep_tree_ = metis_nested_dissection(A, perm_, iperm_, opts);
      break;
    }
    case ReorderingStrategy::SCOTCH: {
#if defined(STRUMPACK_USE_SCOTCH)
      sep_tree_ = scotch_nested_dissection(A, perm_, iperm_, opts);
#else
      std::cerr << "ERROR: STRUMPACK was not configured with Scotch support"
                << std::endl;
      abort();
#endif
      break;
    }
    case ReorderingStrategy::GEOMETRIC: {
      sep_tree_ = geometric_nested_dissection
        (A, nx, ny, nz, components, width, perm_, iperm_, opts);
      if (!sep_tree_) return 1;
      break;
    }
    case ReorderingStrategy::RCM: {
      sep_tree_ = rcm_reordering(A, perm_, iperm_);
      break;
    }
    default:
      std::cerr << "# ERROR: parallel matrix reorderings are"
        " not supported from this interface, \n"
        "\tuse StrumpackSparseSolverMPI or"
        " StrumpackSparseSolverMPIDist instead." << std::endl;
      return 1;
    }
    sep_tree_->check();
    nested_dissection_print(opts, A.nnz(), opts.verbose());
    return 0;
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> int
  MatrixReordering<scalar_t,integer_t>::nested_dissection
  (SPOptions<scalar_t>& opts, const CSRMatrix<scalar_t,integer_t>& A,
   const MPIComm& comm, int nx, int ny, int nz, int components, int width) {
    if (!is_parallel(opts.reordering_method())) {
      if (comm.is_root()) {
        switch (opts.reordering_method()) {
        case ReorderingStrategy::NATURAL: {
          for (integer_t i=0; i<A.size(); i++) perm_[i] = i;
          sep_tree_ = build_sep_tree_from_perm
            (A.ptr(), A.ind(), perm_, iperm_);
          break;
        }
        case ReorderingStrategy::METIS: {
          sep_tree_ = metis_nested_dissection(A, perm_, iperm_, opts);
          break;
        }
        case ReorderingStrategy::SCOTCH: {
#if defined(STRUMPACK_USE_SCOTCH)
          sep_tree_ = scotch_nested_dissection(A, perm_, iperm_, opts);
#else
          std::cerr << "ERROR: STRUMPACK was not configured with Scotch support"
                    << std::endl;
          abort();
#endif
          break;
        }
        case ReorderingStrategy::RCM: {
          sep_tree_ = rcm_reordering(A, perm_, iperm_);
          break;
        }
        default: assert(false);
        }
      }
      comm.broadcast(perm_);
      comm.broadcast(iperm_);
      integer_t nbsep;
      if (comm.is_root()) nbsep = sep_tree_->separators();
      comm.broadcast(nbsep);
      if (comm.is_root())
        sep_tree_ = std::unique_ptr<SeparatorTree<integer_t>>
          (new SeparatorTree<integer_t>(nbsep));
      sep_tree_->broadcast(comm.comm());
    } else {
      if (opts.reordering_method() == ReorderingStrategy::GEOMETRIC) {
        sep_tree_ = geometric_nested_dissection
          (A, nx, ny, nz, components, width, perm_, iperm_, opts);
        if (!sep_tree_) return 1;
      } else {
        CSRMatrixMPI<scalar_t,integer_t> Ampi(&A, comm.comm(), false);
        switch (opts.reordering_method()) {
        case ReorderingStrategy::PARMETIS: {
#if defined(STRUMPACK_USE_PARMETIS)
          parmetis_nested_dissection(Ampi, comm.comm(), false, perm_, opts);
#else
          std::cerr << "ERROR: STRUMPACK was not configured with ParMetis support"
                    << std::endl;
          abort();
#endif
          break;
        }
        case ReorderingStrategy::PTSCOTCH: {
#if defined(STRUMPACK_USE_SCOTCH)
          ptscotch_nested_dissection(Ampi, comm.comm(), false, perm_, opts);
#else
          std::cerr << "ERROR: STRUMPACK was not configured with Scotch support"
                    << std::endl;
          abort();
#endif
          break;
        }
        default: assert(true);
        }
        sep_tree_ = build_sep_tree_from_perm(A.ptr(), A.ind(), perm_, iperm_);
      }
    }
    nested_dissection_print
      (opts, A.nnz(), opts.verbose() && comm.is_root());
    return 0;
  }
#endif

  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::clear_tree_data() {
    sep_tree_ = nullptr;
  }

  // reorder the vertices in the separator to get a better rank structure
  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::separator_reordering
  (const SPOptions<scalar_t>& opts, CSRMatrix<scalar_t,integer_t>& A,
   FrontalMatrix<scalar_t,integer_t>* F) {
    auto N = A.size();
    std::vector<integer_t> sorder(N);
#pragma omp parallel
#pragma omp single
    F->partition_fronts(opts, A, sorder.data());
    for (integer_t i=0; i<N; i++) iperm_[sorder[i]] = i;
    A.permute(iperm_, sorder);
    // product of perm_ and sep_order
    for (integer_t i=0; i<N; i++) iperm_[i] = sorder[perm_[i]];
    for (integer_t i=0; i<N; i++) perm_[iperm_[i]] = i;
    std::swap(perm_, iperm_);
#pragma omp parallel
#pragma omp single
    F->permute_CB(sorder.data());
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::nested_dissection_print
  (const SPOptions<scalar_t>& opts, integer_t nnz, bool verbose) const {
    nested_dissection_print
      (opts, nnz, sep_tree_->levels(), sep_tree_->separators(), verbose);
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::nested_dissection_print
  (const SPOptions<scalar_t>& opts, integer_t nnz, int max_level,
   int total_separators, bool verbose) const {
    if (verbose) {
      std::cout << "# initial matrix:" << std::endl;
      std::cout << "#   - number of unknowns = "
                << number_format_with_commas(perm_.size()) << std::endl;
      std::cout << "#   - number of nonzeros = "
                << number_format_with_commas(nnz) << std::endl;
      std::cout << "# nested dissection reordering:" << std::endl;
      std::cout << "#   - " << get_name(opts.reordering_method())
                << " reordering" << std::endl;
      if (opts.reordering_method() == ReorderingStrategy::METIS) {
        if (opts.use_METIS_NodeNDP()) {
          std::cout << "#      - used METIS_NodeNDP (iso METIS_NodeND)"
                    << std::endl;
          if (opts.use_MUMPS_SYMQAMD())
            std::cout
              << "#      - supernodal tree was built using MUMPS_SYMQAMD "
              << (opts.use_agg_amalg() ? "with" : "without")
              << " aggressive amalgamation" << std::endl;
          else
            std::cout << "#      - supernodal tree from METIS_NodeNDP is used"
                      << std::endl;
        } else {
          std::cout << "#      - used METIS_NodeND (iso METIS_NodeNDP)"
                    << std::endl;
          if (opts.use_MUMPS_SYMQAMD())
            std::cout
              << "#      - supernodal tree was built using MUMPS_SYMQAMD "
              << (opts.use_agg_amalg() ? "with" : "without")
              << " aggressive amalgamation" << std::endl;
          else
            std::cout << "#      - supernodal tree was built from etree"
                      << std::endl;
        }
      }
      std::cout << "#   - strategy parameter = "
                << opts.nd_param() << std::endl;
      std::cout << "#   - number of separators = "
                << number_format_with_commas(total_separators) << std::endl;
      std::cout << "#   - number of levels = "
                << number_format_with_commas(max_level)
                << std::flush << std::endl;
    }
    if (max_level > 50)
      std::cerr
        << "# ***** WARNING ****************************************************" << std::endl
        << "# Detected a large number of levels in the frontal/elimination tree." << std::endl
        << "# STRUMPACK currently does not handle this safely, which" << std::endl
        << "# could lead to segmentation faults due to stack overflows." << std::endl
        << "# As a remedy, you can try to increase the stack size," << std::endl
        << "# or try a different ordering (metis, scotch, ..)." << std::endl
        << "# When using metis, it often helps to use --sp_enable_METIS_NodeNDP," << std::endl
        << "# iso --sp_enable_METIS_NodeND." << std::endl
        << "# ******************************************************************"
        << std::endl;
    if (opts.log_assembly_tree()) {
      std::string filename = "assembly_tree_" +
        get_name(opts.reordering_method());
      if (opts.reordering_method() == ReorderingStrategy::METIS) {
        if (opts.use_METIS_NodeNDP()) {
          filename += "_NodeNDP";
          if (opts.use_MUMPS_SYMQAMD())
            filename += "_SYMQAMD";
          else filename += "_SEPTREE";
        } else {
          filename += "_NodeND";
          if (opts.use_MUMPS_SYMQAMD())
            filename += "_SYMQAMD";
          else filename += "_ETREE";
        }
      }
      sep_tree_->printm(filename);
    }
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::separator_reordering_print
  (integer_t max_nr_neighbours, integer_t max_dim_sep) {
    std::cout << "# separator reordering:" << std::endl;
    std::cout << "#   - maximum connectivity = "
              << max_nr_neighbours << std::endl;
    std::cout << "#   - maximum separator dimension = "
              << max_dim_sep << std::endl;
  }

} // end namespace strumpack

#endif
