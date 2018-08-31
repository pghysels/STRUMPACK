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
     MPI_Comm comm, int nx, int ny, int nz, int components, int width);
#endif

    void separator_reordering
    (const SPOptions<scalar_t>& opts, CSRMatrix<scalar_t,integer_t>& A,
     FrontalMatrix<scalar_t,integer_t>& F);

    void separator_reordering
    (const SPOptions<scalar_t>& opts, CSRMatrix<scalar_t,integer_t>& A,
     bool verbose);

    virtual void clear_tree_data();

    const std::vector<integer_t>& perm() const { return perm_; }
    const std::vector<integer_t>& iperm() const { return iperm_; }

    const SeparatorTree<integer_t>& tree() const { return *sep_tree_; }

  protected:
    virtual void separator_reordering_print
    (integer_t max_nr_neighbours, integer_t max_dim_sep);

    void nested_dissection_print
    (const SPOptions<scalar_t>& opts, integer_t nnz, int max_level,
     int total_separators, bool verbose) const;

    std::vector<integer_t> perm_;
    std::vector<integer_t> iperm_;

    std::unique_ptr<SeparatorTree<integer_t>> sep_tree_;

  private:
    void split_separator
    (const SPOptions<scalar_t>& opts,
     HSS::HSSPartitionTree& hss_tree,
     integer_t& nr_parts, integer_t sep,
     const CSRMatrix<scalar_t,integer_t>& A,
     integer_t part, integer_t count,
     std::vector<integer_t>& sorder);

    void extract_separator
    (const SPOptions<scalar_t>& opts, integer_t part,
     integer_t sep_beg, integer_t sep_end,
     const CSRMatrix<scalar_t,integer_t>& A,
     std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy,
     std::vector<integer_t>& sorder);

    void separator_reordering_recursive
    (const SPOptions<scalar_t>& opts,
     const CSRMatrix<scalar_t,integer_t>& A,
     bool hss_parent, integer_t sep,
     std::vector<integer_t>& sorder);

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
   MPI_Comm comm, int nx, int ny, int nz, int components, int width) {
    if (!is_parallel(opts.reordering_method())) {
      auto rank = mpi_rank(comm);
      if (!rank) {
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
      MPI_Bcast(perm_.data(), perm_.size(), mpi_type<integer_t>(), 0, comm);
      MPI_Bcast(iperm_.data(), iperm_.size(), mpi_type<integer_t>(), 0, comm);
      integer_t nbsep;
      if (!rank) nbsep = sep_tree_->separators();
      MPI_Bcast(&nbsep, 1, mpi_type<integer_t>(), 0, comm);
      if (rank)
        sep_tree_ = std::unique_ptr<SeparatorTree<integer_t>>
          (new SeparatorTree<integer_t>(nbsep));
      sep_tree_->broadcast(comm);
    } else {
      if (opts.reordering_method() == ReorderingStrategy::GEOMETRIC) {
        sep_tree_ = geometric_nested_dissection
          (A, nx, ny, nz, components, width, perm_, iperm_, opts);
        if (!sep_tree_) return 1;
      } else {
        CSRMatrixMPI<scalar_t,integer_t> Ampi(&A, comm, false);
        switch (opts.reordering_method()) {
        case ReorderingStrategy::PARMETIS: {
#if defined(STRUMPACK_USE_PARMETIS)
          parmetis_nested_dissection(Ampi, comm, false, perm_, opts);
#else
          std::cerr << "ERROR: STRUMPACK was not configured with ParMetis support"
                    << std::endl;
          abort();
#endif
          break;
        }
        case ReorderingStrategy::PTSCOTCH: {
#if defined(STRUMPACK_USE_SCOTCH)
          ptscotch_nested_dissection(Ampi, comm, false, perm_, opts);
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
      (opts, A.nnz(), opts.verbose() && !mpi_rank(comm));
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
   bool verbose) {
    if (opts.reordering_method() != ReorderingStrategy::GEOMETRIC) {
      auto N = A.size();
      std::vector<integer_t> sorder(N);
      std::fill(sorder.begin(), sorder.end(), integer_t(0));
      integer_t root = sep_tree_->root();

#pragma omp parallel
#pragma omp single
      separator_reordering_recursive(opts, A, true, root, sorder);

      auto& iwork = iperm_;
      for (integer_t i=0; i<N; i++) sorder[i] = -sorder[i];
      for (integer_t i=0; i<N; i++) iwork[sorder[i]] = i;
      A.permute(iwork, sorder);
      // product of perm_ and sep_order
      for (integer_t i=0; i<N; i++) iwork[i] = sorder[perm_[i]];
      for (integer_t i=0; i<N; i++) perm_[i] = iwork[i];
      for (integer_t i=0; i<N; i++) iperm_[perm_[i]] = i;
    }

    if (opts.use_BLR()) {
      // find which blocks are admissible:
      //  loop over all edges in the separator [sep_begin,sep_end)
      //  if the 2 end vertices of this edge belong to different leafs in
      //  the HSS tree, then the interaction between the corresponding BLR
      //  blocks is not admissible
      for (auto& s : sep_tree_->HSS_trees()) {
        auto sep = s.first;
        auto& hss_tree = s.second;
        auto sep_begin = sep_tree_->sizes(sep);
        auto sep_end = sep_tree_->sizes(sep + 1);
        auto tiles = hss_tree.leaf_sizes();
        integer_t nr_tiles = tiles.size();
        std::vector<bool> adm(nr_tiles * nr_tiles, true);
        for (integer_t t=0; t<nr_tiles; t++)
          adm[t+t*nr_tiles] = false;
        if (opts.BLR_options().admissibility() == BLR::Admissibility::STRONG) {
          std::vector<integer_t> tile_sizes(nr_tiles+1);
          tile_sizes[0] = sep_begin;
          for (integer_t i=0; i<nr_tiles; i++)
            tile_sizes[i+1] = tiles[i] + tile_sizes[i];
          for (integer_t t=0; t<nr_tiles; t++) {
            for (integer_t i=tile_sizes[t]; i<tile_sizes[t+1]; i++) {
              auto Ai = iperm_[i];
              for (integer_t j=A.ptr(Ai); j<A.ptr(Ai+1); j++) {
                auto Aj = perm_[A.ind(j)];
                if (Aj < sep_begin || Aj >= sep_end) continue;
                integer_t tj = std::distance
                  (tile_sizes.begin(), std::upper_bound
                   (tile_sizes.begin(), tile_sizes.end(), Aj)) - 1;
                if (t != tj) adm[t+nr_tiles*tj] = adm[tj+nr_tiles*t] = false;
              }
            }
          }
        }
#if 0
        if (sep == sep_tree_->root()) {
          std::cout << "root_adm_"
                    << BLR::get_name(opts.BLR_options().admissibility())
                    << " = [" << std::endl;
          for (integer_t ti=0; ti<nr_tiles; ti++) {
            for (integer_t tj=0; tj<nr_tiles; tj++)
              std::cout << adm[ti+nr_tiles*tj] << " ";
            std::cout << std::endl;
          }
          std::cout << "];" << std::endl;
        }
#endif
        sep_tree_->admissibility(sep) = std::move(adm);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::separator_reordering_recursive
  (const SPOptions<scalar_t>& opts, const CSRMatrix<scalar_t,integer_t>& A,
   bool hss_parent, integer_t sep, std::vector<integer_t>& sorder) {
    auto sep_begin = sep_tree_->sizes(sep);
    auto sep_end = sep_tree_->sizes(sep + 1);
    auto dim_sep = sep_end - sep_begin;
    bool is_hss = hss_parent && (dim_sep >= opts.HSS_min_sep_size());
    bool is_blr = opts.use_BLR() && (dim_sep >= opts.BLR_min_sep_size());
    if (is_hss || is_blr) {
      int min_sep = is_hss ? opts.HSS_min_sep_size() : opts.BLR_min_sep_size();
      if (sep_tree_->lch(sep) != -1) {
#pragma omp task firstprivate(sep) default(shared)
        separator_reordering_recursive
          (opts, A, is_hss, sep_tree_->lch(sep), sorder);
      }
      if (sep_tree_->rch(sep) != -1) {
#pragma omp task firstprivate(sep) default(shared)
        separator_reordering_recursive
          (opts, A, is_hss, sep_tree_->rch(sep), sorder);
      }
#pragma omp taskwait
      HSS::HSSPartitionTree hss_tree(dim_sep);
      if (dim_sep > 2 * min_sep) {
        integer_t nr_parts = 0;
        split_separator(opts, hss_tree, nr_parts, sep, A, 0, 1, sorder);
        auto count = sep_begin;
        for (integer_t part=0; part<nr_parts; part++)
          for (integer_t i=sep_begin; i<sep_end; i++)
            if (sorder[i] == part) sorder[i] = -count++;
      } else for (integer_t i=sep_begin; i<sep_end; i++) sorder[i] = -i;
#pragma omp critical
      {  // not thread safe!
        sep_tree_->HSS_tree(sep) = std::move(hss_tree);
      }
    } else {
      if (sep_tree_->lch(sep) != -1)
        separator_reordering_recursive
          (opts, A, is_hss, sep_tree_->lch(sep), sorder);
      if (sep_tree_->rch(sep) != -1)
        separator_reordering_recursive
          (opts, A, is_hss, sep_tree_->rch(sep), sorder);
      for (integer_t i=sep_begin; i<sep_end; i++) sorder[i] = -i;
    }
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::split_separator
  (const SPOptions<scalar_t>& opts, HSS::HSSPartitionTree& hss_tree,
   integer_t& nr_parts, integer_t sep, const CSRMatrix<scalar_t,integer_t>& A,
   integer_t part, integer_t count, std::vector<integer_t>& sorder) {
    auto sep_begin = sep_tree_->sizes(sep);
    auto sep_end = sep_tree_->sizes(sep+1);
    std::vector<idx_t> xadj, adjncy;
    extract_separator(opts, part, sep_begin, sep_end, A, xadj, adjncy, sorder);
    idx_t edge_cut = 0, nvtxs=xadj.size()-1;
    std::vector<idx_t> partitioning(nvtxs);
    int info = WRAPPER_METIS_PartGraphRecursive
      (nvtxs, 1, xadj, adjncy, 2, edge_cut, partitioning);
    if (info != METIS_OK) {
      std::cerr << "METIS_PartGraphRecursive for separator"
                << " reordering returned: " << info << std::endl;
      exit(1);
    }
    hss_tree.c.resize(2);
    for (integer_t i=sep_begin, j=0; i<sep_end; i++)
      if (sorder[i] == part) {
        auto p = partitioning[j++];
        sorder[i] = -count - p;
        hss_tree.c[p].size++;
      }
    int leaf = opts.use_HSS() ? opts.HSS_options().leaf_size() :
      opts.BLR_options().leaf_size();
    for (integer_t p=0; p<2; p++)
      if (hss_tree.c[p].size > 2 * leaf)
        split_separator(opts, hss_tree.c[p], nr_parts, sep, A,
                        -count-p, count+2, sorder);
      else
        std::replace
          (&sorder[sep_begin], &sorder[sep_end], -count-p, nr_parts++);
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::extract_separator
  (const SPOptions<scalar_t>& opts, integer_t part,
   integer_t sep_begin, integer_t sep_end,
   const CSRMatrix<scalar_t,integer_t>& A,
   std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy,
   std::vector<integer_t>& sorder) {
    assert(opts.separator_ordering_level() == 0 ||
           opts.separator_ordering_level() == 1);
    auto dim_sep = sep_end - sep_begin;
    std::vector<bool> mark(dim_sep);
    std::vector<integer_t> ind_to_part(dim_sep);
    integer_t nvtxs = 0;
    for (integer_t r=0; r<dim_sep; r++)
      ind_to_part[r] = (sorder[r+sep_begin] == part) ? nvtxs++ : -1;
    xadj.reserve(nvtxs+1);
    adjncy.reserve(5*nvtxs);
    for (integer_t i=sep_begin, e=0; i<sep_end; i++) {
      if (sorder[i] == part) {
        xadj.push_back(e);
        std::fill(mark.begin(), mark.end(), false);
        for (integer_t j=A.ptr(i); j<A.ptr(i+1); j++) {
          auto c = A.ind(j);
          if (c == i) continue;
          auto lc = c - sep_begin;
          if (lc >= 0 && lc < dim_sep && sorder[c]==part && !mark[lc]) {
            mark[lc] = true;
            adjncy.push_back(ind_to_part[lc]);
            e++;
          } else {
            if (opts.separator_ordering_level() > 0) {
              for (integer_t k=A.ptr(c); k<A.ptr(c+1); k++) {
                auto cc = A.ind(k);
                auto lcc = cc - sep_begin;
                if (cc!=i && lcc >= 0 && lcc < dim_sep &&
                    sorder[cc]==part && !mark[lcc]) {
                  mark[lcc] = true;
                  adjncy.push_back(ind_to_part[lcc]);
                  e++;
                }
              }
            }
          }
        }
      }
    }
    xadj.push_back(adjncy.size());
  }


  // reorder the vertices in the separator to get a better rank structure
  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::separator_reordering
  (const SPOptions<scalar_t>& opts, CSRMatrix<scalar_t,integer_t>& A,
   FrontalMatrix<scalar_t,integer_t>& F) {
    auto N = A.size();
    std::vector<integer_t> sorder(N);
    std::fill(sorder.begin(), sorder.end(), integer_t(0));

#pragma omp parallel
#pragma omp single
    F.bisection_partitioning(opts, sorder.data());

    auto& iwork = iperm_;
    for (integer_t i=0; i<N; i++) sorder[i] = -sorder[i];
    for (integer_t i=0; i<N; i++) iwork[sorder[i]] = i;
    A.permute(iwork, sorder);

    // product of perm_ and sep_order
    for (integer_t i=0; i<N; i++) iwork[i] = sorder[perm_[i]];
    for (integer_t i=0; i<N; i++) perm_[i] = iwork[i];
    for (integer_t i=0; i<N; i++) iperm_[perm_[i]] = i;

#pragma omp parallel
#pragma omp single
    F.permute_upd_indices(sorder.data());
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
        << "# (enabled by default) iso --sp_enable_METIS_NodeND." << std::endl
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
