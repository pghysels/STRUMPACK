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

#include "misc/MPIWrapper.hpp"
#include "HSS/HSSPartitionTree.hpp"
#include "StrumpackOptions.hpp"
#include "CSRMatrix.hpp"
#include "CSRMatrixMPI.hpp"
#include "ScotchReordering.hpp"
#include "MetisReordering.hpp"
#include "ParMetisReordering.hpp"
#include "GeometricReordering.hpp"
#include "RCMReordering.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrix;

  template<typename scalar_t,typename integer_t> class MatrixReordering {
  public:
    MatrixReordering(integer_t _n);
    virtual ~MatrixReordering();

    int nested_dissection(SPOptions<scalar_t>& opts,
                          CSRMatrix<scalar_t,integer_t>* A,
                          int nx, int ny, int nz);
    int nested_dissection(SPOptions<scalar_t>& opts,
                          CSRMatrix<scalar_t,integer_t>* A,
                          MPI_Comm comm, int nx, int ny, int nz);

    void separator_reordering(const SPOptions<scalar_t>& opts,
                              CSRMatrix<scalar_t,integer_t>* A,
                              FrontalMatrix<scalar_t,integer_t>* F);
    void separator_reordering(const SPOptions<scalar_t>& opts,
                              CSRMatrix<scalar_t,integer_t>* A,
                              bool verbose);

    virtual void clear_tree_data();

    integer_t n;
    integer_t* perm;
    integer_t* iperm;

    std::unique_ptr<SeparatorTree<integer_t>> sep_tree;

  protected:
    virtual void separator_reordering_print(integer_t max_nr_neighbours,
                                            integer_t max_dim_sep);
    virtual void nested_dissection_print(SPOptions<scalar_t>& opts,
                                         integer_t n, integer_t nnz,
                                         bool verbose);

  private:
    void split_separator(const SPOptions<scalar_t>& opts,
                         HSS::HSSPartitionTree& hss_tree,
                         integer_t& nr_parts, integer_t sep,
                         CSRMatrix<scalar_t,integer_t>* A, integer_t part,
                         integer_t count, integer_t* sorder);
    void extract_separator(const SPOptions<scalar_t>& opts, integer_t part,
                           integer_t sep_beg, integer_t sep_end,
                           CSRMatrix<scalar_t,integer_t>* A,
                           std::vector<idx_t>& xadj,
                           std::vector<idx_t>& adjncy,
                           integer_t* sorder);
    void separator_reordering_recursive(const SPOptions<scalar_t>& opts,
                                        CSRMatrix<scalar_t,integer_t>* A,
                                        bool hss_parent, integer_t sep,
                                        integer_t* sorder);
  };

  template<typename scalar_t,typename integer_t>
  MatrixReordering<scalar_t,integer_t>::MatrixReordering(integer_t _n)
    : n(_n), perm(new integer_t[2*n]), iperm(perm+n) {
  }

  template<typename scalar_t,typename integer_t>
  MatrixReordering<scalar_t,integer_t>::~MatrixReordering() {
    delete[] perm;
  }

  // if running in parallel, only root should call this
  template<typename scalar_t,typename integer_t> int
  MatrixReordering<scalar_t,integer_t>::nested_dissection
  (SPOptions<scalar_t>& opts, CSRMatrix<scalar_t,integer_t>* A,
   int nx, int ny, int nz) {
    switch (opts.reordering_method()) {
    case ReorderingStrategy::NATURAL: {
      for (integer_t i=0; i<A->size(); i++) perm[i] = i;
      sep_tree = build_sep_tree_from_perm
        (A->size(), A->get_ptr(), A->get_ind(), perm, iperm);
      break;
    }
    case ReorderingStrategy::METIS: {
      sep_tree = metis_nested_dissection(A, perm, iperm, opts);
      break;
    }
    case ReorderingStrategy::SCOTCH: {
      sep_tree = scotch_nested_dissection(A, perm, iperm, opts);
      break;
    }
    case ReorderingStrategy::GEOMETRIC: {
      if (nx*ny*nz != A->size()) {
        std::cerr << "# ERROR: Geometric reordering failed. \n"
          "# Geometric reordering only works on"
          " a simple 3 point wide stencil\n"
          "# on a regular grid and you need to provide the mesh sizes."
                  << std::endl;
        return 1;
      }
      sep_tree = geometric_nested_dissection
        (nx, ny, nz, perm, iperm, opts.nd_param());
      break;
    }
    case ReorderingStrategy::RCM: {
      sep_tree = rcm_reordering(A, perm, iperm);
      break;
    }
    default:
      std::cerr << "# ERROR: parallel matrix reorderings are"
        " not supported from this interface, \n"
        "\tuse StrumpackSparseSolverMPI or"
        " StrumpackSparseSolverMPIDist instead." << std::endl;
      return 1;
    }
    sep_tree->check();
    nested_dissection_print(opts, n, A->nnz(), opts.verbose());
    return 0;
  }

  template<typename scalar_t,typename integer_t> int
  MatrixReordering<scalar_t,integer_t>::nested_dissection
  (SPOptions<scalar_t>& opts, CSRMatrix<scalar_t,integer_t>* A,
   MPI_Comm comm, int nx, int ny, int nz) {
    if (!is_parallel(opts.reordering_method())) {
      auto rank = mpi_rank(comm);
      if (!rank) {
        switch (opts.reordering_method()) {
        case ReorderingStrategy::NATURAL: {
          for (integer_t i=0; i<A->size(); i++) perm[i] = i;
          sep_tree = build_sep_tree_from_perm
            (A->size(), A->get_ptr(), A->get_ind(), perm, iperm);
          break;
        }
        case ReorderingStrategy::METIS: {
          sep_tree = metis_nested_dissection(A, perm, iperm, opts);
          break;
        }
        case ReorderingStrategy::SCOTCH: {
          sep_tree = scotch_nested_dissection(A, perm, iperm, opts);
          break;
        }
        case ReorderingStrategy::RCM: {
          sep_tree = rcm_reordering(A, perm, iperm);
          break;
        }
        default: assert(false);
        }
      }
      MPI_Bcast(perm, 2*n, mpi_type<integer_t>(), 0, comm);
      integer_t nbsep;
      if (!rank) nbsep = sep_tree->separators();
      MPI_Bcast(&nbsep, 1, mpi_type<integer_t>(), 0, comm);
      if (rank)
        sep_tree = std::unique_ptr<SeparatorTree<integer_t>>
          (new SeparatorTree<integer_t>(nbsep));
      sep_tree->broadcast(comm);
    } else {
      if (opts.reordering_method() == ReorderingStrategy::GEOMETRIC) {
        if (nx*ny*nz != A->size()) {
          std::cerr << "# ERROR: Geometric reordering failed. \n"
            "# Geometric reordering only works on"
            " a simple 3 point wide stencil\n"
            "# on a regular grid and you need to provide the mesh sizes."
                    << std::endl;
          return 1;
        }
        sep_tree = geometric_nested_dissection
          (nx, ny, nz, perm, iperm, opts.nd_param());
      } else {
        CSRMatrixMPI<scalar_t,integer_t> Ampi(A, comm, false);
        switch (opts.reordering_method()) {
        case ReorderingStrategy::PARMETIS: {
          parmetis_nested_dissection(&Ampi, comm, false, perm, opts);
          break;
        }
        case ReorderingStrategy::PTSCOTCH: {
          ptscotch_nested_dissection(&Ampi, comm, false, perm, opts);
          break;
        }
        default: assert(true);
        }
        sep_tree = build_sep_tree_from_perm
          (n, A->get_ptr(), A->get_ind(), perm, iperm);
      }
    }
    nested_dissection_print(opts, n, A->nnz(),
                            opts.verbose() && !mpi_rank(comm));
    return 0;
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::clear_tree_data() {
    sep_tree.reset(nullptr);
  }


  // reorder the vertices in the separator to get a better rank structure
  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::separator_reordering
  (const SPOptions<scalar_t>& opts, CSRMatrix<scalar_t,integer_t>* A,
   bool verbose) {
    auto N = A->size();
    auto sorder = new integer_t[N];
    std::fill(sorder, sorder+N, integer_t(0));
    integer_t root = sep_tree->root();
#pragma omp parallel
#pragma omp single
    separator_reordering_recursive(opts, A, true, root, sorder);

    auto iwork = iperm;
    for (integer_t i=0; i<N; i++) sorder[i] = -sorder[i];
    for (integer_t i=0; i<N; i++) iwork[sorder[i]] = i;
    A->permute(iwork, sorder);

    // product of perm and sep_order
    for (integer_t i=0; i<N; i++) iwork[i] = sorder[perm[i]];
    for (integer_t i=0; i<N; i++) perm[i] = iwork[i];
    for (integer_t i=0; i<N; i++) iperm[perm[i]] = i;
    delete[] sorder;
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::separator_reordering_recursive
  (const SPOptions<scalar_t>& opts, CSRMatrix<scalar_t,integer_t>* A,
   bool hss_parent, integer_t sep, integer_t* sorder) {
    auto sep_begin = sep_tree->sizes()[sep];
    auto sep_end = sep_tree->sizes()[sep + 1];
    auto dim_sep = sep_end - sep_begin;
    bool is_hss = hss_parent && (dim_sep >= opts.HSS_min_sep_size());
    if (is_hss) {
      if (sep_tree->lch()[sep] != -1) {
#pragma omp task firstprivate(sep) default(shared)
        separator_reordering_recursive
          (opts, A, is_hss, sep_tree->lch()[sep], sorder);
      }
      if (sep_tree->rch()[sep] != -1) {
#pragma omp task firstprivate(sep) default(shared)
        separator_reordering_recursive
          (opts, A, is_hss, sep_tree->rch()[sep], sorder);
      }
#pragma omp taskwait
      HSS::HSSPartitionTree hss_tree(dim_sep);
      if (dim_sep > 2 * opts.HSS_min_sep_size()) {
        integer_t nr_parts = 0;
        split_separator(opts, hss_tree, nr_parts, sep, A, 0, 1, sorder);
        auto count = sep_begin;
        for (integer_t part=0; part<nr_parts; part++)
          for (integer_t i=sep_begin; i<sep_end; i++)
            if (sorder[i] == part) sorder[i] = -count++;
      } else for (integer_t i=sep_begin; i<sep_end; i++) sorder[i] = -i;
#pragma omp critical
      {
        sep_tree->HSS_trees()[sep] = hss_tree; // not thread safe!
      }
    } else {
      if (sep_tree->lch()[sep] != -1)
        separator_reordering_recursive
          (opts, A, is_hss, sep_tree->lch()[sep], sorder);
      if (sep_tree->rch()[sep] != -1)
        separator_reordering_recursive
          (opts, A, is_hss, sep_tree->rch()[sep], sorder);
      for (integer_t i=sep_begin; i<sep_end; i++) sorder[i] = -i;
    }
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::split_separator
  (const SPOptions<scalar_t>& opts, HSS::HSSPartitionTree& hss_tree,
   integer_t& nr_parts, integer_t sep, CSRMatrix<scalar_t,integer_t>* A,
   integer_t part, integer_t count, integer_t* sorder) {
    auto sep_begin = sep_tree->sizes()[sep];
    auto sep_end = sep_tree->sizes()[sep+1];
    std::vector<idx_t> xadj, adjncy;
    extract_separator(opts, part, sep_begin, sep_end, A, xadj, adjncy, sorder);
    idx_t ncon = 1, edge_cut = 0, two = 2, nvtxs=xadj.size()-1;
    auto partitioning = new idx_t[nvtxs];
    int info = METIS_PartGraphRecursive
      (&nvtxs, &ncon, xadj.data(), adjncy.data(), NULL, NULL, NULL,
       &two, NULL, NULL, NULL, &edge_cut, partitioning);
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
    delete[] partitioning;
    for (integer_t p=0; p<2; p++)
      if (hss_tree.c[p].size > 2 * opts.HSS_options().leaf_size())
        split_separator(opts, hss_tree.c[p], nr_parts, sep, A,
                        -count-p, count+2, sorder);
      else std::replace(sorder+sep_begin, sorder+sep_end,
                        -count-p, nr_parts++);
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::extract_separator
  (const SPOptions<scalar_t>& opts, integer_t part,
   integer_t sep_begin, integer_t sep_end, CSRMatrix<scalar_t,integer_t>* A,
   std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy, integer_t* sorder) {
    assert(opts.separator_ordering_level() == 0 ||
           opts.separator_ordering_level() == 1);
    auto dim_sep = sep_end - sep_begin;
    auto mark = new bool[dim_sep];
    auto ind_to_part = new integer_t[dim_sep];
    integer_t nvtxs = 0;
    for (integer_t r=0; r<dim_sep; r++)
      ind_to_part[r] = (sorder[r+sep_begin] == part) ? nvtxs++ : -1;
    xadj.reserve(nvtxs+1);
    adjncy.reserve(5*nvtxs);
    for (integer_t i=sep_begin, e=0; i<sep_end; i++) {
      if (sorder[i] == part) {
        xadj.push_back(e);
        std::fill(mark, mark+dim_sep, false);
        for (integer_t j=A->get_ptr()[i]; j<A->get_ptr()[i+1]; j++) {
          auto c = A->get_ind()[j];
          if (c == i) continue;
          auto lc = c - sep_begin;
          if (lc >= 0 && lc < dim_sep && sorder[c]==part && !mark[lc]) {
            mark[lc] = true;
            adjncy.push_back(ind_to_part[lc]);
            e++;
          } else {
            if (opts.separator_ordering_level() > 0) {
              for (integer_t k=A->get_ptr()[c]; k<A->get_ptr()[c+1]; k++) {
                auto cc = A->get_ind()[k];
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
    delete[] mark;
    delete[] ind_to_part;
  }


  // reorder the vertices in the separator to get a better rank structure
  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::separator_reordering
  (const SPOptions<scalar_t>& opts, CSRMatrix<scalar_t,integer_t>* A,
   FrontalMatrix<scalar_t,integer_t>* F) {
    auto N = A->size();
    auto sorder = new integer_t[N];
    std::fill(sorder, sorder+N, integer_t(0));

#pragma omp parallel
#pragma omp single
    F->bisection_partitioning(opts, sorder);

    auto iwork = iperm;
    for (integer_t i=0; i<N; i++) sorder[i] = -sorder[i];
    for (integer_t i=0; i<N; i++) iwork[sorder[i]] = i;
    A->permute(iwork, sorder);

    // product of perm and sep_order
    for (integer_t i=0; i<N; i++) iwork[i] = sorder[perm[i]];
    for (integer_t i=0; i<N; i++) perm[i] = iwork[i];
    for (integer_t i=0; i<N; i++) iperm[perm[i]] = i;

#pragma omp parallel
#pragma omp single
    F->permute_upd_indices(sorder);

    delete[] sorder;
  }

  template<typename scalar_t,typename integer_t> void
  MatrixReordering<scalar_t,integer_t>::nested_dissection_print
  (SPOptions<scalar_t>& opts, integer_t n, integer_t nnz, bool verbose) {
    if (verbose) {
      std::cout << "# initial matrix:" << std::endl;
      std::cout << "#   - number of unknowns = "
                << number_format_with_commas(n) << std::endl;
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
                << number_format_with_commas(sep_tree->separators())
                << std::endl;
      std::cout << "#   - number of levels = "
                << number_format_with_commas(sep_tree->levels()) << std::endl;
    }
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
      sep_tree->printm(filename);
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
