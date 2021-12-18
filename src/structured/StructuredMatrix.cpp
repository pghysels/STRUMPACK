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
#include <cassert>
#include <memory>
#include <functional>
#include <algorithm>

#include "StructuredMatrix.hpp"
#include "HSS/HSSMatrix.hpp"
#if defined(STRUMPACK_USE_ZFP)
#include "sparse/fronts/FrontalMatrixLossy.hpp"
#endif
#include "BLR/BLRMatrix.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "BLR/BLRMatrixMPI.hpp"
#include "sparse/fronts/ExtendAdd.hpp"
#endif
#if defined(STRUMPACK_USE_BPACK)
#include "HODLR/HODLRMatrix.hpp"
#include "HODLR/ButterflyMatrix.hpp"
#endif


namespace strumpack {
  namespace structured {

    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_dense(const DenseMatrix<scalar_t>& A,
                         const StructuredOptions<scalar_t>& opts,
                         const structured::ClusterTree* row_tree,
                         const structured::ClusterTree* col_tree) {
      switch (opts.type()) {
      case Type::HSS: {
        if (A.rows() != A.cols())
          throw std::invalid_argument
            ("HSS compression only supported for square matrices.");
        HSS::HSSOptions<scalar_t> hss_opts(opts);
        if (!row_tree)
          return std::unique_ptr<StructuredMatrix<scalar_t>>
            (new HSS::HSSMatrix<scalar_t>(A, hss_opts));
        else {
          auto H = new HSS::HSSMatrix<scalar_t>(*row_tree, hss_opts);
          H->compress(A, hss_opts);
          return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
        }
      } break;
      case Type::BLR: {
        BLR::BLROptions<scalar_t> blr_opts(opts);
        auto row_leafs = row_tree ? row_tree->leaf_sizes<std::size_t>() :
          structured::ClusterTree(A.rows()).refine(opts.leaf_size()).
          template leaf_sizes<std::size_t>();
        auto col_leafs = col_tree ? col_tree->leaf_sizes<std::size_t>() :
          structured::ClusterTree(A.cols()).refine(opts.leaf_size()).
          template leaf_sizes<std::size_t>();
        return std::unique_ptr<StructuredMatrix<scalar_t>>
          // TODO if square construct as square and do not compress diag
          // pass admissibility matrix?
          (new BLR::BLRMatrix<scalar_t>
           (const_cast<DenseMatrix<scalar_t>&>(A),
            row_leafs, col_leafs, blr_opts));
      } break;
      case Type::LOSSY: {
#if defined(STRUMPACK_USE_ZFP)
        return std::unique_ptr<StructuredMatrix<scalar_t>>
          (new LossyMatrix<scalar_t>(A, 16 /* TODO */));
#else
        throw std::runtime_error
          ("Lossy compression requires ZFP to be enabled.");
#endif
      } break;
      case Type::LOSSLESS: {
#if defined(STRUMPACK_USE_ZFP)
        return std::unique_ptr<StructuredMatrix<scalar_t>>
          (new LossyMatrix<scalar_t>(A, 0));
#else
        throw std::runtime_error
          ("Lossless compression requires ZFP to be enabled.");
#endif
      } break;
      case Type::HODLR:
        throw std::invalid_argument("Type HODLR requires MPI.");
      case Type::HODBF:
        throw std::invalid_argument("Type HODBF requires MPI.");
      case Type::BUTTERFLY:
        throw std::invalid_argument("Type BUTTERFLY requires MPI.");
      case Type::LR:
        throw std::invalid_argument("Type LR requires MPI.");
      }
      return std::unique_ptr<StructuredMatrix<scalar_t>>(nullptr);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_from_dense(const DenseMatrix<float>&,
                         const StructuredOptions<float>&,
                         const structured::ClusterTree*,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_dense(const DenseMatrix<double>&,
                         const StructuredOptions<double>&,
                         const structured::ClusterTree*,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_dense(const DenseMatrix<std::complex<float>>&,
                         const StructuredOptions<std::complex<float>>&,
                         const structured::ClusterTree*,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_dense(const DenseMatrix<std::complex<double>>&,
                         const StructuredOptions<std::complex<double>>&,
                         const structured::ClusterTree*,
                         const structured::ClusterTree*);


    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_dense(int rows, int cols, const scalar_t* A, int ldA,
                         const StructuredOptions<scalar_t>& opts,
                         const structured::ClusterTree* row_tree,
                         const structured::ClusterTree* col_tree) {
      auto M = ConstDenseMatrixWrapperPtr(rows, cols, A, ldA);
      return construct_from_dense(*M, opts, row_tree, col_tree);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_from_dense(int rows, int cols, const float* A, int ldA,
                         const StructuredOptions<float>& opts,
                         const structured::ClusterTree*,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_dense(int rows, int cols, const double* A, int ldA,
                         const StructuredOptions<double>& opts,
                         const structured::ClusterTree*,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_dense(int rows, int cols, const std::complex<float>* A, int ldA,
                         const StructuredOptions<std::complex<float>>& opts,
                         const structured::ClusterTree*,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_dense(int rows, int cols, const std::complex<double>* A, int ldA,
                         const StructuredOptions<std::complex<double>>& opts,
                         const structured::ClusterTree*,
                         const structured::ClusterTree*);


    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(int rows, int cols,
                            const extract_block_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts,
                            const structured::ClusterTree* row_tree,
                            const structured::ClusterTree* col_tree) {
      switch (opts.type()) {
      case Type::HSS: {
        using DenseM_t = DenseMatrix<scalar_t>;
        using DenseMW_t = DenseMatrixWrapper<scalar_t>;
        auto sample =
          [&A, &opts, &rows, &cols]
          (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc) {
            DenseM_t Adense(opts.leaf_size(), opts.leaf_size());
            std::vector<std::size_t> I, J;
            int B = opts.leaf_size();
            I.reserve(B);
            J.reserve(B);
            Sr.zero();
            Sc.zero();
            // TODO threading
            for (int r=0; r<rows; r+=B) {
              for (int c=0; c<cols; c+=B) {
                int m = std::min(B, rows - r),
                  n = std::min(B, cols - c);
                DenseMW_t Asub(m, n, Adense, 0, 0);
                I.resize(m);
                J.resize(n);
                std::iota(I.begin(), I.end(), r);
                std::iota(J.begin(), J.end(), c);
                A(I, J, Asub);
                DenseMW_t Rrsub(n, Rr.cols(), Rr, c, 0),
                  Srsub(m, Sr.cols(), Sr, r, 0),
                  Rcsub(m, Rc.cols(), Rc, r, 0),
                  Scsub(n, Sc.cols(), Sc, c, 0);
                gemm(Trans::N, Trans::N, scalar_t(1.), Asub, Rrsub,
                     scalar_t(1.), Srsub);
                gemm(Trans::C, Trans::N, scalar_t(1.), Asub, Rcsub,
                     scalar_t(1.), Scsub);
              }
            }
          };
        HSS::HSSOptions<scalar_t> hss_opts(opts);
        auto H = row_tree ?
          new HSS::HSSMatrix<scalar_t>(*row_tree, hss_opts) :
          new HSS::HSSMatrix<scalar_t>(rows, cols, hss_opts);
        H->compress(sample, A, hss_opts);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
      } break;
      case Type::BLR: {
        // TODO
        throw std::logic_error
          ("BLR compression from elements not implemented yet.");
      } break;
      case Type::HODLR:
        throw std::invalid_argument("Type HODLR requires MPI.");
      case Type::HODBF:
        throw std::invalid_argument("Type HODBF requires MPI.");
      case Type::BUTTERFLY:
        throw std::invalid_argument("Type BUTTERFLY requires MPI.");
      case Type::LR:
        throw std::invalid_argument("Type LR requires MPI.");
      case Type::LOSSY:
        throw std::invalid_argument
          ("Type LOSSY does not support construction from elements.");
      case Type::LOSSLESS:
        throw std::invalid_argument
          ("Type LOSSLESS does not support construction from elements.");
      }
      return std::unique_ptr<StructuredMatrix<scalar_t>>(nullptr);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_from_elements(int rows, int cols,
                            const extract_block_t<float>& A,
                            const StructuredOptions<float>& opts,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_elements(int rows, int cols,
                            const extract_block_t<double>& A,
                            const StructuredOptions<double>& opts,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_elements(int rows, int cols,
                            const extract_block_t<std::complex<float>>& A,
                            const StructuredOptions<std::complex<float>>& opts,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_elements(int rows, int cols,
                            const extract_block_t<std::complex<double>>& A,
                            const StructuredOptions<std::complex<double>>& opts,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);



    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(int rows, int cols,
                            const extract_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts,
                            const structured::ClusterTree* row_tree,
                            const structured::ClusterTree* col_tree) {
      auto extract_block =
        [&A](const std::vector<std::size_t>& I,
             const std::vector<std::size_t>& J,
             DenseMatrix<scalar_t>& B) {
          for (std::size_t j=0; j<J.size(); j++)
            for (std::size_t i=0; i<I.size(); i++)
              B(i, j) = A(I[i], J[j]);
        };
      return construct_from_elements<scalar_t>
        (rows, cols, extract_block, opts, row_tree, col_tree);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_from_elements(int, int, const extract_t<float>&,
                            const StructuredOptions<float>&,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_elements(int, int, const extract_t<double>&,
                            const StructuredOptions<double>&,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_elements(int, int, const extract_t<std::complex<float>>&,
                            const StructuredOptions<std::complex<float>>&,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_elements(int, int, const extract_t<std::complex<double>>&,
                            const StructuredOptions<std::complex<double>>&,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);



    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_matrix_free(int rows, int cols,
                          const mult_t<scalar_t>& Amult,
                          const StructuredOptions<scalar_t>& opts,
                          const structured::ClusterTree* row_tree,
                          const structured::ClusterTree* col_tree) {
      switch (opts.type()) {
      case Type::HSS:
        throw std::invalid_argument
          ("Type HSS does not support matrix-free compression.");
      case Type::BLR:
        throw std::invalid_argument
          ("Type BLR does not support matrix-free compression.");
      case Type::HODLR:
        throw std::invalid_argument("Type HODLR requires MPI.");
      case Type::HODBF:
        throw std::invalid_argument("Type HODBF requires MPI.");
      case Type::BUTTERFLY:
        throw std::invalid_argument("Type BUTTERFLY requires MPI.");
      case Type::LR:
        throw std::invalid_argument("Type LR requires MPI.");
      case Type::LOSSY:
        throw std::invalid_argument
          ("Type LOSSY does not support matrix-free compression.");
      case Type::LOSSLESS:
        throw std::invalid_argument
          ("Type LOSSLESS does not support matrix-free compression.");
      }
      return std::unique_ptr<StructuredMatrix<scalar_t>>(nullptr);
    }


    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_matrix_free(int, int, const mult_t<float>&,
                          const StructuredOptions<float>&,
                          const structured::ClusterTree*,
                          const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_matrix_free(int, int, const mult_t<double>&,
                          const StructuredOptions<double>&,
                          const structured::ClusterTree*,
                          const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_matrix_free(int, int, const mult_t<std::complex<float>>&,
                          const StructuredOptions<std::complex<float>>&,
                          const structured::ClusterTree*,
                          const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_matrix_free(int, int, const mult_t<std::complex<double>>&,
                          const StructuredOptions<std::complex<double>>&,
                          const structured::ClusterTree*,
                          const structured::ClusterTree*);



    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<scalar_t>& Amult,
                                    const extract_block_t<scalar_t>& Aelem,
                                    const StructuredOptions<scalar_t>& opts,
                                    const structured::ClusterTree* row_tree,
                                    const structured::ClusterTree* col_tree) {
      switch (opts.type()) {
      case Type::HSS: {
        HSS::HSSOptions<scalar_t> hss_opts(opts);
        auto H = row_tree ?
          new HSS::HSSMatrix<scalar_t>(*row_tree, hss_opts) :
          new HSS::HSSMatrix<scalar_t>(rows, cols, hss_opts);
        using DenseM_t = DenseMatrix<scalar_t>;
        auto sample =
          [&Amult](DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc) {
            Amult(Trans::N, Rr, Sr);
            Amult(Trans::C, Rc, Sc);
          };
        H->compress(sample, Aelem, hss_opts);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
      } break;
      case Type::BLR: {
        return construct_from_elements<scalar_t>(rows, cols, Aelem, opts);
      } break;
      case Type::HODLR:
        throw std::invalid_argument("Type HODLR requires MPI.");
      case Type::HODBF:
        throw std::invalid_argument("Type HODBF requires MPI.");
      case Type::BUTTERFLY:
        throw std::invalid_argument("Type BUTTERFLY requires MPI.");
      case Type::LR:
        throw std::invalid_argument("Type LR requires MPI.");
      case Type::LOSSY:
        throw std::invalid_argument
          ("Type LOSSY does not support partially matrix-free compression.");
      case Type::LOSSLESS:
        throw std::invalid_argument
          ("Type LOSSLESS does not support partially matrix-free compression.");
      }
      return std::unique_ptr<StructuredMatrix<scalar_t>>(nullptr);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_partially_matrix_free(int, int, const mult_t<float>&,
                                    const extract_block_t<float>&,
                                    const StructuredOptions<float>&,
                                    const structured::ClusterTree*,
                                    const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_partially_matrix_free(int, int, const mult_t<double>&,
                                    const extract_block_t<double>&,
                                    const StructuredOptions<double>&,
                                    const structured::ClusterTree*,
                                    const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_partially_matrix_free(int, int, const mult_t<std::complex<float>>&,
                                    const extract_block_t<std::complex<float>>&,
                                    const StructuredOptions<std::complex<float>>&,
                                    const structured::ClusterTree*,
                                    const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_partially_matrix_free(int, int, const mult_t<std::complex<double>>&,
                                    const extract_block_t<std::complex<double>>&,
                                    const StructuredOptions<std::complex<double>>&,
                                    const structured::ClusterTree*,
                                    const structured::ClusterTree*);



    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<scalar_t>& Amult,
                                    const extract_t<scalar_t>& Aelem,
                                    const StructuredOptions<scalar_t>& opts,
                                    const structured::ClusterTree* row_tree,
                                    const structured::ClusterTree* col_tree) {
      auto extract_block =
        [&Aelem](const std::vector<std::size_t>& I,
                 const std::vector<std::size_t>& J,
                 DenseMatrix<scalar_t>& B) {
          for (std::size_t j=0; j<J.size(); j++)
            for (std::size_t i=0; i<I.size(); i++)
              B(i, j) = Aelem(I[i], J[j]);
        };
      return construct_partially_matrix_free<scalar_t>
        (rows, cols, Amult, extract_block, opts, row_tree, col_tree);
    }
    template std::unique_ptr<StructuredMatrix<float>>
    construct_partially_matrix_free(int, int, const mult_t<float>&,
                                    const extract_t<float>&,
                                    const StructuredOptions<float>&,
                                    const structured::ClusterTree*,
                                    const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_partially_matrix_free(int, int, const mult_t<double>&,
                                    const extract_t<double>&,
                                    const StructuredOptions<double>&,
                                    const structured::ClusterTree*,
                                    const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_partially_matrix_free(int, int, const mult_t<std::complex<float>>&,
                                    const extract_t<std::complex<float>>&,
                                    const StructuredOptions<std::complex<float>>&,
                                    const structured::ClusterTree*,
                                    const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_partially_matrix_free(int, int, const mult_t<std::complex<double>>&,
                                    const extract_t<std::complex<double>>&,
                                    const StructuredOptions<std::complex<double>>&,
                                    const structured::ClusterTree*,
                                    const structured::ClusterTree*);

#if defined(STRUMPACK_USE_MPI)
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_dense(const DistributedMatrix<scalar_t>& A,
                         const StructuredOptions<scalar_t>& opts,
                         const structured::ClusterTree* row_tree,
                         const structured::ClusterTree* col_tree) {
#if defined(STRUMPACK_USE_BPACK)
      auto Ablocks =
        [&A](const std::vector<std::vector<std::size_t>>& I,
             const std::vector<std::vector<std::size_t>>& J,
             std::vector<DistributedMatrixWrapper<scalar_t>>& B,
             HODLR::ExtractionMeta&) {
          auto nB = I.size();
          assert(I.size() == J.size());
          for (std::size_t i=0; i<nB; i++) {
            auto Bi = A.extract(I[i], J[i]);
            copy(I[i].size(), J[i].size(), Bi, 0, 0,
                 B[i], 0, 0, A.ctxt_all());
          }
        };
#endif
      switch (opts.type()) {
      case Type::HSS: {
        HSS::HSSOptions<scalar_t> hss_opts(opts);
        return std::unique_ptr<StructuredMatrix<scalar_t>>
          (row_tree ?
           new HSS::HSSMatrixMPI<scalar_t>(*row_tree, A, hss_opts) :
           new HSS::HSSMatrixMPI<scalar_t>(A, hss_opts));
      } break;
      case Type::BLR:
        throw std::logic_error("Not implemented yet.");
      case Type::HODLR: {
#if defined(STRUMPACK_USE_BPACK)
        if (A.rows() != A.cols())
          throw std::invalid_argument
            ("HODLR compression only supported for square matrices.");
        structured::ClusterTree t(A.rows());
        t.refine(opts.leaf_size());
        HODLR::HODLROptions<scalar_t> hodlr_opts(opts);
        auto H = new HODLR::HODLRMatrix<scalar_t>
          (A.Comm(), t, hodlr_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("HODLR compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::HODBF: {
#if defined(STRUMPACK_USE_BPACK)
        if (A.rows() != A.cols())
          throw std::invalid_argument
            ("HODBF compression only supported for square matrices.");
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        hodbf_opts.set_butterfly_levels(1000);
        hodbf_opts.set_BF_entry_n15(true);
        auto H = new HODLR::HODLRMatrix<scalar_t>
          (A.Comm(),
           row_tree ? *row_tree :
           structured::ClusterTree(A.rows()).refine(opts.leaf_size()),
           hodbf_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("HODBF compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::BUTTERFLY: {
#if defined(STRUMPACK_USE_BPACK)
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        hodbf_opts.set_butterfly_levels(1000);
        hodbf_opts.set_BF_entry_n15(true);
        auto H = new HODLR::ButterflyMatrix<scalar_t>
          (A.Comm(),
           row_tree ? *row_tree :
           structured::ClusterTree(A.rows()).refine(opts.leaf_size()),
           col_tree ? *col_tree :
           (row_tree ? *row_tree :
            structured::ClusterTree(A.cols()).refine(opts.leaf_size())),
           hodbf_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("BUTTERFLY compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::LR: {
#if defined(STRUMPACK_USE_BPACK)
        structured::ClusterTree tr(A.rows()), tc(A.cols());
        HODLR::HODLROptions<scalar_t> lr_opts(opts);
        lr_opts.set_butterfly_levels(0);
        lr_opts.set_BF_entry_n15(true);
        auto H = new HODLR::ButterflyMatrix<scalar_t>
          (A.Comm(),
           structured::ClusterTree(A.rows()),
           structured::ClusterTree(A.cols()),
           lr_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("LR compression requires ButterflyPACK to be enabled.");
#endif
      }
      case Type::LOSSY:
        throw std::logic_error("Not implemented yet.");
      case Type::LOSSLESS:
        throw std::logic_error("Not implemented yet.");
      }
      return std::unique_ptr<StructuredMatrix<scalar_t>>(nullptr);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_from_dense(const DistributedMatrix<float>&,
                         const StructuredOptions<float>&,
                         const structured::ClusterTree*,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_dense(const DistributedMatrix<double>&,
                         const StructuredOptions<double>&,
                         const structured::ClusterTree*,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_dense(const DistributedMatrix<std::complex<float>>& A,
                         const StructuredOptions<std::complex<float>>& opts,
                         const structured::ClusterTree*,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_dense(const DistributedMatrix<std::complex<double>>& A,
                         const StructuredOptions<std::complex<double>>& opts,
                         const structured::ClusterTree*,
                         const structured::ClusterTree*);


    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_dist_block_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts,
                            const structured::ClusterTree* row_tree,
                            const structured::ClusterTree* col_tree) {
      switch (opts.type()) {
      case Type::HSS:
        throw std::logic_error("Not implemented yet.");
      case Type::BLR:
        throw std::logic_error("Not implemented yet.");
      case Type::HODLR: {
#if defined(STRUMPACK_USE_BPACK)
        if (rows != cols)
          throw std::invalid_argument
            ("HODLR compression only supported for square matrices.");
        throw std::logic_error("Not implemented yet.");
#else
        throw std::runtime_error
          ("HODLR compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::HODBF:
#if defined(STRUMPACK_USE_BPACK)
        throw std::logic_error("Not implemented yet.");
#else
        throw std::runtime_error
          ("HODBF compression requires ButterflyPACK to be enabled.");
#endif
      case Type::BUTTERFLY:
#if defined(STRUMPACK_USE_BPACK)
        throw std::logic_error("Not implemented yet.");
#else
        throw std::runtime_error
          ("BUTTERFLY compression requires ButterflyPACK to be enabled.");
#endif
      case Type::LR:
#if defined(STRUMPACK_USE_BPACK)
        throw std::logic_error("Not implemented yet.");
#else
        throw std::runtime_error
          ("LR compression requires ButterflyPACK to be enabled.");
#endif
      case Type::LOSSY:
        throw std::invalid_argument
          ("Type LOSSY does not support compression from elements.");
      case Type::LOSSLESS:
        throw std::invalid_argument
          ("Type LOSSLESS does not support compression from elements.");
      }
      return std::unique_ptr<StructuredMatrix<scalar_t>>(nullptr);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_from_elements(const MPIComm&, int, int,
                            const extract_dist_block_t<float>&,
                            const StructuredOptions<float>&,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_elements(const MPIComm&, int, int,
                            const extract_dist_block_t<double>&,
                            const StructuredOptions<double>&,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_elements(const MPIComm&, int, int,
                            const extract_dist_block_t<std::complex<float>>&,
                            const StructuredOptions<std::complex<float>>&,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_elements(const MPIComm&, int, int,
                            const extract_dist_block_t<std::complex<double>>&,
                            const StructuredOptions<std::complex<double>>&,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);


    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts,
                            const structured::ClusterTree* row_tree,
                            const structured::ClusterTree* col_tree) {
#if defined(STRUMPACK_USE_BPACK)
      auto Ablocks =
        [&A](const std::vector<std::vector<std::size_t>>& I,
             const std::vector<std::vector<std::size_t>>& J,
             std::vector<DistributedMatrixWrapper<scalar_t>>& B,
             HODLR::ExtractionMeta&) {
          for (std::size_t i=0; i<I.size(); i++) {
            auto Afill = [&](std::size_t r, std::size_t c) -> scalar_t {
                           return A(I[i][r], J[i][c]);
                         };
            B[i].fill(Afill);
          }
        };
#endif
      switch (opts.type()) {
      case Type::HSS:
        throw std::logic_error("Not implemented yet.");
      case Type::BLR:
        throw std::logic_error("Not implemented yet.");
      case Type::HODLR: {
#if defined(STRUMPACK_USE_BPACK)
        if (rows != cols)
          throw std::invalid_argument
            ("HODLR compression only supported for square matrices.");
        HODLR::HODLROptions<scalar_t> hodlr_opts(opts);
        hodlr_opts.set_BF_entry_n15(true);
        auto H = new HODLR::HODLRMatrix<scalar_t>
          (comm, row_tree ? *row_tree :
           structured::ClusterTree(rows).refine(opts.leaf_size()),
           hodlr_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("HODLR compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::HODBF: {
#if defined(STRUMPACK_USE_BPACK)
        if (rows != cols)
          throw std::invalid_argument
            ("HODBF compression only supported for square matrices.");
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        hodbf_opts.set_butterfly_levels(1000);
        hodbf_opts.set_BF_entry_n15(true);
        auto H = new HODLR::HODLRMatrix<scalar_t>
          (comm, row_tree ? *row_tree :
           structured::ClusterTree(rows).refine(opts.leaf_size()),
           hodbf_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("HODBF compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::BUTTERFLY: {
#if defined(STRUMPACK_USE_BPACK)
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        hodbf_opts.set_butterfly_levels(1000);
        hodbf_opts.set_BF_entry_n15(true);
        auto H = new HODLR::ButterflyMatrix<scalar_t>
          (comm,
           row_tree ? *row_tree : structured::ClusterTree(rows).refine(opts.leaf_size()),
           col_tree ? *col_tree : (row_tree ? *row_tree :
                                   structured::ClusterTree(cols).refine(opts.leaf_size())),
           hodbf_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("BUTTERFLY compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::LR: {
#if defined(STRUMPACK_USE_BPACK)
        structured::ClusterTree tr(rows), tc(cols);
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        auto H = new HODLR::ButterflyMatrix<scalar_t>
          (comm, structured::ClusterTree(rows),
           structured::ClusterTree(cols), hodbf_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("LR compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::LOSSY:
        throw std::invalid_argument
          ("Type LOSSY does not support compression from elements.");
      case Type::LOSSLESS:
        throw std::invalid_argument
          ("Type LOSSLESS does not support compression from elements.");
      }
      return std::unique_ptr<StructuredMatrix<scalar_t>>(nullptr);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_from_elements(const MPIComm&, int, int, const extract_t<float>&,
                            const StructuredOptions<float>&,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_elements(const MPIComm&, int, int, const extract_t<double>&,
                            const StructuredOptions<double>&,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_elements(const MPIComm&, int, int,
                            const extract_t<std::complex<float>>&,
                            const StructuredOptions<std::complex<float>>&,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_elements(const MPIComm&, int, int,
                            const extract_t<std::complex<double>>&,
                            const StructuredOptions<std::complex<double>>&,
                            const structured::ClusterTree*,
                            const structured::ClusterTree*);



    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_matrix_free(const MPIComm& comm, const BLACSGrid* g,
                          int rows, int cols,
                          const mult_2d_t<scalar_t>& A,
                          const StructuredOptions<scalar_t>& opts,
                          const structured::ClusterTree* row_tree,
                          const structured::ClusterTree* col_tree) {
      switch (opts.type()) {
      case Type::HSS:
        throw std::invalid_argument
          ("Type HSS does not support matrix-free compression.");
      case Type::BLR:
        throw std::invalid_argument
          ("Type BLR does not support matrix-free compression.");
      case Type::HODLR: {
#if defined(STRUMPACK_USE_BPACK)
        if (rows != cols)
          throw std::invalid_argument
            ("HODLR compression only supported for square matrices.");
        HODLR::HODLROptions<scalar_t> hodlr_opts(opts);
        hodlr_opts.set_BF_entry_n15(true);
        auto H = new HODLR::HODLRMatrix<scalar_t>
          (comm, row_tree ? *row_tree :
           structured::ClusterTree(rows).refine(opts.leaf_size()),
           hodlr_opts);
        auto Tmult = [&A, &H, &g, &rows, &cols]
          (Trans op, const DenseMatrix<scalar_t>& R,
           DenseMatrix<scalar_t>& S) {
                       DistributedMatrix<scalar_t>
                         R2d(g, op == Trans::N ? cols : rows, R.cols()),
                         S2d(g, op == Trans::N ? rows : cols, R.cols());
                       H->redistribute_1D_to_2D(R, R2d);
                       A(op, R2d, S2d);
                       H->redistribute_2D_to_1D(S2d, S);
                     };
        H->compress(Tmult);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("HODLR compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::HODBF: {
#if defined(STRUMPACK_USE_BPACK)
        if (rows != cols)
          throw std::invalid_argument
            ("HODBF compression only supported for square matrices.");
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        hodbf_opts.set_butterfly_levels(1000);
        hodbf_opts.set_BF_entry_n15(true);
        auto H = new HODLR::HODLRMatrix<scalar_t>
          (comm, row_tree ? *row_tree :
           structured::ClusterTree(rows).refine(opts.leaf_size()),
           hodbf_opts);
        auto Tmult = [&A, &H, &g, &rows, &cols]
          (Trans op, const DenseMatrix<scalar_t>& R,
           DenseMatrix<scalar_t>& S) {
                       DistributedMatrix<scalar_t>
                         R2d(g, op == Trans::N ? cols : rows, R.cols()),
                         S2d(g, op == Trans::N ? rows : cols, R.cols());
                       H->redistribute_1D_to_2D(R, R2d);
                       A(op, R2d, S2d);
                       H->redistribute_2D_to_1D(S2d, S);
                     };
        H->compress(Tmult);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("HODBF compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::BUTTERFLY: {
#if defined(STRUMPACK_USE_BPACK)
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        hodbf_opts.set_butterfly_levels(1000);
        hodbf_opts.set_BF_entry_n15(true);
        auto H = new HODLR::ButterflyMatrix<scalar_t>
          (comm,
           row_tree ? *row_tree :
           structured::ClusterTree(rows).refine(opts.leaf_size()),
           col_tree ? *col_tree :
           (row_tree ? *row_tree :
            structured::ClusterTree(cols).refine(opts.leaf_size())),
           hodbf_opts);
        auto Tmult = [&A, &H, &g, &rows, &cols]
          (Trans op, scalar_t alpha, const DenseMatrix<scalar_t>& R,
           scalar_t beta, DenseMatrix<scalar_t>& S) {
                       DistributedMatrix<scalar_t>
                         R2d(g, op == Trans::N ? cols : rows, R.cols()),
                         S2d(g, op == Trans::N ? rows : cols, R.cols());
                       H->redistribute_1D_to_2D
                         (R, R2d, op == Trans::N ? H->cdist() : H->rdist());
                       A(op, R2d, S2d);
                       H->redistribute_2D_to_1D
                         (alpha, S2d, beta, S,
                          op == Trans::N ? H->rdist() : H->cdist());
                     };
        H->compress(Tmult);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("BUTTERFLY compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::LR: {
#if defined(STRUMPACK_USE_BPACK)
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        auto H = new HODLR::ButterflyMatrix<scalar_t>
          (comm, structured::ClusterTree(rows),
           structured::ClusterTree(cols), hodbf_opts);
        auto Tmult = [&A, &H, &g, &rows, &cols]
          (Trans op, scalar_t alpha,
           const DenseMatrix<scalar_t>& R,
           scalar_t beta, DenseMatrix<scalar_t>& S) {
                       DistributedMatrix<scalar_t>
                         R2d(g, op == Trans::N ? cols : rows, R.cols()),
                         S2d(g, op == Trans::N ? rows : cols, R.cols());
                       H->redistribute_1D_to_2D
                         (R, R2d, op == Trans::N ? H->cdist() : H->rdist());
                       A(op, R2d, S2d);
                       H->redistribute_2D_to_1D
                         (alpha, S2d, beta, S,
                          op == Trans::N ? H->rdist() : H->cdist());
                     };
        H->compress(Tmult);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("LR compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::LOSSY:
        throw std::invalid_argument
          ("Type LOSSY does not support matrix-free compression.");
      case Type::LOSSLESS:
        throw std::invalid_argument
          ("Type LOSSLESS does not support matrix-free compression.");
      }
      return std::unique_ptr<StructuredMatrix<scalar_t>>(nullptr);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_matrix_free(const MPIComm&, const BLACSGrid*, int, int,
                          const mult_2d_t<float>&,
                          const StructuredOptions<float>&,
                          const structured::ClusterTree*,
                          const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_matrix_free(const MPIComm&, const BLACSGrid*, int, int,
                          const mult_2d_t<double>&,
                          const StructuredOptions<double>&,
                          const structured::ClusterTree*,
                          const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_matrix_free(const MPIComm&, const BLACSGrid*, int, int,
                          const mult_2d_t<std::complex<float>>&,
                          const StructuredOptions<std::complex<float>>&,
                          const structured::ClusterTree*,
                          const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_matrix_free(const MPIComm&, const BLACSGrid*, int, int,
                          const mult_2d_t<std::complex<double>>&,
                          const StructuredOptions<std::complex<double>>&,
                          const structured::ClusterTree*,
                          const structured::ClusterTree*);



    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_matrix_free(const MPIComm& comm, int rows, int cols,
                          const mult_1d_t<scalar_t>& A,
                          const StructuredOptions<scalar_t>& opts,
                          const structured::ClusterTree* row_tree,
                          const structured::ClusterTree* col_tree) {
      switch (opts.type()) {
      case Type::HSS:
        throw std::invalid_argument
          ("Type HSS does not support matrix-free compression.");
      case Type::BLR:
        throw std::invalid_argument
          ("Type BLR does not support matrix-free compression.");
      case Type::HODLR: {
        if (rows != cols)
          throw std::invalid_argument
            ("HODLR compression only supported for square matrices.");
#if defined(STRUMPACK_USE_BPACK)
        HODLR::HODLROptions<scalar_t> hodlr_opts(opts);
        hodlr_opts.set_BF_entry_n15(true);
        auto H = new HODLR::HODLRMatrix<scalar_t>
          (comm, row_tree ? *row_tree :
           structured::ClusterTree(rows).refine(opts.leaf_size()),
           hodlr_opts);
        auto Tmult = [&A, &H]
          (Trans op, const DenseMatrix<scalar_t>& R,
           DenseMatrix<scalar_t>& S) {
                       A(op, R, S, H->dist(), H->dist());
                     };
        H->compress(Tmult);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("HODLR compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::HODBF: {
        if (rows != cols)
          throw std::invalid_argument
            ("HODBF compression only supported for square matrices.");
#if defined(STRUMPACK_USE_BPACK)
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        hodbf_opts.set_butterfly_levels(1000);
        hodbf_opts.set_BF_entry_n15(true);
        auto H = new HODLR::HODLRMatrix<scalar_t>
          (comm, row_tree ? *row_tree :
           structured::ClusterTree(rows).refine(opts.leaf_size()),
           hodbf_opts);
        auto Tmult = [&A, &H]
          (Trans op, const DenseMatrix<scalar_t>& R,
           DenseMatrix<scalar_t>& S) {
                       A(op, R, S, H->dist(), H->dist());
                     };
        H->compress(Tmult);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("HODBF compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::BUTTERFLY: {
#if defined(STRUMPACK_USE_BPACK)
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        hodbf_opts.set_butterfly_levels(1000);
        hodbf_opts.set_BF_entry_n15(true);
        auto H = new HODLR::ButterflyMatrix<scalar_t>
          (comm,
           row_tree ? *row_tree :
           structured::ClusterTree(rows).refine(opts.leaf_size()),
           col_tree ? *col_tree :
           (row_tree ? *row_tree :
            structured::ClusterTree(cols).refine(opts.leaf_size())),
           hodbf_opts);
        auto Tmult = [&A, &H]
          (Trans op, scalar_t alpha, const DenseMatrix<scalar_t>& R,
           scalar_t beta, DenseMatrix<scalar_t>& S) {
                       assert(alpha == scalar_t(1.));
                       assert(beta == scalar_t(0.));
                       A(op, R, S, H->rdist(), H->cdist());
                     };
        H->compress(Tmult);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("BUTTERFLY compression requires ButterflyPACK to be enabled.");
#endif
      } break;
      case Type::LR: {
#if defined(STRUMPACK_USE_BPACK)
        structured::ClusterTree tr(rows), tc(cols);
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        auto H = new HODLR::ButterflyMatrix<scalar_t>
          (comm, structured::ClusterTree(rows),
           structured::ClusterTree(cols), hodbf_opts);
        auto Tmult = [&A, &H]
          (Trans op, scalar_t alpha,
           const DenseMatrix<scalar_t>& R,
           scalar_t beta, DenseMatrix<scalar_t>& S) {
                       assert(alpha == scalar_t(1.));
                       assert(beta == scalar_t(0.));
                       A(op, R, S, H->rdist(), H->cdist());
                     };
        H->compress(Tmult);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
#else
        throw std::runtime_error
          ("LR compression requires ButterflyPACK to be enabled.");
#endif

      } break;
      case Type::LOSSY:
        throw std::invalid_argument
          ("Type LOSSY does not support matrix-free compression.");
      case Type::LOSSLESS:
        throw std::invalid_argument
          ("Type LOSSLESS does not support matrix-free compression.");
      }
      return std::unique_ptr<StructuredMatrix<scalar_t>>(nullptr);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_matrix_free(const MPIComm&, int, int, const mult_1d_t<float>&,
                          const StructuredOptions<float>&,
                          const structured::ClusterTree*,
                          const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_matrix_free(const MPIComm&, int, int, const mult_1d_t<double>&,
                          const StructuredOptions<double>&,
                          const structured::ClusterTree*,
                          const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_matrix_free(const MPIComm&, int, int,
                          const mult_1d_t<std::complex<float>>&,
                          const StructuredOptions<std::complex<float>>&,
                          const structured::ClusterTree*,
                          const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_matrix_free(const MPIComm&, int, int,
                          const mult_1d_t<std::complex<double>>&,
                          const StructuredOptions<std::complex<double>>&,
                          const structured::ClusterTree*,
                          const structured::ClusterTree*);
#endif


    template<typename scalar_t> void
    StructuredMatrix<scalar_t>::mult(Trans op, const DenseMatrix<scalar_t>& x,
                                     DenseMatrix<scalar_t>& y) const {
      throw std::invalid_argument
        ("Operation mult not supported for this type.");
    }
    template<typename scalar_t> void
    StructuredMatrix<scalar_t>::mult(Trans op,
                                     int m, const scalar_t* x, int ldx,
                                     scalar_t* y, int ldy) const {
      DenseMatrixWrapper<scalar_t>
        Y(op == Trans::N ? rows() : cols(), m, y, ldy);
      auto X = ConstDenseMatrixWrapperPtr
        (op == Trans::N ? cols() : rows(), m, x, ldx);
      mult(op, *X, Y);
    }

#if defined(STRUMPACK_USE_MPI)
    template<typename scalar_t> void
    StructuredMatrix<scalar_t>::mult(Trans op,
                                     const DistributedMatrix<scalar_t>& x,
                                     DistributedMatrix<scalar_t>& y) const {
      throw std::invalid_argument
        ("Operation mult(Dist) not supported for this type.");
    }
#endif

    template<typename scalar_t> void
    StructuredMatrix<scalar_t>::factor() {
      throw std::invalid_argument
        ("Operation factor not supported for this type.");
    }
    template<typename scalar_t> void
    StructuredMatrix<scalar_t>::solve(DenseMatrix<scalar_t>& b) const {
      throw std::invalid_argument
        ("Operation solve not supported for this type.");
    }
#if defined(STRUMPACK_USE_MPI)
    template<typename scalar_t> void
    StructuredMatrix<scalar_t>::solve(DistributedMatrix<scalar_t>& b) const {
      throw std::invalid_argument
        ("Operation solve(Dist) not supported for this type.");
    }
#endif

    template<typename scalar_t> void
    StructuredMatrix<scalar_t>::shift(scalar_t s) {
      throw std::invalid_argument
        ("Operation shift not supported for this type.");
    }


    // explicit template instantiations
    template class StructuredMatrix<float>;
    template class StructuredMatrix<double>;
    template class StructuredMatrix<std::complex<float>>;
    template class StructuredMatrix<std::complex<double>>;

  } // end namespace structured
} // end namespace strumpack

