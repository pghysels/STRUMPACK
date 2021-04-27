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
#include "HODLR/HODLRMatrix.hpp"
#include "HODLR/ButterflyMatrix.hpp"
#include "sparse/fronts/FrontalMatrixLossy.hpp"
#include "sparse/fronts/ExtendAdd.hpp"

namespace strumpack {
  namespace structured {

    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_dense(const DenseMatrix<scalar_t>& A,
                         const StructuredOptions<scalar_t>& opts,
                         const structured::ClusterTree* tree) {
      switch (opts.type()) {
      case Type::HSS: {
        if (A.rows() != A.cols())
          throw std::invalid_argument
            ("HSS compression only supported for square matrices.");
        HSS::HSSOptions<scalar_t> hss_opts(opts);
        if (!tree)
          return std::unique_ptr<StructuredMatrix<scalar_t>>
            (new HSS::HSSMatrix<scalar_t>(A, hss_opts));
        else {
          auto H = new HSS::HSSMatrix<scalar_t>(*tree, hss_opts);
          H->compress(A, hss_opts);
          return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
        }
      } break;
      case Type::BLR: {
        structured::ClusterTree tr(A.rows()), tc(A.cols());
        tr.refine(opts.leaf_size());
        tc.refine(opts.leaf_size());
        BLR::BLROptions<scalar_t> blr_opts(opts);
        return std::unique_ptr<StructuredMatrix<scalar_t>>
          // TODO if square construct as square and do not compress diag
          // pass admissibility matrix?
          (new BLR::BLRMatrix<scalar_t>
           (const_cast<DenseMatrix<scalar_t>&>(A),
            tr.leaf_sizes<std::size_t>(),
            tr.leaf_sizes<std::size_t>(), blr_opts));
      } break;
      case Type::LOSSY: {
#if defined(STRUMPACK_USE_ZFP)
        return std::unique_ptr<StructuredMatrix<scalar_t>>
          (new LossyMatrix<scalar_t>(A, 16 /* TODO */));
#else
        throw std::exception
          ("Lossy compression requires ZFP to be enabled.");
#endif
      } break;
      case Type::LOSSLESS: {
#if defined(STRUMPACK_USE_ZFP)
        return std::unique_ptr<StructuredMatrix<scalar_t>>
          (new LossyMatrix<scalar_t>(A, 0));
#else
        throw std::exception
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
      default:
        throw std::invalid_argument("Unknown StructuredMatrix type.");
      }
      return std::unique_ptr<StructuredMatrix<scalar_t>>(nullptr);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_from_dense(const DenseMatrix<float>& A,
                         const StructuredOptions<float>& opts,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_dense(const DenseMatrix<double>& A,
                         const StructuredOptions<double>& opts,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_dense(const DenseMatrix<std::complex<float>>& A,
                         const StructuredOptions<std::complex<float>>& opts,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_dense(const DenseMatrix<std::complex<double>>& A,
                         const StructuredOptions<std::complex<double>>& opts,
                         const structured::ClusterTree*);


    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_dense(int rows, int cols, const scalar_t* A, int ldA,
                         const StructuredOptions<scalar_t>& opts,
                         const structured::ClusterTree* tree) {
      auto M = ConstDenseMatrixWrapperPtr(rows, cols, A, ldA);
      return construct_from_dense(*M, opts, tree);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_from_dense(int rows, int cols, const float* A, int ldA,
                         const StructuredOptions<float>& opts,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_dense(int rows, int cols, const double* A, int ldA,
                         const StructuredOptions<double>& opts,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_dense(int rows, int cols, const std::complex<float>* A, int ldA,
                         const StructuredOptions<std::complex<float>>& opts,
                         const structured::ClusterTree*);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_dense(int rows, int cols, const std::complex<double>* A, int ldA,
                         const StructuredOptions<std::complex<double>>& opts,
                         const structured::ClusterTree*);


    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(int rows, int cols,
                            const extract_block_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts) {
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
        auto H = new HSS::HSSMatrix<scalar_t>(rows, cols, hss_opts);
        H->compress(sample, A, hss_opts);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
      } break;
      case Type::BLR: {
        // TODO
        throw std::invalid_argument("Not implemented yet.");
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
                            const StructuredOptions<float>& opts);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_elements(int rows, int cols,
                            const extract_block_t<double>& A,
                            const StructuredOptions<double>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_elements(int rows, int cols,
                            const extract_block_t<std::complex<float>>& A,
                            const StructuredOptions<std::complex<float>>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_elements(int rows, int cols,
                            const extract_block_t<std::complex<double>>& A,
                            const StructuredOptions<std::complex<double>>& opts);



    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(int rows, int cols,
                            const extract_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts) {
      auto extract_block =
        [&A](const std::vector<std::size_t>& I,
             const std::vector<std::size_t>& J,
             DenseMatrix<scalar_t>& B) {
          for (std::size_t j=0; j<J.size(); j++)
            for (std::size_t i=0; i<I.size(); i++)
              B(i, j) = A(I[i], J[j]);
        };
      return construct_from_elements<scalar_t>
        (rows, cols, extract_block, opts);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_from_elements(int rows, int cols,
                            const extract_t<float>& A,
                            const StructuredOptions<float>& opts);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_elements(int rows, int cols,
                            const extract_t<double>& A,
                            const StructuredOptions<double>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_elements(int rows, int cols,
                            const extract_t<std::complex<float>>& A,
                            const StructuredOptions<std::complex<float>>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_elements(int rows, int cols,
                            const extract_t<std::complex<double>>& A,
                            const StructuredOptions<std::complex<double>>& opts);



    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_matrix_free(int rows, int cols,
                          const mult_t<scalar_t>& Amult,
                          const StructuredOptions<scalar_t>& opts) {
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
    construct_matrix_free(int rows, int cols,
                          const mult_t<float>& Amult,
                          const StructuredOptions<float>& opts);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_matrix_free(int rows, int cols,
                          const mult_t<double>& Amult,
                          const StructuredOptions<double>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_matrix_free(int rows, int cols,
                          const mult_t<std::complex<float>>& Amult,
                          const StructuredOptions<std::complex<float>>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_matrix_free(int rows, int cols,
                          const mult_t<std::complex<double>>& Amult,
                          const StructuredOptions<std::complex<double>>& opts);



    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<scalar_t>& Amult,
                                    const extract_block_t<scalar_t>& Aelem,
                                    const StructuredOptions<scalar_t>& opts) {
      switch (opts.type()) {
      case Type::HSS: {
        HSS::HSSOptions<scalar_t> hss_opts(opts);
        auto H = new HSS::HSSMatrix<scalar_t>(rows, cols, hss_opts);
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
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<float>& Amult,
                                    const extract_block_t<float>& Aelem,
                                    const StructuredOptions<float>& opts);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<double>& Amult,
                                    const extract_block_t<double>& Aelem,
                                    const StructuredOptions<double>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<std::complex<float>>& Amult,
                                    const extract_block_t<std::complex<float>>& Aelem,
                                    const StructuredOptions<std::complex<float>>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<std::complex<double>>& Amult,
                                    const extract_block_t<std::complex<double>>& Aelem,
                                    const StructuredOptions<std::complex<double>>& opts);



    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<scalar_t>& Amult,
                                    const extract_t<scalar_t>& Aelem,
                                    const StructuredOptions<scalar_t>& opts) {
      auto extract_block =
        [&Aelem](const std::vector<std::size_t>& I,
                 const std::vector<std::size_t>& J,
                 DenseMatrix<scalar_t>& B) {
          for (std::size_t j=0; j<J.size(); j++)
            for (std::size_t i=0; i<I.size(); i++)
              B(i, j) = Aelem(I[i], J[j]);
        };
      return construct_partially_matrix_free<scalar_t>
        (rows, cols, Amult, extract_block, opts);
    }
    template std::unique_ptr<StructuredMatrix<float>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<float>& Amult,
                                    const extract_t<float>& Aelem,
                                    const StructuredOptions<float>& opts);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<double>& Amult,
                                    const extract_t<double>& Aelem,
                                    const StructuredOptions<double>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<std::complex<float>>& Amult,
                                    const extract_t<std::complex<float>>& Aelem,
                                    const StructuredOptions<std::complex<float>>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_partially_matrix_free(int rows, int cols,
                                    const mult_t<std::complex<double>>& Amult,
                                    const extract_t<std::complex<double>>& Aelem,
                                    const StructuredOptions<std::complex<double>>& opts);

#if defined(STRUMPACK_USE_MPI)
    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_dense(const DistributedMatrix<scalar_t>& A,
                         const StructuredOptions<scalar_t>& opts) {
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
      switch (opts.type()) {
      case Type::HSS: {
        HSS::HSSOptions<scalar_t> hss_opts(opts);
        return std::unique_ptr<StructuredMatrix<scalar_t>>
          (new HSS::HSSMatrixMPI<scalar_t>(A, hss_opts));
      } break;
      case Type::BLR:
        throw std::invalid_argument("Not implemented yet.");
      case Type::HODLR: {
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
      } break;
      case Type::HODBF: {
        if (A.rows() != A.cols())
          throw std::invalid_argument
            ("HODBF compression only supported for square matrices.");
        structured::ClusterTree t(A.rows());
        t.refine(opts.leaf_size());
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        hodbf_opts.set_butterfly_levels(1000);
        auto H = new HODLR::HODLRMatrix<scalar_t>
          (A.Comm(), t, hodbf_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
      } break;
      case Type::BUTTERFLY: {
        structured::ClusterTree tr(A.rows()), tc(A.cols());
        tr.refine(opts.leaf_size());
        tc.refine(opts.leaf_size());
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        hodbf_opts.set_butterfly_levels(1000);
        auto H = new HODLR::ButterflyMatrix<scalar_t>
          (A.Comm(), tr, tc, hodbf_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
      } break;
      case Type::LR: {
        structured::ClusterTree tr(A.rows()), tc(A.cols());
        // tr.refine(opts.leaf_size());
        // tc.refine(opts.leaf_size());
        HODLR::HODLROptions<scalar_t> lr_opts(opts);
        lr_opts.set_butterfly_levels(0);
        auto H = new HODLR::ButterflyMatrix<scalar_t>
          (A.Comm(), tr, tc, lr_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
      }
      case Type::LOSSY:
        throw std::invalid_argument("Not implemented yet.");
      case Type::LOSSLESS:
        throw std::invalid_argument("Not implemented yet.");
      }
      return std::unique_ptr<StructuredMatrix<scalar_t>>(nullptr);
    }

    // explicit template instantiations
    template std::unique_ptr<StructuredMatrix<float>>
    construct_from_dense(const DistributedMatrix<float>& A,
                         const StructuredOptions<float>& opts);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_dense(const DistributedMatrix<double>& A,
                         const StructuredOptions<double>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_dense(const DistributedMatrix<std::complex<float>>& A,
                         const StructuredOptions<std::complex<float>>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_dense(const DistributedMatrix<std::complex<double>>& A,
                         const StructuredOptions<std::complex<double>>& opts);


    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_dist_block_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts) {
      switch (opts.type()) {
      case Type::HSS:
        throw std::invalid_argument("Not implemented yet.");
      case Type::BLR:
        throw std::invalid_argument("Not implemented yet.");
      case Type::HODLR: {
        if (rows != cols)
          throw std::invalid_argument
            ("HODLR compression only supported for square matrices.");
        auto Ablocks =
          [&A](const std::vector<std::vector<std::size_t>>& I,
               const std::vector<std::vector<std::size_t>>& J,
               std::vector<DistributedMatrixWrapper<scalar_t>>& B,
               HODLR::ExtractionMeta&) {
            for (std::size_t i=0; i<I.size(); i++)
              A(I[i], J[i], B[i]);
          };
        structured::ClusterTree t(rows);
        t.refine(opts.leaf_size());
        HODLR::HODLROptions<scalar_t> hodlr_opts(opts);
        auto H = new HODLR::HODLRMatrix<scalar_t>
          (comm, t, hodlr_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
      } break;
      case Type::HODBF:
        throw std::invalid_argument("Not implemented yet.");
      case Type::BUTTERFLY:
        throw std::invalid_argument("Not implemented yet.");
      case Type::LR:
        throw std::invalid_argument("Not implemented yet.");
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
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_dist_block_t<float>& A,
                            const StructuredOptions<float>& opts);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_dist_block_t<double>& A,
                            const StructuredOptions<double>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_dist_block_t<std::complex<float>>& A,
                            const StructuredOptions<std::complex<float>>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_dist_block_t<std::complex<double>>& A,
                            const StructuredOptions<std::complex<double>>& opts);


    template<typename scalar_t> std::unique_ptr<StructuredMatrix<scalar_t>>
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_t<scalar_t>& A,
                            const StructuredOptions<scalar_t>& opts) {
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
      switch (opts.type()) {
      case Type::HSS:
        throw std::invalid_argument("Not implemented yet.");
      case Type::BLR:
        throw std::invalid_argument("Not implemented yet.");
      case Type::HODLR: {
        if (rows != cols)
          throw std::invalid_argument
            ("HODLR compression only supported for square matrices.");
        structured::ClusterTree t(rows);
        t.refine(opts.leaf_size());
        HODLR::HODLROptions<scalar_t> hodlr_opts(opts);
        auto H = new HODLR::HODLRMatrix<scalar_t>(comm, t, hodlr_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
      } break;
      case Type::HODBF: {
        if (rows != cols)
          throw std::invalid_argument
            ("HODBF compression only supported for square matrices.");
        structured::ClusterTree t(rows);
        t.refine(opts.leaf_size());
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        hodbf_opts.set_butterfly_levels(1000);
        auto H = new HODLR::HODLRMatrix<scalar_t>(comm, t, hodbf_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
      } break;
      case Type::BUTTERFLY: {
        structured::ClusterTree tr(rows), tc(cols);
        tr.refine(opts.leaf_size());
        tc.refine(opts.leaf_size());
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        hodbf_opts.set_butterfly_levels(1000);
        auto H = new HODLR::ButterflyMatrix<scalar_t>
          (comm, tr, tc, hodbf_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
      } break;
      case Type::LR: {
        structured::ClusterTree tr(rows), tc(cols);
        HODLR::HODLROptions<scalar_t> hodbf_opts(opts);
        auto H = new HODLR::ButterflyMatrix<scalar_t>
          (comm, tr, tc, hodbf_opts);
        H->compress(Ablocks);
        return std::unique_ptr<StructuredMatrix<scalar_t>>(H);
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
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_t<float>& A,
                            const StructuredOptions<float>& opts);
    template std::unique_ptr<StructuredMatrix<double>>
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_t<double>& A,
                            const StructuredOptions<double>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<float>>>
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_t<std::complex<float>>& A,
                            const StructuredOptions<std::complex<float>>& opts);
    template std::unique_ptr<StructuredMatrix<std::complex<double>>>
    construct_from_elements(const MPIComm& comm, int rows, int cols,
                            const extract_t<std::complex<double>>& A,
                            const StructuredOptions<std::complex<double>>& opts);
#endif


    template<typename scalar_t> void
    StructuredMatrix<scalar_t>::mult(Trans op, const DenseMatrix<scalar_t>& x,
                                     DenseMatrix<scalar_t>& y) const {
      throw std::invalid_argument
        ("Operation mult not implemented for this type.");
    }
    template<typename scalar_t> void
    StructuredMatrix<scalar_t>::mult(Trans op,
                                     const DistributedMatrix<scalar_t>& x,
                                     DistributedMatrix<scalar_t>& y) const {
      throw std::invalid_argument
        ("Operation mult(Dist) not implemented for this type.");
    }
    template<typename scalar_t> void
    StructuredMatrix<scalar_t>::factor() {
      throw std::invalid_argument
        ("Operation factor not implemented for this type.");
    }
    template<typename scalar_t> void
    StructuredMatrix<scalar_t>::solve(DenseMatrix<scalar_t>& b) const {
      throw std::invalid_argument
        ("Operation solve not implemented for this type.");
    }
    template<typename scalar_t> void
    StructuredMatrix<scalar_t>::solve(DistributedMatrix<scalar_t>& b) const {
      throw std::invalid_argument
        ("Operation solve(Dist) not implemented for this type.");
    }


    // explicit template instantiations
    template class StructuredMatrix<float>;
    template class StructuredMatrix<double>;
    template class StructuredMatrix<std::complex<float>>;
    template class StructuredMatrix<std::complex<double>>;

  } // end namespace structured
} // end namespace strumpack

