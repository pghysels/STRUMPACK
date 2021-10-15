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
 * five (5) year renewals, the U.S. Government igs granted for itself
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
#ifndef FRONTAL_MATRIX_DENSE_MPI_HPP
#define FRONTAL_MATRIX_DENSE_MPI_HPP

#include "FrontalMatrixMPI.hpp"
#if defined(STRUMPACK_USE_ZFP)
#include "FrontalMatrixLossy.hpp"
#endif
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
#define LAPACK_COMPLEX_CPP
#include <slate/slate.hh>
#endif

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixBLRMPI;
  namespace BLR {
    template<typename scalar_t> class BLRMatrixMPI;
  }

  template<typename scalar_t,typename integer_t>
  class FrontalMatrixDenseMPI : public FrontalMatrixMPI<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using DistMW_t = DistributedMatrixWrapper<scalar_t>;
    using BLRMPI_t = BLR::BLRMatrixMPI<scalar_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using FDMPI_t = FrontalMatrixDenseMPI<scalar_t,integer_t>;
    using FBLRMPI_t = FrontalMatrixBLRMPI<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using VecVec_t = std::vector<std::vector<std::size_t>>;

  public:
    FrontalMatrixDenseMPI
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd, const MPIComm& comm, int P);
    FrontalMatrixDenseMPI(const FDMPI_t&) = delete;
    FrontalMatrixDenseMPI& operator=(FDMPI_t const&) = delete;

    void release_work_memory() override;

    void extend_add();
    void
    extend_add_copy_to_buffers(std::vector<std::vector<scalar_t>>& sbuf,
                               const FMPI_t* pa) const override;
    void
    extadd_blr_copy_to_buffers(std::vector<std::vector<scalar_t>>& sbuf,
                               const FBLRMPI_t* pa) const override;
    void
    extadd_blr_copy_to_buffers_col(std::vector<std::vector<scalar_t>>& sbuf,
                                   const FBLRMPI_t* pa, integer_t begin_col,
                                   integer_t end_col,
                                   const SPOptions<scalar_t>& opts)
      const override;
    void
    extadd_blr_copy_from_buffers(BLRMPI_t& F11, BLRMPI_t& F12,
                                 BLRMPI_t& F21, BLRMPI_t& F22,
                                 scalar_t** pbuf, const FBLRMPI_t* pa)
      const override;

    void
    extadd_blr_copy_from_buffers_col(BLRMPI_t& F11, BLRMPI_t& F12,
                                     BLRMPI_t& F21, BLRMPI_t& F22,
                                     scalar_t** pbuf, const FBLRMPI_t* pa,
                                     integer_t begin_col, integer_t end_col)
      const override;

    void sample_CB(const DistM_t& R, DistM_t& Sr,
                   DistM_t& Sc, F_t* pa) const override;
    void sample_CB(Trans op, const DistM_t& R, DistM_t& S,
                   FrontalMatrix<scalar_t,integer_t>* pa) const override;

    void
    multifrontal_factorization(const SpMat_t& A,
                               const SPOptions<scalar_t>& opts,
                               int etree_level=0, int task_depth=0)
      override;

    void
    forward_multifrontal_solve(DenseM_t& bloc, DistM_t* bdist,
                               DistM_t& bupd, DenseM_t& seqbupd,
                               int etree_level=0) const override;
    void
    backward_multifrontal_solve(DenseM_t& yloc, DistM_t* ydist,
                                DistM_t& yupd, DenseM_t& seqyupd,
                                int etree_level=0) const override;

    void
    extract_CB_sub_matrix_2d(const VecVec_t& I, const VecVec_t& J,
                             std::vector<DistM_t>& B) const override;

    std::string type() const override { return "FrontalMatrixDenseMPI"; }

    long long node_factor_nonzeros() const override;

    void delete_factors() override;

  private:
    DistM_t F11_, F12_, F21_, F22_;
    std::vector<int> piv;

    void build_front(const SpMat_t& A);
    void partial_factorization(const SPOptions<scalar_t>& opts);

    void fwd_solve_phase2(const DistM_t& F11, const DistM_t& F12,
                          const DistM_t& F21,
                          DistM_t& b, DistM_t& bupd) const;
    void bwd_solve_phase1(const DistM_t& F11, const DistM_t& F12,
                          const DistM_t& F21,
                          DistM_t& y, DistM_t& yupd) const;

#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
    slate::Pivots slate_piv_;
    std::map<slate::Option, slate::Value> slate_opts_;

    slate::Matrix<scalar_t> slate_matrix(const DistM_t& M) const;
#endif

#if defined(STRUMPACK_USE_ZFP)
    LossyMatrix<scalar_t> F11c_, F12c_, F21c_;
    bool compressed_ = false;

    void compress(const SPOptions<scalar_t>& opts);
    void decompress(DistM_t& F11, DistM_t& F12, DistM_t& F21) const;
#endif

    using F_t::lchild_;
    using F_t::rchild_;
    using FMPI_t::visit;
    using FMPI_t::grid;
    using FMPI_t::Comm;
    template<typename _scalar_t,typename _integer_t> friend class ExtendAdd;
  };

} // end namespace strumpack

#endif // FRONTAL_MATRIX_DENSE_MPI_HPP
