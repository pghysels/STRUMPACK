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
#ifndef FRONTAL_MATRIX_HODLR_MPI_HPP
#define FRONTAL_MATRIX_HODLR_MPI_HPP

#include "FrontalMatrixMPI.hpp"
#include "HODLR/HODLRMatrix.hpp"
#include "HODLR/ButterflyMatrix.hpp"

// #define STRUMPACK_PERMUTE_CB

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  class FrontalMatrixHODLRMPI : public FrontalMatrixMPI<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using DistMW_t = DistributedMatrixWrapper<scalar_t>;
    using Opts_t = SPOptions<scalar_t>;
    using VecVec_t = std::vector<std::vector<std::size_t>>;

  public:
    FrontalMatrixHODLRMPI(integer_t sep, integer_t sep_begin,
                          integer_t sep_end, std::vector<integer_t>& upd,
                          const MPIComm& comm, int _total_procs);

    ~FrontalMatrixHODLRMPI();

    FrontalMatrixHODLRMPI(const FrontalMatrixHODLRMPI&) = delete;

    FrontalMatrixHODLRMPI& operator=(FrontalMatrixHODLRMPI const&) = delete;

    void release_work_memory() override;

    void sample_CB(Trans op, const DistM_t& R, DistM_t& S,
                   F_t* pa) const override;
    void sample_children_CB(Trans op, const DistM_t& R, DistM_t& S);

    void skinny_extend_add(DistM_t& cSl, DistM_t& cSr, DistM_t& S);

    void extend_add_copy_to_buffers(std::vector<std::vector<scalar_t>>& sbuf,
                                    const FMPI_t* pa) const override;

    void multifrontal_factorization(const SpMat_t& A, const Opts_t& opts,
                                    int etree_level=0, int task_depth=0)
      override;

    void forward_multifrontal_solve(DenseM_t& bloc, DistM_t* bdist,
                                    DistM_t& bupd, DenseM_t& seqbupd,
                                    int etree_level=0) const override;
    void backward_multifrontal_solve(DenseM_t& yloc, DistM_t* ydist,
                                     DistM_t& yupd, DenseM_t& seqyupd,
                                     int etree_level=0) const override;

    long long node_factor_nonzeros() const override;
    integer_t front_rank(int task_depth=0) const override;
    std::string type() const override { return "FrontalMatrixHODLRMPI"; }

    void extract_CB_sub_matrix_2d(const VecVec_t& I, const VecVec_t& J,
                                  std::vector<DistM_t>& B) const override;

    void partition(const Opts_t& opts, const SpMat_t& A, integer_t* sorder,
                   bool is_root=true, int task_depth=0) override;

  private:
    HODLR::HODLRMatrix<scalar_t> F11_;
    HODLR::ButterflyMatrix<scalar_t> F12_, F21_;
    std::unique_ptr<HODLR::HODLRMatrix<scalar_t>> F22_;
    structured::ClusterTree sep_tree_;
#if defined(STRUMPACK_PERMUTE_CB)
    std::vector<integer_t> CB_perm_, CB_iperm_;
#endif

    void construct_hierarchy(const SpMat_t& A, const Opts_t& opts);
    void compress_sampling(const SpMat_t& A, const Opts_t& opts);
    void compress_extraction(const SpMat_t& A, const Opts_t& opts);
    void compress_flops_F11();
    void compress_flops_F12_F21();
    void compress_flops_F22();
    void compress_flops_Schur(long long int invf11_mult_flops);

    using F_t::lchild_;
    using F_t::rchild_;
    using F_t::sep_begin_;
    using F_t::sep_end_;
    using F_t::dim_sep;
    using F_t::dim_upd;
    using F_t::dim_blk;
    using FMPI_t::visit;
    using FMPI_t::grid;
    using FMPI_t::Comm;
    template<typename _scalar_t,typename _integer_t> friend class ExtendAdd;
  };

} // end namespace strumpack

#endif // FRONTAL_MATRIX_HODLR_MPI_HPP
