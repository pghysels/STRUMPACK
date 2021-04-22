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
#ifndef FRONTAL_MATRIX_HSS_MPI_HPP
#define FRONTAL_MATRIX_HSS_MPI_HPP

#include "FrontalMatrixMPI.hpp"
#include "HSS/HSSMatrixMPI.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  class FrontalMatrixHSSMPI : public FrontalMatrixMPI<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using DistMW_t = DistributedMatrixWrapper<scalar_t>;
    using Opts_t = SPOptions<scalar_t>;

  public:
    FrontalMatrixHSSMPI(integer_t sep, integer_t sep_begin, integer_t sep_end,
                        std::vector<integer_t>& upd, const MPIComm& comm,
                        int _total_procs);
    FrontalMatrixHSSMPI(const FrontalMatrixHSSMPI&) = delete;
    FrontalMatrixHSSMPI& operator=(FrontalMatrixHSSMPI const&) = delete;

    void release_work_memory() override;

    void random_sampling(const SpMat_t& A, const SPOptions<scalar_t>& opts,
                         const DistM_t& R, DistM_t& Sr, DistM_t& Sc);
    void sample_CB(const DistM_t& R, DistM_t& Sr,
                   DistM_t& Sc, F_t* pa) const override;
    void sample_children_CB(const SPOptions<scalar_t>& opts, const DistM_t& R,
                            DistM_t& Sr, DistM_t& Sc);

    void multifrontal_factorization(const SpMat_t& A, const Opts_t& opts,
                                    int etree_level=0,
                                    int task_depth=0) override;

    void forward_multifrontal_solve(DenseM_t& bloc, DistM_t* bdist,
                                    DistM_t& bupd, DenseM_t& seqbupd,
                                    int etree_level=0) const override;
    void backward_multifrontal_solve(DenseM_t& yloc, DistM_t* ydist,
                                     DistM_t& yupd, DenseM_t& seqyupd,
                                     int etree_level=0) const override;

    void element_extraction(const SpMat_t& A,
                            const std::vector<std::vector<std::size_t>>& I,
                            const std::vector<std::vector<std::size_t>>& J,
                            std::vector<DistMW_t>& B) const;

    void
    extract_CB_sub_matrix_2d(const std::vector<std::vector<std::size_t>>& I,
                             const std::vector<std::vector<std::size_t>>& J,
                             std::vector<DistM_t>& B) const override;

    long long node_factor_nonzeros() const override;
    integer_t front_rank(int task_depth=0) const override;
    bool isHSS() const override { return true; };
    std::string type() const override { return "FrontalMatrixHSSMPI"; }

    void partition(const Opts_t& opts, const SpMat_t& A, integer_t* sorder,
                   bool is_root=true, int task_depth=0) override;

  private:
    std::unique_ptr<HSS::HSSMatrixMPI<scalar_t>> H_;

    // TODO get rid of this!!!
    mutable std::unique_ptr<HSS::WorkSolveMPI<scalar_t>> ULVwork_;

    /** Schur complement update:
     *    S = F22 - theta_ * Vhat^C * phi_^C
     **/
    DistM_t theta_, phi_, Vhat_;
    DistM_t thetaVhatC_, VhatCPhiC_, DUB01_;

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

#endif // FRONTAL_MATRIX_HSS_MPI_HPP
