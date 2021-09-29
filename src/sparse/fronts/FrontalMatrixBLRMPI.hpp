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
#ifndef FRONTAL_MATRIX_BLR_MPI_HPP
#define FRONTAL_MATRIX_BLR_MPI_HPP

#include "FrontalMatrixMPI.hpp"
#include "BLR/BLRMatrixMPI.hpp"
#include "BLR/BLRExtendAdd.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  class FrontalMatrixBLRMPI : public FrontalMatrixMPI<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using DistMW_t = DistributedMatrixWrapper<scalar_t>;
    using BLRMPI_t = BLR::BLRMatrixMPI<scalar_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using FBLRMPI_t = FrontalMatrixBLRMPI<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using Opts_t = SPOptions<scalar_t>;
    using VecVec_t = std::vector<std::vector<std::size_t>>;

  public:
    FrontalMatrixBLRMPI
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd, const MPIComm& comm, int P,
     int leaf);

    void release_work_memory() override;

    void build_front(const SpMat_t& A);
    void build_front_cols(const SpMat_t& A, std::size_t i, bool part, std::size_t CP,
                          const std::vector<Triplet<scalar_t>>& r1buf,
                          const std::vector<Triplet<scalar_t>>& r2buf,
                          const std::vector<Triplet<scalar_t>>& r3buf,
                          const Opts_t& opts);

    void extend_add();
    void extend_add_cols(std::size_t i, bool part, std::size_t CP,
                         const Opts_t& opts);
    void extend_add_copy_to_buffers(std::vector<std::vector<scalar_t>>& sbuf,
                                    const FMPI_t* pa) const override;
    void extadd_blr_copy_to_buffers(std::vector<std::vector<scalar_t>>& sbuf,
                                    const FBLRMPI_t* pa) const override;
    void extadd_blr_copy_to_buffers_col(std::vector<std::vector<scalar_t>>& sbuf,
                                        const FBLRMPI_t* pa,
                                        integer_t begin_col, integer_t end_col,
                                        const Opts_t& opts)
      const override;
    void extadd_blr_copy_from_buffers(BLRMPI_t& F11, BLRMPI_t& F12,
                                      BLRMPI_t& F21, BLRMPI_t& F22,
                                      scalar_t** pbuf, const FBLRMPI_t* pa)
      const override;
    void extadd_blr_copy_from_buffers_col(BLRMPI_t& F11, BLRMPI_t& F12,
                                          BLRMPI_t& F21, BLRMPI_t& F22,
                                          scalar_t** pbuf, const FBLRMPI_t* pa,
                                          integer_t begin_col, integer_t end_col)
      const override;

    void multifrontal_factorization(const SpMat_t& A, const Opts_t& opts,
                                    int etree_level=0, int task_depth=0)
      override;

    void forward_multifrontal_solve(DenseM_t& bloc, DistM_t* bdist,
                                    DistM_t& bupd, DenseM_t& seqbupd,
                                    int etree_level=0) const override;
    void backward_multifrontal_solve(DenseM_t& yloc, DistM_t* ydist,
                                     DistM_t& yupd, DenseM_t& seqyupd,
                                     int etree_level=0) const override;

    void sample_CB(const DistM_t& R, DistM_t& Sr, DistM_t& Sc,
                   F_t* pa) const override {
      std::cout << "FrontalMatrixBLRMPI::sample_CB TODO" << std::endl;
    }

    void extract_CB_sub_matrix_2d(const VecVec_t& I, const VecVec_t& J,
                                  std::vector<DistM_t>& B) const override;

    std::string type() const override { return "FrontalMatrixBLRMPI"; }

    void partition(const Opts_t& opts, const SpMat_t& A,
                   integer_t* sorder, bool is_root, int task_depth) override;

    const BLR::ProcessorGrid2D& grid2d() const { return pgrid_; }

    int sep_rg2p(std::size_t i) const { return F11blr_.rg2p(i); }
    int sep_cg2p(std::size_t j) const { return F11blr_.cg2p(j); }

    // might not be active, but still need this for extadd
    int upd_rg2p(std::size_t i) const { return (i/leaf_)%pgrid_.nprows(); }
    int upd_cg2p(std::size_t j) const { return (j/leaf_)%pgrid_.npcols(); }

  private:
    BLRMPI_t F11blr_, F12blr_, F21blr_, F22blr_;
    std::vector<int> piv_;
    std::vector<std::size_t> sep_tiles_, upd_tiles_;
    DenseMatrix<bool> adm_;
    BLR::ProcessorGrid2D pgrid_;
    int leaf_ = 0;

    long long node_factor_nonzeros() const override;

    using F_t::lchild_;
    using F_t::rchild_;
    using F_t::dim_sep;
    using F_t::dim_upd;
    using F_t::sep_begin_;
    using F_t::sep_end_;
    using FMPI_t::visit;
    using FMPI_t::Comm;
    using FMPI_t::grid;

    template<typename _scalar_t,typename _integer_t>
    friend class BLR::BLRExtendAdd;
  };

} // end namespace strumpack

#endif // FRONTAL_MATRIX_BLR_MPI_HPP
