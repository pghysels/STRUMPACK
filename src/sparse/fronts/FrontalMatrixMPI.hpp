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
#ifndef FRONTAL_MATRIX_MPI_HPP
#define FRONTAL_MATRIX_MPI_HPP

#include "FrontalMatrix.hpp"

#include "misc/MPIWrapper.hpp"
#include "dense/DistributedMatrix.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixBLRMPI;
  namespace BLR {
    template<typename scalar_t> class BLRMatrixMPI;
  }

  template<typename scalar_t,typename integer_t>
  class FrontalMatrixMPI : public FrontalMatrix<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using FBLRMPI_t = FrontalMatrixBLRMPI<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using DistMW_t = DistributedMatrixWrapper<scalar_t>;
    using BLRMPI_t = BLR::BLRMatrixMPI<scalar_t>;
    using Opts_t = SPOptions<scalar_t>;
    using Vec_t = std::vector<std::size_t>;
    using VecVec_t = std::vector<std::vector<std::size_t>>;

  public:
    FrontalMatrixMPI
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd, const MPIComm& comm, int P);

    FrontalMatrixMPI(const FrontalMatrixMPI&) = delete;
    FrontalMatrixMPI& operator=(FrontalMatrixMPI const&) = delete;
    virtual ~FrontalMatrixMPI() = default;

    virtual void sample_CB
    (const DistM_t& R, DistM_t& Sr, DistM_t& Sc, F_t* pa) const {}
    void sample_CB
    (const Opts_t& opts, const DistM_t& R,
     DistM_t& Sr, DistM_t& Sc, const DenseM_t& seqR, DenseM_t& seqSr,
     DenseM_t& seqSc, F_t* pa) override {
      sample_CB(R, Sr, Sc, pa);
    }
    virtual void sample_CB
    (Trans op, const DistM_t& R, DistM_t& Sr, F_t* pa) const {};
    void sample_CB
    (Trans op, const DistM_t& R, DistM_t& S, const DenseM_t& Rseq,
     DenseM_t& Sseq, FrontalMatrix<scalar_t,integer_t>* pa) const override {
      sample_CB(op, R, S, pa);
    }

    virtual integer_t maximum_rank(int task_depth) const override;

    void extract_2d
    (const SpMat_t& A, const VecVec_t& I, const VecVec_t& J,
     std::vector<DistMW_t>& B, bool skip_sparse=false) const;
    void get_submatrix_2d
    (const VecVec_t& I, const VecVec_t& J,
     std::vector<DistM_t>& Bdist, std::vector<DenseM_t>& Bseq) const override;
    void extract_CB_sub_matrix
    (const Vec_t& I, const Vec_t& J,
     DenseM_t& B, int task_depth) const override {};
    virtual void extract_CB_sub_matrix_2d
    (const Vec_t& I, const Vec_t& J, DistM_t& B) const {};
    virtual void extract_CB_sub_matrix_2d
    (const VecVec_t& I, const VecVec_t& J, std::vector<DistM_t>& B) const;

    void extend_add_b
    (DistM_t& b, DistM_t& bupd, const DistM_t& CBl, const DistM_t& CBr,
     const DenseM_t& seqCBl, const DenseM_t& seqCBr) const;
    void extract_b
    (const DistM_t& b, const DistM_t& bupd, DistM_t& CBl, DistM_t& CBr,
     DenseM_t& seqCBl, DenseM_t& seqCBr) const;

    void extend_add_copy_from_buffers
    (DistM_t& F11, DistM_t& F12, DistM_t& F21, DistM_t& F22, scalar_t** pbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa) const override;
    void extend_add_column_copy_to_buffers
    (const DistM_t& CB, const DenseM_t& seqCB,
     std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const override;
    void extend_add_column_copy_from_buffers
    (DistM_t& B, DistM_t& Bupd, scalar_t** pbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa) const override;
    void extract_column_copy_to_buffers
    (const DistM_t& b, const DistM_t& bupd, int ch_master,
     std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const override;
    void extract_column_copy_from_buffers
    (const DistM_t& b, DistM_t& CB, DenseM_t& seqCB,
     std::vector<scalar_t*>& pbuf, const FMPI_t* pa) const override;
    void skinny_ea_to_buffers
    (const DistM_t& S, const DenseM_t& seqS,
     std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const override;
    void skinny_ea_from_buffers
    (DistM_t& S, scalar_t** pbuf, const FMPI_t* pa) const override;

    void extract_from_R2D
    (const DistM_t& R, DistM_t& cR, DenseM_t& seqcR,
     const FMPI_t* pa, bool visit) const override;

    bool visit(const F_t* ch) const;
    bool visit(const std::unique_ptr<F_t>& ch) const;
    int master(const F_t* ch) const;
    int master(const std::unique_ptr<F_t>& ch) const;

    void barrier_world() const override {}

    MPIComm& Comm() { return grid()->Comm(); }
    const MPIComm& Comm() const { return grid()->Comm(); }
    BLACSGrid* grid() override { return &blacs_grid_; }
    const BLACSGrid* grid() const override { return &blacs_grid_; }
    int P() const override { return grid()->P(); }

    virtual long long factor_nonzeros(int task_depth=0) const override;
    virtual long long dense_factor_nonzeros(int task_depth=0) const override;
    virtual std::string type() const override { return "FrontalMatrixMPI"; }
    virtual bool isMPI() const override { return true; }

    void partition_fronts
    (const Opts_t& opts, const SpMat_t& A, integer_t* sorder,
     bool is_root=true, int task_depth=0) override;

  protected:
    BLACSGrid blacs_grid_;     // 2D processor grid

    virtual long long node_factor_nonzeros() const override;

    using F_t::lchild_;
    using F_t::rchild_;
    template<typename _scalar_t,typename _integer_t> friend class ExtendAdd;
  };

} // end namespace strumpack

#endif //FRONTAL_MATRIX_MPI_HPP
