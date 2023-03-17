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
#ifndef FRONTAL_MATRIX_DENSE_HPP
#define FRONTAL_MATRIX_DENSE_HPP

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>

#include "FrontalMatrix.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "FrontalMatrixBLRMPI.hpp"
#endif

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixBLRMPI;

  template<typename scalar_t,typename integer_t> class FrontalMatrixDense
    : public FrontalMatrix<scalar_t,integer_t> {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using BLRM_t = BLR::BLRMatrix<scalar_t>;
    using Opts_t = SPOptions<scalar_t>;

  public:
    FrontalMatrixDense(integer_t sep, integer_t sep_begin, integer_t sep_end,
                       std::vector<integer_t>& upd);

    void release_work_memory() override;
    void release_work_memory(VectorPool<scalar_t>& workspace);

    void extend_add_to_dense(DenseM_t& paF11, DenseM_t& paF12,
                             DenseM_t& paF21, DenseM_t& paF22,
                             const F_t* p, VectorPool<scalar_t>& workspace,
                             int task_depth) override;
    void extend_add_to_dense(DenseM_t& paF11, DenseM_t& paF12,
                             DenseM_t& paF21, DenseM_t& paF22,
                             const F_t* p, int task_depth) override;

    void extend_add_to_blr(BLRM_t& paF11, BLRM_t& paF12, BLRM_t& paF21,
                           BLRM_t& paF22, const F_t* p, int task_depth,
                           const Opts_t& opts) override;
    void extend_add_to_blr_col(BLRM_t& paF11, BLRM_t& paF12, BLRM_t& paF21,
                               BLRM_t& paF22, const F_t* p,
                               integer_t begin_col, integer_t end_col,
                               int task_depth, const Opts_t& opts) override;

    void sample_CB(const Opts_t& opts, const DenseM_t& R,
                   DenseM_t& Sr, DenseM_t& Sc, F_t* pa, int task_depth)
      override;
    void sample_CB(Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
                   int task_depth=0) const override;

    void sample_CB_to_F11(Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
                          int task_depth=0) const override;
    void sample_CB_to_F12(Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
                          int task_depth=0) const override;
    void sample_CB_to_F21(Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
                          int task_depth=0) const override;
    void sample_CB_to_F22(Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
                          int task_depth=0) const override;

    virtual ReturnCode
    multifrontal_factorization(const SpMat_t& A, const Opts_t& opts,
                               int etree_level=0, int task_depth=0) override {
      VectorPool<scalar_t> workspace;
      return factor(A, opts, workspace, etree_level, task_depth);
    }
    virtual ReturnCode factor(const SpMat_t& A, const Opts_t& opts,
                              VectorPool<scalar_t>& workspace,
                              int etree_level=0, int task_depth=0) override;

    void
    extract_CB_sub_matrix(const std::vector<std::size_t>& I,
                          const std::vector<std::size_t>& J,
                          DenseM_t& B, int task_depth) const override;

    void delete_factors() override;

    std::string type() const override { return "FrontalMatrixDense"; }

#if defined(STRUMPACK_USE_MPI)
    void
    extend_add_copy_to_buffers(std::vector<std::vector<scalar_t>>& sbuf,
                               const FrontalMatrixMPI<scalar_t,integer_t>* pa)
      const override;
    void
    extadd_blr_copy_to_buffers(std::vector<std::vector<scalar_t>>& sbuf,
                               const FrontalMatrixBLRMPI<scalar_t,integer_t>* pa)
      const override;
    void
    extadd_blr_copy_to_buffers_col(std::vector<std::vector<scalar_t>>& sbuf,
                                   const FrontalMatrixBLRMPI<scalar_t,integer_t>* pa, 
                                   integer_t begin_col, integer_t end_col,
                                   const Opts_t& opts)
      const override;
#endif

  protected:
    DenseM_t F11_, F12_, F21_;
    DenseMW_t F22_;
    std::vector<scalar_t,NoInit<scalar_t>> CBstorage_;
    std::vector<int> piv_; // regular int because it is passed to BLAS

    FrontalMatrixDense(const FrontalMatrixDense&) = delete;
    FrontalMatrixDense& operator=(FrontalMatrixDense const&) = delete;

    ReturnCode factor_phase1(const SpMat_t& A, const Opts_t& opts,
                             VectorPool<scalar_t>& workspace,
                             int etree_level, int task_depth);
    ReturnCode factor_phase2(const SpMat_t& A, const Opts_t& opts,
                             int etree_level, int task_depth);

    virtual void
    fwd_solve_phase2(DenseM_t& b, DenseM_t& bupd, int etree_level,
                     int task_depth) const override;
    virtual void
    bwd_solve_phase1(DenseM_t& y, DenseM_t& yupd, int etree_level,
                     int task_depth) const override;

    ReturnCode matrix_inertia(const DenseM_t& F,
                              integer_t& neg,
                              integer_t& zero,
                              integer_t& pos) const;
    virtual ReturnCode node_inertia(integer_t& neg,
                                    integer_t& zero,
                                    integer_t& pos) const override;

    using F_t::lchild_;
    using F_t::rchild_;
    using F_t::dim_sep;
    using F_t::dim_upd;
  };

} // end namespace strumpack

#endif
