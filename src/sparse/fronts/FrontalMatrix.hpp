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
#ifndef FRONTAL_MATRIX_HPP
#define FRONTAL_MATRIX_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <typeinfo>

#include "StrumpackParameters.hpp"
#include "misc/TaskTimer.hpp"
#include "dense/DenseMatrix.hpp"
#include "sparse/CompressedSparseMatrix.hpp"
#include "BLR/BLRMatrix.hpp"
#if defined(_OPENMP)
#include "omp.h"
#endif
#if defined(STRUMPACK_USE_MPI)
#include "dense/DistributedMatrix.hpp"
#include "BLR/BLRMatrixMPI.hpp"
#endif

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixMPI;
  template<typename scalar_t,typename integer_t> class FrontalMatrixBLRMPI;


  template<typename scalar_t,typename integer_t> class FrontalMatrix {
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using Opts_t = SPOptions<scalar_t>;
    using BLRM_t = BLR::BLRMatrix<scalar_t>;
#if defined(STRUMPACK_USE_MPI)
    using DistM_t = DistributedMatrix<scalar_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using FBLRMPI_t = FrontalMatrixBLRMPI<scalar_t,integer_t>;
    using BLRMPI_t = BLR::BLRMatrixMPI<scalar_t>;
#endif

  public:
    FrontalMatrix(F_t* lchild, F_t* rchild,
                  integer_t sep, integer_t sep_begin,
                  integer_t sep_end, std::vector<integer_t>& upd);
    virtual ~FrontalMatrix() = default;

    integer_t sep_begin() const { return sep_begin_; }
    integer_t sep_end() const { return sep_end_; }
    integer_t dim_sep() const { return sep_end_ - sep_begin_; }
    integer_t dim_upd() const { return upd_.size(); }
    integer_t dim_blk() const { return dim_sep() + dim_upd(); }
    const std::vector<integer_t>& upd() const { return upd_; }

    void draw(std::ostream& of, int etree_level=0) const;

    void find_upd_indices(const std::vector<std::size_t>& I,
                          std::vector<std::size_t>& lI,
                          std::vector<std::size_t>& oI) const;

    void upd_to_parent(const F_t* pa, std::size_t& upd2sep,
                       std::size_t* I) const;
    void upd_to_parent(const F_t* pa, std::size_t* I) const;
    std::vector<std::size_t> upd_to_parent(const F_t* pa,
                                           std::size_t& upd2sep) const;
    std::vector<std::size_t> upd_to_parent(const F_t* pa) const;

    virtual void release_work_memory() = 0;

    virtual ReturnCode
    multifrontal_factorization(const SpMat_t& A, const Opts_t& opts,
                               int etree_level=0, int task_depth=0) = 0;

    virtual ReturnCode factor(const SpMat_t& A, const Opts_t& opts,
                              VectorPool<scalar_t>& workspace,
                              int etree_level=0, int task_depth=0) {
      return multifrontal_factorization(A, opts, etree_level, task_depth);
    };

    virtual void delete_factors() {}

    virtual void multifrontal_solve(DenseM_t& b) const;

    virtual void
    forward_multifrontal_solve(DenseM_t& b, DenseM_t* work,
                               int etree_level=0,
                               int task_depth=0) const;
    virtual void
    backward_multifrontal_solve(DenseM_t& y, DenseM_t* work,
                                int etree_level=0,
                                int task_depth=0) const;

    void fwd_solve_phase1(DenseM_t& b, DenseM_t& bupd, DenseM_t* work,
                          int etree_level, int task_depth) const;
    virtual
    void fwd_solve_phase2(DenseM_t& b, DenseM_t& bupd,
                          int etree_level, int task_depth) const {};
    void bwd_solve_phase2(DenseM_t& y, DenseM_t& yupd, DenseM_t* work,
                          int etree_level, int task_depth) const;
    virtual
    void bwd_solve_phase1(DenseM_t& y, DenseM_t& yupd,
                          int etree_level, int task_depth) const {};

    ReturnCode inertia(integer_t& neg,
                       integer_t& zero,
                       integer_t& pos) const;

    virtual void
    extend_add_to_dense(DenseM_t& paF11, DenseM_t& paF12,
                        DenseM_t& paF21, DenseM_t& paF22,
                        const FrontalMatrix<scalar_t,integer_t>* p,
                        int task_depth) {
      assert(false);
    }
    virtual void
    extend_add_to_dense(DenseM_t& paF11, DenseM_t& paF12,
                        DenseM_t& paF21, DenseM_t& paF22,
                        const FrontalMatrix<scalar_t,integer_t>* p,
                        VectorPool<scalar_t>& workspace,
                        int task_depth) {
      extend_add_to_dense(paF11, paF12, paF21, paF22, p, task_depth);
    }

    virtual void
    extend_add_to_blr(BLRM_t& paF11, BLRM_t& paF12,
                      BLRM_t& paF21, BLRM_t& paF22,
                      const FrontalMatrix<scalar_t,integer_t>* p,
                      int task_depth, const Opts_t& opts) {}
    virtual void
    extend_add_to_blr_col(BLRM_t& paF11, BLRM_t& paF12,
                          BLRM_t& paF21, BLRM_t& paF22,
                          const FrontalMatrix<scalar_t,integer_t>* p,
                          integer_t begin_col, integer_t end_col,
                          int task_depth, const Opts_t& opts) {}

    virtual int random_samples() const { return 0; }

    // TODO why not const? HSS problem?
    virtual void
    sample_CB(const Opts_t& opts, const DenseM_t& R,
              DenseM_t& Sr, DenseM_t& Sc,
              F_t* parent, int task_depth=0) { assert(false); }
    virtual void
    sample_CB(Trans op, const DenseM_t& R, DenseM_t& S, F_t* parent,
              int task_depth=0) const { assert(false); }

    virtual void
    sample_CB_to_F11(Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
                     int task_depth=0) const {}
    virtual void
    sample_CB_to_F12(Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
                     int task_depth=0) const {}
    virtual void
    sample_CB_to_F21(Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
                     int task_depth=0) const {}
    virtual void
    sample_CB_to_F22(Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
                     int task_depth=0) const {}

    virtual void
    extract_CB_sub_matrix(const std::vector<std::size_t>& I,
                          const std::vector<std::size_t>& J,
                          DenseM_t& B, int task_depth) const = 0;

    virtual void
    extract_CB_sub_matrix_blocks(const std::vector<std::vector<std::size_t>>& I,
                                 const std::vector<std::vector<std::size_t>>& J,
                                 std::vector<DenseM_t>& Bseq, int task_depth) const;
    virtual void
    extract_CB_sub_matrix_blocks(const std::vector<std::vector<std::size_t>>& I,
                                 const std::vector<std::vector<std::size_t>>& J,
                                 std::vector<DenseMW_t>& Bseq, int task_depth) const;

    void extend_add_b(DenseM_t& b, DenseM_t& bupd,
                      const DenseM_t& CB, const F_t* pa) const;
    void extract_b(const DenseM_t& y, const DenseM_t& yupd,
                   DenseM_t& CB, const F_t* pa) const;

    virtual integer_t maximum_rank(int task_depth=0) const;
    virtual integer_t front_rank(int task_depth=0) const { return 0; }

    virtual long long factor_nonzeros(int task_depth=0) const;
    virtual long long dense_factor_nonzeros(int task_depth=0) const;
    virtual bool isHSS() const { return false; }
    virtual bool isMPI() const { return false; }
    virtual void print_rank_statistics(std::ostream &out) const {}
    virtual std::string type() const { return "FrontalMatrix"; }

    virtual void
    partition_fronts(const Opts_t& opts, const SpMat_t& A, integer_t* sorder,
                     bool is_root=true, int task_depth=0);
    void permute_CB(const integer_t* perm, int task_depth=0);

    int levels() const {
      int ll = 0, lr = 0;
      if (lchild_) ll = lchild_->levels();
      if (rchild_) lr = rchild_->levels();
      return std::max(ll, lr) + 1;
    }

    void set_lchild(std::unique_ptr<F_t> ch) { lchild_ = std::move(ch); }
    void set_rchild(std::unique_ptr<F_t> ch) { rchild_ = std::move(ch); }

    // TODO compute this (and levels) once, store it
    // maybe compute it when setting pointers to the children
    // create setters/getters for the children
    integer_t max_dim_upd() const {
      integer_t max_dupd = dim_upd();
      if (lchild_) max_dupd = std::max(max_dupd, lchild_->max_dim_upd());
      if (rchild_) max_dupd = std::max(max_dupd, rchild_->max_dim_upd());
      return max_dupd;
    }

    virtual int P() const { return 1; }

    void get_level_fronts(std::vector<const F_t*>& ldata, int elvl, int l=0) const;
    void get_level_fronts(std::vector<F_t*>& ldata, int elvl, int l=0);

#if defined(STRUMPACK_USE_MPI)
    void multifrontal_solve(DenseM_t& bloc, DistM_t* bdist) const;
    virtual void
    forward_multifrontal_solve(DenseM_t& bloc, DistM_t* bdist,
                               DistM_t& bupd, DenseM_t& seqbupd,
                               int etree_level=0) const;
    virtual void
    backward_multifrontal_solve(DenseM_t& yloc, DistM_t* ydist,
                                DistM_t& yupd, DenseM_t& seqyupd,
                                int etree_level=0) const;

    virtual void
    sample_CB(Trans op, const DistM_t& R, DistM_t& S,
              const DenseM_t& seqR, DenseM_t& seqS, F_t* pa) const {
      sample_CB(op, seqR, seqS, pa);
    };
    virtual void
    sample_CB(const Opts_t& opts, const DistM_t& R,
              DistM_t& Sr, DistM_t& Sc, const DenseM_t& seqR,
              DenseM_t& seqSr, DenseM_t& seqSc, F_t* pa) {
      sample_CB(opts, seqR, seqSr, seqSc, pa, 0);
    }

    virtual void
    extend_add_copy_to_buffers(std::vector<std::vector<scalar_t>>& sbuf,
                               const FMPI_t* pa) const {
      std::cerr << "FrontalMatrix::extend_add_copy_to_buffers"
                << " not implemented for this front type: "
                << typeid(*this).name()
                << std::endl;
      abort();
    }
    virtual void
    extend_add_copy_from_buffers(DistM_t& F11, DistM_t& F12,
                                 DistM_t& F21, DistM_t& F22,
                                 scalar_t** pbuf, const FMPI_t* pa) const;

    virtual void
    extadd_blr_copy_to_buffers(std::vector<std::vector<scalar_t>>& sbuf,
                               const FBLRMPI_t* pa) const {
      std::cerr << "FrontalMatrix::extadd_blr_copy_to_buffers"
                << " not implemented for this front type: "
                << typeid(*this).name()
                << std::endl;
      abort();
    }
    virtual void
    extadd_blr_copy_to_buffers_col(std::vector<std::vector<scalar_t>>& sbuf,
                                   const FBLRMPI_t* pa,
                                   integer_t begin_col, integer_t end_col,
                                   const Opts_t& opts) const {
      std::cerr << "FrontalMatrix::extadd_blr_copy_to_buffers_col"
                << " not implemented for this front type: "
                << typeid(*this).name()
                << std::endl;
      abort();
    }
    virtual void
    extadd_blr_copy_from_buffers(BLRMPI_t& F11, BLRMPI_t& F12,
                                 BLRMPI_t& F21, BLRMPI_t& F22,
                                 scalar_t** pbuf, const FBLRMPI_t* pa) const;

    virtual void
    extadd_blr_copy_from_buffers_col(BLRMPI_t& F11, BLRMPI_t& F12,
                                     BLRMPI_t& F21, BLRMPI_t& F22,
                                     scalar_t** pbuf, const FBLRMPI_t* pa,
                                     integer_t begin_col, integer_t end_col) const;

    virtual void
    extend_add_column_copy_to_buffers(const DistM_t& CB, const DenseM_t& seqCB,
                                      std::vector<std::vector<scalar_t>>& sbuf,
                                      const FMPI_t* pa) const;
    virtual void
    extend_add_column_copy_from_buffers(DistM_t& B, DistM_t& Bupd,
                                        scalar_t** pbuf, const FMPI_t* pa) const;
    virtual void
    extract_column_copy_to_buffers(const DistM_t& b, const DistM_t& bupd,
                                   int ch_master,
                                   std::vector<std::vector<scalar_t>>& sbuf,
                                   const FMPI_t* pa) const;
    virtual void
    extract_column_copy_from_buffers(const DistM_t& b, DistM_t& CB,
                                     DenseM_t& seqCB,
                                     std::vector<scalar_t*>& pbuf,
                                     const FMPI_t* pa) const;
    virtual void
    skinny_ea_to_buffers(const DistM_t& S, const DenseM_t& seqS,
                         std::vector<std::vector<scalar_t>>& sbuf,
                         const FMPI_t* pa) const;
    virtual void
    skinny_ea_from_buffers(DistM_t& S, scalar_t** pbuf, const FMPI_t* pa) const;

    virtual void
    extract_from_R2D(const DistM_t& R, DistM_t& cR, DenseM_t& seqcR,
                     const FMPI_t* pa, bool visit) const;

    virtual void
    get_submatrix_2d(const std::vector<std::size_t>& I,
                     const std::vector<std::size_t>& J,
                     DistM_t& Bdist, DenseM_t& Bseq) const;
    virtual void
    get_submatrix_2d(const std::vector<std::vector<std::size_t>>& I,
                     const std::vector<std::vector<std::size_t>>& J,
                     std::vector<DistM_t>& Bdist,
                     std::vector<DenseM_t>& Bseq) const;

    virtual BLACSGrid* grid() { return nullptr; }
    virtual const BLACSGrid* grid() const { return nullptr; }
#endif

  protected:
    integer_t sep_, sep_begin_, sep_end_;
    std::vector<integer_t> upd_;
    std::unique_ptr<F_t> lchild_, rchild_;

    virtual long long node_factor_nonzeros() const {
      return dense_node_factor_nonzeros();
    }

    virtual void partition(const Opts_t& opts, const SpMat_t& A,
                           integer_t* sorder,
                           bool is_root=true, int task_depth=0);

    virtual ReturnCode node_inertia(integer_t& neg,
                                    integer_t& zero,
                                    integer_t& pos) const {
      return ReturnCode::INACCURATE_INERTIA;
    }

  private:
    FrontalMatrix(const FrontalMatrix&) = delete;
    FrontalMatrix& operator=(FrontalMatrix const&) = delete;

    virtual void draw_node(std::ostream& of, bool is_root) const;

    virtual long long dense_node_factor_nonzeros() const {
      long long dsep = dim_sep(), dupd = dim_upd();
      return dsep * (dsep + 2 * dupd);
    }
  };

} // end namespace strumpack

#endif // FRONTAL_MATRIX_HPP
