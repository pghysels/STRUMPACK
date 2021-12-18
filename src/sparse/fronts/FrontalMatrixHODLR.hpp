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
#ifndef FRONTAL_MATRIX_HODLR_HPP
#define FRONTAL_MATRIX_HODLR_HPP

#include "FrontalMatrix.hpp"
#include "HODLR/HODLRMatrix.hpp"
#include "HODLR/ButterflyMatrix.hpp"

// #define STRUMPACK_PERMUTE_CB

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixHODLR
    : public FrontalMatrix<scalar_t,integer_t> {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using VecVec_t = std::vector<std::vector<std::size_t>>;
    using Opts_t = SPOptions<scalar_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;

  public:
    FrontalMatrixHODLR(integer_t sep, integer_t sep_begin, integer_t sep_end,
                       std::vector<integer_t>& upd);

    ~FrontalMatrixHODLR();

    FrontalMatrixHODLR(const FrontalMatrixHODLR&) = delete;

    FrontalMatrixHODLR& operator=(FrontalMatrixHODLR const&) = delete;

    void extend_add_to_dense(DenseM_t& paF11, DenseM_t& paF12,
                             DenseM_t& paF21, DenseM_t& paF22,
                             const F_t* p, int task_depth) override;

    void
    extend_add_copy_to_buffers(std::vector<std::vector<scalar_t>>& sbuf,
                               const FMPI_t* pa) const override;

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

    void element_extraction(const SpMat_t& A,
                            const VecVec_t& I, const VecVec_t& J,
                            std::vector<DenseMW_t>& B, int task_depth,
                            bool skip_sparse=false);

    void extract_CB_sub_matrix(const std::vector<std::size_t>& I,
                               const std::vector<std::size_t>& J,
                               DenseM_t& B, int task_depth) const override;
    void extract_CB_sub_matrix_blocks(const VecVec_t& I, const VecVec_t& J,
                                      std::vector<DenseM_t>& Bseq,
                                      int task_depth) const override;
    void extract_CB_sub_matrix_blocks(const VecVec_t& I, const VecVec_t& J,
                                      std::vector<DenseMW_t>& Bseq,
                                      int task_depth) const override;

    void release_work_memory() override;

    void random_sampling(const SpMat_t& A, const Opts_t& opts, DenseM_t& Rr,
                         DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
                         int etree_level, int task_depth); // TODO const?

    void multifrontal_factorization(const SpMat_t& A, const Opts_t& opts,
                                    int etree_level=0, int task_depth=0)
      override;

    void forward_multifrontal_solve(DenseM_t& b, DenseM_t* work,
                                    int etree_level=0, int task_depth=0)
      const override;

    void backward_multifrontal_solve(DenseM_t& y, DenseM_t* work,
                                     int etree_level=0, int task_depth=0)
      const override;

    integer_t front_rank(int task_depth=0) const override;
    void print_rank_statistics(std::ostream &out) const override;
    std::string type() const override { return "FrontalMatrixHODLR"; }

    void partition(const Opts_t& opts, const SpMat_t& A, integer_t* sorder,
                   bool is_root=true, int task_depth=0) override;

  private:
    HODLR::HODLRMatrix<scalar_t> F11_;
    HODLR::ButterflyMatrix<scalar_t> F12_, F21_;
    std::unique_ptr<HODLR::HODLRMatrix<scalar_t>> F22_;
    MPIComm commself_;
    structured::ClusterTree sep_tree_;
#if defined(STRUMPACK_PERMUTE_CB)
    std::vector<integer_t> CB_perm_, CB_iperm_;
#endif

    void draw_node(std::ostream& of, bool is_root) const override;

    void multifrontal_factorization_node(const SpMat_t& A, const Opts_t& opts,
                                         int etree_level, int task_depth);

    void fwd_solve_node(DenseM_t& b, DenseM_t* work,
                        int etree_level, int task_depth) const;
    void bwd_solve_node(DenseM_t& y, DenseM_t* work,
                        int etree_level, int task_depth) const;

    long long node_factor_nonzeros() const override;

    void construct_hierarchy(const SpMat_t& A, const Opts_t& opts,
                             int task_depth);
    void compress_sampling(const SpMat_t& A, const Opts_t& opts,
                           int task_depth);
    void compress_extraction(const SpMat_t& A, const Opts_t& opts,
                             int task_depth);
    void compress_flops_F11();
    void compress_flops_F12_F21();
    void compress_flops_F22();
    void compress_flops_Schur(long long int invf11_mult_flops);

    DenseM_t get_dense_CB() const;

    using F_t::lchild_;
    using F_t::rchild_;
    using F_t::dim_sep;
    using F_t::dim_upd;
    using F_t::dim_blk;
    using F_t::sep_begin_;
    using F_t::sep_end_;
  };

} // end namespace strumpack

#endif //FRONTAL_MATRIX_HODLR_HPP
