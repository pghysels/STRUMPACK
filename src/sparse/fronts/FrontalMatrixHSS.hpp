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
#ifndef FRONTAL_MATRIX_HSS_HPP
#define FRONTAL_MATRIX_HSS_HPP

#include "FrontalMatrix.hpp"
#include "HSS/HSSMatrix.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixHSS
    : public FrontalMatrix<scalar_t,integer_t> {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using Opts_t = SPOptions<scalar_t>;

  public:
    FrontalMatrixHSS(integer_t sep, integer_t sep_begin, integer_t sep_end,
                     std::vector<integer_t>& upd);

    void extend_add_to_dense(DenseM_t& paF11, DenseM_t& paF12,
                             DenseM_t& paF21, DenseM_t& paF22,
                             const FrontalMatrix<scalar_t,integer_t>* p,
                             int task_depth) override;

    void sample_CB(const Opts_t& opts, const DenseM_t& R,
                   DenseM_t& Sr, DenseM_t& Sc,
                   F_t* pa, int task_depth) override;

    void sample_CB_direct(const DenseM_t& cR, DenseM_t& Sr, DenseM_t& Sc,
                          const std::vector<std::size_t>& I, int task_depth);

    void release_work_memory() override;
    void random_sampling(const SpMat_t& A, const Opts_t& opts, DenseM_t& Rr,
                         DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
                         int etree_level, int task_depth); // TODO const?
    void element_extraction(const SpMat_t& A,
                            const std::vector<std::size_t>& I,
                            const std::vector<std::size_t>& J,
                            DenseM_t& B, int task_depth);
    void extract_CB_sub_matrix(const std::vector<std::size_t>& I,
                               const std::vector<std::size_t>& J,
                               DenseM_t& B, int task_depth) const override;

    void multifrontal_factorization(const SpMat_t& A, const Opts_t& opts,
                                    int etree_level=0,
                                    int task_depth=0) override;

    void forward_multifrontal_solve(DenseM_t& b, DenseM_t* work,
                                    int etree_level=0,
                                    int task_depth=0) const override;
    void backward_multifrontal_solve(DenseM_t& y, DenseM_t* work,
                                     int etree_level=0,
                                     int task_depth=0) const override;

    integer_t front_rank(int task_depth=0) const override;
    void print_rank_statistics(std::ostream &out) const override;
    bool isHSS() const override { return true; };
    std::string type() const override { return "FrontalMatrixHSS"; }

    int random_samples() const override { return R1.cols(); };

    void partition(const Opts_t& opts, const SpMat_t& A, integer_t* sorder,
                   bool is_root=true, int task_depth=0) override;


    // TODO make private?
    HSS::HSSMatrix<scalar_t> H_;

    // TODO do not store this here: makes solve not thread safe!!
    mutable std::unique_ptr<HSS::WorkSolve<scalar_t>> ULVwork_;

    /** Schur complement update:
     *    S = F22 - _Theta * Vhat^C * _Phi^C
     **/
    DenseM_t Theta_, Phi_, ThetaVhatC_or_VhatCPhiC_, DUB01_;

    /** these are saved during/after randomized compression and are
        then later used to sample the Schur complement when
        compressing the parent front */
    DenseM_t R1;        /* top of the random matrix used to construct
                           HSS matrix of this front */
    DenseM_t Sr2, Sc2;  /* bottom of the sample matrix used to
                           construct HSS matrix of this front */
    std::uint32_t sampled_columns_ = 0;

  private:
    FrontalMatrixHSS(const FrontalMatrixHSS&) = delete;
    FrontalMatrixHSS& operator=(FrontalMatrixHSS const&) = delete;

    void draw_node(std::ostream& of, bool is_root) const override;

    void multifrontal_factorization_node(const SpMat_t& A,
                                         const Opts_t& opts, int etree_level,
                                         int task_depth);

    void fwd_solve_node(DenseM_t& b, DenseM_t* work,
                        int etree_level, int task_depth) const;
    void bwd_solve_node(DenseM_t& y, DenseM_t* work,
                        int etree_level, int task_depth) const;

    long long node_factor_nonzeros() const override;

    using F_t::lchild_;
    using F_t::rchild_;
    using F_t::dim_sep;
    using F_t::dim_upd;
    using F_t::dim_blk;
    using F_t::sep_begin_;
    using F_t::sep_end_;
  };

} // end namespace strumpack

#endif // FRONTAL_MATRIX_HSS_HPP
