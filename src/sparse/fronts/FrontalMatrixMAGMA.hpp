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
#ifndef FRONTAL_MATRIX_MAGMA_HPP
#define FRONTAL_MATRIX_MAGMA_HPP

#include "FrontalMatrixDense.hpp"
#if defined(STRUMPACK_USE_CUDA)
#include "dense/CUDAWrapper.hpp"
#endif
#if defined(STRUMPACK_USE_HIP)
#include "dense/HIPWrapper.hpp"
#endif


namespace strumpack {

  template<typename scalar_t, typename integer_t> class LevelInfoMAGMA;


  template<typename scalar_t,typename integer_t> class FrontalMatrixMAGMA
    : public FrontalMatrix<scalar_t,integer_t> {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FM_t = FrontalMatrixMAGMA<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using LInfo_t = LevelInfoMAGMA<scalar_t,integer_t>;

  public:
    FrontalMatrixMAGMA(integer_t sep, integer_t sep_begin, integer_t sep_end,
                     std::vector<integer_t>& upd);
    ~FrontalMatrixMAGMA();

    void release_work_memory() override;

    void extend_add_to_dense(DenseM_t& paF11, DenseM_t& paF12,
                             DenseM_t& paF21, DenseM_t& paF22,
                             const F_t* p, int task_depth) override;

    ReturnCode multifrontal_factorization(const SpMat_t& A,
                                          const SPOptions<scalar_t>& opts,
                                          int etree_level=0,
                                          int task_depth=0) override;

    void extract_CB_sub_matrix(const std::vector<std::size_t>& I,
                               const std::vector<std::size_t>& J,
                               DenseM_t& B, int task_depth) const override {}

    std::string type() const override { return "FrontalMatrixMAGMA"; }

#if defined(STRUMPACK_USE_MPI)
    void
    extend_add_copy_to_buffers(std::vector<std::vector<scalar_t>>& sbuf,
                               const FrontalMatrixMPI<scalar_t,integer_t>* pa)
      const override;
#endif

  private:
    std::unique_ptr<scalar_t[]> host_factors_, host_Schur_;
    DenseMW_t F11_, F12_, F21_, F22_;
    std::vector<int> pivot_mem_;
    int* piv_ = nullptr;
    std::unique_ptr<gpu::DeviceMemory<char>> dev_factors_ = nullptr;

    FrontalMatrixMAGMA(const FrontalMatrixMAGMA&) = delete;
    FrontalMatrixMAGMA& operator=(FrontalMatrixMAGMA const&) = delete;

    void front_assembly(const SpMat_t& A, LInfo_t& L,
                        char* hea_mem, char* dea_mem);

    ReturnCode
    split_smaller(const SpMat_t& A, const SPOptions<scalar_t>& opts,
                  int etree_level=0, int task_depth=0);
    ReturnCode
    factors_on_device(const SpMat_t& A, const SPOptions<scalar_t>& opts,
                      std::vector<LInfo_t>& ldata, std::size_t total_dmem);

    void multifrontal_solve(DenseM_t& b) const override;

    void fwd_solve_phase2(DenseM_t& b, DenseM_t& bupd,
                          int etree_level, int task_depth) const;
    void bwd_solve_phase1(DenseM_t& y, DenseM_t& yupd,
                          int etree_level, int task_depth) const;

    void gpu_solve(DenseM_t& b) const;

    using F_t::lchild_;
    using F_t::rchild_;
    using F_t::dim_sep;
    using F_t::dim_upd;

    template<typename T,typename I> friend class LevelInfoMAGMA;
  };

} // end namespace strumpack

#endif // FRONTAL_MATRIX_MAGMA_HPP
