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
#ifndef FRONTAL_MATRIX_LOSSY_HPP
#define FRONTAL_MATRIX_LOSSY_HPP

#include "FrontalMatrixDense.hpp"
#include "structured/StructuredMatrix.hpp"

namespace strumpack {

  template<typename T> class LossyMatrix
    : public structured::StructuredMatrix<T> {
  public:
    LossyMatrix() {}
    LossyMatrix(const DenseMatrix<T>& F, int prec);
    DenseMatrix<T> decompress() const {
      DenseMatrix<T> F(rows_, cols_);
      decompress(F);
      return F;
    }
    virtual ~LossyMatrix() {
      STRUMPACK_SUB_MEMORY(buffer_.size()*sizeof(unsigned char));
    }
    void decompress(DenseMatrix<T>& F) const;
    std::size_t compressed_size() const { return buffer_.size(); }
    std::size_t memory() const override { return compressed_size(); }
    std::size_t nonzeros() const override { return rows()*cols(); }
    std::size_t rank() const override { return std::min(rows(), cols()); }
    std::size_t rows() const override { return rows_; }
    std::size_t cols() const override { return cols_; }
  private:
    std::size_t rows_ = 0, cols_ = 0;
    int prec_ = 16;
    std::vector<unsigned char> buffer_;
  };

  template<typename T> class LossyMatrix<std::complex<T>>
    : public structured::StructuredMatrix<std::complex<T>> {
  public:
    LossyMatrix() {}
    LossyMatrix(const DenseMatrix<std::complex<T>>& F, int prec);
    DenseMatrix<std::complex<T>> decompress() const {
      DenseMatrix<std::complex<T>> F(rows(), cols());
      decompress(F);
      return F;
    }
    void decompress(DenseMatrix<std::complex<T>>& F) const;
    std::size_t compressed_size() const {
      return Freal_.compressed_size() + Fimag_.compressed_size();
    }
    std::size_t memory() const override { return compressed_size(); }
    std::size_t nonzeros() const override { return rows()*cols(); }
    std::size_t rank() const override { return std::min(rows(), cols()); }
    std::size_t rows() const override { return Freal_.rows(); }
    std::size_t cols() const override { return Freal_.cols(); }
  private:
    LossyMatrix<T> Freal_, Fimag_;
  };


  template<typename scalar_t,typename integer_t> class FrontalMatrixLossy
    : public FrontalMatrixDense<scalar_t,integer_t> {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FD_t = FrontalMatrixDense<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using real_t = typename RealType<scalar_t>::value_type;
    using Opts_t = SPOptions<scalar_t>;

  public:
    FrontalMatrixLossy(integer_t sep, integer_t sep_begin, integer_t sep_end,
                       std::vector<integer_t>& upd);

    ReturnCode factor(const SpMat_t& A, const SPOptions<scalar_t>& opts,
                      VectorPool<scalar_t>& workspace,
                      int etree_level=0, int task_depth=0) override;

    std::string type() const override { return "FrontalMatrixLossy"; }

    void compress(const Opts_t& opts);
    void decompress(DenseM_t& F11, DenseM_t& F12, DenseM_t& F21) const;
    bool compressible(const Opts_t& opts) const;

    long long node_factor_nonzeros() const override;

  private:
    LossyMatrix<scalar_t> F11c_, F12c_, F21c_;

    void fwd_solve_phase2(DenseM_t& b, DenseM_t& bupd,
                          int etree_level, int task_depth) const override;
    void bwd_solve_phase1(DenseM_t& y, DenseM_t& yupd,
                          int etree_level, int task_depth) const override;

    virtual ReturnCode node_inertia(integer_t& neg,
                                    integer_t& zero,
                                    integer_t& pos) const override;

    FrontalMatrixLossy(const FrontalMatrixLossy&) = delete;
    FrontalMatrixLossy& operator=(FrontalMatrixLossy const&) = delete;
  };

} // end namespace strumpack

#endif // FRONTAL_MATRIX_LOSSY_HPP
