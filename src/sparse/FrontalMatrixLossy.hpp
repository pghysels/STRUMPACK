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
#include "zfp.h"
#include "zfparray2.h"

namespace strumpack {

  template<typename T> zfp_type get_zfp_type();
  template<> inline zfp_type get_zfp_type<float>() { return zfp_type_float; }
  template<> inline zfp_type get_zfp_type<double>() { return zfp_type_double; }

  template<typename T> class LossyMatrix {
  public:
    LossyMatrix() {}
    LossyMatrix(const DenseMatrix<T>& F, uint prec)
      : rows_(F.rows()), cols_(F.cols()), prec_(prec) {
      if (!rows_ || !cols_) return;
      zfp_field* f = zfp_field_2d
        (static_cast<void*>(const_cast<T*>(F.data())),
         get_zfp_type<T>(), rows_, cols_);
      zfp_stream* stream = zfp_stream_open(NULL);
      zfp_stream_set_precision(stream, prec_);
      auto bufsize = zfp_stream_maximum_size(stream, f);
      buffer_.resize(bufsize);
      bitstream* bstream = stream_open(buffer_.data(), bufsize);
      zfp_stream_set_bit_stream(stream, bstream);
      zfp_stream_rewind(stream);
      auto comp_size = zfp_compress(stream, f);
      buffer_.resize(comp_size);
      zfp_stream_flush(stream);
      zfp_field_free(f);
      zfp_stream_close(stream);
      stream_close(bstream);
    }
    DenseMatrix<T> decompress() const {
      DenseMatrix<T> F(rows_, cols_);
      decompress(F);
      return F;
    }
    void decompress(DenseMatrix<T>& F) const {
      assert(F.rows() == rows_ && F.cols() == cols_);
      if (!rows_ || !cols_) return;
      zfp_field* f = zfp_field_2d
        (static_cast<void*>(F.data()), get_zfp_type<T>(), rows_, cols_);
      zfp_stream* destream = zfp_stream_open(NULL);
      zfp_stream_set_precision(destream, prec_);
      bitstream* bstream = stream_open
        (static_cast<void*>
         (const_cast<uchar*>(buffer_.data())), buffer_.size());
      zfp_stream_set_bit_stream(destream, bstream);
      zfp_stream_rewind(destream);
      zfp_decompress(destream, f);
      zfp_field_free(f);
      zfp_stream_close(destream);
      stream_close(bstream);
      return;
    }
    std::size_t compressed_size() const {
      return buffer_.size();
    }
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
  private:
    std::size_t rows_ = 0, cols_ = 0;
    uint prec_ = 16;
    std::vector<uchar> buffer_;
  };

  template<typename T> class LossyMatrix<std::complex<T>> {
  public:
    LossyMatrix() {}
    LossyMatrix(const DenseMatrix<std::complex<T>>& F, uint prec) {
      int rows = F.rows(), cols = F.cols();
      DenseMatrix<T> Freal(rows, cols), Fimag(rows, cols);
      for (int j=0; j<cols; j++)
        for (int i=0; i<rows; i++) {
          Freal(i, j) = F(i,j).real();
          Fimag(i, j) = F(i,j).imag();
        }
      Freal_ = LossyMatrix<T>(Freal, prec);
      Fimag_ = LossyMatrix<T>(Fimag, prec);
    }
    DenseMatrix<std::complex<T>> decompress() const {
      DenseMatrix<std::complex<T>> F(rows(), cols());
      decompress(F);
      return F;
    }
    void decompress(DenseMatrix<std::complex<T>>& F) const {
      auto Freal = Freal_.decompress();
      auto Fimag = Fimag_.decompress();
      int rows = Freal_.rows(), cols = Freal_.cols();
      for (int j=0; j<cols; j++)
        for (int i=0; i<rows; i++)
          F(i, j) = std::complex<T>(Freal(i,j), Fimag(i,j));
      return;
    }
    std::size_t compressed_size() const {
      return Freal_.compressed_size() + Fimag_.compressed_size();
    }
    std::size_t rows() const { return Freal_.rows(); }
    std::size_t cols() const { return Freal_.cols(); }
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

  public:
    FrontalMatrixLossy
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd);

    void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level=0, int task_depth=0) override;

    std::string type() const override { return "FrontalMatrixLossy"; }

    void compress(const SPOptions<scalar_t>& opts);
    void decompress(DenseM_t& F11, DenseM_t& F12, DenseM_t& F21) const;
    bool compressible(const SPOptions<scalar_t>& opts) const;

    long long node_factor_nonzeros() const override;

  private:
    LossyMatrix<scalar_t> F11c_, F12c_, F21c_;

    void fwd_solve_phase2
    (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const;
    void bwd_solve_phase1
    (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const;

    FrontalMatrixLossy(const FrontalMatrixLossy&) = delete;
    FrontalMatrixLossy& operator=(FrontalMatrixLossy const&) = delete;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixLossy<scalar_t,integer_t>::FrontalMatrixLossy
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : FD_t(sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixLossy<scalar_t,integer_t>::node_factor_nonzeros() const {
    return (F11c_.compressed_size() + F12c_.compressed_size() +
            F21c_.compressed_size()) / sizeof(scalar_t);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixLossy<scalar_t,integer_t>::compress
  (const SPOptions<scalar_t>& opts) {
    uint prec = opts.lossy_precision();
    F11c_ = LossyMatrix<scalar_t>(this->F11_, prec);
    F12c_ = LossyMatrix<scalar_t>(this->F12_, prec);
    F21c_ = LossyMatrix<scalar_t>(this->F21_, prec);
    this->F11_.clear();
    this->F12_.clear();
    this->F21_.clear();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixLossy<scalar_t,integer_t>::decompress
  (DenseM_t& F11, DenseM_t& F12, DenseM_t& F21) const {
    F11 = F11c_.decompress();
    F12 = F12c_.decompress();
    F21 = F21c_.decompress();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixLossy<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    FD_t::multifrontal_factorization(A, opts, etree_level, task_depth);
    compress(opts);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixLossy<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    DenseM_t F11, F12, F21;
    decompress(F11, F12, F21);
    //FD_t::fwd_solve_phase2(b, bupd, etree_level, task_depth);
    if (this->dim_sep()) {
      DenseMW_t bloc(this->dim_sep(), b.cols(), b, this->sep_begin_, 0);
      bloc.laswp(this->piv, true);
      if (b.cols() == 1) {
        trsv(UpLo::L, Trans::N, Diag::U, F11, bloc, task_depth);
        if (this->dim_upd())
          gemv(Trans::N, scalar_t(-1.), F21, bloc,
               scalar_t(1.), bupd, task_depth);
      } else {
        trsm(Side::L, UpLo::L, Trans::N, Diag::U,
             scalar_t(1.), F11, bloc, task_depth);
        if (this->dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F21, bloc,
               scalar_t(1.), bupd, task_depth);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixLossy<scalar_t,integer_t>::bwd_solve_phase1
  (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const {
    DenseM_t F11, F12, F21;
    decompress(F11, F12, F21);
    // FD_t::bwd_solve_phase1(y, yupd, etree_level, task_depth);
    if (this->dim_sep()) {
      DenseMW_t yloc(this->dim_sep(), y.cols(), y, this->sep_begin_, 0);
      if (y.cols() == 1) {
        if (this->dim_upd())
          gemv(Trans::N, scalar_t(-1.), F12, yupd,
               scalar_t(1.), yloc, task_depth);
        trsv(UpLo::U, Trans::N, Diag::N, F11, yloc, task_depth);
      } else {
        if (this->dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F12, yupd,
               scalar_t(1.), yloc, task_depth);
        trsm(Side::L, UpLo::U, Trans::N, Diag::N, scalar_t(1.),
             F11, yloc, task_depth);
      }
    }
  }

} // end namespace strumpack

#endif
