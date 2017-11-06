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
#ifndef HSS_MATRIX_HPP
#define HSS_MATRIX_HPP

#include <cassert>
#include <utility>
#include <fstream>
#include <string>

#include "DenseMatrix.hpp"
#include "HOptions.hpp"

namespace strumpack {
  namespace H {

    class HBlockPartition {
    public:
      HBlockPartition() {}
      HBlockPartition(std::size_t n)
        : _lo(0), _hi(n) {}
      HBlockPartition(std::size_t lo, std::size_t hi)
        : _lo(lo), _hi(hi) {}
      std::size_t size() const { return _hi - _lo; }
      std::size_t lo() const { return _lo; }
      std::size_t hi() const { return _hi; }
      bool leaf() const { return _c.empty(); }
      const HBlockPartition& c(std::size_t i) const {
        assert(i < _c.size());
        return _c[i];
      }
      HBlockPartition& c(std::size_t i) {
        assert(i < _c.size());
        return _c[i];
      }
      void refine(std::size_t leaf_size) {
        assert(_c.empty());
        if (size() > 2*leaf_size) {
          _c.reserve(2);
          _c.emplace_back(_lo, _lo+size()/2);
          _c.emplace_back(_lo+size()/2, _hi);
          _c[0].refine(leaf_size);
          _c[1].refine(leaf_size);
        }
      }

    private:
      std::size_t _lo = 0, _hi = 0;
      std::vector<HBlockPartition> _c;
    };

    struct WeakAdmissibility {
      bool operator()(const HBlockPartition& a,
                      const HBlockPartition& b) {
        // .. or equal, meaning the parts can touch and still be
        // admissible
        return a.lo() >= b.hi() || a.hi() <= b.lo();
      }
    };

    struct StrongAdmissibility {
      double eta;
      bool operator()(const HBlockPartition& a,
                      const HBlockPartition& b) {
        // parts must be separated to be admissible
        return a.lo() > b.hi() || a.hi() < b.lo();
      }
    };

    template<typename scalar_t> class HMatrixBase {
      using real_t = typename RealType<scalar_t>::value_type;
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using opts_t = HOptions<scalar_t>;
      using adm_t = std::function<bool(const HBlockPartition&,
                                       const HBlockPartition&)>;
    public:
      static std::unique_ptr<HMatrixBase<scalar_t>>
      compress(const DenseM_t& A,
               const HBlockPartition& row_part,
               const HBlockPartition& col_part,
               const adm_t& admissible,
               const opts_t& opts);

      virtual std::size_t rows() const = 0;
      virtual std::size_t cols() const = 0;
      virtual std::size_t rank() const = 0;
      virtual std::size_t memory() const = 0;
      virtual std::size_t levels() const = 0;
      virtual DenseM_t dense() const = 0;

    protected:
      virtual void draw(std::ostream& of,
                        std::size_t rlo=0, std::size_t clo=0) const = 0;
      virtual std::vector<int> LU(int task_depth) = 0;
      virtual void permute_rows_fwd(const std::vector<int>& piv) = 0;
      virtual void call_gemm_A_D_D() const = 0;

      template<typename T>
      friend void draw(std::unique_ptr<HMatrixBase<T>> const&,
                       const std::string&);
      template<typename T> friend
      std::vector<int> LU(std::unique_ptr<HMatrixBase<T>> const&);
      template<typename T> friend class HMatrix;
      template<typename T> friend class HMatrixLR;
      template<typename T> friend class HMatrixDense;
    };

    template<typename scalar_t>
    void draw(std::unique_ptr<HMatrixBase<scalar_t>> const& H,
              const std::string& name) {
      std::ofstream of("plot" + name + ".gnuplot");
      of << "set terminal pdf enhanced color size 5,4" << std::endl;
      of << "set output '" << name << ".pdf'" << std::endl;
      H->draw(of);
      of << "set xrange [0:" << H->cols() << "]" << std::endl;
      of << "set yrange [" << H->rows() << ":0]" << std::endl;
      of << "plot x lt -1 notitle" << std::endl;
      of.close();
    }

    template<typename scalar_t> std::vector<int>
    LU(std::unique_ptr<HMatrixBase<scalar_t>> const& H) {
      return H->LU(0);
    }

    template<typename scalar_t> class HMatrix
      : public HMatrixBase<scalar_t> {
      using real_t = typename RealType<scalar_t>::value_type;
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using opts_t = HOptions<scalar_t>;

    public:
      HMatrix(std::unique_ptr<HMatrixBase<scalar_t>> H00,
              std::unique_ptr<HMatrixBase<scalar_t>> H01,
              std::unique_ptr<HMatrixBase<scalar_t>> H10,
              std::unique_ptr<HMatrixBase<scalar_t>> H11)
        : _H00(std::move(H00)), _H01(std::move(H01)),
          _H10(std::move(H10)), _H11(std::move(H11)) {}

      std::size_t rows() const { return H00().rows() + H10().rows(); }
      std::size_t cols() const { return H00().cols() + H01().cols(); }
      std::size_t rank() const;
      std::size_t memory() const;
      std::size_t levels() const;
      DenseM_t dense() const;

      HMatrixBase<scalar_t>& H00() { return *_H00; }
      HMatrixBase<scalar_t>& H01() { return *_H01; }
      HMatrixBase<scalar_t>& H10() { return *_H10; }
      HMatrixBase<scalar_t>& H11() { return *_H11; }
      const HMatrixBase<scalar_t>& H00() const { return *_H00; }
      const HMatrixBase<scalar_t>& H01() const { return *_H01; }
      const HMatrixBase<scalar_t>& H10() const { return *_H10; }
      const HMatrixBase<scalar_t>& H11() const { return *_H11; }

    protected:
      std::unique_ptr<HMatrixBase<scalar_t>> _H00;
      std::unique_ptr<HMatrixBase<scalar_t>> _H01;
      std::unique_ptr<HMatrixBase<scalar_t>> _H10;
      std::unique_ptr<HMatrixBase<scalar_t>> _H11;
      void draw(std::ostream& of, std::size_t rlo=0, std::size_t clo=0) const;
      std::vector<int> LU(int task_depth);
      void permute_rows_fwd(const std::vector<int>& piv);
      void call_gemm_A_D_D(Trans ta, Trans tb, scalar_t alpha,
                           const DenseMatrix<scalar_t>& b,
                           scalar_t beta, DenseMatrix<scalar_t>& c,
                           int depth=0) const;

      template<typename T>
      friend void draw(std::unique_ptr<HMatrixBase<T>> const&,
                       const std::string&);
    };
    template<typename scalar_t> std::size_t
    HMatrix<scalar_t>::rank() const {
      return std::max(std::max(H00().rank(), H01().rank()),
                      std::max(H10().rank(), H11().rank()));
    }
    template<typename scalar_t> std::size_t
    HMatrix<scalar_t>::memory() const {
      return H00().memory() + H01().memory() +
        H10().memory() + H11().memory();
      }
    template<typename scalar_t> std::size_t
    HMatrix<scalar_t>::levels() const {
      return 1 + std::max(std::max(H00().levels(), H01().levels()),
                          std::max(H10().levels(), H11().levels()));
    }
    template<typename scalar_t> DenseMatrix<scalar_t>
    HMatrix<scalar_t>::dense() const {
      DenseM_t out(rows(), cols());
      DenseMW_t out00(H00().rows(), H00().cols(), out, 0, 0);
      DenseMW_t out01(H01().rows(), H01().cols(), out, 0, H00().cols());
      DenseMW_t out10(H10().rows(), H10().cols(), out, H00().rows(), 0);
      DenseMW_t out11(H11().rows(), H11().cols(), out,
                      H00().rows(), H00().cols());
      out00 = H00().dense();
      out01 = H01().dense();
      out10 = H10().dense();
      out11 = H11().dense();
      return out;
    }
    template<typename scalar_t> void
    HMatrix<scalar_t>::draw(std::ostream& of,
                            std::size_t rlo, std::size_t clo) const {
      H00().draw(of, rlo, clo);
      H01().draw(of, rlo, clo+H00().cols());
      H10().draw(of, rlo+H00().rows(), clo);
      H11().draw(of, rlo+H00().rows(), clo+H00().cols());
    }
    template<typename scalar_t> std::vector<int>
    HMatrix<scalar_t>::LU(int task_depth) {
      auto piv0 = H00().LU(task_depth);
      H01().permute_rows_fwd(piv0);
      // trsm(Side::L, UpLo::L, Trans::N, Diag::U,
      //      scalar_t(1.), H00(), H01(), task_depth);
      // trsm(Side::R, UpLo::U, Trans::N, Diag::N,
      //      scalar_t(1.), H00(), H10(), task_depth);
      // gemm(Trans::N, Trans::N, scalar_t(-1.), H10(), H01(),
      //      scalar_t(1.), H11(), task_depth);
      auto piv1 = H11().LU(task_depth);
      piv1.reserve(piv0.size() + piv1.size());
      auto m0 = H00().rows();
      for (auto i : piv1) piv0.push_back(i + m0);
      return piv0;
    }
    template<typename scalar_t> void
    HMatrix<scalar_t>::permute_rows_fwd(const std::vector<int>& piv) {
      assert(piv.size() == rows());
      std::vector<int> piv0(piv.begin(), piv.begin()+H00().rows());
      std::vector<int> piv1(piv.begin()+H00().rows(), piv.end());
      auto m0 = H00().rows();
      for (auto& i : piv1) i -= m0;
      H00().permute_rows_fwd(piv0);  H01().permute_rows_fwd(piv0);
      H10().permute_rows_fwd(piv1);  H11().permute_rows_fwd(piv1);
    }
    template<typename scalar_t> void
    HMatrix<scalar_t>::call_gemm_A_D_D
    (Trans ta, Trans tb, scalar_t alpha, const DenseMatrix<scalar_t>& b,
     scalar_t beta, DenseMatrix<scalar_t>& c, int depth) const {
      gemm(ta, tb, alpha, *this, b, beta, c, depth);
    }

    template<typename scalar_t> class HMatrixLR
      : public HMatrixBase<scalar_t> {
      using real_t = typename RealType<scalar_t>::value_type;
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using opts_t = HOptions<scalar_t>;

    public:
      HMatrixLR(const DenseM_t& A, const opts_t& opts);

      std::size_t rows() const { return _U.rows(); }
      std::size_t cols() const { return _V.rows(); }
      std::size_t rank() const { return _U.cols(); }
      std::size_t memory() const { return _U.memory() + _V.memory(); }
      std::size_t levels() const { return 1; }
      DenseM_t dense() const;

      DenseM_t& U() { return _U; }
      DenseM_t& V() { return _V; }
      const DenseM_t& U() const { return _U; }
      const DenseM_t& V() const { return _V; }

    protected:
      DenseM_t _U;
      DenseM_t _V;
      void draw(std::ostream& of, std::size_t rlo=0, std::size_t clo=0) const;
      std::vector<int> LU(int task_depth);
      void permute_rows_fwd(const std::vector<int>& piv);

      template<typename T> friend
      void draw(std::unique_ptr<HMatrixBase<T>> const&, const std::string&);
    };

    template<typename scalar_t>
    HMatrixLR<scalar_t>::HMatrixLR(const DenseM_t& A, const opts_t& opts) {
      A.low_rank(_U, _V, opts.rel_tol(), opts.abs_tol(), opts.max_rank(), 0);
    }
    template<typename scalar_t> DenseMatrix<scalar_t>
    HMatrixLR<scalar_t>::dense() const {
      DenseM_t out(rows(), cols());
      gemm(Trans::N, Trans::C, scalar_t(1.), _U, _V, scalar_t(0.), out);
      return out;
    }
    template<typename scalar_t> void
    HMatrixLR<scalar_t>::draw(std::ostream& of,
                              std::size_t rlo, std::size_t clo) const {
      int minmn = std::min(rows(), cols());
      int red = std::floor(255.0 * rank() / minmn);
      assert(red < 256 && red >= 0);
      int blue = 255 - red;
      assert(blue < 256 && blue >= 0);
      char prev = std::cout.fill('0');
      of << "set obj rect from "
         << rlo << ", " << clo << " to "
         << rlo+rows() << ", " << clo+cols()
         << " fc rgb '#"
         << std::hex << std::setw(2) << std::setfill('0') << red
         << "00" << std::setw(2)  << std::setfill('0') << blue
         << "'" << std::dec << std::endl;
      std::cout.fill(prev);
    }
    template<typename scalar_t> std::vector<int>
    HMatrixLR<scalar_t>::LU(int task_depth) {
      assert(false);
      return std::vector<int>();
    }
    template<typename scalar_t> void
    HMatrixLR<scalar_t>::permute_rows_fwd(const std::vector<int>& piv) {
      assert(piv.size() == rows());
      _U.permute_rows_fwd(piv);
    }

    template<typename scalar_t> class HMatrixDense
      : public HMatrixBase<scalar_t> {
      using real_t = typename RealType<scalar_t>::value_type;
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using opts_t = HOptions<scalar_t>;

    public:
      HMatrixDense(const DenseM_t& A) : _D(A) {}

      std::size_t rows() const { return _D.rows(); }
      std::size_t cols() const { return _D.cols(); }
      std::size_t rank() const { return 0; /*std::min(rows(), cols());*/ }
      std::size_t memory() const { return _D.memory(); }
      std::size_t levels() const { return 1; }
      DenseM_t dense() const { return _D; }

      DenseM_t& D() { return _D; }
      const DenseM_t& D() const { return _D; }

    protected:
      DenseM_t _D;
      void draw(std::ostream& of, std::size_t rlo=0, std::size_t clo=0) const;
      std::vector<int> LU(int task_depth);
      void permute_rows_fwd(const std::vector<int>& piv);

      template<typename T> friend
      void draw(std::unique_ptr<HMatrixBase<T>> const&, const std::string&);
    };

    template<typename scalar_t> void
    HMatrixDense<scalar_t>::draw(std::ostream& of,
                                 std::size_t rlo, std::size_t clo) const {
      of << "set obj rect from "
         << rlo << ", " << clo << " to "
         << rlo+rows() << ", " << clo+cols()
         << " fc rgb 'red'" << std::endl;
    }
    template<typename scalar_t> std::vector<int>
    HMatrixDense<scalar_t>::LU(int task_depth) {
      return _D.LU(task_depth);
    }
    template<typename scalar_t> void
    HMatrixDense<scalar_t>::permute_rows_fwd(const std::vector<int>& piv) {
      assert(piv.size() == rows());
      _D.permute_rows_fwd(piv);
    }

    template<typename scalar_t> std::unique_ptr<HMatrixBase<scalar_t>>
    HMatrixBase<scalar_t>::compress(const DenseM_t& A,
                                    const HBlockPartition& row_part,
                                    const HBlockPartition& col_part,
                                    const adm_t& admissible,
                                    const opts_t& opts) {
      if (row_part.leaf() || col_part.leaf()) {
        if (admissible(row_part, col_part))
          return std::unique_ptr<HMatrixLR<scalar_t>>
            (new HMatrixLR<scalar_t>(A, opts));
        else
          return std::unique_ptr<HMatrixDense<scalar_t>>
            (new HMatrixDense<scalar_t>(A));
      } else {
        if (admissible(row_part, col_part))
          return std::unique_ptr<HMatrixLR<scalar_t>>
            (new HMatrixLR<scalar_t>(A, opts));
        else {
          auto Anc = const_cast<DenseM_t&>(A);
          DenseMW_t A00(row_part.c(0).size(), col_part.c(0).size(),
                        Anc, 0, 0);
          DenseMW_t A01(row_part.c(0).size(), col_part.c(1).size(),
                        Anc, 0, col_part.c(0).size());
          DenseMW_t A10(row_part.c(1).size(), col_part.c(0).size(),
                        Anc, row_part.c(0).size(), 0);
          DenseMW_t A11(row_part.c(1).size(), col_part.c(1).size(),
                        Anc, row_part.c(0).size(), col_part.c(0).size());
          return std::unique_ptr<HMatrix<scalar_t>>
            (new HMatrix<scalar_t>
             (HMatrixBase<scalar_t>::compress
              (A00, row_part.c(0), col_part.c(0), admissible, opts),
              HMatrixBase<scalar_t>::compress
              (A01, row_part.c(0), col_part.c(1), admissible, opts),
              HMatrixBase<scalar_t>::compress
              (A10, row_part.c(1), col_part.c(0), admissible, opts),
              HMatrixBase<scalar_t>::compress
              (A11, row_part.c(1), col_part.c(1), admissible, opts)));
        }
      }
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixBase<scalar_t>& a, const HMatrixBase<scalar_t>& b,
         scalar_t beta, HMatrixBase<scalar_t>& c, int depth=0) {
      // for now check the type with if statments??

      //c.add_multiply(ta, tb, alpha, a, b, beta, depth);
    }



    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b, beta, c, depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      using DenseM_t = DenseMatrix<scalar_t>;
      // if real, then ta should be N or T, not C
      // if complex, then ta should be N or C, not T
      if (ta == Trans::N) {
        if (tb == Trans::N) {
          DenseM_t tmp(a.V().cols(), b.cols());
          gemm(Trans::C, Trans::N, scalar_t(1.), a.V(), b,
               scalar_t(0.), tmp, depth);
          gemm(Trans::N, Trans::N, scalar_t(1.), a.U(), tmp,
               scalar_t(0.), c, depth);
        } else {
          DenseM_t tmp(a.V().cols(), b.rows());
          gemm(Trans::C, tb, scalar_t(1.), a.V(), b,
               scalar_t(0.), tmp, depth);
          gemm(Trans::N, Trans::N, scalar_t(1.), a.U(), tmp,
               scalar_t(0.), c, depth);
        }
      } else {
        if (tb == Trans::N) {
          DenseM_t tmp(a.V().cols(), b.cols());
          gemm(Trans::C, Trans::N, scalar_t(1.), a.U(), b,
               scalar_t(0.), tmp, depth);
          gemm(Trans::N, Trans::N, scalar_t(1.), a.V(), tmp,
               scalar_t(0.), c, depth);
        } else {
          DenseM_t tmp(a.V().cols(), b.rows());
          gemm(Trans::C, tb, scalar_t(1.), a.U(), b,
               scalar_t(0.), tmp, depth);
          gemm(Trans::N, Trans::N, scalar_t(1.), a.V(), tmp,
               scalar_t(0.), c, depth);
        }
      }
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      if (ta == Trans::N) {
        if (tb == Trans::N) {
          DenseMW_t c0(a.H00().rows(), b.cols(), c, 0, 0);
          DenseMW_t c1(a.H10().rows(), b.cols(), c,
                      a.H00().rows(), 0);
          DenseMW_t b0(a.H00().cols(), b.cols(), b, 0, 0);
          DenseMW_t b1(a.H01().cols(), b.cols(), b,
                      a.H00().cols(), 0);
          gemm(Trans::N, Trans::N, alpha, a.H00(), b0, beta, c0);
          gemm(Trans::N, Trans::N, alpha, a.H01(), b1, scalar_t(1.), c0);
          gemm(Trans::N, Trans::N, alpha, a.H10(), b0, beta, c1);
          gemm(Trans::N, Trans::N, alpha, a.H11(), b1, scalar_t(1.), c1);
        } else {
          assert(true);
          // TODO
        }
      } else {
        assert(true);
        // TODO
      }
    }

  } // end namespace H
} // end namespace strumpack

#endif // H_MATRIX_HPP
