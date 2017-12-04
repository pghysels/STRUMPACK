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
#ifndef HSS_MATRIX_BASE_HPP
#define HSS_MATRIX_BASE_HPP

#include <cassert>
#include <utility>
#include <fstream>
#include <string>

#include "dense/DenseMatrix.hpp"
#include "HOptions.hpp"

namespace strumpack {
  namespace H {

    template<typename T> class HMatrix;
    template<typename T> class HMatrixLR;
    template<typename T> class HMatrixDense;

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
      bool operator==(const HBlockPartition& other) const {
        // TODO also check the children??
        return _lo == other._lo && _hi == other._hi;
      }
      bool operator!=(const HBlockPartition& other) const {
        return !(*this == other);
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
      bool operator()(const HBlockPartition& a,
                      const HBlockPartition& b) {
        // parts must be separated to be admissible
        return a.lo() > b.hi() || a.hi() < b.lo();
      }
    };

    struct BLRAdmissibility {
      int leaf_size;
      bool operator()(const HBlockPartition& a,
                      const HBlockPartition& b) {
        // parts must be separated to be admissible
        //return a.lo() > b.hi() || a.hi() < b.lo();
        return (a != b) && (a.leaf() || b.leaf());
      }
    };

    template<typename scalar_t> class HMatrixBase {
      using real_t = typename RealType<scalar_t>::value_type;
      using D_t = DenseMatrix<scalar_t>;
      using DW_t = DenseMatrixWrapper<scalar_t>;
      using opts_t = HOptions<scalar_t>;
      using adm_t = std::function<bool(const HBlockPartition&,
                                       const HBlockPartition&)>;
      using HB_t = HMatrixBase<scalar_t>;
      using HLR_t = HMatrixLR<scalar_t>;
      using HD_t = HMatrixDense<scalar_t>;
      using H_t = HMatrix<scalar_t>;

    public:
      static std::unique_ptr<HMatrixBase<scalar_t>>
      compress(const D_t& A, const HBlockPartition& row_part,
               const HBlockPartition& col_part,
               const adm_t& admissible, const opts_t& opts);

      virtual std::size_t rows() const = 0;
      virtual std::size_t cols() const = 0;
      virtual std::size_t rank() const = 0;
      virtual std::size_t memory() const = 0;
      virtual std::size_t levels() const = 0;
      virtual D_t dense() const = 0;
      virtual std::string name() const { return "HMatrixBase"; };

      virtual void AGEMM(Trans ta, Trans tb, scalar_t alpha, const HB_t& b,
                         scalar_t beta, HB_t& c, int depth=0) const = 0;

      virtual void ABGEMM(Trans ta, Trans tb, scalar_t alpha, const H_t& a,
                          scalar_t beta, HB_t& c, int depth=0) const = 0;
      virtual void ABGEMM(Trans ta, Trans tb, scalar_t alpha, const HLR_t& a,
                          scalar_t beta, HB_t& c, int depth=0) const = 0;
      virtual void ABGEMM(Trans ta, Trans tb, scalar_t alpha, const HD_t& a,
                          scalar_t beta, HB_t& c, int depth=0) const = 0;
      virtual void ABGEMM(Trans ta, Trans tb, scalar_t alpha, const D_t& a,
                          scalar_t beta, HB_t& c, int depth=0) const = 0;

      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const H_t& a,
                           const H_t& b, scalar_t beta, int depth=0) = 0;
      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const H_t& a,
                           const HLR_t& b, scalar_t beta, int depth=0) = 0;
      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const H_t& a,
                           const HD_t& b, scalar_t beta, int depth=0) = 0;
      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const H_t& a,
                           const D_t& b, scalar_t beta, int depth=0) = 0;

      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HLR_t& a,
                           const H_t& b, scalar_t beta, int depth=0) = 0;
      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HLR_t& a,
                           const HLR_t& b, scalar_t beta, int depth=0) = 0;
      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HLR_t& a,
                           const HD_t& b, scalar_t beta, int depth=0) = 0;
      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HLR_t& a,
                           const D_t& b, scalar_t beta, int depth=0) = 0;

      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HD_t& a,
                           const H_t& b, scalar_t beta, int depth=0) = 0;
      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HD_t& a,
                           const HLR_t& b, scalar_t beta, int depth=0) = 0;
      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HD_t& a,
                           const HD_t& b, scalar_t beta, int depth=0) = 0;
      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const HD_t& a,
                           const D_t& b, scalar_t beta, int depth=0) = 0;

      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const D_t& a,
                           const H_t& b, scalar_t beta, int depth=0) = 0;
      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const D_t& a,
                           const HLR_t& b, scalar_t beta, int depth=0) = 0;
      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const D_t& a,
                           const HD_t& b, scalar_t beta, int depth=0) = 0;
      virtual void ABCGEMM(Trans ta, Trans tb, scalar_t alpha, const D_t& a,
                           const D_t& b, scalar_t beta, int depth=0) = 0;

    protected:
      virtual void draw
      (std::ostream& of, std::size_t rlo=0, std::size_t clo=0) const = 0;
      virtual std::vector<int> LU(int task_depth) = 0;
      virtual void laswp(const std::vector<int>& piv, bool fwd) = 0;

      template<typename T> friend
      void draw(std::unique_ptr<HMatrixBase<T>> const&, const std::string&);
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

  } // end namespace H
} // end namespace strumpack

#include "HMatrix.hpp"
#include "HMatrixLR.hpp"
#include "HMatrixDense.hpp"
#include "HMatrix.gemm.hpp"

namespace strumpack {
  namespace H {

    template<typename scalar_t> std::unique_ptr<HMatrixBase<scalar_t>>
    HMatrixBase<scalar_t>::compress
    (const D_t& A, const HBlockPartition& row_part,
     const HBlockPartition& col_part,
     const adm_t& admissible, const opts_t& opts) {
      if (row_part.leaf() || col_part.leaf()) {
        if (admissible(row_part, col_part))
          return std::unique_ptr<HMatrixLR<scalar_t>>
            (new HMatrixLR<scalar_t>(A, opts));
        else
          return std::unique_ptr<HMatrixDense<scalar_t>>
            (new HMatrixDense<scalar_t>(A, opts));
      } else {
        if (admissible(row_part, col_part))
          return std::unique_ptr<HMatrixLR<scalar_t>>
            (new HMatrixLR<scalar_t>(A, opts));
        else {
          auto Anc = const_cast<D_t&>(A);
          DW_t A00(row_part.c(0).size(), col_part.c(0).size(),
                   Anc, 0, 0);
          DW_t A01(row_part.c(0).size(), col_part.c(1).size(),
                   Anc, 0, col_part.c(0).size());
          DW_t A10(row_part.c(1).size(), col_part.c(0).size(),
                   Anc, row_part.c(0).size(), 0);
          DW_t A11(row_part.c(1).size(), col_part.c(1).size(),
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
  } // end namespace H
} // end namespace strumpack

#endif // H_MATRIX_BASE_HPP
