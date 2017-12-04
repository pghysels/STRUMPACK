#ifndef HSS_EXTRA_HPP
#define HSS_EXTRA_HPP

#include "dense/DenseMatrix.hpp"

namespace strumpack {
  namespace HSS {

    enum class State : char
      {UNTOUCHED='U', PARTIALLY_COMPRESSED='P', COMPRESSED='C'};

    template<typename T> std::pair<T,T>
    operator+(const std::pair<T,T>& l, const std::pair<T,T>& r) {
      return {l.first+r.first, l.second+r.second};
    }

    template<typename scalar_t> class WorkCompressBase {
    public:
      WorkCompressBase() {}
      virtual ~WorkCompressBase() {}
      virtual void split(const std::pair<std::size_t,std::size_t>& dim) = 0;

      std::vector<std::size_t> Ir, Ic, Jr, Jc;
      std::pair<std::size_t,std::size_t> offset;
      int lvl = 0;
      scalar_t U_r_max, V_r_max;
    };

    template<typename scalar_t> class WorkCompress :
      public WorkCompressBase<scalar_t>  {
    public:
      std::vector<WorkCompress<scalar_t>> c;
      // only needed in the new compression algorithm
      DenseMatrix<scalar_t> Qr, Qc;
      void split(const std::pair<std::size_t,std::size_t>& dim) {
        if (c.empty()) {
          c.resize(2);
          c[0].offset = this->offset;
          c[1].offset = this->offset + dim;
          c[0].lvl = c[1].lvl = this->lvl + 1;
        }
      }
    };

    template<typename scalar_t> class WorkApply {
    public:
      std::pair<std::size_t,std::size_t> offset;
      std::vector<WorkApply<scalar_t>> c;
      DenseMatrix<scalar_t> tmp1, tmp2;
    };

    template<typename scalar_t> class WorkExtract {
    public:
      std::vector<WorkExtract<scalar_t>> c;
      DenseMatrix<scalar_t> y, z;
      std::vector<std::size_t> I, J, rl2g, cl2g, ycols, zcols;
      void split_extraction_sets
      (const std::pair<std::size_t,std::size_t>& dim) {
        if (c.empty()) {
          c.resize(2);
          c[0].I.reserve(I.size());
          c[1].I.reserve(I.size());
          for (auto i : I)
            if (i < dim.first) c[0].I.push_back(i);
            else c[1].I.push_back(i - dim.first);
          c[0].J.reserve(J.size());
          c[1].J.reserve(J.size());
          for (auto j : J)
            if (j < dim.second) c[0].J.push_back(j);
            else c[1].J.push_back(j - dim.second);
        }
      }
    };

    template<typename scalar_t> class WorkFactor {
    public:
      std::vector<WorkFactor<scalar_t>> c;
      DenseMatrix<scalar_t> Dt;  // (U.cols x U.cols) \tilde(D)
      DenseMatrix<scalar_t> Vt1; // (U.cols x V.cols) bottom part of \tilde{V}
    };

    template<typename scalar_t> class HSSFactors {
    public:
      template<typename T> friend class HSSMatrix;
      template<typename T> friend class HSSMatrixBase;
      DenseMatrix<scalar_t>& Vhat() { return _Vt0; }
      const DenseMatrix<scalar_t>& Vhat() const { return _Vt0; }
      std::size_t memory() {
        std::size_t mem = sizeof(*this) + _L.memory() + _Vt0.memory()
          + _W1.memory() + _Q.memory() + _D.memory()
          + sizeof(int)*_piv.size();
        for (auto& c : _ch) mem += c.memory();
        return memory;
      }
      std::size_t nonzeros() const {
        std::size_t nnz = _L.nonzeros() + _Vt0.nonzeros() + _W1.nonzeros()
          + _Q.nonzeros() + _D.nonzeros();
        for (auto& c : _ch) nnz += c.nonzeros();
        return nnz;
      }
    private:
      std::vector<HSSFactors<scalar_t>> _ch;
      DenseMatrix<scalar_t> _L;   // (U.rows-U.cols x U.rows-U.cols),
                                  //  empty at the root
      DenseMatrix<scalar_t> _Vt0; // (U.rows-U.cols x V.cols)
                                  // at the root, _Vt0 stored Vhat
      DenseMatrix<scalar_t> _W1;  // (U.cols x U.rows) bottom part of W
                                  // if (U.rows == U.cols)
                                  // then W == I and is not stored!
      DenseMatrix<scalar_t> _Q;   // (U.rows x U.rows) Q from LQ(W0)
                                  // if (U.rows == U.cols)
                                  // then Q == I and is not stored!
      DenseMatrix<scalar_t> _D;   // (U.rows x U.rows) at the root holds LU(D)
                                  // else empty
      std::vector<int> _piv;      // hold permutation from LU(D) at root
    };

    template<typename scalar_t> class WorkSolve {
    public:
      std::vector<WorkSolve<scalar_t>> c;

      // do we need all these?? x only used in bwd, y only used in fwd??
      DenseMatrix<scalar_t> z;
      DenseMatrix<scalar_t> ft1;
      DenseMatrix<scalar_t> y;
      DenseMatrix<scalar_t> x;

      // DO NOT STORE reduced_rhs here!!!
      DenseMatrix<scalar_t> reduced_rhs;
      std::pair<std::size_t,std::size_t> offset;
    };

    template<typename scalar_t> class AFunctor {
      using DenseM_t = DenseMatrix<scalar_t>;
    public:
      AFunctor(const DenseM_t& A) : _A(A) {}
      const DenseM_t& _A;
      void operator()
      (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc) {
        gemm(Trans::N, Trans::N, scalar_t(1.), _A, Rr, scalar_t(0.), Sr);
        gemm(Trans::C, Trans::N, scalar_t(1.), _A, Rc, scalar_t(0.), Sc);
      }
      void operator()(const std::vector<size_t>& I,
                      const std::vector<size_t>& J, DenseM_t& B) {
        assert(I.size() == B.rows() && J.size() == B.cols());
        for (std::size_t j=0; j<J.size(); j++)
          for (std::size_t i=0; i<I.size(); i++) {
            assert(I[i] >= 0 && I[i] < _A.rows() &&
                   J[j] >= 0 && J[j] < _A.cols());
            B(i,j) = _A(I[i], J[j]);
          }
      }
    };

    template<typename scalar_t> class WorkDense {
    public:
      std::pair<std::size_t,std::size_t> offset;
      std::vector<WorkDense<scalar_t>> c;
      DenseMatrix<scalar_t> tmpU, tmpV;
    };

    template<typename scalar_t> class Triplet {
    public:
      int _r; int _c; scalar_t _v;
      Triplet() {}
      Triplet(int r, int c, scalar_t v): _r(r), _c(c), _v(v) {}
    };

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_EXTRA_HPP
