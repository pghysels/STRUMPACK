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
 */
#ifndef HSS_EXTRA_HPP
#define HSS_EXTRA_HPP

#include "dense/DenseMatrix.hpp"

namespace strumpack {
  namespace HSS {

    /**
     * Enumeration of possible states of an HSS matrix/node. This is
     * used in the adaptive HSS compression algorithms, where a node
     * can be untouched (it is not yet visited by the compression
     * algorithm), partially_compressed (a compression was attempted
     * but failed, so the adaptive algorithm will have to try again),
     * or can be successfully compressed.
     * \ingroup Enumerations
     */
    enum class State : char
      {UNTOUCHED='U',   /*!< Node was not yet visited by the
                           compression algorithm */
       PARTIALLY_COMPRESSED='P', /*!< Compression was attempted for
                                    this node, but failed. The
                                    adaptive compression should try
                                    again. */
       COMPRESSED='C'   /*!< This HSS node was succesfully
                           compressed. */
      };

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename T> std::pair<T,T>
    operator+(const std::pair<T,T>& l, const std::pair<T,T>& r) {
      return {l.first+r.first, l.second+r.second};
    }
#endif // DOXYGEN_SHOULD_SKIP_THIS


#ifndef DOXYGEN_SHOULD_SKIP_THIS
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


    template<typename scalar_t,
             typename real_t=typename RealType<scalar_t>::value_type>
    class WorkCompressANN :
      public WorkCompressBase<scalar_t>  {
    public:
      std::vector<WorkCompressANN<scalar_t>> c;
      DenseMatrix<scalar_t> S;
      std::vector<std::pair<std::size_t,real_t>> ids_scores;
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
#endif // DOXYGEN_SHOULD_SKIP_THIS


    /**
     * \class HSSFactors
     * \brief Contains data related to ULV factorization of an HSS
     * matrix.
     *
     * Class containing data regarding the ULV factorization of an HSS
     * matrix. This is constructed inside the ULV factorization
     * routine and should be passed to the HSS solve routine (along
     * with the original HSS matrix).
     */
    template<typename scalar_t> class HSSFactors {
    public:

      /**
       * Get the amount of memory used to store this data (excluding
       * any metadata). To get the memory for the entire
       * factorization, you should also count the memory of the
       * original HSS matrix, as that is still required to perform a
       * solve.
       */
      std::size_t memory() {
        return sizeof(*this) + L_.memory() + Vt0_.memory()
          + W1_.memory() + Q_.memory() + D_.memory()
          + sizeof(int)*piv_.size();
      }

      /**
       * Get the number of nonzeros in this data. To get the total
       * number of nonzeros for the entire factorization, you should
       * also count the nonzeros of the original HSS matrix, as that
       * is still required to perform a solve.
       */
      std::size_t nonzeros() const {
        return L_.nonzeros() + Vt0_.nonzeros() + W1_.nonzeros()
          + Q_.nonzeros() + D_.nonzeros();
      }

      /**
       * Used in the sparse solver to construct the Schur complement.
       */
      const DenseMatrix<scalar_t>& Vhat() const { return Vt0_; }
      /**
       * Used in the sparse solver to construct the Schur complement.
       */
      DenseMatrix<scalar_t>& Vhat() { return Vt0_; }

    private:
      DenseMatrix<scalar_t> L_;   // (U.rows-U.cols x U.rows-U.cols),
                                  //  empty at the root
      DenseMatrix<scalar_t> Vt0_; // (U.rows-U.cols x V.cols)
                                  // at the root, _Vt0 stored Vhat
      DenseMatrix<scalar_t> W1_;  // (U.cols x U.rows) bottom part of W
                                  // if (U.rows == U.cols)
                                  // then W == I and is not stored!
      DenseMatrix<scalar_t> Q_;   // (U.rows x U.rows) Q from LQ(W0)
                                  // if (U.rows == U.cols)
                                  // then Q == I and is not stored!
      DenseMatrix<scalar_t> D_;   // (U.rows x U.rows) at the root holds LU(D)
                                  // else empty
      std::vector<int> piv_;      // hold permutation from LU(D) at root
      template<typename T> friend class HSSMatrix;
      template<typename T> friend class HSSMatrixBase;
    };

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t> class WorkSolve {
    public:
      std::vector<WorkSolve<scalar_t>> c;

      // do we need all these?? x only used in bwd, y only used in fwd??
      DenseMatrix<scalar_t> z, ft1, y, x;

      // DO NOT STORE reduced_rhs here!!!
      DenseMatrix<scalar_t> reduced_rhs;
      std::pair<std::size_t,std::size_t> offset;
    };
#endif // DOXYGEN_SHOULD_SKIP_THIS


#ifndef DOXYGEN_SHOULD_SKIP_THIS
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
#endif // DOXYGEN_SHOULD_SKIP_THIS


#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t> class WorkDense {
    public:
      std::pair<std::size_t,std::size_t> offset;
      std::vector<WorkDense<scalar_t>> c;
      DenseMatrix<scalar_t> tmpU, tmpV;
    };
#endif // DOXYGEN_SHOULD_SKIP_THIS

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_EXTRA_HPP
