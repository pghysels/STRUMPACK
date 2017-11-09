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
#ifndef HSS_MATRIX_GEMM_HPP
#define HSS_MATRIX_GEMM_HPP

#include <cassert>
#include <utility>
#include <fstream>
#include <string>

#define DEBUG_GEMM

namespace strumpack {
  namespace H {

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixBase<scalar_t>& a, const HMatrixBase<scalar_t>& b,
         scalar_t beta, HMatrixBase<scalar_t>& c, int depth=0) {
      a.AGEMM(ta, tb, alpha, b, beta, c, depth);
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const HMatrixBase<scalar_t>& b,
         scalar_t beta, HMatrixBase<scalar_t>& c, int depth=0) {
      b.ABGEMM(ta, tb, alpha, a, beta, c, depth);
    }


    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      if (ta == Trans::N && tb == Trans::N) {
        a.H00().AGEMM(ta, tb, alpha, b.H00(), beta, c.H00(), depth);
        a.H01().AGEMM(ta, tb, alpha, b.H10(), scalar_t(1.), c.H00(), depth);
        a.H00().AGEMM(ta, tb, alpha, b.H01(), beta, c.H01(), depth);
        a.H01().AGEMM(ta, tb, alpha, b.H11(), scalar_t(1.), c.H01(), depth);
        a.H10().AGEMM(ta, tb, alpha, b.H00(), beta, c.H10(), depth);
        a.H11().AGEMM(ta, tb, alpha, b.H10(), scalar_t(1.), c.H10(), depth);
        a.H10().AGEMM(ta, tb, alpha, b.H01(), beta, c.H11(), depth);
        a.H11().AGEMM(ta, tb, alpha, b.H11(), scalar_t(1.), c.H11(), depth);
      } else {
        assert(false);
      }
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a, b, beta, c.D(), depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", Dense);" << std::endl;
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", Dense);" << std::endl;
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", Dense);" << std::endl;
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", Dense"
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", Dense"
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", Dense"
                << ", " << c.name() << ");" << std::endl;
    }


    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", Dense);" << std::endl;
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      using D_t = DenseMatrix<scalar_t>;
      using DW_t = DenseMatrixWrapper<scalar_t>;
#if defined(DEBUG_GEMM)
      auto cdense = c.dense();
#endif
      if (ta == Trans::N && tb == Trans::N) {
        D_t tmp(a.rank(), b.rank());
        gemm(Trans::C, Trans::N, scalar_t(1.), a.V(), b.U(),
             scalar_t(0.), tmp, depth);
        if (a.rank() < b.rank()) {
          D_t Vt(a.rank(), b.cols());
          gemm(Trans::N, Trans::C, scalar_t(1.), tmp, b.V(),
               scalar_t(0.), Vt, depth);
          auto m0 = c.H00().rows();
          auto m1 = c.H11().rows();
          auto n0 = c.H00().cols();
          auto n1 = c.H11().cols();
          DW_t U0(m0, a.rank(), *const_cast<D_t*>(&a.U()), 0, 0);
          DW_t U1(m1, a.rank(), *const_cast<D_t*>(&a.U()), m0, 0);
          DW_t Vt0(a.rank(), n0, Vt, 0, 0);
          DW_t Vt1(a.rank(), n1, Vt, 0, n0);
          c.H00().ABCGEMM(Trans::N, Trans::N, alpha, U0, Vt0, beta, depth);
          c.H01().ABCGEMM(Trans::N, Trans::N, alpha, U0, Vt1, beta, depth);
          c.H10().ABCGEMM(Trans::N, Trans::N, alpha, U1, Vt0, beta, depth);
          c.H11().ABCGEMM(Trans::N, Trans::N, alpha, U1, Vt1, beta, depth);
        } else {
          D_t U(a.rows(), b.rank());
          gemm(Trans::N, Trans::N, scalar_t(1.), a.U(), tmp,
               scalar_t(0.), U, depth);
          auto m0 = c.H00().rows();
          auto m1 = c.H11().rows();
          auto n0 = c.H00().cols();
          auto n1 = c.H11().cols();
          DW_t U0(m0, b.rank(), U, 0, 0);
          DW_t U1(m1, b.rank(), U, m0, 0);
          DW_t V0(n0, b.rank(), *const_cast<D_t*>(&b.V()), 0, 0);
          DW_t V1(n1, b.rank(), *const_cast<D_t*>(&b.V()), n0, 0);
          c.H00().ABCGEMM(Trans::N, Trans::C, alpha, U0, V0, beta, depth);
          c.H01().ABCGEMM(Trans::N, Trans::C, alpha, U0, V1, beta, depth);
          c.H10().ABCGEMM(Trans::N, Trans::C, alpha, U1, V0, beta, depth);
          c.H11().ABCGEMM(Trans::N, Trans::C, alpha, U1, V1, beta, depth);
        }
      } else {
        assert(false);
      }
#if defined(DEBUG_GEMM)
      gemm(ta, tb, alpha, a.dense(), b.dense(), beta, cdense, depth);
      auto cnorm = cdense.normF();
      cdense.scaled_add(scalar_t(-1.), c.dense());
      auto cdensenorm = cdense.normF();
      std::cout << "LR*LR+H:  ||Cdense-C||/||Cdense|| = "
                << cdensenorm / cnorm << std::endl;
      // using real_t = typename RealType<scalar_t>::value_type;
      // assert(cdensenorm / cnorm < blas::lamch<real_t>('P'));
#endif
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
#if defined(DEBUG_GEMM)
      auto cdense = c.dense();
#endif
      if (ta == Trans::N && tb == Trans::N) {
        DenseMatrix<scalar_t> tmp(a.rank(), b.rank());
        gemm(Trans::C, Trans::N, scalar_t(1.), a.V(), b.U(),
             scalar_t(0.), tmp, depth);
        if (a.rank() < b.rank()) {
          DenseMatrix<scalar_t> Vt(a.rank(), b.cols());
          gemm(Trans::N, Trans::C, scalar_t(1.), tmp, b.V(),
               scalar_t(0.), Vt, depth);
          gemm(Trans::N, Trans::N, alpha, a.U(), Vt, beta, c.D(), depth);
        } else {
          DenseMatrix<scalar_t> U(a.rows(), b.rank());
          gemm(Trans::N, Trans::N, scalar_t(1.), a.U(), tmp,
               scalar_t(0.), U, depth);
          gemm(Trans::N, Trans::C, alpha, U, b.V(), beta, c.D(), depth);
        }
      } else {
        assert(false);
      }
#if defined(DEBUG_GEMM)
      gemm(ta, tb, alpha, a.dense(), b.dense(), beta, cdense, depth);
      auto cnorm = cdense.normF();
      cdense.scaled_add(scalar_t(-1.), c.dense());
      auto cdensenorm = cdense.normF();
      std::cout << "LR*LR+D:  ||Cdense-C||/||Cdense|| = "
                << cdensenorm / cnorm << std::endl;
      // using real_t = typename RealType<scalar_t>::value_type;
      // assert(cdensenorm / cnorm < blas::lamch<real_t>('P'));
#endif
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a, b.D(), beta, c, depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a, b.D(), beta, c, depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a, b.D(), beta, c.D(), depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a, b.D(), beta, c, depth);
    }


    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", Dense"
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", Dense"
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(" << a.name() << ", Dense"
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixLR<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
#if defined(DEBUG_GEMM)
      auto cdense = c;
#endif
      DenseMatrix<scalar_t> tmp
        (a.rank(), (tb == Trans::N) ? b.cols() : b.rows());
      if (ta == Trans::N) {
        gemm(Trans::C, tb, scalar_t(1.), a.V(), b,
             scalar_t(0.), tmp, depth);
        gemm(Trans::N, Trans::N, alpha, a.U(), tmp,
             beta, c, depth);
      } else {
        gemm(Trans::C, tb, scalar_t(1.), a.U(), b,
             scalar_t(0.), tmp, depth);
        gemm(Trans::N, Trans::N, alpha, a.V(), tmp,
             beta, c, depth);
      }
#if defined(DEBUG_GEMM)
      gemm(ta, tb, alpha, a.dense(), b, beta, cdense, depth);
      auto cnorm = cdense.normF();
      cdense.scaled_add(scalar_t(-1.), c);
      auto cdensenorm = cdense.normF();
      std::cout << "LR*D+D:   ||Cdense-C||/||Cdense|| = "
                << cdensenorm / cnorm << std::endl;
      // using real_t = typename RealType<scalar_t>::value_type;
      // assert(cdensenorm / cnorm < blas::lamch<real_t>('P'));
#endif
    }


    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(, " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(, " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a, b, beta, c.D(), depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(Dense, " << b.name()
                << ", Dense);" << std::endl;
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(Dense, " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      std::cout << "TODO GEMM(Dense, " << b.name()
                << ", " << c.name() << ");" << std::endl;
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a, b, beta, c.D(), depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
#if defined(DEBUG_GEMM)
      auto cdense = c;
#endif
      DenseMatrix<scalar_t> tmp
        ((ta == Trans::N) ? a.rows() : a.cols(), b.rank());
      if (tb == Trans::N) {
        gemm(ta, Trans::N, scalar_t(1.), a, b.U(),
             scalar_t(0.), tmp, depth);
        gemm(Trans::N, Trans::C, alpha, tmp, b.V(),
             beta, c, depth);
      } else {
        gemm(ta, Trans::N, scalar_t(1.), a, b.V(),
             scalar_t(0.), tmp, depth);
        gemm(Trans::N, Trans::C, alpha, tmp, b.U(),
             beta, c, depth);
      }
#if defined(DEBUG_GEMM)
      gemm(ta, tb, alpha, a, b.dense(), beta, cdense, depth);
      auto cnorm = cdense.normF();
      cdense.scaled_add(scalar_t(-1.), c);
      auto cdensenorm = cdense.normF();
      std::cout << "D*LR+D:   ||Cdense-C||/||Cdense|| = "
                << cdensenorm / cnorm << std::endl;
      // using real_t = typename RealType<scalar_t>::value_type;
      // assert(cdensenorm / cnorm < blas::lamch<real_t>('P'));
#endif
   }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      using D_t = DenseMatrix<scalar_t>;
      using DW_t = DenseMatrixWrapper<scalar_t>;
#if defined(DEBUG_GEMM)
      auto cdense = c.dense();
#endif
      auto m0 = c.H00().rows();  auto n0 = c.H00().cols();
      auto m1 = c.H11().rows();  auto n1 = c.H11().cols();
      DW_t a0, a1, b0, b1;
      if (ta == Trans::N) {
        a0 = DW_t(m0, a.cols(), *const_cast<D_t*>(&a), 0, 0);
        a1 = DW_t(m1, a.cols(), *const_cast<D_t*>(&a), m0, 0);
      } else {
        a0 = DW_t(a.rows(), m0, *const_cast<D_t*>(&a), 0, 0);
        a1 = DW_t(a.rows(), m1, *const_cast<D_t*>(&a), 0, m0);
      }
      if (tb == Trans::N) {
        b0 = DW_t(b.rows(), n0, *const_cast<D_t*>(&b), 0, 0);
        b1 = DW_t(b.rows(), n1, *const_cast<D_t*>(&b), 0, n0);
      } else {
        b0 = DW_t(n0, b.cols(), *const_cast<D_t*>(&b), 0, 0);
        b1 = DW_t(n1, b.cols(), *const_cast<D_t*>(&b), n0, 0);
      }
      c.H00().ABCGEMM(ta, tb, alpha, a0, b0, beta, depth);
      c.H01().ABCGEMM(ta, tb, alpha, a0, b1, beta, depth);
      c.H10().ABCGEMM(ta, tb, alpha, a1, b0, beta, depth);
      c.H11().ABCGEMM(ta, tb, alpha, a1, b1, beta, depth);
#if defined(DEBUG_GEMM)
      gemm(ta, tb, alpha, a, b, beta, cdense, depth);
      auto cnorm = cdense.normF();
      cdense.scaled_add(scalar_t(-1.), c.dense());
      auto cdensenorm = cdense.normF();
      std::cout << "D*D+H:    ||Cdense-C||/||Cdense|| = "
                << cdensenorm / cnorm << std::endl;
      // using real_t = typename RealType<scalar_t>::value_type;
      // assert(cdensenorm / cnorm < blas::lamch<real_t>('P'));
#endif
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
#if defined(DEBUG_GEMM)
      auto cdense = c.dense();
#endif
      auto abc = c.dense();
      gemm(ta, tb, alpha, a, b, beta, abc, depth);
      auto opts = c.options();
      abc.low_rank(c.U(), c.V(), opts.rel_tol(), opts.abs_tol(),
                   opts.max_rank(), 0);
#if defined(DEBUG_GEMM)
      gemm(ta, tb, alpha, a, b, beta, cdense, depth);
      auto cnorm = cdense.normF();
      cdense.scaled_add(scalar_t(-1.), c.dense());
      auto cdensenorm = cdense.normF();
      std::cout << "D*D+LR:   ||Cdense-C||/||Cdense|| = "
                << cdensenorm / cnorm << std::endl;
      // using real_t = typename RealType<scalar_t>::value_type;
      // assert(cdensenorm / cnorm < blas::lamch<real_t>('P'));
#endif
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a, b, beta, c.D(), depth);
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a, b.D(), beta, c, depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a, b.D(), beta, c, depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a, b.D(), beta, c.D(), depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const DenseMatrix<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a, b.D(), beta, c, depth);
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b, beta, c, depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b, beta, c, depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b, beta, c.D(), depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const HMatrix<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b, beta, c, depth);
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b, beta, c, depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b, beta, c, depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b, beta, c.D(), depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const HMatrixLR<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b, beta, c, depth);
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b.D(), beta, c, depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b.D(), beta, c, depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b.D(), beta, c.D(), depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const HMatrixDense<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b.D(), beta, c, depth);
    }

    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, HMatrix<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b, beta, c, depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, HMatrixLR<scalar_t>& c, int depth=0) {
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, HMatrixDense<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b, beta, c.D(), depth);
    }
    template<typename scalar_t> void
    gemm(Trans ta, Trans tb, scalar_t alpha,
         const HMatrixDense<scalar_t>& a, const DenseMatrix<scalar_t>& b,
         scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
      gemm(ta, tb, alpha, a.D(), b, beta, c, depth);
    }





    // template<typename scalar_t> void
    // gemm(Trans ta, Trans tb, scalar_t alpha,
    //      const HMatrixDense<scalar_t>& a, const DenseMatrix<scalar_t>& b,
    //      scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
    //   gemm(ta, tb, alpha, a.D(), b, beta, c, depth);
    // }
    // template<typename scalar_t> void
    // gemm(Trans ta, Trans tb, scalar_t alpha,
    //      const HMatrixLR<scalar_t>& a, const DenseMatrix<scalar_t>& b,
    //      scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
    //   using DenseM_t = DenseMatrix<scalar_t>;
    //   // if real, then ta should be N or T, not C
    //   // if complex, then ta should be N or C, not T
    //   if (ta == Trans::N) {
    //     if (tb == Trans::N) {
    //       DenseM_t tmp(a.V().cols(), b.cols());
    //       gemm(Trans::C, Trans::N, scalar_t(1.), a.V(), b,
    //            scalar_t(0.), tmp, depth);
    //       gemm(Trans::N, Trans::N, scalar_t(1.), a.U(), tmp,
    //            scalar_t(0.), c, depth);
    //     } else {
    //       DenseM_t tmp(a.V().cols(), b.rows());
    //       gemm(Trans::C, tb, scalar_t(1.), a.V(), b,
    //            scalar_t(0.), tmp, depth);
    //       gemm(Trans::N, Trans::N, scalar_t(1.), a.U(), tmp,
    //            scalar_t(0.), c, depth);
    //     }
    //   } else {
    //     if (tb == Trans::N) {
    //       DenseM_t tmp(a.V().cols(), b.cols());
    //       gemm(Trans::C, Trans::N, scalar_t(1.), a.U(), b,
    //            scalar_t(0.), tmp, depth);
    //       gemm(Trans::N, Trans::N, scalar_t(1.), a.V(), tmp,
    //            scalar_t(0.), c, depth);
    //     } else {
    //       DenseM_t tmp(a.V().cols(), b.rows());
    //       gemm(Trans::C, tb, scalar_t(1.), a.U(), b,
    //            scalar_t(0.), tmp, depth);
    //       gemm(Trans::N, Trans::N, scalar_t(1.), a.V(), tmp,
    //            scalar_t(0.), c, depth);
    //     }
    //   }
    // }
    // template<typename scalar_t> void
    // gemm(Trans ta, Trans tb, scalar_t alpha,
    //      const HMatrix<scalar_t>& a, const DenseMatrix<scalar_t>& b,
    //      scalar_t beta, DenseMatrix<scalar_t>& c, int depth=0) {
    //   using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    //   if (ta == Trans::N) {
    //     if (tb == Trans::N) {
    //       DenseMW_t c0(a.H00().rows(), b.cols(), c, 0, 0);
    //       DenseMW_t c1(a.H10().rows(), b.cols(), c,
    //                   a.H00().rows(), 0);
    //       DenseMW_t b0(a.H00().cols(), b.cols(), b, 0, 0);
    //       DenseMW_t b1(a.H01().cols(), b.cols(), b,
    //                   a.H00().cols(), 0);
    //       gemm(Trans::N, Trans::N, alpha, a.H00(), b0, beta, c0);
    //       gemm(Trans::N, Trans::N, alpha, a.H01(), b1, scalar_t(1.), c0);
    //       gemm(Trans::N, Trans::N, alpha, a.H10(), b0, beta, c1);
    //       gemm(Trans::N, Trans::N, alpha, a.H11(), b1, scalar_t(1.), c1);
    //     } else {
    //       assert(true);
    //       // TODO
    //     }
    //   } else {
    //     assert(true);
    //     // TODO
    //   }
    // }

  } // end namespace H
} // end namespace strumpack

#endif // H_MATRIX_GEMM_HPP
