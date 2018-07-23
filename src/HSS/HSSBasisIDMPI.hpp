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
#ifndef HSS_BASIS_ID_MPI_HPP
#define HSS_BASIS_ID_MPI_HPP

#include <cassert>

#include "dense/DistributedMatrix.hpp"

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> class HSSBasisIDMPI {
      using DistM_t = DistributedMatrix<scalar_t>;
    private:
      std::vector<int> _P;
      DistM_t _E;

    public:
      inline std::size_t rows() const { return _E.cols()+_E.rows(); }
      inline std::size_t cols() const { return _E.cols(); }
      inline const DistM_t& E() const { return _E; }
      inline const std::vector<int>& P() const { return _P; }
      inline DistM_t& E() { return _E; }
      inline std::vector<int>& P() { return _P; }

      void clear();
      void print() const { print("basis"); }
      void print(std::string name) const;

      DistM_t dense() const;
      std::size_t memory() const {
        return E().memory() + sizeof(int)*P().size();
      }
      std::size_t nonzeros() const {
        return E().nonzeros() + P().size();
      }

      // TODO: remove these, the procs that are not active, will not
      // get the correct sized matrix back
      DistM_t apply(const DistM_t& b) const;
      DistM_t applyC(const DistM_t& b) const;
      void apply(const DistM_t& b, DistM_t& c) const;
      void applyC(const DistM_t& b, DistM_t& c) const;

      DistM_t extract_rows
      (const std::vector<std::size_t>& I) const;

      long long int apply_flops(std::size_t nrhs) const;
      long long int applyC_flops(std::size_t nrhs) const;
    };

    template<typename scalar_t> void HSSBasisIDMPI<scalar_t>::clear() {
      _P.clear();
      _E.clear();
    }

    template<typename scalar_t> void
    HSSBasisIDMPI<scalar_t>::print(std::string name) const {
      std::cout << name << " = { "
                << rows() << "x" << cols() << std::endl << "\tP = [";
      for (auto Pi : P()) std::cout << Pi << " ";
      std::cout << "]" << std::endl;
      _E.print("\tE");
      std::cout << "}" << std::endl;
    }

    template<typename scalar_t> DistributedMatrix<scalar_t>
    HSSBasisIDMPI<scalar_t>::dense() const {
      DistM_t ret(E().grid(), rows(), cols());
      ret.eye();
      if (E().grid())
        copy(rows()-cols(), cols(), E(), 0, 0, ret, cols(), 0, E().grid()->ctxt());
      ret.laswp(P(), false);
      return ret;
    }

    template<typename scalar_t> DistributedMatrix<scalar_t>
    HSSBasisIDMPI<scalar_t>::apply(const DistM_t& b) const {
      if (!b.active() || !rows() || !b.cols())
        return DistM_t(b.grid(), rows(), b.cols());
      DistM_t c(b.grid(), rows(), b.cols());
      copy(cols(), b.cols(), b, 0, 0, c, 0, 0, b.grid()->ctxt());
      DistributedMatrixWrapper<scalar_t>
        tmpC(E().rows(), b.cols(), c, cols(), 0);
      if (E().rows())
        gemm(Trans::N, Trans::N, scalar_t(1), E(), b, scalar_t(0.), tmpC);
      c.laswp(P(), false);
      return c;
    }

    template<typename scalar_t> void
    HSSBasisIDMPI<scalar_t>::apply(const DistM_t& b, DistM_t& c) const {
      if (!b.active() || !rows() || !b.cols()) return;
      copy(cols(), b.cols(), b, 0, 0, c, 0, 0, b.grid()->ctxt());
      DistributedMatrixWrapper<scalar_t>
        tmpC(E().rows(), b.cols(), c, cols(), 0);
      if (E().rows())
        gemm(Trans::N, Trans::N, scalar_t(1), E(), b, scalar_t(0.), tmpC);
      c.laswp(P(), false);
    }

    template<typename scalar_t> DistributedMatrix<scalar_t>
    HSSBasisIDMPI<scalar_t>::applyC(const DistM_t& b) const {
      if (!b.active() || !cols() || !b.cols())
        return DistM_t(b.grid(), E().cols(), b.cols());
      assert(b.rows() == int(rows()));
      DistM_t PtB(b);
      PtB.laswp(P(), true);
      if (!E().rows()) return PtB;
      DistM_t c(b.grid(), cols(), b.cols());
      copy(cols(), b.cols(), PtB, 0, 0, c, 0, 0, b.grid()->ctxt());
      auto tmpPtB = ConstDistributedMatrixWrapperPtr
        (E().rows(), b.cols(), PtB, cols(), 0);
      gemm(Trans::C, Trans::N, scalar_t(1.), E(), *tmpPtB, scalar_t(1.), c);
      return c;
    }

    template<typename scalar_t> void
    HSSBasisIDMPI<scalar_t>::applyC(const DistM_t& b, DistM_t& c) const {
      if (!b.active() || !cols() || !b.cols()) return;
      assert(b.cols() == c.cols());
      assert(b.rows() == int(rows()));
      assert(c.rows() == int(cols()));
      if (!E().rows()) {
        copy(b.rows(), b.cols(), b, 0, 0, c, 0, 0, b.grid()->ctxt());
        c.laswp(P(), true);
      } else {
        DistM_t PtB(b);
        PtB.laswp(P(), true);
        copy(cols(), b.cols(), PtB, 0, 0, c, 0, 0, b.grid()->ctxt());
        if (!E().rows()) return;
        auto tmpPtB = ConstDistributedMatrixWrapperPtr
          (E().rows(), b.cols(), PtB, cols(), 0);
        gemm(Trans::C, Trans::N, scalar_t(1.), E(), *tmpPtB, scalar_t(1.), c);
      }
    }

    template<typename scalar_t> DistributedMatrix<scalar_t>
    HSSBasisIDMPI<scalar_t>::extract_rows
    (const std::vector<std::size_t>& I) const {
      // TODO implement this without explicitly forming the dense basis matrix
      return dense().extract_rows(I);
    }

    template<typename scalar_t> long long int
    HSSBasisIDMPI<scalar_t>::apply_flops(std::size_t nrhs) const {
      if (!_E.is_master()) return 0;
      return blas::gemm_flops
        (_E.rows(), nrhs, _E.cols(), scalar_t(1.), scalar_t(0.));
    }

    template<typename scalar_t> long long int
    HSSBasisIDMPI<scalar_t>::applyC_flops(std::size_t nrhs) const {
      if (!_E.is_master()) return 0;
      return blas::gemm_flops
        (_E.cols(), nrhs, _E.rows(), scalar_t(1.), scalar_t(1.));
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_BASIS_ID_MPI_HPP
