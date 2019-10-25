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
#ifndef HSS_DISTELEMMULT_HPP
#define HSS_DISTELEMMULT_HPP

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> class DistElemMult {
      using DistM_t = DistributedMatrix<scalar_t>;
      using DenseM_t = DenseMatrix<scalar_t>;
    public:
      DistElemMult(const DistM_t& A) : _A(A) {}
      const DistM_t& _A;
      void operator()(DistM_t& R, DistM_t& Sr, DistM_t& Sc) {
        gemm(Trans::N, Trans::N, scalar_t(1.), _A, R, scalar_t(0.), Sr);
        gemm(Trans::C, Trans::N, scalar_t(1.), _A, R, scalar_t(0.), Sc);
      }
      void operator()
      (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
       DistM_t& B, DistM_t& A, std::size_t rlo, std::size_t clo, MPI_Comm c) {
        if (!B.active()) return;
        assert(int(I.size()) == B.rows() && int(J.size()) == B.cols());
        std::vector<std::size_t> lI(I);
        std::vector<std::size_t> lJ(J);
        for (auto& i : lI) i -= rlo;
        for (auto& j : lJ) j -= clo;
        auto Asub = A.extract(lI, lJ);
        assert(Asub.ctxt() == B.ctxt());
        B = Asub;
        // TODO: if I do
        // B = A.extract(lI, lJ);
        // it compiles, but doesn't work correctly.
        // This should not be able to compile, since B
        // is a DistMWrapper, and the move constructor is
        // deleted explicitly???
      }
    };

    template<typename scalar_t> class DistElemMultDuplicated {
      using DistM_t = DistributedMatrix<scalar_t>;
      using DenseM_t = DenseMatrix<scalar_t>;
    public:
      DistElemMultDuplicated(const DistM_t& A)
        : A_(A), Ag_(A.all_gather()) {}

      void operator()(DistM_t& R, DistM_t& Sr, DistM_t& Sc) {
        gemm(Trans::N, Trans::N, scalar_t(1.), A_, R, scalar_t(0.), Sr);
        gemm(Trans::C, Trans::N, scalar_t(1.), A_, R, scalar_t(0.), Sc);
      }
      void operator()
      (const std::vector<size_t>& I, const std::vector<size_t>& J,
       DistM_t& B) {
        if (!B.active()) return;
        assert(int(I.size()) == B.rows() && int(J.size()) == B.cols());
        for (std::size_t j=0; j<J.size(); j++)
          for (std::size_t i=0; i<I.size(); i++) {
            assert(I[i] >= 0 && int(I[i]) < A_.rows() &&
                   J[j] >= 0 && int(J[j]) < A_.cols());
            B.global(i, j, Ag_(I[i], J[j]));
          }
      }

    private:
      const DistM_t& A_;
      const DenseM_t Ag_;
    };

    template<typename scalar_t> class LocalElemMult {
      using DistM_t = DistributedMatrix<scalar_t>;
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      using DenseM_t = DenseMatrix<scalar_t>;
      using delemw_t = typename std::function
        <void(const std::vector<std::size_t>& I,
              const std::vector<std::size_t>& J,
              DistM_t& B, DistM_t& A,
              std::size_t rlo, std::size_t clo, MPI_Comm comm)>;
    public:
      LocalElemMult(const delemw_t& Aelem,
                    std::pair<std::size_t,std::size_t>& offset,
                    const BLACSGrid* lg, const DenseM_t& A)
        : dAelem_(Aelem), offset_(offset), grid_local_(lg), A_(A) {}

      void operator()
      (const std::vector<size_t>& I, const std::vector<size_t>& J,
       DenseM_t& B) {
        if (A_.rows() != 0)
          B = A_.extract(I, J);
        else {
          std::vector<std::size_t> gI(I), gJ(J);
          for (auto& i : gI) i += offset_.first;
          for (auto& j : gJ) j += offset_.second;
          DistMW_t dB(grid_local_, I.size(), J.size(), B);
          DistM_t dA;
          dAelem_(gI, gJ, dB, dA, 0, 0, MPI_COMM_NULL);
        }
      }

    private:
      const delemw_t& dAelem_;
      const std::pair<std::size_t,std::size_t> offset_;
      const BLACSGrid* grid_local_;
      const DenseM_t& A_;
    };

  } // end namespace HSS
} // end namespace strumpack

#endif  // HSS_DISTELEMMULT_HPP
