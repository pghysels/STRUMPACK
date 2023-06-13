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

#include <iomanip>

#include "HBSMatrix.hpp"

#include "misc/TaskTimer.hpp"
// #include "HBSMatrix.apply.hpp"
#include "HBSMatrix.compress.hpp"
// #include "HBSMatrix.compress_stable.hpp"
// #include "HBSMatrix.compress_kernel.hpp"
// #include "HBSMatrix.factor.hpp"
// #include "HBSMatrix.solve.hpp"
// #include "HBSMatrix.extract.hpp"
// #include "HBSMatrix.Schur.hpp"


namespace strumpack {
  namespace HBS {

    template<typename scalar_t> HBSMatrix<scalar_t>::HBSMatrix
    (std::size_t m, std::size_t n, bool active)
      : rows_(m), cols_(n), U_state_(State::UNTOUCHED),
        V_state_(State::UNTOUCHED),
        openmp_task_depth_(0), active_(active) {
    }

    template<typename scalar_t> HBSMatrix<scalar_t>::HBSMatrix()
      : HBSMatrix<scalar_t>(0, 0, true) {}

    template<typename scalar_t> HBSMatrix<scalar_t>::HBSMatrix
    (const DenseMatrix<scalar_t>& A, const opts_t& opts)
      : HBSMatrix<scalar_t>(A.rows(), A.cols(), opts) {
      compress(A, opts);
    }

    template<typename scalar_t> HBSMatrix<scalar_t>::HBSMatrix
    (std::size_t m, std::size_t n, const opts_t& opts)
      : HBSMatrix<scalar_t>(m, n, opts, true) { }

    template<typename scalar_t> HBSMatrix<scalar_t>::HBSMatrix
    (std::size_t m, std::size_t n, const opts_t& opts, bool active)
      : HBSMatrix<scalar_t>(m, n, active) {
      if (!active) return;
      if (m > std::size_t(opts.leaf_size()) ||
          n > std::size_t(opts.leaf_size())) {
        this->ch_.reserve(2);
        this->ch_.emplace_back(new HBSMatrix<scalar_t>(m/2, n/2, opts));
        this->ch_.emplace_back(new HBSMatrix<scalar_t>(m-m/2, n-n/2, opts));
      }
    }

    template<typename scalar_t> HBSMatrix<scalar_t>::HBSMatrix
    (const structured::ClusterTree& t, const opts_t& opts, bool active)
      : HBSMatrix<scalar_t>(t.size, t.size, active) {
      if (!active) return;
      if (!t.c.empty()) {
        assert(t.c.size() == 2);
        this->ch_.reserve(2);
        this->ch_.emplace_back(new HBSMatrix<scalar_t>(t.c[0], opts));
        this->ch_.emplace_back(new HBSMatrix<scalar_t>(t.c[1], opts));
      }
    }

    template<typename scalar_t> HBSMatrix<scalar_t>::HBSMatrix
    (const structured::ClusterTree& t, const opts_t& opts)
      : HBSMatrix<scalar_t>(t, opts, true) { }


    template<typename scalar_t>
    HBSMatrix<scalar_t>::HBSMatrix(const HBSMatrix<scalar_t>& other) {
      U_ = other.U_;
      V_ = other.V_;
      D_ = other.D_;
      // B01_ = other.B01_;
      // B10_ = other.B10_;
    }

    template<typename scalar_t> HBSMatrix<scalar_t>&
    HBSMatrix<scalar_t>::operator=(const HBSMatrix<scalar_t>& other) {
      U_ = other.U_;
      V_ = other.V_;
      D_ = other.D_;
      // B01_ = other.B01_;
      // B10_ = other.B10_;
      return *this;
    }

    // template<typename scalar_t> void
    // HBSMatrix<scalar_t>::compress(const DenseM_t& A, const opts_t& opts) {
    //   // TIMER_TIME(TaskType::HBS_COMPRESS, 0, t_compress);
    //   // switch (opts.compression_algorithm()) {
    //   // case CompressionAlgorithm::ORIGINAL:
    //   //   compress_original(A, opts); break;
    //   // case CompressionAlgorithm::STABLE:
    //   //   compress_stable(A, opts); break;
    //   // case CompressionAlgorithm::HARD_RESTART:
    //   //   compress_hard_restart(A, opts); break;
    //   // default:
    //   //   std::cout << "Compression algorithm not recognized!" << std::endl;
    //   // };
    // }

    template<typename scalar_t> void HBSMatrix<scalar_t>::reset() {
      U_.clear();
      V_.clear();
      D_.clear();
      // B01_.clear();
      // B10_.clear();
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    HBSMatrix<scalar_t>::dense() const {
      DenseM_t A(this->rows(), this->cols());
//       WorkDense<scalar_t> w;
// #pragma omp parallel if(!omp_in_parallel())
// #pragma omp single nowait
//       dense_recursive(A, w, true, this->openmp_task_depth_);
      return A;
    }

    template<typename scalar_t> std::size_t
    HBSMatrix<scalar_t>::rank() const {
      if (!this->active()) return 0;
      std::size_t rank = std::max(this->U_.cols(), this->V_.cols());
      for (auto& c : this->ch_) rank = std::max(rank, c->rank());
      return rank;
    }

    template<typename scalar_t> std::size_t
    HBSMatrix<scalar_t>::memory() const {
      if (!this->active()) return 0;
      // std::size_t mem = sizeof(*this) + U_.memory() + V_.memory()
      //   + D_.memory() + B01_.memory() + B10_.memory();
      // for (auto& c : this->ch_) mem += c->memory();
      // return mem;
      return 0;
    }

    template<typename scalar_t> std::size_t
    HBSMatrix<scalar_t>::nonzeros() const {
      if (!this->active()) return 0;
      // std::size_t nnz = sizeof(*this) + U_.nonzeros() + V_.nonzeros()
      //   + D_.nonzeros() + B01_.nonzeros() + B10_.nonzeros();
      // for (auto& c : this->ch_) nnz += c->nonzeros();
      // return nnz;
      return 0;
    }

    template<typename scalar_t> std::size_t
    HBSMatrix<scalar_t>::levels() const {
      if (!this->active()) return 0;
      std::size_t lvls = 0;
      for (auto& c : this->ch_) lvls = std::max(lvls, c->levels());
      return 1 + lvls;
    }

    template<typename scalar_t> void HBSMatrix<scalar_t>::print_info
    (std::ostream &out, std::size_t roff, std::size_t coff) const {
      if (!this->active()) return;
#if defined(STRUMPACK_USE_MPI)
      int flag, rank;
      MPI_Initialized(&flag);
      if (flag) rank = mpi_rank();
      else rank = 0;
#else
      int rank = 0;
#endif
      out << "SEQ rank=" << rank
          << " b = [" << roff << "," << roff+this->rows()
          << " x " << coff << "," << coff+this->cols() << "]  U = "
          << this->U_.rows() << " x " << this->U_.cols() << " V = "
          << this->V_.rows() << " x " << this->V_.cols();
      if (this->leaf()) out << " leaf" << std::endl;
      else out << " non-leaf" << std::endl;
      for (auto& c : this->ch_) {
        c->print_info(out, roff, coff);
        roff += c->rows();
        coff += c->cols();
      }
    }

    template<typename scalar_t> void
    HBSMatrix<scalar_t>::shift(scalar_t sigma) {
      if (!this->active()) return;
      if (this->leaf()) D_.shift(sigma);
      else
        for (auto& c : this->ch_)
          c->shift(sigma);
    }


    template<typename scalar_t> void apply_HBS
    (Trans op, const HBSMatrix<scalar_t>& A, const DenseMatrix<scalar_t>& B,
     scalar_t beta, DenseMatrix<scalar_t>& C) {
//       WorkApply<scalar_t> w;
//       std::atomic<long long int> flops(0);
// #pragma omp parallel if(!omp_in_parallel())
// #pragma omp single nowait
//       {
//         if (op == Trans::N) {
//           A.apply_fwd(B, w, true, A.openmp_task_depth_, flops);
//           A.apply_bwd(B, beta, C, w, true, A.openmp_task_depth_, flops);
//         } else {
//           A.applyT_fwd(B, w, true, A.openmp_task_depth_, flops);
//           A.applyT_bwd(B, beta, C, w, true, A.openmp_task_depth_, flops);
//         }
//       }
    }


    // explicit template instantiations
    template class HBSMatrix<float>;
    template class HBSMatrix<double>;
    template class HBSMatrix<std::complex<float>>;
    template class HBSMatrix<std::complex<double>>;

    template void
    apply_HBS(Trans op, const HBSMatrix<float>& A,
              const DenseMatrix<float>& B,
              float beta, DenseMatrix<float>& C);
    template void
    apply_HBS(Trans op, const HBSMatrix<double>& A,
              const DenseMatrix<double>& B,
              double beta, DenseMatrix<double>& C);
    template void
    apply_HBS(Trans op, const HBSMatrix<std::complex<float>>& A,
              const DenseMatrix<std::complex<float>>& B,
              std::complex<float> beta,
              DenseMatrix<std::complex<float>>& C);
    template void
    apply_HBS(Trans op, const HBSMatrix<std::complex<double>>& A,
              const DenseMatrix<std::complex<double>>& B,
              std::complex<double> beta,
              DenseMatrix<std::complex<double>>& C);

  } // end namespace HBS
} // end namespace strumpack

