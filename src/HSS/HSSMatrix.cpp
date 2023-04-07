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


#include "HSSMatrix.hpp"

#include "misc/TaskTimer.hpp"
#include "HSSMatrix.apply.hpp"
#include "HSSMatrix.compress.hpp"
#include "HSSMatrix.compress_stable.hpp"
#include "HSSMatrix.compress_kernel.hpp"
#include "HSSMatrix.factor.hpp"
#include "HSSMatrix.solve.hpp"
#include "HSSMatrix.extract.hpp"
#include "HSSMatrix.Schur.hpp"


namespace strumpack {
  namespace HSS {

    template<typename scalar_t> HSSMatrix<scalar_t>::HSSMatrix()
      : HSSMatrixBase<scalar_t>(0, 0, true) {}

    template<typename scalar_t> HSSMatrix<scalar_t>::HSSMatrix
    (const DenseMatrix<scalar_t>& A, const opts_t& opts)
      : HSSMatrix<scalar_t>(A.rows(), A.cols(), opts) {
      compress(A, opts);
    }

    template<typename scalar_t> HSSMatrix<scalar_t>::HSSMatrix
    (std::size_t m, std::size_t n, const opts_t& opts)
      : HSSMatrix<scalar_t>(m, n, opts, true) { }

    template<typename scalar_t> HSSMatrix<scalar_t>::HSSMatrix
    (std::size_t m, std::size_t n, const opts_t& opts, bool active)
      : HSSMatrixBase<scalar_t>(m, n, active) {
      if (!active) return;
      if (m > std::size_t(opts.leaf_size()) ||
          n > std::size_t(opts.leaf_size())) {
        this->ch_.reserve(2);
        this->ch_.emplace_back(new HSSMatrix<scalar_t>(m/2, n/2, opts));
        this->ch_.emplace_back(new HSSMatrix<scalar_t>(m-m/2, n-n/2, opts));
      }
    }

    template<typename scalar_t> HSSMatrix<scalar_t>::HSSMatrix
    (const structured::ClusterTree& t, const opts_t& opts, bool active)
      : HSSMatrixBase<scalar_t>(t.size, t.size, active) {
      if (!active) return;
      if (!t.c.empty()) {
        assert(t.c.size() == 2);
        this->ch_.reserve(2);
        this->ch_.emplace_back(new HSSMatrix<scalar_t>(t.c[0], opts));
        this->ch_.emplace_back(new HSSMatrix<scalar_t>(t.c[1], opts));
      }
    }

    template<typename scalar_t> HSSMatrix<scalar_t>::HSSMatrix
    (const structured::ClusterTree& t, const opts_t& opts)
      : HSSMatrix<scalar_t>(t, opts, true) { }

    template<typename scalar_t> HSSMatrix<scalar_t>::HSSMatrix
    (kernel::Kernel<real_t>& K, const opts_t& opts)
      : HSSMatrixBase<scalar_t>(K.n(), K.n(), true) {
      TaskTimer timer("clustering");
      timer.start();
      auto t = binary_tree_clustering
        (opts.clustering_algorithm(), K.data(), K.permutation(), opts.leaf_size());
      K.permute();
      if (opts.verbose())
        std::cout << "# clustering (" << get_name(opts.clustering_algorithm())
                  << ") time = " << timer.elapsed() << std::endl;
      if (!t.c.empty()) {
        assert(t.c.size() == 2);
        this->ch_.reserve(2);
        this->ch_.emplace_back(new HSSMatrix<scalar_t>(t.c[0], opts));
        this->ch_.emplace_back(new HSSMatrix<scalar_t>(t.c[1], opts));
      }
      compress(K, opts);
    }

    template<typename scalar_t>
    HSSMatrix<scalar_t>::HSSMatrix(std::ifstream& is)
      : HSSMatrixBase<scalar_t>(0, 0, true) {
      read(is);
    }

    template<typename scalar_t>
    HSSMatrix<scalar_t>::HSSMatrix(const HSSMatrix<scalar_t>& other)
      : HSSMatrixBase<scalar_t>(other) {
      U_ = other.U_;
      V_ = other.V_;
      D_ = other.D_;
      B01_ = other.B01_;
      B10_ = other.B10_;
    }

    template<typename scalar_t> HSSMatrix<scalar_t>&
    HSSMatrix<scalar_t>::operator=(const HSSMatrix<scalar_t>& other) {
      HSSMatrixBase<scalar_t>::operator=(other);
      U_ = other.U_;
      V_ = other.V_;
      D_ = other.D_;
      B01_ = other.B01_;
      B10_ = other.B10_;
      return *this;
    }

    template<typename scalar_t> std::unique_ptr<HSSMatrixBase<scalar_t>>
    HSSMatrix<scalar_t>::clone() const {
      return std::unique_ptr<HSSMatrixBase<scalar_t>>
        (new HSSMatrix<scalar_t>(*this));
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::delete_trailing_block() {
      B01_.clear();
      B10_.clear();
      HSSMatrixBase<scalar_t>::delete_trailing_block();
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress(const DenseM_t& A, const opts_t& opts) {
      TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_compress);
      switch (opts.compression_algorithm()) {
      case CompressionAlgorithm::ORIGINAL:
        compress_original(A, opts); break;
      case CompressionAlgorithm::STABLE:
        compress_stable(A, opts); break;
      case CompressionAlgorithm::HARD_RESTART:
        compress_hard_restart(A, opts); break;
      default:
        std::cout << "Compression algorithm not recognized!" << std::endl;
      };
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::compress
    (const mult_t& Amult, const elem_t& Aelem, const opts_t& opts) {
      TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_compress);
      switch (opts.compression_algorithm()) {
      case CompressionAlgorithm::ORIGINAL:
        compress_original(Amult, Aelem, opts); break;
      case CompressionAlgorithm::STABLE:
        compress_stable(Amult, Aelem, opts); break;
      case CompressionAlgorithm::HARD_RESTART:
        compress_hard_restart(Amult, Aelem, opts); break;
      default:
        std::cout << "Compression algorithm not recognized!" << std::endl;
      };
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::reset() {
      U_.clear();
      V_.clear();
      D_.clear();
      B01_.clear();
      B10_.clear();
      HSSMatrixBase<scalar_t>::reset();
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    HSSMatrix<scalar_t>::dense() const {
      DenseM_t A(this->rows(), this->cols());
      WorkDense<scalar_t> w;
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      dense_recursive(A, w, true, this->openmp_task_depth_);
      return A;
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::dense_recursive
    (DenseM_t& A, WorkDense<scalar_t>& w, bool isroot, int depth) const {
      if (this->leaf()) {
        copy(D_, A, w.offset.first, w.offset.second);
        w.tmpU = U_.dense();
        w.tmpV = V_.dense();
      } else {
        w.c.resize(2);
        w.c[0].offset = w.offset;
        w.c[1].offset = w.offset + child(0)->dims();
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        child(0)->dense_recursive(A, w.c[0], false, depth+1);
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        child(1)->dense_recursive(A, w.c[1], false, depth+1);
#pragma omp taskwait

#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          DenseM_t tmp01(B01_.rows(), w.c[1].tmpV.rows());
          DenseMW_t A01(child(0)->rows(), child(1)->cols(),
                        A, w.c[0].offset.first, w.c[1].offset.second);
          gemm(Trans::N, Trans::C, scalar_t(1.), B01_, w.c[1].tmpV,
               scalar_t(0.), tmp01, depth);
          gemm(Trans::N, Trans::N, scalar_t(1.), w.c[0].tmpU, tmp01,
               scalar_t(0.), A01, depth);
        }
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          DenseM_t tmp10(B10_.rows(), w.c[0].tmpV.rows());
          DenseMW_t A10(child(1)->rows(), child(0)->cols(), A,
                        w.c[1].offset.first, w.c[0].offset.second);
          gemm(Trans::N, Trans::C, scalar_t(1.), B10_, w.c[0].tmpV,
               scalar_t(0.), tmp10, depth);
          gemm(Trans::N, Trans::N, scalar_t(1.), w.c[1].tmpU, tmp10,
               scalar_t(0.), A10, depth);
        }
        if (!isroot) {
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          {
            w.tmpU = DenseM_t(this->rows(), this->U_rank());
            DenseMW_t wtmpU0(child(0)->rows(), this->U_rank(),
                             w.tmpU, 0, 0);
            DenseMW_t wtmpU1(child(1)->rows(), this->U_rank(), w.tmpU,
                             child(0)->rows(), 0);
            auto Udense = U_.dense();
            DenseMW_t Udense0(child(0)->U_rank(), Udense.cols(),
                              Udense, 0, 0);
            DenseMW_t Udense1(child(1)->U_rank(), Udense.cols(), Udense,
                              child(0)->U_rank(), 0);
            gemm(Trans::N, Trans::N, scalar_t(1.), w.c[0].tmpU, Udense0,
                 scalar_t(0.), wtmpU0, depth);
            gemm(Trans::N, Trans::N, scalar_t(1.), w.c[1].tmpU, Udense1,
                 scalar_t(0.), wtmpU1, depth);
          }
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          {
            w.tmpV = DenseM_t(this->cols(), this->V_rank());
            DenseMW_t wtmpV0(child(0)->cols(), this->V_rank(),
                             w.tmpV, 0, 0);
            DenseMW_t wtmpV1(child(1)->cols(), this->V_rank(),
                             w.tmpV, child(0)->cols(), 0);
            auto Vdense = V_.dense();
            DenseMW_t Vdense0(child(0)->V_rank(), Vdense.cols(),
                              Vdense, 0, 0);
            DenseMW_t Vdense1(child(1)->V_rank(), Vdense.cols(),
                              Vdense, child(0)->V_rank(), 0);
            gemm(Trans::N, Trans::N, scalar_t(1.), w.c[0].tmpV, Vdense0,
                 scalar_t(0.), wtmpV0, depth);
            gemm(Trans::N, Trans::N, scalar_t(1.), w.c[1].tmpV, Vdense1,
                 scalar_t(0.), wtmpV1, depth);
          }
        }
#pragma omp taskwait
        w.c[0].tmpU.clear();
        w.c[0].tmpV.clear();
        w.c[1].tmpU.clear();
        w.c[1].tmpV.clear();
      }
    }

    template<typename scalar_t> std::size_t
    HSSMatrix<scalar_t>::rank() const {
      if (!this->active()) return 0;
      std::size_t rank = std::max(this->U_rank(), this->V_rank());
      for (auto& c : this->ch_) rank = std::max(rank, c->rank());
      return rank;
    }

    template<typename scalar_t> std::size_t
    HSSMatrix<scalar_t>::memory() const {
      if (!this->active()) return 0;
      std::size_t mem = sizeof(*this) + U_.memory() + V_.memory()
        + D_.memory() + B01_.memory() + B10_.memory();
      for (auto& c : this->ch_) mem += c->memory();
      return mem;
    }

    template<typename scalar_t> std::size_t
    HSSMatrix<scalar_t>::nonzeros() const {
      if (!this->active()) return 0;
      std::size_t nnz = sizeof(*this) + U_.nonzeros() + V_.nonzeros()
        + D_.nonzeros() + B01_.nonzeros() + B10_.nonzeros();
      for (auto& c : this->ch_) nnz += c->nonzeros();
      return nnz;
    }

    template<typename scalar_t> std::size_t
    HSSMatrix<scalar_t>::levels() const {
      if (!this->active()) return 0;
      std::size_t lvls = 0;
      for (auto& c : this->ch_) lvls = std::max(lvls, c->levels());
      return 1 + lvls;
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::print_info
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
          << this->U_rows() << " x " << this->U_rank() << " V = "
          << this->V_rows() << " x " << this->V_rank();
      if (this->leaf()) out << " leaf" << std::endl;
      else out << " non-leaf" << std::endl;
      for (auto& c : this->ch_) {
        c->print_info(out, roff, coff);
        roff += c->rows();
        coff += c->cols();
      }
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::shift(scalar_t sigma) {
      if (!this->active()) return;
      if (this->leaf()) D_.shift(sigma);
      else
        for (auto& c : this->ch_)
          c->shift(sigma);
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::draw
    (std::ostream& of, std::size_t rlo, std::size_t clo) const {
      if (!this->leaf()) {
        char prev = std::cout.fill('0');
        int rank0 = std::max(B01_.rows(), B01_.cols());
        int rank1 = std::max(B10_.rows(), B10_.cols());
        int minmn = std::min(this->rows(), this->cols());
        int red = std::floor(255.0 * rank0 / minmn);
        int blue = 255 - red;
        of << "set obj rect from "
           << rlo << ", " << clo+child(0)->cols() << " to "
           << rlo+child(0)->rows()
           << ", " << clo+this->cols()
           << " fc rgb '#"
           << std::hex << std::setw(2) << std::setfill('0') << red
           << "00" << std::setw(2)  << std::setfill('0') << blue
           << "'" << std::dec << std::endl;
        red = std::floor(255.0 * rank1 / minmn);
        blue = 255 - red;
        of << "set obj rect from "
           << rlo+child(0)->rows() << ", " << clo
           << " to " << rlo+this->rows()
           << ", " << clo+child(0)->cols()
           << " fc rgb '#"
           << std::hex << std::setw(2) << std::setfill('0') << red
           << "00" << std::setw(2)  << std::setfill('0') << blue
           << "'" << std::dec << std::endl;
        std::cout.fill(prev);
        child(0)->draw(of, rlo, clo);
        child(1)->draw
          (of, rlo+child(0)->rows(), clo+child(0)->cols());
      } else {
        of << "set obj rect from "
           << rlo << ", " << clo << " to "
           << rlo+this->rows() << ", " << clo+this->cols()
           << " fc rgb 'red'" << std::endl;
      }
    }


    template<typename scalar_t>
    void draw(const HSSMatrix<scalar_t>& H, const std::string& name) {
      std::ofstream of("plot" + name + ".gnuplot");
      of << "set terminal pdf enhanced color size 5,4" << std::endl;
      of << "set output '" << name << ".pdf'" << std::endl;
      H.draw(of);
      of << "set xrange [0:" << H.cols() << "]" << std::endl;
      of << "set yrange [" << H.rows() << ":0]" << std::endl;
      of << "plot x lt -1 notitle" << std::endl;
      of.close();
    }

    template<typename scalar_t> void apply_HSS
    (Trans op, const HSSMatrix<scalar_t>& A, const DenseMatrix<scalar_t>& B,
     scalar_t beta, DenseMatrix<scalar_t>& C) {
      WorkApply<scalar_t> w;
      std::atomic<long long int> flops(0);
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      {
        if (op == Trans::N) {
          A.apply_fwd(B, w, true, A.openmp_task_depth_, flops);
          A.apply_bwd(B, beta, C, w, true, A.openmp_task_depth_, flops);
        } else {
          A.applyT_fwd(B, w, true, A.openmp_task_depth_, flops);
          A.applyT_bwd(B, beta, C, w, true, A.openmp_task_depth_, flops);
        }
      }
    }


    template<typename scalar_t> void
    HSSMatrix<scalar_t>::write(std::ofstream& os) const {
      os.write((const char*)&this->rows_, sizeof(this->rows_));
      os.write((const char*)&this->cols_, sizeof(this->cols_));
      os.write((const char*)&this->U_state_, sizeof(this->U_state_));
      os.write((const char*)&this->V_state_, sizeof(this->V_state_));
      os.write((const char*)&this->openmp_task_depth_, sizeof(this->openmp_task_depth_));
      os.write((const char*)&this->active_, sizeof(this->active_));
      os.write((const char*)&this->U_rank_, sizeof(this->U_rank_));
      os.write((const char*)&this->U_rows_, sizeof(this->U_rows_));
      os.write((const char*)&this->V_rank_, sizeof(this->V_rank_));
      os.write((const char*)&this->V_rows_, sizeof(this->V_rows_));
      os << this->Asub_;
      os << U_ << V_ << D_ << B01_ << B10_;
      int nc = this->ch_.size();
      os.write((const char*)&nc, sizeof(nc));
      for (auto& c : this->ch_)
        c->write(os);
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::read(std::ifstream& is) {
      is.read((char*)&this->rows_, sizeof(this->rows_));
      is.read((char*)&this->cols_, sizeof(this->cols_));
      is.read((char*)&this->U_state_, sizeof(this->U_state_));
      is.read((char*)&this->V_state_, sizeof(this->V_state_));
      is.read((char*)&this->openmp_task_depth_, sizeof(this->openmp_task_depth_));
      is.read((char*)&this->active_, sizeof(this->active_));
      is.read((char*)&this->U_rank_, sizeof(this->U_rank_));
      is.read((char*)&this->U_rows_, sizeof(this->U_rows_));
      is.read((char*)&this->V_rank_, sizeof(this->V_rank_));
      is.read((char*)&this->V_rows_, sizeof(this->V_rows_));
      is >> this->Asub_;
      is >> U_ >> V_ >> D_ >> B01_ >> B10_;
      int nc = 0;
      is.read((char*)&nc, sizeof(nc));
      this->ch_.resize(nc);
      for (auto& c : this->ch_)
        c.reset(new HSSMatrix<scalar_t>(is));
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::write(const std::string& fname) const {
      std::ofstream f(fname, std::ios::out | std::ios::trunc);
      int v[3];
      get_version(v, v+1, v+2);
      f.write((const char*)v, sizeof(v));
      write(f);
    }

    template<typename scalar_t> HSSMatrix<scalar_t>
    HSSMatrix<scalar_t>::read(const std::string& fname) {
      std::ifstream f;
      try {
        f.open(fname);
      } catch (std::ios_base::failure& e) {
        std::cerr << e.what() << std::endl;
      }
      int v[3], vf[3];
      get_version(v+0, v+1, v+2);
      f.read((char*)vf, sizeof(vf));
      if (v[0] != vf[0] || v[1] != vf[1] || v[2] != vf[2]) {
        std::cerr << "Warning, file was created with a different"
                  << " strumpack version (v"
                  << vf[0] << "." << vf[1] << "." << vf[2]
                  << " instead of v"
                  << v[0] << "." << v[1] << "." << v[2]
                  << ")" << std::endl;
      }
      HSSMatrix<scalar_t> H;
      H.read(f);
      return H;
    }

    // explicit template instantiations
    template class HSSMatrix<float>;
    template class HSSMatrix<double>;
    template class HSSMatrix<std::complex<float>>;
    template class HSSMatrix<std::complex<double>>;

    template void
    apply_HSS(Trans op, const HSSMatrix<float>& A,
              const DenseMatrix<float>& B,
              float beta, DenseMatrix<float>& C);
    template void
    apply_HSS(Trans op, const HSSMatrix<double>& A,
              const DenseMatrix<double>& B,
              double beta, DenseMatrix<double>& C);
    template void
    apply_HSS(Trans op, const HSSMatrix<std::complex<float>>& A,
              const DenseMatrix<std::complex<float>>& B,
              std::complex<float> beta,
              DenseMatrix<std::complex<float>>& C);
    template void
    apply_HSS(Trans op, const HSSMatrix<std::complex<double>>& A,
              const DenseMatrix<std::complex<double>>& B,
              std::complex<double> beta,
              DenseMatrix<std::complex<double>>& C);

    template void draw(const HSSMatrix<float>& H, const std::string& name);
    template void draw(const HSSMatrix<double>& H, const std::string& name);
    template void draw(const HSSMatrix<std::complex<float>>& H,
                       const std::string& name);
    template void draw(const HSSMatrix<std::complex<double>>& H,
                       const std::string& name);

  } // end namespace HSS
} // end namespace strumpack

