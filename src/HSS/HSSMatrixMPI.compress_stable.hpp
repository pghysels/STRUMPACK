#ifndef HSS_MATRIX_MPI_COMPRESS_STABLE_HPP
#define HSS_MATRIX_MPI_COMPRESS_STABLE_HPP

#include "misc/RandomWrapper.hpp"
#include "DistSamples.hpp"
#include "DistElemMult.hpp"

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_stable_nosync
    (const dmult_t& Amult, const delem_t& Aelem,
     const opts_t& opts, int Actxt) {
      auto d = opts.d0();
      auto dd = opts.dd();
      DistSamples<scalar_t> RS(d+dd, (Actxt!=-1) ? Actxt :
                               _ctxt, *this, Amult, opts);
      WorkCompressMPI<scalar_t> w;
      while (!this->is_compressed()) {
        if (d != opts.d0()) RS.add_columns(d+dd, opts);
        if (opts.verbose() && !mpi_rank(_comm))
          std::cout << "# compressing with d+dd = " << d << "+" << dd
                    << " (stable)" << std::endl;
        compress_recursive_stable(RS, Aelem, opts, w, d, dd);
        d += dd;
        dd = std::min(dd, opts.max_rank()-d);
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_stable_sync
    (const dmult_t& Amult, const delem_t& Aelem,
     const opts_t& opts, int Actxt) {
      auto d = opts.d0();
      auto dd = opts.dd();
      assert(dd <= d);
      std::unique_ptr<random::RandomGeneratorBase<real_t>> rgen;
      WorkCompressMPI<scalar_t> w;
      DistSamples<scalar_t> RS(d+dd, (Actxt!=-1) ? Actxt :
                               _ctxt, *this, Amult, opts);
      const auto nr_lvls = this->max_levels();
      while (!this->is_compressed()) {
        if (d != opts.d0()) RS.add_columns(d+dd, opts);
        if (opts.verbose() && !mpi_rank(_comm))
          std::cout << "# compressing with d+dd = " << d << "+" << dd
                    << " (stable)" << std::endl;
        for (int lvl=nr_lvls-1; lvl>=0; lvl--) {
          extract_level(Aelem, opts, w, lvl);
          compress_level_stable(RS, opts, w, d, dd, lvl);
        }
        d += dd;
        dd = std::min(dd, opts.max_rank()-d);
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_recursive_stable
    (DistSamples<scalar_t>& RS, const delem_t& Aelem, const opts_t& opts,
     WorkCompressMPI<scalar_t>& w, int d, int dd) {
      if (!this->active()) return;
      if (this->leaf()) {
        if (this->is_untouched()) {
          std::vector<std::size_t> I, J;
          I.reserve(this->rows());
          J.reserve(this->cols());
          for (std::size_t i=0; i<this->rows(); i++)
            I.push_back(i+w.offset.first);
          for (std::size_t j=0; j<this->cols(); j++)
            J.push_back(j+w.offset.second);
          _D = DistM_t(_ctxt, this->rows(), this->cols());
          Aelem(I, J, _D);
        }
      } else {
        w.split(this->_ch[0]->dims());
        this->_ch[0]->compress_recursive_stable(RS, Aelem, opts,
                                                w.c[0], d, dd);
        this->_ch[1]->compress_recursive_stable(RS, Aelem, opts,
                                                w.c[1], d, dd);
        communicate_child_data(w);
        if (!this->_ch[0]->is_compressed() ||
            !this->_ch[1]->is_compressed()) return;
        if (this->is_untouched()) {
          _B01 = DistM_t(_ctxt, w.c[0].Ir.size(), w.c[1].Ic.size());
          _B10 = DistM_t(_ctxt, w.c[1].Ir.size(), w.c[0].Ic.size());
          Aelem(w.c[0].Ir, w.c[1].Ic, _B01);
          Aelem(w.c[1].Ir, w.c[0].Ic, _B10);
        }
      }
      if (w.lvl == 0) this->_U_state = this->_V_state = State::COMPRESSED;
      else {
        if (this->is_untouched()) compute_local_samples(RS, w, d+dd);
        else compute_local_samples(RS, w, dd);
        if (!this->is_compressed()) {
          compute_U_basis_stable(opts, w, d, dd);
          compute_V_basis_stable(opts, w, d, dd);
          notify_inactives_states(w);
          if (this->is_compressed())
            reduce_local_samples(RS, w, d+dd, false);
        } else reduce_local_samples(RS, w, dd, true);
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_level_stable
    (DistSamples<scalar_t>& RS, const opts_t& opts,
     WorkCompressMPI<scalar_t>& w, int d, int dd, int lvl) {
      if (!this->active()) return;
      if (this->leaf()) {
        if (w.lvl < lvl) return;
      } else {
        if (w.lvl < lvl) {
          this->_ch[0]->compress_level_stable(RS, opts, w.c[0], d, dd, lvl);
          this->_ch[1]->compress_level_stable(RS, opts, w.c[1], d, dd, lvl);
          return;
        }
        if (!this->_ch[0]->is_compressed() ||
            !this->_ch[1]->is_compressed()) return;
      }
      if (w.lvl==0) this->_U_state = this->_V_state = State::COMPRESSED;
      else {
        if (this->is_untouched()) compute_local_samples(RS, w, d+dd);
        else compute_local_samples(RS, w, dd);
        if (!this->is_compressed()) {
          compute_U_basis_stable(opts, w, d, dd);
          compute_V_basis_stable(opts, w, d, dd);
          notify_inactives_states(w);
          if (this->is_compressed())
            reduce_local_samples(RS, w, d+dd, false);
        } else reduce_local_samples(RS, w, dd, true);
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compute_U_basis_stable
    (const opts_t& opts, WorkCompressMPI<scalar_t>& w, int d, int dd) {
      if (this->_U_state == State::COMPRESSED) return;
      int u_rows = this->leaf() ? this->rows() :
        this->_ch[0]->U_rank()+this->_ch[1]->U_rank();
      if (!w.Sr.active()) return;
      if (d+dd >= opts.max_rank() || d+dd >= u_rows ||
          update_orthogonal_basis
          (opts, w.U_r_max, w.Sr, w.Qr, d, dd,
           this->_U_state == State::UNTOUCHED)) {
        w.Qr.clear();
        auto f0 = params::flops;
        // TODO pass max_rank to ID in DistributedMatrix
        w.Sr.ID_row(_U.E(), _U.P(), w.Jr, opts.rel_tol(), opts.abs_tol(),
                    /*opts.max_rank(),*/ _ctxt_T);
        params::ID_flops += params::flops - f0;
        this->_U_rank = _U.cols();
        this->_U_rows = _U.rows();
        w.Ir.reserve(_U.cols());
        if (this->leaf())
          for (auto i : w.Jr) w.Ir.push_back(w.offset.first + i);
        else {
          auto r0 = w.c[0].Ir.size();
          for (auto i : w.Jr)
            w.Ir.push_back((i < r0) ? w.c[0].Ir[i] : w.c[1].Ir[i-r0]);
        }
        this->_U_state = State::COMPRESSED;
      } else this->_U_state = State::PARTIALLY_COMPRESSED;
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compute_V_basis_stable
    (const opts_t& opts, WorkCompressMPI<scalar_t>& w, int d, int dd) {
      if (this->_V_state == State::COMPRESSED) return;
      int v_rows = this->leaf() ? this->rows() :
        this->_ch[0]->V_rank()+this->_ch[1]->V_rank();
      if (!w.Sc.active()) return;
      if (d+dd >= opts.max_rank() || d+dd >= v_rows ||
          update_orthogonal_basis
          (opts, w.V_r_max, w.Sc, w.Qc, d, dd,
           this->_V_state == State::UNTOUCHED)) {
        w.Qc.clear();
        auto f0 = params::flops;
        // TODO pass max_rank to ID in DistributedMatrix
        w.Sc.ID_row(_V.E(), _V.P(), w.Jc, opts.rel_tol(),
                    opts.abs_tol(), /*opts.max_rank()*/ _ctxt_T);
        params::ID_flops += params::flops - f0;
        this->_V_rank = _V.cols();
        this->_V_rows = _V.rows();
        w.Ic.reserve(_V.cols());
        if (this->leaf())
          for (auto j : w.Jc) w.Ic.push_back(w.offset.second + j);
        else {
          auto r0 = w.c[0].Ic.size();
          for (auto j : w.Jc)
            w.Ic.push_back((j < r0) ? w.c[0].Ic[j] : w.c[1].Ic[j-r0]);
        }
        this->_V_state = State::COMPRESSED;
      } else this->_V_state = State::PARTIALLY_COMPRESSED;
    }

    template<typename scalar_t> bool
    HSSMatrixMPI<scalar_t>::update_orthogonal_basis
    (const opts_t& opts, scalar_t& r_max_0, const DistM_t& S,
     DistM_t& Q, int d, int dd, bool untouched) {
      int m = S.rows();
      if (d >= m) return true;
      if (Q.cols() == 0) Q = DistM_t(_ctxt, m, d+dd);
      else Q.resize(m, d+dd);
      copy(m, dd, S, 0, d, Q, 0, d, _ctxt);
      DistMW_t Q2, Q12;
      if (untouched) {
        Q2 = DistMW_t(m, std::min(d, m), Q, 0, 0);
        Q12 = DistMW_t(m, std::min(d, m), Q, 0, 0);
        copy(m, d, S, 0, 0, Q, 0, 0, _ctxt);
      } else {
        Q2 = DistMW_t(m, std::min(dd, m-(d-dd)), Q, 0, d-dd);
        Q12 = DistMW_t(m, std::min(d, m), Q, 0, 0);
      }
      auto f0 = params::flops;
      scalar_t r_min, r_max;
      Q2.orthogonalize(r_max, r_min);
      if (untouched) r_max_0 = r_max;
      if (std::abs(r_min) < opts.abs_tol() ||
          std::abs(r_min / r_max_0) < opts.rel_tol())
        return true;
      params::QR_flops += params::flops - f0;
      DistMW_t Q3(m, dd, Q, 0, d);
      DistM_t Q12tQ3(_ctxt, Q12.cols(), Q3.cols());
      auto S3norm = Q3.norm();
      f0 = params::flops;
      gemm(Trans::C, Trans::N, scalar_t(1.), Q12, Q3, scalar_t(0.), Q12tQ3);
      gemm(Trans::N, Trans::N, scalar_t(-1.), Q12, Q12tQ3, scalar_t(1.), Q3);
      // iterated classical Gram-Schmidt
      gemm(Trans::C, Trans::N, scalar_t(1.), Q12, Q3, scalar_t(0.), Q12tQ3);
      gemm(Trans::N, Trans::N, scalar_t(-1.), Q12, Q12tQ3, scalar_t(1.), Q3);
      params::ortho_flops += params::flops - f0;
      auto Q3norm = Q3.norm();
      return (Q3norm / std::sqrt(double(dd)) < opts.abs_tol())
        || (Q3norm / S3norm < opts.rel_tol());
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_MPI_COMPRESS_STABLE_HPP
