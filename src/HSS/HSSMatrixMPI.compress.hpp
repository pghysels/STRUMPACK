#ifndef HSS_MATRIX_MPI_COMPRESS_HPP
#define HSS_MATRIX_MPI_COMPRESS_HPP

#include "misc/RandomWrapper.hpp"
#include "DistSamples.hpp"
#include "DistElemMult.hpp"

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress(const DistM_t& A, const opts_t& opts) {
      auto afunc = DistElemMult<scalar_t>(A, _ctxt_all, _comm);
      switch (opts.compression_algorithm()) {
      case CompressionAlgorithm::ORIGINAL: {
        if (opts.synchronized_compression())
          compress_original_sync(afunc, afunc, opts, A.ctxt());
        else compress_original_nosync(afunc, afunc, opts, A.ctxt());
        // TODO sync should not be necessary in this case
        // compress_original_nosync(afunc, afunc, opts, A.ctxt());
      } break;
      case CompressionAlgorithm::STABLE: {
        if (opts.synchronized_compression())
          compress_stable_sync(afunc, afunc, opts, A.ctxt());
        else compress_stable_nosync(afunc, afunc, opts, A.ctxt());
        // TODO sync should not be necessary in this case
        // compress_stable_nosync(afunc, afunc, opts, A.ctxt());
      } break;
      default:
        std::cout << "Compression algorithm not recognized!" << std::endl;
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::compress
    (const dmult_t& Amult, const delem_t& Aelem,
     const opts_t& opts, int Actxt) {
      switch (opts.compression_algorithm()) {
      case CompressionAlgorithm::ORIGINAL: {
        if (opts.synchronized_compression())
          compress_original_sync(Amult, Aelem, opts, Actxt);
        else compress_original_nosync(Amult, Aelem, opts, Actxt);
      } break;
      case CompressionAlgorithm::STABLE: {
        if (opts.synchronized_compression())
          compress_stable_sync(Amult, Aelem, opts, Actxt);
        else compress_stable_nosync(Amult, Aelem, opts, Actxt);
      } break;
      default:
        std::cout << "Compression algorithm not recognized!" << std::endl;
      };
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_original_nosync
    (const dmult_t& Amult, const delem_t& Aelem,
     const opts_t& opts, int Actxt) {
      // TODO compare with sequential compression, start with d0+dd
      int d_old = 0, d = opts.d0() + opts.dd();
      DistSamples<scalar_t> RS(d, (Actxt!=-1) ? Actxt : _ctxt,
                               *this, Amult, opts);
      WorkCompressMPI<scalar_t> w;
      while (!this->is_compressed()) {
        if (d != opts.d0() + opts.dd()) RS.add_columns(d, opts);
        if (opts.verbose() && !mpi_rank(_comm))
          std::cout << "# compressing with d = " << d-opts.dd()
                    << " + " << opts.dd() << " (original)" << std::endl;
        compress_recursive_original(RS, Aelem, opts, w, d-d_old);
        d_old = d;
        d = 2 * (d_old - opts.dd()) + opts.dd();
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_original_sync
    (const dmult_t& Amult, const delem_t& Aelem,
     const opts_t& opts, int Actxt) {
      WorkCompressMPI<scalar_t> w;
      int d_old = 0, d = opts.d0();
      DistSamples<scalar_t> RS(d, (Actxt!=-1) ? Actxt : _ctxt,
                               *this, Amult, opts);
      const auto nr_lvls = this->max_levels();
      while (!this->is_compressed() && d < opts.max_rank()) {
        if (opts.verbose() && !mpi_rank(_comm))
          std::cout << "# compressing with d = " << d << ", d_old = "
                    << d_old << ", tol = " << opts.rel_tol() << std::endl;
        for (int lvl=nr_lvls-1; lvl>=0; lvl--) {
          extract_level(Aelem, opts, w, lvl);
          compress_level_original(RS, opts, w, d-d_old, lvl);
        }
        if (!this->is_compressed()) {
          d_old = d;
          d = std::min(2*d, opts.max_rank());
          RS.add_columns(d, opts);
        }
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_level
    (const delem_t& Aelem, const opts_t& opts,
     WorkCompressMPI<scalar_t>& w, int lvl) {
      std::vector<std::vector<std::size_t>> lI, lJ, I, J;
      int self = 0, before, after;
      get_extraction_indices(lI, lJ, w, self, lvl);
      allgather_extraction_indices(lI, lJ, I, J, before, self, after);
      DistM_t dummy;
      for (int i=0; i<before; i++)
        Aelem(I[i], J[i], dummy);
      extract_D_B(Aelem, ctxt_loc(), opts, w, lvl);
      for (int i=I.size()-after; i<int(I.size()); i++)
        Aelem(I[i], J[i], dummy);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::get_extraction_indices
    (std::vector<std::vector<std::size_t>>& I,
     std::vector<std::vector<std::size_t>>& J,
     WorkCompressMPI<scalar_t>& w, int& self, int lvl) {
      if (!this->active()) return;
      if (this->leaf()) {
        if (w.lvl == lvl && this->is_untouched()) {
          self++;
          if (mpi_rank(_comm)==0) {
            I.emplace_back();  J.emplace_back();
            I.back().reserve(this->rows());  J.back().reserve(this->cols());
            for (std::size_t i=0; i<this->rows(); i++)
              I.back().push_back(i+w.offset.first);
            for (std::size_t j=0; j<this->cols(); j++)
              J.back().push_back(j+w.offset.second);
          }
        }
      } else {
        w.split(this->_ch[0]->dims());
        if (w.lvl < lvl) {
          this->_ch[0]->get_extraction_indices(I, J, w.c[0], self, lvl);
          this->_ch[1]->get_extraction_indices(I, J, w.c[1], self, lvl);
          return;
        }
        communicate_child_data(w);
        if (!this->_ch[0]->is_compressed() ||
            !this->_ch[1]->is_compressed()) return;
        if (this->is_untouched()) {
          self += 2;
          if (mpi_rank(_comm)==0) {
            I.push_back(w.c[0].Ir);  J.push_back(w.c[1].Ic);
            I.push_back(w.c[1].Ir);  J.push_back(w.c[0].Ic);
          }
        }
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::allgather_extraction_indices
    (std::vector<std::vector<std::size_t>>& lI,
     std::vector<std::vector<std::size_t>>& lJ,
     std::vector<std::vector<std::size_t>>& I,
     std::vector<std::vector<std::size_t>>& J,
     int& before, int self, int& after) {
      auto P = mpi_nprocs(_comm);
      auto rank = mpi_rank(_comm);
      auto rsize = new int[P];
      rsize[rank] = 1;
      for (auto& i : lI) rsize[rank] += i.size() + 1;
      for (auto& j : lJ) rsize[rank] += j.size() + 1;
      MPI_Allgather(MPI_IN_PLACE, 0, mpi_type<int>(), rsize, 1,
                    mpi_type<int>(), _comm);
      auto displs = new int[P];
      displs[0] = 0;
      for (int p=1; p<P; p++) displs[p] = displs[p-1] + rsize[p-1];
      auto sbuf = new std::size_t[std::accumulate(rsize, rsize+P, 0)];
      auto ptr = sbuf + displs[rank];
      *ptr++ = lI.size();
      for (std::size_t i=0; i<lI.size(); i++) {
        *ptr++ = lI[i].size();
        std::copy(lI[i].begin(), lI[i].end(), ptr);
        ptr += lI[i].size();
        *ptr++ = lJ[i].size();
        std::copy(lJ[i].begin(), lJ[i].end(), ptr);
        ptr += lJ[i].size();
      }
      MPI_Allgatherv(MPI_IN_PLACE, 0, mpi_type<std::size_t>(), sbuf,
                     rsize, displs, mpi_type<std::size_t>(), _comm);
      int total = 0;
      for (int p=0; p<P; p++) total += sbuf[displs[p]];
      after = 0;
      for (int p=rank+1; p<P; p++) after += sbuf[displs[p]];
      before = total - self - after;
      I.resize(total);
      J.resize(total);
      ptr = sbuf;
      for (std::size_t p=0, i=0; p<std::size_t(P); p++) {
        auto fromp = *ptr++;
        for (std::size_t ii=i; ii<i+fromp; ii++) {
          I[ii].resize(*ptr++);
          std::copy(ptr, ptr+I[ii].size(), I[ii].data()); ptr += I[ii].size();
          J[ii].resize(*ptr++);
          std::copy(ptr, ptr+J[ii].size(), J[ii].data()); ptr += J[ii].size();
        }
        i += fromp;
      }
      delete[] sbuf;
      delete[] displs;
      delete[] rsize;
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_D_B
    (const delem_t& Aelem, int lctxt, const opts_t& opts,
     WorkCompressMPI<scalar_t>& w, int lvl) {
      if (!this->active()) return;
      if (this->leaf()) {
        if (w.lvl < lvl) return;
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
        if (w.lvl < lvl) {
          this->_ch[0]->extract_D_B(Aelem, lctxt, opts, w.c[0], lvl);
          this->_ch[1]->extract_D_B(Aelem, lctxt, opts, w.c[1], lvl);
          return;
        }
        if (!this->_ch[0]->is_compressed() ||
            !this->_ch[1]->is_compressed()) return;
        if (this->is_untouched()) {
          _B01 = DistM_t(_ctxt, w.c[0].Ir.size(), w.c[1].Ic.size());
          _B10 = DistM_t(_ctxt, w.c[1].Ir.size(), w.c[0].Ic.size());
          Aelem(w.c[0].Ir, w.c[1].Ic, _B01);
          Aelem(w.c[1].Ir, w.c[0].Ic, _B10);
        }
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_recursive_original
    (DistSamples<scalar_t>& RS, const delem_t& Aelem, const opts_t& opts,
     WorkCompressMPI<scalar_t>& w, int dd) {
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
        this->_ch[0]->compress_recursive_original
          (RS, Aelem, opts, w.c[0], dd);
        this->_ch[1]->compress_recursive_original
          (RS, Aelem, opts, w.c[1], dd);
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
        compute_local_samples(RS, w, dd);
        if (!this->is_compressed()) {
          if (compute_U_V_bases(RS.R.cols(), opts, w)) {
            reduce_local_samples(RS, w, dd, false);
            this->_U_state = this->_V_state = State::COMPRESSED;
          } else
            this->_U_state = this->_V_state = State::PARTIALLY_COMPRESSED;
        } else reduce_local_samples(RS, w, dd, true);
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_level_original
    (DistSamples<scalar_t>& RS, const opts_t& opts,
     WorkCompressMPI<scalar_t>& w, int dd, int lvl) {
      if (!this->active()) return;
      if (this->leaf()) {
        if (w.lvl < lvl) return;
      } else {
        if (w.lvl < lvl) {
          this->_ch[0]->compress_level_original(RS, opts, w.c[0], dd, lvl);
          this->_ch[1]->compress_level_original(RS, opts, w.c[1], dd, lvl);
          return;
        }
        communicate_child_data(w);
        if (!this->_ch[0]->is_compressed() ||
            !this->_ch[1]->is_compressed()) return;
      }
      if (w.lvl==0) this->_U_state = this->_V_state = State::COMPRESSED;
      else {
        compute_local_samples(RS, w, dd);
        if (!this->is_compressed()) {
          if (compute_U_V_bases(RS.R.cols(), opts, w)) {
            reduce_local_samples(RS, w, dd, false);
            this->_U_state = this->_V_state = State::COMPRESSED;
          } else
            this->_U_state = this->_V_state = State::PARTIALLY_COMPRESSED;
        } else reduce_local_samples(RS, w, dd, true);
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compute_local_samples
    (const DistSamples<scalar_t>& RS, WorkCompressMPI<scalar_t>& w, int dd) {
      auto d = RS.R.cols();
      auto d_old = d - dd;
      auto c_old = w.Sr.cols();
      assert(d_old >= 0);
      assert(c_old <= d);
      if (this->leaf()) {
        auto wR = ConstDistributedMatrixWrapperPtr
          (this->rows(), dd, RS.leaf_R, 0, d_old);
        if (!c_old) {
          w.Sr = DistM_t(_ctxt, this->rows(), dd);
          w.Sc = DistM_t(_ctxt, this->rows(), dd);
          copy(this->rows(), dd, RS.leaf_Sr, 0, d_old, w.Sr, 0, 0, _ctxt_all);
          copy(this->rows(), dd, RS.leaf_Sc, 0, d_old, w.Sc, 0, 0, _ctxt_all);
          gemm(Trans::N, Trans::N, scalar_t(-1), _D, *wR, scalar_t(1.), w.Sr);
          gemm(Trans::C, Trans::N, scalar_t(-1), _D, *wR, scalar_t(1.), w.Sc);
        } else {
          w.Sr.resize(this->rows(), c_old+dd);
          w.Sc.resize(this->rows(), c_old+dd);
          DistMW_t wSr_new(this->rows(), dd, w.Sr, 0, c_old);
          DistMW_t wSc_new(this->rows(), dd, w.Sc, 0, c_old);
          copy(this->rows(), dd, RS.leaf_Sr, 0, d_old,
               wSr_new, 0, 0, _ctxt_all);
          copy(this->rows(), dd, RS.leaf_Sc, 0, d_old,
               wSc_new, 0, 0, _ctxt_all);
          gemm(Trans::N, Trans::N, scalar_t(-1), _D, *wR,
               scalar_t(1.), wSr_new);
          gemm(Trans::C, Trans::N, scalar_t(-1), _D, *wR,
               scalar_t(1.), wSc_new);
        }
      } else {
        std::size_t c_new, c_lo;
        if (!c_old) {
          w.Sr = DistM_t(_ctxt, w.c[0].Jr.size()+w.c[1].Jr.size(), w.c[0].dS);
          w.Sc = DistM_t(_ctxt, w.c[0].Jc.size()+w.c[1].Jc.size(), w.c[0].dS);
          c_new = w.Sc.cols();
          c_lo = 0;
        } else {
          w.Sr.resize(w.Sr.rows(), c_old+dd);
          w.Sc.resize(w.Sc.rows(), c_old+dd);
          c_new = dd;
          c_lo = c_old;
        }
        DistMW_t wSr_new0(w.c[0].Jr.size(), c_new, w.Sr, 0, c_lo);
        DistMW_t wSr_new1(w.c[1].Jr.size(), c_new, w.Sr,
                          w.c[0].Jr.size(), c_lo);
        {
          // we don't know the number of rows in w.c[0].Sr, but we
          // know the indices of the rows to extract
          int c0Srows = (w.c[0].Jr.empty()) ? 0 :
            1 + *std::max_element(w.c[0].Jr.begin(), w.c[0].Jr.end());
          int c1Srows = (w.c[1].Jr.empty()) ? 0 :
            1 + *std::max_element(w.c[1].Jr.begin(), w.c[1].Jr.end());
          DistM_t wc0Sr_new(_ctxt, c0Srows, c_new);
          DistM_t wc1Sr_new(_ctxt, c1Srows, c_new);
          copy(c0Srows, c_new, w.c[0].Sr, 0, w.c[0].dS-c_new,
               wc0Sr_new, 0, 0, _ctxt_all);
          copy(c1Srows, c_new, w.c[1].Sr, 0, w.c[1].dS-c_new,
               wc1Sr_new, 0, 0, _ctxt_all);
          auto tmpr0 = wc0Sr_new.extract_rows(w.c[0].Jr, comm());
          auto tmpr1 = wc1Sr_new.extract_rows(w.c[1].Jr, comm());
          copy(w.c[0].Jr.size(), c_new, tmpr0, 0, 0,
               wSr_new0, 0, 0, _ctxt_all);
          copy(w.c[1].Jr.size(), c_new, tmpr1, 0, 0,
               wSr_new1, 0, 0, _ctxt_all);
        }

        DistM_t wc1Rr(_ctxt, _B01.cols(), c_new);
        DistM_t wc0Rr(_ctxt, _B10.cols(), c_new);
        copy(_B01.cols(), c_new, w.c[1].Rr, 0, w.c[1].dR-c_new,
             wc1Rr, 0, 0, _ctxt_all);
        copy(_B10.cols(), c_new, w.c[0].Rr, 0, w.c[0].dR-c_new,
             wc0Rr, 0, 0, _ctxt_all);
        gemm(Trans::N, Trans::N, scalar_t(-1.), _B01, wc1Rr,
             scalar_t(1.), wSr_new0);
        gemm(Trans::N, Trans::N, scalar_t(-1.), _B10, wc0Rr,
             scalar_t(1.), wSr_new1);

        DistMW_t wSc_new0(w.c[0].Jc.size(), c_new, w.Sc, 0, c_lo);
        DistMW_t wSc_new1(w.c[1].Jc.size(), c_new, w.Sc,
                          w.c[0].Jc.size(), c_lo);
        {
          int c0Srows = (w.c[0].Jc.empty()) ? 0 :
            1 + *std::max_element(w.c[0].Jc.begin(), w.c[0].Jc.end());
          int c1Srows = (w.c[1].Jc.empty()) ? 0 :
            1 + *std::max_element(w.c[1].Jc.begin(), w.c[1].Jc.end());
          DistM_t wc0Sc_new(_ctxt, c0Srows, c_new);
          DistM_t wc1Sc_new(_ctxt, c1Srows, c_new);
          copy(c0Srows, c_new, w.c[0].Sc, 0, w.c[0].dS-c_new,
               wc0Sc_new, 0, 0, _ctxt_all);
          copy(c1Srows, c_new, w.c[1].Sc, 0, w.c[1].dS-c_new,
               wc1Sc_new, 0, 0, _ctxt_all);
          auto tmpr0 = wc0Sc_new.extract_rows(w.c[0].Jc, comm());
          auto tmpr1 = wc1Sc_new.extract_rows(w.c[1].Jc, comm());
          copy(w.c[0].Jc.size(), c_new, tmpr0, 0, 0,
               wSc_new0, 0, 0, _ctxt_all);
          copy(w.c[1].Jc.size(), c_new, tmpr1, 0, 0,
               wSc_new1, 0, 0, _ctxt_all);
        }

        DistM_t wc1Rc(_ctxt, _B10.rows(), c_new);
        DistM_t wc0Rc(_ctxt, _B01.rows(), c_new);
        copy(_B10.rows(), c_new, w.c[1].Rc, 0, w.c[1].dR-c_new,
             wc1Rc, 0, 0, _ctxt_all);
        copy(_B01.rows(), c_new, w.c[0].Rc, 0, w.c[0].dR-c_new,
             wc0Rc, 0, 0, _ctxt_all);
        gemm(Trans::C, Trans::N, scalar_t(-1.), _B10, wc1Rc,
             scalar_t(1.), wSc_new0);
        gemm(Trans::C, Trans::N, scalar_t(-1.), _B01, wc0Rc,
             scalar_t(1.), wSc_new1);
        w.c[0].Sr.clear(); w.c[0].Sc.clear(); w.c[0].dS = 0;
        w.c[1].Sr.clear(); w.c[1].Sc.clear(); w.c[1].dS = 0;
      }
    }

    template<typename scalar_t> bool HSSMatrixMPI<scalar_t>::compute_U_V_bases
    (int d, const opts_t& opts, WorkCompressMPI<scalar_t>& w) {
      w.Sr.ID_row(_U.E(), _U.P(), w.Jr, opts.rel_tol(),
                  opts.abs_tol(), _ctxt_T);
      w.Sc.ID_row(_V.E(), _V.P(), w.Jc, opts.rel_tol(),
                  opts.abs_tol(), _ctxt_T);
      notify_inactives_J(w);
      if (d-opts.dd() >= opts.max_rank() ||
          (int(w.Jr.size()) <= d - opts.dd() &&
           int(w.Jc.size()) <= d - opts.dd())) {
        this->_U_rank = w.Jr.size();  this->_U_rows = w.Sr.rows();
        this->_V_rank = w.Jc.size();  this->_V_rows = w.Sc.rows();
        w.Ir.reserve(w.Jr.size());
        w.Ic.reserve(w.Jc.size());
        if (this->leaf()) {
          for (auto i : w.Jr) w.Ir.push_back(w.offset.first + i);
          for (auto j : w.Jc) w.Ic.push_back(w.offset.second + j);
        } else {
          auto r0 = w.c[0].Ir.size();
          for (auto i : w.Jr)
            w.Ir.push_back((i < r0) ? w.c[0].Ir[i] : w.c[1].Ir[i-r0]);
          r0 = w.c[0].Ic.size();
          for (auto j : w.Jc)
            w.Ic.push_back((j < r0) ? w.c[0].Ic[j] : w.c[1].Ic[j-r0]);
        }
        // TODO clear w.c[0].Ir, w.c[1].Ir, w.c[0].Ic, w.c[1].Ic
        return true;
      } else {
        w.Jr.clear();
        w.Jc.clear();
        return false;
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::reduce_local_samples
    (const DistSamples<scalar_t>& RS, WorkCompressMPI<scalar_t>& w,
     int dd, bool was_compressed) {
      auto d = RS.R.cols();
      auto d_old = d - dd;
      auto c_old = w.Rr.cols();
      assert(d_old >= 0);
      assert(c_old <= d);
      if (this->leaf()) {
        if (!c_old) {
          auto c_new = (was_compressed) ? dd : d;
          auto tmpR = ConstDistributedMatrixWrapperPtr
            (this->rows(), c_new, RS.leaf_R, 0, d-c_new);
          w.Rr = DistM_t(_ctxt, this->V_rank(), c_new);
          w.Rc = DistM_t(_ctxt, this->U_rank(), c_new);
          _V.applyC(*tmpR, w.Rr);
          _U.applyC(*tmpR, w.Rc);
        } else {
          auto tmpR = ConstDistributedMatrixWrapperPtr
            (this->rows(), dd, RS.leaf_R, 0, d_old);
          DistM_t wRr_new(_ctxt, w.Rr.rows(), dd);
          DistM_t wRc_new(_ctxt, w.Rc.rows(), dd);
          _V.applyC(*tmpR, wRr_new);
          _U.applyC(*tmpR, wRc_new);
          w.Rr.hconcat(wRr_new);
          w.Rc.hconcat(wRc_new);
        }
      } else {
        if (!c_old) {
          auto wRr = vconcat
            (w.c[0].dR, this->_ch[0]->V_rank(), this->_ch[1]->V_rank(),
             w.c[0].Rr, w.c[1].Rr, _ctxt, _ctxt_all);
          auto wRc = vconcat
            (w.c[0].dR, this->_ch[0]->U_rank(), this->_ch[1]->U_rank(),
             w.c[0].Rc, w.c[1].Rc, _ctxt, _ctxt_all);
          w.Rr = DistM_t(_ctxt, this->V_rank(), w.c[0].dR);
          w.Rc = DistM_t(_ctxt, this->U_rank(), w.c[0].dR);
          _V.applyC(wRr, w.Rr);
          _U.applyC(wRc, w.Rc);
        } else {
          DistM_t wRr(_ctxt, this->V_rows(), dd);
          copy(w.c[0].Ic.size(), dd, w.c[0].Rr, 0, w.c[0].dR - dd,
               wRr, 0, 0, _ctxt_all);
          copy(w.c[1].Ic.size(), dd, w.c[1].Rr, 0, w.c[1].dR - dd,
               wRr, w.c[0].Ic.size(), 0, _ctxt_all);
          DistM_t wRc(_ctxt, this->U_rows(), dd);
          copy(w.c[0].Ir.size(), dd, w.c[0].Rc, 0, w.c[0].dR - dd,
               wRc, 0, 0, _ctxt_all);
          copy(w.c[1].Ir.size(), dd, w.c[1].Rc, 0, w.c[1].dR - dd,
               wRc, w.c[0].Ir.size(), 0, _ctxt_all);

          DistM_t wRr_new(_ctxt, w.Rr.rows(), dd);
          DistM_t wRc_new(_ctxt, w.Rc.rows(), dd);
          _V.applyC(wRr, wRr_new);
          _U.applyC(wRc, wRc_new);
          w.Rr.hconcat(wRr_new);
          w.Rc.hconcat(wRc_new);
        }
        w.c[0].Rr.clear(); w.c[0].Rc.clear(); w.c[0].dR = 0;
        w.c[1].Rr.clear(); w.c[1].Rc.clear(); w.c[1].dR = 0;
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_MPI_COMPRESS_HPP
