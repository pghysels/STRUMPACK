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
#ifndef HSS_MATRIX_MPI_COMPRESS_HPP
#define HSS_MATRIX_MPI_COMPRESS_HPP


namespace strumpack {
  namespace HSS {

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::redistribute_to_tree_to_buffers
    (const DistM_t& A, std::size_t Arlo, std::size_t Aclo,
     std::vector<std::vector<scalar_t>>& sbuf, int dest) {
      if (!A.active()) return;
      const auto nprows = grid()->nprows();
      const auto npcols = grid()->npcols();
      if (this->leaf()) {
        const auto B = DistM_t::default_MB;
        const DistMW_t Ad
          (this->rows(), this->cols(), const_cast<DistM_t&>(A), Arlo, Aclo);
        int rlo, rhi, clo, chi;
        Ad.lranges(rlo, rhi, clo, chi);
        // destination rank is:
        //  dest + ((r / B) % prows) + ((c / B) % pcols) * prows
        std::vector<int> destr(rhi-rlo), destc(chi-clo);
        for (int r=rlo; r<rhi; r++)
          destr[r-rlo] = dest + (Ad.rowl2g_fixed(r) / B) % nprows;
        for (int c=clo; c<chi; c++)
          destc[c-clo] = ((Ad.coll2g_fixed(c) / B) % npcols) * nprows;
        {
          std::vector<std::size_t> cnt(sbuf.size());
          for (int c=clo; c<chi; c++)
            for (int r=rlo; r<rhi; r++)
              cnt[destr[r-rlo]+destc[c-clo]]++;
          for (std::size_t p=0; p<sbuf.size(); p++)
            sbuf[p].reserve(sbuf[p].size()+cnt[p]);
        }
        for (int c=clo; c<chi; c++)
          for (int r=rlo; r<rhi; r++)
            sbuf[destr[r-rlo]+destc[c-clo]].push_back(Ad(r,c));
      } else {
        auto m0 = child(0)->rows();
        auto n0 = child(0)->cols();
        auto m1 = child(1)->rows();
        auto n1 = child(1)->cols();
        child(0)->redistribute_to_tree_to_buffers
          (A, Arlo, Aclo, sbuf, dest);
        child(1)->redistribute_to_tree_to_buffers
          (A, Arlo+m0, Aclo+n0, sbuf, dest+Pl());
        assert(A.MB() == DistM_t::default_MB);
        const auto B = DistM_t::default_MB;
        const DistMW_t A01(m0, n1, const_cast<DistM_t&>(A), Arlo, Aclo+n0);
        const DistMW_t A10(m1, n0, const_cast<DistM_t&>(A), Arlo+m0, Aclo);
        int A01rlo, A01rhi, A01clo, A01chi;
        A01.lranges(A01rlo, A01rhi, A01clo, A01chi);
        int A10rlo, A10rhi, A10clo, A10chi;
        A10.lranges(A10rlo, A10rhi, A10clo, A10chi);
        // destination rank is:
        //  dest + ((r / B) % prows) + ((c / B) % pcols) * prows
        std::vector<int> destrA01(A01rhi-A01rlo), destcA01(A01chi-A01clo),
          destrA10(A10rhi-A10rlo), destcA10(A10chi-A10clo);
        for (int r=A01rlo; r<A01rhi; r++)
          destrA01[r-A01rlo] = dest + (A01.rowl2g_fixed(r) / B) % nprows;
        for (int c=A01clo; c<A01chi; c++)
          destcA01[c-A01clo] = ((A01.coll2g_fixed(c) / B) % npcols) * nprows;
        for (int r=A10rlo; r<A10rhi; r++)
          destrA10[r-A10rlo] = dest + (A10.rowl2g_fixed(r) / B) % nprows;
        for (int c=A10clo; c<A10chi; c++)
          destcA10[c-A10clo] = ((A10.coll2g_fixed(c) / B) % npcols) * nprows;
        {
          std::vector<std::size_t> cnt(sbuf.size());
          for (int c=A01clo; c<A01chi; c++)
            for (int r=A01rlo; r<A01rhi; r++)
              cnt[destrA01[r-A01rlo]+destcA01[c-A01clo]]++;
          for (int c=A10clo; c<A10chi; c++)
            for (int r=A10rlo; r<A10rhi; r++)
              cnt[destrA10[r-A10rlo]+destcA10[c-A10clo]]++;
          for (std::size_t p=0; p<sbuf.size(); p++)
            sbuf[p].reserve(sbuf[p].size()+cnt[p]);
        }
        for (int c=A01clo; c<A01chi; c++)
          for (int r=A01rlo; r<A01rhi; r++)
            sbuf[destrA01[r-A01rlo]+destcA01[c-A01clo]].push_back(A01(r,c));
        for (int c=A10clo; c<A10chi; c++)
          for (int r=A10rlo; r<A10rhi; r++)
            sbuf[destrA10[r-A10rlo]+destcA10[c-A10clo]].push_back(A10(r,c));
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::redistribute_to_tree_from_buffers
    (const DistM_t& A, std::size_t Arlo, std::size_t Aclo,
     std::vector<scalar_t*>& pbuf) {
      if (!this->active()) return;
      const auto Aprows = A.grid()->nprows();
      const auto Apcols = A.grid()->npcols();
      if (this->leaf()) {
        A_ = DistM_t(grid(), this->rows(), this->cols());
        const auto B = DistM_t::default_MB;
        int rlo, rhi, clo, chi;
        A_.lranges(rlo, rhi, clo, chi);
        std::vector<int> srcr(rhi-rlo);
        for (int r=rlo; r<rhi; r++)
          srcr[r-rlo] = ((A_.rowl2g_fixed(r) + Arlo) / B) % Aprows;
        for (int c=clo; c<chi; c++)
          for (int srcc=(((A_.coll2g_fixed(c)+Aclo)/B)%Apcols)*Aprows,
                 r=rlo; r<rhi; r++)
            A_(r,c) = *(pbuf[srcr[r-rlo] + srcc]++);
      } else {
        auto m0 = child(0)->rows();
        auto n0 = child(0)->cols();
        auto m1 = child(1)->rows();
        auto n1 = child(1)->cols();
        child(0)->redistribute_to_tree_from_buffers
          (A, Arlo, Aclo, pbuf);
        child(1)->redistribute_to_tree_from_buffers
          (A, Arlo+m0, Aclo+n0, pbuf);
        A01_ = DistM_t(grid(), m0, n1);
        A10_ = DistM_t(grid(), m1, n0);
        assert(A.I() == 1 && A.J() == 1);
        const auto B = DistM_t::default_MB;
        int rlo, rhi, clo, chi;
        A01_.lranges(rlo, rhi, clo, chi);
        std::vector<int> srcr(rhi-rlo);
        for (int r=rlo; r<rhi; r++)
          srcr[r-rlo] = ((A01_.rowl2g_fixed(r) + Arlo) / B) % Aprows;
        for (int c=clo; c<chi; c++)
          for (int srcc=(((A01_.coll2g_fixed(c)+Aclo+n0)/B)%Apcols)*Aprows,
                 r=rlo; r<rhi; r++)
            A01_(r,c) = *(pbuf[srcr[r-rlo] + srcc]++);
        A10_.lranges(rlo, rhi, clo, chi);
        srcr.resize(rhi-rlo);
        for (int r=rlo; r<rhi; r++)
          srcr[r-rlo] = ((A10_.rowl2g_fixed(r) + Arlo+m0) / B) % Aprows;
        for (int c=clo; c<chi; c++)
          for (int srcc=(((A10_.coll2g_fixed(c)+Aclo)/B)%Apcols)*Aprows,
                 r=rlo; r<rhi; r++)
            A10_(r,c) = *(pbuf[srcr[r-rlo]+srcc]++);
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::delete_redistributed_input() {
      if (this->leaf()) A_.clear();
      else {
        child(0)->delete_redistributed_input();
        child(1)->delete_redistributed_input();
        A01_.clear();
        A10_.clear();
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress(const DistM_t& A, const opts_t& opts) {
      TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_compress);
      TIMER_TIME(TaskType::REDIST_2D_TO_HSS, 0, t_redist);
      {
        std::vector<std::vector<scalar_t>> sbuf(Comm().size());
        redistribute_to_tree_to_buffers(A, 0, 0, sbuf);
        std::vector<scalar_t,NoInit<scalar_t>> rbuf;
        std::vector<scalar_t*> pbuf;
        Comm().all_to_all_v(sbuf, rbuf, pbuf);
        redistribute_to_tree_from_buffers(A, 0, 0, pbuf);
      }
      TIMER_STOP(t_redist);
      DistElemMult<scalar_t> Afunc(A);
      switch (opts.compression_algorithm()) {
      case CompressionAlgorithm::ORIGINAL:
        if (opts.synchronized_compression())
          compress_original_sync(Afunc, Afunc, opts);
        else compress_original_nosync(Afunc, Afunc, opts);
        break;
      case CompressionAlgorithm::STABLE:
        if (opts.synchronized_compression())
          compress_stable_sync(Afunc, Afunc, opts);
        else compress_stable_nosync(Afunc, Afunc, opts);
        break;
      case CompressionAlgorithm::HARD_RESTART:
        if (opts.synchronized_compression())
          compress_hard_restart_sync(Afunc, Afunc, opts);
        else compress_hard_restart_nosync(Afunc, Afunc, opts);
        break;
      default:
        std::cout << "Compression algorithm not recognized!" << std::endl;
      };
      delete_redistributed_input();
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::compress
    (const dmult_t& Amult, const delem_blocks_t& Aelem, const opts_t& opts) {
      TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_compress);
      if (!opts.synchronized_compression())
        std::cerr << "WARNING: Non synchronized block-extraction version"
                  << "  of compression not supported,"
                  << " using synchronized extraction routine" << std::endl;
      switch (opts.compression_algorithm()) {
      case CompressionAlgorithm::ORIGINAL:
        compress_original_sync(Amult, Aelem, opts); break;
      case CompressionAlgorithm::STABLE:
        compress_stable_sync(Amult, Aelem, opts); break;
      case CompressionAlgorithm::HARD_RESTART:
        compress_hard_restart_sync(Amult, Aelem, opts); break;
      default:
        std::cout << "Compression algorithm not recognized!" << std::endl;
      };
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::compress
    (const dmult_t& Amult, const delem_t& Aelem, const opts_t& opts) {
      TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_compress);
      auto Aelemw = [&]
        (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
         DistM_t& B, const DistM_t& A, std::size_t rlo, std::size_t clo,
         MPI_Comm comm) {
        Aelem(I, J, B);
      };
      switch (opts.compression_algorithm()) {
      case CompressionAlgorithm::ORIGINAL: {
        if (opts.synchronized_compression())
          compress_original_sync(Amult, Aelemw, opts);
        else compress_original_nosync(Amult, Aelemw, opts);
      } break;
      case CompressionAlgorithm::STABLE: {
        if (opts.synchronized_compression())
          compress_stable_sync(Amult, Aelemw, opts);
        else compress_stable_nosync(Amult, Aelemw, opts);
      } break;
      case CompressionAlgorithm::HARD_RESTART: {
        if (opts.synchronized_compression())
          compress_hard_restart_sync(Amult, Aelemw, opts);
        else compress_hard_restart_nosync(Amult, Aelemw, opts);
      } break;
      default:
        std::cout << "Compression algorithm not recognized!" << std::endl;
      };
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_original_nosync
    (const dmult_t& Amult, const delemw_t& Aelem, const opts_t& opts) {
      // TODO compare with sequential compression, start with d0+p
      int d_old = 0, d = opts.d0() + opts.p();
      DistSamples<scalar_t> RS(d, grid(), *this, Amult, opts);
      WorkCompressMPI<scalar_t> w;
      while (!this->is_compressed()) {
        if (d != opts.d0() + opts.p()) RS.add_columns(d, opts);
        if (opts.verbose() && Comm().is_root())
          std::cout << "# compressing with d = " << d-opts.p()
                    << " + " << opts.p() << " (original)" << std::endl;
        compress_recursive_original(RS, Aelem, opts, w, d-d_old);
        d_old = d;
        d = 2 * (d_old - opts.p()) + opts.p();
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_original_sync
    (const dmult_t& Amult, const delemw_t& Aelem, const opts_t& opts) {
      WorkCompressMPI<scalar_t> w;
      int d_old = 0, d = opts.d0();
      DistSamples<scalar_t> RS(d, grid(), *this, Amult, opts);
      const auto nr_lvls = this->max_levels();
      while (!this->is_compressed() && d < opts.max_rank()) {
        if (opts.verbose() && Comm().is_root())
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

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_original_sync
    (const dmult_t& Amult, const delem_blocks_t& Aelem, const opts_t& opts) {
      WorkCompressMPI<scalar_t> w;
      int d_old = 0, d = opts.d0();
      DistSamples<scalar_t> RS(d, grid(), *this, Amult, opts);
      const auto nr_lvls = this->max_levels();
      while (!this->is_compressed() && d < opts.max_rank()) {
        if (opts.verbose() && Comm().is_root())
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

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_hard_restart_nosync
    (const dmult_t& Amult, const delemw_t& Aelem, const opts_t& opts) {
      int d_old = 0, d = opts.d0() + opts.p();
      DistSamples<scalar_t> RS(d, grid(), *this, Amult, opts, true);
      while (!this->is_compressed()) {
        WorkCompressMPI<scalar_t> w;
        if (d != opts.d0() + opts.p()) RS.add_columns(d, opts);
        if (opts.verbose() && Comm().is_root())
          std::cout << "# compressing with d = " << d-opts.p()
                    << " + " << opts.p() << " (original, hard restart)"
                    << std::endl;
        compress_recursive_original(RS, Aelem, opts, w, d);
        if (!this->is_compressed()) {
          d_old = d;
          d = 2 * (d_old - opts.p()) + opts.p();
          reset();
        }
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_hard_restart_sync
    (const dmult_t& Amult, const delemw_t& Aelem, const opts_t& opts) {
      std::cout << "TODO: HSSMatrixMPI<scalar_t>::compress_hard_restart_sync"
                << std::endl;
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_hard_restart_sync
    (const dmult_t& Amult, const delem_blocks_t& Aelem, const opts_t& opts) {
      std::cout << "TODO: HSSMatrixMPI<scalar_t>::compress_hard_restart_sync"
                << std::endl;
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_level
    (const delemw_t& Aelem, const opts_t& opts,
     WorkCompressMPI<scalar_t>& w, int lvl) {
      std::vector<std::vector<std::size_t>> lI, lJ, I, J;
      int self = 0, before, after;
      get_extraction_indices(lI, lJ, w, self, lvl);
      allgather_extraction_indices(lI, lJ, I, J, before, self, after);
      DistM_t dummy;
      for (int i=0; i<before; i++)
        Aelem(I[i], J[i], dummy, A_, 0, 0, comm());
      if (I.size()-after-before)
        extract_D_B(Aelem, grid_local(), opts, w, lvl);
      for (int i=I.size()-after; i<int(I.size()); i++)
        Aelem(I[i], J[i], dummy, A_, 0, 0, comm());
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_level
    (const delem_blocks_t& Aelem, const opts_t& opts,
     WorkCompressMPI<scalar_t>& w, int lvl) {
      std::vector<std::vector<std::size_t>> lI, lJ, I, J;
      std::vector<DistMW_t> lB;
      int self = 0, before, after;
      get_extraction_indices(lI, lJ, lB, grid_local(), w, self, lvl);
      allgather_extraction_indices(lI, lJ, I, J, before, self, after);
      DistMW_t dummy;
      std::vector<DistMW_t> B(I.size(), dummy);
      std::copy(lB.begin(), lB.end(), B.begin()+before);
      assert(before+lB.size()+after == I.size());
      Aelem(I, J, B);
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
          if (Comm().is_root()) {
            I.emplace_back();
            J.emplace_back();
            I.back().reserve(this->rows());
            J.back().reserve(this->cols());
            for (std::size_t i=0; i<this->rows(); i++)
              I.back().push_back(i+w.offset.first);
            for (std::size_t j=0; j<this->cols(); j++)
              J.back().push_back(j+w.offset.second);
          }
        }
      } else {
        w.split(child(0)->dims());
        if (w.lvl < lvl) {
          child(0)->get_extraction_indices(I, J, w.c[0], self, lvl);
          child(1)->get_extraction_indices(I, J, w.c[1], self, lvl);
          return;
        }
        communicate_child_data(w);
        if (!child(0)->is_compressed() ||
            !child(1)->is_compressed()) return;
        if (this->is_untouched()) {
          self += 2;
          if (Comm().is_root()) {
            I.push_back(w.c[0].Ir);  J.push_back(w.c[1].Ic);
            I.push_back(w.c[1].Ir);  J.push_back(w.c[0].Ic);
          }
        }
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::get_extraction_indices
    (std::vector<std::vector<std::size_t>>& I,
     std::vector<std::vector<std::size_t>>& J, std::vector<DistMW_t>& B,
     const BLACSGrid* lg, WorkCompressMPI<scalar_t>& w, int& self, int lvl) {
      if (!this->active()) return;
      if (this->leaf()) {
        if (w.lvl == lvl && this->is_untouched()) {
          self++;
          if (Comm().is_root()) {
            I.emplace_back();
            J.emplace_back();
            I.back().reserve(this->rows());
            J.back().reserve(this->cols());
            for (std::size_t i=0; i<this->rows(); i++)
              I.back().push_back(i+w.offset.first);
            for (std::size_t j=0; j<this->cols(); j++)
              J.back().push_back(j+w.offset.second);
          }
          D_ = DistM_t(grid(), this->rows(), this->cols());
          B.push_back(DistMW_t(D_));
        }
      } else {
        w.split(child(0)->dims());
        if (w.lvl < lvl) {
          child(0)->get_extraction_indices(I, J, B, lg, w.c[0], self, lvl);
          child(1)->get_extraction_indices(I, J, B, lg, w.c[1], self, lvl);
          return;
        }
        communicate_child_data(w);
        if (!child(0)->is_compressed() ||
            !child(1)->is_compressed()) return;
        if (this->is_untouched()) {
          self += 2;
          if (Comm().is_root()) {
            I.push_back(w.c[0].Ir);  J.push_back(w.c[1].Ic);
            I.push_back(w.c[1].Ir);  J.push_back(w.c[0].Ic);
          }
          B01_ = DistM_t(grid(), w.c[0].Ir.size(), w.c[1].Ic.size());
          B10_ = DistM_t(grid(), w.c[1].Ir.size(), w.c[0].Ic.size());
          B.push_back(DistMW_t(B01_));
          B.push_back(DistMW_t(B10_));
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
      const int rank = Comm().rank();
      const int P = Comm().size();
      std::unique_ptr<int[]> iwork(new int[2*P]);
      auto rsize = iwork.get();
      auto displs = rsize + P;
      rsize[rank] = 1;
      for (auto& i : lI) rsize[rank] += i.size() + 1;
      for (auto& j : lJ) rsize[rank] += j.size() + 1;
      Comm().all_gather(rsize, 1);
      displs[0] = 0;
      for (int p=1; p<P; p++) displs[p] = displs[p-1] + rsize[p-1];
      std::unique_ptr<std::size_t[]> sbuf
        (new std::size_t[std::accumulate(rsize, rsize+P, 0)]);
      auto ptr = sbuf.get() + displs[rank];
      *ptr++ = lI.size();
      for (std::size_t i=0; i<lI.size(); i++) {
        *ptr++ = lI[i].size();
        std::copy(lI[i].begin(), lI[i].end(), ptr);
        ptr += lI[i].size();
        *ptr++ = lJ[i].size();
        std::copy(lJ[i].begin(), lJ[i].end(), ptr);
        ptr += lJ[i].size();
      }
      Comm().all_gather_v(sbuf.get(), rsize, displs);
      int total = 0;
      for (int p=0; p<P; p++) total += sbuf[displs[p]];
      after = 0;
      for (int p=rank+1; p<P; p++) after += sbuf[displs[p]];
      before = total - self - after;
      I.resize(total);
      J.resize(total);
      ptr = sbuf.get();
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
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::extract_D_B
    (const delemw_t& Aelem, const BLACSGrid* lg, const opts_t& opts,
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
          D_ = DistM_t(grid(), this->rows(), this->cols());
          Aelem(I, J, D_, A_, w.offset.first, w.offset.second, comm());
        }
      } else {
        if (w.lvl < lvl) {
          child(0)->extract_D_B(Aelem, lg, opts, w.c[0], lvl);
          child(1)->extract_D_B(Aelem, lg, opts, w.c[1], lvl);
          return;
        }
        if (!child(0)->is_compressed() ||
            !child(1)->is_compressed()) return;
        if (this->is_untouched()) {
          B01_ = DistM_t(grid(), w.c[0].Ir.size(), w.c[1].Ic.size());
          B10_ = DistM_t(grid(), w.c[1].Ir.size(), w.c[0].Ic.size());
          Aelem(w.c[0].Ir, w.c[1].Ic, B01_, A01_,
                w.offset.first, w.offset.second+child(0)->cols(), comm());
          Aelem(w.c[1].Ir, w.c[0].Ic, B10_, A10_,
                w.offset.first+child(0)->rows(), w.offset.second, comm());
        }
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compress_recursive_original
    (DistSamples<scalar_t>& RS, const delemw_t& Aelem, const opts_t& opts,
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
          D_ = DistM_t(grid(), this->rows(), this->cols());
          Aelem(I, J, D_, A_, w.offset.first, w.offset.second, comm());
        }
      } else {
        w.split(child(0)->dims());
        child(0)->compress_recursive_original
          (RS, Aelem, opts, w.c[0], dd);
        child(1)->compress_recursive_original
          (RS, Aelem, opts, w.c[1], dd);
        communicate_child_data(w);
        if (!child(0)->is_compressed() ||
            !child(1)->is_compressed()) return;
        if (this->is_untouched()) {
          B01_ = DistM_t(grid(), w.c[0].Ir.size(), w.c[1].Ic.size());
          B10_ = DistM_t(grid(), w.c[1].Ir.size(), w.c[0].Ic.size());
          Aelem(w.c[0].Ir, w.c[1].Ic, B01_, A01_,
                w.offset.first, w.offset.second+child(0)->cols(), comm());
          Aelem(w.c[1].Ir, w.c[0].Ic, B10_, A10_,
                w.offset.first+child(0)->rows(), w.offset.second, comm());
        }
      }
      if (w.lvl == 0) this->U_state_ = this->V_state_ = State::COMPRESSED;
      else {
        compute_local_samples(RS, w, dd);
        if (!this->is_compressed()) {
          if (compute_U_V_bases(RS.R.cols(), opts, w)) {
            reduce_local_samples(RS, w, dd, false);
            this->U_state_ = this->V_state_ = State::COMPRESSED;
          } else
            this->U_state_ = this->V_state_ = State::PARTIALLY_COMPRESSED;
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
          child(0)->compress_level_original(RS, opts, w.c[0], dd, lvl);
          child(1)->compress_level_original(RS, opts, w.c[1], dd, lvl);
          return;
        }
        communicate_child_data(w);
        if (!child(0)->is_compressed() ||
            !child(1)->is_compressed()) return;
      }
      if (w.lvl==0) this->U_state_ = this->V_state_ = State::COMPRESSED;
      else {
        compute_local_samples(RS, w, dd);
        if (!this->is_compressed()) {
          if (compute_U_V_bases(RS.R.cols(), opts, w)) {
            reduce_local_samples(RS, w, dd, false);
            this->U_state_ = this->V_state_ = State::COMPRESSED;
          } else
            this->U_state_ = this->V_state_ = State::PARTIALLY_COMPRESSED;
        } else reduce_local_samples(RS, w, dd, true);
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::compute_local_samples
    (const DistSamples<scalar_t>& RS, WorkCompressMPI<scalar_t>& w, int dd) {
      TIMER_TIME(TaskType::COMPUTE_SAMPLES, 1, t_compute);
      auto d = RS.R.cols();
      auto d_old = d - dd;
      auto c_old = w.Sr.cols();
      assert(d_old >= 0);
      assert(c_old <= d);
      if (this->leaf()) {
        auto wR = ConstDistributedMatrixWrapperPtr
          (this->rows(), dd, RS.leaf_R, 0, d_old);
        if (!c_old) {
          w.Sr = DistM_t(grid(), this->rows(), dd);
          w.Sc = DistM_t(grid(), this->rows(), dd);
          copy(this->rows(), dd, RS.leaf_Sr, 0, d_old, w.Sr, 0, 0, grid()->ctxt_all());
          copy(this->rows(), dd, RS.leaf_Sc, 0, d_old, w.Sc, 0, 0, grid()->ctxt_all());
          gemm(Trans::N, Trans::N, scalar_t(-1), D_, *wR, scalar_t(1.), w.Sr);
          gemm(Trans::C, Trans::N, scalar_t(-1), D_, *wR, scalar_t(1.), w.Sc);
          STRUMPACK_UPDATE_SAMPLE_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(-1), D_, *wR, scalar_t(1.)) +
             gemm_flops(Trans::C, Trans::N, scalar_t(-1), D_, *wR, scalar_t(1.)));
        } else {
          w.Sr.resize(this->rows(), c_old+dd);
          w.Sc.resize(this->rows(), c_old+dd);
          DistMW_t wSr_new(this->rows(), dd, w.Sr, 0, c_old);
          DistMW_t wSc_new(this->rows(), dd, w.Sc, 0, c_old);
          copy(this->rows(), dd, RS.leaf_Sr, 0, d_old, wSr_new, 0, 0, grid()->ctxt_all());
          copy(this->rows(), dd, RS.leaf_Sc, 0, d_old, wSc_new, 0, 0, grid()->ctxt_all());
          gemm(Trans::N, Trans::N, scalar_t(-1), D_, *wR,
               scalar_t(1.), wSr_new);
          gemm(Trans::C, Trans::N, scalar_t(-1), D_, *wR,
               scalar_t(1.), wSc_new);
          STRUMPACK_UPDATE_SAMPLE_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(-1), D_, *wR, scalar_t(1.)) +
             gemm_flops(Trans::C, Trans::N, scalar_t(-1), D_, *wR, scalar_t(1.)));
        }
      } else {
        std::size_t c_new, c_lo;
        if (!c_old) {
          w.Sr = DistM_t(grid(), w.c[0].Jr.size()+w.c[1].Jr.size(), w.c[0].dS);
          w.Sc = DistM_t(grid(), w.c[0].Jc.size()+w.c[1].Jc.size(), w.c[0].dS);
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
          DistM_t wc0Sr_new(grid(), c0Srows, c_new);
          DistM_t wc1Sr_new(grid(), c1Srows, c_new);
          copy(c0Srows, c_new, w.c[0].Sr, 0, w.c[0].dS-c_new, wc0Sr_new, 0, 0, grid()->ctxt_all());
          copy(c1Srows, c_new, w.c[1].Sr, 0, w.c[1].dS-c_new, wc1Sr_new, 0, 0, grid()->ctxt_all());
          auto tmpr0 = wc0Sr_new.extract_rows(w.c[0].Jr);
          auto tmpr1 = wc1Sr_new.extract_rows(w.c[1].Jr);
          copy(w.c[0].Jr.size(), c_new, tmpr0, 0, 0, wSr_new0, 0, 0, grid()->ctxt_all());
          copy(w.c[1].Jr.size(), c_new, tmpr1, 0, 0, wSr_new1, 0, 0, grid()->ctxt_all());
        }

        DistM_t wc1Rr(grid(), B01_.cols(), c_new);
        DistM_t wc0Rr(grid(), B10_.cols(), c_new);
        copy(B01_.cols(), c_new, w.c[1].Rr, 0, w.c[1].dR-c_new, wc1Rr, 0, 0, grid()->ctxt_all());
        copy(B10_.cols(), c_new, w.c[0].Rr, 0, w.c[0].dR-c_new, wc0Rr, 0, 0, grid()->ctxt_all());
        gemm(Trans::N, Trans::N, scalar_t(-1.), B01_, wc1Rr,
             scalar_t(1.), wSr_new0);
        gemm(Trans::N, Trans::N, scalar_t(-1.), B10_, wc0Rr,
             scalar_t(1.), wSr_new1);
        STRUMPACK_UPDATE_SAMPLE_FLOPS
          (gemm_flops(Trans::N, Trans::N, scalar_t(-1.), B01_, wc1Rr, scalar_t(1.)) +
           gemm_flops(Trans::N, Trans::N, scalar_t(-1.), B10_, wc0Rr, scalar_t(1.)));

        DistMW_t wSc_new0(w.c[0].Jc.size(), c_new, w.Sc, 0, c_lo);
        DistMW_t wSc_new1(w.c[1].Jc.size(), c_new, w.Sc,
                          w.c[0].Jc.size(), c_lo);
        {
          int c0Srows = (w.c[0].Jc.empty()) ? 0 :
            1 + *std::max_element(w.c[0].Jc.begin(), w.c[0].Jc.end());
          int c1Srows = (w.c[1].Jc.empty()) ? 0 :
            1 + *std::max_element(w.c[1].Jc.begin(), w.c[1].Jc.end());
          DistM_t wc0Sc_new(grid(), c0Srows, c_new);
          DistM_t wc1Sc_new(grid(), c1Srows, c_new);
          copy(c0Srows, c_new, w.c[0].Sc, 0, w.c[0].dS-c_new, wc0Sc_new, 0, 0, grid()->ctxt_all());
          copy(c1Srows, c_new, w.c[1].Sc, 0, w.c[1].dS-c_new, wc1Sc_new, 0, 0, grid()->ctxt_all());
          auto tmpr0 = wc0Sc_new.extract_rows(w.c[0].Jc);
          auto tmpr1 = wc1Sc_new.extract_rows(w.c[1].Jc);
          copy(w.c[0].Jc.size(), c_new, tmpr0, 0, 0, wSc_new0, 0, 0, grid()->ctxt_all());
          copy(w.c[1].Jc.size(), c_new, tmpr1, 0, 0, wSc_new1, 0, 0, grid()->ctxt_all());
        }

        DistM_t wc1Rc(grid(), B10_.rows(), c_new);
        DistM_t wc0Rc(grid(), B01_.rows(), c_new);
        copy(B10_.rows(), c_new, w.c[1].Rc, 0, w.c[1].dR-c_new, wc1Rc, 0, 0, grid()->ctxt_all());
        copy(B01_.rows(), c_new, w.c[0].Rc, 0, w.c[0].dR-c_new, wc0Rc, 0, 0, grid()->ctxt_all());
        gemm(Trans::C, Trans::N, scalar_t(-1.), B10_, wc1Rc,
             scalar_t(1.), wSc_new0);
        gemm(Trans::C, Trans::N, scalar_t(-1.), B01_, wc0Rc,
             scalar_t(1.), wSc_new1);
        STRUMPACK_UPDATE_SAMPLE_FLOPS
          (gemm_flops(Trans::C, Trans::N, scalar_t(-1.), B10_, wc1Rc, scalar_t(1.)) +
           gemm_flops(Trans::C, Trans::N, scalar_t(-1.), B01_, wc0Rc, scalar_t(1.)));
        w.c[0].Sr.clear(); w.c[0].Sc.clear(); w.c[0].dS = 0;
        w.c[1].Sr.clear(); w.c[1].Sc.clear(); w.c[1].dS = 0;
      }
    }

    template<typename scalar_t> bool HSSMatrixMPI<scalar_t>::compute_U_V_bases
    (int d, const opts_t& opts, WorkCompressMPI<scalar_t>& w) {
      auto rtol = opts.rel_tol() / w.lvl;
      auto atol = opts.abs_tol() / w.lvl;
      auto gT = grid()->transpose();
      w.Sr.ID_row(U_.E(), U_.P(), w.Jr, rtol, atol, opts.max_rank(), &gT);
      w.Sc.ID_row(V_.E(), V_.P(), w.Jc, rtol, atol, opts.max_rank(), &gT);
      STRUMPACK_ID_FLOPS(ID_row_flops(w.Sr, w.Jr.size()));
      STRUMPACK_ID_FLOPS(ID_row_flops(w.Sc, w.Jc.size()));
      notify_inactives_J(w);
      if (d-opts.p() >= opts.max_rank() ||
          (int(w.Jr.size()) <= d - opts.p() &&
           int(w.Jc.size()) <= d - opts.p())) {
        this->U_rank_ = w.Jr.size();  this->U_rows_ = w.Sr.rows();
        this->V_rank_ = w.Jc.size();  this->V_rows_ = w.Sc.rows();
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
      TIMER_TIME(TaskType::REDUCE_SAMPLES, 1, t_reduce);
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
          w.Rr = DistM_t(grid(), this->V_rank(), c_new);
          w.Rc = DistM_t(grid(), this->U_rank(), c_new);
          V_.applyC(*tmpR, w.Rr);
          U_.applyC(*tmpR, w.Rc);
          STRUMPACK_REDUCE_SAMPLE_FLOPS
            (V_.applyC_flops(c_new) + U_.applyC_flops(c_new));
        } else {
          auto tmpR = ConstDistributedMatrixWrapperPtr
            (this->rows(), dd, RS.leaf_R, 0, d_old);
          DistM_t wRr_new(grid(), w.Rr.rows(), dd);
          DistM_t wRc_new(grid(), w.Rc.rows(), dd);
          V_.applyC(*tmpR, wRr_new);
          U_.applyC(*tmpR, wRc_new);
          STRUMPACK_REDUCE_SAMPLE_FLOPS
            (V_.applyC_flops(dd) + U_.applyC_flops(dd));
          w.Rr.hconcat(wRr_new);
          w.Rc.hconcat(wRc_new);
        }
      } else {
        if (!c_old) {
          auto wRr = vconcat
            (w.c[0].dR, child(0)->V_rank(), child(1)->V_rank(),
             w.c[0].Rr, w.c[1].Rr, grid(), grid()->ctxt_all());
          auto wRc = vconcat
            (w.c[0].dR, child(0)->U_rank(), child(1)->U_rank(),
             w.c[0].Rc, w.c[1].Rc, grid(), grid()->ctxt_all());
          w.Rr = DistM_t(grid(), this->V_rank(), w.c[0].dR);
          w.Rc = DistM_t(grid(), this->U_rank(), w.c[0].dR);
          V_.applyC(wRr, w.Rr);
          U_.applyC(wRc, w.Rc);
          STRUMPACK_REDUCE_SAMPLE_FLOPS
            (V_.applyC_flops(w.c[0].dR) + U_.applyC_flops(w.c[0].dR));
        } else {
          DistM_t wRr(grid(), this->V_rows(), dd);
          copy(w.c[0].Ic.size(), dd, w.c[0].Rr, 0, w.c[0].dR - dd, wRr, 0, 0, grid()->ctxt_all());
          copy(w.c[1].Ic.size(), dd, w.c[1].Rr, 0, w.c[1].dR - dd,
               wRr, w.c[0].Ic.size(), 0, grid()->ctxt_all());
          DistM_t wRc(grid(), this->U_rows(), dd);
          copy(w.c[0].Ir.size(), dd, w.c[0].Rc, 0, w.c[0].dR - dd, wRc, 0, 0, grid()->ctxt_all());
          copy(w.c[1].Ir.size(), dd, w.c[1].Rc, 0, w.c[1].dR - dd,
               wRc, w.c[0].Ir.size(), 0, grid()->ctxt_all());

          DistM_t wRr_new(grid(), w.Rr.rows(), dd);
          DistM_t wRc_new(grid(), w.Rc.rows(), dd);
          V_.applyC(wRr, wRr_new);
          U_.applyC(wRc, wRc_new);
          STRUMPACK_REDUCE_SAMPLE_FLOPS
            (V_.applyC_flops(dd) + U_.applyC_flops(dd));
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
