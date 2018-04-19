#ifndef BLOCK_CYCLIC_2_BLOCK_ROW_HPP
#define BLOCK_CYCLIC_2_BLOCK_ROW_HPP

namespace strumpack {
  namespace HSS {
    namespace BC2BR {

      template<typename scalar_t> void block_cyclic_to_block_row
      (const TreeLocalRanges& ranges, const DistributedMatrix<scalar_t>& dist,
       DenseMatrix<scalar_t>& sub, DistributedMatrix<scalar_t>& leaf,
       int lctxt, MPI_Comm comm) {
        assert(dist.fixed());
        const auto P = mpi_nprocs(comm);
        // TODO if P == 1 create 1 leaf???
        const auto rank = mpi_rank(comm);
        const int MB = DistributedMatrix<scalar_t>::default_MB;
        const auto d = dist.cols();
        int maxr = 0;
        for (int p=0; p<P; p++) {
          const DistributedMatrixWrapper<scalar_t> pdist
            (ranges.chi(p) - ranges.clo(p), d,
             const_cast<DistributedMatrix<scalar_t>&>(dist),
             ranges.clo(p) - ranges.clo(0), 0);
          int rlo, rhi, clo, chi;
          pdist.lranges(rlo, rhi, clo, chi);
          maxr = std::max(maxr, rhi - rlo);
          p += ranges.leaf_procs(p) - 1;
        }
        auto destr = new std::size_t[2*maxr+P];
        auto gr = destr + maxr;
        auto ssize = gr + maxr;
        std::fill(ssize, ssize+P, 0);
        for (int p=0; p<P; p++) {
          const auto m = ranges.chi(p) - ranges.clo(p);
          assert(m >= 0);
          const auto leaf_procs = ranges.leaf_procs(p);
          const auto rbegin = ranges.clo(p) - ranges.clo(0);
          const DistributedMatrixWrapper<scalar_t> pdist
            (m, d, const_cast<DistributedMatrix<scalar_t>&>(dist), rbegin, 0);
          int rlo, rhi, clo, chi;
          pdist.lranges(rlo, rhi, clo, chi);
          if (leaf_procs == 1) {
            if (p == rank) {
              sub = DenseMatrix<scalar_t>(m, d);
              if (dist.active()) {
                for (int r=rlo; r<rhi; r++)
                  destr[r-rlo] = dist.rowl2g_fixed(r) - rbegin;
                for (int c=clo; c<chi; c++)
                  for (int r=rlo, gc=dist.coll2g_fixed(c); r<rhi; r++)
                    sub(destr[r-rlo], gc) = dist(r,c);
              }
            } else if (dist.active())
              ssize[p] += (chi-clo) * (rhi-rlo);
          } else {
            if (p <= rank && rank < p+leaf_procs)
              leaf = DistributedMatrix<scalar_t>(lctxt, m, d);
            if (dist.active()) {
              const int leaf_pcols = std::floor(std::sqrt((float)leaf_procs));
              const int leaf_prows = leaf_procs / leaf_pcols;
              for (int r=rlo; r<rhi; r++)
                destr[r-rlo] = p
                  + (((dist.rowl2g_fixed(r) - rbegin) / MB) % leaf_prows);
              for (int c=clo; c<chi; c++) {
                const auto destc =
                  (((dist.coll2g_fixed(c)) / MB) % leaf_pcols) * leaf_prows;
                for (int r=rlo; r<rhi; r++)
                  ssize[destr[r-rlo]+destc]++;
              }
            }
            p += leaf_procs - 1;
          }
        }
        std::vector<std::vector<Triplet<scalar_t>>> sbuf(P);
        for (int p=0; p<P; p++)
          sbuf[p].reserve(ssize[p]);
        for (int p=0; p<P; p++) {
          const auto m = ranges.chi(p) - ranges.clo(p);
          const auto leaf_procs = ranges.leaf_procs(p);
          const auto rbegin = ranges.clo(p) - ranges.clo(0);
          const DistributedMatrixWrapper<scalar_t> pdist
            (m, d, const_cast<DistributedMatrix<scalar_t>&>(dist), rbegin, 0);
          int rlo, rhi, clo, chi;
          pdist.lranges(rlo, rhi, clo, chi);
          if (leaf_procs == 1) {
            if (p != rank && dist.active()) {
              for (int r=rlo; r<rhi; r++) {
                gr[r-rlo] = dist.rowl2g_fixed(r) - rbegin;
                assert(int(gr[r-rlo]) < m);
              }
              for (int c=clo; c<chi; c++)
                for (int r=rlo, gc=dist.coll2g_fixed(c); r<rhi; r++)
                  sbuf[p].emplace_back(gr[r-rlo], gc, dist(r,c));
            }
          } else {
            if (dist.active()) {
              const int leaf_pcols = std::floor(std::sqrt((float)leaf_procs));
              const int leaf_prows = leaf_procs / leaf_pcols;
              for (int r=rlo; r<rhi; r++) {
                gr[r-rlo] = dist.rowl2g_fixed(r) - rbegin;
                destr[r-rlo] = p + ((gr[r-rlo] / MB) % leaf_prows);
              }
              for (int c=clo; c<chi; c++) {
                const auto gc = dist.coll2g_fixed(c);
                const auto destc = ((gc / MB) % leaf_pcols) * leaf_prows;
                for (int r=rlo; r<rhi; r++)
                  sbuf[destr[r-rlo]+destc].emplace_back
                    (gr[r-rlo], gc, dist(r,c));
              }
            }
            p += leaf_procs - 1;
          }
        }
        for (int p=0; p<P; p++) { assert(ssize[p] == sbuf[p].size()); }
        delete[] destr;
        MPI_Datatype triplet_type;
        create_triplet_mpi_type<scalar_t>(&triplet_type);
        Triplet<scalar_t>* rbuf = nullptr;
        std::size_t totrsize;
        all_to_all_v(sbuf, rbuf, totrsize, comm, triplet_type);
        MPI_Type_free(&triplet_type);
        if (ranges.leaf_procs(rank) == 1) {
          assert((ranges.chi(rank) - ranges.clo(rank)) == int(sub.rows()));
          assert(int(sub.cols()) == dist.cols());
          for (auto t=rbuf; t!=rbuf+totrsize; t++)
            sub(t->_r, t->_c) = t->_v;
        } else if (leaf.active()) {
          const auto rows = leaf.rows();
          const auto cols = leaf.cols();
          auto lr = new int[rows+cols];
          auto lc = lr + rows;
          std::fill(lr, lr+rows+cols, -1);
          for (auto t=rbuf; t!=rbuf+totrsize; t++) {
            int locr = lr[t->_r];
            if (locr == -1) locr = lr[t->_r] = leaf.rowg2l_fixed(t->_r);
            int locc = lc[t->_c];
            if (locc == -1) locc = lc[t->_c] = leaf.colg2l_fixed(t->_c);
            leaf(locr, locc) = t->_v;
          }
          delete lr;
        }
        delete[] rbuf;
      }

      template<typename scalar_t> void block_row_to_block_cyclic
      (const TreeLocalRanges& ranges, DistributedMatrix<scalar_t>& dist,
       const DenseMatrix<scalar_t>& sub,
       const DistributedMatrix<scalar_t>& leaf, MPI_Comm comm) {
        assert(dist.fixed());
        const auto P = mpi_nprocs(comm);
        const auto rank = mpi_rank(comm);
        const int MB = DistributedMatrix<scalar_t>::default_MB;
        const int dist_pcols = std::floor(std::sqrt((float)P));
        const int dist_prows = P / dist_pcols;
        assert(!dist.active() || (dist_pcols == dist.pcols()));
        assert(!dist.active() || (dist_prows == dist.prows()));
        const auto leaf_procs = ranges.leaf_procs(rank);
        auto rbegin = ranges.clo(rank) - ranges.clo(0);
        std::vector<std::vector<Triplet<scalar_t>>> sbuf(P);
        if (leaf_procs == 1) { // sub
          const auto rows = sub.rows();
          const auto cols = sub.cols();
          auto destr = new std::size_t[rows+cols+P];
          auto destc = destr + rows;
          auto ssize = destc + cols;
          std::fill(ssize, ssize+P, 0);
          for (std::size_t r=0; r<rows; r++)
            destr[r] = (((r + rbegin) / MB) % dist_prows);
          for (std::size_t c=0; c<cols; c++)
            destc[c] = ((c / MB) % dist_pcols) * dist_prows;
          for (std::size_t c=0; c<cols; c++)
            for (std::size_t r=0; r<rows; r++)
              ssize[destr[r]+destc[c]]++;
          for (int p=0; p<P; p++)
            sbuf[p].reserve(ssize[p]);
          for (std::size_t c=0; c<cols; c++)
            for (std::size_t r=0; r<rows; r++)
              sbuf[destr[r]+destc[c]].emplace_back(r + rbegin, c, sub(r, c));
          for (int p=0; p<P; p++) { assert(sbuf[p].size() == ssize[p]); }
          delete[] destr;
        } else { // leaf
          if (leaf.active()) {
            const auto lcols = leaf.lcols();
            const auto lrows = leaf.lrows();
            auto destr = new std::size_t[2*lrows+2*lcols+P];
            auto gr    = destr + lrows;
            auto destc = gr + lrows;
            auto gc    = destc + lcols;
            auto ssize = gc + lcols;
            std::fill(ssize, ssize+P, 0);
            for (int r=0; r<lrows; r++) {
              gr[r] = leaf.rowl2g_fixed(r) + rbegin;
              destr[r] = ((gr[r] / MB) % dist_prows);
            }
            for (int c=0; c<lcols; c++) {
              gc[c] = leaf.coll2g_fixed(c);
              destc[c] = ((gc[c] / MB) % dist_pcols) * dist_prows;
            }
            for (int c=0; c<lcols; c++)
              for (int r=0; r<lrows; r++)
                ssize[destr[r]+destc[c]]++;
            for (int p=0; p<P; p++)
              sbuf[p].reserve(ssize[p]);
            for (int c=0; c<lcols; c++)
              for (int r=0; r<lrows; r++)
                sbuf[destr[r]+destc[c]].emplace_back
                  (gr[r], gc[c], leaf(r, c));
            for (int p=0; p<P; p++) { assert(sbuf[p].size() == ssize[p]); }
            delete[] destr;
          }
        }
        MPI_Datatype triplet_type;
        create_triplet_mpi_type<scalar_t>(&triplet_type);
        Triplet<scalar_t>* rbuf = nullptr;
        std::size_t totrsize;
        all_to_all_v(sbuf, rbuf, totrsize, comm, triplet_type);
        MPI_Type_free(&triplet_type);
        if (dist.active()) {
          const auto rows = dist.rows();
          const auto cols = dist.cols();
          auto lr = new int[rows+cols];
          auto lc = lr + rows;
          std::fill(lr, lr+rows+cols, -1);
          for (auto t=rbuf; t!=rbuf+totrsize; t++) {
            int locr = lr[t->_r];
            if (locr == -1) locr = lr[t->_r] = dist.rowg2l_fixed(t->_r);
            int locc = lc[t->_c];
            if (locc == -1) locc = lc[t->_c] = dist.colg2l_fixed(t->_c);
            dist(locr, locc) = t->_v;
          }
          delete[] lr;
        }
        delete[] rbuf;
      }

    } //end namespace BC2BR
  } // end namespace HSS
} //end namespace strumpack

#endif // BLOCK_CYCLIC_2_BLOCK_ROW_HPP
