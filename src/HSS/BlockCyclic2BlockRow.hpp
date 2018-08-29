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
#ifndef BLOCK_CYCLIC_2_BLOCK_ROW_HPP
#define BLOCK_CYCLIC_2_BLOCK_ROW_HPP

namespace strumpack {
  namespace HSS {
    namespace BC2BR {

      template<typename scalar_t> void block_cyclic_to_block_row
      (const TreeLocalRanges& ranges, const DistributedMatrix<scalar_t>& dist,
       DenseMatrix<scalar_t>& sub, DistributedMatrix<scalar_t>& leaf,
       const BLACSGrid* lg, const MPIComm& comm) {
        assert(dist.fixed());
        const auto P = comm.size();
        const auto rank = comm.rank();
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
              leaf = DistributedMatrix<scalar_t>(lg, m, d);
            if (dist.active()) {
              int leaf_prows, leaf_pcols;
              BLACSGrid::layout(leaf_procs, leaf_prows, leaf_pcols);
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
              int leaf_prows, leaf_pcols;
              BLACSGrid::layout(leaf_procs, leaf_prows, leaf_pcols);
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
        std::vector<Triplet<scalar_t>> rbuf;
        std::vector<Triplet<scalar_t>*> pbuf;
        comm.all_to_all_v(sbuf, rbuf, pbuf, triplet_type);
        MPI_Type_free(&triplet_type);
        if (ranges.leaf_procs(rank) == 1) {
          assert((ranges.chi(rank) - ranges.clo(rank)) == int(sub.rows()));
          assert(int(sub.cols()) == dist.cols());
          for (auto& t : rbuf)
            sub(t._r, t._c) = t._v;
        } else if (leaf.active()) {
          const auto rows = leaf.rows();
          const auto cols = leaf.cols();
          auto lr = new int[rows+cols];
          auto lc = lr + rows;
          std::fill(lr, lr+rows+cols, -1);
          for (auto& t : rbuf) {
            int locr = lr[t._r];
            if (locr == -1) locr = lr[t._r] = leaf.rowg2l_fixed(t._r);
            int locc = lc[t._c];
            if (locc == -1) locc = lc[t._c] = leaf.colg2l_fixed(t._c);
            leaf(locr, locc) = t._v;
          }
          delete[] lr;
        }
      }

      template<typename scalar_t> void block_row_to_block_cyclic
      (const TreeLocalRanges& ranges, DistributedMatrix<scalar_t>& dist,
       const DenseMatrix<scalar_t>& sub,
       const DistributedMatrix<scalar_t>& leaf, const MPIComm& comm) {
        assert(dist.fixed());
        const auto P = comm.size();
        const auto rank = comm.rank();
        const int MB = DistributedMatrix<scalar_t>::default_MB;
        const int dist_pcols = dist.grid()->npcols();
        const int dist_prows = dist.grid()->nprows();
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
        std::vector<Triplet<scalar_t>> rbuf;
        std::vector<Triplet<scalar_t>*> pbuf;
        comm.all_to_all_v(sbuf, rbuf, pbuf, triplet_type);
        MPI_Type_free(&triplet_type);
        if (dist.active()) {
          const auto rows = dist.rows();
          const auto cols = dist.cols();
          auto lr = new int[rows+cols];
          auto lc = lr + rows;
          std::fill(lr, lr+rows+cols, -1);
          for (auto& t : rbuf) {
            int locr = lr[t._r];
            if (locr == -1) locr = lr[t._r] = dist.rowg2l_fixed(t._r);
            int locc = lc[t._c];
            if (locc == -1) locc = lc[t._c] = dist.colg2l_fixed(t._c);
            dist(locr, locc) = t._v;
          }
          delete[] lr;
        }
      }

    } //end namespace BC2BR
  } // end namespace HSS
} //end namespace strumpack

#endif // BLOCK_CYCLIC_2_BLOCK_ROW_HPP
