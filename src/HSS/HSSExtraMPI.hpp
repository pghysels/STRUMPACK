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
#ifndef HSS_EXTRA_MPI_HPP
#define HSS_EXTRA_MPI_HPP

#include "HSSExtra.hpp"
#include "dense/DistributedMatrix.hpp"

namespace strumpack {
  namespace HSS {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t>
    class WorkCompressMPI : public WorkCompressBase<scalar_t> {
    public:
      std::vector<WorkCompressMPI<scalar_t>> c;
      DistributedMatrix<scalar_t> Rr, Rc, Sr, Sc;
      DistributedMatrix<scalar_t> Qr, Qc;
      int dR = 0, dS = 0;
      std::unique_ptr<WorkCompress<scalar_t>> w_seq;
      void split(const std::pair<std::size_t,std::size_t>& dim) {
        if (c.empty()) {
          c.resize(2);
          c[0].offset = this->offset;
          c[1].offset = this->offset + dim;
          c[0].lvl = c[1].lvl = this->lvl + 1;
        }
      }
      void create_sequential() {
        if (!w_seq)
          w_seq = std::unique_ptr<WorkCompress<scalar_t>>
            (new WorkCompress<scalar_t>());
        w_seq->lvl = this->lvl;
      }
    };
#endif //DOXYGEN_SHOULD_SKIP_THIS

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t,
             typename real_t=typename RealType<scalar_t>::value_type>
    class WorkCompressMPIANN : public WorkCompressBase<scalar_t> {
    public:
      std::vector<WorkCompressMPIANN<scalar_t>> c;
      DistributedMatrix<scalar_t> S;
      std::vector<std::pair<std::size_t,real_t>> ids_scores;
      std::unique_ptr<WorkCompressANN<scalar_t>> w_seq;

      void split(const std::pair<std::size_t,std::size_t>& dim) {
        if (c.empty()) {
          c.resize(2);
          c[0].offset = this->offset;
          c[1].offset = this->offset + dim;
          c[0].lvl = c[1].lvl = this->lvl + 1;
        }
      }
      void create_sequential() {
        if (!w_seq)
          w_seq = std::unique_ptr<WorkCompressANN<scalar_t>>
            (new WorkCompressANN<scalar_t>());
        w_seq->lvl = this->lvl;
      }
    };
#endif //DOXYGEN_SHOULD_SKIP_THIS

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t> class WorkApplyMPI {
    public:
      std::pair<std::size_t,std::size_t> offset;
      std::vector<WorkApplyMPI<scalar_t>> c;
      DistributedMatrix<scalar_t> tmp1, tmp2;
      std::unique_ptr<WorkApply<scalar_t>> w_seq;
    };
#endif //DOXYGEN_SHOULD_SKIP_THIS


    template<typename scalar_t> class HSSMatrixBase;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t> class DistSubLeaf {
    public:
      DistSubLeaf
      (int cols, const HSSMatrixBase<scalar_t>* H, const BLACSGrid* lg)
        : cols_(cols), hss_(H), grid_loc_(lg) { allocate_block_row(); }
      /** dist should be on the context of H */
      DistSubLeaf
      (int cols, const HSSMatrixBase<scalar_t>* H, const BLACSGrid* lg,
       const DistributedMatrix<scalar_t>& dist)
        : cols_(cols), hss_(H), grid_loc_(lg) { to_block_row(dist); }
      void from_block_row(DistributedMatrix<scalar_t>& dist) const
      { hss_->from_block_row(dist, sub, leaf, grid_loc_); }
      DistributedMatrix<scalar_t> leaf;
      DenseMatrix<scalar_t> sub;
      const BLACSGrid* grid_local() const { return grid_loc_; }
      int cols() const { return cols_; }

    private:
      void allocate_block_row()
      { hss_->allocate_block_row(cols_, sub, leaf); }
      void to_block_row(const DistributedMatrix<scalar_t>& dist)
      { hss_->to_block_row(dist, sub, leaf); }

      const int cols_;
      const HSSMatrixBase<scalar_t>* hss_;
      const BLACSGrid* grid_loc_;
    };
#endif //DOXYGEN_SHOULD_SKIP_THIS


    class TreeLocalRanges {
    public:
      TreeLocalRanges() {}
      TreeLocalRanges(int P) : ranges_(5*P) {}
      void print() const {
        std::cout << "ranges=[";
        for (std::size_t p=0; p<ranges_.size()/5; p++)
          std::cout << rlo(p) << "," << rhi(p) << "/"
                    << clo(p) << "," << chi(p) << "/" << leaf_procs(p) << " ";
        std::cout << "];" << std::endl;
      }
      int rlo(int p) const { return ranges_[5*p+0]; }
      int rhi(int p) const { return ranges_[5*p+1]; }
      int clo(int p) const { return ranges_[5*p+2]; }
      int chi(int p) const { return ranges_[5*p+3]; }
      int leaf_procs(int p) const { return ranges_[5*p+4]; }
      int& rlo(int p) { return ranges_[5*p+0]; }
      int& rhi(int p) { return ranges_[5*p+1]; }
      int& clo(int p) { return ranges_[5*p+2]; }
      int& chi(int p) { return ranges_[5*p+3]; }
      int& leaf_procs(int p) { return ranges_[5*p+4]; }
    private:
      std::vector<int> ranges_; // rlo, rhi, clo, chi, leaf_procs
    };

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t> class WorkFactorMPI {
    public:
      std::vector<WorkFactorMPI<scalar_t>> c;

      // (U.cols x U.cols) \tilde(D)
      DistributedMatrix<scalar_t> Dt;

      // (U.cols x V.cols) bottom part of \tilde{V}
      DistributedMatrix<scalar_t> Vt1;
      std::unique_ptr<WorkFactor<scalar_t>> w_seq;
    };
#endif //DOXYGEN_SHOULD_SKIP_THIS


    /**
     * \class HSSFactorsMPI
     * \brief Contains data related to ULV factorization of a
     * distributed HSS matrix.
     *
     * Class containing data regarding the ULV factorization of an
     * HSSMatrixMPI This is constructed inside the ULV factorization
     * routine and should be passed to the HSS solve routine (along
     * with the original HSS matrix).
     */
    template<typename scalar_t> class HSSFactorsMPI {
      template<typename T> friend class HSSMatrixMPI;
      template<typename T> friend class HSSMatrixBase;

    public:

      /**
       * Get the amount of memory __(per rank)__ used to store this
       * data (excluding any metadata). To get the memory for the
       * entire factorization, you should also count the memory of the
       * original HSS matrix, as that is still required to perform a
       * solve.
       */
      std::size_t memory() {
        return sizeof(*this) + L_.memory() + Vt0_.memory()
          + W1_.memory() + Q_.memory() + D_.memory()
          + sizeof(int)*piv_.size();
      }

      /**
       * Get the number of nonzeros __(per rank)__ in this data. To
       * get the total number of nonzeros for the entire
       * factorization, you should also count the nonzeros of the
       * original HSS matrix, as that is still required to perform a
       * solve.
       */
      std::size_t nonzeros() const {
        return L_.nonzeros() + Vt0_.nonzeros() + W1_.nonzeros()
          + Q_.nonzeros() + D_.nonzeros();
      }

      /**
       * Used in the sparse solver to construct the Schur complement.
       */
      const DistributedMatrix<scalar_t>& Vhat() const { return Vt0_; }

      /**
       * Used in the sparse solver to construct the Schur complement.
       */
      DistributedMatrix<scalar_t>& Vhat() { return Vt0_; }

    private:
      // (U.rows-U.cols x U.rows-U.cols), empty at the root
      DistributedMatrix<scalar_t> L_;

      // (U.rows-U.cols x V.cols)
      // at the root, Vt0_ stored Vhat
      DistributedMatrix<scalar_t> Vt0_;

      // (U.cols x U.rows) bottom part of W
      // if (U.rows == U.cols) then W == I and is not stored!
      DistributedMatrix<scalar_t> W1_;

      // (U.rows x U.rows) Q from LQ(W0)
      // if (U.rows == U.cols) then Q == I and is not stored!
      DistributedMatrix<scalar_t> Q_;

      // (U.rows x U.rows) at the root holds LU(D), else empty
      DistributedMatrix<scalar_t> D_;
      std::vector<int> piv_;     // hold permutation from LU(D) at root
    };


#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t> class WorkSolveMPI {
    public:
      std::vector<WorkSolveMPI<scalar_t>> c;
      std::unique_ptr<WorkSolve<scalar_t>> w_seq;

      // do we need all these?? x only used in bwd, y only used in fwd??
      DistributedMatrix<scalar_t> z;
      DistributedMatrix<scalar_t> ft1;  // TODO document the sizes here
      DistributedMatrix<scalar_t> y;
      DistributedMatrix<scalar_t> x;

      // DO NOT STORE reduced_rhs here!!!
      DistributedMatrix<scalar_t> reduced_rhs;
      std::pair<std::size_t,std::size_t> offset;
    };
#endif //DOXYGEN_SHOULD_SKIP_THIS


#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t> class WorkExtractMPI {
    public:
      std::vector<WorkExtractMPI<scalar_t>> c;
      DistributedMatrix<scalar_t> y, z;
      std::vector<std::size_t> I, J, rl2g, cl2g, ycols, zcols;
      std::unique_ptr<WorkExtract<scalar_t>> w_seq;
      void split_extraction_sets
      (const std::pair<std::size_t,std::size_t>& dim) {
        if (c.empty()) {
          c.resize(2);
          c[0].I.reserve(I.size());
          c[1].I.reserve(I.size());
          for (auto i : I)
            if (i < dim.first) c[0].I.push_back(i);
            else c[1].I.push_back(i - dim.first);
          c[0].J.reserve(J.size());
          c[1].J.reserve(J.size());
          for (auto j : J)
            if (j < dim.second) c[0].J.push_back(j);
            else c[1].J.push_back(j - dim.second);
        }
      }
      void communicate_child_ycols(const MPIComm& comm, int rch1) {
        // TODO optimize these 4 bcasts?
        auto rch0 = 0;
        auto c0ycols = c[0].ycols.size();
        auto c1ycols = c[1].ycols.size();
        comm.broadcast_from(c0ycols, rch0);
        comm.broadcast_from(c1ycols, rch1);
        c[0].ycols.resize(c0ycols);
        c[1].ycols.resize(c1ycols);
        comm.broadcast_from(c[0].ycols, rch0);
        comm.broadcast_from(c[1].ycols, rch1);
      }
      void combine_child_ycols() {
        auto c0ycols = c[0].ycols.size();
        auto c1ycols = c[1].ycols.size();
        ycols.resize(c0ycols + c1ycols);
        std::copy(c[0].ycols.begin(), c[0].ycols.end(), ycols.begin());
        std::copy(c[1].ycols.begin(), c[1].ycols.end(),
                  ycols.begin()+c0ycols);
      }
    };
#endif //DOXYGEN_SHOULD_SKIP_THIS


#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t> class WorkExtractBlocksMPI {
    public:
      WorkExtractBlocksMPI(std::size_t nb) {
        y.resize(nb);
        z.resize(nb);
        I.resize(nb);
        J.resize(nb);
        rl2g.resize(nb);
        cl2g.resize(nb);
        ycols.resize(nb);
        zcols.resize(nb);
      }
      std::vector<WorkExtractBlocksMPI<scalar_t>> c;
      std::vector<DistributedMatrix<scalar_t>> y, z;
      std::vector<std::vector<std::size_t>> I, J, rl2g, cl2g, ycols, zcols;
      std::vector<std::unique_ptr<WorkExtract<scalar_t>>> w_seq;

      void split_extraction_sets
      (const std::pair<std::size_t,std::size_t>& dim) {
        if (c.empty()) {
          auto nb = I.size();
          c.reserve(2);
          c.emplace_back(nb);
          c.emplace_back(nb);
          for (std::size_t k=0; k<nb; k++) {
            c[0].I[k].reserve(I[k].size());
            c[1].I[k].reserve(I[k].size());
            for (auto i : I[k])
              if (i < dim.first) c[0].I[k].push_back(i);
              else c[1].I[k].push_back(i - dim.first);
            c[0].J[k].reserve(J[k].size());
            c[1].J[k].reserve(J[k].size());
            for (auto j : J[k])
              if (j < dim.second) c[0].J[k].push_back(j);
              else c[1].J[k].push_back(j - dim.second);
          }
        }
      }

      void communicate_child_ycols(const MPIComm& comm, int rch1) {
        int rank = comm.rank(), P = comm.size();
        int P0total = rch1, P1total = P - rch1;
        int pcols = std::floor(std::sqrt((float)P0total));
        int prows = P0total / pcols;
        int P0active = prows * pcols;
        pcols = std::floor(std::sqrt((float)P1total));
        prows = P1total / pcols;
        int P1active = prows * pcols;
        std::vector<MPI_Request> sreq;
        sreq.reserve(P);
        int sreqs = 0;
        std::vector<std::size_t> sbuf0, sbuf1;
        if (rank < P0active) {
          if (rank < (P-P0active)) {
            // I'm one of the first P-P0active processes that are active
            // on child0, so I need to send to one or more others which
            // are not active on child0, ie the ones in [P0active,P)
            std::size_t ssize = 0;
            for (std::size_t k=0; k<I.size(); k++)
              ssize += 1 + c[0].ycols[k].size();
            sbuf0.reserve(ssize);
            for (std::size_t k=0; k<I.size(); k++) {
              sbuf0.push_back(c[0].ycols[k].size());
              for (auto i : c[0].ycols[k])
                sbuf0.push_back(i);
            }
            for (int p=P0active; p<P; p++)
              if (rank == (p - P0active) % P0active) {
                sreq.emplace_back();
                comm.isend(sbuf0, p, 0, &sreq.back());
              }
          }
        }
        if (rank >= rch1 && rank < rch1+P1active) {
          if ((rank-rch1) < (P-P1active)) {
            // I'm one of the first P-P1active processes that are active
            // on child1, so I need to send to one or more others which
            // are not active on child1, ie the ones in [0,rch1) union
            // [rch1+P1active,P)
            std::size_t ssize = 0;
            for (std::size_t k=0; k<I.size(); k++)
              ssize += 1 + c[1].ycols[k].size();
            sbuf1.reserve(ssize);
            for (std::size_t k=0; k<I.size(); k++) {
              sbuf1.push_back(c[1].ycols[k].size());
              for (auto i : c[1].ycols[k])
                sbuf1.push_back(i);
            }
            for (int p=0; p<rch1; p++)
              if (rank - rch1 == p % P1active) {
                sreq.emplace_back();
                comm.isend(sbuf1, p, 1, &sreq[sreqs++]);
              }
            for (int p=rch1+P1active; p<P; p++)
              if (rank - rch1 == (p - P1active) % P1active) {
                sreq.emplace_back();
                comm.isend(sbuf1, p, 1, &sreq.back());
              }
          }
        }

        if (rank >= P0active) {
          // I'm not active on child0, so I need to receive
          int dest = -1;
          for (int p=0; p<P0active; p++)
            if (p == (rank - P0active) % P0active) { dest = p; break; }
          assert(dest >= 0);
          auto buf = comm.recv<std::size_t>(dest, 0);
          auto ptr = buf.data();
          for (std::size_t k=0; k<I.size(); k++) {
            auto c0ycols = *ptr++;
            c[0].ycols[k].resize(c0ycols);
            for (std::size_t i=0; i<c0ycols; i++)
              c[0].ycols[k][i] = *ptr++;
          }
        }
        if (!(rank >= rch1 && rank < rch1+P1active)) {
          // I'm not active on child1, so I need to receive
          int dest = -1;
          for (int p=rch1; p<rch1+P1active; p++) {
            if (rank < rch1) {
              if (p - rch1 == rank % P1active) { dest = p; break; }
            } else if (p - rch1 == (rank - P1active) % P1active) {
              dest = p; break;
            }
          }
          assert(dest >= 0);
          auto buf = comm.recv<std::size_t>(dest, 1);
          auto ptr = buf.data();
          for (std::size_t k=0; k<I.size(); k++) {
            auto c1ycols = *ptr++;
            c[1].ycols[k].resize(c1ycols);
            for (std::size_t i=0; i<c1ycols; i++)
              c[1].ycols[k][i] = *ptr++;
          }
        }
        wait_all(sreq);
      }

      void combine_child_ycols(const std::vector<bool>& odiag) {
        for (std::size_t k=0; k<I.size(); k++) {
          if (!odiag[k]) {
            ycols[k].clear();
            continue;
          }
          auto c0ycols = c[0].ycols[k].size();
          auto c1ycols = c[1].ycols[k].size();
          ycols[k].resize(c0ycols + c1ycols);
          std::copy
            (c[0].ycols[k].begin(), c[0].ycols[k].end(), ycols[k].begin());
          std::copy
            (c[1].ycols[k].begin(), c[1].ycols[k].end(),
             ycols[k].begin()+c0ycols);
        }
      }
    };
#endif //DOXYGEN_SHOULD_SKIP_THIS

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_EXTRA_MPI_HPP
