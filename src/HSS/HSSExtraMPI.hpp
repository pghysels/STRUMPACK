#ifndef HSS_EXTRA_MPI_HPP
#define HSS_EXTRA_MPI_HPP

#include "HSSExtra.hpp"
#include "dense/DistributedMatrix.hpp"

namespace strumpack {
  namespace HSS {

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

    template<typename scalar_t> class WorkApplyMPI {
    public:
      std::pair<std::size_t,std::size_t> offset;
      std::vector<WorkApplyMPI<scalar_t>> c;
      DistributedMatrix<scalar_t> tmp1, tmp2;
      std::unique_ptr<WorkApply<scalar_t>> w_seq;
    };

    template<typename scalar_t> class HSSMatrixBase;

    template<typename scalar_t> class DistSubLeaf {
    public:
      DistSubLeaf(int cols, const HSSMatrixBase<scalar_t>* H, int ctxt_loc)
        : _cols(cols), _hss(H), _ctxt_loc(ctxt_loc) { allocate_block_row(); }
      /** dist should be on the context of H */
      DistSubLeaf(int cols, const HSSMatrixBase<scalar_t>* H, int ctxt_loc,
                  const DistributedMatrix<scalar_t>& dist)
        : _cols(cols), _hss(H), _ctxt_loc(ctxt_loc) { to_block_row(dist); }
      void from_block_row(DistributedMatrix<scalar_t>& dist) const
      { _hss->from_block_row(dist, sub, leaf, _ctxt_loc); }
      DistributedMatrix<scalar_t> leaf;
      DenseMatrix<scalar_t> sub;
      int ctxt_loc() const { return _ctxt_loc; }
      int cols() const { return _cols; }
    private:
      void allocate_block_row()
      { _hss->allocate_block_row(_cols, sub, leaf); }
      void to_block_row(const DistributedMatrix<scalar_t>& dist)
      { _hss->to_block_row(dist, sub, leaf); }
      const int _cols;
      const HSSMatrixBase<scalar_t>* _hss;
      const int _ctxt_loc;
    };

    class TreeLocalRanges {
    public:
      TreeLocalRanges() {}
      TreeLocalRanges(int P) : _ranges(5*P) {}
      void print() const {
        std::cout << "ranges=[";
        for (std::size_t p=0; p<_ranges.size()/5; p++)
          std::cout << rlo(p) << "," << rhi(p) << "/"
                    << clo(p) << "," << chi(p) << "/" << leaf_procs(p) << " ";
        std::cout << "];" << std::endl;
      }
      int rlo(int p) const { return _ranges[5*p+0]; }
      int rhi(int p) const { return _ranges[5*p+1]; }
      int clo(int p) const { return _ranges[5*p+2]; }
      int chi(int p) const { return _ranges[5*p+3]; }
      int leaf_procs(int p) const { return _ranges[5*p+4]; }
      int& rlo(int p) { return _ranges[5*p+0]; }
      int& rhi(int p) { return _ranges[5*p+1]; }
      int& clo(int p) { return _ranges[5*p+2]; }
      int& chi(int p) { return _ranges[5*p+3]; }
      int& leaf_procs(int p) { return _ranges[5*p+4]; }
    private:
      std::vector<int> _ranges; // rlo, rhi, clo, chi, leaf_procs
      int _ctxt = -1;
    };

    template<typename scalar_t> class WorkFactorMPI {
    public:
      std::vector<WorkFactorMPI<scalar_t>> c;

      // (U.cols x U.cols) \tilde(D)
      DistributedMatrix<scalar_t> Dt;

      // (U.cols x V.cols) bottom part of \tilde{V}
      DistributedMatrix<scalar_t> Vt1;
      std::unique_ptr<WorkFactor<scalar_t>> w_seq;
    };

    template<typename scalar_t> class HSSFactorsMPI {
      template<typename T> friend class HSSMatrixMPI;
      template<typename T> friend class HSSMatrixBase;
    public:
      const DistributedMatrix<scalar_t>& Vhat() const { return _Vt0; }
      DistributedMatrix<scalar_t>& Vhat() { return _Vt0; }
      std::size_t memory() {
        std::size_t mem = sizeof(*this) + _L.memory() + _Vt0.memory()
          + _W1.memory() + _Q.memory() + _D.memory()
          + sizeof(int)*_piv.size();
        for (auto& c : _ch) mem += c.memory();
        return memory;
      }
      std::size_t nonzeros() const {
        std::size_t nnz = _L.nonzeros() + _Vt0.nonzeros() + _W1.nonzeros()
          + _Q.nonzeros() + _D.nonzeros();
        for (auto& c : _ch) nnz += c.nonzeros();
        return nnz;
      }
    private:
      std::vector<HSSFactorsMPI<scalar_t>> _ch;
      std::unique_ptr<HSSFactors<scalar_t>> _factors_seq;

      // (U.rows-U.cols x U.rows-U.cols), empty at the root
      DistributedMatrix<scalar_t> _L;

      // (U.rows-U.cols x V.cols)
      // at the root, _Vt0 stored Vhat
      DistributedMatrix<scalar_t> _Vt0;

      // (U.cols x U.rows) bottom part of W
      // if (U.rows == U.cols) then W == I and is not stored!
      DistributedMatrix<scalar_t> _W1;

      // (U.rows x U.rows) Q from LQ(W0)
      // if (U.rows == U.cols) then Q == I and is not stored!
      DistributedMatrix<scalar_t> _Q;

      // (U.rows x U.rows) at the root holds LU(D), else empty
      DistributedMatrix<scalar_t> _D;
      std::vector<int> _piv;            // hold permutation from LU(D) at root
    };

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
      void communicate_child_ycols(MPI_Comm comm, int rch1) {
        // TODO optimize these 4 Bcasts!!
        auto rch0 = 0;
        std::size_t c0ycols = c[0].ycols.size();
        MPI_Bcast(&c0ycols, 1, mpi_type<std::size_t>(), rch0, comm);
        std::size_t c1ycols = c[1].ycols.size();
        MPI_Bcast(&c1ycols, 1, mpi_type<std::size_t>(), rch1, comm);
        c[0].ycols.resize(c0ycols);
        c[1].ycols.resize(c1ycols);
        MPI_Bcast(c[0].ycols.data(), c0ycols, mpi_type<std::size_t>(),
                  rch0, comm);
        MPI_Bcast(c[1].ycols.data(), c1ycols, mpi_type<std::size_t>(),
                  rch1, comm);
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

    template<typename scalar_t> void
    create_triplet_mpi_type(MPI_Datatype* triplet_type) {
      MPI_Type_contiguous(sizeof(Triplet<scalar_t>), MPI_BYTE, triplet_type);
      MPI_Type_commit(triplet_type);
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_EXTRA_MPI_HPP
