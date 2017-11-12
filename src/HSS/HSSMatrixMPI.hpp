#ifndef HSS_MATRIX_MPI_HPP
#define HSS_MATRIX_MPI_HPP

#include <cassert>

#include "misc/MPIWrapper.hpp"
#include "HSSExtraMPI.hpp"
#include "DistSamples.hpp"
#include "HSSMatrixBase.hpp"
#include "HSSMatrix.hpp"
#include "HSSPartitionTree.hpp"
#include "HSSBasisIDMPI.hpp"
#include "BlockCyclic2BlockRow.hpp"

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> class HSSMatrixBase;
    template<typename scalar_t> class HSSMatrix;

    template<typename scalar_t>
    class HSSMatrixMPI : public HSSMatrixBase<scalar_t> {
      using real_t = typename RealType<scalar_t>::value_type;
      using DistM_t = DistributedMatrix<scalar_t>;
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using delem_t = typename std::function
        <void(const std::vector<std::size_t>& I,
              const std::vector<std::size_t>& J, DistM_t& B)>;
      using elem_t = typename std::function
        <void(const std::vector<std::size_t>& I,
              const std::vector<std::size_t>& J, DenseM_t& B)>;
      using dmult_t = typename std::function
        <void(DistM_t& R, DistM_t& Sr, DistM_t& Sc)>;
      using opts_t = HSSOptions<scalar_t>;

    public:
      HSSMatrixMPI() : HSSMatrixBase<scalar_t>(0, 0, true) {}
      HSSMatrixMPI(const DistM_t& A, const opts_t& opts, MPI_Comm c);
      HSSMatrixMPI(const HSSPartitionTree& t, const opts_t& opts, MPI_Comm c);
      HSSMatrixMPI(const HSSPartitionTree& t, const DistM_t& A,
                   const opts_t& opts, MPI_Comm c);
      HSSMatrixMPI(std::size_t m, std::size_t n, const dmult_t& Amult,
                   int Actxt, const delem_t& Aelem,
                   const opts_t& opts, MPI_Comm c);
      HSSMatrixMPI(const HSSPartitionTree& t, const dmult_t& Amult, int Actxt,
                   const delem_t& Aelem,
                   const opts_t& opts, MPI_Comm c);
      ~HSSMatrixMPI();

      const HSSMatrixBase<scalar_t>* child(int c) const
      { return this->_ch[c].get(); }
      HSSMatrixBase<scalar_t>* child(int c) { return this->_ch[c].get(); }

      int ctxt() const { return _ctxt; }
      int ctxt(int) const { return _ctxt; }  // this overwrites base class
      int ctxt_loc() const { return _ctxt_loc; }
      int ctxt_all() const { return _ctxt_all; }
      MPI_Comm comm() const { return _comm; }

      void compress(const DistM_t& A, const opts_t& opts);
      void compress(const dmult_t& Amult, const delem_t& Aelem,
                    const opts_t& opts, int Actxt=-1);

      HSSFactorsMPI<scalar_t> factor() const;
      HSSFactorsMPI<scalar_t> partial_factor() const;
      void solve(const HSSFactorsMPI<scalar_t>& ULV, DistM_t& b) const;
      void forward_solve(const HSSFactorsMPI<scalar_t>& ULV,
                         WorkSolveMPI<scalar_t>& w,
                         const DistM_t& b, bool partial) const;
      void backward_solve(const HSSFactorsMPI<scalar_t>& ULV,
                          WorkSolveMPI<scalar_t>& w, DistM_t& x) const;

      DistM_t apply(const DistM_t& b) const;
      DistM_t applyC(const DistM_t& b) const;

      scalar_t get(std::size_t i, std::size_t j) const;
      DistM_t extract(const std::vector<std::size_t>& I,
                      const std::vector<std::size_t>& J,
                      int Bctxt, int Bprows, int Bpcols) const;
      void extract_add(const std::vector<std::size_t>& I,
                       const std::vector<std::size_t>& J,
                       DistM_t& B, int Bprows, int Bpcols) const;

      void Schur_update(const HSSFactorsMPI<scalar_t>& f, DistM_t& Theta,
                        DistM_t& Vhat, DistM_t& DUB01, DistM_t& Phi) const;
      void Schur_product_direct(const DistM_t& Theta, const DistM_t& Vhat,
                                const DistM_t& DUB01, const DistM_t& Phi,
                                const DistM_t&_ThetaVhatC,
                                const DistM_t& VhatCPhiC, const DistM_t& R,
                                DistM_t& Sr, DistM_t& Sc) const;

      std::size_t max_rank() const;        // collective on comm()
      std::size_t total_memory() const;    // collective on comm()
      std::size_t total_nonzeros() const;  // collective on comm()
      std::size_t max_levels() const;      // collective on comm()
      std::size_t rank() const;
      std::size_t memory() const;
      std::size_t nonzeros() const;
      std::size_t levels() const;

      void print_info(std::ostream &out=std::cout,
                      std::size_t roff=0, std::size_t coff=0) const;

      DistM_t dense(int ctxt) const;

      const TreeLocalRanges& tree_ranges() const { return _ranges; }
      void to_block_row(const DistM_t& A, DenseM_t& sub_A,
                        DistM_t& leaf_A) const;
      void allocate_block_row(int d, DenseM_t& sub_A, DistM_t& leaf_A) const;
      void from_block_row(DistM_t& A, const DenseM_t& sub_A,
                          const DistM_t& leaf_A, int lctxt) const;

    private:
      MPI_Comm _comm;
      int _ctxt, _ctxt_all, _ctxt_T, _ctxt_loc;   // TODO default arguments?
      int _prows, _pcols, _active_procs, _nprocs;
      TreeLocalRanges _ranges;

      HSSBasisIDMPI<scalar_t> _U;
      HSSBasisIDMPI<scalar_t> _V;
      DistM_t _D;
      DistM_t _B01;
      DistM_t _B10;

      HSSMatrixMPI(std::size_t m, std::size_t n, const opts_t& opts,
                   MPI_Comm c, int P, bool dup_comm,
                   std::size_t roff, std::size_t coff);
      HSSMatrixMPI(const HSSPartitionTree& t, const opts_t& opts, MPI_Comm c,
                   int P, bool dup_comm, std::size_t roff, std::size_t coff);
      void setup_hierarchy(const opts_t& opts, MPI_Comm c, int P,
                           bool dup_comm, std::size_t roff, std::size_t coff);
      void setup_hierarchy(const HSSPartitionTree& t, const opts_t& opts,
                           MPI_Comm c, int P, bool dup_comm,
                           std::size_t roff, std::size_t coff);
      void setup_contexts(int P);
      void setup_ranges(std::size_t roff, std::size_t coff);

      void compress_original_nosync(const dmult_t& Amult,
                                    const delem_t& Aelem, const opts_t& opts,
                                    int Actxt=-1);
      void compress_original_sync(const dmult_t& Amult, const delem_t& Aelem,
                                  const opts_t& opts, int Actxt=-1);
      void compress_stable_nosync(const dmult_t& Amult, const delem_t& Aelem,
                                  const opts_t& opts, int Actxt=-1);
      void compress_stable_sync(const dmult_t& Amult, const delem_t& Aelem,
                                const opts_t& opts, int Actxt=-1);

      void compress_recursive_original(DistSamples<scalar_t>& RS,
                                       const delem_t& Aelem,
                                       const opts_t& opts,
                                       WorkCompressMPI<scalar_t>& w, int dd);
      void compress_recursive_stable(DistSamples<scalar_t>& RS,
                                     const delem_t& Aelem, const opts_t& opts,
                                     WorkCompressMPI<scalar_t>& w,
                                     int d, int dd);
      void compute_local_samples(const DistSamples<scalar_t>& RS,
                                 WorkCompressMPI<scalar_t>& w, int dd);
      bool compute_U_V_bases(int d, const opts_t& opts,
                             WorkCompressMPI<scalar_t>& w);
      void compute_U_basis_stable(const opts_t& opts,
                                  WorkCompressMPI<scalar_t>& w,
                                  int d, int dd);
      void compute_V_basis_stable(const opts_t& opts,
                                  WorkCompressMPI<scalar_t>& w,
                                  int d, int dd);
      bool update_orthogonal_basis(const opts_t& opts, scalar_t& r_max_0,
                                   const DistM_t& S, DistM_t& Q,
                                   int d, int dd, bool untouched);
      void reduce_local_samples(const DistSamples<scalar_t>& RS,
                                WorkCompressMPI<scalar_t>& w, int dd,
                                bool was_compressed);
      void communicate_child_data(WorkCompressMPI<scalar_t>& w);
      void notify_inactives_J(WorkCompressMPI<scalar_t>& w);
      void notify_inactives_states(WorkCompressMPI<scalar_t>& w);

      void compress_level_original(DistSamples<scalar_t>& RS,
                                   const opts_t& opts,
                                   WorkCompressMPI<scalar_t>& w,
                                   int dd, int lvl);
      void compress_level_stable(DistSamples<scalar_t>& RS,
                                 const opts_t& opts,
                                 WorkCompressMPI<scalar_t>& w,
                                 int d, int dd, int lvl);
      void extract_level(const delem_t& Aelem, const opts_t& opts,
                         WorkCompressMPI<scalar_t>& w, int lvl);
      void get_extraction_indices(std::vector<std::vector<std::size_t>>& I,
                                  std::vector<std::vector<std::size_t>>& J,
                                  WorkCompressMPI<scalar_t>& w,
                                  int& self, int lvl);
      void allgather_extraction_indices
      (std::vector<std::vector<std::size_t>>& lI,
       std::vector<std::vector<std::size_t>>& lJ,
       std::vector<std::vector<std::size_t>>& I,
       std::vector<std::vector<std::size_t>>& J,
       int& before, int self, int& after);
      void extract_D_B(const delem_t& Aelem, int lctxt, const opts_t& opts,
                       WorkCompressMPI<scalar_t>& w, int lvl);

      void factor_recursive(HSSFactorsMPI<scalar_t>& f,
                            WorkFactorMPI<scalar_t>& w, int local_ctxt,
                            bool isroot, bool partial) const;

      void solve_fwd(const HSSFactorsMPI<scalar_t>& ULV,
                     const DistSubLeaf<scalar_t>& b,
                     WorkSolveMPI<scalar_t>& w, bool partial,
                     bool isroot) const;
      void solve_bwd(const HSSFactorsMPI<scalar_t>& ULV,
                     DistSubLeaf<scalar_t>& x, WorkSolveMPI<scalar_t>& w,
                     bool isroot) const;

      void apply_fwd(const DistSubLeaf<scalar_t>& B,
                     WorkApplyMPI<scalar_t>& w, bool isroot) const;
      void apply_bwd(const DistSubLeaf<scalar_t>& B, scalar_t beta,
                     DistSubLeaf<scalar_t>& C,
                     WorkApplyMPI<scalar_t>& w, bool isroot) const;
      void applyT_fwd(const DistSubLeaf<scalar_t>& B,
                      WorkApplyMPI<scalar_t>& w, bool isroot) const;
      void applyT_bwd(const DistSubLeaf<scalar_t>& B, scalar_t beta,
                      DistSubLeaf<scalar_t>& C, WorkApplyMPI<scalar_t>& w,
                      bool isroot) const;

      void extract_fwd(WorkExtractMPI<scalar_t>& w, int lctxt,
                       bool odiag) const;
      void extract_bwd(std::vector<Triplet<scalar_t>>& triplets,
                       int lctxt, WorkExtractMPI<scalar_t>& w) const;
      void triplets_to_DistM(std::vector<Triplet<scalar_t>>& triplets,
                             DistM_t& B, int Bprows, int Bpcols) const;

      void apply_UV_big(DistSubLeaf<scalar_t>& Theta, DistM_t& Uop,
                        DistSubLeaf<scalar_t>& Phi, DistM_t& Vop) const;

      static int Pl(std::size_t n, std::size_t nl, std::size_t nr, int P) {
        return std::max(1, std::min(int(std::round(float(P) * nl / n)), P-1));
      }
      static int Pr(std::size_t n, std::size_t nl, std::size_t nr, int P)
      { return std::max(1, P - Pl(n, nl, nr, P));  }
      int Pl(int P) const {
        return Pl(this->rows(), this->_ch[0]->rows(),
                  this->_ch[1]->rows(), P);
      }
      int Pr(int P) const {
        return Pr(this->rows(), this->_ch[0]->rows(),
                  this->_ch[1]->rows(), P);
      }

      template<typename T> friend
      void apply_HSS(Trans ta, const HSSMatrixMPI<T>& a,
                     const DistributedMatrix<T>& b, T beta,
                     DistributedMatrix<T>& c);
      friend class DistSamples<scalar_t>;
    };

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const DistM_t& A, const opts_t& opts, MPI_Comm c)
      : HSSMatrixBase<scalar_t>(A.rows(), A.cols(), true) {
      setup_hierarchy(opts, c, mpi_nprocs(c), true, 0, 0);
      setup_contexts(mpi_nprocs(c));
      setup_ranges(0, 0);
      compress(A, opts);
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const HSSPartitionTree& t, const opts_t& opts, MPI_Comm c)
      : HSSMatrixBase<scalar_t>(t.size, t.size, true) {
      setup_hierarchy(t, opts, c, mpi_nprocs(c), true, 0, 0);
      setup_contexts(mpi_nprocs(c));
      setup_ranges(0, 0);
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const HSSPartitionTree& t, const DistM_t& A, const opts_t& opts,
     MPI_Comm c) : HSSMatrixBase<scalar_t>(A.rows(), A.cols(), true) {
      assert(t.size == A.rows() && t.size == A.cols());
      setup_hierarchy(t, opts, c, mpi_nprocs(c), true, 0, 0);
      setup_contexts(mpi_nprocs(c));
      setup_ranges(0, 0);
      compress(A, opts);
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (std::size_t m, std::size_t n, const dmult_t& Amult, int Actxt,
     const delem_t& Aelem, const opts_t& opts, MPI_Comm c)
      : HSSMatrixBase<scalar_t>(m, n, true) {
      setup_hierarchy(opts, c, mpi_nprocs(c), true, 0, 0);
      setup_contexts(mpi_nprocs(c));
      setup_ranges(0, 0);
      compress(Amult, Aelem, opts, Actxt);
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const HSSPartitionTree& t, const dmult_t& Amult, int Actxt,
     const delem_t& Aelem, const opts_t& opts, MPI_Comm c)
      : HSSMatrixBase<scalar_t>(t.size, t.size, true) {
      setup_hierarchy(t, opts, c, mpi_nprocs(c), true, 0, 0);
      setup_contexts(mpi_nprocs(c));
      setup_ranges(0, 0);
      compress(Amult, Aelem, opts, Actxt);
    }

    /** private constructor */
    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (std::size_t m, std::size_t n, const opts_t& opts, MPI_Comm c, int P,
     bool dup_comm, std::size_t roff, std::size_t coff)
      : HSSMatrixBase<scalar_t>(m, n, true) {
      setup_hierarchy(opts, c, P, dup_comm, roff, coff);
      setup_contexts(P);
      setup_ranges(roff, coff);
    }

    /** private constructor */
    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const HSSPartitionTree& t, const opts_t& opts, MPI_Comm c, int P,
     bool dup_comm, std::size_t roff, std::size_t coff)
      : HSSMatrixBase<scalar_t>(t.size, t.size, true) {
      setup_hierarchy(t, opts, c, P, dup_comm, roff, coff);
      setup_contexts(P);
      setup_ranges(roff, coff);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::setup_contexts(int P) {
      // P is not always mpi_nprocs(_comm), _comm can be MPI_COMM_NULL
      _nprocs = P;
      _pcols = std::floor(std::sqrt((float)_nprocs));
      _prows = P / _pcols;
      _active_procs = _prows * _pcols;
      assert(_active_procs >= 1 && _active_procs <= _nprocs);
      if (_comm != MPI_COMM_NULL) {
        if (_active_procs < P) {
          auto active_comm = mpi_sub_comm(_comm, 0, _active_procs);
          if (mpi_rank(_comm) < _active_procs) {
            _ctxt = scalapack::Csys2blacs_handle(active_comm);
            scalapack::Cblacs_gridinit(&_ctxt, "C", _prows, _pcols);
            _ctxt_T = scalapack::Csys2blacs_handle(active_comm);
            scalapack::Cblacs_gridinit(&_ctxt_T, "R", _pcols, _prows);
          } else _ctxt = _ctxt_T = -1;
          mpi_free_comm(&active_comm);
        } else {
          _ctxt = scalapack::Csys2blacs_handle(_comm);
          scalapack::Cblacs_gridinit(&_ctxt, "C", _prows, _pcols);
          _ctxt_T = scalapack::Csys2blacs_handle(_comm);
          scalapack::Cblacs_gridinit(&_ctxt_T, "R", _pcols, _prows);
        }
        _ctxt_all = scalapack::Csys2blacs_handle(_comm);
        scalapack::Cblacs_gridinit(&_ctxt_all, "R", 1, P);
        _ctxt_loc = _ctxt;
      } else {
        _ctxt = _ctxt_T = _ctxt_all = _ctxt_loc = -1;
        this->_active = false;
      }
      if (!this->leaf()) {
        if (Pl(P) <= 1) { // child 0 is sequential, create a local context
          if (_comm != MPI_COMM_NULL) {
            MPI_Comm c0;
            int root = 0;
            MPI_Group group, sub_group;
            MPI_Comm_group(_comm, &group);
            MPI_Group_incl(group, 1, &root, &sub_group);
            MPI_Comm_create(_comm, sub_group, &c0);
            if (mpi_rank(_comm) == 0) {
              _ctxt_loc = scalapack::Csys2blacs_handle(c0);
              scalapack::Cblacs_gridinit(&_ctxt_loc, "C", 1, 1);
            }
            MPI_Group_free(&group);
            MPI_Group_free(&sub_group);
            mpi_free_comm(&c0);
          }
        } else {
          auto ch0 = static_cast<HSSMatrixMPI<scalar_t>*>(this->_ch[0].get());
          if (ch0->_active) _ctxt_loc = ch0->ctxt_loc();
        }
        if (Pr(P) <= 1) { // child 1 is sequential, create a local context
          if (_comm != MPI_COMM_NULL) {
            MPI_Comm c1;
            int root = Pl(P);
            MPI_Group group, sub_group;
            MPI_Comm_group(_comm, &group);
            MPI_Group_incl(group, 1, &root, &sub_group);
            MPI_Comm_create(_comm, sub_group, &c1);
            if (mpi_rank(_comm) == root) {
              _ctxt_loc = scalapack::Csys2blacs_handle(c1);
              scalapack::Cblacs_gridinit(&_ctxt_loc, "C", 1, 1);
            }
            MPI_Group_free(&group);
            MPI_Group_free(&sub_group);
            mpi_free_comm(&c1);
          }
        } else {
          auto ch1 = static_cast<HSSMatrixMPI<scalar_t>*>(this->_ch[1].get());
          if (ch1->_active) _ctxt_loc = ch1->ctxt_loc();
        }
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::setup_ranges
    (std::size_t roff, std::size_t coff) {
      auto P = _nprocs;
      _ranges = TreeLocalRanges(P);
      if (this->leaf()) {
        for (int p=0; p<P; p++) {
          _ranges.rlo(p) = roff;
          _ranges.rhi(p) = roff + this->rows();
          _ranges.clo(p) = coff;
          _ranges.chi(p) = coff + this->cols();
          _ranges.leaf_procs(p) = P;
        }
      } else {
        auto pl = Pl(P);
        auto pr = Pr(P);
        if (pl <= 1) {
          _ranges.rlo(0) = roff;
          _ranges.clo(0) = coff;
          if (P > 1) {
            _ranges.rhi(0) = roff + this->_ch[0]->rows();
            _ranges.chi(0) = coff + this->_ch[0]->cols();
          }
          _ranges.leaf_procs(0) = 1;
        } else {
          auto ch0 = static_cast<HSSMatrixMPI<scalar_t>*>(this->_ch[0].get());
          for (int p=0; p<pl; p++) {
            _ranges.rlo(p) = ch0->_ranges.rlo(p);
            _ranges.rhi(p) = ch0->_ranges.rhi(p);
            _ranges.clo(p) = ch0->_ranges.clo(p);
            _ranges.chi(p) = ch0->_ranges.chi(p);
            _ranges.leaf_procs(p) = ch0->_ranges.leaf_procs(p);
          }
        }
        if (pr <= 1) {
          if (P > 1) {
            _ranges.rlo(pl) = roff + this->_ch[0]->rows();
            _ranges.clo(pl) = coff + this->_ch[0]->cols();
          }
          _ranges.rhi(pl) = roff + this->rows();
          _ranges.chi(pl) = coff + this->cols();
          _ranges.leaf_procs(pl) = 1;
        } else {
          auto ch1 = static_cast<HSSMatrixMPI<scalar_t>*>(this->_ch[1].get());
          for (int p=pl; p<P; p++) {
            _ranges.rlo(p) = ch1->_ranges.rlo(p-pl);
            _ranges.rhi(p) = ch1->_ranges.rhi(p-pl);
            _ranges.clo(p) = ch1->_ranges.clo(p-pl);
            _ranges.chi(p) = ch1->_ranges.chi(p-pl);
            _ranges.leaf_procs(p) = ch1->_ranges.leaf_procs(p-pl);
          }
        }
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::setup_hierarchy
    (const opts_t& opts, MPI_Comm c, int P, bool dup_comm,
     std::size_t roff, std::size_t coff) {
      auto m = this->rows();
      auto n = this->cols();
      if (c == MPI_COMM_NULL || !dup_comm) _comm = c;
      else MPI_Comm_dup(c, &_comm);
      if (m > std::size_t(opts.leaf_size()) ||
          n > std::size_t(opts.leaf_size())) {
        this->_ch.reserve(2);
        auto pl = Pl(m, m/2, m-m/2, P);
        auto pr = Pr(m, m/2, m-m/2, P);
        if (pl > 1) {
          auto c0 = mpi_sub_comm(_comm, 0, pl);
          this->_ch.emplace_back
            (new HSSMatrixMPI<scalar_t>
             (m/2, n/2, opts, c0, pl, false, roff, coff));
        } else {
          bool act = (_comm != MPI_COMM_NULL) && (mpi_rank(_comm) == 0);
          this->_ch.emplace_back
            (new HSSMatrix<scalar_t>(m/2, n/2, opts, act));
        }
        if (pr > 1) {
          auto c1 = mpi_sub_comm(_comm, pl, pr);
          this->_ch.emplace_back
            (new HSSMatrixMPI<scalar_t>
             (m-m/2, n-n/2, opts, c1, pr, false, roff+m/2, coff+n/2));
        } else {
          bool act = (_comm != MPI_COMM_NULL) && (mpi_rank(_comm) == pl);
          this->_ch.emplace_back
            (new HSSMatrix<scalar_t>(m-m/2, n-n/2, opts, act));
        }
      }
    }

    // TODO this only works with 1 tree, so all blocks are square!!
    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::setup_hierarchy
    (const HSSPartitionTree& t, const opts_t& opts, MPI_Comm c, int P,
     bool dup_comm, std::size_t roff, std::size_t coff) {
      if (c == MPI_COMM_NULL || !dup_comm) _comm = c;
      else MPI_Comm_dup(c, &_comm);
      if (!t.c.empty()) {
        assert(t.size == t.c[0].size + t.c[1].size);
        auto pl = Pl(t.size, t.c[0].size, t.c[1].size, P);
        auto pr = Pr(t.size, t.c[0].size, t.c[1].size, P);
        this->_ch.reserve(2);
        if (pl > 1) {
          auto c0 = mpi_sub_comm(_comm, 0, pl);
          this->_ch.emplace_back
            (new HSSMatrixMPI<scalar_t>
             (t.c[0], opts, c0, pl, false, roff, coff));
        } else {
          bool act = (_comm != MPI_COMM_NULL) && (mpi_rank(_comm) == 0);
          this->_ch.emplace_back(new HSSMatrix<scalar_t>(t.c[0], opts, act));
        }
        if (pr > 1) {
          auto c1 = mpi_sub_comm(_comm, pl, pr);
          this->_ch.emplace_back
            (new HSSMatrixMPI<scalar_t>
             (t.c[1], opts, c1, pr, false,
              roff+t.c[0].size, coff+t.c[0].size));
        } else {
          bool act = (_comm != MPI_COMM_NULL) && (mpi_rank(_comm) == pl);
          this->_ch.emplace_back(new HSSMatrix<scalar_t>(t.c[1], opts, act));
        }
      }
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::~HSSMatrixMPI() {
      if (_ctxt != -1) scalapack::Cblacs_gridexit(_ctxt);
      if (_ctxt_T != -1) scalapack::Cblacs_gridexit(_ctxt_T);
      if (_ctxt_all != -1) scalapack::Cblacs_gridexit(_ctxt_all);
      mpi_free_comm(&_comm);
    }

    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::max_rank() const {
      std::size_t rmax, r=this->rank();
      MPI_Allreduce(&r, &rmax, 1, mpi_type<std::size_t>(), MPI_MAX, _comm);
      return rmax;
    }
    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::rank() const {
      if (!this->active()) return 0;
      std::size_t rank = std::max(_U.cols(), _V.cols());
      for (auto& c : this->_ch) rank = std::max(rank, c->rank());
      return rank;
    }

    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::total_memory() const {
      std::size_t mtot, m=memory();
      MPI_Allreduce(&m, &mtot, 1, mpi_type<std::size_t>(), MPI_SUM, _comm);
      return mtot;
    }
    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::memory() const {
      if (!this->active()) return 0;
      std::size_t memory = sizeof(*this) + _U.memory() + _V.memory()
        + _D.memory() + _B01.memory() + _B10.memory();
      for (auto& c : this->_ch) memory += c->memory();
      return memory;
    }

    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::total_nonzeros() const {
      std::size_t nnztot, nnz=nonzeros();
      MPI_Allreduce(&nnz, &nnztot, 1, mpi_type<std::size_t>(),
                    MPI_SUM, _comm);
      return nnztot;
    }
    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::nonzeros() const {
      if (!this->active()) return 0;
      std::size_t nnz = sizeof(*this) + _U.nonzeros() + _V.nonzeros()
        + _D.nonzeros() + _B01.nonzeros() + _B10.nonzeros();
      for (auto& c : this->_ch) nnz += c->nonzeros();
      return nnz;
    }

    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::max_levels() const {
      std::size_t ltot, l=levels();
      MPI_Allreduce(&l, &ltot, 1, mpi_type<std::size_t>(), MPI_MAX, _comm);
      return ltot;
    }
    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::levels() const {
      if (!this->active()) return 0;
      std::size_t lvls = 0;
      for (auto& c : this->_ch) lvls = std::max(lvls, c->levels());
      return 1 + lvls;
    }


    /** ctxt should be included in _ctxt_all, which is the case if it
        is included in comm? */
    template<typename scalar_t> DistributedMatrix<scalar_t>
    HSSMatrixMPI<scalar_t>::dense(int ctxt) const {
      // TODO faster implementation?  an implementation similar to the
      // sequential algorithm is difficult, as it will require a lot
      // of communication? Maybe just use the extraction routine??
      DistM_t identity(ctxt, this->cols(), this->cols());
      identity.eye();
      return apply(identity);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::communicate_child_data
    (WorkCompressMPI<scalar_t>& w) {
      w.c[0].dR = w.c[0].Rr.cols();
      w.c[0].dS = w.c[0].Sr.cols();
      w.c[1].dR = w.c[1].Rr.cols();
      w.c[1].dS = w.c[1].Sr.cols();
      int rank = mpi_rank(_comm), P = mpi_nprocs(_comm), root1 = Pl(P);
      int P0total = root1, P1total = P - root1;
      int pcols = std::floor(std::sqrt((float)P0total));
      int prows = P0total / pcols;
      int P0active = prows * pcols;
      pcols = std::floor(std::sqrt((float)P1total));
      prows = P1total / pcols;
      int P1active = prows * pcols;
      // TODO start both sends first, make sure it cannot deadlock
      if (rank < P0active) {
        if (rank < (P-P0active)) {
          // I'm one of the first P-P0active processes that are active
          // on child0, so I need to send to one or more others which
          // are not active on child0, ie the ones in [P0active,P)
          std::vector<std::size_t> buf;
          buf.reserve(8+w.c[0].Ir.size()+w.c[0].Ic.size()+
                      w.c[0].Jr.size()+w.c[0].Jc.size());
          buf.push_back(std::size_t(this->_ch[0]->_U_state));
          buf.push_back(std::size_t(this->_ch[0]->_V_state));
          buf.push_back(this->_ch[0]->_U_rank);
          buf.push_back(this->_ch[0]->_V_rank);
          buf.push_back(this->_ch[0]->_U_rows);
          buf.push_back(this->_ch[0]->_V_rows);
          buf.push_back(w.c[0].dR);
          buf.push_back(w.c[0].dS);
          for (auto i : w.c[0].Ir) buf.push_back(i);
          for (auto i : w.c[0].Ic) buf.push_back(i);
          for (auto i : w.c[0].Jr) buf.push_back(i);
          for (auto i : w.c[0].Jc) buf.push_back(i);
          for (int p=P0active; p<P; p++)
            if (rank == (p - P0active) % P0active)
              MPI_Send(buf.data(), buf.size(), mpi_type<std::size_t>(),
                       p, /*tag*/0, _comm);
        }
      } else {
        // I'm not active on child0, so I need to receive
        MPI_Status stat;
        int dest=-1, msgsize;
        for (int p=0; p<P0active; p++)
          if (p == (rank - P0active) % P0active) { dest = p; break; }
        assert(dest >= 0);
        MPI_Probe(dest, /*tag*/0, _comm, &stat);
        MPI_Get_count(&stat, mpi_type<std::size_t>(), &msgsize);
        auto buf = new std::size_t[msgsize];
        MPI_Recv(buf, msgsize, mpi_type<std::size_t>(), dest, /*tag*/0,
                 _comm, MPI_STATUS_IGNORE);
        auto ptr = buf;
        this->_ch[0]->_U_state = State(*ptr++);
        this->_ch[0]->_V_state = State(*ptr++);
        this->_ch[0]->_U_rank = *ptr++;
        this->_ch[0]->_V_rank = *ptr++;
        this->_ch[0]->_U_rows = *ptr++;
        this->_ch[0]->_V_rows = *ptr++;
        w.c[0].dR = *ptr++;
        w.c[0].dS = *ptr++;
        w.c[0].Ir.resize(this->_ch[0]->_U_rank);
        w.c[0].Ic.resize(this->_ch[0]->_V_rank);
        w.c[0].Jr.resize(this->_ch[0]->_U_rank);
        w.c[0].Jc.resize(this->_ch[0]->_V_rank);
        for (int i=0; i<this->_ch[0]->_U_rank; i++) w.c[0].Ir[i] = *ptr++;
        for (int i=0; i<this->_ch[0]->_V_rank; i++) w.c[0].Ic[i] = *ptr++;
        for (int i=0; i<this->_ch[0]->_U_rank; i++) w.c[0].Jr[i] = *ptr++;
        for (int i=0; i<this->_ch[0]->_V_rank; i++) w.c[0].Jc[i] = *ptr++;
        assert(msgsize == ptr-buf);
        delete[] buf;
      }

      if (rank >= root1 && rank < root1+P1active) {
        if ((rank-root1) < (P-P1active)) {
          // I'm one of the first P-P1active processes that are active
          // on child1, so I need to send to one or more others which
          // are not active on child1, ie the ones in [0,root1) union
          // [root1+P1active,P)
          std::vector<std::size_t> buf;
          buf.reserve(8+w.c[1].Ir.size()+w.c[1].Ic.size()+
                      w.c[1].Jr.size()+w.c[1].Jc.size());
          buf.push_back(std::size_t(this->_ch[1]->_U_state));
          buf.push_back(std::size_t(this->_ch[1]->_V_state));
          buf.push_back(this->_ch[1]->_U_rank);
          buf.push_back(this->_ch[1]->_V_rank);
          buf.push_back(this->_ch[1]->_U_rows);
          buf.push_back(this->_ch[1]->_V_rows);
          buf.push_back(w.c[1].dR);
          buf.push_back(w.c[1].dS);
          for (auto i : w.c[1].Ir) buf.push_back(i);
          for (auto i : w.c[1].Ic) buf.push_back(i);
          for (auto i : w.c[1].Jr) buf.push_back(i);
          for (auto i : w.c[1].Jc) buf.push_back(i);
          for (int p=0; p<root1; p++)
            if (rank - root1 == p % P1active)
              MPI_Send(buf.data(), buf.size(), mpi_type<std::size_t>(),
                       p, /*tag*/1, _comm);
          for (int p=root1+P1active; p<P; p++)
            if (rank - root1 == (p - P1active) % P1active)
              MPI_Send(buf.data(), buf.size(), mpi_type<std::size_t>(),
                       p, /*tag*/1, _comm);
        }
      } else {
        // I'm not active on child1, so I need to receive
        MPI_Status stat;
        int dest=-1, msgsize;
        for (int p=root1; p<root1+P1active; p++) {
          if (rank < root1) {
            if (p - root1 == rank % P1active) { dest = p; break; }
          } else if (p - root1 == (rank - P1active) % P1active) {
            dest = p; break;
          }
        }
        assert(dest >= 0);
        MPI_Probe(dest, /*tag*/1, _comm, &stat);
        MPI_Get_count(&stat, mpi_type<std::size_t>(), &msgsize);
        auto buf = new std::size_t[msgsize];
        MPI_Recv(buf, msgsize, mpi_type<std::size_t>(), dest, /*tag*/1,
                 _comm, MPI_STATUS_IGNORE);
        auto ptr = buf;
        this->_ch[1]->_U_state = State(*ptr++);
        this->_ch[1]->_V_state = State(*ptr++);
        this->_ch[1]->_U_rank = *ptr++;
        this->_ch[1]->_V_rank = *ptr++;
        this->_ch[1]->_U_rows = *ptr++;
        this->_ch[1]->_V_rows = *ptr++;
        w.c[1].dR = *ptr++;
        w.c[1].dS = *ptr++;
        w.c[1].Ir.resize(this->_ch[1]->_U_rank);
        w.c[1].Ic.resize(this->_ch[1]->_V_rank);
        w.c[1].Jr.resize(this->_ch[1]->_U_rank);
        w.c[1].Jc.resize(this->_ch[1]->_V_rank);
        for (int i=0; i<this->_ch[1]->_U_rank; i++) w.c[1].Ir[i] = *ptr++;
        for (int i=0; i<this->_ch[1]->_V_rank; i++) w.c[1].Ic[i] = *ptr++;
        for (int i=0; i<this->_ch[1]->_U_rank; i++) w.c[1].Jr[i] = *ptr++;
        for (int i=0; i<this->_ch[1]->_V_rank; i++) w.c[1].Jc[i] = *ptr++;
        assert(msgsize == ptr-buf);
        delete[] buf;
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::notify_inactives_J(WorkCompressMPI<scalar_t>& w) {
      int rank = mpi_rank(_comm), P = _nprocs, actives = _active_procs;
      int inactives = P - actives;
      if (rank < inactives) {
        std::vector<std::size_t> sbuf;
        sbuf.reserve(2+w.Jr.size()+w.Jc.size());
        sbuf.push_back(w.Jr.size());
        sbuf.push_back(w.Jc.size());
        for (auto i : w.Jr) sbuf.push_back(i);
        for (auto i : w.Jc) sbuf.push_back(i);
        MPI_Send(sbuf.data(), sbuf.size(), mpi_type<std::size_t>(),
                 actives + rank, 0, _comm);
      }
      if (rank >= actives) {
        MPI_Status stat;
        MPI_Probe(rank - actives, 0, _comm, &stat);
        int msgsize;
        MPI_Get_count(&stat, mpi_type<std::size_t>(), &msgsize);
        auto buf = new std::size_t[msgsize];
        MPI_Recv(buf, msgsize, mpi_type<std::size_t>(), rank - actives,
                 0, _comm, MPI_STATUS_IGNORE);
        auto ptr = buf;
        w.Jr.resize(*ptr++);
        w.Jc.resize(*ptr++);
        for (std::size_t i=0; i<w.Jr.size(); i++) w.Jr[i] = *ptr++;
        for (std::size_t i=0; i<w.Jc.size(); i++) w.Jc[i] = *ptr++;
        delete[] buf;
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::notify_inactives_states
    (WorkCompressMPI<scalar_t>& w) {
      int rank = mpi_rank(_comm), P = _nprocs, actives = _active_procs;
      int inactives = P - actives;
      if (rank < inactives) {
        std::vector<std::size_t> sbuf;
        sbuf.reserve(8+w.Ir.size()+w.Ic.size()+w.Jr.size()+w.Jc.size());
        sbuf.push_back(std::size_t(this->_U_state));
        sbuf.push_back(std::size_t(this->_V_state));
        sbuf.push_back(this->_U_rank);
        sbuf.push_back(this->_V_rank);
        sbuf.push_back(this->_U_rows);
        sbuf.push_back(this->_V_rows);
        sbuf.push_back(w.Rr.cols());
        sbuf.push_back(w.Sr.cols());
        for (auto i : w.Ir) sbuf.push_back(i);
        for (auto i : w.Ic) sbuf.push_back(i);
        for (auto i : w.Jr) sbuf.push_back(i);
        for (auto i : w.Jc) sbuf.push_back(i);
        MPI_Send(sbuf.data(), sbuf.size(), mpi_type<std::size_t>(),
                 actives + rank, 0, _comm);
      }
      if (rank >= actives) {
        MPI_Status stat;
        MPI_Probe(rank - actives, 0, _comm, &stat);
        int msgsize;
        MPI_Get_count(&stat, mpi_type<std::size_t>(), &msgsize);
        auto buf = new std::size_t[msgsize];
        MPI_Recv(buf, msgsize, mpi_type<std::size_t>(), rank - actives, 0,
                 _comm, MPI_STATUS_IGNORE);
        auto ptr = buf;
        this->_U_state = State(*ptr++);
        this->_V_state = State(*ptr++);
        this->_U_rank = *ptr++;
        this->_V_rank = *ptr++;
        this->_U_rows = *ptr++;
        this->_V_rows = *ptr++;
        w.dR = *ptr++;
        w.dS = *ptr++;
        w.Ir.resize(this->_U_rank);
        w.Ic.resize(this->_V_rank);
        w.Jr.resize(this->_U_rank);
        w.Jc.resize(this->_V_rank);
        for (int i=0; i<this->_U_rank; i++) w.Ir[i] = *ptr++;
        for (int i=0; i<this->_V_rank; i++) w.Ic[i] = *ptr++;
        for (int i=0; i<this->_U_rank; i++) w.Jr[i] = *ptr++;
        for (int i=0; i<this->_V_rank; i++) w.Jc[i] = *ptr++;
        delete[] buf;
      }
    }

    /**
     * Redistribute a matrix according to the tree of this HSS
     * matrix. This redistribution is based on the column partitioning
     * of the HSS matrix. If this process has a local subtree, then
     * return the local part of A in sub. Else, if this process
     * belongs to a parallel leaf, return the matrix corresponding to
     * that parallel leaf in leaf.
     */
    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::to_block_row
    (const DistM_t& dist, DenseM_t& sub, DistM_t& leaf) const {
      if (!this->active()) return;
      assert(std::size_t(dist.rows())==this->cols());
      BC2BR::block_cyclic_to_block_row
        (_ranges, dist, sub, leaf, ctxt_loc(), _comm);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::allocate_block_row
    (int d, DenseM_t& sub, DistM_t& leaf) const {
      if (!this->active()) return;
      auto rank = mpi_rank(_comm);
      for (int p=0; p<_nprocs; p++) {
        auto m = _ranges.chi(p) - _ranges.clo(p);
        if (_ranges.leaf_procs(p) == 1) {
          if (p == rank) sub = DenseM_t(m, d);
        } else {
          if (p <= rank && rank < p+_ranges.leaf_procs(p))
            leaf = DistM_t(ctxt_loc(), m, d);
          p += _ranges.leaf_procs(p)-1;
        }
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::from_block_row
    (DistM_t& dist, const DenseM_t& sub,
     const DistM_t& leaf, int lctxt) const {
      if (!this->active()) return;
      assert(std::size_t(dist.rows())==this->cols());
      BC2BR::block_row_to_block_cyclic(_ranges, dist, sub, leaf, _comm);
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::print_info
    (std::ostream &out, std::size_t roff, std::size_t coff) const {
      if (!this->active()) return;
      if (!mpi_rank(_comm)) {
        out << "rank = " << mpi_rank(_comm) << "/" << mpi_rank(MPI_COMM_WORLD)
            << " P=" << mpi_nprocs(_comm)
            << " b = [" << roff << "," << roff+this->rows()
            << " x " << coff << "," << coff+this->cols() << "]  U = "
            << this->U_rows() << " x " << this->U_rank() << " V = "
            << this->V_rows() << " x " << this->V_rank();
        if (this->leaf()) std::cout << " leaf" << std::endl;
        else out << " non-leaf" << std::endl;
      }
      for (auto& c : this->_ch) {
        MPI_Barrier(_comm);
        c->print_info(out, roff, coff);
        roff += c->rows();
        coff += c->cols();
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#include "HSSMatrixMPI.apply.hpp"
#include "HSSMatrixMPI.compress.hpp"
#include "HSSMatrixMPI.compress_stable.hpp"
#include "HSSMatrixMPI.factor.hpp"
#include "HSSMatrixMPI.solve.hpp"
#include "HSSMatrixMPI.extract.hpp"
#include "HSSMatrixMPI.Schur.hpp"

#endif // HSS_MATRIX_MPI_HPP
