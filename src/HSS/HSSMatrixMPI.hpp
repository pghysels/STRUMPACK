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
      using delem_blocks_t = typename std::function
        <void(const std::vector<std::vector<std::size_t>>& I,
              const std::vector<std::vector<std::size_t>>& J,
              std::vector<DistMW_t>& B)>;
      using elem_t = typename std::function
        <void(const std::vector<std::size_t>& I,
              const std::vector<std::size_t>& J, DenseM_t& B)>;
      using dmult_t = typename std::function
        <void(DistM_t& R, DistM_t& Sr, DistM_t& Sc)>;
      using opts_t = HSSOptions<scalar_t>;

    public:
      HSSMatrixMPI() : HSSMatrixBase<scalar_t>(0, 0, true) {}
      HSSMatrixMPI(const DistM_t& A, const opts_t& opts);
      HSSMatrixMPI
      (const HSSPartitionTree& t, const DistM_t& A, const opts_t& opts);
      HSSMatrixMPI
      (const HSSPartitionTree& t, const BLACSGrid* g, const opts_t& opts);
      HSSMatrixMPI
      (std::size_t m, std::size_t n, const BLACSGrid* Agrid,
       const dmult_t& Amult, const delem_t& Aelem, const opts_t& opts);
      HSSMatrixMPI
      (std::size_t m, std::size_t n, const BLACSGrid* Agrid,
       const dmult_t& Amult, const delem_blocks_t& Aelem, const opts_t& opts);
      HSSMatrixMPI
      (const HSSPartitionTree& t, const BLACSGrid* Agrid,
       const dmult_t& Amult, const delem_t& Aelem, const opts_t& opts);
      HSSMatrixMPI(const HSSMatrixMPI<scalar_t>& other);
      HSSMatrixMPI(HSSMatrixMPI<scalar_t>&& other) = default;
      virtual ~HSSMatrixMPI() {}

      HSSMatrixMPI<scalar_t>& operator=(const HSSMatrixMPI<scalar_t>& other);
      HSSMatrixMPI<scalar_t>& operator=(HSSMatrixMPI<scalar_t>&& other) = default;
      HSSMatrixBase<scalar_t>* clone() const override;

      const HSSMatrixBase<scalar_t>* child(int c) const
      { return this->_ch[c].get(); }
      HSSMatrixBase<scalar_t>* child(int c) { return this->_ch[c].get(); }

      inline const BLACSGrid* grid() const override { return blacs_grid_; }
      inline const BLACSGrid* grid(const BLACSGrid* grid) const override { return blacs_grid_; }
      inline const BLACSGrid* grid_local() const override { return blacs_grid_local_; }
      inline const MPIComm& Comm() const { return grid()->Comm(); }
      inline MPI_Comm comm() const { return Comm().comm(); }
      int Ptotal() const override { return grid()->P(); }
      int Pactive() const override { return grid()->npactives(); }


      void compress(const DistM_t& A, const opts_t& opts);
      void compress
      (const dmult_t& Amult, const delem_t& Aelem, const opts_t& opts);
      void compress
      (const dmult_t& Amult, const delem_blocks_t& Aelem, const opts_t& opts);

      HSSFactorsMPI<scalar_t> factor() const;
      HSSFactorsMPI<scalar_t> partial_factor() const;
      void solve(const HSSFactorsMPI<scalar_t>& ULV, DistM_t& b) const;
      void forward_solve
      (const HSSFactorsMPI<scalar_t>& ULV, WorkSolveMPI<scalar_t>& w,
       const DistM_t& b, bool partial) const;
      void backward_solve
      (const HSSFactorsMPI<scalar_t>& ULV,
       WorkSolveMPI<scalar_t>& w, DistM_t& x) const;

      DistM_t apply(const DistM_t& b) const;
      DistM_t applyC(const DistM_t& b) const;

      scalar_t get(std::size_t i, std::size_t j) const;
      DistM_t extract
      (const std::vector<std::size_t>& I,
       const std::vector<std::size_t>& J, const BLACSGrid* Bgrid) const;
      std::vector<DistM_t> extract
      (const std::vector<std::vector<std::size_t>>& I,
       const std::vector<std::vector<std::size_t>>& J,
       const BLACSGrid* Bgrid) const;
      void extract_add
      (const std::vector<std::size_t>& I,
       const std::vector<std::size_t>& J, DistM_t& B) const;
      void extract_add
      (const std::vector<std::vector<std::size_t>>& I,
       const std::vector<std::vector<std::size_t>>& J,
       std::vector<DistM_t>& B) const;

      void Schur_update
      (const HSSFactorsMPI<scalar_t>& f, DistM_t& Theta,
       DistM_t& Vhat, DistM_t& DUB01, DistM_t& Phi) const;
      void Schur_product_direct
      (const DistM_t& Theta, const DistM_t& Vhat, const DistM_t& DUB01,
       const DistM_t& Phi, const DistM_t&_ThetaVhatC,
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

      void print_info
      (std::ostream &out=std::cout,
       std::size_t roff=0, std::size_t coff=0) const;

      DistM_t dense() const;

      void shift(scalar_t sigma) override;

      const TreeLocalRanges& tree_ranges() const { return _ranges; }
      void to_block_row
      (const DistM_t& A, DenseM_t& sub_A, DistM_t& leaf_A) const;
      void allocate_block_row(int d, DenseM_t& sub_A, DistM_t& leaf_A) const;
      void from_block_row
      (DistM_t& A, const DenseM_t& sub_A, const DistM_t& leaf_A,
       const BLACSGrid* lgrid) const;

      void delete_trailing_block() override;
      void reset() override;

    private:
      using delemw_t = typename std::function
        <void(const std::vector<std::size_t>& I,
              const std::vector<std::size_t>& J,
              DistM_t& B, DistM_t& A,
              std::size_t rlo, std::size_t clo,
              MPI_Comm comm)>;

      const BLACSGrid* blacs_grid_;
      const BLACSGrid* blacs_grid_local_;
      std::unique_ptr<const BLACSGrid> owned_blacs_grid_;
      std::unique_ptr<const BLACSGrid> owned_blacs_grid_local_;

      TreeLocalRanges _ranges;

      HSSBasisIDMPI<scalar_t> _U;
      HSSBasisIDMPI<scalar_t> _V;
      DistM_t _D;
      DistM_t _B01;
      DistM_t _B10;

      // Used to redistribute the original 2D block cyclic matrix
      // according to the HSS tree
      DistM_t _A;
      DistM_t _A01;
      DistM_t _A10;

      HSSMatrixMPI
      (std::size_t m, std::size_t n, const opts_t& opts,
       const MPIComm& c, int P, std::size_t roff, std::size_t coff);
      HSSMatrixMPI
      (const HSSPartitionTree& t, const opts_t& opts,
       const MPIComm& c, int P, std::size_t roff, std::size_t coff);
      void setup_hierarchy
      (const opts_t& opts, std::size_t roff, std::size_t coff);
      void setup_hierarchy
      (const HSSPartitionTree& t, const opts_t& opts,
       std::size_t roff, std::size_t coff);
      void setup_local_context();
      void setup_ranges(std::size_t roff, std::size_t coff);

      void compress_original_nosync
      (const dmult_t& Amult, const delemw_t& Aelem, const opts_t& opts);
      void compress_original_sync
      (const dmult_t& Amult, const delemw_t& Aelem, const opts_t& opts);
      void compress_original_sync
      (const dmult_t& Amult, const delem_blocks_t& Aelem, const opts_t& opts);
      void compress_stable_nosync
      (const dmult_t& Amult, const delemw_t& Aelem, const opts_t& opts);
      void compress_stable_sync
      (const dmult_t& Amult, const delemw_t& Aelem, const opts_t& opts);
      void compress_stable_sync
      (const dmult_t& Amult, const delem_blocks_t& Aelem, const opts_t& opts);
      void compress_hard_restart_nosync
      (const dmult_t& Amult, const delemw_t& Aelem, const opts_t& opts);
      void compress_hard_restart_sync
      (const dmult_t& Amult, const delemw_t& Aelem, const opts_t& opts);
      void compress_hard_restart_sync
      (const dmult_t& Amult, const delem_blocks_t& Aelem, const opts_t& opts);

      void compress_recursive_original
      (DistSamples<scalar_t>& RS, const delemw_t& Aelem,
       const opts_t& opts, WorkCompressMPI<scalar_t>& w, int dd);
      void compress_recursive_stable
      (DistSamples<scalar_t>& RS, const delemw_t& Aelem, const opts_t& opts,
       WorkCompressMPI<scalar_t>& w, int d, int dd);
      void compute_local_samples
      (const DistSamples<scalar_t>& RS, WorkCompressMPI<scalar_t>& w, int dd);
      bool compute_U_V_bases
      (int d, const opts_t& opts, WorkCompressMPI<scalar_t>& w);
      void compute_U_basis_stable
      (const opts_t& opts, WorkCompressMPI<scalar_t>& w, int d, int dd);
      void compute_V_basis_stable
      (const opts_t& opts, WorkCompressMPI<scalar_t>& w, int d, int dd);
      bool update_orthogonal_basis
      (const opts_t& opts, scalar_t& r_max_0, const DistM_t& S,
       DistM_t& Q, int d, int dd, bool untouched, int L);
      void reduce_local_samples
      (const DistSamples<scalar_t>& RS, WorkCompressMPI<scalar_t>& w,
       int dd, bool was_compressed);
      void communicate_child_data(WorkCompressMPI<scalar_t>& w);
      void notify_inactives_J(WorkCompressMPI<scalar_t>& w);
      void notify_inactives_states(WorkCompressMPI<scalar_t>& w);

      void compress_level_original
      (DistSamples<scalar_t>& RS, const opts_t& opts,
       WorkCompressMPI<scalar_t>& w, int dd, int lvl);
      void compress_level_stable
      (DistSamples<scalar_t>& RS, const opts_t& opts,
       WorkCompressMPI<scalar_t>& w, int d, int dd, int lvl);
      void extract_level
      (const delemw_t& Aelem, const opts_t& opts,
       WorkCompressMPI<scalar_t>& w, int lvl);
      void extract_level
      (const delem_blocks_t& Aelem, const opts_t& opts,
       WorkCompressMPI<scalar_t>& w, int lvl);
      void get_extraction_indices
      (std::vector<std::vector<std::size_t>>& I,
       std::vector<std::vector<std::size_t>>& J,
       WorkCompressMPI<scalar_t>& w, int& self, int lvl);
      void get_extraction_indices
      (std::vector<std::vector<std::size_t>>& I,
       std::vector<std::vector<std::size_t>>& J, std::vector<DistMW_t>& B,
       const BLACSGrid* lg, WorkCompressMPI<scalar_t>& w,
       int& self, int lvl);
      void allgather_extraction_indices
      (std::vector<std::vector<std::size_t>>& lI,
       std::vector<std::vector<std::size_t>>& lJ,
       std::vector<std::vector<std::size_t>>& I,
       std::vector<std::vector<std::size_t>>& J,
       int& before, int self, int& after);
      void extract_D_B
      (const delemw_t& Aelem, const BLACSGrid* lg, const opts_t& opts,
       WorkCompressMPI<scalar_t>& w, int lvl);

      void factor_recursive
      (HSSFactorsMPI<scalar_t>& f, WorkFactorMPI<scalar_t>& w,
       const BLACSGrid* lg, bool isroot, bool partial) const;

      void solve_fwd
      (const HSSFactorsMPI<scalar_t>& ULV, const DistSubLeaf<scalar_t>& b,
       WorkSolveMPI<scalar_t>& w, bool partial, bool isroot) const;
      void solve_bwd
      (const HSSFactorsMPI<scalar_t>& ULV, DistSubLeaf<scalar_t>& x,
       WorkSolveMPI<scalar_t>& w, bool isroot) const;

      void apply_fwd
      (const DistSubLeaf<scalar_t>& B, WorkApplyMPI<scalar_t>& w,
       bool isroot, long long int flops) const;
      void apply_bwd
      (const DistSubLeaf<scalar_t>& B, scalar_t beta,
       DistSubLeaf<scalar_t>& C, WorkApplyMPI<scalar_t>& w,
       bool isroot, long long int flops) const;
      void applyT_fwd
      (const DistSubLeaf<scalar_t>& B, WorkApplyMPI<scalar_t>& w,
       bool isroot, long long int flops) const;
      void applyT_bwd
      (const DistSubLeaf<scalar_t>& B, scalar_t beta,
       DistSubLeaf<scalar_t>& C, WorkApplyMPI<scalar_t>& w,
       bool isroot, long long int flops) const;

      void extract_fwd
      (WorkExtractMPI<scalar_t>& w, const BLACSGrid* lg, bool odiag) const;
      void extract_bwd
      (std::vector<Triplet<scalar_t>>& triplets,
       const BLACSGrid* lg, WorkExtractMPI<scalar_t>& w) const;
      void triplets_to_DistM
      (std::vector<Triplet<scalar_t>>& triplets, DistM_t& B) const;
      void extract_fwd
      (WorkExtractBlocksMPI<scalar_t>& w, const BLACSGrid* lg,
       std::vector<bool>& odiag) const;
      void extract_bwd
      (std::vector<std::vector<Triplet<scalar_t>>>& triplets,
       const BLACSGrid* lg, WorkExtractBlocksMPI<scalar_t>& w) const;
      void triplets_to_DistM
      (std::vector<std::vector<Triplet<scalar_t>>>& triplets,
       std::vector<DistM_t>& B) const;

      void redistribute_to_tree_to_buffers
      (const DistM_t& A, std::size_t Arlo, std::size_t Aclo,
       std::vector<std::vector<scalar_t>>& sbuf, int dest=0) override;
      void redistribute_to_tree_from_buffers
      (const DistM_t& A, std::size_t rlo, std::size_t clo,
       std::vector<scalar_t*>& pbuf) override;
      void delete_redistributed_input() override;

      void apply_UV_big
      (DistSubLeaf<scalar_t>& Theta, DistM_t& Uop,
       DistSubLeaf<scalar_t>& Phi, DistM_t& Vop,
       long long int& flops) const override;

      static int Pl(std::size_t n, std::size_t nl, std::size_t nr, int P) {
        return std::max
          (1, std::min(int(std::round(float(P) * nl / n)), P-1));
      }
      static int Pr(std::size_t n, std::size_t nl, std::size_t nr, int P)
      { return std::max(1, P - Pl(n, nl, nr, P));  }
      int Pl() const {
        return Pl(this->rows(), this->_ch[0]->rows(),
                  this->_ch[1]->rows(), Ptotal());
      }
      int Pr() const {
        return Pr(this->rows(), this->_ch[0]->rows(),
                  this->_ch[1]->rows(), Ptotal());
      }

      template<typename T> friend
      void apply_HSS(Trans ta, const HSSMatrixMPI<T>& a,
                     const DistributedMatrix<T>& b, T beta,
                     DistributedMatrix<T>& c);
      friend class DistSamples<scalar_t>;
    };

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const DistM_t& A, const opts_t& opts)
      : HSSMatrixBase<scalar_t>(A.rows(), A.cols(), true),
      blacs_grid_(A.grid()) {
      setup_hierarchy(opts, 0, 0);
      setup_local_context();
      setup_ranges(0, 0);
      compress(A, opts);
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const HSSPartitionTree& t, const BLACSGrid* g, const opts_t& opts)
      : HSSMatrixBase<scalar_t>(t.size, t.size, true), blacs_grid_(g) {
      setup_hierarchy(t, opts, 0, 0);
      setup_local_context();
      setup_ranges(0, 0);
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const HSSPartitionTree& t, const DistM_t& A, const opts_t& opts)
      : HSSMatrixBase<scalar_t>(A.rows(), A.cols(), true),
      blacs_grid_(A.grid()) {
      assert(t.size == A.rows() && t.size == A.cols());
      setup_hierarchy(t, opts, 0, 0);
      setup_local_context();
      setup_ranges(0, 0);
      compress(A, opts);
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (std::size_t m, std::size_t n, const BLACSGrid* Agrid,
     const dmult_t& Amult, const delem_t& Aelem, const opts_t& opts)
      : HSSMatrixBase<scalar_t>(m, n, true), blacs_grid_(Agrid) {
      setup_hierarchy(opts, 0, 0);
      setup_local_context();
      setup_ranges(0, 0);
      compress(Amult, Aelem, opts);
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const HSSPartitionTree& t, const BLACSGrid* Agrid,
     const dmult_t& Amult, const delem_t& Aelem, const opts_t& opts)
      : HSSMatrixBase<scalar_t>(t.size, t.size, true), blacs_grid_(Agrid) {
      setup_hierarchy(t, opts, 0, 0);
      setup_local_context();
      setup_ranges(0, 0);
      compress(Amult, Aelem, opts);
    }

    template<typename scalar_t>
    HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const HSSMatrixMPI<scalar_t>& other)
      : HSSMatrixBase<scalar_t>(other.rows(), other.cols(), other.active()) {
      *this = other;
    }

    template<typename scalar_t> HSSMatrixMPI<scalar_t>&
    HSSMatrixMPI<scalar_t>::operator=(const HSSMatrixMPI<scalar_t>& other) {
      HSSMatrixBase<scalar_t>::operator=(other);
      blacs_grid_ = other.blacs_grid_;
      blacs_grid_local_ = other.blacs_grid_local_;
      auto& og = *other.owned_blacs_grid_.get();
      owned_blacs_grid_ = std::unique_ptr<BLACSGrid>(new BLACSGrid(og));
      auto& ogl = *other.owned_blacs_grid_.get();
      owned_blacs_grid_local_ = std::unique_ptr<BLACSGrid>(new BLACSGrid(ogl));
      _ranges = other._ranges;
      _U = other._U;
      _V = other._V;
      _D = other._D;
      _B01 = other._B01;
      _B10 = other._B10;
      return *this;
    }

    template<typename scalar_t> HSSMatrixBase<scalar_t>*
    HSSMatrixMPI<scalar_t>::clone() const {
      return new HSSMatrixMPI<scalar_t>(*this);
    }

    /** private constructor */
    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (std::size_t m, std::size_t n, const opts_t& opts,
     const MPIComm& c, int P, std::size_t roff, std::size_t coff)
      : HSSMatrixBase<scalar_t>(m, n, !c.is_null()) {
      owned_blacs_grid_ = std::unique_ptr<BLACSGrid>(new BLACSGrid(c, P));
      blacs_grid_ = owned_blacs_grid_.get();
      setup_hierarchy(opts, roff, coff);
      setup_local_context();
      setup_ranges(roff, coff);
    }

    /** private constructor */
    template<typename scalar_t> HSSMatrixMPI<scalar_t>::HSSMatrixMPI
    (const HSSPartitionTree& t, const opts_t& opts,
     const MPIComm& c, int P, std::size_t roff, std::size_t coff)
      : HSSMatrixBase<scalar_t>(t.size, t.size, !c.is_null()) {
      owned_blacs_grid_ = std::unique_ptr<const BLACSGrid>(new BLACSGrid(c, P));
      blacs_grid_ = owned_blacs_grid_.get();
      setup_hierarchy(t, opts, roff, coff);
      setup_local_context();
      setup_ranges(roff, coff);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::setup_local_context() {
      if (!this->leaf()) {
        if (Pl() <= 1) { // child 0 is sequential, create a local context
          if (!Comm().is_null()) {
            auto c = Comm().sub_self(0);
            if (Comm().rank() == 0) {
              owned_blacs_grid_local_ = std::unique_ptr<const BLACSGrid>
                (new BLACSGrid(std::move(c), 1));
              blacs_grid_local_ = owned_blacs_grid_local_.get();
            }
          }
        } else {
          if (this->_ch[0]->active())
            blacs_grid_local_ = this->_ch[0]->grid_local();
        }
        if (Pr() <= 1) { // child 1 is sequential, create a local context
          if (!Comm().is_null()) {
            auto c = Comm().sub_self(Pl());
            if (Comm().rank() == Pl()) {
              owned_blacs_grid_local_ = std::unique_ptr<const BLACSGrid>
                (new BLACSGrid(std::move(c), 1));
              blacs_grid_local_ = owned_blacs_grid_local_.get();
            }
          }
        } else {
          if (this->_ch[1]->active())
            blacs_grid_local_ = this->_ch[1]->grid_local();
        }
      } else blacs_grid_local_ = blacs_grid_;
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::setup_ranges
    (std::size_t roff, std::size_t coff) {
      const int P = Ptotal();
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
        auto pl = Pl();
        auto pr = Pr();
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
    (const opts_t& opts, std::size_t roff, std::size_t coff) {
      auto m = this->rows();
      auto n = this->cols();
      if (m > std::size_t(opts.leaf_size()) ||
          n > std::size_t(opts.leaf_size())) {
        this->_ch.reserve(2);
        auto pl = Pl(m, m/2, m-m/2, Ptotal());
        auto pr = Pr(m, m/2, m-m/2, Ptotal());
        if (pl > 1) {
          this->_ch.emplace_back
            (new HSSMatrixMPI<scalar_t>
             (m/2, n/2, opts, Comm().sub(0, pl), pl, roff, coff));
        } else {
          bool act = !Comm().is_null() && Comm().rank() == 0;
          this->_ch.emplace_back
            (new HSSMatrix<scalar_t>(m/2, n/2, opts, act));
        }
        if (pr > 1) {
          this->_ch.emplace_back
            (new HSSMatrixMPI<scalar_t>
             (m-m/2, n-n/2, opts, Comm().sub(pl, pr), pr,
              roff+m/2, coff+n/2));
        } else {
          bool act = !Comm().is_null() && Comm().rank() == pl;
          this->_ch.emplace_back
            (new HSSMatrix<scalar_t>(m-m/2, n-n/2, opts, act));
        }
      }
    }

    // TODO this only works with 1 tree, so all blocks are square!!
    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::setup_hierarchy
    (const HSSPartitionTree& t, const opts_t& opts,
     std::size_t roff, std::size_t coff) {
      if (!t.c.empty()) {
        assert(t.size == t.c[0].size + t.c[1].size);
        const int P = grid()->P();
        auto pl = Pl(t.size, t.c[0].size, t.c[1].size, P);
        auto pr = Pr(t.size, t.c[0].size, t.c[1].size, P);
        this->_ch.reserve(2);
        if (pl > 1) {
          this->_ch.emplace_back
            (new HSSMatrixMPI<scalar_t>
             (t.c[0], opts, Comm().sub(0, pl), pl, roff, coff));
        } else {
          bool act = !Comm().is_null() && Comm().rank() == 0;
          this->_ch.emplace_back(new HSSMatrix<scalar_t>(t.c[0], opts, act));
        }
        if (pr > 1) {
          this->_ch.emplace_back
            (new HSSMatrixMPI<scalar_t>
             (t.c[1], opts, Comm().sub(pl, pr), pr,
              roff+t.c[0].size, coff+t.c[0].size));
        } else {
          bool act = !Comm().is_null() && Comm().rank() == pl;
          this->_ch.emplace_back(new HSSMatrix<scalar_t>(t.c[1], opts, act));
        }
      }
    }

    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::max_rank() const {
      return Comm().all_reduce(this->rank(), MPI_MAX);
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
      return Comm().all_reduce(memory(), MPI_SUM);
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
      return Comm().all_reduce(nonzeros(), MPI_SUM);
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
      return Comm().all_reduce(levels(), MPI_MAX);
    }
    template<typename scalar_t> std::size_t
    HSSMatrixMPI<scalar_t>::levels() const {
      if (!this->active()) return 0;
      std::size_t lvls = 0;
      for (auto& c : this->_ch) lvls = std::max(lvls, c->levels());
      return 1 + lvls;
    }

    template<typename scalar_t> DistributedMatrix<scalar_t>
    HSSMatrixMPI<scalar_t>::dense() const {
      // TODO faster implementation?  an implementation similar to the
      // sequential algorithm is difficult, as it will require a lot
      // of communication? Maybe just use the extraction routine??
      DistM_t identity(grid(), this->cols(), this->cols());
      identity.eye();
      return apply(identity);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::shift(scalar_t sigma) {
      if (!this->active()) return;
      if (this->leaf()) _D.shift(sigma);
      else for (auto& c : this->_ch) c->shift(sigma);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::communicate_child_data
    (WorkCompressMPI<scalar_t>& w) {
      w.c[0].dR = w.c[0].Rr.cols();
      w.c[0].dS = w.c[0].Sr.cols();
      w.c[1].dR = w.c[1].Rr.cols();
      w.c[1].dS = w.c[1].Sr.cols();
      int rank = Comm().rank(), P = Ptotal(), root1 = Pl();
      int P0active = this->_ch[0]->Pactive();
      int P1active = this->_ch[1]->Pactive();
      std::vector<MPIRequest> sreq;
      std::vector<std::size_t> sbuf0, sbuf1;
      if (rank < P0active) {
        if (rank < (P-P0active)) {
          // I'm one of the first P-P0active processes that are active
          // on child0, so I need to send to one or more others which
          // are not active on child0, ie the ones in [P0active,P)
          sbuf0.reserve(8+w.c[0].Ir.size()+w.c[0].Ic.size()+
                        w.c[0].Jr.size()+w.c[0].Jc.size());
          sbuf0.push_back(std::size_t(this->_ch[0]->_U_state));
          sbuf0.push_back(std::size_t(this->_ch[0]->_V_state));
          sbuf0.push_back(this->_ch[0]->_U_rank);
          sbuf0.push_back(this->_ch[0]->_V_rank);
          sbuf0.push_back(this->_ch[0]->_U_rows);
          sbuf0.push_back(this->_ch[0]->_V_rows);
          sbuf0.push_back(w.c[0].dR);
          sbuf0.push_back(w.c[0].dS);
          for (auto i : w.c[0].Ir) sbuf0.push_back(i);
          for (auto i : w.c[0].Ic) sbuf0.push_back(i);
          for (auto i : w.c[0].Jr) sbuf0.push_back(i);
          for (auto i : w.c[0].Jc) sbuf0.push_back(i);
          for (int p=P0active; p<P; p++)
            if (rank == (p - P0active) % P0active)
              sreq.emplace_back(Comm().isend(sbuf0, p, 0));
        }
      }
      if (rank >= root1 && rank < root1+P1active) {
        if ((rank-root1) < (P-P1active)) {
          // I'm one of the first P-P1active processes that are active
          // on child1, so I need to send to one or more others which
          // are not active on child1, ie the ones in [0,root1) union
          // [root1+P1active,P)
          sbuf1.reserve(8+w.c[1].Ir.size()+w.c[1].Ic.size()+
                        w.c[1].Jr.size()+w.c[1].Jc.size());
          sbuf1.push_back(std::size_t(this->_ch[1]->_U_state));
          sbuf1.push_back(std::size_t(this->_ch[1]->_V_state));
          sbuf1.push_back(this->_ch[1]->_U_rank);
          sbuf1.push_back(this->_ch[1]->_V_rank);
          sbuf1.push_back(this->_ch[1]->_U_rows);
          sbuf1.push_back(this->_ch[1]->_V_rows);
          sbuf1.push_back(w.c[1].dR);
          sbuf1.push_back(w.c[1].dS);
          for (auto i : w.c[1].Ir) sbuf1.push_back(i);
          for (auto i : w.c[1].Ic) sbuf1.push_back(i);
          for (auto i : w.c[1].Jr) sbuf1.push_back(i);
          for (auto i : w.c[1].Jc) sbuf1.push_back(i);
          for (int p=0; p<root1; p++)
            if (rank - root1 == p % P1active)
              sreq.emplace_back(Comm().isend(sbuf1, p, 1));
          for (int p=root1+P1active; p<P; p++)
            if (rank - root1 == (p - P1active) % P1active)
              sreq.emplace_back(Comm().isend(sbuf1, p, 1));
        }
      }
      if (!(rank < P0active)) {
        // I'm not active on child0, so I need to receive
        int dest = -1;
        for (int p=0; p<P0active; p++)
          if (p == (rank - P0active) % P0active) { dest = p; break; }
        assert(dest >= 0);
        auto buf = Comm().template recv<std::size_t>(dest, 0);
        auto ptr = buf.begin();
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
        //assert(msgsize == std::distance(buf, ptr));
      }
      if (!(rank >= root1 && rank < root1+P1active)) {
        // I'm not active on child1, so I need to receive
        int dest = -1;
        for (int p=root1; p<root1+P1active; p++) {
          if (rank < root1) {
            if (p - root1 == rank % P1active) { dest = p; break; }
          } else if (p - root1 == (rank - P1active) % P1active) {
            dest = p; break;
          }
        }
        assert(dest >= 0);
        auto buf = Comm().template recv<std::size_t>(dest, 1);
        auto ptr = buf.begin();
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
        //assert(msgsize == std::distance(buf, ptr));
      }
      wait_all(sreq);
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::notify_inactives_J(WorkCompressMPI<scalar_t>& w) {
      int rank = Comm().rank(), actives = Pactive();
      int inactives = grid()->P() - actives;
      if (rank < inactives) {
        std::vector<std::size_t> sbuf;
        sbuf.reserve(2+w.Jr.size()+w.Jc.size());
        sbuf.push_back(w.Jr.size());
        sbuf.push_back(w.Jc.size());
        for (auto i : w.Jr) sbuf.push_back(i);
        for (auto i : w.Jc) sbuf.push_back(i);
        Comm().send(sbuf, actives+rank, 0);
      }
      if (rank >= actives) {
        auto buf = Comm().template recv<std::size_t>(rank - actives, 0);
        auto ptr = buf.begin();
        w.Jr.resize(*ptr++);
        w.Jc.resize(*ptr++);
        for (std::size_t i=0; i<w.Jr.size(); i++) w.Jr[i] = *ptr++;
        for (std::size_t i=0; i<w.Jc.size(); i++) w.Jc[i] = *ptr++;
      }
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::notify_inactives_states
    (WorkCompressMPI<scalar_t>& w) {
      int rank = Comm().rank(), actives = Pactive();
      int inactives = Ptotal() - actives;
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
        Comm().send(sbuf, actives+rank, 0);
      }
      if (rank >= actives) {
        auto buf = Comm().template recv<std::size_t>(rank-actives, 0);
        auto ptr = buf.begin();
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
        (_ranges, dist, sub, leaf, grid_local(), Comm());
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::allocate_block_row
    (int d, DenseM_t& sub, DistM_t& leaf) const {
      if (!this->active()) return;
      const int rank = Comm().rank();
      const int P = grid()->P();
      for (int p=0; p<P; p++) {
        auto m = _ranges.chi(p) - _ranges.clo(p);
        if (_ranges.leaf_procs(p) == 1) {
          if (p == rank) sub = DenseM_t(m, d);
        } else {
          if (p <= rank && rank < p+_ranges.leaf_procs(p))
            leaf = DistM_t(grid_local(), m, d);
          p += _ranges.leaf_procs(p)-1;
        }
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::from_block_row
    (DistM_t& dist, const DenseM_t& sub, const DistM_t& leaf,
     const BLACSGrid* lg) const {
      if (!this->active()) return;
      assert(std::size_t(dist.rows())==this->cols());
      BC2BR::block_row_to_block_cyclic(_ranges, dist, sub, leaf, Comm());
    }

    template<typename scalar_t> void
    HSSMatrixMPI<scalar_t>::delete_trailing_block() {
      _B01.clear();
      _B10.clear();
      HSSMatrixBase<scalar_t>::delete_trailing_block();
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::reset() {
      _U.clear();
      _V.clear();
      _D.clear();
      _B01.clear();
      _B10.clear();
      HSSMatrixBase<scalar_t>::reset();
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::print_info
    (std::ostream &out, std::size_t roff, std::size_t coff) const {
      if (!this->active()) return;
      if (Comm().is_root()) {
        out << "rank = " << Comm().rank() << "/" << mpi_rank(MPI_COMM_WORLD)
            << " P=" << Comm().size()
            << " b = [" << roff << "," << roff+this->rows()
            << " x " << coff << "," << coff+this->cols() << "]  U = "
            << this->U_rows() << " x " << this->U_rank() << " V = "
            << this->V_rows() << " x " << this->V_rank();
        if (this->leaf()) std::cout << " leaf" << std::endl;
        else out << " non-leaf" << std::endl;
      }
      for (auto& c : this->_ch) {
        Comm().barrier();
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
#include "HSSMatrixMPI.extract_blocks.hpp"
#include "HSSMatrixMPI.Schur.hpp"

#endif // HSS_MATRIX_MPI_HPP
