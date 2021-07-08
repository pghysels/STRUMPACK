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
/**
 * \file HSSMatrixMPI.hpp
 *
 * \brief This file contains the HSSMatrixMPI class definition as well
 * as implementations for a number of it's member routines. Other
 * member routines are implemented in files such as
 * HSSMatrixMPI.apply.hpp, HSSMatrixMPI.factor.hpp etc.
 */
#ifndef HSS_MATRIX_MPI_HPP
#define HSS_MATRIX_MPI_HPP

#include <cassert>

#include "HSSMatrix.hpp"
#include "misc/MPIWrapper.hpp"
#include "HSSExtraMPI.hpp"
#include "DistSamples.hpp"
#include "DistElemMult.hpp"
#include "HSSBasisIDMPI.hpp"
#include "kernel/Kernel.hpp"

namespace strumpack {
  namespace HSS {

    /**
     * \class HSSMatrixMPI
     *
     * \brief Distributed memory implementation of
     * the HSS (Hierarchically Semi-Separable) matrix format
     *
     * This is for non-symmetric matrices, but can be used with
     * symmetric matrices as well. This class inherits from
     * StructuredMatrix.
     *
     * \tparam scalar_t Can be float, double, std:complex<float> or
     * std::complex<double>.
     *
     * \see HSSMatrix, structured::StructuredMatrix
     */
    template<typename scalar_t> class HSSMatrixMPI
      : public HSSMatrixBase<scalar_t> {
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
      HSSMatrixMPI(const structured::ClusterTree& t,
                   const DistM_t& A, const opts_t& opts);
      HSSMatrixMPI(const structured::ClusterTree& t,
                   const BLACSGrid* g, const opts_t& opts);
      HSSMatrixMPI(std::size_t m, std::size_t n, const BLACSGrid* Agrid,
                   const dmult_t& Amult, const delem_t& Aelem,
                   const opts_t& opts);
      HSSMatrixMPI(std::size_t m, std::size_t n, const BLACSGrid* Agrid,
                   const dmult_t& Amult, const delem_blocks_t& Aelem,
                   const opts_t& opts);
      HSSMatrixMPI(const structured::ClusterTree& t, const BLACSGrid* Agrid,
                   const dmult_t& Amult, const delem_t& Aelem,
                   const opts_t& opts);
      HSSMatrixMPI(kernel::Kernel<real_t>& K, const BLACSGrid* Agrid,
                   const opts_t& opts);
      HSSMatrixMPI(const HSSMatrixMPI<scalar_t>& other);
      HSSMatrixMPI(HSSMatrixMPI<scalar_t>&& other) = default;
      virtual ~HSSMatrixMPI() {}

      HSSMatrixMPI<scalar_t>& operator=(const HSSMatrixMPI<scalar_t>& other);
      HSSMatrixMPI<scalar_t>& operator=(HSSMatrixMPI<scalar_t>&& other) = default;
      std::unique_ptr<HSSMatrixBase<scalar_t>> clone() const override;

      const HSSMatrixBase<scalar_t>* child(int c) const {
        return this->ch_[c].get();
      }
      HSSMatrixBase<scalar_t>* child(int c) { return this->ch_[c].get(); }

      const BLACSGrid* grid() const override { return blacs_grid_; }
      const BLACSGrid* grid(const BLACSGrid* grid) const override { return blacs_grid_; }
      const BLACSGrid* grid_local() const override { return blacs_grid_local_; }
      const MPIComm& Comm() const { return grid()->Comm(); }
      MPI_Comm comm() const { return Comm().comm(); }
      int Ptotal() const override { return grid()->P(); }
      int Pactive() const override { return grid()->npactives(); }


      void compress(const DistM_t& A,
                    const opts_t& opts);
      void compress(const dmult_t& Amult,
                    const delem_t& Aelem,
                    const opts_t& opts);
      void compress(const dmult_t& Amult,
                    const delem_blocks_t& Aelem,
                    const opts_t& opts);
      void compress(const kernel::Kernel<real_t>& K, const opts_t& opts);

      void factor() override;
      void partial_factor();
      void solve(DistM_t& b) const override;
      void forward_solve(WorkSolveMPI<scalar_t>& w, const DistM_t& b,
                         bool partial) const override;
      void backward_solve(WorkSolveMPI<scalar_t>& w,
                          DistM_t& x) const override;

      DistM_t apply(const DistM_t& b) const;
      DistM_t applyC(const DistM_t& b) const;

      void mult(Trans op, const DistM_t& x, DistM_t& y) const override;

      scalar_t get(std::size_t i, std::size_t j) const;
      DistM_t extract(const std::vector<std::size_t>& I,
                      const std::vector<std::size_t>& J,
                      const BLACSGrid* Bgrid) const;
      std::vector<DistM_t>
      extract(const std::vector<std::vector<std::size_t>>& I,
              const std::vector<std::vector<std::size_t>>& J,
              const BLACSGrid* Bgrid) const;
      void extract_add(const std::vector<std::size_t>& I,
                       const std::vector<std::size_t>& J, DistM_t& B) const;
      void extract_add(const std::vector<std::vector<std::size_t>>& I,
                       const std::vector<std::vector<std::size_t>>& J,
                       std::vector<DistM_t>& B) const;

      void Schur_update(DistM_t& Theta, DistM_t& Vhat,
                        DistM_t& DUB01, DistM_t& Phi) const;
      void Schur_product_direct(const DistM_t& Theta,
                                const DistM_t& Vhat,
                                const DistM_t& DUB01,
                                const DistM_t& Phi,
                                const DistM_t&_ThetaVhatC,
                                const DistM_t& VhatCPhiC,
                                const DistM_t& R,
                                DistM_t& Sr, DistM_t& Sc) const;

      std::size_t max_rank() const;        // collective on comm()
      std::size_t total_memory() const;    // collective on comm()
      std::size_t total_nonzeros() const;  // collective on comm()
      std::size_t total_factor_nonzeros() const;  // collective on comm()
      std::size_t max_levels() const;      // collective on comm()
      std::size_t rank() const override;
      std::size_t memory() const override;
      std::size_t nonzeros() const override;
      std::size_t factor_nonzeros() const override;
      std::size_t levels() const override;

      void print_info(std::ostream &out=std::cout,
                      std::size_t roff=0,
                      std::size_t coff=0) const override;

      DistM_t dense() const;

      void shift(scalar_t sigma) override;

      const TreeLocalRanges& tree_ranges() const { return ranges_; }
      void to_block_row(const DistM_t& A,
                        DenseM_t& sub_A,
                        DistM_t& leaf_A) const override;
      void allocate_block_row(int d, DenseM_t& sub_A,
                              DistM_t& leaf_A) const override;
      void from_block_row(DistM_t& A,
                          const DenseM_t& sub_A,
                          const DistM_t& leaf_A,
                          const BLACSGrid* lgrid) const override;

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

      TreeLocalRanges ranges_;

      HSSBasisIDMPI<scalar_t> U_, V_;
      DistM_t D_, B01_, B10_;

      // Used to redistribute the original 2D block cyclic matrix
      // according to the HSS tree
      DistM_t A_, A01_, A10_;

      HSSMatrixMPI(std::size_t m, std::size_t n, const opts_t& opts,
                   const MPIComm& c, int P,
                   std::size_t roff, std::size_t coff);
      HSSMatrixMPI(const structured::ClusterTree& t, const opts_t& opts,
                   const MPIComm& c, int P,
                   std::size_t roff, std::size_t coff);
      void setup_hierarchy(const opts_t& opts,
                           std::size_t roff, std::size_t coff);
      void setup_hierarchy(const structured::ClusterTree& t, const opts_t& opts,
                           std::size_t roff, std::size_t coff);
      void setup_local_context();
      void setup_ranges(std::size_t roff, std::size_t coff);

      void compress_original_nosync(const dmult_t& Amult,
                                    const delemw_t& Aelem,
                                    const opts_t& opts);
      void compress_original_sync(const dmult_t& Amult,
                                  const delemw_t& Aelem,
                                  const opts_t& opts);
      void compress_original_sync(const dmult_t& Amult,
                                  const delem_blocks_t& Aelem,
                                  const opts_t& opts);
      void compress_stable_nosync(const dmult_t& Amult,
                                  const delemw_t& Aelem,
                                  const opts_t& opts);
      void compress_stable_sync(const dmult_t& Amult,
                                const delemw_t& Aelem,
                                const opts_t& opts);
      void compress_stable_sync(const dmult_t& Amult,
                                const delem_blocks_t& Aelem,
                                const opts_t& opts);
      void compress_hard_restart_nosync(const dmult_t& Amult,
                                        const delemw_t& Aelem,
                                        const opts_t& opts);
      void compress_hard_restart_sync(const dmult_t& Amult,
                                      const delemw_t& Aelem,
                                      const opts_t& opts);
      void compress_hard_restart_sync(const dmult_t& Amult,
                                      const delem_blocks_t& Aelem,
                                      const opts_t& opts);

      void compress_recursive_ann(DenseMatrix<std::uint32_t>& ann,
                                  DenseMatrix<real_t>& scores,
                                  const delemw_t& Aelem,
                                  WorkCompressMPIANN<scalar_t>& w,
                                  const opts_t& opts,
                                  const BLACSGrid* lg) override;
      void compute_local_samples_ann(DenseMatrix<std::uint32_t>& ann,
                                     DenseMatrix<real_t>& scores,
                                     WorkCompressMPIANN<scalar_t>& w,
                                     const delemw_t& Aelem,
                                     const opts_t& opts);
      bool compute_U_V_bases_ann(DistM_t& S, const opts_t& opts,
                                 WorkCompressMPIANN<scalar_t>& w);
      void communicate_child_data_ann(WorkCompressMPIANN<scalar_t>& w);

      void compress_recursive_original(DistSamples<scalar_t>& RS,
                                       const delemw_t& Aelem,
                                       const opts_t& opts,
                                       WorkCompressMPI<scalar_t>& w,
                                       int dd) override;
      void compress_recursive_stable(DistSamples<scalar_t>& RS,
                                     const delemw_t& Aelem,
                                     const opts_t& opts,
                                     WorkCompressMPI<scalar_t>& w,
                                     int d, int dd) override;
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
      bool update_orthogonal_basis(const opts_t& opts,
                                   scalar_t& r_max_0, const DistM_t& S,
                                   DistM_t& Q, int d, int dd,
                                   bool untouched, int L);
      void reduce_local_samples(const DistSamples<scalar_t>& RS,
                                WorkCompressMPI<scalar_t>& w,
                                int dd, bool was_compressed);
      void communicate_child_data(WorkCompressMPI<scalar_t>& w);
      void notify_inactives_J(WorkCompressMPI<scalar_t>& w);
      void notify_inactives_J(WorkCompressMPIANN<scalar_t>& w);
      void notify_inactives_states(WorkCompressMPI<scalar_t>& w);

      void compress_level_original(DistSamples<scalar_t>& RS,
                                   const opts_t& opts,
                                   WorkCompressMPI<scalar_t>& w,
                                   int dd, int lvl) override;
      void compress_level_stable(DistSamples<scalar_t>& RS,
                                 const opts_t& opts,
                                 WorkCompressMPI<scalar_t>& w,
                                 int d, int dd, int lvl) override;
      void extract_level(const delemw_t& Aelem, const opts_t& opts,
                         WorkCompressMPI<scalar_t>& w, int lvl);
      void extract_level(const delem_blocks_t& Aelem, const opts_t& opts,
                         WorkCompressMPI<scalar_t>& w, int lvl);
      void get_extraction_indices(std::vector<std::vector<std::size_t>>& I,
                                  std::vector<std::vector<std::size_t>>& J,
                                  WorkCompressMPI<scalar_t>& w,
                                  int& self, int lvl) override;
      void get_extraction_indices(std::vector<std::vector<std::size_t>>& I,
                                  std::vector<std::vector<std::size_t>>& J,
                                  std::vector<DistMW_t>& B,
                                  const BLACSGrid* lg,
                                  WorkCompressMPI<scalar_t>& w,
                                  int& self, int lvl) override;
      void allgather_extraction_indices(std::vector<std::vector<std::size_t>>& lI,
                                        std::vector<std::vector<std::size_t>>& lJ,
                                        std::vector<std::vector<std::size_t>>& I,
                                        std::vector<std::vector<std::size_t>>& J,
                                        int& before, int self, int& after);
      void extract_D_B(const delemw_t& Aelem,
                       const BLACSGrid* lg, const opts_t& opts,
                       WorkCompressMPI<scalar_t>& w, int lvl) override;

      void factor_recursive(WorkFactorMPI<scalar_t>& w,
                            const BLACSGrid* lg,
                            bool isroot, bool partial) override;

      void solve_fwd(const DistSubLeaf<scalar_t>& b,
                     WorkSolveMPI<scalar_t>& w,
                     bool partial, bool isroot) const override;
      void solve_bwd(DistSubLeaf<scalar_t>& x,
                     WorkSolveMPI<scalar_t>& w, bool isroot) const override;

      void apply_fwd(const DistSubLeaf<scalar_t>& B,
                     WorkApplyMPI<scalar_t>& w,
                     bool isroot, long long int flops) const override;
      void apply_bwd(const DistSubLeaf<scalar_t>& B, scalar_t beta,
                     DistSubLeaf<scalar_t>& C, WorkApplyMPI<scalar_t>& w,
                     bool isroot, long long int flops) const override;
      void applyT_fwd(const DistSubLeaf<scalar_t>& B,
                      WorkApplyMPI<scalar_t>& w,
                      bool isroot, long long int flops) const override;
      void applyT_bwd(const DistSubLeaf<scalar_t>& B, scalar_t beta,
                      DistSubLeaf<scalar_t>& C, WorkApplyMPI<scalar_t>& w,
                      bool isroot, long long int flops) const override;

      void extract_fwd(WorkExtractMPI<scalar_t>& w, const BLACSGrid* lg,
                       bool odiag) const override;
      void extract_bwd(std::vector<Triplet<scalar_t>>& triplets,
                       const BLACSGrid* lg,
                       WorkExtractMPI<scalar_t>& w) const override;
      void triplets_to_DistM(std::vector<Triplet<scalar_t>>& triplets,
                             DistM_t& B) const;
      void extract_fwd(WorkExtractBlocksMPI<scalar_t>& w,
                       const BLACSGrid* lg,
                       std::vector<bool>& odiag) const override;
      void extract_bwd(std::vector<std::vector<Triplet<scalar_t>>>& triplets,
                       const BLACSGrid* lg,
                       WorkExtractBlocksMPI<scalar_t>& w) const override;
      void triplets_to_DistM(std::vector<std::vector<Triplet<scalar_t>>>& triplets,
                             std::vector<DistM_t>& B) const;

      void redistribute_to_tree_to_buffers(const DistM_t& A,
                                           std::size_t Arlo, std::size_t Aclo,
                                           std::vector<std::vector<scalar_t>>& sbuf,
                                           int dest=0) override;
      void redistribute_to_tree_from_buffers(const DistM_t& A,
                                             std::size_t rlo, std::size_t clo,
                                             std::vector<scalar_t*>& pbuf)
        override;
      void delete_redistributed_input() override;

      void apply_UV_big(DistSubLeaf<scalar_t>& Theta, DistM_t& Uop,
                        DistSubLeaf<scalar_t>& Phi, DistM_t& Vop,
                        long long int& flops) const override;

      static int Pl(std::size_t n, std::size_t nl, std::size_t nr, int P) {
        return std::max
          (1, std::min(int(std::round(float(P) * nl / n)), P-1));
      }
      static int Pr(std::size_t n, std::size_t nl, std::size_t nr, int P) {
        return std::max(1, P - Pl(n, nl, nr, P));
      }
      int Pl() const {
        return Pl(this->rows(), child(0)->rows(),
                  child(1)->rows(), Ptotal());
      }
      int Pr() const {
        return Pr(this->rows(), child(0)->rows(),
                  child(1)->rows(), Ptotal());
      }

      template<typename T> friend
      void apply_HSS(Trans ta, const HSSMatrixMPI<T>& a,
                     const DistributedMatrix<T>& b, T beta,
                     DistributedMatrix<T>& c);
      friend class DistSamples<scalar_t>;

      using HSSMatrixBase<scalar_t>::child;
    };

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_MPI_HPP
