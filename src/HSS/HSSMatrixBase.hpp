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
 *
 */
/**
 * \file HSSMatrixBase.hpp
 * \brief This file contains the HSSMatrixBase class definition, an
 * abstract class for HSS matrix representation.
 */
#ifndef HSS_MATRIX_BASE_HPP
#define HSS_MATRIX_BASE_HPP

#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <functional>

#include "dense/DenseMatrix.hpp"
#include "misc/Triplet.hpp"
#include "HSSOptions.hpp"
#include "HSSExtra.hpp"
#include "structured/StructuredMatrix.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "dense/DistributedMatrix.hpp"
#include "HSSExtraMPI.hpp"
#include "HSSMatrixMPI.hpp"
#endif

namespace strumpack {
  namespace HSS {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t> class HSSMatrix;
#if defined(STRUMPACK_USE_MPI)
    template<typename scalar_t> class HSSMatrixMPI;
    template<typename scalar_t> class DistSubLeaf;
    template<typename scalar_t> class DistSamples;
#endif //defined(STRUMPACK_USE_MPI)
#endif //DOXYGEN_SHOULD_SKIP_THIS


    /**
     * \class HSSMatrixBase
     *
     * \brief Abstract base class for Hierarchically Semi-Separable
     * (HSS) matrices.
     *
     * This is for non-symmetric HSS matrices, but can be used with
     * symmetric matrices as well.
     *
     * \tparam scalar_t Can be float, double, std:complex<float> or
     * std::complex<double>.
     *
     * \see HSSMatrix, HSSMatrixMPI
     */
    template<typename scalar_t> class HSSMatrixBase
      : public structured::StructuredMatrix<scalar_t> {
      using real_t = typename RealType<scalar_t>::value_type;
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using elem_t = typename std::function
        <void(const std::vector<std::size_t>& I,
              const std::vector<std::size_t>& J, DenseM_t& B)>;
      using opts_t = HSSOptions<scalar_t>;
#if defined(STRUMPACK_USE_MPI)
      using DistM_t = DistributedMatrix<scalar_t>;
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      using delem_t = typename std::function
        <void(const std::vector<std::size_t>& I,
              const std::vector<std::size_t>& J, DistM_t& B)>;
#endif //defined(STRUMPACK_USE_MPI)

    public:
      /**
       * Construct an m x n HSS matrix, not initialized.
       *
       * \param m number of rows
       * \param n number of columns
       * \param active Denote if this matrix (node) is active on this
       * process.
       */
      HSSMatrixBase(std::size_t m, std::size_t n, bool active);

      /**
       * Default virtual destructor.
       */
      virtual ~HSSMatrixBase() = default;

      /**
       * Copy constructor.
       * \param other HSS matrix to copy.
       */
      HSSMatrixBase(const HSSMatrixBase<scalar_t>& other);

      /**
       * Copy assignment operator, makes a deep copy.
       * \param other HSS matrix to copy.
       * \return reference to this HSS matrix.
       */
      HSSMatrixBase<scalar_t>& operator=(const HSSMatrixBase<scalar_t>& other);

      /**
       * Move constructor.
       * \param h HSS matrix to move from, h will be emptied.
       */
      HSSMatrixBase(HSSMatrixBase&& h) = default;

      /**
       * Move assignment operator.
       * \param h HSS matrix to move from, h will be emptied.
       * \return reference to this HSS matrix.
       */
      HSSMatrixBase& operator=(HSSMatrixBase&& h) = default;

      /**
       * Clone this HSS matrix.
       * TODO delete this??
       *
       * \return std::unique_ptr to a clone of this HSS matrix.
       */
      virtual std::unique_ptr<HSSMatrixBase<scalar_t>> clone() const = 0;

      /**
       * Returns the dimensions of this HSS matrix, as a pair.
       *
       * \return pair with number of rows and columns of this HSS
       * matrix.
       */
      std::pair<std::size_t,std::size_t> dims() const {
        return std::make_pair(rows_, cols_);
      }

      /**
       * Return the number of rows in this HSS matrix.
       * \return number of rows
       */
      std::size_t rows() const override { return rows_; }

      /**
       * Return the number of columns in this HSS matrix.
       * \return number of columns
       */
      std::size_t cols() const override { return cols_; }

      /**
       * Check whether this node of the HSS tree is a leaf.
       * \return true if this node is a leaf, false otherwise.
       */
      bool leaf() const { return ch_.empty(); }

      virtual std::size_t factor_nonzeros() const;

      /**
       * Return a const reference to the child (0, or 1) of this HSS
       * matrix. This is only valid when !this->leaf(). It is assumed
       * that a non-leaf node always has exactly 2 children.
       *
       * \param c Number of the child, should be 0 or 1, for the left
       * or the right child.
       * \return Const reference to the child (HSSMatrixBase).
       */
      const HSSMatrixBase<scalar_t>& child(int c) const {
        assert(c>=0 && c<int(ch_.size())); return *(ch_[c]);
      }

      /**
       * Return a reference to the child (0, or 1) of this HSS
       * matrix. This is only valid when !this->leaf(). It is assumed
       * that a non-leaf node always has exactly 2 children.
       *
       * \param c Number of the child, should be 0 or 1, for the left
       * or the right child.
       * \return Reference to the child (HSSMatrixBase).
       */
      HSSMatrixBase<scalar_t>& child(int c) {
        assert(c>=0 && c<int(ch_.size())); return *(ch_[c]);
      }

      /**
       * Check whether the HSS matrix was compressed.
       *
       * \return True if this HSS matrix was succesfully compressed,
       * false otherwise.
       *
       * \see is_untouched
       */
      bool is_compressed() const {
        return U_state_ == State::COMPRESSED &&
          V_state_ == State::COMPRESSED;
      }

      /**
       * Check whether the HSS compression was started for this
       * matrix.
       *
       * \return True if HSS compression was not started yet, false
       * otherwise. False may mean that compression was started but
       * failed, or that compression succeeded.
       *
       * \see is_compressed
       */
      bool is_untouched() const {
        return U_state_ == State::UNTOUCHED &&
          V_state_ == State::UNTOUCHED;
      }

      /**
       * Check if this HSS matrix (or node in the HSS tree) is active
       * on this rank.
       *
       * \return True if this node is active, false otherwise.
       */
      bool active() const { return active_; }

      /**
       * Return the number of levels in the HSS matrix.
       *
       * \return Number of HSS levels (>= 1).
       */
      virtual std::size_t levels() const = 0;

      /**
       * Print info about this HSS matrix, such as tree info, ranks,
       * etc.
       *
       * \param out Stream to print to, defaults to std::cout
       * \param roff Row offset of top left corner, defaults to
       * 0. This is used to recursively print the tree, you can leave
       * this at the default.
       * \param coff Column offset of top left corner, defaults to
       * 0. This is used to recursively print the tree, you can leave
       * this at the default.
       */
      virtual void print_info(std::ostream &out=std::cout,
                              std::size_t roff=0,
                              std::size_t coff=0) const = 0;

      /**
       * Set the depth of openmp nested tasks. This can be used to
       * limit the number of tasks to spawn in the HSS routines, which
       * is can reduce task creation overhead.  This is used in the
       * sparse solver when multiple HSS matrices are created from
       * within multiple openmp tasks. The HSS routines all use openmp
       * tasking to traverse the HSS tree and for parallelism within
       * the HSS nodes as well.
       */
      void set_openmp_task_depth(int depth) { openmp_task_depth_ = depth; }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
      virtual void delete_trailing_block() { if (ch_.size()==2) ch_.resize(1); }
      virtual void reset() {
        U_state_ = V_state_ = State::UNTOUCHED;
        U_rank_ = U_rows_ = V_rank_ = V_rows_ = 0;
        for (auto& c : ch_) c->reset();
      }
#endif

      /**
       * Apply a shift to the diagonal of this matrix. Ie, this +=
       * sigma * I, with I the identity matrix. Call this after
       * compression.
       *
       * \param sigma Shift to be applied to the diagonal.
       */
      virtual void shift(scalar_t sigma) override = 0;

      /**
       * Internal routine to draw this HSS matrix. Do not use this
       * directly. Use HSS::draw.
       *
       * \see HSS::draw
       */
      virtual void draw(std::ostream& of,
                        std::size_t rlo,
                        std::size_t clo) const {}

#if defined(STRUMPACK_USE_MPI)
      virtual void forward_solve(WorkSolveMPI<scalar_t>& w,
                                 const DistM_t& b, bool partial) const;
      virtual void backward_solve(WorkSolveMPI<scalar_t>& w,
                                  DistM_t& x) const;

      virtual const BLACSGrid* grid() const { return nullptr; }
      virtual const BLACSGrid* grid(const BLACSGrid* local_grid) const {
        return active() ? local_grid : nullptr;
      }
      virtual const BLACSGrid* grid_local() const { return nullptr; }
      virtual int Ptotal() const { return 1; }
      virtual int Pactive() const { return 1; }

      virtual void to_block_row(const DistM_t& A, DenseM_t& sub_A,
                                DistM_t& leaf_A) const;
      virtual void allocate_block_row(int d, DenseM_t& sub_A,
                                      DistM_t& leaf_A) const;
      virtual void from_block_row(DistM_t& A, const DenseM_t& sub_A,
                                  const DistM_t& leaf_A,
                                  const BLACSGrid* lg) const;
#endif //defined(STRUMPACK_USE_MPI)

    protected:
      std::size_t rows_, cols_;

      // TODO store children array in the sub-class???
      std::vector<std::unique_ptr<HSSMatrixBase<scalar_t>>> ch_;
      State U_state_, V_state_;
      int openmp_task_depth_;
      bool active_;

      int U_rank_ = 0, U_rows_ = 0, V_rank_ = 0, V_rows_ = 0;

      // Used to redistribute the original 2D block cyclic matrix
      // according to the HSS tree
      DenseM_t Asub_;

      HSSFactors<scalar_t> ULV_;
#if defined(STRUMPACK_USE_MPI)
      HSSFactorsMPI<scalar_t> ULV_mpi_;
#endif //defined(STRUMPACK_USE_MPI)

      virtual std::size_t U_rank() const { return U_rank_; }
      virtual std::size_t V_rank() const { return V_rank_; }
      virtual std::size_t U_rows() const { return U_rows_; }
      virtual std::size_t V_rows() const { return V_rows_; }

      virtual void
      compress_recursive_original(DenseM_t& Rr, DenseM_t& Rc,
                                  DenseM_t& Sr, DenseM_t& Sc,
                                  const elem_t& Aelem, const opts_t& opts,
                                  WorkCompress<scalar_t>& w,
                                  int dd, int depth) {}
      virtual void
      compress_recursive_stable(DenseM_t& Rr, DenseM_t& Rc,
                                DenseM_t& Sr, DenseM_t& Sc,
                                const elem_t& Aelem, const opts_t& opts,
                                WorkCompress<scalar_t>& w,
                                int d, int dd, int depth) {}
      virtual void
      compress_level_original(DenseM_t& Rr, DenseM_t& Rc,
                              DenseM_t& Sr, DenseM_t& Sc,
                              const opts_t& opts, WorkCompress<scalar_t>& w,
                              int dd, int lvl, int depth) {}
      virtual void
      compress_level_stable(DenseM_t& Rr, DenseM_t& Rc,
                            DenseM_t& Sr, DenseM_t& Sc,
                            const opts_t& opts, WorkCompress<scalar_t>& w,
                            int d, int dd, int lvl, int depth) {}
      virtual void
      compress_recursive_ann(DenseMatrix<std::uint32_t>& ann,
                             DenseMatrix<real_t>& scores,
                             const elem_t& Aelem, const opts_t& opts,
                             WorkCompressANN<scalar_t>& w, int depth) {}

      virtual void
      get_extraction_indices(std::vector<std::vector<std::size_t>>& I,
                             std::vector<std::vector<std::size_t>>& J,
                             const std::pair<std::size_t,std::size_t>& off,
                             WorkCompress<scalar_t>& w,
                             int& self, int lvl) {}

      virtual void
      get_extraction_indices(std::vector<std::vector<std::size_t>>& I,
                             std::vector<std::vector<std::size_t>>& J,
                             std::vector<DenseM_t*>& B,
                             const std::pair<std::size_t,std::size_t>& off,
                             WorkCompress<scalar_t>& w,
                             int& self, int lvl) {}
      virtual void extract_D_B(const elem_t& Aelem, const opts_t& opts,
                               WorkCompress<scalar_t>& w, int lvl) {}

      virtual void factor_recursive(WorkFactor<scalar_t>& w,
                                    bool isroot, bool partial,
                                    int depth) {}

      virtual void apply_fwd(const DenseM_t& b, WorkApply<scalar_t>& w,
                             bool isroot, int depth,
                             std::atomic<long long int>& flops) const {}
      virtual void apply_bwd(const DenseM_t& b, scalar_t beta,
                             DenseM_t& c, WorkApply<scalar_t>& w,
                             bool isroot, int depth,
                             std::atomic<long long int>& flops) const {}
      virtual void applyT_fwd(const DenseM_t& b, WorkApply<scalar_t>& w,
                              bool isroot, int depth,
                              std::atomic<long long int>& flops) const {}
      virtual void applyT_bwd(const DenseM_t& b, scalar_t beta,
                              DenseM_t& c, WorkApply<scalar_t>& w,
                              bool isroot, int depth,
                              std::atomic<long long int>& flops) const {}

      virtual void forward_solve(WorkSolve<scalar_t>& w,
                                 const DenseMatrix<scalar_t>& b,
                                 bool partial) const {}
      virtual void backward_solve(WorkSolve<scalar_t>& w,
                                  DenseMatrix<scalar_t>& b) const {}
      virtual void solve_fwd(const DenseM_t& b,
                             WorkSolve<scalar_t>& w, bool partial,
                             bool isroot, int depth) const {}
      virtual void solve_bwd(DenseM_t& x, WorkSolve<scalar_t>& w,
                             bool isroot, int depth) const {}

      virtual void extract_fwd(WorkExtract<scalar_t>& w,
                               bool odiag, int depth) const {}
      virtual void extract_bwd(DenseMatrix<scalar_t>& B,
                               WorkExtract<scalar_t>& w,
                               int depth) const {}
      virtual void extract_bwd(std::vector<Triplet<scalar_t>>& triplets,
                               WorkExtract<scalar_t>& w, int depth) const {}

      virtual void apply_UV_big(DenseM_t& Theta, DenseM_t& Uop,
                                DenseM_t& Phi, DenseM_t& Vop,
                                const std::pair<std::size_t,std::size_t>& offset,
                                int depth,
                                std::atomic<long long int>& flops) const {}
      virtual void apply_UtVt_big(const DenseM_t& A, DenseM_t& UtA,
                                  DenseM_t& VtA,
                                  const std::pair<std::size_t, std::size_t>& offset,
                                  int depth,
                                  std::atomic<long long int>& flops) const {}

      virtual void dense_recursive(DenseM_t& A, WorkDense<scalar_t>& w,
                                   bool isroot, int depth) const {}

      virtual void read(std::ifstream& os) {
        std::cerr << "ERROR read_HSS_node not implemented" << std::endl;
      }
      virtual void write(std::ofstream& os) const {
        std::cerr << "ERROR write_HSS_node not implemented" << std::endl;
      }

      friend class HSSMatrix<scalar_t>;

#if defined(STRUMPACK_USE_MPI)
      using delemw_t = typename std::function
        <void(const std::vector<std::size_t>& I,
              const std::vector<std::size_t>& J,
              DistM_t& B, DistM_t& A,
              std::size_t rlo, std::size_t clo,
              MPI_Comm comm)>;


      virtual void
      compress_recursive_original(DistSamples<scalar_t>& RS,
                                  const delemw_t& Aelem,
                                  const opts_t& opts,
                                  WorkCompressMPI<scalar_t>& w, int dd);
      virtual void
      compress_recursive_stable(DistSamples<scalar_t>& RS,
                                const delemw_t& Aelem,
                                const opts_t& opts,
                                WorkCompressMPI<scalar_t>& w, int d, int dd);
      virtual void
      compress_level_original(DistSamples<scalar_t>& RS, const opts_t& opts,
                              WorkCompressMPI<scalar_t>& w, int dd, int lvl);
      virtual void
      compress_level_stable(DistSamples<scalar_t>& RS, const opts_t& opts,
                            WorkCompressMPI<scalar_t>& w,
                            int d, int dd, int lvl);
      virtual void
      compress_recursive_ann(DenseMatrix<std::uint32_t>& ann,
                             DenseMatrix<real_t>& scores,
                             const delemw_t& Aelem,
                             WorkCompressMPIANN<scalar_t>& w,
                             const opts_t& opts, const BLACSGrid* lg);

      virtual void
      get_extraction_indices(std::vector<std::vector<std::size_t>>& I,
                             std::vector<std::vector<std::size_t>>& J,
                             WorkCompressMPI<scalar_t>& w,
                             int& self, int lvl);
      virtual void
      get_extraction_indices(std::vector<std::vector<std::size_t>>& I,
                             std::vector<std::vector<std::size_t>>& J,
                             std::vector<DistMW_t>& B,
                             const BLACSGrid* lg,
                             WorkCompressMPI<scalar_t>& w,
                             int& self, int lvl);
      virtual void extract_D_B(const delemw_t& Aelem, const BLACSGrid* lg,
                               const opts_t& opts,
                               WorkCompressMPI<scalar_t>& w, int lvl);

      virtual void apply_fwd(const DistSubLeaf<scalar_t>& B,
                             WorkApplyMPI<scalar_t>& w,
                             bool isroot, long long int flops) const;
      virtual void apply_bwd(const DistSubLeaf<scalar_t>& B, scalar_t beta,
                             DistSubLeaf<scalar_t>& C,
                             WorkApplyMPI<scalar_t>& w,
                             bool isroot, long long int flops) const;
      virtual void applyT_fwd(const DistSubLeaf<scalar_t>& B,
                              WorkApplyMPI<scalar_t>& w,
                              bool isroot, long long int flops) const;
      virtual void applyT_bwd(const DistSubLeaf<scalar_t>& B, scalar_t beta,
                              DistSubLeaf<scalar_t>& C,
                              WorkApplyMPI<scalar_t>& w,
                              bool isroot, long long int flops) const;

      virtual void factor_recursive(WorkFactorMPI<scalar_t>& w,
                                    const BLACSGrid* lg, bool isroot,
                                    bool partial);

      virtual void solve_fwd(const DistSubLeaf<scalar_t>& b,
                             WorkSolveMPI<scalar_t>& w,
                             bool partial, bool isroot) const;
      virtual void solve_bwd(DistSubLeaf<scalar_t>& x,
                             WorkSolveMPI<scalar_t>& w, bool isroot) const;

      virtual void extract_fwd(WorkExtractMPI<scalar_t>& w,
                               const BLACSGrid* lg, bool odiag) const;
      virtual void extract_bwd(std::vector<Triplet<scalar_t>>& triplets,
                               const BLACSGrid* lg,
                               WorkExtractMPI<scalar_t>& w) const;
      virtual void extract_fwd(WorkExtractBlocksMPI<scalar_t>& w,
                               const BLACSGrid* lg,
                               std::vector<bool>& odiag) const;
      virtual void extract_bwd(std::vector<std::vector<Triplet<scalar_t>>>& triplets,
                               const BLACSGrid* lg,
                               WorkExtractBlocksMPI<scalar_t>& w) const;

      virtual void apply_UV_big(DistSubLeaf<scalar_t>& Theta, DistM_t& Uop,
                                DistSubLeaf<scalar_t>& Phi, DistM_t& Vop,
                                long long int& flops) const;

      virtual void
      redistribute_to_tree_to_buffers(const DistM_t& A,
                                      std::size_t Arlo, std::size_t Aclo,
                                      std::vector<std::vector<scalar_t>>& sbuf,
                                      int dest);
      virtual void
      redistribute_to_tree_from_buffers(const DistM_t& A,
                                        std::size_t Arlo, std::size_t Aclo,
                                        std::vector<scalar_t*>& pbuf);
      virtual void delete_redistributed_input();

      friend class HSSMatrixMPI<scalar_t>;
#endif //defined(STRUMPACK_USE_MPI)
    };

  } // end namespace HSS
} // end namespace strumpack


#endif // HSS_MATRIX_BASE_HPP
