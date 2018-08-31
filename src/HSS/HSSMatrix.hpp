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
/*! \file HSSMatrix.hpp
 *
 * \brief This file contains the HSSMatrix class definition as well as
 * implementations for a number of it's member routines. Other member
 * routines are implemented in files such as HSSMatrix.apply.hpp,
 * HSSMatrix.factor.hpp etc.
 */
#ifndef HSS_MATRIX_HPP
#define HSS_MATRIX_HPP

#include <cassert>

#include "HSSPartitionTree.hpp"
#include "HSSBasisID.hpp"
#include "HSSOptions.hpp"
#include "HSSExtra.hpp"
#include "HSSMatrixBase.hpp"

namespace strumpack {
  namespace HSS {

    // forward declaration
    template<typename scalar_t> class HSSMatrixMPI;


    /**
     * \class HSSMatrix
     *
     * \brief HSSMatrix is a class to represent a sequential/threaded
     * Hierarchically Semi-Separable matrix.
     *
     * This is for non-symmetric matrices, but can be used with
     * symmetric matrices as well. This class inherits from
     * HSSMatrixBase. The template type can be float, double,
     * std:complex<float> or std::complex<double>. For the distributed
     * memory code for HSS matrices, see HSSMatrixMPI. An HSS matrix
     * is represented recursively, using 2 children which are also
     * HSSMatrix object, except at the lowest level (the leafs), where
     * the HSSMatrix has no children. Hence, the tree representing the
     * HSS matrix is always a binary tree, but not necessarily a
     * complete binary tree.
     */
    template<typename scalar_t> class HSSMatrix
      : public HSSMatrixBase<scalar_t> {
      using real_t = typename RealType<scalar_t>::value_type;
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using elem_t = typename std::function
        <void(const std::vector<std::size_t>& I,
              const std::vector<std::size_t>& J, DenseM_t& B)>;
      using mult_t = typename std::function
        <void(DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc)>;
      using opts_t = HSSOptions<scalar_t>;

    public:
      /**
       * Default constructor, constructs an empty 0 x 0 matrix.
       */
      HSSMatrix();

      /**
       * Construct an HSS representation for a dense matrix A. The HSS
       * tree will be constructed by splitting the row/column set
       * evenly and recursively until the leafs in the HSS tree are
       * smaller than opts.leaf_size(). Alternative constructors can
       * be used to specify a specific HSS partitioning
       * tree. Internaly, this will call the appropriate compress
       * routine to construct the HSS representation, using the
       * options (such as compression tolerance, adaptive compression
       * scheme etc) as specified in the HSSOptions opts object.
       *
       * \param A dense matrix (unmodified) to compress as HSS
       * \param opts object containing a number of options for HSS
       * compression
       * \see DenseMatrix
       * \see HSSOptions
       */
      HSSMatrix(const DenseM_t& A, const opts_t& opts);

      /**
       * Construct an HSS representation for an m x n matrix. The HSS
       * tree will be constructed by splitting the row/column set
       * evenly and recursively until the leafs in the HSS tree are
       * smaller than opts.leaf_size(). After construction, the HSS
       * matrix will be empty, and can be filled by calling one of the
       * compress member routines. Alternative constructors can be
       * used to specify a specific HSS partitioning tree.
       *
       * \param m number of rows in the constructed HSS matrix
       * \param n number of rows in the constructed HSS matrix
       * \param opts object containing a number of options for HSS
       * compression
       * \see HSSOptions
       */
      HSSMatrix(std::size_t m, std::size_t n, const opts_t& opts);

      /**
       * Construct an HSS matrix using a specified HSS tree. After
       * construction, the HSS matrix will be empty, and can be filled
       * by calling one of the compress member routines.
       *
       * \param t tree specifying the HSS matrix partitioning
       * \param opts object containing a number of options for HSS
       * compression
       * \see HSSOptions
       */
      HSSMatrix(const HSSPartitionTree& t, const opts_t& opts);

      /**
       * Copy constructor. Copying an HSSMatrix can be an expensive
       * operation.
       * \param other HSS matrix to be copied
       */
      HSSMatrix(const HSSMatrix<scalar_t>& other);

      /**
       * Copy assignment operator. Copying an HSSMatrix can be an
       * expensive operation.
       * \param other HSS matrix to be copied
       */
      HSSMatrix<scalar_t>& operator=(const HSSMatrix<scalar_t>& other);

      /**
       * Move constructor.
       * \param other HSS matrix to be moved from, will be emptied
       */
      HSSMatrix(HSSMatrix<scalar_t>&& other) = default;

      /**
       * Move assignment operator.
       * \param other HSS matrix to be moved from, will be emptied
       */
      HSSMatrix<scalar_t>& operator=(HSSMatrix<scalar_t>&& other) = default;

      /**
       * Create a clone of this matrix.
       * TODO remove this!!???
       */
      std::unique_ptr<HSSMatrixBase<scalar_t>> clone() const override;

      /**
       * Return a const raw (non-owning) pointer to child c of this
       * HSS matrix. A child of an HSS matrix is itself an HSS
       * matrix. The value of c should be 0 or 1, and this HSS matrix
       * should not be a leaf!
       */
      const HSSMatrix<scalar_t>* child(int c) const {
        return dynamic_cast<HSSMatrix<scalar_t>*>(this->_ch[c].get());
      }

      /**
       * Return a raw (non-owning) pointer to child c of this HSS
       * matrix. A child of an HSS matrix is itself an HSS matrix. The
       * value of c should be 0 or 1, and this HSS matrix should not
       * be a leaf!
       */
      HSSMatrix<scalar_t>* child(int c) {
        return dynamic_cast<HSSMatrix<scalar_t>*>(this->_ch[c].get());
      }

      /**
       * Initialize this HSS matrix as the compressed HSS
       * representation of a given dense matrix. The HSS matrix should
       * have been constructed with the proper sizes, i.e., rows() ==
       * A.rows() and cols() == A.cols().  Internaly, this will call
       * the appropriate compress routine to construct the HSS
       * representation, using the options (such as compression
       * tolerance, adaptive compression scheme etc) as specified in
       * the HSSOptions opts object.
       *
       * \param A dense matrix (unmodified) to compress as HSS
       * \param opts object containing a number of options for HSS
       * compression
       * \see DenseMatrix
       * \see HSSOptions
       */
      void compress(const DenseM_t& A, const opts_t& opts);

      /**
       * Initialize this HSS matrix as the compressed HSS
       * representation of a given dense matrix. The HSS matrix should
       * have been constructed with the proper sizes, i.e., rows() ==
       * A.rows() and cols() == A.cols().  Internaly, this will call
       * the appropriate compress routine to construct the HSS
       * representation, using the options (such as compression
       * tolerance, adaptive compression scheme etc) as specified in
       * the HSSOptions opts object.
       *
       * \param A dense matrix (unmodified) to compress as HSS
       * \param opts object containing a number of options for HSS
       * compression
       * \see DenseMatrix
       * \see HSSOptions
       */
      void compress (const mult_t& Amult, const elem_t& Aelem, const opts_t& opts);


      //void compress_ann (DenseM_t& ann, DenseM_t& scores, const elem_t& Aelem, const opts_t& opts);
      void compress_ann (DenseM_t&  ann, DenseM_t& scores, const elem_t& Aelem, const opts_t& opts);


      /**
       * Reset the matrix to an empty, 0 x 0 matrix, freeing up all
       * it's memory.
       */
      void reset() override;

      /**
       * Compute a ULV factorization of this matrix. The factors are
       * returned as an HSSFactors object, the current HSS matrix is
       * not modified.
       */
      HSSFactors<scalar_t> factor() const;

      /**
       * Compute a partial ULV factorization of this matrix. Only the
       * left child is factored. This is not similar to calling
       * child(0)->factor(), except that the HSSFactors resulting from
       * calling partial_factor can be used to compute the Schur
       * complement. The factors are returned as an HSSFactors object,
       * the current HSS matrix is not modified.
       *
       * \see Schur_update, Schur_product_direct and
       * Schur_product_indirect
       */
      HSSFactors<scalar_t> partial_factor() const;

      /**
       * Solve a linear system with the ULV factorization of this
       * HSSMatrix. The right hand side vector (or matrix) b is
       * overwritten with the solution x of Ax=b.
       *
       * \param ULV ULV factorization of this matrix
       * \param b on input, the right hand side vector, on output the
       * solution of A x = b (with A this HSS matrix). The vector b
       * should be b.rows() == cols().
       * \see factor
       */
      void solve(const HSSFactors<scalar_t>& ULV, DenseM_t& b) const;

      /**
       * Perform only the forward phase of the ULV linear solve. This
       * is for advanced use only, typically to be used in combination
       * with partial_factor. You should really just use factor/solve
       * when possible.
       *
       * \param ULV  ULV factorization of this matrix
       * \param w temporary working storage, to pass information from
       * forward_solve to backward_solve
       * \param b on input, the right hand side vector, on output the
       * intermediate solution. The vector b
       * should be b.rows() == cols().
       * \param partial denotes wether the matrix was fully or
       * partially factored
       * \see factor, partial_factor, backward_solve
       */
      void forward_solve
      (const HSSFactors<scalar_t>& ULV, WorkSolve<scalar_t>& w,
       const DenseM_t& b, bool partial) const;

      /**
       * Perform only the backward phase of the ULV linear solve. This
       * is for advanced use only, typically to be used in combination
       * with partial_factor. You should really just use factor/solve
       * when possible.
       *
       * \param ULV  ULV factorization of this matrix
       * \param w temporary working storage, to pass information from
       * forward_solve to backward_solve
       * \param b on input, the vector obtained from forward_solve, on
       * output the solution of A x = b (with A this HSS matrix). The
       * vector b should be b.rows() == cols().
       * \see factor, partial_factor, backward_solve
       */
      void backward_solve
      (const HSSFactors<scalar_t>& ULV,
       WorkSolve<scalar_t>& w, DenseM_t& x) const;

      DenseM_t apply(const DenseM_t& b) const;
      DenseM_t applyC(const DenseM_t& b) const;

      scalar_t get(std::size_t i, std::size_t j) const;
      DenseM_t extract
      (const std::vector<std::size_t>& I,
       const std::vector<std::size_t>& J) const;
      void extract_add
      (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
       DenseM_t& B) const;

      /**
       * Compute a Schur complement update... TODO
       */
      void Schur_update
      (const HSSFactors<scalar_t>& f, DenseM_t& Theta,
       DenseM_t& DUB01, DenseM_t& Phi) const;
      /**
       * TODO
       */
      void Schur_product_direct
      (const HSSFactors<scalar_t>& f,
       const DenseM_t& Theta, const DenseM_t& DUB01,
       const DenseM_t& Phi, const DenseM_t&_ThetaVhatC_or_VhatCPhiC,
       const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc) const;
      /**
       * TODO
       */
      void Schur_product_indirect
      (const HSSFactors<scalar_t>& f, const DenseM_t& DUB01,
       const DenseM_t& R1, const DenseM_t& R2, const DenseM_t& Sr2,
       const DenseM_t& Sc2, DenseM_t& Sr, DenseM_t& Sc) const;

      std::size_t rank() const;
      std::size_t memory() const;
      std::size_t nonzeros() const;
      std::size_t levels() const;
      void print_info
      (std::ostream &out=std::cout,
       std::size_t roff=0, std::size_t coff=0) const;

      DenseM_t dense() const;
      void delete_trailing_block() override;

      void shift(scalar_t sigma) override;

      void draw(std::ostream& of, std::size_t rlo=0, std::size_t clo=0) const;

    protected:
      HSSMatrix
      (std::size_t m, std::size_t n, const opts_t& opts, bool active);
      HSSMatrix(const HSSPartitionTree& t, const opts_t& opts, bool active);

      HSSBasisID<scalar_t> _U;
      HSSBasisID<scalar_t> _V;
      DenseM_t _D;
      DenseM_t _B01;
      DenseM_t _B10;

      void compress_original(const DenseM_t& A, const opts_t& opts);
      void compress_original
      (const mult_t& Amult, const elem_t& Aelem, const opts_t& opts);
      void compress_stable(const DenseM_t& A, const opts_t& opts);
      void compress_stable
      (const mult_t& Amult, const elem_t& Aelem, const opts_t& opts);
      void compress_hard_restart(const DenseM_t& A, const opts_t& opts);
      void compress_hard_restart
      (const mult_t& Amult, const elem_t& Aelem, const opts_t& opts);

      //NEW
      // void compress_ann
      // (const mult_t& Amult, const elem_t& Aelem,
      //  data_points, ann_indices, scores_indices,
      //  //const kopts_t& kopts);
      //  const opts_t& opts);

      void compress_recursive_ann
      (DenseM_t& ann, DenseM_t&  scores,
       const elem_t& Aelem, const opts_t& opts,
       WorkCompressANN<scalar_t>& w, int d) override;

      void compute_local_samples_ann
      (DenseM_t& ann, DenseM_t& scores, WorkCompressANN<scalar_t>& w,
       const elem_t& Aelem, int d);


      void compress_recursive_original
      (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
       const elem_t& Aelem, const opts_t& opts,
       WorkCompress<scalar_t>& w, int dd, int depth);
      void compress_recursive_stable
      (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
       const elem_t& Aelem, const opts_t& opts,
       WorkCompress<scalar_t>& w, int d, int dd, int depth);
      void compute_local_samples
      (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
       WorkCompress<scalar_t>& w, int d0, int d, int depth);
      bool compute_U_V_bases
      (DenseM_t& Sr, DenseM_t& Sc, const opts_t& opts,
       WorkCompress<scalar_t>& w, int d, int depth);
      bool compute_U_V_bases_ANN
      (DenseM_t& Sr, DenseM_t& Sc, const opts_t& opts,
       WorkCompressANN<scalar_t>& w, int d, int depth);
      void compute_U_basis_stable
      (DenseM_t& Sr, const opts_t& opts,
       WorkCompress<scalar_t>& w, int d, int dd, int depth);
      void compute_V_basis_stable
      (DenseM_t& Sc, const opts_t& opts,
       WorkCompress<scalar_t>& w, int d, int dd, int depth);
      void reduce_local_samples
      (DenseM_t& Rr, DenseM_t& Rc, WorkCompress<scalar_t>& w,
       int d0, int d, int depth);
      bool update_orthogonal_basis
      (const opts_t& opts, scalar_t& r_max_0,
       const DenseM_t& S, DenseM_t& Q, int d, int dd,
       bool untouched, int L, int depth);
      void set_U_full_rank(WorkCompress<scalar_t>& w);
      void set_V_full_rank(WorkCompress<scalar_t>& w);

      void compress_level_original
      (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
       const opts_t& opts, WorkCompress<scalar_t>& w,
       int dd, int lvl, int depth);
      void compress_level_stable
      (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
       const opts_t& opts, WorkCompress<scalar_t>& w,
       int d, int dd, int lvl, int depth);
      void get_extraction_indices
      (std::vector<std::vector<std::size_t>>& I,
       std::vector<std::vector<std::size_t>>& J,
       const std::pair<std::size_t,std::size_t>& off,
       WorkCompress<scalar_t>& w, int& self, int lvl);
      void get_extraction_indices
      (std::vector<std::vector<std::size_t>>& I,
       std::vector<std::vector<std::size_t>>& J, std::vector<DenseM_t*>& B,
       const std::pair<std::size_t,std::size_t>& off,
       WorkCompress<scalar_t>& w, int& self, int lvl);
      void extract_D_B
      (const elem_t& Aelem, const opts_t& opts,
       WorkCompress<scalar_t>& w, int lvl);

      void factor_recursive
      (HSSFactors<scalar_t>& ULV, WorkFactor<scalar_t>& w,
       bool isroot, bool partial, int depth) const;

      void apply_fwd
      (const DenseM_t& b, WorkApply<scalar_t>& w, bool isroot,
       int depth, std::atomic<long long int>& flops) const override;
      void apply_bwd
      (const DenseM_t& b, scalar_t beta, DenseM_t& c,
       WorkApply<scalar_t>& w, bool isroot, int depth,
       std::atomic<long long int>& flops) const override;
      void applyT_fwd
      (const DenseM_t& b, WorkApply<scalar_t>& w, bool isroot,
       int depth, std::atomic<long long int>& flops) const override;
      void applyT_bwd
      (const DenseM_t& b, scalar_t beta, DenseM_t& c,
       WorkApply<scalar_t>& w, bool isroot, int depth,
       std::atomic<long long int>& flops) const override;

      void solve_fwd
      (const HSSFactors<scalar_t>& ULV, const DenseM_t& b,
       WorkSolve<scalar_t>& w,
       bool partial, bool isroot, int depth) const;
      void solve_bwd
      (const HSSFactors<scalar_t>& ULV, DenseM_t& x,
       WorkSolve<scalar_t>& w,
       bool isroot, int depth) const;

      void extract_fwd(WorkExtract<scalar_t>& w, bool odiag, int depth) const;
      void extract_bwd
      (DenseM_t& B, WorkExtract<scalar_t>& w, int depth) const;
      void extract_bwd
      (std::vector<Triplet<scalar_t>>& triplets,
       WorkExtract<scalar_t>& w, int depth) const;
      void extract_bwd_internal(WorkExtract<scalar_t>& w, int depth) const;

      void apply_UV_big
      (DenseM_t& Theta, DenseM_t& Uop, DenseM_t& Phi,
       DenseM_t& Vop, const std::pair<std::size_t, std::size_t>& offset,
       int depth, std::atomic<long long int>& flops) const override;
      void apply_UtVt_big
      (const DenseM_t& A, DenseM_t& UtA, DenseM_t& VtA,
       const std::pair<std::size_t, std::size_t>& offset,
       int depth, std::atomic<long long int>& flops) const override;

      void dense_recursive
      (DenseM_t& A, WorkDense<scalar_t>& w, bool isroot, int depth) const;

      template<typename T> friend
      void apply_HSS
      (Trans ta, const HSSMatrix<T>& a, const DenseMatrix<T>& b,
       T beta, DenseMatrix<T>& c);
      template<typename T> friend
      void draw(const HSSMatrix<T>& H, const std::string& name);

      friend class HSSMatrixMPI<scalar_t>;
    };

    template<typename scalar_t>
    HSSMatrix<scalar_t>::HSSMatrix() : HSSMatrixBase<scalar_t>(0, 0, true) {}

    template<typename scalar_t>
    HSSMatrix<scalar_t>::HSSMatrix
    (const DenseMatrix<scalar_t>& A, const opts_t& opts)
      : HSSMatrix<scalar_t>(A.rows(), A.cols(), opts) {
      compress(A, opts);
    }

    template<typename scalar_t>
    HSSMatrix<scalar_t>::HSSMatrix
    (std::size_t m, std::size_t n, const opts_t& opts)
      : HSSMatrix<scalar_t>(m, n, opts, true) { }

    template<typename scalar_t> HSSMatrix<scalar_t>::HSSMatrix
    (std::size_t m, std::size_t n, const opts_t& opts, bool active)
      : HSSMatrixBase<scalar_t>(m, n, active) {
      if (!active) return;
      if (m > std::size_t(opts.leaf_size()) ||
          n > std::size_t(opts.leaf_size())) {
        this->_ch.reserve(2);
        this->_ch.emplace_back(new HSSMatrix<scalar_t>(m/2, n/2, opts));
        this->_ch.emplace_back(new HSSMatrix<scalar_t>(m-m/2, n-n/2, opts));
      }
    }

    template<typename scalar_t> HSSMatrix<scalar_t>::HSSMatrix
    (const HSSPartitionTree& t, const opts_t& opts, bool active)
      : HSSMatrixBase<scalar_t>(t.size, t.size, active) {
      if (!active) return;
      if (!t.c.empty()) {
        assert(t.c.size() == 2);
        this->_ch.reserve(2);
        this->_ch.emplace_back(new HSSMatrix<scalar_t>(t.c[0], opts));
        this->_ch.emplace_back(new HSSMatrix<scalar_t>(t.c[1], opts));
      }
    }

    template<typename scalar_t>
    HSSMatrix<scalar_t>::HSSMatrix
    (const HSSPartitionTree& t, const opts_t& opts)
      : HSSMatrix<scalar_t>(t, opts, true) { }

    template<typename scalar_t>
    HSSMatrix<scalar_t>::HSSMatrix(const HSSMatrix<scalar_t>& other)
      : HSSMatrixBase<scalar_t>(other) {
      _U = other._U;
      _V = other._V;
      _D = other._D;
      _B01 = other._B01;
      _B10 = other._B10;
    }

    template<typename scalar_t> HSSMatrix<scalar_t>&
    HSSMatrix<scalar_t>::operator=(const HSSMatrix<scalar_t>& other) {
      HSSMatrixBase<scalar_t>::operator=(other);
      _U = other._U;
      _V = other._V;
      _D = other._D;
      _B01 = other._B01;
      _B10 = other._B10;
      return *this;
    }

    template<typename scalar_t> std::unique_ptr<HSSMatrixBase<scalar_t>>
    HSSMatrix<scalar_t>::clone() const {
      return std::unique_ptr<HSSMatrixBase<scalar_t>>
        (new HSSMatrix<scalar_t>(*this));
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::delete_trailing_block() {
      _B01.clear();
      _B10.clear();
      HSSMatrixBase<scalar_t>::delete_trailing_block();
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::compress(const DenseM_t& A, const opts_t& opts) {
      TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_compress);
      switch (opts.compression_algorithm()) {
      case CompressionAlgorithm::ORIGINAL:
        compress_original(A, opts); break;
      case CompressionAlgorithm::STABLE:
        compress_stable(A, opts); break;
      case CompressionAlgorithm::HARD_RESTART:
        compress_hard_restart(A, opts); break;
      default:
        std::cout << "Compression algorithm not recognized!" << std::endl;
      };
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::compress
    (const mult_t& Amult, const elem_t& Aelem, const opts_t& opts) {
      TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_compress);
      switch (opts.compression_algorithm()) {
      case CompressionAlgorithm::ORIGINAL:
        compress_original(Amult, Aelem, opts); break;
      case CompressionAlgorithm::STABLE:
        compress_stable(Amult, Aelem, opts); break;
      case CompressionAlgorithm::HARD_RESTART:
        compress_hard_restart(Amult, Aelem, opts); break;
      default:
        std::cout << "Compression algorithm not recognized!" << std::endl;
      };
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::reset() {
      _U.clear();
      _V.clear();
      _D.clear();
      _B01.clear();
      _B10.clear();
      HSSMatrixBase<scalar_t>::reset();
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    HSSMatrix<scalar_t>::dense() const {
      DenseM_t A(this->rows(), this->cols());
      // apply_HSS(Trans::N, *this, eye<scalar_t>(this->cols(),
      // this->cols()), scalar_t(0.), A);
      WorkDense<scalar_t> w;
      dense_recursive(A, w, true, this->_openmp_task_depth);
      return A;
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::dense_recursive
    (DenseM_t& A, WorkDense<scalar_t>& w, bool isroot, int depth) const {
      if (this->leaf()) {
        copy(_D, A, w.offset.first, w.offset.second);
        w.tmpU = _U.dense();
        w.tmpV = _V.dense();
      } else {
        w.c.resize(2);
        w.c[0].offset = w.offset;
        w.c[1].offset = w.offset + this->_ch[0]->dims();
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        this->_ch[0]->dense_recursive(A, w.c[0], false, depth+1);
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        this->_ch[1]->dense_recursive(A, w.c[1], false, depth+1);
#pragma omp taskwait

#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          DenseM_t tmp01(_B01.rows(), w.c[1].tmpV.rows());
          DenseMW_t A01(this->_ch[0]->rows(), this->_ch[1]->cols(),
                        A, w.c[0].offset.first, w.c[1].offset.second);
          gemm(Trans::N, Trans::C, scalar_t(1.), _B01, w.c[1].tmpV,
               scalar_t(0.), tmp01, depth);
          gemm(Trans::N, Trans::N, scalar_t(1.), w.c[0].tmpU, tmp01,
               scalar_t(0.), A01, depth);
        }
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
        {
          DenseM_t tmp10(_B10.rows(), w.c[0].tmpV.rows());
          DenseMW_t A10(this->_ch[1]->rows(), this->_ch[0]->cols(), A,
                        w.c[1].offset.first, w.c[0].offset.second);
          gemm(Trans::N, Trans::C, scalar_t(1.), _B10, w.c[0].tmpV,
               scalar_t(0.), tmp10, depth);
          gemm(Trans::N, Trans::N, scalar_t(1.), w.c[1].tmpU, tmp10,
               scalar_t(0.), A10, depth);
        }
        if (!isroot) {
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          {
            w.tmpU = DenseM_t(this->rows(), this->U_rank());
            DenseMW_t wtmpU0(this->_ch[0]->rows(), this->U_rank(),
                             w.tmpU, 0, 0);
            DenseMW_t wtmpU1(this->_ch[1]->rows(), this->U_rank(), w.tmpU,
                             this->_ch[0]->rows(), 0);
            auto Udense = _U.dense();
            DenseMW_t Udense0(this->_ch[0]->U_rank(), Udense.cols(),
                              Udense, 0, 0);
            DenseMW_t Udense1(this->_ch[1]->U_rank(), Udense.cols(), Udense,
                              this->_ch[0]->U_rank(), 0);
            gemm(Trans::N, Trans::N, scalar_t(1.), w.c[0].tmpU, Udense0,
                 scalar_t(0.), wtmpU0, depth);
            gemm(Trans::N, Trans::N, scalar_t(1.), w.c[1].tmpU, Udense1,
                 scalar_t(0.), wtmpU1, depth);
          }
#pragma omp task default(shared)                                        \
  if(depth < params::task_recursion_cutoff_level)                       \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          {
            w.tmpV = DenseM_t(this->cols(), this->V_rank());
            DenseMW_t wtmpV0(this->_ch[0]->cols(), this->V_rank(),
                             w.tmpV, 0, 0);
            DenseMW_t wtmpV1(this->_ch[1]->cols(), this->V_rank(),
                             w.tmpV, this->_ch[0]->cols(), 0);
            auto Vdense = _V.dense();
            DenseMW_t Vdense0(this->_ch[0]->V_rank(), Vdense.cols(),
                              Vdense, 0, 0);
            DenseMW_t Vdense1(this->_ch[1]->V_rank(), Vdense.cols(),
                              Vdense, this->_ch[0]->V_rank(), 0);
            gemm(Trans::N, Trans::N, scalar_t(1.), w.c[0].tmpV, Vdense0,
                 scalar_t(0.), wtmpV0, depth);
            gemm(Trans::N, Trans::N, scalar_t(1.), w.c[1].tmpV, Vdense1,
                 scalar_t(0.), wtmpV1, depth);
          }
        }
#pragma omp taskwait
        w.c[0].tmpU.clear();
        w.c[0].tmpV.clear();
        w.c[1].tmpU.clear();
        w.c[1].tmpV.clear();
      }
    }

    template<typename scalar_t> std::size_t
    HSSMatrix<scalar_t>::rank() const {
      if (!this->active()) return 0;
      std::size_t rank = std::max(this->U_rank(), this->V_rank());
      for (auto& c : this->_ch) rank = std::max(rank, c->rank());
      return rank;
    }

    template<typename scalar_t> std::size_t
    HSSMatrix<scalar_t>::memory() const {
      if (!this->active()) return 0;
      std::size_t mem = sizeof(*this) + _U.memory() + _V.memory()
        + _D.memory() + _B01.memory() + _B10.memory();
      for (auto& c : this->_ch) mem += c->memory();
      return mem;
    }

    template<typename scalar_t> std::size_t
    HSSMatrix<scalar_t>::nonzeros() const {
      if (!this->active()) return 0;
      std::size_t nnz = sizeof(*this) + _U.nonzeros() + _V.nonzeros()
        + _D.nonzeros() + _B01.nonzeros() + _B10.nonzeros();
      for (auto& c : this->_ch) nnz += c->nonzeros();
      return nnz;
    }

    template<typename scalar_t> std::size_t
    HSSMatrix<scalar_t>::levels() const {
      if (!this->active()) return 0;
      std::size_t lvls = 0;
      for (auto& c : this->_ch) lvls = std::max(lvls, c->levels());
      return 1 + lvls;
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::print_info
    (std::ostream &out, std::size_t roff, std::size_t coff) const {
      if (!this->active()) return;
      int flag, rank;
      MPI_Initialized(&flag);
      if (flag) rank = mpi_rank();
      else rank = 0;
      out << "SEQ rank=" << rank
          << " b = [" << roff << "," << roff+this->rows()
          << " x " << coff << "," << coff+this->cols() << "]  U = "
          << this->U_rows() << " x " << this->U_rank() << " V = "
          << this->V_rows() << " x " << this->V_rank();
      if (this->leaf()) out << " leaf" << std::endl;
      else out << " non-leaf" << std::endl;
      for (auto& c : this->_ch) {
        c->print_info(out, roff, coff);
        roff += c->rows();
        coff += c->cols();
      }
    }

    template<typename scalar_t> void
    HSSMatrix<scalar_t>::shift(scalar_t sigma) {
      if (!this->active()) return;
      if (this->leaf()) _D.shift(sigma);
      else
        for (auto& c : this->_ch)
          c->shift(sigma);
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::draw
    (std::ostream& of, std::size_t rlo, std::size_t clo) const {
      if (!this->leaf()) {
        char prev = std::cout.fill('0');
        int rank0 = std::max(_B01.rows(), _B01.cols());
        int rank1 = std::max(_B10.rows(), _B10.cols());
        int minmn = std::min(this->rows(), this->cols());
        int red = std::floor(255.0 * rank0 / minmn);
        int blue = 255 - red;
        of << "set obj rect from "
           << rlo << ", " << clo+this->_ch[0]->cols() << " to "
           << rlo+this->_ch[0]->rows()
           << ", " << clo+this->cols()
           << " fc rgb '#"
           << std::hex << std::setw(2) << std::setfill('0') << red
           << "00" << std::setw(2)  << std::setfill('0') << blue
           << "'" << std::dec << std::endl;
        red = std::floor(255.0 * rank1 / minmn);
        blue = 255 - red;
        of << "set obj rect from "
           << rlo+this->_ch[0]->rows() << ", " << clo
           << " to " << rlo+this->rows()
           << ", " << clo+this->_ch[0]->cols()
           << " fc rgb '#"
           << std::hex << std::setw(2) << std::setfill('0') << red
           << "00" << std::setw(2)  << std::setfill('0') << blue
           << "'" << std::dec << std::endl;
        std::cout.fill(prev);
        this->_ch[0]->draw(of, rlo, clo);
        this->_ch[1]->draw
          (of, rlo+this->_ch[0]->rows(), clo+this->_ch[0]->cols());
      } else {
        of << "set obj rect from "
           << rlo << ", " << clo << " to "
           << rlo+this->rows() << ", " << clo+this->cols()
           << " fc rgb 'red'" << std::endl;
      }
    }

    template<typename scalar_t>
    void draw(const HSSMatrix<scalar_t>& H, const std::string& name) {
      std::ofstream of("plot" + name + ".gnuplot");
      of << "set terminal pdf enhanced color size 5,4" << std::endl;
      of << "set output '" << name << ".pdf'" << std::endl;
      H.draw(of);
      of << "set xrange [0:" << H.cols() << "]" << std::endl;
      of << "set yrange [" << H.rows() << ":0]" << std::endl;
      of << "plot x lt -1 notitle" << std::endl;
      of.close();
    }

  } // end namespace HSS
} // end namespace strumpack

#include "HSSMatrix.apply.hpp"
#include "HSSMatrix.compress.hpp"
#include "HSSMatrix.compress_stable.hpp"
#include "HSSMatrix.compress.kernel.hpp"
#include "HSSMatrix.factor.hpp"
#include "HSSMatrix.solve.hpp"
#include "HSSMatrix.extract.hpp"
#include "HSSMatrix.Schur.hpp"

#endif // HSS_MATRIX_HPP
