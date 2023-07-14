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
#include "StrumpackSparseSolverMPIDist.hpp"
#include "misc/TaskTimer.hpp"
#include "sparse/EliminationTreeMPIDist.hpp"
#include "iterative/IterativeSolversMPI.hpp"
#include "sparse/ordering/MatrixReorderingMPI.hpp"
#include "sparse/Redistribute.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  SparseSolverMPIDist<scalar_t,integer_t>::
  SparseSolverMPIDist(MPI_Comm comm, bool verbose) :
    SparseSolverMPIDist<scalar_t,integer_t>
    (comm, 0, nullptr, verbose) {
  }

  template<typename scalar_t,typename integer_t>
  SparseSolverMPIDist<scalar_t,integer_t>::
  ~SparseSolverMPIDist() = default;

  template<typename scalar_t,typename integer_t>
  SparseSolverMPIDist<scalar_t,integer_t>::
  SparseSolverMPIDist
  (MPI_Comm comm, int argc, char* argv[], bool verbose) :
    SparseSolverBase<scalar_t,integer_t>
    (argc, argv, verbose, !mpi_rank(comm)), comm_(comm) {
    if (opts_.verbose() && is_root_)
      std::cout << "# using " << comm_.size()
                << " MPI processes" << std::endl;
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
    int thread_level;
    MPI_Query_thread(&thread_level);
    if (thread_level != MPI_THREAD_MULTIPLE &&
        mpi_rank(comm) == 0)
      std::cerr << "MPI_THREAD_MULTIPLE is required for SLATE"
                << std::endl;
#endif
    // Set the default reordering to PARMETIS?
    //opts_.set_reordering_method(ReorderingStrategy::PARMETIS);
  }

  template<typename scalar_t,typename integer_t> MPI_Comm
  SparseSolverMPIDist<scalar_t,integer_t>::comm() const {
    return comm_.comm();
  }

  template<typename scalar_t,typename integer_t>
  MatrixReordering<scalar_t,integer_t>*
  SparseSolverMPIDist<scalar_t,integer_t>::reordering() {
    return nd_mpi_.get();
  }
  template<typename scalar_t,typename integer_t>
  const MatrixReordering<scalar_t,integer_t>*
  SparseSolverMPIDist<scalar_t,integer_t>::reordering() const {
    return nd_mpi_.get();
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::broadcast_matrix
  (const CSRMatrix<scalar_t,integer_t>& A) {
    mat_mpi_.reset
      (new CSRMatrixMPI<scalar_t,integer_t>(&A, comm_, true));
    this->factored_ = this->reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::broadcast_csr_matrix
  (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, bool symmetric_pattern) {
    CSRMatrix<scalar_t,integer_t> mat_seq
      (N, row_ptr, col_ind, values, symmetric_pattern);
    mat_mpi_.reset
      (new CSRMatrixMPI<scalar_t,integer_t>(&mat_seq, comm_, true));
    this->factored_ = this->reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::set_matrix
  (const CSRMatrixMPI<scalar_t,integer_t>& A) {
    mat_mpi_.reset(new CSRMatrixMPI<scalar_t,integer_t>(A));
    this->factored_ = this->reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::set_distributed_csr_matrix
  (integer_t local_rows, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, const integer_t* dist, bool symmetric_pattern) {
    mat_mpi_.reset
      (new CSRMatrixMPI<scalar_t,integer_t>
       (local_rows, row_ptr, col_ind, values, dist,
        comm_, symmetric_pattern));
    this->factored_ = this->reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::set_MPIAIJ_matrix
  (integer_t local_rows, const integer_t* d_ptr, const integer_t* d_ind,
   const scalar_t* d_val, const integer_t* o_ptr, const integer_t* o_ind,
   const scalar_t* o_val, const integer_t* garray) {
    mat_mpi_.reset
      (new CSRMatrixMPI<scalar_t,integer_t>
       (local_rows, d_ptr, d_ind, d_val, o_ptr, o_ind, o_val,
        garray, comm_));
    this->factored_ = this->reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::update_matrix_values
  (const CSRMatrixMPI<scalar_t,integer_t>& A) {
    if (!(mat_mpi_ && A.local_rows() == mat_mpi_->local_rows() &&
          A.local_nnz() <= mat_mpi_->local_nnz())) {
      // matrix() has been made symmetric, can have more nonzeros
      this->print_wrong_sparsity_error();
      return;
    }
    mat_mpi_.reset(new CSRMatrixMPI<scalar_t,integer_t>(A));
    redistribute_values();
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::update_matrix_values
  (integer_t local_rows, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, const integer_t* dist, bool symmetric_pattern) {
    if (!(mat_mpi_ && local_rows == mat_mpi_->local_rows() &&
          row_ptr[local_rows]-row_ptr[0] <= mat_mpi_->local_nnz())) {
      // matrix() has been made symmetric, can have more nonzeros
      this->print_wrong_sparsity_error();
      return;
    }
    mat_mpi_.reset
      (new CSRMatrixMPI<scalar_t,integer_t>
       (local_rows, row_ptr, col_ind, values, dist,
        comm_, symmetric_pattern));
    redistribute_values();
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::update_MPIAIJ_matrix_values
  (integer_t local_rows, const integer_t* d_ptr, const integer_t* d_ind,
   const scalar_t* d_val, const integer_t* o_ptr, const integer_t* o_ind,
   const scalar_t* o_val, const integer_t* garray) {
    if (!(mat_mpi_ && local_rows == mat_mpi_->local_rows())) {
      // matrix() has been made symmetric, can have more nonzeros
      this->print_wrong_sparsity_error();
      return;
    }
    mat_mpi_.reset
      (new CSRMatrixMPI<scalar_t,integer_t>
       (local_rows, d_ptr, d_ind, d_val, o_ptr, o_ind, o_val,
        garray, comm_));
    redistribute_values();
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::redistribute_values() {
    if (this->reordered_) {
      matrix()->apply_matching(this->matching_);
      matrix()->equilibrate(this->equil_);
      matrix()->symmetrize_sparsity();
      tree_mpi_dist_->update_values(opts_, *mat_mpi_, *nd_mpi_);
      if (opts_.compression() != CompressionType::NONE)
        separator_reordering();
    }
    this->factored_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::setup_reordering() {
    nd_mpi_.reset
      (new MatrixReorderingMPI<scalar_t,integer_t>
       (mat_mpi_->size(), comm_));
  }

  template<typename scalar_t,typename integer_t> int
  SparseSolverMPIDist<scalar_t,integer_t>::compute_reordering
  (const int* p, int base, int nx, int ny, int nz,
   int components, int width) {
    if (p) return nd_mpi_->set_permutation(opts_, *mat_mpi_, p, base);
    return nd_mpi_->nested_dissection
      (opts_, *mat_mpi_, nx, ny, nz, components, width);
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::separator_reordering() {
    // TODO only if not doing MC64 and if enable_replace_tiny_pivots?
    // auto shifted_mat = mat_mpi_->add_missing_diagonal(opts_.pivot_threshold());
    // tree_mpi_dist_->separator_reordering(opts_, *shifted_mat);
    tree_mpi_dist_->separator_reordering(opts_, *mat_mpi_);
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::setup_tree() {
    // TODO only if not doing MC64 and if enable_replace_tiny_pivots?
    // auto shifted_mat = mat_mpi_->add_missing_diagonal(opts_.pivot_threshold());
    // tree_mpi_dist_.reset
    //   (new EliminationTreeMPIDist<scalar_t,integer_t>
    //    (opts_, *shifted_mat, *nd_mpi_, comm_));
    tree_mpi_dist_.reset
      (new EliminationTreeMPIDist<scalar_t,integer_t>
       (opts_, *mat_mpi_, *nd_mpi_, comm_));
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolverMPIDist<scalar_t,integer_t>::solve_internal
  (int nrhs, const scalar_t* b, int ldb,
   scalar_t* x, int ldx, bool use_initial_guess) {
    if (!nrhs) return ReturnCode::SUCCESS;
    auto N = mat_mpi_->local_rows();
    assert(ldb >= N);
    assert(ldx >= N);
    assert(nrhs >= 1);
    auto B = ConstDenseMatrixWrapperPtr(N, nrhs, b, ldb);
    DenseMW_t X(N, nrhs, x, N);
    return this->solve(*B, X, use_initial_guess);
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolverMPIDist<scalar_t,integer_t>::solve_internal
  (const DenseM_t& b, DenseM_t& x, bool use_initial_guess) {
    using real_t = typename RealType<scalar_t>::value_type;

    // reordering has to be called, even for the iterative solvers
    if (!this->reordered_) {
      ReturnCode ierr = this->reorder();
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }

    // factor needs to be called, except for the non-preconditioned
    // solvers
    if (!this->factored_ &&
        opts_.Krylov_solver() != KrylovSolver::GMRES &&
        opts_.Krylov_solver() != KrylovSolver::BICGSTAB) {
      ReturnCode ierr = this->factor();
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }

    assert(std::size_t(mat_mpi_->local_rows()) == b.rows());
    assert(b.rows() == x.rows());
    assert(b.cols() == x.cols());
    TaskTimer t("solve");
    this->perf_counters_start();
    t.start();
    auto nloc = x.rows();
    this->Krylov_its_ = 0;

    auto bloc = b;
    if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
      bloc.scale_rows_real(this->matching_.R);
    if (this->equil_.type == EquilibrationType::ROW ||
        this->equil_.type == EquilibrationType::BOTH)
      bloc.scale_rows_real(this->equil_.R);

    if (use_initial_guess &&
        opts_.Krylov_solver() != KrylovSolver::DIRECT) {
      if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING ||
          this->equil_.type == EquilibrationType::COLUMN ||
          this->equil_.type == EquilibrationType::BOTH) {
        std::vector<real_t> C(nloc, 1.);
        if (this->equil_.type == EquilibrationType::COLUMN ||
            this->equil_.type == EquilibrationType::BOTH)
          for (std::size_t i=0; i<nloc; i++)
            C[i] /= this->equil_.C[i + mat_mpi_->begin_row()];
        if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
          for (std::size_t i=0; i<nloc; i++)
            C[i] /= this->matching_.C[i + mat_mpi_->begin_row()];
        x.scale_rows_real(C);

        // TODO this needs to apply the (inverse) matching permutation!!
      }
    }

    auto spmv = [&](const scalar_t* x, scalar_t* y) {
      mat_mpi_->spmv(x, y);
    };

    auto gmres =
      [&](const std::function<void(scalar_t*)>& prec) {
        assert(x.cols() == 1);
        iterative::GMResMPI<scalar_t>
          (comm_, spmv, prec, nloc, x.data(), bloc.data(),
           opts_.rel_tol(), opts_.abs_tol(),
           this->Krylov_its_, opts_.maxit(),
           opts_.gmres_restart(), opts_.GramSchmidt_type(),
           use_initial_guess, opts_.verbose() && is_root_);
      };
    auto bicgstab =
      [&](const std::function<void(scalar_t*)>& prec) {
        assert(x.cols() == 1);
        iterative::BiCGStabMPI<scalar_t>
          (comm_, spmv, prec, nloc, x.data(), bloc.data(),
           opts_.rel_tol(), opts_.abs_tol(),
           this->Krylov_its_, opts_.maxit(),
           use_initial_guess, opts_.verbose() && is_root_);
      };
    auto MFsolve =
      [&](scalar_t* w) {
        DenseMW_t X(nloc, x.cols(), w, x.ld());
        tree()->multifrontal_solve_dist(X, mat_mpi_->dist());
      };
    auto refine =
      [&]() {
        iterative::IterativeRefinementMPI<scalar_t,integer_t>
          (comm_, *mat_mpi_,
           [&](DenseM_t& w) {
             tree()->multifrontal_solve_dist(w, mat_mpi_->dist()); },
           x, bloc, opts_.rel_tol(), opts_.abs_tol(),
           this->Krylov_its_, opts_.maxit(),
           use_initial_guess, opts_.verbose() && is_root_);
      };

    switch (opts_.Krylov_solver()) {
    case KrylovSolver::AUTO: {
      if (opts_.compression() != CompressionType::NONE && x.cols() == 1)
        gmres(MFsolve);
      else refine();
    }; break;
    case KrylovSolver::REFINE: {
      refine();
    }; break;
    case KrylovSolver::GMRES: {
      gmres([](scalar_t*){});
    }; break;
    case KrylovSolver::PREC_GMRES: {
      gmres(MFsolve);
    }; break;
    case KrylovSolver::BICGSTAB: {
      bicgstab([](scalar_t*){});
    }; break;
    case KrylovSolver::PREC_BICGSTAB: {
      bicgstab(MFsolve);
    }; break;
    case KrylovSolver::DIRECT: {
      // TODO bloc is already a copy, avoid extra copy?
      x = bloc;
      tree()->multifrontal_solve_dist(x, mat_mpi_->dist());
    }; break;
    }

    if (this->equil_.type == EquilibrationType::COLUMN ||
        this->equil_.type == EquilibrationType::BOTH)
      x.scale_rows_real(this->equil_.C.data() + mat_mpi_->begin_row());
    if (opts_.matching() != MatchingJob::NONE) {
      permute_vector(x, this->matching_.Q, mat_mpi_->dist(), comm_);
      if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING)
        x.scale_rows_real(this->matching_.C.data() + mat_mpi_->begin_row());
    }

    t.stop();
    this->perf_counters_stop("DIRECT/GMRES solve");
    this->print_solve_stats(t);
    return ReturnCode::SUCCESS;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolverMPIDist<scalar_t,integer_t>::solve_internal
  (const scalar_t* b, scalar_t* x, bool use_initial_guess) {
    auto N = mat_mpi_->local_rows();
    auto B = ConstDenseMatrixWrapperPtr(N, 1, b, N);
    DenseMW_t X(N, 1, x, N);
    return this->solve(*B, X, use_initial_guess);
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::perf_counters_stop
  (const std::string& s) {
    if (opts_.verbose()) {
#if defined(STRUMPACK_USE_PAPI)
      float rtime1=0., ptime1=0., mflops=0.;
      long_long flpops1=0;
#pragma omp parallel reduction(+:flpops1) reduction(max:rtime1) \
  reduction(max:ptime1)
      PAPI_flops(&rtime1, &ptime1, &flpops1, &mflops);
      float papi_total_flops = flpops1 - this->flpops_;
      papi_total_flops = comm_.all_reduce(papi_total_flops, MPI_SUM);

      // TODO memory usage with PAPI

      if (is_root_) {
        std::cout << "# " << s << " PAPI stats:" << std::endl;
        std::cout << "#   - total flops = " << papi_total_flops << std::endl;
        std::cout << "#   - flop rate = "
                  <<  papi_total_flops/(rtime1-this->rtime_)/1e9
                  << " GFlops/s" << std::endl;
        std::cout << "#   - real time = " << rtime1-this->rtime_
                  << " sec" << std::endl;
        std::cout << "#   - processor time = " << ptime1-this->ptime_
                  << " sec" << std::endl;
      }
#endif
#if defined(STRUMPACK_COUNT_FLOPS)
      auto df = params::flops - this->f0_;
      long long int flopsbytes[2] = {df, params::bytes_moved - this->b0_};
      comm_.all_reduce(flopsbytes, 2, MPI_SUM);
      this->ftot_ = flopsbytes[0];
      this->btot_ = flopsbytes[1];
      this->fmin_ = comm_.all_reduce(df, MPI_MIN);
      this->fmax_ = comm_.all_reduce(df, MPI_MAX);
#endif
    }
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::
  reduce_flop_counters() const {
#if defined(STRUMPACK_COUNT_FLOPS)
    std::array<long long int,19> flops = {
      params::random_flops.load(),
      params::ID_flops.load(),
      params::QR_flops.load(),
      params::ortho_flops.load(),
      params::reduce_sample_flops.load(),
      params::update_sample_flops.load(),
      params::extraction_flops.load(),
      params::CB_sample_flops.load(),
      params::sparse_sample_flops.load(),
      params::ULV_factor_flops.load(),
      params::schur_flops.load(),
      params::full_rank_flops.load(),
      params::f11_fill_flops.load(),
      params::f12_fill_flops.load(),
      params::f21_fill_flops.load(),
      params::f22_fill_flops.load(),
      params::f21_mult_flops.load(),
      params::invf11_mult_flops.load(),
      params::f12_mult_flops.load()
    };
    comm_.reduce(flops.data(), flops.size(), MPI_SUM);
    params::random_flops = flops[0];
    params::ID_flops = flops[1];
    params::QR_flops = flops[2];
    params::ortho_flops = flops[3];
    params::reduce_sample_flops = flops[4];
    params::update_sample_flops = flops[5];
    params::extraction_flops = flops[6];
    params::CB_sample_flops = flops[7];
    params::sparse_sample_flops = flops[8];
    params::ULV_factor_flops = flops[9];
    params::schur_flops = flops[10];
    params::full_rank_flops = flops[11];
    params::f11_fill_flops = flops[12];
    params::f12_fill_flops = flops[13];
    params::f21_fill_flops = flops[14];
    params::f22_fill_flops = flops[15];
    params::f21_mult_flops = flops[16];
    params::invf11_mult_flops = flops[17];
    params::f12_mult_flops = flops[18];
#endif
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverMPIDist<scalar_t,integer_t>::
  delete_factors_internal() {
    tree_mpi_dist_->delete_factors();
  }

  // explicit template instantiations
  template class SparseSolverMPIDist<float,int>;
  template class SparseSolverMPIDist<double,int>;
  template class SparseSolverMPIDist<std::complex<float>,int>;
  template class SparseSolverMPIDist<std::complex<double>,int>;

  template class SparseSolverMPIDist<float,long int>;
  template class SparseSolverMPIDist<double,long int>;
  template class SparseSolverMPIDist<std::complex<float>,long int>;
  template class SparseSolverMPIDist<std::complex<double>,long int>;

  template class SparseSolverMPIDist<float,long long int>;
  template class SparseSolverMPIDist<double,long long int>;
  template class SparseSolverMPIDist<std::complex<float>,long long int>;
  template class SparseSolverMPIDist<std::complex<double>,long long int>;

} // end namespace strumpack
