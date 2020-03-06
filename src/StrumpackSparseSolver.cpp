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

#include "StrumpackSparseSolver.hpp"

#if defined(STRUMPACK_USE_PAPI)
#include <papi.h>
#endif

#include "misc/Tools.hpp"
#include "misc/TaskTimer.hpp"
#include "StrumpackOptions.hpp"
#include "sparse/ordering/MatrixReordering.hpp"
#include "sparse/EliminationTree.hpp"
#include "sparse/iterative/IterativeSolvers.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolver<scalar_t,integer_t>::StrumpackSparseSolver
  (bool verbose, bool root)
    : StrumpackSparseSolver<scalar_t,integer_t>(0, nullptr, verbose, root) {
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolver<scalar_t,integer_t>::StrumpackSparseSolver
  (int argc, char* argv[], bool verbose, bool root)
    : opts_(argc, argv), is_root_(root) {
    opts_.set_verbose(verbose);
    old_handler_ = std::set_new_handler
      ([]{ std::cerr << "STRUMPACK: out of memory!" << std::endl; abort(); });
    papi_initialize();
    if (opts_.verbose() && is_root_) {
      std::cout << "# Initializing STRUMPACK" << std::endl;
#if defined(_OPENMP)
      if (params::num_threads == 1)
        std::cout << "# using " << params::num_threads
                  << " OpenMP thread" << std::endl;
      else
        std::cout << "# using " << params::num_threads
                  << " OpenMP threads" << std::endl;
#else
      std::cout << "# running serially, no OpenMP support!" << std::endl;
#endif
      // a heuristic to set the recursion task cutoff level based on
      // the number of threads
      if (params::num_threads == 1) params::task_recursion_cutoff_level = 0;
      else {
        params::task_recursion_cutoff_level =
          std::log2(params::num_threads) + 3;
        std::cout << "# number of tasking levels = "
                  << params::task_recursion_cutoff_level
                  << " = log_2(#threads) + 3"<< std::endl;
      }
    }
#if defined(STRUMPACK_COUNT_FLOPS)
    // if (!params::flops.is_lock_free())
    //   std::cerr << "# WARNING: the flop counter is not lock free"
    //             << std::endl;
#endif
    opts_.HSS_options().set_synchronized_compression(true);
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolver<scalar_t,integer_t>::~StrumpackSparseSolver() {
    std::set_new_handler(old_handler_);
  }

  template<typename scalar_t,typename integer_t> SPOptions<scalar_t>&
  StrumpackSparseSolver<scalar_t,integer_t>::options() {
    return opts_;
  }

  template<typename scalar_t,typename integer_t> const SPOptions<scalar_t>&
  StrumpackSparseSolver<scalar_t,integer_t>::options() const {
    return opts_;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::set_from_options() {
    opts_.set_from_command_line();
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::set_from_options
  (int argc, char* argv[]) {
    opts_.set_from_command_line(argc, argv);
  }

  template<typename scalar_t,typename integer_t> int
  StrumpackSparseSolver<scalar_t,integer_t>::maximum_rank() const {
    return tree()->maximum_rank();
  }

  template<typename scalar_t,typename integer_t> std::size_t
  StrumpackSparseSolver<scalar_t,integer_t>::factor_nonzeros() const {
    return tree()->factor_nonzeros();
  }

  template<typename scalar_t,typename integer_t> std::size_t
  StrumpackSparseSolver<scalar_t,integer_t>::factor_memory() const {
    return tree()->factor_nonzeros() * sizeof(scalar_t);
  }

  template<typename scalar_t,typename integer_t> int
  StrumpackSparseSolver<scalar_t,integer_t>::Krylov_iterations() const {
    return Krylov_its_;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::draw
  (const std::string& name) const {
    tree()->draw(*matrix(), name);
  }

  template<typename scalar_t,typename integer_t> long long
  StrumpackSparseSolver<scalar_t,integer_t>::dense_factor_nonzeros() const {
    return tree()->dense_factor_nonzeros();
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::setup_tree() {
    tree_.reset(new EliminationTree<scalar_t,integer_t>
                (opts_, *mat_, nd_->tree()));
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::setup_reordering() {
    nd_.reset(new MatrixReordering<scalar_t,integer_t>(matrix()->size()));
  }

  template<typename scalar_t,typename integer_t> int
  StrumpackSparseSolver<scalar_t,integer_t>::compute_reordering
  (const int* p, int base, int nx, int ny, int nz,
   int components, int width) {
    if (p) return nd_->set_permutation(opts_, *mat_, p, base);
    return nd_->nested_dissection
      (opts_, *mat_, nx, ny, nz, components, width);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::separator_reordering() {
    nd_->separator_reordering(opts_, *mat_, tree_->root());
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::papi_initialize() {
#if defined(STRUMPACK_USE_PAPI)
    // TODO call PAPI_library_init???
    float mflops = 0.;
    int retval = PAPI_flops(&rtime_, &ptime_, &_flpops, &mflops);
    if (retval != PAPI_OK) {
      std::cerr << "# WARNING: problem starting PAPI performance counters:"
                << std::endl;
      switch (retval) {
      case PAPI_EINVAL:
        std::cerr << "#   - the counters were already started by"
          << " something other than: PAPI_flips() or PAPI_flops()."
          << std::endl; break;
      case PAPI_ENOEVNT:
        std::cerr << "#   - the floating point operations, floating point"
                  << " instructions or total cycles event does not exist."
                  << std::endl; break;
      case PAPI_ENOMEM:
        std::cerr << "#   - insufficient memory to complete the operation."
                  << std::endl; break;
      default:
        std::cerr << "#   - some other error: " << retval << std::endl;
      }
    }
#endif
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::perf_counters_start() {
#if defined(STRUMPACK_USE_PAPI)
    float mflops = 0., rtime = 0., ptime = 0.;
    long_long flpops = 0; // cannot use class variables in openmp clause
#pragma omp parallel reduction(+:flpops) reduction(max:rtime) \
  reduction(max:ptime)
    PAPI_flops(&rtime, &ptime, &flpops, &mflops);
    _flpops = flpops; rtime_ = rtime; ptime_ = ptime;
#endif
#if defined(STRUMPACK_COUNT_FLOPS)
    f0_ = params::flops;
    b0_ = params::bytes;
#endif
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::perf_counters_stop
  (const std::string& s) {
#if defined(STRUMPACK_USE_PAPI)
    float mflops = 0., rtime = 0., ptime = 0.;
    long_long flpops = 0;
#pragma omp parallel reduction(+:flpops) reduction(max:rtime)  \
  reduction(max:ptime)
    PAPI_flops(&rtime, &ptime, &flpops, &mflops);
    PAPI_dmem_info_t dmem;
    PAPI_get_dmem_info(&dmem);
    if (opts_.verbose() && is_root_) {
      std::cout << "# " << s << " PAPI stats:" << std::endl;
      std::cout << "#   - total flops = "
                << double(flpops-_flpops) << std::endl;
      std::cout << "#   - flop rate = "
                << double(flpops-_flpops)/(rtime-rtime_)/1e9
                << " GFlops/s" << std::endl;
      std::cout << "#   - real time = " << rtime-rtime_
                << " sec" << std::endl;
      std::cout << "#   - processor time = " << ptime-ptime_
                << " sec" << std::endl;
      std::cout << "# mem size:\t\t" << dmem.size << std::endl;
      std::cout << "# mem resident:\t\t" << dmem.resident << std::endl;
      std::cout << "# mem high water mark:\t" << dmem.high_water_mark
                << std::endl;
      std::cout << "# mem shared:\t\t" << dmem.shared << std::endl;
      std::cout << "# mem text:\t\t" << dmem.text << std::endl;
      std::cout << "# mem library:\t\t" << dmem.library << std::endl;
      std::cout << "# mem heap:\t\t" << dmem.heap << std::endl;
      std::cout << "# mem locked:\t\t" << dmem.locked << std::endl;
      std::cout << "# mem stack:\t\t" << dmem.stack << std::endl;
      std::cout << "# mem pagesize:\t\t" << dmem.pagesize << std::endl;
    }
#endif
#if defined(STRUMPACK_COUNT_FLOPS)
    fmin_ = fmax_ = ftot_ = params::flops - f0_;
    bmin_ = bmax_ = btot_ = params::bytes - b0_;
#endif
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::print_solve_stats
  (TaskTimer& t) const {
    double tel = t.elapsed();
    if (opts_.verbose() && is_root_) {
      std::cout << "# DIRECT/GMRES solve:" << std::endl;
      std::cout << "#   - abs_tol = " << opts_.abs_tol()
                << ", rel_tol = " << opts_.rel_tol()
                << ", restart = " << opts_.gmres_restart()
                << ", maxit = " << opts_.maxit() << std::endl;
      std::cout << "#   - number of Krylov iterations = "
                << Krylov_its_ << std::endl;
      std::cout << "#   - solve time = " << tel << std::endl;
#if defined(STRUMPACK_COUNT_FLOPS)
      std::cout << "#   - solve flops = " << double(ftot_) << " min = "
                << double(fmin_) << " max = " << double(fmax_) << std::endl;
      std::cout << "#   - solve flop rate = " << ftot_ / tel / 1e9 << " GFlop/s"
                << std::endl;
      std::cout << "#   - bytes moved = " << double(btot_) / 1e6
                << " MB, min = "<< double(bmin_) / 1e6
                << " MB, max = " << double(bmax_) / 1e6 << " MB" << std::endl;
      std::cout << "#   - byte rate = " << btot_ / tel / 1e9 << " GByte/s"
                << std::endl;
      std::cout << "#   - solve arithmetic intensity = "
                << double(ftot_) / btot_
                << " flop/byte" << std::endl;
#endif
    }
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::set_matrix
  (const CSRMatrix<scalar_t,integer_t>& A) {
    mat_.reset(new CSRMatrix<scalar_t,integer_t>(A));
    factored_ = reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::set_csr_matrix
  (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, bool symmetric_pattern) {
    mat_.reset(new CSRMatrix<scalar_t,integer_t>
               (N, row_ptr, col_ind, values, symmetric_pattern));
    factored_ = reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolver<scalar_t,integer_t>::reorder
  (int nx, int ny, int nz, int components, int width) {
    return internal_reorder(nullptr, 0, nx, ny, nz, components, width);
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolver<scalar_t,integer_t>::reorder
  (const int* p, int base) {
    return internal_reorder(p, base, 1, 1, 1, 1, 1);
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolver<scalar_t,integer_t>::internal_reorder
  (const int* p, int base, int nx, int ny, int nz,
   int components, int width) {
    if (!matrix()) return ReturnCode::MATRIX_NOT_SET;
    TaskTimer t1("permute-scale");
    int ierr;
    if (opts_.matching() != MatchingJob::NONE) {
      if (opts_.verbose() && is_root_)
        std::cout << "# matching job: "
                  << get_description(opts_.matching())
                  << std::endl;
      t1.time([&](){
          ierr = matrix()->permute_and_scale
            (opts_.matching(), matching_cperm_, matching_Dr_, matching_Dc_);
        });
      if (ierr) {
        std::cerr << "ERROR: matching failed" << std::endl;
        return ReturnCode::REORDERING_ERROR;
      }
    }
    auto old_nnz = matrix()->nnz();
    TaskTimer t2("sparsity-symmetrization",
                 [&](){ matrix()->symmetrize_sparsity(); });
    if (matrix()->nnz() != old_nnz && opts_.verbose() && is_root_) {
      std::cout << "# Matrix padded with zeros to get symmetric pattern."
                << std::endl;
      std::cout << "# Number of nonzeros increased from "
                << number_format_with_commas(old_nnz) << " to "
                << number_format_with_commas(matrix()->nnz()) << "."
                << std::endl;
    }

    TaskTimer t3("nested-dissection");
    perf_counters_start();
    t3.start();
    setup_reordering();
    ierr = compute_reordering(p, base, nx, ny, nz, components, width);
    if (ierr) {
      std::cerr << "ERROR: nested dissection went wrong, ierr="
                << ierr << std::endl;
      return ReturnCode::REORDERING_ERROR;
    }
    matrix()->permute(reordering()->iperm(), reordering()->perm());
    t3.stop();
    if (opts_.verbose() && is_root_) {
      std::cout << "#   - nd time = " << t3.elapsed() << std::endl;
      if (opts_.matching() != MatchingJob::NONE)
        std::cout << "#   - matching time = " << t1.elapsed() << std::endl;
      std::cout << "#   - symmetrization time = " << t2.elapsed()
                << std::endl;
    }
    perf_counters_stop("nested dissection");

    perf_counters_start();
    TaskTimer t0("symbolic-factorization", [&](){ setup_tree(); });
    reordering()->clear_tree_data();
    if (opts_.verbose()) {
      auto fc = tree()->front_counter();
      if (is_root_) {
        std::cout << "# symbolic factorization:" << std::endl;
        std::cout << "#   - nr of dense Frontal matrices = "
                  << number_format_with_commas(fc.dense) << std::endl;
        if (fc.HSS)
          std::cout << "#   - nr of HSS Frontal matrices = "
                    << number_format_with_commas(fc.HSS) << std::endl;
        if (fc.BLR)
          std::cout << "#   - nr of BLR Frontal matrices = "
                    << number_format_with_commas(fc.BLR) << std::endl;
        if (fc.HODLR)
          std::cout << "#   - nr of HODLR Frontal matrices = "
                    << number_format_with_commas(fc.HODLR) << std::endl;
        if (fc.lossy)
          std::cout << "#   - nr of lossy Frontal matrices = "
                    << number_format_with_commas(fc.lossy) << std::endl;
        std::cout << "#   - symb-factor time = " << t0.elapsed() << std::endl;
      }
    }
    perf_counters_stop("symbolic factorization");

    if (opts_.compression() != CompressionType::NONE) {
      perf_counters_start();
      // TODO also broadcast this?? is computed with metis
      TaskTimer t4("separator-reordering", [&](){ separator_reordering(); });
      if (opts_.verbose() && is_root_)
        std::cout << "#   - sep-reorder time = "
                  << t4.elapsed() << std::endl;
      perf_counters_stop("separator reordering");
    }

    reordered_ = true;
    return ReturnCode::SUCCESS;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::flop_breakdown_reset() const {
#if defined(STRUMPACK_COUNT_FLOPS)
    params::random_flops = 0;
    params::ID_flops = 0;
    params::QR_flops = 0;
    params::ortho_flops = 0;
    params::reduce_sample_flops = 0;
    params::update_sample_flops = 0;
    params::extraction_flops = 0;
    params::CB_sample_flops = 0;
    params::sparse_sample_flops = 0;
    params::ULV_factor_flops = 0;
    params::schur_flops = 0;
    params::full_rank_flops = 0;
    params::f11_fill_flops = 0;
    params::f12_fill_flops = 0;
    params::f21_fill_flops = 0;
    params::f22_fill_flops = 0;
    params::f21_mult_flops = 0;
    params::invf11_mult_flops = 0;
    params::f12_mult_flops = 0;
#endif
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::print_flop_breakdown_HSS() const {
    reduce_flop_counters();
    if (!is_root_) return;
    float sample_flops = params::CB_sample_flops + params::sparse_sample_flops;
    float compression_flops = params::random_flops + params::ID_flops +
      params::QR_flops + params::ortho_flops + params::reduce_sample_flops +
      params::update_sample_flops + params::extraction_flops + sample_flops;
    std::cout << std::endl;
    std::cout << "# ----- FLOP BREAKDOWN ---------------------" << std::endl;
    std::cout << "# compression           = " << compression_flops << std::endl;
    std::cout << "#    random             = " << float(params::random_flops) << std::endl;
    std::cout << "#    ID                 = " << float(params::ID_flops) << std::endl;
    std::cout << "#    QR                 = " << float(params::QR_flops) << std::endl;
    std::cout << "#    ortho              = " << float(params::ortho_flops) << std::endl;
    std::cout << "#    reduce_samples     = " << float(params::reduce_sample_flops) << std::endl;
    std::cout << "#    update_samples     = " << float(params::update_sample_flops) << std::endl;
    std::cout << "#    extraction         = " << float(params::extraction_flops) << std::endl;
    std::cout << "#    sampling           = " << sample_flops << std::endl;
    std::cout << "#       CB_sample       = " << float(params::CB_sample_flops) << std::endl;
    std::cout << "#       sparse_sampling = " << float(params::sparse_sample_flops) << std::endl;
    std::cout << "# ULV_factor            = " << float(params::ULV_factor_flops) << std::endl;
    std::cout << "# Schur                 = " << float(params::schur_flops) << std::endl;
    std::cout << "# full_rank             = " << float(params::full_rank_flops) << std::endl;
    std::cout << "# --------------------------------------------" << std::endl;
    std::cout << "# total                 = "
              << (compression_flops + params::ULV_factor_flops +
                  params::schur_flops + params::full_rank_flops) << std::endl;
    std::cout << "# --------------------------------------------" << std::endl << std::endl;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolver<scalar_t,integer_t>::print_flop_breakdown_HODLR() const {
    reduce_flop_counters();
    if (!is_root_) return;
    float sample_flops = strumpack::params::CB_sample_flops +
      strumpack::params::schur_flops + strumpack::params::sparse_sample_flops;
    float compression_flops = strumpack::params::f11_fill_flops +
      strumpack::params::f12_fill_flops + strumpack::params::f21_fill_flops +
      strumpack::params::f22_fill_flops + sample_flops;
    std::cout << std::endl;
    std::cout << "# ----- FLOP BREAKDOWN ---------------------" << std::endl;
    std::cout << "# F11_compression       = " << float(params::f11_fill_flops) << std::endl;
    std::cout << "# F12_compression       = " << float(params::f12_fill_flops) << std::endl;
    std::cout << "# F21_compression       = " << float(params::f21_fill_flops) << std::endl;
    std::cout << "# F22_compression       = " << float(params::f22_fill_flops) << std::endl;
    std::cout << "# sampling              = " << sample_flops << std::endl;
    std::cout << "#    CB_sample          = " << float(params::CB_sample_flops) << std::endl;
    std::cout << "#    sparse_sampling    = " << float(params::sparse_sample_flops) << std::endl;
    std::cout << "#    Schur_sampling     = " << float(params::schur_flops) << std::endl;
    std::cout << "#       F21             = " << float(params::f21_mult_flops) << std::endl;
    std::cout << "#       inv(F11)        = " << float(params::invf11_mult_flops) << std::endl;
    std::cout << "#       F12             = " << float(params::f12_mult_flops) << std::endl;
    std::cout << "# HODLR_factor          = " << float(params::ULV_factor_flops) << std::endl;
    std::cout << "# full_rank             = " << float(params::full_rank_flops) << std::endl;
    std::cout << "# --------------------------------------------" << std::endl;
    std::cout << "# total                 = "
              << (compression_flops + strumpack::params::ULV_factor_flops +
                  strumpack::params::full_rank_flops) << std::endl;
    std::cout << "# --------------------------------------------" << std::endl << std::endl;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolver<scalar_t,integer_t>::factor() {
    if (!matrix()) return ReturnCode::MATRIX_NOT_SET;
    if (factored_) return ReturnCode::SUCCESS;
    if (!reordered_) {
      ReturnCode ierr = reorder();
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }
    float dfnnz = 0.;
    if (opts_.verbose()) {
      dfnnz = dense_factor_nonzeros();
      if (is_root_) {
        std::cout << "# multifrontal factorization:" << std::endl;
        std::cout << "#   - estimated memory usage (exact solver) = "
                  << dfnnz * sizeof(scalar_t) / 1.e6 << " MB" << std::endl;
      }
    }
    perf_counters_start();
    flop_breakdown_reset();
    TaskTimer t1("Sparse-factorization", [&]() {
        tree()->multifrontal_factorization(*matrix(), opts_);
      });
    perf_counters_stop("numerical factorization");
    if (opts_.verbose()) {
      auto fnnz = factor_nonzeros();
      auto max_rank = maximum_rank();
      if (is_root_) {
        std::cout << "#   - factor time = " << t1.elapsed() << std::endl;
        std::cout << "#   - factor nonzeros = "
                  << number_format_with_commas(fnnz) << std::endl;
        std::cout << "#   - factor memory = "
                  << float(fnnz) * sizeof(scalar_t) / 1.e6 << " MB" << std::endl;
#if defined(STRUMPACK_COUNT_FLOPS)
        std::cout << "#   - factor flops = " << double(ftot_) << " min = "
                  << double(fmin_) << " max = " << double(fmax_)
                  << std::endl;
        std::cout << "#   - factor flop rate = " << ftot_ / t1.elapsed() / 1e9
                  << " GFlop/s" << std::endl;
#endif
        std::cout << "#   - factor memory/nonzeros = "
                  << float(fnnz) / dfnnz * 100.0
                  << " % of multifrontal" << std::endl;
        std::cout << "#   - compression = " << std::boolalpha
                  << get_name(opts_.compression()) << std::endl;
        if (opts_.compression() == CompressionType::HSS) {
          std::cout << "#   - maximum HSS rank = " << max_rank << std::endl;
          std::cout << "#   - relative compression tolerance = "
                    << opts_.HSS_options().rel_tol() << std::endl;
          std::cout << "#   - absolute compression tolerance = "
                    << opts_.HSS_options().abs_tol() << std::endl;
          std::cout << "#   - "
                    << get_name(opts_.HSS_options().random_distribution())
                    << " distribution with "
                    << get_name(opts_.HSS_options().random_engine())
                    << " engine" << std::endl;
        }
        if (opts_.compression() == CompressionType::BLR) {
          std::cout << "#   - relative compression tolerance = "
                    << opts_.BLR_options().rel_tol() << std::endl;
          std::cout << "#   - absolute compression tolerance = "
                    << opts_.BLR_options().abs_tol() << std::endl;
        }
#if defined(STRUMPACK_USE_BPACK)
        if (opts_.compression() == CompressionType::HODLR) {
          std::cout << "#   - maximum HODLR rank = " << max_rank << std::endl;
          std::cout << "#   - relative compression tolerance = "
                    << opts_.HODLR_options().rel_tol() << std::endl;
          std::cout << "#   - absolute compression tolerance = "
                    << opts_.HODLR_options().abs_tol() << std::endl;
        }
#endif
#if defined(STRUMPACK_USE_ZFP)
        if (opts_.compression() == CompressionType::LOSSY)
          std::cout << "#   - lossy compression precision = "
                    << opts_.lossy_precision() << " bitplanes" << std::endl;
#endif
      }
      if (opts_.compression() == CompressionType::HSS)
        print_flop_breakdown_HSS();
      if (opts_.compression() == CompressionType::HODLR)
        print_flop_breakdown_HODLR();
    }
    if (rank_out_) tree()->print_rank_statistics(*rank_out_);
    factored_ = true;
    return ReturnCode::SUCCESS;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolver<scalar_t,integer_t>::solve
  (const scalar_t* b, scalar_t* x, bool use_initial_guess) {
    auto N = matrix()->size();
    auto B = ConstDenseMatrixWrapperPtr(N, 1, b, N);
    DenseMW_t X(N, 1, x, N);
    return solve(*B, X, use_initial_guess);
  }

  // TODO make this const
  //  Krylov its and flops, bytes, time are modified!!
  // pass those as a pointer to a struct ??
  // this can also call factor if not already factored!!
  template<typename scalar_t,typename integer_t> ReturnCode
  StrumpackSparseSolver<scalar_t,integer_t>::solve
  (const DenseM_t& b, DenseM_t& x, bool use_initial_guess) {
    if (!this->factored_ &&
        opts_.Krylov_solver() != KrylovSolver::GMRES &&
        opts_.Krylov_solver() != KrylovSolver::BICGSTAB) {
      ReturnCode ierr = factor();
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }
    TaskTimer t("solve");
    perf_counters_start();
    t.start();

    integer_t N = matrix()->size(), d = b.cols();
    assert(N < std::numeric_limits<int>::max());

    DenseM_t bloc(b.rows(), b.cols());

    // TODO this fails when the reordering was not done, for instance
    // for iterative solvers!!!
    auto iperm = reordering()->iperm();
    if (use_initial_guess &&
        opts_.Krylov_solver() != KrylovSolver::DIRECT) {
      if (opts_.matching() != MatchingJob::NONE) {
        if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING) {
          for (integer_t j=0; j<d; j++)
#pragma omp parallel for
            for (integer_t i=0; i<N; i++) {
              auto pi = iperm[matching_cperm_[i]];
              bloc(i, j) = x(pi, j) / matching_Dc_[pi];
            }
        } else {
          for (integer_t j=0; j<d; j++)
#pragma omp parallel for
            for (integer_t i=0; i<N; i++)
              bloc(i, j) = x(iperm[matching_cperm_[i]], j);
        }
      } else {
        for (integer_t j=0; j<d; j++)
#pragma omp parallel for
          for (integer_t i=0; i<N; i++)
            bloc(i, j) = x(iperm[i], j);
      }
      x.copy(bloc);
    }
    if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING) {
      for (integer_t j=0; j<d; j++)
#pragma omp parallel for
        for (integer_t i=0; i<N; i++) {
          auto pi = iperm[i];
          bloc(i, j) = matching_Dr_[pi] * b(pi, j);
        }
    } else {
      for (integer_t j=0; j<d; j++)
#pragma omp parallel for
        for (integer_t i=0; i<N; i++)
          bloc(i, j) = b(iperm[i], j);
    }

    Krylov_its_ = 0;

    auto spmv = [&](const scalar_t* x, scalar_t* y) {
      matrix()->spmv(x, y);
    };

    auto gmres_solve =
      [&](const std::function<void(scalar_t*)>& prec) {
        iterative::GMRes<scalar_t>
          (spmv, prec, x.rows(), x.data(), bloc.data(),
           opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
           opts_.gmres_restart(), opts_.GramSchmidt_type(),
           use_initial_guess, opts_.verbose() && is_root_);
      };
    auto bicgstab_solve =
      [&](const std::function<void(scalar_t*)>& prec) {
        iterative::BiCGStab<scalar_t>
          (spmv, prec, x.rows(), x.data(), bloc.data(),
           opts_.rel_tol(), opts_.abs_tol(), Krylov_its_, opts_.maxit(),
           use_initial_guess, opts_.verbose() && is_root_);
      };
    auto MFsolve =
      [&](scalar_t* w) {
        DenseMW_t X(x.rows(), 1, w, x.ld());
        tree()->multifrontal_solve(X);
      };
    auto refine =
      [&]() {
        iterative::IterativeRefinement<scalar_t,integer_t>
          (*matrix(), [&](DenseM_t& w) { tree()->multifrontal_solve(w); },
           x, bloc, opts_.rel_tol(), opts_.abs_tol(),
           Krylov_its_, opts_.maxit(), use_initial_guess,
           opts_.verbose() && is_root_);
      };

    switch (opts_.Krylov_solver()) {
    case KrylovSolver::AUTO: {
      if (opts_.compression() != CompressionType::NONE && x.cols() == 1)
        gmres_solve(MFsolve);
      else refine();
    }; break;
    case KrylovSolver::DIRECT: {
      x = bloc;
      tree()->multifrontal_solve(x);
    }; break;
    case KrylovSolver::REFINE: {
      refine();
    }; break;
    case KrylovSolver::PREC_GMRES: {
      assert(x.cols() == 1);
      gmres_solve(MFsolve);
    }; break;
    case KrylovSolver::GMRES: {
      assert(x.cols() == 1);
      gmres_solve([](scalar_t* x){});
    }; break;
    case KrylovSolver::PREC_BICGSTAB: {
      assert(x.cols() == 1);
      bicgstab_solve(MFsolve);
    }; break;
    case KrylovSolver::BICGSTAB: {
      assert(x.cols() == 1);
      bicgstab_solve([](scalar_t* x){});
    }; break;
    }

    if (opts_.matching() != MatchingJob::NONE) {
      if (opts_.matching() == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING) {
        for (integer_t j=0; j<d; j++)
#pragma omp parallel for
          for (integer_t i=0; i<N; i++) {
            auto ipi = matching_cperm_[iperm[i]];
            bloc(ipi, j) = x(i, j) * matching_Dc_[ipi];
          }
      } else {
        for (integer_t j=0; j<d; j++)
#pragma omp parallel for
          for (integer_t i=0; i<N; i++)
            bloc(matching_cperm_[iperm[i]], j) = x(i, j);
      }
    } else {
      auto perm = reordering()->perm();
      for (integer_t j=0; j<d; j++)
#pragma omp parallel for
        for (integer_t i=0; i<N; i++)
          bloc(i, j) = x(perm[i], j);
    }
    x.copy(bloc);

    t.stop();
    perf_counters_stop("DIRECT/GMRES solve");
    print_solve_stats(t);
    return ReturnCode::SUCCESS;
  }

  // explicit template instantiations
  template class StrumpackSparseSolver<float,int>;
  template class StrumpackSparseSolver<double,int>;
  template class StrumpackSparseSolver<std::complex<float>,int>;
  template class StrumpackSparseSolver<std::complex<double>,int>;

  template class StrumpackSparseSolver<float,long int>;
  template class StrumpackSparseSolver<double,long int>;
  template class StrumpackSparseSolver<std::complex<float>,long int>;
  template class StrumpackSparseSolver<std::complex<double>,long int>;

  template class StrumpackSparseSolver<float,long long int>;
  template class StrumpackSparseSolver<double,long long int>;
  template class StrumpackSparseSolver<std::complex<float>,long long int>;
  template class StrumpackSparseSolver<std::complex<double>,long long int>;

} //end namespace strumpack
