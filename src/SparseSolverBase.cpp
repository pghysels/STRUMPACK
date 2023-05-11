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
#include "SparseSolverBase.hpp"

#if defined(STRUMPACK_USE_PAPI)
#include <papi.h>
#endif

#include "misc/Tools.hpp"
#include "misc/TaskTimer.hpp"
#include "StrumpackOptions.hpp"
#include "sparse/ordering/MatrixReordering.hpp"
#include "sparse/EliminationTree.hpp"
#include "iterative/IterativeSolvers.hpp"
#if defined(STRUMPACK_USE_CUDA)
#include "dense/CUDAWrapper.hpp"
#endif
#if defined(STRUMPACK_USE_HIP)
#include "dense/HIPWrapper.hpp"
#endif
#if defined(STRUMPACK_USE_SYCL)
#include "dense/DPCPPWrapper.hpp"
#endif


namespace strumpack {

  template<typename scalar_t,typename integer_t>
  SparseSolverBase<scalar_t,integer_t>::SparseSolverBase
  (bool verbose, bool root)
    : SparseSolverBase<scalar_t,integer_t>
    (0, nullptr, verbose, root) {
  }

  template<typename scalar_t,typename integer_t>
  SparseSolverBase<scalar_t,integer_t>::SparseSolverBase
  (int argc, char* argv[], bool verbose, bool root)
    : opts_(argc, argv), is_root_(root) {
    opts_.set_verbose(verbose);
    old_handler_ = std::set_new_handler
      ([] {
         std::cerr << "STRUMPACK: out of memory!" << std::endl;
         abort();
       });
    papi_initialize();
    if (opts_.verbose() && is_root_) {
      std::cout << "# Initializing STRUMPACK" << std::endl;
#if defined(_OPENMP)
      std::cout << "# using " << params::num_threads
                << " OpenMP thread(s)" << std::endl;
#else
      std::cout << "# running serially, no OpenMP support!" << std::endl;
#endif
    }
    // a heuristic to set the recursion task cutoff level based on
    // the number of threads
    if (params::num_threads == 1)
      params::task_recursion_cutoff_level = 0;
    else
      params::task_recursion_cutoff_level =
        std::log2(params::num_threads) + 3;
    opts_.HSS_options().set_synchronized_compression(true);
#if defined(STRUMPACK_USE_CUDA) || defined(STRUMPACK_USE_HIP)
    if (opts_.use_gpu()) gpu::init();
#endif
#if defined(STRUMPACK_USE_SYCL)
    if (opts_.use_gpu()) dpcpp::init();
#endif
  }

  template<typename scalar_t,typename integer_t>
  SparseSolverBase<scalar_t,integer_t>::~SparseSolverBase() {
    std::set_new_handler(old_handler_);
  }

  template<typename scalar_t,typename integer_t> SPOptions<scalar_t>&
  SparseSolverBase<scalar_t,integer_t>::options() {
    return opts_;
  }

  template<typename scalar_t,typename integer_t> const SPOptions<scalar_t>&
  SparseSolverBase<scalar_t,integer_t>::options() const {
    return opts_;
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverBase<scalar_t,integer_t>::set_from_options() {
    opts_.set_from_command_line();
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverBase<scalar_t,integer_t>::set_from_options
  (int argc, char* argv[]) {
    opts_.set_from_command_line(argc, argv);
  }

  template<typename scalar_t,typename integer_t> int
  SparseSolverBase<scalar_t,integer_t>::maximum_rank() const {
    return tree()->maximum_rank();
  }

  template<typename scalar_t,typename integer_t> std::size_t
  SparseSolverBase<scalar_t,integer_t>::factor_nonzeros() const {
    return tree()->factor_nonzeros();
  }

  template<typename scalar_t,typename integer_t> std::size_t
  SparseSolverBase<scalar_t,integer_t>::factor_memory() const {
    return tree()->factor_nonzeros() * sizeof(scalar_t);
  }

  template<typename scalar_t,typename integer_t> int
  SparseSolverBase<scalar_t,integer_t>::Krylov_iterations() const {
    return Krylov_its_;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolverBase<scalar_t,integer_t>::inertia
  (integer_t& neg, integer_t& zero, integer_t& pos) {
    neg = zero = pos = 0;
    if (opts_.matching() != MatchingJob::NONE)
      return ReturnCode::INACCURATE_INERTIA;
    if (!this->factored_) {
      ReturnCode ierr = this->factor();
      if (ierr != ReturnCode::SUCCESS) return ierr;
    }
    return tree()->inertia(neg, zero, pos);
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverBase<scalar_t,integer_t>::draw
  (const std::string& name) const {
    tree()->draw(*matrix(), name);
  }

  template<typename scalar_t,typename integer_t> long long
  SparseSolverBase<scalar_t,integer_t>::dense_factor_nonzeros() const {
    return tree()->dense_factor_nonzeros();
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverBase<scalar_t,integer_t>::papi_initialize() {
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
  SparseSolverBase<scalar_t,integer_t>::perf_counters_start() {
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
    b0_ = params::bytes_moved;
#endif
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverBase<scalar_t,integer_t>::perf_counters_stop
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
    bmin_ = bmax_ = btot_ = params::bytes_moved - b0_;
#endif
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverBase<scalar_t,integer_t>::print_solve_stats
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
  SparseSolverBase<scalar_t,integer_t>::print_wrong_sparsity_error() {
    std::cerr
      << "ERROR:\n"
      << "  update_matrix_values should be called with exactly\n"
      << "  the same sparsity pattern as used to set the\n"
      << "  initial matrix, using set_matrix/set_csr_matrix."
      << std::endl;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolverBase<scalar_t,integer_t>::reorder
  (int nx, int ny, int nz, int components, int width) {
    return reorder_internal(nullptr, 0, nx, ny, nz, components, width);
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolverBase<scalar_t,integer_t>::reorder
  (const int* p, int base) {
    return reorder_internal(p, base, 1, 1, 1, 1, 1);
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolverBase<scalar_t,integer_t>::reorder_internal
  (const int* p, int base, int nx, int ny, int nz,
   int components, int width) {
    if (!matrix()) return ReturnCode::MATRIX_NOT_SET;
    if (reordered_) return ReturnCode::SUCCESS;
    TaskTimer t1("permute-scale");
    int ierr;
    if (opts_.verbose() && is_root_)
      std::cout << "# matching job: " << get_description(opts_.matching())
                << std::endl;
    if (opts_.matching() != MatchingJob::NONE) {
      try {
        t1.time([&](){ matching_ = matrix()->matching(opts_.matching()); });
      } catch (std::exception& e) {
        if (is_root_) std::cerr << e.what() << std::endl;
        return ReturnCode::REORDERING_ERROR;
      }
    }

    equil_ = matrix()->equilibration();
    matrix()->equilibrate(equil_);
    if (opts_.verbose() && is_root_)
      std::cout << "# matrix equilibration, r_cond = "
                << equil_.rcond << " , c_cond = " << equil_.ccond
                << " , type = " << char(equil_.type) << std::endl;

    using real_t = typename RealType<scalar_t>::value_type;
    opts_.set_pivot_threshold
      (std::sqrt(blas::lamch<real_t>('E')) * matrix()->norm1());

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
    /* do not clear the tree data, because if we update the matrix
     * values, we want to reuse this information */
    // reordering()->clear_tree_data();
    if (opts_.verbose()) {
      auto fc = tree()->front_counter();
      if (is_root_) {
        std::cout << "# symbolic factorization:" << std::endl;
        std::cout << "#   - nr of dense Frontal matrices = "
                  << number_format_with_commas(fc.dense) << std::endl;
        switch (opts_.compression()) {
        case CompressionType::HSS:
          std::cout << "#   - nr of HSS Frontal matrices = "
                    << number_format_with_commas(fc.HSS) << std::endl;
          break;
        case CompressionType::BLR:
          std::cout << "#   - nr of BLR Frontal matrices = "
                    << number_format_with_commas(fc.BLR) << std::endl;
          break;
        case CompressionType::HODLR:
          std::cout << "#   - nr of HODLR Frontal matrices = "
                    << number_format_with_commas(fc.HODLR) << std::endl;
          break;
        case CompressionType::BLR_HODLR:
          std::cout << "#   - nr of HODLR Frontal matrices = "
                    << number_format_with_commas(fc.HODLR) << std::endl;
          std::cout << "#   - nr of BLR Frontal matrices = "
                    << number_format_with_commas(fc.BLR) << std::endl;
          break;
        case CompressionType::ZFP_BLR_HODLR:
          std::cout << "#   - nr of HODLR Frontal matrices = "
                    << number_format_with_commas(fc.HODLR) << std::endl;
          std::cout << "#   - nr of BLR Frontal matrices = "
                    << number_format_with_commas(fc.BLR) << std::endl;
          std::cout << "#   - nr of ZFP Frontal matrices = "
                    << number_format_with_commas(fc.lossy) << std::endl;
          break;
        case CompressionType::LOSSLESS:
        case CompressionType::LOSSY:
          std::cout << "#   - nr of lossy/lossless Frontal matrices = "
                    << number_format_with_commas(fc.lossy) << std::endl;
          break;
        case CompressionType::NONE:
        default: break;
        }
        std::cout << "#   - symb-factor time = " << t0.elapsed() << std::endl;
      }
    }
    perf_counters_stop("symbolic factorization");

    if (opts_.compression() != CompressionType::NONE) {
      if (is_root_) {
#if !defined(STRUMPACK_USE_BPACK)
        if (opts_.compression() == CompressionType::HODLR ||
            opts_.compression() == CompressionType::BLR_HODLR ||
            opts_.compression() == CompressionType::ZFP_BLR_HODLR) {
          std::cerr << "WARNING: Compression type requires ButterflyPACK, "
            "but STRUMPACK was not configured with ButterflyPACK support!"
                    << std::endl;
        }
#endif
#if !defined(STRUMPACK_USE_ZFP)
        if (opts_.compression() == CompressionType::ZFP_BLR_HODLR ||
            opts_.compression() == CompressionType::LOSSLESS ||
            opts_.compression() == CompressionType::LOSSY) {
          std::cerr << "WARNING: Compression type requires ZFP, "
            "but STRUMPACK was not configured with ZFP support!"
                    << std::endl;
        }
#endif
      }
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
  SparseSolverBase<scalar_t,integer_t>::flop_breakdown_reset() const {
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
  SparseSolverBase<scalar_t,integer_t>::print_flop_breakdown_HSS() const {
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
  SparseSolverBase<scalar_t,integer_t>::print_flop_breakdown_HODLR() const {
    reduce_flop_counters();
    if (!is_root_) return;
    float sample_flops = strumpack::params::CB_sample_flops +
      strumpack::params::schur_flops + strumpack::params::sparse_sample_flops;
    float compression_flops = strumpack::params::f11_fill_flops +
      strumpack::params::f12_fill_flops + strumpack::params::f21_fill_flops +
      strumpack::params::f22_fill_flops + sample_flops + params::extraction_flops;
    std::cout << std::endl;
    std::cout << "# ----- FLOP BREAKDOWN ---------------------" << std::endl;
    std::cout << "# F11_compression       = " << float(params::f11_fill_flops) << std::endl;
    std::cout << "# F12_compression       = " << float(params::f12_fill_flops) << std::endl;
    std::cout << "# F21_compression       = " << float(params::f21_fill_flops) << std::endl;
    std::cout << "# F22_compression       = " << float(params::f22_fill_flops) << std::endl;
    std::cout << "# extraction            = " << float(params::extraction_flops) << std::endl;
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
  SparseSolverBase<scalar_t,integer_t>::factor() {
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
        std::cout << "#   - minimum pivot, sqrt(eps)*|A|_1 = "
                  << opts_.pivot_threshold() << std::endl;
        std::cout << "#   - replacing of small pivots is "
                  << (opts_.replace_tiny_pivots() ? "" : "not")
                  << " enabled" << std::endl;
      }
    }
    perf_counters_start();
    flop_breakdown_reset();
    ReturnCode err_code;
    TaskTimer t1("Sparse-factorization", [&]() {
      // TODO add shift if opts_.replace...
      // auto shifted_mat = matrix_nonzero_diag();
      // err_code = tree()->multifrontal_factorization(*shifted_mat, opts_);
      err_code = tree()->multifrontal_factorization(*matrix(), opts_);
    });
    perf_counters_stop("numerical factorization");
    if (opts_.verbose()) {
      auto fnnz = factor_nonzeros();
      auto max_rank = maximum_rank();
#if defined(STRUMPACK_COUNT_FLOPS)
      auto peak_max = max_peak_memory();
      auto peak_min = min_peak_memory();
#endif
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
        std::cout << "#   - factor peak memory usage (estimate) = "
                  << peak_max / 1.0e6 << " MB (max), "
                  << peak_min / 1.0e6 << " MB (min), imbalance: "
                  << (peak_max / peak_min)
                  << std::endl;
        std::cout << "#   - factor peak device memory usage (estimate) = "
                  << double(params::peak_device_memory)/1.e6
                  << " MB" << std::endl;
#endif
        if (opts_.compression() != CompressionType::NONE) {
          std::cout << "#   - compression = " << std::boolalpha
                    << get_name(opts_.compression()) << std::endl;
          std::cout << "#   - factor memory/nonzeros = "
                    << float(fnnz) / dfnnz * 100.0
                    << " % of multifrontal" << std::endl;
          if (opts_.compression() == CompressionType::HSS) {
            std::cout << "#   - maximum HSS rank = " << max_rank << std::endl;
            std::cout << "#   - HSS relative compression tolerance = "
                      << opts_.HSS_options().rel_tol() << std::endl;
            std::cout << "#   - HSS absolute compression tolerance = "
                      << opts_.HSS_options().abs_tol() << std::endl;
            std::cout << "#   - "
                      << get_name(opts_.HSS_options().random_distribution())
                      << " distribution with "
                      << get_name(opts_.HSS_options().random_engine())
                      << " engine" << std::endl;
          }
          if (opts_.compression() == CompressionType::BLR) {
            std::cout << "#   - BLR relative compression tolerance = "
                      << opts_.BLR_options().rel_tol() << std::endl;
            std::cout << "#   - BLR absolute compression tolerance = "
                      << opts_.BLR_options().abs_tol() << std::endl;
          }
#if defined(STRUMPACK_USE_BPACK)
          if (opts_.compression() == CompressionType::HODLR) {
            std::cout << "#   - maximum HODLR rank = " << max_rank << std::endl;
            std::cout << "#   - relative compression tolerance = "
                      << opts_.HODLR_options().rel_tol() << std::endl;
            std::cout << "#   - absolute compression tolerance = "
                      << opts_.HODLR_options().abs_tol() << std::endl;
          } else if (opts_.compression() == CompressionType::BLR_HODLR) {
            std::cout << "#   - maximum HODLR rank = " << max_rank << std::endl;
            std::cout << "#   - HODLR relative compression tolerance = "
                      << opts_.HODLR_options().rel_tol() << std::endl;
            std::cout << "#   - HODLR absolute compression tolerance = "
                      << opts_.HODLR_options().abs_tol() << std::endl;
            std::cout << "#   - BLR relative compression tolerance = "
                      << opts_.BLR_options().rel_tol() << std::endl;
            std::cout << "#   - BLR absolute compression tolerance = "
                      << opts_.BLR_options().abs_tol() << std::endl;
          }
#endif
#if defined(STRUMPACK_USE_BPACK)
#if defined(STRUMPACK_USE_ZFP)
          if (opts_.compression() == CompressionType::ZFP_BLR_HODLR) {
            std::cout << "#   - maximum HODLR rank = " << max_rank << std::endl;
            std::cout << "#   - HODLR relative compression tolerance = "
                      << opts_.HODLR_options().rel_tol() << std::endl;
            std::cout << "#   - HODLR absolute compression tolerance = "
                      << opts_.HODLR_options().abs_tol() << std::endl;
            std::cout << "#   - BLR relative compression tolerance = "
                      << opts_.BLR_options().rel_tol() << std::endl;
            std::cout << "#   - BLR absolute compression tolerance = "
                      << opts_.BLR_options().abs_tol() << std::endl;
          }
#endif
#endif
#if defined(STRUMPACK_USE_ZFP)
          if (opts_.compression() == CompressionType::LOSSY)
            std::cout << "#   - lossy compression precision = "
                      << opts_.lossy_precision() << " bitplanes" << std::endl;
#endif
        }
      }
    }
    if (rank_out_) tree()->print_rank_statistics(*rank_out_);
    // if (err_code == ReturnCode::SUCCESS)
    factored_ = true;
    return err_code;
  }


  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolverBase<scalar_t,integer_t>::solve
  (const scalar_t* b, scalar_t* x, bool use_initial_guess) {
    return solve_internal(b, x, use_initial_guess);
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolverBase<scalar_t,integer_t>::solve
  (const DenseM_t& b, DenseM_t& x, bool use_initial_guess) {
    return solve_internal(b, x, use_initial_guess);
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolverBase<scalar_t,integer_t>::solve
  (int nrhs, const scalar_t* b, int ldb, scalar_t* x, int ldx,
   bool use_initial_guess) {
    return solve_internal(nrhs, b, ldb, x, ldx, use_initial_guess);
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  SparseSolverBase<scalar_t,integer_t>::solve_internal
  (int nrhs, const scalar_t* b, int ldb, scalar_t* x, int ldx,
   bool use_initial_guess) {
    if (!nrhs) return ReturnCode::SUCCESS;
    auto N = matrix()->size();
    assert(ldb >= N);
    assert(ldx >= N);
    assert(nrhs >= 1);
    auto B = ConstDenseMatrixWrapperPtr(N, nrhs, b, ldb);
    DenseMW_t X(N, nrhs, x, N);
    return this->solve(*B, X, use_initial_guess);
  }

  template<typename scalar_t,typename integer_t> void
  SparseSolverBase<scalar_t,integer_t>::delete_factors() {
    delete_factors_internal();
    factored_ = false;
  }

  // explicit template instantiations
  template class SparseSolverBase<float,int>;
  template class SparseSolverBase<double,int>;
  template class SparseSolverBase<std::complex<float>,int>;
  template class SparseSolverBase<std::complex<double>,int>;

  template class SparseSolverBase<float,long int>;
  template class SparseSolverBase<double,long int>;
  template class SparseSolverBase<std::complex<float>,long int>;
  template class SparseSolverBase<std::complex<double>,long int>;

  template class SparseSolverBase<float,long long int>;
  template class SparseSolverBase<double,long long int>;
  template class SparseSolverBase<std::complex<float>,long long int>;
  template class SparseSolverBase<std::complex<double>,long long int>;

} //end namespace strumpack
