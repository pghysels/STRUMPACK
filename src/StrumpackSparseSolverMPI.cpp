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

#include "StrumpackSparseSolverMPI.hpp"
#include "sparse/EliminationTreeMPI.hpp"
#include "sparse/ordering/MatrixReorderingMPI.hpp"


namespace strumpack {

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolverMPI<scalar_t,integer_t>::StrumpackSparseSolverMPI
  (MPI_Comm comm, bool verbose)
    : StrumpackSparseSolverMPI<scalar_t,integer_t>
    (comm, 0, nullptr, verbose) {
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolverMPI<scalar_t,integer_t>::StrumpackSparseSolverMPI
  (MPI_Comm comm, int argc, char* argv[], bool verbose) :
    StrumpackSparseSolver<scalar_t,integer_t>
    (argc, argv, verbose, mpi_rank(comm) == 0), comm_(comm) {
    if (opts_.verbose() && is_root_)
      std::cout << "# using " << comm_.size()
                << " MPI processes" << std::endl;
  }

  template<typename scalar_t,typename integer_t>
  StrumpackSparseSolverMPI<scalar_t,integer_t>::
  ~StrumpackSparseSolverMPI() = default;

  template<typename scalar_t,typename integer_t> MPI_Comm
  StrumpackSparseSolverMPI<scalar_t,integer_t>::comm() const {
    return comm_.comm();
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::set_matrix
  (const CSRMatrix<scalar_t,integer_t>& A) {
    mat_.reset(new CSRMatrix<scalar_t,integer_t>(A));
    this->factored_ = this->reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::set_csr_matrix
  (integer_t N, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, bool symmetric_pattern) {
    mat_.reset
      (new CSRMatrix<scalar_t,integer_t>
       (N, row_ptr, col_ind, values, symmetric_pattern));
    this->factored_ = this->reordered_ = false;
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::setup_tree() {
    tree_mpi_.reset
      (new EliminationTreeMPI<scalar_t,integer_t>(opts_, *mat_, *nd_, comm_));
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::setup_reordering() {
    nd_.reset(new MatrixReordering<scalar_t,integer_t>(mat_->size()));
  }

  template<typename scalar_t,typename integer_t> int
  StrumpackSparseSolverMPI<scalar_t,integer_t>::compute_reordering
  (const int* p, int base, int nx, int ny, int nz,
   int components, int width) {
    if (p) return nd_->set_permutation(opts_, *mat_, comm_, p, base);
    return nd_->nested_dissection
      (opts_, *mat_, comm_, nx, ny, nz, components, width);
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::separator_reordering() {
    nd_->separator_reordering(opts_, *mat_, tree_mpi_->root());
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::perf_counters_stop
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
      long long int flopsbytes[2] = {df, params::bytes - this->b0_};
      comm_.all_reduce(flopsbytes, 2, MPI_SUM);
      this->ftot_ = flopsbytes[0];
      this->btot_ = flopsbytes[1];
      this->fmin_ = comm_.all_reduce(df, MPI_MIN);
      this->fmax_ = comm_.all_reduce(df, MPI_MAX);
#endif
    }
  }

  template<typename scalar_t,typename integer_t> void
  StrumpackSparseSolverMPI<scalar_t,integer_t>::reduce_flop_counters() const {
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

  // explicit template instantiations
  template class StrumpackSparseSolverMPI<float,int>;
  template class StrumpackSparseSolverMPI<double,int>;
  template class StrumpackSparseSolverMPI<std::complex<float>,int>;
  template class StrumpackSparseSolverMPI<std::complex<double>,int>;

  template class StrumpackSparseSolverMPI<float,long int>;
  template class StrumpackSparseSolverMPI<double,long int>;
  template class StrumpackSparseSolverMPI<std::complex<float>,long int>;
  template class StrumpackSparseSolverMPI<std::complex<double>,long int>;

  template class StrumpackSparseSolverMPI<float,long long int>;
  template class StrumpackSparseSolverMPI<double,long long int>;
  template class StrumpackSparseSolverMPI<std::complex<float>,long long int>;
  template class StrumpackSparseSolverMPI<std::complex<double>,long long int>;

} // end namespace strumpack
