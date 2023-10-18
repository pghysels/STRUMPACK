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
#include <algorithm>
#include <vector>
#include <memory>

#include "EliminationTreeMPI.hpp"

#include "ordering/MatrixReorderingMPI.hpp"
#include "dense/DistributedMatrix.hpp"
#include "fronts/FrontFactory.hpp"
#include "fronts/FrontalMatrixMPI.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  EliminationTreeMPI<scalar_t,integer_t>::EliminationTreeMPI
  (const MPIComm& comm) : EliminationTree<scalar_t,integer_t>(),
    comm_(comm), rank_(comm.rank()), P_(comm.size()) {
  }

  template<typename scalar_t,typename integer_t>
  EliminationTreeMPI<scalar_t,integer_t>::~EliminationTreeMPI() {}

  template<typename scalar_t,typename integer_t> FrontCounter
  EliminationTreeMPI<scalar_t,integer_t>::front_counter() const {
    return this->nr_fronts_.reduce(comm_);
  }

  template<typename scalar_t,typename integer_t> void
  EliminationTreeMPI<scalar_t,integer_t>::update_local_ranges
  (integer_t lo, integer_t hi) {
    local_range_.first  = std::min(local_range_.first, lo);
    local_range_.second = std::max(local_range_.second, hi);
  }

  template<typename scalar_t,typename integer_t> integer_t
  EliminationTreeMPI<scalar_t,integer_t>::maximum_rank() const {
    return comm_.all_reduce
      (EliminationTree<scalar_t,integer_t>::maximum_rank(), MPI_MAX);
  }

  template<typename scalar_t,typename integer_t> long long
  EliminationTreeMPI<scalar_t,integer_t>::factor_nonzeros() const {
    return comm_.all_reduce
      (EliminationTree<scalar_t,integer_t>::factor_nonzeros(), MPI_SUM);
  }

  template<typename scalar_t,typename integer_t> long long
  EliminationTreeMPI<scalar_t,integer_t>::dense_factor_nonzeros() const {
    return comm_.all_reduce
      (EliminationTree<scalar_t,integer_t>::dense_factor_nonzeros(), MPI_SUM);
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  EliminationTreeMPI<scalar_t,integer_t>::inertia
  (integer_t& neg, integer_t& zero, integer_t& pos) const {
    auto info = EliminationTree<scalar_t,integer_t>::inertia(neg, zero, pos);
    neg  = comm_.all_reduce(neg,  MPI_SUM);
    zero = comm_.all_reduce(zero, MPI_SUM);
    pos  = comm_.all_reduce(pos,  MPI_SUM);
    return info;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  EliminationTreeMPI<scalar_t,integer_t>::subnormals
  (std::size_t& ns, std::size_t& nz) const {
    auto info = EliminationTree<scalar_t,integer_t>::subnormals(ns, nz);
    ns = comm_.all_reduce(ns, MPI_SUM);
    nz = comm_.all_reduce(nz, MPI_SUM);
    return info;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  EliminationTreeMPI<scalar_t,integer_t>::pivot_growth
  (scalar_t& pgL, scalar_t& pgU) const {
    auto info = EliminationTree<scalar_t,integer_t>::pivot_growth(pgL, pgU);
    pgL = comm_.all_reduce(pgL, MPI_MAX);
    pgU = comm_.all_reduce(pgU, MPI_MAX);
    return info;
  }


  // explicit template specializations
  template class EliminationTreeMPI<float,int>;
  template class EliminationTreeMPI<double,int>;
  template class EliminationTreeMPI<std::complex<float>,int>;
  template class EliminationTreeMPI<std::complex<double>,int>;

  template class EliminationTreeMPI<float,long int>;
  template class EliminationTreeMPI<double,long int>;
  template class EliminationTreeMPI<std::complex<float>,long int>;
  template class EliminationTreeMPI<std::complex<double>,long int>;

  template class EliminationTreeMPI<float,long long int>;
  template class EliminationTreeMPI<double,long long int>;
  template class EliminationTreeMPI<std::complex<float>,long long int>;
  template class EliminationTreeMPI<std::complex<double>,long long int>;

} // end namespace strumpack
