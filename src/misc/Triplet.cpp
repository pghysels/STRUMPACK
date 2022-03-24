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
#include <cstddef>

#include "StrumpackConfig.hpp"
#include "Triplet.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "MPIWrapper.hpp"
#endif

namespace strumpack {

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> MPI_Datatype
  Triplet<scalar_t,integer_t>::triplet_mpi_type = MPI_DATATYPE_NULL;

  template<typename scalar_t,typename integer_t> void
  Triplet<scalar_t,integer_t>::free_mpi_type() {
    if (triplet_mpi_type != MPI_DATATYPE_NULL) {
      MPI_Type_free(&triplet_mpi_type);
      triplet_mpi_type = MPI_DATATYPE_NULL;
    }
  }

  template<typename scalar_t,typename integer_t> MPI_Datatype
  Triplet<scalar_t,integer_t>::mpi_type() {
    if (triplet_mpi_type == MPI_DATATYPE_NULL) {
      using T = Triplet<scalar_t,integer_t>;
      const int count = 3;
      int b[count] = {1, 1, 1};
      MPI_Datatype t[count] = {strumpack::mpi_type<integer_t>(),
                               strumpack::mpi_type<integer_t>(),
                               strumpack::mpi_type<scalar_t>()};
      MPI_Aint o[count] = {offsetof(T, r), offsetof(T, c), offsetof(T, v)};
      MPI_Datatype tmp_mpi_type;
      MPI_Type_create_struct(count, b, o, t, &tmp_mpi_type);
      MPI_Type_create_resized(tmp_mpi_type, 0, sizeof(T), &triplet_mpi_type);
      MPI_Type_free(&tmp_mpi_type);
      MPI_Type_commit(&triplet_mpi_type);
    }
    return triplet_mpi_type;
  }

  template<typename scalar_t,typename integer_t> MPI_Datatype
  IdxVal<scalar_t,integer_t>::idxval_mpi_type = MPI_DATATYPE_NULL;

  template<typename scalar_t,typename integer_t> void
  IdxVal<scalar_t,integer_t>::free_mpi_type() {
    if (idxval_mpi_type != MPI_DATATYPE_NULL) {
      MPI_Type_free(&idxval_mpi_type);
      idxval_mpi_type = MPI_DATATYPE_NULL;
    }
  }

  template<typename scalar_t,typename integer_t> MPI_Datatype
  IdxVal<scalar_t,integer_t>::mpi_type() {
    if (idxval_mpi_type == MPI_DATATYPE_NULL) {
      using T = IdxVal<scalar_t,integer_t>;
      const int count = 2;
      int b[count] = {1, 1};
      MPI_Datatype t[count] = {strumpack::mpi_type<integer_t>(),
                               strumpack::mpi_type<scalar_t>()};
      MPI_Aint o[count] = {offsetof(T, i), offsetof(T, v)};
      MPI_Datatype tmp_mpi_type;
      MPI_Type_create_struct(count, b, o, t, &tmp_mpi_type);
      MPI_Type_create_resized(tmp_mpi_type, 0, sizeof(T), &idxval_mpi_type);
      MPI_Type_free(&tmp_mpi_type);
      MPI_Type_commit(&idxval_mpi_type);
    }
    return idxval_mpi_type;
  }

  template<typename integer_t> MPI_Datatype
  IdxIJ<integer_t>::idxij_mpi_type = MPI_DATATYPE_NULL;

  template<typename integer_t> void
  IdxIJ<integer_t>::free_mpi_type() {
    if (idxij_mpi_type != MPI_DATATYPE_NULL) {
      MPI_Type_free(&idxij_mpi_type);
      idxij_mpi_type = MPI_DATATYPE_NULL;
    }
  }

  template<typename integer_t> MPI_Datatype
  IdxIJ<integer_t>::mpi_type() {
    if (idxij_mpi_type == MPI_DATATYPE_NULL) {
      MPI_Type_contiguous
        (2, strumpack::mpi_type<integer_t>(), &idxij_mpi_type);
      MPI_Type_commit(&idxij_mpi_type);
    }
    return idxij_mpi_type;
  }


  template<typename scalar_t,typename integer_t> MPI_Datatype
  Quadlet<scalar_t,integer_t>::quadlet_mpi_type = MPI_DATATYPE_NULL;

  template<typename scalar_t,typename integer_t> void
  Quadlet<scalar_t,integer_t>::free_mpi_type() {
    if (quadlet_mpi_type != MPI_DATATYPE_NULL) {
      MPI_Type_free(&quadlet_mpi_type);
      quadlet_mpi_type = MPI_DATATYPE_NULL;
    }
  }

  template<typename scalar_t,typename integer_t> MPI_Datatype
  Quadlet<scalar_t,integer_t>::mpi_type() {
    if (quadlet_mpi_type == MPI_DATATYPE_NULL) {
      using T = Quadlet<scalar_t,integer_t>;
      const int count = 4;
      int b[count] = {1, 1, 1, 1};
      MPI_Datatype t[count] = {strumpack::mpi_type<integer_t>(),
                               strumpack::mpi_type<integer_t>(),
                               strumpack::mpi_type<integer_t>(),
                               strumpack::mpi_type<scalar_t>()};
      MPI_Aint o[count] = {offsetof(T, r), offsetof(T, c),
                           offsetof(T, k), offsetof(T, v)};
      MPI_Datatype tmp_mpi_type;
      MPI_Type_create_struct(count, b, o, t, &tmp_mpi_type);
      MPI_Type_create_resized(tmp_mpi_type, 0, sizeof(T), &quadlet_mpi_type);
      MPI_Type_free(&tmp_mpi_type);
      MPI_Type_commit(&quadlet_mpi_type);
    }
    return quadlet_mpi_type;
  }
#endif

  // explicit template instantiations
  template class Triplet<float,int>;
  template class Triplet<double,int>;
  template class Triplet<std::complex<float>,int>;
  template class Triplet<std::complex<double>,int>;

  template class Triplet<float,long int>;
  template class Triplet<double,long int>;
  template class Triplet<std::complex<float>,long int>;
  template class Triplet<std::complex<double>,long int>;

  template class Triplet<float,long long int>;
  template class Triplet<double,long long int>;
  template class Triplet<std::complex<float>,long long int>;
  template class Triplet<std::complex<double>,long long int>;


  template class IdxVal<float,int>;
  template class IdxVal<double,int>;
  template class IdxVal<std::complex<float>,int>;
  template class IdxVal<std::complex<double>,int>;

  template class IdxVal<float,long int>;
  template class IdxVal<double,long int>;
  template class IdxVal<std::complex<float>,long int>;
  template class IdxVal<std::complex<double>,long int>;

  template class IdxVal<float,long long int>;
  template class IdxVal<double,long long int>;
  template class IdxVal<std::complex<float>,long long int>;
  template class IdxVal<std::complex<double>,long long int>;


  template class IdxIJ<int>;
  template class IdxIJ<long int>;
  template class IdxIJ<long long int>;


  template class Quadlet<float,int>;
  template class Quadlet<double,int>;
  template class Quadlet<std::complex<float>,int>;
  template class Quadlet<std::complex<double>,int>;
} // end namespace strumpack
