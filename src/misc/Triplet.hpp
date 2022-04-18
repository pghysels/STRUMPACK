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
#ifndef STRUMPACK_TRIPLET_HPP
#define STRUMPACK_TRIPLET_HPP

#include "dense/DenseMatrix.hpp"
#include "StrumpackConfig.hpp"
#if defined(STRUMPACK_USE_MPI) && !defined(STRUMPACK_NO_TRIPLET_MPI)
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#endif

namespace strumpack {

  template<typename scalar_t, typename integer_t=int>
  class Triplet {
  public:
    integer_t r, c;
    scalar_t v;
    Triplet() {}
    Triplet(integer_t row, integer_t col, scalar_t value)
      : r(row), c(col), v(value) {}

#if defined(STRUMPACK_USE_MPI) && !defined(STRUMPACK_NO_TRIPLET_MPI)
    static MPI_Datatype triplet_mpi_type;
    static MPI_Datatype mpi_type();
    static void free_mpi_type();
#endif
  };


  template<typename scalar_t, typename integer_t=int>
  class IdxVal {
  public:
    integer_t i;
    scalar_t v;
    IdxVal() {}
    IdxVal(integer_t idx, scalar_t value) : i(idx), v(value) {}

#if defined(STRUMPACK_USE_MPI) && !defined(STRUMPACK_NO_TRIPLET_MPI)
    static MPI_Datatype idxval_mpi_type;
    static MPI_Datatype mpi_type();
    static void free_mpi_type();
#endif
  };

  template<typename integer_t=int>
  class IdxIJ {
  public:
    integer_t i, j;
    IdxIJ() {}
    IdxIJ(integer_t ii, integer_t jj) : i(ii), j(jj) {}

#if defined(STRUMPACK_USE_MPI) && !defined(STRUMPACK_NO_TRIPLET_MPI)
    static MPI_Datatype idxij_mpi_type;
    static MPI_Datatype mpi_type();
    static void free_mpi_type();
#endif
  };

  template<typename scalar_t, typename integer_t=int>
  class Quadlet {
  public:
    integer_t r, c, k;
    scalar_t v;
    Quadlet() {}
    Quadlet(const Triplet<scalar_t,integer_t>& t, integer_t k)
      : Quadlet(t.r, t.c, k, t.v) {}
    Quadlet(integer_t row, integer_t col, integer_t block, scalar_t val)
      : r(row), c(col), k(block), v(val) {}

#if defined(STRUMPACK_USE_MPI) && !defined(STRUMPACK_NO_TRIPLET_MPI)
    static MPI_Datatype quadlet_mpi_type;
    static MPI_Datatype mpi_type();
    static void free_mpi_type();
#endif
  };

} // end namespace strumpack

#endif // STRUMPACK_TRIPLET_HPP
