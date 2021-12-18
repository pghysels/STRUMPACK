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
#include <complex.h>
#undef I
#include <complex>
#include <iostream>

#include "StructuredMatrix.h"
#include "StructuredMatrix.hpp"
using namespace strumpack;
using namespace strumpack::structured;

template<typename scalar_t> void set_def_options(CSPOptions* opts) {
  structured::StructuredOptions<scalar_t> o;
  opts->type = static_cast<SP_STRUCTURED_TYPE>(o.type());
  opts->rel_tol = o.rel_tol();
  opts->abs_tol = o.abs_tol();
  opts->leaf_size = o.leaf_size();
  opts->max_rank = o.max_rank();
  opts->verbose = o.verbose();
}

template<typename scalar_t> structured::StructuredOptions<scalar_t>
get_options(const CSPOptions* o) {
  structured::StructuredOptions<scalar_t> opts;
  opts.set_type(static_cast<structured::Type>(o->type));
  opts.set_rel_tol(o->rel_tol);
  opts.set_abs_tol(o->abs_tol);
  opts.set_leaf_size(o->leaf_size);
  opts.set_max_rank(o->max_rank);
  opts.set_verbose(o->verbose);
  return opts;
}

template<typename scalar_t> struct CStructMat_ {
  std::unique_ptr<structured::StructuredMatrix<scalar_t>> S;
#if defined(STRUMPACK_USE_MPI)
  MPIComm comm;
  BLACSGrid grid;
#endif
};


template<typename scalar_t> structured::StructuredMatrix<scalar_t>*
get_mat(CSPStructMat S) {
  return reinterpret_cast<CStructMat_<scalar_t>*>(S)->S.get();
}

template<typename scalar_t> CStructMat_<scalar_t>*
create_mat() {
  return new CStructMat_<scalar_t>();
}


extern "C" {

  void SP_s_struct_default_options(CSPOptions* opts) { set_def_options<float>(opts); }
  void SP_d_struct_default_options(CSPOptions* opts) { set_def_options<double>(opts); }
  void SP_c_struct_default_options(CSPOptions* opts) { set_def_options<std::complex<float>>(opts); }
  void SP_z_struct_default_options(CSPOptions* opts) { set_def_options<std::complex<double>>(opts); }


  void SP_s_struct_destroy(CSPStructMat* S) {
    delete static_cast<CStructMat_<float>*>(*S);
    *S = NULL;
  }
  void SP_d_struct_destroy(CSPStructMat* S) {
    delete static_cast<CStructMat_<double>*>(*S);
    *S = NULL;
  }
  void SP_c_struct_destroy(CSPStructMat* S) {
    delete static_cast<CStructMat_<std::complex<float>>*>(*S);
    *S = NULL;
  }
  void SP_z_struct_destroy(CSPStructMat* S) {
    delete static_cast<CStructMat_<std::complex<double>>*>(*S);
    *S = NULL;
  }


  int SP_s_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             const float* A, int ldA,
                             const CSPOptions* opts) {
    try {
      auto s = create_mat<float>();
      s->S = construct_from_dense<float>
        (rows, cols, A, ldA, get_options<float>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_d_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             const double* A, int ldA,
                             const CSPOptions* opts) {
    try {
      auto s = create_mat<double>();
      s->S = construct_from_dense<double>
        (rows, cols, A, ldA, get_options<double>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_c_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             const float _Complex* A, int ldA,
                             const CSPOptions* opts) {
    try {
      auto s = create_mat<std::complex<float>>();
      s->S = construct_from_dense<std::complex<float>>
        (rows, cols, reinterpret_cast<const std::complex<float>*>(A), ldA,
         get_options<std::complex<float>>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_z_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             const double _Complex* A, int ldA,
                             const CSPOptions* opts) {
    try {
      auto s = create_mat<std::complex<double>>();
      s->S = construct_from_dense<std::complex<double>>
        (rows, cols, reinterpret_cast<const std::complex<double>*>(A), ldA,
         get_options<std::complex<double>>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }


  int SP_s_struct_from_elements(CSPStructMat* S, int rows, int cols,
                                float A(int,int),
                                const CSPOptions* opts) {
    try {
      auto s = create_mat<float>();
      s->S = construct_from_elements<float>
        (rows, cols, [&A](int i, int j) { return A(i,j); },
         get_options<float>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_d_struct_from_elements(CSPStructMat* S, int rows, int cols,
                                double A(int,int),
                                const CSPOptions* opts) {
    try {
      auto s = create_mat<double>();
      s->S = construct_from_elements<double>
        (rows, cols, [&A](int i, int j) { return A(i,j); },
         get_options<double>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_c_struct_from_elements(CSPStructMat* S, int rows, int cols,
                                float _Complex A(int,int),
                                const CSPOptions* opts) {
    try {
      auto s = create_mat<std::complex<float>>();
      s->S = construct_from_elements<std::complex<float>>
        (rows, cols,
         [&A](int i, int j) -> std::complex<float> {
          auto Aij = A(i,j);
          return reinterpret_cast<std::complex<float>&>(Aij); },
         get_options<std::complex<float>>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_z_struct_from_elements(CSPStructMat* S, int rows, int cols,
                                double _Complex A(int,int),
                                const CSPOptions* opts) {
    try {
      auto s = create_mat<std::complex<double>>();
      s->S = construct_from_elements<std::complex<double>>
        (rows, cols,
         [&A](int i, int j) -> std::complex<double> {
          auto Aij = A(i,j);
          return reinterpret_cast<std::complex<double>&>(Aij); },
         get_options<std::complex<double>>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }


#if defined(STRUMPACK_USE_MPI)
  int SP_s_struct_from_dense2d(CSPStructMat* S, const MPI_Comm comm,
                               int rows, int cols, const float* A,
                               int IA, int JA, int* DESCA,
                               const CSPOptions* opts) {
    try {
      auto s = create_mat<float>();
      s->comm = MPIComm(comm);
      s->grid = BLACSGrid(s->comm);
      DistributedMatrix<float> A2d(&s->grid, rows, cols);
      scalapack::pgemr2d
        (rows, cols, A, IA, JA, DESCA,
         A2d.data(), A2d.I(), A2d.J(), A2d.desc(), A2d.ctxt_all());
      s->S = construct_from_dense<float>
        (A2d, get_options<float>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_d_struct_from_dense2d(CSPStructMat* S, const MPI_Comm comm,
                               int rows, int cols, const double* A,
                               int IA, int JA, int* DESCA,
                               const CSPOptions* opts) {
    try {
      auto s = create_mat<double>();
      s->comm = MPIComm(comm);
      s->grid = BLACSGrid(s->comm);
      DistributedMatrix<double> A2d(&s->grid, rows, cols);
      scalapack::pgemr2d
        (rows, cols, A, IA, JA, DESCA,
         A2d.data(), A2d.I(), A2d.J(), A2d.desc(), A2d.ctxt_all());
      s->S = construct_from_dense<double>
        (A2d, get_options<double>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_c_struct_from_dense2d(CSPStructMat* S, const MPI_Comm comm,
                               int rows, int cols, const float _Complex* A,
                               int IA, int JA, int* DESCA,
                               const CSPOptions* opts) {
    try {
      auto s = create_mat<std::complex<float>>();
      s->comm = MPIComm(comm);
      s->grid = BLACSGrid(s->comm);
      DistributedMatrix<std::complex<float>> A2d(&s->grid, rows, cols);
      scalapack::pgemr2d
        (rows, cols, reinterpret_cast<const std::complex<float>*>(A), IA, JA, DESCA,
         A2d.data(), A2d.I(), A2d.J(), A2d.desc(), A2d.ctxt_all());

      s->S = construct_from_dense<std::complex<float>>
        (A2d, get_options<std::complex<float>>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_z_struct_from_dense2d(CSPStructMat* S, const MPI_Comm comm,
                               int rows, int cols, const double _Complex* A,
                               int IA, int JA, int* DESCA,
                               const CSPOptions* opts) {
    try {
      auto s = create_mat<std::complex<double>>();
      s->comm = MPIComm(comm);
      s->grid = BLACSGrid(s->comm);
      DistributedMatrix<std::complex<double>> A2d(&s->grid, rows, cols);
      scalapack::pgemr2d
        (rows, cols,
         reinterpret_cast<const std::complex<double>*>(A), IA, JA, DESCA,
         A2d.data(), A2d.I(), A2d.J(), A2d.desc(), A2d.ctxt_all());
      s->S = construct_from_dense<std::complex<double>>
        (A2d, get_options<std::complex<double>>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }



  int SP_s_struct_from_elements_mpi(CSPStructMat* S,
                                    const MPI_Comm comm,
                                    int rows, int cols,
                                    float A(int,int),
                                    const CSPOptions* opts) {
    try {
      auto s = create_mat<float>();
      s->comm = MPIComm(comm);
      s->S = construct_from_elements<float>
        (s->comm, rows, cols,
         [&A](int i, int j) { return A(i,j); },
         get_options<float>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_d_struct_from_elements_mpi(CSPStructMat* S,
                                    const MPI_Comm comm,
                                    int rows, int cols,
                                    double A(int,int),
                                    const CSPOptions* opts) {
    try {
      auto s = create_mat<double>();
      s->comm = MPIComm(comm);
      s->S = construct_from_elements<double>
        (s->comm, rows, cols,
         [&A](int i, int j) { return A(i,j); },
         get_options<double>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_c_struct_from_elements_mpi(CSPStructMat* S,
                                    const MPI_Comm comm,
                                    int rows, int cols,
                                    float _Complex A(int,int),
                                    const CSPOptions* opts) {
    try {
      auto s = create_mat<std::complex<float>>();
      s->comm = MPIComm(comm);
      s->S = construct_from_elements<std::complex<float>>
        (s->comm, rows, cols,
         [&A](int i, int j) -> std::complex<float> {
          auto aij = A(i, j);
          return reinterpret_cast<std::complex<float>&>(aij); },
         get_options<std::complex<float>>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_z_struct_from_elements_mpi(CSPStructMat* S,
                                    const MPI_Comm comm,
                                    int rows, int cols,
                                    double _Complex A(int,int),
                                    const CSPOptions* opts) {
    try {
      auto s = create_mat<std::complex<double>>();
      s->comm = MPIComm(comm);
      s->S = construct_from_elements<std::complex<double>>
        (s->comm, rows, cols,
         [&A](int i, int j) -> std::complex<double> {
          auto aij = A(i, j);
          return reinterpret_cast<std::complex<double>&>(aij); },
         get_options<std::complex<double>>(opts));
      *S = s;
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }


  int SP_s_struct_rows(const CSPStructMat S) {
    try {
      return get_mat<float>(S)->rows();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_d_struct_rows(const CSPStructMat S) {
    try {
      return get_mat<double>(S)->rows();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_c_struct_rows(const CSPStructMat S) {
    try {
      return get_mat<std::complex<float>>(S)->rows();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_z_struct_rows(const CSPStructMat S) {
    try {
      return get_mat<std::complex<double>>(S)->rows();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }


  int SP_s_struct_cols(const CSPStructMat S) {
    try {
      return get_mat<float>(S)->cols();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_d_struct_cols(const CSPStructMat S) {
    try {
      return get_mat<double>(S)->cols();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_c_struct_cols(const CSPStructMat S) {
    try {
      return get_mat<std::complex<float>>(S)->cols();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_z_struct_cols(const CSPStructMat S) {
    try {
      return get_mat<std::complex<double>>(S)->cols();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }


  long long int SP_s_struct_memory(const CSPStructMat S) {
    try {
      return get_mat<float>(S)->memory();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  long long int SP_d_struct_memory(const CSPStructMat S) {
    try {
      return get_mat<double>(S)->memory();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  long long int SP_c_struct_memory(const CSPStructMat S) {
    try {
      return get_mat<std::complex<float>>(S)->memory();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  long long int SP_z_struct_memory(const CSPStructMat S) {
    try {
      return get_mat<std::complex<double>>(S)->memory();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }

  long long int SP_s_struct_nonzeros(const CSPStructMat S) {
    try {
      return get_mat<float>(S)->nonzeros();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  long long int SP_d_struct_nonzeros(const CSPStructMat S) {
    try {
      return get_mat<double>(S)->nonzeros();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  long long int SP_c_struct_nonzeros(const CSPStructMat S) {
    try {
      return get_mat<std::complex<float>>(S)->nonzeros();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  long long int SP_z_struct_nonzeros(const CSPStructMat S) {
    try {
      return get_mat<std::complex<double>>(S)->nonzeros();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }

  int SP_s_struct_rank(const CSPStructMat S) {
    try {
      return get_mat<float>(S)->rank();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_d_struct_rank(const CSPStructMat S) {
    try {
      return get_mat<double>(S)->rank();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_c_struct_rank(const CSPStructMat S) {
    try {
      return get_mat<std::complex<float>>(S)->rank();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_z_struct_rank(const CSPStructMat S) {
    try {
      return get_mat<std::complex<double>>(S)->rank();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }

  int SP_s_struct_local_rows(const CSPStructMat S) {
    try {
      return get_mat<float>(S)->local_rows();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_d_struct_local_rows(const CSPStructMat S) {
    try {
      return get_mat<double>(S)->local_rows();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_c_struct_local_rows(const CSPStructMat S) {
    try {
      return get_mat<std::complex<float>>(S)->local_rows();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_z_struct_local_rows(const CSPStructMat S) {
    try {
      return get_mat<std::complex<double>>(S)->local_rows();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }


  int SP_s_struct_begin_row(const CSPStructMat S) {
    try {
      return get_mat<float>(S)->begin_row();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_d_struct_begin_row(const CSPStructMat S) {
    try {
      return get_mat<double>(S)->begin_row();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_c_struct_begin_row(const CSPStructMat S) {
    try {
      return get_mat<std::complex<float>>(S)->begin_row();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }
  int SP_z_struct_begin_row(const CSPStructMat S) {
    try {
      return get_mat<std::complex<double>>(S)->begin_row();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
    }
    return -1;
  }

#endif




  int SP_s_struct_mult(CSPStructMat S, char trans, int m,
                       const float* B, int ldB,
                       float* C, int ldC) {
    try {
      get_mat<float>(S)->mult(c2T(trans), m, B, ldB, C, ldC);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_d_struct_mult(CSPStructMat S, char trans, int m,
                       const double* B, int ldB,
                       double* C, int ldC) {
    try {
      get_mat<double>(S)->mult(c2T(trans), m, B, ldB, C, ldC);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_c_struct_mult(CSPStructMat S, char trans, int m,
                       const float _Complex* B, int ldB,
                       float _Complex* C, int ldC) {
    try {
      get_mat<std::complex<float>>(S)->
        mult(c2T(trans), m,
             reinterpret_cast<const std::complex<float>*>(B), ldB,
             reinterpret_cast<std::complex<float>*>(C), ldC);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_z_struct_mult(CSPStructMat S, char trans, int m,
                       const double _Complex* B, int ldB,
                       double _Complex* C, int ldC) {
    try {
      get_mat<std::complex<double>>(S)->
        mult(c2T(trans), m,
             reinterpret_cast<const std::complex<double>*>(B), ldB,
             reinterpret_cast<std::complex<double>*>(C), ldC);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }

  int SP_s_struct_factor(CSPStructMat S) {
    try {
      get_mat<float>(S)->factor();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_d_struct_factor(CSPStructMat S) {
    try {
      get_mat<double>(S)->factor();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_c_struct_factor(CSPStructMat S) {
    try {
      get_mat<std::complex<float>>(S)->factor();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_z_struct_factor(CSPStructMat S) {
    try {
      get_mat<std::complex<double>>(S)->factor();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }

  int SP_s_struct_solve(CSPStructMat S, int nrhs, float* B, int ldB) {
    try {
      get_mat<float>(S)->solve(nrhs, B, ldB);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_d_struct_solve(CSPStructMat S, int nrhs, double* B, int ldB) {
    try {
      get_mat<double>(S)->solve(nrhs, B, ldB);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_c_struct_solve(CSPStructMat S, int nrhs,
                        float _Complex* B, int ldB) {
    try {
      get_mat<std::complex<float>>(S)->
        solve(nrhs, reinterpret_cast<std::complex<float>*>(B), ldB);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_z_struct_solve(CSPStructMat S, int nrhs,
                        double _Complex* B, int ldB) {
    try {
      get_mat<std::complex<double>>(S)->
        solve(nrhs, reinterpret_cast<std::complex<double>*>(B), ldB);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }


  int SP_s_struct_shift(CSPStructMat S, float s) {
    try {
      get_mat<float>(S)->shift(s);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_d_struct_shift(CSPStructMat S, double s) {
    try {
      get_mat<double>(S)->shift(s);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_c_struct_shift(CSPStructMat S, float _Complex s) {
    try {
      get_mat<std::complex<float>>(S)->shift
        (reinterpret_cast<std::complex<float>&>(s));
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_z_struct_shift(CSPStructMat S, double _Complex s) {
    try {
      get_mat<std::complex<double>>(S)->shift
        (reinterpret_cast<std::complex<double>&>(s));
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }


}

