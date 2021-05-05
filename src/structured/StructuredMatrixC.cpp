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
get_options(CSPOptions* o) {
  structured::StructuredOptions<scalar_t> opts;
  opts.set_type(static_cast<structured::Type>(o->type));
  opts.set_rel_tol(o->rel_tol);
  opts.set_abs_tol(o->abs_tol);
  opts.set_leaf_size(o->leaf_size);
  opts.set_max_rank(o->max_rank);
  opts.set_verbose(o->verbose);
  return opts;
}


extern "C" {

  void SP_s_struct_default_options(CSPOptions* opts) { set_def_options<float>(opts); }
  void SP_d_struct_default_options(CSPOptions* opts) { set_def_options<double>(opts); }
  void SP_c_struct_default_options(CSPOptions* opts) { set_def_options<std::complex<float>>(opts); }
  void SP_z_struct_default_options(CSPOptions* opts) { set_def_options<std::complex<double>>(opts); }


  void SP_s_struct_destroy(CSPStructMat* S) {
    auto H = reinterpret_cast<StructuredMatrix<float>*>(*S);
    delete H;
    H = NULL;
  }
  void SP_d_struct_destroy(CSPStructMat* S) {
    auto H = reinterpret_cast<StructuredMatrix<double>*>(*S);
    delete H;
    H = NULL;
  }
  void SP_c_struct_destroy(CSPStructMat* S) {
    auto H = reinterpret_cast<StructuredMatrix<float _Complex>*>(*S);
    delete H;
    H = NULL;
  }
  void SP_z_struct_destroy(CSPStructMat* S) {
    auto H = reinterpret_cast<StructuredMatrix<double _Complex>*>(*S);
    delete H;
    H = NULL;
  }


  int SP_s_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             float* A, int ldA, CSPOptions* opts) {
    try {
      *S = construct_from_dense<float>
        (rows, cols, A, ldA, get_options<float>(opts)).release();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_d_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             double* A, int ldA, CSPOptions* opts) {
    try {
      *S = construct_from_dense<double>
        (rows, cols, A, ldA, get_options<double>(opts)).release();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_c_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             float _Complex* A, int ldA, CSPOptions* opts) {
    try {
      *S = construct_from_dense<std::complex<float>>
        (rows, cols, reinterpret_cast<std::complex<float>*>(A), ldA,
         get_options<std::complex<float>>(opts)).release();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_z_struct_from_dense(CSPStructMat* S, int rows, int cols,
                             double _Complex* A, int ldA, CSPOptions* opts) {
    try {
      *S = construct_from_dense<std::complex<double>>
        (rows, cols, reinterpret_cast<std::complex<double>*>(A), ldA,
         get_options<std::complex<double>>(opts)).release();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }



  int SP_s_struct_mult(CSPStructMat S, char trans, int m, float* B, int ldB,
                       float* C, int ldC) {
    try {
      reinterpret_cast<StructuredMatrix<float>*>(S)->
        mult(c2T(trans), m, B, ldB, C, ldC);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_d_struct_mult(CSPStructMat S, char trans, int m, double* B, int ldB,
                       double* C, int ldC) {
    try {
      reinterpret_cast<StructuredMatrix<double>*>(S)->
        mult(c2T(trans), m, B, ldB, C, ldC);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_c_struct_mult(CSPStructMat S, char trans, int m, float _Complex* B, int ldB,
                       float _Complex* C, int ldC) {
    try {
      reinterpret_cast<StructuredMatrix<std::complex<float>>*>(S)->
        mult(c2T(trans), m, reinterpret_cast<std::complex<float>*>(B), ldB,
             reinterpret_cast<std::complex<float>*>(C), ldC);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_z_struct_mult(CSPStructMat S, char trans, int m, double _Complex* B, int ldB,
                       double _Complex* C, int ldC) {
    try {
      reinterpret_cast<StructuredMatrix<std::complex<double>>*>(S)->
        mult(c2T(trans), m, reinterpret_cast<std::complex<double>*>(B), ldB,
             reinterpret_cast<std::complex<double>*>(C), ldC);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }

  int SP_s_struct_factor(CSPStructMat S) {
    try {
      reinterpret_cast<StructuredMatrix<float>*>(S)->factor();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_d_struct_factor(CSPStructMat S) {
    try {
      reinterpret_cast<StructuredMatrix<double>*>(S)->factor();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_c_struct_factor(CSPStructMat S) {
    try {
      reinterpret_cast<StructuredMatrix<std::complex<float>>*>(S)->factor();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_z_struct_factor(CSPStructMat S) {
    try {
      reinterpret_cast<StructuredMatrix<std::complex<double>>*>(S)->factor();
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }

  int SP_s_struct_solve(CSPStructMat S, int nrhs, float* B, int ldB) {
    try {
      reinterpret_cast<StructuredMatrix<float>*>(S)->
        solve(nrhs, reinterpret_cast<float*>(B), ldB);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_d_struct_solve(CSPStructMat S, int nrhs, double* B, int ldB) {
    try {
      reinterpret_cast<StructuredMatrix<double>*>(S)->
        solve(nrhs, reinterpret_cast<double*>(B), ldB);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_c_struct_solve(CSPStructMat S, int nrhs, float _Complex* B, int ldB) {
    try {
      reinterpret_cast<StructuredMatrix<std::complex<float>>*>(S)->
        solve(nrhs, reinterpret_cast<std::complex<float>*>(B), ldB);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  int SP_z_struct_solve(CSPStructMat S, int nrhs, double _Complex* B, int ldB) {
    try {
      reinterpret_cast<StructuredMatrix<std::complex<double>>*>(S)->
        solve(nrhs, reinterpret_cast<std::complex<double>*>(B), ldB);
    } catch (std::exception& e) {
      std::cerr << "Operation failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }

}

