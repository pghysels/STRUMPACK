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
/*! \file HODLRWrapper.hpp
 * \brief Interface to Yang Liu's Fortran HODLR and butterfly code.
 */
#ifndef STRUMPACK_HODLR_WRAPPER_HPP
#define STRUMPACK_HODLR_WRAPPER_HPP

#include <cassert>

#include "dense/DenseMatrix.hpp"
#include "dC_BPACK_wrapper.h"
#undef HODLR_WRAP
#include "zC_BPACK_wrapper.h"

namespace strumpack {

  namespace HODLR {

    template<typename scalar_t> void HODLR_createptree
    (int& P, int* groups, MPI_Fint comm, F2Cptr& ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void HODLR_createptree<double>
    (int& P, int* groups, MPI_Fint comm, F2Cptr& ptree) {
      d_c_bpack_createptree(&P, groups, &comm, &ptree);
    }
    template<> inline void HODLR_createptree<std::complex<double>>
    (int& P, int* groups, MPI_Fint comm, F2Cptr& ptree) {
      z_c_bpack_createptree(&P, groups, &comm, &ptree);
    }

    template<typename scalar_t> void HODLR_createoptions(F2Cptr& options) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void HODLR_createoptions<double>(F2Cptr& options) {
      d_c_bpack_createoption(&options);
    }
    template<> inline void HODLR_createoptions<std::complex<double>>(F2Cptr& options) {
      z_c_bpack_createoption(&options);
    }

    template<typename scalar_t> void HODLR_copyoptions(F2Cptr& in, F2Cptr& out) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void HODLR_copyoptions<double>(F2Cptr& in, F2Cptr& out) {
      d_c_bpack_copyoption(&in, &out);
    }
    template<> inline void HODLR_copyoptions<std::complex<double>>(F2Cptr& in, F2Cptr& out) {
      z_c_bpack_copyoption(&in, &out);
    }

    template<typename scalar_t> void HODLR_createstats(F2Cptr& stats) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void HODLR_createstats<double>(F2Cptr& stats) {
      d_c_bpack_createstats(&stats);
    }
    template<> inline void HODLR_createstats<std::complex<double>>(F2Cptr& stats) {
      z_c_bpack_createstats(&stats);
    }

    template<typename scalar_t> void HODLR_set_D_option
    (F2Cptr options, const std::string& opt, double v) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void HODLR_set_D_option<double>
    (F2Cptr options, const std::string& opt, double v) {
      d_c_bpack_set_D_option(&options, opt.c_str(), v);
    }
    template<> inline void HODLR_set_D_option<std::complex<double>>
    (F2Cptr options, const std::string& opt, double v) {
      z_c_bpack_set_D_option(&options, opt.c_str(), v);
    }

    template<typename scalar_t> void HODLR_set_I_option
    (F2Cptr options, const std::string& opt, int v) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void HODLR_set_I_option<double>
    (F2Cptr options, const std::string& opt, int v) {
      d_c_bpack_set_I_option(&options, opt.c_str(), v);
    }
    template<> inline void HODLR_set_I_option<std::complex<double>>
    (F2Cptr options, const std::string& opt, int v) {
      z_c_bpack_set_I_option(&options, opt.c_str(), v);
    }

    template<typename scalar_t> void HODLR_construct_element
    (int n, int d, scalar_t* data, int lvls, int* leafs, int* perm,
     int& lrows, F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, scalar_t*, C2Fptr),
     C2Fptr K, MPI_Fint comm) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void HODLR_construct_element<double>
    (int n, int d, double* data, int lvls, int* leafs, int* perm,
     int& lrows, F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, double*, C2Fptr),
     C2Fptr K, MPI_Fint comm) {
      d_c_bpack_construct_element
        (&n, &d, data, &lvls, leafs, perm, &lrows, &ho_bf, &options,
         &stats, &msh, &kerquant, &ptree,
         C_FuncZmn, K, &comm);
    }
    template<> inline void HODLR_construct_element<std::complex<double>>
    (int n, int d, std::complex<double>* data, int lvls, int* leafs,
     int* perm, int& lrows, F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, std::complex<double>*, C2Fptr),
     C2Fptr K, MPI_Fint comm) {
      //TODO, data should be double??
      // z_c_bpack_construct_element
      //   (&n, &d, data, &lvls, leafs, perm, &lrows, &ho_bf, &options,
      //    &stats, &msh, &kerquant, &ptree,
      //    C_FuncZmn, K, &comm);
    }


    template<typename scalar_t> void HODLR_construct_matvec_init
    (int N, int lvls, int* tree, int* perm, int& lrow,
     F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void HODLR_construct_matvec_init<double>
    (int N, int lvls, int* tree, int* perm, int& lrow,
     F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree) {
      d_c_bpack_construct_matvec_init
        (&N, &lvls, tree, perm, &lrow, &ho_bf, &options,
         &stats, &msh, &kerquant, &ptree);
    }
    template<> inline void HODLR_construct_matvec_init<std::complex<double>>
    (int N, int lvls, int* tree, int* perm, int& lrow,
     F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree) {
      // z_c_bpack_construct_matvec_init
      //   (&N, &lvls, tree, perm, &lrow, &ho_bf, &options,
      //    &stats, &msh, &kerquant, &ptree);
    }

    template<typename scalar_t> void HODLR_construct_matvec_compute
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (char const*, int*, int*, int*, const scalar_t*, scalar_t*, C2Fptr),
     C2Fptr& fdata) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void HODLR_construct_matvec_compute<double>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (char const*, int*, int*, int*, const double*, double*, C2Fptr),
     C2Fptr& fdata) {
      d_c_bpack_construct_matvec_compute
        (&ho_bf, &options, &stats, &msh, &kerquant, &ptree, matvec, fdata);
    }
    template<> inline void HODLR_construct_matvec_compute<std::complex<double>>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (char const*, int*, int*, int*, const std::complex<double>*,
      std::complex<double>*, C2Fptr), C2Fptr& fdata) {
      // z_c_bpack_construct_matvec_compute
      //   (&ho_bf, &options, &stats, &msh, &kerquant, &ptree, matvec, fdata);
    }


    template<typename scalar_t> void LRBF_construct_matvec_init
    (int M, int N, int& lrows, int& lcols, F2Cptr rmsh, F2Cptr cmsh,
     F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void LRBF_construct_matvec_init<double>
    (int M, int N, int& lrows, int& lcols, F2Cptr rmsh, F2Cptr cmsh,
     F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree) {
      d_c_bf_construct_matvec_init
        (&M, &N, &lrows, &lcols, &rmsh, &cmsh, &lr_bf, &options,
         &stats, &msh, &kerquant, &ptree);
    }
    template<> inline void LRBF_construct_matvec_init<std::complex<double>>
    (int M, int N, int& lrows, int& lcols, F2Cptr rmsh, F2Cptr cmsh,
     F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree) {
      // z_c_bf_construct_matvec_init
      //   (&M, &N, &lrows, &lcols_, &rmsh, &cmsh, &lr_bf, &options,
      //    &stats, &msh, &kerquant, &ptree);
    }

    template<typename scalar_t> void LRBF_construct_matvec_compute
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (const char*, int*, int*, int*, const scalar_t*,
      scalar_t*, C2Fptr, scalar_t*, scalar_t*), C2Fptr& fdata) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void LRBF_construct_matvec_compute<double>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (const char*, int*, int*, int*, const double*,
      double*, C2Fptr, double*, double*), C2Fptr& fdata) {
      d_c_bf_construct_matvec_compute
        (&lr_bf, &options, &stats, &msh, &kerquant, &ptree,
         matvec, fdata);
    }
    template<> inline void LRBF_construct_matvec_compute<std::complex<double>>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (char const*, int*, int*, int*, const std::complex<double>*,
      std::complex<double>*, C2Fptr, std::complex<double>*,
      std::complex<double>*), C2Fptr& fdata) {
      // z_c_bf_construct_matvec_compute
      //   (&lr_bf, &options, &stats, &msh, &kerquant, &ptree,
      //    matvec, fdata);
    }

    template<typename scalar_t> void HODLR_deletestats(F2Cptr&);
    template<> inline void HODLR_deletestats<float>(F2Cptr& stats) { std::cout << "TODO: HODLR_deletestats" << std::endl; }
    template<> inline void HODLR_deletestats<double>(F2Cptr& stats) { d_c_bpack_deletestats(&stats); }
    template<> inline void HODLR_deletestats<std::complex<float>>(F2Cptr& stats) { std::cout << "TODO: HODLR_deletestats" << std::endl; }
    template<> inline void HODLR_deletestats<std::complex<double>>(F2Cptr& stats) { z_c_bpack_deletestats(&stats); }

    template<typename scalar_t> void HODLR_deleteproctree(F2Cptr&);
    template<> inline void HODLR_deleteproctree<float>(F2Cptr& ptree) { std::cout << "TODO: HODLR_deleteproctree" << std::endl; }
    template<> inline void HODLR_deleteproctree<double>(F2Cptr& ptree) { d_c_bpack_deleteproctree(&ptree); }
    template<> inline void HODLR_deleteproctree<std::complex<float>>(F2Cptr& ptree) { std::cout << "TODO: HODLR_deleteproctree" << std::endl; }
    template<> inline void HODLR_deleteproctree<std::complex<double>>(F2Cptr& ptree) { z_c_bpack_deleteproctree(&ptree); }

    template<typename scalar_t> void HODLR_deletemesh(F2Cptr&);
    template<> inline void HODLR_deletemesh<float>(F2Cptr& mesh) { std::cout << "TODO: HODLR_deletemesh" << std::endl; }
    template<> inline void HODLR_deletemesh<double>(F2Cptr& mesh) { d_c_bpack_deletemesh(&mesh); }
    template<> inline void HODLR_deletemesh<std::complex<float>>(F2Cptr& mesh) { std::cout << "TODO: HODLR_deletemesh" << std::endl; }
    template<> inline void HODLR_deletemesh<std::complex<double>>(F2Cptr& mesh) { z_c_bpack_deletemesh(&mesh); }

    template<typename scalar_t> void HODLR_deletekernelquant(F2Cptr&);
    template<> inline void HODLR_deletekernelquant<float>(F2Cptr& kerquant) { std::cout << "TODO HODLR_deletekernelquant" << std::endl; }
    template<> inline void HODLR_deletekernelquant<double>(F2Cptr& kerquant) { d_c_bpack_deletekernelquant(&kerquant); }
    template<> inline void HODLR_deletekernelquant<std::complex<float>>(F2Cptr& kerquant) { std::cout << "TODO HODLR_deletekernelquant" << std::endl; }
    template<> inline void HODLR_deletekernelquant<std::complex<double>>(F2Cptr& kerquant) { z_c_bpack_deletekernelquant(&kerquant); }

    template<typename scalar_t> void HODLR_delete(F2Cptr&);
    template<> inline void HODLR_delete<float>(F2Cptr& ho_bf) { std::cout << "TODO HODLR_delete" << std::endl; }
    template<> inline void HODLR_delete<double>(F2Cptr& ho_bf) { d_c_bpack_delete(&ho_bf); }
    template<> inline void HODLR_delete<std::complex<float>>(F2Cptr& ho_bf) { std::cout << "TODO HODLR_delete" << std::endl; }
    template<> inline void HODLR_delete<std::complex<double>>(F2Cptr& ho_bf) { z_c_bpack_delete(&ho_bf); }

    template<typename scalar_t> void LRBF_deletebf(F2Cptr&);
    template<> inline void LRBF_deletebf<float>(F2Cptr& lr_bf) { std::cout << "TODO LRBF_deletebf" << std::endl; }
    template<> inline void LRBF_deletebf<double>(F2Cptr& lr_bf) { d_c_bf_deletebf(&lr_bf); }
    template<> inline void LRBF_deletebf<std::complex<float>>(F2Cptr& lr_bf) { std::cout << "TODO LRBF_deletebf" << std::endl; }
    template<> inline void LRBF_deletebf<std::complex<double>>(F2Cptr& lr_bf) { z_c_bf_deletebf(&lr_bf); }

    template<typename scalar_t> void HODLR_deleteoptions(F2Cptr&);
    template<> inline void HODLR_deleteoptions<float>(F2Cptr& option) { std::cout << "TODO HODLR_deleteoptions" << std::endl; }
    template<> inline void HODLR_deleteoptions<double>(F2Cptr& option) { d_c_bpack_deleteoption(&option); }
    template<> inline void HODLR_deleteoptions<std::complex<float>>(F2Cptr& option) { std::cout << "TODO HODLR_deleteoptions" << std::endl; }
    template<> inline void HODLR_deleteoptions<std::complex<double>>(F2Cptr& option) { z_c_bpack_deleteoption(&option); }

    template<typename scalar_t> void HODLR_mult
    (char op, const scalar_t* X, scalar_t* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void HODLR_mult<double>
    (char op, const double* X, double* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      d_c_bpack_mult(&op, X, Y, &Xlrows, &Ylrows,
                     &cols, &ho_bf, &options, &stats, &ptree);
    }
    template<> inline void HODLR_mult<std::complex<double>>
    (char op, const std::complex<double>* X, std::complex<double>* Y,
     int Xlrows, int Ylrows, int cols, F2Cptr ho_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree) {
      // z_c_bpack_mult
      //   (&op, const_cast<std::complex<double>*>(X), Y, &Xlrows, &Ylrows,
      //    &cols, &ho_bf, &options, &stats, &ptree);
    }

    template<typename scalar_t> void LRBF_mult
    (char op, const scalar_t* X, scalar_t* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr lr_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void LRBF_mult<double>
    (char op, const double* X, double* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr lr_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      d_c_bf_mult(&op, X, Y, &Xlrows, &Ylrows, &cols,
                  &lr_bf, &options, &stats, &ptree);
    }
    template<> inline void LRBF_mult<std::complex<double>>
    (char op, const std::complex<double>* X, std::complex<double>* Y,
     int Xlrows, int Ylrows, int cols, F2Cptr lr_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree) {
      // z_c_bf_mult(&op, X, Y, &Xlrows, &Ylrows, &cols,
      //             &lr_bf, &options, &stats, &ptree);
    }

    template<typename scalar_t> void HODLR_factor
    (F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree, F2Cptr msh) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void HODLR_factor<double>
    (F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree, F2Cptr msh) {
      d_c_bpack_factor(&ho_bf, &options, &stats, &ptree, &msh);
    }
    template<> inline void HODLR_factor<std::complex<double>>
    (F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree, F2Cptr msh) {
      z_c_bpack_factor(&ho_bf, &options, &stats, &ptree, &msh);
    }

    template<typename scalar_t> void HODLR_solve
    (scalar_t* X, const scalar_t* B, int lrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void HODLR_solve<double>
    (double* X, const double* B, int lrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      d_c_bpack_solve(X, const_cast<double*>(B), &lrows, &rhs,
                      &ho_bf, &options, &stats, &ptree);
    }
    template<> inline void HODLR_solve<std::complex<double>>
    (std::complex<double>* X, const std::complex<double>* B,
     int lrows, int rhs, F2Cptr ho_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree) {
      // z_c_bpack_solve(X, const_cast<std::complex<double>*>(B), &lrows, &rhs,
      //                 &ho_bf, &options, &stats, &ptree);
    }

    template<typename scalar_t> void HODLR_inv_mult
    (char op, const scalar_t* B, scalar_t* X, int Xlrows, int Blrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> inline void HODLR_inv_mult<double>
    (char op, const double* B, double* X, int Xlrows, int Blrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      d_c_bpack_inv_mult
        (&op, B, X, &Xlrows, &Blrows, &rhs, &ho_bf, &options, &stats, &ptree);
    }
    template<> inline void HODLR_inv_mult<std::complex<double>>
    (char op, const std::complex<double>* B, std::complex<double>* X,
     int Xlrows, int Blrows, int rhs, F2Cptr ho_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree) {
      // z_c_bpack_inv_mult
      //   (&op, B, X, &Xlrows, &Blrows,
      //    &rhs, &ho_bf, &options, &stats, &ptree);
    }

  } // end namespace HODLR
} // end namespace strumpack

#endif // STRUMPACK_HODLR_WRAPPER_HPP
