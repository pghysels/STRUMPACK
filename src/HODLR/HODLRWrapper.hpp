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
#include "zC_BPACK_wrapper.h"

namespace strumpack {
  namespace HODLR {

    template<typename scalar_t> void HODLR_createptree
    (int& P, int* groups, MPI_Fint comm, F2Cptr& ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_createptree<double>
    (int& P, int* groups, MPI_Fint comm, F2Cptr& ptree);
    template<> void HODLR_createptree<std::complex<double>>
    (int& P, int* groups, MPI_Fint comm, F2Cptr& ptree);

    template<typename scalar_t> void HODLR_createoptions(F2Cptr& options) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_createoptions<double>(F2Cptr& options);
    template<> void HODLR_createoptions<std::complex<double>>(F2Cptr& options);

    template<typename scalar_t> void HODLR_copyoptions(F2Cptr& in, F2Cptr& out) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_copyoptions<double>(F2Cptr& in, F2Cptr& out);
    template<> void HODLR_copyoptions<std::complex<double>>(F2Cptr& in, F2Cptr& out);

    template<typename scalar_t> void HODLR_printoptions(F2Cptr& options, F2Cptr& ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_printoptions<double>(F2Cptr& options, F2Cptr& ptree);
    template<> void HODLR_printoptions<std::complex<double>>(F2Cptr& options, F2Cptr& ptree);

    template<typename scalar_t> void HODLR_printstats(F2Cptr& stats, F2Cptr& ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_printstats<double>(F2Cptr& stats, F2Cptr& ptree);
    template<> void HODLR_printstats<std::complex<double>>(F2Cptr& stats, F2Cptr& ptree);

    template<typename scalar_t> void HODLR_createstats(F2Cptr& stats) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_createstats<double>(F2Cptr& stats);
    template<> void HODLR_createstats<std::complex<double>>(F2Cptr& stats);

    template<typename scalar_t> void HODLR_set_D_option
    (F2Cptr options, const std::string& opt, double v) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_set_D_option<double>
    (F2Cptr options, const std::string& opt, double v);
    template<> void HODLR_set_D_option<std::complex<double>>
    (F2Cptr options, const std::string& opt, double v);

    template<typename scalar_t> void HODLR_set_I_option
    (F2Cptr options, const std::string& opt, int v) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_set_I_option<double>
    (F2Cptr options, const std::string& opt, int v);
    template<> void HODLR_set_I_option<std::complex<double>>
    (F2Cptr options, const std::string& opt, int v);

    /**
     * Possible values:
     *
     *  Time_Fill, Time_Factor, Time_Solve, Time_Sblock, Time_Inv,
     *  Time_SMW, Time_RedistB, Time_RedistV, Time_C_Mult,
     *  Time_Direct_LU, Time_Add_Multiply, Time_Multiply, Time_XLUM,
     *  Time_Split, Time_Comm, Time_Idle
     *
     *  Flop_Fill, Flop_Factor, Flop_Solve, Flop_C_Mult
     *
     *  Mem_Factor, Mem_Fill, Mem_Sblock, Mem_SMW, Mem_Direct_inv,
     *  Mem_Direct_for, Mem_int_vec, Mem_Comp_for
     *
     *  Rank_max
     */
    template<typename scalar_t> double BPACK_get_stat
    (F2Cptr stats, const std::string& name) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
      return 0.;
    }
    template<> double BPACK_get_stat<double>
    (F2Cptr stats, const std::string& name);
    template<> double BPACK_get_stat<std::complex<double>>
    (F2Cptr stats, const std::string& name);


    template<typename scalar_t> void HODLR_construct_init
    (int N, int d, scalar_t* data, int* nns, int lvls, int* tree, int* perm,
     int& lrow, F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr),
     C2Fptr fdata) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_construct_init<double>
    (int N, int d, double* data, int* nns, int lvls, int* tree, int* perm,
     int& lrow, F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr),
     C2Fptr fdata);
    template<> void HODLR_construct_init<std::complex<double>>
    (int N, int d, std::complex<double>* data, int* nns, int lvls, int* tree,
     int* perm, int& lrow, F2Cptr& ho_bf, F2Cptr& options,
     F2Cptr& stats, F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr),
     C2Fptr fdata);

    template<typename scalar_t> void HODLR_construct_element_compute
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, scalar_t*, C2Fptr),
     void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, scalar_t* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr K) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_construct_element_compute<double>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, double* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr K);
    template<> void HODLR_construct_element_compute<std::complex<double>>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, std::complex<double>*, C2Fptr),
     void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, std::complex<double>* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr K);

    template<typename scalar_t> void HODLR_construct_matvec_compute
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*matvec)
     (char const*, int*, int*, int*, const scalar_t*, scalar_t*, C2Fptr),
     C2Fptr fdata) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_construct_matvec_compute<double>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*matvec)
     (char const*, int*, int*, int*, const double*, double*, C2Fptr),
     C2Fptr fdata);
    template<> void HODLR_construct_matvec_compute<std::complex<double>>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*matvec)
     (char const*, int*, int*, int*, const std::complex<double>*,
      std::complex<double>*, C2Fptr), C2Fptr fdata);

    template<typename scalar_t> void LRBF_construct_init
    (int M, int N, int& lrows, int& lcols, int* nsr, int* nnsc,
     F2Cptr rmsh, F2Cptr cmsh, F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr), C2Fptr fdata) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void LRBF_construct_init<double>
    (int M, int N, int& lrows, int& lcols, int* nsr, int* nnsc,
     F2Cptr rmsh, F2Cptr cmsh, F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr), C2Fptr fdata);
    template<> void LRBF_construct_init<std::complex<double>>
    (int M, int N, int& lrows, int& lcols, int* nsr, int* nnsc,
     F2Cptr rmsh, F2Cptr cmsh, F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr), C2Fptr fdata);

    template<typename scalar_t> void LRBF_construct_matvec_compute
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (const char*, int*, int*, int*, const scalar_t*,
      scalar_t*, C2Fptr, scalar_t*, scalar_t*), C2Fptr fdata) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void LRBF_construct_matvec_compute<double>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (const char*, int*, int*, int*, const double*,
      double*, C2Fptr, double*, double*), C2Fptr fdata);
    template<> void LRBF_construct_matvec_compute<std::complex<double>>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (char const*, int*, int*, int*, const std::complex<double>*,
      std::complex<double>*, C2Fptr, std::complex<double>*,
      std::complex<double>*), C2Fptr fdata);

    template<typename scalar_t> void LRBF_construct_element_compute
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*element)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, scalar_t* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void LRBF_construct_element_compute<double>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*element)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, double* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata);
    template<> void LRBF_construct_element_compute<std::complex<double>>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*element)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, std::complex<double>* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata);

    template<typename scalar_t> void HODLR_extract_elements
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, scalar_t* alldat_loc,
     int* rowidx, int* colidx, int* pgidx, int Npmap, int* pmaps) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_extract_elements<double>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, double* alldat_loc,
     int* rowidx, int* colidx, int* pgidx, int Npmap, int* pmaps);
    template<> void HODLR_extract_elements<std::complex<double>>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows,int* allcols, std::complex<double>* alldat_loc,
     int* rowidx, int* colidx, int* pgidx, int Npmap, int* pmaps);

    template<typename scalar_t> void LRBF_extract_elements
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, scalar_t* alldat_loc,
     int* rowidx, int* colidx, int* pgidx, int Npmap, int* pmaps) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void LRBF_extract_elements<double>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, double* alldat_loc,
     int* rowidx, int* colidx, int* pgidx, int Npmap, int* pmaps);
    template<> void LRBF_extract_elements<std::complex<double>>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows,int* allcols, std::complex<double>* alldat_loc,
     int* rowidx, int* colidx, int* pgidx, int Npmap, int* pmaps);

    template<typename scalar_t> void HODLR_deletestats(F2Cptr&);
    template<> void HODLR_deletestats<float>(F2Cptr& stats);
    template<> void HODLR_deletestats<double>(F2Cptr& stats);
    template<> void HODLR_deletestats<std::complex<float>>(F2Cptr& stats);
    template<> void HODLR_deletestats<std::complex<double>>(F2Cptr& stats);

    template<typename scalar_t> void HODLR_deleteproctree(F2Cptr&);
    template<> void HODLR_deleteproctree<float>(F2Cptr& ptree);
    template<> void HODLR_deleteproctree<double>(F2Cptr& ptree);
    template<> void HODLR_deleteproctree<std::complex<float>>(F2Cptr& ptree);
    template<> void HODLR_deleteproctree<std::complex<double>>(F2Cptr& ptree);

    template<typename scalar_t> void HODLR_deletemesh(F2Cptr&);
    template<> void HODLR_deletemesh<float>(F2Cptr& mesh);
    template<> void HODLR_deletemesh<double>(F2Cptr& mesh);
    template<> void HODLR_deletemesh<std::complex<float>>(F2Cptr& mesh);
    template<> void HODLR_deletemesh<std::complex<double>>(F2Cptr& mesh);

    template<typename scalar_t> void HODLR_deletekernelquant(F2Cptr&);
    template<> void HODLR_deletekernelquant<float>(F2Cptr& kerquant);
    template<> void HODLR_deletekernelquant<double>(F2Cptr& kerquant);
    template<> void HODLR_deletekernelquant<std::complex<float>>(F2Cptr& kerquant);
    template<> void HODLR_deletekernelquant<std::complex<double>>(F2Cptr& kerquant);

    template<typename scalar_t> void HODLR_delete(F2Cptr&);
    template<> void HODLR_delete<float>(F2Cptr& ho_bf);
    template<> void HODLR_delete<double>(F2Cptr& ho_bf);
    template<> void HODLR_delete<std::complex<float>>(F2Cptr& ho_bf);
    template<> void HODLR_delete<std::complex<double>>(F2Cptr& ho_bf);

    template<typename scalar_t> void LRBF_deletebf(F2Cptr&);
    template<> void LRBF_deletebf<float>(F2Cptr& lr_bf);
    template<> void LRBF_deletebf<double>(F2Cptr& lr_bf);
    template<> void LRBF_deletebf<std::complex<float>>(F2Cptr& lr_bf);
    template<> void LRBF_deletebf<std::complex<double>>(F2Cptr& lr_bf);

    template<typename scalar_t> void HODLR_deleteoptions(F2Cptr&);
    template<> void HODLR_deleteoptions<float>(F2Cptr& option);
    template<> void HODLR_deleteoptions<double>(F2Cptr& option);
    template<> void HODLR_deleteoptions<std::complex<float>>(F2Cptr& option);
    template<> void HODLR_deleteoptions<std::complex<double>>(F2Cptr& option);

    template<typename scalar_t> void HODLR_mult
    (char op, const scalar_t* X, scalar_t* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_mult<double>
    (char op, const double* X, double* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree);
    template<> void HODLR_mult<std::complex<double>>
    (char op, const std::complex<double>* X, std::complex<double>* Y,
     int Xlrows, int Ylrows, int cols, F2Cptr ho_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree);

    template<typename scalar_t> void LRBF_mult
    (char op, const scalar_t* X, scalar_t* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr lr_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void LRBF_mult<double>
    (char op, const double* X, double* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr lr_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree);
    template<> void LRBF_mult<std::complex<double>>
    (char op, const std::complex<double>* X, std::complex<double>* Y,
     int Xlrows, int Ylrows, int cols, F2Cptr lr_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree);

    template<typename scalar_t> void HODLR_factor
    (F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree, F2Cptr msh) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_factor<double>
    (F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree, F2Cptr msh);
    template<> void HODLR_factor<std::complex<double>>
    (F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree, F2Cptr msh);

    template<typename scalar_t> void HODLR_solve
    (scalar_t* X, const scalar_t* B, int lrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_solve<double>
    (double* X, const double* B, int lrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree);
    template<> void HODLR_solve<std::complex<double>>
    (std::complex<double>* X, const std::complex<double>* B,
     int lrows, int rhs, F2Cptr ho_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree);

    template<typename scalar_t> void HODLR_inv_mult
    (char op, const scalar_t* B, scalar_t* X, int Xlrows, int Blrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      std::cout << "ERROR: HODLR code does not support this precision." << std::endl;
    }
    template<> void HODLR_inv_mult<double>
    (char op, const double* B, double* X, int Xlrows, int Blrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree);
    template<> void HODLR_inv_mult<std::complex<double>>
    (char op, const std::complex<double>* B, std::complex<double>* X,
     int Xlrows, int Blrows, int rhs, F2Cptr ho_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree);

    int LRBF_treeindex_merged2child(int idx_merge);

  } // end namespace HODLR
} // end namespace strumpack

#endif // STRUMPACK_HODLR_WRAPPER_HPP
