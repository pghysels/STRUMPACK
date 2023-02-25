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
#include <complex>

#include "dense/DenseMatrix.hpp"
#include "misc/MPIWrapper.hpp"
typedef void* F2Cptr;
typedef void* C2Fptr;

namespace strumpack {
  namespace HODLR {

    template<typename scalar_t> void HODLR_createptree
    (int& P, int* groups, MPI_Fint comm, F2Cptr& ptree);

    template<typename scalar_t> void HODLR_createoptions(F2Cptr& options);

    template<typename scalar_t> void HODLR_copyoptions(F2Cptr& in, F2Cptr& out);

    template<typename scalar_t> void HODLR_printoptions(F2Cptr& options, F2Cptr& ptree);

    template<typename scalar_t> void HODLR_printstats(F2Cptr& stats, F2Cptr& ptree);

    template<typename scalar_t> void HODLR_createstats(F2Cptr& stats);

    template<typename scalar_t> void HODLR_set_D_option
    (F2Cptr options, const std::string& opt, double v);

    template<typename scalar_t> void HODLR_set_I_option
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
    (F2Cptr stats, const std::string& name);

    template<typename scalar_t,
             typename real_t = typename RealType<scalar_t>::value_type>
    void HODLR_construct_init
    (int N, int d, real_t* data, int* nns, int lvls, int* tree, int* perm,
     int& lrow, F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr), C2Fptr fdata);

    template<typename scalar_t,
             typename real_t = typename RealType<scalar_t>::value_type>
    void HODLR_construct_init_Gram
    (int N, int d, real_t* data, int* nns, int lvls, int* tree, int* perm,
     int& lrow, F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, scalar_t*, C2Fptr),
     void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, scalar_t* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata);

    template<typename scalar_t> void HODLR_construct_element_compute
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, scalar_t*, C2Fptr),
     void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, scalar_t* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr K);

    template<typename scalar_t> void HODLR_construct_matvec_compute
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*matvec)
     (char const*, int*, int*, int*, const scalar_t*, scalar_t*, C2Fptr),
     C2Fptr fdata);

    template<typename scalar_t>
    void LRBF_construct_init
    (int M, int N, int& lrows, int& lcols, int* nsr, int* nnsc,
     F2Cptr rmsh, F2Cptr cmsh, F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr), C2Fptr fdata);

    template<typename scalar_t> void LRBF_construct_matvec_compute
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (const char*, int*, int*, int*, const scalar_t*,
      scalar_t*, C2Fptr, scalar_t*, scalar_t*), C2Fptr fdata);

    template<typename scalar_t> void LRBF_construct_element_compute
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*element)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, scalar_t* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata);

    template<typename scalar_t> void HODLR_extract_elements
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, scalar_t* alldat_loc,
     int* rowidx, int* colidx, int* pgidx, int Npmap, int* pmaps);

    template<typename scalar_t> void LRBF_extract_elements
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, scalar_t* alldat_loc,
     int* rowidx, int* colidx, int* pgidx, int Npmap, int* pmaps);

    template<typename scalar_t> void HODLR_deletestats(F2Cptr&);

    template<typename scalar_t> void HODLR_deleteproctree(F2Cptr&);

    template<typename scalar_t> void HODLR_deletemesh(F2Cptr&);

    template<typename scalar_t> void HODLR_deletekernelquant(F2Cptr&);

    template<typename scalar_t> void HODLR_delete(F2Cptr&);

    template<typename scalar_t> void LRBF_deletebf(F2Cptr&);

    template<typename scalar_t> void HODLR_deleteoptions(F2Cptr&);

    template<typename scalar_t> void HODLR_mult
    (char op, const scalar_t* X, scalar_t* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree);

    template<typename scalar_t> void LRBF_mult
    (char op, const scalar_t* X, scalar_t* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr lr_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree);

    template<typename scalar_t> void HODLR_factor
    (F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree, F2Cptr msh);

    template<typename scalar_t> void HODLR_solve
    (scalar_t* X, const scalar_t* B, int lrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree);

    template<typename scalar_t> void HODLR_inv_mult
    (char op, const scalar_t* B, scalar_t* X, int Xlrows, int Blrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree);

    int LRBF_treeindex_merged2child(int idx_merge);

  } // end namespace HODLR
} // end namespace strumpack

#endif // STRUMPACK_HODLR_WRAPPER_HPP
