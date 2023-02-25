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

#include <cassert>
#include <complex>

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#include "misc/TaskTimer.hpp"
#include "HODLRWrapper.hpp"
#include "sC_BPACK_wrapper.h"
#include "dC_BPACK_wrapper.h"
#include "cC_BPACK_wrapper.h"
#include "zC_BPACK_wrapper.h"

namespace strumpack {
  namespace HODLR {

    template<> void HODLR_createptree<float>
    (int& P, int* groups, MPI_Fint comm, F2Cptr& ptree) {
      TIMER_TIME(TaskType::CONSTRUCT_PTREE, 0, t_construct_h);
      s_c_bpack_createptree(&P, groups, &comm, &ptree);
    }
    template<> void HODLR_createptree<double>
    (int& P, int* groups, MPI_Fint comm, F2Cptr& ptree) {
      TIMER_TIME(TaskType::CONSTRUCT_PTREE, 0, t_construct_h);
      d_c_bpack_createptree(&P, groups, &comm, &ptree);
    }
    template<> void HODLR_createptree<std::complex<float>>
    (int& P, int* groups, MPI_Fint comm, F2Cptr& ptree) {
      TIMER_TIME(TaskType::CONSTRUCT_PTREE, 0, t_construct_h);
      c_c_bpack_createptree(&P, groups, &comm, &ptree);
    }
    template<> void HODLR_createptree<std::complex<double>>
    (int& P, int* groups, MPI_Fint comm, F2Cptr& ptree) {
      TIMER_TIME(TaskType::CONSTRUCT_PTREE, 0, t_construct_h);
      z_c_bpack_createptree(&P, groups, &comm, &ptree);
    }

    template<> void HODLR_createoptions<float>(F2Cptr& options) {
      s_c_bpack_createoption(&options);
    }
    template<> void HODLR_createoptions<double>(F2Cptr& options) {
      d_c_bpack_createoption(&options);
    }
    template<> void HODLR_createoptions<std::complex<float>>(F2Cptr& options) {
      c_c_bpack_createoption(&options);
    }
    template<> void HODLR_createoptions<std::complex<double>>(F2Cptr& options) {
      z_c_bpack_createoption(&options);
    }

    template<> void HODLR_copyoptions<float>(F2Cptr& in, F2Cptr& out) {
      s_c_bpack_copyoption(&in, &out);
    }
    template<> void HODLR_copyoptions<double>(F2Cptr& in, F2Cptr& out) {
      d_c_bpack_copyoption(&in, &out);
    }
    template<> void HODLR_copyoptions<std::complex<float>>(F2Cptr& in, F2Cptr& out) {
      c_c_bpack_copyoption(&in, &out);
    }
    template<> void HODLR_copyoptions<std::complex<double>>(F2Cptr& in, F2Cptr& out) {
      z_c_bpack_copyoption(&in, &out);
    }

    template<> void HODLR_printoptions<float>(F2Cptr& options, F2Cptr& ptree) {
      s_c_bpack_printoption(&options, &ptree);
    }
    template<> void HODLR_printoptions<double>(F2Cptr& options, F2Cptr& ptree) {
      d_c_bpack_printoption(&options, &ptree);
    }
    template<> void HODLR_printoptions<std::complex<float>>(F2Cptr& options, F2Cptr& ptree) {
      c_c_bpack_printoption(&options, &ptree);
    }
    template<> void HODLR_printoptions<std::complex<double>>(F2Cptr& options, F2Cptr& ptree) {
      z_c_bpack_printoption(&options, &ptree);
    }

    template<> void HODLR_printstats<float>(F2Cptr& stats, F2Cptr& ptree) {
      s_c_bpack_printstats(&stats, &ptree);
    }
    template<> void HODLR_printstats<double>(F2Cptr& stats, F2Cptr& ptree) {
      d_c_bpack_printstats(&stats, &ptree);
    }
    template<> void HODLR_printstats<std::complex<float>>(F2Cptr& stats, F2Cptr& ptree) {
      c_c_bpack_printstats(&stats, &ptree);
    }
    template<> void HODLR_printstats<std::complex<double>>(F2Cptr& stats, F2Cptr& ptree) {
      z_c_bpack_printstats(&stats, &ptree);
    }

    template<> void HODLR_createstats<float>(F2Cptr& stats) {
      s_c_bpack_createstats(&stats);
    }
    template<> void HODLR_createstats<double>(F2Cptr& stats) {
      d_c_bpack_createstats(&stats);
    }
    template<> void HODLR_createstats<std::complex<float>>(F2Cptr& stats) {
      c_c_bpack_createstats(&stats);
    }
    template<> void HODLR_createstats<std::complex<double>>(F2Cptr& stats) {
      z_c_bpack_createstats(&stats);
    }

    template<> void HODLR_set_D_option<float>
    (F2Cptr options, const std::string& opt, double v) {
      s_c_bpack_set_D_option(&options, opt.c_str(), v);
    }
    template<> void HODLR_set_D_option<double>
    (F2Cptr options, const std::string& opt, double v) {
      d_c_bpack_set_D_option(&options, opt.c_str(), v);
    }
    template<> void HODLR_set_D_option<std::complex<float>>
    (F2Cptr options, const std::string& opt, double v) {
      c_c_bpack_set_D_option(&options, opt.c_str(), v);
    }
    template<> void HODLR_set_D_option<std::complex<double>>
    (F2Cptr options, const std::string& opt, double v) {
      z_c_bpack_set_D_option(&options, opt.c_str(), v);
    }

    template<> void HODLR_set_I_option<float>
    (F2Cptr options, const std::string& opt, int v) {
      s_c_bpack_set_I_option(&options, opt.c_str(), v);
    }
    template<> void HODLR_set_I_option<double>
    (F2Cptr options, const std::string& opt, int v) {
      d_c_bpack_set_I_option(&options, opt.c_str(), v);
    }
    template<> void HODLR_set_I_option<std::complex<float>>
    (F2Cptr options, const std::string& opt, int v) {
      c_c_bpack_set_I_option(&options, opt.c_str(), v);
    }
    template<> void HODLR_set_I_option<std::complex<double>>
    (F2Cptr options, const std::string& opt, int v) {
      z_c_bpack_set_I_option(&options, opt.c_str(), v);
    }

    template<> double BPACK_get_stat<float>
    (F2Cptr stats, const std::string& name) {
      double val;
      s_c_bpack_getstats(&stats, name.c_str(), &val);
      return val;
    }
    template<> double BPACK_get_stat<double>
    (F2Cptr stats, const std::string& name) {
      double val;
      d_c_bpack_getstats(&stats, name.c_str(), &val);
      return val;
    }
    template<> double BPACK_get_stat<std::complex<float>>
    (F2Cptr stats, const std::string& name) {
      double val;
      c_c_bpack_getstats(&stats, name.c_str(), &val);
      return val;
    }
    template<> double BPACK_get_stat<std::complex<double>>
    (F2Cptr stats, const std::string& name) {
      double val;
      z_c_bpack_getstats(&stats, name.c_str(), &val);
      return val;
    }

    template<> void HODLR_construct_init<float,float>
    (int N, int d, float* data, int* nns, int lvls, int* tree, int* perm,
     int& lrow, F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr), C2Fptr fdata) {
      if (data)
        std::cerr << "ERROR: HODLR_construct_init does "
          "not support single precision" << std::endl;
      else
        s_c_bpack_construct_init
          (&N, &d, nullptr, nns, &lvls, tree, perm, &lrow, &ho_bf, &options,
           &stats, &msh, &kerquant, &ptree, C_FuncDistmn, C_FuncNearFar,
           fdata);
    }
    template<> void HODLR_construct_init<double,double>
    (int N, int d, double* data, int* nns, int lvls, int* tree, int* perm,
     int& lrow, F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr), C2Fptr fdata) {
      d_c_bpack_construct_init
        (&N, &d, data, nns, &lvls, tree, perm, &lrow, &ho_bf, &options,
         &stats, &msh, &kerquant, &ptree, C_FuncDistmn, C_FuncNearFar,
         fdata);
    }
    template<> void HODLR_construct_init<std::complex<float>,float>
    (int N, int d, float* data, int* nns, int lvls, int* tree,
     int* perm, int& lrow, F2Cptr& ho_bf, F2Cptr& options,
     F2Cptr& stats, F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr), C2Fptr fdata) {
      if (data)
        std::cerr << "ERROR: HODLR_construct_init does "
          "not support single precision" << std::endl;
      else
        c_c_bpack_construct_init
          (&N, &d, nullptr, nns, &lvls, tree, perm, &lrow, &ho_bf, &options,
           &stats, &msh, &kerquant, &ptree, C_FuncDistmn,
           C_FuncNearFar, fdata);
    }
    template<> void HODLR_construct_init<std::complex<double>,double>
    (int N, int d, double* data, int* nns, int lvls, int* tree,
     int* perm, int& lrow, F2Cptr& ho_bf, F2Cptr& options,
     F2Cptr& stats, F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr), C2Fptr fdata) {
      z_c_bpack_construct_init
        (&N, &d, data, nns, &lvls, tree, perm, &lrow, &ho_bf, &options,
         &stats, &msh, &kerquant, &ptree, C_FuncDistmn,
         C_FuncNearFar, fdata);
    }

    template<> void HODLR_construct_init_Gram<float,float>
    (int N, int d, float* data, int* nns, int lvls, int* tree, int* perm,
     int& lrow, F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, float*, C2Fptr),
     void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, float* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata) {
      if (data) {
        std::cerr << "ERROR: HODLR_construct_init_Gram "
          "does not support single precision" << std::endl;
      } else
        s_c_bpack_construct_init_gram
          (&N, &d, nullptr, nns, &lvls, tree, perm, &lrow, &ho_bf, &options,
           &stats, &msh, &kerquant, &ptree, C_FuncZmn, C_FuncZmnBlock, fdata);
    }
    template<> void HODLR_construct_init_Gram<double,double>
    (int N, int d, double* data, int* nns, int lvls, int* tree, int* perm,
     int& lrow, F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, double* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata) {
      d_c_bpack_construct_init_gram
        (&N, &d, data, nns, &lvls, tree, perm, &lrow, &ho_bf, &options,
         &stats, &msh, &kerquant, &ptree, C_FuncZmn, C_FuncZmnBlock, fdata);
    }
    template<> void HODLR_construct_init_Gram<std::complex<float>,float>
    (int N, int d, float* data, int* nns, int lvls, int* tree,
     int* perm, int& lrow, F2Cptr& ho_bf, F2Cptr& options,
     F2Cptr& stats, F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, std::complex<float>*, C2Fptr),
     void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, std::complex<float>* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata) {
      if (data)
        std::cerr << "ERROR: HODLR_construct_init_Gram "
          "does not support single precision" << std::endl;
      else
        c_c_bpack_construct_init_gram
          (&N, &d, nullptr, nns, &lvls, tree, perm, &lrow, &ho_bf, &options,
           &stats, &msh, &kerquant, &ptree,
           reinterpret_cast<
           void(*)(int*, int*, _Complex float*, C2Fptr)>(C_FuncZmn),
           reinterpret_cast<
           void(*)(int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
                   int* allrows, int* allcols, _Complex float* alldat_loc,
                   int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
                   C2Fptr elems)>(C_FuncZmnBlock), fdata);
    }
    template<> void HODLR_construct_init_Gram<std::complex<double>,double>
    (int N, int d, double* data, int* nns, int lvls, int* tree,
     int* perm, int& lrow, F2Cptr& ho_bf, F2Cptr& options,
     F2Cptr& stats, F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, std::complex<double>*, C2Fptr),
     void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, std::complex<double>* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata) {
      z_c_bpack_construct_init_gram
        (&N, &d, data, nns, &lvls, tree, perm, &lrow, &ho_bf, &options,
         &stats, &msh, &kerquant, &ptree,
         reinterpret_cast<
         void(*)(int*, int*, _Complex double*, C2Fptr)>(C_FuncZmn),
         reinterpret_cast<
         void(*)(int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
                 int* allrows, int* allcols, _Complex double* alldat_loc,
                 int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
                 C2Fptr elems)>(C_FuncZmnBlock), fdata);
    }


    template<> void HODLR_construct_element_compute<float>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, float*, C2Fptr),
     void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, float* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata) {
      s_c_bpack_construct_element_compute
        (&ho_bf, &options, &stats, &msh, &kerquant, &ptree,
         C_FuncZmn, C_FuncZmnBlock, fdata);
    }
    template<> void HODLR_construct_element_compute<double>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, double* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata) {
      d_c_bpack_construct_element_compute
        (&ho_bf, &options, &stats, &msh, &kerquant, &ptree,
         C_FuncZmn, C_FuncZmnBlock, fdata);
    }
    template<> void HODLR_construct_element_compute<std::complex<float>>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, std::complex<float>*, C2Fptr),
     void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, std::complex<float>* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata) {
      c_c_bpack_construct_element_compute
        (&ho_bf, &options, &stats, &msh, &kerquant, &ptree,
         reinterpret_cast<
         void(*)(int*, int*, _Complex float*, C2Fptr)>(C_FuncZmn),
         reinterpret_cast<
         void(*)(int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
                 int* allrows, int* allcols, _Complex float* alldat_loc,
                 int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
                 C2Fptr elems)>(C_FuncZmnBlock),
         fdata);
    }
    template<> void HODLR_construct_element_compute<std::complex<double>>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncZmn)(int*, int*, std::complex<double>*, C2Fptr),
     void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, std::complex<double>* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata) {
      z_c_bpack_construct_element_compute
        (&ho_bf, &options, &stats, &msh, &kerquant, &ptree,
         reinterpret_cast<
         void(*)(int*, int*, _Complex double*, C2Fptr)>(C_FuncZmn),
         reinterpret_cast<
         void(*)(int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
                 int* allrows, int* allcols, _Complex double* alldat_loc,
                 int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
                 C2Fptr elems)>(C_FuncZmnBlock),
         fdata);
    }

    template<> void HODLR_construct_matvec_compute<float>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (char const*, int*, int*, int*, const float*, float*, C2Fptr),
     C2Fptr fdata) {
      s_c_bpack_construct_matvec_compute
        (&ho_bf, &options, &stats, &msh, &kerquant, &ptree, matvec, fdata);
    }
    template<> void HODLR_construct_matvec_compute<double>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (char const*, int*, int*, int*, const double*, double*, C2Fptr),
     C2Fptr fdata) {
      d_c_bpack_construct_matvec_compute
        (&ho_bf, &options, &stats, &msh, &kerquant, &ptree, matvec, fdata);
    }
    template<> void HODLR_construct_matvec_compute<std::complex<float>>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (char const*, int*, int*, int*, const std::complex<float>*,
      std::complex<float>*, C2Fptr), C2Fptr fdata) {
      c_c_bpack_construct_matvec_compute
        (&ho_bf, &options, &stats, &msh, &kerquant, &ptree,
         reinterpret_cast<
         void(*)(char const*, int*, int*, int*, const _Complex float*,
                 _Complex float*, C2Fptr)>(matvec), fdata);
    }
    template<> void HODLR_construct_matvec_compute<std::complex<double>>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (char const*, int*, int*, int*, const std::complex<double>*,
      std::complex<double>*, C2Fptr), C2Fptr fdata) {
      z_c_bpack_construct_matvec_compute
        (&ho_bf, &options, &stats, &msh, &kerquant, &ptree,
         reinterpret_cast<
         void(*)(char const*, int*, int*, int*, const _Complex double*,
                 _Complex double*, C2Fptr)>(matvec), fdata);
    }

    template<> void LRBF_construct_init<float>
    (int M, int N, int& lrows, int& lcols, int* nnsr, int* nnsc,
     F2Cptr rmsh, F2Cptr cmsh, F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr), C2Fptr fdata) {
      s_c_bf_construct_init
        (&M, &N, &lrows, &lcols, nnsr, nnsc, &rmsh, &cmsh, &lr_bf, &options,
         &stats, &msh, &kerquant, &ptree, C_FuncDistmn, C_FuncNearFar,
         fdata);
    }
    template<> void LRBF_construct_init<double>
    (int M, int N, int& lrows, int& lcols, int* nnsr, int* nnsc,
     F2Cptr rmsh, F2Cptr cmsh, F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr), C2Fptr fdata) {
      d_c_bf_construct_init
        (&M, &N, &lrows, &lcols, nnsr, nnsc, &rmsh, &cmsh, &lr_bf, &options,
         &stats, &msh, &kerquant, &ptree, C_FuncDistmn, C_FuncNearFar,
         fdata);
    }
    template<> void LRBF_construct_init<std::complex<float>>
    (int M, int N, int& lrows, int& lcols, int* nnsr, int* nnsc,
     F2Cptr rmsh, F2Cptr cmsh, F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr), C2Fptr fdata) {
      c_c_bf_construct_init
        (&M, &N, &lrows, &lcols, nnsr, nnsc, &rmsh, &cmsh, &lr_bf, &options,
         &stats, &msh, &kerquant, &ptree, C_FuncDistmn, C_FuncNearFar,
         fdata);
    }
    template<> void LRBF_construct_init<std::complex<double>>
    (int M, int N, int& lrows, int& lcols, int* nnsr, int* nnsc,
     F2Cptr rmsh, F2Cptr cmsh, F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats,
     F2Cptr& msh, F2Cptr& kerquant, F2Cptr& ptree,
     void (*C_FuncDistmn)(int*, int*, double*, C2Fptr),
     void (*C_FuncNearFar)(int*, int*, int*, C2Fptr), C2Fptr fdata) {
      z_c_bf_construct_init
        (&M, &N, &lrows, &lcols, nnsr, nnsc, &rmsh, &cmsh, &lr_bf, &options,
         &stats, &msh, &kerquant, &ptree, C_FuncDistmn, C_FuncNearFar,
         fdata);
    }

    template<> void LRBF_construct_matvec_compute<float>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (const char*, int*, int*, int*, const float*,
      float*, C2Fptr, float*, float*), C2Fptr fdata) {
      s_c_bf_construct_matvec_compute
        (&lr_bf, &options, &stats, &msh, &kerquant, &ptree,
         matvec, fdata);
    }
    template<> void LRBF_construct_matvec_compute<double>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (const char*, int*, int*, int*, const double*,
      double*, C2Fptr, double*, double*), C2Fptr fdata) {
      d_c_bf_construct_matvec_compute
        (&lr_bf, &options, &stats, &msh, &kerquant, &ptree,
         matvec, fdata);
    }
    template<> void LRBF_construct_matvec_compute<std::complex<float>>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (char const*, int*, int*, int*, const std::complex<float>*,
      std::complex<float>*, C2Fptr, std::complex<float>*,
      std::complex<float>*), C2Fptr fdata) {
      c_c_bf_construct_matvec_compute
        (&lr_bf, &options, &stats, &msh, &kerquant, &ptree,
         reinterpret_cast<void(*)
         (char const*, int*, int*, int*, const _Complex float*,
          _Complex float*, C2Fptr, _Complex float*,
          _Complex float*)>(matvec), fdata);
    }
    template<> void LRBF_construct_matvec_compute<std::complex<double>>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*matvec)
     (char const*, int*, int*, int*, const std::complex<double>*,
      std::complex<double>*, C2Fptr, std::complex<double>*,
      std::complex<double>*), C2Fptr fdata) {
      z_c_bf_construct_matvec_compute
        (&lr_bf, &options, &stats, &msh, &kerquant, &ptree,
         reinterpret_cast<void(*)
         (char const*, int*, int*, int*, const _Complex double*,
          _Complex double*, C2Fptr, _Complex double*,
          _Complex double*)>(matvec), fdata);
    }

    template<> void LRBF_construct_element_compute<float>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, float* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata) {
      s_c_bf_construct_element_compute
        (&lr_bf, &options, &stats, &msh, &kerquant, &ptree,
         nullptr, C_FuncZmnBlock, fdata);
    }
    template<> void LRBF_construct_element_compute<double>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, double* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata) {
      d_c_bf_construct_element_compute
        (&lr_bf, &options, &stats, &msh, &kerquant, &ptree,
         nullptr, C_FuncZmnBlock, fdata);
    }
    template<> void LRBF_construct_element_compute<std::complex<float>>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, std::complex<float>* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata) {
      c_c_bf_construct_element_compute
        (&lr_bf, &options, &stats, &msh, &kerquant, &ptree,
         nullptr, reinterpret_cast<void(*)
         (int*, int*, int*, int*, int*, int*, _Complex float*,
          int*, int*, int*, int*, int*, C2Fptr)>(C_FuncZmnBlock), fdata);
    }
    template<> void LRBF_construct_element_compute<std::complex<double>>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& stats, F2Cptr& msh,
     F2Cptr& kerquant, F2Cptr& ptree, void (*C_FuncZmnBlock)
     (int* Ninter, int* Nallrows, int* Nallcols, int* Nalldat_loc,
      int* allrows, int* allcols, std::complex<double>* alldat_loc,
      int* rowids, int* colids, int* pgids, int* Npmap, int* pmaps,
      C2Fptr elems), C2Fptr fdata) {
      z_c_bf_construct_element_compute
        (&lr_bf, &options, &stats, &msh, &kerquant, &ptree,
         nullptr, reinterpret_cast<void(*)
         (int*, int*, int*, int*, int*, int*, _Complex double*,
          int*, int*, int*, int*, int*, C2Fptr)>(C_FuncZmnBlock), fdata);
    }

    template<> void HODLR_extract_elements<float>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, float* alldat_loc, int* rowidx, int* colidx,
     int* pgidx, int Npmap, int* pmaps) {
      s_c_bpack_extractelement
        (&ho_bf, &options, &msh, &stats, &ptree, &Ninter, &Nallrows,
         &Nallcols, &Nalldat_loc, allrows, allcols, alldat_loc,
         rowidx, colidx, pgidx, &Npmap, pmaps);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int f = BPACK_get_stat<float>(stats, "Flop_C_Extract");
      STRUMPACK_FLOPS(f);
      STRUMPACK_EXTRACTION_FLOPS(f);
#endif
    }
    template<> void HODLR_extract_elements<double>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, double* alldat_loc, int* rowidx, int* colidx,
     int* pgidx, int Npmap, int* pmaps) {
      d_c_bpack_extractelement
        (&ho_bf, &options, &msh, &stats, &ptree, &Ninter, &Nallrows,
         &Nallcols, &Nalldat_loc, allrows, allcols, alldat_loc,
         rowidx, colidx, pgidx, &Npmap, pmaps);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int f = BPACK_get_stat<double>(stats, "Flop_C_Extract");
      STRUMPACK_FLOPS(f);
      STRUMPACK_EXTRACTION_FLOPS(f);
#endif
    }
    template<> void HODLR_extract_elements<std::complex<float>>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, std::complex<float>* alldat_loc,
     int* rowidx, int* colidx, int* pgidx, int Npmap, int* pmaps) {
      c_c_bpack_extractelement
        (&ho_bf, &options, &msh, &stats, &ptree,
         &Ninter, &Nallrows, &Nallcols, &Nalldat_loc,
         allrows, allcols, reinterpret_cast<_Complex float*>(alldat_loc),
         rowidx, colidx, pgidx, &Npmap, pmaps);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int f = BPACK_get_stat<float>(stats, "Flop_C_Extract");
      STRUMPACK_FLOPS(f);
      STRUMPACK_EXTRACTION_FLOPS(f);
#endif
    }
    template<> void HODLR_extract_elements<std::complex<double>>
    (F2Cptr& ho_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, std::complex<double>* alldat_loc,
     int* rowidx, int* colidx, int* pgidx, int Npmap, int* pmaps) {
      z_c_bpack_extractelement
        (&ho_bf, &options, &msh, &stats, &ptree,
         &Ninter, &Nallrows, &Nallcols, &Nalldat_loc,
         allrows, allcols, reinterpret_cast<_Complex double*>(alldat_loc),
         rowidx, colidx, pgidx, &Npmap, pmaps);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int f = BPACK_get_stat<double>(stats, "Flop_C_Extract");
      STRUMPACK_FLOPS(f);
      STRUMPACK_EXTRACTION_FLOPS(f);
#endif
    }

    template<> void LRBF_extract_elements<float>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, float* alldat_loc, int* rowidx, int* colidx,
     int* pgidx, int Npmap, int* pmaps) {
      s_c_bf_extractelement
        (&lr_bf, &options, &msh, &stats, &ptree,
         &Ninter, &Nallrows, &Nallcols, &Nalldat_loc,
         allrows, allcols, alldat_loc,
         rowidx, colidx, pgidx, &Npmap, pmaps);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int f = BPACK_get_stat<float>(stats, "Flop_C_Extract");
      STRUMPACK_FLOPS(f);
      STRUMPACK_EXTRACTION_FLOPS(f);
#endif
    }
    template<> void LRBF_extract_elements<double>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, double* alldat_loc, int* rowidx, int* colidx,
     int* pgidx, int Npmap, int* pmaps) {
      d_c_bf_extractelement
        (&lr_bf, &options, &msh, &stats, &ptree,
         &Ninter, &Nallrows, &Nallcols, &Nalldat_loc,
         allrows, allcols, alldat_loc,
         rowidx, colidx, pgidx, &Npmap, pmaps);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int f = BPACK_get_stat<double>(stats, "Flop_C_Extract");
      STRUMPACK_FLOPS(f);
      STRUMPACK_EXTRACTION_FLOPS(f);
#endif
    }
    template<> void LRBF_extract_elements<std::complex<float>>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, std::complex<float>* alldat_loc,
     int* rowidx, int* colidx, int* pgidx, int Npmap, int* pmaps) {
      c_c_bf_extractelement
        (&lr_bf, &options, &msh, &stats, &ptree,
         &Ninter, &Nallrows, &Nallcols, &Nalldat_loc,
         allrows, allcols, reinterpret_cast<_Complex float*>(alldat_loc),
         rowidx, colidx, pgidx, &Npmap, pmaps);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int f = BPACK_get_stat<std::complex<float>>(stats, "Flop_C_Extract");
      STRUMPACK_FLOPS(f);
      STRUMPACK_EXTRACTION_FLOPS(f);
#endif
    }
    template<> void LRBF_extract_elements<std::complex<double>>
    (F2Cptr& lr_bf, F2Cptr& options, F2Cptr& msh, F2Cptr& stats,
     F2Cptr& ptree, int Ninter, int Nallrows, int Nallcols, int Nalldat_loc,
     int* allrows, int* allcols, std::complex<double>* alldat_loc,
     int* rowidx, int* colidx, int* pgidx, int Npmap, int* pmaps) {
      z_c_bf_extractelement
        (&lr_bf, &options, &msh, &stats, &ptree,
         &Ninter, &Nallrows, &Nallcols, &Nalldat_loc,
         allrows, allcols, reinterpret_cast<_Complex double*>(alldat_loc),
         rowidx, colidx, pgidx, &Npmap, pmaps);
#if defined(STRUMPACK_COUNT_FLOPS)
      long long int f = BPACK_get_stat<std::complex<double>>(stats, "Flop_C_Extract");
      STRUMPACK_FLOPS(f);
      STRUMPACK_EXTRACTION_FLOPS(f);
#endif
    }

    template<> void HODLR_deletestats<float>(F2Cptr& stats) { s_c_bpack_deletestats(&stats); }
    template<> void HODLR_deletestats<double>(F2Cptr& stats) { d_c_bpack_deletestats(&stats); }
    template<> void HODLR_deletestats<std::complex<float>>(F2Cptr& stats) { c_c_bpack_deletestats(&stats); }
    template<> void HODLR_deletestats<std::complex<double>>(F2Cptr& stats) { z_c_bpack_deletestats(&stats); }

    template<> void HODLR_deleteproctree<float>(F2Cptr& ptree) { s_c_bpack_deleteproctree(&ptree); }
    template<> void HODLR_deleteproctree<double>(F2Cptr& ptree) { d_c_bpack_deleteproctree(&ptree); }
    template<> void HODLR_deleteproctree<std::complex<float>>(F2Cptr& ptree) { c_c_bpack_deleteproctree(&ptree); }
    template<> void HODLR_deleteproctree<std::complex<double>>(F2Cptr& ptree) { z_c_bpack_deleteproctree(&ptree); }

    template<> void HODLR_deletemesh<float>(F2Cptr& mesh) { s_c_bpack_deletemesh(&mesh); }
    template<> void HODLR_deletemesh<double>(F2Cptr& mesh) { d_c_bpack_deletemesh(&mesh); }
    template<> void HODLR_deletemesh<std::complex<float>>(F2Cptr& mesh) { c_c_bpack_deletemesh(&mesh); }
    template<> void HODLR_deletemesh<std::complex<double>>(F2Cptr& mesh) { z_c_bpack_deletemesh(&mesh); }

    template<> void HODLR_deletekernelquant<float>(F2Cptr& kerquant) { s_c_bpack_deletekernelquant(&kerquant); }
    template<> void HODLR_deletekernelquant<double>(F2Cptr& kerquant) { d_c_bpack_deletekernelquant(&kerquant); }
    template<> void HODLR_deletekernelquant<std::complex<float>>(F2Cptr& kerquant) { c_c_bpack_deletekernelquant(&kerquant); }
    template<> void HODLR_deletekernelquant<std::complex<double>>(F2Cptr& kerquant) { z_c_bpack_deletekernelquant(&kerquant); }

    template<> void HODLR_delete<float>(F2Cptr& ho_bf) { s_c_bpack_delete(&ho_bf); }
    template<> void HODLR_delete<double>(F2Cptr& ho_bf) { d_c_bpack_delete(&ho_bf); }
    template<> void HODLR_delete<std::complex<float>>(F2Cptr& ho_bf) { c_c_bpack_delete(&ho_bf); }
    template<> void HODLR_delete<std::complex<double>>(F2Cptr& ho_bf) { z_c_bpack_delete(&ho_bf); }

    template<> void LRBF_deletebf<float>(F2Cptr& lr_bf) { s_c_bf_deletebf(&lr_bf); }
    template<> void LRBF_deletebf<double>(F2Cptr& lr_bf) { d_c_bf_deletebf(&lr_bf); }
    template<> void LRBF_deletebf<std::complex<float>>(F2Cptr& lr_bf) { c_c_bf_deletebf(&lr_bf); }
    template<> void LRBF_deletebf<std::complex<double>>(F2Cptr& lr_bf) { z_c_bf_deletebf(&lr_bf); }

    template<> void HODLR_deleteoptions<float>(F2Cptr& option) { s_c_bpack_deleteoption(&option); }
    template<> void HODLR_deleteoptions<double>(F2Cptr& option) { d_c_bpack_deleteoption(&option); }
    template<> void HODLR_deleteoptions<std::complex<float>>(F2Cptr& option) { c_c_bpack_deleteoption(&option); }
    template<> void HODLR_deleteoptions<std::complex<double>>(F2Cptr& option) { z_c_bpack_deleteoption(&option); }

    template<> void HODLR_mult<float>
    (char op, const float* X, float* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      s_c_bpack_mult(&op, X, Y, &Xlrows, &Ylrows,
                     &cols, &ho_bf, &options, &stats, &ptree);
    }
    template<> void HODLR_mult<double>
    (char op, const double* X, double* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      d_c_bpack_mult(&op, X, Y, &Xlrows, &Ylrows,
                     &cols, &ho_bf, &options, &stats, &ptree);
    }
    template<> void HODLR_mult<std::complex<float>>
    (char op, const std::complex<float>* X, std::complex<float>* Y,
     int Xlrows, int Ylrows, int cols, F2Cptr ho_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree) {
      c_c_bpack_mult(&op, reinterpret_cast<const _Complex float*>(X),
                     reinterpret_cast<_Complex float*>(Y), &Xlrows, &Ylrows,
                     &cols, &ho_bf, &options, &stats, &ptree);
    }
    template<> void HODLR_mult<std::complex<double>>
    (char op, const std::complex<double>* X, std::complex<double>* Y,
     int Xlrows, int Ylrows, int cols, F2Cptr ho_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree) {
      z_c_bpack_mult(&op, reinterpret_cast<const _Complex double*>(X),
                     reinterpret_cast<_Complex double*>(Y), &Xlrows, &Ylrows,
                     &cols, &ho_bf, &options, &stats, &ptree);
    }

    template<> void LRBF_mult<float>
    (char op, const float* X, float* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr lr_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      s_c_bf_mult(&op, X, Y, &Xlrows, &Ylrows, &cols,
                  &lr_bf, &options, &stats, &ptree);
    }
    template<> void LRBF_mult<double>
    (char op, const double* X, double* Y, int Xlrows, int Ylrows, int cols,
     F2Cptr lr_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      d_c_bf_mult(&op, X, Y, &Xlrows, &Ylrows, &cols,
                  &lr_bf, &options, &stats, &ptree);
    }
    template<> void LRBF_mult<std::complex<float>>
    (char op, const std::complex<float>* X, std::complex<float>* Y,
     int Xlrows, int Ylrows, int cols, F2Cptr lr_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree) {
      c_c_bf_mult(&op, reinterpret_cast<const _Complex float*>(X),
                  reinterpret_cast<_Complex float*>(Y),
                  &Xlrows, &Ylrows, &cols,
                  &lr_bf, &options, &stats, &ptree);
    }
    template<> void LRBF_mult<std::complex<double>>
    (char op, const std::complex<double>* X, std::complex<double>* Y,
     int Xlrows, int Ylrows, int cols, F2Cptr lr_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree) {
      z_c_bf_mult(&op, reinterpret_cast<const _Complex double*>(X),
                  reinterpret_cast<_Complex double*>(Y),
                  &Xlrows, &Ylrows, &cols,
                  &lr_bf, &options, &stats, &ptree);
    }

    template<> void HODLR_factor<float>
    (F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree, F2Cptr msh) {
      s_c_bpack_factor(&ho_bf, &options, &stats, &ptree, &msh);
    }
    template<> void HODLR_factor<double>
    (F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree, F2Cptr msh) {
      d_c_bpack_factor(&ho_bf, &options, &stats, &ptree, &msh);
    }
    template<> void HODLR_factor<std::complex<float>>
    (F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree, F2Cptr msh) {
      c_c_bpack_factor(&ho_bf, &options, &stats, &ptree, &msh);
    }
    template<> void HODLR_factor<std::complex<double>>
    (F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree, F2Cptr msh) {
      z_c_bpack_factor(&ho_bf, &options, &stats, &ptree, &msh);
    }

    template<> void HODLR_solve<float>
    (float* X, const float* B, int lrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      s_c_bpack_solve(X, const_cast<float*>(B), &lrows, &rhs,
                      &ho_bf, &options, &stats, &ptree);
    }
    template<> void HODLR_solve<double>
    (double* X, const double* B, int lrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      d_c_bpack_solve(X, const_cast<double*>(B), &lrows, &rhs,
                      &ho_bf, &options, &stats, &ptree);
    }
    template<> void HODLR_solve<std::complex<float>>
    (std::complex<float>* X, const std::complex<float>* B,
     int lrows, int rhs, F2Cptr ho_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree) {
      c_c_bpack_solve
        (reinterpret_cast<_Complex float*>(X),
         reinterpret_cast<_Complex float*>
         (const_cast<std::complex<float>*>(B)), &lrows, &rhs,
         &ho_bf, &options, &stats, &ptree);
    }
    template<> void HODLR_solve<std::complex<double>>
    (std::complex<double>* X, const std::complex<double>* B,
     int lrows, int rhs, F2Cptr ho_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree) {
      z_c_bpack_solve
        (reinterpret_cast<_Complex double*>(X),
         reinterpret_cast<_Complex double*>
         (const_cast<std::complex<double>*>(B)), &lrows, &rhs,
         &ho_bf, &options, &stats, &ptree);
    }

    template<> void HODLR_inv_mult<float>
    (char op, const float* B, float* X, int Xlrows, int Blrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      s_c_bpack_inv_mult
        (&op, B, X, &Xlrows, &Blrows, &rhs, &ho_bf, &options, &stats, &ptree);
    }
    template<> void HODLR_inv_mult<double>
    (char op, const double* B, double* X, int Xlrows, int Blrows, int rhs,
     F2Cptr ho_bf, F2Cptr options, F2Cptr stats, F2Cptr ptree) {
      d_c_bpack_inv_mult
        (&op, B, X, &Xlrows, &Blrows, &rhs, &ho_bf, &options, &stats, &ptree);
    }
    template<> void HODLR_inv_mult<std::complex<float>>
    (char op, const std::complex<float>* B, std::complex<float>* X,
     int Xlrows, int Blrows, int rhs, F2Cptr ho_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree) {
      c_c_bpack_inv_mult
        (&op, reinterpret_cast<const _Complex float*>(B),
         reinterpret_cast<_Complex float*>(X),
         &Xlrows, &Blrows, &rhs, &ho_bf, &options, &stats, &ptree);
    }
    template<> void HODLR_inv_mult<std::complex<double>>
    (char op, const std::complex<double>* B, std::complex<double>* X,
     int Xlrows, int Blrows, int rhs, F2Cptr ho_bf, F2Cptr options,
     F2Cptr stats, F2Cptr ptree) {
      z_c_bpack_inv_mult
        (&op, reinterpret_cast<const _Complex double*>(B),
         reinterpret_cast<_Complex double*>(X),
         &Xlrows, &Blrows, &rhs, &ho_bf, &options, &stats, &ptree);
    }

    int LRBF_treeindex_merged2child(int idx_merge) {
      int idx_child;
      d_c_bpack_treeindex_merged2child(&idx_merge, &idx_child);
      return idx_child;
    }

  } // end namespace HODLR
} // end namespace strumpack
