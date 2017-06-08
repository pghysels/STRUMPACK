/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The Regents of
 * the University of California, through Lawrence Berkeley National Laboratory
 * (subject to receipt of any required approvals from the U.S. Dept. of Energy).
 * All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Technology Transfer Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, prepare derivative works, and perform publicly and display publicly.
 * Beginning five (5) years after the date permission to assert copyright is
 * obtained from the U.S. Department of Energy, and subject to any subsequent five
 * (5) year renewals, the U.S. Government is granted for itself and others acting
 * on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, prepare derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 *
 */
#include <iostream>
#include <iomanip>
#include <fstream>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "TaskTimer.hpp"
#include "MPI_wrapper.hpp"

using namespace std::chrono;

tpoint TaskTimer::t_begin = GET_TIME_NOW();
TimerList TaskTimer::time_log_list = TimerList();

TaskTimer::TaskTimer(std::string name, int depth)
  : t_name(name), started(false), stopped(false), type(EXPLICITLY_NAMED_TASK), number(depth) {
#if defined(_OPENMP)
  tid = omp_get_thread_num();
#else
  tid = 0;
#endif
}

TaskTimer::TaskTimer(std::string name, std::function<void()> f, int depth)
  : t_name(name), started(false), stopped(false), type(EXPLICITLY_NAMED_TASK), number(depth) {
#if defined(_OPENMP)
  tid = omp_get_thread_num();
#else
  tid = 0;
#endif
  time(f);
}

TaskTimer::TaskTimer(TaskType task_type, int depth)
  : started(false), stopped(false), type(task_type), number(depth) {
#if defined(_OPENMP)
  tid = omp_get_thread_num();
#else
  tid = 0;
#endif
}

TaskTimer::~TaskTimer() {
  if (started && !stopped) stop();
}

void TaskTimer::time(std::function<void()> f) {
  start();
  f();
  stop();
}

double TaskTimer::elapsed() {
#if defined(USE_OPENMP_TIMER)
  double begin_time = t_start - TaskTimer::t_begin;
  double stop_time;
  if (stopped) stop_time = t_stop - TaskTimer::t_begin;
  else stop_time = GET_TIME_NOW() - TaskTimer::t_begin;
  return stop_time - begin_time;
#else
  duration<double> begin_time = duration_cast<duration<double>>(t_start - TaskTimer::t_begin);
  duration<double> stop_time;
  if (stopped) stop_time = duration_cast<duration<double>>(t_stop - TaskTimer::t_begin);
  else stop_time = duration_cast<duration<double>>(GET_TIME_NOW() - TaskTimer::t_begin);
  return stop_time.count() - begin_time.count();
#endif
}

void TaskTimer::print_name(std::ostream& os) {
  if (type == EXPLICITLY_NAMED_TASK) {
    os << t_name;
  } else {
    switch (type) {
    case RANDOM_SAMPLING:           os << "RANDOM_SAMPLING"; break;
    case RANDOM_GENERATE:           os << "RANDOM_GENERATE"; break;
    case FRONT_MULTIPLY_2D:         os << "FRONT_MULTIPLY_2D"; break;
    case UUTXR:                     os << "UUTXR"; break;
    case HSS_SCHUR_PRODUCT:         os << "HSS_SCHUR_PRODUCT"; break;
    case HSS_PRODUCT:               os << "HSS_PRODUCT"; break;
    case SKINNY_EXTEND_ADD_SEQSEQ:  os << "SKINNY_EXTEND_ADD_SEQSEQ"; break;
    case SKINNY_EXTEND_ADD_SEQ1:    os << "SKINNY_EXTEND_ADD_SEQ1"; break;
    case SKINNY_EXTEND_ADD_MPIMPI:  os << "SKINNY_EXTEND_ADD_MPIMPI"; break;
    case SKINNY_EXTEND_ADD_MPI1:    os << "SKINNY_EXTEND_ADD_MPI1"; break;
    case HSS_COMPRESS:              os << "HSS_COMPRESS"; break;
    case HSS_PARHQRINTERPOL:        os << "HSS_PARHQRINTERPOL"; break;
    case HSS_SEQHQRINTERPOL:        os << "HSS_SEQHQRINTERPOL"; break;
    case EXTRACT_2D:                os << "EXTRACT_2D"; break;
    case EXTRACT_SEP_2D:            os << "EXTRACT_SEP_2D"; break;
    case GET_SUBMATRIX_2D:          os << "GET_SUBMATRIX_2D"; break;
    case HSS_EXTRACT_SCHUR:         os << "HSS_EXTRACT_SCHUR"; break;
    case HSS_EXTRACT_FULL:          os << "HSS_EXTRACT_FULL"; break;
    case HSS_PARTIALLY_FACTOR:      os << "HSS_PARTIALLY_FACTOR"; break;
    case HSS_COMPUTE_SCHUR:         os << "HSS_COMPUTE_SCHUR"; break;
    case HSS_FACTOR:                os << "HSS_FACTOR"; break;
    case FORWARD_SOLVE:             os << "FORWARD_SOLVE"; break;
    case LOOK_LEFT:                 os << "LOOK_LEFT"; break;
    case SOLVE_LOWER:               os << "SOLVE_LOWER"; break;
    case SOLVE_LOWER_ROOT:          os << "SOLVE_LOWER_ROOT"; break;
    case BACKWARD_SOLVE:            os << "BACKWARD_SOLVE"; break;
    case SOLVE_UPPER:               os << "SOLVE_UPPER"; break;
    case LOOK_RIGHT:                os << "LOOK_RIGHT"; break;
    case PGEMR2D:                   os << "PGEMR2D"; break;
    case EXCHANGEINDICES:           os << "EXCHANGEINDICES"; break;
    case EXCHANGEINTEGER:	    os << "EXCHANGEINTEGER"; break;
    case PAREXTRACTSUBMATRIX:	    os << "PAREXTRACTSUBMATRIX"; break;
    case MERGEVECTOR:		    os << "MERGEVECTOR"; break;
    case SPLITVECTOR:		    os << "SPLITVECTOR"; break;
    case UPDATECOLUMNS:		    os << "UPDATECOLUMNS"; break;
    case GENERATESAMPLES:	    os << "GENERATESAMPLES"; break;
    case DISTVECTORTREE:	    os << "DISTVECTORTREE"; break;
    case UNDISTVECTORTREE:	    os << "UNDISTVECTORTREE"; break;
    case EXTRACTROWVECTORTREE:	    os << "EXTRACTROWVECTORTREE"; break;
    case EXTRACTLOCALSUBTREE:	    os << "EXTRACTLOCALSUBTREE"; break;
    case RESTART:                   os << "RESTART"; break;
    default: os << "SOMEOTHERTAKSNOTNAMED";
    }
  }
}

std::ostream& operator<<(std::ostream& os, TaskTimer& t) {
  //  if (t.type == EXPLICITLY_NAMED_TASK) return os;
#if defined(USE_OPENMP_TIMER)
  if (!t.stopped) t.stop();
  double begin_time = t.t_start - TaskTimer::t_begin;
  double stop_time = t.t_stop - TaskTimer::t_begin;
  t.print_name(os);
  os << " " << t.number << " [ "
     << std::setprecision(12) << begin_time << " , "
     << std::setprecision(12) << stop_time  << " ] thread: " << t.tid << "\n";
#else
  if (!t.stopped) t.stop();
  duration<double> begin_time = duration_cast<duration<double>>(t.t_start - TaskTimer::t_begin);
  duration<double> stop_time = duration_cast<duration<double>>(t.t_stop - TaskTimer::t_begin);
  t.print_name(os);
  os << " " << t.number << " [ "
     << std::setprecision(12) << begin_time.count() << " , "
     << std::setprecision(12) << stop_time.count()  << " ] thread: " << t.tid << "\n";
#endif
  return os;
}

void TaskTimer::start() {
  t_start = GET_TIME_NOW();
  started = true;
}

void TaskTimer::stop() {
  t_stop = GET_TIME_NOW();
  stopped = true;
  time_log_list.list[tid].push_back(*this);
}


TimerList::TimerList() : is_finalized(false) {
#if defined(_OPENMP)
  int max_t = omp_get_max_threads();
#else
  int max_t = 1;
#endif
  for (int i=0; i<max_t; i++)
    list.push_back(std::list<TaskTimer>());
}

void TimerList::Finalize() {
  TaskTimer::time_log_list.finalize();
}

void TimerList::finalize() {
  if (is_finalized) return;
  is_finalized = true;
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  if (mpi_finalized) {
    std::cerr << "# Warning, not printing out timings, since MPI_Finalized has already been called." << std::endl;
    return;
  }
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    std::ofstream log;
    log.open("time.log", std::ofstream::out);
    for (unsigned int thread=0; thread<list.size(); thread++)
      for (auto timing : list[thread]) log << timing;
    return;
  }

  for (int p=0; p<mpi_rank(); p++) MPI_Barrier(MPI_COMM_WORLD);
  {
    std::ofstream log;
    if (mpi_rank()) log.open("time.log", std::ofstream::out | std::ofstream::app);
    else log.open("time.log", std::ofstream::out);
    log << "# MPI rank " << mpi_rank() << std::endl;
    log << "# ==============" << std::endl;
    for (unsigned int thread=0; thread<list.size(); thread++)
      for (auto timing : list[thread]) log << timing;
    log << std::endl;
    log.close();
  }
  for (int p=mpi_rank(); p<=mpi_nprocs(); p++) MPI_Barrier(MPI_COMM_WORLD);

  int timers = 43;
  auto t = new double[4*timers];
  auto t_sum = t+timers;
  auto t_min = t+2*timers;
  auto t_max = t+3*timers;
  std::fill(t, t+timers, .0);
  int i=0;
  for (unsigned int thread=0; thread<list.size(); thread++)
    for (auto timing : list[thread]) {
      switch (timing.type) {
      case RANDOM_SAMPLING:          t[0]  += timing.elapsed(); break;
      case RANDOM_GENERATE:          t[1]  += timing.elapsed(); break;
      case FRONT_MULTIPLY_2D:        t[2]  += timing.elapsed(); break;
      case UUTXR:                    t[3]  += timing.elapsed(); break;
      case HSS_SCHUR_PRODUCT:        t[4]  += timing.elapsed(); break;
      case HSS_PRODUCT:              t[5]  += timing.elapsed(); break;
      case SKINNY_EXTEND_ADD_SEQSEQ: t[6]  += timing.elapsed(); break;
      case SKINNY_EXTEND_ADD_SEQ1:   t[7]  += timing.elapsed(); break;
      case SKINNY_EXTEND_ADD_MPIMPI: t[8]  += timing.elapsed(); break;
      case SKINNY_EXTEND_ADD_MPI1:   t[9]  += timing.elapsed(); break;
      case HSS_COMPRESS:             t[10] += timing.elapsed(); break;
      case HSS_PARHQRINTERPOL:       t[11] += timing.elapsed(); break;
      case HSS_SEQHQRINTERPOL:       t[12] += timing.elapsed(); break;
      case EXTRACT_2D:               t[13] += timing.elapsed(); break;
      case EXTRACT_SEP_2D:           t[14] += timing.elapsed(); break;
      case GET_SUBMATRIX_2D:         t[15] += timing.elapsed(); break;
      case HSS_EXTRACT_SCHUR:        t[16] += timing.elapsed(); break;
      case HSS_EXTRACT_FULL:         t[17] += timing.elapsed(); break;
      case GET_SUBMATRIX:            t[18] += timing.elapsed(); break;
      case HSS_PARTIALLY_FACTOR:     t[19] += timing.elapsed(); break;
      case HSS_COMPUTE_SCHUR:        t[20] += timing.elapsed(); break;
      case HSS_FACTOR:               t[21] += timing.elapsed(); break;
      case FORWARD_SOLVE:            t[22] += timing.elapsed(); break;
      case LOOK_LEFT:                t[23] += timing.elapsed(); break;
      case SOLVE_LOWER:              t[24] += timing.elapsed(); break;
      case SOLVE_LOWER_ROOT:         t[25] += timing.elapsed(); break;
      case BACKWARD_SOLVE:           t[26] += timing.elapsed(); break;
      case SOLVE_UPPER:              t[27] += timing.elapsed(); break;
      case LOOK_RIGHT:               t[28] += timing.elapsed(); break;
      case PGEMR2D:                  t[29] += timing.elapsed(); break;
      case EXCHANGEINDICES:          t[30] += timing.elapsed(); break;
      case EXCHANGEINTEGER:	     t[31] += timing.elapsed(); break;
      case PAREXTRACTSUBMATRIX:	     t[32] += timing.elapsed(); break;
      case MERGEVECTOR:		     t[33] += timing.elapsed(); break;
      case SPLITVECTOR:		     t[34] += timing.elapsed(); break;
      case UPDATECOLUMNS:	     t[35] += timing.elapsed(); break;
      case GENERATESAMPLES:	     t[36] += timing.elapsed(); break;
      case DISTVECTORTREE:	     t[37] += timing.elapsed(); break;
      case UNDISTVECTORTREE:	     t[38] += timing.elapsed(); break;
      case EXTRACTROWVECTORTREE:     t[39] += timing.elapsed(); break;
      case EXTRACTLOCALSUBTREE:	     t[40] += timing.elapsed(); break;
      case RESTART:		     t[41] += timing.elapsed(); break;
      case DISTMATRIXTREE:           t[42] += timing.elapsed(); break;
      default: break; //std::cerr << "unrecognized timer!!" << std::endl;
      }
    }
  MPI_Reduce(t, t_sum, timers, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(t, t_min, timers, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(t, t_max, timers, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (!mpi_rank()) {
    i=0;
    auto P = mpi_nprocs();
    std::cout << std::scientific << std::setprecision(3) << std::setw(8) << std::endl;
    std::cout << "+================================================================================+\n";
    std::cout << "| Op                         |    total   |     min    |     max    |     avg    |\n";
    std::cout << "|----------------------------+------------+------------+------------+------------|\n";
    std::cout << "| random_sampling            | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|    random_generate         | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|    front_multiply_2d       | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|    UUtxR                   | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|       HSS_Schur_product    | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    //std::cout << "|       HSS_product          | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl;
    i++;
    std::cout << "|       skinny_extend_add_ss | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|       skinny_extend_add_s  | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|       skinny_extend_add_mm | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|       skinny_extend_add_m  | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|----------------------------+------------+------------+------------+------------|\n";
    std::cout << "| HSS_compress               | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|    HSS_parHQRInterpol      | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|    HSS_seqHQRInterpol      | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|    extract_2d              | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|       extract_sep_2d       | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|       get_submatrix_2d     | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|          HSS_extract_Schur | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    //std::cout << "|          HSS_extract       | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl;
    i++;
    std::cout << "|       get_submatrix        | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|----------------------------+------------+------------+------------+------------|\n";
    std::cout << "| HSS_partially_factor       | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "| HSS_compute_Schur          | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "| HSS_factor                 | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|----------------------------+------------+------------+------------+------------|\n";
    std::cout << "| Forward_solve              | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|    look_left               | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|    solve_lower             | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|    solve_lower_root        | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "| Backward_solve             | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|    solve_upper             | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "|    look_right              | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "|----------------------------+------------+------------+------------+------------|\n";
    // std::cout << "| PGEMR2D                    | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "| EXCHANGEINDICES            | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "| EXCHANGEINTEGER            | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "| PAREXTRACTSUBMATRIX        | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "| MERGEVECTOR                | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "| SPLITVECTOR                | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "| UPDATECOLUMNS              | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "| GENERATESAMPLES            | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "| DISTVECTORTREE             | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "| UNDISTVECTORTREE           | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "| EXTRACTROWVECTORTREE       | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "| EXTRACTLOCALSUBTREE        | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "| RESTART                    | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    // std::cout << "| DISTMATRIXTREE             | " << std::setw(10) << t_sum[i] << " | " << std::setw(10) << t_min[i] << " | " << std::setw(10) << t_max[i] << " | " << std::setw(10) << t_sum[i]/P << " |" << std::endl; i++;
    std::cout << "+================================================================================+";
    std::cout << std::endl;
  }
  delete[] t;
}

