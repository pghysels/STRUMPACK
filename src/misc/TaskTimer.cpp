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
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "TaskTimer.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "misc/MPIWrapper.hpp"
#endif

using namespace std::chrono;
using namespace strumpack;

tpoint TaskTimer::t_begin = GET_TIME_NOW();
TimerList TaskTimer::time_log_list = TimerList();

TaskTimer::TaskTimer(std::string name, int depth)
  : t_name(name), started(false), stopped(false),
    type(TaskType::EXPLICITLY_NAMED_TASK), number(depth) {
#if defined(_OPENMP)
  tid = omp_get_thread_num();
#else
  tid = 0;
#endif
}

TaskTimer::TaskTimer(std::string name, std::function<void()> f, int depth)
  : t_name(name), started(false), stopped(false),
    type(TaskType::EXPLICITLY_NAMED_TASK), number(depth) {
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
  assert(started);
#if defined(USE_OPENMP_TIMER)
  double begin_time = t_start - TaskTimer::t_begin;
  double stop_time;
  if (stopped) stop_time = t_stop - TaskTimer::t_begin;
  else stop_time = GET_TIME_NOW() - TaskTimer::t_begin;
  return stop_time - begin_time;
#else
  duration<double> begin_time =
    duration_cast<duration<double>>(t_start - TaskTimer::t_begin);
  duration<double> stop_time;
  if (stopped)
    stop_time = duration_cast<duration<double>>(t_stop - TaskTimer::t_begin);
  else
    stop_time = duration_cast<duration<double>>
      (GET_TIME_NOW() - TaskTimer::t_begin);
  return stop_time.count() - begin_time.count();
#endif
}

void TaskTimer::print_name(std::ostream& os) {
  if (type == TaskType::EXPLICITLY_NAMED_TASK) {
    os << t_name;
  } else {
    switch (type) {
    case TaskType::RANDOM_SAMPLING:       os << "RANDOM_SAMPLING"; break;
    case TaskType::RANDOM_GENERATE:       os << "RANDOM_GENERATE"; break;
    case TaskType::FRONT_MULTIPLY_2D:     os << "FRONT_MULTIPLY_2D"; break;
    case TaskType::UUTXR:                 os << "UUTXR"; break;
    case TaskType::F22_MULT:              os << "F22_MULT"; break;
    case TaskType::HSS_SCHUR_PRODUCT:     os << "HSS_SCHUR_PRODUCT"; break;
    case TaskType::SKINNY_EXTEND_ADD_SEQSEQ: os << "SKINNY_EXTEND_ADD_SEQSEQ"; break;
    case TaskType::SKINNY_EXTEND_ADD_SEQ1:   os << "SKINNY_EXTEND_ADD_SEQ1"; break;
    case TaskType::SKINNY_EXTEND_ADD_MPIMPI: os << "SKINNY_EXTEND_ADD_MPIMPI"; break;
    case TaskType::SKINNY_EXTEND_ADD_MPI1:   os << "SKINNY_EXTEND_ADD_MPI1"; break;
    case TaskType::HSS_COMPRESS:          os << "HSS_COMPRESS"; break;
    case TaskType::HSS_COMPRESS_22:       os << "HSS_COMPRESS_22"; break;
    case TaskType::LRBF_COMPRESS:         os << "LRBF_COMPRESS"; break;
    case TaskType::CONSTRUCT_HIERARCHY:   os << "CONSTRUCT_HIERARCHY"; break;
    case TaskType::CONSTRUCT_INIT:        os << "CONSTRUCT_INIT"; break;
    case TaskType::CONSTRUCT_PTREE:       os << "CONSTRUCT_PTREE"; break;
    case TaskType::NEIGHBOR_SEARCH:       os << "NEIGHBOR_SEARCH"; break;
    case TaskType::HSS_PARHQRINTERPOL:    os << "HSS_PARHQRINTERPOL"; break;
    case TaskType::HSS_SEQHQRINTERPOL:    os << "HSS_SEQHQRINTERPOL"; break;
    case TaskType::EXTRACT_2D:            os << "EXTRACT_2D"; break;
    case TaskType::EXTRACT_2D_A2A:        os << "EXTRACT_2D_A2A"; break;
    case TaskType::EXTRACT_SEP_2D:        os << "EXTRACT_SEP_2D"; break;
    case TaskType::EXTRACT_ELEMS:         os << "EXTRACT_ELEMS"; break;
    case TaskType::BF_EXTRACT_TRAVERSE:   os << "BF_EXTRACT_TRAVERSE"; break;
    case TaskType::BF_EXTRACT_ENTRY:      os << "BF_EXTRACT_ENTRY"; break;
    case TaskType::BF_EXTRACT_COMM:       os << "BF_EXTRACT_COMM"; break;
    case TaskType::GET_SUBMATRIX_2D:      os << "GET_SUBMATRIX_2D"; break;
    case TaskType::GET_SUBMATRIX_2D_A2A:  os << "GET_SUBMATRIX_2D_A2A"; break;
    case TaskType::HSS_EXTRACT_SCHUR:     os << "HSS_EXTRACT_SCHUR"; break;
    case TaskType::HSS_PARTIALLY_FACTOR:  os << "HSS_PARTIALLY_FACTOR"; break;
    case TaskType::F11INV_MULT:           os << "F11INV_MULT"; break;
    case TaskType::HSS_COMPUTE_SCHUR:     os << "HSS_COMPUTE_SCHUR"; break;
    case TaskType::HSS_FACTOR:            os << "HSS_FACTOR"; break;
    case TaskType::FORWARD_SOLVE:         os << "FORWARD_SOLVE"; break;
    case TaskType::LOOK_LEFT:             os << "LOOK_LEFT"; break;
    case TaskType::SOLVE_LOWER:           os << "SOLVE_LOWER"; break;
    case TaskType::SOLVE_LOWER_ROOT:      os << "SOLVE_LOWER_ROOT"; break;
    case TaskType::BACKWARD_SOLVE:        os << "BACKWARD_SOLVE"; break;
    case TaskType::SOLVE_UPPER:           os << "SOLVE_UPPER"; break;
    case TaskType::LOOK_RIGHT:            os << "LOOK_RIGHT"; break;
    case TaskType::DISTMAT_EXTRACT_ROWS:  os << "DISTMAT_EXTRACT_ROWS"; break;
    case TaskType::DISTMAT_EXTRACT_COLS:  os << "DISTMAT_EXTRACT_COLS"; break;
    case TaskType::DISTMAT_EXTRACT:       os << "DISTMAT_EXTRACT"; break;
    case TaskType::QR:                    os << "QR"; break;
    case TaskType::REDUCE_SAMPLES:        os << "REDUCE_SAMPLES"; break;
    case TaskType::COMPUTE_SAMPLES:       os << "COMPUTE_SAMPLES"; break;
    case TaskType::ORTHO:                 os << "ORTHO"; break;
    case TaskType::REDIST_2D_TO_HSS:      os << "REDIST_2D_TO_HSS"; break;
    default: os << "OTHER_TASK";
    }
  }
}

void TaskTimer::print(std::ostream& os) {
  if (type == TaskType::EXPLICITLY_NAMED_TASK) return;
  if (!stopped) stop();
#if defined(USE_OPENMP_TIMER)
  double begin_time = t_start - TaskTimer::t_begin;
  double stop_time = t_stop - TaskTimer::t_begin;
#else
  double begin_time = duration_cast<duration<double>>
    (t_start - TaskTimer::t_begin).count();
  double stop_time = duration_cast<duration<double>>
    (t_stop - TaskTimer::t_begin).count();
#endif
  print_name(os);
#if !defined(STRUMPACK_USE_MPI)
  os << " " << number << " [ "
     << std::setprecision(12) << begin_time << " , "
     << std::setprecision(12) << stop_time  << " ] thread: " << tid << "\n";
#else
  MPIComm c;
  int rank = c.rank();
  os << " " << number << " [ "
     << std::setprecision(12) << begin_time << " , "
     << std::setprecision(12) << stop_time  << " ] thread: " << tid
     << " , rank: " << rank << "\n";
#endif
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

void TaskTimer::set_elapsed(double t) {
#if defined(USE_OPENMP_TIMER)
  started = stopped = true;
  t_start = 0;
  t_stop = t;
  time_log_list.list[tid].push_back(*this);
#endif
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
#if defined(STRUMPACK_TASK_TIMERS)
#if !defined(STRUMPACK_USE_MPI)
  std::ofstream log;
  log.open("time.log", std::ofstream::out);
  for (unsigned int thread=0; thread<list.size(); thread++)
    for (auto timing : list[thread]) //log << timing;
      timing.print(log);
  return;
#else
  if (is_finalized) return;
  is_finalized = true;
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  if (mpi_finalized) {
    std::cerr << "# Warning, not printing out timings,"
              << " since MPI_Finalized has already been called."
              << std::endl;
    return;
  }
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    std::ofstream log;
    log.open("time.log", std::ofstream::out);
    for (unsigned int thread=0; thread<list.size(); thread++)
      for (auto timing : list[thread]) //log << timing;
        timing.print(log);
    return;
  }

  MPIComm c;
  int rank = c.rank(), P = c.size();
  // for (int p=0; p<rank; p++) MPI_Barrier(MPI_COMM_WORLD);
  // {
  //   std::ofstream log;
  //   if (mpi_rank())
  //     log.open("time.log", std::ofstream::out | std::ofstream::app);
  //   else log.open("time.log", std::ofstream::out);
  //   for (unsigned int thread=0; thread<list.size(); thread++)
  //     for (auto timing : list[thread])
  //       timing.print(log);
  //   log.close();
  // }
  // for (int p=rank; p<=P; p++) MPI_Barrier(MPI_COMM_WORLD);

  int timers = int(TaskType::EXPLICITLY_NAMED_TASK);
  std::unique_ptr<double[]> t_(new double[4*timers]);
  auto t = t_.get();
  auto t_sum = t+timers;
  auto t_min = t+2*timers;
  auto t_max = t+3*timers;
  std::fill(t, t+timers, .0);
  int i=0;
  for (unsigned int thread=0; thread<list.size(); thread++)
    for (auto timing : list[thread])
      t[int(timing.type)] += timing.elapsed();
  std::copy(t, t+timers, t_sum);
  std::copy(t, t+timers, t_min);
  std::copy(t, t+timers, t_max);
  c.reduce(t_sum, timers, MPI_SUM);
  c.reduce(t_min, timers, MPI_MIN);
  c.reduce(t_max, timers, MPI_MAX);
  if (!rank) {
    auto print_line = [&](const std::string& op, TaskType idx) {
      std::cout << "|" << op << "| "
      << std::setw(10) << t_sum[int(idx)] << " | "
      << std::setw(10) << t_min[int(idx)] << " | "
      << std::setw(10) << t_max[int(idx)] << " | "
      << std::setw(10) << t_sum[int(idx)]/P << " |"
      << std::endl;
      i++;
    };
    std::string hline = "|----------------------------+------------+"
      "------------+------------+------------|\n";
    std::cout << std::scientific << std::setprecision(3)
              << std::setw(8) << std::endl;
    std::cout << "+=========================================================="
              << "======================+\n";
    std::cout << "| Op                         |    total   |     min    |"
              << "     max    |     avg    |\n";
    std::cout << hline;
    print_line(" random_sampling            ", TaskType::RANDOM_SAMPLING);
    print_line("    random_generate         ", TaskType::RANDOM_GENERATE);
    print_line("    front_multiply          ", TaskType::FRONT_MULTIPLY_2D);
    print_line("    CB_mult                 ", TaskType::UUTXR);
    print_line("       F22_mult             ", TaskType::F22_MULT);
    print_line("       Schur_mult           ", TaskType::HSS_SCHUR_PRODUCT);
    // print_line("       skinny_extend_add_ss ",
    //            TaskType::SKINNY_EXTEND_ADD_SEQSEQ);
    // print_line("       skinny_extend_add_s  ",
    //            TaskType::SKINNY_EXTEND_ADD_SEQ1);
    // print_line("       skinny_extend_add_mm ",
    //            TaskType::SKINNY_EXTEND_ADD_MPIMPI);
    // print_line("       skinny_extend_add_m  ",
    //            TaskType::SKINNY_EXTEND_ADD_MPI1);
    std::cout << hline;
    print_line(" H_compress                 ", TaskType::HSS_COMPRESS);
    print_line(" H_compress_22              ", TaskType::HSS_COMPRESS_22);
    print_line(" Construct_hierarchy        ", TaskType::CONSTRUCT_HIERARCHY);
    print_line("    Extract_graph           ", TaskType::EXTRACT_GRAPH);
    print_line("    Neighbor_search         ", TaskType::NEIGHBOR_SEARCH);
    print_line("    Construct_ptree         ", TaskType::CONSTRUCT_PTREE);
    print_line("    Construct_init          ", TaskType::CONSTRUCT_INIT);
    print_line(" LRBF_compress              ", TaskType::LRBF_COMPRESS);
    print_line("    HSS_parHQRInterpol      ", TaskType::HSS_PARHQRINTERPOL);
    print_line("    HSS_seqHQRInterpol      ", TaskType::HSS_SEQHQRINTERPOL);
    print_line("    QR                      ", TaskType::QR);
    print_line("    REDUCE_SAMPLES          ", TaskType::REDUCE_SAMPLES);
    print_line("    COMPUTE_SAMPLES         ", TaskType::COMPUTE_SAMPLES);
    print_line("    ORTHO                   ", TaskType::ORTHO);
    print_line("    REDIST_2D_TO_HSS        ", TaskType::REDIST_2D_TO_HSS);
    print_line("    extract_2d              ", TaskType::EXTRACT_2D);
    print_line("       extract_2d_a2a       ", TaskType::EXTRACT_2D_A2A);
    print_line("       extract_sep_2d       ", TaskType::EXTRACT_SEP_2D);
    print_line("       get_submatrix_2d     ", TaskType::GET_SUBMATRIX_2D);
    print_line("          extract_elems     ", TaskType::EXTRACT_ELEMS);
    print_line("            bf_traverse     ", TaskType::BF_EXTRACT_TRAVERSE);
    print_line("            bf_entry        ", TaskType::BF_EXTRACT_ENTRY);
    print_line("            bf_comm         ", TaskType::BF_EXTRACT_COMM);
    print_line("          barrier+all2all   ", TaskType::GET_SUBMATRIX_2D_BA2A);
    print_line("             all2all        ", TaskType::GET_SUBMATRIX_2D_A2A);
    print_line("       extract_Schur        ", TaskType::HSS_EXTRACT_SCHUR);
    print_line("       get_submatrix        ", TaskType::GET_SUBMATRIX);
    std::cout << hline;
    print_line(" Partial_factor             ",
               TaskType::HSS_PARTIALLY_FACTOR);
    print_line(" Compute_Schur              ", TaskType::HSS_COMPUTE_SCHUR);
    print_line(" Factor                     ", TaskType::HSS_FACTOR);
    std::cout << hline;
    print_line(" Forward_solve              ", TaskType::FORWARD_SOLVE);
    print_line("    look_left               ", TaskType::LOOK_LEFT);
    print_line("    solve_lower             ", TaskType::SOLVE_LOWER);
    print_line("    solve_lower_root        ", TaskType::SOLVE_LOWER_ROOT);
    print_line(" Backward_solve             ", TaskType::BACKWARD_SOLVE);
    print_line("    solve_upper             ", TaskType::SOLVE_UPPER);
    print_line("    look_right              ", TaskType::LOOK_RIGHT);
    std::cout << hline;
    print_line(" DISTMAT_EXTRACT_ROWS       ",
               TaskType::DISTMAT_EXTRACT_ROWS);
    print_line(" DISTMAT_EXTRACT_COLS       ",
               TaskType::DISTMAT_EXTRACT_COLS);
    print_line(" DISTMAT_EXTRACT            ", TaskType::DISTMAT_EXTRACT);
    std::cout << "+=========================================================="
              << "======================+";
    std::cout << std::endl;
  }
#endif
#endif
}

