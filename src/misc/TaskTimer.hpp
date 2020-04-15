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
#ifndef TASK_TIMER_HPP
#define TASK_TIMER_HPP

#include <list>
#include <vector>
#include <string>
#include <chrono>
#include <functional>
#include "StrumpackConfig.hpp"

namespace strumpack {

#if defined(_OPENMP)
#define USE_OPENMP_TIMER
#endif

#if defined(USE_OPENMP_TIMER)
#define GET_TIME_NOW() omp_get_wtime()
  typedef double tpoint;
#else
  typedef std::chrono::high_resolution_clock timer;
#define GET_TIME_NOW() timer::now()
  typedef timer::time_point tpoint;
#endif

  enum class TaskType : int
    {
     RANDOM_SAMPLING=0, RANDOM_GENERATE, FRONT_MULTIPLY_2D, UUTXR, F22_MULT,
     HSS_SCHUR_PRODUCT, SKINNY_EXTEND_ADD_SEQSEQ, SKINNY_EXTEND_ADD_SEQ1,
     SKINNY_EXTEND_ADD_MPIMPI, SKINNY_EXTEND_ADD_MPI1, HSS_COMPRESS,
     HSS_COMPRESS_22, EXTRACT_GRAPH, CONSTRUCT_PTREE, CONSTRUCT_HIERARCHY,
     NEIGHBOR_SEARCH, LRBF_COMPRESS, CONSTRUCT_INIT, HSS_PARHQRINTERPOL,
     HSS_SEQHQRINTERPOL, EXTRACT_ELEMS, EXTRACT_2D, EXTRACT_2D_A2A,
     EXTRACT_SEP_2D, GET_SUBMATRIX_2D, GET_SUBMATRIX_2D_A2A,
     BF_EXTRACT_TRAVERSE, BF_EXTRACT_ENTRY, BF_EXTRACT_COMM,
     GET_SUBMATRIX_2D_BA2A, HSS_EXTRACT_SCHUR, GET_SUBMATRIX,
     HSS_PARTIALLY_FACTOR, F11INV_MULT, HSS_COMPUTE_SCHUR, HSS_FACTOR,
     FORWARD_SOLVE, LOOK_LEFT, SOLVE_LOWER, SOLVE_LOWER_ROOT, BACKWARD_SOLVE,
     SOLVE_UPPER, LOOK_RIGHT, DISTMAT_EXTRACT_ROWS, DISTMAT_EXTRACT_COLS,
     DISTMAT_EXTRACT, QR, REDUCE_SAMPLES, COMPUTE_SAMPLES, ORTHO,
     REDIST_2D_TO_HSS, EXPLICITLY_NAMED_TASK  // leave this one last
    };

  class TimerList;

  class TaskTimer {
  public:
    TaskTimer(TaskType task_type, int depth=1);
    TaskTimer(std::string name, int depth=1);
    TaskTimer(std::string name, std::function<void()> f, int depth=1);
    ~TaskTimer();

    void time(std::function<void()> f);
    void start();
    void stop();
    void set_elapsed(double t);

    double elapsed();

    std::string t_name;
    tpoint t_start;
    tpoint t_stop;

    bool started;
    bool stopped;

    TaskType type;
    int number;
    int tid;

    static tpoint t_begin;
    static TimerList time_log_list;

    void print_name(std::ostream& os);
    //friend std::ostream& operator<<(std::ostream& os, TaskTimer& t);
    void print(std::ostream& os);
  };

  class TimerList {
  public:
    TimerList();

    static void Finalize();
    void finalize();
    bool is_finalized;
    std::vector<std::list<TaskTimer>> list;
  };

#if !defined(STRUMPACK_TASK_TIMERS)

#define TIMER_TIME(name, nr, timer) (void)0
#define TIMER_DEFINE(name, nr, timer) (void)0
#define TIMER_START(timer) (void)0
#define TIMER_STOP(timer) (void)0

#else // STRUMPACK_TASK_TIMERS

#define TIMER_TIME(type, depth, timer)          \
  TaskTimer timer(type, depth);                 \
  timer.start();
#define TIMER_DEFINE(type, depth, timer)        \
  TaskTimer timer(type, depth);
#define TIMER_START(timer) timer.start();
#define TIMER_STOP(timer) timer.stop();

#endif // STRUMPACK_TASK_TIMERS

} // end namespace strumpack


#endif // TASK_TIMER_HPP
