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
#ifndef STRUMPACK_ORDERING_BLASLAPACKWRAPPER_HPP
#define STRUMPACK_ORDERING_BLASLAPACKWRAPPER_HPP

#include <type_traits>
#include <random>

#include "StrumpackParameters.hpp"
#include "StrumpackFortranCInterface.h"
#if defined(STRUMPACK_USE_MPI)
#include "misc/MPIWrapper.hpp"
#endif
#include "misc/TaskTimer.hpp"

// #include "dense/BLASLAPACKWrapper.hpp"

namespace strumpack {
  namespace ordering {

    enum class ThreadingModel {
      SEQ,   /* No threading. */
      PAR,   /* Create an openmp parallel region. */
      LOOP,  /* Do not create an openmp parallel region, this is called
                from within an existing parallel region, use openmp loop
                parallelism. */
      TASK   /* Do not create an openmp parallel region, this is called
                from within an existing parallel region, from within an
                openmp task. Use openmp task parallelism. */
      //GPU
    };

    template<typename T, typename Comp> void
    merge_sort_recursive(std::size_t n, T* x, const Comp& c) {
      if (n >= (1<<14)) {
        std::size_t half = n/2;
#pragma omp taskgroup
        {
#pragma omp task shared(x) untied
          merge_sort_recursive(half, x, c);
#pragma omp task shared(x) untied
          merge_sort_recursive(n-half, x+half, c);
#pragma omp taskyield
        }
        std::inplace_merge(x, x+half, x+n, c);
      } else std::sort(x, x+n, c);
    }

    template<typename T, typename Comp>
    void sort(ThreadingModel par, std::size_t n, T* x, const Comp& c) {
      TIMER_TIME(TaskType::SORT, 1, tsort);
      switch (par) {
      case ThreadingModel::SEQ:
        std::sort(x, x+n, c); break;
      case ThreadingModel::TASK:
        merge_sort_recursive(n, x, c); break;
      case ThreadingModel::PAR:
#pragma omp parallel
#pragma omp single nowait
        merge_sort_recursive(n, x, c);
        break;
      case ThreadingModel::LOOP:
#pragma omp single
        merge_sort_recursive(n, x, c);
        break;
      }
#if !defined(NDEBUG)
      for (std::size_t i=1; i<n; i++) { assert(!c(x[i], x[i-1])); }
#endif
    }

    template<typename T>
    void sort(ThreadingModel par, std::size_t n, T* x) {
      sort(par, n, x, std::less<T>());
    }

    template<typename T, typename intt, typename Comp>
    void inplace_merge_ranges_recursive
    (T* x, std::size_t s, const intt* offsets, const Comp& c) {
      if (s == 1) return;
      std::size_t half = s/2;
#pragma omp taskgroup
      {
#pragma omp task shared(x,offsets) untied
        inplace_merge_ranges_recursive(x, half, offsets, c);
#pragma omp task shared(x,offsets) untied
        inplace_merge_ranges_recursive(x, s-half, offsets+half, c);
#pragma omp taskyield
      }
      std::inplace_merge(x+offsets[0], x+offsets[half], x+offsets[s], c);
    }

    template<typename T, typename intt, typename Comp>
    void inplace_merge_ranges
    (ThreadingModel par, T* x, std::size_t s,
     const intt* offsets, const Comp& c) {
      TIMER_TIME(TaskType::MERGE_RANGES, 1, tmr);
      //TIMER_TIME(TaskType::SORT, 1, tsort);
      switch (par) {
      case ThreadingModel::SEQ:
        inplace_merge_ranges_recursive(x, s, offsets, c);
        break;
      case ThreadingModel::TASK:
        inplace_merge_ranges_recursive(x, s, offsets, c);
        break;
      case ThreadingModel::PAR:
#pragma omp parallel
#pragma omp single nowait
        inplace_merge_ranges_recursive(x, s, offsets, c);
        break;
      case ThreadingModel::LOOP:
#pragma omp single
        inplace_merge_ranges_recursive(x, s, offsets, c);
        break;
      }
#if !defined(NDEBUG)
      for (intt i=1; i<offsets[s]; i++) { assert(!c(x[i], x[i-1])); }
#endif
    }

    template<typename scalart> void print
    (const std::string& name, int m, int n, scalart* X, int ldX) {
      std::cout << std::setprecision(16) << name << " = [";
      for (int j=0; j<m; j++) {
        for (int i=0; i<n; i++)
          std::cout << X[j+i*ldX] << " ";
        std::cout << std::endl;
      }
      std::cout << "];" << std::endl;
    }

    // convert work[0] to long long lwork
    inline long long lapack_lwork(const double work0) {
      return static_cast<std::size_t>(work0);
    }
    inline long long lapack_lwork(const float work0) { // TODO
      return static_cast<std::size_t>
        (std::nextafter(work0, std::numeric_limits<float>::max()));
    }

    // convert lwork long long work[0] so that long long(work[0]) >= lwork
    template<typename T> T lapack_work0(const long long lwork) {
      T lwrk = T(lwork) ;
      if (lwrk >= lwork) return lwrk;
      else return std::nextafter(lwrk, std::numeric_limits<T>::max());
    }

    extern "C" {
#define SLAMCH_FC FC_GLOBAL(slamch, SLAMCH)
#define DLAMCH_FC FC_GLOBAL(dlamch, DLAMCH)
      float slamch_(char* cmach);
      double dlamch_(char* cmach);
    }

    template<typename real> inline real lamch(char cmach);
    template<> inline float
    lamch<float>(char cmach) { return slamch_(&cmach); }
    template<> inline double
    lamch<double>(char cmach) { return dlamch_(&cmach); }


    template<typename scalart> void
    random_vector(int n, scalart* x) {
      // TODO seeding in parallel
      std::minstd_rand0 gen;
      std::uniform_real_distribution<scalart> dist(-1., 1.);
      std::generate(x, x+n, [&]{ return dist(gen); });
    }
    template<typename scalart, typename alloc = std::allocator<scalart>>
    std::vector<scalart,alloc> random_vector(int n) {
      std::vector<scalart,alloc> x(n);
      random_vector(n, x.data());
      return x;
    }

    // TODO threading
    template<typename scalart> scalart
    median(int n, const scalart* x) {
      std::vector<scalart> tmp;
      tmp.assign(x, x+n);
      std::nth_element(tmp.begin(), tmp.begin()+n/2, tmp.end());
      //__gnu_parallel::nth_element(tmp.begin(), tmp.begin()+n/2, tmp.end());
      return tmp[n/2];
    }

#if defined(PROJECTND_USE_MPI)
    template<typename scalart,typename intt> scalart
    median(int n, const scalart* x, const intt* dist, const MPIComm& comm) {
      scalart m = 0;
      if (comm.is_root()) {
        auto P = comm.size();
        std::unique_ptr<int[]> iwork(new int[2*P]);
        auto rcnts = iwork.get();
        auto displ = rcnts + P;
        for (int p=0; p<P; p++) {
          rcnts[p] = dist[p+1] - dist[p];
          displ[p] = dist[p];
        }
        std::unique_ptr<scalart[]> xg(new scalart[dist[P]]);
        MPI_Gatherv
          (x, n, mpi_type<scalart>(), xg.get(), rcnts,
           displ, mpi_type<scalart>(), 0, comm.comm());
        m = median(n, xg.get());
      } else
        MPI_Gatherv
          (x, n, mpi_type<scalart>(), nullptr, nullptr,
           nullptr, mpi_type<scalart>(), 0, comm.comm());
      comm.broadcast(m);
      return m;
    }
#endif

    // TODO par == loop ??
    template<ThreadingModel par, typename scalart>
    scalart norm_squared(int n, const scalart* x) {
      TIMER_TIME(TaskType::NORM, 1, tnorm);
      if constexpr (par == ThreadingModel::SEQ) {
        double s = 0.;
        for (int i=0; i<n; i++)
          s += double(x[i])*double(x[i]);
        return s;
      }
      else if constexpr (par == ThreadingModel::PAR) {
        double s = 0.;
#pragma omp parallel
#pragma omp for simd reduction(+:s)
        for (int i=0; i<n; i++)
          s += double(x[i])*double(x[i]);
        return s;
      }
      else if constexpr(par == ThreadingModel::TASK) {
#if _OPENMP >= 201811
        double s = 0.;
#pragma omp taskloop simd reduction(+:s) default(shared)
        for (int i=0; i<n; i++)
          s += double(x[i])*double(x[i]);
        return s;
#else
#if defined(_OPENMP)
        int B = 5000; // TODO tune task minimum block size!!
        const int nb = std::min(int(std::ceil(float(n) / B)),
                                omp_get_num_threads()*4);
        B = int(std::ceil(float(n) / nb));
#else
        const int nb = 1;
        const int B = n;
#endif
        double stotal = 0;
#pragma omp taskloop grainsize(1) default(shared)
        for (int b=0; b<nb; b++) {
          const auto lb = b*B;
          const auto ub = std::min((b+1)*B, n);
          double s = 0.;
          for (int i=lb; i<ub; i++)
            s += double(x[i])*double(x[i]);
#pragma omp critical
          stotal += s;
        }
        return stotal;
#endif
      }
    }

#if defined(PROJECTND_USE_MPI)
    template<ThreadingModel par, typename scalart> scalart
    norm_squared(int n, const scalart* x, const MPIComm& comm) {
      TIMER_TIME(TaskType::NORM_MPI, 1, tnorm);
      return comm.all_reduce(norm_squared<par>(n, x), MPI_SUM);
    }
#endif

    template<ThreadingModel par, typename scalart> scalart
    norm(int n, const scalart* x) {
      return std::sqrt(norm_squared<par>(n, x));
    }

#if defined(PROJECTND_USE_MPI)
    template<ThreadingModel par, typename scalart> scalart
    norm(int n, const scalart* x, const MPIComm& comm) {
      return std::sqrt(norm_squared<par>(n, x, comm));
    }
#endif

    template<ThreadingModel par, typename scalart>
    scalart dot(int n, const scalart* x, const scalart* y) {
      TIMER_TIME(TaskType::DOT, 1, tdot);
      if constexpr (par == ThreadingModel::SEQ) {
        double s = 0.;
        for (int i=0; i<n; i++)
          s += double(x[i])*double(y[i]);
        return s;
      }
      else if constexpr (par == ThreadingModel::PAR) {
        double s = 0.;
#pragma omp parallel
#pragma omp for simd reduction(+:s)
        for (int i=0; i<n; i++) s += x[i]*y[i];
        return s;
      }
      else if constexpr (par == ThreadingModel::TASK) {
#if _OPENMP >= 201811
        double s = 0.;
#pragma omp taskloop simd reduction(+:s)
        for (int i=0; i<n; i++) s += x[i]*y[i];
        return s;
#else
#if defined(_OPENMP)
        int B = 5000; // TODO tune task minimum block size!!
        const int nb = std::min(int(std::ceil(float(n) / B)),
                                omp_get_num_threads()*4);
        B = int(std::ceil(float(n) / nb));
#else
        const int nb = 1;
        const int B = n;
#endif
        double stotal = 0;
#pragma omp taskloop grainsize(1) default(shared)
        for (int b=0; b<nb; b++) {
          const auto lb = b*B;
          const auto ub = std::min((b+1)*B, n);
          double s = 0.;
          for (int i=lb; i<ub; i++)
            s += double(x[i])*double(y[i]);
#pragma omp critical
          stotal += s;
        }
        return stotal;
#endif
      }
    }

#if defined(PROJECTND_USE_MPI)
    template<ThreadingModel par, typename scalart> scalart
    dot(int n, const scalart* x, const scalart* y, const MPIComm& comm) {
      TIMER_TIME(TaskType::DOT_MPI, 1, tdot);
      return comm.all_reduce(dot<par>(n, x, y), MPI_SUM);
    }
#endif

    template<ThreadingModel par, typename scalart>
    std::pair<scalart,scalart> dot_norm_squared
    (int n, const scalart* x, const scalart* y) {
      TIMER_TIME(TaskType::DOT, 1, tdot);
      TIMER_TIME(TaskType::NORM, 1, tnorm);
      if constexpr (par == ThreadingModel::SEQ) {
        double d = 0., ns = 0.;
        for (int i=0; i<n; i++) {
          d += double(x[i])*double(y[i]);
          ns += double(x[i])*double(x[i]);
        }
        return {d, ns};
      }
      else if constexpr (par == ThreadingModel::PAR) {
        double d = 0., ns = 0.;
#pragma omp parallel
#pragma omp for simd reduction(+:d,ns)
        for (int i=0; i<n; i++) {
          d += double(x[i])*double(y[i]);
          ns += double(x[i])*double(x[i]);
        }
        return {d, ns};
      }
      else if constexpr (par == ThreadingModel::TASK) {
#if _OPENMP >= 201811
        double d = 0., ns = 0.;
#pragma omp taskloop simd reduction(+:d,ns) default(shared)
        for (int i=0; i<n; i++) {
          d += double(x[i])*double(y[i]);
          ns += double(x[i])*double(x[i]);
        }
        return {d, ns};
#else
#if defined(_OPENMP)
        int B = 5000; // TODO tune task minimum block size!!
        const int nb = std::min(int(std::ceil(float(n) / B)),
                                omp_get_num_threads()*4);
        B = int(std::ceil(float(n) / nb));
#else
        const int nb = 1;
        const int B = n;
#endif
        double dtot = 0., nstot = 0.;
#pragma omp taskloop grainsize(1) default(shared)
        for (int b=0; b<nb; b++) {
          const auto lb = b*B;
          const auto ub = std::min((b+1)*B, n);
          double d = 0., ns = 0.;
          for (int i=lb; i<ub; i++) {
            d += double(x[i])*double(y[i]);
            ns += double(x[i])*double(x[i]);
          }
#pragma omp critical
          {
            dtot += d;
            nstot += ns;
          }
        }
        return {dtot, nstot};
#endif
      }
    }

    template<ThreadingModel par, typename scalart>
    scalart Rayleigh_quotient(int n, const scalart* x, const scalart* y) {
      auto [t0, t1] = dot_norm_squared<par>(n, x, y);
      return t0 / t1;
    }

#if defined(PROJECTND_USE_MPI)
    template<ThreadingModel par, typename scalart> scalart
    Rayleigh_quotient(int n, const scalart* x, const scalart* y,
                      const MPIComm& comm) {
      TIMER_TIME(TaskType::DOT_MPI, 1, tdot);
      double tmp[2];
      std::tie(tmp[0], tmp[1]) = dot_norm_squared<par>(n, x, y);
      comm.all_reduce(tmp, 2, MPI_SUM);
      return tmp[0] / tmp[1];
    }
#endif

    template<ThreadingModel par, typename scalart>
    void scale(int n, scalart* x, scalart alpha) {
      TIMER_TIME(TaskType::SCALE, 1, tadd);
      if constexpr (par == ThreadingModel::SEQ) {
        for (int i=0; i<n; i++) x[i] *= alpha;
      }
      else if constexpr (par == ThreadingModel::PAR) {
#pragma omp parallel for simd
        for (int i=0; i<n; i++) x[i] *= alpha;
      }
      else if constexpr (par == ThreadingModel::TASK) {
#pragma omp taskloop simd default(shared)
        for (int i=0; i<n; i++) x[i] *= alpha;
      }
      else if constexpr (par == ThreadingModel::LOOP) {
#pragma omp for simd
        for (int i=0; i<n; i++) x[i] *= alpha;
      }
    }


    template<ThreadingModel par, typename scalart> void
    normalize(int n, scalart* x) {
      auto s = norm<par>(n, x);
      if (s > lamch<scalart>('P'))
        scale<par>(n, x, scalart(1.)/s);
      else {
        random_vector(n, x);
        normalize<par>(n, x);
      }
    }

#if defined(PROJECTND_USE_MPI)
    template<ThreadingModel par, typename scalart> void
    normalize(int n, scalart* x, const MPIComm& comm) {
      auto s = norm<par>(n, x, comm);
      scale<par>(n, x, scalart(1.)/s);
    }
#endif

    template<ThreadingModel par, typename scalart>
    scalart sum(int n, const scalart* x) {
      TIMER_TIME(TaskType::SUM, 1, tadd);
      if constexpr (par == ThreadingModel::SEQ) {
        double s = 0.;
        for (int i=0; i<n; i++) s += x[i];
        return s;
      }
      else if constexpr (par == ThreadingModel::PAR) {
        double s = 0.;
#pragma omp parallel for simd reduction(+:s)
        for (int i=0; i<n; i++) s += x[i];
        return s;
      }
      else if constexpr (par == ThreadingModel::TASK) {
#if _OPENMP >= 201811
        double s = 0.;
#pragma omp taskloop simd reduction(+:s) default(shared)
        for (int i=0; i<n; i++) s += x[i];
        return s;
#else
#if defined(_OPENMP)
        int B = 5000; // TODO tune task minimum block size!!
        const int nb = std::min(int(std::ceil(float(n) / B)),
                                omp_get_num_threads()*4);
        B = int(std::ceil(float(n) / nb));
#else
        const int nb = 1;
        const int B = n;
#endif
        double stotal = 0;
#pragma omp taskloop grainsize(1) default(shared)
        for (int b=0; b<nb; b++) {
          const auto lb = b*B;
          const auto ub = std::min((b+1)*B, n);
          double s = 0.;
          for (int i=lb; i<ub; i++)
            s += x[i];
#pragma omp critical
          stotal += s;
        }
        return stotal;
#endif
      }
    }

    template<typename scalart> scalart
    sum(ThreadingModel par, int n, const scalart* x) {
      switch (par) {
      case ThreadingModel::SEQ: return sum<ThreadingModel::SEQ>(n, x);
      case ThreadingModel::PAR: return sum<ThreadingModel::PAR>(n, x);
      case ThreadingModel::TASK: return sum<ThreadingModel::TASK>(n, x);
      case ThreadingModel::LOOP:
      default: assert(true);
      }
      return 0.;
    }

#if defined(PROJECTND_USE_MPI)
    template<ThreadingModel par, typename scalart> scalart
    sum(int n, const scalart* x, const MPIComm& comm) {
      return comm.all_reduce(sum<par>(n, x), MPI_SUM);
    }

    template<typename scalart> scalart
    sum(ThreadingModel par, int n, const scalart* x, const MPIComm& comm) {
      return comm.all_reduce(sum(par, n, x), MPI_SUM);
    }
#endif

    template<ThreadingModel par, typename scalart>
    std::pair<scalart,scalart> minmax(int n, const scalart* x) {
      if constexpr (par == ThreadingModel::SEQ) {
        scalart m = std::numeric_limits<scalart>::max(),
          M = std::numeric_limits<scalart>::lowest();
        for (int i=0; i<n; i++) {
          m = std::min(m, x[i]);
          M = std::max(M, x[i]);
        }
        return {m, M};
      }
      else if constexpr (par == ThreadingModel::PAR) {
        scalart m = std::numeric_limits<scalart>::max(),
          M = std::numeric_limits<scalart>::lowest();
#pragma omp parallel for simd reduction(min:m) reduction(max:M)
        for (int i=0; i<n; i++) {
          m = std::min(m, x[i]);
          M = std::max(M, x[i]);
        }
        return {m, M};
      }
      else if constexpr (par == ThreadingModel::TASK) {
#if _OPENMP >= 201811
        scalart m = std::numeric_limits<scalart>::max(),
          M = std::numeric_limits<scalart>::lowest();
#pragma omp taskloop simd reduction(min:m) reduction(max:M) default(shared)
        for (int i=0; i<n; i++) {
          m = std::min(m, x[i]);
          M = std::max(M, x[i]);
        }
        return {m, M};
#else
#if defined(_OPENMP)
        int B = 5000; // TODO tune task minimum block size!!
        const int nb = std::min(int(std::ceil(float(n) / B)),
                                omp_get_num_threads()*4);
        B = int(std::ceil(float(n) / nb));
#else
        const int nb = 1;
        const int B = n;
#endif
        scalart m = std::numeric_limits<scalart>::max(),
          M = std::numeric_limits<scalart>::lowest();
#pragma omp taskloop grainsize(1) default(shared)
        for (int b=0; b<nb; b++) {
          const auto lb = b*B;
          const auto ub = std::min((b+1)*B, n);
          scalart mt = std::numeric_limits<scalart>::max(),
            Mt = std::numeric_limits<scalart>::lowest();
          for (int i=lb; i<ub; i++) {
            mt = std::min(mt, x[i]);
            Mt = std::max(Mt, x[i]);
          }
#pragma omp critical
          {
            m = std::min(m, mt);
            M = std::max(M, Mt);
          }
        }
        return {m, M};
#endif
      }
    }

    template<typename scalart> std::pair<scalart,scalart>
    minmax(ThreadingModel par, int n, const scalart* x) {
      switch (par) {
      case ThreadingModel::SEQ: return minmax<ThreadingModel::SEQ>(n, x);
      case ThreadingModel::PAR: return minmax<ThreadingModel::PAR>(n, x);
      case ThreadingModel::TASK: return minmax<ThreadingModel::TASK>(n, x);
      case ThreadingModel::LOOP:
      default: assert(true);
      }
      return {0., 0.};
    }

#if defined(PROJECTND_USE_MPI)
    template<ThreadingModel par, typename scalart>
    std::pair<scalart,scalart> minmax
    (int n, const scalart* x, const MPIComm& comm) {
      auto [m, M] = minmax<par>(n, x);
      return {comm.all_reduce(m, MPI_MIN), comm.all_reduce(M, MPI_MAX)};
    }

    template<typename scalart>
    std::pair<scalart,scalart> minmax
    (ThreadingModel par, int n, const scalart* x, const MPIComm& comm) {
      auto [m, M] = minmax(par, n, x);
      return {comm.all_reduce(m, MPI_MIN), comm.all_reduce(M, MPI_MAX)};
    }
#endif

    template<ThreadingModel par, typename scalart>
    void add(int n, scalart* x, scalart alpha) {
      TIMER_TIME(TaskType::ADD, 1, tadd);
      if constexpr (par == ThreadingModel::SEQ) {
        for (int i=0; i<n; i++) x[i] += alpha;
      }
      else if constexpr (par == ThreadingModel::PAR) {
#pragma omp parallel for simd
        for (int i=0; i<n; i++) x[i] += alpha;
      }
      else if constexpr (par == ThreadingModel::LOOP) {
#pragma omp for simd
        for (int i=0; i<n; i++) x[i] += alpha;
      }
      else if constexpr (par == ThreadingModel::TASK) {
#pragma omp taskloop simd default(shared)
        for (int i=0; i<n; i++) x[i] += alpha;
      }
    }

    template<ThreadingModel par, typename scalart>
    void axpy(int n, scalart a, const scalart* x, scalart* y) {
      TIMER_TIME(TaskType::AXPY, 1, taxpy);
      if constexpr (par == ThreadingModel::SEQ) {
        for (int i=0; i<n; i++) y[i] += a * x[i];
      }
      else if constexpr (par == ThreadingModel::PAR) {
#pragma omp parallel for simd
        for (int i=0; i<n; i++) y[i] += a * x[i];
      }
      else if constexpr (par == ThreadingModel::LOOP) {
#pragma omp for simd
        for (int i=0; i<n; i++) y[i] += a * x[i];
      }
      else if constexpr (par == ThreadingModel::TASK) {
#pragma omp taskloop simd default(shared)
        for (int i=0; i<n; i++) y[i] += a * x[i];
      }
    }

    template<ThreadingModel par, typename scalart>
    void xpay(int n, scalart a, const scalart* x, scalart* y) {
      TIMER_TIME(TaskType::AXPY, 1, taxpy);
      if constexpr (par == ThreadingModel::SEQ) {
        for (int i=0; i<n; i++) y[i] = x[i] + a * y[i];
      }
      else if constexpr (par == ThreadingModel::PAR) {
#pragma omp parallel for simd
        for (int i=0; i<n; i++) y[i] = x[i] + a * y[i];
      }
      else if constexpr (par == ThreadingModel::LOOP) {
#pragma omp for simd
        for (int i=0; i<n; i++) y[i] = x[i] + a * y[i];
      }
      else if constexpr (par == ThreadingModel::TASK) {
#pragma omp taskloop simd default(shared)
        for (int i=0; i<n; i++) y[i] = x[i] + a * y[i];
      }
    }

    extern "C" {
#define SSWAP_FC STRUMPACK_FC_GLOBAL(sswap, SSWAP)
#define DSWAP_FC STRUMPACK_FC_GLOBAL(dswap, DSWAP)
      void SSWAP_FC(int* n, float* x, int* ldx, float* y, int* ldy);
      void DSWAP_FC(int* n, double* x, int* ldx, double* y, int* ldy);

#define SSTEV_FC STRUMPACK_FC_GLOBAL(sstev, SSTEV)
#define DSTEV_FC STRUMPACK_FC_GLOBAL(dstev, DSTEV)
      void SSTEV_FC(char* JOBZ, int* N, float* D, float* E,
                    float* Z, int* LDZ, float* WORK, int* INFO);
      void DSTEV_FC(char* JOBZ, int* N, double* D, double* E,
                    double* Z, int* LDZ, double* WORK, int* INFO);

#define SSTEVX_FC STRUMPACK_FC_GLOBAL(sstevx, SSTEVX)
#define DSTEVX_FC STRUMPACK_FC_GLOBAL(dstevx, DSTEVX)
      void SSTEVX_FC(char* JOBZ, char* RANGE, int* N,
                     float* D, float* E, float* VL, float* VU,
                     int* IL, int* IU, float* ABSTOL, int* M,
                     float* W, float* Z, int* LDZ, float* WORK,
                     int* IWORK, int* IFAIL, int* INFO);
      void DSTEVX_FC(char* JOBZ, char* RANGE, int* N,
                     double* D, double* E, double* VL, double* VU,
                     int* IL, int* IU, double* ABSTOL, int* M,
                     double* W, double* Z, int* LDZ, double* WORK,
                     int* IWORK, int* IFAIL, int* INFO);

#define SSYEV_FC STRUMPACK_FC_GLOBAL(ssyev, SSYEV)
#define DSYEV_FC STRUMPACK_FC_GLOBAL(dsyev, DSYEV)
      void SSYEV_FC(char* JOBZ, char* UPLO, int* N, float* D, int* LD,
                    float* W, float* WORK, int* LWORK, int* INFO);
      void DSYEV_FC(char* JOBZ, char* UPLO, int* N, double* D, int* LD,
                    double* W, double* WORK, int* LWORK, int* INFO);

#define SSYEVD_FC STRUMPACK_FC_GLOBAL(ssyevd, SSYEVD)
#define DSYEVD_FC STRUMPACK_FC_GLOBAL(dsyevd, DSYEVD)
      void SSYEVD_FC(char* JOBZ, char* UPLO, int* N, float* A, int* LDA,
                     float* W, float* WORK, int* LWORK,
                     int* IWORK, int* LIWORK, int* INFO);
      void DSYEVD_FC(char* JOBZ, char* UPLO, int* N, double* A, int* LDA,
                     double* W, double* WORK, int* LWORK,
                     int* IWORK, int* LIWORK, int* INFO);

#define SSYEVX_FC STRUMPACK_FC_GLOBAL(ssyevx, SSYEVX)
#define DSYEVX_FC STRUMPACK_FC_GLOBAL(dsyevx, DSYEVX)
      void SSYEVX_FC(char* JOBZ, char* RANGE, char* UPLO, int* N,
                     float* A, int* LDA, float* VL, float* VU,
                     int* IL, int* IU, float* ABSTOL, int* M, float* W,
                     float* Z, int* LDZ, float* WORK, int* LWORK,
                     int* IWORK, int* IFAIL, int* INFO);
      void DSYEVX_FC(char* JOBZ, char* RANGE, char* UPLO, int* N,
                     double* A, int* LDA, double* VL, double* VU,
                     int* IL, int* IU, double* ABSTOL, int* M, double* W,
                     double* Z, int* LDZ, double* WORK, int* LWORK,
                     int* IWORK, int* IFAIL, int* INFO);

#define SSYGV_FC STRUMPACK_FC_GLOBAL(ssygv, SSYGV)
#define DSYGV_FC STRUMPACK_FC_GLOBAL(dsygv, DSYGV)
      void SSYGV_FC(int* ITYPE, char* JOBZ, char* UPLO, int* N,
                    float* A, int* LDA, float* B, int* LDB, float* W,
                    float* WORK, int* LWORK, int* INFO);
      void DSYGV_FC(int* ITYPE, char* JOBZ, char* UPLO, int* N,
                    double* A, int* LDA, double* B, int* LDB, double* W,
                    double* WORK, int* LWORK, int* INFO);

#define SGEMV_FC STRUMPACK_FC_GLOBAL(sgemv, SGEMV)
#define DGEMV_FC STRUMPACK_FC_GLOBAL(dgemv, DGEMV)
      void SGEMV_FC(char* TRANS, int* M, int* N, float* alpha,
                    float *A, int* lda, float* X, int* incx,
                    float* beta, float* Y, int* incy);
      void DGEMV_FC(char* TRANS, int* M, int* N, double* alpha,
                    double *A, int* lda, double* X, int* incx,
                    double* beta, double* Y, int* incy);

#define SGEMM_FC STRUMPACK_FC_GLOBAL(sgemm, SGEMM)
#define DGEMM_FC STRUMPACK_FC_GLOBAL(dgemm, DGEMM)
      void SGEMM_FC(char* TRANSA, char* TRANSB, int* M, int* N, int* K,
                    float* ALPHA, const float* A, int* LDA,
                    const float* B, int* LDB,
                    float* BETA, float* C, int* LDC);
      void DGEMM_FC(char* TRANSA, char* TRANSB, int* M, int* N, int* K,
                    double* ALPHA, const double* A, int* LDA,
                    const double* B, int* LDB,
                    double* BETA, double* C, int* LDC);

#define SLANGE_FC STRUMPACK_FC_GLOBAL(slange, SLANGE)
#define DLANGE_FC STRUMPACK_FC_GLOBAL(dlange, DLANGE)
      float SLANGE_FC(char* NORM, int* M, int* N, float* A, int* LDA,
                      float* WORK);
      double DLANGE_FC(char* NORM, int* M, int* N, double* A, int* LDA,
                       double* WORK);

#define SLACPY_FC STRUMPACK_FC_GLOBAL(slacpy, SLACPY)
#define DLACPY_FC STRUMPACK_FC_GLOBAL(dlacpy, DLACPY)
      int SLACPY_FC(char* uplo, int* m, int* n, const float* a, int* lda,
                    float* b, int* ldb);
      int DLACPY_FC(char* uplo, int* m, int* n, const double* a, int* lda,
                    double* b, int* ldb);

#define SGEQRF_FC STRUMPACK_FC_GLOBAL(sgeqrf, SGEQRF)
#define DGEQRF_FC STRUMPACK_FC_GLOBAL(dgeqrf, DGEQRF)
      void SGEQRF_FC(int* m, int* n, float* a, int* lda, float* tau,
                     float* work, int* lwork, int* info);
      void DGEQRF_FC(int* m, int* n, double* a, int* lda, double* tau,
                     double* work, int* lwork, int* info);

#define SORGQR_FC STRUMPACK_FC_GLOBAL(sorgqr, SORGQR)
#define DORGQR_FC STRUMPACK_FC_GLOBAL(dorgqr, DORGQR)
      void SORGQR_FC(int* m, int* n, int* k, float* a, int* lda,
                     const float* tau, float* work, int* lwork, int* info);
      void DORGQR_FC(int* m, int* n, int* k, double* a, int* lda,
                     const double* tau, double* work, int* lwork, int* info);

#define SGELQF_FC STRUMPACK_FC_GLOBAL(sgelqf, SGELQF)
#define DGELQF_FC STRUMPACK_FC_GLOBAL(dgelqf, DGELQF)
      void SGELQF_FC(int* m, int* n, float* a, int* lda, float* tau,
                     float* work, int* lwork, int* info);
      void DGELQF_FC(int* m, int* n, double* a, int* lda, double* tau,
                     double* work, int* lwork, int* info);

#define SORMLQ_FC STRUMPACK_FC_GLOBAL(sormlq, SORMLQ)
#define DORMLQ_FC STRUMPACK_FC_GLOBAL(dormlq, DORMLQ)
      void SORMLQ_FC(char* side, char* trans, int* m, int *n, int* k,
                     const float* a, int* lda, const float* tau,
                     float* c, int* ldc, float* work, int* lwork, int* info);
      void DORMLQ_FC(char* side, char* trans, int* m, int *n, int* k,
                     const double* a, int* lda, const double* tau,
                     double* c, int* ldc, double* work, int* lwork, int* info);
    }

    inline void swap(int n, float* x, int incx, float* y, int incy) {
      SSWAP_FC(&n, x, &incx, y, &incy);
    }
    inline void swap(int n, double* x, int incx, double* y, int incy) {
      DSWAP_FC(&n, x, &incx, y, &incy);
    }

    inline void swap(int n, float* x, float* y) {
      int one = 1;
      SSWAP_FC(&n, x, &one, y, &one);
    }
    inline void swap(int n, double* x, double* y) {
      int one = 1;
      DSWAP_FC(&n, x, &one, y, &one);
    }

    // default implementation is valid for SEQ, PAR, LOOP
    // except that LOOP will run sequentially
    template<ThreadingModel par=ThreadingModel::SEQ> void
    gemv(char trans, int M, int N, float alpha, float *A, int lda,
         float *X, int incx, float beta, float *Y, int incy) {
      TIMER_TIME(TaskType::GEMV, 1, tgemv);
      SGEMV_FC(&trans, &M, &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy);
    }
    template<ThreadingModel par=ThreadingModel::SEQ> void
    gemv(char trans, int M, int N, double alpha, double *A, int lda,
         double *X, int incx, double beta, double *Y, int incy) {
      TIMER_TIME(TaskType::GEMV, 1, tgemv);
      DGEMV_FC(&trans, &M, &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy);
    }

    /** computes all eigenvalues and, optionally, eigenvectors of a real
        symmetric tridiagonal matrix A. */
    inline int
    stev(char JOBZ, int N, float* D, float* E, float* Z, int LDZ) {
      TIMER_TIME(TaskType::STEV, 1, tstev);
      int INFO;
      auto work = new float[std::max(1,2*N-2)];
      SSTEV_FC(&JOBZ, &N, D, E, Z, &LDZ, work, &INFO);
      delete[] work;
      return INFO;
    }
    inline int
    stev(char JOBZ, int N, double* D, double* E, double* Z, int LDZ) {
      TIMER_TIME(TaskType::STEV, 1, tstev);
      int INFO;
      auto work = new double[std::max(1,2*N-2)];
      DSTEV_FC(&JOBZ, &N, D, E, Z, &LDZ, work, &INFO);
      delete[] work;
      return INFO;
    }

    inline int
    stevx_1(int N, float* D, float* E, float* Z, int LDZ, float& LAMBDA) {
      TIMER_TIME(TaskType::STEVX, 1, tstevx);
      int INFO;
      auto WORK = new float[5*N+N];
      auto W = WORK + 5*N; // new float[N]
      auto IWORK = new int[5*N+N];
      auto IFAIL = IWORK + 5*N;
      int IL = 1;
      int IU = std::min(1, N);
      int M;
      char JOBZ = 'V', RANGE = 'I';
      float ABSTOL = -1.;
      SSTEVX_FC(&JOBZ, &RANGE, &N, D, E, NULL, NULL,
                &IL, &IU, &ABSTOL, &M, W, Z, &LDZ,
                WORK, IWORK, IFAIL, &INFO);
      LAMBDA = W[0];
      delete[] WORK;
      delete[] IWORK;
      return INFO;
    }

    /* work[6*N], iwork[6*N] */
    inline int
    stevx(int N, int IL, int IU, float ABSTOL, float* D, float* E,
          float* Z, int LDZ, float* WORK, int* IWORK) {
      TIMER_TIME(TaskType::STEVX, 1, tstevx);
      int INFO;
      auto W = WORK + 5*N; // new float[N]
      auto IFAIL = IWORK + 5*N;
      int M;
      char JOBZ = 'V', RANGE = 'I';
      SSTEVX_FC(&JOBZ, &RANGE, &N, D, E, NULL, NULL,
                &IL, &IU, &ABSTOL, &M, W, Z, &LDZ,
                WORK, IWORK, IFAIL, &INFO);
      return INFO;
    }

    inline int
    stevx_1(int N, double* D, double* E, double* Z, int LDZ, double& LAMBDA) {
      TIMER_TIME(TaskType::STEVX, 1, tstevx);
      int INFO;
      auto WORK = new double[5*N+N];
      auto W = WORK + 5*N; // new double[N]
      auto IWORK = new int[5*N+N];
      auto IFAIL = IWORK + 5*N;
      int IL = 1;
      int IU = std::min(1, N);
      int M;
      char JOBZ = 'V', RANGE = 'I';
      double ABSTOL = -1.;
      DSTEVX_FC(&JOBZ, &RANGE, &N, D, E, NULL, NULL,
                &IL, &IU, &ABSTOL, &M, W, Z, &LDZ,
                WORK, IWORK, IFAIL, &INFO);
      LAMBDA = W[0];
      delete[] WORK;
      delete[] IWORK;
      return INFO;
    }

    /* work[6*N], iwork[6*N] */
    inline int
    stevx(int N, int IL, int IU, double ABSTOL, double* D, double* E,
          double* Z, int LDZ, double* WORK, int* IWORK) {
      TIMER_TIME(TaskType::STEVX, 1, tstevx);
      int INFO;
      auto W = WORK + 5*N; // new double[N]
      auto IFAIL = IWORK + 5*N;
      int M;
      char JOBZ = 'V', RANGE = 'I';
      DSTEVX_FC(&JOBZ, &RANGE, &N, D, E, NULL, NULL,
                &IL, &IU, &ABSTOL, &M, W, Z, &LDZ,
                WORK, IWORK, IFAIL, &INFO);
      return INFO;
    }


    /**
     * Compute the smallest eigenvalue lambda and corresponding
     * eigenvector Z for a tridiagonal matrix. The tridiagonal matrix is
     * given by alpha (diag) and beta (off-diag). D and E define the
     * Cholesky factor.  This will first try maxit inverse iterations,
     * and if that fails to converge, call stevx. work should be 8*N.
     */
    template <typename scalart>
    int stev_Fiedler_inverse_iteration
    (int N, int maxit, double ABSTOL,  scalart* alpha, scalart* beta,
     scalart* D, scalart* E, scalart* Z, scalart& lambda, scalart* work) {
      scalart tmp1, tmp2;
      scalart* TZ = work;
      for (int l=0; l<maxit; l++) {
        Z[0] = Z[0] / D[0];            // solve with G, T = GG'
        for (int k=1; k<N; k++)
          Z[k] = (Z[k] - E[k-1]*Z[k-1]) / D[k];
        Z[N-1] = Z[N-1] / D[N-1];      // solve with G'
        tmp1 = Z[N-1]*Z[N-1];
        for (int k=N-2; k>=0; k--) {
          Z[k] = (Z[k] - E[k]*Z[k+1]) / D[k];
          tmp1 += Z[k] * Z[k];
        }
        tmp1 = scalart(1.) / std::sqrt(tmp1);
        for (int k=0; k<N; k++) Z[k] *= tmp1;
        tmp1 = tmp2 = 0.;
        for (int k=0; k<N; k++) {      // lambda = (Z'*T*Z) / (Z'*Z)
          TZ[k] = alpha[k] * Z[k];
          if (k > 0) TZ[k] += beta[k-1]*Z[k-1];
          if (k < N-1) TZ[k] += beta[k]*Z[k+1];
          tmp1 += Z[k] * TZ[k];
          tmp2 += Z[k] * Z[k];
        }
        lambda = tmp1 / tmp2;
        tmp1 = 0.;                     // tmp1 = ||T*Z - lambda*Z||
        for (int k=0; k<N; k++) {
          tmp2 = TZ[k] - lambda * Z[k];
          tmp1 += tmp2 * tmp2;
        }
        if (std::sqrt(tmp1) < ABSTOL) return 0;
      }
      // fallback to LAPACK solver
      std::unique_ptr<int[]> iwork_(new int[6*N]);
      auto Tdiag = work + 6*N;
      auto Todiag = Tdiag + N;
      std::copy(alpha, alpha+N, Tdiag);
      std::copy(beta, beta+N, Todiag);
      auto info = stevx
        (N, 1, 1, ABSTOL, Tdiag, Todiag, Z, N, work, iwork_.get());
      lambda = work[5*N];
      return info;
    }

    inline int
    syevx(char JOBZ, char RANGE, char UPLO, int N,
          float* A, int LDA, float VL, float VU,
          int IL, int IU, float ABSTOL, int& M, float* W,
          float* Z, int LDZ) {
      TIMER_TIME(TaskType::SYEVX, 1, tsyevx);
      int INFO;
      auto IWORK = new int[5*N+N];
      auto IFAIL = IWORK + 5*N;
      int LWORK = -1;
      float SWORK;
      SSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU,
                &IL, &IU, &ABSTOL, &M, W, Z, &LDZ,
                &SWORK, &LWORK, IWORK, IFAIL, &INFO);
      LWORK = int(SWORK);
      auto WORK = new float[LWORK];
      SSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU,
                &IL, &IU, &ABSTOL, &M, W, Z, &LDZ,
                WORK, &LWORK, IWORK, IFAIL, &INFO);
      delete[] WORK;
      delete[] IWORK;
      return INFO;
    }
    inline int
    syevx(char JOBZ, char RANGE, char UPLO, int N,
          double* A, int LDA, double VL, double VU,
          int IL, int IU, double ABSTOL, int& M, double* W,
          double* Z, int LDZ) {
      TIMER_TIME(TaskType::SYEVX, 1, tsyevx);
      int INFO;
      auto IWORK = new int[5*N+N];
      auto IFAIL = IWORK + 5*N;
      int LWORK = -1;
      double DWORK;
      DSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU,
                &IL, &IU, &ABSTOL, &M, W, Z, &LDZ,
                &DWORK, &LWORK, IWORK, IFAIL, &INFO);
      LWORK = int(DWORK);
      auto WORK = new double[LWORK];
      DSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU,
                &IL, &IU, &ABSTOL, &M, W, Z, &LDZ,
                WORK, &LWORK, IWORK, IFAIL, &INFO);
      delete[] WORK;
      delete[] IWORK;
      return INFO;
    }


    inline int
    syevx(char JOBZ, char RANGE, char UPLO, int N,
          float* A, int LDA, float VL, float VU,
          int IL, int IU, float ABSTOL, int& M, float* W,
          float* Z, int LDZ, float* WORK, int LWORK,
          int* IWORK, int* IFAIL) {
      TIMER_TIME(TaskType::SYEVX, 1, tsyevx);
      int INFO;
      SSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU,
                &IL, &IU, &ABSTOL, &M, W, Z, &LDZ,
                WORK, &LWORK, IWORK, IFAIL, &INFO);
      return INFO;
    }
    inline int
    syevx(char JOBZ, char RANGE, char UPLO, int N,
          double* A, int LDA, double VL, double VU,
          int IL, int IU, double ABSTOL, int& M, double* W,
          double* Z, int LDZ, double* WORK, int LWORK,
          int* IWORK, int* IFAIL) {
      TIMER_TIME(TaskType::SYEVX, 1, tsyevx);
      int INFO;
      DSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU,
                &IL, &IU, &ABSTOL, &M, W, Z, &LDZ,
                WORK, &LWORK, IWORK, IFAIL, &INFO);
      return INFO;
    }

    // find the i-th smallest eigenvector
    // W should be a vector of size N
    inline int
    syevx_i(int I, float ABSTOL, int N, float* A, int LDA,
            float* W, float* Z, int LDZ) {
      int M;
      int INFO = syevx
        ('V', 'I', 'U', N, A, LDA, 0, 0, std::min(I, N),
         std::min(I, N), ABSTOL, M, W, Z, LDZ);
      if (M != 1)
        std::cout << "# WARNING: syevx only found " << M
                  << " eigenvalues" << std::endl;
      return INFO;
    }
    inline int
    syevx_i(int I, double ABSTOL, int N, double* A, int LDA,
            double* W, double* Z, int LDZ) {
      int M;
      int INFO = syevx
        ('V', 'I', 'U', N, A, LDA, 0, 0, std::min(I, N),
         std::min(I, N), ABSTOL, M, W, Z, LDZ);
      if (M != 1)
        std::cout << "# WARNING: syevx only found " << M
                  << " eigenvalues" << std::endl;
      return INFO;
    }

    /** computes all eigenvalues and, optionally, eigenvectors of a real
        symmetric tridiagonal matrix A. */
    inline int
    syev(char JOBZ, char UPLO, int N, float* D, int LD, float* W) {
      TIMER_TIME(TaskType::SYEV, 1, tsyev);
      int info;
      int lwork = -1;
      float swork;
      SSYEV_FC(&JOBZ, &UPLO, &N, D, &LD, W, &swork, &lwork, &info);
      lwork = int(swork);
      auto work = new float[lwork];
      SSYEV_FC(&JOBZ, &UPLO, &N, D, &LD, W, work, &lwork, &info);
      delete[] work;
      return info;
    }
    inline int
    syev(char JOBZ, char UPLO, int N, double* D, int LD, double* W) {
      TIMER_TIME(TaskType::SYEV, 1, tsyev);
      int info;
      int lwork = -1;
      double dwork;
      DSYEV_FC(&JOBZ, &UPLO, &N, D, &LD, W, &dwork, &lwork, &info);
      lwork = int(dwork);
      auto work = new double[lwork];
      DSYEV_FC(&JOBZ, &UPLO, &N, D, &LD, W, work, &lwork, &info);
      delete[] work;
      return info;
    }

    inline int
    syev(char JOBZ, char UPLO, int N, float* D, int LD, float* W,
         float* work, int lwork) {
      TIMER_TIME(TaskType::SYEV, 1, tsyev);
      int info;
      SSYEV_FC(&JOBZ, &UPLO, &N, D, &LD, W, work, &lwork, &info);
      return info;
    }
    inline int
    syev(char JOBZ, char UPLO, int N, double* D, int LD, double* W,
         double* work, int lwork) {
      TIMER_TIME(TaskType::SYEV, 1, tsyev);
      int info;
      DSYEV_FC(&JOBZ, &UPLO, &N, D, &LD, W, work, &lwork, &info);
      return info;
    }

    /** computes all eigenvalues and, optionally, eigenvectors of a real
        symmetric tridiagonal matrix A. Using a divide-and-conquer
        algorithm. */
    inline int
    syevd(char JOBZ, char UPLO, int N, float* A, int LDA, float* W) {
      TIMER_TIME(TaskType::SYEVD, 1, tsyevd);
      int info;
      int lwork = -1, liwork = -1;
      float swork;
      int siwork;
      SSYEVD_FC(&JOBZ, &UPLO, &N, A, &LDA, W, &swork, &lwork,
                &siwork, &liwork, &info);
      lwork = int(swork);
      liwork = int(siwork);
      auto work = new float[lwork];
      auto iwork = new int[liwork];
      SSYEVD_FC(&JOBZ, &UPLO, &N, A, &LDA, W, work, &lwork,
                iwork, &liwork, &info);
      delete[] iwork;
      delete[] work;
      return info;
    }
    inline int
    syevd(char JOBZ, char UPLO, int N, double* A, int LDA, double* W) {
      TIMER_TIME(TaskType::SYEVD, 1, tsyevd);
      int info;
      int lwork = -1, liwork;
      double dwork;
      DSYEVD_FC(&JOBZ, &UPLO, &N, A, &LDA, W, &dwork, &lwork,
                &liwork, &lwork, &info);
      lwork = int(dwork);
      auto work = new double[lwork];
      auto iwork = new int[liwork];
      DSYEVD_FC(&JOBZ, &UPLO, &N, A, &LDA, W, work, &lwork,
                iwork, &liwork, &info);
      delete[] iwork;
      delete[] work;
      return info;
    }


    template<typename scalart> void
    gemv_task(char trans, int M, int N, scalart alpha, scalart *A, int lda,
              scalart *X, int incx, scalart beta,
              scalart *Y, int incy, int depth) {
      // TODO tune parameters
      const int max_task_depth = 3;
      const int gemv_tile_size = 64;
      if (depth >= max_task_depth ||
          2*double(M)*N <= gemv_tile_size * gemv_tile_size) {
        gemv<ThreadingModel::SEQ>
          (trans, M, N, alpha, A, lda, X, incx, beta, Y, incy);
      } else {
        if (trans=='T' || trans=='t' || trans=='C' || trans=='c') {
          if (N >= M) {
#pragma omp task final(depth >= max_task_depth - 1) mergeable
            gemv_task(trans, M, N/2, alpha, A, lda, X, incx,
                      beta, Y, incy, depth+1);
#pragma omp task final(depth >= max_task_depth-1) mergeable
            gemv_task(trans, M, N-N/2, alpha, A+(N/2)*lda, lda, X, incx,
                      beta, Y+(N/2)*incy, incy, depth+1);
#pragma omp taskwait
          } else {
            if (N <= gemv_tile_size) {
              scalart tmp[gemv_tile_size];
#pragma omp task final(depth >= max_task_depth-1) mergeable
              gemv_task(trans, M/2, N, alpha, A, lda, X, incx,
                        beta, Y, incy, depth+1);
#pragma omp task shared(tmp) final(depth >= max_task_depth-1) mergeable
              gemv_task(trans, M-M/2, N, alpha, A+M/2, lda, X+(M/2)*incx, incx,
                        scalart(0.), tmp, 1, depth+1);
#pragma omp taskwait
              for (int i=0; i<N; i++) Y[i*incy] += tmp[i];
            } else {
              gemv_task(trans, M/2, N, alpha, A, lda, X, incx,
                        beta, Y, incy, depth);
              gemv_task(trans, M-M/2, N, alpha, A+M/2, lda, X+(M/2)*incx, incx,
                        scalart(1.), Y, incy, depth);
            }
          }
        } else if (trans=='N' || trans=='n') {
          if (M >= N) {
#pragma omp task final(depth >= max_task_depth-1) mergeable
            gemv_task(trans, M/2, N, alpha, A, lda, X, incx,
                      beta, Y, incy, depth+1);
#pragma omp task final(depth >= max_task_depth-1) mergeable
            gemv_task(trans, M-M/2, N, alpha, A+M/2, lda, X, incx,
                      beta, Y+(M/2)*incy, incy, depth+1);
#pragma omp taskwait
          } else {
            if (M <= gemv_tile_size) {
              scalart tmp[gemv_tile_size];
#pragma omp task final(depth >= max_task_depth-1) mergeable
              gemv_task(trans, M, N/2, alpha, A, lda, X, incx,
                        beta, Y, incy, depth+1);
#pragma omp task shared(tmp) final(depth >= max_task_depth-1) mergeable
              gemv_task(trans, M, N-N/2, alpha, A+(N/2)*lda, lda, X+(N/2)*incx,
                        incx, scalart(0.), tmp, 1, depth+1);
#pragma omp taskwait
              for (int i=0; i<M; i++) Y[i*incy] += tmp[i];
            } else {
              gemv_task(trans, M, N/2, alpha, A, lda, X, incx,
                        beta, Y, incy, depth);
              gemv_task(trans, M, N-N/2, alpha, A+(N/2)*lda, lda,
                        X+(N/2)*incx, incx, scalart(1.), Y, incy, depth);
            }
          }
        }
      }
    }

    template<> inline void gemv<ThreadingModel::TASK>
    (char trans, int M, int N, float alpha, float *A, int lda,
     float *X, int incx, float beta, float *Y, int incy) {
      gemv_task(trans, M, N, alpha, A, lda, X, incx, beta, Y, incy, 0);
    }
    template<> inline void gemv<ThreadingModel::TASK>
    (char trans, int M, int N, double alpha, double *A, int lda,
     double *X, int incx, double beta, double *Y, int incy) {
      gemv_task(trans, M, N, alpha, A, lda, X, incx, beta, Y, incy, 0);
    }

    inline void
    gemm(char transa, char transb, int M, int N, int K,
         float alpha, const float *A, int lda,
         const float *B, int ldb, float beta, float *C, int ldc)
    { TIMER_TIME(TaskType::GEMM, 1, tgemm);
      SGEMM_FC(&transa, &transb, &M, &N, &K, &alpha, A, &lda,
               B, &ldb, &beta, C, &ldc); }
    inline void
    gemm(char transa, char transb, int M, int N, int K,
         double alpha, const double *A, int lda,
         const double *B, int ldb, double beta, double *C, int ldc)
    { TIMER_TIME(TaskType::GEMM, 1, tgemm);
      DGEMM_FC(&transa, &transb, &M, &N, &K, &alpha, A, &lda,
               B, &ldb, &beta, C, &ldc); }

    inline float lange(char NORM, int M, int N, float *A, int LDA) {
      if (NORM == 'I' || NORM == 'i') {
        auto WORK = new float[M];
        auto ret = SLANGE_FC(&NORM, &M, &N, A, &LDA, WORK);
        delete[] WORK;
        return ret;
      } else
        return SLANGE_FC(&NORM, &M, &N, A, &LDA, NULL);
    }
    inline double lange(char NORM, int M, int N, double *A, int LDA) {
      if (NORM == 'I' || NORM == 'i') {
        auto WORK = new double[M];
        auto ret = DLANGE_FC(&NORM, &M, &N, A, &LDA, WORK);
        delete[] WORK;
        return ret;
      } else
        return DLANGE_FC(&NORM, &M, &N, A, &LDA, NULL);
    }

    inline void lacpy
    (char ul, int m, int n, const float* a, int lda, float* b, int ldb) {
      SLACPY_FC(&ul, &m, &n, a, &lda, b, &ldb);
    }
    inline void lacpy
    (char ul, int m, int n, const double* a, int lda, double* b, int ldb) {
      DLACPY_FC(&ul, &m, &n, a, &lda, b, &ldb);
    }

    inline int geqrf
    (int m, int n, float* a, int lda, float* tau,
     float* work, int lwork) {
      int info;
      SGEQRF_FC(&m, &n, a, &lda, tau, work, &lwork, &info);
      return info;
    }
    inline int geqrf
    (int m, int n, double* a, int lda, double* tau,
     double* work, int lwork) {
      int info;
      DGEQRF_FC(&m, &n, a, &lda, tau, work, &lwork, &info);
      return info;
    }

    inline int orgqr
    (int m, int n, int k, float* a, int lda, const float* tau,
     float* work, int lwork) {
      int info;
      SORGQR_FC(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
      return info;
    }
    inline int orgqr
    (int m, int n, int k, double* a, int lda, const double* tau,
     double* work, int lwork) {
      int info;
      DORGQR_FC(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
      return info;
    }

    inline int gelqf
    (int m, int n, float* a, int lda, float* tau,
     float* work, int lwork) {
      int info;
      SGELQF_FC(&m, &n, a, &lda, tau, work, &lwork, &info);
      return info;
    }
    inline int gelqf
    (int m, int n, double* a, int lda, double* tau,
     double* work, int lwork) {
      int info;
      DGELQF_FC(&m, &n, a, &lda, tau, work, &lwork, &info);
      return info;
    }

    inline int ormlq
    (char side, char trans, int m, int n, int k,
     const float* a, int lda, const float* tau,
     float* c, int ldc, float* work, int lwork) {
      int info;
      SORMLQ_FC(&side, &trans, &m, &n, &k, a, &lda, tau,
                c, &ldc, work, &lwork, &info);
      return info;
    }
    inline int ormlq
    (char side, char trans, int m, int n, int k,
     const double* a, int lda, const double* tau,
     double* c, int ldc, double* work, int lwork) {
      int info;
      DORMLQ_FC(&side, &trans, &m, &n, &k, a, &lda, tau,
                c, &ldc, work, &lwork, &info);
      return info;
    }

  } // end namespace ordering
} // end namespace strumpack

#endif // STRUMPACK_ORDERING_BLASLAPACKWRAPPER_HPP
