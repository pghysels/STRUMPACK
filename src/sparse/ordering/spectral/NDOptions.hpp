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
#ifndef STRUMPACK_ORDERING_NDOPTIONS_HPP
#define STRUMPACK_ORDERING_NDOPTIONS_HPP

#include "StrumpackParameters.hpp"
#if defined(_OPENMP)
#include <omp.h>
#endif
// TODO make getopt optional!!
#include <getopt.h>
#include "sparse/ordering/Graph.hpp"

namespace strumpack {
  namespace ordering {

    enum class Ordering
      { NATURAL, RCM, AMD, MMD, MLF, AND, METIS, SCOTCH,
        SPECTRAL, PARMETIS, PTSCOTCH };
    enum class FiedlerSolver
      { AUTO, LANCZOS, IMPLICIT_LANCZOS, CA_LANCZOS, PIPE_LANCZOS,
        SLEPC, S_STEP, LOBPCG, LOBPCG_STABLE };
    enum class CoarseSolver
      { SYEVX, LANCZOS_LINEAR, LANCZOS_RANDOM };
    enum class FiedlerCut
      { MEDIAN, MIDWAY, AVERAGE, ZERO, OPTIMAL, FAST };
    enum class Precision
      { SINGLE, DOUBLE };

    inline std::string name(const Ordering& o) {
      switch (o) {
      case Ordering::NATURAL:  return "natural";
      case Ordering::RCM:      return "rcm";
      case Ordering::AMD:      return "amd";
      case Ordering::MMD:      return "mmd";
      case Ordering::MLF:      return "mlf";
      case Ordering::AND:      return "and";
      case Ordering::METIS:    return "metis";
      case Ordering::SCOTCH:   return "scotch";
      case Ordering::SPECTRAL: return "spectral";
      case Ordering::PARMETIS: return "parmetis";
      case Ordering::PTSCOTCH: return "ptscotch";
      default: return "ordering-not-recognized";
      }
    }

    inline std::string name(const FiedlerSolver& l) {
      switch (l) {
      case FiedlerSolver::AUTO:              return "auto";
      case FiedlerSolver::LANCZOS:           return "Lanczos";
      case FiedlerSolver::IMPLICIT_LANCZOS:  return "implicit-Lanczos";
      case FiedlerSolver::CA_LANCZOS:        return "ca-Lanczos";
      case FiedlerSolver::PIPE_LANCZOS:      return "pipe-Lanczos";
      case FiedlerSolver::SLEPC:             return "slepc";
      case FiedlerSolver::S_STEP:            return "s-step";
      case FiedlerSolver::LOBPCG:            return "lobpcg";
      case FiedlerSolver::LOBPCG_STABLE:     return "lobpcg-stable";
      default: return "solver-not-recognized";
      }
    }

    inline std::string name(const CoarseSolver& s) {
      switch (s) {
      case CoarseSolver::SYEVX:          return "syevx";
      case CoarseSolver::LANCZOS_LINEAR: return "linear";
      case CoarseSolver::LANCZOS_RANDOM: return "random";
      default: return "coarse-solver-not-recognized";
      }
    }

    inline std::string name(const FiedlerCut& s) {
      switch (s) {
      case FiedlerCut::MEDIAN:  return "median";
      case FiedlerCut::MIDWAY:  return "midway";
      case FiedlerCut::AVERAGE: return "average";
      case FiedlerCut::ZERO:    return "zero";
      case FiedlerCut::OPTIMAL: return "optimal";
      case FiedlerCut::FAST:    return "fast";
      default: return "Fiedler-cut-not-recognized";
      }
    }

    inline std::string name(const Precision& p) {
      switch (p) {
      case Precision::SINGLE: return "single";
      case Precision::DOUBLE: return "double";
      default: return "precision-not-recognized";
      }
    }

    class NDOptions {
    public:
      NDOptions() {}
      NDOptions(int argc, char* argv[]) : argc_(argc), argv_(argv) {}

      void set_verbose(bool v) { verbose_ = v; }
      void set_Fiedler_solver(FiedlerSolver s) { Fiedler_solver_ = s; }
      void set_Fiedler_restart(int restart)
      { assert(restart >= 0); Fiedler_restart_ = restart; }
      void set_Fiedler_subspace(int subspace)
      { assert(subspace >= 0); Fiedler_subspace_ = subspace; }
      void set_Fiedler_maxit(int maxit)
      { assert(maxit >= 0); Fiedler_maxit_ = maxit; }
      void set_Fiedler_tol(double tol)
      { assert(tol <= 1. && tol >= 0.); Fiedler_tol_ = tol; }
      void set_Fiedler_reorthogonalize(bool reorth)
      { Fiedler_reorthogonalize_ = reorth; }
      void set_Fiedler_monitor_residual(bool monitor)
      { Fiedler_monitor_residual_ = monitor; }
      void set_Fiedler_monitor_orthogonality(bool monitor)
      { Fiedler_monitor_orthogonality_ = monitor; }
      void set_Fiedler_monitor_true_residual(bool monitor)
      { Fiedler_monitor_true_residual_ = monitor; }
      void set_Fiedler_print_convergence(bool print)
      { Fiedler_print_convergence_ = print; }
      void set_Lanczos_skip(int skip)
      { assert(skip >= 1); Lanczos_skip_ = skip; }
      void set_smoothing_steps(int s)
      { assert(s >= 0); smoothing_steps_ = s; }
      void set_Fiedler_cut(FiedlerCut c) { Fiedler_cut_ = c; }
      void set_max_imbalance(int ratio) { max_imbalance_ = ratio; }
      void set_interpolation(Interpolation interp)
      { interpolation_ = interp; }
      void set_multilevel_cutoff(int c)
      { assert(c >= 0); multilevel_cutoff_ = c; }
      void set_coarse_solver(CoarseSolver s) { coarse_solver_ = s; }
      void set_dissection_cutoff(int c)
      { assert(c >= 0); dissection_cutoff_ = c; }
      void set_sub_ordering(Ordering o) { sub_ordering_ = o; }
      void enable_weighted_coarsening() { weighted_ = true; }
      void disable_weighted_coarsening() { weighted_ = false; }
      void enable_normalized_Laplacian() { normalized_ = true; }
      void disable_normalized_Laplacian() { normalized_ = false; }
      void enable_SLEPc_matrix_free() { SLEPc_mf_ = true; }
      void disable_SLEPc_matrix_free() { SLEPc_mf_ = false; }
      void set_edge_to_vertex(EdgeToVertex e2v) { edge_to_vertex_ = e2v; }
      void set_precision(Precision prec) { precision_ = prec; }
      void enable_gpu() { gpu_ = true; }
      void disable_gpu() { gpu_ = false; }
      void set_gpu_threshold(int t) { gpu_threshold_ = t; }

      bool verbose() const { return verbose_; }
      FiedlerSolver Fiedler_solver() const { return Fiedler_solver_; }
      int Fiedler_restart() const { return Fiedler_restart_; }
      int Fiedler_subspace() const { return Fiedler_subspace_; }
      int Fiedler_maxit() const { return Fiedler_maxit_; }
      double Fiedler_tol() const { return Fiedler_tol_; }
      bool Fiedler_reorthogonalize() const
      { return Fiedler_reorthogonalize_; }
      bool Fiedler_monitor_residual() const
      { return Fiedler_monitor_residual_; }
      bool Fiedler_monitor_orthogonality() const
      { return Fiedler_monitor_orthogonality_; }
      bool Fiedler_monitor_true_residual() const
      { return Fiedler_monitor_true_residual_; }
      bool Fiedler_print_convergence() const
      { return Fiedler_print_convergence_; }
      int Lanczos_skip() const { return Lanczos_skip_; }
      int smoothing_steps() const { return smoothing_steps_; }
      FiedlerCut Fiedler_cut() const { return Fiedler_cut_; }
      int max_imbalance() const { return max_imbalance_; }
      Interpolation interpolation() const { return interpolation_; }
      int multilevel_cutoff() const { return multilevel_cutoff_; }
      CoarseSolver coarse_solver() const { return coarse_solver_; }
      int dissection_cutoff() const { return dissection_cutoff_; }
      Ordering sub_ordering() const { return sub_ordering_; }
      bool weighted_coarsening() const { return weighted_; }
      bool normalized_Laplacian() const { return normalized_; }
      bool SLEPc_matrix_free() const { return SLEPc_mf_; }
      EdgeToVertex edge_to_vertex() const { return edge_to_vertex_; }
      Precision precision() const { return precision_; }
      bool use_gpu() const { return gpu_; }
      int gpu_threshold() const { return gpu_threshold_; }

      int max_task_lvl() const { return max_task_lvl_; }
      int max_threads() const { return max_threads_; }

      void set_from_command_line() { set_from_command_line(argc_, argv_); }
      void set_from_command_line(int argc, const char* const* cargv) {
#if defined(STRUMPACK_USE_GETOPT)
        std::vector<std::unique_ptr<char[]>> argv_data(argc);
        std::vector<char*> argv(argc);
        for (int i=0; i<argc; i++) {
          argv_data[i].reset(new char[strlen(cargv[i])+1]);
          argv[i] = argv_data[i].get();
          strcpy(argv[i], cargv[i]);
        }
        option long_options[] = {
          {"nd_Fiedler_solver",                required_argument, nullptr, 1},
          {"nd_Fiedler_restart",               required_argument, nullptr, 2},
          {"nd_Fiedler_subspace",              required_argument, nullptr, 3},
          {"nd_Fiedler_maxit",                 required_argument, nullptr, 4},
          {"nd_Fiedler_tol",                   required_argument, nullptr, 5},
          {"nd_Fiedler_reorthogonalize",       no_argument, nullptr, 6},
          {"nd_Fiedler_monitor_residual",      no_argument, nullptr, 7},
          {"nd_Fiedler_monitor_orthogonality", no_argument, nullptr, 8},
          {"nd_Fiedler_monitor_true_residual", no_argument, nullptr, 9},
          {"nd_Fiedler_print_convergence",     no_argument, nullptr, 10},
          {"nd_Lanczos_skip",                  required_argument, nullptr, 11},
          {"nd_smoothing_steps",               required_argument, nullptr, 12},
          {"nd_Fiedler_cut",                   required_argument, nullptr, 13},
          {"nd_max_imbalance",                 required_argument, nullptr, 14},
          {"nd_interpolation",                 required_argument, nullptr, 15},
          {"nd_multilevel_cutoff",             required_argument, nullptr, 16},
          {"nd_coarse_solver",                 required_argument, nullptr, 17},
          {"nd_dissection_cutoff",             required_argument, nullptr, 18},
          {"nd_sub_ordering",                  required_argument, nullptr, 19},
          {"nd_enable_weighted_coarsening",    no_argument, nullptr, 20},
          {"nd_disable_weighted_coarsening",   no_argument, nullptr, 21},
          {"nd_enable_normalized_Laplacian",   no_argument, nullptr, 22},
          {"nd_disable_normalized_Laplacian",  no_argument, nullptr, 23},
          {"nd_enable_SLEPc_matrix_free",      no_argument, nullptr, 24},
          {"nd_disable_SLEPc_matrix_free",     no_argument, nullptr, 25},
          {"nd_edge_to_vertex",                required_argument, nullptr, 26},
          {"nd_precision",                     required_argument, nullptr, 27},
          {"nd_enable_gpu",                    no_argument, nullptr, 28},
          {"nd_disable_gpu",                   no_argument, nullptr, 29},
          {"nd_gpu_threshold",                 required_argument, nullptr, 30},
          {"nd_verbose",                       no_argument, nullptr, 'v'},
          {"nd_quiet",                         no_argument, nullptr, 'q'},
          {"help",                             no_argument, nullptr, 'h'},
          {NULL, 0, NULL, 0}
        };
        int c, option_index = 0;
        // bool unrecognized_options = false;
        opterr = 0;
        while
          ((c = getopt_long_only
            (argc, argv.data(), "hvq", long_options, &option_index)) != -1) {
          switch (c) {
          case 1: {
            std::string s; std::istringstream iss(optarg); iss >> s;
            if (s == name(FiedlerSolver::AUTO))
              set_Fiedler_solver(FiedlerSolver::AUTO);
            else if (s == name(FiedlerSolver::LANCZOS))
              set_Fiedler_solver(FiedlerSolver::LANCZOS);
            else if (s == name(FiedlerSolver::IMPLICIT_LANCZOS))
              set_Fiedler_solver(FiedlerSolver::IMPLICIT_LANCZOS);
            else if (s == name(FiedlerSolver::CA_LANCZOS))
              set_Fiedler_solver(FiedlerSolver::CA_LANCZOS);
            else if (s == name(FiedlerSolver::PIPE_LANCZOS))
              set_Fiedler_solver(FiedlerSolver::PIPE_LANCZOS);
            else if (s == name(FiedlerSolver::SLEPC))
              set_Fiedler_solver(FiedlerSolver::SLEPC);
            else if (s == name(FiedlerSolver::S_STEP))
              set_Fiedler_solver(FiedlerSolver::S_STEP);
            else if (s == name(FiedlerSolver::LOBPCG))
              set_Fiedler_solver(FiedlerSolver::LOBPCG);
            else if (s == name(FiedlerSolver::LOBPCG_STABLE))
              set_Fiedler_solver(FiedlerSolver::LOBPCG_STABLE);
            else std::cerr << "# WARNING: Fiedler solver \"" << s
                           << "\" not recognized, using default" << std::endl;
          } break;
          case 2: {
            std::istringstream iss(optarg); iss >> Fiedler_restart_;
            set_Fiedler_restart(Fiedler_restart_);
          } break;
          case 3: {
            std::istringstream iss(optarg); iss >> Fiedler_subspace_;
            set_Fiedler_subspace(Fiedler_subspace_);
          } break;
          case 4: {
            std::istringstream iss(optarg); iss >> Fiedler_maxit_;
            set_Fiedler_maxit(Fiedler_maxit_);
          } break;
          case 5: {
            std::istringstream iss(optarg); iss >> Fiedler_tol_;
            set_Fiedler_tol(Fiedler_tol_);
          } break;
          case 6: set_Fiedler_reorthogonalize(true); break;
          case 7: set_Fiedler_monitor_residual(true); break;
          case 8: set_Fiedler_monitor_orthogonality(true); break;
          case 9: set_Fiedler_monitor_true_residual(true); break;
          case 10: set_Fiedler_print_convergence(true); break;
          case 11: {
            std::istringstream iss(optarg); iss >> Lanczos_skip_;
            set_Lanczos_skip(Lanczos_skip_);
          } break;
          case 12: {
            std::istringstream iss(optarg); iss >> smoothing_steps_;
            set_smoothing_steps(smoothing_steps_);
          } break;
          case 13: {
            std::string s; std::istringstream iss(optarg); iss >> s;
            if (s == "median") set_Fiedler_cut(FiedlerCut::MEDIAN);
            else if (s == "midway") set_Fiedler_cut(FiedlerCut::MIDWAY);
            else if (s == "average") set_Fiedler_cut(FiedlerCut::AVERAGE);
            else if (s == "zero") set_Fiedler_cut(FiedlerCut::ZERO);
            else if (s == "optimal") set_Fiedler_cut(FiedlerCut::OPTIMAL);
            else if (s == "fast") set_Fiedler_cut(FiedlerCut::FAST);
            else std::cerr << "# WARNING: FiedlerCut \"" << s
                           << "\" not recognized, using default" << std::endl;
          } break;
          case 14: {
            std::istringstream iss(optarg); iss >> max_imbalance_;
            set_max_imbalance(max_imbalance_); } break;
          case 15: {
            std::string s; std::istringstream iss(optarg); iss >> s;
            if (s == "constant") set_interpolation(Interpolation::CONSTANT);
            else if (s == "average") set_interpolation(Interpolation::AVERAGE);
            else std::cerr << "# WARNING: Interpolation \"" << s
                           << "\" not recognized, using default" << std::endl;
          } break;
          case 16: {
            std::istringstream iss(optarg); iss >> multilevel_cutoff_;
            set_multilevel_cutoff(multilevel_cutoff_); } break;
          case 17: {
            std::string s; std::istringstream iss(optarg); iss >> s;
            if (s == "syevx") set_coarse_solver(CoarseSolver::SYEVX);
            else if (s == "linear") set_coarse_solver(CoarseSolver::LANCZOS_LINEAR);
            else if (s == "random") set_coarse_solver(CoarseSolver::LANCZOS_RANDOM);
            else std::cerr << "# WARNING: CoarseSolver \"" << s
                           << "\" not recognized, using default" << std::endl;
          } break;
          case 18: { std::istringstream iss(optarg); iss >> dissection_cutoff_;
              set_dissection_cutoff(dissection_cutoff_); } break;
          case 19: {
            std::string s; std::istringstream iss(optarg); iss >> s;
            if (s == "natural") set_sub_ordering(Ordering::NATURAL);
            else if (s == "rcm") set_sub_ordering(Ordering::RCM);
            else if (s == "amd") set_sub_ordering(Ordering::AMD);
            else if (s == "mmd") set_sub_ordering(Ordering::MMD);
            else if (s == "mlf") set_sub_ordering(Ordering::MLF);
            else if (s == "and") set_sub_ordering(Ordering::AND);
            else if (s == "metis") set_sub_ordering(Ordering::METIS);
            else if (s == "scotch") set_sub_ordering(Ordering::SCOTCH);
            else if (s == "spectral") set_sub_ordering(Ordering::SPECTRAL);
            else if (s == "parmetis") set_sub_ordering(Ordering::PARMETIS);
            else if (s == "ptscotch") set_sub_ordering(Ordering::PTSCOTCH);
            else std::cerr << "# WARNING: Ordering \"" << s
                           << "\" not recognized, using default" << std::endl;
          } break;
          case 20: enable_weighted_coarsening(); break;
          case 21: disable_weighted_coarsening(); break;
          case 22: enable_normalized_Laplacian(); break;
          case 23: disable_normalized_Laplacian(); break;
          case 24: enable_SLEPc_matrix_free(); break;
          case 25: disable_SLEPc_matrix_free(); break;
          case 26: {
            std::string s; std::istringstream iss(optarg); iss >> s;
            if (s == "greedy") set_edge_to_vertex(EdgeToVertex::GREEDY);
            else if (s == "onesided") set_edge_to_vertex(EdgeToVertex::ONESIDED);
            else std::cerr << "# WARNING: EdgeToVertex \"" << s
                           << "\" not recognized, using default" << std::endl;
          } break;
          case 27: {
            std::string s; std::istringstream iss(optarg); iss >> s;
            if (s == "single") set_precision(Precision::SINGLE);
            else if (s == "double") set_precision(Precision::DOUBLE);
            else std::cerr << "# WARNING: Precision \"" << s
                           << "\" not recognized, using default" << std::endl;
          } break;
          case 28: enable_gpu(); break;
          case 29: disable_gpu(); break;
          case 30: { std::istringstream iss(optarg);
              iss >> gpu_threshold_;
              set_gpu_threshold(gpu_threshold_);
          } break;
          case 'h': describe_options(); break;
          case 'v': set_verbose(true); break;
          case 'q': set_verbose(false); break;
            // case '?': unrecognized_options = true; break;
          default: break;
          }
        }
        //if (unrecognized_options && mpi_root())
        // if (unrecognized_options)
        //   std::cerr << "# WARNING ProjectND: There were unrecognized options."
        //             << std::endl;
#else
        std::cerr << "WARNING: no support for getopt.h, "
          "not parsing command line options." << std::endl;
#endif
      }

      void describe_options() const {
#if defined(PROJECTND_USE_MPI)
        MPIComm c;
        if (!c.is_root()) return;
#endif
        std::cout << "# Spectral ND options:" << std::endl;
        std::cout << "#   --nd_Fiedler_solver [auto|Lanczos|implicit-Lanczos|"
                  << "ca-Lanczos|pipe-Lanczos|slepc|lobpcg|lobpcg-multi] (default "
                  << name(Fiedler_solver()) << ")" << std::endl;
        std::cout << "#   --nd_Fiedler_restart int (default "
                  << Fiedler_restart() << ")" << std::endl;
        std::cout << "#   --nd_Fiedler_subspace int (default "
                  << Fiedler_subspace() << ")" << std::endl;
        std::cout << "#   --nd_Fiedler_maxit int (default "
                  << Fiedler_maxit() << ")" << std::endl;
        std::cout << "#   --nd_Fiedler_tol realt (default "
                  << Fiedler_tol() << ")" << std::endl;
        std::cout << "#   --nd_Fiedler_reorthogonalize (default "
                  << Fiedler_reorthogonalize() << ")" << std::endl;
        std::cout << "#   --nd_Fiedler_monitor_residual (default "
                  << Fiedler_monitor_residual() << ")" << std::endl;
        std::cout << "#   --nd_Fiedler_monitor_orthogonality (default "
                  << Fiedler_monitor_orthogonality() << ")" << std::endl;
        std::cout << "#   --nd_Fiedler_monitor_true_residual (default "
                  << Fiedler_monitor_true_residual() << ")" << std::endl;
        std::cout << "#   --nd_Fiedler_print_convergence (default "
                  << Fiedler_print_convergence() << ")" << std::endl;
        std::cout << "#   --nd_Lanczos_skip (default "
                  << Lanczos_skip() << ")" << std::endl;
        std::cout << "#   --nd_smoothing_steps (default "
                  << smoothing_steps() << ")" << std::endl;
        std::cout << "#   --nd_Fiedler_cut [median|midway|average|zero|optimal|fast]"
                  << " (default " << name(Fiedler_cut()) << ")" << std::endl;
        std::cout << "#   --nd_max_imbalance realt [1,..) "
                  << "(default " << max_imbalance() << ")" << std::endl;
        std::cout << "#   --nd_interpolation [constant|average]"
                  << " (default " << name(interpolation()) << ")" << std::endl;
        std::cout << "#   --nd_multilevel_cutoff (default "
                  << multilevel_cutoff() << ")" << std::endl;
        std::cout << "#   --nd_coarse_solver [syev|syevx|linear|random]"
                  << " (default " << name(coarse_solver()) << ")" << std::endl;
        std::cout << "#   --nd_dissection_cutoff (default "
                  << dissection_cutoff() << ")" << std::endl;
        std::cout << "#   --nd_sub_ordering [natural|rcm|amd|mmd|mlf|and|"
                  << "metis|scotch|spectral|parmetis|ptscotch]"
                  << " (default " << name(sub_ordering()) << ")"
                  << std::endl;
        // std::cout << "#   --nd_enable_weighted_coarsening (default "
        //           << std::boolalpha << _weighted << ")" << std::endl;
        // std::cout << "#   --nd_disable_weighted_coarsening (default "
        //           << std::boolalpha << (!_weighted) << ")" << std::endl;
        // std::cout << "#   --nd_enable_normalized_Laplacian (default "
        //           << std::boolalpha << _normalized << ")" << std::endl;
        // std::cout << "#   --nd_disable_normalized_Laplacian (default "
        //           << std::boolalpha << (!_normalized) << ")" << std::endl;
        std::cout << "#   --nd_enable_SLEPc_matrix_free (default "
                  << std::boolalpha << SLEPc_mf_ << ")" << std::endl;
        std::cout << "#   --nd_disable_SLEPc_matrix_free (default "
                  << std::boolalpha << (!SLEPc_mf_) << ")" << std::endl;
        std::cout << "#   --nd_edge_to_vertex [greedy|onesided]"
                  << " (default " << name(edge_to_vertex()) << ")" << std::endl;
        std::cout << "#   --nd_precision [single|double]"
                  << " (default " << name(precision()) << ")" << std::endl;
        std::cout << "#   --nd_enable_gpu (default "
                  << std::boolalpha << use_gpu() << ")" << std::endl;
        std::cout << "#   --nd_disable_gpu (default "
                  << std::boolalpha << (!use_gpu()) << ")" << std::endl;
        std::cout << "#   --nd_gpu_threshold (default "
                  << gpu_threshold() << ")" << std::endl;
        std::cout << "#   --nd_verbose or -v (default "
                  << verbose() << ")" << std::endl;
        std::cout << "#   --nd_quiet or -q (default "
                  << !verbose() << ")" << std::endl;
        std::cout << "#   --help or -h" << std::endl;
        std::cout << std::endl;
      }

    private:
      bool verbose_ = false;
      FiedlerSolver Fiedler_solver_ = FiedlerSolver::LANCZOS;
      int Fiedler_restart_ = 50;
      int Fiedler_subspace_ = 10;
      int Fiedler_maxit_ = 500;
      double Fiedler_tol_ = 1e-2;
      bool Fiedler_reorthogonalize_ = false;
      bool Fiedler_monitor_residual_ = false;
      bool Fiedler_monitor_orthogonality_ = false;
      bool Fiedler_monitor_true_residual_ = false;
      bool Fiedler_print_convergence_ = false;
      int Lanczos_skip_ = 1;
      int smoothing_steps_ = 3;
      FiedlerCut Fiedler_cut_ = FiedlerCut::OPTIMAL;
      int max_imbalance_ = 10;
      Interpolation interpolation_ = Interpolation::AVERAGE;
      int multilevel_cutoff_ = 30;
      CoarseSolver coarse_solver_ = CoarseSolver::SYEVX;
      int dissection_cutoff_ = 150;
      Ordering sub_ordering_ = Ordering::MMD;
      bool weighted_ = false;
      bool normalized_ = false;
      bool SLEPc_mf_ = false;
      EdgeToVertex edge_to_vertex_ = EdgeToVertex::GREEDY;
      Precision precision_ = Precision::SINGLE;
// #if _OPENMP >= 201511 // 4.5
//       bool gpu_ = true;
// #else
//       bool gpu_ = false;
// #endif
      bool gpu_ = false;
      int gpu_threshold_ = 5000;

      int argc_ = 0;
      char** argv_ = nullptr;
#if defined(_OPENMP)
      int max_threads_ = omp_get_max_threads();
      int max_task_lvl_ = std::log2(omp_get_max_threads()) + 3;
#else
      int max_threads_ = 1;
      int max_task_lvl_ = 0;
#endif
    };

  } // end namespace ordering
} // end namespace strumpack

#endif // STRUMPACK_ORDERING_NDOPTIONS_HPP
