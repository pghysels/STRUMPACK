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
 */
#include "StructuredOptions.hpp"
#include "StrumpackConfig.hpp"
#if defined(STRUMPACK_USE_GETOPT)
#include <vector>
#include <sstream>
#include <cstring>
#include <getopt.h>
#endif
#include "misc/Tools.hpp"

namespace strumpack {
  namespace structured {

    template<typename scalar_t> void
    StructuredOptions<scalar_t>::set_from_command_line
    (int argc, const char* const* cargv) {
#if defined(STRUMPACK_USE_GETOPT)
      std::vector<std::unique_ptr<char[]>> argv_data(argc);
      std::vector<char*> argv(argc);
      for (int i=0; i<argc; i++) {
        argv_data[i].reset(new char[strlen(cargv[i])+1]);
        argv[i] = argv_data[i].get();
        strcpy(argv[i], cargv[i]);
      }
      option long_options[] =
        {{"structured_rel_tol",               required_argument, 0, 1},
         {"structured_abs_tol",               required_argument, 0, 2},
         {"structured_leaf_size",             required_argument, 0, 3},
         {"structured_max_rank",              required_argument, 0, 4},
         {"structured_type",                  required_argument, 0, 5},
         {"structured_verbose",               no_argument, 0, 'v'},
         {"structured_quiet",                 no_argument, 0, 'q'},
         {"help",                             no_argument, 0, 'h'},
         {NULL, 0, NULL, 0}};
      int c, option_index = 0;
      opterr = optind = 0;
      while ((c = getopt_long_only
              (argc, argv.data(), "hvq",
               long_options, &option_index)) != -1) {
        switch (c) {
        case 1: {
          std::istringstream iss(optarg);
          iss >> rel_tol_; set_rel_tol(rel_tol_);
        } break;
        case 2: {
          std::istringstream iss(optarg);
          iss >> abs_tol_; set_abs_tol(abs_tol_);
        } break;
        case 3: {
          std::istringstream iss(optarg);
          iss >> leaf_size_;
          set_leaf_size(leaf_size_);
        } break;
        case 4: {
          std::istringstream iss(optarg);
          iss >> max_rank_;
          set_max_rank(max_rank_);
        } break;
        case 5: {
          std::istringstream iss(optarg);
          std::string s; iss >> s;
          for (auto& c : s) c = std::toupper(c);
          if (s == "HSS")            set_type(Type::HSS);
          else if (s == "BLR")       set_type(Type::BLR);
          else if (s == "HODLR")     set_type(Type::HODLR);
          else if (s == "HODBF")     set_type(Type::HODBF);
          else if (s == "BUTTERFLY") set_type(Type::BUTTERFLY);
          else if (s == "LR")        set_type(Type::LR);
          else if (s == "LOSSY")     set_type(Type::LOSSY);
          else if (s == "LOSSLESS")  set_type(Type::LOSSLESS);
          else
            std::cerr << "# WARNING: low-rank algorithm not"
                      << " recognized, use 'RRQR', 'ACA' or 'BACA'."
                      << std::endl;
        } break;
        case 'v': set_verbose(true); break;
        case 'q': set_verbose(false); break;
        case 'h': describe_options(); break;
        }
      }
#else
      std::cerr << "WARNING: no support for getopt.h, "
        "not parsing command line options." << std::endl;
#endif
    }

    template<typename scalar_t> void
    StructuredOptions<scalar_t>::describe_options() const {
#if defined(STRUMPACK_USE_GETOPT)
      if (!mpi_root()) return;
      std::cout << "# Structured Options:" << std::endl
                << "#   --structured_rel_tol real_t (default "
                << rel_tol() << ")" << std::endl
                << "#   --structured_abs_tol real_t (default "
                << abs_tol() << ")" << std::endl
                << "#   --structured_leaf_size int (default "
                << leaf_size() << ")" << std::endl
                << "#   --structured_max_rank int (default "
                << max_rank() << ")" << std::endl
                << "#   --structured_type type (default "
                << get_name(type()) << ")" << std::endl
                << "#   --structured_verbose or -v (default "
                << verbose() << ")" << std::endl
                << "#   --structured_quiet or -q (default "
                << !verbose() << ")" << std::endl
                << "#   --help or -h" << std::endl << std::endl;
#endif
    }

    // explicit template instantiations
    template class StructuredOptions<float>;
    template class StructuredOptions<double>;
    template class StructuredOptions<std::complex<float>>;
    template class StructuredOptions<std::complex<double>>;

  } // end namespace structured
} // end namespace strumpack
