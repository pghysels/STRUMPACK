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
#ifndef STRUMPACK_ORDERING_SPECTRAL_STATS_HPP
#define STRUMPACK_ORDERING_SPECTRAL_STATS_HPP

#include <iostream>
#include <atomic>


namespace strumpack {
  namespace ordering {

    class SpectralStats {
    public:
      void print() const {
        std::cout << "+==========================================+"
                  << std::endl;
        std::cout << "| Spectral stats                           |"
                  << std::endl;
        std::cout << "|------------------------------------------|"
                  << std::endl;
        std::cout << "|   " << std::setw(8) << Fiedler_solves.load()
                  << " | Fiedler solves              |" << std::endl;
        std::cout << "|   " << std::setw(8) << Fiedler_failed_solves.load()
                  << " | failed Fiedler solves       |" << std::endl;
        std::cout << "|   " << std::setw(8) << Fiedler_its.load()
                  << " | total Fiedler iterations    |" << std::endl;
        std::cout << "|   " << std::setw(8)
                  << std::fixed << std::setprecision(2)
                  << float(Fiedler_its) / Fiedler_solves.load()
                  << " | avg iterations per Fiedler  |" << std::endl;
        std::cout << "|   " << std::setw(8) << Fiedler_maxits
                  << " | maximum Fiedler iterations  |" << std::endl;
        std::cout << "|   " << std::setw(8) << coarse_solves.load()
                  << " | coarse solves               |" << std::endl;
        std::cout << "|   " << std::setw(8) << coarsenings.load()
                  << " | coarsenings                 |" << std::endl;
        std::cout << "|   " << std::setw(8)
                  << std::fixed << std::setprecision(2)
                  << coarsening_factor/coarsenings.load()
                  << " | avg coarsening factor       |" << std::endl;
        std::cout << "|   " << std::setw(8) << unconnected_graphs.load()
                  << " | unconnected graphs          |" << std::endl;
        std::cout << "|   " << std::setw(8) << unconnected_nodes.load()
                  << " | unconnected nodes           |" << std::endl;
        std::cout << "|   " << std::setw(8) << dense_nodes.load()
                  << " | dense nodes                 |" << std::endl;
        std::cout << "|   " << std::setw(8) << empty_subdomains.load()
                  << " | empty subdomains            |" << std::endl;
        std::cout << "|   " << std::setw(8) << levels
                  << " | dissection levels           |" << std::endl;
        std::cout << "+==========================================+"
                  << std::endl;
      }

#if defined(PROJECTND_USE_MPI)
      void print(const MPIComm& c) {
        Fiedler_solves = c.all_reduce(Fiedler_solves.load(), MPI_SUM);
        Fiedler_failed_solves = c.all_reduce(Fiedler_failed_solves.load(), MPI_SUM);
        Fiedler_its = c.all_reduce(Fiedler_its.load(), MPI_SUM);
        Fiedler_maxits = c.all_reduce(Fiedler_maxits, MPI_MAX);
        coarse_solves = c.all_reduce(coarse_solves.load(), MPI_SUM);
        coarsenings = c.all_reduce(coarsenings.load(), MPI_SUM);
        coarsening_factor = c.all_reduce(coarsening_factor, MPI_SUM);
        coarsening_factor /= c.size();
        unconnected_graphs = c.all_reduce(unconnected_graphs.load(), MPI_SUM);
        unconnected_nodes = c.all_reduce(unconnected_nodes.load(), MPI_SUM);
        dense_nodes = c.all_reduce(dense_nodes.load(), MPI_SUM);
        empty_subdomains = c.all_reduce(empty_subdomains.load(), MPI_SUM);
        if (c.is_root()) print();
      }
#endif

      std::atomic_size_t Fiedler_solves{0};
      std::atomic_size_t Fiedler_failed_solves{0};
      std::atomic_size_t Fiedler_its{0};
      int Fiedler_maxits = 0;
      std::atomic_size_t coarse_solves{0};
      std::atomic_size_t coarsenings{0};
      float coarsening_factor = 0.;        // TODO atomic
      std::atomic_size_t unconnected_graphs{0};
      std::atomic_size_t unconnected_nodes{0};
      std::atomic_size_t dense_nodes{0};
      std::atomic_size_t empty_subdomains{0};
      int levels = 0;                      // TODO atomic
    };

  } // end namespace ordering
} // end namespace strumpack

#endif // STRUMPACK_ORDERING_SPECTRAL_STATS_HPP
