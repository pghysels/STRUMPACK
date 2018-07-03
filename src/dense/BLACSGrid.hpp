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
#ifndef BLACS_GRID_HPP
#define BLACS_GRID_HPP

#include "ScaLAPACKWrapper.hpp"

namespace strumpack {

  class BLACSGrid {
  public:
    BLACSGrid(MPI_Comm comm) : BLACSGrid(comm, mpi_nprocs(comm)) { }
    BLACSGrid(int P) : BLACSGrid(MPI_COMM_NULL, P) {}
    BLACSGrid(MPI_Comm comm, int P) : P_(P) {
      // why floor, why not nearest??
      npcols_ = std::floor(std::sqrt((float)P_));
      nprows_ = P_ / npcols_;
      if (comm == MPI_COMM_NULL) {
        ctxt_ = ctxt_all_ = -1;
        prow_ = pcol_ = -1;
      } else {
        int active_procs = nprows_ * npcols_;
        if (active_procs < P_) {
          auto active_comm = mpi_sub_comm(comm, 0, active_procs);
          if (mpi_rank(comm) < active_procs) {
            ctxt_ = scalapack::Csys2blacs_handle(active_comm);
            scalapack::Cblacs_gridinit(&ctxt_, "C", nprows_, npcols_);
          } else ctxt_ = -1;
          mpi_free_comm(&active_comm);
        } else {
          ctxt_ = scalapack::Csys2blacs_handle(comm);
          scalapack::Cblacs_gridinit(&ctxt_, "C", nprows_, npcols_);
        }
        ctxt_all_ = scalapack::Csys2blacs_handle(comm);
        scalapack::Cblacs_gridinit(&ctxt_all_, "R", 1, P);
      }
    }
    ~BLACSGrid() {
      if (ctxt_ != -1) scalapack::Cblacs_gridexit(ctxt_);
      if (ctxt_all_ != -1) scalapack::Cblacs_gridexit(ctxt_all_);
    }

    int ctxt() const { return ctxt_; }
    int ctxt_all() const { return ctxt_all_; }
    int nprows() const { return nprows_; }
    int npcols() const { return npcols_; }
    int prow() const { return prow_; }
    int pcol() const { return pcol_; }
    bool active() const { return prow_ != -1; }

  private:
    int P_;
    int ctxt_;
    int ctxt_all_;
    int nprows_;
    int npcols_;
    int prow_;
    int pcol_;
  };

} // end namespace strumpack

#endif // BLACS_GRID_HPP
