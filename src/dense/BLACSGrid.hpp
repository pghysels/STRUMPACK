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
#include "misc/MPIWrapper.hpp"

namespace strumpack {

  class BLACSGrid {
  public:
    BLACSGrid() {}
    BLACSGrid(const MPIComm& comm) : BLACSGrid(comm, comm.size()) { }
    BLACSGrid(MPIComm&& comm) : BLACSGrid(comm, comm.size()) { }
    BLACSGrid(const MPIComm& comm, int P) : comm_(comm), P_(P) { setup(); }
    BLACSGrid(MPIComm&& comm, int P) : comm_(comm), P_(P) { setup(); }
    ~BLACSGrid() {
      if (ctxt_ != -1) scalapack::Cblacs_gridexit(ctxt_);
      if (ctxt_all_ != -1) scalapack::Cblacs_gridexit(ctxt_all_);
      if (ctxt_T_ != -1) scalapack::Cblacs_gridexit(ctxt_T_);
    }
    BLACSGrid(const BLACSGrid& grid) { *this = grid; }
    BLACSGrid(BLACSGrid&& grid) { *this = std::move(grid); }
    BLACSGrid& operator=(const BLACSGrid& grid) {
      //std::cout << "WARNING copying a BLACS grid is expensive!!" << std::endl;
      comm_ = grid.Comm();
      P_ = grid.P();
      setup();
      return *this;
    }
    BLACSGrid& operator=(BLACSGrid&& grid) {
      comm_ = std::move(grid.comm_);
      P_ = grid.P_;
      ctxt_ = grid.ctxt_;
      ctxt_all_ = grid.ctxt_all_;
      ctxt_T_ = grid.ctxt_T_;
      nprows_ = grid.nprows_;
      npcols_ = grid.npcols_;
      prow_ = grid.prow_;
      pcol_ = grid.pcol_;
      // make sure that grid's context is not destroyed in its
      // destructor
      grid.ctxt_ = -1;
      grid.ctxt_all_ = -1;
      grid.ctxt_T_ = -1;
      return *this;
    }

    const MPIComm& Comm() const { return comm_; }
    MPIComm& Comm() { return comm_; }
    int ctxt() const { return ctxt_; }
    int ctxt_all() const { return ctxt_all_; }
    int nprows() const { return nprows_; }
    int npcols() const { return npcols_; }
    int prow() const { return prow_; }
    int pcol() const { return pcol_; }
    int P() const { return P_; }
    int npactives() const { return nprows() * npcols(); }
    bool active() const { return prow_ != -1; }


    static void layout(int procs, int& proc_rows, int& proc_cols) {
      // why floor, why not nearest??
      proc_cols = std::floor(std::sqrt((float)procs));
      proc_rows = procs / proc_cols;
    }

    BLACSGrid transpose() const {
      BLACSGrid g(*this);
      g.transpose_inplace();
      return g;
    }

  private:
    MPIComm comm_;
    int P_ = -1;
    int ctxt_ = -1;
    int ctxt_all_ = -1;
    int ctxt_T_ = -1;
    int nprows_ = -1;
    int npcols_ = -1;
    int prow_ = -1;
    int pcol_ = -1;

    void setup() {
      layout(P_, nprows_, npcols_);
      if (comm_.is_null()) {
        ctxt_ = ctxt_all_ = ctxt_T_ = -1;
        prow_ = pcol_ = -1;
      } else {
        int active_procs = nprows_ * npcols_;
        if (active_procs < P_) {
          auto active_comm = comm_.sub(0, active_procs);
          if (comm_.rank() < active_procs) {
            ctxt_ = scalapack::Csys2blacs_handle(active_comm.comm());
            scalapack::Cblacs_gridinit(&ctxt_, "C", nprows_, npcols_);
            ctxt_T_ = scalapack::Csys2blacs_handle(active_comm.comm());
            scalapack::Cblacs_gridinit(&ctxt_T_, "R", npcols_, nprows_);
          } else ctxt_ = ctxt_T_ = -1;
        } else {
          ctxt_ = scalapack::Csys2blacs_handle(comm_.comm());
          scalapack::Cblacs_gridinit(&ctxt_, "C", nprows_, npcols_);
          ctxt_T_ = scalapack::Csys2blacs_handle(comm_.comm());
          scalapack::Cblacs_gridinit(&ctxt_T_, "R", npcols_, nprows_);
        }
        ctxt_all_ = scalapack::Csys2blacs_handle(comm_.comm());
        scalapack::Cblacs_gridinit(&ctxt_all_, "R", 1, P_);
        int dummy1, dummy2;
        scalapack::Cblacs_gridinfo(ctxt_, &dummy1, &dummy2, &prow_, &pcol_);
      }
    }

    void transpose_inplace() {
      std::swap(ctxt_, ctxt_T_);
      std::swap(nprows_, npcols_);
      std::swap(prow_, pcol_);
    }

    friend std::ostream& operator<<(std::ostream& os, const BLACSGrid* g);
  };

  std::ostream& operator<<(std::ostream& os, const BLACSGrid* g) {
    if (!g) os << "null";
    else
      os << " ctxt[" << g->ctxt()
         << "](" << g->nprows() << "x" << g->npcols() << " +"
         << (g->P()-g->npactives()) << ")ctxtT[" << g->ctxt_T_ << "]";
    return os;
  }

} // end namespace strumpack

#endif // BLACS_GRID_HPP
