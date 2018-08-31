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
/*!
 * \file BLACSGrid.hpp
 * \brief Contains a wrapper class around a BLACS grid/context.
 */
#ifndef BLACS_GRID_HPP
#define BLACS_GRID_HPP

#include "ScaLAPACKWrapper.hpp"
#include "misc/MPIWrapper.hpp"

namespace strumpack {

  /**
   * \class BLACSGrid
   * \brief This is a small wrapper class around a BLACS grid and a
   * BLACS context.
   *
   * The main purpose of this wrapper is to handle the (global)
   * resource of BLACS grid, ie, creation and destruction, as well as
   * to encapsulate the BLACS interface into a single small piece of
   * code. This will hopefully make it much easier to at one point in
   * the near future get rid of ScaLAPACK and BLACS, and replace it
   * with SLATE.
   *
   * This class also stores (and owns) an MPIComm object. The BLACS
   * grid is a 2D grid, consisting of nprows x npcols processes, with
   * ranks assigned COLUMN MAJOR!!. However, the total number of
   * processes, P, can be larger than nprows x npcols, meaning some
   * processes are included in the MPIComm, but not in the grid, ie,
   * they are idle. Leaving some processes idle can give a grid with a
   * better aspect ratio, compared to using all available ranks,
   * leading to more scalable code.
   *
   * The MPIComm object can also be MPI_COMM_NULL, while P can still
   * be > 0. This is useful for getting info about a grid (the
   * layout), in which this rank is not active, for instance to know
   * to which rank to communicate.
   */
  class BLACSGrid {
  public:
    /**
     * Default constructor. Does not construct any BLACS grid or
     * context.
     */
    BLACSGrid() {}

    /**
     * Construct a BLACSGrid from an MPIComm object. The MPIComm will
     * be duplicated. The BLACS grid will be created. This operation
     * is collective on comm.
     *
     * \param comm MPIComm used to initialize the grid, this will be
     * duplicated
     */
    BLACSGrid(const MPIComm& comm) : BLACSGrid(comm, comm.size()) { }

    /**
     * Construct a BLACSGrid from an MPIComm object. The MPIComm will
     * be moved from (reset to MPI_COMM_NULL). The BLACS grid will be
     * created. This operation is collective on comm.
     *
     * \param comm MPIComm used to initialize the grid, this will be
     * moved (reset to MPI_COMM_NULL)
     */
    BLACSGrid(MPIComm&& comm) : BLACSGrid(comm, comm.size()) { }

    /**
     * Construct a BLACSGrid from an MPIComm object. The MPIComm will
     * be duplicated. The BLACS grid will be created. This operation
     * is collective on comm. comm can be a null communicator
     * (MPI_COMM_NULL), but P can be used to specify the number of
     * processes in the grid (including potentially idle ranks).
     *
     * \param comm MPIComm used to initialize the grid, this will be
     * duplicated
     * \param P total number of ranks in the new grid (P ==
     * comm.size() or comm.is_null())
     */
    BLACSGrid(const MPIComm& comm, int P) : comm_(comm), P_(P) { setup(); }

    /**
     * Construct a BLACSGrid from an MPIComm object. The MPIComm will
     * be moved. The BLACS grid will be created. This operation is
     * collective on comm. comm can be a null communicator
     * (MPI_COMM_NULL), but P can be used to specify the number of
     * processes in the grid (including potentially idle ranks).
     *
     * \param comm MPIComm used to initialize the grid, this will be
     * moved from, and then reset to a null communicator
     * (MPI_COMM_NULL)
     * \param P total number of ranks in the new grid (P ==
     * comm.size() or comm.is_null())
     */
    BLACSGrid(MPIComm&& comm, int P) : comm_(comm), P_(P) { setup(); }

    /**
     * Destructor, this will free all resources associated with this
     * grid.
     */
    ~BLACSGrid() {
      if (ctxt_ != -1) scalapack::Cblacs_gridexit(ctxt_);
      if (ctxt_all_ != -1) scalapack::Cblacs_gridexit(ctxt_all_);
      if (ctxt_T_ != -1) scalapack::Cblacs_gridexit(ctxt_T_);
    }

    /**
     * Copy constructor. This might be expensive, the entire grid will
     * be copied. The resulting grid will be a new grid, with a
     * different BLACS context, and so cannot be used in the same
     * ScaLAPACK call as the original grid. Also the MPIComm object
     * will be copied. This operation is collective on all processes
     * in the MPIComm.
     *
     * \param grid BLACSGrid to copy from
     */
    BLACSGrid(const BLACSGrid& grid) { *this = grid; }

    /**
     * Move constructor.
     */
    BLACSGrid(BLACSGrid&& grid) { *this = std::move(grid); }

    /**
     * Copy assignment. This might be expensive, the entire grid will
     * be copied. The resulting grid will be a new grid, with a
     * different BLACS context, and so cannot be used in the same
     * ScaLAPACK call as the original grid. Also the MPIComm object
     * will be copied. This operation is collective on all processes
     * in the MPIComm.
     *
     * \param grid BLACSGrid to copy from
     * \return the newly created grid
     */
    BLACSGrid& operator=(const BLACSGrid& grid) {
      //std::cout << "WARNING copying a BLACS grid is expensive!!" << std::endl;
      comm_ = grid.Comm();
      P_ = grid.P();
      setup();
      return *this;
    }

    /**
     * Move assignment, does not make a copy.
     *
     * \param grid grid to move from, this will be reset to a
     * non-existing grid (a -1 context).
     */
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

    /**
     * Return a (const) reference to the MPIComm object.
     */
    const MPIComm& Comm() const { return comm_; }

    /**
     * Return a reference to the MPIComm object.
     */
    MPIComm& Comm() { return comm_; }

    /**
     * Return the blacs context, a context to a blacs grid with nprows
     * processor rows and npcols processor columns (ranks are assigned
     * column major). Or -1 for a non-initialized grid or for ranks
     * which are idle in this grid.
     */
    int ctxt() const { return ctxt_; }

    /**
     * Return a BLACS context for a grid including all the ranks, not
     * just those active on ctxt(), but all the ranks in Comm(). The
     * grid is a single row with Comm().size() columns. Will return -1
     * for a non-initialized grid, or initialized with a null
     * communicator.
     *
     * This can be used as the last argument for a call to P*DGEMR2d.
     */
    int ctxt_all() const { return ctxt_all_; }

    /**
     * Number of processor rows in the grid, -1 if this rank is idle
     * in the grid, or if not initialized.
     */
    int nprows() const { return nprows_; }

    /**
     * Number of processor columns in the grid, -1 if this rank is
     * idle in the grid, or if not initialized.
     */
    int npcols() const { return npcols_; }

    /**
     * This ranks processor row in the grid. -1 if this rank is not
     * part of the grid.
     */
    int prow() const { return prow_; }

    /**
     * This ranks processor column in the grid. -1 if this rank is not
     * part of the grid.
     */
    int pcol() const { return pcol_; }

    /**
     * Total number of ranks in this grid, including any ranks that
     * may be idle. P >= nprows()*npcols(). If Comm() is not a null
     * communicator, then P == Comm.size().
     */
    int P() const { return P_; }

    /**
     * Number of processes active in the grid, simply
     * nprows()*npcols().
     */
    int npactives() const { return nprows() * npcols(); }

    /**
     * Checks whether this rank is active on this grid, ie, the grid
     * has been initialized, and this rank is not one of the idle
     * ranks.
     */
    bool active() const { return prow_ != -1; }

    /**
     * For a given number of processes procs, find a 2D layout. This
     * will try to find a 2D layout using P, or as close to P as
     * possible, number of ranks, while making the grid as square as
     * possible. The number of processes in the 2D layout can be less
     * than procs, ie, some ranks will be idle, to improve
     * parallelism. procs >= proc_rows * proc_cols.
     *
     * \param procs maximum number of processes in the 2d layout
     * \param proc_rows output, number of rows in the 2d layout
     * \param proc_cols output, number of columns in the 2d layout
     */
    static void layout(int procs, int& proc_rows, int& proc_cols) {
      // why floor, why not nearest??
      proc_cols = std::floor(std::sqrt((float)procs));
      proc_rows = procs / proc_cols;
    }

    /**
     * Return a BLACSGrid which is the transpose of the current
     * grid. Ie., has npcols processor rows and nprows processor
     * columns. This can be used to perform operations on the
     * transpose of a distributed matrix. Instead of transposing the
     * matrix, which requires communication, one can transpose the
     * BLACS grid and only transpose the local storage.
     */
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


  /**
   * Print some info about the BLACS grid to stream os. Just used for
   * debugging.
   */
  inline std::ostream& operator<<(std::ostream& os, const BLACSGrid* g) {
    if (!g) os << "null";
    else
      os << " ctxt[" << g->ctxt()
         << "](" << g->nprows() << "x" << g->npcols() << " +"
         << (g->P()-g->npactives()) << ")ctxtT[" << g->ctxt_T_ << "]";
    return os;
  }

} // end namespace strumpack

#endif // BLACS_GRID_HPP
