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
 * \file DistributedMatrix.hpp \brief Contains the DistributedMatrix
 * and DistributedMatrixWrapper classes, wrappers around
 * ScaLAPACK/PBLAS style 2d block cyclic matrices. See also
 * strumpack::BLACSGrid for the processor grid (BLACS functionality).
 */
#ifndef DISTRIBUTED_MATRIX_HPP
#define DISTRIBUTED_MATRIX_HPP

#include <vector>
#include <string>
#include <cmath>
#include <functional>

#include "misc/MPIWrapper.hpp"
#include "misc/RandomWrapper.hpp"
#include "DenseMatrix.hpp"
#include "BLACSGrid.hpp"

namespace strumpack {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  inline int indxl2g(int INDXLOC, int NB,
                     int IPROC, int ISRCPROC, int NPROCS)  {
    return NPROCS*NB*((INDXLOC-1)/NB) + (INDXLOC-1) % NB +
      ((NPROCS+IPROC-ISRCPROC) % NPROCS)*NB + 1;
  }
  inline int indxg2l(int INDXGLOB, int NB,
                     int IPROC, int ISRCPROC, int NPROCS) {
    return NB*((INDXGLOB-1)/(NB*NPROCS)) + (INDXGLOB-1) % NB + 1;
  }
  inline int indxg2p(int INDXGLOB, int NB,
                     int IPROC, int ISRCPROC, int NPROCS) {
    return ( ISRCPROC + (INDXGLOB - 1) / NB ) % NPROCS;
  }
#endif

  /**
   * \class DistributedMatrix
   *
   * \brief 2D block cyclicly distributed matrix, as used by
   * ScaLAPACK.
   *
   * This class represents a 2D block cyclic matrix like used by
   * ScaLAPACK. To create a submatrix of a DistributedMatrix use a
   * DistributedMatrixWrapper.
   *
   * Several routines in this class are collective on all processes in
   * the process grid. The BLACSGrid passed when constructing the
   * matrix should persist as long as the matrix is in use (in scope).
   *
   * \tparam scalar_t Possible values for the scalar_t template type
   * are: float, double, std::complex<float> and std::complex<double>.
   *
   * \see strumpack::BLACSGrid
   */
  template<typename scalar_t> class DistributedMatrix {
    using real_t = typename RealType<scalar_t>::value_type;

  public:

    /**
     * Default constructor. The grid will not be initialized.
     */
    DistributedMatrix();

    /**
     * Construct a Distributed matrix of size MxN on grid g, using the
     * default blocksize
     * strumpack::DistributedMatrix<scalar_t>::default_MB and
     * strumpack::DistributedMatrix::default_NB. The values of the
     * matrix are not set.
     *
     * \param g BLACS grid.
     * \param M number of (global) columns
     * \param N number of (global) rows
     */
    DistributedMatrix(const BLACSGrid* g, int M, int N);

    /**
     * Construct a Distributed matrix of size MxN on grid g, using the
     * default blocksize.  The values of the matrix are initialized
     * using the specified function.
     *
     * \param g BLACS grid.
     * \param M number of (global) columns
     * \param N number of (global) rows

     * \param A Routine to be used to initialize the matrix, by
     * calling this->fill(A).
     */
    DistributedMatrix(const BLACSGrid* g, int M, int N,
                      const std::function<scalar_t(std::size_t,
                                                   std::size_t)>& A);

    /**
     * Create a DistributedMatrix from a DenseMatrix. This only works
     * on a BLACS grid with a single process. Values will be copied
     * from the input DenseMatrix into the new DistributedMatrix.
     *
     * \param g BLACS grid, g->nprows() == 1, g->npcols() == 1.
     * \param m input matrix.
     *
     */
    DistributedMatrix(const BLACSGrid* g, const DenseMatrix<scalar_t>& m);

    /**
     * Create a DistributedMatrix by moving from a DenseMatrix. This
     * only works on a BLACS grid with a single process. The input
     * DenseMatrix will be cleared.
     *
     * \param g BLACS grid, g->nprows() == 1, g->npcols() == 1.
     * \param m input matrix, cleared.
     */
    DistributedMatrix(const BLACSGrid* g, DenseMatrix<scalar_t>&& m);

    /**
     * Create a DistributedMatrix by moving from a
     * DenseMatrixWrapper. This only works on a BLACS grid with a
     * single process. Values will be copied from the input
     * DenseMatrix(Wrapper) into the new DistributedMatrix.
     *
     * \param g BLACS grid, g->nprows() == 1, g->npcols() == 1.
     * \param m input matrix.
     */
    DistributedMatrix(const BLACSGrid* g, DenseMatrixWrapper<scalar_t>&& m);

    /**
     * Construct a new DistributedMatrix as a copy of another. Sizes
     * need to be specified since it can be that not all processes
     * involved know the sizes of the source matrix. The input matrix
     * might be on a different BLACSGrid.
     *
     * \param g BLACS grid of newly to construct matrix
     * \param M (global) number of rows of m, and of new matrix
     * \param N (global) number of rows of m, and of new matrix
     * \param context_all BLACS context containing all processes in g
     * and in m.grid(). If g and m->grid() are the same, then this can
     * be g->ctxt_all(). Ideally this is a BLACS context with all
     * processes arranged as a single row, since the routine called
     * from this constructor, P_GEMR2D, is more efficient when this is
     * the case.
     */
    DistributedMatrix(const BLACSGrid* g, int M, int N,
                      const DistributedMatrix<scalar_t>& m,
                      int context_all);

    /**
     * Construct a new MxN (global sizes) DistributedMatrix on
     * processor grid g, using block sizes MB and NB. Matrix values
     * will not be initialized.
     *
     * Note that some operations, such as LU factorization require MB
     * == NB.
     */
    DistributedMatrix(const BLACSGrid* g, int M, int N, int MB, int NB);


    /**
     * Copy constructor.
     */
    DistributedMatrix(const DistributedMatrix<scalar_t>& m);

    /**
     * Move constructor.
     */
    DistributedMatrix(DistributedMatrix<scalar_t>&& m);

    /**
     * Destructor.
     */
    virtual ~DistributedMatrix();

    /**
     * Copy assignment.
     * \param m matrix to copy from
     */
    DistributedMatrix<scalar_t>&
    operator=(const DistributedMatrix<scalar_t>& m);

    /**
     * Moce assignment.
     * \param m matrix to move from, will be cleared.
     */
    DistributedMatrix<scalar_t>&
    operator=(DistributedMatrix<scalar_t>&& m);


    /**
     * Return the descriptor array.
     * \return the array descriptor for the distributed matrix.
     */
    const int* desc() const { return desc_; }

    /**
     * Check whether this rank is active in the grid on which this
     * matrix is defined. Since the 2d grid build from the MPI
     * communicator is made as square as possible, some processes
     * might be idle, i.e., not active.
     * \return grid() && grid()->active()
     */
    bool active() const { return grid() && grid()->active(); }

    /**
     * Get a pointer to the 2d processor grid used.
     * \return 2d processor grid.
     */
    const BLACSGrid* grid() const { return grid_; }

    /**
     * Get the MPIComm (MPI_Comm wrapper) communicator associated with
     * the grid.
     * \return grid()->Comm()
     */
    const MPIComm& Comm() const { return grid()->Comm(); }

    /**
     * Get the MPI communicator associated with the processor grid.
     * \return grid()->comm()
     */
    MPI_Comm comm() const { return Comm().comm(); }

    /**
     * Get the BLACS context from the grid.
     * \return blacs context if grid() != NULL, otherwise -1
     */
    int ctxt() const { return grid() ? grid()->ctxt() : -1; }

    /**
     * Get the global BLACS context from the grid. The global context
     * is the context that includes all MPI ranks from the
     * communicator used to construct the grid.
     * \return blacs context if grid() != NULL, otherwise -1
     */
    int ctxt_all() const { return grid() ? grid()->ctxt_all() : -1; }

    /**
     * Get the number of global rows in the matrix
     * \return Global number of matrix rows
     */
    virtual int rows() const { return desc_[2]; }
    /**
     * Get the number of global rows in the matrix
     * \return Global number of matrix columns
     */
    virtual int cols() const { return desc_[3]; }
    /**
     * Get the number of local rows in the matrix.
     * \return number of local rows stored on this rank.
     */
    int lrows() const { return lrows_; }
    /**
     * Get the number of local columns in the matrix.
     * \return number of local columns stored on this rank.
     */
    int lcols() const { return lcols_; }
    /**
     * Get the leading dimension for the local storage.
     * \return leading dimension of local storage.
     */
    int ld() const { return lrows_; }

    /**
     * Get the row blocksize.
     * \return row blocksize
     */
    int MB() const { return desc_[4]; }
    /**
     * Get the column blocksize.
     * \return column blocksize
     */
    int NB() const { return desc_[5]; }

    /**
     * Return number of block rows stored on this rank
     * \return number of local row blocks.
     */
    int rowblocks() const { return std::ceil(float(lrows()) / MB()); }
    /**
     * Return number of block columns stored on this rank
     * \return number of local column blocks.
     */
    int colblocks() const { return std::ceil(float(lcols()) / NB()); }

    /**
     * Row index of top left element. This is always 1 for a
     * DistributedMatrix. For a DistributedMatrixWrapper, this can be
     * different from to denote the position in the original matrix.
     * \return Row index of top left element (1-based).
     */
    virtual int I() const { return 1; }
    /**
     * Columns index of top left element. This is always 1 for a
     * DistributedMatrix. For a DistributedMatrixWrapper, this can be
     * different from to denote the position in the original matrix.
     * \return Column index of top left element (1-based).
     */
    virtual int J() const { return 1; }

    /**
     * Get the ranges of local rows and columns. For a
     * DistributedMatrix, this will simply be (0, lrows(), 0,
     * lcols()). For a DistributedMatrixWrapper, this returns the local
     * rows/columns of the DistributedMatrix that correspond the the
     * submatrix represented by the wrapper.
     * The values are 0-based
     *
     * \param rlo output parameter, first local row
     * \param rhi output parameter, one more than the last local row
     * \param clo output parameter, first local column
     * \param chi output parameter, one more than the last local column
     */
    virtual void lranges(int& rlo, int& rhi, int& clo, int& chi) const;

    /**
     * Return raw pointer to local storage.
     * \return pointer to first element of local storage.
     */
    const scalar_t* data() const { return data_; }

    /**
     * Return raw pointer to local storage.
     * \return pointer to first element of local storage.
     */
    scalar_t* data() { return data_; }

    /**
     * Access an element using local indexing. This const version is
     * only for reading the value.
     *
     * \param r local row index
     * \param c local column index
     * \return value at local row r, local column c
     * \see rowg2l(), colg2l(), rowl2g(), coll2g()
     */
    const scalar_t& operator()(int r, int c) const { return data_[r+ld()*c]; }

    /**
     * Access an element using local indexing. This can be used to
     * read or write the value.
     *
     * \param r local row index
     * \param c local column index
     * \return value at local row r, local column c
     * \see rowg2l(), colg2l(), rowl2g(), coll2g()
     */
    scalar_t& operator()(int r, int c) { return data_[r+ld()*c]; }

    /**
     * Get the row of the process in the process grid. This requires
     * grid() != NULL.
     * \return grid()->prow()
     */
    int prow() const { assert(grid()); return grid()->prow(); }
    /**
     * Get the column of the process in the process grid. This
     * requires grid() != NULL.
     * \return grid()->pcol()
     */
    int pcol() const { assert(grid()); return grid()->pcol(); }
    /**
     * Get the number of process rows in the process grid. This
     * requires grid() != NULL.
     * \return grid()->nprows()
     */
    int nprows() const { assert(grid()); return grid()->nprows(); }
    /**
     * Get the number of process columns in the process grid. This
     * requires grid() != NULL.
     * \return grid()->npcols()
     */
    int npcols() const { assert(grid()); return grid()->npcols(); }
    /**
     * Get the number active processes in the process grid. This can
     * be less than Comm().size(), but should be
     * nprows()*npcols(). This requires grid() != NULL.
     * \return grid()->npactives()
     */
    int npactives() const {
      assert(grid());
      return grid()->npactives();
    }

    /**
     * Check whether this process is the root/master, i.e., prow() ==
     * 0 && pcol() == 0. This requires grid() != NULL.
     * \return grid() && prow() == 0 && pcol() == 0
     */
    bool is_master() const {
      return grid() && prow() == 0 && pcol() == 0;
    }
    /**
     * Get the global row from the local row. This is 0-based. This
     * requires grid() != NULL.
     * \param row local row, 0 <= row < lrows()
     * \return global row corresponding to row, 0 <= global row < rows().
     */
    int rowl2g(int row) const {
      assert(grid());
      return indxl2g(row+1, MB(), prow(), 0, nprows()) - I();
    }
    /**
     * Get the global columns from the local columns. This is
     * 0-based. This requires grid() != NULL.
     * \param col local col, 0 <= col < lcols()
     * \return global column corresponding to col, 0 <= global column < cols().
     */
    int coll2g(int col) const {
      assert(grid());
      return indxl2g(col+1, NB(), pcol(), 0, npcols()) - J();
    }
    /**
     * Get the local row from the global row. This is 0-based. This
     * requires grid() != NULL.
     * \param row global row, 0 <= row < rows()
     * \return local row corresponding to row, 0 <= local row < lrows().
     */
    int rowg2l(int row) const {
      assert(grid());
      return indxg2l(row+I(), MB(), prow(), 0, nprows()) - 1;
    }
    /**
     * Get the global column from the local column. This is
     * 0-based. This requires grid() != NULL.
     * \param col global col, 0 <= col < cols()
     * \return local col corresponding to col, 0 <= local col < lcols().
     */
    int colg2l(int col) const {
      assert(grid());
      return indxg2l(col+J(), NB(), pcol(), 0, npcols()) - 1;
    }
    /**
     * Get the process row from the global row. This is 0-based. This
     * requires grid() != NULL.
     * \param row global row, 0 <= row < rows()
     * \return process row corresponding to row, 0 <= process row < grid()->nprows().
     */
    int rowg2p(int row) const {
      assert(grid());
      return indxg2p(row+I(), MB(), prow(), 0, nprows());
    }
    /**
     * Get the process column from the global column. This is
     * 0-based. This requires grid() != NULL.
     * \param col global columns, 0 <= col < cols()
     * \return process column corresponding to col, 0 <= process
     * column < grid()->npcols().
     */
    int colg2p(int col) const {
      assert(grid());
      return indxg2p(col+J(), NB(), pcol(), 0, npcols());
    }
    /**
     * Get the MPI rank in the communicator used to construct the grid
     * corresponding to global row,column element. This assumes a
     * column major ordering of the grid, see strumpack::BLACSGrid.
     * \param r global row
     * \param c global column
     * \return rowg2p(r) + colg2p(c) * nprows()
     */
    int rank(int r, int c) const {
      return rowg2p(r) + colg2p(c) * nprows();
    }
    /**
     * Check whether a global element r,c is local to this rank.
     * \param r global row
     * \param c global column
     * \return rowg2p(r) == prow() && colg2p(c) == pcol()
     */
    bool is_local(int r, int c) const {
      assert(grid());
      return rowg2p(r) == prow() && colg2p(c) == pcol();
    }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    bool fixed() const { return MB()==default_MB && NB()==default_NB; }
    int rowl2g_fixed(int row) const {
      assert(grid() && fixed());
      return indxl2g(row+1, default_MB, prow(), 0, nprows()) - I();
    }
    int coll2g_fixed(int col) const {
      assert(grid() && fixed());
      return indxl2g(col+1, default_NB, pcol(), 0, npcols()) - J();
    }
    int rowg2l_fixed(int row) const {
      assert(grid() && fixed());
      return indxg2l(row+I(), default_MB, prow(), 0, nprows()) - 1;
    }
    int colg2l_fixed(int col) const {
      assert(grid() && fixed());
      return indxg2l(col+J(), default_NB, pcol(), 0, npcols()) - 1;
    }
    int rowg2p_fixed(int row) const {
      assert(grid() && fixed());
      return indxg2p(row+I(), default_MB, prow(), 0, nprows());
    }
    int colg2p_fixed(int col) const {
      assert(grid() && fixed());
      return indxg2p(col+J(), default_NB, pcol(), 0, npcols());
    }
    int rank_fixed(int r, int c) const {
      assert(grid() && fixed()); return rowg2p_fixed(r) + colg2p_fixed(c) * nprows();
    }
    bool is_local_fixed(int r, int c) const {
      assert(grid() && fixed());
      return rowg2p_fixed(r) == prow() && colg2p_fixed(c) == pcol();
    }
    scalar_t& global_fixed(int r, int c) {
      assert(is_local(r, c)); assert(fixed());
      return operator()(rowg2l_fixed(r),colg2l_fixed(c));
    }
#endif

    /**
     * Access global element r,c. This requires global element r,c to
     * be stored locally on this rank (else it will result in
     * undefined behavior or trigger an assertion).
     * Requires is_local(r,c).
     *
     * \param r global row
     * \param c global col
     * \return const reference to global element r,c
     */
    const scalar_t& global(int r, int c) const {
      assert(is_local(r, c));
      return operator()(rowg2l(r),colg2l(c));
    }
    /**
     * Access global element r,c. This requires global element r,c to
     * be stored locally on this rank (else it will result in
     * undefined behavior or trigger an assertion).
     * Requires is_local(r,c).
     *
     * \param r global row
     * \param c global column
     * \return reference to global element r,c
     */
    scalar_t& global(int r, int c) {
      assert(is_local(r, c)); return operator()(rowg2l(r),colg2l(c));
    }
    /**
     * Set global element r,c if global r,c is local to this rank,
     * otherwise do nothing. This requires global element r,c to be
     * stored locally on this rank (else it will result in undefined
     * behavior or trigger an assertion).  Requires is_local(r,c).
     *
     * \param r global row
     * \param c global column
     * \param v value to set at global position r,c
     */
    void global(int r, int c, scalar_t v) {
      if (active() && is_local(r, c)) operator()(rowg2l(r),colg2l(c)) = v;
    }
    /**
     * Return (on all ranks) the value at global position r,c. The
     * ranks that are not active, i.e., not in the BLACSGrid will
     * receive 0. Since the value at position r,c is stored on only
     * one rank, this performs an (expensive) broadcast. So this
     * should not be used in a loop.
     *
     * \param r global row
     * \param c global column
     * \return value at global position r,c (0 is !active())
     */
    scalar_t all_global(int r, int c) const;

    void print() const { print("A"); }
    void print(std::string name, int precision=15) const;
    void print_to_file(std::string name, std::string filename,
                       int width=8) const;
    void print_to_files(std::string name, int precision=16) const;
    void random();
    void random(random::RandomGeneratorBase<typename RealType<scalar_t>::
                value_type>& rgen);
    void zero();
    void fill(scalar_t a);
    void fill(const std::function<scalar_t(std::size_t,
                                           std::size_t)>& A);
    void eye();
    void shift(scalar_t sigma);
    scalar_t trace() const;
    void clear();
    virtual void resize(std::size_t m, std::size_t n);
    virtual void hconcat(const DistributedMatrix<scalar_t>& b);
    DistributedMatrix<scalar_t> transpose() const;

    void mult(Trans op, const DistributedMatrix<scalar_t>& X,
              DistributedMatrix<scalar_t>& Y) const;

    void laswp(const std::vector<int>& P, bool fwd);

    DistributedMatrix<scalar_t>
    extract_rows(const std::vector<std::size_t>& Ir) const;
    DistributedMatrix<scalar_t>
    extract_cols(const std::vector<std::size_t>& Ic) const;

    DistributedMatrix<scalar_t>
    extract(const std::vector<std::size_t>& I,
            const std::vector<std::size_t>& J) const;
    DistributedMatrix<scalar_t>& add(const DistributedMatrix<scalar_t>& B);
    DistributedMatrix<scalar_t>&
    scaled_add(scalar_t alpha, const DistributedMatrix<scalar_t>& B);
    DistributedMatrix<scalar_t>&
    scale_and_add(scalar_t alpha, const DistributedMatrix<scalar_t>& B);

    real_t norm() const;
    real_t normF() const;
    real_t norm1() const;
    real_t normI() const;

    virtual std::size_t memory() const {
      return sizeof(scalar_t)*std::size_t(lrows())*std::size_t(lcols());
    }
    virtual std::size_t total_memory() const {
      return sizeof(scalar_t)*std::size_t(rows())*std::size_t(cols());
    }
    virtual std::size_t nonzeros() const {
      return std::size_t(lrows())*std::size_t(lcols());
    }
    virtual std::size_t total_nonzeros() const {
      return std::size_t(rows())*std::size_t(cols());
    }

    void scatter(const DenseMatrix<scalar_t>& a);
    DenseMatrix<scalar_t> gather() const;
    DenseMatrix<scalar_t> all_gather() const;

    DenseMatrix<scalar_t> dense_and_clear();
    DenseMatrix<scalar_t> dense() const;
    DenseMatrixWrapper<scalar_t> dense_wrapper();

    std::vector<int> LU();
    int LU(std::vector<int>&);

    DistributedMatrix<scalar_t>
    solve(const DistributedMatrix<scalar_t>& b,
          const std::vector<int>& piv) const;

    void LQ(DistributedMatrix<scalar_t>& L,
            DistributedMatrix<scalar_t>& Q) const;

    void orthogonalize(scalar_t& r_max, scalar_t& r_min);

    void ID_column(DistributedMatrix<scalar_t>& X, std::vector<int>& piv,
                   std::vector<std::size_t>& ind,
                   real_t rel_tol, real_t abs_tol, int max_rank);
    void ID_row(DistributedMatrix<scalar_t>& X, std::vector<int>& piv,
                std::vector<std::size_t>& ind, real_t rel_tol, real_t abs_tol,
                int max_rank, const BLACSGrid* grid_T);


    std::size_t subnormals() const;
    std::size_t zeros() const;

    /**
     * Default row blocksize used for 2D block cyclic
     * dustribution. This is set during CMake configuration.
     *
     * The blocksize is set to a larger value when SLATE is used in
     * order to achieve better GPU performance.
     *
     * The default value is used when no blocksize is specified during
     * matrix construction. Since the default value is a power of 2,
     * some index mapping calculations between local/global can be
     * optimized.
     */
    static const int default_MB = STRUMPACK_PBLAS_BLOCKSIZE;
    /**
     * Default columns blocksize used for 2D block cyclic
     * dustribution. This is set during CMake configuration.
     *
     * \see default_MB
     */
    static const int default_NB = STRUMPACK_PBLAS_BLOCKSIZE;

  protected:
    const BLACSGrid* grid_ = nullptr;
    scalar_t* data_ = nullptr;
    int lrows_;
    int lcols_;
    int desc_[9];
  };

  /**
   * copy submatrix of a DistM_t at ia,ja of size m,n into a DenseM_t
   * b at proc dest
   */
  template<typename scalar_t> void
  copy(std::size_t m, std::size_t n, const DistributedMatrix<scalar_t>& a,
       std::size_t ia, std::size_t ja, DenseMatrix<scalar_t>& b,
       int dest, int context_all);

  template<typename scalar_t> void
  copy(std::size_t m, std::size_t n, const DenseMatrix<scalar_t>& a, int src,
       DistributedMatrix<scalar_t>& b, std::size_t ib, std::size_t jb,
       int context_all);

  /** copy submatrix of a at ia,ja of size m,n into b at position ib,jb */
  template<typename scalar_t> void
  copy(std::size_t m, std::size_t n, const DistributedMatrix<scalar_t>& a,
       std::size_t ia, std::size_t ja, DistributedMatrix<scalar_t>& b,
       std::size_t ib, std::size_t jb, int context_all);

  /**
   * Wrapper class does exactly the same as a regular DistributedMatrix,
   * but it is initialized with existing memory, so it does not
   * allocate, own or delete the memory
   */
  template<typename scalar_t>
  class DistributedMatrixWrapper : public DistributedMatrix<scalar_t> {
  private:
    int _rows, _cols, _i, _j;
  public:
    DistributedMatrixWrapper() : DistributedMatrix<scalar_t>(),
      _rows(0), _cols(0), _i(0), _j(0) {}

    DistributedMatrixWrapper(DistributedMatrix<scalar_t>& A);
    DistributedMatrixWrapper(const DistributedMatrixWrapper<scalar_t>& A);
    DistributedMatrixWrapper(DistributedMatrixWrapper<scalar_t>&& A);
    DistributedMatrixWrapper(std::size_t m, std::size_t n,
                             DistributedMatrix<scalar_t>& A,
                             std::size_t i, std::size_t j);
    DistributedMatrixWrapper(const BLACSGrid* g, std::size_t m, std::size_t n,
                             scalar_t* A);
    DistributedMatrixWrapper(const BLACSGrid* g, std::size_t m, std::size_t n,
                             int MB, int NB, scalar_t* A);
    DistributedMatrixWrapper(const BLACSGrid* g, std::size_t m, std::size_t n,
                             DenseMatrix<scalar_t>& A);

    virtual ~DistributedMatrixWrapper() { this->data_ = nullptr; }

    DistributedMatrixWrapper<scalar_t>&
    operator=(const DistributedMatrixWrapper<scalar_t>& A);
    DistributedMatrixWrapper<scalar_t>&
    operator=(DistributedMatrixWrapper<scalar_t>&& A);

    int rows() const override { return _rows; }
    int cols() const override { return _cols; }
    int I() const override { return _i+1; }
    int J() const override { return _j+1; }
    void lranges(int& rlo, int& rhi, int& clo, int& chi) const override;

    void resize(std::size_t m, std::size_t n) override { assert(1); }
    void hconcat(const DistributedMatrix<scalar_t>& b) override { assert(1); }
    void clear()
    { this->data_ = nullptr; DistributedMatrix<scalar_t>::clear(); }
    std::size_t memory() const override { return 0; }
    std::size_t total_memory() const override { return 0; }
    std::size_t nonzeros() const override { return 0; }
    std::size_t total_nonzeros() const override { return 0; }

    DenseMatrix<scalar_t> dense_and_clear() = delete;
    DenseMatrixWrapper<scalar_t> dense_wrapper() = delete;
    DistributedMatrixWrapper<scalar_t>&
    operator=(const DistributedMatrix<scalar_t>&) = delete;
    DistributedMatrixWrapper<scalar_t>&
    operator=(DistributedMatrix<scalar_t>&&) = delete;
  };


  template<typename scalar_t> long long int
  LU_flops(const DistributedMatrix<scalar_t>& a) {
    if (!a.active()) return 0;
    return (is_complex<scalar_t>() ? 4:1) *
      blas::getrf_flops(a.rows(), a.cols()) /
      a.npactives();
  }

  template<typename scalar_t> long long int
  solve_flops(const DistributedMatrix<scalar_t>& b) {
    if (!b.active()) return 0;
    return (is_complex<scalar_t>() ? 4:1) *
      blas::getrs_flops(b.rows(), b.cols()) /
      b.npactives();
  }

  template<typename scalar_t> long long int
  LQ_flops(const DistributedMatrix<scalar_t>& a) {
    if (!a.active()) return 0;
    auto minrc = std::min(a.rows(), a.cols());
    return (is_complex<scalar_t>() ? 4:1) *
      (blas::gelqf_flops(a.rows(), a.cols()) +
       blas::xxglq_flops(a.cols(), a.cols(), minrc)) /
      a.npactives();
  }

  template<typename scalar_t> long long int
  ID_row_flops(const DistributedMatrix<scalar_t>& a, int rank) {
    if (!a.active()) return 0;
    return (is_complex<scalar_t>() ? 4:1) *
      (blas::geqp3_flops(a.cols(), a.rows())
       + blas::trsm_flops(rank, a.cols() - rank, scalar_t(1.), 'L')) /
      a.npactives();
  }

  template<typename scalar_t> long long int
  trsm_flops(Side s, scalar_t alpha, const DistributedMatrix<scalar_t>& a,
             const DistributedMatrix<scalar_t>& b) {
    if (!a.active()) return 0;
    return (is_complex<scalar_t>() ? 4:1) *
      blas::trsm_flops(b.rows(), b.cols(), alpha, char(s)) /
      a.npactives();
  }

  template<typename scalar_t> long long int
  gemm_flops(Trans ta, Trans tb, scalar_t alpha,
             const DistributedMatrix<scalar_t>& a,
             const DistributedMatrix<scalar_t>& b, scalar_t beta) {
    if (!a.active()) return 0;
    return (is_complex<scalar_t>() ? 4:1) *
      blas::gemm_flops
      ((ta==Trans::N) ? a.rows() : a.cols(),
       (tb==Trans::N) ? b.cols() : b.rows(),
       (ta==Trans::N) ? a.cols() : a.rows(), alpha, beta) /
      a.npactives();
  }

  template<typename scalar_t> long long int
  gemv_flops(Trans ta, const DistributedMatrix<scalar_t>& a,
             scalar_t alpha, scalar_t beta) {
    if (!a.active()) return 0;
    auto m = (ta==Trans::N) ? a.rows() : a.cols();
    auto n = (ta==Trans::N) ? a.cols() : a.rows();
    return (is_complex<scalar_t>() ? 4:1) *
      ((alpha != scalar_t(0.)) * m * (n * 2 - 1) +
       (alpha != scalar_t(1.) && alpha != scalar_t(0.)) * m +
       (beta != scalar_t(0.) && beta != scalar_t(1.)) * m +
       (alpha != scalar_t(0.) && beta != scalar_t(0.)) * m) /
      a.npactives();
  }

  template<typename scalar_t> long long int
  orthogonalize_flops(const DistributedMatrix<scalar_t>& a) {
    if (!a.active()) return 0;
    auto minrc = std::min(a.rows(), a.cols());
    return (is_complex<scalar_t>() ? 4:1) *
      (blas::geqrf_flops(a.rows(), minrc) +
       blas::xxgqr_flops(a.rows(), minrc, minrc)) /
      a.npactives();
  }


  template<typename scalar_t>
  std::unique_ptr<const DistributedMatrixWrapper<scalar_t>>
  ConstDistributedMatrixWrapperPtr
  (std::size_t m, std::size_t n, const DistributedMatrix<scalar_t>& D,
   std::size_t i, std::size_t j) {
    return std::unique_ptr<const DistributedMatrixWrapper<scalar_t>>
      (new DistributedMatrixWrapper<scalar_t>
       (m, n, const_cast<DistributedMatrix<scalar_t>&>(D), i, j));
  }


  template<typename scalar_t> void gemm
  (Trans ta, Trans tb, scalar_t alpha, const DistributedMatrix<scalar_t>& A,
   const DistributedMatrix<scalar_t>& B, scalar_t beta,
   DistributedMatrix<scalar_t>& C);

  template<typename scalar_t> void trsm
  (Side s, UpLo u, Trans ta, Diag d, scalar_t alpha,
   const DistributedMatrix<scalar_t>& A, DistributedMatrix<scalar_t>& B);

  template<typename scalar_t> void trsv
  (UpLo ul, Trans ta, Diag d, const DistributedMatrix<scalar_t>& A,
   DistributedMatrix<scalar_t>& B);

  template<typename scalar_t> void gemv
  (Trans ta, scalar_t alpha, const DistributedMatrix<scalar_t>& A,
   const DistributedMatrix<scalar_t>& X, scalar_t beta,
   DistributedMatrix<scalar_t>& Y);

  template<typename scalar_t> DistributedMatrix<scalar_t> vconcat
  (int cols, int arows, int brows, const DistributedMatrix<scalar_t>& a,
   const DistributedMatrix<scalar_t>& b, const BLACSGrid* gnew, int cxt_all);

  template<typename scalar_t> void subgrid_copy_to_buffers
  (const DistributedMatrix<scalar_t>& a, const DistributedMatrix<scalar_t>& b,
   int p0, int npr, int npc, std::vector<std::vector<scalar_t>>& sbuf);

  template<typename scalar_t> void subproc_copy_to_buffers
  (const DenseMatrix<scalar_t>& a, const DistributedMatrix<scalar_t>& b,
   int p0, int npr, int npc, std::vector<std::vector<scalar_t>>& sbuf);

  template<typename scalar_t> void subgrid_add_from_buffers
  (const BLACSGrid* subg, int master, DistributedMatrix<scalar_t>& b,
   std::vector<scalar_t*>& pbuf);

} // end namespace strumpack

#endif // DISTRIBUTED_MATRIX_HPP
