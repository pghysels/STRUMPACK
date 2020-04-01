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
#ifndef STRUMPACK_AWPM_COMBBLAS_HPP
#define STRUMPACK_AWPM_COMBBLAS_HPP

#include "CombBLAS/CombBLAS.h"
#include "BipartiteMatchings/ApproxWeightPerfectMatching.h"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class CSRMatrixMPI;

  /*! \brief
   *
   * <pre>
   * Purpose
   * =======
   *   Re-distribute A from distributed CSR storage to 2D block storage
   *   conforming CombBLAS API.
   *
   * Arguments
   * =========
   *
   * A      (input) Block-row distributed CSR matris (CSRMatrixMPI)
   *        The distributed input matrix A of dimension (A.size(), A.size()).
   *        A may be overwritten by diag(R)*A*diag(C)*Pc^T.
   *
   * Return value
   * ============
   * perm_c (output) integer_t*
   *        Permutation vector describing the transformation to be
   *        performed to the original matrix A.
   *
   * </pre>
   */
  template<typename scalar_t, typename integer_t>
  void GetAWPM(const CSRMatrixMPI<scalar_t,integer_t>& A,
               integer_t* perm_c) {
    integer_t* perm = nullptr; // placeholder for load balancing permutation for CombBLAS
    int procs = A.Comm().size();
    int sqrtP = (int)std::sqrt((double)procs);
    if (sqrtP * sqrtP != procs) {
      if (A.Comm().is_root())
        std::cerr << "# WARNING: Combinatorial BLAS currently only works on "
                  << "a square number of processes. (disabling)" << std::endl;
      std::iota(perm_c, perm_c+A.size(), 0);
      return;
    }

    using real_t = typename RealType<scalar_t>::value_type;
    combblas::SpParMat<integer_t,real_t,combblas::SpDCCols<integer_t,real_t>> Adcsc(A.comm());
    std::vector<std::vector<std::tuple<integer_t,integer_t,real_t>>> data(procs);

    // ------------------------------------------------------------
    //  INITIALIZATION.
    // ------------------------------------------------------------
    integer_t n = A.size(), m_loc = A.local_rows(), fst_row = A.begin_row();

    // ------------------------------------------------------------
    // FIRST PASS OF A:
    // COUNT THE NUMBER OF NONZEROS TO BE SENT TO EACH PROCESS,
    // THEN ALLOCATE SPACE.
    // ------------------------------------------------------------
    integer_t nnz_loc = 0, irow, jcol, lirow, ljcol;
    for (integer_t i=0; i<m_loc; ++i) {
      for (integer_t j=A.ptr(i); j<A.ptr(i+1); ++j) {
        if (perm != NULL) {
          irow = perm[i+fst_row];      // Row number in P*A*P^T
          jcol = perm[A.ind(j)]; // Column number in P*A*P^T
        } else {
          irow = i+fst_row;
          jcol = A.ind(j);
        }
        int p = Adcsc.Owner(n, n, irow, jcol, lirow, ljcol);
        ++nnz_loc;
        data[p].push_back(std::make_tuple(lirow, ljcol, std::real(A.val(j))));
      }
    }

    Adcsc.SparseCommon(data, nnz_loc, n, n, std::plus<real_t>());
    combblas::FullyDistVec<integer_t,integer_t> mateRow2Col
      (Adcsc.getcommgrid(), n, (integer_t) -1);
    combblas::FullyDistVec<integer_t,integer_t> mateCol2Row
      (Adcsc.getcommgrid(), n, (integer_t) -1);
    combblas::AWPM(Adcsc, mateRow2Col, mateCol2Row, true);

    // now gather the matching vector
    MPI_Comm World = mateRow2Col.getcommgrid()->GetWorld();
    std::unique_ptr<int[]> iwork(new int[2*procs]);
    auto rdispls = iwork.get();
    auto recvcnt = rdispls + procs;
    int sendcnt = mateRow2Col.LocArrSize();
    MPI_Allgather(&sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
    rdispls[0] = 0;
    for (int i=0; i<procs-1; ++i)
      rdispls[i+1] = rdispls[i] + recvcnt[i];
    integer_t *senddata = (integer_t*)mateRow2Col.GetLocArr();
    MPI_Allgatherv
      (senddata, sendcnt, combblas::MPIType<integer_t>(),
       perm_c, recvcnt, rdispls, combblas::MPIType<integer_t>(), World);
  }

} // end namespace strumpack

#endif // STRUMPACK_AWPM_COMBBLAS_HPP
