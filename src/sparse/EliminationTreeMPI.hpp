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
#ifndef ELIMINATION_TREE_MPI_HPP
#define ELIMINATION_TREE_MPI_HPP

#include "EliminationTree.hpp"
#include "misc/MPIWrapper.hpp"

namespace strumpack {

  // forward declarations
  template<typename scalar_t,typename integer_t> class MatrixReordering;
  template<typename scalar_t> class DistributedMatrix;
  class BLACSGrid;


  template<typename scalar_t,typename integer_t>
  class EliminationTreeMPI : public EliminationTree<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using Reord_t = MatrixReordering<scalar_t,integer_t>;
    using Tree_t = SeparatorTree<integer_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using SepRange = std::pair<integer_t,integer_t>;

  public:
    EliminationTreeMPI(const MPIComm& comm);

    virtual ~EliminationTreeMPI();

    integer_t maximum_rank() const override;
    long long factor_nonzeros() const override;
    long long dense_factor_nonzeros() const override;
    const MPIComm& Comm() const { return comm_; }

    ReturnCode inertia(integer_t& neg,
                       integer_t& zero,
                       integer_t& pos) const override;
    ReturnCode subnormals(std::size_t& ns,
                          std::size_t& nz) const override;
    ReturnCode pivot_growth(scalar_t& pgL,
                            scalar_t& pgU) const override;

  protected:
    const MPIComm& comm_;
    int rank_, P_;

    std::vector<SepRange> subtree_ranges_;
    SepRange local_range_;

    virtual FrontCounter front_counter() const override;
    void update_local_ranges(integer_t lo, integer_t hi);
  };

} // end namespace strumpack

#endif
