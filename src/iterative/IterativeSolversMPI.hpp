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
#ifndef STRUMPACK_ITERATIVE_SOLVERS_MPI_HPP
#define STRUMPACK_ITERATIVE_SOLVERS_MPI_HPP

#include "IterativeSolvers.hpp"
#include "misc/MPIWrapper.hpp"
#include "sparse/CSRMatrixMPI.hpp"
#include "dense/DenseMatrix.hpp"
#include "dense/DistributedVector.hpp"
#include "dense/DistributedMatrix.hpp"

namespace strumpack {
  namespace iterative {

    /**
     * This is left preconditioned restarted GMRes.
     * Collective operation on comm.
     *
     * Vectors x and b should be divided over the processors in the same
     * way as the matrix,
     *
     * with n the local size (ie number of rows of A stored on this
     * rank).
     *
     * Input vectors x and b have stride 1 and (local) length n
     *
     */
    template<typename scalar_t,
             typename real_t = typename RealType<scalar_t>::value_type>
    real_t GMResMPI(const MPIComm& comm,
                    const std::function
                    <void(const scalar_t*,scalar_t*)>& spmv,
                    const std::function
                    <void(scalar_t*)>& prec,
                    std::size_t n, scalar_t* x, const scalar_t* b,
                    real_t rtol, real_t atol, int& totit, int maxit,
                    int restart, GramSchmidtType GStype,
                    bool non_zero_guess, bool verbose);

    template<typename scalar_t,
             typename real_t = typename RealType<scalar_t>::value_type>
    real_t GMResMPI(const MPIComm& comm,
                    const std::function
                    <void(const DenseMatrix<scalar_t>&,
                          DenseMatrix<scalar_t>&)>& spmv,
                    const std::function
                    <void(DenseMatrix<scalar_t>&)>& prec,
                    DenseMatrix<scalar_t>& x, const DenseMatrix<scalar_t>& b,
                    real_t rtol, real_t atol, int& totit, int maxit,
                    int restart, GramSchmidtType GStype,
                    bool non_zero_guess, bool verbose) {
      assert(x.cols() == 1 && b.cols() == 1);
      assert(x.rows() == b.rows());
      auto n = x.rows();
      return GMResMPI<scalar_t,real_t>
        (comm,
         [&](const scalar_t* v, scalar_t* w){
           DenseMatrixWrapper<scalar_t> W(n, 1, w, n),
             V(n, 1, const_cast<scalar_t*>(v), n);
           spmv(V, W);
         },
         [&](scalar_t* v){
           DenseMatrixWrapper<scalar_t> V(n, 1, v, n);
           prec(V);
         },
         n, x.data(), b.data(), rtol, atol, totit, maxit,
         restart, GStype, non_zero_guess, verbose);
    }


    /**
     * http://www.netlib.org/templates/matlab/bicgstab.m
     */
    template<typename scalar_t,
             typename real_t = typename RealType<scalar_t>::value_type>
    real_t BiCGStabMPI(const MPIComm& comm,
                       const std::function
                       <void(const scalar_t*,scalar_t*)>& spmv,
                       const std::function
                       <void(scalar_t*)>& preconditioner,
                       std::size_t n, scalar_t* x, const scalar_t* b,
                       real_t rtol, real_t atol, int& totit, int maxit,
                       bool non_zero_guess, bool verbose);

    template<typename scalar_t,
             typename real_t = typename RealType<scalar_t>::value_type>
    real_t BiCGStabMPI(const MPIComm& comm,
                       const std::function
                       <void(const DenseMatrix<scalar_t>&,
                             DenseMatrix<scalar_t>&)>& spmv,
                       const std::function
                       <void(DenseMatrix<scalar_t>&)>& prec,
                       DenseMatrix<scalar_t>& x, const DenseMatrix<scalar_t>& b,
                       real_t rtol, real_t atol, int& totit, int maxit,
                       bool non_zero_guess, bool verbose) {
      assert(x.cols() == 1 && b.cols() == 1);
      assert(x.rows() == b.rows());
      auto n = x.rows();
      return BiCGStabMPI<scalar_t,real_t>
        (comm,
         [&](const scalar_t* v, scalar_t* w){
           DenseMatrixWrapper<scalar_t> W(n, 1, w, n),
             V(n, 1, const_cast<scalar_t*>(v), n);
           spmv(V, W);
         },
         [&](scalar_t* v){
           DenseMatrixWrapper<scalar_t> V(n, 1, v, n);
           prec(V);
         },
         n, x.data(), b.data(), rtol, atol, totit, maxit,
         non_zero_guess, verbose);
    }


    /**
     * Iterative refinement.
     * Input vectors x and b have stride 1, length n
     */
    template<typename scalar_t,typename integer_t,
             typename real_t = typename RealType<scalar_t>::value_type>
    void IterativeRefinementMPI(const MPIComm& comm,
                                const CSRMatrixMPI<scalar_t,integer_t>& A,
                                const std::function
                                <void(DenseMatrix<scalar_t>&)>& direct_solve,
                                DenseMatrix<scalar_t>& x,
                                const DenseMatrix<scalar_t>& b,
                                real_t rtol, real_t atol, int& totit, int maxit,
                                bool non_zero_guess, bool verbose);

    /**
     * Iterative refinement with the right-hand side and solution
     * vectors in 2d block cyclic distribution.
     */
    template<typename scalar_t, typename real_t>
    void IterativeRefinementMPI(const MPIComm& comm,
                                const std::function
                                <void(const DistributedMatrix<scalar_t>&,
                                      DistributedMatrix<scalar_t>&)>& A,
                                const std::function
                                <void(DistributedMatrix<scalar_t>&)>& M,
                                DistributedMatrix<scalar_t>& x,
                                const DistributedMatrix<scalar_t>& b,
                                real_t rtol, real_t atol,
                                int& totit, int maxit,
                                bool non_zero_guess, bool verbose);

  } // end namespace iterative
} // end namespace strumpack

#endif // STRUMPACK_ITERATIVE_SOLVERS_MPI_HPP
