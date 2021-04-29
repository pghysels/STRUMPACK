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
/*!
 * \file IterativeSolvers.hpp
 * \brief Contains (sequential/threaded) iterative solvers.
 */
#ifndef STRUMPACK_ITERATIVE_SOLVERS_HPP
#define STRUMPACK_ITERATIVE_SOLVERS_HPP

#include <functional>

#include "StrumpackOptions.hpp" // for GramSchmidtType
#include "sparse/CompressedSparseMatrix.hpp"
#include "dense/DenseMatrix.hpp"

namespace strumpack {
  namespace iterative {

    template<typename T>
    using SPMV = std::function<void(const T*, T*)>;

    template<typename T>
    using PREC = std::function<void(T*)>;

    /*
     * This is left preconditioned restarted GMRes.
     *
     *  Input vectors x and b have stride 1, length n
     */
    template<typename scalar_t,
             typename real_t = typename RealType<scalar_t>::value_type>
    real_t GMRes(const SPMV<scalar_t>& A,
                 const PREC<scalar_t>& M,
                 std::size_t n, scalar_t* x, const scalar_t* b,
                 real_t rtol, real_t atol, int& totit, int maxit,
                 int restart, GramSchmidtType GStype,
                 bool non_zero_guess, bool verbose);


    /**
     * http://www.netlib.org/templates/matlab/bicgstab.m
     */
    template<typename scalar_t,
             typename real_t = typename RealType<scalar_t>::value_type>
    real_t BiCGStab(const SPMV<scalar_t>& A,
                    const PREC<scalar_t>& M,
                    std::size_t n, scalar_t* x, const scalar_t* b,
                    real_t rtol, real_t atol, int& totit, int maxit,
                    bool non_zero_guess, bool verbose);

    /**
     * Iterative refinement, with a sparse matrix, to solve a linear
     * system M^{-1}Ax=M^{-1}b.
     *
     * \tparam scalar_t scalar type
     * \tparam integer_t integer type used in A
     * \tparam real_t real type, can be derived from the scalar_t type
     *
     * \param A dense matrix A
     * \param M routine to apply M^{-1} to a matrix
     * \param x on output this contains the solution, on input this can
     * be the initial guess. This always has to be allocated to the
     * correct size (A.rows() x b.cols())
     * \param b the right hand side, should have A.rows() rows
     * \param rtol relative stopping tolerance
     * \param atol absolute stopping tolerance
     * \param totit on output this will contain the number of iterations
     * that were performed
     * \param maxit maximum number of iterations
     * \param non_zero_guess x use x as an initial guess
     */
    template<typename scalar_t,typename integer_t,
             typename real_t = typename RealType<scalar_t>::value_type>
    void IterativeRefinement(const CompressedSparseMatrix<scalar_t,integer_t>& A,
                             const std::function<void(DenseMatrix<scalar_t>&)>& M,
                             DenseMatrix<scalar_t>& x,
                             const DenseMatrix<scalar_t>& b,
                             real_t rtol, real_t atol, int& totit, int maxit,
                             bool non_zero_guess, bool verbose);


    /**
     * Iterative refinement, with a dense matrix, to solve a linear
     * system M^{-1}Ax=M^{-1}b.
     *
     * \tparam scalar_t scalar type
     * \tparam real_t real type, can be derived from the scalar_t type
     *
     * \param A dense matrix A
     * \param direct_solve routine to apply M^{-1} to a matrix
     * \param x on output this contains the solution, on input this can
     * be the initial guess. This always has to be allocated to the
     * correct size (A.rows() x b.cols())
     * \param b the right hand side, should have A.rows() rows
     * \param rtol relative stopping tolerance
     * \param atol absolute stopping tolerance
     * \param totit on output this will contain the number of iterations
     * that were performed
     * \param maxit maximum number of iterations
     * \param non_zero_guess x use x as an initial guess
     */
    template<typename scalar_t,
             typename real_t = typename RealType<scalar_t>::value_type>
    void IterativeRefinement(const DenseMatrix<scalar_t>& A,
                             const std::function
                             <void(DenseMatrix<scalar_t>&)>& M,
                             DenseMatrix<scalar_t>& x,
                             const DenseMatrix<scalar_t>& b,
                             real_t rtol, real_t atol, int& totit, int maxit,
                             bool non_zero_guess, bool verbose);

  } // end namespace iterative
} // end namespace strumpack

#endif // STRUMPACK_ITERATIVE_SOLVERS_HPP
