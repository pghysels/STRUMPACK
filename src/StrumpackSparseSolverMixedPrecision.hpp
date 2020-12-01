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
/**
 * \file StrumpackSparseSolverMixedPrecision.hpp
 * \brief Contains the mixed precision definition of the
 * sequential/multithreaded sparse solver class.
 */
#ifndef STRUMPACK_SPARSE_SOLVER_MIXED_PRECISION_HPP
#define STRUMPACK_SPARSE_SOLVER_MIXED_PRECISION_HPP

#include <new>
#include <memory>
#include <vector>
#include <string>

#include "sparse/iterative/IterativeSolvers.hpp"
#include "StrumpackOptions.hpp"
#include "StrumpackSparseSolver.hpp"
#include "StrumpackSparseSolverBase.hpp"


namespace strumpack {

  template<typename factor_t,typename refine_t,typename integer_t>
  class StrumpackSparseSolverMixedPrecision {

  public:
    StrumpackSparseSolverMixedPrecision(bool verbose=false, bool root=true);
    ~StrumpackSparseSolverMixedPrecision();

    void solve(const DenseMatrix<refine_t>& b, DenseMatrix<refine_t>& x);
    void factor();
    ReturnCode reorder(int nx=1, int ny=1, int nz=1);

    void set_matrix(const CSRMatrix<refine_t,integer_t>& A);

    SPOptions<refine_t>& options() { return opts_; }
    const SPOptions<refine_t>& options() const { return opts_; }
    SPOptions<factor_t>& solver_options() { return solver_.options(); }
    const SPOptions<factor_t>& solver_options() const { return solver_.options(); }

  private:
    CSRMatrix<refine_t,integer_t> mat_;
    StrumpackSparseSolver<factor_t,integer_t> solver_;
    SPOptions<refine_t> opts_;
  };

} //end namespace strumpack

#endif // STRUMPACK_SPARSE_SOLVER_MIXED_PRECISION_HPP
