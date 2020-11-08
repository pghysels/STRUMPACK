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

#include "StrumpackSparseSolver.hpp"
#include "StrumpackSparseSolverBase.hpp"

namespace strumpack {
  template<typename factor_t,typename refine_t,typename integer_t=int>
  class StrumpackSparseSolverMixedPrecision 
  : public StrumpackSparseSolver<factor_t, integer_t> {

  public:
    StrumpackSparseSolverMixedPrecision
    (int argc, char* argv[], bool verbose=true, bool root=true);
    StrumpackSparseSolverMixedPrecision(bool verbose=true, bool root=true);
    ~StrumpackSparseSolverMixedPrecision();

    void refine(const DenseMatrix<refine_t>& b, DenseMatrix<refine_t>& x); 

  private:
    void create_refine_matrix();
    void create_refine_tree();

    std::unique_ptr<CSRMatrix<refine_t,integer_t>> refine_mat_;
    std::unique_ptr<EliminationTree<refine_t,integer_t>> refine_tree_;

  };

} //end namespace strumpack

#endif // STRUMPACK_SPARSE_SOLVER_MIXED_PRECISION_HPP
