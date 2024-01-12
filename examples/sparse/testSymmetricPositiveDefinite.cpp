//
// Created by tingxuan on 2023/12/24.
//
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
#include <iostream>
#include <type_traits>
#include <random>
#include <cmath>

#include "StrumpackSparseSolver.hpp"
#include "StrumpackSparseSolverMixedPrecision.hpp"
#include "sparse/CSRMatrix.hpp"

using namespace strumpack;


/**
 * Test the STRUMPACK sparse solver, and the mixed precision sparse
 * solver.
 *
 * For working_t == float, the mixed precision solver will
 * compute the factorization in single precision, but do iterative
 * refinement in double precision to give a more accurate results than
 * the standard single precision solver.
 *
 * For working_t == double, the mixed precision solver will compute
 * the factorization in single precision and perform the iterative
 * refinement in double precision. If the problem is not too
 * ill-conditioned, this should be about as accurate, and about twice
 * as fast as the standard double precision solver. The speedup
 * depends on the relative cost of the sparse triangular solver phase
 * compared to the sparse LU factorization phase.
 *
 * TODO long double
 */
template<typename working_t>
void test(CSRMatrix<working_t,int>& A,
          DenseMatrix<working_t>& b, DenseMatrix<working_t>& x_exact,
          int argc, char* argv[]) {
    int m = b.cols();  // number of right-hand sides
    auto N = A.size();
    DenseMatrix<working_t> x(N, m);

    std::cout << std::endl;
    std::cout << "###############################################" << std::endl;
    std::cout << "### Working precision: " <<
              (std::is_same<float,working_t>::value ? "single" : "double")
              << " #################" << std::endl;
    std::cout << "###############################################" << std::endl;

    {
        std::cout << std::endl;
        std::cout << "### MIXED Precision Solver ####################" << std::endl;

        SparseSolverMixedPrecision<float,double,int> spss;
        /** options for the outer solver */
        spss.options().set_Krylov_solver(KrylovSolver::REFINE);
//     spss.options().set_Krylov_solver(KrylovSolver::PREC_BICGSTAB);
//     spss.options().set_Krylov_solver(KrylovSolver::PREC_GMRES);
        spss.options().set_rel_tol(1e-14);
        spss.options().set_from_command_line(argc, argv);

        /** options for the inner solver */
        spss.solver().options().set_Krylov_solver(KrylovSolver::DIRECT);
        spss.solver().options().set_from_command_line(argc, argv);
        spss.options().set_matching(strumpack::MatchingJob::NONE);
        spss.solver().options().set_matching(strumpack::MatchingJob::NONE);
        spss.options().enable_symmetric();
        spss.options().enable_positive_definite();
        spss.solver().options().enable_symmetric();
        spss.solver().options().enable_positive_definite();

        spss.set_matrix(A);
        spss.reorder();
        spss.factor();
        spss.solve(b, x);

        std::cout << "# COMPONENTWISE SCALED RESIDUAL = "
                  << A.max_scaled_residual(x.data(), b.data()) << std::endl;
        strumpack::blas::axpy(N, -1., x_exact.data(), 1, x.data(), 1);
        auto nrm_error = strumpack::blas::nrm2(N, x.data(), 1);
        auto nrm_x_exact = strumpack::blas::nrm2(N, x_exact.data(), 1);
        std::cout << "# RELATIVE ERROR = " << (nrm_error/nrm_x_exact) << std::endl;
    }

    std::cout << std::endl;
}


int main(int argc, char* argv[]) {

    CSRMatrix<double,int> A_d;
    if(argc > 1){
        A_d.read_matrix_market(argv[1]);
    }else{
        int n =3;
        int ptr[4] = {0,2,3,5};
        int Index[5] = {0,2,1,0,2};
        double val[5] = {2.1,1,3.5,1,5.2};
        A_d = CSRMatrix<double,int>(n, ptr, Index, val);
    }


    int N = A_d.size();
    int m = 1; // nr of RHSs
    DenseMatrix<double> b_d(N, m), x_true_d(N, m);


    // set the exact solution, see:
    //   http://www.netlib.org/lapack/lawnspdf/lawn165.pdf
    // page 20
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(0., std::sqrt(24.));
    for (int j=0; j<m; j++) {
        // step 4, use a different tau for each RHS
        double tau = std::pow(dist(gen), 2.);
        for (int i=0; i<N; i++)
            // step 4c
            x_true_d(i, j) = std::pow(tau, -double(i)/(N-1));
    }

    // step 6, but in double, not double-double
    A_d.spmv(x_true_d, b_d);
    {
        DenseMatrix<double> x(N, m);
        // step 7, but in double, not double-double
        SparseSolver<double,int> spss;
        // SparseSolverMixedPrecision<double,long double,int> spss;
        spss.options().enable_symmetric();
        spss.options().enable_positive_definite();
        spss.set_matrix(A_d);
        spss.options().set_matching(strumpack::MatchingJob::NONE);
        spss.options().set_Krylov_solver(KrylovSolver::DIRECT);
        spss.solve(b_d, x);

        std::cout<<"x_true_d=";
        for(int r=0; r<N; r++){
            for(int c=0; c<m; c++){
                std::cout<<x_true_d.data()[r+m*c];
            }
            std::cout<<std::endl;
        }

        std::cout<<"x_solve=";
        for(int r=0; r<N; r++){
            for(int c=0; c<m; c++){
                std::cout<<x.data()[r+m*c];
            }
            std::cout<<std::endl;
        }

    }
    return 0;
}
