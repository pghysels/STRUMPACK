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
#ifndef STRUMPACK_CONFIG_HPP
#define STRUMPACK_CONFIG_HPP

#define STRUMPACK_USE_MPI

#define STRUMPACK_USE_METIS
#define STRUMPACK_USE_PARMETIS
#define STRUMPACK_USE_SCOTCH
#define STRUMPACK_USE_OPENMP
#define STRUMPACK_C_INTERFACE

#define STRUMPACK_USE_OPENMP_TASKLOOP
#define STRUMPACK_USE_OPENMP_TASK_DEPEND

#define STRUMPACK_VERSION_MAJOR 3
#define STRUMPACK_VERSION_MINOR 0
#define STRUMPACK_VERSION_PATCH 2

#endif // STRUMPACK_CONFIG_HPP

