%module strumpack;

/* -------------------------------------------------------------------------
 * Header definition macros
 * ------------------------------------------------------------------------- */

%insert(fbegin)
%{
! STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The
! Regents of the University of California, through Lawrence Berkeley
! National Laboratory (subject to receipt of any required approvals
! from the U.S. Dept. of Energy).  All rights reserved.
!
! If you have questions about your rights to use or distribute this
! software, please contact Berkeley Lab's Technology Transfer
! Department at TTD@lbl.gov.
!
! NOTICE. This software is owned by the U.S. Department of Energy. As
! such, the U.S. Government has been granted for itself and others
! acting on its behalf a paid-up, nonexclusive, irrevocable,
! worldwide license in the Software to reproduce, prepare derivative
! works, and perform publicly and display publicly.  Beginning five
! (5) years after the date permission to assert copyright is obtained
! from the U.S. Department of Energy, and subject to any subsequent
! five (5) year renewals, the U.S. Government is granted for itself
! and others acting on its behalf a paid-up, nonexclusive,
! irrevocable, worldwide license in the Software to reproduce,
! prepare derivative works, distribute copies to the public, perform
! publicly and display publicly, and to permit others to do so.
!
! Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
!             (Lawrence Berkeley National Lab, Computational Research
!             Division).
%}

/* -------------------------------------------------------------------------
 * Exception handling
 * ------------------------------------------------------------------------- */

// Rename the error variables' internal C symbols
#define SWIG_FORTRAN_ERROR_INT strumpack_ierr
#define SWIG_FORTRAN_ERROR_STR strumpack_get_serr

// Restore names in the wrapper code
%rename(ierr) strumpack_ierr;
%rename(get_serr) strumpack_get_serr;

%include <exception.i>

/* -------------------------------------------------------------------------
 * Data types and instantiation
 * ------------------------------------------------------------------------- */

// Note: stdint.i inserts #include <stdint.h>
%include <stdint.i>

// Support std::string
%include <std_string.i>

// Unless otherwise specified, ignore functions that use unknown types
%fortranonlywrapped;

/* -------------------------------------------------------------------------
 * Wrapped files
 * ------------------------------------------------------------------------- */

%include "StrumpackParameters.i"
%include "StrumpackSparseSolver.i"

