# STRUMPACK

STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014-2017, The
Regents of the University of California, through Lawrence Berkeley
National Laboratory (subject to receipt of any required approvals from
the U.S. Dept. of Energy).  All rights reserved.


## Installation instructions:

    See INSTALL.txt


## Website:

    http://portal.nersc.gov/project/sparse/strumpack/


## Current developers:

 - Pieter Ghysels -- pghysels@lbl.gov (Lawrence Berkeley National Laboratory)
 - Xiaoye S. Li -- xsli@lbl.gov (Lawrence Berkeley National Laboratory)
 - Christopher Gorman (UC Santa Barbara)

## Other contributors:

 - Francois-Henry Rouet -- fhrouet@lbl.gov,fhrouet@lstc.com (Livermore
   Software Technology Corp., Lawrence Berkeley National Laboratory)


STRUMPACK -- STRUctured Matrices PACKage - is a package for
computations with sparse and dense structured matrices, i.e., matrices
that exhibit some kind of low-rank property, in particular
Hierarchically Semi-Separable matrices (HSS).  Such matrices appear in
many applications, e.g., Finite Element Methods, Boundary Element
Methods... Exploiting this structure using a compression algorithm
allows for fast solution of linear systems and/or fast computation of
matrix-vector products, which are the two main building blocks of
matrix computations. STRUMPACK has two main components: a
distributed-memory dense matrix computations package and a distributed
memory sparse solver/preconditioner.

##  Components:

 - The sparse solver is documented in doc/manual.pdf. STRUMPACK-sparse
   can be used as a direct solver for sparse linear systems or as a
   preconditioner. It also includes GMRes and BiCGStab iterative
   solvers that can use the preconditioner. The preconditioning
   strategy is based on applying low-rank approximations to the
   fill-in of a sparse multifrontal LU factorization.  The code uses
   MPI+OpenMP for hybrid distributed and shared memory parallelism.
   Main point of contact: Pieter Ghysels (pghysels@lbl.gov).

 - The dense distributed-memory package can be found in the src/HSS
   directory. This is currently not well documented. An older version
   of the dense package is available at:
   http://portal.nersc.gov/project/sparse/strumpack/ as
   STRUMPACK-Dense 1.1.1. We refer to the documentation included there
   for more information on installation and usage. The main point of
   contact for STRUMPACK-Dense is Francois-Henry Rouet
   (fhrouet@lbl.gov).

If you have questions about your rights to use or distribute this
software, please contact Berkeley Lab's Technology Transfer Department
at TTD@lbl.gov.

NOTICE.  This software is owned by the U.S. Department of Energy.  As
such, the U.S. Government has been granted for itself and others
acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide
license in the Software to reproduce, prepare derivative works, and
perform publicly and display publicly.  Beginning five (5) years after
the date permission to assert copyright is obtained from the
U.S. Department of Energy, and subject to any subsequent five (5) year
renewals, the U.S. Government is granted for itself and others acting
on its behalf a paid-up, nonexclusive, irrevocable, worldwide license
in the Software to reproduce, prepare derivative works, distribute
copies to the public, perform publicly and display publicly, and to
permit others to do so.