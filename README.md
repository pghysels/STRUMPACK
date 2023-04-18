# STRUMPACK
STRUMPACK -- STRUctured Matrix PACKage, Copyright (c) 2014-2021, The
Regents of the University of California, through Lawrence Berkeley
National Laboratory (subject to receipt of any required approvals from
the U.S. Dept. of Energy).  All rights reserved.

## Documentation & Installation instructions
   [http://portal.nersc.gov/project/sparse/strumpack/master/](http://portal.nersc.gov/project/sparse/strumpack/master/)

   [http://portal.nersc.gov/project/sparse/strumpack/v7.1.0/](http://portal.nersc.gov/project/sparse/strumpack/v7.1.0/)


## Website
   [http://portal.nersc.gov/project/sparse/strumpack/](http://portal.nersc.gov/project/sparse/strumpack/)


## Current developers - Lawrence Berkeley National Laboratory
 - Pieter Ghysels - pghysels@lbl.gov
 - Xiaoye S. Li - xsli@lbl.gov
 - Yang Liu - liuyangzhuan@lbl.gov
 - Lisa Claus - LClaus@lbl.gov
 - Wajih Boukaram - wajih.boukaram@lbl.gov
 - Yotam Yaniv - yotamya@math.ucla.edu (UCLA)

## Past contributors
 - Ryan Synk
 - Lucy Guo
 - Gustavo Chávez
 - Liza Rebrova - UCLA, University of Michigan
 - François-Henry Rouet - Livermore Software Technology Corp., Ansys
 - Theo Mary - University of Manchester
 - Christopher Gorman - UC Santa Barbara
 - Jonas Actor - Rice University
 - Michael Neuder - Harvard


## Overview

STRUMPACK - STRUctured Matrix PACKage - is a software library
providing linear algebra routines and linear system solvers for sparse
and for dense rank-structured linear systems. Many large dense
matrices are rank structured, meaning they exhibit some kind of
low-rank property, for instance in hierarchically defined
sub-blocks. In sparse direct solvers based on LU factorization, the LU
factors can often also be approximated well using rank-structured
matrix compression, leading to robust preconditioners. The sparse
solver in STRUMPACK can also be used as an exact direct solver, in
which case it functions similarly as for instance
[SuperLU](https://github.com/xiaoyeli/superlu) or
[superlu_dist](https://github.com/xiaoyeli/superlu_dist). The
STRUMPACK sparse direct solver delivers good performance and
distributed memory scalability and provides excellent CUDA support.

Currently, STRUMPACK has support for the Hierarchically Semi-Separable
(HSS), Block Low Rank (BLR), Hierachically Off-Diagonal Low Rank
(HODLR), Butterfly and Hierarchically Off-Diagonal Butterfly (HODBF)
rank-structured matrix formats. Such matrices appear in many
applications, e.g., the Boundary Element Method for discretization of
integral equations, structured matrices like Toeplitz and Cauchy,
kernel and covariance matrices etc. In the LU factorization of sparse
linear systems arising from the discretization of partial differential
equations, the fill-in in the triangular factors often has low-rank
structure. Hence, the sparse linear solve algorithms in STRUMPACK
exploit the different dense rank-structured matrix formats to compress
the fill-in. This leads to purely algebraic, fast and scalable (both
with problem size and compute cores) approximate direct solvers or
preconditioners. These preconditioners are mostly aimed at large
sparse linear systems which result from the discretization of a
partial differential equation, but are not limited to any particular
type of problem. STRUMPACK also provides preconditioned GMRES and
BiCGStab iterative solvers.

Apart from rank-structured compression, the STRUMPACK sparse solver
also support compression of the factors using the
[ZFP](https://computing.llnl.gov/projects/floating-point-compression)
library, a general purpose compression algorithm tuned for floating
point data. This can be used with a specified precision, or with
lossless compression.

The HODLR and Butterfly functionality in STRUMPACK is implemented
through interfaces to the ButterflyPACK package:
    [https://github.com/liuyangzhuan/ButterflyPACK](https://github.com/liuyangzhuan/ButterflyPACK)



## NOTICE

This software is owned by the U.S. Department of Energy.  As
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

If you have questions about your rights to use or distribute this
software, please contact Berkeley Lab's Technology Transfer Department
at TTD@lbl.gov.
