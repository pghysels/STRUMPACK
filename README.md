# STRUMPACK
STRUMPACK -- STRUctured Matrix PACKage, Copyright (c) 2014-2020, The
Regents of the University of California, through Lawrence Berkeley
National Laboratory (subject to receipt of any required approvals from
the U.S. Dept. of Energy).  All rights reserved.

[![Build Status](https://travis-ci.org/pghysels/STRUMPACK.svg?branch=master)](https://travis-ci.org/pghysels/STRUMPACK)

## Documentation & Installation instructions
   [http://portal.nersc.gov/project/sparse/strumpack/master/](http://portal.nersc.gov/project/sparse/strumpack/master/)  
   [http://portal.nersc.gov/project/sparse/strumpack/v3.3.0/](http://portal.nersc.gov/project/sparse/strumpack/v3.3.0/)


## Website
   [http://portal.nersc.gov/project/sparse/strumpack/](http://portal.nersc.gov/project/sparse/strumpack/)


## Current developers
 - Pieter Ghysels - pghysels@lbl.gov (Lawrence Berkeley National Laboratory)
 - Xiaoye S. Li - xsli@lbl.gov (Lawrence Berkeley National Laboratory)
 - Yang Liu - liuyangzhuan@lbl.gov (Lawrence Berkeley National Laboratory)
 - Lisa Claus - LClaus@lbl.gov (Lawrence Berkeley National Laboratory)

## Other contributors
 - Lucy Guo - lcguo@lbl.gov
 - Gustavo Chávez - gichavez@lbl.gov
 - Liza Rebrova - erebrova@umich.edu (University of Michigan)
 - François-Henry Rouet - fhrouet@lbl.gov,fhrouet@lstc.com (Livermore
   Software Technology Corp., Ansys)
 - Theo Mary - theo.mary@manchester.ac.uk (University of Manchester)
 - Christopher Gorman - (UC Santa Barbara)
 - Jonas Actor - (Rice University)

## Overview
STRUMPACK - STRUctured Matrix PACKage - is a software library
providing linear algebra routines for sparse matrices and for dense
rank-structured matrices, i.e., matrices that exhibit some kind of
low-rank property. In particular, STRUMPACK uses the Hierarchically
Semi-Separable matrix format (HSS).  Such matrices appear in many
applications, e.g., Finite Element Methods, Boundary Element Methods
... In sparse matrix factorization, the fill-in in the triangular
factors often has a low-rank structure. Hence, the sparse linear
solve in STRUMPACK exploits the HSS matrix format to compress the
fill-in. Exploiting this structure using a compression algorithm
allows for fast solution of linear systems and/or fast computation of
matrix-vector products, which are two of the main building blocks of
matrix computations. STRUMPACK has two main components: a
distributed-memory dense matrix computations package (for dense
matrices that have the HSS structure) and a distributed memory fully
algebraic sparse general solver and preconditioner. The preconditioner
is mostly aimed at large sparse linear systems which result from the
discretization of a partial differential equation, but is not limited
to any particular type of problem. STRUMPACK also provides
preconditioned GMRES and BiCGStab iterative solvers.

If you have questions about your rights to use or distribute this
software, please contact Berkeley Lab's Technology Transfer Department
at TTD@lbl.gov.

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
