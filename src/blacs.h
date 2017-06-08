/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The Regents of
 * the University of California, through Lawrence Berkeley National Laboratory
 * (subject to receipt of any required approvals from the U.S. Dept. of Energy).
 * All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Technology Transfer Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, prepare derivative works, and perform publicly and display publicly.
 * Beginning five (5) years after the date permission to assert copyright is
 * obtained from the U.S. Department of Energy, and subject to any subsequent five
 * (5) year renewals, the U.S. Government is granted for itself and others acting
 * on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, prepare derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit others to do so.
 *
 * Developers: Francois-Henry Rouet, Xiaoye S. Li, Pieter Ghysels
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 *
 */

#ifndef BLACS_H
#define BLACS_H

/* BLACS routines */
extern "C" {
  void blacs_get_(int *, int *, int *);
  void blacs_gridinit_(int *, const char *, int *, int *);
  void blacs_gridmap_(int *, int*, int *, int *, int *);
  void blacs_gridinfo_(int *, int *, int *, int *, int *);
  void blacs_gridexit_(int *);
  void blacs_exit_(int *);
  MPI_Comm blacs2sys_handle_(int *ctxt);
  void Cblacs_get(int, int, int *);
  void Cblacs_gridinit(int *, const char *, int, int);
  void Cblacs_gridmap(int *, int *, int, int, int);
  void Cblacs_gridinfo(int, int *, int *, int *, int *);
  void Cblacs_gridexit(int);
  void Cblacs_exit(int);
  int Csys2blacs_handle(MPI_Comm);
  MPI_Comm Cblacs2sys_handle(int);
}

#endif // BLACS_H
