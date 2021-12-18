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

program fstructured
  use, intrinsic :: ISO_C_BINDING
  use strumpack_dense
  implicit none

  integer(C_INT) :: n        ! matrix dimension
  integer(C_INT) :: nrhs     ! number of right hand sides
  integer :: r, c, i         ! row, column, index
  integer(C_INT) :: ierr

  ! dense Toeplitz matrix
  ! real(C_DOUBLE), dimension(:,:), allocatable, target :: T
  COMPLEX*16, dimension(:,:), allocatable, target :: T

  ! solution and right hand-side vectors (matrices)
  !real(C_DOUBLE), dimension(:,:), allocatable, target :: X, B
  COMPLEX*16, dimension(:,:), allocatable, target :: X, B


  ! structured matrix options structure
  type(CSPOptions) :: options

  ! structured matrix
  type(C_PTR) :: S

  n = 1000
  nrhs = 10
  write(*,*) "# Creating a ", n, "x", n, " Toeplitz matrix"

  allocate( T( n, n ) )

  ! construct a simple (dense) test matrix
  do c = 1, n
     do r = 1, n
        T( r, c ) = 1. / (1. + abs(r - c))
     end do
  end do

  allocate( B( n, nrhs ) )
  allocate( X( n, nrhs ) )

  do c = 1, nrhs
     do r = 1, n
        X( r, c ) = 1. / n
     end do
  end do


  call SP_d_struct_default_options( options )
  options%type = SP_TYPE_HSS

  ! construct a rank-structured matrix from T
  ierr = SP_z_struct_from_dense( S, n, n, c_loc(T(1, 1)), n, options )

  ! compute the right-hand side B from X as B = H*X
  ierr = SP_z_struct_mult( S, 'N', nrhs, c_loc(X(1, 1)), n, c_loc(B(1, 1)), n )

  ! compute a factorization of H
  ierr = SP_z_struct_factor( S )

  ! solve linear system H * X = B
  write(*,*) "# solving linear system"
  ierr = SP_z_struct_solve( S, nrhs, c_loc(B(1, 1)), n )


  ! TODO check accuracy of compression, solve

  call SP_z_struct_destroy( S )

  deallocate(T)
  deallocate(X)
  deallocate(B)

end program fstructured
