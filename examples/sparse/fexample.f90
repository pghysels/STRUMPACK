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

program fexample
  use, intrinsic :: ISO_C_BINDING
  use strumpack
  implicit none

  integer(c_int) :: k      ! grid dimension
  integer, target :: n     ! matrix dimension
  integer :: nnz           ! matrix nonzeros
  integer :: r, c, i       ! row, column, index
  integer(c_int) :: ierr

  ! compressed sparse row representation
  integer, dimension (:), allocatable, target :: rptr, cind
  real(kind=8), dimension(:), allocatable, target :: val

  ! solution and right hand-side vectors
  real(kind=8), dimension(:), allocatable, target :: x, b

  ! sparse solver object
  type(STRUMPACK_SparseSolver) :: S


  k = 400
  write(*,*) "# Solving a ", k, "^2 Poisson problem\n"

  call STRUMPACK_init_mt(S, STRUMPACK_DOUBLE, STRUMPACK_MT, 0, c_null_ptr, 1)

  ! The 2d Poisson problem does not need static pivoting, the matrix
  ! is already diagonally dominant.
  call STRUMPACK_set_matching(S, STRUMPACK_MATCHING_NONE);

  ! Since we are defining the problem on a regular k^2 grid, we can
  ! use a geometric nested dissection ordering, but then we need to
  ! specify the grid dimension, see below.
  call STRUMPACK_set_reordering_method(S, STRUMPACK_GEOMETRIC);
  ! Alternatively, use metis or scotch:
  ! call STRUMPACK_set_reordering_method(S, STRUMPACK_METIS);

  ! Set compression method. Other options include NONE, HSS, HODLR,
  ! LOSSY, LOSSLESS. HODLR is only supported in parallel, and only
  ! supports double precision (including complex double).
  call STRUMPACK_set_compression(S, STRUMPACK_BLR);

  ! Set the block size and relative compression tolerances for BLR
  ! compression.
  call STRUMPACK_set_compression_leaf_size(S, 64);
  call STRUMPACK_set_compression_rel_tol(S, dble(1.e-2));

  ! Only sub-blocks in the sparse triangular factors corresponing to
  ! separators larger than this minimum separator size will be
  ! compressed. For performance, this value should probably be larger
  ! than 128. This value should be larger for HODLR/HODBF, than for
  ! BLR, since HODLR/HODBF have larger constants in the complexity.
  ! For an n x n 2D domain, the largest separator will correspond to
  ! an n x n sub-block in the sparse factors.
  call STRUMPACK_set_compression_min_sep_size(S, 300);

  n = k * k
  nnz = 5 * n - 4 * k
  allocate(rptr(n+1)) ! row pointers (start of each row in cind and val)
  allocate(cind(nnz)) ! column indices
  allocate(val(nnz))  ! nonzero values

  ! Construct a compressed sparse row representation of a sparse
  ! matrix corresponding to a 5-point stencil, i.e., a second order
  ! finite difference scheme, for the 2-d Poisson equation.
  nnz = 1
  rptr(1) = 0
  do r = 0, k-1
     do c = 0, k-1
        i = c + k * r
        val(nnz) = 4.0
        cind(nnz) = i
        nnz = nnz + 1
        if (c > 0) then      ! left
           val(nnz) = -1.0
           cind(nnz) = i - 1
           nnz = nnz + 1
        end if
        if (c < k-1) then    ! right
           val(nnz) = -1.0
           cind(nnz) = i + 1
           nnz = nnz + 1
        end if
        if (r > 0) then      ! up
           val(nnz) = -1.0
           cind(nnz) = i - k
           nnz = nnz + 1
        end if
        if (r < k-1) then    ! down
           val(nnz) = -1.0
           cind(nnz) = i + k
           nnz = nnz + 1
        end if
        rptr(i+2) = nnz - 1
     end do
  end do

  allocate(b(n))
  allocate(x(n))
  do i = 1, n
     b(i) = 1.
     x(i) = 0.
  end do

  call STRUMPACK_set_csr_matrix &
       (S, c_loc(n), c_loc(rptr), c_loc(cind), c_loc(val), 1);

  ! use geometric nested dissection
  ierr = STRUMPACK_reorder_regular(S, k, k, 1)

  ! Solve will internally call factor (and reorder if necessary).
  ierr = STRUMPACK_solve(S, c_loc(b), c_loc(x), 0);

end program fexample
