program testlapack
  real(8),dimension(3,3) :: A
  integer,dimension(3) :: piv
  integer n, i, info
  n = 3
  do i = 1, n
     do j = 1, n
        A(i,j) = 1. / abs(i-j-1)
     end do
  end do
  call dgetrf(n, n, A, n, piv, info)
  print*, 'info = ', info
end program testlapack
