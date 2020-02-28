program testblas
  real x(10), y(10)
  integer n, i
  n = 10
  do i = 1, n
    x(i) = i
  end do
  call scopy(n, x, 1, y, 1)
  print*, 'y = ', (y(i), i = 1, n)
end program testblas
