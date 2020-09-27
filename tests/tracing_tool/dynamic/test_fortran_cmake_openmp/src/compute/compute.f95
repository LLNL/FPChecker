

function compute()  result(N)
  implicit none

  integer :: N         ! Shared

  N = 1001
  print *, "Before parallel section: N = ", N            

  !$OMP PARALLEL
  N = N + 1
  print *, "Inside parallel section: N = ", N
  !$OMP END PARALLEL

  print *, "After parallel section: N = ", N

end function compute

