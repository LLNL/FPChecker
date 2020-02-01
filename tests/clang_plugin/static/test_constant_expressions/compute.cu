
#include <stdio.h>

__device__
double compute(double x)
{
  double y;
  double z;

  y = _FPC_CHECK_(0., 10, "compute.cu");
  z = _FPC_CHECK_(0, 11, "compute.cu");

  return _FPC_CHECK_(y + z, 13, "compute.cu");
}

