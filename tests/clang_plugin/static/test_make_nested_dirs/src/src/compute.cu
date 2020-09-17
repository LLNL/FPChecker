
#include <stdio.h>

__device__
double compute(double x)
{
  double y = _FPC_CHECK_(x / (x + 1.3), 7, "src/src/compute.cu");
  return y;
}

