
#include <stdio.h>

__device__
double power(double x)
{
  return x*x + 2.0;
}

__global__
void compute(double x)
{
  double y = power(x);
  y = y + x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid == 0) {
    printf("y: %f\n", y);
  }
}
