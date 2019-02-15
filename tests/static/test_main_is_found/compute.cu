
#include <stdio.h>

__device__
double power(double x)
{
  double y = (x-x)/(x-x);
  return x*x + 2.0 + y;
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

int main()
{
  return 0;
}
