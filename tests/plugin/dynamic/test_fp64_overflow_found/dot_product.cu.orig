
#include <stdio.h>

__device__ void mul(double a, double b, double *res)
{
  *res = a * b;
  // Overflow

#ifdef FPC_POSITIVE_OVERFLOW
  *res = (*res) * (1e307 * 1e10);
#else
  *res = (*res) * (1e307 * -1e10);
#endif
}

__global__ void dot_prod(double *x, double *y, int size)
{
  double d;
  for (int i=0; i < size; ++i)
  {
    double tmp;
    mul(x[i], y[i], &tmp);
    d += tmp;
  }

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    printf("dot: %f\n", d);
  }
}
