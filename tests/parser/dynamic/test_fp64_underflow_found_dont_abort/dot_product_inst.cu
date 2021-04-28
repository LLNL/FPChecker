
#include <stdio.h>

__device__ void mul(double a, double b, double *res)
{
  *res = _FPC_CHECK_D_(a * b, 6, "dot_product.cu");
  // underflow
  *res = _FPC_CHECK_D_((*res) * (1e-300 * 1e-22), 8, "dot_product.cu");
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
