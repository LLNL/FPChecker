
#include <stdio.h>

__device__ void mul(double a, double b, double *res)
{
  *res = _FPC_CHECK_(a * b, 6, "dot_product_copy.cu");
  // NaN
  *res = _FPC_CHECK_((*res)-(*res) / (*res)-(*res), 8, "dot_product_copy.cu"); 
}

__global__ void dot_prod(double *x, double *y, int size)
{
  double d;
  for (int i=0; i < size; ++i)
  {
    double tmp;
    mul(x[i], y[i], &tmp);
    d += _FPC_CHECK_(tmp, 18, "dot_product_copy.cu");
  }

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    printf("dot: %f\n", d);
  }
}
