
#include <stdio.h>

__device__ void mul(double a, double b, double *res)
{
  *res = a * b;

#ifdef FPC_ERROR_ADD
  *res = *res + 1e308;
#endif

#ifdef FPC_ERROR_SUB
  *res = *res - (1e308);
#endif

#ifdef FPC_ERROR_MUL
  *res = *res * 1e-323;
#endif

#ifdef FPC_ERROR_DIV
  *res = (*res - *res) / (*res - *res);
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
