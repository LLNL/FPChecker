
#include <stdio.h>

__device__ void mul(double a, double b, double *res)
{
  *res = a * b;
  // NaN
  *res = (*res)-(*res) / (*res)-(*res); 
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

__host__ __device__
void calc(double *x, int s) {
  for (int i=0; i < s; ++i) {
    x[i] = x[i] + 3.1;
  }
}

__host__ 


  __device__

void calc2(double *x, int s) {
  for (int i=0; i < s; ++i) {
    x[i] = x[i] + 3.1;
  }
}

    __device__     __host__     void   comp(double *x, int s) {
  for (int i=0; i < s; ++i) {
    x[i] = x[i] + 3.1;
  }
}

  __device__    


 __host__     void   comp2(double *x, int s) {
  for (int i=0; i < s; ++i) {
    x[i] = x[i] + 3.1;
  }
}
