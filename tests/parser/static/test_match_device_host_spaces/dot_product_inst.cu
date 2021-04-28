
#include <stdio.h>

__device__ void mul(double a, double b, double *res)
{
  *res = _FPC_CHECK_D_(a * b, 6, "/usr/WS1/laguna/fpchecker/FPChecker/tests/parser/static/test_match_device_host_spaces/dot_product.cu");
  // NaN
  *res = _FPC_CHECK_D_((*res)-(*res) / (*res)-(*res), 8, "/usr/WS1/laguna/fpchecker/FPChecker/tests/parser/static/test_match_device_host_spaces/dot_product.cu");
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

/*__host__ __device__
void calc(double *x, int s) {
  for (int i=0; i < s; ++i) {
    x[i] = x[i] + 3.1;
  }
}*/

    __device__     __host__     void   comp(double *x, int s) {
  for (int i=0; i < s; ++i) {
    x[i] = _FPC_CHECK_HD_(x[i] + 3.1, 36, "/usr/WS1/laguna/fpchecker/FPChecker/tests/parser/static/test_match_device_host_spaces/dot_product.cu");
  }
}

  __device__    


 __host__     void   comp2(double *x, int s) {
  for (int i=0; i < s; ++i) {
    x[i] = _FPC_CHECK_HD_(x[i] + 3.1, 45, "/usr/WS1/laguna/fpchecker/FPChecker/tests/parser/static/test_match_device_host_spaces/dot_product.cu");
  }
}


