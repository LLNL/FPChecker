
#include <stdio.h>

__attribute__((device)) void mul(double a, double b, double *res)
{
  *res = _FPC_CHECK_D_(a * b, 6, "/usr/WS1/laguna/fpchecker/FPChecker/tests/parser/static/test_match_device_host_spaces/dot_product.cu");

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

    __attribute__((device)) __attribute__((host)) void comp(double *x, int s) {
  for (int i=0; i < s; ++i) {
    x[i] = _FPC_CHECK_HD_(x[i] + 3.1, 36, "/usr/WS1/laguna/fpchecker/FPChecker/tests/parser/static/test_match_device_host_spaces/dot_product.cu");
  }
}

  __attribute__((device))


 __attribute__((host)) void comp2(double *x, int s) {
  for (int i=0; i < s; ++i) {
    x[i] = _FPC_CHECK_HD_(x[i] + 3.1, 45, "/usr/WS1/laguna/fpchecker/FPChecker/tests/parser/static/test_match_device_host_spaces/dot_product.cu");
  }
}


