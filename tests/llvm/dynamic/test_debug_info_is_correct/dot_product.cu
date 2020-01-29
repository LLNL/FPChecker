
#include <stdio.h>

/* Lorem ipsum lorem cubilia orci cursus nec elementum gravida sociosqu, ad rutrum diam tempor dictumst nostra scelerisque litora, turpis phasellus nunc sagittis nisi ipsum suspendisse ipsum.

Congue justo sociosqu ad posuere proin euismod platea, dui netus conubia hac leo pretium libero, ad cras tortor laoreet lacus euismod mauris mattis luctus donec integer molestie malesuada mi dapibus curabitur aliquam. */

__device__ void mul(double a, double b, double *res)
{

/*
Lorem ipsum lorem cubilia orci cursus nec elementum gravida sociosqu, ad rutrum diam tempor dictumst nostra scelerisque litora, turpis phasellus nunc sagittis nisi ipsum suspendisse ipsum.

Congue justo sociosqu ad posuere proin euismod platea, dui netus conubia hac leo pretium libero, ad cras tortor laoreet lacus euismod mauris mattis luctus donec integer molestie malesuada mi dapibus curabitur aliquam.
*/

  *res = a * b;
  // Overflow
  *res = (*res) * (1e307 * 1e10);
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
