
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


  int &s = size;

    /**
     * Contains information needed for one quadrature set direction.
     */
    struct QuadraturePoint {
      double xcos;              /* Absolute value of the x-direction cosine. */
      double ycos;              /* Absolute value of the y-direction cosine. */
      double zcos;              /* Absolute value of the z-direction cosine. */
      double w;                 /* weight for the quadrature rule.*/
      int id;                   /* direction flag (= 1 if x-direction
                                cosine is positive; = -1 if not). */
      int jd;                   /* direction flag (= 1 if y-direction
                                cosine is positive; = -1 if not). */
      int kd;                   /* direction flag (= 1 if z-direction
                                cosine is positive; = -1 if not). */
      int octant;
    };

double x1 /* x = 9 */ = 8;






 double kd;                   /* direction flag (= 1 if z-direction
                           cosine is positive; = -1 if not). */
 kd = 98.8;
 kd = kd + 1;






/*
* Comment
*
*
*
*
*/
  
}
