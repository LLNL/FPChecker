
#include <stdio.h>
#include <stdlib.h>
#include "compute.h"

int main(int argc, char **argv)
{
  int n = 4;
  int nbytes = n*sizeof(double); 
  double *data = (double *)malloc(nbytes);
  // Initialize
  data[0] = 1e-10;
  data[1] = 1e-50;
  data[2] = 1e-100;
  data[3] = 1e-200;

  printf("Calling kernel\n");
  double result = compute(data, n);
  printf("Result: %f\n", result);

  return 0;
}
