
#include <stdio.h>
#include <stdlib.h>
#include "compute.h"

int main(int argc, char **argv)
{
  int n = 8;
  int nbytes = n*sizeof(double); 
  double *data = (double *)malloc(nbytes);
  for (int i=0; i < n; ++i)
    data[i] = (double)(i+1);
  printf("Calling kernel\n");
  double result = compute(data, n);
  printf("Result: %f\n", result);

  return 0;
}
