
#include <stdio.h>
#include <stdlib.h>
#include "compute.h"

int main(int argc, char **argv)
{
  int n = 8;
  int nbytes = n*sizeof(float); 
  float *data = (float *)malloc(nbytes);
  for (int i=0; i < n; ++i)
    data[i] = (float)(i+1);
  printf("Calling kernel\n");
  float result = compute(data, n);
  printf("Result: %.12g\n", result);

  return 0;
}
