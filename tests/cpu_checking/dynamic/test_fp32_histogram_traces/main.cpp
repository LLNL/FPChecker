
#include <stdio.h>
#include <stdlib.h>
#include "compute.h"

int main(int argc, char **argv)
{
  int n = 3;
  int nbytes = n*sizeof(float); 
  float *data = (float *)malloc(nbytes);
  // Initialize
  data[0] = 0.125;  // 0.125+0.125 = 0.25 = 1 x 2^-2
  data[1] = 0.0625; // 0.0625+0.0625 = 0.125 = 1 x 2^-3
  data[2] = 0.03125; // 0.03125+0.03125 = 0.0625 = 1 x 2^-4

  printf("Calling kernel\n");
  float result = compute(data[0], data[0]);

  printf("Result: %.16g\n", result);
  result = compute(data[1], data[1]);

  printf("Result: %.16g\n", result);

  result = compute(data[2], data[2]);
  printf("Result: %.16g\n", result);

  return 0;
}
