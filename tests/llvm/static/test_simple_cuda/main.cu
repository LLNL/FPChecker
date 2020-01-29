
#include <stdio.h>
#include "compute.h"

int main(int argc, char **argv)
{
  printf("Calling kernel\n");
  double x = atof(argv[1]);
  compute<<<128,128>>>(x);
  cudaDeviceSynchronize();
  printf("done");

  return 0;
}
