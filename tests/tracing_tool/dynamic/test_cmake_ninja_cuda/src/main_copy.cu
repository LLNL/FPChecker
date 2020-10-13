
#include <stdio.h>
#include <stdlib.h>
#include "compute.h"

int main(int argc, char **argv)
{
  int n = 3;
  int nbytes = n*sizeof(double); 
  double *d_a = 0;
  cudaMalloc(&d_a, nbytes);

  double *data = (double *)malloc(nbytes);
  for (int i=0; i < n; ++i)
  {
    data[i] = _FPC_CHECK_((double)(i+1), 16, "../src/main_copy.cu");
  }

  cudaMemcpy((void *)d_a, (void *)data, nbytes, cudaMemcpyHostToDevice);

  printf("Calling kernel\n");
  compute<<<16,16>>>(d_a, d_a, nbytes);
  cudaDeviceSynchronize();
  printf("done\n");

  return 0;
}
