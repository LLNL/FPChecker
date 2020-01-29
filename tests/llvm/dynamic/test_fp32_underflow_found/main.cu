
#include <stdio.h>
#include <stdlib.h>
#include "dot_product.h"

int main(int argc, char **argv)
{
  int n = 3;
  int nbytes = n*sizeof(float); 
  float *d_a = 0;
  cudaMalloc(&d_a, nbytes);

  float *data = (float *)malloc(nbytes);
  for (int i=0; i < n; ++i)
  {
    data[i] = (float)(i+1);
  }

  cudaMemcpy((void *)d_a, (void *)data, nbytes, cudaMemcpyHostToDevice);

  printf("Calling kernel\n");
  dot_prod<<<1,1>>>(d_a, d_a, nbytes);
  cudaDeviceSynchronize();
  printf("done\n");

  return 0;
}
