#include <stdio.h>
#include <stdlib.h>
#include "dot_product.h"
#include "cuda_launch.h"

void launch()
{ 
  int n = 3;
  int nbytes = n*sizeof(double); 
  double *d_a = 0;
  cudaMalloc(&d_a, nbytes);

  double *data = (double *)malloc(nbytes);
  for (int i=0; i < n; ++i)
  {
    data[i] = (double)(i+1);
  }

  cudaMemcpy((void *)d_a, (void *)data, nbytes, cudaMemcpyHostToDevice);

  printf("Calling kernel\n");
  dot_prod<<<16,16>>>(d_a, d_a, nbytes);
  cudaDeviceSynchronize();
  printf("done\n");
}
