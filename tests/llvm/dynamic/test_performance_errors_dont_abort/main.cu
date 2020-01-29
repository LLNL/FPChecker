
#include <stdio.h>
#include <stdlib.h>
#include "arr_sum.h"

int main(int argc, char **argv)
{
  if (argc==1)
	printf("main (n elems) (kernel runs) (block size)\n");

  int n 	= atoi(argv[1]);
  int k_runs 	= atoi(argv[2]);
  int b_size	= atoi(argv[3]);

  for (int i=0; i < k_runs; ++i)
  {
    int nbytes = n*sizeof(double); 
    double *d_a = 0;
    cudaMalloc(&d_a, nbytes);

    double *data = (double *)malloc(nbytes);
    for (int i=0; i < n; ++i)
    {
      data[i] = (double)(i+1);
    }
    cudaMemcpy((void *)d_a, (void *)data, nbytes, cudaMemcpyHostToDevice);
    //printf("Calling kernel\n");
    array_sum<<<b_size,b_size>>>(d_a, d_a);
    cudaDeviceSynchronize();
    //printf("done\n");
    printf(".");
  }
  printf("\ndone\n");

  return 0;
}
