
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "compute.h"

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int n = 8;
  int nbytes = n*sizeof(double); 
  double *data = (double *)malloc(nbytes);
  for (int i=0; i < n; ++i)
    data[i] = (double)(i+1);
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("[Rank %d]: Calling kernel\n", rank);
  double result = compute(data, n);
  printf("Result: %f\n", result);
  MPI_Finalize();
  return 0;
}
