

#include "compute.h"
#include <mpi.h>
#include <stdio.h>

void compute(double x) {

  double recbf;
  MPI_Reduce(&x, &recbf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    printf("Sum: %f\n", recbf);

  for (int i; i < 10; ++i) {
    printf("%d\n", recbf);
  }
}
