
#include <mpi.h>
#include "src/compute.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  double x=1.0;
  compute(x);
  MPI_Finalize();
  return 0;
}
