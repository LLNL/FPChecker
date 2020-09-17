
#include <stdio.h>
#include <stdlib.h>
#include "mpi_stuff.h"
#include "cuda_launch.h"

int main(int argc, char **argv)
{
  
  initMPI(&argc, &argv);

  launch();

  finalizeMPI();

  return 0;
}
