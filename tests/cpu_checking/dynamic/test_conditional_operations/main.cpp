
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "compute.h"

int main(int argc, char **argv)
{
  double *data = (double *)malloc(sizeof(double) * 4);
  data[0] = 1.0;
  data[1] = 0.0;
  data[2] = 2.0;
  data[3] = 3.0;

  printf("Calling kernel\n");
  compute(data, 3);
  printf("Result: %f\n", data[0]);

  return 0;
}
