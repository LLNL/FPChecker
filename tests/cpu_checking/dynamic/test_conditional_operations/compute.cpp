
#include <stdio.h>

double compute(double *x, int n) {

#ifndef INVERSE
  if (x[1])
    x[0] = -1.0/x[1];
  else
    x[0] = 0.0;
#else
  if (x[1] < 1.0)
    x[0] = 0.0;
  else
    x[0] = -1.0/x[1];
#endif

  return x[0];
}


