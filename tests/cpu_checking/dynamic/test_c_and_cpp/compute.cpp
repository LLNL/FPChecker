
#include "compute.h"
#include "compute_2.h"
#include <stdio.h>

double compute(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
    res = res + x[i];
    res = mult(res, res);

    // NaN
    res = (res-res) / (res-res);
  }
  return res;
}


