
#include <stdio.h>

double compute(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
    //res = res + x[i];

    // Cancellation
    double tmp = x[i] - (x[i]+0.0000000001);
    res += tmp;
  }
  return res;
}


