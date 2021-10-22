
#include <stdio.h>

double compute(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // NaN
    res = (res-res) / (res-res);
  }
  return res;
}
