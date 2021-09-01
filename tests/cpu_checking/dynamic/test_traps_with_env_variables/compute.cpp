
#include <stdio.h>

double compute(double *x, int n) {
  double res_1 = 0.0;
  double res_2 = 0.0;
  for (int i=0; i < n; ++i) {
    res_1 += x[i];
    res_2 += x[i];

    // Overflow
    res_2 = res_2 * (1e307 * 1e10);

    // NaN
    res_1 = (res_1-res_1) / (res_1-res_1);
  }
  return (res_1 - res_2);
}


