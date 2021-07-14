
#include <stdio.h>

double compute(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
#ifdef FPC_POSITIVE_OVERFLOW
    res = res + x[i];
#else
    res = res - x[i];
#endif

    // Overflow
    res = res * (1e307 * 1e10);
  }
  return res;
}


