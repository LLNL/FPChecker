
#include <stdio.h>

double compute(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // Comparisons
    if (res == 0.0)
      res += 1.0;
    if (res < x[i])
      res -= 1.0;
    if (res <= x[i])
      res -= 1.0;
    if (res > x[i])
      res -= 1.0;
    if (res >= x[i])
      res -= 1.0;
  }
  return res;
}


