
#include <stdio.h>

float compute(float *x, int n) {
  float res = 0.0f;
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // Comparisons
    if (res == 0.0f)
      res += 1.0f;
    if (res < x[i])
      res -= 1.0f;
    if (res <= x[i])
      res -= 1.0f;
    if (res > x[i])
      res -= 1.0f;
    if (res >= x[i])
      res -= 1.0f;
  }
  return res;
}


