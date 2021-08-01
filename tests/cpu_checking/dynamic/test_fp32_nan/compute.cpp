
#include <stdio.h>

float compute(float *x, int n) {
  float res = 0.0f;
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // NaN
    res = (res-res) / (res-res);
  }
  return res;
}


