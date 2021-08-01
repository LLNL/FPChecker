
#include <stdio.h>

float compute(float *x, int n) {
  float res = 0.0f;
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // subnormal
    res = res * (1e-38f * 1e-5f); 
  }
  return res;
}


