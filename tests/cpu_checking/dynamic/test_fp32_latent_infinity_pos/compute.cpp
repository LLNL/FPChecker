
#include <stdio.h>

float compute(float *x, int n) {
  float res = 0.0f;
  for (int i=0; i < n; ++i) {
    //res = res + x[i];

    // Latent overflow
    res = x[i] * 1e+36f;
  }
  return res;
}


