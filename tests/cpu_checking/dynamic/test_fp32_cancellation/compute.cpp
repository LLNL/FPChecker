
#include <stdio.h>

float compute(float *x, int n) {
  float res = 0.0;
  for (int i=0; i < n; ++i) {
    //res = res + x[i];

    // Cancellation
    float tmp = x[i] - (x[i]+0.0000000001);
    res += tmp;
  }
  return res;
}


