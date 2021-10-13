
#include <stdio.h>

float compute_infinity_pos(float *x, int n) {
  float res = 0.0;
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // Overflow
    res = res * (1e307 * 1e10);
  }
  return res;
}

float compute_infinity_neg(float *x, int n) {
  float res = 0.0;
  for (int i=0; i < n; ++i) {
    res = res - x[i];

    // Overflow
    res = res * (1e307 * 1e10);
  }
  return res;
}

float compute_nan(float *x, int n) {
  float res = 0.0;
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // NaN
    res = (res-res) / (res-res);
  }
  return res;
}

float compute_division_0(float *x, int n) {
  float res = 0.0;
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // Division by zero
    res = res / (res-res);
  }
  return res;
}

float compute_cancellation(float *x, int n) {
  float res = 0.0;
  for (int i=0; i < n; ++i) {
    //res = res + x[i];

    // Cancellation
    float tmp = x[i] - (x[i]+0.0000000001);
    res += tmp;
  }
  return res;
}

float compute_comparison(float *x, int n) {
  float res = 0.0;
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

float compute_underflow(float *x, int n) {
  float res = 0.0;
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // NaN
    res = res * (1e-300 * 1e-22);
  }
  return res;
}

float compute_latent_infinity_pos(float *x, int n) {
  float res = 0.0;
  for (int i=0; i < n; ++i) {
    //res = res + x[i];

    // Latent overflow
    res = x[i] * 1e+300;
  }
  return res;
}

float compute_latent_infinity_neg(float *x, int n) {
  float res = 0.0;
  for (int i=0; i < n; ++i) {
    //res = res + x[i];

    // Latent overflow
    res = x[i] * -1e+300;
  }
  return res;
}

float compute_latent_underflow(float *x, int n) {
  float res = 0.0;
  for (int i=0; i < n; ++i) {
    //res = res + x[i];

    // NaN
    res = x[i] * (1e-300);
  } 
  return res;
}
