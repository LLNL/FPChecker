
#include <stdio.h>

double compute_infinity_pos(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // Overflow
    res = res * (1e307 * 1e10);
  }
  return res;
}

double compute_infinity_neg(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
    res = res - x[i];

    // Overflow
    res = res * (1e307 * 1e10);
  }
  return res;
}

double compute_nan(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // NaN
    res = (res-res) / (res-res);
  }
  return res;
}

double compute_division_0(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // Division by zero
    res = res / (res-res);
  }
  return res;
}

double compute_cancellation(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
    //res = res + x[i];

    // Cancellation
    double tmp = x[i] - (x[i]+0.0000000001);
    res += tmp;
  }
  return res;
}

double compute_comparison(double *x, int n) {
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

double compute_underflow(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // NaN
    res = res * (1e-300 * 1e-22);
  }
  return res;
}

double compute_latent_infinity_pos(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
    //res = res + x[i];

    // Latent overflow
    res = x[i] * 1e+300;
  }
  return res;
}

double compute_latent_infinity_neg(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
    //res = res + x[i];

    // Latent overflow
    res = x[i] * -1e+300;
  }
  return res;
}

double compute_latent_underflow(double *x, int n) {
  double res = 0.0;
  for (int i=0; i < n; ++i) {
    //res = res + x[i];

    // NaN
    res = x[i] * (1e-300);
  } 
  return res;
}
