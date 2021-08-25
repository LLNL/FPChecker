
#include <stdio.h>
#include <string>

double compute(double *x, int n) {
  double res = 0.0;
  std::string s(COMPILE_STRING);
  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // NaN
    res = (res-res) / (res-res);
  }
  printf("string: %s\n", s.c_str());
  return res;
}


