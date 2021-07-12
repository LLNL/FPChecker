
#ifndef HEADER_H_
#define HEADER_H_

#include <stdio.h>

void compute(double x, double y) {
  double tmp;
  tmp = x * y;
  tmp += 3.0;
  printf("result = %f\n", tmp);
}

#endif
