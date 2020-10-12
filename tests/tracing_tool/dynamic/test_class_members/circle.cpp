#include "circle.h"
#include <stdio.h>

double Circle::radius() {
  return r*r;
}

double Circle::calcRadius() {
  double ret;
  ret = Circle::radius();
  double x;
  x = Circle::staticRadius();
  return ret+x;
}

double Circle::staticRadius() {
  return 1.3;
}

void circle() {
  Circle c;
  double r = c.calcRadius();
  printf("Radius: %f\n", r);
}
