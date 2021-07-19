
#include <stdio.h>
#include "header.h"

int main(int agc, char **argv) {
  double x = 1000.0;
  double y = 1e+308;
  compute(x, y);
  return 0;
}
