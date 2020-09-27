
#include <stdio.h>
#include "compute.h"

int main() {

  double x = 8.0;
  double y = 16.0;
  y = compute_(&x, &y);
  printf("y = %f\n", y);

  return 0;
}
