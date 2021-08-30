
#include <stdio.h>
#include <stdlib.h>

int main() {
  double x=0;
  x = x + 1.2 * (double)(rand());
  x = x / (x-x);
  printf("x=%f\n", x);
  return 0;
}
