
#include <stdio.h>

int main(int argc, char *argv[]) {
  double x=0;
  x = x + 1.2 * (double)(argc);
  x = x / (x-x);
  printf("x=%f\n", x);
  return 0;
}
