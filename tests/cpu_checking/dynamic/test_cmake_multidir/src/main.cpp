
#include <cstdio>
#include "util.h"
#include "server.h"

int main() {
  double x = 1.3;
  double y = 10.0;
  double tmp = compute_util(x, y);
  tmp = server_compute(tmp, y);

  printf("simple program: %f\n", tmp);

  return 0;
}
