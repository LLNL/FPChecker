
#include <unistd.h>
#include "comp.h"
#include <stdio.h>

int main()
{
  int x=9, y=3, z=1;

  int res = foo(x, y, z);
  res++;
  int k = res - 1024;
  printf("res = %d, x: %d\n", res, k);

  return 0;
}
