
#include <unistd.h>
#include "comp.h"
#include <stdio.h>

int main()
{
  int x=9, y=3, z=1;

  int res = foo(x, y, z);
  res++;
  printf("res = %d\n", res);

  return 0;
}
