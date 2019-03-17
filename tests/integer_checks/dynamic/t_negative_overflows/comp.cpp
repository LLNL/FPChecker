
#include <limits.h>

int comp(int x, int y) {
   int ret = x*y;
   ret = INT_MIN - ret;
   return ret + 2;
}

int foo(int x, int y, int z)
{
  int k = z + comp(x, y); 
  return k;
}
