
int comp(int x, int y) {
   int ret = x*y;
   ret = ret / 3 * 3 * 3 + 4;
   return ret + 2;
}

int foo(int x, int y, int z)
{
  int k = z + comp(x, y); 
  return k;
}
