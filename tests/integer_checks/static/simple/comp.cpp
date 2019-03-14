
int comp(int x, int y) {
   int ret = x*y;
   return ret + 2.0;
}

int foo(int x, int y, int z)
{
  int k = z + comp(x, y); 
  return k;
}
