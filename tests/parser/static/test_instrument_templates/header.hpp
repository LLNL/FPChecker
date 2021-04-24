
template<typename T>
__device__
double calc(T x, T y, T z) {
  T ret  = x*y;
  ret = ret / z;
  return ret;
}

