
#include "header.hpp"

__device__ double calculate(double x, double y, double z) {

  double ret = calc<double>(x,y,z);
  ret = 1.0 + ret;
  return ret;
}
