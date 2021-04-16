
#include "simple.hpp"

namespace MySpace {

__device__ void compute(Real_ptr x) {
  x[0] = x[1] * x[2];
}

}
