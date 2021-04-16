
#include "simple.hpp"

namespace MySpace {

__device__ void compute(Real_ptr x) {
  x[0] = _FPC_CHECK_D_(x[1] * x[2], 7, "simple.cu");
}

}
