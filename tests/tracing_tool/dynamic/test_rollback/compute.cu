
#include "compute.h"

#ifdef __NVCC__
void compute() {
  return;
}
#else // clang
__host__ __device__ void compute() {}
__device__ void compute() {}
#endif

