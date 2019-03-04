#include <cstdlib>
#include <cstring>
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

const int CUDA_BLOCK_SIZE = 16;

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
  std::cout << "RAJA vector dot product example.\n";
  const int N = 1000000;

  // Allocate and initialize vector data
  int *a = memoryManager::allocate<int>(N);
  int *b = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    a[i] = 1.0;
    b[i] = 1.0;
  }

  double dot = 0.0;

  RAJA::ReduceSum<RAJA::cuda_reduce, double> cudot(0.0);
  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) { 
    cudot += a[i] * b[i];
    // ---- injected underflow --------
    cudot += a[i] * (1e-300 * 1e-20); 
    // --------------------------------
  });    

  cudaDeviceSynchronize();
  dot = cudot.get();
  std::cout << "(a, b) = " << dot << std::endl;

  memoryManager::deallocate(a);
  memoryManager::deallocate(b);

  std::cout << "DONE!...\n";

  return 0;
}


