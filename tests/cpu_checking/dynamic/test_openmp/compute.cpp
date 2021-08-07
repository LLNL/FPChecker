
#include <stdio.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

double compute(double *x, int n) {
  double res = 0.0;

#if defined(_OPENMP)
  omp_set_num_threads(4);
#endif

#pragma omp parallel
  {

#if defined(_OPENMP)
  int num = omp_get_thread_num();
  printf("Thread num: %d\n", num);
#endif

  for (int i=0; i < n; ++i) {
    res = res + x[i];

    // NaN
    res = (res-res) / (res-res);
  }
  }
  return res;
}


