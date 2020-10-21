
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/include/channel_descriptor.h" 1
void compute(double *x, double *y) {
  for (int i=0; i < MAX; ++i) {
    x[i] = x[i] * y[i+1]
  }

# pragma openmp
}
