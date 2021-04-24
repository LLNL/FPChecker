
void compute(float *x, float *y, float a, int N) {
    auto r = 0;

    auto lambda = [=] __host__ __device__ (int i) {
      y[i] = a * x[i] + y[i];
    };

    auto lambda2 = [=] __host__ __device__ {
      double z=3.3;
      z = z*z;
      return y[0] + z;
    };
}

