
#define min(X, Y)  ((X) < (Y) ? \
                    (X) : \
                    (Y))
#define Real_t double

__host__ void compute(Real_t a) {
  Real_t x = a * a + 1.4;

}

__device__ double comp_dev(double *x, double *y) {
  y[0] = 7.7 - y[0] + 1.3; y[1] += y[1];

  x[0] = x[1]+
      // this is a comment
            1.3;
  x[1] = min(x[0], 
                  x[1]
                      );

  int a=1;
  int b=2;
  auto l = [=]() { return a*b; };

  return x[0]; 
}

__device__ double foo_dev(double *x, double *rr) {
  x[0] = rr[1] 
            + 
              1.3;  

  if (x[1] < x[2])
    return 5.9;  
}



static int var1;
static int var2;
