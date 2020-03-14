
#include <stdio.h>

__device__
double power(double x);

__global__
void compute(double x);
