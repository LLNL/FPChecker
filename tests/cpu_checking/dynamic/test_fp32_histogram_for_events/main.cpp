
//#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "compute.h"

int main(int argc, char **argv)
{
  int n = 3;
  int nbytes = n*sizeof(float); 
  float *data = (float *)malloc(nbytes);

  /*// Test for cancellation
  for (int i=0; i < n; ++i)
    data[i] = (float)(pow(2,-i*2));
  float result = compute_cancellation(data, n);
  printf("Cancellation: %f\n", result);
  */

  // Initialization
  for (int i=0; i < n; ++i)
    data[i] = (float)(i+1);
  
  // Test for infinity positive
  float result = compute_infinity_pos(data, n);
  printf("Infinity positive: %f\n", result);

  /*
  // Test for infinity negative
  result = compute_infinity_neg(data, n);
  printf("Infinity negative: %f\n", result);
  */

  // Test for NaN
  result = compute_nan(data, n);
  printf("Nan: %f\n", result);  

  /*
  // Test for division by zero
  result = compute_division_0(data, n);
  printf("Division by 0: %f\n", result);
  

  // Test for comparison
  result = compute_comparison(data, n);
  printf("Comparison: %f\n", result);

  // Test for underflow
  result = compute_underflow(data, n);
  printf("Underflow: %f\n", result);

  // Test for latent infinity positive
  result = compute_latent_infinity_pos(data, n);
  printf("Latent infinity positive: %f\n", result);

  // Test for latent infinity negative
  result = compute_latent_infinity_neg(data, n);
  printf("Latent infinity negative: %f\n", result);

  // Test for latent underflow
  result = compute_latent_underflow(data, n);
  printf("Latent underflow: %f\n", result);
  */

  return 0;
}
