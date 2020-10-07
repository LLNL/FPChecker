
#ifndef SRC_RUNTIME_H_
#define SRC_RUNTIME_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>


//#pragma hd_warning_disable
/*#ifdef __CUDA_ARCH__
template<typename T>
__device__
T _FPC_CHECK_(T t, int x, const char *str) {
  return t;
}
#else
template<typename T>
__host__
T _FPC_CHECK_(T t, int x, const char *str) {
  return t;
}
#endif
*/

//#ifdef __CUDA_ARCH__
//#pragma hd_warning_disable
//template<typename T>
//__host__ __device__
//T _FPC_CHECK_(T t, int x, const char *str) {
//  return t;
//}
//#else
//#define _FPC_CHECK_(f, x, str) (f)
//#endif

/*#pragma hd_warning_disable
template<typename T>
__host__ __device__
T _FPC_CHECK_(T t, int x, const char *str) {
#ifndef __CUDA_ARCH__
  return t;
#else
  return t;
#endif
}*/

//#pragma hd_warning_disable
template<typename T>
__device__
T _FPC_CHECK_DEVICE_(T t, int x, const char *str) {
  //printf("--From device--\n");
  return t;
}

//#pragma hd_warning_disable
template<typename T>
__host__
T _FPC_CHECK_HOST_(T t, int x, const char *str) {
  //printf("--From device--\n");
  return t;
}

#if defined(__CUDA_ARCH__) and defined(__NVCC__)
#define _FPC_CHECK_(f, x, str) _FPC_CHECK_DEVICE_(f, x, str)
#else
#define _FPC_CHECK_(f, x, str) _FPC_CHECK_HOST_(f, x, str)
#endif

/*#ifdef __CUDA_ARCH__
template<typename T>
__noinline__
__host__ __device__
T _FPC_CHECK_(T t, int x, const char *str) {
  printf("--From device--\n");
  return t;
}
#endif
*/

#endif /* SRC_RUNTIME_H_ */
