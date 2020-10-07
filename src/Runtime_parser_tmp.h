/*
 * Runtime.h
 *
 *  Created on: Apr 10, 2018
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#ifndef SRC_RUNTIME_H_
#define SRC_RUNTIME_H_

#ifdef __CUDA_ARCH__
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

#endif
