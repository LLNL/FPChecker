
#ifndef SRC_RUNTIME_CPU_H_
#define SRC_RUNTIME_CPU_H_

#include "FPC_Hashtable.h"
#include <stdio.h>
#include <math.h>

#ifdef FPC_MULTI_THREADED
#include <pthread.h>
#endif

#define FPC_MAX(a,b) (((a)>(b))?(a):(b))

/*----------------------------------------------------------------------------*/
/* Global data                                                                */
/*----------------------------------------------------------------------------*/

/** We store the file name and directory in this variable **/
__attribute__((used)) static char *_FPC_FILE_NAME_;

/** Hash table pointer **/
_FPC_HTABLE_T *_FPC_HTABLE_;

#ifdef FPC_DANGER_ZONE_PERCENT
#define DANGER_ZONE_PERCENTAGE FPC_DANGER_ZONE_PERCENT
#else
#define DANGER_ZONE_PERCENTAGE 0.05
#endif

#ifdef FPC_MULTI_THREADED
pthread_mutex_t fpc_lock;
#endif

/** Program name and input **/
int _FPC_PROG_INPUTS;
char ** _FPC_PROG_ARGS;

/*----------------------------------------------------------------------------*/
/* Initialize                                                                 */
/*----------------------------------------------------------------------------*/

void _FPC_INIT_HASH_TABLE_() {
  printf("#FPCHECKER: Initializing...\n");
  int64_t size = 1000;
  _FPC_HTABLE_ = _FPC_HT_CREATE_(size);

#ifdef FPC_MULTI_THREADED
  if (pthread_mutex_init(&fpc_lock, NULL) != 0) {
    printf("#FPCHECKER: Mutex init failed for multi-threading\n");
  }
#endif
}

void _FPC_INIT_FPCHECKER() {
  _FPC_PROG_INPUTS = 0;
  _FPC_INIT_HASH_TABLE_();
}

void _FPC_INIT_ARGS_FPCHECKER(int argc, char **argv) {
  _FPC_PROG_INPUTS = argc;
  _FPC_PROG_ARGS = argv;
  _FPC_INIT_HASH_TABLE_();
}

void _FPC_PRINT_LOCATIONS_()
{
  printf("#FPCHECKER: Finalizing and writing traces...\n");
  _FPC_PRINT_HASH_TABLE_(_FPC_HTABLE_);
}

/*----------------------------------------------------------------------------*/
/* Checking functions for events (FP32)                                       */
/*----------------------------------------------------------------------------*/

uint32_t _FPC_FP32_GET_EXPONENT(float x) {
  uint32_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1;   // get rid of sign bit
  val = val >> 24;  // get rid of the mantissa bits
  return val;
}

uint32_t _FPC_FP32_GET_MANTISSA(float x) {
  uint32_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 9;   // get rid of sign bit and exponent
  val = val >> 9;
  return val;
}

int _FPC_FP32_IS_INF(float x) {
  if  (_FPC_FP32_GET_EXPONENT(x) == (uint32_t)(255) &&
      _FPC_FP32_GET_MANTISSA(x) == (uint32_t)(0)
      )
    return 1;
  return 0;
}

int _FPC_FP32_IS_INFINITY_POS(float x) {
  if (_FPC_FP32_IS_INF(x))
    if (x > 0)
      return 1;
  return 0;
}

int _FPC_FP32_IS_INFINITY_NEG(float x) {
  if (_FPC_FP32_IS_INF(x))
    if (x < 0)
      return 1;
  return 0;
}

int _FPC_FP32_IS_NAN(float x) {
  if (isnan(x))
      return 1;
  return 0;
}

int _FPC_FP32_IS_DIVISON_ZERO(float y, float z, int op) {
  if (op == 3)
    if (y!=0)
      if (z==0)
        return 1;

  return 0;
}

// Number of cancelled digits calculated as:
//    max{exponent(op1), exponent(op2)} - exponent(res)
// res = result
// A cancellation has happened if the number of canceled digits 
// is greater than zero
int _FPC_FP32_IS_CANCELLATION(float x, float y, float z, int op) {
  if (op==0 || op==1) {
    uint32_t e1 = _FPC_FP32_GET_EXPONENT(y);
    uint32_t e2 = _FPC_FP32_GET_EXPONENT(z);
    uint32_t re = _FPC_FP32_GET_EXPONENT(x);
    if ((FPC_MAX((int)e1,(int)e2) - (int)re) > 30)
      return 1;
  }

  return 0;
}

int _FPC_FP32_IS_COMPARISON(int op) {
  if (op == 4)
    return 1;

  return 0;
}

int _FPC_FP32_IS_SUBNORMAL(float x) {
  int ret = 0;
  uint32_t val = _FPC_FP32_GET_EXPONENT(x);
  if (x != 0.0 && x != -0.0) {
    if (val == 0)
      ret = 1;
  }
  return ret;
}

int _FPC_FP32_IS_LATENT_INFINITY(float x) {
  int ret = 0;
  uint32_t val = _FPC_FP32_GET_EXPONENT(x);
  if (x != 0.0 && x != -0.0){
    uint64_t maxVal = 256 - (uint64_t)(DANGER_ZONE_PERCENTAGE*256.0);
    if (val >= maxVal)
      ret = 1;
  }
  return ret;
}

int _FPC_FP32_IS_LATENT_INFINITY_POS(float x) {
  if (_FPC_FP32_IS_LATENT_INFINITY(x))
    if (x > 0)
      return 1;

  return 0;
}

int _FPC_FP32_IS_LATENT_INFINITY_NEG(float x) {
  if (_FPC_FP32_IS_LATENT_INFINITY(x))
    if (x < 0)
      return 1;

  return 0;
}

int _FPC_FP32_IS_LATENT_SUBNORMAL(float x) {
  int ret = 0;
  uint32_t val = _FPC_FP32_GET_EXPONENT(x);
  if (x != 0.0 && x != -0.0) {
    uint64_t minVal = (uint64_t)(DANGER_ZONE_PERCENTAGE*256.0);
    if (val <= minVal)
      ret = 1;
  }
  return ret;
}

/*----------------------------------------------------------------------------*/
/* Checking functions for events (FP64)                                       */
/*----------------------------------------------------------------------------*/

uint64_t _FPC_FP64_GET_EXPONENT(double x) {
  uint64_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1;   // get rid of sign bit
  val = val >> 53;  // get rid of the mantissa bits
  return val;
}

uint64_t _FPC_FP64_GET_MANTISSA(double x) {
  uint64_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 12;   // get rid of sign bit and exponent
  val = val >> 12;
  return val;
}

int _FPC_FP64_IS_INF(double x) {
  if  (_FPC_FP64_GET_EXPONENT(x) == (uint64_t)(2047) &&
      _FPC_FP64_GET_MANTISSA(x) == (uint64_t)(0)
      )
    return 1;
  return 0;
}

int _FPC_FP64_IS_INFINITY_POS(double x) {
  if (_FPC_FP64_IS_INF(x))
    if (x > 0)
      return 1;
  return 0;
}

int _FPC_FP64_IS_INFINITY_NEG(double x) {
  if (_FPC_FP64_IS_INF(x))
    if (x < 0)
      return 1;
  return 0;
}

int _FPC_FP64_IS_NAN(double x) {
  if (isnan(x))
      return 1;
  return 0;
}

int _FPC_FP64_IS_DIVISON_ZERO(double y, double z, int op) {
  if (op == 3)
    if (y!=0)
      if (z==0)
        return 1;

  return 0;
}

// Number of cancelled digits calculated as:
//    max{exponent(op1), exponent(op2)} - exponent(res)
// res = result
// A cancellation has happened if the number of canceled digits 
// is greater than zero
// Threshold: 10^9 or 2^30, i.e., 9 decimal digits or 30 binary digits
int _FPC_FP64_IS_CANCELLATION(double x, double y, double z, int op) {
  if (op==0 || op==1) {
    uint64_t e1 = _FPC_FP64_GET_EXPONENT(y);
    uint64_t e2 = _FPC_FP64_GET_EXPONENT(z);
    uint64_t re = _FPC_FP64_GET_EXPONENT(x);
    if ((FPC_MAX((int)e1,(int)e2) - (int)re) > 30) {
      return 1;
    }
  }

  return 0;
}

int _FPC_FP64_IS_COMPARISON(int op) {
  if (op == 4)
    return 1;

  return 0;
}

int _FPC_FP64_IS_SUBNORMAL(double x)
{
  int ret = 0;
  uint64_t val = _FPC_FP64_GET_EXPONENT(x);
  //memcpy((void *) &val, (void *) &x, sizeof(val));
  //val = val << 1;   // get rid of sign bit
  //val = val >> 53;  // get rid of the mantissa bits
  if (x != 0.0 && x != -0.0)
  {
    if (val == 0)
      ret = 1;
  }
  return ret;
}

int _FPC_FP64_IS_LATENT_INFINITY(double x)
{
  int ret = 0;
  uint64_t val = _FPC_FP64_GET_EXPONENT(x);
  if (x != 0.0 && x != -0.0) {
    uint64_t maxVal = 2048 - (uint64_t)(DANGER_ZONE_PERCENTAGE*2048.0);
    if (val >= maxVal)
      ret = 1;
  }
  return ret;
}

int _FPC_FP64_IS_LATENT_INFINITY_POS(double x) {
  if (_FPC_FP64_IS_LATENT_INFINITY(x))
    if (x > 0)
      return 1;

  return 0;
}

int _FPC_FP64_IS_LATENT_INFINITY_NEG(double x) {
  if (_FPC_FP64_IS_LATENT_INFINITY(x))
    if (x < 0)
      return 1;

  return 0;
}

int _FPC_FP64_IS_LATENT_SUBNORMAL(double x) {
  int ret = 0;
  uint64_t val = _FPC_FP64_GET_EXPONENT(x);
  if (x != 0.0 && x != -0.0) {
    uint64_t minVal = (uint64_t)(DANGER_ZONE_PERCENTAGE*2048.0);
    if (val <= minVal)
      ret = 1;
  }
  return ret;
}

/*----------------------------------------------------------------------------*/
/* Generic checking functions                                                 */
/*----------------------------------------------------------------------------*/

int _FPC_EVENT_OCURRED(_FPC_ITEM_T_ *item) {
  return (
      item->infinity_pos ||
      item->infinity_neg ||
      item->nan ||
      item->division_zero ||
      item->cancellation ||
      item->comparison ||
      item->underflow ||
      item->latent_infinity_pos ||
      item->latent_infinity_neg ||
      item->latent_underflow
      );
}


/**
 * Operations table
 * -------------------------
 * ADD = 0
 * SUB = 1
 * MUL = 2
 * DIV = 3
 * CMP = 4 (comparison)
 * REM = 5 (reminder)
 * CALL = 6 (function call)
 * -------------------------
 **/

void _FPC_FP32_CHECK_(
    float x, float y, float z, int loc, char *file_name, int op) {
  _FPC_ITEM_T_ item;
  // Set file name and line
  item.file_name = file_name;
  item.line = (uint64_t)loc;

  // Set events
  item.infinity_pos         = (uint64_t)_FPC_FP32_IS_INFINITY_POS(x);
  item.infinity_neg         = (uint64_t)_FPC_FP32_IS_INFINITY_NEG(x);
  item.nan                  = (uint64_t)_FPC_FP32_IS_NAN(x);
  item.division_zero        = (uint64_t)_FPC_FP32_IS_DIVISON_ZERO(y, z, op);
  item.cancellation         = (uint64_t)_FPC_FP32_IS_CANCELLATION(x, y, z, op);
  item.comparison           = (uint64_t)_FPC_FP32_IS_COMPARISON(op);
  item.underflow            = (uint64_t)_FPC_FP32_IS_SUBNORMAL(x);
  item.latent_infinity_pos  = (uint64_t)_FPC_FP32_IS_LATENT_INFINITY_POS(x);
  item.latent_infinity_neg  = (uint64_t)_FPC_FP32_IS_LATENT_INFINITY_NEG(x);
  item.latent_underflow     = (uint64_t)_FPC_FP32_IS_LATENT_SUBNORMAL(x);

 if (_FPC_EVENT_OCURRED(&item)) {
#ifdef FPC_MULTI_THREADED
    pthread_mutex_lock(&fpc_lock);
#endif
    _FPC_HT_SET_(_FPC_HTABLE_, &item);
#ifdef FPC_MULTI_THREADED
    pthread_mutex_unlock(&fpc_lock);
#endif
  }
}

void _FPC_FP64_CHECK_(
    double x, double y, double z, int loc, char *file_name, int op) {
  _FPC_ITEM_T_ item;
  // Set file name and line
  item.file_name = file_name;
  item.line = (uint64_t)loc;

  // Set events
  item.infinity_pos         = (uint64_t)_FPC_FP64_IS_INFINITY_POS(x);
  item.infinity_neg         = (uint64_t)_FPC_FP64_IS_INFINITY_NEG(x);
  item.nan                  = (uint64_t)_FPC_FP64_IS_NAN(x);
  item.division_zero        = (uint64_t)_FPC_FP64_IS_DIVISON_ZERO(y, z, op);
  item.cancellation         = (uint64_t)_FPC_FP64_IS_CANCELLATION(x, y, z, op);
  item.comparison           = (uint64_t)_FPC_FP64_IS_COMPARISON(op);
  item.underflow            = (uint64_t)_FPC_FP64_IS_SUBNORMAL(x);
  item.latent_infinity_pos  = (uint64_t)_FPC_FP64_IS_LATENT_INFINITY_POS(x);
  item.latent_infinity_neg  = (uint64_t)_FPC_FP64_IS_LATENT_INFINITY_NEG(x);
  item.latent_underflow     = (uint64_t)_FPC_FP64_IS_LATENT_SUBNORMAL(x);

   if (_FPC_EVENT_OCURRED(&item)) {
#ifdef FPC_MULTI_THREADED
    pthread_mutex_lock(&fpc_lock);
#endif
    _FPC_HT_SET_(_FPC_HTABLE_, &item);
#ifdef FPC_MULTI_THREADED
    pthread_mutex_unlock(&fpc_lock);
#endif
  }
}


#endif /* SRC_RUNTIME_CPU_H_ */
