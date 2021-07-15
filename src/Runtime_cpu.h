
#ifndef SRC_RUNTIME_CPU_H_
#define SRC_RUNTIME_CPU_H_

#include "FPC_Hashtable.h"
#include <stdio.h>
#include <math.h>

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


/*----------------------------------------------------------------------------*/
/* Initialize                                                                 */
/*----------------------------------------------------------------------------*/

void _FPC_INIT_HASH_TABLE_() {
  int64_t size = 1000;
  _FPC_HTABLE_ = _FPC_HT_CREATE_(size);
  printf("#FPCHECKER: Initializing...\n");
}

void _FPC_PRINT_LOCATIONS_()
{
  _FPC_PRINT_HASH_TABLE_(_FPC_HTABLE_);
}

/*----------------------------------------------------------------------------*/
/* Checking functions for events (FP32)                                       */
/*----------------------------------------------------------------------------*/

int _FPC_FP32_IS_INFINITY_POS(float x) {
  if (isinf(x))
    if (x > 0)
      return 1;
  return 0;
}

int _FPC_FP32_IS_INFINITY_NEG(float x) {
  if (isinf(x))
    if (x < 0)
      return 1;
  return 0;
}

int _FPC_FP32_IS_NAN(float x) {
  if (isnan(x))
      return 1;
  return 0;
}

int _FPC_FP32_IS_DIVISON_ZERO(float x, float y, float z, int op) {
  if (op == 3)
    if (y!=0)
      if (z==0)
        return 1;

  return 0;
}

int _FPC_FP32_IS_CANCELLATION(float x, float y, float z, int op) {

  return 0;
}

int _FPC_FP32_IS_COMPARISON(float x, float y, float z, int op) {
  if (op == 4)
    return 1;

  return 0;
}

int _FPC_FP32_IS_SUBNORMAL(float x) {
  int ret = 0;
  uint32_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1;   // get rid of sign bit
  val = val >> 24;  // get rid of the mantissa bits
  if (x != 0.0 && x != -0.0)
  {
    if (val == 0)
      ret = 1;
  }
  return ret;
}

int _FPC_FP32_IS_LATENT_INFINITY(float x) {
  int ret = 0;
  uint32_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1;   // get rid of sign bit
  val = val >> 24;  // get rid of the mantissa bits
  if (x != 0.0 && x != -0.0)
  {
    int maxVal = 256 - (int)(DANGER_ZONE_PERCENTAGE*256.0);
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
  uint32_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1;   // get rid of sign bit
  val = val >> 24;  // get rid of the mantissa bits
  if (x != 0.0 && x != -0.0)
  {
    int minVal = (int)(DANGER_ZONE_PERCENTAGE*256.0);
    if (val <= minVal)
      ret = 1;
  }
  return ret;
}

/*----------------------------------------------------------------------------*/
/* Checking functions for events (FP64)                                       */
/*----------------------------------------------------------------------------*/

int _FPC_FP64_IS_INFINITY_POS(double x) {
  if (isinf(x))
    if (x > 0)
      return 1;
  return 0;
}

int _FPC_FP64_IS_INFINITY_NEG(double x) {
  if (isinf(x))
    if (x < 0)
      return 1;
  return 0;
}

int _FPC_FP64_IS_NAN(double x) {
  if (isnan(x))
      return 1;
  return 0;
}

int _FPC_FP64_IS_DIVISON_ZERO(double x, double y, double z, int op) {
  if (op == 3)
    if (y!=0)
      if (z==0)
        return 1;

  return 0;
}

int _FPC_FP64_IS_CANCELLATION(double x, double y, double z, int op) {

  return 0;
}

int _FPC_FP64_IS_COMPARISON(double x, double y, double z, int op) {
  if (op == 4)
    return 1;

  return 0;
}

int _FPC_FP64_IS_SUBNORMAL(double x)
{
  int ret = 0;
  uint64_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1;   // get rid of sign bit
  val = val >> 53;  // get rid of the mantissa bits
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
  uint64_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1;   // get rid of sign bit
  val = val >> 53;  // get rid of the mantissa bits
  if (x != 0.0 && x != -0.0)
  {
    int maxVal = 2048 - (int)(DANGER_ZONE_PERCENTAGE*2048.0);
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
  uint64_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1;   // get rid of sign bit
  val = val >> 53;  // get rid of the mantissa bits
  if (x != 0.0 && x != -0.0)
  {
    int minVal = (int)(DANGER_ZONE_PERCENTAGE*2048.0);
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
  item.line = loc;

  // Set events
  item.infinity_pos         = _FPC_FP32_IS_INFINITY_POS(x);
  item.infinity_neg         = _FPC_FP32_IS_INFINITY_NEG(x);
  item.nan                  = _FPC_FP32_IS_NAN(x);
  item.division_zero        = _FPC_FP32_IS_DIVISON_ZERO(x, y, z, op);
  item.cancellation         = _FPC_FP32_IS_CANCELLATION(x, y, z, op);
  item.comparison           = _FPC_FP32_IS_COMPARISON(x, y, z, op);
  item.underflow            = _FPC_FP32_IS_SUBNORMAL(x);
  item.latent_infinity_pos  = _FPC_FP32_IS_LATENT_INFINITY_POS(x);
  item.latent_infinity_neg  = _FPC_FP32_IS_LATENT_INFINITY_NEG(x);
  item.latent_underflow     = _FPC_FP32_IS_LATENT_SUBNORMAL(x);

  if (_FPC_EVENT_OCURRED(&item))
    _FPC_HT_SET_(_FPC_HTABLE_, &item);
}

void _FPC_FP64_CHECK_(
    double x, double y, double z, int loc, char *file_name, int op) {
  _FPC_ITEM_T_ item;
  // Set file name and line
  item.file_name = file_name;
  item.line = loc;

  // Set events
  item.infinity_pos         = _FPC_FP64_IS_INFINITY_POS(x);
  item.infinity_neg         = _FPC_FP64_IS_INFINITY_NEG(x);
  item.nan                  = _FPC_FP64_IS_NAN(x);
  item.division_zero        = _FPC_FP64_IS_DIVISON_ZERO(x, y, z, op);
  item.cancellation         = _FPC_FP64_IS_CANCELLATION(x, y, z, op);
  item.comparison           = _FPC_FP64_IS_COMPARISON(x, y, z, op);
  item.underflow            = _FPC_FP64_IS_SUBNORMAL(x);
  item.latent_infinity_pos  = _FPC_FP64_IS_LATENT_INFINITY_POS(x);
  item.latent_infinity_neg  = _FPC_FP64_IS_LATENT_INFINITY_NEG(x);
  item.latent_underflow     = _FPC_FP64_IS_LATENT_SUBNORMAL(x);

  if (_FPC_EVENT_OCURRED(&item))
    _FPC_HT_SET_(_FPC_HTABLE_, &item);
}


#endif /* SRC_RUNTIME_CPU_H_ */
