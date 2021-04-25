
#ifndef SRC_RUNTIME_H_
#define SRC_RUNTIME_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <type_traits>
#include <limits>

/* --------------- Definitions --------------------------------------------- */
/*
static 
short _FPC_CHECK_(short x, int loc, const char *fileName) {return x;}

static 
unsigned short _FPC_CHECK_(unsigned short x, int loc, const char *fileName) {return x;}

static 
int _FPC_CHECK_(int x, int loc, const char *fileName) {return x;}

static 
unsigned _FPC_CHECK_(unsigned x, int loc, const char *fileName) {return x;}

static 
long _FPC_CHECK_(long x, int loc, const char *fileName) {return x;}

static 
unsigned long _FPC_CHECK_(unsigned long x, int loc, const char *fileName) {return x;}

static 
long long _FPC_CHECK_(long long x, int loc, const char *fileName) {return x;}

static 
unsigned long long _FPC_CHECK_(unsigned long long x, int loc, const char *fileName) {return x;}
*/

////////// HOST DEVICE CHECKS ///////////////////////////
static
double _FPC_CHECK_HD_(double x, int l, const char *str) {
  return x;
}

static
float _FPC_CHECK_HD_(float x, int l, const char *str) {
  return x;
}

template<typename T>
T _FPC_CHECK_HD_(T t, int x, const char *str) {
  return t;
}

////////// DEVICE CHECKS ///////////////////////////
static
double _FPC_CHECK_D_(double x, int l, const char *str) {
  return x;
}

static
float _FPC_CHECK_D_(float x, int l, const char *str) {
  return x;
}

template<typename T>
T _FPC_CHECK_D_(T t, int x, const char *str) {
  return t;
}

/* ----------------------------------------------------------------------- */

#endif
