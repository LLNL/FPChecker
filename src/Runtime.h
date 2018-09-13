/*
 * Runtime.h
 *
 *  Created on: Apr 10, 2018
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#ifndef SRC_RUNTIME_H_
#define SRC_RUNTIME_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

/* ----------------------------- Global Data ------------------------------- */

//char *_FPC_LOCATIONS_TABLE_[100];// = {"NONE1"};
__device__ char *_FPC_FILE_NAME_[1];


/* ------------------------ Generic Functions ------------------------------ */

__device__
void _FPC_INTERRUPT_(int loc)
{
	printf("Error: File: %s, Line: %d\n", _FPC_FILE_NAME_[0], loc);
	asm("trap;");
}


/* ------------------------ FP32 Functions --------------------------------- */

/* Returns non-zero value if FP argument is a sub-normal */
__device__
int _FPC_FP32_IS_SUBNORMAL(double x)
{
	int ret = 0;
	uint32_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1; 	// get rid of sign bit
  val = val >> 24; 	// get rid of the mantissa bits
  if (x != 0.0 && x != -0.0)
  {
  	if (val == 0)
  		ret = 1;
  }
  return ret;
}

__device__
void _FPC_FP32_CHECK_ADD_(float x, float y, float z, int loc)
{
	if (isinf(x))
	{
		puts("ERROR: infinite value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (isnan(x))
	{
		puts("ERROR: NaN value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (_FPC_FP32_IS_SUBNORMAL(x))
	{
		puts("ERROR: Subnormal value!");
		_FPC_INTERRUPT_(loc);
	}
}

__device__
void _FPC_FP32_CHECK_SUB_(float x, float y, float z, int loc)
{
	if (isinf(x))
	{
		puts("ERROR: infinite value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (isnan(x))
	{
		puts("ERROR: NaN value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (_FPC_FP32_IS_SUBNORMAL(x))
	{
		puts("ERROR: Subnormal value!");
		_FPC_INTERRUPT_(loc);
	}
}

__device__
void _FPC_FP32_CHECK_MUL_(float x, float y, float z, int loc)
{
	if (isinf(x))
	{
		puts("ERROR: infinite value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (isnan(x))
	{
		puts("ERROR: NaN value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (_FPC_FP32_IS_SUBNORMAL(x))
	{
		puts("ERROR: Subnormal value!");
		_FPC_INTERRUPT_(loc);
	}
}

__device__
void _FPC_FP32_CHECK_DIV_(float x, float y, float z, int loc)
{
	if (isinf(x))
	{
		puts("ERROR: infinite value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (isnan(x))
	{
		puts("ERROR: NaN value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (_FPC_FP32_IS_SUBNORMAL(x))
	{
		puts("ERROR: Subnormal value!");
		_FPC_INTERRUPT_(loc);
	}
}

/* ------------------------ FP64 Functions --------------------------------- */

/* Returns non-zero value if FP argument is a sub-normal */
__device__
int _FPC_FP64_IS_SUBNORMAL(double x)
{
	int ret = 0;
	uint64_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1; 	// get rid of sign bit
  val = val >> 53; 	// get rid of the mantissa bits
  if (x != 0.0 && x != -0.0)
  {
  	if (val == 0)
  		ret = 1;
  }
  return ret;
}

__device__
void _FPC_FP64_CHECK_ADD_(double x, double y, double z, int loc)
{
	if (isinf(x))
	{
		puts("ERROR: infinite value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (isnan(x))
	{
		puts("ERROR: NaN value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (_FPC_FP64_IS_SUBNORMAL(x))
	{
		puts("ERROR: Subnormal value!");
		_FPC_INTERRUPT_(loc);
	}
}

__device__
void _FPC_FP64_CHECK_SUB_(double x, double y, double z, int loc)
{
	if (isinf(x))
	{
		puts("ERROR: infinite value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (isnan(x))
	{
		puts("ERROR: NaN value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (_FPC_FP64_IS_SUBNORMAL(x))
	{
		puts("ERROR: Subnormal value!");
		_FPC_INTERRUPT_(loc);
	}
}

__device__
void _FPC_FP64_CHECK_MUL_(double x, double y, double z, int loc)
{
	if (isinf(x))
	{
		puts("ERROR: infinite value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (isnan(x))
	{
		puts("ERROR: NaN value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (_FPC_FP64_IS_SUBNORMAL(x))
	{
		puts("ERROR: Subnormal value!");
		_FPC_INTERRUPT_(loc);
	}
}

__device__
void _FPC_FP64_CHECK_DIV_(double x, double y, double z, int loc)
{
	if (isinf(x))
	{
		puts("ERROR: infinite value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (isnan(x))
	{
		puts("ERROR: NaN value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (_FPC_FP64_IS_SUBNORMAL(x))
	{
		puts("ERROR: Subnormal value!");
		_FPC_INTERRUPT_(loc);
	}
}

#endif /* SRC_RUNTIME_H_ */
