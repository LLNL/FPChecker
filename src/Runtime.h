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

/// *** Warning ***
/// Changing this file: Runtime.h
/// When a global __device__ function is added in the this file, some actions
/// must be taken:
/// (1) We need to find this function when initializing Instrumentation pass.
/// (2) Linkage of the function must be set to LinkOnceODRLinkage.
/// (3) We need to add this function to the list of unwanted functions, i.e.,
///     functions we do not instrument in the pass.

__device__ void	_FPC_INTERRUPT_(int loc);
__device__ int	_FPC_FP32_IS_SUBNORMAL(double x);
__device__ void _FPC_FP32_CHECK_ADD_(float x, float y, float z, int loc);
__device__ void _FPC_FP32_CHECK_SUB_(float x, float y, float z, int loc);
__device__ void _FPC_FP32_CHECK_MUL_(float x, float y, float z, int loc);
__device__ void _FPC_FP32_CHECK_DIV_(float x, float y, float z, int loc);
__device__ int	_FPC_FP64_IS_SUBNORMAL(double x);
__device__ void _FPC_FP64_CHECK_ADD_(float x, float y, float z, int loc);
__device__ void _FPC_FP64_CHECK_SUB_(float x, float y, float z, int loc);
__device__ void _FPC_FP64_CHECK_MUL_(float x, float y, float z, int loc);
__device__ void _FPC_FP64_CHECK_DIV_(float x, float y, float z, int loc);

/* ----------------------------- Global Data ------------------------------- */

#define TOOL_NAME "[FPChecker] "

//char *_FPC_LOCATIONS_TABLE_[100];// = {"NONE1"};
__device__ char *_FPC_FILE_NAME_[1];


/* ------------------------ Generic Functions ------------------------------ */

__device__
void _FPC_INTERRUPT_(int loc)
{
	printf(TOOL_NAME "File: %s, Line: %d\n", _FPC_FILE_NAME_[0], loc);
	//fflush(stdout);
	asm("trap;");
}

/// Check the operation.
/// type: 0 for float, 1 for double
/// x,y,z: x=operation result, y=first operand, z=second operand
/// loc: line number
__device__
static void _FPC_CHECK_OPERATION_(int type, float x, float y, float z, int loc)
{
	if (isinf(x))
	{
		printf(TOOL_NAME "ERROR: infinite value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (isnan(x))
	{
		printf(TOOL_NAME "ERROR: NaN value!");
		_FPC_INTERRUPT_(loc);
	}
	else if (type == 0) /// subnormals check
	{
		if (_FPC_FP32_IS_SUBNORMAL(x))
		{
			printf(TOOL_NAME "ERROR: Subnormal value!");
			_FPC_INTERRUPT_(loc);
		}
	}
	else if (type == 1) /// subnormals check
	{
		if (_FPC_FP64_IS_SUBNORMAL(x))
		{
			printf(TOOL_NAME "ERROR: Subnormal value!");
			_FPC_INTERRUPT_(loc);
		}
	}
}

/* ------------------------ FP32 Functions --------------------------------- */

//// Returns non-zero value if FP argument is a sub-normal
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
	_FPC_CHECK_OPERATION_(0, x, y, z, loc);
}

__device__
void _FPC_FP32_CHECK_SUB_(float x, float y, float z, int loc)
{
	_FPC_CHECK_OPERATION_(0, x, y, z, loc);
}

__device__
void _FPC_FP32_CHECK_MUL_(float x, float y, float z, int loc)
{
	_FPC_CHECK_OPERATION_(0, x, y, z, loc);
}

__device__
void _FPC_FP32_CHECK_DIV_(float x, float y, float z, int loc)
{
	_FPC_CHECK_OPERATION_(0, x, y, z, loc);
}

/* ------------------------ FP64 Functions --------------------------------- */

/// Returns non-zero value if FP argument is a sub-normal.
/// Check that the exponent bits are zero.
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
	_FPC_CHECK_OPERATION_(1, x, y, z, loc);
}

__device__
void _FPC_FP64_CHECK_SUB_(double x, double y, double z, int loc)
{
	_FPC_CHECK_OPERATION_(1, x, y, z, loc);
}

__device__
void _FPC_FP64_CHECK_MUL_(double x, double y, double z, int loc)
{
	_FPC_CHECK_OPERATION_(1, x, y, z, loc);
}

__device__
void _FPC_FP64_CHECK_DIV_(double x, double y, double z, int loc)
{
	_FPC_CHECK_OPERATION_(1, x, y, z, loc);
}

#endif /* SRC_RUNTIME_H_ */
