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
#include <cmath>

//char *_FPC_LOCATIONS_TABLE_[100];// = {"NONE1"};
__device__ char *_FPC_FILE_NAME_[1];

__device__
void _FPC_INTERRUPT_(int loc)
{
	//asm("trap;");
	printf("File: %s, Line: %d\n", _FPC_FILE_NAME_[0], loc);
}

/* ------------------------ FP32 Functions --------------------------------- */

__device__
void _FPC_FP32_CHECK_ADD_(float x, float y, float z, int loc)
{
	if (isnan(x))
	{
		_FPC_INTERRUPT_(loc);
	}
}

__device__
void _FPC_FP32_CHECK_SUB_(float x, float y, float z, int loc)
{
	if (isnan(x))
	{
		_FPC_INTERRUPT_(loc);
	}
}

__device__
void _FPC_FP32_CHECK_MUL_(float x, float y, float z, int loc)
{
	if (isnan(x))
	{
		_FPC_INTERRUPT_(loc);
	}
}

__device__
void _FPC_FP32_CHECK_DIV_(float x, float y, float z, int loc)
{
	if (isnan(x))
	{
		_FPC_INTERRUPT_(loc);
	}
}

/* ------------------------ FP64 Functions --------------------------------- */

__device__
void _FPC_FP64_CHECK_ADD_(double x, double y, double z, int loc)
{
	if (isnan(x))
	{
		_FPC_INTERRUPT_(loc);
	}
}

__device__
void _FPC_FP64_CHECK_SUB_(double x, double y, double z, int loc)
{
	if (isnan(x))
	{
		_FPC_INTERRUPT_(loc);
	}
}

__device__
void _FPC_FP64_CHECK_MUL_(double x, double y, double z, int loc)
{
	if (isnan(x))
	{
		_FPC_INTERRUPT_(loc);
	}
}

__device__
void _FPC_FP64_CHECK_DIV_(double x, double y, double z, int loc)
{
	if (isnan(x))
	{
		_FPC_INTERRUPT_(loc);
	}
}

#endif /* SRC_RUNTIME_H_ */
