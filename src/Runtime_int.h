/*
 * Runtime_int.h
 *
 *  Created on: Mar 13, 2019
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#ifndef SRC_RUNTIME_INT_H_
#define SRC_RUNTIME_INT_H_

#include "FPC_Hash.h"
#include <stdio.h>

/* ----------------------------- Global Data ------------------------------- */

/// We store the file name and directory in this variable
static char *_FPC_FILE_NAME_;
_FPC_HTABLE_T *_FPC_HTABLE_;

/* ------------------------ Generic Functions ------------------------------ */

void _FPC_UNUSED_FUNC_()
{
	asm ("");
	printf("#FPCHECKER: %s\n", _FPC_FILE_NAME_);
}

void _FPC_INIT_HASH_TABLE_()
{
	int size = 1000;
	_FPC_HTABLE_ = _FPC_HT_CREATE_(size);

	printf("\n");
	printf("========================================\n");
	printf(" FPChecker (v0.0.4, %s)\n", __DATE__);
	printf("========================================\n");
	printf("\n");
}

void _FPC_PRINT_LOCATIONS_()
{
	_FPC_PRINT_HASH_TABLE_(_FPC_HTABLE_);
}

/// op: 0:ADD, 1:SUB, 2:MUL
int _FPC_CHECK_OVERFLOW_(int x, int y, int z, int op, int64_t *res)
{
	//int64_t _x = (int64_t)x;
	int64_t _y = (int64_t)y;
	int64_t _z = (int64_t)z;
	*res = 0;

	if (op == 0)
		*res = _y + _z;
	else if (op == 1)
		*res = _y - _z;
	else if (op == 2)
		*res = _y * _z;

	if (*res < (int64_t)INT_MIN || *res > (int64_t)INT_MAX)
		return 1;

	return 0;
}

/* ------------------------ FP32 Functions --------------------------------- */

void _FPC_FP32_CHECK_ADD_(int x, int y, int z, int loc, char *fileName)
{
	_FPC_ENTRY_T_ t;
	t.fileName = fileName;
	t.line = loc;
	t.minVal = (int64_t)x;
	t.maxVal = (int64_t)x;

	int64_t res = 0;
	if (_FPC_CHECK_OVERFLOW_(x, y, z, 0, &res))
	{
		t.overflow = 1;
		t.overRes = res;
	}
	else
	{
		t.overflow = 0;
		t.overRes = 0;
	}

	_FPC_HT_SET_(_FPC_HTABLE_, &t);
}

void _FPC_FP32_CHECK_SUB_(int x, int y, int z, int loc, char *fileName)
{
	_FPC_ENTRY_T_ t;
	t.fileName = fileName;
	t.line = loc;
	t.minVal = (int64_t)x;
	t.maxVal = (int64_t)x;

	int64_t res = 0;
	if (_FPC_CHECK_OVERFLOW_(x, y, z, 1, &res))
	{
		t.overflow = 1;
		t.overRes = res;
	}
	else
	{
		t.overflow = 0;
		t.overRes = 0;
	}

	_FPC_HT_SET_(_FPC_HTABLE_, &t);
}

void _FPC_FP32_CHECK_MUL_(int x, int y, int z, int loc, char *fileName)
{
	_FPC_ENTRY_T_ t;
	t.fileName = fileName;
	t.line = loc;
	t.minVal = (int64_t)x;
	t.maxVal = (int64_t)x;

	int64_t res = 0;
	if (_FPC_CHECK_OVERFLOW_(x, y, z, 2, &res))
	{
		t.overflow = 1;
		t.overRes = res;
	}
	else
	{
		t.overflow = 0;
		t.overRes = 0;
	}

	_FPC_HT_SET_(_FPC_HTABLE_, &t);
}

void _FPC_FP32_CHECK_DIV_(int x, int y, int z, int loc, char *fileName)
{
}

/* ------------------------ FP64 Functions --------------------------------- */


#endif /* SRC_RUNTIME_INT_H_ */
