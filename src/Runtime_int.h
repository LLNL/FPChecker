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
}

void _FPC_PRINT_LOCATIONS_()
{
	_FPC_PRINT_HASH_TABLE_(_FPC_HTABLE_);
}


/* ------------------------ FP32 Functions --------------------------------- */

void _FPC_FP32_CHECK_ADD_(int x, int y, int z, int loc, char *fileName)
{
	//printf("fileName: %s %p\n", fileName, fileName);
	_FPC_ENTRY_T_ t;
	t.fileName = fileName;
	t.line = loc;
	t.minVal = (uint64_t)x;
	t.maxVal = (uint64_t)x;
	_FPC_HT_SET_(_FPC_HTABLE_, &t);
}

void _FPC_FP32_CHECK_SUB_(int x, int y, int z, int loc, char *fileName)
{
	_FPC_ENTRY_T_ t;
	t.fileName = fileName;
	t.line = loc;
	t.minVal = (uint64_t)x;
	t.maxVal = (uint64_t)x;
	_FPC_HT_SET_(_FPC_HTABLE_, &t);
}

void _FPC_FP32_CHECK_MUL_(int x, int y, int z, int loc, char *fileName)
{
	_FPC_ENTRY_T_ t;
	t.fileName = fileName;
	t.line = loc;
	t.minVal = (uint64_t)x;
	t.maxVal = (uint64_t)x;
	_FPC_HT_SET_(_FPC_HTABLE_, &t);
}

void _FPC_FP32_CHECK_DIV_(int x, int y, int z, int loc, char *fileName)
{
	//LineLocation lineLoc(fileName, loc);
	//MinMaxPair v(x, x);
	//_FPC_INSERT_LOCATIONS_MAP_(lineLoc, v);
}

/* ------------------------ FP64 Functions --------------------------------- */


#endif /* SRC_RUNTIME_INT_H_ */
