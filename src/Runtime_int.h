/*
 * Runtime_int.h
 *
 *  Created on: Mar 13, 2019
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#ifndef SRC_RUNTIME_INT_H_
#define SRC_RUNTIME_INT_H_

#include <stdio.h>
#include <map>

/* ----------------------------- Global Data ------------------------------- */

/// We store the file name and directory in this variable
static char *_FPC_FILE_NAME_;

typedef std::pair<char*, int> LineLocation;
typedef std::pair<int, int> MinMaxPair;
typedef std::map<LineLocation, MinMaxPair> LocationsMap;
LocationsMap _FPC_LOCATIONS_MAP_;

/* ------------------------ Generic Functions ------------------------------ */

void _FPC_UNUSED_FUNC_()
{
	asm ("");
	printf("#FPCHECKER: %s\n", _FPC_FILE_NAME_);
}

void _FPC_INSERT_LOCATIONS_MAP_(const LineLocation &loc, const MinMaxPair &v)
{
	auto it = _FPC_LOCATIONS_MAP_.find(loc);
	if (it == _FPC_LOCATIONS_MAP_.end())
	{
		std::pair<LineLocation, MinMaxPair> tmp(loc, v);
		_FPC_LOCATIONS_MAP_.insert(tmp);
		//_FPC_LOCATIONS_MAP_[loc] = v;
	}
	else
	{
		int minVal = v.first;
		int maxVal = v.second;
		if (minVal < it->second.first)
			it->second.first = minVal;
		if (maxVal > it->second.second)
			it->second.second = maxVal;
	}
}


/* ------------------------ FP32 Functions --------------------------------- */

void _FPC_FP32_CHECK_ADD_(int x, int y, int z, int loc, char *fileName)
{
	printf("fileName: %s %p\n", fileName, fileName);
	//uint64_t locLargeInt = (uint64_t)fileName;
	LineLocation lineLoc(fileName, loc);
	MinMaxPair v(x, x);
	_FPC_INSERT_LOCATIONS_MAP_(lineLoc, v);
}

void _FPC_FP32_CHECK_SUB_(int x, int y, int z, int loc, char *fileName)
{

}

void _FPC_FP32_CHECK_MUL_(int x, int y, int z, int loc, char *fileName)
{

}

void _FPC_FP32_CHECK_DIV_(int x, int y, int z, int loc, char *fileName)
{

}

/* ------------------------ FP64 Functions --------------------------------- */


#endif /* SRC_RUNTIME_INT_H_ */
