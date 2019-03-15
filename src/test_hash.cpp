/*
 * test_hash.cpp
 *
 *  Created on: Mar 14, 2019
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#include "FPC_Hash.h"
#include <stdlib.h>


int main()
{
	printf("Hash table test\n");
	int size = 1000;
	_FPC_HTABLE_T *table = _FPC_HT_CREATE_(size);
	for (int i=0; i < 200; ++i)
	{
		char *file = (char*)malloc(sizeof(char)*128);
		file[0] = '\0';
		sprintf(file, "file_%d", i);

		int line = i*100;
		int mi = i+1;
		int ma = i+5;

		_FPC_ENTRY_T_ t;
		t.fileName = file;
		t.line = line;
		t.minVal = mi;
		t.maxVal = ma;
		_FPC_HT_SET_(table, &t);
	}

	_FPC_PRINT_HASH_TABLE_(table);

	return 0;
}
