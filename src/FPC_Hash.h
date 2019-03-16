/*
 * FPC_Hash.h
 *
 *  Created on: Mar 14, 2019
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#ifndef SRC_FPC_HASH_H_
#define SRC_FPC_HASH_H_

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <unistd.h>
#include <string.h>


struct _FPC_ENTRY_S_ {
	char *fileName;
	int line;
	int64_t minVal;
	int64_t maxVal;
	int overflow;
	int64_t overRes; /// result of overflow
	struct _FPC_ENTRY_S_ *next;
};

typedef struct _FPC_ENTRY_S_ _FPC_ENTRY_T_;

struct _FPC_HTABLE_S {
	int size;
	int n; // number of entries
	struct _FPC_ENTRY_S_ **table;
};

typedef struct _FPC_HTABLE_S _FPC_HTABLE_T;

_FPC_HTABLE_T *_FPC_HT_CREATE_(int size)
{
	_FPC_HTABLE_T *hashtable = NULL;
	int i;

	if( size < 1 )
		return NULL;

	// Allocate the table itself.
	if( ( hashtable = (_FPC_HTABLE_T*)malloc( sizeof( _FPC_HTABLE_T ) ) ) == NULL )
	{
		printf("#FPCHECKER: out of memory error!");
		exit(EXIT_FAILURE);
		//return NULL;
	}

	// Allocate pointers to the head nodes.
	if( ( hashtable->table = (struct _FPC_ENTRY_S_ **)malloc( sizeof( _FPC_ENTRY_T_ * ) * size ) ) == NULL )
	{
		printf("#FPCHECKER: out of memory error!");
		exit(EXIT_FAILURE);
		//return NULL;
	}
	for(i = 0; i < size; i++)
	{
		hashtable->table[i] = NULL;
	}

	hashtable->size = size;
	hashtable->n = 0;

	return hashtable;
}

int _FPC_HT_HASH_( _FPC_HTABLE_T *hashtable, _FPC_ENTRY_T_ *val)
{
	uint64_t key = (uint64_t)(val->fileName);
	key += val->line;
	return key % hashtable->size;
}

/* Create a key-value pair. */
//entry_t *ht_newpair(char *key, char *value)
_FPC_ENTRY_T_ *_FPC_HT_NEWPAIR_(_FPC_ENTRY_T_ *val)
{
	_FPC_ENTRY_T_ *newpair;

	if( (newpair = (_FPC_ENTRY_T_ *)malloc(sizeof(_FPC_ENTRY_T_ ))) == NULL)
	{
		printf("#FPCHECKER: out of memory error!");
		exit(EXIT_FAILURE);
		//return NULL;
	}

	newpair->fileName = val->fileName;
	newpair->line = val->line;
	newpair->minVal = val->minVal;
	newpair->maxVal = val->maxVal;
	newpair->overflow = val->overflow;
	newpair->overRes = val->overRes;

	newpair->next = NULL;

	return newpair;
}

int _FPC_ITEMS_EQUAL_(_FPC_ENTRY_T_ *x, _FPC_ENTRY_T_ *y)
{
	if ((x->fileName == y->fileName) && (x->line == y->line))
			return 1;
	return 0;
}

/// Insert a key-value pair into a hash table
void _FPC_HT_SET_(_FPC_HTABLE_T *hashtable, _FPC_ENTRY_T_ *newVal)
{
	int bin = 0;
	_FPC_ENTRY_T_ *newpair = NULL;
	_FPC_ENTRY_T_ *next = NULL;
	_FPC_ENTRY_T_ *last = NULL;

	bin = _FPC_HT_HASH_(hashtable, newVal);
	next = hashtable->table[bin];

	//while( next != NULL && next->key != NULL && strcmp( key, next->key ) > 0 )
	while(next != NULL && !_FPC_ITEMS_EQUAL_(newVal, next))
	{
		last = next;
		next = next->next;
	}

	// There's already a pair
	//if( next != NULL && next->key != NULL && strcmp( key, next->key ) == 0)
	if(next != NULL && _FPC_ITEMS_EQUAL_(newVal, next))
	{
		//free( next->value );
		//next->value = strdup( value );
		if (newVal->minVal < next->minVal)
			next->minVal = newVal->minVal;
		if (newVal->maxVal > next->maxVal)
			next->maxVal = newVal->maxVal;
		if (newVal->overflow > next->overflow)
		{
			next->overflow = newVal->overflow;
			next->overRes = newVal->overRes;
		}
	}
	else // Nope, could't find it
	{
		//newpair = ht_newpair( key, value );
		newpair = _FPC_HT_NEWPAIR_(newVal);
		(hashtable->n)++;

		/* We're at the start of the linked list in this bin. */
		if (next == hashtable->table[bin])
		{
			newpair->next = next;
			hashtable->table[bin] = newpair;

		/* We're at the end of the linked list in this bin. */
		}
		else if ( next == NULL )
		{
			last->next = newpair;
		/* We're in the middle of the list. */
		}
		else
		{
			newpair->next = next;
			last->next = newpair;
		}
	}
}

void _FPC_PRINT_HASH_TABLE_(_FPC_HTABLE_T *hashtable)
{
	/// Set filename
	size_t len=128;
	char nodeName[len];
	nodeName[0] = '\0';
	if(gethostname(nodeName, len) != 0)
		strcpy(nodeName, "node-unknown");

	char fileName[len];
	fileName[0] = '\0';
	strcpy(fileName, "fpc_");
	strcat(fileName, nodeName);

	int pid = (int)getpid();
	char pidStr[len];
	pidStr[0] = '\0';
	sprintf(pidStr, "%d", pid);
	strcat(fileName, "_");
	strcat(fileName, pidStr);
	strcat(fileName, ".json");

	int n = hashtable->n;
	int printed = 0;

	FILE *fp;
	fp = fopen(fileName, "w");

	fprintf(fp, "[\n");

	for (int i=0; i < hashtable->size; ++i)
	{
		_FPC_ENTRY_T_ *next;
		next = hashtable->table[i];

		while(next != NULL)
		{
			printf("file: %s, line: %d, min %lld, max %lld\n", next->fileName, next->line, next->minVal, next->maxVal);
			fprintf(fp, "  {\n");
			fprintf(fp, "\t\"file\": \"%s\",\n", next->fileName);
			fprintf(fp, "\t\"line\": %d,\n", next->line);
			fprintf(fp, "\t\"min\": %lld,\n", next->minVal);
			fprintf(fp, "\t\"max\": %lld,\n", next->maxVal);
			fprintf(fp, "\t\"over\": %d,\n", next->overflow);
			fprintf(fp, "\t\"over_res\": %lld\n", next->overRes);

			next = next->next;
			printed++;

			if (printed == n)
				fprintf(fp, "  }\n");
			else
				fprintf(fp, "  },\n");
		}
	}

	fprintf(fp, "]\n");
	fclose(fp);
}

#endif /* SRC_FPC_HASH_H_ */
