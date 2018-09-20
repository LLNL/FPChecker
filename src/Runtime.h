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

#ifdef __CUDA_ARCH__
/// Symbol used to determine whether we are compiling device code or not
__device__ void _FPC_DEVICE_CODE_FUNC_(){};
#endif

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

#define REPORT_LINE_SIZE 80
#define REPORT_COL1_SIZE 15
#define REPORT_COL2_SIZE REPORT_LINE_SIZE-REPORT_COL1_SIZE-1

/// We store the file name and directory in this variable
//char *_FPC_LOCATIONS_TABLE_[100];// = {"NONE1"};
__device__ char *_FPC_FILE_NAME_[1];

__device__ static int lock_state = 0;

/* ------------------------ Generic Functions ------------------------------ */

/// Report format
/// +---------------- FPChecker Error Report ------------------+
///  Error     : NaN
///  Operation : DIV
///  File      : /usr/file/pathForceCalculation.cc
///  Line      : 23
///  Thread ID : 1024
/// +----------------------------------------------------------+

__device__
static void _FPC_PRINT_REPORT_LINE_(const char border)
{
	printf("%c",border);
	for (int i=0; i < REPORT_LINE_SIZE-2; ++i)
		printf("-");
	printf("%c\n",border);
}

__device__
static void _FPC_PRINT_REPORT_HEADER_(int type)
{
	//_FPC_PRINT_REPORT_LINE_('.');

	char msg[255];
	msg[0] = '\0';
	if (type == 0)
		strcpy(msg," FPChecker Error Report ");
	else
		strcpy(msg," FPChecker Warning Report ");

	int l = strlen(msg);
	l = REPORT_LINE_SIZE-l-2;
	char line[255];
	line[0] = '\0';
	strcat(line,"+");
	for (int i=0; i < l/2; ++i)
			strcat(line,"-");
	if (l%2)
		strcat(line,"-");
	strcat(line,msg);
	for (int i=0; i < l/2; ++i)
			strcat(line,"-");
	strcat(line,"+");
	printf("%s\n",line);
}

__device__
static void _FPC_PRINT_REPORT_ROW_(const char *val, int space, int last)
{
	char msg[255];
	msg[0] = '\0';
	strcpy(msg," ");
	strcat(msg, val);
	int rem = strlen(msg);
	for (int i=0; i < space-rem; ++i)
		strcat(msg," ");
	printf("%s",msg);

	if (last==0)
		printf(":");
	else
		printf("\n");
}

/// errorType: 0:NaN, 1:INF, 2:Underflow
/// op: 0:ADD, 1:SUB, 2:MUL, 3:DIV
__device__
void _FPC_INTERRUPT_(int errorType, int op, int loc)
{
	bool blocked = true;
  	while(blocked) {
			if(0 == atomicCAS(&lock_state, 0, 1)) {

				char e[64]; e[0] = '\0';
				char o[64]; o[0] = '\0';
				char l[64]; l[0] = '\0';

				if 			(errorType == 0) strcpy(e, "NaN");
				else if	(errorType == 1) strcpy(e, "INF");
				else if	(errorType == 2) strcpy(e, "Underflow");
				else strcpy(e, "NONE");

				if 			(op == 0) strcpy(o, "ADD");
				else if	(op == 1) strcpy(o, "SUB");
				else if	(op == 2) strcpy(o, "MUL");
				else if	(op == 3) strcpy(o, "DIV");
				else strcpy(o, "NONE");

				sprintf(l, "%d", loc);

				_FPC_PRINT_REPORT_HEADER_(0);
				_FPC_PRINT_REPORT_ROW_("Error", REPORT_COL1_SIZE, 0);
				_FPC_PRINT_REPORT_ROW_(e, REPORT_COL2_SIZE, 1);
				_FPC_PRINT_REPORT_ROW_("Operation", REPORT_COL1_SIZE, 0);
				_FPC_PRINT_REPORT_ROW_(o, REPORT_COL2_SIZE, 1);
				_FPC_PRINT_REPORT_ROW_("File", REPORT_COL1_SIZE, 0);
				_FPC_PRINT_REPORT_ROW_(_FPC_FILE_NAME_[0], REPORT_COL2_SIZE, 1);
				_FPC_PRINT_REPORT_ROW_("Line", REPORT_COL1_SIZE, 0);
				_FPC_PRINT_REPORT_ROW_(l, REPORT_COL2_SIZE, 1);
				_FPC_PRINT_REPORT_LINE_('+');

				//printf(TOOL_NAME "File: %s, Line: %d\n", _FPC_FILE_NAME_[0], loc);

				asm("trap;");
		}
	}
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
		//printf(TOOL_NAME "ERROR: NaN value!");
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
