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
#include <limits.h>
#include <type_traits>
#include <limits>

/* --------------- Definitions --------------------------------------------- */
__host__ __device__
static float _FPC_CHECK_(float x, int loc, const char *fileName);

__host__ __device__
static double _FPC_CHECK_(double x, int loc, const char *fileName);

__host__ __device__ static 
short _FPC_CHECK_(short x, int loc, const char *fileName) {return x;}

__host__ __device__ static 
unsigned short _FPC_CHECK_(unsigned short x, int loc, const char *fileName) {return x;}

__host__ __device__ static 
int _FPC_CHECK_(int x, int loc, const char *fileName) {return x;}

__host__ __device__ static 
unsigned _FPC_CHECK_(unsigned x, int loc, const char *fileName) {return x;}

__host__ __device__ static 
long _FPC_CHECK_(long x, int loc, const char *fileName) {return x;}

__host__ __device__ static 
unsigned long _FPC_CHECK_(unsigned long x, int loc, const char *fileName) {return x;}

__host__ __device__ static 
long long _FPC_CHECK_(long long x, int loc, const char *fileName) {return x;}

__host__ __device__ static 
unsigned long long _FPC_CHECK_(unsigned long long x, int loc, const char *fileName) {return x;}


////////// HOST DEVICE CHECKS ///////////////////////////
__host__ __device__ static
double _FPC_CHECK_HD_(double x, int l, const char *str) {
  return _FPC_CHECK_(x, l, str);
}

__host__ __device__ static
float _FPC_CHECK_HD_(float x, int l, const char *str) {
  return _FPC_CHECK_(x, l, str);
}

template<typename T>
__host__ __device__ 
T _FPC_CHECK_HD_(T t, int x, const char *str) {
  return t;
}

////////// DEVICE CHECKS ///////////////////////////
__host__ __device__ static
double _FPC_CHECK_D_(double x, int l, const char *str) {
  return _FPC_CHECK_(x, l, str);
}

__host__ __device__ static
float _FPC_CHECK_D_(float x, int l, const char *str) {
  return _FPC_CHECK_(x, l, str);
}

template<typename T>
__host__ __device__ 
T _FPC_CHECK_D_(T t, int x, const char *str) {
  return t;
}

/* ----------------------------------------------------------------------- */

#define REPORT_LINE_SIZE 80
#define REPORT_COL1_SIZE 15
#define REPORT_COL2_SIZE REPORT_LINE_SIZE-REPORT_COL1_SIZE-1

#ifdef FPC_DANGER_ZONE_PERCENT
#define DANGER_ZONE_PERCENTAGE FPC_DANGER_ZONE_PERCENT
#else
#define DANGER_ZONE_PERCENTAGE 0.05
#endif

// Enable short reports by default (i.e., errors don't abort)
#ifndef FPC_ERRORS_ABORT
#define FPC_SHORT_REPORTS
#endif

// Disable warning by default
#ifndef FPC_ENABLE_WARNINGS
#define FPC_DISABLE_WARNINGS
#endif


/* ----------------------------- Global Data ------------------------------- */


/// Lock to print from one thread only
//__device__ static int lock_state = 0;
__device__ static int _FPC_LOCK_STATE_ = 0;


/* ------------------------ Generic Functions ------------------------------ */

/// Report format
/// +---------------- FPChecker Error Report ------------------+
///  Error     : NaN
///  Operation : DIV
///  File      : /usr/file/pathForceCalculation.cc
///  Line      : 23
/// +----------------------------------------------------------+

__host__ __device__
static int _FPC_LEN_(const char *s)
{
	int maxLen = 1024; // to check correctness and avid infinite loop
	int i = 0;
	while(s[i] != '\0' && i < maxLen)
		i++;
	return i;
}

__host__ __device__
static void _FPC_CPY_(char *d, const char *s)
{
	int len = _FPC_LEN_(s);
	int i=0;
	for (i=0; i < len; ++i)
		d[i] = s[i];
	d[i] = '\0';
}

__host__ __device__
static void _FPC_CAT_(char *d, const char *s)
{
	int lenS = _FPC_LEN_(s);
	int lenD = _FPC_LEN_(d);
	int i=0;
	for (i=0; i < lenS; ++i)
		d[i+lenD] = s[i];
	d[i+lenD] = '\0';
}

__host__ __device__
static void _FPC_PRINT_REPORT_LINE_(const char border)
{
	printf("%c",border);
	for (int i=0; i < REPORT_LINE_SIZE-2; ++i)
		printf("-");
	printf("%c\n",border);
}

__host__ __device__
static void _FPC_PRINT_REPORT_HEADER_(int type)
{
	//_FPC_PRINT_REPORT_LINE_('.');

	char msg[255];
	msg[0] = '\0';
	if (type == 0)
		_FPC_CPY_(msg," FPChecker Error Report ");
	else
		_FPC_CPY_(msg," FPChecker Warning Report ");

	int l = _FPC_LEN_(msg);
	l = REPORT_LINE_SIZE-l-2;
	char line[255];
	line[0] = '\0';
	_FPC_CAT_(line,"+");
	for (int i=0; i < l/2; ++i)
		_FPC_CAT_(line,"-");
	if (l%2)
		_FPC_CAT_(line,"-");
	_FPC_CAT_(line,msg);
	for (int i=0; i < l/2; ++i)
		_FPC_CAT_(line,"-");
	_FPC_CAT_(line,"+");
	printf("%s\n",line);
}

__host__ __device__
static void _FPC_PRINT_REPORT_ROW_(const char *val, int space, int last, char lastChar)
{
	char msg[255];
	msg[0] = '\0';
	_FPC_CPY_(msg," ");
	_FPC_CAT_(msg, val);
	int rem = _FPC_LEN_(msg);
	for (int i=0; i < space-rem; ++i)
		_FPC_CAT_(msg," ");
	printf("%s",msg);

	if (last==0)
		printf("%c", lastChar);
	else
		printf("\n");
}

__host__ __device__
static void _FPC_PRINT_REPORT_ROW_(int val, int space, int last)
{
	int numChars = floor(log10 ((double)abs (val))) + 1;
	printf(" %d", val);

	char msg[255];
	msg[0] = '\0';
	int rem = numChars + 1;
	for (int i=0; i < space-rem; ++i)
		_FPC_CAT_(msg," ");
	printf("%s",msg);

	if (last==0)
		printf(":");
	else
		printf("\n");
}

__host__ __device__
static void _FPC_PRINT_REPORT_ROW_(float val, int space, int last)
{
	int numChars = 18;
	printf("(%1.9e)", val);

	char msg[255];
	msg[0] = '\0';
	int rem = numChars + 1;
	for (int i=0; i < space-rem; ++i)
		_FPC_CAT_(msg," ");
	printf("%s",msg);

	//if (last==0)
	//	printf("");
	//else
	if (last!=0)
		printf("\n");
}

__host__ __device__
static void _FPC_PRINT_REPORT_ROW_(double val, int space, int last)
{
	int numChars = 18;
	printf("(%1.9e)", val);

	char msg[255];
	msg[0] = '\0';
	int rem = numChars + 1;
	for (int i=0; i < space-rem; ++i)
		_FPC_CAT_(msg," ");
	printf("%s",msg);

	//if (last==0)
	//	printf("");
	//else
	if (last!=0)
		printf("\n");
}

/* ----- END of Report Functons --------------- */

/// errorType: 0:NaN, 1:INF, 2:Underflow
/// op: 0:ADD, 1:SUB, 2:MUL, 3:DIV

/** Calculates the ID of GPU thread */
__host__ __device__
static int _FPC_GET_GLOBAL_IDX_3D_3D()
{
#if defined(__CUDA_ARCH__)
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
 		+ (threadIdx.z * (blockDim.x * blockDim.y))
 		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
#else
  return 0;
#endif
}

/* ------------------------ FP32 Helper Functions --------------------------- */

/// Returns non-zero value if operation returned zero,
/// but none of arguments of the operation were zero
__host__ __device__
static int _FPC_FP32_IS_FLUSH_TO_ZERO(float x, float y, float z, int op)
{
	int ret = 0;
	if (x == 0.0 || x == -0.0)
	{
		if (y != 0.0 && y != -0.0 && z != 0.0 && z != -0.0)
		{
			if (op != 0 && op != 1) // for now, we check on MUL, DIV
				ret = 1;
		}
	}

	return ret;
}

//// Returns non-zero value if FP argument is a sub-normal
__host__ __device__
static int _FPC_FP32_IS_SUBNORMAL(float x)
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

__host__ __device__
static int _FPC_FP32_IS_ALMOST_OVERFLOW(float x)
{
	int ret = 0;
	uint32_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1; 	// get rid of sign bit
  val = val >> 24; 	// get rid of the mantissa bits
  if (x != 0.0 && x != -0.0)
  {
  	int maxVal = 256 - (int)(DANGER_ZONE_PERCENTAGE*256.0);
  	if (val >= maxVal)
  		ret = 1;
  }
  return ret;
}

__host__ __device__
static int _FPC_FP32_IS_ALMOST_SUBNORMAL(float x)
{
	int ret = 0;
	uint32_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1; 	// get rid of sign bit
  val = val >> 24; 	// get rid of the mantissa bits
  if (x != 0.0 && x != -0.0)
  {
  	int minVal = (int)(DANGER_ZONE_PERCENTAGE*256.0);
  	if (val <= minVal)
  		ret = 1;
  }
  return ret;
}

/* ------------------------ FP64 Helper Functions --------------------------- */

/// Returns non-zero value if operation returned zero,
/// but none of arguments of the operation were zero
__host__ __device__
static int _FPC_FP64_IS_FLUSH_TO_ZERO(double x, double y, double z, int op)
{
	int ret = 0;
	if (x == 0.0 || x == -0.0)
	{
		if (y != 0.0 && y != -0.0 && z != 0.0 && z != -0.0)
		{
			if (op != 0 && op != 1) // for now, we check on MUL, DIV
				ret = 1;
		}
	}

	return ret;
}

/// Returns non-zero value if FP argument is a sub-normal.
/// Check that the exponent bits are zero.
__host__ __device__
static int _FPC_FP64_IS_SUBNORMAL(double x)
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

__host__ __device__
static int _FPC_FP64_IS_ALMOST_OVERFLOW(double x)
{
	int ret = 0;
	uint64_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1; 	// get rid of sign bit
  val = val >> 53; 	// get rid of the mantissa bits
  if (x != 0.0 && x != -0.0)
  {
  	int maxVal = 2048 - (int)(DANGER_ZONE_PERCENTAGE*2048.0);
  	if (val >= maxVal)
  		ret = 1;
  }
  return ret;
}

__host__ __device__
static int _FPC_FP64_IS_ALMOST_SUBNORMAL(double x)
{
	int ret = 0;
	uint64_t val;
  memcpy((void *) &val, (void *) &x, sizeof(val));
  val = val << 1; 	// get rid of sign bit
  val = val >> 53; 	// get rid of the mantissa bits
  if (x != 0.0 && x != -0.0)
  {
  	int minVal = (int)(DANGER_ZONE_PERCENTAGE*2048.0);
  	if (val <= minVal)
  		ret = 1;
  }
  return ret;
}


/* --------------------- Clang Plug-in Functions ---------------------- */

/// errorType: 0:NaN, 1:INF, 2:Underflow
/// op: 0:ADD, 1:SUB, 2:MUL, 3:DIV

__host__ __device__
__attribute__((noinline)) static void _FPC_PLUGIN_INTERRUPT_(int errorType, int op, int loc, float fp32_val, double fp64_val, const char* fileName)
{
#if defined(__CUDA_ARCH__)
	//printf("-- _FPC_PLUGIN_INTERRUPT_\n");
	volatile bool blocked = true;
	while(blocked) {
			if(0 == atomicCAS(&_FPC_LOCK_STATE_, 0, 1)) {

				char e[64]; e[0] = '\0';
				char o[64]; o[0] = '\0';

				if 			(errorType == 0) _FPC_CPY_(e, "NaN");
				else if	(errorType == 1) _FPC_CPY_(e, "INF");
				else if	(errorType == 2) _FPC_CPY_(e, "Underflow");
				else _FPC_CPY_(e, "NONE");

				if 			(op == 0) _FPC_CPY_(o, "ADD");
				else if	(op == 1) _FPC_CPY_(o, "SUB");
				else if	(op == 2) _FPC_CPY_(o, "MUL");
				else if	(op == 3) _FPC_CPY_(o, "DIV");
				else _FPC_CPY_(o, "NONE");

				_FPC_PRINT_REPORT_HEADER_(0);
				_FPC_PRINT_REPORT_ROW_("Error", REPORT_COL1_SIZE, 0, ':');
				_FPC_PRINT_REPORT_ROW_(e, REPORT_COL2_SIZE, 1, ' ');
				_FPC_PRINT_REPORT_ROW_("Operation", REPORT_COL1_SIZE, 0, ':');
				if (errorType == 2)
				{
					_FPC_PRINT_REPORT_ROW_(o, 4, 0, ' ');
					if (fp32_val != 0)
						_FPC_PRINT_REPORT_ROW_(fp32_val, REPORT_COL2_SIZE, 1);
					else
						_FPC_PRINT_REPORT_ROW_(fp64_val, REPORT_COL2_SIZE, 1);
				}
				else
				{
					_FPC_PRINT_REPORT_ROW_(o, REPORT_COL2_SIZE, 1, ' ');
				}
				_FPC_PRINT_REPORT_ROW_("File", REPORT_COL1_SIZE, 0, ':');
				_FPC_PRINT_REPORT_ROW_(fileName, REPORT_COL2_SIZE, 1, ' ');
				_FPC_PRINT_REPORT_ROW_("Line", REPORT_COL1_SIZE, 0, ':');
				//_FPC_PRINT_REPORT_ROW_(l, REPORT_COL2_SIZE, 1);
				_FPC_PRINT_REPORT_ROW_(loc, REPORT_COL2_SIZE, 1);
        printf(" Block (%d,%d,%d), Thread (%d,%d,%d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
				_FPC_PRINT_REPORT_LINE_('+');

				asm("trap;");
		}
	}
#endif
}

__host__ __device__
__attribute__((noinline)) static void _FPC_PLUGIN_WARNING_(int errorType, int op, int loc, float fp32_val, double fp64_val, const char* fileName)
{
#if defined(__CUDA_ARCH__)
	//printf("-- _FPC_PLUGIN_WAR_\n");
	volatile bool blocked = true;
  	while(blocked) {
			if(0 == atomicCAS(&_FPC_LOCK_STATE_, 0, 1)) {

				char e[64]; e[0] = '\0';
				char o[64]; o[0] = '\0';

				if 			(errorType == 0) _FPC_CPY_(e, "NaN");
				else if	(errorType == 1) _FPC_CPY_(e, "INF");
				else if	(errorType == 2) _FPC_CPY_(e, "Underflow");
				else _FPC_CPY_(e, "NONE");

				if 			(op == 0) _FPC_CPY_(o, "ADD");
				else if	(op == 1) _FPC_CPY_(o, "SUB");
				else if	(op == 2) _FPC_CPY_(o, "MUL");
				else if	(op == 3) _FPC_CPY_(o, "DIV");
				else _FPC_CPY_(o, "NONE");

				_FPC_PRINT_REPORT_HEADER_(1);
				_FPC_PRINT_REPORT_ROW_("Error", REPORT_COL1_SIZE, 0, ':');
				_FPC_PRINT_REPORT_ROW_(e, REPORT_COL2_SIZE, 1, ' ');
				_FPC_PRINT_REPORT_ROW_("Operation", REPORT_COL1_SIZE, 0, ':');
				if (errorType == 1 || errorType == 2)
				{
					_FPC_PRINT_REPORT_ROW_(o, 4, 0, ' ');
					if (fp32_val != 0)
						_FPC_PRINT_REPORT_ROW_(fp32_val, REPORT_COL2_SIZE, 1);
					else
						_FPC_PRINT_REPORT_ROW_(fp64_val, REPORT_COL2_SIZE, 1);
				}
				else
				{
					_FPC_PRINT_REPORT_ROW_(o, REPORT_COL2_SIZE, 1, ' ');
				}
				_FPC_PRINT_REPORT_ROW_("File", REPORT_COL1_SIZE, 0, ':');
				_FPC_PRINT_REPORT_ROW_(fileName, REPORT_COL2_SIZE, 1, ' ');
				_FPC_PRINT_REPORT_ROW_("Line", REPORT_COL1_SIZE, 0, ':');
				//_FPC_PRINT_REPORT_ROW_(l, REPORT_COL2_SIZE, 1);
				_FPC_PRINT_REPORT_ROW_(loc, REPORT_COL2_SIZE, 1);
        printf(" Block (%d,%d,%d), Thread (%d,%d,%d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
				_FPC_PRINT_REPORT_LINE_('+');

				asm("trap;");
		}
	}
#endif
}

__device__ static int _FPC_HAS_PRINTED_ = 0;


// NOTE: we rely on CUDA function overloading for FP32 and FP64 versions
// FP64 version
__host__ __device__ static
double _FPC_CHECK_(double x, int loc, const char *fileName)
{
#if defined(__CUDA_ARCH__)
#ifdef FPC_DISABLE_CHECKING
  return x;
#else

#ifndef FPC_DISABLE_VERBOSE
  int id = _FPC_GET_GLOBAL_IDX_3D_3D();
  if (id == 0)
  {
    if (_FPC_HAS_PRINTED_==0) {
      printf("#FPCHECKER: checking on %f\n", x);
      _FPC_HAS_PRINTED_=1;
    }
  }
#endif

	//int op = -1;
	if (isinf(x))
	{
#if defined(FPC_SHORT_REPORTS) || defined(FPC_ERRORS_DONT_ABORT)
		printf("#FPCHECKER: INF error: %f @ %s:%d\n", x, fileName, loc);
#else
		_FPC_PLUGIN_INTERRUPT_(1, -1, loc, 0, x, fileName);
#endif
	}
	else if (isnan(x))
	{
#if defined(FPC_SHORT_REPORTS) || defined(FPC_ERRORS_DONT_ABORT)
		printf("#FPCHECKER: NaN error: %f @ %s:%d\n", x, fileName, loc);
#else
		_FPC_PLUGIN_INTERRUPT_(0, -1, loc, 0, x, fileName);
#endif
	}
	else /// subnormals check
	{

#ifndef FPC_DISABLE_SUBNORMAL
		if (_FPC_FP64_IS_SUBNORMAL(x))
		{
#if defined(FPC_SHORT_REPORTS) || defined(FPC_ERRORS_DONT_ABORT)
			printf("#FPCHECKER: underflow error: %f @ %s:%d\n", x, fileName, loc);
#else
			_FPC_PLUGIN_INTERRUPT_(2, -1, loc, 0, x, fileName);
#endif
		}
#endif // FPC_DISABLE_SUBNORMAL

#ifndef FPC_DISABLE_WARNINGS
		if (_FPC_FP64_IS_ALMOST_SUBNORMAL(x))
		{
#if defined(FPC_SHORT_REPORTS) || defined(FPC_ERRORS_DONT_ABORT)
			printf("#FPCHECKER: underflow warning: %f @ %s:%d\n", x, fileName, loc);
#else
			_FPC_PLUGIN_WARNING_(2, -1, loc, 0, x, fileName);
#endif
		}
		else if (_FPC_FP64_IS_ALMOST_OVERFLOW(x))
		{
#if defined(FPC_SHORT_REPORTS) || defined(FPC_ERRORS_DONT_ABORT)
			printf("#FPCHECKER: overflow warning: %f @ %s:%d\n", x, fileName, loc);
#else
			_FPC_PLUGIN_WARNING_(1, -1, loc, 0, x, fileName);
#endif
		}
#endif // FPC_DISABLE_WARNINGS

	}

	return x;
#endif // FPC_DISABLE_CHECKING
#else
  return x; // host code
#endif // end of #if defined(__CUDA_ARCH__)
}

// FP32 version
__host__ __device__ static
float _FPC_CHECK_(float x, int loc, const char *fileName)
{
#if defined(__CUDA_ARCH__)
#ifdef FPC_DISABLE_CHECKING
  return x;
#else

#ifndef FPC_DISABLE_VERBOSE
  int id = _FPC_GET_GLOBAL_IDX_3D_3D();
  if (id == 0)
  {
    if (_FPC_HAS_PRINTED_==0) {
      printf("#FPCHECKER: checking on %f\n", x);
      _FPC_HAS_PRINTED_=1;
    }
  }
#endif

	//int op = -1;
	if (isinf(x))
	{
#if defined(FPC_SHORT_REPORTS) || defined(FPC_ERRORS_DONT_ABORT)
		printf("#FPCHECKER: INF error: %f @ %s:%d\n", x, fileName, loc);
#else
		_FPC_PLUGIN_INTERRUPT_(1, -1, loc, x, 0, fileName);
#endif
	}
	else if (isnan(x))
	{
#if defined(FPC_SHORT_REPORTS) || defined(FPC_ERRORS_DONT_ABORT)
		printf("#FPCHECKER: NaN error: %f @ %s:%d\n", x, fileName, loc);
#else
		_FPC_PLUGIN_INTERRUPT_(0, -1, loc, x, 0, fileName);
#endif
	}
	else /// subnormals check
	{

#ifndef FPC_DISABLE_SUBNORMAL
		if (_FPC_FP64_IS_SUBNORMAL(x))
		{
#if defined(FPC_SHORT_REPORTS) || defined(FPC_ERRORS_DONT_ABORT)
			printf("#FPCHECKER: underflow error: %f @ %s:%d\n", x, fileName, loc);
#else
			_FPC_PLUGIN_INTERRUPT_(2, -1, loc, x, 0, fileName);
#endif
		}
#endif // FPC_DISABLE_SUBNORMAL


#ifndef FPC_DISABLE_WARNINGS
		if (_FPC_FP64_IS_ALMOST_SUBNORMAL(x))
		{
#if defined(FPC_SHORT_REPORTS) || defined(FPC_ERRORS_DONT_ABORT)
			printf("#FPCHECKER: underflow warning: %f @ %s:%d\n", x, fileName, loc);
#else
			_FPC_PLUGIN_WARNING_(2, -1, loc, x, 0, fileName);
#endif
		}
		else if (_FPC_FP64_IS_ALMOST_OVERFLOW(x))
		{
#if defined(FPC_SHORT_REPORTS) || defined(FPC_ERRORS_DONT_ABORT)
			printf("#FPCHECKER: overflow warning: %f @ %s:%d\n", x, fileName, loc);
#else
			_FPC_PLUGIN_WARNING_(1, -1, loc, x, 0, fileName);
#endif
		}
#endif // FPC_DISABLE_WARNINGS

	}

	return x;
#endif // FPC_DISABLE_CHECKING
#else
  return x; // host code
#endif // end of #if defined(__CUDA_ARCH__)
}



#endif /* SRC_RUNTIME_H_ */
