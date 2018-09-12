/*
 * Runtime.cpp
 *
 *  Created on: Apr 10, 2018
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#include "Runtime.cpp"

using namespace CUDAAnalysis;

/****************************************/
#if !defined TABLE_SIZE
#define TABLE_SIZE 30000 // max number of GPU threads
#define LOC_SIZE 4000
#endif
/****************************************/

/* Global table */
__device__ StatType statsTable[LOC_SIZE][TABLE_SIZE];

__device__ void _LOG_FLOATING_POINT_OP_(double x, double y, double z, int OP, int loc)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	assert(loc < TABLE_SIZE && "Invalid table item");
	assert(blockDim.y == 1 && "Invalid dimension!");
	assert(blockDim.z == 1 && "Invalid dimension!");
	assert(gridDim.y == 1 && "Invalid dimension!");
	assert(gridDim.z == 1 && "Invalid dimension!");

	double res_64;
	float res_32;

	if (OP == 0)
	{
		res_64 = x + y;
		res_32 = (float)x + (float)y;
	}
	else if (OP == 1)
	{
		res_64 = x - y;
		res_32 = (float)x - (float)y;
	}
	else if (OP == 2)
	{
		res_64 = x * y;
		res_32 = (float)x * (float)y;
	}
	else if (OP == 3)
	{
		res_64 = x / y;
		res_32 = (float)x / (float)y;
	}
	else if (OP == 4)
	{
		res_64 = __fma_rn(x, y, z);
		res_32 = __fmaf_rn (x, y, z);
	}
	else if (OP == 5)
	{
		res_64 = __dsqrt_rn (x);
		res_32 = __fsqrt_rn (x);
	}
	else if (OP == 6)
	{
		res_64 = rsqrt (x);
		res_32 = __frsqrt_rn (x);
	}
	else
	{
		printf("Incorrect operation in runtime\n");
		asm("trap;");
	}

	//FIXME: check if error should be calculated in this way
	//float error = fabs(((float)res_64 - res_32) / (float)res_64);
	float error;
	if ((float)res_64 != 0)
		error = ((float)res_64 - res_32) / (float)res_64;
	else
		error = 0;
	error = (error < 0.0) ? error*(-1) : error; // get absolute value
	//double error = fabs(res_64 - (double)res_32);

	statsTable[loc][id].total_error += error;
	statsTable[loc][id].n += 1;
}

/*__device__ void _PRINT_TABLE_()
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	printf("My Id: %d\n", id);

	if (id == 0)
	{
		for (int i=1; i < LOC_SIZE; ++i)
		{
			if (statsTable[i][0].n == 0)
				break;

			//float locStat = 0;
			float total_error = 0;
			for (int j=0; j < TABLE_SIZE; ++j)
			{
				if (statsTable[i][j].n == 0)
					break;

				//locStat += statsTable[i][j].average;
				total_error += statsTable[i][j].total_error;
			}

			//printf("LOC %d SUM: %.17g\n", i, locStat);
			printf("LOC %d TOTAL: %.10g\n", i, total_error);
		}
	}
}*/

