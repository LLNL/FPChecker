/*
 * Logging.cpp
 *
 *  Created on: Feb 4, 2019
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */


#include "Logging.h"
#include <llvm/Support/raw_ostream.h>
#include <stdlib.h>

using namespace CUDAAnalysis;
using namespace llvm;

void Logging::info(const char *msg)
{
	outs() << FPC_PREFIX << msg << "\n";
}

void Logging::error(const char *msg)
{
	errs() << FPC_PREFIX << msg << "\n";
	exit(EXIT_FAILURE);
}


