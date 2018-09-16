/*
 * CodeMatching.h
 *
 *  Created on: Sep 15, 2018
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#ifndef SRC_CODEMATCHING_H_
#define SRC_CODEMATCHING_H_

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace CUDAAnalysis {

class CodeMatching
{

public:
	static bool isUnwantedFunction(Function *f);
	static bool isMainFunction(Function *f);
	static bool isAKernelFunction(const Function &F);

	/// Determines if the pass can access device functions in the module
	static bool isDeviceCode(Module *mod);
};

}



#endif /* SRC_CODEMATCHING_H_ */
